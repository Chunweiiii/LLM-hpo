"""
統一介面呼叫四個 LLM:Claude Haiku 4.5 / Claude Sonnet 5 / Claude Opus 5 /
Qwen3-14B(本機 Ollama,通用推理模型,見 docs/decisions.md 為何從
Qwen2.5-Coder 換成 Qwen3)。

不依賴尚未定案的閾值/演算法決策,可以先動工。
"""

import os
import json
from dotenv import load_dotenv

load_dotenv()

CLAUDE_MODELS = {
    # 2026-09 修正:原本寫 "claude-haiku-4-5" 是錯的,Haiku 4.5 早於「4.6 世代」
    # 開始用的無日期 model id 慣例,還是需要帶日期後綴的 dated snapshot 格式,
    # 查證後應為 "claude-haiku-4-5-20251001",見 docs/decisions.md。
    "claude-haiku-4.5": "claude-haiku-4-5-20251001",
    "claude-sonnet-5": "claude-sonnet-5",
    "claude-opus-5": "claude-opus-5",
}


def call_claude(model_key: str, system_prompt: str, user_message: str) -> dict:
    """呼叫 Anthropic API,回傳解析後的 JSON 決策。"""
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    model_id = CLAUDE_MODELS[model_key]

    response = client.messages.create(
        model=model_id,
        # 2026-09 修正:原本是 2048。Sonnet 5 / Opus 5 在這個環境會預設夾帶
        # 一段 thinking 內容(見下面的說明),而 max_tokens 是 thinking+
        # 最終文字輸出的總預算。2048 常常被 thinking 吃掉大半,導致最後要
        # 拿來解析的 JSON 文字被硬生生截斷在字串中間,parse 直接
        # JSONDecodeError: Unterminated string(Sonnet round 0 補救時實測
        # 踩到)。拉高到 8192 讓 thinking + 完整 JSON 都有空間,見
        # docs/decisions.md。
        max_tokens=8192,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

    # 2026-09 修正:原本寫 text = response.content[0].text,假設 content 的
    # 第一個區塊一定是文字。但 Sonnet 5(後來發現 Opus 5 應該也一樣)的回應
    # 有時會多夾帶一個 type="thinking" 的 ThinkingBlock(沒有 .text 屬性,
    # 是模型內部的思考過程,不是要拿來解析 JSON 的答案),而且不保證排在
    # content 的哪個位置,導致 response.content[0].text 直接爆
    # AttributeError: 'ThinkingBlock' object has no attribute 'text'。這裡
    # 沒有主動要求 thinking(呼叫時沒帶 thinking 參數),看起來是這個環境
    # 這幾個模型的預設行為,不是我們特別打開的,見 docs/decisions.md。
    # 改成明確找出 type == "text" 的區塊,不管它出現在第幾個位置、前面有沒有
    # 夾別的區塊類型都不受影響。
    text = None
    for block in response.content:
        if getattr(block, "type", None) == "text":
            text = block.text
            break
    if text is None:
        block_types = [getattr(b, "type", type(b).__name__) for b in response.content]
        raise ValueError(
            f"Claude API 回應裡沒有找到文字內容,content 型別列表:{block_types}"
        )

    return _extract_json(text)


def call_qwen_ollama(system_prompt: str, user_message: str,
                      base_url: str = "http://localhost:11434") -> dict:
    """透過本機 Ollama 呼叫 Qwen3-14B。

    think=True:Qwen3 支援 thinking 模式,Ollama 會把推理過程放在
    message.thinking,最終答案仍乾淨地放在 message.content,兩者不會混在
    一起,所以底下的 JSON 解析不受影響。訓練一輪的時間遠長於 LLM 呼叫,
    多花一點時間換取推理品質划算,而且 thinking 內容本身對論文的質化分析
    也有參考價值,所以順便存起來(見回傳值 "_thinking" 欄位)。
    """
    import requests

    resp = requests.post(
        f"{base_url}/api/chat",
        json={
            "model": "qwen3:14b",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "think": True,
            "stream": False,
        },
    )
    resp.raise_for_status()
    message = resp.json()["message"]
    result = _extract_json(message["content"])
    result["_thinking"] = message.get("thinking", "")
    return result


def _extract_json(text: str) -> dict:
    """從 LLM 回覆文字裡抓出 ```json ... ``` 區塊並解析。"""
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    return json.loads(text.strip())


def propose_hyperparameters(model_key: str, system_prompt: str,
                             report_text: str) -> dict:
    """統一進入點:依 model_key 分派到對應的 LLM。"""
    if model_key == "qwen3-14b":
        return call_qwen_ollama(system_prompt, report_text)
    elif model_key in CLAUDE_MODELS:
        return call_claude(model_key, system_prompt, report_text)
    raise ValueError(f"未知的 model_key: {model_key}")
