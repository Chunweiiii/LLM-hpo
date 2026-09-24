"""
統一介面呼叫 LLM:Claude Haiku 4.5 / Claude Sonnet 5 / Claude Opus 5,
以及本機 Ollama 跑的開源模型(通用推理模型,非程式碼特化)。

2026-09-21 新增:原本 Ollama 這條路只硬寫死 Qwen3-14B 一個模型
(call_qwen_ollama),現在校規排除中國模型後改成通用的 OLLAMA_MODELS
註冊表 + call_ollama(model_key, ...),之後要加 Ministral 3 14B /
Phi-4 14B 等候選,只要在 OLLAMA_MODELS 多加一筆、不用再複製整個函式。
Qwen3-14B 先保留在表裡(還沒決定要不要整個從論文拿掉,只是不能是最終
候選之一),避免動到既有已經跑完的實驗程式碼路徑。

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


# 本機 Ollama 模型註冊表:model_key -> {ollama pull tag, 是否開 think 模式}。
# 加新的本機模型只要在這裡多一筆,不用再寫一個新函式。
# - "think": True 的前提是該模型在 Ollama library 頁面的 Capabilities 有標
#   "thinking"(用 `ollama show <tag>` 或官網 tags 頁確認),不是每個模型
#   都支援 Ollama /api/chat 的 think 參數,亂開可能被忽略或直接報錯。
OLLAMA_MODELS = {
    "qwen3-14b": {"tag": "qwen3:14b", "think": True},
    # 2026-09-21 新增,見上方模組說明。Ollama library 頁面 Capabilities 標
    # 了 "thinking",跟 Qwen3 同樣走 think=True 這條路。
    "gemma4-12b": {"tag": "gemma4:12b", "think": True},
    # 2026-09-21 新增:額外的探索性比較點,不算低階模型主線的一員,專門用來
    # 回答「~32B 等級的本機開源模型能不能接近 Haiku」這個獨立問題,見
    # docs/decisions.md。同屬 gemma4 家族,Capabilities 一樣標 "thinking"。
    # 20GB(Q4_K_M),24GB VRAM 塞得下但餘裕不多(~4GB),第一次跑如果 OOM
    # 記得回報,可能要縮短 context 或換更低的量化版本。
    "gemma4-31b": {"tag": "gemma4:31b", "think": True},
    # 2026-09-22 新增:同屬「~32B 比較組」,但架構是 MoE(總共 26B 參數,
    # 每個 token 只啟動 3.8B),推理時的實際運算量比較接近一個 4B 模型。
    # 跟 gemma4-31b 不是嚴格同量級的 dense 對照,寫論文時要講清楚這個
    # 架構差異,不要跟 gemma4-31b 混在同一個「32B」類別裡討論。19GB。
    "gemma4-26b": {"tag": "gemma4:26b", "think": True},
    # 2026-09-22 新增:Mistral AI(法國)出的 dense 24B 模型,比 gemma4-31b
    # 略小一個量級,當作「~25B 這個量級」的額外對照點。14GB。Ollama 頁面
    # 沒有標「thinking」capability,先當作不支援(think=False),如果之後
    # 發現有支援再改。
    "mistral-small-24b": {"tag": "mistral-small:24b", "think": False},
    # 2026-09-22 新增:低階模型主線候選,跟 gemma4-12b 同一個量級(9.1GB,
    # Q4_K_M),Mistral AI(法國)出品。Ollama 頁面沒標 thinking capability,
    # think=False。
    "ministral-3-14b": {"tag": "ministral-3:14b", "think": False},
    # 2026-09-22 新增:低階模型主線候選,微軟出品(9.1GB,Q4_K_M)。Ollama
    # 頁面沒標 thinking capability,think=False。
    "phi4-14b": {"tag": "phi4:14b", "think": False},
    # 2026-09-22 新增:低階模型主線候選,量級比其他三個小一截(8B,4.9GB,
    # Q4_K_M),Meta 出品。不是推理模型,think=False。
    "llama3.1-8b": {"tag": "llama3.1:8b", "think": False},
}


def call_ollama(model_key: str, system_prompt: str, user_message: str,
                 base_url: str = "http://localhost:11434") -> dict:
    """透過本機 Ollama 呼叫 OLLAMA_MODELS 裡任一個模型。

    think=True 的模型:Ollama 會把推理過程放在 message.thinking,最終答案
    仍乾淨地放在 message.content,兩者不會混在一起,所以底下的 JSON 解析
    不受影響。訓練一輪的時間遠長於 LLM 呼叫,多花一點時間換取推理品質
    划算,而且 thinking 內容本身對論文的質化分析也有參考價值,所以順便
    存起來(見回傳值 "_thinking" 欄位;think=False 的模型這欄會是空字串)。
    """
    import requests

    spec = OLLAMA_MODELS[model_key]
    resp = requests.post(
        f"{base_url}/api/chat",
        json={
            "model": spec["tag"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "think": spec["think"],
            "stream": False,
        },
    )
    resp.raise_for_status()
    message = resp.json()["message"]
    result = _extract_json(message["content"])
    result["_thinking"] = message.get("thinking", "")
    return result


class LLMResponseParseError(ValueError):
    """LLM 回覆的內容無法被解析成合法 JSON(格式跑掉,例如漏引號、多餘逗號、
    夾雜JSON以外的文字,不是超參數數值本身超出範圍那種問題)。

    2026-09-23 新增:原本 _extract_json() 沒有接住 json.loads() 可能丟出的
    json.decoder.JSONDecodeError,任何一次 LLM 輸出格式跑掉都會讓整個
    process 直接崩潰(這裡是唯一確認過會實際發生、且不算罕見的一種 LLM
    輸出格式錯誤——llama3.1-8b 在 hinted 診斷組多次於同一輪撞到)。雖然當時
    決定先不動這裡、靠手動重跑繞過(resume 機制能保住已完成輪次的進度),
    但重跑多次仍然反覆卡在同一輪之後,改為在這裡定義一個專屬例外類別,讓
    hpo_loop.py 的 _propose_with_retry() 可以像現有的「超出合法範圍」
    「缺必要欄位」兩種情況一樣,在 process 內部用同一套重試機制處理,不用
    整個中斷,見 docs/decisions.md。"""

    def __init__(self, raw_text: str, original_error: json.JSONDecodeError):
        self.raw_text = raw_text
        self.original_error = original_error
        super().__init__(
            f"LLM 回覆無法解析為合法 JSON:{original_error}。"
            f"原始回覆前 500 字:{raw_text[:500]!r}"
        )


def _strip_json_comments(text: str) -> str:
    """移除 JSON 文字裡的 `//` 行內註解——不是合法 JSON 語法,但實測發現
    llama3.1-8b 很習慣在數值後面加這種註解說明自己改了什麼,例如:
        "mosaic": 0.25,  // 增加 mosaic 值
    這是目前唯一確認過、且反覆發生的 json.loads() 失敗主因(見
    LLMResponseParseError 的說明、docs/decisions.md 2026-09-23 這筆)。

    用逐字元掃描而不是簡單的 regex,是為了避免誤砍字串內容本身剛好包含
    `//`(例如 reasoning 文字裡寫到網址)——只有「不在字串內」的 `//` 才會
    被當成註解砍掉。"""
    result = []
    in_string = False
    escape = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if in_string:
            result.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            result.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < n and text[i + 1] == "/":
            # 跳到這一行結束(換行本身留著,不影響後面的行號)
            while i < n and text[i] != "\n":
                i += 1
            continue
        result.append(ch)
        i += 1
    return "".join(result)


def _strip_trailing_commas(text: str) -> str:
    """移除 `}` 或 `]` 前面多餘的逗號,例如 `"box": 12.5,\\n  }`——這也不是
    合法 JSON 語法,但砍掉行內註解後常常會剩下這種懸空的逗號(原本逗號是
    接在註解前面,不是接下一個欄位),同一批 llama3.1-8b 回覆裡多次一起
    出現,一併處理。"""
    import re
    return re.sub(r",(\s*[}\]])", r"\1", text)


def _extract_json(text: str) -> dict:
    """從 LLM 回覆文字裡抓出 ```json ... ``` 區塊並解析。

    2026-09-23 修正:json.loads() 失敗時原本讓 json.decoder.JSONDecodeError
    直接往外傳,沒有任何地方接住。這裡改包一層 try/except,轉成
    LLMResponseParseError 往外丟,見 LLMResponseParseError 的說明。

    2026-09-23 再修正:光是接住例外、靠重試請 LLM 重新輸出,對
    llama3.1-8b 習慣性加 `//` 註解這個問題效果不穩定(重試 3 次仍然
    反覆發生)。改成在真的丟出例外之前,先嘗試清掉行內註解跟懸空逗號這
    兩種已知、且經驗證是這個專案裡實際發生過的格式問題,清完再重新
    parse 一次;還是失敗才真的當成 LLMResponseParseError 往外丟(觸發
    hpo_loop.py 的重試機制)。這個清理只處理已知格式問題,不會改動任何
    合法的超參數數值或文字內容本身。"""
    original_text = text
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    cleaned = _strip_trailing_commas(_strip_json_comments(text))
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise LLMResponseParseError(original_text, e) from e


def propose_hyperparameters(model_key: str, system_prompt: str,
                             report_text: str) -> dict:
    """統一進入點:依 model_key 分派到對應的 LLM。"""
    if model_key in OLLAMA_MODELS:
        return call_ollama(model_key, system_prompt, report_text)
    elif model_key in CLAUDE_MODELS:
        return call_claude(model_key, system_prompt, report_text)
    raise ValueError(f"未知的 model_key: {model_key}")
