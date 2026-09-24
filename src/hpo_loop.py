"""
主迴圈:N 輪 propose -> train -> diagnose -> best-tracking -> next round。

2026-09 韌性強化(電腦被 Windows 自動更新強制重開機,導致一次 13 輪的
Qwen HPO 資料遺失後加上,見 docs/decisions.md):
  1. 每輪訓練 + 提案完成後立刻把目前進度寫進 save_path(JSON),不用等
     全部跑完才存檔。
  2. run_hpo_loop() 開始執行時,如果偵測到 save_path 已經有未跑完的進度,
     自動從上次中斷的地方接續(用上次記錄的 LLM 提案當這輪的超參數),
     不需要額外的 CLI 參數,重新執行同一條指令就會自動接續。
  3. LLM 回傳的超參數如果缺 key 或超出 SEARCH_SPACE 範圍,附上錯誤訊息
     重新請 LLM 提案,最多重試 MAX_RETRIES 次,還是不行才真正中斷(這種
     算「有害」錯誤,不能悄悄放行,例如之前 Qwen 漏了 lr0,程式沒檢查,
     結果悄悄改用 Ultralytics 內建預設值,整輪結果因此不可信)。
  3b. 2026-09-23 新增:LLM 回覆本身不是合法 JSON(格式跑掉,不是超參數
     數值問題)原本沒有任何地方接住,json.decoder.JSONDecodeError 會直接
     讓整個 process 崩潰——llama3.1-8b 在 hinted 診斷組多次撞到,靠手動
     重跑繞過(resume 機制保住進度)一開始還行,但同一輪反覆卡住之後
     決定處理。做法比照第 3 點:llm_agent.py 的 _extract_json() 改成把
     json.JSONDecodeError 轉成 LLMResponseParseError 往外丟,這裡一樣
     當「有害」錯誤處理,附上錯誤訊息重新請 LLM 提案,最多重試
     MAX_RETRIES 次,還是不行才真正中斷,見 docs/decisions.md。
  4. LLM 回傳完全重複的超參數(浪費一輪訓練預算但不算「有害」,例如
     中斷前的 13 輪資料裡 round6 跟 round9 完全一樣),一樣重新請 LLM
     提案,重試 MAX_RETRIES 次後如果還是重複,改成「軟接受」照跑,
     不中斷整個流程。
  5. 退步時(這輪 mAP50-95 沒有超過目前歷史最佳),下一輪的報告額外附上
     「HISTORICAL BEST (REFERENCE)」段落,標示歷史最佳的分數與超參數。
     這不是硬性回退/重跑機制,LLM 仍然可以自由決定要不要參考這個點,
     純粹是資訊揭露,避免浪費輪次預算重複驗證同一個已知點。

2026-09-19 加兩個東西,見 docs/decisions.md:
  6. `folder_suffix`:`runs/detect/<run_name>` 的命名原本只有
     `{model_key}_round{round_idx}`,沒有用到 `results/` 底下 json 檔名
     用的 `_r<N>` 這種後綴。這造成一個真實踩到的 bug——用
     `--rounds 10`(數值特意跟預設值一樣,只是想拿到不同檔名)重跑同一個
     模型當第二次獨立複製(replicate)時,json 檔名有靠後綴分開沒有撞名,
     但 `runs/detect/claude-opus-5_round0` 這個資料夾名稱沒有變,
     `train_run()` 發現資料夾已存在會整個刪掉重練,結果把第一次複製的
     原始訓練資料夾(權重、confusion matrix 等)蓋掉了,json 結果檔本身
     没事(那些是分開存的),但原始訓練產物救不回來。加上 `folder_suffix`
     參數(呼叫端傳跟 json 檔名一樣的後綴進來),資料夾名稱也會跟著分開,
     徹底避免撞名,不管是 `--rounds` 重跑還是下面第7點的 budget-aware
     版本都適用。
  7. `include_budget_info`(只有 `run_hpo_loop` 有,`run_score_only_loop`
     沒有,見 docs/decisions.md 討論):使用者觀察到 Opus 5 完整診斷組
     (第一次跑)的推理文字裡,明顯是用「像做嚴謹科學實驗一樣,一次只調
     一個變因,還規劃好幾輪以後要測什麼」的長線策略(例如 round4 的推理
     寫「Next round: if this gains, try box=4.5; if it regresses, ... I
     pivot to attacking the epoch-55 plateau」),但系統提示跟每輪報告
     從來沒有告訴 LLM 這個優化總共只有 10 輪預算——LLM 完全不知道自己
     只剩幾次嘗試機會。這可能是 Opus/Sonnet 完整診斷組表現不如預期
     (甚至輸給純分數組跟 Random Search baseline)的其中一個原因:不知道
     預算快用完,就沒有理由從「謹慎的單變因測試」切換成「更大膽的多變因
     調整」。這次新增 `include_budget_info`,如果是 True,會在
     report_text 最前面插入一段「=== OPTIMIZATION BUDGET ===」,誠實告訴
     LLM 目前是第幾輪、總共幾輪、還剩幾輪,但不指定它該怎麼反應(跟「不給
     診斷類型→參數對應提示」同樣的哲學,只給客觀事實,策略由 LLM 自己
     決定)。這次刻意只加在完整診斷組(`run_hpo_loop`),不動純分數組
     (`run_score_only_loop`)跟其他還沒重跑的模型,先驗證這個假說在
     Opus/Sonnet 身上是否成立,見 docs/decisions.md。

report_text 組成順序(對應 prompts/program.md 的「Previous Rounds History
Format」段落):
  [OPTIMIZATION BUDGET](只有 include_budget_info=True 才有,且一律放最前面)
  + [PREVIOUS ROUNDS HISTORY](如果有的話) + [HISTORICAL BEST REFERENCE]
  (如果這輪退步的話) + [TRAINING RUN FACTS + DIAGNOSTIC FLAGS]

run_score_only_loop() 是對照組(見該函式說明),刻意重用本模組大部分的
輔助函式,確保跟 run_hpo_loop() 除了「有沒有診斷資訊」之外的其他機制
(重試/重複偵測/resume/歷史最佳參考)完全對等。
"""

from __future__ import annotations

import time
from pathlib import Path

from .train_runner import train_run
from .llm_agent import propose_hyperparameters, OLLAMA_MODELS, LLMResponseParseError
from .summarize_log import read_log, compute_facts, diagnose, to_report_text
from .baseline_search import SEARCH_SPACE
from .utils import save_json, load_json

# 2026-09-23 新增(見 docs/decisions.md):本機 Ollama 模型呼叫完
# propose_hyperparameters() 後,即使設了 keep_alive=0 要求立刻卸載,VRAM
# 不保證在這次 Python 呼叫返回的當下就已經真正釋放乾淨——下一輪
# train_run() 緊接著就要跟 YOLO 要訓練用的顯存,曾經在 gemma4-31b
# score-only 實測踩到 CUDA out of memory(round2->round3 交接處,
# torch.nn.utils.clip_grad_norm_ 這步噴錯)。加一個緩衝等待,只在模型是
# 本機 Ollama 模型時才啟用(Claude API 模型不佔用本地 VRAM,不需要等待),
# 純粹是操作面的保險措施,不影響任何實驗結果/超參數本身。
GPU_UNLOAD_BUFFER_SEC = 15

# 每輪 LLM 一定要提出這五個超參數,缺一個都不行(2026-09 從 4 個擴到 5 個,
# 新增 box,見 docs/decisions.md)。2026-09 實測發現 Qwen 有一輪的回覆漏了
# lr0,而程式原本沒檢查,導致 model.train() 收不到 lr0,悄悄改用 Ultralytics
# 內建預設值(0.01),不是 LLM 真正的提案,整輪結果因此不可信。加這個檢查
# 讓這種狀況直接報錯、當場中斷,而不是悄悄跑錯又難以事後察覺。
EXPECTED_HP_KEYS = {"lr0", "weight_decay", "mosaic", "hsv_v", "box"}
TRACKED_KEYS = ["lr0", "weight_decay", "mosaic", "hsv_v", "box"]

MAX_RETRIES = 3


def _format_hp(hp: dict) -> str:
    return ", ".join(f"{k}={hp[k]}" for k in TRACKED_KEYS)


def _format_history(history: list[dict]) -> str:
    """組出 prompts/program.md 「Previous Rounds History Format」要求的
    === PREVIOUS ROUNDS HISTORY === 區塊。round 顯示用 1-indexed(Round 1,
    Round 2, ...),跟資料夾命名(round0, round1, ...)的 0-indexed 是刻意
    分開的,前者是給 LLM 看的人類可讀編號,後者是程式內部/檔案系統用的。
    run_score_only_loop() 也共用這個函式,格式完全一樣。"""
    if not history:
        return ""
    lines = ["=== PREVIOUS ROUNDS HISTORY ==="]
    for entry in history:
        lines.append(
            f"Round {entry['round'] + 1}: {_format_hp(entry['hyperparameters'])}"
            f"  -> mAP50-95={entry['map5095']:.4f}"
        )
    return "\n".join(lines)


def _format_budget(round_idx: int, num_rounds: int) -> str:
    """組出「=== OPTIMIZATION BUDGET ===」區塊,誠實告訴 LLM 目前輪次跟
    總預算,不建議該怎麼反應,見模組開頭第7點說明。round_idx 是 0-indexed
    (跟資料夾命名一致),顯示給 LLM 看時一律轉成 1-indexed,跟
    _format_history 的顯示慣例一致。"""
    completed = round_idx + 1
    remaining = num_rounds - completed
    if remaining > 0:
        remaining_line = f"{remaining} more round(s) will run after this one."
    else:
        remaining_line = (
            "This was the final round of this optimization run; no further "
            "rounds will be trained, though you should still submit a valid "
            "proposal."
        )
    return (
        "=== OPTIMIZATION BUDGET ===\n"
        f"You have just completed round {completed} of {num_rounds} total "
        f"rounds in this optimization run. {remaining_line}\n"
    )


def _format_best_reference(best: dict, current_map5095: float) -> str:
    """只有「這輪沒有超過目前歷史最佳」時才附上,純粹是資訊揭露,不是
    強制要求 LLM 回到這個點,見模組開頭說明。run_score_only_loop() 也
    共用這個函式。"""
    if best["round"] == -1 or current_map5095 > best["map5095"]:
        return ""
    return (
        "\n\n=== HISTORICAL BEST (REFERENCE) ===\n"
        f"This round's mAP50-95 ({current_map5095:.4f}) did not exceed the "
        "historical best. For reference, the best result so far is:\n"
        f"Round {best['round'] + 1}: {_format_hp(best['hyperparameters'])}"
        f"  -> mAP50-95={best['map5095']:.4f}\n"
        "You may use this as a reference point, but you are free to propose "
        "any new configuration within the valid ranges."
    )


def _validate_hyperparameters(new_hp: dict) -> str | None:
    """回傳 None 代表沒問題;回傳字串代表要餵回去給 LLM 的錯誤訊息
    (缺 key / 超出範圍,兩種都算「有害」錯誤)。"""
    missing_keys = EXPECTED_HP_KEYS - set(new_hp.keys())
    if missing_keys:
        return (
            f"你上一次的回覆缺少必要的超參數欄位:{sorted(missing_keys)}。"
            f"new_hyperparameters 必須包含全部五個欄位:{sorted(EXPECTED_HP_KEYS)}。"
            "請重新輸出完整的 JSON,不要省略任何欄位。"
        )

    out_of_range = {}
    for key, (lo, hi) in SEARCH_SPACE.items():
        value = new_hp[key]
        if not (lo <= value <= hi):
            out_of_range[key] = value
    if out_of_range:
        ranges = ", ".join(
            f"{k}: {SEARCH_SPACE[k][0]} to {SEARCH_SPACE[k][1]}" for k in out_of_range
        )
        return (
            f"你上一次提出的超參數超出合法範圍:{out_of_range}。"
            f"合法範圍是 {ranges}。請重新輸出一組在範圍內的超參數。"
        )

    return None


def _check_duplicate(new_hp: dict, history: list[dict]) -> bool:
    """跟「已經實際訓練過」的超參數組合(history 裡的 hyperparameters,不是
    llm_proposal_for_next_round)比對,避免浪費一輪訓練預算重跑一模一樣的
    組合。"""
    for entry in history:
        if all(entry["hyperparameters"][k] == new_hp[k] for k in EXPECTED_HP_KEYS):
            return True
    return False


def _propose_with_retry(model_key: str, system_prompt: str, report_text: str,
                         round_idx: int, history: list[dict]) -> dict:
    """呼叫 LLM 提案。缺 key / 超出範圍 -> 硬重試,MAX_RETRIES 次後中斷
    (有害錯誤,不能悄悄放行)。完全重複 -> 軟重試,MAX_RETRIES 次後照樣
    接受繼續跑(浪費但無害)。"""
    current_report = report_text
    proposal = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            proposal = propose_hyperparameters(model_key, system_prompt, current_report)
        except LLMResponseParseError as e:
            if attempt == MAX_RETRIES:
                raise ValueError(
                    f"round {round_idx}({model_key}):重試 {MAX_RETRIES} 次後,"
                    f"LLM 回覆仍然無法解析成合法 JSON。最後一次錯誤:{e}"
                ) from e
            print(f"  [round {round_idx}] LLM回覆不是合法JSON,重試中"
                  f"({attempt + 1}/{MAX_RETRIES}):{e}")
            current_report = (
                report_text
                + "\n\n=== 上一次提案的錯誤 ===\n"
                "你上一次的回覆無法被解析成合法JSON(可能格式跑掉,例如漏引號、"
                "多餘逗號、或摻雜了JSON以外的文字)。請重新只輸出一份合法的JSON,"
                "不要包含任何JSON以外的文字。"
            )
            continue

        new_hp = proposal.get("new_hyperparameters", {})

        error_msg = _validate_hyperparameters(new_hp)
        if error_msg is not None:
            if attempt == MAX_RETRIES:
                raise ValueError(
                    f"round {round_idx}({model_key}):重試 {MAX_RETRIES} 次後,"
                    f"LLM 提案仍然無效。最後一次錯誤:{error_msg}"
                    f"完整回傳內容: {proposal}"
                )
            print(f"  [round {round_idx}] 提案無效,重試中"
                  f"({attempt + 1}/{MAX_RETRIES}):{error_msg}")
            current_report = report_text + f"\n\n=== 上一次提案的錯誤 ===\n{error_msg}"
            continue

        if _check_duplicate(new_hp, history):
            if attempt == MAX_RETRIES:
                print(f"  [round {round_idx}] 重試 {MAX_RETRIES} 次後仍然是重複提案,"
                      "改為接受照跑(浪費但無害)")
                return proposal
            print(f"  [round {round_idx}] 提案跟先前某一輪完全重複,重試中"
                  f"({attempt + 1}/{MAX_RETRIES})")
            current_report = (
                report_text
                + "\n\n=== 上一次提案的問題 ===\n"
                "你剛剛提出的超參數組合跟先前某一輪完全相同,不會提供新資訊,"
                "請根據歷史紀錄提出一組不同的超參數。"
            )
            continue

        return proposal

    return proposal


def _load_resume_state(save_path: str | None) -> dict | None:
    if save_path is None or not Path(save_path).exists():
        return None
    try:
        return load_json(save_path)
    except Exception as e:
        print(f"[resume] 讀取 {save_path} 失敗,視為沒有進度可接續:{e}")
        return None


def run_hpo_loop(model_key: str, system_prompt: str, settings: dict,
                  initial_hyperparameters: dict, num_rounds: int = 10,
                  save_path: str | None = None,
                  include_budget_info: bool = False,
                  folder_suffix: str = "") -> dict:
    """
    執行完整診斷組的 HPO 迴圈。

    save_path: 每輪結束後即時寫入進度的 JSON 路徑。如果指定,且該路徑已經
    存在未跑完的進度,會自動從上次中斷的地方接續(用上次記錄的
    llm_proposal_for_next_round 當這輪的超參數),不需要額外的 CLI 參數,
    重跑同一條指令就會自動接續。如果要確保是全新開始(例如換了新的
    baseline/起始超參數),要先手動刪掉舊的 save_path 檔案。

    include_budget_info: True 的話,每輪報告最前面會多一段
    「=== OPTIMIZATION BUDGET ===」,告訴 LLM 目前第幾輪/總共幾輪/還剩
    幾輪,見模組開頭第7點說明。

    folder_suffix: 附加在 `runs/detect/<run_name>` 資料夾名稱後面的字串
    (預設空字串,不影響原本的命名),避免跟其他次執行(例如用
    `--rounds` 重跑的獨立複製、或這次的 budget-aware 版本)撞名,見模組
    開頭第6點說明。呼叫端應該傳跟 json 結果檔名一樣的後綴,確保資料夾
    命名跟結果檔名一一對應、容易對照。

    回傳: {"best_round": int, "best_map5095": float, "best_hyperparameters": dict,
           "history": [...]}
    """
    history: list[dict] = []
    best = {"round": -1, "map5095": -1.0, "hyperparameters": None}
    current_hp = initial_hyperparameters
    start_round = 0

    resume_state = _load_resume_state(save_path)
    if resume_state and resume_state.get("history"):
        history = resume_state["history"]
        start_round = len(history)
        current_hp = history[-1]["llm_proposal_for_next_round"]
        for entry in history:
            if entry["map5095"] > best["map5095"]:
                best = {
                    "round": entry["round"],
                    "map5095": entry["map5095"],
                    "hyperparameters": entry["hyperparameters"],
                }
        print(f"[resume] 偵測到 {save_path} 已有 {start_round} 輪進度,"
              f"從 round {start_round} 接續"
              f"(目前歷史最佳 round {best['round']}, mAP50-95={best['map5095']:.4f})")

    for round_idx in range(start_round, num_rounds):
        # 先存下這一輪「實際拿去訓練」的超參數,避免下面被 LLM 的新提案覆蓋掉
        used_hp = current_hp

        run_name = f"{model_key}_round{round_idx}{folder_suffix}"
        train_result = train_run(used_hp, run_name=run_name, settings=settings)
        current_map5095 = train_result["best_map5095"]

        # --- Best Model Tracker:跟 Diagnostic 是平行、獨立的判斷,不需要 LLM ---
        if current_map5095 > best["map5095"]:
            best = {
                "round": round_idx,
                "map5095": current_map5095,
                "hyperparameters": used_hp,
            }
            print(f"[round {round_idx}] mAP50-95={current_map5095:.4f}  -> 進步(新的歷史最佳)")
        else:
            gap = best["map5095"] - current_map5095
            print(f"[round {round_idx}] mAP50-95={current_map5095:.4f}  "
                  f"-> 退步(目前歷史最佳 {best['map5095']:.4f},差 {gap:.4f})")

        # --- Diagnostic Translation(LLM 推理範圍)---
        df = read_log(train_result["results_csv"])
        class_stats = train_result["class_stats"]
        facts = compute_facts(df, class_stats=class_stats)
        flags = diagnose(facts, class_stats=class_stats)
        diagnostic_text = to_report_text(facts, flags)

        budget_text = (
            _format_budget(round_idx, num_rounds) + "\n\n"
            if include_budget_info else ""
        )
        history_text = _format_history(history)
        best_ref_text = _format_best_reference(best, current_map5095)
        report_text = (
            budget_text
            + (history_text + "\n\n" if history_text else "")
            + diagnostic_text
            + best_ref_text
        )

        proposal = _propose_with_retry(model_key, system_prompt, report_text,
                                        round_idx, history)
        new_hp = proposal["new_hyperparameters"]
        current_hp = new_hp
        print(f"[round {round_idx}] LLM 提案下一輪超參數: {_format_hp(new_hp)}")

        history.append({
            "round": round_idx,
            "hyperparameters": used_hp,               # 這一輪實際訓練用的參數
            "map5095": current_map5095,
            "diagnostic_flags": flags,
            "llm_reasoning": proposal.get("reasoning"),
            "llm_base_iteration": proposal.get("base_iteration"),
            "llm_base_rationale": proposal.get("base_rationale"),
            "llm_thinking": proposal.get("_thinking"),  # 只有本機 Ollama 模型有(見 OLLAMA_MODELS 的 think 設定),Claude 模型會是 None
            "llm_proposal_for_next_round": new_hp,     # LLM 對下一輪的提案
        })

        if save_path is not None:
            save_json({
                "model_key": model_key,
                "num_rounds": num_rounds,
                "history": history,
            }, save_path)

        # 見模組開頭 GPU_UNLOAD_BUFFER_SEC 說明:只有本機 Ollama 模型、且
        # 後面還有下一輪要訓練時才需要等待。
        if model_key in OLLAMA_MODELS and round_idx + 1 < num_rounds:
            time.sleep(GPU_UNLOAD_BUFFER_SEC)

    return {
        "best_round": best["round"],
        "best_map5095": best["map5095"],
        "best_hyperparameters": best["hyperparameters"],
        "history": history,
    }


def run_score_only_loop(model_key: str, system_prompt: str, settings: dict,
                         initial_hyperparameters: dict,
                         num_rounds: int = 10,
                         save_path: str | None = None,
                         folder_suffix: str = "") -> dict:
    """
    對照組:只把每輪的最終分數(不含診斷)餵給 LLM。重現 Zhang et al. 2023
    的做法,用來跟 run_hpo_loop 做控制變因對照,驗證 Facts + Diagnostic
    Flags 這套診斷架構本身是否真的有貢獻。

    刻意重用 run_hpo_loop 用的 _format_history / _format_best_reference /
    _validate_hyperparameters / _check_duplicate / _propose_with_retry /
    _load_resume_state,確保兩組的「額外安全機制」(重試/重複偵測/resume/
    歷史最佳參考)完全對等——兩個迴圈唯一的差異只在於有沒有 Facts +
    Diagnostic Flags 那個區塊,不會有其他混淆變因,見 docs/decisions.md。
    2026-09-19 這次沒有加 include_budget_info(刻意只在 run_hpo_loop 測試
    這個假說,見模組開頭第7點說明),但有加 folder_suffix,理由跟
    run_hpo_loop 一樣(避免資料夾撞名),兩邊保持一致。

    save_path / resume 行為跟 run_hpo_loop 一致(見該函式說明)。

    回傳: {"best_round": int, "best_map5095": float, "best_hyperparameters": dict,
           "history": [...]}
    """
    history: list[dict] = []
    best = {"round": -1, "map5095": -1.0, "hyperparameters": None}
    current_hp = initial_hyperparameters
    start_round = 0

    resume_state = _load_resume_state(save_path)
    if resume_state and resume_state.get("history"):
        history = resume_state["history"]
        start_round = len(history)
        current_hp = history[-1]["llm_proposal_for_next_round"]
        for entry in history:
            if entry["map5095"] > best["map5095"]:
                best = {
                    "round": entry["round"],
                    "map5095": entry["map5095"],
                    "hyperparameters": entry["hyperparameters"],
                }
        print(f"[resume] 偵測到 {save_path} 已有 {start_round} 輪進度,"
              f"從 round {start_round} 接續"
              f"(目前歷史最佳 round {best['round']}, mAP50-95={best['map5095']:.4f})")

    for round_idx in range(start_round, num_rounds):
        # 先存下這一輪「實際拿去訓練」的超參數,避免下面被 LLM 的新提案覆蓋掉
        used_hp = current_hp

        run_name = f"score_only_{model_key}_round{round_idx}{folder_suffix}"
        train_result = train_run(used_hp, run_name=run_name, settings=settings)
        current_map5095 = train_result["best_map5095"]

        # --- Best Model Tracker:跟診斷組一樣,是平行、獨立的判斷 ---
        if current_map5095 > best["map5095"]:
            best = {
                "round": round_idx,
                "map5095": current_map5095,
                "hyperparameters": used_hp,
            }
            print(f"[round {round_idx}] mAP50-95={current_map5095:.4f}  -> 進步(新的歷史最佳)")
        else:
            gap = best["map5095"] - current_map5095
            print(f"[round {round_idx}] mAP50-95={current_map5095:.4f}  "
                  f"-> 退步(目前歷史最佳 {best['map5095']:.4f},差 {gap:.4f})")

        # score-only:不算 Facts/Diagnostic Flags(train_run 仍會跑
        # compute_class_stats,但這裡刻意不使用它的回傳值),只把這一輪的
        # 分數跟超參數組成簡單文字,對應 prompts/score_only.md 的
        # 「This Round Result Format」
        this_round_text = (
            "=== THIS ROUND RESULT ===\n"
            f"mAP50-95: {current_map5095:.4f}\n"
            f"Hyperparameters used: {_format_hp(used_hp)}"
        )

        history_text = _format_history(history)
        best_ref_text = _format_best_reference(best, current_map5095)
        report_text = (
            (history_text + "\n\n" if history_text else "")
            + this_round_text
            + best_ref_text
        )

        proposal = _propose_with_retry(model_key, system_prompt, report_text,
                                        round_idx, history)
        new_hp = proposal["new_hyperparameters"]
        current_hp = new_hp
        print(f"[round {round_idx}] LLM 提案下一輪超參數: {_format_hp(new_hp)}")

        history.append({
            "round": round_idx,
            "hyperparameters": used_hp,               # 這一輪實際訓練用的參數
            "map5095": current_map5095,
            "llm_reasoning": proposal.get("reasoning"),
            "llm_base_iteration": proposal.get("base_iteration"),
            "llm_base_rationale": proposal.get("base_rationale"),
            "llm_thinking": proposal.get("_thinking"),  # 只有本機 Ollama 模型有(見 OLLAMA_MODELS 的 think 設定),Claude 模型會是 None
            "llm_proposal_for_next_round": new_hp,     # LLM 對下一輪的提案
        })

        if save_path is not None:
            save_json({
                "model_key": model_key,
                "num_rounds": num_rounds,
                "history": history,
            }, save_path)

        # 見模組開頭 GPU_UNLOAD_BUFFER_SEC 說明:只有本機 Ollama 模型、且
        # 後面還有下一輪要訓練時才需要等待。
        if model_key in OLLAMA_MODELS and round_idx + 1 < num_rounds:
            time.sleep(GPU_UNLOAD_BUFFER_SEC)

    return {
        "best_round": best["round"],
        "best_map5095": best["map5095"],
        "best_hyperparameters": best["hyperparameters"],
        "history": history,
    }
