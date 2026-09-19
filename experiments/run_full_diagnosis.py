"""
完整診斷組進入點:用完整兩層診斷(Facts + 5 Diagnostic Flags)跑 N 輪
HPO 迴圈(N 預設 = config/settings.yaml 的 hpo.num_rounds,可用 --rounds 覆蓋)。

預設依序跑全部四個模型(Claude Haiku 4.5 / Sonnet 5 / Opus 5 / Qwen3-14B),
也可以用 --model 只跑其中一個,例如先跑 Qwen 驗證整條線,之後 Claude API
key 設好了再分開跑其他三個,不用等全部一起跑完。

用法:
    python experiments/run_full_diagnosis.py                    (跑全部四個,預設輪數)
    python experiments/run_full_diagnosis.py --model qwen3-14b  (只跑 Qwen,預設輪數)
    python experiments/run_full_diagnosis.py --model qwen3-14b --rounds 20
        (只跑 Qwen,這次跑 20 輪,不影響 settings.yaml 的預設值,
         也不影響之後跑其他模型時的輪數,避免不小心把 Claude 模型也跑成
         20 輪、多花一倍 API 費用)
    python experiments/run_full_diagnosis.py --model claude-opus-5 --budget-aware
        (2026-09-19 新增,見下方「--budget-aware」說明)

注意:每個模型跑 N 輪,每輪都是一次完整的 100-epoch 訓練,耗時很長,
建議先用 --model 只跑一個確認沒問題,再考慮要不要全部一起跑。

中斷接續:每個模型執行中都會即時把進度寫到 results/progress_<model_key><suffix>.json
(suffix 由 --rounds/--budget-aware 決定,見下方)。如果中途被打斷(當機、
重開機等),重新執行同一條指令就會自動從上次中斷的地方接續,不用加任何
額外參數。如果是要「全新開始」(例如換了新的 baseline/起始超參數,像
2026-09 這次從 SEARCH_SPACE 中點版換成 novice baseline 版),記得先手動
刪掉舊的 results/progress_<model_key>*.json,不然會被誤判成接續舊的進度。

--budget-aware(2026-09-19 新增,見 docs/decisions.md、src/hpo_loop.py 模組
開頭第7點說明):使用者觀察到 Opus 5 完整診斷組(第一次跑)的推理文字裡
明顯採取「像做嚴謹科學實驗一樣,一次只調一個變因、還規劃好幾輪以後要測
什麼」的長線策略,懷疑原因是系統提示從來沒告訴 LLM 這個優化總共只有 10
輪預算。加上這個旗標,會改用 prompts/program_budget_aware.md(比
program.md 多一段「Optimization Budget」說明),每輪報告也會多一段
「=== OPTIMIZATION BUDGET ===」,誠實告知目前第幾輪/總共幾輪/還剩幾輪。
刻意獨立成新的 flag、新的檔名後綴(`_budgetaware`)、新的
`runs/detect/` 資料夾後綴,不會跟第一次(不知道預算)的結果撞名或混淆,
兩份結果可以並列比較。這次刻意只用在完整診斷組(不影響 run_score_only.py),
先只測 Opus/Sonnet 兩個模型,驗證假說是否成立,有效的話再考慮 Haiku/Qwen
要不要也補跑。
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils import load_settings, save_json
from src.hpo_loop import run_hpo_loop

MODELS = ["claude-haiku-4.5", "claude-sonnet-5", "claude-opus-5", "qwen3-14b"]

# 2026-09 教授指出舊版(SEARCH_SPACE 中點)baseline 太高、跟 HPO 上限太接近
# (ceiling effect),改用 Ultralytics 框架本身未經修改的預設值,模擬「完全
# 沒調過參數的新手」狀態,取代原本的中點版數字。這組數字是直接查證
# Ultralytics 原始碼得出的(`python -c "from ultralytics.cfg import
# DEFAULT_CFG; print(DEFAULT_CFG)"`),不是猜測,見 docs/decisions.md
# 「Baseline 起始點重新設計:novice baseline」。務必跟
# scripts/run_baseline_3seed.py、experiments/run_score_only.py 用同一組,
# 否則 HPO 迴圈的起始點會跟 baseline 對不上,比較就失去意義。
INITIAL_HYPERPARAMETERS = {
    "lr0": 0.01,
    "weight_decay": 0.0005,
    "mosaic": 1.0,
    "hsv_v": 0.4,
    "box": 7.5,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, default=None,
                         help="只跑指定模型,不指定的話依序跑全部四個")
    parser.add_argument("--rounds", type=int, default=None,
                         help="覆蓋 settings.yaml 的 hpo.num_rounds,只影響這次執行,"
                              "不會改到設定檔(例如 2026-09 輪數敏感度分析,只用 Qwen "
                              "多跑幾輪,不影響之後 Claude 模型的預設輪數)")
    parser.add_argument("--budget-aware", action="store_true",
                         help="讓 LLM 每輪知道目前第幾輪/總共幾輪/還剩幾輪,"
                              "見上方模組說明跟 docs/decisions.md")
    args = parser.parse_args()

    models_to_run = [args.model] if args.model else MODELS

    settings = load_settings()
    prompt_file = (
        "prompts/program_budget_aware.md" if args.budget_aware
        else "prompts/program.md"
    )
    with open(prompt_file, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    num_rounds = args.rounds if args.rounds is not None else settings["hpo"]["num_rounds"]

    # 檔名/資料夾後綴:--rounds 不是預設值的話加 "_r<N>",--budget-aware
    # 再加 "_budgetaware",兩個可以疊加(目前用不到疊加,但保留彈性)。
    # 這個 suffix 同時用在 results/ 底下的 json 檔名跟
    # runs/detect/<run_name> 資料夾命名(見 folder_suffix 參數),確保兩邊
    # 一一對應、不會撞名,2026-09-19 修正,見 src/hpo_loop.py 模組開頭第6點。
    suffix_parts = []
    if args.rounds is not None:
        suffix_parts.append(f"r{args.rounds}")
    if args.budget_aware:
        suffix_parts.append("budgetaware")
    suffix = ("_" + "_".join(suffix_parts)) if suffix_parts else ""

    for model_key in models_to_run:
        print(f"=== Running full-diagnosis HPO loop: {model_key} "
              f"({num_rounds} rounds{', budget-aware' if args.budget_aware else ''}) ===")
        save_path = f"results/progress_{model_key}{suffix}.json"
        result = run_hpo_loop(
            model_key=model_key,
            system_prompt=system_prompt,
            settings=settings,
            initial_hyperparameters=INITIAL_HYPERPARAMETERS,
            num_rounds=num_rounds,
            save_path=save_path,
            include_budget_info=args.budget_aware,
            folder_suffix=suffix,
        )
        save_json(result, f"results/full_diagnosis_{model_key}{suffix}.json")
        print(f"=== Done: {model_key}, best round={result['best_round']}, "
              f"best_map5095={result['best_map5095']:.4f} ===")


if __name__ == "__main__":
    main()
