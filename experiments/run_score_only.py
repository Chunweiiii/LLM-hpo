"""
score-only 對照組進入點:重現 Zhang et al. 2023 做法,每輪只給最終分數
(不含任何診斷),跑 N 輪(N 預設 = config/settings.yaml 的 hpo.num_rounds,
可用 --rounds 覆蓋),跟 experiments/run_full_diagnosis.py 同一個模型的結果
做控制變因對照,驗證 Facts + Diagnostic Flags 這套診斷架構本身是否有貢獻。

用法跟 run_full_diagnosis.py 一致:
    python experiments/run_score_only.py                    (跑全部四個,預設輪數)
    python experiments/run_score_only.py --model qwen3-14b  (只跑 Qwen,預設輪數)
    python experiments/run_score_only.py --model qwen3-14b --rounds 20

中斷接續:每個模型執行中都會即時把進度寫到
results/progress_score_only_<model_key><suffix>.json(suffix 由 --rounds
決定,見下方)。中途被打斷重新執行同一條指令會自動接續,見 src/hpo_loop.py。

2026-09-19 修正(見 src/hpo_loop.py 模組開頭第6點):suffix 現在也會傳進
`folder_suffix`,讓 `runs/detect/score_only_<model_key>_round<N>` 這個
訓練資料夾名稱跟著 suffix 走,避免用 --rounds 重跑同一個模型當第二次獨立
複製時,撞到第一次的原始訓練資料夾(json 結果檔本身不會撞名,但資料夾
之前會撞,見 docs/decisions.md 的事故記錄)。這支腳本目前沒有
--budget-aware 選項(那個只加在 run_full_diagnosis.py,見該檔案說明)。
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils import load_settings, save_json
from src.hpo_loop import run_score_only_loop

MODELS = ["claude-haiku-4.5", "claude-sonnet-5", "claude-opus-5", "qwen3-14b"]

# 跟 experiments/run_full_diagnosis.py 用同一組起始超參數,兩組唯一該有的
# 差異只能是「有沒有診斷資訊」,起始點、輪數、模型都要對齊,才有控制變因
# 對照的意義,見 docs/decisions.md。
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
                              "不會改到設定檔")
    args = parser.parse_args()

    models_to_run = [args.model] if args.model else MODELS

    settings = load_settings()
    with open("prompts/score_only.md", "r", encoding="utf-8") as f:
        system_prompt = f.read()

    num_rounds = args.rounds if args.rounds is not None else settings["hpo"]["num_rounds"]
    suffix = f"_r{num_rounds}" if args.rounds is not None else ""

    for model_key in models_to_run:
        print(f"=== Running score-only HPO loop: {model_key} "
              f"({num_rounds} rounds) ===")
        save_path = f"results/progress_score_only_{model_key}{suffix}.json"
        result = run_score_only_loop(
            model_key=model_key,
            system_prompt=system_prompt,
            settings=settings,
            initial_hyperparameters=INITIAL_HYPERPARAMETERS,
            num_rounds=num_rounds,
            save_path=save_path,
            folder_suffix=suffix,
        )
        save_json(result, f"results/score_only_{model_key}{suffix}.json")
        print(f"=== Done: {model_key}, best round={result['best_round']}, "
              f"best_map5095={result['best_map5095']:.4f} ===")


if __name__ == "__main__":
    main()
