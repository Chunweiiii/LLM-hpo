"""
補跑獨立 test set(94 張,datasets/test/)的最終驗證,回應雪糕的問題。

重點:HPO/挑超參數全程只碰過 validation set(datasets/valid/,187 張),
test set 完全沒被用來訓練或挑模型——這份腳本是 HPO 結束後才額外執行的,
不會回頭去影響任何一輪訓練或 best_hyperparameters 的選擇,所以 test 的
held-out 性質不受影響。

跑的對象:
  1. baseline:5 個 seed(baseline_3seed_seed0~4)的 best.pt,各自在 test
     set 上跑一次,回報 mean ± std(跟 val 版 baseline 的報告方式一致)。
  2. 四個主線模型各自的最佳一輪(Diagnosis 條件,round 數字取自
     results/full_diagnosis_<model>.json 的 best_round):
       - Llama 3.1 8B  -> round2
       - Phi-4 14B     -> round9
       - Claude Haiku 4.5 -> round7
       - Gemma 4 12B   -> round5

用法(專案根目錄、venv 啟用):
    python scripts\\run_final_test_eval.py

結果存到 results/final_test_eval.json,同時印出每個模型 val vs test 的
對照(方便直接判斷有沒有明顯的 val/test gap)。
"""
import json
from pathlib import Path

import yaml
from ultralytics import YOLO

SETTINGS_PATH = Path("config/settings.yaml")
OUT_PATH = Path("results/final_test_eval.json")

# (run 資料夾名稱, 論文裡對應的 val best_map5095, 方便印出來對照)
BASELINE_SEEDS = [
    "baseline_3seed_seed0",
    "baseline_3seed_seed1",
    "baseline_3seed_seed2",
    "baseline_3seed_seed3",
    "baseline_3seed_seed4",
]

FINAL_MODELS = [
    ("Llama 3.1 8B", "llama3.1-8b_round2_hinted", 0.66959),
    ("Phi-4 14B", "phi4-14b_round9_hinted", 0.66493),
    ("Claude Haiku 4.5", "claude-haiku-4.5_round7_hinted", 0.66337),
    ("Gemma 4 12B", "gemma4-12b_round5_hinted", 0.65932),
]


def load_settings():
    with open(SETTINGS_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def eval_on_test(weights_path: Path, settings: dict) -> float:
    """對一個 best.pt 跑 split='test',回傳 mAP50-95。"""
    model = YOLO(str(weights_path))
    metrics = model.val(
        data=str(Path(settings["dataset"]["root"]) / "data.yaml"),
        split="test",
        batch=settings["training"]["batch_size"],
        workers=settings["training"].get("workers", 4),
        verbose=False,
    )
    return float(metrics.box.map)  # mAP50-95 (all classes)


def main():
    settings = load_settings()
    results = {"baseline": {}, "final_models": {}}

    print("=== Baseline(5-seed,test set)===")
    baseline_test_vals = []
    for run_name in BASELINE_SEEDS:
        weights = Path("runs/detect") / run_name / "weights" / "best.pt"
        if not weights.exists():
            print(f"  [跳過] 找不到 {weights}")
            continue
        test_map = eval_on_test(weights, settings)
        baseline_test_vals.append(test_map)
        results["baseline"][run_name] = test_map
        print(f"  {run_name}: test mAP50-95 = {test_map:.5f}")

    if baseline_test_vals:
        mean = sum(baseline_test_vals) / len(baseline_test_vals)
        std = (sum((v - mean) ** 2 for v in baseline_test_vals) / len(baseline_test_vals)) ** 0.5
        results["baseline_summary"] = {"mean": mean, "std": std, "n": len(baseline_test_vals)}
        print(f"  -> baseline test mean±std = {mean:.5f} ± {std:.5f}  (val 版本: 0.6198 ± 0.0110)")

    print()
    print("=== 四個主線模型(Diagnosis 條件最佳一輪,test set)===")
    for label, run_name, val_map in FINAL_MODELS:
        weights = Path("runs/detect") / run_name / "weights" / "best.pt"
        if not weights.exists():
            print(f"  [跳過] 找不到 {weights}")
            continue
        test_map = eval_on_test(weights, settings)
        gap = test_map - val_map
        results["final_models"][label] = {
            "run_name": run_name,
            "val_map5095": val_map,
            "test_map5095": test_map,
            "val_minus_test_gap": gap,
        }
        print(f"  {label:20s} val={val_map:.5f}  test={test_map:.5f}  gap={gap:+.5f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n存到 {OUT_PATH}")


if __name__ == "__main__":
    main()
