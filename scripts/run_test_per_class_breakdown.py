"""
針對四個主線模型(Diagnosis 條件最佳一輪),在獨立 test set(94 張)上
跑 per-class(紅/黃/綠燈個別)AP50 / AP50-95 / Precision / Recall / n,
用來查 Claude Haiku 4.5 在 test set 上 mAP 掉得特別多(val 0.66337 ->
test 0.59401,-10.46%)是不是跟某個類別(尤其是 yellow light,test set
只有 8 個 instance)特別差有關。

不動 baseline(baseline 不是這次要查的重點,整體 mAP 版的 baseline 已經
跑過,見 results/final_test_eval.json)。

屬性名稱跟 src/train_runner.py 的 compute_class_stats() 是同一套
(ap50/ap/p/r/ap_class_index 在 metrics.box 底下,nt_per_class 在
metrics 底下,8.4.152 版實測過),唯一差異是這裡明確傳 split="test"
(compute_class_stats 原本沒傳 split,預設是 val)。

用法(專案根目錄、venv 啟用):
    python scripts\\run_test_per_class_breakdown.py

結果存到 results/test_per_class_breakdown.json,同時印出每個模型
各類別的對照表。
"""
import json
from pathlib import Path

import yaml
from ultralytics import YOLO

SETTINGS_PATH = Path("config/settings.yaml")
OUT_PATH = Path("results/test_per_class_breakdown.json")

FINAL_MODELS = [
    ("Llama 3.1 8B", "llama3.1-8b_round2_hinted"),
    ("Phi-4 14B", "phi4-14b_round9_hinted"),
    ("Claude Haiku 4.5", "claude-haiku-4.5_round7_hinted"),
    ("Gemma 4 12B", "gemma4-12b_round5_hinted"),
]


def load_settings():
    with open(SETTINGS_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def per_class_on_test(weights_path: Path, settings: dict) -> dict:
    """對一個 best.pt 在 test set 上跑驗證,回傳每個類別的 AP50/AP50-95/P/R/n。"""
    model = YOLO(str(weights_path))
    metrics = model.val(
        data=str(Path(settings["dataset"]["root"]) / "data.yaml"),
        split="test",
        batch=settings["training"]["batch_size"],
        workers=settings["training"].get("workers", 4),
        verbose=False,
    )

    names = metrics.names  # {class_idx: class_name}
    class_stats = {}
    for i, class_idx in enumerate(metrics.box.ap_class_index):
        class_name = names[int(class_idx)]
        n = None
        if hasattr(metrics, "nt_per_class"):
            n = int(metrics.nt_per_class[int(class_idx)])
        class_stats[class_name] = {
            "ap50": float(metrics.box.ap50[i]),
            "ap5095": float(metrics.box.ap[i]),
            "precision": float(metrics.box.p[i]),
            "recall": float(metrics.box.r[i]),
            "n": n,
        }
    return {
        "overall_map5095": float(metrics.box.map),
        "per_class": class_stats,
    }


def main():
    settings = load_settings()
    results = {}

    print("=== 四個主線模型 per-class breakdown(test set,94 張)===\n")
    for label, run_name in FINAL_MODELS:
        weights = Path("runs/detect") / run_name / "weights" / "best.pt"
        if not weights.exists():
            print(f"  [跳過] 找不到 {weights}")
            continue

        stats = per_class_on_test(weights, settings)
        results[label] = {"run_name": run_name, **stats}

        print(f"--- {label} (overall test mAP50-95 = {stats['overall_map5095']:.5f}) ---")
        print(f"  {'class':<14s} {'n':>4s} {'AP50':>8s} {'AP50-95':>8s} {'P':>8s} {'R':>8s}")
        for class_name, c in stats["per_class"].items():
            print(
                f"  {class_name:<14s} {c['n'] if c['n'] is not None else '-':>4} "
                f"{c['ap50']:>8.4f} {c['ap5095']:>8.4f} {c['precision']:>8.4f} {c['recall']:>8.4f}"
            )
        print()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"存到 {OUT_PATH}")


if __name__ == "__main__":
    main()
