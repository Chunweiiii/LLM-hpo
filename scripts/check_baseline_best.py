"""
查詢傳統方法基準線(Random Search/Optuna TPE/Optuna CMA-ES)每一輪的結果,
跟目前最佳。可以查已經跑完的最終結果檔(results/baseline_*.json),也可以
查還在跑的進度檔(results/progress_*.json),格式一樣。

用法:
    python scripts/check_baseline_best.py results/baseline_random_search.json
"""
import json
import sys
from pathlib import Path

TRACKED_KEYS = ["lr0", "weight_decay", "mosaic", "hsv_v", "box"]


def _format_hp(hp: dict) -> str:
    return ", ".join(f"{k}={hp[k]}" for k in TRACKED_KEYS)


def main():
    if len(sys.argv) != 2:
        print("用法:python scripts/check_baseline_best.py <結果json路徑>")
        return

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"找不到 {path}")
        return

    with open(path, encoding="utf-8") as f:
        results = json.load(f)

    if not results:
        print("目前還沒有任何完成的輪次")
        return

    for r in sorted(results, key=lambda r: r["round"]):
        print(f"round{r['round']}: {_format_hp(r['hyperparameters'])}  "
              f"-> best_map5095={r['best_map5095']:.4f}")

    print(f"\n目前共 {len(results)} 輪已完成")
    best = max(results, key=lambda r: r["best_map5095"])
    print(f"最佳: round={best['round']}  best_map5095={best['best_map5095']:.4f}")
    print(f"超參數: {_format_hp(best['hyperparameters'])}")


if __name__ == "__main__":
    main()