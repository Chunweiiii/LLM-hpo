"""
從 3-seed baseline 的結果,推導 D4(Localization Bottleneck)和 D5
(Class-Level Imbalance)的判斷閾值,取代 src/summarize_log.py 跟
prompts/program.md 裡的佔位符(0.20 / 0.10)。

D4:用每個 seed 最終 epoch 的 mAP50 - mAP50-95 落差,算出平均值 + 1 個標準差
   當閾值。意義:baseline 本身就會有正常震盪的落差,只有新一輪訓練的落差
   超過這個正常範圍的上界,才算異常,值得標記為 localization bottleneck。

D5:用每個 seed 的最強類別 vs 最弱類別 AP50-95 落差,算出平均值 + 1 個標準差
   當閾值,邏輯同上。

注意:目前只有 3 個 seed,標準差估計本身會比較不穩定,這是這個階段能做到
最誠實的估計方式,之後如果有更多 seed 資料可以再重算,決策記錄裡會註明。

補算說明:如果 results/baseline_3seed.json 裡的某個 seed 缺 class_stats
(例如在還沒寫 compute_class_stats() 之前就先跑完的舊資料),這個腳本會
自動用該 seed 已經存在的 best.pt 權重補跑一次驗證(不是重新訓練),補齊後
一併寫回 results/baseline_3seed.json,不需要另外開檔案處理。
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd

from src.utils import load_settings
from src.train_runner import compute_class_stats


def _find_map_cols(df):
    cols = df.columns.tolist()
    map5095_col = [c for c in cols if "map50-95" in c.lower()][0]
    map50_col = [c for c in cols if "map50" in c.lower() and "map50-95" not in c.lower()][0]
    return map50_col, map5095_col


def main():
    settings = load_settings()
    summary_path = Path("results/baseline_3seed.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    d4_gaps = []
    d5_gaps = []
    need_resave = False

    for run in summary["per_seed_results"]:
        seed = run["seed"]

        # 缺 class_stats,或雖然有但 n 是 None(例如舊版 compute_class_stats
        # 抓錯屬性位置導致補算失敗)的話,重新補算(用已存在的 best.pt,
        # 不重新訓練)
        missing = "class_stats" not in run
        stale_n = (not missing) and any(
            v.get("n") is None for v in run["class_stats"].values()
        )
        if missing or stale_n:
            results_csv_path = Path(run["results_csv"])
            best_weights = results_csv_path.parent / "weights" / "best.pt"
            print(f"[seed={seed}] 補算 class_stats,用 {best_weights}...")
            run["class_stats"] = compute_class_stats(str(best_weights), settings)
            need_resave = True

        # D4:該 seed 最終 epoch 的 mAP50 / mAP50-95 落差
        df = pd.read_csv(run["results_csv"])
        map50_col, map5095_col = _find_map_cols(df)
        final_map50 = float(df[map50_col].iloc[-1])
        final_map5095 = float(df[map5095_col].iloc[-1])
        gap4 = final_map50 - final_map5095
        d4_gaps.append(gap4)
        print(f"[seed={seed}] D4 gap (final mAP50 - mAP50-95) = {gap4:.4f}")

        # D5:該 seed 最強類別 vs 最弱類別 AP50-95 落差
        class_stats = run["class_stats"]
        ap5095_values = {name: stats["ap5095"] for name, stats in class_stats.items()}
        strongest = max(ap5095_values, key=ap5095_values.get)
        weakest = min(ap5095_values, key=ap5095_values.get)
        gap5 = ap5095_values[strongest] - ap5095_values[weakest]
        d5_gaps.append(gap5)
        print(f"[seed={seed}] D5 gap ({strongest} - {weakest}) = {gap5:.4f}")

    if need_resave:
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\n(已把補算的 class_stats 寫回 {summary_path})")

    d4_mean, d4_std = float(np.mean(d4_gaps)), float(np.std(d4_gaps, ddof=1))
    d5_mean, d5_std = float(np.mean(d5_gaps)), float(np.std(d5_gaps, ddof=1))

    d4_threshold = d4_mean + d4_std
    d5_threshold = d5_mean + d5_std

    print()
    print(f"D4: mean={d4_mean:.4f}, std={d4_std:.4f}  ->  threshold (mean+1std) = {d4_threshold:.4f}")
    print(f"D5: mean={d5_mean:.4f}, std={d5_std:.4f}  ->  threshold (mean+1std) = {d5_threshold:.4f}")

    out = {
        "d4_gaps_per_seed": d4_gaps,
        "d4_mean": d4_mean,
        "d4_std": d4_std,
        "d4_threshold": d4_threshold,
        "d5_gaps_per_seed": d5_gaps,
        "d5_mean": d5_mean,
        "d5_std": d5_std,
        "d5_threshold": d5_threshold,
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/d4_d5_thresholds.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n已存到 results/d4_d5_thresholds.json")


if __name__ == "__main__":
    main()
