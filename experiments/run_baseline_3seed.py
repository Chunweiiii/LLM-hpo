"""
Novice baseline 的 N-seed 穩健性驗證:固定超參數(跟 run_full_diagnosis.py 的
INITIAL_HYPERPARAMETERS 一致),只換隨機種子重複訓練多次,算出 mAP50-95 的
mean/std,推導雜訊門檻(noise_threshold = 2 * std),給 summarize_log.py 的
D1-D5 診斷判斷「這個差距是不是雜訊」用。

2026-09 從原本的 3-seed(seed=0,1,2)擴充到 5-seed(新增 seed=3,4):3-seed
時發現 seed1 明顯偏低,但用 results.csv 的逐 epoch loss 跟 class_stats 檢查過,
確認是正常訓練變異(seed1 的 loss 全面偏高、且落差主要來自 yellow light 這個
樣本數最少的類別,不是訓練中斷或資料損毀),不是 bug。既然是真實變異,
n=3 對這個變異數的估計太不穩定(單一極端值影響力過大),所以擴充到 5 個
seed 稀釋這個影響,不是為了把 seed1 換掉或剔除,見 docs/decisions.md。

用法:
    python experiments/run_baseline_3seed.py                  (預設跑 seed 0~4,共5個)
    python experiments/run_baseline_3seed.py --seeds 0,1,2,3,4 (效果同上,顯式指定)
    python experiments/run_baseline_3seed.py --seeds 3,4       (只補跑新增的兩個)

中斷接續:每個 seed 訓練完就立刻存檔,重新執行同一條指令會自動跳過已經
跑完的 seed,不用加任何參數,也不會覆蓋掉已有的 seed0/1/2 結果。
"""

import argparse
import statistics
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.train_runner import train_run
from src.utils import load_settings, save_json, load_json

RESULT_PATH = "results/baseline_3seed.json"

# 跟 experiments/run_full_diagnosis.py 的 INITIAL_HYPERPARAMETERS 完全一致,
# 3-seed / N-seed 驗證測的是「同一組固定超參數」在不同隨機種子下的變異,
# 不是搜尋,所以必須固定用這組 novice baseline 起始值。
NOVICE_HYPERPARAMETERS = {
    "lr0": 0.01,
    "weight_decay": 0.0005,
    "mosaic": 1.0,
    "hsv_v": 0.4,
    "box": 7.5,
}


def _load_existing() -> dict:
    path = Path(RESULT_PATH)
    if not path.exists():
        return {"per_seed_results": []}
    try:
        return load_json(RESULT_PATH)
    except Exception as e:
        print(f"[resume] 讀取 {RESULT_PATH} 失敗,視為沒有既有結果:{e}")
        return {"per_seed_results": []}


def _recompute_and_save(per_seed_results: list):
    values = [r["best_map5095"] for r in per_seed_results]
    seeds = sorted(r["seed"] for r in per_seed_results)
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    noise_threshold = 2 * std
    save_json({
        "seeds": seeds,
        "per_seed_results": per_seed_results,
        "map5095_mean": mean,
        "map5095_std": std,
        "noise_threshold": noise_threshold,
    }, RESULT_PATH)
    return mean, std, noise_threshold


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4",
                         help="要驗證的 seed 清單,逗號分隔,預設 0,1,2,3,4")
    args = parser.parse_args()
    target_seeds = [int(s) for s in args.seeds.split(",")]

    settings = load_settings()
    existing = _load_existing()
    per_seed_results = existing.get("per_seed_results", [])
    done_seeds = {r["seed"] for r in per_seed_results}

    if done_seeds:
        print(f"[resume] 已有 seed {sorted(done_seeds)} 的結果,"
              f"還需要跑 {[s for s in target_seeds if s not in done_seeds]}")

    for seed in target_seeds:
        if seed in done_seeds:
            continue
        print(f"=== baseline_3seed seed{seed} ===")
        result = train_run(NOVICE_HYPERPARAMETERS, run_name=f"baseline_3seed_seed{seed}",
                            settings=settings, seed=seed)
        result["seed"] = seed
        per_seed_results.append(result)
        mean, std, noise_threshold = _recompute_and_save(per_seed_results)
        print(f"=== Done: seed{seed}, best_map5095={result['best_map5095']:.5f} ===")
        print(f"目前 mean={mean:.4f} std={std:.4f} noise_threshold={noise_threshold:.4f}")

    print(f"\n全部完成,共 {len(per_seed_results)} 個 seed")


if __name__ == "__main__":
    main()