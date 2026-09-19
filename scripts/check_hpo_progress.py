"""
查詢 HPO 迴圈目前跑到第幾輪、每輪用了什麼超參數、跑出來的最佳 mAP50-95,
並在最後統計出目前為止(不管有沒有跑完全部輪次)最佳的一輪。
不用滑終端機或猜測,直接讀 runs/detect/<model_key>_round*/ 底下的
args.yaml 跟 results.csv(還在訓練中、還沒產生完整檔案的那一輪會被標示
「訓練中」,不會報錯中斷)。

用法:
    python scripts/check_hpo_progress.py --model qwen3-14b
"""
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import yaml
import pandas as pd

TRACKED_KEYS = ["lr0", "weight_decay", "mosaic", "hsv_v", "box"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="例如 qwen3-14b")
    args = parser.parse_args()

    round_dirs = sorted(
        Path("runs/detect").glob(f"{args.model}_round*"),
        key=lambda p: int(p.name.split("round")[-1]),
    )

    if not round_dirs:
        print(f"找不到 runs/detect/{args.model}_round* 任何資料夾")
        return

    best_round_idx = None
    best_map = -1.0
    best_hp_str = None
    completed_count = 0

    for round_dir in round_dirs:
        round_idx = round_dir.name.split("round")[-1]
        args_yaml = round_dir / "args.yaml"
        results_csv = round_dir / "results.csv"

        if not args_yaml.exists():
            print(f"round{round_idx}: 還沒開始訓練(沒有 args.yaml)")
            continue

        hp = yaml.safe_load(args_yaml.read_text(encoding="utf-8"))
        hp_str = ", ".join(f"{k}={hp.get(k)}" for k in TRACKED_KEYS)

        if not results_csv.exists():
            print(f"round{round_idx}: {hp_str}  ->  訓練中(還沒有 results.csv)")
            continue

        df = pd.read_csv(results_csv)
        map_col = [c for c in df.columns if "map50-95" in c.lower()][0]
        this_map = df[map_col].max()
        print(f"round{round_idx}: {hp_str}  ->  best_map5095={this_map:.4f}")

        completed_count += 1
        if this_map > best_map:
            best_map = this_map
            best_round_idx = round_idx
            best_hp_str = hp_str

    print("\n=== 統計 ===")
    print(f"目前完整跑完的輪次數: {completed_count}")
    if best_round_idx is not None:
        print(f"目前為止最佳: round{best_round_idx}, mAP50-95={best_map:.4f}")
        print(f"對應超參數: {best_hp_str}")
    else:
        print("目前沒有任何一輪完整跑完。")


if __name__ == "__main__":
    main()