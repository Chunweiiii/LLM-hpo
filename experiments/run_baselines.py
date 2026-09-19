"""
傳統方法基準線進入點:Random Search / Optuna TPE / Optuna CMA-ES。

依序跑三個方法,每個都用跟 LLM 方法一樣的輪數(預設 = config/settings.yaml
的 hpo.num_rounds,可用 --rounds 覆蓋,用法跟 experiments/run_full_diagnosis.py
一致)。三個方法都不依賴 INITIAL_HYPERPARAMETERS(novice baseline 起始點),
從第一輪就在整個 SEARCH_SPACE 隨機/依演算法取樣,這是刻意的設計,見
docs/decisions.md。

用法:
    python experiments/run_baselines.py                        (跑全部三個,預設輪數)
    python experiments/run_baselines.py --rounds 20            (跑全部三個,20輪)
    python experiments/run_baselines.py --method random_search (只跑其中一個)

中斷接續:每個方法執行中都會即時把進度寫到 results/progress_<method>.json
(有用 --rounds 的話檔名會加 _r<N>)。中途被打斷(當機、重開機等),重新
執行同一條指令就會自動從已完成的部分接續,不用加任何額外參數,見
src/baseline_search.py。
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils import load_settings, save_json
from src.baseline_search import random_search, optuna_search

METHODS = ["random_search", "optuna_tpe", "optuna_cmaes"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=None,
                         help="覆蓋 settings.yaml 的 hpo.num_rounds,只影響這次執行,"
                              "不會改到設定檔")
    parser.add_argument("--method", choices=METHODS, default=None,
                         help="只跑指定方法,不指定的話依序跑全部三個")
    args = parser.parse_args()

    settings = load_settings()
    num_rounds = args.rounds if args.rounds is not None else settings["hpo"]["num_rounds"]
    suffix = f"_r{num_rounds}" if args.rounds is not None else ""

    methods_to_run = [args.method] if args.method else METHODS

    if "random_search" in methods_to_run:
        print(f"=== Random Search ({num_rounds} rounds) ===")
        save_path = f"results/progress_random_search{suffix}.json"
        result = random_search(settings, num_rounds=num_rounds, save_path=save_path)
        save_json(result, f"results/baseline_random_search{suffix}.json")
        print(f"=== Done: Random Search, {len(result)} rounds saved ===")

    if "optuna_tpe" in methods_to_run:
        print(f"=== Optuna TPE ({num_rounds} rounds) ===")
        save_path = f"results/progress_optuna_tpe{suffix}.json"
        result = optuna_search(settings, sampler_name="tpe", num_rounds=num_rounds,
                                save_path=save_path)
        save_json(result, f"results/baseline_optuna_tpe{suffix}.json")
        print(f"=== Done: Optuna TPE, {len(result)} rounds saved ===")

    if "optuna_cmaes" in methods_to_run:
        print(f"=== Optuna CMA-ES ({num_rounds} rounds) ===")
        save_path = f"results/progress_optuna_cmaes{suffix}.json"
        result = optuna_search(settings, sampler_name="cmaes", num_rounds=num_rounds,
                                save_path=save_path)
        save_json(result, f"results/baseline_optuna_cmaes{suffix}.json")
        print(f"=== Done: Optuna CMA-ES, {len(result)} rounds saved ===")


if __name__ == "__main__":
    main()