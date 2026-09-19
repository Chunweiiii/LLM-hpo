"""
傳統 HPO 方法基準線:Random Search / Optuna TPE / Optuna CMA-ES。

搜尋空間需跟 LLM 方法用的超參數對齊,才能公平比較(見 docs/decisions.md
「可調超參數組合」那筆紀錄:5 個參數是在「10 輪預算下維度不能太高」跟
「每個參數盡量對應到至少一個 D1-D5 診斷旗標」兩個考量下定案的)。

2026-09 韌性強化(同 src/hpo_loop.py 的理由,電腦可能被 Windows 自動更新
強制重開機):每輪(random search)/每個 trial(optuna)結束後即時把目前
結果寫進 save_path,重跑同一條指令會自動從已完成的部分接續,不會整批
遺失。Optuna 的部分,接續時是用一個全新的 study 只跑剩下的輪數(沒有把
之前已完成的 trial 餵回去讓 TPE/CMA-ES 的取樣模型「記得」,這是刻意簡化
過的版本,換取邏輯簡單、不容易出錯;在 10~20 輪這種小預算下,犧牲的
取樣效率有限)。
"""

from pathlib import Path

import random
import optuna

from .train_runner import train_run
from .utils import save_json, load_json

# 2026-09 定案(見 docs/decisions.md):5 個參數,lr0/weight_decay 對應
# D1/D2/D3(收斂、loss-mAP脫鉤、過擬合),box 對應 D4(定位瓶頸),
# mosaic/hsv_v 對應 D5(類別不平衡,用資料增強手段調整)。
# box 的範圍 (2.0, 15.0) 是以 YOLOv8 官方預設值 7.5 為中心抓的合理區間
# (約 0.27~2 倍),不是文獻實證出來的精確範圍,見 docs/decisions.md 誠實記錄。
SEARCH_SPACE = {
    "lr0": (1e-4, 1e-2),
    "weight_decay": (1e-5, 1e-2),
    "mosaic": (0.0, 1.0),
    "hsv_v": (0.0, 1.0),
    "box": (2.0, 15.0),
}


def _load_resume_results(save_path: str) -> list:
    if save_path is None or not Path(save_path).exists():
        return []
    try:
        return load_json(save_path)
    except Exception as e:
        print(f"[resume] 讀取 {save_path} 失敗,視為沒有進度可接續:{e}")
        return []


def random_search(settings: dict, num_rounds: int = 10, seed: int = 0,
                   save_path: str = None) -> list:
    """隨機搜尋基準線。

    save_path:每輪結束後即時存檔,重跑同一條指令會自動接續(見模組開頭
    說明)。rng 每輪都會抽樣一次(即使是已經跑過、要跳過的輪次),確保
    resume 前後的隨機序列一致,可重現。
    """
    rng = random.Random(seed)
    results = _load_resume_results(save_path)
    start_round = len(results)
    if start_round > 0:
        print(f"[resume] random_search 已有 {start_round} 輪結果,從 round {start_round} 接續")

    for i in range(num_rounds):
        hp = {k: rng.uniform(*v) for k, v in SEARCH_SPACE.items()}
        if i < start_round:
            continue
        result = train_run(hp, run_name=f"random_search_{i}", settings=settings)
        results.append({"round": i, "hyperparameters": hp, **result})
        if save_path is not None:
            save_json(results, save_path)
    return results


def optuna_search(settings: dict, sampler_name: str = "tpe",
                   num_rounds: int = 10, save_path: str = None) -> list:
    """
    Optuna 搜尋基準線,sampler_name 可選 "tpe" 或 "cmaes"。
    CMA-ES 只需要換 sampler,工作量很小。

    save_path:每個 trial 結束後即時存檔,重跑同一條指令會自動接續(見
    模組開頭說明:接續時用全新的 study 只跑剩下的輪數,不會把之前的
    trial 餵回去讓取樣模型記得,是刻意簡化過的版本)。
    """
    sampler = {
        "tpe": optuna.samplers.TPESampler(),
        "cmaes": optuna.samplers.CmaEsSampler(),
    }[sampler_name]

    results = _load_resume_results(save_path)
    start_round = len(results)
    remaining = num_rounds - start_round
    if start_round > 0:
        print(f"[resume] optuna_{sampler_name} 已有 {start_round} 輪結果,"
              f"還需要 {remaining} 輪")
    if remaining <= 0:
        return results

    def objective(trial):
        hp = {k: trial.suggest_float(k, *v) for k, v in SEARCH_SPACE.items()}
        round_idx = start_round + trial.number
        result = train_run(hp, run_name=f"optuna_{sampler_name}_{round_idx}",
                            settings=settings)
        results.append({"round": round_idx, "hyperparameters": hp, **result})
        if save_path is not None:
            save_json(results, save_path)
        return result["best_map5095"]

    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=remaining)
    return results