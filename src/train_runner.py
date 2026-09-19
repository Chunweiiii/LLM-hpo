"""
包裝 YOLOv8n 的 model.train() 呼叫,負責寫入超參數、讀取 args.yaml / results.csv。
"""

import shutil
from pathlib import Path
from ultralytics import YOLO


def train_run(hyperparameters: dict, run_name: str, settings: dict, seed: int = 0) -> dict:
    """
    跑一輪 YOLOv8n 訓練。

    hyperparameters: 例如 {"lr0": 0.001, "weight_decay": 0.0005, "mosaic": 0.5, "hsv_v": 0.6}
    settings: 從 config/settings.yaml 讀進來的 dict
    seed: 隨機種子,3-seed baseline / seed 穩健性驗證用,預設 0

    回傳: {"results_csv": <path>, "args_yaml": <path>, "best_map5095": float}
    """
    run_dir = Path("runs") / "detect" / run_name

    # 2026-09 修正:Ultralytics model.train() 預設 exist_ok=False,如果
    # run_dir 已經存在(例如上次用舊超參數跑過的殘留),不會覆蓋,而是自動
    # 改名另存新資料夾(例如 baseline_3seed_seed0 -> baseline_3seed_seed0-2)。
    # 但下面讀結果的邏輯是照原本要求的 run_name 去找 results.csv,對不上
    # Ultralytics 實際存的地方,輕則直接 FileNotFoundError(這次踩到的狀況),
    # 重則悄悄讀到舊資料夾裡「之前訓練」留下的 results.csv,當作這次的結果,
    # 不會有任何錯誤訊息提示你用錯資料。過去一直是靠「記得手動刪舊資料夾」
    # 來避免這個問題,這次真的忘記刪就爆出來了,改成訓練前自動清空同名
    # 資料夾,徹底避免要求人為記憶這件事,見 docs/decisions.md。
    if run_dir.exists():
        print(f"[train_run] 發現舊的 {run_dir} 資料夾,先刪除再開始訓練,"
              f"避免 Ultralytics exist_ok=False 自動改名導致讀到錯的資料夾。")
        shutil.rmtree(run_dir)

    model = YOLO(settings["training"]["model"])

    # 明確指定 optimizer 跟 pretrained,不要用預設值(見 README 已知陷阱)
    # workers 保守設定為 4:實驗室機器系統 RAM 只有 16GB,Windows 下每個
    # worker 是獨立 process,設太高(Ultralytics 預設有時會抓到 8)容易在
    # 訓練跑起來時把記憶體壓滿,見 docs/decisions.md。
    model.train(
        data=str(Path(settings["dataset"]["root"]) / "data.yaml"),
        epochs=settings["training"]["epochs"],
        batch=settings["training"]["batch_size"],
        imgsz=settings["training"]["imgsz"],
        optimizer=settings["training"]["optimizer"],
        pretrained=settings["training"]["pretrained"],
        workers=settings["training"].get("workers", 4),
        seed=seed,
        name=run_name,
        **hyperparameters,
    )

    results_csv = run_dir / "results.csv"
    args_yaml = run_dir / "args.yaml"
    best_weights = run_dir / "weights" / "best.pt"

    best_map5095 = _read_best_map(results_csv)
    class_stats = compute_class_stats(str(best_weights), settings)

    return {
        "results_csv": str(results_csv),
        "args_yaml": str(args_yaml),
        "best_map5095": best_map5095,
        "class_stats": class_stats,
    }


def compute_class_stats(weights_path: str, settings: dict) -> dict:
    """
    對訓練完的最佳權重(best.pt)跑一次驗證,取得每個類別的
    AP50 / AP50-95 / Precision / Recall / n(instance數),給
    summarize_log.py 的 D5(Class-Level Imbalance)診斷跟 Facts 層用。

    回傳: {class_name: {"ap50": float, "ap5095": float, "precision": float,
                         "recall": float, "n": int}, ...}

    注意:Ultralytics DetMetrics 物件的屬性名稱在不同版本可能略有差異,
    這裡是依 8.4.152(2026-09 實測版本)寫的、也已經實測跑過確認正確:
    ap50/ap/p/r/ap_class_index 在 metrics.box 底下,但 nt_per_class(每個
    類別的 instance 數量)是在 metrics 底下,不是 metrics.box 底下,一開始
    寫錯位置導致 n 全部是 None,已修正。如果之後 ultralytics 版本更新又
    改了屬性名稱或位置,一樣可以用 dir(metrics) / dir(metrics.box) 查。
    """
    model = YOLO(weights_path)
    metrics = model.val(
        data=str(Path(settings["dataset"]["root"]) / "data.yaml"),
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
    return class_stats


def _read_best_map(results_csv: Path) -> float:
    """用內建 csv 模組讀,不依賴 pandas。2026-09 這台機器的 pandas 被
    Application Control Policy 擋住(DLL 載入失敗),Random Search/Optuna
    這三個 baseline 方法不需要 pandas 的複雜功能,改用內建模組繞過去,
    見 docs/decisions.md。LLM 診斷組(summarize_log.py)還是需要完整
    pandas,不受這個修法影響,那邊如果被擋還是要等 IT 處理。"""
    import csv

    with open(results_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"{results_csv} 沒有任何資料列")

    map_col = None
    for col in rows[0].keys():
        if "map50-95" in col.lower():
            map_col = col
            break
    if map_col is None:
        raise KeyError(f"找不到包含 'map50-95' 的欄位,實際欄位:{list(rows[0].keys())}")

    values = [float(row[map_col]) for row in rows if row[map_col].strip() != ""]
    return max(values)