"""
兩層診斷架構核心模組:Facts + 5 個 Diagnostic Flags。

下面幾個閾值常數是用 novice baseline(5-seed,seed=0,1,2,3,4,
`experiments/run_baseline_3seed.py` 產生 `results/baseline_3seed.json`,
`scripts/derive_d4_d5_thresholds.py` 產生 `results/d4_d5_thresholds.json`)
算出來的最新版本,取代 2026-09 初曾經用過的 SEARCH_SPACE 中點版
(BASELINE_MEAN=0.6606/BASELINE_STD=0.0060/D4=0.3155/D5=0.1991,因指導教授
指出天花板效應疑慮而棄用),也取代同年 9/17 算出、後來發現有偏差風險的
3-seed 版(BASELINE_MEAN=0.6214/BASELINE_STD=0.0095/D4=0.3261/D5=0.2353,
3-seed 時 std 估計不穩定,單一極端值 seed1 影響力過大,見 docs/decisions.md
「5-seed 擴充」)。跟 `prompts/program.md`、`prompts/score_only.md` 系統提示
裡寫給 LLM 看的數字保持一致,這裡用完整精度而非四捨五入後的顯示值,避免
邊界值判斷產生誤差。

2026-09 修正紀錄:發現這個檔案在 novice baseline 改版後,常數一直沒有跟著
更新,導致 `results/full_diagnosis_qwen3-14b.json`(2026-09-17 晚間跑的
10 輪正式診斷)當時實際使用的是舊的 SEARCH_SPACE 中點版閾值判斷 D1/D4/D5,
跟 LLM 系統提示裡看到的閾值文字不一致。已改正這裡的常數(先改成 3-seed
版,再改成這裡的 5-seed 最終版),但那次 Qwen 結果本身尚未重跑,見
docs/decisions.md 的處理紀錄與待辦。
"""

import numpy as np
import pandas as pd


# ---- 已確定(novice baseline,5-seed,詳見 results/baseline_3seed.json、
# docs/decisions.md「新手起始點」「5-seed 擴充」)----
NOISE_THRESHOLD = 0.02192252905118387   # = 2 × BASELINE_STD
BASELINE_MEAN = 0.6198480000000001
BASELINE_STD = 0.010961264525591935

D1_PLATEAU_MAX_FRACTION = 0.7    # 已定案:起始點需 < 總epoch × 此比例才算 D1,不受超參數組合影響,不用重算

# ---- 已確定(見 results/d4_d5_thresholds.json,scripts/derive_d4_d5_thresholds.py 算出,5-seed 版)----
D4_LOCALIZATION_GAP_THRESHOLD = 0.326124
D5_CLASS_GAP_THRESHOLD = 0.226932

def read_log(results_csv_path: str) -> pd.DataFrame:
    """讀取 Ultralytics 的 results.csv"""
    return pd.read_csv(results_csv_path)


def _find_col(df: pd.DataFrame, keyword: str, exclude: str = None) -> str:
    """
    在 df 欄位裡找包含 keyword(不分大小寫)的欄位名稱。
    exclude:排除同時包含這個字串的欄位(例如找 "map50" 時要排除 "map50-95",
    因為 "map50-95" 這個字串本身也包含 "map50" 子字串)。
    """
    keyword_lower = keyword.lower()
    matches = [c for c in df.columns if keyword_lower in c.lower()]
    if exclude:
        exclude_lower = exclude.lower()
        matches = [c for c in matches if exclude_lower not in c.lower()]
    if not matches:
        raise KeyError(f"找不到包含 '{keyword}' 的欄位,實際欄位:{list(df.columns)}")
    return matches[0]


def find_key_points(series: pd.Series) -> dict:
    """
    找關鍵抽樣點:起始點、最佳點、最終點、斜率轉折點。
    轉折點演算法已定案:簡單斜率變化(不用 change point detection / PELT)。

    轉折點定義:先用移動平均(window=3)平滑序列,計算逐輪斜率(一階差分),
    以前 25% 訓練輪次的平均斜率當作「早期成長速度」參考基準,找第一個
    「平滑後斜率掉到早期斜率 10% 以下,且之後 70% 以上的時間都維持在
    早期斜率 30% 以下」的 epoch,視為成長明顯趨緩的轉折點(而不是單一雜訊
    波動)。找不到符合條件的點時回傳 None(例如訓練全程都還在穩定成長)。
    """
    values = series.reset_index(drop=True).astype(float)
    n = len(values)

    start_epoch = 0
    best_epoch = int(values.idxmax())
    final_epoch = n - 1

    if n < 4:
        # 輪次太少,轉折點判定沒有意義
        return {
            "start_epoch": start_epoch,
            "best_epoch": best_epoch,
            "final_epoch": final_epoch,
            "inflection_epoch": None,
        }

    smoothed = values.rolling(window=min(3, n), min_periods=1, center=True).mean()
    slope = smoothed.diff()

    early_window = max(1, n // 4)
    early_slope = slope.iloc[1:early_window + 1].mean()

    inflection_epoch = None
    if early_slope is not None and early_slope > 0:
        drop_threshold = early_slope * 0.1
        sustain_threshold = early_slope * 0.3
        for i in range(early_window, n):
            if slope.iloc[i] < drop_threshold:
                remaining = slope.iloc[i:]
                if (remaining < sustain_threshold).mean() > 0.7:
                    inflection_epoch = i
                    break

    return {
        "start_epoch": start_epoch,
        "best_epoch": best_epoch,
        "final_epoch": final_epoch,
        "inflection_epoch": inflection_epoch,
    }


def _find_plateau_start(series: pd.Series, noise_threshold: float) -> int:
    """
    找到平滑後逐輪變化量持續低於 noise_threshold 的起始 epoch(平台期起點)。
    D1 判定專用,跟 find_key_points 的「轉折點」是不同的定義
    (D1 用絕對雜訊門檻,find_key_points 用相對早期斜率的比例)。
    找不到就回傳 None(例如訓練全程都還在明顯成長,沒有進入平台期)。
    """
    values = series.reset_index(drop=True).astype(float)
    n = len(values)
    if n < 2:
        return None

    smoothed = values.rolling(window=min(3, n), min_periods=1, center=True).mean()
    change = smoothed.diff().abs()

    for i in range(1, n):
        remaining = change.iloc[i:]
        if (remaining < noise_threshold).all():
            return i
    return None


def _is_train_loss_still_decreasing(train_loss_series: pd.Series, plateau_start: int,
                                      min_relative_drop: float = 0.05) -> bool:
    """平台期開始之後,train loss 是否仍持續顯著下降(相對下降超過 min_relative_drop)。"""
    if plateau_start is None:
        return False
    n = len(train_loss_series)
    if plateau_start >= n - 1:
        return False
    segment = train_loss_series.iloc[plateau_start:]
    start_val = segment.iloc[0]
    end_val = segment.iloc[-1]
    if start_val == 0:
        return False
    relative_drop = (start_val - end_val) / start_val
    return bool(relative_drop > min_relative_drop)


def _is_val_loss_rising(val_loss_series: pd.Series, tail_fraction: float = 0.3) -> bool:
    """只看 val loss 後段(預設最後30%輪次)的線性趨勢斜率是否為正(轉為上升)。"""
    n = len(val_loss_series)
    tail_n = max(2, int(n * tail_fraction))
    tail = val_loss_series.iloc[-tail_n:].reset_index(drop=True)
    x = np.arange(len(tail))
    slope = np.polyfit(x, tail.values.astype(float), 1)[0]
    return bool(slope > 0)


def compute_facts(df: pd.DataFrame, class_stats: dict) -> dict:
    """
    組出 Facts 層:總epoch數、最佳/最終mAP、關鍵點、train/val loss首尾值、
    per-class AP/P/R(含n)、有效學習率首尾值(線性衰減,非cosine)。
    """
    total_epochs = len(df)

    map5095_col = _find_col(df, "map50-95")
    map50_col = _find_col(df, "map50", exclude="map50-95")
    train_box_col = _find_col(df, "train/box_loss")
    train_cls_col = _find_col(df, "train/cls_loss")
    train_dfl_col = _find_col(df, "train/dfl_loss")
    val_box_col = _find_col(df, "val/box_loss")
    val_cls_col = _find_col(df, "val/cls_loss")
    val_dfl_col = _find_col(df, "val/dfl_loss")
    lr_col = _find_col(df, "lr/pg0")

    map5095_series = df[map5095_col].astype(float)
    map50_series = df[map50_col].astype(float)
    train_loss_series = (df[train_box_col] + df[train_cls_col] + df[train_dfl_col]).astype(float)
    val_loss_series = (df[val_box_col] + df[val_cls_col] + df[val_dfl_col]).astype(float)

    key_points = find_key_points(map5095_series)

    facts = {
        "total_epochs": total_epochs,
        "best_map5095": float(map5095_series.max()),
        "best_epoch": key_points["best_epoch"],
        "final_map5095": float(map5095_series.iloc[-1]),
        "final_map50": float(map50_series.iloc[-1]),
        "key_points": key_points,
        "map5095_series": map5095_series.tolist(),
        "train_loss_start": float(train_loss_series.iloc[0]),
        "train_loss_end": float(train_loss_series.iloc[-1]),
        "train_loss_series": train_loss_series.tolist(),
        "val_loss_start": float(val_loss_series.iloc[0]),
        "val_loss_end": float(val_loss_series.iloc[-1]),
        "val_loss_series": val_loss_series.tolist(),
        "lr_start": float(df[lr_col].iloc[0]),
        "lr_end": float(df[lr_col].iloc[-1]),
        "class_stats": class_stats,
    }
    return facts


def diagnose(facts: dict, class_stats: dict) -> dict:
    """
    產出 5 個 Diagnostic Flags(D1-D5),各自獨立判定,回傳 dict:
    {"D1": {"triggered": bool, "detail": {...}}, ...}
    """
    flags = {}
    total_epochs = facts["total_epochs"]
    map5095_series = pd.Series(facts["map5095_series"])
    train_loss_series = pd.Series(facts["train_loss_series"])
    val_loss_series = pd.Series(facts["val_loss_series"])

    # [D1] Convergence Saturation
    # 判定:平滑後mAP在epoch X之後變化量持續低於 NOISE_THRESHOLD,
    #      且 X < 總epoch數 × D1_PLATEAU_MAX_FRACTION
    plateau_start = _find_plateau_start(map5095_series, NOISE_THRESHOLD)
    d1_triggered = (
        plateau_start is not None
        and plateau_start < total_epochs * D1_PLATEAU_MAX_FRACTION
    )
    flags["D1"] = {
        "triggered": d1_triggered,
        "detail": {
            "plateau_start_epoch": plateau_start,
            "threshold_epoch": total_epochs * D1_PLATEAU_MAX_FRACTION,
            "noise_threshold": NOISE_THRESHOLD,
        },
    }

    # [D2] Loss-Metric Decoupling
    # 判定:train loss持續顯著下降 AND mAP已進入平台期
    d2_triggered = d1_triggered and _is_train_loss_still_decreasing(
        train_loss_series, plateau_start
    )
    flags["D2"] = {
        "triggered": d2_triggered,
        "detail": {
            "mAP_plateaued": d1_triggered,
            "plateau_start_epoch": plateau_start,
        },
    }

    # [D3] Genuine Overfitting
    # 判定:只看 val loss 後段趨勢是否轉為上升(不看 train-val gap 大小)
    d3_triggered = _is_val_loss_rising(val_loss_series)
    flags["D3"] = {
        "triggered": d3_triggered,
        "detail": {"tail_fraction": 0.3},
    }

    # [D4] Localization Bottleneck
    # 判定:mAP@0.5 與 mAP@0.5:0.95 落差 > D4_LOCALIZATION_GAP_THRESHOLD
    d4_gap = facts["final_map50"] - facts["final_map5095"]
    d4_triggered = d4_gap > D4_LOCALIZATION_GAP_THRESHOLD
    flags["D4"] = {
        "triggered": d4_triggered,
        "detail": {
            "gap": d4_gap,
            "threshold": D4_LOCALIZATION_GAP_THRESHOLD,
            "final_map50": facts["final_map50"],
            "final_map5095": facts["final_map5095"],
        },
    }

    # [D5] Class-Level Imbalance Effect
    # 判定:最弱類別AP低於最強類別 > D5_CLASS_GAP_THRESHOLD
    ap5095_by_class = {
        name: stats["ap5095"] for name, stats in class_stats.items()
    }
    if ap5095_by_class:
        weakest_class = min(ap5095_by_class, key=ap5095_by_class.get)
        strongest_class = max(ap5095_by_class, key=ap5095_by_class.get)
        d5_gap = ap5095_by_class[strongest_class] - ap5095_by_class[weakest_class]
        d5_triggered = d5_gap > D5_CLASS_GAP_THRESHOLD
    else:
        weakest_class = strongest_class = None
        d5_gap = None
        d5_triggered = False
    flags["D5"] = {
        "triggered": d5_triggered,
        "detail": {
            "gap": d5_gap,
            "threshold": D5_CLASS_GAP_THRESHOLD,
            "weakest_class": weakest_class,
            "strongest_class": strongest_class,
            "ap5095_by_class": ap5095_by_class,
        },
    }

    return flags


def to_report_text(facts: dict, flags: dict) -> str:
    """把 Facts + Diagnostic Flags 組成結構化文字報告(見 prompts/program.md 範例格式)"""
    kp = facts["key_points"]

    class_lines = []
    for name, stats in facts["class_stats"].items():
        class_lines.append(
            f"  - {name}: AP50-95={stats['ap5095']:.4f} AP50={stats['ap50']:.4f} "
            f"P={stats['precision']:.4f} R={stats['recall']:.4f} n={stats['n']}"
        )
    class_block = "\n".join(class_lines) if class_lines else "  (無 per-class 資料)"

    inflection_str = (
        f"epoch {kp['inflection_epoch']}" if kp["inflection_epoch"] is not None
        else "無明顯轉折點(全程持續成長或全程平坦)"
    )

    lines = [
        "=== TRAINING RUN FACTS ===",
        f"Total epochs: {facts['total_epochs']}",
        f"Best mAP50-95: {facts['best_map5095']:.4f} (epoch {facts['best_epoch']})",
        f"Final mAP50-95: {facts['final_map5095']:.4f}",
        f"Final mAP50: {facts['final_map50']:.4f}",
        f"Key trajectory points: start=epoch {kp['start_epoch']}, "
        f"best=epoch {kp['best_epoch']}, final=epoch {kp['final_epoch']}, "
        f"inflection={inflection_str}",
        f"Train loss: start={facts['train_loss_start']:.4f} -> end={facts['train_loss_end']:.4f}",
        f"Val loss: start={facts['val_loss_start']:.4f} -> end={facts['val_loss_end']:.4f}",
        f"Effective learning rate (linear decay, cos_lr=False): "
        f"start={facts['lr_start']:.6f} -> end={facts['lr_end']:.6f}",
        "Per-class AP/P/R (n):",
        class_block,
        "",
        "=== DIAGNOSTIC FLAGS ===",
    ]

    flag_labels = {
        "D1": "Convergence Saturation",
        "D2": "Loss-Metric Decoupling",
        "D3": "Genuine Overfitting",
        "D4": "Localization Bottleneck",
        "D5": "Class-Level Imbalance Effect",
    }
    for key, label in flag_labels.items():
        flag = flags[key]
        status = "TRIGGERED" if flag["triggered"] else "NOT TRIGGERED"
        lines.append(f"[{key}] {label}: {status}")
        lines.append(f"     detail: {flag['detail']}")

    return "\n".join(lines)
