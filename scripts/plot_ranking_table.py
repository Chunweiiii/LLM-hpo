"""
畫「全專案結果排行榜」表格圖(Rank | Method | Condition | Best mAP50-95 |
Baseline (5-seed)),樣式照 Wayne 原本手動做的表格:無標題列文字、頂/底
粗黑線、表頭下細線、其餘無格線、襯線字體。畫布 16:9。

2026-09-23 再更新:
- 拿掉 Round 欄位。
- Condition 欄位統一簡化成只有 "Diagnosis" / "Score-only" 兩種(不再分
  Hinted/Full-diagnosis 這種 model-specific 的字眼,因為 hinted 現在
  就是三個小模型正式的診斷組)。
- 畫布改成 16:9。
- 除了表頭(Rank/Method/... 那一列)以外都不粗體了,包含原本第一名
  整列粗體、分數粗體、紅色 "(↑X.XX%)" 粗體都拿掉,只留紅色顏色本身。

用法(在專案根目錄執行):
    python scripts/plot_ranking_table.py                  # 預設全部列出來
    python scripts/plot_ranking_table.py --top 4           # 只列前 4 名
    python scripts/plot_ranking_table.py --out my_table.png

資料來源:自動掃描 results/full_diagnosis_*.json、results/score_only_*.json,
不用手動貼數字 —— 之後有新輪次結果,重跑這支script就會自動抓最新的。

2026-09-23 更新:hinted 已經正式取代 full-diagnosis 作為三個小模型
(gemma4-12b / llama3.1-8b / phi4-14b)的診斷組結果(見 docs/decisions.md
「hint 作為我們的診斷組最終結果」的決定),Claude Opus 5 / Claude Sonnet 5 /
Qwen3-14B / Random Search baseline 都不是這四個模型比較的對象,所以
ENTRIES 只留這四個模型(3 個小模型 + Claude Haiku 4.5,各自的
score-only + diagnosis 兩種條件,共 8 筆),不是全專案排行榜。

2026-09-24 更新:results/ 底下三個小模型的 hinted 結果檔已經改名(拿掉
"_hinted" 後綴,例如 full_diagnosis_gemma4-12b_hinted.json ->
full_diagnosis_gemma4-12b.json),因為 hinted 現在就是正式的診斷組,不用
再用檔名特別標記。ENTRIES 已同步改成新檔名。舊檔名(帶 _hinted)跟舊的
未加 hint 版本都已經不在 git 追蹤範圍內,見 .gitignore。

Baseline (5-seed) 欄位顯示 "0.6198"(四捨五入到小數點後4位),
(↑X.XX%) 的百分比也是用這個四捨五入後的 0.6198 當分母算的 —— 這是對照
Wayne 原本手動做的表格圖反推驗證過的寫法,不是用 baseline_3seed.json
裡完整精度的 0.61985(兩種算法到小數點後兩位就會對不上)。
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

BASELINE_DISPLAY = "0.6198"
BASELINE_VALUE = 0.6198

# (檔名, Method 顯示名稱, Condition 顯示名稱, 檔案格式)
# 檔案格式 "single": dict,直接有 best_map5095 / best_round
# 檔案格式 "list":   list of dict(每個 dict 是一個 trial),要自己取 max
#
# 只留 4 個模型,全部統一用 hinted 當診斷組(2026-09-26:Claude Haiku 4.5
# 補跑 --hinted 後,四個模型的診斷組 prompt 完全一致)。Claude Opus 5 /
# Claude Sonnet 5 / Qwen3-14B / Random Search baseline 都不列入,見上面
# module docstring 的說明。Condition 統一只用 "Diagnosis" / "Score-only"
# 兩種字眼。
ENTRIES = [
    ("full_diagnosis_claude-haiku-4.5_hinted.json", "Claude Haiku 4.5", "Diagnosis", "single"),
    ("full_diagnosis_gemma4-12b.json", "Gemma 4 12B", "Diagnosis", "single"),
    ("full_diagnosis_llama3.1-8b.json", "Llama 3.1 8B", "Diagnosis", "single"),
    ("full_diagnosis_phi4-14b.json", "Phi-4 14B", "Diagnosis", "single"),
    ("score_only_claude-haiku-4.5.json", "Claude Haiku 4.5", "Score-only", "single"),
    ("score_only_gemma4-12b.json", "Gemma 4 12B", "Score-only", "single"),
    ("score_only_llama3.1-8b.json", "Llama 3.1 8B", "Score-only", "single"),
    ("score_only_phi4-14b.json", "Phi-4 14B", "Score-only", "single"),
]

FONT = "Liberation Serif"  # 度量相容 Times New Roman,環境沒裝正牌 Times
RED = "#c00000"
BLACK = "#000000"

COL_X = {
    "rank": 0.018,
    "method": 0.140,
    "condition": 0.430,
    "map": 0.650,
    "baseline": 0.870,
}
HEADERS = {
    "rank": "Rank",
    "method": "Method",
    "condition": "Condition",
    "map": "Best mAP50-95",
    "baseline": "Baseline (5-seed)",
}


def collect_rows():
    rows = []
    for filename, method, condition, kind in ENTRIES:
        path = RESULTS_DIR / filename
        if not path.exists():
            print(f"[skip] 找不到 {path}")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))

        if kind == "single":
            best_map = data["best_map5095"]
        else:  # list of trial dicts (baseline_random_search.json 這種格式)
            best_trial = max(data, key=lambda r: r.get("best_map5095", -1))
            best_map = best_trial["best_map5095"]

        baseline_disp = "–" if kind == "list" else BASELINE_DISPLAY

        rows.append({
            "method": method,
            "condition": condition,
            "score": best_map,
            "baseline_disp": baseline_disp,
        })

    rows.sort(key=lambda r: -r["score"])
    return rows


def render(rows, out_path):
    n = len(rows)
    # 16:9 畫布(投影片比例)。用 bbox_inches=None(存整張畫布,不裁切)
    # 才能維持準確的 16:9,裁切(bbox_inches="tight")會依實際文字範圍
    # 改變長寬比,見下面 savefig。
    fig_w, fig_h = 13.33, 7.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
    plt.rcParams["font.family"] = FONT
    # matplotlib 預設會在 axes 四周留一圈 margin(即使 axis("off")、就算
    # savefig 不裁切,那圈 margin 還是算進圖片裡),讓存出來的圖邊框
    # 留白很多。把 axes 撐滿整張畫布,留白完全由下面 top/bottom/COL_X
    # 這些內部座標自己控制。
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    top, bottom = 0.97, 0.03
    header_y = top - 0.075
    rule_under_header_y = header_y - 0.045
    row_top = rule_under_header_y - 0.075
    row_bottom = bottom + 0.03
    row_ys = (
        [row_top - i * (row_top - row_bottom) / (n - 1) for i in range(n)]
        if n > 1 else [(row_top + row_bottom) / 2]
    )

    fs = 15.5

    ax.plot([0, 1], [top, top], color=BLACK, lw=2.8, solid_capstyle="butt", clip_on=False)
    ax.plot([0, 1], [bottom, bottom], color=BLACK, lw=2.8, solid_capstyle="butt", clip_on=False)

    for key, label in HEADERS.items():
        ax.text(COL_X[key], header_y, label, fontsize=fs, fontweight="bold",
                 color=BLACK, ha="left", va="center", fontfamily=FONT)

    ax.plot([0, 1], [rule_under_header_y, rule_under_header_y], color=BLACK, lw=1.1,
             solid_capstyle="butt", clip_on=False)

    for i, row in enumerate(rows):
        y = row_ys[i]
        rank = i + 1

        ax.text(COL_X["rank"], y, str(rank), fontsize=fs, fontweight="normal",
                 color=BLACK, ha="left", va="center", fontfamily=FONT)
        ax.text(COL_X["method"], y, row["method"], fontsize=fs, fontweight="normal",
                 color=BLACK, ha="left", va="center", fontfamily=FONT)
        ax.text(COL_X["condition"], y, row["condition"], fontsize=fs, fontweight="normal",
                 color=BLACK, ha="left", va="center", fontfamily=FONT)
        ax.text(COL_X["baseline"], y, row["baseline_disp"], fontsize=fs, fontweight="normal",
                 color=BLACK, ha="left", va="center", fontfamily=FONT)

        score = row["score"]
        pct = (score - BASELINE_VALUE) / BASELINE_VALUE * 100
        score_text = f"{score:.5f}"
        pct_text = f"(↑ {pct:.2f}%)"

        x = COL_X["map"]
        t1 = ax.text(x, y, score_text, fontsize=fs, fontweight="normal",
                      color=BLACK, ha="left", va="center", fontfamily=FONT)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox = t1.get_window_extent(renderer=renderer)
        inv = ax.transData.inverted()
        (x1_data, _), = inv.transform([[bbox.x1, bbox.y0]])
        ax.text(x1_data, y, pct_text, fontsize=fs, fontweight="normal",
                 color=RED, ha="left", va="center", fontfamily=FONT)

    fig.savefig(out_path, facecolor="white")  # 不用 bbox_inches="tight",維持準確 16:9
    print(f"saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=8, help="列出前幾名,預設 8(4個模型 x 2種條件,全部列出)")
    parser.add_argument("--out", type=str, default="ranking_table.png", help="輸出圖檔路徑")
    args = parser.parse_args()

    rows = collect_rows()
    rows = rows[: args.top]
    render(rows, args.out)


if __name__ == "__main__":
    main()
