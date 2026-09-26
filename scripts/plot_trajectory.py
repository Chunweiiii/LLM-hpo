"""
畫「Best-so-far mAP50-95 隨 Round 變化」的軌跡折線圖,4個模型 x 2種條件
(Diagnosis 實線 / Score-only 虛線),同一模型同色、不同 marker(marker
有黑色描邊),灰色系虛線畫 Novice baseline。

用法(在專案根目錄執行):
    python scripts/plot_trajectory.py
    python scripts/plot_trajectory.py --out-chart my_chart.png --out-legend my_legend.png

資料來源:自動掃描 results/full_diagnosis_*.json、results/score_only_*.json
的 history(每輪實際分數),換算成 best-so-far(累計最大值)畫圖;
results/baseline_3seed.json 的 map5095_mean 當 Novice baseline。

模型範圍跟 plot_ranking_table.py 一樣的 4 個模型 —— 全部統一用 hinted
當診斷組(gemma4-12b / llama3.1-8b / phi4-14b 原本就用 hinted;Claude
Haiku 4.5 於 2026-09-26 補跑 --hinted 後也改用 hinted 版本,四個模型
的診斷組 prompt 完全一致,不再有 hinted/non-hinted 混用的問題)。
Condition 也跟表格一樣簡化成 "Diagnosis" / "Score-only" 兩種字眼。

2026-09-26 更新:Claude Haiku 4.5 的診斷組改成 hinted 版本(best_round=7,
best_map5095=0.66337),取代原本的非 hinted 版本(0.667)。

2026-09-26 再更新:Claude Haiku 4.5 的檔名也跟其他三個小模型統一,拿掉
"_hinted" 後綴(full_diagnosis_claude-haiku-4.5_hinted.json ->
full_diagnosis_claude-haiku-4.5.json),因為 hinted 現在就是四個模型
共同、唯一的正式診斷組,不用再靠檔名區分。舊的非 hinted 版本改名成
full_diagnosis_claude-haiku-4.5_nonhinted_reference.json 留在本機當參考,
不進 git 版控(見 .gitignore)。

顏色用 Wayne 提供的參考圖(Material Design 色票)量出來的:
    粉   #E91E63  Llama 3.1 8B(呼應參考圖裡表現最好的方法用粉紅+菱形)
    橘   #F57C00  Phi-4 14B
    綠   #2E7D32  Gemma 4 12B
    藍   #2196F3  Claude Haiku 4.5
    藍灰 #607D8B  Novice baseline

2026-09-23 更新:曲線圖跟圖例拆成兩張獨立的圖片(--out-chart /
--out-legend),不再合成同一張。曲線圖只留座標軸/格線/線條,不含圖例,
x 軸範圍貼著資料(不用再留空白區);圖例圖是單獨一張裁到剛好包住圖例
本身大小的圖片,兩張圖的顏色、marker、線型定義完全共用同一份資料,
確保拼在一起時對得上。

2026-09-24 更新:results/ 底下三個小模型的 hinted 結果檔已經改名(拿掉
"_hinted" 後綴),因為 hinted 現在就是正式的診斷組。MODELS 已同步改成
新檔名。
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

# (Method 顯示名稱, 診斷組檔名, score-only 檔名, 顏色, marker)
MODELS = [
    ("Llama 3.1 8B", "full_diagnosis_llama3.1-8b.json", "score_only_llama3.1-8b.json", "#E91E63", "D"),
    ("Phi-4 14B", "full_diagnosis_phi4-14b.json", "score_only_phi4-14b.json", "#F57C00", "o"),
    ("Gemma 4 12B", "full_diagnosis_gemma4-12b.json", "score_only_gemma4-12b.json", "#2E7D32", "^"),
    ("Claude Haiku 4.5", "full_diagnosis_claude-haiku-4.5.json", "score_only_claude-haiku-4.5.json", "#2196F3", "*"),
]

GRAY = "#607D8B"
MARKER_EDGE = "black"
MARKER_EDGE_WIDTH = 1.1


def best_so_far(history):
    out = []
    running = None
    for h in history:
        v = h.get("best_map5095", h.get("map5095"))
        running = v if running is None else max(running, v)
        out.append(running)
    return out


def load_series(filename):
    data = json.loads((RESULTS_DIR / filename).read_text(encoding="utf-8"))
    return best_so_far(data["history"])


def load_baseline():
    data = json.loads((RESULTS_DIR / "baseline_3seed.json").read_text(encoding="utf-8"))
    return data["map5095_mean"]


def build_chart(ax):
    """在給定的 ax 上畫所有曲線,回傳 (rounds, legend_handles, legend_names, legend_values)。"""
    baseline = load_baseline()

    legend_handles = []
    legend_names = []
    legend_values = []

    rounds = None
    base_line = ax.axhline(baseline, color=GRAY, linestyle=(0, (5, 3)), linewidth=1.6, zorder=1)
    legend_handles.append(base_line)
    legend_names.append("Baseline")
    legend_values.append(f"{baseline:.4f}")

    for display_name, diag_file, score_file, color, marker in MODELS:
        for filename, cond_label, linestyle in [
            (diag_file, "Diagnosis", "-"),
            (score_file, "Score-only", "--"),
        ]:
            series = load_series(filename)
            if rounds is None:
                rounds = list(range(len(series)))
            (line,) = ax.plot(
                rounds, series,
                color=color, linestyle=linestyle, linewidth=2.2,
                marker=marker, markersize=8 if marker != "*" else 12,
                markeredgecolor=MARKER_EDGE, markeredgewidth=MARKER_EDGE_WIDTH,
                markerfacecolor=color,
                zorder=3,
            )
            legend_handles.append(line)
            legend_names.append(f"{display_name} - {cond_label}")
            legend_values.append(f"{series[-1]:.4f}")

    return rounds, legend_handles, legend_names, legend_values


def style_axes(ax, rounds):
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Best-so-far mAP@0.5:0.95", fontsize=12)
    ax.set_xticks(rounds)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.01))
    ax.set_xlim(-0.3, max(rounds) + 0.3)

    ax.grid(True, color="#d9d9d9", linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)


def build_legend_labels(legend_names, legend_values):
    # 名稱欄寬度依實際最長的名稱動態決定(含表頭 "Method"),等寬字體
    # 對齊成左邊名稱、右邊分數兩欄。
    name_width = max(len("Method"), *(len(n) for n in legend_names)) + 2
    labels = [f"{n:<{name_width}}{v}" for n, v in zip(legend_names, legend_values)]
    header_label = f"{'Method':<{name_width}}Best"
    return header_label, labels


def render(out_chart, out_legend):
    # ---- 曲線圖(不含圖例,x 軸貼著資料範圍)----
    fig, ax = plt.subplots(figsize=(9.6, 6.4), dpi=200)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    rounds, legend_handles, legend_names, legend_values = build_chart(ax)
    style_axes(ax, rounds)

    fig.subplots_adjust(left=0.085, right=0.97, top=0.97, bottom=0.1)
    fig.savefig(out_chart, facecolor="white")
    plt.close(fig)
    print(f"saved {out_chart}")

    # ---- 圖例(單獨一張圖,裁到剛好包住圖例本身)----
    header_label, labels = build_legend_labels(legend_names, legend_values)
    header_handle = Line2D([], [], linestyle="None")
    handles = [header_handle] + legend_handles
    all_labels = [header_label] + labels

    fig2 = plt.figure(figsize=(6.0, 4.0), dpi=200)
    fig2.patch.set_facecolor("white")
    leg = fig2.legend(
        handles, all_labels,
        loc="center", frameon=True, framealpha=1, edgecolor="black",
        prop={"family": "DejaVu Sans Mono", "size": 10.5},
        handlelength=2.4, labelspacing=0.55, borderpad=0.9,
    )
    leg.get_texts()[0].set_fontweight("bold")
    leg.legend_handles[0].set_visible(False)

    fig2.savefig(out_legend, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig2)
    print(f"saved {out_legend}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-chart", type=str, default="trajectory_chart.png")
    parser.add_argument("--out-legend", type=str, default="trajectory_legend.png")
    args = parser.parse_args()
    render(args.out_chart, args.out_legend)


if __name__ == "__main__":
    main()
