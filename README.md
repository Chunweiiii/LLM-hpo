# LLM-HPO Traffic Light Detection

LLM 讀取 YOLOv8n 訓練動態,診斷後提出下一輪超參數建議,迭代 10 輪選出最佳模型。

## 專案狀態(2026-09-26)

資料收集階段已完成,目前進入論文寫作階段。四個模型的完整診斷組
(diagnosis,統一用 hinted prompt,見下方「重要:hinted 統一說明」)+
score-only對照組都已跑完 10 輪,傳統方法基準線(novice baseline / random
search / Optuna TPE)也都跑完,最終結果與排名已定案(見下方「目前結果」)。

論文用的圖表在 `figure/` 資料夾(有進版控),包含:
- `chart.png`:4 模型 best-so-far mAP50-95 收斂曲線圖(含圖例)
- `ranking_table_3.png`:8 列排行榜表格(4 模型 × Diagnosis/Score-only)
- `twelve_panel_comparison_figure.png`:紅/黃/綠燈 3 場景 × baseline+3模型
  最佳輪次的偵測結果對比圖
- `diagnosis.png`:system prompt 結構圖(Task Description / Diagnostic
  Information Module / Guidance Module 三大模組)
- `fig1.png`:整體流程圖(Datasets -> Training -> Diagnosis -> LLM Guide)
- `preview_small.jpg`:資料集樣本圖(4 張真實照片,(a)-(d) 各代表一種
  情境,無 GT 框)

## 目前結果(Diagnosis 條件,4 模型排名)

| Rank | Method | Best mAP50-95 | vs. Baseline (0.6198) |
|---|---|---|---|
| 1 | Llama 3.1 8B | 0.66959 | ↑ 8.03% |
| 2 | Phi-4 14B | 0.66493 | ↑ 7.28% |
| 3 | Claude Haiku 4.5 | 0.66337 | ↑ 7.03% |
| 4 | Gemma 4 12B | 0.65932 | ↑ 6.38% |

完整 8 列(含 Score-only 對照組)數字以 `figure/ranking_table_3.png` 及
`results/` 底下對應 json 為準,這裡只列 Diagnosis 條件方便快速參考。

## 重要:hinted 統一說明

2026-09-23 起,`prompts/program_hinted.md`(比主線 `program.md` 多兩段
「Diagnostic-to-Parameter Guidance」「Exploration Guidance」提示)成為
Llama 3.1 8B / Phi-4 14B / Gemma 4 12B 三個小模型正式的診斷組標準。
2026-09-26,Claude Haiku 4.5 也補跑 `--hinted`,四個模型的診斷組 prompt
現在完全一致,論文裡統一稱為「Diagnosis」,不再需要區分 hinted/non-hinted。

檔名慣例(見 `.gitignore` 底部說明,務必保持一致,不然 clone 下來的人
跑 `scripts/plot_trajectory.py` / `plot_ranking_table.py` 會找不到檔案):
四個模型統一用 `results/full_diagnosis_<model>.json`(不帶 `_hinted`
後綴,因為 hinted 現在就是唯一的正式診斷組)。Claude Haiku 4.5 那份舊的
非 hinted 版本改名成 `full_diagnosis_claude-haiku-4.5_nonhinted_reference.json`,
只留在本機當參考,不進版控。

## 環境設定

1. Python 版本:建議 3.10+
2. 建立虛擬環境並安裝依賴:
   ```
   python -m venv venv
   venv\Scripts\activate        # Windows
   pip install -r requirements.txt
   ```
3. PyTorch 安裝:依實際 GPU 型號到 https://pytorch.org/get-started/locally/ 選對應 CUDA 版本指令,requirements.txt 裡不含 torch,要另外裝。
4. LLM 金鑰設定:
   - Claude 系列:設定環境變數 `ANTHROPIC_API_KEY`
   - Llama 3.1 8B / Phi-4 14B / Gemma 4 12B:本機跑 Ollama,見 `src/llm_agent.py`
5. 資料集:`datasets/` 已隨 repo 一起進版控(937 張圖片,YOLOv8 格式,來源見
   `datasets/README.roboflow.txt`:https://universe.roboflow.com/traffic-light-for-yolo/mix-dataset-9hfip ,
   CC BY 4.0),clone 下來就有,不用另外下載。`config/settings.yaml` 裡的路徑維持相對路徑,不要寫死絕對路徑。
   `runs/`、`weights/`、`yolov8n.pt` 仍不進版控(體積大且可重新產生/下載,見下方「資料夾結構」)。
   預訓練權重(`yolov8n.pt`)會在第一次執行時由 ultralytics 自動下載。

## 已知陷阱

- `optimizer='auto'` 會覆蓋自訂超參數,務必明確指定 optimizer。
- Ultralytics 的 `pretrained` 參數要明確傳入,不要依賴預設行為。
- Windows 下 `workers>0` 的 multiprocessing 需要 `if __name__ == "__main__":` 保護。
- YOLOv8 預設 `cos_lr=False`,學習率是線性衰減不是 cosine,寫 prompt 時注意別誤述。

## 資料夾結構

```
config/       設定檔(路徑、batch size 等)
prompts/      LLM prompt 模板(program.md 主線 / program_hinted.md 現行診斷組標準)
src/          核心程式(診斷、LLM 呼叫、訓練包裝、傳統方法基準線、主迴圈)
experiments/  各組實驗的執行進入點
scripts/      畫圖腳本(plot_trajectory.py / plot_ranking_table.py)+ 一次性工具腳本
results/      實驗結果輸出位置(progress/full_diagnosis/score_only json,論文數據來源)
docs/         決策紀錄(供論文 Method 章節引用)
figure/       論文/簡報用的最終圖表(PNG,有進版控,詳見上方「專案狀態」)
```

以下資料夾不進版控(見 `.gitignore`),體積大且可重新產生/下載:
`venv/`(虛擬環境)、`runs/`(每輪訓練的權重與過程圖,可由 `results/*.json` + `src/hpo_loop.py` 重新產生)、
`weights/`、`yolov8n.pt`(可重新下載)、`Claude outputs/`(產圖過程中的草稿/被否決版本,不是最終交付物,最終版在 `figure/`)。`datasets/` 有進版控。

## 如何執行

```
python experiments/run_full_diagnosis.py --model <model> --hinted   # 完整診斷組(見上方「hinted 統一說明」)
python experiments/run_score_only.py       # score-only 對照組
python experiments/run_baselines.py        # 傳統方法基準線(novice / random search / Optuna TPE)
python scripts/plot_trajectory.py          # 重新產生收斂曲線圖(figure/chart.png 的來源)
python scripts/plot_ranking_table.py       # 重新產生排行榜表格(figure/ranking_table_3.png 的來源)
```
