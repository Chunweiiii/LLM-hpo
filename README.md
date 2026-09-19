# LLM-HPO Traffic Light Detection

LLM 讀取 YOLOv8n 訓練動態,診斷後提出下一輪超參數建議,迭代 10 輪選出最佳模型。

## 專案狀態

骨架已建立,以下項目待補(見 `docs/decisions.md`):
- [x] 用 `scripts/find_batch_size.py` 在新機器上定出 batch_size:實測49,取整為 48
- [ ] batch_size 定案後,在新機器上重跑 3-seed baseline(舊數字是 8GB 筆電測的,不能沿用),用 `scripts/run_baseline_3seed.py`
- [x] `src/summarize_log.py` 的 Facts / D1-D5 診斷邏輯已實作完成,`compute_class_stats()` 也接上了 per-class 資料來源(見 `src/train_runner.py`),尚未實跑驗證過
- [ ] D4 / D5 兩個診斷閾值的具體數字(D1 已定案 0.7,見 `docs/decisions.md`)——邏輯已寫好,只差把佔位符數字換成新機器 baseline 算出來的正式值
- [x] 關鍵點抽樣的轉折點演算法:簡單斜率變化
- [x] `prompts/program.md` 的 Domain Knowledge 段落:保留
- [x] 本機/開源 LLM 選型:Qwen3-14B(取代原本的 Qwen2.5-Coder-7B,理由見 `docs/decisions.md`)

實驗室顯卡:NVIDIA RTX PRO 4000 Blackwell,24GB GDDR7(已確認)。系統 RAM 16GB,YOLO 訓練 `workers` 已保守鎖定為 4。

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
   - Qwen3-14B:本機跑 Ollama(`ollama pull qwen3:14b`),見 `src/llm_agent.py`
5. 資料集下載:`datasets/`、`runs/`、`weights/`、`yolov8n.pt` 已加進 `.gitignore`(體積大且可重新取得,不進版控)。
   資料集(937 張圖片,YOLOv8 格式,CC BY 4.0)從 Roboflow Universe 下載後解壓到 `datasets/`,
   維持 `train/valid/test` 三個子資料夾與 `data.yaml`:
   https://universe.roboflow.com/traffic-light-for-yolo/mix-dataset-9hfip
   （細節見 `datasets/README.roboflow.txt`)。下載完成後在 `config/settings.yaml` 填相對路徑,不要寫死絕對路徑。
   預訓練權重(`yolov8n.pt`)會在第一次執行時由 ultralytics 自動下載。

## 已知陷阱

- `optimizer='auto'` 會覆蓋自訂超參數,務必明確指定 optimizer。
- Ultralytics 的 `pretrained` 參數要明確傳入,不要依賴預設行為。
- Windows 下 `workers>0` 的 multiprocessing 需要 `if __name__ == "__main__":` 保護。
- YOLOv8 預設 `cos_lr=False`,學習率是線性衰減不是 cosine,寫 prompt 時注意別誤述。

## 資料夾結構

```
config/       設定檔(路徑、batch size 等)
prompts/      LLM prompt 模板
src/          核心程式(診斷、LLM 呼叫、訓練包裝、傳統方法基準線、主迴圈)
experiments/  各組實驗的執行進入點
scripts/      一次性工具腳本(例如找 batch size)
results/      實驗結果輸出位置(9 種方法的 progress/history json,論文數據來源)
docs/         決策紀錄(供論文 Method 章節引用)
Claude outputs/  收斂曲線圖與最終成績表(PNG,供論文/簡報使用)
```

以下資料夾不進版控(見 `.gitignore`),體積大且可重新產生/下載:
`venv/`(虛擬環境)、`datasets/`(可從上方連結重新下載)、`runs/`(每輪訓練的權重與過程圖,
可由 `results/*.json` + `src/hpo_loop.py` 重新產生)、`weights/`、`yolov8n.pt`(可重新下載)。

## 如何執行

```
python scripts/find_batch_size.py          # 第一步:在新機器上定出 batch_size
python scripts/run_baseline_3seed.py       # 第二步:batch_size定案後,跑3-seed baseline定出 mAP/雜訊門檻
python experiments/run_score_only.py       # score-only 對照組
python experiments/run_full_diagnosis.py   # 完整診斷組(4 個 LLM)
python experiments/run_baselines.py        # 傳統方法基準線
```
