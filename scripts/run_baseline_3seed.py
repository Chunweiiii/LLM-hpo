"""
[已停用] 這份腳本已經被 experiments/run_baseline_3seed.py 取代,不要再用這份。

原因:這份腳本沒有中斷接續機制、seed 清單寫死從 config/settings.yaml 讀,
2026-09 擴充到 5-seed 時改寫成 experiments/run_baseline_3seed.py(支援
--seeds 參數指定要跑哪些、每個 seed 跑完立刻存檔、重跑會自動跳過已完成的
seed)。如果不小心執行這份舊版,會用沒有接續機制的邏輯重跑全部 seed,
浪費時間也可能跟 experiments/ 那份的結果搞混,所以直接讓它報錯提醒,
不再讓它正常執行。

見 docs/decisions.md「experiments/run_baseline_3seed.py 取代
scripts/run_baseline_3seed.py」。

正確用法:
    python experiments/run_baseline_3seed.py
"""

raise RuntimeError(
    "這份腳本已停用,請改用 python experiments/run_baseline_3seed.py"
    "(支援中斷接續,見檔案開頭說明與 docs/decisions.md)"
)
