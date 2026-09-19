"""
在實驗室新機器(RTX PRO 4000 Blackwell, 24GB VRAM)上跑一次,
用 Ultralytics 內建的 AutoBatch 找出穩定可用的最大 batch size。

找到後把結果**手動**填進 config/settings.yaml 的 training.batch_size,
之後所有實驗都用這個固定值,不要每次都重新自動判定
(否則不同次結果因 batch size 不同而不可比)。

用法:
    python scripts/find_batch_size.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO
from src.utils import load_settings

if __name__ == "__main__":
    settings = load_settings()
    model = YOLO("yolov8n.pt")

    # batch=-1 觸發 AutoBatch:以預設 60% GPU 記憶體使用率為目標,
    # 自動試跑找出建議的 batch size,epochs=1 只是為了讓它跑完偵測流程,
    # 不是正式訓練。
    # data.yaml 路徑從 config/settings.yaml 的 dataset.root 讀,不要寫死,
    # 避免跟 train_runner.py 用到不同路徑(見 docs/decisions.md 的雙層巢狀說明)。
    model.train(
        data=str(Path(settings["dataset"]["root"]) / "data.yaml"),
        epochs=1,
        batch=-1,
        imgsz=640,
        optimizer="AdamW",
        pretrained=True,
        name="autobatch_probe",
    )

    print("\n上面 log 裡 'AutoBatch' 那行會顯示建議的 batch size,")
    print("把這個數字手動填進 config/settings.yaml 的 training.batch_size。")
