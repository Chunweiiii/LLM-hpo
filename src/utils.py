"""共用工具函式。"""

import yaml


def load_settings(path: str = "config/settings.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(obj, path: str):
    """存 JSON。會先寫到暫存檔,成功後才用 os.replace() 原子性地取代目標檔案,
    避免寫到一半當機把原本已經存好的內容也一起毀掉(2026-09 實測發生過,
    見 docs/decisions.md)。default=_default 讓 numpy 型別(numpy.bool_/
    numpy.integer/numpy.floating/numpy.ndarray)自動轉成 Python 原生型別,
    避免 TypeError: Object of type bool/int64/float64 is not JSON serializable。
    """
    import json
    import os
    import numpy as np

    def _default(o):
        if isinstance(o, (np.bool_, np.integer, np.floating)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_default)
    os.replace(tmp_path, path)


def load_json(path: str):
    """讀回 save_json 存的進度檔,hpo_loop.py 的自動接續機制用。"""
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)