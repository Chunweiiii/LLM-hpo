"""
測試工具(可重複使用):驗證「Facts+Flags 診斷報告 -> LLM -> JSON 超參數
提案」這整條線有沒有正確運作。四個 LLM(qwen3-14b / claude-haiku-4.5 /
claude-sonnet-5 / claude-opus-5)共用這一個檔案,用 --model 參數切換,
不用每個模型各開一個檔案。

用法:
    python scripts/test_llm_diagnosis.py --model qwen3-14b       (預設)
    python scripts/test_llm_diagnosis.py --model claude-sonnet-5
    python scripts/test_llm_diagnosis.py --model claude-haiku-4.5
    python scripts/test_llm_diagnosis.py --model claude-opus-5

呼叫 Claude 系列前記得先在 .env 設好 ANTHROPIC_API_KEY。

做法:借用 results/baseline_3seed.json 裡 seed=0 那筆已經存在的訓練紀錄
(results_csv + class_stats)組出報告,丟給指定的 LLM,不會重新訓練或
動用 GPU。
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.summarize_log import read_log, compute_facts, diagnose, to_report_text
from src.llm_agent import propose_hyperparameters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="qwen3-14b",
        choices=["qwen3-14b", "claude-haiku-4.5", "claude-sonnet-5", "claude-opus-5"],
    )
    args = parser.parse_args()

    summary = json.loads(Path("results/baseline_3seed.json").read_text(encoding="utf-8"))
    seed0_run = next(r for r in summary["per_seed_results"] if r["seed"] == 0)

    df = read_log(seed0_run["results_csv"])
    class_stats = seed0_run["class_stats"]
    facts = compute_facts(df, class_stats=class_stats)
    flags = diagnose(facts, class_stats=class_stats)
    report_text = to_report_text(facts, flags)

    print("=" * 60)
    print("組出來的報告(餵給 LLM 的內容):")
    print("=" * 60)
    print(report_text)
    print()

    system_prompt = Path("prompts/program.md").read_text(encoding="utf-8")

    print("=" * 60)
    print(f"呼叫 {args.model} ...")
    print("=" * 60)
    proposal = propose_hyperparameters(args.model, system_prompt, report_text)

    thinking = proposal.pop("_thinking", None)
    if thinking:
        print("--- thinking(推理過程,只印前 2000 字,只有 Qwen 有這欄位)---")
        print(thinking[:2000])
        print()
    print("--- 最終提案(JSON)---")
    print(json.dumps(proposal, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
