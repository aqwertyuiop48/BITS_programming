"""Summarize token usage across LLM demo runs.

This utility scans run folders for metadata.json + params.json and reports:
- Average prompt/response/total tokens per budget mode
- Savings percentages (low vs detailed, low vs balanced, balanced vs detailed)

Examples:
    python token_budget_report.py --task rag_grounded_answer
    python token_budget_report.py --task general_assistant --model gemini-2.5-flash
    python token_budget_report.py --runs-dir runs
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


@dataclass
class RunRecord:
    run_name: str
    task: str | None
    model: str | None
    budget_mode: str
    prompt_tokens: int
    response_tokens: int
    total_tokens: int


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _to_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def collect_records(runs_dir: Path) -> list[RunRecord]:
    records: list[RunRecord] = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        metadata_path = run_dir / "metadata.json"
        if not metadata_path.exists():
            continue

        metadata = _load_json(metadata_path)
        params = _load_json(run_dir / "params.json")

        prompt_tokens = _to_int(metadata.get("prompt_tokens"))
        response_tokens = _to_int(metadata.get("response_tokens"))
        total_tokens = _to_int(metadata.get("total_tokens"))

        if prompt_tokens is None or response_tokens is None or total_tokens is None:
            continue

        task = metadata.get("task")
        model = metadata.get("model") or params.get("model_name")
        budget_mode = (
            metadata.get("budget_mode")
            or params.get("budget_mode")
            or "unknown"
        )

        records.append(
            RunRecord(
                run_name=run_dir.name,
                task=task,
                model=model,
                budget_mode=str(budget_mode),
                prompt_tokens=prompt_tokens,
                response_tokens=response_tokens,
                total_tokens=total_tokens,
            )
        )

    return records


def print_mode_table(records: list[RunRecord]) -> None:
    grouped: dict[str, list[RunRecord]] = {}
    for rec in records:
        grouped.setdefault(rec.budget_mode, []).append(rec)

    print("\nToken summary by budget mode")
    print("=" * 72)
    print(f"{'mode':<12} {'n':>4} {'avg_prompt':>12} {'avg_response':>14} {'avg_total':>10}")
    print("-" * 72)

    for mode in sorted(grouped.keys()):
        rows = grouped[mode]
        avg_prompt = mean(r.prompt_tokens for r in rows)
        avg_response = mean(r.response_tokens for r in rows)
        avg_total = mean(r.total_tokens for r in rows)
        print(
            f"{mode:<12} {len(rows):>4} "
            f"{avg_prompt:>12.1f} {avg_response:>14.1f} {avg_total:>10.1f}"
        )


def _avg_total_for_mode(records: list[RunRecord], mode: str) -> float | None:
    rows = [r.total_tokens for r in records if r.budget_mode == mode]
    if not rows:
        return None
    return mean(rows)


def _print_savings_line(label: str, base: float | None, better: float | None) -> None:
    if base is None or better is None:
        print(f"- {label}: not enough data")
        return
    if base <= 0:
        print(f"- {label}: invalid baseline")
        return

    savings_pct = ((base - better) / base) * 100.0
    token_delta = base - better
    print(f"- {label}: {savings_pct:.1f}% ({token_delta:.1f} tokens avg)")


def print_savings(records: list[RunRecord]) -> None:
    low = _avg_total_for_mode(records, "low")
    balanced = _avg_total_for_mode(records, "balanced")
    detailed = _avg_total_for_mode(records, "detailed")

    print("\nSavings comparisons")
    print("=" * 72)
    _print_savings_line("low vs detailed", detailed, low)
    _print_savings_line("low vs balanced", balanced, low)
    _print_savings_line("balanced vs detailed", detailed, balanced)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize token usage for budget-mode demos")
    parser.add_argument("--runs-dir", default="runs", help="Runs directory path")
    parser.add_argument("--task", default=None, help="Optional task filter (e.g. rag_grounded_answer)")
    parser.add_argument("--model", default=None, help="Optional model filter (e.g. gemini-2.5-flash)")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    records = collect_records(runs_dir)

    if args.task:
        records = [r for r in records if r.task == args.task]
    if args.model:
        records = [r for r in records if r.model == args.model]

    if not records:
        print("No usable metadata rows found.")
        print("Tip: run llm_demo.py / rag_demo.py with --budget-mode low|balanced|detailed first.")
        return

    print(f"Rows analyzed: {len(records)}")
    tasks = sorted({r.task for r in records if r.task})
    models = sorted({r.model for r in records if r.model})
    print(f"Tasks: {', '.join(tasks) if tasks else 'n/a'}")
    print(f"Models: {', '.join(models) if models else 'n/a'}")

    print_mode_table(records)
    print_savings(records)


if __name__ == "__main__":
    main()
