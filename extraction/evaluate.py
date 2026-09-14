"""Scores Layer 1 against held-out labels.

    python -m extraction.evaluate --split test

Reports field-level accuracy overall and broken down by difficulty tier,
template and the specific traps the corpus builds in, using the escalation
threshold fitted by tune.py.
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")

from .ats_parser import parse
from .dataset import load_split
from .metrics import HEADER, Report
from .pdf_text import read_document
from .tune import CONFIG_PATH


def load_threshold(path: Path) -> float:
    if not path.exists():
        return 0.0
    return json.loads(path.read_text(encoding="utf-8")).get("escalation_threshold", 0.0)


def main():
    ap = argparse.ArgumentParser(description="Evaluate the rule-based extractor.")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--threshold", type=float, default=None,
                    help="Override the fitted escalation threshold")
    ap.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = ap.parse_args()

    threshold = args.threshold if args.threshold is not None else load_threshold(args.config)
    samples = load_split(args.split, limit=args.limit)
    print(f"Evaluating {len(samples)} CVs from '{args.split}' "
          f"at escalation threshold {threshold:.2f}\n")

    overall = Report()
    by_difficulty: dict[str, Report] = defaultdict(Report)
    by_template: dict[str, Report] = defaultdict(Report)
    by_trap: dict[str, Report] = defaultdict(Report)

    for i, sample in enumerate(samples, 1):
        document = read_document(sample.pdf_path)
        result = parse(document, threshold=threshold)
        escalated = result.needs_llm

        for report in (overall, by_difficulty[sample.difficulty], by_template[sample.template]):
            report.add(result.record, sample.gold, escalated)
        by_trap["skill years stated" if sample.skill_years_stated
                else "skill years implicit"].add(result.record, sample.gold, escalated)
        if sample.contact_in_header_only:
            by_trap["contact in header only"].add(result.record, sample.gold, escalated)
        if sample.pages > 1:
            by_trap["multi-page"].add(result.record, sample.gold, escalated)
        if i % 100 == 0:
            print(f"  {i}/{len(samples)}")

    print(f"\n{HEADER}")
    print("-" * len(HEADER))
    print(overall.as_row("ALL"))

    print("\nby difficulty")
    for key in ("easy", "medium", "hard", "scanned"):
        if by_difficulty[key].n:
            print(by_difficulty[key].as_row(f"  {key}"))

    print("\nby template")
    for key in sorted(by_template, key=lambda k: -by_template[k].n):
        print(by_template[key].as_row(f"  {key}"))

    print("\nby trap")
    for key in sorted(by_trap):
        print(by_trap[key].as_row(f"  {key}"))

    print(f"\nskills  P={overall.skills.precision:.1%} R={overall.skills.recall:.1%}")
    print(f"jobs    P={overall.companies.precision:.1%} R={overall.companies.recall:.1%}")
    print(f"escalated to LLM: {overall.escalated}/{overall.n} "
          f"({overall.escalated / overall.n:.1%})" if overall.n else "")


if __name__ == "__main__":
    main()
