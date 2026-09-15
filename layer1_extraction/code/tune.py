"""Fits the escalation threshold for Layer 1 on the TRAIN split.

This is what "training" means for a rule-based extractor. There is no
backpropagation: the parser is fixed, and the one free parameter is the
confidence below which a document is handed to the (more expensive) LLM
extractor. We choose it on train data only, then report on test.

The objective is a cost/quality trade-off, not raw accuracy. Escalating every
CV would maximise quality and cost the most; escalating none is free and loses
the hard documents. We pick the threshold that keeps the parse quality of the
documents we *keep* above a target, while escalating as few as possible.

    python -m layer1_extraction.code.tune --target 0.85
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

from .ats_parser import parse
from .dataset import load_split
from .metrics import Report
from .pdf_text import read_document_cached as read_document

CONFIG_PATH = Path(__file__).resolve().parent / "config.json"


def parse_quality(predicted, gold) -> float:
    """Per-document quality in 0..1, averaged over the fields that matter."""
    report = Report()
    report.add(predicted, gold)
    parts = [
        report.name_correct,
        report.degree_correct,
        report.skills.f1,
        report.companies.f1,
    ]
    return sum(parts) / len(parts)


def evaluate_threshold(scored: list[tuple[float, float]], threshold: float) -> tuple[float, float]:
    """Returns (mean quality of kept parses, escalation rate)."""
    kept = [quality for confidence, quality in scored if confidence >= threshold]
    escalated = len(scored) - len(kept)
    mean_quality = sum(kept) / len(kept) if kept else 1.0
    return mean_quality, escalated / len(scored)


def main():
    ap = argparse.ArgumentParser(description="Fit the LLM-escalation threshold on train.")
    ap.add_argument("--target", type=float, default=0.85,
                    help="Minimum mean parse quality required of non-escalated CVs")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=CONFIG_PATH)
    args = ap.parse_args()

    samples = load_split("train", limit=args.limit)
    print(f"Fitting on {len(samples)} training CVs...")

    scored: list[tuple[float, float]] = []
    for index, sample in enumerate(samples, 1):
        document = read_document(sample.pdf_path)
        result = parse(document)
        scored.append((result.confidence, parse_quality(result.record, sample.gold)))
        if index % 100 == 0:
            print(f"  {index}/{len(samples)}")

    print(f"\n{'threshold':>9s} {'kept quality':>13s} {'escalated':>10s}")
    best = None
    for step in range(0, 101, 5):
        threshold = step / 100
        quality, escalation = evaluate_threshold(scored, threshold)
        marker = ""
        if quality >= args.target and (best is None or escalation < best[2]):
            best = (threshold, quality, escalation)
            marker = "  <-"
        print(f"{threshold:>9.2f} {quality:>12.1%} {escalation:>10.1%}{marker}")

    if best is None:
        # Nothing reaches the target; fall back to escalating everything the
        # parser is least sure about rather than silently accepting bad parses.
        best = (1.0, *evaluate_threshold(scored, 1.0))
        print("\nNo threshold reaches the target quality - escalating all low-confidence CVs.")

    threshold, quality, escalation = best
    args.out.write_text(json.dumps({
        "escalation_threshold": threshold,
        "target_quality": args.target,
        "train_kept_quality": round(quality, 4),
        "train_escalation_rate": round(escalation, 4),
        "train_size": len(samples),
    }, indent=2), encoding="utf-8")

    print(f"\nChosen threshold {threshold:.2f} -> keeps {1 - escalation:.1%} of CVs "
          f"at {quality:.1%} mean parse quality")
    print(f"Written to {args.out}")


if __name__ == "__main__":
    main()
