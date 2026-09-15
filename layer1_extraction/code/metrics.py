"""Field-level scoring of an extraction against the ground-truth labels."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from .schema import CVRecord

FUZZY_MATCH = 88   # rapidfuzz ratio at which two employer names are "the same"


def _norm(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _match_sets(predicted: set[str], gold: set[str], fuzzy: bool = False) -> tuple[int, int, int]:
    """Returns (true positives, predicted count, gold count)."""
    remaining = {_norm(name) for name in gold}
    hits = 0
    for item in {_norm(name) for name in predicted}:
        if item in remaining:
            remaining.discard(item)
            hits += 1
        elif fuzzy:
            near = next((candidate for candidate in remaining
                         if fuzz.ratio(item, candidate) >= FUZZY_MATCH), None)
            if near:
                remaining.discard(near)
                hits += 1
    return hits, len(predicted), len(gold)


@dataclass
class Tally:
    """Accumulates counts so precision/recall are computed over the whole split."""
    tp: int = 0
    predicted: int = 0
    gold: int = 0

    def add(self, tp: int, predicted: int, gold: int):
        self.tp += tp
        self.predicted += predicted
        self.gold += gold

    @property
    def precision(self) -> float:
        return self.tp / self.predicted if self.predicted else 0.0

    @property
    def recall(self) -> float:
        return self.tp / self.gold if self.gold else 0.0

    @property
    def f1(self) -> float:
        precision, recall = self.precision, self.recall
        return (2 * precision * recall / (precision + recall)
                if precision + recall else 0.0)


@dataclass
class Report:
    n: int = 0
    name_correct: int = 0
    degree_correct: int = 0
    field_correct: int = 0
    skills: Tally = field(default_factory=Tally)
    companies: Tally = field(default_factory=Tally)
    start_date_correct: int = 0
    start_date_total: int = 0
    years_abs_error: float = 0.0
    years_scored: int = 0
    escalated: int = 0

    def add(self, predicted: CVRecord, gold: CVRecord, escalated: bool = False):
        self.n += 1
        self.escalated += bool(escalated)

        self.name_correct += _norm(predicted.name) == _norm(gold.name)

        gold_top = gold.highest_education
        pred_top = predicted.highest_education
        if gold_top:
            self.degree_correct += bool(pred_top) and _norm(pred_top.level) == _norm(gold_top.level)
            if gold_top.field_of_study:
                self.field_correct += bool(pred_top) and \
                    _norm(pred_top.field_of_study) == _norm(gold_top.field_of_study)

        self.skills.add(*_match_sets(predicted.skill_names(), gold.skill_names()))
        self.companies.add(*_match_sets(predicted.company_names(), gold.company_names(), fuzzy=True))

        gold_starts = {_norm(company.name): company.start_date
                       for company in gold.companies}
        for company in predicted.companies:
            key = _norm(company.name)
            if key in gold_starts and gold_starts[key]:
                self.start_date_total += 1
                self.start_date_correct += (company.start_date or "")[:4] == gold_starts[key][:4]

        if gold.total_years is not None and predicted.total_years is not None:
            self.years_abs_error += abs(predicted.total_years - gold.total_years)
            self.years_scored += 1

    @property
    def years_mae(self) -> float:
        return self.years_abs_error / self.years_scored if self.years_scored else float("nan")

    def as_row(self, label: str) -> str:
        def pct(value):
            return f"{value * 100:5.1f}%"

        return (f"{label:<22s} {self.n:>5d} "
                f"{pct(self.name_correct / self.n if self.n else 0)} "
                f"{pct(self.degree_correct / self.n if self.n else 0)} "
                f"{pct(self.skills.f1)} "
                f"{pct(self.companies.f1)} "
                f"{pct(self.start_date_correct / self.start_date_total if self.start_date_total else 0)} "
                f"{self.years_mae:>6.2f} "
                f"{pct(self.escalated / self.n if self.n else 0)}")


HEADER = (f"{'group':<22s} {'n':>5s} {'name':>6s} {'degree':>6s} {'skillF1':>6s} "
          f"{'compF1':>6s} {'start':>6s} {'yrsMAE':>6s} {'esc':>6s}")
