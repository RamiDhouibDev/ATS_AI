"""Loads Layer 2's training data.

Everything comes from data_layer2/<split>/: candidates.jsonl, jobs.jsonl and
pairs.jsonl. Pairs hold ids plus the four section scores; this module joins them
into Examples, either flat or grouped by job posting.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data_layer2"

# Model targets, in a fixed order. `overall_score` is excluded: it is a weighted
# combination of these four, not a fifth independent thing to predict.
SECTIONS = ["education_score", "relevant_experience_score",
            "stack_experience_score", "companies_score"]


@dataclass
class SkillRequirement:
    """One skill a posting asks for, with its depth and importance."""
    name: str
    min_years: float
    weight: float


@dataclass
class JobPosting:
    id: str
    title: str
    domain: str
    seniority: str
    required_experience_years: int
    preferred_education: str
    preferred_field: str | None          # None when the posting leaves it open
    required_skills: list[SkillRequirement] = field(default_factory=list)
    # Sums to 1, and differs per posting - this is what makes scoring
    # role-conditioned rather than one fixed formula.
    weights: dict[str, float] = field(default_factory=dict)

    def required_skill_names(self) -> set[str]:
        return {requirement.name for requirement in self.required_skills}


@dataclass
class Candidate:
    """A candidate's structured fields - what a CV yields once parsed."""
    id: str
    name: str
    domain: str
    total_years: float
    education: list[dict] = field(default_factory=list)   # highest degree first
    skills: list[dict] = field(default_factory=list)
    companies: list[dict] = field(default_factory=list)

    @property
    def highest_degree(self) -> str | None:
        return self.education[0]["level"] if self.education else None

    def skill_years(self) -> dict[str, float]:
        return {skill["name"]: skill["years"] for skill in self.skills}

    def best_tier(self) -> int | None:
        """Best employer tier reached; 1 is the strongest, None if never employed."""
        return min((employer["tier"] for employer in self.companies), default=None)


@dataclass
class Example:
    """One (candidate, job) pair and the scores it should receive."""
    candidate: Candidate
    job: JobPosting
    labels: dict[str, int]

    @property
    def target_vector(self) -> list[int]:
        return [self.labels[name] for name in SECTIONS]

    @property
    def overall(self) -> int:
        return self.labels["overall_score"]


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_jobs(split: str, data_dir: Path | None = None) -> dict[str, JobPosting]:
    """Job postings for one split, keyed by id."""
    jobs = {}
    for raw in _read_jsonl((data_dir or DATA_DIR) / split / "jobs.jsonl"):
        jobs[raw["id"]] = JobPosting(
            id=raw["id"],
            title=raw["title"],
            domain=raw["domain"],
            seniority=raw["seniority"],
            required_experience_years=raw["required_experience_years"],
            preferred_education=raw["preferred_education"],
            preferred_field=raw.get("preferred_field"),
            required_skills=[SkillRequirement(**requirement)
                             for requirement in raw.get("required_skills", [])],
            weights=raw.get("weights", {}),
        )
    return jobs


def load_candidates(split: str, data_dir: Path | None = None) -> dict[str, Candidate]:
    """Candidates for one split, keyed by id."""
    candidates = {}
    for raw in _read_jsonl((data_dir or DATA_DIR) / split / "candidates.jsonl"):
        general = raw["general_experience"]
        candidates[raw["id"]] = Candidate(
            id=raw["id"],
            name=raw["name"],
            domain=general["domain"],
            total_years=general["total_years"],
            education=raw.get("education", []),
            skills=raw.get("skills", []),
            companies=raw.get("companies", []),
        )
    return candidates


def load_examples(split: str, limit: int | None = None,
                  data_dir: Path | None = None) -> list[Example]:
    """Every scored pair in a split, as a flat list."""
    base = data_dir or DATA_DIR
    jobs = load_jobs(split, base)
    candidates = load_candidates(split, base)

    examples, orphans = [], 0
    for pair in _read_jsonl(base / split / "pairs.jsonl"):
        job = jobs.get(pair["job_id"])
        candidate = candidates.get(pair["candidate_id"])
        # Only happens if the three files were regenerated out of step; a count
        # at the end is more useful than dying on the first bad row.
        if job is None or candidate is None:
            orphans += 1
            continue
        labels = {field: score for field, score in pair.items()
                  if field.endswith("_score")}
        examples.append(Example(candidate=candidate, job=job, labels=labels))
        if limit and len(examples) >= limit:
            break

    if orphans:
        print(f"WARNING: {orphans} pairs in '{split}' reference a missing id - regenerate")
    return examples


def load_ranking_pools(split: str, limit_jobs: int | None = None
                       ) -> list[tuple[JobPosting, list[Example]]]:
    """Examples grouped by posting.

    Ranking metrics like NDCG@20 are only defined within one job's applicant
    pool, so a flat list of pairs cannot express them.
    """
    grouped: dict[str, list[Example]] = defaultdict(list)
    for example in load_examples(split):
        grouped[example.job.id].append(example)

    pools = [(examples[0].job, examples) for examples in grouped.values()]
    pools.sort(key=lambda item: item[0].id)      # stable order across runs
    return pools[:limit_jobs] if limit_jobs else pools


def describe(split: str) -> dict:
    """Shape of a split - a sanity check before training on it."""
    pools = load_ranking_pools(split)
    examples = [example for _, pool in pools for example in pool]
    overall = [example.overall for example in examples]
    return {
        "split": split,
        "jobs": len(pools),
        "candidates": len({example.candidate.id for example in examples}),
        "pairs": len(examples),
        "pool_size": len(examples) // max(1, len(pools)),
        "overall_min": min(overall),
        "overall_mean": round(sum(overall) / len(overall), 1),
        "overall_max": max(overall),
    }


if __name__ == "__main__":
    for split_name in ("train", "test"):
        print(describe(split_name))
