"""Builds Layer 2's training data: job postings and scored candidate-job pairs.

Candidates are not duplicated here - they live in data_layer1/<split>/ and are
referenced by id. Layer 2 consumes the structured fields Layer 1 would produce
from those CVs under correct extraction, which is what keeps the two layers
decoupled and separately trainable.

Splits are kept disjoint on *both* axes: train pairs use train candidates and
train jobs, test pairs use test candidates and test jobs. So the held-out set
measures generalisation to roles the model has never seen, not just to new
applicants for familiar roles.

    python generate_layer2_data.py --train-jobs 200 --test-jobs 50
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import random
from pathlib import Path

import scoring_rules as rules

LAYER1_DIR = Path(__file__).resolve().parents[2] / "data_layer1"
OUT_DIR = Path(__file__).resolve().parent.parent

DOMAIN_TITLES = {
    "Software Development": ["Backend Engineer", "Full-Stack Engineer", "Software Engineer",
                             "Platform Engineer", "Senior Software Engineer"],
    "Data/AI/ML": ["Machine Learning Engineer", "Data Scientist", "Applied Scientist",
                   "MLOps Engineer", "Senior ML Engineer"],
    "DevOps/Infrastructure": ["DevOps Engineer", "Site Reliability Engineer",
                              "Cloud Infrastructure Engineer", "Platform Engineer"],
    "Product/Design": ["Product Manager", "Product Designer", "UX Designer",
                       "Senior Product Manager"],
    "Data Analysis": ["Data Analyst", "Analytics Engineer", "BI Analyst",
                      "Senior Data Analyst"],
    "QA/Testing": ["QA Engineer", "Test Automation Engineer", "SDET", "Quality Lead"],
}

SENIORITY = [
    ("Junior", 1, 3, "Bachelor"),
    ("Mid-level", 3, 6, "Bachelor"),
    ("Senior", 6, 10, "Bachelor"),
    ("Lead", 10, 15, "Master"),
]


def load_candidates(split: str) -> list[dict]:
    path = LAYER1_DIR / split / f"{split}.jsonl"
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def skill_pools(candidates: list[dict]) -> dict[str, list[str]]:
    """Which skills people in each domain actually hold.

    Drawing job requirements from this rather than a hand-written list keeps
    postings plausible and guarantees the population contains people who could
    genuinely match them.
    """
    counter: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for candidate in candidates:
        domain = candidate["general_experience"]["domain"]
        for skill in candidate["skills"]:
            counter[domain][skill["name"]] += 1
    return {domain: [name for name, _ in counts.most_common(22)]
            for domain, counts in counter.items()}


def make_job(index: int, rng: random.Random, pools: dict[str, list[str]], prefix: str) -> dict:
    domain = rng.choice(rules.DOMAINS)
    level, min_years, max_years, min_degree = rng.choice(SENIORITY)
    pool = pools.get(domain) or [s for names in pools.values() for s in names]

    n_skills = rng.randint(3, 7)
    chosen = rng.sample(pool, min(n_skills, len(pool)))
    required_skills = [{
        "name": name,
        "min_years": rng.choice([1, 2, 2, 3, 3, 5]),
        "weight": round(rng.uniform(0.5, 1.0), 2),
    } for name in chosen]

    # Weights vary per posting: some roles are stack-led, others prize seniority.
    raw = {
        "education": rng.uniform(0.5, 1.5),
        "experience": rng.uniform(1.0, 2.5),
        "stack": rng.uniform(1.5, 3.0),
        "company": rng.uniform(0.4, 1.5),
    }
    total = sum(raw.values())
    weights = {k: round(v / total, 4) for k, v in raw.items()}

    return {
        "id": f"{prefix}{index:04d}",
        "title": f"{level} {rng.choice(DOMAIN_TITLES[domain])}",
        "domain": domain,
        "seniority": level,
        "required_experience_years": rng.randint(min_years, max_years),
        "preferred_education": min_degree,
        "preferred_field": (rng.choice(sorted(rules.TECHNICAL_FIELDS))
                            if rng.random() < 0.55 else None),
        "prefers_big_tech": rng.random() < 0.30,
        "required_skills": required_skills,
        "weights": weights,
    }


def flatten(pair: dict, job: dict, candidate: dict) -> dict:
    return {
        "job_id": job["id"],
        "candidate_id": candidate["id"],
        "job_title": job["title"],
        "job_domain": job["domain"],
        "required_years": job["required_experience_years"],
        "candidate_domain": candidate["general_experience"]["domain"],
        "candidate_years": candidate["general_experience"]["total_years"],
        "candidate_degree": candidate["education"][0]["level"] if candidate["education"] else "",
        **pair,
    }


def sample_applicants(job: dict, by_id: dict, by_domain: dict, pool_size: int,
                      rng: random.Random, in_domain_share: float = 0.55) -> list[str]:
    """Build a realistic applicant pool for one posting.

    Sampling candidates uniformly would make most pairs share no required skill
    at all, because a random person is rarely in the right field. Real pools are
    self-selecting: most applicants already work in the domain, alongside a tail
    of speculative applications from elsewhere. That balance also stops the
    labels being dominated by near-zero stack scores.
    """
    want_in_domain = int(pool_size * in_domain_share)
    in_domain = by_domain.get(job["domain"], [])
    chosen = list(rng.sample(in_domain, min(want_in_domain, len(in_domain))))

    remaining = [cid for cid in by_id if cid not in set(chosen)]
    chosen += rng.sample(remaining, min(pool_size - len(chosen), len(remaining)))
    return chosen


def build_split(split: str, n_jobs: int, pool_size: int, seed: int,
                pools: dict[str, list[str]], prefix: str) -> tuple[list, list, list]:
    rng = random.Random(f"{seed}-{split}")
    candidates = load_candidates(split)
    by_id = {c["id"]: c for c in candidates}

    by_domain: dict[str, list[str]] = collections.defaultdict(list)
    for candidate in candidates:
        by_domain[candidate["general_experience"]["domain"]].append(candidate["id"])

    jobs = [make_job(i + 1, rng, pools, prefix) for i in range(n_jobs)]
    pairs, flat = [], []
    for job in jobs:
        sampled = sample_applicants(job, by_id, by_domain, pool_size, rng)
        for candidate_id in sampled:
            candidate = by_id[candidate_id]
            scores = rules.score_pair(candidate, job, rng)
            pairs.append({"job_id": job["id"], "candidate_id": candidate_id, **scores})
            flat.append(flatten(scores, job, candidate))
    return jobs, pairs, flat


def write_split(split: str, jobs: list, pairs: list, flat: list):
    out = OUT_DIR / split
    out.mkdir(parents=True, exist_ok=True)

    with (out / "jobs.jsonl").open("w", encoding="utf-8") as f:
        for job in jobs:
            f.write(json.dumps(job, ensure_ascii=False) + "\n")
    with (out / "pairs.jsonl").open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    with (out / "pairs.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
        writer.writeheader()
        writer.writerows(flat)
    print(f"{split}: {len(jobs)} jobs, {len(pairs)} pairs -> {out}")


def main():
    ap = argparse.ArgumentParser(description="Generate Layer 2 job postings and scored pairs.")
    ap.add_argument("--train-jobs", type=int, default=200)
    ap.add_argument("--test-jobs", type=int, default=50)
    ap.add_argument("--pool-size", type=int, default=250,
                    help="Candidates scored per job posting")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Job requirements are drawn from the training population only, so the test
    # split never influences what the postings ask for.
    pools = skill_pools(load_candidates("train"))

    for split, n_jobs, prefix in (("train", args.train_jobs, "JOBTR"),
                                  ("test", args.test_jobs, "JOBTE")):
        jobs, pairs, flat = build_split(split, n_jobs, args.pool_size, args.seed, pools, prefix)
        write_split(split, jobs, pairs, flat)

        overall = [p["overall_score"] for p in pairs]
        print(f"  overall score: min {min(overall)} / mean {sum(overall) / len(overall):.1f} "
              f"/ max {max(overall)}")


if __name__ == "__main__":
    main()
