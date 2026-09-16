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

# (label, min years, max years, degree usually asked for).
# Required years are drawn from the band, so a "Senior" posting asks for 6-10
# years rather than one fixed number - real postings vary at the same level.
# Only Lead roles typically ask for a master's; below that a bachelor's is the
# common ask, which keeps education from correlating too tightly with seniority.
SENIORITY = [
    ("Junior", 1, 3, "Bachelor"),
    ("Mid-level", 3, 6, "Bachelor"),
    ("Senior", 6, 10, "Bachelor"),
    ("Lead", 10, 15, "Master"),
]


def load_candidates(split: str) -> list[dict]:
    """Layer 1's labelled candidates - the ground-truth structured fields.

    Read straight from data_layer1 rather than copied here, so there is exactly
    one definition of each candidate and the two layers cannot drift apart.
    """
    path = LAYER1_DIR / split / f"{split}.jsonl"
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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

    # Top 22 per domain: broad enough that postings differ from each other,
    # narrow enough to exclude the long tail of one-off tools that nobody else
    # holds - a requirement only one candidate could ever meet teaches the model
    # nothing and just drags every stack score toward zero.
    return {domain: [name for name, _ in counts.most_common(22)]
            for domain, counts in counter.items()}


def make_job(index: int, rng: random.Random, pools: dict[str, list[str]], prefix: str) -> dict:
    """One synthetic job posting.

    Every field here is something the scorer actually conditions on. Anything
    decorative (company name, location, salary) is left out deliberately: it
    would be unused input inviting the model to find noise correlations.
    """
    domain = rng.choice(rules.DOMAINS)
    level, min_years, max_years, min_degree = rng.choice(SENIORITY)
    pool = pools.get(domain) or [name for names in pools.values() for name in names]

    n_skills = rng.randint(3, 7)
    chosen = rng.sample(pool, min(n_skills, len(pool)))
    required_skills = [{
        "name": name,
        # Weighted toward 2-3 years: postings rarely demand 5+ on every item,
        # and a requirement nobody meets carries no ranking information.
        "min_years": rng.choice([1, 2, 2, 3, 3, 5]),
        # Per-skill importance, so a posting can distinguish must-haves from
        # nice-to-haves. Never zero - an unwanted skill would not be listed.
        "weight": round(rng.uniform(0.5, 1.0), 2),
    } for name in chosen]

    # How this posting trades the four sections off against each other. The
    # ranges encode a prior that holds across most real hiring: stack matters
    # most, then experience, with education and employer prestige as tiebreakers
    # rather than drivers. The ranges overlap, so an education-led posting is
    # possible - just uncommon.
    #
    # This is what makes the task genuinely role-conditioned. With one fixed
    # weighting the overall score would be a fixed function of the four
    # sections, and the model could ignore the job entirely when ranking.
    raw = {
        "education": rng.uniform(0.5, 1.5),
        "relevant_experience": rng.uniform(1.0, 2.5),
        "stack_experience": rng.uniform(1.5, 3.0),
        "companies": rng.uniform(0.4, 1.5),
    }
    total = sum(raw.values())
    weights = {section: round(weight / total, 4)                 # normalised to sum to 1
               for section, weight in raw.items()}

    return {
        "id": f"{prefix}{index:04d}",
        "title": f"{level} {rng.choice(DOMAIN_TITLES[domain])}",
        "domain": domain,
        "seniority": level,
        "required_experience_years": rng.randint(min_years, max_years),
        "preferred_education": min_degree,
        # Just over half of postings name a field; the rest leave it open, so
        # the model must handle a missing requirement rather than assume one.
        # sorted() keeps the draw reproducible - set iteration order is not.
        "preferred_field": (rng.choice(sorted(rules.TECHNICAL_FIELDS))
                            if rng.random() < 0.55 else None),
        "required_skills": required_skills,
        "weights": weights,
    }


def flatten(pair: dict, job: dict, candidate: dict) -> dict:
    """A row for pairs.csv: the scores plus enough context to sanity-check them.

    The CSV is for reading, not for training - it carries the handful of fields
    needed to tell at a glance whether a score looks right. Training reads
    pairs.jsonl and joins back to the full candidate and job records.
    """
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
    # Measured: uniform sampling left 69% of pairs sharing no required skill at
    # all, so stack scores piled up at zero and the section carried little
    # signal. At 55% in-domain that falls to 55% and mean stack rises 13 -> 19,
    # without touching the formula itself.
    in_domain = by_domain.get(job["domain"], [])

    # The share is a guarantee, not a request. Asking for a pool bigger than the
    # in-domain population can fill does not raise an error - it quietly fills
    # the remainder from everyone else and hands back a pool with the wrong
    # composition. That is how the test split ended up 17% in-domain against
    # train's 53%: 250 was larger than the whole 200-candidate split, so every
    # pool was simply everybody. Shrinking the pool to what the share can
    # actually support keeps both splits on the same distribution, which is the
    # one thing a held-out set has to get right.
    if in_domain:
        pool_size = min(pool_size, int(len(in_domain) / in_domain_share))

    want_in_domain = int(pool_size * in_domain_share)
    chosen = list(rng.sample(in_domain, min(want_in_domain, len(in_domain))))

    # Top up from everyone else. The exclusion set keeps a candidate from being
    # scored twice against the same posting, which would double-weight them.
    remaining = [cid for cid in by_id if cid not in set(chosen)]
    chosen += rng.sample(remaining, min(pool_size - len(chosen), len(remaining)))
    return chosen


def build_split(split: str, n_jobs: int, pool_size: int, seed: int,
                pools: dict[str, list[str]], prefix: str) -> tuple[list, list, list, list]:
    """Generate one split's postings and score every sampled pair.

    Candidates come from the matching Layer 1 split, so train pairs only ever
    involve train candidates. Combined with the separate job id prefixes, that
    keeps the two splits disjoint on both axes.
    """
    # Seeded per split so train and test draw different postings, and so a
    # rerun with the same seed reproduces the dataset exactly.
    rng = random.Random(f"{seed}-{split}")
    candidates = load_candidates(split)
    by_id = {candidate["id"]: candidate for candidate in candidates}

    # Domain index, built once rather than filtered per posting.
    by_domain: dict[str, list[str]] = collections.defaultdict(list)
    for candidate in candidates:
        by_domain[candidate["general_experience"]["domain"]].append(candidate["id"])

    jobs = [make_job(number, rng, pools, prefix) for number in range(1, n_jobs + 1)]
    pairs, flat, used = [], [], set()
    for job in jobs:
        for candidate_id in sample_applicants(job, by_id, by_domain, pool_size, rng):
            candidate = by_id[candidate_id]
            scores = rules.score_pair(candidate, job, rng)
            # pairs.jsonl stays minimal - ids plus labels. Features are derived
            # at training time from the full records, so a change to feature
            # engineering never means regenerating the dataset.
            pairs.append({"job_id": job["id"], "candidate_id": candidate_id, **scores})
            flat.append(flatten(scores, job, candidate))
            used.add(candidate_id)
    return jobs, pairs, flat, [by_id[cid] for cid in sorted(used)]


def write_split(split: str, jobs: list, pairs: list, flat: list, candidates: list):
    out = OUT_DIR / split
    out.mkdir(parents=True, exist_ok=True)

    # The candidates these pairs reference are copied in, so data_layer2 is
    # self-contained: training Layer 2 never has to read data_layer1.
    with (out / "candidates.jsonl").open("w", encoding="utf-8") as stream:
        for candidate in candidates:
            # Drop the candidate's Layer-1 intrinsic scores: they derive from
            # the same fields as these labels, so leaving them in invites an
            # accidental leak into Layer 2's features.
            trimmed = {field: value for field, value in candidate.items()
                       if field != "scores"}
            stream.write(json.dumps(trimmed, ensure_ascii=False) + "\n")
    with (out / "jobs.jsonl").open("w", encoding="utf-8") as stream:
        for job in jobs:
            stream.write(json.dumps(job, ensure_ascii=False) + "\n")
    with (out / "pairs.jsonl").open("w", encoding="utf-8") as stream:
        for pair in pairs:
            stream.write(json.dumps(pair, ensure_ascii=False) + "\n")
    with (out / "pairs.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0].keys()))
        writer.writeheader()
        writer.writerows(flat)
    print(f"{split}: {len(jobs)} jobs, {len(candidates)} candidates, {len(pairs)} pairs -> {out}")


def main():
    ap = argparse.ArgumentParser(description="Generate Layer 2 job postings and scored pairs.")
    # Postings are the scarce axis for a role-conditioned model, and they are
    # free to generate - unlike candidates, who are capped by the rendered CV
    # corpus. Pool depth is capped too: holding the in-domain share means a pool
    # can be no larger than the in-domain population supports (~60 on test,
    # which has ~33 candidates per domain). So pair count is bought with more
    # postings rather than deeper pools.
    ap.add_argument("--train-jobs", type=int, default=400)
    ap.add_argument("--test-jobs", type=int, default=170)
    ap.add_argument("--pool-size", type=int, default=250,
                    help="Upper bound on candidates scored per posting; the "
                         "in-domain share may reduce it (see sample_applicants)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Skill pools come from the TRAIN population only. Deriving them from all
    # candidates would let the test split shape what postings ask for - a
    # subtle leak that would flatter held-out results.
    pools = skill_pools(load_candidates("train"))

    # Distinct id prefixes make a leak between splits visible at a glance in any
    # output file, rather than something you have to cross-reference to catch.
    for split, n_jobs, prefix in (("train", args.train_jobs, "JOBTR"),
                                  ("test", args.test_jobs, "JOBTE")):
        jobs, pairs, flat, candidates = build_split(
            split, n_jobs, args.pool_size, args.seed, pools, prefix)
        write_split(split, jobs, pairs, flat, candidates)

        overall = [pair["overall_score"] for pair in pairs]
        print(f"  overall score: min {min(overall)} / mean {sum(overall) / len(overall):.1f} "
              f"/ max {max(overall)}")


if __name__ == "__main__":
    main()
