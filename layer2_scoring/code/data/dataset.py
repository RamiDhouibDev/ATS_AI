"""Loads Layer 2 data.

Everything comes from data_layer2/<split>/. One row per scored (candidate, job)
pair, with the job and candidate columns merged in.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[3] / "data_layer2"

# The model's four targets, and all of them. There is no overall score in the
# data: it is a weighted sum of these four, combined at ranking time with
# whatever weights the posting or the UI supplies - arithmetic, not a label.
SECTIONS = ["education_score", "relevant_experience_score",
            "stack_experience_score", "companies_score"]


def load(split: str, data_dir: Path | None = None) -> pd.DataFrame:
    """One row per scored pair, with job and candidate fields merged in."""
    base = (data_dir or DATA_DIR) / split
    pairs = pd.read_json(base / "pairs.jsonl", lines=True)
    jobs = pd.read_json(base / "jobs.jsonl", lines=True).add_prefix("job_")
    candidates = pd.read_json(base / "candidates.jsonl", lines=True).add_prefix("candidate_")

    # validate= turns a broken join into an error rather than silently
    # duplicating or dropping rows.
    return (pairs
            .merge(jobs, on="job_id", validate="many_to_one")
            .merge(candidates, on="candidate_id", validate="many_to_one"))


def targets(frame: pd.DataFrame) -> pd.DataFrame:
    """The four section scores, in a fixed column order."""
    return frame[SECTIONS]


def pools(frame: pd.DataFrame):
    """Rows grouped by posting - ranking metrics are defined within one job."""
    return frame.groupby("job_id", sort=True)


if __name__ == "__main__":
    for split_name in ("train", "test"):
        data = load(split_name)
        print(f"{split_name}: {len(data)} pairs | {data.job_id.nunique()} jobs | "
              f"{data.candidate_id.nunique()} candidates | {data.shape[1]} columns")
