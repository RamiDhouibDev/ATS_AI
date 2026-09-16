"""Tests for the Layer 2 loader.

Reads the real generated data rather than fixtures: the loader's job is joining
three files, and the failure worth catching is those drifting out of step.
"""

import pytest

from layer2_scoring.code.data.dataset import SECTIONS, load, pools, targets


@pytest.fixture(scope="module")
def splits():
    return load("train"), load("test")


def test_split_shapes(splits):
    train, test = splits
    assert (len(train), train.job_id.nunique()) == (94848, 400)
    assert (len(test), test.job_id.nunique()) == (10158, 170)


def test_join_lost_no_rows(splits):
    """A bad join would show up as nulls in the merged columns."""
    for frame in splits:
        assert frame[["job_title", "candidate_name"]].notna().all().all()


def test_splits_share_no_jobs_or_candidates(splits):
    train, test = splits
    assert not set(train.job_id) & set(test.job_id)
    assert not set(train.candidate_id) & set(test.candidate_id)


def test_scores_are_in_range(splits):
    for frame in splits:
        section_scores = targets(frame)
        assert section_scores.min().min() >= 0
        assert section_scores.max().max() <= 100


def test_overall_is_consistent_with_its_parts(splits):
    """The overall must stay explainable from the four sections it combines."""
    train, _ = splits
    for row in train.head(500).itertuples():
        expected = sum(getattr(row, f"{key}_score") * row.job_weights[key]
                       for key in ("education", "relevant_experience",
                                   "stack_experience", "companies"))
        assert abs(expected - row.overall_score) <= 1      # rounding only


def test_pools_are_large_enough_to_rank(splits):
    _, test = splits
    for job_id, pool in pools(test):
        assert len(pool) >= 20
        assert pool.candidate_id.is_unique
        assert pool.overall_score.max() - pool.overall_score.min() > 20


def test_sections_are_the_four_targets():
    assert len(SECTIONS) == 4
    assert "overall_score" not in SECTIONS


def test_companies_score_is_one_number_per_candidate(splits):
    """It reads the candidate alone, so no posting may move it.

    It used to: the per-section jitter was drawn per pair, so one employment
    history came out 13 points apart across postings with nothing having
    changed. Unpredictable by construction, and misleading to anyone reading
    pairs.jsonl.
    """
    for frame in splits:
        spread = frame.groupby("candidate_id").companies_score.nunique()
        assert spread.max() == 1


def test_splits_are_drawn_from_the_same_distribution(splits):
    """A held-out set has to differ from train in its rows, not in its shape.

    The applicant sampler takes an in-domain share, and asking for a pool larger
    than the in-domain population silently filled the rest from everyone else.
    Test had 200 candidates and asked for 250, so its pools were simply
    everybody: 17% in-domain against train's 53%, and a stack score median of 5
    against 14. Nothing raised an error.
    """
    train, test = splits

    def in_domain_share(frame):
        candidate_domain = frame.candidate_general_experience.map(lambda block: block["domain"])
        return (candidate_domain == frame.job_domain).mean()

    assert abs(in_domain_share(train) - in_domain_share(test)) < 0.05
    for section in SECTIONS:
        assert abs(train[section].mean() - test[section].mean()) < 5


def test_pool_composition_holds_in_every_posting(splits):
    """The share is a guarantee, so it must hold per pool, not just on average."""
    for frame in splits:
        for _, pool in pools(frame):
            matching = (pool.candidate_general_experience.map(lambda block: block["domain"])
                        == pool.job_domain).mean()
            assert 0.45 <= matching <= 0.65
