"""Tests for the Layer 2 data loader.

These read the real generated corpus rather than fixtures: the loader joins
three files, and the failure worth catching is those drifting out of step.
"""

import pytest

from layer2_scoring.code.dataset import (
    SECTIONS, Candidate, Example, describe, load_candidates, load_examples,
    load_jobs, load_ranking_pools,
)


@pytest.fixture(scope="module")
def train_examples():
    return load_examples("train", limit=2000)


class TestJobs:
    def test_jobs_load_for_both_splits(self):
        assert len(load_jobs("train")) == 200
        assert len(load_jobs("test")) == 50

    def test_ids_are_prefixed_per_split(self):
        """Distinct prefixes make a train/test leak visible at a glance."""
        assert all(j.startswith("JOBTR") for j in load_jobs("train"))
        assert all(j.startswith("JOBTE") for j in load_jobs("test"))

    def test_weights_sum_to_one(self):
        for job in load_jobs("train").values():
            assert abs(sum(job.weights.values()) - 1.0) < 0.01

    def test_every_posting_asks_for_some_stack(self):
        for job in load_jobs("train").values():
            assert 3 <= len(job.required_skills) <= 7
            assert all(s.weight > 0 for s in job.required_skills)

    def test_open_ended_field_is_none_not_blank(self):
        fields = {job.preferred_field for job in load_jobs("train").values()}
        assert None in fields                       # some postings leave it open
        assert "" not in fields                     # ...represented as None, never ""


class TestCandidates:
    def test_candidates_load_from_layer2_only(self):
        """data_layer2 is self-contained: no read into data_layer1."""
        candidates = load_candidates("train")
        assert len(candidates) == 800
        assert all(isinstance(c, Candidate) for c in candidates.values())

    def test_splits_do_not_share_candidates(self):
        assert not set(load_candidates("train")) & set(load_candidates("test"))


class TestExamples:
    def test_pairs_join_to_real_candidates_and_jobs(self, train_examples):
        assert len(train_examples) == 2000
        assert all(isinstance(e, Example) for e in train_examples)
        assert all(e.candidate is not None and e.job is not None for e in train_examples)

    def test_no_orphan_pairs_in_either_split(self, capsys):
        """A pair pointing at a missing id means the two data folders drifted."""
        load_examples("train", limit=5000)
        assert "WARNING" not in capsys.readouterr().out

    def test_target_vector_is_the_four_sections_in_order(self, train_examples):
        example = train_examples[0]
        assert example.target_vector == [example.labels[name] for name in SECTIONS]
        assert len(example.target_vector) == 4

    def test_scores_are_within_range(self, train_examples):
        for example in train_examples:
            assert all(0 <= v <= 100 for v in example.target_vector)
            assert 0 <= example.overall <= 100

    def test_overall_is_consistent_with_its_parts(self, train_examples):
        """The overall must stay explainable from the four sections it combines."""
        for example in train_examples[:200]:
            weights = example.job.weights
            expected = sum(example.labels[f"{k}_score"] * weights[k]
                           for k in ("education", "relevant_experience",
                                     "stack_experience", "companies"))
            assert abs(expected - example.overall) <= 1     # rounding only


class TestRankingPools:
    def test_pools_are_grouped_by_posting(self):
        pools = load_ranking_pools("test")
        assert len(pools) == 50
        assert all(all(e.job.id == job.id for e in examples) for job, examples in pools)

    def test_pool_is_large_enough_to_pick_a_top_20(self):
        for job, examples in load_ranking_pools("test"):
            assert len(examples) >= 20

    def test_candidates_are_unique_within_a_pool(self):
        """A candidate scored twice for one job would be double-weighted."""
        for job, examples in load_ranking_pools("test", limit_jobs=10):
            ids = [e.candidate.id for e in examples]
            assert len(ids) == len(set(ids))

    def test_ranking_has_something_to_rank(self):
        """A flat pool would make top-20 selection meaningless."""
        for job, examples in load_ranking_pools("test", limit_jobs=10):
            scores = [e.overall for e in examples]
            assert max(scores) - min(scores) > 20


class TestDescribe:
    @pytest.mark.parametrize("split,jobs,pairs", [("train", 200, 50000), ("test", 50, 10000)])
    def test_split_shape(self, split, jobs, pairs):
        summary = describe(split)
        assert summary["jobs"] == jobs
        assert summary["pairs"] == pairs
        assert summary["overall_min"] < summary["overall_mean"] < summary["overall_max"]
