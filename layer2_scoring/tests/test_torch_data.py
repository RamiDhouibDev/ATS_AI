"""Tests for the tensor batching layer.

The loader's failures are quiet ones: an id past the end of an embedding table,
a mask that disagrees with the padding, validation rows sharing a posting with
train. Each of those trains happily and lies about the result, so they are
checked here against the real generated data.
"""

import pytest
import torch

from layer2_scoring.code.data.dataset import SECTIONS, load
from layer2_scoring.code.data.torch_data import (
    DEGREE_IDS,
    RAGGED,
    TARGET_SCALE,
    Vocabulary,
    PairDataset,
    collate,
    fit_vocabulary,
    make_loaders,
    pool_batches,
    text_or_none,
)


@pytest.fixture(scope="module")
def loaders():
    return make_loaders(batch_size=64, val_jobs=60, seed=0)


@pytest.fixture(scope="module")
def vocabulary(loaders):
    return loaders[3]


def batches(loader, count=3):
    return [batch for _, batch in zip(range(count), loader)]


def test_text_or_none_treats_nan_as_absent():
    """pandas turns a JSON null into NaN, which sorts and compares like a float."""
    assert text_or_none("Computer Science") == "Computer Science"
    assert text_or_none(float("nan")) is None
    assert text_or_none(None) is None


def test_validation_is_split_on_both_axes(loaders):
    """Validation must be separated from train exactly the way test is.

    Splitting by posting alone leaves every training candidate present in
    validation under another posting, and `companies_score` reads the candidate
    alone - so the model memorises them and validation reports a number test
    cannot repeat (measured: 1.82 against 3.51).
    """
    train_loader, val_loader, test_loader, _ = loaders
    jobs, candidates = {}, {}
    for name, loader in (("train", train_loader), ("val", val_loader), ("test", test_loader)):
        jobs[name] = {job for batch in loader for job in batch["job_id"]}
        candidates[name] = {item["candidate_id"] for item in loader.dataset.items}

    assert len(jobs["val"]) == 60
    for split in ("val", "test"):
        assert not jobs["train"] & jobs[split]
        assert not candidates["train"] & candidates[split]
    assert not jobs["val"] & jobs["test"]


def test_split_sizes(loaders):
    """Straddling pairs belong to neither side, so the two do not sum to train."""
    train_loader, val_loader, test_loader, _ = loaders
    assert len(train_loader.dataset) + len(val_loader.dataset) < len(load("train"))
    assert len(val_loader.dataset) > 1000              # still enough to select an epoch on
    assert len(test_loader.dataset) == len(load("test"))


def test_vocabulary_is_fitted_on_training_rows_only(loaders):
    """Refitting on the held-out rows must not be able to grow the vocabulary."""
    _, val_loader, _, fitted = loaders
    train_frame = load("train")
    val_jobs = {job for batch in val_loader for job in batch["job_id"]}
    val_candidates = {item["candidate_id"] for item in val_loader.dataset.items}
    refitted = fit_vocabulary(train_frame[train_frame.job_id.isin(val_jobs)
                                       & train_frame.candidate_id.isin(val_candidates)])

    assert set(refitted.skills) <= set(fitted.skills)
    assert set(refitted.fields) <= set(fitted.fields)


def test_unknown_names_fall_back_to_the_padding_row():
    empty = Vocabulary()
    assert empty.skill_id("Fortran") == 0
    assert empty.field_id("Basket Weaving") == 0
    assert empty.field_id(float("nan")) == 0


def test_ids_stay_inside_their_embedding_tables(loaders):
    train_loader, val_loader, test_loader, fitted = loaders
    bounds = [
        ("candidate_skill_ids", fitted.n_skills),
        ("required_skill_ids", fitted.n_skills),
        ("company_tiers", fitted.n_tiers),
    ]
    limits = torch.tensor([fitted.n_domains, fitted.n_domains, fitted.n_seniorities,
                           fitted.n_degrees, fitted.n_degrees,
                           fitted.n_fields, fitted.n_fields])

    for loader in (train_loader, val_loader, test_loader):
        for batch in batches(loader):
            for key, limit in bounds:
                assert batch[key].min() >= 0 and batch[key].max() < limit
            assert (batch["categorical"] >= 0).all()
            assert (batch["categorical"] < limits).all()


def test_masks_agree_with_the_padding(loaders):
    """Padded slots must be zero and outside the mask, or pooling averages junk."""
    train_loader, *_ = loaders
    for batch in batches(train_loader):
        for group, (id_key, value_keys) in RAGGED.items():
            mask = batch[f"{group}_mask"]
            assert mask.shape == batch[id_key].shape
            assert mask.any(dim=1).all()            # no row is entirely padding
            assert (batch[id_key][~mask] == 0).all()
            for value_key in value_keys:
                assert (batch[value_key][~mask] == 0).all()


def test_targets_are_normalised_and_recover_the_labels(vocabulary):
    test_frame = load("test").head(200)
    dataset = PairDataset(test_frame, vocabulary)
    batch = collate([dataset[index] for index in range(len(dataset))])

    assert batch["target"].min() >= 0 and batch["target"].max() <= 1
    recovered = batch["target"] * TARGET_SCALE
    expected = torch.tensor(test_frame[SECTIONS].to_numpy(), dtype=torch.float32)
    assert torch.allclose(recovered, expected, atol=1e-4)


def test_required_skill_years_is_a_lookup_not_a_score(vocabulary):
    """Aligned to the requirement list: the candidate's years, or 0 if not held."""
    frame = load("test").head(100)
    dataset = PairDataset(frame, vocabulary)

    for position, row in enumerate(frame.itertuples()):
        held = {skill["name"]: skill["years"] for skill in row.candidate_skills}
        expected = torch.tensor([float(held.get(item["name"], 0.0))
                                 for item in row.job_required_skills])
        assert torch.allclose(dataset[position]["required_skill_candidate_years"], expected)


def test_degree_ids_are_ordered_by_level():
    levels = ["High School", "Bachelor", "Master", "PhD"]
    assert [DEGREE_IDS[level] for level in levels] == sorted(DEGREE_IDS.values())
    assert 0 not in DEGREE_IDS.values()      # 0 is reserved for "no degree stated"


def test_pool_batches_yield_one_posting_each(vocabulary):
    test_frame = load("test")
    sizes = test_frame.groupby("job_id").size()

    seen = []
    for job_id, batch in pool_batches("test", vocabulary):
        seen.append(job_id)
        assert len(batch["job_id"]) == sizes[job_id]
        assert set(batch["job_id"]) == {job_id}
    assert seen == sorted(sizes.index)


def test_loaders_are_reproducible():
    first = make_loaders(batch_size=64, val_jobs=5, seed=7)[1]
    second = make_loaders(batch_size=64, val_jobs=5, seed=7)[1]
    assert [batch["job_id"] for batch in first] == [batch["job_id"] for batch in second]
