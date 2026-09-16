"""Turns the loaded frames into padded tensor batches for the scoring net.

A DataFrame is the right source but the wrong model input: three fields per row
are variable-length (skills held, skills wanted, employers), so the net needs
padded id tensors with masks rather than lists of dicts.

WHAT IS AND ISN'T DONE FOR THE MODEL
    Everything here is an encoding or a re-indexing of data already present.
    No coverage ratios, no similarity coefficients - those are the labelling
    formula, and feeding them in would hand over the answer.

    The one apparent exception is `required_skill_candidate_years`: the years
    held in each *required* skill, aligned to the requirement list. That is a
    lookup, not a score, and it matters - set intersection is nearly impossible
    to recover from mean-pooled embeddings (measured: stack MAE 9.5 without it,
    3.8 with).

SHARED ID SPACES
    Skills held and skills wanted share one vocabulary, as do the candidate's
    field of study and the job's preferred field. One embedding table per pair
    means "what they have" and "what is wanted" land in the same space and can
    simply be compared.

    train_loader, val_loader, test_loader, encoder = make_loaders()
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .dataset import SECTIONS, load

# Ordered, so the id doubles as "how advanced"; 0 stays free for padding/missing.
DEGREE_IDS = {"High School": 1, "Bachelor": 2, "Master": 3, "PhD": 4}
WEIGHT_KEYS = ["education", "relevant_experience", "stack_experience", "companies"]

# Targets are 0-100; the net trains on 0-1 and multiplies back for reporting.
TARGET_SCALE = 100.0


def text_or_none(value) -> str | None:
    """Missing values arrive as None or, via pandas, as NaN. Both mean absent."""
    return value if isinstance(value, str) else None

RAGGED = {
    "candidate_skill": ("candidate_skill_ids", ["candidate_skill_years"]),
    "required_skill": ("required_skill_ids", ["required_min_years", "required_weights",
                                              "required_skill_candidate_years"]),
    "company": ("company_tiers", ["company_years"]),
}


@dataclass
class Encoder:
    """Id maps and scalar statistics, fitted on the training rows only."""
    skills: dict[str, int] = field(default_factory=dict)
    domains: dict[str, int] = field(default_factory=dict)
    seniorities: dict[str, int] = field(default_factory=dict)
    fields: dict[str, int] = field(default_factory=dict)
    scalar_mean: torch.Tensor = field(default_factory=lambda: torch.zeros(2))
    scalar_std: torch.Tensor = field(default_factory=lambda: torch.ones(2))

    # 0 means padding, missing, or unseen-at-test, so every lookup lands on a
    # real embedding row instead of raising.
    def skill_id(self, name) -> int:
        return self.skills.get(name, 0)

    def field_id(self, name) -> int:
        return self.fields.get(text_or_none(name), 0)

    @property
    def n_skills(self) -> int:
        return len(self.skills) + 1

    @property
    def n_domains(self) -> int:
        return len(self.domains) + 1

    @property
    def n_seniorities(self) -> int:
        return len(self.seniorities) + 1

    @property
    def n_fields(self) -> int:
        return len(self.fields) + 1

    @property
    def n_degrees(self) -> int:
        return len(DEGREE_IDS) + 1

    @property
    def n_tiers(self) -> int:
        return 4        # tiers 1-3 plus padding


def fit_encoder(frame) -> Encoder:
    """Build the id maps and scalar statistics from training rows."""
    skills, domains, seniorities, fields = set(), set(), set(), set()
    for row in frame.itertuples():
        skills.update(skill["name"] for skill in row.candidate_skills)
        skills.update(requirement["name"] for requirement in row.job_required_skills)
        domains.add(row.candidate_general_experience["domain"])
        domains.add(row.job_domain)
        seniorities.add(row.job_seniority)
        fields.update(name for name in
                      (text_or_none(entry["field"]) for entry in row.candidate_education)
                      if name)
        preferred = text_or_none(row.job_preferred_field)
        if preferred:
            fields.add(preferred)

    def index(values) -> dict[str, int]:
        return {name: number for number, name in enumerate(sorted(values), start=1)}

    encoder = Encoder(skills=index(skills), domains=index(domains),
                      seniorities=index(seniorities), fields=index(fields))

    raw = torch.tensor([[float(row.candidate_general_experience["total_years"]),
                         float(row.job_required_experience_years)]
                        for row in frame.itertuples()])
    encoder.scalar_mean = raw.mean(dim=0)
    encoder.scalar_std = raw.std(dim=0).clamp(min=1e-6)
    return encoder


class PairDataset(Dataset):
    """One (candidate, job) pair per item.

    Every row is encoded once at construction rather than on each __getitem__,
    so repeated epochs are pure indexing instead of rebuilding dicts and
    tensors 50,000 times per pass.
    """

    def __init__(self, frame, encoder: Encoder):
        self.items = [self._encode(row, encoder) for row in frame.itertuples()]

    @staticmethod
    def _encode(row, encoder: Encoder) -> dict:
        general = row.candidate_general_experience
        education = row.candidate_education
        held_years = {skill["name"]: skill["years"] for skill in row.candidate_skills}
        required = row.job_required_skills
        top_degree = education[0] if education else {}

        def floats(values) -> torch.Tensor:
            return torch.tensor(values or [0.0], dtype=torch.float32)

        def ids(values) -> torch.Tensor:
            return torch.tensor(values or [0], dtype=torch.long)

        scalars = torch.tensor([float(general["total_years"]),
                                float(row.job_required_experience_years)])

        return {
            "candidate_skill_ids": ids([encoder.skill_id(skill["name"])
                                        for skill in row.candidate_skills]),
            "candidate_skill_years": floats([float(skill["years"])
                                             for skill in row.candidate_skills]),

            "required_skill_ids": ids([encoder.skill_id(item["name"]) for item in required]),
            "required_min_years": floats([float(item["min_years"]) for item in required]),
            "required_weights": floats([float(item["weight"]) for item in required]),
            "required_skill_candidate_years": floats(
                [float(held_years.get(item["name"], 0.0)) for item in required]),

            "company_tiers": ids([int(company["tier"]) for company in row.candidate_companies]),
            "company_years": floats([float(company["years"])
                                     for company in row.candidate_companies]),

            "categorical": torch.tensor([
                encoder.domains.get(general["domain"], 0),
                encoder.domains.get(row.job_domain, 0),
                encoder.seniorities.get(row.job_seniority, 0),
                DEGREE_IDS.get(text_or_none(top_degree.get("level")), 0),
                DEGREE_IDS.get(text_or_none(row.job_preferred_education), 0),
                encoder.field_id(top_degree.get("field")),      # 0 when no degree field
                encoder.field_id(row.job_preferred_field),      # 0 when the job states none
            ]),
            "scalars": (scalars - encoder.scalar_mean) / encoder.scalar_std,
            # Carried so the four heads can be combined into the overall score
            # that ranking is done on.
            "job_weights": torch.tensor([float(row.job_weights[key]) for key in WEIGHT_KEYS]),
            "target": torch.tensor([float(getattr(row, name))
                                    for name in SECTIONS]) / TARGET_SCALE,
            # Identifiers travel with the batch but never reach the model:
            # job_id groups a pool for ranking, candidate_id lets a split be
            # checked for the overlap that made validation optimistic.
            "job_id": row.job_id,
            "candidate_id": row.candidate_id,
        }

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        return self.items[index]


def _pad(sequences: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad a ragged batch and return it with a validity mask."""
    padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    lengths = torch.tensor([len(sequence) for sequence in sequences])
    mask = torch.arange(padded.size(1))[None, :] < lengths[:, None]
    return padded, mask


def collate(items: list[dict]) -> dict:
    """Pad the ragged fields, stack the rest."""
    batch = {}
    for name, (id_key, value_keys) in RAGGED.items():
        padded, mask = _pad([item[id_key] for item in items])
        batch[id_key] = padded
        batch[f"{name}_mask"] = mask
        for value_key in value_keys:
            batch[value_key], _ = _pad([item[value_key] for item in items])

    for key in ("categorical", "scalars", "job_weights", "target"):
        batch[key] = torch.stack([item[key] for item in items])

    for key in ("job_id", "candidate_id"):
        batch[key] = [item[key] for item in items]
    return batch


def make_loaders(batch_size: int = 256, val_jobs: int = 60,
                 val_candidate_share: float = 0.25, seed: int = 0,
                 num_workers: int = 0) -> tuple[DataLoader, DataLoader, DataLoader, Encoder]:
    """Train, validation and test loaders.

    Validation is carved out of train on **both** axes - postings and candidates
    - because that is how test is separated, and a validation set built any
    other way measures something test will not repeat.

    Splitting by posting alone is the tempting version and it is wrong here:
    every training candidate would still appear in validation under a different
    posting. `companies_score` reads the candidate alone, so the model can
    simply memorise those candidates; measured, that reported 1.82 MAE on a
    job-only validation split against 3.51 on test. Selecting an epoch on that
    number is selecting for memorisation.

    Pairs that straddle the two sides (a held-out posting with a training
    candidate, or the reverse) belong to neither and are dropped. That costs
    training rows, which is the price of a validation number that means
    something. The encoder is fitted on the training portion alone.
    """
    train_frame, test_frame = load("train"), load("test")

    picker = random.Random(seed)
    held_jobs = set(picker.sample(sorted(train_frame.job_id.unique()), val_jobs))
    all_candidates = sorted(train_frame.candidate_id.unique())
    held_candidates = set(picker.sample(all_candidates,
                                        round(len(all_candidates) * val_candidate_share)))

    is_held_job = train_frame.job_id.isin(held_jobs)
    is_held_candidate = train_frame.candidate_id.isin(held_candidates)
    val_frame = train_frame[is_held_job & is_held_candidate]
    fit_frame = train_frame[~is_held_job & ~is_held_candidate]

    encoder = fit_encoder(fit_frame)
    generator = torch.Generator().manual_seed(seed)      # reproducible shuffling

    def loader(frame, shuffle: bool) -> DataLoader:
        return DataLoader(PairDataset(frame, encoder), batch_size=batch_size, shuffle=shuffle,
                          collate_fn=collate, num_workers=num_workers,
                          generator=generator if shuffle else None)

    return loader(fit_frame, True), loader(val_frame, False), loader(test_frame, False), encoder


def pool_batches(split: str, encoder: Encoder):
    """One batch per job posting - the shape ranking metrics need."""
    frame = load(split)
    for job_id, pool in frame.groupby("job_id", sort=True):
        dataset = PairDataset(pool, encoder)
        yield job_id, collate([dataset[index] for index in range(len(dataset))])


if __name__ == "__main__":
    train_loader, val_loader, test_loader, encoder = make_loaders(batch_size=4)
    print(f"rows      train {len(train_loader.dataset)} | val {len(val_loader.dataset)} "
          f"| test {len(test_loader.dataset)}")
    print(f"vocab     {encoder.n_skills} skills | {encoder.n_domains} domains | "
          f"{encoder.n_fields} fields | {encoder.n_seniorities} seniorities | "
          f"{encoder.n_degrees} degrees | {encoder.n_tiers} tiers")
    for key, value in next(iter(train_loader)).items():
        shape = tuple(value.shape) if torch.is_tensor(value) else f"list[{len(value)}]"
        print(f"  {key:32s} {shape}")
