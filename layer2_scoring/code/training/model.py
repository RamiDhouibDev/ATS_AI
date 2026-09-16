"""The scoring network - PLACEHOLDER.

Nothing here is implemented. The file exists to fix the shape of the thing and
to record what it has to read, so that filling it in is a matter of writing
layers rather than rediscovering the data format.

The one part that is real is the loader underneath it: `make_loaders()` already
returns batches in exactly the form `forward` will be handed.

WHAT IT HAS TO BE
    One shared trunk, four heads - education, relevant experience, stack
    experience, companies. Not four separate models: the four sections share
    almost all of their input, and four models would each rediscover the same
    representation of a CV from a quarter of the gradient signal.

    The overall score is NOT a fifth head. It is a weighted sum of the four,
    using the posting's own weights, so that a ranking can always be explained
    by the numbers printed beside it.

WHAT forward() RECEIVES
    One dict from `torch_data.collate`. Ragged fields are padded and come with a
    boolean mask; anything pooled over them must respect that mask or padding
    dilutes the average and batch composition becomes a feature.

    candidate_skill_ids        (batch, skills)      + candidate_skill_mask
    candidate_skill_years      (batch, skills)
    required_skill_ids         (batch, required)    + required_skill_mask
    required_min_years         (batch, required)
    required_weights           (batch, required)
    required_skill_candidate_years
                               (batch, required)    years held in each *required*
                                                    skill, aligned to the list
    company_tiers              (batch, employers)   + company_mask
    company_years              (batch, employers)
    categorical                (batch, 7)           candidate domain, job domain,
                                                    seniority, candidate degree,
                                                    job degree, candidate field,
                                                    job preferred field
    scalars                    (batch, 2)           standardised: candidate total
                                                    years, years the posting asks
    job_weights                (batch, 4)           for the overall, not an input
    target                     (batch, 4)           labels, 0-1
    job_id, candidate_id       lists                identifiers, never inputs

WHAT forward() MUST RETURN
    (batch, 4) in **0-1**, ordered by `dataset.SECTIONS`. The loader divides the
    0-100 labels by TARGET_SCALE, so predictions live on that scale and only
    become percentages for reporting.

NOTES WORTH KEEPING WHEN THIS IS BUILT
    - Skills held and skills wanted share one embedding table, as do the
      candidate's field of study and the posting's preferred field. Sharing the
      table puts "has Python" and "wants Python" at the same point, so they can
      be compared instead of the network first learning that two vocabularies
      line up.
    - Mean-pooled sets cannot express their intersection. Whatever pools skills
      needs a path that keeps per-skill identity - an elementwise product of the
      two pooled vectors, or attention over the requirement list.
    - `required_skill_candidate_years` exists for that reason and is the one
      field that looks derived. It is a lookup, not a score.
    - No feature may be computed from the labelling rules in
      `data_layer2/gen/scoring_rules.py`. Coverage ratios and similarity
      coefficients are the answer, not the input.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..data.torch_data import Vocabulary


class SectionScorer(nn.Module):
    """PLACEHOLDER - shared trunk plus four section heads."""

    def __init__(self, vocabulary: Vocabulary):
        super().__init__()
        # The vocabulary carries the vocabulary sizes any embedding table needs:
        # n_skills, n_fields, n_domains, n_seniorities, n_degrees, n_tiers.
        # Index 0 is reserved in every one of them for padding, missing, and
        # unseen-at-test, so every table needs padding_idx=0.
        self.vocabulary = vocabulary

        # TODO: embedding tables, the branches that pool them, the shared trunk,
        # and four heads.

    def forward(self, batch: dict) -> torch.Tensor:
        """PLACEHOLDER - must return (batch, 4) in 0-1."""
        raise NotImplementedError("SectionScorer.forward is not built yet")


def overall_score(sections: torch.Tensor, job_weights: torch.Tensor) -> torch.Tensor:
    """PLACEHOLDER - the four sections combined with the posting's own weights.

    Deliberately never learned. Both arguments are (batch, 4) in the same
    section order, so this is a row dot product.
    """
    raise NotImplementedError("overall_score is not built yet")


if __name__ == "__main__":
    # No model to run. This just shows what a batch looks like, which is the
    # thing `forward` has to consume.
    from ..data.torch_data import make_loaders

    train_loader, _, _, vocabulary = make_loaders(batch_size=4)
    print(f"vocabularies   {vocabulary.n_skills} skills | {vocabulary.n_fields} fields | "
          f"{vocabulary.n_domains} domains | {vocabulary.n_seniorities} seniorities | "
          f"{vocabulary.n_degrees} degrees | {vocabulary.n_tiers} tiers")
    print("one batch:")
    for key, value in next(iter(train_loader)).items():
        shape = tuple(value.shape) if torch.is_tensor(value) else f"list[{len(value)}]"
        print(f"  {key:32s} {shape}")
