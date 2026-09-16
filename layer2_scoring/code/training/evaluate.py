"""Held-out evaluation - PLACEHOLDER.

Nothing is evaluated here. The loading is real: `pool_batches` already yields
one batch per posting, which is the shape the ranking metrics need.

WHY POOLS AND NOT PLAIN BATCHES
    The product does not return scores, it returns twenty people. Candidates are
    only ever compared against other applicants to the *same* posting, never
    across postings, so every ranking metric is computed within one pool.
    `pool_batches(split, vocabulary)` yields exactly that: (job_id, batch).

WHY PER-SECTION MAE IS NOT ENOUGH ON ITS OWN
    It is the number to watch while building, but alone it flatters.
    `stack_experience_score` has a median near 15, so a model that predicts
    "low" everywhere posts a respectable MAE while being useless for ranking.

READ THIS ONCE
    Every run here reads the test split. Tuning anything in response to what it
    prints turns the held-out set into a second validation set, and the next
    number it gives is worth nothing. Tune on validation in `train.py`; come
    here when you are finished.

NOTES WORTH KEEPING WHEN THIS IS BUILT
    - Accumulate absolute error over rows, not over batches. Pools run from ~50
      to ~250 candidates, so averaging per-pool averages weights a small posting
      the same as a large one.
    - Rank on the overall score, which is the four sections combined with *this
      posting's* weights - `batch["job_weights"]`. Two postings can rank the
      same two candidates differently, which is the whole point of Layer 2.
    - Test pools run 50-76 candidates, so a top-k request must be clamped to the
      pool size rather than assumed smaller.
    - Report both: top-20 overlap answers "how many of the right twenty did we
      surface", NDCG@20 answers "and how near the top did we put them".
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..data.dataset import SECTIONS
from ..data.torch_data import TARGET_SCALE, pool_batches

TOP_K = 20      # the product returns twenty candidates


def top_k_overlap(predicted, actual, top_k: int = TOP_K) -> float:
    """PLACEHOLDER - share of the true top-k that the predicted top-k recovers."""
    raise NotImplementedError("top_k_overlap is not built yet")


def ndcg_at_k(predicted, actual, top_k: int = TOP_K) -> float:
    """PLACEHOLDER - ranking quality weighted by position.

    Overlap treats the 1st and 20th slots as equal; this must not. Relevance is
    the true overall score.
    """
    raise NotImplementedError("ndcg_at_k is not built yet")


def evaluate(checkpoint_path: Path, split: str = "test", top_k: int = TOP_K):
    """PLACEHOLDER - score every posting in `split`, report error and ranking."""
    # TODO: payload = torch.load(checkpoint_path, weights_only=False)
    #       vocabulary = payload["vocabulary"]        # the fitted one, never a fresh fit
    #       model = SectionScorer(vocabulary); model.load_state_dict(payload["state"])
    #
    # for job_id, batch in pool_batches(split, vocabulary):
    #     ...                                   # predict, accumulate MAE
    #     ...                                   # rank the pool both ways, score it
    raise NotImplementedError("evaluation is not built yet")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Layer 2 scorer.")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path(__file__).resolve().parents[2] / "checkpoints" / "scorer.pt")
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--top-k", type=int, default=TOP_K)
    arguments = parser.parse_args()

    evaluate(arguments.checkpoint, split=arguments.split, top_k=arguments.top_k)


if __name__ == "__main__":
    # Nothing to evaluate yet. This only shows the pools the metrics will read:
    # the vocabulary normally comes from the checkpoint, so this refits one purely
    # to demonstrate the shape.
    from ..data.torch_data import make_loaders

    _, _, _, vocabulary = make_loaders(batch_size=256)
    sizes = [batch["target"].shape[0] for _, batch in pool_batches("test", vocabulary)]
    print(f"test: {len(sizes)} postings | {sum(sizes):,} pairs | "
          f"pools {min(sizes)}-{max(sizes)} candidates")
    print(f"sections  {SECTIONS}  (labels 0-1, x{TARGET_SCALE:.0f} to report)")
    print(f"top-k     {TOP_K}, clamped to the pool size")
