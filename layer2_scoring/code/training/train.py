"""Training loop - PLACEHOLDER.

Nothing is trained here. The loading is real and finished; everything below it
is a shape to fill in.

THE SPLIT IS ALREADY DECIDED, AND IT MATTERS
    `make_loaders()` returns train, validation and test. Validation is carved
    out of train on **both** axes - postings and candidates - because that is
    how test is separated, and a validation set built any other way measures
    something test will not repeat.

    Splitting by posting alone is the tempting version and it is wrong here:
    every training candidate would still turn up in validation under a different
    posting, and `companies_score` reads the candidate alone, so the model can
    simply memorise them. Measured, before this was fixed: 1.82 MAE on a
    posting-only validation split against 3.51 on test.

WHAT THIS FILE MUST NEVER DO
    Touch the test loader. It is returned so that one function owns the split,
    and it goes straight back unread. Select on validation; `evaluate.py` reads
    test, once, at the end.

NOTES WORTH KEEPING WHEN THIS IS BUILT
    - Huber (SmoothL1) rather than MSE. The labels carry gaussian jitter and the
      section curves saturate at both ends, so a few pairs sit far from anything
      predictable and squared error would let them steer the update.
    - Keep the best epoch by validation, not the last one.
    - Save the encoder alongside the weights. Its id maps are fitted on the
      training rows, so weights loaded against a refitted encoder look up
      different embedding rows - a bug that produces plausible nonsense rather
      than an error.
    - Targets are 0-1. Multiply by TARGET_SCALE only to print.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..data.dataset import SECTIONS
from ..data.torch_data import TARGET_SCALE, make_loaders

CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "checkpoints"


def train(epochs: int = 30, batch_size: int = 256, learning_rate: float = 1e-3,
          seed: int = 0, checkpoint: Path | None = None):
    """PLACEHOLDER - fit the scorer and keep the epoch that validated best."""
    # --- real: the data ------------------------------------------------------
    train_loader, val_loader, test_loader, encoder = make_loaders(
        batch_size=batch_size, seed=seed)
    print(f"rows   train {len(train_loader.dataset):,} | "
          f"val {len(val_loader.dataset):,} | test {len(test_loader.dataset):,} (untouched)")
    print(f"batches per epoch: {len(train_loader)}")

    # --- placeholder: everything else ---------------------------------------
    # TODO: model = SectionScorer(encoder)
    # TODO: optimiser, loss function
    #
    # for epoch in range(1, epochs + 1):
    #     for batch in train_loader:            # batches are ready as-is
    #         ...                               # forward, loss, backward, step
    #     ...                                   # validation pass, keep best epoch
    #
    # TODO: save {"state": ..., "encoder": encoder, "sections": SECTIONS}
    #       to `checkpoint or CHECKPOINT_DIR / "scorer.pt"`
    raise NotImplementedError("training loop is not built yet")


def section_mae(model, loader):
    """PLACEHOLDER - mean absolute error per section, on the 0-100 scale.

    Predictions and targets are both 0-1, so multiply the mean absolute
    difference by TARGET_SCALE before reporting it.
    """
    raise NotImplementedError("section_mae is not built yet")


def main():
    parser = argparse.ArgumentParser(description="Train the Layer 2 section scorer.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=None)
    arguments = parser.parse_args()

    train(epochs=arguments.epochs, batch_size=arguments.batch_size,
          learning_rate=arguments.learning_rate, seed=arguments.seed,
          checkpoint=arguments.checkpoint)


if __name__ == "__main__":
    # Nothing to train yet, so this only shows what the loop will be handed.
    train_loader, val_loader, test_loader, encoder = make_loaders(batch_size=256)
    print(f"train {len(train_loader.dataset):,} rows in {len(train_loader)} batches")
    print(f"val   {len(val_loader.dataset):,} rows in {len(val_loader)} batches")
    print(f"test  {len(test_loader.dataset):,} rows - not read during training")
    print(f"targets   {SECTIONS}, scaled to 0-1 by /{TARGET_SCALE:.0f}")
