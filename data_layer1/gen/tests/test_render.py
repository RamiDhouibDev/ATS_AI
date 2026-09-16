"""Guards the reproducibility of the rendered corpus.

Every CV must render to the same bytes on every run, so that regenerating the
dataset never silently reshuffles what Layer 1 is measured against. The plain
layouts get this from reportlab's `invariant=1`; the scanned ones are the
fragile case, because two separate sources of run-to-run drift live in that
path - see `render_cvs.scan_grain`.
"""

import random
import shutil
from pathlib import Path

import pymupdf as fitz
import pytest

from render_cvs import scan_degrade

SOURCE = Path(__file__).resolve().parents[2] / "train" / "cvs_pdf" / "TRAIN00001.pdf"


@pytest.fixture
def degrade(tmp_path):
    """Degrade a copy of one real CV, always at the same path.

    The path matters: Pillow writes the file's own name into the PDF's /Title,
    so degrading to two different filenames differs by those bytes alone and
    would make this test lie.
    """
    target = tmp_path / "scan.pdf"

    def run(seed):
        shutil.copy(SOURCE, target)
        scan_degrade(target, random.Random(seed))
        return target.read_bytes()

    return run


def test_same_seed_renders_the_same_bytes(degrade):
    assert degrade("seed-a") == degrade("seed-a")


def test_different_seeds_still_vary(degrade):
    """Reproducibility must not have come from pinning every scan to one image."""
    assert degrade("seed-a") != degrade("seed-b")


def test_no_timestamp_is_written(degrade):
    """A clock stamp reproduces within a second and breaks a minute later."""
    assert b"/CreationDate" not in degrade("seed-a")
    assert b"/ModDate" not in degrade("seed-a")


def test_scan_leaves_no_text_layer(degrade, tmp_path):
    """The point of the scanned tier: no parser can read it, forcing the fallback."""
    (tmp_path / "check.pdf").write_bytes(degrade("seed-a"))
    with fitz.open(str(tmp_path / "check.pdf")) as document:
        assert not "".join(page.get_text() for page in document).strip()
