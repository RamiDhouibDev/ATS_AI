"""Pairs the labelled records with their rendered documents.

The labels in data/{train,test}.jsonl and the PDFs in data/cvs_pdf/<split>/ are
joined by candidate id, and the manifest supplies the difficulty metadata that
evaluation reports are broken down by.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from .schema import CVRecord, from_ground_truth

DATA_DIR = Path(__file__).resolve().parents[2] / "data_layer1"


@dataclass
class Sample:
    id: str
    split: str
    pdf_path: Path
    gold: CVRecord
    template: str
    difficulty: str
    skill_years_stated: bool
    contact_in_header_only: bool
    scanned: bool
    pages: int


def _load_manifest(split_dir: Path) -> dict[str, dict]:
    with (split_dir / "manifest.csv").open(encoding="utf-8") as f:
        return {row["id"]: row for row in csv.DictReader(f)}


def load_split(split: str, data_dir: Path | None = None, limit: int | None = None) -> list[Sample]:
    """Load one split, skipping any label whose PDF is missing.

    Each split owns its labels, manifest and documents under
    data_layer1/<split>/.
    """
    split_dir = (data_dir or DATA_DIR) / split
    manifest = _load_manifest(split_dir)

    samples: list[Sample] = []
    with (split_dir / f"{split}.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            meta = manifest.get(record["id"])
            if meta is None:
                continue
            pdf_path = split_dir / meta["pdf_path"].replace("\\", "/")
            if not pdf_path.exists():
                continue
            samples.append(Sample(
                id=record["id"],
                split=split,
                pdf_path=pdf_path,
                gold=from_ground_truth(record),
                template=meta["template"],
                difficulty=meta["difficulty"],
                skill_years_stated=meta["skill_years_stated"] == "True",
                contact_in_header_only=meta["contact_in_header_only"] == "True",
                scanned=meta["scanned"] == "True",
                pages=int(meta["pages"]),
            ))
            if limit and len(samples) >= limit:
                break
    return samples
