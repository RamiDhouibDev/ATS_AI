"""Canonical CV structure shared by every extraction path.

Both the rule-based parser and (later) the LLM extractor must produce a
`CVRecord`; the ground-truth labels are loaded into the same shape so the two
can be compared field by field.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class Education:
    level: str | None = None
    field_of_study: str | None = None
    institution: str | None = None
    start_year: int | None = None
    end_year: int | None = None


@dataclass
class Skill:
    name: str
    years: float | None = None


@dataclass
class Company:
    name: str
    title: str | None = None
    start_date: str | None = None   # "YYYY-MM"
    end_date: str | None = None     # None means the role is current
    tier: int | None = None


@dataclass
class CVRecord:
    name: str | None = None
    education: list[Education] = field(default_factory=list)
    domain: str | None = None
    total_years: float | None = None
    skills: list[Skill] = field(default_factory=list)
    companies: list[Company] = field(default_factory=list)

    @property
    def highest_education(self) -> Education | None:
        return self.education[0] if self.education else None

    def skill_names(self) -> set[str]:
        return {s.name for s in self.skills}

    def company_names(self) -> set[str]:
        return {c.name for c in self.companies}

    def to_dict(self) -> dict:
        return asdict(self)


# Degree levels ordered weakest to strongest, so "highest" is well defined.
DEGREE_ORDER = ["High School", "Bachelor", "Master", "PhD"]


def degree_rank(level: str | None) -> int:
    return DEGREE_ORDER.index(level) if level in DEGREE_ORDER else -1


def from_ground_truth(record: dict) -> CVRecord:
    """Load a labelled record from data/{train,test}.jsonl into a CVRecord."""
    education = [
        Education(
            level=e.get("level"),
            field_of_study=e.get("field"),
            institution=e.get("institution"),
            start_year=e.get("start_year"),
            end_year=e.get("end_year"),
        )
        for e in record.get("education", [])
    ]
    education.sort(key=lambda e: degree_rank(e.level), reverse=True)

    general = record.get("general_experience", {})
    return CVRecord(
        name=record.get("name"),
        education=education,
        domain=general.get("domain"),
        total_years=general.get("total_years"),
        skills=[Skill(name=s["name"], years=s.get("years")) for s in record.get("skills", [])],
        companies=[
            Company(
                name=c["name"],
                title=c.get("title"),
                start_date=c.get("start_date"),
                end_date=c.get("end_date"),
                tier=c.get("tier"),
            )
            for c in record.get("companies", [])
        ],
    )


@dataclass
class LayoutSignals:
    """What the document looks like, independent of what was read off it."""
    char_count: int = 0
    n_pages: int = 1
    max_columns: int = 1
    has_images: bool = False
    has_header_text: bool = False
    has_footer_text: bool = False

    @property
    def is_multi_column(self) -> bool:
        return self.max_columns > 1

    @property
    def is_image_only(self) -> bool:
        return self.char_count == 0


@dataclass
class ExtractionResult:
    record: CVRecord
    signals: LayoutSignals
    confidence: float = 0.0
    mode: str = "ats"           # which path produced this
    needs_llm: bool = False     # confidence below threshold -> escalate
    sections_found: list[str] = field(default_factory=list)
