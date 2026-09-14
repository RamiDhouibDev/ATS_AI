"""Rule-based CV parser - the cheap path that runs on every document.

No training in the gradient sense: this is regex, section-header detection and
a vocabulary lookup. What *is* fitted (in tune.py) is the confidence threshold
below which a document gets escalated to the LLM extractor.
"""

from __future__ import annotations

import re

from . import vocab
from .pdf_text import Document
from .schema import Company, CVRecord, Education, ExtractionResult, Skill

MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}

TITLE_KEYWORDS = re.compile(
    r"\b(engineer|developer|scientist|manager|analyst|designer|researcher|architect|"
    r"consultant|specialist|administrator|sdet|programmer|lead|director)\b", re.I)

BULLET_START = re.compile(r"^\s*[•»–—▪*\-●]\s+")
YEAR = r"(?:19|20)\d{2}"
MONTH_NAME = r"[A-Z][a-z]{2,8}"

# One pattern per date style the corpus emits, plus the open-ended "Present".
DATE_TOKEN = rf"(?:{MONTH_NAME}\s+{YEAR}|\d{{1,2}}/{YEAR}|{YEAR}-\d{{2}}|{MONTH_NAME}\s+'\d{{2}}|{YEAR})"
DATE_RANGE = re.compile(
    rf"({DATE_TOKEN})\s*(?:[–—-]|to)\s*({DATE_TOKEN}|Present|Current|Now)", re.I)
YEAR_RANGE = re.compile(rf"({YEAR})\s*(?:[–—-]|to)\s*({YEAR})")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.\w+")
NAME_LINE = re.compile(r"^[A-Z][A-Za-z'\-]+(?:\s+[A-Z]\.)?(?:\s+[A-Z][A-Za-z'\-]+){1,2}$")

SEPARATORS = re.compile(r"\s*[|,—–]\s*|\s+-\s+")


def _parse_date_token(token: str) -> str | None:
    """Normalise any supported date spelling to 'YYYY-MM'."""
    token = token.strip()
    if re.fullmatch(YEAR, token):
        return f"{token}-01"
    m = re.fullmatch(rf"({YEAR})-(\d{{2}})", token)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    m = re.fullmatch(rf"(\d{{1,2}})/({YEAR})", token)
    if m:
        return f"{m.group(2)}-{int(m.group(1)):02d}"
    m = re.fullmatch(rf"({MONTH_NAME})\s+'(\d{{2}})", token)
    if m:
        month = MONTHS.get(m.group(1)[:3].lower())
        return f"20{m.group(2)}-{month:02d}" if month else None
    m = re.fullmatch(rf"({MONTH_NAME})\s+({YEAR})", token)
    if m:
        month = MONTHS.get(m.group(1)[:3].lower())
        return f"{m.group(2)}-{month:02d}" if month else None
    return None


def find_date_range(text: str) -> tuple[str | None, str | None] | None:
    """Returns (start, end) as 'YYYY-MM'; end is None for a current role."""
    match = DATE_RANGE.search(text)
    if not match:
        return None
    start = _parse_date_token(match.group(1))
    raw_end = match.group(2)
    if raw_end.lower() in ("present", "current", "now"):
        return start, None
    return start, _parse_date_token(raw_end)


def split_sections(lines: list[str]) -> dict[str, list[str]]:
    """Bucket body lines under the section heading that precedes them."""
    sections: dict[str, list[str]] = {}
    current = "_preamble"
    sections[current] = []
    for line in lines:
        heading = vocab.heading_for(line)
        if heading:
            current = heading
            sections.setdefault(current, [])
            continue
        prefix_heading, remainder = vocab.split_heading_prefix(line)
        if prefix_heading:
            current = prefix_heading
            sections.setdefault(current, [])
            if remainder:
                sections[current].append(remainder)
            continue
        sections.setdefault(current, []).append(line)
    return sections


def _name_like(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or EMAIL.search(stripped) or any(ch.isdigit() for ch in stripped):
        return None
    if vocab.heading_for(stripped) or vocab.canonical_skill(stripped):
        return None
    if NAME_LINE.match(stripped):
        return stripped.title() if stripped.isupper() else stripped
    # Templates that upper-case the name lose their capitalisation pattern.
    if stripped.isupper() and 1 < len(stripped.split()) <= 4 and stripped.replace(" ", "").isalpha():
        return stripped.title()
    return None


def parse_name(document: Document, sections: dict[str, list[str]]) -> str | None:
    # A header holds "Name | email | phone" when contact details live up there.
    for line in document.header_text.splitlines():
        candidate = _name_like(SEPARATORS.split(line.strip())[0])
        if candidate:
            return candidate

    for line in sections.get("_preamble", [])[:6]:
        candidate = _name_like(line)
        if candidate:
            return candidate

    # Sidebar layouts emit the sidebar first, so the name lands after a heading
    # rather than in the preamble. Form layouts bury it behind a label on the
    # same line ("PERSONAL INFORMATION Amanda May"), so try that split too.
    for line in document.body_text.splitlines()[:30]:
        candidate = _name_like(line)
        if candidate:
            return candidate
        _, remainder = vocab.split_heading_prefix(line)
        if remainder and remainder != line.strip():
            candidate = _name_like(remainder)
            if candidate:
                return candidate
    return None


def parse_education(lines: list[str]) -> list[Education]:
    entries: list[Education] = []
    for index, line in enumerate(lines):
        level = next((name for name, pattern in vocab.DEGREE_PATTERNS if pattern.search(line)), None)
        if not level:
            continue

        field_of_study = next((f for f in vocab.EDUCATION_FIELDS if f.lower() in line.lower()), None)
        entry = Education(level=level, field_of_study=field_of_study)

        # Institution and dates usually sit on the degree line or the one below.
        for candidate in (line, *lines[index + 1:index + 3]):
            years = YEAR_RANGE.search(candidate)
            if years and entry.start_year is None:
                entry.start_year, entry.end_year = int(years.group(1)), int(years.group(2))
            elif entry.end_year is None:
                lone = re.search(YEAR, candidate)
                if lone and candidate is not line:
                    entry.end_year = int(lone.group(0))
            if entry.institution is None and re.search(
                    r"university|college|institute|school|academy", candidate, re.I):
                entry.institution = SEPARATORS.split(candidate.strip())[0].strip()
        entries.append(entry)

    entries.sort(key=lambda e: vocab.DEGREE_PATTERNS and _degree_rank(e.level), reverse=True)
    return entries


def _degree_rank(level: str | None) -> int:
    order = ["High School", "Bachelor", "Master", "PhD"]
    return order.index(level) if level in order else -1


def parse_skills(lines: list[str]) -> list[Skill]:
    """Canonicalise every recognisable skill token in the skills section."""
    found: dict[str, float | None] = {}
    for line in lines:
        if vocab.NOISE_MARKERS.search(line):
            continue
        # Strip a category prefix such as "Languages: Python, SQL".
        body = line.split(":", 1)[1] if ":" in line and len(line.split(":", 1)[0]) < 25 else line
        # Dashes are split only when spaced, so hyphenated names like
        # "Scikit-learn" survive while "Python - Expert" is separated.
        for chunk in re.split(r"[,|•\n]|\s+[–—-]\s+|\s{2,}", body):
            chunk = chunk.strip()
            if not chunk:
                continue
            years_match = re.search(r"\((\d+(?:\.\d+)?)\s*(?:yrs?|years?|yr exp)\)|"
                                    r"\b(\d+(?:\.\d+)?)\s*(?:yrs?|years?)\b", chunk, re.I)
            years = None
            if years_match:
                years = float(years_match.group(1) or years_match.group(2))
                chunk = chunk[:years_match.start()].strip()
            name = vocab.canonical_skill(chunk)
            if name and (name not in found or found[name] is None):
                found[name] = years
    return [Skill(name=n, years=y) for n, y in found.items()]


def _is_date_only(line: str) -> bool:
    """A line carrying just a date range, as table layouts put in their own column."""
    match = DATE_RANGE.search(line)
    return bool(match) and len(line.strip()) - len(match.group(0)) <= 6


def _employer_like(line: str) -> str | None:
    """A standalone employer line: short, no job title, no dates."""
    stripped = line.strip()
    if (not stripped or BULLET_START.match(stripped) or TITLE_KEYWORDS.search(stripped)
            or DATE_RANGE.search(stripped) or vocab.heading_for(stripped)
            or vocab.NOISE_MARKERS.search(stripped) or len(stripped) > 60
            or EMAIL.search(stripped)):
        return None
    return SEPARATORS.split(stripped)[0].strip() or None


def parse_companies(lines: list[str]) -> list[Company]:
    """Read job entries out of the experience section.

    Employers are identified structurally rather than by lookup, since tier-3
    names are invented: a job header is split on its separators and the part
    *without* a job-title keyword is taken as the employer. Templates order
    these differently - "Title, Employer" on one line, "Employer - Title", or a
    table that puts dates in their own column above a title and employer on
    separate lines - so dates seen just before a title are carried forward.
    """
    companies: list[Company] = []
    pending_dates: tuple[str | None, str | None] | None = None

    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or BULLET_START.match(stripped) or vocab.NOISE_MARKERS.search(stripped):
            continue
        if _is_date_only(stripped):
            pending_dates = find_date_range(stripped)
            continue
        if len(stripped) > 110 or not TITLE_KEYWORDS.search(stripped):
            continue

        parts = [p.strip() for p in SEPARATORS.split(stripped) if p.strip()]
        parts = [p for p in parts if not DATE_RANGE.fullmatch(p)]
        title = next((p for p in parts if TITLE_KEYWORDS.search(p)), None)
        if title is None:
            continue
        employer = next((p for p in parts
                         if p != title and not TITLE_KEYWORDS.search(p)
                         and not DATE_RANGE.search(p) and 1 < len(p) < 60), None)

        dates = find_date_range(stripped)

        # The employer, when not on the title line, is on the line directly
        # below it. Searching further ahead picks up unrelated content once a
        # multi-column reading order interleaves sections.
        if employer is None:
            following = next((ln.strip() for ln in lines[index + 1:index + 3]
                              if ln.strip() and not BULLET_START.match(ln.strip())), None)
            if following:
                employer = _employer_like(following)

        # Dates are more forgiving: they may be a line or two away.
        if dates is None:
            for lookahead in lines[index + 1:index + 4]:
                candidate = lookahead.strip()
                if not candidate or BULLET_START.match(candidate):
                    continue
                dates = find_date_range(candidate)
                if dates or TITLE_KEYWORDS.search(candidate):
                    break

        if dates is None:
            dates, pending_dates = pending_dates, None
        if employer is None:
            continue

        start, end = dates if dates else (None, None)
        companies.append(Company(name=employer, title=title, start_date=start,
                                 end_date=end, tier=vocab.company_tier(employer)))
    return companies


def total_years_from(companies: list[Company], reference: str = "2026-09") -> float | None:
    """Sum tenure across roles, which is how the labels define total experience."""
    months = 0
    for company in companies:
        if not company.start_date:
            continue
        end = company.end_date or reference
        try:
            sy, sm = int(company.start_date[:4]), int(company.start_date[5:7])
            ey, em = int(end[:4]), int(end[5:7])
        except (ValueError, IndexError):
            continue
        span = (ey * 12 + em) - (sy * 12 + sm)
        if 0 < span < 600:
            months += span
    return round(months / 12, 1) if months else None


def score_confidence(record: CVRecord, signals, sections: dict,
                     experience_lines: list[str] | None = None) -> float:
    """How much to trust this parse, in 0..1.

    Field coverage drives it; layout irregularity discounts it, because a
    multi-column or image-bearing document is exactly where reading order and
    silent omissions go wrong.
    """
    if signals.is_image_only:
        return 0.0

    weights = {"name": 0.2, "education": 0.2, "skills": 0.25, "companies": 0.35}

    # A graduate CV genuinely has no employment history. Finding no jobs in a
    # section that contains no dates either is a correct read, not a failure,
    # so the employment weight is redistributed rather than lost.
    jobs_expected = any(DATE_RANGE.search(line) for line in (experience_lines or []))
    if not jobs_expected and not record.companies:
        weights = {"name": 0.3, "education": 0.3, "skills": 0.4, "companies": 0.0}

    score = 0.0
    score += weights["name"] * bool(record.name)
    score += weights["education"] * min(1.0, len(record.education))
    score += weights["skills"] * min(1.0, len(record.skills) / 3)
    score += weights["companies"] * min(1.0, len(record.companies) / 2)

    dated = [c for c in record.companies if c.start_date]
    if record.companies:
        score *= 0.75 + 0.25 * (len(dated) / len(record.companies))
    if signals.is_multi_column:
        score *= 0.85
    if not {"experience", "education", "skills"} & set(sections):
        score *= 0.7
    return round(min(1.0, score), 4)


def parse(document: Document, threshold: float = 0.0) -> ExtractionResult:
    """Full rule-based extraction for one document."""
    lines = document.body_text.splitlines()
    sections = split_sections(lines)

    experience_lines = sections.get("experience", [])
    education_lines = sections.get("education", [])
    skills_lines = sections.get("skills", [])

    # Sections we could not label still carry content on messy layouts.
    if not experience_lines:
        experience_lines = sections.get("_preamble", []) + sum(
            (v for k, v in sections.items() if k not in vocab.SECTION_SYNONYMS), [])

    companies = parse_companies(experience_lines)
    record = CVRecord(
        name=parse_name(document, sections),
        education=parse_education(education_lines or lines),
        skills=parse_skills(skills_lines),
        companies=companies,
        total_years=total_years_from(companies),
    )
    confidence = score_confidence(record, document.signals, sections, experience_lines)
    return ExtractionResult(
        record=record,
        signals=document.signals,
        confidence=confidence,
        mode="ats",
        needs_llm=confidence < threshold,
        sections_found=sorted(k for k in sections if k != "_preamble"),
    )
