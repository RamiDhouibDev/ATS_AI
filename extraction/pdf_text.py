"""PDF -> text, with the layout awareness a naive extractor lacks.

Two things here matter more than anything else in the parser:

1. **Reading order.** `page.extract_text()` walks the page top-to-bottom, so a
   two-column CV comes back with the sidebar interleaved into the job history.
   We detect the gutter and emit whole columns in order instead.
2. **Page furniture.** Contact details parked in a page header are invisible to
   most ATS parsers. We pull headers and footers out separately, so they stop
   polluting the body *and* remain available to read.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber

from .schema import LayoutSignals

LINE_TOLERANCE = 2.5         # points; words within this share a line
MIN_GUTTER_WIDTH = 18.0      # points of empty space needed to call it a column break
MIN_COLUMN_SHARE = 0.10      # each side must hold this fraction of the page's words
MIN_COLUMN_LINES = 4         # ...and this many lines, so stray words don't split a page
MIN_VERTICAL_OVERLAP = 0.55  # columns must genuinely run alongside each other
COLUMN_GAP = 24.0            # horizontal gap that ends a line rather than being a wide space
HEADER_BAND = 52.0          # points from the top that can hold page furniture
FOOTER_BAND = 46.0          # points from the bottom
CID_GLYPH = re.compile(r"\(cid:\d+\)")
CONTACT_HINT = re.compile(r"[\w.+-]+@[\w-]+\.\w+|\+?\d[\d ().-]{7,}\d|page \d+", re.I)


@dataclass
class Document:
    body_text: str = ""
    header_text: str = ""
    footer_text: str = ""
    signals: LayoutSignals = field(default_factory=LayoutSignals)

    @property
    def all_text(self) -> str:
        return "\n".join(t for t in (self.header_text, self.body_text, self.footer_text) if t)


@dataclass
class _Line:
    top: float
    bottom: float
    x0: float
    x1: float
    text: str


def _group_lines(words: list[dict]) -> list[_Line]:
    """Collapse words into lines, preserving left-to-right order within each."""
    if not words:
        return []
    buckets: list[list[dict]] = []
    for word in sorted(words, key=lambda w: (round(w["top"], 1), w["x0"])):
        if buckets and abs(word["top"] - buckets[-1][0]["top"]) <= LINE_TOLERANCE:
            buckets[-1].append(word)
        else:
            buckets.append([word])
    lines = []
    for ws in buckets:
        # Split on wide horizontal gaps: text from two columns can share a
        # baseline, and merging it would produce a phantom full-width line.
        segment = []
        for word in sorted(ws, key=lambda w: w["x0"]):
            if segment and word["x0"] - segment[-1]["x1"] > COLUMN_GAP:
                lines.append(_line_from(segment))
                segment = []
            segment.append(word)
        if segment:
            lines.append(_line_from(segment))
    return sorted(lines, key=lambda ln: (round(ln.top, 1), ln.x0))


def _line_from(words: list[dict]) -> _Line:
    # Symbol fonts without a ToUnicode map surface as "(cid:NNN)"; in practice
    # these are the bullet glyphs, and leaving them in breaks bullet detection.
    text = CID_GLYPH.sub("•", " ".join(w["text"] for w in words))
    return _Line(
        top=min(w["top"] for w in words),
        bottom=max(w["bottom"] for w in words),
        x0=min(w["x0"] for w in words),
        x1=max(w["x1"] for w in words),
        text=text.strip(),
    )


def _find_gutter(lines: list[_Line], page_width: float, page_height: float) -> float | None:
    """Return the x of a vertical gap that splits the page into two columns.

    Tested on whole lines, not words: a wrapped prose line is a run of short
    tokens, so any space landing in the candidate band would otherwise look like
    a column break and scramble a perfectly ordinary single-column CV.

    A gap alone still isn't enough - sparse pages have plenty of empty bands -
    so both sides must carry several lines and genuinely run alongside each
    other. Full-width lines high on the page are tolerated, since two-column
    designs routinely sit under a full-width name block.
    """
    if len(lines) < 10:
        return None

    header_zone = page_height * 0.25
    best, best_score = None, 0.0
    x = page_width * 0.25
    while x <= page_width * 0.75:
        crossing = [ln for ln in lines
                    if ln.x0 < x - MIN_GUTTER_WIDTH / 2 and ln.x1 > x + MIN_GUTTER_WIDTH / 2]
        if any(ln.top > header_zone for ln in crossing):
            x += 4.0
            continue

        left = [ln for ln in lines if ln.x1 <= x and ln not in crossing]
        right = [ln for ln in lines if ln.x0 >= x and ln not in crossing]
        if len(left) < MIN_COLUMN_LINES or len(right) < MIN_COLUMN_LINES:
            x += 4.0
            continue
        share = min(len(left), len(right)) / len(lines)
        if share < MIN_COLUMN_SHARE:
            x += 4.0
            continue

        left_top, left_bottom = min(l.top for l in left), max(l.bottom for l in left)
        right_top, right_bottom = min(r.top for r in right), max(r.bottom for r in right)
        overlap = min(left_bottom, right_bottom) - max(left_top, right_top)
        shorter = min(left_bottom - left_top, right_bottom - right_top)
        if shorter > 0 and overlap / shorter >= MIN_VERTICAL_OVERLAP and share > best_score:
            best, best_score = x, share
        x += 4.0
    return best


def _page_lines(page) -> tuple[list[str], list[str], list[str], int]:
    """Returns (header lines, body lines in reading order, footer lines, column count)."""
    words = page.extract_words(use_text_flow=False, keep_blank_chars=False)
    if not words:
        return [], [], [], 1

    height = page.height
    header_words, footer_words, body_words = [], [], []
    for w in words:
        if w["top"] < HEADER_BAND:
            header_words.append(w)
        elif w["bottom"] > height - FOOTER_BAND:
            footer_words.append(w)
        else:
            body_words.append(w)

    # Only treat a top band as furniture when it actually looks like furniture;
    # otherwise it is just the candidate's name sitting high on the page.
    header_lines = [ln.text for ln in _group_lines(header_words)]
    if header_lines and not any(CONTACT_HINT.search(line) for line in header_lines):
        body_words = header_words + body_words
        header_lines = []

    footer_lines = [ln.text for ln in _group_lines(footer_words)]
    if footer_lines and not any(CONTACT_HINT.search(line) for line in footer_lines):
        body_words = body_words + footer_words
        footer_lines = []

    lines = _group_lines(body_words)
    gutter = _find_gutter(lines, page.width, page.height)
    if gutter is None:
        return header_lines, [ln.text for ln in lines], footer_lines, 1

    spanning = [ln for ln in lines
                if ln.x0 < gutter - MIN_GUTTER_WIDTH / 2 and ln.x1 > gutter + MIN_GUTTER_WIDTH / 2]
    left = [ln for ln in lines if ln.x1 <= gutter and ln not in spanning]
    right = [ln for ln in lines if ln.x0 >= gutter and ln not in spanning]
    ordered = spanning + left + right
    return header_lines, [ln.text for ln in ordered], footer_lines, 2


def read_document(path: str | Path) -> Document:
    """Read a CV into text plus the layout signals the confidence score uses."""
    header, body, footer = [], [], []
    max_columns, has_images, n_pages = 1, False, 0

    with pdfplumber.open(str(path)) as pdf:
        n_pages = len(pdf.pages)
        for page in pdf.pages:
            has_images = has_images or bool(page.images)
            page_header, page_body, page_footer, columns = _page_lines(page)
            max_columns = max(max_columns, columns)
            header += page_header
            body += page_body
            footer += page_footer

    body_text = "\n".join(body)
    signals = LayoutSignals(
        char_count=len(body_text),
        n_pages=n_pages,
        max_columns=max_columns,
        has_images=has_images,
        has_header_text=bool(header),
        has_footer_text=bool(footer),
    )
    return Document(body_text=body_text, header_text="\n".join(header),
                    footer_text="\n".join(footer), signals=signals)
