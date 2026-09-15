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

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pdfplumber

from .schema import LayoutSignals

LINE_TOLERANCE = 2.5         # points; words within this share a line
MIN_GUTTER_WIDTH = 8.0       # empty band width that marks a column break
                             # (safe to be small: a within-line space is still
                             #  covered by other lines at that x)
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
    title_line: str = ""   # largest type on page 1 - almost always the name
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
    size: float = 0.0


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
    sizes = [w["size"] for w in words if w.get("size")]
    return _Line(
        top=min(w["top"] for w in words),
        bottom=max(w["bottom"] for w in words),
        x0=min(w["x0"] for w in words),
        x1=max(w["x1"] for w in words),
        text=text.strip(),
        size=round(sum(sizes) / len(sizes), 2) if sizes else 0.0,
    )


def _word_gutter(words: list[dict], page_width: float) -> float | None:
    """Find the column boundary from raw word coverage.

    Deriving it from assembled lines is circular: two columns sharing a
    baseline merge into one line, which then looks like it spans the gutter.
    Coverage over x has no such problem - a true gutter is a vertical band no
    word occupies, however narrow.
    """
    if len(words) < 30:
        return None

    resolution = 2.0
    n_bins = int(page_width / resolution) + 1
    covered = [False] * n_bins
    for word in words:
        start = max(0, int(word["x0"] / resolution))
        end = min(n_bins - 1, int(word["x1"] / resolution))
        for b in range(start, end + 1):
            covered[b] = True

    low, high = int(page_width * 0.22 / resolution), int(page_width * 0.78 / resolution)
    best, best_score = None, 0.0
    b = low
    while b <= high:
        if covered[b]:
            b += 1
            continue
        run_start = b
        while b <= high and not covered[b]:
            b += 1
        if (b - run_start) * resolution < MIN_GUTTER_WIDTH:
            continue

        left_edge, right_edge = run_start * resolution, b * resolution
        left = [w for w in words if w["x1"] <= left_edge]
        right = [w for w in words if w["x0"] >= right_edge]
        if not left or not right:
            continue
        share = min(len(left), len(right)) / len(words)
        if share < MIN_COLUMN_SHARE:
            continue
        if len(_group_lines(left)) < MIN_COLUMN_LINES or len(_group_lines(right)) < MIN_COLUMN_LINES:
            continue

        left_top, left_bottom = min(w["top"] for w in left), max(w["bottom"] for w in left)
        right_top, right_bottom = min(w["top"] for w in right), max(w["bottom"] for w in right)
        overlap = min(left_bottom, right_bottom) - max(left_top, right_top)
        shorter = min(left_bottom - left_top, right_bottom - right_top)
        if shorter > 0 and overlap / shorter >= MIN_VERTICAL_OVERLAP and share > best_score:
            best, best_score = (left_edge + right_edge) / 2, share
    return best


def _page_lines(page) -> tuple[list[str], list[_Line], list[str], int]:
    """Returns (header lines, body lines in reading order, footer lines, column count)."""
    words = page.extract_words(use_text_flow=False, keep_blank_chars=False,
                               extra_attrs=["size"])
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

    gutter = _word_gutter(body_words, page.width)
    if gutter is None:
        return header_lines, _group_lines(body_words), footer_lines, 1

    # Group each column separately so text sharing a baseline across the gutter
    # never merges, and emit whole columns in reading order.
    spanning = [w for w in body_words if w["x0"] < gutter < w["x1"]]
    left = [w for w in body_words if w["x1"] <= gutter and w not in spanning]
    right = [w for w in body_words if w["x0"] >= gutter and w not in spanning]
    ordered = (_group_lines(spanning) + _group_lines(left) + _group_lines(right))
    return header_lines, ordered, footer_lines, 2


def _largest_line(lines: list[_Line]) -> str:
    """The line set in the biggest type, when it clearly stands out.

    Resumes render the candidate's name larger than everything else, which is a
    far steadier signal than position once a sidebar reorders the page.
    """
    sized = [ln for ln in lines if ln.size and ln.text.strip()]
    if not sized:
        return ""
    biggest = max(sized, key=lambda ln: ln.size)
    typical = sorted(ln.size for ln in sized)[len(sized) // 2]
    return biggest.text if biggest.size >= typical * 1.35 else ""


CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "pdf_text"


def read_document_cached(path: str | Path, cache_dir: Path = CACHE_DIR) -> Document:
    """`read_document` with an on-disk cache.

    Reading a PDF costs far more than parsing its text, and the parser is
    iterated on constantly while the text for a given file never changes. The
    key includes size and mtime, so re-rendering the corpus invalidates it.
    """
    path = Path(path)
    stat = path.stat()
    key = hashlib.sha1(
        f"{path.resolve()}|{stat.st_size}|{int(stat.st_mtime)}".encode()).hexdigest()
    cached = cache_dir / f"{key}.json"

    if cached.exists():
        try:
            payload = json.loads(cached.read_text(encoding="utf-8"))
            return Document(
                body_text=payload["body_text"],
                header_text=payload["header_text"],
                footer_text=payload["footer_text"],
                title_line=payload.get("title_line", ""),
                signals=LayoutSignals(**payload["signals"]),
            )
        except (ValueError, KeyError, TypeError):
            pass   # stale or truncated cache entry; fall through and re-read

    document = read_document(path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps({
        "body_text": document.body_text,
        "header_text": document.header_text,
        "footer_text": document.footer_text,
        "title_line": document.title_line,
        "signals": asdict(document.signals),
    }), encoding="utf-8")
    return document


def read_document(path: str | Path) -> Document:
    """Read a CV into text plus the layout signals the confidence score uses."""
    header, body, footer, title_line = [], [], [], ""
    max_columns, has_images, n_pages = 1, False, 0

    with pdfplumber.open(str(path)) as pdf:
        n_pages = len(pdf.pages)
        for page in pdf.pages:
            has_images = has_images or bool(page.images)
            page_header, page_body, page_footer, columns = _page_lines(page)
            max_columns = max(max_columns, columns)
            header += page_header
            if not body and page_body:
                title_line = _largest_line(page_body)
            body += [ln.text for ln in page_body]
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
                    footer_text="\n".join(footer), title_line=title_line,
                    signals=signals)
