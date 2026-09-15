"""
Renders each mock candidate into a realistic CV document.

Every CV is drawn from one of ten hand-designed layouts (see cv_templates.py)
and then varied further - fonts, colours, heading synonyms, date formats,
section order, how skills are presented, which filler sections appear. A small
slice is degraded into image-only "scanned" PDFs with no text layer at all,
which no text parser can read; those exist to force the vision/LLM fallback.

Ground truth stays in data_layer1/<split>/<split>.jsonl, joined by id. The
manifest records how hard each document should be to parse.

Usage:
    python render_cvs.py --scanned-frac 0.04 --seed 42
"""

import argparse
import csv
import io
import json
import random
import time
from pathlib import Path

import pymupdf as fitz
from faker import Faker
from PIL import Image, ImageEnhance, ImageFilter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.platypus import SimpleDocTemplate

import cv_templates as TPL

TEMPLATE_NAMES = [t[0] for t in TPL.TEMPLATES]
TEMPLATE_WEIGHTS = [t[3] for t in TPL.TEMPLATES]
TEMPLATE_BY_NAME = {t[0]: t for t in TPL.TEMPLATES}


def load_jsonl(path: Path) -> list:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def page_furniture(ctx: dict, record: dict, header_text: str, margin: float):
    """Header/footer painter. Contact details parked in a page header are
    invisible to most text parsers - a real and common resume mistake."""
    def draw(canvas, doc):
        canvas.saveState()
        width, height = ctx["pagesize"]
        canvas.setFillColor(colors.grey)
        if ctx["header_contact"]:
            canvas.setFont(ctx["font"], 8)
            canvas.drawString(margin, height - margin * 0.55,
                              f"{record['name']}  |  {header_text}")
            canvas.setStrokeColor(colors.Color(0.85, 0.85, 0.85))
            canvas.line(margin, height - margin * 0.72, width - margin, height - margin * 0.72)
        if ctx["footer_pagenum"]:
            canvas.setFont(ctx["font"], 8)
            canvas.drawRightString(width - margin, margin * 0.45,
                                   f"{record['name']} — page {doc.page}")
        canvas.restoreState()
    return draw


def write_with_retry(build, out_path: Path, attempts: int = 4):
    """A PDF viewer or OneDrive sync can hold a file open; retry before giving up."""
    for attempt in range(attempts):
        try:
            return build()
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.6 * (attempt + 1))


def build_pdf(record, ctx, template_fn, fake, out_path: Path):
    margin = ctx["rng"].uniform(0.5, 0.85) * inch
    top_margin = margin + (16 if ctx["header_contact"] else 0)
    doc = SimpleDocTemplate(
        str(out_path), pagesize=ctx["pagesize"],
        topMargin=top_margin, bottomMargin=margin, leftMargin=margin, rightMargin=margin,
        title=f"{record['name']} - CV", author=record["name"],
        # Suppress the embedded creation timestamp so a re-run with the same seed
        # produces byte-identical files instead of churning all 1,000 in git.
        invariant=1,
    )
    painter = page_furniture(ctx, record, ctx["header_contact_text"], margin)
    story = template_fn(record, ctx, fake)
    write_with_retry(lambda: doc.build(story, onFirstPage=painter, onLaterPages=painter), out_path)


def scan_degrade(pdf_path: Path, rng: random.Random):
    """Rasterise to images, skew/blur/noise them, and write back an image-only PDF."""
    doc = fitz.open(str(pdf_path))
    pages = []
    for page in doc:
        pix = page.get_pixmap(dpi=rng.choice([120, 140, 150]))
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
        img = img.rotate(rng.uniform(-0.9, 0.9), resample=Image.BICUBIC,
                         expand=False, fillcolor=245)
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.7)))
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.85, 1.15))
        noise = Image.effect_noise(img.size, rng.uniform(6, 14)).convert("L")
        img = Image.blend(img, noise, rng.uniform(0.04, 0.09))
        pages.append(img.convert("RGB"))
    doc.close()
    pages[0].save(str(pdf_path), save_all=True, append_images=pages[1:], resolution=110.0)
    return len(pages)


def page_count(pdf_path: Path) -> int:
    doc = fitz.open(str(pdf_path))
    n = doc.page_count
    doc.close()
    return n


def render_record(record: dict, out_path: Path, seed: int, scanned_frac: float, fake: Faker) -> dict:
    rng = random.Random(f"{seed}-{record['id']}")
    name = rng.choices(TEMPLATE_NAMES, weights=TEMPLATE_WEIGHTS, k=1)[0]
    _, template_fn, difficulty, _ = TEMPLATE_BY_NAME[name]
    ctx = TPL.build_context(record, rng)

    # Contact details and per-job locations need Faker, which lives here.
    contact = TPL.contact_string(record, ctx, fake)
    ctx["header_contact_text"] = contact
    if ctx["header_contact"]:
        ctx["contact"] = ctx["contact_multiline"] = ""   # body carries none of it
    else:
        ctx["contact"] = contact
        ctx["contact_multiline"] = TPL.contact_string(record, ctx, fake, multiline=True)
    ctx["job_locations"] = [
        f"{fake.city()}" + (f" ({rng.choice(TPL.T.WORK_MODES)})" if rng.random() < 0.45 else "")
        for _ in record["companies"]
    ]

    fallback = ""
    try:
        build_pdf(record, ctx, template_fn, fake, out_path)
    except Exception as first_error:
        # Content overflowed an un-splittable layout: retry trimmed down, then plain.
        lean = dict(ctx, bullets_recent=1, show_certifications=False, show_languages=False,
                    show_interests=False, show_awards=False, show_projects=False,
                    show_publications=False, show_volunteering=False, show_highlights=False,
                    show_coursework=False, used_bullets=set())
        try:
            build_pdf(record, lean, template_fn, fake, out_path)
            fallback = f"trimmed: {type(first_error).__name__}"
        except Exception as second_error:
            fallback = f"{name}->classic_ats: {type(second_error).__name__}"
            name, difficulty = "classic_ats", "easy"
            build_pdf(record, lean, TPL.tpl_classic_ats, fake, out_path)

    scanned = rng.random() < scanned_frac
    pages = scan_degrade(out_path, rng) if scanned else page_count(out_path)

    return {
        "template": name,
        "difficulty": "scanned" if scanned else difficulty,
        "pagesize": "A4" if abs(ctx["pagesize"][0] - 595.27) < 1 else "letter",
        "skill_mode": ctx["skill_mode"],
        "skill_years_stated": ctx["skill_mode"] == "years",
        "scanned": scanned,
        "contact_in_header_only": ctx["header_contact"],
        "pages": pages,
        "fallback": fallback,
    }


def build_contact_sheet(samples: list, out_path: Path, cols: int = 3):
    """One page thumbnail grid so the layout variety can be eyeballed at a glance."""
    c = pdfcanvas.Canvas(str(out_path), pagesize=(11 * inch, 8.5 * inch), invariant=1)
    cell_w, cell_h = 3.3 * inch, 3.6 * inch
    margin_x, margin_y = 0.4 * inch, 0.35 * inch
    per_page = cols * 2
    for i, (label, pdf_path) in enumerate(samples):
        if i and i % per_page == 0:
            c.showPage()
        slot = i % per_page
        col, row = slot % cols, slot // cols
        x = margin_x + col * (cell_w + 0.2 * inch)
        y = 8.5 * inch - margin_y - (row + 1) * (cell_h + 0.3 * inch)

        doc = fitz.open(str(pdf_path))
        pix = doc[0].get_pixmap(dpi=72)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        doc.close()
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        scale = min(cell_w / img.width, cell_h / img.height)
        from reportlab.lib.utils import ImageReader
        c.drawImage(ImageReader(buf), x, y, width=img.width * scale, height=img.height * scale)
        c.setFont("Helvetica", 8)
        c.drawString(x, y - 11, label)
    c.save()


def main():
    parser = argparse.ArgumentParser(description="Render mock candidates into varied CV PDFs.")
    parser.add_argument("--scanned-frac", type=float, default=0.04,
                        help="Fraction degraded into image-only scans")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default="..")
    parser.add_argument("--limit", type=int, default=None, help="Render only the first N per split")
    args = parser.parse_args()

    data_dir = (Path(__file__).parent / args.data_dir).resolve()
    fake = Faker()
    Faker.seed(args.seed)

    manifest, seen_templates, blocked = [], {}, []
    for split in ("train", "test"):
        split_dir = data_dir / split
        records = load_jsonl(split_dir / f"{split}.jsonl")
        if args.limit:
            records = records[:args.limit]
        out_dir = split_dir / "cvs_pdf"
        out_dir.mkdir(parents=True, exist_ok=True)

        for record in records:
            out_path = out_dir / f"{record['id']}.pdf"
            try:
                meta = render_record(record, out_path, args.seed, args.scanned_frac, fake)
            except PermissionError:
                blocked.append(str(out_path))
                continue
            manifest.append({"id": record["id"], "split": split,
                             "pdf_path": f"cvs_pdf/{record['id']}.pdf", **meta})
            key = (meta["template"], meta["scanned"])
            if key not in seen_templates:
                seen_templates[key] = out_path
        print(f"{split}: rendered {len(records)} CVs -> {out_dir}")

    # One manifest per split, written beside that split's documents.
    for split in ("train", "test"):
        split_manifest = [row for row in manifest if row["split"] == split]
        if not split_manifest:
            continue
        with (data_dir / split / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(split_manifest[0].keys()))
            w.writeheader()
            w.writerows(split_manifest)

    samples = [(f"{tpl}{' (scanned)' if sc else ''}", path)
               for (tpl, sc), path in sorted(seen_templates.items())]
    sheet = data_dir / "_layout_samples.pdf"
    build_contact_sheet(samples, sheet)

    if blocked:
        print(f"\nERROR: {len(blocked)} file(s) locked by another program and NOT "
              f"regenerated (close any open PDF viewer): {blocked[:3]}")

    degraded = [r for r in manifest if r["fallback"]]
    if degraded:
        print(f"\nWARNING: {len(degraded)} CVs fell back to a simpler layout, e.g. "
              f"{degraded[0]['id']} ({degraded[0]['fallback']})")

    by_difficulty = {}
    for row in manifest:
        by_difficulty[row["difficulty"]] = by_difficulty.get(row["difficulty"], 0) + 1
    print(f"\nTotal {len(manifest)} CVs. Difficulty mix: {by_difficulty}")
    print(f"Manifests: {data_dir / '<split>' / 'manifest.csv'}")
    print(f"Layout sample sheet: {sheet}")


if __name__ == "__main__":
    main()
