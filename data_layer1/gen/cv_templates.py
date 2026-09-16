"""
Ten hand-designed resume layouts, plus the per-document variation that stops
any two CVs looking alike: fonts, colours, heading synonyms, date formats,
section order, how skills are presented, and which sections appear at all.

How much content a CV carries follows the candidate's seniority, the way real
resumes do - juniors lead with education and projects and run to one page,
seniors lead with experience, carry 3-5 bullets on the recent role tapering to
1-2 on older ones, and often spill onto a second page.

Difficulty tiers describe how hard the document is to parse:
  easy   - plain single column, linear reading order
  medium - styled, varied headings/order, still linear
  hard   - sidebars, true multi-column, or table grids where naive text
           extraction scrambles the reading order
"""

import random

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_RIGHT
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable, KeepTogether, ListFlowable, ListItem, Paragraph, Spacer, Table, TableStyle,
)

import cv_text

ACCENTS = ["#1a3c6e", "#2c5f2d", "#6b2737", "#37474f", "#0f4c5c", "#4a148c",
           "#8d5524", "#1b5e20", "#263238", "#7b1fa2"]

FONT_PAIRS = [
    ("Helvetica", "Helvetica-Bold", "Helvetica-Oblique"),
    ("Helvetica", "Helvetica-Bold", "Helvetica-Oblique"),
    ("Times-Roman", "Times-Bold", "Times-Italic"),
]

# "table" and "bars" are the presentations that defeat naive text extraction;
# only "years" states years in words. "none" is rare - most CVs do list skills.
SKILL_MODES = ["years", "plain", "categorised", "bullets", "levels", "bars", "table", "none"]
SKILL_MODE_WEIGHTS = [20, 17, 18, 13, 9, 9, 12, 2]


def seniority(record: dict) -> str:
    years = record["general_experience"]["total_years"]
    if years < 3:
        return "junior"
    return "mid" if years < 8 else "senior"


def build_context(record: dict, rng: random.Random) -> dict:
    """All the per-document randomised choices, decided once up front."""
    base, bold, italic = rng.choice(FONT_PAIRS)
    tier = seniority(record)
    top_level = record["education"][0]["level"]

    # Recent-role bullet count; older roles taper from this.
    bullets_recent = {"junior": rng.randint(2, 4),
                      "mid": rng.randint(3, 4),
                      "senior": rng.randint(3, 5)}[tier]

    # Juniors lean on projects and coursework; seniors on certifications and scope.
    p_projects = {"junior": 0.75, "mid": 0.40, "senior": 0.25}[tier]
    p_certs = {"junior": 0.25, "mid": 0.50, "senior": 0.60}[tier]
    p_coursework = {"junior": 0.50, "mid": 0.15, "senior": 0.04}[tier]
    p_edu_first = {"junior": 0.70, "mid": 0.20, "senior": 0.08}[tier]
    p_highlights = {"junior": 0.0, "mid": 0.12, "senior": 0.30}[tier]

    return {
        "rng": rng,
        "tier": tier,
        "font": base,
        "font_bold": bold,
        "font_italic": italic,
        "accent": colors.HexColor(rng.choice(ACCENTS)),
        "date_style": rng.randint(0, 5),
        "bullet_char": rng.choice(cv_text.BULLET_CHARS),
        "skill_mode": rng.choices(SKILL_MODES, weights=SKILL_MODE_WEIGHTS, k=1)[0],
        "pagesize": A4 if rng.random() < 0.4 else letter,
        "headings": {key: cv_text.heading(rng, key) for key in cv_text.SECTION_HEADINGS},
        "bullets_recent": bullets_recent,
        "education_first": rng.random() < p_edu_first,
        "show_summary": rng.random() < 0.80,
        "show_projects": rng.random() < p_projects,
        "show_certifications": rng.random() < p_certs,
        "show_coursework": rng.random() < p_coursework,
        "show_thesis": top_level in ("Master", "PhD") and rng.random() < 0.45,
        "show_honours": rng.random() < 0.35,
        "show_publications": top_level == "PhD" and rng.random() < 0.60,
        "show_highlights": rng.random() < p_highlights,
        "show_volunteering": rng.random() < 0.22,
        "show_languages": rng.random() < 0.40,
        "show_interests": rng.random() < 0.28,
        "show_awards": rng.random() < 0.22,
        "show_references_line": rng.random() < 0.22,
        "show_job_location": rng.random() < 0.55,
        "uppercase_name": rng.random() < 0.18,
        "uppercase_headings": rng.random() < 0.45,
        "used_bullets": set(),   # shared across jobs so no sentence repeats on one CV
        # Contact details and per-job locations are filled in by the renderer,
        # which owns the Faker instance.
        "contact": "",
        "contact_multiline": "",
        "job_locations": [],
        # Contact in a page header only - invisible to most parsers, and real.
        "header_contact": rng.random() < 0.12,
        "footer_pagenum": rng.random() < 0.30,
    }


def make_styles(ctx: dict, on_dark: bool = False) -> dict:
    fg = colors.white if on_dark else colors.black
    muted = colors.Color(0.85, 0.85, 0.85) if on_dark else colors.dimgrey
    accent = colors.white if on_dark else ctx["accent"]
    return {
        "name": ParagraphStyle("nm", fontName=ctx["font_bold"], fontSize=19, leading=22,
                                textColor=fg, spaceAfter=2),
        "title_line": ParagraphStyle("tl", fontName=ctx["font_italic"], fontSize=10.5,
                                      leading=13, textColor=muted, spaceAfter=6),
        "contact": ParagraphStyle("ct", fontName=ctx["font"], fontSize=8.5, leading=11,
                                   textColor=muted, spaceAfter=8),
        "section": ParagraphStyle("sc", fontName=ctx["font_bold"], fontSize=11.5, leading=14,
                                   textColor=accent, spaceBefore=9, spaceAfter=3),
        "body": ParagraphStyle("bd", fontName=ctx["font"], fontSize=9.5, leading=12.5,
                                textColor=fg, spaceAfter=1),
        "body_bold": ParagraphStyle("bb", fontName=ctx["font_bold"], fontSize=9.5, leading=12.5,
                                     textColor=fg, spaceAfter=1),
        "small": ParagraphStyle("sm", fontName=ctx["font"], fontSize=8.5, leading=11,
                                 textColor=muted, spaceAfter=3),
        "bullet": ParagraphStyle("bu", fontName=ctx["font"], fontSize=9, leading=12,
                                  textColor=fg, spaceAfter=1),
    }


def heading_for(ctx: dict, key: str) -> str:
    text = ctx["headings"][key]
    return text.upper() if ctx["uppercase_headings"] else text


def contact_string(record: dict, ctx: dict, fake, multiline: bool = False) -> str:
    rng = ctx["rng"]
    parts = [fake.email(), fake.phone_number()]
    if rng.random() < 0.6:
        parts.append(fake.city())
    if rng.random() < 0.5:
        handle = record["name"].lower().replace(" ", "-").replace(".", "")
        parts.append(f"linkedin.com/in/{handle}")
    if rng.random() < 0.3:
        parts.append(f"github.com/{record['name'].split()[0].lower()}dev")
    sep = "<br/>" if multiline else rng.choice([" | ", " • ", " – ", "  "])
    return sep.join(parts)


# ---------------------------------------------------------------------------
# Shared content blocks
# ---------------------------------------------------------------------------

def bullet_count(ctx: dict, index: int) -> int:
    """Most recent role carries the most detail; older roles taper."""
    base = ctx["bullets_recent"]
    if index == 0:
        return base
    if index == 1:
        return max(1, base - 1)
    return max(1, base - 2)


def bullet_list(lines: list, ctx: dict, st: dict) -> ListFlowable:
    return ListFlowable(
        [ListItem(Paragraph(line, st["bullet"]), leftIndent=8) for line in lines],
        bulletType="bullet", start=ctx["bullet_char"], leftIndent=12, bulletFontSize=7)


def job_flowables(record: dict, ctx: dict, st: dict, compact: bool = False,
                  keep: bool = True) -> list:
    # `keep=False` is required inside table cells: KeepTogether has no canvas
    # of its own there and blows up during wrap.
    rng = ctx["rng"]
    out = []
    for index, company in enumerate(record["companies"]):
        dates = cv_text.format_date_range(rng, company["start_date"], company["end_date"],
                                          ctx["date_style"])
        location = ""
        if ctx["show_job_location"] and index < len(ctx["job_locations"]):
            location = ctx["job_locations"][index]

        layout = rng.random()
        if layout < 0.45:
            head = f"<b>{company['title']}</b>, {company['name']}"
            sub = f"{dates}  |  {location}" if location else dates
        elif layout < 0.75:
            head = f"<b>{company['name']}</b> — {company['title']}"
            sub = f"{dates}  |  {location}" if location else dates
        else:
            head = f"<b>{company['title']}</b>"
            sub = f"{company['name']}  |  {dates}" + (f"  |  {location}" if location else "")

        block = [Paragraph(head, st["body"]), Paragraph(sub, st["small"])]
        if not compact:
            lines = cv_text.make_bullets(rng, record["general_experience"]["domain"],
                                   record["skills"], bullet_count(ctx, index), ctx["used_bullets"])
            if lines:
                block.append(bullet_list(lines, ctx, st))
        block.append(Spacer(1, 5))
        out.append(KeepTogether(block)) if keep else out.extend(block)

    if not record["companies"]:
        out.append(Paragraph("Seeking a first professional role; see projects below.",
                             st["small"]))
    return out


def education_flowables(record: dict, ctx: dict, st: dict) -> list:
    rng, out = ctx["rng"], []
    for index, entry in enumerate(record["education"]):
        degree = entry["level"] if entry["field"] is None else f"{entry['level']}, {entry['field']}"
        dates = cv_text.format_year_range(rng, entry["start_year"], entry["end_year"], ctx["date_style"])
        out.append(Paragraph(f"<b>{degree}</b>", st["body"]))
        out.append(Paragraph(f"{entry['institution']}  |  {dates}", st["small"]))

        if index == 0 and entry["field"]:
            if ctx["show_honours"]:
                out.append(Paragraph(rng.choice(cv_text.HONOURS), st["small"]))
            if ctx["show_thesis"] and entry["level"] in ("Master", "PhD"):
                topic = rng.choice(cv_text.THESIS_TOPICS.get(entry["field"],
                                                       cv_text.THESIS_TOPICS["Computer Science"]))
                out.append(Paragraph(f"Thesis: “{topic}”", st["small"]))
            if ctx["show_coursework"]:
                courses = cv_text.COURSEWORK.get(entry["field"], cv_text.COURSEWORK["Computer Science"])
                picked = rng.sample(courses, min(4, len(courses)))
                out.append(Paragraph(f"Relevant coursework: {', '.join(picked)}", st["small"]))
        out.append(Spacer(1, 3))
    return out


def rating_bar(level: int, ctx: dict, on_dark: bool = False) -> Table:
    """Five little squares, `level` of them filled - proficiency with no text at all."""
    filled = ctx["accent"] if not on_dark else colors.white
    empty = colors.Color(0.82, 0.82, 0.82)
    style = [("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 0)]
    for index in range(5):
        style.append(("BACKGROUND", (index, 0), (index, 0), filled if index < level else empty))
    table = Table([[""] * 5], colWidths=[9] * 5, rowHeights=[6])
    table.setStyle(TableStyle(style))
    return table


def years_label(raw_years: float, unit: str = "yrs") -> str:
    whole = max(1, int(round(raw_years)))
    if whole == 1:
        return "1 year" if unit in ("yrs", "years") else f"1 {unit}"
    return f"{whole} {unit}"


def skills_flowables(record: dict, ctx: dict, st: dict, on_dark: bool = False) -> list:
    """Skill presentation varies a lot; only the 'years' mode states years in words."""
    rng, mode, skills = ctx["rng"], ctx["skill_mode"], record["skills"]
    if not skills or mode == "none":
        return []

    if mode == "years":
        unit = rng.choice(["yrs", "years", "yr exp"])
        return [Paragraph(", ".join(f"{skill['name']} ({years_label(skill['years'], unit)})"
                                    for skill in skills), st["body"])]
    if mode == "plain":
        return [Paragraph(rng.choice([", ", " • ", " | "]).join(skill["name"] for skill in skills),
                          st["body"])]
    if mode == "categorised":
        return [Paragraph(f"<b>{cat}:</b> {', '.join(names)}", st["body"])
                for cat, names in cv_text.categorise_skills(skills)]
    if mode == "bullets":
        return [bullet_list([skill["name"] for skill in skills], ctx, st)]
    if mode == "levels":
        return [Paragraph(
            f"{skill['name']} — "
            f"{cv_text.SKILL_LEVEL_WORDS[min(4, max(0, 4 - int(skill['years'] // 2)))]}", st["body"])
            for skill in skills]
    if mode == "table":
        # A bordered "Skill | Years" grid - one of the classic ATS parser killers.
        rows = [[Paragraph("<b>Skill</b>", st["small"]), Paragraph("<b>Experience</b>", st["small"])]]
        for skill in skills:
            rows.append([Paragraph(skill["name"], st["body"]),
                         Paragraph(years_label(skill["years"]), st["body"])])
        table = Table(rows, colWidths=[None, 70], splitByRow=1)
        table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.4, colors.Color(0.78, 0.78, 0.78)),
            ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.94, 0.94, 0.94)),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ]))
        return [table]

    # bars
    rows = []
    for skill in skills:
        level = max(1, min(5, int(round(skill["years"] / 2.5)) + 1))
        rows.append([Paragraph(skill["name"], st["body"]), rating_bar(level, ctx, on_dark)])
    table = Table(rows, colWidths=[None, 50], splitByRow=1)
    table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("LEFTPADDING", (0, 0), (-1, -1), 0),
                           ("TOPPADDING", (0, 0), (-1, -1), 1),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
    return [table]


def projects_flowables(record: dict, ctx: dict, st: dict) -> list:
    rng = ctx["rng"]
    count = 3 if ctx["tier"] == "junior" else rng.randint(1, 2)
    out = []
    for name, desc in cv_text.make_projects(rng, record, count):
        out.append(Paragraph(f"<b>{name}</b>", st["body"]))
        out.append(Paragraph(desc, st["small"]))
    return out


def highlights_flowables(record: dict, ctx: dict, st: dict) -> list:
    lines = cv_text.make_bullets(ctx["rng"], record["general_experience"]["domain"],
                           record["skills"], 3, ctx["used_bullets"])
    return [bullet_list(lines, ctx, st)] if lines else []


def extra_section_flowables(record: dict, ctx: dict, st: dict, keys=None) -> list:
    """Certifications / languages / interests / awards / volunteering - parser distractors."""
    rng, out = ctx["rng"], []
    keys = keys or ["certifications", "publications", "awards", "volunteering",
                    "languages", "interests"]

    if "certifications" in keys and ctx["show_certifications"]:
        out.append(Paragraph(heading_for(ctx, "certifications"), st["section"]))
        for cert, issuer in rng.sample(cv_text.CERTIFICATIONS, rng.randint(1, 3)):
            year = rng.randint(cv_text.MIN_CERT_YEAR, cv_text.MAX_CERT_YEAR)
            out.append(Paragraph(f"{cert} — {issuer}, {year}", st["body"]))

    if "publications" in keys and ctx["show_publications"]:
        out.append(Paragraph(heading_for(ctx, "publications"), st["section"]))
        field = record["education"][0]["field"] or "Computer Science"
        for _ in range(rng.randint(1, 2)):
            year = record["education"][0]["end_year"] + rng.randint(0, 2)
            out.append(Paragraph(cv_text.make_publication(rng, field, year), st["small"]))

    if "awards" in keys and ctx["show_awards"]:
        out.append(Paragraph(heading_for(ctx, "awards"), st["section"]))
        for award in rng.sample(cv_text.AWARDS, rng.randint(1, 2)):
            out.append(Paragraph(f"{award}, {rng.randint(cv_text.MIN_CERT_YEAR, cv_text.MAX_CERT_YEAR)}",
                                 st["body"]))

    if "volunteering" in keys and ctx["show_volunteering"]:
        out.append(Paragraph(heading_for(ctx, "volunteering"), st["section"]))
        out.append(Paragraph(cv_text.make_volunteering(rng, record), st["body"]))

    if "languages" in keys and ctx["show_languages"]:
        out.append(Paragraph(heading_for(ctx, "languages"), st["section"]))
        # The CV itself is in English, so English is always listed and always strong.
        out.append(Paragraph(f"English – {rng.choice(['Native', 'Fluent', 'Bilingual', 'C2'])}",
                             st["body"]))
        others = [name for name in cv_text.LANGUAGE_NAMES if name != "English"]
        for lang in rng.sample(others, rng.randint(0, 2)):
            level = rng.choice([name for name in cv_text.LANGUAGE_LEVELS
                                if name not in ("Native", "Bilingual")])
            out.append(Paragraph(f"{lang} – {level}", st["body"]))

    if "interests" in keys and ctx["show_interests"]:
        out.append(Paragraph(heading_for(ctx, "interests"), st["section"]))
        out.append(Paragraph(", ".join(rng.sample(cv_text.INTERESTS, rng.randint(2, 4))), st["body"]))

    if ctx["show_references_line"] and "interests" in keys:
        out.append(Spacer(1, 6))
        out.append(Paragraph("References available upon request.", st["small"]))
    return out


def display_name(record: dict, ctx: dict) -> str:
    return record["name"].upper() if ctx["uppercase_name"] else record["name"]


def titled(ctx: dict, st: dict, key: str, content: list, rule: bool = False) -> list:
    if not content:
        return []
    block = [Paragraph(heading_for(ctx, key), st["section"])]
    if rule:
        block.append(HRFlowable(width="100%", thickness=0.6, color=ctx["accent"], spaceAfter=4))
    return block + content


def main_sections(record: dict, ctx: dict, st: dict, rule: bool = False,
                  keep: bool = True) -> list:
    """Experience + education in the order chosen for this document."""
    exp = titled(ctx, st, "experience", job_flowables(record, ctx, st, keep=keep), rule)
    edu = titled(ctx, st, "education", education_flowables(record, ctx, st), rule)
    return edu + exp if ctx["education_first"] else exp + edu


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

def tpl_classic_ats(record, ctx, fake):
    """Plain, boring, maximally ATS-friendly. The easy case."""
    st = make_styles(ctx)
    story = [Paragraph(display_name(record, ctx), st["name"]),
             Paragraph(ctx["contact"], st["contact"])]
    if ctx["show_summary"]:
        story += titled(ctx, st, "summary",
                        [Paragraph(cv_text.make_summary(ctx["rng"], record), st["body"])])
    if ctx["show_highlights"]:
        story += titled(ctx, st, "highlights", highlights_flowables(record, ctx, st))
    story += main_sections(record, ctx, st)
    story += titled(ctx, st, "skills", skills_flowables(record, ctx, st))
    if ctx["show_projects"]:
        story += titled(ctx, st, "projects", projects_flowables(record, ctx, st))
    story += extra_section_flowables(record, ctx, st)
    return story


def tpl_modern_banner(record, ctx, fake):
    """Full-width colour banner with the name reversed out in white."""
    st = make_styles(ctx)
    banner_name = ParagraphStyle("bn", parent=st["name"], textColor=colors.white, fontSize=21)
    banner_sub = ParagraphStyle("bs", parent=st["contact"],
                                 textColor=colors.Color(0.93, 0.93, 0.93), spaceAfter=0)
    banner = Table([[[Paragraph(display_name(record, ctx), banner_name),
                      Paragraph(ctx["contact"], banner_sub)]]],
                    colWidths=[ctx["pagesize"][0] - 1.0 * inch])
    banner.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), ctx["accent"]),
                                ("LEFTPADDING", (0, 0), (-1, -1), 14),
                                ("RIGHTPADDING", (0, 0), (-1, -1), 14),
                                ("TOPPADDING", (0, 0), (-1, -1), 12),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 12)]))
    story = [banner, Spacer(1, 10)]
    if ctx["show_summary"]:
        story += titled(ctx, st, "summary",
                        [Paragraph(cv_text.make_summary(ctx["rng"], record), st["body"])])
    if ctx["show_highlights"]:
        story += titled(ctx, st, "highlights", highlights_flowables(record, ctx, st))
    story += main_sections(record, ctx, st, rule=True)
    story += titled(ctx, st, "skills", skills_flowables(record, ctx, st), rule=True)
    if ctx["show_projects"]:
        story += titled(ctx, st, "projects", projects_flowables(record, ctx, st), rule=True)
    story += extra_section_flowables(record, ctx, st)
    return story


def _sidebar_layout(record, ctx, fake, dark: bool, side: str):
    """Shared builder for the sidebar templates - genuinely two reading orders."""
    st = make_styles(ctx)
    side_st = make_styles(ctx, on_dark=dark)
    rng = ctx["rng"]

    photo_bg = colors.Color(0.25, 0.28, 0.32) if dark else colors.Color(0.90, 0.90, 0.90)
    photo = Table([[Paragraph('<para align="center">PHOTO</para>', side_st["small"])]],
                  colWidths=[1.1 * inch], rowHeights=[1.1 * inch])
    photo.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), photo_bg),
                               ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
                               ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))

    side_content = []
    if rng.random() < 0.75:
        side_content += [photo, Spacer(1, 10)]
    if ctx["contact_multiline"]:   # empty when contact lives in the page header only
        side_content += titled(ctx, side_st, "contact",
                               [Paragraph(ctx["contact_multiline"], side_st["small"])])
    side_content += titled(ctx, side_st, "skills",
                           skills_flowables(record, ctx, side_st, on_dark=dark))
    side_content += extra_section_flowables(record, ctx, side_st,
                                            keys=["languages", "interests"])

    main_content = [Paragraph(display_name(record, ctx), st["name"])]
    if record["companies"]:
        main_content.append(Paragraph(record["companies"][0]["title"], st["title_line"]))
    if ctx["show_summary"]:
        main_content += titled(ctx, st, "summary",
                               [Paragraph(cv_text.make_summary(rng, record), st["body"])])
    main_content += main_sections(record, ctx, st, keep=False)
    if ctx["show_projects"]:
        main_content += titled(ctx, st, "projects", projects_flowables(record, ctx, st))
    main_content += extra_section_flowables(record, ctx, st,
                                            keys=["certifications", "publications",
                                                  "awards", "volunteering"])

    page_w = ctx["pagesize"][0] - 1.0 * inch
    side_w, main_w = page_w * 0.33, page_w * 0.67
    if side == "left":
        row, widths, shade_col = [side_content, main_content], [side_w, main_w], 0
    else:
        row, widths, shade_col = [main_content, side_content], [main_w, side_w], 1

    table = Table([row], colWidths=widths, splitByRow=1, splitInRow=1)
    bg = colors.HexColor("#2b3a45") if dark else colors.Color(0.945, 0.945, 0.945)
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (shade_col, 0), (shade_col, 0), bg),
        ("LEFTPADDING", (shade_col, 0), (shade_col, 0), 10),
        ("RIGHTPADDING", (shade_col, 0), (shade_col, 0), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (1 - shade_col, 0), (1 - shade_col, 0), 14),
    ]))
    return [table]


def tpl_left_sidebar_dark(record, ctx, fake):
    return _sidebar_layout(record, ctx, fake, dark=True, side="left")


def tpl_right_sidebar_light(record, ctx, fake):
    return _sidebar_layout(record, ctx, fake, dark=False, side="right")


def tpl_two_column_balanced(record, ctx, fake):
    """Even split - the classic multi-column parser killer."""
    st = make_styles(ctx)
    rng = ctx["rng"]
    left = [Paragraph(display_name(record, ctx), st["name"]),
            Paragraph(ctx["contact_multiline"], st["contact"])]
    if ctx["show_summary"]:
        left += titled(ctx, st, "summary", [Paragraph(cv_text.make_summary(rng, record), st["body"])])
    left += titled(ctx, st, "education", education_flowables(record, ctx, st))
    left += titled(ctx, st, "skills", skills_flowables(record, ctx, st))

    right = titled(ctx, st, "experience", job_flowables(record, ctx, st, keep=False))
    if ctx["show_projects"]:
        right += titled(ctx, st, "projects", projects_flowables(record, ctx, st))
    right += extra_section_flowables(record, ctx, st,
                                     keys=["certifications", "awards", "languages", "interests"])

    page_w = ctx["pagesize"][0] - 1.0 * inch
    table = Table([[left, right]], colWidths=[page_w * 0.46, page_w * 0.54],
                  splitByRow=1, splitInRow=1)
    table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                               ("RIGHTPADDING", (0, 0), (0, 0), 16),
                               ("LINEAFTER", (0, 0), (0, 0), 0.5, colors.Color(0.8, 0.8, 0.8))]))
    return [table]


def tpl_table_grid(record, ctx, fake):
    """Experience laid out as a bordered grid - dates in their own column."""
    st = make_styles(ctx)
    rng = ctx["rng"]
    story = [Paragraph(display_name(record, ctx), st["name"]),
             Paragraph(ctx["contact"], st["contact"])]
    if ctx["show_summary"]:
        story += titled(ctx, st, "summary", [Paragraph(cv_text.make_summary(rng, record), st["body"])])
    story.append(Paragraph(heading_for(ctx, "experience"), st["section"]))

    rows = []
    for index, company in enumerate(record["companies"]):
        dates = cv_text.format_date_range(rng, company["start_date"], company["end_date"],
                                          ctx["date_style"])
        cell = [Paragraph(f"<b>{company['title']}</b>", st["body"]), Paragraph(company["name"], st["small"])]
        lines = cv_text.make_bullets(rng, record["general_experience"]["domain"],
                               record["skills"], bullet_count(ctx, index), ctx["used_bullets"])
        if lines:
            cell.append(bullet_list(lines, ctx, st))
        rows.append([Paragraph(dates, st["small"]), cell])
    if rows:
        page_w = ctx["pagesize"][0] - 1.0 * inch
        grid = Table(rows, colWidths=[page_w * 0.26, page_w * 0.74],
                     splitByRow=1, splitInRow=1)
        grid.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.4, colors.Color(0.75, 0.75, 0.75)),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BACKGROUND", (0, 0), (0, -1), colors.Color(0.96, 0.96, 0.96)),
            ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(grid)

    story += titled(ctx, st, "education", education_flowables(record, ctx, st))
    story += titled(ctx, st, "skills", skills_flowables(record, ctx, st))
    if ctx["show_projects"]:
        story += titled(ctx, st, "projects", projects_flowables(record, ctx, st))
    story += extra_section_flowables(record, ctx, st)
    return story


def tpl_minimal_centered(record, ctx, fake):
    """Centred header, hairline rules, generous whitespace."""
    st = make_styles(ctx)
    centered_name = ParagraphStyle("cn", parent=st["name"], alignment=TA_CENTER, fontSize=22)
    centered_contact = ParagraphStyle("cc", parent=st["contact"], alignment=TA_CENTER)
    centered_section = ParagraphStyle("cs", parent=st["section"], alignment=TA_CENTER,
                                       fontSize=10, textColor=colors.dimgrey)
    story = [Spacer(1, 12),
             Paragraph(display_name(record, ctx).upper(), centered_name),
             Paragraph(ctx["contact"], centered_contact),
             HRFlowable(width="40%", thickness=0.7, color=colors.grey, spaceAfter=12,
                        hAlign="CENTER")]
    if ctx["show_summary"]:
        just = ParagraphStyle("ju", parent=st["body"], alignment=TA_JUSTIFY)
        story += [Paragraph(cv_text.make_summary(ctx["rng"], record), just), Spacer(1, 8)]

    def block(key, content):
        if not content:
            return []
        return [Paragraph(heading_for(ctx, key).upper(), centered_section),
                HRFlowable(width="100%", thickness=0.3, color=colors.Color(0.85, 0.85, 0.85),
                           spaceAfter=6)] + content

    experience = block("experience", job_flowables(record, ctx, st))
    education = block("education", education_flowables(record, ctx, st))
    story += education + experience if ctx["education_first"] else experience + education
    story += block("skills", skills_flowables(record, ctx, st))
    if ctx["show_projects"]:
        story += block("projects", projects_flowables(record, ctx, st))
    story += extra_section_flowables(record, ctx, st)
    return story


def tpl_academic_dense(record, ctx, fake):
    """Small type, dense, usually runs to two pages."""
    st = make_styles(ctx)
    for key in ("body", "bullet", "small"):
        st[key] = ParagraphStyle(f"d{key}", parent=st[key], fontSize=8.5, leading=10.5)
    st["section"] = ParagraphStyle("dsec", parent=st["section"], fontSize=10, spaceBefore=7)
    ctx = dict(ctx, bullets_recent=max(4, ctx["bullets_recent"]),
               show_publications=True, show_projects=True, show_certifications=True)

    story = [Paragraph(display_name(record, ctx), st["name"]),
             Paragraph(ctx["contact"], st["contact"])]
    if ctx["show_summary"]:
        story += titled(ctx, st, "summary",
                        [Paragraph(cv_text.make_summary(ctx["rng"], record), st["body"])])
    story += titled(ctx, st, "education", education_flowables(record, ctx, st))
    story += titled(ctx, st, "experience", job_flowables(record, ctx, st))
    story += titled(ctx, st, "skills", skills_flowables(record, ctx, st))
    story += titled(ctx, st, "projects", projects_flowables(record, ctx, st))
    story += extra_section_flowables(record, ctx, st)
    return story


def tpl_europass_style(record, ctx, fake):
    """EU-style form: a label column on the left, values on the right."""
    st = make_styles(ctx)
    rng = ctx["rng"]
    label = ParagraphStyle("lb", parent=st["small"], fontName=ctx["font_bold"],
                            textColor=ctx["accent"], alignment=TA_RIGHT)
    rows = [[Paragraph("PERSONAL INFORMATION", label),
             [Paragraph(f"<b>{display_name(record, ctx)}</b>", st["body"]),
              Paragraph(ctx["contact_multiline"], st["small"])]]]

    if ctx["show_summary"]:
        rows.append([Paragraph("PROFILE", label),
                     [Paragraph(cv_text.make_summary(rng, record), st["body"])]])

    for index, company in enumerate(record["companies"]):
        dates = cv_text.format_date_range(rng, company["start_date"], company["end_date"],
                                          ctx["date_style"])
        cell = [Paragraph(f"<b>{company['title']}</b> — {company['name']}", st["body"]),
                Paragraph(dates, st["small"])]
        lines = cv_text.make_bullets(rng, record["general_experience"]["domain"],
                               record["skills"], bullet_count(ctx, index), ctx["used_bullets"])
        cell += [Paragraph(f"{ctx['bullet_char']} {line}", st["bullet"]) for line in lines]
        rows.append([Paragraph("WORK EXPERIENCE" if index == 0 else "", label), cell])

    for index, entry in enumerate(record["education"]):
        degree = entry["level"] if entry["field"] is None else f"{entry['level']}, {entry['field']}"
        dates = cv_text.format_year_range(rng, entry["start_year"], entry["end_year"], ctx["date_style"])
        rows.append([Paragraph("EDUCATION AND TRAINING" if index == 0 else "", label),
                     [Paragraph(f"<b>{degree}</b>", st["body"]),
                      Paragraph(f"{entry['institution']}  |  {dates}", st["small"])]])

    skills = skills_flowables(record, ctx, st)
    if skills:
        rows.append([Paragraph("PERSONAL SKILLS", label), skills])
    extras = extra_section_flowables(record, ctx, st, keys=["languages", "certifications"])
    if extras:
        rows.append([Paragraph("ADDITIONAL INFORMATION", label), extras])

    page_w = ctx["pagesize"][0] - 1.0 * inch
    table = Table(rows, colWidths=[page_w * 0.28, page_w * 0.72],
                  splitByRow=1, splitInRow=1)
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.Color(0.85, 0.85, 0.85)),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (0, -1), 12),
    ]))
    return [table]


def tpl_compact_boxed(record, ctx, fake):
    """Each section sits in its own shaded panel."""
    st = make_styles(ctx)

    def panel(key, content, shaded=True):
        if not content:
            return []
        inner = [Paragraph(heading_for(ctx, key), st["section"])] + content
        table = Table([[inner]], colWidths=[ctx["pagesize"][0] - 1.0 * inch],
                  splitByRow=1, splitInRow=1)
        style = [("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                 ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]
        if shaded:
            style.append(("BACKGROUND", (0, 0), (-1, -1), colors.Color(0.955, 0.955, 0.955)))
        table.setStyle(TableStyle(style))
        return [table, Spacer(1, 7)]

    story = [Paragraph(display_name(record, ctx), st["name"]),
             Paragraph(ctx["contact"], st["contact"])]
    if ctx["show_summary"]:
        story += panel("summary", [Paragraph(cv_text.make_summary(ctx["rng"], record), st["body"])])
    story += panel("experience", job_flowables(record, ctx, st, keep=False), shaded=False)
    story += panel("education", education_flowables(record, ctx, st))
    story += panel("skills", skills_flowables(record, ctx, st))
    if ctx["show_projects"]:
        story += panel("projects", projects_flowables(record, ctx, st), shaded=False)
    extras = extra_section_flowables(record, ctx, st)
    if extras:
        story += [Spacer(1, 4)] + extras
    return story


TEMPLATES = [
    ("classic_ats", tpl_classic_ats, "easy", 16),
    ("modern_banner", tpl_modern_banner, "easy", 13),
    ("minimal_centered", tpl_minimal_centered, "medium", 11),
    ("compact_boxed", tpl_compact_boxed, "medium", 10),
    ("academic_dense", tpl_academic_dense, "medium", 9),
    ("table_grid", tpl_table_grid, "hard", 11),
    ("two_column_balanced", tpl_two_column_balanced, "hard", 11),
    ("left_sidebar_dark", tpl_left_sidebar_dark, "hard", 9),
    ("right_sidebar_light", tpl_right_sidebar_light, "hard", 6),
    ("europass_style", tpl_europass_style, "hard", 4),
]
