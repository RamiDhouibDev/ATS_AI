"""Unit tests for the extraction layer.

These cover the parsing rules directly. End-to-end accuracy against the corpus
is measured by `python -m layer1_extraction.code.evaluate`, not here - this suite is meant
to stay fast and to pin the behaviours that were expensive to get right.
"""

import pytest

from layer1_extraction.code import vocab
from layer1_extraction.code.ats_parser import (
    canonical_skill_in, find_date_range, parse_companies, parse_education, parse_skills,
    score_confidence, split_sections, total_years_from,
)
from layer1_extraction.code.schema import CVRecord, Company, LayoutSignals


class TestDateParsing:
    @pytest.mark.parametrize("text,expected", [
        ("Mar 2019 - Aug 2021", ("2019-03", "2021-08")),
        ("March 2019 to August 2021", ("2019-03", "2021-08")),
        ("03/2019 - 08/2021", ("2019-03", "2021-08")),
        ("2019-03 - 2021-08", ("2019-03", "2021-08")),
        ("Mar '19 - Aug '21", ("2019-03", "2021-08")),
        ("2019 - 2021", ("2019-01", "2021-01")),
    ])
    def test_every_rendered_format(self, text, expected):
        assert find_date_range(text) == expected

    @pytest.mark.parametrize("word", ["Present", "Current", "Now"])
    def test_open_ended_role_has_no_end(self, word):
        start, end = find_date_range(f"Jan 2021 - {word}")
        assert start == "2021-01"
        assert end is None

    def test_no_date_returns_none(self):
        assert find_date_range("Senior Engineer, Acme") is None


class TestSections:
    def test_headings_bucket_following_lines(self):
        sections = split_sections(["Jane Doe", "EXPERIENCE", "Engineer, Acme", "SKILLS", "Python"])
        assert sections["experience"] == ["Engineer, Acme"]
        assert sections["skills"] == ["Python"]
        assert sections["_preamble"] == ["Jane Doe"]

    def test_heading_synonyms_map_to_one_key(self):
        for heading in ("Employment History", "Career History", "Work Experience"):
            assert vocab.heading_for(heading) == "experience"

    def test_form_label_on_same_line_is_split(self):
        sections = split_sections(["WORK EXPERIENCE Mobile Engineer - Uber"])
        assert sections["experience"] == ["Mobile Engineer - Uber"]

    def test_categorised_skills_line_is_not_a_heading(self):
        """"Languages: Python, SQL" is skills content, not a languages section."""
        heading, _ = vocab.split_heading_prefix("Languages: Python, SQL")
        assert heading is None


class TestSkills:
    def test_comma_separated(self):
        names = {skill.name for skill in parse_skills(["Python, JavaScript, AWS"])}
        assert names == {"Python", "JavaScript", "AWS"}

    def test_years_are_read_when_stated(self):
        skills = parse_skills(["Python (7 yrs), AWS (3 years)"])
        assert {skill.name: skill.years for skill in skills} == {"Python": 7.0, "AWS": 3.0}

    def test_proficiency_words_are_stripped(self):
        names = {skill.name for skill in parse_skills(["Python - Expert", "AWS - Intermediate"])}
        assert names == {"Python", "AWS"}

    def test_category_prefix_is_ignored(self):
        names = {skill.name for skill in parse_skills(["Languages: Python, SQL"])}
        assert names == {"Python", "SQL"}

    def test_aliases_canonicalise(self):
        names = {skill.name for skill in parse_skills(["JS, K8s, Postgres, sklearn"])}
        assert names == {"JavaScript", "Kubernetes", "PostgreSQL", "Scikit-learn"}

    def test_hyphenated_names_survive(self):
        assert {skill.name for skill in parse_skills(["Scikit-learn"])} == {"Scikit-learn"}

    def test_unknown_tokens_are_dropped(self):
        assert parse_skills(["Underwater basket weaving"]) == []


class TestCompanies:
    def test_title_and_employer_on_one_line(self):
        companies = parse_companies(["Senior Engineer, Acme Corp", "Jan 2020 - Mar 2022"])
        assert len(companies) == 1
        assert companies[0].name == "Acme Corp"
        assert companies[0].title == "Senior Engineer"
        assert companies[0].start_date == "2020-01"

    def test_employer_first_ordering(self):
        companies = parse_companies(["Acme Corp - Data Scientist", "2019 - 2021"])
        assert companies[0].name == "Acme Corp"
        assert companies[0].title == "Data Scientist"

    def test_table_layout_puts_dates_above_title(self):
        """Grid templates emit the date column, then title, then employer."""
        companies = parse_companies([
            "Oct '16 - Present", "Platform Engineer", "Cobalt Applications",
            "- did some things",
        ])
        assert len(companies) == 1
        assert companies[0].name == "Cobalt Applications"
        assert companies[0].start_date == "2016-10"
        assert companies[0].end_date is None

    def test_bullets_are_never_read_as_jobs(self):
        companies = parse_companies(["- Mentored 4 junior engineers at the company"])
        assert companies == []

    def test_known_employer_gets_its_tier(self):
        """3 is world class, 1 is unknown - higher is better."""
        companies = parse_companies(["Engineer, Google", "2020 - 2022"])
        assert companies[0].tier == 3

    def test_unknown_employer_defaults_to_the_bottom_tier(self):
        """An unrecognised name is not evidence of quality."""
        companies = parse_companies(["Engineer, Blue Orbit Labs", "2020 - 2022"])
        assert companies[0].tier == 1


class TestAwkwardLayouts:
    """Behaviours that cost real accuracy and are easy to regress."""

    def test_employer_line_carrying_dates_and_location(self):
        """"Adobe | February 2024 to Now | Berlin" - rejecting dated lines lost the job."""
        companies = parse_companies(["Platform Engineer",
                                     "Adobe | February 2024 to Now | North Jorge"])
        assert len(companies) == 1
        assert companies[0].name == "Adobe"
        assert companies[0].start_date == "2024-02"

    def test_skill_rows_split_across_lines_are_paired(self):
        """Skills tables and rating bars emit the name and years as separate lines."""
        skills = parse_skills([], ["Datadog", "15 yrs", "Grafana", "18 yrs"])
        assert {skill.name: skill.years for skill in skills} == {"Datadog": 15.0, "Grafana": 18.0}

    def test_bullet_glyph_rendered_as_a_letter(self):
        """Symbol fonts map their bullet to an arbitrary letter, e.g. "n Vue.js"."""
        assert canonical_skill_in("n Vue.js") == "Vue.js"
        assert canonical_skill_in("Vue.js") == "Vue.js"

    def test_dropping_a_leading_token_cannot_invent_a_skill(self):
        assert canonical_skill_in("Senior Engineer") is None
        assert canonical_skill_in("Spring Boot") == "Spring Boot"

    def test_wrapped_skills_line_is_rejoined(self):
        """Narrow columns wrap mid-entry: "GCP (2" / "years)"."""
        skills = parse_skills(["Java (6 years), GCP (2", "years)"])
        assert {skill.name for skill in skills} >= {"Java", "GCP"}

    def test_prose_mining_only_names_known_skills(self):
        found = vocab.skills_mentioned_in(
            "Migrated 10 services to Kubernetes, cutting spend by 45%.")
        assert found == {"Kubernetes"}

    def test_prose_mining_respects_word_boundaries(self):
        assert vocab.skills_mentioned_in("Wrote Gopher tooling and Rustic scripts") == set()


class TestEducation:
    def test_degree_field_and_years(self):
        entries = parse_education(["Master, Computer Science", "Some University | 2015 - 2017"])
        assert entries[0].level == "Master"
        assert entries[0].field_of_study == "Computer Science"
        assert (entries[0].start_year, entries[0].end_year) == (2015, 2017)

    def test_highest_degree_comes_first(self):
        entries = parse_education([
            "Bachelor, Physics", "A University | 2010 - 2013",
            "PhD, Physics", "B University | 2013 - 2017",
        ])
        assert entries[0].level == "PhD"

    def test_abbreviations_are_recognised(self):
        assert parse_education(["BSc Computer Science"])[0].level == "Bachelor"
        assert parse_education(["Ph.D. in Physics"])[0].level == "PhD"


class TestTenure:
    def test_sums_across_roles(self):
        companies = [
            Company(name="A", start_date="2018-01", end_date="2020-01"),
            Company(name="B", start_date="2020-01", end_date="2021-01"),
        ]
        assert total_years_from(companies) == 3.0

    def test_open_ended_role_runs_to_reference_date(self):
        companies = [Company(name="A", start_date="2024-09", end_date=None)]
        assert total_years_from(companies, reference="2026-09") == 2.0


class TestConfidence:
    def test_image_only_document_scores_zero(self):
        signals = LayoutSignals(char_count=0)
        assert score_confidence(CVRecord(), signals, {}) == 0.0

    def test_graduate_with_no_jobs_is_not_penalised(self):
        """An experience section with no dates means no jobs to find, not a failure."""
        record = CVRecord(name="A B", education=[object()], skills=[object()] * 3)
        signals = LayoutSignals(char_count=900)
        sections = {"experience": [], "education": [], "skills": []}
        assert score_confidence(record, signals, sections,
                                experience_lines=["Seeking a first role."]) > 0.9

    def test_multi_column_is_discounted(self):
        record = CVRecord(name="A B", education=[object()], skills=[object()] * 3,
                          companies=[Company(name="X", start_date="2020-01")] * 2)
        sections = {"experience": [], "education": [], "skills": []}
        single = score_confidence(record, LayoutSignals(char_count=900), sections)
        multi = score_confidence(record, LayoutSignals(char_count=900, max_columns=2), sections)
        assert multi < single


class TestVocab:
    def test_company_tier_ignores_legal_suffixes(self):
        assert vocab.company_tier("Google LLC") == 3
        assert vocab.company_tier("Google, Inc.") == 3

    def test_unknown_company_is_the_bottom_tier(self):
        assert vocab.company_tier("Nonexistent Holdings") == 1
