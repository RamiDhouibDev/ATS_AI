"""Prestige tiers for employers and universities.

ONE SCALE, USED EVERYWHERE
    3  world class    - the name alone carries weight anywhere
    2  good           - well known and respected, not a household name
    1  unknown or below average

    Higher is better, so the number reads the way a rating does. Anything not
    in these tables is tier 1 by default: an unrecognised name is not evidence
    of quality, and a scoring rule that treated "unknown" as neutral would
    reward obscurity.

WHY A TABLE AND NOT THE NAME
    A name is a string with no order - "Google" and "Acme Ltd" are equally far
    apart from a model's point of view, and a vocabulary of every employer on
    earth is unlearnable. A tier is three ordered buckets the model can
    actually use, and the mapping is a lookup anyone can read and argue with.

    Names are matched case-insensitively with punctuation and common suffixes
    (Inc, Ltd, GmbH) stripped - see `tier_for`.
"""

from __future__ import annotations

import re

UNKNOWN = 1

# Employers. Tier 3 is the set whose name alone moves a hiring conversation.
COMPANY_TIERS = {
    # 3 - world class
    "Google": 3, "Meta": 3, "Apple": 3, "Amazon": 3, "Microsoft": 3,
    "Netflix": 3, "OpenAI": 3, "NVIDIA": 3, "DeepMind": 3,
    # 2 - good, well known
    "Uber": 2, "Airbnb": 2, "Salesforce": 2, "Adobe": 2, "IBM": 2,
    "Oracle": 2, "SAP": 2, "Spotify": 2, "Stripe": 2, "Shopify": 2,
    "Atlassian": 2, "Datadog": 2, "Snowflake": 2, "Cloudflare": 2, "Palantir": 2,
    # 1 - everything else in the corpus
    "BrightPath Solutions": 1, "Nova Tech Group": 1, "Summit Digital": 1,
    "Clearwater Systems": 1, "Meridian Software": 1, "Ironwood Labs": 1,
    "Vantage Analytics": 1, "Beacon Interactive": 1, "Harbor Point Tech": 1,
    "Stonebridge IT": 1, "Lumen Works": 1, "Cobalt Systems": 1,
    "Foxglove Media": 1, "Redwood Data": 1, "Keystone Apps": 1,
}

# Universities. Same three buckets, same reasoning: the institution is a weak
# signal, but it is not no signal, and a tier is the only form of it a model can
# use without memorising thousands of names.
UNIVERSITY_TIERS = {
    # 3 - world class
    "Massachusetts Institute of Technology": 3, "Stanford University": 3,
    "Carnegie Mellon University": 3, "University of California, Berkeley": 3,
    "University of Oxford": 3, "University of Cambridge": 3,
    "ETH Zurich": 3, "California Institute of Technology": 3,
    "Harvard University": 3, "Tsinghua University": 3,
    # 2 - good, well regarded
    "University of Toronto": 2, "Technical University of Munich": 2,
    "University of Edinburgh": 2, "Delft University of Technology": 2,
    "KU Leuven": 2, "University of Illinois Urbana-Champaign": 2,
    "Georgia Institute of Technology": 2, "University of Texas at Austin": 2,
    "University of Waterloo": 2, "KTH Royal Institute of Technology": 2,
    "Politecnico di Milano": 2, "Universidad Politecnica de Madrid": 2,
    "University of Manchester": 2, "Aalto University": 2,
    "National University of Singapore": 2,
    # 1 - regional and unremarkable, the bulk of any real applicant pool
    "Riverside State University": 1, "Northgate University": 1,
    "Fairview Institute of Technology": 1, "Lakeshore University": 1,
    "Eastbrook State University": 1, "Pinehurst University": 1,
    "Ashford Technical University": 1, "Granite Bay University": 1,
    "Westfield State University": 1, "Cedar Hollow University": 1,
    "Milbrook Institute of Technology": 1, "Sandalwood University": 1,
    "Thornfield University": 1, "Copperfield State University": 1,
    "Braxton University": 1, "Halloway Institute of Technology": 1,
    "Marchmont University": 1, "Ellsworth State University": 1,
    "Vernon Hills University": 1, "Oakmere University": 1,
}

# Legal-form suffixes carry no information and vary between CVs for the same
# employer, so they are dropped before matching.
_SUFFIXES = re.compile(
    r"\b(inc|llc|ltd|limited|corp|corporation|gmbh|ag|sa|bv|plc|co|company|"
    r"group|holdings|technologies|labs)\b", re.I)
_PUNCTUATION = re.compile(r"[^\w\s]")


def normalise(name: str) -> str:
    """Lowercase, strip punctuation and legal suffixes, squash whitespace."""
    without_punctuation = _PUNCTUATION.sub(" ", name or "")
    return " ".join(_SUFFIXES.sub(" ", without_punctuation).lower().split())


_COMPANY_LOOKUP = {normalise(name): tier for name, tier in COMPANY_TIERS.items()}
_UNIVERSITY_LOOKUP = {normalise(name): tier for name, tier in UNIVERSITY_TIERS.items()}


def company_tier(name: str) -> int:
    """Tier for an employer; unrecognised names are tier 1."""
    return _COMPANY_LOOKUP.get(normalise(name), UNKNOWN)


def university_tier(name: str) -> int:
    """Tier for an institution; unrecognised names are tier 1.

    High schools land here too and are correctly tier 1 - the table holds
    universities, and a school name is not a prestige signal.
    """
    return _UNIVERSITY_LOOKUP.get(normalise(name), UNKNOWN)


def names_by_tier(table: dict[str, int], tier: int) -> list[str]:
    """The names at one tier, sorted - the generator samples from these."""
    return sorted(name for name, value in table.items() if value == tier)


if __name__ == "__main__":
    for label, table in (("companies", COMPANY_TIERS), ("universities", UNIVERSITY_TIERS)):
        print(f"{label}:")
        for tier in (3, 2, 1):
            names = names_by_tier(table, tier)
            print(f"  tier {tier}  {len(names):2d}  {', '.join(names[:3])}...")
    print(f"\nunknown name -> {company_tier('Some Random Startup Ltd')}")
    print(f"suffix ignored -> Google Inc = {company_tier('Google Inc.')}")
