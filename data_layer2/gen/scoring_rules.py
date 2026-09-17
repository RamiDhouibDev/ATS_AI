"""Role-conditioned labelling formula for Layer 2.

WHAT THIS IS
    The ground truth Layer 2 learns from. Layer 1 asks "what does this CV say?";
    Layer 2 asks "how well does this CV fit *this* job?". A ten-year Django
    specialist is a strong hire for a backend role and a weak one for a data
    science role, and the labels must reflect that or the model has nothing to
    learn beyond "rank people by seniority".

    Three of the four sections are therefore functions of the (candidate, job)
    pair. `companies_score` is deliberately not: employer prestige is always a
    plus and no posting can switch it off, so it reads the candidate alone. What
    a posting varies is how heavily it *weights* that section. This is a design
    decision, not an oversight - see the note on its own docstring.

WHY A FORMULA AND NOT HUMAN LABELS
    Hand-scoring 60,000 pairs is not possible, and a formula gives an exactly
    known answer to compare predictions against. The trade-off is that the model
    can in principle recover the formula rather than generalise, which is why
    noise is added in `score_pair` and why the curves below are smooth rather
    than step functions.

WHY THE MODEL IS STILL WORTH TRAINING
    In production Layer 2 never sees these clean fields - it sees Layer 1's
    output, which can be missing a skill, a tier, or a date. A trained network
    degrades gracefully on partial input; this formula would just compute a
    confidently wrong number. The formula is the teacher, not the product.

SHAPE OF EVERY SECTION SCORE
    All four return 0-100 and are built so that "meets the requirement" lands
    around 80-85, leaving headroom above for genuinely exceptional candidates.
    That keeps the four sections on a comparable scale before they are combined
    with the job's own weights.
"""

from __future__ import annotations

import math
import random

DOMAINS = [
    "Software Development", "Data/AI/ML", "DevOps/Infrastructure",
    "Product/Design", "Data Analysis", "QA/Testing",
]

# How much experience in one domain counts toward another, 0-1.
#
# This is the single most opinionated table in the file. It encodes that a
# backend engineer is most of the way to a DevOps role (0.60) but barely closer
# to product management than a stranger (0.35). Without it, "years of
# experience" would be domain-blind and a 20-year designer would outrank a
# 5-year engineer for an engineering job.
#
# Stored one way round only; `domain_similarity` looks up both orderings, so
# transfer is treated as mutual. Anything unlisted falls back to 0.25, i.e. a
# quarter of the experience carries over - not zero, because general seniority
# is worth something everywhere.
_SIMILARITY = {
    ("Software Development", "DevOps/Infrastructure"): 0.60,   # shares tooling, deployment, on-call
    ("Software Development", "Data/AI/ML"): 0.50,              # shared engineering, different maths
    ("Software Development", "QA/Testing"): 0.55,              # same codebase, different objective
    ("Software Development", "Data Analysis"): 0.40,
    ("Software Development", "Product/Design"): 0.35,          # adjacent, rarely interchangeable
    ("Data/AI/ML", "Data Analysis"): 0.70,                     # closest pair here: same data, same SQL
    ("Data/AI/ML", "DevOps/Infrastructure"): 0.35,
    ("Data/AI/ML", "QA/Testing"): 0.30,
    ("Data/AI/ML", "Product/Design"): 0.25,
    ("DevOps/Infrastructure", "QA/Testing"): 0.40,             # both automate pipelines
    ("DevOps/Infrastructure", "Data Analysis"): 0.30,
    ("DevOps/Infrastructure", "Product/Design"): 0.20,         # furthest apart in this taxonomy
    ("Product/Design", "Data Analysis"): 0.40,
    ("Product/Design", "QA/Testing"): 0.30,
    ("Data Analysis", "QA/Testing"): 0.30,
}

# Ordered so that subtracting ranks gives a signed "levels above/below" gap.
DEGREE_RANK = {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3}

# Treated as interchangeable when a posting asks for a specific technical
# degree: a physics graduate applying for a CS role is not penalised the way a
# communications graduate is.
TECHNICAL_FIELDS = {
    "Computer Science", "Data Science", "Software Engineering",
    "Electrical Engineering", "Mathematics", "Physics", "Information Systems",
}

# Prestige tiers, matching data_layer1/gen/tiers.py: 3 is world class, 1 is
# unknown or below average. The gap between 3 and 2 (35pts) is deliberately
# larger than between 2 and 1 (30pts) but not overwhelming - a long tenure at a
# good firm should still beat a brief stint at a famous one.
TIER_POINTS = {3: 100, 2: 65, 1: 35}
UNKNOWN_TIER = 1

# What the university is worth in `education_score`, kept deliberately small
# next to the 70 points that degree level moves - see that function.
INSTITUTION_BONUS = {3: 6.0, 2: 2.0, 1: 0.0}


def domain_similarity(candidate_domain: str, job_domain: str) -> float:
    """Transfer coefficient between two domains, 1.0 for an exact match."""
    if candidate_domain == job_domain:
        return 1.0
    key = (candidate_domain, job_domain)
    # `or` rather than a chained .get: a stored 0.0 would be meaningless here,
    # so falling through on a falsy hit is safe and keeps the lookup symmetric.
    return _SIMILARITY.get(key) or _SIMILARITY.get((job_domain, candidate_domain), 0.25)


def clip(value: float) -> float:
    """Hold a score inside 0-100 after bonuses, penalties and noise."""
    return max(0.0, min(100.0, value))


def education_score(candidate: dict, job: dict) -> float:
    """Degree level against the job's requirement, then field, then institution.

    The three parts are deliberately unequal, and in that order:

        level        70 points of swing   (30 for two levels short, 100 for two over)
        field        18 points            (-8 unrelated, +10 exact match)
        institution   6 points            (0 unknown, +6 world class)

    What you studied to, and in what, outweighs where by an order of magnitude.
    A world-class university cannot lift a candidate over someone a whole degree
    level above them, and it is never the reason one applicant beats another on
    its own - it only separates otherwise equal ones.
    """
    education = candidate["education"]
    if not education:
        return 20.0     # no degree listed at all; not zero, since many good engineers have none

    # education[0] is the highest degree - Layer 1's schema sorts it that way.
    top = education[0]

    # Signed distance in levels: +1 means one level above what was asked for.
    gap = DEGREE_RANK.get(top["level"], 0) - DEGREE_RANK.get(job["preferred_education"], 1)
    if gap >= 0:
        # Meeting the bar earns most of the score. Exceeding it adds a little,
        # capped at two levels: a PhD for a role wanting a bachelor's is not
        # twice as good a hire, and often not a better one at all.
        score = 82.0 + min(gap, 2) * 9.0
    elif gap == -1:
        score = 55.0    # one level short - a real gap, rarely disqualifying
    else:
        score = 30.0    # two or more levels short

    # Field fit only applies when the posting actually names a preferred field;
    # roughly half of them leave it open.
    wanted_field = job.get("preferred_field")
    field = top.get("field")
    if wanted_field:
        if field == wanted_field:
            score += 10
        elif field in TECHNICAL_FIELDS and wanted_field in TECHNICAL_FIELDS:
            score += 4      # adjacent technical degree: most of the credit
        else:
            score -= 8      # unrelated field, e.g. communications for an ML role

    # Institution, last and smallest. Tier 1 is both "unknown" and "below
    # average", so it earns nothing rather than losing anything - not having
    # heard of somewhere is not evidence against the candidate.
    score += INSTITUTION_BONUS.get(top.get("tier"), 0.0)
    return clip(score)


def relevant_experience_score(candidate: dict, job: dict) -> float:
    """Years of experience, discounted when it comes from another domain.

    The discount is what makes this role-conditioned. Without it the score would
    be a pure function of the candidate and identical across every posting.
    """
    general = candidate["general_experience"]
    transfer = domain_similarity(general["domain"], job["domain"])
    effective = general["total_years"] * transfer

    # Guard against a division blow-up on a hypothetical zero-requirement job.
    required = max(1.0, job["required_experience_years"])
    ratio = effective / required

    if ratio >= 1:
        # Saturating curve above the bar: the first year of surplus is worth far
        # more than the tenth. Exponential decay approaches but never reaches
        # 100, so 85 is "meets exactly" and 100 is unattainable in practice -
        # which is what stops long tenure from dominating every other section.
        return clip(85 + 15 * (1 - math.exp(-(ratio - 1))))

    # Below the bar, a gentle concave curve (exponent < 1) rather than a linear
    # drop: someone with 4 of 5 required years is much closer to hireable than
    # a straight proportion would suggest.
    return clip(85 * ratio ** 0.85)


def stack_experience_score(candidate: dict, job: dict) -> float:
    """Weighted coverage of the stack this job actually asks for.

    Only the job's required skills count. A candidate with thirty unrelated
    technologies scores zero here, which is the intended behaviour: breadth is
    not fit.
    """
    required = job["required_skills"]
    if not required:
        return 50.0     # no stated stack; neutral rather than punishing

    held = {skill["name"]: skill["years"] for skill in candidate["skills"]}
    total_weight = sum(requirement["weight"] for requirement in required)

    earned = 0.0
    for requirement in required:
        years = held.get(requirement["name"], 0.0)
        if years <= 0:
            coverage = 0.0                      # doesn't have it at all
        else:
            # Depth against the years this posting asks for, capped at 1.0 -
            # ten years of Python does not beat five when five were wanted.
            coverage = min(1.0, years / max(1.0, requirement["min_years"]))
            # Floor of 0.45: having the skill but less experience than asked is
            # worth most of the credit, because the alternative (not knowing it)
            # is a categorically different problem for a hiring manager.
            coverage = 0.45 + 0.55 * coverage
        # Per-skill importance: postings mark some requirements as must-haves
        # and others as nice-to-haves.
        earned += coverage * requirement["weight"]

    # Normalise by total weight so postings asking for three skills and seven
    # skills stay on the same 0-100 scale.
    return clip(100 * earned / total_weight)


def companies_score(candidate: dict, job: dict) -> float:
    """Tenure-weighted employer prestige.

    Big-tech experience is always a plus, never a per-posting preference: it is
    one of the four sections being scored, so a job cannot switch it off. What a
    job *can* do is weight this section more or less heavily against the other
    three, which is handled by `weights["companies"]`.

    Weighted by tenure rather than counted per employer, so six years at a
    tier-1 firm outranks three one-year stints that happen to include one.
    """
    companies = candidate["companies"]
    if not companies:
        return 18.0     # no employment history - graduates land here

    tenure = sum(employer["years"] for employer in companies) or 1.0
    return clip(sum(TIER_POINTS.get(employer["tier"], TIER_POINTS[UNKNOWN_TIER]) * employer["years"]
                    for employer in companies) / tenure)


def score_pair(candidate: dict, job: dict, rng: random.Random,
               noise_sigma: float = 3.0) -> dict:
    """The four section scores. There is no fifth.

    An overall score is not stored, because it is not a label. It is a weighted
    sum of these four, and the weights are chosen at ranking time - by the
    posting, or by whoever is moving the sliders in the UI. Storing one would
    freeze somebody's weighting into the ground truth and invite a model to
    learn arithmetic it can simply be given.
    """
    sections = {
        "education_score": education_score(candidate, job),
        "relevant_experience_score": relevant_experience_score(candidate, job),
        "stack_experience_score": stack_experience_score(candidate, job),
        "companies_score": companies_score(candidate, job),
    }

    # Gaussian jitter, standing in for the inconsistency of a human rater.
    #
    # Drawn per pair for the three sections that depend on the posting. NOT for
    # `companies_score`: it reads only the candidate, so a fresh draw per posting
    # would show one employment history scored 58 for one job and 71 for the
    # next with nothing whatsoever having changed - variation no model can
    # predict and no reader can explain. Keyed on the candidate id instead, so
    # the section is jittered once and then stays put across every posting.
    jitter = {section: rng.gauss(0, noise_sigma)
              for section in sections if section != "companies_score"}
    jitter["companies_score"] = random.Random(candidate["id"]).gauss(0, noise_sigma)

    return {section: round(clip(score + jitter[section]))
            for section, score in sections.items()}
