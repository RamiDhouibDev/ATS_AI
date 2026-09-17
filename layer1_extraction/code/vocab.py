"""Recognition vocabulary for the extraction layer.

This is the parser's own dictionary, deliberately kept separate from the
generator's taxonomy in data_gen. A production ATS maintains its skill list
and employer tier table independently of whatever a CV happens to contain, and
sharing the generator's copy would leak labels into the evaluation. Where this
list is incomplete, evaluation should show it as reduced recall - that is the
honest signal.
"""

from __future__ import annotations

import re

# Canonical skill names the parser can recognise.
CANONICAL_SKILLS = {
    "Python", "JavaScript", "TypeScript", "Java", "C++", "C#", "Go", "Rust", "Ruby",
    "Swift", "Kotlin", "Scala", "SQL", "Bash", "React", "Vue.js", "Angular", "Next.js",
    "Node.js", "Django", "FastAPI", "Spring Boot", "REST APIs", "GraphQL", "gRPC",
    "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform", "Ansible", "Linux",
    "CI/CD", "Jenkins", "GitHub Actions", "Prometheus", "Grafana", "Datadog", "Nginx",
    "PyTorch", "TensorFlow", "Scikit-learn", "Pandas", "NumPy", "Spark", "Airflow",
    "dbt", "Kafka", "Snowflake", "Databricks", "BigQuery", "Redshift", "MLflow",
    "Hugging Face", "LangChain", "PostgreSQL", "MySQL", "MongoDB", "Redis",
    "Elasticsearch", "DynamoDB", "Tableau", "Power BI", "Looker", "Excel", "Figma",
    "Sketch", "Selenium", "Cypress", "Playwright", "pytest", "Postman", "Jira",
    "Git", "Confluence",
}

# Real CVs write these instead of the canonical name.
SKILL_ALIASES = {
    "js": "JavaScript", "ts": "TypeScript", "node": "Node.js", "nodejs": "Node.js",
    "reactjs": "React", "react.js": "React", "vue": "Vue.js", "vuejs": "Vue.js",
    "nextjs": "Next.js", "k8s": "Kubernetes", "kube": "Kubernetes",
    "postgres": "PostgreSQL", "psql": "PostgreSQL", "postgresql": "PostgreSQL",
    "mongo": "MongoDB", "elastic": "Elasticsearch", "es": "Elasticsearch",
    "sklearn": "Scikit-learn", "scikit learn": "Scikit-learn", "sci-kit learn": "Scikit-learn",
    "tf": "TensorFlow", "torch": "PyTorch", "pytorch lightning": "PyTorch",
    "amazon web services": "AWS", "google cloud": "GCP", "google cloud platform": "GCP",
    "microsoft azure": "Azure", "gh actions": "GitHub Actions", "github action": "GitHub Actions",
    "ci cd": "CI/CD", "cicd": "CI/CD", "continuous integration": "CI/CD",
    "rest": "REST APIs", "rest api": "REST APIs", "restful apis": "REST APIs",
    "powerbi": "Power BI", "ms excel": "Excel", "microsoft excel": "Excel",
    "golang": "Go", "c sharp": "C#", "csharp": "C#", "cpp": "C++",
    "py.test": "pytest", "hugging-face": "Hugging Face", "huggingface": "Hugging Face",
    "spring": "Spring Boot", "terraform cloud": "Terraform",
}

# Employer prestige table. This one IS a shared production resource: the design
# calls for a curated tier list, with an optional LLM lookup for unknown names.
# 3 is world class, 2 is good, 1 is unknown or below average - the same scale
# as data_layer1/gen/tiers.py, which the corpus is generated from.
COMPANY_TIERS = {
    "google": 3, "meta": 3, "facebook": 3, "apple": 3, "amazon": 3, "microsoft": 3,
    "netflix": 3, "nvidia": 3, "openai": 3, "deepmind": 3,
    "uber": 2, "airbnb": 2, "salesforce": 2, "adobe": 2, "ibm": 2, "oracle": 2,
    "sap": 2, "spotify": 2, "stripe": 2, "shopify": 2, "intel": 2, "cisco": 2,
    "linkedin": 2, "palantir": 2, "dell": 2,
    "atlassian": 2, "datadog": 2, "snowflake": 2, "cloudflare": 2,
}
UNKNOWN_TIER = 1

# Section heading synonyms -> canonical section key.
SECTION_SYNONYMS = {
    "summary": ["summary", "profile", "professional summary", "about me", "career objective",
                "personal statement", "professional profile", "objective"],
    "experience": ["experience", "work experience", "professional experience",
                   "employment history", "career history", "relevant experience",
                   "work history", "employment"],
    "education": ["education", "academic background", "education & qualifications",
                  "education and qualifications", "qualifications", "academic history",
                  "education and training", "academic qualifications"],
    "skills": ["skills", "technical skills", "core competencies", "technologies",
               "skills & tools", "skills and tools", "technical proficiencies",
               "key skills", "tech stack", "personal skills", "competencies"],
    "certifications": ["certifications", "certificates", "professional certifications",
                       "training & certifications", "licences & certifications"],
    "languages": ["languages", "language skills"],
    "interests": ["interests", "hobbies & interests", "outside of work",
                  "personal interests", "hobbies"],
    "projects": ["projects", "selected projects", "side projects", "notable projects",
                 "personal projects", "open source"],
    "awards": ["awards", "achievements", "honours & awards", "recognition"],
    "contact": ["contact", "contact details", "personal details", "get in touch",
                "personal information"],
    "volunteering": ["volunteering", "community", "volunteer experience"],
    "publications": ["publications", "selected publications", "research output"],
    "highlights": ["career highlights", "key achievements", "selected highlights"],
}

HEADING_LOOKUP = {
    synonym: section for section, synonyms in SECTION_SYNONYMS.items() for synonym in synonyms
}

DEGREE_PATTERNS = [
    ("PhD", re.compile(r"\b(ph\.?\s?d|doctorate|doctoral)\b", re.I)),
    ("Master", re.compile(r"\b(master'?s?|m\.?sc|m\.?eng|m\.?b\.?a|msc|mba)\b", re.I)),
    ("Bachelor", re.compile(r"\b(bachelor'?s?|b\.?sc|b\.?eng|b\.?a\b|bsc)\b", re.I)),
    ("High School", re.compile(r"\b(high school|secondary school|a-levels|diploma)\b", re.I)),
]

EDUCATION_FIELDS = [
    "Computer Science", "Data Science", "Software Engineering", "Electrical Engineering",
    "Mathematics", "Physics", "Information Systems", "Business Administration",
    "Economics", "Biology", "Communications",
]

# Lines that look like a section body but are filler, used to avoid mistaking a
# certification or interest line for a job or a skill.
NOISE_MARKERS = re.compile(
    r"references available|available upon request|page \d+\s*$", re.I)

_SKILL_LOOKUP = {name.lower(): name for name in CANONICAL_SKILLS}
_SKILL_LOOKUP.update(SKILL_ALIASES)


# Phrase matcher for skills named in prose. Longest first so "Spring Boot"
# wins over "Spring"; lookarounds rather than \b because "C++" and "C#" end in
# non-word characters.
_SKILL_PHRASES = sorted(
    set(CANONICAL_SKILLS) | {alias for alias in SKILL_ALIASES if len(alias) > 3},
    key=len, reverse=True)
SKILL_PHRASE_RE = re.compile(
    r"(?<!\w)(" + "|".join(re.escape(phrase) for phrase in _SKILL_PHRASES) + r")(?!\w)",
    re.I)


def skills_mentioned_in(text: str) -> set[str]:
    """Canonical skills named anywhere in free text.

    Used only when a CV carries no skills section at all - real applicant
    tracking systems read the experience prose in that case, and a tool named
    in an achievement is a genuine claim to it.
    """
    return {canonical_skill(match.group(1)) or ""
            for match in SKILL_PHRASE_RE.finditer(text)} - {""}


def canonical_skill(token: str) -> str | None:
    """Map a raw token to a canonical skill name, or None if unrecognised."""
    cleaned = token.strip().strip(".,;:|•-–—»▪()").strip()
    if not cleaned or len(cleaned) > 40:
        return None
    return _SKILL_LOOKUP.get(cleaned.lower())


def company_tier(name: str) -> int:
    """Tier for an employer name; unrecognised names fall to tier 1.

    The optional LLM lookup described in the README would slot in here, for
    unknown names only.
    """
    key = re.sub(r"\b(inc|llc|ltd|limited|gmbh|corp|corporation|co)\b\.?", "",
                 name.lower()).strip(" .,")
    return COMPANY_TIERS.get(key, UNKNOWN_TIER)


def heading_for(line: str) -> str | None:
    """Return the canonical section key if this line is a section heading."""
    cleaned = line.strip().strip(":").strip()
    if not cleaned or len(cleaned) > 40 or len(cleaned.split()) > 4:
        return None
    return HEADING_LOOKUP.get(cleaned.lower())


# Longest first, so "work experience" wins over "experience".
_PREFIX_HEADINGS = sorted(HEADING_LOOKUP, key=len, reverse=True)


def split_heading_prefix(line: str, require_upper: bool = True) -> tuple[str | None, str]:
    """Split "WORK EXPERIENCE Mobile Engineer" into its heading and content.

    Form-style layouts put the section label in a left-hand column that lands on
    the same text line as its value. The label is required to be upper-case or
    colon-terminated so ordinary prose starting with "Experience in ..." is not
    mistaken for a heading.
    """
    stripped = line.strip()
    lowered = stripped.lower()
    for synonym in _PREFIX_HEADINGS:
        if not lowered.startswith(synonym):
            continue
        rest = stripped[len(synonym):]
        if not rest[:1] in (" ", ":"):
            continue
        # Section splitting requires an upper-case label: a categorised skills
        # line ("Languages: Python, SQL") starts with a heading word too and
        # must not be read as a heading. Name extraction relaxes this, since it
        # validates the remainder looks like a name anyway.
        prefix = stripped[:len(synonym)]
        if require_upper and not prefix.isupper():
            continue
        return HEADING_LOOKUP[synonym], rest.lstrip(" :").strip()
    return None, stripped
