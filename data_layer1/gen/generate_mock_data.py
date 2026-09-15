"""
Generates a mock CV dataset: realistic-looking candidate profiles with a
preset 0-100 score per section (education, general experience, stack,
company/big-tech) plus an overall score.

Scores are role-agnostic (v1) - they reflect intrinsic CV strength, not
fit against a specific job posting. Role-conditioned scoring (candidate x
job pairs) is a later phase - see README.md.

Usage:
    python generate_mock_data.py --n 1000 --train-frac 0.8 --seed 42
"""

import argparse
import csv
import json
import math
import random
from pathlib import Path

from faker import Faker

# ---------------------------------------------------------------------------
# Reference data (small, inline versions of the eventual data_gen/*.py
# taxonomy modules described in README.md)
# ---------------------------------------------------------------------------

EDUCATION_LEVELS = [
    ("High School", 35),
    ("Bachelor", 60),
    ("Master", 78),
    ("PhD", 92),
]
EDUCATION_LEVEL_WEIGHTS = [0.20, 0.45, 0.27, 0.08]

EDUCATION_FIELDS = [
    ("Computer Science", 1.00),
    ("Data Science", 1.00),
    ("Software Engineering", 1.00),
    ("Electrical Engineering", 0.90),
    ("Mathematics", 0.88),
    ("Physics", 0.85),
    ("Information Systems", 0.85),
    ("Business Administration", 0.72),
    ("Economics", 0.70),
    ("Biology", 0.62),
    ("Communications", 0.58),
]

DOMAINS = [
    "Software Development",
    "Data/AI/ML",
    "DevOps/Infrastructure",
    "Product/Design",
    "Data Analysis",
    "QA/Testing",
]

DOMAIN_TITLES = {
    "Software Development": ["Software Engineer", "Backend Developer", "Full-Stack Developer",
                             "Platform Engineer", "Mobile Engineer"],
    "Data/AI/ML": ["Data Scientist", "Machine Learning Engineer", "AI Researcher",
                   "Applied Scientist", "MLOps Engineer"],
    "DevOps/Infrastructure": ["DevOps Engineer", "Site Reliability Engineer",
                              "Infrastructure Engineer", "Cloud Engineer", "Platform Engineer"],
    "Product/Design": ["Product Manager", "UX Designer", "Product Designer",
                       "UX Researcher", "Technical Product Manager"],
    "Data Analysis": ["Data Analyst", "Business Intelligence Analyst", "Analytics Engineer",
                      "Insights Analyst", "Data Engineer"],
    "QA/Testing": ["QA Engineer", "Test Automation Engineer", "QA Analyst",
                   "SDET", "Quality Engineer"],
}

# skill_name -> demand weight (0.5 - 1.0, how strongly it counts toward stack_score)
SKILLS = {
    # languages
    "Python": 1.00, "JavaScript": 0.90, "TypeScript": 0.92, "Java": 0.85,
    "C++": 0.85, "C#": 0.80, "Go": 0.90, "Rust": 0.88, "Ruby": 0.72,
    "Swift": 0.78, "Kotlin": 0.80, "Scala": 0.78, "SQL": 0.90, "Bash": 0.65,
    # web / app frameworks
    "React": 0.92, "Vue.js": 0.78, "Angular": 0.75, "Next.js": 0.85,
    "Node.js": 0.85, "Django": 0.80, "FastAPI": 0.85, "Spring Boot": 0.80,
    "REST APIs": 0.75, "GraphQL": 0.78, "gRPC": 0.75,
    # cloud / infra
    "AWS": 0.95, "Azure": 0.88, "GCP": 0.88, "Docker": 0.92, "Kubernetes": 0.95,
    "Terraform": 0.88, "Ansible": 0.72, "Linux": 0.75, "CI/CD": 0.80,
    "Jenkins": 0.68, "GitHub Actions": 0.80, "Prometheus": 0.78, "Grafana": 0.75,
    "Datadog": 0.75, "Nginx": 0.68,
    # data / ML
    "PyTorch": 0.95, "TensorFlow": 0.90, "Scikit-learn": 0.85, "Pandas": 0.85,
    "NumPy": 0.78, "Spark": 0.88, "Airflow": 0.82, "dbt": 0.82, "Kafka": 0.88,
    "Snowflake": 0.88, "Databricks": 0.88, "BigQuery": 0.82, "Redshift": 0.75,
    "MLflow": 0.80, "Hugging Face": 0.88, "LangChain": 0.82,
    # storage
    "PostgreSQL": 0.82, "MySQL": 0.72, "MongoDB": 0.75, "Redis": 0.80,
    "Elasticsearch": 0.78, "DynamoDB": 0.75,
    # analysis / design / qa / tooling
    "Tableau": 0.72, "Power BI": 0.72, "Looker": 0.75, "Excel": 0.55,
    "Figma": 0.72, "Sketch": 0.60, "Selenium": 0.65, "Cypress": 0.72,
    "Playwright": 0.78, "pytest": 0.72, "Postman": 0.60, "Jira": 0.50,
    "Git": 0.70, "Confluence": 0.48,
}

# company_name -> tier (1 = FAANG/MAANG-level, 2 = well-known large tech, 3 = everything else)
COMPANIES = {
    # Tier 1
    "Google": 1, "Meta": 1, "Apple": 1, "Amazon": 1, "Microsoft": 1,
    "Netflix": 1, "Nvidia": 1, "OpenAI": 1, "DeepMind": 1,
    # Tier 2
    "Uber": 2, "Airbnb": 2, "Salesforce": 2, "Adobe": 2, "IBM": 2,
    "Oracle": 2, "SAP": 2, "Spotify": 2, "Stripe": 2, "Shopify": 2,
    "Intel": 2, "Cisco": 2, "LinkedIn": 2, "Palantir": 2, "Dell": 2,
    # Tier 3 (generic / smaller companies)
    "BrightPath Solutions": 3, "Nova Tech Group": 3, "Summit Digital": 3,
    "Clearwater Systems": 3, "Meridian Software": 3, "Blue Orbit Labs": 3,
    "Ridgeline Analytics": 3, "Ironwood Consulting": 3, "Cobalt Applications": 3,
    "Vantage Point Technologies": 3, "Silverline Data": 3, "Fenwick Digital": 3,
    "Amber Creek Software": 3, "Northbridge IT": 3, "Lumen Works": 3,
}
# Skills a person in each domain actually tends to list. Most of a candidate's
# stack is drawn from here so a DevOps engineer doesn't end up with LangChain
# as their headline skill; the remainder comes from the full taxonomy, since
# real people do carry adjacent and legacy tools.
DOMAIN_SKILLS = {
    "Software Development": [
        "Python", "JavaScript", "TypeScript", "Java", "C#", "Go", "React", "Vue.js",
        "Angular", "Next.js", "Node.js", "Django", "FastAPI", "Spring Boot", "REST APIs",
        "GraphQL", "gRPC", "PostgreSQL", "MySQL", "Redis", "Docker", "Git", "CI/CD", "pytest",
    ],
    "Data/AI/ML": [
        "Python", "PyTorch", "TensorFlow", "Scikit-learn", "Pandas", "NumPy", "Spark",
        "MLflow", "Hugging Face", "LangChain", "Databricks", "SQL", "Airflow", "AWS",
        "Docker", "BigQuery", "Snowflake", "Kafka",
    ],
    "DevOps/Infrastructure": [
        "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform", "Ansible", "Linux",
        "CI/CD", "Jenkins", "GitHub Actions", "Prometheus", "Grafana", "Datadog", "Nginx",
        "Bash", "Python", "Go", "Redis", "Kafka",
    ],
    "Product/Design": [
        "Figma", "Sketch", "Jira", "Confluence", "SQL", "Excel", "Looker", "Tableau",
        "React", "TypeScript", "Power BI", "Git",
    ],
    "Data Analysis": [
        "SQL", "Excel", "Tableau", "Power BI", "Looker", "Python", "Pandas", "NumPy",
        "dbt", "Snowflake", "BigQuery", "Redshift", "Airflow", "Spark", "PostgreSQL",
    ],
    "QA/Testing": [
        "Selenium", "Cypress", "Playwright", "pytest", "Postman", "Jira", "CI/CD",
        "Python", "Java", "JavaScript", "TypeScript", "Git", "Docker", "GitHub Actions",
    ],
}

COMPANY_NAMES = list(COMPANIES.keys())
COMPANY_WEIGHTS = [0.06 if COMPANIES[c] == 1 else 0.28 if COMPANIES[c] == 2 else 0.66
                   for c in COMPANY_NAMES]

TIER_POINTS = {1: 100, 2: 65, 3: 35}
CURRENT_YEAR = 2026

# Career length is drawn per band rather than from one curve, so every stage
# from fresh graduate to 20-year veteran is properly represented.
EXPERIENCE_BANDS = [
    ((0.0, 1.0), 6),    # fresh out of education
    ((1.0, 2.0), 16),   # junior
    ((2.0, 5.0), 24),   # mid
    ((5.0, 10.0), 26),  # senior
    ((10.0, 18.0), 20), # lead / principal
    ((18.0, 30.0), 8),  # veteran
]

# Typical length of each qualification, used to give education a from-to range.
DEGREE_YEARS = {"High School": (3, 4), "Bachelor": (3, 4), "Master": (1, 2), "PhD": (3, 5)}


def clip(x, lo=0, hi=100):
    return max(lo, min(hi, x))


def noisy(value, sigma=4.5):
    return clip(round(value + random.gauss(0, sigma)))


def gen_institution(fake: Faker) -> str:
    suffix = random.choice(["University", "Institute of Technology", "State University", "National University"])
    return f"{fake.city()} {suffix}"


def months_to_ym(total_months: int) -> str:
    """Convert an absolute month count back to a 'YYYY-MM' string."""
    year, month = divmod(total_months, 12)
    return f"{year:04d}-{month + 1:02d}"


def gen_total_years() -> float:
    """Career length, drawn per band so juniors and veterans are both well represented."""
    (low, high), = random.choices([b[0] for b in EXPERIENCE_BANDS],
                                  weights=[b[1] for b in EXPERIENCE_BANDS], k=1)
    return round(random.uniform(low, high), 1)


def gen_education(total_years: float, fake: Faker) -> tuple:
    """Returns (education_list, level_base, field_weight) - list is highest degree first.

    Each entry carries a start_year/end_year range, the way a CV states it.
    """
    level, level_base = random.choices(EDUCATION_LEVELS, weights=EDUCATION_LEVEL_WEIGHTS, k=1)[0]
    end_year = CURRENT_YEAR - int(total_years) - random.randint(0, 2)
    duration = random.randint(*DEGREE_YEARS[level])

    if level == "High School":
        # A high school diploma has no field of study.
        entries = [{
            "level": "High School",
            "field": None,
            "institution": f"{fake.city()} High School",
            "start_year": end_year - duration,
            "end_year": end_year,
        }]
        return entries, level_base, 1.0

    field, field_weight = random.choice(EDUCATION_FIELDS)
    entries = [{
        "level": level,
        "field": field,
        "institution": gen_institution(fake),
        "start_year": end_year - duration,
        "end_year": end_year,
    }]
    # Postgrads also hold an earlier bachelor's degree, finished before this one began.
    if level in ("Master", "PhD"):
        bachelor_end = entries[0]["start_year"] - random.randint(0, 1)
        bachelor_duration = random.randint(*DEGREE_YEARS["Bachelor"])
        entries.append({
            "level": "Bachelor",
            "field": field if random.random() < 0.7 else random.choice(EDUCATION_FIELDS)[0],
            "institution": gen_institution(fake),
            "start_year": bachelor_end - bachelor_duration,
            "end_year": bachelor_end,
        })
    return entries, level_base, field_weight


def gen_companies(total_years: float, domain: str) -> list:
    """Build a reverse-chronological job history with real start/end dates."""
    if total_years < 1:
        return []

    titles = DOMAIN_TITLES[domain]
    # Job count follows career length - a two-year junior can't have had four
    # employers, and a fifteen-year veteran rarely has had only one.
    floor = max(1, int(total_years // 6))
    ceiling = max(floor, min(5, int(total_years // 3) + 1))
    n_companies = random.randint(floor, ceiling)
    tenures, remaining = [], total_years
    for i in range(n_companies):
        if remaining <= 0.2:
            break
        max_tenure = remaining if i == n_companies - 1 else remaining * random.uniform(0.3, 0.8)
        tenure = round(max(0.3, min(remaining, max_tenure)), 1)
        tenures.append(tenure)
        remaining = round(remaining - tenure, 1)

    # Walk backwards from today: most recent job first, with occasional gaps between roles.
    cursor = CURRENT_YEAR * 12 + 8  # September of the current year
    if random.random() < 0.35:
        cursor -= random.randint(1, 8)  # not currently employed

    companies, used_names = [], []
    experience_after = 0.0  # years of experience accumulated after this job started
    for i, tenure in enumerate(tenures):
        months = max(4, round(tenure * 12))
        end_months = cursor
        start_months = end_months - months
        cursor = start_months - random.randint(0, 5)  # gap before the next (older) job

        name = random.choices(COMPANY_NAMES, weights=COMPANY_WEIGHTS, k=1)[0]
        while used_names and name == used_names[-1]:
            name = random.choices(COMPANY_NAMES, weights=COMPANY_WEIGHTS, k=1)[0]
        used_names.append(name)

        # Seniority reflects career stage: experience already held when the role started.
        experience_before = total_years - experience_after - tenure
        title = random.choice(titles)
        if experience_before >= 12:
            title = random.choice(["Principal ", "Lead ", "Staff "]) + title
        elif experience_before >= 6:
            title = "Senior " + title
        elif experience_before < 1.5:
            title = random.choice(["Junior ", "", ""]) + title
        experience_after += tenure

        companies.append({
            "name": name,
            "title": title.strip(),
            "years": tenure,
            "start_date": months_to_ym(start_months),
            "end_date": None if i == 0 and end_months >= CURRENT_YEAR * 12 + 8 else months_to_ym(end_months),
            "tier": COMPANIES[name],
        })
    return companies


def gen_candidate(fake: Faker) -> dict:
    # --- structural attributes ---
    total_years = gen_total_years()

    education, level_base, field_weight = gen_education(total_years, fake)

    domain = random.choice(DOMAINS)
    general_experience = {"domain": domain, "total_years": total_years}

    # Longer careers accumulate a broader stack, mostly within their own domain.
    n_skills = max(4, min(16, int(5 + total_years / 1.8) + random.randint(-1, 2)))
    core_pool = DOMAIN_SKILLS[domain]
    n_core = min(len(core_pool), max(2, int(n_skills * 0.7)))
    skill_names = random.sample(core_pool, n_core)
    adjacent = [s for s in SKILLS if s not in skill_names]
    # Core domain skills stay first: CVs lead with their headline stack, and the
    # summary quotes the first few.
    skill_names += random.sample(adjacent, min(n_skills - n_core, len(adjacent)))
    skills = []
    for s in skill_names:
        years = round(min(total_years, random.uniform(0.5, max(1.0, total_years))), 1)
        skills.append({"name": s, "years": years})

    companies = gen_companies(total_years, domain)

    # Keep the timeline coherent: nobody graduates after their first job started.
    # Shift the whole education block so degree lengths and ordering survive.
    if companies:
        first_job_year = int(companies[-1]["start_date"][:4])
        overlap = education[0]["end_year"] - first_job_year
        if overlap > 0:
            for entry in education:
                entry["start_year"] -= overlap
                entry["end_year"] -= overlap

    # --- scores (deterministic formula + noise) ---
    education_score = noisy(level_base * field_weight)

    general_experience_score = noisy(100 * (1 - math.exp(-total_years / 8)))

    raw_stack = sum((min(sk["years"], 6) / 6) * SKILLS[sk["name"]] for sk in skills)
    stack_score = noisy(100 * (1 - math.exp(-raw_stack / 4)))

    if companies:
        tenure_sum = sum(c["years"] for c in companies)
        company_raw = sum(TIER_POINTS[c["tier"]] * c["years"] for c in companies) / tenure_sum
        company_score = noisy(company_raw)
    else:
        company_score = noisy(20)

    overall_score = round(clip((education_score + general_experience_score + stack_score + company_score) / 4))

    # fake.name() attaches titles/suffixes ("Miss", "MD") that muddy the name label.
    name = f"{fake.first_name()} {fake.last_name()}"
    if random.random() < 0.12:
        name = f"{name.split()[0]} {random.choice('ABCDEFGHJKLMNPRSTW')}. {name.split()[1]}"

    return {
        "id": None,  # assigned after the train/test split, see main()
        "name": name,
        "education": education,
        "general_experience": general_experience,
        "skills": skills,
        "companies": companies,
        "scores": {
            "education_score": education_score,
            "general_experience_score": general_experience_score,
            "stack_score": stack_score,
            "company_score": company_score,
            "overall_score": overall_score,
        },
    }


def flatten(record: dict) -> dict:
    top_skills = sorted(record["skills"], key=lambda s: s["years"], reverse=True)[:5]
    top_skills_str = ";".join(f"{s['name']}:{s['years']}" for s in top_skills)
    if record["companies"]:
        best_company = max(record["companies"], key=lambda c: (c["tier"] == 1, c["years"]))
        best_company_name = best_company["name"]
        best_company_tier = best_company["tier"]
    else:
        best_company_name = ""
        best_company_tier = ""
    top_edu = record["education"][0]
    return {
        "id": record["id"],
        "name": record["name"],
        "education_level": top_edu["level"],
        "education_field": top_edu["field"] or "",
        "education_start_year": top_edu["start_year"],
        "grad_year": top_edu["end_year"],
        "domain": record["general_experience"]["domain"],
        "total_years_experience": record["general_experience"]["total_years"],
        "num_skills": len(record["skills"]),
        "top_skills": top_skills_str,
        "num_companies": len(record["companies"]),
        "best_company": best_company_name,
        "best_company_tier": best_company_tier,
        **record["scores"],
    }


def write_jsonl(records: list, path: Path):
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_csv(records: list, path: Path):
    flat = [flatten(r) for r in records]
    fieldnames = list(flat[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat)


def main():
    parser = argparse.ArgumentParser(description="Generate mock CV train/test data.")
    parser.add_argument("--n", type=int, default=1000, help="Total number of candidates")
    parser.add_argument("--train-frac", type=float, default=0.8, help="Fraction assigned to train")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--out-dir", type=str, default="..", help="Output directory (relative to this script)")
    args = parser.parse_args()

    random.seed(args.seed)
    fake = Faker()
    Faker.seed(args.seed)

    records = [gen_candidate(fake) for _ in range(args.n)]
    random.shuffle(records)

    n_train = int(args.n * args.train_frac)
    train_records, test_records = records[:n_train], records[n_train:]

    # Ids are assigned after the shuffle/split so that row order, id order and
    # PDF filename order all agree: train.jsonl line 1 == train.csv row 1 ==
    # cvs_pdf/train/TRAIN00001.pdf, and likewise for test.
    for prefix, subset in (("TRAIN", train_records), ("TEST", test_records)):
        for i, record in enumerate(subset, 1):
            record["id"] = f"{prefix}{i:05d}"

    out_dir = (Path(__file__).parent / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Each split owns its own folder under train_test_data/.
    for split, records_for_split in (("train", train_records), ("test", test_records)):
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(records_for_split, split_dir / f"{split}.jsonl")
        write_csv(records_for_split, split_dir / f"{split}.csv")

    print(f"Wrote {len(train_records)} train / {len(test_records)} test records to {out_dir}")


if __name__ == "__main__":
    main()
