"""
Content banks for realistic CV documents.

Shaped by how real resumes are actually written: 3-5 quantified bullets on the
most recent role tapering to 1-2 on older ones, a 2-4 sentence summary, skills
grouped by category, projects with enough detail to discuss in an interview,
and the filler sections (certifications, languages, volunteering, interests)
that a parser has to recognise and then ignore.

Everything here only ever names skills the candidate actually holds, and only
where the skill family fits the claim, so the rendered document never
contradicts the ground-truth JSON.
"""

import random

MONTHS_SHORT = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTHS_LONG = ["January", "February", "March", "April", "May", "June", "July",
               "August", "September", "October", "November", "December"]

SECTION_HEADINGS = {
    "summary": ["Summary", "Profile", "Professional Summary", "About Me",
                "Career Objective", "Personal Statement", "Professional Profile"],
    "experience": ["Experience", "Work Experience", "Professional Experience",
                   "Employment History", "Career History", "Relevant Experience"],
    "education": ["Education", "Academic Background", "Education & Qualifications",
                  "Qualifications", "Academic History", "Education and Training"],
    "skills": ["Skills", "Technical Skills", "Core Competencies", "Technologies",
               "Skills & Tools", "Technical Proficiencies", "Key Skills", "Tech Stack"],
    "certifications": ["Certifications", "Certificates", "Professional Certifications",
                       "Training & Certifications", "Licences & Certifications"],
    "languages": ["Languages", "Language Skills"],
    "interests": ["Interests", "Hobbies & Interests", "Outside of Work", "Personal Interests"],
    "projects": ["Projects", "Selected Projects", "Side Projects", "Notable Projects",
                 "Personal Projects", "Open Source"],
    "awards": ["Awards", "Achievements", "Honours & Awards", "Recognition"],
    "contact": ["Contact", "Contact Details", "Personal Details", "Get In Touch"],
    "volunteering": ["Volunteering", "Community", "Volunteer Experience"],
    "publications": ["Publications", "Selected Publications", "Research Output"],
    "highlights": ["Career Highlights", "Key Achievements", "Selected Highlights"],
}

BULLET_CHARS = ["•", "-", "»", "–", "▪"]

# Skill families, so a bullet never claims something absurd like
# "built CI/CD pipelines with Excel".
SKILL_KINDS = {
    "code": {"Python", "JavaScript", "TypeScript", "Java", "C++", "C#", "Go", "Rust", "Ruby",
             "Swift", "Kotlin", "Scala", "React", "Vue.js", "Angular", "Next.js", "Node.js",
             "Django", "FastAPI", "Spring Boot", "REST APIs", "GraphQL", "gRPC"},
    "infra": {"AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform", "Ansible", "Linux",
              "CI/CD", "Jenkins", "GitHub Actions", "Prometheus", "Grafana", "Datadog", "Nginx"},
    "ml": {"PyTorch", "TensorFlow", "Scikit-learn", "Pandas", "NumPy", "Python", "Spark",
           "Hugging Face", "LangChain", "MLflow", "Databricks"},
    "data": {"SQL", "Spark", "Airflow", "dbt", "Kafka", "Snowflake", "Databricks", "BigQuery",
             "Redshift", "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch",
             "DynamoDB", "Pandas", "Excel"},
    "viz": {"Tableau", "Power BI", "Looker", "Excel", "Figma"},
    "test": {"Selenium", "Cypress", "Playwright", "pytest", "Postman", "CI/CD", "Java", "Python"},
    "design": {"Figma", "Sketch", "React", "TypeScript", "Next.js"},
}

# Stand-ins used when a candidate holds no skill of the required family.
GENERIC_SUBJECT = {
    "code": "our internal services", "infra": "the in-house toolchain",
    "ml": "the modelling stack", "data": "the data warehouse",
    "viz": "the reporting suite", "test": "the automation framework",
    "design": "the design system", "any": "the platform",
}

# Each entry is (sentence, required skill family). Metrics follow what real
# engineering resumes quantify: latency, uptime, deploy frequency, users, cost.
ACHIEVEMENTS = {
    "Software Development": [
        ("Designed and shipped {skill} services handling {n}k requests per minute, cutting p99 latency from {ms1}ms to {ms2}ms.", "code"),
        ("Led the migration of a legacy monolith to {skill}, taking deployments from fortnightly to {n} per day.", "code"),
        ("Built a reusable {skill} component library adopted by {n} product teams, removing months of duplicated work.", "code"),
        ("Refactored the billing module behind a feature flag, eliminating {pct}% of recurring production incidents.", "any"),
        ("Introduced contract testing with {skill}, lifting coverage from {low}% to {high_pct}% and cutting integration bugs.", "test"),
        ("Owned the customer-facing API end to end on {skill}, from schema design through on-call support.", "code"),
        ("Mentored {n} junior engineers, ran weekly design reviews, and authored the team's onboarding guide.", "any"),
        ("Cut CI build times by {pct}% by parallelising the {skill} pipeline and caching dependency layers.", "infra"),
        ("Rewrote the search path in {skill}, dropping median response from {ms1}ms to {ms2}ms for {n}M monthly users.", "code"),
        ("Drove the team's adoption of trunk-based development, reducing merge conflicts and lead time by {pct}%.", "any"),
        ("Partnered with product to scope and deliver {n} major releases, each behind staged rollouts.", "any"),
        ("Hardened authentication flows in {skill} after a security review, closing {n} findings before audit.", "code"),
        ("Replaced a third-party dependency with an in-house {skill} module, saving ${n}k per year in licence fees.", "code"),
        ("Set the service-level objectives for {n} services and built the dashboards the team still runs on.", "infra"),
    ],
    "Data/AI/ML": [
        ("Trained and deployed {skill} models serving {n}M predictions per day at under {ms2}ms p95.", "ml"),
        ("Lifted recommendation click-through by {pct}% through feature engineering and retraining cadence in {skill}.", "ml"),
        ("Built an end-to-end training pipeline on {skill}, cutting retraining from days to {n} hours.", "ml"),
        ("Productionised demand forecasting models that reduced stock-outs by {pct}% across {n} regions.", "any"),
        ("Ran A/B tests across {n} model variants; the winning variant added an estimated ${n}0k in annual revenue.", "any"),
        ("Developed an NLP classification system in {skill} reaching {high_pct}% F1, replacing a manual review queue.", "ml"),
        ("Cut inference cost by {pct}% through quantisation, batching and autoscaling on {skill}.", "infra"),
        ("Built the feature store that {n} downstream models now read from, ending duplicated feature logic.", "data"),
        ("Established offline/online evaluation parity, catching {n} silent regressions before release.", "any"),
        ("Translated ambiguous business questions into a modelling roadmap with product and finance stakeholders.", "any"),
        ("Wrote the team's model documentation and review standard, now required before any production launch.", "any"),
        ("Migrated ad-hoc notebooks into versioned {skill} pipelines with reproducible runs and tracked experiments.", "ml"),
        ("Reduced label noise by redesigning the annotation guidelines, improving validation accuracy by {pct}%.", "any"),
        ("Mentored {n} analysts moving into modelling work, with weekly paper reading and code review.", "any"),
    ],
    "DevOps/Infrastructure": [
        ("Migrated {n} services to {skill}, cutting monthly infrastructure spend by {pct}%.", "infra"),
        ("Built CI/CD pipelines with {skill}, bringing mean time to deploy under {n} minutes.", "infra"),
        ("Introduced infrastructure-as-code with {skill}, eliminating manual provisioning entirely.", "infra"),
        ("Improved cluster utilisation by {pct}% through right-sizing and {skill} autoscaling policies.", "infra"),
        ("Reduced incident MTTR from {ms3} hours to under {mins} minutes with better alert routing and runbooks.", "any"),
        ("Hardened {skill} environments ahead of SOC 2, closing {n} critical findings.", "infra"),
        ("Automated {n} manual runbooks, saving an estimated {low} engineer-hours per month.", "any"),
        ("Ran the on-call rotation for {n} production services, sustaining 99.9% availability.", "any"),
        ("Designed the multi-region failover strategy and led the game-day exercises that proved it.", "any"),
        ("Built observability on {skill} so teams could self-serve dashboards instead of filing tickets.", "infra"),
        ("Cut container image sizes by {pct}%, shortening cold starts across the fleet.", "infra"),
        ("Introduced secret rotation and least-privilege IAM across {n} accounts.", "infra"),
        ("Led the Kubernetes upgrade across {n} clusters with zero customer-visible downtime.", "infra"),
        ("Wrote the platform team's paved-road documentation, cutting onboarding from weeks to days.", "any"),
    ],
    "Product/Design": [
        ("Ran discovery interviews with {n} customers, reshaping the following year's roadmap.", "any"),
        ("Redesigned onboarding in {skill}, lifting activation from {low}% to {high_pct}%.", "design"),
        ("Defined north-star metrics and instrumented them across {n} delivery squads.", "any"),
        ("Shipped a design system in {skill} adopted across {n} product surfaces.", "design"),
        ("Prioritised a backlog for {n} engineers, delivering {ms3} releases per quarter.", "any"),
        ("Reduced support contacts by {pct}% by reworking the checkout and error states.", "any"),
        ("Facilitated usability testing that surfaced {n} critical friction points before launch.", "any"),
        ("Wrote the PRDs and acceptance criteria for a platform migration affecting {n}k users.", "any"),
        ("Negotiated scope with engineering and legal to ship a compliance deadline on time.", "any"),
        ("Built the experimentation framework the team uses to size bets before committing.", "any"),
        ("Ran quarterly roadmap reviews with executive stakeholders and published the trade-offs openly.", "any"),
        ("Introduced accessibility standards to the design process, taking {n} flows to WCAG AA.", "design"),
        ("Prototyped {n} concepts in {skill} and killed two early on evidence rather than opinion.", "design"),
        ("Partnered with data to define activation and retention definitions the whole company now uses.", "any"),
    ],
    "Data Analysis": [
        ("Built {skill} dashboards used daily by {n} stakeholders across three departments.", "viz"),
        ("Automated monthly reporting with {skill}, saving {n} analyst-hours per cycle.", "data"),
        ("Analysed churn drivers and informed retention work that reduced churn by {pct}%.", "any"),
        ("Modelled pricing scenarios in {skill}, supporting a {pct}% margin improvement.", "data"),
        ("Defined and documented {n} core business metrics in the semantic layer, ending duplicate definitions.", "any"),
        ("Rebuilt the finance forecast with the FP&A team, improving accuracy by {pct}%.", "any"),
        ("Migrated reporting from spreadsheets to {skill}, giving one trusted source of numbers.", "data"),
        ("Ran cohort analysis that reframed how the business measures activation.", "any"),
        ("Built data quality tests in {skill} that catch upstream breakages before stakeholders see them.", "data"),
        ("Trained {n} non-technical colleagues to self-serve, cutting ad-hoc request volume by {pct}%.", "any"),
        ("Partnered with engineering to fix event tracking, recovering {pct}% of previously lost sessions.", "any"),
        ("Presented findings to the executive team monthly, translating analysis into concrete decisions.", "any"),
        ("Designed the experiment readouts template now used company-wide.", "any"),
        ("Consolidated {n} legacy reports into {ms3} maintained dashboards.", "viz"),
    ],
    "QA/Testing": [
        ("Automated {n} regression suites with {skill}, cutting the manual test cycle by {pct}%.", "test"),
        ("Reduced escaped defects by {pct}% through risk-based test prioritisation.", "any"),
        ("Built a performance harness in {skill} simulating {n}k concurrent users.", "test"),
        ("Integrated automated checks into the {skill} pipeline, blocking {n} faulty releases.", "infra"),
        ("Established test data management practices later adopted by {n} teams.", "any"),
        ("Drove accessibility coverage from {low}% to {high_pct}% of core journeys.", "any"),
        ("Introduced flaky-test quarantine, taking suite reliability from {low}% to {high_pct}%.", "test"),
        ("Owned release sign-off for {n} services, coordinating across product and support.", "any"),
        ("Wrote the team's testing strategy, separating unit, contract and end-to-end responsibilities.", "any"),
        ("Cut end-to-end suite runtime from {n} hours to {ms3} minutes through parallelisation.", "test"),
        ("Set up cross-browser and device coverage in {skill}, catching defects staging had missed.", "test"),
        ("Partnered with developers to shift testing left, embedding QA in refinement sessions.", "any"),
        ("Built the defect triage process and reporting the engineering leads still use.", "any"),
        ("Mentored {n} manual testers into automation roles.", "any"),
    ],
}

# Domain-neutral lines used to top up long careers without repeating a sentence.
GENERIC_ACHIEVEMENTS = [
    ("Represented the team in cross-functional planning with product, design and support.", "any"),
    ("Interviewed and onboarded {n} new hires, shaping the hiring bar for the team.", "any"),
    ("Led a team of {ms3} through a full replatforming with no missed delivery dates.", "any"),
    ("Introduced written design docs and an RFC process, cutting rework late in delivery.", "any"),
    ("Ran retrospectives and drove the follow-up actions to completion, not just to the board.", "any"),
    ("Managed stakeholder communication during a major incident, including the post-mortem.", "any"),
    ("Reduced technical debt on a legacy component that had blocked {n} previous attempts.", "any"),
    ("Presented the team's work at an internal engineering forum of {n}0+ colleagues.", "any"),
    ("Took over an unowned service, documented it and brought it back under SLO.", "any"),
    ("Standardised the team's tooling with {skill}, ending long-running environment drift.", "infra"),
]

SUMMARIES_NO_JOB = [
    "{domain} graduate looking for a first role. Picked up {skill1} and {skill2} through "
    "coursework and side projects rather than commercial work, and I'd like to learn the rest "
    "somewhere that reviews code properly.",
    "Recent graduate moving into {domain}. Most of my practical experience is self-directed - "
    "personal projects in {skill1} and {skill2} - and I'm looking for a team to learn production "
    "engineering with.",
    "Entry-level {domain} candidate. Strongest in {skill1}; comfortable enough with {skill2} to "
    "be useful from week one, and honest about what I still need to learn.",
]

SUMMARIES_JUNIOR = [
    "{domain} graduate with {years_text} of hands-on experience, most recently at {company}. "
    "Comfortable with {skill1} and {skill2}, and keen to work somewhere with strong code review culture.",
    "Early-career {title} with {years_text} in {domain}. Strongest in {skill1}; currently deepening "
    "my {skill2} experience through production work rather than tutorials.",
    "{title} with {years_text} of commercial experience across {domain}. I learn fast, ask a lot of "
    "questions, and care about leaving code clearer than I found it. Core tools: {skill1}, {skill2}.",
]

SUMMARIES_MID = [
    "{title} with {years_text} in {domain}, focused on {skill1} and {skill2}. Comfortable owning a "
    "feature from discovery through to production support.",
    "{domain} specialist with {years_text}+ building and shipping production systems. Deep hands-on "
    "experience with {skill1}, {skill2} and {skill3}, most recently at {company}.",
    "Pragmatic {title} ({years_text} in {domain}). I enjoy untangling messy systems and leaving them "
    "simpler than I found them. Day to day that means {skill1}, {skill2} and a lot of conversations "
    "with the people who use what we build.",
]

SUMMARIES_SENIOR = [
    "{title} with {years_text} in {domain}, currently at {company}. I work at the point where "
    "architecture meets delivery: setting technical direction, then staying close enough to the code "
    "to know whether it held. Strongest in {skill1}, {skill2} and {skill3}.",
    "Senior {domain} professional, {years_text} in. Track record of taking ambiguous problems through "
    "design, delivery and the unglamorous operational work that follows. I mentor deliberately and "
    "write things down. Core stack: {skill1}, {skill2}, {skill3}.",
    "{years_text} in {domain}, most of it spent on systems other teams depend on. Comfortable leading "
    "technical decisions, pushing back on scope when it matters, and carrying a pager for what I build. "
    "Primary tools: {skill1} and {skill2}.",
]

CERTIFICATIONS = [
    ("AWS Certified Solutions Architect - Associate", "Amazon Web Services"),
    ("AWS Certified Developer - Associate", "Amazon Web Services"),
    ("Certified Kubernetes Administrator (CKA)", "Cloud Native Computing Foundation"),
    ("Professional Data Engineer", "Google Cloud"),
    ("Azure Fundamentals (AZ-900)", "Microsoft"),
    ("Azure Solutions Architect Expert (AZ-305)", "Microsoft"),
    ("Professional Scrum Master I (PSM I)", "Scrum.org"),
    ("TensorFlow Developer Certificate", "Google"),
    ("ISTQB Certified Tester - Foundation Level", "ISTQB"),
    ("Tableau Desktop Specialist", "Tableau"),
    ("HashiCorp Certified: Terraform Associate", "HashiCorp"),
    ("Databricks Certified Data Engineer Associate", "Databricks"),
    ("Certified Information Systems Security Professional (CISSP)", "ISC2"),
    ("Google Analytics Certification", "Google"),
    ("Certified ScrumMaster (CSM)", "Scrum Alliance"),
]

LANGUAGE_NAMES = ["English", "Spanish", "French", "German", "Arabic", "Mandarin",
                  "Portuguese", "Hindi", "Dutch", "Italian", "Polish", "Japanese", "Turkish"]
LANGUAGE_LEVELS = ["Native", "Fluent", "Professional working proficiency",
                   "Intermediate (B1)", "Conversational", "Basic (A2)", "C1", "B2", "Bilingual"]

INTERESTS = [
    "Trail running", "Chess", "Amateur photography", "Cycling", "Rock climbing",
    "Open-source contribution", "Cooking", "Jazz guitar", "Board games", "Bouldering",
    "Volunteering at a local coding club", "Long-distance swimming", "Woodworking",
    "Travel photography", "Five-a-side football", "Urban sketching", "Home automation",
    "Competitive pub quizzes", "Baking", "Sailing",
]

AWARDS = [
    "Employee of the Quarter", "Internal Hackathon Winner", "Dean's List",
    "Innovation Award", "Peer Recognition Award", "Best Team Contribution",
    "Engineering Excellence Award", "Graduated with Distinction",
]

VOLUNTEERING = [
    "Mentor, {org} - coaching career changers through their first technical role.",
    "Volunteer instructor at a weekend coding club for {n} secondary school students.",
    "Maintainer of a small open-source {skill} library used by a handful of teams.",
    "Committee member, local {domain} meetup - organise speakers and venue.",
    "STEM ambassador running school workshops twice a term.",
]

VOLUNTEER_ORGS = ["Code First Girls", "a local refugee support charity", "CoderDojo",
                  "a community tech collective", "the university alumni network"]

PROJECTS = [
    ("{skill} utility library", "Open-source helper library for {skill}, ~{n}00 GitHub stars. "
     "Wrote the docs and handled issue triage."),
    ("Personal finance tracker", "Self-hosted spend tracker built with {skill}. "
     "Deliberately kept the data local rather than using a hosted service."),
    ("Charity chatbot", "Built a {skill} assistant for a local charity to answer volunteer "
     "questions; retired it once they moved to a supported platform."),
    ("Home energy dashboard", "Collects meter readings and renders them with {skill}. "
     "Mostly an excuse to learn time-series storage properly."),
    ("Match-day predictor", "Small model in {skill} predicting league results. "
     "Beats the naive baseline; still loses to the bookmakers."),
    ("Technical blog", "Writing up {skill} experiments and failures, roughly {n}k readers a month."),
    ("Recipe search engine", "Full-text search over a scraped recipe corpus using {skill}; "
     "learned more about ranking than about cooking."),
    ("Commute optimiser", "Pulls transit APIs and suggests departure times, built on {skill}."),
    ("Conference talk", "Spoke on {skill} in production at a regional meetup, ~{n}0 attendees."),
]

PUBLICATIONS = [
    "\"{topic}\" - accepted at a peer-reviewed workshop, {year}.",
    "\"{topic}\" - co-authored conference paper, {year}.",
    "\"{topic}\" - technical report, {year}. Cited internally as the basis for the current design.",
]

THESIS_TOPICS = {
    "Computer Science": ["Scheduling strategies for distributed task queues",
                         "Cache coherence under bursty workloads"],
    "Data Science": ["Handling label noise in weakly supervised classification",
                     "Drift detection for production forecasting models"],
    "Software Engineering": ["Measuring the cost of technical debt in delivery throughput",
                             "Contract testing between independently deployed services"],
    "Electrical Engineering": ["Low-power sensor networks for building monitoring",
                               "Signal denoising on embedded hardware"],
    "Mathematics": ["Convergence bounds for stochastic optimisation methods",
                    "Graph partitioning under capacity constraints"],
    "Physics": ["Monte Carlo methods for particle transport",
                "Statistical modelling of detector noise"],
    "Information Systems": ["Data governance in federated organisations",
                            "Adoption barriers for self-service analytics"],
    "Business Administration": ["Pricing strategy in two-sided marketplaces",
                                "Organisational change during platform migration"],
    "Economics": ["Demand elasticity in subscription markets",
                  "Labour mobility and remote work adoption"],
    "Biology": ["Sequence alignment heuristics for large genomes",
                "Population modelling under habitat fragmentation"],
    "Communications": ["Trust signals in digital public messaging",
                       "Framing effects in crisis communication"],
}

COURSEWORK = {
    "Computer Science": ["Algorithms", "Distributed Systems", "Operating Systems", "Databases",
                         "Compilers", "Computer Networks"],
    "Data Science": ["Statistical Learning", "Bayesian Inference", "Data Mining",
                     "Experimental Design", "Natural Language Processing"],
    "Software Engineering": ["Software Architecture", "Requirements Engineering", "Testing",
                             "Human-Computer Interaction", "Agile Delivery"],
    "Electrical Engineering": ["Signal Processing", "Control Systems", "Embedded Systems",
                               "Microelectronics"],
    "Mathematics": ["Linear Algebra", "Real Analysis", "Probability Theory", "Numerical Methods",
                    "Optimisation"],
    "Physics": ["Quantum Mechanics", "Computational Physics", "Thermodynamics", "Electromagnetism"],
    "Information Systems": ["Database Design", "Enterprise Architecture", "IT Governance",
                            "Business Process Modelling"],
    "Business Administration": ["Corporate Finance", "Operations Management", "Marketing Analytics",
                                "Strategic Management"],
    "Economics": ["Econometrics", "Microeconomic Theory", "Game Theory", "Public Economics"],
    "Biology": ["Genetics", "Bioinformatics", "Cell Biology", "Biostatistics"],
    "Communications": ["Media Theory", "Public Relations", "Research Methods", "Digital Strategy"],
}

HONOURS = ["First Class Honours", "2:1 Honours", "Graduated with Distinction", "Cum Laude",
           "GPA 3.8/4.0", "GPA 3.6/4.0", "Dean's List (2 years)", "Magna Cum Laude"]

WORK_MODES = ["Remote", "Hybrid", "On-site", "Remote (UK)", "Hybrid - 2 days on-site"]
EMPLOYMENT_TYPES = ["Full-time", "Contract", "Fixed-term", "Part-time"]

# Plausible window for certifications and awards to have been earned.
MIN_CERT_YEAR, MAX_CERT_YEAR = 2016, 2026

SKILL_LEVEL_WORDS = ["Expert", "Advanced", "Proficient", "Intermediate", "Working knowledge"]

SKILL_CATEGORIES = {
    "Languages": {"Python", "JavaScript", "TypeScript", "Java", "C++", "C#", "Go", "Rust",
                  "Ruby", "Swift", "Kotlin", "Scala", "SQL", "Bash"},
    "Frameworks": {"React", "Vue.js", "Angular", "Next.js", "Node.js", "Django", "FastAPI",
                   "Spring Boot", "PyTorch", "TensorFlow", "Scikit-learn", "Pandas", "NumPy",
                   "Hugging Face", "LangChain"},
    "Cloud & DevOps": {"AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform", "Ansible",
                       "Linux", "CI/CD", "Jenkins", "GitHub Actions", "Nginx", "Prometheus",
                       "Grafana", "Datadog"},
    "Data": {"Spark", "Airflow", "dbt", "Kafka", "Snowflake", "Databricks", "BigQuery",
             "Redshift", "MLflow", "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch",
             "DynamoDB"},
    "Analytics & Design": {"Tableau", "Power BI", "Looker", "Excel", "Figma", "Sketch"},
    "Testing & Tooling": {"Selenium", "Cypress", "Playwright", "pytest", "Postman", "Jira",
                          "Git", "Confluence", "REST APIs", "GraphQL", "gRPC"},
}


def heading(rng: random.Random, key: str) -> str:
    return rng.choice(SECTION_HEADINGS[key])


def format_date_range(rng: random.Random, start_ym: str, end_ym, style: int) -> str:
    """start_ym / end_ym are 'YYYY-MM' strings; end_ym None means current role."""
    sy, sm = int(start_ym[:4]), int(start_ym[5:7])
    present_word = rng.choice(["Present", "Current", "Now"]) if style in (0, 3) else "Present"
    if end_ym is None:
        ey = em = None
    else:
        ey, em = int(end_ym[:4]), int(end_ym[5:7])

    def fmt(year, month):
        if style == 0:
            return f"{MONTHS_SHORT[month - 1]} {year}"
        if style == 1:
            return f"{month:02d}/{year}"
        if style == 2:
            return str(year)
        if style == 3:
            return f"{MONTHS_LONG[month - 1]} {year}"
        if style == 4:
            return f"{MONTHS_SHORT[month - 1]} '{str(year)[2:]}"
        return f"{year}-{month:02d}"

    sep = {0: " – ", 1: " - ", 2: "–", 3: " to ", 4: " – ", 5: " — "}[style]
    end_txt = present_word if ey is None else fmt(ey, em)
    return f"{fmt(sy, sm)}{sep}{end_txt}"


def format_year_range(rng: random.Random, start_year: int, end_year: int, style: int) -> str:
    """Education dates. Real CVs show a range, and sometimes only the finish year."""
    if rng.random() < 0.22:
        return str(end_year)
    sep = {0: " – ", 1: " - ", 2: "–", 3: " to ", 4: " – ", 5: " — "}[style]
    return f"{start_year}{sep}{end_year}"


def _fill(rng: random.Random, template: str, subject: str) -> str:
    ms1 = rng.choice([320, 450, 600, 780, 900, 1200])
    return template.format(
        skill=subject,
        n=rng.choice([2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 40, 50]),
        pct=rng.choice([12, 15, 18, 20, 22, 25, 30, 35, 40, 45, 55, 60, 70]),
        low=rng.choice([4, 6, 8, 10, 20, 30, 40]),
        ms1=ms1,
        ms2=rng.choice([40, 60, 80, 110, 140, 180]),
        ms3=rng.choice([3, 4, 5, 6, 8, 12]),
        mins=rng.choice([30, 40, 45, 60, 90]),
        high_pct=rng.choice([82, 85, 88, 90, 92, 94, 96]),
    )


def make_bullets(rng: random.Random, domain: str, skills: list, count: int,
                 used: set = None) -> list:
    """Achievement lines for one job.

    Only ever names skills the candidate holds, and only where the skill family
    fits the claim. `used` carries across jobs so no sentence repeats on one CV;
    long careers top up from the domain-neutral pool rather than repeating.
    """
    used = used if used is not None else set()
    skill_names = [skill["name"] for skill in skills]

    pool = [entry for entry in ACHIEVEMENTS.get(domain, ACHIEVEMENTS["Software Development"])
            if entry[0] not in used]
    rng.shuffle(pool)
    if len(pool) < count:
        top_up = [entry for entry in GENERIC_ACHIEVEMENTS if entry[0] not in used]
        rng.shuffle(top_up)
        pool += top_up

    out = []
    for template, kind in pool[:count]:
        used.add(template)
        if kind == "any":
            subject = GENERIC_SUBJECT["any"]
        else:
            matching = [name for name in skill_names if name in SKILL_KINDS[kind]]
            subject = rng.choice(matching) if matching else GENERIC_SUBJECT[kind]
        out.append(_fill(rng, template, subject))
    return out


def make_summary(rng: random.Random, record: dict) -> str:
    """Summary tone follows seniority, the way real profiles do."""
    skills = [skill["name"] for skill in record["skills"]] or ["software"]
    while len(skills) < 3:
        skills.append(skills[0])
    years = record["general_experience"]["total_years"]

    # No employment history: never claim a recent employer the CV can't show.
    if not record["companies"]:
        return rng.choice(SUMMARIES_NO_JOB).format(
            domain=record["general_experience"]["domain"],
            skill1=skills[0], skill2=skills[1], skill3=skills[2])

    whole = max(1, int(years))
    years_text = "1 year" if whole == 1 else f"{whole} years"
    bank = SUMMARIES_JUNIOR if years < 3 else SUMMARIES_MID if years < 8 else SUMMARIES_SENIOR
    return rng.choice(bank).format(
        years_text=years_text, domain=record["general_experience"]["domain"],
        skill1=skills[0], skill2=skills[1], skill3=skills[2],
        company=record["companies"][0]["name"], title=record["companies"][0]["title"],
    )


def make_projects(rng: random.Random, record: dict, count: int) -> list:
    """Returns (name, description) pairs referencing the candidate's real stack."""
    skills = [skill["name"] for skill in record["skills"]] or ["Python"]
    chosen = rng.sample(PROJECTS, min(count, len(PROJECTS)))
    out = []
    for name_tpl, desc_tpl in chosen:
        skill = rng.choice(skills)
        out.append((name_tpl.format(skill=skill),
                    desc_tpl.format(skill=skill, n=rng.randint(1, 9))))
    return out


def make_volunteering(rng: random.Random, record: dict) -> str:
    skills = [skill["name"] for skill in record["skills"]] or ["Python"]
    return rng.choice(VOLUNTEERING).format(
        org=rng.choice(VOLUNTEER_ORGS), n=rng.choice([8, 12, 15, 20, 25]),
        skill=rng.choice(skills), domain=record["general_experience"]["domain"].split("/")[0],
    )


def make_publication(rng: random.Random, field: str, year: int) -> str:
    topics = THESIS_TOPICS.get(field, THESIS_TOPICS["Computer Science"])
    return rng.choice(PUBLICATIONS).format(topic=rng.choice(topics), year=year)


def categorise_skills(skills: list) -> list:
    """Group skills into (category, [names]) pairs, dropping empty categories."""
    grouped = []
    for category, members in SKILL_CATEGORIES.items():
        names = [skill["name"] for skill in skills if skill["name"] in members]
        if names:
            grouped.append((category, names))
    return grouped
