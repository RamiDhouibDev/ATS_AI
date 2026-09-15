# Layer 1 — Extraction

Rule-based ATS parser with an LLM-escalation threshold. Code in `code/`, tests in `tests/`.

## Results (held-out test split)

```
python -m layer1_extraction.code.tune --target 0.95     # fit escalation threshold on TRAIN
python -m layer1_extraction.code.evaluate --split test  # score on held-out TEST
pytest layer1_extraction/tests -q                            # 44 unit tests
```

| group | n | name | degree | skills F1 | jobs F1 | start date | yrs MAE | escalated |
|---|---|---|---|---|---|---|---|---|
| all | 200 | 93.5% | 94.0% | 95.7% | 94.9% | 97.9% | 0.36 | 6.0% |
| **all (text CVs)** | **188** | **99.5%** | **100%** | **98.9%** | **98.0%** | **97.9%** | **0.36** | **0%** |
| easy | 60 | 100% | 100% | 98.9% | 97.4% | 97.7% | 0.46 | 0% |
| medium | 50 | 100% | 100% | 98.0% | 100% | 98.9% | 0.12 | 0% |
| hard | 78 | 98.7% | 100% | 99.5% | 97.4% | 97.4% | 0.42 | 0% |
| scanned | 12 | 0% | 0% | 0% | 0% | 0% | — | 100% |

The two headline rows differ only in whether image-only CVs are included. Those carry no text at all, so they score zero on every field by construction and drag the raw average down; **all (text CVs)** is the meaningful figure for the parser, and the scanned row is the measure of how badly the LLM fallback is needed.

Skills run at **99.7% precision / 91.9% recall** and employers at **96.4% / 93.4%** — the parser still under-reads rather than inventing, which is the right failure direction when a downstream model consumes its output.

Escalation is **exactly** the 12 scanned CVs: every document with a text layer is parsed confidently, and every one without is handed on. Notably the **hard** tier (sidebars, two-column, table grids) now matches easy and medium on every field — reading order, not vocabulary, was what made those documents hard.

### What "training" means here

There is no gradient training — the parser is regex, section-header detection and a vocabulary lookup. The one fitted parameter is the **confidence threshold** for escalating a CV to the LLM extractor, chosen on the train split by `layer1_extraction/code/tune.py` and written to `layer1_extraction/code/config.json`. It is a cost/quality trade-off, not an accuracy maximiser:

| threshold | quality of kept parses | escalated |
|---|---|---|
| 0.00 | 94.1% | 0% |
| **0.05** | **97.4%** | **3.4%** |
| 0.85 | 96.0% | 43.4% |

The curve is flat from 0.05 to 0.70: the parser is either confident and right, or looking at a document with no text at all. The fitted threshold is the cheapest one clearing the target quality on train, and it escalates precisely the scanned CVs.

### How it handles the hard layouts

Each of these came from measuring a specific failure, not from guessing.

- **Reading order** — `pdfplumber` walks a page top-to-bottom, so a sidebar interleaves line-by-line into the job history. The column boundary is found from **raw word coverage** across x: a true gutter is a vertical band no word occupies. Deriving it from assembled lines is circular, because two columns sharing a baseline merge into one line that then appears to span the gutter. Each column is grouped and emitted whole.
- **Names** — taken from the **largest type on page one**, which is where every resume puts the name. Position is unreliable once a sidebar reorders the page; before this, sidebar CVs returned a city from the contact block.
- **Page furniture** — headers and footers are separated from the body, so contact details parked in a header (117 CVs) are still read rather than ignored, and page numbers don't pollute the sections.
- **Employer identification** — tier-3 employers are invented names, so lookup is useless. A job header is split on its separators and the part *without* a job-title keyword is the employer. This covers "Title, Employer", "Employer — Title", and grid layouts that put dates in their own column above the title. The employer line usually carries the dates and location too (`Adobe | February 2024 to Now | Berlin`), so only its leading field is judged — rejecting any line containing a date dropped whole jobs.
- **Fragmented skill rows** — tables and rating bars emit `Kubernetes` and `20 yrs` as separate lines, which reading order can strand far from the skills heading. A line that is *exactly* a known skill is a list entry, never prose, so these are recovered safely. Narrow columns also wrap mid-entry (`GCP (2` / `years)`), so the section is parsed both line-by-line and rejoined.
- **Broken glyphs** — symbol fonts without a Unicode map surface as `(cid:127)` or as an arbitrary letter (`n Vue.js`). The first is normalised to a bullet; for the second, the leading token is dropped only when the remainder is itself a known skill, which cannot invent one.
- **No skills section** — ~2% of CVs list no skills at all. Only for those, skills are read from the experience prose, which is what real applicant tracking systems do: a tool named in an achievement is a genuine claim to it. CVs *with* a skills section never mine prose.

---
