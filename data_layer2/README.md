# Layer 2 data — role-conditioned scoring

Layer 1 asks *"what does this CV say?"*. Layer 2 asks *"how well does this CV fit
**this** job?"*. A ten-year Django specialist is a strong hire for a backend role
and a weak one for a data science role, and the labels have to reflect that or
the model has nothing to learn.

Three of the four sections are functions of the (candidate, job) pair.
`companies_score` is deliberately not — employer prestige is always a plus and no
posting can switch it off, so it reads the candidate alone and is **identical
across every posting that candidate appears in**. What a posting varies is how
heavily it weights that section.

## Contents

```
gen/
  scoring_rules.py           the role-conditioned labelling formula
  generate_layer2_data.py    job postings, applicant pools, scored pairs
train/
  jobs.jsonl                 400 job postings
  candidates.jsonl           the 800 candidates these pairs reference
  pairs.jsonl                94,848 scored (candidate, job) pairs
  pairs.csv                  the same, flattened for eyeballing
test/
  jobs.jsonl                 170 job postings
  candidates.jsonl           the 200 candidates these pairs reference
  pairs.jsonl                10,158 scored pairs
  pairs.csv
```

`data_layer2/` is **self-contained** — the candidates these pairs reference are
copied in, so training and evaluating Layer 2 never reads `data_layer1/`. Their
Layer-1 intrinsic scores are stripped on the way in: those are derived from the
same fields as these labels and would be an easy accidental leak.

Regenerate with:

```
cd data_layer2/gen
python generate_layer2_data.py --train-jobs 400 --test-jobs 170 --pool-size 250 --seed 42
```

## Job postings

Each posting carries a target domain, seniority, required experience years, a
preferred degree level and (55% of the time) a field, 3–7 required skills with their own `min_years` and importance
`weight`, and **per-posting section weights**. The weights vary by role: some
postings are stack-led, others prize seniority, which is why the overall score
cannot be a fixed formula across jobs.

Required skills are drawn from what people in that domain actually hold in the
**training** population, so postings are plausible and the test split never
influences what jobs ask for.

## Applicant pools

**55% of every pool comes from the job's own domain**, the rest from anywhere.
Sampling uniformly instead would leave 69% of pairs sharing no required skill at
all, because a random person is rarely in the right field — real pools are
self-selecting, with a tail of speculative applications.

The share is a guarantee, not a target, and it sets the pool size rather than the
other way round: a pool can be no larger than the in-domain population supports.
Train pools run 218–250; test pools 50–76, because the test split has only ~33
candidates per domain. Pair count is therefore bought with **more postings**, not
deeper pools — which is also the axis a role-conditioned model is short of.

Asking for a pool bigger than the in-domain population used to be silent: it
filled the remainder from everyone else. Test asked for 250 out of a 200-candidate
split, so every pool was simply everybody — 17% in-domain against train's 53%, and
a stack median of 5 against 14. A held-out set has to differ from train in its
rows, not its shape, so this is now pinned by a test.

## Split hygiene

Train and test are disjoint on **both** axes — train pairs use train candidates
and train jobs; test pairs use test candidates and test jobs. The held-out set
therefore measures generalisation to roles the model has never seen, not just to
new applicants for familiar roles. Verified: no job, candidate or pair overlap.

## Label quality (train split)

| section | min | mean | sd | max |
|---|---|---|---|---|
| education_score | 12 | 75.5 | 19.2 | 100 |
| relevant_experience_score | 0 | 57.1 | 34.3 | 100 |
| stack_experience_score | 0 | 19.7 | 20.8 | 100 |
| companies_score | 11 | 43.4 | 15.2 | 100 |
| overall_score | 5 | 43.9 | 17.1 | 97 |

The signal is real and monotonic rather than decorative:

- **Stack** tracks required-skill coverage: 0% coverage → mean 5.4, 25% → 27.9,
  50% → 52.8, 75% → 73.9, 100% → 94.6.
- **Employer prestige** is always a plus, never a per-posting preference: it is
  one of the four sections, so a job cannot switch it off. A job can only weight
  the section up or down against the other three.
- **Experience** rewards domain fit: same-domain pairs average 67.0 against 45.0
  cross-domain, via a domain-similarity matrix that gives partial credit for
  adjacent fields (Data/AI/ML ↔ Data Analysis transfers at 0.70; DevOps ↔
  Product/Design at 0.20).
- **Within a job**, overall scores span 71 points on average, so a top 20 is a
  meaningful selection rather than a coin flip.

## Label noise

Small gaussian noise (σ=3) stands in for the inconsistency of a human rater.

It is drawn **per pair** for the three sections that depend on the posting, and
**per candidate** for `companies_score`, which does not. Drawing that one per pair
meant a single employment history came out 13 points apart across postings — one
candidate had 17 different company scores — with nothing having changed. That is
variation no model can predict and no reader can explain, and it is what a reader
notices first on opening `pairs.jsonl`.
