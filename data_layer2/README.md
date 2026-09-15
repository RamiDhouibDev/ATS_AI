# Layer 2 data — role-conditioned scoring

Layer 1 asks *"what does this CV say?"*. Layer 2 asks *"how well does this CV fit
**this** job?"*, so every label here is a function of a (candidate, job) pair.
A ten-year Django specialist is a strong hire for a backend role and a weak one
for a data science role, and the labels have to reflect that or the model has
nothing to learn.

## Contents

```
gen/
  scoring_rules.py           the role-conditioned labelling formula
  generate_layer2_data.py    job postings, applicant pools, scored pairs
train/
  jobs.jsonl                 200 job postings
  pairs.jsonl                50,000 scored (candidate, job) pairs
  pairs.csv                  the same, flattened for eyeballing
test/
  jobs.jsonl                 50 job postings
  pairs.jsonl                10,000 scored pairs
  pairs.csv
```

Candidates are **not duplicated here** — they live in `data_layer1/<split>/` and
are referenced by `candidate_id`. Layer 2 consumes the structured fields Layer 1
would produce from those CVs under correct extraction, which is what keeps the
two layers decoupled and separately trainable.

Regenerate with:

```
cd data_layer2/gen
python generate_layer2_data.py --train-jobs 200 --test-jobs 50 --pool-size 250 --seed 42
```

## Job postings

Each posting carries a target domain, seniority, required experience years, a
preferred degree level and (55% of the time) a field, an optional big-tech
preference, 3–7 required skills with their own `min_years` and importance
`weight`, and **per-posting section weights**. The weights vary by role: some
postings are stack-led, others prize seniority, which is why the overall score
cannot be a fixed formula across jobs.

Required skills are drawn from what people in that domain actually hold in the
**training** population, so postings are plausible and the test split never
influences what jobs ask for.

## Applicant pools

Each posting is scored against 250 candidates: **55% from the job's own domain**,
the rest from anywhere. Sampling uniformly instead would leave 69% of pairs
sharing no required skill at all, because a random person is rarely in the right
field — real pools are self-selecting, with a tail of speculative applications.

## Split hygiene

Train and test are disjoint on **both** axes — train pairs use train candidates
and train jobs; test pairs use test candidates and test jobs. The held-out set
therefore measures generalisation to roles the model has never seen, not just to
new applicants for familiar roles. Verified: no job, candidate or pair overlap.

## Label quality (train split)

| section | min | mean | sd | max |
|---|---|---|---|---|
| education_score | 12 | 75.6 | 19.2 | 100 |
| experience_score | 0 | 56.0 | 34.2 | 100 |
| stack_score | 0 | 19.2 | 20.2 | 100 |
| company_score | 7 | 41.9 | 17.1 | 100 |
| overall_score | 4 | 42.9 | 17.1 | 97 |

The signal is real and monotonic rather than decorative:

- **Stack** tracks required-skill coverage: 0% coverage → mean 5.1, 25% → 27.7,
  50% → 52.7, 75% → 74.7, 100% → 94.7.
- **Experience** rewards domain fit: same-domain pairs average 66.4 against 44.4
  cross-domain, via a domain-similarity matrix that gives partial credit for
  adjacent fields (Data/AI/ML ↔ Data Analysis transfers at 0.70; DevOps ↔
  Product/Design at 0.20).
- **Within a job**, overall scores span 68 points on average, so a top 20 is a
  meaningful selection rather than a coin flip.

Small gaussian noise (σ=3) is added per section so the formula is not perfectly
recoverable — the model should learn the shape of the relationship, not
reproduce arithmetic.
