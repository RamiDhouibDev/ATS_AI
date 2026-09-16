# Project conventions

## Naming

**Never use single-letter variable names.** No exceptions for the usual Python
idioms — loop and comprehension variables, lambda parameters, `except ... as e`,
`with open() as f` all get real names.

```python
# no
for s in skills: ...
labels = {k: v for k, v in pair.items()}
with path.open() as f: ...

# yes
for skill in skills: ...
labels = {field: score for field, score in pair.items()}
with path.open() as handle: ...
```

Prefer iterating objects directly over indexing. Where an index is genuinely
needed, call it `index`.

The one allowed single character is `_`, and only where the value is deliberately
discarded (`for _ in range(n)`, `first, _ = pair`). It is a placeholder, not a name.
