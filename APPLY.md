# Equivariant channel-collapse fix

Replace these files in the repository:

- `cryo_registration/fine_point_matching.py`
- add `tests/test_equivariant_channel_diversity.py`

Then run:

```bash
pytest -q tests/test_fine_point_matching.py \
  tests/test_fine_point_matching_training.py \
  tests/test_equivariant_channel_diversity.py
```

Start a new FinePointMatcher training run. Do not resume the old optimizer state,
because the head forward geometry and equivariant objective have changed even
though the parameter key shapes remain compatible.
