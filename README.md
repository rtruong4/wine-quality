# wine-quality

## Preprocessing

Clean the dataset:

```bash
python preprocessing.py
```

`preprocessing.py` uses only the Python standard library and saves
`wine_quality_cleaned.csv`, preserving the original CSV. It:

- Validates the expected columns and trims surrounding whitespace.
- Normalizes quality-label capitalization.
- Removes rows with missing, nonnumeric, non-finite, or negative feature values,
  unknown quality labels, or malformed CSV fields.
- Rejects zero fixed acidity, density, pH, or alcohol, and pH above 14.
  Other zeros, including citric acid, are retained.
- Removes duplicate rows after normalization.
- Reports rows where free sulfur dioxide exceeds total sulfur dioxide for
  source review; these rows are retained pending confirmation.

These are basic validity rules, not a guarantee of chemical plausibility.
The script prints counts of removed rows. It does not split, scale, encode,
impute, rebalance classes, or remove statistical outliers. Any transformations
that learn from data should be fitted on training data after the future split.
