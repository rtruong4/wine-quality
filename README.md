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

## Training

Run the complete model comparison after preprocessing:

```bash
python train.py
```

`train.py` uses an 80/10/10 training, validation, and test split. Set
`use_stratification` to `True` or `False` in the script to compare stratified
and non-stratified splits.

The script trains and evaluates these models:

- Logistic Regression
- Tuned and default Decision Tree
- Tuned and default Gradient Boosting
- Tuned and default Support Vector Machine

Logistic Regression and SVM use standardization. Decision Tree and Gradient
Boosting models do not require feature scaling. The quality labels remain as
text because the scikit-learn classifiers handle target-label encoding
internally.

Results are written to `results.txt`, including training, validation, and test
accuracy for every model. The script does not generate charts or print the
detailed results to the terminal; it only prints the location of the results
file.
