"""Clean the wine dataset without splitting it or fitting transformations."""

import csv
import math
from pathlib import Path


DATA_PATH = Path(__file__).with_name("wine_quality.csv")
OUTPUT_PATH = Path(__file__).with_name("wine_quality_cleaned.csv")
FEATURES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol",
]
QUALITY_LABELS = {label.lower(): label for label in
                  ["Low", "Medium-Low", "Medium", "Medium-High", "High"]}


def clean_data(input_path=DATA_PATH, output_path=OUTPUT_PATH):
    if Path(input_path).resolve() == Path(output_path).resolve():
        raise ValueError("Input and output file name are the same.")

    cleaned_rows = []
    seen = set()
    counts = {"input": 0, "invalid": 0, "duplicates": 0}

    with Path(input_path).open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        # Normalize surrounding whitespace in column names.
        columns = [name.strip() for name in (reader.fieldnames or [])]
        if len(columns) != len(set(columns)) or set(columns) != set(FEATURES + ["quality"]):
            raise ValueError("Expected 11 wine features and a quality target column.")
        reader.fieldnames = columns

        for row in reader:
            counts["input"] += 1
            # Reject incomplete rows and rows with extra CSV fields.
            if None in row or any(value is None for value in row.values()):
                counts["invalid"] += 1
                continue

            try:
                values = {column: float(row[column].strip()) for column in FEATURES}
            except ValueError:
                counts["invalid"] += 1
                continue

            quality = QUALITY_LABELS.get(row["quality"].strip().lower())
            invalid_number = any(not math.isfinite(v) or v < 0 for v in values.values())
            # These measurements must be positive. Other zeros are retained.
            invalid_zero = any(values[c] == 0 for c in ["fixed acidity", "density", "pH", "alcohol"])
            if quality is None or invalid_number or invalid_zero or values["pH"] > 14:
                counts["invalid"] += 1
                continue

            # Compare parsed values so '1' and '1.0' count as the same value.
            key = tuple(values[column] for column in FEATURES) + (quality,)
            if key in seen:
                counts["duplicates"] += 1
                continue
            seen.add(key)
            cleaned_rows.append({**values, "quality": quality})

    if not cleaned_rows:
        raise ValueError("No valid rows remain; no output file was written.")

    with Path(output_path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FEATURES + ["quality"])
        writer.writeheader()
        writer.writerows(cleaned_rows)

    counts["output"] = len(cleaned_rows)
    return counts


if __name__ == "__main__":
    summary = clean_data()
    print(f"Input samples: {summary['input']}")
    print(f"Invalid rows removed: {summary['invalid']}")
    print(f"Duplicate rows removed: {summary['duplicates']}")
    print(f"Cleaned samples: {summary['output']}")
    print(f"Saved to: {OUTPUT_PATH.name}")
