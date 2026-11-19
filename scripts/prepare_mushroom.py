# Prepare the UCI Mushroom dataset for decision-tree classification.
# Usage: python3 scripts/prepare_mushroom.py
# Source: https://archive.ics.uci.edu/dataset/73/mushroom

import csv
from pathlib import Path

source = Path("data/agaricus-lepiota.data")
output = Path("data/mushroom.csv")

if not source.exists():
    print("Missing data/agaricus-lepiota.data")
    print("Download the original Mushroom dataset from the UCI page:")
    print("https://archive.ics.uci.edu/dataset/73/mushroom")
    print("Save agaricus-lepiota.data in the data/ folder, then rerun.")
    raise SystemExit(1)

# UCI places the class first; our C++ program expects it last.
# '?' represents missing data and stays an explicit category for now.
feature_names = ["feature_" + str(i) for i in range(1, 23)]
rows = []
with source.open(newline="") as input_file:
    for row_number, row in enumerate(csv.reader(input_file), start=1):
        if len(row) != 23:
            raise ValueError(f"Row {row_number} has {len(row)} fields; expected 23")
        label = row[0].strip()
        if label not in ("e", "p"):
            raise ValueError(f"Unknown class label at row {row_number}: {label}")
        features = [value.strip() for value in row[1:]]
        rows.append(features + [label])

if len(rows) != 8124:
    print("Warning: expected 8124 mushroom rows, found", len(rows))

with output.open("w", newline="") as output_file:
    writer = csv.writer(output_file)
    writer.writerow(feature_names + ["class"])
    writer.writerows(rows)

print("Created:", output)
print("Examples:", len(rows))
print("Features:", len(feature_names))
print("Edible:", sum(row[-1] == "e" for row in rows))
print("Poisonous:", sum(row[-1] == "p" for row in rows))
