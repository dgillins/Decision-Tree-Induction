# Compare three standard classifiers on the EXACT split exported from C++.
# Run: python3 scripts/compare_classifiers.py
# Install dependencies: python3 -m pip install pandas scikit-learn

from pathlib import Path
import csv
import time
import resource

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, precision_recall_fscore_support,
)

train_path = Path("data/mushroom_train.csv")
test_path = Path("data/mushroom_test.csv")
if not train_path.exists() or not test_path.exists():
    raise SystemExit(
        "Split files missing. Run: "
        "./decision_tree data/mushroom.csv information_gain export_split"
    )

train = pd.read_csv(train_path, dtype=str, keep_default_na=False)
test = pd.read_csv(test_path, dtype=str, keep_default_na=False)
if list(train.columns) != list(test.columns):
    raise SystemExit("Train and test column names do not match.")
feature_names = list(train.columns[:-1])
label_name = train.columns[-1]
X_train, y_train = train[feature_names], train[label_name]
X_test, y_test = test[feature_names], test[label_name]
labels = sorted(set(y_train) | set(y_test))
print(f"Training examples: {len(train)}")
print(f"Test examples: {len(test)}")
print(f"Class labels: {labels} (e=edible, p=poisonous)")

# One-hot encoding fits ONLY on training data to prevent data leakage.
# These models are all tested with the same encoded categorical features.
models = {
    "Naive Bayes": BernoulliNB(),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
}

output_rows = []
for name, classifier in models.items():
    preprocessing = ColumnTransformer(
        [("categorical", OneHotEncoder(handle_unknown="ignore"), feature_names)]
    )
    model = Pipeline([("encode", preprocessing), ("classifier", classifier)])

    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - start

    start = time.perf_counter()
    predictions = model.predict(X_test)
    prediction_seconds = time.perf_counter() - start

    acc = accuracy_score(y_test, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, labels=labels, average="macro", zero_division=0
    )
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    print(f"\n===== {name} =====")
    print("Confusion matrix (rows=actual, columns=predicted)")
    print("Labels:", ", ".join(labels))
    print(matrix)
    print(f"Test accuracy: {acc * 100:.4f}%")
    print(f"Macro precision: {precision:.4f}")
    print(f"Macro recall/sensitivity: {recall:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print(f"Training time: {train_seconds:.4f} seconds")
    print(f"Prediction time: {prediction_seconds:.4f} seconds")
    print("Peak Python process RSS so far (KB; cumulative across models):", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    output_rows.append({
        "classifier": name,
        "train_examples": len(train),
        "test_examples": len(test),
        "test_accuracy_percent": round(acc * 100, 4),
        "macro_precision": round(float(precision), 4),
        "macro_recall": round(float(recall), 4),
        "macro_f1": round(float(f1), 4),
        "training_seconds": round(train_seconds, 4),
        "prediction_seconds": round(prediction_seconds, 4),
        "confusion_labels": "|".join(labels),
        "confusion_matrix": ";".join("|".join(str(v) for v in row) for row in matrix),
    })

output = Path("results/classifier_comparison.csv")
output.parent.mkdir(exist_ok=True)
with output.open("w", newline="") as result_file:
    writer = csv.DictWriter(result_file, fieldnames=list(output_rows[0]))
    writer.writeheader()
    writer.writerows(output_rows)
print(f"\nSaved results to {output}")
