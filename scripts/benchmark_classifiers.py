# Run each scikit-learn classifier in a fresh process, five times.
# Usage: python3 scripts/benchmark_classifiers.py
# Requires GNU time: sudo apt install time -y
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from statistics import mean

MODELS = ["naive_bayes", "knn", "logistic_regression"]
RUNS = 5
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)
records = []

with tempfile.TemporaryDirectory() as temp_dir:
    for model in MODELS:
        print(f"\n===== {model} =====", flush=True)
        for run in range(1, RUNS + 1):
            json_path = Path(temp_dir) / "metrics.json"
            time_path = Path(temp_dir) / "time.txt"
            command = [
                "/usr/bin/time", "-f", "%e,%M", "-o", str(time_path),
                sys.executable, "scripts/compare_classifiers.py",
                "--model", model, "--json-output", str(json_path), "--quiet"
            ]
            subprocess.run(command, check=True)
            elapsed_text, rss_text = time_path.read_text().strip().split(",")
            metrics = json.loads(json_path.read_text())
            record = {
                "classifier": metrics["classifier"],
                "model_key": model,
                "run": run,
                "train_examples": metrics["train_examples"],
                "test_examples": metrics["test_examples"],
                "test_accuracy_percent": metrics["test_accuracy_percent"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "training_seconds": metrics["training_seconds"],
                "prediction_seconds": metrics["prediction_seconds"],
                "total_process_seconds": float(elapsed_text),
                "peak_memory_kb": int(rss_text),
                "confusion_labels": metrics["confusion_labels"],
                "confusion_matrix": metrics["confusion_matrix"],
            }
            records.append(record)
            print(
                f"Run {run}: total {record['total_process_seconds']:.2f} s, "
                f"train {record['training_seconds']:.4f} s, "
                f"predict {record['prediction_seconds']:.4f} s, "
                f"peak {record['peak_memory_kb']} KB, "
                f"accuracy {record['test_accuracy_percent']:.4f}%",
                flush=True
            )

runs_path = RESULTS / "classifier_benchmark_runs.csv"
with runs_path.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)

summary = []
for model in MODELS:
    group = [row for row in records if row["model_key"] == model]
    if len({(row["confusion_labels"], row["confusion_matrix"]) for row in group}) != 1:
        print("WARNING: Predictions differed between runs for", model)
    summary.append({
        "classifier": group[0]["classifier"],
        "runs": len(group),
        "test_accuracy_percent": group[0]["test_accuracy_percent"],
        "macro_precision": group[0]["macro_precision"],
        "macro_recall": group[0]["macro_recall"],
        "macro_f1": group[0]["macro_f1"],
        "avg_training_seconds": round(mean(r["training_seconds"] for r in group), 4),
        "avg_prediction_seconds": round(mean(r["prediction_seconds"] for r in group), 4),
        "avg_total_process_seconds": round(mean(r["total_process_seconds"] for r in group), 4),
        "avg_peak_memory_kb": round(mean(r["peak_memory_kb"] for r in group), 1),
    })

summary_path = RESULTS / "classifier_benchmark_summary.csv"
with summary_path.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)

print("\n===== FIVE-RUN AVERAGES =====")
for row in summary:
    print(
        f"{row['classifier']}: "
        f"accuracy {row['test_accuracy_percent']:.4f}%, "
        f"train {row['avg_training_seconds']:.4f} s, "
        f"predict {row['avg_prediction_seconds']:.4f} s, "
        f"total {row['avg_total_process_seconds']:.4f} s, "
        f"memory {row['avg_peak_memory_kb']:.1f} KB"
    )
print(f"\nSaved: {runs_path}")
print(f"Saved: {summary_path}")
