# Decision Tree Induction — Project 2

C++ implementation of a Decision Tree Induction classifier for a Machine Learning Algorithms course.

## Current progress
Step 1: Read a labeled CSV dataset, count examples and features, and report class frequencies. The last CSV column is treated as the class label.

The starter program supports simple comma-separated categorical data, without quoted commas.

## Build and run
Run these commands from the repository root:

```bash
g++ -std=c++17 -O2 main.cpp -o decision_tree
./decision_tree data/weather.csv
```

Expected output:

```text
Examples: 8
Features: 2
Class counts:
  No: 4
  Yes: 4
```

## Next steps
- Implement split metrics (tentatively Information Gain, Gain Ratio, and Gini Index; to be confirmed with the instructor).
- Construct the decision tree and add prediction.
- Test predictions and generate confusion matrices, accuracy, precision, recall/sensitivity, and F1-score.
- Benchmark on labeled UCI datasets and compare with three other classifiers.
