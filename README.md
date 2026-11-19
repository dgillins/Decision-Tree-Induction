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

## Step 4: Held-out classification evaluation

The original training demonstration remains available:

```bash
g++ -std=c++17 -O2 main.cpp -o decision_tree
./decision_tree data/weather.csv information_gain
```

For classification evaluation, add `evaluate`:

```bash
./decision_tree data/weather.csv information_gain evaluate
./decision_tree data/weather.csv gain_ratio evaluate
./decision_tree data/weather.csv gini evaluate
```

The evaluation mode creates a reproducible class-stratified ~75/25 train/test split with random seed 42 (small classes may be rounded differently). It fits the tree **only** on training examples and prints a held-out confusion matrix (rows=actual, columns=predicted), overall test accuracy, per-class precision, recall (sensitivity), specificity and F1, and macro averages.

Notes:
- The tiny eight-example weather dataset is intended only for program validation, not a meaningful estimate of generalization accuracy.
- The split is reproducible, but for larger experiments we should run multiple splits or cross-validation.
- Currently, categorical features are expected; continuous UCI features will need threshold splitting or other preprocessing.
- When a denominator for a class metric is zero, the program displays zero by convention.
