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

## Step 5: UCI Mushroom classification benchmark

Dataset: https://archive.ics.uci.edu/dataset/73/mushroom (8,124 labeled mushrooms, 22 categorical features).

1. Download the original `agaricus-lepiota.data` file from UCI into `data/`.
2. From the repository root run:

```bash
python3 scripts/prepare_mushroom.py
g++ -std=c++17 -O2 main.cpp -o decision_tree
./decision_tree data/mushroom.csv information_gain evaluate
./decision_tree data/mushroom.csv gain_ratio evaluate
./decision_tree data/mushroom.csv gini evaluate
```

The converter moves the original first-column class label to the last column to match the C++ reader and keeps missing `?` values as their own category. The file `data/mushroom.csv` is generated locally and does not need to be committed.

**Important:** Evaluation output contains a large decision tree for this dataset. For easier reading, redirect results to files, e.g. `./decision_tree data/mushroom.csv gini evaluate > gini_mushroom_output.txt`.

The current tree supports categorical multiway splits only and has no pruning. Benchmark results should be interpreted with those limitations. Do not treat the eight-example weather experiment as a performance benchmark.

## Step 6: Compare against three reference classifiers

The C++ program exports the **same deterministic stratified split** used in its evaluation mode:

```bash
g++ -std=c++17 -O2 main.cpp -o decision_tree
./decision_tree data/mushroom.csv information_gain export_split
```

This creates `data/mushroom_train.csv` and `data/mushroom_test.csv`.

Install Python dependencies and compare Naive Bayes, KNN (k=5), and Logistic Regression:

```bash
python3 -m pip install pandas scikit-learn
python3 scripts/compare_classifiers.py
```

The script fits a one-hot encoder on the training data only and prints held-out confusion matrices, accuracy, macro precision, macro sensitivity/recall, and macro F1. It saves a CSV summary to `results/classifier_comparison.csv`.

We use `BernoulliNB` for binary one-hot features; KNN uses `k=5` and Euclidean distance on the encoded vectors; Logistic Regression uses `max_iter=1000`. These are **reference implementations** from scikit-learn, not custom C++ implementations.

For fairness, compare accuracy and confusion matrices using the exported split. Be careful comparing C++ total process time or peak RSS directly against Python, which includes interpreter, library and preprocessing overhead. Report training and prediction times separately where possible.
