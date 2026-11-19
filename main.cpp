#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <iomanip>
#include <random>
#include <algorithm>
#include <set>
using namespace std;

// Store one example with its features and correct class
struct Example {
    vector<string> features;
    string label;
};

// Entropy measures how mixed the class labels are
double entropy(map<string, int> counts) {
    int total = 0;
    for (auto entry : counts) total += entry.second;
    if (total == 0) return 0.0;

    double result = 0.0;
    for (auto entry : counts) {
        if (entry.second == 0) continue;
        double p = (double)entry.second / total;
        result -= p * log2(p);
    }
    return result;
}

// Gini impurity is zero when every example has the same class
double gini(map<string, int> counts) {
    int total = 0;
    for (auto entry : counts) total += entry.second;
    if (total == 0) return 0.0;

    double result = 1.0;
    for (auto entry : counts) {
        double p = (double)entry.second / total;
        result -= p * p;
    }
    return result;
}

// Hold the three scores for splitting on a feature
struct SplitScores {
    double informationGain = 0.0;
    double gainRatio = 0.0;
    double giniDecrease = 0.0;
};

// Evaluate one categorical feature against the correct class labels
SplitScores evaluateFeature(vector<Example> data, int featureIndex) {
    map<string, int> parentCounts;
    map<string, map<string, int>> groups;

    for (Example example : data) {
        parentCounts[example.label]++;
        groups[example.features[featureIndex]][example.label]++;
    }

    double parentEntropy = entropy(parentCounts);
    double parentGini = gini(parentCounts);
    double weightedEntropy = 0.0;
    double weightedGini = 0.0;
    double splitInformation = 0.0;

    for (auto group : groups) {
        int groupSize = 0;
        for (auto entry : group.second) groupSize += entry.second;

        double p = (double)groupSize / data.size();
        weightedEntropy += p * entropy(group.second);
        weightedGini += p * gini(group.second);
        if (p > 0.0) splitInformation -= p * log2(p);
    }

    SplitScores scores;
    scores.informationGain = parentEntropy - weightedEntropy;
    scores.giniDecrease = parentGini - weightedGini;
    if (splitInformation > 0.0) {
        scores.gainRatio = scores.informationGain / splitInformation;
    }
    return scores;
}



// A leaf predicts a class. An internal node asks about one feature.
struct TreeNode {
    bool isLeaf = true;
    string prediction;
    int featureIndex = -1;
    map<string, TreeNode*> children;
};

// Find the most common class; alphabetical order breaks a tie.
string majorityClass(vector<Example> data) {
    map<string, int> counts;
    for (Example example : data) counts[example.label]++;
    string best = "";
    int bestCount = -1;
    for (auto entry : counts) {
        if (entry.second > bestCount) {
            best = entry.first;
            bestCount = entry.second;
        }
    }
    return best;
}

// Choose which of the three scores to maximize.
double selectedScore(SplitScores scores, string metric) {
    if (metric == "gain_ratio") return scores.gainRatio;
    if (metric == "gini") return scores.giniDecrease;
    return scores.informationGain;
}

// Build one subtree from its training examples.
TreeNode* buildTree(vector<Example> data, vector<int> availableFeatures,
                    string metric) {
    TreeNode* node = new TreeNode();
    node->prediction = majorityClass(data);

    // Stop when all examples have the same label or no features remain.
    bool sameClass = true;
    for (Example example : data) {
        if (example.label != data[0].label) sameClass = false;
    }
    if (sameClass || availableFeatures.empty()) return node;

    int bestFeature = -1;
    double bestScore = 0.0;
    for (int feature : availableFeatures) {
        SplitScores scores = evaluateFeature(data, feature);
        double score = selectedScore(scores, metric);
        if (score > bestScore + 1e-12) {
            bestScore = score;
            bestFeature = feature;
        }
    }

    // No useful split: predict the majority class.
    if (bestFeature == -1) return node;

    node->isLeaf = false;
    node->featureIndex = bestFeature;
    map<string, vector<Example>> groups;
    for (Example example : data) {
        groups[example.features[bestFeature]].push_back(example);
    }

    // Categorical features are used at most once along a path.
    vector<int> remainingFeatures;
    for (int feature : availableFeatures) {
        if (feature != bestFeature) remainingFeatures.push_back(feature);
    }
    for (auto group : groups) {
        node->children[group.first] = buildTree(group.second, remainingFeatures, metric);
    }
    return node;
}

// Follow branches to get a prediction for a new example.
string predict(TreeNode* node, Example example) {
    while (!node->isLeaf) {
        string value = example.features[node->featureIndex];
        if (node->children.count(value) == 0) {
            return node->prediction; // Unseen category: majority fallback
        }
        node = node->children[value];
    }
    return node->prediction;
}

// Print the rules in an easy-to-read format.
void printTree(TreeNode* node, vector<string> columnNames, int depth = 0) {
    string indent(depth * 2, ' ');
    if (node->isLeaf) {
        cout << indent << "Predict: " << node->prediction << endl;
        return;
    }
    for (auto child : node->children) {
        cout << indent << columnNames[node->featureIndex]
             << " = " << child.first << endl;
        printTree(child.second, columnNames, depth + 1);
    }
}

void deleteTree(TreeNode* node) {
    for (auto child : node->children) deleteTree(child.second);
    delete node;
}


// Make a repeatable 75/25 class-stratified split.
// Keep at least one training example per class whenever possible.
void splitData(vector<Example> data, vector<Example>& training,
               vector<Example>& testing) {
    map<string, vector<Example>> byClass;
    for (Example example : data) byClass[example.label].push_back(example);

    mt19937 rng(42);
    for (auto group : byClass) {
        vector<Example> examples = group.second;
        shuffle(examples.begin(), examples.end(), rng);
        int n = (int)examples.size();
        int testCount = (int)(n * 0.25 + 0.5);
        if (n >= 2 && testCount < 1) testCount = 1;
        if (testCount >= n) testCount = n - 1;

        for (int i = 0; i < n; i++) {
            if (i < testCount) testing.push_back(examples[i]);
            else training.push_back(examples[i]);
        }
    }
}

// A zero denominator means the corresponding measure is undefined.
// Return zero for a consistent display in that case.
double safeDivide(double numerator, double denominator) {
    if (denominator == 0.0) return 0.0;
    return numerator / denominator;
}

// Print a multiclass confusion matrix and one-vs-rest class metrics.
// Matrix rows are actual classes; columns are predicted classes.
void evaluateModel(TreeNode* tree, vector<Example> testing) {
    set<string> classSet;
    for (Example example : testing) {
        classSet.insert(example.label);
        classSet.insert(predict(tree, example));
    }
    vector<string> labels(classSet.begin(), classSet.end());
    map<string, map<string, int>> matrix;
    int correct = 0;
    for (Example example : testing) {
        string predicted = predict(tree, example);
        matrix[example.label][predicted]++;
        if (predicted == example.label) correct++;
    }

    cout << "\nConfusion Matrix (rows = actual, columns = predicted):" << endl;
    cout << setw(14) << "Actual/Pred";
    for (string label : labels) cout << setw(14) << label;
    cout << endl;
    for (string actual : labels) {
        cout << setw(14) << actual;
        for (string predicted : labels) {
            cout << setw(14) << matrix[actual][predicted];
        }
        cout << endl;
    }

    cout << "\nTest accuracy: " << 100.0 * safeDivide(correct, testing.size()) << "%" << endl;
    cout << "Metrics are computed one-vs-rest for each class." << endl;
    cout << left << setw(14) << "Class" << right
         << setw(12) << "Precision" << setw(12) << "Recall"
         << setw(14) << "Specificity" << setw(12) << "F1" << endl;

    double macroPrecision = 0, macroRecall = 0, macroSpecificity = 0, macroF1 = 0;
    int total = (int)testing.size();
    for (string label : labels) {
        int tp = matrix[label][label];
        int fn = 0, fp = 0;
        for (string other : labels) {
            if (other != label) {
                fn += matrix[label][other];
                fp += matrix[other][label];
            }
        }
        int tn = total - tp - fn - fp;
        double precision = safeDivide(tp, tp + fp);
        double recall = safeDivide(tp, tp + fn);
        double specificity = safeDivide(tn, tn + fp);
        double f1 = safeDivide(2 * precision * recall, precision + recall);
        macroPrecision += precision;
        macroRecall += recall;
        macroSpecificity += specificity;
        macroF1 += f1;
        cout << left << setw(14) << label << right
             << setw(12) << precision << setw(12) << recall
             << setw(14) << specificity << setw(12) << f1 << endl;
    }
    double count = labels.size();
    cout << left << setw(14) << "Macro avg" << right
         << setw(12) << macroPrecision / count
         << setw(12) << macroRecall / count
         << setw(14) << macroSpecificity / count
         << setw(12) << macroF1 / count << endl;
}

int main(int argc, char* argv[]) {
    string fileName = "data/weather.csv";
    if (argc > 1) fileName = argv[1];

    ifstream file(fileName);
    if (!file) {
        cout << "Could not open " << fileName << endl;
        return 1;
    }

    // The last column contains the class label
    string line, value;
    getline(file, line);
    stringstream header(line);
    vector<string> columnNames;
    while (getline(header, value, ',')) columnNames.push_back(value);

    if (columnNames.size() < 2) {
        cout << "The dataset needs features and a class label." << endl;
        return 1;
    }

    vector<Example> data;
    while (getline(file, line)) {
        if (line.empty()) continue;
        stringstream row(line);
        vector<string> values;
        while (getline(row, value, ',')) values.push_back(value);
        if (values.size() != columnNames.size()) {
            cout << "Skipping a row with the wrong number of columns." << endl;
            continue;
        }

        Example example;
        for (int i = 0; i < (int)values.size() - 1; i++) {
            example.features.push_back(values[i]);
        }
        example.label = values.back();
        data.push_back(example);
    }
    file.close();

    if (data.empty()) {
        cout << "No examples found." << endl;
        return 1;
    }

    map<string, int> classCounts;
    for (Example example : data) classCounts[example.label]++;

    cout << "Examples: " << data.size() << endl;
    cout << "Features: " << columnNames.size() - 1 << endl;
    cout << "Class counts:" << endl;
    for (auto entry : classCounts) {
        cout << "  " << entry.first << ": " << entry.second << endl;
    }

    // Add 'evaluate' as the third argument for held-out testing.
    bool evaluate = (argc > 3 && string(argv[3]) == "evaluate");
    vector<Example> trainingData, testData;
    if (evaluate) {
        splitData(data, trainingData, testData);
        if (trainingData.empty() || testData.empty()) {
            cout << "Not enough data for a train/test split." << endl;
            return 1;
        }
        cout << "\nTrain/test split (seed 42): "
             << trainingData.size() << " train, " << testData.size() << " test" << endl;
    } else {
        trainingData = data;
    }

    // All evaluation-mode feature scores must use training data only.
    map<string, int> trainingCounts;
    for (Example example : trainingData) trainingCounts[example.label]++;

    cout << fixed << setprecision(4);
    cout << "\nParent entropy: " << entropy(trainingCounts) << endl;
    cout << "Parent Gini: " << gini(trainingCounts) << endl;
    cout << "\nFeature split scores:" << endl;
    cout << "Feature            Info Gain   Gain Ratio   Gini Decrease" << endl;

    double bestGain = -1.0, bestRatio = -1.0, bestGini = -1.0;
    string bestGainFeature, bestRatioFeature, bestGiniFeature;

    for (int i = 0; i < (int)columnNames.size() - 1; i++) {
        SplitScores s = evaluateFeature(trainingData, i);
        cout << left << setw(19) << columnNames[i]
             << right << setw(10) << s.informationGain
             << setw(13) << s.gainRatio
             << setw(16) << s.giniDecrease << endl;

        if (s.informationGain > bestGain) {
            bestGain = s.informationGain;
            bestGainFeature = columnNames[i];
        }
        if (s.gainRatio > bestRatio) {
            bestRatio = s.gainRatio;
            bestRatioFeature = columnNames[i];
        }
        if (s.giniDecrease > bestGini) {
            bestGini = s.giniDecrease;
            bestGiniFeature = columnNames[i];
        }
    }

    cout << "\nBest first split by Information Gain: " << bestGainFeature << endl;
    cout << "Best first split by Gain Ratio: " << bestRatioFeature << endl;
    cout << "Best first split by Gini Decrease: " << bestGiniFeature << endl;
    // Step 3: Learn a tree and predict the labels of our small test dataset.
    // These are training-set predictions, not an unbiased test accuracy.
    string metric = "information_gain";
    if (argc > 2) metric = argv[2];
    if (metric != "information_gain" && metric != "gain_ratio" && metric != "gini") {
        cout << "Metric must be information_gain, gain_ratio, or gini." << endl;
        return 1;
    }
    vector<int> features;
    for (int i = 0; i < (int)columnNames.size() - 1; i++) features.push_back(i);
    TreeNode* tree = buildTree(trainingData, features, metric);

    cout << "\nDecision Tree (metric: " << metric << "):" << endl;
    printTree(tree, columnNames);

    if (evaluate) {
        evaluateModel(tree, testData);
    } else {
        int correct = 0;
        cout << "\nTraining examples: Actual -> Predicted" << endl;
        for (int i = 0; i < (int)data.size(); i++) {
            string predicted = predict(tree, data[i]);
            if (predicted == data[i].label) correct++;
            cout << "Example " << i + 1 << ": " << data[i].label
                 << " -> " << predicted << endl;
        }
        cout << "Training accuracy: " << (100.0 * correct / data.size()) << "%" << endl;
    }

    deleteTree(tree);
    return 0;
}
