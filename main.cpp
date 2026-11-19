#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <cmath>
#include <iomanip>
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

    cout << fixed << setprecision(4);
    cout << "\nParent entropy: " << entropy(classCounts) << endl;
    cout << "Parent Gini: " << gini(classCounts) << endl;
    cout << "\nFeature split scores:" << endl;
    cout << "Feature            Info Gain   Gain Ratio   Gini Decrease" << endl;

    double bestGain = -1.0, bestRatio = -1.0, bestGini = -1.0;
    string bestGainFeature, bestRatioFeature, bestGiniFeature;

    for (int i = 0; i < (int)columnNames.size() - 1; i++) {
        SplitScores s = evaluateFeature(data, i);
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
    return 0;
}
