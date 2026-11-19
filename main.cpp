#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
using namespace std;

// Store one example with its features and correct class
struct Example {
    vector<string> features;
    string label;
};

int main(int argc, char* argv[]) {
    // Step 1: Select the dataset
    string fileName = "data/weather.csv";
    if (argc > 1) fileName = argv[1];

    ifstream file(fileName);
    if (!file) {
        cout << "Could not open " << fileName << endl;
        return 1;
    }

    // Step 2: Read feature names from the first line
    string line, value;
    getline(file, line);
    stringstream header(line);
    vector<string> columnNames;
    while (getline(header, value, ',')) columnNames.push_back(value);

    if (columnNames.size() < 2) {
        cout << "The dataset needs features and a class label." << endl;
        return 1;
    }

    // Step 3: Read each example; the last column is the class
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

    // Step 4: Count the examples in each class
    map<string, int> classCounts;
    for (int i = 0; i < (int)data.size(); i++) {
        classCounts[data[i].label]++;
    }

    cout << "Examples: " << data.size() << endl;
    cout << "Features: " << columnNames.size() - 1 << endl;
    cout << "Class counts:" << endl;
    for (auto entry : classCounts) {
        cout << "  " << entry.first << ": " << entry.second << endl;
    }
    return 0;
}
