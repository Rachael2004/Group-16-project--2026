#include "edu-svm/dataset_loader.hpp"
#include <fstream>
#include <sstream>
#include <iostream>

bool DatasetLoader::loadFromCSV(const std::string& filepath, 
                                 std::vector<std::vector<double>>& outFeatures, 
                                 std::vector<double>& outLabels) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open data file at " << filepath << std::endl;
        return false;
    }

    std::string line;
    bool isHeader = true;

    while (std::getline(file, line)) {
        if (isHeader) {
            isHeader = false;
            continue;
        }
        if (line.empty()) continue;

        std::stringstream ss(line);
        std::string value;
        std::vector<double> featureRow;
        double labelValue = 0.0;

        std::vector<std::string> tokens;
        while (std::getline(ss, value, ',')) {
            tokens.push_back(value);
        }

        if (tokens.size() < 2) continue;

        for (size_t i = 0; i < tokens.size(); ++i) {
            try {
                double val = std::stod(tokens[i]);
                if (i == tokens.size() - 1) {
                    labelValue = val;
                } else {
                    featureRow.push_back(val);
                }
            } catch (const std::exception& e) {
                continue;
            }
        }

        if (!featureRow.empty()) {
            outFeatures.push_back(featureRow);
            outLabels.push_back(labelValue);
        }
    }

    file.close();
    return true;
}
