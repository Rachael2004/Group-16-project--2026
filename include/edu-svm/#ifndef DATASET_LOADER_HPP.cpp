#ifndef DATASET_LOADER_HPP
#define DATASET_LOADER_HPP

#include <string>
#include <vector>

namespace edusvm {

struct Dataset {
    std::vector<std::vector<double>> features; // Rows of students, columns of features
    std::vector<int> labels;                   // 1 for PASS, -1 for FAIL
};

class DatasetLoader {
public:
    // Reads a CSV file and loads data into the Dataset struct
    static Dataset loadFromCSV(const std::string& filepath);

    // Validates that data contains no missing or physically impossible negative values
    static bool validateDataset(const Dataset& dataset);
};

} // namespace edusvm

#endif // DATASET_LOADER_HPP
