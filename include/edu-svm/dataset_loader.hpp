#ifndef DATASET_LOADER_HPP
#define DATASET_LOADER_HPP

#include <string>
#include <vector>

class DatasetLoader {
public:
    bool loadFromCSV(const std::string& filepath, 
                     std::vector<std::vector<double>>& outFeatures, 
                     std::vector<double>& outLabels);
};

#endif // DATASET_LOADER_HPP
