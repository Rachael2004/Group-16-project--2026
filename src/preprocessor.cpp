#include "edu-svm/preprocessor.hpp"
#include <stdexcept>
#include <cmath>
 
namespace edusvm {
 
void MinMaxScaler::fit(const std::vector<std::vector<double>>& features) {
    if (features.empty() || features[0].empty()) {
        throw std::invalid_argument("Error: Cannot fit scaler on an empty feature array.");
    }
 
    size_t num_features = features[0].size();
    m_mins = features[0];
    m_maxs = features[0];
 
    // Scan across columns to track down individual min and max data ranges
    for (const auto& row : features) {
        if (row.size() != num_features) {
            throw std::invalid_argument("Error: Inconsistent row widths detected in dataset array.");
        }
        for (size_t i = 0; i < num_features; ++i) {
            if (row[i] < m_mins[i]) m_mins[i] = row[i];
            if (row[i] > m_maxs[i]) m_maxs[i] = row[i];
        }
    }
}
 
void MinMaxScaler::transform(std::vector<std::vector<double>>& features) const {
    if (features.empty()) return;
 
    size_t num_features = features[0].size();
    for (auto& row : features) {
        for (size_t i = 0; i < num_features; ++i) {
            double range = m_maxs[i] - m_mins[i];
 
            // Strict edge case boundary validation: Avoid division by zero if values are uniform
            if (range == 0.0) {
                row[i] = 0.0;
            } else {
                // Scales the absolute number explicitly into a clean [-1.0, 1.0] mathematical frame
                row[i] = 2.0 * ((row[i] - m_mins[i]) / range) - 1.0;
            }
        }
    }
}
 
} // namespace edusvm
