#ifndef EDUSVM_PREPROCESSOR_HPP
#define EDUSVM_PREPROCESSOR_HPP
 
#include <vector>
 
namespace edusvm {
 
class MinMaxScaler {
public:
    // Calculates scaling parameters (min and max values) from the training dataset features
    void fit(const std::vector<std::vector<double>>& features);
 
    // Transforms dataset features in-place to map them strictly onto a standardized [-1.0, 1.0] scale
    void transform(std::vector<std::vector<double>>& features) const;
 
private:
    std::vector<double> m_mins;
    std::vector<double> m_maxs;
};
 
} // namespace edusvm
 
#endif // EDUSVM_PREPROCESSOR_HPP
