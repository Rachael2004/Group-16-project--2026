#ifndef EDUSVM_SVM_MODEL_HPP
#define EDUSVM_SVM_MODEL_HPP

#include <vector>
#include <cstddef>

namespace edusvm {

class SVMModel {
public:

    explicit SVMModel(std::size_t num_features);

    std::vector<double>& getWeights();

    const std::vector<double>& getWeights() const;

    double& getBias();

    double getBias() const;

    double computeHyperplaneValue(
        const std::vector<double>& x
    ) const;

private:

    std::vector<double> m_weights;
    double m_bias;
};

} // namespace edusvm

#endif