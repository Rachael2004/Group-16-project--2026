#include "edu-svm/optimizer.hpp"

#include <cstddef>
#include <stdexcept>

namespace edusvm {

GradientDescentOptimizer::GradientDescentOptimizer(
    double learning_rate, double lambda_reg, int epochs)
    : m_lr(learning_rate), m_lambda(lambda_reg), m_epochs(epochs) {
    if (m_lr <= 0.0 || m_lambda < 0.0 || m_epochs <= 0) {
        throw std::invalid_argument("Invalid optimizer settings");
    }
}

void GradientDescentOptimizer::train(
    SVMModel& model,
    const std::vector<std::vector<double>>& X,
    const std::vector<int>& y) {
    if (X.empty() || X.size() != y.size()) {
        throw std::invalid_argument("X and y must have the same nonzero size");
    }

    auto& weights = model.getWeights();
    double& bias = model.getBias();

    for (std::size_t i = 0; i < X.size(); ++i) {
        if (X[i].size() != weights.size() || (y[i] != -1 && y[i] != 1)) {
            throw std::invalid_argument("Invalid feature count or label");
        }
    }

    for (int epoch = 0; epoch < m_epochs; ++epoch) {
        for (std::size_t i = 0; i < X.size(); ++i) {
            const double condition = y[i] * model.computeHyperplaneValue(X[i]);

            for (std::size_t j = 0; j < weights.size(); ++j) {
                if (condition < 1.0) {
                    weights[j] -= m_lr * (m_lambda * weights[j] - y[i] * X[i][j]);
                } else {
                    weights[j] -= m_lr * m_lambda * weights[j];
                }
            }

            if (condition < 1.0) {
                bias += m_lr * y[i];
            }
        }
    }
}

} // namespace edusvm