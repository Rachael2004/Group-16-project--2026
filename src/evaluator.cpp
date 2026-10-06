#include "edu-svm/evaluator.hpp"

namespace edusvm {

int Evaluator::predict(
    const SVMModel& model,
    const std::vector<double>& sample
) {
    return (model.computeHyperplaneValue(sample) >= 0.0) ? 1 : -1;
}

double Evaluator::calculateAccuracy(
    const SVMModel& model,
    const std::vector<std::vector<double>>& X,
    const std::vector<int>& y
) {
    if (X.empty()) {
        return 0.0;
    }

    int correct_predictions = 0;

    for (size_t i = 0; i < X.size(); ++i) {
        if (predict(model, X[i]) == y[i]) {
            correct_predictions++;
        }
    }

    return static_cast<double>(correct_predictions) / X.size();
}

} // namespace edusvm