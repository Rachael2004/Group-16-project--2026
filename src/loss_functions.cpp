#include "../include/edu-svm/loss_functions.hpp"
#include "../include/edu-svm/svm_model.hpp" // where SVMModel is defined
#include <algorithm>
#include <cmath>
#include <vector>

using namespace std;
namespace edusvm {

double LossEvaluator::calculateTotalLoss(const SVMModel& model, const vector< vector <double>>& X, 
const vector<int>& y, double lambda_reg) 

{
double hinge_loss_sum = 0.0;
for (size_t i = 0; i < X.size(); ++i) {
double current_val = model.computeHyperplaneValue(X[i]);
// Equation: max(0, 1 - y * (W • X + b))
hinge_loss_sum += std::max(0.0, 1.0 - (y[i] * current_val));

}
    
double l2_regularization = 0.0;
for (double w : model.getWeights()) {
l2_regularization += w * w;
}

return (hinge_loss_sum / X.size()) + (0.5 * lambda_reg * l2_regularization);
}

} // namespace edusvm
