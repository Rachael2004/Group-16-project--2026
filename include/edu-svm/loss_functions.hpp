#ifndef EDUSVM_LOSS_FUNCTIONS_HPP
#define EDUSVM_LOSS_FUNCTIONS_HPP

#include <vector>

using namespace std;
namespace edusvm {

class SVMModel; // Forward declaration of SVMModel class
class LossEvaluator {

public:
// Calculates Hinge Loss + Regularization Penalty
static double calculateTotalLoss(const SVMModel& model, 
const vector< vector <double>>& X, 
const vector<int>& y, 
double lambda_reg);

};

} // namespace edusvm

#endif //EDUSVM_LOSS_FUNCTIONS_HPP
