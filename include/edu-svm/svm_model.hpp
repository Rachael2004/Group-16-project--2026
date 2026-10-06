#ifndef SVM_MODEL_HPP
#define SVM_MODEL_HPP
#include <vector>

using namespace std;
namespace edusvm {

class SVMModel {
public:
 //matches the model.getWeights() 
    SVMModel(const vector<double>& weights, double bias);
    double computeHyperplaneValue(const vector<double>& x) const;
    const vector<double>& getWeights() const;
    double getBias() const;

private:
vector<double> weights;
double bias;

};

} 

#endif // SVM_MODEL_HPP