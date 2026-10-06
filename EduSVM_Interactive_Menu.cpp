
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>
#include <algorithm>

using namespace std;

/*
    EduSVM - Student Performance Classification
    =============================================

    Interactive, menu-driven Linear SVM project.

    Features:
      1. Study hours
      2. Attendance (%)
      3. Assignment score (%)
      4. Previous test score (%)

    Labels:
      +1 = PASS
      -1 = FAIL

    Main project scopes:
      - Dataset input and validation
      - Data preprocessing
      - Feature scaling
      - Hyperplane and decision function
      - Margin
      - Hinge loss
      - L2 regularization
      - Gradient-descent training
      - Support-vector identification
      - PASS/FAIL prediction
      - Model evaluation
      - Confusion matrix
      - Accuracy, precision, recall and F1-score
*/


// ============================================================
// SAMPLE DATA STRUCTURE
// ============================================================

struct Sample {
    vector<double> x;
    int y;
};


// ============================================================
// FEATURE SCALER
// ============================================================

class StandardScaler {
private:
    vector<double> mean;
    vector<double> stdev;

public:

    void fit(const vector<Sample>& data) {

        if (data.empty())
            return;

        size_t numberOfSamples = data.size();
        size_t numberOfFeatures = data[0].x.size();

        mean.assign(numberOfFeatures, 0.0);
        stdev.assign(numberOfFeatures, 0.0);

        // Calculate mean
        for (const auto& sample : data) {
            for (size_t j = 0; j < numberOfFeatures; ++j) {
                mean[j] += sample.x[j];
            }
        }

        for (double& value : mean) {
            value /= static_cast<double>(numberOfSamples);
        }

        // Calculate standard deviation
        for (const auto& sample : data) {
            for (size_t j = 0; j < numberOfFeatures; ++j) {

                double difference = sample.x[j] - mean[j];

                stdev[j] += difference * difference;
            }
        }

        for (double& value : stdev) {

            value = sqrt(value / static_cast<double>(numberOfSamples));

            // Prevent division by zero
            if (value < 1e-12)
                value = 1.0;
        }
    }


    vector<double> transform(const vector<double>& x) const {

        vector<double> result(x.size());

        for (size_t j = 0; j < x.size(); ++j) {

            result[j] =
                (x[j] - mean[j]) / stdev[j];
        }

        return result;
    }


    vector<Sample> transformDataset(
        const vector<Sample>& data) const {

        vector<Sample> result = data;

        for (auto& sample : result) {
            sample.x = transform(sample.x);
        }

        return result;
    }
};


// ============================================================
// LINEAR SVM
// ============================================================

class LinearSVM {

private:

    vector<double> weights;
    double bias;

    double learningRate;
    double lambda;

    int epochs;


public:

    LinearSVM(
        double lr = 0.01,
        double regularization = 0.01,
        int numberOfEpochs = 1000)
        : bias(0.0),
          learningRate(lr),
          lambda(regularization),
          epochs(numberOfEpochs) {}


    // --------------------------------------------------------
    // Decision function
    // f(x) = w.x + b
    // --------------------------------------------------------

    double decisionFunction(
        const vector<double>& x) const {

        double score = bias;

        for (size_t j = 0; j < weights.size(); ++j) {
            score += weights[j] * x[j];
        }

        return score;
    }


    // --------------------------------------------------------
    // Prediction
    // --------------------------------------------------------

    int predict(
        const vector<double>& x) const {

        if (decisionFunction(x) >= 0.0)
            return 1;

        return -1;
    }


    // --------------------------------------------------------
    // Hinge loss
    // L = max(0, 1 - y*f(x))
    // --------------------------------------------------------

    double hingeLoss(
        const Sample& sample) const {

        double margin =
            sample.y * decisionFunction(sample.x);

        return max(0.0, 1.0 - margin);
    }


    // --------------------------------------------------------
    // L2 regularization
    // lambda * 1/2 ||w||^2
    // --------------------------------------------------------

    double regularization() const {

        double sum = 0.0;

        for (double value : weights)
            sum += value * value;

        return 0.5 * lambda * sum;
    }


    // --------------------------------------------------------
    // Complete objective function
    // --------------------------------------------------------

    double objective(
        const vector<Sample>& data) const {

        if (data.empty())
            return 0.0;

        double loss = 0.0;

        for (const auto& sample : data)
            loss += hingeLoss(sample);

        loss /= static_cast<double>(data.size());

        return loss + regularization();
    }


    // --------------------------------------------------------
    // Average hinge loss
    // --------------------------------------------------------

    double averageHingeLoss(
        const vector<Sample>& data) const {

        if (data.empty())
            return 0.0;

        double loss = 0.0;

        for (const auto& sample : data)
            loss += hingeLoss(sample);

        return loss /
               static_cast<double>(data.size());
    }


    // --------------------------------------------------------
    // TRAINING USING GRADIENT DESCENT
    // --------------------------------------------------------

    void train(
        const vector<Sample>& data) {

        if (data.empty())
            return;

        size_t numberOfFeatures =
            data[0].x.size();

        weights.assign(
            numberOfFeatures,
            0.0);

        bias = 0.0;


        for (int epoch = 1;
             epoch <= epochs;
             ++epoch) {

            vector<double> gradientW(
                numberOfFeatures,
                0.0);

            double gradientB = 0.0;


            for (const auto& sample : data) {

                double margin =
                    sample.y *
                    decisionFunction(sample.x);


                if (margin < 1.0) {

                    for (size_t j = 0;
                         j < numberOfFeatures;
                         ++j) {

                        gradientW[j] +=
                            -sample.y *
                            sample.x[j];
                    }

                    gradientB +=
                        -sample.y;
                }
            }


            double n =
                static_cast<double>(
                    data.size());


            for (size_t j = 0;
                 j < numberOfFeatures;
                 ++j) {

                gradientW[j] =
                    gradientW[j] / n
                    + lambda * weights[j];

                weights[j] -=
                    learningRate *
                    gradientW[j];
            }


            gradientB /= n;

            bias -=
                learningRate *
                gradientB;
        }
    }


    // --------------------------------------------------------
    // Margin width
    // Margin = 2 / ||w||
    // --------------------------------------------------------

    double marginWidth() const {

        double norm = 0.0;

        for (double value : weights)
            norm += value * value;

        norm = sqrt(norm);

        if (norm < 1e-12)
            return 0.0;

        return 2.0 / norm;
    }


    // --------------------------------------------------------
    // Support vectors
    // --------------------------------------------------------

    vector<int> supportVectorIndices(
        const vector<Sample>& data) const {

        vector<int> indices;

        for (size_t i = 0;
             i < data.size();
             ++i) {

            double functionalMargin =
                data[i].y *
                decisionFunction(data[i].x);


            /*
                For this educational soft-margin SVM,
                samples with functional margin <= 1
                are treated as support-vector candidates.
            */

            if (functionalMargin <= 1.0 + 1e-6)
                indices.push_back(
                    static_cast<int>(i));
        }

        return indices;
    }


    const vector<double>& getWeights() const {
        return weights;
    }


    double getBias() const {
        return bias;
    }
};


// ============================================================
// MODEL EVALUATION
// ============================================================

struct Metrics {

    int truePositive = 0;
    int trueNegative = 0;
    int falsePositive = 0;
    int falseNegative = 0;


    double accuracy() const {

        int total =
            truePositive +
            trueNegative +
            falsePositive +
            falseNegative;

        if (total == 0)
            return 0.0;

        return static_cast<double>(
            truePositive + trueNegative)
            / total;
    }


    double precision() const {

        int denominator =
            truePositive +
            falsePositive;

        if (denominator == 0)
            return 0.0;

        return static_cast<double>(
            truePositive)
            / denominator;
    }


    double recall() const {

        int denominator =
            truePositive +
            falseNegative;

        if (denominator == 0)
            return 0.0;

        return static_cast<double>(
            truePositive)
            / denominator;
    }


    double f1() const {

        double p = precision();
        double r = recall();

        if (p + r == 0.0)
            return 0.0;

        return 2.0 * p * r /
               (p + r);
    }
};


// ============================================================
// VALIDATE DATASET
// ============================================================

bool validateDataset(
    const vector<Sample>& data) {

    if (data.empty()) {

        cout << "\nError: No student data entered.\n";

        return false;
    }


    size_t featureCount =
        data[0].x.size();


    for (size_t i = 0;
         i < data.size();
         ++i) {

        if (data[i].x.size()
            != featureCount) {

            cout <<
                "\nError: Inconsistent feature count.\n";

            return false;
        }


        if (data[i].y != 1 &&
            data[i].y != -1) {

            cout <<
                "\nError: Result must be 1 or -1.\n";

            return false;
        }
    }

    return true;
}


// ============================================================
// EVALUATION FUNCTION
// ============================================================

Metrics evaluate(
    const LinearSVM& model,
    const vector<Sample>& testData) {

    Metrics metrics;


    for (const auto& sample : testData) {

        int predicted =
            model.predict(sample.x);


        if (sample.y == 1 &&
            predicted == 1) {

            ++metrics.truePositive;
        }

        else if (sample.y == -1 &&
                 predicted == -1) {

            ++metrics.trueNegative;
        }

        else if (sample.y == -1 &&
                 predicted == 1) {

            ++metrics.falsePositive;
        }

        else if (sample.y == 1 &&
                 predicted == -1) {

            ++metrics.falseNegative;
        }
    }

    return metrics;
}


// ============================================================
// ENTER TRAINING DATA
// ============================================================

vector<Sample> enterTrainingData() {

    vector<Sample> data;

    int numberOfStudents;


    cout << "\n============================================\n";
    cout << " ENTER TRAINING DATA\n";
    cout << "============================================\n";


    cout << "\nEnter number of students: ";
    cin >> numberOfStudents;


    while (numberOfStudents < 4) {

        cout <<
            "Please enter at least 4 students: ";

        cin >> numberOfStudents;
    }


    for (int i = 0;
         i < numberOfStudents;
         ++i) {

        Sample sample;

        sample.x.resize(4);


        cout << "\n--------------------------------------------\n";
        cout << "Student " << i + 1 << "\n";
        cout << "--------------------------------------------\n";


        cout << "Study hours: ";
        cin >> sample.x[0];


        cout << "Attendance (%): ";
        cin >> sample.x[1];


        cout << "Assignment score (%): ";
        cin >> sample.x[2];


        cout << "Previous test score (%): ";
        cin >> sample.x[3];


        cout <<
            "Result (1 = PASS, -1 = FAIL): ";

        cin >> sample.y;


        while (sample.y != 1 &&
               sample.y != -1) {

            cout <<
                "Invalid result. Enter 1 for PASS "
                "or -1 for FAIL: ";

            cin >> sample.y;
        }


        data.push_back(sample);
    }


    return data;
}


// ============================================================
// DISPLAY RAW DATA
// ============================================================

void displayDataset(
    const vector<Sample>& data) {

    cout << "\n============================================\n";
    cout << " STUDENT DATASET\n";
    cout << "============================================\n\n";


    cout << left
         << setw(10) << "Student"
         << setw(15) << "Study Hours"
         << setw(15) << "Attendance"
         << setw(18) << "Assignment"
         << setw(18) << "Previous Test"
         << "Result\n";


    cout << string(90, '-') << "\n";


    for (size_t i = 0;
         i < data.size();
         ++i) {

        cout << left
             << setw(10) << i + 1
             << setw(15) << data[i].x[0]
             << setw(15) << data[i].x[1]
             << setw(18) << data[i].x[2]
             << setw(18) << data[i].x[3]
             << (data[i].y == 1
                 ? "PASS"
                 : "FAIL")
             << "\n";
    }
}


// ============================================================
// TRAIN MODEL
// ============================================================

void trainModel(
    const vector<Sample>& rawData,
    LinearSVM& svm,
    StandardScaler& scaler,
    vector<Sample>& trainData,
    vector<Sample>& testData) {


    if (!validateDataset(rawData))
        return;


    if (rawData.size() < 4) {

        cout <<
            "\nAt least 4 students are required.\n";

        return;
    }


    // --------------------------------------------------------
    // 80% training, 20% testing
    // --------------------------------------------------------

    size_t splitPoint =
        rawData.size() * 80 / 100;


    if (splitPoint < 2)
        splitPoint = 2;

    if (rawData.size() - splitPoint < 2)
        splitPoint = rawData.size() - 2;


    trainData.assign(
        rawData.begin(),
        rawData.begin()
        + static_cast<long>(splitPoint));


    testData.assign(
        rawData.begin()
        + static_cast<long>(splitPoint),
        rawData.end());


    // --------------------------------------------------------
    // Scaling
    // --------------------------------------------------------

    scaler.fit(trainData);

    trainData =
        scaler.transformDataset(trainData);

    testData =
        scaler.transformDataset(testData);


    // --------------------------------------------------------
    // Train
    // --------------------------------------------------------

    svm.train(trainData);


    cout <<
        "\nModel training completed successfully.\n";
}


// ============================================================
// DISPLAY MODEL INFORMATION
// ============================================================

void displayModelInformation(
    const LinearSVM& svm,
    const vector<Sample>& trainData) {


    cout << "\n============================================\n";
    cout << " SVM MODEL INFORMATION\n";
    cout << "============================================\n";


    cout << "\n[HYPERPLANE]\n";

    const vector<double>& weights =
        svm.getWeights();


    for (size_t i = 0;
         i < weights.size();
         ++i) {

        cout << "w" << i + 1
             << " = "
             << weights[i]
             << "\n";
    }


    cout << "bias = "
         << svm.getBias()
         << "\n";


    cout <<
        "\nDecision function:\n";

    cout <<
        "f(x) = w^T x + b\n";


    cout << "\n[MARGIN]\n";

    cout <<
        "Margin width = "
        << svm.marginWidth()
        << "\n";

    cout <<
        "Formula: 2 / ||w||\n";


    cout << "\n[HINGE LOSS]\n";

    cout <<
        "Average hinge loss = "
        << svm.averageHingeLoss(trainData)
        << "\n";

    cout <<
        "Formula: max(0, 1 - y*f(x))\n";


    cout << "\n[REGULARIZATION]\n";

    cout <<
        "L2 regularization = "
        << svm.regularization()
        << "\n";

    cout <<
        "Formula: lambda * (1/2)||w||^2\n";


    cout <<
        "Total objective = "
        << svm.objective(trainData)
        << "\n";


    cout << "\n[SUPPORT VECTORS]\n";

    vector<int> supportVectors =
        svm.supportVectorIndices(trainData);


    cout <<
        "Support-vector candidates: "
        << supportVectors.size()
        << " / "
        << trainData.size()
        << "\n";


    for (size_t i = 0;
         i < supportVectors.size();
         ++i) {

        cout <<
            "SV "
            << i + 1
            << " -> training record "
            << supportVectors[i] + 1
            << "\n";
    }
}


// ============================================================
// TEST MODEL
// ============================================================

void testModel(
    const LinearSVM& svm,
    const vector<Sample>& testData) {


    if (testData.empty()) {

        cout <<
            "\nNo testing data available.\n";

        return;
    }


    cout << "\n============================================\n";
    cout << " TEST SET PREDICTIONS\n";
    cout << "============================================\n";


    for (size_t i = 0;
         i < testData.size();
         ++i) {

        double score =
            svm.decisionFunction(
                testData[i].x);


        int prediction =
            svm.predict(testData[i].x);


        cout <<
            "\nStudent "
            << i + 1
            << "\n";


        cout <<
            "Actual result : "
            << (testData[i].y == 1
                ? "PASS"
                : "FAIL")
            << "\n";


        cout <<
            "Decision score: "
            << score
            << "\n";


        cout <<
            "Prediction     : "
            << (prediction == 1
                ? "PASS"
                : "FAIL")
            << "\n";
    }
}


// ============================================================
// EVALUATE MODEL
// ============================================================

void evaluateModel(
    const LinearSVM& svm,
    const vector<Sample>& testData) {


    if (testData.empty()) {

        cout <<
            "\nNo testing data available.\n";

        return;
    }


    Metrics metrics =
        evaluate(svm, testData);


    cout << "\n============================================\n";
    cout << " MODEL EVALUATION\n";
    cout << "============================================\n";


    cout << "\nConfusion Matrix\n\n";

    cout <<
        "                 Predicted\n";

    cout <<
        "                 FAIL    PASS\n";


    cout <<
        "Actual FAIL      "
        << setw(5)
        << metrics.trueNegative
        << "   "
        << setw(5)
        << metrics.falsePositive
        << "\n";


    cout <<
        "Actual PASS      "
        << setw(5)
        << metrics.falseNegative
        << "   "
        << setw(5)
        << metrics.truePositive
        << "\n";


    cout << fixed
         << setprecision(2);


    cout <<
        "\nAccuracy : "
        << metrics.accuracy() * 100
        << "%\n";


    cout <<
        "Precision: "
        << metrics.precision() * 100
        << "%\n";


    cout <<
        "Recall   : "
        << metrics.recall() * 100
        << "%\n";


    cout <<
        "F1-score : "
        << metrics.f1() * 100
        << "%\n";
}


// ============================================================
// PREDICT A NEW STUDENT
// ============================================================

void predictNewStudent(
    const LinearSVM& svm,
    const StandardScaler& scaler) {


    cout << "\n============================================\n";
    cout << " PREDICT NEW STUDENT\n";
    cout << "============================================\n";


    vector<double> rawInput(4);


    cout << "\nEnter student's information:\n\n";


    cout << "Study hours: ";
    cin >> rawInput[0];


    cout << "Attendance (%): ";
    cin >> rawInput[1];


    cout << "Assignment score (%): ";
    cin >> rawInput[2];


    cout << "Previous test score (%): ";
    cin >> rawInput[3];


    vector<double> scaledInput =
        scaler.transform(rawInput);


    double score =
        svm.decisionFunction(
            scaledInput);


    int prediction =
        svm.predict(scaledInput);


    cout << "\n--------------------------------------------\n";

    cout <<
        "Decision score: "
        << score
        << "\n";


    cout <<
        "Prediction: "
        << (prediction == 1
            ? "PASS"
            : "FAIL")
        << "\n";

    cout << "--------------------------------------------\n";
}


// ============================================================
// MAIN MENU
// ============================================================

int main() {

    cout << fixed
         << setprecision(4);


    vector<Sample> rawData;

    vector<Sample> trainData;

    vector<Sample> testData;


    StandardScaler scaler;


    LinearSVM svm(
        0.01,   // learning rate
        0.01,   // regularization
        1000);  // epochs


    bool modelTrained = false;


    int choice;


    do {

        cout << "\n\n";
        cout << "================================================\n";
        cout << "       EduSVM - STUDENT PERFORMANCE SYSTEM\n";
        cout << "================================================\n";

        cout << "\n1. Enter training dataset\n";
        cout << "2. Display entered dataset\n";
        cout << "3. Train SVM model\n";
        cout << "4. Display SVM model information\n";
        cout << "5. Test model\n";
        cout << "6. Evaluate model\n";
        cout << "7. Predict a new student\n";
        cout << "8. Exit\n";

        cout << "\nChoose an option: ";
        cin >> choice;


        switch (choice) {


        // ----------------------------------------------------
        // OPTION 1
        // ----------------------------------------------------

        case 1:

            rawData =
                enterTrainingData();

            modelTrained = false;

            cout <<
                "\nStudent dataset entered successfully.\n";

            break;


        // ----------------------------------------------------
        // OPTION 2
        // ----------------------------------------------------

        case 2:

            if (rawData.empty()) {

                cout <<
                    "\nNo dataset has been entered yet.\n";

            } else {

                displayDataset(rawData);
            }

            break;


        // ----------------------------------------------------
        // OPTION 3
        // ----------------------------------------------------

        case 3:

            if (rawData.empty()) {

                cout <<
                    "\nPlease enter a dataset first.\n";

            } else {

                trainModel(
                    rawData,
                    svm,
                    scaler,
                    trainData,
                    testData);

                modelTrained = true;
            }

            break;


        // ----------------------------------------------------
        // OPTION 4
        // ----------------------------------------------------

        case 4:

            if (!modelTrained) {

                cout <<
                    "\nPlease train the model first.\n";

            } else {

                displayModelInformation(
                    svm,
                    trainData);
            }

            break;


        // ----------------------------------------------------
        // OPTION 5
        // ----------------------------------------------------

        case 5:

            if (!modelTrained) {

                cout <<
                    "\nPlease train the model first.\n";

            } else {

                testModel(
                    svm,
                    testData);
            }

            break;


        // ----------------------------------------------------
        // OPTION 6
        // ----------------------------------------------------

        case 6:

            if (!modelTrained) {

                cout <<
                    "\nPlease train the model first.\n";

            } else {

                evaluateModel(
                    svm,
                    testData);
            }

            break;


        // ----------------------------------------------------
        // OPTION 7
        // ----------------------------------------------------

        case 7:

            if (!modelTrained) {

                cout <<
                    "\nPlease train the model first.\n";

            } else {

                predictNewStudent(
                    svm,
                    scaler);
            }

            break;


        // ----------------------------------------------------
        // OPTION 8
        // ----------------------------------------------------

        case 8:

            cout <<
                "\nThank you for using EduSVM.\n";

            break;


        default:

            cout <<
                "\nInvalid option. Please choose 1-8.\n";

            break;
        }


    } while (choice != 8);


    return 0;
}


// ---------------------------------------------------------------
// Dummy test (checks that the compilation environment works)
// ---------------------------------------------------------------
#include <cassert>

void dummy_test() {
    assert(1 + 1 == 2);
    cout << "Test passed: compilation environment works!" << endl;
}
