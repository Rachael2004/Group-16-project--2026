cmake_minimum_required(VERSION 3.14)

project(EduSVMLibrary VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build the EduSVM library
add_library(edu_svm
    src/dataset_loader.cpp
    src/preprocessor.cpp
    src/svm_model.cpp
    src/loss_functions.cpp
    src/optimizer.cpp
    src/evaluator.cpp
)

# Tell the compiler where the header files are
target_include_directories(edu_svm PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# Build the student demonstration program
add_executable(student_demo
    examples/student_classification_demo.cpp
)

target_link_libraries(student_demo PRIVATE edu_svm)

# Enable testing
enable_testing()

add_executable(test_suite
    tests/test_svm.cpp
)

target_link_libraries(test_suite PRIVATE edu_svm)

add_test(NAME RunLibraryTests COMMAND test_suite)