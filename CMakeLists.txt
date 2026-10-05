cmake_minimum_required(VERSION 3.14)
project(EduSVM VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define the library source targets
file(GLOB_RECURSIVE LIB_SOURCES "src/*.cpp")
add_library(edusvm ${LIB_SOURCES})

target_include_directories(edusvm PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Core demonstration application entry point
add_executable(student_demo examples/student_classification_demo.cpp)
target_link_libraries(student_demo PRIVATE edusvm)
