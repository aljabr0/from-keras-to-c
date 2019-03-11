cmake_minimum_required(VERSION 3.13)
project(keras_native_run)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_PREFIX_PATH ./lib)
include_directories(./include)
find_library (TF_LIB NAMES libtensorflow tensorflow)

add_executable(model_run model_run.cpp)
target_link_libraries(model_run ${TF_LIB})
