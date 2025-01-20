cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(hpim)

find_package(Torch REQUIRED)
# find_package(OpenCV REQUIRED)

# Define library target
add_library(hpim SHARED src/matmul.cpp)
# Enable C++14
target_compile_features(hpim PRIVATE cxx_std_14)
# Link against LibTorch
target_link_libraries(hpim "${TORCH_LIBRARIES}")
