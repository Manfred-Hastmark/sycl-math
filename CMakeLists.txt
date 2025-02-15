
cmake_minimum_required(VERSION 3.10)

project(sycl-math LANGUAGES CXX)

set(CMAKE_C_COMPILER acpp)
set(CMAKE_CXX_COMPILER acpp)

# SYCL math library
add_library(sycl-math)
set_target_properties(sycl-math PROPERTIES CMAKE_CXX_STANDARD 17)
target_include_directories(sycl-math PUBLIC include)
target_sources(sycl-math 
    PRIVATE 
        src/gemm.cpp
)

add_executable(samples samples/main.cpp)
target_link_libraries(samples PRIVATE sycl-math)
