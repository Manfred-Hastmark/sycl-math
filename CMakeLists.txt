# Main cmake file for using/building/testing this library
# @author: Manfred Hästmark (hastmark2001@hotmail.com)
# @created: February 2025

option(
    SMATH_BUILD_TESTS 
    "If ON, build files for unit tests will be generated"
    OFF)

option(
    SMATH_BUILD_SAMPLES
    "If ON, build files for samples will be generated"
    OFF)

option(
    SMATH_BUILD_OMP
    "If ON, build with OMP backend"
    OFF)

cmake_minimum_required(VERSION 3.10)

project(sycl-math LANGUAGES CXX)

set(CMAKE_C_COMPILER acpp)
set(CMAKE_CXX_COMPILER acpp)

if(SMATH_BUILD_OMP)
    add_compile_options(--acpp-targets=omp -fopenmp=libgomp)
endif()

# SYCL math library
add_library(sycl-math INTERFACE)
set_target_properties(sycl-math PROPERTIES CMAKE_CXX_STANDARD 17)
target_include_directories(sycl-math INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(sycl-math INTERFACE -Wpedantic -Wall -Wextra -Werror -Wshadow -Wdeprecated -Wconversion)

add_subdirectory(samples)
add_subdirectory(test)
