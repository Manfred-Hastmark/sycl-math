
cmake_minimum_required(VERSION 3.10)

project(sycl-math LANGUAGES CXX)

set(CMAKE_C_COMPILER acpp)
set(CMAKE_CXX_COMPILER acpp)

# SYCL math library
add_library(sycl-math INTERFACE)
set_target_properties(sycl-math PROPERTIES CMAKE_CXX_STANDARD 17)
target_include_directories(sycl-math INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(samples samples/main.cpp)
target_link_libraries(samples PRIVATE sycl-math)
target_compile_options(samples PRIVATE --acpp-targets=omp)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_subdirectory(test)
endif()
