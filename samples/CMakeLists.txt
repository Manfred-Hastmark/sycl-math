# Main cmake file for using/building/testing this library
# @author: Manfred Hästmark (hastmark2001@hotmail.com)
# @created: February 2025

if(SMATH_BUILD_SAMPLES)
    add_executable(matrix matrix.cpp)
    target_link_libraries(matrix PRIVATE sycl-math)
endif()
