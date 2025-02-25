# Main cmake file for using/building/testing this library
# @author: Manfred Hästmark (hastmark2001@hotmail.com)
# @created: February 2025

if(SMATH_BUILD_TESTS)
    add_executable(test-matrix test_matrix.cpp)
    add_test(NAME TestMatrix COMMAND test-matrix)
    find_package(GTest)
    target_link_libraries(test-matrix PRIVATE GTest::GTest GTest::Main)
    target_link_libraries(test-matrix PRIVATE sycl-math)
endif()
