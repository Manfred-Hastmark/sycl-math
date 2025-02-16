add_executable(test-gemm test_matrix.cpp)
add_test(NAME TestGemm COMMAND test-gemm)

find_package(GTest)
target_link_libraries(test-gemm PRIVATE GTest::GTest GTest::Main)
target_link_libraries(test-gemm PRIVATE sycl-math)
