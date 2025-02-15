#include "smath/core.h"
#include "sycl/sycl.hpp"
#include "gtest/gtest.h"
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <sstream>
#include <string>

constexpr size_t DIM_M{1000};
constexpr size_t DIM_N{1000};
constexpr size_t DIM_Q{1000};

constexpr auto FAULT_TOLERANCE{0.01f};
constexpr auto TEST_RANGE{10.0f};

constexpr uint32_t NO_TEST_CASES{100};
constexpr uint32_t REPORT_EVERY{10};

namespace {
/**
 * Regular implementation of gemm used for verification
 */
template <typename T, size_t M, size_t N, size_t Q>
void gemm(const smath::Matrix<T, M, N> &mat_a,
          const smath::Matrix<T, N, Q> &mat_b, smath::Matrix<T, M, Q> &mat_c) {
  constexpr size_t BLOCK_SIZE{256};
  for (size_t mb_idx = 0; mb_idx < (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
       mb_idx++) {
    for (size_t qb_idx = 0; qb_idx < (Q + BLOCK_SIZE - 1) / BLOCK_SIZE;
         qb_idx++) {
      for (size_t m_idx = mb_idx * BLOCK_SIZE;
           m_idx < std::min((mb_idx + 1) * BLOCK_SIZE, M); m_idx++) {
        for (size_t n_idx = 0; n_idx < N; n_idx++) {
          for (size_t q_idx = qb_idx * BLOCK_SIZE;
               q_idx < std::min((qb_idx + 1) * BLOCK_SIZE, Q); q_idx++) {
            if (n_idx == 0) {
              mat_c.set(m_idx, q_idx, 0);
            }
            auto val = mat_c.get(m_idx, q_idx);
            val += mat_a.get(m_idx, n_idx) * mat_b.get(n_idx, q_idx);
            mat_c.set(m_idx, q_idx, val);
          }
        }
      }
    }
  }
}

/**
 * Compares two matrices
 */
template <typename T, size_t M, size_t N>
bool compare(const smath::Matrix<T, M, N> &mat_a,
             const smath::Matrix<T, M, N> &mat_b) {
  for (size_t m_idx = 0; m_idx < M; m_idx++) {
    for (size_t n_idx = 0; n_idx < N; n_idx++) {
      if (std::abs(mat_a.get(m_idx, n_idx) - mat_b.get(m_idx, n_idx)) >
          FAULT_TOLERANCE) {
        return false;
      }
    }
  }
  return true;
}

template <typename T, size_t M, size_t N>
void init(smath::Matrix<T, M, N> &mat) {
  std::random_device rand;
  std::mt19937 gen(rand());
  std::uniform_real_distribution<float> dist(-TEST_RANGE, TEST_RANGE);
  for (size_t m_idx = 0; m_idx < M; m_idx++) {
    for (size_t n_idx = 0; n_idx < N; n_idx++) {
      mat.set(m_idx, n_idx, dist(gen));
    }
  }
}

template <typename T, size_t M, size_t N>
std::string mat_to_string(const smath::Matrix<T, M, N> &mat) {
  std::ostringstream oss;
  for (size_t m_idx = 0; m_idx < M; m_idx++) {
    oss << "[ ";
    for (size_t n_idx = 0; n_idx < N; n_idx++) {
      oss << mat.get(m_idx, n_idx) << " ";
    }
    oss << "]\n";
  }
  return oss.str();
}

} // namespace

TEST(TestGemm, functional) {
  auto queue = sycl::queue{};
  smath::Matrix<float, DIM_M, DIM_N> mat_a{&queue};
  smath::Matrix<float, DIM_N, DIM_Q> mat_b{&queue};
  smath::Matrix<float, DIM_M, DIM_Q> mat_c{&queue};

  GTEST_LOG_(INFO) << "Running " << NO_TEST_CASES << " cases...";
  double avg_time_us = 0.0;
  for (uint32_t i = 0; i < NO_TEST_CASES; i++) {
    init(mat_a);
    init(mat_b);

    gemm(mat_a, mat_b, mat_c);

    auto start = std::chrono::high_resolution_clock::now();
    auto calc_mat_c = mat_a * mat_b;
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> duration = end - start;

    if (i == 0) {
      avg_time_us = duration.count();
    } else {
      avg_time_us = (avg_time_us * i + duration.count()) / i;
    }

    ASSERT_TRUE(compare(mat_c, calc_mat_c))
        << "Matrix A:\n"
        << mat_to_string(mat_a) << "\nMatrix B:\n"
        << mat_to_string(mat_b) << "\nMatrix C:\n"
        << mat_to_string(mat_c) << "\nReceived:\n"
        << mat_to_string(calc_mat_c);

    if (i % REPORT_EVERY == 0) {
      GTEST_LOG_(INFO) << i
                       << " cases passed... "
                          "Avg time = "
                       << avg_time_us << "us";
    }
  }
}
