
#include "smath/matrix.h"
#include "sycl/sycl.hpp"
#include <cstddef>

constexpr size_t M_DIM{64};
constexpr size_t N_DIM{64};

int main() {
  auto queue = sycl::queue{};
  smath::Matrix<float, M_DIM, N_DIM> a_mat{&queue};
  smath::Matrix<float, M_DIM, N_DIM> b_mat{&queue};
  auto c_mat = a_mat * b_mat;
  return 0;
}
