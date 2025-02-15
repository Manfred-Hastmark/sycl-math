
#include "smath/core.h"
#include <iostream>

int main() {
  smath::Matrix<float, 3, 2> a_mat;
  smath::Matrix<float, 2, 4> b_mat;

  for (auto &row : a_mat.raw()) {
    for (auto &elem : row) {
      elem = 1.0f;
    }
  }

  for (auto &row : b_mat.raw()) {
    for (auto &elem : row) {
      elem = 2.0f;
    }
  }

  auto c_mat = a_mat * b_mat;

  for (const auto &row : c_mat.craw()) {
    for (const auto &elem : row) {
      std::cout << elem << "\n";
    }
  }
  return 0;
}
