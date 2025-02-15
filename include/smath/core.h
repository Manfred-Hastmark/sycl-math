
#include <array>
#include <cstddef>

namespace smath {
template <typename T, size_t M, size_t N> class Matrix {
public:
  Matrix() = default;
  std::array<T, N> &at(size_t index) { return m_matrix.at(index); }
  const std::array<T, N> &cat(size_t index) { return m_matrix.at(index); }
  template <size_t Q> Matrix<T, M, Q> operator*(const Matrix<T, N, Q> &mat);

private:
  std::array<std::array<T, N>, M> m_matrix{};
};
} // namespace smath
