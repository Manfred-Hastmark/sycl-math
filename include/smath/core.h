
#include "sycl/sycl.hpp"
#include <cstddef>
#include <cstdlib>
#include <cstring>

namespace smath {
template <typename T, size_t M, size_t N> class Matrix {
public:
  Matrix(sycl::queue *queue, T *matrix) : m_queue(queue), m_matrix(matrix) {}
  explicit Matrix(sycl::queue *queue)
      : m_queue(queue),
        m_matrix(sycl::malloc_host<T>(M * N, m_queue->get_context())) {
    std::memset(m_matrix, 0, M * N * sizeof(T));
  };
  ~Matrix() {
    if (m_matrix != nullptr) {
      sycl::free(m_matrix, m_queue->get_context());
    }
  }
  Matrix(Matrix & /* other */) = delete;
  Matrix(Matrix &&other) noexcept
      : m_queue(other.m_queue), m_matrix(other.m_matrix) {
    other.m_matrix = nullptr;
  };

  void set(size_t m_idx, size_t n_idx, T val) {
    m_matrix[(m_idx * N) + n_idx] = val;
  }

  T get(size_t m_idx, size_t n_idx) const {
    return m_matrix[(m_idx * N) + n_idx];
  }

  const T *cdata() const { return m_matrix; }

  template <size_t Q> Matrix<T, M, Q> operator*(const Matrix<T, N, Q> &mat) {
    const auto *mat_a = m_matrix;
    const auto *mat_b = mat.cdata();
    T *mat_c = sycl::malloc_host<T>(M * Q, m_queue->get_context());
    constexpr size_t BLOCK_SIZE{64};
    m_queue
        ->submit([&](sycl::handler &cgh) {
          cgh.parallel_for(
              sycl::range<2>{(M + BLOCK_SIZE - 1) / BLOCK_SIZE,
                             (Q + BLOCK_SIZE - 1) / BLOCK_SIZE},
              [=](sycl::id<2> idx) {
                const auto mb_idx = idx[0];
                const auto qb_idx = idx[1];
                for (size_t m_idx = mb_idx * BLOCK_SIZE;
                     m_idx < std::min((mb_idx + 1) * BLOCK_SIZE, M); m_idx++) {
                  for (size_t n_idx = 0; n_idx < N; n_idx++) {
                    for (size_t q_idx = qb_idx * BLOCK_SIZE;
                         q_idx < std::min((qb_idx + 1) * BLOCK_SIZE, Q);
                         q_idx++) {
                      if (n_idx == 0) {
                        mat_c[(m_idx * Q) + q_idx] = 0;
                      }
                      mat_c[(m_idx * Q) + q_idx] += mat_a[(m_idx * N) + n_idx] *
                                                    mat_b[(n_idx * Q) + q_idx];
                    }
                  }
                }
              });
        })
        .wait();
    return {m_queue, mat_c};
  }

private:
  sycl::queue *m_queue;
  T *m_matrix;
};
} // namespace smath
