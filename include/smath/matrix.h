/**
 * @author: Manfred Hästmark (hastmark2001@hotmail.com)
 * @created: February 2025
 */

#include "sycl/sycl.hpp"
#include <cstddef>
#include <cstdlib>
#include <cstring>

/**
 * Class representing a matrix of static size MxN, with a value type of T
 *
 * @note M represents number of rows and N represents number of columns
 * @note Value must be implement operands (*,+,-)
 *
 * Implemented using SYCL, the aim for the class to be portable between
 * different hardware
 */
namespace smath {
template <typename T, size_t M, size_t N> class Matrix {
public:
  /**
   * Constructor to construct a matrix if raw data already exists
   * @param queue: Pointer to sycl queue for submitting tasks
   * @param matrix: Pointer to existing matrix allocated using queue
   * @note Not recomended
   */
  Matrix(sycl::queue *queue, T *matrix) : m_queue(queue), m_matrix(matrix) {}

  /**
   * Allocates and constructs a matrix
   * @param queue: Pointer to sycl queue to use for submitting tasks
   * @note All matrices that this one interacts with, must use the same queue
   */
  explicit Matrix(sycl::queue *queue)
      : m_queue(queue),
        m_matrix(sycl::malloc_host<T>(M * N, m_queue->get_context())) {
    m_queue
        ->submit([&](sycl::handler &cgh) {
          cgh.memset(m_matrix, 0, M * N * sizeof(T));
        })
        .wait();
  };

  /**
   * Destructor, deallocates matrix memory
   */
  ~Matrix() {
    if (m_matrix != nullptr) {
      sycl::free(m_matrix, m_queue->get_context());
    }
  }

  /**
   * Copy constructor
   *  - reuses queue, allocates new matrix memory, memcpy data
   */
  Matrix(Matrix &other)
      : m_queue(other.m_queue),
        m_matrix(sycl::malloc_host<T>(M * N, m_queue->get_context())) {
    m_queue
        ->submit([&](sycl::handler &cgh) {
          cgh.memcpy(m_matrix, other.m_matrix, M * N * sizeof(T));
        })
        .wait();
  }

  /**
   * Move constructor, moves pointers to queue and matrix, very cheap
   */
  Matrix(Matrix &&other) noexcept
      : m_queue(other.m_queue), m_matrix(other.m_matrix) {
    other.m_matrix = nullptr;
  };

  /**
   * Set val at place (m_idx, n_idx)
   */
  void set(size_t m_idx, size_t n_idx, T val) {
    m_matrix[(m_idx * N) + n_idx] = val;
  }

  /**
   * Get val at place (m_idx, n_idx)
   */
  T get(size_t m_idx, size_t n_idx) const {
    return m_matrix[(m_idx * N) + n_idx];
  }

  /**
   * Get pointer to raw data
   */
  const T *cdata() const { return m_matrix; }

  /**
   * Get pointer to raw data
   */
  T *data() { return m_matrix; }

  /**
   * Matrix multiplication, does not allocate new memory
   */
  template <size_t Q>
  void mul(const Matrix<T, N, Q> &in_mat, Matrix<T, M, Q> &out_mat) {
    const auto *mat_a = m_matrix;
    const auto *mat_b = in_mat.cdata();
    auto *mat_c = out_mat.data();
    constexpr size_t BLOCK_SIZE{32};
    sycl::nd_range<2> range{sycl::range<2>(M, Q),
                            sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE)};
    m_queue
        ->submit([&](sycl::handler &cgh) {
          cgh.parallel_for(range, [=](sycl::nd_item<2> iter) {
            const auto m_idx = iter.get_global_id().get(0);
            const auto q_idx = iter.get_global_id().get(1);
            if (m_idx < M && q_idx < Q) {
              T sum = 0;
              for (size_t n_idx = 0; n_idx < N; n_idx++) {
                sum += mat_a[(m_idx * N) + n_idx] * mat_b[(n_idx * Q) + q_idx];
              }
              mat_c[(m_idx * Q) + q_idx] = sum;
            }
          });
        })
        .wait();
  }

  /**
   * Matrix muliplication, tiled implementation, allocates new return matrix
   */
  template <size_t Q> Matrix<T, M, Q> operator*(const Matrix<T, N, Q> &mat) {
    Matrix<T, M, Q> out_mat{m_queue};
    mul(mat, out_mat);
    return std::move(out_mat);
  }

  /**
   * Scalar multiplication, allocates new return matrix
   */
  Matrix<T, M, N> operator*(T scalar) {
    const auto *mat_a = m_matrix;
    T *mat_c = sycl::malloc_host<T>(M * N, m_queue->get_context());
    m_queue
        ->submit([&](sycl::handler &cgh) {
          cgh.parallel_for(sycl::range<2>{M, N}, [=](sycl::id<2> idx) {
            mat_c[(idx[0] * N) + idx[1]] =
                mat_a[(idx[0] * N) + idx[1]] * scalar;
          });
        })
        .wait();
    return {m_queue, mat_c};
  }

  /**
   * Matrix addition, allocates return matrix
   */
  Matrix<T, M, N> operator+(const Matrix<T, M, N> &mat) {
    const auto *mat_a = m_matrix;
    const auto *mat_b = mat.cdata();
    T *mat_c = sycl::malloc_host<T>(M * N, m_queue->get_context());
    m_queue
        ->submit([&](sycl::handler &cgh) {
          cgh.parallel_for(sycl::range<2>{M, N}, [=](sycl::id<2> idx) {
            mat_c[(idx[0] * N) + idx[1]] =
                mat_a[(idx[0] * N) + idx[1]] + mat_b[(idx[0] * N) + idx[1]];
          });
        })
        .wait();
    return {m_queue, mat_c};
  }

  /**
   * Matrix increment, does not allocate new memory
   */
  void operator+=(const Matrix<T, M, N> &mat) {
    auto *mat_a = m_matrix;
    const auto *mat_b = mat.cdata();
    m_queue
        ->submit([&](sycl::handler &cgh) {
          cgh.parallel_for(sycl::range<2>{M, N}, [=](sycl::id<2> idx) {
            mat_a[(idx[0] * N) + idx[1]] += mat_b[(idx[0] * N) + idx[1]];
          });
        })
        .wait();
  }

  /**
   * Matrix subtraction, allocates new return matrix
   */
  Matrix<T, M, N> operator-(const Matrix<T, M, N> &mat) {
    const auto *mat_a = m_matrix;
    const auto *mat_b = mat.cdata();
    T *mat_c = sycl::malloc_host<T>(M * N, m_queue->get_context());
    m_queue
        ->submit([&](sycl::handler &cgh) {
          cgh.parallel_for(sycl::range<2>{M, N}, [=](sycl::id<2> idx) {
            mat_c[(idx[0] * N) + idx[1]] =
                mat_a[(idx[0] * N) + idx[1]] - mat_b[(idx[0] * N) + idx[1]];
          });
        })
        .wait();
    return {m_queue, mat_c};
  }

  /**
   * Matrix decrement, does not allocate new memory
   */
  void operator-=(const Matrix<T, M, N> &mat) {
    auto *mat_a = m_matrix;
    const auto *mat_b = mat.cdata();
    m_queue
        ->submit([&](sycl::handler &cgh) {
          cgh.parallel_for(sycl::range<2>{M, N}, [=](sycl::id<2> idx) {
            mat_a[(idx[0] * N) + idx[1]] -= mat_b[(idx[0] * N) + idx[1]];
          });
        })
        .wait();
  }

private:
  sycl::queue *m_queue;
  T *m_matrix;
};

/**
 * Make scalar multiplication commutative
 */
template <typename T, size_t M, size_t N>
Matrix<T, M, N> operator*(T val, const Matrix<T, M, N> &mat) {
  return mat * val;
}
} // namespace smath
