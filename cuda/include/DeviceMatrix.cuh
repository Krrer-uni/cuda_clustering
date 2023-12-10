//
// Created by Wojciech Rymer on 04.11.23.
//

#ifndef EUCLIDEAN_CLUSTERING_CUDA_INCLUDE_DEVICE_MATRIX_CUH_
#define EUCLIDEAN_CLUSTERING_CUDA_INCLUDE_DEVICE_MATRIX_CUH_
#include "DeviceArray.cuh"
#include "cuda.h"
/**
 * 2d coordinates in matrix
 */
struct MatrixPoint {
  unsigned x;
  unsigned y;
};
/**
 * Class for square matrix for cuda
 */
template<class MatrixType>
class DeviceMatrix {
 public:
  DeviceArray<MatrixType> matrix{};
  unsigned step;  // matrix width
  unsigned size;  // width * height
  void allocateMatrix(size_t size);  // allocate square matrix size * size
  void allocateMatrixZero(size_t size);  // allocate matrix and write it with zeroes
  void free();  // free matrix and assign nullptr
};

/**
 * Submatrix that's a view of a parent matrix
 * @tparam MatrixType
 */
template<class MatrixType>
class DeviceSubmatrixView {
 private:
 public:
  DeviceMatrix<MatrixType> parent_matrix_;
  MatrixPoint origin_;
  unsigned step_;
  unsigned height_;
  DeviceSubmatrixView(DeviceMatrix<MatrixType> parent_matrix,
                      MatrixPoint submatrix_origin,
                      unsigned submatrix_step,
                      unsigned submatrix_height_);
  MatrixType getField(MatrixPoint coordinates);
};

#endif  // EUCLIDEAN_CLUSTERING_CUDA_INCLUDE_DEVICE_MATRIX_CUH_