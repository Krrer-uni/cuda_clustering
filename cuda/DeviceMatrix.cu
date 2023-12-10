//
// Created by Wojciech Rymer on 04.11.23.
//
#include "include/DeviceMatrix.cuh"
#include "include/CudaUtils.cuh"
#include <cstdint>
template<class MatrixType>
void DeviceMatrix<MatrixType>::free() {
  if (matrix.data != nullptr) {
    cudaFree(matrix.data);
    cudaDeviceSynchronize();
    cudaCheckError()
    matrix.data = nullptr;
  }
}

template<class MatrixType>
void DeviceMatrix<MatrixType>::allocateMatrix(size_t size) {
  free();
  this->size = size;
  this->step = std::sqrt(size);
  cudaMallocManaged(&matrix.data, size * sizeof(MatrixType));
  cudaCheckError()
}
template<class MatrixType>
void DeviceMatrix<MatrixType>::allocateMatrixZero(size_t size) {
  allocateMatrix(size);
  cudaMemset(matrix.data, 0u, size * sizeof(MatrixType));
  cudaDeviceSynchronize();
  cudaCheckError()
}

template<class MatrixType>
DeviceSubmatrixView<MatrixType>::DeviceSubmatrixView(DeviceMatrix<MatrixType> parent_matrix,
                                                     MatrixPoint submatrix_origin,
                                                     unsigned int submatrix_step,
                                                     unsigned int submatrix_height) : parent_matrix_(parent_matrix),
                                                                                      origin_(submatrix_origin),
                                                                                      step_(submatrix_step),
                                                                                      height_(submatrix_height) {

}

template<class MatrixType>
MatrixType DeviceSubmatrixView<MatrixType>::getField(MatrixPoint coordinates) {
  unsigned x_view = coordinates.x + origin_.x;
  unsigned y_view = coordinates.y * parent_matrix_.step + origin_.y;
  if (x_view >= parent_matrix_.step || y_view >= parent_matrix_.size) {
    printf("Out of bound submatrix access");
    return 255;
  }
  return parent_matrix_.matrix.data[x_view + y_view];
}

template
class DeviceMatrix<uint8_t>;

template
class DeviceSubmatrixView<uint8_t>;