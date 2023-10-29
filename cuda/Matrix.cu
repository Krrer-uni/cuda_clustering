#include "include/Matrix.cuh"
#include "include/CudaUtils.cuh"
#include <cstdint>
template<class MatrixType>
void Matrix<MatrixType>::free() {
  if(matrix.data != nullptr){
    cudaFree(matrix.data);
    cudaCheckError()
  }
}
template<class MatrixType>
bool Matrix<MatrixType>::is_assigned() {
  return matrix.data != nullptr;
}
template<class MatrixType>
void Matrix<MatrixType>::allocateMatrix(size_t size) {
  this->free();
  this->size = size;
  this->matrix_step = std::sqrt(size);
  cudaMallocManaged(&matrix.data,size * sizeof(MatrixType));
  cudaCheckError()
}
template<class MatrixType>
void Matrix<MatrixType>::allocateMatrixZero(size_t size) {
  allocateMatrix(size);
  cudaMemset(matrix.data, 0u, size * sizeof(MatrixType));
  cudaDeviceSynchronize();
  cudaCheckError()
}


template<class MatrixType>
SubmatrixView<MatrixType>::SubmatrixView(Matrix<MatrixType> parent_matrix,
                                         MatrixPoint submatrix_origin,
                                         unsigned int submatrix_step,
                                         unsigned int submatrix_height) : parent_matrix_(parent_matrix),
                                                                          origin_(submatrix_origin),
                                                                          step_(submatrix_step),
                                                                          height_(submatrix_height) {

}

template<class MatrixType>
MatrixType SubmatrixView<MatrixType>::getField(MatrixPoint coordinates) {
  unsigned x_view = coordinates.x + origin_.x;
  unsigned y_view = coordinates.y * parent_matrix_.matrix_step + origin_.y;
  if(x_view >= parent_matrix_.matrix_step || y_view >= parent_matrix_.size){
    printf("Out of bound submatrix access");
    return 255;
  }
  return parent_matrix_.matrix.data[x_view + y_view];
}


template class Matrix<uint8_t>;

template class SubmatrixView<uint8_t>;