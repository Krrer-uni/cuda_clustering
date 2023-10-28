#pragma once
#include "DeviceArray.cuh"

struct MatrixPoint {
  unsigned x;
  unsigned y;
};

template<class MatrixType>
class Matrix {
 public:
  DeviceArray<MatrixType> matrix{};
  unsigned matrix_step;
  unsigned size;
  void allocateMatrix(size_t size);
  void allocateMatrixZero(size_t size);
  bool is_assigned();
  void free();

};

template<class MatrixType>
class SubmatrixView {
 private:
 public:
  Matrix<MatrixType> parent_matrix_;
  MatrixPoint submatrix_origin_;
  unsigned submatrix_step_;
  unsigned submatrix_height_;
  SubmatrixView(Matrix<MatrixType> parent_matrix,
                MatrixPoint submatrix_origin,
                unsigned submatrix_step,
                unsigned submatrix_height_);
  MatrixType getField(MatrixPoint coordinates);
};
