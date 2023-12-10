#pragma once
#include "DeviceArray.cuh"
#include "cuda.h"

struct MatrixPoint {
  unsigned x;
  unsigned y;
};

template<class MatrixType>
class DeviceMatrix {
 public:
  DeviceArray<MatrixType> matrix{};
  unsigned step;
  unsigned size;
  void allocateMatrix(size_t size);
  void allocateMatrixZero(size_t size);
  void free();
};

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
