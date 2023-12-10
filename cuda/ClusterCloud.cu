//
// Created by Wojciech Rymer on 04.11.23.
//
#include "include/CudaPointCloud.cuh"
#include "include/CudaUtils.cuh"

void CudaPointCloud::free() {
  if (points != nullptr) {
    cudaFree(points);
    cudaDeviceSynchronize();
    cudaCheckError()
    points = nullptr;
  }
  if (labels != nullptr) {
    cudaFree(labels);
    cudaDeviceSynchronize();
    cudaCheckError()
    labels = nullptr;
  }
}