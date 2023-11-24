//
// Created by krrer on 04.11.23.
//
#include "include/ClusterCloud.cuh"
#include "include/CudaUtils.cuh"

void ClusterCloud::free() {
  if(points != nullptr){
    cudaFree(points);
    cudaDeviceSynchronize();
    cudaCheckError()
    points = nullptr;
  }
  if(labels != nullptr){
    cudaFree(labels);
    cudaDeviceSynchronize();
    cudaCheckError()
    labels = nullptr;
  }
}