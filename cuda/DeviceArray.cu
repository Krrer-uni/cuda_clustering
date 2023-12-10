//
// Created by Wojciech Rymer on 04.11.23.
//
#include "include/DeviceArray.cuh"
#include "include/CudaUtils.cuh"

template<typename T>
void DeviceArray<T>::free() {
  if (data != nullptr) {
    cudaFree(data);
    cudaDeviceSynchronize();
    cudaCheckError()
    data = nullptr;
  }
}

template
class DeviceArray<unsigned>;