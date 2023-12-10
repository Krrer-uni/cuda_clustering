//
// Created by Wojciech Rymer on 04.11.23.
//

#ifndef EUCLIDEAN_CLUSTERING_CUDA_INCLUDE_DEVICE_ARRAY_CUH_
#define EUCLIDEAN_CLUSTERING_CUDA_INCLUDE_DEVICE_ARRAY_CUH_
#include <cstddef>
#include <vector>

template<typename T>
class DeviceArray {
 public:
  std::size_t size;
  T *data;
  void free();
};

#endif  // EUCLIDEAN_CLUSTERING_CUDA_INCLUDE_DEVICE_ARRAY_CUH_
