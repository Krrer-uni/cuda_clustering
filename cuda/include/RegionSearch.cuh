//
// Created by krrer on 04.11.23.
//

#ifndef EUCLIDEAN_CLUSTERING_CUDA_INCLUDE_REGIONSEARCH_CUH_
#define EUCLIDEAN_CLUSTERING_CUDA_INCLUDE_REGIONSEARCH_CUH_
#include "ClusterCloud.cuh"
#include "DeviceArray.cuh"

class RegionSearch{
 private:
  ClusterCloud& _cloud;
  DeviceArray<unsigned> _pid;
  DeviceArray<unsigned> _V;
  DeviceArray<unsigned> _S;
  CudaPoint min();
  CudaPoint max();
  size_t block_size_ = 128;
 public:
  RegionSearch(ClusterCloud& cloud);
  void setCloud(ClusterCloud& cloud);
  void build(float threshold);

  void find();
};

#endif //EUCLIDEAN_CLUSTERING_CUDA_INCLUDE_REGIONSEARCH_CUH_
