//
// Created by krrer on 04.11.23.
//

#ifndef EUCLIDEAN_CLUSTERING_CUDA_INCLUDE_CLUSTERCLOUD_CUH_
#define EUCLIDEAN_CLUSTERING_CUDA_INCLUDE_CLUSTERCLOUD_CUH_

struct CudaPoint {
  float x;
  float y;
  float z;
  float i;
};

class ClusterCloud {
 public:
  size_t size;
  CudaPoint *points;
  unsigned *labels;
  void free();
};

#endif //EUCLIDEAN_CLUSTERING_CUDA_INCLUDE_CLUSTERCLOUD_CUH_
