//
// Created by Wojciech Rymer on 04.11.23.
//

#ifndef EUCLIDEAN_CLUSTERING_CUDA_INCLUDE_CUDAPOINTCLOUD_CUH_
#define EUCLIDEAN_CLUSTERING_CUDA_INCLUDE_CUDAPOINTCLOUD_CUH_
#include <pcl/point_types.h>
struct CudaPoint {
  float x;
  float y;
  float z;
  float i;
  CudaPoint() = default;
  CudaPoint(pcl::PointXYZ p) : x(p.x), y(p.y), z(p.z) {};
  CudaPoint(pcl::PointXYZI p) : x(p.x), y(p.y), z(p.z), i(p.intensity) {};
};

class CudaPointCloud {
 public:
  size_t size{};
  CudaPoint *points{};
  unsigned *labels{};
  void free();
};

#endif //EUCLIDEAN_CLUSTERING_CUDA_INCLUDE_CUDAPOINTCLOUD_CUH_
