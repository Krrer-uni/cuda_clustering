//
// Created by Wojciech Rymer on 04.11.23.
//

#ifndef EUCLIDEAN_CLUSTERING_CUDA_INCLUDE_CUDAPOINTCLOUD_CUH_
#define EUCLIDEAN_CLUSTERING_CUDA_INCLUDE_CUDAPOINTCLOUD_CUH_
#include <pcl/point_types.h>
/**
 * Point struct usable on cuda device
 */
struct CudaPoint {
  float x;
  float y;
  float z;
  float i;
  CudaPoint() = default;
  CudaPoint(pcl::PointXYZ p) : x(p.x), y(p.y), z(p.z) {};
  CudaPoint(pcl::PointXYZI p) : x(p.x), y(p.y), z(p.z), i(p.intensity) {};
};

/**
 * Point cloud struct usable on cuda device
 */
class CudaPointCloud {
 public:
  size_t size{};
  CudaPoint *points{};
  unsigned *labels{};
  /**
   * dealocate memory and assign nullptr
   */
  void free();
};

#endif //EUCLIDEAN_CLUSTERING_CUDA_INCLUDE_CUDAPOINTCLOUD_CUH_
