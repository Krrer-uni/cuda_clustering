#pragma once

#include <vector>
#include <pcl/point_cloud.h>

#define BLOCK_SIZE 16

struct CudaPoint {
  float x;
  float y;
  float z;
};

struct ClusterCloud {
  size_t size;
  CudaPoint *points;
  unsigned* labels;
};

template<typename T>
struct DeviceArray{
  size_t size;
  T* data;
};

struct ClusterParams {
  double distance;
};

template<class PointType>
class CudaClustering {
 private:
  ClusterParams params{};
  std::vector<int> clusters;
  std::shared_ptr<ClusterCloud> cluster_cloud;

  void initial_clustering();
  void build_matrix(ClusterCloud& cluster_cloud, const float d_th, DeviceArray<unsigned>& labels_list);
  size_t b{};  // block size

 public:
  CudaClustering();
  ~CudaClustering();
  void setParams(ClusterParams params);
  void setInputCloud(typename pcl::PointCloud<PointType>::Ptr input_cloud);
  void extract(std::vector<unsigned> &indices_clusters);
};
