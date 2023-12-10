//
// Created by Wojciech Rymer on 04.11.23.
//

#ifndef EUCLIDEAN_CLUSTERING_CUDA_INCLUDE_MATRIX_CLUSTERING_CUH_
#define EUCLIDEAN_CLUSTERING_CUDA_INCLUDE_MATRIX_CLUSTERING_CUH_
#include <pcl/point_cloud.h>
#include "include/DeviceArray.cuh"
#include "include/CudaPointCloud.cuh"
#include "DeviceMatrix.cuh"

struct ClusteringConfig {
  double distance;
};

template<class PointType>
class MatrixClustering {
 private:
  ClusteringConfig config_{};
  std::vector<int> labels_;
  std::shared_ptr<CudaPointCloud> cluster_cloud_;
  DeviceArray<unsigned> labels_map_{};   // help array L
  DeviceArray<unsigned> labels_list_{}; // list of remaining labels R
  DeviceMatrix<u_int8_t> d_matrix_{};  //adjacency matrix

  void build_matrix(CudaPointCloud &cluster_cloud,
                    DeviceArray<unsigned int> &labels_list,
                    DeviceMatrix<uint8_t> &matrix,
                    const float d_th);

  void update();

  bool evaluate_layer(std::vector<DeviceSubmatrixView<uint8_t>> &layer);

  /**
   * Get layers of the matrix
   * @return pairs of a top left points in submatrices,
   **/
  std::vector<DeviceSubmatrixView<uint8_t>> get_layer(size_t layer_number, bool &_is_last);

 public:
  MatrixClustering();

  ~MatrixClustering();

  void setConfig(ClusteringConfig config);

  void setInputCloud(typename pcl::PointCloud<PointType>::Ptr input_cloud);

  void extract(std::vector<std::vector<int>> &indices_clusters);
};

#endif  // EUCLIDEAN_CLUSTERING_CUDA_INCLUDE_MATRIX_CLUSTERING_CUH_