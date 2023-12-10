//
// Created by Wojciech Rymer on 04.11.23.
//

#ifndef EUCLIDEAN_CLUSTERING_CUDA_INCLUDE_MATRIX_CLUSTERING_CUH_
#define EUCLIDEAN_CLUSTERING_CUDA_INCLUDE_MATRIX_CLUSTERING_CUH_
#include <pcl/point_cloud.h>
#include "include/DeviceArray.cuh"
#include "include/CudaPointCloud.cuh"
#include "DeviceMatrix.cuh"
/**
 * Parameters for matrix clustering
 */
struct MatrixClusteringConfig {
  double distance;
};
/**
 * Matrix Clustering main class
 * @tparam PointType type of points in input cloud
 */
template<class PointType>
class MatrixClustering {
 private:
  MatrixClusteringConfig config_{};
  std::shared_ptr<CudaPointCloud> cluster_cloud_;
  DeviceArray<unsigned> d_labels_map_{};   // help array L
  DeviceArray<unsigned> d_labels_list_{}; // list of remaining labels R
  DeviceMatrix<u_int8_t> d_matrix_{};  //adjacency matrix

  /**
   * create a matrix for the first time
   */
  void build_matrix();
  /**
   * rebuild matrix and labels list
   */
  void update();

  /**
   * evaluate each submatrix of layer
   * @param layer set of submatricies
   * @return true if a merge occured
   */
  bool evaluate_layer(std::vector<DeviceSubmatrixView<uint8_t>> &layer);

  /**
   * Get layers of the matrix
   * @return pairs of a top left points in submatrices,
   **/
  std::vector<DeviceSubmatrixView<uint8_t>> get_layer(size_t layer_number, bool &_is_last);

 public:
  MatrixClustering();

  ~MatrixClustering();

  void setConfig(MatrixClusteringConfig config);

  void setInputCloud(typename pcl::PointCloud<PointType>::Ptr input_cloud);

  void extract(std::vector<std::vector<int>> &indices_clusters);
};

#endif  // EUCLIDEAN_CLUSTERING_CUDA_INCLUDE_MATRIX_CLUSTERING_CUH_