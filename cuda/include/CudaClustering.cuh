#pragma once

#include <pcl/point_cloud.h>
#include "include/DeviceArray.cuh"
#include "Matrix.cuh"

struct CudaPoint {
  float x;
  float y;
  float z;
};

struct ClusterCloud {
  size_t size;
  CudaPoint *points;
  unsigned *labels;
};


struct ClusterParams {
  double distance;
};

template<class PointType>
class CudaClustering {
 private:
  ClusterParams params_{};
  std::vector<int> clusters_;
  std::shared_ptr<ClusterCloud> cluster_cloud_;
  DeviceArray<unsigned> d_labels_list{}; // list of remaining labels
  Matrix<u_int8_t> d_matrix{};  //adjacency matrix
  size_t submatrix_size_ = 4;  // dimension of submatrix

  void initial_clustering();

  void build_matrix(ClusterCloud &cluster_cloud,
                    DeviceArray<unsigned int> &labels_list,
                    Matrix<uint8_t> &matrix,
                    const float d_th);

  void exclusive_scan(DeviceArray<unsigned> &array);

  void update();

  bool evaluate_layer(std::vector<SubmatrixView<uint8_t>> layer);

  bool evaluate_diagonal_submatrix(MatrixPoint submat_origin);

  bool evaluate_submatrix(MatrixPoint submatrix);

  /**
   * Get layers of the matrix
   * @return pairs of a top left points in submatrices,
   * (-1,-1) as last pair in case it's the last one
   */
  std::vector<SubmatrixView<uint8_t>> get_layer(size_t layer_number, bool &_is_last);

 public:
  CudaClustering();

  ~CudaClustering();

  void setParams(ClusterParams params);

  void setInputCloud(typename pcl::PointCloud<PointType>::Ptr input_cloud);

  void extract(std::vector<unsigned> &indices_clusters);
};
