#include "include/CudaClustering.cuh"
#include "kernels/matrix_clustering.cu"
#include <pcl/point_types.h>
#include "include/CudaUtils.cuh"
#include <cmath>

template<class PointType>
CudaClustering<PointType>::CudaClustering() {
  cluster_cloud_ = std::make_shared<ClusterCloud>();
}

template<class PointType>
CudaClustering<PointType>::~CudaClustering() {
  if (cluster_cloud_ != nullptr && cluster_cloud_->points != nullptr) {
    delete[] cluster_cloud_->points;
  }
}

template<class PointType>
void CudaClustering<PointType>::setParams(ClusterParams params) {
  this->params_ = params;
}

template<class PointType>
void CudaClustering<PointType>::setInputCloud(typename pcl::PointCloud<PointType>::Ptr input_cloud) {
  if (input_cloud == nullptr) {
    return;
  }

  if (cluster_cloud_ != nullptr && cluster_cloud_->points != nullptr) {
    delete[] cluster_cloud_->points;
  }
  cluster_cloud_->size = input_cloud->size();
  cluster_cloud_->points = new CudaPoint[input_cloud->size()];

  for (size_t i = 0; i < input_cloud->size(); i++) {
    cluster_cloud_->points[i] = {input_cloud->points.data()[i].x,
                                 input_cloud->points.data()[i].y,
                                 input_cloud->points.data()[i].z};
  }
}

template<class PointType>
void CudaClustering<PointType>::extract(std::vector<unsigned int> &indices_clusters) {
  size_t points_data_bytes = cluster_cloud_->size * (sizeof(CudaPoint));
  size_t labels_data_bytes = cluster_cloud_->size * sizeof(unsigned);

  ClusterCloud d_cluster_cloud{};  // device cluster cloud
  d_cluster_cloud.size = cluster_cloud_->size;
  cudaMalloc(&d_cluster_cloud.points, points_data_bytes);
  cudaCheckError()
  cudaMemcpy(d_cluster_cloud.points,
             cluster_cloud_->points,
             points_data_bytes,
             cudaMemcpyHostToDevice);
  cudaCheckError()

  cudaMalloc(&d_cluster_cloud.labels, labels_data_bytes);
  cudaCheckError()

  // WARNING DATA ALLOCATED IN build_matrix()
  d_labels_list.data = nullptr;

  int grid_size = std::ceil(((float) d_cluster_cloud.size) / BLOCK_SIZE);
  initial_ec<<<grid_size, BLOCK_SIZE>>>(d_cluster_cloud, params_.distance);

  build_matrix(d_cluster_cloud, d_labels_list, d_matrix, params_.distance);

  bool main_loop;
  while (main_loop) {
    bool md = false;

    bool is_final_layer = false;
    unsigned layer_count = 0;

    while (!md) {
      auto layer = get_layer(layer_count, is_final_layer);
      for (const auto &submatrix : layer) {
        bool layer_md = (layer_count == 0) ? evaluate_submatrix(submatrix)
                                           : evaluate_diagonal_submatrix(submatrix);
        md = md || layer_md;
      }
      if (md) {
        update();
        continue;
      }
      if (is_final_layer) {
        main_loop = false;
        continue;
      }
    }
  }

  cudaMemcpy(indices_clusters.data(),
             d_cluster_cloud.labels,
             labels_data_bytes,
             cudaMemcpyDeviceToHost);

  cudaFree(&d_cluster_cloud.labels);
  cudaFree(&d_cluster_cloud.points);
  if (d_labels_list.data != nullptr) {
    cudaFree(&d_labels_list.data); // ALLOCATED IN build_matrix()
  }
  if (d_matrix.matrix.data != nullptr) {
    cudaFree(&d_matrix.matrix); // ALLOCATED IN build_matrix()
  }
}

template<class PointType>
void CudaClustering<PointType>::build_matrix(ClusterCloud &cluster_cloud,
                                             DeviceArray<unsigned int> &labels_list,
                                             Matrix<uint8_t> &matrix,
                                             const float d_th) {
  if (labels_list.data != nullptr)
    cudaFree(&labels_list.data);
  matrix.free();

  // help array L
  DeviceArray<unsigned> d_labels_pos{};
  d_labels_pos.size = cluster_cloud.size + 1;  // n + 1 elements
  size_t labels_pos_data_size = d_labels_pos.size * sizeof(unsigned);
  cudaMallocManaged(&d_labels_pos.data, labels_pos_data_size);
  cudaMemset(d_labels_pos.data, 0u, labels_pos_data_size);
  cudaCheckError()

  int grid_size = std::ceil(((float) cluster_cloud.size) / BLOCK_SIZE);
  BuildMatrix::set_label_list<<<grid_size, BLOCK_SIZE>>>(cluster_cloud, d_labels_pos);
  cudaDeviceSynchronize();
  cudaCheckError()

  exclusive_scan(d_labels_pos);

  cudaCheckError()

  // R array allocation
  unsigned unique_clusters = d_labels_pos.data[d_labels_pos.size - 1];
  std::cout << "Found " << unique_clusters << " unique labels\n";
  labels_list.size = unique_clusters;
  cudaMalloc(&labels_list.data, labels_list.size * sizeof(unsigned));
  BuildMatrix::set_array_to_tid<<<grid_size, BLOCK_SIZE>>>(labels_list);
  cudaDeviceSynchronize();
  cudaCheckError()

  BuildMatrix::cluster_update<<<grid_size, BLOCK_SIZE>>>(cluster_cloud, d_labels_pos);
  cudaDeviceSynchronize();
  cudaCheckError()

  matrix.allocateMatrixZero(unique_clusters * unique_clusters);
  size_t matrix_grid_size = std::ceil(((float) matrix.size) / BLOCK_SIZE);
  BuildMatrix::populate_matrix<<<matrix_grid_size, BLOCK_SIZE>>>(cluster_cloud,
      matrix,
      d_th);

  cudaFree(&d_labels_pos);

}

template<class PointType>
void CudaClustering<PointType>::initial_clustering() {

}

template<class PointType>
void CudaClustering<PointType>::exclusive_scan(DeviceArray<unsigned> &array) {
  unsigned last_elem = array.data[0];

  array.data[0] = 0;
  for (size_t i = 1; i < array.size; i++) {
    unsigned tmp = array.data[i];
    array.data[i] = last_elem + array.data[i - 1];
    last_elem = tmp;
  }
}

template<class PointType>
bool CudaClustering<PointType>::evaluate_layer(std::vector<SubmatrixView<uint8_t>> layer) {
  return false;
}

template<class PointType>
std::vector<SubmatrixView<uint8_t>> CudaClustering<PointType>::get_layer(size_t layer_number, bool &_is_last) {
  std::vector<SubmatrixView<uint8_t>> layer{};
  for (unsigned x = layer_number * BLOCK_SIZE; x < d_labels_list.size; x += BLOCK_SIZE) {
    unsigned y = x - layer_number * BLOCK_SIZE;
    unsigned blockWidth = std::min((size_t) BLOCK_SIZE, d_labels_list.size - x);
    unsigned blockHeight = std::min((size_t) BLOCK_SIZE, d_labels_list.size - y);
    layer.emplace_back(&d_matrix, MatrixPoint{x, y}, blockHeight, blockWidth);
  }
  return layer;
}

template<class PointType>
bool CudaClustering<PointType>::evaluate_diagonal_submatrix(MatrixPoint submat_origin) {
  bool merge_found = false;

  SubmatrixView<uint8_t> submat(&d_matrix, submat_origin,);
}

/*
 * declarations of used templates
 */
template
class CudaClustering<pcl::PointXYZ>;

template
class CudaClustering<pcl::PointXYZI>;
