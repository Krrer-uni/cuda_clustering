//
// Created by Wojciech Rymer on 04.11.23.
//
#include "include/MatrixClustering.cuh"
#include "kernels/MatrixClusteringKernels.cu"
#include "include/CudaUtils.cuh"
#include <pcl/point_types.h>
#include <cmath>
#include <thrust/scan.h>
template<class PointType>
MatrixClustering<PointType>::MatrixClustering() {
  cluster_cloud_ = std::make_shared<CudaPointCloud>();
  cudaMalloc(&cluster_cloud_->points,12000);
  cluster_cloud_->free();
}

template<class PointType>
MatrixClustering<PointType>::~MatrixClustering() {
  cluster_cloud_->free();
  d_labels_map_.free();
  d_labels_list_.free();
}

template<class PointType>
void MatrixClustering<PointType>::setConfig(MatrixClusteringConfig config) {
  this->config_ = config;
}

template<class PointType>
void MatrixClustering<PointType>::setInputCloud(typename pcl::PointCloud<PointType>::Ptr input_cloud) {
  if (input_cloud == nullptr) {
    return;
  }
  cluster_cloud_->free();

  cluster_cloud_->size = input_cloud->size();
  size_t points_data_bytes = cluster_cloud_->size * sizeof(CudaPoint);
  size_t labels_data_bytes = cluster_cloud_->size * sizeof(unsigned);
  cudaMallocManaged(&cluster_cloud_->points, points_data_bytes);
  cudaDeviceSynchronize();
  cudaCheckError()
  cudaMallocManaged(&cluster_cloud_->labels, labels_data_bytes);
  cudaDeviceSynchronize();
  cudaCheckError()
  int j = 0;
  for(const auto &p : input_cloud->points){
    cluster_cloud_->points[j] = CudaPoint(p);
    j++;
  }
}
template<class PointType>
void MatrixClustering<PointType>::extract(std::vector<std::vector<int>> &indices_clusters) {

  // WARNING DATA ALLOCATED IN build_matrix()
  int grid_size = std::ceil(((float) cluster_cloud_->size) / BLOCK_SIZE);
  initial_ec<<<grid_size, BLOCK_SIZE>>>(*cluster_cloud_, config_.distance);
  cudaDeviceSynchronize();
  cudaCheckError()

  build_matrix();
  cudaDeviceSynchronize();
  cudaCheckError()

  bool main_loop = true;
  while (main_loop) {
    bool merge_found = false;

    bool is_final_layer = false;
    unsigned layer_count = 0;

    while (true) {
      auto layer = get_layer(layer_count, is_final_layer);
      layer_count++;
      if(layer.size() == 0)
        break;
      if (evaluate_layer(layer)) {
        update();
        break;
      }
      if (is_final_layer) {
        main_loop = false;
        break;
      }
    }
  }
  update();
//  std::cout << "number of labels: " << labels_list_.size << std::endl;
  indices_clusters.clear();
  indices_clusters.resize(d_labels_list_.size);
  for (size_t p = 0; p < cluster_cloud_->size; p++) {
    unsigned cl = cluster_cloud_->labels[p];
    indices_clusters[cl].push_back(p);
  }


  d_labels_list_.free();
  d_labels_map_.free();
  d_matrix_.free();
}

template<class PointType>
void MatrixClustering<PointType>::build_matrix() {
  d_labels_list_.free();
  d_labels_map_.free();

  d_labels_map_.size = cluster_cloud_->size + 1;  // n + 1 elements
  size_t labels_pos_data_size = d_labels_map_.size * sizeof(unsigned);
  cudaMallocManaged(&d_labels_map_.data, labels_pos_data_size);
  cudaDeviceSynchronize();
  cudaMemset(d_labels_map_.data, 0u, labels_pos_data_size);
  cudaDeviceSynchronize();
  cudaCheckError()

  int grid_size = std::ceil(((float) cluster_cloud_->size) / BLOCK_SIZE);
  BuildMatrix::set_label_list<<<grid_size, BLOCK_SIZE>>>(*cluster_cloud_, d_labels_map_);
  cudaDeviceSynchronize();
  cudaCheckError()

  thrust::exclusive_scan(d_labels_map_.data, d_labels_map_.data + d_labels_map_.size, d_labels_map_.data);

  cudaCheckError()
  // R array allocation
  unsigned unique_clusters = d_labels_map_.data[d_labels_map_.size - 1];
  d_labels_list_.size = unique_clusters;
  cudaMallocManaged(&d_labels_list_.data, d_labels_list_.size * sizeof(unsigned));
  BuildMatrix::set_array_to_tid<<<grid_size, BLOCK_SIZE>>>(d_labels_list_);
  cudaDeviceSynchronize();
  cudaCheckError()

  BuildMatrix::cluster_update<<<grid_size, BLOCK_SIZE>>>(*cluster_cloud_, d_labels_map_);
  cudaDeviceSynchronize();
  cudaCheckError()

  cudaMemset(d_labels_map_.data, 0u, labels_pos_data_size);
  cudaDeviceSynchronize();
  cudaCheckError()
  BuildMatrix::set_label_list<<<grid_size, BLOCK_SIZE>>>(*cluster_cloud_, d_labels_map_);
  cudaDeviceSynchronize();
  cudaCheckError()
  thrust::exclusive_scan(d_labels_map_.data, d_labels_map_.data + d_labels_map_.size, d_labels_map_.data);

  d_matrix_.allocateMatrixZero(unique_clusters * unique_clusters);
  size_t matrix_grid_size = std::ceil(((float) d_matrix_.size) / BLOCK_SIZE);
  BuildMatrix::populate_matrix<<<matrix_grid_size, BLOCK_SIZE>>>(*cluster_cloud_,
      d_matrix_,
      config_.distance);
  cudaDeviceSynchronize();
  cudaCheckError()
}

/**
 * Evaluate a layer of matrix
 * @tparam PointType Point type of input cloud
 * @param layer vector of submatricies belonging to one layer
 * @return true if a merged occurred, false otherwise
 */
template<class PointType>
bool MatrixClustering<PointType>::evaluate_layer(std::vector<DeviceSubmatrixView<uint8_t>>& layer) {
  bool* merge_found;  // variable in managed memory
  cudaMallocManaged(&merge_found,sizeof merge_found);
  *merge_found = false;
  // allocate and copy submatrices to device memeory
  DeviceSubmatrixView<uint8_t>* d_layer;
  size_t layer_size = layer.size() * sizeof(DeviceSubmatrixView<uint8_t>) ;
  cudaMallocManaged(&d_layer, layer_size);
  cudaMemcpy(d_layer, layer.data(), layer_size, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  cudaCheckError()

  MatrixMerge::launchLayerMerge<<<layer.size(),BLOCK_SIZE>>>(d_layer,merge_found,d_labels_list_);
  cudaDeviceSynchronize();
  cudaCheckError()

  cudaFree(d_layer);
  cudaDeviceSynchronize();
  cudaCheckError()
  return *merge_found;
}
/**
 *  Function to get submatrices of a layer
 * @tparam PointType Point type of input cloud
 * @param layer_number index of a layer
 * @param _is_last idicator if the returned layer is the last one
 * @return vector of submatricies belonging to one layer
 */
template<class PointType>
std::vector<DeviceSubmatrixView<uint8_t>> MatrixClustering<PointType>::get_layer(size_t layer_number, bool &_is_last) {
  std::vector<DeviceSubmatrixView<uint8_t>> layer{};
  for (unsigned x = layer_number * BLOCK_SIZE; x < d_labels_list_.size; x += BLOCK_SIZE) {
    unsigned y = x - layer_number * BLOCK_SIZE;
    unsigned blockWidth = std::min((size_t) BLOCK_SIZE, d_labels_list_.size - x);
    unsigned blockHeight = std::min((size_t) BLOCK_SIZE, d_labels_list_.size - y);
    layer.emplace_back(d_matrix_, MatrixPoint{x, y}, blockWidth, blockHeight);
  }
  if(layer.size() == 1)
    _is_last = true;
  return layer;
}


template<class PointType>
void MatrixClustering<PointType>::update() {
  int cloud_grid_size = std::ceil(((float) cluster_cloud_->size) / BLOCK_SIZE);
  Update::mapClusters<<<cloud_grid_size, BLOCK_SIZE>>>(*cluster_cloud_, d_labels_list_,d_labels_map_);
  cudaDeviceSynchronize();
  cudaCheckError()
  size_t labels_pos_data_size = d_labels_map_.size * sizeof(unsigned);
  cudaMemset(d_labels_map_.data, 0u, labels_pos_data_size);
  cudaDeviceSynchronize();
  cudaCheckError()

  int label_grid_size = std::ceil(((float) d_labels_map_.size) / BLOCK_SIZE);
  Update::set_label_list<<<label_grid_size,BLOCK_SIZE>>>(d_labels_list_, d_labels_map_);
  cudaDeviceSynchronize();
  cudaCheckError()
  thrust::exclusive_scan(d_labels_map_.data, d_labels_map_.data + d_labels_map_.size, d_labels_map_.data);

  unsigned unique_clusters = d_labels_map_.data[d_labels_map_.size - 1];
//  std::cout << "Found " << unique_clusters << " unique labels\n" ;
  DeviceArray<unsigned> labels_list_update{};
  labels_list_update.size = unique_clusters;
  cudaMallocManaged(&labels_list_update.data, labels_list_update.size * sizeof(unsigned));
  cudaDeviceSynchronize();
  cudaCheckError()

  BuildMatrix::set_array_to_tid<<<unique_clusters, BLOCK_SIZE>>>(labels_list_update);
  cudaDeviceSynchronize();
  cudaCheckError()

  DeviceMatrix<uint8_t> matrix_update{};
  matrix_update.allocateMatrixZero(unique_clusters * unique_clusters);

  Update::build_matrix<<<label_grid_size, BLOCK_SIZE>>>(d_matrix_,matrix_update,d_labels_list_,d_labels_map_);
  cudaDeviceSynchronize();
  cudaCheckError()

  d_labels_list_.free();
  d_labels_list_ = labels_list_update;

  d_matrix_.free();
  d_matrix_ = matrix_update;
}

/*
 * declarations of used templates
 */
template
class MatrixClustering<pcl::PointXYZ>;

template
class MatrixClustering<pcl::PointXYZI>;
