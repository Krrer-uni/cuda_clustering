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
    cudaFree(cluster_cloud_->points);
  }
  if (cluster_cloud_ != nullptr && cluster_cloud_->labels != nullptr) {
    cudaFree(cluster_cloud_->labels);
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
    cudaFree(cluster_cloud_->points);
  }
  if (cluster_cloud_ != nullptr && cluster_cloud_->labels != nullptr) {
    cudaFree(cluster_cloud_->labels);
  }

  cluster_cloud_->size = input_cloud->size();
  size_t points_data_bytes = cluster_cloud_->size * sizeof(CudaPoint);
  size_t labels_data_bytes = cluster_cloud_->size * sizeof(unsigned);
  cudaMallocManaged(&cluster_cloud_->points, points_data_bytes);
  cudaDeviceSynchronize();
  cudaCheckError()
  cudaMallocManaged(&cluster_cloud_->labels, labels_data_bytes);
  cudaDeviceSynchronize();
  cudaCheckError()

  for (size_t i = 0; i < input_cloud->size(); i++) {
    cluster_cloud_->points[i] = {input_cloud->points.data()[i].x,
                                 input_cloud->points.data()[i].y,
                                 input_cloud->points.data()[i].z};
  }
}

template<class PointType>
void CudaClustering<PointType>::extract(std::vector<unsigned int> &indices_clusters) {

  // WARNING DATA ALLOCATED IN build_matrix()
  labels_list_.data = nullptr;

  int grid_size = std::ceil(((float) cluster_cloud_->size) / BLOCK_SIZE);
  initial_ec<<<grid_size, BLOCK_SIZE>>>(*cluster_cloud_, params_.distance);

  build_matrix(*cluster_cloud_, labels_list_, d_matrix_, params_.distance);
//
//  for(int i = 0; i < d_matrix_.matrix_step; i++){
//    for(int j = 0; j < d_matrix_.matrix_step; j++){
//      std::cout << (unsigned)d_matrix_.matrix.data[i * d_matrix_.matrix_step + j] << " ";
//    }
//    std::cout << std::endl;
//  }


  bool main_loop = true;
  while (main_loop) {
    bool merge_found = false;

    bool is_final_layer = false;
    unsigned layer_count = 0;

    while (!merge_found) {
      auto layer = get_layer(layer_count, is_final_layer);
      layer_count++;
      if(layer.size() == 0)
        return;
      merge_found = merge_found || evaluate_layer(layer);
      if (merge_found) {
        update();
        continue;
      }
      if (is_final_layer) {
        main_loop = false;
        continue;
      }
    }
  }

  size_t labels_data_bytes = cluster_cloud_->size * sizeof(unsigned);
  cudaMemcpy(indices_clusters.data(),
             cluster_cloud_->labels,
             labels_data_bytes,
             cudaMemcpyDeviceToHost);

  if (labels_list_.data != nullptr) {
    cudaFree(&labels_list_.data); // ALLOCATED IN build_matrix()
  }
  if (d_matrix_.matrix.data != nullptr) {
    cudaFree(&d_matrix_.matrix); // ALLOCATED IN build_matrix()
  }
}

template<class PointType>
void CudaClustering<PointType>::build_matrix(ClusterCloud &cluster_cloud,
                                             DeviceArray<unsigned int> &labels_list,
                                             Matrix<uint8_t> &matrix,
                                             const float d_th) {
  if (labels_list.data != nullptr)
    cudaFree(&labels_list.data);
  if (labels_map_.data != nullptr)
    cudaFree(&labels_map_.data);
  matrix.free();

  labels_map_.size = cluster_cloud.size + 1;  // n + 1 elements
  size_t labels_pos_data_size = labels_map_.size * sizeof(unsigned);
  cudaMallocManaged(&labels_map_.data, labels_pos_data_size);
  cudaMemset(labels_map_.data, 0u, labels_pos_data_size);
  cudaCheckError()

  int grid_size = std::ceil(((float) cluster_cloud.size) / BLOCK_SIZE);
  BuildMatrix::set_label_list<<<grid_size, BLOCK_SIZE>>>(cluster_cloud, labels_map_);
  cudaDeviceSynchronize();
  cudaCheckError()

  exclusive_scan(labels_map_);
  cudaCheckError()
  // R array allocation
  unsigned unique_clusters = labels_map_.data[labels_map_.size - 1];
  std::cout << "Found " << unique_clusters << " unique labels\n";
  labels_list.size = unique_clusters;
  cudaMallocManaged(&labels_list.data, labels_list.size * sizeof(unsigned));
  BuildMatrix::set_array_to_tid<<<grid_size, BLOCK_SIZE>>>(labels_list);
  cudaDeviceSynchronize();
  cudaCheckError()

  BuildMatrix::cluster_update<<<grid_size, BLOCK_SIZE>>>(cluster_cloud, labels_map_);
  cudaDeviceSynchronize();
  cudaCheckError()

  matrix.allocateMatrixZero(unique_clusters * unique_clusters);
  size_t matrix_grid_size = std::ceil(((float) matrix.size) / BLOCK_SIZE);
  BuildMatrix::populate_matrix<<<matrix_grid_size, BLOCK_SIZE>>>(cluster_cloud,
      matrix,
      d_th);
  cudaDeviceSynchronize();
  cudaCheckError()
}


template<class PointType>
void CudaClustering<PointType>::exclusive_scan(DeviceArray<unsigned> &array) {
  if(array.size == 0){
    return;
  }
  unsigned last_elem = array.data[0];

  array.data[0] = 0;
  for (size_t i = 1; i < array.size; i++) {
    unsigned tmp = array.data[i];
    array.data[i] = last_elem + array.data[i - 1];
    last_elem = tmp;
  }
}

template<class PointType>
bool CudaClustering<PointType>::evaluate_layer(std::vector<SubmatrixView<uint8_t>>& layer) {
  bool* merge_found;
  cudaMallocManaged(&merge_found,sizeof merge_found);
  *merge_found = false;
  SubmatrixView<uint8_t>* d_layer;
  size_t layer_size = layer.size() * sizeof(SubmatrixView<uint8_t>) ;
  cudaMallocManaged(&d_layer, layer_size);
  cudaMemcpy(d_layer, layer.data(), layer_size, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  cudaCheckError()
  for(auto i = labels_list_.data; i < labels_list_.data + labels_list_.size; i++ ){
    std::cout << *i << " ";
  }
  std::cout << std::endl;
  MatrixMerge::launchLayerMerge<<<layer.size(),BLOCK_SIZE>>>(d_layer,merge_found,labels_list_);
  cudaDeviceSynchronize();
  cudaCheckError()
  for(auto i = labels_list_.data; i < labels_list_.data + labels_list_.size; i++ ){
    std::cout << *i << " ";
  }
  std::cout << std::endl;
  cudaFree(d_layer);
  return *merge_found;
}

template<class PointType>
std::vector<SubmatrixView<uint8_t>> CudaClustering<PointType>::get_layer(size_t layer_number, bool &_is_last) {
  std::vector<SubmatrixView<uint8_t>> layer{};
  for (unsigned x = layer_number * BLOCK_SIZE; x < labels_list_.size; x += BLOCK_SIZE) {
    unsigned y = x - layer_number * BLOCK_SIZE;
    unsigned blockWidth = std::min((size_t) BLOCK_SIZE, labels_list_.size - x);
    unsigned blockHeight = std::min((size_t) BLOCK_SIZE, labels_list_.size - y);
    layer.emplace_back(d_matrix_, MatrixPoint{x, y}, blockHeight, blockWidth);
  }
  if(layer.size() == 1)
    _is_last = true;
  return layer;
}


template<class PointType>
void CudaClustering<PointType>::update() {

  int cloud_grid_size = std::ceil(((float) cluster_cloud_->size) / BLOCK_SIZE);
  Update::mapClusters<<<cloud_grid_size, BLOCK_SIZE>>>(*cluster_cloud_, labels_list_,labels_map_);
  cudaDeviceSynchronize();
  cudaCheckError()

  size_t labels_pos_data_size = labels_map_.size * sizeof(unsigned);
  cudaMemset(labels_map_.data, 0u, labels_pos_data_size);
  cudaDeviceSynchronize();
  cudaCheckError()

  int label_grid_size = std::ceil(((float) labels_map_.size) / BLOCK_SIZE);
  Update::set_label_list<<<label_grid_size,BLOCK_SIZE>>>(labels_list_, labels_map_);
  cudaDeviceSynchronize();
  cudaCheckError()

  exclusive_scan(labels_map_);

  unsigned unique_clusters = labels_map_.data[labels_map_.size - 1];
  std::cout << "Found " << unique_clusters << " unique labels\n" ;
  DeviceArray<unsigned> labels_list_update{};
  labels_list_update.size = unique_clusters;
  cudaMallocManaged(&labels_list_update.data, labels_list_update.size * sizeof(unsigned));
  cudaDeviceSynchronize();
  cudaCheckError()

  BuildMatrix::set_array_to_tid<<<unique_clusters, BLOCK_SIZE>>>(labels_list_update);
  cudaDeviceSynchronize();
  cudaCheckError()

  Matrix<uint8_t> matrix_update{};
  matrix_update.allocateMatrixZero(unique_clusters * unique_clusters);

  Update::build_matrix<<<label_grid_size, BLOCK_SIZE>>>(d_matrix_,matrix_update,labels_list_,labels_map_);
  cudaDeviceSynchronize();
  cudaCheckError()

  cudaFree(labels_list_.data);
  labels_list_ = labels_list_update;

  d_matrix_.free();
  d_matrix_ = matrix_update;
}

/*
 * declarations of used templates
 */
template
class CudaClustering<pcl::PointXYZ>;

template
class CudaClustering<pcl::PointXYZI>;
