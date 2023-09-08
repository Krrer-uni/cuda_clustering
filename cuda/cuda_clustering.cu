#include "cstdio"
#include "cuda_clustering.cuh"
#include "matrix_clustering.cu"
#include <pcl/point_types.h>

#include <cmath>

#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}
template<class PointType>
CudaClustering<PointType>::CudaClustering() {
  cluster_cloud = std::make_shared<ClusterCloud>();
}

template<class PointType>
CudaClustering<PointType>::~CudaClustering() {
  if (cluster_cloud != nullptr && cluster_cloud->points != nullptr) {
    delete[] cluster_cloud->points;
  }
}

template<class PointType>
void CudaClustering<PointType>::setParams(ClusterParams params) {
  this->params = params;
}

template<class PointType>
void CudaClustering<PointType>::setInputCloud(typename pcl::PointCloud<PointType>::Ptr input_cloud) {
  if (input_cloud == nullptr) {
    return;
  }

  if (cluster_cloud != nullptr && cluster_cloud->points != nullptr) {
    delete[] cluster_cloud->points;
  }
  cluster_cloud->size = input_cloud->size();
  cluster_cloud->points = new CudaPoint[input_cloud->size()];

  for (size_t i = 0; i < input_cloud->size(); i++) {
    cluster_cloud->points[i] = {input_cloud->points.data()[i].x,
                                input_cloud->points.data()[i].y,
                                input_cloud->points.data()[i].z};
  }
}

template<class PointType>
void CudaClustering<PointType>::extract(std::vector<unsigned int> &indices_clusters) {
  size_t points_data_bytes = cluster_cloud->size * (sizeof(CudaPoint));
  size_t labels_data_bytes = cluster_cloud->size * sizeof(unsigned);

  ClusterCloud d_cluster_cloud{};
  d_cluster_cloud.size = cluster_cloud->size;
  cudaMalloc(&d_cluster_cloud.points, points_data_bytes);
  cudaCheckError()
  cudaMemcpy(d_cluster_cloud.points,
             cluster_cloud->points,
             points_data_bytes,
             cudaMemcpyHostToDevice);
  cudaCheckError()

  cudaMalloc(&d_cluster_cloud.labels, labels_data_bytes);
  cudaCheckError()

  DeviceArray<unsigned> d_labels_list{};
  d_labels_list.size = cluster_cloud->size;
  size_t labels_list_data_bytes = cluster_cloud->size * sizeof(*d_labels_list.data);
  cudaMalloc(&d_labels_list.data, labels_data_bytes);
  cudaCheckError()

  int grid_size = std::ceil(((float) d_cluster_cloud.size) / BLOCK_SIZE);
  initial_ec<<<grid_size, BLOCK_SIZE>>>(d_cluster_cloud, params.distance);

  build_matrix(d_cluster_cloud, params.distance, d_labels_list);

  cudaMemcpy(indices_clusters.data(),
             d_cluster_cloud.labels,
             labels_data_bytes,
             cudaMemcpyDeviceToHost);

  cudaFree(&d_cluster_cloud.labels);
  cudaFree(&d_cluster_cloud.points);
  cudaFree(&d_labels_list.data);
}

template<class PointType>
void CudaClustering<PointType>::build_matrix(ClusterCloud &cluster_cloud,
                                             const float d_th,
                                             DeviceArray<unsigned int> &labels_list) {

  DeviceArray<unsigned> d_labels_pos{};
  d_labels_pos.size = labels_list.size;
  cudaMalloc(&labels_list.data, labels_list.size * sizeof (unsigned));
  cudaCheckError()
  cudaMemset(&labels_list.data, 0,labels_list.size * sizeof (unsigned));
  cudaCheckError()

  set_label_list<<<1,d_labels_pos.size>>>(cluster_cloud,d_labels_pos);
  cudaCheckError()

}

template<class PointType>
void CudaClustering<PointType>::initial_clustering() {

}

/*
 * declaration of used templates
 */
template
class CudaClustering<pcl::PointXYZ>;
template
class CudaClustering<pcl::PointXYZI>;