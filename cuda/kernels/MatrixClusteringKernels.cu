#ifndef EUCLIDEAN_CLUSTERING_MATRIX_KERNEL_CU_
#define EUCLIDEAN_CLUSTERING_MATRIX_KERNEL_CU_

#define BLOCK_SIZE 128
#include "kernels/cuda_debug_utills.cu"
#include "include/MatrixClustering.cuh"

__device__ float point_distance(const CudaPoint a, const CudaPoint b) {
  return sqrt((a.x - b.x) * (a.x - b.x) +
      (a.y - b.y) * (a.y - b.y) +
      (a.z - b.z) * (a.z - b.z));
}

/**
 * performs initial clustering, called with 1D grid of 1D blocks.
 * @param cluster_cloud points and their lables
 * @param d_th euclidean clustering distance
 */
__global__ void initial_ec(CudaPointCloud cluster_cloud, float d_th) {
  unsigned tid = threadIdx.x;
  unsigned bid = blockIdx.x;
  unsigned b_top = min((unsigned)cluster_cloud.size - blockDim.x * blockIdx.x, blockDim.x);
  size_t gid = tid + bid * blockDim.x;  // get global thread id
  CudaPoint p{};
  if (gid < cluster_cloud.size){
    p = cluster_cloud.points[gid];
  }

  // allocate shared libraries
  __shared__ CudaPoint shared_points[BLOCK_SIZE];
  __shared__ unsigned labels[BLOCK_SIZE];
  __shared__ bool status[BLOCK_SIZE];

  shared_points[tid] = p;
  labels[tid] = tid;
  __syncthreads();
  for (int j = 0; j < b_top - 1; j++) {
    status[tid] = false;
    CudaPoint q = shared_points[j];
    unsigned cc = labels[tid];  // column cluster
    unsigned rc = labels[j];  // row cluster
    if (tid > j && point_distance(q, p) < d_th && rc != cc) {
      status[cc] = true;
    }
    __syncthreads();
    if (status[cc]) {
      labels[tid] = rc;
    }
    __syncthreads();
  }
  if(gid < cluster_cloud.size){
    cluster_cloud.labels[gid] = labels[tid] + bid * blockDim.x;
  }
}

namespace BuildMatrix {

__global__ void set_array_to_tid(DeviceArray<unsigned> array) {
  size_t global_id = threadIdx.x + blockIdx.x * blockDim.x;
  if (global_id >= array.size)
    return;
  array.data[global_id] = global_id;
}

__global__ void set_label_list(const CudaPointCloud cluster_cloud,
                               DeviceArray<unsigned> labels_pos) {
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= cluster_cloud.size)
    return;
  size_t cluster_id = cluster_cloud.labels[tid];
  if (cluster_id >= labels_pos.size)
    return;
  labels_pos.data[cluster_id] = 1;
}

__global__ void cluster_update(const CudaPointCloud cluster_cloud,
                               DeviceArray<unsigned> cluster_map) {

  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= cluster_cloud.size)
    return;
  cluster_cloud.labels[tid] = cluster_map.data[cluster_cloud.labels[tid]];
}
/**
 * Populates the adjacency matrix
 * @param cluster_cloud points and their labels
 * @param matrix adjacency matrix of clusters
 * @param d_th distance threshold between points
 */
__global__ void
populate_matrix(const CudaPointCloud cluster_cloud, DeviceMatrix<uint8_t> matrix, float d_th) {
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= cluster_cloud.size)  // check if thread is excess
    return;
  CudaPoint p = cluster_cloud.points[tid];
  for (int i = 0; i <= tid; i++) {
    CudaPoint q = cluster_cloud.points[i];
    if (point_distance(p, q) < d_th ) {
      size_t matrix_index = cluster_cloud.labels[tid] + cluster_cloud.labels[i] * matrix.step;
      matrix.matrix.data[matrix_index] = 1;
    }
  }
}

}  // namespace BuildMatrix

namespace MatrixMerge {

__device__ uint8_t getSubmatField(DeviceSubmatrixView<uint8_t> submat, MatrixPoint field) {
  unsigned offset_step = submat.origin_.x + submat.origin_.y * submat.parent_matrix_.step;
  unsigned field_step = field.x + field.y * submat.parent_matrix_.step;
  if (field_step + offset_step >= submat.parent_matrix_.size) {
    printf("Submat{%d,%d}: Error accessing (%d,%d)\n",
           submat.origin_.x,
           submat.origin_.y,
           field.x,
           field.y);
    return 255;
  }
  return submat.parent_matrix_.matrix.data[field_step + offset_step];
}


__device__
void
diagonal(DeviceSubmatrixView<uint8_t> submatrix,
         bool *was_merged,
         DeviceArray<unsigned> label_list) {
  unsigned tid = threadIdx.x;
  bool is_out_of_bound = tid > submatrix.step_-1;

  __shared__ unsigned labels[BLOCK_SIZE];
  __shared__ bool status[BLOCK_SIZE];
  bool thread_merged = false;
  labels[tid] = tid;
  __syncthreads();
  for (unsigned j = 0; j < submatrix.step_ - 1; j++) {
    status[tid] = false;
    __syncthreads();

    unsigned cc = labels[tid];  // current label of thread point
    unsigned rc = labels[j];  // current thread of compared point

    uint8_t label_status = (is_out_of_bound) ? 0 : getSubmatField(submatrix, {tid, j});
    if (tid > j && label_status == 1 && rc != cc) {
      status[cc] = true;
    }
    __syncthreads();

    if (status[cc]) {
      labels[tid] = rc;
      thread_merged = true;
      *was_merged = true;
    }
    __syncthreads();
  }
  if (thread_merged){
//    printf("changed %d to %d\n",label_list.data[tid + submatrix.submatrix_origin_.x], labels[tid]+ submatrix.submatrix_origin_.x);
    label_list.data[tid + submatrix.origin_.x] = labels[tid]+ submatrix.origin_.x;
  }
}

__device__ void offdiagonal(DeviceSubmatrixView<uint8_t> submatrix,
                            bool *was_merged,
                            DeviceArray<unsigned> label_list) {
  unsigned tid = threadIdx.x;
  unsigned submat_x = submatrix.origin_.x / BLOCK_SIZE;  // g1
  unsigned submat_y = submatrix.origin_.y / BLOCK_SIZE;  // g0

  unsigned cl = tid;
  bool merged_thread = false;
  __shared__ bool merged_block;
  __shared__ bool status[2 * BLOCK_SIZE];

  if (tid == 0) merged_block = false;
  __syncthreads();
  for (unsigned j = 0; j < submatrix.height_; j++) {
    status[tid] = false;
    status[tid + BLOCK_SIZE] = false;
    __syncthreads();


    uint8_t label_status = (tid >= submatrix.step_) ? 0 : getSubmatField(submatrix, {tid, j});
    if (label_status == 1) {
      status[cl] = true;

    }
    if (status[cl]) {
      cl = BLOCK_SIZE + j;
      merged_thread = true;
    }
    __syncthreads();
  }
  if (merged_thread) {
    label_list.data[tid + submatrix.origin_.x] = submatrix.origin_.y + cl - BLOCK_SIZE;
    merged_block = true;
  }
  __syncthreads();

  if(merged_block && tid == 0){
    *was_merged = true;
  }
}

__global__ void launchLayerMerge(DeviceSubmatrixView<uint8_t> *layer, bool *was_merged, DeviceArray<unsigned> label_list) {
  unsigned tid = threadIdx.x;
  unsigned bid = blockIdx.x;
//  printf("Launched submat {%d, %d}\n", layer[bid].submatrix_origin_.x,layer[bid].submatrix_origin_.y);
  if (layer[bid].origin_.x == layer[bid].origin_.y) {
    diagonal(layer[bid], was_merged, label_list);
  } else {
    offdiagonal(layer[bid], was_merged, label_list);
  }
}
}  // namespace MatrixMerge

namespace Update {
__global__ void mapClusters(CudaPointCloud cluster_cloud,
                            DeviceArray<unsigned> label_list,
                            DeviceArray<unsigned> label_map) {
  unsigned id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id >= cluster_cloud.size) {
    return;
  }
  cluster_cloud.labels[id] = label_list.data[label_map.data[cluster_cloud.labels[id]]];
}

__global__ void set_label_list(DeviceArray<unsigned> labels_list,
                               DeviceArray<unsigned> labels_pos) {
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= labels_list.size)
    return;
  size_t cluster_id = labels_list.data[tid];
  if (cluster_id >= labels_pos.size)
    return;
  labels_pos.data[cluster_id] = 1;
}

__global__ void build_matrix(DeviceMatrix<uint8_t> matrix,
                             DeviceMatrix<uint8_t> matrix_update,
                             DeviceArray<unsigned> label_list,
                             DeviceArray<unsigned> label_map) {
  unsigned column = threadIdx.x + blockIdx.x * blockDim.x;
  if (column >= matrix.step) {
    return;
  }

  unsigned column_updated = label_map.data[label_list.data[column]];
  for (unsigned row = 0; row < column; row++) {
    unsigned row_updated = label_map.data[label_list.data[row]];

    size_t matrix_index = row * matrix.step + column;
    if (matrix.matrix.data[matrix_index] == 1) {
      size_t matrix_update_index = row_updated * matrix_update.step + column_updated;

      matrix_update.matrix.data[matrix_update_index] = 1;
    }
  }
}
}

#endif //EUCLIDEAN_CLUSTERING_MATRIX_KERNEL_CU_