#define BLOCK_SIZE 16
#include "kernels/cuda_debug_utills.cu"

__device__ float point_distance(const CudaPoint a, const CudaPoint b) {
  return sqrt((a.x - b.x) * (a.x - b.x) +
      (a.y - b.y) * (a.y - b.y) +
      (a.z - b.z) * (a.z - b.z));
}

/**
 * performs initial clustering, called with 1D grid of 1D blocks.
 * @param cluster_cloud
 */
__global__ void initial_ec(ClusterCloud cluster_cloud, float d_th) {
  int b = blockDim.x;
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  size_t global_id = tid + bid * b;
  if (global_id >= cluster_cloud.size)
    return;
  CudaPoint p = cluster_cloud.points[global_id];

  __shared__ CudaPoint shared_points[BLOCK_SIZE];
  __shared__ int labels[BLOCK_SIZE];
  __shared__ int status[BLOCK_SIZE];
  __syncthreads();

  shared_points[tid] = p;
  labels[tid] = tid;
  __syncthreads();
  for (int j = 0; j < b - 1; j++) {
    status[tid] = 0;
    CudaPoint q = shared_points[j];
    int cc = labels[tid];  // current label of thread point
    int rc = labels[j];  // current thread of compared point
    if (tid > j && point_distance(q, p) < d_th && rc != cc) {
      status[cc] = 1;
    }
    __syncthreads();
    if (status[cc] == 1) {
      labels[tid] = rc;
    }
    __syncthreads();
  }
  cluster_cloud.labels[global_id] = labels[tid] + bid * b;
}

namespace BuildMatrix {

__global__ void set_array_to_tid(DeviceArray<unsigned> array) {
  size_t tid = threadIdx.x;
  if (tid >= array.size)
    return;
  array.data[tid] = tid;
}

__global__ void set_label_list(const ClusterCloud cluster_cloud,
                               DeviceArray<unsigned> labels_pos) {
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= cluster_cloud.size)
    return;
  size_t cluster_id = cluster_cloud.labels[tid];
  if (cluster_id >= labels_pos.size)
    return;
  labels_pos.data[cluster_id] = 1;
}

__global__ void cluster_update(const ClusterCloud cluster_cloud,
                               DeviceArray<unsigned> cluster_map) {

  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= cluster_cloud.size)
    return;
  unsigned new_label = cluster_map.data[cluster_cloud.labels[tid]];
  cluster_cloud.labels[tid] = new_label;

}
/**
 * Populates the adjacency matrix
 * @param cluster_cloud points and their labels
 * @param matrix adjacency matrix of clusters
 * @param matrix_dim dimension of square matrix
 * @param d_th distance threshold between points
 */
__global__ void
populate_matrix(const ClusterCloud cluster_cloud, Matrix<uint8_t> matrix, float d_th) {
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= cluster_cloud.size)
    return;
  CudaPoint p = cluster_cloud.points[tid];
  for (int i = 0; i < cluster_cloud.size; i++) {
    CudaPoint q = cluster_cloud.points[i];
    if (point_distance(p, q) < d_th) {
      size_t matrix_index = cluster_cloud.labels[tid] + cluster_cloud.labels[i] * matrix.matrix_step;
      matrix.matrix.data[matrix_index] = 1;
    }
  }
}

}  // namespace BuildMatrix

namespace MatrixMerge {

__device__ uint8_t getSubmatField(SubmatrixView<uint8_t> submat, MatrixPoint field) {
  unsigned offset_step = submat.submatrix_origin_.x + submat.submatrix_origin_.y * submat.parent_matrix_.matrix_step;
  unsigned field_step = field.x + field.y * submat.parent_matrix_.matrix_step;
  if (field_step + offset_step >= submat.parent_matrix_.size) {
    printf("Diagonal{%d,%d}: Error accessing (%d,%d)",
           submat.submatrix_origin_.x,
           submat.submatrix_origin_.y,
           field.x,
           field.y);
    return 255;
  }
  return submat.parent_matrix_.matrix.data[field_step + offset_step];
}


__device__
void
diagonal(SubmatrixView<uint8_t> submatrix,
         bool *was_merged,
         DeviceArray<unsigned> label_list) {
  unsigned tid = threadIdx.x;
//  print_status(tid,blockIdx.x);
//  printf("submat_step %d\n",submatrix.submatrix_step_ );
  __shared__ unsigned labels[BLOCK_SIZE];
  __shared__ bool status[BLOCK_SIZE];
  labels[tid] = tid;
  __syncthreads();
//  printf("first sync\n");
  for (unsigned j = 0; j < submatrix.submatrix_step_ - 1; j++) {
    status[tid] = false;
    __syncthreads();
//    printf("first for sync\n");

    unsigned cc = labels[tid];  // current label of thread point
    unsigned rc = labels[j];  // current thread of compared point
//    printf("Got the cc rc (%d, %d)\n",cc, rc);

    uint8_t label_status = getSubmatField(submatrix, {tid, j});
//    printf("Got the field (%d, %d)\n",tid, j);
    if (tid > j && label_status == 1 && rc != cc) {
      status[cc] = true;
    }
    __syncthreads();
//    printf("second for sync\n");

    if (status[cc]) {
      labels[tid] = rc;
      *was_merged = true;
    }
    __syncthreads();
//    printf("third for sync\n");

  }
  if (tid >= submatrix.submatrix_step_){
    label_list.data[tid + submatrix.submatrix_origin_.x] = labels[tid];
  }
}

__device__ void offdiagonal(SubmatrixView<uint8_t> submatrix,
                            bool *was_merged,
                            DeviceArray<unsigned> label_list) {
  unsigned tid = threadIdx.x;
  unsigned submat_x = submatrix.submatrix_origin_.x / BLOCK_SIZE;  // g1
  unsigned submat_y = submatrix.submatrix_origin_.y / BLOCK_SIZE;  // g0

  if (tid >= submatrix.submatrix_step_)
    return;
  unsigned cl = tid;
  unsigned c = submat_x + tid;
  bool merged_thread = false;
  __shared__ bool merged_block;
  __shared__ bool status[2 * BLOCK_SIZE];

  if (tid == 0) merged_block = 0;
  __syncthreads();
  for (unsigned j = 0; j < submatrix.submatrix_step_; j++) {
    unsigned r = submat_y + j;
    status[tid] = false;
    status[tid + BLOCK_SIZE] = false;
    __syncthreads();

    uint8_t label_status = getSubmatField(submatrix, {tid, j});
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
    label_list.data[tid + submat_x] = submat_y + cl - BLOCK_SIZE;
    merged_block = true;
    if(merged_block && tid == 0){}
    *was_merged = true;
  }
}

__global__ void launchLayerMerge(SubmatrixView<uint8_t> *layer, bool *was_merged, DeviceArray<unsigned> label_list) {
  unsigned tid = threadIdx.x;
  unsigned bid = blockIdx.x;
  printf("Launched submat {%d, %d}\n", layer[bid].submatrix_origin_.x,layer[bid].submatrix_origin_.y);
  if (layer[bid].submatrix_origin_.x == layer[bid].submatrix_origin_.y) {
    diagonal(layer[bid], was_merged, label_list);
  } else {
    offdiagonal(layer[bid], was_merged, label_list);
  }
}
}  // namespace MatrixMerge

namespace Update {
__global__ void mapClusters(ClusterCloud cluster_cloud,
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

__global__ void build_matrix(Matrix<uint8_t> matrix,
                             Matrix<uint8_t> matrix_update,
                             DeviceArray<unsigned> label_list,
                             DeviceArray<unsigned> label_map) {
  unsigned column = threadIdx.x + blockIdx.x * blockDim.x;
  if (column >= matrix.matrix_step) {
    return;
  }

  unsigned column_updated = label_list.data[label_map.data[column]];
  for (unsigned row = 0; row < column; row++) {
    unsigned row_updated = label_list.data[label_map.data[row]];

    size_t matrix_index = row * matrix.matrix_step + column;
    if (matrix.matrix.data[matrix_index] == 1) {
      size_t matrix_update_index = row_updated * matrix_update.matrix_step + column_updated;

      matrix_update.matrix.data[matrix_update_index] = 1;
    }
  }
}
}