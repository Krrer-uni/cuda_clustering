#define BLOCK_SIZE 16

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

__global__ void build_matrix_kernel(ClusterCloud cluster_cloud,
                                    float d_th,
                                    DeviceArray<unsigned> labels_list,
                                    DeviceArray<unsigned> labels_pos) {

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

__device__ void getSubmatField(SubmatrixView<uint8_t> submat, MatrixPoint field, uint8_t* out){
  unsigned offset_step = submat.submatrix_origin_.x + submat.submatrix_origin_.y * submat.parent_matrix_->matrix_step;
  unsigned field_step = field.x + field.y * submat.parent_matrix_->matrix_step;
  if(field_step + offset_step >= submat.parent_matrix_->size){
    *out = 255;
    return;
  }
  *out = submat.parent_matrix_->matrix.data[field_step + offset_step];
}

__global__ void
diagonal(ClusterCloud cluster_cloud,
         SubmatrixView<uint8_t> submatrix,
         bool *was_merged,
         DeviceArray<unsigned> label_map) {
  unsigned b = blockDim.x;
  unsigned tid = threadIdx.x;
  if (tid >= submatrix.submatrix_step_)
    return;

  __shared__ unsigned labels[BLOCK_SIZE];
  __shared__ bool status[BLOCK_SIZE];
  __syncthreads();

  labels[tid] = tid;
  __syncthreads();
  for (unsigned j = 0; j < b - 1; j++) {
    status[tid] = false;

    unsigned cc = labels[tid];  // current label of thread point
    unsigned rc = labels[j];  // current thread of compared point

    uint8_t label_status;
    getSubmatField(submatrix,{tid,j},&label_status);
    if (tid > j && label_status == 1 && rc != cc) {
      status[cc] = true;
    }
    __syncthreads();
    if (status[cc]) {
      labels[tid] = rc;
      *was_merged = true;
    }
    __syncthreads();
  }
  label_map.data[tid + submatrix.submatrix_origin_.x] = labels[tid];
}
}