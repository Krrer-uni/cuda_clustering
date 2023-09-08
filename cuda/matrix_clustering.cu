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
  for (int j = 0; j < b - 1; j++) {
    status[tid] = 0;
    CudaPoint q = shared_points[j];
    int cc = labels[tid];
    int rc = labels[j];
    if (tid > j && point_distance(q, p) < d_th && rc != cc) {
      status[cc] = 1;
    }
    if (status[cc] == 1) {
      labels[tid] = rc;
    }
    __syncthreads();
  }
  cluster_cloud.labels[global_id] = labels[tid];
}

__global__ void build_matrix_kernel(ClusterCloud cluster_cloud,
                                    float d_th,
                                    DeviceArray<unsigned> labels_list,
                                    DeviceArray<unsigned> labels_pos) {

}

__global__ void set_label_list(const ClusterCloud& cluster_cloud, DeviceArray<unsigned>& labels_pos){
  size_t tid = threadIdx.x;
  size_t cluster_id = cluster_cloud.labels[tid];
  if(cluster_id >= labels_pos.size)
    return;
  labels_pos.data[cluster_id] = 1;
}
__global__ void scan(float *g_odata, float *g_idata, int n) {
  extern __shared__ float temp[]; // allocated on invocation
   int thid = threadIdx.x;
   int pout = 0, pin = 1;// Load input into shared memory.
   // This is exclusive scan, so shift right by one
   // and set first element to 0
   temp[pout*n + thid] = (thid > 0) ? g_idata[thid-1] : 0;
   __syncthreads();
   for (int offset = 1; offset < n; offset *= 2)   {
     pout = 1 - pout; // swap double buffer indices
      pin = 1 - pout;
      if (thid >= offset)
        temp[pout*n+thid] += temp[pin*n+thid - offset];
      else
        temp[pout*n+thid] = temp[pin*n+thid];
      __syncthreads();
   }   g_odata[thid] = temp[pout*n+thid]; // write output }
   }
