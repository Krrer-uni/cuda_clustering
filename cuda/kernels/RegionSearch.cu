#include "include/RegionSearch.cuh"
#include <limits>
#include "include/CudaUtils.cuh"

void RegionSearch::setCloud(ClusterCloud &cloud) {
  _cloud = cloud;
}

__global__ void parallelMin(CudaPoint* in, CudaPoint* out, size_t in_size){
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  size_t stride = ceil(in_size/2.0);
  if(tid + stride >= in_size)
    return;

  out[tid].x = min(in[tid].x,in[tid+stride].x);
  out[tid].y = min(in[tid].y,in[tid+stride].y);
  out[tid].z = min(in[tid].z,in[tid+stride].z);
}

__global__ void parallelMax(CudaPoint* in, CudaPoint* out, size_t in_size){
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  size_t stride = ceil(in_size/2.0);
  if(tid + stride >= in_size)
    return;

  out[tid].x = max(in[tid].x,in[tid+stride].x);
  out[tid].y = max(in[tid].y,in[tid+stride].y);
  out[tid].z = max(in[tid].z,in[tid+stride].z);
}


void RegionSearch::build(float threshold) {
  float minX, maxX, minY, maxY, minZ, maxZ;

  minX = minY = minZ = -INFINITY;
  maxX = maxY = maxZ = INFINITY;
  //TODO determine min/max
  auto min = RegionSearch::min();
  auto max = RegionSearch::max();

  size_t rowSize = std::ceil((maxX - minX) / threshold);
  size_t planeSize = rowSize * std::ceil((maxY - minY) / threshold);

  size_t voxelsSize = planeSize * std::ceil((maxZ - minZ) / threshold);
  // voxel position is calculated by x + y * rowSize + z * planeSize
  DeviceArray<unsigned> vid{};
  vid.size = _cloud.size;
  cudaMallocManaged(&vid.data, _cloud.size * sizeof(unsigned));

}

CudaPoint RegionSearch::min(){
  size_t size = _cloud.size;
  size_t new_size = std::ceil(size/2.0);
  size_t grid_size = std::ceil(((float) new_size) / block_size_);
  CudaPoint* reductionArray;
  cudaMallocManaged(&reductionArray,new_size * sizeof(CudaPoint));

  parallelMin<<<grid_size,block_size_>>>(_cloud.points, reductionArray, size);
  cudaDeviceSynchronize();
  cudaCheckError()

  size = new_size;
  new_size = std::ceil(size/2.0);

  while(size > 1){
    parallelMin<<<grid_size,block_size_>>>(reductionArray, reductionArray, size);
    cudaDeviceSynchronize();
    cudaCheckError()

    size = new_size;
    new_size = std::ceil(size/2.0);
  }

  auto output = reductionArray[0];
  cudaFree(reductionArray);
  cudaDeviceSynchronize();
  cudaCheckError()
  return output;
}

CudaPoint RegionSearch::max(){
  size_t size = _cloud.size;
  size_t new_size = std::ceil(size/2.0);
  size_t grid_size = std::ceil(((float) new_size) / block_size_);
  CudaPoint* reductionArray;
  cudaMallocManaged(&reductionArray,new_size * sizeof(CudaPoint));

  parallelMax<<<grid_size,block_size_>>>(_cloud.points, reductionArray, size);
  cudaDeviceSynchronize();
  cudaCheckError()

  size = new_size;
  new_size = std::ceil(size/2.0);

  while(size > 1){
    parallelMax<<<grid_size,block_size_>>>(reductionArray, reductionArray, size);
    cudaDeviceSynchronize();
    cudaCheckError()

    size = new_size;
    new_size = std::ceil(size/2.0);
  }

  auto output = reductionArray[0];
  cudaFree(reductionArray);
  cudaDeviceSynchronize();
  cudaCheckError()
  return output;
}

RegionSearch::RegionSearch(ClusterCloud &cloud) : _cloud(cloud){

}
