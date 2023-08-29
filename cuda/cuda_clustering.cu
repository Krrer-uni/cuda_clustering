#include "cstdio"
#include "cuda_clustering.cuh"

#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}

__global__ void cuda_hello() {
  auto n = threadIdx.x;
  printf("Hello from thread %d!!\n", n);
}

void cuda_wrapper(int n){
  cuda_hello<<<1,n>>>();
  cudaDeviceSynchronize();
  cudaCheckError()
}
