//
// Created by Wojciech Rymer on 08.10.23.
//

#ifndef EUCLIDEAN_CLUSTERING_CUDA_INCLUDE_CUDAUTILS_CUH_
#define EUCLIDEAN_CLUSTERING_CUDA_INCLUDE_CUDAUTILS_CUH_
#include "cstdio"
/**
 * Macro to check for cuda errors, used after every cuda call
 */
#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}

#endif //EUCLIDEAN_CLUSTERING_CUDA_INCLUDE_CUDAUTILS_CUH_
