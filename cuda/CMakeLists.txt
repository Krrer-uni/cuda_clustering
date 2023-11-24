find_package(CUDA)

set(CUDA_CLUSTERING_SRC_FILES CudaClustering.cu DeviceArray.cu Matrix.cu
        ClusterCloud.cu
        kernels/RegionSearch.cu)
set(CUDA_CLUSTERING_INC_FILES include/CudaClustering.cuh include/DeviceArray.cuh include/Matrix.cuh
        include/CudaUtils.cuh
        include/ClusterCloud.cuh
        include/RegionSearch.cuh)

add_library(cuda_clustering ${CUDA_CLUSTERING_INC_FILES} ${CUDA_CLUSTERING_SRC_FILES})

target_link_libraries(cuda_clustering ${CUDA_LIBRARIES})

set_target_properties(cuda_clustering PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
