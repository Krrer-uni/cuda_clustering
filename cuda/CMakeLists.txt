find_package(CUDA)

set(CUDA_CLUSTERING_SRC_FILES
        CudaClustering.cu
        DeviceArray.cu
        DeviceMatrix.cu
        ClusterCloud.cu
)
set(CUDA_CLUSTERING_INC_FILES
        include/MatrixClustering.cuh
        include/DeviceArray.cuh
        include/DeviceMatrix.cuh
        include/CudaUtils.cuh
        include/CudaPointCloud.cuh
)

add_library(cuda_clustering ${CUDA_CLUSTERING_INC_FILES} ${CUDA_CLUSTERING_SRC_FILES})

target_link_libraries(cuda_clustering ${CUDA_LIBRARIES})

set_target_properties(cuda_clustering PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
