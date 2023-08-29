
find_package(CUDA)
add_library(cuda_clustering cuda_clustering.cu cuda_clustering.cuh)

target_link_libraries(cuda_clustering ${CUDA_LIBRARIES})

set_target_properties(cuda_clustering PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
enable_language(CUDA)
