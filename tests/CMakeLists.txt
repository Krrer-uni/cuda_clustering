find_package(CUDA)

set(KERNEL_TEST_FILES kernels/BuildMatrix.cu)
include_directories(${PROJECT_SOURCE_DIR}/cuda/kernels)

add_library(cuda_interface test_interface.cu ${KERNEL_TEST_FILES}
        test_interface.cuh)
target_link_libraries(cuda_interface ${CUDA_LIBRARIES})

set_target_properties(cuda_interface PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

add_executable(tests test_main.cpp)

target_link_libraries(tests ${PCL_LIBRARIES} cuda_interface)
