cmake_minimum_required(VERSION 3.25)
project(euclidean_clustering CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(euclidean_clustering main.cu)

set_target_properties(euclidean_clustering PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
