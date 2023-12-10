cmake_minimum_required(VERSION 3.25)
project(euclidean_clustering LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

find_package(PCL REQUIRED)

include_directories(${PCL_INCLUDE_DIRS} ${PROJECT_BINARY_DIR} ${PROJECT_SOURCE_DIR}/cuda)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_subdirectory(${PROJECT_SOURCE_DIR}/cuda)
add_subdirectory(${PROJECT_SOURCE_DIR}/tests)
set(cxx_flags -Wall -Wextra -Wpedantic -Wno-attributes -Wno-unknown-pragmas)
set(nvcc_flags  --compiler-options -fpic)

add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:${cxx_flags}>"
        "$<$<COMPILE_LANGUAGE:CUDA>:${nvcc_flags}>")
add_executable(main src/main.cpp)

target_link_libraries(main ${PCL_LIBRARIES} cuda_clustering)

