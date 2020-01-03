cmake_minimum_required(VERSION 3.15)
project(cuda_benchmark)

set(CMAKE_CXX_STANDARD 17)
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-extended-lambda")

cuda_add_executable(cuda_benchmark main.cu cuda_benchmark.h)
target_include_directories(cuda_benchmark PRIVATE ${CUDA_INCLUDE_DIRS})
