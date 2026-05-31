cmake_minimum_required(VERSION 3.21)

project(Kerneloom LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(KL_ENABLE_CUDA "Enable CUDA backend" OFF)
option(KL_ENABLE_ROCM "Enable ROCm backend" OFF)

file(GLOB_RECURSE KL_SOURCES CONFIGURE_DEPENDS
    src/core/*.cpp
    src/backend/*.cpp
    src/backend/cpu/*.cpp
    src/cnn/*.cpp
    src/data/*.cpp
    src/ops/*.cpp
    src/kernels/cpu/*.cpp
    src/third_party/*.cpp
)

if(KL_ENABLE_CUDA)
    enable_language(CUDA)
    add_compile_definitions(KL_ENABLE_CUDA)

    file(GLOB_RECURSE KL_CUDA_SOURCES CONFIGURE_DEPENDS
        src/backend/cuda/*.cu
        src/kernels/cuda/*.cu
    )

    list(APPEND KL_SOURCES ${KL_CUDA_SOURCES})
endif()

if(KL_ENABLE_ROCM)
    enable_language(HIP)
    add_compile_definitions(KL_ENABLE_ROCM)

    file(GLOB_RECURSE KL_ROCM_SOURCES CONFIGURE_DEPENDS
        src/backend/rocm/*.hip
        src/kernels/rocm/*.hip
    )

    list(APPEND KL_SOURCES ${KL_ROCM_SOURCES})
endif()

add_library(kerneloom_lib ${KL_SOURCES})

target_include_directories(kerneloom_lib PUBLIC
    include
    third_party
)

add_executable(kerneloom test/test_loss.cpp)

target_link_libraries(kerneloom PRIVATE kerneloom_lib)