cmake_minimum_required(VERSION 3.24)

project(Kerneloom LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(KERNELOOM_BACKEND "CPU" CACHE STRING "Backend to build: CPU, CUDA, or ROCM")
set_property(CACHE KERNELOOM_BACKEND PROPERTY STRINGS CPU CUDA ROCM)

message(STATUS "Kerneloom backend: ${KERNELOOM_BACKEND}")

add_library(kerneloom STATIC
    src/core/device.cpp
    src/core/shape.cpp
    src/core/tensor.cpp
    src/backend/backend.cpp
    src/ops/matmul.cpp
)

target_include_directories(kerneloom
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_definitions(kerneloom
    PUBLIC
        KERNELOOM_HAS_CPU=0
        KERNELOOM_HAS_CUDA=0
        KERNELOOM_HAS_ROCM=0
)

if (KERNELOOM_BACKEND STREQUAL "CPU")
    target_sources(kerneloom PRIVATE
        kernels/cpu/matmul_cpu.cpp
    )

    target_compile_definitions(kerneloom
        PUBLIC
            KERNELOOM_HAS_CPU=1
            KERNELOOM_DEFAULT_BACKEND_CPU=1
    )

elseif (KERNELOOM_BACKEND STREQUAL "CUDA")
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    target_sources(kerneloom PRIVATE
        kernels/cuda/matmul_cuda.cu
    )

    target_link_libraries(kerneloom
        PRIVATE
            CUDA::cudart
    )

    target_compile_definitions(kerneloom
        PUBLIC
            KERNELOOM_HAS_CUDA=1
            KERNELOOM_DEFAULT_BACKEND_CUDA=1
    )

    set_target_properties(kerneloom PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_SEPARABLE_COMPILATION ON
    )

elseif (KERNELOOM_BACKEND STREQUAL "ROCM")
    enable_language(HIP)
    find_package(hip REQUIRED CONFIG)

    target_sources(kerneloom PRIVATE
        kernels/rocm/matmul_rocm.hip
    )

    set_source_files_properties(
        kernels/rocm/matmul_rocm.hip
        PROPERTIES LANGUAGE HIP
    )

    target_link_libraries(kerneloom
        PRIVATE
            hip::host
    )

    target_compile_definitions(kerneloom
        PUBLIC
            KERNELOOM_HAS_ROCM=1
            KERNELOOM_DEFAULT_BACKEND_ROCM=1
    )

    set_target_properties(kerneloom PROPERTIES
        HIP_STANDARD 17
        HIP_STANDARD_REQUIRED ON
    )

else()
    message(FATAL_ERROR "Invalid KERNELOOM_BACKEND='${KERNELOOM_BACKEND}'. Use CPU, CUDA, or ROCM.")
endif()

add_executable(kerneloom_playground
    playground/main.cpp
)

target_link_libraries(kerneloom_playground
    PRIVATE
        kerneloom
)