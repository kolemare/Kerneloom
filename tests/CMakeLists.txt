file(GLOB_RECURSE KL_TEST_SUITE_SOURCES CONFIGURE_DEPENDS
    suites/*.cpp
)

set(KL_TEST_VENDOR_SOURCES)

if(KL_ENABLE_CUDA)
    file(GLOB_RECURSE KL_TEST_CUDA_VENDOR_SOURCES CONFIGURE_DEPENDS
        vendor/cuda/*.cu
    )

    list(APPEND KL_TEST_VENDOR_SOURCES
        ${KL_TEST_CUDA_VENDOR_SOURCES}
    )
endif()

if(KL_ENABLE_ROCM)
    file(GLOB_RECURSE KL_TEST_ROCM_VENDOR_SOURCES CONFIGURE_DEPENDS
        vendor/rocm/*.hip
    )

    list(APPEND KL_TEST_VENDOR_SOURCES
        ${KL_TEST_ROCM_VENDOR_SOURCES}
    )
endif()

add_executable(kerneloom_tests
    ${KL_TEST_SUITE_SOURCES}
    ${KL_TEST_VENDOR_SOURCES}
)

target_link_libraries(kerneloom_tests
    PRIVATE
        kerneloom_lib
        GTest::gtest_main
)

target_include_directories(kerneloom_tests
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}
)

if(KL_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    find_package(CUDNN REQUIRED)

    target_link_libraries(kerneloom_tests
        PRIVATE
            CUDA::cublas
            CUDA::cudnn
            CUDA::curand
    )
endif()

if(KL_ENABLE_ROCM)
    find_package(rocblas REQUIRED)
    find_package(miopen REQUIRED)
    find_package(hiprand REQUIRED)

    target_link_libraries(kerneloom_tests
        PRIVATE
            roc::rocblas
            MIOpen
            hiprand
    )
endif()