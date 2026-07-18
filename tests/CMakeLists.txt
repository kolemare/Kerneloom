file(GLOB_RECURSE KL_TEST_SOURCES CONFIGURE_DEPENDS
    suites/*.cpp
)

add_executable(kerneloom_tests
    test_main.cpp
    ${KL_TEST_SOURCES}
)

target_link_libraries(kerneloom_tests
    PRIVATE
        kerneloom_lib
        GTest::gtest
        GTest::gtest_main
)

target_include_directories(kerneloom_tests
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/third_party
        ${CMAKE_CURRENT_SOURCE_DIR}
)

if(KL_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)

    target_link_libraries(kerneloom_tests
        PRIVATE
            CUDA::cublas
    )
endif()

if(KL_ENABLE_ROCM)
    find_package(rocblas REQUIRED)

    target_link_libraries(kerneloom_tests
        PRIVATE
            roc::rocblas
    )
endif()