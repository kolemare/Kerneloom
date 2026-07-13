add_executable(kerneloom_tests
    test_main.cpp
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

include(GoogleTest)
gtest_discover_tests(kerneloom_tests)