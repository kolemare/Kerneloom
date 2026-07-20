include(FindPackageHandleStandardArgs)

find_path(CUDNN_INCLUDE_DIR
    NAMES
        cudnn.h
    HINTS
        ${CUDNN_ROOT}
        $ENV{CUDNN_ROOT}
        ${CUDAToolkit_INCLUDE_DIRS}
    PATH_SUFFIXES
        include
)

find_library(CUDNN_LIBRARY
    NAMES
        cudnn
    HINTS
        ${CUDNN_ROOT}
        $ENV{CUDNN_ROOT}
        ${CUDAToolkit_LIBRARY_DIR}
    PATH_SUFFIXES
        lib
        lib64
)

find_package_handle_standard_args(CUDNN
    REQUIRED_VARS
        CUDNN_LIBRARY
        CUDNN_INCLUDE_DIR
)

if(CUDNN_FOUND AND NOT TARGET CUDA::cudnn)
    add_library(CUDA::cudnn UNKNOWN IMPORTED)

    set_target_properties(CUDA::cudnn
        PROPERTIES
            IMPORTED_LOCATION "${CUDNN_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${CUDNN_INCLUDE_DIR}"
    )
endif()

mark_as_advanced(
    CUDNN_INCLUDE_DIR
    CUDNN_LIBRARY
)