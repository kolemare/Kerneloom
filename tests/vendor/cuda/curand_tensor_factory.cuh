#ifndef KL_TEST_CURAND_TENSOR_FACTORY_HPP
#define KL_TEST_CURAND_TENSOR_FACTORY_HPP

#ifdef KL_ENABLE_CUDA

#include <core/shape.hpp>
#include <core/tensor.hpp>

namespace kl::test
{

    Tensor makeRandomCudaTensorFloat32(
        Shape shape,
        double min_value,
        double max_value,
        unsigned int seed);

}

#endif // KL_ENABLE_CUDA

#endif // KL_TEST_CURAND_TENSOR_FACTORY_HPP