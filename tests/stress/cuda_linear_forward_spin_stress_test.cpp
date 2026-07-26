#include <gtest/gtest.h>

#ifdef KL_ENABLE_CUDA

#include "common/kernel_spin.hpp"
#include "common/tensor_factory.hpp"
#include "common/test_options.hpp"

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/shape.hpp>
#include <core/tensor.hpp>

#include <kernels/cuda/linear/linear_cuda_float32.cuh>

namespace
{

    namespace linear_options =
        kl::test::options::
            linear_forward_float32::
                large_non_square;

}

TEST(
    CudaKernelSpinStress,
    DISABLED_LinearForwardFloat32WithoutBias)
{
    const auto input =
        kl::test::makeRandomTensor(
            kl::Shape{
                linear_options::batch_size,
                linear_options::input_features},
            kl::DType::Float32,
            kl::Device::cuda(),
            -1.0,
            1.0,
            11);

    const auto weights =
        kl::test::makeRandomTensor(
            kl::Shape{
                linear_options::output_features,
                linear_options::input_features},
            kl::DType::Float32,
            kl::Device::cuda(),
            -1.0,
            1.0,
            22);

    kl::Tensor result(
        kl::Shape{
            linear_options::batch_size,
            linear_options::output_features},
        kl::DType::Float32,
        kl::Device::cuda());

    kl::test::spinKernel(
        kl::Device::cuda(),
        "Linear Forward CUDA Float32 Without Bias",
        [&]()
        {
            kl::linear_cuda_float32(
                input,
                weights,
                nullptr,
                result);
        });
}

#endif // KL_ENABLE_CUDA