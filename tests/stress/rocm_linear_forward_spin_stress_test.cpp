#include <gtest/gtest.h>

#ifdef KL_ENABLE_ROCM

#include "common/kernel_spin.hpp"
#include "common/tensor_factory.hpp"
#include "common/test_options.hpp"

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/shape.hpp>
#include <core/tensor.hpp>

#include <kernels/rocm/linear/linear_rocm_float32.hiph>

namespace
{

    namespace linear_options =
        kl::test::options::
            linear_forward_float32::
                large_non_square;

}

TEST(
    RocmKernelSpinStress,
    DISABLED_LinearForwardFloat32WithoutBias)
{
    const auto input =
        kl::test::makeRandomTensor(
            kl::Shape{
                linear_options::batch_size,
                linear_options::input_features},
            kl::DType::Float32,
            kl::Device::rocm(),
            -1.0,
            1.0,
            11);

    const auto weights =
        kl::test::makeRandomTensor(
            kl::Shape{
                linear_options::output_features,
                linear_options::input_features},
            kl::DType::Float32,
            kl::Device::rocm(),
            -1.0,
            1.0,
            22);

    kl::Tensor result(
        kl::Shape{
            linear_options::batch_size,
            linear_options::output_features},
        kl::DType::Float32,
        kl::Device::rocm());

    kl::test::spinKernel(
        kl::Device::rocm(),
        "Linear Forward ROCm Float32 Without Bias",
        [&]()
        {
            kl::linear_rocm_float32(
                input,
                weights,
                nullptr,
                result);
        });
}

#endif // KL_ENABLE_ROCM