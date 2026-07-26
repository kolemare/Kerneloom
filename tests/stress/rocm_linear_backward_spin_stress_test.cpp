#include <gtest/gtest.h>

#ifdef KL_ENABLE_ROCM

#include "common/kernel_spin.hpp"
#include "common/tensor_factory.hpp"
#include "common/test_options.hpp"

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/shape.hpp>
#include <core/tensor.hpp>

#include <kernels/rocm/linear/backward_linear_grad_bias_rocm_float32.hiph>
#include <kernels/rocm/linear/backward_linear_grad_input_rocm_float32.hiph>
#include <kernels/rocm/linear/backward_linear_grad_weights_rocm_float32.hiph>

namespace
{

    namespace linear_options =
        kl::test::options::
            linear_backward_float32::
                large_non_square;

    namespace grad_bias_options =
        kl::test::options::
            linear_backward_float32::
                grad_bias_large;

}

TEST(
    RocmKernelSpinStress,
    DISABLED_LinearBackwardGradInputRocmFloat32)
{
    const auto weights =
        kl::test::makeRandomTensor(
            kl::Shape{
                linear_options::output_features,
                linear_options::input_features},
            kl::DType::Float32,
            kl::Device::rocm(),
            -1.0,
            1.0,
            11);

    const auto grad_output =
        kl::test::makeRandomTensor(
            kl::Shape{
                linear_options::batch_size,
                linear_options::output_features},
            kl::DType::Float32,
            kl::Device::rocm(),
            -1.0,
            1.0,
            22);

    kl::Tensor grad_input(
        kl::Shape{
            linear_options::batch_size,
            linear_options::input_features},
        kl::DType::Float32,
        kl::Device::rocm());

    kl::test::spinKernel(
        kl::Device::rocm(),
        "Linear Backward Grad Input ROCm Float32",
        [&]()
        {
            kl::backward_linear_grad_input_rocm_float32(
                weights,
                grad_output,
                grad_input);
        });
}

TEST(
    RocmKernelSpinStress,
    DISABLED_LinearBackwardGradWeightsRocmFloat32)
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

    const auto grad_output =
        kl::test::makeRandomTensor(
            kl::Shape{
                linear_options::batch_size,
                linear_options::output_features},
            kl::DType::Float32,
            kl::Device::rocm(),
            -1.0,
            1.0,
            22);

    kl::Tensor grad_weights(
        kl::Shape{
            linear_options::output_features,
            linear_options::input_features},
        kl::DType::Float32,
        kl::Device::rocm());

    kl::test::spinKernel(
        kl::Device::rocm(),
        "Linear Backward Grad Weights ROCm Float32",
        [&]()
        {
            kl::backward_linear_grad_weights_rocm_float32(
                input,
                grad_output,
                grad_weights);
        });
}

TEST(
    RocmKernelSpinStress,
    DISABLED_LinearBackwardGradBiasRocmFloat32)
{
    const auto grad_output =
        kl::test::makeRandomTensor(
            kl::Shape{
                grad_bias_options::batch_size,
                grad_bias_options::output_features},
            kl::DType::Float32,
            kl::Device::rocm(),
            -1.0,
            1.0,
            11);

    kl::Tensor grad_bias(
        kl::Shape{
            grad_bias_options::output_features},
        kl::DType::Float32,
        kl::Device::rocm());

    kl::test::spinKernel(
        kl::Device::rocm(),
        "Linear Backward Grad Bias ROCm Float32",
        [&]()
        {
            kl::backward_linear_grad_bias_rocm_float32(
                grad_output,
                grad_bias);
        });
}

#endif // KL_ENABLE_ROCM