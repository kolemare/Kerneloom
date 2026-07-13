#include <backend/backend.hpp>

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/tensor.hpp>

#include <ops/linear.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>

namespace
{

    void fill_input(kl::Tensor &tensor)
    {
        float *data = static_cast<float *>(tensor.data());

        data[0] = 1.0f;
        data[1] = 2.0f;
        data[2] = 3.0f;

        data[3] = 4.0f;
        data[4] = 5.0f;
        data[5] = 6.0f;
    }

    void fill_weights(kl::Tensor &tensor)
    {
        float *data = static_cast<float *>(tensor.data());

        data[0] = 1.0f;
        data[1] = 1.0f;
        data[2] = 1.0f;

        data[3] = 2.0f;
        data[4] = 2.0f;
        data[5] = 2.0f;

        data[6] = 1.0f;
        data[7] = 0.0f;
        data[8] = 1.0f;

        data[9] = 0.0f;
        data[10] = 1.0f;
        data[11] = 0.0f;
    }

    void fill_bias(kl::Tensor &tensor)
    {
        float *data = static_cast<float *>(tensor.data());

        data[0] = 1.0f;
        data[1] = 2.0f;
        data[2] = 3.0f;
        data[3] = 4.0f;
    }

    bool close_enough(float actual, float expected)
    {
        return std::fabs(actual - expected) < 1.0e-5f;
    }

}

int main()
{
    const kl::Device target = kl::default_device();

    kl::Tensor input_cpu(
        kl::Shape{2, 3},
        kl::DType::Float32,
        kl::Device::cpu(),
        kl::Layout::Unknown,
        kl::Storage::RowMajor);

    kl::Tensor weights_cpu(
        kl::Shape{4, 3},
        kl::DType::Float32,
        kl::Device::cpu(),
        kl::Layout::Unknown,
        kl::Storage::RowMajor);

    kl::Tensor bias_cpu(
        kl::Shape{4},
        kl::DType::Float32,
        kl::Device::cpu(),
        kl::Layout::Unknown,
        kl::Storage::RowMajor);

    fill_input(input_cpu);
    fill_weights(weights_cpu);
    fill_bias(bias_cpu);

    kl::Tensor input = input_cpu.to(target);
    kl::Tensor weights = weights_cpu.to(target);
    kl::Tensor bias = bias_cpu.to(target);

    kl::Tensor result(
        kl::Shape{2, 4},
        kl::DType::Float32,
        target,
        kl::Layout::Unknown,
        kl::Storage::RowMajor);

    kl::linear(
        input,
        weights,
        &bias,
        result);

    kl::Tensor result_cpu = result.to(kl::Device::cpu());

    const float *data = static_cast<const float *>(result_cpu.data());

    const float expected[8] = {
        7.0f, 14.0f, 7.0f, 6.0f,
        16.0f, 32.0f, 13.0f, 9.0f};

    bool passed = true;

    for (std::size_t i = 0; i < 8; ++i)
    {
        if (!close_enough(data[i], expected[i]))
        {
            passed = false;
        }
    }

    std::cout << "target=" << kl::to_string(target.type()) << '\n';

    for (std::size_t n = 0; n < 2; ++n)
    {
        for (std::size_t out = 0; out < 4; ++out)
        {
            std::cout << data[n * 4 + out] << ' ';
        }

        std::cout << '\n';
    }

    if (!passed)
    {
        std::cout << "linear test failed\n";
        return EXIT_FAILURE;
    }

    std::cout << "linear test passed\n";

    return EXIT_SUCCESS;
}