#include <backend/backend.hpp>

#include <cnn/layers/avgpool2d_layer.hpp>
#include <cnn/layers/maxpool2d_layer.hpp>
#include <cnn/options/pooling2d_options.hpp>

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/tensor.hpp>
#include <core/tensor_pool.hpp>

#include <cmath>
#include <cstdlib>
#include <iostream>

namespace
{

    void fill_input(kl::Tensor &tensor)
    {
        float *data = static_cast<float *>(tensor.data());

        for (std::size_t i = 0; i < tensor.numel(); ++i)
        {
            data[i] = static_cast<float>(i + 1);
        }
    }

    void fill_tensor(kl::Tensor &tensor, float value)
    {
        float *data = static_cast<float *>(tensor.data());

        for (std::size_t i = 0; i < tensor.numel(); ++i)
        {
            data[i] = value;
        }
    }

    bool close_enough(float actual, float expected)
    {
        return std::fabs(actual - expected) < 1.0e-5f;
    }

    bool check_tensor(
        const kl::Tensor &tensor,
        const float *expected)
    {
        const float *data = static_cast<const float *>(tensor.data());

        for (std::size_t i = 0; i < tensor.numel(); ++i)
        {
            if (!close_enough(data[i], expected[i]))
            {
                std::cout << "mismatch at " << i
                          << " | actual=" << data[i]
                          << " | expected=" << expected[i]
                          << '\n';

                return false;
            }
        }

        return true;
    }

    bool test_maxpool2d_layer(kl::Device target)
    {
        kl::Pooling2dOptions options;
        options.kernel_h = 2;
        options.kernel_w = 2;
        options.stride_h = 2;
        options.stride_w = 2;
        options.padding_h = 0;
        options.padding_w = 0;

        kl::Tensor input_cpu(
            kl::Shape{1, 1, 4, 4},
            kl::DType::Float32,
            kl::Device::cpu(),
            kl::Layout::NCHW,
            kl::Storage::RowMajor);

        fill_input(input_cpu);

        kl::Tensor input = input_cpu.to(target);

        kl::TensorPool pool;
        kl::MaxPool2dLayer layer(options);

        layer.prepareTraining();

        kl::Tensor &output = layer.forward(input, pool);

        kl::Tensor grad_output_cpu(
            output.shape(),
            output.dtype(),
            kl::Device::cpu(),
            output.layout(),
            output.storage());

        fill_tensor(grad_output_cpu, 1.0f);

        kl::Tensor grad_output = grad_output_cpu.to(target);

        kl::Tensor &grad_input = layer.backward(grad_output, pool);

        kl::Tensor output_cpu = output.to(kl::Device::cpu());
        kl::Tensor grad_input_cpu = grad_input.to(kl::Device::cpu());

        const float expected_output[4] = {
            6.0f, 8.0f,
            14.0f, 16.0f};

        const float expected_grad_input[16] = {
            0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 1.0f,
            0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 1.0f};

        if (!check_tensor(output_cpu, expected_output))
        {
            std::cout << "maxpool forward failed\n";
            return false;
        }

        if (!check_tensor(grad_input_cpu, expected_grad_input))
        {
            std::cout << "maxpool backward failed\n";
            return false;
        }

        return true;
    }

    bool test_avgpool2d_layer(kl::Device target)
    {
        kl::Pooling2dOptions options;
        options.kernel_h = 2;
        options.kernel_w = 2;
        options.stride_h = 2;
        options.stride_w = 2;
        options.padding_h = 0;
        options.padding_w = 0;

        kl::Tensor input_cpu(
            kl::Shape{1, 1, 4, 4},
            kl::DType::Float32,
            kl::Device::cpu(),
            kl::Layout::NCHW,
            kl::Storage::RowMajor);

        fill_input(input_cpu);

        kl::Tensor input = input_cpu.to(target);

        kl::TensorPool pool;
        kl::AvgPool2dLayer layer(options);

        layer.prepareTraining();

        kl::Tensor &output = layer.forward(input, pool);

        kl::Tensor grad_output_cpu(
            output.shape(),
            output.dtype(),
            kl::Device::cpu(),
            output.layout(),
            output.storage());

        fill_tensor(grad_output_cpu, 1.0f);

        kl::Tensor grad_output = grad_output_cpu.to(target);

        kl::Tensor &grad_input = layer.backward(grad_output, pool);

        kl::Tensor output_cpu = output.to(kl::Device::cpu());
        kl::Tensor grad_input_cpu = grad_input.to(kl::Device::cpu());

        const float expected_output[4] = {
            3.5f, 5.5f,
            11.5f, 13.5f};

        const float expected_grad_input[16] = {
            0.25f, 0.25f, 0.25f, 0.25f,
            0.25f, 0.25f, 0.25f, 0.25f,
            0.25f, 0.25f, 0.25f, 0.25f,
            0.25f, 0.25f, 0.25f, 0.25f};

        if (!check_tensor(output_cpu, expected_output))
        {
            std::cout << "avgpool forward failed\n";
            return false;
        }

        if (!check_tensor(grad_input_cpu, expected_grad_input))
        {
            std::cout << "avgpool backward failed\n";
            return false;
        }

        return true;
    }

}

int main()
{
    const kl::Device target = kl::default_device();

    bool passed = true;

    if (!test_maxpool2d_layer(target))
    {
        passed = false;
    }

    if (!test_avgpool2d_layer(target))
    {
        passed = false;
    }

    if (!passed)
    {
        return EXIT_FAILURE;
    }

    std::cout << "pooling layer forward/backward test passed on "
              << kl::to_string(target.type())
              << '\n';

    return EXIT_SUCCESS;
}