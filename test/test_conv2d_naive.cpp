#include <backend/backend.hpp>
#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/tensor.hpp>
#include <ops/conv2d_naive.hpp>

#include <cstdlib>
#include <exception>
#include <iostream>
#include <thread>
#include <chrono>

int main()
{
    try
    {
        size_t batch_size = 32;    // N
        size_t input_channels = 3; // C
        size_t input_h = 256;      // H
        size_t input_w = 256;      // W

        size_t k_filters = 32;
        size_t k_channels = 3;
        size_t k_h = 3;
        size_t k_w = 3;

        kl::Device target = kl::default_device();

        kl::Tensor input(kl::Shape{batch_size, input_channels, input_h, input_w}, kl::DType::Float32, kl::Device::cpu());
        kl::Tensor kernels(kl::Shape{k_filters, k_channels, k_h, k_w}, kl::DType::Float32, kl::Device::cpu());

        float *input_cpu = static_cast<float *>(input.data());
        float *kernels_cpu = static_cast<float *>(kernels.data());

        for (int i = 0; i < batch_size * input_h * input_w * input_channels; i++)
        {
            input_cpu[i] = 1;
        }

        for (int i = 0; i < k_filters * k_channels * k_h * k_w; i++)
        {
            kernels_cpu[i] = 2;
        }

        kl::Tensor input_target = input.to(target);
        kl::Tensor kernels_target = kernels.to(target);

        kl::Tensor result = kl::conv2d_naive(input_target, kernels_target);

        kl::Tensor cpu_result = result.to(kl::Device::cpu());

        const float *result_raw = static_cast<const float *>(cpu_result.data());

        std::cout << result_raw[346] << ": " << result_raw[34] << ": " << result_raw[262] << std::endl;

        return EXIT_SUCCESS;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }
}