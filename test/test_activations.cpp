#include <backend/backend.hpp>

#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/tensor.hpp>

#include <ops/activation.hpp>

#include <cstdlib>
#include <exception>
#include <iostream>

int main()
{
    try
    {
        kl::Device target = kl::default_device();

        std::cout << "Target device: "
                  << kl::to_string(target.type())
                  << '\n';

        kl::Tensor input_cpu(
            kl::Shape{9},
            kl::DType::Float32,
            kl::Device::cpu(),
            kl::Layout::Unknown,
            kl::Storage::RowMajor);

        std::cout << "Tensor numel is: " << input_cpu.numel();

        float *input_data = static_cast<float *>(input_cpu.data());

        input_data[0] = -4.0f;
        input_data[1] = -2.0f;
        input_data[2] = -1.0f;
        input_data[3] = 0.0f;
        input_data[4] = 1.0f;
        input_data[5] = 2.0f;
        input_data[6] = 4.0f;
        input_data[7] = 8.0f;
        input_data[8] = -8.0f;

        {
            kl::Tensor tensor = input_cpu.to(target);

            kl::activation(tensor, kl::ActivationType::ReLU);

            kl::Tensor result_cpu = tensor.to(kl::Device::cpu());
            const float *result = static_cast<const float *>(result_cpu.data());

            std::cout << "ReLU:    ";

            for (std::size_t i = 0; i < result_cpu.numel(); ++i)
            {
                std::cout << result[i] << ' ';
            }

            std::cout << '\n';
        }

        {
            kl::Tensor tensor = input_cpu.to(target);

            kl::activation(tensor, kl::ActivationType::Sigmoid);

            kl::Tensor result_cpu = tensor.to(kl::Device::cpu());
            const float *result = static_cast<const float *>(result_cpu.data());

            std::cout << "Sigmoid: ";

            for (std::size_t i = 0; i < result_cpu.numel(); ++i)
            {
                std::cout << result[i] << ' ';
            }

            std::cout << '\n';
        }

        {
            kl::Tensor tensor = input_cpu.to(target);

            kl::activation(tensor, kl::ActivationType::Tanh);

            kl::Tensor result_cpu = tensor.to(kl::Device::cpu());
            const float *result = static_cast<const float *>(result_cpu.data());

            std::cout << "Tanh:    ";

            for (std::size_t i = 0; i < result_cpu.numel(); ++i)
            {
                std::cout << result[i] << ' ';
            }

            std::cout << '\n';
        }

        return EXIT_SUCCESS;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }
}