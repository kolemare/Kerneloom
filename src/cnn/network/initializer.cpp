#include <cnn/network/initializer.hpp>

#include <core/copy.hpp>
#include <core/device.hpp>
#include <core/dtype.hpp>

#include <cmath>
#include <random>
#include <stdexcept>

namespace kl
{

    void Initializer::initialize(
        Tensor &tensor,
        const InitializerOptions &options)
    {
        if (tensor.dtype() != DType::Float32)
        {
            throw std::runtime_error("Initializer currently supports only Float32 tensors");
        }

        if (tensor.device().type() == DeviceType::CPU)
        {
            initialize_cpu(tensor, options);
            return;
        }

        Tensor cpu_tensor(
            tensor.shape(),
            tensor.dtype(),
            Device::cpu(),
            tensor.layout(),
            tensor.storage());

        initialize_cpu(cpu_tensor, options);

        copy(tensor, cpu_tensor);
    }

    void Initializer::initialize(
        Tensor &tensor,
        InitializerType type)
    {
        InitializerOptions options;
        options.type = type;

        initialize(tensor, options);
    }

    void Initializer::initialize(
        Tensor &tensor,
        InitializerType type,
        float value)
    {
        InitializerOptions options;
        options.type = type;
        options.value = value;

        initialize(tensor, options);
    }

    void Initializer::initialize_cpu(
        Tensor &tensor,
        const InitializerOptions &options)
    {
        switch (options.type)
        {
        case InitializerType::Zeros:
            fill_constant_cpu(tensor, 0.0f);
            return;

        case InitializerType::Ones:
            fill_constant_cpu(tensor, 1.0f);
            return;

        case InitializerType::Constant:
            fill_constant_cpu(tensor, options.value);
            return;

        case InitializerType::XavierUniform:
        {
            const float in = fan_in(tensor);
            const float out = fan_out(tensor);
            const float limit = std::sqrt(6.0f / (in + out));

            fill_uniform_cpu(
                tensor,
                -limit,
                limit,
                options.seed);

            return;
        }

        case InitializerType::KaimingUniform:
        {
            const float in = fan_in(tensor);
            const float limit = std::sqrt(6.0f / in);

            fill_uniform_cpu(
                tensor,
                -limit,
                limit,
                options.seed);

            return;
        }

        default:
            throw std::runtime_error("unknown InitializerType");
        }
    }

    void Initializer::fill_constant_cpu(
        Tensor &tensor,
        float value)
    {
        float *data = static_cast<float *>(tensor.data());

        for (std::size_t i = 0; i < tensor.numel(); ++i)
        {
            data[i] = value;
        }
    }

    void Initializer::fill_uniform_cpu(
        Tensor &tensor,
        float min_value,
        float max_value,
        std::uint32_t seed)
    {
        float *data = static_cast<float *>(tensor.data());

        std::mt19937 generator(seed);
        std::uniform_real_distribution<float> distribution(
            min_value,
            max_value);

        for (std::size_t i = 0; i < tensor.numel(); ++i)
        {
            data[i] = distribution(generator);
        }
    }

    float Initializer::fan_in(
        const Tensor &tensor)
    {
        if (tensor.rank() == 2)
        {
            return static_cast<float>(tensor.shape()[1]);
        }

        if (tensor.rank() == 4)
        {
            return static_cast<float>(
                tensor.shape()[1] *
                tensor.shape()[2] *
                tensor.shape()[3]);
        }

        return static_cast<float>(tensor.numel());
    }

    float Initializer::fan_out(
        const Tensor &tensor)
    {
        if (tensor.rank() == 2)
        {
            return static_cast<float>(tensor.shape()[0]);
        }

        if (tensor.rank() == 4)
        {
            return static_cast<float>(
                tensor.shape()[0] *
                tensor.shape()[2] *
                tensor.shape()[3]);
        }

        return static_cast<float>(tensor.numel());
    }

}