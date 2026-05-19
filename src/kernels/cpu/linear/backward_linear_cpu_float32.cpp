#include <kernels/cpu/linear/backward_linear_cpu_float32.hpp>

#include <backend/cpu/cpu_parallel.hpp>

#include <cstddef>

namespace kl
{

    namespace
    {

        struct GradInputTask
        {
            const float *grad_output;
            const float *weights;
            float *grad_input;

            std::size_t input_features;
            std::size_t output_features;

            void operator()(std::size_t begin, std::size_t end) const
            {
                for (std::size_t index = begin; index < end; ++index)
                {
                    const std::size_t n = index / input_features;
                    const std::size_t in = index % input_features;

                    float sum = 0.0f;

                    for (std::size_t out = 0; out < output_features; ++out)
                    {
                        sum += grad_output[n * output_features + out] *
                               weights[out * input_features + in];
                    }

                    grad_input[index] = sum;
                }
            }
        };

        struct GradWeightsTask
        {
            const float *input;
            const float *grad_output;
            float *grad_weights;

            std::size_t batch_size;
            std::size_t input_features;
            std::size_t output_features;

            void operator()(std::size_t begin, std::size_t end) const
            {
                for (std::size_t index = begin; index < end; ++index)
                {
                    const std::size_t out = index / input_features;
                    const std::size_t in = index % input_features;

                    float sum = 0.0f;

                    for (std::size_t n = 0; n < batch_size; ++n)
                    {
                        sum += grad_output[n * output_features + out] *
                               input[n * input_features + in];
                    }

                    grad_weights[index] = sum;
                }
            }
        };

        struct GradBiasTask
        {
            const float *grad_output;
            float *grad_bias;

            std::size_t batch_size;
            std::size_t output_features;

            void operator()(std::size_t begin, std::size_t end) const
            {
                for (std::size_t out = begin; out < end; ++out)
                {
                    float sum = 0.0f;

                    for (std::size_t n = 0; n < batch_size; ++n)
                    {
                        sum += grad_output[n * output_features + out];
                    }

                    grad_bias[out] = sum;
                }
            }
        };

    }

    void backward_linear_cpu_float32(
        const Tensor &input,
        const Tensor &weights,
        const Tensor &grad_output,
        Tensor &grad_input,
        Tensor &grad_weights,
        Tensor *grad_bias)
    {
        const std::size_t batch_size = input.shape()[0];
        const std::size_t input_features = input.shape()[1];
        const std::size_t output_features = weights.shape()[0];

        const float *input_data =
            static_cast<const float *>(input.data());

        const float *weights_data =
            static_cast<const float *>(weights.data());

        const float *grad_output_data =
            static_cast<const float *>(grad_output.data());

        float *grad_input_data =
            static_cast<float *>(grad_input.data());

        float *grad_weights_data =
            static_cast<float *>(grad_weights.data());

        GradInputTask grad_input_task{
            grad_output_data,
            weights_data,
            grad_input_data,
            input_features,
            output_features};

        cpu_parallel_for(
            0,
            batch_size * input_features,
            grad_input_task);

        GradWeightsTask grad_weights_task{
            input_data,
            grad_output_data,
            grad_weights_data,
            batch_size,
            input_features,
            output_features};

        cpu_parallel_for(
            0,
            output_features * input_features,
            grad_weights_task);

        if (grad_bias != nullptr)
        {
            float *grad_bias_data =
                static_cast<float *>(grad_bias->data());

            GradBiasTask grad_bias_task{
                grad_output_data,
                grad_bias_data,
                batch_size,
                output_features};

            cpu_parallel_for(
                0,
                output_features,
                grad_bias_task);
        }
    }

}