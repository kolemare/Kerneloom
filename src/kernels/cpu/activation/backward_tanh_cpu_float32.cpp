#include <kernels/cpu/activation/backward_tanh_cpu_float32.hpp>

#include <backend/cpu/cpu_parallel.hpp>

#include <cstddef>

namespace kl
{

    namespace
    {

        struct BackwardTanhCpuFloat32Task
        {
            const float *activation_output;
            float *grad;

            void operator()(std::size_t begin, std::size_t end) const
            {
                for (std::size_t i = begin; i < end; ++i)
                {
                    const float y = activation_output[i];
                    grad[i] *= 1.0f - y * y;
                }
            }
        };

    }

    void backward_tanh_cpu_float32(
        const Tensor &activation_output,
        Tensor &grad)
    {
        const float *activation_output_data =
            static_cast<const float *>(activation_output.data());

        float *grad_data =
            static_cast<float *>(grad.data());

        BackwardTanhCpuFloat32Task task{
            activation_output_data,
            grad_data};

        cpu_parallel_for(
            0,
            grad.numel(),
            task);
    }

}