#include <kernels/cpu/optimizers/sgd_update_cpu_float32.hpp>

#include <backend/cpu/cpu_parallel.hpp>

#include <cstddef>

namespace kl
{

    namespace
    {

        struct SgdUpdateCpuFloat32Task
        {
            float *value;
            const float *grad;
            float learning_rate;

            void operator()(std::size_t begin, std::size_t end) const
            {
                for (std::size_t i = begin; i < end; ++i)
                {
                    value[i] -= learning_rate * grad[i];
                }
            }
        };

    }

    void sgd_update_cpu_float32(
        Tensor &value,
        const Tensor &grad,
        float learning_rate)
    {
        float *value_data =
            static_cast<float *>(value.data());

        const float *grad_data =
            static_cast<const float *>(grad.data());

        SgdUpdateCpuFloat32Task task{
            value_data,
            grad_data,
            learning_rate};

        cpu_parallel_for(
            0,
            value.numel(),
            task);
    }

}