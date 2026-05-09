#include <kernels/cpu/activation/sigmoid_cpu_float32.hpp>

#include <backend/cpu/cpu_parallel.hpp>

#include <cmath>
#include <cstddef>

namespace kl
{

    namespace
    {

        struct SigmoidCpuFloat32Task
        {
            float *data;

            void operator()(std::size_t begin, std::size_t end) const
            {
                for (std::size_t i = begin; i < end; ++i)
                {
                    data[i] = 1.0f / (1.0f + std::exp(-data[i]));
                }
            }
        };

    }

    void sigmoid_cpu_float32(Tensor &tensor)
    {
        float *data = static_cast<float *>(tensor.data());

        SigmoidCpuFloat32Task task{data};

        cpu_parallel_for(0, tensor.numel(), task);
    }

}