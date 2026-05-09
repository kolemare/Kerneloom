#include <kernels/cpu/activation/relu_cpu_float32.hpp>

#include <backend/cpu/cpu_parallel.hpp>

#include <cstddef>

namespace kl
{

    namespace
    {

        struct ReLUCpuFloat32Task
        {
            float *data;

            void operator()(std::size_t begin, std::size_t end) const
            {
                for (std::size_t i = begin; i < end; ++i)
                {
                    if (data[i] < 0.0f)
                    {
                        data[i] = 0.0f;
                    }
                }
            }
        };

    }

    void relu_cpu_float32(Tensor &tensor)
    {
        float *data = static_cast<float *>(tensor.data());

        ReLUCpuFloat32Task task{data};

        cpu_parallel_for(0, tensor.numel(), task);
    }

}