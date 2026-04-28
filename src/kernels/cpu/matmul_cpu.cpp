#include <kernels/cpu/matmul_cpu.hpp>

namespace kl
{

    void matmul_cpu_float32(const Tensor &a, const Tensor &b, Tensor &c)
    {
        const std::size_t m = a.shape()[0];
        const std::size_t k = a.shape()[1];
        const std::size_t n = b.shape()[1];

        const float *a_data = static_cast<const float *>(a.data());
        const float *b_data = static_cast<const float *>(b.data());
        float *c_data = static_cast<float *>(c.data());

        for (std::size_t i = 0; i < m; ++i)
        {
            for (std::size_t j = 0; j < n; ++j)
            {
                float sum = 0.0f;

                for (std::size_t p = 0; p < k; ++p)
                {
                    sum += a_data[i * k + p] * b_data[p * n + j];
                }

                c_data[i * n + j] = sum;
            }
        }
    }

}