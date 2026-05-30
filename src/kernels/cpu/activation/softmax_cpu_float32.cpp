#include <kernels/cpu/activation/softmax_cpu_float32.hpp>

#include <backend/cpu/cpu_parallel.hpp>

#include <cmath>
#include <cstddef>
#include <limits>

namespace kl
{

    namespace
    {

        struct SoftmaxCpuFloat32Task
        {
            float *data;
            std::size_t class_count;

            void operator()(
                std::size_t begin,
                std::size_t end) const
            {
                for (std::size_t n = begin; n < end; ++n)
                {
                    float *row =
                        data + n * class_count;

                    float maximum =
                        -std::numeric_limits<float>::infinity();

                    for (std::size_t c = 0; c < class_count; ++c)
                    {
                        if (row[c] > maximum)
                        {
                            maximum = row[c];
                        }
                    }

                    float sum = 0.0f;

                    for (std::size_t c = 0; c < class_count; ++c)
                    {
                        row[c] =
                            std::exp(
                                row[c] - maximum);

                        sum += row[c];
                    }

                    for (std::size_t c = 0; c < class_count; ++c)
                    {
                        row[c] /= sum;
                    }
                }
            }
        };

    }

    void softmax_cpu_float32(
        Tensor &tensor)
    {
        const std::size_t batch_size =
            tensor.shape()[0];

        const std::size_t class_count =
            tensor.shape()[1];

        float *data =
            static_cast<float *>(
                tensor.data());

        SoftmaxCpuFloat32Task task{
            data,
            class_count};

        cpu_parallel_for(
            0,
            batch_size,
            task);
    }

}