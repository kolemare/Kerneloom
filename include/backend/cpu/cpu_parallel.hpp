#ifndef KL_CPU_PARALLEL_HPP
#define KL_CPU_PARALLEL_HPP

#include <cstddef>
#include <functional>

namespace kl
{

    void cpu_parallel_for(
        std::size_t begin,
        std::size_t end,
        const std::function<void(std::size_t begin, std::size_t end)> &function);

}

#endif // KL_CPU_PARALLEL_HPP