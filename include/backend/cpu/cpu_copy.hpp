#ifndef KL_CPU_COPY_HPP
#define KL_CPU_COPY_HPP

#include <cstddef>

namespace kl
{

    void cpu_copy(void *dst, const void *src, std::size_t nbytes);

}

#endif // KL_CPU_COPY_HPP