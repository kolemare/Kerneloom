#include <backend/cpu/cpu_copy.hpp>

#include <cstring>

namespace kl
{

    void cpu_copy(void *dst, const void *src, std::size_t nbytes)
    {
        if (nbytes == 0)
        {
            return;
        }

        std::memcpy(dst, src, nbytes);
    }

}