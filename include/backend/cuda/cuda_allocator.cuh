#ifndef KL_CUDA_ALLOCATOR_HPP
#define KL_CUDA_ALLOCATOR_HPP

#include <core/allocator.hpp>

namespace kl
{

    class CudaAllocator final : public Allocator
    {
    public:
        void *allocate(std::size_t nbytes) override;
        void deallocate(void *ptr) override;
    };

}

#endif // KL_CUDA_ALLOCATOR_HPP