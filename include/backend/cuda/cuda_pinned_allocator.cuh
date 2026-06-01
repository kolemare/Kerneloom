#ifndef KL_CUDA_PINNED_ALLOCATOR_CUH
#define KL_CUDA_PINNED_ALLOCATOR_CUH

#include <core/allocator.hpp>

namespace kl
{

    class CudaPinnedAllocator final
        : public Allocator
    {
    public:
        void *allocate(
            std::size_t nbytes) override;

        void deallocate(
            void *ptr) override;
    };

}

#endif // KL_CUDA_PINNED_ALLOCATOR_CUH