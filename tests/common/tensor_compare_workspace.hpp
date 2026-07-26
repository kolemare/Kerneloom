#ifndef KL_TEST_TENSOR_COMPARE_WORKSPACE_HPP
#define KL_TEST_TENSOR_COMPARE_WORKSPACE_HPP

#include <core/buffer.hpp>
#include <core/device.hpp>
#include <core/memory_type.hpp>

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace kl::test
{

    class TensorCompareWorkspace
    {
    public:
        static constexpr std::size_t default_chunk_bytes =
            64ULL * 1024ULL * 1024ULL;

        TensorCompareWorkspace() = default;

        TensorCompareWorkspace(
            const TensorCompareWorkspace &) = delete;

        TensorCompareWorkspace &operator=(
            const TensorCompareWorkspace &) = delete;

        TensorCompareWorkspace(
            TensorCompareWorkspace &&) = delete;

        TensorCompareWorkspace &operator=(
            TensorCompareWorkspace &&) = delete;

        void prepare(
            DeviceType backend,
            std::size_t required_capacity_bytes =
                default_chunk_bytes)
        {
            if (required_capacity_bytes == 0)
            {
                throw std::runtime_error(
                    "TensorCompareWorkspace capacity must be greater than zero");
            }

            const MemoryType required_memory_type =
                pinned_memory_type(backend);

            if (capacity_bytes_ >= required_capacity_bytes &&
                memory_type_ == required_memory_type &&
                !expected_buffer_.empty() &&
                !actual_buffer_.empty())
            {
                return;
            }

            Buffer new_expected_buffer(
                required_capacity_bytes,
                Device::cpu(),
                required_memory_type);

            Buffer new_actual_buffer(
                required_capacity_bytes,
                Device::cpu(),
                required_memory_type);

            expected_buffer_ =
                std::move(new_expected_buffer);

            actual_buffer_ =
                std::move(new_actual_buffer);

            capacity_bytes_ =
                required_capacity_bytes;

            memory_type_ =
                required_memory_type;
        }

        [[nodiscard]] void *expected_data()
        {
            return expected_buffer_.data();
        }

        [[nodiscard]] void *actual_data()
        {
            return actual_buffer_.data();
        }

        [[nodiscard]] std::size_t capacity_bytes() const
        {
            return capacity_bytes_;
        }

        [[nodiscard]] MemoryType memory_type() const
        {
            return memory_type_;
        }

    private:
        static MemoryType pinned_memory_type(
            DeviceType backend)
        {
            switch (backend)
            {
            case DeviceType::CUDA:
                return MemoryType::CudaPinnedHost;

            case DeviceType::ROCM:
                return MemoryType::RocmPinnedHost;

            default:
                throw std::runtime_error(
                    "TensorCompareWorkspace requires CUDA or ROCm backend");
            }
        }

    private:
        Buffer expected_buffer_;
        Buffer actual_buffer_;

        std::size_t capacity_bytes_ = 0;

        MemoryType memory_type_ =
            MemoryType::Default;
    };

    inline TensorCompareWorkspace &
    tensorCompareWorkspace()
    {
        thread_local TensorCompareWorkspace workspace;

        return workspace;
    }

} // namespace kl::test

#endif // KL_TEST_TENSOR_COMPARE_WORKSPACE_HPP