#ifndef KL_DATA_LOADER_MEMORY_BUDGET_HPP
#define KL_DATA_LOADER_MEMORY_BUDGET_HPP

#include <data/memory_plan.hpp>
#include <data/memory_policy.hpp>

#include <core/device.hpp>

#include <cstddef>
#include <mutex>

namespace kl
{

    struct DataLoaderMemoryReservation
    {
        MemoryPlan plan;

        std::size_t decoded_cache_bytes =
            0;

        std::size_t host_storage_bytes =
            0;

        std::size_t device_storage_bytes =
            0;
    };

    class DataLoaderMemoryBudget
    {
    public:
        DataLoaderMemoryBudget(
            Device device,
            const MemoryPolicy &policy = {},
            std::size_t expected_loader_count = 1);

        DataLoaderMemoryReservation reserve_auto(
            std::size_t batch_bytes,
            std::size_t loader_workers,
            bool uses_device);

        DataLoaderMemoryReservation reserve_manual(
            std::size_t batch_bytes,
            std::size_t decoded_cache_bytes,
            std::size_t host_prefetch_batches,
            std::size_t device_prefetch_batches,
            std::size_t loader_workers,
            bool uses_device);

        void release(
            const DataLoaderMemoryReservation &reservation) noexcept;

        Device device() const;

        std::size_t available_ram_bytes() const;
        std::size_t available_vram_bytes() const;

        std::size_t decoded_cache_capacity_bytes() const;
        std::size_t host_storage_capacity_bytes() const;
        std::size_t device_storage_capacity_bytes() const;

        std::size_t reserved_decoded_cache_bytes() const;
        std::size_t reserved_host_storage_bytes() const;
        std::size_t reserved_device_storage_bytes() const;

    private:
        std::size_t remaining_decoded_cache_bytes() const;
        std::size_t remaining_host_storage_bytes() const;
        std::size_t remaining_device_storage_bytes() const;

        std::size_t fair_decoded_cache_budget_bytes() const;
        std::size_t fair_host_storage_budget_bytes() const;
        std::size_t fair_device_storage_budget_bytes() const;

        std::size_t calculate_auto_prefetch_batches(
            std::size_t budget_bytes,
            std::size_t batch_bytes,
            std::size_t extra_batches,
            std::size_t maximum_prefetch_batches,
            const char *name) const;

        static std::size_t checked_batch_bytes(
            std::size_t batch_count,
            std::size_t batch_bytes,
            const char *name);

        static void validate_policy(
            const MemoryPolicy &policy);

        static void validate_fraction(
            float value,
            const char *name);

    private:
        Device device_;
        MemoryPolicy policy_;

        std::size_t expected_loader_count_ =
            1;

        std::size_t available_ram_bytes_ =
            0;

        std::size_t available_vram_bytes_ =
            0;

        std::size_t decoded_cache_capacity_bytes_ =
            0;

        std::size_t host_storage_capacity_bytes_ =
            0;

        std::size_t device_storage_capacity_bytes_ =
            0;

        mutable std::mutex mutex_;

        std::size_t reserved_decoded_cache_bytes_ =
            0;

        std::size_t reserved_host_storage_bytes_ =
            0;

        std::size_t reserved_device_storage_bytes_ =
            0;
    };

}

#endif // KL_DATA_LOADER_MEMORY_BUDGET_HPP