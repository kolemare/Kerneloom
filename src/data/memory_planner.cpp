#include <data/memory_planner.hpp>

#include <backend/backend_memory.hpp>

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>

namespace kl
{

    namespace
    {

        std::size_t available_ram_bytes()
        {
            std::ifstream file(
                "/proc/meminfo");

            if (!file)
            {
                throw std::runtime_error(
                    "failed to open /proc/meminfo");
            }

            std::string key;
            std::size_t value_kib =
                0;

            std::string unit;

            while (file >>
                   key >>
                   value_kib >>
                   unit)
            {
                if (key ==
                    "MemAvailable:")
                {
                    return value_kib *
                           1024ULL;
                }
            }

            throw std::runtime_error(
                "MemAvailable was not found in /proc/meminfo");
        }

        void validate_fraction(
            float value,
            const char *name)
        {
            if (value < 0.0f ||
                value > 1.0f)
            {
                throw std::runtime_error(
                    std::string(name) +
                    " must be between zero and one");
            }
        }

        std::size_t calculate_batch_capacity(
            std::size_t budget_bytes,
            std::size_t batch_bytes,
            std::size_t maximum_batches)
        {
            if (batch_bytes == 0)
            {
                throw std::runtime_error(
                    "memory planner received zero batch size");
            }

            if (maximum_batches == 0)
            {
                throw std::runtime_error(
                    "memory planner maximum batch count must be greater than zero");
            }

            const std::size_t affordable_batches =
                budget_bytes /
                batch_bytes;

            return std::max<std::size_t>(
                1,
                std::min(
                    affordable_batches,
                    maximum_batches));
        }

    }

    MemoryPlan create_memory_plan(
        Device device,
        std::size_t batch_bytes,
        const MemoryPolicy &policy)
    {
        validate_fraction(
            policy
                .decoded_cache_ram_fraction,
            "decoded_cache_ram_fraction");

        validate_fraction(
            policy
                .host_prefetch_ram_fraction,
            "host_prefetch_ram_fraction");

        validate_fraction(
            policy
                .device_prefetch_vram_fraction,
            "device_prefetch_vram_fraction");

        MemoryPlan plan;

        plan.available_ram_bytes =
            available_ram_bytes();

        plan.available_vram_bytes =
            available_device_memory_bytes(
                device);

        plan.batch_bytes =
            batch_bytes;

        plan.decoded_cache_bytes =
            static_cast<std::size_t>(
                static_cast<double>(
                    plan.available_ram_bytes) *
                policy
                    .decoded_cache_ram_fraction);

        const std::size_t host_budget_bytes =
            static_cast<std::size_t>(
                static_cast<double>(
                    plan.available_ram_bytes) *
                policy
                    .host_prefetch_ram_fraction);

        plan.host_prefetch_batches =
            calculate_batch_capacity(
                host_budget_bytes,
                batch_bytes,
                policy
                    .max_host_prefetch_batches);

        if (device.type() ==
            DeviceType::CPU)
        {
            plan.device_prefetch_batches =
                0;

            return plan;
        }

        const std::size_t device_budget_bytes =
            static_cast<std::size_t>(
                static_cast<double>(
                    plan.available_vram_bytes) *
                policy
                    .device_prefetch_vram_fraction);

        plan.device_prefetch_batches =
            calculate_batch_capacity(
                device_budget_bytes,
                batch_bytes,
                policy
                    .max_device_prefetch_batches);

        return plan;
    }

}