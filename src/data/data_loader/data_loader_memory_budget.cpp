#include <data/data_loader/data_loader_memory_budget.hpp>

#include <backend/backend_memory.hpp>

#include <algorithm>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace kl
{

    namespace data_loader_memory_budget_internal
    {

        std::size_t detect_available_ram_bytes()
        {
            std::ifstream file(
                "/proc/meminfo");

            if (!file)
            {
                throw std::runtime_error(
                    "failed to open /proc/meminfo");
            }

            std::string line;

            while (std::getline(file, line))
            {
                std::istringstream stream(
                    line);

                std::string key;

                std::size_t value_kib =
                    0;

                std::string unit;

                stream >>
                    key >>
                    value_kib >>
                    unit;

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

    }

    DataLoaderMemoryBudget::DataLoaderMemoryBudget(
        Device device,
        const MemoryPolicy &policy,
        std::size_t expected_loader_count)
        : device_(
              device),
          policy_(
              policy),
          expected_loader_count_(
              expected_loader_count)
    {
        validate_policy(
            policy_);

        if (expected_loader_count_ == 0)
        {
            throw std::runtime_error(
                "DataLoaderMemoryBudget expected loader count must be greater than zero");
        }

        available_ram_bytes_ =
            data_loader_memory_budget_internal::detect_available_ram_bytes();

        available_vram_bytes_ =
            available_device_memory_bytes(
                device_);

        decoded_cache_capacity_bytes_ =
            static_cast<std::size_t>(
                static_cast<double>(
                    available_ram_bytes_) *
                policy_
                    .decoded_cache_ram_fraction);

        host_storage_capacity_bytes_ =
            static_cast<std::size_t>(
                static_cast<double>(
                    available_ram_bytes_) *
                policy_
                    .host_prefetch_ram_fraction);

        if (device_.type() ==
            DeviceType::CPU)
        {
            device_storage_capacity_bytes_ =
                0;
        }
        else
        {
            device_storage_capacity_bytes_ =
                static_cast<std::size_t>(
                    static_cast<double>(
                        available_vram_bytes_) *
                    policy_
                        .device_prefetch_vram_fraction);
        }
    }

    DataLoaderMemoryReservation
    DataLoaderMemoryBudget::reserve_auto(
        std::size_t batch_bytes,
        std::size_t loader_workers,
        bool uses_device)
    {
        if (batch_bytes == 0)
        {
            throw std::runtime_error(
                "DataLoaderMemoryBudget received zero batch size");
        }

        if (loader_workers == 0)
        {
            throw std::runtime_error(
                "DataLoaderMemoryBudget received zero loader workers");
        }

        std::lock_guard<std::mutex> lock(
            mutex_);

        DataLoaderMemoryReservation reservation;

        reservation.plan.available_ram_bytes =
            available_ram_bytes_;

        reservation.plan.available_vram_bytes =
            available_vram_bytes_;

        reservation.plan.batch_bytes =
            batch_bytes;

        reservation.plan.decoded_cache_bytes =
            std::min(
                fair_decoded_cache_budget_bytes(),
                remaining_decoded_cache_bytes());

        reservation.decoded_cache_bytes =
            reservation
                .plan
                .decoded_cache_bytes;

        const std::size_t host_budget =
            std::min(
                fair_host_storage_budget_bytes(),
                remaining_host_storage_bytes());

        reservation.plan.host_prefetch_batches =
            calculate_auto_prefetch_batches(
                host_budget,
                batch_bytes,
                loader_workers,
                policy_
                    .max_host_prefetch_batches,
                "host");

        reservation.host_storage_bytes =
            checked_batch_bytes(
                reservation
                        .plan
                        .host_prefetch_batches +
                    loader_workers,
                batch_bytes,
                "host storage reservation");

        if (reservation.host_storage_bytes >
            remaining_host_storage_bytes())
        {
            throw std::runtime_error(
                "DataLoader shared memory budget does not have enough host storage remaining");
        }

        if (uses_device)
        {
            const std::size_t device_budget =
                std::min(
                    fair_device_storage_budget_bytes(),
                    remaining_device_storage_bytes());

            reservation.plan.device_prefetch_batches =
                calculate_auto_prefetch_batches(
                    device_budget,
                    batch_bytes,
                    1,
                    policy_
                        .max_device_prefetch_batches,
                    "device");

            reservation.device_storage_bytes =
                checked_batch_bytes(
                    reservation
                            .plan
                            .device_prefetch_batches +
                        1,
                    batch_bytes,
                    "device storage reservation");

            if (reservation.device_storage_bytes >
                remaining_device_storage_bytes())
            {
                throw std::runtime_error(
                    "DataLoader shared memory budget does not have enough device storage remaining");
            }
        }
        else
        {
            reservation.plan.device_prefetch_batches =
                0;

            reservation.device_storage_bytes =
                0;
        }

        reserved_decoded_cache_bytes_ +=
            reservation
                .decoded_cache_bytes;

        reserved_host_storage_bytes_ +=
            reservation
                .host_storage_bytes;

        reserved_device_storage_bytes_ +=
            reservation
                .device_storage_bytes;

        return reservation;
    }

    DataLoaderMemoryReservation
    DataLoaderMemoryBudget::reserve_manual(
        std::size_t batch_bytes,
        std::size_t decoded_cache_bytes,
        std::size_t host_prefetch_batches,
        std::size_t device_prefetch_batches,
        std::size_t loader_workers,
        bool uses_device)
    {
        if (batch_bytes == 0)
        {
            throw std::runtime_error(
                "DataLoaderMemoryBudget received zero batch size");
        }

        if (loader_workers == 0)
        {
            throw std::runtime_error(
                "DataLoaderMemoryBudget received zero loader workers");
        }

        if (host_prefetch_batches == 0)
        {
            throw std::runtime_error(
                "DataLoaderMemoryBudget host prefetch batch count must be greater than zero");
        }

        if (uses_device &&
            device_prefetch_batches == 0)
        {
            throw std::runtime_error(
                "DataLoaderMemoryBudget device prefetch batch count must be greater than zero");
        }

        std::lock_guard<std::mutex> lock(
            mutex_);

        DataLoaderMemoryReservation reservation;

        reservation.plan.available_ram_bytes =
            available_ram_bytes_;

        reservation.plan.available_vram_bytes =
            available_vram_bytes_;

        reservation.plan.batch_bytes =
            batch_bytes;

        reservation.plan.decoded_cache_bytes =
            std::min(
                decoded_cache_bytes,
                remaining_decoded_cache_bytes());

        reservation.decoded_cache_bytes =
            reservation
                .plan
                .decoded_cache_bytes;

        reservation.plan.host_prefetch_batches =
            host_prefetch_batches;

        reservation.host_storage_bytes =
            checked_batch_bytes(
                host_prefetch_batches +
                    loader_workers,
                batch_bytes,
                "host storage reservation");

        if (reservation.host_storage_bytes >
            remaining_host_storage_bytes())
        {
            throw std::runtime_error(
                "DataLoader shared memory budget does not have enough host storage remaining");
        }

        if (uses_device)
        {
            reservation.plan.device_prefetch_batches =
                device_prefetch_batches;

            reservation.device_storage_bytes =
                checked_batch_bytes(
                    device_prefetch_batches +
                        1,
                    batch_bytes,
                    "device storage reservation");

            if (reservation.device_storage_bytes >
                remaining_device_storage_bytes())
            {
                throw std::runtime_error(
                    "DataLoader shared memory budget does not have enough device storage remaining");
            }
        }
        else
        {
            reservation.plan.device_prefetch_batches =
                0;

            reservation.device_storage_bytes =
                0;
        }

        reserved_decoded_cache_bytes_ +=
            reservation
                .decoded_cache_bytes;

        reserved_host_storage_bytes_ +=
            reservation
                .host_storage_bytes;

        reserved_device_storage_bytes_ +=
            reservation
                .device_storage_bytes;

        return reservation;
    }

    void DataLoaderMemoryBudget::release(
        const DataLoaderMemoryReservation &reservation) noexcept
    {
        std::lock_guard<std::mutex> lock(
            mutex_);

        if (reservation.decoded_cache_bytes >
            reserved_decoded_cache_bytes_)
        {
            reserved_decoded_cache_bytes_ =
                0;
        }
        else
        {
            reserved_decoded_cache_bytes_ -=
                reservation
                    .decoded_cache_bytes;
        }

        if (reservation.host_storage_bytes >
            reserved_host_storage_bytes_)
        {
            reserved_host_storage_bytes_ =
                0;
        }
        else
        {
            reserved_host_storage_bytes_ -=
                reservation
                    .host_storage_bytes;
        }

        if (reservation.device_storage_bytes >
            reserved_device_storage_bytes_)
        {
            reserved_device_storage_bytes_ =
                0;
        }
        else
        {
            reserved_device_storage_bytes_ -=
                reservation
                    .device_storage_bytes;
        }
    }

    Device DataLoaderMemoryBudget::device() const
    {
        return device_;
    }

    std::size_t
    DataLoaderMemoryBudget::available_ram_bytes() const
    {
        return available_ram_bytes_;
    }

    std::size_t
    DataLoaderMemoryBudget::available_vram_bytes() const
    {
        return available_vram_bytes_;
    }

    std::size_t
    DataLoaderMemoryBudget::decoded_cache_capacity_bytes() const
    {
        return decoded_cache_capacity_bytes_;
    }

    std::size_t
    DataLoaderMemoryBudget::host_storage_capacity_bytes() const
    {
        return host_storage_capacity_bytes_;
    }

    std::size_t
    DataLoaderMemoryBudget::device_storage_capacity_bytes() const
    {
        return device_storage_capacity_bytes_;
    }

    std::size_t
    DataLoaderMemoryBudget::reserved_decoded_cache_bytes() const
    {
        std::lock_guard<std::mutex> lock(
            mutex_);

        return reserved_decoded_cache_bytes_;
    }

    std::size_t
    DataLoaderMemoryBudget::reserved_host_storage_bytes() const
    {
        std::lock_guard<std::mutex> lock(
            mutex_);

        return reserved_host_storage_bytes_;
    }

    std::size_t
    DataLoaderMemoryBudget::reserved_device_storage_bytes() const
    {
        std::lock_guard<std::mutex> lock(
            mutex_);

        return reserved_device_storage_bytes_;
    }

    std::size_t
    DataLoaderMemoryBudget::remaining_decoded_cache_bytes() const
    {
        if (reserved_decoded_cache_bytes_ >=
            decoded_cache_capacity_bytes_)
        {
            return 0;
        }

        return decoded_cache_capacity_bytes_ -
               reserved_decoded_cache_bytes_;
    }

    std::size_t
    DataLoaderMemoryBudget::remaining_host_storage_bytes() const
    {
        if (reserved_host_storage_bytes_ >=
            host_storage_capacity_bytes_)
        {
            return 0;
        }

        return host_storage_capacity_bytes_ -
               reserved_host_storage_bytes_;
    }

    std::size_t
    DataLoaderMemoryBudget::remaining_device_storage_bytes() const
    {
        if (reserved_device_storage_bytes_ >=
            device_storage_capacity_bytes_)
        {
            return 0;
        }

        return device_storage_capacity_bytes_ -
               reserved_device_storage_bytes_;
    }

    std::size_t
    DataLoaderMemoryBudget::fair_decoded_cache_budget_bytes() const
    {
        return decoded_cache_capacity_bytes_ /
               expected_loader_count_;
    }

    std::size_t
    DataLoaderMemoryBudget::fair_host_storage_budget_bytes() const
    {
        return host_storage_capacity_bytes_ /
               expected_loader_count_;
    }

    std::size_t
    DataLoaderMemoryBudget::fair_device_storage_budget_bytes() const
    {
        return device_storage_capacity_bytes_ /
               expected_loader_count_;
    }

    std::size_t
    DataLoaderMemoryBudget::calculate_auto_prefetch_batches(
        std::size_t budget_bytes,
        std::size_t batch_bytes,
        std::size_t extra_batches,
        std::size_t maximum_prefetch_batches,
        const char *name) const
    {
        if (maximum_prefetch_batches == 0)
        {
            throw std::runtime_error(
                std::string("DataLoaderMemoryBudget ") +
                name +
                " maximum prefetch batch count must be greater than zero");
        }

        const std::size_t minimum_batches =
            extra_batches +
            1;

        const std::size_t minimum_bytes =
            checked_batch_bytes(
                minimum_batches,
                batch_bytes,
                name);

        if (budget_bytes < minimum_bytes)
        {
            throw std::runtime_error(
                std::string("DataLoader shared memory budget does not have enough ") +
                name +
                " storage remaining");
        }

        const std::size_t affordable_total_batches =
            budget_bytes /
            batch_bytes;

        const std::size_t affordable_prefetch_batches =
            affordable_total_batches -
            extra_batches;

        return std::max<std::size_t>(
            1,
            std::min(
                affordable_prefetch_batches,
                maximum_prefetch_batches));
    }

    std::size_t
    DataLoaderMemoryBudget::checked_batch_bytes(
        std::size_t batch_count,
        std::size_t batch_bytes,
        const char *name)
    {
        if (batch_count != 0 &&
            batch_bytes >
                std::numeric_limits<std::size_t>::max() /
                    batch_count)
        {
            throw std::runtime_error(
                std::string("DataLoaderMemoryBudget overflow while calculating ") +
                name);
        }

        return batch_count *
               batch_bytes;
    }

    void DataLoaderMemoryBudget::validate_policy(
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

        if (policy.max_host_prefetch_batches == 0)
        {
            throw std::runtime_error(
                "max_host_prefetch_batches must be greater than zero");
        }

        if (policy.max_device_prefetch_batches == 0)
        {
            throw std::runtime_error(
                "max_device_prefetch_batches must be greater than zero");
        }

        if (policy.decoded_cache_ram_fraction +
                policy.host_prefetch_ram_fraction >
            1.0f)
        {
            throw std::runtime_error(
                "decoded cache RAM fraction plus host prefetch RAM fraction must not exceed one");
        }
    }

    void DataLoaderMemoryBudget::validate_fraction(
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

}