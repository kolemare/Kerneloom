#include "data/data_loader/data_loader_impl.hpp"

#include <stdexcept>

namespace kl::data_loader_internal
{

    std::mutex &default_budget_mutex()
    {
        static std::mutex mutex;

        return mutex;
    }

    std::size_t &expected_loader_count_storage()
    {
        static std::size_t expected_loader_count =
            1;

        return expected_loader_count;
    }

    std::weak_ptr<DataLoaderMemoryBudget> &
    default_budget_storage(
        Device device)
    {
        static std::weak_ptr<DataLoaderMemoryBudget>
            cpu_budget;

        static std::weak_ptr<DataLoaderMemoryBudget>
            cuda_budget;

        static std::weak_ptr<DataLoaderMemoryBudget>
            rocm_budget;

        switch (device.type())
        {
        case DeviceType::CPU:
            return cpu_budget;

        case DeviceType::CUDA:
            return cuda_budget;

        case DeviceType::ROCM:
            return rocm_budget;

        default:
            throw std::runtime_error(
                "unsupported DataLoader default memory budget device");
        }
    }

    void set_expected_loader_count(
        std::size_t expected_loader_count)
    {
        if (expected_loader_count == 0)
        {
            throw std::runtime_error(
                "DataLoader expected loader count must be greater than zero");
        }

        std::lock_guard<std::mutex> lock(
            default_budget_mutex());

        if (default_budget_storage(
                Device::cpu())
                    .lock() != nullptr ||
            default_budget_storage(
                Device::cuda())
                    .lock() != nullptr ||
            default_budget_storage(
                Device::rocm())
                    .lock() != nullptr)
        {
            throw std::runtime_error(
                "DataLoader expected loader count must be set before creating default DataLoaders");
        }

        expected_loader_count_storage() =
            expected_loader_count;
    }

    std::size_t expected_loader_count()
    {
        std::lock_guard<std::mutex> lock(
            default_budget_mutex());

        return expected_loader_count_storage();
    }

    std::shared_ptr<DataLoaderMemoryBudget>
    default_data_loader_memory_budget(
        Device device,
        const MemoryPolicy &policy)
    {
        std::lock_guard<std::mutex> lock(
            default_budget_mutex());

        std::weak_ptr<DataLoaderMemoryBudget> &slot =
            default_budget_storage(
                device);

        std::shared_ptr<DataLoaderMemoryBudget>
            existing =
                slot.lock();

        if (existing !=
            nullptr)
        {
            return existing;
        }

        std::shared_ptr<DataLoaderMemoryBudget>
            created =
                std::make_shared<DataLoaderMemoryBudget>(
                    device,
                    policy,
                    expected_loader_count_storage());

        slot =
            created;

        return created;
    }

}