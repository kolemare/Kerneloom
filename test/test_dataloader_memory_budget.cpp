#include <backend/backend.hpp>

#include <data/data_loader.hpp>
#include <data/data_loader_options.hpp>
#include <data/image_dataset.hpp>
#include <data/image_transform.hpp>
#include <data/memory_policy.hpp>

#include <core/device.hpp>

#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

int main()
{
    try
    {
        const kl::Device device =
            kl::default_device();

        constexpr size_t loader_count = 10;

        kl::DataLoader::set_expected_loader_count(loader_count);

        kl::MemoryPolicy memory_policy;

        memory_policy.decoded_cache_ram_fraction =
            0.40f;

        memory_policy.host_prefetch_ram_fraction =
            0.40f;

        memory_policy.device_prefetch_vram_fraction =
            0.80f;

        kl::ImageDataset dataset(
            "/media/WDGold/Temp/HaGRIDv2");

        std::cout
            << "Device: "
            << kl::to_string(device.type())
            << '\n';

        std::cout
            << "Samples: "
            << dataset.size()
            << '\n';

        std::cout
            << "Classes: "
            << dataset.class_count()
            << '\n';

        std::vector<std::unique_ptr<kl::DataLoader>>
            loaders;

        for (std::size_t i = 0;
             i < loader_count;
             ++i)
        {
            kl::DataLoaderOptions options;

            options.batch_size =
                80;

            options.shuffle =
                true;

            options.loader_workers =
                4;

            options.pin_host_memory =
                true;

            options.memory =
                memory_policy;

            loaders.push_back(
                std::make_unique<kl::DataLoader>(
                    dataset.samples(),
                    kl::ImageTransform(224, 224),
                    device,
                    options));

            const kl::MemoryPlan &plan =
                loaders.back()->memory_plan();

            std::cout
                << "Loader "
                << i + 1
                << ": decoded_cache="
                << plan.decoded_cache_bytes / (1024 * 1024)
                << " MiB, host_prefetch="
                << plan.host_prefetch_batches
                << ", device_prefetch="
                << plan.device_prefetch_batches
                << '\n';
        }

        std::cout
            << "\nAll loaders created. Press ENTER to exit...\n";

        std::cin.get();

        return EXIT_SUCCESS;
    }
    catch (const std::exception &exception)
    {
        std::cerr
            << exception.what()
            << '\n';

        return EXIT_FAILURE;
    }
}