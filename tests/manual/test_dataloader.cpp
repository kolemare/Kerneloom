#include <backend/backend.hpp>

#include <data/data_loader.hpp>
#include <data/data_loader_options.hpp>
#include <data/image_dataset.hpp>
#include <data/image_transform.hpp>
#include <data/memory_plan.hpp>

#include <core/device.hpp>
#include <core/dtype.hpp>

#include <chrono>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <thread>

namespace
{

    constexpr std::size_t mebibyte =
        1024ULL *
        1024ULL;

    void print_stats(
        const kl::DataLoader &loader)
    {
        const kl::MemoryPlan &plan =
            loader.memory_plan();

        std::cout
            << "Available RAM MiB: "
            << plan.available_ram_bytes /
                   static_cast<double>(
                       mebibyte)
            << '\n';

        std::cout
            << "Available VRAM MiB: "
            << plan.available_vram_bytes /
                   static_cast<double>(
                       mebibyte)
            << '\n';

        std::cout
            << "Batch MiB: "
            << plan.batch_bytes /
                   static_cast<double>(
                       mebibyte)
            << '\n';

        std::cout
            << "Planned decoded cache MiB: "
            << plan.decoded_cache_bytes /
                   static_cast<double>(
                       mebibyte)
            << '\n';

        std::cout
            << "Host prefetched batches: "
            << loader
                   .host_prefetched_batch_count()
            << '\n';

        std::cout
            << "Device prefetched batches: "
            << loader
                   .device_prefetched_batch_count()
            << '\n';

        std::cout
            << "Allocated pooled host batches: "
            << loader
                   .pooled_host_batch_count()
            << '\n';

        std::cout
            << "Allocated pooled device batches: "
            << loader
                   .pooled_device_batch_count()
            << '\n';

        std::cout
            << "Decoded cache images: "
            << loader
                   .decoded_cache_image_count()
            << '\n';

        std::cout
            << "Decoded cache MiB: "
            << loader
                       .decoded_cache_used_bytes() /
                   static_cast<double>(
                       mebibyte)
            << '\n';
    }

}

int main()
{
    try
    {
        const kl::Device target =
            kl::default_device();

        kl::ImageDataset dataset(
            "/media/WDGold/Temp/HaGRIDv2");

        kl::DataLoaderOptions options;

        options.batch_size =
            64;

        options.input_dtype =
            kl::DType::Float32;

        options.shuffle =
            true;

        options.drop_last =
            true;

        options.loader_workers =
            8;

        options.automatic_memory_planning =
            true;

        options.pin_host_memory =
            true;

        kl::DataLoader loader(
            dataset.samples(),
            kl::ImageTransform(
                224,
                224),
            target,
            options);

        std::cout
            << "Device: "
            << kl::to_string(
                   target.type())
            << '\n';

        std::cout
            << "Images: "
            << dataset.size()
            << '\n';

        std::cout
            << "Loading for 3 minutes...\n";

        std::this_thread::sleep_for(
            std::chrono::minutes(
                3));

        print_stats(
            loader);

        std::cout
            << "Holding memory for 3 minutes...\n";

        std::this_thread::sleep_for(
            std::chrono::minutes(
                3));

        return EXIT_SUCCESS;
    }
    catch (const std::exception &e)
    {
        std::cerr
            << e.what()
            << '\n';

        return EXIT_FAILURE;
    }
}