#include <backend/backend.hpp>

#include <data/data_loader.hpp>
#include <data/data_loader_options.hpp>
#include <data/image_dataset.hpp>
#include <data/image_transform.hpp>

#include <core/device.hpp>
#include <core/dtype.hpp>

#include <chrono>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <thread>
#include <vector>

namespace
{

    constexpr std::size_t mebibyte =
        1024ULL *
        1024ULL;

    constexpr std::size_t gibibyte =
        1024ULL *
        mebibyte;

    void print_stats(
        const kl::DataLoader &loader)
    {
        std::cout
            << "Host prefetched batches: "
            << loader
                   .host_prefetched_batch_count()
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

        std::cout
            << "Decoded cache hits: "
            << loader
                   .decoded_cache_hit_count()
            << '\n';

        std::cout
            << "Decoded cache misses: "
            << loader
                   .decoded_cache_miss_count()
            << '\n';
    }

    void wait_and_print_stats(
        const kl::DataLoader &loader,
        const char *message,
        std::size_t seconds)
    {
        std::cout
            << message
            << " | waiting "
            << seconds
            << " seconds...\n";

        std::this_thread::sleep_for(
            std::chrono::seconds(
                seconds));

        print_stats(
            loader);
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
        options.batch_size = 64;
        options.input_dtype =
            kl::DType::Float32;

        options.shuffle = true;
        options.drop_last = true;

        options.loader_workers = 8;
        options.host_prefetch_batches = 64;

        options.decoded_cache_bytes =
            4ULL *
            gibibyte;

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
            << "Classes: "
            << dataset.class_count()
            << '\n';

        std::cout
            << "Images: "
            << dataset.size()
            << '\n';

        std::cout
            << "Batches: "
            << loader.batch_count()
            << '\n';

        wait_and_print_stats(
            loader,
            "Initial queue fill",
            30);

        std::vector<kl::Batch>
            held_batches;

        for (std::size_t i = 0;
             i < 8;
             ++i)
        {
            held_batches.push_back(
                loader.next());
        }

        std::cout
            << "Loaded 8 batches on "
            << kl::to_string(
                   target.type())
            << '\n';

        std::cout
            << "First batch shape: "
            << held_batches[0]
                   .inputs
                   .shape()[0]
            << "x"
            << held_batches[0]
                   .inputs
                   .shape()[1]
            << "x"
            << held_batches[0]
                   .inputs
                   .shape()[2]
            << "x"
            << held_batches[0]
                   .inputs
                   .shape()[3]
            << '\n';

        wait_and_print_stats(
            loader,
            "Queue refill after consuming batches",
            15);

        std::cout
            << "Resetting epoch...\n";

        loader.reset_epoch();

        wait_and_print_stats(
            loader,
            "Queue refill after epoch reset",
            30);

        kl::Batch first_batch_new_epoch =
            loader.next();

        std::cout
            << "Loaded first batch from new epoch\n";

        std::cout
            << "Holding batches for memory inspection for 30 seconds...\n";

        std::this_thread::sleep_for(
            std::chrono::seconds(
                30));

        std::cout
            << "DataLoader Phase 3 test passed\n";

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