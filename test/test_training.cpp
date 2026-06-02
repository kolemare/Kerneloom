#include <backend/backend.hpp>

#include <cnn/losses/categorical_cross_entropy_loss.hpp>
#include <cnn/network/initializer.hpp>
#include <cnn/network/sequential.hpp>
#include <cnn/optimizers/adam_optimizer.hpp>
#include <cnn/training/training.hpp>

#include <data/data_loader.hpp>
#include <data/data_loader_options.hpp>
#include <data/image_dataset.hpp>
#include <data/image_transform.hpp>

#include <core/device.hpp>
#include <core/dtype.hpp>

#include <ops/activation.hpp>

#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>

int main()
{
    try
    {
        const kl::Device target =
            kl::default_device();

        constexpr std::size_t batch_size = 80;
        constexpr std::size_t epoch_count = 20;

        kl::Sequential cnn(
            batch_size,
            3,
            32,
            32,
            kl::DType::Float32,
            target);

        cnn.addConvolutionLayer(32, 3);
        cnn.addActivationLayer(kl::ActivationType::ReLU);
        cnn.addMaxPoolingLayer(2, 2);

        cnn.addConvolutionLayer(64, 3);
        cnn.addActivationLayer(kl::ActivationType::ReLU);
        cnn.addMaxPoolingLayer(2, 2);

        cnn.addConvolutionLayer(64, 3);
        cnn.addActivationLayer(kl::ActivationType::ReLU);
        cnn.addMaxPoolingLayer(2, 2);

        cnn.addFlattenLayer();

        cnn.addFullyConnectedLayer(128);
        cnn.addActivationLayer(kl::ActivationType::ReLU);

        cnn.addFullyConnectedLayer(10);
        cnn.addActivationLayer(kl::ActivationType::Softmax);

        cnn.initializeWeights(
            kl::InitializerType::KaimingUniform);

        cnn.initializeBiases(
            kl::InitializerType::Zeros);

        kl::CategoricalCrossEntropyLoss loss;
        kl::AdamOptimizer optimizer(0.001f);

        kl::Training training(
            cnn,
            loss,
            optimizer);

        kl::ImageDataset dataset(
            "/home/marko/Projects/CNN-CPP/datasets/cifar10/training_set");

        kl::DataLoaderOptions options;
        options.batch_size = batch_size;
        options.input_dtype = kl::DType::Float32;
        options.shuffle = true;
        options.drop_last = true;
        options.loader_workers = 8;
        options.automatic_memory_planning = true;
        options.pin_host_memory = true;

        kl::DataLoader loader(
            dataset.samples(),
            kl::ImageTransform(32, 32),
            target,
            options);

        std::cout
            << "Device: "
            << kl::to_string(target.type())
            << " | Images: "
            << dataset.size()
            << " | Batches: "
            << loader.batch_count()
            << '\n';

        for (std::size_t epoch = 0;
             epoch < epoch_count;
             ++epoch)
        {
            std::size_t batch_index =
                0;

            float total_loss =
                0.0f;

            while (loader.has_next())
            {
                kl::Batch batch =
                    loader.next();

                const kl::TrainingResult result =
                    training.trainBatch(
                        batch);

                ++batch_index;

                total_loss +=
                    result.loss;

                std::cout
                    << '\r'
                    << "epoch "
                    << epoch + 1
                    << "/"
                    << epoch_count
                    << " | batch "
                    << batch_index
                    << "/"
                    << loader.batch_count()
                    << " | loss="
                    << std::fixed
                    << std::setprecision(6)
                    << result.loss
                    << std::flush;
            }

            const float average_loss =
                total_loss /
                static_cast<float>(
                    batch_index);

            std::cout
                << '\r'
                << "epoch "
                << epoch + 1
                << "/"
                << epoch_count
                << " | average_loss="
                << std::fixed
                << std::setprecision(6)
                << average_loss
                << "                    "
                << '\n';

            if (epoch + 1 <
                epoch_count)
            {
                loader.reset_epoch();
            }
        }

        return EXIT_SUCCESS;
    }
    catch (const std::exception &exception)
    {
        std::cerr
            << '\n'
            << exception.what()
            << '\n';

        return EXIT_FAILURE;
    }
}