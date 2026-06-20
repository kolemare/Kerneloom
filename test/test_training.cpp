#include <backend/backend.hpp>

#include <cnn/losses/categorical_cross_entropy_loss.hpp>
#include <cnn/training/training_callbacks.hpp>
#include <cnn/network/initializer.hpp>
#include <cnn/network/sequential.hpp>
#include <cnn/optimizers/adam_optimizer.hpp>
#include <cnn/training/training.hpp>

#include <data/data_loader.hpp>
#include <data/data_loader_options.hpp>
#include <data/dataset_split.hpp>
#include <data/image_dataset.hpp>
#include <data/image_transform.hpp>

#include <core/device.hpp>
#include <core/dtype.hpp>

#include <ops/activation.hpp>

#include <cstdlib>
#include <exception>
#include <iostream>

int main()
{
    try
    {
        const kl::Device device =
            kl::default_device();

        constexpr std::size_t batch_size =
            80;

        kl::ImageDataset dataset(
            "/home/marko/Projects/CNN-CPP/datasets/cifar10/training_set");

        const kl::DatasetSplit split =
            kl::split_dataset(
                dataset.samples(),
                0.8f,
                0.1f,
                1337);

        kl::DataLoaderOptions train_options;
        train_options.batch_size =
            batch_size;

        train_options.shuffle =
            true;

        train_options.drop_last =
            true;

        kl::DataLoaderOptions validation_options;
        validation_options.batch_size =
            batch_size;

        validation_options.shuffle =
            false;

        validation_options.drop_last =
            false;

        kl::DataLoader train_loader(
            split.train,
            kl::ImageTransform(32, 32),
            device,
            train_options);

        kl::DataLoader validation_loader(
            split.validation,
            kl::ImageTransform(32, 32),
            device,
            validation_options);

        kl::Sequential cnn(
            batch_size,
            3,
            32,
            32,
            kl::DType::Float32,
            device);

        cnn.addConvolutionLayer(32, 3);
        cnn.addActivationLayer(kl::ActivationType::ReLU);
        cnn.addMaxPoolingLayer(2);

        cnn.addConvolutionLayer(64, 3);
        cnn.addActivationLayer(kl::ActivationType::ReLU);
        cnn.addMaxPoolingLayer(2);

        cnn.addConvolutionLayer(64, 3);
        cnn.addActivationLayer(kl::ActivationType::ReLU);
        cnn.addMaxPoolingLayer(2);

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
        kl::AdamOptimizer optimizer(
            0.001f);

        kl::Training training(
            cnn,
            loss,
            optimizer);

        training.fit(
            train_loader,
            validation_loader,
            20,
            kl::training_callbacks::terminal_loss());

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