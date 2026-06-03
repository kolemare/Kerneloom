#include <backend/backend.hpp>

#include <cnn/losses/categorical_cross_entropy_loss.hpp>
#include <cnn/training/training_callbacks.hpp>
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
        const kl::Device device =
            kl::default_device();

        constexpr std::size_t batch_size =
            80;

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
        kl::AdamOptimizer optimizer(0.001f);

        kl::Training training(
            cnn,
            loss,
            optimizer);

        kl::ImageDataset dataset(
            "/home/marko/Projects/CNN-CPP/datasets/cifar10/training_set");

        kl::DataLoaderOptions options;
        options.batch_size = batch_size;
        options.shuffle = true;
        options.drop_last = true;

        kl::DataLoader loader(
            dataset.samples(),
            kl::ImageTransform(32, 32),
            device,
            options);

        training.fit(
            loader,
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