#include <cnn/training/training_callbacks.hpp>

#include <iomanip>
#include <iostream>

namespace kl
{

    namespace training_callbacks
    {

        TrainingCallback terminal_loss()
        {
            return [](
                       const TrainingProgress &progress)
            {
                if (progress.epoch_complete)
                {
                    std::cout
                        << '\r'
                        << "epoch "
                        << progress.epoch
                        << "/"
                        << progress.epoch_count
                        << " | train_loss="
                        << std::fixed
                        << std::setprecision(6)
                        << progress.average_loss
                        << " | train_acc="
                        << std::fixed
                        << std::setprecision(4)
                        << progress.average_accuracy;

                    if (progress.has_validation)
                    {
                        std::cout
                            << " | val_loss="
                            << std::fixed
                            << std::setprecision(6)
                            << progress.validation_loss
                            << " | val_acc="
                            << std::fixed
                            << std::setprecision(4)
                            << progress.validation_accuracy;
                    }

                    std::cout
                        << "                    "
                        << '\n';

                    return;
                }

                std::cout
                    << '\r'
                    << "epoch "
                    << progress.epoch
                    << "/"
                    << progress.epoch_count
                    << " | batch "
                    << progress.batch
                    << "/"
                    << progress.batch_count
                    << " | loss="
                    << std::fixed
                    << std::setprecision(6)
                    << progress.batch_loss
                    << " | acc="
                    << std::fixed
                    << std::setprecision(4)
                    << progress.batch_accuracy
                    << std::flush;
            };
        }

        TrainingCallback epoch_loss()
        {
            return [](
                       const TrainingProgress &progress)
            {
                if (!progress.epoch_complete)
                {
                    return;
                }

                std::cout
                    << "epoch "
                    << progress.epoch
                    << "/"
                    << progress.epoch_count
                    << " | train_loss="
                    << std::fixed
                    << std::setprecision(6)
                    << progress.average_loss
                    << " | train_acc="
                    << std::fixed
                    << std::setprecision(4)
                    << progress.average_accuracy;

                if (progress.has_validation)
                {
                    std::cout
                        << " | val_loss="
                        << std::fixed
                        << std::setprecision(6)
                        << progress.validation_loss
                        << " | val_acc="
                        << std::fixed
                        << std::setprecision(4)
                        << progress.validation_accuracy;
                }

                std::cout
                    << '\n';
            };
        }

    }

}