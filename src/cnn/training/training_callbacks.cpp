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
                        << " | average_loss="
                        << std::fixed
                        << std::setprecision(6)
                        << progress.average_loss
                        << " | accuracy="
                        << std::fixed
                        << std::setprecision(4)
                        << progress.average_accuracy
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
                    << " | average_loss="
                    << std::fixed
                    << std::setprecision(6)
                    << progress.average_loss
                    << " | accuracy="
                    << std::fixed
                    << std::setprecision(4)
                    << progress.average_accuracy
                    << '\n';
            };
        }

    }

}