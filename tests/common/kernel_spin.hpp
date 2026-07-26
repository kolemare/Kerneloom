#ifndef KL_TEST_KERNEL_SPIN_HPP
#define KL_TEST_KERNEL_SPIN_HPP

#include <core/device.hpp>
#include <core/synchronize.hpp>

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace kl::test
{

    struct KernelSpinOptions
    {
        std::size_t iterations =
            200;

        std::size_t synchronization_interval =
            10;

        std::size_t progress_interval =
            10;
    };

    inline void validateKernelSpinOptions(
        const KernelSpinOptions &options)
    {
        if (options.iterations == 0)
        {
            throw std::runtime_error(
                "kernel spin iterations must be greater than zero");
        }

        if (options.synchronization_interval == 0)
        {
            throw std::runtime_error(
                "kernel spin synchronization interval must be greater than zero");
        }

        if (options.progress_interval == 0)
        {
            throw std::runtime_error(
                "kernel spin progress interval must be greater than zero");
        }
    }

    template <typename LaunchFn>
    void spinKernel(
        Device device,
        const char *kernel_name,
        const KernelSpinOptions &options,
        LaunchFn &&launch)
    {
        validateKernelSpinOptions(
            options);

        synchronize(
            device);

        std::cout
            << "\n[Kernel Spin Stress]\n"
            << "  Kernel: "
            << kernel_name
            << '\n'
            << "  Iterations: "
            << options.iterations
            << '\n'
            << "  Synchronization interval: "
            << options.synchronization_interval
            << '\n'
            << "  Progress interval: "
            << options.progress_interval
            << '\n';

        for (std::size_t iteration = 0;
             iteration < options.iterations;
             ++iteration)
        {
            std::forward<LaunchFn>(
                launch)();

            const std::size_t completed =
                iteration + 1;

            if (completed %
                    options.synchronization_interval ==
                0)
            {
                synchronize(
                    device);
            }

            if (completed %
                    options.progress_interval ==
                0)
            {
                std::cout
                    << "  Completed "
                    << completed
                    << " / "
                    << options.iterations
                    << " launches\n";
            }
        }

        synchronize(
            device);

        std::cout
            << "  Completed all "
            << options.iterations
            << " launches\n";
    }

    template <typename LaunchFn>
    void spinKernel(
        Device device,
        const char *kernel_name,
        LaunchFn &&launch)
    {
        spinKernel(
            device,
            kernel_name,
            KernelSpinOptions{},
            std::forward<LaunchFn>(
                launch));
    }

}

#endif // KL_TEST_KERNEL_SPIN_HPP