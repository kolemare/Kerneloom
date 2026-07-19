#ifndef KL_TEST_BENCHMARK_TIMER_HPP
#define KL_TEST_BENCHMARK_TIMER_HPP

#include <core/device.hpp>
#include <core/synchronize.hpp>

#include <chrono>
#include <cstddef>
#include <iostream>

namespace kl::test
{

    class BenchmarkTimer
    {
    public:
        void start()
        {
            start_time_ = Clock::now();
        }

        [[nodiscard]] double stopMilliseconds() const
        {
            const auto end_time = Clock::now();

            return std::chrono::duration<double, std::milli>(
                       end_time - start_time_)
                .count();
        }

    private:
        using Clock = std::chrono::steady_clock;

        Clock::time_point start_time_{};
    };

    template <typename Fn>
    double benchmarkGpu(
        Device device,
        std::size_t warmup_iterations,
        std::size_t measured_iterations,
        bool print_each_iteration,
        const char *name,
        Fn &&fn)
    {
        for (std::size_t i = 0; i < warmup_iterations; ++i)
        {
            (void)fn();
        }

        synchronize(device);

        double total_ms = 0.0;

        for (std::size_t i = 0; i < measured_iterations; ++i)
        {
            synchronize(device);

            BenchmarkTimer timer;
            timer.start();

            (void)fn();

            synchronize(device);

            const double iteration_ms =
                timer.stopMilliseconds();

            total_ms += iteration_ms;

            if (print_each_iteration)
            {
                std::cout
                    << "  "
                    << name
                    << " iteration "
                    << i
                    << ": "
                    << iteration_ms
                    << " ms\n";
            }
        }

        return total_ms / static_cast<double>(measured_iterations);
    }

}

#endif // KL_TEST_BENCHMARK_TIMER_HPP