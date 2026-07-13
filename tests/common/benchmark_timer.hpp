#ifndef KL_TEST_BENCHMARK_TIMER_HPP
#define KL_TEST_BENCHMARK_TIMER_HPP

#include <chrono>

namespace kl::test
{

    class BenchmarkTimer
    {
    public:
        using Clock = std::chrono::steady_clock;

        void start()
        {
            start_time_ = Clock::now();
        }

        [[nodiscard]] double stopMilliseconds() const
        {
            const auto end_time = Clock::now();
            const auto elapsed = std::chrono::duration<double, std::milli>(
                end_time - start_time_);

            return elapsed.count();
        }

    private:
        Clock::time_point start_time_{};
    };

}

#endif // KL_TEST_BENCHMARK_TIMER_HPP