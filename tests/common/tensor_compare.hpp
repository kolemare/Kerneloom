#ifndef KL_TEST_TENSOR_COMPARE_HPP
#define KL_TEST_TENSOR_COMPARE_HPP

#include "common/dtype_dispatch.hpp"

#include <core/device.hpp>
#include <core/tensor.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>

namespace kl::test
{

    inline bool tensorCompare(
        const Tensor &expected,
        const Tensor &actual,
        double absolute_tolerance = 1.0e-4,
        double relative_tolerance = 1.0e-4)
    {
        if (expected.shape() != actual.shape())
        {
            std::cout << "Tensor shape mismatch\n";
            return false;
        }

        if (expected.dtype() != actual.dtype())
        {
            std::cout << "Tensor dtype mismatch\n";
            return false;
        }

        Tensor expected_cpu = expected.to(Device::cpu());
        Tensor actual_cpu = actual.to(Device::cpu());

        bool passed = true;

        dispatchFloatDType(expected.dtype(), [&]<typename T>()
                           {
            const T* expected_data =
                static_cast<const T*>(expected_cpu.data());

            const T* actual_data =
                static_cast<const T*>(actual_cpu.data());

            double max_abs_error = 0.0;
            double max_rel_error = 0.0;
            std::size_t mismatches = 0;

            for (std::size_t i = 0; i < expected_cpu.numel(); ++i)
            {
                const double e = static_cast<double>(expected_data[i]);
                const double a = static_cast<double>(actual_data[i]);

                const double abs_error = std::fabs(e - a);
                const double rel_error =
                    abs_error / std::max(std::fabs(e), 1.0);

                max_abs_error = std::max(max_abs_error, abs_error);
                max_rel_error = std::max(max_rel_error, rel_error);

                if (abs_error > absolute_tolerance &&
                    rel_error > relative_tolerance)
                {
                    ++mismatches;
                }
            }

            if (mismatches > 0)
            {
                std::cout
                    << "Tensor comparison failed\n"
                    << "  Elements:       " << expected_cpu.numel() << '\n'
                    << "  Mismatches:     " << mismatches << '\n'
                    << "  Max abs error:  " << max_abs_error << '\n'
                    << "  Max rel error:  " << max_rel_error << '\n';

                passed = false;
            } });

        return passed;
    }

}

#endif // KL_TEST_TENSOR_COMPARE_HPP