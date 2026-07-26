#ifndef KL_TEST_TENSOR_COMPARE_HPP
#define KL_TEST_TENSOR_COMPARE_HPP

#include "common/dtype_dispatch.hpp"
#include "common/tensor_compare_workspace.hpp"

#include <core/copy.hpp>
#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/synchronize.hpp>
#include <core/tensor.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace kl::test
{

    namespace tensor_compare_detail
    {

        struct ComparisonResult
        {
            double max_abs_error = 0.0;
            double max_rel_error = 0.0;

            std::size_t mismatches = 0;
            std::size_t elements = 0;
        };

        inline bool is_accelerator(
            DeviceType type)
        {
            return type == DeviceType::CUDA ||
                   type == DeviceType::ROCM;
        }

        inline DeviceType comparison_backend(
            const Tensor &expected,
            const Tensor &actual)
        {
            const DeviceType expected_type =
                expected.device().type();

            const DeviceType actual_type =
                actual.device().type();

            if (is_accelerator(expected_type) &&
                is_accelerator(actual_type) &&
                expected_type != actual_type)
            {
                throw std::runtime_error(
                    "tensorCompare does not directly compare CUDA and ROCm tensors");
            }

            if (is_accelerator(expected_type))
            {
                return expected_type;
            }

            if (is_accelerator(actual_type))
            {
                return actual_type;
            }

            return DeviceType::CPU;
        }

        inline void synchronize_inputs(
            const Tensor &expected,
            const Tensor &actual)
        {
            const DeviceType expected_type =
                expected.device().type();

            const DeviceType actual_type =
                actual.device().type();

            if (is_accelerator(expected_type))
            {
                synchronize(
                    expected.device());
            }

            if (is_accelerator(actual_type) &&
                actual_type != expected_type)
            {
                synchronize(
                    actual.device());
            }
        }

        inline std::size_t aligned_chunk_bytes(
            std::size_t requested_chunk_bytes,
            std::size_t element_size)
        {
            if (element_size == 0)
            {
                throw std::runtime_error(
                    "tensorCompare encountered zero-sized dtype");
            }

            const std::size_t aligned =
                requested_chunk_bytes -
                requested_chunk_bytes % element_size;

            if (aligned == 0)
            {
                return element_size;
            }

            return aligned;
        }

        template <typename T>
        void compare_elements(
            const T *expected_data,
            const T *actual_data,
            std::size_t element_count,
            double absolute_tolerance,
            double relative_tolerance,
            ComparisonResult &result)
        {
            for (std::size_t i = 0;
                 i < element_count;
                 ++i)
            {
                const double expected_value =
                    static_cast<double>(
                        expected_data[i]);

                const double actual_value =
                    static_cast<double>(
                        actual_data[i]);

                const double abs_error =
                    std::fabs(
                        expected_value -
                        actual_value);

                const double rel_error =
                    abs_error /
                    std::max(
                        std::fabs(expected_value),
                        1.0);

                result.max_abs_error =
                    std::max(
                        result.max_abs_error,
                        abs_error);

                result.max_rel_error =
                    std::max(
                        result.max_rel_error,
                        rel_error);

                if (abs_error > absolute_tolerance &&
                    rel_error > relative_tolerance)
                {
                    ++result.mismatches;
                }
            }

            result.elements +=
                element_count;
        }

        template <typename T>
        ComparisonResult compare_cpu_tensors(
            const Tensor &expected,
            const Tensor &actual,
            double absolute_tolerance,
            double relative_tolerance)
        {
            ComparisonResult result;

            const auto *expected_data =
                static_cast<const T *>(
                    expected.data());

            const auto *actual_data =
                static_cast<const T *>(
                    actual.data());

            compare_elements(
                expected_data,
                actual_data,
                expected.numel(),
                absolute_tolerance,
                relative_tolerance,
                result);

            return result;
        }

        template <typename T>
        ComparisonResult compare_chunked(
            const Tensor &expected,
            const Tensor &actual,
            DeviceType backend,
            double absolute_tolerance,
            double relative_tolerance)
        {
            TensorCompareWorkspace &workspace =
                tensorCompareWorkspace();

            const std::size_t chunk_capacity =
                aligned_chunk_bytes(
                    TensorCompareWorkspace::
                        default_chunk_bytes,
                    sizeof(T));

            workspace.prepare(
                backend,
                chunk_capacity);

            ComparisonResult result;

            const std::size_t total_bytes =
                expected.nbytes();

            for (std::size_t offset = 0;
                 offset < total_bytes;
                 offset += chunk_capacity)
            {
                const std::size_t current_bytes =
                    std::min(
                        chunk_capacity,
                        total_bytes - offset);

                if (current_bytes % sizeof(T) != 0)
                {
                    throw std::runtime_error(
                        "tensorCompare chunk is not aligned to dtype size");
                }

                copy_to_host(
                    workspace.expected_data(),
                    expected,
                    offset,
                    current_bytes);

                copy_to_host(
                    workspace.actual_data(),
                    actual,
                    offset,
                    current_bytes);

                const std::size_t current_elements =
                    current_bytes / sizeof(T);

                const auto *expected_data =
                    static_cast<const T *>(
                        workspace.expected_data());

                const auto *actual_data =
                    static_cast<const T *>(
                        workspace.actual_data());

                compare_elements(
                    expected_data,
                    actual_data,
                    current_elements,
                    absolute_tolerance,
                    relative_tolerance,
                    result);
            }

            return result;
        }

        inline void print_result(
            const ComparisonResult &result,
            double max_mismatch_ratio,
            bool passed)
        {
            if (result.mismatches == 0)
            {
                return;
            }

            const double mismatch_ratio =
                result.elements == 0
                    ? 0.0
                    : static_cast<double>(
                          result.mismatches) /
                          static_cast<double>(
                              result.elements);

            std::cout
                << (passed
                        ? "Tensor comparison passed with tiny mismatch ratio\n"
                        : "Tensor comparison failed\n")
                << "  Elements:        "
                << result.elements
                << '\n'
                << "  Mismatches:      "
                << result.mismatches
                << '\n'
                << "  Mismatch ratio:  "
                << std::scientific
                << mismatch_ratio
                << '\n'
                << "  Allowed ratio:   "
                << std::scientific
                << max_mismatch_ratio
                << '\n'
                << "  Max abs error:   "
                << std::scientific
                << result.max_abs_error
                << '\n'
                << "  Max rel error:   "
                << std::scientific
                << result.max_rel_error
                << '\n';
        }

    } // namespace tensor_compare_detail

    inline bool tensorCompare(
        const Tensor &expected,
        const Tensor &actual,
        double absolute_tolerance = 1.0e-4,
        double relative_tolerance = 1.0e-4,
        double max_mismatch_ratio = 0.0)
    {
        if (expected.shape() != actual.shape())
        {
            std::cout
                << "Tensor shape mismatch\n";

            return false;
        }

        if (expected.dtype() != actual.dtype())
        {
            std::cout
                << "Tensor dtype mismatch\n";

            return false;
        }

        if (expected.nbytes() != actual.nbytes())
        {
            std::cout
                << "Tensor byte-size mismatch\n";

            return false;
        }

        if (expected.numel() == 0)
        {
            return true;
        }

        const DeviceType backend =
            tensor_compare_detail::
                comparison_backend(
                    expected,
                    actual);

        tensor_compare_detail::
            synchronize_inputs(
                expected,
                actual);

        tensor_compare_detail::ComparisonResult result;

        dispatchFloatDType(
            expected.dtype(),
            [&]<typename T>()
            {
                if (backend == DeviceType::CPU)
                {
                    result =
                        tensor_compare_detail::
                            compare_cpu_tensors<T>(
                                expected,
                                actual,
                                absolute_tolerance,
                                relative_tolerance);
                }
                else
                {
                    result =
                        tensor_compare_detail::
                            compare_chunked<T>(
                                expected,
                                actual,
                                backend,
                                absolute_tolerance,
                                relative_tolerance);
                }
            });

        const double mismatch_ratio =
            result.elements == 0
                ? 0.0
                : static_cast<double>(
                      result.mismatches) /
                      static_cast<double>(
                          result.elements);

        const bool passed =
            mismatch_ratio <=
            max_mismatch_ratio;

        tensor_compare_detail::
            print_result(
                result,
                max_mismatch_ratio,
                passed);

        return passed;
    }

} // namespace kl::test

#endif // KL_TEST_TENSOR_COMPARE_HPP