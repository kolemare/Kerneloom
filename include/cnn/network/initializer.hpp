#ifndef KL_INITIALIZER_HPP
#define KL_INITIALIZER_HPP

#include <core/tensor.hpp>

#include <cstdint>

namespace kl
{

    enum class InitializerType
    {
        Zeros,
        Ones,
        Constant,
        XavierUniform,
        KaimingUniform
    };

    struct InitializerOptions
    {
        InitializerType type = InitializerType::Zeros;
        float value = 0.0f;
        std::uint32_t seed = 1234;
    };

    class Initializer
    {
    public:
        static void initialize(
            Tensor &tensor,
            const InitializerOptions &options);

        static void initialize(
            Tensor &tensor,
            InitializerType type);

        static void initialize(
            Tensor &tensor,
            InitializerType type,
            float value);

    private:
        static void initialize_cpu(
            Tensor &tensor,
            const InitializerOptions &options);

        static void fill_constant_cpu(
            Tensor &tensor,
            float value);

        static void fill_uniform_cpu(
            Tensor &tensor,
            float min_value,
            float max_value,
            std::uint32_t seed);

        static float fan_in(
            const Tensor &tensor);

        static float fan_out(
            const Tensor &tensor);
    };

}

#endif // KL_INITIALIZER_HPP