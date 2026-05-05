#include <backend/backend.hpp>
#include <core/device.hpp>
#include <core/dtype.hpp>
#include <core/tensor.hpp>
#include <ops/matmul_naive.hpp>

#include <cstdlib>
#include <exception>
#include <iostream>

int main()
{
    try
    {
        kl::Device target = kl::default_device();

        kl::Tensor a_cpu(kl::Shape{2, 3}, kl::DType::Float32, kl::Device::cpu());
        kl::Tensor b_cpu(kl::Shape{3, 2}, kl::DType::Float32, kl::Device::cpu());

        float *a = static_cast<float *>(a_cpu.data());
        float *b = static_cast<float *>(b_cpu.data());

        a[0] = 1.0f;
        a[1] = 2.0f;
        a[2] = 3.0f;
        a[3] = 4.0f;
        a[4] = 5.0f;
        a[5] = 6.0f;

        b[0] = 7.0f;
        b[1] = 8.0f;
        b[2] = 9.0f;
        b[3] = 10.0f;
        b[4] = 11.0f;
        b[5] = 12.0f;

        kl::Tensor a_target = a_cpu.to(target);
        kl::Tensor b_target = b_cpu.to(target);

        kl::Tensor c_target(
            kl::Shape{a_target.shape()[0], b_target.shape()[1]},
            kl::DType::Float32,
            target,
            kl::Layout::Unknown,
            kl::Storage::RowMajor);

        kl::matmul_naive(a_target, b_target, c_target);

        kl::Tensor c_cpu = c_target.to(kl::Device::cpu());

        const float *c = static_cast<const float *>(c_cpu.data());

        std::cout << c[0] << ' ' << c[1] << '\n';
        std::cout << c[2] << ' ' << c[3] << '\n';

        return EXIT_SUCCESS;
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }
}