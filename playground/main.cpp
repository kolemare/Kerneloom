#include <kerneloom.hpp>
#include <backend/backend.hpp>

#include <chrono>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>

int main()
{
    try
    {
        const kl::Device device = kl::default_device();

        constexpr std::size_t N = 1 << 15;

        const double gib_per_matrix =
            static_cast<double>(N) *
            static_cast<double>(N) *
            sizeof(float) /
            1024.0 / 1024.0 / 1024.0;

        std::cout << "Kerneloom playground\n";
        std::cout << "Backend: " << kl::to_string(device.type()) << "\n";
        std::cout << "Matrix A: " << N << " x " << N << "\n";
        std::cout << "Matrix B: " << N << " x " << N << "\n";
        std::cout << "Matrix C: " << N << " x " << N << "\n\n";

        std::cout << "Approx memory per matrix: "
                  << std::fixed << std::setprecision(2)
                  << gib_per_matrix << " GiB\n";

        std::cout << "Approx host memory for A + B + C: "
                  << gib_per_matrix * 3.0 << " GiB\n\n";

        std::cout << "Allocating tensors...\n";

        kl::Tensor a(kl::Shape2D(N, N), device);
        kl::Tensor b(kl::Shape2D(N, N), device);

        std::cout << "Initializing tensors...\n";

        for (std::size_t i = 0; i < a.size(); ++i)
        {
            a.data()[i] = 1.0f;
        }

        for (std::size_t i = 0; i < b.size(); ++i)
        {
            b.data()[i] = 1.0f;
        }

        std::cout << "Running matmul...\n";

        const auto start = std::chrono::high_resolution_clock::now();

        kl::Tensor c = kl::matmul(a, b);

        const auto end = std::chrono::high_resolution_clock::now();

        const std::chrono::duration<double> elapsed = end - start;

        std::cout << "Done.\n";
        std::cout << "Elapsed: " << elapsed.count() << " seconds\n";

        std::cout << "Sanity check:\n";
        std::cout << "C[0, 0] = " << c(0, 0) << "\n";
        std::cout << "Expected: " << static_cast<float>(N) << "\n";

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Kerneloom error: " << e.what() << "\n";
        return 1;
    }
}