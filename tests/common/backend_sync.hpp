#ifndef KL_TEST_BACKEND_SYNC_HPP
#define KL_TEST_BACKEND_SYNC_HPP

#include "core/synchronize.hpp"
#include "core/device.hpp"

namespace kl::test
{

    inline void synchronizeDevice(const Device &device)
    {
        kl::synchronize(device);
    }

}

#endif // KL_TEST_BACKEND_SYNC_HPP