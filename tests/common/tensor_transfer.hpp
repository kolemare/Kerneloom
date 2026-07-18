#ifndef KL_TEST_TENSOR_TRANSFER_HPP
#define KL_TEST_TENSOR_TRANSFER_HPP

#include "core/device.hpp"
#include "core/tensor.hpp"

namespace kl::test
{

    inline Tensor toCpu(const Tensor &tensor)
    {
        return tensor.to(Device::cpu());
    }

    inline Tensor toCuda(const Tensor &tensor)
    {
        return tensor.to(Device::cuda());
    }

    inline Tensor toRocm(const Tensor &tensor)
    {
        return tensor.to(Device::rocm());
    }

}

#endif // KL_TEST_TENSOR_TRANSFER_HPP