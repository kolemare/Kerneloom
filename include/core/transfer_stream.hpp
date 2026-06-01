#ifndef KL_TRANSFER_STREAM_HPP
#define KL_TRANSFER_STREAM_HPP

#include <core/device.hpp>
#include <core/tensor.hpp>

namespace kl
{

    class TransferStream
    {
    public:
        explicit TransferStream(
            Device device);

        ~TransferStream();

        TransferStream(
            const TransferStream &) = delete;

        TransferStream &operator=(
            const TransferStream &) = delete;

        TransferStream(
            TransferStream &&) = delete;

        TransferStream &operator=(
            TransferStream &&) = delete;

        void copy_async(
            Tensor &destination,
            const Tensor &source);

        void synchronize();

        Device device() const;

    private:
        Device device_;
        void *handle_ = nullptr;
    };

}

#endif // KL_TRANSFER_STREAM_HPP