#ifndef KL_IMAGE_DATASET_HPP
#define KL_IMAGE_DATASET_HPP

#include <data/image_sample.hpp>

#include <filesystem>
#include <string>
#include <vector>

namespace kl
{

    class ImageDataset
    {
    public:
        explicit ImageDataset(
            const std::filesystem::path &root);

        ImageDataset(
            const std::filesystem::path &root,
            std::vector<std::string> class_names);

        const std::vector<ImageSample> &samples() const;
        const std::vector<std::string> &class_names() const;

        std::size_t size() const;
        std::size_t class_count() const;

    private:
        void scan(
            const std::filesystem::path &root);

    private:
        std::vector<ImageSample> samples_;
        std::vector<std::string> class_names_;
    };

}

#endif // KL_IMAGE_DATASET_HPP