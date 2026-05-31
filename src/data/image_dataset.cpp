#include <data/image_dataset.hpp>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace kl
{

    namespace
    {

        bool is_image_file(
            const std::filesystem::path &path)
        {
            std::string extension =
                path.extension().string();

            std::transform(
                extension.begin(),
                extension.end(),
                extension.begin(),
                [](unsigned char value)
                {
                    return static_cast<char>(
                        std::tolower(value));
                });

            return extension == ".jpg" ||
                   extension == ".jpeg" ||
                   extension == ".png" ||
                   extension == ".bmp" ||
                   extension == ".ppm";
        }

        std::vector<std::string> infer_class_names(
            const std::filesystem::path &root)
        {
            std::vector<std::string> class_names;

            for (const std::filesystem::directory_entry &entry :
                 std::filesystem::directory_iterator(root))
            {
                if (entry.is_directory())
                {
                    class_names.push_back(
                        entry.path()
                            .filename()
                            .string());
                }
            }

            std::sort(
                class_names.begin(),
                class_names.end());

            return class_names;
        }

    }

    ImageDataset::ImageDataset(
        const std::filesystem::path &root)
        : class_names_(
              infer_class_names(root))
    {
        scan(root);
    }

    ImageDataset::ImageDataset(
        const std::filesystem::path &root,
        std::vector<std::string> class_names)
        : class_names_(
              std::move(class_names))
    {
        scan(root);
    }

    void ImageDataset::scan(
        const std::filesystem::path &root)
    {
        if (!std::filesystem::exists(root) ||
            !std::filesystem::is_directory(root))
        {
            throw std::runtime_error(
                "ImageDataset root directory does not exist");
        }

        if (class_names_.empty())
        {
            throw std::runtime_error(
                "ImageDataset did not find any classes");
        }

        samples_.clear();

        for (std::size_t label = 0;
             label < class_names_.size();
             ++label)
        {
            const std::filesystem::path class_directory =
                root /
                class_names_[label];

            if (!std::filesystem::exists(
                    class_directory) ||
                !std::filesystem::is_directory(
                    class_directory))
            {
                throw std::runtime_error(
                    "ImageDataset class directory does not exist: " +
                    class_directory.string());
            }

            std::vector<std::filesystem::path> paths;

            for (const std::filesystem::directory_entry &entry :
                 std::filesystem::directory_iterator(
                     class_directory))
            {
                if (entry.is_regular_file() &&
                    is_image_file(entry.path()))
                {
                    paths.push_back(
                        entry.path());
                }
            }

            std::sort(
                paths.begin(),
                paths.end());

            for (const std::filesystem::path &path :
                 paths)
            {
                samples_.push_back(
                    ImageSample{
                        path,
                        static_cast<std::int32_t>(
                            label)});
            }
        }

        if (samples_.empty())
        {
            throw std::runtime_error(
                "ImageDataset did not find any images");
        }
    }

    const std::vector<ImageSample> &
    ImageDataset::samples() const
    {
        return samples_;
    }

    const std::vector<std::string> &
    ImageDataset::class_names() const
    {
        return class_names_;
    }

    std::size_t ImageDataset::size() const
    {
        return samples_.size();
    }

    std::size_t ImageDataset::class_count() const
    {
        return class_names_.size();
    }

}