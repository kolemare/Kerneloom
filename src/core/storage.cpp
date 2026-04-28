#include <core/storage.hpp>

#include <stdexcept>

namespace kl
{

    std::string_view storage_name(Storage storage)
    {
        switch (storage)
        {
        case Storage::RowMajor:
            return "row_major";

        case Storage::ColumnMajor:
            return "column_major";

        case Storage::Tile:
            return "tile";

        default:
            throw std::invalid_argument("unknown Storage");
        }
    }

}