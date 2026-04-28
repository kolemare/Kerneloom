#ifndef KL_STORAGE_HPP
#define KL_STORAGE_HPP

#include <string_view>

namespace kl
{

    enum class Storage
    {
        RowMajor,
        ColumnMajor,
        Tile
    };

    std::string_view storage_name(Storage storage);

}

#endif // KL_STORAGE_HPP