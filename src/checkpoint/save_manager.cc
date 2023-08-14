#include <tostii/checkpoint/save_manager.inl>

namespace tostii
{
    template class SaveManager<
        boost::archive::text_iarchive,
        boost::archive::text_oarchive>;
    template class SaveManager<
        boost::archive::binary_iarchive,
        boost::archive::binary_oarchive>;
}
