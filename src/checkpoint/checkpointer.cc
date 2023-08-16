#include <tostii/checkpoint/checkpointer.inl>

namespace tostii
{
    template class Checkpointer<
        boost::archive::text_iarchive,
        boost::archive::text_oarchive>;
    template class Checkpointer<
        boost::archive::binary_iarchive,
        boost::archive::binary_oarchive>;
}
