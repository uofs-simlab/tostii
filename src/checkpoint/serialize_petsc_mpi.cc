#include <tostii/checkpoint/serialize_petsc_mpi.inl>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

namespace boost::serialization
{
    template void serialize(
        boost::archive::text_iarchive&,
        dealii::PETScWrappers::MPI::Vector&,
        const unsigned int);
    template void serialize(
        boost::archive::text_oarchive&,
        dealii::PETScWrappers::MPI::Vector&,
        const unsigned int);
    template void serialize(
        boost::archive::binary_iarchive&,
        dealii::PETScWrappers::MPI::Vector&,
        const unsigned int);
    template void serialize(
        boost::archive::binary_oarchive&,
        dealii::PETScWrappers::MPI::Vector&,
        const unsigned int);

    template void serialize(
        boost::archive::text_iarchive&,
        dealii::PETScWrappers::MPI::BlockVector&,
        const unsigned int);
    template void serialize(
        boost::archive::text_oarchive&,
        dealii::PETScWrappers::MPI::BlockVector&,
        const unsigned int);
    template void serialize(
        boost::archive::binary_iarchive&,
        dealii::PETScWrappers::MPI::BlockVector&,
        const unsigned int);
    template void serialize(
        boost::archive::binary_oarchive&,
        dealii::PETScWrappers::MPI::BlockVector&,
        const unsigned int);
}
