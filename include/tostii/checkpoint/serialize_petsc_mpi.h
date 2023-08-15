#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_block_vector.h>

namespace boost::serialization
{
    /**
     * Serialize a PETSc vector.
     * 
     * PETSc vectors have a range of locally owned values;
     * this function simply saves those values
     * (i.e., the actual indices, the ghost indices/elements,
     * and the vector's last operation are not stored).
     * As such, when loading a PETSc vector stored by this function,
     * its owned/ghost indices should be set before calling this function.
     */
    template<typename Archive>
    void serialize(
        Archive& ar,
        dealii::PETScWrappers::MPI::Vector& v,
        const unsigned int version);
    
    /**
     * Serialize a PETSc block vector.
     * 
     * Each block of the input vector is saved
     * as described in this function's overload
     * for `PETScWrappers::MPI::Vector`.
     * As in that overload, the number of blocks in \p v
     * should be set before calling this function,
     * as well as the owned/ghosted indices of the blocks.
     */
    template<typename Archive>
    void serialize(
        Archive& ar,
        dealii::PETScWrappers::MPI::BlockVector& v,
        const unsigned int version);
}
