#include <tostii/time_stepping/operator_split_single.inl>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_block_vector.h>

namespace tostii::TimeStepping
{
    template class OperatorSplitSingle<dealii::Vector<double>>;
    template class OperatorSplitSingle<dealii::Vector<std::complex<double>>>;
    template class OperatorSplitSingle<dealii::Vector<std::complex<double>>, std::complex<double>>;

    template class OperatorSplitSingle<dealii::BlockVector<double>>;
    template class OperatorSplitSingle<dealii::BlockVector<std::complex<double>>>;
    template class OperatorSplitSingle<dealii::BlockVector<std::complex<double>>, std::complex<double>>;

    template class OperatorSplitSingle<dealii::PETScWrappers::MPI::Vector>;

    template class OperatorSplitSingle<dealii::PETScWrappers::MPI::BlockVector>;
}
