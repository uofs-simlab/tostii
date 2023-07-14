#include <tostii/time_stepping/exact.inl>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_block_vector.h>

#include <complex>

namespace tostii::TimeStepping
{
    template class Exact<dealii::Vector<double>>;
    template class Exact<dealii::Vector<std::complex<double>>>;
    template class Exact<dealii::Vector<std::complex<double>>, std::complex<double>>;

    template class Exact<dealii::BlockVector<double>>;
    template class Exact<dealii::BlockVector<std::complex<double>>>;
    template class Exact<dealii::BlockVector<std::complex<double>>, std::complex<double>>;

    template class Exact<dealii::PETScWrappers::MPI::Vector>;

    template class Exact<dealii::PETScWrappers::MPI::BlockVector>;
}
