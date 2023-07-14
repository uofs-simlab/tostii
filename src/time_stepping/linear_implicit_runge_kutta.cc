#include <tostii/time_stepping/linear_implicit_runge_kutta.inl>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_block_vector.h>

#include <complex>

namespace tostii::TimeStepping
{
    template class LinearImplicitRungeKutta<dealii::Vector<double>>;
    template class LinearImplicitRungeKutta<dealii::Vector<std::complex<double>>>;
    template class LinearImplicitRungeKutta<dealii::Vector<std::complex<double>>, std::complex<double>>;

    template class LinearImplicitRungeKutta<dealii::BlockVector<double>>;
    template class LinearImplicitRungeKutta<dealii::BlockVector<std::complex<double>>>;
    template class LinearImplicitRungeKutta<dealii::BlockVector<std::complex<double>>, std::complex<double>>;

    template class LinearImplicitRungeKutta<dealii::PETScWrappers::MPI::Vector>;
    template class LinearImplicitRungeKutta<dealii::PETScWrappers::MPI::BlockVector>;
}
