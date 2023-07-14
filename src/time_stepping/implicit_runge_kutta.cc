#include <tostii/time_stepping/implicit_runge_kutta.inl>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_block_vector.h>

#include <complex>

namespace tostii::TimeStepping
{
    template class ImplicitRungeKutta<dealii::Vector<double>>;
    template class ImplicitRungeKutta<dealii::Vector<std::complex<double>>>;
    template class ImplicitRungeKutta<dealii::Vector<std::complex<double>>, std::complex<double>>;

    template class ImplicitRungeKutta<dealii::BlockVector<double>>;
    template class ImplicitRungeKutta<dealii::BlockVector<std::complex<double>>>;
    template class ImplicitRungeKutta<dealii::BlockVector<std::complex<double>>, std::complex<double>>;

    template class ImplicitRungeKutta<dealii::PETScWrappers::MPI::Vector>;
    template class ImplicitRungeKutta<dealii::PETScWrappers::MPI::BlockVector>;
}
