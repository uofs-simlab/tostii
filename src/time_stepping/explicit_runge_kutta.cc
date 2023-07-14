#include <tostii/time_stepping/explicit_runge_kutta.inl>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/petsc_block_vector.h>

#include <complex>

namespace tostii::TimeStepping
{
    template class ExplicitRungeKutta<dealii::Vector<double>>;
    template class ExplicitRungeKutta<dealii::Vector<std::complex<double>>>;
    template class ExplicitRungeKutta<dealii::Vector<std::complex<double>>, std::complex<double>>;

    template class ExplicitRungeKutta<dealii::BlockVector<double>>;
    template class ExplicitRungeKutta<dealii::BlockVector<std::complex<double>>>;
    template class ExplicitRungeKutta<dealii::BlockVector<std::complex<double>>, std::complex<double>>;

    template class ExplicitRungeKutta<dealii::PETScWrappers::MPI::Vector>;

    template class ExplicitRungeKutta<dealii::PETScWrappers::MPI::BlockVector>;
}
