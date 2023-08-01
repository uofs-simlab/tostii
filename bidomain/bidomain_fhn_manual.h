#pragma once

#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/conditional_ostream.h>

#include <deal.II/lac/generic_linear_algebra.h>

namespace LA
{
    using namespace dealii::LinearAlgebraPETSc;
}

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparsity_pattern.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/distributed/tria.h>

#include <tostii/time_stepping/exact.h>
#include <tostii/time_stepping/explicit_runge_kutta.h>
#include <tostii/time_stepping/operator_split_single.h>

#include "parameters.h"
#include "fitzhugh-nagumo.h"

namespace Bidomain
{
    using namespace dealii;

    template<int dim>
    class BaseProblem
    {
    public:
        BaseProblem(const Parameters::AllParameters& param);

    protected:
        const Parameters::AllParameters param;

        MPI_Comm mpi_communicator;

        ConditionalOStream pcout;
        TimerOutput computing_timer;

        parallel::distributed::Triangulation<dim> triangulation;

        const QGauss<dim> quadrature;
    };

    template<int dim>
    class ExplicitProblem
        : virtual public BaseProblem<dim>
    {
    public:
        ExplicitProblem(const Parameters::AllParameters& param);

    protected:
        void rhs_f(
            double t,
            const LA::MPI::Vector& y,
            LA::MPI::Vector& out);

    private:
        static constexpr unsigned int
            transmembrane_component = 0,
            state_variable_component = 1;

        void assemble_system();
        void assemble_membrane_rhs(
            double t,
            const LA::MPI::Vector& y,
            LA::MPI::Vector& out);
        void solve_membrane_lhs(
            const LA::MPI::Vector& y,
            LA::MPI::Vector& out);

        DoFHandler<dim> dof_handler;

        IndexSet locally_owned_dofs;
        IndexSet locally_relevant_dofs;

        const FESystem<dim> fe;

        AffineConstraints<double> constraints;

        LA::MPI::Vector temp;

        SparsityPattern sparsity_pattern;
        LA::MPI::SparseMatrix mass_matrix;
        LA::MPI::SparseMatrix membrane_matrix;
    };

    template<int dim>
    class ImplicitProblem
        : virtual public BaseProblem<dim>
    {
    public:
        ImplicitProblem(const Parameters::AllParameters& param);

    protected:
        void step_tissue(
            double t,
            double tau,
            const LA::MPI::Vector& y,
            LA::MPI::Vector& out);

    private:
        static constexpr unsigned int
            transmembrane_component = 0,
            extracellular_component = 1;

        void assemble_system();

        DoFHandler<dim> dof_handler;

        IndexSet locally_owned_dofs;
        IndexSet locally_relevant_dofs;

        const FESystem<dim> fe;

        AffineConstraints<double> constraints;

        LA::MPI::Vector temp;

        SparsityPattern sparsity_pattern;
        LA::MPI::SparseMatrix mass_matrix;
        LA::MPI::SparseMatrix tissue_matrix;
        LA::MPI::SparseMatrix system_matrix;
    };

    template<int dim>
    class BidomainProblem
        : public ExplicitProblem<dim>
        , public ImplicitProblem<dim>
    {
    public:
        using E = ExplicitProblem<dim>;
        using I = ImplicitProblem<dim>;

        BidomainProblem(const Parameters::AllParameters& param);
        
        void run();

    private:
        static constexpr unsigned int
            transmembrane_component = 0,
            state_variable_component = 1,
            extracellular_component = 2;

        void output_results() const;

        DoFHandler<dim> dof_handler;

        LA::MPI::Vector solution;

        unsigned int timestep_number;
        const double time_step;
        double time;
    };
}
