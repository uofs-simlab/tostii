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

#include "parameters.h"
#include "prescribed_data.h"

namespace Bidomain
{
    using namespace dealii;

    template<int dim>
    class BidomainProblem
    {
    public:
        BidomainProblem(const Parameters::AllParameters& param);

        void run();

    private:
        void setup_system();
        void compute_residual(
            const LA::MPI::Vector& W,
            LA::MPI::Vector& res);
        void assemble_cell_term(
            const typename DoFHandler<dim>::active_cell_iterator& cell,
            FEValues<dim>& fe_v,
            std::vector<types::global_dof_index>& local_dof_indices,
            const LA::MPI::Vector& W,
            LA::MPI::Vector& res);
        void prescribed_residual(
            LA::MPI::Vector& res);
        void prescribed_cell_term(
            const typename DoFHandler<dim>::active_cell_iterator& cell,
            FEValues<dim>& fe_v,
            std::vector<types::global_dof_index>& local_dof_indices,
            const LA::MPI::Vector& W,
            LA::MPI::Vector& res);
        void solve(
            const LA::MPI::Vector& W,
            LA::MPI::Vector& delta_W,
            double tolerance);
        void compute_errors() const;
        void output_results() const;

        const Parameters::AllParameters param;

        MPI_Comm mpi_communicator;

        ConditionalOStream pcout;
        TimerOutput computing_timer;

        parallel::distributed::Triangulation<dim> triangulation;
        DoFHandler<dim> dof_handler;

        IndexSet locally_owned_dofs;
        IndexSet locally_relevant_dofs;

        const FESystem<dim> fe;
        const QGauss<dim> quadrature;

        AffineConstraints<double> constraints;

        LA::MPI::Vector old_solution;
        LA::MPI::Vector solution;

        SparsityPattern sparsity_pattern;
        LA::MPI::SparseMatrix jacobian_matrix;

        unsigned int timestep_number;
        const double time_step;
        double time;
    };
}
