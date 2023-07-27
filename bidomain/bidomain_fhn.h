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
#include "fitzhugh-nagumo.h"

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
        void assemble_system();
        void solve_monolithic_step(
            const double t,
            const double tau,
            const LA::MPI::Vector& y,
            LA::MPI::Vector& out);
        void assemble_membrane_rhs(
            const double t,
            const LA::MPI::Vector& y,
            LA::MPI::Vector& out);
        void solve_membrane_lhs(
            const LA::MPI::Vector& y,
            LA::MPI::Vector& out);
        void assemble_tissue_rhs(
            const LA::MPI::Vector& y,
            LA::MPI::Vector& out);
        void solve_tissue_lhs(
            const LA::MPI::Vector& y,
            LA::MPI::Vector& out);
        void assemble_Jtissue_rhs(
            const LA::MPI::Vector& y,
            LA::MPI::Vector& out);
        void solve_Jtissue_lhs(
            const double tau,
            const LA::MPI::Vector& y,
            LA::MPI::Vector& out);
        void output_results();
        
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

        LA::MPI::Vector solution;
        LA::MPI::Vector locally_owned_temp;
        LA::MPI::Vector locally_relevant_temp;

        SparsityPattern sparsity_pattern;
        LA::MPI::SparseMatrix mass_matrix;
        LA::MPI::SparseMatrix membrane_matrix;
        LA::MPI::SparseMatrix tissue_matrix;
        LA::MPI::SparseMatrix Jtissue_matrix;

        unsigned int timestep_number;
        const double time_step;
        double time;
    };
}
