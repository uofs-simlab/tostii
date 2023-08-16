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
        static constexpr unsigned int
            transmembrane_component = 0,
            state_variable_component = 1,
            extracellular_component = 2;

        BidomainProblem(
            const Parameters::AllParameters& param);
        
        void run();

    private:
        void assemble_system();
        void step_membrane(
            const double t,
            const LA::MPI::BlockVector& y,
            LA::MPI::BlockVector& out);
        void step_tissue(
            const double tau,
            const LA::MPI::BlockVector& y,
            LA::MPI::BlockVector& out);
        void output_results() const;

        static const std::vector<unsigned int> explicit_blocks;
        static const std::vector<unsigned int> implicit_blocks;

        const Parameters::AllParameters param;

        const MPI_Comm mpi_communicator;
        ConditionalOStream pcout;

        mutable TimerOutput computing_timer;

        parallel::distributed::Triangulation<dim> triangulation;
        DoFHandler<dim> dof_handler;

        const FESystem<dim> fe;
        const QGauss<dim> quadrature;

        std::vector<IndexSet> locally_owned_dofs;
        std::vector<IndexSet> locally_relevant_dofs;

        AffineConstraints<double> constraints;

        LA::MPI::BlockVector solution;
        LA::MPI::BlockVector locally_relevant_solution;
        LA::MPI::BlockVector I_stim;
        LA::MPI::BlockVector tissue_rhs;

        LA::MPI::BlockSparseMatrix mass_matrix;
        LA::MPI::BlockSparseMatrix implicit_mass_matrix;
        LA::MPI::BlockSparseMatrix tissue_matrix;
        LA::MPI::BlockSparseMatrix implicit_tissue_matrix;
        LA::MPI::BlockSparseMatrix implicit_system_matrix;

        unsigned int timestep_number;
        const double time_step;
        double time;
    };
}
