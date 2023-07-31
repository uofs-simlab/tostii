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
        static constexpr unsigned int
            explicit_transmembrane_component = 0,
            explicit_state_variable_component = 1;
        static constexpr unsigned int
            implicit_transmembrane_component = 0,
            implicit_extracellular_component = 1;

        BidomainProblem(const Parameters::AllParameters& param);
        
        void run();

    private:
        void setup_system();
        void assemble_system();
        void assemble_membrane_rhs(
            const double t,
            const LA::MPI::BlockVector& y,
            LA::MPI::BlockVector& out);
        void solve_membrane_lhs(
            const LA::MPI::BlockVector& y,
            LA::MPI::BlockVector& out);
        void step_tissue(
            const double tau,
            const LA::MPI::BlockVector& y,
            LA::MPI::BlockVector& out);
        void output_results();

        constexpr types::global_dof_index global_to_component_index(const types::global_dof_index i) const;

        const Parameters::AllParameters param;

        MPI_Comm mpi_communicator;

        ConditionalOStream pcout;
        TimerOutput computing_timer;

        parallel::distributed::Triangulation<dim> triangulation;
        DoFHandler<dim> dof_handler;

        std::vector<IndexSet> locally_owned_dofs;
        std::vector<IndexSet> locally_relevant_dofs;
        std::vector<types::global_dof_index> dofs_per_block;
        std::vector<std::vector<types::global_dof_index>> component_local_dofs;

        const FESystem<dim> fe;
        const QGauss<dim> quadrature;

        AffineConstraints<double> constraints;

        LA::MPI::BlockVector solution;
        LA::MPI::BlockVector relevant_solution;
        LA::MPI::BlockVector membrane_temp;
        LA::MPI::BlockVector relevant_membrane_temp;
        LA::MPI::BlockVector membrane_rhs;
        LA::MPI::BlockVector tissue_rhs;

        LA::MPI::BlockSparseMatrix explicit_mass_matrix;
        LA::MPI::BlockSparseMatrix implicit_mass_matrix;
        LA::MPI::BlockSparseMatrix membrane_matrix;
        LA::MPI::BlockSparseMatrix tissue_matrix;
        LA::MPI::BlockSparseMatrix implicit_matrix;

        unsigned int timestep_number;
        const double time_step;
        double time;
    };
}
