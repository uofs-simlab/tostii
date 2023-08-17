#pragma once

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>

#include <deal.II/lac/generic_linear_algebra.h>

namespace LA
{
    using namespace dealii::LinearAlgebraPETSc;
}

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/numerics/vector_tools.h>

#include <deal.II/distributed/tria.h>

#include <Sacado.hpp>

#include <fstream>
#include <iostream>

#include <tostii/checkpoint/checkpointer.h>

#include "parameters.h"

namespace NSE
{
    template<int dim>
    class NonlinearSchroedingerEquation
        : public tostii::BinaryCheckpointer
    {
    public:
        NonlinearSchroedingerEquation(
            const Parameters::AllParameters& param);
        
        void run();

    private:
        void assemble_system();
        void old_residual();
        void residual(
            const LA::MPI::Vector& y,
            LA::MPI::Vector& out);
        void setup_jacobian();
        void jacobian_solve(
            const LA::MPI::Vector& y,
            LA::MPI::Vector& out,
            const double tolerance);
        void output_results() const;

        friend class boost::serialization::access;
        void serialize(
            boost::archive::binary_iarchive& ar,
            const unsigned int version) override;
        void serialize(
            boost::archive::binary_oarchive& ar,
            const unsigned int version) override;

        const Parameters::AllParameters param;

        const MPI_Comm mpi_communicator;
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
        LA::MPI::Vector ghost_solution;
        bool old_solution_residual_ready;
        LA::MPI::Vector old_solution_residual;
        LA::MPI::Vector temp;

        SparsityPattern sparsity_pattern;
        /** M + h/2 (A + B) */
        LA::MPI::SparseMatrix stiffness_matrix;
        /** -M + h/2 (A + B) */
        LA::MPI::SparseMatrix old_stiffness_matrix;
        /** h/2 J[C] */
        LA::MPI::SparseMatrix jacobian_C;
        /** J[R] */
        LA::MPI::SparseMatrix jacobian_matrix;

        double time;
        const double time_step;
        unsigned int timestep_number;

        const double kappa;
    };
}
