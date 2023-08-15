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
        void rhs(
            const LA::MPI::Vector& y,
            LA::MPI::Vector& out);
        void jacobian_solve(
            const double tau,
            const LA::MPI::Vector& y,
            LA::MPI::Vector& out);
        void output_results() const;

        unsigned int solve(
            const LA::MPI::SparseMatrix& A,
            LA::MPI::Vector& x,
            const LA::MPI::Vector& b) const;

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
        LA::MPI::Vector temp;
        LA::MPI::Vector ghost_temp;

        SparsityPattern sparsity_pattern;
        LA::MPI::SparseMatrix mass_matrix;
        LA::MPI::SparseMatrix minus_A_minus_B;
        LA::MPI::SparseMatrix jacobian_C;
        LA::MPI::SparseMatrix system_matrix;

        double time;
        const double time_step;
        unsigned int timestep_number;

        const double kappa;
    };
}
