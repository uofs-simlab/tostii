#pragma once

#include <deal.II/base/exceptions.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <tostii/time_stepping/exact.h>
#include <tostii/time_stepping/linear_implicit_runge_kutta.h>
#include <tostii/time_stepping/operator_split_single.h>

#include <iostream>
#include <fstream>
#include <filesystem>

#include "bidomain_fhn.h"

namespace Bidomain
{
    template<int dim>
    BidomainProblem<dim>::BidomainProblem(const Parameters::AllParameters& param)
        : param(param)
        , mpi_communicator(MPI_COMM_WORLD)
        , pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        , computing_timer(mpi_communicator, pcout, TimerOutput::never, TimerOutput::wall_times)
        , triangulation(mpi_communicator)
        , dof_handler(triangulation)
        , fe(FE_Q<dim>(param.polynomial_degree), 3)
        , quadrature(param.quadrature_order)
        , timestep_number(param.initial_time_step)
        , time_step(1. / param.n_time_steps)
        , time(timestep_number * time_step)
    { }

    template<int dim>
    void BidomainProblem<dim>::setup_system()
    {
        GridGenerator::hyper_cube(triangulation, 0., 1.);
        triangulation.refine_global(param.global_refinement_level);

        pcout << "Number of active cells: " << triangulation.n_active_cells() << std::endl;

        dof_handler.distribute_dofs(fe);

        pcout << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

        locally_owned_dofs = dof_handler.locally_owned_dofs();
        DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

        constraints.clear();
        constraints.close();

        {
            DynamicSparsityPattern dsp(locally_relevant_dofs);
            DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
            SparsityTools::distribute_sparsity_pattern(dsp, locally_owned_dofs, mpi_communicator, locally_relevant_dofs);
            sparsity_pattern.copy_from(dsp);
        }

        mass_matrix.reinit(sparsity_pattern);
        tissue_matrix.reinit(sparsity_pattern);

        solution.reinit(locally_owned_dofs, mpi_communicator);

        {
            // TODO initial conditions
        }
    }
}
