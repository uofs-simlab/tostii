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

#include <deal.II/sundials/kinsol.h>

#include <tostii/time_stepping/exact.h>
#include <tostii/time_stepping/linear_implicit_runge_kutta.h>
#include <tostii/time_stepping/operator_split_single.h>

#include <iostream>
#include <fstream>
#include <filesystem>

#include "bidomain_godunov.h"

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
        , fe(FE_Q<dim>(param.polynomial_degree), 2)
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
        locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

        constraints.clear();
        constraints.close();

        {
            DynamicSparsityPattern dsp(locally_relevant_dofs);
            DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
            SparsityTools::distribute_sparsity_pattern(dsp, locally_owned_dofs, mpi_communicator, locally_relevant_dofs);
            sparsity_pattern.copy_from(dsp);
        }

        mass_matrix.reinit(locally_owned_dofs, locally_owned_dofs, sparsity_pattern, mpi_communicator);
        system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, sparsity_pattern, mpi_communicator);

        old_solution.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
        solution.reinit(locally_owned_dofs, mpi_communicator);

        {
            PrescribedData::ExactSolution<dim> initial_conditions(time, param);
            VectorTools::interpolate(
                dof_handler,
                initial_conditions,
                solution);
        }
    }

    template<int dim>
    void BidomainProblem<dim>::assemble_system()
    {
        TimerOutput::Scope timer_scope(computing_timer, "Assemble System");
        pcout << "Assembling system matrices... " << std::flush;

        mass_matrix = 0.;
        system_matrix = 0.;

        FEValues<dim> fe_v(fe, quadrature,
            update_values | update_gradients | update_quadrature_points | update_JxW_values);
        
        const unsigned int dofs_per_cell = fe_v.dofs_per_cell;
        const unsigned int n_q_points = fe_v.n_quadrature_points;

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
        FullMatrix<double> cell_system_matrix(dofs_per_cell, dofs_per_cell);

        for (const auto& cell : dof_handler.active_cell_iterators())
        {
            if (cell->is_locally_owned())
            {
                fe_v.reinit(cell);
                cell->get_dof_indices(local_dof_indices);

                cell_mass_matrix = 0.;
                cell_system_matrix = 0.;

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    const unsigned int c_i = fe.system_to_component_index(i).first;

                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                        const unsigned int c_j = fe.system_to_component_index(j).first;

                        double A_ij = 0.;
                        double M_ij = 0.;

                        for (unsigned int q = 0; q < n_q_points; ++q)
                        {
                            const double JxW = fe_v.JxW(q);

                            if (c_i == 0 && c_j == 0)
                            {
                                M_ij += param.chi * param.Cm
                                    * fe_v.shape_value_component(i, q, c_i)
                                    * fe_v.shape_value_component(j, q, c_j)
                                    * JxW;
                                
                                A_ij -= param.chi / param.Rm
                                    * fe_v.shape_value_component(i, q, c_i)
                                    * fe_v.shape_value_component(j, q, c_j)
                                    * JxW;
                            }

                            if (c_i == 1 && c_j == 1)
                            {
                                A_ij -= (param.sigmai + param.sigmae)
                                    * fe_v.shape_grad_component(i, q, c_i)
                                    * fe_v.shape_grad_component(j, q, c_j)
                                    * JxW;
                            }
                            else
                            {
                                A_ij -= param.sigmai
                                    * fe_v.shape_grad_component(i, q, c_i)
                                    * fe_v.shape_grad_component(j, q, c_j)
                                    * JxW;
                            }
                        }

                        cell_mass_matrix(i, j) = M_ij / time_step;
                        cell_system_matrix(i, j) = cell_mass_matrix(i, j) - A_ij;
                    }
                }

                constraints.distribute_local_to_global(cell_mass_matrix, local_dof_indices, mass_matrix);
                constraints.distribute_local_to_global(cell_system_matrix, local_dof_indices, system_matrix);
            }
        }

        mass_matrix.compress(VectorOperation::add);
        system_matrix.compress(VectorOperation::add);

        pcout << "done" << std::endl;
    }

    template<int dim>
    void BidomainProblem<dim>::assemble_membrane_rhs(
        const double t,
        const double tau,
        const LA::MPI::Vector& W,
        LA::MPI::Vector& rhs)
    {
        TimerOutput::Scope timer_scope(computing_timer, "Assemble RHS");
        pcout << "\tAssembling RHS... " << std::flush;

        PrescribedData::TransmembraneRightHandSide<dim> transmembrane_rhs(t + tau, param);
        PrescribedData::ExtracellularRightHandSide<dim> extracellular_rhs;

        rhs = 0.;

        FEValues<dim> fe_v(fe, quadrature,
            update_values | update_quadrature_points | update_JxW_values);
        
        const unsigned int dofs_per_cell = fe_v.dofs_per_cell;
        const unsigned int n_q_points = fe_v.n_quadrature_points;

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        Vector<double> cell_rhs(dofs_per_cell);

        for (const auto& cell : dof_handler.active_cell_iterators())
        {
            if (cell->is_locally_owned())
            {
                fe_v.reinit(cell);
                cell->get_dof_indices(local_dof_indices);

                cell_rhs = 0.;

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    const unsigned int c_i = fe.system_to_component_index(i).first;

                    double f_i = 0.;

                    for (unsigned int q = 0; q < n_q_points; ++q)
                    {
                        const double JxW = fe_v.JxW(q);
                        const Point<dim>& p = fe_v.quadrature_point(q);

                        if (c_i == 0)
                        {
                            f_i += transmembrane_rhs.value(p)
                                * fe_v.shape_value_component(i, q, c_i)
                                * JxW;
                        }
                        else
                        {
                            f_i += extracellular_rhs.value(p)
                                * fe_v.shape_value_component(i, q, c_i)
                                * JxW;
                        }
                    }

                    cell_rhs[i] += f_i;
                }

                constraints.distribute_local_to_global(cell_rhs, local_dof_indices, rhs);
            }
        }

        rhs.compress(VectorOperation::add);

        mass_matrix.vmult_add(rhs, W);

        pcout << "done" << std::endl;
    }

    template<int dim>
    void BidomainProblem<dim>::solve_tissue_system(
        const LA::MPI::Vector& rhs,
        LA::MPI::Vector& W)
    {
        TimerOutput::Scope timer_scope(computing_timer, "Solve Tissue System");
        pcout << "\tSolving tissue system... " << std::flush;

        LA::MPI::PreconditionAMG preconditioner;
        {
            LA::MPI::PreconditionAMG::AdditionalData additional_data;
            preconditioner.initialize(system_matrix, additional_data);
        }

        SolverControl solver_control(param.max_iterations * dof_handler.n_dofs(), param.tolerance);
        LA::SolverGMRES solver(solver_control, mpi_communicator);

        solver.solve(system_matrix, W, rhs, preconditioner);
        constraints.distribute(solution);

        pcout << "done in " << solver_control.last_step() << " iterations" << std::endl;
    }

    template<int dim>
    void BidomainProblem<dim>::compute_errors() const
    {
        pcout << "[t = " << time << "] " << std::flush;

        Vector<double> cellwise_errors(triangulation.n_active_cells());

        LA::MPI::Vector current_solution(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
        current_solution = solution;

        VectorTools::integrate_difference(
            dof_handler,
            current_solution,
            PrescribedData::ExactSolution<dim>(time, param),
            cellwise_errors,
            QGauss<dim>(param.quadrature_order + 1),
            VectorTools::L2_norm);
        const double error = VectorTools::compute_global_error(
            triangulation,
            cellwise_errors,
            VectorTools::L2_norm);
        
        pcout << "Error = " << error << std::endl;
    }

    template<int dim>
    void BidomainProblem<dim>::output_results() const
    {
        DataOut<dim> data_out;

        LA::MPI::Vector current_solution(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
        current_solution = solution;

        data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(current_solution, "w");

        Vector<float> subdomain(triangulation.n_active_cells());
        for (unsigned int i = 0; i < subdomain.size(); ++i)
        {
            subdomain(i) = triangulation.locally_owned_subdomain();
        }
        data_out.add_data_vector(subdomain, "subdomain");

        data_out.build_patches();

        std::filesystem::path path(param.output_prefix);
        std::string filename = path.filename().string();
        std::string prefix = path.remove_filename().string();

        data_out.write_vtu_with_pvtu_record(
            prefix,
            filename,
            timestep_number,
            mpi_communicator,
            5,
            8);
    }

    template<int dim>
    void BidomainProblem<dim>::run()
    {
        setup_system();
        output_results();
        assemble_system();

        const unsigned int steps_per_output = param.n_output_files
            ? param.n_time_steps / param.n_output_files
            : param.n_time_steps + 1;

        /*
        using namespace tostii::TimeStepping;

        Exact<LA::MPI::Vector> membrane_stepper;
        OSOperator<LA::MPI::Vector> membrane_operator = {
            &membrane_stepper,
            [](
                const double,
                const LA::MPI::Vector& W,
                LA::MPI::Vector& rhs) { rhs = W; },
            [this](
                const double t,
                const double tau,
                const LA::MPI::Vector& rhs,
                LA::MPI::Vector& W) { this->assemble_membrane_rhs(t, tau, rhs, W); }
        };

        LinearImplicitRungeKutta<LA::MPI::Vector> tissue_stepper(BACKWARD_EULER);
        OSOperator<LA::MPI::Vector> tissue_operator = {
            &tissue_stepper,
            [](
                const double,
                const LA::MPI::Vector& W,
                LA::MPI::Vector& rhs) { rhs = W; },
            [this](
                const double,
                const double,
                const LA::MPI::Vector& rhs,
                LA::MPI::Vector& W) { this->solve_tissue_system(rhs, W); }
        };

        auto method = os_method<double>::GODUNOV;
        std::vector<OSOperator<LA::MPI::Vector>> operators = { membrane_operator, tissue_operator };
        OperatorSplitSingle<LA::MPI::Vector> stepper(operators, method, solution);
        */

        LA::MPI::Vector rhs(locally_owned_dofs, mpi_communicator);

        while (timestep_number < param.n_time_steps)
        {
            ++timestep_number;

            old_solution = solution;

            pcout << "Time step " << timestep_number << ":" << std::endl;

            assemble_membrane_rhs(time, time_step, solution, rhs);
            solve_tissue_system(rhs, solution);
            time += time_step;

            /*
            time = stepper.evolve_one_time_step(time, time_step, solution);
            */

            compute_errors();

            if (timestep_number % steps_per_output == 0)
            {
                output_results();
            }
        }

        computing_timer.print_summary();
        pcout << std::endl;
    }
}
