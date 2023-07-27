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

#include <tostii/time_stepping/explicit_runge_kutta.h>
#include <tostii/time_stepping/implicit_runge_kutta.h>
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

        mass_matrix.reinit(locally_owned_dofs, locally_owned_dofs, sparsity_pattern, mpi_communicator);
        membrane_matrix.reinit(locally_owned_dofs, locally_owned_dofs, sparsity_pattern, mpi_communicator);
        tissue_matrix.reinit(locally_owned_dofs, locally_owned_dofs, sparsity_pattern, mpi_communicator);
        Jtissue_matrix.reinit(locally_owned_dofs, locally_owned_dofs, sparsity_pattern, mpi_communicator);

        solution.reinit(locally_owned_dofs, mpi_communicator);
        locally_owned_temp.reinit(locally_owned_dofs, mpi_communicator);
        locally_relevant_temp.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);

        {
            VectorTools::interpolate(
                dof_handler,
                FitzHughNagumo::InitialValues<dim>(param),
                solution);
        }
    }

    template<int dim>
    void BidomainProblem<dim>::assemble_system()
    {
        TimerOutput::Scope timer_scope(computing_timer, "Assemble System");
        pcout << "Assembling system matrices... " << std::flush;

        mass_matrix = 0.;
        membrane_matrix = 0.;
        tissue_matrix = 0.;

        FEValues<dim> fe_v(fe, quadrature,
            update_values | update_gradients | update_JxW_values);
        
        const unsigned int dofs_per_cell = fe_v.dofs_per_cell;
        const unsigned int n_q_points = fe_v.n_quadrature_points;

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        FullMatrix<double> cell_mass(dofs_per_cell, dofs_per_cell);
        FullMatrix<double> cell_membrane(dofs_per_cell, dofs_per_cell);
        FullMatrix<double> cell_tissue(dofs_per_cell, dofs_per_cell);

        for (const auto& cell : dof_handler.active_cell_iterators())
        {
            if (cell->is_locally_owned())
            {
                fe_v.reinit(cell);
                cell->get_dof_indices(local_dof_indices);

                cell_mass = 0.;
                cell_membrane = 0.;
                cell_tissue = 0.;

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    const unsigned int c_i = fe.system_to_component_index(i).first;

                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                        const unsigned int c_j = fe.system_to_component_index(j).first;

                        double M_ij = 0.;
                        double A_ij = 0.;
                        double B_ij = 0.;

                        for (unsigned int q = 0; q < n_q_points; ++q)
                        {
                            const double JxW = fe_v.JxW(q);

                            const double shape_value_product = (c_i != 2 && c_j != 2)
                                ? fe_v.shape_value_component(i, q, c_i) * fe_v.shape_value_component(j, q, c_j) * JxW
                                : 0.;
                            const double shape_grad_product = (c_i != 1 && c_j != 1)
                                ? fe_v.shape_grad_component(i, q, c_i) * fe_v.shape_grad_component(j, q, c_j) * JxW
                                : 0.;

                            if (c_i == 0 && c_j == 0)
                            {
                                M_ij += param.chi * param.Cm * shape_value_product;
                            }
                            else if (c_i == 1 && c_j == 1)
                            {
                                M_ij += shape_value_product;
                            }

                            if (c_i == 0 && c_j == 0)
                            {
                                A_ij -= param.chi / param.fhn.epsilon * shape_value_product;
                            }
                            else if (c_i == 1 && c_j == 0)
                            {
                                A_ij += param.fhn.epsilon * shape_value_product;
                            }
                            else if (c_i == 0 && c_j == 1)
                            {
                                A_ij += param.chi / param.fhn.epsilon * shape_value_product;
                            }
                            else if (c_i == 1 && c_j == 1)
                            {
                                A_ij -= param.fhn.epsilon * param.fhn.gamma * shape_value_product;
                            }

                            if ((c_i == 0 || c_i == 2) && (c_j == 0 || c_j == 2))
                            {
                                B_ij -= param.sigmai * shape_grad_product;
                            }
                            if (c_i == 2 && c_j == 2)
                            {
                                B_ij -= param.sigmae * shape_grad_product;
                            }
                        }

                        cell_mass(i, j) = M_ij;
                        cell_membrane(i, j) = A_ij;
                        cell_tissue(i, j) = B_ij;
                    }
                }

                constraints.distribute_local_to_global(cell_mass, local_dof_indices, mass_matrix);
                constraints.distribute_local_to_global(cell_membrane, local_dof_indices, membrane_matrix);
                constraints.distribute_local_to_global(cell_tissue, local_dof_indices, tissue_matrix);
            }
        }

        mass_matrix.compress(VectorOperation::add);
        membrane_matrix.compress(VectorOperation::add);
        tissue_matrix.compress(VectorOperation::add);

        {
            std::ofstream out("mass.mat");
            mass_matrix.print(out);
        }

        pcout << "done." << std::endl;
    }

    template<int dim>
    void BidomainProblem<dim>::solve_monolithic_step(
        const double,
        const double,
        const LA::MPI::Vector&,
        LA::MPI::Vector&)
    {
        Assert(false, StandardExceptions::ExcNotImplemented());
    }

    template<int dim>
    void BidomainProblem<dim>::assemble_membrane_rhs(
        const double t,
        const LA::MPI::Vector& y,
        LA::MPI::Vector& out)
    {
        TimerOutput::Scope timer_scope(computing_timer, "Membrane RHS");
        pcout << "Assembling membrane RHS... " << std::flush;

        FEValues<dim> fe_v(fe, quadrature,
            update_values | update_quadrature_points | update_JxW_values);
        
        const unsigned int dofs_per_cell = fe_v.dofs_per_cell;
        const unsigned int n_q_points = fe_v.n_quadrature_points;

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
        std::vector<Vector<double>> function_values(n_q_points, Vector<double>(3));

        Vector<double> cell_rhs(dofs_per_cell);

        FitzHughNagumo::Stimulus<dim> stimulus(t, param);

        for (const auto& cell : dof_handler.active_cell_iterators())
        {
            if (cell->is_locally_owned())
            {
                fe_v.reinit(cell);
                cell->get_dof_indices(local_dof_indices);

                cell_rhs = 0.;

                fe_v.get_function_values(y, local_dof_indices, function_values);

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    const unsigned int c_i = fe.system_to_component_index(i).first;
                    if (c_i != 0 && c_i != 1) continue;

                    double rhs_i = 0.;

                    for (unsigned int q = 0; q < n_q_points; ++q)
                    {
                        const double JxW = fe_v.JxW(q);
                        const Point<dim>& p = fe_v.quadrature_point(q);

                        if (c_i == 0)
                        {
                            rhs_i += param.chi * JxW
                                * fe_v.shape_value(i, q)
                                * (function_values[q][0]
                                        * function_values[q][0]
                                        * function_values[q][0]
                                        / param.fhn.epsilon / 3.
                                    - stimulus.value(p));
                        }
                        else if (c_i == 1)
                        {
                            rhs_i += param.fhn.epsilon * param.fhn.beta * JxW
                                * fe_v.shape_value(i, q);
                        }
                    }

                    cell_rhs[i] += rhs_i;
                }

                constraints.distribute_local_to_global(cell_rhs, local_dof_indices, out);
            }
        }

        out.compress(VectorOperation::add);

        membrane_matrix.vmult_add(out, y);

        pcout << "done." << std::endl;
    }

    template<int dim>
    void BidomainProblem<dim>::solve_membrane_lhs(
        const LA::MPI::Vector& y,
        LA::MPI::Vector& out)
    {
        TimerOutput::Scope timer_scope(computing_timer, "Membrane LHS");
        pcout << "Solving membrane LHS... " << std::flush;

        LA::MPI::PreconditionAMG preconditioner;
        {
            LA::MPI::PreconditionAMG::AdditionalData additional_data;
            preconditioner.initialize(mass_matrix, additional_data);
        }

        SolverControl solver_control(
            param.max_iterations * dof_handler.n_dofs(),
            param.tolerance);
        LA::SolverGMRES solver(solver_control, mpi_communicator);

        solver.solve(mass_matrix, out, y, preconditioner);
        constraints.distribute(out);

        pcout << "done in " << solver_control.last_step() << " iterations." << std::endl;
    }

    template<int dim>
    void BidomainProblem<dim>::assemble_tissue_rhs(
        const LA::MPI::Vector& y,
        LA::MPI::Vector& out)
    {
        TimerOutput::Scope timer_scope(computing_timer, "Tissue RHS");
        pcout << "Assembling tissue RHS... " << std::flush;

        tissue_matrix.vmult_add(out, y);

        pcout << "done." << std::endl;
    }

    template<int dim>
    void BidomainProblem<dim>::solve_tissue_lhs(
        const LA::MPI::Vector& y,
        LA::MPI::Vector& out)
    {
        TimerOutput::Scope timer_scope(computing_timer, "Tissue LHS");
        pcout << "Solving tissue LHS... " << std::flush;

        LA::MPI::PreconditionAMG preconditioner;
        {
            LA::MPI::PreconditionAMG::AdditionalData additional_data;
            preconditioner.initialize(mass_matrix, additional_data);
        }

        SolverControl solver_control(
            param.max_iterations * dof_handler.n_dofs(),
            param.tolerance);
        LA::SolverGMRES solver(solver_control, mpi_communicator);

        solver.solve(mass_matrix, out, y, preconditioner);
        constraints.distribute(out);

        pcout << "done in " << solver_control.last_step() << " iterations." << std::endl;
    }

    template<int dim>
    void BidomainProblem<dim>::assemble_Jtissue_rhs(
        const LA::MPI::Vector& y,
        LA::MPI::Vector& out)
    {
        TimerOutput::Scope timer_scope(computing_timer, "JTissue RHS");
        pcout << "Assembling Jtissue RHS... " << std::flush;

        mass_matrix.vmult_add(out, y);

        pcout << "done." << std::endl;
    }

    template<int dim>
    void BidomainProblem<dim>::solve_Jtissue_lhs(
        const double tau,
        const LA::MPI::Vector& y,
        LA::MPI::Vector& out)
    {
        TimerOutput::Scope timer_scope(computing_timer, "JTissue LHS");
        pcout << "Solving Jtissue LHS... " << std::flush;

        Jtissue_matrix.copy_from(mass_matrix);
        Jtissue_matrix.add(-tau, tissue_matrix);

        LA::MPI::PreconditionAMG preconditioner;
        {
            LA::MPI::PreconditionAMG::AdditionalData additional_data;
            preconditioner.initialize(Jtissue_matrix, additional_data);
        }

        SolverControl solver_control(
            param.max_iterations * dof_handler.n_dofs(),
            param.tolerance);
        LA::SolverGMRES solver(solver_control, mpi_communicator);

        solver.solve(Jtissue_matrix, out, y, preconditioner);
        constraints.distribute(out);

        pcout << "done in " << solver_control.last_step() << " iterations." << std::endl;
    }

    template<int dim>
    void BidomainProblem<dim>::output_results()
    {
        DataOut<dim> data_out;
        
        locally_relevant_temp = solution;

        const FitzHughNagumo::DataPostprocessors::TransmembranePart<dim> transmembrane_part;
        const FitzHughNagumo::DataPostprocessors::StateVariablePart<dim> state_variable_part;
        const FitzHughNagumo::DataPostprocessors::ExtracellularPart<dim> extracellular_part;

        data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(locally_relevant_temp, transmembrane_part);
        data_out.add_data_vector(locally_relevant_temp, state_variable_part);
        data_out.add_data_vector(locally_relevant_temp, extracellular_part);

        Vector<float> subdomain(triangulation.n_active_cells());
        for (unsigned int i = 0; i < subdomain.size(); ++i)
        {
            subdomain[i] = triangulation.locally_owned_subdomain();
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
        
        using namespace tostii::TimeStepping;

        ExplicitRungeKutta<LA::MPI::Vector> membrane_stepper(param.membrane_stepper);
        OSOperator<LA::MPI::Vector> membrane_operator = {
            &membrane_stepper,
            [this](
                const double t,
                const LA::MPI::Vector& y,
                LA::MPI::Vector& out)
            {
                this->locally_relevant_temp = y;
                this->locally_owned_temp = 0.;
                this->assemble_membrane_rhs(t, this->locally_relevant_temp, this->locally_owned_temp);
                this->solve_membrane_lhs(this->locally_owned_temp, out);
            },
            [](
                const double,
                const double,
                const LA::MPI::Vector&,
                LA::MPI::Vector&)
            { /* no jacobian solver required for explicit method */ }
        };

        ImplicitRungeKutta<LA::MPI::Vector> tissue_stepper(param.tissue_stepper);
        OSOperator<LA::MPI::Vector> tissue_operator = {
            &tissue_stepper,
            [this](
                const double,
                const LA::MPI::Vector& y,
                LA::MPI::Vector& out)
            {
                this->locally_relevant_temp = y;
                this->locally_owned_temp = 0.;
                this->assemble_tissue_rhs(this->locally_relevant_temp, this->locally_owned_temp);
                this->solve_tissue_lhs(this->locally_owned_temp, out);
            },
            [this](
                const double,
                const double tau,
                const LA::MPI::Vector& y,
                LA::MPI::Vector& out)
            {
                this->locally_relevant_temp = y;
                this->locally_owned_temp = 0.;
                this->assemble_Jtissue_rhs(this->locally_relevant_temp, this->locally_owned_temp);
                this->solve_Jtissue_lhs(tau, this->locally_owned_temp, out);
            }
        };

        std::vector<OSOperator<LA::MPI::Vector>> operators = { membrane_operator, tissue_operator };
        OperatorSplitSingle<LA::MPI::Vector> stepper(operators, param.os_stepper, solution);

        while (timestep_number < param.n_time_steps)
        {
            ++timestep_number;

            pcout << "Time step " << timestep_number << ":" << std::endl;

            time = stepper.evolve_one_time_step(time, time_step, solution);

            if (timestep_number % steps_per_output == 0)
            {
                output_results();
            }
        }

        computing_timer.print_summary();
        pcout << std::endl;
    }
}
