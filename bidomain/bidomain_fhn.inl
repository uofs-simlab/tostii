#pragma once

#include <deal.II/base/exceptions.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <tostii/time_stepping/exact.h>
#include <tostii/time_stepping/explicit_runge_kutta.h>
#include <tostii/time_stepping/operator_split.h>

#include <array>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <filesystem>

#include "bidomain_fhn.h"

namespace Bidomain
{
    template<int dim>
    const std::vector<unsigned int> BidomainProblem<dim>::explicit_blocks = {
        transmembrane_component,
        state_variable_component
    };

    template<int dim>
    const std::vector<unsigned int> BidomainProblem<dim>::implicit_blocks = {
        transmembrane_component,
        extracellular_component
    };

    template<int dim>
    BidomainProblem<dim>::BidomainProblem(
        const Parameters::AllParameters& param)
        : param(param)
        , mpi_communicator(MPI_COMM_WORLD)
        , pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        , computing_timer(mpi_communicator, pcout, TimerOutput::never, TimerOutput::wall_times)
        , triangulation(mpi_communicator)
        , dof_handler(triangulation)
        , fe(FE_Q<dim>(param.polynomial_degree), 3)
        , quadrature(param.quadrature_order)
        , timestep_number(param.initial_time_step)
        , time_step(param.final_time / param.n_time_steps)
        , time(timestep_number * time_step)
    {
        GridGenerator::hyper_cube(triangulation, 0., 1.);
        triangulation.refine_global(param.global_refinement_level);

        pcout << "Number of active cells: " << triangulation.n_global_active_cells() << std::endl;

        dof_handler.distribute_dofs(fe);

        pcout << "Numebr of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

        std::vector<unsigned int> blocks = {
            transmembrane_component,
            state_variable_component,
            extracellular_component
        };

        DoFRenumbering::Cuthill_McKee(dof_handler);
        DoFRenumbering::component_wise(dof_handler, blocks);

        std::vector<unsigned int> dofs_per_block
            = DoFTools::count_dofs_per_fe_component(dof_handler, false, blocks);
        
        IndexSet all_owned_dofs = dof_handler.locally_owned_dofs();
        locally_owned_dofs = all_owned_dofs.split_by_block(dofs_per_block);
        IndexSet all_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);
        locally_relevant_dofs = all_relevant_dofs.split_by_block(dofs_per_block);
        
        constraints.clear();
        constraints.close();

        BlockDynamicSparsityPattern bdsp(locally_relevant_dofs);
        DoFTools::make_sparsity_pattern(dof_handler, bdsp, constraints, false);
        SparsityTools::distribute_sparsity_pattern(
            bdsp,
            all_owned_dofs,
            mpi_communicator,
            all_relevant_dofs);
        
        mass_matrix.reinit(
            locally_owned_dofs,
            locally_owned_dofs,
            bdsp,
            mpi_communicator);
        implicit_mass_matrix.reinit(2, 2);
        tissue_matrix.reinit(
            locally_owned_dofs,
            locally_owned_dofs,
            bdsp,
            mpi_communicator);
        implicit_tissue_matrix.reinit(2, 2);
        implicit_system_matrix.reinit(2, 2);
        
        solution.reinit(locally_owned_dofs, mpi_communicator);
        locally_relevant_solution.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
        I_stim.reinit(locally_owned_dofs, mpi_communicator);
        tissue_rhs.reinit(2);
        tissue_rhs.block(0).reinit(solution.block(transmembrane_component));
        tissue_rhs.block(1).reinit(solution.block(extracellular_component));
        tissue_rhs.collect_sizes();

        VectorTools::interpolate(
            dof_handler,
            FitzHughNagumo::InitialValues<dim>(param),
            solution);
        locally_relevant_solution = solution;

        assemble_system();
    }

    template<int dim>
    void BidomainProblem<dim>::assemble_system()
    {
        pcout << "Assembling system matrices... " << std::flush;

        mass_matrix = 0.;
        tissue_matrix = 0.;

        FEValues<dim> fe_v(
            fe,
            quadrature,
            update_values | update_gradients | update_quadrature_points | update_JxW_values);
        
        const unsigned int dofs_per_cell = fe_v.dofs_per_cell;
        const unsigned int n_q_points = fe_v.n_quadrature_points;

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        FullMatrix<double> cell_mass(dofs_per_cell, dofs_per_cell);
        FullMatrix<double> cell_tissue(dofs_per_cell, dofs_per_cell);

        const FitzHughNagumo::IntracellularConductivity<dim> intracellular_conductivity(time, param);
        const FitzHughNagumo::ExtracellularConductivity<dim> extracellular_conductivity(time, param);

        for (const auto& cell : dof_handler.active_cell_iterators())
        {
            if (cell->is_locally_owned())
            {
                fe_v.reinit(cell);
                cell->get_dof_indices(local_dof_indices);

                cell_mass = 0.;
                cell_tissue = 0.;

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    const unsigned int c_i = fe.system_to_component_index(i).first;

                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                        const unsigned int c_j = fe.system_to_component_index(j).first;

                        for (unsigned int q = 0; q < n_q_points; ++q)
                        {
                            const Point<dim>& p = fe_v.quadrature_point(q);

                            if (c_i == transmembrane_component
                                && c_j == transmembrane_component)
                            {
                                cell_mass(i, j) += param.chi * param.Cm * fe_v.JxW(q)
                                    * fe_v.shape_value(i, q)
                                    * fe_v.shape_value(j, q);
                                cell_tissue(i, j) -= fe_v.JxW(q)
                                    * (intracellular_conductivity.value(p)
                                    * fe_v.shape_grad(i, q)
                                    * fe_v.shape_grad(j, q));
                            }
                            else if (c_i == state_variable_component
                                && c_j == state_variable_component)
                            {
                                cell_mass(i, j) += fe_v.JxW(q)
                                    * fe_v.shape_value(i, q)
                                    * fe_v.shape_value(j, q);
                            }
                            else if (c_i == extracellular_component
                                && c_j == extracellular_component)
                            {
                                Tensor<2, dim> conductivity = intracellular_conductivity.value(p);
                                conductivity += extracellular_conductivity.value(p);

                                cell_tissue(i, j) -= fe_v.JxW(q)
                                    * (conductivity
                                    * fe_v.shape_grad(i, q)
                                    * fe_v.shape_grad(j, q));
                            }
                            else if (c_i == transmembrane_component
                                    && c_j == extracellular_component
                                || c_i == extracellular_component
                                    && c_j == transmembrane_component)
                            {
                                cell_tissue(i, j) -= fe_v.JxW(q)
                                    * (intracellular_conductivity.value(p)
                                    * fe_v.shape_grad(i, q)
                                    * fe_v.shape_grad(j, q));
                            }
                        }
                    }
                }

                constraints.distribute_local_to_global(
                    cell_mass,
                    local_dof_indices,
                    mass_matrix);
                constraints.distribute_local_to_global(
                    cell_tissue,
                    local_dof_indices,
                    tissue_matrix);
            }
        }

        mass_matrix.compress(VectorOperation::add);
        tissue_matrix.compress(VectorOperation::add);

        for (unsigned int i = 0; i < implicit_blocks.size(); ++i)
        {
            for (unsigned int j = 0; j < implicit_blocks.size(); ++j)
            {
                implicit_mass_matrix.block(i, j).reinit(
                    mass_matrix.block(implicit_blocks[i], implicit_blocks[j]));
                implicit_mass_matrix.block(i, j).copy_from(
                    mass_matrix.block(implicit_blocks[i], implicit_blocks[j]));
                
                implicit_tissue_matrix.block(i, j).reinit(
                    tissue_matrix.block(implicit_blocks[i], implicit_blocks[j]));
                implicit_tissue_matrix.block(i, j).copy_from(
                    tissue_matrix.block(implicit_blocks[i], implicit_blocks[j]));
                
                implicit_system_matrix.block(i, j).reinit(
                    implicit_tissue_matrix.block(i, j));
            }
        }

        implicit_mass_matrix.collect_sizes();
        implicit_tissue_matrix.collect_sizes();
        implicit_system_matrix.collect_sizes();

        pcout << "done." << std::endl;
    }

    template<int dim>
    void BidomainProblem<dim>::step_membrane(
        const double t,
        const LA::MPI::BlockVector& y,
        LA::MPI::BlockVector& out)
    {
        TimerOutput::Scope timer_scope(computing_timer, "Membrane RHS");
        pcout << "\tAssembling membrane RHS... " << std::flush;

        ComponentMask transmembrane_mask({ true, false, false });
        VectorTools::interpolate(
            dof_handler,
            FitzHughNagumo::Stimulus<dim>(t, param),
            I_stim,
            transmembrane_mask);
        
        const LA::MPI::Vector& v = y.block(0);
        const LA::MPI::Vector& w = y.block(1);
        const LA::MPI::Vector& i_stim = I_stim.block(0);

        LA::MPI::Vector& out_v = out.block(0);
        LA::MPI::Vector& out_w = out.block(1);

        for (auto [i, last] = v.local_range(); i < last; ++i)
        {
            out_v[i] = (v[i]
                    - v[i] * v[i] * v[i] / 3.
                    - w[i]) / param.fhn.epsilon
                + i_stim[i];
            out_w[i] = (v[i]
                + param.fhn.beta
                - param.fhn.gamma * w[i]) * param.fhn.epsilon;
        }

        out.compress(VectorOperation::insert);
        constraints.distribute(out);

        pcout << "done." << std::endl;
    }

    template<int dim>
    void BidomainProblem<dim>::step_tissue(
        const double tau,
        const LA::MPI::BlockVector& y,
        LA::MPI::BlockVector& out)
    {
        TimerOutput::Scope timer_scope(computing_timer, "Tissue Step");
        pcout << "\tSolving tissue system... " << std::flush;

        double theta = 0.;
        switch (param.tissue_stepper)
        {
        case tostii::TimeStepping::BACKWARD_EULER:
            theta = 1.;
            break;
        case tostii::TimeStepping::CRANK_NICOLSON:
            theta = 0.5;
            break;
        default:
            Assert(false, ExcMessage("Must use BACKWARD_EULER or CRANK_NICOLSON time stepping"));
        }

        implicit_system_matrix.copy_from(implicit_mass_matrix);
        if (theta != 1.)
        {
            implicit_system_matrix.add((1. - theta) * tau, implicit_tissue_matrix);
        }
        implicit_system_matrix.vmult(tissue_rhs, y);

        implicit_system_matrix.add(-tau, implicit_tissue_matrix);

        PreconditionIdentity preconditioner;
        {
            PreconditionIdentity::AdditionalData additional_data;
            preconditioner.initialize(implicit_system_matrix, additional_data);
        }

        SolverControl solver_control(
            param.max_iterations * dof_handler.n_dofs(),
            param.tolerance);
        SolverGMRES<LA::MPI::BlockVector> solver(solver_control);

        solver.solve(implicit_system_matrix, out, tissue_rhs, preconditioner);
        constraints.distribute(out);

        pcout << "done in " << solver_control.last_step() << " iterations." << std::endl;
    }

    template<int dim>
    void BidomainProblem<dim>::output_results() const
    {
        TimerOutput::Scope timer_scope(computing_timer, "Output");

        DataOut<dim> data_out;

        const FitzHughNagumo::DataPostprocessors::TransmembranePart<dim> transmembrane_part;
        const FitzHughNagumo::DataPostprocessors::StateVariablePart<dim> state_variable_part;
        const FitzHughNagumo::DataPostprocessors::ExtracellularPart<dim> extracellular_part;

        data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(locally_relevant_solution, transmembrane_part);
        data_out.add_data_vector(locally_relevant_solution, state_variable_part);
        data_out.add_data_vector(locally_relevant_solution, extracellular_part);

        Vector<float> subdomain(triangulation.n_active_cells());
        for (unsigned int i = 0; i < subdomain.size(); ++i)
        {
            subdomain[i] = triangulation.locally_owned_subdomain();
        }
        data_out.add_data_vector(subdomain, "subdomain");

        data_out.build_patches();

        std::filesystem::path prefix(param.output_prefix);
        std::string filename = prefix.filename().string();
        prefix.remove_filename();

        std::filesystem::create_directories(prefix);
        data_out.write_vtu_with_pvtu_record(
            prefix.string(),
            filename,
            timestep_number,
            mpi_communicator,
            int(std::log10(param.n_time_steps)) + 1,
            8);
    }

    template<int dim>
    void BidomainProblem<dim>::run()
    {
        output_results();

        const unsigned int steps_per_output = param.n_output_files
            ? param.n_time_steps / param.n_output_files
            : param.n_time_steps + 1;
        
        using namespace tostii::TimeStepping;

        ExplicitRungeKutta<LA::MPI::BlockVector> membrane_stepper(param.membrane_stepper);
        OSOperator<LA::MPI::BlockVector> membrane_operator = {
            &membrane_stepper,
            [this](
                const double t,
                const LA::MPI::BlockVector& y,
                LA::MPI::BlockVector& out)
            {
                this->step_membrane(t, y, out);
            },
            [this](
                const double,
                const double,
                const LA::MPI::BlockVector&,
                LA::MPI::BlockVector&)
            { /* no jacobian solver required for explicit method */ }
        };

        Exact<LA::MPI::BlockVector> tissue_stepper;
        OSOperator<LA::MPI::BlockVector> tissue_operator = {
            &tissue_stepper,
            [this](
                const double,
                const LA::MPI::BlockVector&,
                LA::MPI::BlockVector&)
            { /* no function required for exact stepper */ },
            [this](
                const double,
                const double tau,
                const LA::MPI::BlockVector& y,
                LA::MPI::BlockVector& out)
            {
                this->step_tissue(tau, y, out);
            }
        };

        std::vector<OSPair<double>> stages;
        {
            std::vector<OSPair<double>> orig_stages = os_method<double>::to_os_pairs(param.os_stepper);

            for (const auto& stage : orig_stages)
            {
                if (stage.op_num == 0)
                {
                    for (unsigned int i = 0; i < 100; ++i)
                    {
                        stages.emplace_back(stage.op_num, stage.alpha / 100.);
                    }
                }
                else
                {
                    stages.push_back(stage);
                }
            }
        }

        std::vector<OSOperator<LA::MPI::BlockVector>> operators = {
            membrane_operator,
            tissue_operator
        };
        std::vector<OSMask> mask = {
            { transmembrane_component, state_variable_component },
            { transmembrane_component, extracellular_component }
        };

        OperatorSplit<LA::MPI::BlockVector> stepper(
            operators,
            stages,
            mask,
            solution);
        
        while (timestep_number < param.n_time_steps)
        {
            pcout << "Time step " << ++timestep_number << ':' << std::endl;

            time = stepper.evolve_one_time_step(time, time_step, solution);

            if (timestep_number % steps_per_output == 0)
            {
                locally_relevant_solution = solution;
                output_results();
            }
        }

        computing_timer.print_summary();
        pcout << std::endl;
    }
}
