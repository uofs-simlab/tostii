#pragma once

#include <deal.II/base/exceptions.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <tostii/time_stepping/explicit_runge_kutta.h>
#include <tostii/time_stepping/implicit_runge_kutta.h>
#include <tostii/time_stepping/operator_split_single.h>

#include <array>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <filesystem>

#include "bidomain_fhn.h"

namespace Bidomain
{
    template<int dim>
    constexpr types::global_dof_index BidomainProblem<dim>::local_to_component_index(const types::global_dof_index i) const
    {
        const unsigned int c_i = fe.system_to_component_index(i).first;

        switch (c_i)
        {
        case transmembrane_component:
            return i;
        case state_variable_component:
            return i - dofs_per_block[transmembrane_component];
        case extracellular_component:
            return i - dofs_per_block[transmembrane_component] - dofs_per_block[state_variable_component];
        default:
            /* guaranteed to throw */
            AssertIndexRange(c_i, 3);
            return 0;
        }
    }

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

        std::vector<unsigned int> blocks = {
            transmembrane_component,
            state_variable_component,
            extracellular_component
        };

        DoFRenumbering::Cuthill_McKee(dof_handler);
        DoFRenumbering::component_wise(dof_handler, blocks);

        dofs_per_block = DoFTools::count_dofs_per_fe_component(dof_handler, false, blocks);

        locally_owned_dofs = dof_handler.locally_owned_dofs().split_by_block(dofs_per_block);
        locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler).split_by_block(dofs_per_block);

        {
            FEValues<dim> fe_v(fe, quadrature, update_default);

            component_local_dofs.resize(3);
            for (unsigned int i = 0; i < fe_v.dofs_per_cell; ++i)
            {
                const unsigned int c_i = fe.system_to_component_index(i).first;

                component_local_dofs[c_i].push_back(local_to_component_index(i));
            }

            for (unsigned int i = 0; i < 3; ++i)
            {
                component_local_dofs[i].shrink_to_fit();
            }
        }

        constraints.clear();
        constraints.close();

        BlockDynamicSparsityPattern bdsp(locally_relevant_dofs);
        DoFTools::make_sparsity_pattern(dof_handler, bdsp, constraints, false);
        SparsityTools::distribute_sparsity_pattern(bdsp, locally_owned_dofs, mpi_communicator, locally_relevant_dofs);
        sparsity_pattern.copy_from(bdsp);

        mass_matrix.reinit(locally_owned_dofs, locally_owned_dofs, sparsity_pattern, mpi_communicator);
        explicit_mass_matrix.reinit(2, 2);
        implicit_mass_matrix.reinit(2, 2);

        {
            std::vector<IndexSet> owned_dofs(2);

            owned_dofs[explicit_transmembrane_component] = locally_owned_dofs[transmembrane_component];
            owned_dofs[explicit_state_variable_component] = locally_owned_dofs[state_variable_component];
            
            membrane_rhs.reinit(owned_dofs, mpi_communicator);

            BlockSparsityPattern explicit_bsp(2, 2);
            explicit_bsp.block(explicit_transmembrane_component, explicit_transmembrane_component) =
                sparsity_pattern.block(transmembrane_component, transmembrane_component);
            explicit_bsp.block(explicit_transmembrane_component, explicit_state_variable_component) =
                sparsity_pattern.block(transmembrane_component, state_variable_component);
            explicit_bsp.block(explicit_state_variable_component, explicit_transmembrane_component) =
                sparsity_pattern.block(state_variable_component, transmembrane_component);
            explicit_bsp.block(explicit_state_variable_component, explicit_state_variable_component) =
                sparsity_pattern.block(state_variable_component, state_variable_component);
            explicit_bsp.collect_sizes();

            membrane_matrix.reinit(owned_dofs, owned_dofs, explicit_bsp, mpi_communicator);
        }

        {
            std::vector<IndexSet> owned_dofs(2);
            
            owned_dofs[implicit_transmembrane_component] = locally_owned_dofs[transmembrane_component];
            owned_dofs[implicit_extracellular_component] = locally_owned_dofs[extracellular_component];

            tissue_rhs.reinit(owned_dofs, mpi_communicator);

            BlockSparsityPattern implicit_bsp(2, 2);
            implicit_bsp.block(implicit_transmembrane_component, implicit_transmembrane_component) =
                sparsity_pattern.block(transmembrane_component, transmembrane_component);
            implicit_bsp.block(implicit_transmembrane_component, implicit_extracellular_component) =
                sparsity_pattern.block(transmembrane_component, extracellular_component);
            implicit_bsp.block(implicit_extracellular_component, implicit_transmembrane_component) =
                sparsity_pattern.block(extracellular_component, transmembrane_component);
            implicit_bsp.block(implicit_extracellular_component, implicit_extracellular_component) =
                sparsity_pattern.block(extracellular_component, extracellular_component);
            implicit_bsp.collect_sizes();

            tissue_matrix.reinit(owned_dofs, owned_dofs, implicit_bsp, mpi_communicator);
            implicit_matrix.reinit(owned_dofs, owned_dofs, implicit_bsp, mpi_communicator);
        }

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
        std::vector<types::global_dof_index> v_indices(dofs_per_block[transmembrane_component]);
        std::vector<types::global_dof_index> w_indices(dofs_per_block[state_variable_component]);
        std::vector<types::global_dof_index> ue_indices(dofs_per_block[extracellular_component]);

        FullMatrix<double> mass_v(v_indices.size(), v_indices.size());
        FullMatrix<double> mass_w(w_indices.size(), w_indices.size());

        FullMatrix<double> membrane_vv(v_indices.size(), v_indices.size());
        FullMatrix<double> membrane_vw(v_indices.size(), w_indices.size());
        FullMatrix<double> membrane_wv(w_indices.size(), v_indices.size());
        FullMatrix<double> membrane_ww(w_indices.size(), w_indices.size());

        FullMatrix<double> tissue_v(v_indices.size(), v_indices.size());
        FullMatrix<double> tissue_ue(ue_indices.size(), ue_indices.size());

        const unsigned int transmembrane_offset = 0;
        const unsigned int state_variable_offset = transmembrane_offset + dofs_per_block[transmembrane_component];
        const unsigned int extracellular_offset = state_variable_offset + dofs_per_block[state_variable_offset];

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
                constraints.distribute_local_to_global(cell_membrane, local_dof_indices, membrane_temp);
                constraints.distribute_local_to_global(cell_tissue, local_dof_indices, tissue_temp);
            }
        }

        mass_matrix.compress(VectorOperation::add);
        membrane_matrix.compress(VectorOperation::add);
        tissue_matrix.compress(VectorOperation::add);

        pcout << "done." << std::endl;
    }

    template<int dim>
    void BidomainProblem<dim>::assemble_membrane_rhs(
        const double t,
        const LA::MPI::BlockVector& y,
        LA::MPI::BlockVector& out)
    {
        TimerOutput::Scope timer_scope(computing_timer, "Membrane RHS");
        pcout << "Assembling membrane RHS... " << std::flush;

        FEValues<dim> fe_v(fe, quadrature,
            update_values | update_quadrature_points | update_JxW_values);
        FEValuesExtractors::Scalar transmembrane_extractor(transmembrane_component);
                
        const unsigned int dofs_per_cell = fe_v.dofs_per_cell;
        const unsigned int n_q_points = fe_v.n_quadrature_points;

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
        std::vector<types::global_dof_index> rhs_v_indices(dofs_per_block[transmembrane_component]);
        std::vector<types::global_dof_index> rhs_w_indices(dofs_per_block[state_variable_component]);

        std::vector<double> function_values(n_q_points);

        Vector<double> rhs_v(rhs_v_indices.size());
        Vector<double> rhs_w(rhs_w_indices.size());

        FitzHughNagumo::Stimulus<dim> stimulus(t, param);

        const unsigned int transmembrane_offset = 0;
        const unsigned int state_variable_offset = transmembrane_offset + dofs_per_block[transmembrane_component];

        for (const auto& cell : dof_handler.active_cell_iterators())
        {
            if (cell->is_locally_owned())
            {
                fe_v.reinit(cell);
                cell->get_dof_indices(local_dof_indices);

                rhs_v = 0.;
                rhs_w = 0.;

                fe_v[transmembrane_extractor].get_function_values(y, function_values);

                for (unsigned int i = 0; i < component_local_dofs[transmembrane_component].size(); ++i)
                {
                    const unsigned int local_i = component_local_dofs[transmembrane_component][i] + transmembrane_offset;

                    double rhs_i = 0.;

                    for (unsigned int q = 0; q < n_q_points; ++q)
                    {
                        const double JxW = fe_v.JxW(q);
                        const Point<dim>& p = fe_v.quadrature_point(q);

                        rhs_i += param.chi * JxW
                            * fe_v.shape_value(local_i, q)
                            * (function_values[q]
                                    * function_values[q]
                                    * function_values[q]
                                    / param.fhn.epsilon / 3.
                                - stimulus.value(p));
                    }

                    rhs_v[i] += rhs_i;
                }

                for (unsigned int i = 0; i < component_local_dofs[state_variable_component].size(); ++i)
                {
                    const unsigned int local_i = component_local_dofs[state_variable_component][i] + state_variable_offset;

                    double rhs_i = 0.;

                    for (unsigned int q = 0; q < n_q_points; ++q)
                    {
                        const double JxW = fe_v.JxW(q);
                        
                        rhs_i += param.fhn.epsilon * param.fhn.beta * JxW
                            * fe_v.shape_value(local_i, q);
                    }

                    rhs_w[i] += rhs_i;
                }

                std::transform(
                    component_local_dofs[transmembrane_component].begin(),
                    component_local_dofs[transmembrane_component].end(),
                    rhs_v_indices.begin(),
                    [&local_dof_indices](unsigned int i) { return local_dof_indices[i]; });
                constraints.distribute_local_to_global(
                    rhs_v,
                    rhs_v_indices,
                    out.block(explicit_transmembrane_component));
                std::transform(
                    component_local_dofs[state_variable_component].begin(),
                    component_local_dofs[state_variable_component].end(),
                    rhs_w_indices.begin(),
                    [&local_dof_indices](unsigned int i) { return local_dof_indices[i]; });
                constraints.distribute_local_to_global(
                    rhs_w,
                    rhs_w_indices,
                    out.block(explicit_state_variable_component));
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
