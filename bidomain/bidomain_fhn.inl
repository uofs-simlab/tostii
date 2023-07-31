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
    constexpr types::global_dof_index BidomainProblem<dim>::global_to_component_index(const types::global_dof_index i) const
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

        IndexSet all_owned_dofs = dof_handler.locally_owned_dofs();
        locally_owned_dofs = all_owned_dofs.split_by_block(dofs_per_block);
        IndexSet all_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);
        locally_relevant_dofs = all_relevant_dofs.split_by_block(dofs_per_block);

        {
            FEValues<dim> fe_v(fe, quadrature, update_default);

            component_local_dofs.resize(3);
            for (unsigned int i = 0; i < fe_v.dofs_per_cell; ++i)
            {
                const unsigned int c_i = fe.system_to_component_index(i).first;

                component_local_dofs[c_i].push_back(i);
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
        SparsityTools::distribute_sparsity_pattern(bdsp, all_owned_dofs, mpi_communicator, all_relevant_dofs);

        {
            std::vector<IndexSet> owned_dofs(2);
            std::vector<IndexSet> relevant_dofs(2);

            owned_dofs[explicit_transmembrane_component] = locally_owned_dofs[transmembrane_component];
            owned_dofs[explicit_state_variable_component] = locally_owned_dofs[state_variable_component];

            relevant_dofs[explicit_transmembrane_component] = locally_relevant_dofs[transmembrane_component];
            relevant_dofs[explicit_state_variable_component] = locally_relevant_dofs[state_variable_component];
            
            membrane_temp.reinit(owned_dofs, mpi_communicator);
            relevant_membrane_temp.reinit(owned_dofs, relevant_dofs, mpi_communicator);
            membrane_rhs.reinit(owned_dofs, mpi_communicator);

            BlockDynamicSparsityPattern explicit_bdsp(2, 2);
            explicit_bdsp.block(explicit_transmembrane_component, explicit_transmembrane_component) =
                bdsp.block(transmembrane_component, transmembrane_component);
            explicit_bdsp.block(explicit_transmembrane_component, explicit_state_variable_component) =
                bdsp.block(transmembrane_component, state_variable_component);
            explicit_bdsp.block(explicit_state_variable_component, explicit_transmembrane_component) =
                bdsp.block(state_variable_component, transmembrane_component);
            explicit_bdsp.block(explicit_state_variable_component, explicit_state_variable_component) =
                bdsp.block(state_variable_component, state_variable_component);
            explicit_bdsp.collect_sizes();

            explicit_mass_matrix.reinit(owned_dofs, owned_dofs, explicit_bdsp, mpi_communicator);
            membrane_matrix.reinit(owned_dofs, owned_dofs, explicit_bdsp, mpi_communicator);
        }

        {
            std::vector<IndexSet> owned_dofs(2);
            
            owned_dofs[implicit_transmembrane_component] = locally_owned_dofs[transmembrane_component];
            owned_dofs[implicit_extracellular_component] = locally_owned_dofs[extracellular_component];

            tissue_rhs.reinit(owned_dofs, mpi_communicator);

            BlockDynamicSparsityPattern implicit_bdsp(2, 2);
            implicit_bdsp.block(implicit_transmembrane_component, implicit_transmembrane_component) =
                bdsp.block(transmembrane_component, transmembrane_component);
            implicit_bdsp.block(implicit_transmembrane_component, implicit_extracellular_component) =
                bdsp.block(transmembrane_component, extracellular_component);
            implicit_bdsp.block(implicit_extracellular_component, implicit_transmembrane_component) =
                bdsp.block(extracellular_component, transmembrane_component);
            implicit_bdsp.block(implicit_extracellular_component, implicit_extracellular_component) =
                bdsp.block(extracellular_component, extracellular_component);
            implicit_bdsp.collect_sizes();

            implicit_mass_matrix.reinit(owned_dofs, owned_dofs, implicit_bdsp, mpi_communicator);
            tissue_matrix.reinit(owned_dofs, owned_dofs, implicit_bdsp, mpi_communicator);
            implicit_matrix.reinit(owned_dofs, owned_dofs, implicit_bdsp, mpi_communicator);
        }

        solution.reinit(locally_owned_dofs, mpi_communicator);
        relevant_solution.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);

        VectorTools::interpolate(
            dof_handler,
            FitzHughNagumo::InitialValues<dim>(param),
            solution);
    }

    template<int dim>
    void BidomainProblem<dim>::assemble_system()
    {
        TimerOutput::Scope timer_scope(computing_timer, "Assemble System");
        pcout << "Assembling system matrices... " << std::flush;

        explicit_mass_matrix = 0.;
        implicit_mass_matrix = 0.;
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

        FullMatrix<double> tissue_vv(v_indices.size(), v_indices.size());
        FullMatrix<double> tissue_vue(v_indices.size(), ue_indices.size());
        FullMatrix<double> tissue_uev(ue_indices.size(), v_indices.size());
        FullMatrix<double> tissue_ueue(ue_indices.size(), ue_indices.size());

        for (const auto& cell : dof_handler.active_cell_iterators())
        {
            if (cell->is_locally_owned())
            {
                fe_v.reinit(cell);
                cell->get_dof_indices(local_dof_indices);

                mass_v = 0.;
                mass_w = 0.;

                membrane_vv = 0.;
                membrane_vw = 0.;
                membrane_wv = 0.;
                membrane_ww = 0.;

                tissue_vv = 0.;
                tissue_vue = 0.;
                tissue_uev = 0.;
                tissue_ueue = 0.;

                /* $\delta_{1, c_i}$ components */
                for (unsigned int i = 0; i < component_local_dofs[transmembrane_component].size(); ++i)
                {
                    const unsigned int local_i = component_local_dofs[transmembrane_component][i];

                    /* $\delta_{1, c_j}$ */
                    for (unsigned int j = 0; j < component_local_dofs[transmembrane_component].size(); ++i)
                    {
                        const unsigned int local_j = component_local_dofs[transmembrane_component][j];

                        double shape_value_product = 0.;
                        double shape_grad_product = 0.;

                        for (unsigned int q = 0; q < n_q_points; ++q)
                        {
                            shape_value_product += fe_v.JxW(q)
                                * fe_v.shape_value(local_i, q)
                                * fe_v.shape_value(local_j, q);
                            
                            shape_grad_product += fe_v.JxW(q)
                                * (fe_v.shape_grad(local_i, q)
                                    * fe_v.shape_grad(local_j, q));
                        }

                        mass_v(i, j) += param.chi * param.Cm * shape_value_product;
                        membrane_vv(i, j) -= param.chi / param.fhn.epsilon * shape_value_product;
                        tissue_vv(i, j) -= param.sigmai * shape_grad_product;
                    }

                    /* $\delta_{2, c_j}$ */
                    for (unsigned int j = 0; j < component_local_dofs[state_variable_component].size(); ++j)
                    {
                        const unsigned int local_j = component_local_dofs[state_variable_component][j];

                        double shape_value_product = 0.;

                        for (unsigned int q = 0; q < n_q_points; ++q)
                        {
                            shape_value_product += fe_v.JxW(q)
                                * fe_v.shape_value(local_i, q);
                                * fe_v.shape_value(local_j, q);
                        }

                        membrane_vw(i, j) += param.chi / param.fhn.epsilon * shape_value_product;
                    }

                    /* $\delta_{3, c_j}$ */
                    for (unsigned int j = 0; j < component_local_dofs[extracellular_component].size(); ++j)
                    {
                        const unsigned int local_j = component_local_dofs[extracellular_component][j];

                        double shape_grad_product = 0.;

                        for (unsigned int q = 0; q < n_q_points; ++q)
                        {
                            shape_grad_product += fe_v.JxW(q)
                                * (fe_v.shape_grad(local_i, q)
                                    * fe_v.shape_grad(local_j, q));
                        }

                        tissue_vue(i, j) -= param.sigmai * shape_grad_product;
                    }
                }

                /* $\delta_{2, c_i}$ components */
                for (unsigned int i = 0; i < component_local_dofs[state_variable_component].size(); ++i)
                {
                    const unsigned int local_i = component_local_dofs[state_variable_component][i];

                    /* $\delta_{1, c_j}$ */
                    for (unsigned int j = 0; j < component_local_dofs[transmembrane_component].size(); ++j)
                    {
                        const unsigned int local_j = component_local_dofs[transmembrane_component][j];

                        double shape_value_product = 0.;

                        for (unsigned int q = 0; q < n_q_points; ++q)
                        {
                            shape_value_product += fe_v.JxW(q)
                                * fe_v.shape_value(local_i, q)
                                * fe_v.shape_value(local_j, q);
                        }

                        membrane_wv(i, j) += param.fhn.epsilon * shape_value_product;
                    }

                    /* $\delta_{2, c_j}$ */
                    for (unsigned int j = 0; j < component_local_dofs[state_variable_component].size(); ++j)
                    {
                        const unsigned int local_j = component_local_dofs[state_variable_component][j];

                        double shape_value_product = 0.;

                        for (unsigned int q = 0; q < n_q_points; ++q)
                        {
                            shape_value_product += fe_v.JxW(q)
                                * fe_v.shape_value(local_i, q)
                                * fe_v.shape_value(local_j, q);
                        }

                        mass_w(i, j) += shape_value_product;
                        membrane_ww(i, j) -= param.fhn.epsilon * param.fhn.gamma * shape_value_product;
                    }
                }

                /* $\delta_{3, c_i}$ components */
                for (unsigned int i = 0; i < component_local_dofs[extracellular_component].size(); ++i)
                {
                    const unsigned int local_i = component_local_dofs[extracellular_component][i];

                    /* $\delta_{1, c_j}$ */
                    for (unsigned int j = 0; j < component_local_dofs[transmembrane_component].size(); ++j)
                    {
                        const unsigned int local_j = component_local_dofs[transmembrane_component][j];

                        double shape_grad_product = 0.;

                        for (unsigned int q = 0; q < n_q_points; ++q)
                        {
                            shape_grad_product += fe_v.JxW(q)
                                * (fe_v.shape_grad(local_i, q)
                                    * fe_v.shape_grad(local_j, q));
                        }

                        tissue_uev(i, j) -= param.sigmai * shape_grad_product;
                    }

                    /* $\delta_{3, c_j}$ */
                    for (unsigned int j = 0; j < component_local_dofs[extracellular_component].size(); ++j)
                    {
                        const unsigned int local_j = component_local_dofs[extracellular_component][j];

                        double shape_grad_product = 0.;

                        for (unsigned int q = 0; q < n_q_points; ++q)
                        {
                            shape_grad_product += fe_v.JxW(q)
                                * (fe_v.shape_grad(local_i, q)
                                    * fe_v.shape_grad(local_j, q));
                        }

                        tissue_ueue(i, j) -= (param.sigmai + param.sigmae) * shape_grad_product;
                    }
                }

                /*
                cell matrices have been assembled assuming they will be scattered by component_local_dofs,
                but they really need to be scattered by:
                component_local_dofs | local_dof_indices | global_to_component_index
                in order to actually end up in the correct positions.
                */
                std::transform(
                    component_local_dofs[transmembrane_component].begin(),
                    component_local_dofs[transmembrane_component].end(),
                    v_indices.begin(),
                    [this, &local_dof_indices](const unsigned int i)
                    {
                        return this->global_to_component_index(local_dof_indices[i]);
                    });
                std::transform(
                    component_local_dofs[state_variable_component].begin(),
                    component_local_dofs[state_variable_component].end(),
                    w_indices.begin(),
                    [this, &local_dof_indices](const unsigned int i)
                    {
                        return this->global_to_component_index(local_dof_indices[i]);
                    });
                std::transform(
                    component_local_dofs[extracellular_component].begin(),
                    component_local_dofs[extracellular_component].end(),
                    ue_indices.begin(),
                    [this, &local_dof_indices](const unsigned int i)
                    {
                        return this->global_to_component_index(local_dof_indices[i]);
                    });

                /* matrix assembly */
                // TODO: if constraints are added, multiple constraints objects may be needed

                /* explicit_mass_matrix = [mass_v, 0; 0, mass_w] */
                constraints.distribute_local_to_global(
                    mass_v,
                    v_indices,
                    explicit_mass_matrix.block(explicit_transmembrane_component, explicit_transmembrane_component));
                constraints.distribute_local_to_global(
                    mass_w,
                    w_indices,
                    explicit_mass_matrix.block(explicit_state_variable_component, explicit_state_variable_component));
                
                /* implicit_mass_matrix = [mass_v, 0; 0, 0] */
                constraints.distribute_local_to_global(
                    mass_v,
                    v_indices,
                    implicit_mass_matrix.block(implicit_transmembrane_component, implicit_transmembrane_component));

                /* membrane_matrix */
                constraints.distribute_local_to_global(
                    membrane_vv,
                    v_indices,
                    v_indices,
                    membrane_matrix.block(explicit_transmembrane_component, explicit_transmembrane_component));
                constraints.distribute_local_to_global(
                    membrane_vw,
                    v_indices,
                    w_indices,
                    membrane_matrix.block(explicit_transmembrane_component, explicit_state_variable_component));
                constraints.distribute_local_to_global(
                    membrane_wv,
                    w_indices,
                    v_indices,
                    membrane_matrix.block(explicit_state_variable_component, explicit_transmembrane_component));
                constraints.distribute_local_to_global(
                    membrane_ww,
                    w_indices,
                    w_indices,
                    membrane_matrix.block(explicit_state_variable_component, explicit_state_variable_component));

                /* tissue matrix */
                constraints.distribute_local_to_global(
                    tissue_vv,
                    v_indices,
                    v_indices,
                    tissue_matrix.block(implicit_transmembrane_component, implicit_transmembrane_component));
                constraints.distribute_local_to_global(
                    tissue_vue,
                    v_indices,
                    ue_indices,
                    tissue_matrix.block(implicit_transmembrane_component, implicit_extracellular_component));
                constraints.distribute_local_to_global(
                    tissue_uev,
                    ue_indices,
                    v_indices,
                    tissue_matrix.block(implicit_extracellular_component, implicit_transmembrane_component));
                constraints.distribute_local_to_global(
                    tissue_ueue,
                    ue_indices,
                    ue_indices,
                    tissue_matrix.block(implicit_extracellular_component, implicit_extracellular_component));
            }
        }

        explicit_mass_matrix.compress(VectorOperation::add);
        implicit_mass_matrix.compress(VectorOperation::add);
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
        std::vector<types::global_dof_index> v_indices(dofs_per_block[transmembrane_component]);
        std::vector<types::global_dof_index> w_indices(dofs_per_block[state_variable_component]);

        std::vector<double> v_values(n_q_points);

        Vector<double> rhs_v(v_indices.size());
        Vector<double> rhs_w(w_indices.size());

        FitzHughNagumo::Stimulus<dim> stimulus(t, param);

        for (const auto& cell : dof_handler.active_cell_iterators())
        {
            if (cell->is_locally_owned())
            {
                fe_v.reinit(cell);
                cell->get_dof_indices(local_dof_indices);

                rhs_v = 0.;
                rhs_w = 0.;

                fe_v[transmembrane_extractor].get_function_values(y, v_values);

                for (unsigned int i = 0; i < component_local_dofs[transmembrane_component].size(); ++i)
                {
                    const unsigned int local_i = component_local_dofs[transmembrane_component][i];

                    double rhs_i = 0.;

                    for (unsigned int q = 0; q < n_q_points; ++q)
                    {
                        const double JxW = fe_v.JxW(q);
                        const Point<dim>& p = fe_v.quadrature_point(q);

                        rhs_i += param.chi * JxW
                            * fe_v.shape_value(local_i, q)
                            * (v_values[q]
                                    * v_values[q]
                                    * v_values[q]
                                    / param.fhn.epsilon / 3.
                                - stimulus.value(p));
                    }

                    rhs_v[i] += rhs_i;
                }

                for (unsigned int i = 0; i < component_local_dofs[state_variable_component].size(); ++i)
                {
                    const unsigned int local_i = component_local_dofs[state_variable_component][i];

                    double rhs_i = 0.;

                    for (unsigned int q = 0; q < n_q_points; ++q)
                    {
                        const double JxW = fe_v.JxW(q);
                        
                        rhs_i += param.fhn.epsilon * param.fhn.beta * JxW
                            * fe_v.shape_value(local_i, q);
                    }

                    rhs_w[i] += rhs_i;
                }

                /* See assemble_system's comment */
                std::transform(
                    component_local_dofs[transmembrane_component].begin(),
                    component_local_dofs[transmembrane_component].end(),
                    v_indices.begin(),
                    [this, &local_dof_indices](const unsigned int i)
                    {
                        return this->global_to_component_index(local_dof_indices[i]);
                    });
                std::transform(
                    component_local_dofs[state_variable_component].begin(),
                    component_local_dofs[state_variable_component].end(),
                    w_indices.begin(),
                    [this, &local_dof_indices](const unsigned int i)
                    {
                        return this->global_to_component_index(local_dof_indices[i]);
                    });

                // TODO: if constraints are added, multiple constraints objects may be needed
                constraints.distribute_local_to_global(
                    rhs_v,
                    v_indices,
                    out.block(explicit_transmembrane_component));
                constraints.distribute_local_to_global(
                    rhs_w,
                    w_indices,
                    out.block(explicit_state_variable_component));
            }
        }

        out.compress(VectorOperation::add);

        membrane_matrix.vmult_add(out, y);

        pcout << "done." << std::endl;
    }

    template<int dim>
    void BidomainProblem<dim>::solve_membrane_lhs(
        const LA::MPI::BlockVector& y,
        LA::MPI::BlockVector& out)
    {
        TimerOutput::Scope timer_scope(computing_timer, "Membrane LHS");
        pcout << "Solving membrane LHS... " << std::flush;

        LA::MPI::PreconditionAMG preconditioner;
        {
            LA::MPI::PreconditionAMG::AdditionalData additional_data;
            preconditioner.initialize(explicit_mass_matrix, additional_data);
        }

        SolverControl solver_control(
            param.max_iterations * dof_handler.n_dofs(),
            param.tolerance);
        LA::SolverGMRES solver(solver_control, mpi_communicator);

        solver.solve(explicit_mass_matrix, out, y, preconditioner);
        constraints.distribute(out);

        pcout << "done in " << solver_control.last_step() << " iterations." << std::endl;
    }

    template<int dim>
    void BidomainProblem<dim>::step_tissue(
        const double tau,
        const LA::MPI::BlockVector& y,
        LA::MPI::BlockVector& out)
    {
        TimerOutput::Scope timer_scope(computing_timer, "Tissue");
        pcout << "Stepping tissue equation... " << std::flush;

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
            Assert(false, ExcMessage("Must use Backward Euler or Crank Nicolson tissue time stepping"));
        }

        implicit_matrix.copy_from(implicit_mass_matrix);
        if (theta != 1.)
        {
            implicit_matrix.add((1. - theta) * tau, tissue_matrix);
        }
        implicit_matrix.vmult(tissue_rhs, y);

        implicit_matrix.add(-tau, tissue_matrix);

        LA::MPI::PreconditionAMG preconditioner;
        {
            LA::MPI::PreconditionAMG::AdditionalData additional_data;
            preconditioner.initialize(implicit_matrix, additional_data);
        }

        SolverControl solver_control(
            param.max_iterations * dof_handler.n_dofs(),
            param.tolerance);
        LA::SolverGMRES solver(solver_control, mpi_communicator);

        solver.solve(implicit_matrix, out, tissue_rhs, preconditioner);
        constraints.distribute(out);

        pcout << "done in " << solver_control.last_step() << " iterations." << std::endl;
    }

    template<int dim>
    void BidomainProblem<dim>::output_results()
    {
        DataOut<dim> data_out;
        
        relevant_solution = solution;

        const FitzHughNagumo::DataPostprocessors::TransmembranePart<dim> transmembrane_part;
        const FitzHughNagumo::DataPostprocessors::StateVariablePart<dim> state_variable_part;
        const FitzHughNagumo::DataPostprocessors::ExtracellularPart<dim> extracellular_part;

        data_out.attach_dof_handler(dof_handler);
        data_out.add_data_vector(relevant_solution, transmembrane_part);
        data_out.add_data_vector(relevant_solution, state_variable_part);
        data_out.add_data_vector(relevant_solution, extracellular_part);

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

        ExplicitRungeKutta<LA::MPI::BlockVector> membrane_stepper(param.membrane_stepper);
        OSOperator<LA::MPI::BlockVector> membrane_operator = {
            &membrane_stepper,
            [this](
                const double t,
                const LA::MPI::BlockVector& y,
                LA::MPI::BlockVector& out)
            {
                this->relevant_membrane_temp = y;
                this->membrane_temp = 0.;
                this->assemble_membrane_rhs(t, this->relevant_membrane_temp, this->membrane_temp);
                this->solve_membrane_lhs(this->membrane_temp, out);
            },
            [](
                const double,
                const double,
                const LA::MPI::BlockVector&,
                LA::MPI::BlockVector&)
            { /* no jacobian solver required for explicit method */ }
        };

        Exact<LA::MPI::BlockVector> tissue_stepper;
        OSOperator<LA::MPI::BlockVector> tissue_operator = {
            &tissue_stepper,
            [](
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
            param.os_stepper,
            mask,
            solution);

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
