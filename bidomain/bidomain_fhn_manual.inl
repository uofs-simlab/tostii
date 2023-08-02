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

#include "bidomain_fhn_manual.h"

namespace Bidomain
{
    template<int dim>
    BaseProblem<dim>::BaseProblem(const Parameters::AllParameters& param)
        : param(param)
        , computing_timer(std::cout, TimerOutput::never, TimerOutput::wall_times)
        , dof_handler(triangulation)
        , dof_offsets(4)
        , fe(FE_Q<dim>(param.polynomial_degree), 3)
        , quadrature(param.quadrature_order)
    {
        GridGenerator::hyper_cube(triangulation, 0., 1.);
        triangulation.refine_global(param.global_refinement_level);

        std::cout << "Number of active cells: " << triangulation.n_active_cells() << std::endl;

        dof_handler.distribute_dofs(fe);

        std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

        std::vector<unsigned int> blocks = {
            transmembrane_component,
            state_variable_component,
            extracellular_component
        };

        dofs_per_block = DoFTools::count_dofs_per_fe_block(dof_handler, blocks);
        std::partial_sum(
            dofs_per_block.begin(),
            dofs_per_block.end(),
            dof_offsets.begin() + 1);

        DoFRenumbering::Cuthill_McKee(dof_handler);
        DoFRenumbering::component_wise(dof_handler, blocks);

        constraints.clear();
        constraints.close();

        sparsity_template.reinit(dof_handler.n_dofs(), dof_handler.n_dofs());
    }

    template<int dim>
    void BaseProblem<dim>::initialize_split(
        const std::vector<unsigned int>& mask,
        DynamicSparsityPattern& dsp,
        std::vector<unsigned int>& component_dof_indices,
        std::function<void(
            const Vector<double>&,
            Vector<double>&)> translate[2]) const
    {
        std::vector<unsigned int> mask_offsets(mask.size() + 1);
        mask_offsets[0] = 0;
        std::partial_sum(
            mask.begin(),
            mask.end(),
            mask_offsets.begin() + 1,
            [this](const unsigned int c) { return this->dofs_per_block[c]; });

        dsp.reinit(mask_offsets.back(), mask_offsets.back());
        
        for (unsigned int I = 0; I < mask.size(); ++I)
        {
            for (unsigned int J = 0; J < mask.size(); ++J)
            {
                for (unsigned int i = 0; i < dofs_per_block[mask[I]]; ++i)
                {
                    for (unsigned int j = 0; j < dofs_per_block[mask[J]]; ++j)
                    {
                        if (sparsity_template.exists(i + dof_offsets[mask[I]], j + dof_offsets[mask[J]]))
                        {
                            dsp.add(i + mask_offsets[I], j + mask_offsets[J]);
                        }
                    }
                }
            }
        }

        const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
        const unsigned int dofs_per_component = dofs_per_cell / fe.n_components();

        component_dof_indices.resize(dofs_per_component * mask.size());

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            const unsigned int c_i = fe.system_to_component_index(i).first;
            auto it = std::find(mask.begin(), mask.end(), c_i);
            if (it != mask.end())
            {
                component_dof_indices[(it - mask.begin()) * dofs_per_component] = i;
            }
        }

        // TODO: create translate functions
    }
}

namespace Bidomain
{
    template<int dim>
    ExplicitProblem<dim>::ExplicitProblem(const Parameters::AllParameters& param)
        : BaseProblem<dim>(param)
        , dof_handler(this->triangulation)
        , fe(FE_Q<dim>(this->param.polynomial_degree), 2)
    {
        dof_handler.distribute_dofs(fe);

        this->pcout << "Number of explicit DoFs: " << dof_handler.n_dofs() << std::endl;

        std::vector<unsigned int> blocks = {
            transmembrane_component,
            state_variable_component
        };

        DoFRenumbering::Cuthill_McKee(dof_handler);
        DoFRenumbering::component_wise(dof_handler, blocks);

        locally_owned_dofs = dof_handler.locally_owned_dofs();
        DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

        constraints.clear();
        constraints.close();

        {
            DynamicSparsityPattern dsp(locally_relevant_dofs);
            DoFTools::make_sparsity_pattern(
                dof_handler,
                dsp,
                constraints,
                false);
            SparsityTools::distribute_sparsity_pattern(
                dsp,
                locally_owned_dofs,
                this->mpi_communicator,
                locally_relevant_dofs);
            sparsity_pattern.copy_from(dsp);
        }

        mass_matrix.reinit(
            locally_owned_dofs,
            locally_owned_dofs,
            sparsity_pattern,
            this->mpi_communicator);
        membrane_matrix.reinit(
            locally_owned_dofs,
            locally_owned_dofs,
            sparsity_pattern,
            this->mpi_communicator);
        
        temp.reinit(
            locally_owned_dofs,
            this->mpi_communicator);
        
        assemble_system();
    }

    template<int dim>
    void ExplicitProblem<dim>::assemble_system()
    {
        TimerOutput::Scope timer_scope(this->computing_timer, "Explicit System");
        this->pcout << "Assembling explicit system... " << std::flush;

        mass_matrix = 0.;
        membrane_matrix = 0.;

        FEValues<dim> fe_v(
            fe,
            this->quadrature,
            update_values | update_JxW_values);
        
        const unsigned int dofs_per_cell = fe_v.dofs_per_cell;
        const unsigned int n_q_points = fe_v.n_quadrature_points;

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        FullMatrix<double> cell_mass(dofs_per_cell, dofs_per_cell);
        FullMatrix<double> cell_membrane(dofs_per_cell, dofs_per_cell);

        for (const auto& cell : dof_handler.active_cell_iterators())
        {
            if (cell->is_locally_owned())
            {
                fe_v.reinit(cell);
                cell->get_dof_indices(local_dof_indices);

                cell_mass = 0.;
                cell_membrane = 0.;

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    const unsigned int c_i = fe.system_to_component_index(i).first;

                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                        const unsigned int c_j = fe.system_to_component_index(j).first;

                        double shape_value_product = 0.;

                        for (unsigned int q = 0; q < n_q_points; ++q)
                        {
                            shape_value_product += fe_v.JxW(q)
                                * fe_v.shape_value(i, q)
                                * fe_v.shape_value(i, q);
                        }

                        switch (c_i << 1 | c_j)
                        {
                        case 0 << 1 | 0:
                            cell_mass(i, j) += this->param.chi * this->param.Cm
                                * shape_value_product;
                            cell_membrane(i, j) -= this->param.chi / this->param.fhn.epsilon
                                * shape_value_product;
                            break;
                        case 0 << 1 | 1:
                            cell_membrane(i, j) += this->param.chi / this->param.fhn.epsilon
                                * shape_value_product;
                            break;
                        case 1 << 1 | 0:
                            cell_membrane(i, j) += this->param.fhn.epsilon
                                * shape_value_product;
                            break;
                        case 1 << 1 | 1:
                            cell_mass(i, j) += shape_value_product;
                            cell_membrane(i, j) -= param.fhn.epsilon * param.fhn.gamma
                                * shape_value_product;
                            break;
                        default:
                            /* one of these is guaranteed to throw */
                            AssertIndexRange(c_i, 2);
                            AssertIndexRange(c_j, 2);
                        }
                    }
                }

                constraints.distribute_local_to_global(
                    cell_mass,
                    local_dof_indices,
                    mass_matrix);
                constraints.distribute_local_to_global(
                    cell_membrane,
                    local_dof_indices,
                    membrane_matrix);
            }
        }

        mass_matrix.compress(VectorOperation::add);
        membrane_matrix.compress(VectorOperation::add);

        this->pcout << "done." << std::endl;
    }

    template<int dim>
    void ExplicitProblem<dim>::assemble_membrane_rhs(
        const double t,
        const LA::MPI::Vector& y,
        LA::MPI::Vector& out)
    {
        TimerOutput::Scope timer_scope(this->computing_timer, "Explicit RHS");
        this->pcout << "\t\tAssembling explicit RHS... " << std::flush;

        out = 0.;

        FEValues<dim> fe_v(
            fe,
            this->quadrature,
            update_values | update_quadrature_points | update_JxW_values);
        
        const unsigned int dofs_per_cell = fe_v.dofs_per_cell;
        const unsigned int n_q_points = fe_v.n_quadrature_points;

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
        std::vector<double> transmembrane_values(n_q_points);

        Vector<double> cell_rhs(dofs_per_cell);

        FitzHughNagumo::Stimulus<dim> stimulus(t, this->param);

        for (const auto& cell : dof_handler.active_cell_iterators())
        {
            if (cell->is_locally_owned())
            {
                fe_v.reinit(cell);
                cell->get_dof_indices(local_dof_indices);

                cell_rhs = 0.;

                for (unsigned int q = 0; q < n_q_points; ++q)
                {
                    transmembrane_values[q] = 0.;

                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                        const unsigned int c_j = fe.system_to_component_index(j).first;

                        if (c_j == transmembrane_component)
                        {
                            transmembrane_values[q] += y[local_dof_indices[j]]
                                * fe_v.shape_value(j, q);
                        }
                    }
                }

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    const unsigned int c_i = fe.system_to_component_index(i).first;

                    for (unsigned int q = 0; q < n_q_points; ++q)
                    {
                        if (c_i == transmembrane_component)
                        {
                            const Point<dim>& p = fe_v.quadrature_point(q);

                            cell_rhs[i] += this->param.chi * fe_v.JxW(q) * (
                                transmembrane_values[q]
                                        * transmembrane_values[q]
                                        * transmembrane_values[q]
                                        / param.fhn.epsilon / 3.
                                    - stimulus.value(p))
                                * fe_v.shape_value(i, q);
                        }
                        else if (c_i == state_variable_component)
                        {
                            cell_rhs[i] += param.fhn.epsilon * param.fhn.beta * fe_v.JxW(q)
                                * fe_v.shape_value(i, q);
                        }
                        else
                        {
                            /* guaranteed to throw */
                            AssertIndexRange(c_i, 2);
                        }
                    }
                }

                constraints.distribute_local_to_global(
                    cell_rhs,
                    local_dof_indices,
                    out);
            }
        }

        out.compress(VectorOperation::add);

        membrane_matrix.vmult_add(out, y);

        this->pcout << "done." << std::endl;
    }

    template<int dim>
    void ExplicitProblem<dim>::solve_membrane_lhs(
        const LA::MPI::Vector& y,
        LA::MPI::Vector& out)
    {
        TimerOutput::Scope timer_scope(this->computing_timer, "Explicit LHS");
        this->pcout << "\t\tSolving explicit LHS... " << std::flush;

        LA::MPI::PreconditionAMG preconditioner;
        {
            LA::MPI::PreconditionAMG::AdditionalData additional_data;
            additional_data.symmetric_operator = true;
            preconditioner.initialize(mass_matrix, additional_data);
        }

        SolverControl solver_control(
            this->param.max_iterations * dof_handler.n_dofs(),
            this->param.tolerance);
        LA::SolverCG solver(solver_control, mpi_communicator);

        solver.solve(mass_matrix, out, y, preconditioner);
        constraints.distribute(out);

        this->pcout << "done in " << solver_control.last_step() << " iterations." << std::endl;
    }

    template<int dim>
    void ExplicitProblem<dim>::rhs_f(
        const double t,
        const LA::MPI::Vector& y,
        LA::MPI::Vector& out)
    {
        TimerOutput::Scope timer_scope(this->computing_timer, "Explicit Step");
        this->pcout << "\tExplicit step... " << std::flush;

        assemble_membrane_rhs(t, y, temp);
        solve_membrane_lhs(temp, out);

        this->pcout << "\tdone." << std::endl;
    }
}

namespace Bidomain
{
    template<int dim>
    ImplicitProblem<dim>::ImplicitProblem(const Parameters::AllParameters& param)
        : BaseProblem<dim>(param)
        , dof_handler(this->triangulation)
        , fe(FE_Q<dim>(this->param.polynomial_degree), 2)
    {
        dof_handler.distribute_dofs(fe);

        this->pcout << "Number of implicit DoFs: " << dof_handler.n_dofs() << std::endl;

        std::vector<unsigned int> blocks = {
            transmembrane_component,
            extracellular_component
        };

        DoFRenumbering::Cuthill_McKee(dof_handler);
        DoFRenumbering::component_wise(dof_handler, blocks);

        locally_owned_dofs = dof_handler.locally_owned_dofs();
        DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

        constraints.clear();
        constraints.close();

        {
            DynamicSparsityPattern dsp(locally_relevant_dofs);
            DoFTools::make_sparsity_pattern(
                dof_handler,
                dsp,
                constraints,
                false);
            SparsityTools::distribute_sparsity_pattern(
                dsp,
                locally_owned_dofs,
                this->mpi_communicator,
                locally_relevant_dofs);
            sparsity_pattern.copy_from(dsp);
        }

        mass_matrix.reinit(
            locally_owned_dofs,
            locally_owned_dofs,
            sparsity_pattern,
            this->mpi_communicator);
        tissue_matrix.reinit(
            locally_owned_dofs,
            locally_owned_dofs,
            sparsity_pattern,
            this->mpi_communicator);
        system_matrix.reinit(
            locally_owned_dofs,
            locally_owned_dofs,
            sparsity_pattern,
            this->mpi_communicator);

        temp.reinit(
            locally_owned_dofs,
            this->mpi_communicator);
        
        assemble_system();
    }

    template<int dim>
    void ImplicitProblem<dim>::assemble_system()
    {
        TimerOutput::Scope timer_scope(this->computing_timer, "Implicit System");
        this->pcout << "Assembling implicit system... " << std::flush;

        mass_matrix = 0.;
        tissue_matrix = 0.;
        system_matrix = 0.;

        FEValues<dim> fe_v(
            fe,
            this->quadrature,
            update_values | update_gradients | update_JxW_values);
        
        const unsigned int dofs_per_cell = fe_v.dofs_per_cell;
        const unsigned int n_q_points = fe_v.n_quadrature_points;

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        FullMatrix<double> cell_mass(dofs_per_cell);
        FullMatrix<double> cell_tissue(dofs_per_cell);

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

                        double shape_value_product = 0.;
                        double shape_grad_product = 0.;

                        for (unsigned int q = 0; q < n_q_points; ++q)
                        {
                            shape_value_product += fe_v.JxW(q)
                                * fe_v.shape_value(i, q)
                                * fe_v.shape_value(i, q);
                            shape_grad_product += fe_v.JxW(q)
                                * fe_v.shape_grad(i, q)
                                * fe_v.shape_grad(i, q);
                        }

                        switch (c_i << 1 | c_j)
                        {
                        case 0 << 1 | 0:
                            cell_mass(i, j) += this->param.chi * this->param.Cm
                                * shape_value_product;
                            [[fallthrough]];
                        case 0 << 1 | 1:
                        case 1 << 1 | 0:
                            cell_tissue(i, j) -= this->param.sigmai
                                * shape_grad_product;
                            break;
                        case 1 << 1 | 1:
                            cell_tissue(i, j) -= (this->param.sigmai + this->param.sigmae)
                                * shape_grad_product;
                            break;
                        default:
                            /* one of these is guaranteed to throw */
                            AssertIndexRange(c_i, 2);
                            AssertIndexRange(c_j, 2);
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

        this->pcout << "done." << std::endl;
    }

    template<int dim>
    void ImplicitProblem<dim>::step_tissue(
        const double,
        const double tau,
        const LA::MPI::Vector& y,
        LA::MPI::Vector& out)
    {
        TimerOutput::Scope timer_scope(this->computing_timer, "Implicit Step");
        this->pcout << "\tSolving implicit step... " << std::flush;

        double theta = 0.;
        switch (this->param.tissue_stepper)
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

        system_matrix.copy_from(mass_matrix);
        if (theta != 1.)
        {
            system_matrix.add((1. - theta) * tau, tissue_matrix);
        }

        system_matrix.vmult(temp, y);

        system_matrix.add(-tau, tissue_matrix);

        LA::MPI::PreconditionAMG preconditioner;
        {
            LA::MPI::PreconditionAMG::AdditionalData additional_data;
            additional_data.symmetric_operator = true;
            preconditioner.initialize(system_matrix, additional_data);
        }

        SolverControl solver_control(
            this->param.max_iterations * dof_handler.n_dofs(),
            this->param.tolerance);
        LA::SolverCG solver(solver_control, this->mpi_communicator);

        solver.solve(system_matrix, out, temp, preconditioner);
        constraints.distribute(out);

        this->pcout << "done in " << solver_control.last_step() << " iterations." << std::endl;
    }

    
}
