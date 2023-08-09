#pragma once

#include <deal.II/base/exceptions.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/grid/grid_generator.h>

#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <tostii/time_stepping/exact.h>
#include <tostii/time_stepping/explicit_runge_kutta.h>
#include <tostii/time_stepping/operator_split_single.h>

#include <array>
#include <functional>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <filesystem>

#include "bidomain_fhn_manual.h"

namespace Bidomain
{
    template<int dim>
    BaseProblem<dim>::BaseProblem(const Parameters::AllParameters& param)
        : param(param)
        , computing_timer(std::cout, TimerOutput::never, TimerOutput::wall_times)
        , dof_offsets(4)
        , dof_handler(triangulation)
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

        constraints_template.clear();
        constraints_template.close();

        sparsity_template.reinit(dof_handler.n_dofs(), dof_handler.n_dofs());

        DoFTools::make_sparsity_pattern(
            dof_handler,
            sparsity_template,
            constraints_template,
            false);
    }

    template<int dim>
    void BaseProblem<dim>::initialize_split(
        const std::vector<unsigned int>& mask,
        AffineConstraints<double>& constraints,
        DynamicSparsityPattern& dsp,
        std::vector<unsigned int>& component_dof_indices,
        std::function<types::global_dof_index(
            types::global_dof_index)>& shift,
        std::function<void(
            const Vector<double>&,
            Vector<double>&)> translate[2]) const
    {
        std::vector<unsigned int> mask_offsets(mask.size() + 1);
        mask_offsets[0] = 0;
        std::transform_inclusive_scan(
            mask.begin(),
            mask.end(),
            mask_offsets.begin() + 1,
            std::plus(),
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

        {
            IndexSet dofs = complete_index_set(dof_handler.n_dofs());
            std::vector<IndexSet> split_dofs = dofs.split_by_block(dofs_per_block);
            dofs.clear();

            for (unsigned int c : mask)
            {
                dofs.add_indices(split_dofs[c]);
            }

            constraints.add_selected_constraints(constraints_template, dofs);
        }

        const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
        const unsigned int dofs_per_component = dofs_per_cell / fe.n_components();

        component_dof_indices.reserve(dofs_per_component * mask.size());

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            const unsigned int c_i = fe.system_to_component_index(i).first;
            auto it = std::find(mask.begin(), mask.end(), c_i);
            if (it != mask.end())
            {
                component_dof_indices.push_back(i);
            }
        }

        {
            std::vector<std::pair<unsigned int, unsigned int>> offset_map(mask.size());
            for (unsigned int i = 0; i < mask.size(); ++i)
            {
                offset_map[i].first = dof_offsets[mask[i]];
                offset_map[i].second = mask_offsets[i];
            }
            std::sort(
                offset_map.begin(),
                offset_map.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; });

            shift = [offset_map](types::global_dof_index i)
            {
                auto it = std::upper_bound(
                    offset_map.begin(),
                    offset_map.end(),
                    i,
                    [](auto v, const auto& e) { return v >= e.first; });
                return i + it->second - it->first;
            };
        }

        {
            std::vector<unsigned int> global_indices(mask_offsets.back());
            auto it = global_indices.begin();
            for (unsigned int i = 0; i < mask.size(); ++i)
            {
                auto last = it + dofs_per_block[mask[i]];
                std::iota(it, last, dof_offsets[mask[i]]);
                it = last;
            }

            translate[0] = [global_indices](
                const Vector<double>& global_v,
                Vector<double>& local_v)
            {
                global_v.extract_subvector_to(
                    global_indices.begin(),
                    global_indices.end(),
                    local_v.begin());
            };
        }

        {
            std::vector<std::pair<unsigned int, unsigned int>> local_ranges(mask.size());
            for (unsigned int i = 0; i < mask.size(); ++i)
            {
                local_ranges[i].first = dof_offsets[mask[i]];
                local_ranges[i].second = dofs_per_block[mask[i]];
            }

            translate[1] = [local_ranges](
                const Vector<double>& local_v,
                Vector<double>& global_v)
            {
                unsigned int first = 0;
                std::vector<unsigned int> indices;
                for (auto [offset, size] : local_ranges)
                {
                    indices.resize(size);
                    std::iota(indices.begin(), indices.end(), first);

                    local_v.extract_subvector_to(
                        indices.begin(),
                        indices.end(),
                        global_v.begin() + offset);
                    first += size;
                }
            };
        }
    }
}

namespace Bidomain
{
    template<int dim>
    ExplicitProblem<dim>::ExplicitProblem(const Parameters::AllParameters& param)
        : BaseProblem<dim>(param)
    {
        DynamicSparsityPattern dsp;
        constraints.clear();
        this->initialize_split(
            { this->transmembrane_component, this->state_variable_component },
            constraints,
            dsp,
            component_dof_indices,
            shift,
            translate);
        sparsity_pattern.copy_from(dsp);
        constraints.close();

        mass_matrix.reinit(sparsity_pattern);
        membrane_matrix.reinit(sparsity_pattern);
        
        const unsigned int N
            = this->dofs_per_block[this->transmembrane_component]
                + this->dofs_per_block[this->state_variable_component];
        temp.reinit(N);
        translate_buffer[0].reinit(N);
        translate_buffer[1].reinit(N);
        
        assemble_system();
    }

    template<int dim>
    void ExplicitProblem<dim>::assemble_system()
    {
        TimerOutput::Scope timer_scope(this->computing_timer, "Explicit System");
        std::cout << "Assembling explicit system... " << std::flush;

        mass_matrix = 0.;
        membrane_matrix = 0.;

        FEValues<dim> fe_v(
            this->fe,
            this->quadrature,
            update_values | update_JxW_values);
        
        const unsigned int dofs_per_cell = fe_v.dofs_per_cell;
        const unsigned int n_q_points = fe_v.n_quadrature_points;
        const unsigned int component_dofs = component_dof_indices.size();

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
        std::vector<types::global_dof_index> component_local_indices(component_dofs);

        FullMatrix<double> cell_mass(component_dofs, component_dofs);
        FullMatrix<double> cell_membrane(component_dofs, component_dofs);

        for (const auto& cell : this->dof_handler.active_cell_iterators())
        {
            if (cell->is_locally_owned())
            {
                fe_v.reinit(cell);
                cell->get_dof_indices(local_dof_indices);
                std::transform(
                    component_dof_indices.begin(),
                    component_dof_indices.end(),
                    component_local_indices.begin(),
                    [this, &local_dof_indices](unsigned int i)
                    {
                        return this->shift(local_dof_indices[i]);
                    });

                cell_mass = 0.;
                cell_membrane = 0.;

                for (unsigned int i = 0; i < component_dofs; ++i)
                {
                    const unsigned int& local_i = component_dof_indices[i];
                    const unsigned int c_i = this->fe.system_to_component_index(local_i).first;

                    for (unsigned int j = 0; j < component_dofs; ++j)
                    {
                        const unsigned int& local_j = component_dof_indices[j];
                        const unsigned int c_j = this->fe.system_to_component_index(local_j).first;

                        double shape_value_product = 0.;

                        for (unsigned int q = 0; q < n_q_points; ++q)
                        {
                            shape_value_product += fe_v.JxW(q)
                                * fe_v.shape_value(local_i, q)
                                * fe_v.shape_value(local_j, q);
                        }

                        switch (c_i << 2 | c_j)
                        {
                        case this->transmembrane_component << 2 | this->transmembrane_component:
                            cell_mass(i, j) += this->param.chi * this->param.Cm
                                * shape_value_product;
                            cell_membrane(i, j) -= this->param.chi / this->param.fhn.epsilon
                                * shape_value_product;
                            break;
                        case this->transmembrane_component << 2 | this->state_variable_component:
                            cell_membrane(i, j) += this->param.chi / this->param.fhn.epsilon
                                * shape_value_product;
                            break;
                        case this->state_variable_component << 2 | this->transmembrane_component:
                            cell_membrane(i, j) += this->param.fhn.epsilon
                                * shape_value_product;
                            break;
                        case this->state_variable_component << 2 | this->state_variable_component:
                            cell_mass(i, j) += shape_value_product;
                            cell_membrane(i, j) -= this->param.fhn.epsilon * this->param.fhn.gamma
                                * shape_value_product;
                            break;
                        default:
                            Assert(false, ExcMessage("Invalid DoF component"));
                        }
                    }
                }

                constraints.distribute_local_to_global(
                    cell_mass,
                    component_local_indices,
                    mass_matrix);
                constraints.distribute_local_to_global(
                    cell_membrane,
                    component_local_indices,
                    membrane_matrix);
            }
        }

        std::cout << "done." << std::endl;
    }

    template<int dim>
    void ExplicitProblem<dim>::assemble_membrane_rhs(
        const double t,
        const Vector<double>& y,
        Vector<double>& out)
    {
        TimerOutput::Scope timer_scope(this->computing_timer, "Explicit RHS");
        std::cout << "\t\tAssembling explicit RHS... " << std::flush;

        out = 0.;

        FEValues<dim> fe_v(
            this->fe,
            this->quadrature,
            update_values | update_quadrature_points | update_JxW_values);
        
        const unsigned int dofs_per_cell = fe_v.dofs_per_cell;
        const unsigned int n_q_points = fe_v.n_quadrature_points;
        const unsigned int component_dofs = component_dof_indices.size();

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
        std::vector<types::global_dof_index> component_local_indices(component_dofs);
        std::vector<double> transmembrane_values(n_q_points);

        Vector<double> cell_rhs(component_dofs);

        FitzHughNagumo::Stimulus<dim> stimulus(t, this->param);

        for (const auto& cell : this->dof_handler.active_cell_iterators())
        {
            if (cell->is_locally_owned())
            {
                fe_v.reinit(cell);
                cell->get_dof_indices(local_dof_indices);
                std::transform(
                    component_dof_indices.begin(),
                    component_dof_indices.end(),
                    component_local_indices.begin(),
                    [&local_dof_indices, this](unsigned int i)
                    {
                        return this->shift(local_dof_indices[i]);
                    });

                cell_rhs = 0.;

                for (unsigned int q = 0; q < n_q_points; ++q)
                {
                    transmembrane_values[q] = 0.;

                    for (unsigned int j = 0; j < component_dofs; ++j)
                    {
                        const unsigned int& local_j = component_dof_indices[j];
                        const unsigned int c_j = this->fe.system_to_component_index(local_j).first;

                        if (c_j == this->transmembrane_component)
                        {
                            transmembrane_values[q] += y[component_local_indices[j]]
                                * fe_v.shape_value(local_j, q);
                        }
                    }
                }

                for (unsigned int i = 0; i < component_dofs; ++i)
                {
                    const unsigned int& local_i = component_dof_indices[i];
                    const unsigned int c_i = this->fe.system_to_component_index(local_i).first;

                    for (unsigned int q = 0; q < n_q_points; ++q)
                    {
                        if (c_i == this->transmembrane_component)
                        {
                            const Point<dim>& p = fe_v.quadrature_point(q);

                            cell_rhs[i] += this->param.chi * fe_v.JxW(q) * (
                                transmembrane_values[q]
                                        * transmembrane_values[q]
                                        * transmembrane_values[q]
                                        / this->param.fhn.epsilon / 3.
                                    - stimulus.value(p))
                                * fe_v.shape_value(local_i, q);
                        }
                        else if (c_i == this->state_variable_component)
                        {
                            cell_rhs[i] += this->param.fhn.epsilon * this->param.fhn.beta * fe_v.JxW(q)
                                * fe_v.shape_value(local_i, q);
                        }
                        else
                        {
                            Assert(false, ExcMessage("Invalid DoF component"));
                        }
                    }
                }

                constraints.distribute_local_to_global(
                    cell_rhs,
                    component_local_indices,
                    out);
            }
        }

        membrane_matrix.vmult_add(out, y);

        std::cout << "done." << std::endl;
    }

    template<int dim>
    void ExplicitProblem<dim>::solve_membrane_lhs(
        const Vector<double>& y,
        Vector<double>& out)
    {
        TimerOutput::Scope timer_scope(this->computing_timer, "Explicit LHS");
        std::cout << "\t\tSolving explicit LHS... " << std::flush;

        PreconditionIdentity preconditioner;
        {
            PreconditionIdentity::AdditionalData additional_data;
            preconditioner.initialize(mass_matrix, additional_data);
        }

        SolverControl solver_control(
            this->param.max_iterations * this->dof_handler.n_dofs(),
            this->param.tolerance);
        SolverGMRES<Vector<double>> solver(solver_control);

        solver.solve(mass_matrix, out, y, preconditioner);
        constraints.distribute(out);

        std::cout << "done in " << solver_control.last_step() << " iterations." << std::endl;
    }

    template<int dim>
    void ExplicitProblem<dim>::rhs_f(
        const double t,
        const Vector<double>& y,
        Vector<double>& out)
    {
        std::cout << "\tExplicit step... " << std::endl;

        translate[0](y, translate_buffer[0]);

        assemble_membrane_rhs(t, translate_buffer[0], temp);
        solve_membrane_lhs(temp, translate_buffer[1]);

        translate[1](translate_buffer[1], out);

        std::cout << "\tdone." << std::endl;
    }
}

namespace Bidomain
{
    template<int dim>
    ImplicitProblem<dim>::ImplicitProblem(const Parameters::AllParameters& param)
        : BaseProblem<dim>(param)
    {
        DynamicSparsityPattern dsp;
        constraints.clear();
        this->initialize_split(
            { this->transmembrane_component, this->extracellular_component },
            constraints,
            dsp,
            component_dof_indices,
            shift,
            translate);
        sparsity_pattern.copy_from(dsp);
        constraints.close();

        mass_matrix.reinit(sparsity_pattern);
        tissue_matrix.reinit(sparsity_pattern);
        system_matrix.reinit(sparsity_pattern);

        const unsigned int N
            = this->dofs_per_block[this->transmembrane_component]
                + this->dofs_per_block[this->extracellular_component];

        temp.reinit(N);
        translate_buffer[0].reinit(N);
        translate_buffer[1].reinit(N);
        
        assemble_system();
    }

    template<int dim>
    void ImplicitProblem<dim>::assemble_system()
    {
        TimerOutput::Scope timer_scope(this->computing_timer, "Implicit System");
        std::cout << "Assembling implicit system... " << std::flush;

        mass_matrix = 0.;
        tissue_matrix = 0.;
        system_matrix = 0.;

        FEValues<dim> fe_v(
            this->fe,
            this->quadrature,
            update_values | update_gradients | update_quadrature_points | update_JxW_values);
        
        const unsigned int dofs_per_cell = fe_v.dofs_per_cell;
        const unsigned int n_q_points = fe_v.n_quadrature_points;
        const unsigned int component_dofs = component_dof_indices.size();

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
        std::vector<types::global_dof_index> component_local_indices(component_dofs);

        FullMatrix<double> cell_mass(component_dofs);
        FullMatrix<double> cell_tissue(component_dofs);

        for (const auto& cell : this->dof_handler.active_cell_iterators())
        {
            if (cell->is_locally_owned())
            {
                fe_v.reinit(cell);
                cell->get_dof_indices(local_dof_indices);
                std::transform(
                    component_dof_indices.begin(),
                    component_dof_indices.end(),
                    component_local_indices.begin(),
                    [this, &local_dof_indices](unsigned int i)
                    {
                        return this->shift(local_dof_indices[i]);
                    });

                cell_mass = 0.;
                cell_tissue = 0.;

                for (unsigned int i = 0; i < component_dofs; ++i)
                {
                    const unsigned int& local_i = component_dof_indices[i];
                    const unsigned int c_i = this->fe.system_to_component_index(local_i).first;
                    
                    for (unsigned int j = 0; j < component_dofs; ++j)
                    {
                        const unsigned int& local_j = component_dof_indices[j];
                        const unsigned int c_j = this->fe.system_to_component_index(local_j).first;

                        for (unsigned int q = 0; q < n_q_points; ++q)
                        {
                            const Tensor<2, dim> intracellular_conductivity
                                = FitzHughNagumo::IntracellularConductivity<dim>(0., this->param)
                                    .value(fe_v.quadrature_point(q));
                            const Tensor<2, dim> extracellular_conductivity
                                = FitzHughNagumo::ExtracellularConductivity<dim>(0., this->param)
                                    .value(fe_v.quadrature_point(q));
                            
                            switch (c_i << 2 | c_j)
                            {
                            case this->transmembrane_component << 2 | this->transmembrane_component:
                                cell_mass(i, j) += this->param.chi * this->param.Cm * fe_v.JxW(q)
                                    * fe_v.shape_value(local_i, q)
                                    * fe_v.shape_value(local_j, q);
                                [[fallthrough]];
                            case this->transmembrane_component << 2 | this->extracellular_component:
                            case this->extracellular_component << 2 | this->transmembrane_component:
                                cell_tissue(i, j) -= fe_v.JxW(q)
                                    * (intracellular_conductivity
                                    * fe_v.shape_grad(local_i, q)
                                    * fe_v.shape_grad(local_j, q));
                                break;
                            case this->extracellular_component << 2 | this->extracellular_component:
                                cell_tissue(i, j) -= fe_v.JxW(q)
                                    * ((intracellular_conductivity + extracellular_conductivity)
                                    * fe_v.shape_grad(local_i, q)
                                    * fe_v.shape_grad(local_j, q));
                                break;
                            default:
                                Assert(false, ExcMessage("Invalid DoF component"));
                            }
                        }
                    }
                }

                constraints.distribute_local_to_global(
                    cell_mass,
                    component_local_indices,
                    mass_matrix);
                constraints.distribute_local_to_global(
                    cell_tissue,
                    component_local_indices,
                    tissue_matrix);
            }
        }

        std::cout << "done." << std::endl;
    }

    template<int dim>
    void ImplicitProblem<dim>::step_tissue(
        const double,
        const double tau,
        const Vector<double>& y,
        Vector<double>& out)
    {
        TimerOutput::Scope timer_scope(this->computing_timer, "Implicit Step");
        std::cout << "\tSolving implicit step... " << std::flush;

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

        translate[0](y, translate_buffer[0]);

        system_matrix.copy_from(mass_matrix);
        if (theta != 1.)
        {
            system_matrix.add((1. - theta) * tau, tissue_matrix);
        }

        system_matrix.vmult(temp, translate_buffer[0]);

        system_matrix.add(-tau, tissue_matrix);

        PreconditionIdentity preconditioner;
        {
            PreconditionIdentity::AdditionalData additional_data;
            preconditioner.initialize(system_matrix, additional_data);
        }

        SolverControl solver_control(
            this->param.max_iterations * this->dof_handler.n_dofs(),
            this->param.tolerance);
        SolverGMRES<Vector<double>> solver(solver_control);

        solver.solve(system_matrix, translate_buffer[1], temp, preconditioner);
        constraints.distribute(translate_buffer[1]);

        translate[1](translate_buffer[1], out);

        std::cout << "done in " << solver_control.last_step() << " iterations." << std::endl;
    }
}

namespace Bidomain
{
    template<int dim>
    BidomainProblem<dim>::BidomainProblem(const Parameters::AllParameters& param)
        : BaseProblem<dim>(param)
        , ExplicitProblem<dim>(param)
        , ImplicitProblem<dim>(param)
        , timestep_number(this->param.initial_time_step)
        , time_step(this->param.final_time / this->param.n_time_steps)
        , time(timestep_number * time_step)
    {
        solution.reinit(this->dof_handler.n_dofs());

        VectorTools::interpolate(
            this->dof_handler,
            FitzHughNagumo::InitialValues<dim>(this->param),
            solution);
    }

    template<int dim>
    void BidomainProblem<dim>::output_results() const
    {
        DataOut<dim> data_out;

        const FitzHughNagumo::DataPostprocessors::TransmembranePart<dim> transmembrane_part;
        const FitzHughNagumo::DataPostprocessors::StateVariablePart<dim> state_variable_part;
        const FitzHughNagumo::DataPostprocessors::ExtracellularPart<dim> extracellular_part;

        data_out.attach_dof_handler(this->dof_handler);
        data_out.add_data_vector(solution, transmembrane_part);
        data_out.add_data_vector(solution, state_variable_part);
        data_out.add_data_vector(solution, extracellular_part);

        data_out.build_patches();

        std::filesystem::path path(this->param.output_prefix);

        std::stringstream filename;
        filename << path.string() << '.'
            << std::setfill('0') << std::setw(5) << timestep_number
            << ".vtu";
        
        std::ofstream out(filename.str());
        data_out.write_vtu(out);
    }

    template<int dim>
    void BidomainProblem<dim>::run()
    {
        output_results();

        const unsigned int steps_per_output = this->param.n_output_files
            ? this->param.n_time_steps / this->param.n_output_files
            : this->param.n_time_steps + 1;
        
        using namespace tostii::TimeStepping;

        ExplicitRungeKutta<Vector<double>> membrane_stepper(this->param.membrane_stepper);
        OSOperator<Vector<double>> membrane_operator = {
            &membrane_stepper,
            [this](
                const double t,
                const Vector<double>& y,
                Vector<double>& out)
            {
                this->rhs_f(t, y, out);
            },
            [](
                const double,
                const double,
                const Vector<double>&,
                Vector<double>&)
            { /* no jacobian solver needed for explicit method */ }
        };

        Exact<Vector<double>> tissue_stepper;
        OSOperator<Vector<double>> tissue_operator = {
            &tissue_stepper,
            [](
                const double,
                const Vector<double>&,
                Vector<double>&)
            { /* no function needed for exact stepper */ },
            [this](
                const double t,
                const double tau,
                const Vector<double>& y,
                Vector<double>& out)
            {
                this->step_tissue(t, tau, y, out);
            }
        };

        std::vector<OSOperator<Vector<double>>> operators = {
            membrane_operator,
            tissue_operator
        };

        std::vector<OSPair<double>> stages;
        {
            const std::vector<OSPair<double>>& orig_stages
                = os_method<double>::to_os_pairs(this->param.os_stepper);

            for (const auto& pair : orig_stages)
            {
                if (pair.op_num == 0)
                {
                    for (unsigned int i = 0; i < 100; ++i)
                    {
                        OSPair<double> sub_pair = { pair.op_num, pair.alpha / 100. };
                        stages.push_back(sub_pair);
                    }
                }
                else
                {
                    stages.push_back(pair);
                }
            }
        }

        OperatorSplitSingle<Vector<double>> stepper(
            operators,
            stages,
            solution);
        
        while (timestep_number < this->param.n_time_steps)
        {
            ++timestep_number;

            std::cout << "Time Step " << timestep_number << ':' << std::endl;

            time = stepper.evolve_one_time_step(time, time_step, solution);

            if (timestep_number % steps_per_output == 0)
            {
                output_results();
            }
        }

        this->computing_timer.print_summary();
        std::cout << std::endl;
    }
}
