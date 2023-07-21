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

#include <iostream>
#include <fstream>
#include <filesystem>

#include "bidomain.h"

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

        pcout << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl << std::endl;

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

        jacobian_matrix.reinit(locally_owned_dofs, locally_owned_dofs, sparsity_pattern, mpi_communicator);

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
    void BidomainProblem<dim>::compute_residual(
        const LA::MPI::Vector& W,
        LA::MPI::Vector& res)
    {
        pcout << "\tComputing residual vector... " << std::flush;

        FEValues<dim> fe_v(fe, quadrature,
            update_values | update_gradients | update_quadrature_points | update_JxW_values);
        
        const unsigned int dofs_per_cell = fe_v.dofs_per_cell;
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        res = 0.;
        jacobian_matrix = 0.;

        for (const auto& cell : dof_handler.active_cell_iterators())
        {
            if (cell->is_locally_owned())
            {
                assemble_cell_term(cell, fe_v, local_dof_indices, W, res);
            }
        }

        res.compress(VectorOperation::add);
        jacobian_matrix.compress(VectorOperation::add);

        pcout << "norm=" << res.l2_norm() << std::endl;
    }

    template<int dim>
    void BidomainProblem<dim>::assemble_cell_term(
        const typename DoFHandler<dim>::active_cell_iterator& cell,
        FEValues<dim>& fe_v,
        std::vector<types::global_dof_index>& local_dof_indices,
        const LA::MPI::Vector& W,
        LA::MPI::Vector& res)
    {
        PrescribedData::TransmembraneRightHandSide<dim> transmembrane_rhs(time, param);
        PrescribedData::ExtracellularRightHandSide<dim> extracellular_rhs;

        fe_v.reinit(cell);
        cell->get_dof_indices(local_dof_indices);

        const unsigned int dofs_per_cell = fe_v.dofs_per_cell;
        const unsigned int n_q_points = fe_v.n_quadrature_points;

        Table<2, Sacado::Fad::DFad<double>> w(n_q_points, 2);
        Table<2, double> w_old(n_q_points, 2);

        Table<3, Sacado::Fad::DFad<double>> grad_w(n_q_points, 2, dim);

        Table<2, double> rhs(n_q_points, 2);

        FullMatrix<double> cell_jacobian(dofs_per_cell, dofs_per_cell);
        Vector<double> cell_residual(dofs_per_cell);

        cell_jacobian = 0.;
        cell_residual = 0.;

        std::vector<Sacado::Fad::DFad<double>> independent_local_dof_values(dofs_per_cell);

        // mark W values as independent for autograd
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            independent_local_dof_values[i] = W(local_dof_indices[i]);
            independent_local_dof_values[i].diff(i, dofs_per_cell);
        }

        // compute quadrature points
        const std::vector<Point<dim>>& q_points = fe_v.get_quadrature_points();

        // zero w's
        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            for (unsigned int c = 0; c < 2; ++c)
            {
                w[q][c] = 0.;
                w_old[q][c] = 0.;

                for (unsigned int d = 0; d < dim; ++d)
                {
                    grad_w[q][c][d] = 0.;
                }
            }
        }

        // compute w's and rhs's
        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const unsigned int c = fe_v.get_fe().system_to_component_index(i).first;

                w[q][c] += independent_local_dof_values[i] * fe_v.shape_value_component(i, q, c);
                w_old[q][c] += old_solution[local_dof_indices[i]] * fe_v.shape_value_component(i, q, c);

                for (unsigned int d = 0; d < dim; ++d)
                {
                    grad_w[q][c][d] += independent_local_dof_values[i] * fe_v.shape_grad_component(i, q, c)[d];
                }
            }

            rhs[q][0] = transmembrane_rhs.value(q_points[q]);
            rhs[q][1] = extracellular_rhs.value(q_points[q]);
        }

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            Sacado::Fad::DFad<double> R_i = 0.;

            const unsigned int c_i = fe_v.get_fe().system_to_component_index(i).first;

            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                const double JxW = fe_v.JxW(q);

                if (c_i == 0)
                {
                    R_i += param.chi * param.Cm
                         * fe_v.shape_value_component(i, q, c_i)
                         * (w[q][c_i] - w_old[q][c_i])
                         * JxW;
                }

                if (c_i == 0)
                {
                    R_i += time_step * param.chi / param.Rm
                         * fe_v.shape_value_component(i, q, c_i)
                         * w[q][c_i]
                         * JxW;
                }

                for (unsigned int d = 0; d < dim; ++d)
                {
                    R_i += time_step * param.sigmai
                         * fe_v.shape_grad_component(i, q, c_i)[d]
                         * grad_w[q][c_i][d]
                         * JxW;
                }

                if (c_i == 1)
                {
                    for (unsigned int d = 0; d < dim; ++d)
                    {
                        R_i += time_step * param.sigmae
                             * fe_v.shape_grad_component(i, q, c_i)[d]
                             * grad_w[q][c_i][d]
                             * JxW;
                    }
                }

                R_i -= time_step
                     * fe_v.shape_value_component(i, q, c_i)
                     * rhs[q][c_i]
                     * JxW;
            }

            for (unsigned int k = 0; k < dofs_per_cell; ++k)
            {
                cell_jacobian(i, k) += R_i.fastAccessDx(k);
            }
            cell_residual(i) += R_i.val();
        }

        constraints.distribute_local_to_global(cell_jacobian, cell_residual, local_dof_indices, jacobian_matrix, res);
    }

    template<int dim>
    void BidomainProblem<dim>::prescribed_residual(
        LA::MPI::Vector& res)
    {
        FEValues<dim> fe_v(fe, quadrature,
            update_values | update_gradients | update_quadrature_points | update_JxW_values);
        
        const unsigned int dofs_per_cell = fe_v.dofs_per_cell;
        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        res = 0.;

        LA::MPI::Vector W_local(locally_owned_dofs, mpi_communicator);
        VectorTools::interpolate(
            dof_handler,
            PrescribedData::ExactSolution<dim>(time, param),
            W_local);
        
        LA::MPI::Vector W(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
        W = W_local;

        for (const auto& cell : dof_handler.active_cell_iterators())
        {
            if (cell->is_locally_owned())
            {
                prescribed_cell_term(cell, fe_v, local_dof_indices, W, res);
            }
        }

        res.compress(VectorOperation::add);
    }

    template<int dim>
    void BidomainProblem<dim>::prescribed_cell_term(
        const typename DoFHandler<dim>::active_cell_iterator& cell,
        FEValues<dim>& fe_v,
        std::vector<types::global_dof_index>& local_dof_indices,
        const LA::MPI::Vector& W,
        LA::MPI::Vector& res)
    {
        PrescribedData::ExactSolution<dim> exact_solution_old(time - time_step, param);
        PrescribedData::TransmembraneRightHandSide<dim> transmembrane_rhs(time, param);
        PrescribedData::ExtracellularRightHandSide<dim> extracellular_rhs;

        fe_v.reinit(cell);
        cell->get_dof_indices(local_dof_indices);

        const unsigned int dofs_per_cell = fe_v.dofs_per_cell;
        const unsigned int n_q_points = fe_v.n_quadrature_points;

        Table<2, double> w(n_q_points, 2);
        Table<2, double> w_old(n_q_points, 2);
        Table<3, double> grad_w(n_q_points, 2, dim);
        Table<2, double> rhs(n_q_points, 2);

        Vector<double> cell_residual(dofs_per_cell);

        cell_residual = 0.;

        // compute quadrature points
        const std::vector<Point<dim>>& q_points = fe_v.get_quadrature_points();

        // zero w's
        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            for (unsigned int c = 0; c < 2; ++c)
            {
                w[q][c] = 0.;
                for (unsigned int d = 0; d < dim; ++d)
                {
                    grad_w[q][c][d] = 0.;
                }
            }
        }

        // compute w's
        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const unsigned int c = fe.system_to_component_index(i).first;

                w[q][c] += W[local_dof_indices[i]] * fe_v.shape_value_component(i, q, c);

                for (unsigned int d = 0; d < dim; ++d)
                {
                    grad_w[q][c][d] += W[local_dof_indices[i]] * fe_v.shape_grad_component(i, q, c)[d];
                }

                if (c == 0)
                {
                    rhs[q][c] += transmembrane_rhs.value(q_points[q]);
                }
            }

            w_old[q][0] = exact_solution_old.value(q_points[q], 0);
            w_old[q][1] = exact_solution_old.value(q_points[q], 1);
            rhs[q][0] = transmembrane_rhs.value(q_points[q]);
            rhs[q][1] = extracellular_rhs.value(q_points[q]);
        }

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
            double R_i = 0.;

            const unsigned int c_i = fe_v.get_fe().system_to_component_index(i).first;

            for (unsigned int q = 0; q < n_q_points; ++q)
            {
                const double JxW = fe_v.JxW(q);

                if (c_i == 0)
                {
                    R_i += param.chi * param.Cm
                         * fe_v.shape_value_component(i, q, c_i)
                         * (w[q][c_i] - w_old[q][c_i])
                         * JxW;
                }

                if (c_i == 0)
                {
                    R_i += time_step * param.chi / param.Rm
                         * fe_v.shape_value_component(i, q, c_i)
                         * w[q][c_i]
                         * JxW;
                }

                for (unsigned int d = 0; d < dim; ++d)
                {
                    R_i += time_step * param.sigmai
                         * fe_v.shape_grad_component(i, q, c_i)[d]
                         * grad_w[q][c_i][d]
                         * JxW;
                }

                if (c_i == 1)
                {
                    for (unsigned int d = 0; d < dim; ++d)
                    {
                        R_i += time_step * param.sigmae
                             * fe_v.shape_grad_component(i, q, c_i)[d]
                             * grad_w[q][c_i][d]
                             * JxW;
                    }
                }

                R_i -= time_step
                     * fe_v.shape_value_component(i, q, c_i)
                     * rhs[q][c_i]
                     * JxW;
            }

            cell_residual(i) += R_i;
        }

        constraints.distribute_local_to_global(cell_residual, local_dof_indices, res);
    }

    template<int dim>
    void BidomainProblem<dim>::solve(
        const LA::MPI::Vector& W,
        LA::MPI::Vector& delta_W,
        double tolerance)
    {
        pcout << "Solving with Jacobian... " << std::flush;

        SolverControl solver_control(param.max_iterations * dof_handler.n_dofs(), tolerance);
        LA::SolverGMRES solver(solver_control, mpi_communicator);

        PETScWrappers::PreconditionNone preconditioner;
        PETScWrappers::PreconditionNone::AdditionalData additional_data;
        preconditioner.initialize(jacobian_matrix, additional_data);

        solver.solve(jacobian_matrix, delta_W, W, preconditioner);
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

        const unsigned int steps_per_output = param.n_output_files
            ? param.n_time_steps / param.n_output_files
            : param.n_time_steps + 1;
        
        SUNDIALS::KINSOL<LA::MPI::Vector>::AdditionalData additional_data;
        additional_data.function_tolerance = param.tolerance;
        SUNDIALS::KINSOL<LA::MPI::Vector> solver(additional_data, mpi_communicator);

        solver.reinit_vector = [this](LA::MPI::Vector& x)
        {
            x.reinit(this->locally_owned_dofs, this->mpi_communicator);
        };

        solver.residual = [this](const LA::MPI::Vector& evaluation_point, LA::MPI::Vector& residual)
        {
            TimerOutput::Scope timer_scope(this->computing_timer, "Residual");

            LA::MPI::Vector relevant_evaluation_point(this->locally_owned_dofs, this->locally_relevant_dofs, this->mpi_communicator);
            relevant_evaluation_point = evaluation_point;
            this->compute_residual(relevant_evaluation_point, residual);

            LA::MPI::Vector pres(this->locally_owned_dofs, this->mpi_communicator);
            this->prescribed_residual(pres);

            std::ofstream out;
            out.open("errors.txt", out.app);
            pres.print(out);

            return 0;
        };

        solver.setup_jacobian = [this](const LA::MPI::Vector&, const LA::MPI::Vector&)
        {
            return 0;
        };

        solver.solve_with_jacobian = [this](const LA::MPI::Vector& rhs, LA::MPI::Vector& solution, const double tolerance)
        {
            TimerOutput::Scope timer_scope(this->computing_timer, "Solve Jacobian System");
            this->solve(rhs, solution, tolerance);
            return 0;
        };

        while (timestep_number < param.n_time_steps)
        {
            ++timestep_number;
            time += time_step;

            old_solution = solution;

            pcout << "Time step " << timestep_number << ":" << std::endl;

            solver.solve(solution);

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
