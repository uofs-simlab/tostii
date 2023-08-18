#pragma once

#include <filesystem>

#include <deal.II/sundials/kinsol.h>

#include <tostii/checkpoint/serialize_petsc_mpi.h>

#include <tostii/time_stepping/implicit_runge_kutta.h>

#include "monolithic_hs.h"
#include "prescribed_data.h"
#include "utils.h"

namespace NSE
{
    template<int dim>
    NonlinearSchroedingerEquation<dim>::NonlinearSchroedingerEquation(
        const Parameters::AllParameters& param)
        : param(param)
        , mpi_communicator(MPI_COMM_WORLD)
        , pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        , computing_timer(mpi_communicator, pcout, TimerOutput::never, TimerOutput::wall_times)
        , triangulation(mpi_communicator)
        , dof_handler(triangulation)
        , fe(FE_Q<dim>(param.polynomial_degree), 2)
        , quadrature(param.quadrature_order)
        , time(0.)
        , time_step(1. / param.n_time_steps)
        , timestep_number(0)
        , kappa(1.)
    {
        GridGenerator::hyper_cube(triangulation, -1., 1.);
        triangulation.refine_global(param.refinement_level);

        pcout << "Refinement level: " << param.refinement_level
            << "\nNumber of active cells: " << triangulation.n_active_cells() << std::endl;

        dof_handler.distribute_dofs(fe);
        DoFRenumbering::Cuthill_McKee(dof_handler);

        pcout << "Number of degrees of freedom: " << dof_handler.n_dofs()
            << "\nNumber of time steps: " << param.n_time_steps
            << std::endl;

        locally_owned_dofs = dof_handler.locally_owned_dofs();
        DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

        constraints.reinit(locally_relevant_dofs);
        VectorTools::interpolate_boundary_values(
            dof_handler,
            0,
            Functions::ZeroFunction<dim>(2),
            constraints);
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
                mpi_communicator,
                locally_relevant_dofs);
            sparsity_pattern.copy_from(dsp);
        }

        stiffness_matrix.reinit(
            locally_owned_dofs,
            locally_owned_dofs,
            sparsity_pattern,
            mpi_communicator);
        old_stiffness_matrix.reinit(
            locally_owned_dofs,
            locally_owned_dofs,
            sparsity_pattern,
            mpi_communicator);
        jacobian_C.reinit(
            locally_owned_dofs,
            locally_owned_dofs,
            sparsity_pattern,
            mpi_communicator);
        jacobian_matrix.reinit(
            locally_owned_dofs,
            locally_owned_dofs,
            sparsity_pattern,
            mpi_communicator);

        solution.reinit(locally_owned_dofs, mpi_communicator);
        ghost_solution.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
        old_solution_residual_ready = false;
        old_solution_residual.reinit(locally_owned_dofs, mpi_communicator);
        temp.reinit(locally_owned_dofs, mpi_communicator);

        this->initialize(
            param.checkpoint_path,
            mpi_communicator,
            3,
            int(std::log10(param.n_time_steps)) + 1);
        
        if (this->n_checkpoints() == 0)
        {
            VectorTools::project(
                dof_handler,
                constraints,
                quadrature,
                PrescribedData::InitialValues<dim>(),
                solution);
        }
        
        assemble_system();
    }

    template<int dim>
    void NonlinearSchroedingerEquation<dim>::assemble_system()
    {
        pcout << "Assembling system matrices... " << std::flush;
        TimerOutput::Scope timer_scope(computing_timer, "Assemble System");

        stiffness_matrix = 0.;
        old_stiffness_matrix = 0.;

        FEValues<dim> fe_v(
            fe,
            quadrature,
            update_values | update_gradients | update_quadrature_points | update_JxW_values);
        
        const unsigned int dofs_per_cell = fe_v.dofs_per_cell;
        const unsigned int n_q_points = fe_v.n_quadrature_points;

        std::vector<types::global_cell_index> local_dof_indices(dofs_per_cell);

        FullMatrix<double> cell_stiffness(
            dofs_per_cell,
            dofs_per_cell);
        FullMatrix<double> old_cell_stiffness(
            dofs_per_cell,
            dofs_per_cell);

        PrescribedData::Potential<dim> potential;

        for (const auto& cell : dof_handler.active_cell_iterators())
        {
            if (cell->is_locally_owned())
            {
                fe_v.reinit(cell);
                cell->get_dof_indices(local_dof_indices);

                cell_stiffness = 0.;
                old_cell_stiffness = 0.;

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    const unsigned int c_i = fe.system_to_component_index(i).first;
                    const double mass_sign = c_i == 0 ? 1. : -1.;

                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                        const unsigned int c_j = fe.system_to_component_index(j).first;

                        if (c_i == c_j)
                        {
                            double shape_grad_product = 0.;
                            double shape_voltage_product = 0.;

                            for (unsigned int q = 0; q < n_q_points; ++q)
                            {
                                const Point<dim>& p = fe_v.quadrature_point(q);

                                shape_grad_product += fe_v.JxW(q)
                                    * (fe_v.shape_grad(i, q)
                                    * fe_v.shape_grad(j, q));
                                shape_voltage_product += fe_v.JxW(q)
                                    * potential.value(p)
                                    * fe_v.shape_value(i, q)
                                    * fe_v.shape_value(j, q);
                            }

                            cell_stiffness(i, j) += 0.5 * time_step
                                * (0.5 * shape_grad_product + shape_voltage_product);
                            old_cell_stiffness(i, j) += 0.5 * time_step
                                * (0.5 * shape_grad_product + shape_voltage_product);
                        }
                        else
                        {
                            double shape_value_product = 0.;

                            for (unsigned int q = 0; q < n_q_points; ++q)
                            {
                                shape_value_product += fe_v.JxW(q)
                                    * fe_v.shape_value(i, q)
                                    * fe_v.shape_value(j, q);
                            }

                            cell_stiffness(i, j) += mass_sign * shape_value_product;
                            old_cell_stiffness(i, j) -= mass_sign * shape_value_product;
                        }
                    }
                }

                constraints.distribute_local_to_global(
                    cell_stiffness,
                    local_dof_indices,
                    stiffness_matrix);
                constraints.distribute_local_to_global(
                    old_cell_stiffness,
                    local_dof_indices,
                    old_stiffness_matrix);
            }
        }

        stiffness_matrix.compress(VectorOperation::add);
        old_stiffness_matrix.compress(VectorOperation::add);

        pcout << "done." << std::endl;
    }

    template<int dim>
    void NonlinearSchroedingerEquation<dim>::old_residual()
    {
        pcout << "\tComputing old residual... " << std::flush;
        TimerOutput::Scope timer_scope(computing_timer, "Old Residual");

        old_solution_residual = 0.;
        
        FEValues<dim> fe_v(
            fe,
            quadrature,
            update_values | update_JxW_values);
        
        const unsigned int dofs_per_cell = fe_v.dofs_per_cell;
        const unsigned int n_q_points = fe_v.n_quadrature_points;

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        Table<2, double> function_values(n_q_points, 2);

        Vector<double> cell_C(dofs_per_cell);

        for (const auto& cell : dof_handler.active_cell_iterators())
        {
            if (cell->is_locally_owned())
            {
                fe_v.reinit(cell);
                cell->get_dof_indices(local_dof_indices);

                cell_C = 0.;

                for (unsigned int q = 0; q < n_q_points; ++q)
                {
                    function_values[q][0] = 0.;
                    function_values[q][1] = 0.;
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                        const unsigned int c_j = fe.system_to_component_index(j).first;

                        function_values[q][c_j] += ghost_solution[local_dof_indices[j]]
                            * fe_v.shape_value(j, q);
                    }
                }

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    const unsigned int c_i = fe.system_to_component_index(i).first;

                    for (unsigned int q = 0; q < n_q_points; ++q)
                    {
                        cell_C[i] += 0.5 * time_step * kappa * fe_v.JxW(q)
                            * fe_v.shape_value(i, q)
                            * (function_values[q][c_i]
                                * (function_values[q][c_i]
                                    * function_values[q][c_i]
                                    + function_values[q][1 - c_i]
                                    * function_values[q][1 - c_i]));
                    }
                }

                constraints.distribute_local_to_global(
                    cell_C,
                    local_dof_indices,
                    old_solution_residual);
            }
        }

        old_solution_residual.compress(VectorOperation::add);

        old_stiffness_matrix.vmult_add(old_solution_residual, solution);

        old_solution_residual_ready = true;
        pcout << "done." << std::endl;
    }

    template<int dim>
    void NonlinearSchroedingerEquation<dim>::residual(
        const LA::MPI::Vector& y,
        LA::MPI::Vector& out)
    {
        Assert(old_solution_residual_ready, ExcInternalError());

        pcout << "\tComputing residual... " << std::flush;
        TimerOutput::Scope timer_scope(computing_timer, "Residual");

        out = 0.;
        jacobian_C = 0.;

        FEValues<dim> fe_v(
            fe,
            quadrature,
            update_values | update_JxW_values);
        
        const unsigned int dofs_per_cell = fe_v.dofs_per_cell;
        const unsigned int n_q_points = fe_v.n_quadrature_points;

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

        std::vector<Sacado::Fad::DFad<double>> dof_values(dofs_per_cell);
        Table<2, Sacado::Fad::DFad<double>> function_values(n_q_points, 2);

        Vector<double> cell_C(dofs_per_cell);
        FullMatrix<double> cell_jacobian(
            dofs_per_cell,
            dofs_per_cell);
        
        for (const auto& cell : dof_handler.active_cell_iterators())
        {
            if (cell->is_locally_owned())
            {
                fe_v.reinit(cell);
                cell->get_dof_indices(local_dof_indices);

                cell_C = 0.;
                cell_jacobian = 0.;

                for (unsigned int k = 0; k < dofs_per_cell; ++k)
                {
                    dof_values[k] = y[local_dof_indices[k]];
                    dof_values[k].diff(k, dofs_per_cell);
                }

                for (unsigned int q = 0; q < n_q_points; ++q)
                {
                    function_values[q][0] = 0.;
                    function_values[q][1] = 0.;
                    for (unsigned int j = 0; j < dofs_per_cell; ++j)
                    {
                        const unsigned int c_j = fe.system_to_component_index(j).first;

                        function_values[q][c_j] += dof_values[j]
                            * fe_v.shape_value(j, q);
                    }
                }

                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                    const unsigned int c_i = fe.system_to_component_index(i).first;

                    Sacado::Fad::DFad<double> C_i = 0.;

                    for (unsigned int q = 0; q < n_q_points; ++q)
                    {
                        C_i += 0.5 * time_step * kappa * fe_v.JxW(q)
                            * fe_v.shape_value(i, q)
                            * (function_values[q][c_i]
                                * (function_values[q][c_i]
                                    * function_values[q][c_i]
                                    + function_values[q][1 - c_i]
                                    * function_values[q][1 - c_i]));
                    }

                    cell_C[i] += C_i.val();
                    for (unsigned int k = 0; k < dofs_per_cell; ++k)
                    {
                        cell_jacobian(i, k) += C_i.fastAccessDx(k);
                    }
                }

                constraints.distribute_local_to_global(
                    cell_C,
                    local_dof_indices,
                    out);
                constraints.distribute_local_to_global(
                    cell_jacobian,
                    local_dof_indices,
                    jacobian_C);
            }
        }

        out.compress(VectorOperation::add);
        jacobian_C.compress(VectorOperation::add);

        stiffness_matrix.vmult_add(out, y);
        out.add(1., old_solution_residual);

        pcout << "norm=" << out.l2_norm() << std::endl;
    }

    template<int dim>
    void NonlinearSchroedingerEquation<dim>::setup_jacobian()
    {
        pcout << "\tSetup Jacobian system... " << std::flush;
        TimerOutput::Scope timer_scope(computing_timer, "Jacobian Setup");

        jacobian_matrix.copy_from(jacobian_C);
        jacobian_matrix.add(1., stiffness_matrix);

        pcout << "done." << std::endl;
    }

    template<int dim>
    void NonlinearSchroedingerEquation<dim>::jacobian_solve(
        const LA::MPI::Vector& y,
        LA::MPI::Vector& out,
        const double tolerance)
    {
        pcout << "\tSolving Jacobian system... " << std::flush;
        TimerOutput::Scope timer_scope(computing_timer, "Jacobian Solve");

        PETScWrappers::PreconditionNone preconditioner;
        {
            PETScWrappers::PreconditionNone::AdditionalData additional_data;
            preconditioner.initialize(jacobian_matrix, additional_data);
        }

        SolverControl solver_control(
            dof_handler.n_dofs() * param.max_iterations,
            tolerance);
        LA::SolverGMRES solver(solver_control, mpi_communicator);

        solver.solve(jacobian_matrix, out, y, preconditioner);
        constraints.distribute(out);

        pcout << "done in " << solver_control.last_step() << " iterations." << std::endl;
    }

    template<int dim>
    void NonlinearSchroedingerEquation<dim>::output_results() const
    {
        DataOut<dim> data_out;
        DataPostprocessors::ComplexRealPart<dim> complex_re("Psi");
        DataPostprocessors::ComplexImagPart<dim> complex_im("Psi");
        DataPostprocessors::ComplexAmplitude<dim> complex_mag("Psi");
        DataPostprocessors::ComplexPhase<dim> complex_arg("Psi");

        data_out.attach_dof_handler(dof_handler);

        data_out.add_data_vector(ghost_solution, complex_re);
        data_out.add_data_vector(ghost_solution, complex_im);
        data_out.add_data_vector(ghost_solution, complex_mag);
        data_out.add_data_vector(ghost_solution, complex_arg);

        Vector<float> subdomain(triangulation.n_active_cells());
        for (unsigned int i = 0; i < subdomain.size(); ++i)
        {
            subdomain(i) = triangulation.locally_owned_subdomain();
        }

        data_out.add_data_vector(subdomain, "subdomain");

        data_out.build_patches();

        std::filesystem::path path(param.output_prefix);
        std::string filename = path.filename().string();
        std::filesystem::path prefix = path.remove_filename();

        std::filesystem::create_directories(prefix);
        data_out.write_vtu_with_pvtu_record(
            prefix.string(),
            filename,
            timestep_number,
            mpi_communicator,
            int(std::log10(param.n_time_steps)) + 1,
            4);
    }

    template<int dim>
    void NonlinearSchroedingerEquation<dim>::serialize(
        boost::archive::binary_iarchive& ar,
        const unsigned int)
    {
        ar >> timestep_number >> solution;
        time = timestep_number * time_step;
    }

    template<int dim>
    void NonlinearSchroedingerEquation<dim>::serialize(
        boost::archive::binary_oarchive& ar,
        const unsigned int)
    {
        ar << timestep_number << solution;
    }

    template<int dim>
    void NonlinearSchroedingerEquation<dim>::run()
    {
        output_results();

        const unsigned int mod_output_steps = param.n_output_files
            ? param.n_time_steps / param.n_output_files
            : param.n_time_steps + 1;
        const unsigned int mod_checkpoint_steps = param.n_checkpoints
            ? param.n_time_steps / param.n_checkpoints
            : param.n_time_steps + 1;

        SUNDIALS::KINSOL<LA::MPI::Vector>::AdditionalData additional_data;
        additional_data.function_tolerance = param.tolerance;
        SUNDIALS::KINSOL<LA::MPI::Vector> solver(
            additional_data,
            mpi_communicator);
        
        solver.reinit_vector = [this](
            LA::MPI::Vector& y)
        {
            y.reinit(this->locally_owned_dofs, this->mpi_communicator);
        };
        solver.residual = [this](
            const LA::MPI::Vector& y,
            LA::MPI::Vector& out)
        {
            this->residual(y, out);
        };
        solver.setup_jacobian = [this](
            const LA::MPI::Vector&,
            const LA::MPI::Vector&)
        {
            this->setup_jacobian();
        };
        solver.solve_with_jacobian = [this](
            const LA::MPI::Vector& y,
            LA::MPI::Vector& out,
            const double tolerance)
        {
            this->jacobian_solve(y, out, tolerance);
        };

        while (timestep_number < param.n_time_steps)
        {
            pcout << "Time step " << ++timestep_number << ':' << std::endl;

            ghost_solution = solution;
            old_residual();

            solver.solve(solution);

            if (timestep_number % mod_output_steps == 0)
            {
                output_results();
            }
            if (timestep_number % mod_checkpoint_steps == 0)
            {
                this->checkpoint(timestep_number);
            }
        }
    }
}
