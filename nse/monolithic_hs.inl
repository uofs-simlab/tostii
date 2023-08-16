#pragma once

#include <filesystem>

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

        mass_matrix.reinit(
            locally_owned_dofs,
            locally_owned_dofs,
            sparsity_pattern,
            mpi_communicator);
        minus_A_minus_B.reinit(
            locally_owned_dofs,
            locally_owned_dofs,
            sparsity_pattern,
            mpi_communicator);
        jacobian_C.reinit(
            locally_owned_dofs,
            locally_owned_dofs,
            sparsity_pattern,
            mpi_communicator);
        system_matrix.reinit(
            locally_owned_dofs,
            locally_owned_dofs,
            sparsity_pattern,
            mpi_communicator);

        solution.reinit(locally_owned_dofs, mpi_communicator);
        ghost_solution.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
        temp.reinit(locally_owned_dofs, mpi_communicator);
        ghost_temp.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);

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

        mass_matrix = 0.;
        minus_A_minus_B = 0.;

        FEValues<dim> fe_v(
            fe,
            quadrature,
            update_values | update_gradients | update_quadrature_points | update_JxW_values);
        
        const unsigned int dofs_per_cell = fe_v.dofs_per_cell;
        const unsigned int n_q_points = fe_v.n_quadrature_points;

        std::vector<types::global_cell_index> local_dof_indices(dofs_per_cell);

        FullMatrix<double> cell_mass(dofs_per_cell);
        FullMatrix<double> cell_AB(dofs_per_cell);

        PrescribedData::Potential<dim> potential;

        for (const auto& cell : dof_handler.active_cell_iterators())
        {
            if (cell->is_locally_owned())
            {
                fe_v.reinit(cell);
                cell->get_dof_indices(local_dof_indices);

                cell_mass = 0.;
                cell_AB = 0.;

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

                            cell_AB(i, j) -= 0.5 * shape_grad_product + shape_voltage_product;
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

                            cell_mass(i, j) += mass_sign * shape_value_product;
                        }
                    }
                }

                constraints.distribute_local_to_global(
                    cell_mass,
                    local_dof_indices,
                    mass_matrix);
                constraints.distribute_local_to_global(
                    cell_AB,
                    local_dof_indices,
                    minus_A_minus_B);
            }
        }

        mass_matrix.compress(VectorOperation::add);
        minus_A_minus_B.compress(VectorOperation::add);

        pcout << "done." << std::endl;
    }

    template<int dim>
    void NonlinearSchroedingerEquation<dim>::rhs(
        const LA::MPI::Vector& y,
        LA::MPI::Vector& out)
    {
        pcout << "\tComputing RHS... " << std::flush;
        TimerOutput::Scope timer_scope(computing_timer, "RHS");

        temp = 0.;
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
        FullMatrix<double> cell_jacobian(dofs_per_cell);

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
                        C_i += kappa * fe_v.JxW(q)
                            * fe_v.shape_value(i, q)
                            * (function_values[q][c_i]
                                * (function_values[q][c_i]
                                    * function_values[q][c_i]
                                    + function_values[q][1 - c_i]
                                    * function_values[q][1 - c_i]));
                    }

                    cell_C(i) -= C_i.val();
                    for (unsigned int k = 0; k < dofs_per_cell; ++k)
                    {
                        cell_jacobian(i, k) -= C_i.fastAccessDx(k);
                    }
                }

                constraints.distribute_local_to_global(
                    cell_C,
                    local_dof_indices,
                    temp);
                constraints.distribute_local_to_global(
                    cell_jacobian,
                    local_dof_indices,
                    jacobian_C);
            }
        }

        temp.compress(VectorOperation::add);
        jacobian_C.compress(VectorOperation::add);

        minus_A_minus_B.vmult_add(temp, y);
        constraints.distribute(temp);
        
        const unsigned int n_iter = solve(mass_matrix, out, temp);

        pcout << "done in " << n_iter << " iterations." << std::endl;
    }

    template<int dim>
    void NonlinearSchroedingerEquation<dim>::jacobian_solve(
        const double tau,
        const LA::MPI::Vector& y,
        LA::MPI::Vector& out)
    {
        pcout << "\tSolving Jacobian system... " << std::flush;
        TimerOutput::Scope timer_scope(computing_timer, "Jacobian Solve");

        mass_matrix.vmult(temp, y);

        system_matrix.copy_from(mass_matrix);
        system_matrix.add(-tau, minus_A_minus_B);
        system_matrix.add(-tau, jacobian_C);

        const unsigned int n_iter = solve(system_matrix, out, temp);

        pcout << "done in " << n_iter << " iterations." << std::endl;
    }

    template<int dim>
    unsigned int NonlinearSchroedingerEquation<dim>::solve(
        const LA::MPI::SparseMatrix& A,
        LA::MPI::Vector& x,
        const LA::MPI::Vector& b) const
    {
        LA::MPI::PreconditionAMG preconditioner;
        {
            LA::MPI::PreconditionAMG::AdditionalData additional_data;
            preconditioner.initialize(A, additional_data);
        }

        SolverControl solver_control(
            param.max_iterations * dof_handler.n_dofs(),
            param.tolerance);
        LA::SolverGMRES solver(solver_control, mpi_communicator);

        solver.solve(A, x, b, preconditioner);

        return solver_control.last_step();
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

        tostii::TimeStepping::ImplicitRungeKutta<LA::MPI::Vector> stepper(param.rk_method);
        const auto stepper_rhs = [this](
            const double,
            const LA::MPI::Vector& y,
            LA::MPI::Vector& out)
        {
            this->ghost_temp = y;
            this->rhs(this->ghost_temp, out);
        };
        const auto stepper_lhs = [this](
            const double,
            const double tau,
            const LA::MPI::Vector& y,
            LA::MPI::Vector& out)
        {
            this->ghost_temp = y;
            this->jacobian_solve(tau, this->ghost_temp, out);
        };

        while (timestep_number < param.n_time_steps)
        {
            pcout << "Time step " << ++timestep_number << ':' << std::endl;

            time = stepper.evolve_one_time_step(
                stepper_rhs,
                stepper_lhs,
                time,
                time_step,
                solution);
            
            if (timestep_number % mod_output_steps == 0)
            {
                ghost_solution = solution;
                output_results();
            }
            if (timestep_number % mod_checkpoint_steps == 0)
            {
                this->checkpoint(timestep_number);
            }
        }
    }
}
