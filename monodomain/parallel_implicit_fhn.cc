/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2025 by the authors listed below
 *
 * This file is NOT part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 *
 * ---------------------------------------------------------------------

 *
 * Author: Mohammad Mahdi Moayeri, University of Saskatchewan, 2021
 *
 * ---------------------------------------------------------------------
 * Parallel Fully Implicit FitzHugh-Nagumo solver using tostii
 * Uses MPI parallel computing and implicit time integration for stability
 * Based on the monodomain.cc parallel structure
 * ---------------------------------------------------------------------
 */

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <cmath>

#include <deal.II/lac/generic_linear_algebra.h>

// Use PETSc for parallel linear algebra
namespace LA
{
  using namespace dealii::LinearAlgebraPETSc;
}

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <iostream>
#include <cmath>
#include <iomanip>

// Include tostii for time integration
#include <tostii/tostii.h>

using namespace dealii;

template <int dim>
class ParallelImplicitFHN {
public:
  ParallelImplicitFHN();
  void run();

private:
  void setup_system();
  void assemble_matrices();
  void set_initial_conditions();
  
  // ODE right-hand side: dy/dt = f(y) = diffusion + reaction
  void evaluate_rhs(const double time,
                    const LA::MPI::Vector& y_in,
                    LA::MPI::Vector& y_out);
  
  // Implicit solver: solve (I - \tau*J_f)*y_out = y_in for tostii
  void id_minus_tau_J_inverse(const double time,
                              const double tau,
                              const LA::MPI::Vector& y_in,
                              LA::MPI::Vector& y_out);
  
  void assemble_jacobian_matrix(const double tau, const LA::MPI::Vector& solution_state);
  
  void output_results(const unsigned int timestep_number) const;

  // MPI communication
  MPI_Comm mpi_communicator;
  
  // Parallel triangulation and finite elements
  parallel::distributed::Triangulation<dim> triangulation;
  FESystem<dim> fe;
  DoFHandler<dim> dof_handler;
  
  // Index sets for parallel computing
  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;
  
  // Constraints for hanging nodes and boundary conditions
  AffineConstraints<double> constraints;

  // Parallel matrices and vectors
  LA::MPI::SparseMatrix mass_matrix;
  LA::MPI::SparseMatrix laplace_matrix;
  LA::MPI::SparseMatrix jacobian_matrix;
  LA::MPI::SparseMatrix system_matrix;
  
  LA::MPI::Vector solution;
  LA::MPI::Vector locally_relevant_solution;
  LA::MPI::Vector system_rhs;
  
  // Output stream that only prints on processor 0
  ConditionalOStream pcout;
  
  // Timer for performance measurement
  TimerOutput computing_timer;
  
  // Kinetic model I parameters (Moayeri et al., 2D test case)
  const double epsilon      = 0.005;  // \epsilon
  const double a_param      = 0.3;    // a
  const double b_param      = 0.01;   // b
  const double alpha_param  = 1.0;    // \alpha
  const double gamma        = 1.0;    // \gamma

  const double D_v = 1.0;
  const double D_w = 0.0;

  const unsigned int n_refinements = 6;   
  const double final_time          = 8.0;
  const double time_step           = 0.0005; 

};

template <int dim>
ParallelImplicitFHN<dim>::ParallelImplicitFHN()
  : mpi_communicator(MPI_COMM_WORLD)
  , triangulation(mpi_communicator,
                  typename Triangulation<dim>::MeshSmoothing(
                    Triangulation<dim>::smoothing_on_refinement |
                    Triangulation<dim>::smoothing_on_coarsening))
  , fe(FE_Q<dim>(2), 2)  // Two components: v and w
  , dof_handler(triangulation)
  , pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
  , computing_timer(mpi_communicator, pcout, TimerOutput::summary, TimerOutput::wall_times)
{}

template <int dim>
void ParallelImplicitFHN<dim>::setup_system()
{
  TimerOutput::Scope t(computing_timer, "Setup system");

  // Create square domain [-30,30] x [-30,30] for wave observation
  GridGenerator::hyper_cube(triangulation, -30, 30);
  triangulation.refine_global(n_refinements);

  // Distribute DoFs in parallel; this gives each MPI rank
  // a contiguous block of global indices (required by PETSc).
  dof_handler.distribute_dofs(fe);

  // Get locally owned and relevant DoFs
  locally_owned_dofs = dof_handler.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

  // (Optional but helpful sanity check)
  Assert(locally_owned_dofs.is_contiguous(),
         ExcMessage("PETSc requires contiguous locally owned DoFs."));

  const std::vector<types::global_dof_index> dofs_per_component =
    DoFTools::count_dofs_per_fe_component(dof_handler);

  pcout << "Number of active cells: "
        << triangulation.n_global_active_cells() << std::endl;
  pcout << "Number of degrees of freedom: " << dof_handler.n_dofs()
        << " (" << dofs_per_component[0] << " + " << dofs_per_component[1]
        << ")" << std::endl;

  // Setup constraints (hanging nodes, boundary conditions)
  constraints.clear();
  constraints.reinit(locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  // Homogeneous Neumann boundary conditions are natural -> no extra constraints
  constraints.close();

  // Create sparsity pattern on locally relevant DoFs
  DynamicSparsityPattern dsp(locally_relevant_dofs);
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);

  // Distribute sparsity to a parallel pattern compatible with PETSc
  SparsityTools::distribute_sparsity_pattern(dsp,
                                             locally_owned_dofs,
                                             mpi_communicator,
                                             locally_relevant_dofs);

  // Initialize matrices with contiguous row/column index sets
  mass_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);
  laplace_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);
  jacobian_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);
  system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);

  // Initialize vectors
  solution.reinit(locally_owned_dofs, mpi_communicator);
  locally_relevant_solution.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
  system_rhs.reinit(locally_owned_dofs, mpi_communicator);
}

template <int dim>
void ParallelImplicitFHN<dim>::assemble_matrices() {
  TimerOutput::Scope t(computing_timer, "Assemble matrices");
  
  mass_matrix = 0;
  laplace_matrix = 0;
  
  const QGauss<dim> quadrature_formula(fe.degree + 1);
  FEValues<dim> fe_values(fe, quadrature_formula,
                         update_values | update_gradients | update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points = quadrature_formula.size();

  FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_laplace_matrix(dofs_per_cell, dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto& cell : dof_handler.active_cell_iterators()) {
    if (cell->is_locally_owned()) {
      fe_values.reinit(cell);
      cell_mass_matrix = 0;
      cell_laplace_matrix = 0;

      for (unsigned int q = 0; q < n_q_points; ++q) {
        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          const unsigned int component_i = fe.system_to_component_index(i).first;
          
          for (unsigned int j = 0; j < dofs_per_cell; ++j) {
            const unsigned int component_j = fe.system_to_component_index(j).first;
            
            // Mass matrix (block diagonal)
            if (component_i == component_j) {
              cell_mass_matrix(i, j) += fe_values.shape_value(i, q) *
                                       fe_values.shape_value(j, q) *
                                       fe_values.JxW(q);
            }
            
            // Laplace matrix with diffusion coefficients
            if (component_i == component_j) {
              double diffusion_coeff = (component_i == 0) ? D_v : D_w;
              cell_laplace_matrix(i, j) -= diffusion_coeff *
                                          fe_values.shape_grad(i, q) *
                                          fe_values.shape_grad(j, q) *
                                          fe_values.JxW(q);
            }
          }
        }
      }

      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(cell_mass_matrix, local_dof_indices, mass_matrix);
      constraints.distribute_local_to_global(cell_laplace_matrix, local_dof_indices, laplace_matrix);
    }
  }

  mass_matrix.compress(VectorOperation::add);
  laplace_matrix.compress(VectorOperation::add);
}

template <int dim>
void ParallelImplicitFHN<dim>::set_initial_conditions()
{
  TimerOutput::Scope t(computing_timer, "Set initial conditions");

  // 2D Kinetic model I initial conditions (Eqs. (5.5)–(5.6))
  // V(x,0) = [1 + exp(4(|x| - δ1))]^{-2} - [1 + exp(4(|x| - δ2))]^{-2}
  //          if (x < 0 OR y > 5), otherwise 0
  //
  // W(x,0) = θ if (x < λ1 AND y < λ2), otherwise 0
  // with δ1 = 5, δ2 = 1, θ = 0.1, λ1 = 1, λ2 = 10.

  class InitialCondition : public Function<dim>
{
public:
  InitialCondition()
    : Function<dim>(2)
    , delta1(5.0)
    , delta2(1.0)
    , theta(0.1)
    , lambda1(1.0)
    , lambda2(10.0)
  {}

  virtual double value(const Point<dim> &p,
                       const unsigned int component) const override
  {
    const double x = p[0];
    const double y = p[1];

    if (component == 0)
      {
        // V initial condition: non-zero only for y < 5
        if (y >= 5.0)
          return 0.0;

        const double abs_x = std::fabs(x);

        const double term1 =
          1.0 / std::pow(1.0 + std::exp(4.0 * (abs_x - delta1)), 2.0);
        const double term2 =
          1.0 / std::pow(1.0 + std::exp(4.0 * (abs_x - delta2)), 2.0);

        return term1 - term2;
      }
    else
      {
        // W initial condition
        if ((x < lambda1) && (y < lambda2))
          return theta;
        else
          return 0.0;
      }
  }

private:
  const double delta1;
  const double delta2;
  const double theta;
  const double lambda1;
  const double lambda2;
};


  VectorTools::interpolate(dof_handler, InitialCondition(), solution);

  // Apply constraints and update parallel vectors
  constraints.distribute(solution);
  solution.compress(VectorOperation::insert);
  locally_relevant_solution = solution;

  pcout << "Initial conditions set according to 2D Kinetic model I test case" << std::endl;
}

template <int dim>
void ParallelImplicitFHN<dim>::evaluate_rhs(const double /*time*/,
                                           const LA::MPI::Vector& y_in,
                                           LA::MPI::Vector& y_out) {
  // Compute f(y) = M^{-1} * [L*y + M*f_reaction(y)]
  // This gives the proper ODE form dy/dt = f(y)
  
  y_out.reinit(locally_owned_dofs, mpi_communicator);
  LA::MPI::Vector diffusion_part(locally_owned_dofs, mpi_communicator);
  LA::MPI::Vector reaction_part(locally_owned_dofs, mpi_communicator);
  
  const std::vector<types::global_dof_index> dofs_per_component =
    DoFTools::count_dofs_per_fe_component(dof_handler);
  const unsigned int n_v = dofs_per_component[0];
  
  // Diffusion part: L*y_in
  laplace_matrix.vmult(diffusion_part, y_in);
  
  // Reaction part: assemble M*f_reaction(y) into reaction_part
  reaction_part = 0;
  
  const QGauss<dim> quadrature_formula(fe.degree + 1);
  FEValues<dim> fe_values(fe, quadrature_formula, update_values | update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points = quadrature_formula.size();

  Vector<double> cell_reaction_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  std::vector<double> local_solution_values(dofs_per_cell);

  // Update ghost values for evaluation
  LA::MPI::Vector y_in_with_ghosts(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
  y_in_with_ghosts = y_in;

  for (const auto& cell : dof_handler.active_cell_iterators()) {
    if (cell->is_locally_owned()) {
      fe_values.reinit(cell);
      cell_reaction_rhs = 0;
      
      cell->get_dof_indices(local_dof_indices);
      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
        local_solution_values[i] = y_in_with_ghosts[local_dof_indices[i]];
      }

      for (unsigned int q = 0; q < n_q_points; ++q) {
        // Compute v and w at quadrature point
        double v_q = 0, w_q = 0;
        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          const unsigned int component_i = fe.system_to_component_index(i).first;
          if (component_i == 0) v_q += local_solution_values[i] * fe_values.shape_value(i, q);
          else w_q += local_solution_values[i] * fe_values.shape_value(i, q);
        }
        
        // FitzHugh-Nagumo reaction terms
        // const double f_v = v_q - v_q * v_q * v_q / 3.0 - w_q;
        // const double f_w = epsilon * (v_q + beta - gamma * w_q);

                // Kinetic model I reaction terms
        const double V_th = (w_q + b_param) / a_param;

        const double f_v =
          (1.0 / epsilon) * v_q * (1.0 - v_q) * (v_q - V_th);

        const double f_w =
          alpha_param * v_q - gamma * w_q;

        
        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          const unsigned int component_i = fe.system_to_component_index(i).first;
          
          if (component_i == 0) {
            cell_reaction_rhs[i] += f_v * fe_values.shape_value(i, q) * fe_values.JxW(q);
          } else {
            cell_reaction_rhs[i] += f_w * fe_values.shape_value(i, q) * fe_values.JxW(q);
          }
        }
      }

      constraints.distribute_local_to_global(cell_reaction_rhs, local_dof_indices, reaction_part);
    }
  }
  
  reaction_part.compress(VectorOperation::add);
  
  // Combine: f(y) = M^{-1} * [L*y + M*f_reaction(y)]
  diffusion_part += reaction_part;
  
  // Solve M * y_out = diffusion_part (apply mass matrix inverse)
  SolverControl solver_control(1000, 1e-12);
  LA::SolverCG solver(solver_control, mpi_communicator);
  LA::MPI::PreconditionAMG preconditioner;
  LA::MPI::PreconditionAMG::AdditionalData data;
  data.symmetric_operator = true;
  preconditioner.initialize(mass_matrix, data);
  
  solver.solve(mass_matrix, y_out, diffusion_part, preconditioner);
  constraints.distribute(y_out);
}

template <int dim>
void ParallelImplicitFHN<dim>::assemble_jacobian_matrix(const double tau, 
                                                       const LA::MPI::Vector& solution_state) {
  jacobian_matrix = 0;
  
  const QGauss<dim> quadrature_formula(fe.degree + 1);
  FEValues<dim> fe_values(fe, quadrature_formula, update_values | update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points = quadrature_formula.size();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  std::vector<double> local_solution_values(dofs_per_cell);

  // Update ghost values for evaluation
  LA::MPI::Vector solution_with_ghosts(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
  solution_with_ghosts = solution_state;

  for (const auto& cell : dof_handler.active_cell_iterators()) {
    if (cell->is_locally_owned()) {
      fe_values.reinit(cell);
      cell_matrix = 0;
      
      cell->get_dof_indices(local_dof_indices);
      
      // Get local solution values
      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
        local_solution_values[i] = solution_with_ghosts[local_dof_indices[i]];
      }

            for (unsigned int q = 0; q < n_q_points; ++q)
        {
          // Compute v and w at quadrature point
          double v_q = 0.0, w_q = 0.0;
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              const unsigned int component_i =
                fe.system_to_component_index(i).first;
              if (component_i == 0)
                v_q += local_solution_values[i] * fe_values.shape_value(i, q);
              else
                w_q += local_solution_values[i] * fe_values.shape_value(i, q);
            }

          const double V_th = (w_q + b_param) / a_param;

          // Derivatives of reaction terms (Kinetic model I)
          const double dfv_dv =
            ((1.0 - 2.0 * v_q) * (v_q - V_th) + v_q - v_q * v_q) / epsilon;

          const double dfv_dw =
            - v_q * (1.0 - v_q) / (a_param * epsilon);

          const double dfw_dv = alpha_param;
          const double dfw_dw = -gamma;

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              const unsigned int component_i =
                fe.system_to_component_index(i).first;

              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  const unsigned int component_j =
                    fe.system_to_component_index(j).first;

                  double jacobian_entry = 0.0;

                  if (component_i == 0 && component_j == 0)
                    jacobian_entry = dfv_dv;
                  else if (component_i == 0 && component_j == 1)
                    jacobian_entry = dfv_dw;
                  else if (component_i == 1 && component_j == 0)
                    jacobian_entry = dfw_dv;
                  else if (component_i == 1 && component_j == 1)
                    jacobian_entry = dfw_dw;

                  cell_matrix(i, j) += jacobian_entry *
                                       fe_values.shape_value(i, q) *
                                       fe_values.shape_value(j, q) *
                                       fe_values.JxW(q);
                }
            }
        }

      constraints.distribute_local_to_global(cell_matrix, local_dof_indices, jacobian_matrix);
    }
  }

  jacobian_matrix.compress(VectorOperation::add);
}

template <int dim>
void ParallelImplicitFHN<dim>::id_minus_tau_J_inverse(const double /*time*/,
                                                      const double tau,
                                                      const LA::MPI::Vector &y_in,
                                                      LA::MPI::Vector       &y_out)
{
  //   mass_matrix    = M
  //   laplace_matrix = L
  //   jacobian_matrix = R'(y)   (assembled in assemble_jacobian_matrix)
  //
  // We solve (I - \tau J_f) y_out = y_in, where
  //
  //   f(y) = M^{-1} (L y + R(y))
  //   J_f(y) = M^{-1} (L + R'(y))
  //
  // Multiply by M:
  //   (M - \tau (L + R'(y))) y_out = M y_in
  //

  // 1. Assemble the reaction Jacobian R'(y) at current state y_in
  assemble_jacobian_matrix(tau, y_in);

  // 2. Form system matrix: A = M - \tau L - \tau R'(y)
  system_matrix.copy_from(mass_matrix);      // A = M
  system_matrix.add(-tau, laplace_matrix);   // A -= \tau L
  system_matrix.add(-tau, jacobian_matrix);  // A -= \tau R'(y)

  // 3. Right-hand side: b = M y_in
  mass_matrix.vmult(system_rhs, y_in);

  // 4. Solve A y_out = b with GMRES + AMG preconditioner
  SolverControl       solver_control(2000, 1e-8);
  LA::SolverGMRES     solver(solver_control, mpi_communicator);
  LA::MPI::PreconditionAMG               preconditioner;
  LA::MPI::PreconditionAMG::AdditionalData data;
  data.symmetric_operator = false; // non-symmetric due to reaction terms

  preconditioner.initialize(system_matrix, data);

  y_out.reinit(locally_owned_dofs, mpi_communicator);
  solver.solve(system_matrix, y_out, system_rhs, preconditioner);

  constraints.distribute(y_out);
}

template <int dim>
void ParallelImplicitFHN<dim>::output_results(const unsigned int timestep_number) const {
  
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  
  std::vector<std::string> component_names(2);
  component_names[0] = "v";
  component_names[1] = "w";
  
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    component_interpretation(2, DataComponentInterpretation::component_is_scalar);
  
  data_out.add_data_vector(locally_relevant_solution, component_names, 
                          DataOut<dim>::type_dof_data, 
                          component_interpretation);
  
  data_out.build_patches();
  
  const std::string filename = "fhn_parallel-" + 
                              Utilities::int_to_string(timestep_number, 4) + 
                              "." + 
                              Utilities::int_to_string(triangulation.locally_owned_subdomain(), 4);
  std::ofstream output((filename + ".vtu").c_str());
  data_out.write_vtu(output);
  
  // Write master record on processor 0
  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
    std::vector<std::string> filenames;
    for (unsigned int i = 0; i < Utilities::MPI::n_mpi_processes(mpi_communicator); ++i) {
      filenames.push_back("fhn_parallel-" + 
                         Utilities::int_to_string(timestep_number, 4) + 
                         "." + 
                         Utilities::int_to_string(i, 4) + 
                         ".vtu");
    }
    std::ofstream master_output(("fhn_parallel-" + 
                                Utilities::int_to_string(timestep_number, 4) + 
                                ".pvtu").c_str());
    data_out.write_pvtu_record(master_output, filenames);
  }
}

template <int dim>
void ParallelImplicitFHN<dim>::run() {
  pcout << "Setting up parallel system..." << std::endl;
  setup_system();
  
  pcout << "Assembling matrices..." << std::endl;
  assemble_matrices();
  
  pcout << "Setting initial conditions..." << std::endl;
  set_initial_conditions();
  
  // Setup fully implicit time integration
  pcout << "Starting parallel fully implicit time integration..." << std::endl;
  
  tostii::runge_kutta_method method = tostii::SDIRK_TWO_STAGES; // Implicit method
  tostii::ImplicitRungeKutta<LA::MPI::Vector, double> time_stepper(method);
  
  double time = 0.0;
  unsigned int timestep_number = 0;
  const unsigned int n_time_steps = static_cast<unsigned int>(final_time / time_step);
  
  // Output initial condition
  locally_relevant_solution = solution;
  output_results(timestep_number);
  
  for (unsigned int step = 0; step < n_time_steps; ++step) {
    // Use tostii for implicit time integration
    time = time_stepper.evolve_one_time_step(
      [this](const double t, const LA::MPI::Vector& y, LA::MPI::Vector& dydt) {
        this->evaluate_rhs(t, y, dydt);  // ODE right-hand side
      },
      [this](const double t, const double tau, const LA::MPI::Vector& y, LA::MPI::Vector& result) {
        this->id_minus_tau_J_inverse(t, tau, y, result);  // Implicit solve
      },
      time, time_step, solution
    );
    
    ++timestep_number;
    
    if (timestep_number % 100 == 0) {
      pcout << "Time step " << timestep_number 
            << " at t = " << time << std::endl;
      locally_relevant_solution = solution;
      output_results(timestep_number);
    }
  }
  
  pcout << "Final time: " << time << std::endl;
  locally_relevant_solution = solution;
  output_results(timestep_number);
  
  computing_timer.print_summary();
  computing_timer.reset();
}

int main(int argc, char* argv[]) {
  try {
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    
    ParallelImplicitFHN<2> fhn_solver;
    fhn_solver.run();
  }
  catch (std::exception& exc) {
    std::cerr << std::endl << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
  catch (...) {
    std::cerr << std::endl << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
  return 0;
}
