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
#include <deal.II/lac/vector_memory.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>
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
#include <deal.II/numerics/error_estimator.h>

#include <fstream>
#include <iostream>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <limits>

// Include tostii for time integration
#include <tostii/tostii.h>

#include "InitialConditionFactory.h"

using namespace dealii;


#include <algorithm>
#include <string>
#include <memory>

namespace
{
  void clear_sundials_petsc_vector_pool(const unsigned int n_vectors = 128)
  {
    GrowingVectorMemory<LA::MPI::Vector> pool;
    std::vector<VectorMemory<LA::MPI::Vector>::Pointer> cached_vectors;
    cached_vectors.reserve(n_vectors);

    for (unsigned int i = 0; i < n_vectors; ++i)
      {
        cached_vectors.emplace_back(pool);
        cached_vectors.back()->clear();
      }
  }
} // namespace

struct ExperimentParameters
{
  // Mesh and time
  double domain_half_length = 30.0;
  unsigned int n_refinements = 6;
  unsigned int fe_degree = 2;
  double final_time = 4.0;
  double time_step = 5.0e-4;
  unsigned int output_stride = 100;
  unsigned int mesh_adaptation_frequency = 0;
  unsigned int max_delta_refinement_level = 2;
  double mesh_refinement_fraction = 0.3;
  double mesh_coarsening_fraction = 0.03;

  // ARKode time integrator
  double arkode_minimum_step_size = 1e-8;
  unsigned int arkode_maximum_order = 2;
  unsigned int arkode_maximum_non_linear_iterations = 10;
  double arkode_absolute_tolerance = 1e-6;
  double arkode_relative_tolerance = 1e-5;
  bool arkode_implicit_function_is_linear = false;
  bool arkode_implicit_function_is_time_independent = false;

  // Diagnostics
  unsigned int diagnostic_stride = 20;
  double excitation_threshold = 0.5;
  double epsilon_area = 1e-14;
  double diagnosis_start_time = 0.5;
  double ignition_max_v_threshold = 0.5;
  double ignition_max_v_fraction_of_initial = 0.25;
  double ignition_excited_fraction_threshold = 1e-3;
  unsigned int required_ignition_hits = 2;

  // Optional voltage limiter used to control nonphysical overshoots.
  bool enable_voltage_limiter = false;
  double voltage_limiter_min = -0.2;
  double voltage_limiter_max = 1.05;

  InitialConditions::Parameters initial_condition;

  // Kinetic model I parameters
  double epsilon = 0.005;
  double a_param = 0.3;
  double b_param = 0.01;
  double alpha_param = 1.0;
  double gamma = 1.0;
  double diffusion_v = 1.0;
  double diffusion_w = 0.0;
  // Kinetic model II parameter
  double      beta_param    = 0.1;
  std::string kinetic_model = "kinetic_I"; // "kinetic_I" or "kinetic_II"

  // Output configuration
  std::string output_directory = "outputs";
  std::string label_suffix = "";

  static void declare_parameters(ParameterHandler &prm);
  void parse_parameters(ParameterHandler &prm);
  std::string make_run_name() const;
  std::filesystem::path make_output_directory() const;

private:
  static std::string format_double_token(const double value);
  static std::string sanitize_token(const std::string &value);
};

struct DiagnosticsRecord
{
  double max_v = 0.0;
  double excited_area = 0.0;
  double excited_fraction = 0.0;
  double relative_excited_area = 0.0;
};

void ExperimentParameters::declare_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Mesh and Time");
  {
    prm.declare_entry("Domain half length",
                      "30.0",
                      Patterns::Double(0.0),
                      "Half-length of the square domain.");
    prm.declare_entry("Global refinements",
                      "6",
                      Patterns::Integer(0),
                      "Number of global mesh refinements.");
    prm.declare_entry("Finite element degree",
                      "2",
                      Patterns::Integer(1),
                      "Polynomial degree for FE_Q.");
    prm.declare_entry("Final time",
                      "4.0",
                      Patterns::Double(0.0),
                      "Final simulation time.");
    prm.declare_entry("Time step",
                      "5e-4",
                      Patterns::Double(0.0),
                      "Time-step size.");
    prm.declare_entry("Output stride",
                      "100",
                      Patterns::Integer(1),
                      "Write output every N time steps.");
    prm.declare_entry("Mesh adaptation frequency",
                      "0",
                      Patterns::Integer(0),
                      "Adapt the mesh every N time steps. Use 0 to disable.");
    prm.declare_entry("Maximum delta refinement level",
                      "2",
                      Patterns::Integer(0),
                      "Maximum adaptive refinement levels above the global base mesh.");
    prm.declare_entry("Mesh refinement fraction",
                      "0.3",
                      Patterns::Double(0.0, 1.0),
                      "Fraction of cells refined by the Kelly estimator.");
    prm.declare_entry("Mesh coarsening fraction",
                      "0.03",
                      Patterns::Double(0.0, 1.0),
                      "Fraction of cells coarsened by the Kelly estimator.");
  }
  prm.leave_subsection();

  prm.enter_subsection("Solution limiter");
  {
    prm.declare_entry("Enable voltage limiter",
                      "false",
                      Patterns::Bool(),
                      "Clip voltage degrees of freedom after accepted ARKode checkpoints.");
    prm.declare_entry("Minimum voltage",
                      "-0.2",
                      Patterns::Double(),
                      "Lower bound for the optional voltage limiter.");
    prm.declare_entry("Maximum voltage",
                      "1.05",
                      Patterns::Double(),
                      "Upper bound for the optional voltage limiter.");
  }
  prm.leave_subsection();

  prm.enter_subsection("Time integrator");
  {
    prm.declare_entry("ARKode minimum step size",
                      "1e-8",
                      Patterns::Double(0.0),
                      "Minimum adaptive ARKode step size.");
    prm.declare_entry("ARKode maximum order",
                      "2",
                      Patterns::Integer(1),
                      "Maximum ARKode method order.");
    prm.declare_entry("ARKode maximum nonlinear iterations",
                      "10",
                      Patterns::Integer(1),
                      "Maximum nonlinear iterations per ARKode step.");
    prm.declare_entry("ARKode absolute tolerance",
                      "1e-6",
                      Patterns::Double(0.0),
                      "Absolute tolerance for ARKode.");
    prm.declare_entry("ARKode relative tolerance",
                      "1e-5",
                      Patterns::Double(0.0),
                      "Relative tolerance for ARKode.");
    prm.declare_entry("ARKode implicit function is linear",
                      "false",
                      Patterns::Bool(),
                      "Whether the implicit RHS depends linearly on the solution.");
    prm.declare_entry("ARKode implicit function is time independent",
                      "false",
                      Patterns::Bool(),
                      "Whether the implicit RHS is time independent.");
  }
  prm.leave_subsection();

  InitialConditions::Parameters::declare_parameters(prm);

  prm.enter_subsection("Output");
  {
    prm.declare_entry("Output directory",
                      "outputs",
                      Patterns::Anything(),
                      "Parent directory for simulation outputs.");
    prm.declare_entry("Label suffix",
                      "",
                      Patterns::Anything(),
                      "Optional suffix appended to the auto-generated run name.");
  }
  prm.leave_subsection();

  prm.enter_subsection("Model");
  {
    prm.declare_entry("Epsilon",
                      "0.005",
                      Patterns::Double(0.0),
                      "Kinetic model epsilon parameter.");
    prm.declare_entry("A parameter",
                      "0.3",
                      Patterns::Double(),
                      "Kinetic model a parameter.");
    prm.declare_entry("B parameter",
                      "0.01",
                      Patterns::Double(),
                      "Kinetic model b parameter.");
    prm.declare_entry("Alpha parameter",
                      "1.0",
                      Patterns::Double(),
                      "Kinetic model alpha parameter.");
    prm.declare_entry("Gamma parameter",
                      "1.0",
                      Patterns::Double(),
                      "Kinetic model gamma parameter.");
    prm.declare_entry("Diffusion v",
                      "1.0",
                      Patterns::Double(),
                      "Diffusion coefficient for v.");
    prm.declare_entry("Diffusion w",
                      "0.0",
                      Patterns::Double(),
                      "Diffusion coefficient for w.");
    prm.declare_entry("Beta parameter",
                      "0.1",
                      Patterns::Double(),
                      "Excitation threshold beta for Kinetic Model II: "
                      "f_v = v(1-v)(v-beta)-w.");
    prm.declare_entry("Kinetic model",
                      "kinetic_I",
                      Patterns::Selection("kinetic_I|kinetic_II"),
                      "Which reaction kinetics to use. "
                      "kinetic_I: (1/eps)*v(1-v)(v-(w+b)/a), alpha*v-gamma*w. "
                      "kinetic_II: v(1-v)(v-beta)-w, gamma*(alpha*v-w).");
  }
  prm.leave_subsection();

  prm.enter_subsection("Diagnostics");
  {
    prm.declare_entry("Diagnostic stride",
                      "20",
                      Patterns::Integer(1),
                      "Write diagnostics every N output checkpoints.");
    prm.declare_entry("Excitation threshold",
                      "0.5",
                      Patterns::Double(),
                      "Threshold on v used to define the excited set.");
    prm.declare_entry("Area epsilon",
                      "1e-14",
                      Patterns::Double(0.0),
                      "Small number to avoid division by zero in normalization.");
    prm.declare_entry("Diagnosis start time",
                      "0.5",
                      Patterns::Double(0.0),
                      "Ignore early transients and begin ignition checks only after this time.");
    prm.declare_entry("Ignition max v threshold",
                      "0.5",
                      Patterns::Double(),
                      "Absolute lower bound on max(v) required for an ignition hit.");
    prm.declare_entry("Ignition max v fraction of initial",
                      "0.25",
                      Patterns::Double(0.0),
                      "Relative max(v) threshold, measured as a fraction of the initial peak.");
    prm.declare_entry("Ignition excited fraction threshold",
                      "1e-3",
                      Patterns::Double(0.0),
                      "Minimum excited-domain fraction required for an ignition hit.");
    prm.declare_entry("Required ignition hits",
                      "2",
                      Patterns::Integer(1),
                      "Number of consecutive diagnostic hits required to classify ignition.");
  }
  prm.leave_subsection();
}

void ExperimentParameters::parse_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Mesh and Time");
  {
    domain_half_length = prm.get_double("Domain half length");
    n_refinements      = prm.get_integer("Global refinements");
    fe_degree          = prm.get_integer("Finite element degree");
    final_time         = prm.get_double("Final time");
    time_step          = prm.get_double("Time step");
    output_stride      = prm.get_integer("Output stride");
    mesh_adaptation_frequency =
      prm.get_integer("Mesh adaptation frequency");
    max_delta_refinement_level =
      prm.get_integer("Maximum delta refinement level");
    mesh_refinement_fraction = prm.get_double("Mesh refinement fraction");
    mesh_coarsening_fraction = prm.get_double("Mesh coarsening fraction");
  }
  prm.leave_subsection();

  prm.enter_subsection("Solution limiter");
  {
    enable_voltage_limiter = prm.get_bool("Enable voltage limiter");
    voltage_limiter_min    = prm.get_double("Minimum voltage");
    voltage_limiter_max    = prm.get_double("Maximum voltage");
  }
  prm.leave_subsection();

  prm.enter_subsection("Time integrator");
  {
    arkode_minimum_step_size =
      prm.get_double("ARKode minimum step size");
    arkode_maximum_order = prm.get_integer("ARKode maximum order");
    arkode_maximum_non_linear_iterations =
      prm.get_integer("ARKode maximum nonlinear iterations");
    arkode_absolute_tolerance =
      prm.get_double("ARKode absolute tolerance");
    arkode_relative_tolerance =
      prm.get_double("ARKode relative tolerance");
    arkode_implicit_function_is_linear =
      prm.get_bool("ARKode implicit function is linear");
    arkode_implicit_function_is_time_independent =
      prm.get_bool("ARKode implicit function is time independent");
  }
  prm.leave_subsection();

  initial_condition.parse_parameters(prm);

  prm.enter_subsection("Output");
  {
    output_directory = prm.get("Output directory");
    label_suffix     = prm.get("Label suffix");
  }
  prm.leave_subsection();

  prm.enter_subsection("Model");
  {
    epsilon     = prm.get_double("Epsilon");
    a_param     = prm.get_double("A parameter");
    b_param     = prm.get_double("B parameter");
    alpha_param = prm.get_double("Alpha parameter");
    gamma       = prm.get_double("Gamma parameter");
    diffusion_v   = prm.get_double("Diffusion v");
    diffusion_w   = prm.get_double("Diffusion w");
    beta_param    = prm.get_double("Beta parameter");
    kinetic_model = prm.get("Kinetic model");
  }
  prm.leave_subsection();

  prm.enter_subsection("Diagnostics");
  {
    diagnostic_stride    = prm.get_integer("Diagnostic stride");
    excitation_threshold = prm.get_double("Excitation threshold");
    epsilon_area         = prm.get_double("Area epsilon");
    diagnosis_start_time = prm.get_double("Diagnosis start time");
    ignition_max_v_threshold =
      prm.get_double("Ignition max v threshold");
    ignition_max_v_fraction_of_initial =
      prm.get_double("Ignition max v fraction of initial");
    ignition_excited_fraction_threshold =
      prm.get_double("Ignition excited fraction threshold");
    required_ignition_hits = prm.get_integer("Required ignition hits");
  }
  prm.leave_subsection();
}

std::string ExperimentParameters::format_double_token(const double value)
{
  std::ostringstream out;
  out << std::fixed << std::setprecision(6) << value;
  std::string token = out.str();

  while (!token.empty() && token.back() == '0')
    token.pop_back();
  if (!token.empty() && token.back() == '.')
    token.pop_back();
  if (token.empty())
    token = "0";

  std::replace(token.begin(), token.end(), '-', 'm');
  std::replace(token.begin(), token.end(), '.', 'p');
  return token;
}

std::string ExperimentParameters::sanitize_token(const std::string &value)
{
  std::string token;
  token.reserve(value.size());

  for (const char ch : value)
    if (std::isalnum(static_cast<unsigned char>(ch)))
      token.push_back(ch);
    else if (ch == '-' || ch == '_')
      token.push_back(ch);

  return token;
}

std::string ExperimentParameters::make_run_name() const
{
  std::ostringstream name;
  name << initial_condition.type
       << "_Av" << format_double_token(initial_condition.amplitude_v)
       << "_Aw" << format_double_token(initial_condition.amplitude_w);

  if (initial_condition.type == "disk")
    name << "_R" << format_double_token(initial_condition.radius)
         << "_SW" << format_double_token(initial_condition.smooth_width);
  else if (initial_condition.type == "gaussian")
    name << "_GW" << format_double_token(initial_condition.gaussian_width);
  else if (initial_condition.type == "broken_wave" ||
           initial_condition.type == "spiral")
    name << "_kx" << format_double_token(initial_condition.steepness_x)
         << "_ky" << format_double_token(initial_condition.steepness_y);

  name << "_Tf" << format_double_token(final_time)
       << "_TSARKode";

  const std::string suffix = sanitize_token(label_suffix);
  if (!suffix.empty())
    name << "_" << suffix;

  return name.str();
}

std::filesystem::path ExperimentParameters::make_output_directory() const
{
  return std::filesystem::path(output_directory) / make_run_name();
}

template <int dim>
class ParallelImplicitFHN {
public:
  explicit ParallelImplicitFHN(const ExperimentParameters &parameters);
  void run();

private:
  void make_grid();
  void setup_system();
  void assemble_matrices();
  void set_initial_conditions();
  void refine_mesh();
  void setup_arkode_jacobian(const double time,
                             const LA::MPI::Vector &y,
                             const LA::MPI::Vector &fy);
  void apply_arkode_jacobian(const LA::MPI::Vector &v,
                             LA::MPI::Vector       &Jv,
                             const double           time,
                             const LA::MPI::Vector &y,
                             const LA::MPI::Vector &fy);
  void solve_arkode_preconditioner(const double           time,
                                   const LA::MPI::Vector &y,
                                   const LA::MPI::Vector &fy,
                                   const LA::MPI::Vector &r,
                                   LA::MPI::Vector       &z,
                                   const double           gamma,
                                   const double           tol,
                                   const int              lr);
  void solve_arkode_linear_system(
    dealii::SUNDIALS::SundialsOperator<LA::MPI::Vector> &op,
    dealii::SUNDIALS::SundialsPreconditioner<LA::MPI::Vector> &prec,
    LA::MPI::Vector       &x,
    const LA::MPI::Vector &b,
    const double           tol);
  
  // ODE right-hand side: dy/dt = f(y) = diffusion + reaction
  void evaluate_rhs(const double time,
                    const LA::MPI::Vector& y_in,
                    LA::MPI::Vector& y_out);

  void assemble_jacobian_matrix(const LA::MPI::Vector& solution_state);
  void apply_solution_limiter(LA::MPI::Vector &state);
  
  void output_results(const double time,
                      const unsigned int timestep_number) const;

  DiagnosticsRecord compute_diagnostics() const;
  void initialize_diagnostics_file() const;
  void append_diagnostics(const double time,
                          const unsigned int timestep_number,
                          const DiagnosticsRecord &diag) const;
  std::filesystem::path diagnostics_file_path() const;
  void update_ignition_state(const double time, const DiagnosticsRecord &diag);
  double effective_ignition_max_v_threshold() const;
  std::filesystem::path ignition_summary_file_path() const;
  void write_ignition_summary(const double final_time,
                              const DiagnosticsRecord &final_diag) const;

  double initial_excited_area = -1.0;
  double initial_max_v = 0.0;
  unsigned int diagnostic_samples_after_start = 0;
  unsigned int consecutive_ignition_hits = 0;
  unsigned int max_consecutive_ignition_hits = 0;
  double max_v_after_start = -std::numeric_limits<double>::infinity();
  double max_excited_fraction_after_start = 0.0;

  ExperimentParameters prm;

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
  Vector<float> estimated_error_per_cell;
  
  // Output stream that only prints on processor 0
  ConditionalOStream pcout;
  
  // Timer for performance measurement
  TimerOutput computing_timer;
  
};

template <int dim>
ParallelImplicitFHN<dim>::ParallelImplicitFHN(const ExperimentParameters &parameters)
  : prm(parameters)
  , mpi_communicator(MPI_COMM_WORLD)
  , triangulation(mpi_communicator,
                  typename Triangulation<dim>::MeshSmoothing(
                    Triangulation<dim>::smoothing_on_refinement |
                    Triangulation<dim>::smoothing_on_coarsening))
  , fe(FE_Q<dim>(prm.fe_degree), 2)
  , dof_handler(triangulation)
  , pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
  , computing_timer(mpi_communicator, pcout, TimerOutput::summary, TimerOutput::wall_times)
{}

template <int dim>
std::filesystem::path
ParallelImplicitFHN<dim>::diagnostics_file_path() const
{
  return prm.make_output_directory() /
         (prm.make_run_name() + "-diagnostics.csv");
}

template <int dim>
std::filesystem::path
ParallelImplicitFHN<dim>::ignition_summary_file_path() const
{
  return prm.make_output_directory() /
         (prm.make_run_name() + "-ignition-summary.txt");
}

template <int dim>
double
ParallelImplicitFHN<dim>::effective_ignition_max_v_threshold() const
{
  return std::max(prm.ignition_max_v_threshold,
                  prm.ignition_max_v_fraction_of_initial * initial_max_v);
}

template <int dim>
void ParallelImplicitFHN<dim>::apply_solution_limiter(LA::MPI::Vector &state)
{
  if (!prm.enable_voltage_limiter)
    return;

  AssertThrow(prm.voltage_limiter_min <= prm.voltage_limiter_max,
              ExcMessage("Minimum voltage limiter must be <= maximum voltage limiter."));

  std::vector<types::global_dof_index> local_dof_indices(fe.dofs_per_cell);

  const auto clip_owned_voltage_dofs = [&]() {
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell->get_dof_indices(local_dof_indices);

          for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
            {
              const unsigned int component_i =
                fe.system_to_component_index(i).first;
              const types::global_dof_index dof_index = local_dof_indices[i];

              if (component_i == 0 && locally_owned_dofs.is_element(dof_index))
                {
                  const double value = state[dof_index];
                  // state[dof_index] =
                  //   std::clamp(value,
                  //              prm.voltage_limiter_min,
                  //              prm.voltage_limiter_max);
                  (void)value;
                }
            }
        }
  };

  clip_owned_voltage_dofs();
  state.compress(VectorOperation::insert);

  constraints.distribute(state);
  state.compress(VectorOperation::insert);

  // Hanging-node distribution can perturb constrained values.
  clip_owned_voltage_dofs();
  state.compress(VectorOperation::insert);
}

template <int dim>
void ParallelImplicitFHN<dim>::initialize_diagnostics_file() const
{
  if (Utilities::MPI::this_mpi_process(mpi_communicator) != 0)
    return;

  std::ofstream out(diagnostics_file_path().string().c_str(), std::ios::out);
  out << "time,step,max_v,excited_area,excited_fraction,relative_excited_area\n";
}

template <int dim>
DiagnosticsRecord
ParallelImplicitFHN<dim>::compute_diagnostics() const
{
  DiagnosticsRecord diag;

  const QGauss<dim> quadrature_formula(fe.degree + 1);
  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  std::vector<double>                  local_solution_values(dofs_per_cell);

  LA::MPI::Vector solution_with_ghosts(locally_owned_dofs,
                                       locally_relevant_dofs,
                                       mpi_communicator);
  solution_with_ghosts = solution;

  double local_max_v        = -std::numeric_limits<double>::infinity();
  double local_excited_area = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices);

        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          local_solution_values[i] = solution_with_ghosts[local_dof_indices[i]];

        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            double v_q = 0.0;

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                const unsigned int component_i =
                  fe.system_to_component_index(i).first;
                if (component_i == 0)
                  v_q += local_solution_values[i] * fe_values.shape_value(i, q);
              }

            local_max_v = std::max(local_max_v, v_q);

            if (v_q > prm.excitation_threshold)
              local_excited_area += fe_values.JxW(q);
          }
      }

  diag.max_v = Utilities::MPI::max(local_max_v, mpi_communicator);
  diag.excited_area =
    Utilities::MPI::sum(local_excited_area, mpi_communicator);

  const double domain_measure =
    std::pow(2.0 * prm.domain_half_length, static_cast<int>(dim));

  diag.excited_fraction = diag.excited_area / domain_measure;

  if (initial_excited_area > prm.epsilon_area)
    diag.relative_excited_area = diag.excited_area / initial_excited_area;
  else
    diag.relative_excited_area =
      (diag.excited_area <= prm.epsilon_area ?
         0.0 :
         std::numeric_limits<double>::quiet_NaN());

  return diag;
}

template <int dim>
void ParallelImplicitFHN<dim>::append_diagnostics(
  const double             time,
  const unsigned int       timestep_number,
  const DiagnosticsRecord &diag) const
{
  if (Utilities::MPI::this_mpi_process(mpi_communicator) != 0)
    return;

  std::ofstream out(diagnostics_file_path().string().c_str(), std::ios::app);
  out << std::setprecision(16)
      << time << ","
      << timestep_number << ","
      << diag.max_v << ","
      << diag.excited_area << ","
      << diag.excited_fraction << ","
      << diag.relative_excited_area << "\n";
}

template <int dim>
void ParallelImplicitFHN<dim>::update_ignition_state(const double             time,
                                                     const DiagnosticsRecord &diag)
{
  const double time_tolerance = 1e-14;
  if (time + time_tolerance < prm.diagnosis_start_time)
    return;

  ++diagnostic_samples_after_start;
  max_v_after_start =
    std::max(max_v_after_start, diag.max_v);
  max_excited_fraction_after_start =
    std::max(max_excited_fraction_after_start, diag.excited_fraction);

  const double effective_max_v_threshold =
    effective_ignition_max_v_threshold();
  const bool ignition_hit =
    (diag.max_v >= effective_max_v_threshold &&
     diag.excited_fraction >= prm.ignition_excited_fraction_threshold);

  if (ignition_hit)
    ++consecutive_ignition_hits;
  else
    consecutive_ignition_hits = 0;

  max_consecutive_ignition_hits =
    std::max(max_consecutive_ignition_hits, consecutive_ignition_hits);
}

template <int dim>
void ParallelImplicitFHN<dim>::write_ignition_summary(
  const double             final_time,
  const DiagnosticsRecord &final_diag) const
{
  if (Utilities::MPI::this_mpi_process(mpi_communicator) != 0)
    return;

  const bool diagnosis_window_reached =
    (diagnostic_samples_after_start > 0);
  const bool ignited =
    diagnosis_window_reached &&
    max_consecutive_ignition_hits >= prm.required_ignition_hits;
  const double effective_max_v_threshold =
    effective_ignition_max_v_threshold();
  const double reported_max_v_after_start =
    diagnosis_window_reached ? max_v_after_start
                             : std::numeric_limits<double>::quiet_NaN();
  const double reported_max_excited_fraction_after_start =
    diagnosis_window_reached ? max_excited_fraction_after_start
                             : std::numeric_limits<double>::quiet_NaN();

  std::ofstream out(ignition_summary_file_path().string().c_str(), std::ios::out);
  out << std::setprecision(16)
      << "diagnosis_window_reached = "
      << (diagnosis_window_reached ? "true" : "false") << "\n"
      << "ignited = " << (ignited ? "true" : "false") << "\n"
      << "diagnosis_start_time = " << prm.diagnosis_start_time << "\n"
      << "ignition_max_v_threshold = " << prm.ignition_max_v_threshold << "\n"
      << "ignition_max_v_fraction_of_initial = "
      << prm.ignition_max_v_fraction_of_initial << "\n"
      << "effective_ignition_max_v_threshold = "
      << effective_max_v_threshold << "\n"
      << "ignition_excited_fraction_threshold = "
      << prm.ignition_excited_fraction_threshold << "\n"
      << "required_ignition_hits = " << prm.required_ignition_hits << "\n"
      << "diagnostic_samples_after_start = " << diagnostic_samples_after_start
      << "\n"
      << "max_consecutive_ignition_hits = " << max_consecutive_ignition_hits
      << "\n"
      << "max_v_after_start = " << reported_max_v_after_start << "\n"
      << "max_excited_fraction_after_start = "
      << reported_max_excited_fraction_after_start << "\n"
      << "final_time = " << final_time << "\n"
      << "final_max_v = " << final_diag.max_v << "\n"
      << "final_excited_area = " << final_diag.excited_area << "\n"
      << "final_excited_fraction = " << final_diag.excited_fraction << "\n";
}

template <int dim>
void ParallelImplicitFHN<dim>::make_grid()
{
  TimerOutput::Scope t(computing_timer, "Make grid");

  GridGenerator::hyper_cube(triangulation,
                            -prm.domain_half_length,
                             prm.domain_half_length);
  triangulation.refine_global(prm.n_refinements);
}

template <int dim>
void ParallelImplicitFHN<dim>::setup_system()
{
  TimerOutput::Scope t(computing_timer, "Setup system");

  // Distribute DoFs in parallel; this gives each MPI rank
  // a contiguous block of global indices (required by PETSc).
  dof_handler.distribute_dofs(fe);

  // Get locally owned and relevant DoFs
  locally_owned_dofs = dof_handler.locally_owned_dofs();
  locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

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
  constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  // Homogeneous Neumann boundary conditions are natural -> no extra constraints
  constraints.make_consistent_in_parallel(locally_owned_dofs,
                                          locally_relevant_dofs,
                                          mpi_communicator);
  constraints.close();
  // locally_relevant_dofs = constraints.get_local_lines();

  estimated_error_per_cell.reinit(triangulation.n_active_cells(), true);

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
              double diffusion_coeff =
                (component_i == 0) ? prm.diffusion_v : prm.diffusion_w;
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

  const auto initial_condition =
    InitialConditions::make_initial_condition<dim>(prm.initial_condition);

  VectorTools::interpolate(dof_handler,
                           *initial_condition,
                           solution);

  constraints.distribute(solution);
  solution.compress(VectorOperation::insert);
  apply_solution_limiter(solution);
  locally_relevant_solution = solution;

  pcout << "Initial condition:\n"
        << "  fe degree   = " << prm.fe_degree << "\n"
        << "  type        = " << prm.initial_condition.type << "\n"
        << "  center      = (" << prm.initial_condition.center_x << ", "
        << prm.initial_condition.center_y << ")\n"
        << "  amplitude_v = " << prm.initial_condition.amplitude_v << "\n"
        << "  amplitude_w = " << prm.initial_condition.amplitude_w << "\n";

  if (prm.initial_condition.type == "disk")
    pcout << "  radius      = " << prm.initial_condition.radius << "\n"
          << "  smooth_width= " << prm.initial_condition.smooth_width << "\n";
  else if (prm.initial_condition.type == "gaussian")
    pcout << "  gauss_width = " << prm.initial_condition.gaussian_width
          << "\n";
  else if (prm.initial_condition.type == "broken_wave" ||
           prm.initial_condition.type == "spiral")
    pcout << "  steepness_x = " << prm.initial_condition.steepness_x << "\n"
          << "  steepness_y = " << prm.initial_condition.steepness_y << "\n";

  pcout
        << "  output dir  = " << prm.make_output_directory().string() << "\n"
        << "  v limiter   = "
        << (prm.enable_voltage_limiter ? "enabled" : "disabled");
  if (prm.enable_voltage_limiter)
    pcout << " [" << prm.voltage_limiter_min << ", "
          << prm.voltage_limiter_max << "]";
  pcout << std::endl;
}

template <int dim>
void ParallelImplicitFHN<dim>::refine_mesh()
{
  TimerOutput::Scope t(computing_timer, "Refine mesh");

  locally_relevant_solution = solution;
  estimated_error_per_cell.grow_or_shrink(triangulation.n_active_cells());

  ComponentMask voltage_mask(fe.n_components(), false);
  voltage_mask.set(0, true);

  KellyErrorEstimator<dim>::estimate(
    dof_handler,
    QGauss<dim - 1>(fe.degree + 1),
    {},
    locally_relevant_solution,
    estimated_error_per_cell,
    voltage_mask,
    nullptr,
    numbers::invalid_unsigned_int,
    numbers::invalid_subdomain_id,
    numbers::invalid_material_id,
    KellyErrorEstimator<dim>::Strategy::face_diameter_over_twice_max_degree);

  parallel::distributed::GridRefinement::refine_and_coarsen_fixed_fraction(
    triangulation,
    estimated_error_per_cell,
    prm.mesh_refinement_fraction,
    prm.mesh_coarsening_fraction);

  const unsigned int min_grid_level = prm.n_refinements;
  const unsigned int max_grid_level =
    prm.n_refinements + prm.max_delta_refinement_level;

  if (triangulation.n_levels() > max_grid_level)
    for (const auto &cell :
         triangulation.active_cell_iterators_on_level(max_grid_level))
      cell->clear_refine_flag();

  for (const auto &cell :
       triangulation.active_cell_iterators_on_level(min_grid_level))
    cell->clear_coarsen_flag();

  LA::MPI::Vector previous_solution(locally_owned_dofs,
                                    locally_relevant_dofs,
                                    mpi_communicator);
  previous_solution = solution;

  parallel::distributed::SolutionTransfer<dim, LA::MPI::Vector>
    solution_transfer(dof_handler);

  triangulation.prepare_coarsening_and_refinement();
  solution_transfer.prepare_for_coarsening_and_refinement(previous_solution);
  triangulation.execute_coarsening_and_refinement();

  setup_system();

  LA::MPI::Vector transferred_solution(locally_owned_dofs, mpi_communicator);
  solution_transfer.interpolate(transferred_solution);

  solution = transferred_solution;
  constraints.distribute(solution);
  solution.compress(VectorOperation::insert);
  apply_solution_limiter(solution);
  locally_relevant_solution = solution;

  assemble_matrices();
}

template <int dim>
void ParallelImplicitFHN<dim>::setup_arkode_jacobian(
  const double           /*time*/,
  const LA::MPI::Vector &y,
  const LA::MPI::Vector & /*fy*/)
{
  assemble_jacobian_matrix(y);
}

template <int dim>
void ParallelImplicitFHN<dim>::apply_arkode_jacobian(
  const LA::MPI::Vector &v,
  LA::MPI::Vector       &Jv,
  const double           /*time*/,
  const LA::MPI::Vector &y,
  const LA::MPI::Vector & /*fy*/)
{
  assemble_jacobian_matrix(y);

  LA::MPI::Vector tmp(locally_owned_dofs, mpi_communicator);
  LA::MPI::Vector reaction_jacobian_part(locally_owned_dofs, mpi_communicator);

  laplace_matrix.vmult(tmp, v);
  jacobian_matrix.vmult(reaction_jacobian_part, v);
  tmp += reaction_jacobian_part;

  Jv.reinit(locally_owned_dofs, mpi_communicator);

  SolverControl solver_control(1000, 1e-12);
  LA::SolverCG solver(solver_control);
  LA::MPI::PreconditionAMG preconditioner;
  LA::MPI::PreconditionAMG::AdditionalData data;
  data.symmetric_operator = true;
  preconditioner.initialize(mass_matrix, data);

  solver.solve(mass_matrix, Jv, tmp, preconditioner);
  constraints.distribute(Jv);
}

template <int dim>
void ParallelImplicitFHN<dim>::solve_arkode_preconditioner(
  const double           /*time*/,
  const LA::MPI::Vector &y,
  const LA::MPI::Vector & /*fy*/,
  const LA::MPI::Vector &r,
  LA::MPI::Vector       &z,
  const double           gamma,
  const double           /*tol*/,
  const int              /*lr*/)
{
  assemble_jacobian_matrix(y);

  system_matrix.copy_from(mass_matrix);
  system_matrix.add(-gamma, laplace_matrix);
  system_matrix.add(-gamma, jacobian_matrix);

  mass_matrix.vmult(system_rhs, r);

  SolverControl solver_control(2000, 1e-8);
  LA::SolverGMRES solver(solver_control);
  LA::MPI::PreconditionAMG preconditioner;
  LA::MPI::PreconditionAMG::AdditionalData data;
  data.symmetric_operator = false;
  preconditioner.initialize(system_matrix, data);

  z.reinit(locally_owned_dofs, mpi_communicator);
  solver.solve(system_matrix, z, system_rhs, preconditioner);
  constraints.distribute(z);
}

template <int dim>
void ParallelImplicitFHN<dim>::solve_arkode_linear_system(
  dealii::SUNDIALS::SundialsOperator<LA::MPI::Vector> & /*op*/,
  dealii::SUNDIALS::SundialsPreconditioner<LA::MPI::Vector> &prec,
  LA::MPI::Vector       &x,
  const LA::MPI::Vector &b,
  const double           /*tol*/)
{
  x.reinit(locally_owned_dofs, mpi_communicator);
  prec.vmult(x, b);
  constraints.distribute(x);
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
        double f_v = 0.0;
        double f_w = 0.0;
        if (prm.kinetic_model == "kinetic_II")
          {
            // Kinetic Model II: f_v = v(1-v)(v-beta)-w
            //                  f_w = gamma*(alpha*v - w)
            f_v = v_q * (1.0 - v_q) * (v_q - prm.beta_param) - w_q;
            f_w = prm.gamma * (prm.alpha_param * v_q - w_q);
          }
        else
          {
            // Kinetic Model I (default)
            const double V_th = (w_q + prm.b_param) / prm.a_param;
            f_v = (1.0 / prm.epsilon) * v_q * (1.0 - v_q) * (v_q - V_th);
            f_w = prm.alpha_param * v_q - prm.gamma * w_q;
          }

        
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
  LA::SolverCG solver(solver_control);
  LA::MPI::PreconditionAMG preconditioner;
  LA::MPI::PreconditionAMG::AdditionalData data;
  data.symmetric_operator = true;
  preconditioner.initialize(mass_matrix, data);
  
  solver.solve(mass_matrix, y_out, diffusion_part, preconditioner);
  constraints.distribute(y_out);
}

template <int dim>
void ParallelImplicitFHN<dim>::assemble_jacobian_matrix(const LA::MPI::Vector& solution_state) {
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

          double dfv_dv = 0.0;
          double dfv_dw = 0.0;
          double dfw_dv = 0.0;
          double dfw_dw = 0.0;
          if (prm.kinetic_model == "kinetic_II")
            {
              // df_v/dv = (1-2v)*(v-beta) + v*(1-v)
              // df_v/dw = -1
              // df_w/dv = gamma*alpha
              // df_w/dw = -gamma
              dfv_dv = (1.0 - 2.0 * v_q) * (v_q - prm.beta_param)
                       + v_q * (1.0 - v_q);
              dfv_dw = -1.0;
              dfw_dv = prm.gamma * prm.alpha_param;
              dfw_dw = -prm.gamma;
            }
          else
            {
              // Kinetic Model I (default)
              const double V_th = (w_q + prm.b_param) / prm.a_param;
              dfv_dv = ((1.0 - 2.0 * v_q) * (v_q - V_th)
                        + v_q - v_q * v_q) / prm.epsilon;
              dfv_dw = -v_q * (1.0 - v_q) / (prm.a_param * prm.epsilon);
              dfw_dv = prm.alpha_param;
              dfw_dw = -prm.gamma;
            }

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
void ParallelImplicitFHN<dim>::output_results(
  const double       time,
  const unsigned int timestep_number) const
{
  const std::filesystem::path output_dir = prm.make_output_directory();

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
  data_out.add_data_vector(estimated_error_per_cell, "error");
  
  data_out.build_patches();
  data_out.set_flags(DataOutBase::VtkFlags(time, timestep_number));
  
  const std::string filename =
    "solution-" +
    Utilities::int_to_string(timestep_number, 4) + ".vtu";
  data_out.write_vtu_in_parallel((output_dir / filename).string(),
                                 mpi_communicator);

  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
    static std::vector<std::pair<double, std::string>> times_and_names;
    times_and_names.emplace_back(time, filename);

    std::ofstream pvd_output((output_dir / "solution.pvd").string());
    DataOutBase::write_pvd_record(pvd_output, times_and_names);
  }
}

template <int dim>
void ParallelImplicitFHN<dim>::run()
{
  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    std::filesystem::create_directories(prm.make_output_directory());
  MPI_Barrier(mpi_communicator);

  make_grid();

  pcout << "Setting up parallel system..." << std::endl;
  setup_system();

  pcout << "Assembling matrices..." << std::endl;
  assemble_matrices();

  pcout << "Setting initial conditions..." << std::endl;
  set_initial_conditions();

  initialize_diagnostics_file();

  DiagnosticsRecord diag0 = compute_diagnostics();
  initial_excited_area = diag0.excited_area;
  initial_max_v = diag0.max_v;
  diag0.relative_excited_area =
    diag0.excited_area / std::max(initial_excited_area, prm.epsilon_area);
  append_diagnostics(0.0, 0, diag0);
  update_ignition_state(0.0, diag0);

  pcout << "Starting time integration with ARKode..." << std::endl;

  using ARKodeStepperType = tostii::ARKodeStepper<LA::MPI::Vector, double>;

  auto make_time_stepper = [this]() -> std::unique_ptr<ARKodeStepperType>
  {
    typename ARKodeStepperType::AdditionalData data(
      0.0,
      prm.final_time,
      prm.time_step,   // initial step size guess
      prm.time_step,   // initial output/advance seed
      prm.arkode_minimum_step_size,
      prm.arkode_maximum_order,
      prm.arkode_maximum_non_linear_iterations,
      prm.arkode_implicit_function_is_linear,
      prm.arkode_implicit_function_is_time_independent,
      false,
      3,
      prm.arkode_absolute_tolerance,
      prm.arkode_relative_tolerance);

    auto arkode_stepper =
      std::make_unique<ARKodeStepperType>(data, mpi_communicator);

    arkode_stepper->set_implicit_function(
      [this](const double t, const LA::MPI::Vector &y, LA::MPI::Vector &f)
      {
        this->evaluate_rhs(t, y, f);
      });

    arkode_stepper->set_jacobian_times_setup(
      [this](const double t,
             const LA::MPI::Vector &y,
             const LA::MPI::Vector &fy)
      {
        this->setup_arkode_jacobian(t, y, fy);
      });

    arkode_stepper->set_jacobian_times_vector(
      [this](const LA::MPI::Vector &v,
             LA::MPI::Vector &Jv,
             const double t,
             const LA::MPI::Vector &y,
             const LA::MPI::Vector &fy)
      {
        this->apply_arkode_jacobian(v, Jv, t, y, fy);
      });

    arkode_stepper->set_jacobian_preconditioner_solve(
      [this](const double t,
             const LA::MPI::Vector &y,
             const LA::MPI::Vector &fy,
             const LA::MPI::Vector &r,
             LA::MPI::Vector &z,
             const double gamma,
             const double tol,
             const int lr)
      {
        this->solve_arkode_preconditioner(t, y, fy, r, z, gamma, tol, lr);
      });

    arkode_stepper->set_solve_linearized_system(
      [this](dealii::SUNDIALS::SundialsOperator<LA::MPI::Vector> &op,
             dealii::SUNDIALS::SundialsPreconditioner<LA::MPI::Vector> &prec,
             LA::MPI::Vector &x,
             const LA::MPI::Vector &b,
             const double tol)
      {
        this->solve_arkode_linear_system(op, prec, x, b, tol);
      });

    return arkode_stepper;
  };

  auto arkode_stepper = make_time_stepper();

  double       time             = 0.0;
  unsigned int output_counter   = 0;
  unsigned int adapt_counter    = 0;
  const double output_interval  = prm.time_step;

  locally_relevant_solution = solution;
  output_results(time, output_counter);

  // Initialize once. After this, ARKode chooses its own internal steps.
  arkode_stepper->reset(time, prm.time_step, solution);

  double next_output_time = std::min(time + output_interval, prm.final_time);

  while (time < prm.final_time - 1e-14)
    {
      // Advance adaptively to the next requested output time.
      arkode_stepper->solve_ode_incrementally(solution, next_output_time);
      apply_solution_limiter(solution);
      time = next_output_time;

      if (prm.enable_voltage_limiter)
        arkode_stepper->reset(time, prm.time_step, solution);

      locally_relevant_solution = solution;
      ++output_counter;
      ++adapt_counter;

      if (prm.output_stride > 0 &&
          (output_counter % prm.output_stride == 0 ||
           time >= prm.final_time - 1e-14))
        output_results(time, output_counter);

      if (output_counter % prm.diagnostic_stride == 0 ||
          time >= prm.final_time - 1e-14)
        {
          const DiagnosticsRecord diag = compute_diagnostics();
          append_diagnostics(time, output_counter, diag);
          update_ignition_state(time, diag);
        }

      const bool do_adapt =
        (prm.mesh_adaptation_frequency > 0 &&
         adapt_counter % prm.mesh_adaptation_frequency == 0 &&
         time < prm.final_time - 1e-14);

      if (do_adapt)
        {
          pcout << "Adapting mesh at output step " << output_counter
                << " (t = " << time << ")..." << std::endl;

          // Destroy the old ARKode object before AMR changes vector layout.
          arkode_stepper.reset();

          refine_mesh();
          clear_sundials_petsc_vector_pool();

          // Rebuild ARKode on the new PETSc layout.
          arkode_stepper = make_time_stepper();
          arkode_stepper->reset(time, prm.time_step, solution);

          locally_relevant_solution = solution;
        }

      next_output_time = std::min(time + output_interval, prm.final_time);
    }

  const DiagnosticsRecord final_diag = compute_diagnostics();
  write_ignition_summary(time, final_diag);

  const bool diagnosis_window_reached =
    (diagnostic_samples_after_start > 0);
  const bool ignited =
    diagnosis_window_reached &&
    max_consecutive_ignition_hits >= prm.required_ignition_hits;

  if (diagnosis_window_reached)
    pcout << "Ignition diagnosis after t >= "
          << prm.diagnosis_start_time << ": "
          << (ignited ? "IGNITED" : "NOT IGNITED")
          << " (max consecutive hits = "
          << max_consecutive_ignition_hits << ", required = "
          << prm.required_ignition_hits << ")" << std::endl;
  else
    pcout << "Ignition diagnosis after t >= "
          << prm.diagnosis_start_time
          << " could not be performed because no diagnostic sample was "
             "recorded in that window."
          << std::endl;

  pcout << "Simulation completed successfully." << std::endl;
}

int main(int argc, char* argv[]) {
  try {
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    const std::string parameter_file =
      (argc > 1) ? argv[1] : "./parallel_implicit_fhn.prm";

    std::ifstream parameter_stream(parameter_file);
    AssertThrow(parameter_stream.is_open(),
                ExcMessage("Could not open parameter file: " + parameter_file));

    ParameterHandler prm;
    ExperimentParameters::declare_parameters(prm);
    prm.parse_input(parameter_file);

    ExperimentParameters params;
    params.parse_parameters(prm);

    ParallelImplicitFHN<2> fhn_solver(params);
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
