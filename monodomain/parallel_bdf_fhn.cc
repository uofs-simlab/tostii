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
 * ---------------------------------------------------------------------
 *
 * Author: Mohammad Mahdi Moayeri, University of Saskatchewan, 2021
 *
 * ---------------------------------------------------------------------
 * Parallel fully implicit FitzHugh-Nagumo solver using PETSc TS BDF
 * Uses MPI parallel computing and PETSc BDF time integration
 * Based on parallel_implicit_fhn.cc and deal.II step-86 / nagumo.cc
 * ---------------------------------------------------------------------
 */

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_ts.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/vector.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "InitialConditionFactory.h"

using namespace dealii;

namespace
{
  using VectorType = PETScWrappers::MPI::Vector;
  using MatrixType = PETScWrappers::MPI::SparseMatrix;
} // namespace

struct ExperimentParameters
{
  // Mesh and time
  double       domain_half_length           = 30.0;
  unsigned int n_refinements                = 6;
  unsigned int fe_degree                    = 2;
  double       final_time                   = 4.0;
  double       time_step                    = 5.0e-4;
  unsigned int output_stride                = 100;
  unsigned int mesh_adaptation_frequency    = 0;
  unsigned int max_delta_refinement_level   = 2;
  double       mesh_refinement_fraction     = 0.3;
  double       mesh_coarsening_fraction     = 0.03;

  // PETSc TS / BDF
  std::string petsc_ts_type           = "bdf";
  std::string petsc_ts_adapt_type     = "none";
  int         petsc_max_steps         = -1;
  bool        petsc_match_step        = false;
  bool        petsc_restart_if_remesh = false;
  double      petsc_minimum_step_size = -1.0;
  double      petsc_maximum_step_size = -1.0;
  double      petsc_absolute_tolerance = -1.0;
  double      petsc_relative_tolerance = -1.0;

  // Diagnostics
  unsigned int diagnostic_stride                  = 20;
  double       excitation_threshold               = 0.5;
  double       epsilon_area                       = 1e-14;
  double       diagnosis_start_time              = 0.5;
  double       ignition_max_v_threshold          = 0.5;
  double       ignition_max_v_fraction_of_initial = 0.25;
  double       ignition_excited_fraction_threshold = 1e-3;
  unsigned int required_ignition_hits            = 2;

  // Optional voltage limiter. The FHN kinetics is very sensitive to
  // numerical overshoots above v=1 when w is elevated.
  bool   enable_voltage_limiter = false;
  double voltage_limiter_min    = -0.2;
  double voltage_limiter_max    = 1.0;

  InitialConditions::Parameters initial_condition;

  // Kinetic model I parameters
  double epsilon     = 0.005;
  double a_param     = 0.3;
  double b_param     = 0.01;
  double alpha_param = 1.0;
  double gamma       = 1.0;
  double diffusion_v = 1.0;
  double diffusion_w = 0.0;
  // Kinetic model II parameter
  double      beta_param    = 0.1;
  std::string kinetic_model = "kinetic_I"; // "kinetic_I" or "kinetic_II"

  // Output configuration
  std::string output_directory = "outputs";
  std::string label_suffix     = "";

  static void declare_parameters(ParameterHandler &prm);
  void        parse_parameters(ParameterHandler &prm);
  std::string make_run_name() const;
  std::filesystem::path make_output_directory() const;

private:
  static std::string format_double_token(const double value);
  static std::string sanitize_token(const std::string &value);
};

struct DiagnosticsRecord
{
  double max_v                  = 0.0;
  double excited_area           = 0.0;
  double excited_fraction       = 0.0;
  double relative_excited_area  = 0.0;
};

struct LimiterStatistics
{
  double calls                   = 0.0;
  double voltage_dof_visits      = 0.0;
  double clipped_low             = 0.0;
  double clipped_high            = 0.0;
  double min_v_before            = std::numeric_limits<double>::infinity();
  double max_v_before            = -std::numeric_limits<double>::infinity();
  double max_low_correction      = 0.0;
  double max_high_correction     = 0.0;
  double total_abs_correction    = 0.0;

  void reset()
  {
    calls                = 0.0;
    voltage_dof_visits   = 0.0;
    clipped_low          = 0.0;
    clipped_high         = 0.0;
    min_v_before         = std::numeric_limits<double>::infinity();
    max_v_before         = -std::numeric_limits<double>::infinity();
    max_low_correction   = 0.0;
    max_high_correction  = 0.0;
    total_abs_correction = 0.0;
  }

  void accumulate(const LimiterStatistics &other)
  {
    calls += other.calls;
    voltage_dof_visits += other.voltage_dof_visits;
    clipped_low += other.clipped_low;
    clipped_high += other.clipped_high;
    min_v_before = std::min(min_v_before, other.min_v_before);
    max_v_before = std::max(max_v_before, other.max_v_before);
    max_low_correction =
      std::max(max_low_correction, other.max_low_correction);
    max_high_correction =
      std::max(max_high_correction, other.max_high_correction);
    total_abs_correction += other.total_abs_correction;
  }

  double clipped_fraction() const
  {
    return (voltage_dof_visits > 0.0 ?
              (clipped_low + clipped_high) / voltage_dof_visits :
              0.0);
  }
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
                      "Polynomial degree for each component.");
    prm.declare_entry("Final time",
                      "4.0",
                      Patterns::Double(0.0),
                      "Final simulation time.");
    prm.declare_entry("Time step",
                      "5e-4",
                      Patterns::Double(0.0),
                      "Initial time-step size.");
    prm.declare_entry("Output stride",
                      "100",
                      Patterns::Integer(1),
                      "Write output every N accepted time steps.");
    prm.declare_entry("Mesh adaptation frequency",
                      "0",
                      Patterns::Integer(0),
                      "Adapt the mesh every N accepted time steps. Use 0 to disable.");
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

  prm.enter_subsection("Time integrator");
  {
    prm.declare_entry("Type",
                      "bdf",
                      Patterns::Selection("bdf"),
                      "PETSc TS method to use.");
    prm.declare_entry("Adaptivity type",
                      "none",
                      Patterns::Anything(),
                      "PETSc TS adaptivity type. Use 'none' for fixed step.");
    prm.declare_entry("Maximum steps",
                      "-1",
                      Patterns::Integer(),
                      "Maximum number of TS steps. Negative uses PETSc defaults.");
    prm.declare_entry("Match step",
                      "false",
                      Patterns::Bool(),
                      "Whether PETSc should stop exactly at the final time.");
    prm.declare_entry("Restart if remesh",
                      "false",
                      Patterns::Bool(),
                      "Whether PETSc should restart the current step after remeshing.");
    prm.declare_entry("Minimum step size",
                      "-1.0",
                      Patterns::Double(),
                      "Minimum adaptive time step. Non-positive uses PETSc defaults.");
    prm.declare_entry("Maximum step size",
                      "-1.0",
                      Patterns::Double(),
                      "Maximum adaptive time step. Non-positive uses PETSc defaults.");
    prm.declare_entry("Absolute tolerance",
                      "-1.0",
                      Patterns::Double(),
                      "Absolute time-integration tolerance. Negative uses PETSc defaults.");
    prm.declare_entry("Relative tolerance",
                      "-1.0",
                      Patterns::Double(),
                      "Relative time-integration tolerance. Negative uses PETSc defaults.");
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
                      "Write diagnostics every N accepted time steps.");
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

  prm.enter_subsection("Solution limiter");
  {
    prm.declare_entry("Enable voltage limiter",
                      "false",
                      Patterns::Bool(),
                      "Clip voltage degrees of freedom after accepted stages.");
    prm.declare_entry("Minimum voltage",
                      "-0.2",
                      Patterns::Double(),
                      "Lower bound for the optional voltage limiter.");
    prm.declare_entry("Maximum voltage",
                      "1.0",
                      Patterns::Double(),
                      "Upper bound for the optional voltage limiter.");
  }
  prm.leave_subsection();
}

void ExperimentParameters::parse_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Mesh and Time");
  {
    domain_half_length        = prm.get_double("Domain half length");
    n_refinements             = prm.get_integer("Global refinements");
    fe_degree                 = prm.get_integer("Finite element degree");
    final_time                = prm.get_double("Final time");
    time_step                 = prm.get_double("Time step");
    output_stride             = prm.get_integer("Output stride");
    mesh_adaptation_frequency = prm.get_integer("Mesh adaptation frequency");
    max_delta_refinement_level =
      prm.get_integer("Maximum delta refinement level");
    mesh_refinement_fraction = prm.get_double("Mesh refinement fraction");
    mesh_coarsening_fraction = prm.get_double("Mesh coarsening fraction");
  }
  prm.leave_subsection();

  prm.enter_subsection("Time integrator");
  {
    petsc_ts_type            = prm.get("Type");
    petsc_ts_adapt_type      = prm.get("Adaptivity type");
    petsc_max_steps          = prm.get_integer("Maximum steps");
    petsc_match_step         = prm.get_bool("Match step");
    petsc_restart_if_remesh  = prm.get_bool("Restart if remesh");
    petsc_minimum_step_size  = prm.get_double("Minimum step size");
    petsc_maximum_step_size  = prm.get_double("Maximum step size");
    petsc_absolute_tolerance = prm.get_double("Absolute tolerance");
    petsc_relative_tolerance = prm.get_double("Relative tolerance");
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
    diagnostic_stride                   = prm.get_integer("Diagnostic stride");
    excitation_threshold                = prm.get_double("Excitation threshold");
    epsilon_area                        = prm.get_double("Area epsilon");
    diagnosis_start_time                = prm.get_double("Diagnosis start time");
    ignition_max_v_threshold            = prm.get_double("Ignition max v threshold");
    ignition_max_v_fraction_of_initial  =
      prm.get_double("Ignition max v fraction of initial");
    ignition_excited_fraction_threshold =
      prm.get_double("Ignition excited fraction threshold");
    required_ignition_hits = prm.get_integer("Required ignition hits");
  }
  prm.leave_subsection();

  prm.enter_subsection("Solution limiter");
  {
    enable_voltage_limiter = prm.get_bool("Enable voltage limiter");
    voltage_limiter_min    = prm.get_double("Minimum voltage");
    voltage_limiter_max    = prm.get_double("Maximum voltage");
  }
  prm.leave_subsection();
}

std::string ExperimentParameters::format_double_token(const double value)
{
  const double cleaned_value = (std::abs(value) < 1e-14 ? 0.0 : value);

  std::ostringstream out;
  out << std::defaultfloat << std::setprecision(4) << cleaned_value;
  std::string token = out.str();
  std::replace(token.begin(), token.end(), '-', 'm');
  std::replace(token.begin(), token.end(), '.', 'p');
  token.erase(std::remove(token.begin(), token.end(), '+'), token.end());
  return token;
}

std::string ExperimentParameters::sanitize_token(const std::string &value)
{
  std::string token;
  token.reserve(value.size());
  for (const char c : value)
    if (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-')
      token.push_back(c);
    else
      token.push_back('_');
  return token;
}

std::string ExperimentParameters::make_run_name() const
{
  std::ostringstream name;
  name << "fhn"
       << "_L" << format_double_token(domain_half_length)
       << "_N" << n_refinements
       << "_p" << fe_degree
       << "_e" << format_double_token(epsilon)
       << "_a" << format_double_token(a_param)
       << "_b" << format_double_token(b_param)
       << "_al" << format_double_token(alpha_param)
       << "_g" << format_double_token(gamma)
       << "_Dv" << format_double_token(diffusion_v)
       << "_Dw" << format_double_token(diffusion_w)
       << "_IC" << sanitize_token(initial_condition.type)
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

  name << "_T" << format_double_token(final_time)
       << "_BDF";

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
class ParallelBdfFHN
{
public:
  explicit ParallelBdfFHN(const ExperimentParameters &parameters);
  void run();

private:
  void make_grid();
  void setup_system();
  void assemble_matrices();
  void set_initial_conditions();

  void assemble_reaction_rhs(const VectorType &state,
                             VectorType       &reaction_rhs) const;
  void prepare_for_coarsening_and_refinement(const VectorType &state);
  void transfer_solution_vectors_to_new_mesh(
    const double                    time,
    const std::vector<VectorType>  &all_in,
    std::vector<VectorType>        &all_out);

  void implicit_function(const double      time,
                         const VectorType &y,
                         const VectorType &y_dot,
                         VectorType       &residual);
  void assemble_implicit_jacobian(const double      time,
                                  const VectorType &y,
                                  const VectorType &y_dot,
                                  const double      alpha);
  void solve_with_jacobian(const VectorType &src, VectorType &dst);
  void apply_solution_limiter(VectorType &state);

  void output_results(const double time, const unsigned int timestep_number) const;

  DiagnosticsRecord compute_diagnostics() const;
  void initialize_diagnostics_file() const;
  void append_diagnostics(const double             time,
                          const unsigned int       timestep_number,
                          const DiagnosticsRecord &diag);
  std::filesystem::path diagnostics_file_path() const;
  void update_ignition_state(const double             time,
                             const DiagnosticsRecord &diag);
  double effective_ignition_max_v_threshold() const;
  std::filesystem::path ignition_summary_file_path() const;
  void write_ignition_summary(const double             final_time,
                              const DiagnosticsRecord &final_diag) const;
  LimiterStatistics collect_global_limiter_statistics(
    const LimiterStatistics &local_statistics) const;

  double       initial_excited_area             = -1.0;
  double       initial_max_v                    = 0.0;
  unsigned int diagnostic_samples_after_start   = 0;
  unsigned int consecutive_ignition_hits        = 0;
  unsigned int max_consecutive_ignition_hits    = 0;
  double       max_v_after_start                =
    -std::numeric_limits<double>::infinity();
  double       max_excited_fraction_after_start = 0.0;
  double       next_remesh_time                 =
    std::numeric_limits<double>::infinity();
  unsigned int last_output_step                 = numbers::invalid_unsigned_int;
  unsigned int last_diagnostic_step             = numbers::invalid_unsigned_int;
  LimiterStatistics limiter_interval_statistics;
  LimiterStatistics limiter_total_statistics;

  ExperimentParameters prm;

  MPI_Comm mpi_communicator;

  parallel::distributed::Triangulation<dim> triangulation;
  FESystem<dim>                             fe;
  DoFHandler<dim>                           dof_handler;

  IndexSet                  locally_owned_dofs;
  IndexSet                  locally_relevant_dofs;
  AffineConstraints<double> constraints;

  MatrixType mass_matrix;
  MatrixType laplace_matrix;
  MatrixType reaction_jacobian_matrix;
  MatrixType jacobian_matrix;

  VectorType solution;
  VectorType locally_relevant_solution;
  Vector<float> estimated_error_per_cell;

  ConditionalOStream pcout;
  TimerOutput        computing_timer;
};

template <int dim>
ParallelBdfFHN<dim>::ParallelBdfFHN(const ExperimentParameters &parameters)
  : prm(parameters)
  , mpi_communicator(MPI_COMM_WORLD)
  , triangulation(mpi_communicator,
                  typename Triangulation<dim>::MeshSmoothing(
                    Triangulation<dim>::smoothing_on_refinement |
                    Triangulation<dim>::smoothing_on_coarsening))
  , fe(FE_Q<dim>(parameters.fe_degree), 2)
  , dof_handler(triangulation)
  , pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
  , computing_timer(mpi_communicator,
                    pcout,
                    TimerOutput::summary,
                    TimerOutput::wall_times)
{}

template <int dim>
std::filesystem::path ParallelBdfFHN<dim>::diagnostics_file_path() const
{
  return prm.make_output_directory() / "diagnostics.csv";
}

template <int dim>
std::filesystem::path ParallelBdfFHN<dim>::ignition_summary_file_path() const
{
  return prm.make_output_directory() / "ignition-summary.txt";
}

template <int dim>
double ParallelBdfFHN<dim>::effective_ignition_max_v_threshold() const
{
  return std::max(prm.ignition_max_v_threshold,
                  prm.ignition_max_v_fraction_of_initial * initial_max_v);
}

template <int dim>
LimiterStatistics ParallelBdfFHN<dim>::collect_global_limiter_statistics(
  const LimiterStatistics &local_statistics) const
{
  LimiterStatistics global_statistics;

  global_statistics.calls =
    Utilities::MPI::sum(local_statistics.calls, mpi_communicator);
  global_statistics.voltage_dof_visits =
    Utilities::MPI::sum(local_statistics.voltage_dof_visits,
                        mpi_communicator);
  global_statistics.clipped_low =
    Utilities::MPI::sum(local_statistics.clipped_low, mpi_communicator);
  global_statistics.clipped_high =
    Utilities::MPI::sum(local_statistics.clipped_high, mpi_communicator);

  const double local_min =
    (local_statistics.voltage_dof_visits > 0.0 ?
       local_statistics.min_v_before :
       std::numeric_limits<double>::infinity());
  const double local_max =
    (local_statistics.voltage_dof_visits > 0.0 ?
       local_statistics.max_v_before :
       -std::numeric_limits<double>::infinity());

  global_statistics.min_v_before =
    -Utilities::MPI::max(-local_min, mpi_communicator);
  global_statistics.max_v_before =
    Utilities::MPI::max(local_max, mpi_communicator);
  global_statistics.max_low_correction =
    Utilities::MPI::max(local_statistics.max_low_correction,
                        mpi_communicator);
  global_statistics.max_high_correction =
    Utilities::MPI::max(local_statistics.max_high_correction,
                        mpi_communicator);
  global_statistics.total_abs_correction =
    Utilities::MPI::sum(local_statistics.total_abs_correction,
                        mpi_communicator);

  if (global_statistics.voltage_dof_visits == 0.0)
    {
      global_statistics.min_v_before =
        std::numeric_limits<double>::quiet_NaN();
      global_statistics.max_v_before =
        std::numeric_limits<double>::quiet_NaN();
    }

  return global_statistics;
}

template <int dim>
void ParallelBdfFHN<dim>::apply_solution_limiter(VectorType &state)
{
  if (!prm.enable_voltage_limiter)
    return;

  AssertThrow(prm.voltage_limiter_min <= prm.voltage_limiter_max,
              ExcMessage("Minimum voltage limiter must be <= maximum voltage limiter."));

  std::vector<types::global_dof_index> local_dof_indices(fe.dofs_per_cell);
  LimiterStatistics                    call_statistics;
  call_statistics.calls = 1.0;

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
                  const double clipped_value =
                    std::clamp(value,
                               prm.voltage_limiter_min,
                               prm.voltage_limiter_max);

                  ++call_statistics.voltage_dof_visits;
                  call_statistics.min_v_before =
                    std::min(call_statistics.min_v_before, value);
                  call_statistics.max_v_before =
                    std::max(call_statistics.max_v_before, value);

                  if (clipped_value > value)
                    {
                      const double correction = clipped_value - value;
                      ++call_statistics.clipped_low;
                      call_statistics.max_low_correction =
                        std::max(call_statistics.max_low_correction,
                                 correction);
                      call_statistics.total_abs_correction += correction;
                    }
                  else if (clipped_value < value)
                    {
                      const double correction = value - clipped_value;
                      ++call_statistics.clipped_high;
                      call_statistics.max_high_correction =
                        std::max(call_statistics.max_high_correction,
                                 correction);
                      call_statistics.total_abs_correction += correction;
                    }

                  state[dof_index] = clipped_value;
                }
            }
        }
  };

  clip_owned_voltage_dofs();
  state.compress(VectorOperation::insert);

  constraints.distribute(state);
  state.compress(VectorOperation::insert);

  // Hanging-node distribution can modify constrained values; clip once more.
  clip_owned_voltage_dofs();
  state.compress(VectorOperation::insert);

  limiter_interval_statistics.accumulate(call_statistics);
  limiter_total_statistics.accumulate(call_statistics);
}

template <int dim>
void ParallelBdfFHN<dim>::initialize_diagnostics_file() const
{
  if (Utilities::MPI::this_mpi_process(mpi_communicator) != 0)
    return;

  std::ofstream out(diagnostics_file_path().string().c_str(), std::ios::out);
  out << "time,step,max_v,excited_area,excited_fraction,"
         "relative_excited_area,limiter_calls,limiter_voltage_dof_visits,"
         "limiter_clipped_low,limiter_clipped_high,"
         "limiter_clipped_fraction,limiter_min_v_before,"
         "limiter_max_v_before,limiter_max_low_correction,"
         "limiter_max_high_correction,limiter_total_abs_correction\n";
}

template <int dim>
DiagnosticsRecord ParallelBdfFHN<dim>::compute_diagnostics() const
{
  DiagnosticsRecord diag;

  const QGauss<dim> quadrature_formula(fe.degree + 1);
  FEValues<dim>     fe_values(fe, quadrature_formula, update_values | update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  std::vector<double>                  local_solution_values(dofs_per_cell);

  VectorType solution_with_ghosts(locally_owned_dofs,
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
              if (fe.system_to_component_index(i).first == 0)
                v_q += local_solution_values[i] * fe_values.shape_value(i, q);

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
void ParallelBdfFHN<dim>::append_diagnostics(const double             time,
                                             const unsigned int       timestep_number,
                                             const DiagnosticsRecord &diag)
{
  const LimiterStatistics limiter_statistics =
    collect_global_limiter_statistics(limiter_interval_statistics);

  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
      std::ofstream out(diagnostics_file_path().string().c_str(), std::ios::app);
      out << std::setprecision(16)
          << time << ","
          << timestep_number << ","
          << diag.max_v << ","
          << diag.excited_area << ","
          << diag.excited_fraction << ","
          << diag.relative_excited_area << ","
          << limiter_statistics.calls << ","
          << limiter_statistics.voltage_dof_visits << ","
          << limiter_statistics.clipped_low << ","
          << limiter_statistics.clipped_high << ","
          << limiter_statistics.clipped_fraction() << ","
          << limiter_statistics.min_v_before << ","
          << limiter_statistics.max_v_before << ","
          << limiter_statistics.max_low_correction << ","
          << limiter_statistics.max_high_correction << ","
          << limiter_statistics.total_abs_correction << "\n";
    }

  limiter_interval_statistics.reset();
}

template <int dim>
void ParallelBdfFHN<dim>::update_ignition_state(const double             time,
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

  const double effective_max_v_threshold = effective_ignition_max_v_threshold();
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
void ParallelBdfFHN<dim>::write_ignition_summary(const double             final_time,
                                                 const DiagnosticsRecord &final_diag) const
{
  const LimiterStatistics limiter_statistics =
    collect_global_limiter_statistics(limiter_total_statistics);

  if (Utilities::MPI::this_mpi_process(mpi_communicator) != 0)
    return;

  const bool diagnosis_window_reached = (diagnostic_samples_after_start > 0);
  const bool ignited =
    diagnosis_window_reached &&
    max_consecutive_ignition_hits >= prm.required_ignition_hits;

  const double effective_max_v_threshold = effective_ignition_max_v_threshold();
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
      << "final_excited_fraction = " << final_diag.excited_fraction << "\n"
      << "voltage_limiter_enabled = "
      << (prm.enable_voltage_limiter ? "true" : "false") << "\n"
      << "voltage_limiter_min = " << prm.voltage_limiter_min << "\n"
      << "voltage_limiter_max = " << prm.voltage_limiter_max << "\n"
      << "limiter_total_calls = " << limiter_statistics.calls << "\n"
      << "limiter_total_voltage_dof_visits = "
      << limiter_statistics.voltage_dof_visits << "\n"
      << "limiter_total_clipped_low = " << limiter_statistics.clipped_low
      << "\n"
      << "limiter_total_clipped_high = " << limiter_statistics.clipped_high
      << "\n"
      << "limiter_total_clipped_fraction = "
      << limiter_statistics.clipped_fraction() << "\n"
      << "limiter_total_min_v_before = " << limiter_statistics.min_v_before
      << "\n"
      << "limiter_total_max_v_before = " << limiter_statistics.max_v_before
      << "\n"
      << "limiter_total_max_low_correction = "
      << limiter_statistics.max_low_correction << "\n"
      << "limiter_total_max_high_correction = "
      << limiter_statistics.max_high_correction << "\n"
      << "limiter_total_abs_correction = "
      << limiter_statistics.total_abs_correction << "\n";
}

template <int dim>
void ParallelBdfFHN<dim>::make_grid()
{
  TimerOutput::Scope t(computing_timer, "make grid");

  GridGenerator::hyper_cube(triangulation,
                            -prm.domain_half_length,
                            prm.domain_half_length);
  triangulation.refine_global(prm.n_refinements);
}

template <int dim>
void ParallelBdfFHN<dim>::setup_system()
{
  TimerOutput::Scope t(computing_timer, "setup system");

  dof_handler.distribute_dofs(fe);

  locally_owned_dofs    = dof_handler.locally_owned_dofs();
  locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

  Assert(locally_owned_dofs.is_contiguous(),
         ExcMessage("PETSc requires contiguous locally owned DoFs."));

  const std::vector<types::global_dof_index> dofs_per_component =
    DoFTools::count_dofs_per_fe_component(dof_handler);

  pcout << "Number of active cells: "
        << triangulation.n_global_active_cells() << std::endl;
  pcout << "Number of degrees of freedom: " << dof_handler.n_dofs()
        << " (" << dofs_per_component[0] << " + " << dofs_per_component[1]
        << ")" << std::endl;

  constraints.clear();
  constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  constraints.make_consistent_in_parallel(locally_owned_dofs,
                                          locally_relevant_dofs,
                                          mpi_communicator);
  constraints.close();

  estimated_error_per_cell.reinit(triangulation.n_active_cells(), true);

  DynamicSparsityPattern dsp(locally_relevant_dofs);
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
  SparsityTools::distribute_sparsity_pattern(dsp,
                                             locally_owned_dofs,
                                             mpi_communicator,
                                             locally_relevant_dofs);

  mass_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);
  laplace_matrix.reinit(locally_owned_dofs,
                        locally_owned_dofs,
                        dsp,
                        mpi_communicator);
  reaction_jacobian_matrix.reinit(locally_owned_dofs,
                                  locally_owned_dofs,
                                  dsp,
                                  mpi_communicator);
  jacobian_matrix.reinit(locally_owned_dofs,
                         locally_owned_dofs,
                         dsp,
                         mpi_communicator);

  solution.reinit(locally_owned_dofs, mpi_communicator);
  locally_relevant_solution.reinit(locally_owned_dofs,
                                   locally_relevant_dofs,
                                   mpi_communicator);
}

template <int dim>
void ParallelBdfFHN<dim>::assemble_matrices()
{
  TimerOutput::Scope t(computing_timer, "assemble matrices");

  mass_matrix    = 0;
  laplace_matrix = 0;

  const QGauss<dim> quadrature_formula(fe.degree + 1);
  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients | update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();

  FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_laplace_matrix(dofs_per_cell, dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        fe_values.reinit(cell);
        cell_mass_matrix    = 0;
        cell_laplace_matrix = 0;

        for (unsigned int q = 0; q < n_q_points; ++q)
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              const unsigned int component_i = fe.system_to_component_index(i).first;
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  const unsigned int component_j =
                    fe.system_to_component_index(j).first;

                  if (component_i == component_j)
                    {
                      cell_mass_matrix(i, j) +=
                        fe_values.shape_value(i, q) *
                        fe_values.shape_value(j, q) * fe_values.JxW(q);

                      const double diffusion_coeff =
                        (component_i == 0) ? prm.diffusion_v : prm.diffusion_w;
                      cell_laplace_matrix(i, j) -=
                        diffusion_coeff * fe_values.shape_grad(i, q) *
                        fe_values.shape_grad(j, q) * fe_values.JxW(q);
                    }
                }
            }

        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(cell_mass_matrix,
                                               local_dof_indices,
                                               mass_matrix);
        constraints.distribute_local_to_global(cell_laplace_matrix,
                                               local_dof_indices,
                                               laplace_matrix);
      }

  mass_matrix.compress(VectorOperation::add);
  laplace_matrix.compress(VectorOperation::add);
}

template <int dim>
void ParallelBdfFHN<dim>::set_initial_conditions()
{
  TimerOutput::Scope t(computing_timer, "set initial conditions");

  const auto initial_condition =
    InitialConditions::make_initial_condition<dim>(prm.initial_condition);

  VectorTools::interpolate(dof_handler, *initial_condition, solution);

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
    pcout << "  gauss_width = " << prm.initial_condition.gaussian_width << "\n";
  else if (prm.initial_condition.type == "broken_wave" ||
           prm.initial_condition.type == "spiral")
    pcout << "  steepness_x = " << prm.initial_condition.steepness_x << "\n"
          << "  steepness_y = " << prm.initial_condition.steepness_y << "\n";

  pcout << "  output dir  = " << prm.make_output_directory().string()
        << "\n"
        << "  v limiter   = "
        << (prm.enable_voltage_limiter ? "enabled" : "disabled");

  if (prm.enable_voltage_limiter)
    pcout << " [" << prm.voltage_limiter_min << ", "
          << prm.voltage_limiter_max << "]";

  pcout
        << std::endl;
}

template <int dim>
void ParallelBdfFHN<dim>::assemble_reaction_rhs(const VectorType &state,
                                                VectorType       &reaction_rhs) const
{
  reaction_rhs.reinit(locally_owned_dofs, mpi_communicator);
  reaction_rhs = 0;

  const QGauss<dim> quadrature_formula(fe.degree + 1);
  FEValues<dim> fe_values(fe, quadrature_formula, update_values | update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();

  Vector<double>                        cell_reaction_rhs(dofs_per_cell);
  std::vector<types::global_dof_index>  local_dof_indices(dofs_per_cell);
  std::vector<double>                   local_solution_values(dofs_per_cell);

  VectorType state_with_ghosts(locally_owned_dofs,
                               locally_relevant_dofs,
                               mpi_communicator);
  state_with_ghosts = state;

  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        fe_values.reinit(cell);
        cell_reaction_rhs = 0;

        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          local_solution_values[i] = state_with_ghosts[local_dof_indices[i]];

        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            double v_q = 0.0;
            double w_q = 0.0;
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                const unsigned int component_i =
                  fe.system_to_component_index(i).first;
                if (component_i == 0)
                  v_q += local_solution_values[i] * fe_values.shape_value(i, q);
                else
                  w_q += local_solution_values[i] * fe_values.shape_value(i, q);
              }

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
                const double v_th = (w_q + prm.b_param) / prm.a_param;
                f_v = (1.0 / prm.epsilon) * v_q * (1.0 - v_q) * (v_q - v_th);
                f_w = prm.alpha_param * v_q - prm.gamma * w_q;
              }

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                const unsigned int component_i =
                  fe.system_to_component_index(i).first;
                const double reaction_value = (component_i == 0) ? f_v : f_w;
                cell_reaction_rhs[i] +=
                  reaction_value * fe_values.shape_value(i, q) * fe_values.JxW(q);
              }
          }

        constraints.distribute_local_to_global(cell_reaction_rhs,
                                               local_dof_indices,
                                               reaction_rhs);
      }

  reaction_rhs.compress(VectorOperation::add);
}

template <int dim>
void ParallelBdfFHN<dim>::prepare_for_coarsening_and_refinement(const VectorType &state)
{
  VectorType locally_relevant_state(locally_owned_dofs,
                                    locally_relevant_dofs,
                                    mpi_communicator);
  locally_relevant_state = state;

  estimated_error_per_cell.grow_or_shrink(triangulation.n_active_cells());

  ComponentMask voltage_mask(fe.n_components(), false);
  voltage_mask.set(0, true);

  KellyErrorEstimator<dim>::estimate(
    dof_handler,
    QGauss<dim - 1>(fe.degree + 1),
    {},
    locally_relevant_state,
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
}

template <int dim>
void ParallelBdfFHN<dim>::transfer_solution_vectors_to_new_mesh(
  const double                    time,
  const std::vector<VectorType>  &all_in,
  std::vector<VectorType>        &all_out)
{
  parallel::distributed::SolutionTransfer<dim, VectorType> solution_transfer(
    dof_handler);

  std::vector<VectorType>        all_in_ghosted(all_in.size());
  std::vector<const VectorType *> all_in_ghosted_ptr(all_in.size());
  std::vector<VectorType *>       all_out_ptr(all_in.size());

  for (unsigned int i = 0; i < all_in.size(); ++i)
    {
      all_in_ghosted[i].reinit(locally_owned_dofs,
                               locally_relevant_dofs,
                               mpi_communicator);
      all_in_ghosted[i] = all_in[i];
      all_in_ghosted_ptr[i] = &all_in_ghosted[i];
    }

  triangulation.prepare_coarsening_and_refinement();
  solution_transfer.prepare_for_coarsening_and_refinement(all_in_ghosted_ptr);
  triangulation.execute_coarsening_and_refinement();

  setup_system();
  assemble_matrices();
  estimated_error_per_cell.reinit(triangulation.n_active_cells(), false);

  all_out.resize(all_in.size());
  for (unsigned int i = 0; i < all_in.size(); ++i)
    {
      all_out[i].reinit(locally_owned_dofs, mpi_communicator);
      all_out_ptr[i] = &all_out[i];
    }

  solution_transfer.interpolate(all_out_ptr);

  for (VectorType &v : all_out)
    constraints.distribute(v);

  if (!all_out.empty())
    {
      apply_solution_limiter(all_out[0]);
      solution = all_out[0];
      solution.compress(VectorOperation::insert);
      locally_relevant_solution = solution;
    }

  (void)time;
}

template <int dim>
void ParallelBdfFHN<dim>::implicit_function(const double      /*time*/,
                                            const VectorType &y,
                                            const VectorType &y_dot,
                                            VectorType       &residual)
{
  TimerOutput::Scope t(computing_timer, "implicit function");

  VectorType tmp_solution(locally_owned_dofs, mpi_communicator);
  VectorType tmp_solution_dot(locally_owned_dofs, mpi_communicator);
  tmp_solution     = y;
  tmp_solution_dot = y_dot;

  constraints.distribute(tmp_solution);
  constraints.distribute(tmp_solution_dot);

  VectorType reaction_rhs(locally_owned_dofs, mpi_communicator);
  VectorType laplace_part(locally_owned_dofs, mpi_communicator);
  VectorType mass_part(locally_owned_dofs, mpi_communicator);

  assemble_reaction_rhs(tmp_solution, reaction_rhs);
  laplace_matrix.vmult(laplace_part, tmp_solution);
  mass_matrix.vmult(mass_part, tmp_solution_dot);

  residual = 0;
  residual += mass_part;
  residual -= laplace_part;
  residual -= reaction_rhs;
  residual.compress(VectorOperation::insert);

  for (const auto &c : constraints.get_lines())
    if (locally_owned_dofs.is_element(c.index))
      residual[c.index] = y[c.index];

  residual.compress(VectorOperation::insert);
}

template <int dim>
void ParallelBdfFHN<dim>::assemble_implicit_jacobian(const double      /*time*/,
                                                     const VectorType &y,
                                                     const VectorType & /*y_dot*/,
                                                     const double      alpha)
{
  TimerOutput::Scope t(computing_timer, "assemble implicit Jacobian");

  reaction_jacobian_matrix = 0;

  const QGauss<dim> quadrature_formula(fe.degree + 1);
  FEValues<dim> fe_values(fe, quadrature_formula, update_values | update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();

  FullMatrix<double>                 cell_matrix(dofs_per_cell, dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  std::vector<double>                  local_solution_values(dofs_per_cell);

  VectorType tmp_solution(locally_owned_dofs, mpi_communicator);
  tmp_solution = y;
  constraints.distribute(tmp_solution);

  VectorType solution_with_ghosts(locally_owned_dofs,
                                  locally_relevant_dofs,
                                  mpi_communicator);
  solution_with_ghosts = tmp_solution;

  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        fe_values.reinit(cell);
        cell_matrix = 0;

        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          local_solution_values[i] = solution_with_ghosts[local_dof_indices[i]];

        for (unsigned int q = 0; q < n_q_points; ++q)
          {
            double v_q = 0.0;
            double w_q = 0.0;
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
                const double v_th = (w_q + prm.b_param) / prm.a_param;
                dfv_dv = ((1.0 - 2.0 * v_q) * (v_q - v_th)
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

                    cell_matrix(i, j) +=
                      jacobian_entry * fe_values.shape_value(i, q) *
                      fe_values.shape_value(j, q) * fe_values.JxW(q);
                  }
              }
          }

        constraints.distribute_local_to_global(cell_matrix,
                                               local_dof_indices,
                                               reaction_jacobian_matrix);
      }

  reaction_jacobian_matrix.compress(VectorOperation::add);

  jacobian_matrix.copy_from(mass_matrix);
  jacobian_matrix *= alpha;
  jacobian_matrix.add(-1.0, laplace_matrix);
  jacobian_matrix.add(-1.0, reaction_jacobian_matrix);

  for (const auto &c : constraints.get_lines())
    if (locally_owned_dofs.is_element(c.index))
      jacobian_matrix.set(c.index, c.index, 1.0);

  jacobian_matrix.compress(VectorOperation::insert);
}

template <int dim>
void ParallelBdfFHN<dim>::solve_with_jacobian(const VectorType &src, VectorType &dst)
{
  TimerOutput::Scope t(computing_timer, "solve with Jacobian");

  SolverControl                solver_control(2000, 1e-8 * src.l2_norm());
  PETScWrappers::SolverGMRES   solver(solver_control);
  solver.set_prefix("user_");

#if defined(PETSC_HAVE_HYPRE)
  PETScWrappers::PreconditionBoomerAMG preconditioner;
  preconditioner.initialize(jacobian_matrix);
#else
  PETScWrappers::PreconditionBlockJacobi preconditioner;
  preconditioner.initialize(jacobian_matrix);
#endif

  dst = 0;
  solver.solve(jacobian_matrix, dst, src, preconditioner);
  constraints.distribute(dst);
}

template <int dim>
void ParallelBdfFHN<dim>::output_results(const double       time,
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

  data_out.add_data_vector(locally_relevant_solution,
                           component_names,
                           DataOut<dim>::type_dof_data,
                           component_interpretation);
  data_out.add_data_vector(estimated_error_per_cell, "error");

  data_out.build_patches();
  data_out.set_flags(DataOutBase::VtkFlags(time, timestep_number));

  const std::string filename =
    "solution-" + Utilities::int_to_string(timestep_number, 4) + ".vtu";
  data_out.write_vtu_in_parallel((output_dir / filename).string(),
                                 mpi_communicator);

  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
      static std::vector<std::pair<double, std::string>> times_and_names;
      times_and_names.emplace_back(time, filename);

      std::ofstream pvd_output((output_dir / "solution.pvd").string());
      DataOutBase::write_pvd_record(pvd_output, times_and_names);
    }
}

template <int dim>
void ParallelBdfFHN<dim>::run()
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
  initial_max_v        = diag0.max_v;
  diag0.relative_excited_area =
    diag0.excited_area / std::max(initial_excited_area, prm.epsilon_area);
  append_diagnostics(0.0, 0, diag0);
  update_ignition_state(0.0, diag0);

  output_results(0.0, 0);
  last_output_step     = 0;
  last_diagnostic_step = 0;
  next_remesh_time =
    (prm.mesh_adaptation_frequency > 0
       ? prm.mesh_adaptation_frequency * prm.time_step
       : std::numeric_limits<double>::infinity());

  pcout << "Starting PETSc TS integration with type '" << prm.petsc_ts_type
        << "'..." << std::endl;

  PETScWrappers::TimeStepperData time_stepper_data(
    "",
    prm.petsc_ts_type,
    0.0,
    prm.final_time,
    prm.time_step,
    prm.petsc_max_steps,
    prm.petsc_match_step,
    prm.petsc_restart_if_remesh,
    prm.petsc_ts_adapt_type,
    prm.petsc_minimum_step_size,
    prm.petsc_maximum_step_size,
    prm.petsc_absolute_tolerance,
    prm.petsc_relative_tolerance,
    true);

  PETScWrappers::TimeStepper<VectorType, MatrixType> petsc_ts(time_stepper_data);
  petsc_ts.set_matrices(jacobian_matrix, jacobian_matrix);

  petsc_ts.implicit_function =
    [&](const double      time,
        const VectorType &y,
        const VectorType &y_dot,
        VectorType       &res)
  {
    this->implicit_function(time, y, y_dot, res);
  };

  petsc_ts.setup_jacobian =
    [&](const double      time,
        const VectorType &y,
        const VectorType &y_dot,
        const double      alpha)
  {
    this->assemble_implicit_jacobian(time, y, y_dot, alpha);
  };

  petsc_ts.solve_with_jacobian =
    [&](const VectorType &src, VectorType &dst)
  {
    this->solve_with_jacobian(src, dst);
  };

  petsc_ts.algebraic_components = [&]() {
    IndexSet algebraic_set(dof_handler.n_dofs());
    algebraic_set.add_indices(DoFTools::extract_hanging_node_dofs(dof_handler));
    return algebraic_set;
  };

  petsc_ts.update_constrained_components =
    [&](const double /*time*/, VectorType &y)
  {
    constraints.distribute(y);
    this->apply_solution_limiter(y);
  };

  petsc_ts.decide_and_prepare_for_remeshing =
    [&](const double      time,
        const unsigned int step_number,
        const VectorType  &y) -> bool
  {
    if (prm.mesh_adaptation_frequency == 0)
      return false;

    const double tol = 1e-12 * std::max(1.0, std::abs(time));

    if (step_number > 0 && time + tol >= next_remesh_time &&
        time < prm.final_time - tol)
      {
        pcout << "Adapting mesh at time step " << step_number
              << " (t = " << time << ")..." 
              << std::endl;
        this->prepare_for_coarsening_and_refinement(y);

        const double remesh_interval =
          prm.mesh_adaptation_frequency * prm.time_step;
        while (next_remesh_time <= time + tol)
          next_remesh_time += remesh_interval;

        return true;
      }

    return false;
  };

  petsc_ts.transfer_solution_vectors_to_new_mesh =
    [&](const double                   time,
        const std::vector<VectorType> &all_in,
        std::vector<VectorType>       &all_out)
  {
    this->transfer_solution_vectors_to_new_mesh(time, all_in, all_out);
  };

  petsc_ts.monitor =
    [&](const double      time,
        const VectorType &y,
        const unsigned int step_number)
  {
    solution = y;
    apply_solution_limiter(solution);
    locally_relevant_solution = solution;

    if (prm.output_stride > 0 &&
        step_number % prm.output_stride == 0 &&
        step_number != last_output_step)
      {
        output_results(time, step_number);
        last_output_step = step_number;
      }

    if (step_number % prm.diagnostic_stride == 0 &&
        step_number != last_diagnostic_step)
      {
        const DiagnosticsRecord diag = compute_diagnostics();
        append_diagnostics(time, step_number, diag);
        update_ignition_state(time, diag);
        last_diagnostic_step = step_number;
      }
  };

  petsc_ts.solve(solution);
  apply_solution_limiter(solution);
  locally_relevant_solution = solution;

  const double       final_time = petsc_ts.get_time();
  const unsigned int final_step = petsc_ts.get_step_number();

  if (final_step != last_output_step)
    output_results(final_time, final_step);

  const DiagnosticsRecord final_diag = compute_diagnostics();
  if (final_step != last_diagnostic_step)
    {
      append_diagnostics(final_time, final_step, final_diag);
      update_ignition_state(final_time, final_diag);
    }

  write_ignition_summary(final_time, final_diag);

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

int main(int argc, char *argv[])
{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      const std::string parameter_file =
        (argc > 1) ? argv[1] : "./parallel_bdf_fhn.prm";

      std::ifstream parameter_stream(parameter_file);
      AssertThrow(parameter_stream.is_open(),
                  ExcMessage("Could not open parameter file: " + parameter_file));

      ParameterHandler prm;
      ExperimentParameters::declare_parameters(prm);
      prm.parse_input(parameter_file);

      ExperimentParameters params;
      params.parse_parameters(prm);

      ParallelBdfFHN<2> fhn_solver(params);
      fhn_solver.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
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
