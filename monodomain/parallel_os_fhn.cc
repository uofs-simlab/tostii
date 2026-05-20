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
 * Author: Mohammad Mahdi Moayeri, University of Saskatchewan, 2025
 *
 * ---------------------------------------------------------------------
 * Parallel Operator-Split FitzHugh-Nagumo solver using tostii
 *
 * Uses operator splitting (default: Strang, 2nd order) to decouple
 * diffusion and reaction:
 *   - Diffusion: solved implicitly via ImplicitRungeKutta (default: Crank-Nicolson)
 *   - Reaction:  solved explicitly via ExplicitRungeKutta (default: RK4)
 *
 * Strang stages:  D(dt/2) -> R(dt) -> D(dt/2)
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

namespace LA
{
  using namespace dealii::LinearAlgebraPETSc;
}

#include <deal.II/lac/vector.h>
#include <deal.II/lac/vector_memory.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
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

#include <tostii/tostii.h>

#include "InitialConditionFactory.h"

using namespace dealii;

#include <algorithm>
#include <string>
#include <memory>

// ============================================================
//  Parameters
// ============================================================

struct ExperimentParameters
{
  // Mesh and time
  double       domain_half_length = 30.0;
  unsigned int n_refinements      = 6;
  unsigned int fe_degree          = 2;
  double       final_time         = 4.0;
  double       time_step          = 5.0e-4;
  unsigned int output_stride      = 100;
  unsigned int mesh_adaptation_frequency   = 0;
  unsigned int max_delta_refinement_level  = 2;
  double       mesh_refinement_fraction    = 0.3;
  double       mesh_coarsening_fraction    = 0.03;

  // Operator-splitting time integrator
  std::string os_method         = "Strang";           // Godunov | Strang | Ruth | Yoshida
  std::string diffusion_method  = "CRANK_NICOLSON";   // ImplicitRK enum name
  std::string reaction_method   = "RK_CLASSIC_FOURTH_ORDER"; // ExplicitRK enum name

  // Diagnostics
  unsigned int diagnostic_stride                 = 20;
  double       excitation_threshold              = 0.5;
  double       epsilon_area                      = 1e-14;
  double       diagnosis_start_time              = 0.5;
  double       ignition_max_v_threshold          = 0.5;
  double       ignition_max_v_fraction_of_initial = 0.25;
  double       ignition_excited_fraction_threshold = 1e-3;
  unsigned int required_ignition_hits            = 2;

  // Optional voltage limiter
  bool   enable_voltage_limiter = false;
  double voltage_limiter_min    = -0.2;
  double voltage_limiter_max    =  1.05;

  InitialConditions::Parameters initial_condition;

  // Kinetic model
  double      epsilon      = 0.005;
  double      a_param      = 0.3;
  double      b_param      = 0.01;
  double      alpha_param  = 1.0;
  double      gamma        = 1.0;
  double      diffusion_v  = 1.0;
  double      diffusion_w  = 0.0;
  double      beta_param   = 0.1;
  std::string kinetic_model = "kinetic_I"; // kinetic_I | kinetic_II

  // Output
  std::string output_directory = "outputs";
  std::string label_suffix     = "";

  static void declare_parameters(ParameterHandler &prm);
  void        parse_parameters(ParameterHandler &prm);
  std::string make_run_name() const;
  std::filesystem::path make_output_directory() const;

private:
  static std::string format_double_token(double value);
  static std::string sanitize_token(const std::string &value);
};

struct DiagnosticsRecord
{
  double max_v              = 0.0;
  double excited_area       = 0.0;
  double excited_fraction   = 0.0;
  double relative_excited_area = 0.0;
};

// ---- declare / parse -------------------------------------------------------

void ExperimentParameters::declare_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Mesh and Time");
  {
    prm.declare_entry("Domain half length",   "30.0",  Patterns::Double(0.0), "");
    prm.declare_entry("Global refinements",   "6",     Patterns::Integer(0),  "");
    prm.declare_entry("Finite element degree","2",     Patterns::Integer(1),  "");
    prm.declare_entry("Final time",           "4.0",   Patterns::Double(0.0), "");
    prm.declare_entry("Time step",            "5e-4",  Patterns::Double(0.0), "");
    prm.declare_entry("Output stride",        "100",   Patterns::Integer(1),  "");
    prm.declare_entry("Mesh adaptation frequency",    "0",   Patterns::Integer(0),       "");
    prm.declare_entry("Maximum delta refinement level","2",  Patterns::Integer(0),       "");
    prm.declare_entry("Mesh refinement fraction",     "0.3", Patterns::Double(0.0,1.0),  "");
    prm.declare_entry("Mesh coarsening fraction",     "0.03",Patterns::Double(0.0,1.0),  "");
  }
  prm.leave_subsection();

  prm.enter_subsection("Time integrator");
  {
    prm.declare_entry("OS method",
                      "Strang",
                      Patterns::Selection("Godunov|Strang|Ruth|Yoshida|BM4_6"),
                      "Operator-splitting scheme.");
    prm.declare_entry("Diffusion method",
                      "CRANK_NICOLSON",
                      Patterns::Selection("BACKWARD_EULER|IMPLICIT_MIDPOINT|"
                                          "CRANK_NICOLSON|SDIRK_TWO_STAGES|"
                                          "SDIRK_THREE_STAGES|SDIRK_3O4|SDIRK_5O4"),
                      "Implicit RK scheme for the diffusion sub-step.");
    prm.declare_entry("Reaction method",
                      "RK_CLASSIC_FOURTH_ORDER",
                      Patterns::Selection("FORWARD_EULER|EXPLICIT_MIDPOINT|HEUN2|"
                                          "RK_THIRD_ORDER|SSP_THIRD_ORDER|"
                                          "RK_CLASSIC_FOURTH_ORDER"),
                      "Explicit RK scheme for the reaction sub-step.");
  }
  prm.leave_subsection();

  prm.enter_subsection("Solution limiter");
  {
    prm.declare_entry("Enable voltage limiter", "false", Patterns::Bool(), "");
    prm.declare_entry("Minimum voltage",        "-0.2",  Patterns::Double(), "");
    prm.declare_entry("Maximum voltage",        "1.05",  Patterns::Double(), "");
  }
  prm.leave_subsection();

  InitialConditions::Parameters::declare_parameters(prm);

  prm.enter_subsection("Output");
  {
    prm.declare_entry("Output directory", "outputs",  Patterns::Anything(), "");
    prm.declare_entry("Label suffix",     "",         Patterns::Anything(), "");
  }
  prm.leave_subsection();

  prm.enter_subsection("Model");
  {
    prm.declare_entry("Epsilon",       "0.005", Patterns::Double(0.0), "");
    prm.declare_entry("A parameter",   "0.3",   Patterns::Double(),    "");
    prm.declare_entry("B parameter",   "0.01",  Patterns::Double(),    "");
    prm.declare_entry("Alpha parameter","1.0",  Patterns::Double(),    "");
    prm.declare_entry("Gamma parameter","1.0",  Patterns::Double(),    "");
    prm.declare_entry("Diffusion v",   "1.0",   Patterns::Double(),    "");
    prm.declare_entry("Diffusion w",   "0.0",   Patterns::Double(),    "");
    prm.declare_entry("Beta parameter","0.1",   Patterns::Double(),    "");
    prm.declare_entry("Kinetic model", "kinetic_I",
                      Patterns::Selection("kinetic_I|kinetic_II"), "");
  }
  prm.leave_subsection();

  prm.enter_subsection("Diagnostics");
  {
    prm.declare_entry("Diagnostic stride",                "20",   Patterns::Integer(1),  "");
    prm.declare_entry("Excitation threshold",             "0.5",  Patterns::Double(),    "");
    prm.declare_entry("Area epsilon",                     "1e-14",Patterns::Double(0.0), "");
    prm.declare_entry("Diagnosis start time",             "0.5",  Patterns::Double(0.0), "");
    prm.declare_entry("Ignition max v threshold",         "0.5",  Patterns::Double(),    "");
    prm.declare_entry("Ignition max v fraction of initial","0.25",Patterns::Double(0.0), "");
    prm.declare_entry("Ignition excited fraction threshold","1e-3",Patterns::Double(0.0),"");
    prm.declare_entry("Required ignition hits",           "2",    Patterns::Integer(1),  "");
  }
  prm.leave_subsection();
}

void ExperimentParameters::parse_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Mesh and Time");
  {
    domain_half_length           = prm.get_double("Domain half length");
    n_refinements                = prm.get_integer("Global refinements");
    fe_degree                    = prm.get_integer("Finite element degree");
    final_time                   = prm.get_double("Final time");
    time_step                    = prm.get_double("Time step");
    output_stride                = prm.get_integer("Output stride");
    mesh_adaptation_frequency    = prm.get_integer("Mesh adaptation frequency");
    max_delta_refinement_level   = prm.get_integer("Maximum delta refinement level");
    mesh_refinement_fraction     = prm.get_double("Mesh refinement fraction");
    mesh_coarsening_fraction     = prm.get_double("Mesh coarsening fraction");
  }
  prm.leave_subsection();

  prm.enter_subsection("Time integrator");
  {
    os_method        = prm.get("OS method");
    diffusion_method = prm.get("Diffusion method");
    reaction_method  = prm.get("Reaction method");
  }
  prm.leave_subsection();

  prm.enter_subsection("Solution limiter");
  {
    enable_voltage_limiter = prm.get_bool("Enable voltage limiter");
    voltage_limiter_min    = prm.get_double("Minimum voltage");
    voltage_limiter_max    = prm.get_double("Maximum voltage");
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
    epsilon       = prm.get_double("Epsilon");
    a_param       = prm.get_double("A parameter");
    b_param       = prm.get_double("B parameter");
    alpha_param   = prm.get_double("Alpha parameter");
    gamma         = prm.get_double("Gamma parameter");
    diffusion_v   = prm.get_double("Diffusion v");
    diffusion_w   = prm.get_double("Diffusion w");
    beta_param    = prm.get_double("Beta parameter");
    kinetic_model = prm.get("Kinetic model");
  }
  prm.leave_subsection();

  prm.enter_subsection("Diagnostics");
  {
    diagnostic_stride                  = prm.get_integer("Diagnostic stride");
    excitation_threshold               = prm.get_double("Excitation threshold");
    epsilon_area                       = prm.get_double("Area epsilon");
    diagnosis_start_time               = prm.get_double("Diagnosis start time");
    ignition_max_v_threshold           = prm.get_double("Ignition max v threshold");
    ignition_max_v_fraction_of_initial = prm.get_double("Ignition max v fraction of initial");
    ignition_excited_fraction_threshold = prm.get_double("Ignition excited fraction threshold");
    required_ignition_hits             = prm.get_integer("Required ignition hits");
  }
  prm.leave_subsection();
}

std::string ExperimentParameters::format_double_token(const double value)
{
  std::ostringstream out;
  out << std::fixed << std::setprecision(6) << value;
  std::string token = out.str();
  while (!token.empty() && token.back() == '0') token.pop_back();
  if (!token.empty() && token.back() == '.') token.pop_back();
  if (token.empty()) token = "0";
  std::replace(token.begin(), token.end(), '-', 'm');
  std::replace(token.begin(), token.end(), '.', 'p');
  return token;
}

std::string ExperimentParameters::sanitize_token(const std::string &value)
{
  std::string token;
  token.reserve(value.size());
  for (const char ch : value)
    if (std::isalnum(static_cast<unsigned char>(ch)) || ch == '-' || ch == '_')
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
    name << "_R"  << format_double_token(initial_condition.radius)
         << "_SW" << format_double_token(initial_condition.smooth_width);
  else if (initial_condition.type == "gaussian")
    name << "_GW" << format_double_token(initial_condition.gaussian_width);
  else if (initial_condition.type == "broken_wave" || initial_condition.type == "spiral")
    name << "_kx" << format_double_token(initial_condition.steepness_x)
         << "_ky" << format_double_token(initial_condition.steepness_y);

  name << "_Tf" << format_double_token(final_time)
       << "_TS" << sanitize_token(os_method);

  const std::string suffix = sanitize_token(label_suffix);
  if (!suffix.empty())
    name << "_" << suffix;

  return name.str();
}

std::filesystem::path ExperimentParameters::make_output_directory() const
{
  return std::filesystem::path(output_directory) / make_run_name();
}

// ============================================================
//  Solver class
// ============================================================

template <int dim>
class ParallelOSFHN
{
public:
  explicit ParallelOSFHN(const ExperimentParameters &parameters);
  void run();

private:
  void make_grid();
  void setup_system();
  void assemble_matrices();
  void set_initial_conditions();
  void refine_mesh();

  // Diffusion sub-step  (operator 0, implicit)
  void solve_diffusion_rhs(const double           time,
                           const LA::MPI::Vector &y,
                           LA::MPI::Vector       &f_out);

  // id_minus_tau_J_inverse for diffusion: solves (M - tau*L)*out = M*y
  void solve_diffusion_lhs(const double           time,
                           const double           tau,
                           const LA::MPI::Vector &y,
                           LA::MPI::Vector       &out);

  // Reaction sub-step (operator 1, explicit)
  void solve_reaction_rhs(const double           time,
                          const LA::MPI::Vector &y,
                          LA::MPI::Vector       &f_out);

  void apply_solution_limiter(LA::MPI::Vector &state);

  void output_results(double time, unsigned int timestep_number) const;

  DiagnosticsRecord compute_diagnostics() const;
  void initialize_diagnostics_file() const;
  void append_diagnostics(double time,
                          unsigned int timestep_number,
                          const DiagnosticsRecord &diag) const;
  std::filesystem::path diagnostics_file_path() const;
  void update_ignition_state(double time, const DiagnosticsRecord &diag);
  double effective_ignition_max_v_threshold() const;
  std::filesystem::path ignition_summary_file_path() const;
  void write_ignition_summary(double final_time,
                              const DiagnosticsRecord &final_diag) const;

  // Ignition tracking state
  double       initial_excited_area  = -1.0;
  double       initial_max_v         = 0.0;
  unsigned int diagnostic_samples_after_start     = 0;
  unsigned int consecutive_ignition_hits           = 0;
  unsigned int max_consecutive_ignition_hits       = 0;
  double       max_v_after_start                   = -std::numeric_limits<double>::infinity();
  double       max_excited_fraction_after_start    = 0.0;

  ExperimentParameters prm;

  MPI_Comm mpi_communicator;

  parallel::distributed::Triangulation<dim> triangulation;
  FESystem<dim>  fe;
  DoFHandler<dim> dof_handler;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  AffineConstraints<double> constraints;

  LA::MPI::SparseMatrix mass_matrix;
  LA::MPI::SparseMatrix laplace_matrix;
  LA::MPI::SparseMatrix system_matrix;  // scratch: used in diffusion LHS

  LA::MPI::PreconditionAMG mass_amg;   // cached; rebuilt only after remesh

  LA::MPI::Vector solution;
  LA::MPI::Vector locally_relevant_solution;
  LA::MPI::Vector system_rhs;
  Vector<float>   estimated_error_per_cell;

  ConditionalOStream pcout;
  TimerOutput        computing_timer;
};

// ---- Constructor -----------------------------------------------------------

template <int dim>
ParallelOSFHN<dim>::ParallelOSFHN(const ExperimentParameters &parameters)
  : prm(parameters)
  , mpi_communicator(MPI_COMM_WORLD)
  , triangulation(mpi_communicator,
                  typename Triangulation<dim>::MeshSmoothing(
                    Triangulation<dim>::smoothing_on_refinement |
                    Triangulation<dim>::smoothing_on_coarsening))
  , fe(FE_Q<dim>(prm.fe_degree), 2)
  , dof_handler(triangulation)
  , pcout(std::cout,
          (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
  , computing_timer(mpi_communicator, pcout,
                    TimerOutput::summary, TimerOutput::wall_times)
{}

// ---- Path helpers ----------------------------------------------------------

template <int dim>
std::filesystem::path ParallelOSFHN<dim>::diagnostics_file_path() const
{
  return prm.make_output_directory() / (prm.make_run_name() + "-diagnostics.csv");
}

template <int dim>
std::filesystem::path ParallelOSFHN<dim>::ignition_summary_file_path() const
{
  return prm.make_output_directory() / (prm.make_run_name() + "-ignition-summary.txt");
}

template <int dim>
double ParallelOSFHN<dim>::effective_ignition_max_v_threshold() const
{
  return std::max(prm.ignition_max_v_threshold,
                  prm.ignition_max_v_fraction_of_initial * initial_max_v);
}

// ---- Voltage limiter -------------------------------------------------------

template <int dim>
void ParallelOSFHN<dim>::apply_solution_limiter(LA::MPI::Vector &state)
{
  if (!prm.enable_voltage_limiter)
    return;

  AssertThrow(prm.voltage_limiter_min <= prm.voltage_limiter_max,
              ExcMessage("Minimum voltage limiter must be <= maximum."));

  std::vector<types::global_dof_index> local_dof_indices(fe.dofs_per_cell);

  const auto clip_owned_voltage_dofs = [&]() {
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        {
          cell->get_dof_indices(local_dof_indices);
          for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
            {
              const unsigned int comp = fe.system_to_component_index(i).first;
              const types::global_dof_index idx = local_dof_indices[i];
              if (comp == 0 && locally_owned_dofs.is_element(idx))
                {
                  const double val = static_cast<double>(state[idx]);
                  state[idx] = std::clamp(val,
                                          prm.voltage_limiter_min,
                                          prm.voltage_limiter_max);
                }
            }
        }
  };

  clip_owned_voltage_dofs();
  state.compress(VectorOperation::insert);
  constraints.distribute(state);
  state.compress(VectorOperation::insert);
  clip_owned_voltage_dofs();
  state.compress(VectorOperation::insert);
}

// ---- Diagnostics -----------------------------------------------------------

template <int dim>
void ParallelOSFHN<dim>::initialize_diagnostics_file() const
{
  if (Utilities::MPI::this_mpi_process(mpi_communicator) != 0)
    return;
  std::ofstream out(diagnostics_file_path().string(), std::ios::out);
  out << "time,step,max_v,excited_area,excited_fraction,relative_excited_area\n";
}

template <int dim>
DiagnosticsRecord ParallelOSFHN<dim>::compute_diagnostics() const
{
  DiagnosticsRecord diag;

  const QGauss<dim> qf(fe.degree + 1);
  FEValues<dim> fev(fe, qf, update_values | update_JxW_values);

  const unsigned int dpc = fe.dofs_per_cell;
  const unsigned int nq  = qf.size();

  std::vector<types::global_dof_index> local_dof_indices(dpc);

  LA::MPI::Vector sol_ghost(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
  sol_ghost = solution;

  double local_max_v        = -std::numeric_limits<double>::infinity();
  double local_excited_area = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        fev.reinit(cell);
        cell->get_dof_indices(local_dof_indices);

        for (unsigned int q = 0; q < nq; ++q)
          {
            double v_q = 0.0;
            for (unsigned int i = 0; i < dpc; ++i)
              if (fe.system_to_component_index(i).first == 0)
                v_q += sol_ghost[local_dof_indices[i]] * fev.shape_value(i, q);

            local_max_v = std::max(local_max_v, v_q);
            if (v_q > prm.excitation_threshold)
              local_excited_area += fev.JxW(q);
          }
      }

  diag.max_v        = Utilities::MPI::max(local_max_v, mpi_communicator);
  diag.excited_area = Utilities::MPI::sum(local_excited_area, mpi_communicator);

  const double domain_measure =
    std::pow(2.0 * prm.domain_half_length, static_cast<int>(dim));
  diag.excited_fraction = diag.excited_area / domain_measure;

  if (initial_excited_area > prm.epsilon_area)
    diag.relative_excited_area = diag.excited_area / initial_excited_area;
  else
    diag.relative_excited_area =
      (diag.excited_area <= prm.epsilon_area)
        ? 0.0
        : std::numeric_limits<double>::quiet_NaN();

  return diag;
}

template <int dim>
void ParallelOSFHN<dim>::append_diagnostics(
  const double             time,
  const unsigned int       step,
  const DiagnosticsRecord &diag) const
{
  if (Utilities::MPI::this_mpi_process(mpi_communicator) != 0)
    return;
  std::ofstream out(diagnostics_file_path().string(), std::ios::app);
  out << std::setprecision(16)
      << time << "," << step << ","
      << diag.max_v << "," << diag.excited_area << ","
      << diag.excited_fraction << "," << diag.relative_excited_area << "\n";
}

template <int dim>
void ParallelOSFHN<dim>::update_ignition_state(const double             time,
                                                   const DiagnosticsRecord &diag)
{
  if (time + 1e-14 < prm.diagnosis_start_time)
    return;

  ++diagnostic_samples_after_start;
  max_v_after_start             = std::max(max_v_after_start, diag.max_v);
  max_excited_fraction_after_start = std::max(max_excited_fraction_after_start,
                                              diag.excited_fraction);

  const bool hit = (diag.max_v >= effective_ignition_max_v_threshold() &&
                    diag.excited_fraction >= prm.ignition_excited_fraction_threshold);
  if (hit)
    ++consecutive_ignition_hits;
  else
    consecutive_ignition_hits = 0;
  max_consecutive_ignition_hits = std::max(max_consecutive_ignition_hits,
                                           consecutive_ignition_hits);
}

template <int dim>
void ParallelOSFHN<dim>::write_ignition_summary(
  const double             final_time,
  const DiagnosticsRecord &final_diag) const
{
  if (Utilities::MPI::this_mpi_process(mpi_communicator) != 0)
    return;

  const bool window_reached = (diagnostic_samples_after_start > 0);
  const bool ignited        = window_reached &&
    max_consecutive_ignition_hits >= prm.required_ignition_hits;

  std::ofstream out(ignition_summary_file_path().string(), std::ios::out);
  out << std::setprecision(16)
      << "diagnosis_window_reached = " << (window_reached ? "true" : "false") << "\n"
      << "ignited = " << (ignited ? "true" : "false") << "\n"
      << "diagnosis_start_time = " << prm.diagnosis_start_time << "\n"
      << "effective_ignition_max_v_threshold = " << effective_ignition_max_v_threshold() << "\n"
      << "required_ignition_hits = " << prm.required_ignition_hits << "\n"
      << "max_consecutive_ignition_hits = " << max_consecutive_ignition_hits << "\n"
      << "max_v_after_start = "
      << (window_reached ? max_v_after_start : std::numeric_limits<double>::quiet_NaN()) << "\n"
      << "final_time = " << final_time << "\n"
      << "final_max_v = " << final_diag.max_v << "\n"
      << "final_excited_area = " << final_diag.excited_area << "\n"
      << "final_excited_fraction = " << final_diag.excited_fraction << "\n";
}

// ---- Grid, system, matrices ------------------------------------------------

template <int dim>
void ParallelOSFHN<dim>::make_grid()
{
  TimerOutput::Scope t(computing_timer, "Make grid");
  GridGenerator::hyper_cube(triangulation,
                            -prm.domain_half_length,
                             prm.domain_half_length);
  triangulation.refine_global(prm.n_refinements);
}

template <int dim>
void ParallelOSFHN<dim>::setup_system()
{
  TimerOutput::Scope t(computing_timer, "Setup system");

  dof_handler.distribute_dofs(fe);
  locally_owned_dofs   = dof_handler.locally_owned_dofs();
  locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

  Assert(locally_owned_dofs.is_contiguous(),
         ExcMessage("PETSc requires contiguous locally owned DoFs."));

  const std::vector<types::global_dof_index> dofs_per_component =
    DoFTools::count_dofs_per_fe_component(dof_handler);
  pcout << "Active cells: " << triangulation.n_global_active_cells()
        << "  DoFs: " << dof_handler.n_dofs()
        << " (" << dofs_per_component[0] << " + " << dofs_per_component[1] << ")\n";

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
  SparsityTools::distribute_sparsity_pattern(dsp, locally_owned_dofs,
                                             mpi_communicator, locally_relevant_dofs);

  mass_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);
  laplace_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);
  system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);

  solution.reinit(locally_owned_dofs, mpi_communicator);
  locally_relevant_solution.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
  system_rhs.reinit(locally_owned_dofs, mpi_communicator);
}

template <int dim>
void ParallelOSFHN<dim>::assemble_matrices()
{
  TimerOutput::Scope t(computing_timer, "Assemble matrices");

  mass_matrix   = 0;
  laplace_matrix = 0;

  const QGauss<dim> qf(fe.degree + 1);
  FEValues<dim> fev(fe, qf, update_values | update_gradients | update_JxW_values);

  const unsigned int dpc = fe.dofs_per_cell;
  const unsigned int nq  = qf.size();

  FullMatrix<double>                   cell_mass(dpc, dpc);
  FullMatrix<double>                   cell_lap(dpc, dpc);
  std::vector<types::global_dof_index> local_dof_indices(dpc);

  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        fev.reinit(cell);
        cell_mass = 0;
        cell_lap  = 0;

        for (unsigned int q = 0; q < nq; ++q)
          for (unsigned int i = 0; i < dpc; ++i)
            {
              const unsigned int ci = fe.system_to_component_index(i).first;
              for (unsigned int j = 0; j < dpc; ++j)
                {
                  const unsigned int cj = fe.system_to_component_index(j).first;
                  if (ci == cj)
                    {
                      cell_mass(i, j) += fev.shape_value(i, q) *
                                         fev.shape_value(j, q) * fev.JxW(q);

                      const double D = (ci == 0) ? prm.diffusion_v : prm.diffusion_w;
                      cell_lap(i, j) -= D *
                                        fev.shape_grad(i, q) *
                                        fev.shape_grad(j, q) * fev.JxW(q);
                    }
                }
            }

        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(cell_mass, local_dof_indices, mass_matrix);
        constraints.distribute_local_to_global(cell_lap,  local_dof_indices, laplace_matrix);
      }

  mass_matrix.compress(VectorOperation::add);
  laplace_matrix.compress(VectorOperation::add);

  LA::MPI::PreconditionAMG::AdditionalData amg_data;
  amg_data.symmetric_operator = true;
  mass_amg.initialize(mass_matrix, amg_data);
}

template <int dim>
void ParallelOSFHN<dim>::set_initial_conditions()
{
  TimerOutput::Scope t(computing_timer, "Set initial conditions");

  const auto ic = InitialConditions::make_initial_condition<dim>(prm.initial_condition);
  VectorTools::interpolate(dof_handler, *ic, solution);
  constraints.distribute(solution);
  solution.compress(VectorOperation::insert);
  apply_solution_limiter(solution);
  locally_relevant_solution = solution;

  pcout << "Initial condition: type=" << prm.initial_condition.type
        << "  amp_v=" << prm.initial_condition.amplitude_v
        << "  amp_w=" << prm.initial_condition.amplitude_w << "\n"
        << "  output dir = " << prm.make_output_directory().string() << "\n";
}

template <int dim>
void ParallelOSFHN<dim>::refine_mesh()
{
  TimerOutput::Scope t(computing_timer, "Refine mesh");

  locally_relevant_solution = solution;
  estimated_error_per_cell.grow_or_shrink(triangulation.n_active_cells());

  ComponentMask vmask(fe.n_components(), false);
  vmask.set(0, true);

  KellyErrorEstimator<dim>::estimate(
    dof_handler, QGauss<dim - 1>(fe.degree + 1), {}, locally_relevant_solution,
    estimated_error_per_cell, vmask, nullptr,
    numbers::invalid_unsigned_int, numbers::invalid_subdomain_id,
    numbers::invalid_material_id,
    KellyErrorEstimator<dim>::Strategy::face_diameter_over_twice_max_degree);

  parallel::distributed::GridRefinement::refine_and_coarsen_fixed_fraction(
    triangulation, estimated_error_per_cell,
    prm.mesh_refinement_fraction, prm.mesh_coarsening_fraction);

  const unsigned int min_level = prm.n_refinements;
  const unsigned int max_level = prm.n_refinements + prm.max_delta_refinement_level;

  if (triangulation.n_levels() > max_level)
    for (const auto &cell : triangulation.active_cell_iterators_on_level(max_level))
      cell->clear_refine_flag();
  for (const auto &cell : triangulation.active_cell_iterators_on_level(min_level))
    cell->clear_coarsen_flag();

  LA::MPI::Vector prev(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
  prev = solution;

  parallel::distributed::SolutionTransfer<dim, LA::MPI::Vector> transfer(dof_handler);
  triangulation.prepare_coarsening_and_refinement();
  transfer.prepare_for_coarsening_and_refinement(prev);
  triangulation.execute_coarsening_and_refinement();

  setup_system();

  LA::MPI::Vector transferred(locally_owned_dofs, mpi_communicator);
  transfer.interpolate(transferred);
  solution = transferred;
  constraints.distribute(solution);
  solution.compress(VectorOperation::insert);
  apply_solution_limiter(solution);
  locally_relevant_solution = solution;

  assemble_matrices();
}

// ---- Diffusion sub-step (implicit) -----------------------------------------
//
//  The diffusion ODE in weak form is:  M * dy/dt = L * y
//  Written as dy/dt = f_diff(y) = M^{-1} * L * y
//
//  The ImplicitRK solver needs:
//    f(t, y, f_out)           -- the RHS
//    id_minus_tau_J^{-1}(t, tau, y, out)  -- solves (I - tau*J)*out = y
//
//  where J = M^{-1} * L, so (I - tau*J)*out = y
//        <=> M*out - tau*L*out = M*y
//        <=> (M - tau*L)*out = M*y

template <int dim>
void ParallelOSFHN<dim>::solve_diffusion_rhs(const double           /*time*/,
                                                 const LA::MPI::Vector &y,
                                                 LA::MPI::Vector       &f_out)
{
  TimerOutput::Scope t(computing_timer, "Diffusion RHS");

  LA::MPI::Vector tmp(locally_owned_dofs, mpi_communicator);
  laplace_matrix.vmult(tmp, y);

  f_out.reinit(locally_owned_dofs, mpi_communicator);

  SolverControl solver_control(1000, 1e-12);
  LA::SolverCG  solver(solver_control);
  solver.solve(mass_matrix, f_out, tmp, mass_amg);
  constraints.distribute(f_out);
}

template <int dim>
void ParallelOSFHN<dim>::solve_diffusion_lhs(const double           /*time*/,
                                                 const double           tau,
                                                 const LA::MPI::Vector &y,
                                                 LA::MPI::Vector       &out)
{
  TimerOutput::Scope t(computing_timer, "Diffusion LHS");

  // Build (M - tau * L)
  // laplace_matrix entries = -D * ∫ ∇φ_i·∇φ_j, so (M - tau*L) is positive definite
  system_matrix.copy_from(mass_matrix);
  system_matrix.add(-tau, laplace_matrix);

  // RHS = M * y
  LA::MPI::Vector rhs(locally_owned_dofs, mpi_communicator);
  mass_matrix.vmult(rhs, y);

  out.reinit(locally_owned_dofs, mpi_communicator);

  SolverControl solver_control(2000, 1e-10);
  LA::SolverCG  solver(solver_control);
  LA::MPI::PreconditionAMG preconditioner;
  LA::MPI::PreconditionAMG::AdditionalData data;
  data.symmetric_operator = true;
  preconditioner.initialize(system_matrix, data);

  solver.solve(system_matrix, out, rhs, preconditioner);
  constraints.distribute(out);
}

// ---- Reaction sub-step (explicit) ------------------------------------------
//
//  Reaction ODE in weak form: M * dy/dt = reaction(y)
//  Written as dy/dt = f_react(y) = M^{-1} * reaction(y)

template <int dim>
void ParallelOSFHN<dim>::solve_reaction_rhs(const double           /*time*/,
                                                const LA::MPI::Vector &y,
                                                LA::MPI::Vector       &f_out)
{
  TimerOutput::Scope t(computing_timer, "Reaction RHS");

  LA::MPI::Vector react_vec(locally_owned_dofs, mpi_communicator);
  react_vec = 0;

  const QGauss<dim> qf(fe.degree + 1);
  FEValues<dim> fev(fe, qf, update_values | update_JxW_values);

  const unsigned int dpc = fe.dofs_per_cell;
  const unsigned int nq  = qf.size();

  Vector<double>                       cell_rhs(dpc);
  std::vector<types::global_dof_index> local_dof_indices(dpc);
  std::vector<double>                  local_vals(dpc);

  LA::MPI::Vector y_ghost(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
  y_ghost = y;

  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      {
        fev.reinit(cell);
        cell_rhs = 0;
        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dpc; ++i)
          local_vals[i] = y_ghost[local_dof_indices[i]];

        for (unsigned int q = 0; q < nq; ++q)
          {
            double v_q = 0.0, w_q = 0.0;
            for (unsigned int i = 0; i < dpc; ++i)
              {
                const unsigned int ci = fe.system_to_component_index(i).first;
                if (ci == 0) v_q += local_vals[i] * fev.shape_value(i, q);
                else         w_q += local_vals[i] * fev.shape_value(i, q);
              }

            double f_v = 0.0, f_w = 0.0;
            if (prm.kinetic_model == "kinetic_II")
              {
                f_v = v_q * (1.0 - v_q) * (v_q - prm.beta_param) - w_q;
                f_w = prm.gamma * (prm.alpha_param * v_q - w_q);
              }
            else
              {
                const double V_th = (w_q + prm.b_param) / prm.a_param;
                f_v = (1.0 / prm.epsilon) * v_q * (1.0 - v_q) * (v_q - V_th);
                f_w = prm.alpha_param * v_q - prm.gamma * w_q;
              }

            for (unsigned int i = 0; i < dpc; ++i)
              {
                const unsigned int ci = fe.system_to_component_index(i).first;
                cell_rhs[i] += (ci == 0 ? f_v : f_w) *
                               fev.shape_value(i, q) * fev.JxW(q);
              }
          }

        constraints.distribute_local_to_global(cell_rhs, local_dof_indices, react_vec);
      }

  react_vec.compress(VectorOperation::add);

  f_out.reinit(locally_owned_dofs, mpi_communicator);

  SolverControl solver_control(1000, 1e-12);
  LA::SolverCG  solver(solver_control);
  solver.solve(mass_matrix, f_out, react_vec, mass_amg);
  constraints.distribute(f_out);
}

// ---- Output ----------------------------------------------------------------

template <int dim>
void ParallelOSFHN<dim>::output_results(const double       time,
                                            const unsigned int step) const
{
  const std::filesystem::path outdir = prm.make_output_directory();

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);

  const std::vector<std::string> names = {"v", "w"};
  const std::vector<DataComponentInterpretation::DataComponentInterpretation>
    interp(2, DataComponentInterpretation::component_is_scalar);

  data_out.add_data_vector(locally_relevant_solution, names,
                           DataOut<dim>::type_dof_data, interp);
  data_out.add_data_vector(estimated_error_per_cell, "error");
  data_out.build_patches();
  data_out.set_flags(DataOutBase::VtkFlags(time, step));

  const std::string fname = "solution-" + Utilities::int_to_string(step, 4) + ".vtu";
  data_out.write_vtu_in_parallel((outdir / fname).string(), mpi_communicator);

  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    {
      static std::vector<std::pair<double, std::string>> times_and_names;
      times_and_names.emplace_back(time, fname);
      std::ofstream pvd((outdir / "solution.pvd").string());
      DataOutBase::write_pvd_record(pvd, times_and_names);
    }
}

// ---- run() -----------------------------------------------------------------

template <int dim>
void ParallelOSFHN<dim>::run()
{
  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    std::filesystem::create_directories(prm.make_output_directory());
  MPI_Barrier(mpi_communicator);

  make_grid();

  pcout << "Setting up parallel system...\n";
  setup_system();

  pcout << "Assembling matrices...\n";
  assemble_matrices();

  pcout << "Setting initial conditions...\n";
  set_initial_conditions();

  initialize_diagnostics_file();

  DiagnosticsRecord diag0 = compute_diagnostics();
  initial_excited_area = diag0.excited_area;
  initial_max_v        = diag0.max_v;
  diag0.relative_excited_area =
    diag0.excited_area / std::max(initial_excited_area, prm.epsilon_area);
  append_diagnostics(0.0, 0, diag0);
  update_ignition_state(0.0, diag0);

  // ------------------------------------------------------------------
  //  Build operator-split time stepper
  // ------------------------------------------------------------------

  pcout << "Starting time integration with operator splitting...\n"
        << "  OS method:        " << prm.os_method << "\n"
        << "  Diffusion method: " << prm.diffusion_method << "\n"
        << "  Reaction method:  " << prm.reaction_method  << "\n";

  tostii::ImplicitRungeKutta<LA::MPI::Vector> diffusion_ts(
    tostii::RK_string_to_enum(prm.diffusion_method));

  tostii::ExplicitRungeKutta<LA::MPI::Vector> reaction_ts(
    tostii::RK_string_to_enum(prm.reaction_method));

  using os_op_t = tostii::OSoperator<LA::MPI::Vector>;

  os_op_t diffusion_op{
    &diffusion_ts,
    [this](const double t, const LA::MPI::Vector &y, LA::MPI::Vector &f) {
      this->solve_diffusion_rhs(t, y, f);
    },
    [this](const double t, const double tau,
           const LA::MPI::Vector &y, LA::MPI::Vector &out) {
      this->solve_diffusion_lhs(t, tau, y, out);
    }
  };

  os_op_t reaction_op{
    &reaction_ts,
    [this](const double t, const LA::MPI::Vector &y, LA::MPI::Vector &f) {
      this->solve_reaction_rhs(t, y, f);
    },
    // Dummy Jacobian solver (not called for explicit methods)
    [](const double, const double, const LA::MPI::Vector &, LA::MPI::Vector &) {}
  };

  // Retrieve the stage sequence for the chosen OS method (e.g. Strang):
  //   op 0 = diffusion (implicit), op 1 = reaction (explicit)
  const std::vector<tostii::OSpair<double>> stages =
    tostii::os_method.at(prm.os_method);

  tostii::OperatorSplitSingle<LA::MPI::Vector> os_stepper(
    solution,                        // reference vector for sizing
    {diffusion_op, reaction_op},
    stages);

  // ------------------------------------------------------------------
  //  Time loop
  // ------------------------------------------------------------------

  double       time           = 0.0;
  unsigned int output_counter = 0;
  unsigned int adapt_counter  = 0;

  locally_relevant_solution = solution;
  output_results(time, output_counter);

  while (time < prm.final_time - 1e-14)
    {
      const double dt = std::min(prm.time_step, prm.final_time - time);

      os_stepper.evolve_one_time_step(time, dt, solution);
      apply_solution_limiter(solution);
      time += dt;

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
          pcout << "Adapting mesh at step " << output_counter
                << " (t = " << time << ")...\n";
          refine_mesh();
          locally_relevant_solution = solution;
        }
    }

  const DiagnosticsRecord final_diag = compute_diagnostics();
  write_ignition_summary(time, final_diag);

  const bool window_reached = (diagnostic_samples_after_start > 0);
  const bool ignited        = window_reached &&
    max_consecutive_ignition_hits >= prm.required_ignition_hits;

  if (window_reached)
    pcout << "Ignition diagnosis after t >= " << prm.diagnosis_start_time << ": "
          << (ignited ? "IGNITED" : "NOT IGNITED")
          << " (max consecutive hits = " << max_consecutive_ignition_hits
          << ", required = " << prm.required_ignition_hits << ")\n";
  else
    pcout << "Ignition window not reached (no samples after t="
          << prm.diagnosis_start_time << ").\n";

  pcout << "Simulation completed.\n";
}

// ============================================================
//  main
// ============================================================

int main(int argc, char *argv[])
{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      const std::string parameter_file =
        (argc > 1) ? argv[1] : "./parallel_os_fhn.prm";

      std::ifstream parameter_stream(parameter_file);
      AssertThrow(parameter_stream.is_open(),
                  ExcMessage("Could not open parameter file: " + parameter_file));

      ParameterHandler prm;
      ExperimentParameters::declare_parameters(prm);
      prm.parse_input(parameter_file);

      ExperimentParameters params;
      params.parse_parameters(prm);

      ParallelOSFHN<2> solver(params);
      solver.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << "\n----------------------------------------------------\n"
                << "Exception: " << exc.what() << "\nAborting!\n"
                << "----------------------------------------------------\n";
      return 1;
    }
  catch (...)
    {
      std::cerr << "\n----------------------------------------------------\n"
                << "Unknown exception!\nAborting!\n"
                << "----------------------------------------------------\n";
      return 1;
    }
  return 0;
}
