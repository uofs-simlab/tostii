/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2021 by the authors listed below
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
 * Authors: Sebastian Dominguez, University of Saskatchewan, 2021
 *          Kevin R. Green, University of Saskatchewan, 2021
 *          Joyce Reimer, University of Saskatchewan, 2021
 *
 * This Monodomain solver code is based on the tutorials step-6,
 * step-7, step-11, step-17, step-18, step-40, and step-52 for the FEM
 * formulations, but uses the tost.II library for its time integration
 */

// @sect3{Include files}
//
// Most of the include files we need for this program have already been
// discussed in previous programs. In particular, all of the following should
// already be familiar friends:

#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
// Header needed to build conductivity tensor
#include <deal.II/base/tensor_function.h>

#include <deal.II/lac/generic_linear_algebra.h>

// Uncomment the following \#define if you have PETSc and Trilinos installed
// and you prefer using Trilinos in this example:
// @code
// #define FORCE_USE_OF_TRILINOS
// @endcode

// This will either import PETSc or TrilinosWrappers into the namespace LA.
// Note that we are defining the macro USE_PETSC_LA so that we can detect
// if we are using PETSc.
//     (See solve_mass_matrix() for an example where this is necessary)
namespace LA {
using namespace dealii::LinearAlgebraPETSc;
} // namespace LA

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/vector.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

// The following, however, will be new or be used in new roles. Let's walk
// through them. The first of these will provide the tools of the
// Utilities::System namespace that we will use to query things like the
// number of processors associated with the current MPI universe, or the
// number within this universe the processor this job runs on is:
#include <deal.II/base/utilities.h>
// The next one provides a class, ConditionOStream that allows us to write
// code that would output things to a stream (such as <code>std::cout</code>
// on every processor but throws the text away on all but one of them. We
// could achieve the same by simply putting an <code>if</code> statement in
// front of each place where we may generate output, but this doesn't make the
// code any prettier. In addition, the condition whether this processor should
// or should not produce output to the screen is the same every time -- and
// consequently it should be simple enough to put it into the statements that
// generate output itself.
#include <deal.II/base/conditional_ostream.h>
// After these preliminaries, here is where it becomes more interesting. As
// mentioned in the @ref distributed module, one of the fundamental truths of
// solving problems on large numbers of processors is that there is no way for
// any processor to store everything (e.g. information about all cells in the
// mesh, all degrees of freedom, or the values of all elements of the solution
// vector). Rather, every processor will <i>own</i> a few of each of these
// and, if necessary, may <i>know</i> about a few more, for example the ones
// that are located on cells adjacent to the ones this processor owns
// itself. We typically call the latter <i>ghost cells</i>, <i>ghost nodes</i>
// or <i>ghost elements of a vector</i>. The point of this discussion here is
// that we need to have a way to indicate which elements a particular
// processor owns or need to know of. This is the realm of the IndexSet class:
// if there are a total of $N$ cells, degrees of freedom, or vector elements,
// associated with (non-negative) integral indices $[0,N)$, then both the set
// of elements the current processor owns as well as the (possibly larger) set
// of indices it needs to know about are subsets of the set $[0,N)$. IndexSet
// is a class that stores subsets of this set in an efficient format:
#include <deal.II/base/index_set.h>
// The next header file is necessary for a single function,
// SparsityTools::distribute_sparsity_pattern. The role of this function will
// be explained below.
#include <deal.II/lac/sparsity_tools.h>

// The final two, new header files provide the class
// parallel::distributed::Triangulation that provides meshes distributed
// across a potentially very large number of processors, while the second
// provides the namespace parallel::distributed::GridRefinement that offers
// functions that can adaptively refine such distributed meshes:
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>

// Include after:
// - namespace LA definition
#include "FitzHughNagumo1961EB2008.h"

// Separated Timestepping
#include <tostii/tostii.h>

#include <fstream>
#include <iostream>

// Implementation of the monodomain problem. The set of equations
// implemented here is
//   chi*Cm*dv/dt + chi*I_ion(v) = lambda/(1+lambda)*div(sigmai*grad(v)) + f,
// with boundary conditions
//   sigmai*grad(v)*n = 0,
// and initial conditions
//   v = v_0 at t = 0.
// Here n denotes the normal unit vector on the boundary of the domain.
// To test convergence, we set the current I_ion to be I_ion = v (passive
// cell model). A toy problem for convergence is considered on the unit square
// [0,1]^dim, dim = 2 or 3, with solution
//   v = t^3 * cos(pi*x) * cos(pi*y) (in 2D),
//       t^3 * cos(pi*x) * cos(pi*y) * cos(pi*z) (in 3D).
// This solution satisfies:
//   - the initial condition, v_0 = zero,
//   - the boundary conditions, sigmai*grad(v)*n|_{boundary} = 0.
// With this chosen solution, the right hand side f must be defined as
// f = (1/t + dim * lambda / (1+lambda) * sigmai * pi^2) v, dim = 2 or 3.
// The conductivity tensor sigmai has been set to be a scaled version of the
// identity tensor, that is sigmai = sigmai * I.

namespace Parameters {
using namespace dealii;

// Parameters for controlling the FEM spatial discretization.
struct FEMParameters {
  unsigned int dim;
  double       length;
  double       width;
  unsigned int polynomial_degree;
  unsigned int quadrature_order;
  unsigned int global_refinement_level;
  bool         adaptive_refinement;
  std::string  boundary_condition_type;

  static void declare_parameters(ParameterHandler& prm);
  void        parse_parameters(ParameterHandler& prm);
};

void FEMParameters::declare_parameters(ParameterHandler& prm) {
  prm.enter_subsection("FEM Parameters");
  {
    prm.declare_entry("Dimension value",   //
                      "2",                 //
                      Patterns::Integer(), //
                      "Problem dimension (Currently setup only for 2)");

    prm.declare_entry("Length",           //
                      "1e0",              //
                      Patterns::Double(), //
                      "Domain length [mm]");

    prm.declare_entry("Width",            //
                      "1e0",              //
                      Patterns::Double(), //
                      "Domain width [mm]");

    prm.declare_entry("Polynomial degree", //
                      "1",                 //
                      Patterns::Integer(), //
                      "Polynogiaml degree for FEM discretization");

    prm.declare_entry("Global refinement value", //
                      "1",                       //
                      Patterns::Integer(),       //
                      "Global refinement level");

    prm.declare_entry("Adaptive refinement", //
                      "true",                //
                      Patterns::Bool(),      //
                      "Apply daptive refinement");

    prm.declare_entry("Boundary condition type",                //
                      "Neumann",                                //
                      Patterns::Selection("Neumann|Dirichlet"), //
                      "Boundary condition type (Neumann or Dirichlet)");
  }
  prm.leave_subsection();
}

void FEMParameters::parse_parameters(ParameterHandler& prm) {
  prm.enter_subsection("FEM Parameters");
  {
    dim                     = prm.get_integer("Dimension value");
    length                  = prm.get_double("Length");
    width                   = prm.get_double("Width");
    polynomial_degree       = prm.get_double("Polynomial degree");
    quadrature_order        = polynomial_degree + 2;
    global_refinement_level = prm.get_double("Global refinement value");
    boundary_condition_type = prm.get("Boundary condition type");
    adaptive_refinement     = prm.get_bool("Adaptive refinement");
  }
  prm.leave_subsection();
}

// Parameters for controlling the time stepping algorithm.
struct TimeSteppingParameters {
  unsigned int n_time_steps;
  double       initial_time;
  double       final_time;

  std::string os_time_stepping_method;
  std::string tissue_time_stepping_method;
  std::string membrane_time_stepping_method;

  static void declare_parameters(ParameterHandler& prm);
  void        parse_parameters(ParameterHandler& prm);
};

void TimeSteppingParameters::declare_parameters(ParameterHandler& prm) {
  prm.enter_subsection("Time Stepping Parameters");
  {
    prm.declare_entry("Number of time steps", //
                      "1",                    //
                      Patterns::Integer(),    //
                      "Number of time steps (positive integer)");

    prm.declare_entry("Initial time value",
                      "0e0",              //
                      Patterns::Double(), //
                      "Initial time t0 (in ms)");

    prm.declare_entry("Final time value", //
                      "1e0",              //
                      Patterns::Double(), //
                      "Final time tf (in ms)");

    prm.declare_entry("OperatorSplit time stepping method",               //
                      "GODUNOV",                                          //
                      Patterns::Selection("GODUNOV|STRANG|STRANG2|RUTH|"
					  "BEST2-2|Emb_3_AK_p|Emb_4_AK_p"), //
                      "OperatorSplit time stepping method (see choices below)");

    prm.declare_entry("Tissue time stepping method",                          //
                      "BACKWARD_EULER",                                       //
                      Patterns::Selection("BACKWARD_EULER|IMPLICIT_MIDPOINT|" //
                                          "CRANK_NICOLSON|SDIRK_TWO_STAGES|"  //
                                          "SDIRK_THREE_STAGES|"               //
                                          "SDIRK_3O4|SDIRK_5O4"),             //
                      "Tissue time stepping method (see choices below)");

    prm.declare_entry("Membrane time stepping method", "FORWARD_EULER", //
                      Patterns::Selection("FORWARD_EULER"               //
                                          "|IMPLICIT_MIDPOINT"          //
                                          "|HEUN2"                      //
                                          "|RK_THIRD_ORDER"             //
                                          "|RK_CLASSIC_FOURTH_ORDER"),  //
                      "Membrane time stepping method (see choices below)");
  }
  prm.leave_subsection();
}

void TimeSteppingParameters::parse_parameters(ParameterHandler& prm) {
  prm.enter_subsection("Time Stepping Parameters");
  {
    n_time_steps                = prm.get_integer("Number of time steps");
    initial_time                = prm.get_double("Initial time value");
    final_time                  = prm.get_double("Final time value");
    os_time_stepping_method     = prm.get("OperatorSplit time stepping method");
    tissue_time_stepping_method = prm.get("Tissue time stepping method");
    membrane_time_stepping_method = prm.get("Membrane time stepping method");
  }
  prm.leave_subsection();
}

// Parameters for controlling the Monodomain tissue model parameters
struct TissueParameters {
  double chi;
  double Cm;
  double sigmaix;
  double sigmaiy;
  double sigmaixy;
  double lambda;

  static void declare_parameters(ParameterHandler& prm);
  void        parse_parameters(ParameterHandler& prm);
};

void TissueParameters::declare_parameters(ParameterHandler& prm) {
  prm.enter_subsection("Tissue Parameters");
  {
    prm.declare_entry("chi value",        //
                      "1e0",              //
                      Patterns::Double(), //
                      "Cells per unit volume (chi, in )");

    prm.declare_entry("Cm value",         //
                      "1e0",              //
                      Patterns::Double(), //
                      "Capacitance (Cm, in )");

    prm.declare_entry("sigmaix value",    //
                      "1e0",              //
                      Patterns::Double(), //
                      "Intracellular conductivity x component");

    prm.declare_entry("sigmaiy value",    //
                      "1e0",              //
                      Patterns::Double(), //
                      "Intracellular conductivity y component");

    prm.declare_entry("sigmaixy value",    //
                      "0e0",              //
                      Patterns::Double(), //
                      "Intracellular conductivity xy component");

    prm.declare_entry("lambda value",     //
                      "1e0",              //
                      Patterns::Double(), //
                      "Multiplier lambda");
  }
  prm.leave_subsection();
}

void TissueParameters::parse_parameters(ParameterHandler& prm) {
  prm.enter_subsection("Tissue Parameters");
  {
    chi      = prm.get_double("chi value");
    Cm       = prm.get_double("Cm value");
    sigmaix  = prm.get_double("sigmaix value");
    sigmaiy  = prm.get_double("sigmaiy value");
    sigmaixy = prm.get_double("sigmaixy value");
    lambda   = prm.get_double("lambda value");
  }
  prm.leave_subsection();
}

// Parameters for controlling the Linear Solver algorithm
struct LinearSolverParameters {
  std::string type_lin;
  double      tol_linear_solver;
  int         max_iterations_linear_solver;

  static void declare_parameters(ParameterHandler& prm);
  void        parse_parameters(ParameterHandler& prm);
};

void LinearSolverParameters::declare_parameters(ParameterHandler& prm) {
  prm.enter_subsection("Linear Solver Parameters");
  {
    prm.declare_entry("Solver type",             //
                      "CG",                      //
                      Patterns::Selection("CG"), //
                      "Type of solver used to solve the linear system");

    prm.declare_entry("Residual",         //
                      "1e-6",             //
                      Patterns::Double(), //
                      "Linear solver residual (scaled by residual norm)");
    prm.declare_entry(
        "Max iteration multiplier", //
        "1",                        //
        Patterns::Integer(),        //
        "Linear solver iterations (multiples of the system matrix size)");
  }
  prm.leave_subsection();
}

void LinearSolverParameters::parse_parameters(ParameterHandler& prm) {
  prm.enter_subsection("Linear Solver Parameters");
  {
    type_lin                     = prm.get("Solver type");
    tol_linear_solver            = prm.get_double("Residual");
    max_iterations_linear_solver = prm.get_integer("Max iteration multiplier");
  }
  prm.leave_subsection();
}

// Parameters for controlling the Output
struct OutputParameters {
  int         n_info_messages;
  bool        output_pvtu_files;
  int         n_pvtu_files;
  bool        output_solution_error_data;
  std::string output_solution_error_filename;
  static void declare_parameters(ParameterHandler& prm);
  void        parse_parameters(ParameterHandler& prm);
};

void OutputParameters::declare_parameters(ParameterHandler& prm) {
  prm.enter_subsection("Output Parameters");
  {
    prm.declare_entry(
        "Number of messages to print", //
        "1",                           //
        Patterns::Integer(),           //
        "Number of messages to print in terminal during computation");

    prm.declare_entry("Output solution pvtu files", //
                      "true",                       //
                      Patterns::Bool(),             //
                      "Flag to output pvtu files");

    prm.declare_entry("Number of pvtu files", //
                      "1",                    //
                      Patterns::Integer(),    //
                      "Number of pvtu files to visualize solution");

    prm.declare_entry("Output solution error data", //
                      "true",                       //
                      Patterns::Bool(),             //
                      "Flag to output solution error data");

    prm.declare_entry("File name",          //
                      "test.txt",           //
                      Patterns::FileName(), //
                      "File name for convergence study data");
  }
  prm.leave_subsection();
}

void OutputParameters::parse_parameters(ParameterHandler& prm) {
  prm.enter_subsection("Output Parameters");
  {
    n_info_messages            = prm.get_integer("Number of messages to print");
    output_pvtu_files          = prm.get_bool("Output solution pvtu files");
    n_pvtu_files               = prm.get_integer("Number of pvtu files");
    output_solution_error_data = prm.get_bool("Output solution error data");
    output_solution_error_filename = prm.get("File name");
  }
  prm.leave_subsection();
}

// Structure to call and gather all parameters
struct AllParameters : public FEMParameters,
                       public TimeSteppingParameters,
                       public TissueParameters,
                       public LinearSolverParameters,
                       public OutputParameters {
  AllParameters(const std::string& input_file);
  static void declare_parameters(ParameterHandler& prm);
  void        parse_parameters(ParameterHandler& prm);
};

AllParameters::AllParameters(const std::string& input_file) {
  ParameterHandler prm;
  declare_parameters(prm);
  prm.parse_input(input_file);
  parse_parameters(prm);
}

void AllParameters::declare_parameters(ParameterHandler& prm) {
  FEMParameters::declare_parameters(prm);
  TimeSteppingParameters::declare_parameters(prm);
  TissueParameters::declare_parameters(prm);
  LinearSolverParameters::declare_parameters(prm);
  OutputParameters::declare_parameters(prm);
}

void AllParameters::parse_parameters(ParameterHandler& prm) {
  FEMParameters::parse_parameters(prm);
  TimeSteppingParameters::parse_parameters(prm);
  TissueParameters::parse_parameters(prm);
  LinearSolverParameters::parse_parameters(prm);
  OutputParameters::parse_parameters(prm);
}
} // namespace Parameters

namespace Monodomain {
using namespace dealii;

namespace PrescribedData {
// Implementation of the right hand side f. The class implemented here works
// well for homogeneous problems.

// Implementation of the Neumann (flux) boundary conditions on the potential.
// This is only used when
//        parameters.boundary_condition_type == "Neumann".
template <int dim> class FluxBoundaryValues : public Function<dim> {
public:
  FluxBoundaryValues() : Function<dim>(1) {}

  virtual double value(const Point<dim>&  p,
                       const unsigned int component = 0) const override;
};

template <int dim>
double
FluxBoundaryValues<dim>::value(const Point<dim>& /* p */,
                               const unsigned int /* component */) const {
  return 0.0;
}

// Implementation of the Dirichlet boundary conditions for the potential.
// This is only used when
//        parameters.boundary_condition_type is equal == "Dirichlet".
template <int dim> class PotentialBoundaryValues : public Function<dim> {
public:
  PotentialBoundaryValues(const double current_time);

  virtual double value(const Point<dim>&  p,
                       const unsigned int component = 0) const override;

private:
  const double current_time;
};

template <int dim>
PotentialBoundaryValues<dim>::PotentialBoundaryValues(const double current_time)
    : Function<dim>(1), //
      current_time(current_time) {}

template <int dim>
double
PotentialBoundaryValues<dim>::value(const Point<dim>& /* p */,
                                    const unsigned int /*component*/) const {
  return 0.0;
}

template <int dim> class Stimulus : public Function<dim, double> {
 private:
  double time;

  double magnitude;
  double start;
  double duration;

  // axis-oriented ellipses for stimulus
  Point<dim> center;
  Point<dim> radius;

public:
  Stimulus(double current_time, Parameters::AllParameters /*parameters*/)
      : Function<dim, double>(1), time(current_time) {

    magnitude = 10.0;
    start = 0.;
    duration  = 1.0;

    // Center of stimulus
    center[0] = 0.;
    center[1] = 0.;

    // radial extents of stimulus
    radius[0] = 1.;
    radius[1] = 1.;
  }

  virtual double value(const Point<dim>&  p,
                       const unsigned int component = 0) const override;
};


template <int dim>
double Stimulus<dim>::value(const Point<dim>&  p,
			    const unsigned int component) const {
  static_assert(dim == 2, "This initial condition only works in 2d.");

  (void)component;
  Assert(component == 0, ExcIndexRange(component, 0, 1));

  double return_val = 0.0;

  if (time < duration) {
    double RSQ = radius[0] * radius[0];
    double xdiff = p[0]-center[0];
    double ydiff = p[1]-center[1];
    double rsq = xdiff*xdiff + ydiff*ydiff;
    if (rsq <= RSQ) {
      return_val = magnitude;
    }
  }
  return return_val;
}



// Implementation of the conductivity tensor sigmai.
template <int dim> class ConductivityTensor : public TensorFunction<2, dim> {
private:

  Tensor<2,dim> conductivity;

public:
  ConductivityTensor(Parameters::AllParameters parameters) : TensorFunction<2, dim>() {
    conductivity[0][0] = parameters.sigmaix;
    conductivity[1][1] = parameters.sigmaiy;

    conductivity[0][1] = parameters.sigmaixy;
    conductivity[1][0] = parameters.sigmaixy;
  }

  virtual void value_list(const std::vector<Point<dim>>& points,
                          std::vector<Tensor<2, dim>>&   values) const override;
};

template <int dim>
void ConductivityTensor<dim>::value_list(
    const std::vector<Point<dim>>& points, //
    std::vector<Tensor<2, dim>>&   values) const {
  (void)points;
  AssertDimension(points.size(), values.size());

  for (auto& value : values)
    value = conductivity;
}

} // namespace PrescribedData

// @sect3{The <code>MonodomainProblem</code> class template}

// Next let's declare the main class of this program. Its structure is
// almost exactly that of the step-6 tutorial program. The only significant
// differences are:
// - The <code>mpi_communicator</code> variable that
//   describes the set of processors we want this code to run on. In practice,
//   this will be MPI_COMM_WORLD, i.e. all processors the batch scheduling
//   system has assigned to this particular job.
// - The presence of the <code>pcout</code> variable of type ConditionOStream.
// - The obvious use of parallel::distributed::Triangulation instead of
// Triangulation.
// - The presence of two IndexSet objects that denote which sets of degrees of
//   freedom (and associated elements of solution and right hand side vectors)
//   we own on the current processor and which we need (as ghost elements) for
//   the algorithms in this program to work.
// - The fact that all matrices and vectors are now distributed. We use
//   either the PETSc or Trilinos wrapper classes so that we can use one of
//   the sophisticated preconditioners offered by Hypre (with PETSc) or ML
//   (with Trilinos). Note that as part of this class, we store a solution
//   vector that does not only contain the degrees of freedom the current
//   processor owns, but also (as ghost elements) all those vector elements
//   that correspond to "locally relevant" degrees of freedom (i.e. all
//   those that live on locally owned cells or the layer of ghost cells that
//   surround it).
template <int dim> class MonodomainProblem {
public:
  MonodomainProblem(Parameters::AllParameters& input_parameters);

  void run();

private:
  using os_pair_t = tostii::OSpair<double>;     //
  using os_t      = std::vector<os_pair_t>;     //
  using ts_mem_t  = tostii::runge_kutta_method; // type for membrane operator integration
  using ts_tis_t  = tostii::runge_kutta_method; // type for tissue operator integration

  void make_grid();
  void refine_grid();
  void setup_system(const double current_time, const unsigned int time_step);
  void assemble_all_system_matrices();
  void assemble_rhs_tissue(const double current_time);
  void assemble_rhs_membrane(const double                current_time, //
                             const LA::MPI::BlockVector& yin,          //
                             LA::MPI::BlockVector&       yout);
  void set_initial_conditions(LA::MPI::BlockVector& yinout);
  void solve_rhs_tissue(const double                current_time, //
                        const LA::MPI::BlockVector& yin,          //
                        LA::MPI::BlockVector&       yout);
  void solve_rhs_membrane(const double                current_time, //
                          const LA::MPI::BlockVector& yin,          //
                          LA::MPI::BlockVector&       yout);
  void solve_lhs_tissue(const double                current_time,
                        const double                tau, //
                        const LA::MPI::BlockVector& yin, //
                        LA::MPI::BlockVector&       yout);
  void os_time_stepping(const os_t         os_method,          //
                        const ts_mem_t     ts_method_membrane, //
                        const ts_tis_t     ts_method_tissue,   //
                        const unsigned int n_time_steps,       //
                        const double       initial_time,       //
                        const double       final_time);
  void time_integrate();
  void output_results(unsigned int time_step, double current_time) const;

  Parameters::AllParameters parameters;

  MPI_Comm mpi_communicator;

  parallel::distributed::Triangulation<dim> triangulation;

  FE_Q<dim>       fe;
  DoFHandler<dim> dof_handler;
  MappingFE<dim>  mapping;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  AffineConstraints<double> constraints;

  LA::MPI::SparseMatrix mass_matrix;
  LA::MPI::SparseMatrix tissue_matrix;
  // system_matrix represents the matrix we need for solving the linear systems
  // involved in implicit time stepping methods.
  // The general form it takes here is (two different operators):
  //   0: system_matrix = mass_matrix - tau * tissue_matrix.
  //   1: system_matrix = mass_matrix.
  // See step-52 for a more detailed discussion on this formulation.
  LA::MPI::SparseMatrix system_matrix;

  LA::MPI::BlockVector locally_relevant_solution;
  LA::MPI::BlockVector diffusion_rhs;
  LA::MPI::BlockVector reaction_rhs;

  ConditionalOStream pcout;
  TimerOutput        computing_timer;

  // Cell Model to be used
  FitzHughNagumo1961EB2008 membrane_model;

  /* removed - does not compile (maybe due to deal.II v9.5.1?)
  std::vector<Point<dim>> evaluation_points;
  Utilities::MPI::RemotePointEvaluation<dim> evaluation_cache;
  std::string eval_points_file;
  */
};

// @sect3{The <code>MonodomainProblem</code> class implementation}

// @sect4{Constructor}

// Constructors and destructors are rather trivial. In addition to what we
// do in step-6, we set the set of processors we want to work on to all
// machines available (MPI_COMM_WORLD); ask the triangulation to ensure that
// the mesh remains smooth and free to refined islands, for example; and
// initialize the <code>pcout</code> variable to only allow processor zero
// to output anything. The final piece is to initialize a timer that we
// use to determine how much compute time the different parts of the program
// take:
template <int dim>
MonodomainProblem<dim>::MonodomainProblem(
    Parameters::AllParameters& input_parameters)
    : parameters(input_parameters),                                      //
      mpi_communicator(MPI_COMM_WORLD),                                  //
      triangulation(mpi_communicator,                                    //
                    typename Triangulation<dim>::MeshSmoothing(          //
                        Triangulation<dim>::smoothing_on_refinement      //
                        | Triangulation<dim>::smoothing_on_coarsening)), //
      fe(parameters.polynomial_degree),                                  //
      dof_handler(triangulation),                                        //
      mapping(fe),
      pcout(std::cout,                                                   //
            (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)),  //
      computing_timer(mpi_communicator,                                  //
                      pcout,                                             //
                      TimerOutput::summary,                              //
                      TimerOutput::wall_times) {}

// @sect4{MonodomainProblem::make_grid}
// - Triangulate a rectangle from (0,0), to (length,width)
// - Refine the triangulation.
//
template <int dim> void MonodomainProblem<dim>::make_grid() {
  TimerOutput::Scope t(computing_timer, "make grid");

  const Point<dim> p0(0,0);
  const Point<dim> p1(parameters.length,parameters.width);
  // GridGenerator::hyper_rectangle(triangulation, p0, p1);
  // GridGenerator::hyper_shell(triangulation, p0, 0.1, 15.);
  GridGenerator::hyper_ball(triangulation, p0, parameters.length, false);
  triangulation.refine_global(parameters.global_refinement_level);
  triangulation.set_all_manifold_ids(1);
}

// @sect4{MonodomainProblem::refine_grid}

// The function that estimates the error and refines the grid is again
// almost exactly like the one in step-6. The only difference is that the
// function that flags cells to be refined is now in namespace
// parallel::distributed::GridRefinement -- a namespace that has functions
// that can communicate between all involved processors and determine global
// thresholds to use in deciding which cells to refine and which to coarsen.
//
// Note that we didn't have to do anything special about the
// KellyErrorEstimator class: we just give it a vector with as many elements
// as the local triangulation has cells (locally owned cells, ghost cells,
// and artificial ones), but it only fills those entries that correspond to
// cells that are locally owned.
template <int dim> void MonodomainProblem<dim>::refine_grid() {
  TimerOutput::Scope t(computing_timer, "refine");

  Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
  KellyErrorEstimator<dim>::estimate(
      dof_handler,                                          //
      QGauss<dim - 1>(parameters.quadrature_order),         //
      std::map<types::boundary_id, const Function<dim>*>(), //
      locally_relevant_solution, estimated_error_per_cell);
  parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
      triangulation,            //
      estimated_error_per_cell, //
      0.3,                      //
      0.03);
  triangulation.execute_coarsening_and_refinement();
}

// @sect4{MonodomainProblem::setup_system}
// The following function is, arguably, the most interesting one in the
// entire program since it goes to the heart of what distinguishes %parallel
// step-40 from sequential step-6.
//
// At the top we do what we always do: tell the DoFHandler object to
// distribute degrees of freedom. Since the triangulation we use here is
// distributed, the DoFHandler object is smart enough to recognize that on
// each processor it can only distribute degrees of freedom on cells it
// owns; this is followed by an exchange step in which processors tell each
// other about degrees of freedom on ghost cell. The result is a DoFHandler
// that knows about the degrees of freedom on locally owned cells and ghost
// cells (i.e. cells adjacent to locally owned cells) but nothing about
// cells that are further away, consistent with the basic philosophy of
// distributed computing that no processor can know everything.
template <int dim>
void MonodomainProblem<dim>::setup_system(const double       current_time,
                                          const unsigned int time_step) {
  TimerOutput::Scope t(computing_timer, "setup");

  dof_handler.distribute_dofs(fe);

  // The next two lines extract some information we will need later on,
  // namely two index sets that provide information about which degrees of
  // freedom are owned by the current processor (this information will be
  // used to initialize solution and right hand side vectors, and the system
  // matrix, indicating which elements to store on the current processor and
  // which to expect to be stored somewhere else); and an index set that
  // indicates which degrees of freedom are locally relevant (i.e. live on
  // cells that the current processor owns or on the layer of ghost cells
  // around the locally owned cells; we need all of these degrees of
  // freedom, for example, to estimate the error on the local cells).
  locally_owned_dofs = dof_handler.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

  // Next, let us initialize the solution and right hand side vectors. As
  // mentioned above, the solution vector we seek does not only store
  // elements we own, but also ghost entries; on the other hand, the right
  // hand side vector only needs to have the entries the current processor
  // owns since all we will ever do is write into it, never read from it on
  // locally owned cells (of course the linear solvers will read from it,
  // but they do not care about the geometric location of degrees of
  // freedom).
  locally_relevant_solution.reinit(
      std::vector<IndexSet>(2, locally_owned_dofs),    //
      std::vector<IndexSet>(2, locally_relevant_dofs), //
      mpi_communicator);
  diffusion_rhs.reinit(std::vector<IndexSet>(1, locally_owned_dofs), //
                       mpi_communicator);
  reaction_rhs.reinit(std::vector<IndexSet>(1, locally_owned_dofs), //
                      mpi_communicator);

  // The next step is to compute hanging node and boundary value
  // constraints, which we combine into a single object storing all
  // constraints.
  //
  // As with all other things in %parallel, the mantra must be that no
  // processor can store all information about the entire universe. As a
  // consequence, we need to tell the AffineConstraints object for which
  // degrees of freedom it can store constraints and for which it may not
  // expect any information to store. In our case, as explained in the
  // @ref distributed module, the degrees of freedom we need to care about on
  // each processor are the locally relevant ones, so we pass this to the
  // AffineConstraints::reinit function. As a side note, if you forget to
  // pass this argument, the AffineConstraints class will allocate an array
  // with length equal to the largest DoF index it has seen so far. For
  // processors with high MPI process number, this may be very large --
  // maybe on the order of billions. The program would then allocate more
  // memory than for likely all other operations combined for this single
  // array.
  constraints.clear();
  constraints.reinit(locally_relevant_dofs);

  // Hanging nodes are part of the mesh if adaptive refinement is
  // used. We keep the line below uncommented for now as it would not affect
  // the system at all for a globally refined triangulation.
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);

  if (parameters.boundary_condition_type == "Dirichlet") {
    VectorTools::interpolate_boundary_values(
        dof_handler,                                                //
        0,                                                          //
        PrescribedData::PotentialBoundaryValues<dim>(current_time), //
        constraints);
  }
  constraints.close();

  // The last part of this function deals with initializing the matrix with
  // accompanying sparsity pattern. As in previous tutorial programs, we use
  // the DynamicSparsityPattern as an intermediate with which we
  // then initialize the system matrix. To do so we have to tell the sparsity
  // pattern its size but as above there is no way the resulting object will
  // be able to store even a single pointer for each global degree of
  // freedom; the best we can hope for is that it stores information about
  // each locally relevant degree of freedom, i.e. all those that we may
  // ever touch in the process of assembling the matrix (the
  // @ref distributed_paper "distributed computing paper" has a long
  // discussion why one really needs the locally relevant, and not the small
  // set of locally active degrees of freedom in this context).
  //
  // So we tell the sparsity pattern its size and what DoFs to store
  // anything for and then ask DoFTools::make_sparsity_pattern to fill it
  // (this function ignores all cells that are not locally owned, mimicking
  // what we will do below in the assembly process). After this, we call a
  // function that exchanges entries in these sparsity pattern between
  // processors so that in the end each processor really knows about all the
  // entries that will exist in that part of the finite element matrix that
  // it will own. The final step is to initialize the matrix with the
  // sparsity pattern.
  DynamicSparsityPattern dsp(locally_relevant_dofs);

  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);

  SparsityTools::distribute_sparsity_pattern(
      dsp,                              //
      dof_handler.locally_owned_dofs(), //
      mpi_communicator,                 //
      locally_relevant_dofs);

  mass_matrix.reinit(locally_owned_dofs, //
                     locally_owned_dofs, //
                     dsp,                //
                     mpi_communicator);
  tissue_matrix.reinit(locally_owned_dofs, //
                       locally_owned_dofs, //
                       dsp,                //
                       mpi_communicator);  //
  system_matrix.reinit(locally_owned_dofs, //
                       locally_owned_dofs, //
                       dsp,                //
                       mpi_communicator);

  /*
     Allocate space for distributed vectors relating to the cell membrane model.
     Initialize the states to equilibrium values;
  */
  // allocate_all_cells(membrane_model, mpi_communicator, locally_owned_dofs,
  //                    membrane_state, membrane_rate, membrane_current, true);

  /*
    Setup the points to output (for i.e., reference solution comparison)
    TODO(krg)
    - generalize number of points
    - generalize to theta\neq 0
  */
  /*
  int n_eval_points = 101;
  evaluation_points.resize(n_eval_points);
  double ee=0.0;
  double de = (10.0 - 0.0)/(n_eval_points-1);
  std::generate(evaluation_points.begin(), evaluation_points.end(),
                [&ee, de]() {
                  auto rval = ee;
                  ee += de;
                  return Point<2>(rval, 0.0);
                });
  */

  // for(auto e : evaluation_points) {
  //   std::cout << e << std::endl;
  // }
  /*
    NOTE: if grid changes, this cache invalidates and must be reinit
  */
  // evaluation_cache.reinit(evaluation_points, triangulation, mapping);

  /**
  std::stringstream streamer;
  streamer << "eval_points_"                                  //
           << parameters.os_time_stepping_method << "_"       //
           << parameters.membrane_time_stepping_method << "_" //
           << parameters.tissue_time_stepping_method << "_"   //
           << parameters.n_time_steps << ".txt";
  eval_points_file = streamer.str();
  */

  // Get file ready for output
  /*
  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
    std::ofstream f(eval_points_file, std::ofstream::out);
    if (f) {
      f << "t x y v s\n";
    }
    f.close();
  }
  */

  if (time_step == 0) {
    pcout << "\n"
          << "\n"
          << "Relevant info:" << std::endl
          << "  Number of active cells:       "
          << triangulation.n_global_active_cells() << std::endl
          << "  Number of degrees of freedom: " << dof_handler.n_dofs() << "\n"
          << std::endl;
  }
}

// @sect4{MonodomainProblem::assemble_system}

// The function that then assembles the linear system is comparatively
// boring, being almost exactly what we've seen before. The points to watch
// out for are:
// - Assembly must only loop over locally owned cells. There
//   are multiple ways to test that; for example, we could compare a cell's
//   subdomain_id against information from the triangulation as in
//   <code>cell->subdomain_id() ==
//   triangulation.locally_owned_subdomain()</code>, or skip all cells for
//   which the condition <code>cell->is_ghost() ||
//   cell->is_artificial()</code> is true. The simplest way, however, is to
//   simply ask the cell whether it is owned by the local processor.
// - Copying local contributions into the global matrix must include
//   distributing constraints and boundary values. In other words, we cannot
//   (as we did in step-6) first copy every local contribution into the global
//   matrix and only in a later step take care of hanging node constraints and
//   boundary values. The reason is, as discussed in step-17, that the
//   parallel vector classes do not provide access to arbitrary elements of
//   the matrix once they have been assembled into it -- in parts because they
//   may simply no longer reside on the current processor but have instead
//   been shipped to a different machine.
// - The way we compute the right hand side (given the
//   formula stated in the introduction) may not be the most elegant but will
//   do for a program whose focus lies somewhere entirely different.
template <int dim> void MonodomainProblem<dim>::assemble_all_system_matrices() {
  TimerOutput::Scope t(computing_timer, "Assemble all matrices");

  mass_matrix   = 0.0;
  tissue_matrix = 0.0;

  const QGauss<dim> quadrature_formula(parameters.quadrature_order);

  FEValues<dim> fe_values(fe, quadrature_formula,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();

  FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_tissue_matrix(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  const PrescribedData::ConductivityTensor<dim> conductivity_tensor(parameters);
  std::vector<Tensor<2, dim>>                   conductivity_values(n_q_points);

  for (const auto& cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned()) {
      cell_tissue_matrix = 0.0;
      cell_mass_matrix   = 0.0;

      fe_values.reinit(cell);

      conductivity_tensor.value_list(fe_values.get_quadrature_points(),
                                     conductivity_values);

      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {

        for (unsigned int i = 0; i < dofs_per_cell; ++i) {
          for (unsigned int j = 0; j < dofs_per_cell; ++j) {
            // Mass matrix contribution from this cell
            cell_mass_matrix(i, j) += parameters.chi * parameters.Cm      //
                                      * fe_values.shape_value(i, q_point) //
                                      * fe_values.shape_value(j, q_point) //
                                      * fe_values.JxW(q_point);           //

            // cell_tissue_matrix contains the contribution from:
            //     -lambda/(1+lambda)*div(sigmai*grad(v))
            // TODO(krg):
            // - Note that Sebastian has an updated way for dealing with
            //   tensor conductivities
            cell_tissue_matrix(i, j) +=
                -parameters.lambda / (1 + parameters.lambda)       //
                * conductivity_values[q_point] //
                * fe_values.shape_grad(i, q_point)                 //
                * fe_values.shape_grad(j, q_point)                 //
                * fe_values.JxW(q_point);                          //
          }
        }
      }

      // Pack the localized entries into the global matrices,
      // respecting constraintes
      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(cell_mass_matrix,  //
                                             local_dof_indices, //
                                             mass_matrix);
      constraints.distribute_local_to_global(cell_tissue_matrix, //
                                             local_dof_indices,  //
                                             tissue_matrix);
    }

  // Notice that the assembling above is just a local operation. So, to
  // form the "global" linear system, a synchronization between all
  // processors is needed. This could be done by invoking the function
  // compress().
  // - See @ref GlossCompress "Compressing distributed objects" for more
  //   information on what is compress() designed to do.
  mass_matrix.compress(VectorOperation::add);
  tissue_matrix.compress(VectorOperation::add);
}

// TODO(krg): Implement use cases that work for nonzero FluxBoundaryValues
template <int dim>
void MonodomainProblem<dim>::assemble_rhs_tissue(
    const double /*current_time*/) {
  TimerOutput::Scope t(computing_timer, "Assemble RHS tissue");

  const QGauss<dim> quadrature_formula(parameters.quadrature_order);

  FEValues<dim> fe_values(fe, quadrature_formula,
                          update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();

  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  std::vector<double> rhs_values(n_q_points);

  for (const auto& cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned()) {
      cell_rhs = 0.0;

      // NO CONTRIBUTION (outside of constraints)

      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(cell_rhs, local_dof_indices,
                                             diffusion_rhs);
    }

  // Notice that the assembling above is just a local operation. So, to
  // form the "global" linear system, a synchronization between all
  // processors is needed. This could be done by invoking the function
  // compress().
  // - See @ref GlossCompress "Compressing distributed objects" for more
  //   information on what is compress() designed to do.
  diffusion_rhs.compress(VectorOperation::add);
}
// TODO: Implement use cases that work for nonzero FluxBoundaryValues
template <int dim>
void MonodomainProblem<dim>::assemble_rhs_membrane(
    const double                current_time, //
    const LA::MPI::BlockVector& y,            //
    LA::MPI::BlockVector&       out) {
  TimerOutput::Scope t(computing_timer, "Assemble RHS membrane");

  /* Ensure out vector has same structure as state evaluation */
  out.reinit(y);

  // Stimulus vector with the same structure as the transmembrane block
  // TODO(krg) move to class member to avoid reallocations
  LA::MPI::Vector i_stim;
  i_stim.reinit(y.block(0));

  // --------------------------------------------------------------------------
  //     Pack the stimulus vector
  //     ------------------------
  VectorTools::interpolate(
      dof_handler,                                             //
      PrescribedData::Stimulus<dim>(current_time, parameters), //
      i_stim);                                                 //

  // --------------------------------------------------------------------------
  //    Evaluate membrane model
  //    -------------------
  // Use the i_stim to evaluate the RHS of the whole model
  evaluate_y_derivatives(membrane_model, y, i_stim, out);
  out.compress(VectorOperation::insert);

}

// Setting the initial conditions of the solution
// Note: only works for transmembrane potential
// TODO(krg) Update to work with vector function on a block vector
template <int dim> class InitialValues : public Function<dim, double> {
public:
  InitialValues() : Function<dim, double>(1) {}

  virtual double value(const Point<dim>&  p,
                       const unsigned int component = 0) const override;
};
template <int dim>
double InitialValues<dim>::value(const Point<dim>&  /* p */,
                                 const unsigned int /* component */ ) const {


  double value = -1.2879118919372559;

  // TODO(krg) Add logic and/or f(x,y,z) for ICs here

  return value;
}

////////////////////////////////////////////////////////////////////////////
// Set the initial conditions in the `yinout` vector
template <int dim>
void MonodomainProblem<dim>::set_initial_conditions(
    LA::MPI::BlockVector& yinout) {

  const QGauss<dim> quadrature_formula(parameters.quadrature_order);

  FEValues<dim> fe_values(fe,                                                //
                          quadrature_formula,                                //
                          update_values | update_gradients |                 //
                              update_quadrature_points | update_JxW_values); //

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  const unsigned int n_q_points    = quadrature_formula.size();


  Vector<double> cell_val(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  // Set everything to the equilibrium
  equilibrium_state(membrane_model, yinout);
  yinout.compress(VectorOperation::insert);

  for (const auto& cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned()) {
      cell_val = 0.0;
      fe_values.reinit(cell);

      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
        for (unsigned int i = 0; i < dofs_per_cell; ++i) {

	  const Point<dim> p = fe_values.quadrature_point(q_point);

	  double value = 0.0;
          // Set a different value in small corner
          if (p[0] < 0.1) {
            if (p[1] < 0.1) {
              value = 1.0;
            }
          }
          // Stimulus contribution to current FEM cell
          cell_val(i) += value;
        }
      }
      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(cell_val, local_dof_indices,
                                             yinout.block(0));
    }
  yinout.compress(VectorOperation::add);
}

// @sect4{MonodomainProblem::solve}

// Even though solving linear systems on potentially tens of thousands of
// processors is by far not a trivial job, the function that does this is --
// at least at the outside -- relatively simple. Most of the parts you've
// seen before. There are really only two things worth mentioning:
// - Solvers and preconditioners are built on the deal.II wrappers of PETSc
//   and Trilinos functionality. It is relatively well known that the
//   primary bottleneck of massively %parallel linear solvers is not
//   actually the communication between processors, but the fact that it is
//   difficult to produce preconditioners that scale well to large numbers
//   of processors. Over the second half of the first decade of the 21st
//   century, it has become clear that algebraic multigrid (AMG) methods
//   turn out to be extremely efficient in this context, and we will use one
//   of them -- either the BoomerAMG implementation of the Hypre package
//   that can be interfaced to through PETSc, or a preconditioner provided
//   by ML, which is part of Trilinos -- for the current program. The rest
//   of the solver itself is boilerplate and has been shown before. Since
//   the linear system is symmetric and positive definite, we can use the CG
//   method as the outer solver.
// - Ultimately, we want a vector that stores not only the elements
//   of the solution for degrees of freedom the current processor owns, but
//   also all other locally relevant degrees of freedom. On the other hand,
//   the solver itself needs a vector that is uniquely split between
//   processors, without any overlap. We therefore create a vector at the
//   beginning of this function that has these properties, use it to solve the
//   linear system, and only assign it to the vector we want at the very
//   end. This last step ensures that all ghost elements are also copied as
//   necessary.

// Compute the solution of the linear system:
//     mass_matrix*rhs = stiffness_matrix*y + stiffness_rhs.
//
// Notes:
// - Should return the "local" rhs in order to be used with a TimeStepping
//   method.
// - Can also be used for an explicit TimeStepping method.
template <int dim>
void MonodomainProblem<dim>::solve_rhs_tissue(
    const double /* current_time */, //
    const LA::MPI::BlockVector& y,   //
    LA::MPI::BlockVector&       yout) {

  TimerOutput::Scope t(computing_timer, "Tissue RHS solve");

  // Construct the RHS of the linear system.
  // Diffusion part:
  LA::MPI::Vector tmp_rhs(locally_owned_dofs, mpi_communicator);
  tmp_rhs = 0.0;
  tissue_matrix.vmult(tmp_rhs, y.block(0));

  // TODO(krg) Add nonzero tissue RHS here (assemble_rhs_tissue)
  // - not needed for fully implicit diffusion
  // - maybe optional depending on particular membrane models?

  SolverControl solver_control(parameters.max_iterations_linear_solver *
                                   dof_handler.n_dofs(),
                               parameters.tol_linear_solver);
  LA::SolverCG solver(solver_control, mpi_communicator);
  LA::MPI::PreconditionAMG                 preconditioner;
  LA::MPI::PreconditionAMG::AdditionalData data;
  data.symmetric_operator = true;

  preconditioner.initialize(mass_matrix, data);
  solver.solve(mass_matrix, yout.block(0), tmp_rhs, preconditioner);
  // TODO: Add options for logging solve stats

  constraints.distribute(yout.block(0));
}

template <int dim>
void MonodomainProblem<dim>::solve_rhs_membrane(const double current_time,
                                                const LA::MPI::BlockVector& y,
                                                LA::MPI::BlockVector& yout) {
  TimerOutput::Scope t(computing_timer, "Membrane RHS eval");

  assemble_rhs_membrane(current_time, y, yout);

  // ---------------------------------------------------------------------
  //    Block 1: Recovery variable
  // for(unsigned int i=0;i< reaction_rhs.n_blocks(); ++i){
  //   // yout.block(i).add(1.0, reaction_rhs.block(i));
  //   std::swap(yout.block(i),reaction_rhs.block(i));
  // }

}

// Compute the solution of the linear system (Jacobian-solve):
//     (mass_matrix - tau*Jacobian)*lhs = mass_matrix*y,
//
// Note: Should return the "local" rhs in order to be used with a TimeStepping
//       method.
template <int dim>
void MonodomainProblem<dim>::solve_lhs_tissue(
    const double /* current_time */, //
    const double                tau, //
    const LA::MPI::BlockVector& y,   //
    LA::MPI::BlockVector&       yout) {

  TimerOutput::Scope t(computing_timer, "Tissue Jac solve");

  // Copy existing matrices correctly to the mat to be solved
  system_matrix.copy_from(tissue_matrix);
  system_matrix *= -tau;
  system_matrix.add(1.0, mass_matrix);

  // Construct the RHS
  LA::MPI::Vector tmp_rhs(locally_owned_dofs, mpi_communicator);
  mass_matrix.vmult(tmp_rhs, y.block(0));

  // Solve into a locally-owned vector
  // LA::MPI::Vector tmp_locally_owned_lhs(locally_owned_dofs,
  // mpi_communicator);
  SolverControl solver_control(parameters.max_iterations_linear_solver //
                                   * dof_handler.n_dofs(),             //
                               parameters.tol_linear_solver);

  LA::SolverCG solver(solver_control, mpi_communicator);
  LA::MPI::PreconditionAMG                 preconditioner;
  LA::MPI::PreconditionAMG::AdditionalData data;
  data.symmetric_operator = true;

  preconditioner.initialize(system_matrix, data);
  solver.solve(system_matrix, yout.block(0), tmp_rhs, preconditioner);
  // TODO: Add options for logging solve stats

  constraints.distribute(yout);
}


// Implicit time stepping to evolve the dynamics of the model from initial time
// to final time.
//
// We make use of the OperatorSplit class that provides different
// operator splitting methods for time integration.
//
template <int dim>
void MonodomainProblem<dim>::os_time_stepping(
    const os_t         os_method,       //
    const ts_mem_t     membrane_method, //
    const ts_tis_t     tissue_method,   //
    const unsigned int n_time_steps,    //
    const double       initial_time,    //
    const double       final_time) {

  const double time_step_size =
      (final_time - initial_time) / static_cast<double>(n_time_steps);
  double time = initial_time;

  tostii::ExplicitRungeKutta<LA::MPI::BlockVector> membrane_ts(membrane_method);
  tostii::ImplicitRungeKutta<LA::MPI::BlockVector> tissue_ts(tissue_method);

  /*
    The function calls need to pull in all of the class data with them to run
    properly
    -> wrap them in a closure that capture 'this'
   */
  using os_op_t = tostii::OSoperator<LA::MPI::BlockVector>;
  os_op_t stepper_membrane{
      &membrane_ts,
      [this](const double                time, //
             const LA::MPI::BlockVector& y,    //
             LA::MPI::BlockVector&       out) {
        this->solve_rhs_membrane(time, y, out);
      },
      [this](const double /*time*/,             //
             const double /*tau*/,              //
             const LA::MPI::BlockVector& /*y*/, //
             LA::MPI::BlockVector& /*out*/) {
        return; // dummy function for explicit integration
      },
  };
  os_op_t stepper_tissue{
      &tissue_ts,                              //
      [this](const double                time, //
             const LA::MPI::BlockVector& y,    //
             LA::MPI::BlockVector&       out) {
        this->solve_rhs_tissue(time, y, out);
      },
      [this](const double                time, //
             const double                tau,  //
             const LA::MPI::BlockVector& y,    //
             LA::MPI::BlockVector&       out) {
        this->solve_lhs_tissue(time, tau, y, out);
      },
  };

  // Determine frequency of pvtu output
  const unsigned int mod_pvtu_files =
      static_cast<unsigned int>(n_time_steps / parameters.n_pvtu_files);
  // Determine frequency of info messages
  const unsigned int mod_info_messages =
      static_cast<unsigned int>(n_time_steps / parameters.n_info_messages);

  // TimeStepping methods work with locally-owned vectors
  LA::MPI::BlockVector tmp_locally_owned_solution(
      std::vector<IndexSet>(2, locally_owned_dofs), //
      mpi_communicator);                            //

  // Determine FEM representation of the initial conditions
  // Sebastian notes that this project method doesn't work well in the Bidomain...
  /// ... maybe due to block vectors?

  // VectorTools::project(dof_handler,                                      //
  //                      constraints,                                      //
  //                      QGauss<dim>(parameters.quadrature_order),         //
  //                      PrescribedData::ExactSolution<dim>(initial_time), //
  //                      tmp_locally_owned_solution);

  // constraints.distribute(tmp_locally_owned_solution);



  //TODO(krg) Apply ICs here!
  equilibrium_state(membrane_model, tmp_locally_owned_solution);
  tmp_locally_owned_solution.compress(VectorOperation::insert);
  locally_relevant_solution.reinit(tmp_locally_owned_solution);
  equilibrium_state(membrane_model, locally_relevant_solution);
  locally_relevant_solution.compress(VectorOperation::insert);

  // Apply V initial conditions
  // VectorTools::interpolate(dof_handler, InitialValues<dim>(), locally_relevant_solution.block(0));
  // VectorTools::interpolate(dof_handler, InitialValues<dim>(), tmp_locally_owned_solution.block(0));


  // Output discretized initial conditions
  if (parameters.output_pvtu_files == true)
    output_results(0,0.0);

  std::vector<tostii::OSmask> os_mask{
      {0, 1}, // Membrane block vars
      {0}     // Tissue block vars
  };          //
  using os_ops_t = std::vector<os_op_t>;
  os_ops_t os_operators{
      stepper_membrane, //
      stepper_tissue  //
  };

  tostii::OperatorSplit<LA::MPI::BlockVector> os_stepper(
      locally_relevant_solution, // global system BlockVector for reference
      os_operators, // Splitting of the global system - MEMBRANE FIRST
      os_method,    // Method
      os_mask       // Mask for each operator
  );

  // Main time loop:
  // - take timestep
  // - apply constraints
  // - output (at determined frequencies)
  for (unsigned int time_step = 0; time_step < n_time_steps; ++time_step) {

    // TODO: Fix adaptive refinement
    if (time_step > 0 && parameters.adaptive_refinement == true) {
      refine_grid();
      setup_system(time, time_step);
    }

    time = os_stepper.evolve_one_time_step(time, time_step_size,
                                           tmp_locally_owned_solution);
    constraints.distribute(tmp_locally_owned_solution);

    if ((time_step + 1) % mod_info_messages == 0 &&
        parameters.output_solution_error_data == true) {
      pcout << "Time step " << time_step + 1 << " / " << n_time_steps
            << " at time t = "
            // << add more info if needed
            << time << std::endl;
    }

    if (((time_step + 1) % mod_pvtu_files == 0) &&
        parameters.output_pvtu_files == true) {
      // Ghosted local vector is needed for producing pvtu output
      locally_relevant_solution = tmp_locally_owned_solution;
      TimerOutput::Scope t(computing_timer, "output");
      output_results(time_step + 1, time);
    }
  }

  // get ready for next time step
  locally_relevant_solution = tmp_locally_owned_solution;
}

// Time integration of the model from initial time to final time
template <int dim> void MonodomainProblem<dim>::time_integrate() {

  tostii::runge_kutta_method tissue_ts_method =
      tostii::RK_string_to_enum(parameters.tissue_time_stepping_method);
  tostii::runge_kutta_method membrane_ts_method =
      tostii::RK_string_to_enum(parameters.membrane_time_stepping_method);

  /* Define a small dictionary of possible OS methods */
  // using os_t         = std::vector<os_pair_t>;
  using os_methods_t = std::unordered_map<std::string, os_t>;

  os_methods_t os_methods;

  // TODO: Move this map to the TimeStepping namespace
  os_methods["SINGLE"]  = os_t{os_pair_t{0, 1.0}}; //
  os_methods["GODUNOV"] = os_t{os_pair_t{0, 1.0},  //
                               os_pair_t{1, 1.0}};
  os_methods["STRANG"]  = os_t{os_pair_t{0, 0.5},  //
                              os_pair_t{1, 1.0},  //
                              os_pair_t{0, 0.5}};
  os_methods["STRANG2"] = os_t{os_pair_t{0, 0.5},  //
                               os_pair_t{1, 0.5},  //
                               os_pair_t{1, 0.5},  //
			       os_pair_t{0, 0.5}};  //
  os_methods["BEST2-2"] = os_t{os_pair_t{0, 1.0 - std::sqrt(2.0) / 2.0}, //
                               os_pair_t{1, std::sqrt(2.0) / 2.0},       //
                               os_pair_t{0, std::sqrt(2.0) / 2.0}, //
                               os_pair_t{1, 1.0 - std::sqrt(2.0) / 2.0}};
  os_methods["RUTH"]    = os_t{os_pair_t{0, 7.0 / 24.0},  //
                            os_pair_t{1, 2.0 / 3.0},   //
                            os_pair_t{0, 3.0 / 4.0},   //
                            os_pair_t{1, -2.0 / 3.0},  //
                            os_pair_t{0, -1.0 / 24.0}, //
                            os_pair_t{1, 1.0}};

  // Emb 4/3 AK p pair
  double a[5]              = {0.125962888700250514,  //
                              0.751193431379145450,  //
                              0.127551831557005609,  //
                              -0.338296598434303506, //
                              0.333588446797901933};
  os_methods["Emb_3_AK_p"] = os_t{os_pair_t{0, a[0]},                  //
                                  os_pair_t{1, a[4]},                  //
                                  os_pair_t{0, a[1]},                  //
                                  os_pair_t{1, a[3]},                  //
                                  os_pair_t{1, 0.261153550449697153},  //
                                  os_pair_t{0, -0.242703571757396124}, //
                                  os_pair_t{1, 0.596114052266110425},  //
                                  os_pair_t{0, 0.365547251678000160},  //
                                  os_pair_t{1, 0.147440548920593995}};
  os_methods["Emb_4_AK_p"] = os_t{os_pair_t{0, a[0]},  //
                                  os_pair_t{1, a[4]},  //
                                  os_pair_t{0, a[1]},  //
                                  os_pair_t{1, a[3]},  //
                                  os_pair_t{0, a[2]},  //
                                  os_pair_t{1, a[2]},  //
                                  os_pair_t{0, a[3]},  //
                                  os_pair_t{1, a[1]},  //
                                  os_pair_t{0, a[4]},  //
                                  os_pair_t{1, a[0]}}; //

  os_t os_ts_method = os_methods.at(parameters.os_time_stepping_method);

  // Call through to the operator splitting time integration
  os_time_stepping(os_ts_method,            //
                   membrane_ts_method,      //
                   tissue_ts_method,        //
                   parameters.n_time_steps, //
                   parameters.initial_time, //
                   parameters.final_time);  //
  }

// @sect4{MonodomainProblem::output_results}

// Compared to the corresponding function in step-6, the one here is a tad
// more complicated. There are two reasons: the first one is that we do not
// just want to output the solution but also for each cell which processor
// owns it (i.e. which "subdomain" it is in). Secondly, as discussed at
// length in step-17 and step-18, generating graphical data can be a
// bottleneck in parallelizing. In step-18, we have moved this step out of
// the actual computation but shifted it into a separate program that later
// combined the output from various processors into a single file. But this
// doesn't scale: if the number of processors is large, this may mean that
// the step of combining data on a single processor later becomes the
// longest running part of the program, or it may produce a file that's so
// large that it can't be visualized any more. We here follow a more
// sensible approach, namely creating individual files for each MPI process
// and leaving it to the visualization program to make sense of that.
//
// To start, the top of the function looks like it usually does. In addition
// to attaching the solution vector (the one that has entries for all locally
// relevant, not only the locally owned, elements), we attach a data vector
// that stores, for each cell, the subdomain the cell belongs to. This is
// slightly tricky, because of course not every processor knows about every
// cell. The vector we attach therefore has an entry for every cell that the
// current processor has in its mesh (locally owned ones, ghost cells, and
// artificial cells), but the DataOut class will ignore all entries that
// correspond to cells that are not owned by the current processor. As a
// consequence, it doesn't actually matter what values we write into these
// vector entries: we simply fill the entire vector with the number of the
// current MPI process (i.e. the subdomain_id of the current process); this
// correctly sets the values we care for, i.e. the entries that correspond
// to locally owned cells, while providing the wrong value for all other
// elements -- but these are then ignored anyway.
template <int dim>
void MonodomainProblem<dim>::output_results(unsigned int time_step, double /*current_time*/) const {
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(locally_relevant_solution.block(0), "v");
  for(unsigned int i=1;i < locally_relevant_solution.n_blocks(); ++i) {
    data_out.add_data_vector(locally_relevant_solution.block(i), "s");
  }

  Vector<float> subdomain(triangulation.n_active_cells());
  for (unsigned int i = 0; i < subdomain.size(); ++i)
    subdomain(i) = triangulation.locally_owned_subdomain();
  data_out.add_data_vector(subdomain, "subdomain");

  data_out.build_patches();

  // The next step is to write this data to disk. We write up to 8 VTU files
  // in parallel with the help of MPI-IO. Additionally a PVTU record is
  // generated, which groups the written VTU files.

  std::stringstream file_base;
  file_base << "solution_"                                     //
            << parameters.n_time_steps << "_"                  //
            << parameters.os_time_stepping_method << "_"       //
            << parameters.membrane_time_stepping_method << "_" //
            << parameters.tissue_time_stepping_method;

  data_out.write_vtu_with_pvtu_record(
      "./",             // Root directory for output.
      file_base.str(),  // Base name for files.
      time_step,        // Number for file output pattern.
      mpi_communicator, // Communicator on which the data is defined.
      3,                // ??
      8);               // ?? Max number of sub-files written per output.

  /*
    Evaluate at points, and write to file
  */
  /*
  auto vout =
      VectorTools::point_values<1, dim>(evaluation_cache, //
                                        dof_handler,      //
                                        locally_relevant_solution.block(0));
  auto sout =
      VectorTools::point_values<1, dim>(evaluation_cache, //
                                        dof_handler,      //
                                        locally_relevant_solution.block(1));
  */

  /*
  std::ofstream f;
  if(Utilities::MPI::this_mpi_process(mpi_communicator) == 0) {
    f.open(eval_points_file, std::ofstream::out | std::ofstream::app);
    auto v = vout.begin();
    auto s = sout.begin();
    for(auto& p : evaluation_points) {
      auto x = p[0];
      auto y = p[1];
      f << std::setprecision(12)                       //
        << std::fixed                                  //
        << current_time << " " << x << " " << y << " " //
        << v[0] << " " << s[0]                         //
        << "\n";
      ++v;
      ++s;
    }
    f.close();
  }
  */
}

// @sect4{MonodomainProblem::run}

// The function that controls the overall behavior of the program is again
// like the one in step-6. The minor difference are the use of
// <code>pcout</code> instead of <code>std::cout</code> for output to the
// console (see also step-17) and that we only generate graphical output if
// at most 32 processors are involved. Without this limit, it would be just
// too easy for people carelessly running this program without reading it
// first to bring down the cluster interconnect and fill any file system
// available :-)
//
// A functional difference to step-6 is the use of a square domain and that
// we start with a slightly finer mesh (5 global refinement cycles) -- there
// just isn't much of a point showing a massively %parallel program starting
// on 4 cells (although admittedly the point is only slightly stronger
// starting on 1024).
template <int dim> void MonodomainProblem<dim>::run() {
  pcout << "==========================================\n"
        << "Monodomain problem:\n"
        << "  Dimensions: " << parameters.dim << "\n"
        << "  MPI rank(s): "
        << Utilities::MPI::n_mpi_processes(mpi_communicator) << "\n"
        << "  LinearSolver lib: "
#ifdef USE_PETSC_LA
        << "PETSc\n"
#else
        << "Trilinos\n"
#endif
    // TODO(krg) Add membrane parameters, decide how best to init them
        << "  Membrane model: FitzHugh--Nagumo\n"
        << "  Spatial discretization:\n"
        << "    Lagrange element degree: " << parameters.polynomial_degree
        << "\n  Time stepping:\n"
        << "    OS TS      : " << parameters.os_time_stepping_method
        << "\n      Tissue TS  : " << parameters.tissue_time_stepping_method
        << "\n      Membrane TS: " << parameters.membrane_time_stepping_method
        << "\n"
        << std::endl;

  // Create the grid, setup the system and assemble the matrices involved in the
  // calculation. These do not depend on time (so far) so we only need to call
  // them once, before the time stepping occurs.
  // TODO: Consider time-dependent assembly
  make_grid();
  setup_system(0.0, 0);
  assemble_all_system_matrices();

  time_integrate();

  pcout << std::endl;
}
} // namespace Monodomain

// @sect4{main()}

// The final function, <code>main()</code>, again has the same structure as in
// all other programs, in particular step-6. Like the other programs that use
// MPI, we have to initialize and finalize MPI, which is done using the helper
// object Utilities::MPI::MPI_InitFinalize. The constructor of that class also
// initializes libraries that depend on MPI, such as p4est, PETSc, SLEPc, and
// Zoltan (though the last two are not used in this tutorial). The order here
// is important: we cannot use any of these libraries until they are
// initialized, so it does not make sense to do anything before creating an
// instance of Utilities::MPI::MPI_InitFinalize.
//
// After the solver finishes, the MonodomainProblem destructor will run followed
// by Utilities::MPI::MPI_InitFinalize::~MPI_InitFinalize(). This order is
// also important: Utilities::MPI::MPI_InitFinalize::~MPI_InitFinalize() calls
// <code>PetscFinalize</code> (and finalization functions for other
// libraries), which will delete any in-use PETSc objects. This must be done
// after we destruct the Laplace solver to avoid double deletion
// errors. Fortunately, due to the order of destructor call rules of C++, we
// do not need to worry about any of this: everything happens in the correct
// order (i.e., the reverse of the order of construction). The last function
// called by Utilities::MPI::MPI_InitFinalize::~MPI_InitFinalize() is
// <code>MPI_Finalize</code>: i.e., once this object is destructed the program
// should exit since MPI will no longer be available.


int main(int argc, char* argv[]) {
  try {
    using namespace dealii;
    using namespace Parameters;
    using namespace Monodomain;

    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    // Read parameters
    Parameters::AllParameters parameters("./monodomain_parameters.prm");

    // ONLY work for 2d for now
    Assert(parameters.dim == 2,                                         //
           ExcMessage("Monodomain problem is implemented to work in 2 " //
                      "dimensions only."));

    MonodomainProblem<2> monodomain_problem(parameters);
    monodomain_problem.run();

  } catch (std::exception& exc) {
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
  } catch (...) {
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
