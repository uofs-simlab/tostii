/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2014 - 2021 by the deal.II authors
 *
 * This file is part of the deal.II library.
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
 * Authors: Kevin R. Green, University of Saskatchewan
 *
 * Modification of step-52 to use tost.II time integration
 * - Comments have generally been kept intact from the original, with
     comments added that are specifically relevant to the OperatorSplit
     integration.
   - See https://www.dealii.org/current/doxygen/deal.II/step_52.html for
     the original code that solved this problem.
 */

// @sect3{Include files}

// deal.ii functionality
#include <deal.II/base/discrete_time.h>  // DisreteTime
#include <deal.II/base/function.h>       // Functions::ZeroFunction<N>
#include <deal.II/base/quadrature_lib.h> // QGauss<N>

#include <deal.II/grid/grid_generator.h> // GridGenerator::hyper_cube
#include <deal.II/grid/tria.h>           // Triangulation<N>

#include <deal.II/dofs/dof_handler.h> // DoFHandler<N>
#include <deal.II/dofs/dof_tools.h>   // DoFTools::make_sparsity_pattern

#include <deal.II/fe/fe_q.h>      // FE_Q<N>
#include <deal.II/fe/fe_values.h> // FEValues<N>

#include <deal.II/lac/affine_constraints.h> // AffineConstraints<T>
#include <deal.II/lac/sparse_direct.h>      // SpareDirectUMFPack, SparseMatrix<T>

#include <deal.II/numerics/data_out.h>     // DataOut<N>
#include <deal.II/numerics/vector_tools.h> // VectorTools::interpolate_boundary_values

// C++ stdlib functionality
#include <cmath>    // std::sin, std::cos
#include <fstream>  // std::ofstream
#include <iostream> // std::cout, std::cerr, std::endl

// Include our separated time-integration library
// (has some overlap with deal.II/base/time_stepping.h)
#include "tostii.h"

namespace Neutron {
using namespace dealii;

// @sect3{The <code>Diffusion</code> class}

// The next piece is the declaration of the main class. Most of the
// functions in this class are not new and have been explained in previous
// tutorials. The only interesting functions are
// <code>evaluate_diffusion()</code> and
// <code>id_minus_tau_J_inverse()</code>. <code>evaluate_diffusion()</code>
// evaluates the diffusion equation, $M^{-1}(f(t,y))$, at a given time and a
// given $y$. <code>id_minus_tau_J_inverse()</code> evaluates $\left(I-\tau
// M^{-1} \frac{\partial f(t,y)}{\partial y}\right)^{-1}$ or equivalently
// $\left(M-\tau \frac{\partial f}{\partial y}\right)^{-1} M$ at a given
// time, for a given $\tau$ and $y$. This function is needed when an
// implicit method is used.
class Diffusion {
public:
  Diffusion(int argc, char* argv[]);

  void run();

private:
  using bvector_t = BlockVector<double>;

  void setup_system();

  void assemble_system();

  double get_source(const double time, const Point<2>& point) const;

  void evaluate_diffusion(const double     time, //
                          const bvector_t& yin,  //
                          bvector_t&       yout) const;

  void evaluate_reaction(const double     time, //
                         const bvector_t& yin,  //
                         bvector_t&       yout) const;
  void id_minus_tau_Jdiffusion_inverse(const double     time, //
                                       const double     tau,  //
                                       const bvector_t& yin,  //
                                       bvector_t&       yout);
  void id_minus_tau_Jreaction_inverse(const double     time, //
                                      const double     tau,  //
                                      const bvector_t& yin,  //
                                      bvector_t&       yout);

  void output_results(const double       time,      //
                      const unsigned int time_step, //
                      std::string        os_method, //
                      tostii::runge_kutta_method method0,   //
                      tostii::runge_kutta_method method1    //
  ) const;

  // This is the main method for running the OS-specific code.
  // It is templated on the integrator types (implicit vs explicit)
  template <typename TS1, typename TS2>
  void os_method_run(const std::string                 os_method_name, //
                     const std::vector<tostii::OSpair<double>> stages,         //
                     const tostii::runge_kutta_method          method0,        //
                     const tostii::runge_kutta_method          method1,        //
                     const unsigned int                n_time_steps,   //
                     const double                      initial_time,   //
                     const double                      final_time);

  const unsigned int fe_degree;

  const double diffusion_coefficient;
  const double absorption_cross_section;

  Triangulation<2> triangulation;

  const FE_Q<2> fe;

  DoFHandler<2> dof_handler;

  AffineConstraints<double> constraint_matrix;

  SparsityPattern sparsity_pattern;

  SparseMatrix<double> system_matrix;
  SparseMatrix<double> diffusion_matrix;
  SparseMatrix<double> reaction_matrix;

  SparseMatrix<double> mass_matrix;
  SparseMatrix<double> mass_minus_tau_Jacobian;

  SparseDirectUMFPACK inverse_mass_matrix;

  bvector_t solution;
};

// We choose quadratic finite elements and we initialize the parameters.
Diffusion::Diffusion(int /*argc*/, char** /*argv*/)
    : fe_degree(2),                    //
      diffusion_coefficient(1. / 30.), //
      absorption_cross_section(1.),    //
      fe(fe_degree),                   //
      dof_handler(triangulation)       //
{}

// @sect4{<code>Diffusion::setup_system</code>}
// Now, we create the constraint matrix and the sparsity pattern. Then, we
// initialize the matrices and the solution vector.
void Diffusion::setup_system() {
  dof_handler.distribute_dofs(fe);

  VectorTools::interpolate_boundary_values(
      dof_handler, 1, Functions::ZeroFunction<2>(), constraint_matrix);
  constraint_matrix.close();

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraint_matrix);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);
  diffusion_matrix.reinit(sparsity_pattern);
  reaction_matrix.reinit(sparsity_pattern);
  mass_matrix.reinit(sparsity_pattern);
  mass_minus_tau_Jacobian.reinit(sparsity_pattern);
  solution.reinit(1, dof_handler.n_dofs());
}

// @sect4{<code>Diffusion::assemble_system</code>}
// In this function, we compute $-\int D \nabla b_i \cdot \nabla b_j
// d\boldsymbol{r} - \int \Sigma_a b_i b_j d\boldsymbol{r}$ and the mass
// matrix $\int b_i b_j d\boldsymbol{r}$. The mass matrix is then
// inverted using a direct solver; the <code>inverse_mass_matrix</code>
// variable will then store the inverse of the mass matrix so that
// $M^{-1}$ can be applied to a vector using the <code>vmult()</code>
// function of that object. (Internally, UMFPACK does not really store
// the inverse of the matrix, but its LU factors; applying the inverse
// matrix is then equivalent to doing one forward and one backward solves
// with these two factors, which has the same complexity as applying an
// explicit inverse of the matrix).
void Diffusion::assemble_system() {
  diffusion_matrix = 0.;
  reaction_matrix  = 0.;
  mass_matrix      = 0.;

  const QGauss<2> quadrature_formula(fe_degree + 1);

  FEValues<2> fe_values(fe,                 //
                        quadrature_formula, //
                        update_values | update_gradients | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  const unsigned int n_q_points    = quadrature_formula.size();

  // assembly is for both the diffusion and the reaction matrices
  // only at the beginning (since this is a linear problem)
  FullMatrix<double> dif_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> rea_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto& cell : dof_handler.active_cell_iterators()) {

    dif_matrix       = 0.;
    rea_matrix       = 0.;
    cell_mass_matrix = 0.;

    fe_values.reinit(cell);

    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
        for (unsigned int j = 0; j < dofs_per_cell; ++j) {
          dif_matrix(i, j) +=                     //
              (-diffusion_coefficient *           // (-D
               fe_values.shape_grad(i, q_point) * //  * grad phi_i
               fe_values.shape_grad(j, q_point)   //  * grad phi_j
               * fe_values.JxW(q_point));         //  * dx)

          rea_matrix(i, j) +=                      //
              (-absorption_cross_section *         //  (-Sigma
               fe_values.shape_value(i, q_point) * //   * phi_i
               fe_values.shape_value(j, q_point)   //   * phi_j
               * fe_values.JxW(q_point));          //   * dx)

          cell_mass_matrix(i, j) += fe_values.shape_value(i, q_point) * // phi_i
                                    fe_values.shape_value(j, q_point) * // phi_j
                                    fe_values.JxW(q_point);             // dx
        }
      }
    }

    cell->get_dof_indices(local_dof_indices);

    constraint_matrix.distribute_local_to_global(dif_matrix,        //
                                                 local_dof_indices, //
                                                 diffusion_matrix);
    constraint_matrix.distribute_local_to_global(rea_matrix,        //
                                                 local_dof_indices, //
                                                 reaction_matrix);
    constraint_matrix.distribute_local_to_global(cell_mass_matrix,  //
                                                 local_dof_indices, //
                                                 mass_matrix);
  }

  inverse_mass_matrix.initialize(mass_matrix);
}

// @sect4{<code>Diffusion::get_source</code>}
//
// In this function, the source term of the equation for a given time and a
// given point is computed.
double Diffusion::get_source(const double time, const Point<2>& point) const {
  const double intensity = 10.;
  const double frequency = numbers::PI / 10.;
  const double b         = 5.;
  const double x         = point(0);

  return intensity * (frequency * std::cos(frequency * time) * (b * x - x * x) +
                      std::sin(frequency * time) *
                          (absorption_cross_section * (b * x - x * x) +
                           2. * diffusion_coefficient));
}

// @sect4{<code>Diffusion::evaluate_diffusion</code>}
//
// Next, we evaluate the weak form of the diffusion equation at a
// given time $t$ and for a given vector $y$. In other words, as
// outlined in the introduction, we evaluate $M^{-1}(-{\cal D}y)$ and
// $M^{-1}(-{\cal A}y + {\cal S})$. For this, we have to apply the
// matrices $-{\cal D}$ and $-{\cal A}$ (previously computed and
// stored in the variables <code>diffusion_matrix</code>, and
// <code>reaction_matrix</code>) to $y$ and then add the source term
// to the reaction which we integrate as we usually do. The results
// are then multiplied by $M^{-1}$. This is split up to separate
// routines for the diffusion and the reaction components.
void Diffusion::evaluate_diffusion(const double /* time */, //
                                   const bvector_t& yin,    //
                                   bvector_t&       yout) const {
  Vector<double> tmp(dof_handler.n_dofs());
  tmp = 0.;
  diffusion_matrix.vmult(tmp, yin.block(0));

  yout.reinit(1, dof_handler.n_dofs());
  inverse_mass_matrix.vmult(yout.block(0), tmp);
}

void Diffusion::evaluate_reaction(const double     time, //
                                  const bvector_t& yin,  //
                                  bvector_t&       yout) const {
  Vector<double> tmp(dof_handler.n_dofs());
  tmp = 0.;
  reaction_matrix.vmult(tmp, yin.block(0));

  const QGauss<2> quadrature_formula(fe_degree + 1);

  FEValues<2> fe_values(fe, quadrature_formula,
                        update_values | update_quadrature_points |
                            update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  const unsigned int n_q_points    = quadrature_formula.size();

  Vector<double> cell_source(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  for (const auto& cell : dof_handler.active_cell_iterators()) {
    cell_source = 0.;

    fe_values.reinit(cell);

    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {

      const double source =
          get_source(time, fe_values.quadrature_point(q_point));
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        cell_source(i) += fe_values.shape_value(i, q_point) * // phi_i(x)
                          source *                            // * S(x)
                          fe_values.JxW(q_point) / 2.0;       // * dx
    }

    cell->get_dof_indices(local_dof_indices);

    constraint_matrix.distribute_local_to_global(cell_source, local_dof_indices,
                                                 tmp);
  }

  yout.reinit(1, dof_handler.n_dofs());
  inverse_mass_matrix.vmult(yout.block(0), tmp);
}

// @sect4{<code>Diffusion::id_minus_tau_Jdiffusion_inverse</code>}
//
// We compute $\left(M-\tau D\right)^{-1} M$. This is done in several steps:
//   - compute $M-\tau D$
//   - invert the matrix to get $\left(M-\tau D\right)^{-1}$
//   - compute $tmp=My$
//   - compute $z=\left(M-\tau D\right)^{-1} tmp =
//   \left(M-\tau D\right)^{-1} My$
//   - return z.
// A similar process is done for the reaction Jacobian in the
// Jreaction routine.
void Diffusion::id_minus_tau_Jdiffusion_inverse(const double /*time*/, //
                                                const double     tau,  //
                                                const bvector_t& yin,  //
                                                bvector_t&       yout) {
  SparseDirectUMFPACK inverse_mass_minus_tau_Jacobian;

  mass_minus_tau_Jacobian.copy_from(mass_matrix);
  mass_minus_tau_Jacobian.add(-tau, diffusion_matrix);

  inverse_mass_minus_tau_Jacobian.initialize(mass_minus_tau_Jacobian);

  Vector<double> tmp(dof_handler.n_dofs());
  mass_matrix.vmult(tmp, yin.block(0));

  yout.reinit(yin);
  inverse_mass_minus_tau_Jacobian.vmult(yout.block(0), tmp);
}
void Diffusion::id_minus_tau_Jreaction_inverse(const double /*time*/, //
                                               const double     tau,  //
                                               const bvector_t& yin,  //
                                               bvector_t&       yout) {
  SparseDirectUMFPACK inverse_mass_minus_tau_Jacobian;

  mass_minus_tau_Jacobian.copy_from(mass_matrix);
  mass_minus_tau_Jacobian.add(-tau, reaction_matrix);

  inverse_mass_minus_tau_Jacobian.initialize(mass_minus_tau_Jacobian);

  Vector<double> tmp(dof_handler.n_dofs());
  mass_matrix.vmult(tmp, yin.block(0));

  yout.reinit(yin);
  inverse_mass_minus_tau_Jacobian.vmult(yout.block(0), tmp);
}

// @sect4{<code>Diffusion::output_results</code>}
//
// The following function outputs the solution in vtu files indexed by the
// number of the time step and the name of the time stepping method.
void Diffusion::output_results(const double               time,           //
                               const unsigned int         time_step,      //
                               std::string                os_method_name, //
                               tostii::runge_kutta_method method0,        //
                               tostii::runge_kutta_method method1) const {

  std::string method0_name{RK_method_enum_to_string(method0)};
  std::string method1_name{RK_method_enum_to_string(method1)};

  DataOut<2> data_out;

  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "phi");

  data_out.build_patches();

  data_out.set_flags(DataOutBase::VtkFlags(time, time_step));

  const std::string output_dir{"output/"};

  const std::string filename = output_dir + os_method_name + "_" +
                               method0_name + "_" + method1_name + "-" +
                               Utilities::int_to_string(time_step, 3) + ".vtu";
  std::ofstream output(filename);
  data_out.write_vtu(output);

  static std::vector<std::pair<double, std::string>> times_and_names;

  static std::string method_name_prev = "";
  static std::string pvd_filename;
  if (method_name_prev != method0_name) {
    times_and_names.clear();
    method_name_prev = method0_name;
    pvd_filename     = output_dir + os_method_name + "_" + method0_name + "_" +
                   method1_name + ".pvd";
  }
  times_and_names.emplace_back(time, filename);
  std::ofstream pvd_output(pvd_filename);
  DataOutBase::write_pvd_record(pvd_output, times_and_names);
}

// @sect4{<code>Diffusion::os_method_run</code>}
//
// Now on to the main part where the Operator split integration
// happens.
template <typename TS1, typename TS2>
void Diffusion::os_method_run(
    const std::string                         os_method_name, //
    const std::vector<tostii::OSpair<double>> stages,         //
    const tostii::runge_kutta_method          method0,        //
    const tostii::runge_kutta_method          method1,        //
    const unsigned int                        n_time_steps,   //
    const double                              initial_time,   //
    const double                              final_time) {
  const double time_step =
      (final_time - initial_time) / static_cast<double>(n_time_steps);

  solution = 0.;
  constraint_matrix.distribute(solution);

  TS1 runge_kutta_diff(method0);
  TS2 runge_kutta_reac(method1);

  /* Define OSoperators to use in the operator split stepper */
  tostii::OSoperator<bvector_t> diffusion_stepper{
      &runge_kutta_diff,       //
      [this](const double     time, //
             const bvector_t& yin,  //
             bvector_t& yout) { this->evaluate_diffusion(time, yin, yout); },
      [this](const double     time, //
             const double     tau,  //
             const bvector_t& yin,  //
             bvector_t&       yout) {
        this->id_minus_tau_Jdiffusion_inverse(time, tau, yin, yout);
      },
  };
  tostii::OSoperator<bvector_t> reaction_stepper{
      &runge_kutta_reac,       //
      [this](const double     time, //
             const bvector_t& yin,  //
             bvector_t& yout) { this->evaluate_reaction(time, yin, yout); },
      [this](const double     time, //
             const double     tau,  //
             const bvector_t& yin,  //
             bvector_t&       yout) {
        this->id_minus_tau_Jreaction_inverse(time, tau, yin, yout);
      },
  };

  std::vector<tostii::OSmask>      os_mask{{0}, {0}};
  tostii::OperatorSplit<bvector_t> os_stepper(
      solution, //
      std::vector<tostii::OSoperator<bvector_t>>{diffusion_stepper,
                                                 reaction_stepper}, //
      stages,                                                       //
      os_mask);

  /* Main time loop */
  output_results(initial_time, 0, os_method_name, method0, method1);
  DiscreteTime time(initial_time, final_time, time_step);
  while (time.is_at_end() == false) {

    os_stepper.evolve_one_time_step(time.get_current_time(), //
                                    time_step,               //
                                    solution);
    time.advance_time();

    constraint_matrix.distribute(solution);

    if (time.get_step_number() % 10 == 0)
      output_results(time.get_current_time(), //
                     time.get_step_number(),  //
                     os_method_name,          //
                     method0,                 //
                     method1);
  }

  // Output summary line for error
  std::cout << ", " << solution.l2_norm();
}
// @sect4{<code>Diffusion::run</code>}
//
// The following is the main function of the program. At the top, we create
// the grid (a [0,5]x[0,5] square) and refine it four times to get a mesh
// that has 16 by 16 cells, for a total of 256.  We then set the boundary
// indicator to 1 for those parts of the boundary where $x=0$ and $x=5$.
void Diffusion::run() {
  GridGenerator::hyper_cube(triangulation, 0., 5.);
  triangulation.refine_global(4);

  for (const auto& cell : triangulation.active_cell_iterators())
    for (const auto& face : cell->face_iterators())
      if (face->at_boundary()) {
        if ((face->center()[0] == 0.) || (face->center()[0] == 5.))
          face->set_boundary_id(1);
        else
          face->set_boundary_id(0);
      }

  // Next, we set up the linear systems and fill them with content so that
  // they can be used throughout the time stepping process:
  setup_system();
  assemble_system();

  // Finally, we solve the diffusion problem using several of the
  // Runge-Kutta methods implemented in namespace TimeStepping, each time
  // outputting the error at the end time. (As explained in the
  // introduction, since the exact solution is zero at the final time, the
  // error equals the numerical solution and can be computed by just taking
  // the $l_2$ norm of the solution vector.)
  // unsigned int       n_steps      = 0;
  const double initial_time = 0.;
  const double final_time   = 10.;

  // Collections of timesteps and methods to use
  std::vector<unsigned int> n_time_steps{50, 100, 200, 400, 800, 1600, 3200};

  std::vector<std::string> method_names{
      "Godunov-BE-FE",       //
      "Strang-SDIRK2-HEUN",  //
      "Ruth-SDIRK3O3-RK3",   //
      "Yoshida-SDIRK5O4-RK4" //
  };

  /****************************************************/
  /****************************************************/
  // Header & method names
  std::cout << "dt";
  for (auto& method : method_names) {
    std::cout << ", " << method;
  }
  std::cout << std::endl;

  // Loop over the collection of n_time_steps
  for (auto& steps : n_time_steps) {
    const double time_step =
        (final_time - initial_time) / static_cast<double>(steps);
    std::cout << time_step;

    // Implicit explicit methods
    os_method_run<tostii::ImplicitRungeKutta<bvector_t>,
                  tostii::ExplicitRungeKutta<bvector_t>> //
        ("Godunov",                                      //
         tostii::os_method.at("Godunov"),                //
         tostii::BACKWARD_EULER,                         //
         tostii::FORWARD_EULER,                          //
         steps,                                          //
         initial_time,                                   //
         final_time);
    os_method_run<tostii::ImplicitRungeKutta<bvector_t>,
                  tostii::ExplicitRungeKutta<bvector_t>> //
        ("Strang",                                       //
         tostii::os_method.at("Strang"),                 //
         tostii::SDIRK_TWO_STAGES,                       //
         tostii::HEUN2,                                  //
         steps,                                          //
         initial_time,                                   //
         final_time);
    os_method_run<tostii::ImplicitRungeKutta<bvector_t>,
                  tostii::ExplicitRungeKutta<bvector_t>> //
        ("Ruth",                                         //
         tostii::os_method.at("Ruth"),                   //
         tostii::SDIRK_THREE_STAGES,                     //
         tostii::RK_THIRD_ORDER,                         //
         steps,                                          //
         initial_time,                                   //
         final_time);
    os_method_run<tostii::ImplicitRungeKutta<bvector_t>,
                  tostii::ExplicitRungeKutta<bvector_t>> //
        ("Yoshida",                                      //
         tostii::os_method.at("Yoshida"),                //
         tostii::SDIRK_5O4,                              //
         tostii::RK_CLASSIC_FOURTH_ORDER,                //
         steps,                                          //
         initial_time,                                   //
         final_time);
    std::cout << std::endl;
  }
}
} // namespace Neutron

// @sect3{The <code>main()</code> function}
//
// The following <code>main</code> function is similar to previous examples
// and need not be commented on.
int main(int argc, char* argv[]) {
  try {

    Neutron::Diffusion diffusion(argc, argv);
    diffusion.run();

  } catch (std::exception& exc) {

    std::cerr << std::endl                                              //
              << std::endl                                              //
              << "----------------------------------------------------" //
              << std::endl;                                             //
    std::cerr << "Exception on processing: " << std::endl               //
              << exc.what() << std::endl                                //
              << "Aborting!" << std::endl                               //
              << "----------------------------------------------------" //
              << std::endl;                                             //
    return 1;
  } catch (...) {

    std::cerr << std::endl                                              //
              << std::endl                                              //
              << "----------------------------------------------------" //
              << std::endl;                                             //
    std::cerr << "Unknown exception!" << std::endl                      //
              << "Aborting!" << std::endl                               //
              << "----------------------------------------------------" //
              << std::endl;                                             //
    return 1;
  };

  return 0;
}
