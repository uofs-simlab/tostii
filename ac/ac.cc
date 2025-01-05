/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2024 by the authors listed below
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
 * Author: Mahdi Moayeri, University of Saskatchewan, 2024
 *
 * This Allen-Cahn solver uses tostii library for its time integration
 */

// @sect3{Include files}
// The program starts with the usual include files, all of which you should
// have seen before by now:
#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/base/data_out_base.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/base/function.h>  

#include <fstream>
#include <iostream>
#include <cmath>
#include <chrono>

// Include our separated time-integration library
// (has some overlap with deal.II/base/time_stepping.h)
#include <tostii/tostii.h>


// Then the usual placing of all content of this program into a namespace and
// the importation of the deal.II namespace into the one we will work in:
namespace AllenCahn
{
  using namespace dealii;

  // Types
  using vector_type  = Vector<double>;
  using matrix_type  = SparseMatrix<double>;

  using time_type = double;


  template <int dim>
  class ACEquation
  {
  public:
    ACEquation(int argc, char* argv[]);
    void run();

  private:
    void setup_system();
    void assemble_matrices();
    void evaluate_diffusion(const time_type, const vector_type&, vector_type&);
    void id_minus_tau_J_inverse(const time_type, const time_type, const vector_type& ,vector_type&);
    void do_half_phase_step(time_type, time_type, const vector_type&, vector_type&);
    void do_full_spatial_step(time_type, time_type, vector_type&);
    void output_results(std::string, const time_type) const;


    Triangulation<dim> triangulation;
    FE_Q<dim>          fe;
    DoFHandler<dim>    dof_handler;

    AffineConstraints<double> constraints;

    SparsityPattern sparsity_pattern;

    matrix_type mass_matrix;
    matrix_type system_jacobian;
    matrix_type mass_minus_tau_Jacobian;

    SparseDirectUMFPACK inverse_mass_matrix;

    vector_type solution;

    time_type    time;
    unsigned int n_time_steps;
    time_type    time_step;
    unsigned int timestep_number;

  };



  // @sect3{Equation data}

  template <int dim>
  class InitialValues : public Function<dim, double>
  {
  public:
    InitialValues()
      : Function<dim, double>(1)
    {}

    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const override;
  };



  template <int dim>
  double
  InitialValues<dim>::value(const Point<dim> & p,
                            const unsigned int component) const
  {

    (void)component;
    Assert(component == 0, ExcIndexRange(component, 0, 1));

    const double x = p[0];
    const double y = p[1];
    double epsilon = 0.03;

    return std::tanh((0.4 - std::sqrt((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5))) / (std::sqrt(2.0) * epsilon)) - 
           std::tanh((0.3 - std::sqrt((x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5))) / (std::sqrt(2.0) * epsilon)) - 1.0;

  }



  // @sect3{Implementation of the <code>ACEquation</code> class}

  template <int dim>
  ACEquation<dim>::ACEquation(int /*argc*/, char** /*argv*/)
    : fe(2)
    , dof_handler(triangulation)
    , time(0)
    , n_time_steps(850)
    , time_step(.006 / n_time_steps)
    , timestep_number(0)
  {}


  // @sect4{Setting up data structures and assembling matrices}

  // The next function is the one that sets up the mesh, DoFHandler, and
  // matrices and vectors at the beginning of the program, i.e. before the
  // first time step. The first few lines are pretty much standard if you've
  // read through the tutorial programs at least up to step-6:
  template <int dim>
  void ACEquation<dim>::setup_system()
  {
    GridGenerator::hyper_cube(triangulation, 0.0, 1.0);
    triangulation.refine_global(6);
    

    std::cout << "Number of active cells: " << triangulation.n_active_cells()
              << std::endl;

    dof_handler.distribute_dofs(fe);

    std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl
              << std::endl;


    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);

    system_jacobian.reinit(sparsity_pattern);
    mass_matrix.reinit(sparsity_pattern);
    mass_minus_tau_Jacobian.reinit(sparsity_pattern);

    solution.reinit(dof_handler.n_dofs());
  }


  // Next, we assemble the relevant matrices.
  template <int dim>
  void ACEquation<dim>::assemble_matrices() {
    mass_matrix = 0.;
    system_jacobian = 0.;

    const QGauss<dim> quadrature_formula(fe.degree + 1);

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                            update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();

    FullMatrix<double> cell_mass_matrix(dofs_per_cell,
                                                     dofs_per_cell);

    FullMatrix<double> cell_matrix_jacobian(dofs_per_cell,
                                                     dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        cell_mass_matrix = 0.;
        cell_matrix_jacobian = 0.;

        fe_values.reinit(cell);

        for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
          {
            for (unsigned int k = 0; k < dofs_per_cell; ++k)
              {
                for (unsigned int l = 0; l < dofs_per_cell; ++l)
                  {

                cell_matrix_jacobian(k,l) += (-1.0 * fe_values.shape_grad(k, q_index) *
                                          fe_values.shape_grad(l, q_index) *
                                          fe_values.JxW(q_index));

                cell_mass_matrix(k,l) +=  fe_values.shape_value(k, q_index) *
                                          fe_values.shape_value(l, q_index) *
                                          fe_values.JxW(q_index);

                  }
              }
          }

        cell->get_dof_indices(local_dof_indices);

        constraints.distribute_local_to_global(cell_mass_matrix,
                                               local_dof_indices,
                                               mass_matrix);
        constraints.distribute_local_to_global(cell_matrix_jacobian,
                                               local_dof_indices,
                                               system_jacobian);

      }

    inverse_mass_matrix.initialize(mass_matrix);

  }

  template <int dim>
  void ACEquation<dim>::evaluate_diffusion(
      const time_type /*time*/,  //
      const vector_type& yin, //
      vector_type&       yout) {
    // Since this is just a linear problem, f = jacobian*y
    vector_type rhs(dof_handler.n_dofs());
    rhs = 0.;

    system_jacobian.vmult(rhs, yin);
    yout.reinit(dof_handler.n_dofs());
    inverse_mass_matrix.vmult(yout, rhs);
  }

  template <int dim>
  void ACEquation<dim>::id_minus_tau_J_inverse(
      const time_type /*time*/,  //
      const time_type       tau, //
      const vector_type& yin, //
      vector_type&       yout) {

    SparseDirectUMFPACK inverse_mass_minus_tau_Jacobian;

    mass_minus_tau_Jacobian.copy_from(mass_matrix);
    mass_minus_tau_Jacobian.add(-tau, system_jacobian);

    inverse_mass_minus_tau_Jacobian.initialize(mass_minus_tau_Jacobian);

    vector_type tmp(dof_handler.n_dofs());
    mass_matrix.vmult(tmp, yin);

    yout.reinit(yin);
  inverse_mass_minus_tau_Jacobian.vmult(yout, tmp);
  }


  template <int dim>
  void ACEquation<dim>::do_half_phase_step(
      time_type /*t*/,                 //
      time_type             step_size, //
      const vector_type& yin,       //
      vector_type&       yout) {

    double epsilon = 0.03;
    yout               = yin;
    double theta = std::exp((-2.0 * step_size) / (epsilon * epsilon));
    for (auto& value : yout) {
      value = value / std::sqrt(theta + (value * value) * (1.0 - theta));
    }
  }


  //We create output as we always
  // do. As in many other time-dependent tutorial programs, we attach flags to
  // DataOut that indicate the number of the time step and the current
  // simulation time.

  template <int dim>
  void ACEquation<dim>::output_results(std::string name, const time_type time) const{
    DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "u");
    data_out.build_patches();

    // Set the precision for VTU output to double precision
    data_out.set_flags(DataOutBase::VtkFlags(time, timestep_number));

    const std::string filename =
      "solution_" + Utilities::int_to_string(n_time_steps) + "_"
      + name + "-" + Utilities::int_to_string(timestep_number, 4) + ".vtu";
    std::ofstream output(filename);
    
    output.precision(16);  // Ensure the precision of the output stream is set to double precision
    data_out.write_vtu(output);
}



  // @sect4{Running the simulation}

  // The remaining step is how we set up the overall logic for this program.
  // It's really relatively simple: Set up the data structures; interpolate the
  // initial conditions onto finite element space; then iterate over all time
  // steps, and on each time step perform
  // splitting method.

  template <int dim>
  void ACEquation<dim>::run(){
    // Start timing here
    auto start_time = std::chrono::high_resolution_clock::now();

    setup_system();
    assemble_matrices();

    time = 0;
    VectorTools::interpolate(dof_handler, InitialValues<dim>(), solution);

    /* Define methods, operators and alpha for operator split */
    tostii::Exact<vector_type, time_type> half_stepper_method;
    tostii::runge_kutta_method            full_step_method{tostii::SDIRK_THREE_STAGES};
    tostii::ImplicitRungeKutta<vector_type, time_type> full_stepper_method(full_step_method);

    /* Define OSoperators to use in the operator split stepper */
    tostii::OSoperator<vector_type, time_type> half_stepper{
        &half_stepper_method,
        [this](const time_type /*t*/,  //
               const vector_type& yin, //
               vector_type&       yout) { yout = yin; },
        [this](const time_type    t,   //
               const time_type    dt,  //
               const vector_type& yin, //
               vector_type&       yout) {
          this->do_half_phase_step(t, dt, yin, yout);
        }};

    tostii::OSoperator<vector_type, time_type> full_stepper{
        &full_stepper_method,
        [this](const time_type    t,   //
               const vector_type& yin, //
               vector_type& yout) { this->evaluate_diffusion(t, yin, yout); },
        [this](const time_type    t,   //
               const time_type    dt,  //
               const vector_type& yin, //
               vector_type&       yout) {
          this->id_minus_tau_J_inverse(t, dt, yin, yout);
        }};

    std::string os_name{"Ruth"};
    auto        os_coeffs = tostii::os_method.at(os_name);
    tostii::OperatorSplitSingle<vector_type, time_type> os_stepper(
							   solution, //
        std::vector<tostii::OSoperator<vector_type, time_type>>{  //
							       full_stepper, half_stepper}, //
        os_coeffs);

    // Step 0 output:
    auto full_step_name{RK_method_enum_to_string(full_step_method)};
    
    // Main time loop
    for (unsigned int itime=1; itime <= n_time_steps; ++itime)
    {
      ++timestep_number;

      time = os_stepper.evolve_one_time_step(time, time_step, solution);

      std::cout << "Time step " << timestep_number << " at t=" << time
                << std::endl;

      if (timestep_number % 10 == 0) {
        output_results(os_name+"_Exact_"+full_step_name, time);
      }
    }

    // End timing here
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    std::cout << "Total execution time: " << elapsed.count() << " seconds" << std::endl;

  }
} // namespace AllenCahn



// @sect4{The main() function}
//
// The rest is again boiler plate and exactly as in almost all of the previous
// tutorial programs:
int main(int argc, char* argv[])
{
  try
    {
      using namespace AllenCahn;

      ACEquation<2> ac(argc,argv);
      ac.run();
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