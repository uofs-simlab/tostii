/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2018 - 2020 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------
 *
 * Author: Mahdi Moayeri, University of Saskatchewan

 * Modification of step-58 to use tost.II time integration for solving a complex PDE
   - See https://www.dealii.org/current/doxygen/deal.II/step_58.html for
     the original code that solved NSE problem.
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
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_tools.h>

#include <fstream>
#include <iostream>

// Include our separated time-integration library
// (has some overlap with deal.II/base/time_stepping.h)
#include <tostii/tostii.h>


// Then the usual placing of all content of this program into a namespace and
// the importation of the deal.II namespace into the one we will work in:
namespace StepOS
{
  using namespace dealii;

  // Types
  using value_type   = std::complex<double>;
  using vector_type  = Vector<value_type>;
  using matrix_type  = SparseMatrix<value_type>;
  using f_fun_type   = std::function<vector_type(const double, const vector_type &)>;
  using f_vfun_type  = std::vector<f_fun_type>;
  using jac_fun_type = std::function<vector_type(const double, const double, const vector_type &)>;
  using jac_vfun_type = std::vector<jac_fun_type>;

  // using time_type = std::complex<double>;
  // #define timereal time.real()
  using time_type = double;
  #define timereal time

  // @sect3{The <code>NonlineaComplexEquation</code> class}
  //
  // linear system are now storing elements of type `std::complex<double>`
  // instead of just `double`.
  template <int dim>
  class NonlineaComplexEquation
  {
  public:
    NonlineaComplexEquation(int argc, char* argv[]);
    void run();

  private:
    void setup_system();
    void assemble_matrices();
    void evaluate_spatial_rhs(const time_type, const vector_type&, vector_type&);
    void id_minus_tau_J_inverse(const time_type, const time_type, const vector_type& ,vector_type&);
    void do_half_phase_step(time_type, time_type, const vector_type&, vector_type&);
    void do_full_spatial_step(time_type, time_type, vector_type&);
    void output_results(std::string) const;
    void compute_error(time_type);
            
    std::vector<Point<dim>> support_points;

    Triangulation<dim> triangulation;
    FE_Q<dim>          fe;
    DoFHandler<dim>    dof_handler;

    AffineConstraints<value_type> constraints;

    SparsityPattern                    sparsity_pattern;

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
  class InitialValues : public Function<dim, value_type>
  {
  public:
    InitialValues()
      : Function<dim, value_type>(1)
    {}

    virtual value_type
    value(const Point<dim> &p, const unsigned int component = 0) const override;
  };



  template <int dim>
  value_type
  InitialValues<dim>::value(const Point<dim> & p,
                            const unsigned int component) const
  {

    (void)component;
    Assert(component == 0, ExcIndexRange(component, 0, 1));

    const double x = p[0];

    const double initial_value = std::exp(-0.5*(x*x));

    return {initial_value, 0.};
  }



  // @sect3{Implementation of the <code>NonlineaComplexEquation</code> class}

  template <int dim>
  NonlineaComplexEquation<dim>::NonlineaComplexEquation(int /*argc*/, char** /*argv*/)
    : fe(2)
    , dof_handler(triangulation)
    , time(0)
    , n_time_steps(20480)
    , time_step(1. / n_time_steps)
    , timestep_number(0)
  {}

  // @sect4{Setting up data structures and assembling matrices}

  // The next function is the one that sets up the mesh, DoFHandler, and
  // matrices and vectors at the beginning of the program, i.e. before the
  // first time step. The first few lines are pretty much standard if you've
  // read through the tutorial programs at least up to step-6:
  template <int dim>
  void NonlineaComplexEquation<dim>::setup_system(){

    GridGenerator::hyper_cube(triangulation, -10.0, 10.0);
    triangulation.refine_global(12);

    
    dof_handler.distribute_dofs(fe);

    constraints.clear();
    // Collect periodic faces and apply constraints
    std::vector<GridTools::PeriodicFacePair<typename DoFHandler<dim>::cell_iterator>> periodic_faces;
    GridTools::collect_periodic_faces(dof_handler, 0, 1, 0, periodic_faces);
    DoFTools::make_periodicity_constraints<dim, dim>(
        periodic_faces, constraints, ComponentMask(), {});


    // Ensure compatibility of constraints
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    constraints.close();

    // Map DoFs to support points
    MappingQ1<dim> mapping;
    support_points.resize(dof_handler.n_dofs());
    DoFTools::map_dofs_to_support_points(mapping, dof_handler, support_points);

    std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl
              << std::endl;

    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, /*keep_constrained_entries=*/false);
    sparsity_pattern.copy_from(dsp);


    system_jacobian.reinit(sparsity_pattern);
    mass_matrix.reinit(sparsity_pattern);
    mass_minus_tau_Jacobian.reinit(sparsity_pattern);

    solution.reinit(dof_handler.n_dofs());

    constraints.close();
  }

  template <int dim>
  void NonlineaComplexEquation<dim>::evaluate_spatial_rhs(
      const time_type /*time*/,  //
      const vector_type& yin, //
      vector_type&       yout) {
    // Since this is just a linear problem, f = jacobian*y
    // yout.reinit(dof_handler.n_dofs());
    vector_type rhs(dof_handler.n_dofs());
    rhs = static_cast<value_type>(0);
    system_jacobian.vmult(rhs, yin);
    // Darn! Has to be done this way due to partial implementation of
    // SparseDirectUMFPack with std::complex
    inverse_mass_matrix.solve(mass_matrix,rhs);
    yout = rhs;
    constraints.distribute(yout);
  }

  template <int dim>
  void NonlineaComplexEquation<dim>::id_minus_tau_J_inverse(
      const time_type /*time*/,  //
      const time_type       tau, //
      const vector_type& yin, //
      vector_type&       yout) {

    SparseDirectUMFPACK inverse_mass_minus_tau_Jacobian;

    mass_minus_tau_Jacobian.copy_from(mass_matrix);
    mass_minus_tau_Jacobian.add(-tau, system_jacobian);

    inverse_mass_minus_tau_Jacobian.initialize(mass_minus_tau_Jacobian);

    vector_type result(dof_handler.n_dofs());
    mass_matrix.vmult(result, yin);

    // Again... a limit of the partial std::complex wrapper to SparseDirectUMFPack
    inverse_mass_minus_tau_Jacobian.solve(mass_minus_tau_Jacobian, result);
    yout = result;
    constraints.distribute(yout);
  }

  template <int dim>
  void NonlineaComplexEquation<dim>::assemble_matrices()
  {
    const QGauss<dim> quadrature_formula(fe.degree + 1);

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();

    FullMatrix<value_type> cell_mass_matrix(dofs_per_cell,
                                                     dofs_per_cell);

    FullMatrix<value_type> cell_matrix_jacobian(dofs_per_cell,
                                                     dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    std::vector<double>                  potential_values(n_q_points);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        cell_mass_matrix = static_cast<value_type>(0);
        cell_matrix_jacobian = static_cast<value_type>(0);

        fe_values.reinit(cell);

        for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
          {
            for (unsigned int k = 0; k < dofs_per_cell; ++k)
              {
                for (unsigned int l = 0; l < dofs_per_cell; ++l)
                  {
                    const value_type i = {0, 1};


                cell_mass_matrix(k,l) +=  fe_values.shape_value(k, q_index) *
                                          fe_values.shape_value(l, q_index) *
                                          fe_values.JxW(q_index);

		cell_matrix_jacobian(k,l) += -(1 + 0.5 * i) *   fe_values.shape_grad(k, q_index) *
                                        fe_values.shape_grad(l, q_index) *
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
  void NonlineaComplexEquation<dim>::do_half_phase_step(
    time_type /*t*/,                 //
    time_type             step_size, //
    const vector_type& yin,       //
    vector_type&       yout) {

    const value_type i = {0, 1};

    // yout.reinit(yin); // Ensure yout is properly sized
    yout               = yin;

    time_type theta = std::exp(-2.0 * step_size);

    for (auto& value : yout) {
      value = value / std::sqrt(theta + (1.0 - i) * (value * value) * (1.0 - theta));
    }
  }



  // We create output as we always
  // do. As in many other time-dependent tutorial programs, we attach flags to
  // DataOut that indicate the number of the time step and the current
  // simulation time.
  template <int dim>
  void NonlineaComplexEquation<dim>::output_results(std::string name) const{
    
    DataOut<dim> data_out;

    data_out.add_data_vector(solution, "Psi");
    data_out.attach_dof_handler(dof_handler);
    
    data_out.set_flags(DataOutBase::VtkFlags(timereal, timestep_number));

    const std::string filename =
      "solution_" + Utilities::int_to_string(n_time_steps) + "_"
      + name + "-" + Utilities::int_to_string(timestep_number, 4) + ".vtu";
    std::ofstream output(filename);
    data_out.write_vtu(output);
  }



  // @sect4{Running the simulation}

  // The remaining step is how we set up the overall logic for this program.
  // It's really relatively simple: Set up the data structures; interpolate the
  // initial conditions onto finite element space
  template <int dim>
  void NonlineaComplexEquation<dim>::run()
  {
    // Start timing here
    auto start_time = std::chrono::high_resolution_clock::now();

    setup_system();
    assemble_matrices();

    time = 0;

    //initial condition
    VectorTools::interpolate(dof_handler, InitialValues<dim>(), solution);

    /* Define methods, operators and alpha for operator split */
    tostii::Exact<vector_type, time_type> half_stepper_method;
    tostii::runge_kutta_method            full_step_method{tostii::SDIRK_TWO_STAGES};
    tostii::ImplicitRungeKutta<vector_type, time_type> full_stepper_method(
        full_step_method, //
        1);               // single iteration (linear system)

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
               vector_type& yout) { this->evaluate_spatial_rhs(t, yin, yout); },
        [this](const time_type    t,   //
               const time_type    dt,  //
               const vector_type& yin, //
               vector_type&       yout) {
          this->id_minus_tau_J_inverse(t, dt, yin, yout);
        }};
    
    //Complex-valued OS methods
    // std::string os_name{"Milne_2_2_c_i"};
    // std::string os_name{"A_3_3_c"};
    // std::string os_name{"Yoshida_c"};
    // auto        os_coeffs = tostii::os_complex.at(os_name);

    // tostii::OperatorSplitSingle<vector_type, time_type> os_stepper(
    //     solution,                                                      //
    //     std::vector<tostii::OSoperator<vector_type, time_type>>{half_stepper,  //
    //                                                     full_stepper}, //
    //     os_coeffs);

    // Real-valued OS methods
    std::string os_name{"Strang"};
    auto        os_coeffs = tostii::os_method.at(os_name);
    tostii::OperatorSplitSingle<vector_type, time_type> os_stepper(
							   solution, //
        std::vector<tostii::OSoperator<vector_type, time_type>>{half_stepper,  //
							       full_stepper}, //
        os_coeffs);

    // Main time loop
    for (unsigned int itime=1; itime <= n_time_steps; ++itime)
      {
        ++timestep_number;

        time = os_stepper.evolve_one_time_step(time, time_step, solution);

        std::cout << "Time step " << timestep_number << " at t=" << time
          << std::endl;

        auto full_step_name{RK_method_enum_to_string(full_step_method)};


        // if (timestep_number % 10 == 0) {
        //   output_results(os_name+"_Exact_"+full_step_name);
	      // }
      }


      auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    std::cout << "Total execution time: " << elapsed.count() << " seconds" << std::endl;


      const std::string filename =
      "solution_" + Utilities::int_to_string(n_time_steps) + "_"
      + os_name +"-" + Utilities::int_to_string(timestep_number, 4) +".csv";
    std::ofstream output(filename);
    output.precision(16);  // Set the precision to 16 digits
  for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i)  
    {
    output << solution[i] << std::endl;
    }
  output.close();

  }
} // namespace StepOS



// @sect4{The main() function}
//
// The rest is again boiler plate and exactly as in almost all of the previous
// tutorial programs:
int main(int argc, char* argv[])
{
  try
    {
      using namespace StepOS;

      NonlineaComplexEquation<1> pde(argc,argv);
      pde.run();
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