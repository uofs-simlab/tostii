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
 * This Brusselator solver uses tostii library for its time integration
 * We use different time integration methods for negative stages
 * It is same as adr.cc just some dictionaries are defined for the time integration methods
 */


// @sect3{Include files}
// The program starts with the usual include files, all of which you should
// have seen before by now:
#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_tools.h>
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
#include <deal.II/base/function.h>  
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/base/utilities.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/base/index_set.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>

#include <fstream>
#include <iostream>
#include <cmath>
#include <chrono>
// Include our separated time-integration library
// (has some overlap with deal.II/base/time_stepping.h)
#include <tostii/tostii.h>


// Then the usual placing of all content of this program into a namespace and
// the importation of the deal.II namespace into the one we will work in:
namespace Adr
{
  using namespace dealii;

  // Types
  using time_type = double;


  template <int dim>
  class AdrEquation
  {
  public:
    AdrEquation(int argc, char* argv[]);
    void run();

  private:
    void setup_system();
    void assemble_matrices();
    void evaluate_diffusion(const time_type, const BlockVector<double>&, BlockVector<double>&);
    void evaluate_advection(const time_type, const BlockVector<double>&, BlockVector<double>&);
    void id_minus_tau_J_diffusion_inverse(const time_type, const time_type, 
                                          const BlockVector<double>& , BlockVector<double>&);
    void id_minus_tau_J_advection_inverse(const time_type, const time_type, 
                                          const BlockVector<double>& ,BlockVector<double>&);
    void do_reaction_step(time_type, const BlockVector<double>&, BlockVector<double>&);
    void output_results(std::string, const time_type) const;


    Triangulation<dim> triangulation;
    FE_Q<dim>          fe;
    DoFHandler<dim>    dof_handler;
    MappingFE<dim>  mapping;

    AffineConstraints<double> constraints;

    SparsityPattern sparsity_pattern;

    SparseMatrix<double> mass_matrix;
    SparseMatrix<double> system_diffusion;
    SparseMatrix<double> system_advection_u;
    SparseMatrix<double> system_advection_v;
    SparseMatrix<double> mass_minus_tau_diffusion;
    SparseMatrix<double> mass_minus_tau_advection_u;
    SparseMatrix<double> mass_minus_tau_advection_v;


    BlockVector<double> solution;
    
    time_type    time;
    unsigned int n_time_steps;
    time_type    time_step;
    unsigned int timestep_number;

  };


  // @sect3{Equation data}

  template <int dim>
  class InitialValuesU : public Function<dim, double>
  {
  public:
    InitialValuesU()
      : Function<dim, double>(1){}

    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const override;
  };



  template <int dim>
  double
  InitialValuesU<dim>::value(const Point<dim> & p,
                            const unsigned int /*component*/) const
  {

    const double y = p[1];

    return 22.0 * y * std::pow((1.0 - y), 1.5);
  }

  template <int dim>
  class InitialValuesV : public Function<dim, double>
  {
  public:
    InitialValuesV()
      : Function<dim, double>(1){}

    virtual double
    value(const Point<dim> &p, const unsigned int component = 0) const override;
  };



  template <int dim>
  double
  InitialValuesV<dim>::value(const Point<dim> & p,
                            const unsigned int /*component*/) const
  {

    const double x = p[0];

    return 27.0 * x * std::pow((1.0 - x), 1.5);
  }


  // @sect3{Implementation of the <code>NonlinearSchroedingerEquation</code> class}

  // We start by specifying the implementation of the constructor
  // of the class. We choose quadratic ($Q_2$) Lagrange elements
  template <int dim>
  AdrEquation<dim>::AdrEquation(int /*argc*/, char** /*argv*/)
    : fe(2),
     dof_handler(triangulation),
     mapping(fe),
     time(0),
     n_time_steps(200),
     time_step(1.0 / n_time_steps),
     timestep_number(0)
     {}


  // @sect4{Setting up data structures and assembling matrices}

  // The next function is the one that sets up the mesh, DoFHandler, and
  // matrices and vectors at the beginning of the program, i.e. before the      
  // first time step. The first few lines are pretty much standard if you've
  // read through the tutorial programs at least up to step-6:
  template <int dim>
    void AdrEquation<dim>::setup_system() {
    // Generate a mesh
    GridGenerator::hyper_cube(triangulation, 0.0, 1.0);

    // Refine globally
    triangulation.refine_global(5);

    // Distribute degrees of freedom
    dof_handler.distribute_dofs(fe);

    constraints.close();

    // Set up the rest of the system (matrices, vectors, sparsity pattern)
    solution.reinit(2, dof_handler.n_dofs());

    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);
    sparsity_pattern.copy_from(dsp);

    // Reinitialize the system matrices
    system_diffusion.reinit(sparsity_pattern);
    system_advection_u.reinit(sparsity_pattern);
    system_advection_v.reinit(sparsity_pattern);
    mass_matrix.reinit(sparsity_pattern);
    mass_minus_tau_diffusion.reinit(sparsity_pattern);
    mass_minus_tau_advection_u.reinit(sparsity_pattern);
    mass_minus_tau_advection_v.reinit(sparsity_pattern);

    // std::cout << "Number of active cells: " << triangulation.n_active_cells() << std::endl;
    // std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;
  }


  template <int dim>
  void AdrEquation<dim>::assemble_matrices(){
    mass_matrix = 0.0;
    system_diffusion = 0.0;
    system_advection_u = 0.0;
    system_advection_v = 0.0;

    const QGauss<dim> quadrature_formula(fe.degree + 1);

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                            update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points    = quadrature_formula.size();

    FullMatrix<double> cell_mass_matrix(dofs_per_cell, dofs_per_cell);

    FullMatrix<double> cell_matrix_diffusion(dofs_per_cell, dofs_per_cell);

    FullMatrix<double> cell_matrix_advection_u(dofs_per_cell, dofs_per_cell);

    FullMatrix<double> cell_matrix_advection_v(dofs_per_cell, dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  
    for (const auto &cell : dof_handler.active_cell_iterators())
      {

            
          cell_mass_matrix = 0.0;
          cell_matrix_diffusion = 0.0;
          cell_matrix_advection_u = 0.0;
          cell_matrix_advection_v = 0.0;

          fe_values.reinit(cell);

          for (unsigned int q_index = 0; q_index < n_q_points; ++q_index)
            {
              for (unsigned int k = 0; k < dofs_per_cell; ++k)
                {
                  for (unsigned int l = 0; l < dofs_per_cell; ++l)
                    {

                      cell_mass_matrix(k,l) +=  fe_values.shape_value(k, q_index) *
                              fe_values.shape_value(l, q_index) *
                              fe_values.JxW(q_index);
                      
                      const double vx_u = -0.5;
                      const double vy_u = 1.0;

                      const double vx_v = 0.4;
                      const double vy_v = 0.7;

                      cell_matrix_diffusion(k,l) += -0.01 * fe_values.shape_grad(k, q_index) *
                                                fe_values.shape_grad(l, q_index) *
                                                fe_values.JxW(q_index);

                    cell_matrix_advection_u(k, l) += 0.5 * fe_values.shape_value(k, q_index) * (
                                                  vx_u * fe_values.shape_grad(l, q_index)[0] +
                                                  vy_u * fe_values.shape_grad(l, q_index)[1]
                                              ) * fe_values.JxW(q_index);

                      
                      cell_matrix_advection_v(k, l) += 0.5 * fe_values.shape_value(k, q_index) * (
                                                    vx_v * fe_values.shape_grad(l, q_index)[0] +
                                                    vy_v * fe_values.shape_grad(l, q_index)[1]
                                                ) * fe_values.JxW(q_index);


                    }
                }
            }

          cell->get_dof_indices(local_dof_indices);

          constraints.distribute_local_to_global(cell_mass_matrix,
                                                local_dof_indices,
                                                mass_matrix);

          constraints.distribute_local_to_global(cell_matrix_diffusion,
                                                local_dof_indices,
                                                system_diffusion);

          constraints.distribute_local_to_global(cell_matrix_advection_u,
                                          local_dof_indices,
                                          system_advection_u);
                                
          constraints.distribute_local_to_global(cell_matrix_advection_v,
                                  local_dof_indices,
                                  system_advection_v);

        
      }
  }

  template <int dim>
  void AdrEquation<dim>::evaluate_diffusion(
    const time_type /*time*/,  
    const BlockVector<double>& yin, 
    BlockVector<double>& yout) {
    Vector<double> rhs(dof_handler.n_dofs());
    rhs = 0.0;
    system_diffusion.vmult(rhs, yin.block(0));

    // Initialize the direct solver
    SparseDirectUMFPACK direct_solver;
    direct_solver.initialize(mass_matrix);
    direct_solver.vmult(yout.block(0), rhs);

    constraints.distribute(yout.block(0));

    rhs = 0.0;
    system_diffusion.vmult(rhs, yin.block(1));

    direct_solver.vmult(yout.block(1), rhs);

    constraints.distribute(yout.block(1));
  }

  template <int dim>
  void AdrEquation<dim>::id_minus_tau_J_diffusion_inverse(
    const time_type /*time*/,  
    const time_type tau, 
    const BlockVector<double>& yin, 
    BlockVector<double>& yout) {
    mass_minus_tau_diffusion.copy_from(system_diffusion);
    mass_minus_tau_diffusion *= -tau;
    mass_minus_tau_diffusion.add(1.0, mass_matrix);

    Vector<double> rhs(dof_handler.n_dofs());
    mass_matrix.vmult(rhs, yin.block(0));

    // Initialize the direct solver for the modified matrix
    SparseDirectUMFPACK direct_solver_diffusion;
    direct_solver_diffusion.initialize(mass_minus_tau_diffusion);
    direct_solver_diffusion.vmult(yout.block(0), rhs);

    constraints.distribute(yout.block(0));

    rhs = 0.0;
    mass_matrix.vmult(rhs, yin.block(1));

    direct_solver_diffusion.vmult(yout.block(1), rhs);

    constraints.distribute(yout.block(1));
  }


  template <int dim>
  void AdrEquation<dim>::evaluate_advection(
    const time_type /*time*/,  
    const BlockVector<double>& yin, 
          BlockVector<double>& yout) {
    // Temporary vectors for RHS
    Vector<double> rhs_u(dof_handler.n_dofs());
    Vector<double> rhs_v(dof_handler.n_dofs());

    // Compute RHS for 'u' component: rhs_u = system_advection_u * yin_u
    system_advection_u.vmult(rhs_u, yin.block(0));

    SparseDirectUMFPACK direct_solver_mass;
    direct_solver_mass.initialize(mass_matrix);

    // Solve M * yout_u = rhs_u using the direct solver for mass matrix
    direct_solver_mass.vmult(yout.block(0), rhs_u);

    // Apply constraints to the solution
    constraints.distribute(yout.block(0));

    // Compute RHS for 'v' component: rhs_v = system_advection_v * yin_v
    system_advection_v.vmult(rhs_v, yin.block(1));

    // Solve M * yout_v = rhs_v using the direct solver for mass matrix
    direct_solver_mass.vmult(yout.block(1), rhs_v);

    // Apply constraints to the solution
    constraints.distribute(yout.block(1));
  }

  template <int dim>
  void AdrEquation<dim>::id_minus_tau_J_advection_inverse(
    const time_type /*time*/,  
    const time_type       tau, 
    const BlockVector<double>& yin, 
          BlockVector<double>&       yout) {
    // Compute (M - tau * J_u) and (M - tau * J_v)
    mass_minus_tau_advection_u.copy_from(system_advection_u);
    mass_minus_tau_advection_u *= -tau;
    mass_minus_tau_advection_u.add(1.0, mass_matrix);
    
    mass_minus_tau_advection_v.copy_from(system_advection_v);
    mass_minus_tau_advection_v *= -tau;
    mass_minus_tau_advection_v.add(1.0, mass_matrix);

    SparseDirectUMFPACK direct_solver_mass_minus_tau_advection_u;
    SparseDirectUMFPACK direct_solver_mass_minus_tau_advection_v;

    direct_solver_mass_minus_tau_advection_u.initialize(mass_minus_tau_advection_u);
    direct_solver_mass_minus_tau_advection_v.initialize(mass_minus_tau_advection_v);

    // Temporary vectors for RHS
    Vector<double> rhs_u(dof_handler.n_dofs());
    Vector<double> rhs_v(dof_handler.n_dofs());

    // Compute RHS for 'u' component: rhs_u = M * yin_u
    mass_matrix.vmult(rhs_u, yin.block(0));

    // Solve (M - tau * J_u) * yout_u = rhs_u
    direct_solver_mass_minus_tau_advection_u.vmult(yout.block(0), rhs_u);

    // Apply constraints to the solution
    constraints.distribute(yout.block(0));

    // Compute RHS for 'v' component: rhs_v = M * yin_v
    mass_matrix.vmult(rhs_v, yin.block(1));

    // Solve (M - tau * J_v) * yout_v = rhs_v
    direct_solver_mass_minus_tau_advection_v.vmult(yout.block(1), rhs_v);

    // Apply constraints to the solution
    constraints.distribute(yout.block(1));

  }

  template <int dim>
  void AdrEquation<dim>::do_reaction_step(
    time_type /*t*/,                 //
    const BlockVector<double>& yin,       //
          BlockVector<double>&       yout) {

      yout.reinit(yin);

      auto& U = yin.block(0); // u
      auto& V = yin.block(1); // v

      auto& U_prime = yout.block(0); // u'
      auto& V_prime = yout.block(1); // v'

        const unsigned int n = V.size();
      for (unsigned int i = 0; i < n; ++i){
        U_prime[i] = 1.3 - 2.0 * U[i] + V[i] * U[i] * U[i];
        V_prime[i] = 1.0 * U[i] - V[i] * U[i] * U[i];
      }

      constraints.distribute(yout.block(0));
      constraints.distribute(yout.block(1));

  }


  // We create output as we always
  // do. As in many other time-dependent tutorial programs, we attach flags to
  // DataOut that indicate the number of the time step and the current
  // simulation time.

  template <int dim>
  void AdrEquation<dim>::output_results(std::string name, const time_type time) const{
    DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution.block(0), "u");
    data_out.add_data_vector(solution.block(1), "v");
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


  template <int dim>
  void AdrEquation<dim>::run(){
    // Start timing here
    auto start_time = std::chrono::high_resolution_clock::now();

    setup_system();
    assemble_matrices();

    time = 0;

    VectorTools::interpolate(dof_handler, InitialValuesU<dim>(), solution.block(0));
    VectorTools::interpolate(dof_handler, InitialValuesV<dim>(), solution.block(1));

    constraints.distribute(solution.block(0));
    constraints.distribute(solution.block(1));

    /* Define methods, operators and alpha for operator split */
    tostii::runge_kutta_method            implicit_rk{tostii::SDIRK_5O4};
    tostii::runge_kutta_method            explicit_rk{tostii::RK_CLASSIC_FOURTH_ORDER};
    tostii::runge_kutta_method             n_explicit{tostii::RK_CLASSIC_FOURTH_ORDER};
    tostii::ImplicitRungeKutta<BlockVector<double>, time_type>  implicit_stepper_method(implicit_rk);
    tostii::ExplicitRungeKutta<BlockVector<double>, time_type>  explicit_stepper_method(explicit_rk);
    tostii::ExplicitRungeKutta<BlockVector<double>, time_type>  explicit_negative_stepper_method(n_explicit);
    
    //Consider explicit methods for negative stages in the Yoshida method
    std::map<int, tostii::TimeStepping<BlockVector<double>, time_type>*> methods = {{4, &explicit_negative_stepper_method},
                                                                                    {5, &explicit_negative_stepper_method},
                                                                                    {7, &explicit_negative_stepper_method},
                                                                                    {8, &explicit_negative_stepper_method}};

    // // Consider explicit methods for negative stages in the PP_3_A_3 method
    // std::map<int, tostii::TimeStepping<BlockVector<double>, time_type>*> methods = {{1, &explicit_negative_stepper_method},
    //                                                                                 {3, &explicit_negative_stepper_method},
    //                                                                                 {6, &explicit_negative_stepper_method},
    //                                                                                 {14, &explicit_negative_stepper_method}
    //                                                                                 {15, &explicit_negative_stepper_method}
    //                                                                                 {16, &explicit_negative_stepper_method}};


    /* Define OSoperators to use in the operator split stepper */
    tostii::OSoperator<BlockVector<double>, time_type> diffusion_stepper{
      &implicit_stepper_method,
        [this](const time_type t,  //
               const BlockVector<double>& yin, //
               BlockVector<double>&       yout) {this->evaluate_diffusion(t, yin, yout);},
        [this](const time_type    t,   //
               const time_type    dt,  //
               const BlockVector<double>& yin, //
               BlockVector<double>&       yout) {this->id_minus_tau_J_diffusion_inverse(t, dt, yin, yout);}
          };

    tostii::OSoperator<BlockVector<double>, time_type> advection_stepper{
      &implicit_stepper_method,
        [this](const time_type    t,   //
               const BlockVector<double>& yin, //
               BlockVector<double>& yout) { this->evaluate_advection(t, yin, yout); },
        [this](const time_type    t,   //
               const time_type    dt,  //
               const BlockVector<double>& yin, //
               BlockVector<double>&       yout) {this->id_minus_tau_J_advection_inverse(t, dt, yin, yout);}
          };

      tostii::OSoperator<BlockVector<double>, time_type> reaction_stepper{
        &explicit_stepper_method,
      [this](const time_type    t,   //
              const BlockVector<double>& yin, //
              BlockVector<double>& yout) { this->do_reaction_step(t, yin, yout); },
      [this](const time_type    /*t*/,   //
              const time_type    /*dt*/,  //`
              const BlockVector<double>& /*yin*/, //
              BlockVector<double>&       /*yout*/) {return;}
          };
      
      std::vector<tostii::OSmask>      os_mask{{0, 1}, {0, 1}, {0, 1}};


    std::string os_name{"Yoshida3"};
    auto        os_coeffs = tostii::os_method.at(os_name);
    tostii::OperatorSplit<BlockVector<double>, time_type> os_stepper(
							   solution, //
        std::vector<tostii::OSoperator<BlockVector<double>, time_type>>{  //
							      diffusion_stepper, advection_stepper, reaction_stepper}, //
                    os_coeffs, //
                    os_mask //
                    );

    // Step 0 output:
    output_results(os_name, time);

    // Main time loop
    for (unsigned int itime=1; itime <= n_time_steps; ++itime)
    {
      ++timestep_number;

      time = os_stepper.evolve_one_time_step(time, time_step, solution, methods);

      // std::cout << "Time step " << timestep_number << " at t=" << time
      //           << std::endl;

      // if (timestep_number % 10 == 0) {
      //   output_results(os_name, time);
      // }
    }

    // End timing here
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    std::cout << "Total execution time: " << elapsed.count() << " seconds" << std::endl;

    // Save solution to CSV files with timestep information
std::ofstream u_output("solution_u_rk" + std::to_string(n_time_steps) + ".csv");
std::ofstream v_output("solution_v_rk" + std::to_string(n_time_steps) + ".csv");
u_output.precision(16);
v_output.precision(16);

for (unsigned int i = 0; i < solution.block(0).size(); ++i)
{
    u_output << solution.block(0)[i] << "\n";
    v_output << solution.block(1)[i] << "\n";
}

u_output.close();
v_output.close();

std::cout << "Solutions saved to solution_u_" << n_time_steps << ".csv and solution_v_" << n_time_steps << ".csv." << std::endl;



  }

} // namespace Adr



// @sect4{The main() function}
//
// The rest is again boiler plate and exactly as in almost all of the previous
// tutorial programs:
int main(int argc, char* argv[])
{
  try
    {
      using namespace Adr;
      AdrEquation<2> adr(argc,argv);
      adr.run();
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