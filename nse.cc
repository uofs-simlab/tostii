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
 * Author: Kevin R. Green, University of Saskatchewan

 * Modification of step-58 to use tost.II time integration
 * - Comments have generally been kept intact from the original, with
     comments added that are specifically relevant to the OperatorSplit
     integration.
   - See https://www.dealii.org/current/doxygen/deal.II/step_58.html for
     the original code that solved this problem.
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

#include <fstream>
#include <iostream>

// Include our separated time-integration library
// (has some overlap with deal.II/base/time_stepping.h)
#include "tostii.h"


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

  // @sect3{The <code>NonlinearSchroedingerEquation</code> class}
  //
  // Then the main class. It looks very much like the corresponding
  // classes in step-4 or step-6, with the only exception that the
  // matrices and vectors and everything else related to the
  // linear system are now storing elements of type `std::complex<double>`
  // instead of just `double`.
  template <int dim>
  class NonlinearSchroedingerEquation
  {
  public:
    NonlinearSchroedingerEquation(int argc, char* argv[]);
    void run();

  private:
    void setup_system();
    void assemble_matrices();
    void evaluate_spatial_rhs(const time_type, const vector_type&, vector_type&);
    void id_minus_tau_J_inverse(const time_type, const time_type, const vector_type& ,vector_type&);
    void do_half_phase_step(time_type, time_type, const vector_type&, vector_type&);
    void do_full_spatial_step(time_type, time_type, vector_type&);
    void output_results(std::string) const;


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

    double kappa;
  };



  // @sect3{Equation data}

  // Before we go on filling in the details of the main class, let us define
  // the equation data corresponding to the problem, i.e. initial values, as
  // well as a right hand side class. (We will reuse the initial conditions
  // also for the boundary values, which we simply keep constant.) We do so
  // using classes derived
  // from the Function class template that has been used many times before, so
  // the following should not look surprising. The only point of interest is
  // that we here have a complex-valued problem, so we have to provide the
  // second template argument of the Function class (which would otherwise
  // default to `double`). Furthermore, the return type of the `value()`
  // functions is then of course also complex.
  //
  // What precisely these functions return has been discussed at the end of
  // the Introduction section.
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
    static_assert(dim == 2, "This initial condition only works in 2d.");

    (void)component;
    Assert(component == 0, ExcIndexRange(component, 0, 1));

    const std::vector<Point<dim>> vortex_centers = {{0, -0.3},
                                                    {0, +0.3},
                                                    {+0.3, 0},
                                                    {-0.3, 0}};

    const double R = 0.1;
    const double alpha =
      1. / (std::pow(R, dim) * std::pow(numbers::PI, dim / 2.));

    double sum = 0;
    for (const auto &vortex_center : vortex_centers)
      {
        const Tensor<1, dim> distance = p - vortex_center;
        const double         r        = distance.norm();

        sum += alpha * std::exp(-(r * r) / (R * R));
      }

    return {std::sqrt(sum), 0.};
  }



  template <int dim>
  class Potential : public Function<dim>
  {
  public:
    Potential() = default;
    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;
  };



  template <int dim>
  double Potential<dim>::value(const Point<dim> & p,
                               const unsigned int component) const
  {
    (void)component;
    Assert(component == 0, ExcIndexRange(component, 0, 1));

    return (Point<dim>().distance(p) > 0.7 ? 1000 : 0);
  }



  // @sect3{Implementation of the <code>NonlinearSchroedingerEquation</code> class}

  // We start by specifying the implementation of the constructor
  // of the class. There is nothing of surprise to see here except
  // perhaps that we choose quadratic ($Q_2$) Lagrange elements --
  // the solution is expected to be smooth, so we choose a higher
  // polynomial degree than the bare minimum.
  template <int dim>
  NonlinearSchroedingerEquation<dim>::NonlinearSchroedingerEquation(int /*argc*/, char** /*argv*/)
    : fe(2)
    , dof_handler(triangulation)
    , time(0)
    , n_time_steps(32)
    , time_step(1. / n_time_steps)
    , timestep_number(0)
    , kappa(1)
  {}


  // @sect4{Setting up data structures and assembling matrices}

  // The next function is the one that sets up the mesh, DoFHandler, and
  // matrices and vectors at the beginning of the program, i.e. before the
  // first time step. The first few lines are pretty much standard if you've
  // read through the tutorial programs at least up to step-6:
  template <int dim>
  void NonlinearSchroedingerEquation<dim>::setup_system()
  {
    GridGenerator::hyper_cube(triangulation, -1, 1);
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

    constraints.close();
  }

  template <int dim>
  void NonlinearSchroedingerEquation<dim>::evaluate_spatial_rhs(
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
  }

  template <int dim>
  void NonlinearSchroedingerEquation<dim>::id_minus_tau_J_inverse(
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
  }

  // Next, we assemble the relevant matrices. The way we have written
  // the Crank-Nicolson discretization of the spatial step of the Strang
  // splitting (i.e., the second of the three partial steps in each time
  // step), we were led to the linear system
  // $\left[ -iM  +  \frac 14 k_{n+1} A + \frac 12 k_{n+1} W \right]
  //   \Psi^{(n,2)}
  //  =
  //  \left[ -iM  -  \frac 14 k_{n+1} A - \frac 12 k_{n+1} W \right]
  //   \Psi^{(n,1)}$.
  // In other words, there are two matrices in play here -- one for the
  // left and one for the right hand side. We build these matrices
  // separately. (One could avoid building the right hand side matrix
  // and instead just form the *action* of the matrix on $\Psi^{(n,1)}$
  // in each time step. This may or may not be more efficient, but
  // efficiency is not foremost on our minds for this program.)
  template <int dim>
  void NonlinearSchroedingerEquation<dim>::assemble_matrices()
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
    const Potential<dim>                 potential;

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        cell_mass_matrix = static_cast<value_type>(0);
        cell_matrix_jacobian = static_cast<value_type>(0);

        fe_values.reinit(cell);

        potential.value_list(fe_values.get_quadrature_points(),
                             potential_values);

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

		cell_matrix_jacobian(k,l) += -i * ( fe_values.shape_grad(k, q_index) *
						    fe_values.shape_grad(l, q_index) / 2.0 +
					          potential_values[q_index] *
						    fe_values.shape_value(k, q_index) *
						    fe_values.shape_value(l, q_index)) *
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


  // @sect4{Implementing the Strang splitting steps}

  // Having set up all data structures above, we are now in a position to
  // implement the partial steps that form the Strang splitting scheme. We
  // start with the half-step to advance the phase, and that is used as the
  // first and last part of each time step.
  //
  // To this end, recall that for the first half step, we needed to
  // compute
  // $\psi^{(n,1)} = e^{-i\kappa|\psi^{(n,0)}|^2 \tfrac
  //  12\Delta t} \; \psi^{(n,0)}$. Here, $\psi^{(n,0)}=\psi^{(n)}$ and
  //  $\psi^{(n,1)}$
  // are functions of space and correspond to the output of the previous
  // complete time step and the result of the first of the three part steps,
  // respectively. A corresponding solution must be computed for the third
  // of the part steps, i.e.
  // $\psi^{(n,3)} = e^{-i\kappa|\psi^{(n,2)}|^2 \tfrac
  //  12\Delta t} \; \psi^{(n,2)}$, where $\psi^{(n,3)}=\psi^{(n+1)}$ is
  // the result of the time step as a whole, and its input $\psi^{(n,2)}$ is
  // the result of the spatial step of the Strang splitting.
  //
  // An important realization is that while $\psi^{(n,0)}(\mathbf x)$ may be a
  // finite element function (i.e., is piecewise polynomial), this may not
  // necessarily be the case for the "rotated" function in which we have updated
  // the phase using the exponential factor (recall that the amplitude of that
  // function remains constant as part of that step). In other words, we could
  // *compute* $\psi^{(n,1)}(\mathbf x)$ at every point $\mathbf x\in\Omega$,
  // but we can't represent it on a mesh because it is not a piecewise
  // polynomial function. The best we can do in a discrete setting is to compute
  // a projection or interpolation. In other words, we can compute
  // $\psi_h^{(n,1)}(\mathbf x) = \Pi_h
  //     \left(e^{-i\kappa|\psi_h^{(n,0)}(\mathbf x)|^2 \tfrac 12\Delta t}
  //     \; \psi_h^{(n,0)}(\mathbf x) \right)$ where $\Pi_h$ is a projection or
  // interpolation operator. The situation is particularly simple if we
  // choose the interpolation: Then, all we need to compute is the value of
  // the right hand side *at the node points* and use these as nodal
  // values for the vector $\Psi^{(n,1)}$ of degrees of freedom. This is
  // easily done because evaluating the right hand side at node points
  // for a Lagrange finite element as used here requires us to only
  // look at a single (complex-valued) entry of the node vector. In other
  // words, what we need to do is to compute
  // $\Psi^{(n,1)}_j = e^{-i\kappa|\Psi^{(n,0)}_j|^2 \tfrac
  //  12\Delta t} \; \Psi^{(n,0)}_j$ where $j$ loops over all of the entries
  // of our solution vector. This is what the function below does -- in fact,
  // it doesn't even use separate vectors for $\Psi^{(n,0)}$ and $\Psi^{(n,1)}$,
  // but just updates the same vector as appropriate.
  template <int dim>
  void NonlinearSchroedingerEquation<dim>::do_half_phase_step(
      time_type /*t*/,                 //
      time_type             step_size, //
      const vector_type& yin,       //
      vector_type&       yout) {

    yout               = yin;
    const value_type i = {0, 1};
    for (auto& value : yout) {
      const double magnitude = std::abs(value);
      value = std::exp(-i * kappa * magnitude * magnitude * step_size) * value;
    }
  }

  // @sect4{Creating graphical output}

  // The last of the helper functions and classes we ought to discuss are the
  // ones that create graphical output. The result of running the half and full
  // steps for the local and spatial parts of the Strang splitting is that we
  // have updated the `solution` vector $\Psi^n$ to the correct value at the end
  // of each time step. Its entries contain complex numbers for the solution at
  // the nodes of the finite element mesh.
  //
  // Complex numbers are not easily visualized. We can output their real and
  // imaginary parts, i.e., the fields $\text{Re}(\psi_h^{(n)}(\mathbf x))$ and
  // $\text{Im}(\psi_h^{(n)}(\mathbf x))$, and that is exactly what the DataOut
  // class does when one attaches as complex-valued vector via
  // DataOut::add_data_vector() and then calls DataOut::build_patches(). That is
  // indeed what we do below.

  // But oftentimes we are not particularly interested in real and imaginary
  // parts of the solution vector, but instead in derived quantities such as the
  // magnitude $|\psi|$ and phase angle $\text{arg}(\psi)$ of the solution. In
  // the context of quantum systems such as here, the magnitude itself is not so
  // interesting, but instead it is the "amplitude", $|\psi|^2$ that is a
  // physical property: it corresponds to the probability density of finding a
  // particle in a particular place of state. The way to put computed quantities
  // into output files for visualization -- as used in numerous previous
  // tutorial programs -- is to use the facilities of the DataPostprocessor and
  // derived classes. Specifically, both the amplitude of a complex number and
  // its phase angles are scalar quantities, and so the DataPostprocessorScalar
  // class is the right tool to base what we want to do on.
  //
  // Consequently, what we do here is to implement two classes
  // `ComplexAmplitude` and `ComplexPhase` that compute for each point at which
  // DataOut decides to generate output, the amplitudes $|\psi_h|^2$ and phases
  // $\text{arg}(\psi_h)$ of the solution for visualization. There is a fair
  // amount of boiler-plate code below, with the only interesting parts of
  // the first of these two classes being how its `evaluate_vector_field()`
  // function computes the `computed_quantities` object.
  //
  // (There is also the rather awkward fact that the <a
  // href="https://en.cppreference.com/w/cpp/numeric/complex/norm">std::norm()</a>
  // function does not compute what one would naively imagine, namely $|\psi|$,
  // but returns $|\psi|^2$ instead. It's certainly quite confusing to have a
  // standard function mis-named in such a way...)
  namespace DataPostprocessors
  {
    template <int dim>
    class ComplexAmplitude : public DataPostprocessorScalar<dim>
    {
    public:
      ComplexAmplitude();

      virtual void evaluate_vector_field(
        const DataPostprocessorInputs::Vector<dim> &inputs,
        std::vector<Vector<double>> &computed_quantities) const override;
    };


    template <int dim>
    ComplexAmplitude<dim>::ComplexAmplitude()
      : DataPostprocessorScalar<dim>("Amplitude", update_values)
    {}


    template <int dim>
    void ComplexAmplitude<dim>::evaluate_vector_field(
      const DataPostprocessorInputs::Vector<dim> &inputs,
      std::vector<Vector<double>> &               computed_quantities) const
    {
      Assert(computed_quantities.size() == inputs.solution_values.size(),
             ExcDimensionMismatch(computed_quantities.size(),
                                  inputs.solution_values.size()));

      for (unsigned int q = 0; q < computed_quantities.size(); ++q)
        {
          Assert(computed_quantities[q].size() == 1,
                 ExcDimensionMismatch(computed_quantities[q].size(), 1));
          Assert(inputs.solution_values[q].size() == 2,
                 ExcDimensionMismatch(inputs.solution_values[q].size(), 2));

          const value_type psi(inputs.solution_values[q](0),
                                         inputs.solution_values[q](1));
          computed_quantities[q](0) = std::norm(psi);
        }
    }



    // The second of these postprocessor classes computes the phase angle
    // of the complex-valued solution at each point. In other words, if we
    // represent $\psi(\mathbf x,t)=r(\mathbf x,t) e^{i\varphi(\mathbf x,t)}$,
    // then this class computes $\varphi(\mathbf x,t)$. The function
    // <a
    // href="https://en.cppreference.com/w/cpp/numeric/complex/arg">std::arg</a>
    // does this for us, and returns the angle as a real number between $-\pi$
    // and $+\pi$.
    //
    // For reasons that we will explain in detail in the results section, we
    // do not actually output this value at each location where output is
    // generated. Rather, we take the maximum over all evaluation points of the
    // phase and then fill each evaluation point's output field with this
    // maximum -- in essence, we output the phase angle as a piecewise constant
    // field, where each cell has its own constant value. The reasons for this
    // will become clear once you read through the discussion further down
    // below.
    template <int dim>
    class ComplexPhase : public DataPostprocessorScalar<dim>
    {
    public:
      ComplexPhase();

      virtual void evaluate_vector_field(
        const DataPostprocessorInputs::Vector<dim> &inputs,
        std::vector<Vector<double>> &computed_quantities) const override;
    };


    template <int dim>
    ComplexPhase<dim>::ComplexPhase()
      : DataPostprocessorScalar<dim>("Phase", update_values)
    {}


    template <int dim>
    void ComplexPhase<dim>::evaluate_vector_field(
      const DataPostprocessorInputs::Vector<dim> &inputs,
      std::vector<Vector<double>> &               computed_quantities) const
    {
      Assert(computed_quantities.size() == inputs.solution_values.size(),
             ExcDimensionMismatch(computed_quantities.size(),
                                  inputs.solution_values.size()));

      double max_phase = -numbers::PI;
      for (unsigned int q = 0; q < computed_quantities.size(); ++q)
        {
          Assert(computed_quantities[q].size() == 1,
                 ExcDimensionMismatch(computed_quantities[q].size(), 1));
          Assert(inputs.solution_values[q].size() == 2,
                 ExcDimensionMismatch(inputs.solution_values[q].size(), 2));

          max_phase =
            std::max(max_phase,
                     std::arg(
                       value_type(inputs.solution_values[q](0),
                                            inputs.solution_values[q](1))));
        }

      for (auto &output : computed_quantities)
        output(0) = max_phase;
    }

  } // namespace DataPostprocessors


  // Having so implemented these post-processors, we create output as we always
  // do. As in many other time-dependent tutorial programs, we attach flags to
  // DataOut that indicate the number of the time step and the current
  // simulation time.
  template <int dim>
  void NonlinearSchroedingerEquation<dim>::output_results(std::string name) const
  {
    const DataPostprocessors::ComplexAmplitude<dim> complex_magnitude;
    const DataPostprocessors::ComplexPhase<dim>     complex_phase;

    DataOut<dim> data_out;

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "Psi");
    data_out.add_data_vector(solution, complex_magnitude);
    data_out.add_data_vector(solution, complex_phase);
    data_out.build_patches();

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
  // initial conditions onto finite element space; then iterate over all time
  // steps, and on each time step perform the three parts of the Strang
  // splitting method. Every tenth time step, we generate graphical output.
  // That's it.
  template <int dim>
  void NonlinearSchroedingerEquation<dim>::run()
  {
    setup_system();
    assemble_matrices();

    time = 0;
    VectorTools::interpolate(dof_handler, InitialValues<dim>(), solution);

    /* Define methods, operators and alpha for operator split */
    tostii::Exact<vector_type, time_type> half_stepper_method;
    tostii::runge_kutta_method            full_step_method{tostii::CRANK_NICOLSON};
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

    // std::string os_type{"A_3-3_c"};
    // std::string os_name{"Milne_2_2_c_i"};
    // auto        os_coeffs = tostii::os_complex.at(os_name);

    // tostii::OperatorSplitSingle<vector_type, time_type> os_stepper(
    //     solution,                                                      //
    //     std::vector<tostii::OSoperator<vector_type, time_type>>{half_stepper,  //
    //                                                     full_stepper}, //
    //     os_coeffs);

    std::string os_name{"Godunov"};
    auto        os_coeffs = tostii::os_method.at(os_name);
    tostii::OperatorSplitSingle<vector_type, time_type> os_stepper(
							   solution, //
        std::vector<tostii::OSoperator<vector_type, time_type>>{half_stepper,  //
							       full_stepper}, //
        os_coeffs);

    // Step 0 output:
    auto full_step_name{RK_method_enum_to_string(full_step_method)};
    output_results(os_name+"_Exact_"+full_step_name);

    // Main time loop
    for (unsigned int itime=0; itime <= n_time_steps; ++itime)
      {
        ++timestep_number;

        std::cout << "Time step " << timestep_number << " at t=" << time
                  << std::endl;

        time = os_stepper.evolve_one_time_step(time, time_step, solution);

        if (timestep_number % 1 == 0) {
          output_results(os_name+"_Exact_"+full_step_name);
	}
      }
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

      NonlinearSchroedingerEquation<2> nse(argc,argv);
      nse.run();
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
