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

 * Here we convert a complex-valued PDE to a system of real-valued PDEs
 * and solve it using the Operator Splitting method wiht tosii. 
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
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/base/index_set.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/solver_gmres.h>


#include <fstream>
#include <iostream>
#include <cmath>
#include <chrono>
// Include our separated time-integration library
// (has some overlap with deal.II/base/time_stepping.h)
#include <tostii/tostiiv2.h>


// Then the usual placing of all content of this program into a namespace and
// the importation of the deal.II namespace into the one we will work in:
namespace StepOS
{
  using namespace dealii;

//   using value_type   = std::complex<double>;
  using value_type   = double;
  using matrix_type  = SparseMatrix<value_type>;

  // Types
//   using time_type = std::complex<double>;
//   #define timereal time.real()
  using time_type = double;
  #define timereal time


  template <int dim>
  class NonlinearRealSystem
  {
  public:
    NonlinearRealSystem(int argc, char* argv[]);
    void run();

  private:
    void setup_system();
    void assemble_matrices();
    void evaluate_diffusion(const time_type, const BlockVector<value_type>&, BlockVector<value_type>&);
    void do_reaction_step(const time_type, const BlockVector<value_type>&, BlockVector<value_type>&);
    void id_minus_tau_J_diffusion_inverse(const time_type, const time_type, 
                                          const BlockVector<value_type>& , BlockVector<value_type>&);
    void output_results(std::string, const time_type) const;

    void convert_block_to_sparse(const BlockSparseMatrix<value_type> &, SparseMatrix<value_type> &);
    void flatten_block_vector(const BlockVector<value_type> &, Vector<value_type> &);
    void reconstruct_block_vector(const Vector<value_type> &, BlockVector<value_type> &);
    void solve_with_sparse_umfpack(const SparseMatrix<value_type> &, const Vector<value_type> &, Vector<value_type> &);


    Triangulation<dim> triangulation;
    FESystem<dim>      fe;
    DoFHandler<dim>    dof_handler;

    AffineConstraints<value_type> constraints;

    BlockSparsityPattern sparsity_pattern;

    BlockSparseMatrix<value_type> mass_matrix;
    BlockSparseMatrix<value_type> system_diffusion;
    BlockSparseMatrix<value_type> mass_minus_tau_diffusion;


    BlockVector<value_type> solution;
    
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
      : Function<dim, value_type>(2){}

    virtual value_type
    value(const Point<dim> &p, const unsigned int component = 0) const override;
  };



  template <int dim>
  value_type
InitialValues<dim>::value(const Point<dim> &p,
                           const unsigned int component) const
{
    const double x = p[0];

    if (component == 0) // Component 0: 'u'
        // if you are using complex-valued OS methods you need to return a complex value
        //return {std::exp(-0.5 * (x * x)), 0.0}; // Initial value for 'u'
        return std::exp(-0.5 * (x * x)); // Initial value for 'u'
    else                // Component 1: 'v'
        //return {0.0, 0.0}; // Initial value for 'v'
        return 0.0;     // No initial value for 'v'
}




  // @sect3{Implementation of the <code>NonlinearRealSystem</code> class}

  template <int dim>
  NonlinearRealSystem<dim>::NonlinearRealSystem(int /*argc*/, char** /*argv*/)
    : fe(FE_Q<dim>(2), 2), // Degree 2, 2 components
     dof_handler(triangulation),
     time(0),
     n_time_steps(10),
     time_step(1.0 / n_time_steps),
     timestep_number(0)
    {}


  // @sect4{Setting up data structures and assembling matrices}

  // The next function is the one that sets up the mesh, DoFHandler, and
  // matrices and vectors at the beginning of the program, i.e. before the      
  // first time step. The first few lines are pretty much standard if you've
  // read through the tutorial programs at least up to step-6:
    template <int dim>
    void NonlinearRealSystem<dim>::setup_system(){
        // Generate the mesh and distribute DoFs
        GridGenerator::hyper_cube(triangulation, -10., 10.);
        triangulation.refine_global(12);

        dof_handler.distribute_dofs(fe);
        DoFRenumbering::component_wise(dof_handler);

        // Count DoFs per block
        const std::vector<types::global_dof_index> dofs_per_component = 
        DoFTools::count_dofs_per_fe_component(dof_handler);

        const unsigned int n_u = dofs_per_component[0];
        const unsigned int n_v = dofs_per_component[1];

        constraints.clear();

        DoFTools::make_hanging_node_constraints(dof_handler, constraints);

        std::vector<GridTools::PeriodicFacePair<typename DoFHandler<dim>::cell_iterator>> periodic_faces;
        GridTools::collect_periodic_faces(dof_handler, 0, 1, 0, periodic_faces);

        DoFTools::make_periodicity_constraints<dim, dim>(
            periodic_faces, constraints);

        constraints.close();

        std::cout << "Number of active cells: " << triangulation.n_active_cells()
                  << std::endl
                  << "Total number of cells: " << triangulation.n_cells()
                  << std::endl
                  << "Number of degrees of freedom: " << dof_handler.n_dofs()
                  << " (" << n_u << '+' << n_v << ')' << std::endl;



        const std::vector<types::global_dof_index> block_sizes = {n_u, n_v};
        BlockDynamicSparsityPattern                dsp(block_sizes, block_sizes);
        DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints);


        sparsity_pattern.copy_from(dsp);

        // Initialize matrices
        mass_matrix.reinit(sparsity_pattern);
        system_diffusion.reinit(sparsity_pattern);
        mass_minus_tau_diffusion.reinit(sparsity_pattern);


        solution.reinit(block_sizes);
        
    }


template <int dim>
void NonlinearRealSystem<dim>::assemble_matrices(){
    mass_matrix = 0.0;
    system_diffusion = 0.0;

    // Quadrature rule for integration
    const QGauss<dim> quadrature_formula(fe.degree + 1);

    // FEValues to access shape functions and derivatives
    FEValues<dim> fe_values(fe, quadrature_formula,
                            update_values | update_gradients | update_JxW_values);

    const unsigned int dofs_per_cell = fe.dofs_per_cell; // Degrees of freedom per cell
    const unsigned int n_q_points = quadrature_formula.size(); // Quadrature points per cell

    // Local matrices for assembling
    FullMatrix<value_type> cell_mass_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<value_type> cell_system_matrix(dofs_per_cell, dofs_per_cell);

    // Vector for local degree of freedom indices
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
        cell_mass_matrix = 0.0;
        cell_system_matrix = 0.0;

        fe_values.reinit(cell);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
            const double JxW = fe_values.JxW(q);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
                const unsigned int component_i = fe.system_to_component_index(i).first;
                const double phi_i = fe_values.shape_value(i, q);
                const Tensor<1, dim> grad_phi_i = fe_values.shape_grad(i, q);

                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                    const unsigned int component_j = fe.system_to_component_index(j).first;
                    const double phi_j = fe_values.shape_value(j, q);
                    const Tensor<1, dim> grad_phi_j = fe_values.shape_grad(j, q);

                    // Mass matrix contributions (block diagonal)
                    if (component_i == component_j)
                    {
                        cell_mass_matrix(i, j) += phi_i * phi_j * JxW;
                    }

                    // Diffusion matrix contributions (coupled blocks)
                    double diffusion_entry = 0.0;

                    if (component_i == 0 && component_j == 0)
                    {
                        // \Delta u term in u equation
                        diffusion_entry -= grad_phi_i * grad_phi_j;
                    }
                    else if (component_i == 0 && component_j == 1)
                    {
                        // 0.5 \Delta v term in u equation
                        diffusion_entry += 0.5 * (grad_phi_i * grad_phi_j);
                    }
                    else if (component_i == 1 && component_j == 0)
                    {
                        // -0.5 \Delta u term in v equation
                        diffusion_entry -= 0.5 * (grad_phi_i * grad_phi_j);
                    }
                    else if (component_i == 1 && component_j == 1)
                    {
                        // \Delta v term in v equation
                        diffusion_entry -= grad_phi_i * grad_phi_j;
                    }

                    cell_system_matrix(i, j) += diffusion_entry * JxW;
                }
            }
        }

        cell->get_dof_indices(local_dof_indices);

        constraints.distribute_local_to_global(cell_mass_matrix,
                                               local_dof_indices,
                                               mass_matrix);

        constraints.distribute_local_to_global(cell_system_matrix,
                                               local_dof_indices,
                                               system_diffusion);
    }

    }

    template <int dim>
    void NonlinearRealSystem<dim>::evaluate_diffusion(
        const time_type /*time*/,  
    const BlockVector<value_type>& yin, 
    BlockVector<value_type>& yout) {
        //We need to flatten the block system to a monolithic system
        // for using the UMFPACK solver
        // Step 1: Get block sizes and compute total size
        const unsigned int n_blocks = yin.n_blocks();
        std::vector<types::global_dof_index> block_sizes(n_blocks);
        for (unsigned int i = 0; i < n_blocks; ++i)
            block_sizes[i] = yin.block(i).size();

        // Compute block offsets
        std::vector<types::global_dof_index> block_offsets(n_blocks + 1);
        block_offsets[0] = 0;
        for (unsigned int i = 0; i < n_blocks; ++i)
            block_offsets[i+1] = block_offsets[i] + block_sizes[i];

        const types::global_dof_index N = block_offsets[n_blocks];

        // Step 2: Create monolithic sparsity pattern
        DynamicSparsityPattern dsp(N, N);

        // Copy sparsity patterns from blocks into monolithic sparsity pattern
        for (unsigned int i = 0; i < system_diffusion.n_block_rows(); ++i)
        {
            for (unsigned int j = 0; j < system_diffusion.n_block_cols(); ++j)
            {
                const SparsityPattern &block_sp = system_diffusion.block(i, j).get_sparsity_pattern();

                const types::global_dof_index row_offset = block_offsets[i];
                const types::global_dof_index col_offset = block_offsets[j];

                for (types::global_dof_index k = 0; k < block_sp.n_rows(); ++k)
                {
                    for (SparsityPattern::iterator it = block_sp.begin(k);
                        it != block_sp.end(k); ++it)
                    {
                        const types::global_dof_index row = row_offset + k;
                        const types::global_dof_index col = col_offset + it->column();
                        dsp.add(row, col);
                    }
                }
            }
        }

        SparsityPattern monolithic_sparsity_pattern;
        monolithic_sparsity_pattern.copy_from(dsp);

        // Step 3: Create monolithic matrices
        SparseMatrix<value_type> monolithic_mass_matrix;
        monolithic_mass_matrix.reinit(monolithic_sparsity_pattern);

        SparseMatrix<value_type> monolithic_system_diffusion;
        monolithic_system_diffusion.reinit(monolithic_sparsity_pattern);

        // Copy data from block matrices into monolithic matrices
        // For mass_matrix
        for (unsigned int i = 0; i < mass_matrix.n_block_rows(); ++i)
        {
            for (unsigned int j = 0; j < mass_matrix.n_block_cols(); ++j)
            {
                const SparseMatrix<value_type> &block_matrix = mass_matrix.block(i, j);

                const types::global_dof_index row_offset = block_offsets[i];
                const types::global_dof_index col_offset = block_offsets[j];

                for (types::global_dof_index k = 0; k < block_matrix.m(); ++k)
                {
                    for (SparseMatrix<value_type>::const_iterator it = block_matrix.begin(k);
                        it != block_matrix.end(k); ++it)
                    {
                        const types::global_dof_index row = row_offset + k;
                        const types::global_dof_index col = col_offset + it->column();
                        monolithic_mass_matrix.add(row, col, it->value());
                    }
                }
            }
        }

        // For system_diffusion
        for (unsigned int i = 0; i < system_diffusion.n_block_rows(); ++i)
        {
            for (unsigned int j = 0; j < system_diffusion.n_block_cols(); ++j)
            {
                const SparseMatrix<value_type> &block_matrix = system_diffusion.block(i, j);

                const types::global_dof_index row_offset = block_offsets[i];
                const types::global_dof_index col_offset = block_offsets[j];

                for (types::global_dof_index k = 0; k < block_matrix.m(); ++k)
                {
                    for (SparseMatrix<value_type>::const_iterator it = block_matrix.begin(k);
                        it != block_matrix.end(k); ++it)
                    {
                        const types::global_dof_index row = row_offset + k;
                        const types::global_dof_index col = col_offset + it->column();
                        monolithic_system_diffusion.add(row, col, it->value());
                    }
                }
            }
        }

        // Step 4: Flatten yin into monolithic_yin
        Vector<value_type> monolithic_yin(N);
        {
            unsigned int offset = 0;
            for (unsigned int b = 0; b < n_blocks; ++b)
            {
                const auto& block = yin.block(b);
                for (unsigned int i = 0; i < block.size(); ++i)
                    monolithic_yin[offset + i] = block[i];
                offset += block_sizes[b];
            }
        }

        // Step 5: Compute rhs = monolithic_system_diffusion * monolithic_yin
        Vector<value_type> monolithic_rhs(N);
        monolithic_rhs = static_cast<value_type>(0);
        monolithic_system_diffusion.vmult(monolithic_rhs, monolithic_yin);

        
        // Step 6: Solve monolithic_mass_matrix * monolithic_yout = monolithic_rhs

        SparseDirectUMFPACK inverse_mass_matrix;
        inverse_mass_matrix.solve(monolithic_mass_matrix, monolithic_rhs);

        Vector<value_type> monolithic_yout(N);
        monolithic_yout = monolithic_rhs;
        

        // Step 7: Reconstruct yout from monolithic_yout
        yout.reinit(yin);
        {
            unsigned int offset = 0;
            for (unsigned int b = 0; b < n_blocks; ++b)
            {
                auto& block = yout.block(b);
                for (unsigned int i = 0; i < block.size(); ++i)
                    block[i] = monolithic_yout[offset + i];
                offset += block_sizes[b];
            }
        }

        // Step 8: Apply Constraints
        constraints.distribute(yout);
    }


    template <int dim>
    void NonlinearRealSystem<dim>::id_minus_tau_J_diffusion_inverse(
    const time_type /*time*/,  
    const time_type tau, 
    const BlockVector<value_type>& yin, 
    BlockVector<value_type>& yout) {

        //Again, we need to flatten the block system to a monolithic system
        // Step 1: Get block sizes and compute total size
        const unsigned int n_blocks = yin.n_blocks();
        std::vector<types::global_dof_index> block_sizes(n_blocks);
        for (unsigned int i = 0; i < n_blocks; ++i)
            block_sizes[i] = yin.block(i).size();

        // Compute block offsets
        std::vector<types::global_dof_index> block_offsets(n_blocks + 1);
        block_offsets[0] = 0;
        for (unsigned int i = 0; i < n_blocks; ++i)
            block_offsets[i+1] = block_offsets[i] + block_sizes[i];

        const types::global_dof_index N = block_offsets[n_blocks];

        // Step 2: Create monolithic sparsity pattern
        DynamicSparsityPattern dsp(N, N);

        // Copy sparsity patterns from blocks into monolithic sparsity pattern
        for (unsigned int i = 0; i < mass_matrix.n_block_rows(); ++i)
        {
            for (unsigned int j = 0; j < mass_matrix.n_block_cols(); ++j)
            {
                const SparsityPattern &block_sp = mass_matrix.block(i, j).get_sparsity_pattern();

                const types::global_dof_index row_offset = block_offsets[i];
                const types::global_dof_index col_offset = block_offsets[j];

                for (types::global_dof_index k = 0; k < block_sp.n_rows(); ++k)
                {
                    for (SparsityPattern::iterator it = block_sp.begin(k);
                        it != block_sp.end(k); ++it)
                    {
                        const types::global_dof_index row = row_offset + k;
                        const types::global_dof_index col = col_offset + it->column();
                        dsp.add(row, col);
                    }
                }
            }
        }

        SparsityPattern monolithic_sparsity_pattern;
        monolithic_sparsity_pattern.copy_from(dsp);

        // Step 3: Create monolithic matrices
        SparseMatrix<value_type> monolithic_mass_matrix;
        monolithic_mass_matrix.reinit(monolithic_sparsity_pattern);

        SparseMatrix<value_type> monolithic_system_diffusion;
        monolithic_system_diffusion.reinit(monolithic_sparsity_pattern);

        // Copy data from block matrices into monolithic matrices
        // For mass_matrix
        for (unsigned int i = 0; i < mass_matrix.n_block_rows(); ++i)
        {
            for (unsigned int j = 0; j < mass_matrix.n_block_cols(); ++j)
            {
                const SparseMatrix<value_type> &block_matrix = mass_matrix.block(i, j);

                const types::global_dof_index row_offset = block_offsets[i];
                const types::global_dof_index col_offset = block_offsets[j];

                for (types::global_dof_index k = 0; k < block_matrix.m(); ++k)
                {
                    for (SparseMatrix<value_type>::const_iterator it = block_matrix.begin(k);
                        it != block_matrix.end(k); ++it)
                    {
                        const types::global_dof_index row = row_offset + k;
                        const types::global_dof_index col = col_offset + it->column();
                        monolithic_mass_matrix.add(row, col, it->value());
                    }
                }
            }
        }

        // For system_diffusion
        for (unsigned int i = 0; i < system_diffusion.n_block_rows(); ++i)
        {
            for (unsigned int j = 0; j < system_diffusion.n_block_cols(); ++j)
            {
                const SparseMatrix<value_type> &block_matrix = system_diffusion.block(i, j);

                const types::global_dof_index row_offset = block_offsets[i];
                const types::global_dof_index col_offset = block_offsets[j];

                for (types::global_dof_index k = 0; k < block_matrix.m(); ++k)
                {
                    for (SparseMatrix<value_type>::const_iterator it = block_matrix.begin(k);
                        it != block_matrix.end(k); ++it)
                    {
                        const types::global_dof_index row = row_offset + k;
                        const types::global_dof_index col = col_offset + it->column();
                        monolithic_system_diffusion.add(row, col, it->value());
                    }
                }
            }
        }

        // Step 4: Compute monolithic_mass_minus_tau_diffusion = monolithic_mass_matrix - tau * monolithic_system_diffusion
        SparseMatrix<value_type> monolithic_mass_minus_tau_diffusion;
        monolithic_mass_minus_tau_diffusion.reinit(monolithic_sparsity_pattern);

        // monolithic_mass_minus_tau_diffusion = monolithic_mass_matrix - tau * monolithic_system_diffusion
        monolithic_mass_minus_tau_diffusion.copy_from(monolithic_mass_matrix);
        monolithic_system_diffusion *= tau; // monolithic_system_diffusion = tau * monolithic_system_diffusion
        monolithic_mass_minus_tau_diffusion.add(-1.0, monolithic_system_diffusion);

        // Step 5: Flatten yin into monolithic_yin
        Vector<value_type> monolithic_yin(N);
        {
            unsigned int offset = 0;
            for (unsigned int b = 0; b < n_blocks; ++b)
            {
                const auto& block = yin.block(b);
                for (unsigned int i = 0; i < block.size(); ++i)
                    monolithic_yin[offset + i] = block[i];
                offset += block_sizes[b];
            }
        }

        // Step 6: Compute rhs = monolithic_mass_matrix * monolithic_yin
        Vector<value_type> monolithic_rhs(N);
        monolithic_rhs = static_cast<value_type>(0);
        monolithic_mass_matrix.vmult(monolithic_rhs, monolithic_yin);

        // Step 7: Solve monolithic_mass_minus_tau_diffusion * monolithic_yout = monolithic_rhs

        SparseDirectUMFPACK inverse_mass_minus_tau_Jacobian;
        inverse_mass_minus_tau_Jacobian.initialize(monolithic_mass_minus_tau_diffusion);
        
        Vector<value_type> monolithic_yout(N);
        inverse_mass_minus_tau_Jacobian.solve(monolithic_mass_minus_tau_diffusion, monolithic_rhs);
        monolithic_yout = monolithic_rhs;

        // Step 8: Reconstruct yout from monolithic_yout
        yout.reinit(yin);
        {
            unsigned int offset = 0;
            for (unsigned int b = 0; b < n_blocks; ++b)
            {
                auto& block = yout.block(b);
                for (unsigned int i = 0; i < block.size(); ++i)
                    block[i] = monolithic_yout[offset + i];
                offset += block_sizes[b];
            }
        }

        // Step 9: Apply Constraints
        constraints.distribute(yout);

    }


  template <int dim>
  void NonlinearRealSystem<dim>::do_reaction_step(
      time_type /*t*/,                 //
      const BlockVector<value_type>& yin,       //
            BlockVector<value_type>&       yout) {

        auto& U = yin.block(0); // u
        auto& V = yin.block(1); // v

        auto& U_prime = yout.block(0); // u'
        auto& V_prime = yout.block(1); // v'

         const unsigned int n = U.size();
        for (unsigned int i = 0; i < n; ++i){
          U_prime[i] = U[i] - U[i] * U[i] * U[i] + 3.0 * U[i] * V[i] * V[i] - 3.0 * U[i] * U[i] * V[i] + V[i] * V[i] * V[i]; 
          V_prime[i] = V[i] + U[i] * U[i] * U[i] - 3.0 * V[i] * U[i] * U[i] - 3.0 * V[i] * V[i] * U[i] + V[i] * V[i] * V[i];
        }
    }


  // We create output as we always
  // do. As in many other time-dependent tutorial programs, we attach flags to
  // DataOut that indicate the number of the time step and the current
  // simulation time.

    template <int dim>
    void NonlinearRealSystem<dim>::output_results(std::string name, const time_type time) const{
        DataOut<dim> data_out;

        data_out.attach_dof_handler(dof_handler);

        // Flatten block vectors into a single vector
        Vector<value_type> flattened_solution(dof_handler.n_dofs());
        unsigned int offset = 0;

        for (unsigned int b = 0; b < solution.n_blocks(); ++b)
        {
            for (unsigned int i = 0; i < solution.block(b).size(); ++i)
            {
                flattened_solution[offset + i] = solution.block(b)[i];
            }
            offset += solution.block(b).size();
        }

        // Add the flattened solution to DataOut
        data_out.add_data_vector(flattened_solution, "solution");

        data_out.build_patches();

        // Set precision for VTU output
        data_out.set_flags(DataOutBase::VtkFlags(time, timestep_number));

        const std::string filename =
            "solution_" + Utilities::int_to_string(n_time_steps) + "_"
            + name + "-" + Utilities::int_to_string(timestep_number, 4) + ".vtu";
        std::ofstream output(filename);

        output.precision(16); // Ensure precision is double
        data_out.write_vtu(output);
    }


  // @sect4{Running the simulation}


  template <int dim>
  void NonlinearRealSystem<dim>::run(){

        // Start timing here
        auto start_time = std::chrono::high_resolution_clock::now();

        setup_system();
        assemble_matrices();

        time = 0;

        VectorTools::interpolate(dof_handler, InitialValues<dim>(), solution);
        constraints.distribute(solution);


        /* Define methods, operators and alpha for operator split */
        tostii::runge_kutta_method            diffusion_step_method{tostii::SDIRK_TWO_STAGES};
        tostii::runge_kutta_method            reaction_step_method{tostii::HEUN2};
        tostii::ImplicitRungeKutta<BlockVector<value_type>, time_type>  diffusion_stepper_method(diffusion_step_method);
        tostii::ExplicitRungeKutta<BlockVector<value_type>, time_type>  reaction_stepper_method(reaction_step_method);

        /* Define OSoperators to use in the operator split stepper */
        tostii::OSoperator<BlockVector<value_type>, time_type> diffusion_stepper{
            &diffusion_stepper_method,
            [this](const time_type t,  //
                    const BlockVector<value_type>& yin, //
                    BlockVector<value_type>&       yout) {this->evaluate_diffusion(t, yin, yout);},
            [this](const time_type    t,   //
                    const time_type    dt,  //
                    const BlockVector<value_type>& yin, //
                    BlockVector<value_type>&       yout) {this->id_minus_tau_J_diffusion_inverse(t, dt, yin, yout);}
                };

        tostii::OSoperator<BlockVector<value_type>, time_type> reaction_stepper{
            &reaction_stepper_method,
            [this](const time_type    t,   //
                    const BlockVector<value_type>& yin, //
                    BlockVector<value_type>& yout) { this->do_reaction_step(t, yin, yout); },
            [this](const time_type    /*t*/,   //
                    const time_type    /*dt*/,  //`
                    const BlockVector<value_type>& /*yin*/, //
                    BlockVector<value_type>&       /*yout*/) {return;}
                };
            
            std::vector<tostii::OSmask>      os_mask{{0, 1}, {0, 1}};

        //  std::string os_name{"Milne_2_2_c_i"};
        // std::string os_name{"A_3_3_c"};
        // std::string os_name{"Yoshida_c"};
        // auto        os_coeffs = tostii::os_complex.at(os_name);

        // tostii::OperatorSplit<BlockVector<value_type>, time_type> os_stepper(
        //     solution,                                                      //
        //     std::vector<tostii::OSoperator<BlockVector<value_type>, time_type>>{
        //         reaction_stepper, diffusion_stepper}, //
        //                 os_coeffs, //
        //                 os_mask //
        //                 );

        std::string os_name{"Strang"};
        auto        os_coeffs = tostii::os_method.at(os_name);
        tostii::OperatorSplit<BlockVector<value_type>, time_type> os_stepper(
                                    solution, //
            std::vector<tostii::OSoperator<BlockVector<value_type>, time_type>>{  //
                                        reaction_stepper, diffusion_stepper}, //
                        os_coeffs, //
                        os_mask //
                        );

        // Step 0 output:
        output_results(os_name, time);
        auto diffusion_step_name{RK_method_enum_to_string(diffusion_step_method)};
        auto reaction_step_name{RK_method_enum_to_string(reaction_step_method)};


          output_results(os_name+diffusion_step_name+reaction_step_name, 0);

        // Main time loop
        for (unsigned int itime=1; itime <= n_time_steps; ++itime)
        {
            ++timestep_number;

            time = os_stepper.evolve_one_time_step(time, time_step, solution);

          std::cout << "Time step " << timestep_number << " at t=" << time
                    << std::endl;

            if (timestep_number % 10 == 0) {
              output_results(os_name+diffusion_step_name+reaction_step_name, time);
            }
        }

        // End timing here
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;

        std::cout << "Total execution time: " << elapsed.count() << " seconds" << std::endl;

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
      NonlinearRealSystem<1> pde(argc,argv);
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
