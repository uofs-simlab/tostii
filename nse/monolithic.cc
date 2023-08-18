#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>

#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
 
#include <deal.II/sundials/kinsol.h>

#include <Sacado.hpp>

#include <deal.II/distributed/tria.h>

#include <fstream>
#include <iostream>

namespace LA
{
	using namespace dealii::LinearAlgebraPETSc;
}

namespace Monolithic
{
	using namespace dealii;

	template<int dim>
	class InitialValues
		: public Function<dim>
	{
	public:
		InitialValues();
		virtual double value(const Point<dim>& p, const unsigned int component = 0) const override;
	};

	template<int dim>
	InitialValues<dim>::InitialValues()
		: Function<dim>(2)
	{ }

	template<int dim>
	double InitialValues<dim>::value(const Point<dim>& p, const unsigned int component) const
	{
		static_assert(dim == 2, "This initial condition only works in 2d.");
		Assert(component < 2, ExcIndexRange(component, 0, 2));

		// imaginary part
		if (component == 1) return 0.;

		static const std::array<Point<dim>, 4> vortex_centers = { {
			{ 0., -0.3 },
			{ 0., +0.3 },
			{ +0.3, 0. },
			{ -0.3, 0. }
		} };

		const double R = 0.1;
		const double alpha = 1. / (std::pow(R, dim) * std::pow(numbers::PI, dim / 2.));

		// real part
		double sum = 0.;
		for (const auto& vortex_center : vortex_centers)
		{
			const Tensor<1, dim> distance = p - vortex_center;
			const double r = distance.norm();

			sum += alpha * std::exp(-(r * r) / (R * R));
		}

		return std::sqrt(sum);
	}

	template<int dim>
	class Potential
		: public Function<dim>
	{
	public:
		Potential();
		virtual double value(const Point<dim>& p, const unsigned int component = 0) const override;
		virtual void value_list(const std::vector<Point<dim>>& points, std::vector<double>& values,
			const unsigned int component = 0) const override;
	};

	template<int dim>
	Potential<dim>::Potential()
		: Function<dim>(1)
	{ }

	template<int dim>
	double Potential<dim>::value(const Point<dim>& p, const unsigned int component) const
	{
		(void)component;
		Assert(component == 0, ExcIndexRange(component, 0, 1));

		return Point<dim>().distance(p) > 0.7 ? 1000. : 0.;
	}

	template<int dim>
	void Potential<dim>::value_list(const std::vector<Point<dim>>& points, std::vector<double>& values,
		const unsigned int component) const
	{
		AssertDimension(points.size(), values.size());

		for (unsigned int i = 0; i < points.size(); ++i)
		{
			values[i] = value(points[i], component);
		}
	}

	namespace DataPostprocessors
	{
		using std::string_literals::operator""s;

		template<int dim>
		class ComplexRealPart
			: public DataPostprocessorScalar<dim>
		{
		public:
			ComplexRealPart(const char* name);

			virtual void evaluate_vector_field(const DataPostprocessorInputs::Vector<dim>& inputs,
				std::vector<Vector<double>>& computed_quantities) const override;
		};

		template<int dim>
		ComplexRealPart<dim>::ComplexRealPart(const char* name)
			: DataPostprocessorScalar<dim>(name + "_Re"s, update_values)
		{

		}

		template<int dim>
		void ComplexRealPart<dim>::evaluate_vector_field(const DataPostprocessorInputs::Vector<dim>& inputs,
			std::vector<Vector<double>>& computed_quantities) const
		{
			AssertDimension(computed_quantities.size(), inputs.solution_values.size());

			for (unsigned int q = 0; q < computed_quantities.size(); ++q)
			{
				AssertDimension(computed_quantities[q].size(), 1);
				AssertDimension(inputs.solution_values[q].size(), 2);

				computed_quantities[q](0) = inputs.solution_values[q](0);
			}
		}

		template<int dim>
		class ComplexImagPart
			: public DataPostprocessorScalar<dim>
		{
		public:
			ComplexImagPart(const char* name);

			virtual void evaluate_vector_field(const DataPostprocessorInputs::Vector<dim>& inputs,
				std::vector<Vector<double>>& computed_quantities) const override;
		};

		template<int dim>
		ComplexImagPart<dim>::ComplexImagPart(const char* name)
			: DataPostprocessorScalar<dim>(name + "_Im"s, update_values)
		{

		}

		template<int dim>
		void ComplexImagPart<dim>::evaluate_vector_field(const DataPostprocessorInputs::Vector<dim>& inputs,
			std::vector<Vector<double>>& computed_quantities) const
		{
			AssertDimension(computed_quantities.size(), inputs.solution_values.size());

			for (unsigned int q = 0; q < computed_quantities.size(); ++q)
			{
				AssertDimension(computed_quantities[q].size(), 1);
				AssertDimension(inputs.solution_values[q].size(), 2);

				computed_quantities[q](0) = inputs.solution_values[q](1);
			}
		}

		template<int dim>
		class ComplexAmplitude
			: public DataPostprocessorScalar<dim>
		{
		public:
			ComplexAmplitude(const char* name);

			virtual void evaluate_vector_field(const DataPostprocessorInputs::Vector<dim>& inputs,
				std::vector<Vector<double>>& computed_quantities) const override;
		};

		template<int dim>
		ComplexAmplitude<dim>::ComplexAmplitude(const char* name)
			: DataPostprocessorScalar<dim>(name + "_Amplitude"s, update_values)
		{

		}

		template<int dim>
		void ComplexAmplitude<dim>::evaluate_vector_field(const DataPostprocessorInputs::Vector<dim>& inputs,
			std::vector<Vector<double>>& computed_quantities) const
		{
			AssertDimension(computed_quantities.size(), inputs.solution_values.size());

			for (unsigned int q = 0; q < computed_quantities.size(); ++q)
			{
				AssertDimension(computed_quantities[q].size(), 1);
				AssertDimension(inputs.solution_values[q].size(), 2);

				const std::complex<double> psi(inputs.solution_values[q](0), inputs.solution_values[q](1));
				computed_quantities[q](0) = std::norm(psi);
			}
		}

		template<int dim>
		class ComplexPhase
			: public DataPostprocessorScalar<dim>
		{
		public:
			ComplexPhase(const char* name);

			virtual void evaluate_vector_field(const DataPostprocessorInputs::Vector<dim>& inputs,
				std::vector<Vector<double>>& computed_quantities) const override;
		};

		template<int dim>
		ComplexPhase<dim>::ComplexPhase(const char* name)
			: DataPostprocessorScalar<dim>(name + "_Phase"s, update_values)
		{

		}

		template<int dim>
		void ComplexPhase<dim>::evaluate_vector_field(const DataPostprocessorInputs::Vector<dim>& inputs,
			std::vector<Vector<double>>& computed_quantities) const
		{
			AssertDimension(computed_quantities.size(), inputs.solution_values.size());

			double max_phase = -numbers::PI;
			for (unsigned int q = 0; q < computed_quantities.size(); ++q)
			{
				AssertDimension(computed_quantities[q].size(), 1);
				AssertDimension(inputs.solution_values[q].size(), 2);

				const std::complex<double> value(inputs.solution_values[q](0), inputs.solution_values[q](1));
				max_phase = std::max(max_phase, std::arg(value));
			}

			for (auto& output : computed_quantities)
			{
				output(0) = max_phase;
			}
		}
	}

	template<int dim>
	class NonlinearSchroedingerEquation
	{
	public:
		NonlinearSchroedingerEquation(int argc, char* argv[]);
		void run();

	private:
		void setup_system();
		void compute_residual(const LA::MPI::Vector& evaluation_point, LA::MPI::Vector& residual);
		void assemble_cell_term(const typename DoFHandler<dim>::active_cell_iterator& cell,
			FEValues<dim>& fe_v, std::vector<types::global_dof_index>& dof_indices,
			const LA::MPI::Vector& evaluation_point, LA::MPI::Vector& residual);
		void factorize_jacobian();
		void solve(const LA::MPI::Vector& rhs, LA::MPI::Vector& solution, const double tolerance);
		void output_results() const;

		MPI_Comm mpi_communicator;

		ConditionalOStream pcout;
		TimerOutput computing_timer;

		parallel::distributed::Triangulation<dim> triangulation;
		DoFHandler<dim> dof_handler;

		IndexSet locally_owned_dofs;
		IndexSet locally_relevant_dofs;

		const FESystem<dim> fe;
		const QGauss<dim> quadrature;

		AffineConstraints<double> constraints;

		LA::MPI::Vector old_solution;
		LA::MPI::Vector current_solution;
		
		SparsityPattern sparsity_pattern;
		LA::MPI::SparseMatrix jacobian_matrix;

		double time;
		double time_step;
		unsigned int timestep_number;

		unsigned int refinement_level;
		unsigned int n_time_steps;
		double tolerance;

		double kappa;
	};

	template<int dim>
	NonlinearSchroedingerEquation<dim>::NonlinearSchroedingerEquation(int argc, char* argv[])
		: mpi_communicator(MPI_COMM_WORLD), pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0),
		computing_timer(mpi_communicator, pcout, TimerOutput::never, TimerOutput::wall_times),
		triangulation(mpi_communicator),
		dof_handler(triangulation), fe(FE_Q<dim>(2), 2), quadrature(fe.degree + 1),
		time(0.), timestep_number(0),
		kappa(1.)
	{
		if (argc != 4)
		{
			using std::string_literals::operator""s;
			throw std::invalid_argument("usage: "s + argv[0] + " refinement_level n_time_steps tolerance");
		}

		char* str_end;
		refinement_level = std::strtoul(argv[1], &str_end, 0);
		if (str_end != argv[1] + std::strlen(argv[1]))
		{
			throw std::invalid_argument("refinement_level: integer expected");
		}

		n_time_steps = std::strtoul(argv[2], &str_end, 0);
		if (str_end != argv[2] + std::strlen(argv[2]))
		{
			throw std::invalid_argument("n_time_steps: integer expected");
		}

		tolerance = std::strtod(argv[3], &str_end);
		if (str_end != argv[3] + std::strlen(argv[3]))
		{
			throw std::invalid_argument("tolerance: double expected");
		}

		time_step = 1. / n_time_steps;
		// tolerance /= n_time_steps;
	}

	template<int dim>
	void NonlinearSchroedingerEquation<dim>::setup_system()
	{
		GridGenerator::hyper_cube(triangulation, -1., 1.);
		triangulation.refine_global(refinement_level);

		pcout << "Number of active cells: " << triangulation.n_active_cells() << std::endl;

		dof_handler.distribute_dofs(fe);

		pcout << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl << std::endl;

		locally_owned_dofs = dof_handler.locally_owned_dofs();
		locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

		constraints.clear();
		constraints.reinit(locally_relevant_dofs);
		VectorTools::interpolate_boundary_values(dof_handler, 0, Functions::ZeroFunction<dim>(2), constraints);
		constraints.close();

		DynamicSparsityPattern dsp(locally_relevant_dofs);
		DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
		SparsityTools::distribute_sparsity_pattern(dsp, dof_handler.locally_owned_dofs(), mpi_communicator, locally_relevant_dofs);

		sparsity_pattern.copy_from(dsp);
		jacobian_matrix.reinit(locally_owned_dofs, locally_owned_dofs, sparsity_pattern, mpi_communicator);

		old_solution.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
		current_solution.reinit(locally_owned_dofs, mpi_communicator);
	}

	template<int dim>
	void NonlinearSchroedingerEquation<dim>::compute_residual(
		const LA::MPI::Vector& evaluation_point, LA::MPI::Vector& residual)
	{
		pcout << "\tComputing residual vector..." << std::flush;

		FEValues<dim> fe_v(fe, quadrature,
			update_values | update_gradients | update_quadrature_points | update_JxW_values);
		
		const unsigned int dofs_per_cell = fe_v.dofs_per_cell;

		std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

		residual = 0.;
		jacobian_matrix = 0.;

		for (const auto& cell : dof_handler.active_cell_iterators())
		{
			if (cell->is_locally_owned())
			{
				assemble_cell_term(cell, fe_v, local_dof_indices, evaluation_point, residual);
			}
		}

		jacobian_matrix.compress(VectorOperation::add);
		residual.compress(VectorOperation::add);

		pcout << " norm=" << residual.l2_norm() << std::endl;
	}

	template<int dim>
	void NonlinearSchroedingerEquation<dim>::assemble_cell_term(const typename DoFHandler<dim>::active_cell_iterator& cell,
		FEValues<dim>& fe_v, std::vector<types::global_dof_index>& dof_indices,
		const LA::MPI::Vector& evaluation_point, LA::MPI::Vector& residual)
	{
		fe_v.reinit(cell);
		cell->get_dof_indices(dof_indices);

		const unsigned int dofs_per_cell = fe_v.dofs_per_cell;
		const unsigned int n_q_points = fe_v.n_quadrature_points;

		Table<2, Sacado::Fad::DFad<double>> W(n_q_points, 2);
		Table<2, double> W_old(n_q_points, 2);

		Table<3, Sacado::Fad::DFad<double>> grad_W(n_q_points, 2, dim);
		Table<3, double> grad_W_old(n_q_points, 2, dim);

		FullMatrix<double> cell_jacobian(dofs_per_cell, dofs_per_cell);
		Vector<double> cell_residual(dofs_per_cell);

		cell_jacobian = 0.;
		cell_residual = 0.;

		std::vector<double> potential(n_q_points);

		std::vector<Sacado::Fad::DFad<double>> independent_local_dof_values(dofs_per_cell);

		// define independent variables for differentiation
		for (unsigned int i = 0; i < dofs_per_cell; ++i)
		{
			independent_local_dof_values[i] = evaluation_point(dof_indices[i]);
			independent_local_dof_values[i].diff(i, dofs_per_cell);
		}

		// zero W's
		for (unsigned int q = 0; q < n_q_points; ++q)
		{
			for (unsigned int c = 0; c < 2; ++c)
			{
				W[q][c] = 0.;
				W_old[q][c] = 0.;
				for (unsigned int d = 0; d < dim; ++d)
				{
					grad_W[q][c][d] = 0.;
					grad_W_old[q][c][d] = 0.;
				}
			}
		}

		// compute W's
		for (unsigned int q = 0; q < n_q_points; ++q)
		{
			for (unsigned int i = 0; i < dofs_per_cell; ++i)
			{
				const unsigned int c = fe_v.get_fe().system_to_component_index(i).first;

				W[q][c] += independent_local_dof_values[i] * fe_v.shape_value_component(i, q, c);
				W_old[q][c] += old_solution(dof_indices[i]) * fe_v.shape_value_component(i, q, c);

				for (unsigned int d = 0; d < dim; ++d)
				{
					grad_W[q][c][d] += independent_local_dof_values[i] * fe_v.shape_grad_component(i, q, c)[d];
					grad_W_old[q][c][d] += old_solution(dof_indices[i]) * fe_v.shape_grad_component(i, q, c)[d];
				}
			}
		}

		// compute quadrature points
		const std::vector<Point<dim>>& q_points = fe_v.get_quadrature_points();

		// compute potential at quadrature points
		Potential<dim>().value_list(q_points, potential);

		// compute residual
		for (unsigned int i = 0; i < dofs_per_cell; ++i)
		{
			Sacado::Fad::DFad<double> R_i = 0.;

			const unsigned int c_i = fe_v.get_fe().system_to_component_index(i).first;
			const double time_sign = c_i ? -1. : 1.;

			for (unsigned int q = 0; q < n_q_points; ++q)
			{
				const double JxW = fe_v.JxW(q);

				R_i += time_sign
				     * fe_v.shape_value_component(i, q, c_i)
				     * (W[q][1 - c_i] - W_old[q][1 - c_i])
					 * JxW;
				
				for (unsigned int d = 0; d < dim; ++d)
				{
					R_i += 0.25 * time_step
					     * fe_v.shape_grad_component(i, q, c_i)[d]
						 * (grad_W[q][c_i][d] + grad_W_old[q][c_i][d])
						 * JxW;
				}

				R_i += 0.5 * potential[q] * time_step
				     * fe_v.shape_value_component(i, q, c_i)
					 * (W[q][c_i] + W_old[q][c_i])
					 * JxW;
				
				R_i += 0.5 * kappa * time_step
				     * fe_v.shape_value_component(i, q, c_i)
					 * (W[q][c_i] * (W[q][c_i] * W[q][c_i] + W[q][1 - c_i] * W[q][1 - c_i]))
					 * JxW;
				
				R_i += 0.5 * kappa * time_step
				     * fe_v.shape_value_component(i, q, c_i)
					 * (W_old[q][c_i] * (W_old[q][c_i] * W_old[q][c_i] + W_old[q][1 - c_i] * W_old[q][1 - c_i]))
					 * JxW;
			}

			for (unsigned int k = 0; k < dofs_per_cell; ++k)
			{
				cell_jacobian(i, k) += R_i.fastAccessDx(k);
			}
			cell_residual(i) += R_i.val();
		}

		constraints.distribute_local_to_global(cell_jacobian, cell_residual, dof_indices, jacobian_matrix, residual);
	}

	template<int dim>
	void NonlinearSchroedingerEquation<dim>::factorize_jacobian()
	{
		pcout << "\tFactorizing Jacobian matrix..." << std::flush;

		pcout << " done\n";
	}

	template<int dim>
	void NonlinearSchroedingerEquation<dim>::solve(const LA::MPI::Vector& rhs, LA::MPI::Vector& solution, const double tol)
	{
		pcout << "\tSolving with Jacobian..." << std::flush;

		LA::MPI::Vector distributed_solution(locally_owned_dofs, mpi_communicator);

		SolverControl solver_control(dof_handler.n_dofs(), tol);

		LA::SolverGMRES solver(solver_control, mpi_communicator);

		// LA::MPI::PreconditionAMG preconditioner;
		// LA::MPI::PreconditionAMG::AdditionalData data;
		PETScWrappers::PreconditionNone preconditioner;
		PETScWrappers::PreconditionNone::AdditionalData data;

		preconditioner.initialize(jacobian_matrix, data);

		solver.solve(jacobian_matrix, distributed_solution, rhs, preconditioner);

		constraints.distribute(distributed_solution);

		solution = distributed_solution;

		pcout << " done in " << solver_control.last_step() << " iterations\n";
	}

	template<int dim>
	void NonlinearSchroedingerEquation<dim>::output_results() const
	{
		DataOut<dim> data_out;
		const DataPostprocessors::ComplexRealPart<dim> complex_re("Psi");
		const DataPostprocessors::ComplexImagPart<dim> complex_im("Psi");
		const DataPostprocessors::ComplexAmplitude<dim> complex_mag("Psi");
		const DataPostprocessors::ComplexPhase<dim> complex_arg("Psi");

		data_out.attach_dof_handler(dof_handler);

		LA::MPI::Vector relevant_solution(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
		relevant_solution = current_solution;

		data_out.add_data_vector(relevant_solution, complex_re);
		data_out.add_data_vector(relevant_solution, complex_im);
		data_out.add_data_vector(relevant_solution, complex_mag);
		data_out.add_data_vector(relevant_solution, complex_arg);

		Vector<float> subdomain(triangulation.n_active_cells());
		for (unsigned int i = 0; i < subdomain.size(); ++i)
		{
			subdomain(i) = triangulation.locally_owned_subdomain();
		}
		data_out.add_data_vector(subdomain, "subdomain");

		data_out.build_patches();

		data_out.write_vtu_with_pvtu_record("./", "solution", timestep_number, mpi_communicator, 5, 8);
	}

	template<int dim>
	void NonlinearSchroedingerEquation<dim>::run()
	{
		pcout << "Refinement level: " << refinement_level
			<< "\nNumber of time steps: " << n_time_steps
			<< "\nNewton step tolerance: " << tolerance << std::endl;

		setup_system();
		VectorTools::interpolate(dof_handler, InitialValues<dim>(), current_solution);
		output_results();

		while (timestep_number < n_time_steps)
		{
			++timestep_number;
			time += time_step;

			old_solution = current_solution;

			pcout << "Time step " << timestep_number << ":\n";

			SUNDIALS::KINSOL<LA::MPI::Vector>::AdditionalData additional_data;
			additional_data.function_tolerance = tolerance;

			SUNDIALS::KINSOL<LA::MPI::Vector> solver(additional_data, mpi_communicator);

			solver.reinit_vector = [this](LA::MPI::Vector& x)
			{
				x.reinit(this->locally_owned_dofs, this->mpi_communicator);
			};

			solver.residual = [this](const LA::MPI::Vector& evaluation_point, LA::MPI::Vector& residual)
			{
				TimerOutput::Scope timer_scope(this->computing_timer, "Residual");
				LA::MPI::Vector relevant_point(this->locally_owned_dofs, this->locally_relevant_dofs, this->mpi_communicator);
				relevant_point = evaluation_point;
				this->compute_residual(relevant_point, residual);
				return 0;
			};

			solver.setup_jacobian = [this](const LA::MPI::Vector&, const LA::MPI::Vector&)
			{
				TimerOutput::Scope timer_scope(this->computing_timer, "Setup Jacobian");
				this->factorize_jacobian();
				return 0;
			};

			solver.solve_with_jacobian = [this](const LA::MPI::Vector& rhs, LA::MPI::Vector& solution, const double tolerance)
			{
				TimerOutput::Scope timer_scope(this->computing_timer, "Solve with Jacobian");
				this->solve(rhs, solution, tolerance);
				return 0;
			};

			solver.solve(current_solution);

			if (timestep_number % (n_time_steps / 8) == 0)
			{
				output_results();
			}
		}

		computing_timer.print_summary();
		pcout << std::endl;
	}
}

int main(int argc, char* argv[])
try
{
	dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

	Monolithic::NonlinearSchroedingerEquation<2> problem(argc, argv);
	problem.run();

	return 0;
}
catch (std::exception& exc)
{
	std::cerr << "----------------------------------------\n"
	             "Uncaught exception in main:\n"
	          << exc.what() << "\nAborting!\n"
	             "----------------------------------------\n";
	return 1;
}
catch (...)
{
	std::cerr << "----------------------------------------\n"
	             "Uncaught error in main\nAborting!\n"
	             "----------------------------------------\n";
	return 1;
}
