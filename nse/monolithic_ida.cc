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
#include <deal.II/sundials/ida.h>

#include <Sacado.hpp>

#include <deal.II/distributed/tria.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>

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
		void compute_residual(
			const LA::MPI::Vector& y,
			const LA::MPI::Vector& y_dot,
			LA::MPI::Vector& res);
		void assemble_cell_term(
			const typename DoFHandler<dim>::active_cell_iterator& cell,
			FEValues<dim>& fe_v,
			std::vector<types::global_dof_index>& dof_indices,
			const LA::MPI::Vector& y,
			const LA::MPI::Vector& y_dot,
			LA::MPI::Vector& res);
		void setup_jacobian(const double alpha);
		void solve(const LA::MPI::Vector& rhs, LA::MPI::Vector& solution, const double tolerance);
		void output_results(double t, const LA::MPI::Vector& y, unsigned int step_number) const;

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
		AffineConstraints<double> dot_constraints;

		LA::MPI::Vector current_solution;
		LA::MPI::Vector current_dot_solution;
		
		SparsityPattern sparsity_pattern;
		LA::MPI::SparseMatrix jacobian_matrix;
		LA::MPI::SparseMatrix dot_jacobian_matrix;
		LA::MPI::SparseMatrix full_jacobian_matrix;

		unsigned int refinement_level;
		double tolerance;
		unsigned int n_time_steps;
		double time_step;

		double kappa;
	};

	template<int dim>
	NonlinearSchroedingerEquation<dim>::NonlinearSchroedingerEquation(int argc, char* argv[])
		: mpi_communicator(MPI_COMM_WORLD), pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0),
		computing_timer(mpi_communicator, pcout, TimerOutput::never, TimerOutput::wall_times),
		triangulation(mpi_communicator),
		dof_handler(triangulation), fe(FE_Q<dim>(2), 2), quadrature(fe.degree + 1),
		kappa(1.)
	{
		if (argc < 3 || argc > 4)
		{
			using std::string_literals::operator""s;
			throw std::invalid_argument("usage: "s + argv[0] + " refinement_level tolerance [n_time_steps]");
		}

		char* str_end;
		refinement_level = std::strtoul(argv[1], &str_end, 0);
		if (str_end != argv[1] + std::strlen(argv[1]))
		{
			throw std::invalid_argument("refinement_level: integer expected");
		}

		tolerance = std::strtod(argv[2], &str_end);
		if (str_end != argv[2] + std::strlen(argv[2]))
		{
			throw std::invalid_argument("tolerance: double expected");
		}

		if (argc > 3)
		{
			n_time_steps = std::strtoul(argv[3], &str_end, 0);
			if (str_end != argv[3] + std::strlen(argv[3]))
			{
				throw std::invalid_argument("n_time_steps: integer expected");
			}
			time_step = 1. / n_time_steps;
		}
		else
		{
			n_time_steps = 0;
			time_step = 0.;
		}
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

		constraints.reinit(locally_relevant_dofs);
		VectorTools::interpolate_boundary_values(dof_handler, 0, Functions::ZeroFunction<dim>(2), constraints);
		constraints.close();

		dot_constraints.reinit(locally_relevant_dofs);
		dot_constraints.close();

		DynamicSparsityPattern dsp(locally_relevant_dofs);
		DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
		SparsityTools::distribute_sparsity_pattern(dsp, dof_handler.locally_owned_dofs(), mpi_communicator, locally_relevant_dofs);

		sparsity_pattern.copy_from(dsp);
		jacobian_matrix.reinit(locally_owned_dofs, locally_owned_dofs, sparsity_pattern, mpi_communicator);
		dot_jacobian_matrix.reinit(locally_owned_dofs, locally_owned_dofs, sparsity_pattern, mpi_communicator);
		full_jacobian_matrix.reinit(locally_owned_dofs, locally_owned_dofs, sparsity_pattern, mpi_communicator);

		current_solution.reinit(locally_owned_dofs, mpi_communicator);
		current_dot_solution.reinit(locally_owned_dofs, mpi_communicator);
	}

	template<int dim>
	void NonlinearSchroedingerEquation<dim>::compute_residual(
		const LA::MPI::Vector& y,
		const LA::MPI::Vector& y_dot,
		LA::MPI::Vector& res)
	{
		pcout << "\tComputing residual... " << std::flush;

		FEValues<dim> fe_v(fe, quadrature,
			update_values | update_gradients | update_quadrature_points | update_JxW_values);
		
		const unsigned int dofs_per_cell = fe_v.dofs_per_cell;

		std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

		res = 0.;
		jacobian_matrix = 0.;
		dot_jacobian_matrix = 0.;

		for (const auto& cell : dof_handler.active_cell_iterators())
		{
			if (cell->is_locally_owned())
			{
				assemble_cell_term(cell, fe_v, local_dof_indices, y, y_dot, res);
			}
		}

		jacobian_matrix.compress(VectorOperation::add);
		dot_jacobian_matrix.compress(VectorOperation::add);
		res.compress(VectorOperation::add);

		pcout << "digits=" << -std::log10(res.l2_norm()) << std::endl;
	}

	template<int dim>
	void NonlinearSchroedingerEquation<dim>::assemble_cell_term(
		const typename DoFHandler<dim>::active_cell_iterator& cell,
		FEValues<dim>& fe_v,
		std::vector<types::global_dof_index>& dof_indices,
		const LA::MPI::Vector& y,
		const LA::MPI::Vector& y_dot,
		LA::MPI::Vector& res)
	{
		fe_v.reinit(cell);
		cell->get_dof_indices(dof_indices);

		const unsigned int dofs_per_cell = fe_v.dofs_per_cell;
		const unsigned int n_q_points = fe_v.n_quadrature_points;

		Table<2, Sacado::Fad::DFad<double>> W(n_q_points, 2);
		Table<2, Sacado::Fad::DFad<double>> W_dot(n_q_points, 2);
		Table<3, Sacado::Fad::DFad<double>> grad_W(n_q_points, 2, dim);

		FullMatrix<double> cell_jacobian(dofs_per_cell, dofs_per_cell);
		FullMatrix<double> cell_dot_jacobian(dofs_per_cell, dofs_per_cell);
		Vector<double> cell_residual(dofs_per_cell);

		cell_jacobian = 0.;
		cell_dot_jacobian = 0.;
		cell_residual = 0.;

		std::vector<double> potential(n_q_points);

		std::vector<Sacado::Fad::DFad<double>> independent_local_dof_values(dofs_per_cell);
		std::vector<Sacado::Fad::DFad<double>> independent_local_dof_dot_values(dofs_per_cell);

		// define independent variables for differentiation
		for (unsigned int k = 0; k < dofs_per_cell; ++k)
		{
			independent_local_dof_values[k] = y(dof_indices[k]);
			independent_local_dof_values[k].diff(k, 2 * dofs_per_cell);

			independent_local_dof_dot_values[k] = y_dot(dof_indices[k]);
			independent_local_dof_dot_values[k].diff(dofs_per_cell + k, 2 * dofs_per_cell);
		}

		// zero W's
		for (unsigned int q = 0; q < n_q_points; ++q)
		{
			for (unsigned int c = 0; c < 2; ++c)
			{
				W[q][c] = 0.;
				W_dot[q][c] = 0.;
				for (unsigned int d = 0; d < dim; ++d)
				{
					grad_W[q][c][d] = 0.;
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
				W_dot[q][c] += independent_local_dof_dot_values[i] * fe_v.shape_value_component(i, q, c);

				for (unsigned int d = 0; d < dim; ++d)
				{
					grad_W[q][c][d] += independent_local_dof_values[i] * fe_v.shape_grad_component(i, q, c)[d];
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
				     * W_dot[q][1 - c_i]
					 * JxW;
				
				for (unsigned int d = 0; d < dim; ++d)
				{
					R_i += 0.5
					     * fe_v.shape_grad_component(i, q, c_i)[d]
						 * grad_W[q][c_i][d]
						 * JxW;
				}

				R_i += potential[q]
				     * fe_v.shape_value_component(i, q, c_i)
					 * W[q][c_i]
					 * JxW;
				
				R_i += kappa
				     * fe_v.shape_value_component(i, q, c_i)
					 * (W[q][c_i] * (W[q][c_i] * W[q][c_i] + W[q][1 - c_i] * W[q][1 - c_i]))
					 * JxW;
			}

			for (unsigned int k = 0; k < dofs_per_cell; ++k)
			{
				cell_jacobian(i, k) += R_i.fastAccessDx(k);
				cell_dot_jacobian(i, k) += R_i.fastAccessDx(dofs_per_cell + k);
			}
			cell_residual(i) += R_i.val();
		}

		constraints.distribute_local_to_global(cell_jacobian, cell_residual, dof_indices, jacobian_matrix, res);
		dot_constraints.distribute_local_to_global(cell_dot_jacobian, dof_indices, dot_jacobian_matrix);
	}

	template<int dim>
	void NonlinearSchroedingerEquation<dim>::setup_jacobian(const double alpha)
	{
		full_jacobian_matrix = 0.;
		full_jacobian_matrix.add(1., jacobian_matrix);
		full_jacobian_matrix.add(alpha, dot_jacobian_matrix);
	}

	template<int dim>
	void NonlinearSchroedingerEquation<dim>::solve(const LA::MPI::Vector& rhs, LA::MPI::Vector& solution, const double tol)
	{
		LA::MPI::Vector distributed_solution(locally_owned_dofs, mpi_communicator);

		SolverControl solver_control(dof_handler.n_dofs(), tol);

		LA::SolverGMRES solver(solver_control, mpi_communicator);

		PETScWrappers::PreconditionNone preconditioner;
		PETScWrappers::PreconditionNone::AdditionalData data;

		preconditioner.initialize(full_jacobian_matrix, data);

		solver.solve(full_jacobian_matrix, distributed_solution, rhs, preconditioner);

		constraints.distribute(distributed_solution);

		solution = distributed_solution;
	}

	template<int dim>
	void NonlinearSchroedingerEquation<dim>::output_results(double t, const LA::MPI::Vector& y, unsigned int step_number) const
	{
		DataOut<dim> data_out;
		const DataPostprocessors::ComplexRealPart<dim> complex_re("Psi");
		const DataPostprocessors::ComplexImagPart<dim> complex_im("Psi");
		const DataPostprocessors::ComplexAmplitude<dim> complex_mag("Psi");
		const DataPostprocessors::ComplexPhase<dim> complex_arg("Psi");

		data_out.attach_dof_handler(dof_handler);

		LA::MPI::Vector relevant_solution(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
		relevant_solution = y;

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

		std::stringstream fname;
		fname << "solution-" << std::setprecision(3) << t;

		data_out.write_vtu_with_pvtu_record("./", fname.str(), step_number, mpi_communicator, 5, 8);
	}

	template<int dim>
	void NonlinearSchroedingerEquation<dim>::run()
	{
		using std::string_literals::operator""s;
		pcout << "Refinement level: " << refinement_level
			<< "\nTime Steps: " << (n_time_steps ? std::to_string(n_time_steps) : "Default"s)
			<< "\nNewton step tolerance: " << tolerance << std::endl;

		setup_system();
		VectorTools::interpolate(dof_handler, InitialValues<dim>(), current_solution);

		SUNDIALS::IDA<LA::MPI::Vector>::AdditionalData additional_data;
		additional_data.absolute_tolerance = tolerance;
		if (n_time_steps)
		{
			additional_data.initial_step_size = time_step;
		}
		additional_data.initial_time = 0.0;
		additional_data.final_time = 1.0;

		SUNDIALS::IDA<LA::MPI::Vector> solver(additional_data, mpi_communicator);

		solver.reinit_vector = [this](LA::MPI::Vector& x)
		{
			x.reinit(this->locally_owned_dofs, this->mpi_communicator);
		};

		solver.residual = [this](
			const double /*t*/,
			const LA::MPI::Vector& y,
			const LA::MPI::Vector& y_dot,
			LA::MPI::Vector& res)
		{
			TimerOutput::Scope timer_scope(this->computing_timer, "Residual");
			LA::MPI::Vector relevant_y(this->locally_owned_dofs, this->locally_relevant_dofs, this->mpi_communicator);
			LA::MPI::Vector relevant_y_dot(this->locally_owned_dofs, this->locally_relevant_dofs, this->mpi_communicator);
			relevant_y = y;
			relevant_y_dot = y_dot;
			this->compute_residual(relevant_y, relevant_y_dot, res);
			return 0;
		};

		solver.setup_jacobian = [this](
			const double /*t*/,
			const LA::MPI::Vector& /*y*/,
			const LA::MPI::Vector& /*y_dot*/,
			const double alpha)
		{
			TimerOutput::Scope timer_scope(this->computing_timer, "Setup Jacobian");
			this->setup_jacobian(alpha);
			return 0;
		};

		solver.solve_with_jacobian = [this](
			const LA::MPI::Vector& rhs,
			LA::MPI::Vector& solution,
			const double tolerance)
		{
			TimerOutput::Scope timer_scope(this->computing_timer, "Solve with Jacobian");
			this->solve(rhs, solution, tolerance);
			return 0;
		};

		solver.output_step = [this](
			const double t,
			const LA::MPI::Vector& y,
			const LA::MPI::Vector& /*y_dot*/,
			const unsigned int step_number)
		{
			this->pcout << "Output step [t=" << t << ", num=" << step_number << "]" << std::endl;
			this->output_results(t, y, step_number);
		};

		solver.solve_dae(current_solution, current_dot_solution);

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
