#pragma once

#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/conditional_ostream.h>

#include <deal.II/lac/generic_linear_algebra.h>

namespace LA
{
    using namespace dealii::LinearAlgebraPETSc;
}

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparsity_pattern.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/distributed/tria.h>

#include <tostii/time_stepping/exact.h>
#include <tostii/time_stepping/explicit_runge_kutta.h>
#include <tostii/time_stepping/operator_split_single.h>

#include "parameters.h"
#include "fitzhugh-nagumo.h"

namespace Bidomain
{
    using namespace dealii;

    template<int dim>
    class BaseProblem
    {
    public:
        BaseProblem(const Parameters::AllParameters& param);

    protected:
        static constexpr unsigned int
            transmembrane_component = 0,
            state_variable_component = 1,
            extracellular_component = 2;

        void initialize_split(
            const std::vector<unsigned int>& mask,
            AffineConstraints<double>& constraints,
            DynamicSparsityPattern& dsp,
            std::vector<unsigned int>& component_dof_indices,
            std::function<types::global_dof_index(
                types::global_dof_index)>& shift,
            std::function<void(
                const Vector<double>&,
                Vector<double>&)> translate[2]) const;

        const Parameters::AllParameters param;

        TimerOutput computing_timer;

        std::vector<unsigned int> dofs_per_block;
        std::vector<unsigned int> dof_offsets;

        Triangulation<dim> triangulation;
        DoFHandler<dim> dof_handler;

        const FESystem<dim> fe;
        const QGauss<dim> quadrature;

        AffineConstraints<double> constraints_template;
        DynamicSparsityPattern sparsity_template;
    };

    template<int dim>
    class ExplicitProblem
        : virtual public BaseProblem<dim>
    {
    public:
        ExplicitProblem(const Parameters::AllParameters& param);

    protected:
        void rhs_f(
            double t,
            const Vector<double>& y,
            Vector<double>& out);

    private:
        void assemble_system();
        void assemble_membrane_rhs(
            double t,
            const Vector<double>& y,
            Vector<double>& out);
        void solve_membrane_lhs(
            const Vector<double>& y,
            Vector<double>& out);

        AffineConstraints<double> constraints;
        std::vector<unsigned int> component_dof_indices;
        std::function<types::global_dof_index(
            types::global_dof_index)> shift;
        std::function<void(
            const Vector<double>&,
            Vector<double>&)> translate[2];

        Vector<double> temp;
        Vector<double> translate_buffer[2];

        SparsityPattern sparsity_pattern;
        SparseMatrix<double> mass_matrix;
        SparseMatrix<double> membrane_matrix;
    };

    template<int dim>
    class ImplicitProblem
        : virtual public BaseProblem<dim>
    {
    public:
        ImplicitProblem(const Parameters::AllParameters& param);

    protected:
        void step_tissue(
            double t,
            double tau,
            const Vector<double>& y,
            Vector<double>& out);

    private:
        void assemble_system();

        AffineConstraints<double> constraints;
        std::vector<unsigned int> component_dof_indices;
        std::function<types::global_dof_index(
            types::global_dof_index)> shift;
        std::function<void(
            const Vector<double>&,
            Vector<double>&)> translate[2];

        Vector<double> temp;
        Vector<double> translate_buffer[2];

        SparsityPattern sparsity_pattern;
        SparseMatrix<double> mass_matrix;
        SparseMatrix<double> tissue_matrix;
        SparseMatrix<double> system_matrix;
    };

    template<int dim>
    class BidomainProblem
        : public ExplicitProblem<dim>
        , public ImplicitProblem<dim>
    {
    public:
        using E = ExplicitProblem<dim>;
        using I = ImplicitProblem<dim>;

        BidomainProblem(const Parameters::AllParameters& param);
        
        void run();

    private:
        void output_results() const;

        Vector<double> solution;

        unsigned int timestep_number;
        const double time_step;
        double time;
    };
}
