#pragma once

#include <deal.II/base/function.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>

#include "parameters.h"

namespace Bidomain::FitzHughNagumo
{
    using namespace dealii;

    /**
     * FitzHugh-Nagumo initial conditions.
     * 
     * This class implements the initial conditions for the Bidomain FitzHugh-
     * Nagumo cell model, which are a set of equilibrium values for v, w, and
     * ue.
     */
    template<int dim>
    class InitialValues
        : public Function<dim>
    {
    public:
        InitialValues(const Parameters::AllParameters& param);
        
        double value(
            const Point<dim>& p,
            const unsigned int component = 0) const override;

        void vector_value(
            const Point<dim>& p,
            Vector<double>& values) const override;
    };

    /**
     * FitzHugh-Nagumo stimulus.
     * 
     * This class implements I_stim, which acts as a forcing term for v.
     * Specifically, for points within a radius of 1. from the origin, this
     * function returns 10, and elsewhere, returns 0.
     */
    template<int dim>
    class Stimulus
        : public Function<dim>
    {
    public:
        Stimulus(
            const double initial_time,
            const Parameters::AllParameters& param);
        
        double value(
            const Point<dim>& p,
            const unsigned int component = 0) const override;
    };

    namespace DataPostprocessors
    {
        using std::string_literals::operator""s;

        template<int dim>
        class TransmembranePart
            : public DataPostprocessorScalar<dim>
        {
        public:
            TransmembranePart();

            virtual void evaluate_vector_field(
                const DataPostprocessorInputs::Vector<dim>& input_data,
                std::vector<Vector<double>>& computed_quantities) const override;
        };

        template<int dim>
        class StateVariablePart
            : public DataPostprocessorScalar<dim>
        {
        public:
            StateVariablePart();

            virtual void evaluate_vector_field(
                const DataPostprocessorInputs::Vector<dim>& input_data,
                std::vector<Vector<double>>& computed_quantities) const override;
        };

        template<int dim>
        class ExtracellularPart
            : public DataPostprocessorScalar<dim>
        {
        public:
            ExtracellularPart();

            virtual void evaluate_vector_field(
                const DataPostprocessorInputs::Vector<dim>& input_data,
                std::vector<Vector<double>>& computed_quantities) const override;
        };
    }
}
