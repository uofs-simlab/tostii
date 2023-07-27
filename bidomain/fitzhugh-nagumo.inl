#pragma once

#include "fitzhugh-nagumo.h"

namespace Bidomain::FitzHughNagumo
{
    template<int dim>
    InitialValues<dim>::InitialValues(const Parameters::AllParameters&)
        : Function<dim>(3)
    { }

    template<int dim>
    double InitialValues<dim>::value(
        const Point<dim>&,
        const unsigned int component) const
    {
        AssertIndexRange(component, 3);

        switch (component)
        {
        case 0:
            return -1.2879118919372559;
        case 1:
            return -0.5758181214332581;
        case 2:
        default:
            return 0.;
        }
    }

    template<int dim>
    void InitialValues<dim>::vector_value(
        const Point<dim>&,
        Vector<double>& values) const
    {
        AssertDimension(values.size(), 3);
        
        values[0] = -1.2879118919372559;
        values[1] = -0.5758181214332581;
        values[2] = 0.;
    }

    template<int dim>
    Stimulus<dim>::Stimulus(
        const double initial_time,
        const Parameters::AllParameters&)
        : Function<dim>(1, initial_time)
    { }

    template<int dim>
    double Stimulus<dim>::value(
        const Point<dim>& p,
        const unsigned int component) const
    {
        (void)component;
        AssertIndexRange(component, 1);

        if (p.norm_square() < 1.)
        {
            return 10.;
        }
        else
        {
            return 0.;
        }
    }
}

namespace Bidomain::FitzHughNagumo::DataPostprocessors
{
    template<int dim>
    TransmembranePart<dim>::TransmembranePart()
        : DataPostprocessorScalar<dim>("v", update_values)
    { }

    template<int dim>
    void TransmembranePart<dim>::evaluate_vector_field(
        const DataPostprocessorInputs::Vector<dim>& input_data,
        std::vector<Vector<double>>& computed_quantities) const
    {
        AssertDimension(computed_quantities.size(), input_data.solution_values.size());

        for (unsigned int i = 0; i < computed_quantities.size(); ++i)
        {
            AssertDimension(computed_quantities[i].size(), 1);
            AssertDimension(input_data.solution_values[i].size(), 3);

            computed_quantities[i][0] = input_data.solution_values[i][0];
        }
    }

    template<int dim>
    StateVariablePart<dim>::StateVariablePart()
        : DataPostprocessorScalar<dim>("w", update_values)
    { }

    template<int dim>
    void StateVariablePart<dim>::evaluate_vector_field(
        const DataPostprocessorInputs::Vector<dim>& input_data,
        std::vector<Vector<double>>& computed_quantities) const
    {
        AssertDimension(computed_quantities.size(), input_data.solution_values.size());

        for (unsigned int i = 0; i < computed_quantities.size(); ++i)
        {
            AssertDimension(computed_quantities[i].size(), 1);
            AssertDimension(input_data.solution_values[i].size(), 3);

            computed_quantities[i][0] = input_data.solution_values[i][1];
        }
    }

    template<int dim>
    ExtracellularPart<dim>::ExtracellularPart()
        : DataPostprocessorScalar<dim>("u_e", update_values)
    { }

    template<int dim>
    void ExtracellularPart<dim>::evaluate_vector_field(
        const DataPostprocessorInputs::Vector<dim>& input_data,
        std::vector<Vector<double>>& computed_quantities) const
    {
        AssertDimension(computed_quantities.size(), input_data.solution_values.size());

        for (unsigned int i = 0; i < computed_quantities.size(); ++i)
        {
            AssertDimension(computed_quantities[i].size(), 1);
            AssertDimension(input_data.solution_values[i].size(), 3);

            computed_quantities[i][0] = input_data.solution_values[i][2];
        }
    }
}
