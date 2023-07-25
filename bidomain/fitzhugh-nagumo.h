#pragma once

#include <deal.II/base/function.h>

#include <deal.II/lac/vector.h>

#include "parameters.h"

namespace Bidomain::FitzHughNagumo
{
    using namespace dealii;

    template<int dim>
    class InitialValues
        : public Function<dim>
    {
    public:
        InitialValues(
            const double initial_time,
            const Parameters::AllParameters& param);
        
        double value(
            const Point<dim>& p,
            const unsigned int component = 0) const override;

        void vector_value(
            const Point<dim>& p,
            Vector<double>& values) const override;
    };

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
}
