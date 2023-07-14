#pragma once

#include <deal.II/base/function.h>

#include <deal.II/lac/vector.h>

#include "parameters.h"

namespace Bidomain::PrescribedData
{
    using namespace dealii;

    template<int dim>
    class ExactSolution
        : public Function<dim>
    {
    public:
        ExactSolution(
            const double initial_time,
            const Parameters::AllParameters& param);
        
        void vector_value(
            const Point<dim>& p,
            Vector<double>& values) const override;
        
    private:
        const double sigmai;
        const double sigmae;
    };

    template<int dim>
    class TransmembraneRightHandSide
        : public Function<dim>
    {
    public:
        TransmembraneRightHandSide(
            const double initial_time,
            const Parameters::AllParameters& param);
        
        double value(
            const Point<dim>& p,
            const unsigned int component = 0) const override;

    private:
        const double chi;
        const double Cm;
        const double Rm;
        const double sigmai;
        const double sigmae;
    };

    template<int dim>
    class ExtracellularRightHandSide
        : public ZeroFunction<dim>
    {
    public:
        ExtracellularRightHandSide();
    };
}
