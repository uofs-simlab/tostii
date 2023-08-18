#pragma once

#include <deal.II/base/function.h>

#include <array>

#include "parameters.h"

namespace NSE::PrescribedData
{
    template<int dim>
    class InitialValues
        : public Function<dim>
    {
    public:
        InitialValues();

        double value(
            const Point<dim>& p,
            const unsigned int component = 0) const override;
            
    private:
        static const std::array<Point<dim>, 4> vortex_centers;
        static const double R;
    };

    template<int dim>
    class Potential
        : public Function<dim>
    {
    public:
        Potential();

        double value(
            const Point<dim>& p,
            const unsigned int component = 0) const override;

    private:
        static const double radius;
        static const double outer_potential;
    };
}
