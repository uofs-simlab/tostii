#pragma once

#include "prescribed_data.h"

#include <deal.II/base/exceptions.h>

namespace Bidomain::PrescribedData
{
    template<int dim>
    ExactSolution<dim>::ExactSolution(
        const double initial_time,
        const Parameters::AllParameters& param)
        : Function<dim>(2, initial_time)
        , sigmai(param.sigmai)
        , sigmae(param.sigmae)
    { }

    template<int dim>
    void ExactSolution<dim>::vector_value(
        const Point<dim>& p,
        Vector<double>& values) const
    {
        AssertDimension(values.size(), 2);

        using std::cos, dealii::numbers::PI;
        const double t = this->get_time();

        double ue = t * t * t;
        for (unsigned int i = 0; i < dim; ++i)
        {
            ue *= cos(PI * p[i]);
        }
        values[1] = ue;
        values[0] = -(sigmai + sigmae) / sigmai * ue;
    }

    template<int dim>
    TransmembraneRightHandSide<dim>::TransmembraneRightHandSide(
        const double initial_time,
        const Parameters::AllParameters& param)
        : Function<dim>(1, initial_time)
        , chi(param.chi)
        , Cm(param.Cm)
        , Rm(param.Rm)
        , sigmai(param.sigmai)
        , sigmae(param.sigmae)
    { }

    template<int dim>
    double TransmembraneRightHandSide<dim>::value(
        const Point<dim>& p,
        const unsigned int component) const
    {
        (void)component;
        AssertIndexRange(component, 1);

        using std::cos, dealii::numbers::PI;
        const double t = this->get_time();

        double res = -(
                chi * (sigmai + sigmae) / sigmai * (
                    3 * Cm + t / Rm
                ) + 2 * sigmae * PI * PI * t
            ) * t * t;
        for (unsigned int i = 0; i < dim; ++i)
        {
            res *= cos(PI * p[i]);
        }
        return res;
    }

    template<int dim>
    ExtracellularRightHandSide<dim>::ExtracellularRightHandSide()
        : ZeroFunction<dim>(1)
    { }
}
