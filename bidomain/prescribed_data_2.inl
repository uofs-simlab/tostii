#include "prescribed_data.h"

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
    double ExactSolution<dim>::value(
        const Point<dim>& p,
        const unsigned int component) const
    {
        AssertIndexRange(component, 2);

        using std::cos, numbers::PI;
        const double t = this->get_time();

        double ue = 0.1 * t;
        for (unsigned int i = 0; i < dim; ++i)
        {
            ue *= cos(PI * p[i]);
        }

        if (component == 0)
        {
            return t * ue;
        }
        else
        {
            return ue;
        }
    }

    template<int dim>
    void ExactSolution<dim>::vector_value(
        const Point<dim>& p,
        Vector<double>& values) const
    {
        AssertDimension(values.size(), 2);

        using std::cos, numbers::PI;
        const double t = this->get_time();

        double ue = 0.1 * t;
        for (unsigned int i = 0; i < dim; ++i)
        {
            ue *= cos(PI * p[i]);
        }

        values[1] = ue;
        values[0] = t * ue;
    }

    template<int dim>
    TransmembraneRightHandSide<dim>::TransmembraneRightHandSide(
        const double initial_time,
        const Parameters::AllParameters& param)
        : Function<dim>(1, initial_time)
        , chi(param.chi)
        , Cm(param.Cm)
        , Rm(param.passive.Rm)
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

        using std::cos, numbers::PI;
        const double t = this->get_time();

        double f = 0.1 * t * (
            2 * chi * Cm
                + chi * t / Rm
                + dim * PI * PI * sigmai * (t + 1)
        );
        for (unsigned int i = 0; i < dim; ++i)
        {
            f *= cos(PI * p[i]);
        }

        return f;
    }

    template<int dim>
    ExtracellularRightHandSide<dim>::ExtracellularRightHandSide(
        const double initial_time,
        const Parameters::AllParameters& param)
        : Function<dim>(1, initial_time)
        , sigmai(param.sigmai)
        , sigmae(param.sigmae)
    { }

    template<int dim>
    double ExtracellularRightHandSide<dim>::value(
        const Point<dim>& p,
        const unsigned int component) const
    {
        (void)component;
        AssertIndexRange(component, 1);

        using std::cos, numbers::PI;
        const double t = this->get_time();

        double fe = 0.1 * t * (
            dim * PI * PI * (
                sigmai * t + sigmai + sigmae
            )
        );
        for (unsigned int i = 0; i < dim; ++i)
        {
            fe *= cos(PI * p[i]);
        }

        return fe;
    }
}
