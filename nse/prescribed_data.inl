#pragma once

#include "prescribed_data.h"

namespace NSE::PrescribedData
{
    template<>
    const std::array<Point<2>, 4> InitialValues<2>::vortex_centers = { {
        { 0., -0.3 },
        { 0., +0.3 },
        { +0.3, 0. },
        { -0.3, 0. }
    } };

    template<>
    const double InitialValues<2>::R = 0.1;

    template<>
    InitialValues<2>::InitialValues()
        : Function<2>(2)
    { }

    template<>
	double InitialValues<2>::value(
        const Point<2>& p,
        const unsigned int component) const
	{
        AssertIndexRange(component, 2);

		// imaginary part
		if (component == 1) return 0.;

		const double alpha = 1. / (R * R * numbers::PI);

		// real part
		double sum = 0.;
		for (const auto& vortex_center : vortex_centers)
		{
			const double r = (p - vortex_center).norm();

			sum += alpha * std::exp(-(r * r) / (R * R));
		}

		return std::sqrt(sum);
	}

    template<int dim>
    const double Potential<dim>::radius = 0.7;

    template<int dim>
    const double Potential<dim>::outer_potential = 1000.;

    template<int dim>
    Potential<dim>::Potential()
        : Function<dim>(1)
    { }

    template<int dim>
    double Potential<dim>::value(
        const Point<dim>& p,
        const unsigned int component) const
    {
        AssertIndexRange(component, 1);
        (void)component;

        return p.norm() > radius
            ? outer_potential
            : 0.;
    }
}
