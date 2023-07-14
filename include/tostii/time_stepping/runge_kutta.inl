#pragma once

#include <tostii/time_stepping/runge_kutta.h>

#include <deal.II/base/exceptions.h>

namespace tostii::TimeStepping
{
    template<typename VectorType, typename TimeType>
    TimeType RungeKutta<VectorType, TimeType>::evolve_one_time_step(
        std::vector<std::function<void(
            const TimeType,
            const VectorType&,
            VectorType&)>>& F,
        std::vector<std::function<void(
            const TimeType,
            const TimeType,
            const VectorType&,
            VectorType&)>>& J_inverse,
        TimeType t,
        TimeType delta_t,
        VectorType& y)
    {
        AssertDimension(F.size(), 1);

        return evolve_one_time_step(F[0], J_inverse[0], t, delta_t, y);
    }
}
