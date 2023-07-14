#pragma once

#include <tostii/time_stepping/exact.h>

#include <deal.II/base/exceptions.h>

namespace tostii::TimeStepping
{
    template<typename VectorType, typename TimeType>
    TimeType Exact<VectorType, TimeType>::evolve_one_time_step(
        std::vector<std::function<void(
            const TimeType,
            const VectorType&,
            VectorType&)>>&,
        std::vector<std::function<void(
            const TimeType,
            const TimeType,
            const VectorType&,
            VectorType&)>>& exact_eval,
        TimeType t,
        TimeType delta_t,
        VectorType& y)
    {
        AssertDimension(exact_eval.size(), 1);

        VectorType old_y(y);

        exact_eval[0](t, delta_t, old_y, y);

        return t + delta_t;
    }

    template<typename VectorType, typename TimeType>
    const typename Exact<VectorType, TimeType>::Status&
    Exact<VectorType, TimeType>::get_status() const
    {
        return status;
    }
}
