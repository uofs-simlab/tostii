#pragma once

#include <tostii/time_stepping/implicit_runge_kutta.inl>
#include <tostii/time_stepping/linear_implicit_runge_kutta.h>

namespace tostii::TimeStepping
{
    template<typename VectorType, typename TimeType>
    LinearImplicitRungeKutta<VectorType, TimeType>::LinearImplicitRungeKutta(
        const runge_kutta_method method)
        : ImplicitRungeKutta<VectorType, TimeType>(method)
    { }

    template<typename VectorType, typename TimeType>
    void LinearImplicitRungeKutta<VectorType, TimeType>::compute_stages(
        const std::function<void(
            const TimeType,
            const VectorType&,
            VectorType&)>& f,
        const std::function<void(
            const TimeType,
            const TimeType,
            const VectorType&,
            VectorType&)>& id_minus_tau_J_inverse,
        TimeType t,
        TimeType delta_t,
        VectorType& y,
        std::vector<VectorType>& f_stages)
    {
        VectorType z(y);
        for (unsigned int i = 0; i < this->n_stages; ++i)
        {
            VectorType old_y(z);
            for (unsigned int j = 0; j < i; ++j)
            {
                old_y.sadd(1., delta_t * this->a[i][j], f_stages[j]);
            }

            // solve the linear system using the provided solver
            const TimeType new_t = t + this->c[i] * delta_t;
            const TimeType new_delta_t = this->a[i][i] * delta_t;
            VectorType& f_stage = f_stages[i];

            f(new_t, old_y, y);
            id_minus_tau_J_inverse(new_t, new_delta_t, y, f_stage);
        }
    }
}
