#pragma once

#include <tostii/time_stepping/runge_kutta.inl>
#include <tostii/time_stepping/explicit_runge_kutta.h>

namespace tostii::TimeStepping
{
    template<typename VectorType, typename TimeType>
    ExplicitRungeKutta<VectorType, TimeType>::ExplicitRungeKutta(
        const runge_kutta_method method)
    {
        ExplicitRungeKutta<VectorType, TimeType>::initialize(method);
    }

    template<typename VectorType, typename TimeType>
    void ExplicitRungeKutta<VectorType, TimeType>::initialize(
        const runge_kutta_method method)
    {
        status.method = method;

        switch (method)
        {
        case FORWARD_EULER:
            {
                this->n_stages = 1;
                this->b.push_back(1.0);
                this->c.push_back(0.0);

                break;
            }
        case EXPLICIT_MIDPOINT:
            {
                this->n_stages = 2;
                this->b.reserve(this->n_stages);
                this->c.reserve(this->n_stages);

                this->b.push_back(0.0);
                this->b.push_back(1.0);

                this->c.push_back(0.0);
                this->c.push_back(0.5);

                std::vector<double> tmp;
                this->a.push_back(tmp);
                tmp.push_back(0.5);
                this->a.push_back(tmp);

                break;
            }
        case HEUN2:
            {
                this->n_stages = 2;
                this->b.reserve(this->n_stages);
                this->c.reserve(this->n_stages);

                this->b.push_back(0.5);
                this->b.push_back(0.5);

                this->c.push_back(0.0);
                this->c.push_back(1.0);

                std::vector<double> tmp;
                this->a.push_back(tmp);
                tmp.push_back(1.0);
                this->a.push_back(tmp);

                break;
            }
        case RK_THIRD_ORDER:
            {
                this->n_stages = 3;
                this->b.reserve(this->n_stages);
                this->c.reserve(this->n_stages);

                this->b.push_back(1.0 / 6.0);
                this->b.push_back(2.0 / 3.0);
                this->b.push_back(1.0 / 6.0);

                this->c.push_back(0.0);
                this->c.push_back(0.5);
                this->c.push_back(1.0);

                std::vector<double> tmp;
                this->a.push_back(tmp);
                tmp.push_back(0.5);
                this->a.push_back(tmp);
                tmp[0] = -1.0;
                tmp.push_back(2.0);
                this->a.push_back(tmp);

                break;
            }
        case SSP_THIRD_ORDER:
            {
                this->n_stages = 3;
                this->b.reserve(this->n_stages);
                this->c.reserve(this->n_stages);

                this->b.push_back(1.0 / 6.0);
                this->b.push_back(1.0 / 6.0);
                this->b.push_back(2.0 / 3.0);

                this->c.push_back(0.0);
                this->c.push_back(1.0);
                this->c.push_back(0.5);

                std::vector<double> tmp;
                this->a.push_back(tmp);
                tmp.push_back(1.0);
                this->a.push_back(tmp);
                tmp[0] = 1.0 / 4.0;
                tmp.push_back(1.0 / 4.0);
                this->a.push_back(tmp);

                break;
            }
        case RK_CLASSIC_FOURTH_ORDER:
            {
                this->n_stages = 4;
                this->b.reserve(this->n_stages);
                this->c.reserve(this->n_stages);

                this->b.push_back(1.0 / 6.0);
                this->b.push_back(1.0 / 3.0);
                this->b.push_back(1.0 / 3.0);
                this->b.push_back(1.0 / 6.0);

                this->c.push_back(0.0);
                this->c.push_back(0.5);
                this->c.push_back(0.5);
                this->c.push_back(1.0);

                std::vector<double> tmp;
                this->a.push_back(tmp);
                tmp.push_back(0.5);
                this->a.push_back(tmp);
                tmp[0] = 0.0;
                tmp.push_back(0.5);
                this->a.push_back(tmp);
                tmp[1] = 0.0;
                tmp.push_back(1.0);
                this->a.push_back(tmp);
                
                break;
            }
        default:
            {
                AssertThrow(false, dealii::ExcMessage("Unimplemented explicit Runge-Kutta method"));
            }
        }
    }

    template<typename VectorType, typename TimeType>
    TimeType ExplicitRungeKutta<VectorType, TimeType>::evolve_one_time_step(
        const std::function<void(
            const TimeType,
            const VectorType&,
            VectorType&)>& f,
        const std::function<void(
            const TimeType,
            const TimeType,
            const VectorType&,
            VectorType&)>& /*id_minus_tau_J_inverse*/,
        TimeType t,
        TimeType delta_t,
        VectorType& y)
    {
        return evolve_one_time_step(f, t, delta_t, y);
    }

    template<typename VectorType, typename TimeType>
    TimeType ExplicitRungeKutta<VectorType, TimeType>::evolve_one_time_step(
        const std::function<void(
            const TimeType,
            const VectorType&,
            VectorType&)>& f,
        TimeType t,
        TimeType delta_t,
        VectorType& y)
    {
        Assert(status.method != runge_kutta_method::INVALID, ExcNoMethodSelected());

        std::vector<VectorType> f_stages(this->n_stages, y);
        // Compute the different stages needed.
        compute_stages(f, t, delta_t, y, f_stages);

        // Linear combination of the stages.
        for (unsigned int i = 0; i < this->n_stages; ++i)
        {
            y.sadd(1., delta_t * this->b[i], f_stages[i]);
        }

        return t + delta_t;
    }

    template<typename VectorType, typename TimeType>
    const typename ExplicitRungeKutta<VectorType, TimeType>::Status&
    ExplicitRungeKutta<VectorType, TimeType>::get_status() const
    {
        return status;
    }

    template <typename VectorType, typename TimeType>
    void ExplicitRungeKutta<VectorType, TimeType>::compute_stages(
        const std::function<void(
            const TimeType,
            const VectorType&,
            VectorType&)>& f,
        const TimeType t,
        const TimeType delta_t,
        const VectorType& y,
        std::vector<VectorType>& f_stages) const
    {
        for (unsigned int i = 0; i < this->n_stages; ++i)
        {
            VectorType Y(y);
            for (unsigned int j = 0; j < i; ++j)
            {
                Y.sadd(1., delta_t * this->a[i][j], f_stages[j]);
            }
            // Evaluate the function f at the point (t+c[i]*delta_t,Y).
            f(t + this->c[i] * delta_t, Y, f_stages[i]);
        }
    }
}
