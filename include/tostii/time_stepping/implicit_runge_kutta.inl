#pragma once

#include <tostii/time_stepping/runge_kutta.inl>
#include <tostii/time_stepping/implicit_runge_kutta.h>

namespace tostii::TimeStepping
{
    template<typename VectorType, typename TimeType>
    ImplicitRungeKutta<VectorType, TimeType>::ImplicitRungeKutta(
        const runge_kutta_method method,
        const unsigned int max_it,
        const double tolerance)
        : max_it(max_it), tolerance(tolerance)
    {
        ImplicitRungeKutta<VectorType, TimeType>::initialize(method);
    }

    template<typename VectorType, typename TimeType>
    void ImplicitRungeKutta<VectorType, TimeType>::initialize(
        const runge_kutta_method method)
    {
        status.method = method;

        switch (method)
        {
        case BACKWARD_EULER:
            {
                this->n_stages = 1;
                this->b.push_back(1.0);
                this->c.push_back(1.0);
                this->a.push_back(std::vector<double>(1, 1.0));

                break;
            }
        case IMPLICIT_MIDPOINT:
            {
                this->n_stages = 1;
                this->b.push_back(1.0);
                this->c.push_back(0.5);
                this->a.push_back(std::vector<double>(1, 0.5));

                break;
            }
        case CRANK_NICOLSON:
            {
                this->n_stages = 2;
                this->b.reserve(this->n_stages);
                this->c.reserve(this->n_stages);

                this->b.push_back(0.5);
                this->b.push_back(0.5);

                this->c.push_back(0.0);
                this->c.push_back(1.0);

                this->a.push_back(std::vector<double>(1, 0.0));
                this->a.push_back(std::vector<double>(2, 0.5));

                break;
            }
        case SDIRK_TWO_STAGES:
            {
                this->n_stages = 2;
                this->b.reserve(this->n_stages);
                this->c.reserve(this->n_stages);

                double const gamma = 1.0 - 1.0 / std::sqrt(2.0);

                this->b.push_back(1.0 - gamma);
                this->b.push_back(gamma);

                this->c.push_back(gamma);
                this->c.push_back(1.0);

                this->a.push_back(std::vector<double>(1, gamma));
                this->a.push_back(this->b);
                
                break;
            }
        case SDIRK_THREE_STAGES:
            {
                this->n_stages = 3;
                this->b.reserve(this->n_stages);
                this->c.reserve(this->n_stages);

                double const gamma = 0.4358665215;

                this->b.push_back(-3.0*gamma*gamma/2.0 + 4.0*gamma - 1./4.);
                this->b.push_back(3.0*gamma*gamma/2.0 - 5.0*gamma + 5.0/4.0 );
                this->b.push_back(gamma);

                this->c.push_back(gamma);
                this->c.push_back(0.5*(1+gamma));
                this->c.push_back(1.0);

                this->a.push_back(std::vector<double>(1, gamma));
                this->a.push_back(std::vector<double>{0.5*(1.0-gamma), gamma});
                this->a.push_back(this->b);

                break;
            }
        case SDIRK_3O4:
            {
                this->n_stages = 3;
                this->a.reserve(this->n_stages);
                this->b.reserve(this->n_stages);
                this->c.reserve(this->n_stages);

                double const gamma = 2.0 * std::cos(dealii::numbers::PI / 18.0) / std::sqrt(3.0);

                this->b.push_back(1.0/(6.0*gamma*gamma));
                this->b.push_back(1.0 - 1.0/(3.0*gamma*gamma));
                this->b.push_back(1.0/(6.0*gamma*gamma));

                this->c.push_back(0.5*(1.0+gamma));
                this->c.push_back(0.5);
                this->c.push_back(0.5*(1.0-gamma));

                this->a.push_back(std::vector<double>(1, 0.5*(1.0+gamma)));
                this->a.push_back(std::vector<double>{-0.5*gamma, 0.5*(1.0+gamma)});
                this->a.push_back(std::vector<double>{1.0+gamma,-1.0+2.0*gamma, 0.5*(1+gamma)});

                break;
            }
        case SDIRK_5O4:
            {
                this->n_stages = 5;
                this->a.reserve(this->n_stages);
                this->b.reserve(this->n_stages);
                this->c.reserve(this->n_stages);

                this->b.push_back(25./24);
                this->b.push_back(-49./48);
                this->b.push_back(125./16);
                this->b.push_back(-85./12);
                this->b.push_back(1./4);

                this->c.push_back(1./4);
                this->c.push_back(3./4);
                this->c.push_back(11./20);
                this->c.push_back(1./2);
                this->c.push_back(1.);

                this->a.push_back(std::vector<double>(1, 1./4));
                this->a.push_back(std::vector<double>{1./2, 1./4});
                this->a.push_back(std::vector<double>{17./50,-1./25, 1./4});
                this->a.push_back(std::vector<double>{371./1360, -137./2720, 15./544, 1./4});
                this->a.push_back(this->b);

                break;
            }
        default:
            {
                AssertThrow(false, dealii::ExcMessage("Unimplemented implicit Runge-Kutta method."));
            }
        }
    }

    template<typename VectorType, typename TimeType>
    TimeType ImplicitRungeKutta<VectorType, TimeType>::evolve_one_time_step(
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
        VectorType& y)
    {
        Assert(status.method != runge_kutta_method::INVALID, ExcNoMethodSelected());

        VectorType old_y(y);
        std::vector<VectorType> f_stages(this->n_stages, y);
        // Compute the different stages needed.
        compute_stages(f, id_minus_tau_J_inverse, t, delta_t, y, f_stages);

        y = old_y;
        for (unsigned int i = 0; i < this->n_stages; ++i)
        {
            y.sadd(1., delta_t * this->b[i], f_stages[i]);
        }

        return t + delta_t;
    }

    template<typename VectorType, typename TimeType>
    void ImplicitRungeKutta<VectorType, TimeType>::set_newton_solver_parameters(
        unsigned int max_it,
        double tolerance)
    {
        this->max_it = max_it;
        this->tolerance = tolerance;
    }

    template<typename VectorType, typename TimeType>
    const typename ImplicitRungeKutta<VectorType, TimeType>::Status&
    ImplicitRungeKutta<VectorType, TimeType>::get_status() const
    {
        return status;
    }

    template<typename VectorType, typename TimeType>
    void ImplicitRungeKutta<VectorType, TimeType>::compute_stages(
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

            // Solve the nonlinear system using Newton's method
            const TimeType new_t = t + this->c[i] * delta_t;
            const TimeType new_delta_t = this->a[i][i] * delta_t;
            VectorType& f_stage = f_stages[i];

            newton_solve(
                [this, &f, new_t, new_delta_t, &old_y, &f_stage](
                    const VectorType& y, VectorType& residual)
                {
                    this->compute_residual(f, new_t, new_delta_t, old_y, y, f_stage, residual);
                },
                [&id_minus_tau_J_inverse, new_t, new_delta_t](const VectorType& y, VectorType& out)
                {
                    id_minus_tau_J_inverse(new_t, new_delta_t, y, out);
                },
                y);
        }
    }

    template<typename VectorType, typename TimeType>
    void ImplicitRungeKutta<VectorType, TimeType>::newton_solve(
        const std::function<void(
            const VectorType&,
            VectorType&)>& get_residual,
        const std::function<void(
            const VectorType&,
            VectorType&)>& id_minus_tau_J_inverse,
        VectorType& y)
    {
        VectorType old_residual(y), residual(y);
        get_residual(y, old_residual);

        unsigned int i = 0;
        const double initial_residual_norm = old_residual.l2_norm();
        double norm_residual = initial_residual_norm;
        while (i < max_it)
        {
            id_minus_tau_J_inverse(old_residual, residual);
            y.sadd(1., -1., residual);
            get_residual(y, residual);
            norm_residual = residual.l2_norm();
            old_residual.swap(residual);

            if (norm_residual < tolerance) break;
            ++i;
        }

        status.n_iterations = i + 1;
        status.norm_residual = norm_residual;
    }

    template<typename VectorType, typename TimeType>
    void ImplicitRungeKutta<VectorType, TimeType>::compute_residual(
        const std::function<void(
            const TimeType,
            const VectorType&,
            VectorType&)>& f,
        TimeType t,
        TimeType delta_t,
        const VectorType& old_y,
        const VectorType& y,
        VectorType& tendency,
        VectorType& residual) const
    {
        // The tendency is stored to save one evaluation of f.
        f(t, y, tendency);
        residual = tendency;
        residual.sadd(delta_t, 1., old_y);
        residual.sadd(-1., 1., y);
    }
}
