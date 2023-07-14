#pragma once

#include <tostii/time_stepping/implicit_runge_kutta.h>

namespace tostii::TimeStepping
{
    /**
     * This class derived from ImplicitRungeKutta and simply overrides compute_stages.
     * Instead of Newton solving, the system is assumed to be linear.
     */
    template<typename VectorType, typename TimeType = double>
    class LinearImplicitRungeKutta
        : public ImplicitRungeKutta<VectorType, TimeType>
    {
    public:
        /**
         * Default constructor. initialize() must be called before the object can be used.
         */
        LinearImplicitRungeKutta() = default;

        /**
         * Constructor. Calls initialize(method)
         */
        LinearImplicitRungeKutta(const runge_kutta_method method);

    protected:
        /**
         * Compute the different stages needed.
         */
        void compute_stages(
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
            std::vector<VectorType>& f_stages) override;
    };
}
