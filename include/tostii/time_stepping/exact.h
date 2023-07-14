#pragma once

#include <tostii/time_stepping/time_stepping.h>

namespace tostii::TimeStepping
{
    /**
     * Exact time integration class.
     */
    template<typename VectorType, typename TimeType = double>
    class Exact
        : public TimeStepping<VectorType, TimeType>
    {
    public:
        /**
         * Constructor.
         */
        Exact() = default;

        /**
         * This function is used to advance from time @p t to t+ @p delta_t. @p
         * nofunc is not used in this routine. @p exact_eval is a function that
         * evaluates the exact solution at time t + delta_t. @p y is state at
         * time t on input and at time t+delta_t on output.
         *
         * evolve_one_time_step returns the time at the end of the time step.
         */
        TimeType evolve_one_time_step(
            std::vector<std::function<void(
                const TimeType,
                const VectorType&,
                VectorType&)>>& f,
            std::vector<std::function<void(
                const TimeType,
                const TimeType,
                const VectorType&,
                VectorType&)>>& id_minus_tau_J_inverse,
            TimeType t,
            TimeType delta_t,
            VectorType& y) override;
        
        struct Status
            : public TimeStepping<VectorType, TimeType>::Status
        { };

        /**
         * Return the status of the current object.
         */
        const Status& get_status() const override;

    private:
        /**
         * Status structure of the object.
         */
        Status status;
    };
}
