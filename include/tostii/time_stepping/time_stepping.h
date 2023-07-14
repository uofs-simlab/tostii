#pragma once

#include <vector>
#include <functional>

namespace tostii::TimeStepping
{
    /**
     * Abstract class for time stepping methods. These methods assume that the
     * equation has the form: $ \frac{\partial y}{\partial t} = f(t,y) $.
     */
    template <typename VectorType, typename TimeType = double>
    class TimeStepping
    {
    public:
        /**
         * Virtual destructor.
         */
        virtual ~TimeStepping() = default;

        /**
         * Purely virtual function. This function is used to advance from time
         * @p t to t+ @p delta_t. @p F is a vector of functions $ f(t,y) $ that
         * should be integrated, the input parameters are the time t and the
         * vector y and the output is value of f at this point. @p J_inverse is
         * a vector functions that compute the inverse of the Jacobians
         * associated to the implicit problems. The input parameters are the
         * time, $ \tau $, and a vector. The output is the value of function at
         * this point. This function returns the time at the end of the time
         * step.
         */
        virtual TimeType evolve_one_time_step(
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
            VectorType& y) = 0;

        /**
         * Empty structure used to store information.
         */
        struct Status { };

        /**
         * Purely virtual function that return Status.
         */
        virtual const Status& get_status() const = 0;
    };
}
