#pragma once

#include <tostii/time_stepping/runge_kutta.h>

namespace tostii::TimeStepping
{
    /**
     * ExplicitRungeKutta is derived from RungeKutta and implement the explicit
     * methods.
     */
    template <typename VectorType, typename TimeType = double>
    class ExplicitRungeKutta
        : public RungeKutta<VectorType,TimeType>
    {
    public:
        /**
         * Default constructor. This constructor creates an object for which
         * you will want to call <code>initialize(runge_kutta_method)</code>
         * before it can be used.
         */
        ExplicitRungeKutta() = default;

        /**
         * Constructor. This function calls initialize(runge_kutta_method).
         */
        ExplicitRungeKutta(const runge_kutta_method method);

        /**
         * Initialize the explicit Runge-Kutta method.
         */
        void initialize(const runge_kutta_method method) override;

        /**
         * Expose base Runge-Kutta evolution function
         */
        using RungeKutta<VectorType, TimeType>::evolve_one_time_step;

        /**
         * This function is used to advance from time @p t to t+ @p delta_t. @p f
         * is the function $ f(t,y) $ that should be integrated, the input
         * parameters are the time t and the vector y and the output is value of f
         * at this point. @p id_minus_tau_J_inverse is a function that computes $
         * inv(I-\tau J)$ where $ I $ is the identity matrix, $ \tau $ is given,
         * and $ J $ is the Jacobian $ \frac{\partial f}{\partial y} $. The input
         * parameter are the time, $ \tau $, and a vector. The output is the value
         * of function at this point. evolve_one_time_step returns the time at the
         * end of the time step.
         *
         * @note @p id_minus_tau_J_inverse is ignored since the method is explicit.
         */
        TimeType evolve_one_time_step(
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
            VectorType& y);

        /**
         * This function is used to advance from time @p t to t+ @p delta_t. This
         * function is similar to the one derived from RungeKutta, but does not
         * required id_minus_tau_J_inverse because it is not used for explicit
         * methods. evolve_one_time_step returns the time at the end of the time
         * step.
         */
        TimeType evolve_one_time_step(
            const std::function<void(
                const TimeType,
                const VectorType&,
                VectorType&)>& f,
            TimeType t,
            TimeType delta_t,
            VectorType& y);

        /**
         * This structure stores the name of the method used.
         */
        struct Status
            : public TimeStepping<VectorType,TimeType>::Status
        {
            Status()
                : method(INVALID)
            { }

            runge_kutta_method method;
        };

        /**
         * Return the status of the current object.
         */
        const Status& get_status() const override;

    private:
        /**
         * Compute the different stages needed.
         */
        void compute_stages(
            const std::function<void(
                const TimeType,
                const VectorType&,
                VectorType&)>& f,
            const TimeType t,
            const TimeType delta_t,
            const VectorType& y,
            std::vector<VectorType>& f_stages) const;

        /**
         * Status structure of the object.
         */
        Status status;
    };
}
