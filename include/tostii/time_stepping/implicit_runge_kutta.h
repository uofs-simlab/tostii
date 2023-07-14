#pragma once

#include <tostii/time_stepping/runge_kutta.h>

#include <deal.II/base/signaling_nan.h>

namespace tostii::TimeStepping
{
    /**
     * This class is derived from RungeKutta and implements the implicit methods.
     * This class works only for Diagonal Implicit Runge-Kutta (DIRK) methods.
     */
    template<typename VectorType, typename TimeType = double>
    class ImplicitRungeKutta
        : public RungeKutta<VectorType, TimeType>
    {
    public:
        /**
         * Default constructor. initial(runge_kutta_method) and
         * set_newton_solver_parameters(unsigned int, double) need to be called
         * before the object can be used.
         */
        ImplicitRungeKutta() = default;

        /**
         * Constructor. This function calls initialize(runge_kutta_method) and
         * initializes the maximum number of iterations and the tolerance of the
         * Newton solver.
         */
        ImplicitRungeKutta(
            const runge_kutta_method method,
            const unsigned int max_it = 100,
            const double tolerance = 1e-6);
        
        /**
         * Initialize the implicit Runge-Kutta method.
         */
        void initialize(const runge_kutta_method method) override;

        using RungeKutta<VectorType, TimeType>::evolve_one_time_step;

        /**
         * This function is used to advance from time @p t to t+ @p delta_t. @p
         * f is the function $ f(t,y) $ that should be integrated, the input
         * parameters are the time t and the vector y and the output is value of
         * f at this point. @p id_minus_tau_J_inverse is a function that
         * computes $ (I-\tau J)^{-1}$ where $ I $ is the identity matrix, $
         * \tau $ is given, and $ J $ is the Jacobian $ \frac{\partial
         * f}{\partial y} $. The input parameters this function receives are the
         * time, $ \tau $, and a vector. The output is the value of function at
         * this point. evolve_one_time_step returns the time at the end of the
         * time step.
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
            VectorType& y) override;
        
        /**
         * Set the maximum number of iterations and the tolerance used by the
         * Newton solver.
         */
        void set_newton_solver_parameters(
            const unsigned int max_it,
            const double tolerance);
        
        struct Status
            : public TimeStepping<VectorType, TimeType>::Status
        {
            Status()
                : method(INVALID)
                , n_iterations(dealii::numbers::invalid_unsigned_int)
                , norm_residual(dealii::numbers::signaling_nan<double>())
            { }

            runge_kutta_method method;
            unsigned int n_iterations;
            double norm_residual;
        };

        const Status& get_status() const override;

    protected:
        /**
         * Compute the different stages needed.
         */
        virtual void compute_stages(
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
            std::vector<VectorType>& f_stages);

    private:
        /**
         * Newton solver used for the implicit stages
         */
        void newton_solve(
            const std::function<void(
                const VectorType&,
                VectorType&)>& get_residual,
            const std::function<void(
                const VectorType&,
                VectorType&)>& id_minus_tau_J_inverse,
            VectorType& y);
        
        /**
         * Compute the residual needed by the Newton solver.
         */
        void compute_residual(
            const std::function<void(
                const TimeType,
                const VectorType&,
                VectorType&)>& f,
            TimeType t,
            TimeType delta_t,
            const VectorType& new_y,
            const VectorType& y,
            VectorType& tendency,
            VectorType& residual) const;
        
        /**
         * Maximum number of iteration of the Newton solver.
         */
        unsigned int max_it;

        /**
         * Tolerance of the Newton solver.
         */
        double tolerance;

        /**
         * Status structure of the object.
         */
        Status status;
    };
}
