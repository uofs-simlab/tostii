#pragma once

#include <tostii/time_stepping/time_stepping.h>

#include <deal.II/base/exceptions.h>

#include <string>
#include <array>
#include <vector>
#include <unordered_map>
#include <functional>

namespace tostii::TimeStepping
{
    DeclExceptionMsg(ExcNoMethodSelected,
        "No method selected. You need to call initialize or pass a runge_kutta_method to the constructor.");

    enum runge_kutta_method
        : unsigned int
    {
        // --------------------------------------------------------------------------
        //  Explicit RK

        /**
         * Forward Euler method, first order.
         */
        FORWARD_EULER,
        /**
         * Explicit midpoint method, 2nd order.
         */
        EXPLICIT_MIDPOINT,
        /**
         * Heun method, 2nd order.
         */
        HEUN2,
        /**
         * Third order Runge-Kutta method.
         */
        RK_THIRD_ORDER,
        /**
         * Third order Strong Stability Preserving (SSP) Runge-Kutta method
         * (SSP time discretizations are also called Total Variation Diminishing
         * (TVD) methods in the literature, see @cite gottlieb2001strong).
         */
        SSP_THIRD_ORDER,
        /**
         * Classical fourth order Runge-Kutta method.
         */
        RK_CLASSIC_FOURTH_ORDER,

        // --------------------------------------------------------------------------
        //  Implicit RK

        /**
         * Backward Euler method, first order.
         */
        BACKWARD_EULER,
        /**
         * Implicit midpoint method, second order.
         */
        IMPLICIT_MIDPOINT,
        /**
         * Crank-Nicolson method, second order.
         */
        CRANK_NICOLSON,
        /**
         * Two stage SDIRK method (short for "singly diagonally implicit
         * Runge-Kutta"), second order.
         */
        SDIRK_TWO_STAGES,
        /**
         * Three stage L-stable SDIRK method, third order.
         */
        SDIRK_THREE_STAGES,
        /**
         * Three stage A-stable SDIRK method, fourth order.
         */
        SDIRK_3O4,
        /**
         * Three stage A-stable SDIRK method, fourth order.
         */
        SDIRK_5O4,

        INVALID
    };

    /**
     * Mapping from runge_kutta_method enum to strings
     */
    extern const std::array<std::string, INVALID> runge_kutta_strings;

    /**
     * Mapping from strings to runge_kutta_method enum
     */
    extern const std::unordered_map<std::string, runge_kutta_method> runge_kutta_enums;

    /**
     * Base class for the Runge-Kutta methods
     */
    template<typename VectorType, typename TimeType = double>
    class RungeKutta
        : public TimeStepping<VectorType, TimeType>
    {
    public:
        /**
         * Virtual destructor.
         */
        virtual ~RungeKutta() override = default;

        /**
         * Purely virtual method used to initialize the Runge-Kutta method.
         */
        virtual void initialize(const runge_kutta_method method) = 0;

        /**
         * This function is used to advance from time @p t to t+ @p delta_t. @p
         * F is a vector of functions $ f(t,y) $ that should be integrated, the
         * input parameters are the time t and the vector y and the output is
         * value of f at this point. @p J_inverse is a vector functions that
         * compute the inverse of the Jacobians associated to the implicit
         * problems. The input parameters are the time, $ \tau $, and a vector.
         * The output is the value of function at this point. This function
         * returns the time at the end of the time step. When using Runge-Kutta
         * methods, @p F and @p J_inverse can only contain one element.
         */
        TimeType evolve_one_time_step(
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
            VectorType& y) override;

        /**
         * Purely virtual function. This function is used to advance from time
         * @p t to t+ @p delta_t. @p f  is the function $ f(t,y) $ that should
         * be integrated, the input parameters are the time t and the vector y
         * and the output is value of f at this point. @p id_minus_tau_J_inverse
         * is a function that computes $ inv(I-\tau J)$ where $ I $ is the
         * identity matrix, $ \tau $ is given, and $ J $ is the Jacobian $
         * \frac{\partial f}{\partial y} $. The input parameters are the time, $
         * \tau $, and a vector. The output is the value of function at this
         * point. evolve_one_time_step returns the time at the end of the time
         * step.
         */
        virtual TimeType evolve_one_time_step(
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
            VectorType& y) = 0;
        
    protected:
        /**
         * Number of stages of the Runge-Kutta method.
         */
        unsigned int n_stages;

        /**
         * Butcher tableau coefficients.
         */
        std::vector<double> b;

        /**
         * Butcher tableau coefficients.
         */
        std::vector<double> c;

        /**
         * Butcher tableau coefficients.
         */
        std::vector<std::vector<double>> a;
    };
}
