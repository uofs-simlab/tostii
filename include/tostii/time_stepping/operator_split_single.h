#pragma once

#include <tostii/time_stepping/time_stepping.h>
#include <tostii/time_stepping/operator_split.h>

namespace tostii::TimeStepping
{
    /**
     * Class for OperatorSplitSingle time stepping
     * - Single-component problem
     */
    template<typename VectorType, typename TimeType = double>
    class OperatorSplitSingle
        : public TimeStepping<VectorType, TimeType>
    {
    public:
        /**
         * Backwards-compatibility typedef.
         */
        typedef std::function<void(
            const TimeType,
            const VectorType&,
            VectorType&)> f_fun_type;
        
        /**
         * Backwards-compatibility typedef.
         */
        typedef std::vector<f_fun_type> f_vfun_type;

        /**
         * Backwards-compatibility typedef.
         */
        typedef std::function<void(
            const TimeType,
            const TimeType,
            const VectorType&,
            VectorType&)> jac_fun_type;
        
        /**
         * Backwards-compatibility typedef.
         */
        typedef std::vector<jac_fun_type> jac_vfun_type;

        /**
         * Constructor for named OS methods.
         */
        OperatorSplitSingle(
            const std::vector<OSOperator<VectorType, TimeType>>& operators,
            const os_method_t<TimeType> method);

        /**
         * Constructor for named OS methods.
         */
        OperatorSplitSingle(
            const std::vector<OSOperator<VectorType, TimeType>>& operators,
            const os_method_t<TimeType> method,
            const VectorType& ref);

        /**
         * Constructor for explicitly-specified stages.
         */
        OperatorSplitSingle(
            const std::vector<OSOperator<VectorType, TimeType>>& operators,
            const std::vector<OSPair<TimeType>>& stages);
        
        /**
         * Constructor for explicitly-specified stages.
         */
        OperatorSplitSingle(
            const std::vector<OSOperator<VectorType, TimeType>>& operators,
            const std::vector<OSPair<TimeType>>& stages,
            const VectorType& ref);

        /**
         * This function is used to advance from time @p t to t+ @p delta_t.
         * @p f is a vector of functions $ f_j(t,y) $ that should be
         * integrated, the input parameters are the time t and the vector y
         * and the output is value of f at this point.
         * @p id_minus_tau_J_inverse is a vector of functions that computes
         * $(I-\tau * J_j)^{-1}$ where $ I $ is the identity matrix, $ \tau $ is
         * given, and $ J_j $ is the Jacobian $ \frac{\partial f_j}{\partial
         * y_j} This function passes the f_j and thier corresponding I-J_j^{-1}
         * functions through to the sub-integrators that were setup on
         * construction.
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
        
        /**
         * Trimmed down function for an OS method that has been set up.
         */
        TimeType evolve_one_time_step(
            TimeType t,
            TimeType delta_t,
            VectorType& y);

        struct Status
            : public TimeStepping<VectorType, TimeType>::Status
        { };

        /**
         * Return the status of the current object.
         */
        const Status& get_status() const override;

    private:
        /**
         * Operator definitions.
         */
        std::vector<OSOperator<VectorType, TimeType>> operators;

        /**
         * Operator splitting stages.
         */
        std::vector<OSPair<TimeType>> stages;

        /**
         * Reference vector for global state properties.
         */
        VectorType ref_state;

        /**
         * Status structure of the object.
         */
        Status status;
    };
}
