#pragma once

#include <tostii/time_stepping/operator_split.inl>
#include <tostii/time_stepping/operator_split_single.h>

namespace tostii::TimeStepping
{
    template<typename VectorType, typename TimeType>
    OperatorSplitSingle<VectorType, TimeType>::OperatorSplitSingle(
        const std::vector<OSOperator<VectorType, TimeType>>& operators,
        const os_method_t<TimeType> method)
        : OperatorSplitSingle(operators, os_method<TimeType>::to_os_pairs(method))
    { }

    template<typename VectorType, typename TimeType>
    OperatorSplitSingle<VectorType, TimeType>::OperatorSplitSingle(
        const std::vector<OSOperator<VectorType, TimeType>>& operators,
        const os_method_t<TimeType> method,
        const VectorType& ref)
        : OperatorSplitSingle(operators, os_method<TimeType>::to_os_pairs(method), ref)
    { }

    template<typename VectorType, typename TimeType>
    OperatorSplitSingle<VectorType, TimeType>::OperatorSplitSingle(
        const std::vector<OSOperator<VectorType, TimeType>>& operators,
        const std::vector<OSPair<TimeType>>& stages)
        : operators(operators)
        , stages(stages)
    { }

    template<typename VectorType, typename TimeType>
    OperatorSplitSingle<VectorType, TimeType>::OperatorSplitSingle(
        const std::vector<OSOperator<VectorType, TimeType>>& operators,
        const std::vector<OSPair<TimeType>>& stages,
        const VectorType& ref)
        : operators(operators)
        , stages(stages)
        , ref_state(ref)
    { }

    template<typename VectorType, typename TimeType>
    TimeType OperatorSplitSingle<VectorType, TimeType>::evolve_one_time_step(
        std::vector<std::function<void(
            const TimeType,
            const VectorType&,
            VectorType&)>>& /*f*/,
        std::vector<std::function<void(
            const TimeType,
            const TimeType,
            const VectorType&,
            VectorType&)>>& /*id_minus_tau_J_inverse*/,
        TimeType t,
        TimeType delta_t,
        VectorType& y)
    {
        return evolve_one_time_step(t, delta_t, y);
    }

    template<typename VectorType, typename TimeType>
    TimeType OperatorSplitSingle<VectorType, TimeType>::evolve_one_time_step(
        TimeType t,
        TimeType delta_t,
        VectorType& y)
    {
        // Current time for each operator
        std::vector<TimeType> op_time(operators.size(), t);

        // Loop over stages
        for (const auto& pair : stages)
        {
            // Get stage info
            unsigned int op = pair.op_num;
            TimeType alpha = pair.alpha;

            // Operator info for this stage
            TimeStepping<VectorType, TimeType>* method = operators[op].method;
            std::vector<f_fun_type> function = { operators[op].function };
            std::vector<jac_fun_type> id_minus_tau_J_inverse = { operators[op].id_minus_tau_J_inverse };

            // Evolve this operator
            op_time[op] = method->evolve_one_time_step(
                function,
                id_minus_tau_J_inverse,
                op_time[op],
                alpha * delta_t,
                y);
        }

        return t + delta_t;
    }

    template<typename VectorType, typename TimeType>
    const typename OperatorSplitSingle<VectorType, TimeType>::Status&
    OperatorSplitSingle<VectorType, TimeType>::get_status() const
    {
        return status;
    }
}
