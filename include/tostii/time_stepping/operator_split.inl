#pragma once

#include <tostii/time_stepping/operator_split.h>

namespace tostii::TimeStepping
{
    template<typename BVectorType, typename TimeType>
    OperatorSplit<BVectorType, TimeType>::OperatorSplit(
        const std::vector<OSOperator<BVectorType, TimeType>>& operators,
        const os_method_t<TimeType> method)
        : OperatorSplit(operators, os_method<TimeType>::to_os_pairs(method))
    { }

    template<typename BVectorType, typename TimeType>
    OperatorSplit<BVectorType, TimeType>::OperatorSplit(
        const std::vector<OSOperator<BVectorType, TimeType>>& operators,
        const os_method_t<TimeType> method,
        const std::vector<OSMask>& mask,
        const BVectorType& ref)
        : OperatorSplit(operators, os_method<TimeType>::to_os_pairs(method), mask, ref)
    { }

    template<typename BVectorType, typename TimeType>
    OperatorSplit<BVectorType, TimeType>::OperatorSplit(
        const std::vector<OSOperator<BVectorType, TimeType>>& operators,
        const std::vector<OSPair<TimeType>>& stages)
        : operators(operators)
        , stages(stages)
    { }

    template<typename BVectorType, typename TimeType>
    OperatorSplit<BVectorType, TimeType>::OperatorSplit(
        const std::vector<OSOperator<BVectorType, TimeType>>& operators,
        const std::vector<OSPair<TimeType>>& stages,
        const std::vector<OSMask>& mask,
        const BVectorType& ref)
        : operators(operators)
        , stages(stages)
        , mask(mask)
        , ref_state(ref)
        , nblocks(mask.size())
        , blockrefs(mask.size())
    {
        // Allocate space for block refs.
        for (size_t i = 0; i < mask.size(); ++i)
        {
            nblocks[i] = mask[i].size();
            blockrefs[i].reinit(nblocks[i]);
        }
    }

    template<typename BVectorType, typename TimeType>
    TimeType OperatorSplit<BVectorType, TimeType>::evolve_one_time_step(
        std::vector<std::function<void(
            const TimeType,
            const BVectorType&,
            BVectorType&)>>& /*f*/,
        std::vector<std::function<void(
            const TimeType,
            const TimeType,
            const BVectorType&,
            BVectorType&)>>& /*id_minus_tau_J_inverse*/,
        TimeType t,
        TimeType delta_t,
        BVectorType& y)
    {
        // Not implemented
        return evolve_one_time_step(t, delta_t, y);
    }

    template<typename BVectorType, typename TimeType>
    TimeType OperatorSplit<BVectorType, TimeType>::evolve_one_time_step(
        TimeType t,
        TimeType delta_t,
        BVectorType& y)
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
            TimeStepping<BVectorType, TimeType>* method = operators[op].method;
            std::vector<f_fun_type> function = { operators[op].function };
            std::vector<jac_fun_type> id_minus_tau_J_inverse = { operators[op].id_minus_tau_J_inverse };

            // Update blockref pointers for this stage's state
            const auto& m = mask[op];
            for (unsigned int j = 0; j < nblocks[op]; ++j)
            {
                std::swap(blockrefs[op].block(j), y.block(m[j]));
            }
            blockrefs[op].collect_sizes();

            // Evolve this operator with the masked sub-blocks
            op_time[op] = method->evolve_one_time_step(
                function,
                id_minus_tau_J_inverse,
                op_time[op],
                alpha * delta_t,
                blockrefs[op]);
            
            // Swap blocks back
            for (unsigned int j = 0; j < nblocks[op]; ++j)
            {
                std::swap(blockrefs[op].block(j), y.block(m[j]));
            }
            blockrefs[op].collect_sizes();
        }

        return t + delta_t;
    }

    template<typename BVectorType, typename TimeType>
    const typename OperatorSplit<BVectorType, TimeType>::Status&
    OperatorSplit<BVectorType, TimeType>::get_status() const
    {
        return status;
    }
}
