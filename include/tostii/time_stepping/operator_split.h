#pragma once

#include <tostii/time_stepping/time_stepping.h>

#include <complex>
#include <string>
#include <array>
#include <vector>
#include <unordered_map>
#include <functional>

/***/
namespace tostii::TimeStepping
{
    /**
     * Pair for encoding an operator split stage:
     * - op_num = Which operator
     * - alpha = alpha value for this stage
    */
    template<typename TimeType = double>
    struct OSPair
    {
        unsigned int op_num;
        TimeType alpha;
    };

    /**
     * Backwards-compatibility typedef.
     */
    template<typename TimeType = double>
    using OSpair = OSPair<TimeType>;

    typedef std::vector<size_t> OSMask;

    /**
     * Backwards-compatibility typedef.
     */
    typedef OSMask OSmask;

    /***/
    template<typename VectorType, typename TimeType = double>
    struct OSOperator
    {
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
        typedef std::function<void(
            const TimeType,
            const TimeType,
            const VectorType&,
            VectorType&)> jac_fun_type;
        
        /**
         * Time-stepping method for this operator
         */
        TimeStepping<VectorType, TimeType>* method;

        /**
         * Function to be integrated
         */
        std::function<void(
            const TimeType,
            const VectorType&,
            VectorType&)> function;
        
        /**
         * Jacobian solver for @var function
         */
        std::function<void(
            const TimeType,
            const TimeType,
            const VectorType&,
            VectorType&)> id_minus_tau_J_inverse;
    };

    template<typename TimeType>
    struct os_method
    { };

    template<>
    struct os_method<double>
    {
        enum type
            : unsigned int
        {
            GODUNOV,
            STRANG,
            RUTH,
            YOSHIDA,
            GODUNOV3,
            STRANG3,
            PP_3_A_3,

            INVALID
        };

        /**
         * Coefficients for Yoshida splitting
         */
        static const std::array<double, 2> yoshida_omega;

        /**
         * Coefficients for PP_3_A_3
         */
        static const std::array<double, 3> a_pp3a3;
        /**
         * Coefficients for PP_3_A_3
         */
        static const std::array<double, 3> b_pp3a3;
        /**
         * Coefficients for PP_3_A_3
         */
        static const std::array<double, 3> c_pp3a3;

        static const std::array<std::pair<std::string, std::vector<OSPair<double>>>, INVALID> info;
        static const std::unordered_map<std::string, type> values;

        static type from_string(const std::string& name);
        static const std::string& to_string(const type method);
        static const std::vector<OSPair<double>>& to_os_pairs(const type method);
    };

    template<>
    struct os_method<std::complex<double>>
    {
        enum type
            : unsigned int
        {
            MILNE_2_2_C_I,
            MILNE_2_2_C_I_ASC,
            MILNE_2_2_C_II,
            A_3_3_C,
            AKT_2_2_C,
            PP_3_A_3_C,

            INVALID
        };

        /**
         * Coefficients for PP_3_A_3_c
         */
        static const std::array<std::complex<double>, 3> a_pp3a3c;

        /**
         * Coefficients for PP_3_A_3_c
         */
        static const std::array<std::complex<double>, 3> b_pp3a3c;

        /**
         * Coefficients for PP_3_A_3_c
         */
        static const std::array<std::complex<double>, 3> c_pp3a3c;

        static const std::array<std::pair<std::string, std::vector<OSPair<std::complex<double>>>>, INVALID> info;
        static const std::unordered_map<std::string, type> values;

        static type from_string(const std::string& name);
        static const std::string& to_string(const type method);
        static const std::vector<OSPair<std::complex<double>>>& to_os_pairs(const type method);
    };

    template<typename TimeType>
    using os_method_t = typename os_method<TimeType>::type;

    /**
     * Class for OperatorSplit time stepping
     * - @tparam BVectorType should be a block vector type
     * - - The number of blocks should equal the number of components
     */
    template<typename BVectorType, typename TimeType = double>
    class OperatorSplit
        : public TimeStepping<BVectorType, TimeType>
    {
    public:
        /**
         * Backwards-compatibility typedef.
         */
        typedef std::function<void(
            const TimeType,
            const BVectorType&,
            BVectorType&)> f_fun_type;
        
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
            const BVectorType&,
            BVectorType&)> jac_fun_type;

        /**
         * Backwards-compatibility typedef.
         */
        typedef std::vector<jac_fun_type> jac_vfun_type;

        /**
         * Constructor for named OS methods.
         */
        OperatorSplit(
            const std::vector<OSOperator<BVectorType, TimeType>>& operators,
            const os_method_t<TimeType> method);
        
        /**
         * Constructor for named OS methods.
         */
        OperatorSplit(
            const std::vector<OSOperator<BVectorType, TimeType>>& operators,
            const os_method_t<TimeType> method,
            const std::vector<OSMask>& mask,
            const BVectorType& ref);
        
        /**
         * Constructor for explicitly-specified stages.
         */
        OperatorSplit(
            const std::vector<OSOperator<BVectorType, TimeType>>& operators,
            const std::vector<OSPair<TimeType>>& stages);
        
        /**
         * Constructor for explicitly-specified stages.
         */
        OperatorSplit(
            const std::vector<OSOperator<BVectorType, TimeType>>& operators,
            const std::vector<OSPair<TimeType>>& stages,
            const std::vector<OSMask>& mask,
            const BVectorType& ref);

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
                const BVectorType&,
                BVectorType&)>>& f,
            std::vector<std::function<void(
                const TimeType,
                const TimeType,
                const BVectorType&,
                BVectorType&)>>& id_minus_tau_J_inverse,
            TimeType t,
            TimeType delta_t,
            BVectorType& y) override;
        
        /**
         * Trimmed down function for an OS method that has been set up.
         */
        TimeType evolve_one_time_step(
            TimeType t,
            TimeType delta_t,
            BVectorType& y);

        struct Status
            : public TimeStepping<BVectorType, TimeType>::Status
        { };

        /**
         * Return the status of the current object.
         */
        const Status& get_status() const override;

    private:
        /**
         * Operator definitions.
         */
        std::vector<OSOperator<BVectorType, TimeType>> operators;

        /**
         * Operator splitting stages
         */
        std::vector<OSPair<TimeType>> stages;

        /**
         * Which blocks participate in which operators.
         */
        std::vector<OSMask> mask;

        /**
         * Reference vector for global state properties.
         */
        BVectorType ref_state;

        /**
         * Number of block vectors in each state.
         */
        std::vector<unsigned int> nblocks;

        /**
         * Vector of references to each operator's needed blocks.
         */
        std::vector<BVectorType> blockrefs;

        /**
         * Name of the method being employed (empty if given explicit method).
         */
        std::string method_name;

        /**
         * Status structure of the object.
         */
        Status status;
    };
}
