#pragma once

#include <deal.II/base/parameter_handler.h>

#include <array>
#include <unordered_map>

namespace Bidomain::Parameters
{
    using namespace dealii;

    struct time_stepper
    {
        enum type
            : unsigned int
        {
            BACKWARD_EULER,
            CRANK_NICOLSON,
            INVALID
        };

        static const std::array<std::pair<std::string, double>, INVALID> info;
        static const std::unordered_map<std::string, type> values;

        static Patterns::Selection pattern();

        static type from_string(const std::string& name);
        static const std::string& to_string(const type value);
        static double to_theta(const type value);
    };

    typedef time_stepper::type time_stepper_t;

    struct FEMParameters
    {
        unsigned int dim;
        unsigned int polynomial_degree;
        unsigned int quadrature_order;
        unsigned int global_refinement_level;

        static void declare_parameters(ParameterHandler& prm);
        void parse_parameters(ParameterHandler& prm);
    };

    struct TimeSteppingParameters
    {
        unsigned int n_time_steps;
        unsigned int initial_time_step;
        double final_time;
        time_stepper_t time_stepping;

        static void declare_parameters(ParameterHandler& prm);
        void parse_parameters(ParameterHandler& prm);
    };

    struct PassiveParameters
    {
        double Rm;

        static void declare_parameters(ParameterHandler& prm);
        void parse_parameters(ParameterHandler& prm);
    };

    struct FHNParameters
    {
        double epsilon;
        double beta;
        double gamma;

        static void declare_parameters(ParameterHandler& prm);
        void parse_parameters(ParameterHandler& prm);
    };

    struct CellParameters
    {
        double chi;
        double Cm;
        double sigmai;
        double sigmae;
        PassiveParameters passive;
        FHNParameters fhn;

        static void declare_parameters(ParameterHandler& prm);
        void parse_parameters(ParameterHandler& prm);
    };

    struct LinearSolverParameters
    {
        double tolerance;
        double max_iterations;

        static void declare_parameters(ParameterHandler& prm);
        void parse_parameters(ParameterHandler& prm);
    };

    struct OutputParameters
    {
        unsigned int n_output_files;
        std::string output_prefix;

        static void declare_parameters(ParameterHandler& prm);
        void parse_parameters(ParameterHandler& prm);
    };

    struct AllParameters
        : public FEMParameters
        , public TimeSteppingParameters
        , public CellParameters
        , public LinearSolverParameters
        , public OutputParameters
    {
        AllParameters(const std::string& input_file);

        static void declare_parameters(ParameterHandler& prm);
        void parse_parameters(ParameterHandler& prm);
    };
}
