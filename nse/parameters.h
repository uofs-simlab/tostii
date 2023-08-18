#pragma once

#include <deal.II/base/parameter_handler.h>

#include <tostii/time_stepping/runge_kutta.h>

namespace NSE
{
    using namespace dealii;
}

namespace NSE::Parameters
{
    using namespace dealii;

    struct FEMParameters
    {
        unsigned int polynomial_degree;
        unsigned int quadrature_order;
        unsigned int refinement_level;

        static void declare_parameters(
            ParameterHandler& prm);
        void parse_parameters(
            ParameterHandler& prm);
    };

    struct TimeSteppingParameters
    {
        unsigned int n_time_steps;
        tostii::TimeStepping::runge_kutta_method rk_method;

        static void declare_parameters(
            ParameterHandler& prm);
        void parse_parameters(
            ParameterHandler& prm);
    };

    struct LinearSolverParameters
    {
        double tolerance;
        double max_iterations;

        static void declare_parameters(
            ParameterHandler& prm);
        void parse_parameters(
            ParameterHandler& prm);
    };

    struct OutputParameters
    {
        unsigned int n_output_files;
        std::string output_prefix;
        unsigned int n_checkpoints;
        std::string checkpoint_path;

        static void declare_parameters(
            ParameterHandler& prm);
        void parse_parameters(
            ParameterHandler& prm);
    };

    struct AllParameters
        : public FEMParameters
        , public TimeSteppingParameters
        , public LinearSolverParameters
        , public OutputParameters
    {
        AllParameters(
            const std::string& path);

        static void declare_parameters(
            ParameterHandler& prm);
        void parse_parameters(
            ParameterHandler& prm);
    };
}
