#pragma once

#include <deal.II/base/parameter_handler.h>

#include <tostii/time_stepping/runge_kutta.h>
#include <tostii/time_stepping/operator_split.h>

#include <array>
#include <unordered_map>

namespace Bidomain::Parameters
{
    using namespace dealii;

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
        tostii::TimeStepping::runge_kutta_method membrane_stepper;
        tostii::TimeStepping::runge_kutta_method tissue_stepper;
        tostii::TimeStepping::os_method_t<double> os_stepper;

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
        double sigmaix;
        double sigmaiy;
        double sigmaex;
        double sigmaey;
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
