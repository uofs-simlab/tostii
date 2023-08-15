#include "parameters.h"

namespace NSE::Parameters
{
    void FEMParameters::declare_parameters(
        ParameterHandler& prm)
    {
        prm.enter_subsection("Finite Element parameters");

        prm.declare_entry(
            "Polynomial degree",
            "2",
            Patterns::Integer(1),
            "Finite Element polynomial degree");
        prm.declare_entry(
            "Refinement level",
            "8",
            Patterns::Integer(0),
            "Number of global refinements");

        prm.leave_subsection();
    }

    void FEMParameters::parse_parameters(
        ParameterHandler& prm)
    {
        prm.enter_subsection("Finite Element parameters");

        polynomial_degree = prm.get_integer("Polynomial degree");
        quadrature_order = polynomial_degree + 2;
        refinement_level = prm.get_integer("Refinement level");

        prm.leave_subsection();
    }

    void TimeSteppingParameters::declare_parameters(
        ParameterHandler& prm)
    {
        prm.enter_subsection("Time Stepping parameters");

        prm.declare_entry(
            "Number of time steps",
            "2097152",
            Patterns::Integer(1),
            "Number of time steps.\n"
            "Delta t is 1 / n_time_steps");
        prm.declare_entry(
            "Runge-Kutta method",
            "CRANK_NICOLSON",
            Patterns::Anything(),
            "Runge-Kutta method to use for time stepping");

        prm.leave_subsection();
    }

    void TimeSteppingParameters::parse_parameters(
        ParameterHandler& prm)
    {
        prm.enter_subsection("Time Stepping parameters");

        n_time_steps = prm.get_integer("Number of time steps");
        rk_method = tostii::TimeStepping::runge_kutta_enums.at(prm.get("Runge-Kutta method"));

        prm.leave_subsection();
    }

    void LinearSolverParameters::declare_parameters(
        ParameterHandler& prm)
    {
        prm.enter_subsection("Linear Solver parameters");

        prm.declare_entry(
            "Solver tolerance",
            "1e-12",
            Patterns::Double(0.),
            "Solver tolerance");
        prm.declare_entry(
            "Max iteration multiplier",
            "100",
            Patterns::Integer(1),
            "Maximum number of linear solver iterations.\n"
            "Scaled by total DoF count");

        prm.leave_subsection();
    }

    void LinearSolverParameters::parse_parameters(
        ParameterHandler& prm)
    {
        prm.enter_subsection("Linear Solver parameters");

        tolerance = prm.get_double("Solver tolerance");
        max_iterations = prm.get_integer("Max iteration multiplier");

        prm.leave_subsection();
    }

    void OutputParameters::declare_parameters(
        ParameterHandler& prm)
    {
        prm.enter_subsection("Output parameters");

        prm.declare_entry(
            "Number of output files",
            "64",
            Patterns::Integer(0),
            "Number of pvtu output files");
        prm.declare_entry(
            "Output prefix",
            "./results/solution",
            Patterns::FileName(Patterns::FileName::output),
            "Prefix of solution files");
        prm.declare_entry(
            "Number of checkpoints",
            "64",
            Patterns::Integer(0),
            "Number of checkpoints to create");
        prm.declare_entry(
            "Checkpoint path",
            "./saves",
            Patterns::DirectoryName(),
            "Checkpoint directory");

        prm.leave_subsection();
    }

    void OutputParameters::parse_parameters(
        ParameterHandler& prm)
    {
        prm.enter_subsection("Output parameters");

        n_output_files = prm.get_integer("Number of output files");
        output_prefix = prm.get("Output prefix");
        n_checkpoints = prm.get_integer("Number of checkpoints");
        checkpoint_path = prm.get("Checkpoint path");

        prm.leave_subsection();
    }

    AllParameters::AllParameters(
        const std::string& path)
    {
        ParameterHandler prm;
        declare_parameters(prm);

        prm.parse_input(path);
        parse_parameters(prm);
    }

    void AllParameters::declare_parameters(
        ParameterHandler& prm)
    {
        FEMParameters::declare_parameters(prm);
        TimeSteppingParameters::declare_parameters(prm);
        LinearSolverParameters::declare_parameters(prm);
        OutputParameters::declare_parameters(prm);
    }

    void AllParameters::parse_parameters(
        ParameterHandler& prm)
    {
        FEMParameters::parse_parameters(prm);
        TimeSteppingParameters::parse_parameters(prm);
        LinearSolverParameters::parse_parameters(prm);
        OutputParameters::parse_parameters(prm);
    }
}
