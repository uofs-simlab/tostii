#include "parameters.h"

#include <algorithm>
#include <numeric>

namespace Bidomain::Parameters
{
    void FEMParameters::declare_parameters(ParameterHandler& prm)
    {
        prm.enter_subsection("FEM Parameters");

        prm.declare_entry(
            "Dimension value",
            "2",
            Patterns::Integer(),
            "Problem dimension (2 or 3)");
        prm.declare_entry(
            "Polynomial degree",
            "1",
            Patterns::Integer(),
            "Polynomial degree for FEM discretization");
        prm.declare_entry(
            "Global refinement value",
            "1",
            Patterns::Integer(),
            "Global refinement level");
        prm.declare_entry(
            "Adaptive refinement",
            "true",
            Patterns::Bool(),
            "Apply adaptive refinement");
        prm.declare_entry(
            "Boundary condition type",
            "Neumann",
            Patterns::Selection("Neumann|Dirichlet"),
            "Boundary condition type (Neumann or Dirichlet)");

        prm.leave_subsection();
    }

    void FEMParameters::parse_parameters(ParameterHandler& prm)
    {
        prm.enter_subsection("FEM Parameters");

        dim = prm.get_integer("Dimension value");
        polynomial_degree = prm.get_integer("Polynomial degree");
        quadrature_order = polynomial_degree + 2;
        global_refinement_level = prm.get_integer("Global refinement value");

        prm.leave_subsection();
    }

    void TimeSteppingParameters::declare_parameters(ParameterHandler& prm)
    {
        prm.enter_subsection("Time Stepping Parameters");
        
        prm.declare_entry(
            "Number of time steps",
            "1",
            Patterns::Integer(),
            "Number of time steps (positive integer)");
        prm.declare_entry(
            "Initial time step",
            "0",
            Patterns::Integer(),
            "Starting time step");
        prm.declare_entry(
            "Final time value",
            "1e0",
            Patterns::Double(),
            "Final time tf (in ms)");
        prm.declare_entry(
            "Membrane stepper",
            "FORWARD_EULER",
            Patterns::Anything(),
            "Membrane time stepping method");
        prm.declare_entry(
            "Tissue stepper",
            "BACKWARD_EULER",
            Patterns::Anything(),
            "Tissue time stepping method");
        prm.declare_entry(
            "OS stepper",
            "Godunov",
            Patterns::Anything(),
            "Operator split time stepping method");

        prm.leave_subsection();
    }

    void TimeSteppingParameters::parse_parameters(ParameterHandler& prm)
    {
        prm.enter_subsection("Time Stepping Parameters");

        n_time_steps = prm.get_integer("Number of time steps");
        initial_time_step = prm.get_integer("Initial time step");
        final_time = prm.get_double("Final time value");

        {
            using namespace tostii::TimeStepping;
            membrane_stepper = runge_kutta_enums.at(prm.get("Membrane stepper"));
            tissue_stepper = runge_kutta_enums.at(prm.get("Tissue stepper"));
            os_stepper = os_method<double>::from_string(prm.get("OS stepper"));
        }

        prm.leave_subsection();
    }

    void PassiveParameters::declare_parameters(ParameterHandler& prm)
    {
        prm.enter_subsection("Passive Cell Model");

        prm.declare_entry(
            "Rm value",
            "1e0",
            Patterns::Double(),
            "Resistance");

        prm.leave_subsection();
    }

    void PassiveParameters::parse_parameters(ParameterHandler& prm)
    {
        prm.enter_subsection("Passive Cell Model");

        Rm = prm.get_double("Rm value");

        prm.leave_subsection();
    }

    void FHNParameters::declare_parameters(ParameterHandler& prm)
    {
        prm.enter_subsection("FitzHugh-Nagumo Cell Model");

        prm.declare_entry(
            "epsilon value",
            "0.1",
            Patterns::Double(),
            "FitzHugh-Nagumo epsilon");
        prm.declare_entry(
            "beta value",
            "1.0",
            Patterns::Double(),
            "FitzHugh-Nagumo beta");
        prm.declare_entry(
            "gamma value",
            "0.5",
            Patterns::Double(),
            "FitzHugh-Nagumo gamma");

        prm.leave_subsection();
    }

    void FHNParameters::parse_parameters(ParameterHandler& prm)
    {
        prm.enter_subsection("FitzHugh-Nagumo Cell Model");

        epsilon = prm.get_double("epsilon value");
        beta = prm.get_double("beta value");
        gamma = prm.get_double("gamma value");

        prm.leave_subsection();
    }

    void CellParameters::declare_parameters(ParameterHandler& prm)
    {
        prm.enter_subsection("Cell Parameters");

        prm.declare_entry(
            "chi value",
            "1e0",
            Patterns::Double(),
            "Cells per unit volume");
        prm.declare_entry(
            "Cm value",
            "1e0",
            Patterns::Double(),
            "Capacitance");
        prm.declare_entry(
            "sigmai value",
            "1e0",
            Patterns::Double(),
            "Intracellular active conductivity");
        prm.declare_entry(
            "sigmae value",
            "1e0",
            Patterns::Double(),
            "Extracellular active conductivity");
        prm.declare_entry(
            "sigmaix value",
            "1e0",
            Patterns::Double(),
            "Intracellular active conductivity (xx component)");
        prm.declare_entry(
            "sigmaiy value",
            "1e0",
            Patterns::Double(),
            "Intracellular active conductivity (yy component)");
        prm.declare_entry(
            "sigmaex value",
            "1e0",
            Patterns::Double(),
            "Extracellular active conductivity (xx component)");
        prm.declare_entry(
            "sigmaey value",
            "1e0",
            Patterns::Double(),
            "Extracellular active conductivity (yy component)");

        PassiveParameters::declare_parameters(prm);
        FHNParameters::declare_parameters(prm);

        prm.leave_subsection();
    }

    void CellParameters::parse_parameters(ParameterHandler& prm)
    {
        prm.enter_subsection("Cell Parameters");

        chi = prm.get_double("chi value");
        Cm = prm.get_double("Cm value");
        sigmai = prm.get_double("sigmai value");
        sigmae = prm.get_double("sigmae value");
        sigmaix = prm.get_double("sigmaix value");
        sigmaiy = prm.get_double("sigmaiy value");
        sigmaex = prm.get_double("sigmaex value");
        sigmaey = prm.get_double("sigmaey value");

        passive.parse_parameters(prm);
        fhn.parse_parameters(prm);

        prm.leave_subsection();
    }

    void LinearSolverParameters::declare_parameters(ParameterHandler& prm)
    {
        prm.enter_subsection("Linear Solver Parameters");

        prm.declare_entry(
            "Tolerance",
            "1e-12",
            Patterns::Double(),
            "KINSOL tolerance");
        prm.declare_entry(
            "Max iteration multiplier",
            "10",
            Patterns::Integer(),
            "Linear solver iterations (multiples of the system matrix size)");

        prm.leave_subsection();
    }

    void LinearSolverParameters::parse_parameters(ParameterHandler& prm)
    {
        prm.enter_subsection("Linear Solver Parameters");

        tolerance = prm.get_double("Tolerance");
        max_iterations = prm.get_double("Max iteration multiplier");

        prm.leave_subsection();
    }

    void OutputParameters::declare_parameters(ParameterHandler& prm)
    {
        prm.enter_subsection("Output Parameters");

        prm.declare_entry(
            "Number of pvtu files",
            "1",
            Patterns::Integer(),
            "Number of pvtu files to visualize solution");
        prm.declare_entry(
            "Output prefix",
            "./solution",
            Patterns::FileName(Patterns::FileName::output),
            "pvtu filename prefix");

        prm.leave_subsection();
    }

    void OutputParameters::parse_parameters(ParameterHandler& prm)
    {
        prm.enter_subsection("Output Parameters");

        n_output_files = prm.get_integer("Number of pvtu files");
        output_prefix = prm.get("Output prefix");

        prm.leave_subsection();
    }

    AllParameters::AllParameters(const std::string& input_file)
    {
        ParameterHandler prm;
        declare_parameters(prm);

        prm.parse_input(input_file);
        parse_parameters(prm);
    }

    void AllParameters::declare_parameters(ParameterHandler& prm)
    {
        FEMParameters::declare_parameters(prm);
        TimeSteppingParameters::declare_parameters(prm);
        CellParameters::declare_parameters(prm);
        LinearSolverParameters::declare_parameters(prm);
        OutputParameters::declare_parameters(prm);
    }

    void AllParameters::parse_parameters(ParameterHandler& prm)
    {
        FEMParameters::parse_parameters(prm);
        TimeSteppingParameters::parse_parameters(prm);
        CellParameters::parse_parameters(prm);
        LinearSolverParameters::parse_parameters(prm);
        OutputParameters::parse_parameters(prm);
    }
}
