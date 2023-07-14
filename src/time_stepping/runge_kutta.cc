#include <tostii/time_stepping/runge_kutta.h>

namespace tostii::TimeStepping
{
    const std::array<std::string, INVALID> runge_kutta_strings = { {
        "FORWARD_EULER",
        "EXPLICIT_MIDPOINT",
        "HEUN2",
        "RK_THIRD_ORDER",
        "SSP_THIRD_ORDER",
        "RK_CLASSIC_FOURTH_ORDER",
        "BACKWARD_EULER",
        "IMPLICIT_MIDPOINT",
        "CRANK_NICOLSON",
        "SDIRK_TWO_STAGES",
        "SDIRK_THREE_STAGES",
        "SDIRK_3O4",
        "SDIRK_5O4"
    } };

    const std::unordered_map<std::string, runge_kutta_method> runge_kutta_enums = {
        { "FORWARD_EULER", FORWARD_EULER },
        { "EXPLICIT_MIDPOINT", EXPLICIT_MIDPOINT },
        { "HEUN2", HEUN2 },
        { "RK_THIRD_ORDER", RK_THIRD_ORDER },
        { "SSP_THIRD_ORDER", SSP_THIRD_ORDER },
        { "RK_CLASSIC_FOURTH_ORDER", RK_CLASSIC_FOURTH_ORDER },
        { "BACKWARD_EULER", BACKWARD_EULER },
        { "IMPLICIT_MIDPOINT", IMPLICIT_MIDPOINT },
        { "CRANK_NICOLSON", CRANK_NICOLSON },
        { "SDIRK_TWO_STAGES", SDIRK_TWO_STAGES },
        { "SDIRK_THREE_STAGES", SDIRK_THREE_STAGES },
        { "SDIRK_3O4", SDIRK_3O4 },
        { "SDIRK_5O4", SDIRK_5O4 }
    };
}
