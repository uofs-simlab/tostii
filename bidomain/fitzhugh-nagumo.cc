#include "fitzhugh-nagumo.inl"

namespace Bidomain::FitzHughNagumo
{
    template class InitialValues<2>;
    template class InitialValues<3>;

    template class Stimulus<2>;
    template class Stimulus<3>;

    template class IntracellularConductivity<2>;
    template class IntracellularConductivity<3>;

    template class ExtracellularConductivity<2>;
    template class ExtracellularConductivity<3>;

    namespace DataPostprocessors
    {
        template class TransmembranePart<2>;
        template class TransmembranePart<3>;

        template class StateVariablePart<2>;
        template class StateVariablePart<3>;

        template class ExtracellularPart<2>;
        template class ExtracellularPart<3>;
    }
}
