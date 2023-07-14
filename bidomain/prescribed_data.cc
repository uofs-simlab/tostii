#include "prescribed_data.inl"

namespace Bidomain::PrescribedData
{
    template class ExactSolution<2>;
    template class ExactSolution<3>;

    template class TransmembraneRightHandSide<2>;
    template class TransmembraneRightHandSide<3>;

    template class ExtracellularRightHandSide<2>;
    template class ExtracellularRightHandSide<3>;
}
