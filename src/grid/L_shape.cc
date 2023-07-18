#include <tostii/grid/L_shape.inl>

namespace tostii::GridGenerator
{
    template void hyper_L(
        dealii::Triangulation<1, 1>&,
        const double,
        const double,
        const unsigned int,
        const bool);
    template void hyper_L(
        dealii::Triangulation<1, 2>&,
        const double,
        const double,
        const unsigned int,
        const bool);
    template void hyper_L(
        dealii::Triangulation<1, 3>&,
        const double,
        const double,
        const unsigned int,
        const bool);

    template void hyper_L(
        dealii::Triangulation<2, 2>&,
        const double,
        const double,
        const unsigned int,
        const bool);
    template void hyper_L(
        dealii::Triangulation<2, 3>&,
        const double,
        const double,
        const unsigned int,
        const bool);

    template void hyper_L(
        dealii::Triangulation<3, 3>&,
        const double,
        const double,
        const unsigned int,
        const bool);
}
