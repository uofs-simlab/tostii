#include <tostii/grid/L_shape.inl>

namespace tostii::GridGenerator
{
    template void hyper_L(
        dealii::Triangulation<1, 1>&,
        const double,
        const double,
        const bool,
        const unsigned int);
    template void hyper_L(
        dealii::Triangulation<1, 2>&,
        const double,
        const double,
        const bool,
        const unsigned int);
    template void hyper_L(
        dealii::Triangulation<1, 3>&,
        const double,
        const double,
        const bool,
        const unsigned int);
    
    template void hyper_L(
        dealii::Triangulation<2, 2>&,
        const double,
        const double,
        const bool,
        const unsigned int);
    template void hyper_L(
        dealii::Triangulation<2, 3>&,
        const double,
        const double,
        const bool,
        const unsigned int);
    
    template void hyper_L(
        dealii::Triangulation<3, 3>&,
        const double,
        const double,
        const bool,
        const unsigned int);

    template void hyper_L(
        dealii::Triangulation<1, 1>&,
        const dealii::Point<1>&,
        const dealii::Point<1>&,
        const bool,
        const unsigned int);
    template void hyper_L(
        dealii::Triangulation<1, 2>&,
        const dealii::Point<1>&,
        const dealii::Point<1>&,
        const bool,
        const unsigned int);
    template void hyper_L(
        dealii::Triangulation<1, 3>&,
        const dealii::Point<1>&,
        const dealii::Point<1>&,
        const bool,
        const unsigned int);

    template void hyper_L(
        dealii::Triangulation<2, 2>&,
        const dealii::Point<2>&,
        const dealii::Point<2>&,
        const bool,
        const unsigned int);
    template void hyper_L(
        dealii::Triangulation<2, 3>&,
        const dealii::Point<2>&,
        const dealii::Point<2>&,
        const bool,
        const unsigned int);

    template void hyper_L(
        dealii::Triangulation<3, 3>&,
        const dealii::Point<3>&,
        const dealii::Point<3>&,
        const bool,
        const unsigned int);
}
