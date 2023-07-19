#pragma once

#include <deal.II/base/geometry_info.h>

#include <deal.II/grid/tria.h>

namespace tostii::GridGenerator
{
    /**
     * Creates a @p dim -dimensional hyper cube $[left, right]^dim$,
     * missing the rightmost corner ($2^-dim$ of the total volume).
     * By specifying a number of @p refinements,
     * the cells nearest the bend of the L are selectively refined.
     * If the @p colorize flag is set to @p true ,
     * the faces in the $x_k$-direction are assigned the boundary id:
     *  - $2^{2k}(1)$ if the face touches the leftmost vertex.
     *  - $2^{2k}(2)$ if the face would touch the rightmost vertex, if it were included.
     *  - $2^{2k}(3)$ if the face is exposed by removing the rightmost corner.
     */
    template<int dim, int spacedim>
    void hyper_L(
        dealii::Triangulation<dim, spacedim>& tria,
        const double left = -1.,
        const double right = 1.,
        const bool colorize = false,
        const unsigned int refinements = 0);

    /**
     * Creates a @p dim -dimension hyper rectangle spanned by $[p_1, p_2]$,
     * missing the corner closest to $p_2$ ($2^-dim$ of the total volume).
     * @p colorize and @p refinements are as with the above overload.
     */
    template<int dim, int spacedim>
    void hyper_L(
        dealii::Triangulation<dim, spacedim>& tria,
        const dealii::Point<dim>& p_1,
        const dealii::Point<dim>& p_2,
        const bool colorize = false,
        const unsigned int refinements = 0);
}
