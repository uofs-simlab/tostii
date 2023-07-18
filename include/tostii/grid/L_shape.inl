#pragma once

#include <tostii/grid/L_shape.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/iterator_range.h>

#include <deal.II/grid/tria_description.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/vector.h>

#include <algorithm>

namespace tostii::GridGenerator
{
    template<int dim, int spacedim>
    void hyper_L(
        dealii::Triangulation<dim, spacedim>& tria,
        const double left,
        const double right,
        const unsigned int refinements,
        const bool colorize)
    {
        dealii::Point<spacedim> center;

        {
            dealii::Point<spacedim> p1, p2;
            for (unsigned int i = 0; i < dim; ++i)
            {
                p1[i] = left;
                p2[i] = right;
            }

            /* TODO: reserve with 3^dim - 1 points and 2^dim - 1 cells? */
            std::vector<dealii::Point<spacedim>> points;
            std::vector<dealii::CellData<dim>> cells;

            switch (dim)
            {
            case 1:
                points.reserve(2);
                cells.reserve(1);

                /* corners */

                // left
                points[0] = p1;
                cells[0].vertices[0] = 0;

                /* edges */

                // middle
                points[1] = (p1 + p2) / 2.;
                cells[0].vertices[1] = 1;
                center = points[1];

                break;
            case 2:
                points.reserve(8);
                cells.reserve(3);

                /* corners */

                // lower left
                points[0] = p1;
                cells[0].vertices[0] = 0;

                // lower right
                points[2][0] = p2[0]; points[2][1] = p1[1];
                cells[1].vertices[1] = 2;

                // upper left
                points[6][0] = p1[0]; points[6][1] = p2[1];
                cells[2].vertices[2] = 6;

                /* edges */

                // lower middle
                points[1] = (points[0] + points[2]) / 2.;
                cells[0].vertices[1] = 1;
                cells[1].vertices[0] = 1;

                // middle left
                points[3] = (points[0] + points[6]) / 2.;
                cells[0].vertices[2] = 3;
                cells[2].vertices[0] = 3;

                // middle right
                points[5] = (points[2] + p2) / 2.;
                cells[1].vertices[3] = 5;

                // upper middle
                points[7] = (points[6] + p2) / 2.;
                cells[2].vertices[3] = 7;

                /* faces */

                // middle middle
                points[4] = (points[0] + p2) / 2.;
                cells[0].vertices[3] = 4;
                cells[1].vertices[2] = 4;
                cells[2].vertices[1] = 4;
                center = points[4];

                break;
            case 3:
                points.reserve(26);
                cells.reserve(7);

                /* corners */

                // near lower left
                points[0] = p1;
                cells[0].vertices[0] = 0;

                // near lower right
                points[2][0] = p2[0]; points[2][1] = p1[1]; points[2][2] = p1[2];
                cells[1].vertices[1] = 2;

                // near upper left
                points[6][0] = p1[0]; points[6][1] = p2[1]; points[6][2] = p1[2];
                cells[2].vertices[2] = 6;

                // near upper right
                points[8][0] = p2[0]; points[8][1] = p2[1]; points[8][2] = p1[2];
                cells[3].vertices[3] = 8;

                // far lower left
                points[18][0] = p1[0]; points[18][1] = p1[1]; points[18][2] = p2[2];
                cells[4].vertices[4] = 18;

                // far lower right
                points[20][0] = p2[0]; points[20][1] = p1[1]; points[20][2] = p2[2];
                cells[5].vertices[5] = 20;

                // far upper left
                points[24][0] = p1[0]; points[24][1] = p2[1]; points[24][2] = p2[2];
                cells[6].vertices[6] = 24;

                /* edges */

                // near lower middle
                points[1] = (points[0] + points[2]) / 2.;
                cells[0].vertices[1] = 1;
                cells[1].vertices[0] = 1;

                // near middle left
                points[3] = (points[0] + points[6]) / 2.;
                cells[0].vertices[2] = 3;
                cells[2].vertices[0] = 3;

                // near middle right
                points[5] = (points[2] + points[8]) / 2.;
                cells[1].vertices[3] = 5;
                cells[3].vertices[1] = 5;

                // near upper middle
                points[7] = (points[6] + points[8]) / 2.;
                cells[2].vertices[3] = 7;
                cells[3].vertices[2] = 7;

                // middle lower left
                points[9] = (points[0] + points[18]) / 2.;
                cells[0].vertices[4] = 9;
                cells[4].vertices[0] = 9;

                // middle lower right
                points[11] = (points[2] + points[20]) / 2.;
                cells[1].vertices[5] = 11;
                cells[5].vertices[1] = 11;

                // middle upper left
                points[15] = (points[6] + points[21]) / 2.;
                cells[2].vertices[6] = 15;
                cells[6].vertices[2] = 15;

                // middle upper right
                points[17] = (points[8] + p2) / 2.;
                cells[3].vertices[7] = 17;

                // far lower middle
                points[19] = (points[18] + points[20]) / 2.;
                cells[4].vertices[5] = 19;
                cells[5].vertices[4] = 19;

                // far middle left
                points[21] = (points[18] + points[24]) / 2.;
                cells[4].vertices[6] = 21;
                cells[6].vertices[4] = 21;

                // far middle right
                points[23] = (points[20] + p2) / 2.;
                cells[5].vertices[7] = 23;

                // far upper middle
                points[25] = (points[24] + p2) / 2.;
                cells[6].vertices[7] = 25;

                /* faces */

                // near middle middle
                points[4] = (points[0] + points[8]) / 2.;
                cells[0].vertices[3] = 4;
                cells[1].vertices[2] = 4;
                cells[2].vertices[3] = 4;
                cells[3].vertices[0] = 4;

                // middle lower middle
                points[10] = (points[0] + points[20]) / 2.;
                cells[0].vertices[5] = 10;
                cells[1].vertices[4] = 10;
                cells[4].vertices[1] = 10;
                cells[5].vertices[0] = 10;

                // middle middle left
                points[12] = (points[0] + points[24]) / 2.;
                cells[0].vertices[6] = 12;
                cells[2].vertices[4] = 12;
                cells[4].vertices[2] = 12;
                cells[6].vertices[0] = 12;

                // middle middle right
                points[14] = (points[2] + p2) / 2.;
                cells[1].vertices[7] = 14;
                cells[3].vertices[5] = 14;
                cells[5].vertices[3] = 14;

                // middle upper middle
                points[16] = (points[6] + p2) / 2.;
                cells[2].vertices[7] = 16;
                cells[3].vertices[6] = 16;
                cells[6].vertices[3] = 16;

                // far middle middle
                points[22] = (points[18] + p2) / 2.;
                cells[4].vertices[7] = 22;
                cells[5].vertices[6] = 22;
                cells[6].vertices[5] = 22;

                /* center */

                // middle middle middle
                points[13] = (points[0] + p2) / 2.;
                cells[0].vertices[7] = 13;
                cells[1].vertices[6] = 13;
                cells[2].vertices[5] = 13;
                cells[3].vertices[4] = 13;
                cells[4].vertices[3] = 13;
                cells[5].vertices[2] = 13;
                cells[6].vertices[1] = 13;
                center = points[13];

                break;
            default:
                Assert(false, dealii::StandardExceptions::ExcNotImplemented());
            }

            dealii::SubCellData boundaries;

            if (colorize)
            {
                switch (dim)
                {
                case 1:
                    break;
                case 2:
                    boundaries.boundary_lines.reserve(8);

                    /* x-boundaries */

                    boundaries.boundary_lines[0].vertices = { 0, 1 };
                    boundaries.boundary_lines[0].boundary_id = 1;

                    boundaries.boundary_lines[1].vertices = { 1, 2 };
                    boundaries.boundary_lines[1].boundary_id = 1;

                    boundaries.boundary_lines[2].vertices = { 6, 7 };
                    boundaries.boundary_lines[2].boundary_id = 2;

                    boundaries.boundary_lines[3].vertices = { 4, 5 };
                    boundaries.boundary_lines[3].boundary_id = 3;

                    /* y-boundaries */

                    boundaries.boundary_lines[4].vertices = { 0, 3 };
                    boundaries.boundary_lines[4].boundary_id = 4;

                    boundaries.boundary_lines[5].vertices = { 3, 6 };
                    boundaries.boundary_lines[5].boundary_id = 4;

                    boundaries.boundary_lines[6].vertices = { 2, 5 };
                    boundaries.boundary_lines[6].boundary_id = 8;

                    boundaries.boundary_lines[7].vertices = { 4, 7 };
                    boundaries.boundary_lines[7].boundary_id = 12;

                    break;
                case 3:
                    boundaries.boundary_quads.reserve(24);

                    /* x-boundaries */

                    boundaries.boundary_quads[0].vertices = { 0, 1, 9, 10 };
                    boundaries.boundary_quads[0].boundary_id = 1;

                    boundaries.boundary_quads[1].vertices = { 1, 2, 10, 11 };
                    boundaries.boundary_quads[1].boundary_id = 1;

                    boundaries.boundary_quads[2].vertices = { 9, 10, 18, 19 };
                    boundaries.boundary_quads[2].boundary_id = 1;

                    boundaries.boundary_quads[3].vertices = { 10, 11, 19, 20 };
                    boundaries.boundary_quads[3].boundary_id = 1;

                    boundaries.boundary_quads[4].vertices = { 6, 7, 15, 16 };
                    boundaries.boundary_quads[4].boundary_id = 2;

                    boundaries.boundary_quads[5].vertices = { 7, 8, 16, 17 };
                    boundaries.boundary_quads[5].boundary_id = 2;

                    boundaries.boundary_quads[6].vertices = { 15, 16, 24, 25 };
                    boundaries.boundary_quads[6].boundary_id = 2;

                    boundaries.boundary_quads[7].vertices = { 13, 14, 22, 23 };
                    boundaries.boundary_quads[7].boundary_id = 3;

                    /* y-boundaries */

                    boundaries.boundary_quads[8].vertices = { 0, 3, 9, 12 };
                    boundaries.boundary_quads[8].boundary_id = 4;

                    boundaries.boundary_quads[9].vertices = { 3, 6, 12, 15 };
                    boundaries.boundary_quads[9].boundary_id = 4;

                    boundaries.boundary_quads[10].vertices = { 9, 12, 18, 21 };
                    boundaries.boundary_quads[10].boundary_id = 4;

                    boundaries.boundary_quads[11].vertices = { 12, 15, 21, 24 };
                    boundaries.boundary_quads[11].boundary_id = 4;

                    boundaries.boundary_quads[12].vertices = { 2, 5, 11, 14 };
                    boundaries.boundary_quads[12].boundary_id = 8;

                    boundaries.boundary_quads[13].vertices = { 5, 8, 14, 17 };
                    boundaries.boundary_quads[13].boundary_id = 8;

                    boundaries.boundary_quads[14].vertices = { 11, 14, 20, 23 };
                    boundaries.boundary_quads[14].boundary_id = 8;

                    boundaries.boundary_quads[15].vertices = { 13, 16, 22, 25 };
                    boundaries.boundary_quads[15].boundary_id = 12;

                    /* z-boundaries */

                    boundaries.boundary_quads[16].vertices = { 0, 1, 3, 4 };
                    boundaries.boundary_quads[16].boundary_id = 16;

                    boundaries.boundary_quads[17].vertices = { 1, 2, 4, 5 };
                    boundaries.boundary_quads[17].boundary_id = 16;

                    boundaries.boundary_quads[18].vertices = { 3, 4, 6, 7 };
                    boundaries.boundary_quads[18].boundary_id = 16;

                    boundaries.boundary_quads[19].vertices = { 4, 5, 7, 8 };
                    boundaries.boundary_quads[19].boundary_id = 16;

                    boundaries.boundary_quads[20].vertices = { 18, 19, 21, 22 };
                    boundaries.boundary_quads[20].boundary_id = 32;

                    boundaries.boundary_quads[21].vertices = { 19, 20, 22, 23 };
                    boundaries.boundary_quads[21].boundary_id = 32;

                    boundaries.boundary_quads[22].vertices = { 21, 22, 24, 25 };
                    boundaries.boundary_quads[22].boundary_id = 32;

                    boundaries.boundary_quads[23].vertices = { 13, 14, 16, 17 };
                    boundaries.boundary_quads[23].boundary_id = 48;
                }
            }

            tria.create_triangulation(points, cells, boundaries);
        }

        const double radius = std::abs(right - left);
        dealii::Vector<double> temp(spacedim);

        for (unsigned int i = 0; i < refinements; ++i)
        {
            for (auto& cell : tria.active_cell_iterators())
            {
                const dealii::Tensor<1, spacedim> delta = cell->center() - center;

                delta.unroll(temp.begin(), temp.end());
                if (temp.linfty_norm() < radius)
                {
                    cell->set_refine_flag();
                }
            }

            tria.execute_coarsening_and_refinement();
        }
    }
}
