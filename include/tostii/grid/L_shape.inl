#pragma once

#include <tostii/grid/L_shape.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/iterator_range.h>
#include <deal.II/base/index_set.h>

#include <deal.II/grid/tria_description.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/vector.h>

#include <algorithm>
#include <fstream>

namespace tostii::GridGenerator
{
    template<int dim, int spacedim>
    void hyper_L(
        dealii::Triangulation<dim, spacedim>& tria,
        const double left,
        const double right,
        const bool colorize,
        const unsigned int refinements)
    {
        dealii::Point<dim> bottom_left, top_right;
        for (unsigned int i = 0; i < dim; ++i)
        {
            bottom_left[i] = left;
            top_right[i] = right;
        }

        hyper_L(
            tria,
            bottom_left,
            top_right,
            colorize,
            refinements);
    }

    template<int dim, int spacedim>
    void hyper_L(
        dealii::Triangulation<dim, spacedim>& tria,
        const dealii::Point<dim>& p_1,
        const dealii::Point<dim>& p_2,
        const bool colorize,
        const unsigned int refinements)
    {
        /**
         * deal.II convention is that $(-1)^(1-b) x_(i+1)$ boundary is given by the index $2i + b$.
         */
        dealii::IndexSet outer_boundary(2 * dim);

        {
            dealii::Point<dim> bottom_left;
            dealii::Point<dim> top_right;
            std::vector<int> n_cells_to_remove(dim);
            for (unsigned int i = 0; i < dim; ++i)
            {
                if (p_2[i] > p_1[i])
                {
                    bottom_left[i] = p_1[i];
                    top_right[i] = p_2[i];
                    n_cells_to_remove[i] = -1;
                    outer_boundary.add_index(2 * i);
                }
                else
                {
                    bottom_left[i] = p_2[i];
                    top_right[i] = p_1[i];
                    n_cells_to_remove[i] = 1;
                    outer_boundary.add_index(2 * i + 1);
                }
            }

            std::vector<unsigned int> repetitions(dim, 2);

            dealii::GridGenerator::subdivided_hyper_L(
                tria,
                repetitions,
                bottom_left,
                top_right,
                n_cells_to_remove);
        }
        
        dealii::Tensor<1, dim> epsilon;
        std::transform(p_1.begin_raw(), p_1.end_raw(), p_2.begin_raw(), epsilon.begin_raw(),
            [](double a, double b) { return std::abs(a - b) / 2.; });
        const dealii::Point<dim> center = (p_1 + p_2) / 2.;

        auto near_center = [&center, &epsilon](const dealii::Point<spacedim>& p)
        {
            bool near = true;
            for (unsigned int i = 0; near && i < dim; ++i)
            {
                near = std::abs(center[i] - p[i]) < epsilon[i];
            }
            return near;
        };

        if (colorize)
        {
            for (auto& cell : tria.active_cell_iterators())
            {
                for (auto face_index : dealii::GeometryInfo<dim>::face_indices())
                {
                    auto face = cell->face(face_index);
                    if (face->at_boundary())
                    {
                        if (outer_boundary.is_element(face_index))
                        {
                            face->set_boundary_id(1 << (2 * (face_index / 2)));
                        }
                        else if (near_center(face->center()))
                        {
                            face->set_boundary_id(3 << (2 * (face_index / 2)));
                        }
                        else
                        {
                            face->set_boundary_id(2 << (2 * (face_index / 2)));
                        }
                    }
                }
            }
        }

        for (unsigned int i = 0; i < refinements; ++i)
        {
            for (auto& cell : tria.active_cell_iterators())
            {
                if (near_center(cell->center()))
                {
                    cell->set_refine_flag();
                }
            }

            tria.execute_coarsening_and_refinement();
            epsilon /= 2.;
        }
    }
}
