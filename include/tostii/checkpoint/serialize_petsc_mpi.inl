#include <tostii/checkpoint/serialize_petsc_mpi.h>

#include <numeric>

namespace boost::serialization
{
    template<typename Archive>
    void serialize(
        Archive& ar,
        dealii::PETScWrappers::MPI::Vector& v,
        const unsigned int)
    {
        auto [first, last] = v.local_range();

        std::vector<unsigned int> indices(last - first);
        std::iota(indices.begin(), indices.end(), first);

        std::vector<double> values(last - first);
        
        if constexpr (Archive::is_loading::value)
        {
            ar & values;
            v.set(indices, values);
        }
        else if constexpr (Archive::is_saving::value)
        {
            v.extract_subvector_to(indices, values);
            ar & values;
        }
        else
        {
            /* guarenteed to produce compiler error */
            static_assert(Archive::is_saving::value || Archive::is_loading::value);
        }
    }

    template<typename Archive>
    void serialize(
        Archive& ar,
        dealii::PETScWrappers::MPI::BlockVector& v,
        const unsigned int)
    {
        for (unsigned int i = 0; i < v.n_blocks(); ++i)
        {
            ar & v.block(i);
        }
    }
}
