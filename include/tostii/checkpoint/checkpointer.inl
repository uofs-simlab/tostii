#pragma once

#include <tostii/checkpoint/checkpointer.h>

namespace tostii
{
    template<typename IArchive, typename OArchive>
    Checkpointer<IArchive, OArchive>::Checkpointer()
    { }

    template<typename IArchive, typename OArchive>
    template<typename PathType>
    Checkpointer<IArchive, OArchive>::Checkpointer(
        const PathType& path,
        const unsigned int n_saves,
        const unsigned int n_digits_for_counter)
    {
        initialize(path, n_saves, n_digits_for_counter);
    }

    template<typename IArchive, typename OArchive>
    template<typename PathType>
    Checkpointer<IArchive, OArchive>::Checkpointer(
        const PathType& path,
        const MPI_Comm mpi_communicator,
        const unsigned int n_saves,
        const unsigned int n_digits_for_counter)
    {
        initialize(path, mpi_communicator, n_saves, n_digits_for_counter);
    }

    template<typename IArchive, typename OArchive>
    template<typename PathType>
    void Checkpointer<IArchive, OArchive>::initialize(
        const PathType& path,
        const unsigned int n_saves,
        const unsigned int n_digits_for_counter)
    {
        save_manager.initialize(path, n_saves, n_digits_for_counter);

        if (save_manager.n_checkpoints() > 0)
        {
            load();
        }
    }

    template<typename IArchive, typename OArchive>
    template<typename PathType>
    void Checkpointer<IArchive, OArchive>::initialize(
        const PathType& path,
        const MPI_Comm mpi_communicator,
        const unsigned int n_saves,
        const unsigned int n_digits_for_counter)
    {
        save_manager.initialize(path, n_saves, n_digits_for_counter);

        if (save_manager.n_checkpoints() > 0)
        {
            load();
        }
    }

    template<typename IArchive, typename OArchive>
    void Checkpointer<IArchive, OArchive>::checkpoint(
        const unsigned int counter)
    {
        OArchive& ar = save_manager.save_file(counter);

        try
        {
            ar & *this;
        }
        catch (...)
        {
            save_manager.abort_file();
            throw;
        }

        save_manager.close_file();
    }

    template<typename IArchive, typename OArchive>
    void Checkpointer<IArchive, OArchive>::load()
    {
        IArchive& ar = save_manager.load_file();

        try
        {
            ar & *this;
        }
        catch (...)
        {
            save_manager.close_file();
            throw;
        }

        save_manager.close_file();
    }

    template<typename IArchive, typename OArchive>
    void Checkpointer<IArchive, OArchive>::serialize(
        IArchive& ar,
        const unsigned int)
    {
        ar & save_manager;
    }

    template<typename IArchive, typename OArchive>
    void Checkpointer<IArchive, OArchive>::serialize(
        OArchive& ar,
        const unsigned int)
    {
        ar & save_manager;
    }
}
