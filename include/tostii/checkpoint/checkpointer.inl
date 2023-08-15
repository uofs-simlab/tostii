#pragma once

#include <tostii/checkpoint/checkpointer.h>

namespace tostii
{
    template<typename IArchive, typename OArchive>
    Checkpointer<IArchive, OArchive>::Checkpointer()
    { }

    template<typename IArchive, typename OArchive>
    void Checkpointer<IArchive, OArchive>::initialize(
        const std::string& path,
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
    void Checkpointer<IArchive, OArchive>::initialize(
        const std::string& path,
        const MPI_Comm mpi_communicator,
        const unsigned int n_saves,
        const unsigned int n_digits_for_counter)
    {
        save_manager.initialize(path, mpi_communicator, n_saves, n_digits_for_counter);

        if (save_manager.n_checkpoints() > 0)
        {
            load();
        }
    }

    template<typename IArchive, typename OArchive>
    unsigned int Checkpointer<IArchive, OArchive>::last_checkpoint() const noexcept
    {
        return save_manager.last_checkpoint();
    }

    template<typename IArchive, typename OArchive>
    unsigned int Checkpointer<IArchive, OArchive>::n_checkpoints() const noexcept
    {
        return save_manager.n_checkpoints();
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
