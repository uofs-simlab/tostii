#pragma once

#include <tostii/checkpoint/save_manager.h>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include <string>
#include <iterator>
#include <sstream>
#include <fstream>
#include <iomanip>

namespace tostii
{
    template<>
    struct ArchiveTraits<boost::archive::text_iarchive>
    {
        static constexpr std::ios::openmode stream_flags = (std::ios::openmode)0;
    };

    template<>
    struct ArchiveTraits<boost::archive::text_oarchive>
    {
        static constexpr std::ios::openmode stream_flags = (std::ios::openmode)0;
    };

    template<>
    struct ArchiveTraits<boost::archive::binary_iarchive>
    {
        static constexpr std::ios::openmode stream_flags = std::ios::binary;
    };

    template<>
    struct ArchiveTraits<boost::archive::binary_oarchive>
    {
        static constexpr std::ios::openmode stream_flags = std::ios::binary;
    };

    template<typename IArchive, typename OArchive>
    const std::regex SaveManager<IArchive, OArchive>::info_line_re("^\\[(\\d+)\\] (.*)$");

    template<typename IArchive, typename OArchive>
    const std::regex SaveManager<IArchive, OArchive>::save_line_re("^\\[(\\d+)\\] (.*)$");

    template<typename IArchive, typename OArchive>
    SaveManager<IArchive, OArchive>::SaveManager()
    { }

    template<typename IArchive, typename OArchive>
    SaveManager<IArchive, OArchive>::SaveManager(
        const std::string& path,
        const unsigned int n_saves,
        const unsigned int n_digits_for_counter)
    {
        initialize(path, n_saves, n_digits_for_counter);
    }

    template<typename IArchive, typename OArchive>
    SaveManager<IArchive, OArchive>::SaveManager(
        const std::string& path,
        const MPI_Comm mpi_communicator,
        const unsigned int n_saves,
        const unsigned int n_digits_for_counter)
    {
        initialize(path, mpi_communicator, n_saves, n_digits_for_counter);
    }

    template<typename IArchive, typename OArchive>
    void SaveManager<IArchive, OArchive>::initialize(
        const std::string& path,
        const unsigned int n_saves,
        const unsigned int n_digits_for_counter)
    {
        this->mpi_communicator = MPI_COMM_NULL;
        this->path = path;
        this->n_digits_for_counter = n_digits_for_counter;
        this->n_saves = n_saves;

        this->last_counter = 0;
        this->tracked_saves.clear();

        read_directory();
    }

    template<typename IArchive, typename OArchive>
    void SaveManager<IArchive, OArchive>::initialize(
        const std::string& path,
        const MPI_Comm mpi_communicator,
        const unsigned int n_saves,
        const unsigned int n_digits_for_counter)
    {
        this->mpi_communicator = mpi_communicator;
        this->path = path;
        this->n_digits_for_counter = n_digits_for_counter;
        this->n_saves = n_saves;

        this->last_counter = 0;
        this->tracked_saves.clear();

        read_directory();
    }

    template<typename IArchive, typename OArchive>
    constexpr bool SaveManager<IArchive, OArchive>::is_parallel() const noexcept
    {
        return mpi_communicator != MPI_COMM_NULL;
    }

    template<typename IArchive, typename OArchive>
    inline bool SaveManager<IArchive, OArchive>::is_rank_zero() const
    {
        using dealii::Utilities::MPI::this_mpi_process;
        return !is_parallel() || this_mpi_process(mpi_communicator) == 0;
    }

    template<typename IArchive, typename OArchive>
    void SaveManager<IArchive, OArchive>::synchronize() const
    {
        if (is_parallel() && MPI_Barrier(mpi_communicator) != MPI_SUCCESS)
        {
            throw std::runtime_error("MPI_Barrier did not return MPI_SUCCESS");
        }
    }

    template<typename IArchive, typename OArchive>
    void SaveManager<IArchive, OArchive>::read_directory()
    {
        if (is_rank_zero())
        {
            initialize_directory();
        }
        synchronize();

        {
            std::ifstream info(path / info_filename);
            std::string line;
            std::smatch match;
            while (std::getline(info, line))
            {
                if (line.empty()) continue;
                if (!std::regex_match(line, match, info_line_re))
                {
                    throw std::runtime_error((path / info_filename).string() + ": invalid file");
                }

                const unsigned int counter = std::stoul(match[1]);
                if (counter > last_counter)
                {
                    last_counter = counter;
                }

                const std::ssub_match& fname = match[2];
                const auto [it, inserted] = tracked_saves.emplace(counter, fname);
                if (!inserted)
                {
                    throw std::runtime_error((path / info_filename).string() + ": invalid file");
                }
            }
        }
    }

    template<typename IArchive, typename OArchive>
    void SaveManager<IArchive, OArchive>::initialize_directory()
    {
        auto stat = std::filesystem::status(path);

        switch (stat.type())
        {
        case std::filesystem::file_type::directory:
            break;
        case std::filesystem::file_type::not_found:
            if (!std::filesystem::create_directories(path))
            {
                throw std::runtime_error(path.string() + ": could not create directory");
            }
            {
                std::ofstream info(path / info_filename);
            }
            break;
        case std::filesystem::file_type::unknown:
            throw std::runtime_error(path.string() + ": could not stat");
        default:
            throw std::runtime_error(path.string() + ": not a directory");
        }
    }

    template<typename IArchive, typename OArchive>
    unsigned int SaveManager<IArchive, OArchive>::last_checkpoint() const noexcept
    {
        return last_counter;
    }

    template<typename IArchive, typename OArchive>
    unsigned int SaveManager<IArchive, OArchive>::n_checkpoints() const noexcept
    {
        return tracked_saves.size();
    }

    template<typename IArchive, typename OArchive>
    void SaveManager<IArchive, OArchive>::commit() const
    {
        if (!is_rank_zero())
        {
            synchronize();
            return;
        }

        std::map<unsigned int, std::string> old_info;
        {
            std::ifstream fp(path / info_filename);
            std::string line;
            std::smatch match;
            while (std::getline(fp, line))
            {
                if (line.empty()) continue;
                if (!std::regex_match(line, match, info_line_re))
                {
                    throw std::runtime_error((path / info_filename).string() + ": invalid file");
                }

                old_info.emplace(std::stoul(match[1]), match[2]);
            }
        }

        std::vector<std::pair<unsigned int, std::string>>
            insertions,
            deletions;
        for (const auto& [counter, fname] : tracked_saves)
        {
            if (old_info.find(counter) == old_info.end())
            {
                insertions.emplace_back(counter, fname);
            }
        }
        while (!old_info.empty())
        {
            auto nh = old_info.extract(old_info.begin());
            if (tracked_saves.find(nh.key()) == tracked_saves.end())
            {
                deletions.emplace_back(nh.key(), nh.mapped());
            }
        }

        for (const auto& [counter, fname] : insertions)
        {
            commit_save(counter, fname);
        }
        {
            std::ofstream fp(path / info_filename);
            for (const auto& [counter, fname] : tracked_saves)
            {
                fp << '[' << counter << "] " << fname << '\n';
            }
        }
        for (const auto& [counter, fname] : deletions)
        {
            commit_delete(fname);
        }

        synchronize();
    }

    template<typename IArchive, typename OArchive>
    void SaveManager<IArchive, OArchive>::commit_save(
        const unsigned int counter,
        const std::string& fname) const
    {
        using dealii::Utilities::MPI::this_mpi_process,
            dealii::Utilities::MPI::n_mpi_processes;
        const unsigned int nproc = n_mpi_processes(mpi_communicator);
        const unsigned int rank_digits = int(std::log10(nproc)) + 1;

        std::ofstream fp(path / fname);
        for (unsigned int i = 0; i < nproc; ++i)
        {
            std::stringstream group_fname;
            fp << '[' << i << "] "
                << "save_"
                << std::setw(n_digits_for_counter) << std::setfill('0') << counter
                << "."
                << std::setw(rank_digits) << std::setfill('0') << i
                << ".dat\n";
        }
    }

    template<typename IArchive, typename OArchive>
    void SaveManager<IArchive, OArchive>::commit_delete(
        const std::string& fname) const
    {
        std::ifstream fp(path / fname);
        std::string line;
        std::smatch match;
        while (std::getline(fp, line))
        {
            if (line.empty()) continue;
            if (!std::regex_match(line, match, save_line_re))
            {
                throw std::runtime_error((path / fname).string() + ": invalid file");
            }

            std::filesystem::remove(path / match[2].str());
        }
        std::filesystem::remove(path / fname);
    }

    template<typename IArchive, typename OArchive>
    std::string SaveManager<IArchive, OArchive>::load_filename() const
    {
        if (n_checkpoints() == 0)
        {
            throw std::runtime_error(path.string() + ": no checkpoint to load");
        }

        const std::string& fname = tracked_saves.at(last_checkpoint());

        if (!is_parallel())
        {
            return fname;
        }
        else
        {
            std::ifstream psave(path / fname);
            std::string line;
            std::smatch match;
            using dealii::Utilities::MPI::this_mpi_process;
            const unsigned int rank = this_mpi_process(mpi_communicator);
            while (std::getline(psave, line))
            {
                if (line.empty()) continue;
                if (!std::regex_match(line, match, save_line_re))
                {
                    throw std::runtime_error((path / fname).string() + ": invalid file");
                }

                if (std::stoul(match[1]) == rank)
                {
                    return match[2];
                }
            }
            throw std::runtime_error((path / fname).string() + ": invalid file");
        }
    }

    template<typename IArchive, typename OArchive>
    IArchive& SaveManager<IArchive, OArchive>::load_file()
    {
        if (open_file.get() != nullptr)
        {
            throw std::runtime_error(path.string() + ": invalid usage");
        }

        const std::string fname = load_filename();
        
        OpenLoadFile* file_ptr = new OpenLoadFile;
        open_file.reset(file_ptr);
        auto& [stream, ar_ptr] = *file_ptr;
        auto stream_flags = ArchiveTraits<IArchive>::stream_flags;
        stream.open(path / fname, stream_flags);
        ar_ptr.reset(new IArchive(stream));

        return *ar_ptr;
    }

    template<typename IArchive, typename OArchive>
    std::string SaveManager<IArchive, OArchive>::save_filename() const
    {
        if (!is_parallel())
        {
            std::stringstream fname;
            fname << "save_"
                << std::setw(n_digits_for_counter) << std::setfill('0') << last_counter
                << ".dat";
            
            return fname.str();
        }
        else
        {
            using dealii::Utilities::MPI::n_mpi_processes,
                dealii::Utilities::MPI::this_mpi_process;
            
            std::stringstream pfname;
            pfname << "save_"
                << std::setw(n_digits_for_counter) << std::setfill('0') << last_counter
                << '.'
                << std::setw(int(std::log10(n_mpi_processes(mpi_communicator))) + 1)
                << std::setfill('0') << this_mpi_process(mpi_communicator)
                << ".dat";
            
            return pfname.str();
        }
    }

    template<typename IArchive, typename OArchive>
    OArchive& SaveManager<IArchive, OArchive>::save_file(
        const unsigned int counter)
    {
        if (counter < last_counter)
        {
            throw std::invalid_argument(path.string() + ": counter must increase");
        }
        if (open_file.get() != nullptr)
        {
            throw std::runtime_error(path.string() + ": invalid usage");
        }

        last_counter = counter;
        const std::string fname = save_filename();
        
        OpenSaveFile* file_ptr = new OpenSaveFile;
        open_file.reset(file_ptr);
        auto& [stream, ar_ptr] = *file_ptr;
        auto stream_flags = ArchiveTraits<OArchive>::stream_flags;
        stream.open(path / fname, stream_flags);
        ar_ptr.reset(new OArchive(stream));

        return *ar_ptr;
    }

    template<typename IArchive, typename OArchive>
    void SaveManager<IArchive, OArchive>::close_file()
    {
        OpenFile* file_ptr = open_file.get();
        OpenLoadFile* ifile_ptr;
        OpenSaveFile* ofile_ptr;

        if (ifile_ptr = dynamic_cast<OpenLoadFile*>(file_ptr))
        {
            open_file.reset();
        }
        else if (ofile_ptr = dynamic_cast<OpenSaveFile*>(file_ptr))
        {
            open_file.reset();

            {
                std::stringstream fname;
                fname << "save_"
                    << std::setw(n_digits_for_counter) << std::setfill('0') << last_counter
                    << ".txt";
                tracked_saves[last_counter] = fname.str();
            }
            if (n_saves > 0 && tracked_saves.size() > n_saves)
            {
                tracked_saves.erase(tracked_saves.begin());
            }
            commit();
        }
        else
        {
            throw std::runtime_error(path.string() + ": invalid usage");
        }
    }

    template<typename IArchive, typename OArchive>
    void SaveManager<IArchive, OArchive>::abort_file()
    {
        OpenFile* file_ptr = open_file.get();
        OpenLoadFile* ifile_ptr;
        OpenSaveFile* ofile_ptr;

        if (ifile_ptr = dynamic_cast<OpenLoadFile*>(file_ptr))
        {
            throw std::runtime_error(path.string() + ": cannot abort load");
        }
        else if (ofile_ptr = dynamic_cast<OpenSaveFile*>(file_ptr))
        {
            open_file.reset();

            {
                using dealii::Utilities::MPI::n_mpi_processes,
                    dealii::Utilities::MPI::this_mpi_process;

                std::stringstream fname;
                fname << "save_"
                    << std::setw(n_digits_for_counter) << std::setfill('0') << last_counter
                    << '.'
                    << std::setw(int(std::log10(n_mpi_processes(mpi_communicator))) + 1)
                    << std::setfill('0') << this_mpi_process(mpi_communicator)
                    << ".dat";
                std::filesystem::remove(path / fname.str());
            }

            auto it = tracked_saves.end();
            last_counter = (--it)->first;
        }
        else
        {
            throw std::runtime_error(path.string() + ": invalid usage");
        }
    }

    template<typename IArchive, typename OArchive>
    void SaveManager<IArchive, OArchive>::serialize(
        IArchive& ar,
        const unsigned int)
    {
        ar & last_counter;

        auto sentry = tracked_saves.begin(),
            it = tracked_saves.end();
        while (it != sentry && (--it)->first > last_counter);

        if (it->first != last_counter)
        {
            throw std::runtime_error(path.string() + ": no such checkpoint");
        }

        tracked_saves.erase(it, tracked_saves.end());
    }

    template<typename IArchive, typename OArchive>
    void SaveManager<IArchive, OArchive>::serialize(
        OArchive& ar,
        const unsigned int)
    {
        ar & last_counter;
    }
}
