#pragma once

#include <deal.II/base/mpi.h>

#include <boost/serialization/access.hpp>

#include <variant>
#include <memory>
#include <regex>
#include <map>
#include <fstream>
#include <filesystem>

namespace tostii
{
    /**
     * Add instantiations of this struct with the field:
     * ```c++
     * static std::ios::openmode stream_flags;
     * ```
     * to generate SaveManager code for other archive types.
     * These flags are used when opening archive I/O streams.
     * Instantiations are provided for boost's
     * `text_[io]archive` and `binary_[io]archive`.
     */
    template<typename Archive>
    struct ArchiveTraits;

    /**
     * Manages checkpoint files.
     * 
     * When an object of this class is initialized (by initialize()),
     * it scans the directory given by the \p path argument for save files.
     * Afterwards, the directory is populated by information about the
     * saved checkpoints and the checkpoint files themselves.
     * 
     * \p IArchive and \p OArchive are the types of the input and output
     * archives used by the class respectively. They should be of the same type,
     * e.g., \p boost::archive::text_iarchive and \p boost::archive::text_oarchive.
     */
    template<typename IArchive, typename OArchive>
    class SaveManager
    {
    public:
        /**
         * Default constructor.
         * 
         * initialize() must be called before using this object.
         */
        SaveManager();
        /**
         * Serial constructor.
         * 
         * This constructor calls the corresponding overload of initialize().
         */
        SaveManager(
            const std::string& path,
            const unsigned int n_saves = 0,
            const unsigned int n_digits_for_counter = 1);
        /**
         * Parallel constructor.
         * 
         * This constructor calls the corresponding overload of initialize().
         */
        SaveManager(
            const std::string& path,
            const MPI_Comm mpi_communicator,
            const unsigned int n_saves = 0,
            const unsigned int n_digits_for_counter = 1);

        /**
         * Initializes this SaveManager.
         * 
         * This function initializes the current object for use in serial computations,
         * i.e., programs not utilizing MPI.
         * \p path is a path to a directory in which checkpoints will be saved
         * (the directory is created if it does not exist).
         * \p n_saves is the number of checkpoints to keep (\p 0 for no limit).
         * If more than \p n_saves checkpoints would be saved, the oldest checkpoint is
         * deleted after the newest checkpoint has been saved (see save_file()).
         * \p n_digits_for_counter specifies the minimum field width of
         * the checkpoint number in each file name; if the counter has fewer digits,
         * the checkpoint number is padded with leading zeros.
         */
        void initialize(
            const std::string& path,
            const unsigned int n_saves = 0,
            const unsigned int n_digits_for_counter = 1);
        /**
         * Initializes the SaveManager.
         * 
         * This function initializes the current object for use in parallel computations,
         * i.e., programs utilizing MPI.
         * \p path , \p n_saves , and \p n_digits_for_counter are as in the above overload.
         * \p mpi_communicator is the MPI communicator for all processes involved in the
         * computation.
        */
        void initialize(
            const std::string& path,
            const MPI_Comm mpi_communicator,
            const unsigned int n_saves = 0,
            const unsigned int n_digits_for_counter = 1);
        
        /**
         * Returns the last value of \p counter passed to save_file().
         */
        unsigned int last_checkpoint() const noexcept;
        /**
         * Returns the number of tracked checkpoints
         */
        unsigned int n_checkpoints() const noexcept;

        /**
         * Opens the latest checkpoint file for loading.
         * 
         * This function opens the latest checkpoint file
         * as an IArchive.
         * 
         * close_file() must be called between calling this function
         * and calling either this function or save_file().
         */
        IArchive& load_file();
        /**
         * Opens a checkpoint file for saving.
         * 
         * This function opens a file suitable for saving a checkpoint.
         * \p counter must be greater than the value returned by last_checkpoint().
         * 
         * close_file() must be called between calling this function
         * and calling either this function or load_file().
         */
        OArchive& save_file(
            const unsigned int counter);
        /**
         * Closes the currently open file.
         * 
         * This function must be called after calling
         * either load_file() or save_file().
         */
        void close_file();
        /**
         * Closes the currently owned file in the event of an error.
         * 
         * If an error occurs while saving,
         * this function can be called to cancel the checkpoint,
         * i.e., the file is closed and removed.
         * This function cannot be used to cancel loading a checkpoint.
         */
        void abort_file();

    private:
        friend class boost::serialization::access;

        /**
         * Loads this object from a Boost archive.
         * 
         * This function only serializes its \p last_counter field,
         * since other fields are assumed to be known from initialization.
         */
        void serialize(
            IArchive& ar,
            const unsigned int version);
        /**
         * Saves this object to a Boost archive.
         * 
         * If the \p last_counter field in \p ar is older than the current
         * \p last_counter , more recent checkpoints are deleted.
         */
        void serialize(
            OArchive& ar,
            const unsigned int version);

        static constexpr const char info_filename[] = "info.txt";
        static constexpr const char save_pattern[] = "save_%%0%dd.dat";
        static constexpr const char group_save_pattern[] = "save_%%0%dd.%%0%dd.dat";
        static constexpr const char psave_pattern[] = "save_%%0%dd.txt";

        static const std::regex info_line_re;
        static const std::regex save_line_re;

        constexpr bool is_parallel() const noexcept;
        inline bool is_rank_zero() const;

        void synchronize() const;

        void read_directory();
        void initialize_directory();

        void commit() const;
        void commit_save(
            const unsigned int counter,
            const std::string& fname) const;
        void commit_delete(
            const std::string& fname) const;

        std::string load_filename() const;
        std::string save_filename() const;

        MPI_Comm mpi_communicator;

        std::filesystem::path path;
        unsigned int n_digits_for_counter;
        unsigned int n_saves;

        unsigned int last_counter;
        std::map<unsigned int, std::string> tracked_saves;

        struct OpenFile
        {
            virtual ~OpenFile() = default;
        };
        struct OpenLoadFile
            : public OpenFile
        {
            std::ifstream stream;
            std::unique_ptr<IArchive> ar_ptr;
        };
        struct OpenSaveFile
            : public OpenFile
        {
            std::ofstream stream;
            std::unique_ptr<OArchive> ar_ptr;
        };

        std::unique_ptr<OpenFile> open_file;
    };
}
