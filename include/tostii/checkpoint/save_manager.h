#pragma once

#include <deal.II/base/mpi.h>

#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include <memory>
#include <deque>
#include <map>
#include <fstream>
#include <filesystem>

namespace tostii
{
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
        template<typename PathType>
        SaveManager(
            const PathType& path,
            const unsigned int n_saves = 0,
            const unsigned int n_digits_for_counter = 1);
        /**
         * Parallel constructor.
         * 
         * This constructor calls the corresponding overload of initialize().
         */
        template<typename PathType>
        SaveManager(
            const PathType& path,
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
        template<typename PathType>
        void initialize(
            const PathType& path,
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
        template<typename PathType>
        void initialize(
            const PathType& path,
            const MPI_Comm mpi_communicator,
            const unsigned int n_saves = 0,
            const unsigned int n_digits_for_counter = 1);
        
        /**
         * Returns the last value of \p counter passed to save_file().
         */
        unsigned int last_checkpoint() const noexcept;

        /**
         * Opens the latest checkpoint file for loading.
         * 
         * This function opens the latest checkpoint file
         * as an IArchive and returns it as a `shared_ptr`.
         */
        std::shared_ptr<IArchive> load_file();
        /**
         * Opens a checkpoint file for saving.
         * 
         * This function opens a file suitable for saving a checkpoint.
         * \p counter must be greater than the value returned by last_checkpoint().
         * 
         * The `shared_ptr` returned by this function is given a custom deleter
         * which deletes the oldest checkpoint if there would be more checkpoints
         * than permitted by \p n_saves .
         */
        std::shared_ptr<OArchive> save_file(
            const unsigned int counter);

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

    private:
        static constexpr const char info[] = "info.txt";
        static constexpr const char save_pattern[] = "save_{:0{}d}.dat";
        static constexpr const char group_save_pattern[] = "save_{:0{}d}.{:0{}d}.dat";
        static constexpr const char psave_pattern[] = "save_{:0{}d}.pdat";

        MPI_Comm mpi_communicator;

        std::filesystem::path path;
        unsigned int n_digits_for_counter;
        unsigned int n_saves;
        unsigned int n_groups;

        unsigned int last_counter;
        std::deque<unsigned int> tracked_saves;
    };
}
