#pragma once

#include <tostii/checkpoint/save_manager.h>

#include <boost/serialization/access.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>

namespace tostii
{
    /**
     * A base class to be used to easily checkpoint driver objects.
     */
    template<typename IArchive, typename OArchive>
    class Checkpointer
    {
    public:
        friend class boost::serialization::access;

        /**
         * Default constructor.
         * 
         * initialize() must be called before using this object.
         */
        Checkpointer();
        /**
         * Constructor.
         * 
         * Calls initialize().
         */
        template<typename PathType>
        Checkpointer(
            const PathType& path,
            const unsigned int n_saves = 0,
            const unsigned int n_digits_for_counter = 1);

        /**
         * Initializes this Checkpointer.
         * 
         * This function initializes the internal SaveManager
         * with the provided arguments. Then,
         * if checkpoints already exist, the latest checkpoint is loaded.
         */
        template<typename PathType>
        void initialize(
            const PathType& path,
            const unsigned int n_saves = 0,
            const unsigned int n_digits_for_counter = 1);
        /**
         * Initializes this Checkpointer.
         * 
         * This function initializes the internal SaveManager
         * with the provided arguments. Then,
         * if checkpoints already exist, the latest checkpoint is loaded.
         */
        template<typename PathType>
        void initialize(
            const PathType& path,
            const MPI_Comm mpi_communicator,
            const unsigned int n_saves = 0,
            const unsigned int n_digits_for_counter = 1);

        /**
         * Create a checkpoint for this object.
         * 
         * A save file is obtained using the internal SaveManager,
         * then \p this is serialized to the given \p OArchive .
         */
        void checkpoint(
            const unsigned int counter);

        /**
         * Serializes (loads) this object.
         * 
         * This method should be overridden in derived classes.
         * The first argument is specified here explicitly as \p IArchive
         * since template functions cannot be virtual,
         * but derived base classes may choose to use a templated override.
         */
        virtual void serialize(
            IArchive& ar,
            const unsigned int version);
        /**
         * Serializes (saves) this object.
         * 
         * See the above overload.
         */
        virtual void serialize(
            OArchive& ar,
            const unsigned int version);

    private:
        SaveManager<IArchive, OArchive> save_manager;
    };

    using TextCheckpointer = Checkpointer<
        boost::archive::text_iarchive,
        boost::archive::text_oarchive>;
    using BinaryCheckpointer = Checkpointer<
        boost::archive::binary_iarchive,
        boost::archive::binary_oarchive>;
    using XMLCheckpointer = Checkpointer<
        boost::archive::xml_iarchive,
        boost::archive::xml_oarchive>;
}
