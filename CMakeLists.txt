cmake_minimum_required(VERSION 3.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED true)

find_package(deal.II 9.4.0
    HINTS "${deal.II_DIR}" "${DEAL_II_DIR}" "$ENV{DEAL_II_DIR}")

if (NOT ${deal.II_FOUND})
    message(FATAL_ERROR "\n"
        "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
        "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
        "or set an environment variable \"DEAL_II_DIR\" that contains this path.")
endif()

if (NOT (${DEAL_II_WITH_UMFPACK}
    AND ${DEAL_II_WITH_COMPLEX_VALUES}
    AND ${DEAL_II_WITH_PETSC}
    AND ${DEAL_II_WITH_SUNDIALS}))

    message(FATAL_ERROR "\n"
        "Error! This tutorial requires a deal.II library that was configured with the following options:\n"
        "\tDEAL_II_WITH_UMFPACK        = ON\n"
        "\tDEAL_II_WITH_COMPLEX_VALUES = ON\n"
        "\tDEAL_II_WITH_PETSC          = ON\n"
        "However, the deal.II library found at ${DEAL_II_PATH} was configured with these options\n"
        "\tDEAL_II_WITH_UMFPACK        = ${DEAL_II_WITH_UMFPACK}\n"
        "\tDEAL_II_WITH_COMPLEX_VALUES = ${DEAL_II_WITH_COMPLEX_VALUES}\n"
        "\tDEAL_II_WITH_PETSC          = ${DEAL_II_WITH_PETSC}\n"
        "which conflict with the requirements.")
endif()

deal_ii_initialize_cached_variables()

project(tostii-demo)

add_subdirectory("src")

function(deal_II_executable target)
    add_executable(${target} ${ARGN})
    deal_II_setup_target(${target})
    set_target_properties(${target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")
endfunction()

function(tostii_executable target)
    deal_II_executable(${target}
        ${ARGN})
    target_include_directories(${target} PUBLIC
        "${CMAKE_SOURCE_DIR}/include")
    target_link_libraries(${target}
        tostii)
endfunction()

add_subdirectory("nse")
add_subdirectory("monodomain")
add_subdirectory("bidomain")
add_subdirectory("neutron")
