cmake_minimum_required(VERSION 3.1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED true)

find_package(deal.II 9.4.0 REQUIRED
    HINTS $ENV{DEAL_II_DIR})

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

deal_II_initialize_cached_variables()

project(tostii
    VERSION 0.0.0)

add_subdirectory("cmake")
add_subdirectory("src")

file(GLOB macro_files "cmake/macros/*.cmake")
foreach(macro_file ${macro_files})
    include(${macro_file})
endforeach()

add_subdirectory("nse")
add_subdirectory("monodomain")
add_subdirectory("bidomain")
add_subdirectory("neutron")

add_subdirectory("doc")
