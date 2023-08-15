
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

#
# Directoy names relative to project root
#
set(TOSTII_INCLUDE_RELDIR "include")
set(TOSTII_LIBRARY_RELDIR "lib")
set(TOSTII_SHARE_RELDIR "share/tostii")
set(TOSTII_PROJECT_CONFIG_RELDIR "lib/cmake/tostii")

#
# Determine TOSTII_PATH from CMAKE_CURRENT_LIST_DIR and TOSTII_PROJECT_CONFIG_RELDIR
#
set(TOSTII_PATH "${CMAKE_CURRENT_LIST_DIR}")
set(_path "${TOSTII_PROJECT_CONFIG_RELDIR}")
while(NOT "${_path}" STREQUAL "")
    get_filename_component(TOSTII_PATH "${TOSTII_PATH}" PATH)
    get_filename_component(_path "${_path}" PATH)
endwhile()

#
# Set absolute directory names
#
set(TOSTII_INCLUDE_DIR "${TOSTII_PATH}/${TOSTII_INCLUDE_RELDIR}")
set(TOSTII_LIBRARY_DIR "${TOSTII_PATH}/${TOSTII_LIBRARY_RELDIR}")
set(TOSTII_SHARE_DIR "${TOSTII_PATH}/${TOSTII_SHARE_RELDIR}")

find_library(tostii_lib "tostii"
    HINTS ${TOSTII_LIBRARY_DIR})
add_library(tostii UNKNOWN IMPORTED)
set_target_properties(tostii PROPERTIES
    IMPORTED_LOCATION ${tostii_lib})

file(GLOB macro_files "${TOSTII_SHARE_DIR}/macros/*.cmake")
foreach(macro_file ${macro_files})
    include(${macro_file})
endforeach()
