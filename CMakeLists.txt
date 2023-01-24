##
#  CMake script for the OS codes program:
##
CMAKE_MINIMUM_REQUIRED(VERSION 3.1.0)

SET(TOSTII_SRC
  tostii.cc
  tostii.h
)

#
# Locate a deal.ii installation
#
FIND_PACKAGE(deal.II 9.3.0
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

#
# Ensure it enables the functionality we use?
#
IF(NOT (DEAL_II_WITH_UMFPACK
      AND DEAL_II_WITH_COMPLEX_VALUES
      AND DEAL_II_WITH_PETSC)) # keep in one line
  MESSAGE(FATAL_ERROR "
Error! This tutorial requires a deal.II library that was configured with the following options:
    DEAL_II_WITH_UMFPACK        = ON
    DEAL_II_WITH_COMPLEX_VALUES = ON
    DEAL_II_WITH_PETSC          = ON
However, the deal.II library found at ${DEAL_II_PATH} was configured with these options
    DEAL_II_WITH_UMFPACK        = ${DEAL_II_WITH_UMFPACK}
    DEAL_II_WITH_COMPLEX_VALUES = ${DEAL_II_WITH_COMPLEX_VALUES}
    DEAL_II_WITH_PETSC          = ${DEAL_II_WITH_PETSC}
which conflict with the requirements."
    )
ENDIF()


DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(tostii_demo_project)

# Specify pattern for compilation
function(os_code main_src)
  ADD_EXECUTABLE(${main_src}
    ${main_src}.cc
    ${TOSTII_SRC}
  )
  DEAL_II_SETUP_TARGET(${main_src})
endfunction()

os_code(neutron)

os_code(nse)
os_code(nse_three)

os_code(monodomain)
os_code(monodomain_direct)
