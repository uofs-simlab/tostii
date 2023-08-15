
set(TOSTII_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")

add_subdirectory("checkpoint")
add_subdirectory("grid")
add_subdirectory("time_stepping")

add_library(tostii
    "tostii.cc" "${TOSTII_INCLUDE_DIR}/tostii/tostii.h")
deal_II_setup_target(tostii)
target_include_directories(tostii PUBLIC
    "${TOSTII_INCLUDE_DIR}")
target_link_libraries(tostii
    tostii_checkpoint
    tostii_grid
    tostii_time_stepping)

install(TARGETS tostii
    DESTINATION "lib")

set(TOSTII_INCLUDE_DIR "${TOSTII_INCLUDE_DIR}" PARENT_SCOPE)
