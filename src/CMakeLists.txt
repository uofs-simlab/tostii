
set(TOSTII_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include/tostii")

add_subdirectory("grid")
add_subdirectory("time_stepping")

add_library(tostii
    "tostii.cc" "${TOSTII_INCLUDE_DIR}/tostii.h")
deal_II_setup_target(tostii)
target_include_directories(tostii PUBLIC
    "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(tostii
    tostii_grid
    tostii_time_stepping)

install(TARGETS tostii
    DESTINATION "lib")

set(TOSTII_INCLUDE_DIR "${TOSTII_INCLUDE_DIR}" PARENT_SCOPE)
