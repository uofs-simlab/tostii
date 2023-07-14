
set(TOSTII_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include/tostii")

add_subdirectory("time_stepping")

add_library(tostii
    "tostii.cc" "${TOSTII_INCLUDE_DIR}/tostii.h")
DEAL_II_SETUP_TARGET(tostii)
target_include_directories(tostii PUBLIC
    ${TOSTII_INCLUDE_DIR})
target_link_libraries(tostii
    time_stepping)
set_target_properties(tostii PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
