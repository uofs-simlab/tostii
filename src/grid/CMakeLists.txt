
set(INCLUDE_DIR "${TOSTII_INCLUDE_DIR}/tostii/grid")
set(SOURCES
    "L_shape.cc")
set(HEADERS
    "${INCLUDE_DIR}/L_shape.inl"
    "${INCLUDE_DIR}/L_shape.h")

add_library(tostii_grid OBJECT ${SOURCES} ${HEADERS})
deal_II_setup_target(tostii_grid)
target_include_directories(tostii_grid PUBLIC
    "${CMAKE_SOURCE_DIR}/include")

install(FILES ${HEADERS}
    DESTINATION "include/tostii/grid")
