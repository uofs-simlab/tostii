
set(INCLUDE_DIR "${TOSTII_INCLUDE_DIR}/time_stepping")
set(SOURCES
    "runge_kutta.cc"
    "explicit_runge_kutta.cc"
    "implicit_runge_kutta.cc"
    "linear_implicit_runge_kutta.cc"
    "exact.cc"
    "operator_split.cc"
    "operator_split_single.cc")
set(HEADERS
    "${INCLUDE_DIR}/time_stepping.h"
    "${INCLUDE_DIR}/runge_kutta.h"
    "${INCLUDE_DIR}/runge_kutta.inl"
    "${INCLUDE_DIR}/explicit_runge_kutta.h"
    "${INCLUDE_DIR}/explicit_runge_kutta.inl"
    "${INCLUDE_DIR}/implicit_runge_kutta.h"
    "${INCLUDE_DIR}/implicit_runge_kutta.inl"
    "${INCLUDE_DIR}/linear_implicit_runge_kutta.h"
    "${INCLUDE_DIR}/linear_implicit_runge_kutta.inl"
    "${INCLUDE_DIR}/exact.h"
    "${INCLUDE_DIR}/exact.inl"
    "${INCLUDE_DIR}/operator_split.h"
    "${INCLUDE_DIR}/operator_split.inl"
    "${INCLUDE_DIR}/operator_split_single.h"
    "${INCLUDE_DIR}/operator_split_single.inl")

add_library(tostii_time_stepping OBJECT ${SOURCES} ${HEADERS})
deal_II_setup_target(tostii_time_stepping)
target_include_directories(tostii_time_stepping PUBLIC
    "${CMAKE_SOURCE_DIR}/include")

install(FILES ${HEADERS}
    DESTINATION "include/tostii/time_stepping")
