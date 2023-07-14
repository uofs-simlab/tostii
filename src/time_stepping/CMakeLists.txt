
set(INCLUDE_DIR "${TOSTII_INCLUDE_DIR}/time_stepping")
set(SOURCES
    "${INCLUDE_DIR}/time_stepping.h"
    "${INCLUDE_DIR}/runge_kutta.h"
    "${INCLUDE_DIR}/runge_kutta.inl"
    "runge_kutta.cc"
    "${INCLUDE_DIR}/explicit_runge_kutta.h"
    "${INCLUDE_DIR}/explicit_runge_kutta.inl"
    "explicit_runge_kutta.cc"
    "${INCLUDE_DIR}/implicit_runge_kutta.h"
    "${INCLUDE_DIR}/implicit_runge_kutta.inl"
    "implicit_runge_kutta.cc"
    "${INCLUDE_DIR}/linear_implicit_runge_kutta.h"
    "${INCLUDE_DIR}/linear_implicit_runge_kutta.inl"
    "linear_implicit_runge_kutta.cc"
    "${INCLUDE_DIR}/exact.h"
    "${INCLUDE_DIR}/exact.inl"
    "exact.cc"
    "${INCLUDE_DIR}/operator_split.h"
    "${INCLUDE_DIR}/operator_split.inl"
    "operator_split.cc"
    "${INCLUDE_DIR}/operator_split_single.h"
    "${INCLUDE_DIR}/operator_split_single.inl"
    "operator_split_single.cc")

add_library(time_stepping STATIC ${SOURCES})
deal_II_setup_target(time_stepping)
target_include_directories(time_stepping PUBLIC
    "${CMAKE_SOURCE_DIR}/include")
