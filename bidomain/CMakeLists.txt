
add_library(bidomain_passive_utils OBJECT
    "parameters.cc" "parameters.h"
    "prescribed_data.cc" "prescribed_data.inl" "prescribed_data.h")
deal_II_setup_target(bidomain_passive_utils)
target_include_directories(bidomain_passive_utils PUBLIC
    "${TOSTII_INCLUDE_DIR}")
target_link_libraries(bidomain_passive_utils
    tostii)

add_library(bidomain_passive_utils_2 OBJECT
    "parameters.cc" "parameters.h"
    "prescribed_data_2.cc" "prescribed_data_2.inl" "prescribed_data.h")
deal_II_setup_target(bidomain_passive_utils_2)
target_include_directories(bidomain_passive_utils_2 PUBLIC
    "${TOSTII_INCLUDE_DIR}")
target_link_libraries(bidomain_passive_utils_2
    tostii)

tostii_executable(bidomain
    "bidomain.cc" "bidomain.inl" "bidomain.h")
target_link_libraries(bidomain
    bidomain_passive_utils)

tostii_executable(bidomain_2
    "bidomain.cc" "bidomain.inl" "bidomain.h")
target_link_libraries(bidomain_2
    bidomain_passive_utils_2)

tostii_executable(bidomain_godunov
    "bidomain_godunov.cc" "bidomain_godunov.inl" "bidomain_godunov.h")
target_link_libraries(bidomain_godunov
    bidomain_passive_utils)

tostii_executable(bidomain_godunov_2
    "bidomain_godunov.cc" "bidomain_godunov.inl" "bidomain_godunov.h")
target_link_libraries(bidomain_godunov_2
    bidomain_passive_utils_2)

add_library(bidomain_fhn_utils OBJECT
    "parameters.cc" "parameters.h"
    "fitzhugh-nagumo.cc" "fitzhugh-nagumo.inl" "fitzhugh-nagumo.h")
deal_II_setup_target(bidomain_fhn_utils)
target_include_directories(bidomain_fhn_utils PUBLIC
    "${TOSTII_INCLUDE_DIR}")
target_link_libraries(bidomain_fhn_utils
    tostii)

tostii_executable(bidomain_fhn
    "bidomain_fhn.cc" "bidomain_fhn.inl" "bidomain_fhn.h")
target_link_libraries(bidomain_fhn
    bidomain_fhn_utils)

tostii_executable(bidomain_fhn_manual
    "bidomain_fhn_manual.cc" "bidomain_fhn_manual.inl" "bidomain_fhn_manual.h")
target_link_libraries(bidomain_fhn_manual
    bidomain_fhn_utils)
