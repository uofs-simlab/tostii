
deal_II_executable(step-58
    "step-58.cc")

deal_II_executable(monolithic
    "monolithic.cc")

deal_II_executable(monolithic_arkode
    "monolithic_arkode.cc")

add_library(nse_utils OBJECT
    "parameters.h" "parameters.cc"
    "prescribed_data.h" "prescribed_data.inl" "prescribed_data.cc"
    "utils.h" "utils.inl" "utils.cc")
deal_II_setup_target(nse_utils)
target_include_directories(nse_utils PUBLIC
    "${TOSTII_INCLUDE_DIR}")

tostii_executable(monolithic_hs
    "monolithic_hs.h" "monolithic_hs.inl" "monolithic_hs.cc")
target_link_libraries(monolithic_hs
    nse_utils)

tostii_executable(nse
    "nse.cc")

tostii_executable(nse_three
    "nse_three.cc")
