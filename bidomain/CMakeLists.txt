
deal_II_executable(bidomain
    "parameters.cc" "parameters.h"
    "prescribed_data.cc" "prescribed_data.inl" "prescribed_data.h"
    "bidomain.cc" "bidomain.inl" "bidomain.h")

deal_II_executable(bidomain_2
    "parameters.cc" "parameters.h"
    "prescribed_data_2.cc" "prescribed_data_2.inl" "prescribed_data.h"
    "bidomain.cc" "bidomain.inl" "bidomain.h")

tostii_executable(bidomain_godunov
    "parameters.cc" "parameters.h"
    "prescribed_data.cc" "prescribed_data.inl" "prescribed_data.h"
    "bidomain_godunov.cc" "bidomain_godunov.inl" "bidomain_godunov.h")

tostii_executable(bidomain_godunov_2
    "parameters.cc" "parameters.h"
    "prescribed_data_2.cc" "prescribed_data_2.inl" "prescribed_data.h"
    "bidomain_godunov.cc" "bidomain_godunov.inl" "bidomain_godunov.h")

tostii_executable(bidomain_fhn
    "parameters.cc" "parameters.h"
    "fitzhugh-nagumo.h" "fitzhugh-nagumo.inl" "fitzhugh-nagumo.cc"
    "bidomain_fhn.cc" "bidomain_fhn.inl" "bidomain_fhn.h")
