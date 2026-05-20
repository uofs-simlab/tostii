
tostii_executable(monodomain
    "monodomain.cc"
    "FitzHughNagumo.h")

tostii_executable(parallel_implicit_fhn
    "parallel_implicit_fhn.cc")

tostii_executable(parallel_bdf_fhn
    "parallel_bdf_fhn.cc")

tostii_executable(parallel_os_fhn
    "parallel_os_fhn.cc")

tostii_executable(parallel_sdirk_fhn
    "parallel_sdirk_fhn.cc")
    
