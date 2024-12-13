
macro(tostii_executable target)
    deal_II_executable(${target} ${ARGN})
    target_include_directories(${target} PUBLIC
        "${TOSTII_INCLUDE_DIR}")
    target_link_libraries(${target}
        tostii)
endmacro()
