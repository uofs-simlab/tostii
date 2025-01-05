
macro(deal_II_executable target)
    add_executable(${target} ${ARGN})
    deal_II_setup_target(${target})
    set_target_properties(${target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")
endmacro()
