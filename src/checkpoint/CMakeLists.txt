
set(INCLUDE_DIR "${TOSTII_INCLUDE_DIR}/checkpoint")
set(SOURCES
    "save_manager.cc"
    "checkpointer.cc")
set(HEADERS
    "${INCLUDE_DIR}/save_manager.h"
    "${INCLUDE_DIR}/save_manager.inl"
    "${INCLUDE_DIR}/checkpointer.h"
    "${INCLUDE_DIR}/checkpointer.inl")

add_library(tostii_checkpoint OBJECT ${SOURCES} ${HEADERS})
deal_II_setup_target(tostii_checkpoint)
target_include_directories(tostii_checkpoint PUBLIC
    "${CMAKE_SOURCE_DIR}/include")

install(FILES ${HEADERS}
    DESTINATION "include/tostii/checkpoint")
