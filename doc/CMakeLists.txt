
find_package(Doxygen)

if (${DOXYGEN_FOUND})
    set(DOXYGEN_PROJECT_NAME tostii)
    doxygen_add_docs(tostii_doc
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/include/tostii")
    
    install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/html"
        DESTINATION "doc/tostii")
endif()
