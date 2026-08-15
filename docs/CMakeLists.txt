find_program(SPHINX_EXECUTABLE
  NAMES sphinx-build
  HINTS ${PROJECT_SOURCE_DIR}/.venv/bin
  REQUIRED)
find_package(Doxygen REQUIRED)

set(WARPFORTH_DOC_SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/source)
set(WARPFORTH_DOC_HTML_DIR ${CMAKE_CURRENT_BINARY_DIR}/html)
set(WARPFORTH_DIALECT_DOC
  ${PROJECT_BINARY_DIR}/include/warpforth/Dialect/Forth/ForthDialect.md)
set(WARPFORTH_STAGED_DIALECT_DOC
  ${WARPFORTH_DOC_SOURCE_DIR}/ForthDialect.md)

file(MAKE_DIRECTORY ${WARPFORTH_DOC_SOURCE_DIR}/_templates/sidebar)
configure_file(conf.py ${WARPFORTH_DOC_SOURCE_DIR}/conf.py COPYONLY)
configure_file(index.md ${WARPFORTH_DOC_SOURCE_DIR}/index.md COPYONLY)
configure_file(language.md ${WARPFORTH_DOC_SOURCE_DIR}/language.md COPYONLY)
configure_file(demo.md ${WARPFORTH_DOC_SOURCE_DIR}/demo.md COPYONLY)
configure_file(
  _templates/sidebar/navigation.html
  ${WARPFORTH_DOC_SOURCE_DIR}/_templates/sidebar/navigation.html
  COPYONLY)
configure_file(Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)

add_custom_command(
  OUTPUT ${WARPFORTH_STAGED_DIALECT_DOC}
  COMMAND ${CMAKE_COMMAND}
    -DINPUT=${WARPFORTH_DIALECT_DOC}
    -DOUTPUT=${WARPFORTH_STAGED_DIALECT_DOC}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/stage-dialect-doc.cmake
  DEPENDS
    ForthDialectDocGen
    ${WARPFORTH_DIALECT_DOC}
    ${CMAKE_CURRENT_SOURCE_DIR}/stage-dialect-doc.cmake
  COMMENT "Staging generated Forth dialect documentation"
  VERBATIM)
add_custom_target(warpforth-doc-sources
  DEPENDS ${WARPFORTH_STAGED_DIALECT_DOC})

add_custom_target(docs-warpforth-api
  COMMAND ${CMAKE_COMMAND} -E make_directory
    ${WARPFORTH_DOC_HTML_DIR}/api
  COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
  COMMENT "Generating WarpForth C++ API documentation"
  VERBATIM)

add_custom_target(docs-warpforth-html
  COMMAND ${SPHINX_EXECUTABLE}
    -b html
    -a
    -W
    --keep-going
    -d ${CMAKE_CURRENT_BINARY_DIR}/doctrees
    ${WARPFORTH_DOC_SOURCE_DIR}
    ${WARPFORTH_DOC_HTML_DIR}
  DEPENDS
    warpforth-doc-sources
    docs-warpforth-api
  COMMENT "Generating WarpForth HTML documentation"
  VERBATIM)

add_custom_target(docs-warpforth DEPENDS docs-warpforth-html)

set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES
  ${WARPFORTH_DOC_SOURCE_DIR}
  ${WARPFORTH_DOC_HTML_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}/doctrees)
