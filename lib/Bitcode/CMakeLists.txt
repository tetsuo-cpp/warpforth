set(WARPFORTH_LIBDEVICE_PATH
  "${CMAKE_CURRENT_BINARY_DIR}/libdevice.10.bc" PARENT_SCOPE)

configure_file(libdevice.10.bc
  "${CMAKE_CURRENT_BINARY_DIR}/libdevice.10.bc" COPYONLY)
