# FindMLIR.cmake
# Locates MLIR installation
#
# This module defines:
#  MLIR_FOUND - System has MLIR
#  MLIR_INCLUDE_DIRS - The MLIR include directories
#  MLIR_LIBRARY_DIRS - The MLIR library directories
#  MLIR_DEFINITIONS - Compiler switches required for using MLIR

# Try to find MLIR via llvm-config if available
find_program(LLVM_CONFIG_EXECUTABLE NAMES llvm-config DOC "llvm-config executable")

if(LLVM_CONFIG_EXECUTABLE)
    execute_process(
        COMMAND ${LLVM_CONFIG_EXECUTABLE} --prefix
        OUTPUT_VARIABLE LLVM_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(MLIR_DIR "${LLVM_PREFIX}/lib/cmake/mlir" CACHE PATH "Path to MLIR cmake directory")
endif()

# Let CMake find the config
find_package(MLIR CONFIG)

if(MLIR_FOUND)
    set(MLIR_INCLUDE_DIRS ${MLIR_INCLUDE_DIRS})
    set(MLIR_LIBRARY_DIRS ${MLIR_LIBRARY_DIRS})
    message(STATUS "Found MLIR: ${MLIR_DIR}")
else()
    message(FATAL_ERROR "MLIR not found. Please set MLIR_DIR to the MLIR cmake directory.")
endif()
