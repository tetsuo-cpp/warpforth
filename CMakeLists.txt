cmake_minimum_required(VERSION 3.20.0)
project(warpforth LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
set(LLVM_ENABLE_WERROR ON)
include(HandleLLVMOptions)

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)

option(WARPFORTH_ENABLE_DOCS "Build WarpForth documentation" OFF)
if(WARPFORTH_ENABLE_DOCS)
  add_subdirectory(docs)
endif()

# Add clang-tidy target after the TableGen targets it depends on are defined.
find_program(CLANG_TIDY_EXECUTABLE
  NAMES clang-tidy-${LLVM_VERSION_MAJOR} clang-tidy
  HINTS ${LLVM_TOOLS_BINARY_DIR}
  REQUIRED)
find_program(RUN_CLANG_TIDY_EXECUTABLE
  NAMES run-clang-tidy-${LLVM_VERSION_MAJOR}.py run-clang-tidy
  HINTS ${LLVM_TOOLS_BINARY_DIR}
  REQUIRED)
add_custom_target(check-clang-tidy
  COMMAND ${RUN_CLANG_TIDY_EXECUTABLE}
    -clang-tidy-binary ${CLANG_TIDY_EXECUTABLE}
    -config-file ${PROJECT_SOURCE_DIR}/.clang-tidy
    -exclude-header-filter ^${PROJECT_BINARY_DIR}/
    -p ${PROJECT_BINARY_DIR}
    -j 2
    -quiet
  DEPENDS
    MLIRConversionPassIncGen
    MLIRForthOpsIncGen
  COMMENT "Running clang-tidy"
  VERBATIM
)

find_program(IWYU_EXECUTABLE NAMES include-what-you-use)
find_program(IWYU_TOOL_EXECUTABLE NAMES iwyu_tool.py iwyu_tool)
if(IWYU_EXECUTABLE AND IWYU_TOOL_EXECUTABLE)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_custom_target(check-iwyu
    COMMAND ${CMAKE_COMMAND} -E env IWYU_BINARY=${IWYU_EXECUTABLE}
      ${Python3_EXECUTABLE}
      ${IWYU_TOOL_EXECUTABLE}
      -p ${PROJECT_BINARY_DIR}
      -j 2
      --
      -Xiwyu --no_fwd_decls
    DEPENDS
      MLIRConversionPassIncGen
      MLIRForthOpsIncGen
    COMMENT "Checking C++ includes with include-what-you-use"
    VERBATIM
  )
else()
  message(STATUS "include-what-you-use not found, skipping check-iwyu target")
endif()

# Test infrastructure (lit + FileCheck)
find_program(LIT_COMMAND NAMES lit llvm-lit
  HINTS ${CMAKE_SOURCE_DIR}/.venv/bin)
find_program(FILECHECK_PATH NAMES FileCheck
  HINTS /opt/homebrew/opt/llvm/bin ${LLVM_TOOLS_BINARY_DIR})
if(LIT_COMMAND AND FILECHECK_PATH)
  message(STATUS "Found lit: ${LIT_COMMAND}")
  message(STATUS "Found FileCheck: ${FILECHECK_PATH}")
  add_subdirectory(test)
else()
  message(STATUS "lit or FileCheck not found, skipping tests")
endif()

# Add format target to run clang-format on all source files
find_program(CLANG_FORMAT_EXECUTABLE
  NAMES clang-format-${LLVM_VERSION_MAJOR} clang-format)
if(CLANG_FORMAT_EXECUTABLE)
  file(GLOB_RECURSE ALL_SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/include/**/*.h
    ${PROJECT_SOURCE_DIR}/lib/**/*.cpp
    ${PROJECT_SOURCE_DIR}/tools/**/*.cpp
  )
  add_custom_target(format
    COMMAND ${CLANG_FORMAT_EXECUTABLE} -i ${ALL_SOURCE_FILES}
    COMMENT "Running clang-format on all source files"
    VERBATIM
  )
  add_custom_target(check-format
    COMMAND ${CLANG_FORMAT_EXECUTABLE} --dry-run --Werror ${ALL_SOURCE_FILES}
    COMMENT "Checking clang-format on all source files"
    VERBATIM
  )
endif()
