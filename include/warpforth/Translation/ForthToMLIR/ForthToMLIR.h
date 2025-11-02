//===- ForthToMLIR.h - Forth to MLIR translation ----------------*- C++ -*-===//
//
// This file declares the registration function for Forth-to-MLIR translation.
//
//===----------------------------------------------------------------------===//

#pragma once

namespace mlir {
namespace forth {

/// Register the Forth to MLIR translation.
/// This enables the --forth-to-mlir flag in mlir-translate tools.
void registerForthToMLIRTranslation();

} // namespace forth
} // namespace mlir
