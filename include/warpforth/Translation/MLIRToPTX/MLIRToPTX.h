//===- MLIRToPTX.h - MLIR to PTX translation --------------------*- C++ -*-===//
//
// This file declares the registration function for MLIR-to-PTX translation.
// Extracts PTX assembly from gpu.binary operations.
//
//===----------------------------------------------------------------------===//

#pragma once

namespace mlir {
namespace warpforth {

/// Register the MLIR to PTX translation.
/// This enables the --mlir-to-ptx flag in warpforth-translate.
/// Extracts PTX assembly from gpu.binary operations and writes to output.
void registerMLIRToPTXTranslation();

} // namespace warpforth
} // namespace mlir
