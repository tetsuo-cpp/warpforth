//===- MLIRToPTX.h - MLIR to PTX translation --------------------*- C++ -*-===//
//
// This file declares the registration function for MLIR-to-PTX translation.
// Extracts PTX assembly from gpu.binary operations.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace warpforth {

/// Register the MLIR to PTX translation.
/// This enables the --mlir-to-ptx flag in warpforth-translate.
/// Extracts PTX assembly from gpu.binary operations and writes to output.
void registerMLIRToPTXTranslation();

/// Extract PTX assembly from all gpu.binary operations in the module.
LogicalResult extractPTXFromModule(ModuleOp module, llvm::raw_ostream &output);

} // namespace warpforth
} // namespace mlir
