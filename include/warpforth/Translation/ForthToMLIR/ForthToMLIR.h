//===- ForthToMLIR.h - Forth to MLIR translation ----------------*- C++ -*-===//
//
// This file declares the registration function for Forth-to-MLIR translation.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/Support/SourceMgr.h"

namespace mlir {
class MLIRContext;
class Operation;

namespace forth {

/// Register the Forth to MLIR translation.
/// This enables the --forth-to-mlir flag in mlir-translate tools.
void registerForthToMLIRTranslation();

/// Parse Forth source and convert to MLIR.
OwningOpRef<ModuleOp> parseForthSource(llvm::SourceMgr &sourceMgr,
                                       MLIRContext *context);

} // namespace forth
} // namespace mlir
