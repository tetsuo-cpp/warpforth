//===- ForthImporter.h - Forth source to MLIR importer ---------*- C++ -*-===//
//
// This file declares the interface for importing Forth source code into
// MLIR Forth dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef WARPFORTH_TRANSLATION_FORTHIMPORTER_H
#define WARPFORTH_TRANSLATION_FORTHIMPORTER_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SourceMgr.h"
#include <memory>

namespace mlir {
namespace forth {

/// Import Forth source code into MLIR Forth dialect.
///
/// \param sourceMgr - Source manager containing the Forth source code
/// \param context - MLIR context to create operations in
/// \return The parsed MLIR module, or nullptr on error
mlir::OwningOpRef<mlir::ModuleOp> importForth(llvm::SourceMgr &sourceMgr,
                                              mlir::MLIRContext *context);

} // namespace forth
} // namespace mlir

#endif // WARPFORTH_TRANSLATION_FORTHIMPORTER_H
