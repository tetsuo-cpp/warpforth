//===- ForthDialect.h - Forth dialect ---------------------------*- C++ -*-===//
//
// This file defines the Forth dialect.
//
//===----------------------------------------------------------------------===//

#ifndef WARPFORTH_DIALECT_FORTH_FORTHDIALECT_H
#define WARPFORTH_DIALECT_FORTH_FORTHDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"    // IWYU pragma: keep
#include "mlir/Interfaces/SideEffectInterfaces.h" // IWYU pragma: keep
#include <cstdint>

#include "warpforth/Dialect/Forth/ForthOpsDialect.h.inc" // IWYU pragma: keep

#define GET_TYPEDEF_CLASSES
#include "warpforth/Dialect/Forth/ForthOpsTypes.h.inc"

#include "warpforth/Dialect/Forth/ForthOpsEnums.h.inc" // IWYU pragma: keep

#define GET_OP_CLASSES
#include "warpforth/Dialect/Forth/ForthOps.h.inc"

namespace mlir::forth {
inline constexpr int64_t kWarpSize = 32;
} // namespace mlir::forth

#endif
