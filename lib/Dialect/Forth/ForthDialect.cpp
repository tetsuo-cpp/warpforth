//===- ForthDialect.cpp - Forth dialect ----------------------------------===//
//
// This file implements the Forth dialect.
//
//===----------------------------------------------------------------------===//

#include "warpforth/Dialect/Forth/ForthDialect.h"
#include "mlir/IR/Builders.h"              // IWYU pragma: keep
#include "mlir/IR/DialectImplementation.h" // IWYU pragma: keep
#include "llvm/ADT/TypeSwitch.h"           // IWYU pragma: keep

using namespace mlir;
using namespace mlir::forth;

#include "warpforth/Dialect/Forth/ForthOpsDialect.cpp.inc" // IWYU pragma: keep

#define GET_TYPEDEF_CLASSES
#include "warpforth/Dialect/Forth/ForthOpsTypes.cpp.inc"

#define GET_OP_CLASSES
#include "warpforth/Dialect/Forth/ForthOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Forth dialect.
//===----------------------------------------------------------------------===//

void ForthDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "warpforth/Dialect/Forth/ForthOpsTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "warpforth/Dialect/Forth/ForthOps.cpp.inc"
      >();
}
