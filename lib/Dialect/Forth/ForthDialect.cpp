//===- ForthDialect.cpp - Forth dialect ----------------------------------===//
//
// This file implements the Forth dialect.
//
//===----------------------------------------------------------------------===//

#include "warpforth/Dialect/Forth/ForthDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::forth;

#include "warpforth/Dialect/Forth/ForthOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Forth dialect.
//===----------------------------------------------------------------------===//

void ForthDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "warpforth/Dialect/Forth/ForthOps.cpp.inc"
      >();
}
