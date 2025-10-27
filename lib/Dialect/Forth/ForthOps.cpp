//===- ForthOps.cpp - Forth dialect ops -------------------------*- C++ -*-===//
//
// This file implements the operations for the Forth dialect.
//
//===----------------------------------------------------------------------===//

#include "warpforth/Dialect/Forth/ForthOps.h"
#include "mlir/IR/OpImplementation.h"
#include "warpforth/Dialect/Forth/ForthDialect.h"

using namespace mlir;
using namespace mlir::forth;

#define GET_OP_CLASSES
#include "warpforth/Dialect/Forth/ForthOps.cpp.inc"
