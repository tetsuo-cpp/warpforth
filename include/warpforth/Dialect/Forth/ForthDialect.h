//===- ForthDialect.h - Forth dialect ---------------------------*- C++ -*-===//
//
// This file defines the Forth dialect.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "warpforth/Dialect/Forth/ForthOpsDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "warpforth/Dialect/Forth/ForthOpsTypes.h.inc"

#define GET_OP_CLASSES
#include "warpforth/Dialect/Forth/ForthOps.h.inc"
