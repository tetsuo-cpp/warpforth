#ifndef WARPFORTH_DIALECT_FORTH_FORTHOPS_H
#define WARPFORTH_DIALECT_FORTH_FORTHOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "warpforth/Dialect/Forth/ForthOps.h.inc"

#endif // WARPFORTH_DIALECT_FORTH_FORTHOPS_H
