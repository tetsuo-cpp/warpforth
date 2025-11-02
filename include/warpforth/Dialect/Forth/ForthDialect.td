//===- ForthDialect.td - Forth dialect -------------------*- tablegen -*-===//
//
// This file defines the Forth dialect.
//
//===----------------------------------------------------------------------===//

#ifndef FORTH_DIALECT
#define FORTH_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Forth dialect definition.
//===----------------------------------------------------------------------===//

def Forth_Dialect : Dialect {
  let name = "forth";
  let summary = "A Forth dialect.";
  let description = [{
        This dialect provides operations for Forth language constructs,
        designed for GPU kernel programming.
    }];
  let cppNamespace = "::mlir::forth";
}

//===----------------------------------------------------------------------===//
// Base forth operation definition.
//===----------------------------------------------------------------------===//

class Forth_Op<string mnemonic, list<Trait> traits = []>
    : Op<Forth_Dialect, mnemonic, traits>;

#endif // FORTH_DIALECT
