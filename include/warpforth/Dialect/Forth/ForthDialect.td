//===- ForthDialect.td - Forth dialect -------------------*- tablegen -*-===//
//
// This file defines the Forth dialect.
//
//===----------------------------------------------------------------------===//

#ifndef FORTH_DIALECT
#define FORTH_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

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
  let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Base forth operation definition.
//===----------------------------------------------------------------------===//

class Forth_Op<string mnemonic, list<Trait> traits = []>
    : Op<Forth_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Forth type definitions.
//===----------------------------------------------------------------------===//

class Forth_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Forth_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Forth_StackType : Forth_Type<"Stack", "stack"> {
  let summary = "Forth stack type";
  let description = [{
    Represents the Forth data stack. This is an untyped stack where
    elements can be of any type. It is the programmer's responsibility
    to ensure type correctness when manipulating the stack.
  }];
}

#endif // FORTH_DIALECT
