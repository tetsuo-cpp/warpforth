//===- ForthOps.td - Forth dialect ops -----------*- tablegen -*-===//
//
// This file defines the operations for the Forth dialect.
//
//===----------------------------------------------------------------------===//

#ifndef FORTH_OPS
#define FORTH_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Forth Dialect Definition
//===----------------------------------------------------------------------===//

def Forth_Dialect : Dialect {
  let name = "forth";
  let summary = "A FORTH dialect for GPU kernel programming";
  let description = [{
    The Forth dialect provides stack-based operations similar to the FORTH
    programming language, designed to be lowered to GPU kernels via MLIR's
    GPU dialect.
  }];
  let cppNamespace = "::mlir::forth";
}

//===----------------------------------------------------------------------===//
// Base Forth Operation
//===----------------------------------------------------------------------===//

class Forth_Op<string mnemonic, list<Trait> traits = []> :
    Op<Forth_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Forth Operations
//===----------------------------------------------------------------------===//

def Forth_ConstantOp : Forth_Op<"constant", [Pure]> {
  let summary = "Push a constant value onto the stack";
  let description = [{
    The `forth.constant` operation pushes a compile-time constant value
    onto the Forth stack.

    Example:
    ```mlir
    %0 = forth.constant 42 : i32
    ```
  }];

  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyType:$result);

  let assemblyFormat = "`(` $value `)` attr-dict `:` type($result)";
}

def Forth_AddOp : Forth_Op<"add", [Pure]> {
  let summary = "Add two values from the stack";
  let description = [{
    The `forth.add` operation pops two values from the stack, adds them,
    and pushes the result back onto the stack.

    Example:
    ```mlir
    %2 = forth.add %0, %1 : i32
    ```
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)";
}

def Forth_SubOp : Forth_Op<"sub", [Pure]> {
  let summary = "Subtract two values from the stack";
  let description = [{
    The `forth.sub` operation pops two values from the stack, subtracts
    the second from the first, and pushes the result back onto the stack.

    Example:
    ```mlir
    %2 = forth.sub %0, %1 : i32
    ```
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)";
}

def Forth_MulOp : Forth_Op<"mul", [Pure]> {
  let summary = "Multiply two values from the stack";
  let description = [{
    The `forth.mul` operation pops two values from the stack, multiplies them,
    and pushes the result back onto the stack.

    Example:
    ```mlir
    %2 = forth.mul %0, %1 : i32
    ```
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)";
}

def Forth_DivOp : Forth_Op<"div", [Pure]> {
  let summary = "Divide two values from the stack";
  let description = [{
    The `forth.div` operation pops two values from the stack, divides
    the first by the second, and pushes the result back onto the stack.

    Example:
    ```mlir
    %2 = forth.div %0, %1 : i32
    ```
  }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)";
}

def Forth_DupOp : Forth_Op<"dup", [Pure, SameOperandsAndResultType]> {
  let summary = "Duplicate the top stack value";
  let description = [{
    The `forth.dup` operation duplicates the value on top of the stack.

    Example:
    ```mlir
    %1 = forth.dup %0 : i32
    ```
  }];

  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);

  let assemblyFormat = "$input attr-dict `:` type($input)";
}

def Forth_DropOp : Forth_Op<"drop"> {
  let summary = "Drop the top stack value";
  let description = [{
    The `forth.drop` operation removes the value on top of the stack.

    Example:
    ```mlir
    forth.drop %0 : i32
    ```
  }];

  let arguments = (ins AnyType:$input);

  let assemblyFormat = "$input attr-dict `:` type($input)";
}

def Forth_SwapOp : Forth_Op<"swap", [Pure]> {
  let summary = "Swap the top two stack values";
  let description = [{
    The `forth.swap` operation swaps the top two values on the stack.

    Example:
    ```mlir
    %2, %3 = forth.swap %0, %1 : i32, i32
    ```
  }];

  let arguments = (ins AnyType:$first, AnyType:$second);
  let results = (outs AnyType:$result_first, AnyType:$result_second);

  let assemblyFormat = "$first `,` $second attr-dict `:` `(` type($first) `,` type($second) `)` `->` `(` type($result_first) `,` type($result_second) `)`";
}

#endif // FORTH_OPS
