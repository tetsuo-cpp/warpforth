//===- ForthOps.td - Forth dialect ops -------------------*- tablegen -*-===//
//
// This file defines operations in the Forth dialect.
//
//===----------------------------------------------------------------------===//

#ifndef FORTH_OPS
#define FORTH_OPS

include "warpforth/Dialect/Forth/ForthDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Stack manipulation operations.
//===----------------------------------------------------------------------===//

def Forth_StackOp : Forth_Op<"stack", [Pure]> {
  let summary = "Initializes the stack";
  let description = [{
    Initializes the stack.
  }];

  let results = (outs Forth_StackType:$output_stack);

  let assemblyFormat = [{
    attr-dict type($output_stack)
  }];
}

def Forth_DupOp : Forth_Op<"dup", [Pure]> {
  let summary = "Duplicate top stack element";
  let description = [{
    Duplicates the top element of the stack.
    Forth semantics: ( a -- a a )
  }];

  let arguments = (ins Forth_StackType:$input_stack);
  let results = (outs Forth_StackType:$output_stack);

  let assemblyFormat = [{
    $input_stack attr-dict `:` type($input_stack) `->` type($output_stack)
  }];
}

def Forth_DropOp : Forth_Op<"drop", [Pure]> {
  let summary = "Remove top stack element";
  let description = [{
    Removes the top element from the stack.
    Forth semantics: ( a -- )
  }];

  let arguments = (ins Forth_StackType:$input_stack);
  let results = (outs Forth_StackType:$output_stack);

  let assemblyFormat = [{
    $input_stack attr-dict `:` type($input_stack) `->` type($output_stack)
  }];
}

def Forth_SwapOp : Forth_Op<"swap", [Pure]> {
  let summary = "Swap top two stack elements";
  let description = [{
    Swaps the top two elements of the stack.
    Forth semantics: ( a b -- b a )
  }];

  let arguments = (ins Forth_StackType:$input_stack);
  let results = (outs Forth_StackType:$output_stack);

  let assemblyFormat = [{
    $input_stack attr-dict `:` type($input_stack) `->` type($output_stack)
  }];
}

def Forth_OverOp : Forth_Op<"over", [Pure]> {
  let summary = "Copy second element to top";
  let description = [{
    Copies the second element to the top of the stack.
    Forth semantics: ( a b -- a b a )
  }];

  let arguments = (ins Forth_StackType:$input_stack);
  let results = (outs Forth_StackType:$output_stack);

  let assemblyFormat = [{
    $input_stack attr-dict `:` type($input_stack) `->` type($output_stack)
  }];
}

def Forth_RotOp : Forth_Op<"rot", [Pure]> {
  let summary = "Rotate top three elements";
  let description = [{
    Rotates the top three elements of the stack.
    Forth semantics: ( a b c -- b c a )
  }];

  let arguments = (ins Forth_StackType:$input_stack);
  let results = (outs Forth_StackType:$output_stack);

  let assemblyFormat = [{
    $input_stack attr-dict `:` type($input_stack) `->` type($output_stack)
  }];
}

//===----------------------------------------------------------------------===//
// Literal operations.
//===----------------------------------------------------------------------===//

def Forth_LiteralOp : Forth_Op<"literal", [Pure]> {
  let summary = "Push a literal value onto the stack";
  let description = [{
    Pushes a literal integer value onto the stack.
    Forth semantics: ( -- n )
  }];

  let arguments = (ins Forth_StackType:$input_stack, I64Attr:$value);
  let results = (outs Forth_StackType:$output_stack);

  let assemblyFormat = [{
    $input_stack $value attr-dict `:` type($input_stack) `->` type($output_stack)
  }];
}

//===----------------------------------------------------------------------===//
// Arithmetic operations.
//===----------------------------------------------------------------------===//

def Forth_AddOp : Forth_Op<"add", [Pure]> {
  let summary = "Add top two stack elements";
  let description = [{
    Pops the top two elements, adds them, and pushes the result.
    Forth semantics: ( a b -- a+b )
  }];

  let arguments = (ins Forth_StackType:$input_stack);
  let results = (outs Forth_StackType:$output_stack);

  let assemblyFormat = [{
    $input_stack attr-dict `:` type($input_stack) `->` type($output_stack)
  }];
}

def Forth_SubOp : Forth_Op<"sub", [Pure]> {
  let summary = "Subtract top two stack elements";
  let description = [{
    Pops the top two elements, subtracts them (a - b), and pushes the result.
    Forth semantics: ( a b -- a-b )
  }];

  let arguments = (ins Forth_StackType:$input_stack);
  let results = (outs Forth_StackType:$output_stack);

  let assemblyFormat = [{
    $input_stack attr-dict `:` type($input_stack) `->` type($output_stack)
  }];
}

def Forth_MulOp : Forth_Op<"mul", [Pure]> {
  let summary = "Multiply top two stack elements";
  let description = [{
    Pops the top two elements, multiplies them, and pushes the result.
    Forth semantics: ( a b -- a*b )
  }];

  let arguments = (ins Forth_StackType:$input_stack);
  let results = (outs Forth_StackType:$output_stack);

  let assemblyFormat = [{
    $input_stack attr-dict `:` type($input_stack) `->` type($output_stack)
  }];
}

def Forth_DivOp : Forth_Op<"div", [Pure]> {
  let summary = "Divide top two stack elements";
  let description = [{
    Pops the top two elements, divides them (a / b), and pushes the result.
    Forth semantics: ( a b -- a/b )
  }];

  let arguments = (ins Forth_StackType:$input_stack);
  let results = (outs Forth_StackType:$output_stack);

  let assemblyFormat = [{
    $input_stack attr-dict `:` type($input_stack) `->` type($output_stack)
  }];
}

def Forth_ModOp : Forth_Op<"mod", [Pure]> {
  let summary = "Modulo of top two stack elements";
  let description = [{
    Pops the top two elements, computes modulo (a % b), and pushes the result.
    Forth semantics: ( a b -- a%b )
  }];

  let arguments = (ins Forth_StackType:$input_stack);
  let results = (outs Forth_StackType:$output_stack);

  let assemblyFormat = [{
    $input_stack attr-dict `:` type($input_stack) `->` type($output_stack)
  }];
}

#endif // FORTH_OPS
