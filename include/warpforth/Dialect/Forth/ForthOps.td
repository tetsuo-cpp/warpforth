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

//===----------------------------------------------------------------------===//
// Memory operations.
//===----------------------------------------------------------------------===//

def Forth_LoadOp : Forth_Op<"load", [Pure]> {
  let summary = "Load value from memory buffer";
  let description = [{
    Pops an address from the stack, loads a value from the memory buffer at that address,
    and pushes the loaded value onto the stack.
    Forth semantics: ( addr -- value )
  }];

  let arguments = (ins Forth_StackType:$input_stack);
  let results = (outs Forth_StackType:$output_stack);

  let assemblyFormat = [{
    $input_stack attr-dict `:` type($input_stack) `->` type($output_stack)
  }];
}

def Forth_StoreOp : Forth_Op<"store", [Pure]> {
  let summary = "Store value to memory buffer";
  let description = [{
    Pops an address and value from the stack, stores the value to the memory buffer
    at the specified address.
    Forth semantics: ( x addr -- )
  }];

  let arguments = (ins Forth_StackType:$input_stack);
  let results = (outs Forth_StackType:$output_stack);

  let assemblyFormat = [{
    $input_stack attr-dict `:` type($input_stack) `->` type($output_stack)
  }];
}

def Forth_ParamRefOp : Forth_Op<"param_ref", [Pure]> {
  let summary = "Push kernel parameter address onto stack";
  let description = [{
    Pushes the byte address of a named kernel parameter onto the stack.
    Forth semantics: ( -- addr )
  }];
  let arguments = (ins Forth_StackType:$input_stack, StrAttr:$param_name);
  let results = (outs Forth_StackType:$output_stack);
  let assemblyFormat = [{
    $input_stack $param_name attr-dict `:` type($input_stack) `->` type($output_stack)
  }];
}

//===----------------------------------------------------------------------===//
// GPU indexing operations.
//===----------------------------------------------------------------------===//

def Forth_ThreadIdXOp : Forth_Op<"thread_id_x", [Pure]> {
  let summary = "Push x-dimension thread index";
  let description = [{
    Pushes the x-dimension thread index within the current block.
    Corresponds to CUDA's threadIdx.x.
    Forth semantics: ( -- thread_idx_x )
  }];
  let arguments = (ins Forth_StackType:$input_stack);
  let results = (outs Forth_StackType:$output_stack);
  let assemblyFormat = [{
    $input_stack attr-dict `:` type($input_stack) `->` type($output_stack)
  }];
}

def Forth_ThreadIdYOp : Forth_Op<"thread_id_y", [Pure]> {
  let summary = "Push y-dimension thread index";
  let description = [{
    Pushes the y-dimension thread index within the current block.
    Corresponds to CUDA's threadIdx.y.
    Forth semantics: ( -- thread_idx_y )
  }];
  let arguments = (ins Forth_StackType:$input_stack);
  let results = (outs Forth_StackType:$output_stack);
  let assemblyFormat = [{
    $input_stack attr-dict `:` type($input_stack) `->` type($output_stack)
  }];
}

def Forth_ThreadIdZOp : Forth_Op<"thread_id_z", [Pure]> {
  let summary = "Push z-dimension thread index";
  let description = [{
    Pushes the z-dimension thread index within the current block.
    Corresponds to CUDA's threadIdx.z.
    Forth semantics: ( -- thread_idx_z )
  }];
  let arguments = (ins Forth_StackType:$input_stack);
  let results = (outs Forth_StackType:$output_stack);
  let assemblyFormat = [{
    $input_stack attr-dict `:` type($input_stack) `->` type($output_stack)
  }];
}

def Forth_BlockIdXOp : Forth_Op<"block_id_x", [Pure]> {
  let summary = "Push x-dimension block index";
  let description = [{
    Pushes the x-dimension block index within the grid.
    Corresponds to CUDA's blockIdx.x.
    Forth semantics: ( -- block_idx_x )
  }];
  let arguments = (ins Forth_StackType:$input_stack);
  let results = (outs Forth_StackType:$output_stack);
  let assemblyFormat = [{
    $input_stack attr-dict `:` type($input_stack) `->` type($output_stack)
  }];
}

def Forth_BlockIdYOp : Forth_Op<"block_id_y", [Pure]> {
  let summary = "Push y-dimension block index";
  let description = [{
    Pushes the y-dimension block index within the grid.
    Corresponds to CUDA's blockIdx.y.
    Forth semantics: ( -- block_idx_y )
  }];
  let arguments = (ins Forth_StackType:$input_stack);
  let results = (outs Forth_StackType:$output_stack);
  let assemblyFormat = [{
    $input_stack attr-dict `:` type($input_stack) `->` type($output_stack)
  }];
}

def Forth_BlockIdZOp : Forth_Op<"block_id_z", [Pure]> {
  let summary = "Push z-dimension block index";
  let description = [{
    Pushes the z-dimension block index within the grid.
    Corresponds to CUDA's blockIdx.z.
    Forth semantics: ( -- block_idx_z )
  }];
  let arguments = (ins Forth_StackType:$input_stack);
  let results = (outs Forth_StackType:$output_stack);
  let assemblyFormat = [{
    $input_stack attr-dict `:` type($input_stack) `->` type($output_stack)
  }];
}

def Forth_BlockDimXOp : Forth_Op<"block_dim_x", [Pure]> {
  let summary = "Push x-dimension block dimension";
  let description = [{
    Pushes the number of threads in the x-dimension of a block.
    Corresponds to CUDA's blockDim.x.
    Forth semantics: ( -- block_dim_x )
  }];
  let arguments = (ins Forth_StackType:$input_stack);
  let results = (outs Forth_StackType:$output_stack);
  let assemblyFormat = [{
    $input_stack attr-dict `:` type($input_stack) `->` type($output_stack)
  }];
}

def Forth_BlockDimYOp : Forth_Op<"block_dim_y", [Pure]> {
  let summary = "Push y-dimension block dimension";
  let description = [{
    Pushes the number of threads in the y-dimension of a block.
    Corresponds to CUDA's blockDim.y.
    Forth semantics: ( -- block_dim_y )
  }];
  let arguments = (ins Forth_StackType:$input_stack);
  let results = (outs Forth_StackType:$output_stack);
  let assemblyFormat = [{
    $input_stack attr-dict `:` type($input_stack) `->` type($output_stack)
  }];
}

def Forth_BlockDimZOp : Forth_Op<"block_dim_z", [Pure]> {
  let summary = "Push z-dimension block dimension";
  let description = [{
    Pushes the number of threads in the z-dimension of a block.
    Corresponds to CUDA's blockDim.z.
    Forth semantics: ( -- block_dim_z )
  }];
  let arguments = (ins Forth_StackType:$input_stack);
  let results = (outs Forth_StackType:$output_stack);
  let assemblyFormat = [{
    $input_stack attr-dict `:` type($input_stack) `->` type($output_stack)
  }];
}

def Forth_GridDimXOp : Forth_Op<"grid_dim_x", [Pure]> {
  let summary = "Push x-dimension grid dimension";
  let description = [{
    Pushes the number of blocks in the x-dimension of the grid.
    Corresponds to CUDA's gridDim.x.
    Forth semantics: ( -- grid_dim_x )
  }];
  let arguments = (ins Forth_StackType:$input_stack);
  let results = (outs Forth_StackType:$output_stack);
  let assemblyFormat = [{
    $input_stack attr-dict `:` type($input_stack) `->` type($output_stack)
  }];
}

def Forth_GridDimYOp : Forth_Op<"grid_dim_y", [Pure]> {
  let summary = "Push y-dimension grid dimension";
  let description = [{
    Pushes the number of blocks in the y-dimension of the grid.
    Corresponds to CUDA's gridDim.y.
    Forth semantics: ( -- grid_dim_y )
  }];
  let arguments = (ins Forth_StackType:$input_stack);
  let results = (outs Forth_StackType:$output_stack);
  let assemblyFormat = [{
    $input_stack attr-dict `:` type($input_stack) `->` type($output_stack)
  }];
}

def Forth_GridDimZOp : Forth_Op<"grid_dim_z", [Pure]> {
  let summary = "Push z-dimension grid dimension";
  let description = [{
    Pushes the number of blocks in the z-dimension of the grid.
    Corresponds to CUDA's gridDim.z.
    Forth semantics: ( -- grid_dim_z )
  }];
  let arguments = (ins Forth_StackType:$input_stack);
  let results = (outs Forth_StackType:$output_stack);
  let assemblyFormat = [{
    $input_stack attr-dict `:` type($input_stack) `->` type($output_stack)
  }];
}

def Forth_GlobalIdOp : Forth_Op<"global_id", [Pure]> {
  let summary = "Push 1D global thread ID";
  let description = [{
    Computes and pushes the 1D global thread ID.
    Equivalent to blockIdx.x * blockDim.x + threadIdx.x.
    Forth semantics: ( -- global_id )
  }];
  let arguments = (ins Forth_StackType:$input_stack);
  let results = (outs Forth_StackType:$output_stack);
  let assemblyFormat = [{
    $input_stack attr-dict `:` type($input_stack) `->` type($output_stack)
  }];
}

//===----------------------------------------------------------------------===//
// Intrinsic operations.
//===----------------------------------------------------------------------===//

def Forth_IntrinsicOp : Forth_Op<"intrinsic", [Pure]> {
  let summary = "Push intrinsic value onto the stack";
  let description = [{
    Pushes a hardware intrinsic value onto the stack.
    The intrinsic name is specified as a string attribute.
    Forth semantics: ( -- value )
  }];
  let arguments = (ins StrAttr:$intrinsic);
  let results = (outs Index:$value);
  let assemblyFormat = [{
    $intrinsic attr-dict `:` type($value)
  }];
}

#endif // FORTH_OPS
