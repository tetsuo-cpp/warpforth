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
// Base class for standard stack-to-stack operations.
//===----------------------------------------------------------------------===//

class Forth_StackOpBase<string mnemonic, list<Trait> traits = []>
    : Forth_Op<mnemonic, !listconcat([Pure], traits)> {
  let arguments = (ins Forth_StackType:$input_stack);
  let results = (outs Forth_StackType:$output_stack);
  let assemblyFormat = [{
    $input_stack attr-dict `:` type($input_stack) `->` type($output_stack)
  }];
}

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

def Forth_DupOp : Forth_StackOpBase<"dup"> {
  let summary = "Duplicate top stack element";
  let description = [{
    Duplicates the top element of the stack.
    Forth semantics: ( a -- a a )
  }];
}

def Forth_DropOp : Forth_StackOpBase<"drop"> {
  let summary = "Remove top stack element";
  let description = [{
    Removes the top element from the stack.
    Forth semantics: ( a -- )
  }];
}

def Forth_SwapOp : Forth_StackOpBase<"swap"> {
  let summary = "Swap top two stack elements";
  let description = [{
    Swaps the top two elements of the stack.
    Forth semantics: ( a b -- b a )
  }];
}

def Forth_OverOp : Forth_StackOpBase<"over"> {
  let summary = "Copy second element to top";
  let description = [{
    Copies the second element to the top of the stack.
    Forth semantics: ( a b -- a b a )
  }];
}

def Forth_RotOp : Forth_StackOpBase<"rot"> {
  let summary = "Rotate top three elements";
  let description = [{
    Rotates the top three elements of the stack.
    Forth semantics: ( a b c -- b c a )
  }];
}

def Forth_NipOp : Forth_StackOpBase<"nip"> {
  let summary = "Remove second stack element";
  let description = [{
    Removes the second element from the stack, keeping the top.
    Forth semantics: ( a b -- b )
  }];
}

def Forth_TuckOp : Forth_StackOpBase<"tuck"> {
  let summary = "Copy top element before second";
  let description = [{
    Copies the top element and inserts it before the second element.
    Forth semantics: ( a b -- b a b )
  }];
}

def Forth_PickOp : Forth_StackOpBase<"pick"> {
  let summary = "Copy nth element to top";
  let description = [{
    Pops n from the stack, then copies the nth element to the top.
    Forth semantics: ( xn ... x0 n -- xn ... x0 xn )
  }];
}

def Forth_RollOp : Forth_StackOpBase<"roll"> {
  let summary = "Rotate nth element to top";
  let description = [{
    Pops n from the stack, then rotates the nth element to the top,
    shifting elements above it down.
    Forth semantics: ( xn ... x0 n -- xn-1 ... x0 xn )
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

def Forth_AddOp : Forth_StackOpBase<"add"> {
  let summary = "Add top two stack elements";
  let description = [{
    Pops the top two elements, adds them, and pushes the result.
    Forth semantics: ( a b -- a+b )
  }];
}

def Forth_SubOp : Forth_StackOpBase<"sub"> {
  let summary = "Subtract top two stack elements";
  let description = [{
    Pops the top two elements, subtracts them (a - b), and pushes the result.
    Forth semantics: ( a b -- a-b )
  }];
}

def Forth_MulOp : Forth_StackOpBase<"mul"> {
  let summary = "Multiply top two stack elements";
  let description = [{
    Pops the top two elements, multiplies them, and pushes the result.
    Forth semantics: ( a b -- a*b )
  }];
}

def Forth_DivOp : Forth_StackOpBase<"div"> {
  let summary = "Divide top two stack elements";
  let description = [{
    Pops the top two elements, divides them (a / b), and pushes the result.
    Forth semantics: ( a b -- a/b )
  }];
}

def Forth_ModOp : Forth_StackOpBase<"mod"> {
  let summary = "Modulo of top two stack elements";
  let description = [{
    Pops the top two elements, computes modulo (a % b), and pushes the result.
    Forth semantics: ( a b -- a%b )
  }];
}

//===----------------------------------------------------------------------===//
// Bitwise operations.
//===----------------------------------------------------------------------===//

def Forth_AndOp : Forth_StackOpBase<"and"> {
  let summary = "Bitwise AND of top two stack elements";
  let description = [{
    Pops the top two elements, performs bitwise AND, and pushes the result.
    Forth semantics: ( a b -- a&b )
  }];
}

def Forth_OrOp : Forth_StackOpBase<"or"> {
  let summary = "Bitwise OR of top two stack elements";
  let description = [{
    Pops the top two elements, performs bitwise OR, and pushes the result.
    Forth semantics: ( a b -- a|b )
  }];
}

def Forth_XorOp : Forth_StackOpBase<"xor"> {
  let summary = "Bitwise XOR of top two stack elements";
  let description = [{
    Pops the top two elements, performs bitwise XOR, and pushes the result.
    Forth semantics: ( a b -- a^b )
  }];
}

def Forth_NotOp : Forth_StackOpBase<"not"> {
  let summary = "Bitwise NOT of top stack element";
  let description = [{
    Pops the top element, performs bitwise NOT (complement), and pushes the result.
    Forth semantics: ( a -- ~a )
  }];
}

def Forth_LshiftOp : Forth_StackOpBase<"lshift"> {
  let summary = "Left shift";
  let description = [{
    Pops shift amount and value, shifts value left, and pushes the result.
    Forth semantics: ( a n -- a<<n )
  }];
}

def Forth_RshiftOp : Forth_StackOpBase<"rshift"> {
  let summary = "Logical right shift";
  let description = [{
    Pops shift amount and value, shifts value right (logical/unsigned), and pushes the result.
    Forth semantics: ( a n -- a>>n )
  }];
}

//===----------------------------------------------------------------------===//
// Memory operations.
//===----------------------------------------------------------------------===//

def Forth_LoadOp : Forth_StackOpBase<"load"> {
  let summary = "Load value from memory buffer";
  let description = [{
    Pops an address from the stack, loads a value from the memory buffer at that address,
    and pushes the loaded value onto the stack.
    Forth semantics: ( addr -- value )
  }];
}

def Forth_StoreOp : Forth_StackOpBase<"store"> {
  let summary = "Store value to memory buffer";
  let description = [{
    Pops an address and value from the stack, stores the value to the memory buffer
    at the specified address.
    Forth semantics: ( x addr -- )
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

def Forth_ThreadIdXOp : Forth_StackOpBase<"thread_id_x"> {
  let summary = "Push x-dimension thread index";
  let description = [{
    Pushes the x-dimension thread index within the current block.
    Corresponds to CUDA's threadIdx.x.
    Forth semantics: ( -- thread_idx_x )
  }];
}

def Forth_ThreadIdYOp : Forth_StackOpBase<"thread_id_y"> {
  let summary = "Push y-dimension thread index";
  let description = [{
    Pushes the y-dimension thread index within the current block.
    Corresponds to CUDA's threadIdx.y.
    Forth semantics: ( -- thread_idx_y )
  }];
}

def Forth_ThreadIdZOp : Forth_StackOpBase<"thread_id_z"> {
  let summary = "Push z-dimension thread index";
  let description = [{
    Pushes the z-dimension thread index within the current block.
    Corresponds to CUDA's threadIdx.z.
    Forth semantics: ( -- thread_idx_z )
  }];
}

def Forth_BlockIdXOp : Forth_StackOpBase<"block_id_x"> {
  let summary = "Push x-dimension block index";
  let description = [{
    Pushes the x-dimension block index within the grid.
    Corresponds to CUDA's blockIdx.x.
    Forth semantics: ( -- block_idx_x )
  }];
}

def Forth_BlockIdYOp : Forth_StackOpBase<"block_id_y"> {
  let summary = "Push y-dimension block index";
  let description = [{
    Pushes the y-dimension block index within the grid.
    Corresponds to CUDA's blockIdx.y.
    Forth semantics: ( -- block_idx_y )
  }];
}

def Forth_BlockIdZOp : Forth_StackOpBase<"block_id_z"> {
  let summary = "Push z-dimension block index";
  let description = [{
    Pushes the z-dimension block index within the grid.
    Corresponds to CUDA's blockIdx.z.
    Forth semantics: ( -- block_idx_z )
  }];
}

def Forth_BlockDimXOp : Forth_StackOpBase<"block_dim_x"> {
  let summary = "Push x-dimension block dimension";
  let description = [{
    Pushes the number of threads in the x-dimension of a block.
    Corresponds to CUDA's blockDim.x.
    Forth semantics: ( -- block_dim_x )
  }];
}

def Forth_BlockDimYOp : Forth_StackOpBase<"block_dim_y"> {
  let summary = "Push y-dimension block dimension";
  let description = [{
    Pushes the number of threads in the y-dimension of a block.
    Corresponds to CUDA's blockDim.y.
    Forth semantics: ( -- block_dim_y )
  }];
}

def Forth_BlockDimZOp : Forth_StackOpBase<"block_dim_z"> {
  let summary = "Push z-dimension block dimension";
  let description = [{
    Pushes the number of threads in the z-dimension of a block.
    Corresponds to CUDA's blockDim.z.
    Forth semantics: ( -- block_dim_z )
  }];
}

def Forth_GridDimXOp : Forth_StackOpBase<"grid_dim_x"> {
  let summary = "Push x-dimension grid dimension";
  let description = [{
    Pushes the number of blocks in the x-dimension of the grid.
    Corresponds to CUDA's gridDim.x.
    Forth semantics: ( -- grid_dim_x )
  }];
}

def Forth_GridDimYOp : Forth_StackOpBase<"grid_dim_y"> {
  let summary = "Push y-dimension grid dimension";
  let description = [{
    Pushes the number of blocks in the y-dimension of the grid.
    Corresponds to CUDA's gridDim.y.
    Forth semantics: ( -- grid_dim_y )
  }];
}

def Forth_GridDimZOp : Forth_StackOpBase<"grid_dim_z"> {
  let summary = "Push z-dimension grid dimension";
  let description = [{
    Pushes the number of blocks in the z-dimension of the grid.
    Corresponds to CUDA's gridDim.z.
    Forth semantics: ( -- grid_dim_z )
  }];
}

def Forth_GlobalIdOp : Forth_StackOpBase<"global_id"> {
  let summary = "Push 1D global thread ID";
  let description = [{
    Computes and pushes the 1D global thread ID.
    Equivalent to blockIdx.x * blockDim.x + threadIdx.x.
    Forth semantics: ( -- global_id )
  }];
}

//===----------------------------------------------------------------------===//
// Synchronization operations.
//===----------------------------------------------------------------------===//

def Forth_BarrierOp : Forth_Op<"barrier", []> {
  let summary = "GPU thread synchronization barrier";
  let description = [{
    Synchronizes all threads in a block.
    Corresponds to CUDA's __syncthreads().
    Forth semantics: ( -- )
  }];
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// Comparison operations.
//===----------------------------------------------------------------------===//

def Forth_EqOp : Forth_StackOpBase<"eq"> {
  let summary = "Test equality of top two stack elements";
  let description = [{
    Pops two values, pushes -1 (true) if equal, 0 (false) otherwise.
    Forth semantics: ( a b -- flag )
  }];
}

def Forth_LtOp : Forth_StackOpBase<"lt"> {
  let summary = "Test less-than of top two stack elements";
  let description = [{
    Pops two values, pushes -1 (true) if a < b, 0 (false) otherwise.
    Forth semantics: ( a b -- flag )
  }];
}

def Forth_GtOp : Forth_StackOpBase<"gt"> {
  let summary = "Test greater-than of top two stack elements";
  let description = [{
    Pops two values, pushes -1 (true) if a > b, 0 (false) otherwise.
    Forth semantics: ( a b -- flag )
  }];
}

def Forth_NeOp : Forth_StackOpBase<"ne"> {
  let summary = "Test inequality of top two stack elements";
  let description = [{
    Pops two values, pushes -1 (true) if not equal, 0 (false) otherwise.
    Forth semantics: ( a b -- flag )
  }];
}

def Forth_LeOp : Forth_StackOpBase<"le"> {
  let summary = "Test less-than-or-equal of top two stack elements";
  let description = [{
    Pops two values, pushes -1 (true) if a <= b, 0 (false) otherwise.
    Forth semantics: ( a b -- flag )
  }];
}

def Forth_GeOp : Forth_StackOpBase<"ge"> {
  let summary = "Test greater-than-or-equal of top two stack elements";
  let description = [{
    Pops two values, pushes -1 (true) if a >= b, 0 (false) otherwise.
    Forth semantics: ( a b -- flag )
  }];
}

def Forth_ZeroEqOp : Forth_StackOpBase<"zero_eq"> {
  let summary = "Test if top of stack is zero";
  let description = [{
    Pops one value, pushes -1 (true) if zero, 0 (false) otherwise.
    Forth semantics: ( a -- flag )
  }];
}

//===----------------------------------------------------------------------===//
// Control flow support operations.
//===----------------------------------------------------------------------===//

def Forth_PopFlagOp : Forth_Op<"pop_flag", [Pure]> {
  let summary = "Pop top of stack as boolean flag";
  let description = [{
    Pops the top value from the stack and returns it as an i1 flag
    (non-zero = true, zero = false). Used by IF, UNTIL, WHILE.
    Forth semantics: ( flag -- )
  }];
  let arguments = (ins Forth_StackType:$input_stack);
  let results = (outs Forth_StackType:$output_stack, I1:$flag);
  let assemblyFormat = [{
    $input_stack attr-dict `:` type($input_stack) `->` type($output_stack) `,` type($flag)
  }];
}

def Forth_PopOp : Forth_Op<"pop", [Pure]> {
  let summary = "Pop top of stack as i64 value";
  let description = [{
    Pops the top value from the stack and returns it as an i64.
    Used by DO to pop start and limit.
    Forth semantics: ( x -- )
  }];
  let arguments = (ins Forth_StackType:$input_stack);
  let results = (outs Forth_StackType:$output_stack, I64:$value);
  let assemblyFormat = [{
    $input_stack attr-dict `:` type($input_stack) `->` type($output_stack) `,` type($value)
  }];
}

def Forth_PushValueOp : Forth_Op<"push_value", [Pure]> {
  let summary = "Push dynamic i64 value onto stack";
  let description = [{
    Pushes a dynamic i64 value onto the stack. Used by I/J/K to push
    the loop counter value.
    Forth semantics: ( -- x )
  }];
  let arguments = (ins Forth_StackType:$input_stack, I64:$value);
  let results = (outs Forth_StackType:$output_stack);
  let assemblyFormat = [{
    $input_stack `,` $value attr-dict `:` type($input_stack) `,` type($value) `->` type($output_stack)
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
