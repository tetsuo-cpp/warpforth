// RUN: %warpforth-opt --lower-forth-stack %s | %FileCheck %s

// Each function pushes 1 2 3, runs one stack word, and stores the cells that
// remain, top first. The stores read the literals themselves: the word only
// reorders SSA values, so no stack buffer exists.

// ( 1 2 3 -- 1 2 3 3 )
// CHECK-LABEL: func.func @dup(
// CHECK-NOT: memref.alloca
// CHECK: %[[ONE:.*]] = arith.constant 1 : i64
// CHECK: %[[TWO:.*]] = arith.constant 2 : i64
// CHECK: %[[THREE:.*]] = arith.constant 3 : i64
// CHECK: llvm.store %[[THREE]],
// CHECK: llvm.store %[[THREE]],
// CHECK: llvm.store %[[TWO]],
// CHECK: llvm.store %[[ONE]],
// CHECK-NEXT: return
func.func @dup(%out: memref<1xi64>) {
  %s0 = forth.stack !forth.stack
  %s1 = forth.constant %s0(1 : i64) : !forth.stack -> !forth.stack
  %s2 = forth.constant %s1(2 : i64) : !forth.stack -> !forth.stack
  %s3 = forth.constant %s2(3 : i64) : !forth.stack -> !forth.stack
  %s4 = forth.dup %s3 : !forth.stack -> !forth.stack
  %s5 = forth.param_ref %s4 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s6 = forth.storei %s5 : !forth.stack -> !forth.stack
  %s7 = forth.param_ref %s6 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s8 = forth.storei %s7 : !forth.stack -> !forth.stack
  %s9 = forth.param_ref %s8 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s10 = forth.storei %s9 : !forth.stack -> !forth.stack
  %s11 = forth.param_ref %s10 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s12 = forth.storei %s11 : !forth.stack -> !forth.stack
  return
}

// ( 1 2 3 -- 1 2 )
// CHECK-LABEL: func.func @drop(
// CHECK-NOT: memref.alloca
// CHECK: %[[ONE:.*]] = arith.constant 1 : i64
// CHECK: %[[TWO:.*]] = arith.constant 2 : i64
// CHECK: %[[THREE:.*]] = arith.constant 3 : i64
// CHECK: llvm.store %[[TWO]],
// CHECK: llvm.store %[[ONE]],
// CHECK-NEXT: return
func.func @drop(%out: memref<1xi64>) {
  %s0 = forth.stack !forth.stack
  %s1 = forth.constant %s0(1 : i64) : !forth.stack -> !forth.stack
  %s2 = forth.constant %s1(2 : i64) : !forth.stack -> !forth.stack
  %s3 = forth.constant %s2(3 : i64) : !forth.stack -> !forth.stack
  %s4 = forth.drop %s3 : !forth.stack -> !forth.stack
  %s5 = forth.param_ref %s4 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s6 = forth.storei %s5 : !forth.stack -> !forth.stack
  %s7 = forth.param_ref %s6 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s8 = forth.storei %s7 : !forth.stack -> !forth.stack
  return
}

// ( 1 2 3 -- 1 3 2 )
// CHECK-LABEL: func.func @swap(
// CHECK-NOT: memref.alloca
// CHECK: %[[ONE:.*]] = arith.constant 1 : i64
// CHECK: %[[TWO:.*]] = arith.constant 2 : i64
// CHECK: %[[THREE:.*]] = arith.constant 3 : i64
// CHECK: llvm.store %[[TWO]],
// CHECK: llvm.store %[[THREE]],
// CHECK: llvm.store %[[ONE]],
// CHECK-NEXT: return
func.func @swap(%out: memref<1xi64>) {
  %s0 = forth.stack !forth.stack
  %s1 = forth.constant %s0(1 : i64) : !forth.stack -> !forth.stack
  %s2 = forth.constant %s1(2 : i64) : !forth.stack -> !forth.stack
  %s3 = forth.constant %s2(3 : i64) : !forth.stack -> !forth.stack
  %s4 = forth.swap %s3 : !forth.stack -> !forth.stack
  %s5 = forth.param_ref %s4 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s6 = forth.storei %s5 : !forth.stack -> !forth.stack
  %s7 = forth.param_ref %s6 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s8 = forth.storei %s7 : !forth.stack -> !forth.stack
  %s9 = forth.param_ref %s8 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s10 = forth.storei %s9 : !forth.stack -> !forth.stack
  return
}

// ( 1 2 3 -- 1 2 3 2 )
// CHECK-LABEL: func.func @over(
// CHECK-NOT: memref.alloca
// CHECK: %[[ONE:.*]] = arith.constant 1 : i64
// CHECK: %[[TWO:.*]] = arith.constant 2 : i64
// CHECK: %[[THREE:.*]] = arith.constant 3 : i64
// CHECK: llvm.store %[[TWO]],
// CHECK: llvm.store %[[THREE]],
// CHECK: llvm.store %[[TWO]],
// CHECK: llvm.store %[[ONE]],
// CHECK-NEXT: return
func.func @over(%out: memref<1xi64>) {
  %s0 = forth.stack !forth.stack
  %s1 = forth.constant %s0(1 : i64) : !forth.stack -> !forth.stack
  %s2 = forth.constant %s1(2 : i64) : !forth.stack -> !forth.stack
  %s3 = forth.constant %s2(3 : i64) : !forth.stack -> !forth.stack
  %s4 = forth.over %s3 : !forth.stack -> !forth.stack
  %s5 = forth.param_ref %s4 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s6 = forth.storei %s5 : !forth.stack -> !forth.stack
  %s7 = forth.param_ref %s6 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s8 = forth.storei %s7 : !forth.stack -> !forth.stack
  %s9 = forth.param_ref %s8 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s10 = forth.storei %s9 : !forth.stack -> !forth.stack
  %s11 = forth.param_ref %s10 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s12 = forth.storei %s11 : !forth.stack -> !forth.stack
  return
}

// ( 1 2 3 -- 2 3 1 )
// CHECK-LABEL: func.func @rot(
// CHECK-NOT: memref.alloca
// CHECK: %[[ONE:.*]] = arith.constant 1 : i64
// CHECK: %[[TWO:.*]] = arith.constant 2 : i64
// CHECK: %[[THREE:.*]] = arith.constant 3 : i64
// CHECK: llvm.store %[[ONE]],
// CHECK: llvm.store %[[THREE]],
// CHECK: llvm.store %[[TWO]],
// CHECK-NEXT: return
func.func @rot(%out: memref<1xi64>) {
  %s0 = forth.stack !forth.stack
  %s1 = forth.constant %s0(1 : i64) : !forth.stack -> !forth.stack
  %s2 = forth.constant %s1(2 : i64) : !forth.stack -> !forth.stack
  %s3 = forth.constant %s2(3 : i64) : !forth.stack -> !forth.stack
  %s4 = forth.rot %s3 : !forth.stack -> !forth.stack
  %s5 = forth.param_ref %s4 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s6 = forth.storei %s5 : !forth.stack -> !forth.stack
  %s7 = forth.param_ref %s6 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s8 = forth.storei %s7 : !forth.stack -> !forth.stack
  %s9 = forth.param_ref %s8 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s10 = forth.storei %s9 : !forth.stack -> !forth.stack
  return
}

// ( 1 2 3 -- 1 3 )
// CHECK-LABEL: func.func @nip(
// CHECK-NOT: memref.alloca
// CHECK: %[[ONE:.*]] = arith.constant 1 : i64
// CHECK: %[[TWO:.*]] = arith.constant 2 : i64
// CHECK: %[[THREE:.*]] = arith.constant 3 : i64
// CHECK: llvm.store %[[THREE]],
// CHECK: llvm.store %[[ONE]],
// CHECK-NEXT: return
func.func @nip(%out: memref<1xi64>) {
  %s0 = forth.stack !forth.stack
  %s1 = forth.constant %s0(1 : i64) : !forth.stack -> !forth.stack
  %s2 = forth.constant %s1(2 : i64) : !forth.stack -> !forth.stack
  %s3 = forth.constant %s2(3 : i64) : !forth.stack -> !forth.stack
  %s4 = forth.nip %s3 : !forth.stack -> !forth.stack
  %s5 = forth.param_ref %s4 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s6 = forth.storei %s5 : !forth.stack -> !forth.stack
  %s7 = forth.param_ref %s6 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s8 = forth.storei %s7 : !forth.stack -> !forth.stack
  return
}

// ( 1 2 3 -- 1 3 2 3 )
// CHECK-LABEL: func.func @tuck(
// CHECK-NOT: memref.alloca
// CHECK: %[[ONE:.*]] = arith.constant 1 : i64
// CHECK: %[[TWO:.*]] = arith.constant 2 : i64
// CHECK: %[[THREE:.*]] = arith.constant 3 : i64
// CHECK: llvm.store %[[THREE]],
// CHECK: llvm.store %[[TWO]],
// CHECK: llvm.store %[[THREE]],
// CHECK: llvm.store %[[ONE]],
// CHECK-NEXT: return
func.func @tuck(%out: memref<1xi64>) {
  %s0 = forth.stack !forth.stack
  %s1 = forth.constant %s0(1 : i64) : !forth.stack -> !forth.stack
  %s2 = forth.constant %s1(2 : i64) : !forth.stack -> !forth.stack
  %s3 = forth.constant %s2(3 : i64) : !forth.stack -> !forth.stack
  %s4 = forth.tuck %s3 : !forth.stack -> !forth.stack
  %s5 = forth.param_ref %s4 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s6 = forth.storei %s5 : !forth.stack -> !forth.stack
  %s7 = forth.param_ref %s6 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s8 = forth.storei %s7 : !forth.stack -> !forth.stack
  %s9 = forth.param_ref %s8 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s10 = forth.storei %s9 : !forth.stack -> !forth.stack
  %s11 = forth.param_ref %s10 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s12 = forth.storei %s11 : !forth.stack -> !forth.stack
  return
}
