// RUN: %warpforth-opt --lower-forth-stack %s | %FileCheck %s

// `A B op OUT !` for each binary bitwise word. RSHIFT is a logical shift.
// `A NOT` flips every bit of A.
// CHECK-LABEL: func.func @bitwise(
// CHECK-SAME: %[[A:[^:]*]]: i64, %[[B:[^:]*]]: i64
// CHECK: %[[AND:.*]] = arith.andi %[[A]], %[[B]] : i64
// CHECK: llvm.store %[[AND]],
// CHECK: %[[OR:.*]] = arith.ori %[[A]], %[[B]] : i64
// CHECK: llvm.store %[[OR]],
// CHECK: %[[XOR:.*]] = arith.xori %[[A]], %[[B]] : i64
// CHECK: llvm.store %[[XOR]],
// CHECK: %[[SHL:.*]] = arith.shli %[[A]], %[[B]] : i64
// CHECK: llvm.store %[[SHL]],
// CHECK: %[[SHR:.*]] = arith.shrui %[[A]], %[[B]] : i64
// CHECK: llvm.store %[[SHR]],
// CHECK: %[[ONES:.*]] = arith.constant -1 : i64
// CHECK: %[[NOT:.*]] = arith.xori %[[A]], %[[ONES]] : i64
// CHECK: llvm.store %[[NOT]],
func.func @bitwise(%a: i64, %b: i64, %out: memref<1xi64>) {
  %s0 = forth.stack !forth.stack
  %s1 = forth.param_ref %s0 %a : !forth.stack, i64 -> !forth.stack
  %s2 = forth.param_ref %s1 %b : !forth.stack, i64 -> !forth.stack
  %s3 = forth.and %s2 : !forth.stack -> !forth.stack
  %s4 = forth.param_ref %s3 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s5 = forth.storei %s4 : !forth.stack -> !forth.stack
  %s6 = forth.param_ref %s5 %a : !forth.stack, i64 -> !forth.stack
  %s7 = forth.param_ref %s6 %b : !forth.stack, i64 -> !forth.stack
  %s8 = forth.or %s7 : !forth.stack -> !forth.stack
  %s9 = forth.param_ref %s8 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s10 = forth.storei %s9 : !forth.stack -> !forth.stack
  %s11 = forth.param_ref %s10 %a : !forth.stack, i64 -> !forth.stack
  %s12 = forth.param_ref %s11 %b : !forth.stack, i64 -> !forth.stack
  %s13 = forth.xor %s12 : !forth.stack -> !forth.stack
  %s14 = forth.param_ref %s13 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s15 = forth.storei %s14 : !forth.stack -> !forth.stack
  %s16 = forth.param_ref %s15 %a : !forth.stack, i64 -> !forth.stack
  %s17 = forth.param_ref %s16 %b : !forth.stack, i64 -> !forth.stack
  %s18 = forth.lshift %s17 : !forth.stack -> !forth.stack
  %s19 = forth.param_ref %s18 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s20 = forth.storei %s19 : !forth.stack -> !forth.stack
  %s21 = forth.param_ref %s20 %a : !forth.stack, i64 -> !forth.stack
  %s22 = forth.param_ref %s21 %b : !forth.stack, i64 -> !forth.stack
  %s23 = forth.rshift %s22 : !forth.stack -> !forth.stack
  %s24 = forth.param_ref %s23 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s25 = forth.storei %s24 : !forth.stack -> !forth.stack
  %s26 = forth.param_ref %s25 %a : !forth.stack, i64 -> !forth.stack
  %s27 = forth.not %s26 : !forth.stack -> !forth.stack
  %s28 = forth.param_ref %s27 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s29 = forth.storei %s28 : !forth.stack -> !forth.stack
  return
}
