// RUN: %warpforth-opt --lower-forth-stack %s | %FileCheck %s

// `A B op OUT !` for each integer comparison. The deeper cell is the left
// operand, and the flag is the sign-extended i1, so true is -1.
// CHECK-LABEL: func.func @int_comparisons(
// CHECK-SAME: %[[A:[^:]*]]: i64, %[[B:[^:]*]]: i64
// CHECK: %[[EQ:.*]] = arith.cmpi eq, %[[A]], %[[B]] : i64
// CHECK: %[[EQ_FLAG:.*]] = arith.extsi %[[EQ]] : i1 to i64
// CHECK: llvm.store %[[EQ_FLAG]],
// CHECK: %[[LT:.*]] = arith.cmpi slt, %[[A]], %[[B]] : i64
// CHECK: %[[LT_FLAG:.*]] = arith.extsi %[[LT]] : i1 to i64
// CHECK: llvm.store %[[LT_FLAG]],
// CHECK: %[[GT:.*]] = arith.cmpi sgt, %[[A]], %[[B]] : i64
// CHECK: %[[GT_FLAG:.*]] = arith.extsi %[[GT]] : i1 to i64
// CHECK: llvm.store %[[GT_FLAG]],
// CHECK: %[[NE:.*]] = arith.cmpi ne, %[[A]], %[[B]] : i64
// CHECK: %[[NE_FLAG:.*]] = arith.extsi %[[NE]] : i1 to i64
// CHECK: llvm.store %[[NE_FLAG]],
// CHECK: %[[LE:.*]] = arith.cmpi sle, %[[A]], %[[B]] : i64
// CHECK: %[[LE_FLAG:.*]] = arith.extsi %[[LE]] : i1 to i64
// CHECK: llvm.store %[[LE_FLAG]],
// CHECK: %[[GE:.*]] = arith.cmpi sge, %[[A]], %[[B]] : i64
// CHECK: %[[GE_FLAG:.*]] = arith.extsi %[[GE]] : i1 to i64
// CHECK: llvm.store %[[GE_FLAG]],
// `A 0= OUT !` compares A with zero.
// CHECK: %[[ZERO:.*]] = arith.constant 0 : i64
// CHECK: %[[IS_ZERO:.*]] = arith.cmpi eq, %[[A]], %[[ZERO]] : i64
// CHECK: %[[ZERO_FLAG:.*]] = arith.extsi %[[IS_ZERO]] : i1 to i64
// CHECK: llvm.store %[[ZERO_FLAG]],
func.func @int_comparisons(%a: i64, %b: i64, %out: memref<1xi64>) {
  %s0 = forth.stack !forth.stack
  %s1 = forth.param_ref %s0 %a : !forth.stack, i64 -> !forth.stack
  %s2 = forth.param_ref %s1 %b : !forth.stack, i64 -> !forth.stack
  %s3 = forth.eqi %s2 : !forth.stack -> !forth.stack
  %s4 = forth.param_ref %s3 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s5 = forth.storei %s4 : !forth.stack -> !forth.stack
  %s6 = forth.param_ref %s5 %a : !forth.stack, i64 -> !forth.stack
  %s7 = forth.param_ref %s6 %b : !forth.stack, i64 -> !forth.stack
  %s8 = forth.lti %s7 : !forth.stack -> !forth.stack
  %s9 = forth.param_ref %s8 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s10 = forth.storei %s9 : !forth.stack -> !forth.stack
  %s11 = forth.param_ref %s10 %a : !forth.stack, i64 -> !forth.stack
  %s12 = forth.param_ref %s11 %b : !forth.stack, i64 -> !forth.stack
  %s13 = forth.gti %s12 : !forth.stack -> !forth.stack
  %s14 = forth.param_ref %s13 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s15 = forth.storei %s14 : !forth.stack -> !forth.stack
  %s16 = forth.param_ref %s15 %a : !forth.stack, i64 -> !forth.stack
  %s17 = forth.param_ref %s16 %b : !forth.stack, i64 -> !forth.stack
  %s18 = forth.nei %s17 : !forth.stack -> !forth.stack
  %s19 = forth.param_ref %s18 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s20 = forth.storei %s19 : !forth.stack -> !forth.stack
  %s21 = forth.param_ref %s20 %a : !forth.stack, i64 -> !forth.stack
  %s22 = forth.param_ref %s21 %b : !forth.stack, i64 -> !forth.stack
  %s23 = forth.lei %s22 : !forth.stack -> !forth.stack
  %s24 = forth.param_ref %s23 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s25 = forth.storei %s24 : !forth.stack -> !forth.stack
  %s26 = forth.param_ref %s25 %a : !forth.stack, i64 -> !forth.stack
  %s27 = forth.param_ref %s26 %b : !forth.stack, i64 -> !forth.stack
  %s28 = forth.gei %s27 : !forth.stack -> !forth.stack
  %s29 = forth.param_ref %s28 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s30 = forth.storei %s29 : !forth.stack -> !forth.stack
  %s31 = forth.param_ref %s30 %a : !forth.stack, i64 -> !forth.stack
  %s32 = forth.zero_eq %s31 : !forth.stack -> !forth.stack
  %s33 = forth.param_ref %s32 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s34 = forth.storei %s33 : !forth.stack -> !forth.stack
  return
}
