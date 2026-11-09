// RUN: %warpforth-opt --lower-forth-stack --canonicalize %s | %FileCheck %s

// A float cell holds f64 bits. Every float word bitcasts its cells to f64 and
// its result back to i64, so after canonicalization the round trips are gone
// and each word applies to the f64 parameters directly.

// `X Y op OUT !`: the deeper cell is the left operand.
// CHECK-LABEL: func.func @float_arithmetic(
// CHECK-SAME: %[[X:[^:]*]]: f64, %[[Y:[^:]*]]: f64
// CHECK: %[[SUM:.*]] = arith.addf %[[X]], %[[Y]] : f64
// CHECK: %[[SUM_BITS:.*]] = arith.bitcast %[[SUM]] : f64 to i64
// CHECK: llvm.store %[[SUM_BITS]],
// CHECK: %[[DIFF:.*]] = arith.subf %[[X]], %[[Y]] : f64
// CHECK: %[[DIFF_BITS:.*]] = arith.bitcast %[[DIFF]] : f64 to i64
// CHECK: llvm.store %[[DIFF_BITS]],
// CHECK: %[[PROD:.*]] = arith.mulf %[[X]], %[[Y]] : f64
// CHECK: %[[PROD_BITS:.*]] = arith.bitcast %[[PROD]] : f64 to i64
// CHECK: llvm.store %[[PROD_BITS]],
// CHECK: %[[QUOT:.*]] = arith.divf %[[X]], %[[Y]] : f64
// CHECK: %[[QUOT_BITS:.*]] = arith.bitcast %[[QUOT]] : f64 to i64
// CHECK: llvm.store %[[QUOT_BITS]],
// CHECK: %[[MAX:.*]] = arith.maximumf %[[X]], %[[Y]] : f64
// CHECK: %[[MAX_BITS:.*]] = arith.bitcast %[[MAX]] : f64 to i64
// CHECK: llvm.store %[[MAX_BITS]],
// CHECK: %[[MIN:.*]] = arith.minimumf %[[X]], %[[Y]] : f64
// CHECK: %[[MIN_BITS:.*]] = arith.bitcast %[[MIN]] : f64 to i64
// CHECK: llvm.store %[[MIN_BITS]],
func.func @float_arithmetic(%x: f64, %y: f64, %out: memref<1xi64>) {
  %s0 = forth.stack !forth.stack
  %s1 = forth.param_ref %s0 %x : !forth.stack, f64 -> !forth.stack
  %s2 = forth.param_ref %s1 %y : !forth.stack, f64 -> !forth.stack
  %s3 = forth.addf %s2 : !forth.stack -> !forth.stack
  %s4 = forth.param_ref %s3 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s5 = forth.storei %s4 : !forth.stack -> !forth.stack
  %s6 = forth.param_ref %s5 %x : !forth.stack, f64 -> !forth.stack
  %s7 = forth.param_ref %s6 %y : !forth.stack, f64 -> !forth.stack
  %s8 = forth.subf %s7 : !forth.stack -> !forth.stack
  %s9 = forth.param_ref %s8 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s10 = forth.storei %s9 : !forth.stack -> !forth.stack
  %s11 = forth.param_ref %s10 %x : !forth.stack, f64 -> !forth.stack
  %s12 = forth.param_ref %s11 %y : !forth.stack, f64 -> !forth.stack
  %s13 = forth.mulf %s12 : !forth.stack -> !forth.stack
  %s14 = forth.param_ref %s13 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s15 = forth.storei %s14 : !forth.stack -> !forth.stack
  %s16 = forth.param_ref %s15 %x : !forth.stack, f64 -> !forth.stack
  %s17 = forth.param_ref %s16 %y : !forth.stack, f64 -> !forth.stack
  %s18 = forth.divf %s17 : !forth.stack -> !forth.stack
  %s19 = forth.param_ref %s18 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s20 = forth.storei %s19 : !forth.stack -> !forth.stack
  %s21 = forth.param_ref %s20 %x : !forth.stack, f64 -> !forth.stack
  %s22 = forth.param_ref %s21 %y : !forth.stack, f64 -> !forth.stack
  %s23 = forth.maxf %s22 : !forth.stack -> !forth.stack
  %s24 = forth.param_ref %s23 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s25 = forth.storei %s24 : !forth.stack -> !forth.stack
  %s26 = forth.param_ref %s25 %x : !forth.stack, f64 -> !forth.stack
  %s27 = forth.param_ref %s26 %y : !forth.stack, f64 -> !forth.stack
  %s28 = forth.minf %s27 : !forth.stack -> !forth.stack
  %s29 = forth.param_ref %s28 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s30 = forth.storei %s29 : !forth.stack -> !forth.stack
  return
}

// CHECK-LABEL: func.func @float_math(
// CHECK-SAME: %[[X:[^:]*]]: f64
// CHECK: %[[EXP:.*]] = math.exp %[[X]] : f64
// CHECK: %[[EXP_BITS:.*]] = arith.bitcast %[[EXP]] : f64 to i64
// CHECK: llvm.store %[[EXP_BITS]],
// CHECK: %[[SQRT:.*]] = math.sqrt %[[X]] : f64
// CHECK: %[[SQRT_BITS:.*]] = arith.bitcast %[[SQRT]] : f64 to i64
// CHECK: llvm.store %[[SQRT_BITS]],
// CHECK: %[[LOG:.*]] = math.log %[[X]] : f64
// CHECK: %[[LOG_BITS:.*]] = arith.bitcast %[[LOG]] : f64 to i64
// CHECK: llvm.store %[[LOG_BITS]],
// CHECK: %[[ABS:.*]] = math.absf %[[X]] : f64
// CHECK: %[[ABS_BITS:.*]] = arith.bitcast %[[ABS]] : f64 to i64
// CHECK: llvm.store %[[ABS_BITS]],
// CHECK: %[[NEG:.*]] = arith.negf %[[X]] : f64
// CHECK: %[[NEG_BITS:.*]] = arith.bitcast %[[NEG]] : f64 to i64
// CHECK: llvm.store %[[NEG_BITS]],
func.func @float_math(%x: f64, %out: memref<1xi64>) {
  %s0 = forth.stack !forth.stack
  %s1 = forth.param_ref %s0 %x : !forth.stack, f64 -> !forth.stack
  %s2 = forth.expf %s1 : !forth.stack -> !forth.stack
  %s3 = forth.param_ref %s2 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s4 = forth.storei %s3 : !forth.stack -> !forth.stack
  %s5 = forth.param_ref %s4 %x : !forth.stack, f64 -> !forth.stack
  %s6 = forth.sqrtf %s5 : !forth.stack -> !forth.stack
  %s7 = forth.param_ref %s6 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s8 = forth.storei %s7 : !forth.stack -> !forth.stack
  %s9 = forth.param_ref %s8 %x : !forth.stack, f64 -> !forth.stack
  %s10 = forth.logf %s9 : !forth.stack -> !forth.stack
  %s11 = forth.param_ref %s10 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s12 = forth.storei %s11 : !forth.stack -> !forth.stack
  %s13 = forth.param_ref %s12 %x : !forth.stack, f64 -> !forth.stack
  %s14 = forth.absf %s13 : !forth.stack -> !forth.stack
  %s15 = forth.param_ref %s14 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s16 = forth.storei %s15 : !forth.stack -> !forth.stack
  %s17 = forth.param_ref %s16 %x : !forth.stack, f64 -> !forth.stack
  %s18 = forth.negf %s17 : !forth.stack -> !forth.stack
  %s19 = forth.param_ref %s18 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s20 = forth.storei %s19 : !forth.stack -> !forth.stack
  return
}

// Float comparisons are ordered, and the flag is the sign-extended i1.
// CHECK-LABEL: func.func @float_comparisons(
// CHECK-SAME: %[[X:[^:]*]]: f64, %[[Y:[^:]*]]: f64
// CHECK: %[[EQ:.*]] = arith.cmpf oeq, %[[X]], %[[Y]] : f64
// CHECK: %[[EQ_FLAG:.*]] = arith.extsi %[[EQ]] : i1 to i64
// CHECK: llvm.store %[[EQ_FLAG]],
// CHECK: %[[LT:.*]] = arith.cmpf olt, %[[X]], %[[Y]] : f64
// CHECK: %[[LT_FLAG:.*]] = arith.extsi %[[LT]] : i1 to i64
// CHECK: llvm.store %[[LT_FLAG]],
// CHECK: %[[GT:.*]] = arith.cmpf ogt, %[[X]], %[[Y]] : f64
// CHECK: %[[GT_FLAG:.*]] = arith.extsi %[[GT]] : i1 to i64
// CHECK: llvm.store %[[GT_FLAG]],
// CHECK: %[[NE:.*]] = arith.cmpf one, %[[X]], %[[Y]] : f64
// CHECK: %[[NE_FLAG:.*]] = arith.extsi %[[NE]] : i1 to i64
// CHECK: llvm.store %[[NE_FLAG]],
// CHECK: %[[LE:.*]] = arith.cmpf ole, %[[X]], %[[Y]] : f64
// CHECK: %[[LE_FLAG:.*]] = arith.extsi %[[LE]] : i1 to i64
// CHECK: llvm.store %[[LE_FLAG]],
// CHECK: %[[GE:.*]] = arith.cmpf oge, %[[X]], %[[Y]] : f64
// CHECK: %[[GE_FLAG:.*]] = arith.extsi %[[GE]] : i1 to i64
// CHECK: llvm.store %[[GE_FLAG]],
func.func @float_comparisons(%x: f64, %y: f64, %out: memref<1xi64>) {
  %s0 = forth.stack !forth.stack
  %s1 = forth.param_ref %s0 %x : !forth.stack, f64 -> !forth.stack
  %s2 = forth.param_ref %s1 %y : !forth.stack, f64 -> !forth.stack
  %s3 = forth.eqf %s2 : !forth.stack -> !forth.stack
  %s4 = forth.param_ref %s3 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s5 = forth.storei %s4 : !forth.stack -> !forth.stack
  %s6 = forth.param_ref %s5 %x : !forth.stack, f64 -> !forth.stack
  %s7 = forth.param_ref %s6 %y : !forth.stack, f64 -> !forth.stack
  %s8 = forth.ltf %s7 : !forth.stack -> !forth.stack
  %s9 = forth.param_ref %s8 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s10 = forth.storei %s9 : !forth.stack -> !forth.stack
  %s11 = forth.param_ref %s10 %x : !forth.stack, f64 -> !forth.stack
  %s12 = forth.param_ref %s11 %y : !forth.stack, f64 -> !forth.stack
  %s13 = forth.gtf %s12 : !forth.stack -> !forth.stack
  %s14 = forth.param_ref %s13 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s15 = forth.storei %s14 : !forth.stack -> !forth.stack
  %s16 = forth.param_ref %s15 %x : !forth.stack, f64 -> !forth.stack
  %s17 = forth.param_ref %s16 %y : !forth.stack, f64 -> !forth.stack
  %s18 = forth.nef %s17 : !forth.stack -> !forth.stack
  %s19 = forth.param_ref %s18 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s20 = forth.storei %s19 : !forth.stack -> !forth.stack
  %s21 = forth.param_ref %s20 %x : !forth.stack, f64 -> !forth.stack
  %s22 = forth.param_ref %s21 %y : !forth.stack, f64 -> !forth.stack
  %s23 = forth.lef %s22 : !forth.stack -> !forth.stack
  %s24 = forth.param_ref %s23 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s25 = forth.storei %s24 : !forth.stack -> !forth.stack
  %s26 = forth.param_ref %s25 %x : !forth.stack, f64 -> !forth.stack
  %s27 = forth.param_ref %s26 %y : !forth.stack, f64 -> !forth.stack
  %s28 = forth.gef %s27 : !forth.stack -> !forth.stack
  %s29 = forth.param_ref %s28 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s30 = forth.storei %s29 : !forth.stack -> !forth.stack
  return
}

// `A S>F OUT !` then `X F>S OUT !`.
// CHECK-LABEL: func.func @conversions(
// CHECK-SAME: %[[A:[^:]*]]: i64, %[[X:[^:]*]]: f64
// CHECK: %[[A_FLOAT:.*]] = arith.sitofp %[[A]] : i64 to f64
// CHECK: %[[A_BITS:.*]] = arith.bitcast %[[A_FLOAT]] : f64 to i64
// CHECK: llvm.store %[[A_BITS]],
// CHECK: %[[X_INT:.*]] = arith.fptosi %[[X]] : f64 to i64
// CHECK: llvm.store %[[X_INT]],
func.func @conversions(%a: i64, %x: f64, %out: memref<1xi64>) {
  %s0 = forth.stack !forth.stack
  %s1 = forth.param_ref %s0 %a : !forth.stack, i64 -> !forth.stack
  %s2 = forth.itof %s1 : !forth.stack -> !forth.stack
  %s3 = forth.param_ref %s2 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s4 = forth.storei %s3 : !forth.stack -> !forth.stack
  %s5 = forth.param_ref %s4 %x : !forth.stack, f64 -> !forth.stack
  %s6 = forth.ftoi %s5 : !forth.stack -> !forth.stack
  %s7 = forth.param_ref %s6 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %s8 = forth.storei %s7 : !forth.stack -> !forth.stack
  return
}
