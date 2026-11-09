// RUN: %warpforth-opt --lower-forth-stack %s | %FileCheck %s

// The stored value is the sum itself, with no stack buffer in between.
// CHECK-LABEL: func.func @add_then_store(
// CHECK-SAME: %[[OUT:[^:]*]]: memref<1xi64>
// CHECK-NOT: memref.alloca
// CHECK: %[[TWO:.*]] = arith.constant 2 : i64
// CHECK: %[[THREE:.*]] = arith.constant 3 : i64
// CHECK: %[[SUM:.*]] = arith.addi %[[TWO]], %[[THREE]] : i64
// CHECK: %[[BASE:.*]] = memref.extract_aligned_pointer_as_index %[[OUT]]
// CHECK: %[[ADDRESS:.*]] = arith.index_cast %[[BASE]] : index to i64
// CHECK: %[[POINTER:.*]] = llvm.inttoptr %[[ADDRESS]] : i64 to !llvm.ptr<1>
// CHECK: llvm.store %[[SUM]], %[[POINTER]] : i64, !llvm.ptr<1>
// CHECK-NEXT: return
func.func @add_then_store(%out: memref<1xi64>) {
  %0 = forth.stack !forth.stack
  %1 = forth.constant %0(2 : i64) : !forth.stack -> !forth.stack
  %2 = forth.constant %1(3 : i64) : !forth.stack -> !forth.stack
  %3 = forth.addi %2 : !forth.stack -> !forth.stack
  %4 = forth.param_ref %3 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %5 = forth.storei %4 : !forth.stack -> !forth.stack
  return
}

// A binary word takes the deeper cell as its left operand, so `A B -` is A - B.
// CHECK-LABEL: func.func @operand_order(
// CHECK-SAME: %[[A:[^:]*]]: i64, %[[B:[^:]*]]: i64
// CHECK: %[[DIFF:.*]] = arith.subi %[[A]], %[[B]] : i64
// CHECK: %[[QUOT:.*]] = arith.divsi %[[DIFF]], %[[B]] : i64
// CHECK: %[[REM:.*]] = arith.remsi %[[QUOT]], %[[A]] : i64
// CHECK: %[[PROD:.*]] = arith.muli %[[REM]], %[[B]] : i64
// CHECK: llvm.store %[[PROD]], %{{.*}} : i64, !llvm.ptr<1>
func.func @operand_order(%a: i64, %b: i64, %out: memref<1xi64>) {
  %0 = forth.stack !forth.stack
  %1 = forth.param_ref %0 %a : !forth.stack, i64 -> !forth.stack
  %2 = forth.param_ref %1 %b : !forth.stack, i64 -> !forth.stack
  %3 = forth.subi %2 : !forth.stack -> !forth.stack
  %4 = forth.param_ref %3 %b : !forth.stack, i64 -> !forth.stack
  %5 = forth.divi %4 : !forth.stack -> !forth.stack
  %6 = forth.param_ref %5 %a : !forth.stack, i64 -> !forth.stack
  %7 = forth.mod %6 : !forth.stack -> !forth.stack
  %8 = forth.param_ref %7 %b : !forth.stack, i64 -> !forth.stack
  %9 = forth.muli %8 : !forth.stack -> !forth.stack
  %10 = forth.param_ref %9 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %11 = forth.storei %10 : !forth.stack -> !forth.stack
  return
}

// A float literal and an f64 parameter are cells holding their f64 bits.
// CHECK-LABEL: func.func @float_cells(
// CHECK-SAME: %[[SCALE:[^:]*]]: f64
// CHECK: %[[LITERAL:.*]] = arith.constant 1.500000e+00 : f64
// CHECK: %[[LITERAL_BITS:.*]] = arith.bitcast %[[LITERAL]] : f64 to i64
// CHECK: llvm.store %[[LITERAL_BITS]], %{{.*}} : i64, !llvm.ptr<1>
// CHECK: %[[SCALE_BITS:.*]] = arith.bitcast %[[SCALE]] : f64 to i64
// CHECK: llvm.store %[[SCALE_BITS]], %{{.*}} : i64, !llvm.ptr<1>
func.func @float_cells(%scale: f64, %out: memref<1xi64>) {
  %0 = forth.stack !forth.stack
  %1 = forth.constant %0(1.500000e+00 : f64) : !forth.stack -> !forth.stack
  %2 = forth.param_ref %1 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %3 = forth.storei %2 : !forth.stack -> !forth.stack
  %4 = forth.param_ref %3 %scale : !forth.stack, f64 -> !forth.stack
  %5 = forth.param_ref %4 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %6 = forth.storei %5 : !forth.stack -> !forth.stack
  return
}
