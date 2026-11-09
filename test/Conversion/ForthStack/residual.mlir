// RUN: %warpforth-opt --lower-forth-stack %s | %FileCheck %s

// `5 + A !`: + needs a second cell that the stack does not have. The pop
// widens the stack to the residual buffer and loads index 0.
// CHECK-LABEL: func.func @underflow(
// CHECK: %[[BUFFER:.*]] = memref.alloca() : memref<256xi64>
// CHECK: %[[FIVE:.*]] = arith.constant 5 : i64
// CHECK: %[[EMPTY:.*]] = arith.constant 0 : index
// CHECK: %[[BELOW:.*]] = memref.load %[[BUFFER]][%[[EMPTY]]]
// CHECK: %[[SUM:.*]] = arith.addi %[[BELOW]], %[[FIVE]] : i64
// CHECK: llvm.store %[[SUM]], %{{.*}} : i64, !llvm.ptr<1>
func.func @underflow(%a: memref<1xi64>) {
  %0 = forth.stack !forth.stack
  %1 = forth.constant %0(5 : i64) : !forth.stack -> !forth.stack
  %2 = forth.addi %1 : !forth.stack -> !forth.stack
  %3 = forth.param_ref %2 %a : !forth.stack, memref<1xi64> -> !forth.stack
  %4 = forth.storei %3 : !forth.stack -> !forth.stack
  return
}

// `7 FLAG IF 1 THEN + A !`: the arms reach THEN with two cells and with one,
// so the join keeps the one top cell both share in a register and takes the
// memory depth as an index. The arm with two cells stores 7 at index 1.
// CHECK-LABEL: func.func @unequal_join(
// CHECK: %[[BUFFER:.*]] = memref.alloca() : memref<256xi64>
// CHECK: %[[SEVEN:.*]] = arith.constant 7 : i64
// CHECK: %[[EMPTY:.*]] = arith.constant 0 : index
// CHECK: cf.cond_br %{{.*}}, ^[[THEN:bb[0-9]+]](%[[SEVEN]] : i64), ^[[JOIN:bb[0-9]+]](%[[SEVEN]], %[[EMPTY]] : i64, index)
// CHECK: ^[[THEN]](%[[KEPT:.*]]: i64):
// CHECK: %[[ONE:.*]] = arith.constant 1 : i64
// CHECK: %[[SLOT:.*]] = arith.addi %{{.*}}, %{{.*}} : index
// CHECK: memref.store %[[KEPT]], %[[BUFFER]][%[[SLOT]]]
// CHECK: cf.br ^[[JOIN]](%[[ONE]], %[[SLOT]] : i64, index)
// CHECK: ^[[JOIN]](%[[TOP:.*]]: i64, %[[DEPTH:.*]]: index):
// CHECK: %[[BELOW:.*]] = memref.load %[[BUFFER]][%[[DEPTH]]]
// CHECK: %[[SUM:.*]] = arith.addi %[[BELOW]], %[[TOP]] : i64
// CHECK: llvm.store %[[SUM]], %{{.*}} : i64, !llvm.ptr<1>
func.func @unequal_join(%flag: i64, %a: memref<1xi64>) {
  %0 = forth.stack !forth.stack
  %1 = forth.constant %0(7 : i64) : !forth.stack -> !forth.stack
  %2 = forth.param_ref %1 %flag : !forth.stack, i64 -> !forth.stack
  %output_stack, %cond = forth.pop_flag %2 : !forth.stack -> !forth.stack, i1
  cf.cond_br %cond, ^bb1(%output_stack : !forth.stack), ^bb2(%output_stack : !forth.stack)
^bb1(%3: !forth.stack):
  %4 = forth.constant %3(1 : i64) : !forth.stack -> !forth.stack
  cf.br ^bb2(%4 : !forth.stack)
^bb2(%5: !forth.stack):
  %6 = forth.addi %5 : !forth.stack -> !forth.stack
  %7 = forth.param_ref %6 %a : !forth.stack, memref<1xi64> -> !forth.stack
  %8 = forth.storei %7 : !forth.stack -> !forth.stack
  return
}
