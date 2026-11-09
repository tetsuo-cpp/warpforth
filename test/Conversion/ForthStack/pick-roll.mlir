// RUN: %warpforth-opt --lower-forth-stack %s | %FileCheck %s

// `10 20 30 2 PICK A !`: a literal index names a register cell, so PICK is
// another use of that cell's value.
// CHECK-LABEL: func.func @literal_pick(
// CHECK-NOT: memref.alloca
// CHECK: %[[TEN:.*]] = arith.constant 10 : i64
// CHECK-NOT: memref.alloca
// CHECK: llvm.store %[[TEN]], %{{.*}} : i64, !llvm.ptr<1>
// CHECK-NEXT: return
func.func @literal_pick(%a: memref<1xi64>) {
  %0 = forth.stack !forth.stack
  %1 = forth.constant %0(10 : i64) : !forth.stack -> !forth.stack
  %2 = forth.constant %1(20 : i64) : !forth.stack -> !forth.stack
  %3 = forth.constant %2(30 : i64) : !forth.stack -> !forth.stack
  %4 = forth.constant %3(2 : i64) : !forth.stack -> !forth.stack
  %5 = forth.pick %4 : !forth.stack -> !forth.stack
  %6 = forth.param_ref %5 %a : !forth.stack, memref<1xi64> -> !forth.stack
  %7 = forth.storei %6 : !forth.stack -> !forth.stack
  return
}

// `10 20 30 2 ROLL A ! B ! C !`: a literal ROLL reorders register cells, so
// the stack becomes 20 30 10 with no memory traffic.
// CHECK-LABEL: func.func @literal_roll(
// CHECK-NOT: memref.alloca
// CHECK: %[[TEN:.*]] = arith.constant 10 : i64
// CHECK: %[[TWENTY:.*]] = arith.constant 20 : i64
// CHECK: %[[THIRTY:.*]] = arith.constant 30 : i64
// CHECK-NOT: memref.alloca
// CHECK: llvm.store %[[TEN]], %{{.*}} : i64, !llvm.ptr<1>
// CHECK: llvm.store %[[THIRTY]], %{{.*}} : i64, !llvm.ptr<1>
// CHECK: llvm.store %[[TWENTY]], %{{.*}} : i64, !llvm.ptr<1>
// CHECK-NEXT: return
func.func @literal_roll(%a: memref<1xi64>, %b: memref<1xi64>, %c: memref<1xi64>) {
  %0 = forth.stack !forth.stack
  %1 = forth.constant %0(10 : i64) : !forth.stack -> !forth.stack
  %2 = forth.constant %1(20 : i64) : !forth.stack -> !forth.stack
  %3 = forth.constant %2(30 : i64) : !forth.stack -> !forth.stack
  %4 = forth.constant %3(2 : i64) : !forth.stack -> !forth.stack
  %5 = forth.roll %4 : !forth.stack -> !forth.stack
  %6 = forth.param_ref %5 %a : !forth.stack, memref<1xi64> -> !forth.stack
  %7 = forth.storei %6 : !forth.stack -> !forth.stack
  %8 = forth.param_ref %7 %b : !forth.stack, memref<1xi64> -> !forth.stack
  %9 = forth.storei %8 : !forth.stack -> !forth.stack
  %10 = forth.param_ref %9 %c : !forth.stack, memref<1xi64> -> !forth.stack
  %11 = forth.storei %10 : !forth.stack -> !forth.stack
  return
}

// `10 20 30 N PICK A ! B !`: a runtime index needs addressable cells, so the
// three cells are stored at indices 1 to 3 as the image for one load at
// 3 - N. The cells stay in registers, so B receives 30 directly.
// CHECK-LABEL: func.func @dynamic_pick(
// CHECK-SAME: %[[N:[^:]*]]: i64
// CHECK: %[[BUFFER:.*]] = memref.alloca() : memref<256xi64>
// CHECK: %[[TEN:.*]] = arith.constant 10 : i64
// CHECK: %[[TWENTY:.*]] = arith.constant 20 : i64
// CHECK: %[[THIRTY:.*]] = arith.constant 30 : i64
// CHECK: %[[EMPTY:.*]] = arith.constant 0 : index
// CHECK: %[[SLOT1:.*]] = arith.addi %[[EMPTY]], %{{.*}} : index
// CHECK: memref.store %[[TEN]], %[[BUFFER]][%[[SLOT1]]]
// CHECK: %[[SLOT2:.*]] = arith.addi %[[SLOT1]], %{{.*}} : index
// CHECK: memref.store %[[TWENTY]], %[[BUFFER]][%[[SLOT2]]]
// CHECK: %[[SLOT3:.*]] = arith.addi %[[SLOT2]], %{{.*}} : index
// CHECK: memref.store %[[THIRTY]], %[[BUFFER]][%[[SLOT3]]]
// CHECK: %[[OFFSET:.*]] = arith.index_cast %[[N]] : i64 to index
// CHECK: %[[INDEX:.*]] = arith.subi %[[SLOT3]], %[[OFFSET]] : index
// CHECK: %[[PICKED:.*]] = memref.load %[[BUFFER]][%[[INDEX]]]
// CHECK: llvm.store %[[PICKED]], %{{.*}} : i64, !llvm.ptr<1>
// CHECK: llvm.store %[[THIRTY]], %{{.*}} : i64, !llvm.ptr<1>
func.func @dynamic_pick(%n: i64, %a: memref<1xi64>, %b: memref<1xi64>) {
  %0 = forth.stack !forth.stack
  %1 = forth.constant %0(10 : i64) : !forth.stack -> !forth.stack
  %2 = forth.constant %1(20 : i64) : !forth.stack -> !forth.stack
  %3 = forth.constant %2(30 : i64) : !forth.stack -> !forth.stack
  %4 = forth.param_ref %3 %n : !forth.stack, i64 -> !forth.stack
  %5 = forth.pick %4 : !forth.stack -> !forth.stack
  %6 = forth.param_ref %5 %a : !forth.stack, memref<1xi64> -> !forth.stack
  %7 = forth.storei %6 : !forth.stack -> !forth.stack
  %8 = forth.param_ref %7 %b : !forth.stack, memref<1xi64> -> !forth.stack
  %9 = forth.storei %8 : !forth.stack -> !forth.stack
  return
}

// `10 20 30 N ROLL A ! B ! C !`: a runtime ROLL stores the cells, saves the
// cell at 3 - N, shifts the cells above it down, stores the saved cell on top,
// and reloads the three cells.
// CHECK-LABEL: func.func @dynamic_roll(
// CHECK-SAME: %[[N:[^:]*]]: i64
// CHECK: %[[BUFFER:.*]] = memref.alloca() : memref<256xi64>
// CHECK: %[[EMPTY:.*]] = arith.constant 0 : index
// CHECK: memref.store
// CHECK: memref.store
// CHECK: %[[TOP:.*]] = arith.addi %{{.*}}, %{{.*}} : index
// CHECK: memref.store %{{.*}}, %[[BUFFER]][%[[TOP]]]
// CHECK: %[[OFFSET:.*]] = arith.index_cast %[[N]] : i64 to index
// CHECK: %[[FROM:.*]] = arith.subi %[[TOP]], %[[OFFSET]] : index
// CHECK: %[[ROLLED:.*]] = memref.load %[[BUFFER]][%[[FROM]]]
// CHECK: cf.br ^[[HEADER:bb[0-9]+]](%[[FROM]] : index)
// CHECK: ^[[HEADER]](%[[IV:.*]]: index):
// CHECK: %[[MORE:.*]] = arith.cmpi slt, %[[IV]], %[[TOP]] : index
// CHECK: cf.cond_br %[[MORE]], ^[[BODY:bb[0-9]+]](%[[IV]] : index), ^[[DONE:bb[0-9]+]]
// CHECK: ^[[BODY]](%[[SLOT:.*]]: index):
// CHECK: %[[NEXT:.*]] = arith.addi %[[SLOT]], %{{.*}} : index
// CHECK: %[[SHIFTED:.*]] = memref.load %[[BUFFER]][%[[NEXT]]]
// CHECK: memref.store %[[SHIFTED]], %[[BUFFER]][%[[SLOT]]]
// CHECK: cf.br ^[[HEADER]](%[[NEXT]] : index)
// CHECK: ^[[DONE]]:
// CHECK: memref.store %[[ROLLED]], %[[BUFFER]][%[[TOP]]]
// CHECK: %[[RELOAD1:.*]] = arith.addi %[[EMPTY]], %{{.*}} : index
// CHECK: %[[BOTTOM:.*]] = memref.load %[[BUFFER]][%[[RELOAD1]]]
// CHECK: %[[MIDDLE:.*]] = memref.load
// CHECK: %[[NEWTOP:.*]] = memref.load
// CHECK: llvm.store %[[NEWTOP]], %{{.*}} : i64, !llvm.ptr<1>
// CHECK: llvm.store %[[MIDDLE]], %{{.*}} : i64, !llvm.ptr<1>
// CHECK: llvm.store %[[BOTTOM]], %{{.*}} : i64, !llvm.ptr<1>
func.func @dynamic_roll(%n: i64, %a: memref<1xi64>, %b: memref<1xi64>, %c: memref<1xi64>) {
  %0 = forth.stack !forth.stack
  %1 = forth.constant %0(10 : i64) : !forth.stack -> !forth.stack
  %2 = forth.constant %1(20 : i64) : !forth.stack -> !forth.stack
  %3 = forth.constant %2(30 : i64) : !forth.stack -> !forth.stack
  %4 = forth.param_ref %3 %n : !forth.stack, i64 -> !forth.stack
  %5 = forth.roll %4 : !forth.stack -> !forth.stack
  %6 = forth.param_ref %5 %a : !forth.stack, memref<1xi64> -> !forth.stack
  %7 = forth.storei %6 : !forth.stack -> !forth.stack
  %8 = forth.param_ref %7 %b : !forth.stack, memref<1xi64> -> !forth.stack
  %9 = forth.storei %8 : !forth.stack -> !forth.stack
  %10 = forth.param_ref %9 %c : !forth.stack, memref<1xi64> -> !forth.stack
  %11 = forth.storei %10 : !forth.stack -> !forth.stack
  return
}
