// RUN: %warpforth-opt --tag-forth-stack-depth --convert-forth-to-memref %s | %FileCheck %s

// CHECK-LABEL: func.func private @main
// CHECK-NOT: memref<256xi64>
// CHECK: %[[TWO:.*]] = arith.constant 2 : i64
// CHECK: %[[THREE:.*]] = arith.constant 3 : i64
// CHECK: %[[SUM:.*]] = arith.addi %[[TWO]], %[[THREE]] : i64
// CHECK: %[[ADDR:.*]] = arith.constant 100 : i64
// CHECK: %[[PTR:.*]] = llvm.inttoptr %[[ADDR]] : i64 to !llvm.ptr<1>
// CHECK: llvm.store %[[SUM]], %[[PTR]] : i64, !llvm.ptr<1>
// CHECK-NOT: memref<256xi64>

module {
  func.func private @main() {
    %0 = forth.stack !forth.stack
    %1 = forth.constant %0(2 : i64) : !forth.stack -> !forth.stack
    %2 = forth.constant %1(3 : i64) : !forth.stack -> !forth.stack
    %3 = forth.addi %2 : !forth.stack -> !forth.stack
    %4 = forth.constant %3(100 : i64) : !forth.stack -> !forth.stack
    %5 = forth.storei %4 : !forth.stack -> !forth.stack
    return
  }
}
