// RUN: %warpforth-opt --convert-forth-to-memref %s | %FileCheck %s

// CHECK-LABEL: func.func private @main
// CHECK-NOT: memref<256xi64>

// load (@) from global address 1
// CHECK: %[[A1:.*]] = arith.constant 1 : i64
// CHECK: %[[P1:.*]] = llvm.inttoptr %[[A1]] : i64 to !llvm.ptr<1>
// CHECK: llvm.load %[[P1]] : !llvm.ptr<1> -> i64

// store 42 to global address 100
// CHECK: %[[V42:.*]] = arith.constant 42 : i64
// CHECK: %[[A100:.*]] = arith.constant 100 : i64
// CHECK: %[[P100:.*]] = llvm.inttoptr %[[A100]] : i64 to !llvm.ptr<1>
// CHECK: llvm.store %[[V42]], %[[P100]] : i64, !llvm.ptr<1>

// shared load (S@) from address 2
// CHECK: %[[A2:.*]] = arith.constant 2 : i64
// CHECK: %[[P2:.*]] = llvm.inttoptr %[[A2]] : i64 to !llvm.ptr<3>
// CHECK: llvm.load %[[P2]] : !llvm.ptr<3> -> i64

// shared store 9 to address 3
// CHECK: %[[V9:.*]] = arith.constant 9 : i64
// CHECK: %[[A3:.*]] = arith.constant 3 : i64
// CHECK: %[[P3:.*]] = llvm.inttoptr %[[A3]] : i64 to !llvm.ptr<3>
// CHECK: llvm.store %[[V9]], %[[P3]] : i64, !llvm.ptr<3>
// CHECK-NOT: memref<256xi64>

module {
  func.func private @main() {
    %0 = forth.stack !forth.stack
    %1 = forth.constant %0(1 : i64) : !forth.stack -> !forth.stack
    %2 = forth.loadi %1 : !forth.stack -> !forth.stack
    %3 = forth.constant %2(42 : i64) : !forth.stack -> !forth.stack
    %4 = forth.constant %3(100 : i64) : !forth.stack -> !forth.stack
    %5 = forth.storei %4 : !forth.stack -> !forth.stack
    %6 = forth.constant %5(2 : i64) : !forth.stack -> !forth.stack
    %7 = forth.shared_loadi %6 : !forth.stack -> !forth.stack
    %8 = forth.constant %7(9 : i64) : !forth.stack -> !forth.stack
    %9 = forth.constant %8(3 : i64) : !forth.stack -> !forth.stack
    %10 = forth.shared_storei %9 : !forth.stack -> !forth.stack
    return
  }
}
