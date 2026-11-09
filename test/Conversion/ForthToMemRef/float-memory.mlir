// RUN: %warpforth-opt --tag-forth-stack-depth --convert-forth-to-memref %s | %FileCheck %s

// CHECK-LABEL: func.func private @main
// CHECK-NOT: memref<256xi64>

// CHECK: %[[A1000:.*]] = arith.constant 1000 : i64
// CHECK: %[[P1000:.*]] = llvm.inttoptr %[[A1000]] : i64 to !llvm.ptr<1>
// CHECK: %[[LOADED:.*]] = llvm.load %[[P1000]] : !llvm.ptr<1> -> f64
// CHECK: arith.bitcast %[[LOADED]] : f64 to i64

// CHECK: %[[PI:.*]] = arith.constant 3.140000e+00 : f64
// CHECK: %[[BITS:.*]] = arith.bitcast %[[PI]] : f64 to i64
// CHECK: %[[A2000:.*]] = arith.constant 2000 : i64
// CHECK: %[[P2000:.*]] = llvm.inttoptr %[[A2000]] : i64 to !llvm.ptr<1>
// CHECK: %[[FPI:.*]] = arith.bitcast %[[BITS]] : i64 to f64
// CHECK: llvm.store %[[FPI]], %[[P2000]] : f64, !llvm.ptr<1>
// CHECK-NOT: memref<256xi64>

module {
  func.func private @main() {
    %0 = forth.stack !forth.stack
    %1 = forth.constant %0(1000 : i64) : !forth.stack -> !forth.stack
    %2 = forth.loadf %1 : !forth.stack -> !forth.stack
    %3 = forth.constant %2(3.140000e+00 : f64) : !forth.stack -> !forth.stack
    %4 = forth.constant %3(2000 : i64) : !forth.stack -> !forth.stack
    %5 = forth.storef %4 : !forth.stack -> !forth.stack
    return
  }
}
