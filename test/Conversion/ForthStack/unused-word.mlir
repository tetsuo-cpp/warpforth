// RUN: %warpforth-opt --lower-forth-stack %s | %FileCheck %s

// CHECK-NOT: func.func private
// CHECK-LABEL: func.func @main(
// CHECK: %[[FIVE:.*]] = arith.constant 5 : i64
// CHECK: llvm.store %[[FIVE]], %{{.*}} : i64, !llvm.ptr<1>

module {
  func.func private @a(%s: !forth.stack) -> !forth.stack {
    %0 = forth.dup %s : !forth.stack -> !forth.stack
    return %0 : !forth.stack
  }
  func.func private @b(%s: !forth.stack) -> !forth.stack {
    %0 = func.call @a(%s) : (!forth.stack) -> !forth.stack
    return %0 : !forth.stack
  }
  func.func @main(%out: memref<1xi64>) attributes {forth.kernel} {
    %0 = forth.stack !forth.stack
    %1 = forth.constant %0(5 : i64) : !forth.stack -> !forth.stack
    %2 = forth.param_ref %1 %out : !forth.stack, memref<1xi64> -> !forth.stack
    %3 = forth.storei %2 : !forth.stack -> !forth.stack
    return
  }
}
