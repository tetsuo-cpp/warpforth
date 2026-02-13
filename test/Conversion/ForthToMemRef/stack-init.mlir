// RUN: %warpforth-opt --convert-forth-to-memref %s | %FileCheck %s

// CHECK-LABEL: func.func private @main
// CHECK: %[[BUF:.*]] = memref.alloca() : memref<256xi64>
// CHECK: %[[SP:.*]] = arith.constant 0 : index

module {
  func.func private @main() {
    %0 = forth.stack !forth.stack
    return
  }
}
