// RUN: %warpforth-opt --convert-forth-to-memref %s | %FileCheck %s

// An empty stack has no data-stack buffer.
// CHECK-LABEL: func.func private @main
// CHECK-NOT: memref<256xi64>
// CHECK: return

module {
  func.func private @main() {
    %0 = forth.stack !forth.stack
    return
  }
}
