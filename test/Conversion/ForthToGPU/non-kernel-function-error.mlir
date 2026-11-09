// RUN: %not %warpforth-opt --convert-forth-to-gpu %s 2>&1 | %FileCheck %s

// CHECK: error: @helper is not the kernel; inline every Forth word before --convert-forth-to-gpu

module {
  func.func private @helper(%arg0: memref<256xi64>, %arg1: index) -> (memref<256xi64>, index) {
    return %arg0, %arg1 : memref<256xi64>, index
  }
  func.func private @main() attributes {forth.kernel} {
    return
  }
}
