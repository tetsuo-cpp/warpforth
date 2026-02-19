// RUN: %warpforth-opt --convert-forth-to-memref %s | %FileCheck %s
// CHECK-LABEL: func.func private @main
// CHECK: forth.barrier
func.func private @main() {
  %0 = forth.stack !forth.stack
  forth.barrier
  forth.drop %0 : !forth.stack -> !forth.stack
  return
}
