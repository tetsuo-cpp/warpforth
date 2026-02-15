// RUN: %warpforth-opt --convert-forth-to-memref %s | %FileCheck %s

// CHECK-LABEL: func.func private @main

// Verify scf.while with index iter arg:
// CHECK: scf.while (%{{.*}} = %{{.*}}) : (index) -> index {

// Body: operations + flag pop + condition
// CHECK:   arith.subi
// CHECK:   arith.cmpi eq
// CHECK:   arith.extsi
// CHECK:   memref.load
// CHECK:   arith.subi
// CHECK:   arith.cmpi eq
// CHECK:   scf.condition(%{{.*}}) %{{.*}} : index

// After region: identity yield
// CHECK: } do {
// CHECK:   scf.yield %{{.*}} : index
// CHECK: }

module {
  func.func private @main() {
    %0 = forth.stack !forth.stack
    %1 = forth.literal %0 10 : !forth.stack -> !forth.stack
    %2 = forth.begin_until %1 : !forth.stack -> !forth.stack {
    ^bb0(%arg0: !forth.stack):
      %3 = forth.literal %arg0 1 : !forth.stack -> !forth.stack
      %4 = forth.sub %3 : !forth.stack -> !forth.stack
      %5 = forth.dup %4 : !forth.stack -> !forth.stack
      %6 = forth.zero_eq %5 : !forth.stack -> !forth.stack
      forth.yield %6 : !forth.stack
    }
    return
  }
}
