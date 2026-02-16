// RUN: %warpforth-opt --convert-forth-to-memref %s | %FileCheck %s

// CHECK-LABEL: func.func private @main

// Verify scf.while with index iter arg:
// CHECK: scf.while (%{{.*}} = %{{.*}}) : (index) -> index {

// Condition region: operations + flag pop + condition (ne for WHILE)
// CHECK:   memref.load
// CHECK:   arith.cmpi sgt
// CHECK:   arith.extsi
// CHECK:   memref.load
// CHECK:   arith.subi
// CHECK:   arith.cmpi ne
// CHECK:   scf.condition(%{{.*}}) %{{.*}} : index

// Body region: operations + yield
// CHECK: } do {
// CHECK:   arith.addi
// CHECK:   memref.store
// CHECK:   memref.load
// CHECK:   arith.subi
// CHECK:   scf.yield %{{.*}} : index
// CHECK: }

module {
  func.func private @main() {
    %0 = forth.stack !forth.stack
    %1 = forth.literal %0 10 : !forth.stack -> !forth.stack
    %2 = forth.begin_while_repeat %1 : !forth.stack -> !forth.stack {
    ^bb0(%arg0: !forth.stack):
      %3 = forth.dup %arg0 : !forth.stack -> !forth.stack
      %4 = forth.literal %3 0 : !forth.stack -> !forth.stack
      %5 = forth.gt %4 : !forth.stack -> !forth.stack
      forth.yield %5 while_cond : !forth.stack
    } do {
    ^bb0(%arg1: !forth.stack):
      %6 = forth.literal %arg1 1 : !forth.stack -> !forth.stack
      %7 = forth.sub %6 : !forth.stack -> !forth.stack
      forth.yield %7 : !forth.stack
    }
    return
  }
}
