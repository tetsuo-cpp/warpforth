// RUN: %warpforth-opt --convert-forth-to-memref %s | %FileCheck %s

// CHECK-LABEL: func.func private @main
// CHECK-NOT: memref<256xi64>
// CHECK: %[[ONE:.*]] = arith.constant 1 : i64
// CHECK: %[[ZERO:.*]] = arith.constant 0 : i64
// CHECK: %[[COND:.*]] = arith.cmpi ne, %[[ONE]], %[[ZERO]] : i64
// CHECK: cf.cond_br %[[COND]], ^[[THEN:bb[0-9]+]], ^[[ELSE:bb[0-9]+]]
// CHECK: ^[[THEN]]:
// CHECK: %[[TAKEN:.*]] = arith.constant 42 : i64
// CHECK: cf.br ^[[JOIN:bb[0-9]+]](%[[TAKEN]] : i64)
// CHECK: ^[[ELSE]]:
// CHECK: %[[OTHER:.*]] = arith.constant 99 : i64
// CHECK: cf.br ^[[JOIN]](%[[OTHER]] : i64)
// CHECK: ^[[JOIN]](%[[CELL:.*]]: i64):
// CHECK: %[[ADDR:.*]] = arith.constant 100 : i64
// CHECK: %[[PTR:.*]] = llvm.inttoptr %[[ADDR]] : i64 to !llvm.ptr<1>
// CHECK: llvm.store %[[CELL]], %[[PTR]] : i64, !llvm.ptr<1>
// CHECK-NOT: memref<256xi64>

module {
  func.func private @main() {
    %0 = forth.stack !forth.stack
    %1 = forth.constant %0(1 : i64) : !forth.stack -> !forth.stack
    %stack, %flag = forth.pop_flag %1 : !forth.stack -> !forth.stack, i1
    cf.cond_br %flag, ^then(%stack : !forth.stack), ^else(%stack : !forth.stack)
  ^then(%2: !forth.stack):
    %3 = forth.constant %2(42 : i64) : !forth.stack -> !forth.stack
    cf.br ^join(%3 : !forth.stack)
  ^else(%4: !forth.stack):
    %5 = forth.constant %4(99 : i64) : !forth.stack -> !forth.stack
    cf.br ^join(%5 : !forth.stack)
  ^join(%6: !forth.stack):
    %7 = forth.constant %6(100 : i64) : !forth.stack -> !forth.stack
    %8 = forth.storei %7 : !forth.stack -> !forth.stack
    return
  }
}
