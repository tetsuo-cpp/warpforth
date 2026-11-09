// RUN: %warpforth-opt --convert-forth-to-memref %s | %FileCheck %s

// CHECK-LABEL: func.func private @main
// CHECK: memref.alloca() : memref<256xi64>

module {
  func.func private @main() {
    %0 = forth.stack !forth.stack
    %1 = forth.constant %0(1 : i64) : !forth.stack -> !forth.stack
    %stack, %flag = forth.pop_flag %1 : !forth.stack -> !forth.stack, i1
    cf.cond_br %flag, ^then(%stack : !forth.stack), ^join(%stack : !forth.stack)
  ^then(%2: !forth.stack):
    %3 = forth.constant %2(42 : i64) : !forth.stack -> !forth.stack
    cf.br ^join(%3 : !forth.stack)
  ^join(%4: !forth.stack):
    return
  }
}
