// RUN: %warpforth-opt --canonicalize %s | %FileCheck %s

// Memory operations must survive canonicalization when their stack result is
// unused. The dead addi also verifies that ordinary stack operations remain
// pure by default.

// CHECK-LABEL: func.func private @global_memory_effects
// CHECK-NOT: forth.addi
// CHECK: forth.loadi
// CHECK: forth.storei
// CHECK: return
func.func private @global_memory_effects() {
  %stack = forth.stack !forth.stack
  %address = forth.constant %stack(1 : i64) : !forth.stack -> !forth.stack
  %dead = forth.addi %address : !forth.stack -> !forth.stack
  %loaded = forth.loadi %address : !forth.stack -> !forth.stack
  %value = forth.constant %loaded(42 : i64) : !forth.stack -> !forth.stack
  %store_address = forth.constant %value(2 : i64) : !forth.stack -> !forth.stack
  %stored = forth.storei %store_address : !forth.stack -> !forth.stack
  return
}

// CHECK-LABEL: func.func private @shared_reduced_width_effects
// CHECK: forth.shared_load_f32
// CHECK: forth.shared_store_i8
// CHECK: return
func.func private @shared_reduced_width_effects() {
  %stack = forth.stack !forth.stack
  %address = forth.constant %stack(3 : i64) : !forth.stack -> !forth.stack
  %loaded = forth.shared_load_f32 %address : !forth.stack -> !forth.stack
  %value = forth.constant %loaded(7 : i64) : !forth.stack -> !forth.stack
  %store_address = forth.constant %value(4 : i64) : !forth.stack -> !forth.stack
  %stored = forth.shared_store_i8 %store_address : !forth.stack -> !forth.stack
  return
}
