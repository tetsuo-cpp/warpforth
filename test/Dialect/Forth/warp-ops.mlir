// RUN: %warpforth-opt --split-input-file --verify-diagnostics %s | %warpforth-opt | %FileCheck %s

// CHECK-LABEL: func.func private @warp_ops
// CHECK: forth.shuffle down %{{.*}} : !forth.stack -> !forth.stack
// CHECK: forth.shuffle up %{{.*}} : !forth.stack -> !forth.stack
// CHECK: forth.shuffle xor %{{.*}} : !forth.stack -> !forth.stack
// CHECK: forth.shuffle idx %{{.*}} : !forth.stack -> !forth.stack
// CHECK: forth.lane_id %{{.*}} : !forth.stack -> !forth.stack
func.func private @warp_ops(%s: !forth.stack) -> !forth.stack {
  %0 = forth.shuffle down %s : !forth.stack -> !forth.stack
  %1 = forth.shuffle up %0 : !forth.stack -> !forth.stack
  %2 = forth.shuffle xor %1 : !forth.stack -> !forth.stack
  %3 = forth.shuffle idx %2 : !forth.stack -> !forth.stack
  %4 = forth.lane_id %3 : !forth.stack -> !forth.stack
  return %4 : !forth.stack
}

// -----

func.func private @unknown_mode(%s: !forth.stack) -> !forth.stack {
  // expected-error @+1 {{expected string or keyword containing one of the following enum values for attribute 'mode' [down, up, xor, idx]}}
  %0 = forth.shuffle sideways %s : !forth.stack -> !forth.stack
  return %0 : !forth.stack
}
