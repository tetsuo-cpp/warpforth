// RUN: %warpforth-opt --convert-forth-to-memref %s | %FileCheck %s

// CHECK-LABEL: func.func private @shuffle_xor
// CHECK: %[[MEM:.*]] = memref.alloca() : memref<256xi64>
// CHECK: %[[B:.*]] = memref.load %[[MEM]][%[[SP:.*]]] : memref<256xi64>
// CHECK: %[[SPM1:.*]] = arith.subi %[[SP]], %{{.*}} : index
// CHECK: %[[X:.*]] = memref.load %[[MEM]][%[[SPM1]]] : memref<256xi64>
// CHECK: %[[B32:.*]] = arith.trunci %[[B]] : i64 to i32
// CHECK: %[[MASK:.*]] = arith.constant 31 : i32
// CHECK: %[[BMOD:.*]] = arith.andi %[[B32]], %[[MASK]] : i32
// CHECK: %[[W:.*]] = arith.constant 32 : i32
// CHECK: %[[Y:.*]], %{{.*}} = gpu.shuffle xor %[[X]], %[[BMOD]], %[[W]] : i64
// CHECK: memref.store %[[Y]], %[[MEM]][%[[SPM1]]] : memref<256xi64>
// CHECK: %[[LANE:.*]] = forth.intrinsic "lane-id" : index
// CHECK: %[[LANE64:.*]] = arith.index_cast %[[LANE]] : index to i64
// CHECK: %[[SP2:.*]] = arith.addi %[[SPM1]], %{{.*}} : index
// CHECK: memref.store %[[LANE64]], %[[MEM]][%[[SP2]]] : memref<256xi64>
func.func private @shuffle_xor() {
  %0 = forth.stack !forth.stack
  %1 = forth.constant %0(7 : i64) : !forth.stack -> !forth.stack
  %2 = forth.constant %1(3 : i64) : !forth.stack -> !forth.stack
  %3 = forth.shuffle xor %2 : !forth.stack -> !forth.stack
  %4 = forth.lane_id %3 : !forth.stack -> !forth.stack
  return
}

// CHECK-LABEL: func.func private @shuffle_modes
// CHECK: gpu.shuffle down
// CHECK: gpu.shuffle up
// CHECK: gpu.shuffle idx
func.func private @shuffle_modes() {
  %0 = forth.stack !forth.stack
  %1 = forth.constant %0(7 : i64) : !forth.stack -> !forth.stack
  %2 = forth.constant %1(1 : i64) : !forth.stack -> !forth.stack
  %3 = forth.shuffle down %2 : !forth.stack -> !forth.stack
  %4 = forth.constant %3(1 : i64) : !forth.stack -> !forth.stack
  %5 = forth.shuffle up %4 : !forth.stack -> !forth.stack
  %6 = forth.constant %5(0 : i64) : !forth.stack -> !forth.stack
  %7 = forth.shuffle idx %6 : !forth.stack -> !forth.stack
  return
}
