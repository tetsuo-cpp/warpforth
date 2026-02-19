// RUN: %warpforth-opt --convert-forth-to-memref %s | %FileCheck %s

// CHECK-LABEL: func.func private @main

// dup: load top, increment SP, store copy
// CHECK: memref.load %{{.*}}[%{{.*}}] : memref<256xi64>
// CHECK: arith.addi
// CHECK: memref.store

// drop: decrement SP
// CHECK: arith.subi %{{.*}}, %{{.*}} : index

// swap: load top and second, store crossed
// CHECK: %[[TOP_VAL:.*]] = memref.load %{{.*}}[%[[SWAP_SP:.*]]] : memref<256xi64>
// CHECK: %[[SWAP_SP1:.*]] = arith.subi
// CHECK: %[[SEC_VAL:.*]] = memref.load %{{.*}}[%[[SWAP_SP1]]] : memref<256xi64>
// CHECK: memref.store %[[SEC_VAL]], %{{.*}}[%[[SWAP_SP]]] : memref<256xi64>
// CHECK: memref.store %[[TOP_VAL]], %{{.*}}[%[[SWAP_SP1]]] : memref<256xi64>

// over: load second element, increment SP, store copy
// CHECK: arith.subi
// CHECK: memref.load %{{.*}}[%{{.*}}] : memref<256xi64>
// CHECK: arith.addi
// CHECK: memref.store

// rot: load three values, store rotated (a b c -- b c a)
// CHECK: %[[ROT_C:.*]] = memref.load %{{.*}}[%[[ROT_SP:.*]]] : memref<256xi64>
// CHECK: %[[ROT_SP1:.*]] = arith.subi %[[ROT_SP]]
// CHECK: %[[ROT_B:.*]] = memref.load %{{.*}}[%[[ROT_SP1]]] : memref<256xi64>
// CHECK: %[[ROT_SP2:.*]] = arith.subi %[[ROT_SP]]
// CHECK: %[[ROT_A:.*]] = memref.load %{{.*}}[%[[ROT_SP2]]] : memref<256xi64>
// CHECK: memref.store %[[ROT_B]], %{{.*}}[%[[ROT_SP2]]] : memref<256xi64>
// CHECK: memref.store %[[ROT_C]], %{{.*}}[%[[ROT_SP1]]] : memref<256xi64>
// CHECK: memref.store %[[ROT_A]], %{{.*}}[%[[ROT_SP]]] : memref<256xi64>

// nip: load top, subi SP, store at SP-1 (a b -- b)
// CHECK: %[[NIP_B:.*]] = memref.load %{{.*}}[%{{.*}}] : memref<256xi64>
// CHECK: %[[NIP_SP1:.*]] = arith.subi
// CHECK: memref.store %[[NIP_B]], %{{.*}}[%[[NIP_SP1]]] : memref<256xi64>

// tuck: load b and a, store b/a/b (a b -- b a b)
// CHECK: %[[TUCK_B:.*]] = memref.load %{{.*}}[%[[TUCK_SP:.*]]] : memref<256xi64>
// CHECK: %[[TUCK_SP1:.*]] = arith.subi
// CHECK: %[[TUCK_A:.*]] = memref.load %{{.*}}[%[[TUCK_SP1]]] : memref<256xi64>
// CHECK: memref.store %[[TUCK_B]], %{{.*}}[%[[TUCK_SP1]]] : memref<256xi64>
// CHECK: memref.store %[[TUCK_A]], %{{.*}}[%[[TUCK_SP]]] : memref<256xi64>
// CHECK: %[[TUCK_NSP:.*]] = arith.addi
// CHECK: memref.store %[[TUCK_B]], %{{.*}}[%[[TUCK_NSP]]] : memref<256xi64>

// pick: load n, index_cast, subi (dynamic), load picked, store
// CHECK: %[[PICK_N:.*]] = memref.load %{{.*}}[%{{.*}}] : memref<256xi64>
// CHECK: %[[PICK_SP1:.*]] = arith.subi
// CHECK: %[[PICK_NIDX:.*]] = arith.index_cast %[[PICK_N]]
// CHECK: %[[PICK_ADDR:.*]] = arith.subi %[[PICK_SP1]], %[[PICK_NIDX]]
// CHECK: %[[PICK_VAL:.*]] = memref.load %{{.*}}[%[[PICK_ADDR]]] : memref<256xi64>
// CHECK: memref.store %[[PICK_VAL]]

// roll: load n, index_cast, subi (dynamic), load saved, cf loop with load/store, store saved
// CHECK: %[[ROLL_N:.*]] = memref.load %{{.*}}[%{{.*}}] : memref<256xi64>
// CHECK: %[[ROLL_SP1:.*]] = arith.subi
// CHECK: %[[ROLL_NIDX:.*]] = arith.index_cast %[[ROLL_N]]
// CHECK: %[[ROLL_ADDR:.*]] = arith.subi %[[ROLL_SP1]], %[[ROLL_NIDX]]
// CHECK: %[[ROLL_SAVED:.*]] = memref.load %{{.*}}[%[[ROLL_ADDR]]] : memref<256xi64>
// CHECK: cf.br ^[[ROLL_HDR:.*]](%[[ROLL_ADDR]] : index)
// CHECK: ^[[ROLL_HDR]](%[[ROLL_IV:.*]]: index):
// CHECK: %[[ROLL_CMP:.*]] = arith.cmpi slt, %[[ROLL_IV]], %[[ROLL_SP1]] : index
// CHECK: cf.cond_br %[[ROLL_CMP]], ^[[ROLL_BODY:.*]](%[[ROLL_IV]] : index), ^[[ROLL_EXIT:.*]]
// CHECK: ^[[ROLL_BODY]](%[[ROLL_BIV:.*]]: index):
// CHECK: %[[ROLL_NEXT:.*]] = arith.addi %[[ROLL_BIV]]
// CHECK: %[[ROLL_SHIFTED:.*]] = memref.load %{{.*}}[%[[ROLL_NEXT]]] : memref<256xi64>
// CHECK: memref.store %[[ROLL_SHIFTED]], %{{.*}}[%[[ROLL_BIV]]] : memref<256xi64>
// CHECK: cf.br ^[[ROLL_HDR]](%[[ROLL_NEXT]] : index)
// CHECK: ^[[ROLL_EXIT]]:
// CHECK: memref.store %[[ROLL_SAVED]], %{{.*}}[%[[ROLL_SP1]]] : memref<256xi64>

module {
  func.func private @main() {
    %0 = forth.stack !forth.stack
    %1 = forth.literal %0 1 : !forth.stack -> !forth.stack
    %2 = forth.literal %1 2 : !forth.stack -> !forth.stack
    %3 = forth.literal %2 3 : !forth.stack -> !forth.stack
    %4 = forth.dup %3 : !forth.stack -> !forth.stack
    %5 = forth.drop %4 : !forth.stack -> !forth.stack
    %6 = forth.swap %5 : !forth.stack -> !forth.stack
    %7 = forth.over %6 : !forth.stack -> !forth.stack
    %8 = forth.rot %7 : !forth.stack -> !forth.stack
    %9 = forth.nip %8 : !forth.stack -> !forth.stack
    %10 = forth.tuck %9 : !forth.stack -> !forth.stack
    %11 = forth.literal %10 2 : !forth.stack -> !forth.stack
    %12 = forth.pick %11 : !forth.stack -> !forth.stack
    %13 = forth.literal %12 2 : !forth.stack -> !forth.stack
    %14 = forth.roll %13 : !forth.stack -> !forth.stack
    return
  }
}
