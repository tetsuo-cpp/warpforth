// RUN: %warpforth-opt --convert-forth-to-memref %s | %FileCheck %s

// === Nested IF: 1 IF 2 IF 3 THEN THEN ===
// CHECK-LABEL: func.func private @TEST__NESTED__IF
// CHECK: arith.constant 1 : i32
// CHECK: arith.extsi %{{.*}} : i32 to i64
// CHECK: memref.store
// CHECK: arith.cmpi ne
// CHECK: cf.cond_br %{{.*}}, ^bb1({{.*}}), ^bb2({{.*}})

// Inner IF: push 2, pop_flag, cond_br
// CHECK: ^bb1(%{{.*}}: memref<256xi64>, %{{.*}}: index):
// CHECK: arith.constant 2 : i32
// CHECK: arith.extsi %{{.*}} : i32 to i64
// CHECK: memref.store
// CHECK: arith.cmpi ne
// CHECK: cf.cond_br %{{.*}}, ^bb3({{.*}}), ^bb4({{.*}})

// Outer merge -> return
// CHECK: ^bb2(%{{.*}}: memref<256xi64>, %{{.*}}: index):
// CHECK: return

// Inner true: push 3
// CHECK: ^bb3(%{{.*}}: memref<256xi64>, %{{.*}}: index):
// CHECK: arith.constant 3 : i32
// CHECK: arith.extsi %{{.*}} : i32 to i64
// CHECK: memref.store
// CHECK: cf.br ^bb4

// Inner merge -> outer merge
// CHECK: ^bb4(%{{.*}}: memref<256xi64>, %{{.*}}: index):
// CHECK: cf.br ^bb2

// === IF inside DO: 10 0 DO I 5 > IF I THEN LOOP ===
// CHECK-LABEL: func.func private @TEST__IF__INSIDE__DO

// DO loop setup: pop start/limit, alloca counter
// CHECK: arith.constant 10 : i32
// CHECK: arith.extsi %{{.*}} : i32 to i64
// CHECK: arith.constant 0 : i32
// CHECK: arith.extsi %{{.*}} : i32 to i64
// CHECK: %[[COUNTER1:.*]] = memref.alloca() : memref<1xi64>
// CHECK: memref.store %{{.*}}, %[[COUNTER1]]
// CHECK: cf.br ^bb1

// DO loop header: check counter < limit
// CHECK: ^bb1(%{{.*}}: memref<256xi64>, %{{.*}}: index):
// CHECK: memref.load %[[COUNTER1]]
// CHECK: arith.cmpi slt
// CHECK: cf.cond_br

// DO loop body: push I, push 5, compare >, pop_flag, cond_br (IF)
// CHECK: ^bb2(%{{.*}}: memref<256xi64>, %{{.*}}: index):
// CHECK: memref.load %[[COUNTER1]]
// CHECK: memref.store
// CHECK: arith.constant 5 : i32
// CHECK: arith.extsi %{{.*}} : i32 to i64
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: arith.cmpi sgt
// CHECK: arith.cmpi ne
// CHECK: cf.cond_br

// DO loop exit -> return
// CHECK: ^bb3(%{{.*}}: memref<256xi64>, %{{.*}}: index):
// CHECK: return

// IF true: push I
// CHECK: ^bb4(%{{.*}}: memref<256xi64>, %{{.*}}: index):
// CHECK: memref.load %[[COUNTER1]]
// CHECK: memref.store
// CHECK: cf.br ^bb5

// IF merge: increment counter, loop back
// CHECK: ^bb5(%{{.*}}: memref<256xi64>, %{{.*}}: index):
// CHECK: memref.load %[[COUNTER1]]
// CHECK: arith.addi
// CHECK: memref.store %{{.*}}, %[[COUNTER1]]
// CHECK: cf.br ^bb1

// === Nested DO with J: 3 0 DO 4 0 DO J I + LOOP LOOP ===
// CHECK-LABEL: func.func private @TEST__NESTED__DO__J

// Outer DO setup
// CHECK: arith.constant 3 : i32
// CHECK: arith.extsi %{{.*}} : i32 to i64
// CHECK: arith.constant 0 : i32
// CHECK: arith.extsi %{{.*}} : i32 to i64
// CHECK: %[[OUTER:.*]] = memref.alloca() : memref<1xi64>
// CHECK: memref.store %{{.*}}, %[[OUTER]]
// CHECK: cf.br ^bb1

// Outer loop header: check outer counter < outer limit
// CHECK: ^bb1(%{{.*}}: memref<256xi64>, %{{.*}}: index):
// CHECK: memref.load %[[OUTER]]
// CHECK: arith.cmpi slt
// CHECK: cf.cond_br

// Outer loop body: inner DO setup (4 0 DO)
// CHECK: ^bb2(%{{.*}}: memref<256xi64>, %{{.*}}: index):
// CHECK: arith.constant 4 : i32
// CHECK: arith.extsi %{{.*}} : i32 to i64
// CHECK: arith.constant 0 : i32
// CHECK: arith.extsi %{{.*}} : i32 to i64
// CHECK: %[[INNER:.*]] = memref.alloca() : memref<1xi64>
// CHECK: memref.store %{{.*}}, %[[INNER]]
// CHECK: cf.br ^bb4

// Outer loop exit -> return
// CHECK: ^bb3(%{{.*}}: memref<256xi64>, %{{.*}}: index):
// CHECK: return

// Inner loop header: check inner counter < inner limit
// CHECK: ^bb4(%{{.*}}: memref<256xi64>, %{{.*}}: index):
// CHECK: memref.load %[[INNER]]
// CHECK: arith.cmpi slt
// CHECK: cf.cond_br

// Inner loop body: J (load outer counter), I (load inner counter), add
// CHECK: ^bb5(%{{.*}}: memref<256xi64>, %{{.*}}: index):
// CHECK: memref.load %[[OUTER]]
// CHECK: memref.store
// CHECK: memref.load %[[INNER]]
// CHECK: memref.store
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: arith.addi %{{.*}}, %{{.*}} : i32
// CHECK: arith.extsi %{{.*}} : i32 to i64
// CHECK: memref.store
// CHECK: memref.load %[[INNER]]
// CHECK: arith.addi
// CHECK: memref.store %{{.*}}, %[[INNER]]
// CHECK: cf.br ^bb4

// Inner loop exit -> increment outer counter, loop back
// CHECK: ^bb6(%{{.*}}: memref<256xi64>, %{{.*}}: index):
// CHECK: memref.load %[[OUTER]]
// CHECK: arith.addi
// CHECK: memref.store %{{.*}}, %[[OUTER]]
// CHECK: cf.br ^bb1

// === BEGIN/WHILE/REPEAT inside IF: 5 IF BEGIN DUP WHILE 1 - REPEAT THEN ===
// CHECK-LABEL: func.func private @TEST__WHILE__INSIDE__IF

// Push 5, pop_flag, cond_br (IF)
// CHECK: arith.constant 5 : i32
// CHECK: arith.extsi %{{.*}} : i32 to i64
// CHECK: memref.store
// CHECK: arith.cmpi ne
// CHECK: cf.cond_br %{{.*}}, ^bb1({{.*}}), ^bb2({{.*}})

// IF true -> jump to WHILE header
// CHECK: ^bb1(%{{.*}}: memref<256xi64>, %{{.*}}: index):
// CHECK: cf.br ^bb3

// IF false / WHILE exit merge -> return
// CHECK: ^bb2(%{{.*}}: memref<256xi64>, %{{.*}}: index):
// CHECK: return

// WHILE condition: DUP, pop_flag, cond_br
// CHECK: ^bb3(%{{.*}}: memref<256xi64>, %{{.*}}: index):
// CHECK: memref.load
// CHECK: memref.store
// CHECK: arith.cmpi ne
// CHECK: cf.cond_br %{{.*}}, ^bb4({{.*}}), ^bb5({{.*}})

// WHILE body: push 1, subtract, loop back
// CHECK: ^bb4(%{{.*}}: memref<256xi64>, %{{.*}}: index):
// CHECK: arith.constant 1 : i32
// CHECK: arith.extsi %{{.*}} : i32 to i64
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: arith.trunci %{{.*}} : i64 to i32
// CHECK: arith.subi %{{.*}}, %{{.*}} : i32
// CHECK: arith.extsi %{{.*}} : i32 to i64
// CHECK: cf.br ^bb3

// WHILE exit -> merge with IF false
// CHECK: ^bb5(%{{.*}}: memref<256xi64>, %{{.*}}: index):
// CHECK: cf.br ^bb2

module {
  func.func private @TEST__NESTED__IF(%arg0: !forth.stack) -> !forth.stack {
    %0 = forth.constant %arg0(1 : i32) : !forth.stack -> !forth.stack
    %output_stack, %flag = forth.pop_flag %0 : !forth.stack -> !forth.stack, i1
    cf.cond_br %flag, ^bb1(%output_stack : !forth.stack), ^bb2(%output_stack : !forth.stack)
  ^bb1(%1: !forth.stack):
    %2 = forth.constant %1(2 : i32) : !forth.stack -> !forth.stack
    %output_stack_0, %flag_1 = forth.pop_flag %2 : !forth.stack -> !forth.stack, i1
    cf.cond_br %flag_1, ^bb3(%output_stack_0 : !forth.stack), ^bb4(%output_stack_0 : !forth.stack)
  ^bb2(%3: !forth.stack):
    return %3 : !forth.stack
  ^bb3(%4: !forth.stack):
    %5 = forth.constant %4(3 : i32) : !forth.stack -> !forth.stack
    cf.br ^bb4(%5 : !forth.stack)
  ^bb4(%6: !forth.stack):
    cf.br ^bb2(%6 : !forth.stack)
  }
  func.func private @TEST__IF__INSIDE__DO(%arg0: !forth.stack) -> !forth.stack {
    %0 = forth.constant %arg0(10 : i32) : !forth.stack -> !forth.stack
    %1 = forth.constant %0(0 : i32) : !forth.stack -> !forth.stack
    %output_stack, %value = forth.pop %1 : !forth.stack -> !forth.stack, i64
    %output_stack_0, %value_1 = forth.pop %output_stack : !forth.stack -> !forth.stack, i64
    %alloca = memref.alloca() : memref<1xi64>
    %c0 = arith.constant 0 : index
    memref.store %value, %alloca[%c0] : memref<1xi64>
    cf.br ^bb1(%output_stack_0 : !forth.stack)
  ^bb1(%2: !forth.stack):
    %c0_2 = arith.constant 0 : index
    %3 = memref.load %alloca[%c0_2] : memref<1xi64>
    %4 = arith.cmpi slt, %3, %value_1 : i64
    cf.cond_br %4, ^bb2(%2 : !forth.stack), ^bb3(%2 : !forth.stack)
  ^bb2(%5: !forth.stack):
    %c0_3 = arith.constant 0 : index
    %6 = memref.load %alloca[%c0_3] : memref<1xi64>
    %7 = forth.push_value %5, %6 : !forth.stack, i64 -> !forth.stack
    %8 = forth.constant %7(5 : i32) : !forth.stack -> !forth.stack
    %9 = forth.gti %8 : !forth.stack -> !forth.stack
    %output_stack_4, %flag = forth.pop_flag %9 : !forth.stack -> !forth.stack, i1
    cf.cond_br %flag, ^bb4(%output_stack_4 : !forth.stack), ^bb5(%output_stack_4 : !forth.stack)
  ^bb3(%10: !forth.stack):
    return %10 : !forth.stack
  ^bb4(%11: !forth.stack):
    %c0_5 = arith.constant 0 : index
    %12 = memref.load %alloca[%c0_5] : memref<1xi64>
    %13 = forth.push_value %11, %12 : !forth.stack, i64 -> !forth.stack
    cf.br ^bb5(%13 : !forth.stack)
  ^bb5(%14: !forth.stack):
    %c0_6 = arith.constant 0 : index
    %15 = memref.load %alloca[%c0_6] : memref<1xi64>
    %c1_i64 = arith.constant 1 : i64
    %16 = arith.addi %15, %c1_i64 : i64
    memref.store %16, %alloca[%c0_6] : memref<1xi64>
    cf.br ^bb1(%14 : !forth.stack)
  }
  func.func private @TEST__NESTED__DO__J(%arg0: !forth.stack) -> !forth.stack {
    %0 = forth.constant %arg0(3 : i32) : !forth.stack -> !forth.stack
    %1 = forth.constant %0(0 : i32) : !forth.stack -> !forth.stack
    %output_stack, %value = forth.pop %1 : !forth.stack -> !forth.stack, i64
    %output_stack_0, %value_1 = forth.pop %output_stack : !forth.stack -> !forth.stack, i64
    %alloca = memref.alloca() : memref<1xi64>
    %c0 = arith.constant 0 : index
    memref.store %value, %alloca[%c0] : memref<1xi64>
    cf.br ^bb1(%output_stack_0 : !forth.stack)
  ^bb1(%2: !forth.stack):
    %c0_2 = arith.constant 0 : index
    %3 = memref.load %alloca[%c0_2] : memref<1xi64>
    %4 = arith.cmpi slt, %3, %value_1 : i64
    cf.cond_br %4, ^bb2(%2 : !forth.stack), ^bb3(%2 : !forth.stack)
  ^bb2(%5: !forth.stack):
    %6 = forth.constant %5(4 : i32) : !forth.stack -> !forth.stack
    %7 = forth.constant %6(0 : i32) : !forth.stack -> !forth.stack
    %output_stack_3, %value_4 = forth.pop %7 : !forth.stack -> !forth.stack, i64
    %output_stack_5, %value_6 = forth.pop %output_stack_3 : !forth.stack -> !forth.stack, i64
    %alloca_7 = memref.alloca() : memref<1xi64>
    %c0_8 = arith.constant 0 : index
    memref.store %value_4, %alloca_7[%c0_8] : memref<1xi64>
    cf.br ^bb4(%output_stack_5 : !forth.stack)
  ^bb3(%8: !forth.stack):
    return %8 : !forth.stack
  ^bb4(%9: !forth.stack):
    %c0_9 = arith.constant 0 : index
    %10 = memref.load %alloca_7[%c0_9] : memref<1xi64>
    %11 = arith.cmpi slt, %10, %value_6 : i64
    cf.cond_br %11, ^bb5(%9 : !forth.stack), ^bb6(%9 : !forth.stack)
  ^bb5(%12: !forth.stack):
    %c0_10 = arith.constant 0 : index
    %13 = memref.load %alloca[%c0_10] : memref<1xi64>
    %14 = forth.push_value %12, %13 : !forth.stack, i64 -> !forth.stack
    %c0_11 = arith.constant 0 : index
    %15 = memref.load %alloca_7[%c0_11] : memref<1xi64>
    %16 = forth.push_value %14, %15 : !forth.stack, i64 -> !forth.stack
    %17 = forth.addi %16 : !forth.stack -> !forth.stack
    %c0_12 = arith.constant 0 : index
    %18 = memref.load %alloca_7[%c0_12] : memref<1xi64>
    %c1_i64 = arith.constant 1 : i64
    %19 = arith.addi %18, %c1_i64 : i64
    memref.store %19, %alloca_7[%c0_12] : memref<1xi64>
    cf.br ^bb4(%17 : !forth.stack)
  ^bb6(%20: !forth.stack):
    %c0_13 = arith.constant 0 : index
    %21 = memref.load %alloca[%c0_13] : memref<1xi64>
    %c1_i64_14 = arith.constant 1 : i64
    %22 = arith.addi %21, %c1_i64_14 : i64
    memref.store %22, %alloca[%c0_13] : memref<1xi64>
    cf.br ^bb1(%20 : !forth.stack)
  }
  func.func private @TEST__WHILE__INSIDE__IF(%arg0: !forth.stack) -> !forth.stack {
    %0 = forth.constant %arg0(5 : i32) : !forth.stack -> !forth.stack
    %output_stack, %flag = forth.pop_flag %0 : !forth.stack -> !forth.stack, i1
    cf.cond_br %flag, ^bb1(%output_stack : !forth.stack), ^bb2(%output_stack : !forth.stack)
  ^bb1(%1: !forth.stack):
    cf.br ^bb3(%1 : !forth.stack)
  ^bb2(%2: !forth.stack):
    return %2 : !forth.stack
  ^bb3(%3: !forth.stack):
    %4 = forth.dup %3 : !forth.stack -> !forth.stack
    %output_stack_0, %flag_1 = forth.pop_flag %4 : !forth.stack -> !forth.stack, i1
    cf.cond_br %flag_1, ^bb4(%output_stack_0 : !forth.stack), ^bb5(%output_stack_0 : !forth.stack)
  ^bb4(%5: !forth.stack):
    %6 = forth.constant %5(1 : i32) : !forth.stack -> !forth.stack
    %7 = forth.subi %6 : !forth.stack -> !forth.stack
    cf.br ^bb3(%7 : !forth.stack)
  ^bb5(%8: !forth.stack):
    cf.br ^bb2(%8 : !forth.stack)
  }
  func.func private @main() {
    %0 = forth.stack !forth.stack
    return
  }
}
