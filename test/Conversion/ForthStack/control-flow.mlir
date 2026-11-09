// RUN: %warpforth-opt --lower-forth-stack %s | %FileCheck %s
// RUN: %warpforth-opt --lower-forth-stack --canonicalize %s | %FileCheck %s --check-prefix=FOLD

// `1 IF 42 ELSE 99 THEN OUT !`: each arm leaves one cell, so the join takes
// that cell as one i64 argument and stores it. The arms take no arguments
// because popping the flag left the stack empty.
// CHECK-LABEL: func.func @if_else(
// CHECK: %[[FLAG:.*]] = arith.cmpi ne, %{{.*}}, %{{.*}} : i64
// CHECK: cf.cond_br %[[FLAG]], ^[[THEN:bb[0-9]+]], ^[[ELSE:bb[0-9]+]]
// CHECK: ^[[THEN]]:
// CHECK: %[[C42:.*]] = arith.constant 42 : i64
// CHECK: cf.br ^[[JOIN:bb[0-9]+]](%[[C42]] : i64)
// CHECK: ^[[ELSE]]:
// CHECK: %[[C99:.*]] = arith.constant 99 : i64
// CHECK: cf.br ^[[JOIN]](%[[C99]] : i64)
// CHECK: ^[[JOIN]](%[[RESULT:.*]]: i64):
// CHECK: llvm.store %[[RESULT]], %{{.*}} : i64, !llvm.ptr<1>

// FOLD-LABEL: func.func @if_else(
// FOLD: %[[C42:.*]] = arith.constant 42 : i64
// FOLD: llvm.store %[[C42]], %{{.*}} : i64, !llvm.ptr<1>
func.func @if_else(%out: memref<1xi64>) {
  %0 = forth.stack !forth.stack
  %1 = forth.constant %0(1 : i64) : !forth.stack -> !forth.stack
  %output_stack, %flag = forth.pop_flag %1 : !forth.stack -> !forth.stack, i1
  cf.cond_br %flag, ^bb1(%output_stack : !forth.stack), ^bb2(%output_stack : !forth.stack)
^bb1(%2: !forth.stack):
  %3 = forth.constant %2(42 : i64) : !forth.stack -> !forth.stack
  cf.br ^bb3(%3 : !forth.stack)
^bb2(%4: !forth.stack):
  %5 = forth.constant %4(99 : i64) : !forth.stack -> !forth.stack
  cf.br ^bb3(%5 : !forth.stack)
^bb3(%6: !forth.stack):
  %7 = forth.param_ref %6 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %8 = forth.storei %7 : !forth.stack -> !forth.stack
  return
}

// `5 10 2 DO I + LOOP OUT !`: the body pushes I and adds it, so every
// iteration starts and ends with one cell. The loop header carries that cell
// as an i64 argument, and the loop counter stays in memory.
// CHECK-LABEL: func.func @do_loop_sum(
// CHECK: %[[INITIAL:.*]] = arith.constant 5 : i64
// CHECK: %[[LIMIT:.*]] = arith.constant 10 : i64
// CHECK: %[[START:.*]] = arith.constant 2 : i64
// CHECK: %[[COUNTER:.*]] = memref.alloca() : memref<1xi64>
// CHECK: memref.store %[[START]], %[[COUNTER]]
// CHECK: cf.br ^[[HEADER:bb[0-9]+]](%[[INITIAL]] : i64)
// CHECK: ^[[HEADER]](%[[ACC:.*]]: i64):
// CHECK: %[[I:.*]] = memref.load %[[COUNTER]]
// CHECK: %[[SUM:.*]] = arith.addi %[[ACC]], %[[I]] : i64
// CHECK: arith.subi %{{.*}}, %[[LIMIT]] : i64
// CHECK: cf.cond_br %{{.*}}, ^[[EXIT:bb[0-9]+]](%[[SUM]] : i64), ^[[HEADER]](%[[SUM]] : i64)
// CHECK: ^[[EXIT]](%[[TOTAL:.*]]: i64):
// CHECK: llvm.store %[[TOTAL]], %{{.*}} : i64, !llvm.ptr<1>
func.func @do_loop_sum(%out: memref<1xi64>) {
  %0 = forth.stack !forth.stack
  %1 = forth.constant %0(5 : i64) : !forth.stack -> !forth.stack
  %2 = forth.constant %1(10 : i64) : !forth.stack -> !forth.stack
  %3 = forth.constant %2(2 : i64) : !forth.stack -> !forth.stack
  %output_stack, %value = forth.pop %3 : !forth.stack -> !forth.stack, i64
  %output_stack_0, %value_1 = forth.pop %output_stack : !forth.stack -> !forth.stack, i64
  %alloca = memref.alloca() : memref<1xi64>
  %c0 = arith.constant 0 : index
  memref.store %value, %alloca[%c0] : memref<1xi64>
  cf.br ^bb1(%output_stack_0 : !forth.stack)
^bb1(%4: !forth.stack):
  %c0_2 = arith.constant 0 : index
  %5 = memref.load %alloca[%c0_2] : memref<1xi64>
  %6 = forth.push_value %4, %5 : !forth.stack, i64 -> !forth.stack
  %7 = forth.addi %6 : !forth.stack -> !forth.stack
  %c1_i64 = arith.constant 1 : i64
  %c0_3 = arith.constant 0 : index
  %8 = memref.load %alloca[%c0_3] : memref<1xi64>
  %9 = arith.addi %8, %c1_i64 : i64
  memref.store %9, %alloca[%c0_3] : memref<1xi64>
  %10 = arith.subi %8, %value_1 : i64
  %11 = arith.subi %9, %value_1 : i64
  %12 = arith.xori %10, %11 : i64
  %c0_i64 = arith.constant 0 : i64
  %13 = arith.cmpi slt, %12, %c0_i64 : i64
  cf.cond_br %13, ^bb2(%7 : !forth.stack), ^bb1(%7 : !forth.stack)
^bb2(%14: !forth.stack):
  %15 = forth.param_ref %14 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %16 = forth.storei %15 : !forth.stack -> !forth.stack
  return
}

// `10 BEGIN 1 - DUP 0= UNTIL OUT !`: DUP copies a register, so the loop
// carries one i64 and the flag reads the decremented cell directly.
// CHECK-LABEL: func.func @begin_until(
// CHECK: %[[TEN:.*]] = arith.constant 10 : i64
// CHECK: cf.br ^[[LOOP:bb[0-9]+]](%[[TEN]] : i64)
// CHECK: ^[[LOOP]](%[[N:.*]]: i64):
// CHECK: %[[NEXT:.*]] = arith.subi %[[N]], %{{.*}} : i64
// CHECK: %[[IS_ZERO:.*]] = arith.cmpi eq, %[[NEXT]], %{{.*}} : i64
// CHECK: %[[FLAG:.*]] = arith.extsi %[[IS_ZERO]] : i1 to i64
// CHECK: %[[DONE:.*]] = arith.cmpi ne, %[[FLAG]], %{{.*}} : i64
// CHECK: cf.cond_br %[[DONE]], ^[[EXIT:bb[0-9]+]](%[[NEXT]] : i64), ^[[LOOP]](%[[NEXT]] : i64)
// CHECK: ^[[EXIT]](%[[LAST:.*]]: i64):
// CHECK: llvm.store %[[LAST]], %{{.*}} : i64, !llvm.ptr<1>
func.func @begin_until(%out: memref<1xi64>) {
  %0 = forth.stack !forth.stack
  %1 = forth.constant %0(10 : i64) : !forth.stack -> !forth.stack
  cf.br ^bb1(%1 : !forth.stack)
^bb1(%2: !forth.stack):
  %3 = forth.constant %2(1 : i64) : !forth.stack -> !forth.stack
  %4 = forth.subi %3 : !forth.stack -> !forth.stack
  %5 = forth.dup %4 : !forth.stack -> !forth.stack
  %6 = forth.zero_eq %5 : !forth.stack -> !forth.stack
  %output_stack, %flag = forth.pop_flag %6 : !forth.stack -> !forth.stack, i1
  cf.cond_br %flag, ^bb2(%output_stack : !forth.stack), ^bb1(%output_stack : !forth.stack)
^bb2(%7: !forth.stack):
  %8 = forth.param_ref %7 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %9 = forth.storei %8 : !forth.stack -> !forth.stack
  return
}

// `10 BEGIN DUP 0 > WHILE 1 - REPEAT OUT !`: the test, the body, and the exit
// each take the one cell as an i64 argument.
// CHECK-LABEL: func.func @begin_while_repeat(
// CHECK: cf.br ^[[TEST:bb[0-9]+]](%{{.*}} : i64)
// CHECK: ^[[TEST]](%[[N:.*]]: i64):
// CHECK: arith.cmpi sgt, %[[N]], %{{.*}} : i64
// CHECK: cf.cond_br %{{.*}}, ^[[BODY:bb[0-9]+]](%[[N]] : i64), ^[[EXIT:bb[0-9]+]](%[[N]] : i64)
// CHECK: ^[[BODY]](%[[M:.*]]: i64):
// CHECK: %[[DECREMENTED:.*]] = arith.subi %[[M]], %{{.*}} : i64
// CHECK: cf.br ^[[TEST]](%[[DECREMENTED]] : i64)
// CHECK: ^[[EXIT]](%[[LAST:.*]]: i64):
// CHECK: llvm.store %[[LAST]], %{{.*}} : i64, !llvm.ptr<1>
func.func @begin_while_repeat(%out: memref<1xi64>) {
  %0 = forth.stack !forth.stack
  %1 = forth.constant %0(10 : i64) : !forth.stack -> !forth.stack
  cf.br ^bb1(%1 : !forth.stack)
^bb1(%2: !forth.stack):
  %3 = forth.dup %2 : !forth.stack -> !forth.stack
  %4 = forth.constant %3(0 : i64) : !forth.stack -> !forth.stack
  %5 = forth.gti %4 : !forth.stack -> !forth.stack
  %output_stack, %flag = forth.pop_flag %5 : !forth.stack -> !forth.stack, i1
  cf.cond_br %flag, ^bb2(%output_stack : !forth.stack), ^bb3(%output_stack : !forth.stack)
^bb2(%6: !forth.stack):
  %7 = forth.constant %6(1 : i64) : !forth.stack -> !forth.stack
  %8 = forth.subi %7 : !forth.stack -> !forth.stack
  cf.br ^bb1(%8 : !forth.stack)
^bb3(%9: !forth.stack):
  %10 = forth.param_ref %9 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %11 = forth.storei %10 : !forth.stack -> !forth.stack
  return
}

// `0 10 0 DO I + I 3 = IF LEAVE THEN LOOP OUT !`: LEAVE and the loop's own
// exit both reach the store with the running sum as one i64.
// CHECK-LABEL: func.func @leave(
// CHECK: cf.br ^[[BODY:bb[0-9]+]](%{{.*}} : i64)
// CHECK: ^[[BODY]](%[[ACC:.*]]: i64):
// CHECK: %[[SUM:.*]] = arith.addi %[[ACC]], %{{.*}} : i64
// CHECK: cf.cond_br %{{.*}}, ^[[LEAVE:bb[0-9]+]](%[[SUM]] : i64), ^[[NEXT:bb[0-9]+]](%[[SUM]] : i64)
// CHECK: ^[[EXIT:bb[0-9]+]](%[[TOTAL:.*]]: i64):
// CHECK: llvm.store %[[TOTAL]], %{{.*}} : i64, !llvm.ptr<1>
// CHECK: ^[[LEAVE]](%[[LEFT:.*]]: i64):
// CHECK-NEXT: cf.br ^[[EXIT]](%[[LEFT]] : i64)
// CHECK: ^[[NEXT]](%[[CARRIED:.*]]: i64):
// CHECK: cf.cond_br %{{.*}}, ^[[EXIT]](%[[CARRIED]] : i64), ^[[BODY]](%[[CARRIED]] : i64)
func.func @leave(%out: memref<1xi64>) {
  %0 = forth.stack !forth.stack
  %1 = forth.constant %0(0 : i64) : !forth.stack -> !forth.stack
  %2 = forth.constant %1(10 : i64) : !forth.stack -> !forth.stack
  %3 = forth.constant %2(0 : i64) : !forth.stack -> !forth.stack
  %output_stack, %value = forth.pop %3 : !forth.stack -> !forth.stack, i64
  %output_stack_0, %value_1 = forth.pop %output_stack : !forth.stack -> !forth.stack, i64
  %alloca = memref.alloca() : memref<1xi64>
  %c0 = arith.constant 0 : index
  memref.store %value, %alloca[%c0] : memref<1xi64>
  cf.br ^bb1(%output_stack_0 : !forth.stack)
^bb1(%4: !forth.stack):
  %c0_2 = arith.constant 0 : index
  %5 = memref.load %alloca[%c0_2] : memref<1xi64>
  %6 = forth.push_value %4, %5 : !forth.stack, i64 -> !forth.stack
  %7 = forth.addi %6 : !forth.stack -> !forth.stack
  %c0_3 = arith.constant 0 : index
  %8 = memref.load %alloca[%c0_3] : memref<1xi64>
  %9 = forth.push_value %7, %8 : !forth.stack, i64 -> !forth.stack
  %10 = forth.constant %9(3 : i64) : !forth.stack -> !forth.stack
  %11 = forth.eqi %10 : !forth.stack -> !forth.stack
  %output_stack_4, %flag = forth.pop_flag %11 : !forth.stack -> !forth.stack, i1
  cf.cond_br %flag, ^bb3(%output_stack_4 : !forth.stack), ^bb4(%output_stack_4 : !forth.stack)
^bb2(%12: !forth.stack):
  %13 = forth.param_ref %12 %out : !forth.stack, memref<1xi64> -> !forth.stack
  %14 = forth.storei %13 : !forth.stack -> !forth.stack
  return
^bb3(%15: !forth.stack):
  cf.br ^bb2(%15 : !forth.stack)
^bb4(%16: !forth.stack):
  %c1_i64 = arith.constant 1 : i64
  %c0_5 = arith.constant 0 : index
  %17 = memref.load %alloca[%c0_5] : memref<1xi64>
  %18 = arith.addi %17, %c1_i64 : i64
  memref.store %18, %alloca[%c0_5] : memref<1xi64>
  %19 = arith.subi %17, %value_1 : i64
  %20 = arith.subi %18, %value_1 : i64
  %21 = arith.xori %19, %20 : i64
  %c0_i64 = arith.constant 0 : i64
  %22 = arith.cmpi slt, %21, %c0_i64 : i64
  cf.cond_br %22, ^bb2(%16 : !forth.stack), ^bb1(%16 : !forth.stack)
^bb5:
  %23 = forth.stack !forth.stack
  cf.br ^bb4(%23 : !forth.stack)
}
