// RUN: %warpforth-opt --convert-forth-to-memref %s | %FileCheck %s

// === Nested IF → nested scf.if ===
// CHECK-LABEL: func.func private @test_nested_if
// CHECK: scf.if %{{.*}} -> (index) {
// CHECK:   scf.if %{{.*}} -> (index) {
// CHECK:     scf.yield
// CHECK:   } else {
// CHECK:     scf.yield
// CHECK:   }
// CHECK:   scf.yield
// CHECK: } else {
// CHECK:   scf.yield
// CHECK: }

module {
  func.func private @test_nested_if() {
    %0 = forth.stack !forth.stack
    %1 = forth.literal %0 1 : !forth.stack -> !forth.stack
    %2 = forth.if %1 : !forth.stack -> !forth.stack {
    ^bb0(%arg0: !forth.stack):
      %3 = forth.drop %arg0 : !forth.stack -> !forth.stack
      %4 = forth.literal %3 1 : !forth.stack -> !forth.stack
      %5 = forth.if %4 : !forth.stack -> !forth.stack {
      ^bb0(%arg1: !forth.stack):
        %6 = forth.drop %arg1 : !forth.stack -> !forth.stack
        %7 = forth.literal %6 42 : !forth.stack -> !forth.stack
        forth.yield %7 : !forth.stack
      } else {
      ^bb0(%arg1: !forth.stack):
        %6 = forth.drop %arg1 : !forth.stack -> !forth.stack
        forth.yield %6 : !forth.stack
      }
      forth.yield %5 : !forth.stack
    } else {
    ^bb0(%arg0: !forth.stack):
      %3 = forth.drop %arg0 : !forth.stack -> !forth.stack
      forth.yield %3 : !forth.stack
    }
    return
  }

  // === IF inside DO/LOOP → scf.if inside scf.for ===
  // CHECK-LABEL: func.func private @test_if_inside_do
  // CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (index) {
  // CHECK:   scf.if %{{.*}} -> (index) {
  // CHECK:     scf.yield
  // CHECK:   } else {
  // CHECK:     scf.yield
  // CHECK:   }
  // CHECK:   scf.yield
  // CHECK: }

  func.func private @test_if_inside_do() {
    %0 = forth.stack !forth.stack
    %1 = forth.literal %0 10 : !forth.stack -> !forth.stack
    %2 = forth.literal %1 0 : !forth.stack -> !forth.stack
    %3 = forth.do_loop %2 : !forth.stack -> !forth.stack {
    ^bb0(%arg0: !forth.stack):
      %4 = forth.loop_index %arg0 : !forth.stack -> !forth.stack
      %5 = forth.literal %4 5 : !forth.stack -> !forth.stack
      %6 = forth.gt %5 : !forth.stack -> !forth.stack
      %7 = forth.if %6 : !forth.stack -> !forth.stack {
      ^bb0(%arg1: !forth.stack):
        %8 = forth.drop %arg1 : !forth.stack -> !forth.stack
        %9 = forth.literal %8 99 : !forth.stack -> !forth.stack
        forth.yield %9 : !forth.stack
      } else {
      ^bb0(%arg1: !forth.stack):
        %8 = forth.drop %arg1 : !forth.stack -> !forth.stack
        forth.yield %8 : !forth.stack
      }
      forth.yield %7 : !forth.stack
    }
    return
  }

  // === Nested DO/LOOP with J (depth=1) ===
  // CHECK-LABEL: func.func private @test_nested_do_j
  // Outer scf.for
  // CHECK: scf.for %[[OUTER_IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (index) {
  //   Inner scf.for
  // CHECK:   scf.for %[[INNER_IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (index) {
  //     J pushes outer IV, I pushes inner IV
  // CHECK:     arith.index_cast %[[OUTER_IV]]
  // CHECK:     arith.index_cast %[[INNER_IV]]
  // CHECK:     scf.yield
  // CHECK:   }
  // CHECK:   scf.yield
  // CHECK: }

  func.func private @test_nested_do_j() {
    %0 = forth.stack !forth.stack
    %1 = forth.literal %0 3 : !forth.stack -> !forth.stack
    %2 = forth.literal %1 0 : !forth.stack -> !forth.stack
    %3 = forth.do_loop %2 : !forth.stack -> !forth.stack {
    ^bb0(%arg0: !forth.stack):
      %4 = forth.literal %arg0 4 : !forth.stack -> !forth.stack
      %5 = forth.literal %4 0 : !forth.stack -> !forth.stack
      %6 = forth.do_loop %5 : !forth.stack -> !forth.stack {
      ^bb0(%arg1: !forth.stack):
        %7 = forth.loop_index %arg1 {depth = 1 : i64} : !forth.stack -> !forth.stack
        %8 = forth.loop_index %7 : !forth.stack -> !forth.stack
        %9 = forth.add %8 : !forth.stack -> !forth.stack
        forth.yield %9 : !forth.stack
      }
      forth.yield %6 : !forth.stack
    }
    return
  }

  // === BEGIN/WHILE/REPEAT inside IF ===
  // CHECK-LABEL: func.func private @test_while_inside_if
  // CHECK: scf.if %{{.*}} -> (index) {
  // CHECK:   scf.while
  // CHECK:     scf.condition
  // CHECK:   } do {
  // CHECK:     scf.yield
  // CHECK:   }
  // CHECK:   scf.yield
  // CHECK: } else {
  // CHECK:   scf.yield
  // CHECK: }

  func.func private @test_while_inside_if() {
    %0 = forth.stack !forth.stack
    %1 = forth.literal %0 1 : !forth.stack -> !forth.stack
    %2 = forth.if %1 : !forth.stack -> !forth.stack {
    ^bb0(%arg0: !forth.stack):
      %3 = forth.drop %arg0 : !forth.stack -> !forth.stack
      %4 = forth.begin_while_repeat %3 : !forth.stack -> !forth.stack {
      ^bb0(%arg1: !forth.stack):
        %5 = forth.dup %arg1 : !forth.stack -> !forth.stack
        forth.yield %5 while_cond : !forth.stack
      } do {
      ^bb0(%arg1: !forth.stack):
        %5 = forth.literal %arg1 1 : !forth.stack -> !forth.stack
        %6 = forth.sub %5 : !forth.stack -> !forth.stack
        forth.yield %6 : !forth.stack
      }
      forth.yield %4 : !forth.stack
    } else {
    ^bb0(%arg0: !forth.stack):
      %3 = forth.drop %arg0 : !forth.stack -> !forth.stack
      forth.yield %3 : !forth.stack
    }
    return
  }
}
