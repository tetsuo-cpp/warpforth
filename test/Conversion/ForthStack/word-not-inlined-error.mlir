// RUN: %not %warpforth-opt --lower-forth-stack %s 2>&1 | %FileCheck %s

// The lowered stack has no calling convention, so every word must be inlined
// before the stack is lowered.
// CHECK: error: Forth word @double was not inlined; run --inline before --lower-forth-stack

func.func private @double(%s: !forth.stack) -> !forth.stack {
  %0 = forth.dup %s : !forth.stack -> !forth.stack
  %1 = forth.addi %0 : !forth.stack -> !forth.stack
  return %1 : !forth.stack
}

func.func @main() attributes {forth.kernel} {
  %0 = forth.stack !forth.stack
  %1 = forth.constant %0(5 : i64) : !forth.stack -> !forth.stack
  %2 = call @double(%1) : (!forth.stack) -> !forth.stack
  return
}
