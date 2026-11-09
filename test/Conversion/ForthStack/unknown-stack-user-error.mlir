// RUN: %not %warpforth-opt --allow-unregistered-dialect --lower-forth-stack %s 2>&1 | %FileCheck %s

// Only Forth words and branches may use a stack, so an op outside the table is
// an error instead of a stack value with no cells.
// CHECK: error: 'test.consume' op uses a !forth.stack but has no stack-word row
func.func @main() {
  %0 = forth.stack !forth.stack
  "test.consume"(%0) : (!forth.stack) -> ()
  return
}
