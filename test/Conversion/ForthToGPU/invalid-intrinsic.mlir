// RUN: %not %warpforth-opt --convert-forth-to-gpu %s > %t.out 2> %t.err
// RUN: %FileCheck %s < %t.err
// RUN: test ! -s %t.out

// CHECK: failed to legalize operation 'forth.intrinsic'

module {
  func.func private @main() attributes {forth.kernel} {
    %0 = forth.intrinsic "unknown" : index
    return
  }
}
