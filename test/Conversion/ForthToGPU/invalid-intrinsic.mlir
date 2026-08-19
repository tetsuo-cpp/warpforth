// RUN: %not %warpforth-opt --convert-forth-to-gpu %s > %t.out 2> %t.err
// RUN: %FileCheck %s < %t.err
// RUN: test ! -s %t.out

// CHECK: invalid-intrinsic.forth:12:7: error: failed to legalize operation 'forth.intrinsic'

module {
  func.func private @main() attributes {forth.kernel} {
    %0 = forth.intrinsic "unknown" : index loc("invalid-intrinsic.forth":12:7)
    return
  }
}
