// RUN: %not %warpforth-opt --convert-forth-to-gpu %s 2>&1 | %FileCheck %s

// CHECK: error: custom op 'forth.intrinsic' is unknown

module {
  func.func private @main() attributes {forth.kernel} {
    %0 = forth.intrinsic "tid-x" : index
    return
  }
}
