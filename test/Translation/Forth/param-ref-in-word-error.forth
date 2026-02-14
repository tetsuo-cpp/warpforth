\ RUN: %not %warpforth-translate --forth-to-mlir %s 2>&1 | %FileCheck %s
\ CHECK: parameter 'data' cannot be referenced inside a word definition
param data 256
: bad-word data @ ;
bad-word
