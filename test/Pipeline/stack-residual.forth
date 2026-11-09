\ RUN: %warpforth-translate --forth-to-mlir %s | %warpforth-opt --inline --lower-forth-stack --canonicalize | %FileCheck %s

\ The PICK index `1 1 +` is not a constant while the stack is lowered, so the
\ three cells are stored at indices 1 to 3 of the residual buffer and one load
\ reads index 3 - n. After folding, the load reads index 1, where 10 was
\ stored, so OUT receives 10.

\! kernel main
\! param OUT i64[1]
10 20 30 1 1 + PICK OUT !

\ CHECK-LABEL: func.func @main(
\ CHECK-DAG: %[[INDEX1:.*]] = arith.constant 1 : index
\ CHECK-DAG: %[[INDEX2:.*]] = arith.constant 2 : index
\ CHECK-DAG: %[[INDEX3:.*]] = arith.constant 3 : index
\ CHECK-DAG: %[[TEN:.*]] = arith.constant 10 : i64
\ CHECK-DAG: %[[TWENTY:.*]] = arith.constant 20 : i64
\ CHECK-DAG: %[[THIRTY:.*]] = arith.constant 30 : i64
\ CHECK: %[[BUFFER:.*]] = memref.alloca() : memref<256xi64>
\ CHECK-NEXT: memref.store %[[TEN]], %[[BUFFER]][%[[INDEX1]]]
\ CHECK-NEXT: memref.store %[[TWENTY]], %[[BUFFER]][%[[INDEX2]]]
\ CHECK-NEXT: memref.store %[[THIRTY]], %[[BUFFER]][%[[INDEX3]]]
\ CHECK-NEXT: %[[PICKED:.*]] = memref.load %[[BUFFER]][%[[INDEX1]]]
\ CHECK: llvm.store %[[PICKED]], %{{.*}} : i64, !llvm.ptr<1>
