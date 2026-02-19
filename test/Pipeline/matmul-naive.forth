\ RUN: %warpforth-translate --forth-to-mlir %s | %warpforth-opt --warpforth-pipeline | %FileCheck %s
\ RUN: %warpforth-translate --forth-to-mlir %s | %warpforth-opt --convert-forth-to-memref --convert-forth-to-gpu | %FileCheck %s --check-prefix=MID

\ Verify that a naive integer matmul kernel survives the full pipeline.
\ CHECK: gpu.binary @warpforth_module

\ Verify the kernel signature at the memref+gpu stage.
\ MID: gpu.func @main(%arg0: memref<8xi64> {forth.param_name = "A"}, %arg1: memref<12xi64> {forth.param_name = "B"}, %arg2: memref<6xi64> {forth.param_name = "C"}) kernel

\! kernel main
\! param A i64[8]
\! param B i64[12]
\! param C i64[6]

\ M=2, N=3, K=4. One thread computes C[row, col] where gid = row*N + col.
GLOBAL-ID
DUP 3 /
SWAP 3 MOD
0
4 0 DO
  2 PICK
  I SWAP 4 * +
  CELLS A + @
  I 3 * 3 PICK + CELLS B + @
  * +
LOOP
2 PICK 3 * 2 PICK +
CELLS C + !
