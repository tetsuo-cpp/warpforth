\ RUN: %warpforth-translate --forth-to-mlir %s | %warpforth-opt --warpforth-pipeline | %FileCheck %s

\ Verify that a naive attention kernel with shared memory and float intrinsics
\ survives the full pipeline to gpu.binary.
\ CHECK: gpu.binary @warpforth_module

\! kernel attention
\! param Q f32[16]
\! param K f32[16]
\! param V f32[16]
\! param O f32[16]
\! param SEQ_LEN i32
\! param HEAD_DIM i32
\! shared SCORES f32[4]
\! shared SCRATCH f32[4]

\ row = BID-X, t = TID-X
BID-X
TID-X

\ --- Dot product: Q[row,:] . K[t,:] ---
0.0
HEAD_DIM 0 DO
  2 PICK HEAD_DIM * I + CELLS Q + F@
  2 PICK HEAD_DIM * I + CELLS K + F@
  F* F+
LOOP
HEAD_DIM S>F FSQRT F/

\ --- Causal mask: if t > row, score = -inf ---
OVER 3 PICK >
IF DROP -1.0e30 THEN

\ --- Store score to shared memory ---
OVER CELLS SCORES + SF!
BARRIER

\ --- Softmax: max reduction (thread 0) ---
TID-X 0= IF
  0 CELLS SCORES + SF@
  SEQ_LEN 1 DO I CELLS SCORES + SF@ FMAX LOOP
  0 CELLS SCRATCH + SF!
THEN
BARRIER

\ --- Softmax: exp(score - max) ---
DUP CELLS SCORES + SF@
0 CELLS SCRATCH + SF@
F- FEXP
OVER CELLS SCORES + SF!
BARRIER

\ --- Softmax: sum reduction (thread 0) ---
TID-X 0= IF
  0.0
  SEQ_LEN 0 DO I CELLS SCORES + SF@ F+ LOOP
  0 CELLS SCRATCH + SF!
THEN
BARRIER

\ --- Softmax: normalize ---
DUP CELLS SCORES + SF@
0 CELLS SCRATCH + SF@
F/
OVER CELLS SCORES + SF!
BARRIER

\ --- V accumulation: O[row,col] = sum_j SCORES[j] * V[j*HD + col] ---
\ Stride over head_dim columns: col = t, t+BDIM-X, t+2*BDIM-X, ...
DUP BEGIN DUP HEAD_DIM < WHILE
  0.0
  SEQ_LEN 0 DO
    I CELLS SCORES + SF@
    I HEAD_DIM * 3 PICK + CELLS V + F@
    F* F+
  LOOP
  OVER 4 PICK HEAD_DIM * + CELLS O + F!
  BDIM-X +
REPEAT
DROP DROP DROP
