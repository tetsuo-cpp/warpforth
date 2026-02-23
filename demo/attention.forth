\ GPT-2 attention kernel with f32 global memory, f64 shared memory for softmax.
\ Adapted from test/Pipeline/attention.forth — 4 lines changed for f32 access.
\
\ Q/K/V/O are f32 arrays passed as raw byte buffers (i64[]).
\ Global loads/stores use F32@/F32! with 4* byte addressing (f32 = 4 bytes).
\ Shared memory stays f64 for softmax precision, using SF@/SF! with CELLS.

\! kernel attention
\! param Q i64[32768]
\! param K i64[32768]
\! param V i64[32768]
\! param O i64[32768]
\! param SEQ_LEN i64
\! param HEAD_DIM i64
\! shared SCORES f64[1024]
\! shared SCRATCH f64[1024]

\ row = BID-X, t = TID-X
BID-X
TID-X

\ --- Dot product: Q[row,:] . K[t,:] ---
0.0
HEAD_DIM 0 DO
  2 PICK HEAD_DIM * I + 4 * Q + F32@
  2 PICK HEAD_DIM * I + 4 * K + F32@
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
    I HEAD_DIM * 3 PICK + 4 * V + F32@
    F* F+
  LOOP
  OVER 4 PICK HEAD_DIM * + 4 * O + F32!
  BDIM-X +
REPEAT
DROP DROP DROP
