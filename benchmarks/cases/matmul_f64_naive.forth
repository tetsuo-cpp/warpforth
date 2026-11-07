\! kernel matmul_f64_naive
\! param N i64
\! param A f64[16384]
\! param B f64[16384]
\! param C f64[16384]

\ ( -- ) C[row][col] = sum_k A[row][k] * B[k][col]. One thread per output element and no staging,
\ so both operands round-trip through the stack slot on every multiply.
\ Bases and dimensions travel into the word as arguments, because a param name cannot be
\ referenced inside a word definition and a local may not reuse a param name.
: MCELL { row col pa pb pc dim -- }
  0.0
  dim 0 DO
    pa row dim * I + CELLS + F@
    pb I dim * col + CELLS + F@
    F* F+
  LOOP
  pc row dim * col + CELLS + F!
;

GLOBAL-ID
BEGIN DUP N N * < WHILE
  DUP N /
  OVER N MOD
  A B C N MCELL
  GDIM-X BDIM-X * +
REPEAT
DROP
