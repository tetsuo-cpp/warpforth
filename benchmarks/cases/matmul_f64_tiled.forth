\! kernel matmul_f64_tiled
\! param N i64
\! param A f64[65536]
\! param B f64[65536]
\! param C f64[65536]
\! shared SA f64[256]
\! shared SB f64[256]

\ ( -- ) C = A * B in 16x16 shared tiles, 256 threads per block, grid-strided over tiles so the
\ launch stays inside the 2 KiB per-thread stack budget whatever N is.
\ Lane (ty, tx) stages A[brow*16+ty][kk*16+tx] and B[kk*16+ty][bcol*16+tx], both at shared index
\ ty*16 + tx, which is exactly TID-X for the 256-thread block.
: BTILE { tile pa pb pc tsa tsb dim ty tx tiles -- }
  0.0
  dim 16 / 0 DO
    BARRIER
    pa tile tiles / 16 * ty + dim * I 16 * + tx + CELLS + F@
    ty 16 * tx + CELLS tsa + SF!
    pb I 16 * ty + dim * tile tiles MOD 16 * + tx + CELLS + F@
    ty 16 * tx + CELLS tsb + SF!
    BARRIER
    16 0 DO
      tsa ty 16 * I + CELLS + SF@
      tsb I 16 * tx + CELLS + SF@
      F* F+
    LOOP
  LOOP
  pc tile tiles / 16 * ty + dim * tile tiles MOD 16 * + tx + CELLS + F!
;

BID-X
BEGIN DUP N 16 / DUP * < WHILE
  DUP A B C SA SB N TID-X 16 / TID-X 16 MOD N 16 / BTILE
  GDIM-X +
REPEAT
DROP
