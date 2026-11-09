\! kernel reduce_sum_f64
\! param N i64
\! param INPUT f64[1048576]
\! param OUTPUT f64[1]
\! shared PART f64[1024]

\ ( -- ) OUTPUT[0] = sum of INPUT[0..N). One block, because folding partial sums across blocks
\ needs an atomic or a device-wide arrival counter and the word set has neither.
0.0
GLOBAL-ID
BEGIN DUP N < WHILE
  DUP CELLS INPUT + F@
  ROT F+
  SWAP
  BDIM-X +
REPEAT
DROP
TID-X CELLS PART + SF!
BARRIER

\ Shared tree fold, halving the stride each level. Each level costs a block barrier.
BDIM-X 2 /
BEGIN DUP 0 > WHILE
  DUP TID-X >
  IF
    TID-X OVER + CELLS PART + SF@
    TID-X CELLS PART + SF@
    F+
    TID-X CELLS PART + SF!
  THEN
  BARRIER
  2 /
REPEAT
DROP

TID-X 0= IF
  0 CELLS PART + SF@
  0 CELLS OUTPUT + F!
THEN
