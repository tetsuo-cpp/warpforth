\! kernel attention_f32
\! param NROW i64
\! param NKEY i64
\! param HDIM i64
\! param SCALE f64
\! param QMAT i64[16384]
\! param KMAT i64[16384]
\! param VMAT i64[16384]
\! param OMAT i64[16384]
\! shared SCORES f64[2048]
\! shared RED f64[512]

\ ( -- ) O = softmax(SCALE * Q K^T) V for f32 operands, one block per query row, strided over rows
\ by GDIM-X. Buffers hold f32, so each i64 cell count is half the element count and every pointer
\ step multiplies by 4. SCALE is 1/sqrt(HDIM) from the caller, because the word set has no
\ reciprocal square root. SCORES needs NKEY entries and RED one per thread.
\ HDIM must be a multiple of BDIM, because thread d owns output dimension d and threads with
\ d >= HDIM skip the output pass.
\ Each reduction is a shared tree, because the word set has no warp shuffle, and the row of scores
\ lives in shared memory, because there are no registers to hold it. Every word takes its result
\ out of shared memory rather than the data stack, so an iteration adds nothing to the stack and
\ the loop cannot overrun the 2 KiB per-thread buffer. RED[0] carries the row maximum and then the
\ row sum between stages.

: FDOT { pa pb dim -- dot }
  0.0
  dim 0 DO
    pa I 4 * + F32@
    pb I 4 * + F32@
    F* F+
  LOOP
;

: ROWSCORES { row pq pk ps len hd bd td sc -- }
  td
  BEGIN DUP len < WHILE
    DUP hd * 4 * pk +
    row hd * 4 * pq +
    SWAP
    hd FDOT
    sc F*
    OVER CELLS ps + SF!
    bd +
  REPEAT
  DROP
;

: ROWMAX { row ps len bd td -- max }
  -1.0e30
  td
  BEGIN DUP len < WHILE
    DUP CELLS ps + SF@
    ROT FMAX
    SWAP
    bd +
  REPEAT
  DROP
;

: ROWSUM { row ps len bd td -- partial }
  0.0
  td
  BEGIN DUP len < WHILE
    DUP CELLS ps + SF@
    ROT F+
    SWAP
    bd +
  REPEAT
  DROP
;

\ Softmax in place about RED[0].
: ROWEXP { row rdz ps len bd td -- }
  td
  BEGIN DUP len < WHILE
    DUP CELLS ps +
    DUP SF@ 0 CELLS rdz + SF@ F- FEXP
    SWAP SF!
    bd +
  REPEAT
  DROP
;

\ Halving tree fold over RED. Each level ends with a barrier so every thread sees the settled value.
: FOLDMAX { rdz bd td -- max }
  bd 2 /
  BEGIN DUP 0 > WHILE
    DUP td >
    IF
      td OVER + CELLS rdz + SF@
      td CELLS rdz + SF@
      FMAX
      td CELLS rdz + SF!
    THEN
    BARRIER
    2 /
  REPEAT
  DROP
  0 CELLS rdz + SF@
;

: FOLDSUM { rdz bd td -- sum }
  bd 2 /
  BEGIN DUP 0 > WHILE
    DUP td >
    IF
      td OVER + CELLS rdz + SF@
      td CELLS rdz + SF@
      F+
      td CELLS rdz + SF!
    THEN
    BARRIER
    2 /
  REPEAT
  DROP
  0 CELLS rdz + SF@
;

: ROWOUT { sum row ps pv po len hd td -- }
  td hd <
  IF
    0.0
    len 0 DO
      I CELLS ps + SF@
      I hd * td + 4 * pv + F32@
      F* F+
    LOOP
    sum F/
    row hd * 4 * po + td 4 * + F32!
  THEN
;

BID-X NROW <
IF
  NROW BID-X
  DO
    I QMAT KMAT SCORES NKEY HDIM BDIM-X TID-X SCALE ROWSCORES
    BARRIER

    I SCORES NKEY BDIM-X TID-X ROWMAX
    TID-X CELLS RED + SF!
    BARRIER
    RED BDIM-X TID-X FOLDMAX
    0 CELLS RED + SF!
    BARRIER

    I RED SCORES NKEY BDIM-X TID-X ROWEXP
    BARRIER
    I SCORES NKEY BDIM-X TID-X ROWSUM
    TID-X CELLS RED + SF!
    BARRIER
    RED BDIM-X TID-X FOLDSUM
    I SCORES VMAT OMAT NKEY HDIM TID-X ROWOUT
    BARRIER
    GDIM-X
  +LOOP
THEN
