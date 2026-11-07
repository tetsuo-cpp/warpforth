\! kernel scale_i64
\! param N i64
\! param SCALE i64
\! param INPUT i64[4194304]
\! param OUTPUT i64[4194304]

\ ( -- ) OUT[i] = IN[i] * SCALE, one element per thread per grid-stride turn because the
\ word set has no vector load.
GLOBAL-ID
BEGIN DUP N < WHILE
  DUP CELLS INPUT + @
  SCALE *
  OVER CELLS OUTPUT + !
  GDIM-X BDIM-X * +
REPEAT
DROP
