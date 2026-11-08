\! kernel acceptance_scale
\! param N i64
\! param SCALE i64
\! param INPUT i64[4194304]
\! param OUTPUT i64[4194304]

\ ( -- ) The fast arm of the acceptance fixture: OUT[i] = IN[i] * SCALE, one element per thread per
\ grid-stride turn, with nothing between the load and the store but the scale.
GLOBAL-ID
BEGIN DUP N < WHILE
  DUP CELLS INPUT + @
  SCALE *
  OVER CELLS OUTPUT + !
  GDIM-X BDIM-X * +
REPEAT
DROP
