\! kernel axpy_f64
\! param N i64
\! param ALPHA f64
\! param X f64[4194304]
\! param Y f64[4194304]

\ ( -- ) Y[i] = ALPHA * X[i] + Y[i]. F* then F+ because the word set has no fused multiply-add.
GLOBAL-ID
BEGIN DUP N < WHILE
  DUP CELLS X + F@
  ALPHA F*
  OVER CELLS Y + F@
  F+
  OVER CELLS Y + F!
  GDIM-X BDIM-X * +
REPEAT
DROP
