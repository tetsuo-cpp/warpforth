\! kernel acceptance_scale
\! param N i64
\! param SCALE i64
\! param INPUT i64[4194304]
\! param OUTPUT i64[4194304]

\ ( -- ) The slow arm of the acceptance fixture. Same buffers, same launch, same answer, and a
\ 512-trip chain per element: each trip converts the running value to f64 and back to i64, so every
\ trip depends on the one before it and none can be folded away. That chain costs orders of
\ magnitude more per element than the load and store the fast arm is bound by, so the pair lands
\ well past the fixture's 2x floor. The measured ratio is recorded in the PR that added it.
GLOBAL-ID
BEGIN DUP N < WHILE
  DUP CELLS INPUT + @
  512 0 DO S>F F>S LOOP
  SCALE *
  OVER CELLS OUTPUT + !
  GDIM-X BDIM-X * +
REPEAT
DROP
