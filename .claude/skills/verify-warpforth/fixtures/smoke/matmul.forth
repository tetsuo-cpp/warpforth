\! kernel main
\! param A i64[8]
\! param B i64[12]
\! param C i64[6]

\ README quickstart: C = A(2x4) * B(4x3), one thread per output element.
GLOBAL-ID
DUP 3 /
SWAP 3 MOD
0
4 0 DO
  2 PICK
  I SWAP 4 * +
  CELLS A + @
  I 3 * 3 PICK + CELLS B + @
  * +
LOOP
2 PICK 3 * 2 PICK +
CELLS C + !
