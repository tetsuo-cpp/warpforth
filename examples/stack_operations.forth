\ Stack operations demonstration
\ This shows DUP, SWAP, and DROP

10          ( push 10 )
dup         ( duplicate: 10 10 )
20          ( push 20: 10 10 20 )
swap        ( swap top two: 10 20 10 )
+           ( add top two: 10 30 )
drop        ( drop top: 10 )
