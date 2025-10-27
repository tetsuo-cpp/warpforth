\ Comprehensive test of all supported operations
\ This file demonstrates every operation in the Forth dialect

( Constants and basic arithmetic )
10 20 +     ( addition: result 30 )
100 25 -    ( subtraction: result 75 )
6 7 *       ( multiplication: result 42 )
84 2 /      ( division: result 42 )

( Stack manipulation )
42          ( push constant )
dup         ( duplicate top value )
+           ( add duplicates: 84 )

15 30       ( push two values )
swap        ( swap them )
-           ( subtract: 30 - 15 = 15 )

999         ( push a value )
drop        ( drop it )

( Negative numbers )
-10 5 +     ( -10 + 5 = -5 )
