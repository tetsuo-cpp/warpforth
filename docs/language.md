# WarpForth Language Reference

## Kernel Header

Every WarpForth program begins with header directives prefixed by `\!`. The header is required because Forth's stack-passing convention doesn't provide typed parameter declarations for the kernel interface. It's also used to capture information about the kernel that is better specified in a declarative manner rather than with Forth semantics, such as what shared memory buffers are used by the kernel.

### Kernel Declaration

```forth
\! kernel main
```

Required. Must appear first. Names the GPU kernel entry point.

### Parameters

```forth
\! param DATA i64[256]     \ array of 256 i64 → memref<256xi64>
\! param N i64             \ scalar i64
\! param WEIGHTS f64[128]  \ array of 128 f64 → memref<128xf64>
\! param SCALE f64         \ scalar f64 (bitcast to i64 on stack)
```

- Array parameters become `memref` arguments. Using the name as a word pushes the base address.
- Scalar parameters become value arguments. Using the name as a word pushes the value.
- `f64` scalars are bitcast to i64 when pushed to the stack; use `F`-prefixed words to operate on them.

### Shared Memory

```forth
\! shared SCRATCH i64[64]
\! shared SCORES f64[1024]
```

Declares GPU shared memory. Using the name as a word pushes its base address. Access with `S@`/`S!` (i64) or `SF@`/`SF!` (f64). Cannot be referenced inside word definitions.

## Literals

### Integer Literals

Plain numbers are parsed as i64:

```forth
42 -1 0 255
```

### Float Literals

Numbers containing `.` or `e`/`E` are parsed as f64 and stored on the stack as i64 bit patterns:

```forth
3.14 -2.0 1.0e-5 1e3
```

Use `F`-prefixed words (`F+`, `F*`, etc.) to operate on float values.

## Stack Operations

| Word | Stack Effect | Description |
|------|-------------|-------------|
| `DUP` | `( a -- a a )` | Duplicate top |
| `DROP` | `( a -- )` | Discard top |
| `SWAP` | `( a b -- b a )` | Swap top two |
| `OVER` | `( a b -- a b a )` | Copy second to top |
| `ROT` | `( a b c -- b c a )` | Rotate third to top |
| `NIP` | `( a b -- b )` | Drop second |
| `TUCK` | `( a b -- b a b )` | Copy top below second |
| `PICK` | `( xn ... x0 n -- xn ... x0 xn )` | Copy nth item to top |
| `ROLL` | `( xn ... x0 n -- xn-1 ... x0 xn )` | Move nth item to top |

## Arithmetic

### Integer Arithmetic

| Word | Stack Effect | Description |
|------|-------------|-------------|
| `+` | `( a b -- a+b )` | Add |
| `-` | `( a b -- a-b )` | Subtract |
| `*` | `( a b -- a*b )` | Multiply |
| `/` | `( a b -- a/b )` | Divide |
| `MOD` | `( a b -- a%b )` | Modulo |

### Float Arithmetic

| Word | Stack Effect | Description |
|------|-------------|-------------|
| `F+` | `( a b -- a+b )` | Float add |
| `F-` | `( a b -- a-b )` | Float subtract |
| `F*` | `( a b -- a*b )` | Float multiply |
| `F/` | `( a b -- a/b )` | Float divide |

### Float Math Intrinsics

| Word | Stack Effect | Description |
|------|-------------|-------------|
| `FEXP` | `( a -- exp(a) )` | Exponential |
| `FSQRT` | `( a -- sqrt(a) )` | Square root |
| `FLOG` | `( a -- log(a) )` | Natural logarithm |
| `FABS` | `( a -- |a| )` | Absolute value |
| `FNEG` | `( a -- -a )` | Negate |
| `FMAX` | `( a b -- max(a,b) )` | Maximum |
| `FMIN` | `( a b -- min(a,b) )` | Minimum |

## Bitwise Operations

| Word | Stack Effect | Description |
|------|-------------|-------------|
| `AND` | `( a b -- a&b )` | Bitwise AND |
| `OR` | `( a b -- a\|b )` | Bitwise OR |
| `XOR` | `( a b -- a^b )` | Bitwise XOR |
| `NOT` | `( a -- ~a )` | Bitwise NOT |
| `LSHIFT` | `( a n -- a<<n )` | Left shift |
| `RSHIFT` | `( a n -- a>>n )` | Right shift |

## Comparison

### Integer Comparison

All comparisons push 1 (true) or 0 (false).

| Word | Stack Effect | Description |
|------|-------------|-------------|
| `=` | `( a b -- flag )` | Equal |
| `<` | `( a b -- flag )` | Less than |
| `>` | `( a b -- flag )` | Greater than |
| `<>` | `( a b -- flag )` | Not equal |
| `<=` | `( a b -- flag )` | Less or equal |
| `>=` | `( a b -- flag )` | Greater or equal |
| `0=` | `( a -- flag )` | Equal to zero |

### Float Comparison

| Word | Stack Effect | Description |
|------|-------------|-------------|
| `F=` | `( a b -- flag )` | Float equal |
| `F<` | `( a b -- flag )` | Float less than |
| `F>` | `( a b -- flag )` | Float greater than |
| `F<>` | `( a b -- flag )` | Float not equal |
| `F<=` | `( a b -- flag )` | Float less or equal |
| `F>=` | `( a b -- flag )` | Float greater or equal |

## Type Conversion

| Word | Stack Effect | Description |
|------|-------------|-------------|
| `S>F` | `( n -- f )` | Integer to float (i64 → f64 bit pattern) |
| `F>S` | `( f -- n )` | Float to integer (f64 bit pattern → i64) |

## Memory Access

### Address Arithmetic

| Word | Stack Effect | Description |
|------|-------------|-------------|
| `CELLS` | `( n -- n*8 )` | Convert cell index to byte offset (8 bytes per cell) |

### Global Memory (i64)

| Word | Stack Effect | Description |
|------|-------------|-------------|
| `@` | `( addr -- value )` | Load i64 from global memory |
| `!` | `( value addr -- )` | Store i64 to global memory |

### Global Memory (f64)

| Word | Stack Effect | Description |
|------|-------------|-------------|
| `F@` | `( addr -- value )` | Load f64 from global memory (as i64 bit pattern) |
| `F!` | `( value addr -- )` | Store f64 to global memory (from i64 bit pattern) |

### Shared Memory (i64)

| Word | Stack Effect | Description |
|------|-------------|-------------|
| `S@` | `( addr -- value )` | Load i64 from shared memory |
| `S!` | `( value addr -- )` | Store i64 to shared memory |

### Shared Memory (f64)

| Word | Stack Effect | Description |
|------|-------------|-------------|
| `SF@` | `( addr -- value )` | Load f64 from shared memory (as i64 bit pattern) |
| `SF!` | `( value addr -- )` | Store f64 to shared memory (from i64 bit pattern) |

### Reduced-Width Memory

These words load/store narrower types, converting to/from the stack's native i64.

**Integer types** — load sign-extends to i64, store truncates from i64:

| Word | Width | Memory | Description |
|------|-------|--------|-------------|
| `I8@` / `I8!` | 8-bit | Global | Load/store i8 |
| `SI8@` / `SI8!` | 8-bit | Shared | Load/store i8 (shared) |
| `I16@` / `I16!` | 16-bit | Global | Load/store i16 |
| `SI16@` / `SI16!` | 16-bit | Shared | Load/store i16 (shared) |
| `I32@` / `I32!` | 32-bit | Global | Load/store i32 |
| `SI32@` / `SI32!` | 32-bit | Shared | Load/store i32 (shared) |

**Float types** — load extends to f64 then bitcasts to i64, store bitcasts i64 to f64 then truncates:

| Word | Width | Memory | Description |
|------|-------|--------|-------------|
| `HF@` / `HF!` | 16-bit | Global | Load/store f16 |
| `SHF@` / `SHF!` | 16-bit | Shared | Load/store f16 (shared) |
| `BF@` / `BF!` | 16-bit | Global | Load/store bf16 |
| `SBF@` / `SBF!` | 16-bit | Shared | Load/store bf16 (shared) |
| `F32@` / `F32!` | 32-bit | Global | Load/store f32 |
| `SF32@` / `SF32!` | 32-bit | Shared | Load/store f32 (shared) |

## Control Flow

### Conditionals

```forth
condition IF
  \ executed when condition is nonzero
THEN

condition IF
  \ true branch
ELSE
  \ false branch
THEN
```

### Post-Test Loop

```forth
BEGIN
  \ loop body
condition UNTIL   \ exits when condition is nonzero
```

### Pre-Test Loop

```forth
BEGIN condition WHILE
  \ loop body
REPEAT
```

### Counted Loop

```forth
limit start DO
  \ loop body — I is the current index
LOOP

limit start DO
  \ loop body
n +LOOP   \ increment index by n instead of 1
```

| Word | Description |
|------|-------------|
| `I` | Current loop index (innermost loop) |
| `J` | Index of next outer loop |
| `K` | Index of second outer loop |
| `LEAVE` | Exit the innermost loop immediately |
| `UNLOOP` | Discard loop parameters before `EXIT` |
| `EXIT` | Return from the current word |

## User-Defined Words

```forth
: square  DUP * ;
: add3  + + ;
```

### Local Variables

```forth
: dot-product { a-addr b-addr n -- }
  0
  n 0 DO
    I CELLS a-addr + @
    I CELLS b-addr + @
    * +
  LOOP
;
```

`{ name1 name2 ... -- }` at the start of a word definition binds read-only locals. Values are popped from the stack in reverse name order. Locals work across all control flow structures and map directly to GPU registers.

## GPU Operations

### Thread and Block Indexing

| Word | Stack Effect | Description |
|------|-------------|-------------|
| `TID-X` | `( -- id )` | Thread index in X dimension |
| `TID-Y` | `( -- id )` | Thread index in Y dimension |
| `TID-Z` | `( -- id )` | Thread index in Z dimension |
| `BID-X` | `( -- id )` | Block index in X dimension |
| `BID-Y` | `( -- id )` | Block index in Y dimension |
| `BID-Z` | `( -- id )` | Block index in Z dimension |
| `BDIM-X` | `( -- dim )` | Block dimension in X |
| `BDIM-Y` | `( -- dim )` | Block dimension in Y |
| `BDIM-Z` | `( -- dim )` | Block dimension in Z |
| `GDIM-X` | `( -- dim )` | Grid dimension in X |
| `GDIM-Y` | `( -- dim )` | Grid dimension in Y |
| `GDIM-Z` | `( -- dim )` | Grid dimension in Z |
| `GLOBAL-ID` | `( -- id )` | `BID-X * BDIM-X + TID-X` |

### Synchronization

| Word | Stack Effect | Description |
|------|-------------|-------------|
| `BARRIER` | `( -- )` | Thread block barrier (`__syncthreads`) |
