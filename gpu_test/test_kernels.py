"""End-to-end GPU execution tests for the WarpForth compiler."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from conftest import KernelRunner

pytestmark = pytest.mark.gpu


# --- Arithmetic ---


def test_addition(kernel_runner: KernelRunner) -> None:
    """3 + 4 = 7."""
    result = kernel_runner.run(
        forth_source="\\! kernel main\n\\! param DATA i32[256]\n3 4 +\n0 CELLS DATA + !",
    )
    assert result[0] == 7


def test_subtraction(kernel_runner: KernelRunner) -> None:
    """10 - 3 = 7."""
    result = kernel_runner.run(
        forth_source="\\! kernel main\n\\! param DATA i32[256]\n10 3 -\n0 CELLS DATA + !",
    )
    assert result[0] == 7


def test_multiplication(kernel_runner: KernelRunner) -> None:
    """6 * 7 = 42."""
    result = kernel_runner.run(
        forth_source="\\! kernel main\n\\! param DATA i32[256]\n6 7 *\n0 CELLS DATA + !",
    )
    assert result[0] == 42


def test_division(kernel_runner: KernelRunner) -> None:
    """42 / 6 = 7."""
    result = kernel_runner.run(
        forth_source="\\! kernel main\n\\! param DATA i32[256]\n42 6 /\n0 CELLS DATA + !",
    )
    assert result[0] == 7


def test_modulo(kernel_runner: KernelRunner) -> None:
    """17 MOD 5 = 2."""
    result = kernel_runner.run(
        forth_source="\\! kernel main\n\\! param DATA i32[256]\n17 5 MOD\n0 CELLS DATA + !",
    )
    assert result[0] == 2


# --- Stack Manipulation ---


def test_dup(kernel_runner: KernelRunner) -> None:
    """DUP duplicates top of stack: 5 DUP → [5, 5]."""
    result = kernel_runner.run(
        forth_source=(
            "\\! kernel main\n\\! param DATA i32[256]\n5 DUP\n1 CELLS DATA + !\n0 CELLS DATA + !"
        ),
        output_count=2,
    )
    assert result == [5, 5]


def test_swap(kernel_runner: KernelRunner) -> None:
    """SWAP exchanges top two: 1 2 SWAP → [2, 1]."""
    result = kernel_runner.run(
        forth_source=(
            "\\! kernel main\n\\! param DATA i32[256]\n1 2 SWAP\n1 CELLS DATA + !\n0 CELLS DATA + !"
        ),
        output_count=2,
    )
    assert result == [2, 1]


def test_over(kernel_runner: KernelRunner) -> None:
    """OVER copies second element: 1 2 OVER → [1, 2, 1]."""
    result = kernel_runner.run(
        forth_source=(
            "\\! kernel main\n"
            "\\! param DATA i32[256]\n"
            "1 2 OVER\n"
            "2 CELLS DATA + !\n"
            "1 CELLS DATA + !\n"
            "0 CELLS DATA + !"
        ),
        output_count=3,
    )
    assert result == [1, 2, 1]


def test_rot(kernel_runner: KernelRunner) -> None:
    """ROT rotates top three: 1 2 3 ROT → [2, 3, 1]."""
    result = kernel_runner.run(
        forth_source=(
            "\\! kernel main\n"
            "\\! param DATA i32[256]\n"
            "1 2 3 ROT\n"
            "2 CELLS DATA + !\n"
            "1 CELLS DATA + !\n"
            "0 CELLS DATA + !"
        ),
        output_count=3,
    )
    assert result == [2, 3, 1]


def test_drop(kernel_runner: KernelRunner) -> None:
    """DROP removes top: 1 2 DROP → [1]."""
    result = kernel_runner.run(
        forth_source=("\\! kernel main\n\\! param DATA i32[256]\n1 2 DROP\n0 CELLS DATA + !"),
    )
    assert result[0] == 1


# --- Comparisons ---


def test_comparisons(kernel_runner: KernelRunner) -> None:
    """Test =, <, >, 0= in a single kernel. True = -1, False = 0."""
    result = kernel_runner.run(
        forth_source=(
            "\\! kernel main\n\\! param DATA i32[256]\n"
            "5 5 =  0 CELLS DATA + !\n"
            "3 5 <  1 CELLS DATA + !\n"
            "5 3 >  2 CELLS DATA + !\n"
            "0 0=   3 CELLS DATA + !"
        ),
        output_count=4,
    )
    assert result == [-1, -1, -1, -1]


# --- Control Flow ---


def test_if_else_then(kernel_runner: KernelRunner) -> None:
    """IF/ELSE/THEN: if DATA[0] > 0, write 1 to DATA[1], else write 2."""
    result = kernel_runner.run(
        forth_source=(
            "\\! kernel main\n"
            "\\! param DATA i32[256]\n"
            "0 CELLS DATA + @\n"
            "0 >\n"
            "IF 1 ELSE 2 THEN\n"
            "1 CELLS DATA + !"
        ),
        params={"DATA": [5]},
        output_count=2,
    )
    assert result[1] == 1


def test_begin_until(kernel_runner: KernelRunner) -> None:
    """BEGIN/UNTIL countdown: 10 BEGIN 1- DUP 0= UNTIL → final value is 0."""
    result = kernel_runner.run(
        forth_source=(
            "\\! kernel main\n\\! param DATA i32[256]\n10 BEGIN 1 - DUP 0= UNTIL\n0 CELLS DATA + !"
        ),
    )
    assert result[0] == 0


def test_do_loop(kernel_runner: KernelRunner) -> None:
    """DO/LOOP: write I values 0..4 to DATA[0..4]."""
    result = kernel_runner.run(
        forth_source=(
            "\\! kernel main\n\\! param DATA i32[256]\n5 0 DO\n  I I CELLS DATA + !\nLOOP"
        ),
        output_count=5,
    )
    assert result == [0, 1, 2, 3, 4]


def test_do_plus_loop(kernel_runner: KernelRunner) -> None:
    """DO/+LOOP: write I values 0, 2, 4, 6, 8 to DATA[0..4]."""
    result = kernel_runner.run(
        forth_source=(
            "\\! kernel main\n"
            "\\! param DATA i32[256]\n"
            "0\n"
            "10 0 DO\n"
            "  I OVER CELLS DATA + !\n"
            "  1 +\n"
            "2 +LOOP\n"
            "DROP"
        ),
        output_count=5,
    )
    assert result == [0, 2, 4, 6, 8]


def test_do_plus_loop_negative(kernel_runner: KernelRunner) -> None:
    """DO/+LOOP with negative step: count down from 10 to 1."""
    result = kernel_runner.run(
        forth_source=(
            "\\! kernel main\n"
            "\\! param DATA i32[256]\n"
            "0\n"
            "0 10 DO\n"
            "  I OVER CELLS DATA + !\n"
            "  1 +\n"
            "-1 +LOOP\n"
            "DROP"
        ),
        output_count=10,
    )
    assert result == [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]


def test_multi_while(kernel_runner: KernelRunner) -> None:
    """Multi-WHILE: two exit conditions from the same loop (interleaved CF).

    20 BEGIN DUP 10 > WHILE DUP 2 MOD 0= WHILE 1 - REPEAT THEN
    Decrements while >10 AND even. 20→19 (odd, WHILE(2) exit) → result 19.
    """
    result = kernel_runner.run(
        forth_source=(
            "\\! kernel main\n\\! param DATA i32[256]\n"
            "20 BEGIN DUP 10 > WHILE DUP 2 MOD 0= WHILE 1 - REPEAT THEN\n"
            "0 CELLS DATA + !"
        ),
    )
    assert result[0] == 19


def test_while_until(kernel_runner: KernelRunner) -> None:
    """WHILE+UNTIL: two different exit mechanisms from the same loop (interleaved CF).

    10 BEGIN DUP 0 > WHILE 1 - DUP 5 = UNTIL THEN
    Decrements while >0, stops early at 5. 10→9→…→5 (UNTIL exit) → result 5.
    """
    result = kernel_runner.run(
        forth_source=(
            "\\! kernel main\n"
            "\\! param DATA i32[256]\n"
            "10 BEGIN DUP 0 > WHILE 1 - DUP 5 = UNTIL THEN\n"
            "0 CELLS DATA + !"
        ),
    )
    assert result[0] == 5


# --- GPU Indexing ---


def test_global_id(kernel_runner: KernelRunner) -> None:
    """4 threads each write GLOBAL-ID to DATA[GLOBAL-ID]."""
    result = kernel_runner.run(
        forth_source=("\\! kernel main\n\\! param DATA i32[256]\nGLOBAL-ID\nDUP CELLS DATA + !"),
        block=(4, 1, 1),
        output_count=4,
    )
    assert result == [0, 1, 2, 3]


def test_multi_param(kernel_runner: KernelRunner) -> None:
    """Two params: each thread reads INPUT[i], doubles it, writes OUTPUT[i]."""
    result = kernel_runner.run(
        forth_source=(
            "\\! kernel main\n\\! param INPUT i32[4]\n"
            "\\! param OUTPUT i32[4]\n"
            "GLOBAL-ID\n"
            "DUP CELLS INPUT + @\n"
            "DUP +\n"
            "SWAP CELLS OUTPUT + !"
        ),
        params={"INPUT": [10, 20, 30, 40]},
        block=(4, 1, 1),
        output_param=1,
        output_count=4,
    )
    assert result == [20, 40, 60, 80]


def test_scalar_param(kernel_runner: KernelRunner) -> None:
    """Scalar + array params: each thread multiplies INPUT[i] by SCALE, writes OUTPUT[i]."""
    result = kernel_runner.run(
        forth_source=(
            "\\! kernel main\n\\! param SCALE i32\n"
            "\\! param INPUT i32[4]\n"
            "\\! param OUTPUT i32[4]\n"
            "GLOBAL-ID\n"
            "DUP CELLS INPUT + @\n"
            "SCALE *\n"
            "SWAP CELLS OUTPUT + !"
        ),
        params={"SCALE": 3, "INPUT": [10, 20, 30, 40]},
        block=(4, 1, 1),
        output_param=2,
        output_count=4,
    )
    assert result == [30, 60, 90, 120]


# --- Matmul ---


def test_naive_matmul_i32(kernel_runner: KernelRunner) -> None:
    """Naive i32 matmul: C = A(2x4) * B(4x3) -> C(2x3)."""
    # Work partition: one thread per output element.
    # GLOBAL-ID maps to (row, col) with row = gid / N, col = gid MOD N.
    result = kernel_runner.run(
        forth_source=(
            "\\! kernel main\n\\! param A i32[8]\n"
            "\\! param B i32[12]\n"
            "\\! param C i32[6]\n"
            "GLOBAL-ID\n"
            "DUP 3 /\n"
            "SWAP 3 MOD\n"
            "0\n"
            "4 0 DO\n"
            "2 PICK\n"
            "I SWAP 4 * +\n"
            "CELLS A + @\n"
            "I 3 * 3 PICK + CELLS B + @\n"
            "* +\n"
            "LOOP\n"
            "2 PICK 3 * 2 PICK +\n"
            "CELLS C + !"
        ),
        params={
            "A": [1, 2, 3, 4, 5, 6, 7, 8],
            "B": [1, 0, 2, 0, 1, 2, 1, 0, 1, 2, 1, 0],
        },
        block=(6, 1, 1),
        output_param=2,
        output_count=6,
    )
    assert result == [12, 6, 9, 28, 14, 29]


def test_tiled_matmul_i32(kernel_runner: KernelRunner) -> None:
    """Tiled i32 matmul with shared memory: C = A(4x4) * B(4x4) -> C(4x4).

    Uses 2x2 tiles, shared memory for A/B tiles, and BARRIER for sync.
    Grid: (2,2,1), Block: (2,2,1) — 4 blocks of 4 threads each.
    """
    result = kernel_runner.run(
        forth_source=(
            "\\! kernel main\n"
            "\\! param A i32[16]\n"
            "\\! param B i32[16]\n"
            "\\! param C i32[16]\n"
            "\\! shared SA i32[4]\n"
            "\\! shared SB i32[4]\n"
            "BID-Y 2 * TID-Y +\n"
            "BID-X 2 * TID-X +\n"
            "0\n"
            "2 0 DO\n"
            "  2 PICK 4 * I 2 * + TID-X + CELLS A + @\n"
            "  TID-Y 2 * TID-X + CELLS SA + S!\n"
            "  I 2 * TID-Y + 4 * 2 PICK + CELLS B + @\n"
            "  TID-Y 2 * TID-X + CELLS SB + S!\n"
            "  BARRIER\n"
            "  2 0 DO\n"
            "    TID-Y 2 * I + CELLS SA + S@\n"
            "    I 2 * TID-X + CELLS SB + S@\n"
            "    * +\n"
            "  LOOP\n"
            "  BARRIER\n"
            "LOOP\n"
            "ROT 4 * ROT + CELLS C + !"
        ),
        params={
            "A": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            "B": [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
        },
        grid=(2, 2, 1),
        block=(2, 2, 1),
        output_param=2,
        output_count=16,
    )
    expected = [
        250,
        260,
        270,
        280,
        618,
        644,
        670,
        696,
        986,
        1028,
        1070,
        1112,
        1354,
        1412,
        1470,
        1528,
    ]
    assert result == expected


def test_tiled_matmul_f32(kernel_runner: KernelRunner) -> None:
    """Tiled f32 matmul with shared memory: C = A(4x4) * B(4x4) -> C(4x4).

    Uses 2x2 tiles, float shared memory for A/B tiles, and BARRIER for sync.
    Grid: (2,2,1), Block: (2,2,1) — 4 blocks of 4 threads each.
    """
    result = kernel_runner.run(
        forth_source=(
            "\\! kernel main\n"
            "\\! param A f32[16]\n"
            "\\! param B f32[16]\n"
            "\\! param C f32[16]\n"
            "\\! shared SA f32[4]\n"
            "\\! shared SB f32[4]\n"
            "BID-Y 2 * TID-Y +\n"
            "BID-X 2 * TID-X +\n"
            "0.0\n"
            "2 0 DO\n"
            "  2 PICK 4 * I 2 * + TID-X + CELLS A + F@\n"
            "  TID-Y 2 * TID-X + CELLS SA + SF!\n"
            "  I 2 * TID-Y + 4 * 2 PICK + CELLS B + F@\n"
            "  TID-Y 2 * TID-X + CELLS SB + SF!\n"
            "  BARRIER\n"
            "  2 0 DO\n"
            "    TID-Y 2 * I + CELLS SA + SF@\n"
            "    I 2 * TID-X + CELLS SB + SF@\n"
            "    F* F+\n"
            "  LOOP\n"
            "  BARRIER\n"
            "LOOP\n"
            "ROT 4 * ROT + CELLS C + F!"
        ),
        params={
            "A": [1.5, 2.0, 0.5, 3.0, 4.0, 1.5, 2.5, 0.5, 0.5, 3.0, 1.0, 2.0, 2.0, 0.5, 3.5, 1.5],
            "B": [1.0, 0.5, 2.0, 1.5, 3.0, 1.0, 0.5, 2.0, 0.5, 2.5, 1.0, 0.5, 2.0, 1.5, 3.0, 1.0],
        },
        grid=(2, 2, 1),
        block=(2, 2, 1),
        output_param=2,
        output_count=16,
    )
    expected = [
        13.75,
        8.5,
        13.5,
        9.5,
        10.75,
        10.5,
        12.75,
        10.75,
        14.0,
        8.75,
        9.5,
        9.25,
        8.25,
        12.5,
        12.25,
        7.25,
    ]
    assert result == [pytest.approx(v) for v in expected]


# --- User-Defined Words ---


def test_user_defined_word(kernel_runner: KernelRunner) -> None:
    """: DOUBLE DUP + ; then 5 DOUBLE → 10."""
    result = kernel_runner.run(
        forth_source=(
            "\\! kernel main\n\\! param DATA i32[256]\n: DOUBLE DUP + ;\n5 DOUBLE\n0 CELLS DATA + !"
        ),
    )
    assert result[0] == 10


# --- Float Arithmetic ---


def test_float_addition(kernel_runner: KernelRunner) -> None:
    """F+: 1.5 + 2.5 = 4.0."""
    result = kernel_runner.run(
        forth_source="\\! kernel main\n\\! param DATA f32[256]\n1.5 2.5 F+\n0 CELLS DATA + F!",
    )
    assert result[0] == pytest.approx(4.0)


def test_float_subtraction(kernel_runner: KernelRunner) -> None:
    """F-: 10.0 - 3.5 = 6.5."""
    result = kernel_runner.run(
        forth_source="\\! kernel main\n\\! param DATA f32[256]\n10.0 3.5 F-\n0 CELLS DATA + F!",
    )
    assert result[0] == pytest.approx(6.5)


def test_float_multiplication(kernel_runner: KernelRunner) -> None:
    """F*: 6.0 * 7.5 = 45.0."""
    result = kernel_runner.run(
        forth_source="\\! kernel main\n\\! param DATA f32[256]\n6.0 7.5 F*\n0 CELLS DATA + F!",
    )
    assert result[0] == pytest.approx(45.0)


def test_float_division(kernel_runner: KernelRunner) -> None:
    """F/: 42.0 / 6.0 = 7.0."""
    result = kernel_runner.run(
        forth_source="\\! kernel main\n\\! param DATA f32[256]\n42.0 6.0 F/\n0 CELLS DATA + F!",
    )
    assert result[0] == pytest.approx(7.0)


# --- Float Memory ---


def test_float_load_store(kernel_runner: KernelRunner) -> None:
    """F@ and F!: read from DATA[0], multiply by 2, write to DATA[1]."""
    result = kernel_runner.run(
        forth_source=(
            "\\! kernel main\n\\! param DATA f32[256]\n0 CELLS DATA + F@\n2.0 F*\n1 CELLS DATA + F!"
        ),
        params={"DATA": [3.14]},
        output_count=2,
    )
    assert result[1] == pytest.approx(6.28)


# --- Float Scalar Params ---


def test_float_scalar_param(kernel_runner: KernelRunner) -> None:
    """Scalar f32 param: each thread scales DATA[i] by SCALE."""
    result = kernel_runner.run(
        forth_source=(
            "\\! kernel main\n\\! param DATA f32[256]\n\\! param SCALE f32\n"
            "GLOBAL-ID\n"
            "DUP CELLS DATA + F@\n"
            "SCALE F*\n"
            "SWAP CELLS DATA + F!"
        ),
        params={"DATA": [1.0, 2.0, 3.0, 4.0], "SCALE": 2.5},
        block=(4, 1, 1),
        output_count=4,
    )
    assert result == [
        pytest.approx(2.5),
        pytest.approx(5.0),
        pytest.approx(7.5),
        pytest.approx(10.0),
    ]


# --- Float Comparisons ---


def test_float_comparisons(kernel_runner: KernelRunner) -> None:
    """F=, F<, F>: True = -1, False = 0 (pushed as i32 on the stack)."""
    result = kernel_runner.run(
        forth_source=(
            "\\! kernel main\n\\! param DATA i32[256]\n"
            "3.14 3.14 F=  0 CELLS DATA + !\n"
            "1.0 2.0 F<    1 CELLS DATA + !\n"
            "5.0 3.0 F>    2 CELLS DATA + !"
        ),
        output_count=3,
    )
    assert result == [-1, -1, -1]


# --- Float Conversion ---


def test_int_to_float_conversion(kernel_runner: KernelRunner) -> None:
    """S>F: convert int 7 to float, multiply by 1.5, store as f32."""
    result = kernel_runner.run(
        forth_source=("\\! kernel main\n\\! param DATA f32[256]\n7 S>F 1.5 F*\n0 CELLS DATA + F!"),
    )
    assert result[0] == pytest.approx(10.5)


def test_float_to_int_conversion(kernel_runner: KernelRunner) -> None:
    """F>S: convert float 7.9 to int (truncates to 7), store as i32."""
    result = kernel_runner.run(
        forth_source=("\\! kernel main\n\\! param DATA i32[256]\n7.9 F>S\n0 CELLS DATA + !"),
    )
    assert result[0] == 7


# --- Attention ---

_ATTENTION_KERNEL = """\
\\! kernel attention
\\! param Q f32[{n}]
\\! param K f32[{n}]
\\! param V f32[{n}]
\\! param O f32[{n}]
\\! param SEQ_LEN i32
\\! param HEAD_DIM i32
\\! shared SCORES f32[{seq_len}]
\\! shared SCRATCH f32[{seq_len}]
BID-X
TID-X
0.0
HEAD_DIM 0 DO
  2 PICK HEAD_DIM * I + CELLS Q + F@
  2 PICK HEAD_DIM * I + CELLS K + F@
  F* F+
LOOP
HEAD_DIM S>F FSQRT F/
OVER 3 PICK >
IF DROP -1.0e30 THEN
OVER CELLS SCORES + SF!
BARRIER
TID-X 0= IF
  0 CELLS SCORES + SF@
  SEQ_LEN 1 DO I CELLS SCORES + SF@ FMAX LOOP
  0 CELLS SCRATCH + SF!
THEN
BARRIER
DUP CELLS SCORES + SF@
0 CELLS SCRATCH + SF@
F- FEXP
OVER CELLS SCORES + SF!
BARRIER
TID-X 0= IF
  0.0
  SEQ_LEN 0 DO I CELLS SCORES + SF@ F+ LOOP
  0 CELLS SCRATCH + SF!
THEN
BARRIER
DUP CELLS SCORES + SF@
0 CELLS SCRATCH + SF@
F/
OVER CELLS SCORES + SF!
BARRIER
DUP BEGIN DUP HEAD_DIM < WHILE
  0.0
  SEQ_LEN 0 DO
    I CELLS SCORES + SF@
    I HEAD_DIM * 3 PICK + CELLS V + F@
    F* F+
  LOOP
  OVER 4 PICK HEAD_DIM * + CELLS O + F!
  BDIM-X +
REPEAT
DROP DROP DROP
"""


def _attention_reference(q: np.ndarray, k: np.ndarray, v: np.ndarray, seq_len: int) -> list[float]:
    """Compute scaled dot-product attention with causal mask (NumPy reference, f32)."""
    q = q.astype(np.float32)
    k = k.astype(np.float32)
    v = v.astype(np.float32)
    head_dim = q.shape[1]
    scores = q @ k.T / np.sqrt(np.float32(head_dim))
    causal_mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    scores[causal_mask] = np.float32(-1e30)
    exp_scores = np.exp(scores - scores.max(axis=1, keepdims=True))
    attn = exp_scores / exp_scores.sum(axis=1, keepdims=True)
    return (attn @ v).flatten().tolist()


def test_naive_attention_f32(kernel_runner: KernelRunner) -> None:
    """Naive scaled dot-product attention with causal mask.

    O = softmax(Q @ K^T / sqrt(d_k)) @ V, seq_len=4, head_dim=4.
    One block per query row, one thread per key position.
    """
    seq_len, head_dim = 4, 4

    q = np.array(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )
    k = np.array(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )
    v = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]
    )

    expected = _attention_reference(q, k, v, seq_len)
    n = seq_len * head_dim

    result = kernel_runner.run(
        forth_source=_ATTENTION_KERNEL.format(n=n, seq_len=seq_len),
        params={
            "Q": q.flatten().tolist(),
            "K": k.flatten().tolist(),
            "V": v.flatten().tolist(),
            "SEQ_LEN": seq_len,
            "HEAD_DIM": head_dim,
        },
        grid=(seq_len, 1, 1),
        block=(seq_len, 1, 1),
        output_param=3,
        output_count=n,
    )
    assert result == [pytest.approx(v, rel=1e-4) for v in expected]


def test_naive_attention_f32_16x64(kernel_runner: KernelRunner) -> None:
    """Naive scaled dot-product attention, seq_len=16, head_dim=64."""
    seq_len, head_dim = 16, 64

    rng = np.random.default_rng(42)
    q = rng.standard_normal((seq_len, head_dim))
    k = rng.standard_normal((seq_len, head_dim))
    v = rng.standard_normal((seq_len, head_dim))

    expected = _attention_reference(q, k, v, seq_len)
    n = seq_len * head_dim

    result = kernel_runner.run(
        forth_source=_ATTENTION_KERNEL.format(n=n, seq_len=seq_len),
        params={
            "Q": q.flatten().tolist(),
            "K": k.flatten().tolist(),
            "V": v.flatten().tolist(),
            "SEQ_LEN": seq_len,
            "HEAD_DIM": head_dim,
        },
        grid=(seq_len, 1, 1),
        block=(seq_len, 1, 1),
        output_param=3,
        output_count=n,
    )
    assert result == [pytest.approx(v, rel=1e-3) for v in expected]
