"""End-to-end GPU execution tests for the WarpForth compiler."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from conftest import KernelRunner

pytestmark = pytest.mark.gpu


# --- Arithmetic ---


def test_addition(kernel_runner: KernelRunner) -> None:
    """3 + 4 = 7."""
    result = kernel_runner.run(
        forth_source="PARAM DATA 256\n3 4 +\n0 CELLS DATA + !",
    )
    assert result[0] == 7


def test_subtraction(kernel_runner: KernelRunner) -> None:
    """10 - 3 = 7."""
    result = kernel_runner.run(
        forth_source="PARAM DATA 256\n10 3 -\n0 CELLS DATA + !",
    )
    assert result[0] == 7


def test_multiplication(kernel_runner: KernelRunner) -> None:
    """6 * 7 = 42."""
    result = kernel_runner.run(
        forth_source="PARAM DATA 256\n6 7 *\n0 CELLS DATA + !",
    )
    assert result[0] == 42


def test_division(kernel_runner: KernelRunner) -> None:
    """42 / 6 = 7."""
    result = kernel_runner.run(
        forth_source="PARAM DATA 256\n42 6 /\n0 CELLS DATA + !",
    )
    assert result[0] == 7


def test_modulo(kernel_runner: KernelRunner) -> None:
    """17 MOD 5 = 2."""
    result = kernel_runner.run(
        forth_source="PARAM DATA 256\n17 5 MOD\n0 CELLS DATA + !",
    )
    assert result[0] == 2


# --- Stack Manipulation ---


def test_dup(kernel_runner: KernelRunner) -> None:
    """DUP duplicates top of stack: 5 DUP → [5, 5]."""
    result = kernel_runner.run(
        forth_source=("PARAM DATA 256\n5 DUP\n1 CELLS DATA + !\n0 CELLS DATA + !"),
        output_count=2,
    )
    assert result == [5, 5]


def test_swap(kernel_runner: KernelRunner) -> None:
    """SWAP exchanges top two: 1 2 SWAP → [2, 1]."""
    result = kernel_runner.run(
        forth_source=("PARAM DATA 256\n1 2 SWAP\n1 CELLS DATA + !\n0 CELLS DATA + !"),
        output_count=2,
    )
    assert result == [2, 1]


def test_over(kernel_runner: KernelRunner) -> None:
    """OVER copies second element: 1 2 OVER → [1, 2, 1]."""
    result = kernel_runner.run(
        forth_source=(
            "PARAM DATA 256\n1 2 OVER\n2 CELLS DATA + !\n1 CELLS DATA + !\n0 CELLS DATA + !"
        ),
        output_count=3,
    )
    assert result == [1, 2, 1]


def test_rot(kernel_runner: KernelRunner) -> None:
    """ROT rotates top three: 1 2 3 ROT → [2, 3, 1]."""
    result = kernel_runner.run(
        forth_source=(
            "PARAM DATA 256\n1 2 3 ROT\n2 CELLS DATA + !\n1 CELLS DATA + !\n0 CELLS DATA + !"
        ),
        output_count=3,
    )
    assert result == [2, 3, 1]


def test_drop(kernel_runner: KernelRunner) -> None:
    """DROP removes top: 1 2 DROP → [1]."""
    result = kernel_runner.run(
        forth_source=("PARAM DATA 256\n1 2 DROP\n0 CELLS DATA + !"),
    )
    assert result[0] == 1


# --- Comparisons ---


def test_comparisons(kernel_runner: KernelRunner) -> None:
    """Test =, <, >, 0= in a single kernel. True = -1, False = 0."""
    result = kernel_runner.run(
        forth_source=(
            "PARAM DATA 256\n"
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
        forth_source=("PARAM DATA 256\n0 CELLS DATA + @\n0 >\nIF 1 ELSE 2 THEN\n1 CELLS DATA + !"),
        params={"DATA": [5]},
        output_count=2,
    )
    assert result[1] == 1


def test_begin_until(kernel_runner: KernelRunner) -> None:
    """BEGIN/UNTIL countdown: 10 BEGIN 1- DUP 0= UNTIL → final value is 0."""
    result = kernel_runner.run(
        forth_source=("PARAM DATA 256\n10 BEGIN 1 - DUP 0= UNTIL\n0 CELLS DATA + !"),
    )
    assert result[0] == 0


def test_do_loop(kernel_runner: KernelRunner) -> None:
    """DO/LOOP: write I values 0..4 to DATA[0..4]."""
    result = kernel_runner.run(
        forth_source=("PARAM DATA 256\n5 0 DO\n  I I CELLS DATA + !\nLOOP"),
        output_count=5,
    )
    assert result == [0, 1, 2, 3, 4]


def test_multi_while(kernel_runner: KernelRunner) -> None:
    """Multi-WHILE: two exit conditions from the same loop (interleaved CF).

    20 BEGIN DUP 10 > WHILE DUP 2 MOD 0= WHILE 1 - REPEAT THEN
    Decrements while >10 AND even. 20→19 (odd, WHILE(2) exit) → result 19.
    """
    result = kernel_runner.run(
        forth_source=(
            "PARAM DATA 256\n"
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
            "PARAM DATA 256\n10 BEGIN DUP 0 > WHILE 1 - DUP 5 = UNTIL THEN\n0 CELLS DATA + !"
        ),
    )
    assert result[0] == 5


# --- GPU Indexing ---


def test_global_id(kernel_runner: KernelRunner) -> None:
    """4 threads each write GLOBAL-ID to DATA[GLOBAL-ID]."""
    result = kernel_runner.run(
        forth_source=("PARAM DATA 256\nGLOBAL-ID\nDUP CELLS DATA + !"),
        block=(4, 1, 1),
        output_count=4,
    )
    assert result == [0, 1, 2, 3]


def test_multi_param(kernel_runner: KernelRunner) -> None:
    """Two params: each thread reads INPUT[i], doubles it, writes OUTPUT[i]."""
    result = kernel_runner.run(
        forth_source=(
            "PARAM INPUT 4\n"
            "PARAM OUTPUT 4\n"
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


# --- User-Defined Words ---


def test_user_defined_word(kernel_runner: KernelRunner) -> None:
    """: DOUBLE DUP + ; then 5 DOUBLE → 10."""
    result = kernel_runner.run(
        forth_source=("PARAM DATA 256\n: DOUBLE DUP + ;\n5 DOUBLE\n0 CELLS DATA + !"),
    )
    assert result[0] == 10
