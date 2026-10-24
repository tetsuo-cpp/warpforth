#!/usr/bin/env bash
# Compile a Forth file the way a user does, keeping every stage as evidence.
#   compile.sh <kernel.forth> [arch]     (arch defaults to sm_70)
# Writes build/verify-evidence/<run-id>/{kernel.forth,kernel.mlir,lowered.mlir,
# kernel.ptx,warpforthc.stderr,rc} and prints the directory. Exit code is warpforthc's.
set -u
repo="$(cd "$(dirname "$0")/../../../.." && pwd)"
src="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
arch="${2:-sm_70}"
out="$repo/build/verify-evidence/$(date -u +%Y%m%dT%H%M%SZ)-compile-$(basename "$src" .forth)"
mkdir -p "$out"
cp "$src" "$out/kernel.forth"
bin="$repo/build/bin"

"$bin/warpforthc" "$src" --arch "$arch" -o "$out/kernel.ptx" 2>"$out/warpforthc.stderr"
rc=$?
echo "$rc" > "$out/rc"
if [ "$rc" -eq 0 ]; then
  # Intermediate stages for inspection; warpforthc's PTX above is the user-facing result.
  "$bin/warpforth-translate" --forth-to-mlir "$src" > "$out/kernel.mlir" 2>&1
  "$bin/warpforth-opt" \
    --pass-pipeline="builtin.module(warpforth-pipeline{chip=$arch})" \
    "$out/kernel.mlir" > "$out/lowered.mlir" 2>&1
fi
echo "evidence: $out (rc=$rc)"
exit "$rc"
