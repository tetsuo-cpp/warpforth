#!/usr/bin/env bash
# Read-only health check for verifying WarpForth. Run from anywhere.
#   doctor.sh           local checks (build, binaries, smoke compile)
#   doctor.sh --remote  also checks the Vast.ai key and lists warpforth-test-* instances
set -u
repo="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$repo" || exit 1
fail=0
ok() { printf 'ok    %s\n' "$1"; }
bad() { printf 'FAIL  %s\n' "$1"; fail=1; }

for bin in warpforthc warpforth-translate warpforth-opt; do
  if [ -x "build/bin/$bin" ]; then ok "build/bin/$bin exists"; else bad "build/bin/$bin missing (cmake --build build)"; fi
done

# ninja -n is a dry run: it reports pending work without building.
pending="$(ninja -C build -n 2>/dev/null | grep -c '^\[')"
if [ "$pending" = "0" ]; then ok "build is up to date with sources"; else bad "build is stale: $pending step(s) pending (cmake --build build)"; fi

smoke="$(mktemp -d)"
printf '\\! kernel main\n\\! param DATA i64[1]\n3 4 +\n0 CELLS DATA + !\n' > "$smoke/k.forth"
if build/bin/warpforthc "$smoke/k.forth" --arch sm_70 -o "$smoke/k.ptx" 2>"$smoke/err" \
   && grep -q '^\.visible \.entry main' "$smoke/k.ptx"; then
  ok "smoke compile produces a .entry main PTX kernel"
else
  bad "smoke compile failed: $(cat "$smoke/err")"
fi
rm -rf "$smoke"

if [ "${1:-}" = "--remote" ]; then
  command -v uv >/dev/null && ok "uv on PATH" || bad "uv missing"
  command -v ssh >/dev/null && command -v scp >/dev/null && ok "ssh/scp on PATH" || bad "ssh/scp missing"
  uv run --quiet python - <<'PY' || fail=1
import sys
sys.path.insert(0, ".claude/skills/verify-warpforth/scripts")
from gpu_run import load_api_key
from vastai import VastAI
key = load_api_key()
if not key:
    print("FAIL  VASTAI_API_KEY not set (env or .env)"); sys.exit(1)
try:
    insts = VastAI(key, retry=1).show_instances()
except Exception as e:
    print(f"FAIL  Vast.ai API call failed: {e}"); sys.exit(1)
print("ok    Vast.ai key accepted")
ours = [i for i in insts if str(i.get("label", "")).startswith("warpforth-test-")]
for i in ours:
    print(f"warn  existing instance id={i.get('id')} label={i.get('label')} "
          f"status={i.get('actual_status')} dph=${float(i.get('dph_total') or 0):.3f} "
          "(another run's, or an orphan: do not destroy unless its label is yours)")
if not ours:
    print("ok    no warpforth-test-* instances running")
PY
fi
exit $fail
