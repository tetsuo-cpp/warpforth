---
name: review-mlir
description: Reviews C++ and TableGen changes for idiomatic, version-appropriate MLIR usage and existing upstream functionality. Use when reviewing MLIR dialects, operations, rewrites, conversions, passes, analyses, parsers, printers, or translations.
---

# Review MLIR

Review MLIR code against the codebase's actual LLVM/MLIR revision. Find correctness issues first, then opportunities to replace custom machinery with supported MLIR or LLVM facilities.

## Review Contract

- Treat "modern" as idiomatic for the revision the codebase builds against, not automatically LLVM `main`.
- Verify API availability before recommending it. Do not propose an API from a newer release without identifying the compatibility constraint.
- Prefer the narrowest upstream abstraction that expresses the required semantics. Do not recommend replacement merely because an API has a similar name.
- Preserve semantics and intentional compatibility surfaces. Do not require incidental behavior, implementation quirks, or exact printed-IR formatting to remain unchanged when idiomatic MLIR can simplify them.
- Identify observable changes introduced by a cleanup and distinguish harmless canonicalization or presentation changes from semantic or compatibility changes that require a deliberate decision.
- Report only actionable findings supported by code or upstream documentation/source. Separate definite defects from optional improvements.
- Avoid broad style churn, speculative abstraction, and migration advice unrelated to the reviewed change.

## Workflow

### 1. Establish the compatibility baseline

Determine the LLVM/MLIR revision before judging API usage. Inspect, in order of relevance:

1. Dependency lockfiles, submodules, package manifests, toolchain files, and CI images.
2. CMake configuration and `find_package(MLIR ...)` constraints.
3. `llvm-config --version`, MLIR CMake package metadata, or installed headers when the build environment is authoritative.
4. Existing code patterns only as secondary evidence; they may themselves be legacy.

State the detected revision or release in the review. If it cannot be established, explicitly mark version-sensitive advice as conditional.

### 2. Understand the change in context

Read the complete diff and enough surrounding code to establish:

- the owning dialect, pass, conversion, analysis, or translation boundary;
- operation invariants and expected input/output IR;
- pass pipeline ordering and legality assumptions;
- whether textual IR, bytecode, generated APIs, or downstream users form an intentional compatibility surface rather than merely exposing incidental current behavior;
- tests that define current behavior.

Trace referenced helpers and generated definitions. Do not infer an operation's contract from one call site.

### 3. Search for existing functionality

For each nontrivial custom helper or algorithm introduced by the change, formulate the semantic job it performs, then search in this order:

1. Existing helpers and conventions in the local codebase.
2. The pinned MLIR and LLVM headers, sources, tests, and examples available locally.
3. Official documentation for the pinned release.
4. Upstream LLVM source and history when local sources are absent or when checking deprecation/replacement history.

Search by semantics as well as names. Check likely facilities including:

- operation traits, interfaces, declarative assembly formats, builders, folders, canonicalizers, and verifiers;
- `PatternRewriter`, rewrite patterns, dialect conversion, type converters, legality, and materializations;
- `OpBuilder`, `ImplicitLocOpBuilder`, `IRMapping`, `TypeSwitch`, symbol-table utilities, walk APIs, and region/block utilities;
- pass infrastructure, analyses, data-flow frameworks, transform infrastructure, and dialect interfaces;
- LLVM ADTs, casting, ranges, scope-exit, error handling, and debug facilities.

Confirm that an upstream facility satisfies the required semantics. Check ownership, mutation behavior, traversal order, failure behavior, diagnostics, location propagation, and asymptotic cost, but do not preserve differences that are incidental to the custom implementation. Document any meaningful behavior that the replacement intentionally changes.

### 4. Review MLIR invariants

Use [the detailed checklist](reference/review-checklist.md) selectively. Prioritize checks relevant to the changed code rather than mechanically commenting on every category.

Pay particular attention to:

- IR mutation through the correct rewriter and preservation of use-def, dominance, isolation, and symbol invariants;
- safe iteration when operations, blocks, or regions may be erased or replaced;
- precise pattern match failure with no mutation before a successful match is established;
- conversion legality and type-conversion materializations at every boundary;
- operation verification, parser/printer round trips, generated accessor use, and diagnostic quality;
- pass state, analysis invalidation, multithreading assumptions, and deterministic output.

### 5. Validate findings

Use the repository's documented build and test entry points. Prefer the smallest decisive checks first:

- compile the affected target to catch generated-API and version mismatches;
- run focused `lit` or unit tests for the changed pass/dialect;
- run parser/printer round-trip, verifier-negative, and pass-pipeline tests when applicable;
- run broader checks only when the blast radius warrants them.

When recommending an upstream API, verify it by inspecting its declaration and at least one representative upstream use or test for the pinned revision. A search-result name alone is insufficient evidence.

## Finding Format

Order findings by severity. For each finding provide:

1. **Severity and location** — exact file and line or changed construct.
2. **Observed issue** — what the current code does.
3. **Why it matters** — concrete failure mode, maintenance cost, or missed invariant.
4. **Recommended change** — smallest correct fix or upstream facility.
5. **Evidence** — relevant API declaration, upstream example/test, or executed check; include version applicability.

Use these categories:

- **Correctness**: can produce invalid IR, miscompile, crash, lose diagnostics, or violate a pass contract.
- **Compatibility**: uses an unavailable/deprecated API or unintentionally changes IR/API compatibility.
- **Idiomatic MLIR**: duplicates a supported abstraction or bypasses framework lifecycle/invariants.
- **Optional simplification**: cleanup with a clear payoff whose semantic and compatibility effects are understood, even if incidental output or implementation behavior changes.

If no actionable findings remain, say so directly and list the revision checked and validation performed. Do not invent low-value comments to fill the review.

## Applying Fixes

When asked to fix findings:

1. Preserve required semantics and intentional compatibility surfaces when replacing custom code with an upstream facility; simplify incidental behavior rather than recreating it.
2. Make intentional observable changes explicit and update brittle tests that encode incidental formatting or implementation details.
3. Add or adjust tests for the invariant that exposed the issue, not for implementation details.
4. Format changed C++ or TableGen using the repository's configured formatter.
