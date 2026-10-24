# MLIR Review Checklist

Use only the sections relevant to the change. This checklist supplements direct reasoning; it is not a substitute for understanding the IR contract.

## Operations and ODS

- Could a trait or interface express and verify a structural invariant now implemented manually?
- Are arguments, results, regions, successors, properties, and attributes represented declaratively where practical?
- Does the operation use generated typed accessors instead of string-based attribute lookup or positional operand/result access where generated APIs exist?
- Are builders minimal and unambiguous? Do inferred result types use an inference interface when appropriate?
- Is custom parsing/printing necessary, or can declarative assembly format express the syntax?
- If custom assembly is needed, does it round-trip all semantic state, handle optional/default values appropriately, and emit useful parse errors? Do not require incidental whitespace, ordering, or spelling choices to remain identical unless they are a declared compatibility surface.
- Are verifier checks split appropriately between structural verification and region verification? Do diagnostics identify the violated invariant and relevant value/type?
- Are fold and canonicalization rules terminating, semantics-preserving, and placed on the operation that owns the invariant?
- Do traits such as purity, memory effects, speculatability, commutativity, and terminator/successor traits accurately describe behavior?

## Builders, Locations, and Diagnostics

- Is the insertion point explicit and valid, including for nested region construction?
- Would `ImplicitLocOpBuilder` reduce repeated locations without hiding meaningful source locations?
- Are source locations preserved or fused rather than replaced with unknown locations?
- Are diagnostics emitted through the operation/context facilities and attached to the most relevant operation or location?
- Are expected non-matches returned as failure rather than emitted as errors?
- Are unchecked casts and `getDefiningOp` assumptions justified by verified invariants?

## Rewrites and Canonicalization

- Does all mutation occur through the provided rewriter when running inside a rewrite framework?
- Is the IR left untouched when `match` fails? If mutation is needed during matching, is it transactional or deferred?
- Are in-place modifications wrapped with the rewriter's modification protocol?
- Are replacements type-compatible, and are all result uses handled deliberately?
- Is erasure safe with respect to remaining uses and iteration invalidation?
- Can the pattern reapply forever or increase IR without a bounded-recursion declaration and proof?
- Is pattern benefit meaningful, or is correctness accidentally dependent on application order?
- Would a fold, declarative rewrite, interface, or canonicalization pattern be a smaller ownership boundary?

## Dialect Conversion and Type Conversion

- Are legal and illegal dialects/operations/types specified precisely enough that conversion cannot silently leave unsupported IR?
- Is dynamic legality based on the post-conversion invariant rather than merely operation names?
- Does each pattern convert operands, results, region arguments, successor operands, and nested signatures that cross the boundary?
- Are source, target, and argument materializations defined only where genuinely needed and guaranteed to be reconciled?
- Are signature conversion and region type conversion performed with framework utilities rather than manual block surgery?
- Is one-to-many type conversion handled by facilities available in the pinned revision instead of ad hoc packing?
- Does partial versus full conversion match the pass contract?
- Are unrealized conversion casts reconciled, and do tests ensure none leak unexpectedly?

## Regions, Blocks, Symbols, and SSA

- Are block arguments, terminators, successors, and region kinds valid after every transformation?
- Is dominance preserved when moving or cloning operations? Is `IRMapping` used for cloning/remapping SSA values and blocks?
- Are operation walks safe when callbacks erase, replace, or move visited operations?
- Are symbol lookup and mutation performed with symbol-table APIs, respecting visibility, nesting, and `IsolatedFromAbove`?
- Are symbol references updated when symbols are renamed or moved?
- Is region isolation respected when capturing values or creating nested operations?

## Passes and Analyses

- Does the pass operate at the narrowest appropriate operation type?
- Is mutable per-run state local or reset, so a pass instance is safe under pass-manager reuse and multithreading?
- Are dependent dialects declared, and are pass options/statistics/debug output registered through pass infrastructure?
- Are analyses requested rather than recomputed manually? Are preserved analyses declared only when truly preserved?
- Does failure call `signalPassFailure()` after emitting a useful diagnostic?
- Is traversal deterministic where output order is observable?
- Does the pass avoid global mutable state and context mutation during parallel execution?
- Are pipeline prerequisites explicit rather than accidental assumptions about earlier passes?

## Types, Attributes, and Storage

- Are types and attributes uniqued through the context rather than manually allocated or cached?
- Are storage keys complete, immutable, and consistent between equality and construction?
- Is mutable storage actually required and synchronized according to MLIR's context rules?
- Are parse/print/verify hooks consistent and round-trip tested?
- Could a builtin type/attribute, data-layout query, or existing dialect interface replace custom representation logic?

## Interfaces and Effects

- Would an operation, dialect, type, or attribute interface decouple behavior from concrete operation-name switches?
- Are external models appropriate when behavior belongs to a consumer rather than the defining dialect?
- Are memory effects complete enough for dead-code elimination, scheduling, and alias-sensitive transforms?
- Are side-effect-free or speculatable claims valid for all operands, attributes, and nested regions?
- Is callability, branch behavior, region control flow, or loop behavior modeled through the corresponding interface when consumers need it?

## Performance and Ownership

- Is there repeated IR scanning where an analysis, symbol table, dominance structure, or listener can provide the information?
- Are expensive objects and analyses reused within their valid lifetime without stale caches?
- Are `SmallVector`, `ArrayRef`, `MutableArrayRef`, ranges, and `DenseMap`/`DenseSet` used with valid lifetimes and suitable ownership?
- Do callbacks capture values with sufficient lifetime, especially in deferred rewrite or pass-pipeline construction?
- Does a proposed upstream replacement preserve complexity and memory behavior on large IR?

## Testing

- Is there a positive test for the intended transformation and focused negative tests for rejected forms?
- Do verifier tests use expected diagnostics and test malformed IR at the correct abstraction boundary?
- Do transformation tests check semantic structure rather than incidental SSA names or unstable printing details?
- Is custom syntax round-trip tested for semantic equivalence rather than byte-for-byte reproduction?
- Are type-conversion boundaries, regions, symbols, and multi-result cases represented?
- Is test coverage pinned to behavior rather than a private helper implementation?
