# Future Work / Not Implemented

## Phase 7: 64-bit Instruction Support (Deferred)
The original plan included 64-bit instruction support using `RVInst64` and a `getInstruction64()` method. This was deferred because:
- The current disassembler returns `Fail` for 64-bit instructions (line 851-853 in `RISCVDisassembler.cpp`)
- No infrastructure exists for 64-bit custom instruction decoding
- The 32-bit instruction set covers the core functionality

To implement later:
1. Add `getInstruction64()` method to `RISCVDisassembler` (modeled after `getInstruction48()`)
2. Define `THRVMInst64` base class using `RVInst64`
3. Add 64-bit decoder list and dispatch in `getInstruction()`
4. Define 64-bit instruction variants

## Register Allocation Flexibility (IMPLEMENTED)
All intrinsics and builtins now accept register index parameters (ImmArg), replacing the previous hardcoded fixed-register convention. This is COMPLETE:

**What was done**:
- Every intrinsic/builtin accepts an ImmArg register index specifying which matrix register to use (e.g., `__builtin_riscv_th_mld(base, stride, regidx)`)
- Register index constants defined: `__RVM_TR0` (0) through `__RVM_TR3` (3), `__RVM_ACC0` (4) through `__RVM_ACC3` (7)
- Sema constraint validation ensures register indices are valid per instruction category (e.g., matmul operands must use the correct register roles; EW arithmetic requires acc registers)
- Multi-accumulator support: users can now target different accumulator/tile registers in the same function, enabling concurrent matrix operations
- ISel `selectTHMatrix()` reads register index from intrinsic arguments and maps to the correct physical register

**Bug fix**: `mzero` was previously hardcoded to target tr0, which is incorrect per the spec (mzero should target accumulator registers). Now correctly uses the register index parameter to target acc0 (or any specified accumulator).

**Remaining future work**:
- Introduce pseudo-instructions with virtual matrix register operands that resolve during register allocation (full register allocator integration)

## Higher-level Matrix API (IMPLEMENTED - Phase A/B/C COMPLETE)
The `<thead_matrix.h>` header is now available with 414 API functions/macros implementing the RVM Intrinsic API Reference Manual v0.2. Uses native `__rvm_*_t` built-in types (Phase B/C complete).

### Phase B: Native Built-in Types (IMPLEMENTED)
22 first-class Clang built-in types have been implemented following the RVV pattern.

**Types**: 11 single-register (`__rvm_int8_t` through `__rvm_float64_t`) + 11 pair (`__rvm_int8x2_t` through `__rvm_float64x2_t`)

**Key implementation details**:
- New file: `clang/include/clang/Basic/RISCVMatrixTypes.def` with 22 type entries
- ~27 Clang source files modified (Headers, AST, CodeGen, Sema, Serialization, Misc)
- Key changes: TargetInfo (hasRISCVMatrixTypes), ASTContext, Type.cpp, CodeGenTypes, CGDebugInfo, Sema, Serialization, ItaniumMangle
- Types are opaque: Width=0, Align=8, map to `llvm::TargetExtType("riscv.matrix")` in CodeGen
- Feature-gated via `checkRVMTypeSupport` requiring `"experimental-xtheadmatrix"`
- `NumOfBuiltinTypeBits` bumped 9 to 10, `NUM_PREDEF_TYPE_IDS` bumped 514 to 560
- Test: `clang/test/CodeGen/RISCV/thead-matrix-builtin-types.c` verifies all 22 types compile

### Phase C: Typed Builtins (IMPLEMENTED + AUDITED)
121 low-level builtins now accept and return native `__rvm_*_t` matrix types, enabling Sema-level type checking at both the builtin and API levels. 22 `mundef` builtins create undefined matrix values. The `<thead_matrix.h>` header uses native built-in type aliases.

**Key implementation details**:
- Type encoding: `Qm<N>` in `ASTContext.cpp` DecodeTypeFromStr (22 entries mapping to CanQualType singletons)
- Tablegen: `ClangBuiltinsEmitter.cpp` StringSwitch maps `__rvm_*_t` names to `Qm<N>` encoding
- CGBuiltin: `IsTypedMatrixBuiltin` flag on 121 cases; post-switch handler filters TargetExtType args, calls void intrinsic, returns PoisonValue
- TargetExtType `riscv.matrix`: HasZeroInit + CanBeLocal properties (no ISel lowering needed)
- Load builtins return `__rvm_int*_t` (by EEW); unsigned/FP header wrappers use mundef for type conversion
- EW MVI typed builtins: `IntrinsicTypes = {Ops[5]->getType()}` (imm arg at index 5, after 3 reg indices + 2 matrix args)
- Matrix types cannot cross function boundaries (no ISel support for TargetExtType in function params)
- Files: Type.cpp, ASTContext.cpp, ClangBuiltinsEmitter.cpp, BuiltinsRISCVXTHeadMatrix.td, RISCV.cpp (CGBuiltin), thead_matrix.h
- Tests: 16 total (11 Clang CodeGen + 1 Sema + 4 LLVM), all passing

**Audit fixes applied**:
- N4clip signed variants (`mn4clipl_w_mm`, `mn4cliph_w_mm`) now correctly return `mint8_t` (was `muint8_t`)
- Reinterpret cast functions now accept a source parameter per spec (changed from `void()` to macro `(src)`)
- 4 new test files added (2,193 lines, 91 test functions) for comprehensive coverage

### Spec-API (ManagedRA) Coverage (IMPLEMENTED — 2026-03-04)

The spec-API surface has been expanded to cover:
- **All 11 types** (i8/i16/i32/i64/u8/u16/u32/u64/f16/f32/f64) for load/store/zero
- **B-tile loads** (`mld_b_spec_*` → `mlbe_internal`, sets K/N dimensions)
- **All matmul variants**: INT8→INT32 (4 sign), INT16→INT64 (4 sign), partial (4 sign), bypass (2), FP native (h/s/d), widening typed (s_h, d_s), widening opaque (8 FP8/BF16/TF32 variants)
- **Bug fix**: Matmul operand swap (A→ms1, B→ms2) for correct non-commutative behavior
- **Bug fix**: Conversion pseudo register classes (THRVMMR → THRVMACC)

See `13-verification-and-fixes.md` for full details.

### Further API Improvements
- Provide tiling and loop-nest abstractions for common GEMM patterns
- Auto-select matmul variants based on data type and accumulation semantics
- Integrate with MLIR linalg or similar frameworks for automatic matrix tiling
- Generic `__riscv_th_mld` with compiler register allocation (requires register allocator support)

## Current Limitations (as of 2026-03-04)

1. **No 64-bit instruction format**: `inst64_format.adoc` defines extended formats; not implemented.
2. **Matrix types cannot cross function boundaries**: No ABI support for `target("riscv.matrix")` in function params/returns.
3. **DirectReg typed builtins use PoisonValue**: Matrix SSA values are opaque tokens; actual data flows through physical registers.
4. **No auto-matmul from C loops**: Users must explicitly use builtins or inline assembly.
5. **All matrix registers reserved in DirectReg mode**: 8 registers removed from allocator pool.
6. **Limited register file (4+4)**: High register pressure for complex multi-tile kernels.
7. **Whole-register spill granularity**: No partial-register spill optimization (8192-bit spills).
8. **No `.mx` or `.d` integer EW operations**: Spec aspirational API lists these but no hardware instructions exist in RVM 0.6.
9. **No stream load/store**: Spec mentions `msld`/`msst` but no instructions exist in RVM 0.6.

See `13-verification-and-fixes.md` for detailed comparison with spec intrinsic API.

## Attribute/Arch Tests
The `-march` acceptance of `xtheadmatrix0p6` was not explicitly tested in a separate attribute test file. It is implicitly tested via the `--mattr=+experimental-xtheadmatrix` flag used in all test files.

## Encoding Verification Status
All 227 instruction encodings have been verified against the RVM 0.6 spec (two independent audits):
- Audit #1 (Gemini): 24/24 programmatic bit-field checks passed, 0 conflicts
- Audit #2 (Claude Opus 4.6): every bit field re-verified against spec source files, no discrepancies
- Known spec errata documented: matmul uop=01 in instruction_list.adoc should be uop=10; mfmin.s/mfmin.h names swapped in spec
