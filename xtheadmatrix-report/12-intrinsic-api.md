# Higher-Level Intrinsic API (`<thead_matrix.h>`)

## Overview

- Implements the RVM Intrinsic API Reference Manual v0.2
- Provides 414 API functions/macros via `<thead_matrix.h>` header
- Phase C: uses native `__rvm_*_t` built-in type aliases (not structs)
- 121 builtins have typed signatures for Sema-level type checking
- 22 `mundef` builtins for creating undefined matrix values

## Implementation Details

### Types (22 matrix types)

- 11 single-register: mint8_t through mfloat64_t
- 11 register-pair: mint8x2_t through mfloat64x2_t
- Dimension types: mrow_t, mcol_t (typedef size_t)
- CSR enum: RVM_CSR with all 13 CSR addresses

### Key Design Decisions

1. **Native built-in types**: `mint32_t` is now `typedef __rvm_int32_t`, not a struct
2. **Role-specific loads**: `mld_a`/`mld_b`/`mld_c` instead of generic `mld`
3. **Config auto-management**: loads and compute functions set tile dimensions
4. **ImmArg constraint**: MVI, slide, and broadcast functions are macros (not inline functions) to preserve compile-time constant requirement
5. **Role-specific stores only**: generic `mst` dispatch removed (no `__mreg` field in opaque types)
6. **Typed builtins (Phase C)**: 121 builtins accept/return `__rvm_*_t` types, enabling Sema type checking
7. **mundef builtins**: 22 builtins return PoisonValue tokens for creating undefined matrix values
8. **Register index parameters**: All builtins accept ImmArg register index arguments for flexible register selection (no hardcoded register assignments); Sema validates index constraints per instruction category
9. **Register index constants**: `__RVM_TR0` (0) through `__RVM_TR3` (3) for tile registers, `__RVM_ACC0` (4) through `__RVM_ACC3` (7) for accumulator registers

### Register Index Constants

All builtins and API functions accept register index parameters to specify which physical matrix register to use. The following constants are defined in `<thead_matrix.h>`:

| Constant | Value | Register | Role |
|---|---|---|---|
| `__RVM_TR0` | 0 | tr0 | Tile register 0 |
| `__RVM_TR1` | 1 | tr1 | Tile register 1 |
| `__RVM_TR2` | 2 | tr2 | Tile register 2 |
| `__RVM_TR3` | 3 | tr3 | Tile register 3 |
| `__RVM_ACC0` | 4 | acc0 | Accumulator register 0 |
| `__RVM_ACC1` | 5 | acc1 | Accumulator register 1 |
| `__RVM_ACC2` | 6 | acc2 | Accumulator register 2 |
| `__RVM_ACC3` | 7 | acc3 | Accumulator register 3 |

Sema validates that register indices are within valid ranges for each instruction category. For example, matmul instructions require specific register roles (accumulator for destination, tile for sources), and element-wise instructions require accumulator registers for all operands.

### Function Count by Category

| Category | Count | Notes |
|---|---|---|
| Config | 7 | msetmrow_m/n, msetmcol_e8/16/32/64, mrelease |
| CSR | 5 | mread_csr, mwrite_csr, xmlenb, xrlenb, xmsize |
| Zero | 22 | All types |
| Undefined | 22 | All types |
| Reinterpret | 22 | All types |
| Tuple | 22 | mget + mset for all x2 types |
| Load | 77 | 11 types x (a/b/c + at/bt/ct + whole) |
| Store | 77 | 11 types x (a/b/c + at/bt/ct + whole) — generic dispatch removed |
| FP Matmul | 13 | fmmacc + fwmmacc variants |
| INT Matmul | 14 | mmaqa + pmmaqa + bypass |
| INT EW | 22 | 11 ops x (mm + mv macro) |
| FP EW | 30 | 5 ops x 3 sizes x (mm + mv macro) |
| FP Convert | 26 | |
| Float-Int | 12 | |
| N4Clip | 8 | 4 mm + 4 mv macro |
| Packed Conv | 4 | |
| Move/Dup | 13 | mmov_mm, mmov_x_m, mmov_m_x, mdup_m_x |
| Pack | 3 | |
| Slide | 10 | macros (ImmArg) |
| Broadcast | 5 | macros (ImmArg) |

## Spec-API (ManagedRA) — Register-Allocator-Managed Intrinsics (2026-03-04)

The spec-API provides a higher-level programming model where the compiler's register
allocator manages matrix registers. Matrix values are returned and passed as opaque
types (`mint32_t` etc.) with proper SSA dataflow.

### Function Count

| Category | Count | Notes |
|----------|-------|-------|
| Load A-tile (mlae) | 11 | All types (i/u/f × 8/16/32/64) |
| Load B-tile (mlbe) | 11 | All types, sets K/N dimensions |
| Load Acc (mlce) | 11 | All types, sets M/N dimensions |
| Store (msce) | 11 | All types, sets M/N dimensions |
| INT matmul (w_b) | 4 | INT8→INT32, 4 sign variants |
| INT matmul (d_h) | 4 | INT16→INT64, 4 sign variants |
| Partial INT matmul | 4 | INT8→INT32, 4 sign variants |
| Bypass INT matmul | 2 | ss, uu |
| FP matmul native | 3 | h, s, d |
| FP matmul widening typed | 2 | s_h, d_s |
| FP matmul widening opaque | 8 | h_e4/e5, bf16_e4/e5, s_bf16/e4/e5/tf32 |
| Zero | 11 | All types |
| **Total** | **82** | |

### Key Design Decisions

1. **Three load roles**: `mld` (A-tile, M×K), `mld_b` (B-tile, K×N), `mld_acc` (C/acc, M×N)
2. **Dimension auto-config**: Each builtin emits `msettilem`/`msettilek`/`msettilen` calls
3. **Operand ordering**: Intrinsic signature is `(acc, ms2, ms1)`; codegen passes `{acc, b, a}` so A→ms1, B→ms2 per spec
4. **Opaque widening matmul**: FP8/BF16/TF32 sources have no C type representation; builtin takes `(acc, m, k, n)` only
5. **Type-neutral hardware**: All type variants at same EEW map to same instruction (e.g., `mld_spec_i32` and `mld_spec_f32` both emit `mlae_internal32`)

### C Header Wrappers

All spec-API functions are defined in `<thead_matrix.h>` Section 23 using macro patterns:
- `__THEAD_SPEC_MLD(SUFFIX, CTYPE, MTYPE, BUILTIN)` — A-tile loads
- `__THEAD_SPEC_MLD_B(SUFFIX, CTYPE, MTYPE, BUILTIN)` — B-tile loads
- `__THEAD_SPEC_MLD_ACC(SUFFIX, CTYPE, MTYPE, BUILTIN)` — Accumulator loads
- `__THEAD_SPEC_MST(SUFFIX, CTYPE, MTYPE, BUILTIN)` — Stores
- `__THEAD_SPEC_MMAQA(SUFFIX, ATYPE, BTYPE, CTYPE, BUILTIN)` — INT matmul
- `__THEAD_SPEC_FMMAQA(SUFFIX, ATYPE, BTYPE, CTYPE, BUILTIN)` — FP matmul (typed)
- `__THEAD_SPEC_FMMAQA_WIDEN(SUFFIX, CTYPE, BUILTIN)` — FP matmul (opaque sources)
- `__THEAD_SPEC_MZERO(SUFFIX, MTYPE, BUILTIN)` — Zero constructors

## Limitations / Not Mapped

- Stream load/store (msld/msst) — no RVM 0.6 instruction
- Matrix-scalar EW (.mx) — no RVM 0.6 instruction
- 64-bit INT EW (.d variants) — not in RVM 0.6
- 64-bit instruction support — deferred (no LLVM infrastructure for custom-1 64-bit)

## Files

- Header: `clang/lib/Headers/thead_matrix.h`
- CMake: `clang/lib/Headers/CMakeLists.txt` (1 line added)
- Tests:
  - `clang/test/CodeGen/RISCV/thead-matrix-api.c` — original API test
  - `clang/test/CodeGen/RISCV/thead-matrix-api-comprehensive.c` — exhaustive API coverage (44 tests)
  - `clang/test/CodeGen/RISCV/thead-matrix-api-patterns.c` — 26 real-world API usage patterns
  - `clang/test/CodeGen/RISCV/thead-matrix-builtins-exhaustive.c` — all builtins (24 tests)
  - `clang/test/CodeGen/RISCV/thead-matrix-comprehensive-codegen.c` — 18 instruction-category codegen tests
  - `clang/test/CodeGen/RISCV/thead-matrix-corner-cases.c` — 20 corner-case and complex pipeline tests
  - `clang/test/CodeGen/RISCV/thead-matrix-examples.c` — 18 real-world examples
  - `clang/test/CodeGen/RISCV/thead-matrix-register-model.c` — register model tests
  - `clang/test/CodeGen/RISCV/thead-matrix-types-extended.c` — type system tests (5 tests)
  - `clang/test/CodeGen/RISCV/thead-matrix-builtin-types.c` — built-in type declarations
  - `clang/test/CodeGen/RISCV/xtheadmatrix-codegen.c` — low-level builtin codegen
  - `clang/test/Sema/riscv-xtheadmatrix-reg-constraints.c` — Sema register constraint validation

## Verification

- Build: clang builds cleanly
- Include: `#include <thead_matrix.h>` compiles with xtheadmatrix feature
- Tests: 12 Clang test files pass (11 CodeGen + 1 Sema)
- Tests: 4 LLVM test files pass (3 MC + 1 CodeGen ISel)
- Total: 16 tests, all passing
- Regression: 557/557 RISCV MC tests pass
- Audit: cross-verified against RVM 0.6 spec and Intrinsic API Reference Manual v0.2

## Phase B: Native Built-in Types

### Overview
22 first-class Clang built-in types implemented following the RVV pattern.

### Types
11 single-register: `__rvm_int8_t`, `__rvm_int16_t`, `__rvm_int32_t`, `__rvm_int64_t`, `__rvm_uint8_t`, `__rvm_uint16_t`, `__rvm_uint32_t`, `__rvm_uint64_t`, `__rvm_float16_t`, `__rvm_float32_t`, `__rvm_float64_t`
11 pair: `__rvm_int8x2_t` through `__rvm_float64x2_t`

### Files Modified (~27)
- New: `clang/include/clang/Basic/RISCVMatrixTypes.def`
- Headers: TargetInfo.h, ASTContext.h, TypeBase.h, ASTBitCodes.h, SemaRISCV.h, module.modulemap
- AST: ASTContext.cpp, Type.cpp, ItaniumMangle.cpp, ASTImporter.cpp, ExprConstant.cpp, NSAPI.cpp, TypeLoc.cpp, PrintfFormatString.cpp, TypeProperties.td
- CodeGen: CodeGenTypes.cpp, CGDebugInfo.cpp, ItaniumCXXABI.cpp, CIRGenItaniumCXXABI.cpp
- Sema: Sema.cpp, SemaRISCV.cpp, SemaDecl.cpp, SemaExpr.cpp
- Serialization: ASTCommon.cpp, ASTReader.cpp
- Misc: TargetInfo.cpp, RISCV.h, USRGeneration.cpp, CIndex.cpp

### Key Design Decisions
- Types are opaque: Width=0, Align=8
- Map to `llvm::TargetExtType("riscv.matrix")` in CodeGen
- Feature-gated via `checkRVMTypeSupport` requiring `"experimental-xtheadmatrix"`
- NumOfBuiltinTypeBits bumped 9 to 10, NUM_PREDEF_TYPE_IDS bumped 514 to 560

### Test
`clang/test/CodeGen/RISCV/thead-matrix-builtin-types.c` -- verifies all 22 types compile

## Phase C: Typed Builtins

### Overview
121 low-level builtins (loads, matmul, EW arithmetic, conversions) now accept and return native `__rvm_*_t` matrix types, enabling Sema-level type checking. 22 `mundef` builtins create undefined matrix values (`PoisonValue`). Together, 143 builtins are typed. The remaining 106 builtins (stores, config, zero, misc, FP8/BF16/TF32) retain void signatures. The `<thead_matrix.h>` header uses native built-in type aliases.

### Typed Builtin Categories

| Category | Typed | Untyped | Notes |
|---|---|---|---|
| Loads | 28 | 0 | Return `__rvm_int<W>_t` by EEW |
| FP Matmul | 5 | 8 | h,s,d,s_h,d_s typed; FP8/BF16/TF32 void |
| INT Matmul | 14 | 0 | All typed with signed/unsigned Qm codes |
| INT EW MM | 11 | 0 | All `__rvm_int32_t` |
| INT EW MVI | 11 | 0 | `__rvm_int32_t` + `unsigned int` imm |
| FP EW MM | 15 | 0 | By precision (h/s/d) |
| FP EW MVI | 15 | 0 | By precision + `unsigned int` imm |
| FP-FP Convert | 8 | 18 | h↔s, s↔d typed; FP8/BF16/TF32 void |
| Float-Int Convert | 12 | 0 | All typed |
| mundef | 22 | — | Return PoisonValue tokens |
| **Total** | **141** | **26** | Plus 106 unchanged void builtins |

### Files Modified
- `llvm/lib/IR/Type.cpp` — TargetExtType "riscv.matrix" properties
- `clang/lib/AST/ASTContext.cpp` — `Qm<N>` type encoding
- `clang/utils/TableGen/ClangBuiltinsEmitter.cpp` — type name → encoding mapping
- `clang/include/clang/Basic/BuiltinsRISCVXTHeadMatrix.td` — typed prototypes
- `clang/lib/CodeGen/TargetBuiltins/RISCV.cpp` — CGBuiltin handler
- `clang/lib/Headers/thead_matrix.h` — native type aliases + typed API
- `clang/test/CodeGen/RISCV/xtheadmatrix-codegen.c` — updated e2e test
- `clang/test/CodeGen/RISCV/thead-matrix-api.c` — updated API test

### Verification
- Build: clean (0 errors)
- All 16 XTHeadMatrix tests pass (12 Clang + 4 LLVM)
- 557/557 RISCV MC regression tests pass
- Audit fixes: N4clip signed return types corrected, reinterpret casts accept src parameter
