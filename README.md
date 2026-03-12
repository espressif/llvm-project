# XuanTie RVM 0.6 (RISC-V Matrix Extension) — Implementation Report

## Project Summary

Full assembler/disassembler, LLVM IR intrinsics, Clang builtins, ISel/CodeGen,
and higher-level C API for the XuanTie RVM 0.6 (RISC-V Matrix Extension) in
LLVM, targeting AI/ML workloads. 257 instructions (227 base + 30 Zmpanel)
cover matrix multiplication, load/store, configuration, data movement, and
element-wise arithmetic across data types from FP8 to FP64 and INT8 to INT64.
The Zmpanel extension adds panel-aware 2x2 tiling support for efficient GEMM
pipelines, with fire-and-forget macro instructions that handle load, compute,
and store operations on 2x2 matrix panels.

## Key Facts

| Item | Value |
|------|-------|
| Extension | `xtheadmatrix` (experimental, v0.6) |
| Panel extension | `xtheadzmpanel` (experimental, v0.6) |
| Decoder namespace | `XTHeadMatrix` + `XTHeadZmpanel` |
| Major opcode | `OPC_CUSTOM_1` (0b0101011) |
| Instructions | 257 (227 base + 30 Zmpanel panel-aware) |
| Registers | 8 matrix: tr0-tr3 (tile) + acc0-acc3 (accumulator), 3-bit encoded |
| CSRs | 13 base + 18 Zmpanel panel-aware |
| Programming model | Spec-API (ManagedRA) exclusively |
| LLVM intrinsics | ~220 `_internal` + 7 config + 30 Zmpanel |
| Clang builtins | ~220 Spec-API + 22 mget/mset + 22 mundef + 7 config + 30 Zmpanel ≈ 302 |
| Built-in types | 22 native (`__rvm_int8_t` .. `__rvm_float64x2_t`) |
| C header | `<thead_matrix.h>` with 450+ API functions/macros |
| ISel | `selectTHMatrixInternal()`, 16 dispatch categories, ~257 table entries |
| Pseudo expansion | ~220 PTH_*_V → TH_* post-RA |

## Spec-API Coverage

All 227 base hardware operations are covered:

- **Loads**: A-tile `__riscv_th_mld_a_*`, B-tile `__riscv_th_mld_b_*`, accumulator `__riscv_th_mld_acc_*` (11 types each)
- **Stores**: `__riscv_th_mst_*` (11 types)
- **Matmul**: INT `__riscv_th_mmaq_*` (14 + 4 x2), FP `__riscv_th_mfmaqa_*` (13 + 3 x2)
- **Tuple**: `__riscv_th_mget_*` / `__riscv_th_mset_*` (11 types each, extract/insert from x2 pairs)
- **Zero**: `__riscv_th_mzeros_*` (11 types)
- **EW integer**: `__riscv_th_madd_w_mm` etc. (11 ops × .mm/.mv.i)
- **EW FP**: `__riscv_th_mfadd_s_mm` etc. (5 ops × 3 precisions × .mm/.mv.i)
- **Conversions**: format (26), float-int (12), packed (4)
- **N4clip**: 8 variants (signed/unsigned × low/high × .mm/.mv.i)
- **Data movement**: move, dup (4 sizes), pack (3), slide (row + 4 col sizes × up/down), broadcast (row + 4 col sizes)

## Zmpanel Extension (Panel-Aware 2x2 Tiling)

The `xtheadzmpanel` extension adds 30 fire-and-forget macro instructions for
efficient 2x2 panel GEMM pipelines. Unlike the base `xtheadmatrix` instructions
which operate on individual matrix registers via the compiler's register
allocator, Zmpanel instructions configure memory addresses, strides, and
dimensions through dedicated panel state, then execute load-compute-store
sequences as opaque operations. The hardware manages internal panel registers
(tr4-tr7) automatically -- the compiler does not allocate or track them.

Panel load/store/compute instructions carry implicit Defs/Uses on matrix
registers to prevent incorrect reordering by the compiler. ISel uses the
`THMI_PanelFireForget` dispatch category (replacing `THMI_NoArgs` for panel
non-config instructions). A mixed-mode conflict detection emits a fatal error
if a function attempts to use both ManagedRA (base XTHeadMatrix) and
Zmpanel fire-and-forget instructions, since the two models have incompatible
assumptions about matrix register ownership.

### Instruction Summary (30 total)

| Category | Count | Instructions |
|----------|-------|-------------|
| Config (address) | 3 | `th.mset22adra`, `th.mset22adrb`, `th.mset22adrd` |
| Config (stride) | 3 | `th.mset22rsba`, `th.mset22rsbb`, `th.mset22rsbd` |
| Config (dimension) | 3 | `th.mset22m`, `th.mset22n`, `th.mset22k` |
| Config (control) | 3 | `th.msetrstptr`, `th.msetaccum`, `th.msetoob` |
| Load | 2 | `th.ml22e8`, `th.ml22e16` |
| Store | 2 | `th.msc22e16`, `th.msc22e32` |
| FP compute (ext. operand) | 6 | `th.mfmacc22.{h,bf16,s}.{e5,e4}` |
| FP compute (standard) | 4 | `th.mfmacc22.h`, `th.mfmacc22.s.h`, `th.mfmacc22.s.bf16`, `th.mfmacc22.s` |
| INT compute | 4 | `th.mmacc22.w.b`, `th.mmaccu22.w.b`, `th.mmaccus22.w.b`, `th.mmaccsu22.w.b` |

### C API Naming

All Zmpanel functions are exposed via `<thead_matrix.h>`:

- **Config**: `__riscv_th_mset22adra(val)`, `__riscv_th_mset22adrb(val)`, ..., `__riscv_th_msetoob(val)`
- **Load**: `__riscv_th_ml22e8()`, `__riscv_th_ml22e16()`
- **Store**: `__riscv_th_msc22e16()`, `__riscv_th_msc22e32()`
- **FP matmul**: `__riscv_th_mfmacc22_h()`, `__riscv_th_mfmacc22_s_h()`, `__riscv_th_mfmacc22_h_e5()`, ...
- **INT matmul**: `__riscv_th_mmacc22_w_b()`, `__riscv_th_mmaccu22_w_b()`, ...

### Usage Example: INT8 Panel GEMM

```c
#include <thead_matrix.h>

// Compute D[m x n] += A[m x k] * B[k x n] using 2x2 panel tiling (INT8->INT32)
void panel_gemm_int8(void *a, void *b, void *d,
                     size_t rsba, size_t rsbb, size_t rsbd,
                     size_t m, size_t n, size_t k) {
  // 1. Configure panel addresses
  __riscv_th_mset22adra((size_t)a);
  __riscv_th_mset22adrb((size_t)b);
  __riscv_th_mset22adrd((size_t)d);

  // 2. Configure row strides (bytes between consecutive rows)
  __riscv_th_mset22rsba(rsba);
  __riscv_th_mset22rsbb(rsbb);
  __riscv_th_mset22rsbd(rsbd);

  // 3. Configure dimensions
  __riscv_th_mset22m(m);
  __riscv_th_mset22n(n);
  __riscv_th_mset22k(k);

  // 4. Configure accumulation mode (0 = zero-init, 1 = accumulate)
  __riscv_th_msetaccum(0);

  // 5. Reset internal pointers
  __riscv_th_msetrstptr(1);

  // 6. Execute: load A/B panels, compute matmul, store D panel
  __riscv_th_ml22e8();          // Load INT8 A and B panels
  __riscv_th_mmacc22_w_b();     // INT8 matmul -> INT32 accumulate
  __riscv_th_msc22e32();        // Store INT32 result panel
}
```

### Usage Example: FP16 Panel GEMM

```c
#include <thead_matrix.h>

// Compute D[m x n] += A[m x k] * B[k x n] using 2x2 panel tiling (FP16)
void panel_gemm_fp16(void *a, void *b, void *d,
                     size_t rsba, size_t rsbb, size_t rsbd,
                     size_t m, size_t n, size_t k) {
  __riscv_th_mset22adra((size_t)a);
  __riscv_th_mset22adrb((size_t)b);
  __riscv_th_mset22adrd((size_t)d);
  __riscv_th_mset22rsba(rsba);
  __riscv_th_mset22rsbb(rsbb);
  __riscv_th_mset22rsbd(rsbd);
  __riscv_th_mset22m(m);
  __riscv_th_mset22n(n);
  __riscv_th_mset22k(k);
  __riscv_th_msetaccum(0);
  __riscv_th_msetrstptr(1);

  __riscv_th_ml22e16();         // Load FP16 A and B panels
  __riscv_th_mfmacc22_h();      // FP16 matmul-accumulate
  __riscv_th_msc22e16();        // Store FP16 result panel
}
```

## Final Test Status

| Test suite | Result |
|------------|--------|
| `xtheadmatrix-spec-api.c` (23 test cases) | PASS |
| `xtheadmatrix-x2-types.c` (15 test cases, O0+O2) | PASS |
| `xtheadmatrix-spec-api-example.c` (e2e widening matmul) | PASS |
| `xtheadmatrix-managed-ra*.ll` (4 tests) | PASS |
| `xtheadmatrix-lower-O0.ll` | PASS |
| `thead-matrix-builtin-types.c` | PASS |
| `thead-matrix-types-extended.c` | PASS |
| `xtheadmatrix-valid.s` (1154 lines) | PASS |
| `xtheadmatrix-invalid.s` | PASS |
| `xtheadmatrix-csr.s` | PASS |
| RISCV MC full suite | 555/555 PASS |
| `xtheadmatrix-inline-asm.c` (inline asm constraints) | PASS |
| End-to-end RA (EW, conversions, data movement, matmul pipeline) | PASS |
| Spill-pressure test (5 ACC values, 4 regs) | PASS |
| `xtheadzmpanel-valid.s` (30 instructions, encoding/disassembly) | PASS |
| `xtheadzmpanel-builtins.c` (30 builtins + 2 pipeline tests) | PASS |
| `xtheadzmpanel-intrinsics.ll` (30 intrinsics + 2 combined patterns) | PASS |
| `xtheadzmpanel-header-api.c` (header API + panel GEMM pipeline) | PASS |
| `xtheadzmpanel-inline-asm.c` (inline assembly) | PASS |

## Verification History

Seven independent verification rounds were completed:

1. **Gemini (2026-03-04)**: Found 2 HIGH bugs (conversion pseudo register classes THRVMMR→THRVMACC; matmul operand swap), filled 3 coverage gaps (B-tile load, FP/unsigned variants, matmul variants).

2. **Claude Opus 4.6 #1**: Full re-verification of all 227 encodings, 220 pseudo expansions, spec-API codegen. No new bugs. Documented limitations and spec differences.

3. **Claude Opus 4.6 #2**: End-to-end compilation + usability audit. Fixed forward declaration, 32 name collisions, type inconsistencies, config intrinsics forcing DirectReg mode (critical RA fix).

4. **Claude Opus 4.6 #3**: DirectReg removal + Spec-API completion. Removed entire DirectReg model (~3000 lines). Added ~130 new Spec-API builtins for all remaining operations. Fixed FP EW .mv.i signatures, immediate type legalization, config intrinsic ISel dispatch. All tests pass.

5. **Claude Opus 4.6 #4**: Fixed HIGH-severity `SpecAPIMatmulWiden` bug — 8 widening FP matmul builtins passed acc as all 3 intrinsic args `{acc, acc, acc}`. Changed to 6-arg `(acc, a, b, m, k, n)`, unified through `SpecAPIMatmul`. Added 11 new tests (8 widening + 3 ISel + 1 e2e example). All 8 xtheadmatrix tests pass.

6. **Claude Opus 4.6 #5**: Implemented proper x2 (register-pair) type support. Added `RVM_X2_TYPE` macro differentiation in `.def` file. x2 types now lower to `{ target("riscv.matrix"), target("riscv.matrix") }` struct IR. Added 22 `mget`/`mset` builtins (replacing mundef stubs) with extractvalue/insertvalue+select codegen. Added 7 x2 matmul wrapper functions (4 INT16→INT64 + 3 FP). Comprehensive `xtheadmatrix-x2-types.c` test file (15 test cases, O0+O2) + 3 new tests in `xtheadmatrix-spec-api.c`.

7. **Claude Opus 4.6 #6**: Implemented Zmpanel panel-aware 2x2 tiling extension (30 instructions). Added implicit Defs/Uses on panel load/store/compute instructions to prevent reordering. Introduced `THMI_PanelFireForget` ISel dispatch category. Added mixed-mode conflict detection (fatal error when ManagedRA + Zmpanel fire-and-forget used in same function). Added `UsesZmpanelFireAndForget` flag to `RISCVMachineFunctionInfo`. Header API guarded with `#if defined(__riscv_xtheadzmpanel)`. All 20 tests pass (5 new Zmpanel + 15 existing), zero regressions.

## Current Limitations

1. No 64-bit instruction format (spec defines but not implemented)
2. Matrix types cannot cross function boundaries (no ABI support)
3. No auto-matmul from C loops (explicit builtins required)
4. Limited register file (4 tile + 4 accumulator)
5. Whole-register spill granularity (8192-bit per spill)
6. `-O0` support limited (RISCVLowerMatrixType pass provides basic support)
7. Known spec errata: matmul uop=01 should be 10; mfmin.s/mfmin.h names swapped
8. Zmpanel is fire-and-forget: the compiler does not manage panel registers tr4-tr7; all panel state (addresses, strides, dimensions) is configured explicitly by the programmer and executed opaquely by hardware. Mixed-mode usage (ManagedRA base instructions + Zmpanel fire-and-forget in the same function) is detected and rejected with a fatal error at ISel time
