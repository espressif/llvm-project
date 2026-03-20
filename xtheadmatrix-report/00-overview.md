# XuanTie RVM 0.6 (RISC-V Matrix Extension) — Implementation Report

## Project Summary

Full assembler/disassembler, LLVM IR intrinsics, Clang builtins, ISel/CodeGen,
and higher-level C API for the XuanTie RVM 0.6 (RISC-V Matrix Extension) in
LLVM, targeting AI/ML workloads. 257 instructions (227 base XTHeadMatrix +
30 Zmpanel panel-aware) cover matrix multiplication, load/store, configuration,
data movement, element-wise arithmetic, and panel-aware 2x2 tiling across
data types from FP8 to FP64 and INT8 to INT64.

## Key Facts

| Item | Value |
|------|-------|
| Extension | `xtheadmatrix` (experimental, v0.6) + `xtheadzmpanel` |
| Decoder namespace | `XTHeadMatrix` + `XTHeadZmpanel` |
| Major opcode | `OPC_CUSTOM_1` (0b0101011) |
| Instructions | 257 total: 227 base (7 config, 56 load/store, 27 matmul, 35 misc, 102 EW) + 30 Zmpanel (12 config, 2 load, 2 store, 14 compute) |
| Registers | 8 matrix: tr0-tr3 (tile) + acc0-acc3 (accumulator), 3-bit encoded |
| CSRs | 31 (13 base + 18 Zmpanel), prefixed `th.` |
| Programming model | Spec-API (ManagedRA) exclusively; Zmpanel uses implicit HW state |
| LLVM intrinsics | ~220 `_internal` + 7 config + 30 Zmpanel |
| Clang builtins | ~220 Spec-API + 22 mget/mset + 22 mundef + 7 config + 30 Zmpanel ≈ 302 |
| Built-in types | 22 native (`__rvm_int8_t` .. `__rvm_float64x2_t`) |
| C header | `<thead_matrix.h>` with 450+ API functions/macros |
| ISel | `selectTHMatrixInternal()`, 16 dispatch categories, ~257 table entries |
| Pseudo expansion | ~220 PTH_*_V → TH_* post-RA (Zmpanel: no pseudos needed) |

## Zmpanel Extension

The Zmpanel extension adds 30 fire-and-forget macro instructions for panel-aware
2x2 matrix tiling. These operate on implicit hardware state (not compiler-managed
matrix register values) and require no pseudo instructions.

- **Config (12)**: `mset22adra/b/d`, `mset22rsba/b/d`, `mset22m/n/k`, `msetrstptr`, `msetaccum`, `msetoob`
- **Load (2)**: `ml22e8`, `ml22e16`
- **Store (2)**: `msc22e16`, `msc22e32`
- **Compute INT (4)**: `mmacc22_w_b` (ss/uu/us/su)
- **Compute FP (10)**: `mfmacc22_h/bf16/s` with E5/E4 FP8 sources + `mfmacc22_h`, `mfmacc22_s_h`, `mfmacc22_s_bf16`, `mfmacc22_s`

All use func3=010 (panel macro) to distinguish from standard RVM (func3=000).
Separate `XTHeadZmpanel` decoder namespace. Feature `FeatureVendorXTHeadZmpanel`
implies `FeatureVendorXTHeadMatrix`.

Panel load/store/compute instructions carry implicit Defs/Uses on matrix registers
to prevent incorrect reordering by the compiler. ISel uses the `THMI_PanelFireForget`
dispatch category for these instructions (replacing `THMI_NoArgs` which is only used
for `mrelease`). A mixed-mode conflict detection emits a fatal error if a function
attempts to use both ManagedRA (base XTHeadMatrix) and Zmpanel fire-and-forget
instructions. The `UsesZmpanelFireAndForget` flag in `RISCVMachineFunctionInfo`
tracks panel usage for this check. The C header API is guarded with
`#if defined(__riscv_xtheadzmpanel)`.

## Spec-API Coverage

All 227 base hardware operations are covered:

- **Loads**: A-tile `__riscv_th_mld_a_*`, B-tile `__riscv_th_mld_b_*`, accumulator `__riscv_th_mld_acc_*` (11 types each)
- **Stores**: `__riscv_th_mst_*` (11 types)
- **Matmul**: INT `__riscv_th_mmacc_*` / `mmaccu_*` / `mmaccus_*` / `mmaccsu_*` (14 + 4 x2), FP `__riscv_th_mfmacc_*` (13 + 3 x2)
- **Tuple**: `__riscv_th_mget_*` / `__riscv_th_mset_*` (11 types each, extract/insert from x2 pairs)
- **Zero**: `__riscv_th_mzeros_*` (11 single + 11 x2 types)
- **EW integer**: `__riscv_th_madd_w_mm` etc. (11 ops × .mm/.mv.i)
- **EW FP**: `__riscv_th_mfadd_s_mm` etc. (5 ops × 3 precisions × .mm/.mv.i)
- **Conversions**: format (26), float-int (12), packed (4)
- **N4clip**: 8 variants (signed/unsigned × low/high × .mm/.mv.i)
- **Data movement**: move, dup (4 sizes), pack (3), slide (row + 4 col sizes × up/down), broadcast (row + 4 col sizes)

## Final Test Status

| Test suite | Result |
|------------|--------|
| `xtheadmatrix-spec-api.c` (23 test cases) | PASS |
| `xtheadmatrix-spec-api-full.c` (extended API coverage) | PASS |
| `xtheadmatrix-c-api-pipeline.c` (register alloc, dependency, CSR config) | PASS |
| `xtheadmatrix-api-coverage.c` (full API function coverage) | PASS |
| `xtheadmatrix-verification-fixes.c` (verification bug fix tests) | PASS |
| `xtheadmatrix-x2-types.c` (15 test cases, O0+O2) | PASS |
| `xtheadmatrix-spec-api-example.c` (e2e widening matmul) | PASS |
| `xtheadmatrix-managed-ra.ll` (basic RA) | PASS |
| `xtheadmatrix-managed-ra-full.ll` (comprehensive RA) | PASS |
| `xtheadmatrix-managed-ra-misc.ll` (misc RA patterns) | PASS |
| `xtheadmatrix-managed-ra-regclass.ll` (register class) | PASS |
| `xtheadmatrix-managed-ra-pipeline.ll` (pipeline/scheduling) | PASS |
| `xtheadmatrix-managed-ra-spill.ll` (spill/reload) | PASS |
| `xtheadmatrix-lower-O0.ll` | PASS |
| `thead-matrix-builtin-types.c` | PASS |
| `thead-matrix-types-extended.c` | PASS |
| `xtheadmatrix-valid.s` (227 instruction encodings) | PASS |
| `xtheadmatrix-invalid.s` | PASS |
| `xtheadmatrix-csr.s` | PASS |
| `xtheadmatrix-inline-asm.c` (inline asm constraints) | PASS |
| `xtheadmatrix-zmpanel-api.c` (Zmpanel C header API) | PASS |
| `xtheadzmpanel-valid.s` (30 Zmpanel encoding tests) | PASS |
| `xtheadzmpanel-intrinsics.ll` (Zmpanel intrinsic codegen) | PASS |
| `xtheadzmpanel-builtins.c` (Zmpanel builtin codegen) | PASS |
| `xtheadzmpanel-inline-asm.c` (Zmpanel inline asm) | PASS |
| `xtheadzmpanel-header-api.c` (Zmpanel header API) | PASS |
| `xtheadzmpanel-csr.s` (18 panel CSR name tests) | PASS |

## Verification History

Twelve independent verification rounds were completed:

1. **Gemini (2026-03-04)**: Found 2 HIGH bugs (conversion pseudo register classes THRVMMR→THRVMACC; matmul operand swap), filled 3 coverage gaps (B-tile load, FP/unsigned variants, matmul variants).

2. **Claude Opus 4.6 #1**: Full re-verification of all 227 encodings, 220 pseudo expansions, spec-API codegen. No new bugs. Documented limitations and spec differences.

3. **Claude Opus 4.6 #2**: End-to-end compilation + usability audit. Fixed forward declaration, 32 name collisions, type inconsistencies, config intrinsics forcing DirectReg mode (critical RA fix).

4. **Claude Opus 4.6 #3**: DirectReg removal + Spec-API completion. Removed entire DirectReg model (~3000 lines). Added ~130 new Spec-API builtins for all remaining operations. Fixed FP EW .mv.i signatures, immediate type legalization, config intrinsic ISel dispatch. All tests pass.

5. **Claude Opus 4.6 #4**: Fixed HIGH-severity `SpecAPIMatmulWiden` bug — 8 widening FP matmul builtins passed acc as all 3 intrinsic args `{acc, acc, acc}`. Changed to 6-arg `(acc, a, b, m, k, n)`, unified through `SpecAPIMatmul`. Added 11 new tests (8 widening + 3 ISel + 1 e2e example). All 8 xtheadmatrix tests pass.

6. **Claude Opus 4.6 #5**: Implemented proper x2 (register-pair) type support. Added `RVM_X2_TYPE` macro differentiation in `.def` file. x2 types now lower to `{ target("riscv.matrix"), target("riscv.matrix") }` struct IR. Added 22 `mget`/`mset` builtins (replacing mundef stubs) with extractvalue/insertvalue+select codegen. Added 7 x2 matmul wrapper functions (4 INT16→INT64 + 3 FP). Comprehensive `xtheadmatrix-x2-types.c` test file (15 test cases, O0+O2) + 3 new tests in `xtheadmatrix-spec-api.c`.

7. **Claude Opus 4.6 #6**: Implemented and verified Zmpanel panel-aware 2x2 matrix tiling extension. 30 new instructions (12 config, 2 load, 2 store, 14 compute). Added implicit Defs/Uses on panel load/store/compute instructions. Introduced `THMI_PanelFireForget` ISel dispatch category. Added mixed-mode conflict detection (fatal error when ManagedRA + fire-and-forget used in same function). Added `UsesZmpanelFireAndForget` flag to `RISCVMachineFunctionInfo`. Header API guarded with `#if defined(__riscv_xtheadzmpanel)`. Compute instructions confirmed to use uop=10 (matching standard matmul pattern). All 20 tests pass (5 new + 15 existing), zero regressions.

8. **Claude Opus 4.6 #7**: Full-stack verification across all layers. Fixed 4 bugs (mreinterpret data loss, xmsize CSR, x2 matmul element 1 ignored, mzero naming). Verified encoding, intrinsic, builtin, ISel, pseudo, type-lowering, and Zmpanel layers.

9. **Claude Opus 4.6 #8 (2026-03-19)**: Golden spec cross-reference audit. Read ALL spec files (`spec/*.adoc` + `doc/intrinsic/rvm-intrinsic-api.adoc`) and cross-referenced against the full implementation. **No new correctness bugs found.** Identified and documented comprehensive list of C API naming/signature differences from the spec intrinsic API (see `13-verification-and-fixes.md`). Key findings: (a) spec's canonical function names (`__riscv_th_fmmacc`, `__riscv_th_mmaqa`, etc.) not provided as aliases; (b) matmul dimension param order is (M,K,N) vs spec's (M,N,K); (c) EW operations rely on prior CSR state instead of taking dimension params; (d) mzero takes (m,n) params vs spec's no-param signature; (e) spec's intrinsic API doc uses older instruction mnemonics that differ from RVM 0.6. All issues are API naming/convention — no hardware behavior errors.

10. **Claude Opus 4.6 #9 (2026-03-20)**: Full independent re-verification with five parallel verification agents. Verified: (a) 40+ instruction encodings bit-by-bit — 0 bugs; (b) all register class constraints (tile vs accumulator) — 0 bugs; (c) all 30 Zmpanel instructions, 18 CSRs, C API — 0 bugs; (d) stream load/store correctly omitted; (e) C API naming confirmed correct per RVM 0.6 mnemonics. **No new bugs found.** All previously documented differences and limitations remain accurate. Noted gap: no MC-level round-trip encoding tests for Zmpanel panel instructions.

11. **Claude Opus 4.6 #10 (2026-03-20)**: Comprehensive end-to-end verification with parallel spec-comparison agents. Verified: (a) ALL 257 instruction encodings against golden spec — every opcode, func3, func4, uop, d_size, s_size field confirmed correct; (b) complete builtin→intrinsic→instruction chain — every builtin maps to the correct intrinsic with correct argument types and ordering; (c) configuration emission (msettilem/k/n) verified correct before every operation; (d) matmul operand ordering `{acc, B, A}` confirmed correct for `md = md + ms1 × ms2^T`; (e) panel-aware 2x2 macro decomposition verified (load/store/compute register modeling correct, tr4-tr7 correctly handled as hardware-only); (f) pointer types in C API verified correct across all load/store functions; (g) EEW selection correct for all type variants. Re-confirmed 3 spec document errata (matmul uop, mfmin swap, pmmaccus typo). **No new bugs found.** All 26 tests pass. Fixed test CHECK patterns for intrinsic name mangling (`.i64` suffix) and `_Float16*` → `uint16_t*` pointer compatibility.

12. **Claude Opus 4.6 #11 (2026-03-20)**: Final review and cleanup with 8 parallel verification agents. Re-verified all 257 encodings, full dimension parameter flow (M/K/N traced end-to-end), complete intrinsic/builtin chain (407 entries), all 30 Zmpanel instructions, and ManagedRA pass. Three actionable findings addressed: (a) removed 11 broken x2 reinterpret macros from `thead_matrix.h` (x2 struct types cannot fit single `"tr"` inline asm constraint — users should use mget/mset decomposition); (b) registered 18 Zmpanel panel CSRs (0xcc4-0xcd5) in `RISCVSystemOperands.td` under `FeatureVendorXTHeadZmpanel` guard — now accessible by name in inline asm (e.g., `csrr a0, th.panel_m`); (c) documented `mfmacc_h_x2` spec divergence as intentional (x2 on accumulator instead of B operand, for API consistency with fp64/int64 x2 variants). Fixed report documentation error: mreinterpret uses `"=tr"/"tr"` constraint pair, not `"0"` tied constraint. Added `xtheadzmpanel-csr.s` test. All 27 tests pass.

## Current Limitations

1. No 64-bit instruction format (spec defines but not implemented)
2. Matrix types cannot cross function boundaries (no ABI support)
3. No auto-matmul from C loops (explicit builtins required)
4. Limited register file (4 tile + 4 accumulator)
5. Whole-register spill granularity (8192-bit per spill)
6. `-O0` support limited (RISCVLowerMatrixType pass provides basic support)
7. Known spec errata (5): matmul uop=01→10; mfmin.s/mfmin.h names swapped; pmmaccus typo; mbce no encoding; Zmpanel compute rs1/s_size field mislabeled
8. Zmpanel is fire-and-forget: the compiler does not manage panel registers tr4-tr7; all panel state is configured explicitly by the programmer. Mixed-mode usage (ManagedRA + Zmpanel fire-and-forget in the same function) is detected and rejected with a fatal error at ISel time
9. C API naming follows RVM 0.6 assembly mnemonics, not the spec intrinsic API document (`rvm-intrinsic-api.adoc`). The spec's canonical function names (`__riscv_th_fmmacc`, `__riscv_th_mmaqa`, unified `__riscv_th_mld`, etc.) are not provided. Code written against the spec intrinsic API will not compile without adaptation. See `13-verification-and-fixes.md` for the full mapping table
10. EW operations do not auto-configure mtilem/mtilen CSRs; they rely on prior load/matmul operations having set the correct tile dimensions. For standalone EW use, users must manually call config functions first
11. Not implemented: stream load/store (`msld`/`msst`), matrix-scalar EW (`.mx`), 64-bit INT EW (`.d.mm`), `mmov.mv` row move, `mbce` element broadcast (no encoding in spec)
12. x2 reinterpret not available (x2 struct types cannot fit single `"tr"` inline asm constraint; use mget/mset decomposition)
13. `mfmacc_h_x2` puts x2 on accumulator (intentional divergence from spec which puts x2 on B operand); consistent with fp64/int64 x2 variants
