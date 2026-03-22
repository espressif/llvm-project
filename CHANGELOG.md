# Changelog

All notable changes to the AICS LLVM Toolchain (XTHeadMatrix / RVM 0.6) are documented here.

## [aics_llvm_toolchain_v0.2.1] - 2026-03-22

### Breaking Changes

- **CSR addresses remapped** (0x802–0x80a → 0x806–0x80e): All 9 RVM read/write
  matrix CSRs have been relocated to match latest spike/hardware mapping. Code
  using hardcoded CSR numbers (e.g., raw `csrr a0, 0x802`) will silently
  read/write the wrong register. Code using named CSRs (`csrr a0, th.xmcsr`)
  or the C API (`RVM_CSR_XMCSR`) is automatically correct. New mapping:

  | CSR | Old | New |
  |-----|-----|-----|
  | xmcsr | 0x802 | 0x806 |
  | mtilem | 0x803 | 0x807 |
  | mtilen | 0x804 | 0x808 |
  | mtilek | 0x805 | 0x809 |
  | xmxrm | 0x806 | 0x80a |
  | xmsat | 0x807 | 0x80b |
  | xmfflags | 0x808 | 0x80c |
  | xmfrm | 0x809 | 0x80d |
  | xmsaten | 0x80a | 0x80e |

  Read-only CSRs (0xcc0–0xcc3) and Zmpanel CSRs (0xcc4–0xcd5) are unchanged.

- **All integer matmul C API functions renamed** (`mmaq_*` → `mmacc_*`): Renamed
  to align with RVM 0.6 assembly mnemonics. Backward-compatible `#define` aliases
  are provided for all old names. Examples:
  - `__riscv_th_mmaq_ss_w_b` → `__riscv_th_mmacc_w_b`
  - `__riscv_th_mmaq_uu_w_b` → `__riscv_th_mmaccu_w_b`
  - `__riscv_th_mmaq_us_w_b` → `__riscv_th_mmaccus_w_b`
  - `__riscv_th_mmaq_su_w_b` → `__riscv_th_mmaccsu_w_b`
  - `__riscv_th_mmaq_ss_d_h` → `__riscv_th_mmacc_d_h` (and all sign variants)
  - `__riscv_th_mmaq_p_*` → `__riscv_th_pmmacc_*` (panelized)
  - `__riscv_th_mmaq_bp_*` → `__riscv_th_mmacc_w_bp` / `mmaccu_w_bp` (bypass)
  - All `_x2` variants similarly renamed

- **All FP matmul C API functions renamed** (`mfmaqa_*` → `mfmacc_*`): Renamed
  to align with RVM 0.6 assembly mnemonics. Backward-compatible `#define` aliases
  are provided. Examples:
  - `__riscv_th_mfmaqa_h` → `__riscv_th_mfmacc_h`
  - `__riscv_th_mfmaqa_s` → `__riscv_th_mfmacc_s`
  - `__riscv_th_mfmaqa_d` → `__riscv_th_mfmacc_d`
  - `__riscv_th_mfmaqa_s_h` → `__riscv_th_mfmacc_s_h` (widening)
  - `__riscv_th_mfmaqa_h_e4` → `__riscv_th_mfmacc_h_e4` (FP8 variants)
  - All `_x2` and widening variants similarly renamed

- **`mfmacc_h_x2` (fp16 x2 matmul) signature completely changed**: The x2
  position moved from B operand to accumulator, and all types changed:
  - Old: `mfloat16_t mfmaqa_h_x2(mfloat16_t c, mfloat16_t a, mfloat16x2_t b, ...)`
  - New: `mfloat16x2_t mfmacc_h_x2(mfloat16x2_t c, mfloat16_t a, mfloat16_t b, ...)`
  - This is an intentional divergence from the spec for API consistency: all x2
    matmul variants (fp16, fp64, int64) now uniformly place x2 on the accumulator.

- **All x2 matmul functions now process BOTH pair elements**: Previously, x2
  matmul variants only operated on element 0 of the pair, silently ignoring
  element 1. Now they correctly extract both elements, call the underlying
  builtin on each, and reassemble the result. This changes runtime behavior.
  Affected: `mfmacc_h_x2`, `mfmacc_d_x2`, `mfmacc_d_s_x2`,
  `mmacc_d_h_x2`, `mmaccu_d_h_x2`, `mmaccus_d_h_x2`, `mmaccsu_d_h_x2`.

- **x2 reinterpret macros removed** (no replacement): 11
  `__riscv_th_mreinterpret_*x2` macros removed because x2 struct types cannot
  fit a single `"tr"` inline asm constraint. Workaround: decompose with `mget`,
  reinterpret individual elements, reassemble with `mset`.

- **`mreinterpret` implementation rewritten** (data-preserving now): Single-register
  reinterpret macros previously discarded the source value and returned poison
  (`mundef`). Now uses empty inline asm with `"=tr"/"tr"` constraints for
  zero-copy type punning. Existing code that relied on the (broken) old behavior
  of getting an undefined value will now correctly get the source bits preserved.

- **Whole-register load/store gained mandatory stride operand**:
  - Old assembly: `th.mlmee8 md, (rs1)` — no stride
  - New assembly: `th.mlmee8 md, (rs1), rs2` — stride operand required
  - Old intrinsic: `mlme_internal8(ptr)` → new: `mlme_internal8(ptr, stride)`
  - Old intrinsic: `msme_internal8(val, ptr)` → new: `msme_internal8(val, ptr, stride)`
  - C API: `__riscv_th_mld_m_*(base, stride)` and `__riscv_th_mst_m_*(base, stride, val)`
    — stride parameter is new. Spill/reload uses stride=0 for contiguous access.

- **`__riscv_th_xmsize()` renamed to `__riscv_th_xmisa()`**: The old name
  referenced a CSR (`xmsize`) that does not exist in RVM 0.6. The canonical
  function is now `xmisa()`. `xmsize()` is preserved as a compatibility wrapper.

### Added

- **110+ new C API functions** in `<thead_matrix.h>`:
  - **A-tile stores** (11 types): `__riscv_th_mst_a_*(base, stride, val, m, k)`
  - **B-tile stores** (11 types): `__riscv_th_mst_b_*(base, stride, val, k, n)`
  - **Transposed A-tile loads** (11): `__riscv_th_mld_at_*(base, stride, m, k)`
  - **Transposed B-tile loads** (11): `__riscv_th_mld_bt_*(base, stride, k, n)`
  - **Transposed C-tile loads** (11): `__riscv_th_mld_ct_*(base, stride, m, n)`
  - **Transposed A/B/C-tile stores** (33): `__riscv_th_mst_at/bt/ct_*`
  - **Whole-register loads** (11): `__riscv_th_mld_m_*(base, stride)`
  - **Whole-register stores** (11): `__riscv_th_mst_m_*(base, stride, val)`
  - **Zero x2 constructors** (11): `__riscv_th_mzeros_*x2(m, n)`
  - **`mzero` aliases** (22): `__riscv_th_mzero_*` → `mzeros_*` (spec naming)
  - **Immediate config macros** (3): `msettilemi`, `msettileki`, `msettileni`

- **110 new Clang builtins** for all new load/store variants (store A/B,
  transposed load/store A/B/C, whole-register load/store).

- **18 Zmpanel panel CSRs registered** in `RISCVSystemOperands.td` under
  `FeatureVendorXTHeadZmpanel`. Panel CSRs now accessible by name in inline asm
  (e.g., `csrr a0, th.panel_m`). Total named CSRs: 31 (13 base + 18 Zmpanel).

- **Inline asm constraint validation** for matrix types: `SemaStmtAsm.cpp` now
  allows tied constraints between different RVM matrix subtypes, enabling the
  `mreinterpret` inline asm pattern.

- **Intrinsic memory attributes**: Load intrinsics gained `IntrReadMem,
  IntrArgMemOnly`; store intrinsics gained `IntrWriteMem, IntrArgMemOnly`,
  enabling better LLVM optimization (dead store elimination, load hoisting).

- **8 new test files** (~4200 lines):
  - `xtheadmatrix-api-coverage.c` (760 lines) — full C API function coverage
  - `xtheadmatrix-c-api-pipeline.c` (500 lines) — 21 end-to-end pipeline tests
  - `xtheadmatrix-spec-api-full.c` (981 lines) — complete spec API coverage
  - `xtheadmatrix-verification-fixes.c` (592 lines) — regression tests
  - `xtheadmatrix-zmpanel-api.c` (336 lines) — Zmpanel C header API tests
  - `xtheadmatrix-managed-ra-full.ll` (814 lines) — comprehensive RA tests
  - `xtheadmatrix-managed-ra-pipeline.ll` (291 lines) — 10 RA pipeline tests
  - `xtheadzmpanel-csr.s` (60 lines) — 18 panel CSR name round-trip tests

### Fixed

- **x2 matmul silently discarded data**: All `_x2` matmul variants previously
  only processed element 0, ignoring element 1. Fixed to process both elements.

- **mreinterpret discarded source data**: Previously returned `mundef` (poison),
  now correctly preserves register value via empty inline asm.

- **mreinterpret inline asm constraint**: Changed from tied `"0"` constraint
  (fails when input/output are different matrix subtypes) to separate
  `"=tr"/"tr"` constraint pair.

- **macOS toolchain build** (`riscv-toolchain-build.sh`): Fixed 5 build failures
  on macOS — bash 3.2 empty array expansion, `-static-libgcc` on clang, stale
  CMake cache, Darwin SDK detection for RISC-V cross-compile, Apple `libtool`
  rejection of ELF objects. All fixes cross-platform compatible.

- **Test CHECK patterns**: Fixed intrinsic name mangling (`.i64` suffix),
  `_Float16*` → `uint16_t*` pointer types.

### Changed

- **Spec erratum #5 documented**: Zmpanel compute encoding table in
  `zmpanel.adoc` mislabels bits [19:15] as `rs1=00000`; bits [19:18] carry
  `s_size`. Implementation correctly follows standard matmul encoding format.

- **Documentation and reports updated** through verification rounds 10–13:
  CSR address tables, x2 divergence docs, x2 reinterpret limitation,
  spec errata count (4→5), test count (26→27), mreinterpret constraint
  correction, comprehensive verification history entries.

- **Verification round 13 (comprehensive full-stack audit)**: 11 parallel
  verification agents independently audited every implementation layer against
  the golden RVM 0.6 spec. ALL 257 instruction encodings, ALL 31 CSRs, ALL
  267 ISel table entries, ALL 26 inline asm blocks, ALL 14 load/store
  families, and the complete managed RA model verified correct. No new bugs
  found. 13th verification round with 0 encoding errors across all rounds.
  Reports and documentation updated with verification status annotations.

## [aics_llvm_toolchain_v0.2] - 2026-03-13

### Added

- **XTHeadZmpanel extension (xtheadzmpanel v0.6)**: Full LLVM/Clang support for panel-aware 2x2 matrix tiling.
  - 30 fire-and-forget macro instructions for efficient 2x2 panel GEMM pipelines.
  - 12 config instructions (address/stride/dimension), 2 panel loads, 2 panel stores, 10 FP compute, 4 INT compute variants.
  - Panel load/store/compute instructions operate on implicit hardware state with no explicit matrix register operands.
  - New `THMI_PanelFireForget` ISel dispatch category.
  - Implicit Defs/Uses on matrix registers to prevent reordering.
  - Mixed-mode conflict detection (ManagedRA vs Zmpanel fire-and-forget).
  - `XTHeadZmpanel` decoder namespace for disassembler.
  - 30 LLVM intrinsics, 30 Clang builtins, C header API wrappers.
  - Tests: MC encoding/disassembly, intrinsics, builtins, header API, inline assembly.

- **x2 matrix types and builtins for XTHeadMatrix**: Software-level x2 type support for matmul variants requiring paired matrix operands.
  - New x2 types: `mfloat16x2_t`, `mint64x2_t`, etc., mapped to `{ target("riscv.matrix"), target("riscv.matrix") }` struct at IR level.
  - `mget`/`mset` builtins for x2 component access via extractvalue/insertvalue.
  - x2 matmul variants: FP16 `mfmacc.h` (x2 B), FP64 `mfmacc.d` (x2 dest), FP64 `mfmacc.d.s` (x2 dest), INT16->INT64 all 4 sign variants (x2 dest).
  - Spec-API wrappers extract component 0 for the hardware instruction.

## [aics_llvm_toolchain_v0.1] - 2026-03-06

Initial tagged release of AICS LLVM Toolchain with XTHeadMatrix (RVM 0.6) support.
