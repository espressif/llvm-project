# Changelog

All notable changes to the AICS LLVM Toolchain (XTHeadMatrix / RVM 0.6) are documented here.

## [aics_llvm_toolchain_v0.2.1] - 2026-03-20

### Added

- **Zmpanel panel CSR named access**: Registered 18 Zmpanel panel CSRs
  (0xcc4–0xcd5) in `RISCVSystemOperands.td` under `FeatureVendorXTHeadZmpanel`
  guard. Panel CSRs are now accessible by name in inline assembly (e.g.,
  `csrr a0, th.panel_m`). Total named CSRs: 31 (13 base + 18 Zmpanel).

- **Panel CSR test** (`xtheadzmpanel-csr.s`): 18 CSR name round-trip tests
  verifying all `th.custom_ctrl` through `th.addr_d` panel CSR names.

- **C API pipeline tests** (`xtheadmatrix-c-api-pipeline.c`, 21 tests): Comprehensive
  `<thead_matrix.h>` C-level API tests verifying CSR config emission, matmul operand
  ordering (A/B swap), dependency chains (matmul→EW→clip→store), register pressure
  (dual GEMM with shared A), transposed load/store variants, mixed INT+FP matmul in
  one function, EW FP chain (matmul→fmul→fadd→fmax), backward-compat aliases,
  partial/bypass matmul, column slide/broadcast, GPR↔matrix moves, float-int
  conversion round-trip, whole-register ops, and tile store variants.

- **Register allocation pipeline tests** (`xtheadmatrix-managed-ra-pipeline.ll`, 10 tests):
  LLVM IR → assembly tests verifying register class enforcement (A/B→tr, C→acc,
  EW→acc, conversions→acc, n4clip→acc, FP EW→acc), spill behavior under register
  pressure (3 ACC no-spill vs 5 ACC forced spill), dependency chain ordering
  preservation, chained matmul accumulate (same acc register), and dual matmul
  with both results live.

### Fixed

- **RVM matrix CSR addresses updated** (0x802–0x80a → 0x806–0x80e): Updated all 9
  read/write matrix CSR addresses in `RISCVSystemOperands.td` and `thead_matrix.h`
  to match the latest spike/hardware mapping. Read-only CSRs (0xcc0–0xcc3) and
  Zmpanel CSRs (0xcc4–0xcd5) are unchanged. Address mapping:
  xmcsr 0x806, mtilem 0x807, mtilen 0x808, mtilek 0x809, xmxrm 0x80a,
  xmsat 0x80b, xmfflags 0x80c, xmfrm 0x80d, xmsaten 0x80e.

- **Removed broken x2 reinterpret macros**: Removed 11 `__riscv_th_mreinterpret_*x2`
  macros from `thead_matrix.h` that could not work (x2 struct types cannot fit a
  single `"tr"` inline asm constraint). Users should decompose x2 values with
  `mget`, reinterpret individual elements, and reassemble with `mset`.

- **macOS toolchain build** (`riscv-toolchain-build.sh`):
  - Non-portable mode failed with "unbound variable" on macOS bash 3.2 due to empty
    array expansion with `set -u`. Fixed by always populating the flags array (both
    portable and non-portable branches set explicit values).
  - Portable mode failed with `clang++: error: unsupported option '-static-libgcc'`
    on macOS. Fixed by making portable flags OS-aware: `-static-libgcc` only added on
    Linux when host compiler is GCC (detected via `cc -v`); macOS gets static
    zlib/zstd only.
  - Stale CMake cache from previous `--portable` run caused non-portable build to
    fail. Fixed by always explicitly setting flags (ON or OFF) to override cached values.
  - Stage 2 (compiler-rt) failed on macOS: Darwin SDK detection activated instead of
    cross-compiling for RISC-V. Fixed by adding `CMAKE_SYSTEM_NAME=Generic`.
  - Stage 4 (C++ runtimes) failed on macOS: Apple `libtool` rejected RISC-V ELF
    objects. Root cause: LLVM's `UseLibtool.cmake` (included when `CMAKE_HOST_APPLE
    AND APPLE`) finds Apple libtool via `xcrun` and overrides `CMAKE_*_CREATE_STATIC_LIBRARY`.
    Fixed by using a CMake toolchain file with `CMAKE_SYSTEM_NAME=Linux` (makes
    `APPLE=false`, preventing `UseLibtool.cmake` from loading) and `CMAKE_AR=llvm-ar`.
  - All fixes are compatible with Linux + GCC, Linux + Clang, macOS + Clang, and
    macOS + GCC (Homebrew).

### Changed

- **`mfmacc_h_x2` spec divergence documented as intentional**: The spec puts x2 on
  src2 (B operand) for fp16 matmul; the implementation puts x2 on the accumulator.
  This is intentional for API consistency — all x2 matmul variants (fp16, fp64,
  int64) uniformly place x2 on the accumulator/result.

- **Spec erratum #5 added**: Zmpanel compute encoding table in `zmpanel.adoc`
  mislabels bits [19:15] as `rs1=00000`; bits [19:18] carry `s_size` (non-zero
  for fp16/bf16/fp32 source types). Implementation correctly follows standard
  matmul encoding format.

- **Documentation and reports updated** with verification rounds 10–12 findings:
  - Updated `xtheadmatrix-doc/RISCVXTHeadMatrix.md`: CSR address table, x2
    divergence documentation, x2 reinterpret limitation, spec errata (4→5),
    test count (26→27), verification history entries #8–#10.
  - Updated all report files (`00-overview.md`, `02-phase1-infrastructure.md`,
    `08-files-changed.md`, `09-instruction-encoding-reference.md`,
    `10-future-work.md`, `13-verification-and-fixes.md`) with new CSR addresses,
    round 12 findings, and corrected mreinterpret constraint documentation
    (actual: `"=tr"/"tr"`, not `"0"` tied constraint).

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
