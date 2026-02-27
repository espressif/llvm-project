# XTHeadMatrix (RVM 0.6) Extension

## Introduction

XTHeadMatrix is the T-Head (Alibaba DAMO Academy) RISC-V Matrix Extension,
version 0.6 (RVM 0.6). It provides hardware-accelerated matrix operations
including matrix multiply-accumulate, element-wise arithmetic, format
conversions, and data movement instructions. The extension is designed
primarily for machine learning inference and high-performance computing
workloads.

The extension defines 8 matrix registers -- 4 tile registers (`tr0`-`tr3`)
and 4 accumulator registers (`acc0`-`acc3`) -- encoded with 3 bits each.
It also defines 13 control/status registers for matrix configuration,
rounding modes, and hardware capability queries.

**Status**: Experimental (requires `+experimental-xtheadmatrix` to enable).

**Specification**: The official specification is the
[RISC-V Matrix Extension Spec (RVM)](https://github.com/AII-SDU/riscv-matrix-extension-spec)
maintained by C-SKY MicroSystems / T-Head.

**Implementation summary**: 227 instructions, 227 LLVM intrinsics, and
227 Clang builtins are implemented, covering MC-layer assembly/disassembly
and builtin-to-intrinsic code generation. No ISel/CodeGen patterns exist
yet -- programs must use builtins or inline assembly.

## Building the Compiler

### Prerequisites

- CMake >= 3.20
- A C++ compiler with C++17 support (GCC >= 8 or Clang >= 10)
- Python 3
- Ninja (recommended)

### CMake Configuration

```bash
cmake -S llvm -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_TARGETS_TO_BUILD=RISCV \
  -DLLVM_ENABLE_PROJECTS="clang" \
  -DCMAKE_INSTALL_PREFIX=/path/to/install
```

### Build

```bash
cmake --build build -- -j$(nproc) clang llvm-mc llvm-objdump
```

Or with Ninja directly:

```bash
cd build && ninja -j$(nproc) clang llvm-mc llvm-objdump
```

### Verify Extension Availability

```bash
./build/bin/clang --print-supported-extensions 2>&1 | grep xtheadmatrix
```

Expected output:

```
xtheadmatrix             0.6       'XTHeadMatrix' (T-Head Matrix Extension)
```

## Compiling Programs

### The `-march` Flag

XTHeadMatrix is an experimental vendor extension. Two flags are needed:

1. `-march=rv64gc_xtheadmatrix0p6` -- include the extension with its version
2. `-menable-experimental-extensions` -- opt in to experimental support

```bash
clang --target=riscv64 -march=rv64gc_xtheadmatrix0p6 \
  -menable-experimental-extensions ...
```

The `0p6` suffix denotes version 0.6 and is required for experimental
extensions. Combined with the vector extension:

```bash
clang --target=riscv64 -march=rv64gcv_xtheadmatrix0p6 \
  -menable-experimental-extensions ...
```

### Assembling `.s` Files

```bash
clang --target=riscv64 -march=rv64gc_xtheadmatrix0p6 \
  -menable-experimental-extensions -c matrix_kernel.s -o matrix_kernel.o
```

Or using `llvm-mc` directly:

```bash
llvm-mc -triple=riscv64 -mattr=+experimental-xtheadmatrix \
  -filetype=obj matrix_kernel.s -o matrix_kernel.o
```

### Compiling C/C++ with Builtins

The builtins compile C code to LLVM IR containing matrix intrinsics:

```bash
# Generate LLVM IR (works today)
clang --target=riscv64 -march=rv64gc_xtheadmatrix0p6 \
  -menable-experimental-extensions -emit-llvm -S matrix_kernel.c -o matrix_kernel.ll
```

**Note**: Full compilation to native assembly (`-S`) or object code (`-c`)
is not yet supported because no ISel/CodeGen patterns exist for the matrix
intrinsics. This is planned for future work. Assembly files can be
assembled directly (see above).

### Combining RVV and RVM

Both the vector (V) and matrix (XTHeadMatrix) extensions can be enabled
simultaneously:

```bash
clang --target=riscv64 -march=rv64gcv_xtheadmatrix0p6 \
  -menable-experimental-extensions -c mixed_kernel.c -o mixed_kernel.o
```

### Inline Assembly

Matrix registers can be used in inline assembly:

```c
// Read a CSR
unsigned long isa;
asm volatile("csrr %0, th.xmisa" : "=r"(isa));

// Configure and perform a matrix operation
asm volatile(
    "th.msettilemi 4\n\t"
    "th.msettileni 4\n\t"
    "th.msettileki 4\n\t"
    "th.mzero tr0\n\t"
    "th.mlae32 tr1, (%0), %1\n\t"
    "th.mlbe32 tr2, (%2), %3\n\t"
    "th.mfmacc.s acc0, tr1, tr2\n\t"
    "th.msae32 acc0, (%4), %5"
    :
    : "r"(a_ptr), "r"(a_stride),
      "r"(b_ptr), "r"(b_stride),
      "r"(c_ptr), "r"(c_stride)
    : "memory"
);
```

### Cross-Compilation Notes

For cross-compiling to a RISC-V target from a host machine:

```bash
clang --target=riscv64-unknown-linux-gnu \
  --sysroot=/path/to/riscv-sysroot \
  -march=rv64gc_xtheadmatrix0p6 \
  -menable-experimental-extensions \
  -c kernel.c -o kernel.o
```

When linking, ensure the linker can find RISC-V libraries via `--sysroot`
or `-L` flags. Since no runtime library support exists for XTHeadMatrix
types, all matrix operations must go through builtins or inline assembly.

## Programming Model

### Overview

Matrix register state is implicit -- the 8 matrix registers (`tr0`-`tr3`,
`acc0`-`acc3`) are not exposed as C types in the current low-level builtin
API. All builtins operate on hardware matrix registers directly. The
hardware manages register allocation based on the matrix dimension
configuration.

### Typical Workflow

A typical matrix computation follows these steps:

1. **Configure** -- Set the tile dimensions (M, K, N) using `msettile*`
2. **Zero** -- Initialize accumulators using `mzero`
3. **Load** -- Load matrix tiles from memory using `mla`/`mlb`/`mlc`
4. **Compute** -- Perform matrix multiply or element-wise operations
5. **Store** -- Store results back to memory using `msa`/`msb`/`msc`
6. **Release** -- Release the matrix unit using `mrelease`

### Matrix Dimensions

The matrix unit operates on tiles whose dimensions are configured via CSRs:

- **M** (rows of A and C): set by `msettilemi`/`msettilem`
- **K** (columns of A, rows of B): set by `msettileki`/`msettilek`
- **N** (columns of B and C): set by `msettileni`/`msettilen`

The hardware clamps these values to implementation-defined maximums.
Query `th.xtlenb` and `th.xtrlenb` CSRs for the hardware tile dimensions.

### Low-Level vs. Spec API

The specification defines a higher-level intrinsic API with C++ matrix
types (`mint32_t`, `mfloat16_t`, etc.) and overloaded functions that carry
dimension parameters. Our current implementation provides **low-level
instruction-mapped builtins** where:

- Matrix registers are implicit (not returned/passed as C values)
- Each builtin maps 1:1 to a single assembly instruction
- Users manage register allocation through instruction ordering

The higher-level spec API (with `__riscv_th_mld`, `__riscv_th_mmul_mm`,
etc.) is a future goal.

## Builtins Reference

All builtins use the `__builtin_riscv_th_` prefix and are available when
the `xtheadmatrix` target feature is enabled. Matrix registers are implicit
operands -- only GPR values (pointers, strides, scalars, immediates)
appear as explicit builtin parameters.

### Configuration (7 builtins)

| Builtin | Prototype | Assembly | Description |
|---------|-----------|----------|-------------|
| `__builtin_riscv_th_mrelease` | `void()` | `th.mrelease` | Release the matrix unit |
| `__builtin_riscv_th_msettilemi` | `void(size_t)` | `th.msettilemi uimm` | Set tile M dimension (immediate) |
| `__builtin_riscv_th_msettilem` | `void(size_t)` | `th.msettilem rs` | Set tile M dimension (register) |
| `__builtin_riscv_th_msettileki` | `void(size_t)` | `th.msettileki uimm` | Set tile K dimension (immediate) |
| `__builtin_riscv_th_msettilek` | `void(size_t)` | `th.msettilek rs` | Set tile K dimension (register) |
| `__builtin_riscv_th_msettileni` | `void(size_t)` | `th.msettileni uimm` | Set tile N dimension (immediate) |
| `__builtin_riscv_th_msettilen` | `void(size_t)` | `th.msettilen rs` | Set tile N dimension (register) |

### Load Instructions (28 builtins)

All load builtins load data from memory into matrix registers.

#### Element-Stride Loads (12 builtins)

Prototype: `void(void *base, size_t stride)`

| Builtin | Assembly | Description |
|---------|----------|-------------|
| `__builtin_riscv_th_mlae{8,16,32,64}` | `th.mlae{8,16,32,64} md, (rs1), rs2` | Load A matrix, element stride |
| `__builtin_riscv_th_mlbe{8,16,32,64}` | `th.mlbe{8,16,32,64} md, (rs1), rs2` | Load B matrix, element stride |
| `__builtin_riscv_th_mlce{8,16,32,64}` | `th.mlce{8,16,32,64} md, (rs1), rs2` | Load C matrix, element stride |

#### Tile-Stride (Transposed) Loads (12 builtins)

Prototype: `void(void *base, size_t stride)`

| Builtin | Assembly | Description |
|---------|----------|-------------|
| `__builtin_riscv_th_mlate{8,16,32,64}` | `th.mlate{8,16,32,64} md, (rs1), rs2` | Load A transposed, tile stride |
| `__builtin_riscv_th_mlbte{8,16,32,64}` | `th.mlbte{8,16,32,64} md, (rs1), rs2` | Load B transposed, tile stride |
| `__builtin_riscv_th_mlcte{8,16,32,64}` | `th.mlcte{8,16,32,64} md, (rs1), rs2` | Load C transposed, tile stride |

#### Whole-Register Loads (4 builtins)

Prototype: `void(void *base)`

| Builtin | Assembly | Description |
|---------|----------|-------------|
| `__builtin_riscv_th_mlme{8,16,32,64}` | `th.mlme{8,16,32,64} md, (rs1)` | Load whole matrix register |

### Store Instructions (28 builtins)

All store builtins store matrix register data to memory.

#### Element-Stride Stores (12 builtins)

Prototype: `void(void *base, size_t stride)`

| Builtin | Assembly | Description |
|---------|----------|-------------|
| `__builtin_riscv_th_msae{8,16,32,64}` | `th.msae{8,16,32,64} ms3, (rs1), rs2` | Store A matrix, element stride |
| `__builtin_riscv_th_msbe{8,16,32,64}` | `th.msbe{8,16,32,64} ms3, (rs1), rs2` | Store B matrix, element stride |
| `__builtin_riscv_th_msce{8,16,32,64}` | `th.msce{8,16,32,64} ms3, (rs1), rs2` | Store C matrix, element stride |

#### Tile-Stride (Transposed) Stores (12 builtins)

Prototype: `void(void *base, size_t stride)`

| Builtin | Assembly | Description |
|---------|----------|-------------|
| `__builtin_riscv_th_msate{8,16,32,64}` | `th.msate{8,16,32,64} ms3, (rs1), rs2` | Store A transposed, tile stride |
| `__builtin_riscv_th_msbte{8,16,32,64}` | `th.msbte{8,16,32,64} ms3, (rs1), rs2` | Store B transposed, tile stride |
| `__builtin_riscv_th_mscte{8,16,32,64}` | `th.mscte{8,16,32,64} ms3, (rs1), rs2` | Store C transposed, tile stride |

#### Whole-Register Stores (4 builtins)

Prototype: `void(void *base)`

| Builtin | Assembly | Description |
|---------|----------|-------------|
| `__builtin_riscv_th_msme{8,16,32,64}` | `th.msme{8,16,32,64} ms3, (rs1)` | Store whole matrix register |

### Matrix Multiply-Accumulate (27 builtins)

All matrix multiply builtins have prototype `void()` -- all operands
(md, ms2, ms1) are implicit matrix registers.

#### FP Matrix Multiply (13 builtins)

| Builtin | Assembly | Description |
|---------|----------|-------------|
| `__builtin_riscv_th_mfmacc_h` | `th.mfmacc.h` | FP16 * FP16 &rarr; FP16 |
| `__builtin_riscv_th_mfmacc_s` | `th.mfmacc.s` | FP32 * FP32 &rarr; FP32 |
| `__builtin_riscv_th_mfmacc_d` | `th.mfmacc.d` | FP64 * FP64 &rarr; FP64 |
| `__builtin_riscv_th_mfmacc_h_e4` | `th.mfmacc.h.e4` | E4M3 * E4M3 &rarr; FP16 |
| `__builtin_riscv_th_mfmacc_h_e5` | `th.mfmacc.h.e5` | E5M2 * E5M2 &rarr; FP16 |
| `__builtin_riscv_th_mfmacc_bf16_e4` | `th.mfmacc.bf16.e4` | E4M3 * E4M3 &rarr; BF16 |
| `__builtin_riscv_th_mfmacc_bf16_e5` | `th.mfmacc.bf16.e5` | E5M2 * E5M2 &rarr; BF16 |
| `__builtin_riscv_th_mfmacc_s_h` | `th.mfmacc.s.h` | FP16 * FP16 &rarr; FP32 |
| `__builtin_riscv_th_mfmacc_s_bf16` | `th.mfmacc.s.bf16` | BF16 * BF16 &rarr; FP32 |
| `__builtin_riscv_th_mfmacc_s_e4` | `th.mfmacc.s.e4` | E4M3 * E4M3 &rarr; FP32 |
| `__builtin_riscv_th_mfmacc_s_e5` | `th.mfmacc.s.e5` | E5M2 * E5M2 &rarr; FP32 |
| `__builtin_riscv_th_mfmacc_s_tf32` | `th.mfmacc.s.tf32` | TF32 * TF32 &rarr; FP32 |
| `__builtin_riscv_th_mfmacc_d_s` | `th.mfmacc.d.s` | FP32 * FP32 &rarr; FP64 |

#### Integer Matrix Multiply (8 builtins)

| Builtin | Assembly | Description |
|---------|----------|-------------|
| `__builtin_riscv_th_mmacc_w_b` | `th.mmacc.w.b` | INT8 * INT8 &rarr; INT32 (signed) |
| `__builtin_riscv_th_mmaccu_w_b` | `th.mmaccu.w.b` | UINT8 * UINT8 &rarr; INT32 |
| `__builtin_riscv_th_mmaccus_w_b` | `th.mmaccus.w.b` | UINT8 * INT8 &rarr; INT32 |
| `__builtin_riscv_th_mmaccsu_w_b` | `th.mmaccsu.w.b` | INT8 * UINT8 &rarr; INT32 |
| `__builtin_riscv_th_mmacc_d_h` | `th.mmacc.d.h` | INT16 * INT16 &rarr; INT64 (signed) |
| `__builtin_riscv_th_mmaccu_d_h` | `th.mmaccu.d.h` | UINT16 * UINT16 &rarr; INT64 |
| `__builtin_riscv_th_mmaccus_d_h` | `th.mmaccus.d.h` | UINT16 * INT16 &rarr; INT64 |
| `__builtin_riscv_th_mmaccsu_d_h` | `th.mmaccsu.d.h` | INT16 * UINT16 &rarr; INT64 |

#### Partial Integer Matrix Multiply (4 builtins)

| Builtin | Assembly | Description |
|---------|----------|-------------|
| `__builtin_riscv_th_pmmacc_w_b` | `th.pmmacc.w.b` | Partial INT8 * INT8 &rarr; INT32 (signed) |
| `__builtin_riscv_th_pmmaccu_w_b` | `th.pmmaccu.w.b` | Partial UINT8 * UINT8 &rarr; INT32 |
| `__builtin_riscv_th_pmmaccus_w_b` | `th.pmmaccus.w.b` | Partial UINT8 * INT8 &rarr; INT32 |
| `__builtin_riscv_th_pmmaccsu_w_b` | `th.pmmaccsu.w.b` | Partial INT8 * UINT8 &rarr; INT32 |

#### Bypass Matrix Multiply (2 builtins)

| Builtin | Assembly | Description |
|---------|----------|-------------|
| `__builtin_riscv_th_mmacc_w_bp` | `th.mmacc.w.bp` | Bypass INT8 matmul (signed) |
| `__builtin_riscv_th_mmaccu_w_bp` | `th.mmaccu.w.bp` | Bypass INT8 matmul (unsigned) |

### Data Movement (17 builtins)

#### Zero (4 builtins)

Prototype: `void()`

| Builtin | Assembly | Description |
|---------|----------|-------------|
| `__builtin_riscv_th_mzero` | `th.mzero md` | Zero 1 matrix register |
| `__builtin_riscv_th_mzero2r` | `th.mzero2r md` | Zero 2 matrix registers |
| `__builtin_riscv_th_mzero4r` | `th.mzero4r md` | Zero 4 matrix registers |
| `__builtin_riscv_th_mzero8r` | `th.mzero8r md` | Zero all 8 matrix registers |

#### Matrix-to-Matrix Move (1 builtin)

Prototype: `void()`

| Builtin | Assembly | Description |
|---------|----------|-------------|
| `__builtin_riscv_th_mmov_mm` | `th.mmov.mm md, ms1` | Copy matrix register |

#### Matrix-to-GPR Move (4 builtins)

Prototype: `size_t(size_t index)`

| Builtin | Assembly | Description |
|---------|----------|-------------|
| `__builtin_riscv_th_mmovb_x_m` | `th.mmovb.x.m rd, ms2, rs1` | Read byte from matrix |
| `__builtin_riscv_th_mmovh_x_m` | `th.mmovh.x.m rd, ms2, rs1` | Read halfword from matrix |
| `__builtin_riscv_th_mmovw_x_m` | `th.mmovw.x.m rd, ms2, rs1` | Read word from matrix |
| `__builtin_riscv_th_mmovd_x_m` | `th.mmovd.x.m rd, ms2, rs1` | Read doubleword from matrix |

#### GPR-to-Matrix Move (4 builtins)

Prototype: `void(size_t data, size_t index)`

| Builtin | Assembly | Description |
|---------|----------|-------------|
| `__builtin_riscv_th_mmovb_m_x` | `th.mmovb.m.x md, rs2, rs1` | Write byte to matrix |
| `__builtin_riscv_th_mmovh_m_x` | `th.mmovh.m.x md, rs2, rs1` | Write halfword to matrix |
| `__builtin_riscv_th_mmovw_m_x` | `th.mmovw.m.x md, rs2, rs1` | Write word to matrix |
| `__builtin_riscv_th_mmovd_m_x` | `th.mmovd.m.x md, rs2, rs1` | Write doubleword to matrix |

#### GPR Broadcast to Matrix Column (4 builtins)

Prototype: `void(size_t data)`

| Builtin | Assembly | Description |
|---------|----------|-------------|
| `__builtin_riscv_th_mdupb_m_x` | `th.mdupb.m.x md, rs2` | Broadcast byte to matrix |
| `__builtin_riscv_th_mduph_m_x` | `th.mduph.m.x md, rs2` | Broadcast halfword to matrix |
| `__builtin_riscv_th_mdupw_m_x` | `th.mdupw.m.x md, rs2` | Broadcast word to matrix |
| `__builtin_riscv_th_mdupd_m_x` | `th.mdupd.m.x md, rs2` | Broadcast doubleword to matrix |

### Pack (3 builtins)

Prototype: `void()`

| Builtin | Assembly | Description |
|---------|----------|-------------|
| `__builtin_riscv_th_mpack` | `th.mpack md, ms2, ms1` | Pack low halves |
| `__builtin_riscv_th_mpackhl` | `th.mpackhl md, ms2, ms1` | Pack high-low halves |
| `__builtin_riscv_th_mpackhh` | `th.mpackhh md, ms2, ms1` | Pack high halves |

### Slide and Broadcast (15 builtins)

All take a 3-bit unsigned immediate operand.

#### Row Slide (2 builtins)

Prototype: `void(unsigned int imm3)`

| Builtin | Assembly | Description |
|---------|----------|-------------|
| `__builtin_riscv_th_mrslidedown` | `th.mrslidedown md, ms1, imm3` | Slide rows down |
| `__builtin_riscv_th_mrslideup` | `th.mrslideup md, ms1, imm3` | Slide rows up |

#### Column Slide (8 builtins)

Prototype: `void(unsigned int imm3)`

| Builtin | Assembly | Description |
|---------|----------|-------------|
| `__builtin_riscv_th_mcslidedown_b` | `th.mcslidedown.b md, ms1, imm3` | Slide columns down (byte) |
| `__builtin_riscv_th_mcslideup_b` | `th.mcslideup.b md, ms1, imm3` | Slide columns up (byte) |
| `__builtin_riscv_th_mcslidedown_h` | `th.mcslidedown.h md, ms1, imm3` | Slide columns down (half) |
| `__builtin_riscv_th_mcslideup_h` | `th.mcslideup.h md, ms1, imm3` | Slide columns up (half) |
| `__builtin_riscv_th_mcslidedown_w` | `th.mcslidedown.w md, ms1, imm3` | Slide columns down (word) |
| `__builtin_riscv_th_mcslideup_w` | `th.mcslideup.w md, ms1, imm3` | Slide columns up (word) |
| `__builtin_riscv_th_mcslidedown_d` | `th.mcslidedown.d md, ms1, imm3` | Slide columns down (dword) |
| `__builtin_riscv_th_mcslideup_d` | `th.mcslideup.d md, ms1, imm3` | Slide columns up (dword) |

#### Broadcast (5 builtins)

Prototype: `void(unsigned int imm3)`

| Builtin | Assembly | Description |
|---------|----------|-------------|
| `__builtin_riscv_th_mrbca_mv_i` | `th.mrbca.mv.i md, ms1, imm3` | Row broadcast |
| `__builtin_riscv_th_mcbcab_mv_i` | `th.mcbcab.mv.i md, ms1, imm3` | Column broadcast (byte) |
| `__builtin_riscv_th_mcbcah_mv_i` | `th.mcbcah.mv.i md, ms1, imm3` | Column broadcast (half) |
| `__builtin_riscv_th_mcbcaw_mv_i` | `th.mcbcaw.mv.i md, ms1, imm3` | Column broadcast (word) |
| `__builtin_riscv_th_mcbcad_mv_i` | `th.mcbcad.mv.i md, ms1, imm3` | Column broadcast (dword) |

### FP Format Conversions (26 builtins)

All have prototype `void()`. The `l` (low) and `h` (high) suffixes
indicate which half of a widening/narrowing pair is operated on.

#### FP8 &harr; FP16 (8 builtins)

| Builtin | Assembly | Description |
|---------|----------|-------------|
| `__builtin_riscv_th_mfcvtl_h_e4` | `th.mfcvtl.h.e4` | E4M3 &rarr; FP16, low half |
| `__builtin_riscv_th_mfcvth_h_e4` | `th.mfcvth.h.e4` | E4M3 &rarr; FP16, high half |
| `__builtin_riscv_th_mfcvtl_h_e5` | `th.mfcvtl.h.e5` | E5M2 &rarr; FP16, low half |
| `__builtin_riscv_th_mfcvth_h_e5` | `th.mfcvth.h.e5` | E5M2 &rarr; FP16, high half |
| `__builtin_riscv_th_mfcvtl_e4_h` | `th.mfcvtl.e4.h` | FP16 &rarr; E4M3, low half |
| `__builtin_riscv_th_mfcvth_e4_h` | `th.mfcvth.e4.h` | FP16 &rarr; E4M3, high half |
| `__builtin_riscv_th_mfcvtl_e5_h` | `th.mfcvtl.e5.h` | FP16 &rarr; E5M2, low half |
| `__builtin_riscv_th_mfcvth_e5_h` | `th.mfcvth.e5.h` | FP16 &rarr; E5M2, high half |

#### FP16/BF16 &harr; FP32 (8 builtins)

| Builtin | Assembly | Description |
|---------|----------|-------------|
| `__builtin_riscv_th_mfcvtl_s_h` | `th.mfcvtl.s.h` | FP16 &rarr; FP32, low half |
| `__builtin_riscv_th_mfcvth_s_h` | `th.mfcvth.s.h` | FP16 &rarr; FP32, high half |
| `__builtin_riscv_th_mfcvtl_h_s` | `th.mfcvtl.h.s` | FP32 &rarr; FP16, low half |
| `__builtin_riscv_th_mfcvth_h_s` | `th.mfcvth.h.s` | FP32 &rarr; FP16, high half |
| `__builtin_riscv_th_mfcvtl_s_bf16` | `th.mfcvtl.s.bf16` | BF16 &rarr; FP32, low half |
| `__builtin_riscv_th_mfcvth_s_bf16` | `th.mfcvth.s.bf16` | BF16 &rarr; FP32, high half |
| `__builtin_riscv_th_mfcvtl_bf16_s` | `th.mfcvtl.bf16.s` | FP32 &rarr; BF16, low half |
| `__builtin_riscv_th_mfcvth_bf16_s` | `th.mfcvth.bf16.s` | FP32 &rarr; BF16, high half |

#### FP32 &harr; FP8 (4 builtins)

| Builtin | Assembly | Description |
|---------|----------|-------------|
| `__builtin_riscv_th_mfcvtl_e4_s` | `th.mfcvtl.e4.s` | FP32 &rarr; E4M3, low half |
| `__builtin_riscv_th_mfcvth_e4_s` | `th.mfcvth.e4.s` | FP32 &rarr; E4M3, high half |
| `__builtin_riscv_th_mfcvtl_e5_s` | `th.mfcvtl.e5.s` | FP32 &rarr; E5M2, low half |
| `__builtin_riscv_th_mfcvth_e5_s` | `th.mfcvth.e5.s` | FP32 &rarr; E5M2, high half |

#### FP32 &harr; FP64 (4 builtins)

| Builtin | Assembly | Description |
|---------|----------|-------------|
| `__builtin_riscv_th_mfcvtl_d_s` | `th.mfcvtl.d.s` | FP32 &rarr; FP64, low half |
| `__builtin_riscv_th_mfcvth_d_s` | `th.mfcvth.d.s` | FP32 &rarr; FP64, high half |
| `__builtin_riscv_th_mfcvtl_s_d` | `th.mfcvtl.s.d` | FP64 &rarr; FP32, low half |
| `__builtin_riscv_th_mfcvth_s_d` | `th.mfcvth.s.d` | FP64 &rarr; FP32, high half |

#### TF32 &harr; FP32 (2 builtins)

| Builtin | Assembly | Description |
|---------|----------|-------------|
| `__builtin_riscv_th_mfcvt_s_tf32` | `th.mfcvt.s.tf32` | TF32 &rarr; FP32 |
| `__builtin_riscv_th_mfcvt_tf32_s` | `th.mfcvt.tf32.s` | FP32 &rarr; TF32 |

### Float-Integer Conversions (12 builtins)

All have prototype `void()`.

#### INT8 &harr; FP16 (8 builtins)

| Builtin | Assembly | Description |
|---------|----------|-------------|
| `__builtin_riscv_th_mufcvtl_h_b` | `th.mufcvtl.h.b` | UINT8 &rarr; FP16, low half |
| `__builtin_riscv_th_mufcvth_h_b` | `th.mufcvth.h.b` | UINT8 &rarr; FP16, high half |
| `__builtin_riscv_th_msfcvtl_h_b` | `th.msfcvtl.h.b` | SINT8 &rarr; FP16, low half |
| `__builtin_riscv_th_msfcvth_h_b` | `th.msfcvth.h.b` | SINT8 &rarr; FP16, high half |
| `__builtin_riscv_th_mfucvtl_b_h` | `th.mfucvtl.b.h` | FP16 &rarr; UINT8, low half |
| `__builtin_riscv_th_mfucvth_b_h` | `th.mfucvth.b.h` | FP16 &rarr; UINT8, high half |
| `__builtin_riscv_th_mfscvtl_b_h` | `th.mfscvtl.b.h` | FP16 &rarr; SINT8, low half |
| `__builtin_riscv_th_mfscvth_b_h` | `th.mfscvth.b.h` | FP16 &rarr; SINT8, high half |

#### INT32 &harr; FP32 (4 builtins)

| Builtin | Assembly | Description |
|---------|----------|-------------|
| `__builtin_riscv_th_msfcvt_s_w` | `th.msfcvt.s.w` | SINT32 &rarr; FP32 |
| `__builtin_riscv_th_mufcvt_s_w` | `th.mufcvt.s.w` | UINT32 &rarr; FP32 |
| `__builtin_riscv_th_mfscvt_w_s` | `th.mfscvt.w.s` | FP32 &rarr; SINT32 |
| `__builtin_riscv_th_mfucvt_w_s` | `th.mfucvt.w.s` | FP32 &rarr; UINT32 |

### Fixed-Point Clip (8 builtins)

#### Matrix-Matrix (4 builtins)

Prototype: `void()`

| Builtin | Assembly | Description |
|---------|----------|-------------|
| `__builtin_riscv_th_mn4clipl_w_mm` | `th.mn4clipl.w.mm` | N4 clip low, signed |
| `__builtin_riscv_th_mn4cliph_w_mm` | `th.mn4cliph.w.mm` | N4 clip high, signed |
| `__builtin_riscv_th_mn4cliplu_w_mm` | `th.mn4cliplu.w.mm` | N4 clip low, unsigned |
| `__builtin_riscv_th_mn4cliphu_w_mm` | `th.mn4cliphu.w.mm` | N4 clip high, unsigned |

#### Matrix-Vector with Immediate (4 builtins)

Prototype: `void(unsigned int imm3)`

| Builtin | Assembly | Description |
|---------|----------|-------------|
| `__builtin_riscv_th_mn4clipl_w_mv_i` | `th.mn4clipl.w.mv.i` | N4 clip low, signed (imm) |
| `__builtin_riscv_th_mn4cliph_w_mv_i` | `th.mn4cliph.w.mv.i` | N4 clip high, signed (imm) |
| `__builtin_riscv_th_mn4cliplu_w_mv_i` | `th.mn4cliplu.w.mv.i` | N4 clip low, unsigned (imm) |
| `__builtin_riscv_th_mn4cliphu_w_mv_i` | `th.mn4cliphu.w.mv.i` | N4 clip high, unsigned (imm) |

### Packed Conversions (4 builtins)

Prototype: `void()`

| Builtin | Assembly | Description |
|---------|----------|-------------|
| `__builtin_riscv_th_mucvtl_b_p` | `th.mucvtl.b.p` | Unsigned packed to byte, low |
| `__builtin_riscv_th_mscvtl_b_p` | `th.mscvtl.b.p` | Signed packed to byte, low |
| `__builtin_riscv_th_mucvth_b_p` | `th.mucvth.b.p` | Unsigned packed to byte, high |
| `__builtin_riscv_th_mscvth_b_p` | `th.mscvth.b.p` | Signed packed to byte, high |

### Integer Element-Wise Arithmetic (22 builtins)

Each operation has two variants:
- `.w.mm` (matrix-matrix): Prototype `void()`
- `.w.mv.i` (matrix-vector, immediate index): Prototype `void(unsigned int imm3)`

| Operation | `.w.mm` Builtin | `.w.mv.i` Builtin | Description |
|-----------|-----------------|-------------------|-------------|
| Add | `__builtin_riscv_th_madd_w_mm` | `__builtin_riscv_th_madd_w_mv_i` | Element-wise add |
| Sub | `__builtin_riscv_th_msub_w_mm` | `__builtin_riscv_th_msub_w_mv_i` | Element-wise subtract |
| Mul | `__builtin_riscv_th_mmul_w_mm` | `__builtin_riscv_th_mmul_w_mv_i` | Element-wise multiply (low) |
| MulH | `__builtin_riscv_th_mmulh_w_mm` | `__builtin_riscv_th_mmulh_w_mv_i` | Element-wise multiply (high) |
| Max | `__builtin_riscv_th_mmax_w_mm` | `__builtin_riscv_th_mmax_w_mv_i` | Signed maximum |
| UMax | `__builtin_riscv_th_mumax_w_mm` | `__builtin_riscv_th_mumax_w_mv_i` | Unsigned maximum |
| Min | `__builtin_riscv_th_mmin_w_mm` | `__builtin_riscv_th_mmin_w_mv_i` | Signed minimum |
| UMin | `__builtin_riscv_th_mumin_w_mm` | `__builtin_riscv_th_mumin_w_mv_i` | Unsigned minimum |
| SRL | `__builtin_riscv_th_msrl_w_mm` | `__builtin_riscv_th_msrl_w_mv_i` | Shift right logical |
| SLL | `__builtin_riscv_th_msll_w_mm` | `__builtin_riscv_th_msll_w_mv_i` | Shift left logical |
| SRA | `__builtin_riscv_th_msra_w_mm` | `__builtin_riscv_th_msra_w_mv_i` | Shift right arithmetic |

### FP Element-Wise Arithmetic (30 builtins)

Each operation has three precision levels (`.h` FP16, `.s` FP32, `.d` FP64)
and two variants (`.mm` matrix-matrix, `.mv.i` matrix-vector with immediate).

- `.mm` Prototype: `void()`
- `.mv.i` Prototype: `void(unsigned int imm3)`

| Operation | `.h.mm` | `.h.mv.i` | `.s.mm` | `.s.mv.i` | `.d.mm` | `.d.mv.i` |
|-----------|---------|-----------|---------|-----------|---------|-----------|
| FP Add | `mfadd_h_mm` | `mfadd_h_mv_i` | `mfadd_s_mm` | `mfadd_s_mv_i` | `mfadd_d_mm` | `mfadd_d_mv_i` |
| FP Sub | `mfsub_h_mm` | `mfsub_h_mv_i` | `mfsub_s_mm` | `mfsub_s_mv_i` | `mfsub_d_mm` | `mfsub_d_mv_i` |
| FP Mul | `mfmul_h_mm` | `mfmul_h_mv_i` | `mfmul_s_mm` | `mfmul_s_mv_i` | `mfmul_d_mm` | `mfmul_d_mv_i` |
| FP Max | `mfmax_h_mm` | `mfmax_h_mv_i` | `mfmax_s_mm` | `mfmax_s_mv_i` | `mfmax_d_mm` | `mfmax_d_mv_i` |
| FP Min | `mfmin_h_mm` | `mfmin_h_mv_i` | `mfmin_s_mm` | `mfmin_s_mv_i` | `mfmin_d_mm` | `mfmin_d_mv_i` |

All builtins above are prefixed with `__builtin_riscv_th_`. For example,
the full name for FP32 add (matrix-matrix) is
`__builtin_riscv_th_mfadd_s_mm`.

## Code Examples

### Example 1: INT8 GEMM Kernel

This example performs an INT8 matrix multiply-accumulate using builtins.

```c
#include <stddef.h>
#include <stdint.h>

// INT8 GEMM: C[M,N] += A[M,K] * B[K,N] (fixed tile size)
void int8_gemm_4x4(const int8_t *A, size_t a_stride,
                    const int8_t *B, size_t b_stride,
                    int32_t *C, size_t c_stride) {
    // Step 1: Configure tile dimensions (compile-time constants for imm variants)
    __builtin_riscv_th_msettilemi(4);
    __builtin_riscv_th_msettileki(4);
    __builtin_riscv_th_msettileni(4);

    // Step 2: Zero the accumulator
    __builtin_riscv_th_mzero();

    // Step 3: Load matrix tiles
    __builtin_riscv_th_mlae8((void *)A, a_stride);    // Load A
    __builtin_riscv_th_mlbe8((void *)B, b_stride);    // Load B

    // Step 4: Matrix multiply-accumulate (signed INT8 -> INT32)
    __builtin_riscv_th_mmacc_w_b();                   // acc0 += A * B

    // Step 5: Store result
    __builtin_riscv_th_msae32((void *)C, c_stride);   // Store to C

    // Step 6: Release
    __builtin_riscv_th_mrelease();
}

// Runtime-variable tile dimensions use the register variants
void int8_gemm(const int8_t *A, size_t a_stride,
               const int8_t *B, size_t b_stride,
               int32_t *C, size_t c_stride,
               size_t M, size_t K, size_t N) {
    __builtin_riscv_th_msettilem(M);    // register variant for runtime values
    __builtin_riscv_th_msettilek(K);
    __builtin_riscv_th_msettilen(N);

    __builtin_riscv_th_mzero();
    __builtin_riscv_th_mlae8((void *)A, a_stride);
    __builtin_riscv_th_mlbe8((void *)B, b_stride);
    __builtin_riscv_th_mmacc_w_b();
    __builtin_riscv_th_msae32((void *)C, c_stride);
    __builtin_riscv_th_mrelease();
}
```

Compile to LLVM IR with:
```bash
clang --target=riscv64 -march=rv64gc_xtheadmatrix0p6 \
  -menable-experimental-extensions -O2 -emit-llvm -S int8_gemm.c -o int8_gemm.ll
```

### Example 2: FP16 Element-Wise Operations

```c
#include <stddef.h>

// Element-wise FP16: C = A + B, then C = max(C, D)
void fp16_ewise(void *A, void *B, void *C, void *D,
                size_t stride, size_t M, size_t N) {
    __builtin_riscv_th_msettilem(M);
    __builtin_riscv_th_msettilen(N);
    __builtin_riscv_th_msettilek(N);

    // Load matrices
    __builtin_riscv_th_mlae16(A, stride);
    __builtin_riscv_th_mlbe16(B, stride);

    // Element-wise add: C = A + B
    __builtin_riscv_th_mfadd_h_mm();

    // Load D for max operation
    __builtin_riscv_th_mlce16(D, stride);

    // Element-wise max
    __builtin_riscv_th_mfmax_h_mm();

    // Store result
    __builtin_riscv_th_msae16(C, stride);

    __builtin_riscv_th_mrelease();
}
```

### Example 3: Inline Assembly CSR Access

```c
#include <stddef.h>
#include <stdio.h>

void query_matrix_hw(void) {
    unsigned long xmisa, xtlenb, xtrlenb, xalenb;

    asm volatile("csrr %0, th.xmisa"   : "=r"(xmisa));
    asm volatile("csrr %0, th.xtlenb"  : "=r"(xtlenb));
    asm volatile("csrr %0, th.xtrlenb" : "=r"(xtrlenb));
    asm volatile("csrr %0, th.xalenb"  : "=r"(xalenb));

    printf("Matrix ISA:        0x%lx\n", xmisa);
    printf("Tile length:       %lu bytes\n", xtlenb);
    printf("Tile row length:   %lu bytes\n", xtrlenb);
    printf("Accum length:      %lu bytes\n", xalenb);
}
```

### Example 4: Mixed RVV + RVM Program

```c
#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

// Vector preprocessing + matrix compute
// Quantize FP32 input to INT8 using RVV, then use RVM for matmul
void quantize_and_matmul(const float *input, size_t n,
                         const int8_t *weights, size_t w_stride,
                         int32_t *output, size_t o_stride,
                         float scale, size_t M, size_t K, size_t N) {
    int8_t quantized[1024];

    // Phase 1: RVV quantization (FP32 -> INT8)
    size_t remaining = n;
    const float *in = input;
    int8_t *out = quantized;
    while (remaining > 0) {
        size_t vl = __riscv_vsetvl_e32m4(remaining);
        vfloat32m4_t v = __riscv_vle32_v_f32m4(in, vl);
        v = __riscv_vfmul_vf_f32m4(v, scale, vl);
        vint32m4_t vi = __riscv_vfcvt_x_f_v_i32m4(v, vl);
        vint8m1_t vn = __riscv_vncvt_x_x_w_i8m1(
            __riscv_vncvt_x_x_w_i16m2(vi, vl), vl);
        __riscv_vse8_v_i8m1(out, vn, vl);
        in += vl;
        out += vl;
        remaining -= vl;
    }

    // Phase 2: RVM matrix multiply
    __builtin_riscv_th_msettilem(M);
    __builtin_riscv_th_msettilek(K);
    __builtin_riscv_th_msettilen(N);
    __builtin_riscv_th_mzero();

    __builtin_riscv_th_mlae8((void *)quantized, K);
    __builtin_riscv_th_mlbe8((void *)weights, w_stride);
    __builtin_riscv_th_mmacc_w_b();
    __builtin_riscv_th_msae32((void *)output, o_stride);

    __builtin_riscv_th_mrelease();
}
```

Compile to LLVM IR with both V and XTHeadMatrix enabled:
```bash
clang --target=riscv64 -march=rv64gcv_xtheadmatrix0p6 \
  -menable-experimental-extensions -O2 -emit-llvm -S mixed_kernel.c -o mixed_kernel.ll
```

## CSR Reference

XTHeadMatrix defines 13 CSRs, all prefixed with `th.`.

### Read/Write CSRs (addresses 0x802-0x80a)

| CSR Name | Address | Description |
|----------|---------|-------------|
| `th.xmcsr` | 0x802 | Matrix control/status register |
| `th.mtilem` | 0x803 | Tile M dimension (rows of A/C) |
| `th.mtilen` | 0x804 | Tile N dimension (columns of B/C) |
| `th.mtilek` | 0x805 | Tile K dimension (cols of A, rows of B) |
| `th.xmxrm` | 0x806 | Fixed-point rounding mode |
| `th.xmsat` | 0x807 | Saturation flag |
| `th.xmfflags` | 0x808 | FP exception flags |
| `th.xmfrm` | 0x809 | FP rounding mode |
| `th.xmsaten` | 0x80a | Saturation enable |

### Read-Only CSRs (addresses 0xcc0-0xcc3)

| CSR Name | Address | Description |
|----------|---------|-------------|
| `th.xmisa` | 0xcc0 | Matrix ISA capabilities |
| `th.xtlenb` | 0xcc1 | Tile length in bytes |
| `th.xtrlenb` | 0xcc2 | Tile row length in bytes |
| `th.xalenb` | 0xcc3 | Accumulator length in bytes |

### Accessing CSRs

CSRs are accessed via inline assembly since no builtin accessors exist:

```c
// Read a CSR
unsigned long val;
asm volatile("csrr %0, th.xmisa" : "=r"(val));

// Write a CSR
unsigned long rm = 0x1; // Round-to-nearest
asm volatile("csrw th.xmfrm, %0" :: "r"(rm));
```

## Assembly Instruction Quick Reference

All assembly mnemonics use the `th.` prefix. The test files in
`llvm/test/MC/RISCV/xtheadmatrix-{valid,invalid,csr}.s` provide
comprehensive usage examples.

| Category | Count | Mnemonics |
|----------|-------|-----------|
| Configuration | 7 | `mrelease`, `msettile{m,k,n}{i,}` |
| Load (strided) | 24 | `ml{a,b,c,at,bt,ct}e{8,16,32,64}` |
| Load (whole) | 4 | `mlme{8,16,32,64}` |
| Store (strided) | 24 | `ms{a,b,c,at,bt,ct}e{8,16,32,64}` |
| Store (whole) | 4 | `msme{8,16,32,64}` |
| FP Matmul | 13 | `mfmacc.{h,s,d,...}` |
| INT Matmul | 12 | `m{u,}macc{us,su,}.{w.b,d.h}`, `pmmacc...` |
| Bypass Matmul | 2 | `m{u,}macc.w.bp` |
| Zero | 4 | `mzero{,2r,4r,8r}` |
| Move | 9 | `mmov.mm`, `mmov{b,h,w,d}.{x.m,m.x}` |
| Duplicate | 4 | `mdup{b,h,w,d}.m.x` |
| Pack | 3 | `mpack{,hl,hh}` |
| Slide | 10 | `m{r,c}slide{down,up}{,.b,.h,.w,.d}` |
| Broadcast | 5 | `mrbca.mv.i`, `mcbca{b,h,w,d}.mv.i` |
| FP Conversions | 26 | `mfcvt{l,h}.{dst}.{src}` |
| Float-Int Conv | 12 | `m{u,s}fcvt{l,h}.{dst}.{src}` |
| N4Clip | 8 | `mn4clip{l,h}{,u}.w.{mm,mv.i}` |
| Packed Conv | 4 | `m{u,s}cvt{l,h}.b.p` |
| INT EW Arith | 22 | `m{add,sub,mul,...}.w.{mm,mv.i}` |
| FP EW Arith | 30 | `mf{add,sub,mul,max,min}.{h,s,d}.{mm,mv.i}` |
| **Total** | **227** | |

## Limitations and Notes

- **Experimental status**: The extension requires `+experimental-xtheadmatrix`
  and may change in future LLVM releases as the spec evolves.

- **No CodeGen/ISel patterns**: There are no SelectionDAG or GlobalISel
  patterns. Code generation from plain C arithmetic (e.g., matrix loops)
  will not automatically use matrix instructions. Users must use builtins
  or inline assembly.

- **Low-level builtins only**: The current implementation provides low-level
  instruction-mapped builtins where matrix registers are implicit. The spec
  defines a higher-level API with C++ matrix types (`mint32_t`,
  `mfloat16_t`, etc.) and overloaded functions -- this is not yet
  implemented.

- **Known spec errata**:
  - The matmul instruction group uses `uop=10` (binary), but the spec text
    in one place incorrectly states `uop=01`. The encoding in the spec's
    tables is correct.
  - The `mfmin.h`/`mfmin.s` labels are swapped in one table of the spec.

- **No runtime library**: There is no `<thead_matrix.h>` header or runtime
  support library. All operations must use `__builtin_riscv_th_*` builtins
  or inline assembly.

- **Future work**: Higher-level matrix types, CodeGen integration, and a
  `<thead_matrix.h>` compatibility header are planned.
