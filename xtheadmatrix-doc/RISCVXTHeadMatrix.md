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
249 Clang builtins (227 original + 22 `mundef` constructors) are
implemented with full ISel/CodeGen support. All intrinsics and builtins
accept register index parameters that specify which matrix register
(0-7) to use, enabling flexible register selection and multi-accumulator
kernels. 121 builtins have typed signatures using native `__rvm_*_t`
types, enabling Sema-level type checking. Compile-time Sema validation
enforces register type constraints (tile vs. accumulator) per the RVM
0.6 spec. A `<thead_matrix.h>` header provides over 400
higher-level API functions and macros with C matrix types (`mint8_t`,
`mint32_t`, `mfloat16_t`, etc.) backed by native built-in types. 22 native Clang
built-in types (`__rvm_int8_t` through `__rvm_float64x2_t`) provide
first-class type identity, mangling, and debug info support. Programs
can use either the high-level API, the low-level Clang builtins, or
inline assembly and compile directly to native assembly or object code.

## Building and Installing the Toolchain

### Prerequisites

- CMake >= 3.20
- A C++ compiler with C++17 support (GCC >= 8 or Clang >= 10)
- Python 3
- Ninja (recommended)

### Option A: Minimal Development Build

This builds only the tools needed for development and testing of
XTHeadMatrix. It does **not** produce a linker or runtime libraries,
so it cannot link executables on its own.

```bash
cmake -S llvm -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_TARGETS_TO_BUILD=RISCV \
  -DLLVM_ENABLE_PROJECTS="clang"

cmake --build build -- -j$(nproc) clang llvm-mc llvm-objdump
```

What this builds:
- `clang` -- compiler with integrated assembler (can produce `.o` / `.s`)
- `llvm-mc` -- standalone assembler (for MC-layer testing)
- `llvm-objdump` -- disassembler (for encoding verification)

What this does **not** build: linker (`lld`), archiver (`llvm-ar`),
runtime libraries (`compiler-rt`, `libc`, `libcxx`), or any other
LLVM tools (`opt`, `llc`, `llvm-nm`, `llvm-readelf`, etc.).

### Option B: Full Toolchain Build and Install

This builds the complete RISC-V cross-compilation toolchain -- compiler,
linker, assembler, binutils, and runtime libraries -- and installs it to
a prefix directory ready for use.

#### 1. Configure

```bash
cmake -S llvm -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_TARGETS_TO_BUILD=RISCV \
  -DLLVM_ENABLE_PROJECTS="clang;lld;llvm" \
  -DLLVM_ENABLE_RUNTIMES="compiler-rt" \
  -DCMAKE_INSTALL_PREFIX=/opt/riscv-llvm \
  -DLLVM_INSTALL_TOOLCHAIN_ONLY=OFF
```

Key options explained:

| Option | Purpose |
|--------|---------|
| `LLVM_TARGETS_TO_BUILD=RISCV` | Build only the RISC-V backend (faster build) |
| `LLVM_ENABLE_PROJECTS="clang;lld;llvm"` | Compiler + linker + all LLVM tools |
| `LLVM_ENABLE_RUNTIMES="compiler-rt"` | Builtins/sanitizer runtime for RISC-V |
| `CMAKE_INSTALL_PREFIX` | Where `make install` places binaries, libs, headers |

To also build the C++ standard library for RISC-V bare-metal or Linux
targets, add `"libcxx;libcxxabi;libunwind"` to `LLVM_ENABLE_RUNTIMES`.

#### 2. Build

```bash
cmake --build build -- -j$(nproc)
```

Or with Ninja directly:

```bash
cd build && ninja -j$(nproc)
```

This builds everything: `clang`, `lld` (linker), `llvm-ar` (archiver),
`llvm-nm`, `llvm-objcopy`, `llvm-objdump`, `llvm-readelf`,
`llvm-strip`, `llvm-mc`, `opt`, `llc`, and runtime libraries.

#### 3. Install

```bash
cmake --install build
```

Or equivalently:

```bash
cd build && ninja install
```

This installs into the `CMAKE_INSTALL_PREFIX` directory (e.g.
`/opt/riscv-llvm/`). The installed layout:

```
/opt/riscv-llvm/
├── bin/          # clang, lld, llvm-ar, llvm-mc, llvm-objdump, ...
├── include/      # Clang/LLVM headers
├── lib/          # Libraries, compiler-rt builtins, clang resource dir
└── share/        # Documentation, CMake modules
```

#### 4. Add to PATH

```bash
export PATH="/opt/riscv-llvm/bin:$PATH"
```

### Verify Extension Availability

```bash
clang --print-supported-extensions 2>&1 | grep xtheadmatrix
```

Expected output:

```
xtheadmatrix             0.6       'XTHeadMatrix' (T-Head Matrix Extension)
```

### What Each Tool Does

| Tool | Role |
|------|------|
| `clang` | C/C++ compiler with integrated assembler (source → `.o` ELF) |
| `lld` / `ld.lld` | Linker (`.o` files → linked ELF executable/shared lib) |
| `llvm-ar` | Archiver (create/manage `.a` static libraries) |
| `llvm-nm` | List symbols in object files |
| `llvm-objdump` | Disassembler (inspect machine code in ELF files) |
| `llvm-objcopy` | Copy/transform object files (strip, convert formats) |
| `llvm-readelf` | Display ELF file headers, sections, symbols |
| `llvm-strip` | Strip symbols from binaries |
| `llvm-mc` | Standalone assembler (`.s` → `.o`, useful for testing) |
| `opt` | LLVM IR optimizer (for pass development/debugging) |
| `llc` | LLVM IR → machine code compiler (for backend debugging) |

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

### Compiling C/C++ Programs

Programs using `#include <thead_matrix.h>` or the low-level
`__builtin_riscv_th_*` builtins compile directly to native code:

```bash
# Compile to assembly
clang --target=riscv64 -march=rv64gc_xtheadmatrix0p6 \
  -menable-experimental-extensions -S matrix_kernel.c -o matrix_kernel.s

# Compile to object code
clang --target=riscv64 -march=rv64gc_xtheadmatrix0p6 \
  -menable-experimental-extensions -c matrix_kernel.c -o matrix_kernel.o

# Generate LLVM IR
clang --target=riscv64 -march=rv64gc_xtheadmatrix0p6 \
  -menable-experimental-extensions -emit-llvm -S matrix_kernel.c -o matrix_kernel.ll
```

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
    "th.mzero acc0\n\t"
    "th.mlae32 tr0, (%0), %1\n\t"
    "th.mlbe32 tr1, (%2), %3\n\t"
    "th.mfmacc.s acc0, tr1, tr0\n\t"
    "th.msce32 acc0, (%4), %5"
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
or `-L` flags.

## Higher-Level Intrinsic API

The `<thead_matrix.h>` header provides a higher-level C API on top of
the low-level `__builtin_riscv_th_*` builtins. It defines matrix type
aliases (backed by native `__rvm_*_t` built-in types) and approximately
over 400 wrapper functions and macros that carry dimension parameters, making matrix code
more readable and less error-prone. Since Phase C, the header uses typed
builtins internally, providing Sema-level type checking.

### Including the Header

```c
#include <thead_matrix.h>
```

The header is available when compiling with `-march=...xtheadmatrix0p6`
and `-menable-experimental-extensions`.

### Matrix Types

The header defines 22 matrix types backed by native `__rvm_*_t` built-in types:

| Type | Backed by | Description |
|------|-----------|-------------|
| `mint8_t` | `__rvm_int8_t` | Signed 8-bit integer matrix |
| `muint8_t` | `__rvm_uint8_t` | Unsigned 8-bit integer matrix |
| `mint16_t` | `__rvm_int16_t` | Signed 16-bit integer matrix |
| `muint16_t` | `__rvm_uint16_t` | Unsigned 16-bit integer matrix |
| `mint32_t` | `__rvm_int32_t` | Signed 32-bit integer matrix |
| `muint32_t` | `__rvm_uint32_t` | Unsigned 32-bit integer matrix |
| `mint64_t` | `__rvm_int64_t` | Signed 64-bit integer matrix |
| `muint64_t` | `__rvm_uint64_t` | Unsigned 64-bit integer matrix |
| `mfloat16_t` | `__rvm_float16_t` | IEEE FP16 matrix |
| `mfloat32_t` | `__rvm_float32_t` | FP32 matrix |
| `mfloat64_t` | `__rvm_float64_t` | FP64 matrix |
| `mint8x2_t` | `__rvm_int8x2_t` | 2-register group, INT8 |
| `mint16x2_t` | `__rvm_int16x2_t` | 2-register group, INT16 |
| `mint32x2_t` | `__rvm_int32x2_t` | 2-register group, INT32 |
| `mint64x2_t` | `__rvm_int64x2_t` | 2-register group, INT64 |
| `muint8x2_t` | `__rvm_uint8x2_t` | 2-register group, UINT8 |
| `muint16x2_t` | `__rvm_uint16x2_t` | 2-register group, UINT16 |
| `muint32x2_t` | `__rvm_uint32x2_t` | 2-register group, UINT32 |
| `muint64x2_t` | `__rvm_uint64x2_t` | 2-register group, UINT64 |
| `mfloat16x2_t` | `__rvm_float16x2_t` | 2-register group, FP16 |
| `mfloat32x2_t` | `__rvm_float32x2_t` | 2-register group, FP32 |
| `mfloat64x2_t` | `__rvm_float64x2_t` | 2-register group, FP64 |

Two dimension types are also defined:

| Type | Description |
|------|-------------|
| `mrow_t` | Row dimension (M or K) |
| `mcol_t` | Column dimension (N) |

### Example: INT8 GEMM Using the Higher-Level API

```c
#include <thead_matrix.h>

void int8_gemm(const int8_t *A, long a_stride,
               const int8_t *B, long b_stride,
               int32_t *C, long c_stride,
               mrow_t M, mrow_t K, mcol_t N) {
    mint8_t a = __riscv_th_mld_a_i8(A, a_stride, M, K);
    mint8_t b = __riscv_th_mld_b_i8(B, b_stride, K, N);
    mint32_t c = __riscv_th_mzero_i32();
    c = __riscv_th_mmaqa_ss_w_b(c, a, b, M, K, N);
    __riscv_th_mst_c_i32(C, c_stride, c, M, N);
    __riscv_th_mrelease();
}
```

Compare this with the equivalent low-level builtin version in
[Example 1](#example-1-int8-gemm-kernel) below -- the higher-level API
carries dimension parameters and returns typed matrix values, making the
data flow explicit.

### API Categories

The 414 functions and macros in `<thead_matrix.h>` are organized into
these categories:

| Category | Examples | Description |
|----------|----------|-------------|
| Configuration | `__riscv_th_msettilem`, `__riscv_th_mrelease` | Set tile dimensions, release matrix unit |
| CSR access | `__riscv_th_mread_csr`, `__riscv_th_mwrite_csr` | Read/write matrix CSRs |
| Load | `__riscv_th_mld_a_i8`, `__riscv_th_mld_b_f32` | Load tiles from memory with type and role |
| Store | `__riscv_th_mst_c_i32`, `__riscv_th_mst_a_f16` | Store tiles to memory |
| Matrix multiply | `__riscv_th_mmaqa_ss_w_b`, `__riscv_th_mfmacc_s` | Matmul-accumulate (integer and FP) |
| EW arithmetic | `__riscv_th_madd_w`, `__riscv_th_mfmul_s` | Element-wise add, sub, mul, min, max, shift |
| Conversions | `__riscv_th_mfcvt_s_h`, `__riscv_th_msfcvt_h_b` | FP and integer format conversions |
| Data movement | `__riscv_th_mzero_i32`, `__riscv_th_mmov` | Zero, move, pack, slide, broadcast |

### Immediate Arguments (ImmArg)

Several functions require compile-time constant arguments (e.g., the
immediate tile-dimension setters `__riscv_th_msettilemi`, slide functions,
broadcast functions, and MVI variants). These are implemented as macros
in `<thead_matrix.h>` and will produce a compilation error if passed a
non-constant value.

### Native Built-in Types (Phase B)

22 first-class Clang built-in types are now available for the XTHeadMatrix
extension. These types are named `__rvm_int8_t`, `__rvm_uint8_t`,
`__rvm_int16_t`, ..., through `__rvm_float64x2_t`, mirroring the 22
matrix element types listed in the [Matrix Types](#matrix-types) table
above.

The types are registered via `RISCVMatrixTypes.def`, following the same
pattern used by RVV's `__rvv_int32m1_t` types in `RISCVVTypes.def`. They
are gated on the `xtheadmatrix` target feature -- using them without the
feature enabled produces a diagnostic.

The API types (`mint32_t`, `mfloat16_t`, etc.) defined in
`<thead_matrix.h>` are now aliases to these built-in types (since Phase C).
The native built-in types provide:

- Proper type identity in the Clang AST
- Itanium name mangling (enabling C++ overloading and templates)
- AST serialization and deserialization (precompiled headers, modules)
- Debug info (DWARF) generation
- libclang / `CXType` support for tooling

Example:

```c
// Available with -march=rv64gc_xtheadmatrix0p6 -menable-experimental-extensions
__rvm_int32_t my_matrix;
__rvm_float32_t result = my_func(my_matrix);
```

## Programming Model

### Overview

Matrix register state is implicit -- the 8 matrix registers (`tr0`-`tr3`,
`acc0`-`acc3`) are mapped to opaque `__rvm_*_t` built-in types at the
Clang level. These types provide compile-time type checking but map to
`PoisonValue` tokens in LLVM IR. Register selection is controlled through
register index parameters (0-7) passed to each intrinsic/builtin, where
0-3 select tile registers (`tr0`-`tr3`) and 4-7 select accumulator
registers (`acc0`-`acc3`). The `<thead_matrix.h>` API passes appropriate
defaults (e.g., loads to tile registers, matmul destinations to
accumulator registers), but users of the low-level builtins can specify
any valid register index.

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

### Low-Level Builtins vs. Higher-Level API

Two programming interfaces are available. **The higher-level API is
recommended for most users.**

1. **Higher-level API** (`<thead_matrix.h>`, **recommended**): Provides
   over 400 functions/macros with C matrix types (`mint8_t`,
   `mfloat16_t`, etc.) and dimension parameters (`mrow_t`, `mcol_t`).
   Functions like `__riscv_th_mld_a_i8`, `__riscv_th_mmaqa_ss_w_b`, and
   `__riscv_th_mst_c_i32` carry dimension information and return typed
   values, making matrix code readable and type-safe. See the
   [Higher-Level Intrinsic API](#higher-level-intrinsic-api) section and
   all [Code Examples](#code-examples).

2. **Low-level builtins** (`__builtin_riscv_th_*`): Each builtin maps
   1:1 to a single assembly instruction. 121 builtins accept and return
   native `__rvm_*_t` matrix types, enabling compile-time type checking.
   106 builtins for stores, configuration, zero, misc, and FP8/BF16/TF32
   variants retain void signatures. Useful when precise instruction
   control is needed. See the [Builtins Reference](#builtins-reference)
   section below.

## Builtins Reference

All builtins use the `__builtin_riscv_th_` prefix and are available when
the `xtheadmatrix` target feature is enabled.

**Register index parameters**: All non-configuration builtins accept 1-3
`unsigned int` register index parameters as their first arguments,
specifying which matrix register (0-7) to use for each operand. These
are compile-time constants (ImmArg). Sema validation enforces register
type constraints: load-A/B require tile registers (0-3), load-C
requires accumulator registers (4-7), matmul destinations must be
accumulators, and element-wise operands must all be accumulators.

**Typed builtins (Phase C)**: 121 builtins accept and return native
`__rvm_*_t` matrix types (e.g.,
`__rvm_int32_t __builtin_riscv_th_mlae32(unsigned int, void*, size_t)`).
This enables Sema-level type checking -- passing a `__rvm_float32_t` where
`__rvm_int32_t` is expected produces a compile-time error. The LLVM IR
intrinsics remain void; the CGBuiltin handler bridges the gap by filtering
matrix-typed arguments and returning `PoisonValue` tokens. 22 `mundef`
builtins (`__builtin_riscv_th_mundef_i8` through
`__builtin_riscv_th_mundef_f64x2`) create undefined matrix values for
initializing variables.

**Untyped builtins**: 106 builtins retain void signatures -- stores,
configuration, zero, misc, and FP8/BF16/TF32 variants that lack native
matrix types.

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

All load builtins load data from memory into matrix registers and
return a typed matrix value (Phase C).

#### Element-Stride Loads (12 builtins)

Prototype: `__rvm_int<W>_t(unsigned int md_idx, void *base, size_t stride)` where W matches the EEW

| Builtin | Assembly | Description |
|---------|----------|-------------|
| `__builtin_riscv_th_mlae{8,16,32,64}` | `th.mlae{8,16,32,64} md, (rs1), rs2` | Load A matrix, element stride |
| `__builtin_riscv_th_mlbe{8,16,32,64}` | `th.mlbe{8,16,32,64} md, (rs1), rs2` | Load B matrix, element stride |
| `__builtin_riscv_th_mlce{8,16,32,64}` | `th.mlce{8,16,32,64} md, (rs1), rs2` | Load C matrix, element stride |

#### Tile-Stride (Transposed) Loads (12 builtins)

Prototype: `__rvm_int<W>_t(unsigned int md_idx, void *base, size_t stride)` where W matches the EEW

| Builtin | Assembly | Description |
|---------|----------|-------------|
| `__builtin_riscv_th_mlate{8,16,32,64}` | `th.mlate{8,16,32,64} md, (rs1), rs2` | Load A transposed, tile stride |
| `__builtin_riscv_th_mlbte{8,16,32,64}` | `th.mlbte{8,16,32,64} md, (rs1), rs2` | Load B transposed, tile stride |
| `__builtin_riscv_th_mlcte{8,16,32,64}` | `th.mlcte{8,16,32,64} md, (rs1), rs2` | Load C transposed, tile stride |

#### Whole-Register Loads (4 builtins)

Prototype: `__rvm_int<W>_t(unsigned int md_idx, void *base)` where W matches the EEW

| Builtin | Assembly | Description |
|---------|----------|-------------|
| `__builtin_riscv_th_mlme{8,16,32,64}` | `th.mlme{8,16,32,64} md, (rs1)` | Load whole matrix register |

### Store Instructions (28 builtins)

All store builtins store matrix register data to memory.

#### Element-Stride Stores (12 builtins)

Prototype: `void(unsigned int ms3_idx, void *base, size_t stride)`

| Builtin | Assembly | Description |
|---------|----------|-------------|
| `__builtin_riscv_th_msae{8,16,32,64}` | `th.msae{8,16,32,64} ms3, (rs1), rs2` | Store A matrix, element stride |
| `__builtin_riscv_th_msbe{8,16,32,64}` | `th.msbe{8,16,32,64} ms3, (rs1), rs2` | Store B matrix, element stride |
| `__builtin_riscv_th_msce{8,16,32,64}` | `th.msce{8,16,32,64} ms3, (rs1), rs2` | Store C matrix, element stride |

#### Tile-Stride (Transposed) Stores (12 builtins)

Prototype: `void(unsigned int ms3_idx, void *base, size_t stride)`

| Builtin | Assembly | Description |
|---------|----------|-------------|
| `__builtin_riscv_th_msate{8,16,32,64}` | `th.msate{8,16,32,64} ms3, (rs1), rs2` | Store A transposed, tile stride |
| `__builtin_riscv_th_msbte{8,16,32,64}` | `th.msbte{8,16,32,64} ms3, (rs1), rs2` | Store B transposed, tile stride |
| `__builtin_riscv_th_mscte{8,16,32,64}` | `th.mscte{8,16,32,64} ms3, (rs1), rs2` | Store C transposed, tile stride |

#### Whole-Register Stores (4 builtins)

Prototype: `void(unsigned int ms3_idx, void *base)`

| Builtin | Assembly | Description |
|---------|----------|-------------|
| `__builtin_riscv_th_msme{8,16,32,64}` | `th.msme{8,16,32,64} ms3, (rs1)` | Store whole matrix register |

### Matrix Multiply-Accumulate (27 builtins)

Matrix multiply builtins with native types (h, s, d, s_h, d_s, and all
integer variants) accept and return `__rvm_*_t` typed arguments (Phase C).
FP8/BF16/TF32 variants remain `void()` since no native matrix types exist
for these formats.

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

Typed conversions (FP16&harr;FP32, FP32&harr;FP64) accept and return
`__rvm_float*_t` types. FP8/BF16/TF32 conversions have `void()`
prototypes since no native matrix types exist for these formats. The
`l` (low) and `h` (high) suffixes indicate which half of a
widening/narrowing pair is operated on.

#### FP8 &harr; FP16 (8 builtins, void())

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

#### FP16/BF16 &harr; FP32 (8 builtins: FP16 typed, BF16 void())

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

#### FP32 &harr; FP8 (4 builtins, void())

| Builtin | Assembly | Description |
|---------|----------|-------------|
| `__builtin_riscv_th_mfcvtl_e4_s` | `th.mfcvtl.e4.s` | FP32 &rarr; E4M3, low half |
| `__builtin_riscv_th_mfcvth_e4_s` | `th.mfcvth.e4.s` | FP32 &rarr; E4M3, high half |
| `__builtin_riscv_th_mfcvtl_e5_s` | `th.mfcvtl.e5.s` | FP32 &rarr; E5M2, low half |
| `__builtin_riscv_th_mfcvth_e5_s` | `th.mfcvth.e5.s` | FP32 &rarr; E5M2, high half |

#### FP32 &harr; FP64 (4 builtins, typed)

| Builtin | Assembly | Description |
|---------|----------|-------------|
| `__builtin_riscv_th_mfcvtl_d_s` | `th.mfcvtl.d.s` | FP32 &rarr; FP64, low half |
| `__builtin_riscv_th_mfcvth_d_s` | `th.mfcvth.d.s` | FP32 &rarr; FP64, high half |
| `__builtin_riscv_th_mfcvtl_s_d` | `th.mfcvtl.s.d` | FP64 &rarr; FP32, low half |
| `__builtin_riscv_th_mfcvth_s_d` | `th.mfcvth.s.d` | FP64 &rarr; FP32, high half |

#### TF32 &harr; FP32 (2 builtins, void())

| Builtin | Assembly | Description |
|---------|----------|-------------|
| `__builtin_riscv_th_mfcvt_s_tf32` | `th.mfcvt.s.tf32` | TF32 &rarr; FP32 |
| `__builtin_riscv_th_mfcvt_tf32_s` | `th.mfcvt.tf32.s` | FP32 &rarr; TF32 |

### Float-Integer Conversions (12 builtins, all typed)

All accept and return `__rvm_*_t` types (e.g.,
`__rvm_float16_t(__rvm_uint8_t)` for `mufcvtl_h_b`).

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

### Integer Element-Wise Arithmetic (22 builtins, all typed)

Each operation has two variants:
- `.w.mm` (matrix-matrix): Prototype `__rvm_int32_t(__rvm_int32_t, __rvm_int32_t, __rvm_int32_t)`
- `.w.mv.i` (matrix-vector, immediate index): Prototype `__rvm_int32_t(__rvm_int32_t, __rvm_int32_t, unsigned int)`

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

### FP Element-Wise Arithmetic (30 builtins, all typed)

Each operation has three precision levels (`.h` FP16, `.s` FP32, `.d` FP64)
and two variants (`.mm` matrix-matrix, `.mv.i` matrix-vector with immediate).

- `.h.mm` Prototype: `__rvm_float16_t(__rvm_float16_t, __rvm_float16_t, __rvm_float16_t)`
- `.s.mm` Prototype: `__rvm_float32_t(__rvm_float32_t, __rvm_float32_t, __rvm_float32_t)`
- `.d.mm` Prototype: `__rvm_float64_t(__rvm_float64_t, __rvm_float64_t, __rvm_float64_t)`
- `.{h,s,d}.mv.i` Prototype: `__rvm_float{16,32,64}_t(__rvm_float{16,32,64}_t, __rvm_float{16,32,64}_t, unsigned int)`

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

All examples use the `<thead_matrix.h>` higher-level API, which is the
recommended programming interface. The API wraps the low-level
`__builtin_riscv_th_*` builtins with typed functions, dimension parameters,
and role-specific load/store names for a natural programming experience.

### Example 1: INT8 GEMM Kernel

```c
#include <thead_matrix.h>

// INT8 GEMM: C[M,N] = A[M,K] * B[K,N]
void int8_gemm(const int8_t *A, long a_stride,
               const int8_t *B, long b_stride,
               int32_t *C, long c_stride,
               mrow_t M, mrow_t K, mcol_t N) {
    // Load input matrices (tile config is automatic)
    mint8_t a = __riscv_th_mld_a_i8(A, a_stride, M, K);
    mint8_t b = __riscv_th_mld_b_i8(B, b_stride, K, N);

    // Zero accumulator and compute: c = a * b
    mint32_t c = __riscv_th_mzero_i32();
    c = __riscv_th_mmaqa_ss_w_b(c, a, b, M, K, N);

    // Store result and release
    __riscv_th_mst_c_i32(C, c_stride, c, M, N);
    __riscv_th_mrelease();
}
```

### Example 2: FP32 GEMM with Accumulation

```c
#include <thead_matrix.h>

// FP32 GEMM: C[M,N] += A[M,K] * B[K,N]
void fp32_gemm_acc(const float *A, long a_stride,
                   const float *B, long b_stride,
                   float *C, long c_stride,
                   mrow_t M, mrow_t K, mcol_t N) {
    // Load existing accumulator from memory
    mfloat32_t c = __riscv_th_mld_c_f32(C, c_stride, M, N);

    // Load input tiles
    mfloat32_t a = __riscv_th_mld_a_f32(A, a_stride, M, K);
    mfloat32_t b = __riscv_th_mld_b_f32(B, b_stride, K, N);

    // Accumulate: c += a * b
    c = __riscv_th_fmmacc_s(c, a, b, M, K, N);

    // Store and release
    __riscv_th_mst_c_f32(C, c_stride, c, M, N);
    __riscv_th_mrelease();
}
```

### Example 3: FP16→FP32 Widening Matmul

```c
#include <thead_matrix.h>

// Widening matmul: FP16 inputs, FP32 output
void fp16_to_fp32_gemm(const void *A, long a_stride,
                       const void *B, long b_stride,
                       float *C, long c_stride,
                       mrow_t M, mrow_t K, mcol_t N) {
    mfloat16_t a = __riscv_th_mld_a_f16(A, a_stride, M, K);
    mfloat16_t b = __riscv_th_mld_b_f16(B, b_stride, K, N);
    mfloat32_t c = __riscv_th_mzero_f32();

    // Widening matmul: FP16 * FP16 → FP32
    c = __riscv_th_fwmmacc_s_h(c, a, b, M, K, N);

    __riscv_th_mst_c_f32(C, c_stride, c, M, N);
    __riscv_th_mrelease();
}
```

### Example 4: Element-Wise Operations and Type Conversions

```c
#include <thead_matrix.h>

// Post-matmul pipeline: int32 result → add bias → shift → clip to int8
void quantize_output(int32_t *result, long r_stride,
                     int32_t *bias, long b_stride,
                     void *output, long o_stride,
                     mrow_t M, mcol_t N) {
    // Load matmul result and bias
    mint32_t r = __riscv_th_mld_c_i32(result, r_stride, M, N);
    mint32_t b = __riscv_th_mld_c_i32(bias, b_stride, M, N);

    // Element-wise add: r = r + bias
    r = __riscv_th_madd_w_mm(r, r, b);

    // Element-wise shift right: r >>= shift_amounts
    mint32_t shift = __riscv_th_mundefined_i32();
    r = __riscv_th_msra_w_mm(r, r, shift);

    // N4Clip to signed int8
    mint8_t clipped = __riscv_th_mn4clipl_w_mm(r, shift);

    // Store int8 output
    __riscv_th_mst_a_i8(output, o_stride, clipped, M, N);
    __riscv_th_mrelease();
}
```

### Example 5: CSR Access and Hardware Query

```c
#include <thead_matrix.h>
#include <stdio.h>

void query_matrix_hw(void) {
    // Use API functions to read hardware capabilities
    unsigned long tlen = __riscv_th_xmlenb();   // tile register size in bytes
    unsigned long rlen = __riscv_th_xrlenb();   // tile row length in bytes
    unsigned long isa  = __riscv_th_xmsize();   // matrix ISA capabilities

    printf("Tile length:     %lu bytes\n", tlen);
    printf("Tile row length: %lu bytes\n", rlen);
    printf("Max rows:        %lu\n", tlen / rlen);
    printf("Matrix ISA:      0x%lx\n", isa);

    // Set FP rounding mode via CSR macro
    __riscv_th_mwrite_csr(RVM_CSR_XMFRM, 0);  // round-to-nearest

    // Enable saturation for integer matmul
    __riscv_th_mwrite_csr(RVM_CSR_XMSATEN, 1);
}
```

### Example 6: Mixed RVV + RVM Program

```c
#include <riscv_vector.h>
#include <thead_matrix.h>

// Vector preprocessing + matrix compute
// Quantize FP32 input to INT8 using RVV, then use RVM for matmul
void quantize_and_matmul(const float *input, size_t n,
                         const int8_t *weights, long w_stride,
                         int32_t *output, long o_stride,
                         float scale, mrow_t M, mrow_t K, mcol_t N) {
    int8_t quantized[1024];

    // Phase 1: RVV quantization (FP32 → INT8)
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

    // Phase 2: RVM matrix multiply (using high-level API)
    mint8_t a = __riscv_th_mld_a_i8(quantized, K, M, K);
    mint8_t b = __riscv_th_mld_b_i8(weights, w_stride, K, N);
    mint32_t c = __riscv_th_mzero_i32();
    c = __riscv_th_mmaqa_ss_w_b(c, a, b, M, K, N);
    __riscv_th_mst_c_i32(output, o_stride, c, M, N);
    __riscv_th_mrelease();
}
```

Compile with both V and XTHeadMatrix enabled:
```bash
clang --target=riscv64 -march=rv64gcv_xtheadmatrix0p6 \
  -menable-experimental-extensions -O2 -c mixed_kernel.c -o mixed_kernel.o
```

### Example 7: Multi-Accumulator GEMM

Using register index parameters, different matmul operations can target
different accumulator registers simultaneously:

```c
#include <thead_matrix.h>

// Compute two independent GEMMs sharing the same A matrix
// C1[M,N] = A[M,K] * B1[K,N]
// C2[M,N] = A[M,K] * B2[K,N]
void dual_gemm(const int8_t *A, long a_stride,
               const int8_t *B1, long b1_stride,
               const int8_t *B2, long b2_stride,
               int32_t *C1, long c1_stride,
               int32_t *C2, long c2_stride,
               mrow_t M, mrow_t K, mcol_t N) {
    // Configure dimensions
    __builtin_riscv_th_msettilem(M);
    __builtin_riscv_th_msettilek(K);
    __builtin_riscv_th_msettilen(N);

    // Load shared A matrix into tr0
    __builtin_riscv_th_mlae8(__RVM_TR0, (void *)A, a_stride);

    // Zero acc0, load B1 into tr1, INT8 matmul into acc0
    __builtin_riscv_th_mzero(__RVM_ACC0);
    __builtin_riscv_th_mlbe8(__RVM_TR1, (void *)B1, b1_stride);
    __rvm_int32_t i32 = __builtin_riscv_th_mundef_i32();
    __rvm_int8_t i8 = __builtin_riscv_th_mundef_i8();
    (void)__builtin_riscv_th_mmacc_w_b(__RVM_ACC0, __RVM_TR1, __RVM_TR0,
                                        i32, i8, i8);

    // Zero acc1, load B2 into tr1, INT8 matmul into acc1
    __builtin_riscv_th_mzero(__RVM_ACC1);
    __builtin_riscv_th_mlbe8(__RVM_TR1, (void *)B2, b2_stride);
    (void)__builtin_riscv_th_mmacc_w_b(__RVM_ACC1, __RVM_TR1, __RVM_TR0,
                                        i32, i8, i8);

    // Store both results
    __builtin_riscv_th_msce32(__RVM_ACC0, (void *)C1, c1_stride);
    __builtin_riscv_th_msce32(__RVM_ACC1, (void *)C2, c2_stride);
    __builtin_riscv_th_mrelease();
}
```

This pattern was not possible with the previous fixed register assignment
where all matmul operations were hardcoded to use acc0.

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

CSRs are accessed via the `<thead_matrix.h>` API macros or inline assembly:

```c
#include <thead_matrix.h>

// Using API macros (recommended)
unsigned long isa = __riscv_th_mread_csr(RVM_CSR_XMISA);
__riscv_th_mwrite_csr(RVM_CSR_XMFRM, 0x1);  // round-to-nearest

// Using convenience functions
unsigned long tlen = __riscv_th_xmlenb();
unsigned long rlen = __riscv_th_xrlenb();

// Using inline assembly (also works)
unsigned long val;
asm volatile("csrr %0, th.xmisa" : "=r"(val));
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

- **Flexible register selection**: All builtins accept register index
  parameters (0-7) specifying which matrix register to use. The
  `<thead_matrix.h>` API passes default register assignments matching the
  standard programming model (load A→tr0, load B→tr1, matmul dest→acc0,
  etc.), but users of the low-level builtins can select any valid register.
  This enables multi-accumulator kernels where different matmul operations
  target different accumulator registers (e.g., acc0 and acc1 in parallel).
  Code generation from plain C arithmetic (e.g., matrix loops) will not
  automatically use matrix instructions -- users must use builtins or
  inline assembly.

- **Sema register constraint validation**: The compiler enforces RVM 0.6
  register type constraints at compile time:
  - **Load A/B** (`mla*`/`mlb*`): `md` must be a tile register (0-3)
  - **Load C** (`mlc*`): `md` must be an accumulator register (4-7)
  - **Store A/B** (`msa*`/`msb*`): `ms3` must be a tile register (0-3)
  - **Store C** (`msc*`): `ms3` must be an accumulator register (4-7)
  - **Matmul**: `md` must be accumulator (4-7), `ms1`/`ms2` must be tile (0-3)
  - **Element-wise** (arithmetic, conversions, N4clip): all `md/ms1/ms2` must be accumulator (4-7)
  - **Load/Store M** (`mlm*`/`msm*`), **zero**, **move**, **duplicate**: any register (0-7)
  - **Slides**, **broadcasts**, **pack**: any register (0-7)
  Invalid register usage produces a compile-time error.

- **Register index constants**: The `<thead_matrix.h>` header defines
  named constants for register indices:
  ```c
  #define __RVM_TR0  0   // Tile register 0
  #define __RVM_TR1  1   // Tile register 1
  #define __RVM_TR2  2   // Tile register 2
  #define __RVM_TR3  3   // Tile register 3
  #define __RVM_ACC0 4   // Accumulator register 0
  #define __RVM_ACC1 5   // Accumulator register 1
  #define __RVM_ACC2 6   // Accumulator register 2
  #define __RVM_ACC3 7   // Accumulator register 3
  ```

- **Two API levels plus native built-in types**: Both low-level
  instruction-mapped builtins (`__builtin_riscv_th_*`) and the
  higher-level `<thead_matrix.h>` API with C matrix types (`mint32_t`,
  `mfloat16_t`, etc.) are available. 121 low-level builtins accept and
  return native `__rvm_*_t` matrix types, enabling compile-time type
  checking at both API levels. The higher-level API wraps the low-level
  builtins and does not add runtime overhead. Additionally, 22 native
  Clang built-in types (`__rvm_int8_t` through `__rvm_float64x2_t`)
  provide first-class type identity, mangling, debug info, and tooling
  support.

- **Known spec errata**:
  - The matmul instruction group uses `uop=10` (binary), but the spec
    encoding table incorrectly shows `uop=01` (a typo; `uop=01` is
    Load/Store). The format description text correctly states `uop=10`.
  - The `mfmin.h`/`mfmin.s` labels are swapped in the encoding table
    (encodings are correct, names are wrong).
  - `pmmaaccus.w.b` has an extra 'a' (typo for `pmmaccus.w.b`).

- **Header-only, no runtime library**: The `<thead_matrix.h>` header is a
  header-only wrapper around the compiler builtins. There is no separate
  runtime support library -- all matrix operations compile directly to
  inline instructions.

- **Test coverage**: 16 test files (12 Clang + 4 LLVM) covering:
  - Assembly encoding: all 227 instructions validated (`xtheadmatrix-valid.s`)
  - Invalid encoding: error diagnostics verified (`xtheadmatrix-invalid.s`)
  - CSR names: all 13 CSRs tested (`xtheadmatrix-csr.s`)
  - ISel patterns: all 227 intrinsic-to-instruction mappings (`xtheadmatrix-isel.ll`)
  - Low-level builtin codegen: end-to-end C → assembly (`xtheadmatrix-codegen.c`)
  - High-level API: 26+ API usage patterns (`thead-matrix-api-patterns.c`)
  - Corner cases: 20 complex pipeline tests (`thead-matrix-corner-cases.c`)
  - Comprehensive codegen: all instruction categories (`thead-matrix-comprehensive-codegen.c`)
  - Sema validation: register constraint error detection (`riscv-xtheadmatrix-reg-constraints.c`)
  - Built-in types: all 22 types compile and mangle correctly (`thead-matrix-builtin-types.c`)

- **Future work**: Auto-vectorization/auto-matmul support is planned.
  Typed builtins (Phase C) are complete -- 121 builtins accept and return
  `__rvm_*_t` matrix types. Register flexibility is fully implemented.
