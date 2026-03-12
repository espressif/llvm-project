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

**Status**: Experimental (requires `+experimental-xtheadmatrix` to enable;
Zmpanel additionally requires `+experimental-xtheadzmpanel`).

**Specification**: The official specification is the
[RISC-V Matrix Extension Spec (RVM)](https://github.com/AII-SDU/riscv-matrix-extension-spec)
maintained by C-SKY MicroSystems / T-Head.

**Implementation summary**: 257 instructions (227 base + 30 Zmpanel),
~220 `_internal` LLVM intrinsics, 7 config intrinsics, 30 Zmpanel
intrinsics, 22 `mundef` builtins, 22 `mget`/`mset` tuple builtins,
~220 Spec-API Clang builtins, and 30 Zmpanel builtins are implemented
with full ISel/CodeGen support.

The **Spec-API (ManagedRA)** programming model is used exclusively:
`_internal` intrinsics produce/consume `target("riscv.matrix")` SSA
values. The register allocator manages tr0-tr3 / acc0-acc3
automatically. ~220 pseudo-instructions with proper register class
constraints (THRVMTR/THRVMACC/THRVMMR) expand post-RA to hardware
instructions. Spec-API builtins cover all operations: loads (A/B/C
tile, whole-register), stores, all 27 matmul variants, element-wise
arithmetic (integer + FP), format conversions, data movement (move,
duplicate, pack, slide, broadcast), fixed-point clip, and zero.

A `<thead_matrix.h>` header provides over 450 higher-level API
functions and macros with C matrix types (`mint8_t`, `mint32_t`,
`mfloat16_t`, etc.) backed by 22 native Clang built-in types
(`__rvm_int8_t` through `__rvm_float64x2_t`). The x2 (register-pair)
types map to `{ target("riscv.matrix"), target("riscv.matrix") }`
structs at the LLVM IR level, with `mget`/`mset` builtins for
extracting and inserting individual registers. Programs can use either
the high-level API or inline assembly.

The **Zmpanel** extension adds 30 panel-aware 2x2 matrix tiling
instructions that operate as fire-and-forget macro instructions on
implicit hardware state (extended tile registers tr4-tr7 and panel
CSRs). These instructions load/compute/store entire 2x2 blocks of
tiles in a single instruction dispatch for higher compute throughput.
Panel load/store/compute instructions carry implicit Defs/Uses on
matrix registers to prevent incorrect reordering. ISel uses the
`THMI_PanelFireForget` dispatch category for these instructions.
Mixed-mode usage (ManagedRA base instructions combined with Zmpanel
fire-and-forget in the same function) is detected and rejected with
a fatal error at ISel time.

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

This builds the complete RISC-V bare-metal cross-compilation toolchain
-- compiler, linker, assembler, debugger, binutils, runtime libraries,
C library, and C++ standard library -- and installs it to a prefix
directory ready for use. The toolchain uses **multilib** to provide
libraries for multiple arch+ABI combinations from a single install.

The build has four stages, each building all multilib variants:

1. **LLVM toolchain** -- clang, lld, lldb, all LLVM tools (no runtimes)
2. **compiler-rt builtins** -- built standalone for each multilib variant
3. **Newlib + libgloss** -- bare-metal C library (libc, libm) and `libnosys` for each variant
4. **C++ runtimes** -- libunwind, libcxxabi, libcxx (static, bare-metal, no threads) for each variant

The default multilib variants are:

| Variant | `-march` | `-mabi` | Use case |
|---------|----------|---------|----------|
| `rv64imafdc/lp64d` | `rv64gc` | `lp64d` | 64-bit with double-precision FPU |
| `rv32imafdc/ilp32d` | `rv32gc` | `ilp32d` | 32-bit with double-precision FPU |
| `rv64imafc/lp64f` | `rv64imafc` | `lp64f` | 64-bit with single-precision FPU only |
| `rv32imafc/ilp32f` | `rv32imafc` | `ilp32f` | 32-bit with single-precision FPU only |
| `rv64imac/lp64` | `rv64imac` | `lp64` | 64-bit soft-float |
| `rv32imac/ilp32` | `rv32imac` | `ilp32` | 32-bit soft-float |

Clang automatically selects the correct variant based on `-march` and
`-mabi` flags via the `multilib.yaml` configuration file.

A convenience script `riscv-toolchain-build.sh` is provided at the
repository root. It accepts an optional install prefix argument
(default: `${HOME}/opt/llvm`):

```bash
./riscv-toolchain-build.sh                    # install to ~/opt/llvm
./riscv-toolchain-build.sh /path/to/install   # custom prefix
./riscv-toolchain-build.sh --portable         # portable build (static deps)
```

The `--portable` flag statically links libstdc++, libgcc, zlib, and zstd
into the host tools, so the installed toolchain can be tarred and moved
to another Linux system (only glibc remains dynamically linked).

The script runs all four stages for all variants automatically. The key
cmake options for each stage are described below.

#### Stage 1: LLVM Toolchain (no runtimes)

compiler-rt **cannot** be included in `LLVM_ENABLE_RUNTIMES` for
bare-metal cross-compilation because the in-tree runtimes build tries
to cross-compile test programs (`test_target_arch`) which fail without
a C library. Instead, compiler-rt is built standalone in Stage 2.

```bash
cmake -S llvm -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_TARGETS_TO_BUILD=RISCV \
  -DLLVM_ENABLE_PROJECTS="clang;lld;llvm;lldb" \
  -DCMAKE_INSTALL_PREFIX=${HOME}/opt/llvm \
  -DLLVM_INSTALL_TOOLCHAIN_ONLY=OFF \
  -DLLVM_DEFAULT_TARGET_TRIPLE=riscv64-unknown-elf

cmake --build build -- -j12
cmake --install build
```

Key options:

| Option | Purpose |
|--------|---------|
| `LLVM_TARGETS_TO_BUILD=RISCV` | Build only the RISC-V backend (faster build) |
| `LLVM_ENABLE_PROJECTS="clang;lld;llvm;lldb"` | Compiler + linker + all LLVM tools + debugger |
| `CMAKE_INSTALL_PREFIX` | Where `cmake --install` places binaries, libs, headers |
| `LLVM_DEFAULT_TARGET_TRIPLE=riscv64-unknown-elf` | Default target triple (required when building only the RISC-V backend, avoids undefined variable errors in LLDB CMake) |

Note: `LLVM_ENABLE_RUNTIMES` is deliberately **not set** here -- see
Stage 2 for why.

This installs: `clang`, `lld` (linker), `lldb` (debugger),
`llvm-ar` (archiver), `llvm-nm`, `llvm-objcopy`, `llvm-objdump`,
`llvm-readelf`, `llvm-strip`, `llvm-mc`, `opt`, `llc`.

#### Stage 2: compiler-rt Builtins (one build per multilib variant)

compiler-rt is built as a standalone CMake project against the
`compiler-rt/` source directory for **each multilib variant**. Each
build uses the variant's `-march`/`-mabi` flags and installs
`libclang_rt.builtins.a` into the variant's `lib/` directory under
`lib/clang-runtimes/`. Clang's `getCompilerRT()` searches the multilib
library paths first, so it finds the correct variant automatically.

Two critical settings:

- `CMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY` -- avoids link-time
  test programs that fail without a C library
- `COMPILER_RT_BAREMETAL_BUILD=ON` -- skips builtins needing libc headers

Key options:

| Option | Purpose |
|--------|---------|
| `CMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY` | Skip link-time tests (no libc available yet) |
| `COMPILER_RT_BAREMETAL_BUILD=ON` | Skip builtins that need libc headers (`sys/mman.h`, etc.) |
| `COMPILER_RT_DEFAULT_TARGET_ONLY=ON` | Build only for the specified target triple |
| `CMAKE_C_FLAGS="-march=... -mabi=..."` | Match the multilib variant's arch and ABI |

Built once per variant (4 total). The builtins `.a` file is copied from
a temporary install prefix into the multilib `lib/` directory.

#### Stage 3: Newlib + libgloss (per multilib variant)

The script automatically clones
[newlib](https://sourceware.org/newlib/) and builds both **newlib**
(libc, libm) and **libgloss** (board support, including `libnosys`)
for each multilib variant using the freshly-built clang. Each variant
is built to a temporary prefix (newlib always installs to
`<prefix>/<target>/`) and the headers and libraries are copied to the
variant's multilib directory.

Key configure options:

- `--disable-newlib-supplied-syscalls` -- libc.a does **not** embed
  syscall implementations; they come from an external library instead
- `--enable-newlib-io-long-long` -- `printf` supports `%lld`
- `--disable-newlib-multithread` -- no threading support (bare-metal)
- `CFLAGS_FOR_TARGET="-O2 -march=... -mabi=..."` -- variant-specific flags

The build uses `make all-target-newlib all-target-libgloss` to build
both components, and `make install-target-newlib install-target-libgloss`
to install them.

**libnosys** (part of libgloss) provides empty stub implementations of
all POSIX syscalls that newlib requires (`_read`, `_write`, `_sbrk`,
`_close`, `_fstat`, `_exit`, etc.). This allows code using `printf`,
`malloc`, etc. to **link** for bare-metal targets. By default output
goes nowhere -- override `_write` in your application to redirect it:

```c
// Override in your application -- printf output goes to UART.
// Your definition takes priority over the stub in libnosys.a
// (standard static archive linker behavior).
int _write(int fd, const char *buf, int len) {
    for (int i = 0; i < len; i++) uart_putc(buf[i]);
    return len;
}
```

#### Stage 4: C++ Runtimes (per multilib variant)

Once newlib is installed, the script builds `libunwind`, `libcxxabi`,
and `libcxx` for each multilib variant using a standalone CMake
invocation against the `runtimes/` directory. Each variant's
`CMAKE_INSTALL_PREFIX` points to its multilib directory.

Four critical gotchas discovered during bring-up:

1. **Do NOT use `CMAKE_SYSROOT`** -- the runtimes build system
   overrides the compiler path when sysroot is set. Pass `--sysroot`
   via `CMAKE_C_FLAGS` / `CMAKE_CXX_FLAGS` / `CMAKE_ASM_FLAGS` instead.
4. **`--sysroot` must point to the `clang-runtimes/` base directory**,
   not the individual variant directory. Pointing it to a variant dir
   causes clang to look for `multilib.yaml` inside the variant
   (where it doesn't exist) and doubles the variant suffix on include
   paths. The multilib mechanism selects the correct variant
   automatically based on `-march`/`-mabi`.
2. **`LIBUNWIND_IS_BAREMETAL=ON` is required** -- without it,
   libunwind tries to include `dlfcn.h` which does not exist in newlib.
3. **`LLVM_INCLUDE_TESTS=OFF` is required** -- avoids a dependency on
   system Clang/LLVM packages for test infrastructure.

The runtimes are configured for bare-metal:

- **Static only** -- no shared libraries (`ENABLE_SHARED=OFF`)
- **No threads** -- `ENABLE_THREADS=OFF` (bare-metal, no OS)
- **No filesystem / locale / wide chars** -- stripped for embedded use
- **Uses compiler-rt** -- not libgcc (`USE_COMPILER_RT=ON`)
- **libcxxabi uses libunwind** -- `USE_LLVM_UNWINDER=ON`
- **`LIBUNWIND_IS_BAREMETAL=ON`** -- skips `dlfcn.h` and OS-specific includes

The C++ headers install to `<variant-dir>/include/c++/v1/` and
libraries to `<variant-dir>/lib/`, so clang finds them automatically
via the multilib selection.

#### Multilib Configuration

The `multilib.yaml` file at `lib/clang-runtimes/multilib.yaml` tells
clang how to map `-march`/`-mabi` flags to library directories. It
uses regex-based `Mappings` to normalize the canonical `-march` string
(with version numbers like `rv64i2p1_m2p0_a2p1_f2p2_d2p2_...`) to
simplified flags that the `Variants` can match.

Use `clang -print-multi-flags-experimental` to see the flags clang
generates, and `clang -print-multi-directory` to see which variant is
selected:

```bash
clang --target=riscv64-unknown-elf -march=rv64gc -print-multi-directory
# rv64imafdc/lp64d
```

#### Installed Layout

```
~/opt/llvm/
├── bin/                                     # clang, clang++, lld, lldb, ...
├── include/                                 # Clang/LLVM headers
├── lib/
│   └── clang-runtimes/
│       ├── multilib.yaml                    # multilib configuration
│       ├── rv64imafdc/lp64d/                # rv64gc + lp64d variant
│       │   ├── include/
│       │   │   ├── stdio.h, stdlib.h, ...   # newlib C headers
│       │   │   └── c++/v1/                  # libc++ C++ headers
│       │   └── lib/
│       │       ├── libclang_rt.builtins.a   # compiler-rt builtins
│       │       ├── libc.a                   # newlib C library
│       │       ├── libm.a                   # newlib math library
│       │       ├── libnosys.a               # empty syscall stubs
│       │       ├── libc++.a                 # C++ standard library
│       │       ├── libc++abi.a              # C++ ABI library
│       │       └── libunwind.a              # stack unwinder
│       ├── rv32imafdc/ilp32d/               # rv32gc + ilp32d variant
│       │   ├── include/                     # (same structure)
│       │   └── lib/
│       ├── rv64imafc/lp64f/                 # rv64imafc + lp64f variant
│       │   ├── include/
│       │   └── lib/
│       ├── rv32imafc/ilp32f/                # rv32imafc + ilp32f variant
│       │   ├── include/
│       │   └── lib/
│       ├── rv64imac/lp64/                   # rv64imac + lp64 variant
│       │   ├── include/
│       │   └── lib/
│       └── rv32imac/ilp32/                  # rv32imac + ilp32 variant
│           ├── include/
│           └── lib/
└── share/
```

#### Add to PATH

```bash
export PATH="${HOME}/opt/llvm/bin:$PATH"
```

#### Relocatable Toolchain

The installed toolchain is fully relocatable -- it can be tarred up and
moved to a different directory or machine without rebuilding:

```bash
# Package
tar -czf riscv-llvm-toolchain.tar.gz -C ~/opt riscv-llvm

# Deploy to another machine (same or newer Linux version)
tar -xzf riscv-llvm-toolchain.tar.gz -C /some/path
export PATH="/some/path/riscv-llvm/bin:$PATH"
```

This works because:

- **No hardcoded paths** -- clang finds all resources (multilib.yaml,
  headers, libraries) relative to its own binary location
  (`bin/../lib/clang-runtimes/`, `bin/../lib/clang/<ver>/`)
- **All target libraries are static** -- `.a` archives have no rpaths
  or dynamic linker dependencies
- **multilib.yaml uses relative `Dir`** -- variant paths like
  `rv64imafdc/lp64d` are resolved relative to the sysroot

The only host requirement is a compatible Linux (same or newer kernel
and glibc) since the host tools (`clang`, `lld`, `lldb`) are
dynamically linked against the host C library.

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
| `lldb` | Debugger (like GDB but LLVM-native, supports RISC-V targets) |
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

Matrix registers can be used in inline assembly with register constraints or
hardcoded register names.

#### Register Constraints

Three constraint letters are available for matrix registers:

| Constraint | Register Class | Registers |
|------------|---------------|-----------|
| `tr` | THRVMMR (any matrix) | tr0-tr3, acc0-acc3 |
| `tt` | THRVMTR (tile only) | tr0-tr3 |
| `ta` | THRVMACC (accumulator only) | acc0-acc3 |

Clobber constraints (`"tr0"`, `"acc0"`) are also supported. Named register
constraints (`{tr0}`, `{acc1}`, etc.) are available at the LLVM IR level via
`getRegForInlineAsmConstraint`.

```c
#include <thead_matrix.h>

// Constrained output — register allocator picks a matrix register
mint32_t result;
asm volatile("th.mlae8 %0, (%1), %2"
             : "=tr"(result)
             : "r"(base), "r"(stride));

// Constrained input
asm volatile("th.msce8 %0, (%1), %2"
             : /* no outputs */
             : "tr"(tile), "r"(base), "r"(stride));

// Accumulator-only constraint
mint32_t acc;
asm volatile("th.mlae8 %0, (%1), %2"
             : "=ta"(acc)
             : "r"(base), "r"(stride));

// Read-write constraint
asm volatile("th.mmov.mm %0, %0" : "+tr"(val));

// Clobber constraints
asm volatile("th.mcfg zero, zero, zero" ::: "tr0", "acc0");
```

#### Hardcoded Register Names (Legacy)

For pure string-template inline assembly without register allocator interaction:

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
               mrow_t M, mcol_t K, mcol_t N) {
    mint8_t  a = __riscv_th_mld_a_i8(A, a_stride, M, K);
    mint8_t  b = __riscv_th_mld_b_i8(B, b_stride, K, N);
    mint32_t c = __riscv_th_mzeros_i32(M, N);
    c = __riscv_th_mmaq_ss_w_b(c, a, b, M, K, N);
    __riscv_th_mst_i32(C, c_stride, c, M, N);
    __riscv_th_mrelease();
}
```

Compare this with the equivalent low-level builtin version in
[Example 1](#example-1-int8-gemm-kernel) below -- the higher-level API
carries dimension parameters and returns typed matrix values, making the
data flow explicit.

### Example: FP64 GEMM Using x2 Types

Some matmul variants use x2 (register-pair) types for the accumulator
or B operand. The `mget`/`mset` functions extract and insert individual
registers from x2 pairs. At `-O2`, these struct operations are folded
away to direct value passing.

```c
#include <thead_matrix.h>

void fp64_gemm_x2(const double *A, long a_stride,
                  const double *B, long b_stride,
                  double *C, long c_stride,
                  mrow_t M, mcol_t K, mcol_t N) {
    mfloat64_t ta = __riscv_th_mld_a_f64(A, a_stride, M, K);
    mfloat64_t tb = __riscv_th_mld_b_f64(B, b_stride, K, N);
    mfloat64_t c0 = __riscv_th_mld_acc_f64(C, c_stride, M, N);
    // Build an x2 pair with c0 in slot 0
    mfloat64x2_t c_pair = __riscv_th_mset_f64(
        __builtin_riscv_th_mundef_f64x2(), 0, c0);
    // x2 matmul — operates on component 0 internally
    mfloat64x2_t res_pair = __riscv_th_mfmaqa_d_x2(c_pair, ta, tb, M, K, N);
    mfloat64_t res = __riscv_th_mget_f64(res_pair, 0);
    __riscv_th_mst_f64(C, c_stride, res, M, N);
}
```

### API Categories

The 421 functions and macros in `<thead_matrix.h>` are organized into
these categories:

| Category | Examples | Description |
|----------|----------|-------------|
| Configuration | `__riscv_th_msettilem`, `__riscv_th_mrelease` | Set tile dimensions, release matrix unit |
| CSR access | `__riscv_th_mread_csr`, `__riscv_th_mwrite_csr` | Read/write matrix CSRs |
| Load | `__riscv_th_mld_a_i8`, `__riscv_th_mld_b_f32` | Load tiles from memory with type and role |
| Store | `__riscv_th_mst_i32`, `__riscv_th_mst_f16` | Store tiles to memory |
| Matrix multiply | `__riscv_th_mmaq_ss_w_b`, `__riscv_th_mfmaqa_s` | Matmul-accumulate (integer and FP) |
| EW arithmetic | `__riscv_th_madd_w_mm`, `__riscv_th_mfmul_s_mm` | Element-wise add, sub, mul, min, max, shift |
| Conversions | `__riscv_th_mfcvtl_s_h`, `__riscv_th_msfcvt_s_w` | FP and integer format conversions |
| Tuple ops | `__riscv_th_mget_f16`, `__riscv_th_mset_f64` | Extract/insert single register from x2 pair |
| x2 matmul | `__riscv_th_mfmaqa_d_x2`, `__riscv_th_mmaq_ss_d_h_x2` | Matmul with x2 pair operand/result |
| Data movement | `__riscv_th_mzeros_i32`, `__riscv_th_mmov_mm` | Zero, move, pack, slide, broadcast |

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

The implementation uses the **Spec-API (ManagedRA)** programming model
exclusively. Matrix values are represented as opaque `__rvm_*_t` built-in
types at the Clang level, which map to `target("riscv.matrix")` SSA
values in LLVM IR. The register allocator automatically assigns matrix
registers (tr0-tr3 for tile operations, acc0-acc3 for accumulator
operations) based on register class constraints in the pseudo-instructions.

No manual register index management is needed. The compiler handles all
register assignment, spilling, and reloading automatically.

### Typical Workflow

A typical matrix computation follows these steps:

1. **Load** -- Load matrix tiles from memory (dimension CSRs are set automatically)
2. **Zero** -- Initialize accumulators using `__riscv_th_mzeros_*`
3. **Compute** -- Perform matrix multiply or element-wise operations
4. **Store** -- Store results back to memory
5. **Release** -- Release the matrix unit using `mrelease` (optional)

### Matrix Dimensions

The matrix unit operates on tiles whose dimensions are configured via CSRs:

- **M** (rows of A and C): set by `msettilemi`/`msettilem`
- **K** (columns of A, rows of B): set by `msettileki`/`msettilek`
- **N** (columns of B and C): set by `msettileni`/`msettilen`

The hardware clamps these values to implementation-defined maximums.
Query `th.xtlenb` and `th.xtrlenb` CSRs for the hardware tile dimensions.

The Spec-API wrapper functions automatically emit the appropriate CSR
configuration calls (msettilem/k/n) before each load/store/matmul/zero
operation, so users do not need to call them manually.

### The `<thead_matrix.h>` Header

The `<thead_matrix.h>` header provides the complete programming
interface with C matrix types (`mint8_t`, `mfloat16_t`, etc.) and
dimension parameters (`mrow_t`, `mcol_t`). Functions like
`__riscv_th_mld_a_i8`, `__riscv_th_mmaq_ss_w_b`, and
`__riscv_th_mst_i32` carry dimension information, return typed values,
and automatically configure tile dimensions, making matrix code
readable and type-safe. See the
[Higher-Level Intrinsic API](#higher-level-intrinsic-api) section and
all [Code Examples](#code-examples).

## Builtins Reference

> **Note**: The tables below document the 227 hardware instructions and
> their assembly mnemonics. The actual Spec-API Clang builtins (defined
> in `BuiltinsRISCVXTHeadMatrix.td`) use `_spec` suffixed names (e.g.,
> `__builtin_riscv_th_mfmaqa_spec_h` rather than
> `__builtin_riscv_th_mfmacc_h`). Users should use the `<thead_matrix.h>`
> C API (e.g., `__riscv_th_mfmaqa_h`) rather than calling builtins
> directly.

All Spec-API builtins use the `__builtin_riscv_th_` prefix and are
available when the `xtheadmatrix` target feature is enabled. They accept
and return `__rvm_*_t` matrix types; no register index parameters are
needed. The builtins map to `_internal` intrinsics that produce/consume
`target("riscv.matrix")` SSA values.

22 `mundef` builtins (`__builtin_riscv_th_mundef_i8` through
`__builtin_riscv_th_mundef_f64x2`) create undefined matrix values for
initializing variables.

22 `mget`/`mset` tuple builtins (`__builtin_riscv_th_mget_spec_i8`
through `__builtin_riscv_th_mset_spec_f64`) extract and insert single
registers from x2 (register-pair) types. At the IR level, `mget` emits
`extractvalue` + `select` and `mset` emits `insertvalue` + `select`;
at `-O2` with constant indices, these fold to direct struct access.

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

Matrix multiply builtins accept and return `__rvm_*_t` typed arguments.
Native-precision (h, s, d), typed-widening (s_h, d_s), and all integer
variants use the corresponding FP or integer matrix types for all operands.
FP8/BF16/TF32 widening variants use `__rvm_int32_t` (opaque) for the A/B
source tile operands since no native matrix types exist for these formats;
the accumulator operand uses the appropriate output type (`__rvm_float16_t`
or `__rvm_float32_t`).

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
               mrow_t M, mcol_t K, mcol_t N) {
    // Load input matrices (tile config is automatic)
    mint8_t a = __riscv_th_mld_a_i8(A, a_stride, M, K);
    mint8_t b = __riscv_th_mld_b_i8(B, b_stride, K, N);

    // Zero accumulator and compute: c = a * b
    mint32_t c = __riscv_th_mzeros_i32(M, N);
    c = __riscv_th_mmaq_ss_w_b(c, a, b, M, K, N);

    // Store result and release
    __riscv_th_mst_i32(C, c_stride, c, M, N);
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
                   mrow_t M, mcol_t K, mcol_t N) {
    // Load existing accumulator from memory
    mfloat32_t c = __riscv_th_mld_acc_f32(C, c_stride, M, N);

    // Load input tiles
    mfloat32_t a = __riscv_th_mld_a_f32(A, a_stride, M, K);
    mfloat32_t b = __riscv_th_mld_b_f32(B, b_stride, K, N);

    // Accumulate: c += a * b
    c = __riscv_th_mfmaqa_s(c, a, b, M, K, N);

    // Store and release
    __riscv_th_mst_f32(C, c_stride, c, M, N);
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
                       mrow_t M, mcol_t K, mcol_t N) {
    mfloat16_t a = __riscv_th_mld_a_f16(A, a_stride, M, K);
    mfloat16_t b = __riscv_th_mld_b_f16(B, b_stride, K, N);
    mfloat32_t c = __riscv_th_mzeros_f32(M, N);

    // Widening matmul: FP16 * FP16 → FP32
    c = __riscv_th_mfmaqa_s_h(c, a, b, M, K, N);

    __riscv_th_mst_f32(C, c_stride, c, M, N);
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
    mint32_t r = __riscv_th_mld_acc_i32(result, r_stride, M, N);
    mint32_t b = __riscv_th_mld_acc_i32(bias, b_stride, M, N);

    // Element-wise add: r = r + bias
    r = __riscv_th_madd_w_mm(r, r, b);

    // Element-wise shift right: r >>= shift_amounts
    mint32_t shift = __riscv_th_mld_acc_i32(bias, b_stride, M, N);
    r = __riscv_th_msra_w_mm(r, r, shift);

    // N4Clip: (acc, scale, data) -> clipped acc (still int32)
    mint32_t scale = __riscv_th_mld_acc_i32(bias, b_stride, M, N);
    mint32_t clipped = __riscv_th_mn4clipl_w_mm(r, scale, shift);

    // Store int32 output (use packed conversion for int8 output)
    __riscv_th_mst_i32(output, o_stride, clipped, M, N);
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
                         float scale, mrow_t M, mcol_t K, mcol_t N) {
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
    mint32_t c = __riscv_th_mzeros_i32(M, N);
    c = __riscv_th_mmaq_ss_w_b(c, a, b, M, K, N);
    __riscv_th_mst_i32(output, o_stride, c, M, N);
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
               mrow_t M, mcol_t K, mcol_t N) {
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

### Example 8: Zmpanel INT8 Panel GEMM (Fire-and-Forget)

The Zmpanel extension provides panel-aware 2x2 tiling instructions that
operate on implicit hardware state. A single load instruction loads all 8
tile registers (tr0-tr7), a single compute instruction performs the full
2x2 block GEMM, and a single store instruction writes all 4 accumulator
results. This is significantly more efficient than orchestrating individual
tile operations.

```c
#include <thead_matrix.h>

// Panel 2x2 INT8 GEMM: D[2M,2N] += A[2M,2K] * B[2K,2N]
// Uses fire-and-forget macro instructions (Zmpanel)
void panel_gemm_int8(const void *A, const void *B, void *D,
                     size_t stride_a, size_t stride_b, size_t stride_d,
                     size_t M, size_t N, size_t K) {
    // 1. Configure panel parameters (addresses, strides, dimensions)
    __riscv_th_mset22adra((size_t)A);
    __riscv_th_mset22adrb((size_t)B);
    __riscv_th_mset22adrd((size_t)D);
    __riscv_th_mset22rsba(stride_a);
    __riscv_th_mset22rsbb(stride_b);
    __riscv_th_mset22rsbd(stride_d);
    __riscv_th_mset22m(M);
    __riscv_th_mset22n(N);
    __riscv_th_mset22k(K);
    __riscv_th_msetaccum(0);       // zero mode: zero acc before first compute
    __riscv_th_msetoob(2);         // load=zero-pad (1), store=skip (0) -> 0b10 = 2
    __riscv_th_msetrstptr(1);      // reset all HW pointers

    // 2. Execute the panel pipeline:
    //    - ml22e8: loads tr0-tr7 (8 tile regs) from A and B memory
    //    - mmacc22.w.b: computes 2x2 block MAC → acc0-acc3
    //    - msc22e32: stores acc0-acc3 to D memory
    __riscv_th_ml22e8();
    __riscv_th_mmacc22_w_b();
    __riscv_th_msc22e32();

    // 3. Fence before reading results from memory
    asm volatile("fence");
}
```

Compile with Zmpanel enabled:
```bash
clang --target=riscv64 -march=rv64i_xtheadzmpanel0p6 \
  -menable-experimental-extensions -O2 -c panel_gemm.c -o panel_gemm.o
```

### Example 9: Zmpanel FP16 Panel GEMM with K-Loop

For larger K dimensions, the panel GEMM can be iterated in a loop. The
hardware pointer CSRs automatically advance through the K dimension on
each load/compute iteration.

```c
#include <thead_matrix.h>

// Panel 2x2 FP16 GEMM with K-dimension loop
// The hardware automatically advances pointers through K iterations
void panel_gemm_fp16_k_loop(const void *A, const void *B, void *D,
                            size_t stride_a, size_t stride_b, size_t stride_d,
                            size_t M, size_t N, size_t K,
                            size_t k_iters) {
    // Setup
    __riscv_th_mset22adra((size_t)A);
    __riscv_th_mset22adrb((size_t)B);
    __riscv_th_mset22adrd((size_t)D);
    __riscv_th_mset22rsba(stride_a);
    __riscv_th_mset22rsbb(stride_b);
    __riscv_th_mset22rsbd(stride_d);
    __riscv_th_mset22m(M);
    __riscv_th_mset22n(N);
    __riscv_th_mset22k(K);
    __riscv_th_msetaccum(0);       // zero acc on first iter
    __riscv_th_msetoob(2);
    __riscv_th_msetrstptr(1);

    // K-dimension loop: load and accumulate
    for (size_t ki = 0; ki < k_iters; ki++) {
        __riscv_th_ml22e16();
        __riscv_th_mfmacc22_s_h();     // fp16 -> fp32 widening MAC
    }

    // Store accumulated results
    __riscv_th_msc22e32();

    asm volatile("fence");
}
```

### Example 10: Zmpanel with Inline Assembly

For low-level control, Zmpanel instructions can be used directly via
inline assembly. This is useful for hand-tuned kernels.

```c
#include <stddef.h>

void panel_gemm_asm(const void *A, const void *B, void *D,
                    size_t stride_a, size_t stride_b, size_t stride_d,
                    size_t M, size_t N, size_t K) {
    asm volatile(
        // Configure
        "th.mset22adra  %[a]\n\t"
        "th.mset22adrb  %[b]\n\t"
        "th.mset22adrd  %[d]\n\t"
        "th.mset22rsba  %[sa]\n\t"
        "th.mset22rsbb  %[sb]\n\t"
        "th.mset22rsbd  %[sd]\n\t"
        "th.mset22m     %[m]\n\t"
        "th.mset22n     %[n]\n\t"
        "th.mset22k     %[k]\n\t"
        "th.msetaccum   zero\n\t"       // zero mode
        "li             t0, 2\n\t"
        "th.msetoob     t0\n\t"         // load=zero-pad, store=skip
        "li             t0, 1\n\t"
        "th.msetrstptr  t0\n\t"         // reset pointers
        // Execute
        "th.ml22e8\n\t"                 // load 2x2 tiles
        "th.mmacc22.w.b\n\t"           // int8 panel MAC
        "th.msc22e32\n\t"              // store results
        "fence\n\t"
        :
        : [a] "r"(A), [b] "r"(B), [d] "r"(D),
          [sa] "r"(stride_a), [sb] "r"(stride_b), [sd] "r"(stride_d),
          [m] "r"(M), [n] "r"(N), [k] "r"(K)
        : "t0", "memory"
    );
}
```

## CSR Reference

XTHeadMatrix defines 13 base CSRs, all prefixed with `th.`.
The Zmpanel extension adds 18 additional CSRs (addresses 0xcc4-0xcd5).

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

### Zmpanel Panel-Aware CSRs (addresses 0xcc4-0xcd5, all URO)

Written via panel configuration instructions (`th.mset22*`), read-only via CSR reads.

| Address | Name | Description |
|---------|------|-------------|
| 0xcc4 | CUSTOM_CTRL | `[0]`: custom_cap, `[1]`: accum_mode, `[2]`: oob_load, `[3]`: oob_store |
| 0xcc5 | BASE_ADDR_A | Base address of matrix A |
| 0xcc6 | BASE_ADDR_B | Base address of matrix B |
| 0xcc7 | BASE_ADDR_D | Base address of matrix D (output) |
| 0xcc8 | RSTRIDEB_A | Row stride of matrix A in bytes |
| 0xcc9 | RSTRIDEB_B | Row stride of matrix B in bytes |
| 0xcca | RSTRIDEB_D | Row stride of matrix D in bytes |
| 0xccb | PANEL_M | Panel M dimension |
| 0xccc | PANEL_N | Panel N dimension |
| 0xccd | PANEL_K | Panel K dimension |
| 0xcce | MPTR_LD | `[16:1]`=PTR_M_LD, `[0]`=PTR_22M_LD |
| 0xccf | NPTR_LD | `[16:1]`=PTR_N_LD, `[0]`=PTR_22N_LD |
| 0xcd0 | KPTR_LD | `[16:1]`=PTR_K_LD, `[0]`=PTR_22K_LD |
| 0xcd1 | MPTR_ST | `[16:1]`=PTR_M_ST, `[0]`=PTR_22M_ST |
| 0xcd2 | NPTR_ST | `[16:1]`=PTR_N_ST, `[0]`=PTR_22N_ST |
| 0xcd3 | ADDR_A | Current computed address of A |
| 0xcd4 | ADDR_B | Current computed address of B |
| 0xcd5 | ADDR_D | Current computed address of D |

C header defines: `__RVM_CSR_CUSTOM_CTRL` (0xcc4) through `__RVM_CSR_ADDR_D` (0xcd5).

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
| **Base Total** | **227** | |

### Zmpanel Panel-Aware Instructions (func3=010)

| Category | Count | Mnemonics |
|----------|-------|-----------|
| Panel Config | 12 | `mset22adr{a,b,d}`, `mset22rsb{a,b,d}`, `mset22{m,n,k}`, `msetrstptr`, `msetaccum`, `msetoob` |
| Panel Load | 2 | `ml22e{8,16}` |
| Panel Store | 2 | `msc22e{16,32}` |
| Panel FP Compute | 10 | `mfmacc22.{h.e5,h.e4,bf16.e5,bf16.e4,s.e5,s.e4,h,s.h,s.bf16,s}` |
| Panel INT Compute | 4 | `m{,u}macc{us,su,u,}22.w.b` |
| **Zmpanel Total** | **30** | |
| **Grand Total** | **257** | |

## Limitations and Notes

### Implementation Limitations

- **Experimental status**: The extension requires `+experimental-xtheadmatrix`
  and may change in future LLVM releases as the spec evolves.

- **No auto-matmul from C loops**: Code generation from plain C arithmetic
  (e.g., `for (i) for (j) C[i][j] += A[i][k]*B[k][j]`) will not automatically
  use matrix instructions. Users must use builtins, the `<thead_matrix.h>` API,
  or inline assembly.

- **Matrix types cannot cross function boundaries**: `target("riscv.matrix")` /
  `__rvm_*_t` values have no ABI-level calling convention. They cannot be passed
  as function parameters or returned from functions. Matrix operations must stay
  within a single function scope (or be inlined via `always_inline`).

- **Limited register file (4+4)**: Only 4 tile registers and 4 accumulator
  registers are available. Complex multi-tile GEMM kernels face high register
  pressure.

- **Whole-register spill granularity**: Spill and reload use whole-register
  load/store (`TH_MSME_E8`/`TH_MLME_E8`). No partial-register spill
  optimization is available. Each spill/reload transfers the full 8192-bit
  (1024-byte) register.

- **64-bit instruction format not supported**: The spec defines 64-bit
  instruction formats (`inst64_format.adoc`), but only the 32-bit format
  is implemented. All 257 defined instructions use 32-bit encoding.

- **Zmpanel is fire-and-forget**: Panel-aware instructions operate on
  implicit hardware state (extended tile registers tr4-tr7 and panel CSRs).
  The compiler cannot schedule, reorder, or optimize across panel macro
  instructions. No register allocation is performed for panel registers --
  the hardware manages them autonomously. Panel load/store/compute
  instructions carry implicit Defs/Uses to prevent reordering. Mixed-mode
  usage (combining ManagedRA base instructions with Zmpanel fire-and-forget
  in the same function) is detected at ISel time and rejected with a fatal
  error, since the two models have incompatible assumptions about matrix
  register ownership.

- **`-O0` support is limited for ManagedRA**: The `RISCVLowerMatrixType` pass
  provides basic `-O0` support by lowering `target("riscv.matrix")` to allocas.
  Full optimization (`-O1` or higher) is recommended for the ManagedRA model.

### Differences from Spec Intrinsic API (rvm-intrinsic-api.adoc v0.2)

The implementation provides complete coverage of all RVM 0.6 hardware
instructions. The following differences exist relative to the spec's
C intrinsic API definition:

- **No C++ overloading**: The spec envisions C++ overloaded functions
  (e.g., a single `__riscv_th_mld` overloaded by pointer type). The
  implementation uses separate C functions per type (`__riscv_th_mld_a_i8`,
  `__riscv_th_mld_a_i16`, etc.) since C does not support overloading.

- **Role-specific loads**: The spec uses a unified
  `__riscv_th_mld(base, stride, row, col)`. The implementation uses
  role-specific functions because each maps to a different hardware
  instruction: `__riscv_th_mld_a_*` (A-tile, mlae), `__riscv_th_mld_b_*`
  (B-tile, mlbe), and `__riscv_th_mld_acc_*` (C-tile, mlce).

- **Matmul naming convention**: The spec uses `mmaqa`/`mmaqau`/`mmaqaus`/
  `mmaqasu` naming for the high-level API. The hardware instructions use
  `mmacc`/`mmaccu`/`mmaccus`/`mmaccsu`. The implementation uses `mmacc_*`
  for low-level builtins (matching hardware mnemonics) and `mmaqa_*` for
  the high-level `<thead_matrix.h>` API (matching the spec naming).

- **Spec features with no hardware instructions** (not implementation gaps):
  - **Stream load/store** (`msld`, `msst`): Mentioned in the spec but no
    instructions exist in `instruction_list.adoc`.
  - **Matrix-scalar EW operations** (`.mx` variants): Listed in the spec
    API but no corresponding hardware instructions in RVM 0.6.
  - **64-bit integer EW arithmetic** (`.d` variants): The spec lists
    `mint64_t` variants for `madd`, `msub`, etc. but `instruction_list.adoc`
    only defines `.w` (32-bit) integer element-wise operations.
  - **`mmov.mv.x` / `mmov.mv.i`**: Mentioned in the spec but not present
    as separate instructions. The functionality can be achieved via
    `mrslidedown`/`mrbca` instructions.

### Register Constraints and Sema Validation

The compiler enforces RVM 0.6 register type constraints at compile time:
  - **Load A/B** (`mla*`/`mlb*`): `md` must be a tile register (0-3)
  - **Load C** (`mlc*`): `md` must be an accumulator register (4-7)
  - **Store A/B** (`msa*`/`msb*`): `ms3` must be a tile register (0-3)
  - **Store C** (`msc*`): `ms3` must be an accumulator register (4-7)
  - **Matmul**: `md` must be accumulator (4-7), `ms1`/`ms2` must be tile (0-3)
  - **Element-wise** (arithmetic, conversions, N4clip): all `md/ms1/ms2` must be accumulator (4-7)
  - **Load/Store M** (`mlm*`/`msm*`), **zero**, **move**, **duplicate**: any register (0-7)
  - **Slides**, **broadcasts**, **pack**: any register (0-7)

Invalid register usage produces a compile-time error.

### Register Index Constants

The `<thead_matrix.h>` header defines named constants:
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

### Known Spec Errata

Three errata exist in the RVM 0.6 spec:
1. **Matmul uop field**: `instruction_list.adoc` shows `uop=01` for all
   matmul instructions. The correct value is `uop=10` (per
   `inst32_format.adoc`; `uop=01` is Load/Store). LLVM uses the correct `10`.
2. **mfmin name swap**: `mfmin.h`/`mfmin.s` labels are swapped in the
   encoding table (encodings are correct, names are wrong). LLVM correctly
   maps `.h`→s_size=01, `.s`→s_size=10.
3. **Typo**: `pmmaaccus.w.b` has an extra 'a' (should be `pmmaccus.w.b`).

### Other Notes

- **Header-only, no runtime library**: The `<thead_matrix.h>` header is a
  header-only wrapper around compiler builtins. All matrix operations compile
  directly to inline instructions. No separate runtime support library exists.

- **Two API levels**: Both low-level instruction-mapped builtins
  (`__builtin_riscv_th_*`) and the higher-level `<thead_matrix.h>` API with
  C matrix types (`mint32_t`, `mfloat16_t`, etc.) are available. 22 native
  Clang built-in types (`__rvm_int8_t` through `__rvm_float64x2_t`) provide
  first-class type identity, mangling, debug info, and tooling support.

- **ManagedRA / Spec-API**: Full register-allocator-managed programming model
  with A-tile loads (`__riscv_th_mld_a_*`, sets M/K), B-tile loads
  (`__riscv_th_mld_b_*`, sets K/N), accumulator loads (`__riscv_th_mld_acc_*`,
  sets M/N), stores (`__riscv_th_mst_*`, sets M/N), INT matmul
  (`__riscv_th_mmaq_*`), FP matmul (`__riscv_th_mfmaqa_*`), and zero
  constructors (`__riscv_th_mzeros_*`). All operations available in 11 types
  (i8/i16/i32/i64/u8/u16/u32/u64/f16/f32/f64).

- **Test coverage**: 17 test files cover assembly encoding (227 instructions),
  error diagnostics, CSR names (13 CSRs), ISel patterns, end-to-end builtin
  codegen (including all 8 widening FP matmul variants), inline assembly
  constraints, API usage patterns, corner cases, and built-in type support.

### Verification History

- **Verification #1** (2026-03-04, Gemini): Found and fixed 2 bugs (42
  conversion pseudo register classes, spec-API matmul operand swap), filled 3
  coverage gaps (B-tile load, FP/unsigned types, all matmul variants).
- **Verification #2** (2026-03-04, Claude Opus 4.6): Full re-verification against
  spec source files. All 227 encodings, 447 ISel entries, 220 pseudo expansion
  entries, spec-API codegen, 13 CSRs, and 447 intrinsic signatures confirmed
  correct. No new bugs found.
- **Verification #3** (2026-03-04, Claude Opus 4.6): End-to-end compilation
  verification and spec-API usability audit. Found and fixed:
  - **Build error**: Forward declaration of `lookupTHMatrixPseudo()` missing in
    `RISCVExpandPseudoInsts.cpp` (called at line 229, defined at line 874).
  - **32 name collisions** in `thead_matrix.h`: DirectReg and Spec-API defined
    functions with the same names (`mld_b_*`, `mmaqa_*`, `mzero_*`). Fixed by
    renaming Spec-API functions: B-load `mld_b_*` → `mldb_*`, INT matmul
    `mmaqa_*` → `mmaq_*`, zero `mzero_*` → `mzeros_*`.
  - **Type inconsistencies**: `__riscv_th_msetmrow_n` return/param type
    `mrow_t` → `mcol_t`; K dimension param in 7 DirectReg matmul macros
    `mrow_t` → `mcol_t`.
  - **Spec-API test rewritten**: `xtheadmatrix-spec-api.c` now uses
    `#include <thead_matrix.h>`, native types (`mint8_t`, `mfloat32_t`), and
    the public C wrapper functions (7 test cases covering INT8/INT16/UINT8/FP32
    matmul pipelines, zero init, and shorthand aliases).
  - **Critical RA fix**: Config intrinsics (`msettilem/k/n`) no longer force
    DirectReg programming model, which was preventing ManagedRA register
    allocation from working entirely. Now Spec-API code compiles with correct
    tile/accumulator register assignment and automatic spill/reload under
    register pressure.

4. **DirectReg removal and Spec-API completion (2026-03-04)**: Removed the
   DirectReg programming model entirely. All DirectReg intrinsics (~160),
   builtins (~265), ISel infrastructure (~530 lines), Sema validation, and
   test files were deleted. Added ~130 new Spec-API builtins for all
   remaining operations (EW arithmetic, format conversions, float-int
   conversions, packed conversions, n4clip, data movement, pack,
   slide/broadcast). The Spec-API now covers ALL 227 hardware instructions.
   Config intrinsics (msettilem/k/n, mrelease) were moved into the
   ManagedRA ISel dispatch. Verified: 9 regression tests + 555 MC tests
   pass; end-to-end RA tests confirm correct register allocation with
   spill/reload under pressure for all operation categories.
5. **Widening FP matmul fix (2026-03-04, Claude Opus 4.6)**: Found and
   fixed a HIGH-severity bug in 8 widening FP matmul builtins
   (FP8/BF16/TF32 variants). The `SpecAPIMatmulWiden` lambda passed
   the accumulator SSA value as all three intrinsic operands `{acc, acc, acc}`,
   causing register class conflicts (THRVMACC vs THRVMTR). Fix: changed
   builtin prototypes from 4-arg `(acc, m, k, n)` to 6-arg
   `(acc, a, b, m, k, n)` with `__rvm_int32_t` for opaque A/B tile
   operands, updated the `__THEAD_SPEC_FMMAQA_WIDEN` macro, removed the
   `SpecAPIMatmulWiden` lambda, and unified all matmul dispatch through
   `SpecAPIMatmul`. Added 8 widening matmul tests to
   `xtheadmatrix-spec-api.c`, 3 ISel tests to `xtheadmatrix-managed-ra.ll`,
   and a new end-to-end example test `xtheadmatrix-spec-api-example.c`.
   All 8 xtheadmatrix tests pass.
6. **Independent verification (2026-03-04, Claude Opus 4.6 #5)**: Full
   independent audit of all 227 instruction encodings, 224 intrinsics, 233
   Spec-API functions, register class constraints, CSR setting logic, matmul
   operand swap, signedness mapping, inline asm constraints, and pseudo
   expansion. **No correctness bugs found.** Design notes documented below.

## Design Notes

### Element-wise tied constraint (optimization opportunity)

Element-wise pseudo instructions (`PTH_MADD_W_MM_V`, `PTH_MSUB_W_MM_V`,
etc.) use a `$src1 = $dst` tied constraint identical to the matmul
accumulate pattern. However, the hardware computes `md = ms2 op ms1`
**without reading the old md value** — these are pure binary operations.
The spec says e.g. "madd performs the addition of src1 and src2" with no
`md +` on the RHS.

The tied constraint forces the RA to co-locate the output with the `acc`
input, causing unnecessary spills when `acc` is still live. A dedicated
`THMI_BinaryAcc` ISel category with untied
`(outs THRVMACC:$dst), (ins THRVMACC:$ms2, THRVMACC:$ms1)` would give the
RA more freedom. This is a minor performance concern, not a correctness bug.

### Unimplemented optional extensions

- **Zmint4**: `mmacc.w.q` (INT4→INT32 matmul) and INT4↔INT8 conversion
  instructions are not implemented. Zmint4 is optional (xmisa bit 0).
- **`.mv.x` element-wise variants**: Appear in the intrinsic API design doc
  (`rvm-intrinsic-api.adoc`) but are absent from the actual instruction list
  (`instruction_list.adoc`). Correctly omitted.

### Element-wise `msub` operand order

`__riscv_th_msub_w_mm(acc, s2, s1)` computes `ms1 - ms2` (i.e. `s1 - s2`),
so the subtrahend appears before the minuend in the parameter list. This
matches the hardware encoding `th.msub.w.mm md, ms2, ms1` directly but may
be unintuitive for users. The parameter names `__s2`, `__s1` clarify the
intended mapping.
