# Build History and Issues Resolved

## Build Configuration

### Development Build (Minimal)

Used during development for fast iteration. Builds only the compiler,
assembler, and disassembler -- enough for testing XTHeadMatrix encoding,
intrinsics, and builtins but **not** sufficient for linking executables.

```bash
cmake -G Ninja ../llvm \
  -DCMAKE_BUILD_TYPE=Debug \
  -DLLVM_TARGETS_TO_BUILD=RISCV \
  -DLLVM_ENABLE_PROJECTS="clang" \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++

ninja -j12 clang llvm-mc llvm-objdump
```

What this builds: `clang` (compiler + integrated assembler), `llvm-mc`
(standalone assembler), `llvm-objdump` (disassembler).

What this does **not** build: linker (`lld`), archiver (`llvm-ar`),
other binutils (`llvm-nm`, `llvm-readelf`, `llvm-strip`, `llvm-objcopy`),
runtime libraries (`compiler-rt`), or optimizer/codegen tools (`opt`, `llc`).

### Full Toolchain Build and Install (Multilib)

Builds the complete RISC-V bare-metal cross-compilation toolchain
(compiler, linker, debugger, all binutils, runtime libraries, C library,
and C++ standard library) with **multilib** support for multiple
arch+ABI combinations, and installs to a prefix. A convenience script
`riscv-toolchain-build.sh` at the repo root automates all four stages:

```bash
./riscv-toolchain-build.sh                    # install to ~/opt/riscv-llvm
./riscv-toolchain-build.sh /path/to/install   # custom prefix
```

**Multilib variants:**

| Variant dir | `-march` | `-mabi` | Use case |
|-------------|----------|---------|----------|
| `rv64imafdc/lp64d` | `rv64gc` | `lp64d` | 64-bit with double FPU |
| `rv32imafdc/ilp32d` | `rv32gc` | `ilp32d` | 32-bit with double FPU |
| `rv64imafc/lp64f` | `rv64imafc` | `lp64f` | 64-bit with single FPU only |
| `rv32imafc/ilp32f` | `rv32imafc` | `ilp32f` | 32-bit with single FPU only |
| `rv64imac/lp64` | `rv64imac` | `lp64` | 64-bit soft-float |
| `rv32imac/ilp32` | `rv32imac` | `ilp32` | 32-bit soft-float |

Clang selects the correct variant automatically via `multilib.yaml`
based on the `-march` and `-mabi` flags.

**Stage 1: LLVM toolchain (no runtimes)**

compiler-rt cannot be built in-tree (`LLVM_ENABLE_RUNTIMES="compiler-rt"`)
for bare-metal cross-compilation because the runtimes build tries to
cross-compile test programs (`test_target_arch`) which fail without a
C library. It is built standalone in Stage 2 instead.

Key options:
- `LLVM_DEFAULT_TARGET_TRIPLE=riscv64-unknown-elf` -- required when
  building only the RISC-V backend (fixes LLDB `cmake_path` errors)
- No `LLVM_ENABLE_RUNTIMES` -- compiler-rt is built standalone in Stage 2

**Stage 2: compiler-rt builtins (one build per multilib variant)**

Built as a standalone CMake project against `compiler-rt/` with
`CMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY` to skip link-time tests.
Each variant's `libclang_rt.builtins.a` is installed into its multilib
`lib/` directory (clang finds it via `getLibraryPaths()`).

**Stage 3: Newlib + libgloss (per multilib variant)**

Newlib is cloned from [sourceware](https://sourceware.org/git/newlib-cygwin.git)
and built for each multilib variant. Since newlib always installs to
`<prefix>/<target>/`, each variant uses a temp prefix and the headers
and libraries are copied to the multilib directory.

**Stage 4: C++ runtimes (per multilib variant)**

Built as a standalone CMake invocation against `runtimes/`. Each
variant's `CMAKE_INSTALL_PREFIX` points to its multilib directory.

Key gotchas discovered during bring-up:
- **Do NOT use `CMAKE_SYSROOT`** -- the runtimes build system overrides
  the compiler path when sysroot is set. Pass `--sysroot` via
  `CMAKE_C_FLAGS` / `CMAKE_CXX_FLAGS` / `CMAKE_ASM_FLAGS` instead.
- **`LIBUNWIND_IS_BAREMETAL=ON` is required** -- without it, libunwind
  tries to include `dlfcn.h` which does not exist in newlib.
- **`LLVM_INCLUDE_TESTS=OFF` is required** -- avoids a dependency on
  system Clang/LLVM packages for test infrastructure.

**Installed layout:**

```
~/opt/riscv-llvm/
├── bin/                                     # clang, clang++, lld, lldb, ...
├── include/                                 # Clang/LLVM headers
├── lib/
│   └── clang-runtimes/
│       ├── multilib.yaml                    # multilib configuration
│       ├── rv64imafdc/lp64d/                # rv64gc + lp64d variant
│       │   ├── include/                     # newlib + libc++ headers
│       │   └── lib/                         # all libraries
│       │       ├── libclang_rt.builtins.a
│       │       ├── libc.a, libm.a, libnosys.a
│       │       ├── libc++.a, libc++abi.a, libunwind.a
│       ├── rv32imafdc/ilp32d/               # (same structure)
│       ├── rv64imafc/lp64f/
│       ├── rv32imafc/ilp32f/
│       ├── rv64imac/lp64/
│       └── rv32imac/ilp32/
└── share/
```

**Usage:**

```bash
# RV64 with double FPU (auto-selects rv64imafdc/lp64d)
clang --target=riscv64-unknown-elf -march=rv64gc -mabi=lp64d -O2 test.c -lnosys -o test

# RV32 single-precision FPU only (auto-selects rv32imafc/ilp32f)
clang --target=riscv32-unknown-elf -march=rv32imafc -mabi=ilp32f -O2 test.c -lnosys -o test

# RV64 soft-float (auto-selects rv64imac/lp64)
clang --target=riscv64-unknown-elf -march=rv64imac -mabi=lp64 -O2 test.c -lnosys -o test

# C++ with libc++
clang++ --target=riscv64-unknown-elf -march=rv64gc -mabi=lp64d -O2 -stdlib=libc++ test.cpp -lnosys -o test

# Check which variant is selected
clang --target=riscv64-unknown-elf -march=rv64gc -print-multi-directory

# Override _write in your code to redirect printf to UART;
# your definition takes priority over the stub in libnosys.a
```

## Build Iterations and Issues

### Iteration 1: TableGen errors
**Error**: `could not find field for operand 'imm3'`

**Root cause**: The `THRVMInstEWBinMVI` class declared `bits<3> imm3` but never assigned it to any `Inst{...}` bits.

**Fix**:
- Changed the MVI format to have `md, ms1, imm3` (3 operands) instead of `md, ms2, ms1, imm3` (4 operands)
- Assigned `imm3` to bits [22:20] (the same position as `ms2` in the MM variant)
- Updated all MVI instruction definitions and multiclasses accordingly

### Iteration 2: AsmParser errors
**Error**: `no member named 'isTHRVMUimm10' in 'RISCVOperand'`

**Root cause**: Custom `AsmOperandClass` definitions generate `is<Name>()` predicate methods that must exist in `RISCVAsmParser.cpp`. The custom classes `THRVMUimm10AsmOperand` and `THRVMUimm3AsmOperand` had unique names.

**Fix**: Replaced custom operand classes with standard `RISCVUImmOp<N>`:
```tablegen
// BEFORE (broken):
def THRVMUimm10AsmOperand : AsmOperandClass { ... }
def thrvmuimm10 : RISCVOp { ... }

// AFTER (working):
def thrvmuimm10 : RISCVUImmOp<10>;
def thrvmuimm3  : RISCVUImmOp<3>;
```

### Iteration 3: Decoding conflicts
**Error**: `Decoding conflict encountered` - MLAE_E8 vs MLATE_E8

**Root cause**: Load/store instructions used 3-bit GPR fields (only encoding lower 3 bits of 5-bit GPR index), and the `ctrl` field differentiation was lost when restructuring.

**Fix**: Redesigned load/store encoding:
- Changed from 3-bit ctrl field to 1-bit `stride` flag (bit 25)
- Moved GPRs to standard R-type positions: rs1=[19:15], rs2=[24:20] (full 5-bit)
- This gives unique encodings: element-stride has bit 25=0, tile-stride has bit 25=1

### Iteration 4: GPR decode issues
**Error**: Configuration instructions decoded `a0` as `sp` in disassembly

**Root cause**: Config instruction `rd` field only used bits [9:7] (3 bits) for a 5-bit GPR. GPR index 10 (a0) was being truncated to 2 (sp).

**Fix**: Changed config instructions to use bits [11:7] for the full 5-bit GPR rd.

### Iteration 5: Register name issues
**Error**: `acc0` displayed as `v0` in disassembly

**Root cause**: Register decoder used arithmetic offset `RISCV::THRVM_TR0 + RegNo` which didn't correctly map to ACC registers due to enum ordering.

**Fix**: Used explicit lookup table:
```cpp
static constexpr MCPhysReg THRVMRegs[] = {
    RISCV::THRVM_TR0,  RISCV::THRVM_TR1,  RISCV::THRVM_TR2,
    RISCV::THRVM_TR3,  RISCV::THRVM_ACC0, RISCV::THRVM_ACC1,
    RISCV::THRVM_ACC2, RISCV::THRVM_ACC3};
```

### Iteration 6: Unused template parameter warnings
**Warning**: Multiple `unused template argument` warnings for `dsize`, `ssize` parameters

**Fix**: Removed unused parameters from class definitions and updated all instantiation sites.

### Iteration 7: Encoding rework (full spec alignment)
**Issue**: Systematic encoding verification against the RVM 0.6 spec revealed discrepancies across all 5 instruction categories.

**Discrepancies found and fixed**:
1. **CONFIG**: `th.mrelease` had wrong func4 (was 0001, fixed to 0000); immediate config instructions needed `bit[25]=0` enforcement
2. **LOAD/STORE**: Tile-stride mnemonic concatenation produced double-e (e.g. `th.mlatee8` instead of `th.mlate8`); fix: shorten base mnemonic from `"mlate"` to `"mlat"` so multiclass appending `"e8"` yields correct `"th.mlate8"`
3. **MATMUL**: Bypass variants needed func4=0010 (was 0001); signed/unsigned ctrl field values corrected
4. **MISC**: `th.mzero` needed func4=0000 with imm3=000; multi-register zero variants needed correct imm3 encoding (001/011/111); `th.mdupw.m.x` needed bit[25]=0 with rs1=00000
5. **ELEMENT-WISE**: Conversion source/dest size fields corrected; n4clip ctrl field values aligned with spec

**Additional issue**: A pre-commit linter reverted the tile-stride mnemonic fix twice by modifying the .td file. Required re-reading exact file content and re-applying edits with correct context after each linter pass.

**Verification**: 24/24 programmatic bit-field checks passed; 110 instructions assembled with 0 errors in comprehensive smoke test.

## Final Build Status

```
LLVM build: 0 errors (2 pre-existing unused function warnings in disassembler)
Tests: All 16 XTHeadMatrix tests pass (11 Clang CodeGen + 1 Sema + 4 LLVM)
Instruction count: 227 total (119 standalone defs + 108 multiclass expansions)
Encoding verification: 227/227 verified against RVM 0.6 spec, 0 conflicts
Full toolchain: 4-stage multilib build (LLVM tools + compiler-rt + newlib/libgloss + C++ runtimes) for 6 variants (rv64gc/lp64d, rv32gc/ilp32d, rv64imafc/lp64f, rv32imafc/ilp32f, rv64imac/lp64, rv32imac/ilp32)
```
