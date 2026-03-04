# Independent Verification and Bug Fixes (2026-03-04)

## Overview

An independent verification audit was performed against the RVM 0.6 specification
to validate the correctness and alignment of the XTHeadMatrix implementation across
all layers: instruction encodings, register constraints, ISel, pseudo expansion,
builtins, spec-API codegen, and the C header API.

**2 correctness bugs were found and fixed**, **2 spec errata were documented**,
and **3 coverage gaps were eliminated**.

---

## Bug Fix 1: Conversion Pseudo Register Classes (42 pseudos)

**Severity**: HIGH — could produce invalid machine code in ManagedRA mode.

**Spec reference** (`type_convert.adoc` line 1):
> "The input and output matrices of type-convert instructions are both accumulation registers."

**Problem**: All 42 conversion pseudo-instructions (`PTH_MFCVT*_V`, `PTH_MUFCVT*_V`,
`PTH_MSFCVT*_V`, `PTH_MFUCVT*_V`, `PTH_MFSCVT*_V`, `PTH_MUCVT*_V`, `PTH_MSCVT*_V`)
used `THRVMMR` (all 8 registers including tr0-tr3) for both input and output register
classes. The register allocator could assign tile registers (tr0-tr3) to conversion
instructions, producing machine code the hardware would reject.

**Affected pseudo-instructions** (42 total):
- 26 float-point conversion pseudos (FP8↔FP16, FP16↔FP32, BF16↔FP32, FP8↔FP32, FP32↔FP64, TF32↔FP32)
- 12 float-int conversion pseudos (UINT8↔FP16, SINT8↔FP16, INT32↔FP32)
- 4 packed conversion pseudos (INT8↔INT4 pack/unpack)

**Fix**: Changed `THRVMMR` to `THRVMACC` in all 42 definitions.

**File**: `llvm/lib/Target/RISCV/RISCVInstrInfoXTHeadMatrix.td` (lines 1167-1212)

**Note**: The DirectReg (non-pseudo) `TH_*` hardware instructions correctly use `THRVMMR`
since register selection is done via ImmArg index, not register allocation. Only the
ManagedRA `PTH_*_V` pseudos needed this fix.

---

## Bug Fix 2: Spec-API Matmul Operand Swap

**Severity**: HIGH — produced wrong computation results for non-commutative integer
matmul variants (`mmaccus`, `mmaccsu`) in the ManagedRA/spec-API path.

**Spec reference** (`matmul.adoc`): `md = md + ms1 * ms2`

**Encoding reference** (`inst32_format.adoc`):
- `mmaccus`: ms1=unsigned, ms2=signed (size_sup bit[24]=0, bit[23]=1)
- `mmaccsu`: ms1=signed, ms2=unsigned

**Problem**: The `_internal` intrinsic signature is `(acc, ms2, ms1)`. The spec-API
codegen passed user operands `(acc, a, b)` directly as `{Ops[0], Ops[1], Ops[2]}`,
placing A into ms2 and B into ms1. But per the DirectReg convention (which is correct),
A maps to ms1 and B maps to ms2.

For `mmaccus` (ms1=unsigned, ms2=signed):
- **Before (wrong)**: ms2=A(unsigned), ms1=B(signed) → hardware computes unsigned(B) × signed(A)
- **After (correct)**: ms2=B(signed), ms1=A(unsigned) → hardware computes unsigned(A) × signed(B)

**Fix**: Swapped `Ops[1]` and `Ops[2]` in all matmul `CreateCall` invocations:
```cpp
// Before (wrong):
return Builder.CreateCall(F, {Ops[0], Ops[1], Ops[2]});
// After (correct):
return Builder.CreateCall(F, {Ops[0], Ops[2], Ops[1]});
```

This affects all matmul spec-API variants (not just us/su), keeping A→ms1 and B→ms2
consistent with the DirectReg convention. For commutative variants (ss, uu, FP) the
swap is harmless but semantically correct.

**File**: `clang/lib/CodeGen/TargetBuiltins/RISCV.cpp`

---

## Coverage Gap 1: B-tile Spec-API Load (NEW)

**Problem**: The spec-API only provided `mld_spec_*` (A-role, sets M/K via `mlae`)
and `mld_acc_spec_*` (C-role, sets M/N via `mlce`). There was no B-tile load variant,
meaning users could not correctly load B-tiles with K×N dimensions in ManagedRA mode.

**Fix**: Added `mld_b_spec_{i8..f64}` builtins (11 type variants) that:
- Set K and N dimensions (`SetK(Ops[2]); SetN(Ops[3])`)
- Emit `mlbe_internal` (B-role element-stride load)

**Files**:
- `clang/include/clang/Basic/BuiltinsRISCVXTHeadMatrix.td` — 11 new builtin definitions
- `clang/lib/CodeGen/TargetBuiltins/RISCV.cpp` — `SpecAPILoadB()` handler
- `clang/lib/Headers/thead_matrix.h` — `__riscv_th_mld_b_*()` C wrappers
- `clang/test/CodeGen/RISCV/xtheadmatrix-spec-api.c` — test updated to use B-tile load

---

## Coverage Gap 2: FP/Unsigned Type Variants (NEW)

**Problem**: Spec-API load/store/zero only covered signed integer types (i8/i16/i32/i64).
No floating-point (f16/f32/f64) or unsigned (u8/u16/u32/u64) variants existed.

**Fix**: Extended all spec-API operations from 4 types to 11:

| Operation | Before | After |
|-----------|--------|-------|
| Load A-tile | i8/i16/i32/i64 | + u8/u16/u32/u64/f16/f32/f64 |
| Load B-tile | (none) | i8/i16/i32/i64/u8/u16/u32/u64/f16/f32/f64 |
| Load Acc | i8/i16/i32/i64 | + u8/u16/u32/u64/f16/f32/f64 |
| Store | i8/i16/i32/i64 | + u8/u16/u32/u64/f16/f32/f64 |
| Zero | i32/f32 | + i8/i16/i64/u8/u16/u32/u64/f16/f64 |

All type variants at the same EEW map to the same hardware instruction (e.g., both
`mld_spec_i32` and `mld_spec_f32` emit `mlae_internal32`).

**Files**: Same as Coverage Gap 1 (builtins .td, codegen, header).

---

## Coverage Gap 3: Missing Matmul Variants (NEW)

**Problem**: Spec-API matmul only covered INT8→INT32 (4 sign variants) and native FP
(h, s). Missing: INT16→INT64, partial, bypass, widening FP, FP64.

**Fix**: Added all remaining matmul variants to the spec-API:

| Category | Variants Added | Count |
|----------|---------------|-------|
| INT16→INT64 | ss, uu, us, su | 4 |
| Partial INT8→INT32 | ss, uu, us, su | 4 |
| Bypass INT | ss, uu | 2 |
| FP64 native | d | 1 |
| Widening FP (typed) | s_h, d_s | 2 |
| Widening FP (opaque) | h_e4, h_e5, bf16_e4, bf16_e5, s_bf16, s_e4, s_e5, s_tf32 | 8 |
| **Total** | | **21** |

For opaque-source widening matmul (FP8/BF16/TF32), only the accumulator is an SSA
value. The builtin signature is `(acc, m, k, n) -> acc` since source tile types have
no standard C representation.

**Files**: Same as above.

---

## Spec Errata Documented (Not Bugs)

1. **Matmul uop in instruction_list.adoc**: Table shows `uop=01` for all matmul
   instructions, but the format description text (`inst32_format.adoc`) says `uop=10`.
   LLVM correctly uses `10`.

2. **mfmin.s / mfmin.h name swap in instruction_list.adoc**: `mfmin.s` is listed with
   s_size/d_size=01 (half-precision) and `mfmin.h` with 10 (single-precision). LLVM
   correctly uses `.h`=01, `.s`=10 consistent with all other FP operations.

---

## Verification Results (All Passing)

| Area | Status | Details |
|------|--------|---------|
| Instruction encodings | PASS | 227/227 verified against spec |
| Base opcode | PASS | custom-1 (0b0101011) |
| Field positions | PASS | All bit fields at correct positions |
| Config encodings | PASS | func4 values match spec |
| Load/store uop | PASS | uop=01 |
| Matmul uop | PASS | uop=10 (matches spec text, not table) |
| Misc uop | PASS | uop=11 |
| EW func3 | PASS | func3=001 for all element-wise |
| Register classes (loads) | PASS | MLA/MLB→THRVMTR, MLC→THRVMACC, MLM→THRVMMR |
| Register classes (stores) | PASS | MSA/MSB→THRVMTR, MSC→THRVMACC, MSM→THRVMMR |
| Register classes (matmul) | PASS | md/acc→THRVMACC, ms1/ms2→THRVMTR |
| Register classes (EW arith) | PASS | All→THRVMACC |
| Register classes (conversions) | PASS | All→THRVMACC (was THRVMMR, now fixed) |
| Register classes (misc) | PASS | All→THRVMMR |
| ISel completeness | PASS | All 220 _internal intrinsics handled |
| ISel operand ordering | PASS | Correct for all 15 categories |
| Tied operand constraints | PASS | matmul, EW arith, N4Clip, mmov.m.x, mdup.m.x |
| Pseudo expansion table | PASS | 220 entries, all map to correct TH_* |
| Spill/reload expansion | PASS | Correct MSME_E8/MLME_E8 with liveness |
| DirectReg C API | PASS | Dimension configs correct |
| Spec-API config calls | PASS | load-tile=M/K, load-b=K/N, load-acc=M/N |
| Spec-API matmul operands | PASS | A→ms1, B→ms2 (now fixed) |
| Assembly mnemonics | PASS | 227 mnemonics match spec |
| CSR names | PASS | All 13 verified |

---

## Second Independent Verification (2026-03-04, Claude Opus 4.6)

A second independent verification was performed directly against the RVM 0.6 spec
source files (`spec/instruction_list.adoc`, `spec/inst32_format.adoc`,
`spec/matmul.adoc`, `spec/type_convert.adoc`, `doc/intrinsic/rvm-intrinsic-api.adoc`).

### Methodology

Six parallel verification passes were executed:
1. Instruction encodings: every field (func4, uop, ctrl, s_size, d_size) cross-referenced bit-by-bit
2. ISel tables: all 227 DirectReg + 220 ManagedRA entries checked for category, opcode, and operand order
3. Pseudo expansion: all 220 THMatrixPseudoTable entries verified for SkipTiedInput correctness
4. Builtin codegen: spec-API matmul operand swap, load/store intrinsic selection, CSR calls, type signatures
5. CSR definitions: all 13 addresses verified against spec
6. Intrinsic definitions: all 447 signatures (227 DirectReg + 220 _internal) checked

### Results: No New Bugs Found

All six areas passed verification. The two bugs found and fixed in the first audit
(conversion register classes, matmul operand swap) are confirmed correctly fixed.

| Area | Entries Checked | Status |
|------|----------------|--------|
| Instruction encodings | 227 instructions, every bit field | PASS |
| DirectReg ISel table | 227 entries (14 categories) | PASS |
| ManagedRA ISel table | 220 entries (13 categories) | PASS |
| Pseudo expansion table | 220 entries (SkipTiedInput flags) | PASS |
| Pseudo register classes | 42 conversion, 27 matmul, 60 EW | PASS |
| Spec-API matmul operand swap | `{Ops[0], Ops[2], Ops[1]}` verified | PASS |
| Spec-API load/store intrinsic selection | mlae/mlbe/mlce/msce verified | PASS |
| Spec-API CSR dimension calls | SetM/SetK/SetN per role | PASS |
| CSR addresses | 13 CSRs at 0x802-0x80a, 0xcc0-0xcc3 | PASS |
| Intrinsic signatures | 447 definitions, operand types/counts | PASS |
| Assembly/disassembly tests | 227 instructions + 13 CSRs | PASS |
| DirectReg typed builtin codegen | IsTypedMatrixBuiltin path | PASS |

### Key Findings Confirmed

- **mfmin name errata**: `mfmin.h`→s_size=01, `mfmin.s`→s_size=10 in implementation is correct;
  spec `instruction_list.adoc` has names swapped (encodings are right, labels wrong)
- **Matmul uop errata**: Implementation uses uop=10 (correct per `inst32_format.adoc`);
  `instruction_list.adoc` erroneously shows uop=01 for all matmul entries
- **EW .mm pseudos**: Correctly categorized as `THMI_MulAcc` (not `THMI_Binary`), matching
  their tied-accumulator constraint `$src1 = $dst` in the .td file
- **SkipTiedInput flags**: 95 entries `true` (tied ops), 125 entries `false` (non-tied). Each
  flag verified against the corresponding pseudo's `Constraints` declaration in the .td file
- **Non-commutative matmul**: `mmaccus_us_w_b(acc, a=uint8, b=int8)` correctly swaps to
  `_internal(acc, ms2=b=signed, ms1=a=unsigned)` — ms1=unsigned, ms2=signed per spec encoding

---

## Limitations and Differences from Spec

### Implementation Limitations

1. **64-bit instruction format not supported**: `spec/inst64_format.adoc` defines extended
   instruction formats. These are not implemented (deferred Phase 7). The 32-bit instruction
   set covers all 227 defined instructions.

2. **Matrix types cannot cross function boundaries**: `target("riscv.matrix")` / `__rvm_*_t`
   values have no ABI-level calling convention. They cannot be passed as function parameters
   or returned. Matrix operations must stay within a single function scope (or be inlined).

3. **DirectReg typed builtins use PoisonValue**: The `IsTypedMatrixBuiltin` codegen path
   filters out `TargetExtType` (matrix) arguments and calls the void intrinsic with only
   register index args. The returned matrix value is `PoisonValue`. This is by design for
   the DirectReg model where data flows through physical registers, not SSA values. Users
   must independently manage register loading via separate load calls.

4. **No auto-vectorization**: C matrix loops (e.g., `for (i) for (j) C[i][j] += A[i][k]*B[k][j]`)
   are not automatically lowered to matrix instructions. Users must explicitly use builtins,
   the `<thead_matrix.h>` API, or inline assembly.

5. **All 8 matrix registers reserved in DirectReg mode**: When any DirectReg intrinsic is
   used, `RISCVRegisterInfo::getReservedRegs()` reserves all 8 matrix registers unconditionally.
   In ManagedRA mode, only registers actively in use are allocated.

6. **Limited register file**: Only 4 tile + 4 accumulator registers. High register pressure
   for complex kernels (e.g., multi-tile GEMM with multiple accumulators).

7. **Spill granularity**: Spill/reload uses whole-register load/store (`TH_MSME_E8`/`TH_MLME_E8`).
   No partial-register spill optimization. Spill cost is proportional to register size (8192 bits).

8. **No `-O0` register allocator support for ManagedRA**: The `RISCVLowerMatrixType` pass
   provides basic `-O0` support by lowering `target("riscv.matrix")` to dummy allocas,
   but this is limited. Full optimization (`-O1` and above) is recommended.

### Differences from Spec Intrinsic API (rvm-intrinsic-api.adoc v0.2)

1. **No C++ overloading**: The spec envisions C++ overloaded functions (e.g., single
   `__riscv_th_mld` overloaded by pointer type to return `mint8_t`/`mint16_t`/etc.).
   The implementation uses separate C functions per type (`__riscv_th_mld_i8`,
   `__riscv_th_mld_i16`, etc.). This is necessary because C does not support overloading.

2. **Role-specific loads in DirectReg API**: The spec uses a unified `__riscv_th_mld(base,
   stride, row, col)` for all load types. The DirectReg implementation uses role-specific
   functions (`mld_a`, `mld_b`, `mld_c`, `mld_at`, `mld_bt`, `mld_ct`, `mld_whole`) because
   each maps to a different hardware instruction (`mlae`, `mlbe`, `mlce`, etc.). The Spec-API
   provides `__riscv_th_mld_*` (A-tile), `__riscv_th_mld_b_*` (B-tile), and
   `__riscv_th_mld_acc_*` (C-tile) as a middle ground.

3. **Stream load/store not implemented**: The spec mentions `msld<b/h/w/d>` and
   `msst<b/h/w/d>` (stream matrix load/store). These instructions are NOT in
   `instruction_list.adoc` and appear to be a planned future extension. Not a gap — the
   instructions simply do not exist in RVM 0.6.

4. **Matrix-scalar EW operations (`.mx`) not implemented**: The spec intrinsic API lists
   `__riscv_th_madd_mx`, `__riscv_th_msub_mx`, etc. (matrix-scalar arithmetic). These
   instructions are NOT in `instruction_list.adoc`. Only `.mm` (matrix-matrix) and `.mv.i`
   (matrix-vector-by-index) variants exist as hardware instructions.

5. **64-bit integer EW arithmetic not implemented**: The spec lists `mint64_t` variants
   for `madd`, `msub`, etc. However, `instruction_list.adoc` only defines `.w` (32-bit)
   integer element-wise instructions (s_size=10, d_size=10). No `.d` (64-bit) integer EW
   instructions exist in RVM 0.6.

6. **Matmul instruction naming convention differs**: The spec intrinsic API uses `mmaqa`
   (matrix multiply-accumulate quad-widen A-type) naming, while the hardware instructions
   use `mmacc` (matrix multiply-accumulate). Both refer to the same operations.
   Implementation uses `mmacc` for low-level builtins (matching hardware) and `mmaqa`
   for high-level API (matching spec intrinsic naming).

7. **Config API matches spec**: `__riscv_th_msetmrow_m`, `__riscv_th_msetmrow_n`,
   `__riscv_th_msetmcol_e8/16/32/64` are all present in `<thead_matrix.h>`, matching
   the spec intrinsic API names. They map to `msettilem`/`msettilen`/`msettilek`
   hardware instructions.

8. **No `mmov.mv.x` / `mmov.mv.i` as separate functions**: The spec mentions
   `__riscv_th_mmov_mv(src, index)` for extracting a row vector from a matrix. The
   implementation does not have a dedicated instruction for this in `instruction_list.adoc`.
   This functionality can be achieved via `mrslidedown`/`mrbca` instructions.

### What IS Aligned with the Spec

- All 227 hardware instructions fully implemented with correct encodings
- All 13 CSRs at correct addresses with `th.` prefix
- Register encoding: tr0-tr3 = 000-011, acc0-acc3 = 100-111
- Matmul formula: `md = md + ms1 * ms2` correctly implemented
- Register class constraints: matmul (ms1/ms2=tile, md=acc), EW (all acc), conversions (all acc)
- Configuration instructions: `mrelease`, `msettilem/n/k` with both immediate and register forms
- All data types: FP8 (E4M3/E5M2), FP16, BF16, TF32, FP32, FP64, INT8, INT16, INT32, INT64
- Assembly syntax: all mnemonics match spec with `th.` prefix

---

## Files Changed

| File | Change Type | Description |
|------|-------------|-------------|
| `llvm/lib/Target/RISCV/RISCVInstrInfoXTHeadMatrix.td` | Bug fix | 42 conversion pseudos: THRVMMR → THRVMACC |
| `clang/lib/CodeGen/TargetBuiltins/RISCV.cpp` | Bug fix + New | Matmul operand swap; all new spec-API codegen |
| `clang/include/clang/Basic/BuiltinsRISCVXTHeadMatrix.td` | New | B-tile loads, FP/unsigned types, all matmul variants |
| `clang/lib/Headers/thead_matrix.h` | New | B-tile load wrappers, all type variants, all matmul wrappers |
| `clang/test/CodeGen/RISCV/xtheadmatrix-spec-api.c` | Updated | Uses B-tile load; updated CHECK lines |
