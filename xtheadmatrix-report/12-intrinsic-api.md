# Higher-Level Intrinsic API (`<thead_matrix.h>`)

## Overview

The `<thead_matrix.h>` header provides 450+ C API functions/macros. Function
naming follows **RVM 0.6 assembly mnemonics** (e.g., `mmacc`, `mfmacc`,
`mlae`/`mlbe`/`mlce`) rather than the older intrinsic API spec
(`rvm-intrinsic-api.adoc` v0.2) naming (e.g., `mmaqa`, `fmmacc`, `mld`).
All operations use the Spec-API (ManagedRA) programming model — no manual
register index management needed.

## Types

### Matrix types (22 native built-in types)

| Single | Pair | C alias (single) | C alias (pair) |
|--------|------|-------------------|----------------|
| `__rvm_int8_t` | `__rvm_int8x2_t` | `mint8_t` | `mint8x2_t` |
| `__rvm_int16_t` | `__rvm_int16x2_t` | `mint16_t` | `mint16x2_t` |
| `__rvm_int32_t` | `__rvm_int32x2_t` | `mint32_t` | `mint32x2_t` |
| `__rvm_int64_t` | `__rvm_int64x2_t` | `mint64_t` | `mint64x2_t` |
| `__rvm_uint8_t` | `__rvm_uint8x2_t` | `muint8_t` | `muint8x2_t` |
| `__rvm_uint16_t` | `__rvm_uint16x2_t` | `muint16_t` | `muint16x2_t` |
| `__rvm_uint32_t` | `__rvm_uint32x2_t` | `muint32_t` | `muint32x2_t` |
| `__rvm_uint64_t` | `__rvm_uint64x2_t` | `muint64_t` | `muint64x2_t` |
| `__rvm_float16_t` | `__rvm_float16x2_t` | `mfloat16_t` | `mfloat16x2_t` |
| `__rvm_float32_t` | `__rvm_float32x2_t` | `mfloat32_t` | `mfloat32x2_t` |
| `__rvm_float64_t` | `__rvm_float64x2_t` | `mfloat64_t` | `mfloat64x2_t` |

### Dimension types

- `mrow_t` — row count (alias for `unsigned long`)
- `mcol_t` — column count (alias for `unsigned long`)

## API Categories

### Configuration (7 functions)

```c
mrow_t __riscv_th_msetmrow_m(mrow_t m);   // th.msettilem
mcol_t __riscv_th_msetmcol_e8(mcol_t k);  // th.msettilek (+ e16/e32/e64)
void   __riscv_th_mrelease(void);          // th.mrelease
```

### Loads (33 functions)

```c
mint32_t   __riscv_th_mld_a_i32(const int32_t *base, long stride, mrow_t m, mcol_t k);
mint32_t   __riscv_th_mld_b_i32(const int32_t *base, long stride, mcol_t k, mcol_t n);
mint32_t   __riscv_th_mld_acc_i32(const int32_t *base, long stride, mrow_t m, mcol_t n);
```
Each in 11 type variants (i8/i16/i32/i64/u8/u16/u32/u64/f16/f32/f64).
CSR configuration (msettilem/k/n) is emitted automatically.

### Stores (11 functions)

```c
void __riscv_th_mst_i32(int32_t *base, long stride, mint32_t val, mrow_t m, mcol_t n);
```

### Matrix Multiply (27 functions)

Function names follow RVM 0.6 assembly mnemonics (`mmacc`, `mmaccu`, `mfmacc`, etc.):

```c
// INT matmul (typed sources) — names match assembly: mmacc.w.b, mmaccu.w.b, mmaccus.w.b, mmaccsu.w.b
mint32_t   __riscv_th_mmacc_w_b(mint32_t acc, mint8_t a, mint8_t b, mrow_t m, mcol_t k, mcol_t n);
// FP matmul (native precision) — names match assembly: mfmacc.s, mfmacc.h, mfmacc.d
mfloat32_t __riscv_th_mfmacc_s(mfloat32_t acc, mfloat32_t a, mfloat32_t b, mrow_t m, mcol_t k, mcol_t n);
// FP matmul (widening, opaque FP8/BF16/TF32 sources use mint32_t for A/B)
mfloat16_t __riscv_th_mfmacc_h_e4(mfloat16_t acc, mint32_t a, mint32_t b, mrow_t m, mcol_t k, mcol_t n);
mfloat32_t __riscv_th_mfmacc_s_tf32(mfloat32_t acc, mint32_t a, mint32_t b, mrow_t m, mcol_t k, mcol_t n);
```
Shorthand aliases: `__riscv_th_mmacc` → `__riscv_th_mmacc_w_b`.
Backward-compat aliases from old naming (`mmaq_ss_w_b` → `mmacc_w_b`, `mfmaqa_s` → `mfmacc_s`, etc.) are provided.

### Zero (22 functions: 11 single + 11 x2)

```c
mint32_t     __riscv_th_mzeros_i32(mrow_t m, mcol_t n);      // single register
mint32x2_t   __riscv_th_mzeros_i32x2(mrow_t m, mcol_t n);    // register pair (x2)
```
x2 variants zero two registers and return a pair type. Aliases: `__riscv_th_mzero_*` / `__riscv_th_mzero_*x2`.

### Element-Wise Integer (22 functions)

```c
mint32_t __riscv_th_madd_w_mm(mint32_t acc, mint32_t s2, mint32_t s1);        // .mm
mint32_t __riscv_th_madd_w_mv_i(mint32_t acc, mint32_t s2, mint32_t s1, unsigned int imm);  // .mv.i
```
11 operations: madd, msub, mmul, mmulh, mmax, mumax, mmin, mumin, msrl, msll, msra.

### Element-Wise FP (30 functions)

```c
mfloat32_t __riscv_th_mfadd_s_mm(mfloat32_t acc, mfloat32_t s2, mfloat32_t s1);
mfloat32_t __riscv_th_mfadd_s_mv_i(mfloat32_t acc, mfloat32_t s2, mfloat32_t s1, unsigned int imm);
```
5 operations × 3 precisions (h/s/d) × .mm/.mv.i.

### Format Conversions (26 functions)

```c
mfloat32_t __riscv_th_mfcvtl_s_h(mfloat16_t src);  // FP16→FP32 lower
mint32_t   __riscv_th_mfcvtl_h_e4(mint32_t src);    // FP8→FP16 (opaque)
```

### Float-Int Conversions (12 functions)

```c
mfloat32_t __riscv_th_msfcvt_s_w(mint32_t src);    // INT32→FP32 signed
muint32_t  __riscv_th_mfucvt_w_s(mfloat32_t src);  // FP32→UINT32
```

### N4Clip (8 functions)

```c
mint32_t __riscv_th_mn4clipl_w_mm(mint32_t acc, mint32_t s2, mint32_t s1);
mint32_t __riscv_th_mn4clipl_w_mv_i(mint32_t acc, mint32_t s2, mint32_t s1, unsigned int imm);
```

### Data Movement

```c
mint32_t      __riscv_th_mmov_mm(mint32_t src);
unsigned long __riscv_th_mmovw_x_m(mint32_t src, unsigned long idx);
mint32_t      __riscv_th_mmovw_m_x(mint32_t dst, unsigned long data, unsigned long idx);
mint32_t      __riscv_th_mdupw_m_x(mint32_t dst, unsigned long data);
mint32_t      __riscv_th_mpack(mint32_t s2, mint32_t s1);
mint32_t      __riscv_th_mrslidedown(mint32_t src, unsigned int imm);
mint32_t      __riscv_th_mrbca(mint32_t src, unsigned int imm);
mint32_t      __riscv_th_mcbca_w(mint32_t src, unsigned int imm);
```

### Tuple Operations (mget/mset — 22 functions)

Extract or insert a single register from/to an x2 (register-pair) type.
Backed by 22 dedicated Clang builtins (`mget_spec_*` / `mset_spec_*`).
At the IR level, these emit `extractvalue`/`insertvalue` with a `select`
on the index. At `-O2`, constant indices are folded to direct struct access.

```c
mfloat16_t  __riscv_th_mget_f16(mfloat16x2_t pair, size_t idx);  // extract
mfloat16x2_t __riscv_th_mset_f16(mfloat16x2_t pair, size_t idx, mfloat16_t val); // insert
```
11 type variants each (i8/i16/i32/i64/u8/u16/u32/u64/f16/f32/f64).

### x2 Matrix Multiply Overloads (7 functions)

Spec-aligned x2 wrapper functions for matmul variants that use register-pair
types. These are software-level pair abstractions — the hardware instruction
uses a single 3-bit register. Each wrapper extracts component 0, calls the
single-register matmul builtin, then inserts the result back.

```c
// FP16: x2 B operand
mfloat16_t __riscv_th_mfmacc_h_x2(mfloat16_t c, mfloat16_t a, mfloat16x2_t b,
                                    mrow_t m, mcol_t k, mcol_t n);
// FP64: x2 dest
mfloat64x2_t __riscv_th_mfmacc_d_x2(mfloat64x2_t c, mfloat64_t a, mfloat64_t b,
                                      mrow_t m, mcol_t k, mcol_t n);
// FP64 widening: x2 dest
mfloat64x2_t __riscv_th_mfmacc_d_s_x2(mfloat64x2_t c, mfloat32_t a, mfloat32_t b,
                                        mrow_t m, mcol_t k, mcol_t n);
// INT16→INT64: x2 dest (4 sign variants: signed/unsigned/us/su)
mint64x2_t __riscv_th_mmacc_d_h_x2(mint64x2_t c, mint16_t a, mint16_t b,
                                     mrow_t m, mcol_t k, mcol_t n);
```

### Utility

```c
mint32_t __riscv_th_mundef_i32(void);    // undefined value (PoisonValue)
```

## Behavioral Notes

### EW Operations and CSR State

Element-wise operations (`__riscv_th_madd_w_mm`, `__riscv_th_mfadd_s_mm`, etc.) do
NOT configure mtilem/mtilen CSRs — they rely on the CSRs being already set by prior
load/matmul operations. In a typical pipeline (load→matmul→EW), this is correct.
For standalone EW use, manually call `__riscv_th_msetmrow_m(M)` and
`__riscv_th_msetmrow_n(N)` first.

The `acc` parameter in EW functions (`mint32_t __riscv_th_madd_w_mm(mint32_t acc, ...)`)
is for register allocation tying — the output physical register is constrained to be the
same as `acc`. The `acc` value is NOT used in the computation; the hardware computes
`md = ms2 op ms1` without reading old `md`.

### Matmul Dimension Parameter Order

Matmul functions use `(acc, a, b, M, K, N)` parameter order, where M=mtilem, K=mtilek,
N=mtilen. The spec intrinsic API uses `(dest, src1, src2, M, N, K)` — K and N are
swapped relative to the implementation. The codegen correctly maps each to its CSR
regardless of C parameter position.

## Differences from Spec Intrinsic API

The implementation follows **RVM 0.6 assembly mnemonics** for function naming (e.g.,
`mfmacc`, `mmacc`) rather than the spec intrinsic API document's older names (e.g.,
`fmmacc`, `mmaqa`). See `13-verification-and-fixes.md` for the full mapping table
and detailed analysis.

Key differences:
- Spec's `__riscv_th_fmmacc` → implementation's `__riscv_th_mfmacc_*`
- Spec's `__riscv_th_mmaqa` → implementation's `__riscv_th_mmacc_w_b`
- Spec's unified `__riscv_th_mld` → implementation's role-specific `__riscv_th_mld_a/b/acc_*`
- Spec's `__riscv_th_mzero_*()` (no params) → implementation's `__riscv_th_mzero_*(m, n)`
- Spec's `(M, N, K)` dimension order → implementation's `(M, K, N)`

## Limitations

- Matrix types cannot cross function boundaries
- No C++ overloading (separate functions per type)
- Not implemented: stream load/store, matrix-scalar EW (`.mx`), 64-bit INT EW (`.d.mm`), `mmov.mv` row move

## Zmpanel Panel-Aware API

The Zmpanel extension adds 30 fire-and-forget macro instructions for panel-aware 2x2 matrix tiling.
These operate on implicit hardware state and do not use compiler-managed matrix register values.

### Panel Configuration (12 functions)

```c
void __riscv_th_mset22adra(size_t addr);   // Set base address of matrix A
void __riscv_th_mset22adrb(size_t addr);   // Set base address of matrix B
void __riscv_th_mset22adrd(size_t addr);   // Set base address of matrix D
void __riscv_th_mset22rsba(size_t stride); // Set row stride of A in bytes
void __riscv_th_mset22rsbb(size_t stride); // Set row stride of B in bytes
void __riscv_th_mset22rsbd(size_t stride); // Set row stride of D in bytes
void __riscv_th_mset22m(size_t m);         // Set panel M dimension
void __riscv_th_mset22n(size_t n);         // Set panel N dimension
void __riscv_th_mset22k(size_t k);         // Set panel K dimension
void __riscv_th_msetrstptr(size_t val);    // Reset all HW pointers
void __riscv_th_msetaccum(size_t mode);    // 0=zero, 1=preload
void __riscv_th_msetoob(size_t policy);    // OOB policy
```

### Panel Load/Store (4 functions)

```c
void __riscv_th_ml22e8(void);    // Load 2x2 panel tiles, 8-bit
void __riscv_th_ml22e16(void);   // Load 2x2 panel tiles, 16-bit
void __riscv_th_msc22e16(void);  // Store 2x2 panel results, 16-bit
void __riscv_th_msc22e32(void);  // Store 2x2 panel results, 32-bit
```

### Panel Compute (14 functions)

```c
// INT8 -> INT32
void __riscv_th_mmacc22_w_b(void);     // signed x signed
void __riscv_th_mmaccu22_w_b(void);    // unsigned x unsigned
void __riscv_th_mmaccus22_w_b(void);   // unsigned x signed
void __riscv_th_mmaccsu22_w_b(void);   // signed x unsigned

// FP8 -> FP16/BF16/FP32
void __riscv_th_mfmacc22_h_e5(void);   // fp8(E5M2) -> fp16
void __riscv_th_mfmacc22_h_e4(void);   // fp8(E4M3) -> fp16
void __riscv_th_mfmacc22_bf16_e5(void);// fp8(E5M2) -> bf16
void __riscv_th_mfmacc22_bf16_e4(void);// fp8(E4M3) -> bf16
void __riscv_th_mfmacc22_s_e5(void);   // fp8(E5M2) -> fp32
void __riscv_th_mfmacc22_s_e4(void);   // fp8(E4M3) -> fp32

// Standard FP
void __riscv_th_mfmacc22_h(void);      // fp16 -> fp16
void __riscv_th_mfmacc22_s_h(void);    // fp16 -> fp32
void __riscv_th_mfmacc22_s_bf16(void); // bf16 -> fp32
void __riscv_th_mfmacc22_s(void);      // fp32 -> fp32
```

### Usage Example: Panel GEMM Pipeline

```c
#include <thead_matrix.h>

void panel_gemm_int8(const void *A, const void *B, void *D,
                     size_t rsa, size_t rsb, size_t rsd,
                     size_t M, size_t N, size_t K) {
  // 1. Configure panel parameters
  __riscv_th_mset22adra((size_t)A);
  __riscv_th_mset22adrb((size_t)B);
  __riscv_th_mset22adrd((size_t)D);
  __riscv_th_mset22rsba(rsa);
  __riscv_th_mset22rsbb(rsb);
  __riscv_th_mset22rsbd(rsd);
  __riscv_th_mset22m(M);
  __riscv_th_mset22n(N);
  __riscv_th_mset22k(K);
  __riscv_th_msetaccum(0);     // zero mode
  __riscv_th_msetoob(2);       // load=zero-pad, store=skip
  __riscv_th_msetrstptr(1);    // reset HW pointers

  // 2. Execute: load 2x2 tiles -> compute -> store results
  __riscv_th_ml22e8();         // load 8 tile registers
  __riscv_th_mmacc22_w_b();    // signed int8 panel MAC
  __riscv_th_msc22e32();       // store 4 acc registers

  // 3. Fence before reading results
  asm volatile("fence");
}
```
