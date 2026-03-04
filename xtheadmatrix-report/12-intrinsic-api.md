# Higher-Level Intrinsic API (`<thead_matrix.h>`)

## Overview

The `<thead_matrix.h>` header provides 300+ C API functions/macros following
the RVM Intrinsic API Reference Manual v0.2. All operations use the Spec-API
(ManagedRA) programming model — no manual register index management needed.

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

```c
// INT matmul (typed sources)
mint32_t   __riscv_th_mmaq_ss_w_b(mint32_t acc, mint8_t a, mint8_t b, mrow_t m, mcol_t k, mcol_t n);
// FP matmul (native precision, typed sources)
mfloat32_t __riscv_th_mfmaqa_s(mfloat32_t acc, mfloat32_t a, mfloat32_t b, mrow_t m, mcol_t k, mcol_t n);
// FP matmul (widening, opaque FP8/BF16/TF32 sources use mint32_t for A/B)
mfloat16_t __riscv_th_mfmaqa_h_e4(mfloat16_t acc, mint32_t a, mint32_t b, mrow_t m, mcol_t k, mcol_t n);
mfloat32_t __riscv_th_mfmaqa_s_tf32(mfloat32_t acc, mint32_t a, mint32_t b, mrow_t m, mcol_t k, mcol_t n);
```
Shorthand aliases: `__riscv_th_mmaq_ss` → `__riscv_th_mmaq_ss_w_b`.

### Zero (11 functions)

```c
mint32_t __riscv_th_mzeros_i32(mrow_t m, mcol_t n);
```

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

### Utility

```c
mint32_t __riscv_th_mundef_i32(void);    // undefined value (PoisonValue)
// + reinterpret casts, tuple get/set
```

## Limitations

- Matrix types cannot cross function boundaries
- No C++ overloading (separate functions per type)
- No stream load/store (not in RVM 0.6 spec)
