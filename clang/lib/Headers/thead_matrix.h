/*===---- thead_matrix.h - T-Head Matrix Extension (RVM 0.6) intrinsic API ---===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===------------------------------------------------------------------------===
 *
 * Higher-level C intrinsic API for the T-Head RISC-V Matrix Extension (RVM 0.6)
 * as specified in the RVM Intrinsic API Reference Manual v0.2.
 *
 * This header provides:
 *   - 22 matrix data types (11 single + 11 pair/x2)
 *   - Dimension types (mrow_t, mcol_t)
 *   - CSR access functions
 *   - Configuration functions
 *   - Typed load/store with role-specific variants (A/B/C matrices)
 *   - Matrix multiply-accumulate (FP and INT)
 *   - Element-wise arithmetic (INT and FP)
 *   - Format conversions (FP, float-int, packed, N4clip)
 *   - Data movement (move, duplicate, pack, slide, broadcast)
 *   - Zero and undefined value constructors
 *   - Reinterpret casts between matrix types
 *   - Tuple (x2) get/set operations
 *
 * Usage:
 *   #include <thead_matrix.h>
 *   Compile with: -march=rv64gc_xtheadmatrix0p6 -menable-experimental-extensions
 *
 *===------------------------------------------------------------------------===*/

#ifndef __THEAD_MATRIX_H
#define __THEAD_MATRIX_H

#if defined(__riscv_xtheadmatrix)

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Section 1: Matrix Data Types
 *
 * Types are native compiler built-in types (__rvm_*_t) that map to opaque
 * target("riscv.matrix") LLVM IR types. These are used directly as function
 * parameters and return values.
 * ============================================================================ */

/* Single-register matrix types (11 types) */
typedef __rvm_int8_t     mint8_t;
typedef __rvm_int16_t    mint16_t;
typedef __rvm_int32_t    mint32_t;
typedef __rvm_int64_t    mint64_t;
typedef __rvm_uint8_t    muint8_t;
typedef __rvm_uint16_t   muint16_t;
typedef __rvm_uint32_t   muint32_t;
typedef __rvm_uint64_t   muint64_t;
typedef __rvm_float16_t  mfloat16_t;
typedef __rvm_float32_t  mfloat32_t;
typedef __rvm_float64_t  mfloat64_t;

/* Register-pair (x2) matrix types (11 types) */
typedef __rvm_int8x2_t    mint8x2_t;
typedef __rvm_int16x2_t   mint16x2_t;
typedef __rvm_int32x2_t   mint32x2_t;
typedef __rvm_int64x2_t   mint64x2_t;
typedef __rvm_uint8x2_t   muint8x2_t;
typedef __rvm_uint16x2_t  muint16x2_t;
typedef __rvm_uint32x2_t  muint32x2_t;
typedef __rvm_uint64x2_t  muint64x2_t;
typedef __rvm_float16x2_t mfloat16x2_t;
typedef __rvm_float32x2_t mfloat32x2_t;
typedef __rvm_float64x2_t mfloat64x2_t;

/* Dimension types */
typedef size_t mrow_t;
typedef size_t mcol_t;

/* CSR enumeration */
enum RVM_CSR {
  RVM_CSR_XMCSR    = 0x806,
  RVM_CSR_MTILEM   = 0x807,
  RVM_CSR_MTILEN   = 0x808,
  RVM_CSR_MTILEK   = 0x809,
  RVM_CSR_XMXRM    = 0x80a,
  RVM_CSR_XMSAT    = 0x80b,
  RVM_CSR_XMFFLAGS = 0x80c,
  RVM_CSR_XMFRM    = 0x80d,
  RVM_CSR_XMSATEN  = 0x80e,
  RVM_CSR_XMISA    = 0xcc0,
  RVM_CSR_XTLENB   = 0xcc1,
  RVM_CSR_XTRLENB  = 0xcc2,
  RVM_CSR_XALENB   = 0xcc3
};

/* Register index constants for builtin register arguments */
#define __RVM_TR0  0
#define __RVM_TR1  1
#define __RVM_TR2  2
#define __RVM_TR3  3
#define __RVM_ACC0 4
#define __RVM_ACC1 5
#define __RVM_ACC2 6
#define __RVM_ACC3 7

/* ============================================================================
 * Section 2: Configuration Functions
 *
 * Set tile dimensions before matrix operations.  In managed-RA mode these
 * are normally emitted automatically by load/store/matmul wrappers, so
 * explicit calls are only needed for advanced manual control.
 *
 *   mrow_t m = __riscv_th_msetmrow_m(M);   // mtilem = M (row count)
 *   mrow_t n = __riscv_th_msetmrow_n(N);   // mtilen = N (row count)
 *   mcol_t k = __riscv_th_msetmcol_e8(K);  // mtilek = K (col count for 8-bit elems)
 *   __riscv_th_mrelease();                  // release matrix unit context
 * ============================================================================ */

/* Set mtilem CSR (M-dimension row count). Returns the value written. */
static __inline__ __attribute__((__always_inline__, __nodebug__))
mrow_t __riscv_th_msetmrow_m(mrow_t m) {
  __builtin_riscv_th_msettilem(m);
  return m;
}

/* Set mtilen CSR (N-dimension row count). Returns the value written. */
static __inline__ __attribute__((__always_inline__, __nodebug__))
mrow_t __riscv_th_msetmrow_n(mrow_t n) {
  __builtin_riscv_th_msettilen(n);
  return n;
}

/* Set mtilek CSR (K-dimension column count).  The e8/e16/e32/e64 suffix
 * indicates the element width; the caller passes the column count in
 * elements (not bytes).  All four variants write the same CSR. */
static __inline__ __attribute__((__always_inline__, __nodebug__))
mcol_t __riscv_th_msetmcol_e8(mcol_t c) {
  __builtin_riscv_th_msettilek(c);
  return c;
}

static __inline__ __attribute__((__always_inline__, __nodebug__))
mcol_t __riscv_th_msetmcol_e16(mcol_t c) {
  __builtin_riscv_th_msettilek(c);
  return c;
}

static __inline__ __attribute__((__always_inline__, __nodebug__))
mcol_t __riscv_th_msetmcol_e32(mcol_t c) {
  __builtin_riscv_th_msettilek(c);
  return c;
}

static __inline__ __attribute__((__always_inline__, __nodebug__))
mcol_t __riscv_th_msetmcol_e64(mcol_t c) {
  __builtin_riscv_th_msettilek(c);
  return c;
}

/* Release the matrix unit context (frees hardware resources). */
static __inline__ __attribute__((__always_inline__, __nodebug__))
void __riscv_th_mrelease(void) {
  __builtin_riscv_th_mrelease();
}

/* Immediate-operand config wrappers (imm must be a compile-time constant). */
#define __riscv_th_msettilemi(imm) __builtin_riscv_th_msettilemi(imm)
#define __riscv_th_msettileki(imm) __builtin_riscv_th_msettileki(imm)
#define __riscv_th_msettileni(imm) __builtin_riscv_th_msettileni(imm)

/* ============================================================================
 * Section 3: CSR Access Functions
 *
 *   unsigned long v = __riscv_th_mread_csr(RVM_CSR_XMCSR);  // read CSR
 *   __riscv_th_mwrite_csr(RVM_CSR_XMFRM, 0);                // write CSR
 *   unsigned long tlen = __riscv_th_xmlenb();   // tile register size in bytes
 *   unsigned long rlen = __riscv_th_xrlenb();   // tile row width in bytes
 *   unsigned long isa  = __riscv_th_xmisa();    // ISA feature bits
 *
 * Read-only CSRs (0xcc0-0xcc3): xmisa, xtlenb, xtrlenb, xalenb.
 * Read-write CSRs (0x806-0x80e): xmcsr, mtilem/n/k, xmxrm, xmsat, etc.
 * ============================================================================ */

#define __riscv_th_mread_csr(csr)                                              \
  __extension__({                                                              \
    unsigned long __val;                                                        \
    switch (csr) {                                                             \
    case RVM_CSR_XMCSR:                                                        \
      __asm__ __volatile__("csrr %0, th.xmcsr" : "=r"(__val));                \
      break;                                                                   \
    case RVM_CSR_MTILEM:                                                       \
      __asm__ __volatile__("csrr %0, th.mtilem" : "=r"(__val));               \
      break;                                                                   \
    case RVM_CSR_MTILEN:                                                       \
      __asm__ __volatile__("csrr %0, th.mtilen" : "=r"(__val));               \
      break;                                                                   \
    case RVM_CSR_MTILEK:                                                       \
      __asm__ __volatile__("csrr %0, th.mtilek" : "=r"(__val));               \
      break;                                                                   \
    case RVM_CSR_XMXRM:                                                        \
      __asm__ __volatile__("csrr %0, th.xmxrm" : "=r"(__val));                \
      break;                                                                   \
    case RVM_CSR_XMSAT:                                                        \
      __asm__ __volatile__("csrr %0, th.xmsat" : "=r"(__val));                \
      break;                                                                   \
    case RVM_CSR_XMFFLAGS:                                                     \
      __asm__ __volatile__("csrr %0, th.xmfflags" : "=r"(__val));             \
      break;                                                                   \
    case RVM_CSR_XMFRM:                                                        \
      __asm__ __volatile__("csrr %0, th.xmfrm" : "=r"(__val));                \
      break;                                                                   \
    case RVM_CSR_XMSATEN:                                                      \
      __asm__ __volatile__("csrr %0, th.xmsaten" : "=r"(__val));              \
      break;                                                                   \
    case RVM_CSR_XMISA:                                                        \
      __asm__ __volatile__("csrr %0, th.xmisa" : "=r"(__val));                \
      break;                                                                   \
    case RVM_CSR_XTLENB:                                                       \
      __asm__ __volatile__("csrr %0, th.xtlenb" : "=r"(__val));               \
      break;                                                                   \
    case RVM_CSR_XTRLENB:                                                      \
      __asm__ __volatile__("csrr %0, th.xtrlenb" : "=r"(__val));              \
      break;                                                                   \
    case RVM_CSR_XALENB:                                                       \
      __asm__ __volatile__("csrr %0, th.xalenb" : "=r"(__val));               \
      break;                                                                   \
    default:                                                                   \
      __val = 0;                                                               \
      break;                                                                   \
    }                                                                          \
    __val;                                                                     \
  })

#define __riscv_th_mwrite_csr(csr, value)                                      \
  do {                                                                         \
    unsigned long __val = (unsigned long)(value);                               \
    switch (csr) {                                                             \
    case RVM_CSR_XMCSR:                                                        \
      __asm__ __volatile__("csrw th.xmcsr, %0" ::"r"(__val));                 \
      break;                                                                   \
    case RVM_CSR_MTILEM:                                                       \
      __asm__ __volatile__("csrw th.mtilem, %0" ::"r"(__val));                \
      break;                                                                   \
    case RVM_CSR_MTILEN:                                                       \
      __asm__ __volatile__("csrw th.mtilen, %0" ::"r"(__val));                \
      break;                                                                   \
    case RVM_CSR_MTILEK:                                                       \
      __asm__ __volatile__("csrw th.mtilek, %0" ::"r"(__val));                \
      break;                                                                   \
    case RVM_CSR_XMXRM:                                                        \
      __asm__ __volatile__("csrw th.xmxrm, %0" ::"r"(__val));                 \
      break;                                                                   \
    case RVM_CSR_XMSAT:                                                        \
      __asm__ __volatile__("csrw th.xmsat, %0" ::"r"(__val));                 \
      break;                                                                   \
    case RVM_CSR_XMFFLAGS:                                                     \
      __asm__ __volatile__("csrw th.xmfflags, %0" ::"r"(__val));              \
      break;                                                                   \
    case RVM_CSR_XMFRM:                                                        \
      __asm__ __volatile__("csrw th.xmfrm, %0" ::"r"(__val));                 \
      break;                                                                   \
    case RVM_CSR_XMSATEN:                                                      \
      __asm__ __volatile__("csrw th.xmsaten, %0" ::"r"(__val));               \
      break;                                                                   \
    default:                                                                   \
      break;                                                                   \
    }                                                                          \
  } while (0)

static __inline__ __attribute__((__always_inline__, __nodebug__))
unsigned long __riscv_th_xmlenb(void) {
  unsigned long __val;
  __asm__ __volatile__("csrr %0, th.xtlenb" : "=r"(__val));
  return __val;
}

static __inline__ __attribute__((__always_inline__, __nodebug__))
unsigned long __riscv_th_xrlenb(void) {
  unsigned long __val;
  __asm__ __volatile__("csrr %0, th.xtrlenb" : "=r"(__val));
  return __val;
}

/* __riscv_th_xmisa: Read the matrix ISA feature bits (xmisa, CSR 0xcc0).
 * This is the canonical function for reading the matrix ISA register. */
static __inline__ __attribute__((__always_inline__, __nodebug__))
unsigned long __riscv_th_xmisa(void) {
  unsigned long __val;
  __asm__ __volatile__("csrr %0, th.xmisa" : "=r"(__val));
  return __val;
}

/* __riscv_th_xmsize: Compatibility alias from the older intrinsic API spec.
 * The "xmsize" CSR does not exist in RVM 0.6; this maps to xmisa. */
static __inline__ __attribute__((__always_inline__, __nodebug__))
unsigned long __riscv_th_xmsize(void) {
  return __riscv_th_xmisa();
}

/* ============================================================================
 * Section 5: Undefined Functions (22)
 *
 * Return an undefined (poison) matrix value.  Used as a placeholder when
 * a value will be fully overwritten before use (e.g., initializing an x2
 * pair before inserting both elements).
 *
 *   mint32_t u = __riscv_th_mundefined_i32();
 * ============================================================================ */

#define __THEAD_MUNDEFINED_SINGLE(SUFFIX, TYPE, MUNDEF)                        \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  TYPE __riscv_th_mundefined_##SUFFIX(void) {                                  \
    return __builtin_riscv_th_##MUNDEF();                                      \
  }

#define __THEAD_MUNDEFINED_PAIR(SUFFIX, TYPE, MUNDEF)                          \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  TYPE __riscv_th_mundefined_##SUFFIX(void) {                                  \
    return __builtin_riscv_th_##MUNDEF();                                      \
  }

__THEAD_MUNDEFINED_SINGLE(i8,  mint8_t,     mundef_i8)
__THEAD_MUNDEFINED_SINGLE(i16, mint16_t,    mundef_i16)
__THEAD_MUNDEFINED_SINGLE(i32, mint32_t,    mundef_i32)
__THEAD_MUNDEFINED_SINGLE(i64, mint64_t,    mundef_i64)
__THEAD_MUNDEFINED_SINGLE(u8,  muint8_t,    mundef_u8)
__THEAD_MUNDEFINED_SINGLE(u16, muint16_t,   mundef_u16)
__THEAD_MUNDEFINED_SINGLE(u32, muint32_t,   mundef_u32)
__THEAD_MUNDEFINED_SINGLE(u64, muint64_t,   mundef_u64)
__THEAD_MUNDEFINED_SINGLE(f16, mfloat16_t,  mundef_f16)
__THEAD_MUNDEFINED_SINGLE(f32, mfloat32_t,  mundef_f32)
__THEAD_MUNDEFINED_SINGLE(f64, mfloat64_t,  mundef_f64)

__THEAD_MUNDEFINED_PAIR(i8x2,  mint8x2_t,    mundef_i8x2)
__THEAD_MUNDEFINED_PAIR(i16x2, mint16x2_t,   mundef_i16x2)
__THEAD_MUNDEFINED_PAIR(i32x2, mint32x2_t,   mundef_i32x2)
__THEAD_MUNDEFINED_PAIR(i64x2, mint64x2_t,   mundef_i64x2)
__THEAD_MUNDEFINED_PAIR(u8x2,  muint8x2_t,   mundef_u8x2)
__THEAD_MUNDEFINED_PAIR(u16x2, muint16x2_t,  mundef_u16x2)
__THEAD_MUNDEFINED_PAIR(u32x2, muint32x2_t,  mundef_u32x2)
__THEAD_MUNDEFINED_PAIR(u64x2, muint64x2_t,  mundef_u64x2)
__THEAD_MUNDEFINED_PAIR(f16x2, mfloat16x2_t, mundef_f16x2)
__THEAD_MUNDEFINED_PAIR(f32x2, mfloat32x2_t, mundef_f32x2)
__THEAD_MUNDEFINED_PAIR(f64x2, mfloat64x2_t, mundef_f64x2)

/* ============================================================================
 * Section 6: Reinterpret Functions (22)
 *
 * Bitwise reinterpretation between matrix types. All single-register matrix
 * types share the same physical register, so reinterpretation is a no-op at
 * the hardware level. We use inline asm with tied constraints ("0") to
 * preserve the register value while changing the C-level type.
 *
 * Per spec: "The type of SRC can be any matrix type with the same number
 * of registers."
 * ============================================================================ */

#define __THEAD_MREINTERPRET(DST_TYPE, src)                                    \
  __extension__({                                                              \
    DST_TYPE __result;                                                         \
    __asm__("" : "=tr"(__result) : "tr"(src));                                 \
    __result;                                                                  \
  })

#define __riscv_th_mreinterpret_i8(src)    __THEAD_MREINTERPRET(mint8_t, src)
#define __riscv_th_mreinterpret_i16(src)   __THEAD_MREINTERPRET(mint16_t, src)
#define __riscv_th_mreinterpret_i32(src)   __THEAD_MREINTERPRET(mint32_t, src)
#define __riscv_th_mreinterpret_i64(src)   __THEAD_MREINTERPRET(mint64_t, src)
#define __riscv_th_mreinterpret_u8(src)    __THEAD_MREINTERPRET(muint8_t, src)
#define __riscv_th_mreinterpret_u16(src)   __THEAD_MREINTERPRET(muint16_t, src)
#define __riscv_th_mreinterpret_u32(src)   __THEAD_MREINTERPRET(muint32_t, src)
#define __riscv_th_mreinterpret_u64(src)   __THEAD_MREINTERPRET(muint64_t, src)
#define __riscv_th_mreinterpret_f16(src)   __THEAD_MREINTERPRET(mfloat16_t, src)
#define __riscv_th_mreinterpret_f32(src)   __THEAD_MREINTERPRET(mfloat32_t, src)
#define __riscv_th_mreinterpret_f64(src)   __THEAD_MREINTERPRET(mfloat64_t, src)

// x2 reinterpret macros removed: x2 types are struct types that cannot fit a
// single "tr" inline asm constraint. Use mget/mset to decompose x2 values,
// reinterpret individual elements, and reassemble.

/* ============================================================================
 * Section 7: Tuple Get/Set Functions (22)
 *
 * Extract or insert a single register from/to a register-pair (x2) type.
 *
 *   mfloat16_t elem  = __riscv_th_mget_f16(pair, 0);      // extract slot 0
 *   pair = __riscv_th_mset_f16(pair, 1, new_val);          // insert into slot 1
 *
 * At -O2 with constant index, these fold to direct struct access.
 * ============================================================================ */

#define __THEAD_MGET(SUFFIX, STYPE, PTYPE)                                     \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  STYPE __riscv_th_mget_##SUFFIX(PTYPE __src, size_t __index) {                \
    return __builtin_riscv_th_mget_spec_##SUFFIX(__src, __index);               \
  }

#define __THEAD_MSET(SUFFIX, STYPE, PTYPE)                                     \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  PTYPE __riscv_th_mset_##SUFFIX(PTYPE __src, size_t __index,                  \
                                 STYPE __val) {                                \
    return __builtin_riscv_th_mset_spec_##SUFFIX(__src, __index, __val);        \
  }

__THEAD_MGET(i8,  mint8_t,     mint8x2_t)
__THEAD_MGET(i16, mint16_t,    mint16x2_t)
__THEAD_MGET(i32, mint32_t,    mint32x2_t)
__THEAD_MGET(i64, mint64_t,    mint64x2_t)
__THEAD_MGET(u8,  muint8_t,    muint8x2_t)
__THEAD_MGET(u16, muint16_t,   muint16x2_t)
__THEAD_MGET(u32, muint32_t,   muint32x2_t)
__THEAD_MGET(u64, muint64_t,   muint64x2_t)
__THEAD_MGET(f16, mfloat16_t,  mfloat16x2_t)
__THEAD_MGET(f32, mfloat32_t,  mfloat32x2_t)
__THEAD_MGET(f64, mfloat64_t,  mfloat64x2_t)

__THEAD_MSET(i8,  mint8_t,     mint8x2_t)
__THEAD_MSET(i16, mint16_t,    mint16x2_t)
__THEAD_MSET(i32, mint32_t,    mint32x2_t)
__THEAD_MSET(i64, mint64_t,    mint64x2_t)
__THEAD_MSET(u8,  muint8_t,    muint8x2_t)
__THEAD_MSET(u16, muint16_t,   muint16x2_t)
__THEAD_MSET(u32, muint32_t,   muint32x2_t)
__THEAD_MSET(u64, muint64_t,   muint64x2_t)
__THEAD_MSET(f16, mfloat16_t,  mfloat16x2_t)
__THEAD_MSET(f32, mfloat32_t,  mfloat32x2_t)
__THEAD_MSET(f64, mfloat64_t,  mfloat64x2_t)

/* ============================================================================
 * Section 8: Spec-API — Register-Allocator-Managed Intrinsics
 *
 * The compiler's register allocator manages matrix registers (TR0-TR3,
 * ACC0-ACC3). Matrix values are returned and passed as opaque types
 * (mint32_t etc.) with proper SSA dataflow. No manual register index
 * management is needed.  Tile config CSRs (mtilem/n/k) are emitted
 * automatically by each wrapper.
 *
 * Typical usage:
 *   mint8_t  a = __riscv_th_mld_a_i8(A, stride_a, M, K);   // load A-tile
 *   mint8_t  b = __riscv_th_mld_b_i8(B, stride_b, K, N);   // load B-tile
 *   mint32_t c = __riscv_th_mzero_i32(M, N);                // zero accumulator
 *   c = __riscv_th_mmacc_w_b(c, a, b, M, K, N);            // c += a * b^T
 *   __riscv_th_mst_i32(C, stride_c, c, M, N);              // store result
 *   __riscv_th_mrelease();
 * ============================================================================ */

/* --- A-tile loads (mlae: M×K dimensions) ---
 * Load matrix A from memory into a tile register (tr0-tr3).
 *   mint8_t a = __riscv_th_mld_a_i8(base, stride, M, K);
 * Parameters: base=pointer, stride=row stride in bytes, m=rows, k=cols. */
#define __THEAD_SPEC_MLD(SUFFIX, CTYPE, MTYPE, BUILTIN)                        \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  MTYPE __riscv_th_mld_a_##SUFFIX(const CTYPE *__base, long __stride,          \
                                   mrow_t __m, mcol_t __k) {                    \
    return __builtin_riscv_th_##BUILTIN((void *)__base, __stride, __m, __k);   \
  }

__THEAD_SPEC_MLD(i8,  int8_t,     mint8_t,     mld_spec_i8)
__THEAD_SPEC_MLD(i16, int16_t,    mint16_t,    mld_spec_i16)
__THEAD_SPEC_MLD(i32, int32_t,    mint32_t,    mld_spec_i32)
__THEAD_SPEC_MLD(i64, int64_t,    mint64_t,    mld_spec_i64)
__THEAD_SPEC_MLD(u8,  uint8_t,    muint8_t,    mld_spec_u8)
__THEAD_SPEC_MLD(u16, uint16_t,   muint16_t,   mld_spec_u16)
__THEAD_SPEC_MLD(u32, uint32_t,   muint32_t,   mld_spec_u32)
__THEAD_SPEC_MLD(u64, uint64_t,   muint64_t,   mld_spec_u64)
__THEAD_SPEC_MLD(f16, uint16_t,   mfloat16_t,  mld_spec_f16)
__THEAD_SPEC_MLD(f32, float,      mfloat32_t,  mld_spec_f32)
__THEAD_SPEC_MLD(f64, double,     mfloat64_t,  mld_spec_f64)

/* --- B-tile loads (mlbe: K×N dimensions) ---
 * Load matrix B into a tile register (tr0-tr3).
 *   mint8_t b = __riscv_th_mld_b_i8(base, stride, K, N);  */
#define __THEAD_SPEC_MLD_B(SUFFIX, CTYPE, MTYPE, BUILTIN)                      \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  MTYPE __riscv_th_mld_b_##SUFFIX(const CTYPE *__base, long __stride,          \
                                   mcol_t __k, mcol_t __n) {                    \
    return __builtin_riscv_th_##BUILTIN((void *)__base, __stride, __k, __n);   \
  }

__THEAD_SPEC_MLD_B(i8,  int8_t,     mint8_t,     mld_b_spec_i8)
__THEAD_SPEC_MLD_B(i16, int16_t,    mint16_t,    mld_b_spec_i16)
__THEAD_SPEC_MLD_B(i32, int32_t,    mint32_t,    mld_b_spec_i32)
__THEAD_SPEC_MLD_B(i64, int64_t,    mint64_t,    mld_b_spec_i64)
__THEAD_SPEC_MLD_B(u8,  uint8_t,    muint8_t,    mld_b_spec_u8)
__THEAD_SPEC_MLD_B(u16, uint16_t,   muint16_t,   mld_b_spec_u16)
__THEAD_SPEC_MLD_B(u32, uint32_t,   muint32_t,   mld_b_spec_u32)
__THEAD_SPEC_MLD_B(u64, uint64_t,   muint64_t,   mld_b_spec_u64)
__THEAD_SPEC_MLD_B(f16, uint16_t,   mfloat16_t,  mld_b_spec_f16)
__THEAD_SPEC_MLD_B(f32, float,      mfloat32_t,  mld_b_spec_f32)
__THEAD_SPEC_MLD_B(f64, double,     mfloat64_t,  mld_b_spec_f64)

/* --- Accumulator loads (mlce: M×N dimensions) ---
 * Load matrix C into an accumulator register (acc0-acc3).
 *   mint32_t c = __riscv_th_mld_acc_i32(base, stride, M, N);  */
#define __THEAD_SPEC_MLD_ACC(SUFFIX, CTYPE, MTYPE, BUILTIN)                    \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  MTYPE __riscv_th_mld_acc_##SUFFIX(const CTYPE *__base, long __stride,        \
                                     mrow_t __m, mcol_t __n) {                  \
    return __builtin_riscv_th_##BUILTIN((void *)__base, __stride, __m, __n);   \
  }

__THEAD_SPEC_MLD_ACC(i8,  int8_t,     mint8_t,     mld_acc_spec_i8)
__THEAD_SPEC_MLD_ACC(i16, int16_t,    mint16_t,    mld_acc_spec_i16)
__THEAD_SPEC_MLD_ACC(i32, int32_t,    mint32_t,    mld_acc_spec_i32)
__THEAD_SPEC_MLD_ACC(i64, int64_t,    mint64_t,    mld_acc_spec_i64)
__THEAD_SPEC_MLD_ACC(u8,  uint8_t,    muint8_t,    mld_acc_spec_u8)
__THEAD_SPEC_MLD_ACC(u16, uint16_t,   muint16_t,   mld_acc_spec_u16)
__THEAD_SPEC_MLD_ACC(u32, uint32_t,   muint32_t,   mld_acc_spec_u32)
__THEAD_SPEC_MLD_ACC(u64, uint64_t,   muint64_t,   mld_acc_spec_u64)
__THEAD_SPEC_MLD_ACC(f16, uint16_t,   mfloat16_t,  mld_acc_spec_f16)
__THEAD_SPEC_MLD_ACC(f32, float,      mfloat32_t,  mld_acc_spec_f32)
__THEAD_SPEC_MLD_ACC(f64, double,     mfloat64_t,  mld_acc_spec_f64)

/* --- Stores (msce: M×N dimensions) ---
 * Store accumulator to memory.
 *   __riscv_th_mst_i32(base, stride, val, M, N);  */
#define __THEAD_SPEC_MST(SUFFIX, CTYPE, MTYPE, BUILTIN)                        \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  void __riscv_th_mst_##SUFFIX(CTYPE *__base, long __stride, MTYPE __val,      \
                                mrow_t __m, mcol_t __n) {                       \
    __builtin_riscv_th_##BUILTIN((void *)__base, __stride, __val, __m, __n);   \
  }

__THEAD_SPEC_MST(i8,  int8_t,     mint8_t,     mst_spec_i8)
__THEAD_SPEC_MST(i16, int16_t,    mint16_t,    mst_spec_i16)
__THEAD_SPEC_MST(i32, int32_t,    mint32_t,    mst_spec_i32)
__THEAD_SPEC_MST(i64, int64_t,    mint64_t,    mst_spec_i64)
__THEAD_SPEC_MST(u8,  uint8_t,    muint8_t,    mst_spec_u8)
__THEAD_SPEC_MST(u16, uint16_t,   muint16_t,   mst_spec_u16)
__THEAD_SPEC_MST(u32, uint32_t,   muint32_t,   mst_spec_u32)
__THEAD_SPEC_MST(u64, uint64_t,   muint64_t,   mst_spec_u64)
__THEAD_SPEC_MST(f16, uint16_t,   mfloat16_t,  mst_spec_f16)
__THEAD_SPEC_MST(f32, float,      mfloat32_t,  mst_spec_f32)
__THEAD_SPEC_MST(f64, double,     mfloat64_t,  mst_spec_f64)

/* --- A-tile stores (msae: M×K dimensions) ---
 * Store tile register as A-tile to memory. */
#define __THEAD_SPEC_MST_A(SUFFIX, CTYPE, MTYPE, BUILTIN)                      \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  void __riscv_th_mst_a_##SUFFIX(CTYPE *__base, long __stride, MTYPE __val,    \
                                  mrow_t __m, mcol_t __k) {                     \
    __builtin_riscv_th_##BUILTIN((void *)__base, __stride, __val, __m, __k);   \
  }

__THEAD_SPEC_MST_A(i8,  int8_t,     mint8_t,     mst_a_spec_i8)
__THEAD_SPEC_MST_A(i16, int16_t,    mint16_t,    mst_a_spec_i16)
__THEAD_SPEC_MST_A(i32, int32_t,    mint32_t,    mst_a_spec_i32)
__THEAD_SPEC_MST_A(i64, int64_t,    mint64_t,    mst_a_spec_i64)
__THEAD_SPEC_MST_A(u8,  uint8_t,    muint8_t,    mst_a_spec_u8)
__THEAD_SPEC_MST_A(u16, uint16_t,   muint16_t,   mst_a_spec_u16)
__THEAD_SPEC_MST_A(u32, uint32_t,   muint32_t,   mst_a_spec_u32)
__THEAD_SPEC_MST_A(u64, uint64_t,   muint64_t,   mst_a_spec_u64)
__THEAD_SPEC_MST_A(f16, uint16_t,   mfloat16_t,  mst_a_spec_f16)
__THEAD_SPEC_MST_A(f32, float,      mfloat32_t,  mst_a_spec_f32)
__THEAD_SPEC_MST_A(f64, double,     mfloat64_t,  mst_a_spec_f64)

/* --- B-tile stores (msbe: K×N dimensions) ---
 * Store tile register as B-tile to memory. */
#define __THEAD_SPEC_MST_B(SUFFIX, CTYPE, MTYPE, BUILTIN)                      \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  void __riscv_th_mst_b_##SUFFIX(CTYPE *__base, long __stride, MTYPE __val,    \
                                  mcol_t __k, mcol_t __n) {                     \
    __builtin_riscv_th_##BUILTIN((void *)__base, __stride, __val, __k, __n);   \
  }

__THEAD_SPEC_MST_B(i8,  int8_t,     mint8_t,     mst_b_spec_i8)
__THEAD_SPEC_MST_B(i16, int16_t,    mint16_t,    mst_b_spec_i16)
__THEAD_SPEC_MST_B(i32, int32_t,    mint32_t,    mst_b_spec_i32)
__THEAD_SPEC_MST_B(i64, int64_t,    mint64_t,    mst_b_spec_i64)
__THEAD_SPEC_MST_B(u8,  uint8_t,    muint8_t,    mst_b_spec_u8)
__THEAD_SPEC_MST_B(u16, uint16_t,   muint16_t,   mst_b_spec_u16)
__THEAD_SPEC_MST_B(u32, uint32_t,   muint32_t,   mst_b_spec_u32)
__THEAD_SPEC_MST_B(u64, uint64_t,   muint64_t,   mst_b_spec_u64)
__THEAD_SPEC_MST_B(f16, uint16_t,   mfloat16_t,  mst_b_spec_f16)
__THEAD_SPEC_MST_B(f32, float,      mfloat32_t,  mst_b_spec_f32)
__THEAD_SPEC_MST_B(f64, double,     mfloat64_t,  mst_b_spec_f64)

/* --- Transposed A-tile loads (mlate: M×K dimensions) ---
 * Load matrix A with transposition into a tile register. */
#define __THEAD_SPEC_MLD_AT(SUFFIX, CTYPE, MTYPE, BUILTIN)                     \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  MTYPE __riscv_th_mld_at_##SUFFIX(const CTYPE *__base, long __stride,         \
                                    mrow_t __m, mcol_t __k) {                   \
    return __builtin_riscv_th_##BUILTIN((void *)__base, __stride, __m, __k);   \
  }

__THEAD_SPEC_MLD_AT(i8,  int8_t,     mint8_t,     mld_at_spec_i8)
__THEAD_SPEC_MLD_AT(i16, int16_t,    mint16_t,    mld_at_spec_i16)
__THEAD_SPEC_MLD_AT(i32, int32_t,    mint32_t,    mld_at_spec_i32)
__THEAD_SPEC_MLD_AT(i64, int64_t,    mint64_t,    mld_at_spec_i64)
__THEAD_SPEC_MLD_AT(u8,  uint8_t,    muint8_t,    mld_at_spec_u8)
__THEAD_SPEC_MLD_AT(u16, uint16_t,   muint16_t,   mld_at_spec_u16)
__THEAD_SPEC_MLD_AT(u32, uint32_t,   muint32_t,   mld_at_spec_u32)
__THEAD_SPEC_MLD_AT(u64, uint64_t,   muint64_t,   mld_at_spec_u64)
__THEAD_SPEC_MLD_AT(f16, uint16_t,   mfloat16_t,  mld_at_spec_f16)
__THEAD_SPEC_MLD_AT(f32, float,      mfloat32_t,  mld_at_spec_f32)
__THEAD_SPEC_MLD_AT(f64, double,     mfloat64_t,  mld_at_spec_f64)

/* --- Transposed B-tile loads (mlbte: K×N dimensions) ---
 * Load matrix B with transposition into a tile register. */
#define __THEAD_SPEC_MLD_BT(SUFFIX, CTYPE, MTYPE, BUILTIN)                     \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  MTYPE __riscv_th_mld_bt_##SUFFIX(const CTYPE *__base, long __stride,         \
                                    mcol_t __k, mcol_t __n) {                   \
    return __builtin_riscv_th_##BUILTIN((void *)__base, __stride, __k, __n);   \
  }

__THEAD_SPEC_MLD_BT(i8,  int8_t,     mint8_t,     mld_bt_spec_i8)
__THEAD_SPEC_MLD_BT(i16, int16_t,    mint16_t,    mld_bt_spec_i16)
__THEAD_SPEC_MLD_BT(i32, int32_t,    mint32_t,    mld_bt_spec_i32)
__THEAD_SPEC_MLD_BT(i64, int64_t,    mint64_t,    mld_bt_spec_i64)
__THEAD_SPEC_MLD_BT(u8,  uint8_t,    muint8_t,    mld_bt_spec_u8)
__THEAD_SPEC_MLD_BT(u16, uint16_t,   muint16_t,   mld_bt_spec_u16)
__THEAD_SPEC_MLD_BT(u32, uint32_t,   muint32_t,   mld_bt_spec_u32)
__THEAD_SPEC_MLD_BT(u64, uint64_t,   muint64_t,   mld_bt_spec_u64)
__THEAD_SPEC_MLD_BT(f16, uint16_t,   mfloat16_t,  mld_bt_spec_f16)
__THEAD_SPEC_MLD_BT(f32, float,      mfloat32_t,  mld_bt_spec_f32)
__THEAD_SPEC_MLD_BT(f64, double,     mfloat64_t,  mld_bt_spec_f64)

/* --- Transposed C-tile loads (mlcte: M×N dimensions) ---
 * Load matrix C with transposition into an accumulator register. */
#define __THEAD_SPEC_MLD_CT(SUFFIX, CTYPE, MTYPE, BUILTIN)                     \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  MTYPE __riscv_th_mld_ct_##SUFFIX(const CTYPE *__base, long __stride,         \
                                    mrow_t __m, mcol_t __n) {                   \
    return __builtin_riscv_th_##BUILTIN((void *)__base, __stride, __m, __n);   \
  }

__THEAD_SPEC_MLD_CT(i8,  int8_t,     mint8_t,     mld_ct_spec_i8)
__THEAD_SPEC_MLD_CT(i16, int16_t,    mint16_t,    mld_ct_spec_i16)
__THEAD_SPEC_MLD_CT(i32, int32_t,    mint32_t,    mld_ct_spec_i32)
__THEAD_SPEC_MLD_CT(i64, int64_t,    mint64_t,    mld_ct_spec_i64)
__THEAD_SPEC_MLD_CT(u8,  uint8_t,    muint8_t,    mld_ct_spec_u8)
__THEAD_SPEC_MLD_CT(u16, uint16_t,   muint16_t,   mld_ct_spec_u16)
__THEAD_SPEC_MLD_CT(u32, uint32_t,   muint32_t,   mld_ct_spec_u32)
__THEAD_SPEC_MLD_CT(u64, uint64_t,   muint64_t,   mld_ct_spec_u64)
__THEAD_SPEC_MLD_CT(f16, uint16_t,   mfloat16_t,  mld_ct_spec_f16)
__THEAD_SPEC_MLD_CT(f32, float,      mfloat32_t,  mld_ct_spec_f32)
__THEAD_SPEC_MLD_CT(f64, double,     mfloat64_t,  mld_ct_spec_f64)

/* --- Transposed A-tile stores (msate: M×K dimensions) ---
 * Store tile register as transposed A-tile to memory. */
#define __THEAD_SPEC_MST_AT(SUFFIX, CTYPE, MTYPE, BUILTIN)                     \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  void __riscv_th_mst_at_##SUFFIX(CTYPE *__base, long __stride, MTYPE __val,   \
                                   mrow_t __m, mcol_t __k) {                    \
    __builtin_riscv_th_##BUILTIN((void *)__base, __stride, __val, __m, __k);   \
  }

__THEAD_SPEC_MST_AT(i8,  int8_t,     mint8_t,     mst_at_spec_i8)
__THEAD_SPEC_MST_AT(i16, int16_t,    mint16_t,    mst_at_spec_i16)
__THEAD_SPEC_MST_AT(i32, int32_t,    mint32_t,    mst_at_spec_i32)
__THEAD_SPEC_MST_AT(i64, int64_t,    mint64_t,    mst_at_spec_i64)
__THEAD_SPEC_MST_AT(u8,  uint8_t,    muint8_t,    mst_at_spec_u8)
__THEAD_SPEC_MST_AT(u16, uint16_t,   muint16_t,   mst_at_spec_u16)
__THEAD_SPEC_MST_AT(u32, uint32_t,   muint32_t,   mst_at_spec_u32)
__THEAD_SPEC_MST_AT(u64, uint64_t,   muint64_t,   mst_at_spec_u64)
__THEAD_SPEC_MST_AT(f16, uint16_t,   mfloat16_t,  mst_at_spec_f16)
__THEAD_SPEC_MST_AT(f32, float,      mfloat32_t,  mst_at_spec_f32)
__THEAD_SPEC_MST_AT(f64, double,     mfloat64_t,  mst_at_spec_f64)

/* --- Transposed B-tile stores (msbte: K×N dimensions) ---
 * Store tile register as transposed B-tile to memory. */
#define __THEAD_SPEC_MST_BT(SUFFIX, CTYPE, MTYPE, BUILTIN)                     \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  void __riscv_th_mst_bt_##SUFFIX(CTYPE *__base, long __stride, MTYPE __val,   \
                                   mcol_t __k, mcol_t __n) {                    \
    __builtin_riscv_th_##BUILTIN((void *)__base, __stride, __val, __k, __n);   \
  }

__THEAD_SPEC_MST_BT(i8,  int8_t,     mint8_t,     mst_bt_spec_i8)
__THEAD_SPEC_MST_BT(i16, int16_t,    mint16_t,    mst_bt_spec_i16)
__THEAD_SPEC_MST_BT(i32, int32_t,    mint32_t,    mst_bt_spec_i32)
__THEAD_SPEC_MST_BT(i64, int64_t,    mint64_t,    mst_bt_spec_i64)
__THEAD_SPEC_MST_BT(u8,  uint8_t,    muint8_t,    mst_bt_spec_u8)
__THEAD_SPEC_MST_BT(u16, uint16_t,   muint16_t,   mst_bt_spec_u16)
__THEAD_SPEC_MST_BT(u32, uint32_t,   muint32_t,   mst_bt_spec_u32)
__THEAD_SPEC_MST_BT(u64, uint64_t,   muint64_t,   mst_bt_spec_u64)
__THEAD_SPEC_MST_BT(f16, uint16_t,   mfloat16_t,  mst_bt_spec_f16)
__THEAD_SPEC_MST_BT(f32, float,      mfloat32_t,  mst_bt_spec_f32)
__THEAD_SPEC_MST_BT(f64, double,     mfloat64_t,  mst_bt_spec_f64)

/* --- Transposed C-tile stores (mscte: M×N dimensions) ---
 * Store accumulator as transposed C-tile to memory. */
#define __THEAD_SPEC_MST_CT(SUFFIX, CTYPE, MTYPE, BUILTIN)                     \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  void __riscv_th_mst_ct_##SUFFIX(CTYPE *__base, long __stride, MTYPE __val,   \
                                   mrow_t __m, mcol_t __n) {                    \
    __builtin_riscv_th_##BUILTIN((void *)__base, __stride, __val, __m, __n);   \
  }

__THEAD_SPEC_MST_CT(i8,  int8_t,     mint8_t,     mst_ct_spec_i8)
__THEAD_SPEC_MST_CT(i16, int16_t,    mint16_t,    mst_ct_spec_i16)
__THEAD_SPEC_MST_CT(i32, int32_t,    mint32_t,    mst_ct_spec_i32)
__THEAD_SPEC_MST_CT(i64, int64_t,    mint64_t,    mst_ct_spec_i64)
__THEAD_SPEC_MST_CT(u8,  uint8_t,    muint8_t,    mst_ct_spec_u8)
__THEAD_SPEC_MST_CT(u16, uint16_t,   muint16_t,   mst_ct_spec_u16)
__THEAD_SPEC_MST_CT(u32, uint32_t,   muint32_t,   mst_ct_spec_u32)
__THEAD_SPEC_MST_CT(u64, uint64_t,   muint64_t,   mst_ct_spec_u64)
__THEAD_SPEC_MST_CT(f16, uint16_t,   mfloat16_t,  mst_ct_spec_f16)
__THEAD_SPEC_MST_CT(f32, float,      mfloat32_t,  mst_ct_spec_f32)
__THEAD_SPEC_MST_CT(f64, double,     mfloat64_t,  mst_ct_spec_f64)

/* --- Whole-register loads (mlme) ---
 * Load an entire matrix register (any tr/acc) without role distinction.
 *   mint32_t r = __riscv_th_mld_m_i32(base, stride);  */
#define __THEAD_SPEC_MLD_M(SUFFIX, CTYPE, MTYPE, BUILTIN)                      \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  MTYPE __riscv_th_mld_m_##SUFFIX(const CTYPE *__base, long __stride) {        \
    return __builtin_riscv_th_##BUILTIN((void *)__base, __stride);             \
  }

__THEAD_SPEC_MLD_M(i8,  int8_t,     mint8_t,     mld_m_spec_i8)
__THEAD_SPEC_MLD_M(i16, int16_t,    mint16_t,    mld_m_spec_i16)
__THEAD_SPEC_MLD_M(i32, int32_t,    mint32_t,    mld_m_spec_i32)
__THEAD_SPEC_MLD_M(i64, int64_t,    mint64_t,    mld_m_spec_i64)
__THEAD_SPEC_MLD_M(u8,  uint8_t,    muint8_t,    mld_m_spec_u8)
__THEAD_SPEC_MLD_M(u16, uint16_t,   muint16_t,   mld_m_spec_u16)
__THEAD_SPEC_MLD_M(u32, uint32_t,   muint32_t,   mld_m_spec_u32)
__THEAD_SPEC_MLD_M(u64, uint64_t,   muint64_t,   mld_m_spec_u64)
__THEAD_SPEC_MLD_M(f16, uint16_t,   mfloat16_t,  mld_m_spec_f16)
__THEAD_SPEC_MLD_M(f32, float,      mfloat32_t,  mld_m_spec_f32)
__THEAD_SPEC_MLD_M(f64, double,     mfloat64_t,  mld_m_spec_f64)

/* --- Whole-register stores (msme) ---
 * Store an entire matrix register without role distinction. */
#define __THEAD_SPEC_MST_M(SUFFIX, CTYPE, MTYPE, BUILTIN)                      \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  void __riscv_th_mst_m_##SUFFIX(CTYPE *__base, long __stride, MTYPE __val) {  \
    __builtin_riscv_th_##BUILTIN((void *)__base, __stride, __val);             \
  }

__THEAD_SPEC_MST_M(i8,  int8_t,     mint8_t,     mst_m_spec_i8)
__THEAD_SPEC_MST_M(i16, int16_t,    mint16_t,    mst_m_spec_i16)
__THEAD_SPEC_MST_M(i32, int32_t,    mint32_t,    mst_m_spec_i32)
__THEAD_SPEC_MST_M(i64, int64_t,    mint64_t,    mst_m_spec_i64)
__THEAD_SPEC_MST_M(u8,  uint8_t,    muint8_t,    mst_m_spec_u8)
__THEAD_SPEC_MST_M(u16, uint16_t,   muint16_t,   mst_m_spec_u16)
__THEAD_SPEC_MST_M(u32, uint32_t,   muint32_t,   mst_m_spec_u32)
__THEAD_SPEC_MST_M(u64, uint64_t,   muint64_t,   mst_m_spec_u64)
__THEAD_SPEC_MST_M(f16, uint16_t,   mfloat16_t,  mst_m_spec_f16)
__THEAD_SPEC_MST_M(f32, float,      mfloat32_t,  mst_m_spec_f32)
__THEAD_SPEC_MST_M(f64, double,     mfloat64_t,  mst_m_spec_f64)

/* --- INT matmul: acc += A * B^T ---
 * Names follow RVM 0.6 assembly mnemonics.  Operand order: (acc, a, b, m, k, n).
 *   mint32_t c = __riscv_th_mmacc_w_b(c, a, b, M, K, N);     // signed i8→i32
 *   muint32_t c = __riscv_th_mmaccu_w_b(c, a, b, M, K, N);   // unsigned i8→i32
 *   mint32_t c = __riscv_th_mmaccus_w_b(c, a, b, M, K, N);   // a=unsigned, b=signed
 *   mint32_t c = __riscv_th_mmaccsu_w_b(c, a, b, M, K, N);   // a=signed, b=unsigned
 * Hardware semantics: md = md + ms1 * ms2^T (A→ms1, B→ms2). */
#define __THEAD_SPEC_MMACC(NAME, ATYPE, BTYPE, CTYPE, BUILTIN)                \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  CTYPE __riscv_th_##NAME(CTYPE __c, ATYPE __a, BTYPE __b,                    \
                           mrow_t __m, mcol_t __k, mcol_t __n) {               \
    return __builtin_riscv_th_##BUILTIN(__c, __a, __b, __m, __k, __n);         \
  }

/* INT8 -> INT32 (quad-widen) */
__THEAD_SPEC_MMACC(mmacc_w_b,    mint8_t,  mint8_t,  mint32_t,  mmaqa_spec_ss_w_b)
__THEAD_SPEC_MMACC(mmaccu_w_b,   muint8_t, muint8_t, muint32_t, mmaqa_spec_uu_w_b)
__THEAD_SPEC_MMACC(mmaccus_w_b,  muint8_t, mint8_t,  mint32_t,  mmaqa_spec_us_w_b)
__THEAD_SPEC_MMACC(mmaccsu_w_b,  mint8_t,  muint8_t, mint32_t,  mmaqa_spec_su_w_b)
/* INT16 -> INT64 (single-register — maps directly to hardware) */
__THEAD_SPEC_MMACC(mmacc_d_h,    mint16_t,  mint16_t,  mint64_t,  mmaqa_spec_ss_d_h)
__THEAD_SPEC_MMACC(mmaccu_d_h,   muint16_t, muint16_t, muint64_t, mmaqa_spec_uu_d_h)
__THEAD_SPEC_MMACC(mmaccus_d_h,  muint16_t, mint16_t,  mint64_t,  mmaqa_spec_us_d_h)
__THEAD_SPEC_MMACC(mmaccsu_d_h,  mint16_t,  muint16_t, mint64_t,  mmaqa_spec_su_d_h)
/* INT16 -> INT64 spec-aligned x2 overloads (operate on both elements) */
#define __THEAD_SPEC_MMACC_X2(NAME, ATYPE, BTYPE, CTYPE, CTYPE_X2,            \
                              GET_SUFFIX, SET_SUFFIX, BUILTIN)                 \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  CTYPE_X2 __riscv_th_##NAME##_x2(CTYPE_X2 __c, ATYPE __a,                    \
                                   BTYPE __b,                                  \
                                   mrow_t __m, mcol_t __k,                     \
                                   mcol_t __n) {                               \
    CTYPE __c0 = __riscv_th_mget_##GET_SUFFIX(__c, 0);                         \
    CTYPE __c1 = __riscv_th_mget_##GET_SUFFIX(__c, 1);                         \
    CTYPE __r0 = __builtin_riscv_th_##BUILTIN(__c0, __a, __b,                  \
                                               __m, __k, __n);                \
    CTYPE __r1 = __builtin_riscv_th_##BUILTIN(__c1, __a, __b,                  \
                                               __m, __k, __n);                \
    CTYPE_X2 __result = __riscv_th_mset_##SET_SUFFIX(__c, 0, __r0);            \
    return __riscv_th_mset_##SET_SUFFIX(__result, 1, __r1);                    \
  }
__THEAD_SPEC_MMACC_X2(mmacc_d_h,    mint16_t,  mint16_t,  mint64_t,  mint64x2_t,
                       i64, i64, mmaqa_spec_ss_d_h)
__THEAD_SPEC_MMACC_X2(mmaccu_d_h,   muint16_t, muint16_t, muint64_t, muint64x2_t,
                       u64, u64, mmaqa_spec_uu_d_h)
__THEAD_SPEC_MMACC_X2(mmaccus_d_h,  muint16_t, mint16_t,  mint64_t,  mint64x2_t,
                       i64, i64, mmaqa_spec_us_d_h)
__THEAD_SPEC_MMACC_X2(mmaccsu_d_h,  mint16_t,  muint16_t, mint64_t,  mint64x2_t,
                       i64, i64, mmaqa_spec_su_d_h)
/* Partial INT8 -> INT32 (panelized) */
__THEAD_SPEC_MMACC(pmmacc_w_b,    mint8_t,  mint8_t,  mint32_t,  pmmaqa_spec_ss_w_b)
__THEAD_SPEC_MMACC(pmmaccu_w_b,   muint8_t, muint8_t, muint32_t, pmmaqa_spec_uu_w_b)
__THEAD_SPEC_MMACC(pmmaccus_w_b,  muint8_t, mint8_t,  mint32_t,  pmmaqa_spec_us_w_b)
__THEAD_SPEC_MMACC(pmmaccsu_w_b,  mint8_t,  muint8_t, mint32_t,  pmmaqa_spec_su_w_b)
/* Bypass INT */
__THEAD_SPEC_MMACC(mmacc_w_bp,    mint8_t,  mint8_t,  mint32_t,  mmaqa_spec_bp_ss)
__THEAD_SPEC_MMACC(mmaccu_w_bp,   muint8_t, muint8_t, muint32_t, mmaqa_spec_bp_uu)
/* Shorthand aliases */
#define __riscv_th_mmacc  __riscv_th_mmacc_w_b
#define __riscv_th_mmaccu __riscv_th_mmaccu_w_b

/* Backward compatibility: old names (pre-mnemonic-alignment) */
#define __riscv_th_mmaq_ss_w_b      __riscv_th_mmacc_w_b
#define __riscv_th_mmaq_uu_w_b      __riscv_th_mmaccu_w_b
#define __riscv_th_mmaq_us_w_b      __riscv_th_mmaccus_w_b
#define __riscv_th_mmaq_su_w_b      __riscv_th_mmaccsu_w_b
#define __riscv_th_mmaq_ss_d_h      __riscv_th_mmacc_d_h
#define __riscv_th_mmaq_uu_d_h      __riscv_th_mmaccu_d_h
#define __riscv_th_mmaq_us_d_h      __riscv_th_mmaccus_d_h
#define __riscv_th_mmaq_su_d_h      __riscv_th_mmaccsu_d_h
#define __riscv_th_mmaq_ss_d_h_x2   __riscv_th_mmacc_d_h_x2
#define __riscv_th_mmaq_uu_d_h_x2   __riscv_th_mmaccu_d_h_x2
#define __riscv_th_mmaq_us_d_h_x2   __riscv_th_mmaccus_d_h_x2
#define __riscv_th_mmaq_su_d_h_x2   __riscv_th_mmaccsu_d_h_x2
#define __riscv_th_mmaq_p_ss_w_b    __riscv_th_pmmacc_w_b
#define __riscv_th_mmaq_p_uu_w_b    __riscv_th_pmmaccu_w_b
#define __riscv_th_mmaq_p_us_w_b    __riscv_th_pmmaccus_w_b
#define __riscv_th_mmaq_p_su_w_b    __riscv_th_pmmaccsu_w_b
#define __riscv_th_mmaq_bp_ss       __riscv_th_mmacc_w_bp
#define __riscv_th_mmaq_bp_uu       __riscv_th_mmaccu_w_bp
#define __riscv_th_mmaq_ss          __riscv_th_mmacc_w_b
#define __riscv_th_mmaq_uu          __riscv_th_mmaccu_w_b

/* --- FP matmul: acc += A * B^T (names follow RVM 0.6 assembly: mfmacc.*) ---
 *   mfloat32_t c = __riscv_th_mfmacc_s(c, a, b, M, K, N);     // fp32
 *   mfloat16_t c = __riscv_th_mfmacc_h(c, a, b, M, K, N);     // fp16
 *   mfloat32_t c = __riscv_th_mfmacc_s_h(c, a, b, M, K, N);   // fp16→fp32 widen
 *   mfloat16_t c = __riscv_th_mfmacc_h_e4(c, a, b, M, K, N);  // fp8(E4M3)→fp16
 * Opaque source types (FP8/BF16/TF32) use mint32_t for A/B operands. */
#define __THEAD_SPEC_FMMACC(SUFFIX, ATYPE, BTYPE, CTYPE, BUILTIN)             \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  CTYPE __riscv_th_mfmacc_##SUFFIX(CTYPE __c, ATYPE __a, BTYPE __b,           \
                                    mrow_t __m, mcol_t __k, mcol_t __n) {      \
    return __builtin_riscv_th_##BUILTIN(__c, __a, __b, __m, __k, __n);         \
  }

/* Native-precision (single-register — maps directly to hardware) */
__THEAD_SPEC_FMMACC(h, mfloat16_t, mfloat16_t, mfloat16_t, mfmaqa_spec_h)
__THEAD_SPEC_FMMACC(s, mfloat32_t, mfloat32_t, mfloat32_t, mfmaqa_spec_s)
__THEAD_SPEC_FMMACC(d, mfloat64_t, mfloat64_t, mfloat64_t, mfmaqa_spec_d)
/* Widening (typed sources, single-register) */
__THEAD_SPEC_FMMACC(s_h, mfloat16_t, mfloat16_t, mfloat32_t, mfmaqa_spec_s_h)
__THEAD_SPEC_FMMACC(d_s, mfloat32_t, mfloat32_t, mfloat64_t, mfmaqa_spec_d_s)

/* Spec-aligned x2 overloads (software-level pair abstraction per intrinsic API).
   These wrap the single-register builtins, operating on both pair elements. */
static __inline__ __attribute__((__always_inline__, __nodebug__))
mfloat16x2_t __riscv_th_mfmacc_h_x2(mfloat16x2_t __c, mfloat16_t __a,
                                      mfloat16_t __b,
                                      mrow_t __m, mcol_t __k, mcol_t __n) {
  mfloat16_t __c0 = __riscv_th_mget_f16(__c, 0);
  mfloat16_t __c1 = __riscv_th_mget_f16(__c, 1);
  mfloat16_t __r0 = __builtin_riscv_th_mfmaqa_spec_h(__c0, __a, __b,
                                                       __m, __k, __n);
  mfloat16_t __r1 = __builtin_riscv_th_mfmaqa_spec_h(__c1, __a, __b,
                                                       __m, __k, __n);
  mfloat16x2_t __result = __riscv_th_mset_f16(__c, 0, __r0);
  return __riscv_th_mset_f16(__result, 1, __r1);
}
static __inline__ __attribute__((__always_inline__, __nodebug__))
mfloat64x2_t __riscv_th_mfmacc_d_x2(mfloat64x2_t __c, mfloat64_t __a,
                                      mfloat64_t __b,
                                      mrow_t __m, mcol_t __k, mcol_t __n) {
  mfloat64_t __c0 = __riscv_th_mget_f64(__c, 0);
  mfloat64_t __c1 = __riscv_th_mget_f64(__c, 1);
  mfloat64_t __r0 = __builtin_riscv_th_mfmaqa_spec_d(__c0, __a, __b,
                                                       __m, __k, __n);
  mfloat64_t __r1 = __builtin_riscv_th_mfmaqa_spec_d(__c1, __a, __b,
                                                       __m, __k, __n);
  mfloat64x2_t __result = __riscv_th_mset_f64(__c, 0, __r0);
  return __riscv_th_mset_f64(__result, 1, __r1);
}
static __inline__ __attribute__((__always_inline__, __nodebug__))
mfloat64x2_t __riscv_th_mfmacc_d_s_x2(mfloat64x2_t __c, mfloat32_t __a,
                                        mfloat32_t __b,
                                        mrow_t __m, mcol_t __k, mcol_t __n) {
  mfloat64_t __c0 = __riscv_th_mget_f64(__c, 0);
  mfloat64_t __c1 = __riscv_th_mget_f64(__c, 1);
  mfloat64_t __r0 = __builtin_riscv_th_mfmaqa_spec_d_s(__c0, __a, __b,
                                                         __m, __k, __n);
  mfloat64_t __r1 = __builtin_riscv_th_mfmaqa_spec_d_s(__c1, __a, __b,
                                                         __m, __k, __n);
  mfloat64x2_t __result = __riscv_th_mset_f64(__c, 0, __r0);
  return __riscv_th_mset_f64(__result, 1, __r1);
}

/* Widening FP matmul with opaque source types (FP8/BF16/TF32) */
#define __THEAD_SPEC_FMMACC_WIDEN(SUFFIX, CTYPE, BUILTIN)                     \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  CTYPE __riscv_th_mfmacc_##SUFFIX(CTYPE __c, mint32_t __a, mint32_t __b,     \
                                    mrow_t __m, mcol_t __k, mcol_t __n) {      \
    return __builtin_riscv_th_##BUILTIN(__c, __a, __b, __m, __k, __n);         \
  }

__THEAD_SPEC_FMMACC_WIDEN(h_e4,    mfloat16_t, mfmaqa_spec_h_e4)
__THEAD_SPEC_FMMACC_WIDEN(h_e5,    mfloat16_t, mfmaqa_spec_h_e5)
__THEAD_SPEC_FMMACC_WIDEN(bf16_e4, mfloat16_t, mfmaqa_spec_bf16_e4)
__THEAD_SPEC_FMMACC_WIDEN(bf16_e5, mfloat16_t, mfmaqa_spec_bf16_e5)
__THEAD_SPEC_FMMACC_WIDEN(s_bf16,  mfloat32_t, mfmaqa_spec_s_bf16)
__THEAD_SPEC_FMMACC_WIDEN(s_e4,    mfloat32_t, mfmaqa_spec_s_e4)
__THEAD_SPEC_FMMACC_WIDEN(s_e5,    mfloat32_t, mfmaqa_spec_s_e5)
__THEAD_SPEC_FMMACC_WIDEN(s_tf32,  mfloat32_t, mfmaqa_spec_s_tf32)

/* Backward compatibility: old FP matmul names */
#define __riscv_th_mfmaqa_h          __riscv_th_mfmacc_h
#define __riscv_th_mfmaqa_s          __riscv_th_mfmacc_s
#define __riscv_th_mfmaqa_d          __riscv_th_mfmacc_d
#define __riscv_th_mfmaqa_s_h        __riscv_th_mfmacc_s_h
#define __riscv_th_mfmaqa_d_s        __riscv_th_mfmacc_d_s
#define __riscv_th_mfmaqa_h_x2       __riscv_th_mfmacc_h_x2
#define __riscv_th_mfmaqa_d_x2       __riscv_th_mfmacc_d_x2
#define __riscv_th_mfmaqa_d_s_x2     __riscv_th_mfmacc_d_s_x2
#define __riscv_th_mfmaqa_h_e4       __riscv_th_mfmacc_h_e4
#define __riscv_th_mfmaqa_h_e5       __riscv_th_mfmacc_h_e5
#define __riscv_th_mfmaqa_bf16_e4    __riscv_th_mfmacc_bf16_e4
#define __riscv_th_mfmaqa_bf16_e5    __riscv_th_mfmacc_bf16_e5
#define __riscv_th_mfmaqa_s_bf16     __riscv_th_mfmacc_s_bf16
#define __riscv_th_mfmaqa_s_e4       __riscv_th_mfmacc_s_e4
#define __riscv_th_mfmaqa_s_e5       __riscv_th_mfmacc_s_e5
#define __riscv_th_mfmaqa_s_tf32     __riscv_th_mfmacc_s_tf32

/* --- Zero: produce a fully-zeroed matrix register ---
 *   mint32_t c = __riscv_th_mzero_i32(M, N);     // single register
 *   mint32x2_t p = __riscv_th_mzero_i32x2(M, N); // register pair
 * Emits th.mzero which zeroes all TLEN bits of the register.
 * The (m, n) params set tile config for the managed-RA model.
 * Note: primary name is 'mzeros' to avoid collision with DirectReg mzero. */
#define __THEAD_SPEC_MZERO(SUFFIX, MTYPE, BUILTIN)                             \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  MTYPE __riscv_th_mzeros_##SUFFIX(mrow_t __m, mcol_t __n) {                   \
    return __builtin_riscv_th_##BUILTIN(__m, __n);                             \
  }

__THEAD_SPEC_MZERO(i8,  mint8_t,     mzero_spec_i8)
__THEAD_SPEC_MZERO(i16, mint16_t,    mzero_spec_i16)
__THEAD_SPEC_MZERO(i32, mint32_t,    mzero_spec_i32)
__THEAD_SPEC_MZERO(i64, mint64_t,    mzero_spec_i64)
__THEAD_SPEC_MZERO(u8,  muint8_t,    mzero_spec_u8)
__THEAD_SPEC_MZERO(u16, muint16_t,   mzero_spec_u16)
__THEAD_SPEC_MZERO(u32, muint32_t,   mzero_spec_u32)
__THEAD_SPEC_MZERO(u64, muint64_t,   mzero_spec_u64)
__THEAD_SPEC_MZERO(f16, mfloat16_t,  mzero_spec_f16)
__THEAD_SPEC_MZERO(f32, mfloat32_t,  mzero_spec_f32)
__THEAD_SPEC_MZERO(f64, mfloat64_t,  mzero_spec_f64)

/* Spec-compatible aliases: __riscv_th_mzero_* (spec uses mzero, not mzeros) */
#define __riscv_th_mzero_i8(m, n)  __riscv_th_mzeros_i8(m, n)
#define __riscv_th_mzero_i16(m, n) __riscv_th_mzeros_i16(m, n)
#define __riscv_th_mzero_i32(m, n) __riscv_th_mzeros_i32(m, n)
#define __riscv_th_mzero_i64(m, n) __riscv_th_mzeros_i64(m, n)
#define __riscv_th_mzero_u8(m, n)  __riscv_th_mzeros_u8(m, n)
#define __riscv_th_mzero_u16(m, n) __riscv_th_mzeros_u16(m, n)
#define __riscv_th_mzero_u32(m, n) __riscv_th_mzeros_u32(m, n)
#define __riscv_th_mzero_u64(m, n) __riscv_th_mzeros_u64(m, n)
#define __riscv_th_mzero_f16(m, n) __riscv_th_mzeros_f16(m, n)
#define __riscv_th_mzero_f32(m, n) __riscv_th_mzeros_f32(m, n)
#define __riscv_th_mzero_f64(m, n) __riscv_th_mzeros_f64(m, n)

/* --- Zero x2 variants (register-pair) --- */
#define __THEAD_SPEC_MZERO_X2(SUFFIX, STYPE, PTYPE, SINGLE_SUFFIX,            \
                              SET_SUFFIX)                                      \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  PTYPE __riscv_th_mzeros_##SUFFIX(mrow_t __m, mcol_t __n) {                   \
    STYPE __z0 = __riscv_th_mzeros_##SINGLE_SUFFIX(__m, __n);                  \
    STYPE __z1 = __riscv_th_mzeros_##SINGLE_SUFFIX(__m, __n);                  \
    PTYPE __result = __riscv_th_mundefined_##SUFFIX();                          \
    __result = __riscv_th_mset_##SET_SUFFIX(__result, 0, __z0);                 \
    return __riscv_th_mset_##SET_SUFFIX(__result, 1, __z1);                     \
  }

__THEAD_SPEC_MZERO_X2(i8x2,  mint8_t,     mint8x2_t,    i8,  i8)
__THEAD_SPEC_MZERO_X2(i16x2, mint16_t,    mint16x2_t,   i16, i16)
__THEAD_SPEC_MZERO_X2(i32x2, mint32_t,    mint32x2_t,   i32, i32)
__THEAD_SPEC_MZERO_X2(i64x2, mint64_t,    mint64x2_t,   i64, i64)
__THEAD_SPEC_MZERO_X2(u8x2,  muint8_t,    muint8x2_t,   u8,  u8)
__THEAD_SPEC_MZERO_X2(u16x2, muint16_t,   muint16x2_t,  u16, u16)
__THEAD_SPEC_MZERO_X2(u32x2, muint32_t,   muint32x2_t,  u32, u32)
__THEAD_SPEC_MZERO_X2(u64x2, muint64_t,   muint64x2_t,  u64, u64)
__THEAD_SPEC_MZERO_X2(f16x2, mfloat16_t,  mfloat16x2_t, f16, f16)
__THEAD_SPEC_MZERO_X2(f32x2, mfloat32_t,  mfloat32x2_t, f32, f32)
__THEAD_SPEC_MZERO_X2(f64x2, mfloat64_t,  mfloat64x2_t, f64, f64)

/* Spec-compatible aliases for x2 mzero */
#define __riscv_th_mzero_i8x2(m, n)  __riscv_th_mzeros_i8x2(m, n)
#define __riscv_th_mzero_i16x2(m, n) __riscv_th_mzeros_i16x2(m, n)
#define __riscv_th_mzero_i32x2(m, n) __riscv_th_mzeros_i32x2(m, n)
#define __riscv_th_mzero_i64x2(m, n) __riscv_th_mzeros_i64x2(m, n)
#define __riscv_th_mzero_u8x2(m, n)  __riscv_th_mzeros_u8x2(m, n)
#define __riscv_th_mzero_u16x2(m, n) __riscv_th_mzeros_u16x2(m, n)
#define __riscv_th_mzero_u32x2(m, n) __riscv_th_mzeros_u32x2(m, n)
#define __riscv_th_mzero_u64x2(m, n) __riscv_th_mzeros_u64x2(m, n)
#define __riscv_th_mzero_f16x2(m, n) __riscv_th_mzeros_f16x2(m, n)
#define __riscv_th_mzero_f32x2(m, n) __riscv_th_mzeros_f32x2(m, n)
#define __riscv_th_mzero_f64x2(m, n) __riscv_th_mzeros_f64x2(m, n)

/* --- Move / Copy ---
 *   mint32_t copy = __riscv_th_mmov_mm(src);  // copy matrix register */
static __inline__ __attribute__((__always_inline__, __nodebug__))
mint32_t __riscv_th_mmov_mm(mint32_t __src) {
  return __builtin_riscv_th_mmov_mm_spec(__src);
}

/* --- Matrix-to-GPR (element extract) ---
 *   unsigned long val = __riscv_th_mmovw_x_m(src, idx);  // extract word at idx */
#define __THEAD_SPEC_MMOV_X_M(SUFFIX)                                           \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  unsigned long __riscv_th_mmov##SUFFIX##_x_m(mint32_t __src,                  \
                                               unsigned long __idx) {          \
    return __builtin_riscv_th_mmov##SUFFIX##_x_m_spec(__src, __idx);           \
  }

__THEAD_SPEC_MMOV_X_M(b)
__THEAD_SPEC_MMOV_X_M(h)
__THEAD_SPEC_MMOV_X_M(w)
__THEAD_SPEC_MMOV_X_M(d)

/* --- GPR-to-matrix (element insert) ---
 *   mint32_t r = __riscv_th_mmovw_m_x(dst, data, idx);  // insert word at idx */
#define __THEAD_SPEC_MMOV_M_X(SUFFIX)                                           \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  mint32_t __riscv_th_mmov##SUFFIX##_m_x(mint32_t __dst,                       \
                                          unsigned long __data,                \
                                          unsigned long __idx) {               \
    return __builtin_riscv_th_mmov##SUFFIX##_m_x_spec(__dst, __data, __idx);   \
  }

__THEAD_SPEC_MMOV_M_X(b)
__THEAD_SPEC_MMOV_M_X(h)
__THEAD_SPEC_MMOV_M_X(w)
__THEAD_SPEC_MMOV_M_X(d)

/* --- Duplicate GPR to matrix column ---
 *   mint32_t r = __riscv_th_mdupw_m_x(dst, data);  // broadcast word to column */
#define __THEAD_SPEC_MDUP_M_X(SUFFIX)                                           \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  mint32_t __riscv_th_mdup##SUFFIX##_m_x(mint32_t __dst,                       \
                                          unsigned long __data) {              \
    return __builtin_riscv_th_mdup##SUFFIX##_m_x_spec(__dst, __data);          \
  }

__THEAD_SPEC_MDUP_M_X(b)
__THEAD_SPEC_MDUP_M_X(h)
__THEAD_SPEC_MDUP_M_X(w)
__THEAD_SPEC_MDUP_M_X(d)

/* --- Pack: combine halves of two registers ---
 *   mint32_t r = __riscv_th_mpack(s2, s1);    // low(s2) | low(s1)
 *   mint32_t r = __riscv_th_mpackhl(s2, s1);  // high(s2) | low(s1)
 *   mint32_t r = __riscv_th_mpackhh(s2, s1);  // high(s2) | high(s1) */
static __inline__ __attribute__((__always_inline__, __nodebug__))
mint32_t __riscv_th_mpack(mint32_t __s2, mint32_t __s1) {
  return __builtin_riscv_th_mpack_spec(__s2, __s1);
}
static __inline__ __attribute__((__always_inline__, __nodebug__))
mint32_t __riscv_th_mpackhl(mint32_t __s2, mint32_t __s1) {
  return __builtin_riscv_th_mpackhl_spec(__s2, __s1);
}
static __inline__ __attribute__((__always_inline__, __nodebug__))
mint32_t __riscv_th_mpackhh(mint32_t __s2, mint32_t __s1) {
  return __builtin_riscv_th_mpackhh_spec(__s2, __s1);
}

/* --- Row slide: shift rows up/down by imm positions ---
 *   mint32_t r = __riscv_th_mrslidedown(src, imm);  // row[i] = row[i+imm]
 *   mint32_t r = __riscv_th_mrslideup(src, imm);    // row[i] = row[i-imm] */
static __inline__ __attribute__((__always_inline__, __nodebug__))
mint32_t __riscv_th_mrslidedown(mint32_t __src, unsigned int __imm) {
  return __builtin_riscv_th_mrslidedown_spec(__src, __imm);
}
static __inline__ __attribute__((__always_inline__, __nodebug__))
mint32_t __riscv_th_mrslideup(mint32_t __src, unsigned int __imm) {
  return __builtin_riscv_th_mrslideup_spec(__src, __imm);
}

/* --- Column slide: shift columns up/down by imm positions ---
 *   mint32_t r = __riscv_th_mcslidedown_w(src, imm);  // col[i] = col[i+imm]
 * Suffix b/h/w/d selects element width (8/16/32/64 bit). */
#define __THEAD_SPEC_MCSLIDE(SUFFIX, DIR)                                      \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  mint32_t __riscv_th_mcslide##DIR##_##SUFFIX(mint32_t __src,                  \
                                               unsigned int __imm) {           \
    return __builtin_riscv_th_mcslide##DIR##_##SUFFIX##_spec(__src, __imm);    \
  }

__THEAD_SPEC_MCSLIDE(b, down)
__THEAD_SPEC_MCSLIDE(h, down)
__THEAD_SPEC_MCSLIDE(w, down)
__THEAD_SPEC_MCSLIDE(d, down)
__THEAD_SPEC_MCSLIDE(b, up)
__THEAD_SPEC_MCSLIDE(h, up)
__THEAD_SPEC_MCSLIDE(w, up)
__THEAD_SPEC_MCSLIDE(d, up)

/* --- Row broadcast: replicate row[imm] to all rows ---
 *   mint32_t r = __riscv_th_mrbca(src, imm);  */
static __inline__ __attribute__((__always_inline__, __nodebug__))
mint32_t __riscv_th_mrbca(mint32_t __src, unsigned int __imm) {
  return __builtin_riscv_th_mrbca_mv_i_spec(__src, __imm);
}

/* --- Column broadcast: replicate col[imm] to all columns ---
 *   mint32_t r = __riscv_th_mcbca_w(src, imm);
 * Suffix b/h/w/d selects element width. */
#define __THEAD_SPEC_MCBCA(SUFFIX)                                             \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  mint32_t __riscv_th_mcbca_##SUFFIX(mint32_t __src, unsigned int __imm) {     \
    return __builtin_riscv_th_mcbca##SUFFIX##_mv_i_spec(__src, __imm);         \
  }

__THEAD_SPEC_MCBCA(b)
__THEAD_SPEC_MCBCA(h)
__THEAD_SPEC_MCBCA(w)
__THEAD_SPEC_MCBCA(d)

/* --- FP format conversions (unary: src -> dst) ---
 * Convert between FP formats.  'l'=lower half, 'h'=upper half for narrowing/widening.
 *   mfloat32_t w = __riscv_th_mfcvtl_s_h(fp16_src);  // FP16→FP32 (lower half)
 *   mfloat16_t n = __riscv_th_mfcvtl_h_s(fp32_src);  // FP32→FP16 (lower half)
 * Opaque types (FP8/BF16/TF32) use mint32_t. */
#define __THEAD_SPEC_FCVT(NAME, RET, ARG)                                      \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  RET __riscv_th_##NAME(ARG __src) {                                           \
    return __builtin_riscv_th_##NAME##_spec(__src);                            \
  }

/* FP8 <-> FP16 (opaque types, use mint32_t) */
__THEAD_SPEC_FCVT(mfcvtl_h_e4, mint32_t, mint32_t)
__THEAD_SPEC_FCVT(mfcvth_h_e4, mint32_t, mint32_t)
__THEAD_SPEC_FCVT(mfcvtl_h_e5, mint32_t, mint32_t)
__THEAD_SPEC_FCVT(mfcvth_h_e5, mint32_t, mint32_t)
__THEAD_SPEC_FCVT(mfcvtl_e4_h, mint32_t, mint32_t)
__THEAD_SPEC_FCVT(mfcvth_e4_h, mint32_t, mint32_t)
__THEAD_SPEC_FCVT(mfcvtl_e5_h, mint32_t, mint32_t)
__THEAD_SPEC_FCVT(mfcvth_e5_h, mint32_t, mint32_t)
/* FP16 <-> FP32 */
__THEAD_SPEC_FCVT(mfcvtl_s_h, mfloat32_t, mfloat16_t)
__THEAD_SPEC_FCVT(mfcvth_s_h, mfloat32_t, mfloat16_t)
__THEAD_SPEC_FCVT(mfcvtl_h_s, mfloat16_t, mfloat32_t)
__THEAD_SPEC_FCVT(mfcvth_h_s, mfloat16_t, mfloat32_t)
/* BF16 <-> FP32 (opaque) */
__THEAD_SPEC_FCVT(mfcvtl_s_bf16, mint32_t, mint32_t)
__THEAD_SPEC_FCVT(mfcvth_s_bf16, mint32_t, mint32_t)
__THEAD_SPEC_FCVT(mfcvtl_bf16_s, mint32_t, mint32_t)
__THEAD_SPEC_FCVT(mfcvth_bf16_s, mint32_t, mint32_t)
/* FP32 <-> FP8 (opaque) */
__THEAD_SPEC_FCVT(mfcvtl_e4_s, mint32_t, mint32_t)
__THEAD_SPEC_FCVT(mfcvth_e4_s, mint32_t, mint32_t)
__THEAD_SPEC_FCVT(mfcvtl_e5_s, mint32_t, mint32_t)
__THEAD_SPEC_FCVT(mfcvth_e5_s, mint32_t, mint32_t)
/* FP32 <-> FP64 */
__THEAD_SPEC_FCVT(mfcvtl_d_s, mfloat64_t, mfloat32_t)
__THEAD_SPEC_FCVT(mfcvth_d_s, mfloat64_t, mfloat32_t)
__THEAD_SPEC_FCVT(mfcvtl_s_d, mfloat32_t, mfloat64_t)
__THEAD_SPEC_FCVT(mfcvth_s_d, mfloat32_t, mfloat64_t)
/* TF32 <-> FP32 (opaque) */
__THEAD_SPEC_FCVT(mfcvt_s_tf32, mint32_t, mint32_t)
__THEAD_SPEC_FCVT(mfcvt_tf32_s, mint32_t, mint32_t)

/* --- Float-int conversions ---
 *   mfloat32_t f = __riscv_th_msfcvt_s_w(int32_src);  // signed INT32→FP32
 *   muint32_t  u = __riscv_th_mfucvt_w_s(fp32_src);   // FP32→unsigned INT32
 * Naming: ms=signed-to-float, mu=unsigned-to-float, mfs=float-to-signed, mfu=float-to-unsigned. */
__THEAD_SPEC_FCVT(mufcvtl_h_b, mfloat16_t, muint8_t)
__THEAD_SPEC_FCVT(mufcvth_h_b, mfloat16_t, muint8_t)
__THEAD_SPEC_FCVT(mfucvtl_b_h, muint8_t, mfloat16_t)
__THEAD_SPEC_FCVT(mfucvth_b_h, muint8_t, mfloat16_t)
__THEAD_SPEC_FCVT(msfcvtl_h_b, mfloat16_t, mint8_t)
__THEAD_SPEC_FCVT(msfcvth_h_b, mfloat16_t, mint8_t)
__THEAD_SPEC_FCVT(mfscvtl_b_h, mint8_t, mfloat16_t)
__THEAD_SPEC_FCVT(mfscvth_b_h, mint8_t, mfloat16_t)
__THEAD_SPEC_FCVT(msfcvt_s_w, mfloat32_t, mint32_t)
__THEAD_SPEC_FCVT(mufcvt_s_w, mfloat32_t, muint32_t)
__THEAD_SPEC_FCVT(mfscvt_w_s, mint32_t, mfloat32_t)
__THEAD_SPEC_FCVT(mfucvt_w_s, muint32_t, mfloat32_t)

/* --- Packed conversions (INT4↔INT8) --- */
__THEAD_SPEC_FCVT(mucvtl_b_p, mint32_t, mint32_t)
__THEAD_SPEC_FCVT(mscvtl_b_p, mint32_t, mint32_t)
__THEAD_SPEC_FCVT(mucvth_b_p, mint32_t, mint32_t)
__THEAD_SPEC_FCVT(mscvth_b_p, mint32_t, mint32_t)

/* --- N4clip: narrow-and-clip to 4-bit with rounding ---
 * Clips wider integers to 4-bit range.  'l'=low quarter, 'h'=high quarter.
 * 'u' suffix = unsigned clip.
 *   mint32_t r = __riscv_th_mn4clipl_w_mm(acc, s2, s1);      // .mm variant
 *   mint32_t r = __riscv_th_mn4clipl_w_mv_i(acc, s2, s1, imm); // .mv.i variant */
#define __THEAD_SPEC_N4CLIP_MM(NAME, BUILTIN)                                  \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  mint32_t __riscv_th_##NAME(mint32_t __acc, mint32_t __s2, mint32_t __s1) {   \
    return __builtin_riscv_th_##BUILTIN(__acc, __s2, __s1);                    \
  }

__THEAD_SPEC_N4CLIP_MM(mn4clipl_w_mm,  mn4clipl_w_mm_spec)
__THEAD_SPEC_N4CLIP_MM(mn4cliph_w_mm,  mn4cliph_w_mm_spec)
__THEAD_SPEC_N4CLIP_MM(mn4cliplu_w_mm, mn4cliplu_w_mm_spec)
__THEAD_SPEC_N4CLIP_MM(mn4cliphu_w_mm, mn4cliphu_w_mm_spec)

/* --- N4clip .mv.i: (acc, ms2, ms1, imm) -> acc --- */
#define __THEAD_SPEC_N4CLIP_MVI(NAME, BUILTIN)                                 \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  mint32_t __riscv_th_##NAME(mint32_t __acc, mint32_t __s2,                    \
                              mint32_t __s1, unsigned int __imm) {             \
    return __builtin_riscv_th_##BUILTIN(__acc, __s2, __s1, __imm);             \
  }

__THEAD_SPEC_N4CLIP_MVI(mn4clipl_w_mv_i,  mn4clipl_w_mv_i_spec)
__THEAD_SPEC_N4CLIP_MVI(mn4cliph_w_mv_i,  mn4cliph_w_mv_i_spec)
__THEAD_SPEC_N4CLIP_MVI(mn4cliplu_w_mv_i, mn4cliplu_w_mv_i_spec)
__THEAD_SPEC_N4CLIP_MVI(mn4cliphu_w_mv_i, mn4cliphu_w_mv_i_spec)

/* --- Integer element-wise .w.mm: (acc, ms2, ms1) -> acc ---
 * Element-wise operations on 32-bit integer accumulator registers.
 * .mm = matrix-matrix, .mv.i = matrix-vector with immediate row index.
 * Operations: madd, msub, mmul, mmulh, mmax, mumax, mmin, mumin, msrl, msll, msra.
 *   mint32_t r = __riscv_th_madd_w_mm(acc, s2, s1);           // md = ms1 + ms2
 *   mint32_t r = __riscv_th_msub_w_mm(acc, s2, s1);           // md = ms1 - ms2
 * NOTE: result is ms1 op ms2 (NOT acc op something); acc is tied to output register. */
#define __THEAD_SPEC_EW_INT_MM(NAME, BUILTIN)                                  \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  mint32_t __riscv_th_##NAME(mint32_t __acc, mint32_t __s2, mint32_t __s1) {   \
    return __builtin_riscv_th_##BUILTIN(__acc, __s2, __s1);                    \
  }

__THEAD_SPEC_EW_INT_MM(madd_w_mm,   madd_w_mm_spec)
__THEAD_SPEC_EW_INT_MM(msub_w_mm,   msub_w_mm_spec)
__THEAD_SPEC_EW_INT_MM(mmul_w_mm,   mmul_w_mm_spec)
__THEAD_SPEC_EW_INT_MM(mmulh_w_mm,  mmulh_w_mm_spec)
__THEAD_SPEC_EW_INT_MM(mmax_w_mm,   mmax_w_mm_spec)
__THEAD_SPEC_EW_INT_MM(mumax_w_mm,  mumax_w_mm_spec)
__THEAD_SPEC_EW_INT_MM(mmin_w_mm,   mmin_w_mm_spec)
__THEAD_SPEC_EW_INT_MM(mumin_w_mm,  mumin_w_mm_spec)
__THEAD_SPEC_EW_INT_MM(msrl_w_mm,   msrl_w_mm_spec)
__THEAD_SPEC_EW_INT_MM(msll_w_mm,   msll_w_mm_spec)
__THEAD_SPEC_EW_INT_MM(msra_w_mm,   msra_w_mm_spec)

/* --- Integer element-wise .w.mv.i: (acc, ms2, ms1, imm) -> acc --- */
#define __THEAD_SPEC_EW_INT_MVI(NAME, BUILTIN)                                 \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  mint32_t __riscv_th_##NAME(mint32_t __acc, mint32_t __s2,                    \
                              mint32_t __s1, unsigned int __imm) {             \
    return __builtin_riscv_th_##BUILTIN(__acc, __s2, __s1, __imm);             \
  }

__THEAD_SPEC_EW_INT_MVI(madd_w_mv_i,   madd_w_mv_i_spec)
__THEAD_SPEC_EW_INT_MVI(msub_w_mv_i,   msub_w_mv_i_spec)
__THEAD_SPEC_EW_INT_MVI(mmul_w_mv_i,   mmul_w_mv_i_spec)
__THEAD_SPEC_EW_INT_MVI(mmulh_w_mv_i,  mmulh_w_mv_i_spec)
__THEAD_SPEC_EW_INT_MVI(mmax_w_mv_i,   mmax_w_mv_i_spec)
__THEAD_SPEC_EW_INT_MVI(mumax_w_mv_i,  mumax_w_mv_i_spec)
__THEAD_SPEC_EW_INT_MVI(mmin_w_mv_i,   mmin_w_mv_i_spec)
__THEAD_SPEC_EW_INT_MVI(mumin_w_mv_i,  mumin_w_mv_i_spec)
__THEAD_SPEC_EW_INT_MVI(msrl_w_mv_i,   msrl_w_mv_i_spec)
__THEAD_SPEC_EW_INT_MVI(msll_w_mv_i,   msll_w_mv_i_spec)
__THEAD_SPEC_EW_INT_MVI(msra_w_mv_i,   msra_w_mv_i_spec)

/* --- FP element-wise .mm: (acc, ms2, ms1) -> acc ---
 * Element-wise FP operations on accumulator registers.  h/s/d = fp16/fp32/fp64.
 * Operations: mfadd, mfsub, mfmul, mfmax, mfmin.
 *   mfloat32_t r = __riscv_th_mfadd_s_mm(acc, s2, s1);  // md = ms1 + ms2 */
#define __THEAD_SPEC_EW_FP_MM(NAME, MTYPE, BUILTIN)                            \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  MTYPE __riscv_th_##NAME(MTYPE __acc, MTYPE __s2, MTYPE __s1) {              \
    return __builtin_riscv_th_##BUILTIN(__acc, __s2, __s1);                    \
  }

__THEAD_SPEC_EW_FP_MM(mfadd_h_mm, mfloat16_t, mfadd_h_mm_spec)
__THEAD_SPEC_EW_FP_MM(mfsub_h_mm, mfloat16_t, mfsub_h_mm_spec)
__THEAD_SPEC_EW_FP_MM(mfmul_h_mm, mfloat16_t, mfmul_h_mm_spec)
__THEAD_SPEC_EW_FP_MM(mfmax_h_mm, mfloat16_t, mfmax_h_mm_spec)
__THEAD_SPEC_EW_FP_MM(mfmin_h_mm, mfloat16_t, mfmin_h_mm_spec)
__THEAD_SPEC_EW_FP_MM(mfadd_s_mm, mfloat32_t, mfadd_s_mm_spec)
__THEAD_SPEC_EW_FP_MM(mfsub_s_mm, mfloat32_t, mfsub_s_mm_spec)
__THEAD_SPEC_EW_FP_MM(mfmul_s_mm, mfloat32_t, mfmul_s_mm_spec)
__THEAD_SPEC_EW_FP_MM(mfmax_s_mm, mfloat32_t, mfmax_s_mm_spec)
__THEAD_SPEC_EW_FP_MM(mfmin_s_mm, mfloat32_t, mfmin_s_mm_spec)
__THEAD_SPEC_EW_FP_MM(mfadd_d_mm, mfloat64_t, mfadd_d_mm_spec)
__THEAD_SPEC_EW_FP_MM(mfsub_d_mm, mfloat64_t, mfsub_d_mm_spec)
__THEAD_SPEC_EW_FP_MM(mfmul_d_mm, mfloat64_t, mfmul_d_mm_spec)
__THEAD_SPEC_EW_FP_MM(mfmax_d_mm, mfloat64_t, mfmax_d_mm_spec)
__THEAD_SPEC_EW_FP_MM(mfmin_d_mm, mfloat64_t, mfmin_d_mm_spec)

/* --- FP element-wise .mv.i: (acc, ms2, ms1, imm) -> acc --- */
#define __THEAD_SPEC_EW_FP_MVI(NAME, MTYPE, BUILTIN)                           \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  MTYPE __riscv_th_##NAME(MTYPE __acc, MTYPE __s2,                            \
                           MTYPE __s1, unsigned int __imm) {                   \
    return __builtin_riscv_th_##BUILTIN(__acc, __s2, __s1, __imm);             \
  }

__THEAD_SPEC_EW_FP_MVI(mfadd_h_mv_i, mfloat16_t, mfadd_h_mv_i_spec)
__THEAD_SPEC_EW_FP_MVI(mfsub_h_mv_i, mfloat16_t, mfsub_h_mv_i_spec)
__THEAD_SPEC_EW_FP_MVI(mfmul_h_mv_i, mfloat16_t, mfmul_h_mv_i_spec)
__THEAD_SPEC_EW_FP_MVI(mfmax_h_mv_i, mfloat16_t, mfmax_h_mv_i_spec)
__THEAD_SPEC_EW_FP_MVI(mfmin_h_mv_i, mfloat16_t, mfmin_h_mv_i_spec)
__THEAD_SPEC_EW_FP_MVI(mfadd_s_mv_i, mfloat32_t, mfadd_s_mv_i_spec)
__THEAD_SPEC_EW_FP_MVI(mfsub_s_mv_i, mfloat32_t, mfsub_s_mv_i_spec)
__THEAD_SPEC_EW_FP_MVI(mfmul_s_mv_i, mfloat32_t, mfmul_s_mv_i_spec)
__THEAD_SPEC_EW_FP_MVI(mfmax_s_mv_i, mfloat32_t, mfmax_s_mv_i_spec)
__THEAD_SPEC_EW_FP_MVI(mfmin_s_mv_i, mfloat32_t, mfmin_s_mv_i_spec)
__THEAD_SPEC_EW_FP_MVI(mfadd_d_mv_i, mfloat64_t, mfadd_d_mv_i_spec)
__THEAD_SPEC_EW_FP_MVI(mfsub_d_mv_i, mfloat64_t, mfsub_d_mv_i_spec)
__THEAD_SPEC_EW_FP_MVI(mfmul_d_mv_i, mfloat64_t, mfmul_d_mv_i_spec)
__THEAD_SPEC_EW_FP_MVI(mfmax_d_mv_i, mfloat64_t, mfmax_d_mv_i_spec)
__THEAD_SPEC_EW_FP_MVI(mfmin_d_mv_i, mfloat64_t, mfmin_d_mv_i_spec)

/* ===== Section: Zmpanel Panel-Aware 2x2 Matrix Tiling ===== */
#if defined(__riscv_xtheadzmpanel)

/* Panel-Aware CSR addresses */
#define __RVM_CSR_CUSTOM_CTRL 0xcc4
#define __RVM_CSR_BASE_ADDR_A 0xcc5
#define __RVM_CSR_BASE_ADDR_B 0xcc6
#define __RVM_CSR_BASE_ADDR_D 0xcc7
#define __RVM_CSR_RSTRIDEB_A  0xcc8
#define __RVM_CSR_RSTRIDEB_B  0xcc9
#define __RVM_CSR_RSTRIDEB_D  0xcca
#define __RVM_CSR_PANEL_M     0xccb
#define __RVM_CSR_PANEL_N     0xccc
#define __RVM_CSR_PANEL_K     0xccd
#define __RVM_CSR_MPTR_LD     0xcce
#define __RVM_CSR_NPTR_LD     0xccf
#define __RVM_CSR_KPTR_LD     0xcd0
#define __RVM_CSR_MPTR_ST     0xcd1
#define __RVM_CSR_NPTR_ST     0xcd2
#define __RVM_CSR_ADDR_A      0xcd3
#define __RVM_CSR_ADDR_B      0xcd4
#define __RVM_CSR_ADDR_D      0xcd5

/* --- Panel configuration --- */
static __inline__ __attribute__((__always_inline__, __nodebug__))
void __riscv_th_mset22adra(size_t val) {
  __builtin_riscv_th_mset22adra(val);
}

static __inline__ __attribute__((__always_inline__, __nodebug__))
void __riscv_th_mset22adrb(size_t val) {
  __builtin_riscv_th_mset22adrb(val);
}

static __inline__ __attribute__((__always_inline__, __nodebug__))
void __riscv_th_mset22adrd(size_t val) {
  __builtin_riscv_th_mset22adrd(val);
}

static __inline__ __attribute__((__always_inline__, __nodebug__))
void __riscv_th_mset22rsba(size_t val) {
  __builtin_riscv_th_mset22rsba(val);
}

static __inline__ __attribute__((__always_inline__, __nodebug__))
void __riscv_th_mset22rsbb(size_t val) {
  __builtin_riscv_th_mset22rsbb(val);
}

static __inline__ __attribute__((__always_inline__, __nodebug__))
void __riscv_th_mset22rsbd(size_t val) {
  __builtin_riscv_th_mset22rsbd(val);
}

static __inline__ __attribute__((__always_inline__, __nodebug__))
void __riscv_th_mset22m(size_t val) {
  __builtin_riscv_th_mset22m(val);
}

static __inline__ __attribute__((__always_inline__, __nodebug__))
void __riscv_th_mset22n(size_t val) {
  __builtin_riscv_th_mset22n(val);
}

static __inline__ __attribute__((__always_inline__, __nodebug__))
void __riscv_th_mset22k(size_t val) {
  __builtin_riscv_th_mset22k(val);
}

static __inline__ __attribute__((__always_inline__, __nodebug__))
void __riscv_th_msetrstptr(size_t val) {
  __builtin_riscv_th_msetrstptr(val);
}

static __inline__ __attribute__((__always_inline__, __nodebug__))
void __riscv_th_msetaccum(size_t val) {
  __builtin_riscv_th_msetaccum(val);
}

static __inline__ __attribute__((__always_inline__, __nodebug__))
void __riscv_th_msetoob(size_t val) {
  __builtin_riscv_th_msetoob(val);
}

/* --- Panel load --- */
static __inline__ __attribute__((__always_inline__, __nodebug__))
void __riscv_th_ml22e8(void) {
  __builtin_riscv_th_ml22e8();
}

static __inline__ __attribute__((__always_inline__, __nodebug__))
void __riscv_th_ml22e16(void) {
  __builtin_riscv_th_ml22e16();
}

/* --- Panel store --- */
static __inline__ __attribute__((__always_inline__, __nodebug__))
void __riscv_th_msc22e16(void) {
  __builtin_riscv_th_msc22e16();
}

static __inline__ __attribute__((__always_inline__, __nodebug__))
void __riscv_th_msc22e32(void) {
  __builtin_riscv_th_msc22e32();
}

/* --- Panel FP compute --- */
static __inline__ __attribute__((__always_inline__, __nodebug__))
void __riscv_th_mfmacc22_h_e5(void) {
  __builtin_riscv_th_mfmacc22_h_e5();
}

static __inline__ __attribute__((__always_inline__, __nodebug__))
void __riscv_th_mfmacc22_h_e4(void) {
  __builtin_riscv_th_mfmacc22_h_e4();
}

static __inline__ __attribute__((__always_inline__, __nodebug__))
void __riscv_th_mfmacc22_bf16_e5(void) {
  __builtin_riscv_th_mfmacc22_bf16_e5();
}

static __inline__ __attribute__((__always_inline__, __nodebug__))
void __riscv_th_mfmacc22_bf16_e4(void) {
  __builtin_riscv_th_mfmacc22_bf16_e4();
}

static __inline__ __attribute__((__always_inline__, __nodebug__))
void __riscv_th_mfmacc22_s_e5(void) {
  __builtin_riscv_th_mfmacc22_s_e5();
}

static __inline__ __attribute__((__always_inline__, __nodebug__))
void __riscv_th_mfmacc22_s_e4(void) {
  __builtin_riscv_th_mfmacc22_s_e4();
}

static __inline__ __attribute__((__always_inline__, __nodebug__))
void __riscv_th_mfmacc22_h(void) {
  __builtin_riscv_th_mfmacc22_h();
}

static __inline__ __attribute__((__always_inline__, __nodebug__))
void __riscv_th_mfmacc22_s_h(void) {
  __builtin_riscv_th_mfmacc22_s_h();
}

static __inline__ __attribute__((__always_inline__, __nodebug__))
void __riscv_th_mfmacc22_s_bf16(void) {
  __builtin_riscv_th_mfmacc22_s_bf16();
}

static __inline__ __attribute__((__always_inline__, __nodebug__))
void __riscv_th_mfmacc22_s(void) {
  __builtin_riscv_th_mfmacc22_s();
}

/* --- Panel INT compute --- */
static __inline__ __attribute__((__always_inline__, __nodebug__))
void __riscv_th_mmacc22_w_b(void) {
  __builtin_riscv_th_mmacc22_w_b();
}

static __inline__ __attribute__((__always_inline__, __nodebug__))
void __riscv_th_mmaccu22_w_b(void) {
  __builtin_riscv_th_mmaccu22_w_b();
}

static __inline__ __attribute__((__always_inline__, __nodebug__))
void __riscv_th_mmaccus22_w_b(void) {
  __builtin_riscv_th_mmaccus22_w_b();
}

static __inline__ __attribute__((__always_inline__, __nodebug__))
void __riscv_th_mmaccsu22_w_b(void) {
  __builtin_riscv_th_mmaccsu22_w_b();
}

#endif /* defined(__riscv_xtheadzmpanel) */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* defined(__riscv_xtheadmatrix) */

#endif /* __THEAD_MATRIX_H */
