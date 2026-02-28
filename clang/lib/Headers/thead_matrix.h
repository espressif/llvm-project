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
  RVM_CSR_XMCSR    = 0x802,
  RVM_CSR_MTILEM   = 0x803,
  RVM_CSR_MTILEN   = 0x804,
  RVM_CSR_MTILEK   = 0x805,
  RVM_CSR_XMXRM    = 0x806,
  RVM_CSR_XMSAT    = 0x807,
  RVM_CSR_XMFFLAGS = 0x808,
  RVM_CSR_XMFRM    = 0x809,
  RVM_CSR_XMSATEN  = 0x80a,
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
 * ============================================================================ */

static __inline__ __attribute__((__always_inline__, __nodebug__))
mrow_t __riscv_th_msetmrow_m(mrow_t m) {
  __builtin_riscv_th_msettilem(m);
  return m;
}

static __inline__ __attribute__((__always_inline__, __nodebug__))
mrow_t __riscv_th_msetmrow_n(mrow_t n) {
  __builtin_riscv_th_msettilen(n);
  return n;
}

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

static __inline__ __attribute__((__always_inline__, __nodebug__))
void __riscv_th_mrelease(void) {
  __builtin_riscv_th_mrelease();
}

/* ============================================================================
 * Section 3: CSR Access Functions
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

static __inline__ __attribute__((__always_inline__, __nodebug__))
unsigned long __riscv_th_xmsize(void) {
  unsigned long __val;
  __asm__ __volatile__("csrr %0, th.xmisa" : "=r"(__val));
  return __val;
}

/* ============================================================================
 * Section 4: Zero Functions (22)
 * ============================================================================ */

#define __THEAD_MZERO_SINGLE(SUFFIX, TYPE, MUNDEF)                             \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  TYPE __riscv_th_mzero_##SUFFIX(void) {                                       \
    __builtin_riscv_th_mzero(__RVM_ACC0);                                      \
    return __builtin_riscv_th_##MUNDEF();                                      \
  }

#define __THEAD_MZERO_PAIR(SUFFIX, TYPE, MUNDEF)                               \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  TYPE __riscv_th_mzero_##SUFFIX(void) {                                       \
    __builtin_riscv_th_mzero2r(__RVM_ACC0);                                    \
    return __builtin_riscv_th_##MUNDEF();                                      \
  }

__THEAD_MZERO_SINGLE(i8,  mint8_t,     mundef_i8)
__THEAD_MZERO_SINGLE(i16, mint16_t,    mundef_i16)
__THEAD_MZERO_SINGLE(i32, mint32_t,    mundef_i32)
__THEAD_MZERO_SINGLE(i64, mint64_t,    mundef_i64)
__THEAD_MZERO_SINGLE(u8,  muint8_t,    mundef_u8)
__THEAD_MZERO_SINGLE(u16, muint16_t,   mundef_u16)
__THEAD_MZERO_SINGLE(u32, muint32_t,   mundef_u32)
__THEAD_MZERO_SINGLE(u64, muint64_t,   mundef_u64)
__THEAD_MZERO_SINGLE(f16, mfloat16_t,  mundef_f16)
__THEAD_MZERO_SINGLE(f32, mfloat32_t,  mundef_f32)
__THEAD_MZERO_SINGLE(f64, mfloat64_t,  mundef_f64)

__THEAD_MZERO_PAIR(i8x2,  mint8x2_t,    mundef_i8x2)
__THEAD_MZERO_PAIR(i16x2, mint16x2_t,   mundef_i16x2)
__THEAD_MZERO_PAIR(i32x2, mint32x2_t,   mundef_i32x2)
__THEAD_MZERO_PAIR(i64x2, mint64x2_t,   mundef_i64x2)
__THEAD_MZERO_PAIR(u8x2,  muint8x2_t,   mundef_u8x2)
__THEAD_MZERO_PAIR(u16x2, muint16x2_t,  mundef_u16x2)
__THEAD_MZERO_PAIR(u32x2, muint32x2_t,  mundef_u32x2)
__THEAD_MZERO_PAIR(u64x2, muint64x2_t,  mundef_u64x2)
__THEAD_MZERO_PAIR(f16x2, mfloat16x2_t, mundef_f16x2)
__THEAD_MZERO_PAIR(f32x2, mfloat32x2_t, mundef_f32x2)
__THEAD_MZERO_PAIR(f64x2, mfloat64x2_t, mundef_f64x2)

/* ============================================================================
 * Section 5: Undefined Functions (22)
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
 * With opaque native types, reinterpret returns a mundef token of the target
 * type. The hardware doesn't need to move data -- types are just annotations.
 *
 * The spec defines reinterpret(src) with a source parameter. Since C cannot
 * overload on different source types, we use macros that accept any expression
 * as src and simply discard it (the cast is a type-level operation only).
 * ============================================================================ */

#define __riscv_th_mreinterpret_i8(src)    ((void)(src), __builtin_riscv_th_mundef_i8())
#define __riscv_th_mreinterpret_i16(src)   ((void)(src), __builtin_riscv_th_mundef_i16())
#define __riscv_th_mreinterpret_i32(src)   ((void)(src), __builtin_riscv_th_mundef_i32())
#define __riscv_th_mreinterpret_i64(src)   ((void)(src), __builtin_riscv_th_mundef_i64())
#define __riscv_th_mreinterpret_u8(src)    ((void)(src), __builtin_riscv_th_mundef_u8())
#define __riscv_th_mreinterpret_u16(src)   ((void)(src), __builtin_riscv_th_mundef_u16())
#define __riscv_th_mreinterpret_u32(src)   ((void)(src), __builtin_riscv_th_mundef_u32())
#define __riscv_th_mreinterpret_u64(src)   ((void)(src), __builtin_riscv_th_mundef_u64())
#define __riscv_th_mreinterpret_f16(src)   ((void)(src), __builtin_riscv_th_mundef_f16())
#define __riscv_th_mreinterpret_f32(src)   ((void)(src), __builtin_riscv_th_mundef_f32())
#define __riscv_th_mreinterpret_f64(src)   ((void)(src), __builtin_riscv_th_mundef_f64())

#define __riscv_th_mreinterpret_i8x2(src)  ((void)(src), __builtin_riscv_th_mundef_i8x2())
#define __riscv_th_mreinterpret_i16x2(src) ((void)(src), __builtin_riscv_th_mundef_i16x2())
#define __riscv_th_mreinterpret_i32x2(src) ((void)(src), __builtin_riscv_th_mundef_i32x2())
#define __riscv_th_mreinterpret_i64x2(src) ((void)(src), __builtin_riscv_th_mundef_i64x2())
#define __riscv_th_mreinterpret_u8x2(src)  ((void)(src), __builtin_riscv_th_mundef_u8x2())
#define __riscv_th_mreinterpret_u16x2(src) ((void)(src), __builtin_riscv_th_mundef_u16x2())
#define __riscv_th_mreinterpret_u32x2(src) ((void)(src), __builtin_riscv_th_mundef_u32x2())
#define __riscv_th_mreinterpret_u64x2(src) ((void)(src), __builtin_riscv_th_mundef_u64x2())
#define __riscv_th_mreinterpret_f16x2(src) ((void)(src), __builtin_riscv_th_mundef_f16x2())
#define __riscv_th_mreinterpret_f32x2(src) ((void)(src), __builtin_riscv_th_mundef_f32x2())
#define __riscv_th_mreinterpret_f64x2(src) ((void)(src), __builtin_riscv_th_mundef_f64x2())

/* Reinterpret macros defined above -- no instantiations needed */

/* ============================================================================
 * Section 7: Tuple Get/Set Functions (22)
 *
 * With opaque native types, tuple operations return mundef tokens.
 * ============================================================================ */

#define __THEAD_MGET(SUFFIX, STYPE, PTYPE, MUNDEF_S)                           \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  STYPE __riscv_th_mget_##SUFFIX(PTYPE __src, size_t __index) {                \
    (void)__src; (void)__index;                                                \
    return __builtin_riscv_th_##MUNDEF_S();                                    \
  }

#define __THEAD_MSET(SUFFIX, STYPE, PTYPE, MUNDEF_P)                           \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  PTYPE __riscv_th_mset_##SUFFIX(PTYPE __src, size_t __index,                  \
                                 STYPE __val) {                                \
    (void)__src; (void)__index; (void)__val;                                   \
    return __builtin_riscv_th_##MUNDEF_P();                                    \
  }

__THEAD_MGET(i8,  mint8_t,     mint8x2_t,    mundef_i8)
__THEAD_MGET(i16, mint16_t,    mint16x2_t,   mundef_i16)
__THEAD_MGET(i32, mint32_t,    mint32x2_t,   mundef_i32)
__THEAD_MGET(i64, mint64_t,    mint64x2_t,   mundef_i64)
__THEAD_MGET(u8,  muint8_t,    muint8x2_t,   mundef_u8)
__THEAD_MGET(u16, muint16_t,   muint16x2_t,  mundef_u16)
__THEAD_MGET(u32, muint32_t,   muint32x2_t,  mundef_u32)
__THEAD_MGET(u64, muint64_t,   muint64x2_t,  mundef_u64)
__THEAD_MGET(f16, mfloat16_t,  mfloat16x2_t, mundef_f16)
__THEAD_MGET(f32, mfloat32_t,  mfloat32x2_t, mundef_f32)
__THEAD_MGET(f64, mfloat64_t,  mfloat64x2_t, mundef_f64)

__THEAD_MSET(i8,  mint8_t,     mint8x2_t,    mundef_i8x2)
__THEAD_MSET(i16, mint16_t,    mint16x2_t,   mundef_i16x2)
__THEAD_MSET(i32, mint32_t,    mint32x2_t,   mundef_i32x2)
__THEAD_MSET(i64, mint64_t,    mint64x2_t,   mundef_i64x2)
__THEAD_MSET(u8,  muint8_t,    muint8x2_t,   mundef_u8x2)
__THEAD_MSET(u16, muint16_t,   muint16x2_t,  mundef_u16x2)
__THEAD_MSET(u32, muint32_t,   muint32x2_t,  mundef_u32x2)
__THEAD_MSET(u64, muint64_t,   muint64x2_t,  mundef_u64x2)
__THEAD_MSET(f16, mfloat16_t,  mfloat16x2_t, mundef_f16x2)
__THEAD_MSET(f32, mfloat32_t,  mfloat32x2_t, mundef_f32x2)
__THEAD_MSET(f64, mfloat64_t,  mfloat64x2_t, mundef_f64x2)

/* ============================================================================
 * Section 8: Load Functions
 *
 * Role-specific loads:
 *   mld_a_TYPE  : Load A matrix (to tr0, sets M rows, K cols)
 *   mld_b_TYPE  : Load B matrix (to tr1, sets K rows, N cols)
 *   mld_c_TYPE  : Load C/accumulator (to acc0, sets M rows, N cols)
 * Transposed variants: mld_at_TYPE, mld_bt_TYPE, mld_ct_TYPE
 * Whole-register load: mld_whole_TYPE
 *
 * EEW (element width) is encoded in the suffix: i8/i16/i32/i64/u8/u16/u32/u64
 * FP types: f16/f32/f64 map to e16/e32/e64
 *
 * Load builtins return Qm types (signed int by EEW). For unsigned/float API
 * types, the load is called for its side effect and mundef returns the token.
 * ============================================================================ */

/* Element-stride loads: A matrix (tr0) */
#define __THEAD_MLD_A(SUFFIX, TYPE, EEW, MUNDEF)                               \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  TYPE __riscv_th_mld_a_##SUFFIX(const void *__base, long __stride,            \
                                 mrow_t __m, mcol_t __k) {                     \
    __builtin_riscv_th_msettilem(__m);                                         \
    __builtin_riscv_th_msettilek(__k);                                         \
    __builtin_riscv_th_mlae##EEW(__RVM_TR0, (void *)__base, (size_t)__stride); \
    return __builtin_riscv_th_##MUNDEF();                                      \
  }

/* Element-stride loads: B matrix (tr1) */
#define __THEAD_MLD_B(SUFFIX, TYPE, EEW, MUNDEF)                               \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  TYPE __riscv_th_mld_b_##SUFFIX(const void *__base, long __stride,            \
                                 mrow_t __k, mcol_t __n) {                     \
    __builtin_riscv_th_msettilek(__k);                                         \
    __builtin_riscv_th_msettilen(__n);                                         \
    __builtin_riscv_th_mlbe##EEW(__RVM_TR1, (void *)__base, (size_t)__stride); \
    return __builtin_riscv_th_##MUNDEF();                                      \
  }

/* Element-stride loads: C/accumulator (acc0) */
#define __THEAD_MLD_C(SUFFIX, TYPE, EEW, MUNDEF)                               \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  TYPE __riscv_th_mld_c_##SUFFIX(const void *__base, long __stride,            \
                                 mrow_t __m, mcol_t __n) {                     \
    __builtin_riscv_th_msettilem(__m);                                         \
    __builtin_riscv_th_msettilen(__n);                                         \
    __builtin_riscv_th_mlce##EEW(__RVM_ACC0, (void *)__base, (size_t)__stride); \
    return __builtin_riscv_th_##MUNDEF();                                      \
  }

/* Transposed loads: A-transposed (tr2) */
#define __THEAD_MLD_AT(SUFFIX, TYPE, EEW, MUNDEF)                              \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  TYPE __riscv_th_mld_at_##SUFFIX(const void *__base, long __stride,           \
                                  mrow_t __m, mcol_t __k) {                    \
    __builtin_riscv_th_msettilem(__m);                                         \
    __builtin_riscv_th_msettilek(__k);                                         \
    __builtin_riscv_th_mlate##EEW(__RVM_TR2, (void *)__base, (size_t)__stride); \
    return __builtin_riscv_th_##MUNDEF();                                      \
  }

/* Transposed loads: B-transposed (tr3) */
#define __THEAD_MLD_BT(SUFFIX, TYPE, EEW, MUNDEF)                              \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  TYPE __riscv_th_mld_bt_##SUFFIX(const void *__base, long __stride,           \
                                  mrow_t __k, mcol_t __n) {                    \
    __builtin_riscv_th_msettilek(__k);                                         \
    __builtin_riscv_th_msettilen(__n);                                         \
    __builtin_riscv_th_mlbte##EEW(__RVM_TR3, (void *)__base, (size_t)__stride); \
    return __builtin_riscv_th_##MUNDEF();                                      \
  }

/* Transposed loads: C-transposed (acc1) */
#define __THEAD_MLD_CT(SUFFIX, TYPE, EEW, MUNDEF)                              \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  TYPE __riscv_th_mld_ct_##SUFFIX(const void *__base, long __stride,           \
                                  mrow_t __m, mcol_t __n) {                    \
    __builtin_riscv_th_msettilem(__m);                                         \
    __builtin_riscv_th_msettilen(__n);                                         \
    __builtin_riscv_th_mlcte##EEW(__RVM_ACC1, (void *)__base, (size_t)__stride); \
    return __builtin_riscv_th_##MUNDEF();                                      \
  }

/* Whole-register loads */
#define __THEAD_MLD_WHOLE(SUFFIX, TYPE, EEW, MUNDEF)                           \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  TYPE __riscv_th_mld_whole_##SUFFIX(const void *__base) {                     \
    __builtin_riscv_th_mlme##EEW(__RVM_TR0, (void *)__base);                   \
    return __builtin_riscv_th_##MUNDEF();                                      \
  }

/* Instantiate loads for all types */
/* INT8 */
__THEAD_MLD_A(i8,  mint8_t,  8, mundef_i8)
__THEAD_MLD_B(i8,  mint8_t,  8, mundef_i8)
__THEAD_MLD_C(i8,  mint8_t,  8, mundef_i8)
__THEAD_MLD_AT(i8, mint8_t,  8, mundef_i8)
__THEAD_MLD_BT(i8, mint8_t,  8, mundef_i8)
__THEAD_MLD_CT(i8, mint8_t,  8, mundef_i8)
__THEAD_MLD_WHOLE(i8, mint8_t, 8, mundef_i8)
/* INT16 */
__THEAD_MLD_A(i16,  mint16_t,  16, mundef_i16)
__THEAD_MLD_B(i16,  mint16_t,  16, mundef_i16)
__THEAD_MLD_C(i16,  mint16_t,  16, mundef_i16)
__THEAD_MLD_AT(i16, mint16_t,  16, mundef_i16)
__THEAD_MLD_BT(i16, mint16_t,  16, mundef_i16)
__THEAD_MLD_CT(i16, mint16_t,  16, mundef_i16)
__THEAD_MLD_WHOLE(i16, mint16_t, 16, mundef_i16)
/* INT32 */
__THEAD_MLD_A(i32,  mint32_t,  32, mundef_i32)
__THEAD_MLD_B(i32,  mint32_t,  32, mundef_i32)
__THEAD_MLD_C(i32,  mint32_t,  32, mundef_i32)
__THEAD_MLD_AT(i32, mint32_t,  32, mundef_i32)
__THEAD_MLD_BT(i32, mint32_t,  32, mundef_i32)
__THEAD_MLD_CT(i32, mint32_t,  32, mundef_i32)
__THEAD_MLD_WHOLE(i32, mint32_t, 32, mundef_i32)
/* INT64 */
__THEAD_MLD_A(i64,  mint64_t,  64, mundef_i64)
__THEAD_MLD_B(i64,  mint64_t,  64, mundef_i64)
__THEAD_MLD_C(i64,  mint64_t,  64, mundef_i64)
__THEAD_MLD_AT(i64, mint64_t,  64, mundef_i64)
__THEAD_MLD_BT(i64, mint64_t,  64, mundef_i64)
__THEAD_MLD_CT(i64, mint64_t,  64, mundef_i64)
__THEAD_MLD_WHOLE(i64, mint64_t, 64, mundef_i64)
/* UINT8 */
__THEAD_MLD_A(u8,  muint8_t,  8, mundef_u8)
__THEAD_MLD_B(u8,  muint8_t,  8, mundef_u8)
__THEAD_MLD_C(u8,  muint8_t,  8, mundef_u8)
__THEAD_MLD_AT(u8, muint8_t,  8, mundef_u8)
__THEAD_MLD_BT(u8, muint8_t,  8, mundef_u8)
__THEAD_MLD_CT(u8, muint8_t,  8, mundef_u8)
__THEAD_MLD_WHOLE(u8, muint8_t, 8, mundef_u8)
/* UINT16 */
__THEAD_MLD_A(u16,  muint16_t,  16, mundef_u16)
__THEAD_MLD_B(u16,  muint16_t,  16, mundef_u16)
__THEAD_MLD_C(u16,  muint16_t,  16, mundef_u16)
__THEAD_MLD_AT(u16, muint16_t,  16, mundef_u16)
__THEAD_MLD_BT(u16, muint16_t,  16, mundef_u16)
__THEAD_MLD_CT(u16, muint16_t,  16, mundef_u16)
__THEAD_MLD_WHOLE(u16, muint16_t, 16, mundef_u16)
/* UINT32 */
__THEAD_MLD_A(u32,  muint32_t,  32, mundef_u32)
__THEAD_MLD_B(u32,  muint32_t,  32, mundef_u32)
__THEAD_MLD_C(u32,  muint32_t,  32, mundef_u32)
__THEAD_MLD_AT(u32, muint32_t,  32, mundef_u32)
__THEAD_MLD_BT(u32, muint32_t,  32, mundef_u32)
__THEAD_MLD_CT(u32, muint32_t,  32, mundef_u32)
__THEAD_MLD_WHOLE(u32, muint32_t, 32, mundef_u32)
/* UINT64 */
__THEAD_MLD_A(u64,  muint64_t,  64, mundef_u64)
__THEAD_MLD_B(u64,  muint64_t,  64, mundef_u64)
__THEAD_MLD_C(u64,  muint64_t,  64, mundef_u64)
__THEAD_MLD_AT(u64, muint64_t,  64, mundef_u64)
__THEAD_MLD_BT(u64, muint64_t,  64, mundef_u64)
__THEAD_MLD_CT(u64, muint64_t,  64, mundef_u64)
__THEAD_MLD_WHOLE(u64, muint64_t, 64, mundef_u64)
/* FP16 */
__THEAD_MLD_A(f16,  mfloat16_t,  16, mundef_f16)
__THEAD_MLD_B(f16,  mfloat16_t,  16, mundef_f16)
__THEAD_MLD_C(f16,  mfloat16_t,  16, mundef_f16)
__THEAD_MLD_AT(f16, mfloat16_t,  16, mundef_f16)
__THEAD_MLD_BT(f16, mfloat16_t,  16, mundef_f16)
__THEAD_MLD_CT(f16, mfloat16_t,  16, mundef_f16)
__THEAD_MLD_WHOLE(f16, mfloat16_t, 16, mundef_f16)
/* FP32 */
__THEAD_MLD_A(f32,  mfloat32_t,  32, mundef_f32)
__THEAD_MLD_B(f32,  mfloat32_t,  32, mundef_f32)
__THEAD_MLD_C(f32,  mfloat32_t,  32, mundef_f32)
__THEAD_MLD_AT(f32, mfloat32_t,  32, mundef_f32)
__THEAD_MLD_BT(f32, mfloat32_t,  32, mundef_f32)
__THEAD_MLD_CT(f32, mfloat32_t,  32, mundef_f32)
__THEAD_MLD_WHOLE(f32, mfloat32_t, 32, mundef_f32)
/* FP64 */
__THEAD_MLD_A(f64,  mfloat64_t,  64, mundef_f64)
__THEAD_MLD_B(f64,  mfloat64_t,  64, mundef_f64)
__THEAD_MLD_C(f64,  mfloat64_t,  64, mundef_f64)
__THEAD_MLD_AT(f64, mfloat64_t,  64, mundef_f64)
__THEAD_MLD_BT(f64, mfloat64_t,  64, mundef_f64)
__THEAD_MLD_CT(f64, mfloat64_t,  64, mundef_f64)
__THEAD_MLD_WHOLE(f64, mfloat64_t, 64, mundef_f64)

/* ============================================================================
 * Section 9: Store Functions
 *
 * Role-specific stores only (no generic dispatch -- opaque types have no tag).
 * ============================================================================ */

/* Role-specific stores */
#define __THEAD_MST_A(SUFFIX, TYPE, EEW)                                       \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  void __riscv_th_mst_a_##SUFFIX(void *__base, long __stride,                  \
                                 TYPE __val, mrow_t __m, mcol_t __k) {         \
    (void)__val;                                                               \
    __builtin_riscv_th_msettilem(__m);                                         \
    __builtin_riscv_th_msettilek(__k);                                         \
    __builtin_riscv_th_msae##EEW(__RVM_TR0, __base, (size_t)__stride);         \
  }

#define __THEAD_MST_B(SUFFIX, TYPE, EEW)                                       \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  void __riscv_th_mst_b_##SUFFIX(void *__base, long __stride,                  \
                                 TYPE __val, mrow_t __k, mcol_t __n) {         \
    (void)__val;                                                               \
    __builtin_riscv_th_msettilek(__k);                                         \
    __builtin_riscv_th_msettilen(__n);                                         \
    __builtin_riscv_th_msbe##EEW(__RVM_TR1, __base, (size_t)__stride);         \
  }

#define __THEAD_MST_C(SUFFIX, TYPE, EEW)                                       \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  void __riscv_th_mst_c_##SUFFIX(void *__base, long __stride,                  \
                                 TYPE __val, mrow_t __m, mcol_t __n) {         \
    (void)__val;                                                               \
    __builtin_riscv_th_msettilem(__m);                                         \
    __builtin_riscv_th_msettilen(__n);                                         \
    __builtin_riscv_th_msce##EEW(__RVM_ACC0, __base, (size_t)__stride);        \
  }

/* Transposed stores */
#define __THEAD_MST_AT(SUFFIX, TYPE, EEW)                                      \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  void __riscv_th_mst_at_##SUFFIX(void *__base, long __stride,                 \
                                  TYPE __val, mrow_t __m, mcol_t __k) {        \
    (void)__val;                                                               \
    __builtin_riscv_th_msettilem(__m);                                         \
    __builtin_riscv_th_msettilek(__k);                                         \
    __builtin_riscv_th_msate##EEW(__RVM_TR2, __base, (size_t)__stride);        \
  }

#define __THEAD_MST_BT(SUFFIX, TYPE, EEW)                                      \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  void __riscv_th_mst_bt_##SUFFIX(void *__base, long __stride,                 \
                                  TYPE __val, mrow_t __k, mcol_t __n) {        \
    (void)__val;                                                               \
    __builtin_riscv_th_msettilek(__k);                                         \
    __builtin_riscv_th_msettilen(__n);                                         \
    __builtin_riscv_th_msbte##EEW(__RVM_TR3, __base, (size_t)__stride);        \
  }

#define __THEAD_MST_CT(SUFFIX, TYPE, EEW)                                      \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  void __riscv_th_mst_ct_##SUFFIX(void *__base, long __stride,                 \
                                  TYPE __val, mrow_t __m, mcol_t __n) {        \
    (void)__val;                                                               \
    __builtin_riscv_th_msettilem(__m);                                         \
    __builtin_riscv_th_msettilen(__n);                                         \
    __builtin_riscv_th_mscte##EEW(__RVM_ACC1, __base, (size_t)__stride);       \
  }

/* Whole-register store */
#define __THEAD_MST_WHOLE(SUFFIX, TYPE, EEW)                                   \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  void __riscv_th_mst_whole_##SUFFIX(void *__base, TYPE __val) {               \
    (void)__val;                                                               \
    __builtin_riscv_th_msme##EEW(__RVM_TR0, __base);                           \
  }

/* Instantiate stores for all types */
#define __THEAD_MST_ALL(SUFFIX, TYPE, EEW)                                     \
  __THEAD_MST_A(SUFFIX, TYPE, EEW)                                            \
  __THEAD_MST_B(SUFFIX, TYPE, EEW)                                            \
  __THEAD_MST_C(SUFFIX, TYPE, EEW)                                            \
  __THEAD_MST_AT(SUFFIX, TYPE, EEW)                                           \
  __THEAD_MST_BT(SUFFIX, TYPE, EEW)                                           \
  __THEAD_MST_CT(SUFFIX, TYPE, EEW)                                           \
  __THEAD_MST_WHOLE(SUFFIX, TYPE, EEW)

__THEAD_MST_ALL(i8,  mint8_t,     8)
__THEAD_MST_ALL(i16, mint16_t,    16)
__THEAD_MST_ALL(i32, mint32_t,    32)
__THEAD_MST_ALL(i64, mint64_t,    64)
__THEAD_MST_ALL(u8,  muint8_t,    8)
__THEAD_MST_ALL(u16, muint16_t,   16)
__THEAD_MST_ALL(u32, muint32_t,   32)
__THEAD_MST_ALL(u64, muint64_t,   64)
__THEAD_MST_ALL(f16, mfloat16_t,  16)
__THEAD_MST_ALL(f32, mfloat32_t,  32)
__THEAD_MST_ALL(f64, mfloat64_t,  64)

/* ============================================================================
 * Section 10: FP Matrix Multiply-Accumulate (13 functions)
 *
 * All matmul: acc0 = acc0 + tr1 * tr0
 * Params: dest (acc), src1 (A/tr0), src2 (B/tr1), row1(M), row2(K), col(N)
 * ============================================================================ */

/* Typed FP matmul (builtins take and return Qm types) */
#define __THEAD_FMMACC(SUFFIX, ATYPE, BTYPE, CTYPE, BUILTIN)                  \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  CTYPE __riscv_th_fmmacc_##SUFFIX(CTYPE __dest, ATYPE __a, BTYPE __b,        \
                                   mrow_t __m, mrow_t __k, mcol_t __n) {      \
    __builtin_riscv_th_msettilem(__m);                                         \
    __builtin_riscv_th_msettilek(__k);                                         \
    __builtin_riscv_th_msettilen(__n);                                         \
    return BUILTIN(__RVM_ACC0, __RVM_TR1, __RVM_TR0, __dest, __a, __b);        \
  }

/* Native-precision FP matmul */
__THEAD_FMMACC(h,   mfloat16_t, mfloat16_t, mfloat16_t, __builtin_riscv_th_mfmacc_h)
__THEAD_FMMACC(s,   mfloat32_t, mfloat32_t, mfloat32_t, __builtin_riscv_th_mfmacc_s)
__THEAD_FMMACC(d,   mfloat64_t, mfloat64_t, mfloat64_t, __builtin_riscv_th_mfmacc_d)

/* Typed widening FP matmul (builtins take and return Qm types) */
#define __THEAD_FWMMACC_TYPED(SUFFIX, ATYPE, BTYPE, CTYPE, BUILTIN)           \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  CTYPE __riscv_th_fwmmacc_##SUFFIX(CTYPE __dest, ATYPE __a, BTYPE __b,       \
                                    mrow_t __m, mrow_t __k, mcol_t __n) {     \
    __builtin_riscv_th_msettilem(__m);                                         \
    __builtin_riscv_th_msettilek(__k);                                         \
    __builtin_riscv_th_msettilen(__n);                                         \
    return BUILTIN(__RVM_ACC0, __RVM_TR1, __RVM_TR0, __dest, __a, __b);        \
  }

/* Untyped widening FP matmul (void builtins for FP8/BF16/TF32) */
#define __THEAD_FWMMACC(SUFFIX, ATYPE, BTYPE, CTYPE, BUILTIN, MUNDEF)         \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  CTYPE __riscv_th_fwmmacc_##SUFFIX(CTYPE __dest, ATYPE __a, BTYPE __b,       \
                                    mrow_t __m, mrow_t __k, mcol_t __n) {     \
    (void)__dest; (void)__a; (void)__b;                                        \
    __builtin_riscv_th_msettilem(__m);                                         \
    __builtin_riscv_th_msettilek(__k);                                         \
    __builtin_riscv_th_msettilen(__n);                                         \
    BUILTIN(__RVM_ACC0, __RVM_TR1, __RVM_TR0);                                 \
    return __builtin_riscv_th_##MUNDEF();                                      \
  }

/* FP8 -> FP16 widening */
__THEAD_FWMMACC(h_e4,     muint8_t,    muint8_t,    mfloat16_t,  __builtin_riscv_th_mfmacc_h_e4,  mundef_f16)
__THEAD_FWMMACC(h_e5,     muint8_t,    muint8_t,    mfloat16_t,  __builtin_riscv_th_mfmacc_h_e5,  mundef_f16)
/* FP8 -> BF16 widening */
__THEAD_FWMMACC(bf16_e4,  muint8_t,    muint8_t,    mfloat16_t,  __builtin_riscv_th_mfmacc_bf16_e4, mundef_f16)
__THEAD_FWMMACC(bf16_e5,  muint8_t,    muint8_t,    mfloat16_t,  __builtin_riscv_th_mfmacc_bf16_e5, mundef_f16)
/* FP16 -> FP32 widening (typed) */
__THEAD_FWMMACC_TYPED(s_h, mfloat16_t, mfloat16_t, mfloat32_t, __builtin_riscv_th_mfmacc_s_h)
/* BF16/FP8/TF32 -> FP32 widening */
__THEAD_FWMMACC(s_bf16,   mfloat16_t,  mfloat16_t,  mfloat32_t,  __builtin_riscv_th_mfmacc_s_bf16, mundef_f32)
__THEAD_FWMMACC(s_e4,     muint8_t,    muint8_t,    mfloat32_t,  __builtin_riscv_th_mfmacc_s_e4,  mundef_f32)
__THEAD_FWMMACC(s_e5,     muint8_t,    muint8_t,    mfloat32_t,  __builtin_riscv_th_mfmacc_s_e5,  mundef_f32)
__THEAD_FWMMACC(s_tf32,   mfloat32_t,  mfloat32_t,  mfloat32_t,  __builtin_riscv_th_mfmacc_s_tf32, mundef_f32)
/* FP32 -> FP64 widening (typed) */
__THEAD_FWMMACC_TYPED(d_s, mfloat32_t, mfloat32_t, mfloat64_t, __builtin_riscv_th_mfmacc_d_s)

/* ============================================================================
 * Section 11: Integer Matrix Multiply-Accumulate (14 functions)
 *
 * All INT matmul builtins are now typed. For unsigned-unsigned (uu) variants,
 * the accumulator type matches the builtin (unsigned).
 * ============================================================================ */

#define __THEAD_MMAQA(SUFFIX, ATYPE, BTYPE, CTYPE, BUILTIN)                   \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  CTYPE __riscv_th_mmaqa_##SUFFIX(CTYPE __dest, ATYPE __a, BTYPE __b,         \
                                  mrow_t __m, mrow_t __k, mcol_t __n) {       \
    __builtin_riscv_th_msettilem(__m);                                         \
    __builtin_riscv_th_msettilek(__k);                                         \
    __builtin_riscv_th_msettilen(__n);                                         \
    return BUILTIN(__RVM_ACC0, __RVM_TR1, __RVM_TR0, __dest, __a, __b);        \
  }

/* INT8 -> INT32 (4 sign variants) */
__THEAD_MMAQA(ss_w_b,  mint8_t,   mint8_t,   mint32_t,   __builtin_riscv_th_mmacc_w_b)
__THEAD_MMAQA(uu_w_b,  muint8_t,  muint8_t,  muint32_t,  __builtin_riscv_th_mmaccu_w_b)
__THEAD_MMAQA(us_w_b,  muint8_t,  mint8_t,   mint32_t,   __builtin_riscv_th_mmaccus_w_b)
__THEAD_MMAQA(su_w_b,  mint8_t,   muint8_t,  mint32_t,   __builtin_riscv_th_mmaccsu_w_b)

/* INT16 -> INT64 (4 sign variants) */
__THEAD_MMAQA(ss_d_h,  mint16_t,  mint16_t,  mint64_t,   __builtin_riscv_th_mmacc_d_h)
__THEAD_MMAQA(uu_d_h,  muint16_t, muint16_t, muint64_t,  __builtin_riscv_th_mmaccu_d_h)
__THEAD_MMAQA(us_d_h,  muint16_t, mint16_t,  mint64_t,   __builtin_riscv_th_mmaccus_d_h)
__THEAD_MMAQA(su_d_h,  mint16_t,  muint16_t, mint64_t,   __builtin_riscv_th_mmaccsu_d_h)

/* Partial INT matmul (INT8 -> INT32, 4 sign variants) */
#define __THEAD_PMMAQA(SUFFIX, ATYPE, BTYPE, CTYPE, BUILTIN)                  \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  CTYPE __riscv_th_pmmaqa_##SUFFIX(CTYPE __dest, ATYPE __a, BTYPE __b,        \
                                   mrow_t __m, mrow_t __k, mcol_t __n) {      \
    __builtin_riscv_th_msettilem(__m);                                         \
    __builtin_riscv_th_msettilek(__k);                                         \
    __builtin_riscv_th_msettilen(__n);                                         \
    return BUILTIN(__RVM_ACC0, __RVM_TR1, __RVM_TR0, __dest, __a, __b);        \
  }

__THEAD_PMMAQA(ss_w_b, mint8_t,  mint8_t,  mint32_t,  __builtin_riscv_th_pmmacc_w_b)
__THEAD_PMMAQA(uu_w_b, muint8_t, muint8_t, muint32_t, __builtin_riscv_th_pmmaccu_w_b)
__THEAD_PMMAQA(us_w_b, muint8_t, mint8_t,  mint32_t,  __builtin_riscv_th_pmmaccus_w_b)
__THEAD_PMMAQA(su_w_b, mint8_t,  muint8_t, mint32_t,  __builtin_riscv_th_pmmaccsu_w_b)

/* Bypass INT matmul (2 functions) */
static __inline__ __attribute__((__always_inline__, __nodebug__))
mint32_t __riscv_th_mmaqa_bp_ss(mint32_t __dest, mint8_t __a, mint8_t __b,
                                mrow_t __m, mrow_t __k, mcol_t __n) {
  __builtin_riscv_th_msettilem(__m);
  __builtin_riscv_th_msettilek(__k);
  __builtin_riscv_th_msettilen(__n);
  return __builtin_riscv_th_mmacc_w_bp(__RVM_ACC0, __RVM_TR1, __RVM_TR0,
                                       __dest, __a, __b);
}

static __inline__ __attribute__((__always_inline__, __nodebug__))
muint32_t __riscv_th_mmaqa_bp_uu(muint32_t __dest, muint8_t __a, muint8_t __b,
                                 mrow_t __m, mrow_t __k, mcol_t __n) {
  __builtin_riscv_th_msettilem(__m);
  __builtin_riscv_th_msettilek(__k);
  __builtin_riscv_th_msettilen(__n);
  return __builtin_riscv_th_mmaccu_w_bp(__RVM_ACC0, __RVM_TR1, __RVM_TR0,
                                        __dest, __a, __b);
}

/* ============================================================================
 * Section 12: Integer Element-Wise Arithmetic (22 functions)
 *
 * All EW ops use accumulator registers: Md=ACC0, Ms1=ACC1, Ms2=ACC2
 * MM variants: typed Qm2(Qm2, Qm2, Qm2)
 * MV.I variants: typed Qm2(Qm2, Qm2, unsigned int) -- must be macros (ImmArg)
 * ============================================================================ */

#define __THEAD_INT_EW_MM(NAME, BUILTIN)                                       \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  mint32_t __riscv_th_##NAME##_w_mm(mint32_t __dest, mint32_t __src1,          \
                                    mint32_t __src2) {                         \
    return BUILTIN(__RVM_ACC0, __RVM_ACC2, __RVM_ACC1,                         \
                   __dest, __src1, __src2);                                     \
  }

#define __THEAD_INT_EW_MVI(NAME, BUILTIN)                                      \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  mint32_t __riscv_th_##NAME##_w_mv(mint32_t __dest, mint32_t __src1,          \
                                    unsigned int __imm)                        \
      __attribute__((__unavailable__(                                           \
          "use macro form: __riscv_th_" #NAME "_w_mv(d, s, imm)")));

/* Macro forms for ImmArg int EW MV.I functions */
#define __riscv_th_madd_w_mv(d, s1, imm)                                      \
  __extension__({                                                              \
    __builtin_riscv_th_madd_w_mv_i(__RVM_ACC0, __RVM_ACC2, __RVM_ACC1,        \
                                   (d), (s1), (imm));                          \
  })
#define __riscv_th_msub_w_mv(d, s1, imm)                                      \
  __extension__({                                                              \
    __builtin_riscv_th_msub_w_mv_i(__RVM_ACC0, __RVM_ACC2, __RVM_ACC1,        \
                                   (d), (s1), (imm));                          \
  })
#define __riscv_th_mmul_w_mv(d, s1, imm)                                      \
  __extension__({                                                              \
    __builtin_riscv_th_mmul_w_mv_i(__RVM_ACC0, __RVM_ACC2, __RVM_ACC1,        \
                                   (d), (s1), (imm));                          \
  })
#define __riscv_th_mmulh_w_mv(d, s1, imm)                                     \
  __extension__({                                                              \
    __builtin_riscv_th_mmulh_w_mv_i(__RVM_ACC0, __RVM_ACC2, __RVM_ACC1,       \
                                    (d), (s1), (imm));                         \
  })
#define __riscv_th_mmax_w_mv(d, s1, imm)                                      \
  __extension__({                                                              \
    __builtin_riscv_th_mmax_w_mv_i(__RVM_ACC0, __RVM_ACC2, __RVM_ACC1,        \
                                   (d), (s1), (imm));                          \
  })
#define __riscv_th_mumax_w_mv(d, s1, imm)                                     \
  __extension__({                                                              \
    __builtin_riscv_th_mumax_w_mv_i(__RVM_ACC0, __RVM_ACC2, __RVM_ACC1,       \
                                    (d), (s1), (imm));                         \
  })
#define __riscv_th_mmin_w_mv(d, s1, imm)                                      \
  __extension__({                                                              \
    __builtin_riscv_th_mmin_w_mv_i(__RVM_ACC0, __RVM_ACC2, __RVM_ACC1,        \
                                   (d), (s1), (imm));                          \
  })
#define __riscv_th_mumin_w_mv(d, s1, imm)                                     \
  __extension__({                                                              \
    __builtin_riscv_th_mumin_w_mv_i(__RVM_ACC0, __RVM_ACC2, __RVM_ACC1,       \
                                    (d), (s1), (imm));                         \
  })
#define __riscv_th_msrl_w_mv(d, s1, imm)                                      \
  __extension__({                                                              \
    __builtin_riscv_th_msrl_w_mv_i(__RVM_ACC0, __RVM_ACC2, __RVM_ACC1,        \
                                   (d), (s1), (imm));                          \
  })
#define __riscv_th_msll_w_mv(d, s1, imm)                                      \
  __extension__({                                                              \
    __builtin_riscv_th_msll_w_mv_i(__RVM_ACC0, __RVM_ACC2, __RVM_ACC1,        \
                                   (d), (s1), (imm));                          \
  })
#define __riscv_th_msra_w_mv(d, s1, imm)                                      \
  __extension__({                                                              \
    __builtin_riscv_th_msra_w_mv_i(__RVM_ACC0, __RVM_ACC2, __RVM_ACC1,        \
                                   (d), (s1), (imm));                          \
  })

__THEAD_INT_EW_MM(madd,   __builtin_riscv_th_madd_w_mm)
__THEAD_INT_EW_MM(msub,   __builtin_riscv_th_msub_w_mm)
__THEAD_INT_EW_MM(mmul,   __builtin_riscv_th_mmul_w_mm)
__THEAD_INT_EW_MM(mmulh,  __builtin_riscv_th_mmulh_w_mm)
__THEAD_INT_EW_MM(mmax,   __builtin_riscv_th_mmax_w_mm)
__THEAD_INT_EW_MM(mumax,  __builtin_riscv_th_mumax_w_mm)
__THEAD_INT_EW_MM(mmin,   __builtin_riscv_th_mmin_w_mm)
__THEAD_INT_EW_MM(mumin,  __builtin_riscv_th_mumin_w_mm)
__THEAD_INT_EW_MM(msrl,   __builtin_riscv_th_msrl_w_mm)
__THEAD_INT_EW_MM(msll,   __builtin_riscv_th_msll_w_mm)
__THEAD_INT_EW_MM(msra,   __builtin_riscv_th_msra_w_mm)

/* ============================================================================
 * Section 13: FP Element-Wise Arithmetic (30 functions)
 * ============================================================================ */

#define __THEAD_FP_EW_MM(NAME, FPSUFFIX, FPTYPE, BUILTIN)                      \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  FPTYPE __riscv_th_##NAME##_##FPSUFFIX##_mm(                                  \
      FPTYPE __dest, FPTYPE __src1, FPTYPE __src2) {                           \
    return BUILTIN(__RVM_ACC0, __RVM_ACC2, __RVM_ACC1,                         \
                   __dest, __src1, __src2);                                     \
  }

/* FP add */
__THEAD_FP_EW_MM(mfadd,  h, mfloat16_t, __builtin_riscv_th_mfadd_h_mm)
__THEAD_FP_EW_MM(mfadd,  s, mfloat32_t, __builtin_riscv_th_mfadd_s_mm)
__THEAD_FP_EW_MM(mfadd,  d, mfloat64_t, __builtin_riscv_th_mfadd_d_mm)
/* FP sub */
__THEAD_FP_EW_MM(mfsub,  h, mfloat16_t, __builtin_riscv_th_mfsub_h_mm)
__THEAD_FP_EW_MM(mfsub,  s, mfloat32_t, __builtin_riscv_th_mfsub_s_mm)
__THEAD_FP_EW_MM(mfsub,  d, mfloat64_t, __builtin_riscv_th_mfsub_d_mm)
/* FP mul */
__THEAD_FP_EW_MM(mfmul,  h, mfloat16_t, __builtin_riscv_th_mfmul_h_mm)
__THEAD_FP_EW_MM(mfmul,  s, mfloat32_t, __builtin_riscv_th_mfmul_s_mm)
__THEAD_FP_EW_MM(mfmul,  d, mfloat64_t, __builtin_riscv_th_mfmul_d_mm)
/* FP max */
__THEAD_FP_EW_MM(mfmax,  h, mfloat16_t, __builtin_riscv_th_mfmax_h_mm)
__THEAD_FP_EW_MM(mfmax,  s, mfloat32_t, __builtin_riscv_th_mfmax_s_mm)
__THEAD_FP_EW_MM(mfmax,  d, mfloat64_t, __builtin_riscv_th_mfmax_d_mm)
/* FP min */
__THEAD_FP_EW_MM(mfmin,  h, mfloat16_t, __builtin_riscv_th_mfmin_h_mm)
__THEAD_FP_EW_MM(mfmin,  s, mfloat32_t, __builtin_riscv_th_mfmin_s_mm)
__THEAD_FP_EW_MM(mfmin,  d, mfloat64_t, __builtin_riscv_th_mfmin_d_mm)

/* FP EW MV.I macro forms (ImmArg requires compile-time constants) */
#define __riscv_th_mfadd_h_mv(d, s1, imm)                                    \
  __extension__({ __builtin_riscv_th_mfadd_h_mv_i(                           \
      __RVM_ACC0, __RVM_ACC2, __RVM_ACC1, (d), (s1), (imm)); })
#define __riscv_th_mfadd_s_mv(d, s1, imm)                                    \
  __extension__({ __builtin_riscv_th_mfadd_s_mv_i(                           \
      __RVM_ACC0, __RVM_ACC2, __RVM_ACC1, (d), (s1), (imm)); })
#define __riscv_th_mfadd_d_mv(d, s1, imm)                                    \
  __extension__({ __builtin_riscv_th_mfadd_d_mv_i(                           \
      __RVM_ACC0, __RVM_ACC2, __RVM_ACC1, (d), (s1), (imm)); })
#define __riscv_th_mfsub_h_mv(d, s1, imm)                                    \
  __extension__({ __builtin_riscv_th_mfsub_h_mv_i(                           \
      __RVM_ACC0, __RVM_ACC2, __RVM_ACC1, (d), (s1), (imm)); })
#define __riscv_th_mfsub_s_mv(d, s1, imm)                                    \
  __extension__({ __builtin_riscv_th_mfsub_s_mv_i(                           \
      __RVM_ACC0, __RVM_ACC2, __RVM_ACC1, (d), (s1), (imm)); })
#define __riscv_th_mfsub_d_mv(d, s1, imm)                                    \
  __extension__({ __builtin_riscv_th_mfsub_d_mv_i(                           \
      __RVM_ACC0, __RVM_ACC2, __RVM_ACC1, (d), (s1), (imm)); })
#define __riscv_th_mfmul_h_mv(d, s1, imm)                                    \
  __extension__({ __builtin_riscv_th_mfmul_h_mv_i(                           \
      __RVM_ACC0, __RVM_ACC2, __RVM_ACC1, (d), (s1), (imm)); })
#define __riscv_th_mfmul_s_mv(d, s1, imm)                                    \
  __extension__({ __builtin_riscv_th_mfmul_s_mv_i(                           \
      __RVM_ACC0, __RVM_ACC2, __RVM_ACC1, (d), (s1), (imm)); })
#define __riscv_th_mfmul_d_mv(d, s1, imm)                                    \
  __extension__({ __builtin_riscv_th_mfmul_d_mv_i(                           \
      __RVM_ACC0, __RVM_ACC2, __RVM_ACC1, (d), (s1), (imm)); })
#define __riscv_th_mfmax_h_mv(d, s1, imm)                                    \
  __extension__({ __builtin_riscv_th_mfmax_h_mv_i(                           \
      __RVM_ACC0, __RVM_ACC2, __RVM_ACC1, (d), (s1), (imm)); })
#define __riscv_th_mfmax_s_mv(d, s1, imm)                                    \
  __extension__({ __builtin_riscv_th_mfmax_s_mv_i(                           \
      __RVM_ACC0, __RVM_ACC2, __RVM_ACC1, (d), (s1), (imm)); })
#define __riscv_th_mfmax_d_mv(d, s1, imm)                                    \
  __extension__({ __builtin_riscv_th_mfmax_d_mv_i(                           \
      __RVM_ACC0, __RVM_ACC2, __RVM_ACC1, (d), (s1), (imm)); })
#define __riscv_th_mfmin_h_mv(d, s1, imm)                                    \
  __extension__({ __builtin_riscv_th_mfmin_h_mv_i(                           \
      __RVM_ACC0, __RVM_ACC2, __RVM_ACC1, (d), (s1), (imm)); })
#define __riscv_th_mfmin_s_mv(d, s1, imm)                                    \
  __extension__({ __builtin_riscv_th_mfmin_s_mv_i(                           \
      __RVM_ACC0, __RVM_ACC2, __RVM_ACC1, (d), (s1), (imm)); })
#define __riscv_th_mfmin_d_mv(d, s1, imm)                                    \
  __extension__({ __builtin_riscv_th_mfmin_d_mv_i(                           \
      __RVM_ACC0, __RVM_ACC2, __RVM_ACC1, (d), (s1), (imm)); })

/* ============================================================================
 * Section 14: FP Format Conversions (26 functions)
 * ============================================================================ */

/* Typed conversions (builtins take/return Qm types) */
#define __THEAD_MFCVT_TYPED(NAME, DTYPE, STYPE, BUILTIN)                      \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  DTYPE __riscv_th_##NAME(STYPE __src) {                                       \
    return BUILTIN(__RVM_ACC0, __RVM_ACC1, __src);                             \
  }

/* Untyped conversions (void builtins for FP8/BF16/TF32) */
#define __THEAD_MFCVT(NAME, DTYPE, STYPE, BUILTIN, MUNDEF)                    \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  DTYPE __riscv_th_##NAME(STYPE __src) {                                       \
    (void)__src;                                                               \
    BUILTIN(__RVM_ACC0, __RVM_ACC1);                                           \
    return __builtin_riscv_th_##MUNDEF();                                      \
  }

/* FP8(e4m3) <-> FP16 */
__THEAD_MFCVT(mfcvtl_h_e4,  mfloat16_t, muint8_t,    __builtin_riscv_th_mfcvtl_h_e4,  mundef_f16)
__THEAD_MFCVT(mfcvth_h_e4,  mfloat16_t, muint8_t,    __builtin_riscv_th_mfcvth_h_e4,  mundef_f16)
__THEAD_MFCVT(mfcvtl_h_e5,  mfloat16_t, muint8_t,    __builtin_riscv_th_mfcvtl_h_e5,  mundef_f16)
__THEAD_MFCVT(mfcvth_h_e5,  mfloat16_t, muint8_t,    __builtin_riscv_th_mfcvth_h_e5,  mundef_f16)
__THEAD_MFCVT(mfcvtl_e4_h,  muint8_t,   mfloat16_t,  __builtin_riscv_th_mfcvtl_e4_h,  mundef_u8)
__THEAD_MFCVT(mfcvth_e4_h,  muint8_t,   mfloat16_t,  __builtin_riscv_th_mfcvth_e4_h,  mundef_u8)
__THEAD_MFCVT(mfcvtl_e5_h,  muint8_t,   mfloat16_t,  __builtin_riscv_th_mfcvtl_e5_h,  mundef_u8)
__THEAD_MFCVT(mfcvth_e5_h,  muint8_t,   mfloat16_t,  __builtin_riscv_th_mfcvth_e5_h,  mundef_u8)
/* FP16 <-> FP32 (typed) */
__THEAD_MFCVT_TYPED(mfcvtl_s_h,   mfloat32_t, mfloat16_t,  __builtin_riscv_th_mfcvtl_s_h)
__THEAD_MFCVT_TYPED(mfcvth_s_h,   mfloat32_t, mfloat16_t,  __builtin_riscv_th_mfcvth_s_h)
__THEAD_MFCVT_TYPED(mfcvtl_h_s,   mfloat16_t, mfloat32_t,  __builtin_riscv_th_mfcvtl_h_s)
__THEAD_MFCVT_TYPED(mfcvth_h_s,   mfloat16_t, mfloat32_t,  __builtin_riscv_th_mfcvth_h_s)
/* BF16 <-> FP32 */
__THEAD_MFCVT(mfcvtl_s_bf16, mfloat32_t, mfloat16_t, __builtin_riscv_th_mfcvtl_s_bf16, mundef_f32)
__THEAD_MFCVT(mfcvth_s_bf16, mfloat32_t, mfloat16_t, __builtin_riscv_th_mfcvth_s_bf16, mundef_f32)
__THEAD_MFCVT(mfcvtl_bf16_s, mfloat16_t, mfloat32_t, __builtin_riscv_th_mfcvtl_bf16_s, mundef_f16)
__THEAD_MFCVT(mfcvth_bf16_s, mfloat16_t, mfloat32_t, __builtin_riscv_th_mfcvth_bf16_s, mundef_f16)
/* FP32 <-> FP8 */
__THEAD_MFCVT(mfcvtl_e4_s,  muint8_t,   mfloat32_t,  __builtin_riscv_th_mfcvtl_e4_s,  mundef_u8)
__THEAD_MFCVT(mfcvth_e4_s,  muint8_t,   mfloat32_t,  __builtin_riscv_th_mfcvth_e4_s,  mundef_u8)
__THEAD_MFCVT(mfcvtl_e5_s,  muint8_t,   mfloat32_t,  __builtin_riscv_th_mfcvtl_e5_s,  mundef_u8)
__THEAD_MFCVT(mfcvth_e5_s,  muint8_t,   mfloat32_t,  __builtin_riscv_th_mfcvth_e5_s,  mundef_u8)
/* FP32 <-> FP64 (typed) */
__THEAD_MFCVT_TYPED(mfcvtl_d_s,   mfloat64_t, mfloat32_t,  __builtin_riscv_th_mfcvtl_d_s)
__THEAD_MFCVT_TYPED(mfcvth_d_s,   mfloat64_t, mfloat32_t,  __builtin_riscv_th_mfcvth_d_s)
__THEAD_MFCVT_TYPED(mfcvtl_s_d,   mfloat32_t, mfloat64_t,  __builtin_riscv_th_mfcvtl_s_d)
__THEAD_MFCVT_TYPED(mfcvth_s_d,   mfloat32_t, mfloat64_t,  __builtin_riscv_th_mfcvth_s_d)
/* TF32 <-> FP32 */
__THEAD_MFCVT(mfcvt_s_tf32, mfloat32_t, mfloat32_t,  __builtin_riscv_th_mfcvt_s_tf32, mundef_f32)
__THEAD_MFCVT(mfcvt_tf32_s, mfloat32_t, mfloat32_t,  __builtin_riscv_th_mfcvt_tf32_s, mundef_f32)

/* ============================================================================
 * Section 15: Float-Integer Conversions (12 functions)
 * ============================================================================ */

/* UINT8 <-> FP16 (typed) */
__THEAD_MFCVT_TYPED(mufcvtl_h_b,  mfloat16_t, muint8_t,    __builtin_riscv_th_mufcvtl_h_b)
__THEAD_MFCVT_TYPED(mufcvth_h_b,  mfloat16_t, muint8_t,    __builtin_riscv_th_mufcvth_h_b)
__THEAD_MFCVT_TYPED(mfucvtl_b_h,  muint8_t,   mfloat16_t,  __builtin_riscv_th_mfucvtl_b_h)
__THEAD_MFCVT_TYPED(mfucvth_b_h,  muint8_t,   mfloat16_t,  __builtin_riscv_th_mfucvth_b_h)
/* SINT8 <-> FP16 (typed) */
__THEAD_MFCVT_TYPED(msfcvtl_h_b,  mfloat16_t, mint8_t,     __builtin_riscv_th_msfcvtl_h_b)
__THEAD_MFCVT_TYPED(msfcvth_h_b,  mfloat16_t, mint8_t,     __builtin_riscv_th_msfcvth_h_b)
__THEAD_MFCVT_TYPED(mfscvtl_b_h,  mint8_t,    mfloat16_t,  __builtin_riscv_th_mfscvtl_b_h)
__THEAD_MFCVT_TYPED(mfscvth_b_h,  mint8_t,    mfloat16_t,  __builtin_riscv_th_mfscvth_b_h)
/* INT32 <-> FP32 (typed) */
__THEAD_MFCVT_TYPED(msfcvt_s_w,   mfloat32_t, mint32_t,    __builtin_riscv_th_msfcvt_s_w)
__THEAD_MFCVT_TYPED(mufcvt_s_w,   mfloat32_t, muint32_t,   __builtin_riscv_th_mufcvt_s_w)
__THEAD_MFCVT_TYPED(mfscvt_w_s,   mint32_t,   mfloat32_t,  __builtin_riscv_th_mfscvt_w_s)
__THEAD_MFCVT_TYPED(mfucvt_w_s,   muint32_t,  mfloat32_t,  __builtin_riscv_th_mfucvt_w_s)

/* ============================================================================
 * Section 16: Packed Conversions (4 functions)
 * ============================================================================ */

__THEAD_MFCVT(mucvtl_b_p,   muint8_t,   muint8_t,    __builtin_riscv_th_mucvtl_b_p,   mundef_u8)
__THEAD_MFCVT(mscvtl_b_p,   mint8_t,    muint8_t,    __builtin_riscv_th_mscvtl_b_p,   mundef_i8)
__THEAD_MFCVT(mucvth_b_p,   muint8_t,   muint8_t,    __builtin_riscv_th_mucvth_b_p,   mundef_u8)
__THEAD_MFCVT(mscvth_b_p,   mint8_t,    muint8_t,    __builtin_riscv_th_mscvth_b_p,   mundef_i8)

/* ============================================================================
 * Section 17: N4Clip Functions (8 functions)
 * ============================================================================ */

/* N4Clip MM variants (signed: return mint8_t, unsigned: return muint8_t) */
#define __THEAD_N4CLIP_MM(NAME, RTYPE, BUILTIN, MUNDEF)                        \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  RTYPE __riscv_th_##NAME(mint32_t __src1, mint32_t __src2) {                  \
    (void)__src1; (void)__src2;                                                \
    BUILTIN(__RVM_ACC0, __RVM_ACC2, __RVM_ACC1);                               \
    return __builtin_riscv_th_##MUNDEF();                                      \
  }

__THEAD_N4CLIP_MM(mn4clipl_w_mm,   mint8_t,  __builtin_riscv_th_mn4clipl_w_mm,  mundef_i8)
__THEAD_N4CLIP_MM(mn4cliph_w_mm,   mint8_t,  __builtin_riscv_th_mn4cliph_w_mm,  mundef_i8)
__THEAD_N4CLIP_MM(mn4cliplu_w_mm,  muint8_t, __builtin_riscv_th_mn4cliplu_w_mm, mundef_u8)
__THEAD_N4CLIP_MM(mn4cliphu_w_mm,  muint8_t, __builtin_riscv_th_mn4cliphu_w_mm, mundef_u8)

/* N4Clip MV.I macro forms (ImmArg requires compile-time constants) */
/* Signed variants return mint8_t, unsigned variants return muint8_t */
#define __riscv_th_mn4clipl_w_mv(s1, imm)                                     \
  __extension__({ (void)(s1); __builtin_riscv_th_mn4clipl_w_mv_i(             \
      __RVM_ACC0, __RVM_ACC2, __RVM_ACC1, (imm));                             \
      __builtin_riscv_th_mundef_i8(); })
#define __riscv_th_mn4cliph_w_mv(s1, imm)                                     \
  __extension__({ (void)(s1); __builtin_riscv_th_mn4cliph_w_mv_i(             \
      __RVM_ACC0, __RVM_ACC2, __RVM_ACC1, (imm));                             \
      __builtin_riscv_th_mundef_i8(); })
#define __riscv_th_mn4cliplu_w_mv(s1, imm)                                    \
  __extension__({ (void)(s1); __builtin_riscv_th_mn4cliplu_w_mv_i(            \
      __RVM_ACC0, __RVM_ACC2, __RVM_ACC1, (imm));                             \
      __builtin_riscv_th_mundef_u8(); })
#define __riscv_th_mn4cliphu_w_mv(s1, imm)                                    \
  __extension__({ (void)(s1); __builtin_riscv_th_mn4cliphu_w_mv_i(            \
      __RVM_ACC0, __RVM_ACC2, __RVM_ACC1, (imm));                             \
      __builtin_riscv_th_mundef_u8(); })

/* ============================================================================
 * Section 18: Move / Duplicate Functions (13 functions)
 * ============================================================================ */

/* mmov_mm: copy matrix register (md = ms1) */
static __inline__ __attribute__((__always_inline__, __nodebug__))
void __riscv_th_mmov_mm(void) {
  __builtin_riscv_th_mmov_mm(__RVM_TR0, __RVM_TR1);
}

/* mmov_x_m: extract element from matrix to GPR */
#define __THEAD_MMOV_X_M(SUFFIX, STYPE, BUILTIN)                              \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  size_t __riscv_th_mmov_x_m_##SUFFIX(STYPE __src, size_t __index) {          \
    (void)__src;                                                               \
    return BUILTIN(__RVM_TR0, __index);                                         \
  }

__THEAD_MMOV_X_M(b, mint8_t,  __builtin_riscv_th_mmovb_x_m)
__THEAD_MMOV_X_M(h, mint16_t, __builtin_riscv_th_mmovh_x_m)
__THEAD_MMOV_X_M(w, mint32_t, __builtin_riscv_th_mmovw_x_m)
__THEAD_MMOV_X_M(d, mint64_t, __builtin_riscv_th_mmovd_x_m)

/* mmov_m_x: insert scalar into matrix element */
#define __THEAD_MMOV_M_X(SUFFIX, DTYPE, BUILTIN, MUNDEF)                      \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  DTYPE __riscv_th_mmov_m_x_##SUFFIX(DTYPE __dest, size_t __data,             \
                                     size_t __index) {                         \
    (void)__dest;                                                              \
    BUILTIN(__RVM_TR0, __data, __index);                                        \
    return __builtin_riscv_th_##MUNDEF();                                      \
  }

__THEAD_MMOV_M_X(b, mint8_t,  __builtin_riscv_th_mmovb_m_x, mundef_i8)
__THEAD_MMOV_M_X(h, mint16_t, __builtin_riscv_th_mmovh_m_x, mundef_i16)
__THEAD_MMOV_M_X(w, mint32_t, __builtin_riscv_th_mmovw_m_x, mundef_i32)
__THEAD_MMOV_M_X(d, mint64_t, __builtin_riscv_th_mmovd_m_x, mundef_i64)

/* mdup_m_x: broadcast scalar to all elements in a matrix column */
#define __THEAD_MDUP(SUFFIX, DTYPE, BUILTIN, MUNDEF)                           \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  DTYPE __riscv_th_mdup_m_x_##SUFFIX(size_t __data) {                         \
    BUILTIN(__RVM_TR0, __data);                                                 \
    return __builtin_riscv_th_##MUNDEF();                                      \
  }

__THEAD_MDUP(b, mint8_t,  __builtin_riscv_th_mdupb_m_x, mundef_i8)
__THEAD_MDUP(h, mint16_t, __builtin_riscv_th_mduph_m_x, mundef_i16)
__THEAD_MDUP(w, mint32_t, __builtin_riscv_th_mdupw_m_x, mundef_i32)
__THEAD_MDUP(d, mint64_t, __builtin_riscv_th_mdupd_m_x, mundef_i64)

/* ============================================================================
 * Section 19: Pack Functions (3 functions)
 * ============================================================================ */

static __inline__ __attribute__((__always_inline__, __nodebug__))
void __riscv_th_mpack(void) {
  __builtin_riscv_th_mpack(__RVM_TR0, __RVM_TR2, __RVM_TR1);
}

static __inline__ __attribute__((__always_inline__, __nodebug__))
void __riscv_th_mpackhl(void) {
  __builtin_riscv_th_mpackhl(__RVM_TR0, __RVM_TR2, __RVM_TR1);
}

static __inline__ __attribute__((__always_inline__, __nodebug__))
void __riscv_th_mpackhh(void) {
  __builtin_riscv_th_mpackhh(__RVM_TR0, __RVM_TR2, __RVM_TR1);
}

/* ============================================================================
 * Section 20: Slide Functions (10 functions)
 * ============================================================================ */

/* Row slide (ImmArg: must be compile-time constants) */
#define __riscv_th_mrslidedown(imm) \
  __builtin_riscv_th_mrslidedown(__RVM_TR0, __RVM_TR1, (imm))
#define __riscv_th_mrslideup(imm) \
  __builtin_riscv_th_mrslideup(__RVM_TR0, __RVM_TR1, (imm))

/* Column slide (ImmArg: must be compile-time constants) */
#define __riscv_th_mcslidedown_b(imm) \
  __builtin_riscv_th_mcslidedown_b(__RVM_TR0, __RVM_TR1, (imm))
#define __riscv_th_mcslideup_b(imm) \
  __builtin_riscv_th_mcslideup_b(__RVM_TR0, __RVM_TR1, (imm))
#define __riscv_th_mcslidedown_h(imm) \
  __builtin_riscv_th_mcslidedown_h(__RVM_TR0, __RVM_TR1, (imm))
#define __riscv_th_mcslideup_h(imm) \
  __builtin_riscv_th_mcslideup_h(__RVM_TR0, __RVM_TR1, (imm))
#define __riscv_th_mcslidedown_w(imm) \
  __builtin_riscv_th_mcslidedown_w(__RVM_TR0, __RVM_TR1, (imm))
#define __riscv_th_mcslideup_w(imm) \
  __builtin_riscv_th_mcslideup_w(__RVM_TR0, __RVM_TR1, (imm))
#define __riscv_th_mcslidedown_d(imm) \
  __builtin_riscv_th_mcslidedown_d(__RVM_TR0, __RVM_TR1, (imm))
#define __riscv_th_mcslideup_d(imm) \
  __builtin_riscv_th_mcslideup_d(__RVM_TR0, __RVM_TR1, (imm))

/* ============================================================================
 * Section 21: Broadcast Functions (5 functions)
 * ============================================================================ */

/* Row broadcast (ImmArg: must be compile-time constants) */
#define __riscv_th_mrbca(imm) \
  __builtin_riscv_th_mrbca_mv_i(__RVM_TR0, __RVM_TR1, (imm))

/* Column broadcast (ImmArg: must be compile-time constants) */
#define __riscv_th_mcbca_b(imm) \
  __builtin_riscv_th_mcbcab_mv_i(__RVM_TR0, __RVM_TR1, (imm))
#define __riscv_th_mcbca_h(imm) \
  __builtin_riscv_th_mcbcah_mv_i(__RVM_TR0, __RVM_TR1, (imm))
#define __riscv_th_mcbca_w(imm) \
  __builtin_riscv_th_mcbcaw_mv_i(__RVM_TR0, __RVM_TR1, (imm))
#define __riscv_th_mcbca_d(imm) \
  __builtin_riscv_th_mcbcad_mv_i(__RVM_TR0, __RVM_TR1, (imm))

/* ============================================================================
 * Section 22: mmov_mv - Matrix-Vector Move (row extract)
 * ============================================================================ */

/* mmov_mv: not directly available as a single builtin in RVM 0.6.
 * Use mrbca (row broadcast) + mmov_x_m to extract individual rows.
 * This is documented as a limitation. */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* defined(__riscv_xtheadmatrix) */

#endif /* __THEAD_MATRIX_H */
