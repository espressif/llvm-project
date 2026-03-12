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
mcol_t __riscv_th_msetmrow_n(mcol_t n) {
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
 * management is needed.
 * ============================================================================ */

/* --- A-tile loads (mlae: M×K dimensions) --- */
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

/* --- B-tile loads (mlbe: K×N dimensions) --- */
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

/* --- Accumulator loads (mlce: M×N dimensions) --- */
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

/* --- Stores (msce: M×N dimensions) --- */
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

/* --- INT matmul: acc = acc + A * B --- */
/* Note: uses 'mmaq' to avoid collision with DirectReg mmaqa */
#define __THEAD_SPEC_MMAQA(SUFFIX, ATYPE, BTYPE, CTYPE, BUILTIN)              \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  CTYPE __riscv_th_mmaq_##SUFFIX(CTYPE __c, ATYPE __a, BTYPE __b,             \
                                  mrow_t __m, mcol_t __k, mcol_t __n) {        \
    return __builtin_riscv_th_##BUILTIN(__c, __a, __b, __m, __k, __n);         \
  }

/* INT8 -> INT32 */
__THEAD_SPEC_MMAQA(ss_w_b, mint8_t,  mint8_t,  mint32_t,  mmaqa_spec_ss_w_b)
__THEAD_SPEC_MMAQA(uu_w_b, muint8_t, muint8_t, muint32_t, mmaqa_spec_uu_w_b)
__THEAD_SPEC_MMAQA(us_w_b, muint8_t, mint8_t,  mint32_t,  mmaqa_spec_us_w_b)
__THEAD_SPEC_MMAQA(su_w_b, mint8_t,  muint8_t, mint32_t,  mmaqa_spec_su_w_b)
/* INT16 -> INT64 (single-register — maps directly to hardware) */
__THEAD_SPEC_MMAQA(ss_d_h, mint16_t,  mint16_t,  mint64_t,  mmaqa_spec_ss_d_h)
__THEAD_SPEC_MMAQA(uu_d_h, muint16_t, muint16_t, muint64_t, mmaqa_spec_uu_d_h)
__THEAD_SPEC_MMAQA(us_d_h, muint16_t, mint16_t,  mint64_t,  mmaqa_spec_us_d_h)
__THEAD_SPEC_MMAQA(su_d_h, mint16_t,  muint16_t, mint64_t,  mmaqa_spec_su_d_h)
/* INT16 -> INT64 spec-aligned x2 overloads */
#define __THEAD_SPEC_MMAQA_X2(SUFFIX, ATYPE, BTYPE, CTYPE, CTYPE_X2,          \
                              GET_SUFFIX, SET_SUFFIX, BUILTIN)                 \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  CTYPE_X2 __riscv_th_mmaq_##SUFFIX##_x2(CTYPE_X2 __c, ATYPE __a,             \
                                          BTYPE __b,                           \
                                          mrow_t __m, mcol_t __k,              \
                                          mcol_t __n) {                        \
    CTYPE __c0 = __riscv_th_mget_##GET_SUFFIX(__c, 0);                         \
    CTYPE __r0 = __builtin_riscv_th_##BUILTIN(__c0, __a, __b,                  \
                                               __m, __k, __n);                \
    return __riscv_th_mset_##SET_SUFFIX(__c, 0, __r0);                         \
  }
__THEAD_SPEC_MMAQA_X2(ss_d_h, mint16_t,  mint16_t,  mint64_t,  mint64x2_t,
                       i64, i64, mmaqa_spec_ss_d_h)
__THEAD_SPEC_MMAQA_X2(uu_d_h, muint16_t, muint16_t, muint64_t, muint64x2_t,
                       u64, u64, mmaqa_spec_uu_d_h)
__THEAD_SPEC_MMAQA_X2(us_d_h, muint16_t, mint16_t,  mint64_t,  mint64x2_t,
                       i64, i64, mmaqa_spec_us_d_h)
__THEAD_SPEC_MMAQA_X2(su_d_h, mint16_t,  muint16_t, mint64_t,  mint64x2_t,
                       i64, i64, mmaqa_spec_su_d_h)
/* Partial INT8 -> INT32 */
__THEAD_SPEC_MMAQA(p_ss_w_b, mint8_t,  mint8_t,  mint32_t,  pmmaqa_spec_ss_w_b)
__THEAD_SPEC_MMAQA(p_uu_w_b, muint8_t, muint8_t, muint32_t, pmmaqa_spec_uu_w_b)
__THEAD_SPEC_MMAQA(p_us_w_b, muint8_t, mint8_t,  mint32_t,  pmmaqa_spec_us_w_b)
__THEAD_SPEC_MMAQA(p_su_w_b, mint8_t,  muint8_t, mint32_t,  pmmaqa_spec_su_w_b)
/* Bypass INT */
__THEAD_SPEC_MMAQA(bp_ss, mint8_t,  mint8_t,  mint32_t,  mmaqa_spec_bp_ss)
__THEAD_SPEC_MMAQA(bp_uu, muint8_t, muint8_t, muint32_t, mmaqa_spec_bp_uu)
/* Shorthand aliases */
#define __riscv_th_mmaq_ss __riscv_th_mmaq_ss_w_b
#define __riscv_th_mmaq_uu __riscv_th_mmaq_uu_w_b

/* --- FP matmul --- */
#define __THEAD_SPEC_FMMAQA(SUFFIX, ATYPE, BTYPE, CTYPE, BUILTIN)             \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  CTYPE __riscv_th_mfmaqa_##SUFFIX(CTYPE __c, ATYPE __a, BTYPE __b,           \
                                    mrow_t __m, mcol_t __k, mcol_t __n) {      \
    return __builtin_riscv_th_##BUILTIN(__c, __a, __b, __m, __k, __n);         \
  }

/* Native-precision (single-register — maps directly to hardware) */
__THEAD_SPEC_FMMAQA(h, mfloat16_t, mfloat16_t, mfloat16_t, mfmaqa_spec_h)
__THEAD_SPEC_FMMAQA(s, mfloat32_t, mfloat32_t, mfloat32_t, mfmaqa_spec_s)
__THEAD_SPEC_FMMAQA(d, mfloat64_t, mfloat64_t, mfloat64_t, mfmaqa_spec_d)
/* Widening (typed sources, single-register) */
__THEAD_SPEC_FMMAQA(s_h, mfloat16_t, mfloat16_t, mfloat32_t, mfmaqa_spec_s_h)
__THEAD_SPEC_FMMAQA(d_s, mfloat32_t, mfloat32_t, mfloat64_t, mfmaqa_spec_d_s)

/* Spec-aligned x2 overloads (software-level pair abstraction per intrinsic API).
   These wrap the single-register builtins, extracting/inserting component 0. */
static __inline__ __attribute__((__always_inline__, __nodebug__))
mfloat16_t __riscv_th_mfmaqa_h_x2(mfloat16_t __c, mfloat16_t __a,
                                    mfloat16x2_t __b,
                                    mrow_t __m, mcol_t __k, mcol_t __n) {
  mfloat16_t __b0 = __riscv_th_mget_f16(__b, 0);
  return __builtin_riscv_th_mfmaqa_spec_h(__c, __a, __b0, __m, __k, __n);
}
static __inline__ __attribute__((__always_inline__, __nodebug__))
mfloat64x2_t __riscv_th_mfmaqa_d_x2(mfloat64x2_t __c, mfloat64_t __a,
                                      mfloat64_t __b,
                                      mrow_t __m, mcol_t __k, mcol_t __n) {
  mfloat64_t __c0 = __riscv_th_mget_f64(__c, 0);
  mfloat64_t __r0 = __builtin_riscv_th_mfmaqa_spec_d(__c0, __a, __b,
                                                       __m, __k, __n);
  return __riscv_th_mset_f64(__c, 0, __r0);
}
static __inline__ __attribute__((__always_inline__, __nodebug__))
mfloat64x2_t __riscv_th_mfmaqa_d_s_x2(mfloat64x2_t __c, mfloat32_t __a,
                                        mfloat32_t __b,
                                        mrow_t __m, mcol_t __k, mcol_t __n) {
  mfloat64_t __c0 = __riscv_th_mget_f64(__c, 0);
  mfloat64_t __r0 = __builtin_riscv_th_mfmaqa_spec_d_s(__c0, __a, __b,
                                                         __m, __k, __n);
  return __riscv_th_mset_f64(__c, 0, __r0);
}

/* Widening FP matmul with opaque source types (FP8/BF16/TF32) */
#define __THEAD_SPEC_FMMAQA_WIDEN(SUFFIX, CTYPE, BUILTIN)                     \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  CTYPE __riscv_th_mfmaqa_##SUFFIX(CTYPE __c, mint32_t __a, mint32_t __b,     \
                                    mrow_t __m, mcol_t __k, mcol_t __n) {      \
    return __builtin_riscv_th_##BUILTIN(__c, __a, __b, __m, __k, __n);         \
  }

__THEAD_SPEC_FMMAQA_WIDEN(h_e4,    mfloat16_t, mfmaqa_spec_h_e4)
__THEAD_SPEC_FMMAQA_WIDEN(h_e5,    mfloat16_t, mfmaqa_spec_h_e5)
__THEAD_SPEC_FMMAQA_WIDEN(bf16_e4, mfloat16_t, mfmaqa_spec_bf16_e4)
__THEAD_SPEC_FMMAQA_WIDEN(bf16_e5, mfloat16_t, mfmaqa_spec_bf16_e5)
__THEAD_SPEC_FMMAQA_WIDEN(s_bf16,  mfloat32_t, mfmaqa_spec_s_bf16)
__THEAD_SPEC_FMMAQA_WIDEN(s_e4,    mfloat32_t, mfmaqa_spec_s_e4)
__THEAD_SPEC_FMMAQA_WIDEN(s_e5,    mfloat32_t, mfmaqa_spec_s_e5)
__THEAD_SPEC_FMMAQA_WIDEN(s_tf32,  mfloat32_t, mfmaqa_spec_s_tf32)

/* --- Zero --- */
/* Note: uses 'mzeros' to avoid collision with DirectReg mzero */
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

/* --- Move / Copy --- */
static __inline__ __attribute__((__always_inline__, __nodebug__))
mint32_t __riscv_th_mmov_mm(mint32_t __src) {
  return __builtin_riscv_th_mmov_mm_spec(__src);
}

/* --- Matrix-to-GPR (element extract) --- */
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

/* --- GPR-to-matrix (element insert) --- */
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

/* --- Duplicate GPR to matrix column --- */
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

/* --- Pack --- */
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

/* --- Row slide --- */
static __inline__ __attribute__((__always_inline__, __nodebug__))
mint32_t __riscv_th_mrslidedown(mint32_t __src, unsigned int __imm) {
  return __builtin_riscv_th_mrslidedown_spec(__src, __imm);
}
static __inline__ __attribute__((__always_inline__, __nodebug__))
mint32_t __riscv_th_mrslideup(mint32_t __src, unsigned int __imm) {
  return __builtin_riscv_th_mrslideup_spec(__src, __imm);
}

/* --- Column slide --- */
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

/* --- Row broadcast --- */
static __inline__ __attribute__((__always_inline__, __nodebug__))
mint32_t __riscv_th_mrbca(mint32_t __src, unsigned int __imm) {
  return __builtin_riscv_th_mrbca_mv_i_spec(__src, __imm);
}

/* --- Column broadcast --- */
#define __THEAD_SPEC_MCBCA(SUFFIX)                                             \
  static __inline__ __attribute__((__always_inline__, __nodebug__))             \
  mint32_t __riscv_th_mcbca_##SUFFIX(mint32_t __src, unsigned int __imm) {     \
    return __builtin_riscv_th_mcbca##SUFFIX##_mv_i_spec(__src, __imm);         \
  }

__THEAD_SPEC_MCBCA(b)
__THEAD_SPEC_MCBCA(h)
__THEAD_SPEC_MCBCA(w)
__THEAD_SPEC_MCBCA(d)

/* --- FP format conversions (unary: src -> dst) --- */
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

/* --- Float-int conversions --- */
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

/* --- Packed conversions --- */
__THEAD_SPEC_FCVT(mucvtl_b_p, mint32_t, mint32_t)
__THEAD_SPEC_FCVT(mscvtl_b_p, mint32_t, mint32_t)
__THEAD_SPEC_FCVT(mucvth_b_p, mint32_t, mint32_t)
__THEAD_SPEC_FCVT(mscvth_b_p, mint32_t, mint32_t)

/* --- N4clip .mm: (acc, ms2, ms1) -> acc --- */
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

/* --- Integer element-wise .w.mm: (acc, ms2, ms1) -> acc --- */
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

/* --- FP element-wise .mm: (acc, ms2, ms1) -> acc --- */
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

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* defined(__riscv_xtheadmatrix) */

#endif /* __THEAD_MATRIX_H */
