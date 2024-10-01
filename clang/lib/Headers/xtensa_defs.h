/*===---- xtensa_defs.h - Xtensa definitions -------------------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */
#ifndef __XTENSAHIFI3_H
#define __XTENSAHIFI3_H

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__((vector_size(4)));
typedef int ae_int32x2 __attribute__((vector_size(8)));
typedef short ae_int16 __attribute__((vector_size(2)));
typedef short ae_int16x2 __attribute__((vector_size(4)));
typedef short ae_int16x4 __attribute__((vector_size(8)));
typedef long long ae_int64 __attribute__((vector_size(8)));
typedef unsigned char ae_valign __attribute__((vector_size(8)));
typedef ae_int16x4 ae_f16x4;
typedef ae_int32x2 ae_f32x2;
typedef ae_int32 ae_f32;
typedef ae_int64 ae_f64;
typedef ae_int32 ae_f24;
typedef ae_int32x2 ae_f24x2;
typedef ae_int16 ae_f16;

#include <xtensa_protos.h>

#define AE_SETCBEGIN0(x) WUR_AE_CBEGIN0(x)
#define AE_SETCEND0(x) WUR_AE_CEND0(x)
#define AE_ZERO64(x) AE_MOVI(0)
#define AE_ZERO24(x) AE_MOVI(0)
#define AE_ZERO32(x) AE_MOVI(0)
#define AE_ZERO16(x) AE_MOVI(0)
#define AE_ZEROQ56(x) AE_ZERO64(x)

#define AE_SEL32_L(a)                                                          \
  ({                                                                           \
    ae_int32x2 _a = a;                                                         \
    __builtin_shufflevector(_a, _a, 0);                                        \
  })

#define AE_INT32(a) __builtin_xtensa_ae_int32(a);
#define AE_INT32X2(a) __builtin_xtensa_ae_int32x2(a);

#define AE_F32X2 AE_INT32X2
#define AE_F32 AE_INT32

#define AE_MOVINT16X4_FROMINT32X2(a) ((ae_int32x2)(a))

#define AE_F32_ADDS_F32(s1, s2)                                                \
  AE_F32(AE_ADD32S(AE_INT32X2(s1), AE_INT32X2(s2)))

typedef float xtfloat;

#define XT_xtfloat_storeip(x, a, i) ({ a = __builtin_xtensa_xt_ssip(x, a, i); })
#define XT_xtfloat_loadip(x, a, i)                                             \
  ({ x = __builtin_xtensa_xt_lsip((xtfloat **)&a, i); })
#define XT_xtfloat_loadxp(x, a, i)                                             \
  ({ x = __builtin_xtensa_xt_lsxp((xtfloat **)&a, i); })

#endif /* __XTENSAHIFI3_H */