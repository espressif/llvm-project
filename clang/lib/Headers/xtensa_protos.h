
/*===---- xtensa_protos.h - Xtensa intrinsics -------------------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */
#ifndef __XTENSAHIFI3INTRIN_H
#define __XTENSAHIFI3INTRIN_H

#define AE_ABS16S(ae_arth_v1)                                                  \
  ({                                                                           \
    ae_int16x4 __ae_arth_v1 = (ae_arth_v1);                                    \
    __builtin_xtensa_ae_abs16s((__ae_arth_v1));                                \
  })

#define AE_ABS24S(ae_arth_v1)                                                  \
  ({                                                                           \
    ae_int32x2 __ae_arth_v1 = (ae_arth_v1);                                    \
    __builtin_xtensa_ae_abs24s((__ae_arth_v1));                                \
  })

#define AE_ABS32(ae_arth_v1)                                                   \
  ({                                                                           \
    ae_int32x2 __ae_arth_v1 = (ae_arth_v1);                                    \
    __builtin_xtensa_ae_abs32((__ae_arth_v1));                                 \
  })

#define AE_ABS32S(ae_arth_v1)                                                  \
  ({                                                                           \
    ae_int32x2 __ae_arth_v1 = (ae_arth_v1);                                    \
    __builtin_xtensa_ae_abs32s((__ae_arth_v1));                                \
  })

#define AE_ABS64(ae_arth_v1)                                                   \
  ({                                                                           \
    ae_int64 __ae_arth_v1 = (ae_arth_v1);                                      \
    __builtin_xtensa_ae_abs64((__ae_arth_v1));                                 \
  })

#define AE_ABS64S(ae_arth_v1)                                                  \
  ({                                                                           \
    ae_int64 __ae_arth_v1 = (ae_arth_v1);                                      \
    __builtin_xtensa_ae_abs64s((__ae_arth_v1));                                \
  })

#define AE_ADD16(ae_arth_v0, ae_arth_v1)                                       \
  ({                                                                           \
    ae_int16x4 __ae_arth_v0 = (ae_arth_v0);                                    \
    ae_int16x4 __ae_arth_v1 = (ae_arth_v1);                                    \
    __builtin_xtensa_ae_add16((__ae_arth_v0), (__ae_arth_v1));                 \
  })

#define AE_ADD16S(ae_arth_v0, ae_arth_v1)                                      \
  ({                                                                           \
    ae_int16x4 __ae_arth_v0 = (ae_arth_v0);                                    \
    ae_int16x4 __ae_arth_v1 = (ae_arth_v1);                                    \
    __builtin_xtensa_ae_add16s((__ae_arth_v0), (__ae_arth_v1));                \
  })

#define AE_ADD24S(ae_arth_v0, ae_arth_v1)                                      \
  ({                                                                           \
    ae_int32x2 __ae_arth_v0 = (ae_arth_v0);                                    \
    ae_int32x2 __ae_arth_v1 = (ae_arth_v1);                                    \
    __builtin_xtensa_ae_add24s((__ae_arth_v0), (__ae_arth_v1));                \
  })

#define AE_ADD32(ae_arth_v0, ae_arth_v1)                                       \
  ({                                                                           \
    ae_int32x2 __ae_arth_v0 = (ae_arth_v0);                                    \
    ae_int32x2 __ae_arth_v1 = (ae_arth_v1);                                    \
    __builtin_xtensa_ae_add32((__ae_arth_v0), (__ae_arth_v1));                 \
  })

#define AE_ADD32_HL_LH(ae_arth_v0, ae_arth_v1)                                 \
  ({                                                                           \
    ae_int32x2 __ae_arth_v0 = (ae_arth_v0);                                    \
    ae_int32x2 __ae_arth_v1 = (ae_arth_v1);                                    \
    __builtin_xtensa_ae_add32_hl_lh((__ae_arth_v0), (__ae_arth_v1));           \
  })

#define AE_ADD32S(ae_arth_v0, ae_arth_v1)                                      \
  ({                                                                           \
    ae_int32x2 __ae_arth_v0 = (ae_arth_v0);                                    \
    ae_int32x2 __ae_arth_v1 = (ae_arth_v1);                                    \
    __builtin_xtensa_ae_add32s((__ae_arth_v0), (__ae_arth_v1));                \
  })

#define AE_ADD64(ae_arth_v0, ae_arth_v1)                                       \
  ({                                                                           \
    ae_int64 __ae_arth_v0 = (ae_arth_v0);                                      \
    ae_int64 __ae_arth_v1 = (ae_arth_v1);                                      \
    __builtin_xtensa_ae_add64((__ae_arth_v0), (__ae_arth_v1));                 \
  })

#define AE_ADD64S(ae_arth_v0, ae_arth_v1)                                      \
  ({                                                                           \
    ae_int64 __ae_arth_v0 = (ae_arth_v0);                                      \
    ae_int64 __ae_arth_v1 = (ae_arth_v1);                                      \
    __builtin_xtensa_ae_add64s((__ae_arth_v0), (__ae_arth_v1));                \
  })

#define AE_ADDBRBA32(art, ars)                                                 \
  ({                                                                           \
    int __art = (int)(art);                                                    \
    int __ars = (int)(ars);                                                    \
    __builtin_xtensa_ae_addbrba32((__art), (__ars));                           \
  })

#define AE_ADDSUB32(ae_arth_v0, ae_arth_v1)                                    \
  ({                                                                           \
    ae_int32x2 __ae_arth_v0 = (ae_arth_v0);                                    \
    ae_int32x2 __ae_arth_v1 = (ae_arth_v1);                                    \
    __builtin_xtensa_ae_addsub32((__ae_arth_v0), (__ae_arth_v1));              \
  })

#define AE_ADDSUB32S(ae_arth_v0, ae_arth_v1)                                   \
  ({                                                                           \
    ae_int32x2 __ae_arth_v0 = (ae_arth_v0);                                    \
    ae_int32x2 __ae_arth_v1 = (ae_arth_v1);                                    \
    __builtin_xtensa_ae_addsub32s((__ae_arth_v0), (__ae_arth_v1));             \
  })

#define AE_AND(ae_dr_to_dr_v0, ae_dr_to_dr_v1)                                 \
  ({                                                                           \
    ae_int64 __ae_dr_to_dr_v0 = (ae_dr_to_dr_v0);                              \
    ae_int64 __ae_dr_to_dr_v1 = (ae_dr_to_dr_v1);                              \
    __builtin_xtensa_ae_and((__ae_dr_to_dr_v0), (__ae_dr_to_dr_v1));           \
  })

#define AE_CVT32X2F16_10(ae_to_dr_v0)                                          \
  ({                                                                           \
    ae_int16x4 __ae_to_dr_v0 = (ae_to_dr_v0);                                  \
    __builtin_xtensa_ae_cvt32x2f16_10((__ae_to_dr_v0));                        \
  })

#define AE_CVT32X2F16_32(ae_to_dr_v0)                                          \
  ({                                                                           \
    ae_int16x4 __ae_to_dr_v0 = (ae_to_dr_v0);                                  \
    __builtin_xtensa_ae_cvt32x2f16_32((__ae_to_dr_v0));                        \
  })

#define AE_CVT48A32(ars)                                                       \
  ({                                                                           \
    int __ars = (int)(ars);                                                    \
    __builtin_xtensa_ae_cvt48a32((__ars));                                     \
  })

#define AE_CVT64A32(ars)                                                       \
  ({                                                                           \
    int __ars = (int)(ars);                                                    \
    __builtin_xtensa_ae_cvt64a32((__ars));                                     \
  })

#define AE_CVT64F32_H(ae_dr_to_dr_v0)                                          \
  ({                                                                           \
    ae_int32x2 __ae_dr_to_dr_v0 = (ae_dr_to_dr_v0);                            \
    __builtin_xtensa_ae_cvt64f32_h((__ae_dr_to_dr_v0));                        \
  })

#define AE_CVTA32F24S_H(ae_dr_to_ar_v0)                                        \
  ({                                                                           \
    ae_int32x2 __ae_dr_to_ar_v0 = (ae_dr_to_ar_v0);                            \
    __builtin_xtensa_ae_cvta32f24s_h((__ae_dr_to_ar_v0));                      \
  })

#define AE_CVTA32F24S_L(ae_dr_to_ar_v0)                                        \
  ({                                                                           \
    ae_int32x2 __ae_dr_to_ar_v0 = (ae_dr_to_ar_v0);                            \
    __builtin_xtensa_ae_cvta32f24s_l((__ae_dr_to_ar_v0));                      \
  })

#define AE_CVTQ56A32S(ars)                                                     \
  ({                                                                           \
    int __ars = (int)(ars);                                                    \
    __builtin_xtensa_ae_cvtq56a32s((__ars));                                   \
  })

#define AE_CVTQ56P32S_H(ae_dr_to_dr_v0)                                        \
  ({                                                                           \
    ae_int32x2 __ae_dr_to_dr_v0 = (ae_dr_to_dr_v0);                            \
    __builtin_xtensa_ae_cvtq56p32s_h((__ae_dr_to_dr_v0));                      \
  })

#define AE_CVTQ56P32S_L(ae_dr_to_dr_v0)                                        \
  ({                                                                           \
    ae_int32x2 __ae_dr_to_dr_v0 = (ae_dr_to_dr_v0);                            \
    __builtin_xtensa_ae_cvtq56p32s_l((__ae_dr_to_dr_v0));                      \
  })

#define AE_DB(ars, art)                                                        \
  ({                                                                           \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_db((const short **)&(ars), (__art));                   \
  })

#define AE_DB_IC(ars, art)                                                     \
  ({                                                                           \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_db_ic((const short **)&(ars), (__art));                \
  })

#define AE_DB_IP(ars, art)                                                     \
  ({                                                                           \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_db_ip((const short **)&(ars), (__art));                \
  })

#define AE_DBI(ars, ae_ohba)                                                   \
  ({ __builtin_xtensa_ae_dbi((const short **)&(ars), (ae_ohba)); })

#define AE_DBI_IC(ars, ae_ohba)                                                \
  ({ __builtin_xtensa_ae_dbi_ic((const short **)&(ars), (ae_ohba)); })

#define AE_DBI_IP(ars, ae_ohba)                                                \
  ({ __builtin_xtensa_ae_dbi_ip((const short **)&(ars), (ae_ohba)); })

#define AE_DIV64D32_H(ae_arth_v, ae_arth_v1)                                   \
  ({                                                                           \
    ae_int32x2 __ae_arth_v1 = (ae_arth_v1);                                    \
    __builtin_xtensa_ae_div64d32_h((ae_int64 *)&(ae_arth_v), (__ae_arth_v1));  \
  })

#define AE_DIV64D32_L(ae_arth_v, ae_arth_v1)                                   \
  ({                                                                           \
    ae_int32x2 __ae_arth_v1 = (ae_arth_v1);                                    \
    __builtin_xtensa_ae_div64d32_l((ae_int64 *)&(ae_arth_v), (__ae_arth_v1));  \
  })

#define AE_EQ16(ae_cmpp_v0, ae_cmpp_v1)                                        \
  ({                                                                           \
    ae_int16x4 __ae_cmpp_v0 = (ae_cmpp_v0);                                    \
    ae_int16x4 __ae_cmpp_v1 = (ae_cmpp_v1);                                    \
    __builtin_xtensa_ae_eq16((__ae_cmpp_v0), (__ae_cmpp_v1));                  \
  })

#define AE_EQ32(ae_cmpp_v0, ae_cmpp_v1)                                        \
  ({                                                                           \
    ae_int32x2 __ae_cmpp_v0 = (ae_cmpp_v0);                                    \
    ae_int32x2 __ae_cmpp_v1 = (ae_cmpp_v1);                                    \
    __builtin_xtensa_ae_eq32((__ae_cmpp_v0), (__ae_cmpp_v1));                  \
  })

#define AE_EQ64(ae_cmpp_v0, ae_cmpp_v1)                                        \
  ({                                                                           \
    ae_int64 __ae_cmpp_v0 = (ae_cmpp_v0);                                      \
    ae_int64 __ae_cmpp_v1 = (ae_cmpp_v1);                                      \
    __builtin_xtensa_ae_eq64((__ae_cmpp_v0), (__ae_cmpp_v1));                  \
  })

#define AE_L16_I(ars, ae_immls16)                                              \
  ({                                                                           \
    ae_int16 *__ars = (ars);                                                   \
    __builtin_xtensa_ae_l16_i((__ars), (ae_immls16));                          \
  })

#define AE_L16_IP(ae_ls_v, ars, ae_immls16)                                    \
  ({                                                                           \
    __builtin_xtensa_ae_l16_ip((ae_int16x4 *)&(ae_ls_v),                       \
                               (const ae_int16 **)&(ars), (ae_immls16));       \
  })

#define AE_L16_X(ars, art)                                                     \
  ({                                                                           \
    ae_int16 *__ars = (ars);                                                   \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_l16_x((__ars), (__art));                               \
  })

#define AE_L16_XC(ae_ls_v, ars, art)                                           \
  ({                                                                           \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_l16_xc((ae_int16x4 *)&(ae_ls_v),                       \
                               (const ae_int16 **)&(ars), (__art));            \
  })

#define AE_L16_XP(ae_ls_v, ars, art)                                           \
  ({                                                                           \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_l16_xp((ae_int16x4 *)&(ae_ls_v),                       \
                               (const ae_int16 **)&(ars), (__art));            \
  })

#define AE_L16M_I(ars, ae_immls16)                                             \
  ({                                                                           \
    ae_int16 *__ars = (ars);                                                   \
    __builtin_xtensa_ae_l16m_i((__ars), (ae_immls16));                         \
  })

#define AE_L16M_IU(ae_ls_v, ars, ae_immls16)                                   \
  ({                                                                           \
    __builtin_xtensa_ae_l16m_iu((ae_int32x2 *)&(ae_ls_v),                      \
                                (const ae_int16 **)&(ars), (ae_immls16));      \
  })

#define AE_L16M_X(ars, art)                                                    \
  ({                                                                           \
    ae_int16 *__ars = (ars);                                                   \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_l16m_x((__ars), (__art));                              \
  })

#define AE_L16M_XC(ae_ls_v, ars, art)                                          \
  ({                                                                           \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_l16m_xc((ae_int32x2 *)&(ae_ls_v),                      \
                                (const ae_int16 **)&(ars), (__art));           \
  })

#define AE_L16M_XU(ae_ls_v, ars, art)                                          \
  ({                                                                           \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_l16m_xu((ae_int32x2 *)&(ae_ls_v),                      \
                                (const ae_int16 **)&(ars), (__art));           \
  })

#define AE_L16X2M_I(ars, ae_immls32)                                           \
  ({                                                                           \
    ae_int16x2 *__ars = (ars);                                                 \
    __builtin_xtensa_ae_l16x2m_i((__ars), (ae_immls32));                       \
  })

#define AE_L16X2M_IU(ae_ls_v, ars, ae_immls32)                                 \
  ({                                                                           \
    __builtin_xtensa_ae_l16x2m_iu((ae_int32x2 *)&(ae_ls_v),                    \
                                  (const ae_int16x2 **)&(ars), (ae_immls32));  \
  })

#define AE_L16X2M_X(ars, art)                                                  \
  ({                                                                           \
    ae_int16x2 *__ars = (ars);                                                 \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_l16x2m_x((__ars), (__art));                            \
  })

#define AE_L16X2M_XC(ae_ls_v, ars, art)                                        \
  ({                                                                           \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_l16x2m_xc((ae_int32x2 *)&(ae_ls_v),                    \
                                  (const ae_int16x2 **)&(ars), (__art));       \
  })

#define AE_L16X2M_XU(ae_ls_v, ars, art)                                        \
  ({                                                                           \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_l16x2m_xu((ae_int32x2 *)&(ae_ls_v),                    \
                                  (const ae_int16x2 **)&(ars), (__art));       \
  })

#define AE_L16X4_I(ars, ae_immls64)                                            \
  ({                                                                           \
    ae_int16x4 *__ars = (ars);                                                 \
    __builtin_xtensa_ae_l16x4_i((__ars), (ae_immls64));                        \
  })

#define AE_L16X4_IP(ae_ls_v, ars, ae_immls64pos)                               \
  ({                                                                           \
    __builtin_xtensa_ae_l16x4_ip((ae_int16x4 *)&(ae_ls_v),                     \
                                 (const ae_int16x4 **)&(ars),                  \
                                 (ae_immls64pos));                             \
  })

#define AE_L16X4_RIC(ae_ls_v, ars)                                             \
  ({                                                                           \
    __builtin_xtensa_ae_l16x4_ric((ae_int16x4 *)&(ae_ls_v),                    \
                                  (const ae_int16x4 **)&(ars));                \
  })

#define AE_L16X4_RIP(ae_ls_v, ars)                                             \
  ({                                                                           \
    __builtin_xtensa_ae_l16x4_rip((ae_int16x4 *)&(ae_ls_v),                    \
                                  (const ae_int16x4 **)&(ars));                \
  })

#define AE_L16X4_X(ars, art)                                                   \
  ({                                                                           \
    ae_int16x4 *__ars = (ars);                                                 \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_l16x4_x((__ars), (__art));                             \
  })

#define AE_L16X4_XC(ae_ls_v, ars, art)                                         \
  ({                                                                           \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_l16x4_xc((ae_int16x4 *)&(ae_ls_v),                     \
                                 (const ae_int16x4 **)&(ars), (__art));        \
  })

#define AE_L16X4_XP(ae_ls_v, ars, art)                                         \
  ({                                                                           \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_l16x4_xp((ae_int16x4 *)&(ae_ls_v),                     \
                                 (const ae_int16x4 **)&(ars), (__art));        \
  })

#define AE_L32_I(ars, ae_immls32)                                              \
  ({                                                                           \
    ae_int32 *__ars = (ars);                                                   \
    __builtin_xtensa_ae_l32_i((__ars), (ae_immls32));                          \
  })

#define AE_L32_IP(ae_ls_v, ars, ae_immls32)                                    \
  ({                                                                           \
    __builtin_xtensa_ae_l32_ip((ae_int32x2 *)&(ae_ls_v),                       \
                               (const ae_int32 **)&(ars), (ae_immls32));       \
  })

#define AE_L32_X(ars, art)                                                     \
  ({                                                                           \
    ae_int32 *__ars = (ars);                                                   \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_l32_x((__ars), (__art));                               \
  })

#define AE_L32_XC(ae_ls_v, ars, art)                                           \
  ({                                                                           \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_l32_xc((ae_int32x2 *)&(ae_ls_v),                       \
                               (const ae_int32 **)&(ars), (__art));            \
  })

#define AE_L32_XP(ae_ls_v, ars, art)                                           \
  ({                                                                           \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_l32_xp((ae_int32x2 *)&(ae_ls_v),                       \
                               (const ae_int32 **)&(ars), (__art));            \
  })

#define AE_L32F24_I(ars, ae_immls32)                                           \
  ({                                                                           \
    ae_int32 *__ars = (ars);                                                   \
    __builtin_xtensa_ae_l32f24_i((__ars), (ae_immls32));                       \
  })

#define AE_L32F24_IP(ae_ls_v, ars, ae_immls32)                                 \
  ({                                                                           \
    __builtin_xtensa_ae_l32f24_ip((ae_int32x2 *)&(ae_ls_v),                    \
                                  (const ae_int32 **)&(ars), (ae_immls32));    \
  })

#define AE_L32F24_X(ars, art)                                                  \
  ({                                                                           \
    ae_int32 *__ars = (ars);                                                   \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_l32f24_x((__ars), (__art));                            \
  })

#define AE_L32F24_XC(ae_ls_v, ars, art)                                        \
  ({                                                                           \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_l32f24_xc((ae_int32x2 *)&(ae_ls_v),                    \
                                  (const ae_int32 **)&(ars), (__art));         \
  })

#define AE_L32F24_XP(ae_ls_v, ars, art)                                        \
  ({                                                                           \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_l32f24_xp((ae_int32x2 *)&(ae_ls_v),                    \
                                  (const ae_int32 **)&(ars), (__art));         \
  })

#define AE_L32M_I(ars, ae_immls32)                                             \
  ({                                                                           \
    ae_int32 *__ars = (ars);                                                   \
    __builtin_xtensa_ae_l32m_i((__ars), (ae_immls32));                         \
  })

#define AE_L32M_IU(ae_ls_v, ars, ae_immls32)                                   \
  ({                                                                           \
    __builtin_xtensa_ae_l32m_iu((ae_int64 *)&(ae_ls_v),                        \
                                (const ae_int32 **)&(ars), (ae_immls32));      \
  })

#define AE_L32M_X(ars, art)                                                    \
  ({                                                                           \
    ae_int32 *__ars = (ars);                                                   \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_l32m_x((__ars), (__art));                              \
  })

#define AE_L32M_XC(ae_ls_v, ars, art)                                          \
  ({                                                                           \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_l32m_xc((ae_int64 *)&(ae_ls_v),                        \
                                (const ae_int32 **)&(ars), (__art));           \
  })

#define AE_L32M_XU(ae_ls_v, ars, art)                                          \
  ({                                                                           \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_l32m_xu((ae_int64 *)&(ae_ls_v),                        \
                                (const ae_int32 **)&(ars), (__art));           \
  })

#define AE_L32X2_I(ars, ae_immls64)                                            \
  ({                                                                           \
    ae_int32x2 *__ars = (ars);                                                 \
    __builtin_xtensa_ae_l32x2_i((__ars), (ae_immls64));                        \
  })

#define AE_L32X2_IP(ae_ls_v, ars, ae_immls64pos)                               \
  ({                                                                           \
    __builtin_xtensa_ae_l32x2_ip((ae_int32x2 *)&(ae_ls_v),                     \
                                 (const ae_int32x2 **)&(ars),                  \
                                 (ae_immls64pos));                             \
  })

#define AE_L32X2_RIC(ae_ls_v, ars)                                             \
  ({                                                                           \
    __builtin_xtensa_ae_l32x2_ric((ae_int32x2 *)&(ae_ls_v),                    \
                                  (const ae_int32x2 **)&(ars));                \
  })

#define AE_L32X2_RIP(ae_ls_v, ars)                                             \
  ({                                                                           \
    __builtin_xtensa_ae_l32x2_rip((ae_int32x2 *)&(ae_ls_v),                    \
                                  (const ae_int32x2 **)&(ars));                \
  })

#define AE_L32X2_X(ars, art)                                                   \
  ({                                                                           \
    ae_int32x2 *__ars = (ars);                                                 \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_l32x2_x((__ars), (__art));                             \
  })

#define AE_L32X2_XC(ae_ls_v, ars, art)                                         \
  ({                                                                           \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_l32x2_xc((ae_int32x2 *)&(ae_ls_v),                     \
                                 (const ae_int32x2 **)&(ars), (__art));        \
  })

#define AE_L32X2_XP(ae_ls_v, ars, art)                                         \
  ({                                                                           \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_l32x2_xp((ae_int32x2 *)&(ae_ls_v),                     \
                                 (const ae_int32x2 **)&(ars), (__art));        \
  })

#define AE_L32X2F24_I(ars, ae_immls64)                                         \
  ({                                                                           \
    ae_int32x2 *__ars = (ars);                                                 \
    __builtin_xtensa_ae_l32x2f24_i((__ars), (ae_immls64));                     \
  })

#define AE_L32X2F24_IP(ae_ls_v, ars, ae_immls64pos)                            \
  ({                                                                           \
    __builtin_xtensa_ae_l32x2f24_ip((ae_int32x2 *)&(ae_ls_v),                  \
                                    (const ae_int32x2 **)&(ars),               \
                                    (ae_immls64pos));                          \
  })

#define AE_L32X2F24_RIC(ae_ls_v, ars)                                          \
  ({                                                                           \
    __builtin_xtensa_ae_l32x2f24_ric((ae_int32x2 *)&(ae_ls_v),                 \
                                     (const ae_int32x2 **)&(ars));             \
  })

#define AE_L32X2F24_RIP(ae_ls_v, ars)                                          \
  ({                                                                           \
    __builtin_xtensa_ae_l32x2f24_rip((ae_int32x2 *)&(ae_ls_v),                 \
                                     (const ae_int32x2 **)&(ars));             \
  })

#define AE_L32X2F24_X(ars, art)                                                \
  ({                                                                           \
    ae_int32x2 *__ars = (ars);                                                 \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_l32x2f24_x((__ars), (__art));                          \
  })

#define AE_L32X2F24_XC(ae_ls_v, ars, art)                                      \
  ({                                                                           \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_l32x2f24_xc((ae_int32x2 *)&(ae_ls_v),                  \
                                    (const ae_int32x2 **)&(ars), (__art));     \
  })

#define AE_L32X2F24_XP(ae_ls_v, ars, art)                                      \
  ({                                                                           \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_l32x2f24_xp((ae_int32x2 *)&(ae_ls_v),                  \
                                    (const ae_int32x2 **)&(ars), (__art));     \
  })

#define AE_L64_I(ars, ae_immls64)                                              \
  ({                                                                           \
    ae_int64 *__ars = (ars);                                                   \
    __builtin_xtensa_ae_l64_i((__ars), (ae_immls64));                          \
  })

#define AE_L64_IP(ae_ls_v, ars, ae_immls64)                                    \
  ({                                                                           \
    __builtin_xtensa_ae_l64_ip((ae_int64 *)&(ae_ls_v),                         \
                               (const ae_int64 **)&(ars), (ae_immls64));       \
  })

#define AE_L64_X(ars, art)                                                     \
  ({                                                                           \
    ae_int64 *__ars = (ars);                                                   \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_l64_x((__ars), (__art));                               \
  })

#define AE_L64_XC(ae_ls_v, ars, art)                                           \
  ({                                                                           \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_l64_xc((ae_int64 *)&(ae_ls_v),                         \
                               (const ae_int64 **)&(ars), (__art));            \
  })

#define AE_L64_XP(ae_ls_v, ars, art)                                           \
  ({                                                                           \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_l64_xp((ae_int64 *)&(ae_ls_v),                         \
                               (const ae_int64 **)&(ars), (__art));            \
  })

#define AE_LA16X4_IC(ae_ls_av, ae_ls_uu, ars)                                  \
  ({                                                                           \
    __builtin_xtensa_ae_la16x4_ic((ae_int16x4 *)&(ae_ls_av),                   \
                                  (ae_valign *)&(ae_ls_uu),                    \
                                  (const ae_int16x4 **)&(ars));                \
  })

#define AE_LA16X4_IP(ae_ls_av, ae_ls_uu, ars)                                  \
  ({                                                                           \
    __builtin_xtensa_ae_la16x4_ip((ae_int16x4 *)&(ae_ls_av),                   \
                                  (ae_valign *)&(ae_ls_uu),                    \
                                  (const ae_int16x4 **)&(ars));                \
  })

#define AE_LA16X4_RIC(ae_ls_av, ae_ls_uu, ars)                                 \
  ({                                                                           \
    __builtin_xtensa_ae_la16x4_ric((ae_int16x4 *)&(ae_ls_av),                  \
                                   (ae_valign *)&(ae_ls_uu),                   \
                                   (const ae_int16x4 **)&(ars));               \
  })

#define AE_LA16X4_RIP(ae_ls_av, ae_ls_uu, ars)                                 \
  ({                                                                           \
    __builtin_xtensa_ae_la16x4_rip((ae_int16x4 *)&(ae_ls_av),                  \
                                   (ae_valign *)&(ae_ls_uu),                   \
                                   (const ae_int16x4 **)&(ars));               \
  })

#define AE_LA16X4NEG_PC(ae_ls_uu, ars)                                         \
  ({                                                                           \
    __builtin_xtensa_ae_la16x4neg_pc((ae_valign *)&(ae_ls_uu),                 \
                                     (const ae_int16x4 **)&(ars));             \
  })

#define AE_LA16X4POS_PC(ae_ls_uu, ars)                                         \
  ({                                                                           \
    __builtin_xtensa_ae_la16x4pos_pc((ae_valign *)&(ae_ls_uu),                 \
                                     (const ae_int16x4 **)&(ars));             \
  })

#define AE_LA24_IC(ae_ls_av, ae_ls_uu, ars)                                    \
  ({                                                                           \
    __builtin_xtensa_ae_la24_ic((ae_int32x2 *)&(ae_ls_av),                     \
                                (ae_valign *)&(ae_ls_uu),                      \
                                (const void **)&(ars));                        \
  })

#define AE_LA24_IP(ae_ls_av, ae_ls_uu, ars)                                    \
  ({                                                                           \
    __builtin_xtensa_ae_la24_ip((ae_int32x2 *)&(ae_ls_av),                     \
                                (ae_valign *)&(ae_ls_uu),                      \
                                (const void **)&(ars));                        \
  })

#define AE_LA24_RIC(ae_ls_av, ae_ls_uu, ars)                                   \
  ({                                                                           \
    __builtin_xtensa_ae_la24_ric((ae_int32x2 *)&(ae_ls_av),                    \
                                 (ae_valign *)&(ae_ls_uu),                     \
                                 (const void **)&(ars));                       \
  })

#define AE_LA24_RIP(ae_ls_av, ae_ls_uu, ars)                                   \
  ({                                                                           \
    __builtin_xtensa_ae_la24_rip((ae_int32x2 *)&(ae_ls_av),                    \
                                 (ae_valign *)&(ae_ls_uu),                     \
                                 (const void **)&(ars));                       \
  })

#define AE_LA24NEG_PC(ae_ls_uu, ars)                                           \
  ({                                                                           \
    __builtin_xtensa_ae_la24neg_pc((ae_valign *)&(ae_ls_uu),                   \
                                   (const void **)&(ars));                     \
  })

#define AE_LA24POS_PC(ae_ls_uu, ars)                                           \
  ({                                                                           \
    __builtin_xtensa_ae_la24pos_pc((ae_valign *)&(ae_ls_uu),                   \
                                   (const void **)&(ars));                     \
  })

#define AE_LA24X2_IC(ae_ls_av, ae_ls_uu, ars)                                  \
  ({                                                                           \
    __builtin_xtensa_ae_la24x2_ic((ae_int32x2 *)&(ae_ls_av),                   \
                                  (ae_valign *)&(ae_ls_uu),                    \
                                  (const void **)&(ars));                      \
  })

#define AE_LA24X2_IP(ae_ls_av, ae_ls_uu, ars)                                  \
  ({                                                                           \
    __builtin_xtensa_ae_la24x2_ip((ae_int32x2 *)&(ae_ls_av),                   \
                                  (ae_valign *)&(ae_ls_uu),                    \
                                  (const void **)&(ars));                      \
  })

#define AE_LA24X2_RIC(ae_ls_av, ae_ls_uu, ars)                                 \
  ({                                                                           \
    __builtin_xtensa_ae_la24x2_ric((ae_int32x2 *)&(ae_ls_av),                  \
                                   (ae_valign *)&(ae_ls_uu),                   \
                                   (const void **)&(ars));                     \
  })

#define AE_LA24X2_RIP(ae_ls_av, ae_ls_uu, ars)                                 \
  ({                                                                           \
    __builtin_xtensa_ae_la24x2_rip((ae_int32x2 *)&(ae_ls_av),                  \
                                   (ae_valign *)&(ae_ls_uu),                   \
                                   (const void **)&(ars));                     \
  })

#define AE_LA24X2NEG_PC(ae_ls_uu, ars)                                         \
  ({                                                                           \
    __builtin_xtensa_ae_la24x2neg_pc((ae_valign *)&(ae_ls_uu),                 \
                                     (const void **)&(ars));                   \
  })

#define AE_LA24X2POS_PC(ae_ls_uu, ars)                                         \
  ({                                                                           \
    __builtin_xtensa_ae_la24x2pos_pc((ae_valign *)&(ae_ls_uu),                 \
                                     (const void **)&(ars));                   \
  })

#define AE_LA32X2_IC(ae_ls_av, ae_ls_uu, ars)                                  \
  ({                                                                           \
    __builtin_xtensa_ae_la32x2_ic((ae_int32x2 *)&(ae_ls_av),                   \
                                  (ae_valign *)&(ae_ls_uu),                    \
                                  (const ae_int32x2 **)&(ars));                \
  })

#define AE_LA32X2_IP(ae_ls_av, ae_ls_uu, ars)                                  \
  ({                                                                           \
    __builtin_xtensa_ae_la32x2_ip((ae_int32x2 *)&(ae_ls_av),                   \
                                  (ae_valign *)&(ae_ls_uu),                    \
                                  (const ae_int32x2 **)&(ars));                \
  })

#define AE_LA32X2_RIC(ae_ls_av, ae_ls_uu, ars)                                 \
  ({                                                                           \
    __builtin_xtensa_ae_la32x2_ric((ae_int32x2 *)&(ae_ls_av),                  \
                                   (ae_valign *)&(ae_ls_uu),                   \
                                   (const ae_int32x2 **)&(ars));               \
  })

#define AE_LA32X2_RIP(ae_ls_av, ae_ls_uu, ars)                                 \
  ({                                                                           \
    __builtin_xtensa_ae_la32x2_rip((ae_int32x2 *)&(ae_ls_av),                  \
                                   (ae_valign *)&(ae_ls_uu),                   \
                                   (const ae_int32x2 **)&(ars));               \
  })

#define AE_LA32X2F24_IC(ae_ls_av, ae_ls_uu, ars)                               \
  ({                                                                           \
    __builtin_xtensa_ae_la32x2f24_ic((ae_int32x2 *)&(ae_ls_av),                \
                                     (ae_valign *)&(ae_ls_uu),                 \
                                     (const ae_int32x2 **)&(ars));             \
  })

#define AE_LA32X2F24_IP(ae_ls_av, ae_ls_uu, ars)                               \
  ({                                                                           \
    __builtin_xtensa_ae_la32x2f24_ip((ae_int32x2 *)&(ae_ls_av),                \
                                     (ae_valign *)&(ae_ls_uu),                 \
                                     (const ae_int32x2 **)&(ars));             \
  })

#define AE_LA32X2F24_RIC(ae_ls_av, ae_ls_uu, ars)                              \
  ({                                                                           \
    __builtin_xtensa_ae_la32x2f24_ric((ae_int32x2 *)&(ae_ls_av),               \
                                      (ae_valign *)&(ae_ls_uu),                \
                                      (const ae_int32x2 **)&(ars));            \
  })

#define AE_LA32X2F24_RIP(ae_ls_av, ae_ls_uu, ars)                              \
  ({                                                                           \
    __builtin_xtensa_ae_la32x2f24_rip((ae_int32x2 *)&(ae_ls_av),               \
                                      (ae_valign *)&(ae_ls_uu),                \
                                      (const ae_int32x2 **)&(ars));            \
  })

#define AE_LA32X2NEG_PC(ae_ls_uu, ars)                                         \
  ({                                                                           \
    __builtin_xtensa_ae_la32x2neg_pc((ae_valign *)&(ae_ls_uu),                 \
                                     (const ae_int32x2 **)&(ars));             \
  })

#define AE_LA32X2POS_PC(ae_ls_uu, ars)                                         \
  ({                                                                           \
    __builtin_xtensa_ae_la32x2pos_pc((ae_valign *)&(ae_ls_uu),                 \
                                     (const ae_int32x2 **)&(ars));             \
  })

#define AE_LA64_PP(ars)                                                        \
  ({                                                                           \
    void *__ars = (void *)(ars);                                               \
    __builtin_xtensa_ae_la64_pp((__ars));                                      \
  })

#define AE_LALIGN64_I(ars, ae_immls64)                                         \
  ({                                                                           \
    ae_valign *__ars = (ars);                                                  \
    __builtin_xtensa_ae_lalign64_i((__ars), (ae_immls64));                     \
  })

#define AE_LB(art)                                                             \
  ({                                                                           \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_lb((__art));                                           \
  })

#define AE_LBI(ae_ohba) ({ __builtin_xtensa_ae_lbi((ae_ohba)); })

#define AE_LBK(ars, art)                                                       \
  ({                                                                           \
    int __ars = (int)(ars);                                                    \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_lbk((__ars), (__art));                                 \
  })

#define AE_LBKI(ars, ae_ohba)                                                  \
  ({                                                                           \
    int __ars = (int)(ars);                                                    \
    __builtin_xtensa_ae_lbki((__ars), (ae_ohba));                              \
  })

#define AE_LBS(art)                                                            \
  ({                                                                           \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_lbs((__art));                                          \
  })

#define AE_LBSI(ae_ohba) ({ __builtin_xtensa_ae_lbsi((ae_ohba)); })

#define AE_LE16(ae_cmpp_v0, ae_cmpp_v1)                                        \
  ({                                                                           \
    ae_int16x4 __ae_cmpp_v0 = (ae_cmpp_v0);                                    \
    ae_int16x4 __ae_cmpp_v1 = (ae_cmpp_v1);                                    \
    __builtin_xtensa_ae_le16((__ae_cmpp_v0), (__ae_cmpp_v1));                  \
  })

#define AE_LE32(ae_cmpp_v0, ae_cmpp_v1)                                        \
  ({                                                                           \
    ae_int32x2 __ae_cmpp_v0 = (ae_cmpp_v0);                                    \
    ae_int32x2 __ae_cmpp_v1 = (ae_cmpp_v1);                                    \
    __builtin_xtensa_ae_le32((__ae_cmpp_v0), (__ae_cmpp_v1));                  \
  })

#define AE_LE64(ae_cmpp_v0, ae_cmpp_v1)                                        \
  ({                                                                           \
    ae_int64 __ae_cmpp_v0 = (ae_cmpp_v0);                                      \
    ae_int64 __ae_cmpp_v1 = (ae_cmpp_v1);                                      \
    __builtin_xtensa_ae_le64((__ae_cmpp_v0), (__ae_cmpp_v1));                  \
  })

#define AE_LT16(ae_cmpp_v0, ae_cmpp_v1)                                        \
  ({                                                                           \
    ae_int16x4 __ae_cmpp_v0 = (ae_cmpp_v0);                                    \
    ae_int16x4 __ae_cmpp_v1 = (ae_cmpp_v1);                                    \
    __builtin_xtensa_ae_lt16((__ae_cmpp_v0), (__ae_cmpp_v1));                  \
  })

#define AE_LT32(ae_cmpp_v0, ae_cmpp_v1)                                        \
  ({                                                                           \
    ae_int32x2 __ae_cmpp_v0 = (ae_cmpp_v0);                                    \
    ae_int32x2 __ae_cmpp_v1 = (ae_cmpp_v1);                                    \
    __builtin_xtensa_ae_lt32((__ae_cmpp_v0), (__ae_cmpp_v1));                  \
  })

#define AE_LT64(ae_cmpp_v0, ae_cmpp_v1)                                        \
  ({                                                                           \
    ae_int64 __ae_cmpp_v0 = (ae_cmpp_v0);                                      \
    ae_int64 __ae_cmpp_v1 = (ae_cmpp_v1);                                      \
    __builtin_xtensa_ae_lt64((__ae_cmpp_v0), (__ae_cmpp_v1));                  \
  })

#define AE_MAX32(ae_cmpp_v0, ae_cmpp_v1)                                       \
  ({                                                                           \
    ae_int32x2 __ae_cmpp_v0 = (ae_cmpp_v0);                                    \
    ae_int32x2 __ae_cmpp_v1 = (ae_cmpp_v1);                                    \
    __builtin_xtensa_ae_max32((__ae_cmpp_v0), (__ae_cmpp_v1));                 \
  })

#define AE_MAX64(ae_cmpp_v0, ae_cmpp_v1)                                       \
  ({                                                                           \
    ae_int64 __ae_cmpp_v0 = (ae_cmpp_v0);                                      \
    ae_int64 __ae_cmpp_v1 = (ae_cmpp_v1);                                      \
    __builtin_xtensa_ae_max64((__ae_cmpp_v0), (__ae_cmpp_v1));                 \
  })

#define AE_MAXABS32S(ae_cmpp_v0, ae_cmpp_v1)                                   \
  ({                                                                           \
    ae_int32x2 __ae_cmpp_v0 = (ae_cmpp_v0);                                    \
    ae_int32x2 __ae_cmpp_v1 = (ae_cmpp_v1);                                    \
    __builtin_xtensa_ae_maxabs32s((__ae_cmpp_v0), (__ae_cmpp_v1));             \
  })

#define AE_MAXABS64S(ae_cmpp_v0, ae_cmpp_v1)                                   \
  ({                                                                           \
    ae_int64 __ae_cmpp_v0 = (ae_cmpp_v0);                                      \
    ae_int64 __ae_cmpp_v1 = (ae_cmpp_v1);                                      \
    __builtin_xtensa_ae_maxabs64s((__ae_cmpp_v0), (__ae_cmpp_v1));             \
  })

#define AE_MIN32(ae_cmpp_v0, ae_cmpp_v1)                                       \
  ({                                                                           \
    ae_int32x2 __ae_cmpp_v0 = (ae_cmpp_v0);                                    \
    ae_int32x2 __ae_cmpp_v1 = (ae_cmpp_v1);                                    \
    __builtin_xtensa_ae_min32((__ae_cmpp_v0), (__ae_cmpp_v1));                 \
  })

#define AE_MIN64(ae_cmpp_v0, ae_cmpp_v1)                                       \
  ({                                                                           \
    ae_int64 __ae_cmpp_v0 = (ae_cmpp_v0);                                      \
    ae_int64 __ae_cmpp_v1 = (ae_cmpp_v1);                                      \
    __builtin_xtensa_ae_min64((__ae_cmpp_v0), (__ae_cmpp_v1));                 \
  })

#define AE_MINABS32S(ae_cmpp_v0, ae_cmpp_v1)                                   \
  ({                                                                           \
    ae_int32x2 __ae_cmpp_v0 = (ae_cmpp_v0);                                    \
    ae_int32x2 __ae_cmpp_v1 = (ae_cmpp_v1);                                    \
    __builtin_xtensa_ae_minabs32s((__ae_cmpp_v0), (__ae_cmpp_v1));             \
  })

#define AE_MINABS64S(ae_cmpp_v0, ae_cmpp_v1)                                   \
  ({                                                                           \
    ae_int64 __ae_cmpp_v0 = (ae_cmpp_v0);                                      \
    ae_int64 __ae_cmpp_v1 = (ae_cmpp_v1);                                      \
    __builtin_xtensa_ae_minabs64s((__ae_cmpp_v0), (__ae_cmpp_v1));             \
  })

#define AE_MOV(ae_to_dr_v0)                                                    \
  ({                                                                           \
    ae_int64 __ae_to_dr_v0 = (ae_to_dr_v0);                                    \
    __builtin_xtensa_ae_mov((__ae_to_dr_v0));                                  \
  })

#define AE_MOVAD16_0(ae_dr_to_ar_v0)                                           \
  ({                                                                           \
    ae_int16x4 __ae_dr_to_ar_v0 = (ae_dr_to_ar_v0);                            \
    __builtin_xtensa_ae_movad16_0((__ae_dr_to_ar_v0));                         \
  })

#define AE_MOVAD16_1(ae_dr_to_ar_v0)                                           \
  ({                                                                           \
    ae_int16x4 __ae_dr_to_ar_v0 = (ae_dr_to_ar_v0);                            \
    __builtin_xtensa_ae_movad16_1((__ae_dr_to_ar_v0));                         \
  })

#define AE_MOVAD16_2(ae_dr_to_ar_v0)                                           \
  ({                                                                           \
    ae_int16x4 __ae_dr_to_ar_v0 = (ae_dr_to_ar_v0);                            \
    __builtin_xtensa_ae_movad16_2((__ae_dr_to_ar_v0));                         \
  })

#define AE_MOVAD16_3(ae_dr_to_ar_v0)                                           \
  ({                                                                           \
    ae_int16x4 __ae_dr_to_ar_v0 = (ae_dr_to_ar_v0);                            \
    __builtin_xtensa_ae_movad16_3((__ae_dr_to_ar_v0));                         \
  })

#define AE_MOVAD32_H(ae_dr_to_ar_v0)                                           \
  ({                                                                           \
    ae_int32x2 __ae_dr_to_ar_v0 = (ae_dr_to_ar_v0);                            \
    __builtin_xtensa_ae_movad32_h((__ae_dr_to_ar_v0));                         \
  })

#define AE_MOVAD32_L(ae_dr_to_ar_v0)                                           \
  ({                                                                           \
    ae_int32x2 __ae_dr_to_ar_v0 = (ae_dr_to_ar_v0);                            \
    __builtin_xtensa_ae_movad32_l((__ae_dr_to_ar_v0));                         \
  })

#define AE_MOVALIGN(ae_uu_v)                                                   \
  ({                                                                           \
    ae_valign __ae_uu_v = (ae_uu_v);                                           \
    __builtin_xtensa_ae_movalign((__ae_uu_v));                                 \
  })

#define AE_MOVDA16(ars)                                                        \
  ({                                                                           \
    int __ars = (int)(ars);                                                    \
    __builtin_xtensa_ae_movda16((__ars));                                      \
  })

#define AE_MOVDA16X2(ars, art)                                                 \
  ({                                                                           \
    int __ars = (int)(ars);                                                    \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_movda16x2((__ars), (__art));                           \
  })

#define AE_MOVDA32(ars)                                                        \
  ({                                                                           \
    int __ars = (int)(ars);                                                    \
    __builtin_xtensa_ae_movda32((__ars));                                      \
  })

#define AE_MOVDA32X2(ars, art)                                                 \
  ({                                                                           \
    int __ars = (int)(ars);                                                    \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_movda32x2((__ars), (__art));                           \
  })

#define AE_MOVF16X4(ae_cmov_v, ae_cmov_v0, bt4)                                \
  ({                                                                           \
    ae_int16x4 __ae_cmov_v0 = (ae_cmov_v0);                                    \
    xtbool4 __bt4 = (bt4);                                                     \
    __builtin_xtensa_ae_movf16x4((ae_int16x4 *)&(ae_cmov_v), (__ae_cmov_v0),   \
                                 (__bt4));                                     \
  })

#define AE_MOVF32X2(ae_cmov_v, ae_cmov_v0, bt2)                                \
  ({                                                                           \
    ae_int32x2 __ae_cmov_v0 = (ae_cmov_v0);                                    \
    xtbool2 __bt2 = (bt2);                                                     \
    __builtin_xtensa_ae_movf32x2((ae_int32x2 *)&(ae_cmov_v), (__ae_cmov_v0),   \
                                 (__bt2));                                     \
  })

#define AE_MOVF64(ae_cmov_v, ae_cmov_v0, bt)                                   \
  ({                                                                           \
    ae_int64 __ae_cmov_v0 = (ae_cmov_v0);                                      \
    xtbool __bt = (bt);                                                        \
    __builtin_xtensa_ae_movf64((ae_int64 *)&(ae_cmov_v), (__ae_cmov_v0),       \
                               (__bt));                                        \
  })

#define AE_MOVI(movi_imm) ({ __builtin_xtensa_ae_movi((movi_imm)); })

#define AE_MOVT16X4(ae_cmov_v, ae_cmov_v0, bt4)                                \
  ({                                                                           \
    ae_int16x4 __ae_cmov_v0 = (ae_cmov_v0);                                    \
    xtbool4 __bt4 = (bt4);                                                     \
    __builtin_xtensa_ae_movt16x4((ae_int16x4 *)&(ae_cmov_v), (__ae_cmov_v0),   \
                                 (__bt4));                                     \
  })

#define AE_MOVT32X2(ae_cmov_v, ae_cmov_v0, bt2)                                \
  ({                                                                           \
    ae_int32x2 __ae_cmov_v0 = (ae_cmov_v0);                                    \
    xtbool2 __bt2 = (bt2);                                                     \
    __builtin_xtensa_ae_movt32x2((ae_int32x2 *)&(ae_cmov_v), (__ae_cmov_v0),   \
                                 (__bt2));                                     \
  })

#define AE_MOVT64(ae_cmov_v, ae_cmov_v0, bt)                                   \
  ({                                                                           \
    ae_int64 __ae_cmov_v0 = (ae_cmov_v0);                                      \
    xtbool __bt = (bt);                                                        \
    __builtin_xtensa_ae_movt64((ae_int64 *)&(ae_cmov_v), (__ae_cmov_v0),       \
                               (__bt));                                        \
  })

#define AE_MUL16X4(ae_mul_d1, ae_mul_d0)                                       \
  ({                                                                           \
    ae_int16x4 __ae_mul_d1 = (ae_mul_d1);                                      \
    ae_int16x4 __ae_mul_d0 = (ae_mul_d0);                                      \
    __builtin_xtensa_ae_mul16x4((__ae_mul_d1), (__ae_mul_d0));                 \
  })

#define AE_MUL32_HH(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1)        \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mul32_hh((__opnd_ae_sem_mul_x2_S1_d0),                 \
                                 (__opnd_ae_sem_mul_x2_S1_d1));                \
  })

#define AE_MUL32_LH(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1)        \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mul32_lh((__opnd_ae_sem_mul_x2_S1_d0),                 \
                                 (__opnd_ae_sem_mul_x2_S1_d1));                \
  })

#define AE_MUL32_LL(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1)        \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mul32_ll((__opnd_ae_sem_mul_x2_S1_d0),                 \
                                 (__opnd_ae_sem_mul_x2_S1_d1));                \
  })

#define AE_MUL32_LL_S2(ae_mul_S2_d0, ae_mul_S2_d1)                             \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mul32_ll_s2((__ae_mul_S2_d0), (__ae_mul_S2_d1));       \
  })

#define AE_MUL32U_LL(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1)       \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mul32u_ll((__opnd_ae_sem_mul_x2_S1_d0),                \
                                  (__opnd_ae_sem_mul_x2_S1_d1));               \
  })

#define AE_MUL32X16_H0(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1)     \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mul32x16_h0((__opnd_ae_sem_mul_x2_S1_d0),              \
                                    (__opnd_ae_sem_mul_x2_S1_d1));             \
  })

#define AE_MUL32X16_H0_S2(ae_mul_S2_d0, ae_mul_S2_d1)                          \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mul32x16_h0_s2((__ae_mul_S2_d0), (__ae_mul_S2_d1));    \
  })

#define AE_MUL32X16_H1(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1)     \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mul32x16_h1((__opnd_ae_sem_mul_x2_S1_d0),              \
                                    (__opnd_ae_sem_mul_x2_S1_d1));             \
  })

#define AE_MUL32X16_H1_S2(ae_mul_S2_d0, ae_mul_S2_d1)                          \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mul32x16_h1_s2((__ae_mul_S2_d0), (__ae_mul_S2_d1));    \
  })

#define AE_MUL32X16_H2(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1)     \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mul32x16_h2((__opnd_ae_sem_mul_x2_S1_d0),              \
                                    (__opnd_ae_sem_mul_x2_S1_d1));             \
  })

#define AE_MUL32X16_H2_S2(ae_mul_S2_d0, ae_mul_S2_d1)                          \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mul32x16_h2_s2((__ae_mul_S2_d0), (__ae_mul_S2_d1));    \
  })

#define AE_MUL32X16_H3(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1)     \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mul32x16_h3((__opnd_ae_sem_mul_x2_S1_d0),              \
                                    (__opnd_ae_sem_mul_x2_S1_d1));             \
  })

#define AE_MUL32X16_H3_S2(ae_mul_S2_d0, ae_mul_S2_d1)                          \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mul32x16_h3_s2((__ae_mul_S2_d0), (__ae_mul_S2_d1));    \
  })

#define AE_MUL32X16_L0(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1)     \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mul32x16_l0((__opnd_ae_sem_mul_x2_S1_d0),              \
                                    (__opnd_ae_sem_mul_x2_S1_d1));             \
  })

#define AE_MUL32X16_L0_S2(ae_mul_S2_d0, ae_mul_S2_d1)                          \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mul32x16_l0_s2((__ae_mul_S2_d0), (__ae_mul_S2_d1));    \
  })

#define AE_MUL32X16_L1(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1)     \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mul32x16_l1((__opnd_ae_sem_mul_x2_S1_d0),              \
                                    (__opnd_ae_sem_mul_x2_S1_d1));             \
  })

#define AE_MUL32X16_L1_S2(ae_mul_S2_d0, ae_mul_S2_d1)                          \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mul32x16_l1_s2((__ae_mul_S2_d0), (__ae_mul_S2_d1));    \
  })

#define AE_MUL32X16_L2(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1)     \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mul32x16_l2((__opnd_ae_sem_mul_x2_S1_d0),              \
                                    (__opnd_ae_sem_mul_x2_S1_d1));             \
  })

#define AE_MUL32X16_L2_S2(ae_mul_S2_d0, ae_mul_S2_d1)                          \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mul32x16_l2_s2((__ae_mul_S2_d0), (__ae_mul_S2_d1));    \
  })

#define AE_MUL32X16_L3(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1)     \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mul32x16_l3((__opnd_ae_sem_mul_x2_S1_d0),              \
                                    (__opnd_ae_sem_mul_x2_S1_d1));             \
  })

#define AE_MUL32X16_L3_S2(ae_mul_S2_d0, ae_mul_S2_d1)                          \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mul32x16_l3_s2((__ae_mul_S2_d0), (__ae_mul_S2_d1));    \
  })

#define AE_MULA16X4(ae_mul_q1, ae_mul_q0, ae_mul_d1, ae_mul_d0)                \
  ({                                                                           \
    ae_int16x4 __ae_mul_d1 = (ae_mul_d1);                                      \
    ae_int16x4 __ae_mul_d0 = (ae_mul_d0);                                      \
    __builtin_xtensa_ae_mula16x4((ae_int32x2 *)&(ae_mul_q1),                   \
                                 (ae_int32x2 *)&(ae_mul_q0), (__ae_mul_d1),    \
                                 (__ae_mul_d0));                               \
  })

#define AE_MULA32_HH(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,       \
                     opnd_ae_sem_mul_x2_S1_d1)                                 \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mula32_hh((ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0),     \
                                  (__opnd_ae_sem_mul_x2_S1_d0),                \
                                  (__opnd_ae_sem_mul_x2_S1_d1));               \
  })

#define AE_MULA32_LH(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,       \
                     opnd_ae_sem_mul_x2_S1_d1)                                 \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mula32_lh((ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0),     \
                                  (__opnd_ae_sem_mul_x2_S1_d0),                \
                                  (__opnd_ae_sem_mul_x2_S1_d1));               \
  })

#define AE_MULA32_LL(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,       \
                     opnd_ae_sem_mul_x2_S1_d1)                                 \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mula32_ll((ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0),     \
                                  (__opnd_ae_sem_mul_x2_S1_d0),                \
                                  (__opnd_ae_sem_mul_x2_S1_d1));               \
  })

#define AE_MULA32_LL_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)              \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mula32_ll_s2((ae_int64 *)&(ae_mul_S2_q0),              \
                                     (__ae_mul_S2_d0), (__ae_mul_S2_d1));      \
  })

#define AE_MULA32U_LL(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,      \
                      opnd_ae_sem_mul_x2_S1_d1)                                \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mula32u_ll((ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0),    \
                                   (__opnd_ae_sem_mul_x2_S1_d0),               \
                                   (__opnd_ae_sem_mul_x2_S1_d1));              \
  })

#define AE_MULA32X16_H0(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,    \
                        opnd_ae_sem_mul_x2_S1_d1)                              \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mula32x16_h0((ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0),  \
                                     (__opnd_ae_sem_mul_x2_S1_d0),             \
                                     (__opnd_ae_sem_mul_x2_S1_d1));            \
  })

#define AE_MULA32X16_H0_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)           \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mula32x16_h0_s2((ae_int64 *)&(ae_mul_S2_q0),           \
                                        (__ae_mul_S2_d0), (__ae_mul_S2_d1));   \
  })

#define AE_MULA32X16_H1(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,    \
                        opnd_ae_sem_mul_x2_S1_d1)                              \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mula32x16_h1((ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0),  \
                                     (__opnd_ae_sem_mul_x2_S1_d0),             \
                                     (__opnd_ae_sem_mul_x2_S1_d1));            \
  })

#define AE_MULA32X16_H1_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)           \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mula32x16_h1_s2((ae_int64 *)&(ae_mul_S2_q0),           \
                                        (__ae_mul_S2_d0), (__ae_mul_S2_d1));   \
  })

#define AE_MULA32X16_H2(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,    \
                        opnd_ae_sem_mul_x2_S1_d1)                              \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mula32x16_h2((ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0),  \
                                     (__opnd_ae_sem_mul_x2_S1_d0),             \
                                     (__opnd_ae_sem_mul_x2_S1_d1));            \
  })

#define AE_MULA32X16_H2_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)           \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mula32x16_h2_s2((ae_int64 *)&(ae_mul_S2_q0),           \
                                        (__ae_mul_S2_d0), (__ae_mul_S2_d1));   \
  })

#define AE_MULA32X16_H3(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,    \
                        opnd_ae_sem_mul_x2_S1_d1)                              \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mula32x16_h3((ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0),  \
                                     (__opnd_ae_sem_mul_x2_S1_d0),             \
                                     (__opnd_ae_sem_mul_x2_S1_d1));            \
  })

#define AE_MULA32X16_H3_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)           \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mula32x16_h3_s2((ae_int64 *)&(ae_mul_S2_q0),           \
                                        (__ae_mul_S2_d0), (__ae_mul_S2_d1));   \
  })

#define AE_MULA32X16_L0(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,    \
                        opnd_ae_sem_mul_x2_S1_d1)                              \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mula32x16_l0((ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0),  \
                                     (__opnd_ae_sem_mul_x2_S1_d0),             \
                                     (__opnd_ae_sem_mul_x2_S1_d1));            \
  })

#define AE_MULA32X16_L0_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)           \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mula32x16_l0_s2((ae_int64 *)&(ae_mul_S2_q0),           \
                                        (__ae_mul_S2_d0), (__ae_mul_S2_d1));   \
  })

#define AE_MULA32X16_L1(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,    \
                        opnd_ae_sem_mul_x2_S1_d1)                              \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mula32x16_l1((ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0),  \
                                     (__opnd_ae_sem_mul_x2_S1_d0),             \
                                     (__opnd_ae_sem_mul_x2_S1_d1));            \
  })

#define AE_MULA32X16_L1_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)           \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mula32x16_l1_s2((ae_int64 *)&(ae_mul_S2_q0),           \
                                        (__ae_mul_S2_d0), (__ae_mul_S2_d1));   \
  })

#define AE_MULA32X16_L2(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,    \
                        opnd_ae_sem_mul_x2_S1_d1)                              \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mula32x16_l2((ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0),  \
                                     (__opnd_ae_sem_mul_x2_S1_d0),             \
                                     (__opnd_ae_sem_mul_x2_S1_d1));            \
  })

#define AE_MULA32X16_L2_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)           \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mula32x16_l2_s2((ae_int64 *)&(ae_mul_S2_q0),           \
                                        (__ae_mul_S2_d0), (__ae_mul_S2_d1));   \
  })

#define AE_MULA32X16_L3(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,    \
                        opnd_ae_sem_mul_x2_S1_d1)                              \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mula32x16_l3((ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0),  \
                                     (__opnd_ae_sem_mul_x2_S1_d0),             \
                                     (__opnd_ae_sem_mul_x2_S1_d1));            \
  })

#define AE_MULA32X16_L3_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)           \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mula32x16_l3_s2((ae_int64 *)&(ae_mul_S2_q0),           \
                                        (__ae_mul_S2_d0), (__ae_mul_S2_d1));   \
  })

#define AE_MULAAD24_HH_LL(ae_mul_q0, ae_mul_d0, ae_mul_d1)                     \
  ({                                                                           \
    ae_int32x2 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int32x2 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulaad24_hh_ll((ae_int64 *)&(ae_mul_q0),               \
                                       (__ae_mul_d0), (__ae_mul_d1));          \
  })

#define AE_MULAAD24_HH_LL_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)         \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulaad24_hh_ll_s2((ae_int64 *)&(ae_mul_S2_q0),         \
                                          (__ae_mul_S2_d0), (__ae_mul_S2_d1)); \
  })

#define AE_MULAAD24_HL_LH(ae_mul_q0, ae_mul_d0, ae_mul_d1)                     \
  ({                                                                           \
    ae_int32x2 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int32x2 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulaad24_hl_lh((ae_int64 *)&(ae_mul_q0),               \
                                       (__ae_mul_d0), (__ae_mul_d1));          \
  })

#define AE_MULAAD24_HL_LH_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)         \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulaad24_hl_lh_s2((ae_int64 *)&(ae_mul_S2_q0),         \
                                          (__ae_mul_S2_d0), (__ae_mul_S2_d1)); \
  })

#define AE_MULAAD32X16_H0_L1(opnd_ae_sem_mul_x2_S1_q0,                         \
                             opnd_ae_sem_mul_x2_S1_d0,                         \
                             opnd_ae_sem_mul_x2_S1_d1)                         \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulaad32x16_h0_l1(                                     \
        (ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0), (__opnd_ae_sem_mul_x2_S1_d0), \
        (__opnd_ae_sem_mul_x2_S1_d1));                                         \
  })

#define AE_MULAAD32X16_H0_L1_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)      \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulaad32x16_h0_l1_s2(                                  \
        (ae_int64 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));      \
  })

#define AE_MULAAD32X16_H1_L0(opnd_ae_sem_mul_x2_S1_q0,                         \
                             opnd_ae_sem_mul_x2_S1_d0,                         \
                             opnd_ae_sem_mul_x2_S1_d1)                         \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulaad32x16_h1_l0(                                     \
        (ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0), (__opnd_ae_sem_mul_x2_S1_d0), \
        (__opnd_ae_sem_mul_x2_S1_d1));                                         \
  })

#define AE_MULAAD32X16_H1_L0_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)      \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulaad32x16_h1_l0_s2(                                  \
        (ae_int64 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));      \
  })

#define AE_MULAAD32X16_H2_L3(opnd_ae_sem_mul_x2_S1_q0,                         \
                             opnd_ae_sem_mul_x2_S1_d0,                         \
                             opnd_ae_sem_mul_x2_S1_d1)                         \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulaad32x16_h2_l3(                                     \
        (ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0), (__opnd_ae_sem_mul_x2_S1_d0), \
        (__opnd_ae_sem_mul_x2_S1_d1));                                         \
  })

#define AE_MULAAD32X16_H2_L3_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)      \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulaad32x16_h2_l3_s2(                                  \
        (ae_int64 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));      \
  })

#define AE_MULAAD32X16_H3_L2(opnd_ae_sem_mul_x2_S1_q0,                         \
                             opnd_ae_sem_mul_x2_S1_d0,                         \
                             opnd_ae_sem_mul_x2_S1_d1)                         \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulaad32x16_h3_l2(                                     \
        (ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0), (__opnd_ae_sem_mul_x2_S1_d0), \
        (__opnd_ae_sem_mul_x2_S1_d1));                                         \
  })

#define AE_MULAAD32X16_H3_L2_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)      \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulaad32x16_h3_l2_s2(                                  \
        (ae_int64 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));      \
  })

#define AE_MULAAFD16SS_11_00(opnd_ae_sem_mul_x2_S1_q0,                         \
                             opnd_ae_sem_mul_x2_S1_d0,                         \
                             opnd_ae_sem_mul_x2_S1_d1)                         \
  ({                                                                           \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulaafd16ss_11_00(                                     \
        (ae_int32x2 *)&(opnd_ae_sem_mul_x2_S1_q0),                             \
        (__opnd_ae_sem_mul_x2_S1_d0), (__opnd_ae_sem_mul_x2_S1_d1));           \
  })

#define AE_MULAAFD16SS_11_00_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)      \
  ({                                                                           \
    ae_int16x4 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulaafd16ss_11_00_s2(                                  \
        (ae_int32x2 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));    \
  })

#define AE_MULAAFD16SS_13_02(opnd_ae_sem_mul_x2_S1_q0,                         \
                             opnd_ae_sem_mul_x2_S1_d0,                         \
                             opnd_ae_sem_mul_x2_S1_d1)                         \
  ({                                                                           \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulaafd16ss_13_02(                                     \
        (ae_int32x2 *)&(opnd_ae_sem_mul_x2_S1_q0),                             \
        (__opnd_ae_sem_mul_x2_S1_d0), (__opnd_ae_sem_mul_x2_S1_d1));           \
  })

#define AE_MULAAFD16SS_13_02_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)      \
  ({                                                                           \
    ae_int16x4 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulaafd16ss_13_02_s2(                                  \
        (ae_int32x2 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));    \
  })

#define AE_MULAAFD16SS_33_22(opnd_ae_sem_mul_x2_S1_q0,                         \
                             opnd_ae_sem_mul_x2_S1_d0,                         \
                             opnd_ae_sem_mul_x2_S1_d1)                         \
  ({                                                                           \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulaafd16ss_33_22(                                     \
        (ae_int32x2 *)&(opnd_ae_sem_mul_x2_S1_q0),                             \
        (__opnd_ae_sem_mul_x2_S1_d0), (__opnd_ae_sem_mul_x2_S1_d1));           \
  })

#define AE_MULAAFD16SS_33_22_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)      \
  ({                                                                           \
    ae_int16x4 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulaafd16ss_33_22_s2(                                  \
        (ae_int32x2 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));    \
  })

#define AE_MULAAFD24_HH_LL(ae_mul_q0, ae_mul_d0, ae_mul_d1)                    \
  ({                                                                           \
    ae_int32x2 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int32x2 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulaafd24_hh_ll((ae_int64 *)&(ae_mul_q0),              \
                                        (__ae_mul_d0), (__ae_mul_d1));         \
  })

#define AE_MULAAFD24_HH_LL_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)        \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulaafd24_hh_ll_s2(                                    \
        (ae_int64 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));      \
  })

#define AE_MULAAFD24_HL_LH(ae_mul_q0, ae_mul_d0, ae_mul_d1)                    \
  ({                                                                           \
    ae_int32x2 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int32x2 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulaafd24_hl_lh((ae_int64 *)&(ae_mul_q0),              \
                                        (__ae_mul_d0), (__ae_mul_d1));         \
  })

#define AE_MULAAFD24_HL_LH_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)        \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulaafd24_hl_lh_s2(                                    \
        (ae_int64 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));      \
  })

#define AE_MULAAFD32X16_H0_L1(opnd_ae_sem_mul_x2_S1_q0,                        \
                              opnd_ae_sem_mul_x2_S1_d0,                        \
                              opnd_ae_sem_mul_x2_S1_d1)                        \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulaafd32x16_h0_l1(                                    \
        (ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0), (__opnd_ae_sem_mul_x2_S1_d0), \
        (__opnd_ae_sem_mul_x2_S1_d1));                                         \
  })

#define AE_MULAAFD32X16_H0_L1_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)     \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulaafd32x16_h0_l1_s2(                                 \
        (ae_int64 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));      \
  })

#define AE_MULAAFD32X16_H1_L0(opnd_ae_sem_mul_x2_S1_q0,                        \
                              opnd_ae_sem_mul_x2_S1_d0,                        \
                              opnd_ae_sem_mul_x2_S1_d1)                        \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulaafd32x16_h1_l0(                                    \
        (ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0), (__opnd_ae_sem_mul_x2_S1_d0), \
        (__opnd_ae_sem_mul_x2_S1_d1));                                         \
  })

#define AE_MULAAFD32X16_H1_L0_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)     \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulaafd32x16_h1_l0_s2(                                 \
        (ae_int64 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));      \
  })

#define AE_MULAAFD32X16_H2_L3(opnd_ae_sem_mul_x2_S1_q0,                        \
                              opnd_ae_sem_mul_x2_S1_d0,                        \
                              opnd_ae_sem_mul_x2_S1_d1)                        \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulaafd32x16_h2_l3(                                    \
        (ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0), (__opnd_ae_sem_mul_x2_S1_d0), \
        (__opnd_ae_sem_mul_x2_S1_d1));                                         \
  })

#define AE_MULAAFD32X16_H2_L3_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)     \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulaafd32x16_h2_l3_s2(                                 \
        (ae_int64 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));      \
  })

#define AE_MULAAFD32X16_H3_L2(opnd_ae_sem_mul_x2_S1_q0,                        \
                              opnd_ae_sem_mul_x2_S1_d0,                        \
                              opnd_ae_sem_mul_x2_S1_d1)                        \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulaafd32x16_h3_l2(                                    \
        (ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0), (__opnd_ae_sem_mul_x2_S1_d0), \
        (__opnd_ae_sem_mul_x2_S1_d1));                                         \
  })

#define AE_MULAAFD32X16_H3_L2_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)     \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulaafd32x16_h3_l2_s2(                                 \
        (ae_int64 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));      \
  })

#define AE_MULAC24(ae_mul_q0, ae_mul_d0, ae_mul_d1)                            \
  ({                                                                           \
    ae_int32x2 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int32x2 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulac24((ae_int32x2 *)&(ae_mul_q0), (__ae_mul_d0),     \
                                (__ae_mul_d1));                                \
  })

#define AE_MULAC32X16_H(opnd_ae_sem_mul_x4_q0, opnd_ae_sem_mul_x4_d0,          \
                        opnd_ae_sem_mul_x4_d1)                                 \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x4_d0 = (opnd_ae_sem_mul_x4_d0);              \
    ae_int16x4 __opnd_ae_sem_mul_x4_d1 = (opnd_ae_sem_mul_x4_d1);              \
    __builtin_xtensa_ae_mulac32x16_h((ae_int32x2 *)&(opnd_ae_sem_mul_x4_q0),   \
                                     (__opnd_ae_sem_mul_x4_d0),                \
                                     (__opnd_ae_sem_mul_x4_d1));               \
  })

#define AE_MULAC32X16_L(opnd_ae_sem_mul_x4_q0, opnd_ae_sem_mul_x4_d0,          \
                        opnd_ae_sem_mul_x4_d1)                                 \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x4_d0 = (opnd_ae_sem_mul_x4_d0);              \
    ae_int16x4 __opnd_ae_sem_mul_x4_d1 = (opnd_ae_sem_mul_x4_d1);              \
    __builtin_xtensa_ae_mulac32x16_l((ae_int32x2 *)&(opnd_ae_sem_mul_x4_q0),   \
                                     (__opnd_ae_sem_mul_x4_d0),                \
                                     (__opnd_ae_sem_mul_x4_d1));               \
  })

#define AE_MULAF16SS_00(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,    \
                        opnd_ae_sem_mul_x2_S1_d1)                              \
  ({                                                                           \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulaf16ss_00(                                          \
        (ae_int32x2 *)&(opnd_ae_sem_mul_x2_S1_q0),                             \
        (__opnd_ae_sem_mul_x2_S1_d0), (__opnd_ae_sem_mul_x2_S1_d1));           \
  })

#define AE_MULAF16SS_00_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)           \
  ({                                                                           \
    ae_int16x4 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulaf16ss_00_s2((ae_int32x2 *)&(ae_mul_S2_q0),         \
                                        (__ae_mul_S2_d0), (__ae_mul_S2_d1));   \
  })

#define AE_MULAF16SS_10(ae_mul_q0, ae_mul_d0, ae_mul_d1)                       \
  ({                                                                           \
    ae_int16x4 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int16x4 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulaf16ss_10((ae_int32x2 *)&(ae_mul_q0),               \
                                     (__ae_mul_d0), (__ae_mul_d1));            \
  })

#define AE_MULAF16SS_11(ae_mul_q0, ae_mul_d0, ae_mul_d1)                       \
  ({                                                                           \
    ae_int16x4 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int16x4 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulaf16ss_11((ae_int32x2 *)&(ae_mul_q0),               \
                                     (__ae_mul_d0), (__ae_mul_d1));            \
  })

#define AE_MULAF16SS_20(ae_mul_q0, ae_mul_d0, ae_mul_d1)                       \
  ({                                                                           \
    ae_int16x4 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int16x4 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulaf16ss_20((ae_int32x2 *)&(ae_mul_q0),               \
                                     (__ae_mul_d0), (__ae_mul_d1));            \
  })

#define AE_MULAF16SS_21(ae_mul_q0, ae_mul_d0, ae_mul_d1)                       \
  ({                                                                           \
    ae_int16x4 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int16x4 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulaf16ss_21((ae_int32x2 *)&(ae_mul_q0),               \
                                     (__ae_mul_d0), (__ae_mul_d1));            \
  })

#define AE_MULAF16SS_22(ae_mul_q0, ae_mul_d0, ae_mul_d1)                       \
  ({                                                                           \
    ae_int16x4 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int16x4 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulaf16ss_22((ae_int32x2 *)&(ae_mul_q0),               \
                                     (__ae_mul_d0), (__ae_mul_d1));            \
  })

#define AE_MULAF16SS_30(ae_mul_q0, ae_mul_d0, ae_mul_d1)                       \
  ({                                                                           \
    ae_int16x4 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int16x4 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulaf16ss_30((ae_int32x2 *)&(ae_mul_q0),               \
                                     (__ae_mul_d0), (__ae_mul_d1));            \
  })

#define AE_MULAF16SS_31(ae_mul_q0, ae_mul_d0, ae_mul_d1)                       \
  ({                                                                           \
    ae_int16x4 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int16x4 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulaf16ss_31((ae_int32x2 *)&(ae_mul_q0),               \
                                     (__ae_mul_d0), (__ae_mul_d1));            \
  })

#define AE_MULAF16SS_32(ae_mul_q0, ae_mul_d0, ae_mul_d1)                       \
  ({                                                                           \
    ae_int16x4 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int16x4 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulaf16ss_32((ae_int32x2 *)&(ae_mul_q0),               \
                                     (__ae_mul_d0), (__ae_mul_d1));            \
  })

#define AE_MULAF16SS_33(ae_mul_q0, ae_mul_d0, ae_mul_d1)                       \
  ({                                                                           \
    ae_int16x4 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int16x4 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulaf16ss_33((ae_int32x2 *)&(ae_mul_q0),               \
                                     (__ae_mul_d0), (__ae_mul_d1));            \
  })

#define AE_MULAF16X4SS(ae_mul_q1, ae_mul_q0, ae_mul_d1, ae_mul_d0)             \
  ({                                                                           \
    ae_int16x4 __ae_mul_d1 = (ae_mul_d1);                                      \
    ae_int16x4 __ae_mul_d0 = (ae_mul_d0);                                      \
    __builtin_xtensa_ae_mulaf16x4ss((ae_int32x2 *)&(ae_mul_q1),                \
                                    (ae_int32x2 *)&(ae_mul_q0), (__ae_mul_d1), \
                                    (__ae_mul_d0));                            \
  })

#define AE_MULAF32R_HH(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,     \
                       opnd_ae_sem_mul_x2_S1_d1)                               \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulaf32r_hh((ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0),   \
                                    (__opnd_ae_sem_mul_x2_S1_d0),              \
                                    (__opnd_ae_sem_mul_x2_S1_d1));             \
  })

#define AE_MULAF32R_LH(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,     \
                       opnd_ae_sem_mul_x2_S1_d1)                               \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulaf32r_lh((ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0),   \
                                    (__opnd_ae_sem_mul_x2_S1_d0),              \
                                    (__opnd_ae_sem_mul_x2_S1_d1));             \
  })

#define AE_MULAF32R_LL(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,     \
                       opnd_ae_sem_mul_x2_S1_d1)                               \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulaf32r_ll((ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0),   \
                                    (__opnd_ae_sem_mul_x2_S1_d0),              \
                                    (__opnd_ae_sem_mul_x2_S1_d1));             \
  })

#define AE_MULAF32R_LL_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)            \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulaf32r_ll_s2((ae_int64 *)&(ae_mul_S2_q0),            \
                                       (__ae_mul_S2_d0), (__ae_mul_S2_d1));    \
  })

#define AE_MULAF32S_HH(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,     \
                       opnd_ae_sem_mul_x2_S1_d1)                               \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulaf32s_hh((ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0),   \
                                    (__opnd_ae_sem_mul_x2_S1_d0),              \
                                    (__opnd_ae_sem_mul_x2_S1_d1));             \
  })

#define AE_MULAF32S_LH(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,     \
                       opnd_ae_sem_mul_x2_S1_d1)                               \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulaf32s_lh((ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0),   \
                                    (__opnd_ae_sem_mul_x2_S1_d0),              \
                                    (__opnd_ae_sem_mul_x2_S1_d1));             \
  })

#define AE_MULAF32S_LL(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,     \
                       opnd_ae_sem_mul_x2_S1_d1)                               \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulaf32s_ll((ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0),   \
                                    (__opnd_ae_sem_mul_x2_S1_d0),              \
                                    (__opnd_ae_sem_mul_x2_S1_d1));             \
  })

#define AE_MULAF32S_LL_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)            \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulaf32s_ll_s2((ae_int64 *)&(ae_mul_S2_q0),            \
                                       (__ae_mul_S2_d0), (__ae_mul_S2_d1));    \
  })

#define AE_MULAF32X16_H0(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,   \
                         opnd_ae_sem_mul_x2_S1_d1)                             \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulaf32x16_h0((ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0), \
                                      (__opnd_ae_sem_mul_x2_S1_d0),            \
                                      (__opnd_ae_sem_mul_x2_S1_d1));           \
  })

#define AE_MULAF32X16_H0_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)          \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulaf32x16_h0_s2((ae_int64 *)&(ae_mul_S2_q0),          \
                                         (__ae_mul_S2_d0), (__ae_mul_S2_d1));  \
  })

#define AE_MULAF32X16_H1(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,   \
                         opnd_ae_sem_mul_x2_S1_d1)                             \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulaf32x16_h1((ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0), \
                                      (__opnd_ae_sem_mul_x2_S1_d0),            \
                                      (__opnd_ae_sem_mul_x2_S1_d1));           \
  })

#define AE_MULAF32X16_H1_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)          \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulaf32x16_h1_s2((ae_int64 *)&(ae_mul_S2_q0),          \
                                         (__ae_mul_S2_d0), (__ae_mul_S2_d1));  \
  })

#define AE_MULAF32X16_H2(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,   \
                         opnd_ae_sem_mul_x2_S1_d1)                             \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulaf32x16_h2((ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0), \
                                      (__opnd_ae_sem_mul_x2_S1_d0),            \
                                      (__opnd_ae_sem_mul_x2_S1_d1));           \
  })

#define AE_MULAF32X16_H2_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)          \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulaf32x16_h2_s2((ae_int64 *)&(ae_mul_S2_q0),          \
                                         (__ae_mul_S2_d0), (__ae_mul_S2_d1));  \
  })

#define AE_MULAF32X16_H3(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,   \
                         opnd_ae_sem_mul_x2_S1_d1)                             \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulaf32x16_h3((ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0), \
                                      (__opnd_ae_sem_mul_x2_S1_d0),            \
                                      (__opnd_ae_sem_mul_x2_S1_d1));           \
  })

#define AE_MULAF32X16_H3_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)          \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulaf32x16_h3_s2((ae_int64 *)&(ae_mul_S2_q0),          \
                                         (__ae_mul_S2_d0), (__ae_mul_S2_d1));  \
  })

#define AE_MULAF32X16_L0(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,   \
                         opnd_ae_sem_mul_x2_S1_d1)                             \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulaf32x16_l0((ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0), \
                                      (__opnd_ae_sem_mul_x2_S1_d0),            \
                                      (__opnd_ae_sem_mul_x2_S1_d1));           \
  })

#define AE_MULAF32X16_L0_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)          \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulaf32x16_l0_s2((ae_int64 *)&(ae_mul_S2_q0),          \
                                         (__ae_mul_S2_d0), (__ae_mul_S2_d1));  \
  })

#define AE_MULAF32X16_L1(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,   \
                         opnd_ae_sem_mul_x2_S1_d1)                             \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulaf32x16_l1((ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0), \
                                      (__opnd_ae_sem_mul_x2_S1_d0),            \
                                      (__opnd_ae_sem_mul_x2_S1_d1));           \
  })

#define AE_MULAF32X16_L1_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)          \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulaf32x16_l1_s2((ae_int64 *)&(ae_mul_S2_q0),          \
                                         (__ae_mul_S2_d0), (__ae_mul_S2_d1));  \
  })

#define AE_MULAF32X16_L2(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,   \
                         opnd_ae_sem_mul_x2_S1_d1)                             \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulaf32x16_l2((ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0), \
                                      (__opnd_ae_sem_mul_x2_S1_d0),            \
                                      (__opnd_ae_sem_mul_x2_S1_d1));           \
  })

#define AE_MULAF32X16_L2_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)          \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulaf32x16_l2_s2((ae_int64 *)&(ae_mul_S2_q0),          \
                                         (__ae_mul_S2_d0), (__ae_mul_S2_d1));  \
  })

#define AE_MULAF32X16_L3(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,   \
                         opnd_ae_sem_mul_x2_S1_d1)                             \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulaf32x16_l3((ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0), \
                                      (__opnd_ae_sem_mul_x2_S1_d0),            \
                                      (__opnd_ae_sem_mul_x2_S1_d1));           \
  })

#define AE_MULAF32X16_L3_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)          \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulaf32x16_l3_s2((ae_int64 *)&(ae_mul_S2_q0),          \
                                         (__ae_mul_S2_d0), (__ae_mul_S2_d1));  \
  })

#define AE_MULAF48Q32SP16S_L(opnd_ae_sem_mul_x2_S1_q0,                         \
                             opnd_ae_sem_mul_x2_S1_d0,                         \
                             opnd_ae_sem_mul_x2_S1_d1)                         \
  ({                                                                           \
    ae_int64 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);          \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulaf48q32sp16s_l(                                     \
        (ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0), (__opnd_ae_sem_mul_x2_S1_d0), \
        (__opnd_ae_sem_mul_x2_S1_d1));                                         \
  })

#define AE_MULAF48Q32SP16S_L_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)      \
  ({                                                                           \
    ae_int64 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                  \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulaf48q32sp16s_l_s2(                                  \
        (ae_int64 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));      \
  })

#define AE_MULAF48Q32SP16U_L(opnd_ae_sem_mul_x2_S1_q0,                         \
                             opnd_ae_sem_mul_x2_S1_d0,                         \
                             opnd_ae_sem_mul_x2_S1_d1)                         \
  ({                                                                           \
    ae_int64 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);          \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulaf48q32sp16u_l(                                     \
        (ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0), (__opnd_ae_sem_mul_x2_S1_d0), \
        (__opnd_ae_sem_mul_x2_S1_d1));                                         \
  })

#define AE_MULAF48Q32SP16U_L_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)      \
  ({                                                                           \
    ae_int64 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                  \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulaf48q32sp16u_l_s2(                                  \
        (ae_int64 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));      \
  })

#define AE_MULAFC24RA(opnd_ae_sem_mul_x4_q0, opnd_ae_sem_mul_x4_d0,            \
                      opnd_ae_sem_mul_x4_d1)                                   \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x4_d0 = (opnd_ae_sem_mul_x4_d0);              \
    ae_int32x2 __opnd_ae_sem_mul_x4_d1 = (opnd_ae_sem_mul_x4_d1);              \
    __builtin_xtensa_ae_mulafc24ra((ae_int32x2 *)&(opnd_ae_sem_mul_x4_q0),     \
                                   (__opnd_ae_sem_mul_x4_d0),                  \
                                   (__opnd_ae_sem_mul_x4_d1));                 \
  })

#define AE_MULAFC32X16RAS_H(opnd_ae_sem_mul_x4_q0, opnd_ae_sem_mul_x4_d0,      \
                            opnd_ae_sem_mul_x4_d1)                             \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x4_d0 = (opnd_ae_sem_mul_x4_d0);              \
    ae_int16x4 __opnd_ae_sem_mul_x4_d1 = (opnd_ae_sem_mul_x4_d1);              \
    __builtin_xtensa_ae_mulafc32x16ras_h(                                      \
        (ae_int32x2 *)&(opnd_ae_sem_mul_x4_q0), (__opnd_ae_sem_mul_x4_d0),     \
        (__opnd_ae_sem_mul_x4_d1));                                            \
  })

#define AE_MULAFC32X16RAS_L(opnd_ae_sem_mul_x4_q0, opnd_ae_sem_mul_x4_d0,      \
                            opnd_ae_sem_mul_x4_d1)                             \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x4_d0 = (opnd_ae_sem_mul_x4_d0);              \
    ae_int16x4 __opnd_ae_sem_mul_x4_d1 = (opnd_ae_sem_mul_x4_d1);              \
    __builtin_xtensa_ae_mulafc32x16ras_l(                                      \
        (ae_int32x2 *)&(opnd_ae_sem_mul_x4_q0), (__opnd_ae_sem_mul_x4_d0),     \
        (__opnd_ae_sem_mul_x4_d1));                                            \
  })

#define AE_MULAFD24X2_FIR_H(ae_mul_q0, ae_mul_q1, ae_mul_d0, ae_mul_d1,        \
                            ae_mul_d2)                                         \
  ({                                                                           \
    ae_int32x2 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int32x2 __ae_mul_d1 = (ae_mul_d1);                                      \
    ae_int32x2 __ae_mul_d2 = (ae_mul_d2);                                      \
    __builtin_xtensa_ae_mulafd24x2_fir_h(                                      \
        (ae_int64 *)&(ae_mul_q0), (ae_int64 *)&(ae_mul_q1), (__ae_mul_d0),     \
        (__ae_mul_d1), (__ae_mul_d2));                                         \
  })

#define AE_MULAFD24X2_FIR_L(ae_mul_q0, ae_mul_q1, ae_mul_d0, ae_mul_d1,        \
                            ae_mul_d2)                                         \
  ({                                                                           \
    ae_int32x2 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int32x2 __ae_mul_d1 = (ae_mul_d1);                                      \
    ae_int32x2 __ae_mul_d2 = (ae_mul_d2);                                      \
    __builtin_xtensa_ae_mulafd24x2_fir_l(                                      \
        (ae_int64 *)&(ae_mul_q0), (ae_int64 *)&(ae_mul_q1), (__ae_mul_d0),     \
        (__ae_mul_d1), (__ae_mul_d2));                                         \
  })

#define AE_MULAFD32X16X2_FIR_HH(ae_mul_q0, ae_mul_q1, ae_mul_d0, ae_mul_d1,    \
                                ae_mul_d2)                                     \
  ({                                                                           \
    ae_int32x2 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int32x2 __ae_mul_d1 = (ae_mul_d1);                                      \
    ae_int16x4 __ae_mul_d2 = (ae_mul_d2);                                      \
    __builtin_xtensa_ae_mulafd32x16x2_fir_hh(                                  \
        (ae_int64 *)&(ae_mul_q0), (ae_int64 *)&(ae_mul_q1), (__ae_mul_d0),     \
        (__ae_mul_d1), (__ae_mul_d2));                                         \
  })

#define AE_MULAFD32X16X2_FIR_HL(ae_mul_q0, ae_mul_q1, ae_mul_d0, ae_mul_d1,    \
                                ae_mul_d2)                                     \
  ({                                                                           \
    ae_int32x2 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int32x2 __ae_mul_d1 = (ae_mul_d1);                                      \
    ae_int16x4 __ae_mul_d2 = (ae_mul_d2);                                      \
    __builtin_xtensa_ae_mulafd32x16x2_fir_hl(                                  \
        (ae_int64 *)&(ae_mul_q0), (ae_int64 *)&(ae_mul_q1), (__ae_mul_d0),     \
        (__ae_mul_d1), (__ae_mul_d2));                                         \
  })

#define AE_MULAFD32X16X2_FIR_LH(ae_mul_q0, ae_mul_q1, ae_mul_d0, ae_mul_d1,    \
                                ae_mul_d2)                                     \
  ({                                                                           \
    ae_int32x2 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int32x2 __ae_mul_d1 = (ae_mul_d1);                                      \
    ae_int16x4 __ae_mul_d2 = (ae_mul_d2);                                      \
    __builtin_xtensa_ae_mulafd32x16x2_fir_lh(                                  \
        (ae_int64 *)&(ae_mul_q0), (ae_int64 *)&(ae_mul_q1), (__ae_mul_d0),     \
        (__ae_mul_d1), (__ae_mul_d2));                                         \
  })

#define AE_MULAFD32X16X2_FIR_LL(ae_mul_q0, ae_mul_q1, ae_mul_d0, ae_mul_d1,    \
                                ae_mul_d2)                                     \
  ({                                                                           \
    ae_int32x2 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int32x2 __ae_mul_d1 = (ae_mul_d1);                                      \
    ae_int16x4 __ae_mul_d2 = (ae_mul_d2);                                      \
    __builtin_xtensa_ae_mulafd32x16x2_fir_ll(                                  \
        (ae_int64 *)&(ae_mul_q0), (ae_int64 *)&(ae_mul_q1), (__ae_mul_d0),     \
        (__ae_mul_d1), (__ae_mul_d2));                                         \
  })

#define AE_MULAFP24X2R(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,     \
                       opnd_ae_sem_mul_x2_S1_d1)                               \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulafp24x2r((ae_int32x2 *)&(opnd_ae_sem_mul_x2_S1_q0), \
                                    (__opnd_ae_sem_mul_x2_S1_d0),              \
                                    (__opnd_ae_sem_mul_x2_S1_d1));             \
  })

#define AE_MULAFP24X2R_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)            \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulafp24x2r_s2((ae_int32x2 *)&(ae_mul_S2_q0),          \
                                       (__ae_mul_S2_d0), (__ae_mul_S2_d1));    \
  })

#define AE_MULAFP24X2RA(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,    \
                        opnd_ae_sem_mul_x2_S1_d1)                              \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulafp24x2ra(                                          \
        (ae_int32x2 *)&(opnd_ae_sem_mul_x2_S1_q0),                             \
        (__opnd_ae_sem_mul_x2_S1_d0), (__opnd_ae_sem_mul_x2_S1_d1));           \
  })

#define AE_MULAFP24X2RA_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)           \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulafp24x2ra_s2((ae_int32x2 *)&(ae_mul_S2_q0),         \
                                        (__ae_mul_S2_d0), (__ae_mul_S2_d1));   \
  })

#define AE_MULAFP32X16X2RAS_H(opnd_ae_sem_mul_x2_S1_q0,                        \
                              opnd_ae_sem_mul_x2_S1_d0,                        \
                              opnd_ae_sem_mul_x2_S1_d1)                        \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulafp32x16x2ras_h(                                    \
        (ae_int32x2 *)&(opnd_ae_sem_mul_x2_S1_q0),                             \
        (__opnd_ae_sem_mul_x2_S1_d0), (__opnd_ae_sem_mul_x2_S1_d1));           \
  })

#define AE_MULAFP32X16X2RAS_H_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)     \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulafp32x16x2ras_h_s2(                                 \
        (ae_int32x2 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));    \
  })

#define AE_MULAFP32X16X2RAS_L(opnd_ae_sem_mul_x2_S1_q0,                        \
                              opnd_ae_sem_mul_x2_S1_d0,                        \
                              opnd_ae_sem_mul_x2_S1_d1)                        \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulafp32x16x2ras_l(                                    \
        (ae_int32x2 *)&(opnd_ae_sem_mul_x2_S1_q0),                             \
        (__opnd_ae_sem_mul_x2_S1_d0), (__opnd_ae_sem_mul_x2_S1_d1));           \
  })

#define AE_MULAFP32X16X2RAS_L_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)     \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulafp32x16x2ras_l_s2(                                 \
        (ae_int32x2 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));    \
  })

#define AE_MULAFP32X16X2RS_H(opnd_ae_sem_mul_x2_S1_q0,                         \
                             opnd_ae_sem_mul_x2_S1_d0,                         \
                             opnd_ae_sem_mul_x2_S1_d1)                         \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulafp32x16x2rs_h(                                     \
        (ae_int32x2 *)&(opnd_ae_sem_mul_x2_S1_q0),                             \
        (__opnd_ae_sem_mul_x2_S1_d0), (__opnd_ae_sem_mul_x2_S1_d1));           \
  })

#define AE_MULAFP32X16X2RS_H_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)      \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulafp32x16x2rs_h_s2(                                  \
        (ae_int32x2 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));    \
  })

#define AE_MULAFP32X16X2RS_L(opnd_ae_sem_mul_x2_S1_q0,                         \
                             opnd_ae_sem_mul_x2_S1_d0,                         \
                             opnd_ae_sem_mul_x2_S1_d1)                         \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulafp32x16x2rs_l(                                     \
        (ae_int32x2 *)&(opnd_ae_sem_mul_x2_S1_q0),                             \
        (__opnd_ae_sem_mul_x2_S1_d0), (__opnd_ae_sem_mul_x2_S1_d1));           \
  })

#define AE_MULAFP32X16X2RS_L_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)      \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulafp32x16x2rs_l_s2(                                  \
        (ae_int32x2 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));    \
  })

#define AE_MULAFP32X2RAS(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,   \
                         opnd_ae_sem_mul_x2_S1_d1)                             \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulafp32x2ras(                                         \
        (ae_int32x2 *)&(opnd_ae_sem_mul_x2_S1_q0),                             \
        (__opnd_ae_sem_mul_x2_S1_d0), (__opnd_ae_sem_mul_x2_S1_d1));           \
  })

#define AE_MULAFP32X2RS(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,    \
                        opnd_ae_sem_mul_x2_S1_d1)                              \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulafp32x2rs(                                          \
        (ae_int32x2 *)&(opnd_ae_sem_mul_x2_S1_q0),                             \
        (__opnd_ae_sem_mul_x2_S1_d0), (__opnd_ae_sem_mul_x2_S1_d1));           \
  })

#define AE_MULAFQ32SP24S_H_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)        \
  ({                                                                           \
    ae_int64 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                  \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulafq32sp24s_h_s2(                                    \
        (ae_int64 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));      \
  })

#define AE_MULAFQ32SP24S_L_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)        \
  ({                                                                           \
    ae_int64 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                  \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulafq32sp24s_l_s2(                                    \
        (ae_int64 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));      \
  })

#define AE_MULAP24X2(ae_mul_q0, ae_mul_d0, ae_mul_d1)                          \
  ({                                                                           \
    ae_int32x2 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int32x2 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulap24x2((ae_int32x2 *)&(ae_mul_q0), (__ae_mul_d0),   \
                                  (__ae_mul_d1));                              \
  })

#define AE_MULAP24X2_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)              \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulap24x2_s2((ae_int32x2 *)&(ae_mul_S2_q0),            \
                                     (__ae_mul_S2_d0), (__ae_mul_S2_d1));      \
  })

#define AE_MULAP32X16X2_H(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,  \
                          opnd_ae_sem_mul_x2_S1_d1)                            \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulap32x16x2_h(                                        \
        (ae_int32x2 *)&(opnd_ae_sem_mul_x2_S1_q0),                             \
        (__opnd_ae_sem_mul_x2_S1_d0), (__opnd_ae_sem_mul_x2_S1_d1));           \
  })

#define AE_MULAP32X16X2_L(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,  \
                          opnd_ae_sem_mul_x2_S1_d1)                            \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulap32x16x2_l(                                        \
        (ae_int32x2 *)&(opnd_ae_sem_mul_x2_S1_q0),                             \
        (__opnd_ae_sem_mul_x2_S1_d0), (__opnd_ae_sem_mul_x2_S1_d1));           \
  })

#define AE_MULAP32X2(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,       \
                     opnd_ae_sem_mul_x2_S1_d1)                                 \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulap32x2((ae_int32x2 *)&(opnd_ae_sem_mul_x2_S1_q0),   \
                                  (__opnd_ae_sem_mul_x2_S1_d0),                \
                                  (__opnd_ae_sem_mul_x2_S1_d1));               \
  })

#define AE_MULAQ32SP16S_L_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)         \
  ({                                                                           \
    ae_int64 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                  \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulaq32sp16s_l_s2((ae_int64 *)&(ae_mul_S2_q0),         \
                                          (__ae_mul_S2_d0), (__ae_mul_S2_d1)); \
  })

#define AE_MULAQ32SP16U_L_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)         \
  ({                                                                           \
    ae_int64 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                  \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulaq32sp16u_l_s2((ae_int64 *)&(ae_mul_S2_q0),         \
                                          (__ae_mul_S2_d0), (__ae_mul_S2_d1)); \
  })

#define AE_MULARFQ32SP24S_H_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)       \
  ({                                                                           \
    ae_int64 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                  \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mularfq32sp24s_h_s2(                                   \
        (ae_int64 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));      \
  })

#define AE_MULARFQ32SP24S_L_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)       \
  ({                                                                           \
    ae_int64 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                  \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mularfq32sp24s_l_s2(                                   \
        (ae_int64 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));      \
  })

#define AE_MULAS32F48P16S_HH(opnd_ae_sem_mul_x2_S1_q0,                         \
                             opnd_ae_sem_mul_x2_S1_d0,                         \
                             opnd_ae_sem_mul_x2_S1_d1)                         \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulas32f48p16s_hh(                                     \
        (ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0), (__opnd_ae_sem_mul_x2_S1_d0), \
        (__opnd_ae_sem_mul_x2_S1_d1));                                         \
  })

#define AE_MULAS32F48P16S_HH_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)      \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulas32f48p16s_hh_s2(                                  \
        (ae_int64 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));      \
  })

#define AE_MULAS32F48P16S_LH(opnd_ae_sem_mul_x2_S1_q0,                         \
                             opnd_ae_sem_mul_x2_S1_d0,                         \
                             opnd_ae_sem_mul_x2_S1_d1)                         \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulas32f48p16s_lh(                                     \
        (ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0), (__opnd_ae_sem_mul_x2_S1_d0), \
        (__opnd_ae_sem_mul_x2_S1_d1));                                         \
  })

#define AE_MULAS32F48P16S_LH_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)      \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulas32f48p16s_lh_s2(                                  \
        (ae_int64 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));      \
  })

#define AE_MULAS32F48P16S_LL(opnd_ae_sem_mul_x2_S1_q0,                         \
                             opnd_ae_sem_mul_x2_S1_d0,                         \
                             opnd_ae_sem_mul_x2_S1_d1)                         \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulas32f48p16s_ll(                                     \
        (ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0), (__opnd_ae_sem_mul_x2_S1_d0), \
        (__opnd_ae_sem_mul_x2_S1_d1));                                         \
  })

#define AE_MULAS32F48P16S_LL_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)      \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulas32f48p16s_ll_s2(                                  \
        (ae_int64 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));      \
  })

#define AE_MULASD24_HH_LL(ae_mul_q0, ae_mul_d0, ae_mul_d1)                     \
  ({                                                                           \
    ae_int32x2 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int32x2 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulasd24_hh_ll((ae_int64 *)&(ae_mul_q0),               \
                                       (__ae_mul_d0), (__ae_mul_d1));          \
  })

#define AE_MULASD24_HH_LL_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)         \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulasd24_hh_ll_s2((ae_int64 *)&(ae_mul_S2_q0),         \
                                          (__ae_mul_S2_d0), (__ae_mul_S2_d1)); \
  })

#define AE_MULASD24_HL_LH(ae_mul_q0, ae_mul_d0, ae_mul_d1)                     \
  ({                                                                           \
    ae_int32x2 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int32x2 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulasd24_hl_lh((ae_int64 *)&(ae_mul_q0),               \
                                       (__ae_mul_d0), (__ae_mul_d1));          \
  })

#define AE_MULASD24_HL_LH_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)         \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulasd24_hl_lh_s2((ae_int64 *)&(ae_mul_S2_q0),         \
                                          (__ae_mul_S2_d0), (__ae_mul_S2_d1)); \
  })

#define AE_MULASD32X16_H1_L0(opnd_ae_sem_mul_x2_S1_q0,                         \
                             opnd_ae_sem_mul_x2_S1_d0,                         \
                             opnd_ae_sem_mul_x2_S1_d1)                         \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulasd32x16_h1_l0(                                     \
        (ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0), (__opnd_ae_sem_mul_x2_S1_d0), \
        (__opnd_ae_sem_mul_x2_S1_d1));                                         \
  })

#define AE_MULASD32X16_H1_L0_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)      \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulasd32x16_h1_l0_s2(                                  \
        (ae_int64 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));      \
  })

#define AE_MULASD32X16_H3_L2(opnd_ae_sem_mul_x2_S1_q0,                         \
                             opnd_ae_sem_mul_x2_S1_d0,                         \
                             opnd_ae_sem_mul_x2_S1_d1)                         \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulasd32x16_h3_l2(                                     \
        (ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0), (__opnd_ae_sem_mul_x2_S1_d0), \
        (__opnd_ae_sem_mul_x2_S1_d1));                                         \
  })

#define AE_MULASD32X16_H3_L2_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)      \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulasd32x16_h3_l2_s2(                                  \
        (ae_int64 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));      \
  })

#define AE_MULASFD24_HH_LL(ae_mul_q0, ae_mul_d0, ae_mul_d1)                    \
  ({                                                                           \
    ae_int32x2 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int32x2 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulasfd24_hh_ll((ae_int64 *)&(ae_mul_q0),              \
                                        (__ae_mul_d0), (__ae_mul_d1));         \
  })

#define AE_MULASFD24_HH_LL_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)        \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulasfd24_hh_ll_s2(                                    \
        (ae_int64 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));      \
  })

#define AE_MULASFD24_HL_LH(ae_mul_q0, ae_mul_d0, ae_mul_d1)                    \
  ({                                                                           \
    ae_int32x2 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int32x2 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulasfd24_hl_lh((ae_int64 *)&(ae_mul_q0),              \
                                        (__ae_mul_d0), (__ae_mul_d1));         \
  })

#define AE_MULASFD24_HL_LH_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)        \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulasfd24_hl_lh_s2(                                    \
        (ae_int64 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));      \
  })

#define AE_MULASFD32X16_H1_L0(opnd_ae_sem_mul_x2_S1_q0,                        \
                              opnd_ae_sem_mul_x2_S1_d0,                        \
                              opnd_ae_sem_mul_x2_S1_d1)                        \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulasfd32x16_h1_l0(                                    \
        (ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0), (__opnd_ae_sem_mul_x2_S1_d0), \
        (__opnd_ae_sem_mul_x2_S1_d1));                                         \
  })

#define AE_MULASFD32X16_H1_L0_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)     \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulasfd32x16_h1_l0_s2(                                 \
        (ae_int64 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));      \
  })

#define AE_MULASFD32X16_H3_L2(opnd_ae_sem_mul_x2_S1_q0,                        \
                              opnd_ae_sem_mul_x2_S1_d0,                        \
                              opnd_ae_sem_mul_x2_S1_d1)                        \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulasfd32x16_h3_l2(                                    \
        (ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0), (__opnd_ae_sem_mul_x2_S1_d0), \
        (__opnd_ae_sem_mul_x2_S1_d1));                                         \
  })

#define AE_MULASFD32X16_H3_L2_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)     \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulasfd32x16_h3_l2_s2(                                 \
        (ae_int64 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));      \
  })

#define AE_MULC24(ae_mul_d0, ae_mul_d1)                                        \
  ({                                                                           \
    ae_int32x2 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int32x2 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulc24((__ae_mul_d0), (__ae_mul_d1));                  \
  })

#define AE_MULC32X16_H(opnd_ae_sem_mul_x4_d0, opnd_ae_sem_mul_x4_d1)           \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x4_d0 = (opnd_ae_sem_mul_x4_d0);              \
    ae_int16x4 __opnd_ae_sem_mul_x4_d1 = (opnd_ae_sem_mul_x4_d1);              \
    __builtin_xtensa_ae_mulc32x16_h((__opnd_ae_sem_mul_x4_d0),                 \
                                    (__opnd_ae_sem_mul_x4_d1));                \
  })

#define AE_MULC32X16_L(opnd_ae_sem_mul_x4_d0, opnd_ae_sem_mul_x4_d1)           \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x4_d0 = (opnd_ae_sem_mul_x4_d0);              \
    ae_int16x4 __opnd_ae_sem_mul_x4_d1 = (opnd_ae_sem_mul_x4_d1);              \
    __builtin_xtensa_ae_mulc32x16_l((__opnd_ae_sem_mul_x4_d0),                 \
                                    (__opnd_ae_sem_mul_x4_d1));                \
  })

#define AE_MULF16SS_00(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1)     \
  ({                                                                           \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulf16ss_00((__opnd_ae_sem_mul_x2_S1_d0),              \
                                    (__opnd_ae_sem_mul_x2_S1_d1));             \
  })

#define AE_MULF16SS_00_S2(ae_mul_S2_d0, ae_mul_S2_d1)                          \
  ({                                                                           \
    ae_int16x4 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulf16ss_00_s2((__ae_mul_S2_d0), (__ae_mul_S2_d1));    \
  })

#define AE_MULF16SS_10(ae_mul_d0, ae_mul_d1)                                   \
  ({                                                                           \
    ae_int16x4 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int16x4 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulf16ss_10((__ae_mul_d0), (__ae_mul_d1));             \
  })

#define AE_MULF16SS_11(ae_mul_d0, ae_mul_d1)                                   \
  ({                                                                           \
    ae_int16x4 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int16x4 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulf16ss_11((__ae_mul_d0), (__ae_mul_d1));             \
  })

#define AE_MULF16SS_20(ae_mul_d0, ae_mul_d1)                                   \
  ({                                                                           \
    ae_int16x4 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int16x4 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulf16ss_20((__ae_mul_d0), (__ae_mul_d1));             \
  })

#define AE_MULF16SS_21(ae_mul_d0, ae_mul_d1)                                   \
  ({                                                                           \
    ae_int16x4 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int16x4 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulf16ss_21((__ae_mul_d0), (__ae_mul_d1));             \
  })

#define AE_MULF16SS_22(ae_mul_d0, ae_mul_d1)                                   \
  ({                                                                           \
    ae_int16x4 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int16x4 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulf16ss_22((__ae_mul_d0), (__ae_mul_d1));             \
  })

#define AE_MULF16SS_30(ae_mul_d0, ae_mul_d1)                                   \
  ({                                                                           \
    ae_int16x4 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int16x4 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulf16ss_30((__ae_mul_d0), (__ae_mul_d1));             \
  })

#define AE_MULF16SS_31(ae_mul_d0, ae_mul_d1)                                   \
  ({                                                                           \
    ae_int16x4 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int16x4 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulf16ss_31((__ae_mul_d0), (__ae_mul_d1));             \
  })

#define AE_MULF16SS_32(ae_mul_d0, ae_mul_d1)                                   \
  ({                                                                           \
    ae_int16x4 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int16x4 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulf16ss_32((__ae_mul_d0), (__ae_mul_d1));             \
  })

#define AE_MULF16SS_33(ae_mul_d0, ae_mul_d1)                                   \
  ({                                                                           \
    ae_int16x4 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int16x4 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulf16ss_33((__ae_mul_d0), (__ae_mul_d1));             \
  })

#define AE_MULF16X4SS(ae_mul_d1, ae_mul_d0)                                    \
  ({                                                                           \
    ae_int16x4 __ae_mul_d1 = (ae_mul_d1);                                      \
    ae_int16x4 __ae_mul_d0 = (ae_mul_d0);                                      \
    __builtin_xtensa_ae_mulf16x4ss((__ae_mul_d1), (__ae_mul_d0));              \
  })

#define AE_MULF32R_HH(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1)      \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulf32r_hh((__opnd_ae_sem_mul_x2_S1_d0),               \
                                   (__opnd_ae_sem_mul_x2_S1_d1));              \
  })

#define AE_MULF32R_LH(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1)      \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulf32r_lh((__opnd_ae_sem_mul_x2_S1_d0),               \
                                   (__opnd_ae_sem_mul_x2_S1_d1));              \
  })

#define AE_MULF32R_LL(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1)      \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulf32r_ll((__opnd_ae_sem_mul_x2_S1_d0),               \
                                   (__opnd_ae_sem_mul_x2_S1_d1));              \
  })

#define AE_MULF32R_LL_S2(ae_mul_S2_d0, ae_mul_S2_d1)                           \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulf32r_ll_s2((__ae_mul_S2_d0), (__ae_mul_S2_d1));     \
  })

#define AE_MULF32S_HH(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1)      \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulf32s_hh((__opnd_ae_sem_mul_x2_S1_d0),               \
                                   (__opnd_ae_sem_mul_x2_S1_d1));              \
  })

#define AE_MULF32S_LH(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1)      \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulf32s_lh((__opnd_ae_sem_mul_x2_S1_d0),               \
                                   (__opnd_ae_sem_mul_x2_S1_d1));              \
  })

#define AE_MULF32S_LL(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1)      \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulf32s_ll((__opnd_ae_sem_mul_x2_S1_d0),               \
                                   (__opnd_ae_sem_mul_x2_S1_d1));              \
  })

#define AE_MULF32S_LL_S2(ae_mul_S2_d0, ae_mul_S2_d1)                           \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulf32s_ll_s2((__ae_mul_S2_d0), (__ae_mul_S2_d1));     \
  })

#define AE_MULF32X16_H0(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1)    \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulf32x16_h0((__opnd_ae_sem_mul_x2_S1_d0),             \
                                     (__opnd_ae_sem_mul_x2_S1_d1));            \
  })

#define AE_MULF32X16_H0_S2(ae_mul_S2_d0, ae_mul_S2_d1)                         \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulf32x16_h0_s2((__ae_mul_S2_d0), (__ae_mul_S2_d1));   \
  })

#define AE_MULF32X16_H1(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1)    \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulf32x16_h1((__opnd_ae_sem_mul_x2_S1_d0),             \
                                     (__opnd_ae_sem_mul_x2_S1_d1));            \
  })

#define AE_MULF32X16_H1_S2(ae_mul_S2_d0, ae_mul_S2_d1)                         \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulf32x16_h1_s2((__ae_mul_S2_d0), (__ae_mul_S2_d1));   \
  })

#define AE_MULF32X16_H2(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1)    \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulf32x16_h2((__opnd_ae_sem_mul_x2_S1_d0),             \
                                     (__opnd_ae_sem_mul_x2_S1_d1));            \
  })

#define AE_MULF32X16_H2_S2(ae_mul_S2_d0, ae_mul_S2_d1)                         \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulf32x16_h2_s2((__ae_mul_S2_d0), (__ae_mul_S2_d1));   \
  })

#define AE_MULF32X16_H3(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1)    \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulf32x16_h3((__opnd_ae_sem_mul_x2_S1_d0),             \
                                     (__opnd_ae_sem_mul_x2_S1_d1));            \
  })

#define AE_MULF32X16_H3_S2(ae_mul_S2_d0, ae_mul_S2_d1)                         \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulf32x16_h3_s2((__ae_mul_S2_d0), (__ae_mul_S2_d1));   \
  })

#define AE_MULF32X16_L0(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1)    \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulf32x16_l0((__opnd_ae_sem_mul_x2_S1_d0),             \
                                     (__opnd_ae_sem_mul_x2_S1_d1));            \
  })

#define AE_MULF32X16_L0_S2(ae_mul_S2_d0, ae_mul_S2_d1)                         \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulf32x16_l0_s2((__ae_mul_S2_d0), (__ae_mul_S2_d1));   \
  })

#define AE_MULF32X16_L1(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1)    \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulf32x16_l1((__opnd_ae_sem_mul_x2_S1_d0),             \
                                     (__opnd_ae_sem_mul_x2_S1_d1));            \
  })

#define AE_MULF32X16_L1_S2(ae_mul_S2_d0, ae_mul_S2_d1)                         \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulf32x16_l1_s2((__ae_mul_S2_d0), (__ae_mul_S2_d1));   \
  })

#define AE_MULF32X16_L2(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1)    \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulf32x16_l2((__opnd_ae_sem_mul_x2_S1_d0),             \
                                     (__opnd_ae_sem_mul_x2_S1_d1));            \
  })

#define AE_MULF32X16_L2_S2(ae_mul_S2_d0, ae_mul_S2_d1)                         \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulf32x16_l2_s2((__ae_mul_S2_d0), (__ae_mul_S2_d1));   \
  })

#define AE_MULF32X16_L3(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1)    \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulf32x16_l3((__opnd_ae_sem_mul_x2_S1_d0),             \
                                     (__opnd_ae_sem_mul_x2_S1_d1));            \
  })

#define AE_MULF32X16_L3_S2(ae_mul_S2_d0, ae_mul_S2_d1)                         \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulf32x16_l3_s2((__ae_mul_S2_d0), (__ae_mul_S2_d1));   \
  })

#define AE_MULF48Q32SP16S_L(opnd_ae_sem_mul_x2_S1_d0,                          \
                            opnd_ae_sem_mul_x2_S1_d1)                          \
  ({                                                                           \
    ae_int64 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);          \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulf48q32sp16s_l((__opnd_ae_sem_mul_x2_S1_d0),         \
                                         (__opnd_ae_sem_mul_x2_S1_d1));        \
  })

#define AE_MULF48Q32SP16S_L_S2(ae_mul_S2_d0, ae_mul_S2_d1)                     \
  ({                                                                           \
    ae_int64 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                  \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulf48q32sp16s_l_s2((__ae_mul_S2_d0),                  \
                                            (__ae_mul_S2_d1));                 \
  })

#define AE_MULF48Q32SP16U_L(opnd_ae_sem_mul_x2_S1_d0,                          \
                            opnd_ae_sem_mul_x2_S1_d1)                          \
  ({                                                                           \
    ae_int64 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);          \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulf48q32sp16u_l((__opnd_ae_sem_mul_x2_S1_d0),         \
                                         (__opnd_ae_sem_mul_x2_S1_d1));        \
  })

#define AE_MULF48Q32SP16U_L_S2(ae_mul_S2_d0, ae_mul_S2_d1)                     \
  ({                                                                           \
    ae_int64 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                  \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulf48q32sp16u_l_s2((__ae_mul_S2_d0),                  \
                                            (__ae_mul_S2_d1));                 \
  })

#define AE_MULFC24RA(opnd_ae_sem_mul_x4_d0, opnd_ae_sem_mul_x4_d1)             \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x4_d0 = (opnd_ae_sem_mul_x4_d0);              \
    ae_int32x2 __opnd_ae_sem_mul_x4_d1 = (opnd_ae_sem_mul_x4_d1);              \
    __builtin_xtensa_ae_mulfc24ra((__opnd_ae_sem_mul_x4_d0),                   \
                                  (__opnd_ae_sem_mul_x4_d1));                  \
  })

#define AE_MULFC32X16RAS_H(opnd_ae_sem_mul_x4_d0, opnd_ae_sem_mul_x4_d1)       \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x4_d0 = (opnd_ae_sem_mul_x4_d0);              \
    ae_int16x4 __opnd_ae_sem_mul_x4_d1 = (opnd_ae_sem_mul_x4_d1);              \
    __builtin_xtensa_ae_mulfc32x16ras_h((__opnd_ae_sem_mul_x4_d0),             \
                                        (__opnd_ae_sem_mul_x4_d1));            \
  })

#define AE_MULFC32X16RAS_L(opnd_ae_sem_mul_x4_d0, opnd_ae_sem_mul_x4_d1)       \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x4_d0 = (opnd_ae_sem_mul_x4_d0);              \
    ae_int16x4 __opnd_ae_sem_mul_x4_d1 = (opnd_ae_sem_mul_x4_d1);              \
    __builtin_xtensa_ae_mulfc32x16ras_l((__opnd_ae_sem_mul_x4_d0),             \
                                        (__opnd_ae_sem_mul_x4_d1));            \
  })

#define AE_MULFD24X2_FIR_H(ae_mul_d0, ae_mul_d1, ae_mul_d2)                    \
  ({                                                                           \
    ae_int32x2 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int32x2 __ae_mul_d1 = (ae_mul_d1);                                      \
    ae_int32x2 __ae_mul_d2 = (ae_mul_d2);                                      \
    __builtin_xtensa_ae_mulfd24x2_fir_h((__ae_mul_d0), (__ae_mul_d1),          \
                                        (__ae_mul_d2));                        \
  })

#define AE_MULFD24X2_FIR_L(ae_mul_d0, ae_mul_d1, ae_mul_d2)                    \
  ({                                                                           \
    ae_int32x2 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int32x2 __ae_mul_d1 = (ae_mul_d1);                                      \
    ae_int32x2 __ae_mul_d2 = (ae_mul_d2);                                      \
    __builtin_xtensa_ae_mulfd24x2_fir_l((__ae_mul_d0), (__ae_mul_d1),          \
                                        (__ae_mul_d2));                        \
  })

#define AE_MULFD32X16X2_FIR_HH(ae_mul_d0, ae_mul_d1, ae_mul_d2)                \
  ({                                                                           \
    ae_int32x2 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int32x2 __ae_mul_d1 = (ae_mul_d1);                                      \
    ae_int16x4 __ae_mul_d2 = (ae_mul_d2);                                      \
    __builtin_xtensa_ae_mulfd32x16x2_fir_hh((__ae_mul_d0), (__ae_mul_d1),      \
                                            (__ae_mul_d2));                    \
  })

#define AE_MULFD32X16X2_FIR_HL(ae_mul_d0, ae_mul_d1, ae_mul_d2)                \
  ({                                                                           \
    ae_int32x2 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int32x2 __ae_mul_d1 = (ae_mul_d1);                                      \
    ae_int16x4 __ae_mul_d2 = (ae_mul_d2);                                      \
    __builtin_xtensa_ae_mulfd32x16x2_fir_hl((__ae_mul_d0), (__ae_mul_d1),      \
                                            (__ae_mul_d2));                    \
  })

#define AE_MULFD32X16X2_FIR_LH(ae_mul_d0, ae_mul_d1, ae_mul_d2)                \
  ({                                                                           \
    ae_int32x2 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int32x2 __ae_mul_d1 = (ae_mul_d1);                                      \
    ae_int16x4 __ae_mul_d2 = (ae_mul_d2);                                      \
    __builtin_xtensa_ae_mulfd32x16x2_fir_lh((__ae_mul_d0), (__ae_mul_d1),      \
                                            (__ae_mul_d2));                    \
  })

#define AE_MULFD32X16X2_FIR_LL(ae_mul_d0, ae_mul_d1, ae_mul_d2)                \
  ({                                                                           \
    ae_int32x2 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int32x2 __ae_mul_d1 = (ae_mul_d1);                                      \
    ae_int16x4 __ae_mul_d2 = (ae_mul_d2);                                      \
    __builtin_xtensa_ae_mulfd32x16x2_fir_ll((__ae_mul_d0), (__ae_mul_d1),      \
                                            (__ae_mul_d2));                    \
  })

#define AE_MULFP16X4RAS(ae_mul_d0, ae_mul_d1)                                  \
  ({                                                                           \
    ae_int16x4 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int16x4 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulfp16x4ras((__ae_mul_d0), (__ae_mul_d1));            \
  })

#define AE_MULFP16X4S(ae_mul_d0, ae_mul_d1)                                    \
  ({                                                                           \
    ae_int16x4 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int16x4 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulfp16x4s((__ae_mul_d0), (__ae_mul_d1));              \
  })

#define AE_MULFP24X2R(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1)      \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulfp24x2r((__opnd_ae_sem_mul_x2_S1_d0),               \
                                   (__opnd_ae_sem_mul_x2_S1_d1));              \
  })

#define AE_MULFP24X2R_S2(ae_mul_S2_d0, ae_mul_S2_d1)                           \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulfp24x2r_s2((__ae_mul_S2_d0), (__ae_mul_S2_d1));     \
  })

#define AE_MULFP24X2RA(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1)     \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulfp24x2ra((__opnd_ae_sem_mul_x2_S1_d0),              \
                                    (__opnd_ae_sem_mul_x2_S1_d1));             \
  })

#define AE_MULFP24X2RA_S2(ae_mul_S2_d0, ae_mul_S2_d1)                          \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulfp24x2ra_s2((__ae_mul_S2_d0), (__ae_mul_S2_d1));    \
  })

#define AE_MULFP32X16X2RAS_H(opnd_ae_sem_mul_x2_S1_d0,                         \
                             opnd_ae_sem_mul_x2_S1_d1)                         \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulfp32x16x2ras_h((__opnd_ae_sem_mul_x2_S1_d0),        \
                                          (__opnd_ae_sem_mul_x2_S1_d1));       \
  })

#define AE_MULFP32X16X2RAS_H_S2(ae_mul_S2_d0, ae_mul_S2_d1)                    \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulfp32x16x2ras_h_s2((__ae_mul_S2_d0),                 \
                                             (__ae_mul_S2_d1));                \
  })

#define AE_MULFP32X16X2RAS_L(opnd_ae_sem_mul_x2_S1_d0,                         \
                             opnd_ae_sem_mul_x2_S1_d1)                         \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulfp32x16x2ras_l((__opnd_ae_sem_mul_x2_S1_d0),        \
                                          (__opnd_ae_sem_mul_x2_S1_d1));       \
  })

#define AE_MULFP32X16X2RAS_L_S2(ae_mul_S2_d0, ae_mul_S2_d1)                    \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulfp32x16x2ras_l_s2((__ae_mul_S2_d0),                 \
                                             (__ae_mul_S2_d1));                \
  })

#define AE_MULFP32X16X2RS_H(opnd_ae_sem_mul_x2_S1_d0,                          \
                            opnd_ae_sem_mul_x2_S1_d1)                          \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulfp32x16x2rs_h((__opnd_ae_sem_mul_x2_S1_d0),         \
                                         (__opnd_ae_sem_mul_x2_S1_d1));        \
  })

#define AE_MULFP32X16X2RS_H_S2(ae_mul_S2_d0, ae_mul_S2_d1)                     \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulfp32x16x2rs_h_s2((__ae_mul_S2_d0),                  \
                                            (__ae_mul_S2_d1));                 \
  })

#define AE_MULFP32X16X2RS_L(opnd_ae_sem_mul_x2_S1_d0,                          \
                            opnd_ae_sem_mul_x2_S1_d1)                          \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulfp32x16x2rs_l((__opnd_ae_sem_mul_x2_S1_d0),         \
                                         (__opnd_ae_sem_mul_x2_S1_d1));        \
  })

#define AE_MULFP32X16X2RS_L_S2(ae_mul_S2_d0, ae_mul_S2_d1)                     \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulfp32x16x2rs_l_s2((__ae_mul_S2_d0),                  \
                                            (__ae_mul_S2_d1));                 \
  })

#define AE_MULFP32X2RAS(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1)    \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulfp32x2ras((__opnd_ae_sem_mul_x2_S1_d0),             \
                                     (__opnd_ae_sem_mul_x2_S1_d1));            \
  })

#define AE_MULFP32X2RS(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1)     \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulfp32x2rs((__opnd_ae_sem_mul_x2_S1_d0),              \
                                    (__opnd_ae_sem_mul_x2_S1_d1));             \
  })

#define AE_MULFQ32SP24S_H_S2(ae_mul_S2_d0, ae_mul_S2_d1)                       \
  ({                                                                           \
    ae_int64 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                  \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulfq32sp24s_h_s2((__ae_mul_S2_d0), (__ae_mul_S2_d1)); \
  })

#define AE_MULFQ32SP24S_L_S2(ae_mul_S2_d0, ae_mul_S2_d1)                       \
  ({                                                                           \
    ae_int64 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                  \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulfq32sp24s_l_s2((__ae_mul_S2_d0), (__ae_mul_S2_d1)); \
  })

#define AE_MULP24X2(ae_mul_d0, ae_mul_d1)                                      \
  ({                                                                           \
    ae_int32x2 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int32x2 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulp24x2((__ae_mul_d0), (__ae_mul_d1));                \
  })

#define AE_MULP24X2_S2(ae_mul_S2_d0, ae_mul_S2_d1)                             \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulp24x2_s2((__ae_mul_S2_d0), (__ae_mul_S2_d1));       \
  })

#define AE_MULP32X16X2_H(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1)   \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulp32x16x2_h((__opnd_ae_sem_mul_x2_S1_d0),            \
                                      (__opnd_ae_sem_mul_x2_S1_d1));           \
  })

#define AE_MULP32X16X2_L(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1)   \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulp32x16x2_l((__opnd_ae_sem_mul_x2_S1_d0),            \
                                      (__opnd_ae_sem_mul_x2_S1_d1));           \
  })

#define AE_MULP32X2(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1)        \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulp32x2((__opnd_ae_sem_mul_x2_S1_d0),                 \
                                 (__opnd_ae_sem_mul_x2_S1_d1));                \
  })

#define AE_MULQ32SP16S_L_S2(ae_mul_S2_d0, ae_mul_S2_d1)                        \
  ({                                                                           \
    ae_int64 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                  \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulq32sp16s_l_s2((__ae_mul_S2_d0), (__ae_mul_S2_d1));  \
  })

#define AE_MULQ32SP16U_L_S2(ae_mul_S2_d0, ae_mul_S2_d1)                        \
  ({                                                                           \
    ae_int64 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                  \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulq32sp16u_l_s2((__ae_mul_S2_d0), (__ae_mul_S2_d1));  \
  })

#define AE_MULRFQ32SP24S_H_S2(ae_mul_S2_d0, ae_mul_S2_d1)                      \
  ({                                                                           \
    ae_int64 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                  \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulrfq32sp24s_h_s2((__ae_mul_S2_d0),                   \
                                           (__ae_mul_S2_d1));                  \
  })

#define AE_MULRFQ32SP24S_L_S2(ae_mul_S2_d0, ae_mul_S2_d1)                      \
  ({                                                                           \
    ae_int64 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                  \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulrfq32sp24s_l_s2((__ae_mul_S2_d0),                   \
                                           (__ae_mul_S2_d1));                  \
  })

#define AE_MULS16X4(ae_mul_q1, ae_mul_q0, ae_mul_d1, ae_mul_d0)                \
  ({                                                                           \
    ae_int16x4 __ae_mul_d1 = (ae_mul_d1);                                      \
    ae_int16x4 __ae_mul_d0 = (ae_mul_d0);                                      \
    __builtin_xtensa_ae_muls16x4((ae_int32x2 *)&(ae_mul_q1),                   \
                                 (ae_int32x2 *)&(ae_mul_q0), (__ae_mul_d1),    \
                                 (__ae_mul_d0));                               \
  })

#define AE_MULS32_HH(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,       \
                     opnd_ae_sem_mul_x2_S1_d1)                                 \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_muls32_hh((ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0),     \
                                  (__opnd_ae_sem_mul_x2_S1_d0),                \
                                  (__opnd_ae_sem_mul_x2_S1_d1));               \
  })

#define AE_MULS32_LH(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,       \
                     opnd_ae_sem_mul_x2_S1_d1)                                 \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_muls32_lh((ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0),     \
                                  (__opnd_ae_sem_mul_x2_S1_d0),                \
                                  (__opnd_ae_sem_mul_x2_S1_d1));               \
  })

#define AE_MULS32_LL(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,       \
                     opnd_ae_sem_mul_x2_S1_d1)                                 \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_muls32_ll((ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0),     \
                                  (__opnd_ae_sem_mul_x2_S1_d0),                \
                                  (__opnd_ae_sem_mul_x2_S1_d1));               \
  })

#define AE_MULS32F48P16S_HH(opnd_ae_sem_mul_x2_S1_d0,                          \
                            opnd_ae_sem_mul_x2_S1_d1)                          \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_muls32f48p16s_hh((__opnd_ae_sem_mul_x2_S1_d0),         \
                                         (__opnd_ae_sem_mul_x2_S1_d1));        \
  })

#define AE_MULS32F48P16S_HH_S2(ae_mul_S2_d0, ae_mul_S2_d1)                     \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_muls32f48p16s_hh_s2((__ae_mul_S2_d0),                  \
                                            (__ae_mul_S2_d1));                 \
  })

#define AE_MULS32F48P16S_LH(opnd_ae_sem_mul_x2_S1_d0,                          \
                            opnd_ae_sem_mul_x2_S1_d1)                          \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_muls32f48p16s_lh((__opnd_ae_sem_mul_x2_S1_d0),         \
                                         (__opnd_ae_sem_mul_x2_S1_d1));        \
  })

#define AE_MULS32F48P16S_LH_S2(ae_mul_S2_d0, ae_mul_S2_d1)                     \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_muls32f48p16s_lh_s2((__ae_mul_S2_d0),                  \
                                            (__ae_mul_S2_d1));                 \
  })

#define AE_MULS32F48P16S_LL(opnd_ae_sem_mul_x2_S1_d0,                          \
                            opnd_ae_sem_mul_x2_S1_d1)                          \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_muls32f48p16s_ll((__opnd_ae_sem_mul_x2_S1_d0),         \
                                         (__opnd_ae_sem_mul_x2_S1_d1));        \
  })

#define AE_MULS32F48P16S_LL_S2(ae_mul_S2_d0, ae_mul_S2_d1)                     \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_muls32f48p16s_ll_s2((__ae_mul_S2_d0),                  \
                                            (__ae_mul_S2_d1));                 \
  })

#define AE_MULS32U_LL(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,      \
                      opnd_ae_sem_mul_x2_S1_d1)                                \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_muls32u_ll((ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0),    \
                                   (__opnd_ae_sem_mul_x2_S1_d0),               \
                                   (__opnd_ae_sem_mul_x2_S1_d1));              \
  })

#define AE_MULS32X16_H0(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,    \
                        opnd_ae_sem_mul_x2_S1_d1)                              \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_muls32x16_h0((ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0),  \
                                     (__opnd_ae_sem_mul_x2_S1_d0),             \
                                     (__opnd_ae_sem_mul_x2_S1_d1));            \
  })

#define AE_MULS32X16_H0_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)           \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_muls32x16_h0_s2((ae_int64 *)&(ae_mul_S2_q0),           \
                                        (__ae_mul_S2_d0), (__ae_mul_S2_d1));   \
  })

#define AE_MULS32X16_H1(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,    \
                        opnd_ae_sem_mul_x2_S1_d1)                              \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_muls32x16_h1((ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0),  \
                                     (__opnd_ae_sem_mul_x2_S1_d0),             \
                                     (__opnd_ae_sem_mul_x2_S1_d1));            \
  })

#define AE_MULS32X16_H1_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)           \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_muls32x16_h1_s2((ae_int64 *)&(ae_mul_S2_q0),           \
                                        (__ae_mul_S2_d0), (__ae_mul_S2_d1));   \
  })

#define AE_MULS32X16_H2(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,    \
                        opnd_ae_sem_mul_x2_S1_d1)                              \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_muls32x16_h2((ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0),  \
                                     (__opnd_ae_sem_mul_x2_S1_d0),             \
                                     (__opnd_ae_sem_mul_x2_S1_d1));            \
  })

#define AE_MULS32X16_H2_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)           \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_muls32x16_h2_s2((ae_int64 *)&(ae_mul_S2_q0),           \
                                        (__ae_mul_S2_d0), (__ae_mul_S2_d1));   \
  })

#define AE_MULS32X16_H3(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,    \
                        opnd_ae_sem_mul_x2_S1_d1)                              \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_muls32x16_h3((ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0),  \
                                     (__opnd_ae_sem_mul_x2_S1_d0),             \
                                     (__opnd_ae_sem_mul_x2_S1_d1));            \
  })

#define AE_MULS32X16_H3_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)           \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_muls32x16_h3_s2((ae_int64 *)&(ae_mul_S2_q0),           \
                                        (__ae_mul_S2_d0), (__ae_mul_S2_d1));   \
  })

#define AE_MULS32X16_L0(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,    \
                        opnd_ae_sem_mul_x2_S1_d1)                              \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_muls32x16_l0((ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0),  \
                                     (__opnd_ae_sem_mul_x2_S1_d0),             \
                                     (__opnd_ae_sem_mul_x2_S1_d1));            \
  })

#define AE_MULS32X16_L0_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)           \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_muls32x16_l0_s2((ae_int64 *)&(ae_mul_S2_q0),           \
                                        (__ae_mul_S2_d0), (__ae_mul_S2_d1));   \
  })

#define AE_MULS32X16_L1(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,    \
                        opnd_ae_sem_mul_x2_S1_d1)                              \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_muls32x16_l1((ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0),  \
                                     (__opnd_ae_sem_mul_x2_S1_d0),             \
                                     (__opnd_ae_sem_mul_x2_S1_d1));            \
  })

#define AE_MULS32X16_L1_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)           \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_muls32x16_l1_s2((ae_int64 *)&(ae_mul_S2_q0),           \
                                        (__ae_mul_S2_d0), (__ae_mul_S2_d1));   \
  })

#define AE_MULS32X16_L2(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,    \
                        opnd_ae_sem_mul_x2_S1_d1)                              \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_muls32x16_l2((ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0),  \
                                     (__opnd_ae_sem_mul_x2_S1_d0),             \
                                     (__opnd_ae_sem_mul_x2_S1_d1));            \
  })

#define AE_MULS32X16_L2_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)           \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_muls32x16_l2_s2((ae_int64 *)&(ae_mul_S2_q0),           \
                                        (__ae_mul_S2_d0), (__ae_mul_S2_d1));   \
  })

#define AE_MULS32X16_L3(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,    \
                        opnd_ae_sem_mul_x2_S1_d1)                              \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_muls32x16_l3((ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0),  \
                                     (__opnd_ae_sem_mul_x2_S1_d0),             \
                                     (__opnd_ae_sem_mul_x2_S1_d1));            \
  })

#define AE_MULS32X16_L3_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)           \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_muls32x16_l3_s2((ae_int64 *)&(ae_mul_S2_q0),           \
                                        (__ae_mul_S2_d0), (__ae_mul_S2_d1));   \
  })

#define AE_MULSAD24_HH_LL(ae_mul_q0, ae_mul_d0, ae_mul_d1)                     \
  ({                                                                           \
    ae_int32x2 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int32x2 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulsad24_hh_ll((ae_int64 *)&(ae_mul_q0),               \
                                       (__ae_mul_d0), (__ae_mul_d1));          \
  })

#define AE_MULSAD24_HH_LL_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)         \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulsad24_hh_ll_s2((ae_int64 *)&(ae_mul_S2_q0),         \
                                          (__ae_mul_S2_d0), (__ae_mul_S2_d1)); \
  })

#define AE_MULSAD32X16_H1_L0(ae_mul_q0, ae_mul_d0, ae_mul_d1)                  \
  ({                                                                           \
    ae_int32x2 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int16x4 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulsad32x16_h1_l0((ae_int64 *)&(ae_mul_q0),            \
                                          (__ae_mul_d0), (__ae_mul_d1));       \
  })

#define AE_MULSAD32X16_H1_L0_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)      \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulsad32x16_h1_l0_s2(                                  \
        (ae_int64 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));      \
  })

#define AE_MULSAD32X16_H3_L2(ae_mul_q0, ae_mul_d0, ae_mul_d1)                  \
  ({                                                                           \
    ae_int32x2 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int16x4 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulsad32x16_h3_l2((ae_int64 *)&(ae_mul_q0),            \
                                          (__ae_mul_d0), (__ae_mul_d1));       \
  })

#define AE_MULSAD32X16_H3_L2_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)      \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulsad32x16_h3_l2_s2(                                  \
        (ae_int64 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));      \
  })

#define AE_MULSAFD24_HH_LL(ae_mul_q0, ae_mul_d0, ae_mul_d1)                    \
  ({                                                                           \
    ae_int32x2 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int32x2 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulsafd24_hh_ll((ae_int64 *)&(ae_mul_q0),              \
                                        (__ae_mul_d0), (__ae_mul_d1));         \
  })

#define AE_MULSAFD24_HH_LL_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)        \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulsafd24_hh_ll_s2(                                    \
        (ae_int64 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));      \
  })

#define AE_MULSAFD32X16_H1_L0(opnd_ae_sem_mul_x2_S1_q0,                        \
                              opnd_ae_sem_mul_x2_S1_d0,                        \
                              opnd_ae_sem_mul_x2_S1_d1)                        \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulsafd32x16_h1_l0(                                    \
        (ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0), (__opnd_ae_sem_mul_x2_S1_d0), \
        (__opnd_ae_sem_mul_x2_S1_d1));                                         \
  })

#define AE_MULSAFD32X16_H1_L0_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)     \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulsafd32x16_h1_l0_s2(                                 \
        (ae_int64 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));      \
  })

#define AE_MULSAFD32X16_H3_L2(opnd_ae_sem_mul_x2_S1_q0,                        \
                              opnd_ae_sem_mul_x2_S1_d0,                        \
                              opnd_ae_sem_mul_x2_S1_d1)                        \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulsafd32x16_h3_l2(                                    \
        (ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0), (__opnd_ae_sem_mul_x2_S1_d0), \
        (__opnd_ae_sem_mul_x2_S1_d1));                                         \
  })

#define AE_MULSAFD32X16_H3_L2_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)     \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulsafd32x16_h3_l2_s2(                                 \
        (ae_int64 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));      \
  })

#define AE_MULSF16SS_00(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,    \
                        opnd_ae_sem_mul_x2_S1_d1)                              \
  ({                                                                           \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulsf16ss_00(                                          \
        (ae_int32x2 *)&(opnd_ae_sem_mul_x2_S1_q0),                             \
        (__opnd_ae_sem_mul_x2_S1_d0), (__opnd_ae_sem_mul_x2_S1_d1));           \
  })

#define AE_MULSF16SS_00_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)           \
  ({                                                                           \
    ae_int16x4 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulsf16ss_00_s2((ae_int32x2 *)&(ae_mul_S2_q0),         \
                                        (__ae_mul_S2_d0), (__ae_mul_S2_d1));   \
  })

#define AE_MULSF16SS_10(ae_mul_q0, ae_mul_d0, ae_mul_d1)                       \
  ({                                                                           \
    ae_int16x4 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int16x4 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulsf16ss_10((ae_int32x2 *)&(ae_mul_q0),               \
                                     (__ae_mul_d0), (__ae_mul_d1));            \
  })

#define AE_MULSF16SS_11(ae_mul_q0, ae_mul_d0, ae_mul_d1)                       \
  ({                                                                           \
    ae_int16x4 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int16x4 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulsf16ss_11((ae_int32x2 *)&(ae_mul_q0),               \
                                     (__ae_mul_d0), (__ae_mul_d1));            \
  })

#define AE_MULSF16SS_20(ae_mul_q0, ae_mul_d0, ae_mul_d1)                       \
  ({                                                                           \
    ae_int16x4 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int16x4 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulsf16ss_20((ae_int32x2 *)&(ae_mul_q0),               \
                                     (__ae_mul_d0), (__ae_mul_d1));            \
  })

#define AE_MULSF16SS_21(ae_mul_q0, ae_mul_d0, ae_mul_d1)                       \
  ({                                                                           \
    ae_int16x4 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int16x4 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulsf16ss_21((ae_int32x2 *)&(ae_mul_q0),               \
                                     (__ae_mul_d0), (__ae_mul_d1));            \
  })

#define AE_MULSF16SS_22(ae_mul_q0, ae_mul_d0, ae_mul_d1)                       \
  ({                                                                           \
    ae_int16x4 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int16x4 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulsf16ss_22((ae_int32x2 *)&(ae_mul_q0),               \
                                     (__ae_mul_d0), (__ae_mul_d1));            \
  })

#define AE_MULSF16SS_30(ae_mul_q0, ae_mul_d0, ae_mul_d1)                       \
  ({                                                                           \
    ae_int16x4 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int16x4 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulsf16ss_30((ae_int32x2 *)&(ae_mul_q0),               \
                                     (__ae_mul_d0), (__ae_mul_d1));            \
  })

#define AE_MULSF16SS_31(ae_mul_q0, ae_mul_d0, ae_mul_d1)                       \
  ({                                                                           \
    ae_int16x4 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int16x4 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulsf16ss_31((ae_int32x2 *)&(ae_mul_q0),               \
                                     (__ae_mul_d0), (__ae_mul_d1));            \
  })

#define AE_MULSF16SS_32(ae_mul_q0, ae_mul_d0, ae_mul_d1)                       \
  ({                                                                           \
    ae_int16x4 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int16x4 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulsf16ss_32((ae_int32x2 *)&(ae_mul_q0),               \
                                     (__ae_mul_d0), (__ae_mul_d1));            \
  })

#define AE_MULSF16SS_33(ae_mul_q0, ae_mul_d0, ae_mul_d1)                       \
  ({                                                                           \
    ae_int16x4 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int16x4 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulsf16ss_33((ae_int32x2 *)&(ae_mul_q0),               \
                                     (__ae_mul_d0), (__ae_mul_d1));            \
  })

#define AE_MULSF16X4SS(ae_mul_q1, ae_mul_q0, ae_mul_d1, ae_mul_d0)             \
  ({                                                                           \
    ae_int16x4 __ae_mul_d1 = (ae_mul_d1);                                      \
    ae_int16x4 __ae_mul_d0 = (ae_mul_d0);                                      \
    __builtin_xtensa_ae_mulsf16x4ss((ae_int32x2 *)&(ae_mul_q1),                \
                                    (ae_int32x2 *)&(ae_mul_q0), (__ae_mul_d1), \
                                    (__ae_mul_d0));                            \
  })

#define AE_MULSF32R_HH(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,     \
                       opnd_ae_sem_mul_x2_S1_d1)                               \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulsf32r_hh((ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0),   \
                                    (__opnd_ae_sem_mul_x2_S1_d0),              \
                                    (__opnd_ae_sem_mul_x2_S1_d1));             \
  })

#define AE_MULSF32R_LH(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,     \
                       opnd_ae_sem_mul_x2_S1_d1)                               \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulsf32r_lh((ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0),   \
                                    (__opnd_ae_sem_mul_x2_S1_d0),              \
                                    (__opnd_ae_sem_mul_x2_S1_d1));             \
  })

#define AE_MULSF32R_LL(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,     \
                       opnd_ae_sem_mul_x2_S1_d1)                               \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulsf32r_ll((ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0),   \
                                    (__opnd_ae_sem_mul_x2_S1_d0),              \
                                    (__opnd_ae_sem_mul_x2_S1_d1));             \
  })

#define AE_MULSF32R_LL_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)            \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulsf32r_ll_s2((ae_int64 *)&(ae_mul_S2_q0),            \
                                       (__ae_mul_S2_d0), (__ae_mul_S2_d1));    \
  })

#define AE_MULSF32S_HH(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,     \
                       opnd_ae_sem_mul_x2_S1_d1)                               \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulsf32s_hh((ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0),   \
                                    (__opnd_ae_sem_mul_x2_S1_d0),              \
                                    (__opnd_ae_sem_mul_x2_S1_d1));             \
  })

#define AE_MULSF32S_LH(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,     \
                       opnd_ae_sem_mul_x2_S1_d1)                               \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulsf32s_lh((ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0),   \
                                    (__opnd_ae_sem_mul_x2_S1_d0),              \
                                    (__opnd_ae_sem_mul_x2_S1_d1));             \
  })

#define AE_MULSF32S_LL(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,     \
                       opnd_ae_sem_mul_x2_S1_d1)                               \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulsf32s_ll((ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0),   \
                                    (__opnd_ae_sem_mul_x2_S1_d0),              \
                                    (__opnd_ae_sem_mul_x2_S1_d1));             \
  })

#define AE_MULSF32X16_H0(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,   \
                         opnd_ae_sem_mul_x2_S1_d1)                             \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulsf32x16_h0((ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0), \
                                      (__opnd_ae_sem_mul_x2_S1_d0),            \
                                      (__opnd_ae_sem_mul_x2_S1_d1));           \
  })

#define AE_MULSF32X16_H0_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)          \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulsf32x16_h0_s2((ae_int64 *)&(ae_mul_S2_q0),          \
                                         (__ae_mul_S2_d0), (__ae_mul_S2_d1));  \
  })

#define AE_MULSF32X16_H1(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,   \
                         opnd_ae_sem_mul_x2_S1_d1)                             \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulsf32x16_h1((ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0), \
                                      (__opnd_ae_sem_mul_x2_S1_d0),            \
                                      (__opnd_ae_sem_mul_x2_S1_d1));           \
  })

#define AE_MULSF32X16_H1_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)          \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulsf32x16_h1_s2((ae_int64 *)&(ae_mul_S2_q0),          \
                                         (__ae_mul_S2_d0), (__ae_mul_S2_d1));  \
  })

#define AE_MULSF32X16_H2(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,   \
                         opnd_ae_sem_mul_x2_S1_d1)                             \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulsf32x16_h2((ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0), \
                                      (__opnd_ae_sem_mul_x2_S1_d0),            \
                                      (__opnd_ae_sem_mul_x2_S1_d1));           \
  })

#define AE_MULSF32X16_H2_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)          \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulsf32x16_h2_s2((ae_int64 *)&(ae_mul_S2_q0),          \
                                         (__ae_mul_S2_d0), (__ae_mul_S2_d1));  \
  })

#define AE_MULSF32X16_H3(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,   \
                         opnd_ae_sem_mul_x2_S1_d1)                             \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulsf32x16_h3((ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0), \
                                      (__opnd_ae_sem_mul_x2_S1_d0),            \
                                      (__opnd_ae_sem_mul_x2_S1_d1));           \
  })

#define AE_MULSF32X16_H3_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)          \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulsf32x16_h3_s2((ae_int64 *)&(ae_mul_S2_q0),          \
                                         (__ae_mul_S2_d0), (__ae_mul_S2_d1));  \
  })

#define AE_MULSF32X16_L0(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,   \
                         opnd_ae_sem_mul_x2_S1_d1)                             \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulsf32x16_l0((ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0), \
                                      (__opnd_ae_sem_mul_x2_S1_d0),            \
                                      (__opnd_ae_sem_mul_x2_S1_d1));           \
  })

#define AE_MULSF32X16_L0_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)          \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulsf32x16_l0_s2((ae_int64 *)&(ae_mul_S2_q0),          \
                                         (__ae_mul_S2_d0), (__ae_mul_S2_d1));  \
  })

#define AE_MULSF32X16_L1(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,   \
                         opnd_ae_sem_mul_x2_S1_d1)                             \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulsf32x16_l1((ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0), \
                                      (__opnd_ae_sem_mul_x2_S1_d0),            \
                                      (__opnd_ae_sem_mul_x2_S1_d1));           \
  })

#define AE_MULSF32X16_L1_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)          \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulsf32x16_l1_s2((ae_int64 *)&(ae_mul_S2_q0),          \
                                         (__ae_mul_S2_d0), (__ae_mul_S2_d1));  \
  })

#define AE_MULSF32X16_L2(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,   \
                         opnd_ae_sem_mul_x2_S1_d1)                             \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulsf32x16_l2((ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0), \
                                      (__opnd_ae_sem_mul_x2_S1_d0),            \
                                      (__opnd_ae_sem_mul_x2_S1_d1));           \
  })

#define AE_MULSF32X16_L2_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)          \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulsf32x16_l2_s2((ae_int64 *)&(ae_mul_S2_q0),          \
                                         (__ae_mul_S2_d0), (__ae_mul_S2_d1));  \
  })

#define AE_MULSF32X16_L3(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,   \
                         opnd_ae_sem_mul_x2_S1_d1)                             \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulsf32x16_l3((ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0), \
                                      (__opnd_ae_sem_mul_x2_S1_d0),            \
                                      (__opnd_ae_sem_mul_x2_S1_d1));           \
  })

#define AE_MULSF32X16_L3_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)          \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulsf32x16_l3_s2((ae_int64 *)&(ae_mul_S2_q0),          \
                                         (__ae_mul_S2_d0), (__ae_mul_S2_d1));  \
  })

#define AE_MULSF48Q32SP16S_L(opnd_ae_sem_mul_x2_S1_q0,                         \
                             opnd_ae_sem_mul_x2_S1_d0,                         \
                             opnd_ae_sem_mul_x2_S1_d1)                         \
  ({                                                                           \
    ae_int64 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);          \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulsf48q32sp16s_l(                                     \
        (ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0), (__opnd_ae_sem_mul_x2_S1_d0), \
        (__opnd_ae_sem_mul_x2_S1_d1));                                         \
  })

#define AE_MULSF48Q32SP16S_L_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)      \
  ({                                                                           \
    ae_int64 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                  \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulsf48q32sp16s_l_s2(                                  \
        (ae_int64 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));      \
  })

#define AE_MULSF48Q32SP16U_L(opnd_ae_sem_mul_x2_S1_q0,                         \
                             opnd_ae_sem_mul_x2_S1_d0,                         \
                             opnd_ae_sem_mul_x2_S1_d1)                         \
  ({                                                                           \
    ae_int64 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);          \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulsf48q32sp16u_l(                                     \
        (ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0), (__opnd_ae_sem_mul_x2_S1_d0), \
        (__opnd_ae_sem_mul_x2_S1_d1));                                         \
  })

#define AE_MULSF48Q32SP16U_L_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)      \
  ({                                                                           \
    ae_int64 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                  \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulsf48q32sp16u_l_s2(                                  \
        (ae_int64 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));      \
  })

#define AE_MULSFP24X2R(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,     \
                       opnd_ae_sem_mul_x2_S1_d1)                               \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulsfp24x2r((ae_int32x2 *)&(opnd_ae_sem_mul_x2_S1_q0), \
                                    (__opnd_ae_sem_mul_x2_S1_d0),              \
                                    (__opnd_ae_sem_mul_x2_S1_d1));             \
  })

#define AE_MULSFP24X2R_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)            \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulsfp24x2r_s2((ae_int32x2 *)&(ae_mul_S2_q0),          \
                                       (__ae_mul_S2_d0), (__ae_mul_S2_d1));    \
  })

#define AE_MULSFP24X2RA(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,    \
                        opnd_ae_sem_mul_x2_S1_d1)                              \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulsfp24x2ra(                                          \
        (ae_int32x2 *)&(opnd_ae_sem_mul_x2_S1_q0),                             \
        (__opnd_ae_sem_mul_x2_S1_d0), (__opnd_ae_sem_mul_x2_S1_d1));           \
  })

#define AE_MULSFP24X2RA_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)           \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulsfp24x2ra_s2((ae_int32x2 *)&(ae_mul_S2_q0),         \
                                        (__ae_mul_S2_d0), (__ae_mul_S2_d1));   \
  })

#define AE_MULSFP32X16X2RAS_H(opnd_ae_sem_mul_x2_S1_q0,                        \
                              opnd_ae_sem_mul_x2_S1_d0,                        \
                              opnd_ae_sem_mul_x2_S1_d1)                        \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulsfp32x16x2ras_h(                                    \
        (ae_int32x2 *)&(opnd_ae_sem_mul_x2_S1_q0),                             \
        (__opnd_ae_sem_mul_x2_S1_d0), (__opnd_ae_sem_mul_x2_S1_d1));           \
  })

#define AE_MULSFP32X16X2RAS_H_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)     \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulsfp32x16x2ras_h_s2(                                 \
        (ae_int32x2 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));    \
  })

#define AE_MULSFP32X16X2RAS_L(opnd_ae_sem_mul_x2_S1_q0,                        \
                              opnd_ae_sem_mul_x2_S1_d0,                        \
                              opnd_ae_sem_mul_x2_S1_d1)                        \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulsfp32x16x2ras_l(                                    \
        (ae_int32x2 *)&(opnd_ae_sem_mul_x2_S1_q0),                             \
        (__opnd_ae_sem_mul_x2_S1_d0), (__opnd_ae_sem_mul_x2_S1_d1));           \
  })

#define AE_MULSFP32X16X2RAS_L_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)     \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulsfp32x16x2ras_l_s2(                                 \
        (ae_int32x2 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));    \
  })

#define AE_MULSFP32X16X2RS_H(opnd_ae_sem_mul_x2_S1_q0,                         \
                             opnd_ae_sem_mul_x2_S1_d0,                         \
                             opnd_ae_sem_mul_x2_S1_d1)                         \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulsfp32x16x2rs_h(                                     \
        (ae_int32x2 *)&(opnd_ae_sem_mul_x2_S1_q0),                             \
        (__opnd_ae_sem_mul_x2_S1_d0), (__opnd_ae_sem_mul_x2_S1_d1));           \
  })

#define AE_MULSFP32X16X2RS_H_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)      \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulsfp32x16x2rs_h_s2(                                  \
        (ae_int32x2 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));    \
  })

#define AE_MULSFP32X16X2RS_L(opnd_ae_sem_mul_x2_S1_q0,                         \
                             opnd_ae_sem_mul_x2_S1_d0,                         \
                             opnd_ae_sem_mul_x2_S1_d1)                         \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulsfp32x16x2rs_l(                                     \
        (ae_int32x2 *)&(opnd_ae_sem_mul_x2_S1_q0),                             \
        (__opnd_ae_sem_mul_x2_S1_d0), (__opnd_ae_sem_mul_x2_S1_d1));           \
  })

#define AE_MULSFP32X16X2RS_L_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)      \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulsfp32x16x2rs_l_s2(                                  \
        (ae_int32x2 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));    \
  })

#define AE_MULSFP32X2RAS(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,   \
                         opnd_ae_sem_mul_x2_S1_d1)                             \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulsfp32x2ras(                                         \
        (ae_int32x2 *)&(opnd_ae_sem_mul_x2_S1_q0),                             \
        (__opnd_ae_sem_mul_x2_S1_d0), (__opnd_ae_sem_mul_x2_S1_d1));           \
  })

#define AE_MULSFP32X2RS(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,    \
                        opnd_ae_sem_mul_x2_S1_d1)                              \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulsfp32x2rs(                                          \
        (ae_int32x2 *)&(opnd_ae_sem_mul_x2_S1_q0),                             \
        (__opnd_ae_sem_mul_x2_S1_d0), (__opnd_ae_sem_mul_x2_S1_d1));           \
  })

#define AE_MULSFQ32SP24S_H_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)        \
  ({                                                                           \
    ae_int64 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                  \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulsfq32sp24s_h_s2(                                    \
        (ae_int64 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));      \
  })

#define AE_MULSFQ32SP24S_L_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)        \
  ({                                                                           \
    ae_int64 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                  \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulsfq32sp24s_l_s2(                                    \
        (ae_int64 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));      \
  })

#define AE_MULSP24X2(ae_mul_q0, ae_mul_d0, ae_mul_d1)                          \
  ({                                                                           \
    ae_int32x2 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int32x2 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulsp24x2((ae_int32x2 *)&(ae_mul_q0), (__ae_mul_d0),   \
                                  (__ae_mul_d1));                              \
  })

#define AE_MULSP24X2_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)              \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulsp24x2_s2((ae_int32x2 *)&(ae_mul_S2_q0),            \
                                     (__ae_mul_S2_d0), (__ae_mul_S2_d1));      \
  })

#define AE_MULSP32X16X2_H(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,  \
                          opnd_ae_sem_mul_x2_S1_d1)                            \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulsp32x16x2_h(                                        \
        (ae_int32x2 *)&(opnd_ae_sem_mul_x2_S1_q0),                             \
        (__opnd_ae_sem_mul_x2_S1_d0), (__opnd_ae_sem_mul_x2_S1_d1));           \
  })

#define AE_MULSP32X16X2_L(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,  \
                          opnd_ae_sem_mul_x2_S1_d1)                            \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulsp32x16x2_l(                                        \
        (ae_int32x2 *)&(opnd_ae_sem_mul_x2_S1_q0),                             \
        (__opnd_ae_sem_mul_x2_S1_d0), (__opnd_ae_sem_mul_x2_S1_d1));           \
  })

#define AE_MULSP32X2(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0,       \
                     opnd_ae_sem_mul_x2_S1_d1)                                 \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulsp32x2((ae_int32x2 *)&(opnd_ae_sem_mul_x2_S1_q0),   \
                                  (__opnd_ae_sem_mul_x2_S1_d0),                \
                                  (__opnd_ae_sem_mul_x2_S1_d1));               \
  })

#define AE_MULSQ32SP16S_L_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)         \
  ({                                                                           \
    ae_int64 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                  \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulsq32sp16s_l_s2((ae_int64 *)&(ae_mul_S2_q0),         \
                                          (__ae_mul_S2_d0), (__ae_mul_S2_d1)); \
  })

#define AE_MULSQ32SP16U_L_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)         \
  ({                                                                           \
    ae_int64 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                  \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulsq32sp16u_l_s2((ae_int64 *)&(ae_mul_S2_q0),         \
                                          (__ae_mul_S2_d0), (__ae_mul_S2_d1)); \
  })

#define AE_MULSRFQ32SP24S_H_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)       \
  ({                                                                           \
    ae_int64 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                  \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulsrfq32sp24s_h_s2(                                   \
        (ae_int64 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));      \
  })

#define AE_MULSRFQ32SP24S_L_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)       \
  ({                                                                           \
    ae_int64 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                  \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulsrfq32sp24s_l_s2(                                   \
        (ae_int64 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));      \
  })

#define AE_MULSS32F48P16S_HH(opnd_ae_sem_mul_x2_S1_q0,                         \
                             opnd_ae_sem_mul_x2_S1_d0,                         \
                             opnd_ae_sem_mul_x2_S1_d1)                         \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulss32f48p16s_hh(                                     \
        (ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0), (__opnd_ae_sem_mul_x2_S1_d0), \
        (__opnd_ae_sem_mul_x2_S1_d1));                                         \
  })

#define AE_MULSS32F48P16S_HH_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)      \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulss32f48p16s_hh_s2(                                  \
        (ae_int64 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));      \
  })

#define AE_MULSS32F48P16S_LH(opnd_ae_sem_mul_x2_S1_q0,                         \
                             opnd_ae_sem_mul_x2_S1_d0,                         \
                             opnd_ae_sem_mul_x2_S1_d1)                         \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulss32f48p16s_lh(                                     \
        (ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0), (__opnd_ae_sem_mul_x2_S1_d0), \
        (__opnd_ae_sem_mul_x2_S1_d1));                                         \
  })

#define AE_MULSS32F48P16S_LH_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)      \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulss32f48p16s_lh_s2(                                  \
        (ae_int64 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));      \
  })

#define AE_MULSS32F48P16S_LL(opnd_ae_sem_mul_x2_S1_q0,                         \
                             opnd_ae_sem_mul_x2_S1_d0,                         \
                             opnd_ae_sem_mul_x2_S1_d1)                         \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulss32f48p16s_ll(                                     \
        (ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0), (__opnd_ae_sem_mul_x2_S1_d0), \
        (__opnd_ae_sem_mul_x2_S1_d1));                                         \
  })

#define AE_MULSS32F48P16S_LL_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)      \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulss32f48p16s_ll_s2(                                  \
        (ae_int64 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));      \
  })

#define AE_MULSSD24_HH_LL(ae_mul_q0, ae_mul_d0, ae_mul_d1)                     \
  ({                                                                           \
    ae_int32x2 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int32x2 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulssd24_hh_ll((ae_int64 *)&(ae_mul_q0),               \
                                       (__ae_mul_d0), (__ae_mul_d1));          \
  })

#define AE_MULSSD24_HH_LL_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)         \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulssd24_hh_ll_s2((ae_int64 *)&(ae_mul_S2_q0),         \
                                          (__ae_mul_S2_d0), (__ae_mul_S2_d1)); \
  })

#define AE_MULSSD24_HL_LH(ae_mul_q0, ae_mul_d0, ae_mul_d1)                     \
  ({                                                                           \
    ae_int32x2 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int32x2 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulssd24_hl_lh((ae_int64 *)&(ae_mul_q0),               \
                                       (__ae_mul_d0), (__ae_mul_d1));          \
  })

#define AE_MULSSD24_HL_LH_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)         \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulssd24_hl_lh_s2((ae_int64 *)&(ae_mul_S2_q0),         \
                                          (__ae_mul_S2_d0), (__ae_mul_S2_d1)); \
  })

#define AE_MULSSD32X16_H1_L0(opnd_ae_sem_mul_x2_S1_q0,                         \
                             opnd_ae_sem_mul_x2_S1_d0,                         \
                             opnd_ae_sem_mul_x2_S1_d1)                         \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulssd32x16_h1_l0(                                     \
        (ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0), (__opnd_ae_sem_mul_x2_S1_d0), \
        (__opnd_ae_sem_mul_x2_S1_d1));                                         \
  })

#define AE_MULSSD32X16_H1_L0_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)      \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulssd32x16_h1_l0_s2(                                  \
        (ae_int64 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));      \
  })

#define AE_MULSSD32X16_H3_L2(opnd_ae_sem_mul_x2_S1_q0,                         \
                             opnd_ae_sem_mul_x2_S1_d0,                         \
                             opnd_ae_sem_mul_x2_S1_d1)                         \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulssd32x16_h3_l2(                                     \
        (ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0), (__opnd_ae_sem_mul_x2_S1_d0), \
        (__opnd_ae_sem_mul_x2_S1_d1));                                         \
  })

#define AE_MULSSD32X16_H3_L2_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)      \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulssd32x16_h3_l2_s2(                                  \
        (ae_int64 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));      \
  })

#define AE_MULSSFD16SS_11_00(opnd_ae_sem_mul_x2_S1_q0,                         \
                             opnd_ae_sem_mul_x2_S1_d0,                         \
                             opnd_ae_sem_mul_x2_S1_d1)                         \
  ({                                                                           \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulssfd16ss_11_00(                                     \
        (ae_int32x2 *)&(opnd_ae_sem_mul_x2_S1_q0),                             \
        (__opnd_ae_sem_mul_x2_S1_d0), (__opnd_ae_sem_mul_x2_S1_d1));           \
  })

#define AE_MULSSFD16SS_11_00_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)      \
  ({                                                                           \
    ae_int16x4 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulssfd16ss_11_00_s2(                                  \
        (ae_int32x2 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));    \
  })

#define AE_MULSSFD16SS_13_02(opnd_ae_sem_mul_x2_S1_q0,                         \
                             opnd_ae_sem_mul_x2_S1_d0,                         \
                             opnd_ae_sem_mul_x2_S1_d1)                         \
  ({                                                                           \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulssfd16ss_13_02(                                     \
        (ae_int32x2 *)&(opnd_ae_sem_mul_x2_S1_q0),                             \
        (__opnd_ae_sem_mul_x2_S1_d0), (__opnd_ae_sem_mul_x2_S1_d1));           \
  })

#define AE_MULSSFD16SS_13_02_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)      \
  ({                                                                           \
    ae_int16x4 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulssfd16ss_13_02_s2(                                  \
        (ae_int32x2 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));    \
  })

#define AE_MULSSFD16SS_33_22(opnd_ae_sem_mul_x2_S1_q0,                         \
                             opnd_ae_sem_mul_x2_S1_d0,                         \
                             opnd_ae_sem_mul_x2_S1_d1)                         \
  ({                                                                           \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulssfd16ss_33_22(                                     \
        (ae_int32x2 *)&(opnd_ae_sem_mul_x2_S1_q0),                             \
        (__opnd_ae_sem_mul_x2_S1_d0), (__opnd_ae_sem_mul_x2_S1_d1));           \
  })

#define AE_MULSSFD16SS_33_22_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)      \
  ({                                                                           \
    ae_int16x4 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulssfd16ss_33_22_s2(                                  \
        (ae_int32x2 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));    \
  })

#define AE_MULSSFD24_HH_LL(ae_mul_q0, ae_mul_d0, ae_mul_d1)                    \
  ({                                                                           \
    ae_int32x2 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int32x2 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulssfd24_hh_ll((ae_int64 *)&(ae_mul_q0),              \
                                        (__ae_mul_d0), (__ae_mul_d1));         \
  })

#define AE_MULSSFD24_HH_LL_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)        \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulssfd24_hh_ll_s2(                                    \
        (ae_int64 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));      \
  })

#define AE_MULSSFD24_HL_LH(ae_mul_q0, ae_mul_d0, ae_mul_d1)                    \
  ({                                                                           \
    ae_int32x2 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int32x2 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulssfd24_hl_lh((ae_int64 *)&(ae_mul_q0),              \
                                        (__ae_mul_d0), (__ae_mul_d1));         \
  })

#define AE_MULSSFD24_HL_LH_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)        \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulssfd24_hl_lh_s2(                                    \
        (ae_int64 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));      \
  })

#define AE_MULSSFD32X16_H1_L0(opnd_ae_sem_mul_x2_S1_q0,                        \
                              opnd_ae_sem_mul_x2_S1_d0,                        \
                              opnd_ae_sem_mul_x2_S1_d1)                        \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulssfd32x16_h1_l0(                                    \
        (ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0), (__opnd_ae_sem_mul_x2_S1_d0), \
        (__opnd_ae_sem_mul_x2_S1_d1));                                         \
  })

#define AE_MULSSFD32X16_H1_L0_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)     \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulssfd32x16_h1_l0_s2(                                 \
        (ae_int64 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));      \
  })

#define AE_MULSSFD32X16_H3_L2(opnd_ae_sem_mul_x2_S1_q0,                        \
                              opnd_ae_sem_mul_x2_S1_d0,                        \
                              opnd_ae_sem_mul_x2_S1_d1)                        \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulssfd32x16_h3_l2(                                    \
        (ae_int64 *)&(opnd_ae_sem_mul_x2_S1_q0), (__opnd_ae_sem_mul_x2_S1_d0), \
        (__opnd_ae_sem_mul_x2_S1_d1));                                         \
  })

#define AE_MULSSFD32X16_H3_L2_S2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1)     \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulssfd32x16_h3_l2_s2(                                 \
        (ae_int64 *)&(ae_mul_S2_q0), (__ae_mul_S2_d0), (__ae_mul_S2_d1));      \
  })

#define AE_MULZAAD24_HH_LL(ae_mul_d0, ae_mul_d1)                               \
  ({                                                                           \
    ae_int32x2 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int32x2 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulzaad24_hh_ll((__ae_mul_d0), (__ae_mul_d1));         \
  })

#define AE_MULZAAD24_HH_LL_S2(ae_mul_S2_d0, ae_mul_S2_d1)                      \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulzaad24_hh_ll_s2((__ae_mul_S2_d0),                   \
                                           (__ae_mul_S2_d1));                  \
  })

#define AE_MULZAAD24_HL_LH(ae_mul_d0, ae_mul_d1)                               \
  ({                                                                           \
    ae_int32x2 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int32x2 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulzaad24_hl_lh((__ae_mul_d0), (__ae_mul_d1));         \
  })

#define AE_MULZAAD24_HL_LH_S2(ae_mul_S2_d0, ae_mul_S2_d1)                      \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulzaad24_hl_lh_s2((__ae_mul_S2_d0),                   \
                                           (__ae_mul_S2_d1));                  \
  })

#define AE_MULZAAD32X16_H0_L1(opnd_ae_sem_mul_x2_S1_d0,                        \
                              opnd_ae_sem_mul_x2_S1_d1)                        \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulzaad32x16_h0_l1((__opnd_ae_sem_mul_x2_S1_d0),       \
                                           (__opnd_ae_sem_mul_x2_S1_d1));      \
  })

#define AE_MULZAAD32X16_H0_L1_S2(ae_mul_S2_d0, ae_mul_S2_d1)                   \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulzaad32x16_h0_l1_s2((__ae_mul_S2_d0),                \
                                              (__ae_mul_S2_d1));               \
  })

#define AE_MULZAAD32X16_H1_L0(opnd_ae_sem_mul_x2_S1_d0,                        \
                              opnd_ae_sem_mul_x2_S1_d1)                        \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulzaad32x16_h1_l0((__opnd_ae_sem_mul_x2_S1_d0),       \
                                           (__opnd_ae_sem_mul_x2_S1_d1));      \
  })

#define AE_MULZAAD32X16_H1_L0_S2(ae_mul_S2_d0, ae_mul_S2_d1)                   \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulzaad32x16_h1_l0_s2((__ae_mul_S2_d0),                \
                                              (__ae_mul_S2_d1));               \
  })

#define AE_MULZAAD32X16_H2_L3(opnd_ae_sem_mul_x2_S1_d0,                        \
                              opnd_ae_sem_mul_x2_S1_d1)                        \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulzaad32x16_h2_l3((__opnd_ae_sem_mul_x2_S1_d0),       \
                                           (__opnd_ae_sem_mul_x2_S1_d1));      \
  })

#define AE_MULZAAD32X16_H2_L3_S2(ae_mul_S2_d0, ae_mul_S2_d1)                   \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulzaad32x16_h2_l3_s2((__ae_mul_S2_d0),                \
                                              (__ae_mul_S2_d1));               \
  })

#define AE_MULZAAD32X16_H3_L2(opnd_ae_sem_mul_x2_S1_d0,                        \
                              opnd_ae_sem_mul_x2_S1_d1)                        \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulzaad32x16_h3_l2((__opnd_ae_sem_mul_x2_S1_d0),       \
                                           (__opnd_ae_sem_mul_x2_S1_d1));      \
  })

#define AE_MULZAAD32X16_H3_L2_S2(ae_mul_S2_d0, ae_mul_S2_d1)                   \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulzaad32x16_h3_l2_s2((__ae_mul_S2_d0),                \
                                              (__ae_mul_S2_d1));               \
  })

#define AE_MULZAAFD16SS_11_00(opnd_ae_sem_mul_x2_S1_d0,                        \
                              opnd_ae_sem_mul_x2_S1_d1)                        \
  ({                                                                           \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulzaafd16ss_11_00((__opnd_ae_sem_mul_x2_S1_d0),       \
                                           (__opnd_ae_sem_mul_x2_S1_d1));      \
  })

#define AE_MULZAAFD16SS_11_00_S2(ae_mul_S2_d0, ae_mul_S2_d1)                   \
  ({                                                                           \
    ae_int16x4 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulzaafd16ss_11_00_s2((__ae_mul_S2_d0),                \
                                              (__ae_mul_S2_d1));               \
  })

#define AE_MULZAAFD16SS_13_02(opnd_ae_sem_mul_x2_S1_d0,                        \
                              opnd_ae_sem_mul_x2_S1_d1)                        \
  ({                                                                           \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulzaafd16ss_13_02((__opnd_ae_sem_mul_x2_S1_d0),       \
                                           (__opnd_ae_sem_mul_x2_S1_d1));      \
  })

#define AE_MULZAAFD16SS_13_02_S2(ae_mul_S2_d0, ae_mul_S2_d1)                   \
  ({                                                                           \
    ae_int16x4 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulzaafd16ss_13_02_s2((__ae_mul_S2_d0),                \
                                              (__ae_mul_S2_d1));               \
  })

#define AE_MULZAAFD16SS_33_22(opnd_ae_sem_mul_x2_S1_d0,                        \
                              opnd_ae_sem_mul_x2_S1_d1)                        \
  ({                                                                           \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulzaafd16ss_33_22((__opnd_ae_sem_mul_x2_S1_d0),       \
                                           (__opnd_ae_sem_mul_x2_S1_d1));      \
  })

#define AE_MULZAAFD16SS_33_22_S2(ae_mul_S2_d0, ae_mul_S2_d1)                   \
  ({                                                                           \
    ae_int16x4 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulzaafd16ss_33_22_s2((__ae_mul_S2_d0),                \
                                              (__ae_mul_S2_d1));               \
  })

#define AE_MULZAAFD24_HH_LL(ae_mul_d0, ae_mul_d1)                              \
  ({                                                                           \
    ae_int32x2 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int32x2 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulzaafd24_hh_ll((__ae_mul_d0), (__ae_mul_d1));        \
  })

#define AE_MULZAAFD24_HH_LL_S2(ae_mul_S2_d0, ae_mul_S2_d1)                     \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulzaafd24_hh_ll_s2((__ae_mul_S2_d0),                  \
                                            (__ae_mul_S2_d1));                 \
  })

#define AE_MULZAAFD24_HL_LH(ae_mul_d0, ae_mul_d1)                              \
  ({                                                                           \
    ae_int32x2 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int32x2 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulzaafd24_hl_lh((__ae_mul_d0), (__ae_mul_d1));        \
  })

#define AE_MULZAAFD24_HL_LH_S2(ae_mul_S2_d0, ae_mul_S2_d1)                     \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulzaafd24_hl_lh_s2((__ae_mul_S2_d0),                  \
                                            (__ae_mul_S2_d1));                 \
  })

#define AE_MULZAAFD32X16_H0_L1(opnd_ae_sem_mul_x2_S1_d0,                       \
                               opnd_ae_sem_mul_x2_S1_d1)                       \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulzaafd32x16_h0_l1((__opnd_ae_sem_mul_x2_S1_d0),      \
                                            (__opnd_ae_sem_mul_x2_S1_d1));     \
  })

#define AE_MULZAAFD32X16_H0_L1_S2(ae_mul_S2_d0, ae_mul_S2_d1)                  \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulzaafd32x16_h0_l1_s2((__ae_mul_S2_d0),               \
                                               (__ae_mul_S2_d1));              \
  })

#define AE_MULZAAFD32X16_H1_L0(opnd_ae_sem_mul_x2_S1_d0,                       \
                               opnd_ae_sem_mul_x2_S1_d1)                       \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulzaafd32x16_h1_l0((__opnd_ae_sem_mul_x2_S1_d0),      \
                                            (__opnd_ae_sem_mul_x2_S1_d1));     \
  })

#define AE_MULZAAFD32X16_H1_L0_S2(ae_mul_S2_d0, ae_mul_S2_d1)                  \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulzaafd32x16_h1_l0_s2((__ae_mul_S2_d0),               \
                                               (__ae_mul_S2_d1));              \
  })

#define AE_MULZAAFD32X16_H2_L3(opnd_ae_sem_mul_x2_S1_d0,                       \
                               opnd_ae_sem_mul_x2_S1_d1)                       \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulzaafd32x16_h2_l3((__opnd_ae_sem_mul_x2_S1_d0),      \
                                            (__opnd_ae_sem_mul_x2_S1_d1));     \
  })

#define AE_MULZAAFD32X16_H2_L3_S2(ae_mul_S2_d0, ae_mul_S2_d1)                  \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulzaafd32x16_h2_l3_s2((__ae_mul_S2_d0),               \
                                               (__ae_mul_S2_d1));              \
  })

#define AE_MULZAAFD32X16_H3_L2(opnd_ae_sem_mul_x2_S1_d0,                       \
                               opnd_ae_sem_mul_x2_S1_d1)                       \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulzaafd32x16_h3_l2((__opnd_ae_sem_mul_x2_S1_d0),      \
                                            (__opnd_ae_sem_mul_x2_S1_d1));     \
  })

#define AE_MULZAAFD32X16_H3_L2_S2(ae_mul_S2_d0, ae_mul_S2_d1)                  \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulzaafd32x16_h3_l2_s2((__ae_mul_S2_d0),               \
                                               (__ae_mul_S2_d1));              \
  })

#define AE_MULZASD24_HH_LL(ae_mul_d0, ae_mul_d1)                               \
  ({                                                                           \
    ae_int32x2 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int32x2 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulzasd24_hh_ll((__ae_mul_d0), (__ae_mul_d1));         \
  })

#define AE_MULZASD24_HH_LL_S2(ae_mul_S2_d0, ae_mul_S2_d1)                      \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulzasd24_hh_ll_s2((__ae_mul_S2_d0),                   \
                                           (__ae_mul_S2_d1));                  \
  })

#define AE_MULZASD24_HL_LH(ae_mul_d0, ae_mul_d1)                               \
  ({                                                                           \
    ae_int32x2 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int32x2 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulzasd24_hl_lh((__ae_mul_d0), (__ae_mul_d1));         \
  })

#define AE_MULZASD24_HL_LH_S2(ae_mul_S2_d0, ae_mul_S2_d1)                      \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulzasd24_hl_lh_s2((__ae_mul_S2_d0),                   \
                                           (__ae_mul_S2_d1));                  \
  })

#define AE_MULZASD32X16_H1_L0(opnd_ae_sem_mul_x2_S1_d0,                        \
                              opnd_ae_sem_mul_x2_S1_d1)                        \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulzasd32x16_h1_l0((__opnd_ae_sem_mul_x2_S1_d0),       \
                                           (__opnd_ae_sem_mul_x2_S1_d1));      \
  })

#define AE_MULZASD32X16_H1_L0_S2(ae_mul_S2_d0, ae_mul_S2_d1)                   \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulzasd32x16_h1_l0_s2((__ae_mul_S2_d0),                \
                                              (__ae_mul_S2_d1));               \
  })

#define AE_MULZASD32X16_H3_L2(opnd_ae_sem_mul_x2_S1_d0,                        \
                              opnd_ae_sem_mul_x2_S1_d1)                        \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulzasd32x16_h3_l2((__opnd_ae_sem_mul_x2_S1_d0),       \
                                           (__opnd_ae_sem_mul_x2_S1_d1));      \
  })

#define AE_MULZASD32X16_H3_L2_S2(ae_mul_S2_d0, ae_mul_S2_d1)                   \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulzasd32x16_h3_l2_s2((__ae_mul_S2_d0),                \
                                              (__ae_mul_S2_d1));               \
  })

#define AE_MULZASFD24_HH_LL(ae_mul_d0, ae_mul_d1)                              \
  ({                                                                           \
    ae_int32x2 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int32x2 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulzasfd24_hh_ll((__ae_mul_d0), (__ae_mul_d1));        \
  })

#define AE_MULZASFD24_HH_LL_S2(ae_mul_S2_d0, ae_mul_S2_d1)                     \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulzasfd24_hh_ll_s2((__ae_mul_S2_d0),                  \
                                            (__ae_mul_S2_d1));                 \
  })

#define AE_MULZASFD24_HL_LH(ae_mul_d0, ae_mul_d1)                              \
  ({                                                                           \
    ae_int32x2 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int32x2 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulzasfd24_hl_lh((__ae_mul_d0), (__ae_mul_d1));        \
  })

#define AE_MULZASFD24_HL_LH_S2(ae_mul_S2_d0, ae_mul_S2_d1)                     \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulzasfd24_hl_lh_s2((__ae_mul_S2_d0),                  \
                                            (__ae_mul_S2_d1));                 \
  })

#define AE_MULZASFD32X16_H1_L0(opnd_ae_sem_mul_x2_S1_d0,                       \
                               opnd_ae_sem_mul_x2_S1_d1)                       \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulzasfd32x16_h1_l0((__opnd_ae_sem_mul_x2_S1_d0),      \
                                            (__opnd_ae_sem_mul_x2_S1_d1));     \
  })

#define AE_MULZASFD32X16_H1_L0_S2(ae_mul_S2_d0, ae_mul_S2_d1)                  \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulzasfd32x16_h1_l0_s2((__ae_mul_S2_d0),               \
                                               (__ae_mul_S2_d1));              \
  })

#define AE_MULZASFD32X16_H3_L2(opnd_ae_sem_mul_x2_S1_d0,                       \
                               opnd_ae_sem_mul_x2_S1_d1)                       \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulzasfd32x16_h3_l2((__opnd_ae_sem_mul_x2_S1_d0),      \
                                            (__opnd_ae_sem_mul_x2_S1_d1));     \
  })

#define AE_MULZASFD32X16_H3_L2_S2(ae_mul_S2_d0, ae_mul_S2_d1)                  \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulzasfd32x16_h3_l2_s2((__ae_mul_S2_d0),               \
                                               (__ae_mul_S2_d1));              \
  })

#define AE_MULZSAD24_HH_LL(ae_mul_d0, ae_mul_d1)                               \
  ({                                                                           \
    ae_int32x2 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int32x2 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulzsad24_hh_ll((__ae_mul_d0), (__ae_mul_d1));         \
  })

#define AE_MULZSAD24_HH_LL_S2(ae_mul_S2_d0, ae_mul_S2_d1)                      \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulzsad24_hh_ll_s2((__ae_mul_S2_d0),                   \
                                           (__ae_mul_S2_d1));                  \
  })

#define AE_MULZSAD32X16_H1_L0(ae_mul_d0, ae_mul_d1)                            \
  ({                                                                           \
    ae_int32x2 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int16x4 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulzsad32x16_h1_l0((__ae_mul_d0), (__ae_mul_d1));      \
  })

#define AE_MULZSAD32X16_H1_L0_S2(ae_mul_S2_d0, ae_mul_S2_d1)                   \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulzsad32x16_h1_l0_s2((__ae_mul_S2_d0),                \
                                              (__ae_mul_S2_d1));               \
  })

#define AE_MULZSAD32X16_H3_L2(ae_mul_d0, ae_mul_d1)                            \
  ({                                                                           \
    ae_int32x2 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int16x4 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulzsad32x16_h3_l2((__ae_mul_d0), (__ae_mul_d1));      \
  })

#define AE_MULZSAD32X16_H3_L2_S2(ae_mul_S2_d0, ae_mul_S2_d1)                   \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulzsad32x16_h3_l2_s2((__ae_mul_S2_d0),                \
                                              (__ae_mul_S2_d1));               \
  })

#define AE_MULZSAFD24_HH_LL(ae_mul_d0, ae_mul_d1)                              \
  ({                                                                           \
    ae_int32x2 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int32x2 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulzsafd24_hh_ll((__ae_mul_d0), (__ae_mul_d1));        \
  })

#define AE_MULZSAFD24_HH_LL_S2(ae_mul_S2_d0, ae_mul_S2_d1)                     \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulzsafd24_hh_ll_s2((__ae_mul_S2_d0),                  \
                                            (__ae_mul_S2_d1));                 \
  })

#define AE_MULZSAFD32X16_H1_L0(opnd_ae_sem_mul_x2_S1_d0,                       \
                               opnd_ae_sem_mul_x2_S1_d1)                       \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulzsafd32x16_h1_l0((__opnd_ae_sem_mul_x2_S1_d0),      \
                                            (__opnd_ae_sem_mul_x2_S1_d1));     \
  })

#define AE_MULZSAFD32X16_H1_L0_S2(ae_mul_S2_d0, ae_mul_S2_d1)                  \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulzsafd32x16_h1_l0_s2((__ae_mul_S2_d0),               \
                                               (__ae_mul_S2_d1));              \
  })

#define AE_MULZSAFD32X16_H3_L2(opnd_ae_sem_mul_x2_S1_d0,                       \
                               opnd_ae_sem_mul_x2_S1_d1)                       \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulzsafd32x16_h3_l2((__opnd_ae_sem_mul_x2_S1_d0),      \
                                            (__opnd_ae_sem_mul_x2_S1_d1));     \
  })

#define AE_MULZSAFD32X16_H3_L2_S2(ae_mul_S2_d0, ae_mul_S2_d1)                  \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulzsafd32x16_h3_l2_s2((__ae_mul_S2_d0),               \
                                               (__ae_mul_S2_d1));              \
  })

#define AE_MULZSSD24_HH_LL(ae_mul_d0, ae_mul_d1)                               \
  ({                                                                           \
    ae_int32x2 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int32x2 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulzssd24_hh_ll((__ae_mul_d0), (__ae_mul_d1));         \
  })

#define AE_MULZSSD24_HH_LL_S2(ae_mul_S2_d0, ae_mul_S2_d1)                      \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulzssd24_hh_ll_s2((__ae_mul_S2_d0),                   \
                                           (__ae_mul_S2_d1));                  \
  })

#define AE_MULZSSD24_HL_LH(ae_mul_d0, ae_mul_d1)                               \
  ({                                                                           \
    ae_int32x2 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int32x2 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulzssd24_hl_lh((__ae_mul_d0), (__ae_mul_d1));         \
  })

#define AE_MULZSSD24_HL_LH_S2(ae_mul_S2_d0, ae_mul_S2_d1)                      \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulzssd24_hl_lh_s2((__ae_mul_S2_d0),                   \
                                           (__ae_mul_S2_d1));                  \
  })

#define AE_MULZSSD32X16_H1_L0(opnd_ae_sem_mul_x2_S1_d0,                        \
                              opnd_ae_sem_mul_x2_S1_d1)                        \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulzssd32x16_h1_l0((__opnd_ae_sem_mul_x2_S1_d0),       \
                                           (__opnd_ae_sem_mul_x2_S1_d1));      \
  })

#define AE_MULZSSD32X16_H1_L0_S2(ae_mul_S2_d0, ae_mul_S2_d1)                   \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulzssd32x16_h1_l0_s2((__ae_mul_S2_d0),                \
                                              (__ae_mul_S2_d1));               \
  })

#define AE_MULZSSD32X16_H3_L2(opnd_ae_sem_mul_x2_S1_d0,                        \
                              opnd_ae_sem_mul_x2_S1_d1)                        \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulzssd32x16_h3_l2((__opnd_ae_sem_mul_x2_S1_d0),       \
                                           (__opnd_ae_sem_mul_x2_S1_d1));      \
  })

#define AE_MULZSSD32X16_H3_L2_S2(ae_mul_S2_d0, ae_mul_S2_d1)                   \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulzssd32x16_h3_l2_s2((__ae_mul_S2_d0),                \
                                              (__ae_mul_S2_d1));               \
  })

#define AE_MULZSSFD16SS_11_00(opnd_ae_sem_mul_x2_S1_d0,                        \
                              opnd_ae_sem_mul_x2_S1_d1)                        \
  ({                                                                           \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulzssfd16ss_11_00((__opnd_ae_sem_mul_x2_S1_d0),       \
                                           (__opnd_ae_sem_mul_x2_S1_d1));      \
  })

#define AE_MULZSSFD16SS_11_00_S2(ae_mul_S2_d0, ae_mul_S2_d1)                   \
  ({                                                                           \
    ae_int16x4 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulzssfd16ss_11_00_s2((__ae_mul_S2_d0),                \
                                              (__ae_mul_S2_d1));               \
  })

#define AE_MULZSSFD16SS_13_02(opnd_ae_sem_mul_x2_S1_d0,                        \
                              opnd_ae_sem_mul_x2_S1_d1)                        \
  ({                                                                           \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulzssfd16ss_13_02((__opnd_ae_sem_mul_x2_S1_d0),       \
                                           (__opnd_ae_sem_mul_x2_S1_d1));      \
  })

#define AE_MULZSSFD16SS_13_02_S2(ae_mul_S2_d0, ae_mul_S2_d1)                   \
  ({                                                                           \
    ae_int16x4 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulzssfd16ss_13_02_s2((__ae_mul_S2_d0),                \
                                              (__ae_mul_S2_d1));               \
  })

#define AE_MULZSSFD16SS_33_22(opnd_ae_sem_mul_x2_S1_d0,                        \
                              opnd_ae_sem_mul_x2_S1_d1)                        \
  ({                                                                           \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulzssfd16ss_33_22((__opnd_ae_sem_mul_x2_S1_d0),       \
                                           (__opnd_ae_sem_mul_x2_S1_d1));      \
  })

#define AE_MULZSSFD16SS_33_22_S2(ae_mul_S2_d0, ae_mul_S2_d1)                   \
  ({                                                                           \
    ae_int16x4 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulzssfd16ss_33_22_s2((__ae_mul_S2_d0),                \
                                              (__ae_mul_S2_d1));               \
  })

#define AE_MULZSSFD24_HH_LL(ae_mul_d0, ae_mul_d1)                              \
  ({                                                                           \
    ae_int32x2 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int32x2 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulzssfd24_hh_ll((__ae_mul_d0), (__ae_mul_d1));        \
  })

#define AE_MULZSSFD24_HH_LL_S2(ae_mul_S2_d0, ae_mul_S2_d1)                     \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulzssfd24_hh_ll_s2((__ae_mul_S2_d0),                  \
                                            (__ae_mul_S2_d1));                 \
  })

#define AE_MULZSSFD24_HL_LH(ae_mul_d0, ae_mul_d1)                              \
  ({                                                                           \
    ae_int32x2 __ae_mul_d0 = (ae_mul_d0);                                      \
    ae_int32x2 __ae_mul_d1 = (ae_mul_d1);                                      \
    __builtin_xtensa_ae_mulzssfd24_hl_lh((__ae_mul_d0), (__ae_mul_d1));        \
  })

#define AE_MULZSSFD24_HL_LH_S2(ae_mul_S2_d0, ae_mul_S2_d1)                     \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int32x2 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulzssfd24_hl_lh_s2((__ae_mul_S2_d0),                  \
                                            (__ae_mul_S2_d1));                 \
  })

#define AE_MULZSSFD32X16_H1_L0(opnd_ae_sem_mul_x2_S1_d0,                       \
                               opnd_ae_sem_mul_x2_S1_d1)                       \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulzssfd32x16_h1_l0((__opnd_ae_sem_mul_x2_S1_d0),      \
                                            (__opnd_ae_sem_mul_x2_S1_d1));     \
  })

#define AE_MULZSSFD32X16_H1_L0_S2(ae_mul_S2_d0, ae_mul_S2_d1)                  \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulzssfd32x16_h1_l0_s2((__ae_mul_S2_d0),               \
                                               (__ae_mul_S2_d1));              \
  })

#define AE_MULZSSFD32X16_H3_L2(opnd_ae_sem_mul_x2_S1_d0,                       \
                               opnd_ae_sem_mul_x2_S1_d1)                       \
  ({                                                                           \
    ae_int32x2 __opnd_ae_sem_mul_x2_S1_d0 = (opnd_ae_sem_mul_x2_S1_d0);        \
    ae_int16x4 __opnd_ae_sem_mul_x2_S1_d1 = (opnd_ae_sem_mul_x2_S1_d1);        \
    __builtin_xtensa_ae_mulzssfd32x16_h3_l2((__opnd_ae_sem_mul_x2_S1_d0),      \
                                            (__opnd_ae_sem_mul_x2_S1_d1));     \
  })

#define AE_MULZSSFD32X16_H3_L2_S2(ae_mul_S2_d0, ae_mul_S2_d1)                  \
  ({                                                                           \
    ae_int32x2 __ae_mul_S2_d0 = (ae_mul_S2_d0);                                \
    ae_int16x4 __ae_mul_S2_d1 = (ae_mul_S2_d1);                                \
    __builtin_xtensa_ae_mulzssfd32x16_h3_l2_s2((__ae_mul_S2_d0),               \
                                               (__ae_mul_S2_d1));              \
  })

#define AE_NAND(ae_dr_to_dr_v0, ae_dr_to_dr_v1)                                \
  ({                                                                           \
    ae_int64 __ae_dr_to_dr_v0 = (ae_dr_to_dr_v0);                              \
    ae_int64 __ae_dr_to_dr_v1 = (ae_dr_to_dr_v1);                              \
    __builtin_xtensa_ae_nand((__ae_dr_to_dr_v0), (__ae_dr_to_dr_v1));          \
  })

#define AE_NEG16S(ae_arth_v1)                                                  \
  ({                                                                           \
    ae_int16x4 __ae_arth_v1 = (ae_arth_v1);                                    \
    __builtin_xtensa_ae_neg16s((__ae_arth_v1));                                \
  })

#define AE_NEG24S(ae_arth_v1)                                                  \
  ({                                                                           \
    ae_int32x2 __ae_arth_v1 = (ae_arth_v1);                                    \
    __builtin_xtensa_ae_neg24s((__ae_arth_v1));                                \
  })

#define AE_NEG32(ae_arth_v1)                                                   \
  ({                                                                           \
    ae_int32x2 __ae_arth_v1 = (ae_arth_v1);                                    \
    __builtin_xtensa_ae_neg32((__ae_arth_v1));                                 \
  })

#define AE_NEG32S(ae_arth_v1)                                                  \
  ({                                                                           \
    ae_int32x2 __ae_arth_v1 = (ae_arth_v1);                                    \
    __builtin_xtensa_ae_neg32s((__ae_arth_v1));                                \
  })

#define AE_NEG64(ae_arth_v1)                                                   \
  ({                                                                           \
    ae_int64 __ae_arth_v1 = (ae_arth_v1);                                      \
    __builtin_xtensa_ae_neg64((__ae_arth_v1));                                 \
  })

#define AE_NEG64S(ae_arth_v1)                                                  \
  ({                                                                           \
    ae_int64 __ae_arth_v1 = (ae_arth_v1);                                      \
    __builtin_xtensa_ae_neg64s((__ae_arth_v1));                                \
  })

#define AE_NSA64(ae_dr_to_ar_v0)                                               \
  ({                                                                           \
    ae_int64 __ae_dr_to_ar_v0 = (ae_dr_to_ar_v0);                              \
    __builtin_xtensa_ae_nsa64((__ae_dr_to_ar_v0));                             \
  })

#define AE_NSAZ16_0(ae_dr_to_ar_v0)                                            \
  ({                                                                           \
    ae_int16x4 __ae_dr_to_ar_v0 = (ae_dr_to_ar_v0);                            \
    __builtin_xtensa_ae_nsaz16_0((__ae_dr_to_ar_v0));                          \
  })

#define AE_NSAZ32_L(ae_dr_to_ar_v0)                                            \
  ({                                                                           \
    ae_int32x2 __ae_dr_to_ar_v0 = (ae_dr_to_ar_v0);                            \
    __builtin_xtensa_ae_nsaz32_l((__ae_dr_to_ar_v0));                          \
  })

#define AE_OR(ae_dr_to_dr_v0, ae_dr_to_dr_v1)                                  \
  ({                                                                           \
    ae_int64 __ae_dr_to_dr_v0 = (ae_dr_to_dr_v0);                              \
    ae_int64 __ae_dr_to_dr_v1 = (ae_dr_to_dr_v1);                              \
    __builtin_xtensa_ae_or((__ae_dr_to_dr_v0), (__ae_dr_to_dr_v1));            \
  })

#define AE_PKSR24(ae_pks_d, ae_pks_s, ae_imm2)                                 \
  ({                                                                           \
    ae_int64 __ae_pks_s = (ae_pks_s);                                          \
    __builtin_xtensa_ae_pksr24((ae_int32x2 *)&(ae_pks_d), (__ae_pks_s),        \
                               (ae_imm2));                                     \
  })

#define AE_PKSR32(ae_pks_d, ae_pks_s, ae_imm2)                                 \
  ({                                                                           \
    ae_int64 __ae_pks_s = (ae_pks_s);                                          \
    __builtin_xtensa_ae_pksr32((ae_int32x2 *)&(ae_pks_d), (__ae_pks_s),        \
                               (ae_imm2));                                     \
  })

#define AE_ROUND16X4F32SASYM(ae_arth_v1, ae_arth_v0)                           \
  ({                                                                           \
    ae_int32x2 __ae_arth_v1 = (ae_arth_v1);                                    \
    ae_int32x2 __ae_arth_v0 = (ae_arth_v0);                                    \
    __builtin_xtensa_ae_round16x4f32sasym((__ae_arth_v1), (__ae_arth_v0));     \
  })

#define AE_ROUND16X4F32SSYM(ae_arth_v1, ae_arth_v0)                            \
  ({                                                                           \
    ae_int32x2 __ae_arth_v1 = (ae_arth_v1);                                    \
    ae_int32x2 __ae_arth_v0 = (ae_arth_v0);                                    \
    __builtin_xtensa_ae_round16x4f32ssym((__ae_arth_v1), (__ae_arth_v0));      \
  })

#define AE_ROUND24X2F48SASYM(ae_arth_v0, ae_arth_v1)                           \
  ({                                                                           \
    ae_int64 __ae_arth_v0 = (ae_arth_v0);                                      \
    ae_int64 __ae_arth_v1 = (ae_arth_v1);                                      \
    __builtin_xtensa_ae_round24x2f48sasym((__ae_arth_v0), (__ae_arth_v1));     \
  })

#define AE_ROUND24X2F48SSYM(ae_arth_v0, ae_arth_v1)                            \
  ({                                                                           \
    ae_int64 __ae_arth_v0 = (ae_arth_v0);                                      \
    ae_int64 __ae_arth_v1 = (ae_arth_v1);                                      \
    __builtin_xtensa_ae_round24x2f48ssym((__ae_arth_v0), (__ae_arth_v1));      \
  })

#define AE_ROUND32X2F48SASYM(ae_arth_v0, ae_arth_v1)                           \
  ({                                                                           \
    ae_int64 __ae_arth_v0 = (ae_arth_v0);                                      \
    ae_int64 __ae_arth_v1 = (ae_arth_v1);                                      \
    __builtin_xtensa_ae_round32x2f48sasym((__ae_arth_v0), (__ae_arth_v1));     \
  })

#define AE_ROUND32X2F48SSYM(ae_arth_v0, ae_arth_v1)                            \
  ({                                                                           \
    ae_int64 __ae_arth_v0 = (ae_arth_v0);                                      \
    ae_int64 __ae_arth_v1 = (ae_arth_v1);                                      \
    __builtin_xtensa_ae_round32x2f48ssym((__ae_arth_v0), (__ae_arth_v1));      \
  })

#define AE_ROUND32X2F64SASYM(ae_arth_v0, ae_arth_v1)                           \
  ({                                                                           \
    ae_int64 __ae_arth_v0 = (ae_arth_v0);                                      \
    ae_int64 __ae_arth_v1 = (ae_arth_v1);                                      \
    __builtin_xtensa_ae_round32x2f64sasym((__ae_arth_v0), (__ae_arth_v1));     \
  })

#define AE_ROUND32X2F64SSYM(ae_arth_v0, ae_arth_v1)                            \
  ({                                                                           \
    ae_int64 __ae_arth_v0 = (ae_arth_v0);                                      \
    ae_int64 __ae_arth_v1 = (ae_arth_v1);                                      \
    __builtin_xtensa_ae_round32x2f64ssym((__ae_arth_v0), (__ae_arth_v1));      \
  })

#define AE_ROUNDSP16F24ASYM(ae_arth_v0)                                        \
  ({                                                                           \
    ae_int32x2 __ae_arth_v0 = (ae_arth_v0);                                    \
    __builtin_xtensa_ae_roundsp16f24asym((__ae_arth_v0));                      \
  })

#define AE_ROUNDSP16F24SYM(ae_arth_v0)                                         \
  ({                                                                           \
    ae_int32x2 __ae_arth_v0 = (ae_arth_v0);                                    \
    __builtin_xtensa_ae_roundsp16f24sym((__ae_arth_v0));                       \
  })

#define AE_ROUNDSP16Q48X2ASYM(ae_arth_v0, ae_arth_v1)                          \
  ({                                                                           \
    ae_int64 __ae_arth_v0 = (ae_arth_v0);                                      \
    ae_int64 __ae_arth_v1 = (ae_arth_v1);                                      \
    __builtin_xtensa_ae_roundsp16q48x2asym((__ae_arth_v0), (__ae_arth_v1));    \
  })

#define AE_ROUNDSP16Q48X2SYM(ae_arth_v0, ae_arth_v1)                           \
  ({                                                                           \
    ae_int64 __ae_arth_v0 = (ae_arth_v0);                                      \
    ae_int64 __ae_arth_v1 = (ae_arth_v1);                                      \
    __builtin_xtensa_ae_roundsp16q48x2sym((__ae_arth_v0), (__ae_arth_v1));     \
  })

#define AE_ROUNDSQ32F48ASYM(ae_arth_v1)                                        \
  ({                                                                           \
    ae_int64 __ae_arth_v1 = (ae_arth_v1);                                      \
    __builtin_xtensa_ae_roundsq32f48asym((__ae_arth_v1));                      \
  })

#define AE_ROUNDSQ32F48SYM(ae_arth_v1)                                         \
  ({                                                                           \
    ae_int64 __ae_arth_v1 = (ae_arth_v1);                                      \
    __builtin_xtensa_ae_roundsq32f48sym((__ae_arth_v1));                       \
  })

#define AE_S16_0_I(ae_ls_v, ars, ae_immls16)                                   \
  ({                                                                           \
    ae_int16x4 __ae_ls_v = (ae_ls_v);                                          \
    ae_int16 *__ars = (ars);                                                   \
    __builtin_xtensa_ae_s16_0_i((__ae_ls_v), (__ars), (ae_immls16));           \
  })

#define AE_S16_0_IP(ae_ls_v, ars, ae_immls16)                                  \
  ({                                                                           \
    ae_int16x4 __ae_ls_v = (ae_ls_v);                                          \
    __builtin_xtensa_ae_s16_0_ip((__ae_ls_v), (ae_int16 **)&(ars),             \
                                 (ae_immls16));                                \
  })

#define AE_S16_0_X(ae_ls_v, ars, art)                                          \
  ({                                                                           \
    ae_int16x4 __ae_ls_v = (ae_ls_v);                                          \
    ae_int16 *__ars = (ars);                                                   \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_s16_0_x((__ae_ls_v), (__ars), (__art));                \
  })

#define AE_S16_0_XC(ae_ls_v, ars, art)                                         \
  ({                                                                           \
    ae_int16x4 __ae_ls_v = (ae_ls_v);                                          \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_s16_0_xc((__ae_ls_v), (ae_int16 **)&(ars), (__art));   \
  })

#define AE_S16_0_XP(ae_ls_v, ars, art)                                         \
  ({                                                                           \
    ae_int16x4 __ae_ls_v = (ae_ls_v);                                          \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_s16_0_xp((__ae_ls_v), (ae_int16 **)&(ars), (__art));   \
  })

#define AE_S16M_L_I(ae_ls_v, ars, ae_immls16)                                  \
  ({                                                                           \
    ae_int32x2 __ae_ls_v = (ae_ls_v);                                          \
    ae_int16 *__ars = (ars);                                                   \
    __builtin_xtensa_ae_s16m_l_i((__ae_ls_v), (__ars), (ae_immls16));          \
  })

#define AE_S16M_L_IU(ae_ls_v, ars, ae_immls16)                                 \
  ({                                                                           \
    ae_int32x2 __ae_ls_v = (ae_ls_v);                                          \
    __builtin_xtensa_ae_s16m_l_iu((__ae_ls_v), (ae_int16 **)&(ars),            \
                                  (ae_immls16));                               \
  })

#define AE_S16M_L_X(ae_ls_v, ars, art)                                         \
  ({                                                                           \
    ae_int32x2 __ae_ls_v = (ae_ls_v);                                          \
    ae_int16 *__ars = (ars);                                                   \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_s16m_l_x((__ae_ls_v), (__ars), (__art));               \
  })

#define AE_S16M_L_XC(ae_ls_v, ars, art)                                        \
  ({                                                                           \
    ae_int32x2 __ae_ls_v = (ae_ls_v);                                          \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_s16m_l_xc((__ae_ls_v), (ae_int16 **)&(ars), (__art));  \
  })

#define AE_S16M_L_XU(ae_ls_v, ars, art)                                        \
  ({                                                                           \
    ae_int32x2 __ae_ls_v = (ae_ls_v);                                          \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_s16m_l_xu((__ae_ls_v), (ae_int16 **)&(ars), (__art));  \
  })

#define AE_S16X2M_I(ae_ls_v, ars, ae_immls32)                                  \
  ({                                                                           \
    ae_int32x2 __ae_ls_v = (ae_ls_v);                                          \
    ae_int16x2 *__ars = (ars);                                                 \
    __builtin_xtensa_ae_s16x2m_i((__ae_ls_v), (__ars), (ae_immls32));          \
  })

#define AE_S16X2M_IU(ae_ls_v, ars, ae_immls32)                                 \
  ({                                                                           \
    ae_int32x2 __ae_ls_v = (ae_ls_v);                                          \
    __builtin_xtensa_ae_s16x2m_iu((__ae_ls_v), (ae_int16x2 **)&(ars),          \
                                  (ae_immls32));                               \
  })

#define AE_S16X2M_X(ae_ls_v, ars, art)                                         \
  ({                                                                           \
    ae_int32x2 __ae_ls_v = (ae_ls_v);                                          \
    ae_int16x2 *__ars = (ars);                                                 \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_s16x2m_x((__ae_ls_v), (__ars), (__art));               \
  })

#define AE_S16X2M_XC(ae_ls_v, ars, art)                                        \
  ({                                                                           \
    ae_int32x2 __ae_ls_v = (ae_ls_v);                                          \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_s16x2m_xc((__ae_ls_v), (ae_int16x2 **)&(ars),          \
                                  (__art));                                    \
  })

#define AE_S16X2M_XU(ae_ls_v, ars, art)                                        \
  ({                                                                           \
    ae_int32x2 __ae_ls_v = (ae_ls_v);                                          \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_s16x2m_xu((__ae_ls_v), (ae_int16x2 **)&(ars),          \
                                  (__art));                                    \
  })

#define AE_S16X4_I(ae_ls_v, ars, ae_immls64)                                   \
  ({                                                                           \
    ae_int16x4 __ae_ls_v = (ae_ls_v);                                          \
    ae_int16x4 *__ars = (ars);                                                 \
    __builtin_xtensa_ae_s16x4_i((__ae_ls_v), (__ars), (ae_immls64));           \
  })

#define AE_S16X4_IP(ae_ls_v, ars, ae_immls64pos)                               \
  ({                                                                           \
    ae_int16x4 __ae_ls_v = (ae_ls_v);                                          \
    __builtin_xtensa_ae_s16x4_ip((__ae_ls_v), (ae_int16x4 **)&(ars),           \
                                 (ae_immls64pos));                             \
  })

#define AE_S16X4_RIC(ae_ls_v, ars)                                             \
  ({                                                                           \
    ae_int16x4 __ae_ls_v = (ae_ls_v);                                          \
    __builtin_xtensa_ae_s16x4_ric((__ae_ls_v), (ae_int16x4 **)&(ars));         \
  })

#define AE_S16X4_RIP(ae_ls_v, ars)                                             \
  ({                                                                           \
    ae_int16x4 __ae_ls_v = (ae_ls_v);                                          \
    __builtin_xtensa_ae_s16x4_rip((__ae_ls_v), (ae_int16x4 **)&(ars));         \
  })

#define AE_S16X4_X(ae_ls_v, ars, art)                                          \
  ({                                                                           \
    ae_int16x4 __ae_ls_v = (ae_ls_v);                                          \
    ae_int16x4 *__ars = (ars);                                                 \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_s16x4_x((__ae_ls_v), (__ars), (__art));                \
  })

#define AE_S16X4_XC(ae_ls_v, ars, art)                                         \
  ({                                                                           \
    ae_int16x4 __ae_ls_v = (ae_ls_v);                                          \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_s16x4_xc((__ae_ls_v), (ae_int16x4 **)&(ars), (__art)); \
  })

#define AE_S16X4_XP(ae_ls_v, ars, art)                                         \
  ({                                                                           \
    ae_int16x4 __ae_ls_v = (ae_ls_v);                                          \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_s16x4_xp((__ae_ls_v), (ae_int16x4 **)&(ars), (__art)); \
  })

#define AE_S24RA64S_I(ae_ls_v1, ars, ae_immls32)                               \
  ({                                                                           \
    ae_int64 __ae_ls_v1 = (ae_ls_v1);                                          \
    ae_int32 *__ars = (ars);                                                   \
    __builtin_xtensa_ae_s24ra64s_i((__ae_ls_v1), (__ars), (ae_immls32));       \
  })

#define AE_S24RA64S_IP(ae_ls_v1, ars, ae_immls32)                              \
  ({                                                                           \
    ae_int64 __ae_ls_v1 = (ae_ls_v1);                                          \
    __builtin_xtensa_ae_s24ra64s_ip((__ae_ls_v1), (ae_int32 **)&(ars),         \
                                    (ae_immls32));                             \
  })

#define AE_S24RA64S_X(ae_ls_v1, ars, art)                                      \
  ({                                                                           \
    ae_int64 __ae_ls_v1 = (ae_ls_v1);                                          \
    ae_int32 *__ars = (ars);                                                   \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_s24ra64s_x((__ae_ls_v1), (__ars), (__art));            \
  })

#define AE_S24RA64S_XC(ae_ls_v1, ars, art)                                     \
  ({                                                                           \
    ae_int64 __ae_ls_v1 = (ae_ls_v1);                                          \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_s24ra64s_xc((__ae_ls_v1), (ae_int32 **)&(ars),         \
                                    (__art));                                  \
  })

#define AE_S24RA64S_XP(ae_ls_v1, ars, art)                                     \
  ({                                                                           \
    ae_int64 __ae_ls_v1 = (ae_ls_v1);                                          \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_s24ra64s_xp((__ae_ls_v1), (ae_int32 **)&(ars),         \
                                    (__art));                                  \
  })

#define AE_S24X2RA64S_IP(ae_ls_v2, ae_ls_v1, ars)                              \
  ({                                                                           \
    ae_int64 __ae_ls_v2 = (ae_ls_v2);                                          \
    ae_int64 __ae_ls_v1 = (ae_ls_v1);                                          \
    __builtin_xtensa_ae_s24x2ra64s_ip((__ae_ls_v2), (__ae_ls_v1),              \
                                      (ae_int32x2 **)&(ars));                  \
  })

#define AE_S32_L_I(ae_ls_v, ars, ae_immls32)                                   \
  ({                                                                           \
    ae_int32x2 __ae_ls_v = (ae_ls_v);                                          \
    ae_int32 *__ars = (ars);                                                   \
    __builtin_xtensa_ae_s32_l_i((__ae_ls_v), (__ars), (ae_immls32));           \
  })

#define AE_S32_L_IP(ae_ls_v, ars, ae_immls32)                                  \
  ({                                                                           \
    ae_int32x2 __ae_ls_v = (ae_ls_v);                                          \
    __builtin_xtensa_ae_s32_l_ip((__ae_ls_v), (ae_int32 **)&(ars),             \
                                 (ae_immls32));                                \
  })

#define AE_S32_L_X(ae_ls_v, ars, art)                                          \
  ({                                                                           \
    ae_int32x2 __ae_ls_v = (ae_ls_v);                                          \
    ae_int32 *__ars = (ars);                                                   \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_s32_l_x((__ae_ls_v), (__ars), (__art));                \
  })

#define AE_S32_L_XC(ae_ls_v, ars, art)                                         \
  ({                                                                           \
    ae_int32x2 __ae_ls_v = (ae_ls_v);                                          \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_s32_l_xc((__ae_ls_v), (ae_int32 **)&(ars), (__art));   \
  })

#define AE_S32_L_XP(ae_ls_v, ars, art)                                         \
  ({                                                                           \
    ae_int32x2 __ae_ls_v = (ae_ls_v);                                          \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_s32_l_xp((__ae_ls_v), (ae_int32 **)&(ars), (__art));   \
  })

#define AE_S32F24_L_I(ae_ls_v, ars, ae_immls32)                                \
  ({                                                                           \
    ae_int32x2 __ae_ls_v = (ae_ls_v);                                          \
    ae_int32 *__ars = (ars);                                                   \
    __builtin_xtensa_ae_s32f24_l_i((__ae_ls_v), (__ars), (ae_immls32));        \
  })

#define AE_S32F24_L_IP(ae_ls_v, ars, ae_immls32)                               \
  ({                                                                           \
    ae_int32x2 __ae_ls_v = (ae_ls_v);                                          \
    __builtin_xtensa_ae_s32f24_l_ip((__ae_ls_v), (ae_int32 **)&(ars),          \
                                    (ae_immls32));                             \
  })

#define AE_S32F24_L_X(ae_ls_v, ars, art)                                       \
  ({                                                                           \
    ae_int32x2 __ae_ls_v = (ae_ls_v);                                          \
    ae_int32 *__ars = (ars);                                                   \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_s32f24_l_x((__ae_ls_v), (__ars), (__art));             \
  })

#define AE_S32F24_L_XC(ae_ls_v, ars, art)                                      \
  ({                                                                           \
    ae_int32x2 __ae_ls_v = (ae_ls_v);                                          \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_s32f24_l_xc((__ae_ls_v), (ae_int32 **)&(ars),          \
                                    (__art));                                  \
  })

#define AE_S32F24_L_XP(ae_ls_v, ars, art)                                      \
  ({                                                                           \
    ae_int32x2 __ae_ls_v = (ae_ls_v);                                          \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_s32f24_l_xp((__ae_ls_v), (ae_int32 **)&(ars),          \
                                    (__art));                                  \
  })

#define AE_S32M_I(ae_ls_v, ars, ae_immls32)                                    \
  ({                                                                           \
    ae_int64 __ae_ls_v = (ae_ls_v);                                            \
    ae_int32 *__ars = (ars);                                                   \
    __builtin_xtensa_ae_s32m_i((__ae_ls_v), (__ars), (ae_immls32));            \
  })

#define AE_S32M_IU(ae_ls_v, ars, ae_immls32)                                   \
  ({                                                                           \
    ae_int64 __ae_ls_v = (ae_ls_v);                                            \
    __builtin_xtensa_ae_s32m_iu((__ae_ls_v), (ae_int32 **)&(ars),              \
                                (ae_immls32));                                 \
  })

#define AE_S32M_X(ae_ls_v, ars, art)                                           \
  ({                                                                           \
    ae_int64 __ae_ls_v = (ae_ls_v);                                            \
    ae_int32 *__ars = (ars);                                                   \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_s32m_x((__ae_ls_v), (__ars), (__art));                 \
  })

#define AE_S32M_XC(ae_ls_v, ars, art)                                          \
  ({                                                                           \
    ae_int64 __ae_ls_v = (ae_ls_v);                                            \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_s32m_xc((__ae_ls_v), (ae_int32 **)&(ars), (__art));    \
  })

#define AE_S32M_XU(ae_ls_v, ars, art)                                          \
  ({                                                                           \
    ae_int64 __ae_ls_v = (ae_ls_v);                                            \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_s32m_xu((__ae_ls_v), (ae_int32 **)&(ars), (__art));    \
  })

#define AE_S32RA64S_I(ae_ls_v1, ars, ae_immls32)                               \
  ({                                                                           \
    ae_int64 __ae_ls_v1 = (ae_ls_v1);                                          \
    ae_int32 *__ars = (ars);                                                   \
    __builtin_xtensa_ae_s32ra64s_i((__ae_ls_v1), (__ars), (ae_immls32));       \
  })

#define AE_S32RA64S_IP(ae_ls_v1, ars, ae_immls32)                              \
  ({                                                                           \
    ae_int64 __ae_ls_v1 = (ae_ls_v1);                                          \
    __builtin_xtensa_ae_s32ra64s_ip((__ae_ls_v1), (ae_int32 **)&(ars),         \
                                    (ae_immls32));                             \
  })

#define AE_S32RA64S_X(ae_ls_v1, ars, art)                                      \
  ({                                                                           \
    ae_int64 __ae_ls_v1 = (ae_ls_v1);                                          \
    ae_int32 *__ars = (ars);                                                   \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_s32ra64s_x((__ae_ls_v1), (__ars), (__art));            \
  })

#define AE_S32RA64S_XC(ae_ls_v1, ars, art)                                     \
  ({                                                                           \
    ae_int64 __ae_ls_v1 = (ae_ls_v1);                                          \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_s32ra64s_xc((__ae_ls_v1), (ae_int32 **)&(ars),         \
                                    (__art));                                  \
  })

#define AE_S32RA64S_XP(ae_ls_v1, ars, art)                                     \
  ({                                                                           \
    ae_int64 __ae_ls_v1 = (ae_ls_v1);                                          \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_s32ra64s_xp((__ae_ls_v1), (ae_int32 **)&(ars),         \
                                    (__art));                                  \
  })

#define AE_S32X2_I(ae_ls_v, ars, ae_immls64)                                   \
  ({                                                                           \
    ae_int32x2 __ae_ls_v = (ae_ls_v);                                          \
    ae_int32x2 *__ars = (ars);                                                 \
    __builtin_xtensa_ae_s32x2_i((__ae_ls_v), (__ars), (ae_immls64));           \
  })

#define AE_S32X2_IP(ae_ls_v, ars, ae_immls64pos)                               \
  ({                                                                           \
    ae_int32x2 __ae_ls_v = (ae_ls_v);                                          \
    __builtin_xtensa_ae_s32x2_ip((__ae_ls_v), (ae_int32x2 **)&(ars),           \
                                 (ae_immls64pos));                             \
  })

#define AE_S32X2_RIC(ae_ls_v, ars)                                             \
  ({                                                                           \
    ae_int32x2 __ae_ls_v = (ae_ls_v);                                          \
    __builtin_xtensa_ae_s32x2_ric((__ae_ls_v), (ae_int32x2 **)&(ars));         \
  })

#define AE_S32X2_RIP(ae_ls_v, ars)                                             \
  ({                                                                           \
    ae_int32x2 __ae_ls_v = (ae_ls_v);                                          \
    __builtin_xtensa_ae_s32x2_rip((__ae_ls_v), (ae_int32x2 **)&(ars));         \
  })

#define AE_S32X2_X(ae_ls_v, ars, art)                                          \
  ({                                                                           \
    ae_int32x2 __ae_ls_v = (ae_ls_v);                                          \
    ae_int32x2 *__ars = (ars);                                                 \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_s32x2_x((__ae_ls_v), (__ars), (__art));                \
  })

#define AE_S32X2_XC(ae_ls_v, ars, art)                                         \
  ({                                                                           \
    ae_int32x2 __ae_ls_v = (ae_ls_v);                                          \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_s32x2_xc((__ae_ls_v), (ae_int32x2 **)&(ars), (__art)); \
  })

#define AE_S32X2_XP(ae_ls_v, ars, art)                                         \
  ({                                                                           \
    ae_int32x2 __ae_ls_v = (ae_ls_v);                                          \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_s32x2_xp((__ae_ls_v), (ae_int32x2 **)&(ars), (__art)); \
  })

#define AE_S32X2F24_I(ae_ls_v, ars, ae_immls64)                                \
  ({                                                                           \
    ae_int32x2 __ae_ls_v = (ae_ls_v);                                          \
    ae_int32x2 *__ars = (ars);                                                 \
    __builtin_xtensa_ae_s32x2f24_i((__ae_ls_v), (__ars), (ae_immls64));        \
  })

#define AE_S32X2F24_IP(ae_ls_v, ars, ae_immls64pos)                            \
  ({                                                                           \
    ae_int32x2 __ae_ls_v = (ae_ls_v);                                          \
    __builtin_xtensa_ae_s32x2f24_ip((__ae_ls_v), (ae_int32x2 **)&(ars),        \
                                    (ae_immls64pos));                          \
  })

#define AE_S32X2F24_RIC(ae_ls_v, ars)                                          \
  ({                                                                           \
    ae_int32x2 __ae_ls_v = (ae_ls_v);                                          \
    __builtin_xtensa_ae_s32x2f24_ric((__ae_ls_v), (ae_int32x2 **)&(ars));      \
  })

#define AE_S32X2F24_RIP(ae_ls_v, ars)                                          \
  ({                                                                           \
    ae_int32x2 __ae_ls_v = (ae_ls_v);                                          \
    __builtin_xtensa_ae_s32x2f24_rip((__ae_ls_v), (ae_int32x2 **)&(ars));      \
  })

#define AE_S32X2F24_X(ae_ls_v, ars, art)                                       \
  ({                                                                           \
    ae_int32x2 __ae_ls_v = (ae_ls_v);                                          \
    ae_int32x2 *__ars = (ars);                                                 \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_s32x2f24_x((__ae_ls_v), (__ars), (__art));             \
  })

#define AE_S32X2F24_XC(ae_ls_v, ars, art)                                      \
  ({                                                                           \
    ae_int32x2 __ae_ls_v = (ae_ls_v);                                          \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_s32x2f24_xc((__ae_ls_v), (ae_int32x2 **)&(ars),        \
                                    (__art));                                  \
  })

#define AE_S32X2F24_XP(ae_ls_v, ars, art)                                      \
  ({                                                                           \
    ae_int32x2 __ae_ls_v = (ae_ls_v);                                          \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_s32x2f24_xp((__ae_ls_v), (ae_int32x2 **)&(ars),        \
                                    (__art));                                  \
  })

#define AE_S32X2RA64S_IP(ae_ls_v2, ae_ls_v1, ars)                              \
  ({                                                                           \
    ae_int64 __ae_ls_v2 = (ae_ls_v2);                                          \
    ae_int64 __ae_ls_v1 = (ae_ls_v1);                                          \
    __builtin_xtensa_ae_s32x2ra64s_ip((__ae_ls_v2), (__ae_ls_v1),              \
                                      (ae_int32x2 **)&(ars));                  \
  })

#define AE_S64_I(ae_ls_v, ars, ae_immls64)                                     \
  ({                                                                           \
    ae_int64 __ae_ls_v = (ae_ls_v);                                            \
    ae_int64 *__ars = (ars);                                                   \
    __builtin_xtensa_ae_s64_i((__ae_ls_v), (__ars), (ae_immls64));             \
  })

#define AE_S64_IP(ae_ls_v, ars, ae_immls64)                                    \
  ({                                                                           \
    ae_int64 __ae_ls_v = (ae_ls_v);                                            \
    __builtin_xtensa_ae_s64_ip((__ae_ls_v), (ae_int64 **)&(ars),               \
                               (ae_immls64));                                  \
  })

#define AE_S64_X(ae_ls_v, ars, art)                                            \
  ({                                                                           \
    ae_int64 __ae_ls_v = (ae_ls_v);                                            \
    ae_int64 *__ars = (ars);                                                   \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_s64_x((__ae_ls_v), (__ars), (__art));                  \
  })

#define AE_S64_XC(ae_ls_v, ars, art)                                           \
  ({                                                                           \
    ae_int64 __ae_ls_v = (ae_ls_v);                                            \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_s64_xc((__ae_ls_v), (ae_int64 **)&(ars), (__art));     \
  })

#define AE_S64_XP(ae_ls_v, ars, art)                                           \
  ({                                                                           \
    ae_int64 __ae_ls_v = (ae_ls_v);                                            \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_s64_xp((__ae_ls_v), (ae_int64 **)&(ars), (__art));     \
  })

#define AE_SA16X4_IC(ae_ls_v, ae_ls_su, ars)                                   \
  ({                                                                           \
    ae_int16x4 __ae_ls_v = (ae_ls_v);                                          \
    __builtin_xtensa_ae_sa16x4_ic((__ae_ls_v), (ae_valign *)&(ae_ls_su),       \
                                  (ae_int16x4 **)&(ars));                      \
  })

#define AE_SA16X4_IP(ae_ls_v, ae_ls_su, ars)                                   \
  ({                                                                           \
    ae_int16x4 __ae_ls_v = (ae_ls_v);                                          \
    __builtin_xtensa_ae_sa16x4_ip((__ae_ls_v), (ae_valign *)&(ae_ls_su),       \
                                  (ae_int16x4 **)&(ars));                      \
  })

#define AE_SA16X4_RIC(ae_ls_v, ae_ls_su, ars)                                  \
  ({                                                                           \
    ae_int16x4 __ae_ls_v = (ae_ls_v);                                          \
    __builtin_xtensa_ae_sa16x4_ric((__ae_ls_v), (ae_valign *)&(ae_ls_su),      \
                                   (ae_int16x4 **)&(ars));                     \
  })

#define AE_SA16X4_RIP(ae_ls_v, ae_ls_su, ars)                                  \
  ({                                                                           \
    ae_int16x4 __ae_ls_v = (ae_ls_v);                                          \
    __builtin_xtensa_ae_sa16x4_rip((__ae_ls_v), (ae_valign *)&(ae_ls_su),      \
                                   (ae_int16x4 **)&(ars));                     \
  })

#define AE_SA24_L_IC(ae_ls_v, ae_ls_su, ars)                                   \
  ({                                                                           \
    ae_int32x2 __ae_ls_v = (ae_ls_v);                                          \
    __builtin_xtensa_ae_sa24_l_ic((__ae_ls_v), (ae_valign *)&(ae_ls_su),       \
                                  (void **)&(ars));                            \
  })

#define AE_SA24_L_IP(ae_ls_v, ae_ls_su, ars)                                   \
  ({                                                                           \
    ae_int32x2 __ae_ls_v = (ae_ls_v);                                          \
    __builtin_xtensa_ae_sa24_l_ip((__ae_ls_v), (ae_valign *)&(ae_ls_su),       \
                                  (void **)&(ars));                            \
  })

#define AE_SA24_L_RIC(ae_ls_v, ae_ls_su, ars)                                  \
  ({                                                                           \
    ae_int32x2 __ae_ls_v = (ae_ls_v);                                          \
    __builtin_xtensa_ae_sa24_l_ric((__ae_ls_v), (ae_valign *)&(ae_ls_su),      \
                                   (void **)&(ars));                           \
  })

#define AE_SA24_L_RIP(ae_ls_v, ae_ls_su, ars)                                  \
  ({                                                                           \
    ae_int32x2 __ae_ls_v = (ae_ls_v);                                          \
    __builtin_xtensa_ae_sa24_l_rip((__ae_ls_v), (ae_valign *)&(ae_ls_su),      \
                                   (void **)&(ars));                           \
  })

#define AE_SA24X2_IC(ae_ls_v, ae_ls_su, ars)                                   \
  ({                                                                           \
    ae_int32x2 __ae_ls_v = (ae_ls_v);                                          \
    __builtin_xtensa_ae_sa24x2_ic((__ae_ls_v), (ae_valign *)&(ae_ls_su),       \
                                  (void **)&(ars));                            \
  })

#define AE_SA24X2_IP(ae_ls_v, ae_ls_su, ars)                                   \
  ({                                                                           \
    ae_int32x2 __ae_ls_v = (ae_ls_v);                                          \
    __builtin_xtensa_ae_sa24x2_ip((__ae_ls_v), (ae_valign *)&(ae_ls_su),       \
                                  (void **)&(ars));                            \
  })

#define AE_SA24X2_RIC(ae_ls_v, ae_ls_su, ars)                                  \
  ({                                                                           \
    ae_int32x2 __ae_ls_v = (ae_ls_v);                                          \
    __builtin_xtensa_ae_sa24x2_ric((__ae_ls_v), (ae_valign *)&(ae_ls_su),      \
                                   (void **)&(ars));                           \
  })

#define AE_SA24X2_RIP(ae_ls_v, ae_ls_su, ars)                                  \
  ({                                                                           \
    ae_int32x2 __ae_ls_v = (ae_ls_v);                                          \
    __builtin_xtensa_ae_sa24x2_rip((__ae_ls_v), (ae_valign *)&(ae_ls_su),      \
                                   (void **)&(ars));                           \
  })

#define AE_SA32X2_IC(ae_ls_v, ae_ls_su, ars)                                   \
  ({                                                                           \
    ae_int32x2 __ae_ls_v = (ae_ls_v);                                          \
    __builtin_xtensa_ae_sa32x2_ic((__ae_ls_v), (ae_valign *)&(ae_ls_su),       \
                                  (ae_int32x2 **)&(ars));                      \
  })

#define AE_SA32X2_IP(ae_ls_v, ae_ls_su, ars)                                   \
  ({                                                                           \
    ae_int32x2 __ae_ls_v = (ae_ls_v);                                          \
    __builtin_xtensa_ae_sa32x2_ip((__ae_ls_v), (ae_valign *)&(ae_ls_su),       \
                                  (ae_int32x2 **)&(ars));                      \
  })

#define AE_SA32X2_RIC(ae_ls_v, ae_ls_su, ars)                                  \
  ({                                                                           \
    ae_int32x2 __ae_ls_v = (ae_ls_v);                                          \
    __builtin_xtensa_ae_sa32x2_ric((__ae_ls_v), (ae_valign *)&(ae_ls_su),      \
                                   (ae_int32x2 **)&(ars));                     \
  })

#define AE_SA32X2_RIP(ae_ls_v, ae_ls_su, ars)                                  \
  ({                                                                           \
    ae_int32x2 __ae_ls_v = (ae_ls_v);                                          \
    __builtin_xtensa_ae_sa32x2_rip((__ae_ls_v), (ae_valign *)&(ae_ls_su),      \
                                   (ae_int32x2 **)&(ars));                     \
  })

#define AE_SA32X2F24_IC(ae_ls_v, ae_ls_su, ars)                                \
  ({                                                                           \
    ae_int32x2 __ae_ls_v = (ae_ls_v);                                          \
    __builtin_xtensa_ae_sa32x2f24_ic((__ae_ls_v), (ae_valign *)&(ae_ls_su),    \
                                     (ae_int32x2 **)&(ars));                   \
  })

#define AE_SA32X2F24_IP(ae_ls_v, ae_ls_su, ars)                                \
  ({                                                                           \
    ae_int32x2 __ae_ls_v = (ae_ls_v);                                          \
    __builtin_xtensa_ae_sa32x2f24_ip((__ae_ls_v), (ae_valign *)&(ae_ls_su),    \
                                     (ae_int32x2 **)&(ars));                   \
  })

#define AE_SA32X2F24_RIC(ae_ls_v, ae_ls_su, ars)                               \
  ({                                                                           \
    ae_int32x2 __ae_ls_v = (ae_ls_v);                                          \
    __builtin_xtensa_ae_sa32x2f24_ric((__ae_ls_v), (ae_valign *)&(ae_ls_su),   \
                                      (ae_int32x2 **)&(ars));                  \
  })

#define AE_SA32X2F24_RIP(ae_ls_v, ae_ls_su, ars)                               \
  ({                                                                           \
    ae_int32x2 __ae_ls_v = (ae_ls_v);                                          \
    __builtin_xtensa_ae_sa32x2f24_rip((__ae_ls_v), (ae_valign *)&(ae_ls_su),   \
                                      (ae_int32x2 **)&(ars));                  \
  })

#define AE_SA64NEG_FP(ae_ls_su, ars)                                           \
  ({                                                                           \
    void *__ars = (void *)(ars);                                               \
    __builtin_xtensa_ae_sa64neg_fp((ae_valign *)&(ae_ls_su), (__ars));         \
  })

#define AE_SA64POS_FP(ae_ls_su, ars)                                           \
  ({                                                                           \
    void *__ars = (void *)(ars);                                               \
    __builtin_xtensa_ae_sa64pos_fp((ae_valign *)&(ae_ls_su), (__ars));         \
  })

#define AE_SALIGN64_I(ae_ls_su, ars, ae_immls64)                               \
  ({                                                                           \
    ae_valign __ae_ls_su = (ae_ls_su);                                         \
    ae_valign *__ars = (ars);                                                  \
    __builtin_xtensa_ae_salign64_i((__ae_ls_su), (__ars), (ae_immls64));       \
  })

#define AE_SAT16X4(ae_arth_v0, ae_arth_v1)                                     \
  ({                                                                           \
    ae_int32x2 __ae_arth_v0 = (ae_arth_v0);                                    \
    ae_int32x2 __ae_arth_v1 = (ae_arth_v1);                                    \
    __builtin_xtensa_ae_sat16x4((__ae_arth_v0), (__ae_arth_v1));               \
  })

#define AE_SAT24S(ae_arth_v1)                                                  \
  ({                                                                           \
    ae_int32x2 __ae_arth_v1 = (ae_arth_v1);                                    \
    __builtin_xtensa_ae_sat24s((__ae_arth_v1));                                \
  })

#define AE_SAT48S(ae_arth_v1)                                                  \
  ({                                                                           \
    ae_int64 __ae_arth_v1 = (ae_arth_v1);                                      \
    __builtin_xtensa_ae_sat48s((__ae_arth_v1));                                \
  })

#define AE_SATQ56S(ae_arth_v1)                                                 \
  ({                                                                           \
    ae_int64 __ae_arth_v1 = (ae_arth_v1);                                      \
    __builtin_xtensa_ae_satq56s((__ae_arth_v1));                               \
  })

#define AE_SB(ars, art)                                                        \
  ({                                                                           \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_sb((short **)&(ars), (__art));                         \
  })

#define AE_SB_IC(ars, art)                                                     \
  ({                                                                           \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_sb_ic((short **)&(ars), (__art));                      \
  })

#define AE_SB_IP(ars, art)                                                     \
  ({                                                                           \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_sb_ip((short **)&(ars), (__art));                      \
  })

#define AE_SBF(ars) ({ __builtin_xtensa_ae_sbf((short **)&(ars)); })

#define AE_SBF_IC(ars) ({ __builtin_xtensa_ae_sbf_ic((short **)&(ars)); })

#define AE_SBF_IP(ars) ({ __builtin_xtensa_ae_sbf_ip((short **)&(ars)); })

#define AE_SBI(ars, art, ae_ohba2)                                             \
  ({                                                                           \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_sbi((short **)&(ars), (__art), (ae_ohba2));            \
  })

#define AE_SBI_IC(ars, art, ae_ohba2)                                          \
  ({                                                                           \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_sbi_ic((short **)&(ars), (__art), (ae_ohba2));         \
  })

#define AE_SBI_IP(ars, art, ae_ohba2)                                          \
  ({                                                                           \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_sbi_ip((short **)&(ars), (__art), (ae_ohba2));         \
  })

#define AE_SEL16I(ae_dr_to_dr_v0, ae_dr_to_dr_v1, ae_selimm)                   \
  ({                                                                           \
    ae_int16x4 __ae_dr_to_dr_v0 = (ae_dr_to_dr_v0);                            \
    ae_int16x4 __ae_dr_to_dr_v1 = (ae_dr_to_dr_v1);                            \
    __builtin_xtensa_ae_sel16i((__ae_dr_to_dr_v0), (__ae_dr_to_dr_v1),         \
                               (ae_selimm));                                   \
  })

#define AE_SEL16I_N(ae_dr_to_dr_v0, ae_dr_to_dr_v1, ae_selimm_N)               \
  ({                                                                           \
    ae_int16x4 __ae_dr_to_dr_v0 = (ae_dr_to_dr_v0);                            \
    ae_int16x4 __ae_dr_to_dr_v1 = (ae_dr_to_dr_v1);                            \
    __builtin_xtensa_ae_sel16i_n((__ae_dr_to_dr_v0), (__ae_dr_to_dr_v1),       \
                                 (ae_selimm_N));                               \
  })

#define AE_SEXT32(ae_dr_to_dr_v0, ae_opnd_tp7)                                 \
  ({                                                                           \
    ae_int32x2 __ae_dr_to_dr_v0 = (ae_dr_to_dr_v0);                            \
    __builtin_xtensa_ae_sext32((__ae_dr_to_dr_v0), (ae_opnd_tp7));             \
  })

#define AE_SEXT32X2D16_10(ae_to_dr_v0)                                         \
  ({                                                                           \
    ae_int16x4 __ae_to_dr_v0 = (ae_to_dr_v0);                                  \
    __builtin_xtensa_ae_sext32x2d16_10((__ae_to_dr_v0));                       \
  })

#define AE_SEXT32X2D16_32(ae_to_dr_v0)                                         \
  ({                                                                           \
    ae_int16x4 __ae_to_dr_v0 = (ae_to_dr_v0);                                  \
    __builtin_xtensa_ae_sext32x2d16_32((__ae_to_dr_v0));                       \
  })

#define AE_SHA32(ars)                                                          \
  ({                                                                           \
    int __ars = (int)(ars);                                                    \
    __builtin_xtensa_ae_sha32((__ars));                                        \
  })

#define AE_SHORTSWAP(ae_to_dr_v0)                                              \
  ({                                                                           \
    ae_int16x4 __ae_to_dr_v0 = (ae_to_dr_v0);                                  \
    __builtin_xtensa_ae_shortswap((__ae_to_dr_v0));                            \
  })

#define AE_SLAA16S(ae_shift_d0, ars)                                           \
  ({                                                                           \
    ae_int16x4 __ae_shift_d0 = (ae_shift_d0);                                  \
    int __ars = (int)(ars);                                                    \
    __builtin_xtensa_ae_slaa16s((__ae_shift_d0), (__ars));                     \
  })

#define AE_SLAA32(ae_shift_d0, ars)                                            \
  ({                                                                           \
    ae_int32x2 __ae_shift_d0 = (ae_shift_d0);                                  \
    int __ars = (int)(ars);                                                    \
    __builtin_xtensa_ae_slaa32((__ae_shift_d0), (__ars));                      \
  })

#define AE_SLAA32S(ae_shift_d0, ars)                                           \
  ({                                                                           \
    ae_int32x2 __ae_shift_d0 = (ae_shift_d0);                                  \
    int __ars = (int)(ars);                                                    \
    __builtin_xtensa_ae_slaa32s((__ae_shift_d0), (__ars));                     \
  })

#define AE_SLAA64(ae_shift_d0, ars)                                            \
  ({                                                                           \
    ae_int64 __ae_shift_d0 = (ae_shift_d0);                                    \
    int __ars = (int)(ars);                                                    \
    __builtin_xtensa_ae_slaa64((__ae_shift_d0), (__ars));                      \
  })

#define AE_SLAA64S(ae_shift_d0, ars)                                           \
  ({                                                                           \
    ae_int64 __ae_shift_d0 = (ae_shift_d0);                                    \
    int __ars = (int)(ars);                                                    \
    __builtin_xtensa_ae_slaa64s((__ae_shift_d0), (__ars));                     \
  })

#define AE_SLAAQ56(ae_shift_d0, ars)                                           \
  ({                                                                           \
    ae_int64 __ae_shift_d0 = (ae_shift_d0);                                    \
    int __ars = (int)(ars);                                                    \
    __builtin_xtensa_ae_slaaq56((__ae_shift_d0), (__ars));                     \
  })

#define AE_SLAI16S(ae_shift_d0, ae_osa16)                                      \
  ({                                                                           \
    ae_int16x4 __ae_shift_d0 = (ae_shift_d0);                                  \
    __builtin_xtensa_ae_slai16s((__ae_shift_d0), (ae_osa16));                  \
  })

#define AE_SLAI24(ae_shift_d0, ae_osa32)                                       \
  ({                                                                           \
    ae_int32x2 __ae_shift_d0 = (ae_shift_d0);                                  \
    __builtin_xtensa_ae_slai24((__ae_shift_d0), (ae_osa32));                   \
  })

#define AE_SLAI24S(ae_shift_d0, ae_osa32)                                      \
  ({                                                                           \
    ae_int32x2 __ae_shift_d0 = (ae_shift_d0);                                  \
    __builtin_xtensa_ae_slai24s((__ae_shift_d0), (ae_osa32));                  \
  })

#define AE_SLAI32(ae_shift_d0, ae_osa32)                                       \
  ({                                                                           \
    ae_int32x2 __ae_shift_d0 = (ae_shift_d0);                                  \
    __builtin_xtensa_ae_slai32((__ae_shift_d0), (ae_osa32));                   \
  })

#define AE_SLAI32S(ae_shift_d0, ae_osa32)                                      \
  ({                                                                           \
    ae_int32x2 __ae_shift_d0 = (ae_shift_d0);                                  \
    __builtin_xtensa_ae_slai32s((__ae_shift_d0), (ae_osa32));                  \
  })

#define AE_SLAI64(ae_shift_d0, ae_osa64)                                       \
  ({                                                                           \
    ae_int64 __ae_shift_d0 = (ae_shift_d0);                                    \
    __builtin_xtensa_ae_slai64((__ae_shift_d0), (ae_osa64));                   \
  })

#define AE_SLAI64S(ae_shift_d0, ae_osa64)                                      \
  ({                                                                           \
    ae_int64 __ae_shift_d0 = (ae_shift_d0);                                    \
    __builtin_xtensa_ae_slai64s((__ae_shift_d0), (ae_osa64));                  \
  })

#define AE_SLAISQ56S(ae_shift_d0, ae_osa64)                                    \
  ({                                                                           \
    ae_int64 __ae_shift_d0 = (ae_shift_d0);                                    \
    __builtin_xtensa_ae_slaisq56s((__ae_shift_d0), (ae_osa64));                \
  })

#define AE_SLAS24(ae_shift_d0)                                                 \
  ({                                                                           \
    ae_int32x2 __ae_shift_d0 = (ae_shift_d0);                                  \
    __builtin_xtensa_ae_slas24((__ae_shift_d0));                               \
  })

#define AE_SLAS24S(ae_shift_d0)                                                \
  ({                                                                           \
    ae_int32x2 __ae_shift_d0 = (ae_shift_d0);                                  \
    __builtin_xtensa_ae_slas24s((__ae_shift_d0));                              \
  })

#define AE_SLAS32(ae_shift_d0)                                                 \
  ({                                                                           \
    ae_int32x2 __ae_shift_d0 = (ae_shift_d0);                                  \
    __builtin_xtensa_ae_slas32((__ae_shift_d0));                               \
  })

#define AE_SLAS32S(ae_shift_d0)                                                \
  ({                                                                           \
    ae_int32x2 __ae_shift_d0 = (ae_shift_d0);                                  \
    __builtin_xtensa_ae_slas32s((__ae_shift_d0));                              \
  })

#define AE_SLAS64(ae_shift_d0)                                                 \
  ({                                                                           \
    ae_int64 __ae_shift_d0 = (ae_shift_d0);                                    \
    __builtin_xtensa_ae_slas64((__ae_shift_d0));                               \
  })

#define AE_SLAS64S(ae_shift_d0)                                                \
  ({                                                                           \
    ae_int64 __ae_shift_d0 = (ae_shift_d0);                                    \
    __builtin_xtensa_ae_slas64s((__ae_shift_d0));                              \
  })

#define AE_SLASQ56(ae_shift_d0)                                                \
  ({                                                                           \
    ae_int64 __ae_shift_d0 = (ae_shift_d0);                                    \
    __builtin_xtensa_ae_slasq56((__ae_shift_d0));                              \
  })

#define AE_SLASSQ56S(ae_shift_d0)                                              \
  ({                                                                           \
    ae_int64 __ae_shift_d0 = (ae_shift_d0);                                    \
    __builtin_xtensa_ae_slassq56s((__ae_shift_d0));                            \
  })

#define AE_SRA64_32(ae_shift_d0, ars)                                          \
  ({                                                                           \
    ae_int32x2 __ae_shift_d0 = (ae_shift_d0);                                  \
    int __ars = (int)(ars);                                                    \
    __builtin_xtensa_ae_sra64_32((__ae_shift_d0), (__ars));                    \
  })

#define AE_SRAA16RS(ae_shift_d0, ars)                                          \
  ({                                                                           \
    ae_int16x4 __ae_shift_d0 = (ae_shift_d0);                                  \
    int __ars = (int)(ars);                                                    \
    __builtin_xtensa_ae_sraa16rs((__ae_shift_d0), (__ars));                    \
  })

#define AE_SRAA16S(ae_shift_d0, ars)                                           \
  ({                                                                           \
    ae_int16x4 __ae_shift_d0 = (ae_shift_d0);                                  \
    int __ars = (int)(ars);                                                    \
    __builtin_xtensa_ae_sraa16s((__ae_shift_d0), (__ars));                     \
  })

#define AE_SRAA32(ae_shift_d0, ars)                                            \
  ({                                                                           \
    ae_int32x2 __ae_shift_d0 = (ae_shift_d0);                                  \
    int __ars = (int)(ars);                                                    \
    __builtin_xtensa_ae_sraa32((__ae_shift_d0), (__ars));                      \
  })

#define AE_SRAA32RS(ae_shift_d0, ars)                                          \
  ({                                                                           \
    ae_int32x2 __ae_shift_d0 = (ae_shift_d0);                                  \
    int __ars = (int)(ars);                                                    \
    __builtin_xtensa_ae_sraa32rs((__ae_shift_d0), (__ars));                    \
  })

#define AE_SRAA32S(ae_shift_d0, ars)                                           \
  ({                                                                           \
    ae_int32x2 __ae_shift_d0 = (ae_shift_d0);                                  \
    int __ars = (int)(ars);                                                    \
    __builtin_xtensa_ae_sraa32s((__ae_shift_d0), (__ars));                     \
  })

#define AE_SRAA64(ae_shift_d0, ars)                                            \
  ({                                                                           \
    ae_int64 __ae_shift_d0 = (ae_shift_d0);                                    \
    int __ars = (int)(ars);                                                    \
    __builtin_xtensa_ae_sraa64((__ae_shift_d0), (__ars));                      \
  })

#define AE_SRAI16(ae_shift_d0, ae_osa16)                                       \
  ({                                                                           \
    ae_int16x4 __ae_shift_d0 = (ae_shift_d0);                                  \
    __builtin_xtensa_ae_srai16((__ae_shift_d0), (ae_osa16));                   \
  })

#define AE_SRAI16R(ae_shift_d0, ae_osa16)                                      \
  ({                                                                           \
    ae_int16x4 __ae_shift_d0 = (ae_shift_d0);                                  \
    __builtin_xtensa_ae_srai16r((__ae_shift_d0), (ae_osa16));                  \
  })

#define AE_SRAI24(ae_shift_d0, ae_osa32)                                       \
  ({                                                                           \
    ae_int32x2 __ae_shift_d0 = (ae_shift_d0);                                  \
    __builtin_xtensa_ae_srai24((__ae_shift_d0), (ae_osa32));                   \
  })

#define AE_SRAI32(ae_shift_d0, ae_osa32)                                       \
  ({                                                                           \
    ae_int32x2 __ae_shift_d0 = (ae_shift_d0);                                  \
    __builtin_xtensa_ae_srai32((__ae_shift_d0), (ae_osa32));                   \
  })

#define AE_SRAI32R(ae_shift_d0, ae_osa32)                                      \
  ({                                                                           \
    ae_int32x2 __ae_shift_d0 = (ae_shift_d0);                                  \
    __builtin_xtensa_ae_srai32r((__ae_shift_d0), (ae_osa32));                  \
  })

#define AE_SRAI64(ae_shift_d0, ae_osa64)                                       \
  ({                                                                           \
    ae_int64 __ae_shift_d0 = (ae_shift_d0);                                    \
    __builtin_xtensa_ae_srai64((__ae_shift_d0), (ae_osa64));                   \
  })

#define AE_SRAS24(ae_shift_d0)                                                 \
  ({                                                                           \
    ae_int32x2 __ae_shift_d0 = (ae_shift_d0);                                  \
    __builtin_xtensa_ae_sras24((__ae_shift_d0));                               \
  })

#define AE_SRAS32(ae_shift_d0)                                                 \
  ({                                                                           \
    ae_int32x2 __ae_shift_d0 = (ae_shift_d0);                                  \
    __builtin_xtensa_ae_sras32((__ae_shift_d0));                               \
  })

#define AE_SRAS64(ae_shift_d0)                                                 \
  ({                                                                           \
    ae_int64 __ae_shift_d0 = (ae_shift_d0);                                    \
    __builtin_xtensa_ae_sras64((__ae_shift_d0));                               \
  })

#define AE_SRLA32(ae_shift_d0, ars)                                            \
  ({                                                                           \
    ae_int32x2 __ae_shift_d0 = (ae_shift_d0);                                  \
    int __ars = (int)(ars);                                                    \
    __builtin_xtensa_ae_srla32((__ae_shift_d0), (__ars));                      \
  })

#define AE_SRLA64(ae_shift_d0, ars)                                            \
  ({                                                                           \
    ae_int64 __ae_shift_d0 = (ae_shift_d0);                                    \
    int __ars = (int)(ars);                                                    \
    __builtin_xtensa_ae_srla64((__ae_shift_d0), (__ars));                      \
  })

#define AE_SRLI24(ae_shift_d0, ae_osa32)                                       \
  ({                                                                           \
    ae_int32x2 __ae_shift_d0 = (ae_shift_d0);                                  \
    __builtin_xtensa_ae_srli24((__ae_shift_d0), (ae_osa32));                   \
  })

#define AE_SRLI32(ae_shift_d0, ae_osa32)                                       \
  ({                                                                           \
    ae_int32x2 __ae_shift_d0 = (ae_shift_d0);                                  \
    __builtin_xtensa_ae_srli32((__ae_shift_d0), (ae_osa32));                   \
  })

#define AE_SRLI64(ae_shift_d0, ae_osa64)                                       \
  ({                                                                           \
    ae_int64 __ae_shift_d0 = (ae_shift_d0);                                    \
    __builtin_xtensa_ae_srli64((__ae_shift_d0), (ae_osa64));                   \
  })

#define AE_SRLS24(ae_shift_d0)                                                 \
  ({                                                                           \
    ae_int32x2 __ae_shift_d0 = (ae_shift_d0);                                  \
    __builtin_xtensa_ae_srls24((__ae_shift_d0));                               \
  })

#define AE_SRLS32(ae_shift_d0)                                                 \
  ({                                                                           \
    ae_int32x2 __ae_shift_d0 = (ae_shift_d0);                                  \
    __builtin_xtensa_ae_srls32((__ae_shift_d0));                               \
  })

#define AE_SRLS64(ae_shift_d0)                                                 \
  ({                                                                           \
    ae_int64 __ae_shift_d0 = (ae_shift_d0);                                    \
    __builtin_xtensa_ae_srls64((__ae_shift_d0));                               \
  })

#define AE_SUB16(ae_arth_v0, ae_arth_v1)                                       \
  ({                                                                           \
    ae_int16x4 __ae_arth_v0 = (ae_arth_v0);                                    \
    ae_int16x4 __ae_arth_v1 = (ae_arth_v1);                                    \
    __builtin_xtensa_ae_sub16((__ae_arth_v0), (__ae_arth_v1));                 \
  })

#define AE_SUB16S(ae_arth_v0, ae_arth_v1)                                      \
  ({                                                                           \
    ae_int16x4 __ae_arth_v0 = (ae_arth_v0);                                    \
    ae_int16x4 __ae_arth_v1 = (ae_arth_v1);                                    \
    __builtin_xtensa_ae_sub16s((__ae_arth_v0), (__ae_arth_v1));                \
  })

#define AE_SUB24S(ae_arth_v0, ae_arth_v1)                                      \
  ({                                                                           \
    ae_int32x2 __ae_arth_v0 = (ae_arth_v0);                                    \
    ae_int32x2 __ae_arth_v1 = (ae_arth_v1);                                    \
    __builtin_xtensa_ae_sub24s((__ae_arth_v0), (__ae_arth_v1));                \
  })

#define AE_SUB32(ae_arth_v0, ae_arth_v1)                                       \
  ({                                                                           \
    ae_int32x2 __ae_arth_v0 = (ae_arth_v0);                                    \
    ae_int32x2 __ae_arth_v1 = (ae_arth_v1);                                    \
    __builtin_xtensa_ae_sub32((__ae_arth_v0), (__ae_arth_v1));                 \
  })

#define AE_SUB32S(ae_arth_v0, ae_arth_v1)                                      \
  ({                                                                           \
    ae_int32x2 __ae_arth_v0 = (ae_arth_v0);                                    \
    ae_int32x2 __ae_arth_v1 = (ae_arth_v1);                                    \
    __builtin_xtensa_ae_sub32s((__ae_arth_v0), (__ae_arth_v1));                \
  })

#define AE_SUB64(ae_arth_v0, ae_arth_v1)                                       \
  ({                                                                           \
    ae_int64 __ae_arth_v0 = (ae_arth_v0);                                      \
    ae_int64 __ae_arth_v1 = (ae_arth_v1);                                      \
    __builtin_xtensa_ae_sub64((__ae_arth_v0), (__ae_arth_v1));                 \
  })

#define AE_SUB64S(ae_arth_v0, ae_arth_v1)                                      \
  ({                                                                           \
    ae_int64 __ae_arth_v0 = (ae_arth_v0);                                      \
    ae_int64 __ae_arth_v1 = (ae_arth_v1);                                      \
    __builtin_xtensa_ae_sub64s((__ae_arth_v0), (__ae_arth_v1));                \
  })

#define AE_SUBADD32(ae_arth_v0, ae_arth_v1)                                    \
  ({                                                                           \
    ae_int32x2 __ae_arth_v0 = (ae_arth_v0);                                    \
    ae_int32x2 __ae_arth_v1 = (ae_arth_v1);                                    \
    __builtin_xtensa_ae_subadd32((__ae_arth_v0), (__ae_arth_v1));              \
  })

#define AE_SUBADD32S(ae_arth_v0, ae_arth_v1)                                   \
  ({                                                                           \
    ae_int32x2 __ae_arth_v0 = (ae_arth_v0);                                    \
    ae_int32x2 __ae_arth_v1 = (ae_arth_v1);                                    \
    __builtin_xtensa_ae_subadd32s((__ae_arth_v0), (__ae_arth_v1));             \
  })

#define AE_TRUNCA32F64S_L(ae_shift_d0, ae_shift_sd, ars)                       \
  ({                                                                           \
    ae_int32x2 __ae_shift_d0 = (ae_shift_d0);                                  \
    ae_int64 __ae_shift_sd = (ae_shift_sd);                                    \
    int __ars = (int)(ars);                                                    \
    __builtin_xtensa_ae_trunca32f64s_l((__ae_shift_d0), (__ae_shift_sd),       \
                                       (__ars));                               \
  })

#define AE_TRUNCA32X2F64S(ae_shift_d0, ae_shift_sd, ars)                       \
  ({                                                                           \
    ae_int64 __ae_shift_d0 = (ae_shift_d0);                                    \
    ae_int64 __ae_shift_sd = (ae_shift_sd);                                    \
    int __ars = (int)(ars);                                                    \
    __builtin_xtensa_ae_trunca32x2f64s((__ae_shift_d0), (__ae_shift_sd),       \
                                       (__ars));                               \
  })

#define AE_TRUNCI32F64S_L(ae_shift_d0, ae_shift_sd, ae_osa16)                  \
  ({                                                                           \
    ae_int32x2 __ae_shift_d0 = (ae_shift_d0);                                  \
    ae_int64 __ae_shift_sd = (ae_shift_sd);                                    \
    __builtin_xtensa_ae_trunci32f64s_l((__ae_shift_d0), (__ae_shift_sd),       \
                                       (ae_osa16));                            \
  })

#define AE_TRUNCI32X2F64S(ae_shift_d0, ae_shift_sd, ae_osa16)                  \
  ({                                                                           \
    ae_int64 __ae_shift_d0 = (ae_shift_d0);                                    \
    ae_int64 __ae_shift_sd = (ae_shift_sd);                                    \
    __builtin_xtensa_ae_trunci32x2f64s((__ae_shift_d0), (__ae_shift_sd),       \
                                       (ae_osa16));                            \
  })

#define AE_VLDL16C(ars)                                                        \
  ({ __builtin_xtensa_ae_vldl16c((const short **)&(ars)); })

#define AE_VLDL16C_IC(ars)                                                     \
  ({ __builtin_xtensa_ae_vldl16c_ic((const short **)&(ars)); })

#define AE_VLDL16C_IP(ars)                                                     \
  ({ __builtin_xtensa_ae_vldl16c_ip((const short **)&(ars)); })

#define AE_VLDL16T(ars)                                                        \
  ({                                                                           \
    short *__ars = (short *)(ars);                                             \
    __builtin_xtensa_ae_vldl16t((__ars));                                      \
  })

#define AE_VLDL32T(ars)                                                        \
  ({                                                                           \
    int *__ars = (int *)(ars);                                                 \
    __builtin_xtensa_ae_vldl32t((__ars));                                      \
  })

#define AE_VLDSHT(art)                                                         \
  ({                                                                           \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_ae_vldsht((__art));                                       \
  })

#define AE_VLEL16T(br, art, ars)                                               \
  ({                                                                           \
    short *__ars = (short *)(ars);                                             \
    __builtin_xtensa_ae_vlel16t((xtbool *)&(br), (int *)&(art), (__ars));      \
  })

#define AE_VLEL32T(br, art, ars)                                               \
  ({                                                                           \
    int *__ars = (int *)(ars);                                                 \
    __builtin_xtensa_ae_vlel32t((xtbool *)&(br), (int *)&(art), (__ars));      \
  })

#define AE_VLES16C(ars) ({ __builtin_xtensa_ae_vles16c((short **)&(ars)); })

#define AE_VLES16C_IC(ars)                                                     \
  ({ __builtin_xtensa_ae_vles16c_ic((short **)&(ars)); })

#define AE_VLES16C_IP(ars)                                                     \
  ({ __builtin_xtensa_ae_vles16c_ip((short **)&(ars)); })

#define AE_XOR(ae_dr_to_dr_v0, ae_dr_to_dr_v1)                                 \
  ({                                                                           \
    ae_int64 __ae_dr_to_dr_v0 = (ae_dr_to_dr_v0);                              \
    ae_int64 __ae_dr_to_dr_v1 = (ae_dr_to_dr_v1);                              \
    __builtin_xtensa_ae_xor((__ae_dr_to_dr_v0), (__ae_dr_to_dr_v1));           \
  })

#define AE_ZALIGN64() ({ __builtin_xtensa_ae_zalign64(); })

#define RUR_AE_BITHEAD() ({ __builtin_xtensa_rur_ae_bithead(); })

#define RUR_AE_BITPTR() ({ __builtin_xtensa_rur_ae_bitptr(); })

#define RUR_AE_BITSUSED() ({ __builtin_xtensa_rur_ae_bitsused(); })

#define RUR_AE_CBEGIN0() ({ __builtin_xtensa_rur_ae_cbegin0(); })

#define RUR_AE_CEND0() ({ __builtin_xtensa_rur_ae_cend0(); })

#define RUR_AE_CW_SD_NO() ({ __builtin_xtensa_rur_ae_cw_sd_no(); })

#define RUR_AE_CWRAP() ({ __builtin_xtensa_rur_ae_cwrap(); })

#define RUR_AE_FIRST_TS() ({ __builtin_xtensa_rur_ae_first_ts(); })

#define RUR_AE_NEXTOFFSET() ({ __builtin_xtensa_rur_ae_nextoffset(); })

#define RUR_AE_OVERFLOW() ({ __builtin_xtensa_rur_ae_overflow(); })

#define RUR_AE_OVF_SAR() ({ __builtin_xtensa_rur_ae_ovf_sar(); })

#define RUR_AE_SAR() ({ __builtin_xtensa_rur_ae_sar(); })

#define RUR_AE_SEARCHDONE() ({ __builtin_xtensa_rur_ae_searchdone(); })

#define RUR_AE_TABLESIZE() ({ __builtin_xtensa_rur_ae_tablesize(); })

#define RUR_AE_TS_FTS_BU_BP() ({ __builtin_xtensa_rur_ae_ts_fts_bu_bp(); })

#define WUR_AE_BITHEAD(art)                                                    \
  ({                                                                           \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_wur_ae_bithead((__art));                                  \
  })

#define WUR_AE_BITPTR(art)                                                     \
  ({                                                                           \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_wur_ae_bitptr((__art));                                   \
  })

#define WUR_AE_BITSUSED(art)                                                   \
  ({                                                                           \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_wur_ae_bitsused((__art));                                 \
  })

#define WUR_AE_CBEGIN0(art)                                                    \
  ({                                                                           \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_wur_ae_cbegin0((__art));                                  \
  })

#define WUR_AE_CEND0(art)                                                      \
  ({                                                                           \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_wur_ae_cend0((__art));                                    \
  })

#define WUR_AE_CW_SD_NO(art)                                                   \
  ({                                                                           \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_wur_ae_cw_sd_no((__art));                                 \
  })

#define WUR_AE_CWRAP(art)                                                      \
  ({                                                                           \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_wur_ae_cwrap((__art));                                    \
  })

#define WUR_AE_FIRST_TS(art)                                                   \
  ({                                                                           \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_wur_ae_first_ts((__art));                                 \
  })

#define WUR_AE_NEXTOFFSET(art)                                                 \
  ({                                                                           \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_wur_ae_nextoffset((__art));                               \
  })

#define WUR_AE_OVERFLOW(art)                                                   \
  ({                                                                           \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_wur_ae_overflow((__art));                                 \
  })

#define WUR_AE_OVF_SAR(art)                                                    \
  ({                                                                           \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_wur_ae_ovf_sar((__art));                                  \
  })

#define WUR_AE_SAR(art)                                                        \
  ({                                                                           \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_wur_ae_sar((__art));                                      \
  })

#define WUR_AE_SEARCHDONE(art)                                                 \
  ({                                                                           \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_wur_ae_searchdone((__art));                               \
  })

#define WUR_AE_TABLESIZE(art)                                                  \
  ({                                                                           \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_wur_ae_tablesize((__art));                                \
  })

#define WUR_AE_TS_FTS_BU_BP(art)                                               \
  ({                                                                           \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_wur_ae_ts_fts_bu_bp((__art));                             \
  })

#define XT_ABS_S(frs)                                                          \
  ({                                                                           \
    xtfloat __frs = (xtfloat)(frs);                                            \
    __builtin_xtensa_xt_abs_s((__frs));                                        \
  })

#define XT_ADD_S(frs, frt)                                                     \
  ({                                                                           \
    xtfloat __frs = (xtfloat)(frs);                                            \
    xtfloat __frt = (xtfloat)(frt);                                            \
    __builtin_xtensa_xt_add_s((__frs), (__frt));                               \
  })

#define XT_ADDEXP_S(frr, frs)                                                  \
  ({                                                                           \
    xtfloat __frs = (xtfloat)(frs);                                            \
    __builtin_xtensa_xt_addexp_s((xtfloat *)&(frr), (__frs));                  \
  })

#define XT_ADDEXPM_S(frr, frs)                                                 \
  ({                                                                           \
    xtfloat __frs = (xtfloat)(frs);                                            \
    __builtin_xtensa_xt_addexpm_s((xtfloat *)&(frr), (__frs));                 \
  })

#define XT_CEIL_S(frs, imm_t)                                                  \
  ({                                                                           \
    xtfloat __frs = (xtfloat)(frs);                                            \
    __builtin_xtensa_xt_ceil_s((__frs), (imm_t));                              \
  })

#define XT_DIV0_S(frs)                                                         \
  ({                                                                           \
    xtfloat __frs = (xtfloat)(frs);                                            \
    __builtin_xtensa_xt_div0_s((__frs));                                       \
  })

#define XT_DIVN_S(frr, frs, frt)                                               \
  ({                                                                           \
    xtfloat __frs = (xtfloat)(frs);                                            \
    xtfloat __frt = (xtfloat)(frt);                                            \
    __builtin_xtensa_xt_divn_s((xtfloat *)&(frr), (__frs), (__frt));           \
  })

#define XT_FLOAT_S(ars, imm_t)                                                 \
  ({                                                                           \
    int __ars = (int)(ars);                                                    \
    __builtin_xtensa_xt_float_s((__ars), (imm_t));                             \
  })

#define XT_FLOOR_S(frs, imm_t)                                                 \
  ({                                                                           \
    xtfloat __frs = (xtfloat)(frs);                                            \
    __builtin_xtensa_xt_floor_s((__frs), (imm_t));                             \
  })

#define XT_LSI(ars, imm8x4)                                                    \
  ({                                                                           \
    xtfloat *__ars = (xtfloat *)(ars);                                         \
    __builtin_xtensa_xt_lsi((__ars), (imm8x4));                                \
  })

#define XT_LSIP(frt, ars, imm8x4)                                              \
  ({                                                                           \
    __builtin_xtensa_xt_lsip((xtfloat *)&(frt), (const xtfloat **)&(ars),      \
                             (imm8x4));                                        \
  })

#define XT_LSX(ars, art)                                                       \
  ({                                                                           \
    xtfloat *__ars = (xtfloat *)(ars);                                         \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_xt_lsx((__ars), (__art));                                 \
  })

#define XT_LSXP(frr, ars, art)                                                 \
  ({                                                                           \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_xt_lsxp((xtfloat *)&(frr), (const xtfloat **)&(ars),      \
                             (__art));                                         \
  })

#define XT_MADD_S(frr, frs, frt)                                               \
  ({                                                                           \
    xtfloat __frs = (xtfloat)(frs);                                            \
    xtfloat __frt = (xtfloat)(frt);                                            \
    __builtin_xtensa_xt_madd_s((xtfloat *)&(frr), (__frs), (__frt));           \
  })

#define XT_MADDN_S(frr, frs, frt)                                              \
  ({                                                                           \
    xtfloat __frs = (xtfloat)(frs);                                            \
    xtfloat __frt = (xtfloat)(frt);                                            \
    __builtin_xtensa_xt_maddn_s((xtfloat *)&(frr), (__frs), (__frt));          \
  })

#define XT_MKDADJ_S(frr, frs)                                                  \
  ({                                                                           \
    xtfloat __frs = (xtfloat)(frs);                                            \
    __builtin_xtensa_xt_mkdadj_s((xtfloat *)&(frr), (__frs));                  \
  })

#define XT_MKSADJ_S(frs)                                                       \
  ({                                                                           \
    xtfloat __frs = (xtfloat)(frs);                                            \
    __builtin_xtensa_xt_mksadj_s((__frs));                                     \
  })

#define XT_MOV_S(frs)                                                          \
  ({                                                                           \
    xtfloat __frs = (xtfloat)(frs);                                            \
    __builtin_xtensa_xt_mov_s((__frs));                                        \
  })

#define XT_MOVEQZ_S(frr, frs, art)                                             \
  ({                                                                           \
    xtfloat __frs = (xtfloat)(frs);                                            \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_xt_moveqz_s((xtfloat *)&(frr), (__frs), (__art));         \
  })

#define XT_MOVF_S(frr, frs, bt)                                                \
  ({                                                                           \
    xtfloat __frs = (xtfloat)(frs);                                            \
    xtbool __bt = (bt);                                                        \
    __builtin_xtensa_xt_movf_s((xtfloat *)&(frr), (__frs), (__bt));            \
  })

#define XT_MOVGEZ_S(frr, frs, art)                                             \
  ({                                                                           \
    xtfloat __frs = (xtfloat)(frs);                                            \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_xt_movgez_s((xtfloat *)&(frr), (__frs), (__art));         \
  })

#define XT_MOVLTZ_S(frr, frs, art)                                             \
  ({                                                                           \
    xtfloat __frs = (xtfloat)(frs);                                            \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_xt_movltz_s((xtfloat *)&(frr), (__frs), (__art));         \
  })

#define XT_MOVNEZ_S(frr, frs, art)                                             \
  ({                                                                           \
    xtfloat __frs = (xtfloat)(frs);                                            \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_xt_movnez_s((xtfloat *)&(frr), (__frs), (__art));         \
  })

#define XT_MOVT_S(frr, frs, bt)                                                \
  ({                                                                           \
    xtfloat __frs = (xtfloat)(frs);                                            \
    xtbool __bt = (bt);                                                        \
    __builtin_xtensa_xt_movt_s((xtfloat *)&(frr), (__frs), (__bt));            \
  })

#define XT_MSUB_S(frr, frs, frt)                                               \
  ({                                                                           \
    xtfloat __frs = (xtfloat)(frs);                                            \
    xtfloat __frt = (xtfloat)(frt);                                            \
    __builtin_xtensa_xt_msub_s((xtfloat *)&(frr), (__frs), (__frt));           \
  })

#define XT_MUL_S(frs, frt)                                                     \
  ({                                                                           \
    xtfloat __frs = (xtfloat)(frs);                                            \
    xtfloat __frt = (xtfloat)(frt);                                            \
    __builtin_xtensa_xt_mul_s((__frs), (__frt));                               \
  })

#define XT_NEG_S(frs)                                                          \
  ({                                                                           \
    xtfloat __frs = (xtfloat)(frs);                                            \
    __builtin_xtensa_xt_neg_s((__frs));                                        \
  })

#define XT_NEXP01_S(frs)                                                       \
  ({                                                                           \
    xtfloat __frs = (xtfloat)(frs);                                            \
    __builtin_xtensa_xt_nexp01_s((__frs));                                     \
  })

#define XT_OEQ_S(frs, frt)                                                     \
  ({                                                                           \
    xtfloat __frs = (xtfloat)(frs);                                            \
    xtfloat __frt = (xtfloat)(frt);                                            \
    __builtin_xtensa_xt_oeq_s((__frs), (__frt));                               \
  })

#define XT_OLE_S(frs, frt)                                                     \
  ({                                                                           \
    xtfloat __frs = (xtfloat)(frs);                                            \
    xtfloat __frt = (xtfloat)(frt);                                            \
    __builtin_xtensa_xt_ole_s((__frs), (__frt));                               \
  })

#define XT_OLT_S(frs, frt)                                                     \
  ({                                                                           \
    xtfloat __frs = (xtfloat)(frs);                                            \
    xtfloat __frt = (xtfloat)(frt);                                            \
    __builtin_xtensa_xt_olt_s((__frs), (__frt));                               \
  })

#define XT_RECIP0_S(frs)                                                       \
  ({                                                                           \
    xtfloat __frs = (xtfloat)(frs);                                            \
    __builtin_xtensa_xt_recip0_s((__frs));                                     \
  })

#define XT_RFR(frs)                                                            \
  ({                                                                           \
    xtfloat __frs = (xtfloat)(frs);                                            \
    __builtin_xtensa_xt_rfr((__frs));                                          \
  })

#define XT_ROUND_S(frs, imm_t)                                                 \
  ({                                                                           \
    xtfloat __frs = (xtfloat)(frs);                                            \
    __builtin_xtensa_xt_round_s((__frs), (imm_t));                             \
  })

#define XT_RSQRT0_S(frs)                                                       \
  ({                                                                           \
    xtfloat __frs = (xtfloat)(frs);                                            \
    __builtin_xtensa_xt_rsqrt0_s((__frs));                                     \
  })

#define XT_RUR_FCR() ({ __builtin_xtensa_xt_rur_fcr(); })

#define XT_RUR_FSR() ({ __builtin_xtensa_xt_rur_fsr(); })

#define XT_SQRT0_S(frs)                                                        \
  ({                                                                           \
    xtfloat __frs = (xtfloat)(frs);                                            \
    __builtin_xtensa_xt_sqrt0_s((__frs));                                      \
  })

#define XT_SSI(frt, ars, imm8x4)                                               \
  ({                                                                           \
    xtfloat __frt = (xtfloat)(frt);                                            \
    xtfloat *__ars = (xtfloat *)(ars);                                         \
    __builtin_xtensa_xt_ssi((__frt), (__ars), (imm8x4));                       \
  })

#define XT_SSIP(frt, ars, imm8x4)                                              \
  ({                                                                           \
    xtfloat __frt = (xtfloat)(frt);                                            \
    __builtin_xtensa_xt_ssip((__frt), (xtfloat **)&(ars), (imm8x4));           \
  })

#define XT_SSX(frr, ars, art)                                                  \
  ({                                                                           \
    xtfloat __frr = (xtfloat)(frr);                                            \
    xtfloat *__ars = (xtfloat *)(ars);                                         \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_xt_ssx((__frr), (__ars), (__art));                        \
  })

#define XT_SSXP(frr, ars, art)                                                 \
  ({                                                                           \
    xtfloat __frr = (xtfloat)(frr);                                            \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_xt_ssxp((__frr), (xtfloat **)&(ars), (__art));            \
  })

#define XT_SUB_S(frs, frt)                                                     \
  ({                                                                           \
    xtfloat __frs = (xtfloat)(frs);                                            \
    xtfloat __frt = (xtfloat)(frt);                                            \
    __builtin_xtensa_xt_sub_s((__frs), (__frt));                               \
  })

#define XT_TRUNC_S(frs, imm_t)                                                 \
  ({                                                                           \
    xtfloat __frs = (xtfloat)(frs);                                            \
    __builtin_xtensa_xt_trunc_s((__frs), (imm_t));                             \
  })

#define XT_UEQ_S(frs, frt)                                                     \
  ({                                                                           \
    xtfloat __frs = (xtfloat)(frs);                                            \
    xtfloat __frt = (xtfloat)(frt);                                            \
    __builtin_xtensa_xt_ueq_s((__frs), (__frt));                               \
  })

#define XT_UFLOAT_S(ars, imm_t)                                                \
  ({                                                                           \
    int __ars = (int)(ars);                                                    \
    __builtin_xtensa_xt_ufloat_s((__ars), (imm_t));                            \
  })

#define XT_ULE_S(frs, frt)                                                     \
  ({                                                                           \
    xtfloat __frs = (xtfloat)(frs);                                            \
    xtfloat __frt = (xtfloat)(frt);                                            \
    __builtin_xtensa_xt_ule_s((__frs), (__frt));                               \
  })

#define XT_ULT_S(frs, frt)                                                     \
  ({                                                                           \
    xtfloat __frs = (xtfloat)(frs);                                            \
    xtfloat __frt = (xtfloat)(frt);                                            \
    __builtin_xtensa_xt_ult_s((__frs), (__frt));                               \
  })

#define XT_UN_S(frs, frt)                                                      \
  ({                                                                           \
    xtfloat __frs = (xtfloat)(frs);                                            \
    xtfloat __frt = (xtfloat)(frt);                                            \
    __builtin_xtensa_xt_un_s((__frs), (__frt));                                \
  })

#define XT_UTRUNC_S(frs, imm_t)                                                \
  ({                                                                           \
    xtfloat __frs = (xtfloat)(frs);                                            \
    __builtin_xtensa_xt_utrunc_s((__frs), (imm_t));                            \
  })

#define XT_WFR(ars)                                                            \
  ({                                                                           \
    int __ars = (int)(ars);                                                    \
    __builtin_xtensa_xt_wfr((__ars));                                          \
  })

#define XT_WUR_FCR(art)                                                        \
  ({                                                                           \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_xt_wur_fcr((__art));                                      \
  })

#define XT_WUR_FSR(art)                                                        \
  ({                                                                           \
    int __art = (int)(art);                                                    \
    __builtin_xtensa_xt_wur_fsr((__art));                                      \
  })

#endif /* __XTENSAHIFI3INTRIN_H */
