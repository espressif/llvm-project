//===-- XtensaSemaCheck.inc - Clang semantic checks for Xtensa arch ----*- C++
//-*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

case Xtensa::BI__builtin_xtensa_ae_dbi:
return SemaBuiltinConstantArgRange(TheCall, 1, 1, 16);
case Xtensa::BI__builtin_xtensa_ae_dbi_ic:
return SemaBuiltinConstantArgRange(TheCall, 1, 1, 16);
case Xtensa::BI__builtin_xtensa_ae_dbi_ip:
return SemaBuiltinConstantArgRange(TheCall, 1, 1, 16);
case Xtensa::BI__builtin_xtensa_ae_l16_i:
return SemaBuiltinConstantArgRange(TheCall, 1, -16, 14) ||
       SemaBuiltinConstantArgMultiple(TheCall, 1, 2);
case Xtensa::BI__builtin_xtensa_ae_l16_ip:
return SemaBuiltinConstantArgRange(TheCall, 2, -16, 14) ||
       SemaBuiltinConstantArgMultiple(TheCall, 2, 2);
case Xtensa::BI__builtin_xtensa_ae_l16m_i:
return SemaBuiltinConstantArgRange(TheCall, 1, -16, 14) ||
       SemaBuiltinConstantArgMultiple(TheCall, 1, 2);
case Xtensa::BI__builtin_xtensa_ae_l16m_iu:
return SemaBuiltinConstantArgRange(TheCall, 2, -16, 14) ||
       SemaBuiltinConstantArgMultiple(TheCall, 2, 2);
case Xtensa::BI__builtin_xtensa_ae_l16x2m_i:
return SemaBuiltinConstantArgRange(TheCall, 1, -32, 28) ||
       SemaBuiltinConstantArgMultiple(TheCall, 1, 4);
case Xtensa::BI__builtin_xtensa_ae_l16x2m_iu:
return SemaBuiltinConstantArgRange(TheCall, 2, -32, 28) ||
       SemaBuiltinConstantArgMultiple(TheCall, 2, 4);
case Xtensa::BI__builtin_xtensa_ae_l16x4_i:
return SemaBuiltinConstantArgRange(TheCall, 1, -64, 56) ||
       SemaBuiltinConstantArgMultiple(TheCall, 1, 8);
case Xtensa::BI__builtin_xtensa_ae_l16x4_ip:
return SemaBuiltinConstantArgRange(TheCall, 2, 0, 56) ||
       SemaBuiltinConstantArgMultiple(TheCall, 2, 8);
case Xtensa::BI__builtin_xtensa_ae_l32_i:
return SemaBuiltinConstantArgRange(TheCall, 1, -32, 28) ||
       SemaBuiltinConstantArgMultiple(TheCall, 1, 4);
case Xtensa::BI__builtin_xtensa_ae_l32_ip:
return SemaBuiltinConstantArgRange(TheCall, 2, -32, 28) ||
       SemaBuiltinConstantArgMultiple(TheCall, 2, 4);
case Xtensa::BI__builtin_xtensa_ae_l32f24_i:
return SemaBuiltinConstantArgRange(TheCall, 1, -32, 28) ||
       SemaBuiltinConstantArgMultiple(TheCall, 1, 4);
case Xtensa::BI__builtin_xtensa_ae_l32f24_ip:
return SemaBuiltinConstantArgRange(TheCall, 2, -32, 28) ||
       SemaBuiltinConstantArgMultiple(TheCall, 2, 4);
case Xtensa::BI__builtin_xtensa_ae_l32m_i:
return SemaBuiltinConstantArgRange(TheCall, 1, -32, 28) ||
       SemaBuiltinConstantArgMultiple(TheCall, 1, 4);
case Xtensa::BI__builtin_xtensa_ae_l32m_iu:
return SemaBuiltinConstantArgRange(TheCall, 2, -32, 28) ||
       SemaBuiltinConstantArgMultiple(TheCall, 2, 4);
case Xtensa::BI__builtin_xtensa_ae_l32x2_i:
return SemaBuiltinConstantArgRange(TheCall, 1, -64, 56) ||
       SemaBuiltinConstantArgMultiple(TheCall, 1, 8);
case Xtensa::BI__builtin_xtensa_ae_l32x2_ip:
return SemaBuiltinConstantArgRange(TheCall, 2, 0, 56) ||
       SemaBuiltinConstantArgMultiple(TheCall, 2, 8);
case Xtensa::BI__builtin_xtensa_ae_l32x2f24_i:
return SemaBuiltinConstantArgRange(TheCall, 1, -64, 56) ||
       SemaBuiltinConstantArgMultiple(TheCall, 1, 8);
case Xtensa::BI__builtin_xtensa_ae_l32x2f24_ip:
return SemaBuiltinConstantArgRange(TheCall, 2, 0, 56) ||
       SemaBuiltinConstantArgMultiple(TheCall, 2, 8);
case Xtensa::BI__builtin_xtensa_ae_l64_i:
return SemaBuiltinConstantArgRange(TheCall, 1, -64, 56) ||
       SemaBuiltinConstantArgMultiple(TheCall, 1, 8);
case Xtensa::BI__builtin_xtensa_ae_l64_ip:
return SemaBuiltinConstantArgRange(TheCall, 2, -64, 56) ||
       SemaBuiltinConstantArgMultiple(TheCall, 2, 8);
case Xtensa::BI__builtin_xtensa_ae_lalign64_i:
return SemaBuiltinConstantArgRange(TheCall, 1, -64, 56) ||
       SemaBuiltinConstantArgMultiple(TheCall, 1, 8);
case Xtensa::BI__builtin_xtensa_ae_lbi:
return SemaBuiltinConstantArgRange(TheCall, 0, 1, 16);
case Xtensa::BI__builtin_xtensa_ae_lbki:
return SemaBuiltinConstantArgRange(TheCall, 1, 1, 16);
case Xtensa::BI__builtin_xtensa_ae_lbsi:
return SemaBuiltinConstantArgRange(TheCall, 0, 1, 16);
case Xtensa::BI__builtin_xtensa_ae_movi:
return SemaBuiltinConstantArgRange(TheCall, 0, -16, 47);
case Xtensa::BI__builtin_xtensa_ae_pksr24:
return SemaBuiltinConstantArgRange(TheCall, 2, 0, 3);
case Xtensa::BI__builtin_xtensa_ae_pksr32:
return SemaBuiltinConstantArgRange(TheCall, 2, 0, 3);
case Xtensa::BI__builtin_xtensa_ae_s16_0_i:
return SemaBuiltinConstantArgRange(TheCall, 2, -16, 14) ||
       SemaBuiltinConstantArgMultiple(TheCall, 2, 2);
case Xtensa::BI__builtin_xtensa_ae_s16_0_ip:
return SemaBuiltinConstantArgRange(TheCall, 2, -16, 14) ||
       SemaBuiltinConstantArgMultiple(TheCall, 2, 2);
case Xtensa::BI__builtin_xtensa_ae_s16m_l_i:
return SemaBuiltinConstantArgRange(TheCall, 2, -16, 14) ||
       SemaBuiltinConstantArgMultiple(TheCall, 2, 2);
case Xtensa::BI__builtin_xtensa_ae_s16m_l_iu:
return SemaBuiltinConstantArgRange(TheCall, 2, -16, 14) ||
       SemaBuiltinConstantArgMultiple(TheCall, 2, 2);
case Xtensa::BI__builtin_xtensa_ae_s16x2m_i:
return SemaBuiltinConstantArgRange(TheCall, 2, -32, 28) ||
       SemaBuiltinConstantArgMultiple(TheCall, 2, 4);
case Xtensa::BI__builtin_xtensa_ae_s16x2m_iu:
return SemaBuiltinConstantArgRange(TheCall, 2, -32, 28) ||
       SemaBuiltinConstantArgMultiple(TheCall, 2, 4);
case Xtensa::BI__builtin_xtensa_ae_s16x4_i:
return SemaBuiltinConstantArgRange(TheCall, 2, -64, 56) ||
       SemaBuiltinConstantArgMultiple(TheCall, 2, 8);
case Xtensa::BI__builtin_xtensa_ae_s16x4_ip:
return SemaBuiltinConstantArgRange(TheCall, 2, 0, 56) ||
       SemaBuiltinConstantArgMultiple(TheCall, 2, 8);
case Xtensa::BI__builtin_xtensa_ae_s24ra64s_i:
return SemaBuiltinConstantArgRange(TheCall, 2, -32, 28) ||
       SemaBuiltinConstantArgMultiple(TheCall, 2, 4);
case Xtensa::BI__builtin_xtensa_ae_s24ra64s_ip:
return SemaBuiltinConstantArgRange(TheCall, 2, -32, 28) ||
       SemaBuiltinConstantArgMultiple(TheCall, 2, 4);
case Xtensa::BI__builtin_xtensa_ae_s32_l_i:
return SemaBuiltinConstantArgRange(TheCall, 2, -32, 28) ||
       SemaBuiltinConstantArgMultiple(TheCall, 2, 4);
case Xtensa::BI__builtin_xtensa_ae_s32_l_ip:
return SemaBuiltinConstantArgRange(TheCall, 2, -32, 28) ||
       SemaBuiltinConstantArgMultiple(TheCall, 2, 4);
case Xtensa::BI__builtin_xtensa_ae_s32f24_l_i:
return SemaBuiltinConstantArgRange(TheCall, 2, -32, 28) ||
       SemaBuiltinConstantArgMultiple(TheCall, 2, 4);
case Xtensa::BI__builtin_xtensa_ae_s32f24_l_ip:
return SemaBuiltinConstantArgRange(TheCall, 2, -32, 28) ||
       SemaBuiltinConstantArgMultiple(TheCall, 2, 4);
case Xtensa::BI__builtin_xtensa_ae_s32m_i:
return SemaBuiltinConstantArgRange(TheCall, 2, -32, 28) ||
       SemaBuiltinConstantArgMultiple(TheCall, 2, 4);
case Xtensa::BI__builtin_xtensa_ae_s32m_iu:
return SemaBuiltinConstantArgRange(TheCall, 2, -32, 28) ||
       SemaBuiltinConstantArgMultiple(TheCall, 2, 4);
case Xtensa::BI__builtin_xtensa_ae_s32ra64s_i:
return SemaBuiltinConstantArgRange(TheCall, 2, -32, 28) ||
       SemaBuiltinConstantArgMultiple(TheCall, 2, 4);
case Xtensa::BI__builtin_xtensa_ae_s32ra64s_ip:
return SemaBuiltinConstantArgRange(TheCall, 2, -32, 28) ||
       SemaBuiltinConstantArgMultiple(TheCall, 2, 4);
case Xtensa::BI__builtin_xtensa_ae_s32x2_i:
return SemaBuiltinConstantArgRange(TheCall, 2, -64, 56) ||
       SemaBuiltinConstantArgMultiple(TheCall, 2, 8);
case Xtensa::BI__builtin_xtensa_ae_s32x2_ip:
return SemaBuiltinConstantArgRange(TheCall, 2, 0, 56) ||
       SemaBuiltinConstantArgMultiple(TheCall, 2, 8);
case Xtensa::BI__builtin_xtensa_ae_s32x2f24_i:
return SemaBuiltinConstantArgRange(TheCall, 2, -64, 56) ||
       SemaBuiltinConstantArgMultiple(TheCall, 2, 8);
case Xtensa::BI__builtin_xtensa_ae_s32x2f24_ip:
return SemaBuiltinConstantArgRange(TheCall, 2, 0, 56) ||
       SemaBuiltinConstantArgMultiple(TheCall, 2, 8);
case Xtensa::BI__builtin_xtensa_ae_s64_i:
return SemaBuiltinConstantArgRange(TheCall, 2, -64, 56) ||
       SemaBuiltinConstantArgMultiple(TheCall, 2, 8);
case Xtensa::BI__builtin_xtensa_ae_s64_ip:
return SemaBuiltinConstantArgRange(TheCall, 2, -64, 56) ||
       SemaBuiltinConstantArgMultiple(TheCall, 2, 8);
case Xtensa::BI__builtin_xtensa_ae_salign64_i:
return SemaBuiltinConstantArgRange(TheCall, 2, -64, 56) ||
       SemaBuiltinConstantArgMultiple(TheCall, 2, 8);
case Xtensa::BI__builtin_xtensa_ae_sbi:
return SemaBuiltinConstantArgRange(TheCall, 2, 1, 16);
case Xtensa::BI__builtin_xtensa_ae_sbi_ic:
return SemaBuiltinConstantArgRange(TheCall, 2, 1, 16);
case Xtensa::BI__builtin_xtensa_ae_sbi_ip:
return SemaBuiltinConstantArgRange(TheCall, 2, 1, 16);
case Xtensa::BI__builtin_xtensa_ae_sel16i:
return SemaBuiltinConstantArgRange(TheCall, 2, 0, 15);
case Xtensa::BI__builtin_xtensa_ae_sel16i_n:
return SemaBuiltinConstantArgRange(TheCall, 2, 0, 3);
case Xtensa::BI__builtin_xtensa_ae_sext32:
return SemaBuiltinConstantArgRange(TheCall, 1, 7, 22);
case Xtensa::BI__builtin_xtensa_ae_slai16s:
return SemaBuiltinConstantArgRange(TheCall, 1, 0, 15);
case Xtensa::BI__builtin_xtensa_ae_slai24:
return SemaBuiltinConstantArgRange(TheCall, 1, 0, 31);
case Xtensa::BI__builtin_xtensa_ae_slai24s:
return SemaBuiltinConstantArgRange(TheCall, 1, 0, 31);
case Xtensa::BI__builtin_xtensa_ae_slai32:
return SemaBuiltinConstantArgRange(TheCall, 1, 0, 31);
case Xtensa::BI__builtin_xtensa_ae_slai32s:
return SemaBuiltinConstantArgRange(TheCall, 1, 0, 31);
case Xtensa::BI__builtin_xtensa_ae_slai64:
return SemaBuiltinConstantArgRange(TheCall, 1, 0, 63);
case Xtensa::BI__builtin_xtensa_ae_slai64s:
return SemaBuiltinConstantArgRange(TheCall, 1, 0, 63);
case Xtensa::BI__builtin_xtensa_ae_slaisq56s:
return SemaBuiltinConstantArgRange(TheCall, 1, 0, 63);
case Xtensa::BI__builtin_xtensa_ae_srai16:
return SemaBuiltinConstantArgRange(TheCall, 1, 0, 15);
case Xtensa::BI__builtin_xtensa_ae_srai16r:
return SemaBuiltinConstantArgRange(TheCall, 1, 0, 15);
case Xtensa::BI__builtin_xtensa_ae_srai24:
return SemaBuiltinConstantArgRange(TheCall, 1, 0, 31);
case Xtensa::BI__builtin_xtensa_ae_srai32:
return SemaBuiltinConstantArgRange(TheCall, 1, 0, 31);
case Xtensa::BI__builtin_xtensa_ae_srai32r:
return SemaBuiltinConstantArgRange(TheCall, 1, 0, 31);
case Xtensa::BI__builtin_xtensa_ae_srai64:
return SemaBuiltinConstantArgRange(TheCall, 1, 0, 63);
case Xtensa::BI__builtin_xtensa_ae_srli24:
return SemaBuiltinConstantArgRange(TheCall, 1, 0, 31);
case Xtensa::BI__builtin_xtensa_ae_srli32:
return SemaBuiltinConstantArgRange(TheCall, 1, 0, 31);
case Xtensa::BI__builtin_xtensa_ae_srli64:
return SemaBuiltinConstantArgRange(TheCall, 1, 0, 63);
case Xtensa::BI__builtin_xtensa_ae_trunci32f64s_l:
return SemaBuiltinConstantArgRange(TheCall, 2, 0, 15);
case Xtensa::BI__builtin_xtensa_ae_trunci32x2f64s:
return SemaBuiltinConstantArgRange(TheCall, 2, 0, 15);
