//===-- XtensaSemaCheck.inc - Clang semantic checks for Xtensa arch ----*- C++
//-*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

case Xtensa::BI__builtin_xtensa_ae_dbi:
return SemaRef.BuiltinConstantArgRange(TheCall, 1, 1, 16);
case Xtensa::BI__builtin_xtensa_ae_dbi_ic:
return SemaRef.BuiltinConstantArgRange(TheCall, 1, 1, 16);
case Xtensa::BI__builtin_xtensa_ae_dbi_ip:
return SemaRef.BuiltinConstantArgRange(TheCall, 1, 1, 16);
case Xtensa::BI__builtin_xtensa_ae_l16_i:
return SemaRef.BuiltinConstantArgRange(TheCall, 1, -16, 14) ||
       SemaRef.BuiltinConstantArgMultiple(TheCall, 1, 2);
case Xtensa::BI__builtin_xtensa_ae_l16_ip:
return SemaRef.BuiltinConstantArgRange(TheCall, 2, -16, 14) ||
       SemaRef.BuiltinConstantArgMultiple(TheCall, 2, 2);
case Xtensa::BI__builtin_xtensa_ae_l16m_i:
return SemaRef.BuiltinConstantArgRange(TheCall, 1, -16, 14) ||
       SemaRef.BuiltinConstantArgMultiple(TheCall, 1, 2);
case Xtensa::BI__builtin_xtensa_ae_l16m_iu:
return SemaRef.BuiltinConstantArgRange(TheCall, 2, -16, 14) ||
       SemaRef.BuiltinConstantArgMultiple(TheCall, 2, 2);
case Xtensa::BI__builtin_xtensa_ae_l16x2m_i:
return SemaRef.BuiltinConstantArgRange(TheCall, 1, -32, 28) ||
       SemaRef.BuiltinConstantArgMultiple(TheCall, 1, 4);
case Xtensa::BI__builtin_xtensa_ae_l16x2m_iu:
return SemaRef.BuiltinConstantArgRange(TheCall, 2, -32, 28) ||
       SemaRef.BuiltinConstantArgMultiple(TheCall, 2, 4);
case Xtensa::BI__builtin_xtensa_ae_l16x4_i:
return SemaRef.BuiltinConstantArgRange(TheCall, 1, -64, 56) ||
       SemaRef.BuiltinConstantArgMultiple(TheCall, 1, 8);
case Xtensa::BI__builtin_xtensa_ae_l16x4_ip:
return SemaRef.BuiltinConstantArgRange(TheCall, 2, 0, 56) ||
       SemaRef.BuiltinConstantArgMultiple(TheCall, 2, 8);
case Xtensa::BI__builtin_xtensa_ae_l32_i:
return SemaRef.BuiltinConstantArgRange(TheCall, 1, -32, 28) ||
       SemaRef.BuiltinConstantArgMultiple(TheCall, 1, 4);
case Xtensa::BI__builtin_xtensa_ae_l32_ip:
return SemaRef.BuiltinConstantArgRange(TheCall, 2, -32, 28) ||
       SemaRef.BuiltinConstantArgMultiple(TheCall, 2, 4);
case Xtensa::BI__builtin_xtensa_ae_l32f24_i:
return SemaRef.BuiltinConstantArgRange(TheCall, 1, -32, 28) ||
       SemaRef.BuiltinConstantArgMultiple(TheCall, 1, 4);
case Xtensa::BI__builtin_xtensa_ae_l32f24_ip:
return SemaRef.BuiltinConstantArgRange(TheCall, 2, -32, 28) ||
       SemaRef.BuiltinConstantArgMultiple(TheCall, 2, 4);
case Xtensa::BI__builtin_xtensa_ae_l32m_i:
return SemaRef.BuiltinConstantArgRange(TheCall, 1, -32, 28) ||
       SemaRef.BuiltinConstantArgMultiple(TheCall, 1, 4);
case Xtensa::BI__builtin_xtensa_ae_l32m_iu:
return SemaRef.BuiltinConstantArgRange(TheCall, 2, -32, 28) ||
       SemaRef.BuiltinConstantArgMultiple(TheCall, 2, 4);
case Xtensa::BI__builtin_xtensa_ae_l32x2_i:
return SemaRef.BuiltinConstantArgRange(TheCall, 1, -64, 56) ||
       SemaRef.BuiltinConstantArgMultiple(TheCall, 1, 8);
case Xtensa::BI__builtin_xtensa_ae_l32x2_ip:
return SemaRef.BuiltinConstantArgRange(TheCall, 2, 0, 56) ||
       SemaRef.BuiltinConstantArgMultiple(TheCall, 2, 8);
case Xtensa::BI__builtin_xtensa_ae_l32x2f24_i:
return SemaRef.BuiltinConstantArgRange(TheCall, 1, -64, 56) ||
       SemaRef.BuiltinConstantArgMultiple(TheCall, 1, 8);
case Xtensa::BI__builtin_xtensa_ae_l32x2f24_ip:
return SemaRef.BuiltinConstantArgRange(TheCall, 2, 0, 56) ||
       SemaRef.BuiltinConstantArgMultiple(TheCall, 2, 8);
case Xtensa::BI__builtin_xtensa_ae_l64_i:
return SemaRef.BuiltinConstantArgRange(TheCall, 1, -64, 56) ||
       SemaRef.BuiltinConstantArgMultiple(TheCall, 1, 8);
case Xtensa::BI__builtin_xtensa_ae_l64_ip:
return SemaRef.BuiltinConstantArgRange(TheCall, 2, -64, 56) ||
       SemaRef.BuiltinConstantArgMultiple(TheCall, 2, 8);
case Xtensa::BI__builtin_xtensa_ae_lalign64_i:
return SemaRef.BuiltinConstantArgRange(TheCall, 1, -64, 56) ||
       SemaRef.BuiltinConstantArgMultiple(TheCall, 1, 8);
case Xtensa::BI__builtin_xtensa_ae_lbi:
return SemaRef.BuiltinConstantArgRange(TheCall, 0, 1, 16);
case Xtensa::BI__builtin_xtensa_ae_lbki:
return SemaRef.BuiltinConstantArgRange(TheCall, 1, 1, 16);
case Xtensa::BI__builtin_xtensa_ae_lbsi:
return SemaRef.BuiltinConstantArgRange(TheCall, 0, 1, 16);
case Xtensa::BI__builtin_xtensa_ae_movi:
return SemaRef.BuiltinConstantArgRange(TheCall, 0, -16, 47);
case Xtensa::BI__builtin_xtensa_ae_pksr24:
return SemaRef.BuiltinConstantArgRange(TheCall, 2, 0, 3);
case Xtensa::BI__builtin_xtensa_ae_pksr32:
return SemaRef.BuiltinConstantArgRange(TheCall, 2, 0, 3);
case Xtensa::BI__builtin_xtensa_ae_s16_0_i:
return SemaRef.BuiltinConstantArgRange(TheCall, 2, -16, 14) ||
       SemaRef.BuiltinConstantArgMultiple(TheCall, 2, 2);
case Xtensa::BI__builtin_xtensa_ae_s16_0_ip:
return SemaRef.BuiltinConstantArgRange(TheCall, 2, -16, 14) ||
       SemaRef.BuiltinConstantArgMultiple(TheCall, 2, 2);
case Xtensa::BI__builtin_xtensa_ae_s16m_l_i:
return SemaRef.BuiltinConstantArgRange(TheCall, 2, -16, 14) ||
       SemaRef.BuiltinConstantArgMultiple(TheCall, 2, 2);
case Xtensa::BI__builtin_xtensa_ae_s16m_l_iu:
return SemaRef.BuiltinConstantArgRange(TheCall, 2, -16, 14) ||
       SemaRef.BuiltinConstantArgMultiple(TheCall, 2, 2);
case Xtensa::BI__builtin_xtensa_ae_s16x2m_i:
return SemaRef.BuiltinConstantArgRange(TheCall, 2, -32, 28) ||
       SemaRef.BuiltinConstantArgMultiple(TheCall, 2, 4);
case Xtensa::BI__builtin_xtensa_ae_s16x2m_iu:
return SemaRef.BuiltinConstantArgRange(TheCall, 2, -32, 28) ||
       SemaRef.BuiltinConstantArgMultiple(TheCall, 2, 4);
case Xtensa::BI__builtin_xtensa_ae_s16x4_i:
return SemaRef.BuiltinConstantArgRange(TheCall, 2, -64, 56) ||
       SemaRef.BuiltinConstantArgMultiple(TheCall, 2, 8);
case Xtensa::BI__builtin_xtensa_ae_s16x4_ip:
return SemaRef.BuiltinConstantArgRange(TheCall, 2, 0, 56) ||
       SemaRef.BuiltinConstantArgMultiple(TheCall, 2, 8);
case Xtensa::BI__builtin_xtensa_ae_s24ra64s_i:
return SemaRef.BuiltinConstantArgRange(TheCall, 2, -32, 28) ||
       SemaRef.BuiltinConstantArgMultiple(TheCall, 2, 4);
case Xtensa::BI__builtin_xtensa_ae_s24ra64s_ip:
return SemaRef.BuiltinConstantArgRange(TheCall, 2, -32, 28) ||
       SemaRef.BuiltinConstantArgMultiple(TheCall, 2, 4);
case Xtensa::BI__builtin_xtensa_ae_s32_l_i:
return SemaRef.BuiltinConstantArgRange(TheCall, 2, -32, 28) ||
       SemaRef.BuiltinConstantArgMultiple(TheCall, 2, 4);
case Xtensa::BI__builtin_xtensa_ae_s32_l_ip:
return SemaRef.BuiltinConstantArgRange(TheCall, 2, -32, 28) ||
       SemaRef.BuiltinConstantArgMultiple(TheCall, 2, 4);
case Xtensa::BI__builtin_xtensa_ae_s32f24_l_i:
return SemaRef.BuiltinConstantArgRange(TheCall, 2, -32, 28) ||
       SemaRef.BuiltinConstantArgMultiple(TheCall, 2, 4);
case Xtensa::BI__builtin_xtensa_ae_s32f24_l_ip:
return SemaRef.BuiltinConstantArgRange(TheCall, 2, -32, 28) ||
       SemaRef.BuiltinConstantArgMultiple(TheCall, 2, 4);
case Xtensa::BI__builtin_xtensa_ae_s32m_i:
return SemaRef.BuiltinConstantArgRange(TheCall, 2, -32, 28) ||
       SemaRef.BuiltinConstantArgMultiple(TheCall, 2, 4);
case Xtensa::BI__builtin_xtensa_ae_s32m_iu:
return SemaRef.BuiltinConstantArgRange(TheCall, 2, -32, 28) ||
       SemaRef.BuiltinConstantArgMultiple(TheCall, 2, 4);
case Xtensa::BI__builtin_xtensa_ae_s32ra64s_i:
return SemaRef.BuiltinConstantArgRange(TheCall, 2, -32, 28) ||
       SemaRef.BuiltinConstantArgMultiple(TheCall, 2, 4);
case Xtensa::BI__builtin_xtensa_ae_s32ra64s_ip:
return SemaRef.BuiltinConstantArgRange(TheCall, 2, -32, 28) ||
       SemaRef.BuiltinConstantArgMultiple(TheCall, 2, 4);
case Xtensa::BI__builtin_xtensa_ae_s32x2_i:
return SemaRef.BuiltinConstantArgRange(TheCall, 2, -64, 56) ||
       SemaRef.BuiltinConstantArgMultiple(TheCall, 2, 8);
case Xtensa::BI__builtin_xtensa_ae_s32x2_ip:
return SemaRef.BuiltinConstantArgRange(TheCall, 2, 0, 56) ||
       SemaRef.BuiltinConstantArgMultiple(TheCall, 2, 8);
case Xtensa::BI__builtin_xtensa_ae_s32x2f24_i:
return SemaRef.BuiltinConstantArgRange(TheCall, 2, -64, 56) ||
       SemaRef.BuiltinConstantArgMultiple(TheCall, 2, 8);
case Xtensa::BI__builtin_xtensa_ae_s32x2f24_ip:
return SemaRef.BuiltinConstantArgRange(TheCall, 2, 0, 56) ||
       SemaRef.BuiltinConstantArgMultiple(TheCall, 2, 8);
case Xtensa::BI__builtin_xtensa_ae_s64_i:
return SemaRef.BuiltinConstantArgRange(TheCall, 2, -64, 56) ||
       SemaRef.BuiltinConstantArgMultiple(TheCall, 2, 8);
case Xtensa::BI__builtin_xtensa_ae_s64_ip:
return SemaRef.BuiltinConstantArgRange(TheCall, 2, -64, 56) ||
       SemaRef.BuiltinConstantArgMultiple(TheCall, 2, 8);
case Xtensa::BI__builtin_xtensa_ae_salign64_i:
return SemaRef.BuiltinConstantArgRange(TheCall, 2, -64, 56) ||
       SemaRef.BuiltinConstantArgMultiple(TheCall, 2, 8);
case Xtensa::BI__builtin_xtensa_ae_sbi:
return SemaRef.BuiltinConstantArgRange(TheCall, 2, 1, 16);
case Xtensa::BI__builtin_xtensa_ae_sbi_ic:
return SemaRef.BuiltinConstantArgRange(TheCall, 2, 1, 16);
case Xtensa::BI__builtin_xtensa_ae_sbi_ip:
return SemaRef.BuiltinConstantArgRange(TheCall, 2, 1, 16);
case Xtensa::BI__builtin_xtensa_ae_sel16i:
return SemaRef.BuiltinConstantArgRange(TheCall, 2, 0, 15);
case Xtensa::BI__builtin_xtensa_ae_sel16i_n:
return SemaRef.BuiltinConstantArgRange(TheCall, 2, 0, 3);
case Xtensa::BI__builtin_xtensa_ae_sext32:
return SemaRef.BuiltinConstantArgRange(TheCall, 1, 7, 22);
case Xtensa::BI__builtin_xtensa_ae_slai16s:
return SemaRef.BuiltinConstantArgRange(TheCall, 1, 0, 15);
case Xtensa::BI__builtin_xtensa_ae_slai24:
return SemaRef.BuiltinConstantArgRange(TheCall, 1, 0, 31);
case Xtensa::BI__builtin_xtensa_ae_slai24s:
return SemaRef.BuiltinConstantArgRange(TheCall, 1, 0, 31);
case Xtensa::BI__builtin_xtensa_ae_slai32:
return SemaRef.BuiltinConstantArgRange(TheCall, 1, 0, 31);
case Xtensa::BI__builtin_xtensa_ae_slai32s:
return SemaRef.BuiltinConstantArgRange(TheCall, 1, 0, 31);
case Xtensa::BI__builtin_xtensa_ae_slai64:
return SemaRef.BuiltinConstantArgRange(TheCall, 1, 0, 63);
case Xtensa::BI__builtin_xtensa_ae_slai64s:
return SemaRef.BuiltinConstantArgRange(TheCall, 1, 0, 63);
case Xtensa::BI__builtin_xtensa_ae_slaisq56s:
return SemaRef.BuiltinConstantArgRange(TheCall, 1, 0, 63);
case Xtensa::BI__builtin_xtensa_ae_srai16:
return SemaRef.BuiltinConstantArgRange(TheCall, 1, 0, 15);
case Xtensa::BI__builtin_xtensa_ae_srai16r:
return SemaRef.BuiltinConstantArgRange(TheCall, 1, 0, 15);
case Xtensa::BI__builtin_xtensa_ae_srai24:
return SemaRef.BuiltinConstantArgRange(TheCall, 1, 0, 31);
case Xtensa::BI__builtin_xtensa_ae_srai32:
return SemaRef.BuiltinConstantArgRange(TheCall, 1, 0, 31);
case Xtensa::BI__builtin_xtensa_ae_srai32r:
return SemaRef.BuiltinConstantArgRange(TheCall, 1, 0, 31);
case Xtensa::BI__builtin_xtensa_ae_srai64:
return SemaRef.BuiltinConstantArgRange(TheCall, 1, 0, 63);
case Xtensa::BI__builtin_xtensa_ae_srli24:
return SemaRef.BuiltinConstantArgRange(TheCall, 1, 0, 31);
case Xtensa::BI__builtin_xtensa_ae_srli32:
return SemaRef.BuiltinConstantArgRange(TheCall, 1, 0, 31);
case Xtensa::BI__builtin_xtensa_ae_srli64:
return SemaRef.BuiltinConstantArgRange(TheCall, 1, 0, 63);
case Xtensa::BI__builtin_xtensa_ae_trunci32f64s_l:
return SemaRef.BuiltinConstantArgRange(TheCall, 2, 0, 15);
case Xtensa::BI__builtin_xtensa_ae_trunci32x2f64s:
return SemaRef.BuiltinConstantArgRange(TheCall, 2, 0, 15);
