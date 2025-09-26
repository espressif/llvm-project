//===-- XtensaBuiltins.inc - Clang intrinsic database for Xtensa arch ----*- C++
//-*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

case Xtensa::BI__builtin_xtensa_ae_abs16s:
return {Intrinsic::xtensa_ae_abs16s, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_abs24s:
return {Intrinsic::xtensa_ae_abs24s, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_abs32:
return {Intrinsic::xtensa_ae_abs32, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_abs32s:
return {Intrinsic::xtensa_ae_abs32s, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_abs64:
return {Intrinsic::xtensa_ae_abs64, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_abs64s:
return {Intrinsic::xtensa_ae_abs64s, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_add16:
return {Intrinsic::xtensa_ae_add16, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_add16s:
return {Intrinsic::xtensa_ae_add16s, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_add24s:
return {Intrinsic::xtensa_ae_add24s, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_add32:
return {Intrinsic::xtensa_ae_add32, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_add32_hl_lh:
return {Intrinsic::xtensa_ae_add32_hl_lh, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_add32s:
return {Intrinsic::xtensa_ae_add32s, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_add64:
return {Intrinsic::xtensa_ae_add64, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_add64s:
return {Intrinsic::xtensa_ae_add64s, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_addbrba32:
return {Intrinsic::xtensa_ae_addbrba32, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_addsub32:
return {Intrinsic::xtensa_ae_addsub32, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_addsub32s:
return {Intrinsic::xtensa_ae_addsub32s, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_and:
return {Intrinsic::xtensa_ae_and, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_cvt32x2f16_10:
return {Intrinsic::xtensa_ae_cvt32x2f16_10, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_cvt32x2f16_32:
return {Intrinsic::xtensa_ae_cvt32x2f16_32, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_cvt48a32:
return {Intrinsic::xtensa_ae_cvt48a32, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_cvt64a32:
return {Intrinsic::xtensa_ae_cvt64a32, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_cvt64f32_h:
return {Intrinsic::xtensa_ae_cvt64f32_h, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_cvta32f24s_h:
return {Intrinsic::xtensa_ae_cvta32f24s_h, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_cvta32f24s_l:
return {Intrinsic::xtensa_ae_cvta32f24s_l, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_cvtq56a32s:
return {Intrinsic::xtensa_ae_cvtq56a32s, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_cvtq56p32s_h:
return {Intrinsic::xtensa_ae_cvtq56p32s_h, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_cvtq56p32s_l:
return {Intrinsic::xtensa_ae_cvtq56p32s_l, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_db:
return {Intrinsic::xtensa_ae_db, 0, 0x20100};
case Xtensa::BI__builtin_xtensa_ae_db_ic:
return {Intrinsic::xtensa_ae_db_ic, 0, 0x20100};
case Xtensa::BI__builtin_xtensa_ae_db_ip:
return {Intrinsic::xtensa_ae_db_ip, 0, 0x20100};
case Xtensa::BI__builtin_xtensa_ae_dbi:
return {Intrinsic::xtensa_ae_dbi, 0, 0x20100};
case Xtensa::BI__builtin_xtensa_ae_dbi_ic:
return {Intrinsic::xtensa_ae_dbi_ic, 0, 0x20100};
case Xtensa::BI__builtin_xtensa_ae_dbi_ip:
return {Intrinsic::xtensa_ae_dbi_ip, 0, 0x20100};
case Xtensa::BI__builtin_xtensa_ae_div64d32_h:
return {Intrinsic::xtensa_ae_div64d32_h, 0, 0x20100};
case Xtensa::BI__builtin_xtensa_ae_div64d32_l:
return {Intrinsic::xtensa_ae_div64d32_l, 0, 0x20100};
case Xtensa::BI__builtin_xtensa_ae_eq16:
return {Intrinsic::xtensa_ae_eq16, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_eq32:
return {Intrinsic::xtensa_ae_eq32, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_eq64:
return {Intrinsic::xtensa_ae_eq64, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_l16_i:
return {Intrinsic::xtensa_ae_l16_i, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_l16_ip:
return {Intrinsic::xtensa_ae_l16_ip, 0, 0x40201};
case Xtensa::BI__builtin_xtensa_ae_l16_x:
return {Intrinsic::xtensa_ae_l16_x, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_l16_xc:
return {Intrinsic::xtensa_ae_l16_xc, 0, 0x40201};
case Xtensa::BI__builtin_xtensa_ae_l16_xp:
return {Intrinsic::xtensa_ae_l16_xp, 0, 0x40201};
case Xtensa::BI__builtin_xtensa_ae_l16m_i:
return {Intrinsic::xtensa_ae_l16m_i, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_l16m_iu:
return {Intrinsic::xtensa_ae_l16m_iu, 0, 0x40201};
case Xtensa::BI__builtin_xtensa_ae_l16m_x:
return {Intrinsic::xtensa_ae_l16m_x, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_l16m_xc:
return {Intrinsic::xtensa_ae_l16m_xc, 0, 0x40201};
case Xtensa::BI__builtin_xtensa_ae_l16m_xu:
return {Intrinsic::xtensa_ae_l16m_xu, 0, 0x40201};
case Xtensa::BI__builtin_xtensa_ae_l16x2m_i:
return {Intrinsic::xtensa_ae_l16x2m_i, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_l16x2m_iu:
return {Intrinsic::xtensa_ae_l16x2m_iu, 0, 0x40201};
case Xtensa::BI__builtin_xtensa_ae_l16x2m_x:
return {Intrinsic::xtensa_ae_l16x2m_x, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_l16x2m_xc:
return {Intrinsic::xtensa_ae_l16x2m_xc, 0, 0x40201};
case Xtensa::BI__builtin_xtensa_ae_l16x2m_xu:
return {Intrinsic::xtensa_ae_l16x2m_xu, 0, 0x40201};
case Xtensa::BI__builtin_xtensa_ae_l16x4_i:
return {Intrinsic::xtensa_ae_l16x4_i, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_l16x4_ip:
return {Intrinsic::xtensa_ae_l16x4_ip, 0, 0x40201};
case Xtensa::BI__builtin_xtensa_ae_l16x4_ric:
return {Intrinsic::xtensa_ae_l16x4_ric, 0, 0x201};
case Xtensa::BI__builtin_xtensa_ae_l16x4_rip:
return {Intrinsic::xtensa_ae_l16x4_rip, 0, 0x201};
case Xtensa::BI__builtin_xtensa_ae_l16x4_x:
return {Intrinsic::xtensa_ae_l16x4_x, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_l16x4_xc:
return {Intrinsic::xtensa_ae_l16x4_xc, 0, 0x40201};
case Xtensa::BI__builtin_xtensa_ae_l16x4_xp:
return {Intrinsic::xtensa_ae_l16x4_xp, 0, 0x40201};
case Xtensa::BI__builtin_xtensa_ae_l32_i:
return {Intrinsic::xtensa_ae_l32_i, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_l32_ip:
return {Intrinsic::xtensa_ae_l32_ip, 0, 0x40201};
case Xtensa::BI__builtin_xtensa_ae_l32_x:
return {Intrinsic::xtensa_ae_l32_x, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_l32_xc:
return {Intrinsic::xtensa_ae_l32_xc, 0, 0x40201};
case Xtensa::BI__builtin_xtensa_ae_l32_xp:
return {Intrinsic::xtensa_ae_l32_xp, 0, 0x40201};
case Xtensa::BI__builtin_xtensa_ae_l32f24_i:
return {Intrinsic::xtensa_ae_l32f24_i, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_l32f24_ip:
return {Intrinsic::xtensa_ae_l32f24_ip, 0, 0x40201};
case Xtensa::BI__builtin_xtensa_ae_l32f24_x:
return {Intrinsic::xtensa_ae_l32f24_x, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_l32f24_xc:
return {Intrinsic::xtensa_ae_l32f24_xc, 0, 0x40201};
case Xtensa::BI__builtin_xtensa_ae_l32f24_xp:
return {Intrinsic::xtensa_ae_l32f24_xp, 0, 0x40201};
case Xtensa::BI__builtin_xtensa_ae_l32m_i:
return {Intrinsic::xtensa_ae_l32m_i, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_l32m_iu:
return {Intrinsic::xtensa_ae_l32m_iu, 0, 0x40201};
case Xtensa::BI__builtin_xtensa_ae_l32m_x:
return {Intrinsic::xtensa_ae_l32m_x, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_l32m_xc:
return {Intrinsic::xtensa_ae_l32m_xc, 0, 0x40201};
case Xtensa::BI__builtin_xtensa_ae_l32m_xu:
return {Intrinsic::xtensa_ae_l32m_xu, 0, 0x40201};
case Xtensa::BI__builtin_xtensa_ae_l32x2_i:
return {Intrinsic::xtensa_ae_l32x2_i, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_l32x2_ip:
return {Intrinsic::xtensa_ae_l32x2_ip, 0, 0x40201};
case Xtensa::BI__builtin_xtensa_ae_l32x2_ric:
return {Intrinsic::xtensa_ae_l32x2_ric, 0, 0x201};
case Xtensa::BI__builtin_xtensa_ae_l32x2_rip:
return {Intrinsic::xtensa_ae_l32x2_rip, 0, 0x201};
case Xtensa::BI__builtin_xtensa_ae_l32x2_x:
return {Intrinsic::xtensa_ae_l32x2_x, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_l32x2_xc:
return {Intrinsic::xtensa_ae_l32x2_xc, 0, 0x40201};
case Xtensa::BI__builtin_xtensa_ae_l32x2_xp:
return {Intrinsic::xtensa_ae_l32x2_xp, 0, 0x40201};
case Xtensa::BI__builtin_xtensa_ae_l32x2f24_i:
return {Intrinsic::xtensa_ae_l32x2f24_i, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_l32x2f24_ip:
return {Intrinsic::xtensa_ae_l32x2f24_ip, 0, 0x40201};
case Xtensa::BI__builtin_xtensa_ae_l32x2f24_ric:
return {Intrinsic::xtensa_ae_l32x2f24_ric, 0, 0x201};
case Xtensa::BI__builtin_xtensa_ae_l32x2f24_rip:
return {Intrinsic::xtensa_ae_l32x2f24_rip, 0, 0x201};
case Xtensa::BI__builtin_xtensa_ae_l32x2f24_x:
return {Intrinsic::xtensa_ae_l32x2f24_x, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_l32x2f24_xc:
return {Intrinsic::xtensa_ae_l32x2f24_xc, 0, 0x40201};
case Xtensa::BI__builtin_xtensa_ae_l32x2f24_xp:
return {Intrinsic::xtensa_ae_l32x2f24_xp, 0, 0x40201};
case Xtensa::BI__builtin_xtensa_ae_l64_i:
return {Intrinsic::xtensa_ae_l64_i, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_l64_ip:
return {Intrinsic::xtensa_ae_l64_ip, 0, 0x40201};
case Xtensa::BI__builtin_xtensa_ae_l64_x:
return {Intrinsic::xtensa_ae_l64_x, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_l64_xc:
return {Intrinsic::xtensa_ae_l64_xc, 0, 0x40201};
case Xtensa::BI__builtin_xtensa_ae_l64_xp:
return {Intrinsic::xtensa_ae_l64_xp, 0, 0x40201};
case Xtensa::BI__builtin_xtensa_ae_la16x4_ic:
return {Intrinsic::xtensa_ae_la16x4_ic, 0, 0x601};
case Xtensa::BI__builtin_xtensa_ae_la16x4_ip:
return {Intrinsic::xtensa_ae_la16x4_ip, 0, 0x601};
case Xtensa::BI__builtin_xtensa_ae_la16x4_ric:
return {Intrinsic::xtensa_ae_la16x4_ric, 0, 0x601};
case Xtensa::BI__builtin_xtensa_ae_la16x4_rip:
return {Intrinsic::xtensa_ae_la16x4_rip, 0, 0x601};
case Xtensa::BI__builtin_xtensa_ae_la16x4neg_pc:
return {Intrinsic::xtensa_ae_la16x4neg_pc, 0, 0x201};
case Xtensa::BI__builtin_xtensa_ae_la16x4pos_pc:
return {Intrinsic::xtensa_ae_la16x4pos_pc, 0, 0x201};
case Xtensa::BI__builtin_xtensa_ae_la24_ic:
return {Intrinsic::xtensa_ae_la24_ic, 0, 0x601};
case Xtensa::BI__builtin_xtensa_ae_la24_ip:
return {Intrinsic::xtensa_ae_la24_ip, 0, 0x601};
case Xtensa::BI__builtin_xtensa_ae_la24_ric:
return {Intrinsic::xtensa_ae_la24_ric, 0, 0x601};
case Xtensa::BI__builtin_xtensa_ae_la24_rip:
return {Intrinsic::xtensa_ae_la24_rip, 0, 0x601};
case Xtensa::BI__builtin_xtensa_ae_la24neg_pc:
return {Intrinsic::xtensa_ae_la24neg_pc, 0, 0x201};
case Xtensa::BI__builtin_xtensa_ae_la24pos_pc:
return {Intrinsic::xtensa_ae_la24pos_pc, 0, 0x201};
case Xtensa::BI__builtin_xtensa_ae_la24x2_ic:
return {Intrinsic::xtensa_ae_la24x2_ic, 0, 0x601};
case Xtensa::BI__builtin_xtensa_ae_la24x2_ip:
return {Intrinsic::xtensa_ae_la24x2_ip, 0, 0x601};
case Xtensa::BI__builtin_xtensa_ae_la24x2_ric:
return {Intrinsic::xtensa_ae_la24x2_ric, 0, 0x601};
case Xtensa::BI__builtin_xtensa_ae_la24x2_rip:
return {Intrinsic::xtensa_ae_la24x2_rip, 0, 0x601};
case Xtensa::BI__builtin_xtensa_ae_la24x2neg_pc:
return {Intrinsic::xtensa_ae_la24x2neg_pc, 0, 0x201};
case Xtensa::BI__builtin_xtensa_ae_la24x2pos_pc:
return {Intrinsic::xtensa_ae_la24x2pos_pc, 0, 0x201};
case Xtensa::BI__builtin_xtensa_ae_la32x2_ic:
return {Intrinsic::xtensa_ae_la32x2_ic, 0, 0x601};
case Xtensa::BI__builtin_xtensa_ae_la32x2_ip:
return {Intrinsic::xtensa_ae_la32x2_ip, 0, 0x601};
case Xtensa::BI__builtin_xtensa_ae_la32x2_ric:
return {Intrinsic::xtensa_ae_la32x2_ric, 0, 0x601};
case Xtensa::BI__builtin_xtensa_ae_la32x2_rip:
return {Intrinsic::xtensa_ae_la32x2_rip, 0, 0x601};
case Xtensa::BI__builtin_xtensa_ae_la32x2f24_ic:
return {Intrinsic::xtensa_ae_la32x2f24_ic, 0, 0x601};
case Xtensa::BI__builtin_xtensa_ae_la32x2f24_ip:
return {Intrinsic::xtensa_ae_la32x2f24_ip, 0, 0x601};
case Xtensa::BI__builtin_xtensa_ae_la32x2f24_ric:
return {Intrinsic::xtensa_ae_la32x2f24_ric, 0, 0x601};
case Xtensa::BI__builtin_xtensa_ae_la32x2f24_rip:
return {Intrinsic::xtensa_ae_la32x2f24_rip, 0, 0x601};
case Xtensa::BI__builtin_xtensa_ae_la32x2neg_pc:
return {Intrinsic::xtensa_ae_la32x2neg_pc, 0, 0x201};
case Xtensa::BI__builtin_xtensa_ae_la32x2pos_pc:
return {Intrinsic::xtensa_ae_la32x2pos_pc, 0, 0x201};
case Xtensa::BI__builtin_xtensa_ae_la64_pp:
return {Intrinsic::xtensa_ae_la64_pp, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_lalign64_i:
return {Intrinsic::xtensa_ae_lalign64_i, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_lb:
return {Intrinsic::xtensa_ae_lb, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_lbi:
return {Intrinsic::xtensa_ae_lbi, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_lbk:
return {Intrinsic::xtensa_ae_lbk, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_lbki:
return {Intrinsic::xtensa_ae_lbki, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_lbs:
return {Intrinsic::xtensa_ae_lbs, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_lbsi:
return {Intrinsic::xtensa_ae_lbsi, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_le16:
return {Intrinsic::xtensa_ae_le16, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_le32:
return {Intrinsic::xtensa_ae_le32, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_le64:
return {Intrinsic::xtensa_ae_le64, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_lt16:
return {Intrinsic::xtensa_ae_lt16, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_lt32:
return {Intrinsic::xtensa_ae_lt32, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_lt64:
return {Intrinsic::xtensa_ae_lt64, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_max32:
return {Intrinsic::xtensa_ae_max32, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_max64:
return {Intrinsic::xtensa_ae_max64, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_maxabs32s:
return {Intrinsic::xtensa_ae_maxabs32s, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_maxabs64s:
return {Intrinsic::xtensa_ae_maxabs64s, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_min32:
return {Intrinsic::xtensa_ae_min32, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_min64:
return {Intrinsic::xtensa_ae_min64, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_minabs32s:
return {Intrinsic::xtensa_ae_minabs32s, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_minabs64s:
return {Intrinsic::xtensa_ae_minabs64s, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mov:
return {Intrinsic::xtensa_ae_mov, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_movad16_0:
return {Intrinsic::xtensa_ae_movad16_0, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_movad16_1:
return {Intrinsic::xtensa_ae_movad16_1, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_movad16_2:
return {Intrinsic::xtensa_ae_movad16_2, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_movad16_3:
return {Intrinsic::xtensa_ae_movad16_3, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_movad32_h:
return {Intrinsic::xtensa_ae_movad32_h, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_movad32_l:
return {Intrinsic::xtensa_ae_movad32_l, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_movalign:
return {Intrinsic::xtensa_ae_movalign, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_movda16:
return {Intrinsic::xtensa_ae_movda16, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_movda16x2:
return {Intrinsic::xtensa_ae_movda16x2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_movda32:
return {Intrinsic::xtensa_ae_movda32, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_movda32x2:
return {Intrinsic::xtensa_ae_movda32x2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_movf16x4:
return {Intrinsic::xtensa_ae_movf16x4, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_movf32x2:
return {Intrinsic::xtensa_ae_movf32x2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_movf64:
return {Intrinsic::xtensa_ae_movf64, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_movi:
return {Intrinsic::xtensa_ae_movi, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_movt16x4:
return {Intrinsic::xtensa_ae_movt16x4, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_movt32x2:
return {Intrinsic::xtensa_ae_movt32x2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_movt64:
return {Intrinsic::xtensa_ae_movt64, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mul16x4:
return {Intrinsic::xtensa_ae_mul16x4, 0, 0xc0003};
case Xtensa::BI__builtin_xtensa_ae_mul32_hh:
return {Intrinsic::xtensa_ae_mul32_hh, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mul32_lh:
return {Intrinsic::xtensa_ae_mul32_lh, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mul32_ll:
return {Intrinsic::xtensa_ae_mul32_ll, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mul32_ll_s2:
return {Intrinsic::xtensa_ae_mul32_ll_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mul32u_ll:
return {Intrinsic::xtensa_ae_mul32u_ll, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mul32x16_h0:
return {Intrinsic::xtensa_ae_mul32x16_h0, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mul32x16_h0_s2:
return {Intrinsic::xtensa_ae_mul32x16_h0_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mul32x16_h1:
return {Intrinsic::xtensa_ae_mul32x16_h1, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mul32x16_h1_s2:
return {Intrinsic::xtensa_ae_mul32x16_h1_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mul32x16_h2:
return {Intrinsic::xtensa_ae_mul32x16_h2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mul32x16_h2_s2:
return {Intrinsic::xtensa_ae_mul32x16_h2_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mul32x16_h3:
return {Intrinsic::xtensa_ae_mul32x16_h3, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mul32x16_h3_s2:
return {Intrinsic::xtensa_ae_mul32x16_h3_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mul32x16_l0:
return {Intrinsic::xtensa_ae_mul32x16_l0, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mul32x16_l0_s2:
return {Intrinsic::xtensa_ae_mul32x16_l0_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mul32x16_l1:
return {Intrinsic::xtensa_ae_mul32x16_l1, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mul32x16_l1_s2:
return {Intrinsic::xtensa_ae_mul32x16_l1_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mul32x16_l2:
return {Intrinsic::xtensa_ae_mul32x16_l2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mul32x16_l2_s2:
return {Intrinsic::xtensa_ae_mul32x16_l2_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mul32x16_l3:
return {Intrinsic::xtensa_ae_mul32x16_l3, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mul32x16_l3_s2:
return {Intrinsic::xtensa_ae_mul32x16_l3_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mula16x4:
return {Intrinsic::xtensa_ae_mula16x4, 0, 0xc0300};
case Xtensa::BI__builtin_xtensa_ae_mula32_hh:
return {Intrinsic::xtensa_ae_mula32_hh, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mula32_lh:
return {Intrinsic::xtensa_ae_mula32_lh, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mula32_ll:
return {Intrinsic::xtensa_ae_mula32_ll, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mula32_ll_s2:
return {Intrinsic::xtensa_ae_mula32_ll_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mula32u_ll:
return {Intrinsic::xtensa_ae_mula32u_ll, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mula32x16_h0:
return {Intrinsic::xtensa_ae_mula32x16_h0, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mula32x16_h0_s2:
return {Intrinsic::xtensa_ae_mula32x16_h0_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mula32x16_h1:
return {Intrinsic::xtensa_ae_mula32x16_h1, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mula32x16_h1_s2:
return {Intrinsic::xtensa_ae_mula32x16_h1_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mula32x16_h2:
return {Intrinsic::xtensa_ae_mula32x16_h2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mula32x16_h2_s2:
return {Intrinsic::xtensa_ae_mula32x16_h2_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mula32x16_h3:
return {Intrinsic::xtensa_ae_mula32x16_h3, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mula32x16_h3_s2:
return {Intrinsic::xtensa_ae_mula32x16_h3_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mula32x16_l0:
return {Intrinsic::xtensa_ae_mula32x16_l0, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mula32x16_l0_s2:
return {Intrinsic::xtensa_ae_mula32x16_l0_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mula32x16_l1:
return {Intrinsic::xtensa_ae_mula32x16_l1, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mula32x16_l1_s2:
return {Intrinsic::xtensa_ae_mula32x16_l1_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mula32x16_l2:
return {Intrinsic::xtensa_ae_mula32x16_l2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mula32x16_l2_s2:
return {Intrinsic::xtensa_ae_mula32x16_l2_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mula32x16_l3:
return {Intrinsic::xtensa_ae_mula32x16_l3, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mula32x16_l3_s2:
return {Intrinsic::xtensa_ae_mula32x16_l3_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaad24_hh_ll:
return {Intrinsic::xtensa_ae_mulaad24_hh_ll, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaad24_hh_ll_s2:
return {Intrinsic::xtensa_ae_mulaad24_hh_ll_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaad24_hl_lh:
return {Intrinsic::xtensa_ae_mulaad24_hl_lh, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaad24_hl_lh_s2:
return {Intrinsic::xtensa_ae_mulaad24_hl_lh_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaad32x16_h0_l1:
return {Intrinsic::xtensa_ae_mulaad32x16_h0_l1, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaad32x16_h0_l1_s2:
return {Intrinsic::xtensa_ae_mulaad32x16_h0_l1_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaad32x16_h1_l0:
return {Intrinsic::xtensa_ae_mulaad32x16_h1_l0, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaad32x16_h1_l0_s2:
return {Intrinsic::xtensa_ae_mulaad32x16_h1_l0_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaad32x16_h2_l3:
return {Intrinsic::xtensa_ae_mulaad32x16_h2_l3, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaad32x16_h2_l3_s2:
return {Intrinsic::xtensa_ae_mulaad32x16_h2_l3_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaad32x16_h3_l2:
return {Intrinsic::xtensa_ae_mulaad32x16_h3_l2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaad32x16_h3_l2_s2:
return {Intrinsic::xtensa_ae_mulaad32x16_h3_l2_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaafd16ss_11_00:
return {Intrinsic::xtensa_ae_mulaafd16ss_11_00, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaafd16ss_11_00_s2:
return {Intrinsic::xtensa_ae_mulaafd16ss_11_00_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaafd16ss_13_02:
return {Intrinsic::xtensa_ae_mulaafd16ss_13_02, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaafd16ss_13_02_s2:
return {Intrinsic::xtensa_ae_mulaafd16ss_13_02_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaafd16ss_33_22:
return {Intrinsic::xtensa_ae_mulaafd16ss_33_22, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaafd16ss_33_22_s2:
return {Intrinsic::xtensa_ae_mulaafd16ss_33_22_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaafd24_hh_ll:
return {Intrinsic::xtensa_ae_mulaafd24_hh_ll, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaafd24_hh_ll_s2:
return {Intrinsic::xtensa_ae_mulaafd24_hh_ll_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaafd24_hl_lh:
return {Intrinsic::xtensa_ae_mulaafd24_hl_lh, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaafd24_hl_lh_s2:
return {Intrinsic::xtensa_ae_mulaafd24_hl_lh_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaafd32x16_h0_l1:
return {Intrinsic::xtensa_ae_mulaafd32x16_h0_l1, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaafd32x16_h0_l1_s2:
return {Intrinsic::xtensa_ae_mulaafd32x16_h0_l1_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaafd32x16_h1_l0:
return {Intrinsic::xtensa_ae_mulaafd32x16_h1_l0, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaafd32x16_h1_l0_s2:
return {Intrinsic::xtensa_ae_mulaafd32x16_h1_l0_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaafd32x16_h2_l3:
return {Intrinsic::xtensa_ae_mulaafd32x16_h2_l3, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaafd32x16_h2_l3_s2:
return {Intrinsic::xtensa_ae_mulaafd32x16_h2_l3_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaafd32x16_h3_l2:
return {Intrinsic::xtensa_ae_mulaafd32x16_h3_l2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaafd32x16_h3_l2_s2:
return {Intrinsic::xtensa_ae_mulaafd32x16_h3_l2_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulac24:
return {Intrinsic::xtensa_ae_mulac24, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulac32x16_h:
return {Intrinsic::xtensa_ae_mulac32x16_h, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulac32x16_l:
return {Intrinsic::xtensa_ae_mulac32x16_l, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaf16ss_00:
return {Intrinsic::xtensa_ae_mulaf16ss_00, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaf16ss_00_s2:
return {Intrinsic::xtensa_ae_mulaf16ss_00_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaf16ss_10:
return {Intrinsic::xtensa_ae_mulaf16ss_10, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaf16ss_11:
return {Intrinsic::xtensa_ae_mulaf16ss_11, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaf16ss_20:
return {Intrinsic::xtensa_ae_mulaf16ss_20, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaf16ss_21:
return {Intrinsic::xtensa_ae_mulaf16ss_21, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaf16ss_22:
return {Intrinsic::xtensa_ae_mulaf16ss_22, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaf16ss_30:
return {Intrinsic::xtensa_ae_mulaf16ss_30, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaf16ss_31:
return {Intrinsic::xtensa_ae_mulaf16ss_31, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaf16ss_32:
return {Intrinsic::xtensa_ae_mulaf16ss_32, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaf16ss_33:
return {Intrinsic::xtensa_ae_mulaf16ss_33, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaf16x4ss:
return {Intrinsic::xtensa_ae_mulaf16x4ss, 0, 0xc0300};
case Xtensa::BI__builtin_xtensa_ae_mulaf32r_hh:
return {Intrinsic::xtensa_ae_mulaf32r_hh, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaf32r_lh:
return {Intrinsic::xtensa_ae_mulaf32r_lh, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaf32r_ll:
return {Intrinsic::xtensa_ae_mulaf32r_ll, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaf32r_ll_s2:
return {Intrinsic::xtensa_ae_mulaf32r_ll_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaf32s_hh:
return {Intrinsic::xtensa_ae_mulaf32s_hh, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaf32s_lh:
return {Intrinsic::xtensa_ae_mulaf32s_lh, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaf32s_ll:
return {Intrinsic::xtensa_ae_mulaf32s_ll, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaf32s_ll_s2:
return {Intrinsic::xtensa_ae_mulaf32s_ll_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaf32x16_h0:
return {Intrinsic::xtensa_ae_mulaf32x16_h0, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaf32x16_h0_s2:
return {Intrinsic::xtensa_ae_mulaf32x16_h0_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaf32x16_h1:
return {Intrinsic::xtensa_ae_mulaf32x16_h1, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaf32x16_h1_s2:
return {Intrinsic::xtensa_ae_mulaf32x16_h1_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaf32x16_h2:
return {Intrinsic::xtensa_ae_mulaf32x16_h2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaf32x16_h2_s2:
return {Intrinsic::xtensa_ae_mulaf32x16_h2_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaf32x16_h3:
return {Intrinsic::xtensa_ae_mulaf32x16_h3, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaf32x16_h3_s2:
return {Intrinsic::xtensa_ae_mulaf32x16_h3_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaf32x16_l0:
return {Intrinsic::xtensa_ae_mulaf32x16_l0, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaf32x16_l0_s2:
return {Intrinsic::xtensa_ae_mulaf32x16_l0_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaf32x16_l1:
return {Intrinsic::xtensa_ae_mulaf32x16_l1, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaf32x16_l1_s2:
return {Intrinsic::xtensa_ae_mulaf32x16_l1_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaf32x16_l2:
return {Intrinsic::xtensa_ae_mulaf32x16_l2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaf32x16_l2_s2:
return {Intrinsic::xtensa_ae_mulaf32x16_l2_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaf32x16_l3:
return {Intrinsic::xtensa_ae_mulaf32x16_l3, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaf32x16_l3_s2:
return {Intrinsic::xtensa_ae_mulaf32x16_l3_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaf48q32sp16s_l:
return {Intrinsic::xtensa_ae_mulaf48q32sp16s_l, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaf48q32sp16s_l_s2:
return {Intrinsic::xtensa_ae_mulaf48q32sp16s_l_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaf48q32sp16u_l:
return {Intrinsic::xtensa_ae_mulaf48q32sp16u_l, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaf48q32sp16u_l_s2:
return {Intrinsic::xtensa_ae_mulaf48q32sp16u_l_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulafc24ra:
return {Intrinsic::xtensa_ae_mulafc24ra, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulafc32x16ras_h:
return {Intrinsic::xtensa_ae_mulafc32x16ras_h, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulafc32x16ras_l:
return {Intrinsic::xtensa_ae_mulafc32x16ras_l, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulafd24x2_fir_h:
return {Intrinsic::xtensa_ae_mulafd24x2_fir_h, 0, 0x1c0300};
case Xtensa::BI__builtin_xtensa_ae_mulafd24x2_fir_l:
return {Intrinsic::xtensa_ae_mulafd24x2_fir_l, 0, 0x1c0300};
case Xtensa::BI__builtin_xtensa_ae_mulafd32x16x2_fir_hh:
return {Intrinsic::xtensa_ae_mulafd32x16x2_fir_hh, 0, 0x1c0300};
case Xtensa::BI__builtin_xtensa_ae_mulafd32x16x2_fir_hl:
return {Intrinsic::xtensa_ae_mulafd32x16x2_fir_hl, 0, 0x1c0300};
case Xtensa::BI__builtin_xtensa_ae_mulafd32x16x2_fir_lh:
return {Intrinsic::xtensa_ae_mulafd32x16x2_fir_lh, 0, 0x1c0300};
case Xtensa::BI__builtin_xtensa_ae_mulafd32x16x2_fir_ll:
return {Intrinsic::xtensa_ae_mulafd32x16x2_fir_ll, 0, 0x1c0300};
case Xtensa::BI__builtin_xtensa_ae_mulafp24x2r:
return {Intrinsic::xtensa_ae_mulafp24x2r, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulafp24x2r_s2:
return {Intrinsic::xtensa_ae_mulafp24x2r_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulafp24x2ra:
return {Intrinsic::xtensa_ae_mulafp24x2ra, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulafp24x2ra_s2:
return {Intrinsic::xtensa_ae_mulafp24x2ra_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulafp32x16x2ras_h:
return {Intrinsic::xtensa_ae_mulafp32x16x2ras_h, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulafp32x16x2ras_h_s2:
return {Intrinsic::xtensa_ae_mulafp32x16x2ras_h_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulafp32x16x2ras_l:
return {Intrinsic::xtensa_ae_mulafp32x16x2ras_l, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulafp32x16x2ras_l_s2:
return {Intrinsic::xtensa_ae_mulafp32x16x2ras_l_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulafp32x16x2rs_h:
return {Intrinsic::xtensa_ae_mulafp32x16x2rs_h, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulafp32x16x2rs_h_s2:
return {Intrinsic::xtensa_ae_mulafp32x16x2rs_h_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulafp32x16x2rs_l:
return {Intrinsic::xtensa_ae_mulafp32x16x2rs_l, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulafp32x16x2rs_l_s2:
return {Intrinsic::xtensa_ae_mulafp32x16x2rs_l_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulafp32x2ras:
return {Intrinsic::xtensa_ae_mulafp32x2ras, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulafp32x2rs:
return {Intrinsic::xtensa_ae_mulafp32x2rs, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulafq32sp24s_h_s2:
return {Intrinsic::xtensa_ae_mulafq32sp24s_h_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulafq32sp24s_l_s2:
return {Intrinsic::xtensa_ae_mulafq32sp24s_l_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulap24x2:
return {Intrinsic::xtensa_ae_mulap24x2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulap24x2_s2:
return {Intrinsic::xtensa_ae_mulap24x2_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulap32x16x2_h:
return {Intrinsic::xtensa_ae_mulap32x16x2_h, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulap32x16x2_l:
return {Intrinsic::xtensa_ae_mulap32x16x2_l, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulap32x2:
return {Intrinsic::xtensa_ae_mulap32x2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaq32sp16s_l_s2:
return {Intrinsic::xtensa_ae_mulaq32sp16s_l_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulaq32sp16u_l_s2:
return {Intrinsic::xtensa_ae_mulaq32sp16u_l_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mularfq32sp24s_h_s2:
return {Intrinsic::xtensa_ae_mularfq32sp24s_h_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mularfq32sp24s_l_s2:
return {Intrinsic::xtensa_ae_mularfq32sp24s_l_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulas32f48p16s_hh:
return {Intrinsic::xtensa_ae_mulas32f48p16s_hh, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulas32f48p16s_hh_s2:
return {Intrinsic::xtensa_ae_mulas32f48p16s_hh_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulas32f48p16s_lh:
return {Intrinsic::xtensa_ae_mulas32f48p16s_lh, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulas32f48p16s_lh_s2:
return {Intrinsic::xtensa_ae_mulas32f48p16s_lh_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulas32f48p16s_ll:
return {Intrinsic::xtensa_ae_mulas32f48p16s_ll, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulas32f48p16s_ll_s2:
return {Intrinsic::xtensa_ae_mulas32f48p16s_ll_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulasd24_hh_ll:
return {Intrinsic::xtensa_ae_mulasd24_hh_ll, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulasd24_hh_ll_s2:
return {Intrinsic::xtensa_ae_mulasd24_hh_ll_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulasd24_hl_lh:
return {Intrinsic::xtensa_ae_mulasd24_hl_lh, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulasd24_hl_lh_s2:
return {Intrinsic::xtensa_ae_mulasd24_hl_lh_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulasd32x16_h1_l0:
return {Intrinsic::xtensa_ae_mulasd32x16_h1_l0, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulasd32x16_h1_l0_s2:
return {Intrinsic::xtensa_ae_mulasd32x16_h1_l0_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulasd32x16_h3_l2:
return {Intrinsic::xtensa_ae_mulasd32x16_h3_l2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulasd32x16_h3_l2_s2:
return {Intrinsic::xtensa_ae_mulasd32x16_h3_l2_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulasfd24_hh_ll:
return {Intrinsic::xtensa_ae_mulasfd24_hh_ll, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulasfd24_hh_ll_s2:
return {Intrinsic::xtensa_ae_mulasfd24_hh_ll_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulasfd24_hl_lh:
return {Intrinsic::xtensa_ae_mulasfd24_hl_lh, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulasfd24_hl_lh_s2:
return {Intrinsic::xtensa_ae_mulasfd24_hl_lh_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulasfd32x16_h1_l0:
return {Intrinsic::xtensa_ae_mulasfd32x16_h1_l0, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulasfd32x16_h1_l0_s2:
return {Intrinsic::xtensa_ae_mulasfd32x16_h1_l0_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulasfd32x16_h3_l2:
return {Intrinsic::xtensa_ae_mulasfd32x16_h3_l2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulasfd32x16_h3_l2_s2:
return {Intrinsic::xtensa_ae_mulasfd32x16_h3_l2_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulc24:
return {Intrinsic::xtensa_ae_mulc24, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulc32x16_h:
return {Intrinsic::xtensa_ae_mulc32x16_h, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulc32x16_l:
return {Intrinsic::xtensa_ae_mulc32x16_l, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulf16ss_00:
return {Intrinsic::xtensa_ae_mulf16ss_00, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulf16ss_00_s2:
return {Intrinsic::xtensa_ae_mulf16ss_00_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulf16ss_10:
return {Intrinsic::xtensa_ae_mulf16ss_10, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulf16ss_11:
return {Intrinsic::xtensa_ae_mulf16ss_11, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulf16ss_20:
return {Intrinsic::xtensa_ae_mulf16ss_20, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulf16ss_21:
return {Intrinsic::xtensa_ae_mulf16ss_21, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulf16ss_22:
return {Intrinsic::xtensa_ae_mulf16ss_22, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulf16ss_30:
return {Intrinsic::xtensa_ae_mulf16ss_30, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulf16ss_31:
return {Intrinsic::xtensa_ae_mulf16ss_31, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulf16ss_32:
return {Intrinsic::xtensa_ae_mulf16ss_32, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulf16ss_33:
return {Intrinsic::xtensa_ae_mulf16ss_33, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulf16x4ss:
return {Intrinsic::xtensa_ae_mulf16x4ss, 0, 0xc0003};
case Xtensa::BI__builtin_xtensa_ae_mulf32r_hh:
return {Intrinsic::xtensa_ae_mulf32r_hh, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulf32r_lh:
return {Intrinsic::xtensa_ae_mulf32r_lh, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulf32r_ll:
return {Intrinsic::xtensa_ae_mulf32r_ll, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulf32r_ll_s2:
return {Intrinsic::xtensa_ae_mulf32r_ll_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulf32s_hh:
return {Intrinsic::xtensa_ae_mulf32s_hh, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulf32s_lh:
return {Intrinsic::xtensa_ae_mulf32s_lh, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulf32s_ll:
return {Intrinsic::xtensa_ae_mulf32s_ll, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulf32s_ll_s2:
return {Intrinsic::xtensa_ae_mulf32s_ll_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulf32x16_h0:
return {Intrinsic::xtensa_ae_mulf32x16_h0, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulf32x16_h0_s2:
return {Intrinsic::xtensa_ae_mulf32x16_h0_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulf32x16_h1:
return {Intrinsic::xtensa_ae_mulf32x16_h1, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulf32x16_h1_s2:
return {Intrinsic::xtensa_ae_mulf32x16_h1_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulf32x16_h2:
return {Intrinsic::xtensa_ae_mulf32x16_h2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulf32x16_h2_s2:
return {Intrinsic::xtensa_ae_mulf32x16_h2_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulf32x16_h3:
return {Intrinsic::xtensa_ae_mulf32x16_h3, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulf32x16_h3_s2:
return {Intrinsic::xtensa_ae_mulf32x16_h3_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulf32x16_l0:
return {Intrinsic::xtensa_ae_mulf32x16_l0, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulf32x16_l0_s2:
return {Intrinsic::xtensa_ae_mulf32x16_l0_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulf32x16_l1:
return {Intrinsic::xtensa_ae_mulf32x16_l1, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulf32x16_l1_s2:
return {Intrinsic::xtensa_ae_mulf32x16_l1_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulf32x16_l2:
return {Intrinsic::xtensa_ae_mulf32x16_l2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulf32x16_l2_s2:
return {Intrinsic::xtensa_ae_mulf32x16_l2_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulf32x16_l3:
return {Intrinsic::xtensa_ae_mulf32x16_l3, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulf32x16_l3_s2:
return {Intrinsic::xtensa_ae_mulf32x16_l3_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulf48q32sp16s_l:
return {Intrinsic::xtensa_ae_mulf48q32sp16s_l, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulf48q32sp16s_l_s2:
return {Intrinsic::xtensa_ae_mulf48q32sp16s_l_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulf48q32sp16u_l:
return {Intrinsic::xtensa_ae_mulf48q32sp16u_l, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulf48q32sp16u_l_s2:
return {Intrinsic::xtensa_ae_mulf48q32sp16u_l_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulfc24ra:
return {Intrinsic::xtensa_ae_mulfc24ra, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulfc32x16ras_h:
return {Intrinsic::xtensa_ae_mulfc32x16ras_h, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulfc32x16ras_l:
return {Intrinsic::xtensa_ae_mulfc32x16ras_l, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulfd24x2_fir_h:
return {Intrinsic::xtensa_ae_mulfd24x2_fir_h, 0, 0x1c0003};
case Xtensa::BI__builtin_xtensa_ae_mulfd24x2_fir_l:
return {Intrinsic::xtensa_ae_mulfd24x2_fir_l, 0, 0x1c0003};
case Xtensa::BI__builtin_xtensa_ae_mulfd32x16x2_fir_hh:
return {Intrinsic::xtensa_ae_mulfd32x16x2_fir_hh, 0, 0x1c0003};
case Xtensa::BI__builtin_xtensa_ae_mulfd32x16x2_fir_hl:
return {Intrinsic::xtensa_ae_mulfd32x16x2_fir_hl, 0, 0x1c0003};
case Xtensa::BI__builtin_xtensa_ae_mulfd32x16x2_fir_lh:
return {Intrinsic::xtensa_ae_mulfd32x16x2_fir_lh, 0, 0x1c0003};
case Xtensa::BI__builtin_xtensa_ae_mulfd32x16x2_fir_ll:
return {Intrinsic::xtensa_ae_mulfd32x16x2_fir_ll, 0, 0x1c0003};
case Xtensa::BI__builtin_xtensa_ae_mulfp16x4ras:
return {Intrinsic::xtensa_ae_mulfp16x4ras, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulfp16x4s:
return {Intrinsic::xtensa_ae_mulfp16x4s, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulfp24x2r:
return {Intrinsic::xtensa_ae_mulfp24x2r, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulfp24x2r_s2:
return {Intrinsic::xtensa_ae_mulfp24x2r_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulfp24x2ra:
return {Intrinsic::xtensa_ae_mulfp24x2ra, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulfp24x2ra_s2:
return {Intrinsic::xtensa_ae_mulfp24x2ra_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulfp32x16x2ras_h:
return {Intrinsic::xtensa_ae_mulfp32x16x2ras_h, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulfp32x16x2ras_h_s2:
return {Intrinsic::xtensa_ae_mulfp32x16x2ras_h_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulfp32x16x2ras_l:
return {Intrinsic::xtensa_ae_mulfp32x16x2ras_l, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulfp32x16x2ras_l_s2:
return {Intrinsic::xtensa_ae_mulfp32x16x2ras_l_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulfp32x16x2rs_h:
return {Intrinsic::xtensa_ae_mulfp32x16x2rs_h, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulfp32x16x2rs_h_s2:
return {Intrinsic::xtensa_ae_mulfp32x16x2rs_h_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulfp32x16x2rs_l:
return {Intrinsic::xtensa_ae_mulfp32x16x2rs_l, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulfp32x16x2rs_l_s2:
return {Intrinsic::xtensa_ae_mulfp32x16x2rs_l_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulfp32x2ras:
return {Intrinsic::xtensa_ae_mulfp32x2ras, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulfp32x2rs:
return {Intrinsic::xtensa_ae_mulfp32x2rs, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulfq32sp24s_h_s2:
return {Intrinsic::xtensa_ae_mulfq32sp24s_h_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulfq32sp24s_l_s2:
return {Intrinsic::xtensa_ae_mulfq32sp24s_l_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulp24x2:
return {Intrinsic::xtensa_ae_mulp24x2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulp24x2_s2:
return {Intrinsic::xtensa_ae_mulp24x2_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulp32x16x2_h:
return {Intrinsic::xtensa_ae_mulp32x16x2_h, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulp32x16x2_l:
return {Intrinsic::xtensa_ae_mulp32x16x2_l, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulp32x2:
return {Intrinsic::xtensa_ae_mulp32x2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulq32sp16s_l_s2:
return {Intrinsic::xtensa_ae_mulq32sp16s_l_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulq32sp16u_l_s2:
return {Intrinsic::xtensa_ae_mulq32sp16u_l_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulrfq32sp24s_h_s2:
return {Intrinsic::xtensa_ae_mulrfq32sp24s_h_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulrfq32sp24s_l_s2:
return {Intrinsic::xtensa_ae_mulrfq32sp24s_l_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_muls16x4:
return {Intrinsic::xtensa_ae_muls16x4, 0, 0xc0300};
case Xtensa::BI__builtin_xtensa_ae_muls32_hh:
return {Intrinsic::xtensa_ae_muls32_hh, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_muls32_lh:
return {Intrinsic::xtensa_ae_muls32_lh, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_muls32_ll:
return {Intrinsic::xtensa_ae_muls32_ll, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_muls32f48p16s_hh:
return {Intrinsic::xtensa_ae_muls32f48p16s_hh, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_muls32f48p16s_hh_s2:
return {Intrinsic::xtensa_ae_muls32f48p16s_hh_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_muls32f48p16s_lh:
return {Intrinsic::xtensa_ae_muls32f48p16s_lh, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_muls32f48p16s_lh_s2:
return {Intrinsic::xtensa_ae_muls32f48p16s_lh_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_muls32f48p16s_ll:
return {Intrinsic::xtensa_ae_muls32f48p16s_ll, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_muls32f48p16s_ll_s2:
return {Intrinsic::xtensa_ae_muls32f48p16s_ll_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_muls32u_ll:
return {Intrinsic::xtensa_ae_muls32u_ll, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_muls32x16_h0:
return {Intrinsic::xtensa_ae_muls32x16_h0, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_muls32x16_h0_s2:
return {Intrinsic::xtensa_ae_muls32x16_h0_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_muls32x16_h1:
return {Intrinsic::xtensa_ae_muls32x16_h1, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_muls32x16_h1_s2:
return {Intrinsic::xtensa_ae_muls32x16_h1_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_muls32x16_h2:
return {Intrinsic::xtensa_ae_muls32x16_h2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_muls32x16_h2_s2:
return {Intrinsic::xtensa_ae_muls32x16_h2_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_muls32x16_h3:
return {Intrinsic::xtensa_ae_muls32x16_h3, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_muls32x16_h3_s2:
return {Intrinsic::xtensa_ae_muls32x16_h3_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_muls32x16_l0:
return {Intrinsic::xtensa_ae_muls32x16_l0, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_muls32x16_l0_s2:
return {Intrinsic::xtensa_ae_muls32x16_l0_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_muls32x16_l1:
return {Intrinsic::xtensa_ae_muls32x16_l1, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_muls32x16_l1_s2:
return {Intrinsic::xtensa_ae_muls32x16_l1_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_muls32x16_l2:
return {Intrinsic::xtensa_ae_muls32x16_l2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_muls32x16_l2_s2:
return {Intrinsic::xtensa_ae_muls32x16_l2_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_muls32x16_l3:
return {Intrinsic::xtensa_ae_muls32x16_l3, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_muls32x16_l3_s2:
return {Intrinsic::xtensa_ae_muls32x16_l3_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsad24_hh_ll:
return {Intrinsic::xtensa_ae_mulsad24_hh_ll, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsad24_hh_ll_s2:
return {Intrinsic::xtensa_ae_mulsad24_hh_ll_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsad32x16_h1_l0:
return {Intrinsic::xtensa_ae_mulsad32x16_h1_l0, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsad32x16_h1_l0_s2:
return {Intrinsic::xtensa_ae_mulsad32x16_h1_l0_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsad32x16_h3_l2:
return {Intrinsic::xtensa_ae_mulsad32x16_h3_l2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsad32x16_h3_l2_s2:
return {Intrinsic::xtensa_ae_mulsad32x16_h3_l2_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsafd24_hh_ll:
return {Intrinsic::xtensa_ae_mulsafd24_hh_ll, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsafd24_hh_ll_s2:
return {Intrinsic::xtensa_ae_mulsafd24_hh_ll_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsafd32x16_h1_l0:
return {Intrinsic::xtensa_ae_mulsafd32x16_h1_l0, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsafd32x16_h1_l0_s2:
return {Intrinsic::xtensa_ae_mulsafd32x16_h1_l0_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsafd32x16_h3_l2:
return {Intrinsic::xtensa_ae_mulsafd32x16_h3_l2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsafd32x16_h3_l2_s2:
return {Intrinsic::xtensa_ae_mulsafd32x16_h3_l2_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsf16ss_00:
return {Intrinsic::xtensa_ae_mulsf16ss_00, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsf16ss_00_s2:
return {Intrinsic::xtensa_ae_mulsf16ss_00_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsf16ss_10:
return {Intrinsic::xtensa_ae_mulsf16ss_10, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsf16ss_11:
return {Intrinsic::xtensa_ae_mulsf16ss_11, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsf16ss_20:
return {Intrinsic::xtensa_ae_mulsf16ss_20, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsf16ss_21:
return {Intrinsic::xtensa_ae_mulsf16ss_21, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsf16ss_22:
return {Intrinsic::xtensa_ae_mulsf16ss_22, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsf16ss_30:
return {Intrinsic::xtensa_ae_mulsf16ss_30, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsf16ss_31:
return {Intrinsic::xtensa_ae_mulsf16ss_31, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsf16ss_32:
return {Intrinsic::xtensa_ae_mulsf16ss_32, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsf16ss_33:
return {Intrinsic::xtensa_ae_mulsf16ss_33, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsf16x4ss:
return {Intrinsic::xtensa_ae_mulsf16x4ss, 0, 0xc0300};
case Xtensa::BI__builtin_xtensa_ae_mulsf32r_hh:
return {Intrinsic::xtensa_ae_mulsf32r_hh, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsf32r_lh:
return {Intrinsic::xtensa_ae_mulsf32r_lh, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsf32r_ll:
return {Intrinsic::xtensa_ae_mulsf32r_ll, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsf32r_ll_s2:
return {Intrinsic::xtensa_ae_mulsf32r_ll_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsf32s_hh:
return {Intrinsic::xtensa_ae_mulsf32s_hh, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsf32s_lh:
return {Intrinsic::xtensa_ae_mulsf32s_lh, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsf32s_ll:
return {Intrinsic::xtensa_ae_mulsf32s_ll, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsf32x16_h0:
return {Intrinsic::xtensa_ae_mulsf32x16_h0, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsf32x16_h0_s2:
return {Intrinsic::xtensa_ae_mulsf32x16_h0_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsf32x16_h1:
return {Intrinsic::xtensa_ae_mulsf32x16_h1, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsf32x16_h1_s2:
return {Intrinsic::xtensa_ae_mulsf32x16_h1_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsf32x16_h2:
return {Intrinsic::xtensa_ae_mulsf32x16_h2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsf32x16_h2_s2:
return {Intrinsic::xtensa_ae_mulsf32x16_h2_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsf32x16_h3:
return {Intrinsic::xtensa_ae_mulsf32x16_h3, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsf32x16_h3_s2:
return {Intrinsic::xtensa_ae_mulsf32x16_h3_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsf32x16_l0:
return {Intrinsic::xtensa_ae_mulsf32x16_l0, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsf32x16_l0_s2:
return {Intrinsic::xtensa_ae_mulsf32x16_l0_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsf32x16_l1:
return {Intrinsic::xtensa_ae_mulsf32x16_l1, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsf32x16_l1_s2:
return {Intrinsic::xtensa_ae_mulsf32x16_l1_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsf32x16_l2:
return {Intrinsic::xtensa_ae_mulsf32x16_l2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsf32x16_l2_s2:
return {Intrinsic::xtensa_ae_mulsf32x16_l2_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsf32x16_l3:
return {Intrinsic::xtensa_ae_mulsf32x16_l3, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsf32x16_l3_s2:
return {Intrinsic::xtensa_ae_mulsf32x16_l3_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsf48q32sp16s_l:
return {Intrinsic::xtensa_ae_mulsf48q32sp16s_l, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsf48q32sp16s_l_s2:
return {Intrinsic::xtensa_ae_mulsf48q32sp16s_l_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsf48q32sp16u_l:
return {Intrinsic::xtensa_ae_mulsf48q32sp16u_l, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsf48q32sp16u_l_s2:
return {Intrinsic::xtensa_ae_mulsf48q32sp16u_l_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsfp24x2r:
return {Intrinsic::xtensa_ae_mulsfp24x2r, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsfp24x2r_s2:
return {Intrinsic::xtensa_ae_mulsfp24x2r_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsfp24x2ra:
return {Intrinsic::xtensa_ae_mulsfp24x2ra, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsfp24x2ra_s2:
return {Intrinsic::xtensa_ae_mulsfp24x2ra_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsfp32x16x2ras_h:
return {Intrinsic::xtensa_ae_mulsfp32x16x2ras_h, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsfp32x16x2ras_h_s2:
return {Intrinsic::xtensa_ae_mulsfp32x16x2ras_h_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsfp32x16x2ras_l:
return {Intrinsic::xtensa_ae_mulsfp32x16x2ras_l, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsfp32x16x2ras_l_s2:
return {Intrinsic::xtensa_ae_mulsfp32x16x2ras_l_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsfp32x16x2rs_h:
return {Intrinsic::xtensa_ae_mulsfp32x16x2rs_h, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsfp32x16x2rs_h_s2:
return {Intrinsic::xtensa_ae_mulsfp32x16x2rs_h_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsfp32x16x2rs_l:
return {Intrinsic::xtensa_ae_mulsfp32x16x2rs_l, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsfp32x16x2rs_l_s2:
return {Intrinsic::xtensa_ae_mulsfp32x16x2rs_l_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsfp32x2ras:
return {Intrinsic::xtensa_ae_mulsfp32x2ras, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsfp32x2rs:
return {Intrinsic::xtensa_ae_mulsfp32x2rs, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsfq32sp24s_h_s2:
return {Intrinsic::xtensa_ae_mulsfq32sp24s_h_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsfq32sp24s_l_s2:
return {Intrinsic::xtensa_ae_mulsfq32sp24s_l_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsp24x2:
return {Intrinsic::xtensa_ae_mulsp24x2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsp24x2_s2:
return {Intrinsic::xtensa_ae_mulsp24x2_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsp32x16x2_h:
return {Intrinsic::xtensa_ae_mulsp32x16x2_h, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsp32x16x2_l:
return {Intrinsic::xtensa_ae_mulsp32x16x2_l, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsp32x2:
return {Intrinsic::xtensa_ae_mulsp32x2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsq32sp16s_l_s2:
return {Intrinsic::xtensa_ae_mulsq32sp16s_l_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsq32sp16u_l_s2:
return {Intrinsic::xtensa_ae_mulsq32sp16u_l_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsrfq32sp24s_h_s2:
return {Intrinsic::xtensa_ae_mulsrfq32sp24s_h_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulsrfq32sp24s_l_s2:
return {Intrinsic::xtensa_ae_mulsrfq32sp24s_l_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulss32f48p16s_hh:
return {Intrinsic::xtensa_ae_mulss32f48p16s_hh, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulss32f48p16s_hh_s2:
return {Intrinsic::xtensa_ae_mulss32f48p16s_hh_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulss32f48p16s_lh:
return {Intrinsic::xtensa_ae_mulss32f48p16s_lh, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulss32f48p16s_lh_s2:
return {Intrinsic::xtensa_ae_mulss32f48p16s_lh_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulss32f48p16s_ll:
return {Intrinsic::xtensa_ae_mulss32f48p16s_ll, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulss32f48p16s_ll_s2:
return {Intrinsic::xtensa_ae_mulss32f48p16s_ll_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulssd24_hh_ll:
return {Intrinsic::xtensa_ae_mulssd24_hh_ll, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulssd24_hh_ll_s2:
return {Intrinsic::xtensa_ae_mulssd24_hh_ll_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulssd24_hl_lh:
return {Intrinsic::xtensa_ae_mulssd24_hl_lh, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulssd24_hl_lh_s2:
return {Intrinsic::xtensa_ae_mulssd24_hl_lh_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulssd32x16_h1_l0:
return {Intrinsic::xtensa_ae_mulssd32x16_h1_l0, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulssd32x16_h1_l0_s2:
return {Intrinsic::xtensa_ae_mulssd32x16_h1_l0_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulssd32x16_h3_l2:
return {Intrinsic::xtensa_ae_mulssd32x16_h3_l2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulssd32x16_h3_l2_s2:
return {Intrinsic::xtensa_ae_mulssd32x16_h3_l2_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulssfd16ss_11_00:
return {Intrinsic::xtensa_ae_mulssfd16ss_11_00, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulssfd16ss_11_00_s2:
return {Intrinsic::xtensa_ae_mulssfd16ss_11_00_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulssfd16ss_13_02:
return {Intrinsic::xtensa_ae_mulssfd16ss_13_02, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulssfd16ss_13_02_s2:
return {Intrinsic::xtensa_ae_mulssfd16ss_13_02_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulssfd16ss_33_22:
return {Intrinsic::xtensa_ae_mulssfd16ss_33_22, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulssfd16ss_33_22_s2:
return {Intrinsic::xtensa_ae_mulssfd16ss_33_22_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulssfd24_hh_ll:
return {Intrinsic::xtensa_ae_mulssfd24_hh_ll, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulssfd24_hh_ll_s2:
return {Intrinsic::xtensa_ae_mulssfd24_hh_ll_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulssfd24_hl_lh:
return {Intrinsic::xtensa_ae_mulssfd24_hl_lh, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulssfd24_hl_lh_s2:
return {Intrinsic::xtensa_ae_mulssfd24_hl_lh_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulssfd32x16_h1_l0:
return {Intrinsic::xtensa_ae_mulssfd32x16_h1_l0, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulssfd32x16_h1_l0_s2:
return {Intrinsic::xtensa_ae_mulssfd32x16_h1_l0_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulssfd32x16_h3_l2:
return {Intrinsic::xtensa_ae_mulssfd32x16_h3_l2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulssfd32x16_h3_l2_s2:
return {Intrinsic::xtensa_ae_mulssfd32x16_h3_l2_s2, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_mulzaad24_hh_ll:
return {Intrinsic::xtensa_ae_mulzaad24_hh_ll, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzaad24_hh_ll_s2:
return {Intrinsic::xtensa_ae_mulzaad24_hh_ll_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzaad24_hl_lh:
return {Intrinsic::xtensa_ae_mulzaad24_hl_lh, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzaad24_hl_lh_s2:
return {Intrinsic::xtensa_ae_mulzaad24_hl_lh_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzaad32x16_h0_l1:
return {Intrinsic::xtensa_ae_mulzaad32x16_h0_l1, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzaad32x16_h0_l1_s2:
return {Intrinsic::xtensa_ae_mulzaad32x16_h0_l1_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzaad32x16_h1_l0:
return {Intrinsic::xtensa_ae_mulzaad32x16_h1_l0, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzaad32x16_h1_l0_s2:
return {Intrinsic::xtensa_ae_mulzaad32x16_h1_l0_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzaad32x16_h2_l3:
return {Intrinsic::xtensa_ae_mulzaad32x16_h2_l3, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzaad32x16_h2_l3_s2:
return {Intrinsic::xtensa_ae_mulzaad32x16_h2_l3_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzaad32x16_h3_l2:
return {Intrinsic::xtensa_ae_mulzaad32x16_h3_l2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzaad32x16_h3_l2_s2:
return {Intrinsic::xtensa_ae_mulzaad32x16_h3_l2_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzaafd16ss_11_00:
return {Intrinsic::xtensa_ae_mulzaafd16ss_11_00, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzaafd16ss_11_00_s2:
return {Intrinsic::xtensa_ae_mulzaafd16ss_11_00_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzaafd16ss_13_02:
return {Intrinsic::xtensa_ae_mulzaafd16ss_13_02, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzaafd16ss_13_02_s2:
return {Intrinsic::xtensa_ae_mulzaafd16ss_13_02_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzaafd16ss_33_22:
return {Intrinsic::xtensa_ae_mulzaafd16ss_33_22, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzaafd16ss_33_22_s2:
return {Intrinsic::xtensa_ae_mulzaafd16ss_33_22_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzaafd24_hh_ll:
return {Intrinsic::xtensa_ae_mulzaafd24_hh_ll, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzaafd24_hh_ll_s2:
return {Intrinsic::xtensa_ae_mulzaafd24_hh_ll_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzaafd24_hl_lh:
return {Intrinsic::xtensa_ae_mulzaafd24_hl_lh, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzaafd24_hl_lh_s2:
return {Intrinsic::xtensa_ae_mulzaafd24_hl_lh_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzaafd32x16_h0_l1:
return {Intrinsic::xtensa_ae_mulzaafd32x16_h0_l1, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzaafd32x16_h0_l1_s2:
return {Intrinsic::xtensa_ae_mulzaafd32x16_h0_l1_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzaafd32x16_h1_l0:
return {Intrinsic::xtensa_ae_mulzaafd32x16_h1_l0, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzaafd32x16_h1_l0_s2:
return {Intrinsic::xtensa_ae_mulzaafd32x16_h1_l0_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzaafd32x16_h2_l3:
return {Intrinsic::xtensa_ae_mulzaafd32x16_h2_l3, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzaafd32x16_h2_l3_s2:
return {Intrinsic::xtensa_ae_mulzaafd32x16_h2_l3_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzaafd32x16_h3_l2:
return {Intrinsic::xtensa_ae_mulzaafd32x16_h3_l2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzaafd32x16_h3_l2_s2:
return {Intrinsic::xtensa_ae_mulzaafd32x16_h3_l2_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzasd24_hh_ll:
return {Intrinsic::xtensa_ae_mulzasd24_hh_ll, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzasd24_hh_ll_s2:
return {Intrinsic::xtensa_ae_mulzasd24_hh_ll_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzasd24_hl_lh:
return {Intrinsic::xtensa_ae_mulzasd24_hl_lh, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzasd24_hl_lh_s2:
return {Intrinsic::xtensa_ae_mulzasd24_hl_lh_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzasd32x16_h1_l0:
return {Intrinsic::xtensa_ae_mulzasd32x16_h1_l0, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzasd32x16_h1_l0_s2:
return {Intrinsic::xtensa_ae_mulzasd32x16_h1_l0_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzasd32x16_h3_l2:
return {Intrinsic::xtensa_ae_mulzasd32x16_h3_l2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzasd32x16_h3_l2_s2:
return {Intrinsic::xtensa_ae_mulzasd32x16_h3_l2_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzasfd24_hh_ll:
return {Intrinsic::xtensa_ae_mulzasfd24_hh_ll, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzasfd24_hh_ll_s2:
return {Intrinsic::xtensa_ae_mulzasfd24_hh_ll_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzasfd24_hl_lh:
return {Intrinsic::xtensa_ae_mulzasfd24_hl_lh, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzasfd24_hl_lh_s2:
return {Intrinsic::xtensa_ae_mulzasfd24_hl_lh_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzasfd32x16_h1_l0:
return {Intrinsic::xtensa_ae_mulzasfd32x16_h1_l0, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzasfd32x16_h1_l0_s2:
return {Intrinsic::xtensa_ae_mulzasfd32x16_h1_l0_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzasfd32x16_h3_l2:
return {Intrinsic::xtensa_ae_mulzasfd32x16_h3_l2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzasfd32x16_h3_l2_s2:
return {Intrinsic::xtensa_ae_mulzasfd32x16_h3_l2_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzsad24_hh_ll:
return {Intrinsic::xtensa_ae_mulzsad24_hh_ll, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzsad24_hh_ll_s2:
return {Intrinsic::xtensa_ae_mulzsad24_hh_ll_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzsad32x16_h1_l0:
return {Intrinsic::xtensa_ae_mulzsad32x16_h1_l0, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzsad32x16_h1_l0_s2:
return {Intrinsic::xtensa_ae_mulzsad32x16_h1_l0_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzsad32x16_h3_l2:
return {Intrinsic::xtensa_ae_mulzsad32x16_h3_l2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzsad32x16_h3_l2_s2:
return {Intrinsic::xtensa_ae_mulzsad32x16_h3_l2_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzsafd24_hh_ll:
return {Intrinsic::xtensa_ae_mulzsafd24_hh_ll, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzsafd24_hh_ll_s2:
return {Intrinsic::xtensa_ae_mulzsafd24_hh_ll_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzsafd32x16_h1_l0:
return {Intrinsic::xtensa_ae_mulzsafd32x16_h1_l0, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzsafd32x16_h1_l0_s2:
return {Intrinsic::xtensa_ae_mulzsafd32x16_h1_l0_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzsafd32x16_h3_l2:
return {Intrinsic::xtensa_ae_mulzsafd32x16_h3_l2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzsafd32x16_h3_l2_s2:
return {Intrinsic::xtensa_ae_mulzsafd32x16_h3_l2_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzssd24_hh_ll:
return {Intrinsic::xtensa_ae_mulzssd24_hh_ll, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzssd24_hh_ll_s2:
return {Intrinsic::xtensa_ae_mulzssd24_hh_ll_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzssd24_hl_lh:
return {Intrinsic::xtensa_ae_mulzssd24_hl_lh, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzssd24_hl_lh_s2:
return {Intrinsic::xtensa_ae_mulzssd24_hl_lh_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzssd32x16_h1_l0:
return {Intrinsic::xtensa_ae_mulzssd32x16_h1_l0, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzssd32x16_h1_l0_s2:
return {Intrinsic::xtensa_ae_mulzssd32x16_h1_l0_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzssd32x16_h3_l2:
return {Intrinsic::xtensa_ae_mulzssd32x16_h3_l2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzssd32x16_h3_l2_s2:
return {Intrinsic::xtensa_ae_mulzssd32x16_h3_l2_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzssfd16ss_11_00:
return {Intrinsic::xtensa_ae_mulzssfd16ss_11_00, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzssfd16ss_11_00_s2:
return {Intrinsic::xtensa_ae_mulzssfd16ss_11_00_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzssfd16ss_13_02:
return {Intrinsic::xtensa_ae_mulzssfd16ss_13_02, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzssfd16ss_13_02_s2:
return {Intrinsic::xtensa_ae_mulzssfd16ss_13_02_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzssfd16ss_33_22:
return {Intrinsic::xtensa_ae_mulzssfd16ss_33_22, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzssfd16ss_33_22_s2:
return {Intrinsic::xtensa_ae_mulzssfd16ss_33_22_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzssfd24_hh_ll:
return {Intrinsic::xtensa_ae_mulzssfd24_hh_ll, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzssfd24_hh_ll_s2:
return {Intrinsic::xtensa_ae_mulzssfd24_hh_ll_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzssfd24_hl_lh:
return {Intrinsic::xtensa_ae_mulzssfd24_hl_lh, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzssfd24_hl_lh_s2:
return {Intrinsic::xtensa_ae_mulzssfd24_hl_lh_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzssfd32x16_h1_l0:
return {Intrinsic::xtensa_ae_mulzssfd32x16_h1_l0, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzssfd32x16_h1_l0_s2:
return {Intrinsic::xtensa_ae_mulzssfd32x16_h1_l0_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzssfd32x16_h3_l2:
return {Intrinsic::xtensa_ae_mulzssfd32x16_h3_l2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_mulzssfd32x16_h3_l2_s2:
return {Intrinsic::xtensa_ae_mulzssfd32x16_h3_l2_s2, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_nand:
return {Intrinsic::xtensa_ae_nand, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_neg16s:
return {Intrinsic::xtensa_ae_neg16s, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_neg24s:
return {Intrinsic::xtensa_ae_neg24s, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_neg32:
return {Intrinsic::xtensa_ae_neg32, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_neg32s:
return {Intrinsic::xtensa_ae_neg32s, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_neg64:
return {Intrinsic::xtensa_ae_neg64, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_neg64s:
return {Intrinsic::xtensa_ae_neg64s, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_nsa64:
return {Intrinsic::xtensa_ae_nsa64, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_nsaz16_0:
return {Intrinsic::xtensa_ae_nsaz16_0, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_nsaz32_l:
return {Intrinsic::xtensa_ae_nsaz32_l, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_or:
return {Intrinsic::xtensa_ae_or, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_pksr24:
return {Intrinsic::xtensa_ae_pksr24, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_pksr32:
return {Intrinsic::xtensa_ae_pksr32, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_round16x4f32sasym:
return {Intrinsic::xtensa_ae_round16x4f32sasym, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_round16x4f32ssym:
return {Intrinsic::xtensa_ae_round16x4f32ssym, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_round24x2f48sasym:
return {Intrinsic::xtensa_ae_round24x2f48sasym, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_round24x2f48ssym:
return {Intrinsic::xtensa_ae_round24x2f48ssym, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_round32x2f48sasym:
return {Intrinsic::xtensa_ae_round32x2f48sasym, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_round32x2f48ssym:
return {Intrinsic::xtensa_ae_round32x2f48ssym, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_round32x2f64sasym:
return {Intrinsic::xtensa_ae_round32x2f64sasym, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_round32x2f64ssym:
return {Intrinsic::xtensa_ae_round32x2f64ssym, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_roundsp16f24asym:
return {Intrinsic::xtensa_ae_roundsp16f24asym, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_roundsp16f24sym:
return {Intrinsic::xtensa_ae_roundsp16f24sym, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_roundsp16q48x2asym:
return {Intrinsic::xtensa_ae_roundsp16q48x2asym, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_roundsp16q48x2sym:
return {Intrinsic::xtensa_ae_roundsp16q48x2sym, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_roundsq32f48asym:
return {Intrinsic::xtensa_ae_roundsq32f48asym, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_roundsq32f48sym:
return {Intrinsic::xtensa_ae_roundsq32f48sym, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_s16_0_i:
return {Intrinsic::xtensa_ae_s16_0_i, 0, 0x70000};
case Xtensa::BI__builtin_xtensa_ae_s16_0_ip:
return {Intrinsic::xtensa_ae_s16_0_ip, 0, 0x50200};
case Xtensa::BI__builtin_xtensa_ae_s16_0_x:
return {Intrinsic::xtensa_ae_s16_0_x, 0, 0x70000};
case Xtensa::BI__builtin_xtensa_ae_s16_0_xc:
return {Intrinsic::xtensa_ae_s16_0_xc, 0, 0x50200};
case Xtensa::BI__builtin_xtensa_ae_s16_0_xp:
return {Intrinsic::xtensa_ae_s16_0_xp, 0, 0x50200};
case Xtensa::BI__builtin_xtensa_ae_s16m_l_i:
return {Intrinsic::xtensa_ae_s16m_l_i, 0, 0x70000};
case Xtensa::BI__builtin_xtensa_ae_s16m_l_iu:
return {Intrinsic::xtensa_ae_s16m_l_iu, 0, 0x50200};
case Xtensa::BI__builtin_xtensa_ae_s16m_l_x:
return {Intrinsic::xtensa_ae_s16m_l_x, 0, 0x70000};
case Xtensa::BI__builtin_xtensa_ae_s16m_l_xc:
return {Intrinsic::xtensa_ae_s16m_l_xc, 0, 0x50200};
case Xtensa::BI__builtin_xtensa_ae_s16m_l_xu:
return {Intrinsic::xtensa_ae_s16m_l_xu, 0, 0x50200};
case Xtensa::BI__builtin_xtensa_ae_s16x2m_i:
return {Intrinsic::xtensa_ae_s16x2m_i, 0, 0x70000};
case Xtensa::BI__builtin_xtensa_ae_s16x2m_iu:
return {Intrinsic::xtensa_ae_s16x2m_iu, 0, 0x50200};
case Xtensa::BI__builtin_xtensa_ae_s16x2m_x:
return {Intrinsic::xtensa_ae_s16x2m_x, 0, 0x70000};
case Xtensa::BI__builtin_xtensa_ae_s16x2m_xc:
return {Intrinsic::xtensa_ae_s16x2m_xc, 0, 0x50200};
case Xtensa::BI__builtin_xtensa_ae_s16x2m_xu:
return {Intrinsic::xtensa_ae_s16x2m_xu, 0, 0x50200};
case Xtensa::BI__builtin_xtensa_ae_s16x4_i:
return {Intrinsic::xtensa_ae_s16x4_i, 0, 0x70000};
case Xtensa::BI__builtin_xtensa_ae_s16x4_ip:
return {Intrinsic::xtensa_ae_s16x4_ip, 0, 0x50200};
case Xtensa::BI__builtin_xtensa_ae_s16x4_ric:
return {Intrinsic::xtensa_ae_s16x4_ric, 0, 0x10200};
case Xtensa::BI__builtin_xtensa_ae_s16x4_rip:
return {Intrinsic::xtensa_ae_s16x4_rip, 0, 0x10200};
case Xtensa::BI__builtin_xtensa_ae_s16x4_x:
return {Intrinsic::xtensa_ae_s16x4_x, 0, 0x70000};
case Xtensa::BI__builtin_xtensa_ae_s16x4_xc:
return {Intrinsic::xtensa_ae_s16x4_xc, 0, 0x50200};
case Xtensa::BI__builtin_xtensa_ae_s16x4_xp:
return {Intrinsic::xtensa_ae_s16x4_xp, 0, 0x50200};
case Xtensa::BI__builtin_xtensa_ae_s24ra64s_i:
return {Intrinsic::xtensa_ae_s24ra64s_i, 0, 0x70000};
case Xtensa::BI__builtin_xtensa_ae_s24ra64s_ip:
return {Intrinsic::xtensa_ae_s24ra64s_ip, 0, 0x50200};
case Xtensa::BI__builtin_xtensa_ae_s24ra64s_x:
return {Intrinsic::xtensa_ae_s24ra64s_x, 0, 0x70000};
case Xtensa::BI__builtin_xtensa_ae_s24ra64s_xc:
return {Intrinsic::xtensa_ae_s24ra64s_xc, 0, 0x50200};
case Xtensa::BI__builtin_xtensa_ae_s24ra64s_xp:
return {Intrinsic::xtensa_ae_s24ra64s_xp, 0, 0x50200};
case Xtensa::BI__builtin_xtensa_ae_s24x2ra64s_ip:
return {Intrinsic::xtensa_ae_s24x2ra64s_ip, 0, 0x30400};
case Xtensa::BI__builtin_xtensa_ae_s32_l_i:
return {Intrinsic::xtensa_ae_s32_l_i, 0, 0x70000};
case Xtensa::BI__builtin_xtensa_ae_s32_l_ip:
return {Intrinsic::xtensa_ae_s32_l_ip, 0, 0x50200};
case Xtensa::BI__builtin_xtensa_ae_s32_l_x:
return {Intrinsic::xtensa_ae_s32_l_x, 0, 0x70000};
case Xtensa::BI__builtin_xtensa_ae_s32_l_xc:
return {Intrinsic::xtensa_ae_s32_l_xc, 0, 0x50200};
case Xtensa::BI__builtin_xtensa_ae_s32_l_xp:
return {Intrinsic::xtensa_ae_s32_l_xp, 0, 0x50200};
case Xtensa::BI__builtin_xtensa_ae_s32f24_l_i:
return {Intrinsic::xtensa_ae_s32f24_l_i, 0, 0x70000};
case Xtensa::BI__builtin_xtensa_ae_s32f24_l_ip:
return {Intrinsic::xtensa_ae_s32f24_l_ip, 0, 0x50200};
case Xtensa::BI__builtin_xtensa_ae_s32f24_l_x:
return {Intrinsic::xtensa_ae_s32f24_l_x, 0, 0x70000};
case Xtensa::BI__builtin_xtensa_ae_s32f24_l_xc:
return {Intrinsic::xtensa_ae_s32f24_l_xc, 0, 0x50200};
case Xtensa::BI__builtin_xtensa_ae_s32f24_l_xp:
return {Intrinsic::xtensa_ae_s32f24_l_xp, 0, 0x50200};
case Xtensa::BI__builtin_xtensa_ae_s32m_i:
return {Intrinsic::xtensa_ae_s32m_i, 0, 0x70000};
case Xtensa::BI__builtin_xtensa_ae_s32m_iu:
return {Intrinsic::xtensa_ae_s32m_iu, 0, 0x50200};
case Xtensa::BI__builtin_xtensa_ae_s32m_x:
return {Intrinsic::xtensa_ae_s32m_x, 0, 0x70000};
case Xtensa::BI__builtin_xtensa_ae_s32m_xc:
return {Intrinsic::xtensa_ae_s32m_xc, 0, 0x50200};
case Xtensa::BI__builtin_xtensa_ae_s32m_xu:
return {Intrinsic::xtensa_ae_s32m_xu, 0, 0x50200};
case Xtensa::BI__builtin_xtensa_ae_s32ra64s_i:
return {Intrinsic::xtensa_ae_s32ra64s_i, 0, 0x70000};
case Xtensa::BI__builtin_xtensa_ae_s32ra64s_ip:
return {Intrinsic::xtensa_ae_s32ra64s_ip, 0, 0x50200};
case Xtensa::BI__builtin_xtensa_ae_s32ra64s_x:
return {Intrinsic::xtensa_ae_s32ra64s_x, 0, 0x70000};
case Xtensa::BI__builtin_xtensa_ae_s32ra64s_xc:
return {Intrinsic::xtensa_ae_s32ra64s_xc, 0, 0x50200};
case Xtensa::BI__builtin_xtensa_ae_s32ra64s_xp:
return {Intrinsic::xtensa_ae_s32ra64s_xp, 0, 0x50200};
case Xtensa::BI__builtin_xtensa_ae_s32x2_i:
return {Intrinsic::xtensa_ae_s32x2_i, 0, 0x70000};
case Xtensa::BI__builtin_xtensa_ae_s32x2_ip:
return {Intrinsic::xtensa_ae_s32x2_ip, 0, 0x50200};
case Xtensa::BI__builtin_xtensa_ae_s32x2_ric:
return {Intrinsic::xtensa_ae_s32x2_ric, 0, 0x10200};
case Xtensa::BI__builtin_xtensa_ae_s32x2_rip:
return {Intrinsic::xtensa_ae_s32x2_rip, 0, 0x10200};
case Xtensa::BI__builtin_xtensa_ae_s32x2_x:
return {Intrinsic::xtensa_ae_s32x2_x, 0, 0x70000};
case Xtensa::BI__builtin_xtensa_ae_s32x2_xc:
return {Intrinsic::xtensa_ae_s32x2_xc, 0, 0x50200};
case Xtensa::BI__builtin_xtensa_ae_s32x2_xp:
return {Intrinsic::xtensa_ae_s32x2_xp, 0, 0x50200};
case Xtensa::BI__builtin_xtensa_ae_s32x2f24_i:
return {Intrinsic::xtensa_ae_s32x2f24_i, 0, 0x70000};
case Xtensa::BI__builtin_xtensa_ae_s32x2f24_ip:
return {Intrinsic::xtensa_ae_s32x2f24_ip, 0, 0x50200};
case Xtensa::BI__builtin_xtensa_ae_s32x2f24_ric:
return {Intrinsic::xtensa_ae_s32x2f24_ric, 0, 0x10200};
case Xtensa::BI__builtin_xtensa_ae_s32x2f24_rip:
return {Intrinsic::xtensa_ae_s32x2f24_rip, 0, 0x10200};
case Xtensa::BI__builtin_xtensa_ae_s32x2f24_x:
return {Intrinsic::xtensa_ae_s32x2f24_x, 0, 0x70000};
case Xtensa::BI__builtin_xtensa_ae_s32x2f24_xc:
return {Intrinsic::xtensa_ae_s32x2f24_xc, 0, 0x50200};
case Xtensa::BI__builtin_xtensa_ae_s32x2f24_xp:
return {Intrinsic::xtensa_ae_s32x2f24_xp, 0, 0x50200};
case Xtensa::BI__builtin_xtensa_ae_s32x2ra64s_ip:
return {Intrinsic::xtensa_ae_s32x2ra64s_ip, 0, 0x30400};
case Xtensa::BI__builtin_xtensa_ae_s64_i:
return {Intrinsic::xtensa_ae_s64_i, 0, 0x70000};
case Xtensa::BI__builtin_xtensa_ae_s64_ip:
return {Intrinsic::xtensa_ae_s64_ip, 0, 0x50200};
case Xtensa::BI__builtin_xtensa_ae_s64_x:
return {Intrinsic::xtensa_ae_s64_x, 0, 0x70000};
case Xtensa::BI__builtin_xtensa_ae_s64_xc:
return {Intrinsic::xtensa_ae_s64_xc, 0, 0x50200};
case Xtensa::BI__builtin_xtensa_ae_s64_xp:
return {Intrinsic::xtensa_ae_s64_xp, 0, 0x50200};
case Xtensa::BI__builtin_xtensa_ae_sa16x4_ic:
return {Intrinsic::xtensa_ae_sa16x4_ic, 0, 0x10600};
case Xtensa::BI__builtin_xtensa_ae_sa16x4_ip:
return {Intrinsic::xtensa_ae_sa16x4_ip, 0, 0x10600};
case Xtensa::BI__builtin_xtensa_ae_sa16x4_ric:
return {Intrinsic::xtensa_ae_sa16x4_ric, 0, 0x10600};
case Xtensa::BI__builtin_xtensa_ae_sa16x4_rip:
return {Intrinsic::xtensa_ae_sa16x4_rip, 0, 0x10600};
case Xtensa::BI__builtin_xtensa_ae_sa24_l_ic:
return {Intrinsic::xtensa_ae_sa24_l_ic, 0, 0x10600};
case Xtensa::BI__builtin_xtensa_ae_sa24_l_ip:
return {Intrinsic::xtensa_ae_sa24_l_ip, 0, 0x10600};
case Xtensa::BI__builtin_xtensa_ae_sa24_l_ric:
return {Intrinsic::xtensa_ae_sa24_l_ric, 0, 0x10600};
case Xtensa::BI__builtin_xtensa_ae_sa24_l_rip:
return {Intrinsic::xtensa_ae_sa24_l_rip, 0, 0x10600};
case Xtensa::BI__builtin_xtensa_ae_sa24x2_ic:
return {Intrinsic::xtensa_ae_sa24x2_ic, 0, 0x10600};
case Xtensa::BI__builtin_xtensa_ae_sa24x2_ip:
return {Intrinsic::xtensa_ae_sa24x2_ip, 0, 0x10600};
case Xtensa::BI__builtin_xtensa_ae_sa24x2_ric:
return {Intrinsic::xtensa_ae_sa24x2_ric, 0, 0x10600};
case Xtensa::BI__builtin_xtensa_ae_sa24x2_rip:
return {Intrinsic::xtensa_ae_sa24x2_rip, 0, 0x10600};
case Xtensa::BI__builtin_xtensa_ae_sa32x2_ic:
return {Intrinsic::xtensa_ae_sa32x2_ic, 0, 0x10600};
case Xtensa::BI__builtin_xtensa_ae_sa32x2_ip:
return {Intrinsic::xtensa_ae_sa32x2_ip, 0, 0x10600};
case Xtensa::BI__builtin_xtensa_ae_sa32x2_ric:
return {Intrinsic::xtensa_ae_sa32x2_ric, 0, 0x10600};
case Xtensa::BI__builtin_xtensa_ae_sa32x2_rip:
return {Intrinsic::xtensa_ae_sa32x2_rip, 0, 0x10600};
case Xtensa::BI__builtin_xtensa_ae_sa32x2f24_ic:
return {Intrinsic::xtensa_ae_sa32x2f24_ic, 0, 0x10600};
case Xtensa::BI__builtin_xtensa_ae_sa32x2f24_ip:
return {Intrinsic::xtensa_ae_sa32x2f24_ip, 0, 0x10600};
case Xtensa::BI__builtin_xtensa_ae_sa32x2f24_ric:
return {Intrinsic::xtensa_ae_sa32x2f24_ric, 0, 0x10600};
case Xtensa::BI__builtin_xtensa_ae_sa32x2f24_rip:
return {Intrinsic::xtensa_ae_sa32x2f24_rip, 0, 0x10600};
case Xtensa::BI__builtin_xtensa_ae_sa64neg_fp:
return {Intrinsic::xtensa_ae_sa64neg_fp, 0, 0x20100};
case Xtensa::BI__builtin_xtensa_ae_sa64pos_fp:
return {Intrinsic::xtensa_ae_sa64pos_fp, 0, 0x20100};
case Xtensa::BI__builtin_xtensa_ae_salign64_i:
return {Intrinsic::xtensa_ae_salign64_i, 0, 0x70000};
case Xtensa::BI__builtin_xtensa_ae_sat16x4:
return {Intrinsic::xtensa_ae_sat16x4, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_sat24s:
return {Intrinsic::xtensa_ae_sat24s, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_sat48s:
return {Intrinsic::xtensa_ae_sat48s, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_satq56s:
return {Intrinsic::xtensa_ae_satq56s, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_sb:
return {Intrinsic::xtensa_ae_sb, 0, 0x20100};
case Xtensa::BI__builtin_xtensa_ae_sb_ic:
return {Intrinsic::xtensa_ae_sb_ic, 0, 0x20100};
case Xtensa::BI__builtin_xtensa_ae_sb_ip:
return {Intrinsic::xtensa_ae_sb_ip, 0, 0x20100};
case Xtensa::BI__builtin_xtensa_ae_sbf:
return {Intrinsic::xtensa_ae_sbf, 0, 0x100};
case Xtensa::BI__builtin_xtensa_ae_sbf_ic:
return {Intrinsic::xtensa_ae_sbf_ic, 0, 0x100};
case Xtensa::BI__builtin_xtensa_ae_sbf_ip:
return {Intrinsic::xtensa_ae_sbf_ip, 0, 0x100};
case Xtensa::BI__builtin_xtensa_ae_sbi:
return {Intrinsic::xtensa_ae_sbi, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_sbi_ic:
return {Intrinsic::xtensa_ae_sbi_ic, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_sbi_ip:
return {Intrinsic::xtensa_ae_sbi_ip, 0, 0x60100};
case Xtensa::BI__builtin_xtensa_ae_sel16i:
return {Intrinsic::xtensa_ae_sel16i, 1, 0xe0001};
case Xtensa::BI__builtin_xtensa_ae_sel16i_n:
return {Intrinsic::xtensa_ae_sel16i_n, 1, 0xe0001};
case Xtensa::BI__builtin_xtensa_ae_sext32:
return {Intrinsic::xtensa_ae_sext32, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_sext32x2d16_10:
return {Intrinsic::xtensa_ae_sext32x2d16_10, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_sext32x2d16_32:
return {Intrinsic::xtensa_ae_sext32x2d16_32, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_sha32:
return {Intrinsic::xtensa_ae_sha32, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_shortswap:
return {Intrinsic::xtensa_ae_shortswap, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_slaa16s:
return {Intrinsic::xtensa_ae_slaa16s, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_slaa32:
return {Intrinsic::xtensa_ae_slaa32, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_slaa32s:
return {Intrinsic::xtensa_ae_slaa32s, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_slaa64:
return {Intrinsic::xtensa_ae_slaa64, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_slaa64s:
return {Intrinsic::xtensa_ae_slaa64s, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_slaaq56:
return {Intrinsic::xtensa_ae_slaaq56, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_slai16s:
return {Intrinsic::xtensa_ae_slai16s, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_slai24:
return {Intrinsic::xtensa_ae_slai24, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_slai24s:
return {Intrinsic::xtensa_ae_slai24s, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_slai32:
return {Intrinsic::xtensa_ae_slai32, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_slai32s:
return {Intrinsic::xtensa_ae_slai32s, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_slai64:
return {Intrinsic::xtensa_ae_slai64, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_slai64s:
return {Intrinsic::xtensa_ae_slai64s, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_slaisq56s:
return {Intrinsic::xtensa_ae_slaisq56s, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_slas24:
return {Intrinsic::xtensa_ae_slas24, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_slas24s:
return {Intrinsic::xtensa_ae_slas24s, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_slas32:
return {Intrinsic::xtensa_ae_slas32, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_slas32s:
return {Intrinsic::xtensa_ae_slas32s, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_slas64:
return {Intrinsic::xtensa_ae_slas64, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_slas64s:
return {Intrinsic::xtensa_ae_slas64s, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_slasq56:
return {Intrinsic::xtensa_ae_slasq56, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_slassq56s:
return {Intrinsic::xtensa_ae_slassq56s, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_sra64_32:
return {Intrinsic::xtensa_ae_sra64_32, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_sraa16rs:
return {Intrinsic::xtensa_ae_sraa16rs, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_sraa16s:
return {Intrinsic::xtensa_ae_sraa16s, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_sraa32:
return {Intrinsic::xtensa_ae_sraa32, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_sraa32rs:
return {Intrinsic::xtensa_ae_sraa32rs, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_sraa32s:
return {Intrinsic::xtensa_ae_sraa32s, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_sraa64:
return {Intrinsic::xtensa_ae_sraa64, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_srai16:
return {Intrinsic::xtensa_ae_srai16, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_srai16r:
return {Intrinsic::xtensa_ae_srai16r, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_srai24:
return {Intrinsic::xtensa_ae_srai24, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_srai32:
return {Intrinsic::xtensa_ae_srai32, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_srai32r:
return {Intrinsic::xtensa_ae_srai32r, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_srai64:
return {Intrinsic::xtensa_ae_srai64, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_sras24:
return {Intrinsic::xtensa_ae_sras24, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_sras32:
return {Intrinsic::xtensa_ae_sras32, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_sras64:
return {Intrinsic::xtensa_ae_sras64, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_srla32:
return {Intrinsic::xtensa_ae_srla32, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_srla64:
return {Intrinsic::xtensa_ae_srla64, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_srli24:
return {Intrinsic::xtensa_ae_srli24, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_srli32:
return {Intrinsic::xtensa_ae_srli32, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_srli64:
return {Intrinsic::xtensa_ae_srli64, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_srls24:
return {Intrinsic::xtensa_ae_srls24, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_srls32:
return {Intrinsic::xtensa_ae_srls32, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_srls64:
return {Intrinsic::xtensa_ae_srls64, 1, 0x20001};
case Xtensa::BI__builtin_xtensa_ae_sub16:
return {Intrinsic::xtensa_ae_sub16, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_sub16s:
return {Intrinsic::xtensa_ae_sub16s, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_sub24s:
return {Intrinsic::xtensa_ae_sub24s, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_sub32:
return {Intrinsic::xtensa_ae_sub32, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_sub32s:
return {Intrinsic::xtensa_ae_sub32s, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_sub64:
return {Intrinsic::xtensa_ae_sub64, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_sub64s:
return {Intrinsic::xtensa_ae_sub64s, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_subadd32:
return {Intrinsic::xtensa_ae_subadd32, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_subadd32s:
return {Intrinsic::xtensa_ae_subadd32s, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_trunca32f64s_l:
return {Intrinsic::xtensa_ae_trunca32f64s_l, 1, 0xe0001};
case Xtensa::BI__builtin_xtensa_ae_trunca32x2f64s:
return {Intrinsic::xtensa_ae_trunca32x2f64s, 1, 0xe0001};
case Xtensa::BI__builtin_xtensa_ae_trunci32f64s_l:
return {Intrinsic::xtensa_ae_trunci32f64s_l, 1, 0xe0001};
case Xtensa::BI__builtin_xtensa_ae_trunci32x2f64s:
return {Intrinsic::xtensa_ae_trunci32x2f64s, 1, 0xe0001};
case Xtensa::BI__builtin_xtensa_ae_vldl16c:
return {Intrinsic::xtensa_ae_vldl16c, 0, 0x100};
case Xtensa::BI__builtin_xtensa_ae_vldl16c_ic:
return {Intrinsic::xtensa_ae_vldl16c_ic, 0, 0x100};
case Xtensa::BI__builtin_xtensa_ae_vldl16c_ip:
return {Intrinsic::xtensa_ae_vldl16c_ip, 0, 0x100};
case Xtensa::BI__builtin_xtensa_ae_vldl16t:
return {Intrinsic::xtensa_ae_vldl16t, 0, 0x40003};
case Xtensa::BI__builtin_xtensa_ae_vldl32t:
return {Intrinsic::xtensa_ae_vldl32t, 0, 0x40003};
case Xtensa::BI__builtin_xtensa_ae_vldsht:
return {Intrinsic::xtensa_ae_vldsht, 0, 0x10000};
case Xtensa::BI__builtin_xtensa_ae_vlel16t:
return {Intrinsic::xtensa_ae_vlel16t, 0, 0x40201};
case Xtensa::BI__builtin_xtensa_ae_vlel32t:
return {Intrinsic::xtensa_ae_vlel32t, 0, 0x40201};
case Xtensa::BI__builtin_xtensa_ae_vles16c:
return {Intrinsic::xtensa_ae_vles16c, 0, 0x100};
case Xtensa::BI__builtin_xtensa_ae_vles16c_ic:
return {Intrinsic::xtensa_ae_vles16c_ic, 0, 0x100};
case Xtensa::BI__builtin_xtensa_ae_vles16c_ip:
return {Intrinsic::xtensa_ae_vles16c_ip, 0, 0x100};
case Xtensa::BI__builtin_xtensa_ae_xor:
return {Intrinsic::xtensa_ae_xor, 1, 0x60001};
case Xtensa::BI__builtin_xtensa_ae_zalign64:
return {Intrinsic::xtensa_ae_zalign64, 1, 0x1};
case Xtensa::BI__builtin_xtensa_rur_ae_bithead:
return {Intrinsic::xtensa_rur_ae_bithead, 1, 0x1};
case Xtensa::BI__builtin_xtensa_rur_ae_bitptr:
return {Intrinsic::xtensa_rur_ae_bitptr, 1, 0x1};
case Xtensa::BI__builtin_xtensa_rur_ae_bitsused:
return {Intrinsic::xtensa_rur_ae_bitsused, 1, 0x1};
case Xtensa::BI__builtin_xtensa_rur_ae_cbegin0:
return {Intrinsic::xtensa_rur_ae_cbegin0, 1, 0x1};
case Xtensa::BI__builtin_xtensa_rur_ae_cend0:
return {Intrinsic::xtensa_rur_ae_cend0, 1, 0x1};
case Xtensa::BI__builtin_xtensa_rur_ae_cw_sd_no:
return {Intrinsic::xtensa_rur_ae_cw_sd_no, 1, 0x1};
case Xtensa::BI__builtin_xtensa_rur_ae_cwrap:
return {Intrinsic::xtensa_rur_ae_cwrap, 1, 0x1};
case Xtensa::BI__builtin_xtensa_rur_ae_first_ts:
return {Intrinsic::xtensa_rur_ae_first_ts, 1, 0x1};
case Xtensa::BI__builtin_xtensa_rur_ae_nextoffset:
return {Intrinsic::xtensa_rur_ae_nextoffset, 1, 0x1};
case Xtensa::BI__builtin_xtensa_rur_ae_overflow:
return {Intrinsic::xtensa_rur_ae_overflow, 1, 0x1};
case Xtensa::BI__builtin_xtensa_rur_ae_ovf_sar:
return {Intrinsic::xtensa_rur_ae_ovf_sar, 1, 0x1};
case Xtensa::BI__builtin_xtensa_rur_ae_sar:
return {Intrinsic::xtensa_rur_ae_sar, 1, 0x1};
case Xtensa::BI__builtin_xtensa_rur_ae_searchdone:
return {Intrinsic::xtensa_rur_ae_searchdone, 1, 0x1};
case Xtensa::BI__builtin_xtensa_rur_ae_tablesize:
return {Intrinsic::xtensa_rur_ae_tablesize, 1, 0x1};
case Xtensa::BI__builtin_xtensa_rur_ae_ts_fts_bu_bp:
return {Intrinsic::xtensa_rur_ae_ts_fts_bu_bp, 1, 0x1};
case Xtensa::BI__builtin_xtensa_wur_ae_bithead:
return {Intrinsic::xtensa_wur_ae_bithead, 0, 0x10000};
case Xtensa::BI__builtin_xtensa_wur_ae_bitptr:
return {Intrinsic::xtensa_wur_ae_bitptr, 0, 0x10000};
case Xtensa::BI__builtin_xtensa_wur_ae_bitsused:
return {Intrinsic::xtensa_wur_ae_bitsused, 0, 0x10000};
case Xtensa::BI__builtin_xtensa_wur_ae_cbegin0:
return {Intrinsic::xtensa_wur_ae_cbegin0, 0, 0x10000};
case Xtensa::BI__builtin_xtensa_wur_ae_cend0:
return {Intrinsic::xtensa_wur_ae_cend0, 0, 0x10000};
case Xtensa::BI__builtin_xtensa_wur_ae_cw_sd_no:
return {Intrinsic::xtensa_wur_ae_cw_sd_no, 0, 0x10000};
case Xtensa::BI__builtin_xtensa_wur_ae_cwrap:
return {Intrinsic::xtensa_wur_ae_cwrap, 0, 0x10000};
case Xtensa::BI__builtin_xtensa_wur_ae_first_ts:
return {Intrinsic::xtensa_wur_ae_first_ts, 0, 0x10000};
case Xtensa::BI__builtin_xtensa_wur_ae_nextoffset:
return {Intrinsic::xtensa_wur_ae_nextoffset, 0, 0x10000};
case Xtensa::BI__builtin_xtensa_wur_ae_overflow:
return {Intrinsic::xtensa_wur_ae_overflow, 0, 0x10000};
case Xtensa::BI__builtin_xtensa_wur_ae_ovf_sar:
return {Intrinsic::xtensa_wur_ae_ovf_sar, 0, 0x10000};
case Xtensa::BI__builtin_xtensa_wur_ae_sar:
return {Intrinsic::xtensa_wur_ae_sar, 0, 0x10000};
case Xtensa::BI__builtin_xtensa_wur_ae_searchdone:
return {Intrinsic::xtensa_wur_ae_searchdone, 0, 0x10000};
case Xtensa::BI__builtin_xtensa_wur_ae_tablesize:
return {Intrinsic::xtensa_wur_ae_tablesize, 0, 0x10000};
case Xtensa::BI__builtin_xtensa_wur_ae_ts_fts_bu_bp:
return {Intrinsic::xtensa_wur_ae_ts_fts_bu_bp, 0, 0x10000};
