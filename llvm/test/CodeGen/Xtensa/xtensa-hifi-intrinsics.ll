; RUN: split-file %s %t
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_abs16s.ll -o - | FileCheck %t/ae_abs16s.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_abs24s.ll -o - | FileCheck %t/ae_abs24s.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_abs32.ll -o - | FileCheck %t/ae_abs32.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_abs32s.ll -o - | FileCheck %t/ae_abs32s.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_abs64.ll -o - | FileCheck %t/ae_abs64.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_abs64s.ll -o - | FileCheck %t/ae_abs64s.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_add16.ll -o - | FileCheck %t/ae_add16.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_add16s.ll -o - | FileCheck %t/ae_add16s.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_add24s.ll -o - | FileCheck %t/ae_add24s.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_add32.ll -o - | FileCheck %t/ae_add32.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_add32_hl_lh.ll -o - | FileCheck %t/ae_add32_hl_lh.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_add32s.ll -o - | FileCheck %t/ae_add32s.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_add64.ll -o - | FileCheck %t/ae_add64.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_add64s.ll -o - | FileCheck %t/ae_add64s.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_addbrba32.ll -o - | FileCheck %t/ae_addbrba32.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_addsub32.ll -o - | FileCheck %t/ae_addsub32.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_addsub32s.ll -o - | FileCheck %t/ae_addsub32s.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_and.ll -o - | FileCheck %t/ae_and.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_cvt32x2f16_10.ll -o - | FileCheck %t/ae_cvt32x2f16_10.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_cvt32x2f16_32.ll -o - | FileCheck %t/ae_cvt32x2f16_32.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_cvt48a32.ll -o - | FileCheck %t/ae_cvt48a32.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_cvt64a32.ll -o - | FileCheck %t/ae_cvt64a32.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_cvt64f32_h.ll -o - | FileCheck %t/ae_cvt64f32_h.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_cvta32f24s_h.ll -o - | FileCheck %t/ae_cvta32f24s_h.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_cvta32f24s_l.ll -o - | FileCheck %t/ae_cvta32f24s_l.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_cvtq56a32s.ll -o - | FileCheck %t/ae_cvtq56a32s.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_cvtq56p32s_h.ll -o - | FileCheck %t/ae_cvtq56p32s_h.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_cvtq56p32s_l.ll -o - | FileCheck %t/ae_cvtq56p32s_l.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_db.ll -o - | FileCheck %t/ae_db.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_db_ic.ll -o - | FileCheck %t/ae_db_ic.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_db_ip.ll -o - | FileCheck %t/ae_db_ip.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_dbi.ll -o - | FileCheck %t/ae_dbi.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_dbi_ic.ll -o - | FileCheck %t/ae_dbi_ic.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_dbi_ip.ll -o - | FileCheck %t/ae_dbi_ip.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_div64d32_h.ll -o - | FileCheck %t/ae_div64d32_h.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_div64d32_l.ll -o - | FileCheck %t/ae_div64d32_l.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_eq64.ll -o - | FileCheck %t/ae_eq64.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l16_i.ll -o - | FileCheck %t/ae_l16_i.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l16_ip.ll -o - | FileCheck %t/ae_l16_ip.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l16_x.ll -o - | FileCheck %t/ae_l16_x.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l16_xc.ll -o - | FileCheck %t/ae_l16_xc.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l16_xp.ll -o - | FileCheck %t/ae_l16_xp.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l16m_i.ll -o - | FileCheck %t/ae_l16m_i.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l16m_iu.ll -o - | FileCheck %t/ae_l16m_iu.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l16m_x.ll -o - | FileCheck %t/ae_l16m_x.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l16m_xc.ll -o - | FileCheck %t/ae_l16m_xc.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l16m_xu.ll -o - | FileCheck %t/ae_l16m_xu.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l16x2m_i.ll -o - | FileCheck %t/ae_l16x2m_i.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l16x2m_iu.ll -o - | FileCheck %t/ae_l16x2m_iu.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l16x2m_x.ll -o - | FileCheck %t/ae_l16x2m_x.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l16x2m_xc.ll -o - | FileCheck %t/ae_l16x2m_xc.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l16x2m_xu.ll -o - | FileCheck %t/ae_l16x2m_xu.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l16x4_i.ll -o - | FileCheck %t/ae_l16x4_i.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l16x4_ip.ll -o - | FileCheck %t/ae_l16x4_ip.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l16x4_ric.ll -o - | FileCheck %t/ae_l16x4_ric.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l16x4_rip.ll -o - | FileCheck %t/ae_l16x4_rip.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l16x4_x.ll -o - | FileCheck %t/ae_l16x4_x.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l16x4_xc.ll -o - | FileCheck %t/ae_l16x4_xc.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l16x4_xp.ll -o - | FileCheck %t/ae_l16x4_xp.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l32_i.ll -o - | FileCheck %t/ae_l32_i.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l32_ip.ll -o - | FileCheck %t/ae_l32_ip.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l32_x.ll -o - | FileCheck %t/ae_l32_x.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l32_xc.ll -o - | FileCheck %t/ae_l32_xc.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l32_xp.ll -o - | FileCheck %t/ae_l32_xp.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l32f24_i.ll -o - | FileCheck %t/ae_l32f24_i.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l32f24_ip.ll -o - | FileCheck %t/ae_l32f24_ip.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l32f24_x.ll -o - | FileCheck %t/ae_l32f24_x.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l32f24_xc.ll -o - | FileCheck %t/ae_l32f24_xc.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l32f24_xp.ll -o - | FileCheck %t/ae_l32f24_xp.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l32m_i.ll -o - | FileCheck %t/ae_l32m_i.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l32m_iu.ll -o - | FileCheck %t/ae_l32m_iu.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l32m_x.ll -o - | FileCheck %t/ae_l32m_x.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l32m_xc.ll -o - | FileCheck %t/ae_l32m_xc.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l32m_xu.ll -o - | FileCheck %t/ae_l32m_xu.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l32x2_i.ll -o - | FileCheck %t/ae_l32x2_i.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l32x2_ip.ll -o - | FileCheck %t/ae_l32x2_ip.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l32x2_ric.ll -o - | FileCheck %t/ae_l32x2_ric.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l32x2_rip.ll -o - | FileCheck %t/ae_l32x2_rip.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l32x2_x.ll -o - | FileCheck %t/ae_l32x2_x.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l32x2_xc.ll -o - | FileCheck %t/ae_l32x2_xc.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l32x2_xp.ll -o - | FileCheck %t/ae_l32x2_xp.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l32x2f24_i.ll -o - | FileCheck %t/ae_l32x2f24_i.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l32x2f24_ip.ll -o - | FileCheck %t/ae_l32x2f24_ip.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l32x2f24_ric.ll -o - | FileCheck %t/ae_l32x2f24_ric.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l32x2f24_rip.ll -o - | FileCheck %t/ae_l32x2f24_rip.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l32x2f24_x.ll -o - | FileCheck %t/ae_l32x2f24_x.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l32x2f24_xc.ll -o - | FileCheck %t/ae_l32x2f24_xc.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l32x2f24_xp.ll -o - | FileCheck %t/ae_l32x2f24_xp.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l64_i.ll -o - | FileCheck %t/ae_l64_i.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l64_ip.ll -o - | FileCheck %t/ae_l64_ip.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l64_x.ll -o - | FileCheck %t/ae_l64_x.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l64_xc.ll -o - | FileCheck %t/ae_l64_xc.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_l64_xp.ll -o - | FileCheck %t/ae_l64_xp.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_la16x4_ic.ll -o - | FileCheck %t/ae_la16x4_ic.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_la16x4_ip.ll -o - | FileCheck %t/ae_la16x4_ip.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_la16x4_ric.ll -o - | FileCheck %t/ae_la16x4_ric.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_la16x4_rip.ll -o - | FileCheck %t/ae_la16x4_rip.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_la16x4neg_pc.ll -o - | FileCheck %t/ae_la16x4neg_pc.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_la16x4pos_pc.ll -o - | FileCheck %t/ae_la16x4pos_pc.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_la24_ic.ll -o - | FileCheck %t/ae_la24_ic.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_la24_ip.ll -o - | FileCheck %t/ae_la24_ip.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_la24_ric.ll -o - | FileCheck %t/ae_la24_ric.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_la24_rip.ll -o - | FileCheck %t/ae_la24_rip.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_la24neg_pc.ll -o - | FileCheck %t/ae_la24neg_pc.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_la24pos_pc.ll -o - | FileCheck %t/ae_la24pos_pc.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_la24x2_ic.ll -o - | FileCheck %t/ae_la24x2_ic.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_la24x2_ip.ll -o - | FileCheck %t/ae_la24x2_ip.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_la24x2_ric.ll -o - | FileCheck %t/ae_la24x2_ric.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_la24x2_rip.ll -o - | FileCheck %t/ae_la24x2_rip.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_la24x2neg_pc.ll -o - | FileCheck %t/ae_la24x2neg_pc.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_la24x2pos_pc.ll -o - | FileCheck %t/ae_la24x2pos_pc.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_la32x2_ic.ll -o - | FileCheck %t/ae_la32x2_ic.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_la32x2_ip.ll -o - | FileCheck %t/ae_la32x2_ip.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_la32x2_ric.ll -o - | FileCheck %t/ae_la32x2_ric.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_la32x2_rip.ll -o - | FileCheck %t/ae_la32x2_rip.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_la32x2f24_ic.ll -o - | FileCheck %t/ae_la32x2f24_ic.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_la32x2f24_ip.ll -o - | FileCheck %t/ae_la32x2f24_ip.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_la32x2f24_ric.ll -o - | FileCheck %t/ae_la32x2f24_ric.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_la32x2f24_rip.ll -o - | FileCheck %t/ae_la32x2f24_rip.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_la32x2neg_pc.ll -o - | FileCheck %t/ae_la32x2neg_pc.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_la32x2pos_pc.ll -o - | FileCheck %t/ae_la32x2pos_pc.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_la64_pp.ll -o - | FileCheck %t/ae_la64_pp.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_lalign64_i.ll -o - | FileCheck %t/ae_lalign64_i.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_lb.ll -o - | FileCheck %t/ae_lb.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_lbi.ll -o - | FileCheck %t/ae_lbi.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_lbk.ll -o - | FileCheck %t/ae_lbk.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_lbki.ll -o - | FileCheck %t/ae_lbki.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_lbs.ll -o - | FileCheck %t/ae_lbs.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_lbsi.ll -o - | FileCheck %t/ae_lbsi.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_le64.ll -o - | FileCheck %t/ae_le64.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_lt64.ll -o - | FileCheck %t/ae_lt64.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_max32.ll -o - | FileCheck %t/ae_max32.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_max64.ll -o - | FileCheck %t/ae_max64.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_maxabs32s.ll -o - | FileCheck %t/ae_maxabs32s.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_maxabs64s.ll -o - | FileCheck %t/ae_maxabs64s.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_min32.ll -o - | FileCheck %t/ae_min32.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_min64.ll -o - | FileCheck %t/ae_min64.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_minabs32s.ll -o - | FileCheck %t/ae_minabs32s.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_minabs64s.ll -o - | FileCheck %t/ae_minabs64s.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mov.ll -o - | FileCheck %t/ae_mov.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_movad16_0.ll -o - | FileCheck %t/ae_movad16_0.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_movad16_1.ll -o - | FileCheck %t/ae_movad16_1.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_movad16_2.ll -o - | FileCheck %t/ae_movad16_2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_movad16_3.ll -o - | FileCheck %t/ae_movad16_3.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_movad32_h.ll -o - | FileCheck %t/ae_movad32_h.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_movad32_l.ll -o - | FileCheck %t/ae_movad32_l.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_movalign.ll -o - | FileCheck %t/ae_movalign.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_movda16.ll -o - | FileCheck %t/ae_movda16.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_movda16x2.ll -o - | FileCheck %t/ae_movda16x2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_movda32.ll -o - | FileCheck %t/ae_movda32.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_movda32x2.ll -o - | FileCheck %t/ae_movda32x2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_movf64.ll -o - | FileCheck %t/ae_movf64.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_movi.ll -o - | FileCheck %t/ae_movi.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_movt64.ll -o - | FileCheck %t/ae_movt64.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mul16x4.ll -o - | FileCheck %t/ae_mul16x4.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mul32_hh.ll -o - | FileCheck %t/ae_mul32_hh.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mul32_lh.ll -o - | FileCheck %t/ae_mul32_lh.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mul32_ll.ll -o - | FileCheck %t/ae_mul32_ll.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mul32_ll_s2.ll -o - | FileCheck %t/ae_mul32_ll_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mul32u_ll.ll -o - | FileCheck %t/ae_mul32u_ll.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mul32x16_h0.ll -o - | FileCheck %t/ae_mul32x16_h0.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mul32x16_h0_s2.ll -o - | FileCheck %t/ae_mul32x16_h0_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mul32x16_h1.ll -o - | FileCheck %t/ae_mul32x16_h1.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mul32x16_h1_s2.ll -o - | FileCheck %t/ae_mul32x16_h1_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mul32x16_h2.ll -o - | FileCheck %t/ae_mul32x16_h2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mul32x16_h2_s2.ll -o - | FileCheck %t/ae_mul32x16_h2_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mul32x16_h3.ll -o - | FileCheck %t/ae_mul32x16_h3.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mul32x16_h3_s2.ll -o - | FileCheck %t/ae_mul32x16_h3_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mul32x16_l0.ll -o - | FileCheck %t/ae_mul32x16_l0.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mul32x16_l0_s2.ll -o - | FileCheck %t/ae_mul32x16_l0_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mul32x16_l1.ll -o - | FileCheck %t/ae_mul32x16_l1.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mul32x16_l1_s2.ll -o - | FileCheck %t/ae_mul32x16_l1_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mul32x16_l2.ll -o - | FileCheck %t/ae_mul32x16_l2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mul32x16_l2_s2.ll -o - | FileCheck %t/ae_mul32x16_l2_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mul32x16_l3.ll -o - | FileCheck %t/ae_mul32x16_l3.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mul32x16_l3_s2.ll -o - | FileCheck %t/ae_mul32x16_l3_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mula16x4.ll -o - | FileCheck %t/ae_mula16x4.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mula32_hh.ll -o - | FileCheck %t/ae_mula32_hh.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mula32_lh.ll -o - | FileCheck %t/ae_mula32_lh.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mula32_ll.ll -o - | FileCheck %t/ae_mula32_ll.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mula32_ll_s2.ll -o - | FileCheck %t/ae_mula32_ll_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mula32u_ll.ll -o - | FileCheck %t/ae_mula32u_ll.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mula32x16_h0.ll -o - | FileCheck %t/ae_mula32x16_h0.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mula32x16_h0_s2.ll -o - | FileCheck %t/ae_mula32x16_h0_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mula32x16_h1.ll -o - | FileCheck %t/ae_mula32x16_h1.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mula32x16_h1_s2.ll -o - | FileCheck %t/ae_mula32x16_h1_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mula32x16_h2.ll -o - | FileCheck %t/ae_mula32x16_h2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mula32x16_h2_s2.ll -o - | FileCheck %t/ae_mula32x16_h2_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mula32x16_h3.ll -o - | FileCheck %t/ae_mula32x16_h3.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mula32x16_h3_s2.ll -o - | FileCheck %t/ae_mula32x16_h3_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mula32x16_l0.ll -o - | FileCheck %t/ae_mula32x16_l0.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mula32x16_l0_s2.ll -o - | FileCheck %t/ae_mula32x16_l0_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mula32x16_l1.ll -o - | FileCheck %t/ae_mula32x16_l1.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mula32x16_l1_s2.ll -o - | FileCheck %t/ae_mula32x16_l1_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mula32x16_l2.ll -o - | FileCheck %t/ae_mula32x16_l2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mula32x16_l2_s2.ll -o - | FileCheck %t/ae_mula32x16_l2_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mula32x16_l3.ll -o - | FileCheck %t/ae_mula32x16_l3.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mula32x16_l3_s2.ll -o - | FileCheck %t/ae_mula32x16_l3_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaad24_hh_ll.ll -o - | FileCheck %t/ae_mulaad24_hh_ll.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaad24_hh_ll_s2.ll -o - | FileCheck %t/ae_mulaad24_hh_ll_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaad24_hl_lh.ll -o - | FileCheck %t/ae_mulaad24_hl_lh.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaad24_hl_lh_s2.ll -o - | FileCheck %t/ae_mulaad24_hl_lh_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaad32x16_h0_l1.ll -o - | FileCheck %t/ae_mulaad32x16_h0_l1.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaad32x16_h0_l1_s2.ll -o - | FileCheck %t/ae_mulaad32x16_h0_l1_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaad32x16_h1_l0.ll -o - | FileCheck %t/ae_mulaad32x16_h1_l0.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaad32x16_h1_l0_s2.ll -o - | FileCheck %t/ae_mulaad32x16_h1_l0_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaad32x16_h2_l3.ll -o - | FileCheck %t/ae_mulaad32x16_h2_l3.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaad32x16_h2_l3_s2.ll -o - | FileCheck %t/ae_mulaad32x16_h2_l3_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaad32x16_h3_l2.ll -o - | FileCheck %t/ae_mulaad32x16_h3_l2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaad32x16_h3_l2_s2.ll -o - | FileCheck %t/ae_mulaad32x16_h3_l2_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaafd16ss_11_00.ll -o - | FileCheck %t/ae_mulaafd16ss_11_00.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaafd16ss_11_00_s2.ll -o - | FileCheck %t/ae_mulaafd16ss_11_00_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaafd16ss_13_02.ll -o - | FileCheck %t/ae_mulaafd16ss_13_02.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaafd16ss_13_02_s2.ll -o - | FileCheck %t/ae_mulaafd16ss_13_02_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaafd16ss_33_22.ll -o - | FileCheck %t/ae_mulaafd16ss_33_22.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaafd16ss_33_22_s2.ll -o - | FileCheck %t/ae_mulaafd16ss_33_22_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaafd24_hh_ll.ll -o - | FileCheck %t/ae_mulaafd24_hh_ll.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaafd24_hh_ll_s2.ll -o - | FileCheck %t/ae_mulaafd24_hh_ll_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaafd24_hl_lh.ll -o - | FileCheck %t/ae_mulaafd24_hl_lh.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaafd24_hl_lh_s2.ll -o - | FileCheck %t/ae_mulaafd24_hl_lh_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaafd32x16_h0_l1.ll -o - | FileCheck %t/ae_mulaafd32x16_h0_l1.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaafd32x16_h0_l1_s2.ll -o - | FileCheck %t/ae_mulaafd32x16_h0_l1_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaafd32x16_h1_l0.ll -o - | FileCheck %t/ae_mulaafd32x16_h1_l0.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaafd32x16_h1_l0_s2.ll -o - | FileCheck %t/ae_mulaafd32x16_h1_l0_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaafd32x16_h2_l3.ll -o - | FileCheck %t/ae_mulaafd32x16_h2_l3.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaafd32x16_h2_l3_s2.ll -o - | FileCheck %t/ae_mulaafd32x16_h2_l3_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaafd32x16_h3_l2.ll -o - | FileCheck %t/ae_mulaafd32x16_h3_l2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaafd32x16_h3_l2_s2.ll -o - | FileCheck %t/ae_mulaafd32x16_h3_l2_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulac24.ll -o - | FileCheck %t/ae_mulac24.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulac32x16_h.ll -o - | FileCheck %t/ae_mulac32x16_h.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulac32x16_l.ll -o - | FileCheck %t/ae_mulac32x16_l.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaf16ss_00.ll -o - | FileCheck %t/ae_mulaf16ss_00.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaf16ss_00_s2.ll -o - | FileCheck %t/ae_mulaf16ss_00_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaf16ss_10.ll -o - | FileCheck %t/ae_mulaf16ss_10.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaf16ss_11.ll -o - | FileCheck %t/ae_mulaf16ss_11.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaf16ss_20.ll -o - | FileCheck %t/ae_mulaf16ss_20.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaf16ss_21.ll -o - | FileCheck %t/ae_mulaf16ss_21.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaf16ss_22.ll -o - | FileCheck %t/ae_mulaf16ss_22.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaf16ss_30.ll -o - | FileCheck %t/ae_mulaf16ss_30.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaf16ss_31.ll -o - | FileCheck %t/ae_mulaf16ss_31.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaf16ss_32.ll -o - | FileCheck %t/ae_mulaf16ss_32.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaf16ss_33.ll -o - | FileCheck %t/ae_mulaf16ss_33.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaf16x4ss.ll -o - | FileCheck %t/ae_mulaf16x4ss.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaf32r_hh.ll -o - | FileCheck %t/ae_mulaf32r_hh.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaf32r_lh.ll -o - | FileCheck %t/ae_mulaf32r_lh.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaf32r_ll.ll -o - | FileCheck %t/ae_mulaf32r_ll.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaf32r_ll_s2.ll -o - | FileCheck %t/ae_mulaf32r_ll_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaf32s_hh.ll -o - | FileCheck %t/ae_mulaf32s_hh.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaf32s_lh.ll -o - | FileCheck %t/ae_mulaf32s_lh.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaf32s_ll.ll -o - | FileCheck %t/ae_mulaf32s_ll.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaf32s_ll_s2.ll -o - | FileCheck %t/ae_mulaf32s_ll_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaf32x16_h0.ll -o - | FileCheck %t/ae_mulaf32x16_h0.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaf32x16_h0_s2.ll -o - | FileCheck %t/ae_mulaf32x16_h0_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaf32x16_h1.ll -o - | FileCheck %t/ae_mulaf32x16_h1.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaf32x16_h1_s2.ll -o - | FileCheck %t/ae_mulaf32x16_h1_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaf32x16_h2.ll -o - | FileCheck %t/ae_mulaf32x16_h2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaf32x16_h2_s2.ll -o - | FileCheck %t/ae_mulaf32x16_h2_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaf32x16_h3.ll -o - | FileCheck %t/ae_mulaf32x16_h3.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaf32x16_h3_s2.ll -o - | FileCheck %t/ae_mulaf32x16_h3_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaf32x16_l0.ll -o - | FileCheck %t/ae_mulaf32x16_l0.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaf32x16_l0_s2.ll -o - | FileCheck %t/ae_mulaf32x16_l0_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaf32x16_l1.ll -o - | FileCheck %t/ae_mulaf32x16_l1.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaf32x16_l1_s2.ll -o - | FileCheck %t/ae_mulaf32x16_l1_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaf32x16_l2.ll -o - | FileCheck %t/ae_mulaf32x16_l2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaf32x16_l2_s2.ll -o - | FileCheck %t/ae_mulaf32x16_l2_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaf32x16_l3.ll -o - | FileCheck %t/ae_mulaf32x16_l3.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaf32x16_l3_s2.ll -o - | FileCheck %t/ae_mulaf32x16_l3_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaf48q32sp16s_l.ll -o - | FileCheck %t/ae_mulaf48q32sp16s_l.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaf48q32sp16s_l_s2.ll -o - | FileCheck %t/ae_mulaf48q32sp16s_l_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaf48q32sp16u_l.ll -o - | FileCheck %t/ae_mulaf48q32sp16u_l.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaf48q32sp16u_l_s2.ll -o - | FileCheck %t/ae_mulaf48q32sp16u_l_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulafc24ra.ll -o - | FileCheck %t/ae_mulafc24ra.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulafc32x16ras_h.ll -o - | FileCheck %t/ae_mulafc32x16ras_h.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulafc32x16ras_l.ll -o - | FileCheck %t/ae_mulafc32x16ras_l.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulafd24x2_fir_h.ll -o - | FileCheck %t/ae_mulafd24x2_fir_h.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulafd24x2_fir_l.ll -o - | FileCheck %t/ae_mulafd24x2_fir_l.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulafd32x16x2_fir_hh.ll -o - | FileCheck %t/ae_mulafd32x16x2_fir_hh.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulafd32x16x2_fir_hl.ll -o - | FileCheck %t/ae_mulafd32x16x2_fir_hl.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulafd32x16x2_fir_lh.ll -o - | FileCheck %t/ae_mulafd32x16x2_fir_lh.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulafd32x16x2_fir_ll.ll -o - | FileCheck %t/ae_mulafd32x16x2_fir_ll.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulafp24x2r.ll -o - | FileCheck %t/ae_mulafp24x2r.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulafp24x2r_s2.ll -o - | FileCheck %t/ae_mulafp24x2r_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulafp24x2ra.ll -o - | FileCheck %t/ae_mulafp24x2ra.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulafp24x2ra_s2.ll -o - | FileCheck %t/ae_mulafp24x2ra_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulafp32x16x2ras_h.ll -o - | FileCheck %t/ae_mulafp32x16x2ras_h.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulafp32x16x2ras_h_s2.ll -o - | FileCheck %t/ae_mulafp32x16x2ras_h_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulafp32x16x2ras_l.ll -o - | FileCheck %t/ae_mulafp32x16x2ras_l.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulafp32x16x2ras_l_s2.ll -o - | FileCheck %t/ae_mulafp32x16x2ras_l_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulafp32x16x2rs_h.ll -o - | FileCheck %t/ae_mulafp32x16x2rs_h.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulafp32x16x2rs_h_s2.ll -o - | FileCheck %t/ae_mulafp32x16x2rs_h_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulafp32x16x2rs_l.ll -o - | FileCheck %t/ae_mulafp32x16x2rs_l.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulafp32x16x2rs_l_s2.ll -o - | FileCheck %t/ae_mulafp32x16x2rs_l_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulafp32x2ras.ll -o - | FileCheck %t/ae_mulafp32x2ras.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulafp32x2rs.ll -o - | FileCheck %t/ae_mulafp32x2rs.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulafq32sp24s_h_s2.ll -o - | FileCheck %t/ae_mulafq32sp24s_h_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulafq32sp24s_l_s2.ll -o - | FileCheck %t/ae_mulafq32sp24s_l_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulap24x2.ll -o - | FileCheck %t/ae_mulap24x2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulap24x2_s2.ll -o - | FileCheck %t/ae_mulap24x2_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulap32x16x2_h.ll -o - | FileCheck %t/ae_mulap32x16x2_h.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulap32x16x2_l.ll -o - | FileCheck %t/ae_mulap32x16x2_l.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulap32x2.ll -o - | FileCheck %t/ae_mulap32x2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaq32sp16s_l_s2.ll -o - | FileCheck %t/ae_mulaq32sp16s_l_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulaq32sp16u_l_s2.ll -o - | FileCheck %t/ae_mulaq32sp16u_l_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mularfq32sp24s_h_s2.ll -o - | FileCheck %t/ae_mularfq32sp24s_h_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mularfq32sp24s_l_s2.ll -o - | FileCheck %t/ae_mularfq32sp24s_l_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulas32f48p16s_hh.ll -o - | FileCheck %t/ae_mulas32f48p16s_hh.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulas32f48p16s_hh_s2.ll -o - | FileCheck %t/ae_mulas32f48p16s_hh_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulas32f48p16s_lh.ll -o - | FileCheck %t/ae_mulas32f48p16s_lh.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulas32f48p16s_lh_s2.ll -o - | FileCheck %t/ae_mulas32f48p16s_lh_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulas32f48p16s_ll.ll -o - | FileCheck %t/ae_mulas32f48p16s_ll.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulas32f48p16s_ll_s2.ll -o - | FileCheck %t/ae_mulas32f48p16s_ll_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulasd24_hh_ll.ll -o - | FileCheck %t/ae_mulasd24_hh_ll.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulasd24_hh_ll_s2.ll -o - | FileCheck %t/ae_mulasd24_hh_ll_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulasd24_hl_lh.ll -o - | FileCheck %t/ae_mulasd24_hl_lh.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulasd24_hl_lh_s2.ll -o - | FileCheck %t/ae_mulasd24_hl_lh_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulasd32x16_h1_l0.ll -o - | FileCheck %t/ae_mulasd32x16_h1_l0.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulasd32x16_h1_l0_s2.ll -o - | FileCheck %t/ae_mulasd32x16_h1_l0_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulasd32x16_h3_l2.ll -o - | FileCheck %t/ae_mulasd32x16_h3_l2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulasd32x16_h3_l2_s2.ll -o - | FileCheck %t/ae_mulasd32x16_h3_l2_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulasfd24_hh_ll.ll -o - | FileCheck %t/ae_mulasfd24_hh_ll.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulasfd24_hh_ll_s2.ll -o - | FileCheck %t/ae_mulasfd24_hh_ll_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulasfd24_hl_lh.ll -o - | FileCheck %t/ae_mulasfd24_hl_lh.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulasfd24_hl_lh_s2.ll -o - | FileCheck %t/ae_mulasfd24_hl_lh_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulasfd32x16_h1_l0.ll -o - | FileCheck %t/ae_mulasfd32x16_h1_l0.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulasfd32x16_h1_l0_s2.ll -o - | FileCheck %t/ae_mulasfd32x16_h1_l0_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulasfd32x16_h3_l2.ll -o - | FileCheck %t/ae_mulasfd32x16_h3_l2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulasfd32x16_h3_l2_s2.ll -o - | FileCheck %t/ae_mulasfd32x16_h3_l2_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulc24.ll -o - | FileCheck %t/ae_mulc24.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulc32x16_h.ll -o - | FileCheck %t/ae_mulc32x16_h.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulc32x16_l.ll -o - | FileCheck %t/ae_mulc32x16_l.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulf16ss_00.ll -o - | FileCheck %t/ae_mulf16ss_00.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulf16ss_00_s2.ll -o - | FileCheck %t/ae_mulf16ss_00_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulf16ss_10.ll -o - | FileCheck %t/ae_mulf16ss_10.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulf16ss_11.ll -o - | FileCheck %t/ae_mulf16ss_11.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulf16ss_20.ll -o - | FileCheck %t/ae_mulf16ss_20.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulf16ss_21.ll -o - | FileCheck %t/ae_mulf16ss_21.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulf16ss_22.ll -o - | FileCheck %t/ae_mulf16ss_22.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulf16ss_30.ll -o - | FileCheck %t/ae_mulf16ss_30.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulf16ss_31.ll -o - | FileCheck %t/ae_mulf16ss_31.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulf16ss_32.ll -o - | FileCheck %t/ae_mulf16ss_32.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulf16ss_33.ll -o - | FileCheck %t/ae_mulf16ss_33.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulf16x4ss.ll -o - | FileCheck %t/ae_mulf16x4ss.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulf32r_hh.ll -o - | FileCheck %t/ae_mulf32r_hh.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulf32r_lh.ll -o - | FileCheck %t/ae_mulf32r_lh.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulf32r_ll.ll -o - | FileCheck %t/ae_mulf32r_ll.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulf32r_ll_s2.ll -o - | FileCheck %t/ae_mulf32r_ll_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulf32s_hh.ll -o - | FileCheck %t/ae_mulf32s_hh.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulf32s_lh.ll -o - | FileCheck %t/ae_mulf32s_lh.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulf32s_ll.ll -o - | FileCheck %t/ae_mulf32s_ll.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulf32s_ll_s2.ll -o - | FileCheck %t/ae_mulf32s_ll_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulf32x16_h0.ll -o - | FileCheck %t/ae_mulf32x16_h0.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulf32x16_h0_s2.ll -o - | FileCheck %t/ae_mulf32x16_h0_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulf32x16_h1.ll -o - | FileCheck %t/ae_mulf32x16_h1.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulf32x16_h1_s2.ll -o - | FileCheck %t/ae_mulf32x16_h1_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulf32x16_h2.ll -o - | FileCheck %t/ae_mulf32x16_h2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulf32x16_h2_s2.ll -o - | FileCheck %t/ae_mulf32x16_h2_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulf32x16_h3.ll -o - | FileCheck %t/ae_mulf32x16_h3.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulf32x16_h3_s2.ll -o - | FileCheck %t/ae_mulf32x16_h3_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulf32x16_l0.ll -o - | FileCheck %t/ae_mulf32x16_l0.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulf32x16_l0_s2.ll -o - | FileCheck %t/ae_mulf32x16_l0_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulf32x16_l1.ll -o - | FileCheck %t/ae_mulf32x16_l1.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulf32x16_l1_s2.ll -o - | FileCheck %t/ae_mulf32x16_l1_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulf32x16_l2.ll -o - | FileCheck %t/ae_mulf32x16_l2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulf32x16_l2_s2.ll -o - | FileCheck %t/ae_mulf32x16_l2_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulf32x16_l3.ll -o - | FileCheck %t/ae_mulf32x16_l3.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulf32x16_l3_s2.ll -o - | FileCheck %t/ae_mulf32x16_l3_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulf48q32sp16s_l.ll -o - | FileCheck %t/ae_mulf48q32sp16s_l.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulf48q32sp16s_l_s2.ll -o - | FileCheck %t/ae_mulf48q32sp16s_l_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulf48q32sp16u_l.ll -o - | FileCheck %t/ae_mulf48q32sp16u_l.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulf48q32sp16u_l_s2.ll -o - | FileCheck %t/ae_mulf48q32sp16u_l_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulfc24ra.ll -o - | FileCheck %t/ae_mulfc24ra.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulfc32x16ras_h.ll -o - | FileCheck %t/ae_mulfc32x16ras_h.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulfc32x16ras_l.ll -o - | FileCheck %t/ae_mulfc32x16ras_l.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulfd24x2_fir_h.ll -o - | FileCheck %t/ae_mulfd24x2_fir_h.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulfd24x2_fir_l.ll -o - | FileCheck %t/ae_mulfd24x2_fir_l.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulfd32x16x2_fir_hh.ll -o - | FileCheck %t/ae_mulfd32x16x2_fir_hh.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulfd32x16x2_fir_hl.ll -o - | FileCheck %t/ae_mulfd32x16x2_fir_hl.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulfd32x16x2_fir_lh.ll -o - | FileCheck %t/ae_mulfd32x16x2_fir_lh.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulfd32x16x2_fir_ll.ll -o - | FileCheck %t/ae_mulfd32x16x2_fir_ll.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulfp16x4ras.ll -o - | FileCheck %t/ae_mulfp16x4ras.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulfp16x4s.ll -o - | FileCheck %t/ae_mulfp16x4s.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulfp24x2r.ll -o - | FileCheck %t/ae_mulfp24x2r.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulfp24x2r_s2.ll -o - | FileCheck %t/ae_mulfp24x2r_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulfp24x2ra.ll -o - | FileCheck %t/ae_mulfp24x2ra.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulfp24x2ra_s2.ll -o - | FileCheck %t/ae_mulfp24x2ra_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulfp32x16x2ras_h.ll -o - | FileCheck %t/ae_mulfp32x16x2ras_h.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulfp32x16x2ras_h_s2.ll -o - | FileCheck %t/ae_mulfp32x16x2ras_h_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulfp32x16x2ras_l.ll -o - | FileCheck %t/ae_mulfp32x16x2ras_l.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulfp32x16x2ras_l_s2.ll -o - | FileCheck %t/ae_mulfp32x16x2ras_l_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulfp32x16x2rs_h.ll -o - | FileCheck %t/ae_mulfp32x16x2rs_h.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulfp32x16x2rs_h_s2.ll -o - | FileCheck %t/ae_mulfp32x16x2rs_h_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulfp32x16x2rs_l.ll -o - | FileCheck %t/ae_mulfp32x16x2rs_l.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulfp32x16x2rs_l_s2.ll -o - | FileCheck %t/ae_mulfp32x16x2rs_l_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulfp32x2ras.ll -o - | FileCheck %t/ae_mulfp32x2ras.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulfp32x2rs.ll -o - | FileCheck %t/ae_mulfp32x2rs.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulfq32sp24s_h_s2.ll -o - | FileCheck %t/ae_mulfq32sp24s_h_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulfq32sp24s_l_s2.ll -o - | FileCheck %t/ae_mulfq32sp24s_l_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulp24x2.ll -o - | FileCheck %t/ae_mulp24x2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulp24x2_s2.ll -o - | FileCheck %t/ae_mulp24x2_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulp32x16x2_h.ll -o - | FileCheck %t/ae_mulp32x16x2_h.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulp32x16x2_l.ll -o - | FileCheck %t/ae_mulp32x16x2_l.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulp32x2.ll -o - | FileCheck %t/ae_mulp32x2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulq32sp16s_l_s2.ll -o - | FileCheck %t/ae_mulq32sp16s_l_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulq32sp16u_l_s2.ll -o - | FileCheck %t/ae_mulq32sp16u_l_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulrfq32sp24s_h_s2.ll -o - | FileCheck %t/ae_mulrfq32sp24s_h_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulrfq32sp24s_l_s2.ll -o - | FileCheck %t/ae_mulrfq32sp24s_l_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_muls16x4.ll -o - | FileCheck %t/ae_muls16x4.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_muls32_hh.ll -o - | FileCheck %t/ae_muls32_hh.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_muls32_lh.ll -o - | FileCheck %t/ae_muls32_lh.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_muls32_ll.ll -o - | FileCheck %t/ae_muls32_ll.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_muls32f48p16s_hh.ll -o - | FileCheck %t/ae_muls32f48p16s_hh.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_muls32f48p16s_hh_s2.ll -o - | FileCheck %t/ae_muls32f48p16s_hh_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_muls32f48p16s_lh.ll -o - | FileCheck %t/ae_muls32f48p16s_lh.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_muls32f48p16s_lh_s2.ll -o - | FileCheck %t/ae_muls32f48p16s_lh_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_muls32f48p16s_ll.ll -o - | FileCheck %t/ae_muls32f48p16s_ll.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_muls32f48p16s_ll_s2.ll -o - | FileCheck %t/ae_muls32f48p16s_ll_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_muls32u_ll.ll -o - | FileCheck %t/ae_muls32u_ll.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_muls32x16_h0.ll -o - | FileCheck %t/ae_muls32x16_h0.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_muls32x16_h0_s2.ll -o - | FileCheck %t/ae_muls32x16_h0_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_muls32x16_h1.ll -o - | FileCheck %t/ae_muls32x16_h1.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_muls32x16_h1_s2.ll -o - | FileCheck %t/ae_muls32x16_h1_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_muls32x16_h2.ll -o - | FileCheck %t/ae_muls32x16_h2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_muls32x16_h2_s2.ll -o - | FileCheck %t/ae_muls32x16_h2_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_muls32x16_h3.ll -o - | FileCheck %t/ae_muls32x16_h3.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_muls32x16_h3_s2.ll -o - | FileCheck %t/ae_muls32x16_h3_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_muls32x16_l0.ll -o - | FileCheck %t/ae_muls32x16_l0.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_muls32x16_l0_s2.ll -o - | FileCheck %t/ae_muls32x16_l0_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_muls32x16_l1.ll -o - | FileCheck %t/ae_muls32x16_l1.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_muls32x16_l1_s2.ll -o - | FileCheck %t/ae_muls32x16_l1_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_muls32x16_l2.ll -o - | FileCheck %t/ae_muls32x16_l2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_muls32x16_l2_s2.ll -o - | FileCheck %t/ae_muls32x16_l2_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_muls32x16_l3.ll -o - | FileCheck %t/ae_muls32x16_l3.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_muls32x16_l3_s2.ll -o - | FileCheck %t/ae_muls32x16_l3_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsad24_hh_ll.ll -o - | FileCheck %t/ae_mulsad24_hh_ll.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsad24_hh_ll_s2.ll -o - | FileCheck %t/ae_mulsad24_hh_ll_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsad32x16_h1_l0.ll -o - | FileCheck %t/ae_mulsad32x16_h1_l0.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsad32x16_h1_l0_s2.ll -o - | FileCheck %t/ae_mulsad32x16_h1_l0_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsad32x16_h3_l2.ll -o - | FileCheck %t/ae_mulsad32x16_h3_l2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsad32x16_h3_l2_s2.ll -o - | FileCheck %t/ae_mulsad32x16_h3_l2_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsafd24_hh_ll.ll -o - | FileCheck %t/ae_mulsafd24_hh_ll.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsafd24_hh_ll_s2.ll -o - | FileCheck %t/ae_mulsafd24_hh_ll_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsafd32x16_h1_l0.ll -o - | FileCheck %t/ae_mulsafd32x16_h1_l0.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsafd32x16_h1_l0_s2.ll -o - | FileCheck %t/ae_mulsafd32x16_h1_l0_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsafd32x16_h3_l2.ll -o - | FileCheck %t/ae_mulsafd32x16_h3_l2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsafd32x16_h3_l2_s2.ll -o - | FileCheck %t/ae_mulsafd32x16_h3_l2_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsf16ss_00.ll -o - | FileCheck %t/ae_mulsf16ss_00.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsf16ss_00_s2.ll -o - | FileCheck %t/ae_mulsf16ss_00_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsf16ss_10.ll -o - | FileCheck %t/ae_mulsf16ss_10.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsf16ss_11.ll -o - | FileCheck %t/ae_mulsf16ss_11.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsf16ss_20.ll -o - | FileCheck %t/ae_mulsf16ss_20.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsf16ss_21.ll -o - | FileCheck %t/ae_mulsf16ss_21.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsf16ss_22.ll -o - | FileCheck %t/ae_mulsf16ss_22.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsf16ss_30.ll -o - | FileCheck %t/ae_mulsf16ss_30.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsf16ss_31.ll -o - | FileCheck %t/ae_mulsf16ss_31.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsf16ss_32.ll -o - | FileCheck %t/ae_mulsf16ss_32.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsf16ss_33.ll -o - | FileCheck %t/ae_mulsf16ss_33.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsf16x4ss.ll -o - | FileCheck %t/ae_mulsf16x4ss.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsf32r_hh.ll -o - | FileCheck %t/ae_mulsf32r_hh.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsf32r_lh.ll -o - | FileCheck %t/ae_mulsf32r_lh.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsf32r_ll.ll -o - | FileCheck %t/ae_mulsf32r_ll.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsf32r_ll_s2.ll -o - | FileCheck %t/ae_mulsf32r_ll_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsf32s_hh.ll -o - | FileCheck %t/ae_mulsf32s_hh.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsf32s_lh.ll -o - | FileCheck %t/ae_mulsf32s_lh.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsf32s_ll.ll -o - | FileCheck %t/ae_mulsf32s_ll.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsf32x16_h0.ll -o - | FileCheck %t/ae_mulsf32x16_h0.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsf32x16_h0_s2.ll -o - | FileCheck %t/ae_mulsf32x16_h0_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsf32x16_h1.ll -o - | FileCheck %t/ae_mulsf32x16_h1.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsf32x16_h1_s2.ll -o - | FileCheck %t/ae_mulsf32x16_h1_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsf32x16_h2.ll -o - | FileCheck %t/ae_mulsf32x16_h2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsf32x16_h2_s2.ll -o - | FileCheck %t/ae_mulsf32x16_h2_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsf32x16_h3.ll -o - | FileCheck %t/ae_mulsf32x16_h3.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsf32x16_h3_s2.ll -o - | FileCheck %t/ae_mulsf32x16_h3_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsf32x16_l0.ll -o - | FileCheck %t/ae_mulsf32x16_l0.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsf32x16_l0_s2.ll -o - | FileCheck %t/ae_mulsf32x16_l0_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsf32x16_l1.ll -o - | FileCheck %t/ae_mulsf32x16_l1.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsf32x16_l1_s2.ll -o - | FileCheck %t/ae_mulsf32x16_l1_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsf32x16_l2.ll -o - | FileCheck %t/ae_mulsf32x16_l2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsf32x16_l2_s2.ll -o - | FileCheck %t/ae_mulsf32x16_l2_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsf32x16_l3.ll -o - | FileCheck %t/ae_mulsf32x16_l3.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsf32x16_l3_s2.ll -o - | FileCheck %t/ae_mulsf32x16_l3_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsf48q32sp16s_l.ll -o - | FileCheck %t/ae_mulsf48q32sp16s_l.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsf48q32sp16s_l_s2.ll -o - | FileCheck %t/ae_mulsf48q32sp16s_l_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsf48q32sp16u_l.ll -o - | FileCheck %t/ae_mulsf48q32sp16u_l.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsf48q32sp16u_l_s2.ll -o - | FileCheck %t/ae_mulsf48q32sp16u_l_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsfp24x2r.ll -o - | FileCheck %t/ae_mulsfp24x2r.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsfp24x2r_s2.ll -o - | FileCheck %t/ae_mulsfp24x2r_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsfp24x2ra.ll -o - | FileCheck %t/ae_mulsfp24x2ra.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsfp24x2ra_s2.ll -o - | FileCheck %t/ae_mulsfp24x2ra_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsfp32x16x2ras_h.ll -o - | FileCheck %t/ae_mulsfp32x16x2ras_h.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsfp32x16x2ras_h_s2.ll -o - | FileCheck %t/ae_mulsfp32x16x2ras_h_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsfp32x16x2ras_l.ll -o - | FileCheck %t/ae_mulsfp32x16x2ras_l.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsfp32x16x2ras_l_s2.ll -o - | FileCheck %t/ae_mulsfp32x16x2ras_l_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsfp32x16x2rs_h.ll -o - | FileCheck %t/ae_mulsfp32x16x2rs_h.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsfp32x16x2rs_h_s2.ll -o - | FileCheck %t/ae_mulsfp32x16x2rs_h_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsfp32x16x2rs_l.ll -o - | FileCheck %t/ae_mulsfp32x16x2rs_l.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsfp32x16x2rs_l_s2.ll -o - | FileCheck %t/ae_mulsfp32x16x2rs_l_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsfp32x2ras.ll -o - | FileCheck %t/ae_mulsfp32x2ras.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsfp32x2rs.ll -o - | FileCheck %t/ae_mulsfp32x2rs.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsfq32sp24s_h_s2.ll -o - | FileCheck %t/ae_mulsfq32sp24s_h_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsfq32sp24s_l_s2.ll -o - | FileCheck %t/ae_mulsfq32sp24s_l_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsp24x2.ll -o - | FileCheck %t/ae_mulsp24x2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsp24x2_s2.ll -o - | FileCheck %t/ae_mulsp24x2_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsp32x16x2_h.ll -o - | FileCheck %t/ae_mulsp32x16x2_h.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsp32x16x2_l.ll -o - | FileCheck %t/ae_mulsp32x16x2_l.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsp32x2.ll -o - | FileCheck %t/ae_mulsp32x2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsq32sp16s_l_s2.ll -o - | FileCheck %t/ae_mulsq32sp16s_l_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsq32sp16u_l_s2.ll -o - | FileCheck %t/ae_mulsq32sp16u_l_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsrfq32sp24s_h_s2.ll -o - | FileCheck %t/ae_mulsrfq32sp24s_h_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulsrfq32sp24s_l_s2.ll -o - | FileCheck %t/ae_mulsrfq32sp24s_l_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulss32f48p16s_hh.ll -o - | FileCheck %t/ae_mulss32f48p16s_hh.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulss32f48p16s_hh_s2.ll -o - | FileCheck %t/ae_mulss32f48p16s_hh_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulss32f48p16s_lh.ll -o - | FileCheck %t/ae_mulss32f48p16s_lh.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulss32f48p16s_lh_s2.ll -o - | FileCheck %t/ae_mulss32f48p16s_lh_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulss32f48p16s_ll.ll -o - | FileCheck %t/ae_mulss32f48p16s_ll.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulss32f48p16s_ll_s2.ll -o - | FileCheck %t/ae_mulss32f48p16s_ll_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulssd24_hh_ll.ll -o - | FileCheck %t/ae_mulssd24_hh_ll.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulssd24_hh_ll_s2.ll -o - | FileCheck %t/ae_mulssd24_hh_ll_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulssd24_hl_lh.ll -o - | FileCheck %t/ae_mulssd24_hl_lh.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulssd24_hl_lh_s2.ll -o - | FileCheck %t/ae_mulssd24_hl_lh_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulssd32x16_h1_l0.ll -o - | FileCheck %t/ae_mulssd32x16_h1_l0.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulssd32x16_h1_l0_s2.ll -o - | FileCheck %t/ae_mulssd32x16_h1_l0_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulssd32x16_h3_l2.ll -o - | FileCheck %t/ae_mulssd32x16_h3_l2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulssd32x16_h3_l2_s2.ll -o - | FileCheck %t/ae_mulssd32x16_h3_l2_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulssfd16ss_11_00.ll -o - | FileCheck %t/ae_mulssfd16ss_11_00.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulssfd16ss_11_00_s2.ll -o - | FileCheck %t/ae_mulssfd16ss_11_00_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulssfd16ss_13_02.ll -o - | FileCheck %t/ae_mulssfd16ss_13_02.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulssfd16ss_13_02_s2.ll -o - | FileCheck %t/ae_mulssfd16ss_13_02_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulssfd16ss_33_22.ll -o - | FileCheck %t/ae_mulssfd16ss_33_22.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulssfd16ss_33_22_s2.ll -o - | FileCheck %t/ae_mulssfd16ss_33_22_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulssfd24_hh_ll.ll -o - | FileCheck %t/ae_mulssfd24_hh_ll.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulssfd24_hh_ll_s2.ll -o - | FileCheck %t/ae_mulssfd24_hh_ll_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulssfd24_hl_lh.ll -o - | FileCheck %t/ae_mulssfd24_hl_lh.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulssfd24_hl_lh_s2.ll -o - | FileCheck %t/ae_mulssfd24_hl_lh_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulssfd32x16_h1_l0.ll -o - | FileCheck %t/ae_mulssfd32x16_h1_l0.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulssfd32x16_h1_l0_s2.ll -o - | FileCheck %t/ae_mulssfd32x16_h1_l0_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulssfd32x16_h3_l2.ll -o - | FileCheck %t/ae_mulssfd32x16_h3_l2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulssfd32x16_h3_l2_s2.ll -o - | FileCheck %t/ae_mulssfd32x16_h3_l2_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzaad24_hh_ll.ll -o - | FileCheck %t/ae_mulzaad24_hh_ll.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzaad24_hh_ll_s2.ll -o - | FileCheck %t/ae_mulzaad24_hh_ll_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzaad24_hl_lh.ll -o - | FileCheck %t/ae_mulzaad24_hl_lh.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzaad24_hl_lh_s2.ll -o - | FileCheck %t/ae_mulzaad24_hl_lh_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzaad32x16_h0_l1.ll -o - | FileCheck %t/ae_mulzaad32x16_h0_l1.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzaad32x16_h0_l1_s2.ll -o - | FileCheck %t/ae_mulzaad32x16_h0_l1_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzaad32x16_h1_l0.ll -o - | FileCheck %t/ae_mulzaad32x16_h1_l0.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzaad32x16_h1_l0_s2.ll -o - | FileCheck %t/ae_mulzaad32x16_h1_l0_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzaad32x16_h2_l3.ll -o - | FileCheck %t/ae_mulzaad32x16_h2_l3.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzaad32x16_h2_l3_s2.ll -o - | FileCheck %t/ae_mulzaad32x16_h2_l3_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzaad32x16_h3_l2.ll -o - | FileCheck %t/ae_mulzaad32x16_h3_l2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzaad32x16_h3_l2_s2.ll -o - | FileCheck %t/ae_mulzaad32x16_h3_l2_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzaafd16ss_11_00.ll -o - | FileCheck %t/ae_mulzaafd16ss_11_00.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzaafd16ss_11_00_s2.ll -o - | FileCheck %t/ae_mulzaafd16ss_11_00_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzaafd16ss_13_02.ll -o - | FileCheck %t/ae_mulzaafd16ss_13_02.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzaafd16ss_13_02_s2.ll -o - | FileCheck %t/ae_mulzaafd16ss_13_02_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzaafd16ss_33_22.ll -o - | FileCheck %t/ae_mulzaafd16ss_33_22.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzaafd16ss_33_22_s2.ll -o - | FileCheck %t/ae_mulzaafd16ss_33_22_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzaafd24_hh_ll.ll -o - | FileCheck %t/ae_mulzaafd24_hh_ll.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzaafd24_hh_ll_s2.ll -o - | FileCheck %t/ae_mulzaafd24_hh_ll_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzaafd24_hl_lh.ll -o - | FileCheck %t/ae_mulzaafd24_hl_lh.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzaafd24_hl_lh_s2.ll -o - | FileCheck %t/ae_mulzaafd24_hl_lh_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzaafd32x16_h0_l1.ll -o - | FileCheck %t/ae_mulzaafd32x16_h0_l1.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzaafd32x16_h0_l1_s2.ll -o - | FileCheck %t/ae_mulzaafd32x16_h0_l1_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzaafd32x16_h1_l0.ll -o - | FileCheck %t/ae_mulzaafd32x16_h1_l0.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzaafd32x16_h1_l0_s2.ll -o - | FileCheck %t/ae_mulzaafd32x16_h1_l0_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzaafd32x16_h2_l3.ll -o - | FileCheck %t/ae_mulzaafd32x16_h2_l3.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzaafd32x16_h2_l3_s2.ll -o - | FileCheck %t/ae_mulzaafd32x16_h2_l3_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzaafd32x16_h3_l2.ll -o - | FileCheck %t/ae_mulzaafd32x16_h3_l2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzaafd32x16_h3_l2_s2.ll -o - | FileCheck %t/ae_mulzaafd32x16_h3_l2_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzasd24_hh_ll.ll -o - | FileCheck %t/ae_mulzasd24_hh_ll.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzasd24_hh_ll_s2.ll -o - | FileCheck %t/ae_mulzasd24_hh_ll_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzasd24_hl_lh.ll -o - | FileCheck %t/ae_mulzasd24_hl_lh.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzasd24_hl_lh_s2.ll -o - | FileCheck %t/ae_mulzasd24_hl_lh_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzasd32x16_h1_l0.ll -o - | FileCheck %t/ae_mulzasd32x16_h1_l0.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzasd32x16_h1_l0_s2.ll -o - | FileCheck %t/ae_mulzasd32x16_h1_l0_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzasd32x16_h3_l2.ll -o - | FileCheck %t/ae_mulzasd32x16_h3_l2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzasd32x16_h3_l2_s2.ll -o - | FileCheck %t/ae_mulzasd32x16_h3_l2_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzasfd24_hh_ll.ll -o - | FileCheck %t/ae_mulzasfd24_hh_ll.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzasfd24_hh_ll_s2.ll -o - | FileCheck %t/ae_mulzasfd24_hh_ll_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzasfd24_hl_lh.ll -o - | FileCheck %t/ae_mulzasfd24_hl_lh.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzasfd24_hl_lh_s2.ll -o - | FileCheck %t/ae_mulzasfd24_hl_lh_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzasfd32x16_h1_l0.ll -o - | FileCheck %t/ae_mulzasfd32x16_h1_l0.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzasfd32x16_h1_l0_s2.ll -o - | FileCheck %t/ae_mulzasfd32x16_h1_l0_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzasfd32x16_h3_l2.ll -o - | FileCheck %t/ae_mulzasfd32x16_h3_l2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzasfd32x16_h3_l2_s2.ll -o - | FileCheck %t/ae_mulzasfd32x16_h3_l2_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzsad24_hh_ll.ll -o - | FileCheck %t/ae_mulzsad24_hh_ll.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzsad24_hh_ll_s2.ll -o - | FileCheck %t/ae_mulzsad24_hh_ll_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzsad32x16_h1_l0.ll -o - | FileCheck %t/ae_mulzsad32x16_h1_l0.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzsad32x16_h1_l0_s2.ll -o - | FileCheck %t/ae_mulzsad32x16_h1_l0_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzsad32x16_h3_l2.ll -o - | FileCheck %t/ae_mulzsad32x16_h3_l2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzsad32x16_h3_l2_s2.ll -o - | FileCheck %t/ae_mulzsad32x16_h3_l2_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzsafd24_hh_ll.ll -o - | FileCheck %t/ae_mulzsafd24_hh_ll.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzsafd24_hh_ll_s2.ll -o - | FileCheck %t/ae_mulzsafd24_hh_ll_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzsafd32x16_h1_l0.ll -o - | FileCheck %t/ae_mulzsafd32x16_h1_l0.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzsafd32x16_h1_l0_s2.ll -o - | FileCheck %t/ae_mulzsafd32x16_h1_l0_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzsafd32x16_h3_l2.ll -o - | FileCheck %t/ae_mulzsafd32x16_h3_l2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzsafd32x16_h3_l2_s2.ll -o - | FileCheck %t/ae_mulzsafd32x16_h3_l2_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzssd24_hh_ll.ll -o - | FileCheck %t/ae_mulzssd24_hh_ll.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzssd24_hh_ll_s2.ll -o - | FileCheck %t/ae_mulzssd24_hh_ll_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzssd24_hl_lh.ll -o - | FileCheck %t/ae_mulzssd24_hl_lh.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzssd24_hl_lh_s2.ll -o - | FileCheck %t/ae_mulzssd24_hl_lh_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzssd32x16_h1_l0.ll -o - | FileCheck %t/ae_mulzssd32x16_h1_l0.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzssd32x16_h1_l0_s2.ll -o - | FileCheck %t/ae_mulzssd32x16_h1_l0_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzssd32x16_h3_l2.ll -o - | FileCheck %t/ae_mulzssd32x16_h3_l2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzssd32x16_h3_l2_s2.ll -o - | FileCheck %t/ae_mulzssd32x16_h3_l2_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzssfd16ss_11_00.ll -o - | FileCheck %t/ae_mulzssfd16ss_11_00.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzssfd16ss_11_00_s2.ll -o - | FileCheck %t/ae_mulzssfd16ss_11_00_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzssfd16ss_13_02.ll -o - | FileCheck %t/ae_mulzssfd16ss_13_02.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzssfd16ss_13_02_s2.ll -o - | FileCheck %t/ae_mulzssfd16ss_13_02_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzssfd16ss_33_22.ll -o - | FileCheck %t/ae_mulzssfd16ss_33_22.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzssfd16ss_33_22_s2.ll -o - | FileCheck %t/ae_mulzssfd16ss_33_22_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzssfd24_hh_ll.ll -o - | FileCheck %t/ae_mulzssfd24_hh_ll.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzssfd24_hh_ll_s2.ll -o - | FileCheck %t/ae_mulzssfd24_hh_ll_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzssfd24_hl_lh.ll -o - | FileCheck %t/ae_mulzssfd24_hl_lh.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzssfd24_hl_lh_s2.ll -o - | FileCheck %t/ae_mulzssfd24_hl_lh_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzssfd32x16_h1_l0.ll -o - | FileCheck %t/ae_mulzssfd32x16_h1_l0.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzssfd32x16_h1_l0_s2.ll -o - | FileCheck %t/ae_mulzssfd32x16_h1_l0_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzssfd32x16_h3_l2.ll -o - | FileCheck %t/ae_mulzssfd32x16_h3_l2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_mulzssfd32x16_h3_l2_s2.ll -o - | FileCheck %t/ae_mulzssfd32x16_h3_l2_s2.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_nand.ll -o - | FileCheck %t/ae_nand.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_neg16s.ll -o - | FileCheck %t/ae_neg16s.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_neg24s.ll -o - | FileCheck %t/ae_neg24s.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_neg32.ll -o - | FileCheck %t/ae_neg32.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_neg32s.ll -o - | FileCheck %t/ae_neg32s.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_neg64.ll -o - | FileCheck %t/ae_neg64.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_neg64s.ll -o - | FileCheck %t/ae_neg64s.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_nsa64.ll -o - | FileCheck %t/ae_nsa64.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_nsaz16_0.ll -o - | FileCheck %t/ae_nsaz16_0.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_nsaz32_l.ll -o - | FileCheck %t/ae_nsaz32_l.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_or.ll -o - | FileCheck %t/ae_or.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_pksr24.ll -o - | FileCheck %t/ae_pksr24.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_pksr32.ll -o - | FileCheck %t/ae_pksr32.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_round16x4f32sasym.ll -o - | FileCheck %t/ae_round16x4f32sasym.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_round16x4f32ssym.ll -o - | FileCheck %t/ae_round16x4f32ssym.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_round24x2f48sasym.ll -o - | FileCheck %t/ae_round24x2f48sasym.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_round24x2f48ssym.ll -o - | FileCheck %t/ae_round24x2f48ssym.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_round32x2f48sasym.ll -o - | FileCheck %t/ae_round32x2f48sasym.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_round32x2f48ssym.ll -o - | FileCheck %t/ae_round32x2f48ssym.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_round32x2f64sasym.ll -o - | FileCheck %t/ae_round32x2f64sasym.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_round32x2f64ssym.ll -o - | FileCheck %t/ae_round32x2f64ssym.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_roundsp16f24asym.ll -o - | FileCheck %t/ae_roundsp16f24asym.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_roundsp16f24sym.ll -o - | FileCheck %t/ae_roundsp16f24sym.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_roundsp16q48x2asym.ll -o - | FileCheck %t/ae_roundsp16q48x2asym.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_roundsp16q48x2sym.ll -o - | FileCheck %t/ae_roundsp16q48x2sym.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_roundsq32f48asym.ll -o - | FileCheck %t/ae_roundsq32f48asym.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_roundsq32f48sym.ll -o - | FileCheck %t/ae_roundsq32f48sym.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s16_0_i.ll -o - | FileCheck %t/ae_s16_0_i.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s16_0_ip.ll -o - | FileCheck %t/ae_s16_0_ip.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s16_0_x.ll -o - | FileCheck %t/ae_s16_0_x.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s16_0_xc.ll -o - | FileCheck %t/ae_s16_0_xc.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s16_0_xp.ll -o - | FileCheck %t/ae_s16_0_xp.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s16m_l_i.ll -o - | FileCheck %t/ae_s16m_l_i.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s16m_l_iu.ll -o - | FileCheck %t/ae_s16m_l_iu.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s16m_l_x.ll -o - | FileCheck %t/ae_s16m_l_x.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s16m_l_xc.ll -o - | FileCheck %t/ae_s16m_l_xc.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s16m_l_xu.ll -o - | FileCheck %t/ae_s16m_l_xu.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s16x2m_i.ll -o - | FileCheck %t/ae_s16x2m_i.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s16x2m_iu.ll -o - | FileCheck %t/ae_s16x2m_iu.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s16x2m_x.ll -o - | FileCheck %t/ae_s16x2m_x.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s16x2m_xc.ll -o - | FileCheck %t/ae_s16x2m_xc.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s16x2m_xu.ll -o - | FileCheck %t/ae_s16x2m_xu.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s16x4_i.ll -o - | FileCheck %t/ae_s16x4_i.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s16x4_ip.ll -o - | FileCheck %t/ae_s16x4_ip.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s16x4_ric.ll -o - | FileCheck %t/ae_s16x4_ric.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s16x4_rip.ll -o - | FileCheck %t/ae_s16x4_rip.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s16x4_x.ll -o - | FileCheck %t/ae_s16x4_x.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s16x4_xc.ll -o - | FileCheck %t/ae_s16x4_xc.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s16x4_xp.ll -o - | FileCheck %t/ae_s16x4_xp.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s24ra64s_i.ll -o - | FileCheck %t/ae_s24ra64s_i.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s24ra64s_ip.ll -o - | FileCheck %t/ae_s24ra64s_ip.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s24ra64s_x.ll -o - | FileCheck %t/ae_s24ra64s_x.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s24ra64s_xc.ll -o - | FileCheck %t/ae_s24ra64s_xc.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s24ra64s_xp.ll -o - | FileCheck %t/ae_s24ra64s_xp.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s24x2ra64s_ip.ll -o - | FileCheck %t/ae_s24x2ra64s_ip.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s32_l_i.ll -o - | FileCheck %t/ae_s32_l_i.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s32_l_ip.ll -o - | FileCheck %t/ae_s32_l_ip.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s32_l_x.ll -o - | FileCheck %t/ae_s32_l_x.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s32_l_xc.ll -o - | FileCheck %t/ae_s32_l_xc.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s32_l_xp.ll -o - | FileCheck %t/ae_s32_l_xp.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s32f24_l_i.ll -o - | FileCheck %t/ae_s32f24_l_i.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s32f24_l_ip.ll -o - | FileCheck %t/ae_s32f24_l_ip.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s32f24_l_x.ll -o - | FileCheck %t/ae_s32f24_l_x.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s32f24_l_xc.ll -o - | FileCheck %t/ae_s32f24_l_xc.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s32f24_l_xp.ll -o - | FileCheck %t/ae_s32f24_l_xp.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s32m_i.ll -o - | FileCheck %t/ae_s32m_i.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s32m_iu.ll -o - | FileCheck %t/ae_s32m_iu.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s32m_x.ll -o - | FileCheck %t/ae_s32m_x.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s32m_xc.ll -o - | FileCheck %t/ae_s32m_xc.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s32m_xu.ll -o - | FileCheck %t/ae_s32m_xu.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s32ra64s_i.ll -o - | FileCheck %t/ae_s32ra64s_i.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s32ra64s_ip.ll -o - | FileCheck %t/ae_s32ra64s_ip.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s32ra64s_x.ll -o - | FileCheck %t/ae_s32ra64s_x.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s32ra64s_xc.ll -o - | FileCheck %t/ae_s32ra64s_xc.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s32ra64s_xp.ll -o - | FileCheck %t/ae_s32ra64s_xp.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s32x2_i.ll -o - | FileCheck %t/ae_s32x2_i.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s32x2_ip.ll -o - | FileCheck %t/ae_s32x2_ip.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s32x2_ric.ll -o - | FileCheck %t/ae_s32x2_ric.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s32x2_rip.ll -o - | FileCheck %t/ae_s32x2_rip.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s32x2_x.ll -o - | FileCheck %t/ae_s32x2_x.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s32x2_xc.ll -o - | FileCheck %t/ae_s32x2_xc.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s32x2_xp.ll -o - | FileCheck %t/ae_s32x2_xp.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s32x2f24_i.ll -o - | FileCheck %t/ae_s32x2f24_i.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s32x2f24_ip.ll -o - | FileCheck %t/ae_s32x2f24_ip.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s32x2f24_ric.ll -o - | FileCheck %t/ae_s32x2f24_ric.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s32x2f24_rip.ll -o - | FileCheck %t/ae_s32x2f24_rip.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s32x2f24_x.ll -o - | FileCheck %t/ae_s32x2f24_x.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s32x2f24_xc.ll -o - | FileCheck %t/ae_s32x2f24_xc.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s32x2f24_xp.ll -o - | FileCheck %t/ae_s32x2f24_xp.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s32x2ra64s_ip.ll -o - | FileCheck %t/ae_s32x2ra64s_ip.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s64_i.ll -o - | FileCheck %t/ae_s64_i.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s64_ip.ll -o - | FileCheck %t/ae_s64_ip.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s64_x.ll -o - | FileCheck %t/ae_s64_x.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s64_xc.ll -o - | FileCheck %t/ae_s64_xc.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_s64_xp.ll -o - | FileCheck %t/ae_s64_xp.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sa16x4_ic.ll -o - | FileCheck %t/ae_sa16x4_ic.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sa16x4_ip.ll -o - | FileCheck %t/ae_sa16x4_ip.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sa16x4_ric.ll -o - | FileCheck %t/ae_sa16x4_ric.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sa16x4_rip.ll -o - | FileCheck %t/ae_sa16x4_rip.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sa24_l_ic.ll -o - | FileCheck %t/ae_sa24_l_ic.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sa24_l_ip.ll -o - | FileCheck %t/ae_sa24_l_ip.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sa24_l_ric.ll -o - | FileCheck %t/ae_sa24_l_ric.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sa24_l_rip.ll -o - | FileCheck %t/ae_sa24_l_rip.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sa24x2_ic.ll -o - | FileCheck %t/ae_sa24x2_ic.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sa24x2_ip.ll -o - | FileCheck %t/ae_sa24x2_ip.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sa24x2_ric.ll -o - | FileCheck %t/ae_sa24x2_ric.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sa24x2_rip.ll -o - | FileCheck %t/ae_sa24x2_rip.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sa32x2_ic.ll -o - | FileCheck %t/ae_sa32x2_ic.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sa32x2_ip.ll -o - | FileCheck %t/ae_sa32x2_ip.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sa32x2_ric.ll -o - | FileCheck %t/ae_sa32x2_ric.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sa32x2_rip.ll -o - | FileCheck %t/ae_sa32x2_rip.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sa32x2f24_ic.ll -o - | FileCheck %t/ae_sa32x2f24_ic.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sa32x2f24_ip.ll -o - | FileCheck %t/ae_sa32x2f24_ip.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sa32x2f24_ric.ll -o - | FileCheck %t/ae_sa32x2f24_ric.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sa32x2f24_rip.ll -o - | FileCheck %t/ae_sa32x2f24_rip.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sa64neg_fp.ll -o - | FileCheck %t/ae_sa64neg_fp.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sa64pos_fp.ll -o - | FileCheck %t/ae_sa64pos_fp.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_salign64_i.ll -o - | FileCheck %t/ae_salign64_i.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sat16x4.ll -o - | FileCheck %t/ae_sat16x4.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sat24s.ll -o - | FileCheck %t/ae_sat24s.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sat48s.ll -o - | FileCheck %t/ae_sat48s.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_satq56s.ll -o - | FileCheck %t/ae_satq56s.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sb.ll -o - | FileCheck %t/ae_sb.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sb_ic.ll -o - | FileCheck %t/ae_sb_ic.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sb_ip.ll -o - | FileCheck %t/ae_sb_ip.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sbf.ll -o - | FileCheck %t/ae_sbf.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sbf_ic.ll -o - | FileCheck %t/ae_sbf_ic.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sbf_ip.ll -o - | FileCheck %t/ae_sbf_ip.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sbi.ll -o - | FileCheck %t/ae_sbi.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sbi_ic.ll -o - | FileCheck %t/ae_sbi_ic.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sbi_ip.ll -o - | FileCheck %t/ae_sbi_ip.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sel16i.ll -o - | FileCheck %t/ae_sel16i.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sel16i_n.ll -o - | FileCheck %t/ae_sel16i_n.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sext32.ll -o - | FileCheck %t/ae_sext32.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sext32x2d16_10.ll -o - | FileCheck %t/ae_sext32x2d16_10.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sext32x2d16_32.ll -o - | FileCheck %t/ae_sext32x2d16_32.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sha32.ll -o - | FileCheck %t/ae_sha32.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_shortswap.ll -o - | FileCheck %t/ae_shortswap.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_slaa16s.ll -o - | FileCheck %t/ae_slaa16s.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_slaa32.ll -o - | FileCheck %t/ae_slaa32.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_slaa32s.ll -o - | FileCheck %t/ae_slaa32s.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_slaa64.ll -o - | FileCheck %t/ae_slaa64.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_slaa64s.ll -o - | FileCheck %t/ae_slaa64s.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_slaaq56.ll -o - | FileCheck %t/ae_slaaq56.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_slai16s.ll -o - | FileCheck %t/ae_slai16s.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_slai24.ll -o - | FileCheck %t/ae_slai24.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_slai24s.ll -o - | FileCheck %t/ae_slai24s.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_slai32.ll -o - | FileCheck %t/ae_slai32.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_slai32s.ll -o - | FileCheck %t/ae_slai32s.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_slai64.ll -o - | FileCheck %t/ae_slai64.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_slai64s.ll -o - | FileCheck %t/ae_slai64s.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_slaisq56s.ll -o - | FileCheck %t/ae_slaisq56s.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_slas24.ll -o - | FileCheck %t/ae_slas24.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_slas24s.ll -o - | FileCheck %t/ae_slas24s.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_slas32.ll -o - | FileCheck %t/ae_slas32.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_slas32s.ll -o - | FileCheck %t/ae_slas32s.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_slas64.ll -o - | FileCheck %t/ae_slas64.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_slas64s.ll -o - | FileCheck %t/ae_slas64s.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_slasq56.ll -o - | FileCheck %t/ae_slasq56.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_slassq56s.ll -o - | FileCheck %t/ae_slassq56s.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sra64_32.ll -o - | FileCheck %t/ae_sra64_32.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sraa16rs.ll -o - | FileCheck %t/ae_sraa16rs.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sraa16s.ll -o - | FileCheck %t/ae_sraa16s.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sraa32.ll -o - | FileCheck %t/ae_sraa32.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sraa32rs.ll -o - | FileCheck %t/ae_sraa32rs.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sraa32s.ll -o - | FileCheck %t/ae_sraa32s.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sraa64.ll -o - | FileCheck %t/ae_sraa64.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_srai16.ll -o - | FileCheck %t/ae_srai16.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_srai16r.ll -o - | FileCheck %t/ae_srai16r.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_srai24.ll -o - | FileCheck %t/ae_srai24.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_srai32.ll -o - | FileCheck %t/ae_srai32.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_srai32r.ll -o - | FileCheck %t/ae_srai32r.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_srai64.ll -o - | FileCheck %t/ae_srai64.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sras24.ll -o - | FileCheck %t/ae_sras24.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sras32.ll -o - | FileCheck %t/ae_sras32.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sras64.ll -o - | FileCheck %t/ae_sras64.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_srla32.ll -o - | FileCheck %t/ae_srla32.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_srla64.ll -o - | FileCheck %t/ae_srla64.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_srli24.ll -o - | FileCheck %t/ae_srli24.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_srli32.ll -o - | FileCheck %t/ae_srli32.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_srli64.ll -o - | FileCheck %t/ae_srli64.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_srls24.ll -o - | FileCheck %t/ae_srls24.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_srls32.ll -o - | FileCheck %t/ae_srls32.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_srls64.ll -o - | FileCheck %t/ae_srls64.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sub16.ll -o - | FileCheck %t/ae_sub16.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sub16s.ll -o - | FileCheck %t/ae_sub16s.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sub24s.ll -o - | FileCheck %t/ae_sub24s.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sub32.ll -o - | FileCheck %t/ae_sub32.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sub32s.ll -o - | FileCheck %t/ae_sub32s.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sub64.ll -o - | FileCheck %t/ae_sub64.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_sub64s.ll -o - | FileCheck %t/ae_sub64s.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_subadd32.ll -o - | FileCheck %t/ae_subadd32.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_subadd32s.ll -o - | FileCheck %t/ae_subadd32s.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_trunca32f64s_l.ll -o - | FileCheck %t/ae_trunca32f64s_l.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_trunca32x2f64s.ll -o - | FileCheck %t/ae_trunca32x2f64s.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_trunci32f64s_l.ll -o - | FileCheck %t/ae_trunci32f64s_l.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_trunci32x2f64s.ll -o - | FileCheck %t/ae_trunci32x2f64s.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_vldl16c.ll -o - | FileCheck %t/ae_vldl16c.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_vldl16c_ic.ll -o - | FileCheck %t/ae_vldl16c_ic.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_vldl16c_ip.ll -o - | FileCheck %t/ae_vldl16c_ip.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_vldl16t.ll -o - | FileCheck %t/ae_vldl16t.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_vldl32t.ll -o - | FileCheck %t/ae_vldl32t.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_vldsht.ll -o - | FileCheck %t/ae_vldsht.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_vlel16t.ll -o - | FileCheck %t/ae_vlel16t.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_vlel32t.ll -o - | FileCheck %t/ae_vlel32t.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_vles16c.ll -o - | FileCheck %t/ae_vles16c.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_vles16c_ic.ll -o - | FileCheck %t/ae_vles16c_ic.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_vles16c_ip.ll -o - | FileCheck %t/ae_vles16c_ip.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_xor.ll -o - | FileCheck %t/ae_xor.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/ae_zalign64.ll -o - | FileCheck %t/ae_zalign64.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/rur_ae_bithead.ll -o - | FileCheck %t/rur_ae_bithead.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/rur_ae_bitptr.ll -o - | FileCheck %t/rur_ae_bitptr.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/rur_ae_bitsused.ll -o - | FileCheck %t/rur_ae_bitsused.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/rur_ae_cbegin0.ll -o - | FileCheck %t/rur_ae_cbegin0.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/rur_ae_cend0.ll -o - | FileCheck %t/rur_ae_cend0.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/rur_ae_cw_sd_no.ll -o - | FileCheck %t/rur_ae_cw_sd_no.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/rur_ae_cwrap.ll -o - | FileCheck %t/rur_ae_cwrap.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/rur_ae_first_ts.ll -o - | FileCheck %t/rur_ae_first_ts.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/rur_ae_nextoffset.ll -o - | FileCheck %t/rur_ae_nextoffset.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/rur_ae_overflow.ll -o - | FileCheck %t/rur_ae_overflow.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/rur_ae_ovf_sar.ll -o - | FileCheck %t/rur_ae_ovf_sar.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/rur_ae_sar.ll -o - | FileCheck %t/rur_ae_sar.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/rur_ae_searchdone.ll -o - | FileCheck %t/rur_ae_searchdone.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/rur_ae_tablesize.ll -o - | FileCheck %t/rur_ae_tablesize.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/rur_ae_ts_fts_bu_bp.ll -o - | FileCheck %t/rur_ae_ts_fts_bu_bp.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/wur_ae_bithead.ll -o - | FileCheck %t/wur_ae_bithead.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/wur_ae_bitptr.ll -o - | FileCheck %t/wur_ae_bitptr.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/wur_ae_bitsused.ll -o - | FileCheck %t/wur_ae_bitsused.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/wur_ae_cbegin0.ll -o - | FileCheck %t/wur_ae_cbegin0.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/wur_ae_cend0.ll -o - | FileCheck %t/wur_ae_cend0.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/wur_ae_cw_sd_no.ll -o - | FileCheck %t/wur_ae_cw_sd_no.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/wur_ae_cwrap.ll -o - | FileCheck %t/wur_ae_cwrap.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/wur_ae_first_ts.ll -o - | FileCheck %t/wur_ae_first_ts.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/wur_ae_nextoffset.ll -o - | FileCheck %t/wur_ae_nextoffset.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/wur_ae_overflow.ll -o - | FileCheck %t/wur_ae_overflow.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/wur_ae_ovf_sar.ll -o - | FileCheck %t/wur_ae_ovf_sar.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/wur_ae_sar.ll -o - | FileCheck %t/wur_ae_sar.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/wur_ae_searchdone.ll -o - | FileCheck %t/wur_ae_searchdone.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/wur_ae_tablesize.ll -o - | FileCheck %t/wur_ae_tablesize.ll
; RUN: llc -mtriple=xtensa -mcpu=cnl %t/wur_ae_ts_fts_bu_bp.ll -o - | FileCheck %t/wur_ae_ts_fts_bu_bp.ll
;--- ae_abs16s.ll
declare <4 x i16> @llvm.xtensa.ae.abs16s(<4 x i16>)
define <4 x i16> @test_xtensa_ae_abs16s(<4 x i16> %ae_arth_v1) {
; CHECK-LABEL: test_xtensa_ae_abs16s:

%ret = call <4 x i16> @llvm.xtensa.ae.abs16s(<4 x i16> %ae_arth_v1)
; CHECK: ae_abs16s aed{{[0-9]+}}, aed{{[0-9]+}}

ret <4 x i16> %ret
}


;--- ae_abs24s.ll
declare <2 x i32> @llvm.xtensa.ae.abs24s(<2 x i32>)
define <2 x i32> @test_xtensa_ae_abs24s(<2 x i32> %ae_arth_v1) {
; CHECK-LABEL: test_xtensa_ae_abs24s:

%ret = call <2 x i32> @llvm.xtensa.ae.abs24s(<2 x i32> %ae_arth_v1)
; CHECK: ae_abs24s aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_abs32.ll
declare <2 x i32> @llvm.xtensa.ae.abs32(<2 x i32>)
define <2 x i32> @test_xtensa_ae_abs32(<2 x i32> %ae_arth_v1) {
; CHECK-LABEL: test_xtensa_ae_abs32:

%ret = call <2 x i32> @llvm.xtensa.ae.abs32(<2 x i32> %ae_arth_v1)
; CHECK: ae_abs32 aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_abs32s.ll
declare <2 x i32> @llvm.xtensa.ae.abs32s(<2 x i32>)
define <2 x i32> @test_xtensa_ae_abs32s(<2 x i32> %ae_arth_v1) {
; CHECK-LABEL: test_xtensa_ae_abs32s:

%ret = call <2 x i32> @llvm.xtensa.ae.abs32s(<2 x i32> %ae_arth_v1)
; CHECK: ae_abs32s aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_abs64.ll
declare <1 x i64> @llvm.xtensa.ae.abs64(<1 x i64>)
define <1 x i64> @test_xtensa_ae_abs64(<1 x i64> %ae_arth_v1) {
; CHECK-LABEL: test_xtensa_ae_abs64:

%ret = call <1 x i64> @llvm.xtensa.ae.abs64(<1 x i64> %ae_arth_v1)
; CHECK: ae_abs64 aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_abs64s.ll
declare <1 x i64> @llvm.xtensa.ae.abs64s(<1 x i64>)
define <1 x i64> @test_xtensa_ae_abs64s(<1 x i64> %ae_arth_v1) {
; CHECK-LABEL: test_xtensa_ae_abs64s:

%ret = call <1 x i64> @llvm.xtensa.ae.abs64s(<1 x i64> %ae_arth_v1)
; CHECK: ae_abs64s aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_add16.ll
declare <4 x i16> @llvm.xtensa.ae.add16(<4 x i16>, <4 x i16>)
define <4 x i16> @test_xtensa_ae_add16(<4 x i16> %ae_arth_v0, <4 x i16> %ae_arth_v1) {
; CHECK-LABEL: test_xtensa_ae_add16:


%ret = call <4 x i16> @llvm.xtensa.ae.add16(<4 x i16> %ae_arth_v0, <4 x i16> %ae_arth_v1)
; CHECK: ae_add16 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <4 x i16> %ret
}


;--- ae_add16s.ll
declare <4 x i16> @llvm.xtensa.ae.add16s(<4 x i16>, <4 x i16>)
define <4 x i16> @test_xtensa_ae_add16s(<4 x i16> %ae_arth_v0, <4 x i16> %ae_arth_v1) {
; CHECK-LABEL: test_xtensa_ae_add16s:


%ret = call <4 x i16> @llvm.xtensa.ae.add16s(<4 x i16> %ae_arth_v0, <4 x i16> %ae_arth_v1)
; CHECK: ae_add16s aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <4 x i16> %ret
}


;--- ae_add24s.ll
declare <2 x i32> @llvm.xtensa.ae.add24s(<2 x i32>, <2 x i32>)
define <2 x i32> @test_xtensa_ae_add24s(<2 x i32> %ae_arth_v0, <2 x i32> %ae_arth_v1) {
; CHECK-LABEL: test_xtensa_ae_add24s:


%ret = call <2 x i32> @llvm.xtensa.ae.add24s(<2 x i32> %ae_arth_v0, <2 x i32> %ae_arth_v1)
; CHECK: ae_add24s aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_add32.ll
declare <2 x i32> @llvm.xtensa.ae.add32(<2 x i32>, <2 x i32>)
define <2 x i32> @test_xtensa_ae_add32(<2 x i32> %ae_arth_v0, <2 x i32> %ae_arth_v1) {
; CHECK-LABEL: test_xtensa_ae_add32:


%ret = call <2 x i32> @llvm.xtensa.ae.add32(<2 x i32> %ae_arth_v0, <2 x i32> %ae_arth_v1)
; CHECK: ae_add32 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_add32_hl_lh.ll
declare <2 x i32> @llvm.xtensa.ae.add32.hl.lh(<2 x i32>, <2 x i32>)
define <2 x i32> @test_xtensa_ae_add32_hl_lh(<2 x i32> %ae_arth_v0, <2 x i32> %ae_arth_v1) {
; CHECK-LABEL: test_xtensa_ae_add32_hl_lh:


%ret = call <2 x i32> @llvm.xtensa.ae.add32.hl.lh(<2 x i32> %ae_arth_v0, <2 x i32> %ae_arth_v1)
; CHECK: ae_add32_hl_lh aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_add32s.ll
declare <2 x i32> @llvm.xtensa.ae.add32s(<2 x i32>, <2 x i32>)
define <2 x i32> @test_xtensa_ae_add32s(<2 x i32> %ae_arth_v0, <2 x i32> %ae_arth_v1) {
; CHECK-LABEL: test_xtensa_ae_add32s:


%ret = call <2 x i32> @llvm.xtensa.ae.add32s(<2 x i32> %ae_arth_v0, <2 x i32> %ae_arth_v1)
; CHECK: ae_add32s aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_add64.ll
declare <1 x i64> @llvm.xtensa.ae.add64(<1 x i64>, <1 x i64>)
define <1 x i64> @test_xtensa_ae_add64(<1 x i64> %ae_arth_v0, <1 x i64> %ae_arth_v1) {
; CHECK-LABEL: test_xtensa_ae_add64:


%ret = call <1 x i64> @llvm.xtensa.ae.add64(<1 x i64> %ae_arth_v0, <1 x i64> %ae_arth_v1)
; CHECK: ae_add64 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_add64s.ll
declare <1 x i64> @llvm.xtensa.ae.add64s(<1 x i64>, <1 x i64>)
define <1 x i64> @test_xtensa_ae_add64s(<1 x i64> %ae_arth_v0, <1 x i64> %ae_arth_v1) {
; CHECK-LABEL: test_xtensa_ae_add64s:


%ret = call <1 x i64> @llvm.xtensa.ae.add64s(<1 x i64> %ae_arth_v0, <1 x i64> %ae_arth_v1)
; CHECK: ae_add64s aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_addbrba32.ll
declare i32 @llvm.xtensa.ae.addbrba32(i32, i32)
define i32 @test_xtensa_ae_addbrba32(i32 %art, i32 %ars) {
; CHECK-LABEL: test_xtensa_ae_addbrba32:


%ret = call i32 @llvm.xtensa.ae.addbrba32(i32 %art, i32 %ars)
; CHECK: ae_addbrba32 a{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}

ret i32 %ret
}


;--- ae_addsub32.ll
declare <2 x i32> @llvm.xtensa.ae.addsub32(<2 x i32>, <2 x i32>)
define <2 x i32> @test_xtensa_ae_addsub32(<2 x i32> %ae_arth_v0, <2 x i32> %ae_arth_v1) {
; CHECK-LABEL: test_xtensa_ae_addsub32:


%ret = call <2 x i32> @llvm.xtensa.ae.addsub32(<2 x i32> %ae_arth_v0, <2 x i32> %ae_arth_v1)
; CHECK: ae_addsub32 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_addsub32s.ll
declare <2 x i32> @llvm.xtensa.ae.addsub32s(<2 x i32>, <2 x i32>)
define <2 x i32> @test_xtensa_ae_addsub32s(<2 x i32> %ae_arth_v0, <2 x i32> %ae_arth_v1) {
; CHECK-LABEL: test_xtensa_ae_addsub32s:


%ret = call <2 x i32> @llvm.xtensa.ae.addsub32s(<2 x i32> %ae_arth_v0, <2 x i32> %ae_arth_v1)
; CHECK: ae_addsub32s aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_and.ll
declare <1 x i64> @llvm.xtensa.ae.and(<1 x i64>, <1 x i64>)
define <1 x i64> @test_xtensa_ae_and(<1 x i64> %ae_dr_to_dr_v0, <1 x i64> %ae_dr_to_dr_v1) {
; CHECK-LABEL: test_xtensa_ae_and:


%ret = call <1 x i64> @llvm.xtensa.ae.and(<1 x i64> %ae_dr_to_dr_v0, <1 x i64> %ae_dr_to_dr_v1)
; CHECK: ae_and aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_cvt32x2f16_10.ll
declare <2 x i32> @llvm.xtensa.ae.cvt32x2f16.10(<4 x i16>)
define <2 x i32> @test_xtensa_ae_cvt32x2f16_10(<4 x i16> %ae_to_dr_v0) {
; CHECK-LABEL: test_xtensa_ae_cvt32x2f16_10:

%ret = call <2 x i32> @llvm.xtensa.ae.cvt32x2f16.10(<4 x i16> %ae_to_dr_v0)
; CHECK: ae_cvt32x2f16.10 aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_cvt32x2f16_32.ll
declare <2 x i32> @llvm.xtensa.ae.cvt32x2f16.32(<4 x i16>)
define <2 x i32> @test_xtensa_ae_cvt32x2f16_32(<4 x i16> %ae_to_dr_v0) {
; CHECK-LABEL: test_xtensa_ae_cvt32x2f16_32:

%ret = call <2 x i32> @llvm.xtensa.ae.cvt32x2f16.32(<4 x i16> %ae_to_dr_v0)
; CHECK: ae_cvt32x2f16.32 aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_cvt48a32.ll
declare <1 x i64> @llvm.xtensa.ae.cvt48a32(i32)
define <1 x i64> @test_xtensa_ae_cvt48a32(i32 %ars) {
; CHECK-LABEL: test_xtensa_ae_cvt48a32:

%ret = call <1 x i64> @llvm.xtensa.ae.cvt48a32(i32 %ars)
; CHECK: ae_cvt48a32 aed{{[0-9]+}}, a{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_cvt64a32.ll
declare <1 x i64> @llvm.xtensa.ae.cvt64a32(i32)
define <1 x i64> @test_xtensa_ae_cvt64a32(i32 %ars) {
; CHECK-LABEL: test_xtensa_ae_cvt64a32:

%ret = call <1 x i64> @llvm.xtensa.ae.cvt64a32(i32 %ars)
; CHECK: ae_cvt64a32 aed{{[0-9]+}}, a{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_cvt64f32_h.ll
declare <1 x i64> @llvm.xtensa.ae.cvt64f32.h(<2 x i32>)
define <1 x i64> @test_xtensa_ae_cvt64f32_h(<2 x i32> %ae_dr_to_dr_v0) {
; CHECK-LABEL: test_xtensa_ae_cvt64f32_h:

%ret = call <1 x i64> @llvm.xtensa.ae.cvt64f32.h(<2 x i32> %ae_dr_to_dr_v0)
; CHECK: ae_cvt64f32.h aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_cvta32f24s_h.ll
declare i32 @llvm.xtensa.ae.cvta32f24s.h(<2 x i32>)
define i32 @test_xtensa_ae_cvta32f24s_h(<2 x i32> %ae_dr_to_ar_v0) {
; CHECK-LABEL: test_xtensa_ae_cvta32f24s_h:

%ret = call i32 @llvm.xtensa.ae.cvta32f24s.h(<2 x i32> %ae_dr_to_ar_v0)
; CHECK: ae_cvta32f24s.h a{{[0-9]+}}, aed{{[0-9]+}}

ret i32 %ret
}


;--- ae_cvta32f24s_l.ll
declare i32 @llvm.xtensa.ae.cvta32f24s.l(<2 x i32>)
define i32 @test_xtensa_ae_cvta32f24s_l(<2 x i32> %ae_dr_to_ar_v0) {
; CHECK-LABEL: test_xtensa_ae_cvta32f24s_l:

%ret = call i32 @llvm.xtensa.ae.cvta32f24s.l(<2 x i32> %ae_dr_to_ar_v0)
; CHECK: ae_cvta32f24s.l a{{[0-9]+}}, aed{{[0-9]+}}

ret i32 %ret
}


;--- ae_cvtq56a32s.ll
declare <1 x i64> @llvm.xtensa.ae.cvtq56a32s(i32)
define <1 x i64> @test_xtensa_ae_cvtq56a32s(i32 %ars) {
; CHECK-LABEL: test_xtensa_ae_cvtq56a32s:

%ret = call <1 x i64> @llvm.xtensa.ae.cvtq56a32s(i32 %ars)
; CHECK: ae_cvtq56a32s aed{{[0-9]+}}, a{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_cvtq56p32s_h.ll
declare <1 x i64> @llvm.xtensa.ae.cvtq56p32s.h(<2 x i32>)
define <1 x i64> @test_xtensa_ae_cvtq56p32s_h(<2 x i32> %ae_dr_to_dr_v0) {
; CHECK-LABEL: test_xtensa_ae_cvtq56p32s_h:

%ret = call <1 x i64> @llvm.xtensa.ae.cvtq56p32s.h(<2 x i32> %ae_dr_to_dr_v0)
; CHECK: ae_cvtq56p32s.h aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_cvtq56p32s_l.ll
declare <1 x i64> @llvm.xtensa.ae.cvtq56p32s.l(<2 x i32>)
define <1 x i64> @test_xtensa_ae_cvtq56p32s_l(<2 x i32> %ae_dr_to_dr_v0) {
; CHECK-LABEL: test_xtensa_ae_cvtq56p32s_l:

%ret = call <1 x i64> @llvm.xtensa.ae.cvtq56p32s.l(<2 x i32> %ae_dr_to_dr_v0)
; CHECK: ae_cvtq56p32s.l aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_db.ll
declare ptr @llvm.xtensa.ae.db(ptr, i32)
define ptr @test_xtensa_ae_db(ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_db:


%ret = call ptr @llvm.xtensa.ae.db(ptr %ars, i32 %art)
; CHECK: ae_db a{{[0-9]+}}, a{{[0-9]+}}

ret ptr %ret
}


;--- ae_db_ic.ll
declare ptr @llvm.xtensa.ae.db.ic(ptr, i32)
define ptr @test_xtensa_ae_db_ic(ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_db_ic:


%ret = call ptr @llvm.xtensa.ae.db.ic(ptr %ars, i32 %art)
; CHECK: ae_db.ic a{{[0-9]+}}, a{{[0-9]+}}

ret ptr %ret
}


;--- ae_db_ip.ll
declare ptr @llvm.xtensa.ae.db.ip(ptr, i32)
define ptr @test_xtensa_ae_db_ip(ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_db_ip:


%ret = call ptr @llvm.xtensa.ae.db.ip(ptr %ars, i32 %art)
; CHECK: ae_db.ip a{{[0-9]+}}, a{{[0-9]+}}

ret ptr %ret
}


;--- ae_dbi.ll
declare ptr @llvm.xtensa.ae.dbi(ptr, i32)
define ptr @test_xtensa_ae_dbi(ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_dbi:


%ret = call ptr @llvm.xtensa.ae.dbi(ptr %ars, i32 1)
; CHECK: ae_dbi a{{[0-9]+}}, {{-?[0-9]+}}

ret ptr %ret
}


;--- ae_dbi_ic.ll
declare ptr @llvm.xtensa.ae.dbi.ic(ptr, i32)
define ptr @test_xtensa_ae_dbi_ic(ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_dbi_ic:


%ret = call ptr @llvm.xtensa.ae.dbi.ic(ptr %ars, i32 1)
; CHECK: ae_dbi.ic a{{[0-9]+}}, {{-?[0-9]+}}

ret ptr %ret
}


;--- ae_dbi_ip.ll
declare ptr @llvm.xtensa.ae.dbi.ip(ptr, i32)
define ptr @test_xtensa_ae_dbi_ip(ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_dbi_ip:


%ret = call ptr @llvm.xtensa.ae.dbi.ip(ptr %ars, i32 1)
; CHECK: ae_dbi.ip a{{[0-9]+}}, {{-?[0-9]+}}

ret ptr %ret
}


;--- ae_div64d32_h.ll
declare <1 x i64> @llvm.xtensa.ae.div64d32.h(<1 x i64>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_div64d32_h(<1 x i64> %ae_arth_v, <2 x i32> %ae_arth_v1) {
; CHECK-LABEL: test_xtensa_ae_div64d32_h:


%ret = call <1 x i64> @llvm.xtensa.ae.div64d32.h(<1 x i64> %ae_arth_v, <2 x i32> %ae_arth_v1)
; CHECK: ae_div64d32.h aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_div64d32_l.ll
declare <1 x i64> @llvm.xtensa.ae.div64d32.l(<1 x i64>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_div64d32_l(<1 x i64> %ae_arth_v, <2 x i32> %ae_arth_v1) {
; CHECK-LABEL: test_xtensa_ae_div64d32_l:


%ret = call <1 x i64> @llvm.xtensa.ae.div64d32.l(<1 x i64> %ae_arth_v, <2 x i32> %ae_arth_v1)
; CHECK: ae_div64d32.l aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_eq64.ll
declare <1 x i1> @llvm.xtensa.ae.eq64(<1 x i64>, <1 x i64>)
define <1 x i1> @test_xtensa_ae_eq64(<1 x i64> %ae_cmpp_v0, <1 x i64> %ae_cmpp_v1) {
; CHECK-LABEL: test_xtensa_ae_eq64:


%ret = call <1 x i1> @llvm.xtensa.ae.eq64(<1 x i64> %ae_cmpp_v0, <1 x i64> %ae_cmpp_v1)
; CHECK: ae_eq64 b{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i1> %ret
}


;--- ae_l16_i.ll
declare <4 x i16> @llvm.xtensa.ae.l16.i(ptr, i32)
define <4 x i16> @test_xtensa_ae_l16_i(ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_l16_i:


%ret = call <4 x i16> @llvm.xtensa.ae.l16.i(ptr %ars, i32 -16)
; CHECK: ae_l16.i aed{{[0-9]+}}, a{{[0-9]+}}, {{-?[0-9]+}}

ret <4 x i16> %ret
}


;--- ae_l16_ip.ll
declare { <4 x i16>, ptr } @llvm.xtensa.ae.l16.ip(ptr, i32)
define <4 x i16> @test_xtensa_ae_l16_ip(ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_l16_ip:


%ret = call { <4 x i16>, ptr } @llvm.xtensa.ae.l16.ip(ptr %ars, i32 -16)
; CHECK: ae_l16.ip aed{{[0-9]+}}, a{{[0-9]+}}, {{-?[0-9]+}}
%ev = extractvalue { <4 x i16>, ptr } %ret, 0
ret <4 x i16> %ev
}


;--- ae_l16_x.ll
declare <4 x i16> @llvm.xtensa.ae.l16.x(ptr, i32)
define <4 x i16> @test_xtensa_ae_l16_x(ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_l16_x:


%ret = call <4 x i16> @llvm.xtensa.ae.l16.x(ptr %ars, i32 %art)
; CHECK: ae_l16.x aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}

ret <4 x i16> %ret
}


;--- ae_l16_xc.ll
declare { <4 x i16>, ptr } @llvm.xtensa.ae.l16.xc(ptr, i32)
define <4 x i16> @test_xtensa_ae_l16_xc(ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_l16_xc:


%ret = call { <4 x i16>, ptr } @llvm.xtensa.ae.l16.xc(ptr %ars, i32 %art)
; CHECK: ae_l16.xc aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <4 x i16>, ptr } %ret, 0
ret <4 x i16> %ev
}


;--- ae_l16_xp.ll
declare { <4 x i16>, ptr } @llvm.xtensa.ae.l16.xp(ptr, i32)
define <4 x i16> @test_xtensa_ae_l16_xp(ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_l16_xp:


%ret = call { <4 x i16>, ptr } @llvm.xtensa.ae.l16.xp(ptr %ars, i32 %art)
; CHECK: ae_l16.xp aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <4 x i16>, ptr } %ret, 0
ret <4 x i16> %ev
}


;--- ae_l16m_i.ll
declare <2 x i32> @llvm.xtensa.ae.l16m.i(ptr, i32)
define <2 x i32> @test_xtensa_ae_l16m_i(ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_l16m_i:


%ret = call <2 x i32> @llvm.xtensa.ae.l16m.i(ptr %ars, i32 -16)
; CHECK: ae_l16m.i aed{{[0-9]+}}, a{{[0-9]+}}, {{-?[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_l16m_iu.ll
declare { <2 x i32>, ptr } @llvm.xtensa.ae.l16m.iu(ptr, i32)
define <2 x i32> @test_xtensa_ae_l16m_iu(ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_l16m_iu:


%ret = call { <2 x i32>, ptr } @llvm.xtensa.ae.l16m.iu(ptr %ars, i32 -16)
; CHECK: ae_l16m.iu aed{{[0-9]+}}, a{{[0-9]+}}, {{-?[0-9]+}}
%ev = extractvalue { <2 x i32>, ptr } %ret, 0
ret <2 x i32> %ev
}


;--- ae_l16m_x.ll
declare <2 x i32> @llvm.xtensa.ae.l16m.x(ptr, i32)
define <2 x i32> @test_xtensa_ae_l16m_x(ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_l16m_x:


%ret = call <2 x i32> @llvm.xtensa.ae.l16m.x(ptr %ars, i32 %art)
; CHECK: ae_l16m.x aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_l16m_xc.ll
declare { <2 x i32>, ptr } @llvm.xtensa.ae.l16m.xc(ptr, i32)
define <2 x i32> @test_xtensa_ae_l16m_xc(ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_l16m_xc:


%ret = call { <2 x i32>, ptr } @llvm.xtensa.ae.l16m.xc(ptr %ars, i32 %art)
; CHECK: ae_l16m.xc aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <2 x i32>, ptr } %ret, 0
ret <2 x i32> %ev
}


;--- ae_l16m_xu.ll
declare { <2 x i32>, ptr } @llvm.xtensa.ae.l16m.xu(ptr, i32)
define <2 x i32> @test_xtensa_ae_l16m_xu(ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_l16m_xu:


%ret = call { <2 x i32>, ptr } @llvm.xtensa.ae.l16m.xu(ptr %ars, i32 %art)
; CHECK: ae_l16m.xu aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <2 x i32>, ptr } %ret, 0
ret <2 x i32> %ev
}


;--- ae_l16x2m_i.ll
declare <2 x i32> @llvm.xtensa.ae.l16x2m.i(ptr, i32)
define <2 x i32> @test_xtensa_ae_l16x2m_i(ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_l16x2m_i:


%ret = call <2 x i32> @llvm.xtensa.ae.l16x2m.i(ptr %ars, i32 -32)
; CHECK: ae_l16x2m.i aed{{[0-9]+}}, a{{[0-9]+}}, {{-?[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_l16x2m_iu.ll
declare { <2 x i32>, ptr } @llvm.xtensa.ae.l16x2m.iu(ptr, i32)
define <2 x i32> @test_xtensa_ae_l16x2m_iu(ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_l16x2m_iu:


%ret = call { <2 x i32>, ptr } @llvm.xtensa.ae.l16x2m.iu(ptr %ars, i32 -32)
; CHECK: ae_l16x2m.iu aed{{[0-9]+}}, a{{[0-9]+}}, {{-?[0-9]+}}
%ev = extractvalue { <2 x i32>, ptr } %ret, 0
ret <2 x i32> %ev
}


;--- ae_l16x2m_x.ll
declare <2 x i32> @llvm.xtensa.ae.l16x2m.x(ptr, i32)
define <2 x i32> @test_xtensa_ae_l16x2m_x(ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_l16x2m_x:


%ret = call <2 x i32> @llvm.xtensa.ae.l16x2m.x(ptr %ars, i32 %art)
; CHECK: ae_l16x2m.x aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_l16x2m_xc.ll
declare { <2 x i32>, ptr } @llvm.xtensa.ae.l16x2m.xc(ptr, i32)
define <2 x i32> @test_xtensa_ae_l16x2m_xc(ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_l16x2m_xc:


%ret = call { <2 x i32>, ptr } @llvm.xtensa.ae.l16x2m.xc(ptr %ars, i32 %art)
; CHECK: ae_l16x2m.xc aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <2 x i32>, ptr } %ret, 0
ret <2 x i32> %ev
}


;--- ae_l16x2m_xu.ll
declare { <2 x i32>, ptr } @llvm.xtensa.ae.l16x2m.xu(ptr, i32)
define <2 x i32> @test_xtensa_ae_l16x2m_xu(ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_l16x2m_xu:


%ret = call { <2 x i32>, ptr } @llvm.xtensa.ae.l16x2m.xu(ptr %ars, i32 %art)
; CHECK: ae_l16x2m.xu aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <2 x i32>, ptr } %ret, 0
ret <2 x i32> %ev
}


;--- ae_l16x4_i.ll
declare <4 x i16> @llvm.xtensa.ae.l16x4.i(ptr, i32)
define <4 x i16> @test_xtensa_ae_l16x4_i(ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_l16x4_i:


%ret = call <4 x i16> @llvm.xtensa.ae.l16x4.i(ptr %ars, i32 -64)
; CHECK: ae_l16x4.i aed{{[0-9]+}}, a{{[0-9]+}}, {{-?[0-9]+}}

ret <4 x i16> %ret
}


;--- ae_l16x4_ip.ll
declare { <4 x i16>, ptr } @llvm.xtensa.ae.l16x4.ip(ptr, i32)
define <4 x i16> @test_xtensa_ae_l16x4_ip(ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_l16x4_ip:


%ret = call { <4 x i16>, ptr } @llvm.xtensa.ae.l16x4.ip(ptr %ars, i32 0)
; CHECK: ae_l16x4.ip aed{{[0-9]+}}, a{{[0-9]+}}, {{-?[0-9]+}}
%ev = extractvalue { <4 x i16>, ptr } %ret, 0
ret <4 x i16> %ev
}


;--- ae_l16x4_ric.ll
declare { <4 x i16>, ptr } @llvm.xtensa.ae.l16x4.ric(ptr)
define <4 x i16> @test_xtensa_ae_l16x4_ric(ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_l16x4_ric:

%ret = call { <4 x i16>, ptr } @llvm.xtensa.ae.l16x4.ric(ptr %ars)
; CHECK: ae_l16x4.ric aed{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <4 x i16>, ptr } %ret, 0
ret <4 x i16> %ev
}


;--- ae_l16x4_rip.ll
declare { <4 x i16>, ptr } @llvm.xtensa.ae.l16x4.rip(ptr)
define <4 x i16> @test_xtensa_ae_l16x4_rip(ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_l16x4_rip:

%ret = call { <4 x i16>, ptr } @llvm.xtensa.ae.l16x4.rip(ptr %ars)
; CHECK: ae_l16x4.rip aed{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <4 x i16>, ptr } %ret, 0
ret <4 x i16> %ev
}


;--- ae_l16x4_x.ll
declare <4 x i16> @llvm.xtensa.ae.l16x4.x(ptr, i32)
define <4 x i16> @test_xtensa_ae_l16x4_x(ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_l16x4_x:


%ret = call <4 x i16> @llvm.xtensa.ae.l16x4.x(ptr %ars, i32 %art)
; CHECK: ae_l16x4.x aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}

ret <4 x i16> %ret
}


;--- ae_l16x4_xc.ll
declare { <4 x i16>, ptr } @llvm.xtensa.ae.l16x4.xc(ptr, i32)
define <4 x i16> @test_xtensa_ae_l16x4_xc(ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_l16x4_xc:


%ret = call { <4 x i16>, ptr } @llvm.xtensa.ae.l16x4.xc(ptr %ars, i32 %art)
; CHECK: ae_l16x4.xc aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <4 x i16>, ptr } %ret, 0
ret <4 x i16> %ev
}


;--- ae_l16x4_xp.ll
declare { <4 x i16>, ptr } @llvm.xtensa.ae.l16x4.xp(ptr, i32)
define <4 x i16> @test_xtensa_ae_l16x4_xp(ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_l16x4_xp:


%ret = call { <4 x i16>, ptr } @llvm.xtensa.ae.l16x4.xp(ptr %ars, i32 %art)
; CHECK: ae_l16x4.xp aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <4 x i16>, ptr } %ret, 0
ret <4 x i16> %ev
}


;--- ae_l32_i.ll
declare <2 x i32> @llvm.xtensa.ae.l32.i(ptr, i32)
define <2 x i32> @test_xtensa_ae_l32_i(ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_l32_i:


%ret = call <2 x i32> @llvm.xtensa.ae.l32.i(ptr %ars, i32 -32)
; CHECK: ae_l32.i aed{{[0-9]+}}, a{{[0-9]+}}, {{-?[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_l32_ip.ll
declare { <2 x i32>, ptr } @llvm.xtensa.ae.l32.ip(ptr, i32)
define <2 x i32> @test_xtensa_ae_l32_ip(ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_l32_ip:


%ret = call { <2 x i32>, ptr } @llvm.xtensa.ae.l32.ip(ptr %ars, i32 -32)
; CHECK: ae_l32.ip aed{{[0-9]+}}, a{{[0-9]+}}, {{-?[0-9]+}}
%ev = extractvalue { <2 x i32>, ptr } %ret, 0
ret <2 x i32> %ev
}


;--- ae_l32_x.ll
declare <2 x i32> @llvm.xtensa.ae.l32.x(ptr, i32)
define <2 x i32> @test_xtensa_ae_l32_x(ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_l32_x:


%ret = call <2 x i32> @llvm.xtensa.ae.l32.x(ptr %ars, i32 %art)
; CHECK: ae_l32.x aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_l32_xc.ll
declare { <2 x i32>, ptr } @llvm.xtensa.ae.l32.xc(ptr, i32)
define <2 x i32> @test_xtensa_ae_l32_xc(ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_l32_xc:


%ret = call { <2 x i32>, ptr } @llvm.xtensa.ae.l32.xc(ptr %ars, i32 %art)
; CHECK: ae_l32.xc aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <2 x i32>, ptr } %ret, 0
ret <2 x i32> %ev
}


;--- ae_l32_xp.ll
declare { <2 x i32>, ptr } @llvm.xtensa.ae.l32.xp(ptr, i32)
define <2 x i32> @test_xtensa_ae_l32_xp(ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_l32_xp:


%ret = call { <2 x i32>, ptr } @llvm.xtensa.ae.l32.xp(ptr %ars, i32 %art)
; CHECK: ae_l32.xp aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <2 x i32>, ptr } %ret, 0
ret <2 x i32> %ev
}


;--- ae_l32f24_i.ll
declare <2 x i32> @llvm.xtensa.ae.l32f24.i(ptr, i32)
define <2 x i32> @test_xtensa_ae_l32f24_i(ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_l32f24_i:


%ret = call <2 x i32> @llvm.xtensa.ae.l32f24.i(ptr %ars, i32 -32)
; CHECK: ae_l32f24.i aed{{[0-9]+}}, a{{[0-9]+}}, {{-?[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_l32f24_ip.ll
declare { <2 x i32>, ptr } @llvm.xtensa.ae.l32f24.ip(ptr, i32)
define <2 x i32> @test_xtensa_ae_l32f24_ip(ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_l32f24_ip:


%ret = call { <2 x i32>, ptr } @llvm.xtensa.ae.l32f24.ip(ptr %ars, i32 -32)
; CHECK: ae_l32f24.ip aed{{[0-9]+}}, a{{[0-9]+}}, {{-?[0-9]+}}
%ev = extractvalue { <2 x i32>, ptr } %ret, 0
ret <2 x i32> %ev
}


;--- ae_l32f24_x.ll
declare <2 x i32> @llvm.xtensa.ae.l32f24.x(ptr, i32)
define <2 x i32> @test_xtensa_ae_l32f24_x(ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_l32f24_x:


%ret = call <2 x i32> @llvm.xtensa.ae.l32f24.x(ptr %ars, i32 %art)
; CHECK: ae_l32f24.x aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_l32f24_xc.ll
declare { <2 x i32>, ptr } @llvm.xtensa.ae.l32f24.xc(ptr, i32)
define <2 x i32> @test_xtensa_ae_l32f24_xc(ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_l32f24_xc:


%ret = call { <2 x i32>, ptr } @llvm.xtensa.ae.l32f24.xc(ptr %ars, i32 %art)
; CHECK: ae_l32f24.xc aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <2 x i32>, ptr } %ret, 0
ret <2 x i32> %ev
}


;--- ae_l32f24_xp.ll
declare { <2 x i32>, ptr } @llvm.xtensa.ae.l32f24.xp(ptr, i32)
define <2 x i32> @test_xtensa_ae_l32f24_xp(ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_l32f24_xp:


%ret = call { <2 x i32>, ptr } @llvm.xtensa.ae.l32f24.xp(ptr %ars, i32 %art)
; CHECK: ae_l32f24.xp aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <2 x i32>, ptr } %ret, 0
ret <2 x i32> %ev
}


;--- ae_l32m_i.ll
declare <1 x i64> @llvm.xtensa.ae.l32m.i(ptr, i32)
define <1 x i64> @test_xtensa_ae_l32m_i(ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_l32m_i:


%ret = call <1 x i64> @llvm.xtensa.ae.l32m.i(ptr %ars, i32 -32)
; CHECK: ae_l32m.i aed{{[0-9]+}}, a{{[0-9]+}}, {{-?[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_l32m_iu.ll
declare { <1 x i64>, ptr } @llvm.xtensa.ae.l32m.iu(ptr, i32)
define <1 x i64> @test_xtensa_ae_l32m_iu(ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_l32m_iu:


%ret = call { <1 x i64>, ptr } @llvm.xtensa.ae.l32m.iu(ptr %ars, i32 -32)
; CHECK: ae_l32m.iu aed{{[0-9]+}}, a{{[0-9]+}}, {{-?[0-9]+}}
%ev = extractvalue { <1 x i64>, ptr } %ret, 0
ret <1 x i64> %ev
}


;--- ae_l32m_x.ll
declare <1 x i64> @llvm.xtensa.ae.l32m.x(ptr, i32)
define <1 x i64> @test_xtensa_ae_l32m_x(ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_l32m_x:


%ret = call <1 x i64> @llvm.xtensa.ae.l32m.x(ptr %ars, i32 %art)
; CHECK: ae_l32m.x aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_l32m_xc.ll
declare { <1 x i64>, ptr } @llvm.xtensa.ae.l32m.xc(ptr, i32)
define <1 x i64> @test_xtensa_ae_l32m_xc(ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_l32m_xc:


%ret = call { <1 x i64>, ptr } @llvm.xtensa.ae.l32m.xc(ptr %ars, i32 %art)
; CHECK: ae_l32m.xc aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <1 x i64>, ptr } %ret, 0
ret <1 x i64> %ev
}


;--- ae_l32m_xu.ll
declare { <1 x i64>, ptr } @llvm.xtensa.ae.l32m.xu(ptr, i32)
define <1 x i64> @test_xtensa_ae_l32m_xu(ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_l32m_xu:


%ret = call { <1 x i64>, ptr } @llvm.xtensa.ae.l32m.xu(ptr %ars, i32 %art)
; CHECK: ae_l32m.xu aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <1 x i64>, ptr } %ret, 0
ret <1 x i64> %ev
}


;--- ae_l32x2_i.ll
declare <2 x i32> @llvm.xtensa.ae.l32x2.i(ptr, i32)
define <2 x i32> @test_xtensa_ae_l32x2_i(ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_l32x2_i:


%ret = call <2 x i32> @llvm.xtensa.ae.l32x2.i(ptr %ars, i32 -64)
; CHECK: ae_l32x2.i aed{{[0-9]+}}, a{{[0-9]+}}, {{-?[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_l32x2_ip.ll
declare { <2 x i32>, ptr } @llvm.xtensa.ae.l32x2.ip(ptr, i32)
define <2 x i32> @test_xtensa_ae_l32x2_ip(ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_l32x2_ip:


%ret = call { <2 x i32>, ptr } @llvm.xtensa.ae.l32x2.ip(ptr %ars, i32 0)
; CHECK: ae_l32x2.ip aed{{[0-9]+}}, a{{[0-9]+}}, {{-?[0-9]+}}
%ev = extractvalue { <2 x i32>, ptr } %ret, 0
ret <2 x i32> %ev
}


;--- ae_l32x2_ric.ll
declare { <2 x i32>, ptr } @llvm.xtensa.ae.l32x2.ric(ptr)
define <2 x i32> @test_xtensa_ae_l32x2_ric(ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_l32x2_ric:

%ret = call { <2 x i32>, ptr } @llvm.xtensa.ae.l32x2.ric(ptr %ars)
; CHECK: ae_l32x2.ric aed{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <2 x i32>, ptr } %ret, 0
ret <2 x i32> %ev
}


;--- ae_l32x2_rip.ll
declare { <2 x i32>, ptr } @llvm.xtensa.ae.l32x2.rip(ptr)
define <2 x i32> @test_xtensa_ae_l32x2_rip(ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_l32x2_rip:

%ret = call { <2 x i32>, ptr } @llvm.xtensa.ae.l32x2.rip(ptr %ars)
; CHECK: ae_l32x2.rip aed{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <2 x i32>, ptr } %ret, 0
ret <2 x i32> %ev
}


;--- ae_l32x2_x.ll
declare <2 x i32> @llvm.xtensa.ae.l32x2.x(ptr, i32)
define <2 x i32> @test_xtensa_ae_l32x2_x(ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_l32x2_x:


%ret = call <2 x i32> @llvm.xtensa.ae.l32x2.x(ptr %ars, i32 %art)
; CHECK: ae_l32x2.x aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_l32x2_xc.ll
declare { <2 x i32>, ptr } @llvm.xtensa.ae.l32x2.xc(ptr, i32)
define <2 x i32> @test_xtensa_ae_l32x2_xc(ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_l32x2_xc:


%ret = call { <2 x i32>, ptr } @llvm.xtensa.ae.l32x2.xc(ptr %ars, i32 %art)
; CHECK: ae_l32x2.xc aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <2 x i32>, ptr } %ret, 0
ret <2 x i32> %ev
}


;--- ae_l32x2_xp.ll
declare { <2 x i32>, ptr } @llvm.xtensa.ae.l32x2.xp(ptr, i32)
define <2 x i32> @test_xtensa_ae_l32x2_xp(ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_l32x2_xp:


%ret = call { <2 x i32>, ptr } @llvm.xtensa.ae.l32x2.xp(ptr %ars, i32 %art)
; CHECK: ae_l32x2.xp aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <2 x i32>, ptr } %ret, 0
ret <2 x i32> %ev
}


;--- ae_l32x2f24_i.ll
declare <2 x i32> @llvm.xtensa.ae.l32x2f24.i(ptr, i32)
define <2 x i32> @test_xtensa_ae_l32x2f24_i(ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_l32x2f24_i:


%ret = call <2 x i32> @llvm.xtensa.ae.l32x2f24.i(ptr %ars, i32 -64)
; CHECK: ae_l32x2f24.i aed{{[0-9]+}}, a{{[0-9]+}}, {{-?[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_l32x2f24_ip.ll
declare { <2 x i32>, ptr } @llvm.xtensa.ae.l32x2f24.ip(ptr, i32)
define <2 x i32> @test_xtensa_ae_l32x2f24_ip(ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_l32x2f24_ip:


%ret = call { <2 x i32>, ptr } @llvm.xtensa.ae.l32x2f24.ip(ptr %ars, i32 0)
; CHECK: ae_l32x2f24.ip aed{{[0-9]+}}, a{{[0-9]+}}, {{-?[0-9]+}}
%ev = extractvalue { <2 x i32>, ptr } %ret, 0
ret <2 x i32> %ev
}


;--- ae_l32x2f24_ric.ll
declare { <2 x i32>, ptr } @llvm.xtensa.ae.l32x2f24.ric(ptr)
define <2 x i32> @test_xtensa_ae_l32x2f24_ric(ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_l32x2f24_ric:

%ret = call { <2 x i32>, ptr } @llvm.xtensa.ae.l32x2f24.ric(ptr %ars)
; CHECK: ae_l32x2f24.ric aed{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <2 x i32>, ptr } %ret, 0
ret <2 x i32> %ev
}


;--- ae_l32x2f24_rip.ll
declare { <2 x i32>, ptr } @llvm.xtensa.ae.l32x2f24.rip(ptr)
define <2 x i32> @test_xtensa_ae_l32x2f24_rip(ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_l32x2f24_rip:

%ret = call { <2 x i32>, ptr } @llvm.xtensa.ae.l32x2f24.rip(ptr %ars)
; CHECK: ae_l32x2f24.rip aed{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <2 x i32>, ptr } %ret, 0
ret <2 x i32> %ev
}


;--- ae_l32x2f24_x.ll
declare <2 x i32> @llvm.xtensa.ae.l32x2f24.x(ptr, i32)
define <2 x i32> @test_xtensa_ae_l32x2f24_x(ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_l32x2f24_x:


%ret = call <2 x i32> @llvm.xtensa.ae.l32x2f24.x(ptr %ars, i32 %art)
; CHECK: ae_l32x2f24.x aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_l32x2f24_xc.ll
declare { <2 x i32>, ptr } @llvm.xtensa.ae.l32x2f24.xc(ptr, i32)
define <2 x i32> @test_xtensa_ae_l32x2f24_xc(ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_l32x2f24_xc:


%ret = call { <2 x i32>, ptr } @llvm.xtensa.ae.l32x2f24.xc(ptr %ars, i32 %art)
; CHECK: ae_l32x2f24.xc aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <2 x i32>, ptr } %ret, 0
ret <2 x i32> %ev
}


;--- ae_l32x2f24_xp.ll
declare { <2 x i32>, ptr } @llvm.xtensa.ae.l32x2f24.xp(ptr, i32)
define <2 x i32> @test_xtensa_ae_l32x2f24_xp(ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_l32x2f24_xp:


%ret = call { <2 x i32>, ptr } @llvm.xtensa.ae.l32x2f24.xp(ptr %ars, i32 %art)
; CHECK: ae_l32x2f24.xp aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <2 x i32>, ptr } %ret, 0
ret <2 x i32> %ev
}


;--- ae_l64_i.ll
declare <1 x i64> @llvm.xtensa.ae.l64.i(ptr, i32)
define <1 x i64> @test_xtensa_ae_l64_i(ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_l64_i:


%ret = call <1 x i64> @llvm.xtensa.ae.l64.i(ptr %ars, i32 -64)
; CHECK: ae_l64.i aed{{[0-9]+}}, a{{[0-9]+}}, {{-?[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_l64_ip.ll
declare { <1 x i64>, ptr } @llvm.xtensa.ae.l64.ip(ptr, i32)
define <1 x i64> @test_xtensa_ae_l64_ip(ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_l64_ip:


%ret = call { <1 x i64>, ptr } @llvm.xtensa.ae.l64.ip(ptr %ars, i32 -64)
; CHECK: ae_l64.ip aed{{[0-9]+}}, a{{[0-9]+}}, {{-?[0-9]+}}
%ev = extractvalue { <1 x i64>, ptr } %ret, 0
ret <1 x i64> %ev
}


;--- ae_l64_x.ll
declare <1 x i64> @llvm.xtensa.ae.l64.x(ptr, i32)
define <1 x i64> @test_xtensa_ae_l64_x(ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_l64_x:


%ret = call <1 x i64> @llvm.xtensa.ae.l64.x(ptr %ars, i32 %art)
; CHECK: ae_l64.x aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_l64_xc.ll
declare { <1 x i64>, ptr } @llvm.xtensa.ae.l64.xc(ptr, i32)
define <1 x i64> @test_xtensa_ae_l64_xc(ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_l64_xc:


%ret = call { <1 x i64>, ptr } @llvm.xtensa.ae.l64.xc(ptr %ars, i32 %art)
; CHECK: ae_l64.xc aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <1 x i64>, ptr } %ret, 0
ret <1 x i64> %ev
}


;--- ae_l64_xp.ll
declare { <1 x i64>, ptr } @llvm.xtensa.ae.l64.xp(ptr, i32)
define <1 x i64> @test_xtensa_ae_l64_xp(ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_l64_xp:


%ret = call { <1 x i64>, ptr } @llvm.xtensa.ae.l64.xp(ptr %ars, i32 %art)
; CHECK: ae_l64.xp aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <1 x i64>, ptr } %ret, 0
ret <1 x i64> %ev
}


;--- ae_la16x4_ic.ll
declare { <4 x i16>, <8 x i8>, ptr } @llvm.xtensa.ae.la16x4.ic(<8 x i8>, ptr)
define <4 x i16> @test_xtensa_ae_la16x4_ic(<1 x i64> %ae_ls_uu, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_la16x4_ic:
%wrap0 = bitcast <1 x i64> %ae_ls_uu to <8 x i8>

%ret = call { <4 x i16>, <8 x i8>, ptr } @llvm.xtensa.ae.la16x4.ic(<8 x i8> %wrap0, ptr %ars)
; CHECK: ae_la16x4.ic aed{{[0-9]+}}, u{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <4 x i16>, <8 x i8>, ptr } %ret, 0
ret <4 x i16> %ev
}


;--- ae_la16x4_ip.ll
declare { <4 x i16>, <8 x i8>, ptr } @llvm.xtensa.ae.la16x4.ip(<8 x i8>, ptr)
define <4 x i16> @test_xtensa_ae_la16x4_ip(<1 x i64> %ae_ls_uu, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_la16x4_ip:
%wrap0 = bitcast <1 x i64> %ae_ls_uu to <8 x i8>

%ret = call { <4 x i16>, <8 x i8>, ptr } @llvm.xtensa.ae.la16x4.ip(<8 x i8> %wrap0, ptr %ars)
; CHECK: ae_la16x4.ip aed{{[0-9]+}}, u{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <4 x i16>, <8 x i8>, ptr } %ret, 0
ret <4 x i16> %ev
}


;--- ae_la16x4_ric.ll
declare { <4 x i16>, <8 x i8>, ptr } @llvm.xtensa.ae.la16x4.ric(<8 x i8>, ptr)
define <4 x i16> @test_xtensa_ae_la16x4_ric(<1 x i64> %ae_ls_uu, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_la16x4_ric:
%wrap0 = bitcast <1 x i64> %ae_ls_uu to <8 x i8>

%ret = call { <4 x i16>, <8 x i8>, ptr } @llvm.xtensa.ae.la16x4.ric(<8 x i8> %wrap0, ptr %ars)
; CHECK: ae_la16x4.ric aed{{[0-9]+}}, u{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <4 x i16>, <8 x i8>, ptr } %ret, 0
ret <4 x i16> %ev
}


;--- ae_la16x4_rip.ll
declare { <4 x i16>, <8 x i8>, ptr } @llvm.xtensa.ae.la16x4.rip(<8 x i8>, ptr)
define <4 x i16> @test_xtensa_ae_la16x4_rip(<1 x i64> %ae_ls_uu, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_la16x4_rip:
%wrap0 = bitcast <1 x i64> %ae_ls_uu to <8 x i8>

%ret = call { <4 x i16>, <8 x i8>, ptr } @llvm.xtensa.ae.la16x4.rip(<8 x i8> %wrap0, ptr %ars)
; CHECK: ae_la16x4.rip aed{{[0-9]+}}, u{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <4 x i16>, <8 x i8>, ptr } %ret, 0
ret <4 x i16> %ev
}


;--- ae_la16x4neg_pc.ll
declare { <8 x i8>, ptr } @llvm.xtensa.ae.la16x4neg.pc(ptr)
define <1 x i64> @test_xtensa_ae_la16x4neg_pc(ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_la16x4neg_pc:

%ret = call { <8 x i8>, ptr } @llvm.xtensa.ae.la16x4neg.pc(ptr %ars)
; CHECK: ae_la16x4neg.pc u{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <8 x i8>, ptr } %ret, 0
%unwrap = bitcast <8 x i8> %ev to <1 x i64>
ret <1 x i64> %unwrap
}


;--- ae_la16x4pos_pc.ll
declare { <8 x i8>, ptr } @llvm.xtensa.ae.la16x4pos.pc(ptr)
define <1 x i64> @test_xtensa_ae_la16x4pos_pc(ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_la16x4pos_pc:

%ret = call { <8 x i8>, ptr } @llvm.xtensa.ae.la16x4pos.pc(ptr %ars)
; CHECK: ae_la16x4pos.pc u{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <8 x i8>, ptr } %ret, 0
%unwrap = bitcast <8 x i8> %ev to <1 x i64>
ret <1 x i64> %unwrap
}


;--- ae_la24_ic.ll
declare { <2 x i32>, <8 x i8>, ptr } @llvm.xtensa.ae.la24.ic(<8 x i8>, ptr)
define <2 x i32> @test_xtensa_ae_la24_ic(<1 x i64> %ae_ls_uu, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_la24_ic:
%wrap0 = bitcast <1 x i64> %ae_ls_uu to <8 x i8>

%ret = call { <2 x i32>, <8 x i8>, ptr } @llvm.xtensa.ae.la24.ic(<8 x i8> %wrap0, ptr %ars)
; CHECK: ae_la24.ic aed{{[0-9]+}}, u{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <2 x i32>, <8 x i8>, ptr } %ret, 0
ret <2 x i32> %ev
}


;--- ae_la24_ip.ll
declare { <2 x i32>, <8 x i8>, ptr } @llvm.xtensa.ae.la24.ip(<8 x i8>, ptr)
define <2 x i32> @test_xtensa_ae_la24_ip(<1 x i64> %ae_ls_uu, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_la24_ip:
%wrap0 = bitcast <1 x i64> %ae_ls_uu to <8 x i8>

%ret = call { <2 x i32>, <8 x i8>, ptr } @llvm.xtensa.ae.la24.ip(<8 x i8> %wrap0, ptr %ars)
; CHECK: ae_la24.ip aed{{[0-9]+}}, u{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <2 x i32>, <8 x i8>, ptr } %ret, 0
ret <2 x i32> %ev
}


;--- ae_la24_ric.ll
declare { <2 x i32>, <8 x i8>, ptr } @llvm.xtensa.ae.la24.ric(<8 x i8>, ptr)
define <2 x i32> @test_xtensa_ae_la24_ric(<1 x i64> %ae_ls_uu, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_la24_ric:
%wrap0 = bitcast <1 x i64> %ae_ls_uu to <8 x i8>

%ret = call { <2 x i32>, <8 x i8>, ptr } @llvm.xtensa.ae.la24.ric(<8 x i8> %wrap0, ptr %ars)
; CHECK: ae_la24.ric aed{{[0-9]+}}, u{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <2 x i32>, <8 x i8>, ptr } %ret, 0
ret <2 x i32> %ev
}


;--- ae_la24_rip.ll
declare { <2 x i32>, <8 x i8>, ptr } @llvm.xtensa.ae.la24.rip(<8 x i8>, ptr)
define <2 x i32> @test_xtensa_ae_la24_rip(<1 x i64> %ae_ls_uu, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_la24_rip:
%wrap0 = bitcast <1 x i64> %ae_ls_uu to <8 x i8>

%ret = call { <2 x i32>, <8 x i8>, ptr } @llvm.xtensa.ae.la24.rip(<8 x i8> %wrap0, ptr %ars)
; CHECK: ae_la24.rip aed{{[0-9]+}}, u{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <2 x i32>, <8 x i8>, ptr } %ret, 0
ret <2 x i32> %ev
}


;--- ae_la24neg_pc.ll
declare { <8 x i8>, ptr } @llvm.xtensa.ae.la24neg.pc(ptr)
define <1 x i64> @test_xtensa_ae_la24neg_pc(ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_la24neg_pc:

%ret = call { <8 x i8>, ptr } @llvm.xtensa.ae.la24neg.pc(ptr %ars)
; CHECK: ae_la24neg.pc u{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <8 x i8>, ptr } %ret, 0
%unwrap = bitcast <8 x i8> %ev to <1 x i64>
ret <1 x i64> %unwrap
}


;--- ae_la24pos_pc.ll
declare { <8 x i8>, ptr } @llvm.xtensa.ae.la24pos.pc(ptr)
define <1 x i64> @test_xtensa_ae_la24pos_pc(ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_la24pos_pc:

%ret = call { <8 x i8>, ptr } @llvm.xtensa.ae.la24pos.pc(ptr %ars)
; CHECK: ae_la24pos.pc u{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <8 x i8>, ptr } %ret, 0
%unwrap = bitcast <8 x i8> %ev to <1 x i64>
ret <1 x i64> %unwrap
}


;--- ae_la24x2_ic.ll
declare { <2 x i32>, <8 x i8>, ptr } @llvm.xtensa.ae.la24x2.ic(<8 x i8>, ptr)
define <2 x i32> @test_xtensa_ae_la24x2_ic(<1 x i64> %ae_ls_uu, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_la24x2_ic:
%wrap0 = bitcast <1 x i64> %ae_ls_uu to <8 x i8>

%ret = call { <2 x i32>, <8 x i8>, ptr } @llvm.xtensa.ae.la24x2.ic(<8 x i8> %wrap0, ptr %ars)
; CHECK: ae_la24x2.ic aed{{[0-9]+}}, u{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <2 x i32>, <8 x i8>, ptr } %ret, 0
ret <2 x i32> %ev
}


;--- ae_la24x2_ip.ll
declare { <2 x i32>, <8 x i8>, ptr } @llvm.xtensa.ae.la24x2.ip(<8 x i8>, ptr)
define <2 x i32> @test_xtensa_ae_la24x2_ip(<1 x i64> %ae_ls_uu, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_la24x2_ip:
%wrap0 = bitcast <1 x i64> %ae_ls_uu to <8 x i8>

%ret = call { <2 x i32>, <8 x i8>, ptr } @llvm.xtensa.ae.la24x2.ip(<8 x i8> %wrap0, ptr %ars)
; CHECK: ae_la24x2.ip aed{{[0-9]+}}, u{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <2 x i32>, <8 x i8>, ptr } %ret, 0
ret <2 x i32> %ev
}


;--- ae_la24x2_ric.ll
declare { <2 x i32>, <8 x i8>, ptr } @llvm.xtensa.ae.la24x2.ric(<8 x i8>, ptr)
define <2 x i32> @test_xtensa_ae_la24x2_ric(<1 x i64> %ae_ls_uu, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_la24x2_ric:
%wrap0 = bitcast <1 x i64> %ae_ls_uu to <8 x i8>

%ret = call { <2 x i32>, <8 x i8>, ptr } @llvm.xtensa.ae.la24x2.ric(<8 x i8> %wrap0, ptr %ars)
; CHECK: ae_la24x2.ric aed{{[0-9]+}}, u{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <2 x i32>, <8 x i8>, ptr } %ret, 0
ret <2 x i32> %ev
}


;--- ae_la24x2_rip.ll
declare { <2 x i32>, <8 x i8>, ptr } @llvm.xtensa.ae.la24x2.rip(<8 x i8>, ptr)
define <2 x i32> @test_xtensa_ae_la24x2_rip(<1 x i64> %ae_ls_uu, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_la24x2_rip:
%wrap0 = bitcast <1 x i64> %ae_ls_uu to <8 x i8>

%ret = call { <2 x i32>, <8 x i8>, ptr } @llvm.xtensa.ae.la24x2.rip(<8 x i8> %wrap0, ptr %ars)
; CHECK: ae_la24x2.rip aed{{[0-9]+}}, u{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <2 x i32>, <8 x i8>, ptr } %ret, 0
ret <2 x i32> %ev
}


;--- ae_la24x2neg_pc.ll
declare { <8 x i8>, ptr } @llvm.xtensa.ae.la24x2neg.pc(ptr)
define <1 x i64> @test_xtensa_ae_la24x2neg_pc(ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_la24x2neg_pc:

%ret = call { <8 x i8>, ptr } @llvm.xtensa.ae.la24x2neg.pc(ptr %ars)
; CHECK: ae_la24x2neg.pc u{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <8 x i8>, ptr } %ret, 0
%unwrap = bitcast <8 x i8> %ev to <1 x i64>
ret <1 x i64> %unwrap
}


;--- ae_la24x2pos_pc.ll
declare { <8 x i8>, ptr } @llvm.xtensa.ae.la24x2pos.pc(ptr)
define <1 x i64> @test_xtensa_ae_la24x2pos_pc(ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_la24x2pos_pc:

%ret = call { <8 x i8>, ptr } @llvm.xtensa.ae.la24x2pos.pc(ptr %ars)
; CHECK: ae_la24x2pos.pc u{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <8 x i8>, ptr } %ret, 0
%unwrap = bitcast <8 x i8> %ev to <1 x i64>
ret <1 x i64> %unwrap
}


;--- ae_la32x2_ic.ll
declare { <2 x i32>, <8 x i8>, ptr } @llvm.xtensa.ae.la32x2.ic(<8 x i8>, ptr)
define <2 x i32> @test_xtensa_ae_la32x2_ic(<1 x i64> %ae_ls_uu, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_la32x2_ic:
%wrap0 = bitcast <1 x i64> %ae_ls_uu to <8 x i8>

%ret = call { <2 x i32>, <8 x i8>, ptr } @llvm.xtensa.ae.la32x2.ic(<8 x i8> %wrap0, ptr %ars)
; CHECK: ae_la32x2.ic aed{{[0-9]+}}, u{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <2 x i32>, <8 x i8>, ptr } %ret, 0
ret <2 x i32> %ev
}


;--- ae_la32x2_ip.ll
declare { <2 x i32>, <8 x i8>, ptr } @llvm.xtensa.ae.la32x2.ip(<8 x i8>, ptr)
define <2 x i32> @test_xtensa_ae_la32x2_ip(<1 x i64> %ae_ls_uu, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_la32x2_ip:
%wrap0 = bitcast <1 x i64> %ae_ls_uu to <8 x i8>

%ret = call { <2 x i32>, <8 x i8>, ptr } @llvm.xtensa.ae.la32x2.ip(<8 x i8> %wrap0, ptr %ars)
; CHECK: ae_la32x2.ip aed{{[0-9]+}}, u{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <2 x i32>, <8 x i8>, ptr } %ret, 0
ret <2 x i32> %ev
}


;--- ae_la32x2_ric.ll
declare { <2 x i32>, <8 x i8>, ptr } @llvm.xtensa.ae.la32x2.ric(<8 x i8>, ptr)
define <2 x i32> @test_xtensa_ae_la32x2_ric(<1 x i64> %ae_ls_uu, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_la32x2_ric:
%wrap0 = bitcast <1 x i64> %ae_ls_uu to <8 x i8>

%ret = call { <2 x i32>, <8 x i8>, ptr } @llvm.xtensa.ae.la32x2.ric(<8 x i8> %wrap0, ptr %ars)
; CHECK: ae_la32x2.ric aed{{[0-9]+}}, u{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <2 x i32>, <8 x i8>, ptr } %ret, 0
ret <2 x i32> %ev
}


;--- ae_la32x2_rip.ll
declare { <2 x i32>, <8 x i8>, ptr } @llvm.xtensa.ae.la32x2.rip(<8 x i8>, ptr)
define <2 x i32> @test_xtensa_ae_la32x2_rip(<1 x i64> %ae_ls_uu, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_la32x2_rip:
%wrap0 = bitcast <1 x i64> %ae_ls_uu to <8 x i8>

%ret = call { <2 x i32>, <8 x i8>, ptr } @llvm.xtensa.ae.la32x2.rip(<8 x i8> %wrap0, ptr %ars)
; CHECK: ae_la32x2.rip aed{{[0-9]+}}, u{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <2 x i32>, <8 x i8>, ptr } %ret, 0
ret <2 x i32> %ev
}


;--- ae_la32x2f24_ic.ll
declare { <2 x i32>, <8 x i8>, ptr } @llvm.xtensa.ae.la32x2f24.ic(<8 x i8>, ptr)
define <2 x i32> @test_xtensa_ae_la32x2f24_ic(<1 x i64> %ae_ls_uu, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_la32x2f24_ic:
%wrap0 = bitcast <1 x i64> %ae_ls_uu to <8 x i8>

%ret = call { <2 x i32>, <8 x i8>, ptr } @llvm.xtensa.ae.la32x2f24.ic(<8 x i8> %wrap0, ptr %ars)
; CHECK: ae_la32x2f24.ic aed{{[0-9]+}}, u{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <2 x i32>, <8 x i8>, ptr } %ret, 0
ret <2 x i32> %ev
}


;--- ae_la32x2f24_ip.ll
declare { <2 x i32>, <8 x i8>, ptr } @llvm.xtensa.ae.la32x2f24.ip(<8 x i8>, ptr)
define <2 x i32> @test_xtensa_ae_la32x2f24_ip(<1 x i64> %ae_ls_uu, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_la32x2f24_ip:
%wrap0 = bitcast <1 x i64> %ae_ls_uu to <8 x i8>

%ret = call { <2 x i32>, <8 x i8>, ptr } @llvm.xtensa.ae.la32x2f24.ip(<8 x i8> %wrap0, ptr %ars)
; CHECK: ae_la32x2f24.ip aed{{[0-9]+}}, u{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <2 x i32>, <8 x i8>, ptr } %ret, 0
ret <2 x i32> %ev
}


;--- ae_la32x2f24_ric.ll
declare { <2 x i32>, <8 x i8>, ptr } @llvm.xtensa.ae.la32x2f24.ric(<8 x i8>, ptr)
define <2 x i32> @test_xtensa_ae_la32x2f24_ric(<1 x i64> %ae_ls_uu, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_la32x2f24_ric:
%wrap0 = bitcast <1 x i64> %ae_ls_uu to <8 x i8>

%ret = call { <2 x i32>, <8 x i8>, ptr } @llvm.xtensa.ae.la32x2f24.ric(<8 x i8> %wrap0, ptr %ars)
; CHECK: ae_la32x2f24.ric aed{{[0-9]+}}, u{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <2 x i32>, <8 x i8>, ptr } %ret, 0
ret <2 x i32> %ev
}


;--- ae_la32x2f24_rip.ll
declare { <2 x i32>, <8 x i8>, ptr } @llvm.xtensa.ae.la32x2f24.rip(<8 x i8>, ptr)
define <2 x i32> @test_xtensa_ae_la32x2f24_rip(<1 x i64> %ae_ls_uu, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_la32x2f24_rip:
%wrap0 = bitcast <1 x i64> %ae_ls_uu to <8 x i8>

%ret = call { <2 x i32>, <8 x i8>, ptr } @llvm.xtensa.ae.la32x2f24.rip(<8 x i8> %wrap0, ptr %ars)
; CHECK: ae_la32x2f24.rip aed{{[0-9]+}}, u{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <2 x i32>, <8 x i8>, ptr } %ret, 0
ret <2 x i32> %ev
}


;--- ae_la32x2neg_pc.ll
declare { <8 x i8>, ptr } @llvm.xtensa.ae.la32x2neg.pc(ptr)
define <1 x i64> @test_xtensa_ae_la32x2neg_pc(ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_la32x2neg_pc:

%ret = call { <8 x i8>, ptr } @llvm.xtensa.ae.la32x2neg.pc(ptr %ars)
; CHECK: ae_la32x2neg.pc u{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <8 x i8>, ptr } %ret, 0
%unwrap = bitcast <8 x i8> %ev to <1 x i64>
ret <1 x i64> %unwrap
}


;--- ae_la32x2pos_pc.ll
declare { <8 x i8>, ptr } @llvm.xtensa.ae.la32x2pos.pc(ptr)
define <1 x i64> @test_xtensa_ae_la32x2pos_pc(ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_la32x2pos_pc:

%ret = call { <8 x i8>, ptr } @llvm.xtensa.ae.la32x2pos.pc(ptr %ars)
; CHECK: ae_la32x2pos.pc u{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <8 x i8>, ptr } %ret, 0
%unwrap = bitcast <8 x i8> %ev to <1 x i64>
ret <1 x i64> %unwrap
}


;--- ae_la64_pp.ll
declare <8 x i8> @llvm.xtensa.ae.la64.pp(ptr)
define <8 x i8> @test_xtensa_ae_la64_pp(ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_la64_pp:

%ret = call <8 x i8> @llvm.xtensa.ae.la64.pp(ptr %ars)
; CHECK: ae_la64.pp u{{[0-9]+}}, a{{[0-9]+}}

ret <8 x i8> %ret
}


;--- ae_lalign64_i.ll
declare <8 x i8> @llvm.xtensa.ae.lalign64.i(ptr, i32)
define <8 x i8> @test_xtensa_ae_lalign64_i(ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_lalign64_i:


%ret = call <8 x i8> @llvm.xtensa.ae.lalign64.i(ptr %ars, i32 -64)
; CHECK: ae_lalign64.i u{{[0-9]+}}, a{{[0-9]+}}, {{-?[0-9]+}}

ret <8 x i8> %ret
}


;--- ae_lb.ll
declare i32 @llvm.xtensa.ae.lb(i32)
define i32 @test_xtensa_ae_lb(i32 %art) {
; CHECK-LABEL: test_xtensa_ae_lb:

%ret = call i32 @llvm.xtensa.ae.lb(i32 %art)
; CHECK: ae_lb a{{[0-9]+}}, a{{[0-9]+}}

ret i32 %ret
}


;--- ae_lbi.ll
declare i32 @llvm.xtensa.ae.lbi(i32)
define i32 @test_xtensa_ae_lbi() {
; CHECK-LABEL: test_xtensa_ae_lbi:

%ret = call i32 @llvm.xtensa.ae.lbi(i32 1)
; CHECK: ae_lbi a{{[0-9]+}}, {{-?[0-9]+}}

ret i32 %ret
}


;--- ae_lbk.ll
declare i32 @llvm.xtensa.ae.lbk(i32, i32)
define i32 @test_xtensa_ae_lbk(i32 %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_lbk:


%ret = call i32 @llvm.xtensa.ae.lbk(i32 %ars, i32 %art)
; CHECK: ae_lbk a{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}

ret i32 %ret
}


;--- ae_lbki.ll
declare i32 @llvm.xtensa.ae.lbki(i32, i32)
define i32 @test_xtensa_ae_lbki(i32 %ars) {
; CHECK-LABEL: test_xtensa_ae_lbki:


%ret = call i32 @llvm.xtensa.ae.lbki(i32 %ars, i32 1)
; CHECK: ae_lbki a{{[0-9]+}}, a{{[0-9]+}}, {{-?[0-9]+}}

ret i32 %ret
}


;--- ae_lbs.ll
declare i32 @llvm.xtensa.ae.lbs(i32)
define i32 @test_xtensa_ae_lbs(i32 %art) {
; CHECK-LABEL: test_xtensa_ae_lbs:

%ret = call i32 @llvm.xtensa.ae.lbs(i32 %art)
; CHECK: ae_lbs a{{[0-9]+}}, a{{[0-9]+}}

ret i32 %ret
}


;--- ae_lbsi.ll
declare i32 @llvm.xtensa.ae.lbsi(i32)
define i32 @test_xtensa_ae_lbsi() {
; CHECK-LABEL: test_xtensa_ae_lbsi:

%ret = call i32 @llvm.xtensa.ae.lbsi(i32 1)
; CHECK: ae_lbsi a{{[0-9]+}}, {{-?[0-9]+}}

ret i32 %ret
}


;--- ae_le64.ll
declare <1 x i1> @llvm.xtensa.ae.le64(<1 x i64>, <1 x i64>)
define <1 x i1> @test_xtensa_ae_le64(<1 x i64> %ae_cmpp_v0, <1 x i64> %ae_cmpp_v1) {
; CHECK-LABEL: test_xtensa_ae_le64:


%ret = call <1 x i1> @llvm.xtensa.ae.le64(<1 x i64> %ae_cmpp_v0, <1 x i64> %ae_cmpp_v1)
; CHECK: ae_le64 b{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i1> %ret
}


;--- ae_lt64.ll
declare <1 x i1> @llvm.xtensa.ae.lt64(<1 x i64>, <1 x i64>)
define <1 x i1> @test_xtensa_ae_lt64(<1 x i64> %ae_cmpp_v0, <1 x i64> %ae_cmpp_v1) {
; CHECK-LABEL: test_xtensa_ae_lt64:


%ret = call <1 x i1> @llvm.xtensa.ae.lt64(<1 x i64> %ae_cmpp_v0, <1 x i64> %ae_cmpp_v1)
; CHECK: ae_lt64 b{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i1> %ret
}


;--- ae_max32.ll
declare <2 x i32> @llvm.xtensa.ae.max32(<2 x i32>, <2 x i32>)
define <2 x i32> @test_xtensa_ae_max32(<2 x i32> %ae_cmpp_v0, <2 x i32> %ae_cmpp_v1) {
; CHECK-LABEL: test_xtensa_ae_max32:


%ret = call <2 x i32> @llvm.xtensa.ae.max32(<2 x i32> %ae_cmpp_v0, <2 x i32> %ae_cmpp_v1)
; CHECK: ae_max32 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_max64.ll
declare <1 x i64> @llvm.xtensa.ae.max64(<1 x i64>, <1 x i64>)
define <1 x i64> @test_xtensa_ae_max64(<1 x i64> %ae_cmpp_v0, <1 x i64> %ae_cmpp_v1) {
; CHECK-LABEL: test_xtensa_ae_max64:


%ret = call <1 x i64> @llvm.xtensa.ae.max64(<1 x i64> %ae_cmpp_v0, <1 x i64> %ae_cmpp_v1)
; CHECK: ae_max64 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_maxabs32s.ll
declare <2 x i32> @llvm.xtensa.ae.maxabs32s(<2 x i32>, <2 x i32>)
define <2 x i32> @test_xtensa_ae_maxabs32s(<2 x i32> %ae_cmpp_v0, <2 x i32> %ae_cmpp_v1) {
; CHECK-LABEL: test_xtensa_ae_maxabs32s:


%ret = call <2 x i32> @llvm.xtensa.ae.maxabs32s(<2 x i32> %ae_cmpp_v0, <2 x i32> %ae_cmpp_v1)
; CHECK: ae_maxabs32s aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_maxabs64s.ll
declare <1 x i64> @llvm.xtensa.ae.maxabs64s(<1 x i64>, <1 x i64>)
define <1 x i64> @test_xtensa_ae_maxabs64s(<1 x i64> %ae_cmpp_v0, <1 x i64> %ae_cmpp_v1) {
; CHECK-LABEL: test_xtensa_ae_maxabs64s:


%ret = call <1 x i64> @llvm.xtensa.ae.maxabs64s(<1 x i64> %ae_cmpp_v0, <1 x i64> %ae_cmpp_v1)
; CHECK: ae_maxabs64s aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_min32.ll
declare <2 x i32> @llvm.xtensa.ae.min32(<2 x i32>, <2 x i32>)
define <2 x i32> @test_xtensa_ae_min32(<2 x i32> %ae_cmpp_v0, <2 x i32> %ae_cmpp_v1) {
; CHECK-LABEL: test_xtensa_ae_min32:


%ret = call <2 x i32> @llvm.xtensa.ae.min32(<2 x i32> %ae_cmpp_v0, <2 x i32> %ae_cmpp_v1)
; CHECK: ae_min32 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_min64.ll
declare <1 x i64> @llvm.xtensa.ae.min64(<1 x i64>, <1 x i64>)
define <1 x i64> @test_xtensa_ae_min64(<1 x i64> %ae_cmpp_v0, <1 x i64> %ae_cmpp_v1) {
; CHECK-LABEL: test_xtensa_ae_min64:


%ret = call <1 x i64> @llvm.xtensa.ae.min64(<1 x i64> %ae_cmpp_v0, <1 x i64> %ae_cmpp_v1)
; CHECK: ae_min64 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_minabs32s.ll
declare <2 x i32> @llvm.xtensa.ae.minabs32s(<2 x i32>, <2 x i32>)
define <2 x i32> @test_xtensa_ae_minabs32s(<2 x i32> %ae_cmpp_v0, <2 x i32> %ae_cmpp_v1) {
; CHECK-LABEL: test_xtensa_ae_minabs32s:


%ret = call <2 x i32> @llvm.xtensa.ae.minabs32s(<2 x i32> %ae_cmpp_v0, <2 x i32> %ae_cmpp_v1)
; CHECK: ae_minabs32s aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_minabs64s.ll
declare <1 x i64> @llvm.xtensa.ae.minabs64s(<1 x i64>, <1 x i64>)
define <1 x i64> @test_xtensa_ae_minabs64s(<1 x i64> %ae_cmpp_v0, <1 x i64> %ae_cmpp_v1) {
; CHECK-LABEL: test_xtensa_ae_minabs64s:


%ret = call <1 x i64> @llvm.xtensa.ae.minabs64s(<1 x i64> %ae_cmpp_v0, <1 x i64> %ae_cmpp_v1)
; CHECK: ae_minabs64s aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mov.ll
declare <1 x i64> @llvm.xtensa.ae.mov(<1 x i64>)
define <1 x i64> @test_xtensa_ae_mov(<1 x i64> %ae_to_dr_v0) {
; CHECK-LABEL: test_xtensa_ae_mov:

%ret = call <1 x i64> @llvm.xtensa.ae.mov(<1 x i64> %ae_to_dr_v0)
; CHECK: ae_mov aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_movad16_0.ll
declare i32 @llvm.xtensa.ae.movad16.0(<4 x i16>)
define i32 @test_xtensa_ae_movad16_0(<4 x i16> %ae_dr_to_ar_v0) {
; CHECK-LABEL: test_xtensa_ae_movad16_0:

%ret = call i32 @llvm.xtensa.ae.movad16.0(<4 x i16> %ae_dr_to_ar_v0)
; CHECK: ae_movad16.0 a{{[0-9]+}}, aed{{[0-9]+}}

ret i32 %ret
}


;--- ae_movad16_1.ll
declare i32 @llvm.xtensa.ae.movad16.1(<4 x i16>)
define i32 @test_xtensa_ae_movad16_1(<4 x i16> %ae_dr_to_ar_v0) {
; CHECK-LABEL: test_xtensa_ae_movad16_1:

%ret = call i32 @llvm.xtensa.ae.movad16.1(<4 x i16> %ae_dr_to_ar_v0)
; CHECK: ae_movad16.1 a{{[0-9]+}}, aed{{[0-9]+}}

ret i32 %ret
}


;--- ae_movad16_2.ll
declare i32 @llvm.xtensa.ae.movad16.2(<4 x i16>)
define i32 @test_xtensa_ae_movad16_2(<4 x i16> %ae_dr_to_ar_v0) {
; CHECK-LABEL: test_xtensa_ae_movad16_2:

%ret = call i32 @llvm.xtensa.ae.movad16.2(<4 x i16> %ae_dr_to_ar_v0)
; CHECK: ae_movad16.2 a{{[0-9]+}}, aed{{[0-9]+}}

ret i32 %ret
}


;--- ae_movad16_3.ll
declare i32 @llvm.xtensa.ae.movad16.3(<4 x i16>)
define i32 @test_xtensa_ae_movad16_3(<4 x i16> %ae_dr_to_ar_v0) {
; CHECK-LABEL: test_xtensa_ae_movad16_3:

%ret = call i32 @llvm.xtensa.ae.movad16.3(<4 x i16> %ae_dr_to_ar_v0)
; CHECK: ae_movad16.3 a{{[0-9]+}}, aed{{[0-9]+}}

ret i32 %ret
}


;--- ae_movad32_h.ll
declare i32 @llvm.xtensa.ae.movad32.h(<2 x i32>)
define i32 @test_xtensa_ae_movad32_h(<2 x i32> %ae_dr_to_ar_v0) {
; CHECK-LABEL: test_xtensa_ae_movad32_h:

%ret = call i32 @llvm.xtensa.ae.movad32.h(<2 x i32> %ae_dr_to_ar_v0)
; CHECK: ae_movad32.h a{{[0-9]+}}, aed{{[0-9]+}}

ret i32 %ret
}


;--- ae_movad32_l.ll
declare i32 @llvm.xtensa.ae.movad32.l(<2 x i32>)
define i32 @test_xtensa_ae_movad32_l(<2 x i32> %ae_dr_to_ar_v0) {
; CHECK-LABEL: test_xtensa_ae_movad32_l:

%ret = call i32 @llvm.xtensa.ae.movad32.l(<2 x i32> %ae_dr_to_ar_v0)
; CHECK: ae_movad32.l a{{[0-9]+}}, aed{{[0-9]+}}

ret i32 %ret
}


;--- ae_movalign.ll
declare <8 x i8> @llvm.xtensa.ae.movalign(<8 x i8>)
define <8 x i8> @test_xtensa_ae_movalign(<1 x i64> %ae_uu_v) {
; CHECK-LABEL: test_xtensa_ae_movalign:
%wrap0 = bitcast <1 x i64> %ae_uu_v to <8 x i8>
%ret = call <8 x i8> @llvm.xtensa.ae.movalign(<8 x i8> %wrap0)
; CHECK: ae_movalign u{{[0-9]+}}, u{{[0-9]+}}

ret <8 x i8> %ret
}


;--- ae_movda16.ll
declare <4 x i16> @llvm.xtensa.ae.movda16(i32)
define <4 x i16> @test_xtensa_ae_movda16(i32 %ars) {
; CHECK-LABEL: test_xtensa_ae_movda16:

%ret = call <4 x i16> @llvm.xtensa.ae.movda16(i32 %ars)
; CHECK: ae_movda16 aed{{[0-9]+}}, a{{[0-9]+}}

ret <4 x i16> %ret
}


;--- ae_movda16x2.ll
declare <4 x i16> @llvm.xtensa.ae.movda16x2(i32, i32)
define <4 x i16> @test_xtensa_ae_movda16x2(i32 %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_movda16x2:


%ret = call <4 x i16> @llvm.xtensa.ae.movda16x2(i32 %ars, i32 %art)
; CHECK: ae_movda16x2 aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}

ret <4 x i16> %ret
}


;--- ae_movda32.ll
declare <1 x i32> @llvm.xtensa.ae.movda32(i32)
define <1 x i32> @test_xtensa_ae_movda32(i32 %ars) {
; CHECK-LABEL: test_xtensa_ae_movda32:

%ret = call <1 x i32> @llvm.xtensa.ae.movda32(i32 %ars)
; CHECK: ae_movda32 aed{{[0-9]+}}, a{{[0-9]+}}

ret <1 x i32> %ret
}


;--- ae_movda32x2.ll
declare <2 x i32> @llvm.xtensa.ae.movda32x2(i32, i32)
define <2 x i32> @test_xtensa_ae_movda32x2(i32 %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_movda32x2:


%ret = call <2 x i32> @llvm.xtensa.ae.movda32x2(i32 %ars, i32 %art)
; CHECK: ae_movda32x2 aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_movf64.ll
declare <1 x i64> @llvm.xtensa.ae.movf64(<1 x i64>, <1 x i64>, <1 x i1>)
define <1 x i64> @test_xtensa_ae_movf64(<1 x i64> %ae_cmov_v, <1 x i64> %ae_cmov_v0, <1 x i1> %bt) {
; CHECK-LABEL: test_xtensa_ae_movf64:



%ret = call <1 x i64> @llvm.xtensa.ae.movf64(<1 x i64> %ae_cmov_v, <1 x i64> %ae_cmov_v0, <1 x i1> %bt)
; CHECK: ae_movf64 aed{{[0-9]+}}, aed{{[0-9]+}}, b{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_movi.ll
declare <2 x i32> @llvm.xtensa.ae.movi(i32)
define <2 x i32> @test_xtensa_ae_movi() {
; CHECK-LABEL: test_xtensa_ae_movi:

%ret = call <2 x i32> @llvm.xtensa.ae.movi(i32 -16)
; CHECK: ae_movi aed{{[0-9]+}}, {{-?[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_movt64.ll
declare <1 x i64> @llvm.xtensa.ae.movt64(<1 x i64>, <1 x i64>, <1 x i1>)
define <1 x i64> @test_xtensa_ae_movt64(<1 x i64> %ae_cmov_v, <1 x i64> %ae_cmov_v0, <1 x i1> %bt) {
; CHECK-LABEL: test_xtensa_ae_movt64:



%ret = call <1 x i64> @llvm.xtensa.ae.movt64(<1 x i64> %ae_cmov_v, <1 x i64> %ae_cmov_v0, <1 x i1> %bt)
; CHECK: ae_movt64 aed{{[0-9]+}}, aed{{[0-9]+}}, b{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mul16x4.ll
declare { <2 x i32>, <2 x i32> } @llvm.xtensa.ae.mul16x4(<4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mul16x4(<4 x i16> %ae_mul_d1, <4 x i16> %ae_mul_d0) {
; CHECK-LABEL: test_xtensa_ae_mul16x4:


%ret = call { <2 x i32>, <2 x i32> } @llvm.xtensa.ae.mul16x4(<4 x i16> %ae_mul_d1, <4 x i16> %ae_mul_d0)
; CHECK: ae_mul16x4 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}
%ev = extractvalue { <2 x i32>, <2 x i32> } %ret, 0
ret <2 x i32> %ev
}


;--- ae_mul32_hh.ll
declare <1 x i64> @llvm.xtensa.ae.mul32.hh(<2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mul32_hh(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mul32_hh:


%ret = call <1 x i64> @llvm.xtensa.ae.mul32.hh(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mul32.hh aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mul32_lh.ll
declare <1 x i64> @llvm.xtensa.ae.mul32.lh(<2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mul32_lh(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mul32_lh:


%ret = call <1 x i64> @llvm.xtensa.ae.mul32.lh(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mul32.lh aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mul32_ll.ll
declare <1 x i64> @llvm.xtensa.ae.mul32.ll(<2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mul32_ll(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mul32_ll:


%ret = call <1 x i64> @llvm.xtensa.ae.mul32.ll(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mul32.ll aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mul32_ll_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mul32.ll.s2(<2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mul32_ll_s2(<2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mul32_ll_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mul32.ll.s2(<2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mul32.ll_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mul32u_ll.ll
declare <1 x i64> @llvm.xtensa.ae.mul32u.ll(<2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mul32u_ll(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mul32u_ll:


%ret = call <1 x i64> @llvm.xtensa.ae.mul32u.ll(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mul32u.ll aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mul32x16_h0.ll
declare <1 x i64> @llvm.xtensa.ae.mul32x16.h0(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mul32x16_h0(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mul32x16_h0:


%ret = call <1 x i64> @llvm.xtensa.ae.mul32x16.h0(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mul32x16.h0 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mul32x16_h0_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mul32x16.h0.s2(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mul32x16_h0_s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mul32x16_h0_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mul32x16.h0.s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mul32x16.h0_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mul32x16_h1.ll
declare <1 x i64> @llvm.xtensa.ae.mul32x16.h1(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mul32x16_h1(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mul32x16_h1:


%ret = call <1 x i64> @llvm.xtensa.ae.mul32x16.h1(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mul32x16.h1 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mul32x16_h1_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mul32x16.h1.s2(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mul32x16_h1_s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mul32x16_h1_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mul32x16.h1.s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mul32x16.h1_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mul32x16_h2.ll
declare <1 x i64> @llvm.xtensa.ae.mul32x16.h2(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mul32x16_h2(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mul32x16_h2:


%ret = call <1 x i64> @llvm.xtensa.ae.mul32x16.h2(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mul32x16.h2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mul32x16_h2_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mul32x16.h2.s2(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mul32x16_h2_s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mul32x16_h2_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mul32x16.h2.s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mul32x16.h2_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mul32x16_h3.ll
declare <1 x i64> @llvm.xtensa.ae.mul32x16.h3(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mul32x16_h3(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mul32x16_h3:


%ret = call <1 x i64> @llvm.xtensa.ae.mul32x16.h3(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mul32x16.h3 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mul32x16_h3_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mul32x16.h3.s2(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mul32x16_h3_s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mul32x16_h3_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mul32x16.h3.s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mul32x16.h3_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mul32x16_l0.ll
declare <1 x i64> @llvm.xtensa.ae.mul32x16.l0(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mul32x16_l0(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mul32x16_l0:


%ret = call <1 x i64> @llvm.xtensa.ae.mul32x16.l0(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mul32x16.l0 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mul32x16_l0_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mul32x16.l0.s2(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mul32x16_l0_s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mul32x16_l0_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mul32x16.l0.s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mul32x16.l0_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mul32x16_l1.ll
declare <1 x i64> @llvm.xtensa.ae.mul32x16.l1(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mul32x16_l1(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mul32x16_l1:


%ret = call <1 x i64> @llvm.xtensa.ae.mul32x16.l1(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mul32x16.l1 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mul32x16_l1_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mul32x16.l1.s2(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mul32x16_l1_s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mul32x16_l1_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mul32x16.l1.s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mul32x16.l1_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mul32x16_l2.ll
declare <1 x i64> @llvm.xtensa.ae.mul32x16.l2(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mul32x16_l2(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mul32x16_l2:


%ret = call <1 x i64> @llvm.xtensa.ae.mul32x16.l2(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mul32x16.l2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mul32x16_l2_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mul32x16.l2.s2(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mul32x16_l2_s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mul32x16_l2_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mul32x16.l2.s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mul32x16.l2_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mul32x16_l3.ll
declare <1 x i64> @llvm.xtensa.ae.mul32x16.l3(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mul32x16_l3(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mul32x16_l3:


%ret = call <1 x i64> @llvm.xtensa.ae.mul32x16.l3(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mul32x16.l3 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mul32x16_l3_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mul32x16.l3.s2(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mul32x16_l3_s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mul32x16_l3_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mul32x16.l3.s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mul32x16.l3_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mula16x4.ll
declare { <2 x i32>, <2 x i32> } @llvm.xtensa.ae.mula16x4(<2 x i32>, <2 x i32>, <4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mula16x4(<2 x i32> %ae_mul_q1, <2 x i32> %ae_mul_q0, <4 x i16> %ae_mul_d1, <4 x i16> %ae_mul_d0) {
; CHECK-LABEL: test_xtensa_ae_mula16x4:




%ret = call { <2 x i32>, <2 x i32> } @llvm.xtensa.ae.mula16x4(<2 x i32> %ae_mul_q1, <2 x i32> %ae_mul_q0, <4 x i16> %ae_mul_d1, <4 x i16> %ae_mul_d0)
; CHECK: ae_mula16x4 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}
%ev = extractvalue { <2 x i32>, <2 x i32> } %ret, 0
ret <2 x i32> %ev
}


;--- ae_mula32_hh.ll
declare <1 x i64> @llvm.xtensa.ae.mula32.hh(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mula32_hh(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mula32_hh:



%ret = call <1 x i64> @llvm.xtensa.ae.mula32.hh(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mula32.hh aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mula32_lh.ll
declare <1 x i64> @llvm.xtensa.ae.mula32.lh(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mula32_lh(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mula32_lh:



%ret = call <1 x i64> @llvm.xtensa.ae.mula32.lh(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mula32.lh aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mula32_ll.ll
declare <1 x i64> @llvm.xtensa.ae.mula32.ll(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mula32_ll(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mula32_ll:



%ret = call <1 x i64> @llvm.xtensa.ae.mula32.ll(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mula32.ll aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mula32_ll_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mula32.ll.s2(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mula32_ll_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mula32_ll_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mula32.ll.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mula32.ll_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mula32u_ll.ll
declare <1 x i64> @llvm.xtensa.ae.mula32u.ll(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mula32u_ll(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mula32u_ll:



%ret = call <1 x i64> @llvm.xtensa.ae.mula32u.ll(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mula32u.ll aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mula32x16_h0.ll
declare <1 x i64> @llvm.xtensa.ae.mula32x16.h0(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mula32x16_h0(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mula32x16_h0:



%ret = call <1 x i64> @llvm.xtensa.ae.mula32x16.h0(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mula32x16.h0 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mula32x16_h0_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mula32x16.h0.s2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mula32x16_h0_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mula32x16_h0_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mula32x16.h0.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mula32x16.h0_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mula32x16_h1.ll
declare <1 x i64> @llvm.xtensa.ae.mula32x16.h1(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mula32x16_h1(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mula32x16_h1:



%ret = call <1 x i64> @llvm.xtensa.ae.mula32x16.h1(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mula32x16.h1 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mula32x16_h1_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mula32x16.h1.s2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mula32x16_h1_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mula32x16_h1_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mula32x16.h1.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mula32x16.h1_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mula32x16_h2.ll
declare <1 x i64> @llvm.xtensa.ae.mula32x16.h2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mula32x16_h2(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mula32x16_h2:



%ret = call <1 x i64> @llvm.xtensa.ae.mula32x16.h2(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mula32x16.h2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mula32x16_h2_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mula32x16.h2.s2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mula32x16_h2_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mula32x16_h2_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mula32x16.h2.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mula32x16.h2_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mula32x16_h3.ll
declare <1 x i64> @llvm.xtensa.ae.mula32x16.h3(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mula32x16_h3(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mula32x16_h3:



%ret = call <1 x i64> @llvm.xtensa.ae.mula32x16.h3(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mula32x16.h3 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mula32x16_h3_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mula32x16.h3.s2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mula32x16_h3_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mula32x16_h3_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mula32x16.h3.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mula32x16.h3_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mula32x16_l0.ll
declare <1 x i64> @llvm.xtensa.ae.mula32x16.l0(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mula32x16_l0(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mula32x16_l0:



%ret = call <1 x i64> @llvm.xtensa.ae.mula32x16.l0(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mula32x16.l0 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mula32x16_l0_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mula32x16.l0.s2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mula32x16_l0_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mula32x16_l0_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mula32x16.l0.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mula32x16.l0_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mula32x16_l1.ll
declare <1 x i64> @llvm.xtensa.ae.mula32x16.l1(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mula32x16_l1(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mula32x16_l1:



%ret = call <1 x i64> @llvm.xtensa.ae.mula32x16.l1(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mula32x16.l1 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mula32x16_l1_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mula32x16.l1.s2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mula32x16_l1_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mula32x16_l1_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mula32x16.l1.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mula32x16.l1_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mula32x16_l2.ll
declare <1 x i64> @llvm.xtensa.ae.mula32x16.l2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mula32x16_l2(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mula32x16_l2:



%ret = call <1 x i64> @llvm.xtensa.ae.mula32x16.l2(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mula32x16.l2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mula32x16_l2_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mula32x16.l2.s2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mula32x16_l2_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mula32x16_l2_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mula32x16.l2.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mula32x16.l2_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mula32x16_l3.ll
declare <1 x i64> @llvm.xtensa.ae.mula32x16.l3(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mula32x16_l3(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mula32x16_l3:



%ret = call <1 x i64> @llvm.xtensa.ae.mula32x16.l3(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mula32x16.l3 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mula32x16_l3_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mula32x16.l3.s2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mula32x16_l3_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mula32x16_l3_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mula32x16.l3.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mula32x16.l3_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulaad24_hh_ll.ll
declare <1 x i64> @llvm.xtensa.ae.mulaad24.hh.ll(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulaad24_hh_ll(<1 x i64> %ae_mul_q0, <2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaad24_hh_ll:



%ret = call <1 x i64> @llvm.xtensa.ae.mulaad24.hh.ll(<1 x i64> %ae_mul_q0, <2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1)
; CHECK: ae_mulaad24.hh.ll aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulaad24_hh_ll_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulaad24.hh.ll.s2(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulaad24_hh_ll_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaad24_hh_ll_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulaad24.hh.ll.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulaad24.hh.ll_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulaad24_hl_lh.ll
declare <1 x i64> @llvm.xtensa.ae.mulaad24.hl.lh(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulaad24_hl_lh(<1 x i64> %ae_mul_q0, <2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaad24_hl_lh:



%ret = call <1 x i64> @llvm.xtensa.ae.mulaad24.hl.lh(<1 x i64> %ae_mul_q0, <2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1)
; CHECK: ae_mulaad24.hl.lh aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulaad24_hl_lh_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulaad24.hl.lh.s2(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulaad24_hl_lh_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaad24_hl_lh_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulaad24.hl.lh.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulaad24.hl.lh_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulaad32x16_h0_l1.ll
declare <1 x i64> @llvm.xtensa.ae.mulaad32x16.h0.l1(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulaad32x16_h0_l1(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaad32x16_h0_l1:



%ret = call <1 x i64> @llvm.xtensa.ae.mulaad32x16.h0.l1(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulaad32x16.h0.l1 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulaad32x16_h0_l1_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulaad32x16.h0.l1.s2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulaad32x16_h0_l1_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaad32x16_h0_l1_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulaad32x16.h0.l1.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulaad32x16.h0.l1_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulaad32x16_h1_l0.ll
declare <1 x i64> @llvm.xtensa.ae.mulaad32x16.h1.l0(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulaad32x16_h1_l0(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaad32x16_h1_l0:



%ret = call <1 x i64> @llvm.xtensa.ae.mulaad32x16.h1.l0(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulaad32x16.h1.l0 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulaad32x16_h1_l0_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulaad32x16.h1.l0.s2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulaad32x16_h1_l0_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaad32x16_h1_l0_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulaad32x16.h1.l0.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulaad32x16.h1.l0_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulaad32x16_h2_l3.ll
declare <1 x i64> @llvm.xtensa.ae.mulaad32x16.h2.l3(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulaad32x16_h2_l3(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaad32x16_h2_l3:



%ret = call <1 x i64> @llvm.xtensa.ae.mulaad32x16.h2.l3(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulaad32x16.h2.l3 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulaad32x16_h2_l3_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulaad32x16.h2.l3.s2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulaad32x16_h2_l3_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaad32x16_h2_l3_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulaad32x16.h2.l3.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulaad32x16.h2.l3_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulaad32x16_h3_l2.ll
declare <1 x i64> @llvm.xtensa.ae.mulaad32x16.h3.l2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulaad32x16_h3_l2(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaad32x16_h3_l2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulaad32x16.h3.l2(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulaad32x16.h3.l2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulaad32x16_h3_l2_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulaad32x16.h3.l2.s2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulaad32x16_h3_l2_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaad32x16_h3_l2_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulaad32x16.h3.l2.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulaad32x16.h3.l2_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulaafd16ss_11_00.ll
declare <2 x i32> @llvm.xtensa.ae.mulaafd16ss.11.00(<2 x i32>, <4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulaafd16ss_11_00(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaafd16ss_11_00:



%ret = call <2 x i32> @llvm.xtensa.ae.mulaafd16ss.11.00(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulaafd16ss.11_00 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulaafd16ss_11_00_s2.ll
declare <2 x i32> @llvm.xtensa.ae.mulaafd16ss.11.00.s2(<2 x i32>, <4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulaafd16ss_11_00_s2(<2 x i32> %ae_mul_S2_q0, <4 x i16> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaafd16ss_11_00_s2:



%ret = call <2 x i32> @llvm.xtensa.ae.mulaafd16ss.11.00.s2(<2 x i32> %ae_mul_S2_q0, <4 x i16> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulaafd16ss.11_00_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulaafd16ss_13_02.ll
declare <2 x i32> @llvm.xtensa.ae.mulaafd16ss.13.02(<2 x i32>, <4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulaafd16ss_13_02(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaafd16ss_13_02:



%ret = call <2 x i32> @llvm.xtensa.ae.mulaafd16ss.13.02(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulaafd16ss.13_02 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulaafd16ss_13_02_s2.ll
declare <2 x i32> @llvm.xtensa.ae.mulaafd16ss.13.02.s2(<2 x i32>, <4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulaafd16ss_13_02_s2(<2 x i32> %ae_mul_S2_q0, <4 x i16> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaafd16ss_13_02_s2:



%ret = call <2 x i32> @llvm.xtensa.ae.mulaafd16ss.13.02.s2(<2 x i32> %ae_mul_S2_q0, <4 x i16> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulaafd16ss.13_02_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulaafd16ss_33_22.ll
declare <2 x i32> @llvm.xtensa.ae.mulaafd16ss.33.22(<2 x i32>, <4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulaafd16ss_33_22(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaafd16ss_33_22:



%ret = call <2 x i32> @llvm.xtensa.ae.mulaafd16ss.33.22(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulaafd16ss.33_22 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulaafd16ss_33_22_s2.ll
declare <2 x i32> @llvm.xtensa.ae.mulaafd16ss.33.22.s2(<2 x i32>, <4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulaafd16ss_33_22_s2(<2 x i32> %ae_mul_S2_q0, <4 x i16> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaafd16ss_33_22_s2:



%ret = call <2 x i32> @llvm.xtensa.ae.mulaafd16ss.33.22.s2(<2 x i32> %ae_mul_S2_q0, <4 x i16> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulaafd16ss.33_22_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulaafd24_hh_ll.ll
declare <1 x i64> @llvm.xtensa.ae.mulaafd24.hh.ll(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulaafd24_hh_ll(<1 x i64> %ae_mul_q0, <2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaafd24_hh_ll:



%ret = call <1 x i64> @llvm.xtensa.ae.mulaafd24.hh.ll(<1 x i64> %ae_mul_q0, <2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1)
; CHECK: ae_mulaafd24.hh.ll aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulaafd24_hh_ll_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulaafd24.hh.ll.s2(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulaafd24_hh_ll_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaafd24_hh_ll_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulaafd24.hh.ll.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulaafd24.hh.ll_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulaafd24_hl_lh.ll
declare <1 x i64> @llvm.xtensa.ae.mulaafd24.hl.lh(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulaafd24_hl_lh(<1 x i64> %ae_mul_q0, <2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaafd24_hl_lh:



%ret = call <1 x i64> @llvm.xtensa.ae.mulaafd24.hl.lh(<1 x i64> %ae_mul_q0, <2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1)
; CHECK: ae_mulaafd24.hl.lh aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulaafd24_hl_lh_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulaafd24.hl.lh.s2(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulaafd24_hl_lh_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaafd24_hl_lh_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulaafd24.hl.lh.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulaafd24.hl.lh_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulaafd32x16_h0_l1.ll
declare <1 x i64> @llvm.xtensa.ae.mulaafd32x16.h0.l1(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulaafd32x16_h0_l1(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaafd32x16_h0_l1:



%ret = call <1 x i64> @llvm.xtensa.ae.mulaafd32x16.h0.l1(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulaafd32x16.h0.l1 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulaafd32x16_h0_l1_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulaafd32x16.h0.l1.s2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulaafd32x16_h0_l1_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaafd32x16_h0_l1_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulaafd32x16.h0.l1.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulaafd32x16.h0.l1_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulaafd32x16_h1_l0.ll
declare <1 x i64> @llvm.xtensa.ae.mulaafd32x16.h1.l0(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulaafd32x16_h1_l0(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaafd32x16_h1_l0:



%ret = call <1 x i64> @llvm.xtensa.ae.mulaafd32x16.h1.l0(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulaafd32x16.h1.l0 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulaafd32x16_h1_l0_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulaafd32x16.h1.l0.s2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulaafd32x16_h1_l0_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaafd32x16_h1_l0_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulaafd32x16.h1.l0.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulaafd32x16.h1.l0_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulaafd32x16_h2_l3.ll
declare <1 x i64> @llvm.xtensa.ae.mulaafd32x16.h2.l3(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulaafd32x16_h2_l3(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaafd32x16_h2_l3:



%ret = call <1 x i64> @llvm.xtensa.ae.mulaafd32x16.h2.l3(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulaafd32x16.h2.l3 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulaafd32x16_h2_l3_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulaafd32x16.h2.l3.s2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulaafd32x16_h2_l3_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaafd32x16_h2_l3_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulaafd32x16.h2.l3.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulaafd32x16.h2.l3_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulaafd32x16_h3_l2.ll
declare <1 x i64> @llvm.xtensa.ae.mulaafd32x16.h3.l2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulaafd32x16_h3_l2(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaafd32x16_h3_l2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulaafd32x16.h3.l2(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulaafd32x16.h3.l2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulaafd32x16_h3_l2_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulaafd32x16.h3.l2.s2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulaafd32x16_h3_l2_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaafd32x16_h3_l2_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulaafd32x16.h3.l2.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulaafd32x16.h3.l2_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulac24.ll
declare <2 x i32> @llvm.xtensa.ae.mulac24(<2 x i32>, <2 x i32>, <2 x i32>)
define <2 x i32> @test_xtensa_ae_mulac24(<2 x i32> %ae_mul_q0, <2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulac24:



%ret = call <2 x i32> @llvm.xtensa.ae.mulac24(<2 x i32> %ae_mul_q0, <2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1)
; CHECK: ae_mulac24 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulac32x16_h.ll
declare <2 x i32> @llvm.xtensa.ae.mulac32x16.h(<2 x i32>, <2 x i32>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulac32x16_h(<2 x i32> %opnd_ae_sem_mul_x4_q0, <2 x i32> %opnd_ae_sem_mul_x4_d0, <4 x i16> %opnd_ae_sem_mul_x4_d1) {
; CHECK-LABEL: test_xtensa_ae_mulac32x16_h:



%ret = call <2 x i32> @llvm.xtensa.ae.mulac32x16.h(<2 x i32> %opnd_ae_sem_mul_x4_q0, <2 x i32> %opnd_ae_sem_mul_x4_d0, <4 x i16> %opnd_ae_sem_mul_x4_d1)
; CHECK: ae_mulac32x16.h aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulac32x16_l.ll
declare <2 x i32> @llvm.xtensa.ae.mulac32x16.l(<2 x i32>, <2 x i32>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulac32x16_l(<2 x i32> %opnd_ae_sem_mul_x4_q0, <2 x i32> %opnd_ae_sem_mul_x4_d0, <4 x i16> %opnd_ae_sem_mul_x4_d1) {
; CHECK-LABEL: test_xtensa_ae_mulac32x16_l:



%ret = call <2 x i32> @llvm.xtensa.ae.mulac32x16.l(<2 x i32> %opnd_ae_sem_mul_x4_q0, <2 x i32> %opnd_ae_sem_mul_x4_d0, <4 x i16> %opnd_ae_sem_mul_x4_d1)
; CHECK: ae_mulac32x16.l aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulaf16ss_00.ll
declare <2 x i32> @llvm.xtensa.ae.mulaf16ss.00(<2 x i32>, <4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulaf16ss_00(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaf16ss_00:



%ret = call <2 x i32> @llvm.xtensa.ae.mulaf16ss.00(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulaf16ss.00 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulaf16ss_00_s2.ll
declare <2 x i32> @llvm.xtensa.ae.mulaf16ss.00.s2(<2 x i32>, <4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulaf16ss_00_s2(<2 x i32> %ae_mul_S2_q0, <4 x i16> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaf16ss_00_s2:



%ret = call <2 x i32> @llvm.xtensa.ae.mulaf16ss.00.s2(<2 x i32> %ae_mul_S2_q0, <4 x i16> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulaf16ss.00_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulaf16ss_10.ll
declare <2 x i32> @llvm.xtensa.ae.mulaf16ss.10(<2 x i32>, <4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulaf16ss_10(<2 x i32> %ae_mul_q0, <4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaf16ss_10:



%ret = call <2 x i32> @llvm.xtensa.ae.mulaf16ss.10(<2 x i32> %ae_mul_q0, <4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1)
; CHECK: ae_mulaf16ss.10 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulaf16ss_11.ll
declare <2 x i32> @llvm.xtensa.ae.mulaf16ss.11(<2 x i32>, <4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulaf16ss_11(<2 x i32> %ae_mul_q0, <4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaf16ss_11:



%ret = call <2 x i32> @llvm.xtensa.ae.mulaf16ss.11(<2 x i32> %ae_mul_q0, <4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1)
; CHECK: ae_mulaf16ss.11 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulaf16ss_20.ll
declare <2 x i32> @llvm.xtensa.ae.mulaf16ss.20(<2 x i32>, <4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulaf16ss_20(<2 x i32> %ae_mul_q0, <4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaf16ss_20:



%ret = call <2 x i32> @llvm.xtensa.ae.mulaf16ss.20(<2 x i32> %ae_mul_q0, <4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1)
; CHECK: ae_mulaf16ss.20 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulaf16ss_21.ll
declare <2 x i32> @llvm.xtensa.ae.mulaf16ss.21(<2 x i32>, <4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulaf16ss_21(<2 x i32> %ae_mul_q0, <4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaf16ss_21:



%ret = call <2 x i32> @llvm.xtensa.ae.mulaf16ss.21(<2 x i32> %ae_mul_q0, <4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1)
; CHECK: ae_mulaf16ss.21 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulaf16ss_22.ll
declare <2 x i32> @llvm.xtensa.ae.mulaf16ss.22(<2 x i32>, <4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulaf16ss_22(<2 x i32> %ae_mul_q0, <4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaf16ss_22:



%ret = call <2 x i32> @llvm.xtensa.ae.mulaf16ss.22(<2 x i32> %ae_mul_q0, <4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1)
; CHECK: ae_mulaf16ss.22 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulaf16ss_30.ll
declare <2 x i32> @llvm.xtensa.ae.mulaf16ss.30(<2 x i32>, <4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulaf16ss_30(<2 x i32> %ae_mul_q0, <4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaf16ss_30:



%ret = call <2 x i32> @llvm.xtensa.ae.mulaf16ss.30(<2 x i32> %ae_mul_q0, <4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1)
; CHECK: ae_mulaf16ss.30 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulaf16ss_31.ll
declare <2 x i32> @llvm.xtensa.ae.mulaf16ss.31(<2 x i32>, <4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulaf16ss_31(<2 x i32> %ae_mul_q0, <4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaf16ss_31:



%ret = call <2 x i32> @llvm.xtensa.ae.mulaf16ss.31(<2 x i32> %ae_mul_q0, <4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1)
; CHECK: ae_mulaf16ss.31 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulaf16ss_32.ll
declare <2 x i32> @llvm.xtensa.ae.mulaf16ss.32(<2 x i32>, <4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulaf16ss_32(<2 x i32> %ae_mul_q0, <4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaf16ss_32:



%ret = call <2 x i32> @llvm.xtensa.ae.mulaf16ss.32(<2 x i32> %ae_mul_q0, <4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1)
; CHECK: ae_mulaf16ss.32 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulaf16ss_33.ll
declare <2 x i32> @llvm.xtensa.ae.mulaf16ss.33(<2 x i32>, <4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulaf16ss_33(<2 x i32> %ae_mul_q0, <4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaf16ss_33:



%ret = call <2 x i32> @llvm.xtensa.ae.mulaf16ss.33(<2 x i32> %ae_mul_q0, <4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1)
; CHECK: ae_mulaf16ss.33 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulaf16x4ss.ll
declare { <2 x i32>, <2 x i32> } @llvm.xtensa.ae.mulaf16x4ss(<2 x i32>, <2 x i32>, <4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulaf16x4ss(<2 x i32> %ae_mul_q1, <2 x i32> %ae_mul_q0, <4 x i16> %ae_mul_d1, <4 x i16> %ae_mul_d0) {
; CHECK-LABEL: test_xtensa_ae_mulaf16x4ss:




%ret = call { <2 x i32>, <2 x i32> } @llvm.xtensa.ae.mulaf16x4ss(<2 x i32> %ae_mul_q1, <2 x i32> %ae_mul_q0, <4 x i16> %ae_mul_d1, <4 x i16> %ae_mul_d0)
; CHECK: ae_mulaf16x4ss aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}
%ev = extractvalue { <2 x i32>, <2 x i32> } %ret, 0
ret <2 x i32> %ev
}


;--- ae_mulaf32r_hh.ll
declare <1 x i64> @llvm.xtensa.ae.mulaf32r.hh(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulaf32r_hh(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaf32r_hh:



%ret = call <1 x i64> @llvm.xtensa.ae.mulaf32r.hh(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulaf32r.hh aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulaf32r_lh.ll
declare <1 x i64> @llvm.xtensa.ae.mulaf32r.lh(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulaf32r_lh(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaf32r_lh:



%ret = call <1 x i64> @llvm.xtensa.ae.mulaf32r.lh(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulaf32r.lh aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulaf32r_ll.ll
declare <1 x i64> @llvm.xtensa.ae.mulaf32r.ll(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulaf32r_ll(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaf32r_ll:



%ret = call <1 x i64> @llvm.xtensa.ae.mulaf32r.ll(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulaf32r.ll aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulaf32r_ll_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulaf32r.ll.s2(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulaf32r_ll_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaf32r_ll_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulaf32r.ll.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulaf32r.ll_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulaf32s_hh.ll
declare <1 x i64> @llvm.xtensa.ae.mulaf32s.hh(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulaf32s_hh(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaf32s_hh:



%ret = call <1 x i64> @llvm.xtensa.ae.mulaf32s.hh(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulaf32s.hh aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulaf32s_lh.ll
declare <1 x i64> @llvm.xtensa.ae.mulaf32s.lh(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulaf32s_lh(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaf32s_lh:



%ret = call <1 x i64> @llvm.xtensa.ae.mulaf32s.lh(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulaf32s.lh aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulaf32s_ll.ll
declare <1 x i64> @llvm.xtensa.ae.mulaf32s.ll(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulaf32s_ll(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaf32s_ll:



%ret = call <1 x i64> @llvm.xtensa.ae.mulaf32s.ll(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulaf32s.ll aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulaf32s_ll_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulaf32s.ll.s2(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulaf32s_ll_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaf32s_ll_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulaf32s.ll.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulaf32s.ll_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulaf32x16_h0.ll
declare <1 x i64> @llvm.xtensa.ae.mulaf32x16.h0(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulaf32x16_h0(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaf32x16_h0:



%ret = call <1 x i64> @llvm.xtensa.ae.mulaf32x16.h0(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulaf32x16.h0 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulaf32x16_h0_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulaf32x16.h0.s2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulaf32x16_h0_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaf32x16_h0_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulaf32x16.h0.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulaf32x16.h0_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulaf32x16_h1.ll
declare <1 x i64> @llvm.xtensa.ae.mulaf32x16.h1(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulaf32x16_h1(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaf32x16_h1:



%ret = call <1 x i64> @llvm.xtensa.ae.mulaf32x16.h1(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulaf32x16.h1 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulaf32x16_h1_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulaf32x16.h1.s2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulaf32x16_h1_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaf32x16_h1_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulaf32x16.h1.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulaf32x16.h1_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulaf32x16_h2.ll
declare <1 x i64> @llvm.xtensa.ae.mulaf32x16.h2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulaf32x16_h2(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaf32x16_h2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulaf32x16.h2(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulaf32x16.h2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulaf32x16_h2_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulaf32x16.h2.s2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulaf32x16_h2_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaf32x16_h2_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulaf32x16.h2.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulaf32x16.h2_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulaf32x16_h3.ll
declare <1 x i64> @llvm.xtensa.ae.mulaf32x16.h3(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulaf32x16_h3(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaf32x16_h3:



%ret = call <1 x i64> @llvm.xtensa.ae.mulaf32x16.h3(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulaf32x16.h3 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulaf32x16_h3_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulaf32x16.h3.s2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulaf32x16_h3_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaf32x16_h3_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulaf32x16.h3.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulaf32x16.h3_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulaf32x16_l0.ll
declare <1 x i64> @llvm.xtensa.ae.mulaf32x16.l0(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulaf32x16_l0(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaf32x16_l0:



%ret = call <1 x i64> @llvm.xtensa.ae.mulaf32x16.l0(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulaf32x16.l0 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulaf32x16_l0_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulaf32x16.l0.s2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulaf32x16_l0_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaf32x16_l0_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulaf32x16.l0.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulaf32x16.l0_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulaf32x16_l1.ll
declare <1 x i64> @llvm.xtensa.ae.mulaf32x16.l1(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulaf32x16_l1(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaf32x16_l1:



%ret = call <1 x i64> @llvm.xtensa.ae.mulaf32x16.l1(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulaf32x16.l1 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulaf32x16_l1_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulaf32x16.l1.s2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulaf32x16_l1_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaf32x16_l1_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulaf32x16.l1.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulaf32x16.l1_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulaf32x16_l2.ll
declare <1 x i64> @llvm.xtensa.ae.mulaf32x16.l2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulaf32x16_l2(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaf32x16_l2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulaf32x16.l2(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulaf32x16.l2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulaf32x16_l2_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulaf32x16.l2.s2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulaf32x16_l2_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaf32x16_l2_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulaf32x16.l2.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulaf32x16.l2_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulaf32x16_l3.ll
declare <1 x i64> @llvm.xtensa.ae.mulaf32x16.l3(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulaf32x16_l3(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaf32x16_l3:



%ret = call <1 x i64> @llvm.xtensa.ae.mulaf32x16.l3(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulaf32x16.l3 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulaf32x16_l3_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulaf32x16.l3.s2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulaf32x16_l3_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaf32x16_l3_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulaf32x16.l3.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulaf32x16.l3_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulaf48q32sp16s_l.ll
declare <1 x i64> @llvm.xtensa.ae.mulaf48q32sp16s.l(<1 x i64>, <1 x i64>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulaf48q32sp16s_l(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <1 x i64> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaf48q32sp16s_l:



%ret = call <1 x i64> @llvm.xtensa.ae.mulaf48q32sp16s.l(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <1 x i64> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulaf48q32sp16s.l aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulaf48q32sp16s_l_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulaf48q32sp16s.l.s2(<1 x i64>, <1 x i64>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulaf48q32sp16s_l_s2(<1 x i64> %ae_mul_S2_q0, <1 x i64> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaf48q32sp16s_l_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulaf48q32sp16s.l.s2(<1 x i64> %ae_mul_S2_q0, <1 x i64> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulaf48q32sp16s.l_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulaf48q32sp16u_l.ll
declare <1 x i64> @llvm.xtensa.ae.mulaf48q32sp16u.l(<1 x i64>, <1 x i64>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulaf48q32sp16u_l(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <1 x i64> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaf48q32sp16u_l:



%ret = call <1 x i64> @llvm.xtensa.ae.mulaf48q32sp16u.l(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <1 x i64> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulaf48q32sp16u.l aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulaf48q32sp16u_l_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulaf48q32sp16u.l.s2(<1 x i64>, <1 x i64>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulaf48q32sp16u_l_s2(<1 x i64> %ae_mul_S2_q0, <1 x i64> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaf48q32sp16u_l_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulaf48q32sp16u.l.s2(<1 x i64> %ae_mul_S2_q0, <1 x i64> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulaf48q32sp16u.l_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulafc24ra.ll
declare <2 x i32> @llvm.xtensa.ae.mulafc24ra(<2 x i32>, <2 x i32>, <2 x i32>)
define <2 x i32> @test_xtensa_ae_mulafc24ra(<2 x i32> %opnd_ae_sem_mul_x4_q0, <2 x i32> %opnd_ae_sem_mul_x4_d0, <2 x i32> %opnd_ae_sem_mul_x4_d1) {
; CHECK-LABEL: test_xtensa_ae_mulafc24ra:



%ret = call <2 x i32> @llvm.xtensa.ae.mulafc24ra(<2 x i32> %opnd_ae_sem_mul_x4_q0, <2 x i32> %opnd_ae_sem_mul_x4_d0, <2 x i32> %opnd_ae_sem_mul_x4_d1)
; CHECK: ae_mulafc24ra aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulafc32x16ras_h.ll
declare <2 x i32> @llvm.xtensa.ae.mulafc32x16ras.h(<2 x i32>, <2 x i32>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulafc32x16ras_h(<2 x i32> %opnd_ae_sem_mul_x4_q0, <2 x i32> %opnd_ae_sem_mul_x4_d0, <4 x i16> %opnd_ae_sem_mul_x4_d1) {
; CHECK-LABEL: test_xtensa_ae_mulafc32x16ras_h:



%ret = call <2 x i32> @llvm.xtensa.ae.mulafc32x16ras.h(<2 x i32> %opnd_ae_sem_mul_x4_q0, <2 x i32> %opnd_ae_sem_mul_x4_d0, <4 x i16> %opnd_ae_sem_mul_x4_d1)
; CHECK: ae_mulafc32x16ras.h aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulafc32x16ras_l.ll
declare <2 x i32> @llvm.xtensa.ae.mulafc32x16ras.l(<2 x i32>, <2 x i32>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulafc32x16ras_l(<2 x i32> %opnd_ae_sem_mul_x4_q0, <2 x i32> %opnd_ae_sem_mul_x4_d0, <4 x i16> %opnd_ae_sem_mul_x4_d1) {
; CHECK-LABEL: test_xtensa_ae_mulafc32x16ras_l:



%ret = call <2 x i32> @llvm.xtensa.ae.mulafc32x16ras.l(<2 x i32> %opnd_ae_sem_mul_x4_q0, <2 x i32> %opnd_ae_sem_mul_x4_d0, <4 x i16> %opnd_ae_sem_mul_x4_d1)
; CHECK: ae_mulafc32x16ras.l aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulafd24x2_fir_h.ll
declare { <1 x i64>, <1 x i64> } @llvm.xtensa.ae.mulafd24x2.fir.h(<1 x i64>, <1 x i64>, <2 x i32>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulafd24x2_fir_h(<1 x i64> %ae_mul_q0, <1 x i64> %ae_mul_q1, <2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1, <2 x i32> %ae_mul_d2) {
; CHECK-LABEL: test_xtensa_ae_mulafd24x2_fir_h:





%ret = call { <1 x i64>, <1 x i64> } @llvm.xtensa.ae.mulafd24x2.fir.h(<1 x i64> %ae_mul_q0, <1 x i64> %ae_mul_q1, <2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1, <2 x i32> %ae_mul_d2)
; CHECK: ae_mulafd24x2.fir.h aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}
%ev = extractvalue { <1 x i64>, <1 x i64> } %ret, 0
ret <1 x i64> %ev
}


;--- ae_mulafd24x2_fir_l.ll
declare { <1 x i64>, <1 x i64> } @llvm.xtensa.ae.mulafd24x2.fir.l(<1 x i64>, <1 x i64>, <2 x i32>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulafd24x2_fir_l(<1 x i64> %ae_mul_q0, <1 x i64> %ae_mul_q1, <2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1, <2 x i32> %ae_mul_d2) {
; CHECK-LABEL: test_xtensa_ae_mulafd24x2_fir_l:





%ret = call { <1 x i64>, <1 x i64> } @llvm.xtensa.ae.mulafd24x2.fir.l(<1 x i64> %ae_mul_q0, <1 x i64> %ae_mul_q1, <2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1, <2 x i32> %ae_mul_d2)
; CHECK: ae_mulafd24x2.fir.l aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}
%ev = extractvalue { <1 x i64>, <1 x i64> } %ret, 0
ret <1 x i64> %ev
}


;--- ae_mulafd32x16x2_fir_hh.ll
declare { <1 x i64>, <1 x i64> } @llvm.xtensa.ae.mulafd32x16x2.fir.hh(<1 x i64>, <1 x i64>, <2 x i32>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulafd32x16x2_fir_hh(<1 x i64> %ae_mul_q0, <1 x i64> %ae_mul_q1, <2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1, <4 x i16> %ae_mul_d2) {
; CHECK-LABEL: test_xtensa_ae_mulafd32x16x2_fir_hh:





%ret = call { <1 x i64>, <1 x i64> } @llvm.xtensa.ae.mulafd32x16x2.fir.hh(<1 x i64> %ae_mul_q0, <1 x i64> %ae_mul_q1, <2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1, <4 x i16> %ae_mul_d2)
; CHECK: ae_mulafd32x16x2.fir.hh aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}
%ev = extractvalue { <1 x i64>, <1 x i64> } %ret, 0
ret <1 x i64> %ev
}


;--- ae_mulafd32x16x2_fir_hl.ll
declare { <1 x i64>, <1 x i64> } @llvm.xtensa.ae.mulafd32x16x2.fir.hl(<1 x i64>, <1 x i64>, <2 x i32>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulafd32x16x2_fir_hl(<1 x i64> %ae_mul_q0, <1 x i64> %ae_mul_q1, <2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1, <4 x i16> %ae_mul_d2) {
; CHECK-LABEL: test_xtensa_ae_mulafd32x16x2_fir_hl:





%ret = call { <1 x i64>, <1 x i64> } @llvm.xtensa.ae.mulafd32x16x2.fir.hl(<1 x i64> %ae_mul_q0, <1 x i64> %ae_mul_q1, <2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1, <4 x i16> %ae_mul_d2)
; CHECK: ae_mulafd32x16x2.fir.hl aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}
%ev = extractvalue { <1 x i64>, <1 x i64> } %ret, 0
ret <1 x i64> %ev
}


;--- ae_mulafd32x16x2_fir_lh.ll
declare { <1 x i64>, <1 x i64> } @llvm.xtensa.ae.mulafd32x16x2.fir.lh(<1 x i64>, <1 x i64>, <2 x i32>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulafd32x16x2_fir_lh(<1 x i64> %ae_mul_q0, <1 x i64> %ae_mul_q1, <2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1, <4 x i16> %ae_mul_d2) {
; CHECK-LABEL: test_xtensa_ae_mulafd32x16x2_fir_lh:





%ret = call { <1 x i64>, <1 x i64> } @llvm.xtensa.ae.mulafd32x16x2.fir.lh(<1 x i64> %ae_mul_q0, <1 x i64> %ae_mul_q1, <2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1, <4 x i16> %ae_mul_d2)
; CHECK: ae_mulafd32x16x2.fir.lh aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}
%ev = extractvalue { <1 x i64>, <1 x i64> } %ret, 0
ret <1 x i64> %ev
}


;--- ae_mulafd32x16x2_fir_ll.ll
declare { <1 x i64>, <1 x i64> } @llvm.xtensa.ae.mulafd32x16x2.fir.ll(<1 x i64>, <1 x i64>, <2 x i32>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulafd32x16x2_fir_ll(<1 x i64> %ae_mul_q0, <1 x i64> %ae_mul_q1, <2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1, <4 x i16> %ae_mul_d2) {
; CHECK-LABEL: test_xtensa_ae_mulafd32x16x2_fir_ll:





%ret = call { <1 x i64>, <1 x i64> } @llvm.xtensa.ae.mulafd32x16x2.fir.ll(<1 x i64> %ae_mul_q0, <1 x i64> %ae_mul_q1, <2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1, <4 x i16> %ae_mul_d2)
; CHECK: ae_mulafd32x16x2.fir.ll aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}
%ev = extractvalue { <1 x i64>, <1 x i64> } %ret, 0
ret <1 x i64> %ev
}


;--- ae_mulafp24x2r.ll
declare <2 x i32> @llvm.xtensa.ae.mulafp24x2r(<2 x i32>, <2 x i32>, <2 x i32>)
define <2 x i32> @test_xtensa_ae_mulafp24x2r(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulafp24x2r:



%ret = call <2 x i32> @llvm.xtensa.ae.mulafp24x2r(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulafp24x2r aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulafp24x2r_s2.ll
declare <2 x i32> @llvm.xtensa.ae.mulafp24x2r.s2(<2 x i32>, <2 x i32>, <2 x i32>)
define <2 x i32> @test_xtensa_ae_mulafp24x2r_s2(<2 x i32> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulafp24x2r_s2:



%ret = call <2 x i32> @llvm.xtensa.ae.mulafp24x2r.s2(<2 x i32> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulafp24x2r_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulafp24x2ra.ll
declare <2 x i32> @llvm.xtensa.ae.mulafp24x2ra(<2 x i32>, <2 x i32>, <2 x i32>)
define <2 x i32> @test_xtensa_ae_mulafp24x2ra(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulafp24x2ra:



%ret = call <2 x i32> @llvm.xtensa.ae.mulafp24x2ra(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulafp24x2ra aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulafp24x2ra_s2.ll
declare <2 x i32> @llvm.xtensa.ae.mulafp24x2ra.s2(<2 x i32>, <2 x i32>, <2 x i32>)
define <2 x i32> @test_xtensa_ae_mulafp24x2ra_s2(<2 x i32> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulafp24x2ra_s2:



%ret = call <2 x i32> @llvm.xtensa.ae.mulafp24x2ra.s2(<2 x i32> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulafp24x2ra_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulafp32x16x2ras_h.ll
declare <2 x i32> @llvm.xtensa.ae.mulafp32x16x2ras.h(<2 x i32>, <2 x i32>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulafp32x16x2ras_h(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulafp32x16x2ras_h:



%ret = call <2 x i32> @llvm.xtensa.ae.mulafp32x16x2ras.h(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulafp32x16x2ras.h aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulafp32x16x2ras_h_s2.ll
declare <2 x i32> @llvm.xtensa.ae.mulafp32x16x2ras.h.s2(<2 x i32>, <2 x i32>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulafp32x16x2ras_h_s2(<2 x i32> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulafp32x16x2ras_h_s2:



%ret = call <2 x i32> @llvm.xtensa.ae.mulafp32x16x2ras.h.s2(<2 x i32> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulafp32x16x2ras.h_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulafp32x16x2ras_l.ll
declare <2 x i32> @llvm.xtensa.ae.mulafp32x16x2ras.l(<2 x i32>, <2 x i32>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulafp32x16x2ras_l(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulafp32x16x2ras_l:



%ret = call <2 x i32> @llvm.xtensa.ae.mulafp32x16x2ras.l(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulafp32x16x2ras.l aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulafp32x16x2ras_l_s2.ll
declare <2 x i32> @llvm.xtensa.ae.mulafp32x16x2ras.l.s2(<2 x i32>, <2 x i32>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulafp32x16x2ras_l_s2(<2 x i32> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulafp32x16x2ras_l_s2:



%ret = call <2 x i32> @llvm.xtensa.ae.mulafp32x16x2ras.l.s2(<2 x i32> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulafp32x16x2ras.l_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulafp32x16x2rs_h.ll
declare <2 x i32> @llvm.xtensa.ae.mulafp32x16x2rs.h(<2 x i32>, <2 x i32>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulafp32x16x2rs_h(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulafp32x16x2rs_h:



%ret = call <2 x i32> @llvm.xtensa.ae.mulafp32x16x2rs.h(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulafp32x16x2rs.h aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulafp32x16x2rs_h_s2.ll
declare <2 x i32> @llvm.xtensa.ae.mulafp32x16x2rs.h.s2(<2 x i32>, <2 x i32>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulafp32x16x2rs_h_s2(<2 x i32> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulafp32x16x2rs_h_s2:



%ret = call <2 x i32> @llvm.xtensa.ae.mulafp32x16x2rs.h.s2(<2 x i32> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulafp32x16x2rs.h_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulafp32x16x2rs_l.ll
declare <2 x i32> @llvm.xtensa.ae.mulafp32x16x2rs.l(<2 x i32>, <2 x i32>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulafp32x16x2rs_l(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulafp32x16x2rs_l:



%ret = call <2 x i32> @llvm.xtensa.ae.mulafp32x16x2rs.l(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulafp32x16x2rs.l aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulafp32x16x2rs_l_s2.ll
declare <2 x i32> @llvm.xtensa.ae.mulafp32x16x2rs.l.s2(<2 x i32>, <2 x i32>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulafp32x16x2rs_l_s2(<2 x i32> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulafp32x16x2rs_l_s2:



%ret = call <2 x i32> @llvm.xtensa.ae.mulafp32x16x2rs.l.s2(<2 x i32> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulafp32x16x2rs.l_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulafp32x2ras.ll
declare <2 x i32> @llvm.xtensa.ae.mulafp32x2ras(<2 x i32>, <2 x i32>, <2 x i32>)
define <2 x i32> @test_xtensa_ae_mulafp32x2ras(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulafp32x2ras:



%ret = call <2 x i32> @llvm.xtensa.ae.mulafp32x2ras(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulafp32x2ras aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulafp32x2rs.ll
declare <2 x i32> @llvm.xtensa.ae.mulafp32x2rs(<2 x i32>, <2 x i32>, <2 x i32>)
define <2 x i32> @test_xtensa_ae_mulafp32x2rs(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulafp32x2rs:



%ret = call <2 x i32> @llvm.xtensa.ae.mulafp32x2rs(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulafp32x2rs aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulafq32sp24s_h_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulafq32sp24s.h.s2(<1 x i64>, <1 x i64>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulafq32sp24s_h_s2(<1 x i64> %ae_mul_S2_q0, <1 x i64> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulafq32sp24s_h_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulafq32sp24s.h.s2(<1 x i64> %ae_mul_S2_q0, <1 x i64> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulafq32sp24s.h_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulafq32sp24s_l_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulafq32sp24s.l.s2(<1 x i64>, <1 x i64>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulafq32sp24s_l_s2(<1 x i64> %ae_mul_S2_q0, <1 x i64> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulafq32sp24s_l_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulafq32sp24s.l.s2(<1 x i64> %ae_mul_S2_q0, <1 x i64> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulafq32sp24s.l_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulap24x2.ll
declare <2 x i32> @llvm.xtensa.ae.mulap24x2(<2 x i32>, <2 x i32>, <2 x i32>)
define <2 x i32> @test_xtensa_ae_mulap24x2(<2 x i32> %ae_mul_q0, <2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulap24x2:



%ret = call <2 x i32> @llvm.xtensa.ae.mulap24x2(<2 x i32> %ae_mul_q0, <2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1)
; CHECK: ae_mulap24x2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulap24x2_s2.ll
declare <2 x i32> @llvm.xtensa.ae.mulap24x2.s2(<2 x i32>, <2 x i32>, <2 x i32>)
define <2 x i32> @test_xtensa_ae_mulap24x2_s2(<2 x i32> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulap24x2_s2:



%ret = call <2 x i32> @llvm.xtensa.ae.mulap24x2.s2(<2 x i32> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulap24x2_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulap32x16x2_h.ll
declare <2 x i32> @llvm.xtensa.ae.mulap32x16x2.h(<2 x i32>, <2 x i32>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulap32x16x2_h(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulap32x16x2_h:



%ret = call <2 x i32> @llvm.xtensa.ae.mulap32x16x2.h(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulap32x16x2.h aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulap32x16x2_l.ll
declare <2 x i32> @llvm.xtensa.ae.mulap32x16x2.l(<2 x i32>, <2 x i32>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulap32x16x2_l(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulap32x16x2_l:



%ret = call <2 x i32> @llvm.xtensa.ae.mulap32x16x2.l(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulap32x16x2.l aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulap32x2.ll
declare <2 x i32> @llvm.xtensa.ae.mulap32x2(<2 x i32>, <2 x i32>, <2 x i32>)
define <2 x i32> @test_xtensa_ae_mulap32x2(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulap32x2:



%ret = call <2 x i32> @llvm.xtensa.ae.mulap32x2(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulap32x2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulaq32sp16s_l_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulaq32sp16s.l.s2(<1 x i64>, <1 x i64>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulaq32sp16s_l_s2(<1 x i64> %ae_mul_S2_q0, <1 x i64> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaq32sp16s_l_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulaq32sp16s.l.s2(<1 x i64> %ae_mul_S2_q0, <1 x i64> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulaq32sp16s.l_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulaq32sp16u_l_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulaq32sp16u.l.s2(<1 x i64>, <1 x i64>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulaq32sp16u_l_s2(<1 x i64> %ae_mul_S2_q0, <1 x i64> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulaq32sp16u_l_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulaq32sp16u.l.s2(<1 x i64> %ae_mul_S2_q0, <1 x i64> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulaq32sp16u.l_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mularfq32sp24s_h_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mularfq32sp24s.h.s2(<1 x i64>, <1 x i64>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mularfq32sp24s_h_s2(<1 x i64> %ae_mul_S2_q0, <1 x i64> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mularfq32sp24s_h_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mularfq32sp24s.h.s2(<1 x i64> %ae_mul_S2_q0, <1 x i64> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mularfq32sp24s.h_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mularfq32sp24s_l_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mularfq32sp24s.l.s2(<1 x i64>, <1 x i64>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mularfq32sp24s_l_s2(<1 x i64> %ae_mul_S2_q0, <1 x i64> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mularfq32sp24s_l_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mularfq32sp24s.l.s2(<1 x i64> %ae_mul_S2_q0, <1 x i64> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mularfq32sp24s.l_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulas32f48p16s_hh.ll
declare <1 x i64> @llvm.xtensa.ae.mulas32f48p16s.hh(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulas32f48p16s_hh(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulas32f48p16s_hh:



%ret = call <1 x i64> @llvm.xtensa.ae.mulas32f48p16s.hh(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulas32f48p16s.hh aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulas32f48p16s_hh_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulas32f48p16s.hh.s2(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulas32f48p16s_hh_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulas32f48p16s_hh_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulas32f48p16s.hh.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulas32f48p16s.hh_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulas32f48p16s_lh.ll
declare <1 x i64> @llvm.xtensa.ae.mulas32f48p16s.lh(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulas32f48p16s_lh(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulas32f48p16s_lh:



%ret = call <1 x i64> @llvm.xtensa.ae.mulas32f48p16s.lh(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulas32f48p16s.lh aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulas32f48p16s_lh_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulas32f48p16s.lh.s2(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulas32f48p16s_lh_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulas32f48p16s_lh_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulas32f48p16s.lh.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulas32f48p16s.lh_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulas32f48p16s_ll.ll
declare <1 x i64> @llvm.xtensa.ae.mulas32f48p16s.ll(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulas32f48p16s_ll(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulas32f48p16s_ll:



%ret = call <1 x i64> @llvm.xtensa.ae.mulas32f48p16s.ll(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulas32f48p16s.ll aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulas32f48p16s_ll_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulas32f48p16s.ll.s2(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulas32f48p16s_ll_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulas32f48p16s_ll_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulas32f48p16s.ll.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulas32f48p16s.ll_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulasd24_hh_ll.ll
declare <1 x i64> @llvm.xtensa.ae.mulasd24.hh.ll(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulasd24_hh_ll(<1 x i64> %ae_mul_q0, <2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulasd24_hh_ll:



%ret = call <1 x i64> @llvm.xtensa.ae.mulasd24.hh.ll(<1 x i64> %ae_mul_q0, <2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1)
; CHECK: ae_mulasd24.hh.ll aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulasd24_hh_ll_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulasd24.hh.ll.s2(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulasd24_hh_ll_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulasd24_hh_ll_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulasd24.hh.ll.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulasd24.hh.ll_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulasd24_hl_lh.ll
declare <1 x i64> @llvm.xtensa.ae.mulasd24.hl.lh(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulasd24_hl_lh(<1 x i64> %ae_mul_q0, <2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulasd24_hl_lh:



%ret = call <1 x i64> @llvm.xtensa.ae.mulasd24.hl.lh(<1 x i64> %ae_mul_q0, <2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1)
; CHECK: ae_mulasd24.hl.lh aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulasd24_hl_lh_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulasd24.hl.lh.s2(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulasd24_hl_lh_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulasd24_hl_lh_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulasd24.hl.lh.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulasd24.hl.lh_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulasd32x16_h1_l0.ll
declare <1 x i64> @llvm.xtensa.ae.mulasd32x16.h1.l0(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulasd32x16_h1_l0(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulasd32x16_h1_l0:



%ret = call <1 x i64> @llvm.xtensa.ae.mulasd32x16.h1.l0(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulasd32x16.h1.l0 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulasd32x16_h1_l0_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulasd32x16.h1.l0.s2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulasd32x16_h1_l0_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulasd32x16_h1_l0_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulasd32x16.h1.l0.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulasd32x16.h1.l0_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulasd32x16_h3_l2.ll
declare <1 x i64> @llvm.xtensa.ae.mulasd32x16.h3.l2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulasd32x16_h3_l2(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulasd32x16_h3_l2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulasd32x16.h3.l2(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulasd32x16.h3.l2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulasd32x16_h3_l2_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulasd32x16.h3.l2.s2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulasd32x16_h3_l2_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulasd32x16_h3_l2_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulasd32x16.h3.l2.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulasd32x16.h3.l2_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulasfd24_hh_ll.ll
declare <1 x i64> @llvm.xtensa.ae.mulasfd24.hh.ll(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulasfd24_hh_ll(<1 x i64> %ae_mul_q0, <2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulasfd24_hh_ll:



%ret = call <1 x i64> @llvm.xtensa.ae.mulasfd24.hh.ll(<1 x i64> %ae_mul_q0, <2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1)
; CHECK: ae_mulasfd24.hh.ll aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulasfd24_hh_ll_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulasfd24.hh.ll.s2(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulasfd24_hh_ll_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulasfd24_hh_ll_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulasfd24.hh.ll.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulasfd24.hh.ll_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulasfd24_hl_lh.ll
declare <1 x i64> @llvm.xtensa.ae.mulasfd24.hl.lh(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulasfd24_hl_lh(<1 x i64> %ae_mul_q0, <2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulasfd24_hl_lh:



%ret = call <1 x i64> @llvm.xtensa.ae.mulasfd24.hl.lh(<1 x i64> %ae_mul_q0, <2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1)
; CHECK: ae_mulasfd24.hl.lh aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulasfd24_hl_lh_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulasfd24.hl.lh.s2(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulasfd24_hl_lh_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulasfd24_hl_lh_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulasfd24.hl.lh.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulasfd24.hl.lh_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulasfd32x16_h1_l0.ll
declare <1 x i64> @llvm.xtensa.ae.mulasfd32x16.h1.l0(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulasfd32x16_h1_l0(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulasfd32x16_h1_l0:



%ret = call <1 x i64> @llvm.xtensa.ae.mulasfd32x16.h1.l0(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulasfd32x16.h1.l0 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulasfd32x16_h1_l0_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulasfd32x16.h1.l0.s2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulasfd32x16_h1_l0_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulasfd32x16_h1_l0_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulasfd32x16.h1.l0.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulasfd32x16.h1.l0_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulasfd32x16_h3_l2.ll
declare <1 x i64> @llvm.xtensa.ae.mulasfd32x16.h3.l2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulasfd32x16_h3_l2(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulasfd32x16_h3_l2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulasfd32x16.h3.l2(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulasfd32x16.h3.l2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulasfd32x16_h3_l2_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulasfd32x16.h3.l2.s2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulasfd32x16_h3_l2_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulasfd32x16_h3_l2_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulasfd32x16.h3.l2.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulasfd32x16.h3.l2_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulc24.ll
declare <2 x i32> @llvm.xtensa.ae.mulc24(<2 x i32>, <2 x i32>)
define <2 x i32> @test_xtensa_ae_mulc24(<2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulc24:


%ret = call <2 x i32> @llvm.xtensa.ae.mulc24(<2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1)
; CHECK: ae_mulc24 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulc32x16_h.ll
declare <2 x i32> @llvm.xtensa.ae.mulc32x16.h(<2 x i32>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulc32x16_h(<2 x i32> %opnd_ae_sem_mul_x4_d0, <4 x i16> %opnd_ae_sem_mul_x4_d1) {
; CHECK-LABEL: test_xtensa_ae_mulc32x16_h:


%ret = call <2 x i32> @llvm.xtensa.ae.mulc32x16.h(<2 x i32> %opnd_ae_sem_mul_x4_d0, <4 x i16> %opnd_ae_sem_mul_x4_d1)
; CHECK: ae_mulc32x16.h aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulc32x16_l.ll
declare <2 x i32> @llvm.xtensa.ae.mulc32x16.l(<2 x i32>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulc32x16_l(<2 x i32> %opnd_ae_sem_mul_x4_d0, <4 x i16> %opnd_ae_sem_mul_x4_d1) {
; CHECK-LABEL: test_xtensa_ae_mulc32x16_l:


%ret = call <2 x i32> @llvm.xtensa.ae.mulc32x16.l(<2 x i32> %opnd_ae_sem_mul_x4_d0, <4 x i16> %opnd_ae_sem_mul_x4_d1)
; CHECK: ae_mulc32x16.l aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulf16ss_00.ll
declare <2 x i32> @llvm.xtensa.ae.mulf16ss.00(<4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulf16ss_00(<4 x i16> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulf16ss_00:


%ret = call <2 x i32> @llvm.xtensa.ae.mulf16ss.00(<4 x i16> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulf16ss.00 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulf16ss_00_s2.ll
declare <2 x i32> @llvm.xtensa.ae.mulf16ss.00.s2(<4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulf16ss_00_s2(<4 x i16> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulf16ss_00_s2:


%ret = call <2 x i32> @llvm.xtensa.ae.mulf16ss.00.s2(<4 x i16> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulf16ss.00_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulf16ss_10.ll
declare <2 x i32> @llvm.xtensa.ae.mulf16ss.10(<4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulf16ss_10(<4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulf16ss_10:


%ret = call <2 x i32> @llvm.xtensa.ae.mulf16ss.10(<4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1)
; CHECK: ae_mulf16ss.10 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulf16ss_11.ll
declare <2 x i32> @llvm.xtensa.ae.mulf16ss.11(<4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulf16ss_11(<4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulf16ss_11:


%ret = call <2 x i32> @llvm.xtensa.ae.mulf16ss.11(<4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1)
; CHECK: ae_mulf16ss.11 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulf16ss_20.ll
declare <2 x i32> @llvm.xtensa.ae.mulf16ss.20(<4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulf16ss_20(<4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulf16ss_20:


%ret = call <2 x i32> @llvm.xtensa.ae.mulf16ss.20(<4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1)
; CHECK: ae_mulf16ss.20 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulf16ss_21.ll
declare <2 x i32> @llvm.xtensa.ae.mulf16ss.21(<4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulf16ss_21(<4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulf16ss_21:


%ret = call <2 x i32> @llvm.xtensa.ae.mulf16ss.21(<4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1)
; CHECK: ae_mulf16ss.21 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulf16ss_22.ll
declare <2 x i32> @llvm.xtensa.ae.mulf16ss.22(<4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulf16ss_22(<4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulf16ss_22:


%ret = call <2 x i32> @llvm.xtensa.ae.mulf16ss.22(<4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1)
; CHECK: ae_mulf16ss.22 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulf16ss_30.ll
declare <2 x i32> @llvm.xtensa.ae.mulf16ss.30(<4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulf16ss_30(<4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulf16ss_30:


%ret = call <2 x i32> @llvm.xtensa.ae.mulf16ss.30(<4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1)
; CHECK: ae_mulf16ss.30 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulf16ss_31.ll
declare <2 x i32> @llvm.xtensa.ae.mulf16ss.31(<4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulf16ss_31(<4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulf16ss_31:


%ret = call <2 x i32> @llvm.xtensa.ae.mulf16ss.31(<4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1)
; CHECK: ae_mulf16ss.31 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulf16ss_32.ll
declare <2 x i32> @llvm.xtensa.ae.mulf16ss.32(<4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulf16ss_32(<4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulf16ss_32:


%ret = call <2 x i32> @llvm.xtensa.ae.mulf16ss.32(<4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1)
; CHECK: ae_mulf16ss.32 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulf16ss_33.ll
declare <2 x i32> @llvm.xtensa.ae.mulf16ss.33(<4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulf16ss_33(<4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulf16ss_33:


%ret = call <2 x i32> @llvm.xtensa.ae.mulf16ss.33(<4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1)
; CHECK: ae_mulf16ss.33 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulf16x4ss.ll
declare { <2 x i32>, <2 x i32> } @llvm.xtensa.ae.mulf16x4ss(<4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulf16x4ss(<4 x i16> %ae_mul_d1, <4 x i16> %ae_mul_d0) {
; CHECK-LABEL: test_xtensa_ae_mulf16x4ss:


%ret = call { <2 x i32>, <2 x i32> } @llvm.xtensa.ae.mulf16x4ss(<4 x i16> %ae_mul_d1, <4 x i16> %ae_mul_d0)
; CHECK: ae_mulf16x4ss aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}
%ev = extractvalue { <2 x i32>, <2 x i32> } %ret, 0
ret <2 x i32> %ev
}


;--- ae_mulf32r_hh.ll
declare <1 x i64> @llvm.xtensa.ae.mulf32r.hh(<2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulf32r_hh(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulf32r_hh:


%ret = call <1 x i64> @llvm.xtensa.ae.mulf32r.hh(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulf32r.hh aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulf32r_lh.ll
declare <1 x i64> @llvm.xtensa.ae.mulf32r.lh(<2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulf32r_lh(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulf32r_lh:


%ret = call <1 x i64> @llvm.xtensa.ae.mulf32r.lh(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulf32r.lh aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulf32r_ll.ll
declare <1 x i64> @llvm.xtensa.ae.mulf32r.ll(<2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulf32r_ll(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulf32r_ll:


%ret = call <1 x i64> @llvm.xtensa.ae.mulf32r.ll(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulf32r.ll aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulf32r_ll_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulf32r.ll.s2(<2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulf32r_ll_s2(<2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulf32r_ll_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulf32r.ll.s2(<2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulf32r.ll_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulf32s_hh.ll
declare <1 x i64> @llvm.xtensa.ae.mulf32s.hh(<2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulf32s_hh(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulf32s_hh:


%ret = call <1 x i64> @llvm.xtensa.ae.mulf32s.hh(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulf32s.hh aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulf32s_lh.ll
declare <1 x i64> @llvm.xtensa.ae.mulf32s.lh(<2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulf32s_lh(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulf32s_lh:


%ret = call <1 x i64> @llvm.xtensa.ae.mulf32s.lh(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulf32s.lh aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulf32s_ll.ll
declare <1 x i64> @llvm.xtensa.ae.mulf32s.ll(<2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulf32s_ll(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulf32s_ll:


%ret = call <1 x i64> @llvm.xtensa.ae.mulf32s.ll(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulf32s.ll aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulf32s_ll_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulf32s.ll.s2(<2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulf32s_ll_s2(<2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulf32s_ll_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulf32s.ll.s2(<2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulf32s.ll_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulf32x16_h0.ll
declare <1 x i64> @llvm.xtensa.ae.mulf32x16.h0(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulf32x16_h0(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulf32x16_h0:


%ret = call <1 x i64> @llvm.xtensa.ae.mulf32x16.h0(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulf32x16.h0 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulf32x16_h0_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulf32x16.h0.s2(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulf32x16_h0_s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulf32x16_h0_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulf32x16.h0.s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulf32x16.h0_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulf32x16_h1.ll
declare <1 x i64> @llvm.xtensa.ae.mulf32x16.h1(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulf32x16_h1(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulf32x16_h1:


%ret = call <1 x i64> @llvm.xtensa.ae.mulf32x16.h1(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulf32x16.h1 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulf32x16_h1_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulf32x16.h1.s2(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulf32x16_h1_s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulf32x16_h1_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulf32x16.h1.s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulf32x16.h1_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulf32x16_h2.ll
declare <1 x i64> @llvm.xtensa.ae.mulf32x16.h2(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulf32x16_h2(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulf32x16_h2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulf32x16.h2(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulf32x16.h2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulf32x16_h2_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulf32x16.h2.s2(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulf32x16_h2_s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulf32x16_h2_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulf32x16.h2.s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulf32x16.h2_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulf32x16_h3.ll
declare <1 x i64> @llvm.xtensa.ae.mulf32x16.h3(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulf32x16_h3(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulf32x16_h3:


%ret = call <1 x i64> @llvm.xtensa.ae.mulf32x16.h3(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulf32x16.h3 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulf32x16_h3_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulf32x16.h3.s2(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulf32x16_h3_s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulf32x16_h3_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulf32x16.h3.s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulf32x16.h3_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulf32x16_l0.ll
declare <1 x i64> @llvm.xtensa.ae.mulf32x16.l0(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulf32x16_l0(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulf32x16_l0:


%ret = call <1 x i64> @llvm.xtensa.ae.mulf32x16.l0(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulf32x16.l0 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulf32x16_l0_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulf32x16.l0.s2(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulf32x16_l0_s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulf32x16_l0_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulf32x16.l0.s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulf32x16.l0_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulf32x16_l1.ll
declare <1 x i64> @llvm.xtensa.ae.mulf32x16.l1(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulf32x16_l1(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulf32x16_l1:


%ret = call <1 x i64> @llvm.xtensa.ae.mulf32x16.l1(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulf32x16.l1 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulf32x16_l1_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulf32x16.l1.s2(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulf32x16_l1_s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulf32x16_l1_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulf32x16.l1.s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulf32x16.l1_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulf32x16_l2.ll
declare <1 x i64> @llvm.xtensa.ae.mulf32x16.l2(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulf32x16_l2(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulf32x16_l2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulf32x16.l2(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulf32x16.l2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulf32x16_l2_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulf32x16.l2.s2(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulf32x16_l2_s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulf32x16_l2_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulf32x16.l2.s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulf32x16.l2_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulf32x16_l3.ll
declare <1 x i64> @llvm.xtensa.ae.mulf32x16.l3(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulf32x16_l3(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulf32x16_l3:


%ret = call <1 x i64> @llvm.xtensa.ae.mulf32x16.l3(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulf32x16.l3 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulf32x16_l3_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulf32x16.l3.s2(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulf32x16_l3_s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulf32x16_l3_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulf32x16.l3.s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulf32x16.l3_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulf48q32sp16s_l.ll
declare <1 x i64> @llvm.xtensa.ae.mulf48q32sp16s.l(<1 x i64>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulf48q32sp16s_l(<1 x i64> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulf48q32sp16s_l:


%ret = call <1 x i64> @llvm.xtensa.ae.mulf48q32sp16s.l(<1 x i64> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulf48q32sp16s.l aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulf48q32sp16s_l_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulf48q32sp16s.l.s2(<1 x i64>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulf48q32sp16s_l_s2(<1 x i64> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulf48q32sp16s_l_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulf48q32sp16s.l.s2(<1 x i64> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulf48q32sp16s.l_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulf48q32sp16u_l.ll
declare <1 x i64> @llvm.xtensa.ae.mulf48q32sp16u.l(<1 x i64>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulf48q32sp16u_l(<1 x i64> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulf48q32sp16u_l:


%ret = call <1 x i64> @llvm.xtensa.ae.mulf48q32sp16u.l(<1 x i64> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulf48q32sp16u.l aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulf48q32sp16u_l_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulf48q32sp16u.l.s2(<1 x i64>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulf48q32sp16u_l_s2(<1 x i64> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulf48q32sp16u_l_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulf48q32sp16u.l.s2(<1 x i64> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulf48q32sp16u.l_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulfc24ra.ll
declare <2 x i32> @llvm.xtensa.ae.mulfc24ra(<2 x i32>, <2 x i32>)
define <2 x i32> @test_xtensa_ae_mulfc24ra(<2 x i32> %opnd_ae_sem_mul_x4_d0, <2 x i32> %opnd_ae_sem_mul_x4_d1) {
; CHECK-LABEL: test_xtensa_ae_mulfc24ra:


%ret = call <2 x i32> @llvm.xtensa.ae.mulfc24ra(<2 x i32> %opnd_ae_sem_mul_x4_d0, <2 x i32> %opnd_ae_sem_mul_x4_d1)
; CHECK: ae_mulfc24ra aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulfc32x16ras_h.ll
declare <2 x i32> @llvm.xtensa.ae.mulfc32x16ras.h(<2 x i32>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulfc32x16ras_h(<2 x i32> %opnd_ae_sem_mul_x4_d0, <4 x i16> %opnd_ae_sem_mul_x4_d1) {
; CHECK-LABEL: test_xtensa_ae_mulfc32x16ras_h:


%ret = call <2 x i32> @llvm.xtensa.ae.mulfc32x16ras.h(<2 x i32> %opnd_ae_sem_mul_x4_d0, <4 x i16> %opnd_ae_sem_mul_x4_d1)
; CHECK: ae_mulfc32x16ras.h aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulfc32x16ras_l.ll
declare <2 x i32> @llvm.xtensa.ae.mulfc32x16ras.l(<2 x i32>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulfc32x16ras_l(<2 x i32> %opnd_ae_sem_mul_x4_d0, <4 x i16> %opnd_ae_sem_mul_x4_d1) {
; CHECK-LABEL: test_xtensa_ae_mulfc32x16ras_l:


%ret = call <2 x i32> @llvm.xtensa.ae.mulfc32x16ras.l(<2 x i32> %opnd_ae_sem_mul_x4_d0, <4 x i16> %opnd_ae_sem_mul_x4_d1)
; CHECK: ae_mulfc32x16ras.l aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulfd24x2_fir_h.ll
declare { <1 x i64>, <1 x i64> } @llvm.xtensa.ae.mulfd24x2.fir.h(<2 x i32>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulfd24x2_fir_h(<2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1, <2 x i32> %ae_mul_d2) {
; CHECK-LABEL: test_xtensa_ae_mulfd24x2_fir_h:



%ret = call { <1 x i64>, <1 x i64> } @llvm.xtensa.ae.mulfd24x2.fir.h(<2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1, <2 x i32> %ae_mul_d2)
; CHECK: ae_mulfd24x2.fir.h aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}
%ev = extractvalue { <1 x i64>, <1 x i64> } %ret, 0
ret <1 x i64> %ev
}


;--- ae_mulfd24x2_fir_l.ll
declare { <1 x i64>, <1 x i64> } @llvm.xtensa.ae.mulfd24x2.fir.l(<2 x i32>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulfd24x2_fir_l(<2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1, <2 x i32> %ae_mul_d2) {
; CHECK-LABEL: test_xtensa_ae_mulfd24x2_fir_l:



%ret = call { <1 x i64>, <1 x i64> } @llvm.xtensa.ae.mulfd24x2.fir.l(<2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1, <2 x i32> %ae_mul_d2)
; CHECK: ae_mulfd24x2.fir.l aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}
%ev = extractvalue { <1 x i64>, <1 x i64> } %ret, 0
ret <1 x i64> %ev
}


;--- ae_mulfd32x16x2_fir_hh.ll
declare { <1 x i64>, <1 x i64> } @llvm.xtensa.ae.mulfd32x16x2.fir.hh(<2 x i32>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulfd32x16x2_fir_hh(<2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1, <4 x i16> %ae_mul_d2) {
; CHECK-LABEL: test_xtensa_ae_mulfd32x16x2_fir_hh:



%ret = call { <1 x i64>, <1 x i64> } @llvm.xtensa.ae.mulfd32x16x2.fir.hh(<2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1, <4 x i16> %ae_mul_d2)
; CHECK: ae_mulfd32x16x2.fir.hh aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}
%ev = extractvalue { <1 x i64>, <1 x i64> } %ret, 0
ret <1 x i64> %ev
}


;--- ae_mulfd32x16x2_fir_hl.ll
declare { <1 x i64>, <1 x i64> } @llvm.xtensa.ae.mulfd32x16x2.fir.hl(<2 x i32>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulfd32x16x2_fir_hl(<2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1, <4 x i16> %ae_mul_d2) {
; CHECK-LABEL: test_xtensa_ae_mulfd32x16x2_fir_hl:



%ret = call { <1 x i64>, <1 x i64> } @llvm.xtensa.ae.mulfd32x16x2.fir.hl(<2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1, <4 x i16> %ae_mul_d2)
; CHECK: ae_mulfd32x16x2.fir.hl aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}
%ev = extractvalue { <1 x i64>, <1 x i64> } %ret, 0
ret <1 x i64> %ev
}


;--- ae_mulfd32x16x2_fir_lh.ll
declare { <1 x i64>, <1 x i64> } @llvm.xtensa.ae.mulfd32x16x2.fir.lh(<2 x i32>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulfd32x16x2_fir_lh(<2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1, <4 x i16> %ae_mul_d2) {
; CHECK-LABEL: test_xtensa_ae_mulfd32x16x2_fir_lh:



%ret = call { <1 x i64>, <1 x i64> } @llvm.xtensa.ae.mulfd32x16x2.fir.lh(<2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1, <4 x i16> %ae_mul_d2)
; CHECK: ae_mulfd32x16x2.fir.lh aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}
%ev = extractvalue { <1 x i64>, <1 x i64> } %ret, 0
ret <1 x i64> %ev
}


;--- ae_mulfd32x16x2_fir_ll.ll
declare { <1 x i64>, <1 x i64> } @llvm.xtensa.ae.mulfd32x16x2.fir.ll(<2 x i32>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulfd32x16x2_fir_ll(<2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1, <4 x i16> %ae_mul_d2) {
; CHECK-LABEL: test_xtensa_ae_mulfd32x16x2_fir_ll:



%ret = call { <1 x i64>, <1 x i64> } @llvm.xtensa.ae.mulfd32x16x2.fir.ll(<2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1, <4 x i16> %ae_mul_d2)
; CHECK: ae_mulfd32x16x2.fir.ll aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}
%ev = extractvalue { <1 x i64>, <1 x i64> } %ret, 0
ret <1 x i64> %ev
}


;--- ae_mulfp16x4ras.ll
declare <4 x i16> @llvm.xtensa.ae.mulfp16x4ras(<4 x i16>, <4 x i16>)
define <4 x i16> @test_xtensa_ae_mulfp16x4ras(<4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulfp16x4ras:


%ret = call <4 x i16> @llvm.xtensa.ae.mulfp16x4ras(<4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1)
; CHECK: ae_mulfp16x4ras aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <4 x i16> %ret
}


;--- ae_mulfp16x4s.ll
declare <4 x i16> @llvm.xtensa.ae.mulfp16x4s(<4 x i16>, <4 x i16>)
define <4 x i16> @test_xtensa_ae_mulfp16x4s(<4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulfp16x4s:


%ret = call <4 x i16> @llvm.xtensa.ae.mulfp16x4s(<4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1)
; CHECK: ae_mulfp16x4s aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <4 x i16> %ret
}


;--- ae_mulfp24x2r.ll
declare <2 x i32> @llvm.xtensa.ae.mulfp24x2r(<2 x i32>, <2 x i32>)
define <2 x i32> @test_xtensa_ae_mulfp24x2r(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulfp24x2r:


%ret = call <2 x i32> @llvm.xtensa.ae.mulfp24x2r(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulfp24x2r aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulfp24x2r_s2.ll
declare <2 x i32> @llvm.xtensa.ae.mulfp24x2r.s2(<2 x i32>, <2 x i32>)
define <2 x i32> @test_xtensa_ae_mulfp24x2r_s2(<2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulfp24x2r_s2:


%ret = call <2 x i32> @llvm.xtensa.ae.mulfp24x2r.s2(<2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulfp24x2r_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulfp24x2ra.ll
declare <2 x i32> @llvm.xtensa.ae.mulfp24x2ra(<2 x i32>, <2 x i32>)
define <2 x i32> @test_xtensa_ae_mulfp24x2ra(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulfp24x2ra:


%ret = call <2 x i32> @llvm.xtensa.ae.mulfp24x2ra(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulfp24x2ra aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulfp24x2ra_s2.ll
declare <2 x i32> @llvm.xtensa.ae.mulfp24x2ra.s2(<2 x i32>, <2 x i32>)
define <2 x i32> @test_xtensa_ae_mulfp24x2ra_s2(<2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulfp24x2ra_s2:


%ret = call <2 x i32> @llvm.xtensa.ae.mulfp24x2ra.s2(<2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulfp24x2ra_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulfp32x16x2ras_h.ll
declare <2 x i32> @llvm.xtensa.ae.mulfp32x16x2ras.h(<2 x i32>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulfp32x16x2ras_h(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulfp32x16x2ras_h:


%ret = call <2 x i32> @llvm.xtensa.ae.mulfp32x16x2ras.h(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulfp32x16x2ras.h aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulfp32x16x2ras_h_s2.ll
declare <2 x i32> @llvm.xtensa.ae.mulfp32x16x2ras.h.s2(<2 x i32>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulfp32x16x2ras_h_s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulfp32x16x2ras_h_s2:


%ret = call <2 x i32> @llvm.xtensa.ae.mulfp32x16x2ras.h.s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulfp32x16x2ras.h_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulfp32x16x2ras_l.ll
declare <2 x i32> @llvm.xtensa.ae.mulfp32x16x2ras.l(<2 x i32>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulfp32x16x2ras_l(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulfp32x16x2ras_l:


%ret = call <2 x i32> @llvm.xtensa.ae.mulfp32x16x2ras.l(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulfp32x16x2ras.l aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulfp32x16x2ras_l_s2.ll
declare <2 x i32> @llvm.xtensa.ae.mulfp32x16x2ras.l.s2(<2 x i32>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulfp32x16x2ras_l_s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulfp32x16x2ras_l_s2:


%ret = call <2 x i32> @llvm.xtensa.ae.mulfp32x16x2ras.l.s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulfp32x16x2ras.l_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulfp32x16x2rs_h.ll
declare <2 x i32> @llvm.xtensa.ae.mulfp32x16x2rs.h(<2 x i32>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulfp32x16x2rs_h(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulfp32x16x2rs_h:


%ret = call <2 x i32> @llvm.xtensa.ae.mulfp32x16x2rs.h(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulfp32x16x2rs.h aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulfp32x16x2rs_h_s2.ll
declare <2 x i32> @llvm.xtensa.ae.mulfp32x16x2rs.h.s2(<2 x i32>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulfp32x16x2rs_h_s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulfp32x16x2rs_h_s2:


%ret = call <2 x i32> @llvm.xtensa.ae.mulfp32x16x2rs.h.s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulfp32x16x2rs.h_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulfp32x16x2rs_l.ll
declare <2 x i32> @llvm.xtensa.ae.mulfp32x16x2rs.l(<2 x i32>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulfp32x16x2rs_l(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulfp32x16x2rs_l:


%ret = call <2 x i32> @llvm.xtensa.ae.mulfp32x16x2rs.l(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulfp32x16x2rs.l aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulfp32x16x2rs_l_s2.ll
declare <2 x i32> @llvm.xtensa.ae.mulfp32x16x2rs.l.s2(<2 x i32>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulfp32x16x2rs_l_s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulfp32x16x2rs_l_s2:


%ret = call <2 x i32> @llvm.xtensa.ae.mulfp32x16x2rs.l.s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulfp32x16x2rs.l_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulfp32x2ras.ll
declare <2 x i32> @llvm.xtensa.ae.mulfp32x2ras(<2 x i32>, <2 x i32>)
define <2 x i32> @test_xtensa_ae_mulfp32x2ras(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulfp32x2ras:


%ret = call <2 x i32> @llvm.xtensa.ae.mulfp32x2ras(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulfp32x2ras aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulfp32x2rs.ll
declare <2 x i32> @llvm.xtensa.ae.mulfp32x2rs(<2 x i32>, <2 x i32>)
define <2 x i32> @test_xtensa_ae_mulfp32x2rs(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulfp32x2rs:


%ret = call <2 x i32> @llvm.xtensa.ae.mulfp32x2rs(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulfp32x2rs aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulfq32sp24s_h_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulfq32sp24s.h.s2(<1 x i64>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulfq32sp24s_h_s2(<1 x i64> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulfq32sp24s_h_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulfq32sp24s.h.s2(<1 x i64> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulfq32sp24s.h_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulfq32sp24s_l_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulfq32sp24s.l.s2(<1 x i64>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulfq32sp24s_l_s2(<1 x i64> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulfq32sp24s_l_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulfq32sp24s.l.s2(<1 x i64> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulfq32sp24s.l_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulp24x2.ll
declare <2 x i32> @llvm.xtensa.ae.mulp24x2(<2 x i32>, <2 x i32>)
define <2 x i32> @test_xtensa_ae_mulp24x2(<2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulp24x2:


%ret = call <2 x i32> @llvm.xtensa.ae.mulp24x2(<2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1)
; CHECK: ae_mulp24x2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulp24x2_s2.ll
declare <2 x i32> @llvm.xtensa.ae.mulp24x2.s2(<2 x i32>, <2 x i32>)
define <2 x i32> @test_xtensa_ae_mulp24x2_s2(<2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulp24x2_s2:


%ret = call <2 x i32> @llvm.xtensa.ae.mulp24x2.s2(<2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulp24x2_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulp32x16x2_h.ll
declare <2 x i32> @llvm.xtensa.ae.mulp32x16x2.h(<2 x i32>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulp32x16x2_h(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulp32x16x2_h:


%ret = call <2 x i32> @llvm.xtensa.ae.mulp32x16x2.h(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulp32x16x2.h aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulp32x16x2_l.ll
declare <2 x i32> @llvm.xtensa.ae.mulp32x16x2.l(<2 x i32>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulp32x16x2_l(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulp32x16x2_l:


%ret = call <2 x i32> @llvm.xtensa.ae.mulp32x16x2.l(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulp32x16x2.l aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulp32x2.ll
declare <2 x i32> @llvm.xtensa.ae.mulp32x2(<2 x i32>, <2 x i32>)
define <2 x i32> @test_xtensa_ae_mulp32x2(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulp32x2:


%ret = call <2 x i32> @llvm.xtensa.ae.mulp32x2(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulp32x2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulq32sp16s_l_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulq32sp16s.l.s2(<1 x i64>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulq32sp16s_l_s2(<1 x i64> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulq32sp16s_l_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulq32sp16s.l.s2(<1 x i64> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulq32sp16s.l_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulq32sp16u_l_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulq32sp16u.l.s2(<1 x i64>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulq32sp16u_l_s2(<1 x i64> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulq32sp16u_l_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulq32sp16u.l.s2(<1 x i64> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulq32sp16u.l_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulrfq32sp24s_h_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulrfq32sp24s.h.s2(<1 x i64>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulrfq32sp24s_h_s2(<1 x i64> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulrfq32sp24s_h_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulrfq32sp24s.h.s2(<1 x i64> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulrfq32sp24s.h_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulrfq32sp24s_l_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulrfq32sp24s.l.s2(<1 x i64>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulrfq32sp24s_l_s2(<1 x i64> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulrfq32sp24s_l_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulrfq32sp24s.l.s2(<1 x i64> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulrfq32sp24s.l_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_muls16x4.ll
declare { <2 x i32>, <2 x i32> } @llvm.xtensa.ae.muls16x4(<2 x i32>, <2 x i32>, <4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_muls16x4(<2 x i32> %ae_mul_q1, <2 x i32> %ae_mul_q0, <4 x i16> %ae_mul_d1, <4 x i16> %ae_mul_d0) {
; CHECK-LABEL: test_xtensa_ae_muls16x4:




%ret = call { <2 x i32>, <2 x i32> } @llvm.xtensa.ae.muls16x4(<2 x i32> %ae_mul_q1, <2 x i32> %ae_mul_q0, <4 x i16> %ae_mul_d1, <4 x i16> %ae_mul_d0)
; CHECK: ae_muls16x4 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}
%ev = extractvalue { <2 x i32>, <2 x i32> } %ret, 0
ret <2 x i32> %ev
}


;--- ae_muls32_hh.ll
declare <1 x i64> @llvm.xtensa.ae.muls32.hh(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_muls32_hh(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_muls32_hh:



%ret = call <1 x i64> @llvm.xtensa.ae.muls32.hh(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_muls32.hh aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_muls32_lh.ll
declare <1 x i64> @llvm.xtensa.ae.muls32.lh(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_muls32_lh(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_muls32_lh:



%ret = call <1 x i64> @llvm.xtensa.ae.muls32.lh(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_muls32.lh aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_muls32_ll.ll
declare <1 x i64> @llvm.xtensa.ae.muls32.ll(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_muls32_ll(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_muls32_ll:



%ret = call <1 x i64> @llvm.xtensa.ae.muls32.ll(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_muls32.ll aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_muls32f48p16s_hh.ll
declare <1 x i64> @llvm.xtensa.ae.muls32f48p16s.hh(<2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_muls32f48p16s_hh(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_muls32f48p16s_hh:


%ret = call <1 x i64> @llvm.xtensa.ae.muls32f48p16s.hh(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_muls32f48p16s.hh aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_muls32f48p16s_hh_s2.ll
declare <1 x i64> @llvm.xtensa.ae.muls32f48p16s.hh.s2(<2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_muls32f48p16s_hh_s2(<2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_muls32f48p16s_hh_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.muls32f48p16s.hh.s2(<2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_muls32f48p16s.hh_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_muls32f48p16s_lh.ll
declare <1 x i64> @llvm.xtensa.ae.muls32f48p16s.lh(<2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_muls32f48p16s_lh(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_muls32f48p16s_lh:


%ret = call <1 x i64> @llvm.xtensa.ae.muls32f48p16s.lh(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_muls32f48p16s.lh aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_muls32f48p16s_lh_s2.ll
declare <1 x i64> @llvm.xtensa.ae.muls32f48p16s.lh.s2(<2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_muls32f48p16s_lh_s2(<2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_muls32f48p16s_lh_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.muls32f48p16s.lh.s2(<2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_muls32f48p16s.lh_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_muls32f48p16s_ll.ll
declare <1 x i64> @llvm.xtensa.ae.muls32f48p16s.ll(<2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_muls32f48p16s_ll(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_muls32f48p16s_ll:


%ret = call <1 x i64> @llvm.xtensa.ae.muls32f48p16s.ll(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_muls32f48p16s.ll aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_muls32f48p16s_ll_s2.ll
declare <1 x i64> @llvm.xtensa.ae.muls32f48p16s.ll.s2(<2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_muls32f48p16s_ll_s2(<2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_muls32f48p16s_ll_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.muls32f48p16s.ll.s2(<2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_muls32f48p16s.ll_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_muls32u_ll.ll
declare <1 x i64> @llvm.xtensa.ae.muls32u.ll(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_muls32u_ll(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_muls32u_ll:



%ret = call <1 x i64> @llvm.xtensa.ae.muls32u.ll(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_muls32u.ll aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_muls32x16_h0.ll
declare <1 x i64> @llvm.xtensa.ae.muls32x16.h0(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_muls32x16_h0(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_muls32x16_h0:



%ret = call <1 x i64> @llvm.xtensa.ae.muls32x16.h0(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_muls32x16.h0 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_muls32x16_h0_s2.ll
declare <1 x i64> @llvm.xtensa.ae.muls32x16.h0.s2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_muls32x16_h0_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_muls32x16_h0_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.muls32x16.h0.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_muls32x16.h0_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_muls32x16_h1.ll
declare <1 x i64> @llvm.xtensa.ae.muls32x16.h1(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_muls32x16_h1(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_muls32x16_h1:



%ret = call <1 x i64> @llvm.xtensa.ae.muls32x16.h1(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_muls32x16.h1 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_muls32x16_h1_s2.ll
declare <1 x i64> @llvm.xtensa.ae.muls32x16.h1.s2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_muls32x16_h1_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_muls32x16_h1_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.muls32x16.h1.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_muls32x16.h1_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_muls32x16_h2.ll
declare <1 x i64> @llvm.xtensa.ae.muls32x16.h2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_muls32x16_h2(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_muls32x16_h2:



%ret = call <1 x i64> @llvm.xtensa.ae.muls32x16.h2(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_muls32x16.h2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_muls32x16_h2_s2.ll
declare <1 x i64> @llvm.xtensa.ae.muls32x16.h2.s2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_muls32x16_h2_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_muls32x16_h2_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.muls32x16.h2.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_muls32x16.h2_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_muls32x16_h3.ll
declare <1 x i64> @llvm.xtensa.ae.muls32x16.h3(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_muls32x16_h3(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_muls32x16_h3:



%ret = call <1 x i64> @llvm.xtensa.ae.muls32x16.h3(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_muls32x16.h3 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_muls32x16_h3_s2.ll
declare <1 x i64> @llvm.xtensa.ae.muls32x16.h3.s2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_muls32x16_h3_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_muls32x16_h3_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.muls32x16.h3.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_muls32x16.h3_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_muls32x16_l0.ll
declare <1 x i64> @llvm.xtensa.ae.muls32x16.l0(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_muls32x16_l0(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_muls32x16_l0:



%ret = call <1 x i64> @llvm.xtensa.ae.muls32x16.l0(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_muls32x16.l0 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_muls32x16_l0_s2.ll
declare <1 x i64> @llvm.xtensa.ae.muls32x16.l0.s2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_muls32x16_l0_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_muls32x16_l0_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.muls32x16.l0.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_muls32x16.l0_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_muls32x16_l1.ll
declare <1 x i64> @llvm.xtensa.ae.muls32x16.l1(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_muls32x16_l1(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_muls32x16_l1:



%ret = call <1 x i64> @llvm.xtensa.ae.muls32x16.l1(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_muls32x16.l1 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_muls32x16_l1_s2.ll
declare <1 x i64> @llvm.xtensa.ae.muls32x16.l1.s2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_muls32x16_l1_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_muls32x16_l1_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.muls32x16.l1.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_muls32x16.l1_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_muls32x16_l2.ll
declare <1 x i64> @llvm.xtensa.ae.muls32x16.l2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_muls32x16_l2(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_muls32x16_l2:



%ret = call <1 x i64> @llvm.xtensa.ae.muls32x16.l2(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_muls32x16.l2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_muls32x16_l2_s2.ll
declare <1 x i64> @llvm.xtensa.ae.muls32x16.l2.s2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_muls32x16_l2_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_muls32x16_l2_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.muls32x16.l2.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_muls32x16.l2_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_muls32x16_l3.ll
declare <1 x i64> @llvm.xtensa.ae.muls32x16.l3(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_muls32x16_l3(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_muls32x16_l3:



%ret = call <1 x i64> @llvm.xtensa.ae.muls32x16.l3(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_muls32x16.l3 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_muls32x16_l3_s2.ll
declare <1 x i64> @llvm.xtensa.ae.muls32x16.l3.s2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_muls32x16_l3_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_muls32x16_l3_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.muls32x16.l3.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_muls32x16.l3_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulsad24_hh_ll.ll
declare <1 x i64> @llvm.xtensa.ae.mulsad24.hh.ll(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulsad24_hh_ll(<1 x i64> %ae_mul_q0, <2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsad24_hh_ll:



%ret = call <1 x i64> @llvm.xtensa.ae.mulsad24.hh.ll(<1 x i64> %ae_mul_q0, <2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1)
; CHECK: ae_mulsad24.hh.ll aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulsad24_hh_ll_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulsad24.hh.ll.s2(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulsad24_hh_ll_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsad24_hh_ll_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulsad24.hh.ll.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulsad24.hh.ll_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulsad32x16_h1_l0.ll
declare <1 x i64> @llvm.xtensa.ae.mulsad32x16.h1.l0(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulsad32x16_h1_l0(<1 x i64> %ae_mul_q0, <2 x i32> %ae_mul_d0, <4 x i16> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsad32x16_h1_l0:



%ret = call <1 x i64> @llvm.xtensa.ae.mulsad32x16.h1.l0(<1 x i64> %ae_mul_q0, <2 x i32> %ae_mul_d0, <4 x i16> %ae_mul_d1)
; CHECK: ae_mulsad32x16.h1.l0 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulsad32x16_h1_l0_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulsad32x16.h1.l0.s2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulsad32x16_h1_l0_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsad32x16_h1_l0_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulsad32x16.h1.l0.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulsad32x16.h1.l0_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulsad32x16_h3_l2.ll
declare <1 x i64> @llvm.xtensa.ae.mulsad32x16.h3.l2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulsad32x16_h3_l2(<1 x i64> %ae_mul_q0, <2 x i32> %ae_mul_d0, <4 x i16> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsad32x16_h3_l2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulsad32x16.h3.l2(<1 x i64> %ae_mul_q0, <2 x i32> %ae_mul_d0, <4 x i16> %ae_mul_d1)
; CHECK: ae_mulsad32x16.h3.l2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulsad32x16_h3_l2_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulsad32x16.h3.l2.s2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulsad32x16_h3_l2_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsad32x16_h3_l2_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulsad32x16.h3.l2.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulsad32x16.h3.l2_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulsafd24_hh_ll.ll
declare <1 x i64> @llvm.xtensa.ae.mulsafd24.hh.ll(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulsafd24_hh_ll(<1 x i64> %ae_mul_q0, <2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsafd24_hh_ll:



%ret = call <1 x i64> @llvm.xtensa.ae.mulsafd24.hh.ll(<1 x i64> %ae_mul_q0, <2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1)
; CHECK: ae_mulsafd24.hh.ll aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulsafd24_hh_ll_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulsafd24.hh.ll.s2(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulsafd24_hh_ll_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsafd24_hh_ll_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulsafd24.hh.ll.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulsafd24.hh.ll_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulsafd32x16_h1_l0.ll
declare <1 x i64> @llvm.xtensa.ae.mulsafd32x16.h1.l0(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulsafd32x16_h1_l0(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsafd32x16_h1_l0:



%ret = call <1 x i64> @llvm.xtensa.ae.mulsafd32x16.h1.l0(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulsafd32x16.h1.l0 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulsafd32x16_h1_l0_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulsafd32x16.h1.l0.s2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulsafd32x16_h1_l0_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsafd32x16_h1_l0_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulsafd32x16.h1.l0.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulsafd32x16.h1.l0_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulsafd32x16_h3_l2.ll
declare <1 x i64> @llvm.xtensa.ae.mulsafd32x16.h3.l2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulsafd32x16_h3_l2(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsafd32x16_h3_l2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulsafd32x16.h3.l2(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulsafd32x16.h3.l2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulsafd32x16_h3_l2_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulsafd32x16.h3.l2.s2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulsafd32x16_h3_l2_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsafd32x16_h3_l2_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulsafd32x16.h3.l2.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulsafd32x16.h3.l2_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulsf16ss_00.ll
declare <2 x i32> @llvm.xtensa.ae.mulsf16ss.00(<2 x i32>, <4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulsf16ss_00(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsf16ss_00:



%ret = call <2 x i32> @llvm.xtensa.ae.mulsf16ss.00(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulsf16ss.00 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulsf16ss_00_s2.ll
declare <2 x i32> @llvm.xtensa.ae.mulsf16ss.00.s2(<2 x i32>, <4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulsf16ss_00_s2(<2 x i32> %ae_mul_S2_q0, <4 x i16> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsf16ss_00_s2:



%ret = call <2 x i32> @llvm.xtensa.ae.mulsf16ss.00.s2(<2 x i32> %ae_mul_S2_q0, <4 x i16> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulsf16ss.00_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulsf16ss_10.ll
declare <2 x i32> @llvm.xtensa.ae.mulsf16ss.10(<2 x i32>, <4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulsf16ss_10(<2 x i32> %ae_mul_q0, <4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsf16ss_10:



%ret = call <2 x i32> @llvm.xtensa.ae.mulsf16ss.10(<2 x i32> %ae_mul_q0, <4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1)
; CHECK: ae_mulsf16ss.10 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulsf16ss_11.ll
declare <2 x i32> @llvm.xtensa.ae.mulsf16ss.11(<2 x i32>, <4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulsf16ss_11(<2 x i32> %ae_mul_q0, <4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsf16ss_11:



%ret = call <2 x i32> @llvm.xtensa.ae.mulsf16ss.11(<2 x i32> %ae_mul_q0, <4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1)
; CHECK: ae_mulsf16ss.11 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulsf16ss_20.ll
declare <2 x i32> @llvm.xtensa.ae.mulsf16ss.20(<2 x i32>, <4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulsf16ss_20(<2 x i32> %ae_mul_q0, <4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsf16ss_20:



%ret = call <2 x i32> @llvm.xtensa.ae.mulsf16ss.20(<2 x i32> %ae_mul_q0, <4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1)
; CHECK: ae_mulsf16ss.20 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulsf16ss_21.ll
declare <2 x i32> @llvm.xtensa.ae.mulsf16ss.21(<2 x i32>, <4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulsf16ss_21(<2 x i32> %ae_mul_q0, <4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsf16ss_21:



%ret = call <2 x i32> @llvm.xtensa.ae.mulsf16ss.21(<2 x i32> %ae_mul_q0, <4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1)
; CHECK: ae_mulsf16ss.21 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulsf16ss_22.ll
declare <2 x i32> @llvm.xtensa.ae.mulsf16ss.22(<2 x i32>, <4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulsf16ss_22(<2 x i32> %ae_mul_q0, <4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsf16ss_22:



%ret = call <2 x i32> @llvm.xtensa.ae.mulsf16ss.22(<2 x i32> %ae_mul_q0, <4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1)
; CHECK: ae_mulsf16ss.22 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulsf16ss_30.ll
declare <2 x i32> @llvm.xtensa.ae.mulsf16ss.30(<2 x i32>, <4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulsf16ss_30(<2 x i32> %ae_mul_q0, <4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsf16ss_30:



%ret = call <2 x i32> @llvm.xtensa.ae.mulsf16ss.30(<2 x i32> %ae_mul_q0, <4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1)
; CHECK: ae_mulsf16ss.30 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulsf16ss_31.ll
declare <2 x i32> @llvm.xtensa.ae.mulsf16ss.31(<2 x i32>, <4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulsf16ss_31(<2 x i32> %ae_mul_q0, <4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsf16ss_31:



%ret = call <2 x i32> @llvm.xtensa.ae.mulsf16ss.31(<2 x i32> %ae_mul_q0, <4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1)
; CHECK: ae_mulsf16ss.31 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulsf16ss_32.ll
declare <2 x i32> @llvm.xtensa.ae.mulsf16ss.32(<2 x i32>, <4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulsf16ss_32(<2 x i32> %ae_mul_q0, <4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsf16ss_32:



%ret = call <2 x i32> @llvm.xtensa.ae.mulsf16ss.32(<2 x i32> %ae_mul_q0, <4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1)
; CHECK: ae_mulsf16ss.32 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulsf16ss_33.ll
declare <2 x i32> @llvm.xtensa.ae.mulsf16ss.33(<2 x i32>, <4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulsf16ss_33(<2 x i32> %ae_mul_q0, <4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsf16ss_33:



%ret = call <2 x i32> @llvm.xtensa.ae.mulsf16ss.33(<2 x i32> %ae_mul_q0, <4 x i16> %ae_mul_d0, <4 x i16> %ae_mul_d1)
; CHECK: ae_mulsf16ss.33 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulsf16x4ss.ll
declare { <2 x i32>, <2 x i32> } @llvm.xtensa.ae.mulsf16x4ss(<2 x i32>, <2 x i32>, <4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulsf16x4ss(<2 x i32> %ae_mul_q1, <2 x i32> %ae_mul_q0, <4 x i16> %ae_mul_d1, <4 x i16> %ae_mul_d0) {
; CHECK-LABEL: test_xtensa_ae_mulsf16x4ss:




%ret = call { <2 x i32>, <2 x i32> } @llvm.xtensa.ae.mulsf16x4ss(<2 x i32> %ae_mul_q1, <2 x i32> %ae_mul_q0, <4 x i16> %ae_mul_d1, <4 x i16> %ae_mul_d0)
; CHECK: ae_mulsf16x4ss aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}
%ev = extractvalue { <2 x i32>, <2 x i32> } %ret, 0
ret <2 x i32> %ev
}


;--- ae_mulsf32r_hh.ll
declare <1 x i64> @llvm.xtensa.ae.mulsf32r.hh(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulsf32r_hh(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsf32r_hh:



%ret = call <1 x i64> @llvm.xtensa.ae.mulsf32r.hh(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulsf32r.hh aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulsf32r_lh.ll
declare <1 x i64> @llvm.xtensa.ae.mulsf32r.lh(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulsf32r_lh(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsf32r_lh:



%ret = call <1 x i64> @llvm.xtensa.ae.mulsf32r.lh(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulsf32r.lh aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulsf32r_ll.ll
declare <1 x i64> @llvm.xtensa.ae.mulsf32r.ll(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulsf32r_ll(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsf32r_ll:



%ret = call <1 x i64> @llvm.xtensa.ae.mulsf32r.ll(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulsf32r.ll aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulsf32r_ll_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulsf32r.ll.s2(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulsf32r_ll_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsf32r_ll_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulsf32r.ll.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulsf32r.ll_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulsf32s_hh.ll
declare <1 x i64> @llvm.xtensa.ae.mulsf32s.hh(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulsf32s_hh(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsf32s_hh:



%ret = call <1 x i64> @llvm.xtensa.ae.mulsf32s.hh(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulsf32s.hh aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulsf32s_lh.ll
declare <1 x i64> @llvm.xtensa.ae.mulsf32s.lh(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulsf32s_lh(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsf32s_lh:



%ret = call <1 x i64> @llvm.xtensa.ae.mulsf32s.lh(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulsf32s.lh aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulsf32s_ll.ll
declare <1 x i64> @llvm.xtensa.ae.mulsf32s.ll(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulsf32s_ll(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsf32s_ll:



%ret = call <1 x i64> @llvm.xtensa.ae.mulsf32s.ll(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulsf32s.ll aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulsf32x16_h0.ll
declare <1 x i64> @llvm.xtensa.ae.mulsf32x16.h0(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulsf32x16_h0(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsf32x16_h0:



%ret = call <1 x i64> @llvm.xtensa.ae.mulsf32x16.h0(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulsf32x16.h0 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulsf32x16_h0_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulsf32x16.h0.s2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulsf32x16_h0_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsf32x16_h0_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulsf32x16.h0.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulsf32x16.h0_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulsf32x16_h1.ll
declare <1 x i64> @llvm.xtensa.ae.mulsf32x16.h1(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulsf32x16_h1(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsf32x16_h1:



%ret = call <1 x i64> @llvm.xtensa.ae.mulsf32x16.h1(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulsf32x16.h1 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulsf32x16_h1_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulsf32x16.h1.s2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulsf32x16_h1_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsf32x16_h1_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulsf32x16.h1.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulsf32x16.h1_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulsf32x16_h2.ll
declare <1 x i64> @llvm.xtensa.ae.mulsf32x16.h2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulsf32x16_h2(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsf32x16_h2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulsf32x16.h2(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulsf32x16.h2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulsf32x16_h2_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulsf32x16.h2.s2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulsf32x16_h2_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsf32x16_h2_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulsf32x16.h2.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulsf32x16.h2_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulsf32x16_h3.ll
declare <1 x i64> @llvm.xtensa.ae.mulsf32x16.h3(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulsf32x16_h3(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsf32x16_h3:



%ret = call <1 x i64> @llvm.xtensa.ae.mulsf32x16.h3(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulsf32x16.h3 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulsf32x16_h3_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulsf32x16.h3.s2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulsf32x16_h3_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsf32x16_h3_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulsf32x16.h3.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulsf32x16.h3_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulsf32x16_l0.ll
declare <1 x i64> @llvm.xtensa.ae.mulsf32x16.l0(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulsf32x16_l0(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsf32x16_l0:



%ret = call <1 x i64> @llvm.xtensa.ae.mulsf32x16.l0(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulsf32x16.l0 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulsf32x16_l0_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulsf32x16.l0.s2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulsf32x16_l0_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsf32x16_l0_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulsf32x16.l0.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulsf32x16.l0_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulsf32x16_l1.ll
declare <1 x i64> @llvm.xtensa.ae.mulsf32x16.l1(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulsf32x16_l1(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsf32x16_l1:



%ret = call <1 x i64> @llvm.xtensa.ae.mulsf32x16.l1(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulsf32x16.l1 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulsf32x16_l1_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulsf32x16.l1.s2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulsf32x16_l1_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsf32x16_l1_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulsf32x16.l1.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulsf32x16.l1_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulsf32x16_l2.ll
declare <1 x i64> @llvm.xtensa.ae.mulsf32x16.l2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulsf32x16_l2(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsf32x16_l2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulsf32x16.l2(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulsf32x16.l2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulsf32x16_l2_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulsf32x16.l2.s2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulsf32x16_l2_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsf32x16_l2_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulsf32x16.l2.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulsf32x16.l2_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulsf32x16_l3.ll
declare <1 x i64> @llvm.xtensa.ae.mulsf32x16.l3(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulsf32x16_l3(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsf32x16_l3:



%ret = call <1 x i64> @llvm.xtensa.ae.mulsf32x16.l3(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulsf32x16.l3 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulsf32x16_l3_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulsf32x16.l3.s2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulsf32x16_l3_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsf32x16_l3_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulsf32x16.l3.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulsf32x16.l3_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulsf48q32sp16s_l.ll
declare <1 x i64> @llvm.xtensa.ae.mulsf48q32sp16s.l(<1 x i64>, <1 x i64>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulsf48q32sp16s_l(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <1 x i64> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsf48q32sp16s_l:



%ret = call <1 x i64> @llvm.xtensa.ae.mulsf48q32sp16s.l(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <1 x i64> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulsf48q32sp16s.l aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulsf48q32sp16s_l_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulsf48q32sp16s.l.s2(<1 x i64>, <1 x i64>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulsf48q32sp16s_l_s2(<1 x i64> %ae_mul_S2_q0, <1 x i64> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsf48q32sp16s_l_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulsf48q32sp16s.l.s2(<1 x i64> %ae_mul_S2_q0, <1 x i64> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulsf48q32sp16s.l_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulsf48q32sp16u_l.ll
declare <1 x i64> @llvm.xtensa.ae.mulsf48q32sp16u.l(<1 x i64>, <1 x i64>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulsf48q32sp16u_l(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <1 x i64> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsf48q32sp16u_l:



%ret = call <1 x i64> @llvm.xtensa.ae.mulsf48q32sp16u.l(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <1 x i64> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulsf48q32sp16u.l aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulsf48q32sp16u_l_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulsf48q32sp16u.l.s2(<1 x i64>, <1 x i64>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulsf48q32sp16u_l_s2(<1 x i64> %ae_mul_S2_q0, <1 x i64> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsf48q32sp16u_l_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulsf48q32sp16u.l.s2(<1 x i64> %ae_mul_S2_q0, <1 x i64> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulsf48q32sp16u.l_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulsfp24x2r.ll
declare <2 x i32> @llvm.xtensa.ae.mulsfp24x2r(<2 x i32>, <2 x i32>, <2 x i32>)
define <2 x i32> @test_xtensa_ae_mulsfp24x2r(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsfp24x2r:



%ret = call <2 x i32> @llvm.xtensa.ae.mulsfp24x2r(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulsfp24x2r aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulsfp24x2r_s2.ll
declare <2 x i32> @llvm.xtensa.ae.mulsfp24x2r.s2(<2 x i32>, <2 x i32>, <2 x i32>)
define <2 x i32> @test_xtensa_ae_mulsfp24x2r_s2(<2 x i32> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsfp24x2r_s2:



%ret = call <2 x i32> @llvm.xtensa.ae.mulsfp24x2r.s2(<2 x i32> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulsfp24x2r_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulsfp24x2ra.ll
declare <2 x i32> @llvm.xtensa.ae.mulsfp24x2ra(<2 x i32>, <2 x i32>, <2 x i32>)
define <2 x i32> @test_xtensa_ae_mulsfp24x2ra(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsfp24x2ra:



%ret = call <2 x i32> @llvm.xtensa.ae.mulsfp24x2ra(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulsfp24x2ra aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulsfp24x2ra_s2.ll
declare <2 x i32> @llvm.xtensa.ae.mulsfp24x2ra.s2(<2 x i32>, <2 x i32>, <2 x i32>)
define <2 x i32> @test_xtensa_ae_mulsfp24x2ra_s2(<2 x i32> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsfp24x2ra_s2:



%ret = call <2 x i32> @llvm.xtensa.ae.mulsfp24x2ra.s2(<2 x i32> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulsfp24x2ra_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulsfp32x16x2ras_h.ll
declare <2 x i32> @llvm.xtensa.ae.mulsfp32x16x2ras.h(<2 x i32>, <2 x i32>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulsfp32x16x2ras_h(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsfp32x16x2ras_h:



%ret = call <2 x i32> @llvm.xtensa.ae.mulsfp32x16x2ras.h(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulsfp32x16x2ras.h aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulsfp32x16x2ras_h_s2.ll
declare <2 x i32> @llvm.xtensa.ae.mulsfp32x16x2ras.h.s2(<2 x i32>, <2 x i32>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulsfp32x16x2ras_h_s2(<2 x i32> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsfp32x16x2ras_h_s2:



%ret = call <2 x i32> @llvm.xtensa.ae.mulsfp32x16x2ras.h.s2(<2 x i32> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulsfp32x16x2ras.h_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulsfp32x16x2ras_l.ll
declare <2 x i32> @llvm.xtensa.ae.mulsfp32x16x2ras.l(<2 x i32>, <2 x i32>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulsfp32x16x2ras_l(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsfp32x16x2ras_l:



%ret = call <2 x i32> @llvm.xtensa.ae.mulsfp32x16x2ras.l(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulsfp32x16x2ras.l aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulsfp32x16x2ras_l_s2.ll
declare <2 x i32> @llvm.xtensa.ae.mulsfp32x16x2ras.l.s2(<2 x i32>, <2 x i32>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulsfp32x16x2ras_l_s2(<2 x i32> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsfp32x16x2ras_l_s2:



%ret = call <2 x i32> @llvm.xtensa.ae.mulsfp32x16x2ras.l.s2(<2 x i32> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulsfp32x16x2ras.l_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulsfp32x16x2rs_h.ll
declare <2 x i32> @llvm.xtensa.ae.mulsfp32x16x2rs.h(<2 x i32>, <2 x i32>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulsfp32x16x2rs_h(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsfp32x16x2rs_h:



%ret = call <2 x i32> @llvm.xtensa.ae.mulsfp32x16x2rs.h(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulsfp32x16x2rs.h aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulsfp32x16x2rs_h_s2.ll
declare <2 x i32> @llvm.xtensa.ae.mulsfp32x16x2rs.h.s2(<2 x i32>, <2 x i32>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulsfp32x16x2rs_h_s2(<2 x i32> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsfp32x16x2rs_h_s2:



%ret = call <2 x i32> @llvm.xtensa.ae.mulsfp32x16x2rs.h.s2(<2 x i32> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulsfp32x16x2rs.h_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulsfp32x16x2rs_l.ll
declare <2 x i32> @llvm.xtensa.ae.mulsfp32x16x2rs.l(<2 x i32>, <2 x i32>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulsfp32x16x2rs_l(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsfp32x16x2rs_l:



%ret = call <2 x i32> @llvm.xtensa.ae.mulsfp32x16x2rs.l(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulsfp32x16x2rs.l aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulsfp32x16x2rs_l_s2.ll
declare <2 x i32> @llvm.xtensa.ae.mulsfp32x16x2rs.l.s2(<2 x i32>, <2 x i32>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulsfp32x16x2rs_l_s2(<2 x i32> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsfp32x16x2rs_l_s2:



%ret = call <2 x i32> @llvm.xtensa.ae.mulsfp32x16x2rs.l.s2(<2 x i32> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulsfp32x16x2rs.l_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulsfp32x2ras.ll
declare <2 x i32> @llvm.xtensa.ae.mulsfp32x2ras(<2 x i32>, <2 x i32>, <2 x i32>)
define <2 x i32> @test_xtensa_ae_mulsfp32x2ras(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsfp32x2ras:



%ret = call <2 x i32> @llvm.xtensa.ae.mulsfp32x2ras(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulsfp32x2ras aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulsfp32x2rs.ll
declare <2 x i32> @llvm.xtensa.ae.mulsfp32x2rs(<2 x i32>, <2 x i32>, <2 x i32>)
define <2 x i32> @test_xtensa_ae_mulsfp32x2rs(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsfp32x2rs:



%ret = call <2 x i32> @llvm.xtensa.ae.mulsfp32x2rs(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulsfp32x2rs aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulsfq32sp24s_h_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulsfq32sp24s.h.s2(<1 x i64>, <1 x i64>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulsfq32sp24s_h_s2(<1 x i64> %ae_mul_S2_q0, <1 x i64> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsfq32sp24s_h_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulsfq32sp24s.h.s2(<1 x i64> %ae_mul_S2_q0, <1 x i64> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulsfq32sp24s.h_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulsfq32sp24s_l_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulsfq32sp24s.l.s2(<1 x i64>, <1 x i64>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulsfq32sp24s_l_s2(<1 x i64> %ae_mul_S2_q0, <1 x i64> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsfq32sp24s_l_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulsfq32sp24s.l.s2(<1 x i64> %ae_mul_S2_q0, <1 x i64> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulsfq32sp24s.l_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulsp24x2.ll
declare <2 x i32> @llvm.xtensa.ae.mulsp24x2(<2 x i32>, <2 x i32>, <2 x i32>)
define <2 x i32> @test_xtensa_ae_mulsp24x2(<2 x i32> %ae_mul_q0, <2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsp24x2:



%ret = call <2 x i32> @llvm.xtensa.ae.mulsp24x2(<2 x i32> %ae_mul_q0, <2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1)
; CHECK: ae_mulsp24x2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulsp24x2_s2.ll
declare <2 x i32> @llvm.xtensa.ae.mulsp24x2.s2(<2 x i32>, <2 x i32>, <2 x i32>)
define <2 x i32> @test_xtensa_ae_mulsp24x2_s2(<2 x i32> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsp24x2_s2:



%ret = call <2 x i32> @llvm.xtensa.ae.mulsp24x2.s2(<2 x i32> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulsp24x2_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulsp32x16x2_h.ll
declare <2 x i32> @llvm.xtensa.ae.mulsp32x16x2.h(<2 x i32>, <2 x i32>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulsp32x16x2_h(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsp32x16x2_h:



%ret = call <2 x i32> @llvm.xtensa.ae.mulsp32x16x2.h(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulsp32x16x2.h aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulsp32x16x2_l.ll
declare <2 x i32> @llvm.xtensa.ae.mulsp32x16x2.l(<2 x i32>, <2 x i32>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulsp32x16x2_l(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsp32x16x2_l:



%ret = call <2 x i32> @llvm.xtensa.ae.mulsp32x16x2.l(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulsp32x16x2.l aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulsp32x2.ll
declare <2 x i32> @llvm.xtensa.ae.mulsp32x2(<2 x i32>, <2 x i32>, <2 x i32>)
define <2 x i32> @test_xtensa_ae_mulsp32x2(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsp32x2:



%ret = call <2 x i32> @llvm.xtensa.ae.mulsp32x2(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulsp32x2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulsq32sp16s_l_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulsq32sp16s.l.s2(<1 x i64>, <1 x i64>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulsq32sp16s_l_s2(<1 x i64> %ae_mul_S2_q0, <1 x i64> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsq32sp16s_l_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulsq32sp16s.l.s2(<1 x i64> %ae_mul_S2_q0, <1 x i64> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulsq32sp16s.l_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulsq32sp16u_l_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulsq32sp16u.l.s2(<1 x i64>, <1 x i64>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulsq32sp16u_l_s2(<1 x i64> %ae_mul_S2_q0, <1 x i64> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsq32sp16u_l_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulsq32sp16u.l.s2(<1 x i64> %ae_mul_S2_q0, <1 x i64> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulsq32sp16u.l_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulsrfq32sp24s_h_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulsrfq32sp24s.h.s2(<1 x i64>, <1 x i64>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulsrfq32sp24s_h_s2(<1 x i64> %ae_mul_S2_q0, <1 x i64> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsrfq32sp24s_h_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulsrfq32sp24s.h.s2(<1 x i64> %ae_mul_S2_q0, <1 x i64> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulsrfq32sp24s.h_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulsrfq32sp24s_l_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulsrfq32sp24s.l.s2(<1 x i64>, <1 x i64>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulsrfq32sp24s_l_s2(<1 x i64> %ae_mul_S2_q0, <1 x i64> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulsrfq32sp24s_l_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulsrfq32sp24s.l.s2(<1 x i64> %ae_mul_S2_q0, <1 x i64> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulsrfq32sp24s.l_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulss32f48p16s_hh.ll
declare <1 x i64> @llvm.xtensa.ae.mulss32f48p16s.hh(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulss32f48p16s_hh(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulss32f48p16s_hh:



%ret = call <1 x i64> @llvm.xtensa.ae.mulss32f48p16s.hh(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulss32f48p16s.hh aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulss32f48p16s_hh_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulss32f48p16s.hh.s2(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulss32f48p16s_hh_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulss32f48p16s_hh_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulss32f48p16s.hh.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulss32f48p16s.hh_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulss32f48p16s_lh.ll
declare <1 x i64> @llvm.xtensa.ae.mulss32f48p16s.lh(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulss32f48p16s_lh(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulss32f48p16s_lh:



%ret = call <1 x i64> @llvm.xtensa.ae.mulss32f48p16s.lh(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulss32f48p16s.lh aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulss32f48p16s_lh_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulss32f48p16s.lh.s2(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulss32f48p16s_lh_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulss32f48p16s_lh_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulss32f48p16s.lh.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulss32f48p16s.lh_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulss32f48p16s_ll.ll
declare <1 x i64> @llvm.xtensa.ae.mulss32f48p16s.ll(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulss32f48p16s_ll(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulss32f48p16s_ll:



%ret = call <1 x i64> @llvm.xtensa.ae.mulss32f48p16s.ll(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulss32f48p16s.ll aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulss32f48p16s_ll_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulss32f48p16s.ll.s2(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulss32f48p16s_ll_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulss32f48p16s_ll_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulss32f48p16s.ll.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulss32f48p16s.ll_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulssd24_hh_ll.ll
declare <1 x i64> @llvm.xtensa.ae.mulssd24.hh.ll(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulssd24_hh_ll(<1 x i64> %ae_mul_q0, <2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulssd24_hh_ll:



%ret = call <1 x i64> @llvm.xtensa.ae.mulssd24.hh.ll(<1 x i64> %ae_mul_q0, <2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1)
; CHECK: ae_mulssd24.hh.ll aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulssd24_hh_ll_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulssd24.hh.ll.s2(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulssd24_hh_ll_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulssd24_hh_ll_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulssd24.hh.ll.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulssd24.hh.ll_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulssd24_hl_lh.ll
declare <1 x i64> @llvm.xtensa.ae.mulssd24.hl.lh(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulssd24_hl_lh(<1 x i64> %ae_mul_q0, <2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulssd24_hl_lh:



%ret = call <1 x i64> @llvm.xtensa.ae.mulssd24.hl.lh(<1 x i64> %ae_mul_q0, <2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1)
; CHECK: ae_mulssd24.hl.lh aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulssd24_hl_lh_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulssd24.hl.lh.s2(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulssd24_hl_lh_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulssd24_hl_lh_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulssd24.hl.lh.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulssd24.hl.lh_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulssd32x16_h1_l0.ll
declare <1 x i64> @llvm.xtensa.ae.mulssd32x16.h1.l0(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulssd32x16_h1_l0(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulssd32x16_h1_l0:



%ret = call <1 x i64> @llvm.xtensa.ae.mulssd32x16.h1.l0(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulssd32x16.h1.l0 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulssd32x16_h1_l0_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulssd32x16.h1.l0.s2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulssd32x16_h1_l0_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulssd32x16_h1_l0_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulssd32x16.h1.l0.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulssd32x16.h1.l0_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulssd32x16_h3_l2.ll
declare <1 x i64> @llvm.xtensa.ae.mulssd32x16.h3.l2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulssd32x16_h3_l2(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulssd32x16_h3_l2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulssd32x16.h3.l2(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulssd32x16.h3.l2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulssd32x16_h3_l2_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulssd32x16.h3.l2.s2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulssd32x16_h3_l2_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulssd32x16_h3_l2_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulssd32x16.h3.l2.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulssd32x16.h3.l2_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulssfd16ss_11_00.ll
declare <2 x i32> @llvm.xtensa.ae.mulssfd16ss.11.00(<2 x i32>, <4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulssfd16ss_11_00(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulssfd16ss_11_00:



%ret = call <2 x i32> @llvm.xtensa.ae.mulssfd16ss.11.00(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulssfd16ss.11_00 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulssfd16ss_11_00_s2.ll
declare <2 x i32> @llvm.xtensa.ae.mulssfd16ss.11.00.s2(<2 x i32>, <4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulssfd16ss_11_00_s2(<2 x i32> %ae_mul_S2_q0, <4 x i16> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulssfd16ss_11_00_s2:



%ret = call <2 x i32> @llvm.xtensa.ae.mulssfd16ss.11.00.s2(<2 x i32> %ae_mul_S2_q0, <4 x i16> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulssfd16ss.11_00_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulssfd16ss_13_02.ll
declare <2 x i32> @llvm.xtensa.ae.mulssfd16ss.13.02(<2 x i32>, <4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulssfd16ss_13_02(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulssfd16ss_13_02:



%ret = call <2 x i32> @llvm.xtensa.ae.mulssfd16ss.13.02(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulssfd16ss.13_02 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulssfd16ss_13_02_s2.ll
declare <2 x i32> @llvm.xtensa.ae.mulssfd16ss.13.02.s2(<2 x i32>, <4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulssfd16ss_13_02_s2(<2 x i32> %ae_mul_S2_q0, <4 x i16> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulssfd16ss_13_02_s2:



%ret = call <2 x i32> @llvm.xtensa.ae.mulssfd16ss.13.02.s2(<2 x i32> %ae_mul_S2_q0, <4 x i16> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulssfd16ss.13_02_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulssfd16ss_33_22.ll
declare <2 x i32> @llvm.xtensa.ae.mulssfd16ss.33.22(<2 x i32>, <4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulssfd16ss_33_22(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulssfd16ss_33_22:



%ret = call <2 x i32> @llvm.xtensa.ae.mulssfd16ss.33.22(<2 x i32> %opnd_ae_sem_mul_x2_S1_q0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulssfd16ss.33_22 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulssfd16ss_33_22_s2.ll
declare <2 x i32> @llvm.xtensa.ae.mulssfd16ss.33.22.s2(<2 x i32>, <4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulssfd16ss_33_22_s2(<2 x i32> %ae_mul_S2_q0, <4 x i16> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulssfd16ss_33_22_s2:



%ret = call <2 x i32> @llvm.xtensa.ae.mulssfd16ss.33.22.s2(<2 x i32> %ae_mul_S2_q0, <4 x i16> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulssfd16ss.33_22_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulssfd24_hh_ll.ll
declare <1 x i64> @llvm.xtensa.ae.mulssfd24.hh.ll(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulssfd24_hh_ll(<1 x i64> %ae_mul_q0, <2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulssfd24_hh_ll:



%ret = call <1 x i64> @llvm.xtensa.ae.mulssfd24.hh.ll(<1 x i64> %ae_mul_q0, <2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1)
; CHECK: ae_mulssfd24.hh.ll aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulssfd24_hh_ll_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulssfd24.hh.ll.s2(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulssfd24_hh_ll_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulssfd24_hh_ll_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulssfd24.hh.ll.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulssfd24.hh.ll_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulssfd24_hl_lh.ll
declare <1 x i64> @llvm.xtensa.ae.mulssfd24.hl.lh(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulssfd24_hl_lh(<1 x i64> %ae_mul_q0, <2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulssfd24_hl_lh:



%ret = call <1 x i64> @llvm.xtensa.ae.mulssfd24.hl.lh(<1 x i64> %ae_mul_q0, <2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1)
; CHECK: ae_mulssfd24.hl.lh aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulssfd24_hl_lh_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulssfd24.hl.lh.s2(<1 x i64>, <2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulssfd24_hl_lh_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulssfd24_hl_lh_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulssfd24.hl.lh.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulssfd24.hl.lh_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulssfd32x16_h1_l0.ll
declare <1 x i64> @llvm.xtensa.ae.mulssfd32x16.h1.l0(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulssfd32x16_h1_l0(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulssfd32x16_h1_l0:



%ret = call <1 x i64> @llvm.xtensa.ae.mulssfd32x16.h1.l0(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulssfd32x16.h1.l0 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulssfd32x16_h1_l0_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulssfd32x16.h1.l0.s2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulssfd32x16_h1_l0_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulssfd32x16_h1_l0_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulssfd32x16.h1.l0.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulssfd32x16.h1.l0_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulssfd32x16_h3_l2.ll
declare <1 x i64> @llvm.xtensa.ae.mulssfd32x16.h3.l2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulssfd32x16_h3_l2(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulssfd32x16_h3_l2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulssfd32x16.h3.l2(<1 x i64> %opnd_ae_sem_mul_x2_S1_q0, <2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulssfd32x16.h3.l2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulssfd32x16_h3_l2_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulssfd32x16.h3.l2.s2(<1 x i64>, <2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulssfd32x16_h3_l2_s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulssfd32x16_h3_l2_s2:



%ret = call <1 x i64> @llvm.xtensa.ae.mulssfd32x16.h3.l2.s2(<1 x i64> %ae_mul_S2_q0, <2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulssfd32x16.h3.l2_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzaad24_hh_ll.ll
declare <1 x i64> @llvm.xtensa.ae.mulzaad24.hh.ll(<2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulzaad24_hh_ll(<2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzaad24_hh_ll:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzaad24.hh.ll(<2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1)
; CHECK: ae_mulzaad24.hh.ll aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzaad24_hh_ll_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulzaad24.hh.ll.s2(<2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulzaad24_hh_ll_s2(<2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzaad24_hh_ll_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzaad24.hh.ll.s2(<2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulzaad24.hh.ll_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzaad24_hl_lh.ll
declare <1 x i64> @llvm.xtensa.ae.mulzaad24.hl.lh(<2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulzaad24_hl_lh(<2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzaad24_hl_lh:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzaad24.hl.lh(<2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1)
; CHECK: ae_mulzaad24.hl.lh aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzaad24_hl_lh_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulzaad24.hl.lh.s2(<2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulzaad24_hl_lh_s2(<2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzaad24_hl_lh_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzaad24.hl.lh.s2(<2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulzaad24.hl.lh_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzaad32x16_h0_l1.ll
declare <1 x i64> @llvm.xtensa.ae.mulzaad32x16.h0.l1(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulzaad32x16_h0_l1(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzaad32x16_h0_l1:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzaad32x16.h0.l1(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulzaad32x16.h0.l1 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzaad32x16_h0_l1_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulzaad32x16.h0.l1.s2(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulzaad32x16_h0_l1_s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzaad32x16_h0_l1_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzaad32x16.h0.l1.s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulzaad32x16.h0.l1_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzaad32x16_h1_l0.ll
declare <1 x i64> @llvm.xtensa.ae.mulzaad32x16.h1.l0(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulzaad32x16_h1_l0(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzaad32x16_h1_l0:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzaad32x16.h1.l0(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulzaad32x16.h1.l0 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzaad32x16_h1_l0_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulzaad32x16.h1.l0.s2(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulzaad32x16_h1_l0_s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzaad32x16_h1_l0_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzaad32x16.h1.l0.s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulzaad32x16.h1.l0_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzaad32x16_h2_l3.ll
declare <1 x i64> @llvm.xtensa.ae.mulzaad32x16.h2.l3(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulzaad32x16_h2_l3(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzaad32x16_h2_l3:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzaad32x16.h2.l3(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulzaad32x16.h2.l3 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzaad32x16_h2_l3_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulzaad32x16.h2.l3.s2(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulzaad32x16_h2_l3_s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzaad32x16_h2_l3_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzaad32x16.h2.l3.s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulzaad32x16.h2.l3_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzaad32x16_h3_l2.ll
declare <1 x i64> @llvm.xtensa.ae.mulzaad32x16.h3.l2(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulzaad32x16_h3_l2(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzaad32x16_h3_l2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzaad32x16.h3.l2(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulzaad32x16.h3.l2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzaad32x16_h3_l2_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulzaad32x16.h3.l2.s2(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulzaad32x16_h3_l2_s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzaad32x16_h3_l2_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzaad32x16.h3.l2.s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulzaad32x16.h3.l2_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzaafd16ss_11_00.ll
declare <2 x i32> @llvm.xtensa.ae.mulzaafd16ss.11.00(<4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulzaafd16ss_11_00(<4 x i16> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzaafd16ss_11_00:


%ret = call <2 x i32> @llvm.xtensa.ae.mulzaafd16ss.11.00(<4 x i16> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulzaafd16ss.11_00 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulzaafd16ss_11_00_s2.ll
declare <2 x i32> @llvm.xtensa.ae.mulzaafd16ss.11.00.s2(<4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulzaafd16ss_11_00_s2(<4 x i16> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzaafd16ss_11_00_s2:


%ret = call <2 x i32> @llvm.xtensa.ae.mulzaafd16ss.11.00.s2(<4 x i16> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulzaafd16ss.11_00_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulzaafd16ss_13_02.ll
declare <2 x i32> @llvm.xtensa.ae.mulzaafd16ss.13.02(<4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulzaafd16ss_13_02(<4 x i16> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzaafd16ss_13_02:


%ret = call <2 x i32> @llvm.xtensa.ae.mulzaafd16ss.13.02(<4 x i16> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulzaafd16ss.13_02 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulzaafd16ss_13_02_s2.ll
declare <2 x i32> @llvm.xtensa.ae.mulzaafd16ss.13.02.s2(<4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulzaafd16ss_13_02_s2(<4 x i16> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzaafd16ss_13_02_s2:


%ret = call <2 x i32> @llvm.xtensa.ae.mulzaafd16ss.13.02.s2(<4 x i16> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulzaafd16ss.13_02_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulzaafd16ss_33_22.ll
declare <2 x i32> @llvm.xtensa.ae.mulzaafd16ss.33.22(<4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulzaafd16ss_33_22(<4 x i16> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzaafd16ss_33_22:


%ret = call <2 x i32> @llvm.xtensa.ae.mulzaafd16ss.33.22(<4 x i16> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulzaafd16ss.33_22 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulzaafd16ss_33_22_s2.ll
declare <2 x i32> @llvm.xtensa.ae.mulzaafd16ss.33.22.s2(<4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulzaafd16ss_33_22_s2(<4 x i16> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzaafd16ss_33_22_s2:


%ret = call <2 x i32> @llvm.xtensa.ae.mulzaafd16ss.33.22.s2(<4 x i16> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulzaafd16ss.33_22_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulzaafd24_hh_ll.ll
declare <1 x i64> @llvm.xtensa.ae.mulzaafd24.hh.ll(<2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulzaafd24_hh_ll(<2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzaafd24_hh_ll:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzaafd24.hh.ll(<2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1)
; CHECK: ae_mulzaafd24.hh.ll aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzaafd24_hh_ll_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulzaafd24.hh.ll.s2(<2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulzaafd24_hh_ll_s2(<2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzaafd24_hh_ll_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzaafd24.hh.ll.s2(<2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulzaafd24.hh.ll_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzaafd24_hl_lh.ll
declare <1 x i64> @llvm.xtensa.ae.mulzaafd24.hl.lh(<2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulzaafd24_hl_lh(<2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzaafd24_hl_lh:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzaafd24.hl.lh(<2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1)
; CHECK: ae_mulzaafd24.hl.lh aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzaafd24_hl_lh_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulzaafd24.hl.lh.s2(<2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulzaafd24_hl_lh_s2(<2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzaafd24_hl_lh_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzaafd24.hl.lh.s2(<2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulzaafd24.hl.lh_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzaafd32x16_h0_l1.ll
declare <1 x i64> @llvm.xtensa.ae.mulzaafd32x16.h0.l1(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulzaafd32x16_h0_l1(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzaafd32x16_h0_l1:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzaafd32x16.h0.l1(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulzaafd32x16.h0.l1 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzaafd32x16_h0_l1_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulzaafd32x16.h0.l1.s2(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulzaafd32x16_h0_l1_s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzaafd32x16_h0_l1_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzaafd32x16.h0.l1.s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulzaafd32x16.h0.l1_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzaafd32x16_h1_l0.ll
declare <1 x i64> @llvm.xtensa.ae.mulzaafd32x16.h1.l0(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulzaafd32x16_h1_l0(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzaafd32x16_h1_l0:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzaafd32x16.h1.l0(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulzaafd32x16.h1.l0 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzaafd32x16_h1_l0_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulzaafd32x16.h1.l0.s2(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulzaafd32x16_h1_l0_s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzaafd32x16_h1_l0_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzaafd32x16.h1.l0.s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulzaafd32x16.h1.l0_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzaafd32x16_h2_l3.ll
declare <1 x i64> @llvm.xtensa.ae.mulzaafd32x16.h2.l3(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulzaafd32x16_h2_l3(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzaafd32x16_h2_l3:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzaafd32x16.h2.l3(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulzaafd32x16.h2.l3 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzaafd32x16_h2_l3_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulzaafd32x16.h2.l3.s2(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulzaafd32x16_h2_l3_s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzaafd32x16_h2_l3_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzaafd32x16.h2.l3.s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulzaafd32x16.h2.l3_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzaafd32x16_h3_l2.ll
declare <1 x i64> @llvm.xtensa.ae.mulzaafd32x16.h3.l2(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulzaafd32x16_h3_l2(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzaafd32x16_h3_l2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzaafd32x16.h3.l2(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulzaafd32x16.h3.l2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzaafd32x16_h3_l2_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulzaafd32x16.h3.l2.s2(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulzaafd32x16_h3_l2_s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzaafd32x16_h3_l2_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzaafd32x16.h3.l2.s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulzaafd32x16.h3.l2_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzasd24_hh_ll.ll
declare <1 x i64> @llvm.xtensa.ae.mulzasd24.hh.ll(<2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulzasd24_hh_ll(<2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzasd24_hh_ll:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzasd24.hh.ll(<2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1)
; CHECK: ae_mulzasd24.hh.ll aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzasd24_hh_ll_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulzasd24.hh.ll.s2(<2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulzasd24_hh_ll_s2(<2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzasd24_hh_ll_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzasd24.hh.ll.s2(<2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulzasd24.hh.ll_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzasd24_hl_lh.ll
declare <1 x i64> @llvm.xtensa.ae.mulzasd24.hl.lh(<2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulzasd24_hl_lh(<2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzasd24_hl_lh:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzasd24.hl.lh(<2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1)
; CHECK: ae_mulzasd24.hl.lh aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzasd24_hl_lh_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulzasd24.hl.lh.s2(<2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulzasd24_hl_lh_s2(<2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzasd24_hl_lh_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzasd24.hl.lh.s2(<2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulzasd24.hl.lh_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzasd32x16_h1_l0.ll
declare <1 x i64> @llvm.xtensa.ae.mulzasd32x16.h1.l0(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulzasd32x16_h1_l0(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzasd32x16_h1_l0:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzasd32x16.h1.l0(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulzasd32x16.h1.l0 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzasd32x16_h1_l0_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulzasd32x16.h1.l0.s2(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulzasd32x16_h1_l0_s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzasd32x16_h1_l0_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzasd32x16.h1.l0.s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulzasd32x16.h1.l0_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzasd32x16_h3_l2.ll
declare <1 x i64> @llvm.xtensa.ae.mulzasd32x16.h3.l2(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulzasd32x16_h3_l2(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzasd32x16_h3_l2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzasd32x16.h3.l2(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulzasd32x16.h3.l2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzasd32x16_h3_l2_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulzasd32x16.h3.l2.s2(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulzasd32x16_h3_l2_s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzasd32x16_h3_l2_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzasd32x16.h3.l2.s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulzasd32x16.h3.l2_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzasfd24_hh_ll.ll
declare <1 x i64> @llvm.xtensa.ae.mulzasfd24.hh.ll(<2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulzasfd24_hh_ll(<2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzasfd24_hh_ll:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzasfd24.hh.ll(<2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1)
; CHECK: ae_mulzasfd24.hh.ll aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzasfd24_hh_ll_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulzasfd24.hh.ll.s2(<2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulzasfd24_hh_ll_s2(<2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzasfd24_hh_ll_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzasfd24.hh.ll.s2(<2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulzasfd24.hh.ll_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzasfd24_hl_lh.ll
declare <1 x i64> @llvm.xtensa.ae.mulzasfd24.hl.lh(<2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulzasfd24_hl_lh(<2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzasfd24_hl_lh:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzasfd24.hl.lh(<2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1)
; CHECK: ae_mulzasfd24.hl.lh aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzasfd24_hl_lh_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulzasfd24.hl.lh.s2(<2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulzasfd24_hl_lh_s2(<2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzasfd24_hl_lh_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzasfd24.hl.lh.s2(<2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulzasfd24.hl.lh_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzasfd32x16_h1_l0.ll
declare <1 x i64> @llvm.xtensa.ae.mulzasfd32x16.h1.l0(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulzasfd32x16_h1_l0(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzasfd32x16_h1_l0:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzasfd32x16.h1.l0(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulzasfd32x16.h1.l0 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzasfd32x16_h1_l0_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulzasfd32x16.h1.l0.s2(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulzasfd32x16_h1_l0_s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzasfd32x16_h1_l0_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzasfd32x16.h1.l0.s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulzasfd32x16.h1.l0_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzasfd32x16_h3_l2.ll
declare <1 x i64> @llvm.xtensa.ae.mulzasfd32x16.h3.l2(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulzasfd32x16_h3_l2(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzasfd32x16_h3_l2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzasfd32x16.h3.l2(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulzasfd32x16.h3.l2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzasfd32x16_h3_l2_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulzasfd32x16.h3.l2.s2(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulzasfd32x16_h3_l2_s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzasfd32x16_h3_l2_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzasfd32x16.h3.l2.s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulzasfd32x16.h3.l2_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzsad24_hh_ll.ll
declare <1 x i64> @llvm.xtensa.ae.mulzsad24.hh.ll(<2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulzsad24_hh_ll(<2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzsad24_hh_ll:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzsad24.hh.ll(<2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1)
; CHECK: ae_mulzsad24.hh.ll aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzsad24_hh_ll_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulzsad24.hh.ll.s2(<2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulzsad24_hh_ll_s2(<2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzsad24_hh_ll_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzsad24.hh.ll.s2(<2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulzsad24.hh.ll_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzsad32x16_h1_l0.ll
declare <1 x i64> @llvm.xtensa.ae.mulzsad32x16.h1.l0(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulzsad32x16_h1_l0(<2 x i32> %ae_mul_d0, <4 x i16> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzsad32x16_h1_l0:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzsad32x16.h1.l0(<2 x i32> %ae_mul_d0, <4 x i16> %ae_mul_d1)
; CHECK: ae_mulzsad32x16.h1.l0 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzsad32x16_h1_l0_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulzsad32x16.h1.l0.s2(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulzsad32x16_h1_l0_s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzsad32x16_h1_l0_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzsad32x16.h1.l0.s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulzsad32x16.h1.l0_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzsad32x16_h3_l2.ll
declare <1 x i64> @llvm.xtensa.ae.mulzsad32x16.h3.l2(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulzsad32x16_h3_l2(<2 x i32> %ae_mul_d0, <4 x i16> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzsad32x16_h3_l2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzsad32x16.h3.l2(<2 x i32> %ae_mul_d0, <4 x i16> %ae_mul_d1)
; CHECK: ae_mulzsad32x16.h3.l2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzsad32x16_h3_l2_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulzsad32x16.h3.l2.s2(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulzsad32x16_h3_l2_s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzsad32x16_h3_l2_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzsad32x16.h3.l2.s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulzsad32x16.h3.l2_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzsafd24_hh_ll.ll
declare <1 x i64> @llvm.xtensa.ae.mulzsafd24.hh.ll(<2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulzsafd24_hh_ll(<2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzsafd24_hh_ll:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzsafd24.hh.ll(<2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1)
; CHECK: ae_mulzsafd24.hh.ll aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzsafd24_hh_ll_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulzsafd24.hh.ll.s2(<2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulzsafd24_hh_ll_s2(<2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzsafd24_hh_ll_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzsafd24.hh.ll.s2(<2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulzsafd24.hh.ll_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzsafd32x16_h1_l0.ll
declare <1 x i64> @llvm.xtensa.ae.mulzsafd32x16.h1.l0(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulzsafd32x16_h1_l0(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzsafd32x16_h1_l0:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzsafd32x16.h1.l0(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulzsafd32x16.h1.l0 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzsafd32x16_h1_l0_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulzsafd32x16.h1.l0.s2(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulzsafd32x16_h1_l0_s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzsafd32x16_h1_l0_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzsafd32x16.h1.l0.s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulzsafd32x16.h1.l0_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzsafd32x16_h3_l2.ll
declare <1 x i64> @llvm.xtensa.ae.mulzsafd32x16.h3.l2(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulzsafd32x16_h3_l2(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzsafd32x16_h3_l2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzsafd32x16.h3.l2(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulzsafd32x16.h3.l2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzsafd32x16_h3_l2_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulzsafd32x16.h3.l2.s2(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulzsafd32x16_h3_l2_s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzsafd32x16_h3_l2_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzsafd32x16.h3.l2.s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulzsafd32x16.h3.l2_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzssd24_hh_ll.ll
declare <1 x i64> @llvm.xtensa.ae.mulzssd24.hh.ll(<2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulzssd24_hh_ll(<2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzssd24_hh_ll:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzssd24.hh.ll(<2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1)
; CHECK: ae_mulzssd24.hh.ll aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzssd24_hh_ll_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulzssd24.hh.ll.s2(<2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulzssd24_hh_ll_s2(<2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzssd24_hh_ll_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzssd24.hh.ll.s2(<2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulzssd24.hh.ll_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzssd24_hl_lh.ll
declare <1 x i64> @llvm.xtensa.ae.mulzssd24.hl.lh(<2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulzssd24_hl_lh(<2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzssd24_hl_lh:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzssd24.hl.lh(<2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1)
; CHECK: ae_mulzssd24.hl.lh aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzssd24_hl_lh_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulzssd24.hl.lh.s2(<2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulzssd24_hl_lh_s2(<2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzssd24_hl_lh_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzssd24.hl.lh.s2(<2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulzssd24.hl.lh_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzssd32x16_h1_l0.ll
declare <1 x i64> @llvm.xtensa.ae.mulzssd32x16.h1.l0(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulzssd32x16_h1_l0(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzssd32x16_h1_l0:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzssd32x16.h1.l0(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulzssd32x16.h1.l0 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzssd32x16_h1_l0_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulzssd32x16.h1.l0.s2(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulzssd32x16_h1_l0_s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzssd32x16_h1_l0_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzssd32x16.h1.l0.s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulzssd32x16.h1.l0_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzssd32x16_h3_l2.ll
declare <1 x i64> @llvm.xtensa.ae.mulzssd32x16.h3.l2(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulzssd32x16_h3_l2(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzssd32x16_h3_l2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzssd32x16.h3.l2(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulzssd32x16.h3.l2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzssd32x16_h3_l2_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulzssd32x16.h3.l2.s2(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulzssd32x16_h3_l2_s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzssd32x16_h3_l2_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzssd32x16.h3.l2.s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulzssd32x16.h3.l2_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzssfd16ss_11_00.ll
declare <2 x i32> @llvm.xtensa.ae.mulzssfd16ss.11.00(<4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulzssfd16ss_11_00(<4 x i16> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzssfd16ss_11_00:


%ret = call <2 x i32> @llvm.xtensa.ae.mulzssfd16ss.11.00(<4 x i16> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulzssfd16ss.11_00 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulzssfd16ss_11_00_s2.ll
declare <2 x i32> @llvm.xtensa.ae.mulzssfd16ss.11.00.s2(<4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulzssfd16ss_11_00_s2(<4 x i16> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzssfd16ss_11_00_s2:


%ret = call <2 x i32> @llvm.xtensa.ae.mulzssfd16ss.11.00.s2(<4 x i16> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulzssfd16ss.11_00_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulzssfd16ss_13_02.ll
declare <2 x i32> @llvm.xtensa.ae.mulzssfd16ss.13.02(<4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulzssfd16ss_13_02(<4 x i16> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzssfd16ss_13_02:


%ret = call <2 x i32> @llvm.xtensa.ae.mulzssfd16ss.13.02(<4 x i16> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulzssfd16ss.13_02 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulzssfd16ss_13_02_s2.ll
declare <2 x i32> @llvm.xtensa.ae.mulzssfd16ss.13.02.s2(<4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulzssfd16ss_13_02_s2(<4 x i16> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzssfd16ss_13_02_s2:


%ret = call <2 x i32> @llvm.xtensa.ae.mulzssfd16ss.13.02.s2(<4 x i16> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulzssfd16ss.13_02_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulzssfd16ss_33_22.ll
declare <2 x i32> @llvm.xtensa.ae.mulzssfd16ss.33.22(<4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulzssfd16ss_33_22(<4 x i16> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzssfd16ss_33_22:


%ret = call <2 x i32> @llvm.xtensa.ae.mulzssfd16ss.33.22(<4 x i16> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulzssfd16ss.33_22 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulzssfd16ss_33_22_s2.ll
declare <2 x i32> @llvm.xtensa.ae.mulzssfd16ss.33.22.s2(<4 x i16>, <4 x i16>)
define <2 x i32> @test_xtensa_ae_mulzssfd16ss_33_22_s2(<4 x i16> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzssfd16ss_33_22_s2:


%ret = call <2 x i32> @llvm.xtensa.ae.mulzssfd16ss.33.22.s2(<4 x i16> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulzssfd16ss.33_22_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_mulzssfd24_hh_ll.ll
declare <1 x i64> @llvm.xtensa.ae.mulzssfd24.hh.ll(<2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulzssfd24_hh_ll(<2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzssfd24_hh_ll:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzssfd24.hh.ll(<2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1)
; CHECK: ae_mulzssfd24.hh.ll aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzssfd24_hh_ll_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulzssfd24.hh.ll.s2(<2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulzssfd24_hh_ll_s2(<2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzssfd24_hh_ll_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzssfd24.hh.ll.s2(<2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulzssfd24.hh.ll_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzssfd24_hl_lh.ll
declare <1 x i64> @llvm.xtensa.ae.mulzssfd24.hl.lh(<2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulzssfd24_hl_lh(<2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzssfd24_hl_lh:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzssfd24.hl.lh(<2 x i32> %ae_mul_d0, <2 x i32> %ae_mul_d1)
; CHECK: ae_mulzssfd24.hl.lh aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzssfd24_hl_lh_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulzssfd24.hl.lh.s2(<2 x i32>, <2 x i32>)
define <1 x i64> @test_xtensa_ae_mulzssfd24_hl_lh_s2(<2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzssfd24_hl_lh_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzssfd24.hl.lh.s2(<2 x i32> %ae_mul_S2_d0, <2 x i32> %ae_mul_S2_d1)
; CHECK: ae_mulzssfd24.hl.lh_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzssfd32x16_h1_l0.ll
declare <1 x i64> @llvm.xtensa.ae.mulzssfd32x16.h1.l0(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulzssfd32x16_h1_l0(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzssfd32x16_h1_l0:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzssfd32x16.h1.l0(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulzssfd32x16.h1.l0 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzssfd32x16_h1_l0_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulzssfd32x16.h1.l0.s2(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulzssfd32x16_h1_l0_s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzssfd32x16_h1_l0_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzssfd32x16.h1.l0.s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulzssfd32x16.h1.l0_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzssfd32x16_h3_l2.ll
declare <1 x i64> @llvm.xtensa.ae.mulzssfd32x16.h3.l2(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulzssfd32x16_h3_l2(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzssfd32x16_h3_l2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzssfd32x16.h3.l2(<2 x i32> %opnd_ae_sem_mul_x2_S1_d0, <4 x i16> %opnd_ae_sem_mul_x2_S1_d1)
; CHECK: ae_mulzssfd32x16.h3.l2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_mulzssfd32x16_h3_l2_s2.ll
declare <1 x i64> @llvm.xtensa.ae.mulzssfd32x16.h3.l2.s2(<2 x i32>, <4 x i16>)
define <1 x i64> @test_xtensa_ae_mulzssfd32x16_h3_l2_s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1) {
; CHECK-LABEL: test_xtensa_ae_mulzssfd32x16_h3_l2_s2:


%ret = call <1 x i64> @llvm.xtensa.ae.mulzssfd32x16.h3.l2.s2(<2 x i32> %ae_mul_S2_d0, <4 x i16> %ae_mul_S2_d1)
; CHECK: ae_mulzssfd32x16.h3.l2_s2 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_nand.ll
declare <1 x i64> @llvm.xtensa.ae.nand(<1 x i64>, <1 x i64>)
define <1 x i64> @test_xtensa_ae_nand(<1 x i64> %ae_dr_to_dr_v0, <1 x i64> %ae_dr_to_dr_v1) {
; CHECK-LABEL: test_xtensa_ae_nand:


%ret = call <1 x i64> @llvm.xtensa.ae.nand(<1 x i64> %ae_dr_to_dr_v0, <1 x i64> %ae_dr_to_dr_v1)
; CHECK: ae_nand aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_neg16s.ll
declare <4 x i16> @llvm.xtensa.ae.neg16s(<4 x i16>)
define <4 x i16> @test_xtensa_ae_neg16s(<4 x i16> %ae_arth_v1) {
; CHECK-LABEL: test_xtensa_ae_neg16s:

%ret = call <4 x i16> @llvm.xtensa.ae.neg16s(<4 x i16> %ae_arth_v1)
; CHECK: ae_neg16s aed{{[0-9]+}}, aed{{[0-9]+}}

ret <4 x i16> %ret
}


;--- ae_neg24s.ll
declare <2 x i32> @llvm.xtensa.ae.neg24s(<2 x i32>)
define <2 x i32> @test_xtensa_ae_neg24s(<2 x i32> %ae_arth_v1) {
; CHECK-LABEL: test_xtensa_ae_neg24s:

%ret = call <2 x i32> @llvm.xtensa.ae.neg24s(<2 x i32> %ae_arth_v1)
; CHECK: ae_neg24s aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_neg32.ll
declare <2 x i32> @llvm.xtensa.ae.neg32(<2 x i32>)
define <2 x i32> @test_xtensa_ae_neg32(<2 x i32> %ae_arth_v1) {
; CHECK-LABEL: test_xtensa_ae_neg32:

%ret = call <2 x i32> @llvm.xtensa.ae.neg32(<2 x i32> %ae_arth_v1)
; CHECK: ae_neg32 aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_neg32s.ll
declare <2 x i32> @llvm.xtensa.ae.neg32s(<2 x i32>)
define <2 x i32> @test_xtensa_ae_neg32s(<2 x i32> %ae_arth_v1) {
; CHECK-LABEL: test_xtensa_ae_neg32s:

%ret = call <2 x i32> @llvm.xtensa.ae.neg32s(<2 x i32> %ae_arth_v1)
; CHECK: ae_neg32s aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_neg64.ll
declare <1 x i64> @llvm.xtensa.ae.neg64(<1 x i64>)
define <1 x i64> @test_xtensa_ae_neg64(<1 x i64> %ae_arth_v1) {
; CHECK-LABEL: test_xtensa_ae_neg64:

%ret = call <1 x i64> @llvm.xtensa.ae.neg64(<1 x i64> %ae_arth_v1)
; CHECK: ae_neg64 aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_neg64s.ll
declare <1 x i64> @llvm.xtensa.ae.neg64s(<1 x i64>)
define <1 x i64> @test_xtensa_ae_neg64s(<1 x i64> %ae_arth_v1) {
; CHECK-LABEL: test_xtensa_ae_neg64s:

%ret = call <1 x i64> @llvm.xtensa.ae.neg64s(<1 x i64> %ae_arth_v1)
; CHECK: ae_neg64s aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_nsa64.ll
declare i32 @llvm.xtensa.ae.nsa64(<1 x i64>)
define i32 @test_xtensa_ae_nsa64(<1 x i64> %ae_dr_to_ar_v0) {
; CHECK-LABEL: test_xtensa_ae_nsa64:

%ret = call i32 @llvm.xtensa.ae.nsa64(<1 x i64> %ae_dr_to_ar_v0)
; CHECK: ae_nsa64 a{{[0-9]+}}, aed{{[0-9]+}}

ret i32 %ret
}


;--- ae_nsaz16_0.ll
declare i32 @llvm.xtensa.ae.nsaz16.0(<4 x i16>)
define i32 @test_xtensa_ae_nsaz16_0(<4 x i16> %ae_dr_to_ar_v0) {
; CHECK-LABEL: test_xtensa_ae_nsaz16_0:

%ret = call i32 @llvm.xtensa.ae.nsaz16.0(<4 x i16> %ae_dr_to_ar_v0)
; CHECK: ae_nsaz16.0 a{{[0-9]+}}, aed{{[0-9]+}}

ret i32 %ret
}


;--- ae_nsaz32_l.ll
declare i32 @llvm.xtensa.ae.nsaz32.l(<2 x i32>)
define i32 @test_xtensa_ae_nsaz32_l(<2 x i32> %ae_dr_to_ar_v0) {
; CHECK-LABEL: test_xtensa_ae_nsaz32_l:

%ret = call i32 @llvm.xtensa.ae.nsaz32.l(<2 x i32> %ae_dr_to_ar_v0)
; CHECK: ae_nsaz32.l a{{[0-9]+}}, aed{{[0-9]+}}

ret i32 %ret
}


;--- ae_or.ll
declare <1 x i64> @llvm.xtensa.ae.or(<1 x i64>, <1 x i64>)
define <1 x i64> @test_xtensa_ae_or(<1 x i64> %ae_dr_to_dr_v0, <1 x i64> %ae_dr_to_dr_v1) {
; CHECK-LABEL: test_xtensa_ae_or:


%ret = call <1 x i64> @llvm.xtensa.ae.or(<1 x i64> %ae_dr_to_dr_v0, <1 x i64> %ae_dr_to_dr_v1)
; CHECK: ae_or aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_pksr24.ll
declare <2 x i32> @llvm.xtensa.ae.pksr24(<2 x i32>, <1 x i64>, i32)
define <2 x i32> @test_xtensa_ae_pksr24(<2 x i32> %ae_pks_d, <1 x i64> %ae_pks_s) {
; CHECK-LABEL: test_xtensa_ae_pksr24:



%ret = call <2 x i32> @llvm.xtensa.ae.pksr24(<2 x i32> %ae_pks_d, <1 x i64> %ae_pks_s, i32 0)
; CHECK: ae_pksr24 aed{{[0-9]+}}, aed{{[0-9]+}}, {{-?[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_pksr32.ll
declare <2 x i32> @llvm.xtensa.ae.pksr32(<2 x i32>, <1 x i64>, i32)
define <2 x i32> @test_xtensa_ae_pksr32(<2 x i32> %ae_pks_d, <1 x i64> %ae_pks_s) {
; CHECK-LABEL: test_xtensa_ae_pksr32:



%ret = call <2 x i32> @llvm.xtensa.ae.pksr32(<2 x i32> %ae_pks_d, <1 x i64> %ae_pks_s, i32 0)
; CHECK: ae_pksr32 aed{{[0-9]+}}, aed{{[0-9]+}}, {{-?[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_round16x4f32sasym.ll
declare <4 x i16> @llvm.xtensa.ae.round16x4f32sasym(<2 x i32>, <2 x i32>)
define <4 x i16> @test_xtensa_ae_round16x4f32sasym(<2 x i32> %ae_arth_v1, <2 x i32> %ae_arth_v0) {
; CHECK-LABEL: test_xtensa_ae_round16x4f32sasym:


%ret = call <4 x i16> @llvm.xtensa.ae.round16x4f32sasym(<2 x i32> %ae_arth_v1, <2 x i32> %ae_arth_v0)
; CHECK: ae_round16x4f32sasym aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <4 x i16> %ret
}


;--- ae_round16x4f32ssym.ll
declare <4 x i16> @llvm.xtensa.ae.round16x4f32ssym(<2 x i32>, <2 x i32>)
define <4 x i16> @test_xtensa_ae_round16x4f32ssym(<2 x i32> %ae_arth_v1, <2 x i32> %ae_arth_v0) {
; CHECK-LABEL: test_xtensa_ae_round16x4f32ssym:


%ret = call <4 x i16> @llvm.xtensa.ae.round16x4f32ssym(<2 x i32> %ae_arth_v1, <2 x i32> %ae_arth_v0)
; CHECK: ae_round16x4f32ssym aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <4 x i16> %ret
}


;--- ae_round24x2f48sasym.ll
declare <2 x i32> @llvm.xtensa.ae.round24x2f48sasym(<1 x i64>, <1 x i64>)
define <2 x i32> @test_xtensa_ae_round24x2f48sasym(<1 x i64> %ae_arth_v0, <1 x i64> %ae_arth_v1) {
; CHECK-LABEL: test_xtensa_ae_round24x2f48sasym:


%ret = call <2 x i32> @llvm.xtensa.ae.round24x2f48sasym(<1 x i64> %ae_arth_v0, <1 x i64> %ae_arth_v1)
; CHECK: ae_round24x2f48sasym aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_round24x2f48ssym.ll
declare <2 x i32> @llvm.xtensa.ae.round24x2f48ssym(<1 x i64>, <1 x i64>)
define <2 x i32> @test_xtensa_ae_round24x2f48ssym(<1 x i64> %ae_arth_v0, <1 x i64> %ae_arth_v1) {
; CHECK-LABEL: test_xtensa_ae_round24x2f48ssym:


%ret = call <2 x i32> @llvm.xtensa.ae.round24x2f48ssym(<1 x i64> %ae_arth_v0, <1 x i64> %ae_arth_v1)
; CHECK: ae_round24x2f48ssym aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_round32x2f48sasym.ll
declare <2 x i32> @llvm.xtensa.ae.round32x2f48sasym(<1 x i64>, <1 x i64>)
define <2 x i32> @test_xtensa_ae_round32x2f48sasym(<1 x i64> %ae_arth_v0, <1 x i64> %ae_arth_v1) {
; CHECK-LABEL: test_xtensa_ae_round32x2f48sasym:


%ret = call <2 x i32> @llvm.xtensa.ae.round32x2f48sasym(<1 x i64> %ae_arth_v0, <1 x i64> %ae_arth_v1)
; CHECK: ae_round32x2f48sasym aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_round32x2f48ssym.ll
declare <2 x i32> @llvm.xtensa.ae.round32x2f48ssym(<1 x i64>, <1 x i64>)
define <2 x i32> @test_xtensa_ae_round32x2f48ssym(<1 x i64> %ae_arth_v0, <1 x i64> %ae_arth_v1) {
; CHECK-LABEL: test_xtensa_ae_round32x2f48ssym:


%ret = call <2 x i32> @llvm.xtensa.ae.round32x2f48ssym(<1 x i64> %ae_arth_v0, <1 x i64> %ae_arth_v1)
; CHECK: ae_round32x2f48ssym aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_round32x2f64sasym.ll
declare <2 x i32> @llvm.xtensa.ae.round32x2f64sasym(<1 x i64>, <1 x i64>)
define <2 x i32> @test_xtensa_ae_round32x2f64sasym(<1 x i64> %ae_arth_v0, <1 x i64> %ae_arth_v1) {
; CHECK-LABEL: test_xtensa_ae_round32x2f64sasym:


%ret = call <2 x i32> @llvm.xtensa.ae.round32x2f64sasym(<1 x i64> %ae_arth_v0, <1 x i64> %ae_arth_v1)
; CHECK: ae_round32x2f64sasym aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_round32x2f64ssym.ll
declare <2 x i32> @llvm.xtensa.ae.round32x2f64ssym(<1 x i64>, <1 x i64>)
define <2 x i32> @test_xtensa_ae_round32x2f64ssym(<1 x i64> %ae_arth_v0, <1 x i64> %ae_arth_v1) {
; CHECK-LABEL: test_xtensa_ae_round32x2f64ssym:


%ret = call <2 x i32> @llvm.xtensa.ae.round32x2f64ssym(<1 x i64> %ae_arth_v0, <1 x i64> %ae_arth_v1)
; CHECK: ae_round32x2f64ssym aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_roundsp16f24asym.ll
declare <2 x i32> @llvm.xtensa.ae.roundsp16f24asym(<2 x i32>)
define <2 x i32> @test_xtensa_ae_roundsp16f24asym(<2 x i32> %ae_arth_v0) {
; CHECK-LABEL: test_xtensa_ae_roundsp16f24asym:

%ret = call <2 x i32> @llvm.xtensa.ae.roundsp16f24asym(<2 x i32> %ae_arth_v0)
; CHECK: ae_roundsp16f24asym aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_roundsp16f24sym.ll
declare <2 x i32> @llvm.xtensa.ae.roundsp16f24sym(<2 x i32>)
define <2 x i32> @test_xtensa_ae_roundsp16f24sym(<2 x i32> %ae_arth_v0) {
; CHECK-LABEL: test_xtensa_ae_roundsp16f24sym:

%ret = call <2 x i32> @llvm.xtensa.ae.roundsp16f24sym(<2 x i32> %ae_arth_v0)
; CHECK: ae_roundsp16f24sym aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_roundsp16q48x2asym.ll
declare <2 x i32> @llvm.xtensa.ae.roundsp16q48x2asym(<1 x i64>, <1 x i64>)
define <2 x i32> @test_xtensa_ae_roundsp16q48x2asym(<1 x i64> %ae_arth_v0, <1 x i64> %ae_arth_v1) {
; CHECK-LABEL: test_xtensa_ae_roundsp16q48x2asym:


%ret = call <2 x i32> @llvm.xtensa.ae.roundsp16q48x2asym(<1 x i64> %ae_arth_v0, <1 x i64> %ae_arth_v1)
; CHECK: ae_roundsp16q48x2asym aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_roundsp16q48x2sym.ll
declare <2 x i32> @llvm.xtensa.ae.roundsp16q48x2sym(<1 x i64>, <1 x i64>)
define <2 x i32> @test_xtensa_ae_roundsp16q48x2sym(<1 x i64> %ae_arth_v0, <1 x i64> %ae_arth_v1) {
; CHECK-LABEL: test_xtensa_ae_roundsp16q48x2sym:


%ret = call <2 x i32> @llvm.xtensa.ae.roundsp16q48x2sym(<1 x i64> %ae_arth_v0, <1 x i64> %ae_arth_v1)
; CHECK: ae_roundsp16q48x2sym aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_roundsq32f48asym.ll
declare <1 x i64> @llvm.xtensa.ae.roundsq32f48asym(<1 x i64>)
define <1 x i64> @test_xtensa_ae_roundsq32f48asym(<1 x i64> %ae_arth_v1) {
; CHECK-LABEL: test_xtensa_ae_roundsq32f48asym:

%ret = call <1 x i64> @llvm.xtensa.ae.roundsq32f48asym(<1 x i64> %ae_arth_v1)
; CHECK: ae_roundsq32f48asym aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_roundsq32f48sym.ll
declare <1 x i64> @llvm.xtensa.ae.roundsq32f48sym(<1 x i64>)
define <1 x i64> @test_xtensa_ae_roundsq32f48sym(<1 x i64> %ae_arth_v1) {
; CHECK-LABEL: test_xtensa_ae_roundsq32f48sym:

%ret = call <1 x i64> @llvm.xtensa.ae.roundsq32f48sym(<1 x i64> %ae_arth_v1)
; CHECK: ae_roundsq32f48sym aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_s16_0_i.ll
declare void @llvm.xtensa.ae.s16.0.i(<4 x i16>, ptr, i32)
define void @test_xtensa_ae_s16_0_i(<4 x i16> %ae_ls_v, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_s16_0_i:



 call void @llvm.xtensa.ae.s16.0.i(<4 x i16> %ae_ls_v, ptr %ars, i32 -16)
; CHECK: ae_s16.0.i aed{{[0-9]+}}, a{{[0-9]+}}, {{-?[0-9]+}}

ret void
}


;--- ae_s16_0_ip.ll
declare ptr @llvm.xtensa.ae.s16.0.ip(<4 x i16>, ptr, i32)
define ptr @test_xtensa_ae_s16_0_ip(<4 x i16> %ae_ls_v, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_s16_0_ip:



%ret = call ptr @llvm.xtensa.ae.s16.0.ip(<4 x i16> %ae_ls_v, ptr %ars, i32 -16)
; CHECK: ae_s16.0.ip aed{{[0-9]+}}, a{{[0-9]+}}, {{-?[0-9]+}}

ret ptr %ret
}


;--- ae_s16_0_x.ll
declare void @llvm.xtensa.ae.s16.0.x(<4 x i16>, ptr, i32)
define void @test_xtensa_ae_s16_0_x(<4 x i16> %ae_ls_v, ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_s16_0_x:



 call void @llvm.xtensa.ae.s16.0.x(<4 x i16> %ae_ls_v, ptr %ars, i32 %art)
; CHECK: ae_s16.0.x aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}

ret void
}


;--- ae_s16_0_xc.ll
declare ptr @llvm.xtensa.ae.s16.0.xc(<4 x i16>, ptr, i32)
define ptr @test_xtensa_ae_s16_0_xc(<4 x i16> %ae_ls_v, ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_s16_0_xc:



%ret = call ptr @llvm.xtensa.ae.s16.0.xc(<4 x i16> %ae_ls_v, ptr %ars, i32 %art)
; CHECK: ae_s16.0.xc aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}

ret ptr %ret
}


;--- ae_s16_0_xp.ll
declare ptr @llvm.xtensa.ae.s16.0.xp(<4 x i16>, ptr, i32)
define ptr @test_xtensa_ae_s16_0_xp(<4 x i16> %ae_ls_v, ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_s16_0_xp:



%ret = call ptr @llvm.xtensa.ae.s16.0.xp(<4 x i16> %ae_ls_v, ptr %ars, i32 %art)
; CHECK: ae_s16.0.xp aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}

ret ptr %ret
}


;--- ae_s16m_l_i.ll
declare void @llvm.xtensa.ae.s16m.l.i(<2 x i32>, ptr, i32)
define void @test_xtensa_ae_s16m_l_i(<2 x i32> %ae_ls_v, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_s16m_l_i:



 call void @llvm.xtensa.ae.s16m.l.i(<2 x i32> %ae_ls_v, ptr %ars, i32 -16)
; CHECK: ae_s16m.l.i aed{{[0-9]+}}, a{{[0-9]+}}, {{-?[0-9]+}}

ret void
}


;--- ae_s16m_l_iu.ll
declare ptr @llvm.xtensa.ae.s16m.l.iu(<2 x i32>, ptr, i32)
define ptr @test_xtensa_ae_s16m_l_iu(<2 x i32> %ae_ls_v, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_s16m_l_iu:



%ret = call ptr @llvm.xtensa.ae.s16m.l.iu(<2 x i32> %ae_ls_v, ptr %ars, i32 -16)
; CHECK: ae_s16m.l.iu aed{{[0-9]+}}, a{{[0-9]+}}, {{-?[0-9]+}}

ret ptr %ret
}


;--- ae_s16m_l_x.ll
declare void @llvm.xtensa.ae.s16m.l.x(<2 x i32>, ptr, i32)
define void @test_xtensa_ae_s16m_l_x(<2 x i32> %ae_ls_v, ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_s16m_l_x:



 call void @llvm.xtensa.ae.s16m.l.x(<2 x i32> %ae_ls_v, ptr %ars, i32 %art)
; CHECK: ae_s16m.l.x aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}

ret void
}


;--- ae_s16m_l_xc.ll
declare ptr @llvm.xtensa.ae.s16m.l.xc(<2 x i32>, ptr, i32)
define ptr @test_xtensa_ae_s16m_l_xc(<2 x i32> %ae_ls_v, ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_s16m_l_xc:



%ret = call ptr @llvm.xtensa.ae.s16m.l.xc(<2 x i32> %ae_ls_v, ptr %ars, i32 %art)
; CHECK: ae_s16m.l.xc aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}

ret ptr %ret
}


;--- ae_s16m_l_xu.ll
declare ptr @llvm.xtensa.ae.s16m.l.xu(<2 x i32>, ptr, i32)
define ptr @test_xtensa_ae_s16m_l_xu(<2 x i32> %ae_ls_v, ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_s16m_l_xu:



%ret = call ptr @llvm.xtensa.ae.s16m.l.xu(<2 x i32> %ae_ls_v, ptr %ars, i32 %art)
; CHECK: ae_s16m.l.xu aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}

ret ptr %ret
}


;--- ae_s16x2m_i.ll
declare void @llvm.xtensa.ae.s16x2m.i(<2 x i32>, ptr, i32)
define void @test_xtensa_ae_s16x2m_i(<2 x i32> %ae_ls_v, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_s16x2m_i:



 call void @llvm.xtensa.ae.s16x2m.i(<2 x i32> %ae_ls_v, ptr %ars, i32 -32)
; CHECK: ae_s16x2m.i aed{{[0-9]+}}, a{{[0-9]+}}, {{-?[0-9]+}}

ret void
}


;--- ae_s16x2m_iu.ll
declare ptr @llvm.xtensa.ae.s16x2m.iu(<2 x i32>, ptr, i32)
define ptr @test_xtensa_ae_s16x2m_iu(<2 x i32> %ae_ls_v, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_s16x2m_iu:



%ret = call ptr @llvm.xtensa.ae.s16x2m.iu(<2 x i32> %ae_ls_v, ptr %ars, i32 -32)
; CHECK: ae_s16x2m.iu aed{{[0-9]+}}, a{{[0-9]+}}, {{-?[0-9]+}}

ret ptr %ret
}


;--- ae_s16x2m_x.ll
declare void @llvm.xtensa.ae.s16x2m.x(<2 x i32>, ptr, i32)
define void @test_xtensa_ae_s16x2m_x(<2 x i32> %ae_ls_v, ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_s16x2m_x:



 call void @llvm.xtensa.ae.s16x2m.x(<2 x i32> %ae_ls_v, ptr %ars, i32 %art)
; CHECK: ae_s16x2m.x aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}

ret void
}


;--- ae_s16x2m_xc.ll
declare ptr @llvm.xtensa.ae.s16x2m.xc(<2 x i32>, ptr, i32)
define ptr @test_xtensa_ae_s16x2m_xc(<2 x i32> %ae_ls_v, ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_s16x2m_xc:



%ret = call ptr @llvm.xtensa.ae.s16x2m.xc(<2 x i32> %ae_ls_v, ptr %ars, i32 %art)
; CHECK: ae_s16x2m.xc aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}

ret ptr %ret
}


;--- ae_s16x2m_xu.ll
declare ptr @llvm.xtensa.ae.s16x2m.xu(<2 x i32>, ptr, i32)
define ptr @test_xtensa_ae_s16x2m_xu(<2 x i32> %ae_ls_v, ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_s16x2m_xu:



%ret = call ptr @llvm.xtensa.ae.s16x2m.xu(<2 x i32> %ae_ls_v, ptr %ars, i32 %art)
; CHECK: ae_s16x2m.xu aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}

ret ptr %ret
}


;--- ae_s16x4_i.ll
declare void @llvm.xtensa.ae.s16x4.i(<4 x i16>, ptr, i32)
define void @test_xtensa_ae_s16x4_i(<4 x i16> %ae_ls_v, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_s16x4_i:



 call void @llvm.xtensa.ae.s16x4.i(<4 x i16> %ae_ls_v, ptr %ars, i32 -64)
; CHECK: ae_s16x4.i aed{{[0-9]+}}, a{{[0-9]+}}, {{-?[0-9]+}}

ret void
}


;--- ae_s16x4_ip.ll
declare ptr @llvm.xtensa.ae.s16x4.ip(<4 x i16>, ptr, i32)
define ptr @test_xtensa_ae_s16x4_ip(<4 x i16> %ae_ls_v, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_s16x4_ip:



%ret = call ptr @llvm.xtensa.ae.s16x4.ip(<4 x i16> %ae_ls_v, ptr %ars, i32 0)
; CHECK: ae_s16x4.ip aed{{[0-9]+}}, a{{[0-9]+}}, {{-?[0-9]+}}

ret ptr %ret
}


;--- ae_s16x4_ric.ll
declare ptr @llvm.xtensa.ae.s16x4.ric(<4 x i16>, ptr)
define ptr @test_xtensa_ae_s16x4_ric(<4 x i16> %ae_ls_v, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_s16x4_ric:


%ret = call ptr @llvm.xtensa.ae.s16x4.ric(<4 x i16> %ae_ls_v, ptr %ars)
; CHECK: ae_s16x4.ric aed{{[0-9]+}}, a{{[0-9]+}}

ret ptr %ret
}


;--- ae_s16x4_rip.ll
declare ptr @llvm.xtensa.ae.s16x4.rip(<4 x i16>, ptr)
define ptr @test_xtensa_ae_s16x4_rip(<4 x i16> %ae_ls_v, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_s16x4_rip:


%ret = call ptr @llvm.xtensa.ae.s16x4.rip(<4 x i16> %ae_ls_v, ptr %ars)
; CHECK: ae_s16x4.rip aed{{[0-9]+}}, a{{[0-9]+}}

ret ptr %ret
}


;--- ae_s16x4_x.ll
declare void @llvm.xtensa.ae.s16x4.x(<4 x i16>, ptr, i32)
define void @test_xtensa_ae_s16x4_x(<4 x i16> %ae_ls_v, ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_s16x4_x:



 call void @llvm.xtensa.ae.s16x4.x(<4 x i16> %ae_ls_v, ptr %ars, i32 %art)
; CHECK: ae_s16x4.x aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}

ret void
}


;--- ae_s16x4_xc.ll
declare ptr @llvm.xtensa.ae.s16x4.xc(<4 x i16>, ptr, i32)
define ptr @test_xtensa_ae_s16x4_xc(<4 x i16> %ae_ls_v, ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_s16x4_xc:



%ret = call ptr @llvm.xtensa.ae.s16x4.xc(<4 x i16> %ae_ls_v, ptr %ars, i32 %art)
; CHECK: ae_s16x4.xc aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}

ret ptr %ret
}


;--- ae_s16x4_xp.ll
declare ptr @llvm.xtensa.ae.s16x4.xp(<4 x i16>, ptr, i32)
define ptr @test_xtensa_ae_s16x4_xp(<4 x i16> %ae_ls_v, ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_s16x4_xp:



%ret = call ptr @llvm.xtensa.ae.s16x4.xp(<4 x i16> %ae_ls_v, ptr %ars, i32 %art)
; CHECK: ae_s16x4.xp aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}

ret ptr %ret
}


;--- ae_s24ra64s_i.ll
declare void @llvm.xtensa.ae.s24ra64s.i(<1 x i64>, ptr, i32)
define void @test_xtensa_ae_s24ra64s_i(<1 x i64> %ae_ls_v1, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_s24ra64s_i:



 call void @llvm.xtensa.ae.s24ra64s.i(<1 x i64> %ae_ls_v1, ptr %ars, i32 -32)
; CHECK: ae_s24ra64s.i aed{{[0-9]+}}, a{{[0-9]+}}, {{-?[0-9]+}}

ret void
}


;--- ae_s24ra64s_ip.ll
declare ptr @llvm.xtensa.ae.s24ra64s.ip(<1 x i64>, ptr, i32)
define ptr @test_xtensa_ae_s24ra64s_ip(<1 x i64> %ae_ls_v1, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_s24ra64s_ip:



%ret = call ptr @llvm.xtensa.ae.s24ra64s.ip(<1 x i64> %ae_ls_v1, ptr %ars, i32 -32)
; CHECK: ae_s24ra64s.ip aed{{[0-9]+}}, a{{[0-9]+}}, {{-?[0-9]+}}

ret ptr %ret
}


;--- ae_s24ra64s_x.ll
declare void @llvm.xtensa.ae.s24ra64s.x(<1 x i64>, ptr, i32)
define void @test_xtensa_ae_s24ra64s_x(<1 x i64> %ae_ls_v1, ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_s24ra64s_x:



 call void @llvm.xtensa.ae.s24ra64s.x(<1 x i64> %ae_ls_v1, ptr %ars, i32 %art)
; CHECK: ae_s24ra64s.x aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}

ret void
}


;--- ae_s24ra64s_xc.ll
declare ptr @llvm.xtensa.ae.s24ra64s.xc(<1 x i64>, ptr, i32)
define ptr @test_xtensa_ae_s24ra64s_xc(<1 x i64> %ae_ls_v1, ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_s24ra64s_xc:



%ret = call ptr @llvm.xtensa.ae.s24ra64s.xc(<1 x i64> %ae_ls_v1, ptr %ars, i32 %art)
; CHECK: ae_s24ra64s.xc aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}

ret ptr %ret
}


;--- ae_s24ra64s_xp.ll
declare ptr @llvm.xtensa.ae.s24ra64s.xp(<1 x i64>, ptr, i32)
define ptr @test_xtensa_ae_s24ra64s_xp(<1 x i64> %ae_ls_v1, ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_s24ra64s_xp:



%ret = call ptr @llvm.xtensa.ae.s24ra64s.xp(<1 x i64> %ae_ls_v1, ptr %ars, i32 %art)
; CHECK: ae_s24ra64s.xp aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}

ret ptr %ret
}


;--- ae_s24x2ra64s_ip.ll
declare ptr @llvm.xtensa.ae.s24x2ra64s.ip(<1 x i64>, <1 x i64>, ptr)
define ptr @test_xtensa_ae_s24x2ra64s_ip(<1 x i64> %ae_ls_v2, <1 x i64> %ae_ls_v1, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_s24x2ra64s_ip:



%ret = call ptr @llvm.xtensa.ae.s24x2ra64s.ip(<1 x i64> %ae_ls_v2, <1 x i64> %ae_ls_v1, ptr %ars)
; CHECK: ae_s24x2ra64s.ip aed{{[0-9]+}}, aed{{[0-9]+}}, a{{[0-9]+}}

ret ptr %ret
}


;--- ae_s32_l_i.ll
declare void @llvm.xtensa.ae.s32.l.i(<2 x i32>, ptr, i32)
define void @test_xtensa_ae_s32_l_i(<2 x i32> %ae_ls_v, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_s32_l_i:



 call void @llvm.xtensa.ae.s32.l.i(<2 x i32> %ae_ls_v, ptr %ars, i32 -32)
; CHECK: ae_s32.l.i aed{{[0-9]+}}, a{{[0-9]+}}, {{-?[0-9]+}}

ret void
}


;--- ae_s32_l_ip.ll
declare ptr @llvm.xtensa.ae.s32.l.ip(<2 x i32>, ptr, i32)
define ptr @test_xtensa_ae_s32_l_ip(<2 x i32> %ae_ls_v, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_s32_l_ip:



%ret = call ptr @llvm.xtensa.ae.s32.l.ip(<2 x i32> %ae_ls_v, ptr %ars, i32 -32)
; CHECK: ae_s32.l.ip aed{{[0-9]+}}, a{{[0-9]+}}, {{-?[0-9]+}}

ret ptr %ret
}


;--- ae_s32_l_x.ll
declare void @llvm.xtensa.ae.s32.l.x(<2 x i32>, ptr, i32)
define void @test_xtensa_ae_s32_l_x(<2 x i32> %ae_ls_v, ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_s32_l_x:



 call void @llvm.xtensa.ae.s32.l.x(<2 x i32> %ae_ls_v, ptr %ars, i32 %art)
; CHECK: ae_s32.l.x aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}

ret void
}


;--- ae_s32_l_xc.ll
declare ptr @llvm.xtensa.ae.s32.l.xc(<2 x i32>, ptr, i32)
define ptr @test_xtensa_ae_s32_l_xc(<2 x i32> %ae_ls_v, ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_s32_l_xc:



%ret = call ptr @llvm.xtensa.ae.s32.l.xc(<2 x i32> %ae_ls_v, ptr %ars, i32 %art)
; CHECK: ae_s32.l.xc aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}

ret ptr %ret
}


;--- ae_s32_l_xp.ll
declare ptr @llvm.xtensa.ae.s32.l.xp(<2 x i32>, ptr, i32)
define ptr @test_xtensa_ae_s32_l_xp(<2 x i32> %ae_ls_v, ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_s32_l_xp:



%ret = call ptr @llvm.xtensa.ae.s32.l.xp(<2 x i32> %ae_ls_v, ptr %ars, i32 %art)
; CHECK: ae_s32.l.xp aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}

ret ptr %ret
}


;--- ae_s32f24_l_i.ll
declare void @llvm.xtensa.ae.s32f24.l.i(<2 x i32>, ptr, i32)
define void @test_xtensa_ae_s32f24_l_i(<2 x i32> %ae_ls_v, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_s32f24_l_i:



 call void @llvm.xtensa.ae.s32f24.l.i(<2 x i32> %ae_ls_v, ptr %ars, i32 -32)
; CHECK: ae_s32f24.l.i aed{{[0-9]+}}, a{{[0-9]+}}, {{-?[0-9]+}}

ret void
}


;--- ae_s32f24_l_ip.ll
declare ptr @llvm.xtensa.ae.s32f24.l.ip(<2 x i32>, ptr, i32)
define ptr @test_xtensa_ae_s32f24_l_ip(<2 x i32> %ae_ls_v, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_s32f24_l_ip:



%ret = call ptr @llvm.xtensa.ae.s32f24.l.ip(<2 x i32> %ae_ls_v, ptr %ars, i32 -32)
; CHECK: ae_s32f24.l.ip aed{{[0-9]+}}, a{{[0-9]+}}, {{-?[0-9]+}}

ret ptr %ret
}


;--- ae_s32f24_l_x.ll
declare void @llvm.xtensa.ae.s32f24.l.x(<2 x i32>, ptr, i32)
define void @test_xtensa_ae_s32f24_l_x(<2 x i32> %ae_ls_v, ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_s32f24_l_x:



 call void @llvm.xtensa.ae.s32f24.l.x(<2 x i32> %ae_ls_v, ptr %ars, i32 %art)
; CHECK: ae_s32f24.l.x aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}

ret void
}


;--- ae_s32f24_l_xc.ll
declare ptr @llvm.xtensa.ae.s32f24.l.xc(<2 x i32>, ptr, i32)
define ptr @test_xtensa_ae_s32f24_l_xc(<2 x i32> %ae_ls_v, ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_s32f24_l_xc:



%ret = call ptr @llvm.xtensa.ae.s32f24.l.xc(<2 x i32> %ae_ls_v, ptr %ars, i32 %art)
; CHECK: ae_s32f24.l.xc aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}

ret ptr %ret
}


;--- ae_s32f24_l_xp.ll
declare ptr @llvm.xtensa.ae.s32f24.l.xp(<2 x i32>, ptr, i32)
define ptr @test_xtensa_ae_s32f24_l_xp(<2 x i32> %ae_ls_v, ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_s32f24_l_xp:



%ret = call ptr @llvm.xtensa.ae.s32f24.l.xp(<2 x i32> %ae_ls_v, ptr %ars, i32 %art)
; CHECK: ae_s32f24.l.xp aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}

ret ptr %ret
}


;--- ae_s32m_i.ll
declare void @llvm.xtensa.ae.s32m.i(<1 x i64>, ptr, i32)
define void @test_xtensa_ae_s32m_i(<1 x i64> %ae_ls_v, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_s32m_i:



 call void @llvm.xtensa.ae.s32m.i(<1 x i64> %ae_ls_v, ptr %ars, i32 -32)
; CHECK: ae_s32m.i aed{{[0-9]+}}, a{{[0-9]+}}, {{-?[0-9]+}}

ret void
}


;--- ae_s32m_iu.ll
declare ptr @llvm.xtensa.ae.s32m.iu(<1 x i64>, ptr, i32)
define ptr @test_xtensa_ae_s32m_iu(<1 x i64> %ae_ls_v, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_s32m_iu:



%ret = call ptr @llvm.xtensa.ae.s32m.iu(<1 x i64> %ae_ls_v, ptr %ars, i32 -32)
; CHECK: ae_s32m.iu aed{{[0-9]+}}, a{{[0-9]+}}, {{-?[0-9]+}}

ret ptr %ret
}


;--- ae_s32m_x.ll
declare void @llvm.xtensa.ae.s32m.x(<1 x i64>, ptr, i32)
define void @test_xtensa_ae_s32m_x(<1 x i64> %ae_ls_v, ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_s32m_x:



 call void @llvm.xtensa.ae.s32m.x(<1 x i64> %ae_ls_v, ptr %ars, i32 %art)
; CHECK: ae_s32m.x aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}

ret void
}


;--- ae_s32m_xc.ll
declare ptr @llvm.xtensa.ae.s32m.xc(<1 x i64>, ptr, i32)
define ptr @test_xtensa_ae_s32m_xc(<1 x i64> %ae_ls_v, ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_s32m_xc:



%ret = call ptr @llvm.xtensa.ae.s32m.xc(<1 x i64> %ae_ls_v, ptr %ars, i32 %art)
; CHECK: ae_s32m.xc aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}

ret ptr %ret
}


;--- ae_s32m_xu.ll
declare ptr @llvm.xtensa.ae.s32m.xu(<1 x i64>, ptr, i32)
define ptr @test_xtensa_ae_s32m_xu(<1 x i64> %ae_ls_v, ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_s32m_xu:



%ret = call ptr @llvm.xtensa.ae.s32m.xu(<1 x i64> %ae_ls_v, ptr %ars, i32 %art)
; CHECK: ae_s32m.xu aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}

ret ptr %ret
}


;--- ae_s32ra64s_i.ll
declare void @llvm.xtensa.ae.s32ra64s.i(<1 x i64>, ptr, i32)
define void @test_xtensa_ae_s32ra64s_i(<1 x i64> %ae_ls_v1, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_s32ra64s_i:



 call void @llvm.xtensa.ae.s32ra64s.i(<1 x i64> %ae_ls_v1, ptr %ars, i32 -32)
; CHECK: ae_s32ra64s.i aed{{[0-9]+}}, a{{[0-9]+}}, {{-?[0-9]+}}

ret void
}


;--- ae_s32ra64s_ip.ll
declare ptr @llvm.xtensa.ae.s32ra64s.ip(<1 x i64>, ptr, i32)
define ptr @test_xtensa_ae_s32ra64s_ip(<1 x i64> %ae_ls_v1, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_s32ra64s_ip:



%ret = call ptr @llvm.xtensa.ae.s32ra64s.ip(<1 x i64> %ae_ls_v1, ptr %ars, i32 -32)
; CHECK: ae_s32ra64s.ip aed{{[0-9]+}}, a{{[0-9]+}}, {{-?[0-9]+}}

ret ptr %ret
}


;--- ae_s32ra64s_x.ll
declare void @llvm.xtensa.ae.s32ra64s.x(<1 x i64>, ptr, i32)
define void @test_xtensa_ae_s32ra64s_x(<1 x i64> %ae_ls_v1, ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_s32ra64s_x:



 call void @llvm.xtensa.ae.s32ra64s.x(<1 x i64> %ae_ls_v1, ptr %ars, i32 %art)
; CHECK: ae_s32ra64s.x aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}

ret void
}


;--- ae_s32ra64s_xc.ll
declare ptr @llvm.xtensa.ae.s32ra64s.xc(<1 x i64>, ptr, i32)
define ptr @test_xtensa_ae_s32ra64s_xc(<1 x i64> %ae_ls_v1, ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_s32ra64s_xc:



%ret = call ptr @llvm.xtensa.ae.s32ra64s.xc(<1 x i64> %ae_ls_v1, ptr %ars, i32 %art)
; CHECK: ae_s32ra64s.xc aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}

ret ptr %ret
}


;--- ae_s32ra64s_xp.ll
declare ptr @llvm.xtensa.ae.s32ra64s.xp(<1 x i64>, ptr, i32)
define ptr @test_xtensa_ae_s32ra64s_xp(<1 x i64> %ae_ls_v1, ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_s32ra64s_xp:



%ret = call ptr @llvm.xtensa.ae.s32ra64s.xp(<1 x i64> %ae_ls_v1, ptr %ars, i32 %art)
; CHECK: ae_s32ra64s.xp aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}

ret ptr %ret
}


;--- ae_s32x2_i.ll
declare void @llvm.xtensa.ae.s32x2.i(<2 x i32>, ptr, i32)
define void @test_xtensa_ae_s32x2_i(<2 x i32> %ae_ls_v, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_s32x2_i:



 call void @llvm.xtensa.ae.s32x2.i(<2 x i32> %ae_ls_v, ptr %ars, i32 -64)
; CHECK: ae_s32x2.i aed{{[0-9]+}}, a{{[0-9]+}}, {{-?[0-9]+}}

ret void
}


;--- ae_s32x2_ip.ll
declare ptr @llvm.xtensa.ae.s32x2.ip(<2 x i32>, ptr, i32)
define ptr @test_xtensa_ae_s32x2_ip(<2 x i32> %ae_ls_v, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_s32x2_ip:



%ret = call ptr @llvm.xtensa.ae.s32x2.ip(<2 x i32> %ae_ls_v, ptr %ars, i32 0)
; CHECK: ae_s32x2.ip aed{{[0-9]+}}, a{{[0-9]+}}, {{-?[0-9]+}}

ret ptr %ret
}


;--- ae_s32x2_ric.ll
declare ptr @llvm.xtensa.ae.s32x2.ric(<2 x i32>, ptr)
define ptr @test_xtensa_ae_s32x2_ric(<2 x i32> %ae_ls_v, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_s32x2_ric:


%ret = call ptr @llvm.xtensa.ae.s32x2.ric(<2 x i32> %ae_ls_v, ptr %ars)
; CHECK: ae_s32x2.ric aed{{[0-9]+}}, a{{[0-9]+}}

ret ptr %ret
}


;--- ae_s32x2_rip.ll
declare ptr @llvm.xtensa.ae.s32x2.rip(<2 x i32>, ptr)
define ptr @test_xtensa_ae_s32x2_rip(<2 x i32> %ae_ls_v, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_s32x2_rip:


%ret = call ptr @llvm.xtensa.ae.s32x2.rip(<2 x i32> %ae_ls_v, ptr %ars)
; CHECK: ae_s32x2.rip aed{{[0-9]+}}, a{{[0-9]+}}

ret ptr %ret
}


;--- ae_s32x2_x.ll
declare void @llvm.xtensa.ae.s32x2.x(<2 x i32>, ptr, i32)
define void @test_xtensa_ae_s32x2_x(<2 x i32> %ae_ls_v, ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_s32x2_x:



 call void @llvm.xtensa.ae.s32x2.x(<2 x i32> %ae_ls_v, ptr %ars, i32 %art)
; CHECK: ae_s32x2.x aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}

ret void
}


;--- ae_s32x2_xc.ll
declare ptr @llvm.xtensa.ae.s32x2.xc(<2 x i32>, ptr, i32)
define ptr @test_xtensa_ae_s32x2_xc(<2 x i32> %ae_ls_v, ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_s32x2_xc:



%ret = call ptr @llvm.xtensa.ae.s32x2.xc(<2 x i32> %ae_ls_v, ptr %ars, i32 %art)
; CHECK: ae_s32x2.xc aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}

ret ptr %ret
}


;--- ae_s32x2_xp.ll
declare ptr @llvm.xtensa.ae.s32x2.xp(<2 x i32>, ptr, i32)
define ptr @test_xtensa_ae_s32x2_xp(<2 x i32> %ae_ls_v, ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_s32x2_xp:



%ret = call ptr @llvm.xtensa.ae.s32x2.xp(<2 x i32> %ae_ls_v, ptr %ars, i32 %art)
; CHECK: ae_s32x2.xp aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}

ret ptr %ret
}


;--- ae_s32x2f24_i.ll
declare void @llvm.xtensa.ae.s32x2f24.i(<2 x i32>, ptr, i32)
define void @test_xtensa_ae_s32x2f24_i(<2 x i32> %ae_ls_v, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_s32x2f24_i:



 call void @llvm.xtensa.ae.s32x2f24.i(<2 x i32> %ae_ls_v, ptr %ars, i32 -64)
; CHECK: ae_s32x2f24.i aed{{[0-9]+}}, a{{[0-9]+}}, {{-?[0-9]+}}

ret void
}


;--- ae_s32x2f24_ip.ll
declare ptr @llvm.xtensa.ae.s32x2f24.ip(<2 x i32>, ptr, i32)
define ptr @test_xtensa_ae_s32x2f24_ip(<2 x i32> %ae_ls_v, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_s32x2f24_ip:



%ret = call ptr @llvm.xtensa.ae.s32x2f24.ip(<2 x i32> %ae_ls_v, ptr %ars, i32 0)
; CHECK: ae_s32x2f24.ip aed{{[0-9]+}}, a{{[0-9]+}}, {{-?[0-9]+}}

ret ptr %ret
}


;--- ae_s32x2f24_ric.ll
declare ptr @llvm.xtensa.ae.s32x2f24.ric(<2 x i32>, ptr)
define ptr @test_xtensa_ae_s32x2f24_ric(<2 x i32> %ae_ls_v, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_s32x2f24_ric:


%ret = call ptr @llvm.xtensa.ae.s32x2f24.ric(<2 x i32> %ae_ls_v, ptr %ars)
; CHECK: ae_s32x2f24.ric aed{{[0-9]+}}, a{{[0-9]+}}

ret ptr %ret
}


;--- ae_s32x2f24_rip.ll
declare ptr @llvm.xtensa.ae.s32x2f24.rip(<2 x i32>, ptr)
define ptr @test_xtensa_ae_s32x2f24_rip(<2 x i32> %ae_ls_v, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_s32x2f24_rip:


%ret = call ptr @llvm.xtensa.ae.s32x2f24.rip(<2 x i32> %ae_ls_v, ptr %ars)
; CHECK: ae_s32x2f24.rip aed{{[0-9]+}}, a{{[0-9]+}}

ret ptr %ret
}


;--- ae_s32x2f24_x.ll
declare void @llvm.xtensa.ae.s32x2f24.x(<2 x i32>, ptr, i32)
define void @test_xtensa_ae_s32x2f24_x(<2 x i32> %ae_ls_v, ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_s32x2f24_x:



 call void @llvm.xtensa.ae.s32x2f24.x(<2 x i32> %ae_ls_v, ptr %ars, i32 %art)
; CHECK: ae_s32x2f24.x aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}

ret void
}


;--- ae_s32x2f24_xc.ll
declare ptr @llvm.xtensa.ae.s32x2f24.xc(<2 x i32>, ptr, i32)
define ptr @test_xtensa_ae_s32x2f24_xc(<2 x i32> %ae_ls_v, ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_s32x2f24_xc:



%ret = call ptr @llvm.xtensa.ae.s32x2f24.xc(<2 x i32> %ae_ls_v, ptr %ars, i32 %art)
; CHECK: ae_s32x2f24.xc aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}

ret ptr %ret
}


;--- ae_s32x2f24_xp.ll
declare ptr @llvm.xtensa.ae.s32x2f24.xp(<2 x i32>, ptr, i32)
define ptr @test_xtensa_ae_s32x2f24_xp(<2 x i32> %ae_ls_v, ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_s32x2f24_xp:



%ret = call ptr @llvm.xtensa.ae.s32x2f24.xp(<2 x i32> %ae_ls_v, ptr %ars, i32 %art)
; CHECK: ae_s32x2f24.xp aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}

ret ptr %ret
}


;--- ae_s32x2ra64s_ip.ll
declare ptr @llvm.xtensa.ae.s32x2ra64s.ip(<1 x i64>, <1 x i64>, ptr)
define ptr @test_xtensa_ae_s32x2ra64s_ip(<1 x i64> %ae_ls_v2, <1 x i64> %ae_ls_v1, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_s32x2ra64s_ip:



%ret = call ptr @llvm.xtensa.ae.s32x2ra64s.ip(<1 x i64> %ae_ls_v2, <1 x i64> %ae_ls_v1, ptr %ars)
; CHECK: ae_s32x2ra64s.ip aed{{[0-9]+}}, aed{{[0-9]+}}, a{{[0-9]+}}

ret ptr %ret
}


;--- ae_s64_i.ll
declare void @llvm.xtensa.ae.s64.i(<1 x i64>, ptr, i32)
define void @test_xtensa_ae_s64_i(<1 x i64> %ae_ls_v, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_s64_i:



 call void @llvm.xtensa.ae.s64.i(<1 x i64> %ae_ls_v, ptr %ars, i32 -64)
; CHECK: ae_s64.i aed{{[0-9]+}}, a{{[0-9]+}}, {{-?[0-9]+}}

ret void
}


;--- ae_s64_ip.ll
declare ptr @llvm.xtensa.ae.s64.ip(<1 x i64>, ptr, i32)
define ptr @test_xtensa_ae_s64_ip(<1 x i64> %ae_ls_v, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_s64_ip:



%ret = call ptr @llvm.xtensa.ae.s64.ip(<1 x i64> %ae_ls_v, ptr %ars, i32 -64)
; CHECK: ae_s64.ip aed{{[0-9]+}}, a{{[0-9]+}}, {{-?[0-9]+}}

ret ptr %ret
}


;--- ae_s64_x.ll
declare void @llvm.xtensa.ae.s64.x(<1 x i64>, ptr, i32)
define void @test_xtensa_ae_s64_x(<1 x i64> %ae_ls_v, ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_s64_x:



 call void @llvm.xtensa.ae.s64.x(<1 x i64> %ae_ls_v, ptr %ars, i32 %art)
; CHECK: ae_s64.x aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}

ret void
}


;--- ae_s64_xc.ll
declare ptr @llvm.xtensa.ae.s64.xc(<1 x i64>, ptr, i32)
define ptr @test_xtensa_ae_s64_xc(<1 x i64> %ae_ls_v, ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_s64_xc:



%ret = call ptr @llvm.xtensa.ae.s64.xc(<1 x i64> %ae_ls_v, ptr %ars, i32 %art)
; CHECK: ae_s64.xc aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}

ret ptr %ret
}


;--- ae_s64_xp.ll
declare ptr @llvm.xtensa.ae.s64.xp(<1 x i64>, ptr, i32)
define ptr @test_xtensa_ae_s64_xp(<1 x i64> %ae_ls_v, ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_s64_xp:



%ret = call ptr @llvm.xtensa.ae.s64.xp(<1 x i64> %ae_ls_v, ptr %ars, i32 %art)
; CHECK: ae_s64.xp aed{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}

ret ptr %ret
}


;--- ae_sa16x4_ic.ll
declare { <8 x i8>, ptr } @llvm.xtensa.ae.sa16x4.ic(<4 x i16>, <8 x i8>, ptr)
define <1 x i64> @test_xtensa_ae_sa16x4_ic(<4 x i16> %ae_ls_v, <1 x i64> %ae_ls_su, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_sa16x4_ic:

%wrap1 = bitcast <1 x i64> %ae_ls_su to <8 x i8>

%ret = call { <8 x i8>, ptr } @llvm.xtensa.ae.sa16x4.ic(<4 x i16> %ae_ls_v, <8 x i8> %wrap1, ptr %ars)
; CHECK: ae_sa16x4.ic aed{{[0-9]+}}, u{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <8 x i8>, ptr } %ret, 0
%unwrap = bitcast <8 x i8> %ev to <1 x i64>
ret <1 x i64> %unwrap
}


;--- ae_sa16x4_ip.ll
declare { <8 x i8>, ptr } @llvm.xtensa.ae.sa16x4.ip(<4 x i16>, <8 x i8>, ptr)
define <1 x i64> @test_xtensa_ae_sa16x4_ip(<4 x i16> %ae_ls_v, <1 x i64> %ae_ls_su, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_sa16x4_ip:

%wrap1 = bitcast <1 x i64> %ae_ls_su to <8 x i8>

%ret = call { <8 x i8>, ptr } @llvm.xtensa.ae.sa16x4.ip(<4 x i16> %ae_ls_v, <8 x i8> %wrap1, ptr %ars)
; CHECK: ae_sa16x4.ip aed{{[0-9]+}}, u{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <8 x i8>, ptr } %ret, 0
%unwrap = bitcast <8 x i8> %ev to <1 x i64>
ret <1 x i64> %unwrap
}


;--- ae_sa16x4_ric.ll
declare { <8 x i8>, ptr } @llvm.xtensa.ae.sa16x4.ric(<4 x i16>, <8 x i8>, ptr)
define <1 x i64> @test_xtensa_ae_sa16x4_ric(<4 x i16> %ae_ls_v, <1 x i64> %ae_ls_su, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_sa16x4_ric:

%wrap1 = bitcast <1 x i64> %ae_ls_su to <8 x i8>

%ret = call { <8 x i8>, ptr } @llvm.xtensa.ae.sa16x4.ric(<4 x i16> %ae_ls_v, <8 x i8> %wrap1, ptr %ars)
; CHECK: ae_sa16x4.ric aed{{[0-9]+}}, u{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <8 x i8>, ptr } %ret, 0
%unwrap = bitcast <8 x i8> %ev to <1 x i64>
ret <1 x i64> %unwrap
}


;--- ae_sa16x4_rip.ll
declare { <8 x i8>, ptr } @llvm.xtensa.ae.sa16x4.rip(<4 x i16>, <8 x i8>, ptr)
define <1 x i64> @test_xtensa_ae_sa16x4_rip(<4 x i16> %ae_ls_v, <1 x i64> %ae_ls_su, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_sa16x4_rip:

%wrap1 = bitcast <1 x i64> %ae_ls_su to <8 x i8>

%ret = call { <8 x i8>, ptr } @llvm.xtensa.ae.sa16x4.rip(<4 x i16> %ae_ls_v, <8 x i8> %wrap1, ptr %ars)
; CHECK: ae_sa16x4.rip aed{{[0-9]+}}, u{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <8 x i8>, ptr } %ret, 0
%unwrap = bitcast <8 x i8> %ev to <1 x i64>
ret <1 x i64> %unwrap
}


;--- ae_sa24_l_ic.ll
declare { <8 x i8>, ptr } @llvm.xtensa.ae.sa24.l.ic(<2 x i32>, <8 x i8>, ptr)
define <1 x i64> @test_xtensa_ae_sa24_l_ic(<2 x i32> %ae_ls_v, <1 x i64> %ae_ls_su, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_sa24_l_ic:

%wrap1 = bitcast <1 x i64> %ae_ls_su to <8 x i8>

%ret = call { <8 x i8>, ptr } @llvm.xtensa.ae.sa24.l.ic(<2 x i32> %ae_ls_v, <8 x i8> %wrap1, ptr %ars)
; CHECK: ae_sa24.l.ic aed{{[0-9]+}}, u{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <8 x i8>, ptr } %ret, 0
%unwrap = bitcast <8 x i8> %ev to <1 x i64>
ret <1 x i64> %unwrap
}


;--- ae_sa24_l_ip.ll
declare { <8 x i8>, ptr } @llvm.xtensa.ae.sa24.l.ip(<2 x i32>, <8 x i8>, ptr)
define <1 x i64> @test_xtensa_ae_sa24_l_ip(<2 x i32> %ae_ls_v, <1 x i64> %ae_ls_su, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_sa24_l_ip:

%wrap1 = bitcast <1 x i64> %ae_ls_su to <8 x i8>

%ret = call { <8 x i8>, ptr } @llvm.xtensa.ae.sa24.l.ip(<2 x i32> %ae_ls_v, <8 x i8> %wrap1, ptr %ars)
; CHECK: ae_sa24.l.ip aed{{[0-9]+}}, u{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <8 x i8>, ptr } %ret, 0
%unwrap = bitcast <8 x i8> %ev to <1 x i64>
ret <1 x i64> %unwrap
}


;--- ae_sa24_l_ric.ll
declare { <8 x i8>, ptr } @llvm.xtensa.ae.sa24.l.ric(<2 x i32>, <8 x i8>, ptr)
define <1 x i64> @test_xtensa_ae_sa24_l_ric(<2 x i32> %ae_ls_v, <1 x i64> %ae_ls_su, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_sa24_l_ric:

%wrap1 = bitcast <1 x i64> %ae_ls_su to <8 x i8>

%ret = call { <8 x i8>, ptr } @llvm.xtensa.ae.sa24.l.ric(<2 x i32> %ae_ls_v, <8 x i8> %wrap1, ptr %ars)
; CHECK: ae_sa24.l.ric aed{{[0-9]+}}, u{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <8 x i8>, ptr } %ret, 0
%unwrap = bitcast <8 x i8> %ev to <1 x i64>
ret <1 x i64> %unwrap
}


;--- ae_sa24_l_rip.ll
declare { <8 x i8>, ptr } @llvm.xtensa.ae.sa24.l.rip(<2 x i32>, <8 x i8>, ptr)
define <1 x i64> @test_xtensa_ae_sa24_l_rip(<2 x i32> %ae_ls_v, <1 x i64> %ae_ls_su, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_sa24_l_rip:

%wrap1 = bitcast <1 x i64> %ae_ls_su to <8 x i8>

%ret = call { <8 x i8>, ptr } @llvm.xtensa.ae.sa24.l.rip(<2 x i32> %ae_ls_v, <8 x i8> %wrap1, ptr %ars)
; CHECK: ae_sa24.l.rip aed{{[0-9]+}}, u{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <8 x i8>, ptr } %ret, 0
%unwrap = bitcast <8 x i8> %ev to <1 x i64>
ret <1 x i64> %unwrap
}


;--- ae_sa24x2_ic.ll
declare { <8 x i8>, ptr } @llvm.xtensa.ae.sa24x2.ic(<2 x i32>, <8 x i8>, ptr)
define <1 x i64> @test_xtensa_ae_sa24x2_ic(<2 x i32> %ae_ls_v, <1 x i64> %ae_ls_su, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_sa24x2_ic:

%wrap1 = bitcast <1 x i64> %ae_ls_su to <8 x i8>

%ret = call { <8 x i8>, ptr } @llvm.xtensa.ae.sa24x2.ic(<2 x i32> %ae_ls_v, <8 x i8> %wrap1, ptr %ars)
; CHECK: ae_sa24x2.ic aed{{[0-9]+}}, u{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <8 x i8>, ptr } %ret, 0
%unwrap = bitcast <8 x i8> %ev to <1 x i64>
ret <1 x i64> %unwrap
}


;--- ae_sa24x2_ip.ll
declare { <8 x i8>, ptr } @llvm.xtensa.ae.sa24x2.ip(<2 x i32>, <8 x i8>, ptr)
define <1 x i64> @test_xtensa_ae_sa24x2_ip(<2 x i32> %ae_ls_v, <1 x i64> %ae_ls_su, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_sa24x2_ip:

%wrap1 = bitcast <1 x i64> %ae_ls_su to <8 x i8>

%ret = call { <8 x i8>, ptr } @llvm.xtensa.ae.sa24x2.ip(<2 x i32> %ae_ls_v, <8 x i8> %wrap1, ptr %ars)
; CHECK: ae_sa24x2.ip aed{{[0-9]+}}, u{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <8 x i8>, ptr } %ret, 0
%unwrap = bitcast <8 x i8> %ev to <1 x i64>
ret <1 x i64> %unwrap
}


;--- ae_sa24x2_ric.ll
declare { <8 x i8>, ptr } @llvm.xtensa.ae.sa24x2.ric(<2 x i32>, <8 x i8>, ptr)
define <1 x i64> @test_xtensa_ae_sa24x2_ric(<2 x i32> %ae_ls_v, <1 x i64> %ae_ls_su, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_sa24x2_ric:

%wrap1 = bitcast <1 x i64> %ae_ls_su to <8 x i8>

%ret = call { <8 x i8>, ptr } @llvm.xtensa.ae.sa24x2.ric(<2 x i32> %ae_ls_v, <8 x i8> %wrap1, ptr %ars)
; CHECK: ae_sa24x2.ric aed{{[0-9]+}}, u{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <8 x i8>, ptr } %ret, 0
%unwrap = bitcast <8 x i8> %ev to <1 x i64>
ret <1 x i64> %unwrap
}


;--- ae_sa24x2_rip.ll
declare { <8 x i8>, ptr } @llvm.xtensa.ae.sa24x2.rip(<2 x i32>, <8 x i8>, ptr)
define <1 x i64> @test_xtensa_ae_sa24x2_rip(<2 x i32> %ae_ls_v, <1 x i64> %ae_ls_su, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_sa24x2_rip:

%wrap1 = bitcast <1 x i64> %ae_ls_su to <8 x i8>

%ret = call { <8 x i8>, ptr } @llvm.xtensa.ae.sa24x2.rip(<2 x i32> %ae_ls_v, <8 x i8> %wrap1, ptr %ars)
; CHECK: ae_sa24x2.rip aed{{[0-9]+}}, u{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <8 x i8>, ptr } %ret, 0
%unwrap = bitcast <8 x i8> %ev to <1 x i64>
ret <1 x i64> %unwrap
}


;--- ae_sa32x2_ic.ll
declare { <8 x i8>, ptr } @llvm.xtensa.ae.sa32x2.ic(<2 x i32>, <8 x i8>, ptr)
define <1 x i64> @test_xtensa_ae_sa32x2_ic(<2 x i32> %ae_ls_v, <1 x i64> %ae_ls_su, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_sa32x2_ic:

%wrap1 = bitcast <1 x i64> %ae_ls_su to <8 x i8>

%ret = call { <8 x i8>, ptr } @llvm.xtensa.ae.sa32x2.ic(<2 x i32> %ae_ls_v, <8 x i8> %wrap1, ptr %ars)
; CHECK: ae_sa32x2.ic aed{{[0-9]+}}, u{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <8 x i8>, ptr } %ret, 0
%unwrap = bitcast <8 x i8> %ev to <1 x i64>
ret <1 x i64> %unwrap
}


;--- ae_sa32x2_ip.ll
declare { <8 x i8>, ptr } @llvm.xtensa.ae.sa32x2.ip(<2 x i32>, <8 x i8>, ptr)
define <1 x i64> @test_xtensa_ae_sa32x2_ip(<2 x i32> %ae_ls_v, <1 x i64> %ae_ls_su, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_sa32x2_ip:

%wrap1 = bitcast <1 x i64> %ae_ls_su to <8 x i8>

%ret = call { <8 x i8>, ptr } @llvm.xtensa.ae.sa32x2.ip(<2 x i32> %ae_ls_v, <8 x i8> %wrap1, ptr %ars)
; CHECK: ae_sa32x2.ip aed{{[0-9]+}}, u{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <8 x i8>, ptr } %ret, 0
%unwrap = bitcast <8 x i8> %ev to <1 x i64>
ret <1 x i64> %unwrap
}


;--- ae_sa32x2_ric.ll
declare { <8 x i8>, ptr } @llvm.xtensa.ae.sa32x2.ric(<2 x i32>, <8 x i8>, ptr)
define <1 x i64> @test_xtensa_ae_sa32x2_ric(<2 x i32> %ae_ls_v, <1 x i64> %ae_ls_su, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_sa32x2_ric:

%wrap1 = bitcast <1 x i64> %ae_ls_su to <8 x i8>

%ret = call { <8 x i8>, ptr } @llvm.xtensa.ae.sa32x2.ric(<2 x i32> %ae_ls_v, <8 x i8> %wrap1, ptr %ars)
; CHECK: ae_sa32x2.ric aed{{[0-9]+}}, u{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <8 x i8>, ptr } %ret, 0
%unwrap = bitcast <8 x i8> %ev to <1 x i64>
ret <1 x i64> %unwrap
}


;--- ae_sa32x2_rip.ll
declare { <8 x i8>, ptr } @llvm.xtensa.ae.sa32x2.rip(<2 x i32>, <8 x i8>, ptr)
define <1 x i64> @test_xtensa_ae_sa32x2_rip(<2 x i32> %ae_ls_v, <1 x i64> %ae_ls_su, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_sa32x2_rip:

%wrap1 = bitcast <1 x i64> %ae_ls_su to <8 x i8>

%ret = call { <8 x i8>, ptr } @llvm.xtensa.ae.sa32x2.rip(<2 x i32> %ae_ls_v, <8 x i8> %wrap1, ptr %ars)
; CHECK: ae_sa32x2.rip aed{{[0-9]+}}, u{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <8 x i8>, ptr } %ret, 0
%unwrap = bitcast <8 x i8> %ev to <1 x i64>
ret <1 x i64> %unwrap
}


;--- ae_sa32x2f24_ic.ll
declare { <8 x i8>, ptr } @llvm.xtensa.ae.sa32x2f24.ic(<2 x i32>, <8 x i8>, ptr)
define <1 x i64> @test_xtensa_ae_sa32x2f24_ic(<2 x i32> %ae_ls_v, <1 x i64> %ae_ls_su, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_sa32x2f24_ic:

%wrap1 = bitcast <1 x i64> %ae_ls_su to <8 x i8>

%ret = call { <8 x i8>, ptr } @llvm.xtensa.ae.sa32x2f24.ic(<2 x i32> %ae_ls_v, <8 x i8> %wrap1, ptr %ars)
; CHECK: ae_sa32x2f24.ic aed{{[0-9]+}}, u{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <8 x i8>, ptr } %ret, 0
%unwrap = bitcast <8 x i8> %ev to <1 x i64>
ret <1 x i64> %unwrap
}


;--- ae_sa32x2f24_ip.ll
declare { <8 x i8>, ptr } @llvm.xtensa.ae.sa32x2f24.ip(<2 x i32>, <8 x i8>, ptr)
define <1 x i64> @test_xtensa_ae_sa32x2f24_ip(<2 x i32> %ae_ls_v, <1 x i64> %ae_ls_su, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_sa32x2f24_ip:

%wrap1 = bitcast <1 x i64> %ae_ls_su to <8 x i8>

%ret = call { <8 x i8>, ptr } @llvm.xtensa.ae.sa32x2f24.ip(<2 x i32> %ae_ls_v, <8 x i8> %wrap1, ptr %ars)
; CHECK: ae_sa32x2f24.ip aed{{[0-9]+}}, u{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <8 x i8>, ptr } %ret, 0
%unwrap = bitcast <8 x i8> %ev to <1 x i64>
ret <1 x i64> %unwrap
}


;--- ae_sa32x2f24_ric.ll
declare { <8 x i8>, ptr } @llvm.xtensa.ae.sa32x2f24.ric(<2 x i32>, <8 x i8>, ptr)
define <1 x i64> @test_xtensa_ae_sa32x2f24_ric(<2 x i32> %ae_ls_v, <1 x i64> %ae_ls_su, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_sa32x2f24_ric:

%wrap1 = bitcast <1 x i64> %ae_ls_su to <8 x i8>

%ret = call { <8 x i8>, ptr } @llvm.xtensa.ae.sa32x2f24.ric(<2 x i32> %ae_ls_v, <8 x i8> %wrap1, ptr %ars)
; CHECK: ae_sa32x2f24.ric aed{{[0-9]+}}, u{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <8 x i8>, ptr } %ret, 0
%unwrap = bitcast <8 x i8> %ev to <1 x i64>
ret <1 x i64> %unwrap
}


;--- ae_sa32x2f24_rip.ll
declare { <8 x i8>, ptr } @llvm.xtensa.ae.sa32x2f24.rip(<2 x i32>, <8 x i8>, ptr)
define <1 x i64> @test_xtensa_ae_sa32x2f24_rip(<2 x i32> %ae_ls_v, <1 x i64> %ae_ls_su, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_sa32x2f24_rip:

%wrap1 = bitcast <1 x i64> %ae_ls_su to <8 x i8>

%ret = call { <8 x i8>, ptr } @llvm.xtensa.ae.sa32x2f24.rip(<2 x i32> %ae_ls_v, <8 x i8> %wrap1, ptr %ars)
; CHECK: ae_sa32x2f24.rip aed{{[0-9]+}}, u{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <8 x i8>, ptr } %ret, 0
%unwrap = bitcast <8 x i8> %ev to <1 x i64>
ret <1 x i64> %unwrap
}


;--- ae_sa64neg_fp.ll
declare <8 x i8> @llvm.xtensa.ae.sa64neg.fp(<8 x i8>, ptr)
define <8 x i8> @test_xtensa_ae_sa64neg_fp(<1 x i64> %ae_ls_su, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_sa64neg_fp:
%wrap0 = bitcast <1 x i64> %ae_ls_su to <8 x i8>

%ret = call <8 x i8> @llvm.xtensa.ae.sa64neg.fp(<8 x i8> %wrap0, ptr %ars)
; CHECK: ae_sa64neg.fp u{{[0-9]+}}, a{{[0-9]+}}

ret <8 x i8> %ret
}


;--- ae_sa64pos_fp.ll
declare <8 x i8> @llvm.xtensa.ae.sa64pos.fp(<8 x i8>, ptr)
define <8 x i8> @test_xtensa_ae_sa64pos_fp(<1 x i64> %ae_ls_su, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_sa64pos_fp:
%wrap0 = bitcast <1 x i64> %ae_ls_su to <8 x i8>

%ret = call <8 x i8> @llvm.xtensa.ae.sa64pos.fp(<8 x i8> %wrap0, ptr %ars)
; CHECK: ae_sa64pos.fp u{{[0-9]+}}, a{{[0-9]+}}

ret <8 x i8> %ret
}


;--- ae_salign64_i.ll
declare void @llvm.xtensa.ae.salign64.i(<8 x i8>, ptr, i32)
define void @test_xtensa_ae_salign64_i(<1 x i64> %ae_ls_su, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_salign64_i:
%wrap0 = bitcast <1 x i64> %ae_ls_su to <8 x i8>


 call void @llvm.xtensa.ae.salign64.i(<8 x i8> %wrap0, ptr %ars, i32 -64)
; CHECK: ae_salign64.i u{{[0-9]+}}, a{{[0-9]+}}, {{-?[0-9]+}}

ret void
}


;--- ae_sat16x4.ll
declare <4 x i16> @llvm.xtensa.ae.sat16x4(<2 x i32>, <2 x i32>)
define <4 x i16> @test_xtensa_ae_sat16x4(<2 x i32> %ae_arth_v0, <2 x i32> %ae_arth_v1) {
; CHECK-LABEL: test_xtensa_ae_sat16x4:


%ret = call <4 x i16> @llvm.xtensa.ae.sat16x4(<2 x i32> %ae_arth_v0, <2 x i32> %ae_arth_v1)
; CHECK: ae_sat16x4 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <4 x i16> %ret
}


;--- ae_sat24s.ll
declare <2 x i32> @llvm.xtensa.ae.sat24s(<2 x i32>)
define <2 x i32> @test_xtensa_ae_sat24s(<2 x i32> %ae_arth_v1) {
; CHECK-LABEL: test_xtensa_ae_sat24s:

%ret = call <2 x i32> @llvm.xtensa.ae.sat24s(<2 x i32> %ae_arth_v1)
; CHECK: ae_sat24s aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_sat48s.ll
declare <1 x i64> @llvm.xtensa.ae.sat48s(<1 x i64>)
define <1 x i64> @test_xtensa_ae_sat48s(<1 x i64> %ae_arth_v1) {
; CHECK-LABEL: test_xtensa_ae_sat48s:

%ret = call <1 x i64> @llvm.xtensa.ae.sat48s(<1 x i64> %ae_arth_v1)
; CHECK: ae_sat48s aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_satq56s.ll
declare <1 x i64> @llvm.xtensa.ae.satq56s(<1 x i64>)
define <1 x i64> @test_xtensa_ae_satq56s(<1 x i64> %ae_arth_v1) {
; CHECK-LABEL: test_xtensa_ae_satq56s:

%ret = call <1 x i64> @llvm.xtensa.ae.satq56s(<1 x i64> %ae_arth_v1)
; CHECK: ae_satq56s aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_sb.ll
declare ptr @llvm.xtensa.ae.sb(ptr, i32)
define ptr @test_xtensa_ae_sb(ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_sb:


%ret = call ptr @llvm.xtensa.ae.sb(ptr %ars, i32 %art)
; CHECK: ae_sb a{{[0-9]+}}, a{{[0-9]+}}

ret ptr %ret
}


;--- ae_sb_ic.ll
declare ptr @llvm.xtensa.ae.sb.ic(ptr, i32)
define ptr @test_xtensa_ae_sb_ic(ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_sb_ic:


%ret = call ptr @llvm.xtensa.ae.sb.ic(ptr %ars, i32 %art)
; CHECK: ae_sb.ic a{{[0-9]+}}, a{{[0-9]+}}

ret ptr %ret
}


;--- ae_sb_ip.ll
declare ptr @llvm.xtensa.ae.sb.ip(ptr, i32)
define ptr @test_xtensa_ae_sb_ip(ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_sb_ip:


%ret = call ptr @llvm.xtensa.ae.sb.ip(ptr %ars, i32 %art)
; CHECK: ae_sb.ip a{{[0-9]+}}, a{{[0-9]+}}

ret ptr %ret
}


;--- ae_sbf.ll
declare ptr @llvm.xtensa.ae.sbf(ptr)
define ptr @test_xtensa_ae_sbf(ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_sbf:

%ret = call ptr @llvm.xtensa.ae.sbf(ptr %ars)
; CHECK: ae_sbf a{{[0-9]+}}

ret ptr %ret
}


;--- ae_sbf_ic.ll
declare ptr @llvm.xtensa.ae.sbf.ic(ptr)
define ptr @test_xtensa_ae_sbf_ic(ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_sbf_ic:

%ret = call ptr @llvm.xtensa.ae.sbf.ic(ptr %ars)
; CHECK: ae_sbf.ic a{{[0-9]+}}

ret ptr %ret
}


;--- ae_sbf_ip.ll
declare ptr @llvm.xtensa.ae.sbf.ip(ptr)
define ptr @test_xtensa_ae_sbf_ip(ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_sbf_ip:

%ret = call ptr @llvm.xtensa.ae.sbf.ip(ptr %ars)
; CHECK: ae_sbf.ip a{{[0-9]+}}

ret ptr %ret
}


;--- ae_sbi.ll
declare ptr @llvm.xtensa.ae.sbi(ptr, i32, i32)
define ptr @test_xtensa_ae_sbi(ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_sbi:



%ret = call ptr @llvm.xtensa.ae.sbi(ptr %ars, i32 %art, i32 1)
; CHECK: ae_sbi a{{[0-9]+}}, a{{[0-9]+}}, {{-?[0-9]+}}

ret ptr %ret
}


;--- ae_sbi_ic.ll
declare ptr @llvm.xtensa.ae.sbi.ic(ptr, i32, i32)
define ptr @test_xtensa_ae_sbi_ic(ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_sbi_ic:



%ret = call ptr @llvm.xtensa.ae.sbi.ic(ptr %ars, i32 %art, i32 1)
; CHECK: ae_sbi.ic a{{[0-9]+}}, a{{[0-9]+}}, {{-?[0-9]+}}

ret ptr %ret
}


;--- ae_sbi_ip.ll
declare ptr @llvm.xtensa.ae.sbi.ip(ptr, i32, i32)
define ptr @test_xtensa_ae_sbi_ip(ptr %ars, i32 %art) {
; CHECK-LABEL: test_xtensa_ae_sbi_ip:



%ret = call ptr @llvm.xtensa.ae.sbi.ip(ptr %ars, i32 %art, i32 1)
; CHECK: ae_sbi.ip a{{[0-9]+}}, a{{[0-9]+}}, {{-?[0-9]+}}

ret ptr %ret
}


;--- ae_sel16i.ll
declare <4 x i16> @llvm.xtensa.ae.sel16i(<4 x i16>, <4 x i16>, i32)
define <4 x i16> @test_xtensa_ae_sel16i(<4 x i16> %ae_dr_to_dr_v0, <4 x i16> %ae_dr_to_dr_v1) {
; CHECK-LABEL: test_xtensa_ae_sel16i:



%ret = call <4 x i16> @llvm.xtensa.ae.sel16i(<4 x i16> %ae_dr_to_dr_v0, <4 x i16> %ae_dr_to_dr_v1, i32 0)
; CHECK: ae_sel16i aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}, {{-?[0-9]+}}

ret <4 x i16> %ret
}


;--- ae_sel16i_n.ll
declare <4 x i16> @llvm.xtensa.ae.sel16i.n(<4 x i16>, <4 x i16>, i32)
define <4 x i16> @test_xtensa_ae_sel16i_n(<4 x i16> %ae_dr_to_dr_v0, <4 x i16> %ae_dr_to_dr_v1) {
; CHECK-LABEL: test_xtensa_ae_sel16i_n:



%ret = call <4 x i16> @llvm.xtensa.ae.sel16i.n(<4 x i16> %ae_dr_to_dr_v0, <4 x i16> %ae_dr_to_dr_v1, i32 0)
; CHECK: ae_sel16i.n aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}, {{-?[0-9]+}}

ret <4 x i16> %ret
}


;--- ae_sext32.ll
declare <2 x i32> @llvm.xtensa.ae.sext32(<2 x i32>, i32)
define <2 x i32> @test_xtensa_ae_sext32(<2 x i32> %ae_dr_to_dr_v0) {
; CHECK-LABEL: test_xtensa_ae_sext32:


%ret = call <2 x i32> @llvm.xtensa.ae.sext32(<2 x i32> %ae_dr_to_dr_v0, i32 7)
; CHECK: ae_sext32 aed{{[0-9]+}}, aed{{[0-9]+}}, {{-?[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_sext32x2d16_10.ll
declare <2 x i32> @llvm.xtensa.ae.sext32x2d16.10(<4 x i16>)
define <2 x i32> @test_xtensa_ae_sext32x2d16_10(<4 x i16> %ae_to_dr_v0) {
; CHECK-LABEL: test_xtensa_ae_sext32x2d16_10:

%ret = call <2 x i32> @llvm.xtensa.ae.sext32x2d16.10(<4 x i16> %ae_to_dr_v0)
; CHECK: ae_sext32x2d16.10 aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_sext32x2d16_32.ll
declare <2 x i32> @llvm.xtensa.ae.sext32x2d16.32(<4 x i16>)
define <2 x i32> @test_xtensa_ae_sext32x2d16_32(<4 x i16> %ae_to_dr_v0) {
; CHECK-LABEL: test_xtensa_ae_sext32x2d16_32:

%ret = call <2 x i32> @llvm.xtensa.ae.sext32x2d16.32(<4 x i16> %ae_to_dr_v0)
; CHECK: ae_sext32x2d16.32 aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_sha32.ll
declare i32 @llvm.xtensa.ae.sha32(i32)
define i32 @test_xtensa_ae_sha32(i32 %ars) {
; CHECK-LABEL: test_xtensa_ae_sha32:

%ret = call i32 @llvm.xtensa.ae.sha32(i32 %ars)
; CHECK: ae_sha32 a{{[0-9]+}}, a{{[0-9]+}}

ret i32 %ret
}


;--- ae_shortswap.ll
declare <4 x i16> @llvm.xtensa.ae.shortswap(<4 x i16>)
define <4 x i16> @test_xtensa_ae_shortswap(<4 x i16> %ae_to_dr_v0) {
; CHECK-LABEL: test_xtensa_ae_shortswap:

%ret = call <4 x i16> @llvm.xtensa.ae.shortswap(<4 x i16> %ae_to_dr_v0)
; CHECK: ae_shortswap aed{{[0-9]+}}, aed{{[0-9]+}}

ret <4 x i16> %ret
}


;--- ae_slaa16s.ll
declare <4 x i16> @llvm.xtensa.ae.slaa16s(<4 x i16>, i32)
define <4 x i16> @test_xtensa_ae_slaa16s(<4 x i16> %ae_shift_d0, i32 %ars) {
; CHECK-LABEL: test_xtensa_ae_slaa16s:


%ret = call <4 x i16> @llvm.xtensa.ae.slaa16s(<4 x i16> %ae_shift_d0, i32 %ars)
; CHECK: ae_slaa16s aed{{[0-9]+}}, aed{{[0-9]+}}, a{{[0-9]+}}

ret <4 x i16> %ret
}


;--- ae_slaa32.ll
declare <2 x i32> @llvm.xtensa.ae.slaa32(<2 x i32>, i32)
define <2 x i32> @test_xtensa_ae_slaa32(<2 x i32> %ae_shift_d0, i32 %ars) {
; CHECK-LABEL: test_xtensa_ae_slaa32:


%ret = call <2 x i32> @llvm.xtensa.ae.slaa32(<2 x i32> %ae_shift_d0, i32 %ars)
; CHECK: ae_slaa32 aed{{[0-9]+}}, aed{{[0-9]+}}, a{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_slaa32s.ll
declare <2 x i32> @llvm.xtensa.ae.slaa32s(<2 x i32>, i32)
define <2 x i32> @test_xtensa_ae_slaa32s(<2 x i32> %ae_shift_d0, i32 %ars) {
; CHECK-LABEL: test_xtensa_ae_slaa32s:


%ret = call <2 x i32> @llvm.xtensa.ae.slaa32s(<2 x i32> %ae_shift_d0, i32 %ars)
; CHECK: ae_slaa32s aed{{[0-9]+}}, aed{{[0-9]+}}, a{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_slaa64.ll
declare <1 x i64> @llvm.xtensa.ae.slaa64(<1 x i64>, i32)
define <1 x i64> @test_xtensa_ae_slaa64(<1 x i64> %ae_shift_d0, i32 %ars) {
; CHECK-LABEL: test_xtensa_ae_slaa64:


%ret = call <1 x i64> @llvm.xtensa.ae.slaa64(<1 x i64> %ae_shift_d0, i32 %ars)
; CHECK: ae_slaa64 aed{{[0-9]+}}, aed{{[0-9]+}}, a{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_slaa64s.ll
declare <1 x i64> @llvm.xtensa.ae.slaa64s(<1 x i64>, i32)
define <1 x i64> @test_xtensa_ae_slaa64s(<1 x i64> %ae_shift_d0, i32 %ars) {
; CHECK-LABEL: test_xtensa_ae_slaa64s:


%ret = call <1 x i64> @llvm.xtensa.ae.slaa64s(<1 x i64> %ae_shift_d0, i32 %ars)
; CHECK: ae_slaa64s aed{{[0-9]+}}, aed{{[0-9]+}}, a{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_slaaq56.ll
declare <1 x i64> @llvm.xtensa.ae.slaaq56(<1 x i64>, i32)
define <1 x i64> @test_xtensa_ae_slaaq56(<1 x i64> %ae_shift_d0, i32 %ars) {
; CHECK-LABEL: test_xtensa_ae_slaaq56:


%ret = call <1 x i64> @llvm.xtensa.ae.slaaq56(<1 x i64> %ae_shift_d0, i32 %ars)
; CHECK: ae_slaaq56 aed{{[0-9]+}}, aed{{[0-9]+}}, a{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_slai16s.ll
declare <4 x i16> @llvm.xtensa.ae.slai16s(<4 x i16>, i32)
define <4 x i16> @test_xtensa_ae_slai16s(<4 x i16> %ae_shift_d0) {
; CHECK-LABEL: test_xtensa_ae_slai16s:


%ret = call <4 x i16> @llvm.xtensa.ae.slai16s(<4 x i16> %ae_shift_d0, i32 0)
; CHECK: ae_slai16s aed{{[0-9]+}}, aed{{[0-9]+}}, {{-?[0-9]+}}

ret <4 x i16> %ret
}


;--- ae_slai24.ll
declare <2 x i32> @llvm.xtensa.ae.slai24(<2 x i32>, i32)
define <2 x i32> @test_xtensa_ae_slai24(<2 x i32> %ae_shift_d0) {
; CHECK-LABEL: test_xtensa_ae_slai24:


%ret = call <2 x i32> @llvm.xtensa.ae.slai24(<2 x i32> %ae_shift_d0, i32 0)
; CHECK: ae_slai24 aed{{[0-9]+}}, aed{{[0-9]+}}, {{-?[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_slai24s.ll
declare <2 x i32> @llvm.xtensa.ae.slai24s(<2 x i32>, i32)
define <2 x i32> @test_xtensa_ae_slai24s(<2 x i32> %ae_shift_d0) {
; CHECK-LABEL: test_xtensa_ae_slai24s:


%ret = call <2 x i32> @llvm.xtensa.ae.slai24s(<2 x i32> %ae_shift_d0, i32 0)
; CHECK: ae_slai24s aed{{[0-9]+}}, aed{{[0-9]+}}, {{-?[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_slai32.ll
declare <2 x i32> @llvm.xtensa.ae.slai32(<2 x i32>, i32)
define <2 x i32> @test_xtensa_ae_slai32(<2 x i32> %ae_shift_d0) {
; CHECK-LABEL: test_xtensa_ae_slai32:


%ret = call <2 x i32> @llvm.xtensa.ae.slai32(<2 x i32> %ae_shift_d0, i32 0)
; CHECK: ae_slai32 aed{{[0-9]+}}, aed{{[0-9]+}}, {{-?[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_slai32s.ll
declare <2 x i32> @llvm.xtensa.ae.slai32s(<2 x i32>, i32)
define <2 x i32> @test_xtensa_ae_slai32s(<2 x i32> %ae_shift_d0) {
; CHECK-LABEL: test_xtensa_ae_slai32s:


%ret = call <2 x i32> @llvm.xtensa.ae.slai32s(<2 x i32> %ae_shift_d0, i32 0)
; CHECK: ae_slai32s aed{{[0-9]+}}, aed{{[0-9]+}}, {{-?[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_slai64.ll
declare <1 x i64> @llvm.xtensa.ae.slai64(<1 x i64>, i32)
define <1 x i64> @test_xtensa_ae_slai64(<1 x i64> %ae_shift_d0) {
; CHECK-LABEL: test_xtensa_ae_slai64:


%ret = call <1 x i64> @llvm.xtensa.ae.slai64(<1 x i64> %ae_shift_d0, i32 0)
; CHECK: ae_slai64 aed{{[0-9]+}}, aed{{[0-9]+}}, {{-?[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_slai64s.ll
declare <1 x i64> @llvm.xtensa.ae.slai64s(<1 x i64>, i32)
define <1 x i64> @test_xtensa_ae_slai64s(<1 x i64> %ae_shift_d0) {
; CHECK-LABEL: test_xtensa_ae_slai64s:


%ret = call <1 x i64> @llvm.xtensa.ae.slai64s(<1 x i64> %ae_shift_d0, i32 0)
; CHECK: ae_slai64s aed{{[0-9]+}}, aed{{[0-9]+}}, {{-?[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_slaisq56s.ll
declare <1 x i64> @llvm.xtensa.ae.slaisq56s(<1 x i64>, i32)
define <1 x i64> @test_xtensa_ae_slaisq56s(<1 x i64> %ae_shift_d0) {
; CHECK-LABEL: test_xtensa_ae_slaisq56s:


%ret = call <1 x i64> @llvm.xtensa.ae.slaisq56s(<1 x i64> %ae_shift_d0, i32 0)
; CHECK: ae_slaisq56s aed{{[0-9]+}}, aed{{[0-9]+}}, {{-?[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_slas24.ll
declare <2 x i32> @llvm.xtensa.ae.slas24(<2 x i32>)
define <2 x i32> @test_xtensa_ae_slas24(<2 x i32> %ae_shift_d0) {
; CHECK-LABEL: test_xtensa_ae_slas24:

%ret = call <2 x i32> @llvm.xtensa.ae.slas24(<2 x i32> %ae_shift_d0)
; CHECK: ae_slas24 aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_slas24s.ll
declare <2 x i32> @llvm.xtensa.ae.slas24s(<2 x i32>)
define <2 x i32> @test_xtensa_ae_slas24s(<2 x i32> %ae_shift_d0) {
; CHECK-LABEL: test_xtensa_ae_slas24s:

%ret = call <2 x i32> @llvm.xtensa.ae.slas24s(<2 x i32> %ae_shift_d0)
; CHECK: ae_slas24s aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_slas32.ll
declare <2 x i32> @llvm.xtensa.ae.slas32(<2 x i32>)
define <2 x i32> @test_xtensa_ae_slas32(<2 x i32> %ae_shift_d0) {
; CHECK-LABEL: test_xtensa_ae_slas32:

%ret = call <2 x i32> @llvm.xtensa.ae.slas32(<2 x i32> %ae_shift_d0)
; CHECK: ae_slas32 aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_slas32s.ll
declare <2 x i32> @llvm.xtensa.ae.slas32s(<2 x i32>)
define <2 x i32> @test_xtensa_ae_slas32s(<2 x i32> %ae_shift_d0) {
; CHECK-LABEL: test_xtensa_ae_slas32s:

%ret = call <2 x i32> @llvm.xtensa.ae.slas32s(<2 x i32> %ae_shift_d0)
; CHECK: ae_slas32s aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_slas64.ll
declare <1 x i64> @llvm.xtensa.ae.slas64(<1 x i64>)
define <1 x i64> @test_xtensa_ae_slas64(<1 x i64> %ae_shift_d0) {
; CHECK-LABEL: test_xtensa_ae_slas64:

%ret = call <1 x i64> @llvm.xtensa.ae.slas64(<1 x i64> %ae_shift_d0)
; CHECK: ae_slas64 aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_slas64s.ll
declare <1 x i64> @llvm.xtensa.ae.slas64s(<1 x i64>)
define <1 x i64> @test_xtensa_ae_slas64s(<1 x i64> %ae_shift_d0) {
; CHECK-LABEL: test_xtensa_ae_slas64s:

%ret = call <1 x i64> @llvm.xtensa.ae.slas64s(<1 x i64> %ae_shift_d0)
; CHECK: ae_slas64s aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_slasq56.ll
declare <1 x i64> @llvm.xtensa.ae.slasq56(<1 x i64>)
define <1 x i64> @test_xtensa_ae_slasq56(<1 x i64> %ae_shift_d0) {
; CHECK-LABEL: test_xtensa_ae_slasq56:

%ret = call <1 x i64> @llvm.xtensa.ae.slasq56(<1 x i64> %ae_shift_d0)
; CHECK: ae_slasq56 aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_slassq56s.ll
declare <1 x i64> @llvm.xtensa.ae.slassq56s(<1 x i64>)
define <1 x i64> @test_xtensa_ae_slassq56s(<1 x i64> %ae_shift_d0) {
; CHECK-LABEL: test_xtensa_ae_slassq56s:

%ret = call <1 x i64> @llvm.xtensa.ae.slassq56s(<1 x i64> %ae_shift_d0)
; CHECK: ae_slassq56s aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_sra64_32.ll
declare <1 x i64> @llvm.xtensa.ae.sra64.32(<2 x i32>, i32)
define <1 x i64> @test_xtensa_ae_sra64_32(<2 x i32> %ae_shift_d0, i32 %ars) {
; CHECK-LABEL: test_xtensa_ae_sra64_32:


%ret = call <1 x i64> @llvm.xtensa.ae.sra64.32(<2 x i32> %ae_shift_d0, i32 %ars)
; CHECK: ae_sra64_32 aed{{[0-9]+}}, aed{{[0-9]+}}, a{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_sraa16rs.ll
declare <4 x i16> @llvm.xtensa.ae.sraa16rs(<4 x i16>, i32)
define <4 x i16> @test_xtensa_ae_sraa16rs(<4 x i16> %ae_shift_d0, i32 %ars) {
; CHECK-LABEL: test_xtensa_ae_sraa16rs:


%ret = call <4 x i16> @llvm.xtensa.ae.sraa16rs(<4 x i16> %ae_shift_d0, i32 %ars)
; CHECK: ae_sraa16rs aed{{[0-9]+}}, aed{{[0-9]+}}, a{{[0-9]+}}

ret <4 x i16> %ret
}


;--- ae_sraa16s.ll
declare <4 x i16> @llvm.xtensa.ae.sraa16s(<4 x i16>, i32)
define <4 x i16> @test_xtensa_ae_sraa16s(<4 x i16> %ae_shift_d0, i32 %ars) {
; CHECK-LABEL: test_xtensa_ae_sraa16s:


%ret = call <4 x i16> @llvm.xtensa.ae.sraa16s(<4 x i16> %ae_shift_d0, i32 %ars)
; CHECK: ae_sraa16s aed{{[0-9]+}}, aed{{[0-9]+}}, a{{[0-9]+}}

ret <4 x i16> %ret
}


;--- ae_sraa32.ll
declare <2 x i32> @llvm.xtensa.ae.sraa32(<2 x i32>, i32)
define <2 x i32> @test_xtensa_ae_sraa32(<2 x i32> %ae_shift_d0, i32 %ars) {
; CHECK-LABEL: test_xtensa_ae_sraa32:


%ret = call <2 x i32> @llvm.xtensa.ae.sraa32(<2 x i32> %ae_shift_d0, i32 %ars)
; CHECK: ae_sraa32 aed{{[0-9]+}}, aed{{[0-9]+}}, a{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_sraa32rs.ll
declare <2 x i32> @llvm.xtensa.ae.sraa32rs(<2 x i32>, i32)
define <2 x i32> @test_xtensa_ae_sraa32rs(<2 x i32> %ae_shift_d0, i32 %ars) {
; CHECK-LABEL: test_xtensa_ae_sraa32rs:


%ret = call <2 x i32> @llvm.xtensa.ae.sraa32rs(<2 x i32> %ae_shift_d0, i32 %ars)
; CHECK: ae_sraa32rs aed{{[0-9]+}}, aed{{[0-9]+}}, a{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_sraa32s.ll
declare <2 x i32> @llvm.xtensa.ae.sraa32s(<2 x i32>, i32)
define <2 x i32> @test_xtensa_ae_sraa32s(<2 x i32> %ae_shift_d0, i32 %ars) {
; CHECK-LABEL: test_xtensa_ae_sraa32s:


%ret = call <2 x i32> @llvm.xtensa.ae.sraa32s(<2 x i32> %ae_shift_d0, i32 %ars)
; CHECK: ae_sraa32s aed{{[0-9]+}}, aed{{[0-9]+}}, a{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_sraa64.ll
declare <1 x i64> @llvm.xtensa.ae.sraa64(<1 x i64>, i32)
define <1 x i64> @test_xtensa_ae_sraa64(<1 x i64> %ae_shift_d0, i32 %ars) {
; CHECK-LABEL: test_xtensa_ae_sraa64:


%ret = call <1 x i64> @llvm.xtensa.ae.sraa64(<1 x i64> %ae_shift_d0, i32 %ars)
; CHECK: ae_sraa64 aed{{[0-9]+}}, aed{{[0-9]+}}, a{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_srai16.ll
declare <4 x i16> @llvm.xtensa.ae.srai16(<4 x i16>, i32)
define <4 x i16> @test_xtensa_ae_srai16(<4 x i16> %ae_shift_d0) {
; CHECK-LABEL: test_xtensa_ae_srai16:


%ret = call <4 x i16> @llvm.xtensa.ae.srai16(<4 x i16> %ae_shift_d0, i32 0)
; CHECK: ae_srai16 aed{{[0-9]+}}, aed{{[0-9]+}}, {{-?[0-9]+}}

ret <4 x i16> %ret
}


;--- ae_srai16r.ll
declare <4 x i16> @llvm.xtensa.ae.srai16r(<4 x i16>, i32)
define <4 x i16> @test_xtensa_ae_srai16r(<4 x i16> %ae_shift_d0) {
; CHECK-LABEL: test_xtensa_ae_srai16r:


%ret = call <4 x i16> @llvm.xtensa.ae.srai16r(<4 x i16> %ae_shift_d0, i32 0)
; CHECK: ae_srai16r aed{{[0-9]+}}, aed{{[0-9]+}}, {{-?[0-9]+}}

ret <4 x i16> %ret
}


;--- ae_srai24.ll
declare <2 x i32> @llvm.xtensa.ae.srai24(<2 x i32>, i32)
define <2 x i32> @test_xtensa_ae_srai24(<2 x i32> %ae_shift_d0) {
; CHECK-LABEL: test_xtensa_ae_srai24:


%ret = call <2 x i32> @llvm.xtensa.ae.srai24(<2 x i32> %ae_shift_d0, i32 0)
; CHECK: ae_srai24 aed{{[0-9]+}}, aed{{[0-9]+}}, {{-?[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_srai32.ll
declare <2 x i32> @llvm.xtensa.ae.srai32(<2 x i32>, i32)
define <2 x i32> @test_xtensa_ae_srai32(<2 x i32> %ae_shift_d0) {
; CHECK-LABEL: test_xtensa_ae_srai32:


%ret = call <2 x i32> @llvm.xtensa.ae.srai32(<2 x i32> %ae_shift_d0, i32 0)
; CHECK: ae_srai32 aed{{[0-9]+}}, aed{{[0-9]+}}, {{-?[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_srai32r.ll
declare <2 x i32> @llvm.xtensa.ae.srai32r(<2 x i32>, i32)
define <2 x i32> @test_xtensa_ae_srai32r(<2 x i32> %ae_shift_d0) {
; CHECK-LABEL: test_xtensa_ae_srai32r:


%ret = call <2 x i32> @llvm.xtensa.ae.srai32r(<2 x i32> %ae_shift_d0, i32 0)
; CHECK: ae_srai32r aed{{[0-9]+}}, aed{{[0-9]+}}, {{-?[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_srai64.ll
declare <1 x i64> @llvm.xtensa.ae.srai64(<1 x i64>, i32)
define <1 x i64> @test_xtensa_ae_srai64(<1 x i64> %ae_shift_d0) {
; CHECK-LABEL: test_xtensa_ae_srai64:


%ret = call <1 x i64> @llvm.xtensa.ae.srai64(<1 x i64> %ae_shift_d0, i32 0)
; CHECK: ae_srai64 aed{{[0-9]+}}, aed{{[0-9]+}}, {{-?[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_sras24.ll
declare <2 x i32> @llvm.xtensa.ae.sras24(<2 x i32>)
define <2 x i32> @test_xtensa_ae_sras24(<2 x i32> %ae_shift_d0) {
; CHECK-LABEL: test_xtensa_ae_sras24:

%ret = call <2 x i32> @llvm.xtensa.ae.sras24(<2 x i32> %ae_shift_d0)
; CHECK: ae_sras24 aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_sras32.ll
declare <2 x i32> @llvm.xtensa.ae.sras32(<2 x i32>)
define <2 x i32> @test_xtensa_ae_sras32(<2 x i32> %ae_shift_d0) {
; CHECK-LABEL: test_xtensa_ae_sras32:

%ret = call <2 x i32> @llvm.xtensa.ae.sras32(<2 x i32> %ae_shift_d0)
; CHECK: ae_sras32 aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_sras64.ll
declare <1 x i64> @llvm.xtensa.ae.sras64(<1 x i64>)
define <1 x i64> @test_xtensa_ae_sras64(<1 x i64> %ae_shift_d0) {
; CHECK-LABEL: test_xtensa_ae_sras64:

%ret = call <1 x i64> @llvm.xtensa.ae.sras64(<1 x i64> %ae_shift_d0)
; CHECK: ae_sras64 aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_srla32.ll
declare <2 x i32> @llvm.xtensa.ae.srla32(<2 x i32>, i32)
define <2 x i32> @test_xtensa_ae_srla32(<2 x i32> %ae_shift_d0, i32 %ars) {
; CHECK-LABEL: test_xtensa_ae_srla32:


%ret = call <2 x i32> @llvm.xtensa.ae.srla32(<2 x i32> %ae_shift_d0, i32 %ars)
; CHECK: ae_srla32 aed{{[0-9]+}}, aed{{[0-9]+}}, a{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_srla64.ll
declare <1 x i64> @llvm.xtensa.ae.srla64(<1 x i64>, i32)
define <1 x i64> @test_xtensa_ae_srla64(<1 x i64> %ae_shift_d0, i32 %ars) {
; CHECK-LABEL: test_xtensa_ae_srla64:


%ret = call <1 x i64> @llvm.xtensa.ae.srla64(<1 x i64> %ae_shift_d0, i32 %ars)
; CHECK: ae_srla64 aed{{[0-9]+}}, aed{{[0-9]+}}, a{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_srli24.ll
declare <2 x i32> @llvm.xtensa.ae.srli24(<2 x i32>, i32)
define <2 x i32> @test_xtensa_ae_srli24(<2 x i32> %ae_shift_d0) {
; CHECK-LABEL: test_xtensa_ae_srli24:


%ret = call <2 x i32> @llvm.xtensa.ae.srli24(<2 x i32> %ae_shift_d0, i32 0)
; CHECK: ae_srli24 aed{{[0-9]+}}, aed{{[0-9]+}}, {{-?[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_srli32.ll
declare <2 x i32> @llvm.xtensa.ae.srli32(<2 x i32>, i32)
define <2 x i32> @test_xtensa_ae_srli32(<2 x i32> %ae_shift_d0) {
; CHECK-LABEL: test_xtensa_ae_srli32:


%ret = call <2 x i32> @llvm.xtensa.ae.srli32(<2 x i32> %ae_shift_d0, i32 0)
; CHECK: ae_srli32 aed{{[0-9]+}}, aed{{[0-9]+}}, {{-?[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_srli64.ll
declare <1 x i64> @llvm.xtensa.ae.srli64(<1 x i64>, i32)
define <1 x i64> @test_xtensa_ae_srli64(<1 x i64> %ae_shift_d0) {
; CHECK-LABEL: test_xtensa_ae_srli64:


%ret = call <1 x i64> @llvm.xtensa.ae.srli64(<1 x i64> %ae_shift_d0, i32 0)
; CHECK: ae_srli64 aed{{[0-9]+}}, aed{{[0-9]+}}, {{-?[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_srls24.ll
declare <2 x i32> @llvm.xtensa.ae.srls24(<2 x i32>)
define <2 x i32> @test_xtensa_ae_srls24(<2 x i32> %ae_shift_d0) {
; CHECK-LABEL: test_xtensa_ae_srls24:

%ret = call <2 x i32> @llvm.xtensa.ae.srls24(<2 x i32> %ae_shift_d0)
; CHECK: ae_srls24 aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_srls32.ll
declare <2 x i32> @llvm.xtensa.ae.srls32(<2 x i32>)
define <2 x i32> @test_xtensa_ae_srls32(<2 x i32> %ae_shift_d0) {
; CHECK-LABEL: test_xtensa_ae_srls32:

%ret = call <2 x i32> @llvm.xtensa.ae.srls32(<2 x i32> %ae_shift_d0)
; CHECK: ae_srls32 aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_srls64.ll
declare <1 x i64> @llvm.xtensa.ae.srls64(<1 x i64>)
define <1 x i64> @test_xtensa_ae_srls64(<1 x i64> %ae_shift_d0) {
; CHECK-LABEL: test_xtensa_ae_srls64:

%ret = call <1 x i64> @llvm.xtensa.ae.srls64(<1 x i64> %ae_shift_d0)
; CHECK: ae_srls64 aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_sub16.ll
declare <4 x i16> @llvm.xtensa.ae.sub16(<4 x i16>, <4 x i16>)
define <4 x i16> @test_xtensa_ae_sub16(<4 x i16> %ae_arth_v0, <4 x i16> %ae_arth_v1) {
; CHECK-LABEL: test_xtensa_ae_sub16:


%ret = call <4 x i16> @llvm.xtensa.ae.sub16(<4 x i16> %ae_arth_v0, <4 x i16> %ae_arth_v1)
; CHECK: ae_sub16 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <4 x i16> %ret
}


;--- ae_sub16s.ll
declare <4 x i16> @llvm.xtensa.ae.sub16s(<4 x i16>, <4 x i16>)
define <4 x i16> @test_xtensa_ae_sub16s(<4 x i16> %ae_arth_v0, <4 x i16> %ae_arth_v1) {
; CHECK-LABEL: test_xtensa_ae_sub16s:


%ret = call <4 x i16> @llvm.xtensa.ae.sub16s(<4 x i16> %ae_arth_v0, <4 x i16> %ae_arth_v1)
; CHECK: ae_sub16s aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <4 x i16> %ret
}


;--- ae_sub24s.ll
declare <2 x i32> @llvm.xtensa.ae.sub24s(<2 x i32>, <2 x i32>)
define <2 x i32> @test_xtensa_ae_sub24s(<2 x i32> %ae_arth_v0, <2 x i32> %ae_arth_v1) {
; CHECK-LABEL: test_xtensa_ae_sub24s:


%ret = call <2 x i32> @llvm.xtensa.ae.sub24s(<2 x i32> %ae_arth_v0, <2 x i32> %ae_arth_v1)
; CHECK: ae_sub24s aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_sub32.ll
declare <2 x i32> @llvm.xtensa.ae.sub32(<2 x i32>, <2 x i32>)
define <2 x i32> @test_xtensa_ae_sub32(<2 x i32> %ae_arth_v0, <2 x i32> %ae_arth_v1) {
; CHECK-LABEL: test_xtensa_ae_sub32:


%ret = call <2 x i32> @llvm.xtensa.ae.sub32(<2 x i32> %ae_arth_v0, <2 x i32> %ae_arth_v1)
; CHECK: ae_sub32 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_sub32s.ll
declare <2 x i32> @llvm.xtensa.ae.sub32s(<2 x i32>, <2 x i32>)
define <2 x i32> @test_xtensa_ae_sub32s(<2 x i32> %ae_arth_v0, <2 x i32> %ae_arth_v1) {
; CHECK-LABEL: test_xtensa_ae_sub32s:


%ret = call <2 x i32> @llvm.xtensa.ae.sub32s(<2 x i32> %ae_arth_v0, <2 x i32> %ae_arth_v1)
; CHECK: ae_sub32s aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_sub64.ll
declare <1 x i64> @llvm.xtensa.ae.sub64(<1 x i64>, <1 x i64>)
define <1 x i64> @test_xtensa_ae_sub64(<1 x i64> %ae_arth_v0, <1 x i64> %ae_arth_v1) {
; CHECK-LABEL: test_xtensa_ae_sub64:


%ret = call <1 x i64> @llvm.xtensa.ae.sub64(<1 x i64> %ae_arth_v0, <1 x i64> %ae_arth_v1)
; CHECK: ae_sub64 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_sub64s.ll
declare <1 x i64> @llvm.xtensa.ae.sub64s(<1 x i64>, <1 x i64>)
define <1 x i64> @test_xtensa_ae_sub64s(<1 x i64> %ae_arth_v0, <1 x i64> %ae_arth_v1) {
; CHECK-LABEL: test_xtensa_ae_sub64s:


%ret = call <1 x i64> @llvm.xtensa.ae.sub64s(<1 x i64> %ae_arth_v0, <1 x i64> %ae_arth_v1)
; CHECK: ae_sub64s aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_subadd32.ll
declare <2 x i32> @llvm.xtensa.ae.subadd32(<2 x i32>, <2 x i32>)
define <2 x i32> @test_xtensa_ae_subadd32(<2 x i32> %ae_arth_v0, <2 x i32> %ae_arth_v1) {
; CHECK-LABEL: test_xtensa_ae_subadd32:


%ret = call <2 x i32> @llvm.xtensa.ae.subadd32(<2 x i32> %ae_arth_v0, <2 x i32> %ae_arth_v1)
; CHECK: ae_subadd32 aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_subadd32s.ll
declare <2 x i32> @llvm.xtensa.ae.subadd32s(<2 x i32>, <2 x i32>)
define <2 x i32> @test_xtensa_ae_subadd32s(<2 x i32> %ae_arth_v0, <2 x i32> %ae_arth_v1) {
; CHECK-LABEL: test_xtensa_ae_subadd32s:


%ret = call <2 x i32> @llvm.xtensa.ae.subadd32s(<2 x i32> %ae_arth_v0, <2 x i32> %ae_arth_v1)
; CHECK: ae_subadd32s aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_trunca32f64s_l.ll
declare <2 x i32> @llvm.xtensa.ae.trunca32f64s.l(<2 x i32>, <1 x i64>, i32)
define <2 x i32> @test_xtensa_ae_trunca32f64s_l(<2 x i32> %ae_shift_d0, <1 x i64> %ae_shift_sd, i32 %ars) {
; CHECK-LABEL: test_xtensa_ae_trunca32f64s_l:



%ret = call <2 x i32> @llvm.xtensa.ae.trunca32f64s.l(<2 x i32> %ae_shift_d0, <1 x i64> %ae_shift_sd, i32 %ars)
; CHECK: ae_trunca32f64s.l aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}, a{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_trunca32x2f64s.ll
declare <2 x i32> @llvm.xtensa.ae.trunca32x2f64s(<1 x i64>, <1 x i64>, i32)
define <2 x i32> @test_xtensa_ae_trunca32x2f64s(<1 x i64> %ae_shift_d0, <1 x i64> %ae_shift_sd, i32 %ars) {
; CHECK-LABEL: test_xtensa_ae_trunca32x2f64s:



%ret = call <2 x i32> @llvm.xtensa.ae.trunca32x2f64s(<1 x i64> %ae_shift_d0, <1 x i64> %ae_shift_sd, i32 %ars)
; CHECK: ae_trunca32x2f64s aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}, a{{[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_trunci32f64s_l.ll
declare <2 x i32> @llvm.xtensa.ae.trunci32f64s.l(<2 x i32>, <1 x i64>, i32)
define <2 x i32> @test_xtensa_ae_trunci32f64s_l(<2 x i32> %ae_shift_d0, <1 x i64> %ae_shift_sd) {
; CHECK-LABEL: test_xtensa_ae_trunci32f64s_l:



%ret = call <2 x i32> @llvm.xtensa.ae.trunci32f64s.l(<2 x i32> %ae_shift_d0, <1 x i64> %ae_shift_sd, i32 0)
; CHECK: ae_trunci32f64s.l aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}, {{-?[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_trunci32x2f64s.ll
declare <2 x i32> @llvm.xtensa.ae.trunci32x2f64s(<1 x i64>, <1 x i64>, i32)
define <2 x i32> @test_xtensa_ae_trunci32x2f64s(<1 x i64> %ae_shift_d0, <1 x i64> %ae_shift_sd) {
; CHECK-LABEL: test_xtensa_ae_trunci32x2f64s:



%ret = call <2 x i32> @llvm.xtensa.ae.trunci32x2f64s(<1 x i64> %ae_shift_d0, <1 x i64> %ae_shift_sd, i32 0)
; CHECK: ae_trunci32x2f64s aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}, {{-?[0-9]+}}

ret <2 x i32> %ret
}


;--- ae_vldl16c.ll
declare ptr @llvm.xtensa.ae.vldl16c(ptr)
define ptr @test_xtensa_ae_vldl16c(ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_vldl16c:

%ret = call ptr @llvm.xtensa.ae.vldl16c(ptr %ars)
; CHECK: ae_vldl16c a{{[0-9]+}}

ret ptr %ret
}


;--- ae_vldl16c_ic.ll
declare ptr @llvm.xtensa.ae.vldl16c.ic(ptr)
define ptr @test_xtensa_ae_vldl16c_ic(ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_vldl16c_ic:

%ret = call ptr @llvm.xtensa.ae.vldl16c.ic(ptr %ars)
; CHECK: ae_vldl16c.ic a{{[0-9]+}}

ret ptr %ret
}


;--- ae_vldl16c_ip.ll
declare ptr @llvm.xtensa.ae.vldl16c.ip(ptr)
define ptr @test_xtensa_ae_vldl16c_ip(ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_vldl16c_ip:

%ret = call ptr @llvm.xtensa.ae.vldl16c.ip(ptr %ars)
; CHECK: ae_vldl16c.ip a{{[0-9]+}}

ret ptr %ret
}


;--- ae_vldl16t.ll
declare { <1 x i1>, i32 } @llvm.xtensa.ae.vldl16t(ptr)
define <1 x i1> @test_xtensa_ae_vldl16t(ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_vldl16t:

%ret = call { <1 x i1>, i32 } @llvm.xtensa.ae.vldl16t(ptr %ars)
; CHECK: ae_vldl16t b{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <1 x i1>, i32 } %ret, 0
ret <1 x i1> %ev
}


;--- ae_vldl32t.ll
declare { <1 x i1>, i32 } @llvm.xtensa.ae.vldl32t(ptr)
define <1 x i1> @test_xtensa_ae_vldl32t(ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_vldl32t:

%ret = call { <1 x i1>, i32 } @llvm.xtensa.ae.vldl32t(ptr %ars)
; CHECK: ae_vldl32t b{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <1 x i1>, i32 } %ret, 0
ret <1 x i1> %ev
}


;--- ae_vldsht.ll
declare void @llvm.xtensa.ae.vldsht(i32)
define void @test_xtensa_ae_vldsht(i32 %art) {
; CHECK-LABEL: test_xtensa_ae_vldsht:

 call void @llvm.xtensa.ae.vldsht(i32 %art)
; CHECK: ae_vldsht a{{[0-9]+}}

ret void
}


;--- ae_vlel16t.ll
declare { <1 x i1>, i32 } @llvm.xtensa.ae.vlel16t(i32, ptr)
define <1 x i1> @test_xtensa_ae_vlel16t(i32 %art, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_vlel16t:


%ret = call { <1 x i1>, i32 } @llvm.xtensa.ae.vlel16t(i32 %art, ptr %ars)
; CHECK: ae_vlel16t b{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <1 x i1>, i32 } %ret, 0
ret <1 x i1> %ev
}


;--- ae_vlel32t.ll
declare { <1 x i1>, i32 } @llvm.xtensa.ae.vlel32t(i32, ptr)
define <1 x i1> @test_xtensa_ae_vlel32t(i32 %art, ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_vlel32t:


%ret = call { <1 x i1>, i32 } @llvm.xtensa.ae.vlel32t(i32 %art, ptr %ars)
; CHECK: ae_vlel32t b{{[0-9]+}}, a{{[0-9]+}}, a{{[0-9]+}}
%ev = extractvalue { <1 x i1>, i32 } %ret, 0
ret <1 x i1> %ev
}


;--- ae_vles16c.ll
declare ptr @llvm.xtensa.ae.vles16c(ptr)
define ptr @test_xtensa_ae_vles16c(ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_vles16c:

%ret = call ptr @llvm.xtensa.ae.vles16c(ptr %ars)
; CHECK: ae_vles16c a{{[0-9]+}}

ret ptr %ret
}


;--- ae_vles16c_ic.ll
declare ptr @llvm.xtensa.ae.vles16c.ic(ptr)
define ptr @test_xtensa_ae_vles16c_ic(ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_vles16c_ic:

%ret = call ptr @llvm.xtensa.ae.vles16c.ic(ptr %ars)
; CHECK: ae_vles16c.ic a{{[0-9]+}}

ret ptr %ret
}


;--- ae_vles16c_ip.ll
declare ptr @llvm.xtensa.ae.vles16c.ip(ptr)
define ptr @test_xtensa_ae_vles16c_ip(ptr %ars) {
; CHECK-LABEL: test_xtensa_ae_vles16c_ip:

%ret = call ptr @llvm.xtensa.ae.vles16c.ip(ptr %ars)
; CHECK: ae_vles16c.ip a{{[0-9]+}}

ret ptr %ret
}


;--- ae_xor.ll
declare <1 x i64> @llvm.xtensa.ae.xor(<1 x i64>, <1 x i64>)
define <1 x i64> @test_xtensa_ae_xor(<1 x i64> %ae_dr_to_dr_v0, <1 x i64> %ae_dr_to_dr_v1) {
; CHECK-LABEL: test_xtensa_ae_xor:


%ret = call <1 x i64> @llvm.xtensa.ae.xor(<1 x i64> %ae_dr_to_dr_v0, <1 x i64> %ae_dr_to_dr_v1)
; CHECK: ae_xor aed{{[0-9]+}}, aed{{[0-9]+}}, aed{{[0-9]+}}

ret <1 x i64> %ret
}


;--- ae_zalign64.ll
declare <8 x i8> @llvm.xtensa.ae.zalign64()
define <8 x i8> @test_xtensa_ae_zalign64() {
; CHECK-LABEL: test_xtensa_ae_zalign64:

%ret = call <8 x i8> @llvm.xtensa.ae.zalign64()
; CHECK: ae_zalign64 u{{[0-9]+}}

ret <8 x i8> %ret
}


;--- rur_ae_bithead.ll
declare i32 @llvm.xtensa.rur.ae.bithead()
define i32 @test_xtensa_rur_ae_bithead() {
; CHECK-LABEL: test_xtensa_rur_ae_bithead:

%ret = call i32 @llvm.xtensa.rur.ae.bithead()
; CHECK: rur.ae_bithead a{{[0-9]+}}

ret i32 %ret
}


;--- rur_ae_bitptr.ll
declare i32 @llvm.xtensa.rur.ae.bitptr()
define i32 @test_xtensa_rur_ae_bitptr() {
; CHECK-LABEL: test_xtensa_rur_ae_bitptr:

%ret = call i32 @llvm.xtensa.rur.ae.bitptr()
; CHECK: rur.ae_bitptr a{{[0-9]+}}

ret i32 %ret
}


;--- rur_ae_bitsused.ll
declare i32 @llvm.xtensa.rur.ae.bitsused()
define i32 @test_xtensa_rur_ae_bitsused() {
; CHECK-LABEL: test_xtensa_rur_ae_bitsused:

%ret = call i32 @llvm.xtensa.rur.ae.bitsused()
; CHECK: rur.ae_bitsused a{{[0-9]+}}

ret i32 %ret
}


;--- rur_ae_cbegin0.ll
declare i32 @llvm.xtensa.rur.ae.cbegin0()
define i32 @test_xtensa_rur_ae_cbegin0() {
; CHECK-LABEL: test_xtensa_rur_ae_cbegin0:

%ret = call i32 @llvm.xtensa.rur.ae.cbegin0()
; CHECK: rur.ae_cbegin0 a{{[0-9]+}}

ret i32 %ret
}


;--- rur_ae_cend0.ll
declare i32 @llvm.xtensa.rur.ae.cend0()
define i32 @test_xtensa_rur_ae_cend0() {
; CHECK-LABEL: test_xtensa_rur_ae_cend0:

%ret = call i32 @llvm.xtensa.rur.ae.cend0()
; CHECK: rur.ae_cend0 a{{[0-9]+}}

ret i32 %ret
}


;--- rur_ae_cw_sd_no.ll
declare i32 @llvm.xtensa.rur.ae.cw.sd.no()
define i32 @test_xtensa_rur_ae_cw_sd_no() {
; CHECK-LABEL: test_xtensa_rur_ae_cw_sd_no:

%ret = call i32 @llvm.xtensa.rur.ae.cw.sd.no()
; CHECK: rur.ae_cw_sd_no a{{[0-9]+}}

ret i32 %ret
}


;--- rur_ae_cwrap.ll
declare i32 @llvm.xtensa.rur.ae.cwrap()
define i32 @test_xtensa_rur_ae_cwrap() {
; CHECK-LABEL: test_xtensa_rur_ae_cwrap:

%ret = call i32 @llvm.xtensa.rur.ae.cwrap()
; CHECK: rur.ae_cwrap a{{[0-9]+}}

ret i32 %ret
}


;--- rur_ae_first_ts.ll
declare i32 @llvm.xtensa.rur.ae.first.ts()
define i32 @test_xtensa_rur_ae_first_ts() {
; CHECK-LABEL: test_xtensa_rur_ae_first_ts:

%ret = call i32 @llvm.xtensa.rur.ae.first.ts()
; CHECK: rur.ae_first_ts a{{[0-9]+}}

ret i32 %ret
}


;--- rur_ae_nextoffset.ll
declare i32 @llvm.xtensa.rur.ae.nextoffset()
define i32 @test_xtensa_rur_ae_nextoffset() {
; CHECK-LABEL: test_xtensa_rur_ae_nextoffset:

%ret = call i32 @llvm.xtensa.rur.ae.nextoffset()
; CHECK: rur.ae_nextoffset a{{[0-9]+}}

ret i32 %ret
}


;--- rur_ae_overflow.ll
declare i32 @llvm.xtensa.rur.ae.overflow()
define i32 @test_xtensa_rur_ae_overflow() {
; CHECK-LABEL: test_xtensa_rur_ae_overflow:

%ret = call i32 @llvm.xtensa.rur.ae.overflow()
; CHECK: rur.ae_overflow a{{[0-9]+}}

ret i32 %ret
}


;--- rur_ae_ovf_sar.ll
declare i32 @llvm.xtensa.rur.ae.ovf.sar()
define i32 @test_xtensa_rur_ae_ovf_sar() {
; CHECK-LABEL: test_xtensa_rur_ae_ovf_sar:

%ret = call i32 @llvm.xtensa.rur.ae.ovf.sar()
; CHECK: rur.ae_ovf_sar a{{[0-9]+}}

ret i32 %ret
}


;--- rur_ae_sar.ll
declare i32 @llvm.xtensa.rur.ae.sar()
define i32 @test_xtensa_rur_ae_sar() {
; CHECK-LABEL: test_xtensa_rur_ae_sar:

%ret = call i32 @llvm.xtensa.rur.ae.sar()
; CHECK: rur.ae_sar a{{[0-9]+}}

ret i32 %ret
}


;--- rur_ae_searchdone.ll
declare i32 @llvm.xtensa.rur.ae.searchdone()
define i32 @test_xtensa_rur_ae_searchdone() {
; CHECK-LABEL: test_xtensa_rur_ae_searchdone:

%ret = call i32 @llvm.xtensa.rur.ae.searchdone()
; CHECK: rur.ae_searchdone a{{[0-9]+}}

ret i32 %ret
}


;--- rur_ae_tablesize.ll
declare i32 @llvm.xtensa.rur.ae.tablesize()
define i32 @test_xtensa_rur_ae_tablesize() {
; CHECK-LABEL: test_xtensa_rur_ae_tablesize:

%ret = call i32 @llvm.xtensa.rur.ae.tablesize()
; CHECK: rur.ae_tablesize a{{[0-9]+}}

ret i32 %ret
}


;--- rur_ae_ts_fts_bu_bp.ll
declare i32 @llvm.xtensa.rur.ae.ts.fts.bu.bp()
define i32 @test_xtensa_rur_ae_ts_fts_bu_bp() {
; CHECK-LABEL: test_xtensa_rur_ae_ts_fts_bu_bp:

%ret = call i32 @llvm.xtensa.rur.ae.ts.fts.bu.bp()
; CHECK: rur.ae_ts_fts_bu_bp a{{[0-9]+}}

ret i32 %ret
}


;--- wur_ae_bithead.ll
declare void @llvm.xtensa.wur.ae.bithead(i32)
define void @test_xtensa_wur_ae_bithead(i32 %art) {
; CHECK-LABEL: test_xtensa_wur_ae_bithead:

 call void @llvm.xtensa.wur.ae.bithead(i32 %art)
; CHECK: wur.ae_bithead a{{[0-9]+}}

ret void
}


;--- wur_ae_bitptr.ll
declare void @llvm.xtensa.wur.ae.bitptr(i32)
define void @test_xtensa_wur_ae_bitptr(i32 %art) {
; CHECK-LABEL: test_xtensa_wur_ae_bitptr:

 call void @llvm.xtensa.wur.ae.bitptr(i32 %art)
; CHECK: wur.ae_bitptr a{{[0-9]+}}

ret void
}


;--- wur_ae_bitsused.ll
declare void @llvm.xtensa.wur.ae.bitsused(i32)
define void @test_xtensa_wur_ae_bitsused(i32 %art) {
; CHECK-LABEL: test_xtensa_wur_ae_bitsused:

 call void @llvm.xtensa.wur.ae.bitsused(i32 %art)
; CHECK: wur.ae_bitsused a{{[0-9]+}}

ret void
}


;--- wur_ae_cbegin0.ll
declare void @llvm.xtensa.wur.ae.cbegin0(i32)
define void @test_xtensa_wur_ae_cbegin0(i32 %art) {
; CHECK-LABEL: test_xtensa_wur_ae_cbegin0:

 call void @llvm.xtensa.wur.ae.cbegin0(i32 %art)
; CHECK: wur.ae_cbegin0 a{{[0-9]+}}

ret void
}


;--- wur_ae_cend0.ll
declare void @llvm.xtensa.wur.ae.cend0(i32)
define void @test_xtensa_wur_ae_cend0(i32 %art) {
; CHECK-LABEL: test_xtensa_wur_ae_cend0:

 call void @llvm.xtensa.wur.ae.cend0(i32 %art)
; CHECK: wur.ae_cend0 a{{[0-9]+}}

ret void
}


;--- wur_ae_cw_sd_no.ll
declare void @llvm.xtensa.wur.ae.cw.sd.no(i32)
define void @test_xtensa_wur_ae_cw_sd_no(i32 %art) {
; CHECK-LABEL: test_xtensa_wur_ae_cw_sd_no:

 call void @llvm.xtensa.wur.ae.cw.sd.no(i32 %art)
; CHECK: wur.ae_cw_sd_no a{{[0-9]+}}

ret void
}


;--- wur_ae_cwrap.ll
declare void @llvm.xtensa.wur.ae.cwrap(i32)
define void @test_xtensa_wur_ae_cwrap(i32 %art) {
; CHECK-LABEL: test_xtensa_wur_ae_cwrap:

 call void @llvm.xtensa.wur.ae.cwrap(i32 %art)
; CHECK: wur.ae_cwrap a{{[0-9]+}}

ret void
}


;--- wur_ae_first_ts.ll
declare void @llvm.xtensa.wur.ae.first.ts(i32)
define void @test_xtensa_wur_ae_first_ts(i32 %art) {
; CHECK-LABEL: test_xtensa_wur_ae_first_ts:

 call void @llvm.xtensa.wur.ae.first.ts(i32 %art)
; CHECK: wur.ae_first_ts a{{[0-9]+}}

ret void
}


;--- wur_ae_nextoffset.ll
declare void @llvm.xtensa.wur.ae.nextoffset(i32)
define void @test_xtensa_wur_ae_nextoffset(i32 %art) {
; CHECK-LABEL: test_xtensa_wur_ae_nextoffset:

 call void @llvm.xtensa.wur.ae.nextoffset(i32 %art)
; CHECK: wur.ae_nextoffset a{{[0-9]+}}

ret void
}


;--- wur_ae_overflow.ll
declare void @llvm.xtensa.wur.ae.overflow(i32)
define void @test_xtensa_wur_ae_overflow(i32 %art) {
; CHECK-LABEL: test_xtensa_wur_ae_overflow:

 call void @llvm.xtensa.wur.ae.overflow(i32 %art)
; CHECK: wur.ae_overflow a{{[0-9]+}}

ret void
}


;--- wur_ae_ovf_sar.ll
declare void @llvm.xtensa.wur.ae.ovf.sar(i32)
define void @test_xtensa_wur_ae_ovf_sar(i32 %art) {
; CHECK-LABEL: test_xtensa_wur_ae_ovf_sar:

 call void @llvm.xtensa.wur.ae.ovf.sar(i32 %art)
; CHECK: wur.ae_ovf_sar a{{[0-9]+}}

ret void
}


;--- wur_ae_sar.ll
declare void @llvm.xtensa.wur.ae.sar(i32)
define void @test_xtensa_wur_ae_sar(i32 %art) {
; CHECK-LABEL: test_xtensa_wur_ae_sar:

 call void @llvm.xtensa.wur.ae.sar(i32 %art)
; CHECK: wur.ae_sar a{{[0-9]+}}

ret void
}


;--- wur_ae_searchdone.ll
declare void @llvm.xtensa.wur.ae.searchdone(i32)
define void @test_xtensa_wur_ae_searchdone(i32 %art) {
; CHECK-LABEL: test_xtensa_wur_ae_searchdone:

 call void @llvm.xtensa.wur.ae.searchdone(i32 %art)
; CHECK: wur.ae_searchdone a{{[0-9]+}}

ret void
}


;--- wur_ae_tablesize.ll
declare void @llvm.xtensa.wur.ae.tablesize(i32)
define void @test_xtensa_wur_ae_tablesize(i32 %art) {
; CHECK-LABEL: test_xtensa_wur_ae_tablesize:

 call void @llvm.xtensa.wur.ae.tablesize(i32 %art)
; CHECK: wur.ae_tablesize a{{[0-9]+}}

ret void
}


;--- wur_ae_ts_fts_bu_bp.ll
declare void @llvm.xtensa.wur.ae.ts.fts.bu.bp(i32)
define void @test_xtensa_wur_ae_ts_fts_bu_bp(i32 %art) {
; CHECK-LABEL: test_xtensa_wur_ae_ts_fts_bu_bp:

 call void @llvm.xtensa.wur.ae.ts.fts.bu.bp(i32 %art)
; CHECK: wur.ae_ts_fts_bu_bp a{{[0-9]+}}

ret void
}


