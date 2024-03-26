// RUN: split-file %s %t
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_abs16s.c | FileCheck %t/ae_abs16s.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_abs24s.c | FileCheck %t/ae_abs24s.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_abs32.c | FileCheck %t/ae_abs32.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_abs32s.c | FileCheck %t/ae_abs32s.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_abs64.c | FileCheck %t/ae_abs64.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_abs64s.c | FileCheck %t/ae_abs64s.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_add16.c | FileCheck %t/ae_add16.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_add16s.c | FileCheck %t/ae_add16s.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_add24s.c | FileCheck %t/ae_add24s.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_add32.c | FileCheck %t/ae_add32.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_add32_hl_lh.c | FileCheck %t/ae_add32_hl_lh.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_add32s.c | FileCheck %t/ae_add32s.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_add64.c | FileCheck %t/ae_add64.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_add64s.c | FileCheck %t/ae_add64s.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_addbrba32.c | FileCheck %t/ae_addbrba32.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_addsub32.c | FileCheck %t/ae_addsub32.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_addsub32s.c | FileCheck %t/ae_addsub32s.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_and.c | FileCheck %t/ae_and.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_cvt32x2f16_10.c | FileCheck %t/ae_cvt32x2f16_10.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_cvt32x2f16_32.c | FileCheck %t/ae_cvt32x2f16_32.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_cvt48a32.c | FileCheck %t/ae_cvt48a32.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_cvt64a32.c | FileCheck %t/ae_cvt64a32.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_cvt64f32_h.c | FileCheck %t/ae_cvt64f32_h.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_cvta32f24s_h.c | FileCheck %t/ae_cvta32f24s_h.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_cvta32f24s_l.c | FileCheck %t/ae_cvta32f24s_l.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_cvtq56a32s.c | FileCheck %t/ae_cvtq56a32s.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_cvtq56p32s_h.c | FileCheck %t/ae_cvtq56p32s_h.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_cvtq56p32s_l.c | FileCheck %t/ae_cvtq56p32s_l.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_db.c | FileCheck %t/ae_db.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_db_ic.c | FileCheck %t/ae_db_ic.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_db_ip.c | FileCheck %t/ae_db_ip.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_dbi.c | FileCheck %t/ae_dbi.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_dbi_ic.c | FileCheck %t/ae_dbi_ic.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_dbi_ip.c | FileCheck %t/ae_dbi_ip.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_div64d32_h.c | FileCheck %t/ae_div64d32_h.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_div64d32_l.c | FileCheck %t/ae_div64d32_l.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_eq16.c | FileCheck %t/ae_eq16.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_eq32.c | FileCheck %t/ae_eq32.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_eq64.c | FileCheck %t/ae_eq64.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l16_i.c | FileCheck %t/ae_l16_i.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l16_ip.c | FileCheck %t/ae_l16_ip.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l16_x.c | FileCheck %t/ae_l16_x.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l16_xc.c | FileCheck %t/ae_l16_xc.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l16_xp.c | FileCheck %t/ae_l16_xp.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l16m_i.c | FileCheck %t/ae_l16m_i.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l16m_iu.c | FileCheck %t/ae_l16m_iu.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l16m_x.c | FileCheck %t/ae_l16m_x.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l16m_xc.c | FileCheck %t/ae_l16m_xc.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l16m_xu.c | FileCheck %t/ae_l16m_xu.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l16x2m_i.c | FileCheck %t/ae_l16x2m_i.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l16x2m_iu.c | FileCheck %t/ae_l16x2m_iu.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l16x2m_x.c | FileCheck %t/ae_l16x2m_x.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l16x2m_xc.c | FileCheck %t/ae_l16x2m_xc.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l16x2m_xu.c | FileCheck %t/ae_l16x2m_xu.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l16x4_i.c | FileCheck %t/ae_l16x4_i.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l16x4_ip.c | FileCheck %t/ae_l16x4_ip.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l16x4_ric.c | FileCheck %t/ae_l16x4_ric.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l16x4_rip.c | FileCheck %t/ae_l16x4_rip.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l16x4_x.c | FileCheck %t/ae_l16x4_x.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l16x4_xc.c | FileCheck %t/ae_l16x4_xc.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l16x4_xp.c | FileCheck %t/ae_l16x4_xp.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l32_i.c | FileCheck %t/ae_l32_i.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l32_ip.c | FileCheck %t/ae_l32_ip.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l32_x.c | FileCheck %t/ae_l32_x.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l32_xc.c | FileCheck %t/ae_l32_xc.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l32_xp.c | FileCheck %t/ae_l32_xp.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l32f24_i.c | FileCheck %t/ae_l32f24_i.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l32f24_ip.c | FileCheck %t/ae_l32f24_ip.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l32f24_x.c | FileCheck %t/ae_l32f24_x.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l32f24_xc.c | FileCheck %t/ae_l32f24_xc.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l32f24_xp.c | FileCheck %t/ae_l32f24_xp.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l32m_i.c | FileCheck %t/ae_l32m_i.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l32m_iu.c | FileCheck %t/ae_l32m_iu.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l32m_x.c | FileCheck %t/ae_l32m_x.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l32m_xc.c | FileCheck %t/ae_l32m_xc.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l32m_xu.c | FileCheck %t/ae_l32m_xu.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l32x2_i.c | FileCheck %t/ae_l32x2_i.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l32x2_ip.c | FileCheck %t/ae_l32x2_ip.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l32x2_ric.c | FileCheck %t/ae_l32x2_ric.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l32x2_rip.c | FileCheck %t/ae_l32x2_rip.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l32x2_x.c | FileCheck %t/ae_l32x2_x.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l32x2_xc.c | FileCheck %t/ae_l32x2_xc.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l32x2_xp.c | FileCheck %t/ae_l32x2_xp.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l32x2f24_i.c | FileCheck %t/ae_l32x2f24_i.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l32x2f24_ip.c | FileCheck %t/ae_l32x2f24_ip.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l32x2f24_ric.c | FileCheck %t/ae_l32x2f24_ric.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l32x2f24_rip.c | FileCheck %t/ae_l32x2f24_rip.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l32x2f24_x.c | FileCheck %t/ae_l32x2f24_x.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l32x2f24_xc.c | FileCheck %t/ae_l32x2f24_xc.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l32x2f24_xp.c | FileCheck %t/ae_l32x2f24_xp.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l64_i.c | FileCheck %t/ae_l64_i.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l64_ip.c | FileCheck %t/ae_l64_ip.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l64_x.c | FileCheck %t/ae_l64_x.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l64_xc.c | FileCheck %t/ae_l64_xc.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_l64_xp.c | FileCheck %t/ae_l64_xp.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_la16x4_ic.c | FileCheck %t/ae_la16x4_ic.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_la16x4_ip.c | FileCheck %t/ae_la16x4_ip.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_la16x4_ric.c | FileCheck %t/ae_la16x4_ric.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_la16x4_rip.c | FileCheck %t/ae_la16x4_rip.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_la16x4neg_pc.c | FileCheck %t/ae_la16x4neg_pc.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_la16x4pos_pc.c | FileCheck %t/ae_la16x4pos_pc.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_la24_ic.c | FileCheck %t/ae_la24_ic.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_la24_ip.c | FileCheck %t/ae_la24_ip.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_la24_ric.c | FileCheck %t/ae_la24_ric.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_la24_rip.c | FileCheck %t/ae_la24_rip.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_la24neg_pc.c | FileCheck %t/ae_la24neg_pc.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_la24pos_pc.c | FileCheck %t/ae_la24pos_pc.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_la24x2_ic.c | FileCheck %t/ae_la24x2_ic.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_la24x2_ip.c | FileCheck %t/ae_la24x2_ip.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_la24x2_ric.c | FileCheck %t/ae_la24x2_ric.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_la24x2_rip.c | FileCheck %t/ae_la24x2_rip.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_la24x2neg_pc.c | FileCheck %t/ae_la24x2neg_pc.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_la24x2pos_pc.c | FileCheck %t/ae_la24x2pos_pc.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_la32x2_ic.c | FileCheck %t/ae_la32x2_ic.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_la32x2_ip.c | FileCheck %t/ae_la32x2_ip.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_la32x2_ric.c | FileCheck %t/ae_la32x2_ric.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_la32x2_rip.c | FileCheck %t/ae_la32x2_rip.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_la32x2f24_ic.c | FileCheck %t/ae_la32x2f24_ic.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_la32x2f24_ip.c | FileCheck %t/ae_la32x2f24_ip.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_la32x2f24_ric.c | FileCheck %t/ae_la32x2f24_ric.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_la32x2f24_rip.c | FileCheck %t/ae_la32x2f24_rip.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_la32x2neg_pc.c | FileCheck %t/ae_la32x2neg_pc.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_la32x2pos_pc.c | FileCheck %t/ae_la32x2pos_pc.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_la64_pp.c | FileCheck %t/ae_la64_pp.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_lalign64_i.c | FileCheck %t/ae_lalign64_i.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_lb.c | FileCheck %t/ae_lb.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_lbi.c | FileCheck %t/ae_lbi.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_lbk.c | FileCheck %t/ae_lbk.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_lbki.c | FileCheck %t/ae_lbki.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_lbs.c | FileCheck %t/ae_lbs.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_lbsi.c | FileCheck %t/ae_lbsi.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_le16.c | FileCheck %t/ae_le16.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_le32.c | FileCheck %t/ae_le32.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_le64.c | FileCheck %t/ae_le64.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_lt16.c | FileCheck %t/ae_lt16.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_lt32.c | FileCheck %t/ae_lt32.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_lt64.c | FileCheck %t/ae_lt64.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_max32.c | FileCheck %t/ae_max32.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_max64.c | FileCheck %t/ae_max64.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_maxabs32s.c | FileCheck %t/ae_maxabs32s.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_maxabs64s.c | FileCheck %t/ae_maxabs64s.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_min32.c | FileCheck %t/ae_min32.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_min64.c | FileCheck %t/ae_min64.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_minabs32s.c | FileCheck %t/ae_minabs32s.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_minabs64s.c | FileCheck %t/ae_minabs64s.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mov.c | FileCheck %t/ae_mov.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_movad16_0.c | FileCheck %t/ae_movad16_0.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_movad16_1.c | FileCheck %t/ae_movad16_1.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_movad16_2.c | FileCheck %t/ae_movad16_2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_movad16_3.c | FileCheck %t/ae_movad16_3.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_movad32_h.c | FileCheck %t/ae_movad32_h.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_movad32_l.c | FileCheck %t/ae_movad32_l.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_movalign.c | FileCheck %t/ae_movalign.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_movda16.c | FileCheck %t/ae_movda16.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_movda16x2.c | FileCheck %t/ae_movda16x2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_movda32.c | FileCheck %t/ae_movda32.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_movda32x2.c | FileCheck %t/ae_movda32x2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_movf16x4.c | FileCheck %t/ae_movf16x4.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_movf32x2.c | FileCheck %t/ae_movf32x2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_movf64.c | FileCheck %t/ae_movf64.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_movi.c | FileCheck %t/ae_movi.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_movt16x4.c | FileCheck %t/ae_movt16x4.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_movt32x2.c | FileCheck %t/ae_movt32x2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_movt64.c | FileCheck %t/ae_movt64.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mul16x4.c | FileCheck %t/ae_mul16x4.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mul32_hh.c | FileCheck %t/ae_mul32_hh.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mul32_lh.c | FileCheck %t/ae_mul32_lh.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mul32_ll.c | FileCheck %t/ae_mul32_ll.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mul32_ll_s2.c | FileCheck %t/ae_mul32_ll_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mul32u_ll.c | FileCheck %t/ae_mul32u_ll.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mul32x16_h0.c | FileCheck %t/ae_mul32x16_h0.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mul32x16_h0_s2.c | FileCheck %t/ae_mul32x16_h0_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mul32x16_h1.c | FileCheck %t/ae_mul32x16_h1.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mul32x16_h1_s2.c | FileCheck %t/ae_mul32x16_h1_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mul32x16_h2.c | FileCheck %t/ae_mul32x16_h2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mul32x16_h2_s2.c | FileCheck %t/ae_mul32x16_h2_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mul32x16_h3.c | FileCheck %t/ae_mul32x16_h3.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mul32x16_h3_s2.c | FileCheck %t/ae_mul32x16_h3_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mul32x16_l0.c | FileCheck %t/ae_mul32x16_l0.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mul32x16_l0_s2.c | FileCheck %t/ae_mul32x16_l0_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mul32x16_l1.c | FileCheck %t/ae_mul32x16_l1.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mul32x16_l1_s2.c | FileCheck %t/ae_mul32x16_l1_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mul32x16_l2.c | FileCheck %t/ae_mul32x16_l2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mul32x16_l2_s2.c | FileCheck %t/ae_mul32x16_l2_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mul32x16_l3.c | FileCheck %t/ae_mul32x16_l3.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mul32x16_l3_s2.c | FileCheck %t/ae_mul32x16_l3_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mula16x4.c | FileCheck %t/ae_mula16x4.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mula32_hh.c | FileCheck %t/ae_mula32_hh.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mula32_lh.c | FileCheck %t/ae_mula32_lh.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mula32_ll.c | FileCheck %t/ae_mula32_ll.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mula32_ll_s2.c | FileCheck %t/ae_mula32_ll_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mula32u_ll.c | FileCheck %t/ae_mula32u_ll.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mula32x16_h0.c | FileCheck %t/ae_mula32x16_h0.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mula32x16_h0_s2.c | FileCheck %t/ae_mula32x16_h0_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mula32x16_h1.c | FileCheck %t/ae_mula32x16_h1.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mula32x16_h1_s2.c | FileCheck %t/ae_mula32x16_h1_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mula32x16_h2.c | FileCheck %t/ae_mula32x16_h2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mula32x16_h2_s2.c | FileCheck %t/ae_mula32x16_h2_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mula32x16_h3.c | FileCheck %t/ae_mula32x16_h3.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mula32x16_h3_s2.c | FileCheck %t/ae_mula32x16_h3_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mula32x16_l0.c | FileCheck %t/ae_mula32x16_l0.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mula32x16_l0_s2.c | FileCheck %t/ae_mula32x16_l0_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mula32x16_l1.c | FileCheck %t/ae_mula32x16_l1.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mula32x16_l1_s2.c | FileCheck %t/ae_mula32x16_l1_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mula32x16_l2.c | FileCheck %t/ae_mula32x16_l2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mula32x16_l2_s2.c | FileCheck %t/ae_mula32x16_l2_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mula32x16_l3.c | FileCheck %t/ae_mula32x16_l3.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mula32x16_l3_s2.c | FileCheck %t/ae_mula32x16_l3_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaad24_hh_ll.c | FileCheck %t/ae_mulaad24_hh_ll.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaad24_hh_ll_s2.c | FileCheck %t/ae_mulaad24_hh_ll_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaad24_hl_lh.c | FileCheck %t/ae_mulaad24_hl_lh.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaad24_hl_lh_s2.c | FileCheck %t/ae_mulaad24_hl_lh_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaad32x16_h0_l1.c | FileCheck %t/ae_mulaad32x16_h0_l1.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaad32x16_h0_l1_s2.c | FileCheck %t/ae_mulaad32x16_h0_l1_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaad32x16_h1_l0.c | FileCheck %t/ae_mulaad32x16_h1_l0.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaad32x16_h1_l0_s2.c | FileCheck %t/ae_mulaad32x16_h1_l0_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaad32x16_h2_l3.c | FileCheck %t/ae_mulaad32x16_h2_l3.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaad32x16_h2_l3_s2.c | FileCheck %t/ae_mulaad32x16_h2_l3_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaad32x16_h3_l2.c | FileCheck %t/ae_mulaad32x16_h3_l2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaad32x16_h3_l2_s2.c | FileCheck %t/ae_mulaad32x16_h3_l2_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaafd16ss_11_00.c | FileCheck %t/ae_mulaafd16ss_11_00.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaafd16ss_11_00_s2.c | FileCheck %t/ae_mulaafd16ss_11_00_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaafd16ss_13_02.c | FileCheck %t/ae_mulaafd16ss_13_02.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaafd16ss_13_02_s2.c | FileCheck %t/ae_mulaafd16ss_13_02_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaafd16ss_33_22.c | FileCheck %t/ae_mulaafd16ss_33_22.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaafd16ss_33_22_s2.c | FileCheck %t/ae_mulaafd16ss_33_22_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaafd24_hh_ll.c | FileCheck %t/ae_mulaafd24_hh_ll.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaafd24_hh_ll_s2.c | FileCheck %t/ae_mulaafd24_hh_ll_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaafd24_hl_lh.c | FileCheck %t/ae_mulaafd24_hl_lh.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaafd24_hl_lh_s2.c | FileCheck %t/ae_mulaafd24_hl_lh_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaafd32x16_h0_l1.c | FileCheck %t/ae_mulaafd32x16_h0_l1.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaafd32x16_h0_l1_s2.c | FileCheck %t/ae_mulaafd32x16_h0_l1_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaafd32x16_h1_l0.c | FileCheck %t/ae_mulaafd32x16_h1_l0.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaafd32x16_h1_l0_s2.c | FileCheck %t/ae_mulaafd32x16_h1_l0_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaafd32x16_h2_l3.c | FileCheck %t/ae_mulaafd32x16_h2_l3.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaafd32x16_h2_l3_s2.c | FileCheck %t/ae_mulaafd32x16_h2_l3_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaafd32x16_h3_l2.c | FileCheck %t/ae_mulaafd32x16_h3_l2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaafd32x16_h3_l2_s2.c | FileCheck %t/ae_mulaafd32x16_h3_l2_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulac24.c | FileCheck %t/ae_mulac24.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulac32x16_h.c | FileCheck %t/ae_mulac32x16_h.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulac32x16_l.c | FileCheck %t/ae_mulac32x16_l.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaf16ss_00.c | FileCheck %t/ae_mulaf16ss_00.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaf16ss_00_s2.c | FileCheck %t/ae_mulaf16ss_00_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaf16ss_10.c | FileCheck %t/ae_mulaf16ss_10.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaf16ss_11.c | FileCheck %t/ae_mulaf16ss_11.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaf16ss_20.c | FileCheck %t/ae_mulaf16ss_20.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaf16ss_21.c | FileCheck %t/ae_mulaf16ss_21.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaf16ss_22.c | FileCheck %t/ae_mulaf16ss_22.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaf16ss_30.c | FileCheck %t/ae_mulaf16ss_30.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaf16ss_31.c | FileCheck %t/ae_mulaf16ss_31.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaf16ss_32.c | FileCheck %t/ae_mulaf16ss_32.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaf16ss_33.c | FileCheck %t/ae_mulaf16ss_33.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaf16x4ss.c | FileCheck %t/ae_mulaf16x4ss.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaf32r_hh.c | FileCheck %t/ae_mulaf32r_hh.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaf32r_lh.c | FileCheck %t/ae_mulaf32r_lh.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaf32r_ll.c | FileCheck %t/ae_mulaf32r_ll.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaf32r_ll_s2.c | FileCheck %t/ae_mulaf32r_ll_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaf32s_hh.c | FileCheck %t/ae_mulaf32s_hh.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaf32s_lh.c | FileCheck %t/ae_mulaf32s_lh.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaf32s_ll.c | FileCheck %t/ae_mulaf32s_ll.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaf32s_ll_s2.c | FileCheck %t/ae_mulaf32s_ll_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaf32x16_h0.c | FileCheck %t/ae_mulaf32x16_h0.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaf32x16_h0_s2.c | FileCheck %t/ae_mulaf32x16_h0_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaf32x16_h1.c | FileCheck %t/ae_mulaf32x16_h1.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaf32x16_h1_s2.c | FileCheck %t/ae_mulaf32x16_h1_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaf32x16_h2.c | FileCheck %t/ae_mulaf32x16_h2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaf32x16_h2_s2.c | FileCheck %t/ae_mulaf32x16_h2_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaf32x16_h3.c | FileCheck %t/ae_mulaf32x16_h3.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaf32x16_h3_s2.c | FileCheck %t/ae_mulaf32x16_h3_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaf32x16_l0.c | FileCheck %t/ae_mulaf32x16_l0.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaf32x16_l0_s2.c | FileCheck %t/ae_mulaf32x16_l0_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaf32x16_l1.c | FileCheck %t/ae_mulaf32x16_l1.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaf32x16_l1_s2.c | FileCheck %t/ae_mulaf32x16_l1_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaf32x16_l2.c | FileCheck %t/ae_mulaf32x16_l2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaf32x16_l2_s2.c | FileCheck %t/ae_mulaf32x16_l2_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaf32x16_l3.c | FileCheck %t/ae_mulaf32x16_l3.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaf32x16_l3_s2.c | FileCheck %t/ae_mulaf32x16_l3_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaf48q32sp16s_l.c | FileCheck %t/ae_mulaf48q32sp16s_l.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaf48q32sp16s_l_s2.c | FileCheck %t/ae_mulaf48q32sp16s_l_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaf48q32sp16u_l.c | FileCheck %t/ae_mulaf48q32sp16u_l.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaf48q32sp16u_l_s2.c | FileCheck %t/ae_mulaf48q32sp16u_l_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulafc24ra.c | FileCheck %t/ae_mulafc24ra.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulafc32x16ras_h.c | FileCheck %t/ae_mulafc32x16ras_h.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulafc32x16ras_l.c | FileCheck %t/ae_mulafc32x16ras_l.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulafd24x2_fir_h.c | FileCheck %t/ae_mulafd24x2_fir_h.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulafd24x2_fir_l.c | FileCheck %t/ae_mulafd24x2_fir_l.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulafd32x16x2_fir_hh.c | FileCheck %t/ae_mulafd32x16x2_fir_hh.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulafd32x16x2_fir_hl.c | FileCheck %t/ae_mulafd32x16x2_fir_hl.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulafd32x16x2_fir_lh.c | FileCheck %t/ae_mulafd32x16x2_fir_lh.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulafd32x16x2_fir_ll.c | FileCheck %t/ae_mulafd32x16x2_fir_ll.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulafp24x2r.c | FileCheck %t/ae_mulafp24x2r.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulafp24x2r_s2.c | FileCheck %t/ae_mulafp24x2r_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulafp24x2ra.c | FileCheck %t/ae_mulafp24x2ra.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulafp24x2ra_s2.c | FileCheck %t/ae_mulafp24x2ra_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulafp32x16x2ras_h.c | FileCheck %t/ae_mulafp32x16x2ras_h.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulafp32x16x2ras_h_s2.c | FileCheck %t/ae_mulafp32x16x2ras_h_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulafp32x16x2ras_l.c | FileCheck %t/ae_mulafp32x16x2ras_l.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulafp32x16x2ras_l_s2.c | FileCheck %t/ae_mulafp32x16x2ras_l_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulafp32x16x2rs_h.c | FileCheck %t/ae_mulafp32x16x2rs_h.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulafp32x16x2rs_h_s2.c | FileCheck %t/ae_mulafp32x16x2rs_h_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulafp32x16x2rs_l.c | FileCheck %t/ae_mulafp32x16x2rs_l.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulafp32x16x2rs_l_s2.c | FileCheck %t/ae_mulafp32x16x2rs_l_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulafp32x2ras.c | FileCheck %t/ae_mulafp32x2ras.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulafp32x2rs.c | FileCheck %t/ae_mulafp32x2rs.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulafq32sp24s_h_s2.c | FileCheck %t/ae_mulafq32sp24s_h_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulafq32sp24s_l_s2.c | FileCheck %t/ae_mulafq32sp24s_l_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulap24x2.c | FileCheck %t/ae_mulap24x2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulap24x2_s2.c | FileCheck %t/ae_mulap24x2_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulap32x16x2_h.c | FileCheck %t/ae_mulap32x16x2_h.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulap32x16x2_l.c | FileCheck %t/ae_mulap32x16x2_l.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulap32x2.c | FileCheck %t/ae_mulap32x2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaq32sp16s_l_s2.c | FileCheck %t/ae_mulaq32sp16s_l_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulaq32sp16u_l_s2.c | FileCheck %t/ae_mulaq32sp16u_l_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mularfq32sp24s_h_s2.c | FileCheck %t/ae_mularfq32sp24s_h_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mularfq32sp24s_l_s2.c | FileCheck %t/ae_mularfq32sp24s_l_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulas32f48p16s_hh.c | FileCheck %t/ae_mulas32f48p16s_hh.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulas32f48p16s_hh_s2.c | FileCheck %t/ae_mulas32f48p16s_hh_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulas32f48p16s_lh.c | FileCheck %t/ae_mulas32f48p16s_lh.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulas32f48p16s_lh_s2.c | FileCheck %t/ae_mulas32f48p16s_lh_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulas32f48p16s_ll.c | FileCheck %t/ae_mulas32f48p16s_ll.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulas32f48p16s_ll_s2.c | FileCheck %t/ae_mulas32f48p16s_ll_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulasd24_hh_ll.c | FileCheck %t/ae_mulasd24_hh_ll.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulasd24_hh_ll_s2.c | FileCheck %t/ae_mulasd24_hh_ll_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulasd24_hl_lh.c | FileCheck %t/ae_mulasd24_hl_lh.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulasd24_hl_lh_s2.c | FileCheck %t/ae_mulasd24_hl_lh_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulasd32x16_h1_l0.c | FileCheck %t/ae_mulasd32x16_h1_l0.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulasd32x16_h1_l0_s2.c | FileCheck %t/ae_mulasd32x16_h1_l0_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulasd32x16_h3_l2.c | FileCheck %t/ae_mulasd32x16_h3_l2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulasd32x16_h3_l2_s2.c | FileCheck %t/ae_mulasd32x16_h3_l2_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulasfd24_hh_ll.c | FileCheck %t/ae_mulasfd24_hh_ll.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulasfd24_hh_ll_s2.c | FileCheck %t/ae_mulasfd24_hh_ll_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulasfd24_hl_lh.c | FileCheck %t/ae_mulasfd24_hl_lh.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulasfd24_hl_lh_s2.c | FileCheck %t/ae_mulasfd24_hl_lh_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulasfd32x16_h1_l0.c | FileCheck %t/ae_mulasfd32x16_h1_l0.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulasfd32x16_h1_l0_s2.c | FileCheck %t/ae_mulasfd32x16_h1_l0_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulasfd32x16_h3_l2.c | FileCheck %t/ae_mulasfd32x16_h3_l2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulasfd32x16_h3_l2_s2.c | FileCheck %t/ae_mulasfd32x16_h3_l2_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulc24.c | FileCheck %t/ae_mulc24.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulc32x16_h.c | FileCheck %t/ae_mulc32x16_h.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulc32x16_l.c | FileCheck %t/ae_mulc32x16_l.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulf16ss_00.c | FileCheck %t/ae_mulf16ss_00.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulf16ss_00_s2.c | FileCheck %t/ae_mulf16ss_00_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulf16ss_10.c | FileCheck %t/ae_mulf16ss_10.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulf16ss_11.c | FileCheck %t/ae_mulf16ss_11.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulf16ss_20.c | FileCheck %t/ae_mulf16ss_20.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulf16ss_21.c | FileCheck %t/ae_mulf16ss_21.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulf16ss_22.c | FileCheck %t/ae_mulf16ss_22.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulf16ss_30.c | FileCheck %t/ae_mulf16ss_30.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulf16ss_31.c | FileCheck %t/ae_mulf16ss_31.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulf16ss_32.c | FileCheck %t/ae_mulf16ss_32.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulf16ss_33.c | FileCheck %t/ae_mulf16ss_33.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulf16x4ss.c | FileCheck %t/ae_mulf16x4ss.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulf32r_hh.c | FileCheck %t/ae_mulf32r_hh.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulf32r_lh.c | FileCheck %t/ae_mulf32r_lh.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulf32r_ll.c | FileCheck %t/ae_mulf32r_ll.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulf32r_ll_s2.c | FileCheck %t/ae_mulf32r_ll_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulf32s_hh.c | FileCheck %t/ae_mulf32s_hh.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulf32s_lh.c | FileCheck %t/ae_mulf32s_lh.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulf32s_ll.c | FileCheck %t/ae_mulf32s_ll.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulf32s_ll_s2.c | FileCheck %t/ae_mulf32s_ll_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulf32x16_h0.c | FileCheck %t/ae_mulf32x16_h0.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulf32x16_h0_s2.c | FileCheck %t/ae_mulf32x16_h0_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulf32x16_h1.c | FileCheck %t/ae_mulf32x16_h1.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulf32x16_h1_s2.c | FileCheck %t/ae_mulf32x16_h1_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulf32x16_h2.c | FileCheck %t/ae_mulf32x16_h2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulf32x16_h2_s2.c | FileCheck %t/ae_mulf32x16_h2_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulf32x16_h3.c | FileCheck %t/ae_mulf32x16_h3.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulf32x16_h3_s2.c | FileCheck %t/ae_mulf32x16_h3_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulf32x16_l0.c | FileCheck %t/ae_mulf32x16_l0.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulf32x16_l0_s2.c | FileCheck %t/ae_mulf32x16_l0_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulf32x16_l1.c | FileCheck %t/ae_mulf32x16_l1.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulf32x16_l1_s2.c | FileCheck %t/ae_mulf32x16_l1_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulf32x16_l2.c | FileCheck %t/ae_mulf32x16_l2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulf32x16_l2_s2.c | FileCheck %t/ae_mulf32x16_l2_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulf32x16_l3.c | FileCheck %t/ae_mulf32x16_l3.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulf32x16_l3_s2.c | FileCheck %t/ae_mulf32x16_l3_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulf48q32sp16s_l.c | FileCheck %t/ae_mulf48q32sp16s_l.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulf48q32sp16s_l_s2.c | FileCheck %t/ae_mulf48q32sp16s_l_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulf48q32sp16u_l.c | FileCheck %t/ae_mulf48q32sp16u_l.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulf48q32sp16u_l_s2.c | FileCheck %t/ae_mulf48q32sp16u_l_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulfc24ra.c | FileCheck %t/ae_mulfc24ra.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulfc32x16ras_h.c | FileCheck %t/ae_mulfc32x16ras_h.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulfc32x16ras_l.c | FileCheck %t/ae_mulfc32x16ras_l.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulfd24x2_fir_h.c | FileCheck %t/ae_mulfd24x2_fir_h.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulfd24x2_fir_l.c | FileCheck %t/ae_mulfd24x2_fir_l.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulfd32x16x2_fir_hh.c | FileCheck %t/ae_mulfd32x16x2_fir_hh.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulfd32x16x2_fir_hl.c | FileCheck %t/ae_mulfd32x16x2_fir_hl.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulfd32x16x2_fir_lh.c | FileCheck %t/ae_mulfd32x16x2_fir_lh.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulfd32x16x2_fir_ll.c | FileCheck %t/ae_mulfd32x16x2_fir_ll.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulfp16x4ras.c | FileCheck %t/ae_mulfp16x4ras.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulfp16x4s.c | FileCheck %t/ae_mulfp16x4s.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulfp24x2r.c | FileCheck %t/ae_mulfp24x2r.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulfp24x2r_s2.c | FileCheck %t/ae_mulfp24x2r_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulfp24x2ra.c | FileCheck %t/ae_mulfp24x2ra.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulfp24x2ra_s2.c | FileCheck %t/ae_mulfp24x2ra_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulfp32x16x2ras_h.c | FileCheck %t/ae_mulfp32x16x2ras_h.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulfp32x16x2ras_h_s2.c | FileCheck %t/ae_mulfp32x16x2ras_h_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulfp32x16x2ras_l.c | FileCheck %t/ae_mulfp32x16x2ras_l.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulfp32x16x2ras_l_s2.c | FileCheck %t/ae_mulfp32x16x2ras_l_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulfp32x16x2rs_h.c | FileCheck %t/ae_mulfp32x16x2rs_h.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulfp32x16x2rs_h_s2.c | FileCheck %t/ae_mulfp32x16x2rs_h_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulfp32x16x2rs_l.c | FileCheck %t/ae_mulfp32x16x2rs_l.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulfp32x16x2rs_l_s2.c | FileCheck %t/ae_mulfp32x16x2rs_l_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulfp32x2ras.c | FileCheck %t/ae_mulfp32x2ras.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulfp32x2rs.c | FileCheck %t/ae_mulfp32x2rs.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulfq32sp24s_h_s2.c | FileCheck %t/ae_mulfq32sp24s_h_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulfq32sp24s_l_s2.c | FileCheck %t/ae_mulfq32sp24s_l_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulp24x2.c | FileCheck %t/ae_mulp24x2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulp24x2_s2.c | FileCheck %t/ae_mulp24x2_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulp32x16x2_h.c | FileCheck %t/ae_mulp32x16x2_h.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulp32x16x2_l.c | FileCheck %t/ae_mulp32x16x2_l.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulp32x2.c | FileCheck %t/ae_mulp32x2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulq32sp16s_l_s2.c | FileCheck %t/ae_mulq32sp16s_l_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulq32sp16u_l_s2.c | FileCheck %t/ae_mulq32sp16u_l_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulrfq32sp24s_h_s2.c | FileCheck %t/ae_mulrfq32sp24s_h_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulrfq32sp24s_l_s2.c | FileCheck %t/ae_mulrfq32sp24s_l_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_muls16x4.c | FileCheck %t/ae_muls16x4.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_muls32_hh.c | FileCheck %t/ae_muls32_hh.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_muls32_lh.c | FileCheck %t/ae_muls32_lh.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_muls32_ll.c | FileCheck %t/ae_muls32_ll.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_muls32f48p16s_hh.c | FileCheck %t/ae_muls32f48p16s_hh.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_muls32f48p16s_hh_s2.c | FileCheck %t/ae_muls32f48p16s_hh_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_muls32f48p16s_lh.c | FileCheck %t/ae_muls32f48p16s_lh.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_muls32f48p16s_lh_s2.c | FileCheck %t/ae_muls32f48p16s_lh_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_muls32f48p16s_ll.c | FileCheck %t/ae_muls32f48p16s_ll.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_muls32f48p16s_ll_s2.c | FileCheck %t/ae_muls32f48p16s_ll_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_muls32u_ll.c | FileCheck %t/ae_muls32u_ll.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_muls32x16_h0.c | FileCheck %t/ae_muls32x16_h0.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_muls32x16_h0_s2.c | FileCheck %t/ae_muls32x16_h0_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_muls32x16_h1.c | FileCheck %t/ae_muls32x16_h1.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_muls32x16_h1_s2.c | FileCheck %t/ae_muls32x16_h1_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_muls32x16_h2.c | FileCheck %t/ae_muls32x16_h2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_muls32x16_h2_s2.c | FileCheck %t/ae_muls32x16_h2_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_muls32x16_h3.c | FileCheck %t/ae_muls32x16_h3.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_muls32x16_h3_s2.c | FileCheck %t/ae_muls32x16_h3_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_muls32x16_l0.c | FileCheck %t/ae_muls32x16_l0.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_muls32x16_l0_s2.c | FileCheck %t/ae_muls32x16_l0_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_muls32x16_l1.c | FileCheck %t/ae_muls32x16_l1.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_muls32x16_l1_s2.c | FileCheck %t/ae_muls32x16_l1_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_muls32x16_l2.c | FileCheck %t/ae_muls32x16_l2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_muls32x16_l2_s2.c | FileCheck %t/ae_muls32x16_l2_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_muls32x16_l3.c | FileCheck %t/ae_muls32x16_l3.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_muls32x16_l3_s2.c | FileCheck %t/ae_muls32x16_l3_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsad24_hh_ll.c | FileCheck %t/ae_mulsad24_hh_ll.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsad24_hh_ll_s2.c | FileCheck %t/ae_mulsad24_hh_ll_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsad32x16_h1_l0.c | FileCheck %t/ae_mulsad32x16_h1_l0.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsad32x16_h1_l0_s2.c | FileCheck %t/ae_mulsad32x16_h1_l0_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsad32x16_h3_l2.c | FileCheck %t/ae_mulsad32x16_h3_l2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsad32x16_h3_l2_s2.c | FileCheck %t/ae_mulsad32x16_h3_l2_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsafd24_hh_ll.c | FileCheck %t/ae_mulsafd24_hh_ll.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsafd24_hh_ll_s2.c | FileCheck %t/ae_mulsafd24_hh_ll_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsafd32x16_h1_l0.c | FileCheck %t/ae_mulsafd32x16_h1_l0.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsafd32x16_h1_l0_s2.c | FileCheck %t/ae_mulsafd32x16_h1_l0_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsafd32x16_h3_l2.c | FileCheck %t/ae_mulsafd32x16_h3_l2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsafd32x16_h3_l2_s2.c | FileCheck %t/ae_mulsafd32x16_h3_l2_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsf16ss_00.c | FileCheck %t/ae_mulsf16ss_00.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsf16ss_00_s2.c | FileCheck %t/ae_mulsf16ss_00_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsf16ss_10.c | FileCheck %t/ae_mulsf16ss_10.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsf16ss_11.c | FileCheck %t/ae_mulsf16ss_11.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsf16ss_20.c | FileCheck %t/ae_mulsf16ss_20.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsf16ss_21.c | FileCheck %t/ae_mulsf16ss_21.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsf16ss_22.c | FileCheck %t/ae_mulsf16ss_22.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsf16ss_30.c | FileCheck %t/ae_mulsf16ss_30.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsf16ss_31.c | FileCheck %t/ae_mulsf16ss_31.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsf16ss_32.c | FileCheck %t/ae_mulsf16ss_32.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsf16ss_33.c | FileCheck %t/ae_mulsf16ss_33.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsf16x4ss.c | FileCheck %t/ae_mulsf16x4ss.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsf32r_hh.c | FileCheck %t/ae_mulsf32r_hh.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsf32r_lh.c | FileCheck %t/ae_mulsf32r_lh.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsf32r_ll.c | FileCheck %t/ae_mulsf32r_ll.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsf32r_ll_s2.c | FileCheck %t/ae_mulsf32r_ll_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsf32s_hh.c | FileCheck %t/ae_mulsf32s_hh.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsf32s_lh.c | FileCheck %t/ae_mulsf32s_lh.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsf32s_ll.c | FileCheck %t/ae_mulsf32s_ll.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsf32x16_h0.c | FileCheck %t/ae_mulsf32x16_h0.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsf32x16_h0_s2.c | FileCheck %t/ae_mulsf32x16_h0_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsf32x16_h1.c | FileCheck %t/ae_mulsf32x16_h1.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsf32x16_h1_s2.c | FileCheck %t/ae_mulsf32x16_h1_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsf32x16_h2.c | FileCheck %t/ae_mulsf32x16_h2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsf32x16_h2_s2.c | FileCheck %t/ae_mulsf32x16_h2_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsf32x16_h3.c | FileCheck %t/ae_mulsf32x16_h3.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsf32x16_h3_s2.c | FileCheck %t/ae_mulsf32x16_h3_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsf32x16_l0.c | FileCheck %t/ae_mulsf32x16_l0.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsf32x16_l0_s2.c | FileCheck %t/ae_mulsf32x16_l0_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsf32x16_l1.c | FileCheck %t/ae_mulsf32x16_l1.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsf32x16_l1_s2.c | FileCheck %t/ae_mulsf32x16_l1_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsf32x16_l2.c | FileCheck %t/ae_mulsf32x16_l2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsf32x16_l2_s2.c | FileCheck %t/ae_mulsf32x16_l2_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsf32x16_l3.c | FileCheck %t/ae_mulsf32x16_l3.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsf32x16_l3_s2.c | FileCheck %t/ae_mulsf32x16_l3_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsf48q32sp16s_l.c | FileCheck %t/ae_mulsf48q32sp16s_l.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsf48q32sp16s_l_s2.c | FileCheck %t/ae_mulsf48q32sp16s_l_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsf48q32sp16u_l.c | FileCheck %t/ae_mulsf48q32sp16u_l.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsf48q32sp16u_l_s2.c | FileCheck %t/ae_mulsf48q32sp16u_l_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsfp24x2r.c | FileCheck %t/ae_mulsfp24x2r.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsfp24x2r_s2.c | FileCheck %t/ae_mulsfp24x2r_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsfp24x2ra.c | FileCheck %t/ae_mulsfp24x2ra.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsfp24x2ra_s2.c | FileCheck %t/ae_mulsfp24x2ra_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsfp32x16x2ras_h.c | FileCheck %t/ae_mulsfp32x16x2ras_h.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsfp32x16x2ras_h_s2.c | FileCheck %t/ae_mulsfp32x16x2ras_h_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsfp32x16x2ras_l.c | FileCheck %t/ae_mulsfp32x16x2ras_l.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsfp32x16x2ras_l_s2.c | FileCheck %t/ae_mulsfp32x16x2ras_l_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsfp32x16x2rs_h.c | FileCheck %t/ae_mulsfp32x16x2rs_h.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsfp32x16x2rs_h_s2.c | FileCheck %t/ae_mulsfp32x16x2rs_h_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsfp32x16x2rs_l.c | FileCheck %t/ae_mulsfp32x16x2rs_l.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsfp32x16x2rs_l_s2.c | FileCheck %t/ae_mulsfp32x16x2rs_l_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsfp32x2ras.c | FileCheck %t/ae_mulsfp32x2ras.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsfp32x2rs.c | FileCheck %t/ae_mulsfp32x2rs.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsfq32sp24s_h_s2.c | FileCheck %t/ae_mulsfq32sp24s_h_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsfq32sp24s_l_s2.c | FileCheck %t/ae_mulsfq32sp24s_l_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsp24x2.c | FileCheck %t/ae_mulsp24x2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsp24x2_s2.c | FileCheck %t/ae_mulsp24x2_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsp32x16x2_h.c | FileCheck %t/ae_mulsp32x16x2_h.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsp32x16x2_l.c | FileCheck %t/ae_mulsp32x16x2_l.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsp32x2.c | FileCheck %t/ae_mulsp32x2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsq32sp16s_l_s2.c | FileCheck %t/ae_mulsq32sp16s_l_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsq32sp16u_l_s2.c | FileCheck %t/ae_mulsq32sp16u_l_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsrfq32sp24s_h_s2.c | FileCheck %t/ae_mulsrfq32sp24s_h_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulsrfq32sp24s_l_s2.c | FileCheck %t/ae_mulsrfq32sp24s_l_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulss32f48p16s_hh.c | FileCheck %t/ae_mulss32f48p16s_hh.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulss32f48p16s_hh_s2.c | FileCheck %t/ae_mulss32f48p16s_hh_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulss32f48p16s_lh.c | FileCheck %t/ae_mulss32f48p16s_lh.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulss32f48p16s_lh_s2.c | FileCheck %t/ae_mulss32f48p16s_lh_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulss32f48p16s_ll.c | FileCheck %t/ae_mulss32f48p16s_ll.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulss32f48p16s_ll_s2.c | FileCheck %t/ae_mulss32f48p16s_ll_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulssd24_hh_ll.c | FileCheck %t/ae_mulssd24_hh_ll.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulssd24_hh_ll_s2.c | FileCheck %t/ae_mulssd24_hh_ll_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulssd24_hl_lh.c | FileCheck %t/ae_mulssd24_hl_lh.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulssd24_hl_lh_s2.c | FileCheck %t/ae_mulssd24_hl_lh_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulssd32x16_h1_l0.c | FileCheck %t/ae_mulssd32x16_h1_l0.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulssd32x16_h1_l0_s2.c | FileCheck %t/ae_mulssd32x16_h1_l0_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulssd32x16_h3_l2.c | FileCheck %t/ae_mulssd32x16_h3_l2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulssd32x16_h3_l2_s2.c | FileCheck %t/ae_mulssd32x16_h3_l2_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulssfd16ss_11_00.c | FileCheck %t/ae_mulssfd16ss_11_00.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulssfd16ss_11_00_s2.c | FileCheck %t/ae_mulssfd16ss_11_00_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulssfd16ss_13_02.c | FileCheck %t/ae_mulssfd16ss_13_02.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulssfd16ss_13_02_s2.c | FileCheck %t/ae_mulssfd16ss_13_02_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulssfd16ss_33_22.c | FileCheck %t/ae_mulssfd16ss_33_22.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulssfd16ss_33_22_s2.c | FileCheck %t/ae_mulssfd16ss_33_22_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulssfd24_hh_ll.c | FileCheck %t/ae_mulssfd24_hh_ll.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulssfd24_hh_ll_s2.c | FileCheck %t/ae_mulssfd24_hh_ll_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulssfd24_hl_lh.c | FileCheck %t/ae_mulssfd24_hl_lh.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulssfd24_hl_lh_s2.c | FileCheck %t/ae_mulssfd24_hl_lh_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulssfd32x16_h1_l0.c | FileCheck %t/ae_mulssfd32x16_h1_l0.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulssfd32x16_h1_l0_s2.c | FileCheck %t/ae_mulssfd32x16_h1_l0_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulssfd32x16_h3_l2.c | FileCheck %t/ae_mulssfd32x16_h3_l2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulssfd32x16_h3_l2_s2.c | FileCheck %t/ae_mulssfd32x16_h3_l2_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzaad24_hh_ll.c | FileCheck %t/ae_mulzaad24_hh_ll.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzaad24_hh_ll_s2.c | FileCheck %t/ae_mulzaad24_hh_ll_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzaad24_hl_lh.c | FileCheck %t/ae_mulzaad24_hl_lh.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzaad24_hl_lh_s2.c | FileCheck %t/ae_mulzaad24_hl_lh_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzaad32x16_h0_l1.c | FileCheck %t/ae_mulzaad32x16_h0_l1.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzaad32x16_h0_l1_s2.c | FileCheck %t/ae_mulzaad32x16_h0_l1_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzaad32x16_h1_l0.c | FileCheck %t/ae_mulzaad32x16_h1_l0.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzaad32x16_h1_l0_s2.c | FileCheck %t/ae_mulzaad32x16_h1_l0_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzaad32x16_h2_l3.c | FileCheck %t/ae_mulzaad32x16_h2_l3.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzaad32x16_h2_l3_s2.c | FileCheck %t/ae_mulzaad32x16_h2_l3_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzaad32x16_h3_l2.c | FileCheck %t/ae_mulzaad32x16_h3_l2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzaad32x16_h3_l2_s2.c | FileCheck %t/ae_mulzaad32x16_h3_l2_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzaafd16ss_11_00.c | FileCheck %t/ae_mulzaafd16ss_11_00.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzaafd16ss_11_00_s2.c | FileCheck %t/ae_mulzaafd16ss_11_00_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzaafd16ss_13_02.c | FileCheck %t/ae_mulzaafd16ss_13_02.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzaafd16ss_13_02_s2.c | FileCheck %t/ae_mulzaafd16ss_13_02_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzaafd16ss_33_22.c | FileCheck %t/ae_mulzaafd16ss_33_22.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzaafd16ss_33_22_s2.c | FileCheck %t/ae_mulzaafd16ss_33_22_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzaafd24_hh_ll.c | FileCheck %t/ae_mulzaafd24_hh_ll.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzaafd24_hh_ll_s2.c | FileCheck %t/ae_mulzaafd24_hh_ll_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzaafd24_hl_lh.c | FileCheck %t/ae_mulzaafd24_hl_lh.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzaafd24_hl_lh_s2.c | FileCheck %t/ae_mulzaafd24_hl_lh_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzaafd32x16_h0_l1.c | FileCheck %t/ae_mulzaafd32x16_h0_l1.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzaafd32x16_h0_l1_s2.c | FileCheck %t/ae_mulzaafd32x16_h0_l1_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzaafd32x16_h1_l0.c | FileCheck %t/ae_mulzaafd32x16_h1_l0.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzaafd32x16_h1_l0_s2.c | FileCheck %t/ae_mulzaafd32x16_h1_l0_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzaafd32x16_h2_l3.c | FileCheck %t/ae_mulzaafd32x16_h2_l3.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzaafd32x16_h2_l3_s2.c | FileCheck %t/ae_mulzaafd32x16_h2_l3_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzaafd32x16_h3_l2.c | FileCheck %t/ae_mulzaafd32x16_h3_l2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzaafd32x16_h3_l2_s2.c | FileCheck %t/ae_mulzaafd32x16_h3_l2_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzasd24_hh_ll.c | FileCheck %t/ae_mulzasd24_hh_ll.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzasd24_hh_ll_s2.c | FileCheck %t/ae_mulzasd24_hh_ll_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzasd24_hl_lh.c | FileCheck %t/ae_mulzasd24_hl_lh.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzasd24_hl_lh_s2.c | FileCheck %t/ae_mulzasd24_hl_lh_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzasd32x16_h1_l0.c | FileCheck %t/ae_mulzasd32x16_h1_l0.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzasd32x16_h1_l0_s2.c | FileCheck %t/ae_mulzasd32x16_h1_l0_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzasd32x16_h3_l2.c | FileCheck %t/ae_mulzasd32x16_h3_l2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzasd32x16_h3_l2_s2.c | FileCheck %t/ae_mulzasd32x16_h3_l2_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzasfd24_hh_ll.c | FileCheck %t/ae_mulzasfd24_hh_ll.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzasfd24_hh_ll_s2.c | FileCheck %t/ae_mulzasfd24_hh_ll_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzasfd24_hl_lh.c | FileCheck %t/ae_mulzasfd24_hl_lh.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzasfd24_hl_lh_s2.c | FileCheck %t/ae_mulzasfd24_hl_lh_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzasfd32x16_h1_l0.c | FileCheck %t/ae_mulzasfd32x16_h1_l0.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzasfd32x16_h1_l0_s2.c | FileCheck %t/ae_mulzasfd32x16_h1_l0_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzasfd32x16_h3_l2.c | FileCheck %t/ae_mulzasfd32x16_h3_l2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzasfd32x16_h3_l2_s2.c | FileCheck %t/ae_mulzasfd32x16_h3_l2_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzsad24_hh_ll.c | FileCheck %t/ae_mulzsad24_hh_ll.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzsad24_hh_ll_s2.c | FileCheck %t/ae_mulzsad24_hh_ll_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzsad32x16_h1_l0.c | FileCheck %t/ae_mulzsad32x16_h1_l0.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzsad32x16_h1_l0_s2.c | FileCheck %t/ae_mulzsad32x16_h1_l0_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzsad32x16_h3_l2.c | FileCheck %t/ae_mulzsad32x16_h3_l2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzsad32x16_h3_l2_s2.c | FileCheck %t/ae_mulzsad32x16_h3_l2_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzsafd24_hh_ll.c | FileCheck %t/ae_mulzsafd24_hh_ll.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzsafd24_hh_ll_s2.c | FileCheck %t/ae_mulzsafd24_hh_ll_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzsafd32x16_h1_l0.c | FileCheck %t/ae_mulzsafd32x16_h1_l0.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzsafd32x16_h1_l0_s2.c | FileCheck %t/ae_mulzsafd32x16_h1_l0_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzsafd32x16_h3_l2.c | FileCheck %t/ae_mulzsafd32x16_h3_l2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzsafd32x16_h3_l2_s2.c | FileCheck %t/ae_mulzsafd32x16_h3_l2_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzssd24_hh_ll.c | FileCheck %t/ae_mulzssd24_hh_ll.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzssd24_hh_ll_s2.c | FileCheck %t/ae_mulzssd24_hh_ll_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzssd24_hl_lh.c | FileCheck %t/ae_mulzssd24_hl_lh.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzssd24_hl_lh_s2.c | FileCheck %t/ae_mulzssd24_hl_lh_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzssd32x16_h1_l0.c | FileCheck %t/ae_mulzssd32x16_h1_l0.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzssd32x16_h1_l0_s2.c | FileCheck %t/ae_mulzssd32x16_h1_l0_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzssd32x16_h3_l2.c | FileCheck %t/ae_mulzssd32x16_h3_l2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzssd32x16_h3_l2_s2.c | FileCheck %t/ae_mulzssd32x16_h3_l2_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzssfd16ss_11_00.c | FileCheck %t/ae_mulzssfd16ss_11_00.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzssfd16ss_11_00_s2.c | FileCheck %t/ae_mulzssfd16ss_11_00_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzssfd16ss_13_02.c | FileCheck %t/ae_mulzssfd16ss_13_02.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzssfd16ss_13_02_s2.c | FileCheck %t/ae_mulzssfd16ss_13_02_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzssfd16ss_33_22.c | FileCheck %t/ae_mulzssfd16ss_33_22.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzssfd16ss_33_22_s2.c | FileCheck %t/ae_mulzssfd16ss_33_22_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzssfd24_hh_ll.c | FileCheck %t/ae_mulzssfd24_hh_ll.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzssfd24_hh_ll_s2.c | FileCheck %t/ae_mulzssfd24_hh_ll_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzssfd24_hl_lh.c | FileCheck %t/ae_mulzssfd24_hl_lh.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzssfd24_hl_lh_s2.c | FileCheck %t/ae_mulzssfd24_hl_lh_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzssfd32x16_h1_l0.c | FileCheck %t/ae_mulzssfd32x16_h1_l0.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzssfd32x16_h1_l0_s2.c | FileCheck %t/ae_mulzssfd32x16_h1_l0_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzssfd32x16_h3_l2.c | FileCheck %t/ae_mulzssfd32x16_h3_l2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_mulzssfd32x16_h3_l2_s2.c | FileCheck %t/ae_mulzssfd32x16_h3_l2_s2.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_nand.c | FileCheck %t/ae_nand.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_neg16s.c | FileCheck %t/ae_neg16s.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_neg24s.c | FileCheck %t/ae_neg24s.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_neg32.c | FileCheck %t/ae_neg32.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_neg32s.c | FileCheck %t/ae_neg32s.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_neg64.c | FileCheck %t/ae_neg64.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_neg64s.c | FileCheck %t/ae_neg64s.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_nsa64.c | FileCheck %t/ae_nsa64.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_nsaz16_0.c | FileCheck %t/ae_nsaz16_0.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_nsaz32_l.c | FileCheck %t/ae_nsaz32_l.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_or.c | FileCheck %t/ae_or.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_pksr24.c | FileCheck %t/ae_pksr24.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_pksr32.c | FileCheck %t/ae_pksr32.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_round16x4f32sasym.c | FileCheck %t/ae_round16x4f32sasym.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_round16x4f32ssym.c | FileCheck %t/ae_round16x4f32ssym.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_round24x2f48sasym.c | FileCheck %t/ae_round24x2f48sasym.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_round24x2f48ssym.c | FileCheck %t/ae_round24x2f48ssym.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_round32x2f48sasym.c | FileCheck %t/ae_round32x2f48sasym.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_round32x2f48ssym.c | FileCheck %t/ae_round32x2f48ssym.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_round32x2f64sasym.c | FileCheck %t/ae_round32x2f64sasym.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_round32x2f64ssym.c | FileCheck %t/ae_round32x2f64ssym.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_roundsp16f24asym.c | FileCheck %t/ae_roundsp16f24asym.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_roundsp16f24sym.c | FileCheck %t/ae_roundsp16f24sym.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_roundsp16q48x2asym.c | FileCheck %t/ae_roundsp16q48x2asym.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_roundsp16q48x2sym.c | FileCheck %t/ae_roundsp16q48x2sym.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_roundsq32f48asym.c | FileCheck %t/ae_roundsq32f48asym.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_roundsq32f48sym.c | FileCheck %t/ae_roundsq32f48sym.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s16_0_i.c | FileCheck %t/ae_s16_0_i.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s16_0_ip.c | FileCheck %t/ae_s16_0_ip.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s16_0_x.c | FileCheck %t/ae_s16_0_x.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s16_0_xc.c | FileCheck %t/ae_s16_0_xc.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s16_0_xp.c | FileCheck %t/ae_s16_0_xp.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s16m_l_i.c | FileCheck %t/ae_s16m_l_i.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s16m_l_iu.c | FileCheck %t/ae_s16m_l_iu.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s16m_l_x.c | FileCheck %t/ae_s16m_l_x.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s16m_l_xc.c | FileCheck %t/ae_s16m_l_xc.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s16m_l_xu.c | FileCheck %t/ae_s16m_l_xu.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s16x2m_i.c | FileCheck %t/ae_s16x2m_i.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s16x2m_iu.c | FileCheck %t/ae_s16x2m_iu.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s16x2m_x.c | FileCheck %t/ae_s16x2m_x.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s16x2m_xc.c | FileCheck %t/ae_s16x2m_xc.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s16x2m_xu.c | FileCheck %t/ae_s16x2m_xu.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s16x4_i.c | FileCheck %t/ae_s16x4_i.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s16x4_ip.c | FileCheck %t/ae_s16x4_ip.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s16x4_ric.c | FileCheck %t/ae_s16x4_ric.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s16x4_rip.c | FileCheck %t/ae_s16x4_rip.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s16x4_x.c | FileCheck %t/ae_s16x4_x.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s16x4_xc.c | FileCheck %t/ae_s16x4_xc.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s16x4_xp.c | FileCheck %t/ae_s16x4_xp.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s24ra64s_i.c | FileCheck %t/ae_s24ra64s_i.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s24ra64s_ip.c | FileCheck %t/ae_s24ra64s_ip.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s24ra64s_x.c | FileCheck %t/ae_s24ra64s_x.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s24ra64s_xc.c | FileCheck %t/ae_s24ra64s_xc.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s24ra64s_xp.c | FileCheck %t/ae_s24ra64s_xp.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s24x2ra64s_ip.c | FileCheck %t/ae_s24x2ra64s_ip.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s32_l_i.c | FileCheck %t/ae_s32_l_i.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s32_l_ip.c | FileCheck %t/ae_s32_l_ip.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s32_l_x.c | FileCheck %t/ae_s32_l_x.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s32_l_xc.c | FileCheck %t/ae_s32_l_xc.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s32_l_xp.c | FileCheck %t/ae_s32_l_xp.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s32f24_l_i.c | FileCheck %t/ae_s32f24_l_i.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s32f24_l_ip.c | FileCheck %t/ae_s32f24_l_ip.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s32f24_l_x.c | FileCheck %t/ae_s32f24_l_x.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s32f24_l_xc.c | FileCheck %t/ae_s32f24_l_xc.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s32f24_l_xp.c | FileCheck %t/ae_s32f24_l_xp.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s32m_i.c | FileCheck %t/ae_s32m_i.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s32m_iu.c | FileCheck %t/ae_s32m_iu.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s32m_x.c | FileCheck %t/ae_s32m_x.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s32m_xc.c | FileCheck %t/ae_s32m_xc.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s32m_xu.c | FileCheck %t/ae_s32m_xu.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s32ra64s_i.c | FileCheck %t/ae_s32ra64s_i.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s32ra64s_ip.c | FileCheck %t/ae_s32ra64s_ip.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s32ra64s_x.c | FileCheck %t/ae_s32ra64s_x.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s32ra64s_xc.c | FileCheck %t/ae_s32ra64s_xc.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s32ra64s_xp.c | FileCheck %t/ae_s32ra64s_xp.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s32x2_i.c | FileCheck %t/ae_s32x2_i.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s32x2_ip.c | FileCheck %t/ae_s32x2_ip.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s32x2_ric.c | FileCheck %t/ae_s32x2_ric.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s32x2_rip.c | FileCheck %t/ae_s32x2_rip.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s32x2_x.c | FileCheck %t/ae_s32x2_x.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s32x2_xc.c | FileCheck %t/ae_s32x2_xc.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s32x2_xp.c | FileCheck %t/ae_s32x2_xp.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s32x2f24_i.c | FileCheck %t/ae_s32x2f24_i.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s32x2f24_ip.c | FileCheck %t/ae_s32x2f24_ip.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s32x2f24_ric.c | FileCheck %t/ae_s32x2f24_ric.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s32x2f24_rip.c | FileCheck %t/ae_s32x2f24_rip.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s32x2f24_x.c | FileCheck %t/ae_s32x2f24_x.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s32x2f24_xc.c | FileCheck %t/ae_s32x2f24_xc.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s32x2f24_xp.c | FileCheck %t/ae_s32x2f24_xp.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s32x2ra64s_ip.c | FileCheck %t/ae_s32x2ra64s_ip.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s64_i.c | FileCheck %t/ae_s64_i.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s64_ip.c | FileCheck %t/ae_s64_ip.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s64_x.c | FileCheck %t/ae_s64_x.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s64_xc.c | FileCheck %t/ae_s64_xc.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_s64_xp.c | FileCheck %t/ae_s64_xp.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sa16x4_ic.c | FileCheck %t/ae_sa16x4_ic.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sa16x4_ip.c | FileCheck %t/ae_sa16x4_ip.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sa16x4_ric.c | FileCheck %t/ae_sa16x4_ric.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sa16x4_rip.c | FileCheck %t/ae_sa16x4_rip.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sa24_l_ic.c | FileCheck %t/ae_sa24_l_ic.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sa24_l_ip.c | FileCheck %t/ae_sa24_l_ip.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sa24_l_ric.c | FileCheck %t/ae_sa24_l_ric.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sa24_l_rip.c | FileCheck %t/ae_sa24_l_rip.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sa24x2_ic.c | FileCheck %t/ae_sa24x2_ic.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sa24x2_ip.c | FileCheck %t/ae_sa24x2_ip.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sa24x2_ric.c | FileCheck %t/ae_sa24x2_ric.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sa24x2_rip.c | FileCheck %t/ae_sa24x2_rip.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sa32x2_ic.c | FileCheck %t/ae_sa32x2_ic.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sa32x2_ip.c | FileCheck %t/ae_sa32x2_ip.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sa32x2_ric.c | FileCheck %t/ae_sa32x2_ric.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sa32x2_rip.c | FileCheck %t/ae_sa32x2_rip.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sa32x2f24_ic.c | FileCheck %t/ae_sa32x2f24_ic.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sa32x2f24_ip.c | FileCheck %t/ae_sa32x2f24_ip.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sa32x2f24_ric.c | FileCheck %t/ae_sa32x2f24_ric.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sa32x2f24_rip.c | FileCheck %t/ae_sa32x2f24_rip.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sa64neg_fp.c | FileCheck %t/ae_sa64neg_fp.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sa64pos_fp.c | FileCheck %t/ae_sa64pos_fp.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_salign64_i.c | FileCheck %t/ae_salign64_i.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sat16x4.c | FileCheck %t/ae_sat16x4.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sat24s.c | FileCheck %t/ae_sat24s.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sat48s.c | FileCheck %t/ae_sat48s.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_satq56s.c | FileCheck %t/ae_satq56s.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sb.c | FileCheck %t/ae_sb.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sb_ic.c | FileCheck %t/ae_sb_ic.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sb_ip.c | FileCheck %t/ae_sb_ip.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sbf.c | FileCheck %t/ae_sbf.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sbf_ic.c | FileCheck %t/ae_sbf_ic.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sbf_ip.c | FileCheck %t/ae_sbf_ip.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sbi.c | FileCheck %t/ae_sbi.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sbi_ic.c | FileCheck %t/ae_sbi_ic.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sbi_ip.c | FileCheck %t/ae_sbi_ip.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sel16i.c | FileCheck %t/ae_sel16i.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sel16i_n.c | FileCheck %t/ae_sel16i_n.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sext32.c | FileCheck %t/ae_sext32.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sext32x2d16_10.c | FileCheck %t/ae_sext32x2d16_10.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sext32x2d16_32.c | FileCheck %t/ae_sext32x2d16_32.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sha32.c | FileCheck %t/ae_sha32.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_shortswap.c | FileCheck %t/ae_shortswap.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_slaa16s.c | FileCheck %t/ae_slaa16s.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_slaa32.c | FileCheck %t/ae_slaa32.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_slaa32s.c | FileCheck %t/ae_slaa32s.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_slaa64.c | FileCheck %t/ae_slaa64.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_slaa64s.c | FileCheck %t/ae_slaa64s.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_slaaq56.c | FileCheck %t/ae_slaaq56.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_slai16s.c | FileCheck %t/ae_slai16s.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_slai24.c | FileCheck %t/ae_slai24.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_slai24s.c | FileCheck %t/ae_slai24s.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_slai32.c | FileCheck %t/ae_slai32.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_slai32s.c | FileCheck %t/ae_slai32s.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_slai64.c | FileCheck %t/ae_slai64.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_slai64s.c | FileCheck %t/ae_slai64s.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_slaisq56s.c | FileCheck %t/ae_slaisq56s.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_slas24.c | FileCheck %t/ae_slas24.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_slas24s.c | FileCheck %t/ae_slas24s.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_slas32.c | FileCheck %t/ae_slas32.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_slas32s.c | FileCheck %t/ae_slas32s.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_slas64.c | FileCheck %t/ae_slas64.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_slas64s.c | FileCheck %t/ae_slas64s.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_slasq56.c | FileCheck %t/ae_slasq56.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_slassq56s.c | FileCheck %t/ae_slassq56s.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sra64_32.c | FileCheck %t/ae_sra64_32.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sraa16rs.c | FileCheck %t/ae_sraa16rs.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sraa16s.c | FileCheck %t/ae_sraa16s.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sraa32.c | FileCheck %t/ae_sraa32.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sraa32rs.c | FileCheck %t/ae_sraa32rs.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sraa32s.c | FileCheck %t/ae_sraa32s.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sraa64.c | FileCheck %t/ae_sraa64.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_srai16.c | FileCheck %t/ae_srai16.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_srai16r.c | FileCheck %t/ae_srai16r.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_srai24.c | FileCheck %t/ae_srai24.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_srai32.c | FileCheck %t/ae_srai32.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_srai32r.c | FileCheck %t/ae_srai32r.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_srai64.c | FileCheck %t/ae_srai64.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sras24.c | FileCheck %t/ae_sras24.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sras32.c | FileCheck %t/ae_sras32.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sras64.c | FileCheck %t/ae_sras64.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_srla32.c | FileCheck %t/ae_srla32.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_srla64.c | FileCheck %t/ae_srla64.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_srli24.c | FileCheck %t/ae_srli24.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_srli32.c | FileCheck %t/ae_srli32.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_srli64.c | FileCheck %t/ae_srli64.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_srls24.c | FileCheck %t/ae_srls24.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_srls32.c | FileCheck %t/ae_srls32.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_srls64.c | FileCheck %t/ae_srls64.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sub16.c | FileCheck %t/ae_sub16.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sub16s.c | FileCheck %t/ae_sub16s.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sub24s.c | FileCheck %t/ae_sub24s.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sub32.c | FileCheck %t/ae_sub32.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sub32s.c | FileCheck %t/ae_sub32s.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sub64.c | FileCheck %t/ae_sub64.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_sub64s.c | FileCheck %t/ae_sub64s.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_subadd32.c | FileCheck %t/ae_subadd32.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_subadd32s.c | FileCheck %t/ae_subadd32s.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_trunca32f64s_l.c | FileCheck %t/ae_trunca32f64s_l.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_trunca32x2f64s.c | FileCheck %t/ae_trunca32x2f64s.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_trunci32f64s_l.c | FileCheck %t/ae_trunci32f64s_l.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_trunci32x2f64s.c | FileCheck %t/ae_trunci32x2f64s.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_vldl16c.c | FileCheck %t/ae_vldl16c.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_vldl16c_ic.c | FileCheck %t/ae_vldl16c_ic.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_vldl16c_ip.c | FileCheck %t/ae_vldl16c_ip.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_vldl16t.c | FileCheck %t/ae_vldl16t.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_vldl32t.c | FileCheck %t/ae_vldl32t.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_vldsht.c | FileCheck %t/ae_vldsht.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_vlel16t.c | FileCheck %t/ae_vlel16t.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_vlel32t.c | FileCheck %t/ae_vlel32t.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_vles16c.c | FileCheck %t/ae_vles16c.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_vles16c_ic.c | FileCheck %t/ae_vles16c_ic.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_vles16c_ip.c | FileCheck %t/ae_vles16c_ip.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_xor.c | FileCheck %t/ae_xor.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/ae_zalign64.c | FileCheck %t/ae_zalign64.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/rur_ae_bithead.c | FileCheck %t/rur_ae_bithead.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/rur_ae_bitptr.c | FileCheck %t/rur_ae_bitptr.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/rur_ae_bitsused.c | FileCheck %t/rur_ae_bitsused.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/rur_ae_cbegin0.c | FileCheck %t/rur_ae_cbegin0.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/rur_ae_cend0.c | FileCheck %t/rur_ae_cend0.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/rur_ae_cw_sd_no.c | FileCheck %t/rur_ae_cw_sd_no.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/rur_ae_cwrap.c | FileCheck %t/rur_ae_cwrap.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/rur_ae_first_ts.c | FileCheck %t/rur_ae_first_ts.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/rur_ae_nextoffset.c | FileCheck %t/rur_ae_nextoffset.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/rur_ae_overflow.c | FileCheck %t/rur_ae_overflow.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/rur_ae_ovf_sar.c | FileCheck %t/rur_ae_ovf_sar.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/rur_ae_sar.c | FileCheck %t/rur_ae_sar.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/rur_ae_searchdone.c | FileCheck %t/rur_ae_searchdone.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/rur_ae_tablesize.c | FileCheck %t/rur_ae_tablesize.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/rur_ae_ts_fts_bu_bp.c | FileCheck %t/rur_ae_ts_fts_bu_bp.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/wur_ae_bithead.c | FileCheck %t/wur_ae_bithead.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/wur_ae_bitptr.c | FileCheck %t/wur_ae_bitptr.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/wur_ae_bitsused.c | FileCheck %t/wur_ae_bitsused.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/wur_ae_cbegin0.c | FileCheck %t/wur_ae_cbegin0.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/wur_ae_cend0.c | FileCheck %t/wur_ae_cend0.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/wur_ae_cw_sd_no.c | FileCheck %t/wur_ae_cw_sd_no.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/wur_ae_cwrap.c | FileCheck %t/wur_ae_cwrap.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/wur_ae_first_ts.c | FileCheck %t/wur_ae_first_ts.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/wur_ae_nextoffset.c | FileCheck %t/wur_ae_nextoffset.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/wur_ae_overflow.c | FileCheck %t/wur_ae_overflow.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/wur_ae_ovf_sar.c | FileCheck %t/wur_ae_ovf_sar.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/wur_ae_sar.c | FileCheck %t/wur_ae_sar.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/wur_ae_searchdone.c | FileCheck %t/wur_ae_searchdone.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/wur_ae_tablesize.c | FileCheck %t/wur_ae_tablesize.c
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/wur_ae_ts_fts_bu_bp.c | FileCheck %t/wur_ae_ts_fts_bu_bp.c
//--- ae_abs16s.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int16x4 test_ae_abs16s(ae_int16x4 ae_arth_v1) {
// CHECK-LABEL: test_ae_abs16s
// CHECK: %[[RET:.*]] = {{(tail)?}} call <4 x i16> @llvm.xtensa.ae.abs16s(<4 x i16> {{.*}})
// CHECK: ret <4 x i16> %[[RET]]
return __builtin_xtensa_ae_abs16s(ae_arth_v1);
}

//--- ae_abs24s.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_abs24s(ae_int32x2 ae_arth_v1) {
// CHECK-LABEL: test_ae_abs24s
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.abs24s(<2 x i32> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_abs24s(ae_arth_v1);
}

//--- ae_abs32.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_abs32(ae_int32x2 ae_arth_v1) {
// CHECK-LABEL: test_ae_abs32
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.abs32(<2 x i32> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_abs32(ae_arth_v1);
}

//--- ae_abs32s.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_abs32s(ae_int32x2 ae_arth_v1) {
// CHECK-LABEL: test_ae_abs32s
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.abs32s(<2 x i32> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_abs32s(ae_arth_v1);
}

//--- ae_abs64.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_abs64(ae_int64 ae_arth_v1) {
// CHECK-LABEL: test_ae_abs64
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.abs64(<1 x i64> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_abs64(ae_arth_v1);
}

//--- ae_abs64s.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_abs64s(ae_int64 ae_arth_v1) {
// CHECK-LABEL: test_ae_abs64s
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.abs64s(<1 x i64> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_abs64s(ae_arth_v1);
}

//--- ae_add16.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int16x4 test_ae_add16(ae_int16x4 ae_arth_v0,ae_int16x4 ae_arth_v1) {
// CHECK-LABEL: test_ae_add16
// CHECK: %[[RET:.*]] = {{(tail)?}} call <4 x i16> @llvm.xtensa.ae.add16(<4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <4 x i16> %[[RET]]
return __builtin_xtensa_ae_add16(ae_arth_v0, ae_arth_v1);
}

//--- ae_add16s.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int16x4 test_ae_add16s(ae_int16x4 ae_arth_v0,ae_int16x4 ae_arth_v1) {
// CHECK-LABEL: test_ae_add16s
// CHECK: %[[RET:.*]] = {{(tail)?}} call <4 x i16> @llvm.xtensa.ae.add16s(<4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <4 x i16> %[[RET]]
return __builtin_xtensa_ae_add16s(ae_arth_v0, ae_arth_v1);
}

//--- ae_add24s.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_add24s(ae_int32x2 ae_arth_v0,ae_int32x2 ae_arth_v1) {
// CHECK-LABEL: test_ae_add24s
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.add24s(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_add24s(ae_arth_v0, ae_arth_v1);
}

//--- ae_add32.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_add32(ae_int32x2 ae_arth_v0,ae_int32x2 ae_arth_v1) {
// CHECK-LABEL: test_ae_add32
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.add32(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_add32(ae_arth_v0, ae_arth_v1);
}

//--- ae_add32_hl_lh.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_add32_hl_lh(ae_int32x2 ae_arth_v0,ae_int32x2 ae_arth_v1) {
// CHECK-LABEL: test_ae_add32_hl_lh
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.add32.hl.lh(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_add32_hl_lh(ae_arth_v0, ae_arth_v1);
}

//--- ae_add32s.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_add32s(ae_int32x2 ae_arth_v0,ae_int32x2 ae_arth_v1) {
// CHECK-LABEL: test_ae_add32s
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.add32s(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_add32s(ae_arth_v0, ae_arth_v1);
}

//--- ae_add64.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_add64(ae_int64 ae_arth_v0,ae_int64 ae_arth_v1) {
// CHECK-LABEL: test_ae_add64
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.add64(<1 x i64> {{.*}}, <1 x i64> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_add64(ae_arth_v0, ae_arth_v1);
}

//--- ae_add64s.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_add64s(ae_int64 ae_arth_v0,ae_int64 ae_arth_v1) {
// CHECK-LABEL: test_ae_add64s
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.add64s(<1 x i64> {{.*}}, <1 x i64> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_add64s(ae_arth_v0, ae_arth_v1);
}

//--- ae_addbrba32.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

int test_ae_addbrba32(int art,int ars) {
// CHECK-LABEL: test_ae_addbrba32
// CHECK: %[[RET:.*]] = {{(tail)?}} call i32 @llvm.xtensa.ae.addbrba32(i32 {{.*}}, i32 {{.*}})
// CHECK: ret i32 %[[RET]]
return __builtin_xtensa_ae_addbrba32(art, ars);
}

//--- ae_addsub32.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_addsub32(ae_int32x2 ae_arth_v0,ae_int32x2 ae_arth_v1) {
// CHECK-LABEL: test_ae_addsub32
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.addsub32(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_addsub32(ae_arth_v0, ae_arth_v1);
}

//--- ae_addsub32s.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_addsub32s(ae_int32x2 ae_arth_v0,ae_int32x2 ae_arth_v1) {
// CHECK-LABEL: test_ae_addsub32s
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.addsub32s(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_addsub32s(ae_arth_v0, ae_arth_v1);
}

//--- ae_and.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_and(ae_int64 ae_dr_to_dr_v0,ae_int64 ae_dr_to_dr_v1) {
// CHECK-LABEL: test_ae_and
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.and(<1 x i64> {{.*}}, <1 x i64> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_and(ae_dr_to_dr_v0, ae_dr_to_dr_v1);
}

//--- ae_cvt32x2f16_10.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_cvt32x2f16_10(ae_int16x4 ae_to_dr_v0) {
// CHECK-LABEL: test_ae_cvt32x2f16_10
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.cvt32x2f16.10(<4 x i16> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_cvt32x2f16_10(ae_to_dr_v0);
}

//--- ae_cvt32x2f16_32.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_cvt32x2f16_32(ae_int16x4 ae_to_dr_v0) {
// CHECK-LABEL: test_ae_cvt32x2f16_32
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.cvt32x2f16.32(<4 x i16> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_cvt32x2f16_32(ae_to_dr_v0);
}

//--- ae_cvt48a32.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_cvt48a32(int ars) {
// CHECK-LABEL: test_ae_cvt48a32
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.cvt48a32(i32 {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_cvt48a32(ars);
}

//--- ae_cvt64a32.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_cvt64a32(int ars) {
// CHECK-LABEL: test_ae_cvt64a32
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.cvt64a32(i32 {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_cvt64a32(ars);
}

//--- ae_cvt64f32_h.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_cvt64f32_h(ae_int32x2 ae_dr_to_dr_v0) {
// CHECK-LABEL: test_ae_cvt64f32_h
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.cvt64f32.h(<2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_cvt64f32_h(ae_dr_to_dr_v0);
}

//--- ae_cvta32f24s_h.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

int test_ae_cvta32f24s_h(ae_int32x2 ae_dr_to_ar_v0) {
// CHECK-LABEL: test_ae_cvta32f24s_h
// CHECK: %[[RET:.*]] = {{(tail)?}} call i32 @llvm.xtensa.ae.cvta32f24s.h(<2 x i32> {{.*}})
// CHECK: ret i32 %[[RET]]
return __builtin_xtensa_ae_cvta32f24s_h(ae_dr_to_ar_v0);
}

//--- ae_cvta32f24s_l.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

int test_ae_cvta32f24s_l(ae_int32x2 ae_dr_to_ar_v0) {
// CHECK-LABEL: test_ae_cvta32f24s_l
// CHECK: %[[RET:.*]] = {{(tail)?}} call i32 @llvm.xtensa.ae.cvta32f24s.l(<2 x i32> {{.*}})
// CHECK: ret i32 %[[RET]]
return __builtin_xtensa_ae_cvta32f24s_l(ae_dr_to_ar_v0);
}

//--- ae_cvtq56a32s.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_cvtq56a32s(int ars) {
// CHECK-LABEL: test_ae_cvtq56a32s
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.cvtq56a32s(i32 {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_cvtq56a32s(ars);
}

//--- ae_cvtq56p32s_h.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_cvtq56p32s_h(ae_int32x2 ae_dr_to_dr_v0) {
// CHECK-LABEL: test_ae_cvtq56p32s_h
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.cvtq56p32s.h(<2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_cvtq56p32s_h(ae_dr_to_dr_v0);
}

//--- ae_cvtq56p32s_l.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_cvtq56p32s_l(ae_int32x2 ae_dr_to_dr_v0) {
// CHECK-LABEL: test_ae_cvtq56p32s_l
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.cvtq56p32s.l(<2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_cvtq56p32s_l(ae_dr_to_dr_v0);
}

//--- ae_db.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_db(const short** ars,int art) {
// CHECK-LABEL: test_ae_db
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.db(ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_db(ars, art);
}

//--- ae_db_ic.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_db_ic(const short** ars,int art) {
// CHECK-LABEL: test_ae_db_ic
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.db.ic(ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_db_ic(ars, art);
}

//--- ae_db_ip.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_db_ip(const short** ars,int art) {
// CHECK-LABEL: test_ae_db_ip
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.db.ip(ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_db_ip(ars, art);
}

//--- ae_dbi.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_dbi(const short** ars,immediate ae_ohba) {
// CHECK-LABEL: test_ae_dbi
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.dbi(ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_dbi(ars, 1);
}

//--- ae_dbi_ic.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_dbi_ic(const short** ars,immediate ae_ohba) {
// CHECK-LABEL: test_ae_dbi_ic
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.dbi.ic(ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_dbi_ic(ars, 1);
}

//--- ae_dbi_ip.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_dbi_ip(const short** ars,immediate ae_ohba) {
// CHECK-LABEL: test_ae_dbi_ip
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.dbi.ip(ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_dbi_ip(ars, 1);
}

//--- ae_div64d32_h.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_div64d32_h(ae_int64* ae_arth_v,ae_int32x2 ae_arth_v1) {
// CHECK-LABEL: test_ae_div64d32_h
// CHECK: %[[LD_AE_ARTH_V:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.div64d32.h(<1 x i64> %[[LD_AE_ARTH_V]], <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_div64d32_h(ae_arth_v, ae_arth_v1);
}

//--- ae_div64d32_l.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_div64d32_l(ae_int64* ae_arth_v,ae_int32x2 ae_arth_v1) {
// CHECK-LABEL: test_ae_div64d32_l
// CHECK: %[[LD_AE_ARTH_V:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.div64d32.l(<1 x i64> %[[LD_AE_ARTH_V]], <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_div64d32_l(ae_arth_v, ae_arth_v1);
}

//--- ae_eq16.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

xtbool4 test_ae_eq16(ae_int16x4 ae_cmpp_v0,ae_int16x4 ae_cmpp_v1) {
// CHECK-LABEL: test_ae_eq16
// CHECK: %[[RET:.*]] = {{(tail)?}} call <4 x i1> @llvm.xtensa.ae.eq16(<4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <4 x i1> %[[RET]]
return __builtin_xtensa_ae_eq16(ae_cmpp_v0, ae_cmpp_v1);
}

//--- ae_eq32.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

xtbool2 test_ae_eq32(ae_int32x2 ae_cmpp_v0,ae_int32x2 ae_cmpp_v1) {
// CHECK-LABEL: test_ae_eq32
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i1> @llvm.xtensa.ae.eq32(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <2 x i1> %[[RET]]
return __builtin_xtensa_ae_eq32(ae_cmpp_v0, ae_cmpp_v1);
}

//--- ae_eq64.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

xtbool test_ae_eq64(ae_int64 ae_cmpp_v0,ae_int64 ae_cmpp_v1) {
// CHECK-LABEL: test_ae_eq64
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i1> @llvm.xtensa.ae.eq64(<1 x i64> {{.*}}, <1 x i64> {{.*}})
// CHECK: ret <1 x i1> %[[RET]]
return __builtin_xtensa_ae_eq64(ae_cmpp_v0, ae_cmpp_v1);
}

//--- ae_l16_i.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int16x4 test_ae_l16_i(const ae_int16* ars,immediate ae_immls16) {
// CHECK-LABEL: test_ae_l16_i
// CHECK: %[[RET:.*]] = {{(tail)?}} call <4 x i16> @llvm.xtensa.ae.l16.i(ptr {{.*}}, i32 {{.*}})
// CHECK: ret <4 x i16> %[[RET]]
return __builtin_xtensa_ae_l16_i(ars, -16);
}

//--- ae_l16_ip.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_l16_ip(ae_int16x4* ae_ls_v,const ae_int16** ars,immediate ae_immls16) {
// CHECK-LABEL: test_ae_l16_ip
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <4 x i16>, ptr } @llvm.xtensa.ae.l16.ip(ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: %[[EV:.*]] = extractvalue { <4 x i16>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_l16_ip(ae_ls_v, ars, -16);
}

//--- ae_l16_x.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int16x4 test_ae_l16_x(const ae_int16* ars,int art) {
// CHECK-LABEL: test_ae_l16_x
// CHECK: %[[RET:.*]] = {{(tail)?}} call <4 x i16> @llvm.xtensa.ae.l16.x(ptr {{.*}}, i32 {{.*}})
// CHECK: ret <4 x i16> %[[RET]]
return __builtin_xtensa_ae_l16_x(ars, art);
}

//--- ae_l16_xc.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_l16_xc(ae_int16x4* ae_ls_v,const ae_int16** ars,int art) {
// CHECK-LABEL: test_ae_l16_xc
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <4 x i16>, ptr } @llvm.xtensa.ae.l16.xc(ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: %[[EV:.*]] = extractvalue { <4 x i16>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_l16_xc(ae_ls_v, ars, art);
}

//--- ae_l16_xp.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_l16_xp(ae_int16x4* ae_ls_v,const ae_int16** ars,int art) {
// CHECK-LABEL: test_ae_l16_xp
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <4 x i16>, ptr } @llvm.xtensa.ae.l16.xp(ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: %[[EV:.*]] = extractvalue { <4 x i16>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_l16_xp(ae_ls_v, ars, art);
}

//--- ae_l16m_i.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_l16m_i(const ae_int16* ars,immediate ae_immls16) {
// CHECK-LABEL: test_ae_l16m_i
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.l16m.i(ptr {{.*}}, i32 {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_l16m_i(ars, -16);
}

//--- ae_l16m_iu.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_l16m_iu(ae_int32x2* ae_ls_v,const ae_int16** ars,immediate ae_immls16) {
// CHECK-LABEL: test_ae_l16m_iu
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <2 x i32>, ptr } @llvm.xtensa.ae.l16m.iu(ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: %[[EV:.*]] = extractvalue { <2 x i32>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_l16m_iu(ae_ls_v, ars, -16);
}

//--- ae_l16m_x.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_l16m_x(const ae_int16* ars,int art) {
// CHECK-LABEL: test_ae_l16m_x
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.l16m.x(ptr {{.*}}, i32 {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_l16m_x(ars, art);
}

//--- ae_l16m_xc.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_l16m_xc(ae_int32x2* ae_ls_v,const ae_int16** ars,int art) {
// CHECK-LABEL: test_ae_l16m_xc
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <2 x i32>, ptr } @llvm.xtensa.ae.l16m.xc(ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: %[[EV:.*]] = extractvalue { <2 x i32>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_l16m_xc(ae_ls_v, ars, art);
}

//--- ae_l16m_xu.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_l16m_xu(ae_int32x2* ae_ls_v,const ae_int16** ars,int art) {
// CHECK-LABEL: test_ae_l16m_xu
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <2 x i32>, ptr } @llvm.xtensa.ae.l16m.xu(ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: %[[EV:.*]] = extractvalue { <2 x i32>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_l16m_xu(ae_ls_v, ars, art);
}

//--- ae_l16x2m_i.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_l16x2m_i(const ae_int16x2* ars,immediate ae_immls32) {
// CHECK-LABEL: test_ae_l16x2m_i
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.l16x2m.i(ptr {{.*}}, i32 {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_l16x2m_i(ars, -32);
}

//--- ae_l16x2m_iu.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_l16x2m_iu(ae_int32x2* ae_ls_v,const ae_int16x2** ars,immediate ae_immls32) {
// CHECK-LABEL: test_ae_l16x2m_iu
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <2 x i32>, ptr } @llvm.xtensa.ae.l16x2m.iu(ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: %[[EV:.*]] = extractvalue { <2 x i32>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_l16x2m_iu(ae_ls_v, ars, -32);
}

//--- ae_l16x2m_x.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_l16x2m_x(const ae_int16x2* ars,int art) {
// CHECK-LABEL: test_ae_l16x2m_x
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.l16x2m.x(ptr {{.*}}, i32 {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_l16x2m_x(ars, art);
}

//--- ae_l16x2m_xc.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_l16x2m_xc(ae_int32x2* ae_ls_v,const ae_int16x2** ars,int art) {
// CHECK-LABEL: test_ae_l16x2m_xc
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <2 x i32>, ptr } @llvm.xtensa.ae.l16x2m.xc(ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: %[[EV:.*]] = extractvalue { <2 x i32>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_l16x2m_xc(ae_ls_v, ars, art);
}

//--- ae_l16x2m_xu.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_l16x2m_xu(ae_int32x2* ae_ls_v,const ae_int16x2** ars,int art) {
// CHECK-LABEL: test_ae_l16x2m_xu
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <2 x i32>, ptr } @llvm.xtensa.ae.l16x2m.xu(ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: %[[EV:.*]] = extractvalue { <2 x i32>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_l16x2m_xu(ae_ls_v, ars, art);
}

//--- ae_l16x4_i.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int16x4 test_ae_l16x4_i(const ae_int16x4* ars,immediate ae_immls64) {
// CHECK-LABEL: test_ae_l16x4_i
// CHECK: %[[RET:.*]] = {{(tail)?}} call <4 x i16> @llvm.xtensa.ae.l16x4.i(ptr {{.*}}, i32 {{.*}})
// CHECK: ret <4 x i16> %[[RET]]
return __builtin_xtensa_ae_l16x4_i(ars, -64);
}

//--- ae_l16x4_ip.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_l16x4_ip(ae_int16x4* ae_ls_v,const ae_int16x4** ars,immediate ae_immls64pos) {
// CHECK-LABEL: test_ae_l16x4_ip
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <4 x i16>, ptr } @llvm.xtensa.ae.l16x4.ip(ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: %[[EV:.*]] = extractvalue { <4 x i16>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_l16x4_ip(ae_ls_v, ars, 0);
}

//--- ae_l16x4_ric.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_l16x4_ric(ae_int16x4* ae_ls_v,const ae_int16x4** ars) {
// CHECK-LABEL: test_ae_l16x4_ric
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <4 x i16>, ptr } @llvm.xtensa.ae.l16x4.ric(ptr %[[LD_ARS]])
// CHECK: %[[EV:.*]] = extractvalue { <4 x i16>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_l16x4_ric(ae_ls_v, ars);
}

//--- ae_l16x4_rip.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_l16x4_rip(ae_int16x4* ae_ls_v,const ae_int16x4** ars) {
// CHECK-LABEL: test_ae_l16x4_rip
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <4 x i16>, ptr } @llvm.xtensa.ae.l16x4.rip(ptr %[[LD_ARS]])
// CHECK: %[[EV:.*]] = extractvalue { <4 x i16>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_l16x4_rip(ae_ls_v, ars);
}

//--- ae_l16x4_x.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int16x4 test_ae_l16x4_x(const ae_int16x4* ars,int art) {
// CHECK-LABEL: test_ae_l16x4_x
// CHECK: %[[RET:.*]] = {{(tail)?}} call <4 x i16> @llvm.xtensa.ae.l16x4.x(ptr {{.*}}, i32 {{.*}})
// CHECK: ret <4 x i16> %[[RET]]
return __builtin_xtensa_ae_l16x4_x(ars, art);
}

//--- ae_l16x4_xc.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_l16x4_xc(ae_int16x4* ae_ls_v,const ae_int16x4** ars,int art) {
// CHECK-LABEL: test_ae_l16x4_xc
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <4 x i16>, ptr } @llvm.xtensa.ae.l16x4.xc(ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: %[[EV:.*]] = extractvalue { <4 x i16>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_l16x4_xc(ae_ls_v, ars, art);
}

//--- ae_l16x4_xp.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_l16x4_xp(ae_int16x4* ae_ls_v,const ae_int16x4** ars,int art) {
// CHECK-LABEL: test_ae_l16x4_xp
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <4 x i16>, ptr } @llvm.xtensa.ae.l16x4.xp(ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: %[[EV:.*]] = extractvalue { <4 x i16>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_l16x4_xp(ae_ls_v, ars, art);
}

//--- ae_l32_i.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_l32_i(const ae_int32* ars,immediate ae_immls32) {
// CHECK-LABEL: test_ae_l32_i
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.l32.i(ptr {{.*}}, i32 {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_l32_i(ars, -32);
}

//--- ae_l32_ip.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_l32_ip(ae_int32x2* ae_ls_v,const ae_int32** ars,immediate ae_immls32) {
// CHECK-LABEL: test_ae_l32_ip
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <2 x i32>, ptr } @llvm.xtensa.ae.l32.ip(ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: %[[EV:.*]] = extractvalue { <2 x i32>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_l32_ip(ae_ls_v, ars, -32);
}

//--- ae_l32_x.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_l32_x(const ae_int32* ars,int art) {
// CHECK-LABEL: test_ae_l32_x
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.l32.x(ptr {{.*}}, i32 {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_l32_x(ars, art);
}

//--- ae_l32_xc.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_l32_xc(ae_int32x2* ae_ls_v,const ae_int32** ars,int art) {
// CHECK-LABEL: test_ae_l32_xc
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <2 x i32>, ptr } @llvm.xtensa.ae.l32.xc(ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: %[[EV:.*]] = extractvalue { <2 x i32>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_l32_xc(ae_ls_v, ars, art);
}

//--- ae_l32_xp.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_l32_xp(ae_int32x2* ae_ls_v,const ae_int32** ars,int art) {
// CHECK-LABEL: test_ae_l32_xp
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <2 x i32>, ptr } @llvm.xtensa.ae.l32.xp(ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: %[[EV:.*]] = extractvalue { <2 x i32>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_l32_xp(ae_ls_v, ars, art);
}

//--- ae_l32f24_i.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_l32f24_i(const ae_int32* ars,immediate ae_immls32) {
// CHECK-LABEL: test_ae_l32f24_i
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.l32f24.i(ptr {{.*}}, i32 {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_l32f24_i(ars, -32);
}

//--- ae_l32f24_ip.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_l32f24_ip(ae_int32x2* ae_ls_v,const ae_int32** ars,immediate ae_immls32) {
// CHECK-LABEL: test_ae_l32f24_ip
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <2 x i32>, ptr } @llvm.xtensa.ae.l32f24.ip(ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: %[[EV:.*]] = extractvalue { <2 x i32>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_l32f24_ip(ae_ls_v, ars, -32);
}

//--- ae_l32f24_x.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_l32f24_x(const ae_int32* ars,int art) {
// CHECK-LABEL: test_ae_l32f24_x
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.l32f24.x(ptr {{.*}}, i32 {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_l32f24_x(ars, art);
}

//--- ae_l32f24_xc.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_l32f24_xc(ae_int32x2* ae_ls_v,const ae_int32** ars,int art) {
// CHECK-LABEL: test_ae_l32f24_xc
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <2 x i32>, ptr } @llvm.xtensa.ae.l32f24.xc(ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: %[[EV:.*]] = extractvalue { <2 x i32>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_l32f24_xc(ae_ls_v, ars, art);
}

//--- ae_l32f24_xp.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_l32f24_xp(ae_int32x2* ae_ls_v,const ae_int32** ars,int art) {
// CHECK-LABEL: test_ae_l32f24_xp
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <2 x i32>, ptr } @llvm.xtensa.ae.l32f24.xp(ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: %[[EV:.*]] = extractvalue { <2 x i32>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_l32f24_xp(ae_ls_v, ars, art);
}

//--- ae_l32m_i.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_l32m_i(const ae_int32* ars,immediate ae_immls32) {
// CHECK-LABEL: test_ae_l32m_i
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.l32m.i(ptr {{.*}}, i32 {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_l32m_i(ars, -32);
}

//--- ae_l32m_iu.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_l32m_iu(ae_int64* ae_ls_v,const ae_int32** ars,immediate ae_immls32) {
// CHECK-LABEL: test_ae_l32m_iu
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <1 x i64>, ptr } @llvm.xtensa.ae.l32m.iu(ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: %[[EV:.*]] = extractvalue { <1 x i64>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_l32m_iu(ae_ls_v, ars, -32);
}

//--- ae_l32m_x.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_l32m_x(const ae_int32* ars,int art) {
// CHECK-LABEL: test_ae_l32m_x
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.l32m.x(ptr {{.*}}, i32 {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_l32m_x(ars, art);
}

//--- ae_l32m_xc.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_l32m_xc(ae_int64* ae_ls_v,const ae_int32** ars,int art) {
// CHECK-LABEL: test_ae_l32m_xc
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <1 x i64>, ptr } @llvm.xtensa.ae.l32m.xc(ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: %[[EV:.*]] = extractvalue { <1 x i64>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_l32m_xc(ae_ls_v, ars, art);
}

//--- ae_l32m_xu.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_l32m_xu(ae_int64* ae_ls_v,const ae_int32** ars,int art) {
// CHECK-LABEL: test_ae_l32m_xu
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <1 x i64>, ptr } @llvm.xtensa.ae.l32m.xu(ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: %[[EV:.*]] = extractvalue { <1 x i64>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_l32m_xu(ae_ls_v, ars, art);
}

//--- ae_l32x2_i.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_l32x2_i(const ae_int32x2* ars,immediate ae_immls64) {
// CHECK-LABEL: test_ae_l32x2_i
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.l32x2.i(ptr {{.*}}, i32 {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_l32x2_i(ars, -64);
}

//--- ae_l32x2_ip.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_l32x2_ip(ae_int32x2* ae_ls_v,const ae_int32x2** ars,immediate ae_immls64pos) {
// CHECK-LABEL: test_ae_l32x2_ip
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <2 x i32>, ptr } @llvm.xtensa.ae.l32x2.ip(ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: %[[EV:.*]] = extractvalue { <2 x i32>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_l32x2_ip(ae_ls_v, ars, 0);
}

//--- ae_l32x2_ric.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_l32x2_ric(ae_int32x2* ae_ls_v,const ae_int32x2** ars) {
// CHECK-LABEL: test_ae_l32x2_ric
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <2 x i32>, ptr } @llvm.xtensa.ae.l32x2.ric(ptr %[[LD_ARS]])
// CHECK: %[[EV:.*]] = extractvalue { <2 x i32>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_l32x2_ric(ae_ls_v, ars);
}

//--- ae_l32x2_rip.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_l32x2_rip(ae_int32x2* ae_ls_v,const ae_int32x2** ars) {
// CHECK-LABEL: test_ae_l32x2_rip
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <2 x i32>, ptr } @llvm.xtensa.ae.l32x2.rip(ptr %[[LD_ARS]])
// CHECK: %[[EV:.*]] = extractvalue { <2 x i32>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_l32x2_rip(ae_ls_v, ars);
}

//--- ae_l32x2_x.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_l32x2_x(const ae_int32x2* ars,int art) {
// CHECK-LABEL: test_ae_l32x2_x
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.l32x2.x(ptr {{.*}}, i32 {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_l32x2_x(ars, art);
}

//--- ae_l32x2_xc.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_l32x2_xc(ae_int32x2* ae_ls_v,const ae_int32x2** ars,int art) {
// CHECK-LABEL: test_ae_l32x2_xc
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <2 x i32>, ptr } @llvm.xtensa.ae.l32x2.xc(ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: %[[EV:.*]] = extractvalue { <2 x i32>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_l32x2_xc(ae_ls_v, ars, art);
}

//--- ae_l32x2_xp.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_l32x2_xp(ae_int32x2* ae_ls_v,const ae_int32x2** ars,int art) {
// CHECK-LABEL: test_ae_l32x2_xp
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <2 x i32>, ptr } @llvm.xtensa.ae.l32x2.xp(ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: %[[EV:.*]] = extractvalue { <2 x i32>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_l32x2_xp(ae_ls_v, ars, art);
}

//--- ae_l32x2f24_i.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_l32x2f24_i(const ae_int32x2* ars,immediate ae_immls64) {
// CHECK-LABEL: test_ae_l32x2f24_i
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.l32x2f24.i(ptr {{.*}}, i32 {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_l32x2f24_i(ars, -64);
}

//--- ae_l32x2f24_ip.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_l32x2f24_ip(ae_int32x2* ae_ls_v,const ae_int32x2** ars,immediate ae_immls64pos) {
// CHECK-LABEL: test_ae_l32x2f24_ip
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <2 x i32>, ptr } @llvm.xtensa.ae.l32x2f24.ip(ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: %[[EV:.*]] = extractvalue { <2 x i32>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_l32x2f24_ip(ae_ls_v, ars, 0);
}

//--- ae_l32x2f24_ric.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_l32x2f24_ric(ae_int32x2* ae_ls_v,const ae_int32x2** ars) {
// CHECK-LABEL: test_ae_l32x2f24_ric
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <2 x i32>, ptr } @llvm.xtensa.ae.l32x2f24.ric(ptr %[[LD_ARS]])
// CHECK: %[[EV:.*]] = extractvalue { <2 x i32>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_l32x2f24_ric(ae_ls_v, ars);
}

//--- ae_l32x2f24_rip.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_l32x2f24_rip(ae_int32x2* ae_ls_v,const ae_int32x2** ars) {
// CHECK-LABEL: test_ae_l32x2f24_rip
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <2 x i32>, ptr } @llvm.xtensa.ae.l32x2f24.rip(ptr %[[LD_ARS]])
// CHECK: %[[EV:.*]] = extractvalue { <2 x i32>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_l32x2f24_rip(ae_ls_v, ars);
}

//--- ae_l32x2f24_x.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_l32x2f24_x(const ae_int32x2* ars,int art) {
// CHECK-LABEL: test_ae_l32x2f24_x
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.l32x2f24.x(ptr {{.*}}, i32 {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_l32x2f24_x(ars, art);
}

//--- ae_l32x2f24_xc.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_l32x2f24_xc(ae_int32x2* ae_ls_v,const ae_int32x2** ars,int art) {
// CHECK-LABEL: test_ae_l32x2f24_xc
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <2 x i32>, ptr } @llvm.xtensa.ae.l32x2f24.xc(ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: %[[EV:.*]] = extractvalue { <2 x i32>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_l32x2f24_xc(ae_ls_v, ars, art);
}

//--- ae_l32x2f24_xp.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_l32x2f24_xp(ae_int32x2* ae_ls_v,const ae_int32x2** ars,int art) {
// CHECK-LABEL: test_ae_l32x2f24_xp
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <2 x i32>, ptr } @llvm.xtensa.ae.l32x2f24.xp(ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: %[[EV:.*]] = extractvalue { <2 x i32>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_l32x2f24_xp(ae_ls_v, ars, art);
}

//--- ae_l64_i.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_l64_i(const ae_int64* ars,immediate ae_immls64) {
// CHECK-LABEL: test_ae_l64_i
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.l64.i(ptr {{.*}}, i32 {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_l64_i(ars, -64);
}

//--- ae_l64_ip.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_l64_ip(ae_int64* ae_ls_v,const ae_int64** ars,immediate ae_immls64) {
// CHECK-LABEL: test_ae_l64_ip
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <1 x i64>, ptr } @llvm.xtensa.ae.l64.ip(ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: %[[EV:.*]] = extractvalue { <1 x i64>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_l64_ip(ae_ls_v, ars, -64);
}

//--- ae_l64_x.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_l64_x(const ae_int64* ars,int art) {
// CHECK-LABEL: test_ae_l64_x
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.l64.x(ptr {{.*}}, i32 {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_l64_x(ars, art);
}

//--- ae_l64_xc.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_l64_xc(ae_int64* ae_ls_v,const ae_int64** ars,int art) {
// CHECK-LABEL: test_ae_l64_xc
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <1 x i64>, ptr } @llvm.xtensa.ae.l64.xc(ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: %[[EV:.*]] = extractvalue { <1 x i64>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_l64_xc(ae_ls_v, ars, art);
}

//--- ae_l64_xp.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_l64_xp(ae_int64* ae_ls_v,const ae_int64** ars,int art) {
// CHECK-LABEL: test_ae_l64_xp
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <1 x i64>, ptr } @llvm.xtensa.ae.l64.xp(ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: %[[EV:.*]] = extractvalue { <1 x i64>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_l64_xp(ae_ls_v, ars, art);
}

//--- ae_la16x4_ic.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_la16x4_ic(ae_int16x4* ae_ls_av,ae_valign* ae_ls_uu,const ae_int16x4** ars) {
// CHECK-LABEL: test_ae_la16x4_ic
// CHECK: %[[LD_AE_LS_UU:.*]] = load <8 x i8>, ptr {{.*}}
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <4 x i16>, <8 x i8>, ptr } @llvm.xtensa.ae.la16x4.ic(<8 x i8> %[[LD_AE_LS_UU]], ptr %[[LD_ARS]])
// CHECK: %[[EV:.*]] = extractvalue { <4 x i16>, <8 x i8>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_la16x4_ic(ae_ls_av, ae_ls_uu, ars);
}

//--- ae_la16x4_ip.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_la16x4_ip(ae_int16x4* ae_ls_av,ae_valign* ae_ls_uu,const ae_int16x4** ars) {
// CHECK-LABEL: test_ae_la16x4_ip
// CHECK: %[[LD_AE_LS_UU:.*]] = load <8 x i8>, ptr {{.*}}
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <4 x i16>, <8 x i8>, ptr } @llvm.xtensa.ae.la16x4.ip(<8 x i8> %[[LD_AE_LS_UU]], ptr %[[LD_ARS]])
// CHECK: %[[EV:.*]] = extractvalue { <4 x i16>, <8 x i8>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_la16x4_ip(ae_ls_av, ae_ls_uu, ars);
}

//--- ae_la16x4_ric.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_la16x4_ric(ae_int16x4* ae_ls_av,ae_valign* ae_ls_uu,const ae_int16x4** ars) {
// CHECK-LABEL: test_ae_la16x4_ric
// CHECK: %[[LD_AE_LS_UU:.*]] = load <8 x i8>, ptr {{.*}}
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <4 x i16>, <8 x i8>, ptr } @llvm.xtensa.ae.la16x4.ric(<8 x i8> %[[LD_AE_LS_UU]], ptr %[[LD_ARS]])
// CHECK: %[[EV:.*]] = extractvalue { <4 x i16>, <8 x i8>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_la16x4_ric(ae_ls_av, ae_ls_uu, ars);
}

//--- ae_la16x4_rip.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_la16x4_rip(ae_int16x4* ae_ls_av,ae_valign* ae_ls_uu,const ae_int16x4** ars) {
// CHECK-LABEL: test_ae_la16x4_rip
// CHECK: %[[LD_AE_LS_UU:.*]] = load <8 x i8>, ptr {{.*}}
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <4 x i16>, <8 x i8>, ptr } @llvm.xtensa.ae.la16x4.rip(<8 x i8> %[[LD_AE_LS_UU]], ptr %[[LD_ARS]])
// CHECK: %[[EV:.*]] = extractvalue { <4 x i16>, <8 x i8>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_la16x4_rip(ae_ls_av, ae_ls_uu, ars);
}

//--- ae_la16x4neg_pc.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_la16x4neg_pc(ae_valign* ae_ls_uu,const ae_int16x4** ars) {
// CHECK-LABEL: test_ae_la16x4neg_pc
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <8 x i8>, ptr } @llvm.xtensa.ae.la16x4neg.pc(ptr %[[LD_ARS]])
// CHECK: %[[EV:.*]] = extractvalue { <8 x i8>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_la16x4neg_pc(ae_ls_uu, ars);
}

//--- ae_la16x4pos_pc.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_la16x4pos_pc(ae_valign* ae_ls_uu,const ae_int16x4** ars) {
// CHECK-LABEL: test_ae_la16x4pos_pc
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <8 x i8>, ptr } @llvm.xtensa.ae.la16x4pos.pc(ptr %[[LD_ARS]])
// CHECK: %[[EV:.*]] = extractvalue { <8 x i8>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_la16x4pos_pc(ae_ls_uu, ars);
}

//--- ae_la24_ic.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_la24_ic(ae_int32x2* ae_ls_av,ae_valign* ae_ls_uu,const void** ars) {
// CHECK-LABEL: test_ae_la24_ic
// CHECK: %[[LD_AE_LS_UU:.*]] = load <8 x i8>, ptr {{.*}}
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <2 x i32>, <8 x i8>, ptr } @llvm.xtensa.ae.la24.ic(<8 x i8> %[[LD_AE_LS_UU]], ptr %[[LD_ARS]])
// CHECK: %[[EV:.*]] = extractvalue { <2 x i32>, <8 x i8>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_la24_ic(ae_ls_av, ae_ls_uu, ars);
}

//--- ae_la24_ip.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_la24_ip(ae_int32x2* ae_ls_av,ae_valign* ae_ls_uu,const void** ars) {
// CHECK-LABEL: test_ae_la24_ip
// CHECK: %[[LD_AE_LS_UU:.*]] = load <8 x i8>, ptr {{.*}}
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <2 x i32>, <8 x i8>, ptr } @llvm.xtensa.ae.la24.ip(<8 x i8> %[[LD_AE_LS_UU]], ptr %[[LD_ARS]])
// CHECK: %[[EV:.*]] = extractvalue { <2 x i32>, <8 x i8>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_la24_ip(ae_ls_av, ae_ls_uu, ars);
}

//--- ae_la24_ric.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_la24_ric(ae_int32x2* ae_ls_av,ae_valign* ae_ls_uu,const void** ars) {
// CHECK-LABEL: test_ae_la24_ric
// CHECK: %[[LD_AE_LS_UU:.*]] = load <8 x i8>, ptr {{.*}}
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <2 x i32>, <8 x i8>, ptr } @llvm.xtensa.ae.la24.ric(<8 x i8> %[[LD_AE_LS_UU]], ptr %[[LD_ARS]])
// CHECK: %[[EV:.*]] = extractvalue { <2 x i32>, <8 x i8>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_la24_ric(ae_ls_av, ae_ls_uu, ars);
}

//--- ae_la24_rip.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_la24_rip(ae_int32x2* ae_ls_av,ae_valign* ae_ls_uu,const void** ars) {
// CHECK-LABEL: test_ae_la24_rip
// CHECK: %[[LD_AE_LS_UU:.*]] = load <8 x i8>, ptr {{.*}}
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <2 x i32>, <8 x i8>, ptr } @llvm.xtensa.ae.la24.rip(<8 x i8> %[[LD_AE_LS_UU]], ptr %[[LD_ARS]])
// CHECK: %[[EV:.*]] = extractvalue { <2 x i32>, <8 x i8>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_la24_rip(ae_ls_av, ae_ls_uu, ars);
}

//--- ae_la24neg_pc.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_la24neg_pc(ae_valign* ae_ls_uu,const void** ars) {
// CHECK-LABEL: test_ae_la24neg_pc
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <8 x i8>, ptr } @llvm.xtensa.ae.la24neg.pc(ptr %[[LD_ARS]])
// CHECK: %[[EV:.*]] = extractvalue { <8 x i8>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_la24neg_pc(ae_ls_uu, ars);
}

//--- ae_la24pos_pc.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_la24pos_pc(ae_valign* ae_ls_uu,const void** ars) {
// CHECK-LABEL: test_ae_la24pos_pc
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <8 x i8>, ptr } @llvm.xtensa.ae.la24pos.pc(ptr %[[LD_ARS]])
// CHECK: %[[EV:.*]] = extractvalue { <8 x i8>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_la24pos_pc(ae_ls_uu, ars);
}

//--- ae_la24x2_ic.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_la24x2_ic(ae_int32x2* ae_ls_av,ae_valign* ae_ls_uu,const void** ars) {
// CHECK-LABEL: test_ae_la24x2_ic
// CHECK: %[[LD_AE_LS_UU:.*]] = load <8 x i8>, ptr {{.*}}
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <2 x i32>, <8 x i8>, ptr } @llvm.xtensa.ae.la24x2.ic(<8 x i8> %[[LD_AE_LS_UU]], ptr %[[LD_ARS]])
// CHECK: %[[EV:.*]] = extractvalue { <2 x i32>, <8 x i8>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_la24x2_ic(ae_ls_av, ae_ls_uu, ars);
}

//--- ae_la24x2_ip.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_la24x2_ip(ae_int32x2* ae_ls_av,ae_valign* ae_ls_uu,const void** ars) {
// CHECK-LABEL: test_ae_la24x2_ip
// CHECK: %[[LD_AE_LS_UU:.*]] = load <8 x i8>, ptr {{.*}}
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <2 x i32>, <8 x i8>, ptr } @llvm.xtensa.ae.la24x2.ip(<8 x i8> %[[LD_AE_LS_UU]], ptr %[[LD_ARS]])
// CHECK: %[[EV:.*]] = extractvalue { <2 x i32>, <8 x i8>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_la24x2_ip(ae_ls_av, ae_ls_uu, ars);
}

//--- ae_la24x2_ric.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_la24x2_ric(ae_int32x2* ae_ls_av,ae_valign* ae_ls_uu,const void** ars) {
// CHECK-LABEL: test_ae_la24x2_ric
// CHECK: %[[LD_AE_LS_UU:.*]] = load <8 x i8>, ptr {{.*}}
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <2 x i32>, <8 x i8>, ptr } @llvm.xtensa.ae.la24x2.ric(<8 x i8> %[[LD_AE_LS_UU]], ptr %[[LD_ARS]])
// CHECK: %[[EV:.*]] = extractvalue { <2 x i32>, <8 x i8>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_la24x2_ric(ae_ls_av, ae_ls_uu, ars);
}

//--- ae_la24x2_rip.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_la24x2_rip(ae_int32x2* ae_ls_av,ae_valign* ae_ls_uu,const void** ars) {
// CHECK-LABEL: test_ae_la24x2_rip
// CHECK: %[[LD_AE_LS_UU:.*]] = load <8 x i8>, ptr {{.*}}
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <2 x i32>, <8 x i8>, ptr } @llvm.xtensa.ae.la24x2.rip(<8 x i8> %[[LD_AE_LS_UU]], ptr %[[LD_ARS]])
// CHECK: %[[EV:.*]] = extractvalue { <2 x i32>, <8 x i8>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_la24x2_rip(ae_ls_av, ae_ls_uu, ars);
}

//--- ae_la24x2neg_pc.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_la24x2neg_pc(ae_valign* ae_ls_uu,const void** ars) {
// CHECK-LABEL: test_ae_la24x2neg_pc
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <8 x i8>, ptr } @llvm.xtensa.ae.la24x2neg.pc(ptr %[[LD_ARS]])
// CHECK: %[[EV:.*]] = extractvalue { <8 x i8>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_la24x2neg_pc(ae_ls_uu, ars);
}

//--- ae_la24x2pos_pc.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_la24x2pos_pc(ae_valign* ae_ls_uu,const void** ars) {
// CHECK-LABEL: test_ae_la24x2pos_pc
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <8 x i8>, ptr } @llvm.xtensa.ae.la24x2pos.pc(ptr %[[LD_ARS]])
// CHECK: %[[EV:.*]] = extractvalue { <8 x i8>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_la24x2pos_pc(ae_ls_uu, ars);
}

//--- ae_la32x2_ic.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_la32x2_ic(ae_int32x2* ae_ls_av,ae_valign* ae_ls_uu,const ae_int32x2** ars) {
// CHECK-LABEL: test_ae_la32x2_ic
// CHECK: %[[LD_AE_LS_UU:.*]] = load <8 x i8>, ptr {{.*}}
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <2 x i32>, <8 x i8>, ptr } @llvm.xtensa.ae.la32x2.ic(<8 x i8> %[[LD_AE_LS_UU]], ptr %[[LD_ARS]])
// CHECK: %[[EV:.*]] = extractvalue { <2 x i32>, <8 x i8>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_la32x2_ic(ae_ls_av, ae_ls_uu, ars);
}

//--- ae_la32x2_ip.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_la32x2_ip(ae_int32x2* ae_ls_av,ae_valign* ae_ls_uu,const ae_int32x2** ars) {
// CHECK-LABEL: test_ae_la32x2_ip
// CHECK: %[[LD_AE_LS_UU:.*]] = load <8 x i8>, ptr {{.*}}
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <2 x i32>, <8 x i8>, ptr } @llvm.xtensa.ae.la32x2.ip(<8 x i8> %[[LD_AE_LS_UU]], ptr %[[LD_ARS]])
// CHECK: %[[EV:.*]] = extractvalue { <2 x i32>, <8 x i8>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_la32x2_ip(ae_ls_av, ae_ls_uu, ars);
}

//--- ae_la32x2_ric.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_la32x2_ric(ae_int32x2* ae_ls_av,ae_valign* ae_ls_uu,const ae_int32x2** ars) {
// CHECK-LABEL: test_ae_la32x2_ric
// CHECK: %[[LD_AE_LS_UU:.*]] = load <8 x i8>, ptr {{.*}}
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <2 x i32>, <8 x i8>, ptr } @llvm.xtensa.ae.la32x2.ric(<8 x i8> %[[LD_AE_LS_UU]], ptr %[[LD_ARS]])
// CHECK: %[[EV:.*]] = extractvalue { <2 x i32>, <8 x i8>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_la32x2_ric(ae_ls_av, ae_ls_uu, ars);
}

//--- ae_la32x2_rip.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_la32x2_rip(ae_int32x2* ae_ls_av,ae_valign* ae_ls_uu,const ae_int32x2** ars) {
// CHECK-LABEL: test_ae_la32x2_rip
// CHECK: %[[LD_AE_LS_UU:.*]] = load <8 x i8>, ptr {{.*}}
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <2 x i32>, <8 x i8>, ptr } @llvm.xtensa.ae.la32x2.rip(<8 x i8> %[[LD_AE_LS_UU]], ptr %[[LD_ARS]])
// CHECK: %[[EV:.*]] = extractvalue { <2 x i32>, <8 x i8>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_la32x2_rip(ae_ls_av, ae_ls_uu, ars);
}

//--- ae_la32x2f24_ic.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_la32x2f24_ic(ae_int32x2* ae_ls_av,ae_valign* ae_ls_uu,const ae_int32x2** ars) {
// CHECK-LABEL: test_ae_la32x2f24_ic
// CHECK: %[[LD_AE_LS_UU:.*]] = load <8 x i8>, ptr {{.*}}
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <2 x i32>, <8 x i8>, ptr } @llvm.xtensa.ae.la32x2f24.ic(<8 x i8> %[[LD_AE_LS_UU]], ptr %[[LD_ARS]])
// CHECK: %[[EV:.*]] = extractvalue { <2 x i32>, <8 x i8>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_la32x2f24_ic(ae_ls_av, ae_ls_uu, ars);
}

//--- ae_la32x2f24_ip.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_la32x2f24_ip(ae_int32x2* ae_ls_av,ae_valign* ae_ls_uu,const ae_int32x2** ars) {
// CHECK-LABEL: test_ae_la32x2f24_ip
// CHECK: %[[LD_AE_LS_UU:.*]] = load <8 x i8>, ptr {{.*}}
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <2 x i32>, <8 x i8>, ptr } @llvm.xtensa.ae.la32x2f24.ip(<8 x i8> %[[LD_AE_LS_UU]], ptr %[[LD_ARS]])
// CHECK: %[[EV:.*]] = extractvalue { <2 x i32>, <8 x i8>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_la32x2f24_ip(ae_ls_av, ae_ls_uu, ars);
}

//--- ae_la32x2f24_ric.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_la32x2f24_ric(ae_int32x2* ae_ls_av,ae_valign* ae_ls_uu,const ae_int32x2** ars) {
// CHECK-LABEL: test_ae_la32x2f24_ric
// CHECK: %[[LD_AE_LS_UU:.*]] = load <8 x i8>, ptr {{.*}}
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <2 x i32>, <8 x i8>, ptr } @llvm.xtensa.ae.la32x2f24.ric(<8 x i8> %[[LD_AE_LS_UU]], ptr %[[LD_ARS]])
// CHECK: %[[EV:.*]] = extractvalue { <2 x i32>, <8 x i8>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_la32x2f24_ric(ae_ls_av, ae_ls_uu, ars);
}

//--- ae_la32x2f24_rip.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_la32x2f24_rip(ae_int32x2* ae_ls_av,ae_valign* ae_ls_uu,const ae_int32x2** ars) {
// CHECK-LABEL: test_ae_la32x2f24_rip
// CHECK: %[[LD_AE_LS_UU:.*]] = load <8 x i8>, ptr {{.*}}
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <2 x i32>, <8 x i8>, ptr } @llvm.xtensa.ae.la32x2f24.rip(<8 x i8> %[[LD_AE_LS_UU]], ptr %[[LD_ARS]])
// CHECK: %[[EV:.*]] = extractvalue { <2 x i32>, <8 x i8>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_la32x2f24_rip(ae_ls_av, ae_ls_uu, ars);
}

//--- ae_la32x2neg_pc.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_la32x2neg_pc(ae_valign* ae_ls_uu,const ae_int32x2** ars) {
// CHECK-LABEL: test_ae_la32x2neg_pc
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <8 x i8>, ptr } @llvm.xtensa.ae.la32x2neg.pc(ptr %[[LD_ARS]])
// CHECK: %[[EV:.*]] = extractvalue { <8 x i8>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_la32x2neg_pc(ae_ls_uu, ars);
}

//--- ae_la32x2pos_pc.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_la32x2pos_pc(ae_valign* ae_ls_uu,const ae_int32x2** ars) {
// CHECK-LABEL: test_ae_la32x2pos_pc
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <8 x i8>, ptr } @llvm.xtensa.ae.la32x2pos.pc(ptr %[[LD_ARS]])
// CHECK: %[[EV:.*]] = extractvalue { <8 x i8>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_la32x2pos_pc(ae_ls_uu, ars);
}

//--- ae_la64_pp.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_valign test_ae_la64_pp(const void* ars) {
// CHECK-LABEL: test_ae_la64_pp
// CHECK: %[[RET:.*]] = {{(tail)?}} call <8 x i8> @llvm.xtensa.ae.la64.pp(ptr {{.*}})
// CHECK: %[[CAST:.*]] = bitcast <8 x i8> %[[RET]] to i64
// CHECK: ret i64 %[[CAST]]
return __builtin_xtensa_ae_la64_pp(ars);
}

//--- ae_lalign64_i.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_valign test_ae_lalign64_i(const ae_valign* ars,immediate ae_immls64) {
// CHECK-LABEL: test_ae_lalign64_i
// CHECK: %[[RET:.*]] = {{(tail)?}} call <8 x i8> @llvm.xtensa.ae.lalign64.i(ptr {{.*}}, i32 {{.*}})
// CHECK: %[[CAST:.*]] = bitcast <8 x i8> %[[RET]] to i64
// CHECK: ret i64 %[[CAST]]
return __builtin_xtensa_ae_lalign64_i(ars, -64);
}

//--- ae_lb.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

int test_ae_lb(int art) {
// CHECK-LABEL: test_ae_lb
// CHECK: %[[RET:.*]] = {{(tail)?}} call i32 @llvm.xtensa.ae.lb(i32 {{.*}})
// CHECK: ret i32 %[[RET]]
return __builtin_xtensa_ae_lb(art);
}

//--- ae_lbi.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

int test_ae_lbi(immediate ae_ohba) {
// CHECK-LABEL: test_ae_lbi
// CHECK: %[[RET:.*]] = {{(tail)?}} call i32 @llvm.xtensa.ae.lbi(i32 {{.*}})
// CHECK: ret i32 %[[RET]]
return __builtin_xtensa_ae_lbi(1);
}

//--- ae_lbk.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

int test_ae_lbk(int ars,int art) {
// CHECK-LABEL: test_ae_lbk
// CHECK: %[[RET:.*]] = {{(tail)?}} call i32 @llvm.xtensa.ae.lbk(i32 {{.*}}, i32 {{.*}})
// CHECK: ret i32 %[[RET]]
return __builtin_xtensa_ae_lbk(ars, art);
}

//--- ae_lbki.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

int test_ae_lbki(int ars,immediate ae_ohba) {
// CHECK-LABEL: test_ae_lbki
// CHECK: %[[RET:.*]] = {{(tail)?}} call i32 @llvm.xtensa.ae.lbki(i32 {{.*}}, i32 {{.*}})
// CHECK: ret i32 %[[RET]]
return __builtin_xtensa_ae_lbki(ars, 1);
}

//--- ae_lbs.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

int test_ae_lbs(int art) {
// CHECK-LABEL: test_ae_lbs
// CHECK: %[[RET:.*]] = {{(tail)?}} call i32 @llvm.xtensa.ae.lbs(i32 {{.*}})
// CHECK: ret i32 %[[RET]]
return __builtin_xtensa_ae_lbs(art);
}

//--- ae_lbsi.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

int test_ae_lbsi(immediate ae_ohba) {
// CHECK-LABEL: test_ae_lbsi
// CHECK: %[[RET:.*]] = {{(tail)?}} call i32 @llvm.xtensa.ae.lbsi(i32 {{.*}})
// CHECK: ret i32 %[[RET]]
return __builtin_xtensa_ae_lbsi(1);
}

//--- ae_le16.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

xtbool4 test_ae_le16(ae_int16x4 ae_cmpp_v0,ae_int16x4 ae_cmpp_v1) {
// CHECK-LABEL: test_ae_le16
// CHECK: %[[RET:.*]] = {{(tail)?}} call <4 x i1> @llvm.xtensa.ae.le16(<4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <4 x i1> %[[RET]]
return __builtin_xtensa_ae_le16(ae_cmpp_v0, ae_cmpp_v1);
}

//--- ae_le32.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

xtbool2 test_ae_le32(ae_int32x2 ae_cmpp_v0,ae_int32x2 ae_cmpp_v1) {
// CHECK-LABEL: test_ae_le32
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i1> @llvm.xtensa.ae.le32(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <2 x i1> %[[RET]]
return __builtin_xtensa_ae_le32(ae_cmpp_v0, ae_cmpp_v1);
}

//--- ae_le64.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

xtbool test_ae_le64(ae_int64 ae_cmpp_v0,ae_int64 ae_cmpp_v1) {
// CHECK-LABEL: test_ae_le64
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i1> @llvm.xtensa.ae.le64(<1 x i64> {{.*}}, <1 x i64> {{.*}})
// CHECK: ret <1 x i1> %[[RET]]
return __builtin_xtensa_ae_le64(ae_cmpp_v0, ae_cmpp_v1);
}

//--- ae_lt16.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

xtbool4 test_ae_lt16(ae_int16x4 ae_cmpp_v0,ae_int16x4 ae_cmpp_v1) {
// CHECK-LABEL: test_ae_lt16
// CHECK: %[[RET:.*]] = {{(tail)?}} call <4 x i1> @llvm.xtensa.ae.lt16(<4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <4 x i1> %[[RET]]
return __builtin_xtensa_ae_lt16(ae_cmpp_v0, ae_cmpp_v1);
}

//--- ae_lt32.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

xtbool2 test_ae_lt32(ae_int32x2 ae_cmpp_v0,ae_int32x2 ae_cmpp_v1) {
// CHECK-LABEL: test_ae_lt32
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i1> @llvm.xtensa.ae.lt32(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <2 x i1> %[[RET]]
return __builtin_xtensa_ae_lt32(ae_cmpp_v0, ae_cmpp_v1);
}

//--- ae_lt64.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

xtbool test_ae_lt64(ae_int64 ae_cmpp_v0,ae_int64 ae_cmpp_v1) {
// CHECK-LABEL: test_ae_lt64
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i1> @llvm.xtensa.ae.lt64(<1 x i64> {{.*}}, <1 x i64> {{.*}})
// CHECK: ret <1 x i1> %[[RET]]
return __builtin_xtensa_ae_lt64(ae_cmpp_v0, ae_cmpp_v1);
}

//--- ae_max32.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_max32(ae_int32x2 ae_cmpp_v0,ae_int32x2 ae_cmpp_v1) {
// CHECK-LABEL: test_ae_max32
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.max32(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_max32(ae_cmpp_v0, ae_cmpp_v1);
}

//--- ae_max64.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_max64(ae_int64 ae_cmpp_v0,ae_int64 ae_cmpp_v1) {
// CHECK-LABEL: test_ae_max64
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.max64(<1 x i64> {{.*}}, <1 x i64> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_max64(ae_cmpp_v0, ae_cmpp_v1);
}

//--- ae_maxabs32s.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_maxabs32s(ae_int32x2 ae_cmpp_v0,ae_int32x2 ae_cmpp_v1) {
// CHECK-LABEL: test_ae_maxabs32s
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.maxabs32s(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_maxabs32s(ae_cmpp_v0, ae_cmpp_v1);
}

//--- ae_maxabs64s.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_maxabs64s(ae_int64 ae_cmpp_v0,ae_int64 ae_cmpp_v1) {
// CHECK-LABEL: test_ae_maxabs64s
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.maxabs64s(<1 x i64> {{.*}}, <1 x i64> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_maxabs64s(ae_cmpp_v0, ae_cmpp_v1);
}

//--- ae_min32.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_min32(ae_int32x2 ae_cmpp_v0,ae_int32x2 ae_cmpp_v1) {
// CHECK-LABEL: test_ae_min32
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.min32(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_min32(ae_cmpp_v0, ae_cmpp_v1);
}

//--- ae_min64.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_min64(ae_int64 ae_cmpp_v0,ae_int64 ae_cmpp_v1) {
// CHECK-LABEL: test_ae_min64
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.min64(<1 x i64> {{.*}}, <1 x i64> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_min64(ae_cmpp_v0, ae_cmpp_v1);
}

//--- ae_minabs32s.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_minabs32s(ae_int32x2 ae_cmpp_v0,ae_int32x2 ae_cmpp_v1) {
// CHECK-LABEL: test_ae_minabs32s
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.minabs32s(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_minabs32s(ae_cmpp_v0, ae_cmpp_v1);
}

//--- ae_minabs64s.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_minabs64s(ae_int64 ae_cmpp_v0,ae_int64 ae_cmpp_v1) {
// CHECK-LABEL: test_ae_minabs64s
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.minabs64s(<1 x i64> {{.*}}, <1 x i64> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_minabs64s(ae_cmpp_v0, ae_cmpp_v1);
}

//--- ae_mov.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mov(ae_int64 ae_to_dr_v0) {
// CHECK-LABEL: test_ae_mov
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mov(<1 x i64> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mov(ae_to_dr_v0);
}

//--- ae_movad16_0.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

int test_ae_movad16_0(ae_int16x4 ae_dr_to_ar_v0) {
// CHECK-LABEL: test_ae_movad16_0
// CHECK: %[[RET:.*]] = {{(tail)?}} call i32 @llvm.xtensa.ae.movad16.0(<4 x i16> {{.*}})
// CHECK: ret i32 %[[RET]]
return __builtin_xtensa_ae_movad16_0(ae_dr_to_ar_v0);
}

//--- ae_movad16_1.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

int test_ae_movad16_1(ae_int16x4 ae_dr_to_ar_v0) {
// CHECK-LABEL: test_ae_movad16_1
// CHECK: %[[RET:.*]] = {{(tail)?}} call i32 @llvm.xtensa.ae.movad16.1(<4 x i16> {{.*}})
// CHECK: ret i32 %[[RET]]
return __builtin_xtensa_ae_movad16_1(ae_dr_to_ar_v0);
}

//--- ae_movad16_2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

int test_ae_movad16_2(ae_int16x4 ae_dr_to_ar_v0) {
// CHECK-LABEL: test_ae_movad16_2
// CHECK: %[[RET:.*]] = {{(tail)?}} call i32 @llvm.xtensa.ae.movad16.2(<4 x i16> {{.*}})
// CHECK: ret i32 %[[RET]]
return __builtin_xtensa_ae_movad16_2(ae_dr_to_ar_v0);
}

//--- ae_movad16_3.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

int test_ae_movad16_3(ae_int16x4 ae_dr_to_ar_v0) {
// CHECK-LABEL: test_ae_movad16_3
// CHECK: %[[RET:.*]] = {{(tail)?}} call i32 @llvm.xtensa.ae.movad16.3(<4 x i16> {{.*}})
// CHECK: ret i32 %[[RET]]
return __builtin_xtensa_ae_movad16_3(ae_dr_to_ar_v0);
}

//--- ae_movad32_h.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

int test_ae_movad32_h(ae_int32x2 ae_dr_to_ar_v0) {
// CHECK-LABEL: test_ae_movad32_h
// CHECK: %[[RET:.*]] = {{(tail)?}} call i32 @llvm.xtensa.ae.movad32.h(<2 x i32> {{.*}})
// CHECK: ret i32 %[[RET]]
return __builtin_xtensa_ae_movad32_h(ae_dr_to_ar_v0);
}

//--- ae_movad32_l.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

int test_ae_movad32_l(ae_int32x2 ae_dr_to_ar_v0) {
// CHECK-LABEL: test_ae_movad32_l
// CHECK: %[[RET:.*]] = {{(tail)?}} call i32 @llvm.xtensa.ae.movad32.l(<2 x i32> {{.*}})
// CHECK: ret i32 %[[RET]]
return __builtin_xtensa_ae_movad32_l(ae_dr_to_ar_v0);
}

//--- ae_movalign.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_valign test_ae_movalign(ae_valign ae_uu_v) {
// CHECK-LABEL: test_ae_movalign
// CHECK: %[[RET:.*]] = {{(tail)?}} call <8 x i8> @llvm.xtensa.ae.movalign(<8 x i8> {{.*}})
// CHECK: %[[CAST:.*]] = bitcast <8 x i8> %[[RET]] to i64
// CHECK: ret i64 %[[CAST]]
return __builtin_xtensa_ae_movalign(ae_uu_v);
}

//--- ae_movda16.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int16x4 test_ae_movda16(int ars) {
// CHECK-LABEL: test_ae_movda16
// CHECK: %[[RET:.*]] = {{(tail)?}} call <4 x i16> @llvm.xtensa.ae.movda16(i32 {{.*}})
// CHECK: ret <4 x i16> %[[RET]]
return __builtin_xtensa_ae_movda16(ars);
}

//--- ae_movda16x2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int16x4 test_ae_movda16x2(int ars,int art) {
// CHECK-LABEL: test_ae_movda16x2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <4 x i16> @llvm.xtensa.ae.movda16x2(i32 {{.*}}, i32 {{.*}})
// CHECK: ret <4 x i16> %[[RET]]
return __builtin_xtensa_ae_movda16x2(ars, art);
}

//--- ae_movda32.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32 test_ae_movda32(int ars) {
// CHECK-LABEL: test_ae_movda32
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i32> @llvm.xtensa.ae.movda32(i32 {{.*}})
// CHECK: ret <1 x i32> %[[RET]]
return __builtin_xtensa_ae_movda32(ars);
}

//--- ae_movda32x2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_movda32x2(int ars,int art) {
// CHECK-LABEL: test_ae_movda32x2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.movda32x2(i32 {{.*}}, i32 {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_movda32x2(ars, art);
}

//--- ae_movf16x4.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_movf16x4(ae_int16x4* ae_cmov_v,ae_int16x4 ae_cmov_v0,xtbool4 bt4) {
// CHECK-LABEL: test_ae_movf16x4
// CHECK: %[[LD_AE_CMOV_V:.*]] = load <4 x i16>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <4 x i16> @llvm.xtensa.ae.movf16x4(<4 x i16> %[[LD_AE_CMOV_V]], <4 x i16> {{.*}}, <4 x i1> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_movf16x4(ae_cmov_v, ae_cmov_v0, bt4);
}

//--- ae_movf32x2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_movf32x2(ae_int32x2* ae_cmov_v,ae_int32x2 ae_cmov_v0,xtbool2 bt2) {
// CHECK-LABEL: test_ae_movf32x2
// CHECK: %[[LD_AE_CMOV_V:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.movf32x2(<2 x i32> %[[LD_AE_CMOV_V]], <2 x i32> {{.*}}, <2 x i1> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_movf32x2(ae_cmov_v, ae_cmov_v0, bt2);
}

//--- ae_movf64.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_movf64(ae_int64* ae_cmov_v,ae_int64 ae_cmov_v0,xtbool bt) {
// CHECK-LABEL: test_ae_movf64
// CHECK: %[[LD_AE_CMOV_V:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.movf64(<1 x i64> %[[LD_AE_CMOV_V]], <1 x i64> {{.*}}, <1 x i1> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_movf64(ae_cmov_v, ae_cmov_v0, bt);
}

//--- ae_movi.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_movi(immediate movi_imm) {
// CHECK-LABEL: test_ae_movi
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.movi(i32 {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_movi(-16);
}

//--- ae_movt16x4.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_movt16x4(ae_int16x4* ae_cmov_v,ae_int16x4 ae_cmov_v0,xtbool4 bt4) {
// CHECK-LABEL: test_ae_movt16x4
// CHECK: %[[LD_AE_CMOV_V:.*]] = load <4 x i16>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <4 x i16> @llvm.xtensa.ae.movt16x4(<4 x i16> %[[LD_AE_CMOV_V]], <4 x i16> {{.*}}, <4 x i1> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_movt16x4(ae_cmov_v, ae_cmov_v0, bt4);
}

//--- ae_movt32x2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_movt32x2(ae_int32x2* ae_cmov_v,ae_int32x2 ae_cmov_v0,xtbool2 bt2) {
// CHECK-LABEL: test_ae_movt32x2
// CHECK: %[[LD_AE_CMOV_V:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.movt32x2(<2 x i32> %[[LD_AE_CMOV_V]], <2 x i32> {{.*}}, <2 x i1> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_movt32x2(ae_cmov_v, ae_cmov_v0, bt2);
}

//--- ae_movt64.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_movt64(ae_int64* ae_cmov_v,ae_int64 ae_cmov_v0,xtbool bt) {
// CHECK-LABEL: test_ae_movt64
// CHECK: %[[LD_AE_CMOV_V:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.movt64(<1 x i64> %[[LD_AE_CMOV_V]], <1 x i64> {{.*}}, <1 x i1> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_movt64(ae_cmov_v, ae_cmov_v0, bt);
}

//--- ae_mul16x4.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mul16x4(ae_int32x2* ae_mul_q1,ae_int32x2* ae_mul_q0,ae_int16x4 ae_mul_d1,ae_int16x4 ae_mul_d0) {
// CHECK-LABEL: test_ae_mul16x4
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <2 x i32>, <2 x i32> } @llvm.xtensa.ae.mul16x4(<4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: %[[EV:.*]] = extractvalue { <2 x i32>, <2 x i32> } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_mul16x4(ae_mul_q1, ae_mul_q0, ae_mul_d1, ae_mul_d0);
}

//--- ae_mul32_hh.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mul32_hh(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mul32_hh
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mul32.hh(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mul32_hh(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mul32_lh.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mul32_lh(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mul32_lh
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mul32.lh(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mul32_lh(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mul32_ll.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mul32_ll(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mul32_ll
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mul32.ll(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mul32_ll(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mul32_ll_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mul32_ll_s2(ae_int32x2 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mul32_ll_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mul32.ll.s2(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mul32_ll_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mul32u_ll.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mul32u_ll(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mul32u_ll
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mul32u.ll(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mul32u_ll(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mul32x16_h0.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mul32x16_h0(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mul32x16_h0
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mul32x16.h0(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mul32x16_h0(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mul32x16_h0_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mul32x16_h0_s2(ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mul32x16_h0_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mul32x16.h0.s2(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mul32x16_h0_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mul32x16_h1.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mul32x16_h1(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mul32x16_h1
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mul32x16.h1(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mul32x16_h1(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mul32x16_h1_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mul32x16_h1_s2(ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mul32x16_h1_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mul32x16.h1.s2(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mul32x16_h1_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mul32x16_h2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mul32x16_h2(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mul32x16_h2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mul32x16.h2(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mul32x16_h2(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mul32x16_h2_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mul32x16_h2_s2(ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mul32x16_h2_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mul32x16.h2.s2(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mul32x16_h2_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mul32x16_h3.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mul32x16_h3(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mul32x16_h3
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mul32x16.h3(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mul32x16_h3(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mul32x16_h3_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mul32x16_h3_s2(ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mul32x16_h3_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mul32x16.h3.s2(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mul32x16_h3_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mul32x16_l0.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mul32x16_l0(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mul32x16_l0
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mul32x16.l0(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mul32x16_l0(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mul32x16_l0_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mul32x16_l0_s2(ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mul32x16_l0_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mul32x16.l0.s2(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mul32x16_l0_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mul32x16_l1.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mul32x16_l1(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mul32x16_l1
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mul32x16.l1(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mul32x16_l1(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mul32x16_l1_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mul32x16_l1_s2(ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mul32x16_l1_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mul32x16.l1.s2(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mul32x16_l1_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mul32x16_l2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mul32x16_l2(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mul32x16_l2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mul32x16.l2(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mul32x16_l2(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mul32x16_l2_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mul32x16_l2_s2(ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mul32x16_l2_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mul32x16.l2.s2(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mul32x16_l2_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mul32x16_l3.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mul32x16_l3(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mul32x16_l3
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mul32x16.l3(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mul32x16_l3(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mul32x16_l3_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mul32x16_l3_s2(ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mul32x16_l3_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mul32x16.l3.s2(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mul32x16_l3_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mula16x4.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mula16x4(ae_int32x2* ae_mul_q1,ae_int32x2* ae_mul_q0,ae_int16x4 ae_mul_d1,ae_int16x4 ae_mul_d0) {
// CHECK-LABEL: test_ae_mula16x4
// CHECK: %[[LD_AE_MUL_Q1:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[LD_AE_MUL_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <2 x i32>, <2 x i32> } @llvm.xtensa.ae.mula16x4(<2 x i32> %[[LD_AE_MUL_Q1]], <2 x i32> %[[LD_AE_MUL_Q0]], <4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: %[[EV:.*]] = extractvalue { <2 x i32>, <2 x i32> } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_mula16x4(ae_mul_q1, ae_mul_q0, ae_mul_d1, ae_mul_d0);
}

//--- ae_mula32_hh.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mula32_hh(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mula32_hh
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mula32.hh(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mula32_hh(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mula32_lh.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mula32_lh(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mula32_lh
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mula32.lh(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mula32_lh(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mula32_ll.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mula32_ll(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mula32_ll
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mula32.ll(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mula32_ll(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mula32_ll_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mula32_ll_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mula32_ll_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mula32.ll.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mula32_ll_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mula32u_ll.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mula32u_ll(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mula32u_ll
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mula32u.ll(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mula32u_ll(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mula32x16_h0.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mula32x16_h0(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mula32x16_h0
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mula32x16.h0(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mula32x16_h0(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mula32x16_h0_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mula32x16_h0_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mula32x16_h0_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mula32x16.h0.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mula32x16_h0_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mula32x16_h1.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mula32x16_h1(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mula32x16_h1
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mula32x16.h1(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mula32x16_h1(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mula32x16_h1_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mula32x16_h1_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mula32x16_h1_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mula32x16.h1.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mula32x16_h1_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mula32x16_h2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mula32x16_h2(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mula32x16_h2
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mula32x16.h2(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mula32x16_h2(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mula32x16_h2_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mula32x16_h2_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mula32x16_h2_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mula32x16.h2.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mula32x16_h2_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mula32x16_h3.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mula32x16_h3(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mula32x16_h3
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mula32x16.h3(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mula32x16_h3(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mula32x16_h3_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mula32x16_h3_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mula32x16_h3_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mula32x16.h3.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mula32x16_h3_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mula32x16_l0.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mula32x16_l0(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mula32x16_l0
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mula32x16.l0(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mula32x16_l0(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mula32x16_l0_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mula32x16_l0_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mula32x16_l0_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mula32x16.l0.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mula32x16_l0_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mula32x16_l1.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mula32x16_l1(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mula32x16_l1
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mula32x16.l1(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mula32x16_l1(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mula32x16_l1_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mula32x16_l1_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mula32x16_l1_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mula32x16.l1.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mula32x16_l1_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mula32x16_l2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mula32x16_l2(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mula32x16_l2
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mula32x16.l2(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mula32x16_l2(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mula32x16_l2_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mula32x16_l2_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mula32x16_l2_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mula32x16.l2.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mula32x16_l2_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mula32x16_l3.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mula32x16_l3(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mula32x16_l3
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mula32x16.l3(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mula32x16_l3(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mula32x16_l3_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mula32x16_l3_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mula32x16_l3_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mula32x16.l3.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mula32x16_l3_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulaad24_hh_ll.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaad24_hh_ll(ae_int64* ae_mul_q0,ae_int32x2 ae_mul_d0,ae_int32x2 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulaad24_hh_ll
// CHECK: %[[LD_AE_MUL_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulaad24.hh.ll(<1 x i64> %[[LD_AE_MUL_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaad24_hh_ll(ae_mul_q0, ae_mul_d0, ae_mul_d1);
}

//--- ae_mulaad24_hh_ll_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaad24_hh_ll_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulaad24_hh_ll_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulaad24.hh.ll.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaad24_hh_ll_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulaad24_hl_lh.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaad24_hl_lh(ae_int64* ae_mul_q0,ae_int32x2 ae_mul_d0,ae_int32x2 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulaad24_hl_lh
// CHECK: %[[LD_AE_MUL_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulaad24.hl.lh(<1 x i64> %[[LD_AE_MUL_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaad24_hl_lh(ae_mul_q0, ae_mul_d0, ae_mul_d1);
}

//--- ae_mulaad24_hl_lh_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaad24_hl_lh_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulaad24_hl_lh_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulaad24.hl.lh.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaad24_hl_lh_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulaad32x16_h0_l1.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaad32x16_h0_l1(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulaad32x16_h0_l1
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulaad32x16.h0.l1(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaad32x16_h0_l1(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulaad32x16_h0_l1_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaad32x16_h0_l1_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulaad32x16_h0_l1_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulaad32x16.h0.l1.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaad32x16_h0_l1_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulaad32x16_h1_l0.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaad32x16_h1_l0(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulaad32x16_h1_l0
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulaad32x16.h1.l0(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaad32x16_h1_l0(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulaad32x16_h1_l0_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaad32x16_h1_l0_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulaad32x16_h1_l0_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulaad32x16.h1.l0.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaad32x16_h1_l0_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulaad32x16_h2_l3.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaad32x16_h2_l3(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulaad32x16_h2_l3
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulaad32x16.h2.l3(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaad32x16_h2_l3(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulaad32x16_h2_l3_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaad32x16_h2_l3_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulaad32x16_h2_l3_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulaad32x16.h2.l3.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaad32x16_h2_l3_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulaad32x16_h3_l2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaad32x16_h3_l2(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulaad32x16_h3_l2
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulaad32x16.h3.l2(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaad32x16_h3_l2(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulaad32x16_h3_l2_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaad32x16_h3_l2_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulaad32x16_h3_l2_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulaad32x16.h3.l2.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaad32x16_h3_l2_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulaafd16ss_11_00.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaafd16ss_11_00(ae_int32x2* opnd_ae_sem_mul_x2_S1_q0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulaafd16ss_11_00
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulaafd16ss.11.00(<2 x i32> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaafd16ss_11_00(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulaafd16ss_11_00_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaafd16ss_11_00_s2(ae_int32x2* ae_mul_S2_q0,ae_int16x4 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulaafd16ss_11_00_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulaafd16ss.11.00.s2(<2 x i32> %[[LD_AE_MUL_S2_Q0]], <4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaafd16ss_11_00_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulaafd16ss_13_02.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaafd16ss_13_02(ae_int32x2* opnd_ae_sem_mul_x2_S1_q0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulaafd16ss_13_02
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulaafd16ss.13.02(<2 x i32> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaafd16ss_13_02(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulaafd16ss_13_02_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaafd16ss_13_02_s2(ae_int32x2* ae_mul_S2_q0,ae_int16x4 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulaafd16ss_13_02_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulaafd16ss.13.02.s2(<2 x i32> %[[LD_AE_MUL_S2_Q0]], <4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaafd16ss_13_02_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulaafd16ss_33_22.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaafd16ss_33_22(ae_int32x2* opnd_ae_sem_mul_x2_S1_q0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulaafd16ss_33_22
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulaafd16ss.33.22(<2 x i32> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaafd16ss_33_22(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulaafd16ss_33_22_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaafd16ss_33_22_s2(ae_int32x2* ae_mul_S2_q0,ae_int16x4 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulaafd16ss_33_22_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulaafd16ss.33.22.s2(<2 x i32> %[[LD_AE_MUL_S2_Q0]], <4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaafd16ss_33_22_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulaafd24_hh_ll.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaafd24_hh_ll(ae_int64* ae_mul_q0,ae_int32x2 ae_mul_d0,ae_int32x2 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulaafd24_hh_ll
// CHECK: %[[LD_AE_MUL_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulaafd24.hh.ll(<1 x i64> %[[LD_AE_MUL_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaafd24_hh_ll(ae_mul_q0, ae_mul_d0, ae_mul_d1);
}

//--- ae_mulaafd24_hh_ll_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaafd24_hh_ll_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulaafd24_hh_ll_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulaafd24.hh.ll.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaafd24_hh_ll_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulaafd24_hl_lh.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaafd24_hl_lh(ae_int64* ae_mul_q0,ae_int32x2 ae_mul_d0,ae_int32x2 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulaafd24_hl_lh
// CHECK: %[[LD_AE_MUL_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulaafd24.hl.lh(<1 x i64> %[[LD_AE_MUL_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaafd24_hl_lh(ae_mul_q0, ae_mul_d0, ae_mul_d1);
}

//--- ae_mulaafd24_hl_lh_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaafd24_hl_lh_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulaafd24_hl_lh_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulaafd24.hl.lh.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaafd24_hl_lh_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulaafd32x16_h0_l1.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaafd32x16_h0_l1(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulaafd32x16_h0_l1
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulaafd32x16.h0.l1(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaafd32x16_h0_l1(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulaafd32x16_h0_l1_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaafd32x16_h0_l1_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulaafd32x16_h0_l1_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulaafd32x16.h0.l1.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaafd32x16_h0_l1_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulaafd32x16_h1_l0.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaafd32x16_h1_l0(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulaafd32x16_h1_l0
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulaafd32x16.h1.l0(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaafd32x16_h1_l0(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulaafd32x16_h1_l0_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaafd32x16_h1_l0_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulaafd32x16_h1_l0_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulaafd32x16.h1.l0.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaafd32x16_h1_l0_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulaafd32x16_h2_l3.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaafd32x16_h2_l3(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulaafd32x16_h2_l3
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulaafd32x16.h2.l3(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaafd32x16_h2_l3(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulaafd32x16_h2_l3_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaafd32x16_h2_l3_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulaafd32x16_h2_l3_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulaafd32x16.h2.l3.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaafd32x16_h2_l3_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulaafd32x16_h3_l2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaafd32x16_h3_l2(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulaafd32x16_h3_l2
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulaafd32x16.h3.l2(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaafd32x16_h3_l2(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulaafd32x16_h3_l2_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaafd32x16_h3_l2_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulaafd32x16_h3_l2_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulaafd32x16.h3.l2.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaafd32x16_h3_l2_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulac24.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulac24(ae_int32x2* ae_mul_q0,ae_int32x2 ae_mul_d0,ae_int32x2 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulac24
// CHECK: %[[LD_AE_MUL_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulac24(<2 x i32> %[[LD_AE_MUL_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulac24(ae_mul_q0, ae_mul_d0, ae_mul_d1);
}

//--- ae_mulac32x16_h.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulac32x16_h(ae_int32x2* opnd_ae_sem_mul_x4_q0,ae_int32x2 opnd_ae_sem_mul_x4_d0,ae_int16x4 opnd_ae_sem_mul_x4_d1) {
// CHECK-LABEL: test_ae_mulac32x16_h
// CHECK: %[[LD_OPND_AE_SEM_MUL_X4_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulac32x16.h(<2 x i32> %[[LD_OPND_AE_SEM_MUL_X4_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulac32x16_h(opnd_ae_sem_mul_x4_q0, opnd_ae_sem_mul_x4_d0, opnd_ae_sem_mul_x4_d1);
}

//--- ae_mulac32x16_l.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulac32x16_l(ae_int32x2* opnd_ae_sem_mul_x4_q0,ae_int32x2 opnd_ae_sem_mul_x4_d0,ae_int16x4 opnd_ae_sem_mul_x4_d1) {
// CHECK-LABEL: test_ae_mulac32x16_l
// CHECK: %[[LD_OPND_AE_SEM_MUL_X4_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulac32x16.l(<2 x i32> %[[LD_OPND_AE_SEM_MUL_X4_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulac32x16_l(opnd_ae_sem_mul_x4_q0, opnd_ae_sem_mul_x4_d0, opnd_ae_sem_mul_x4_d1);
}

//--- ae_mulaf16ss_00.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaf16ss_00(ae_int32x2* opnd_ae_sem_mul_x2_S1_q0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulaf16ss_00
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulaf16ss.00(<2 x i32> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaf16ss_00(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulaf16ss_00_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaf16ss_00_s2(ae_int32x2* ae_mul_S2_q0,ae_int16x4 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulaf16ss_00_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulaf16ss.00.s2(<2 x i32> %[[LD_AE_MUL_S2_Q0]], <4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaf16ss_00_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulaf16ss_10.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaf16ss_10(ae_int32x2* ae_mul_q0,ae_int16x4 ae_mul_d0,ae_int16x4 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulaf16ss_10
// CHECK: %[[LD_AE_MUL_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulaf16ss.10(<2 x i32> %[[LD_AE_MUL_Q0]], <4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaf16ss_10(ae_mul_q0, ae_mul_d0, ae_mul_d1);
}

//--- ae_mulaf16ss_11.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaf16ss_11(ae_int32x2* ae_mul_q0,ae_int16x4 ae_mul_d0,ae_int16x4 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulaf16ss_11
// CHECK: %[[LD_AE_MUL_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulaf16ss.11(<2 x i32> %[[LD_AE_MUL_Q0]], <4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaf16ss_11(ae_mul_q0, ae_mul_d0, ae_mul_d1);
}

//--- ae_mulaf16ss_20.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaf16ss_20(ae_int32x2* ae_mul_q0,ae_int16x4 ae_mul_d0,ae_int16x4 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulaf16ss_20
// CHECK: %[[LD_AE_MUL_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulaf16ss.20(<2 x i32> %[[LD_AE_MUL_Q0]], <4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaf16ss_20(ae_mul_q0, ae_mul_d0, ae_mul_d1);
}

//--- ae_mulaf16ss_21.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaf16ss_21(ae_int32x2* ae_mul_q0,ae_int16x4 ae_mul_d0,ae_int16x4 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulaf16ss_21
// CHECK: %[[LD_AE_MUL_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulaf16ss.21(<2 x i32> %[[LD_AE_MUL_Q0]], <4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaf16ss_21(ae_mul_q0, ae_mul_d0, ae_mul_d1);
}

//--- ae_mulaf16ss_22.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaf16ss_22(ae_int32x2* ae_mul_q0,ae_int16x4 ae_mul_d0,ae_int16x4 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulaf16ss_22
// CHECK: %[[LD_AE_MUL_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulaf16ss.22(<2 x i32> %[[LD_AE_MUL_Q0]], <4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaf16ss_22(ae_mul_q0, ae_mul_d0, ae_mul_d1);
}

//--- ae_mulaf16ss_30.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaf16ss_30(ae_int32x2* ae_mul_q0,ae_int16x4 ae_mul_d0,ae_int16x4 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulaf16ss_30
// CHECK: %[[LD_AE_MUL_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulaf16ss.30(<2 x i32> %[[LD_AE_MUL_Q0]], <4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaf16ss_30(ae_mul_q0, ae_mul_d0, ae_mul_d1);
}

//--- ae_mulaf16ss_31.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaf16ss_31(ae_int32x2* ae_mul_q0,ae_int16x4 ae_mul_d0,ae_int16x4 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulaf16ss_31
// CHECK: %[[LD_AE_MUL_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulaf16ss.31(<2 x i32> %[[LD_AE_MUL_Q0]], <4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaf16ss_31(ae_mul_q0, ae_mul_d0, ae_mul_d1);
}

//--- ae_mulaf16ss_32.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaf16ss_32(ae_int32x2* ae_mul_q0,ae_int16x4 ae_mul_d0,ae_int16x4 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulaf16ss_32
// CHECK: %[[LD_AE_MUL_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulaf16ss.32(<2 x i32> %[[LD_AE_MUL_Q0]], <4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaf16ss_32(ae_mul_q0, ae_mul_d0, ae_mul_d1);
}

//--- ae_mulaf16ss_33.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaf16ss_33(ae_int32x2* ae_mul_q0,ae_int16x4 ae_mul_d0,ae_int16x4 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulaf16ss_33
// CHECK: %[[LD_AE_MUL_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulaf16ss.33(<2 x i32> %[[LD_AE_MUL_Q0]], <4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaf16ss_33(ae_mul_q0, ae_mul_d0, ae_mul_d1);
}

//--- ae_mulaf16x4ss.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaf16x4ss(ae_int32x2* ae_mul_q1,ae_int32x2* ae_mul_q0,ae_int16x4 ae_mul_d1,ae_int16x4 ae_mul_d0) {
// CHECK-LABEL: test_ae_mulaf16x4ss
// CHECK: %[[LD_AE_MUL_Q1:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[LD_AE_MUL_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <2 x i32>, <2 x i32> } @llvm.xtensa.ae.mulaf16x4ss(<2 x i32> %[[LD_AE_MUL_Q1]], <2 x i32> %[[LD_AE_MUL_Q0]], <4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: %[[EV:.*]] = extractvalue { <2 x i32>, <2 x i32> } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_mulaf16x4ss(ae_mul_q1, ae_mul_q0, ae_mul_d1, ae_mul_d0);
}

//--- ae_mulaf32r_hh.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaf32r_hh(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulaf32r_hh
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulaf32r.hh(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaf32r_hh(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulaf32r_lh.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaf32r_lh(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulaf32r_lh
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulaf32r.lh(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaf32r_lh(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulaf32r_ll.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaf32r_ll(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulaf32r_ll
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulaf32r.ll(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaf32r_ll(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulaf32r_ll_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaf32r_ll_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulaf32r_ll_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulaf32r.ll.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaf32r_ll_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulaf32s_hh.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaf32s_hh(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulaf32s_hh
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulaf32s.hh(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaf32s_hh(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulaf32s_lh.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaf32s_lh(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulaf32s_lh
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulaf32s.lh(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaf32s_lh(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulaf32s_ll.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaf32s_ll(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulaf32s_ll
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulaf32s.ll(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaf32s_ll(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulaf32s_ll_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaf32s_ll_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulaf32s_ll_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulaf32s.ll.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaf32s_ll_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulaf32x16_h0.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaf32x16_h0(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulaf32x16_h0
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulaf32x16.h0(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaf32x16_h0(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulaf32x16_h0_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaf32x16_h0_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulaf32x16_h0_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulaf32x16.h0.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaf32x16_h0_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulaf32x16_h1.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaf32x16_h1(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulaf32x16_h1
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulaf32x16.h1(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaf32x16_h1(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulaf32x16_h1_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaf32x16_h1_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulaf32x16_h1_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulaf32x16.h1.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaf32x16_h1_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulaf32x16_h2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaf32x16_h2(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulaf32x16_h2
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulaf32x16.h2(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaf32x16_h2(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulaf32x16_h2_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaf32x16_h2_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulaf32x16_h2_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulaf32x16.h2.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaf32x16_h2_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulaf32x16_h3.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaf32x16_h3(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulaf32x16_h3
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulaf32x16.h3(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaf32x16_h3(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulaf32x16_h3_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaf32x16_h3_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulaf32x16_h3_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulaf32x16.h3.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaf32x16_h3_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulaf32x16_l0.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaf32x16_l0(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulaf32x16_l0
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulaf32x16.l0(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaf32x16_l0(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulaf32x16_l0_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaf32x16_l0_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulaf32x16_l0_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulaf32x16.l0.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaf32x16_l0_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulaf32x16_l1.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaf32x16_l1(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulaf32x16_l1
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulaf32x16.l1(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaf32x16_l1(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulaf32x16_l1_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaf32x16_l1_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulaf32x16_l1_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulaf32x16.l1.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaf32x16_l1_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulaf32x16_l2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaf32x16_l2(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulaf32x16_l2
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulaf32x16.l2(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaf32x16_l2(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulaf32x16_l2_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaf32x16_l2_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulaf32x16_l2_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulaf32x16.l2.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaf32x16_l2_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulaf32x16_l3.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaf32x16_l3(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulaf32x16_l3
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulaf32x16.l3(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaf32x16_l3(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulaf32x16_l3_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaf32x16_l3_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulaf32x16_l3_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulaf32x16.l3.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaf32x16_l3_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulaf48q32sp16s_l.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaf48q32sp16s_l(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int64 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulaf48q32sp16s_l
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulaf48q32sp16s.l(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <1 x i64> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaf48q32sp16s_l(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulaf48q32sp16s_l_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaf48q32sp16s_l_s2(ae_int64* ae_mul_S2_q0,ae_int64 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulaf48q32sp16s_l_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulaf48q32sp16s.l.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <1 x i64> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaf48q32sp16s_l_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulaf48q32sp16u_l.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaf48q32sp16u_l(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int64 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulaf48q32sp16u_l
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulaf48q32sp16u.l(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <1 x i64> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaf48q32sp16u_l(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulaf48q32sp16u_l_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaf48q32sp16u_l_s2(ae_int64* ae_mul_S2_q0,ae_int64 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulaf48q32sp16u_l_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulaf48q32sp16u.l.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <1 x i64> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaf48q32sp16u_l_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulafc24ra.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulafc24ra(ae_int32x2* opnd_ae_sem_mul_x4_q0,ae_int32x2 opnd_ae_sem_mul_x4_d0,ae_int32x2 opnd_ae_sem_mul_x4_d1) {
// CHECK-LABEL: test_ae_mulafc24ra
// CHECK: %[[LD_OPND_AE_SEM_MUL_X4_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulafc24ra(<2 x i32> %[[LD_OPND_AE_SEM_MUL_X4_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulafc24ra(opnd_ae_sem_mul_x4_q0, opnd_ae_sem_mul_x4_d0, opnd_ae_sem_mul_x4_d1);
}

//--- ae_mulafc32x16ras_h.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulafc32x16ras_h(ae_int32x2* opnd_ae_sem_mul_x4_q0,ae_int32x2 opnd_ae_sem_mul_x4_d0,ae_int16x4 opnd_ae_sem_mul_x4_d1) {
// CHECK-LABEL: test_ae_mulafc32x16ras_h
// CHECK: %[[LD_OPND_AE_SEM_MUL_X4_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulafc32x16ras.h(<2 x i32> %[[LD_OPND_AE_SEM_MUL_X4_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulafc32x16ras_h(opnd_ae_sem_mul_x4_q0, opnd_ae_sem_mul_x4_d0, opnd_ae_sem_mul_x4_d1);
}

//--- ae_mulafc32x16ras_l.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulafc32x16ras_l(ae_int32x2* opnd_ae_sem_mul_x4_q0,ae_int32x2 opnd_ae_sem_mul_x4_d0,ae_int16x4 opnd_ae_sem_mul_x4_d1) {
// CHECK-LABEL: test_ae_mulafc32x16ras_l
// CHECK: %[[LD_OPND_AE_SEM_MUL_X4_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulafc32x16ras.l(<2 x i32> %[[LD_OPND_AE_SEM_MUL_X4_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulafc32x16ras_l(opnd_ae_sem_mul_x4_q0, opnd_ae_sem_mul_x4_d0, opnd_ae_sem_mul_x4_d1);
}

//--- ae_mulafd24x2_fir_h.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulafd24x2_fir_h(ae_int64* ae_mul_q0,ae_int64* ae_mul_q1,ae_int32x2 ae_mul_d0,ae_int32x2 ae_mul_d1,ae_int32x2 ae_mul_d2) {
// CHECK-LABEL: test_ae_mulafd24x2_fir_h
// CHECK: %[[LD_AE_MUL_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[LD_AE_MUL_Q1:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <1 x i64>, <1 x i64> } @llvm.xtensa.ae.mulafd24x2.fir.h(<1 x i64> %[[LD_AE_MUL_Q0]], <1 x i64> %[[LD_AE_MUL_Q1]], <2 x i32> {{.*}}, <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: %[[EV:.*]] = extractvalue { <1 x i64>, <1 x i64> } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_mulafd24x2_fir_h(ae_mul_q0, ae_mul_q1, ae_mul_d0, ae_mul_d1, ae_mul_d2);
}

//--- ae_mulafd24x2_fir_l.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulafd24x2_fir_l(ae_int64* ae_mul_q0,ae_int64* ae_mul_q1,ae_int32x2 ae_mul_d0,ae_int32x2 ae_mul_d1,ae_int32x2 ae_mul_d2) {
// CHECK-LABEL: test_ae_mulafd24x2_fir_l
// CHECK: %[[LD_AE_MUL_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[LD_AE_MUL_Q1:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <1 x i64>, <1 x i64> } @llvm.xtensa.ae.mulafd24x2.fir.l(<1 x i64> %[[LD_AE_MUL_Q0]], <1 x i64> %[[LD_AE_MUL_Q1]], <2 x i32> {{.*}}, <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: %[[EV:.*]] = extractvalue { <1 x i64>, <1 x i64> } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_mulafd24x2_fir_l(ae_mul_q0, ae_mul_q1, ae_mul_d0, ae_mul_d1, ae_mul_d2);
}

//--- ae_mulafd32x16x2_fir_hh.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulafd32x16x2_fir_hh(ae_int64* ae_mul_q0,ae_int64* ae_mul_q1,ae_int32x2 ae_mul_d0,ae_int32x2 ae_mul_d1,ae_int16x4 ae_mul_d2) {
// CHECK-LABEL: test_ae_mulafd32x16x2_fir_hh
// CHECK: %[[LD_AE_MUL_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[LD_AE_MUL_Q1:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <1 x i64>, <1 x i64> } @llvm.xtensa.ae.mulafd32x16x2.fir.hh(<1 x i64> %[[LD_AE_MUL_Q0]], <1 x i64> %[[LD_AE_MUL_Q1]], <2 x i32> {{.*}}, <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: %[[EV:.*]] = extractvalue { <1 x i64>, <1 x i64> } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_mulafd32x16x2_fir_hh(ae_mul_q0, ae_mul_q1, ae_mul_d0, ae_mul_d1, ae_mul_d2);
}

//--- ae_mulafd32x16x2_fir_hl.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulafd32x16x2_fir_hl(ae_int64* ae_mul_q0,ae_int64* ae_mul_q1,ae_int32x2 ae_mul_d0,ae_int32x2 ae_mul_d1,ae_int16x4 ae_mul_d2) {
// CHECK-LABEL: test_ae_mulafd32x16x2_fir_hl
// CHECK: %[[LD_AE_MUL_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[LD_AE_MUL_Q1:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <1 x i64>, <1 x i64> } @llvm.xtensa.ae.mulafd32x16x2.fir.hl(<1 x i64> %[[LD_AE_MUL_Q0]], <1 x i64> %[[LD_AE_MUL_Q1]], <2 x i32> {{.*}}, <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: %[[EV:.*]] = extractvalue { <1 x i64>, <1 x i64> } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_mulafd32x16x2_fir_hl(ae_mul_q0, ae_mul_q1, ae_mul_d0, ae_mul_d1, ae_mul_d2);
}

//--- ae_mulafd32x16x2_fir_lh.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulafd32x16x2_fir_lh(ae_int64* ae_mul_q0,ae_int64* ae_mul_q1,ae_int32x2 ae_mul_d0,ae_int32x2 ae_mul_d1,ae_int16x4 ae_mul_d2) {
// CHECK-LABEL: test_ae_mulafd32x16x2_fir_lh
// CHECK: %[[LD_AE_MUL_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[LD_AE_MUL_Q1:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <1 x i64>, <1 x i64> } @llvm.xtensa.ae.mulafd32x16x2.fir.lh(<1 x i64> %[[LD_AE_MUL_Q0]], <1 x i64> %[[LD_AE_MUL_Q1]], <2 x i32> {{.*}}, <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: %[[EV:.*]] = extractvalue { <1 x i64>, <1 x i64> } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_mulafd32x16x2_fir_lh(ae_mul_q0, ae_mul_q1, ae_mul_d0, ae_mul_d1, ae_mul_d2);
}

//--- ae_mulafd32x16x2_fir_ll.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulafd32x16x2_fir_ll(ae_int64* ae_mul_q0,ae_int64* ae_mul_q1,ae_int32x2 ae_mul_d0,ae_int32x2 ae_mul_d1,ae_int16x4 ae_mul_d2) {
// CHECK-LABEL: test_ae_mulafd32x16x2_fir_ll
// CHECK: %[[LD_AE_MUL_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[LD_AE_MUL_Q1:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <1 x i64>, <1 x i64> } @llvm.xtensa.ae.mulafd32x16x2.fir.ll(<1 x i64> %[[LD_AE_MUL_Q0]], <1 x i64> %[[LD_AE_MUL_Q1]], <2 x i32> {{.*}}, <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: %[[EV:.*]] = extractvalue { <1 x i64>, <1 x i64> } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_mulafd32x16x2_fir_ll(ae_mul_q0, ae_mul_q1, ae_mul_d0, ae_mul_d1, ae_mul_d2);
}

//--- ae_mulafp24x2r.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulafp24x2r(ae_int32x2* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulafp24x2r
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulafp24x2r(<2 x i32> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulafp24x2r(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulafp24x2r_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulafp24x2r_s2(ae_int32x2* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulafp24x2r_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulafp24x2r.s2(<2 x i32> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulafp24x2r_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulafp24x2ra.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulafp24x2ra(ae_int32x2* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulafp24x2ra
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulafp24x2ra(<2 x i32> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulafp24x2ra(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulafp24x2ra_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulafp24x2ra_s2(ae_int32x2* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulafp24x2ra_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulafp24x2ra.s2(<2 x i32> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulafp24x2ra_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulafp32x16x2ras_h.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulafp32x16x2ras_h(ae_int32x2* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulafp32x16x2ras_h
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulafp32x16x2ras.h(<2 x i32> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulafp32x16x2ras_h(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulafp32x16x2ras_h_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulafp32x16x2ras_h_s2(ae_int32x2* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulafp32x16x2ras_h_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulafp32x16x2ras.h.s2(<2 x i32> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulafp32x16x2ras_h_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulafp32x16x2ras_l.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulafp32x16x2ras_l(ae_int32x2* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulafp32x16x2ras_l
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulafp32x16x2ras.l(<2 x i32> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulafp32x16x2ras_l(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulafp32x16x2ras_l_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulafp32x16x2ras_l_s2(ae_int32x2* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulafp32x16x2ras_l_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulafp32x16x2ras.l.s2(<2 x i32> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulafp32x16x2ras_l_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulafp32x16x2rs_h.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulafp32x16x2rs_h(ae_int32x2* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulafp32x16x2rs_h
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulafp32x16x2rs.h(<2 x i32> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulafp32x16x2rs_h(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulafp32x16x2rs_h_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulafp32x16x2rs_h_s2(ae_int32x2* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulafp32x16x2rs_h_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulafp32x16x2rs.h.s2(<2 x i32> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulafp32x16x2rs_h_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulafp32x16x2rs_l.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulafp32x16x2rs_l(ae_int32x2* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulafp32x16x2rs_l
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulafp32x16x2rs.l(<2 x i32> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulafp32x16x2rs_l(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulafp32x16x2rs_l_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulafp32x16x2rs_l_s2(ae_int32x2* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulafp32x16x2rs_l_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulafp32x16x2rs.l.s2(<2 x i32> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulafp32x16x2rs_l_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulafp32x2ras.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulafp32x2ras(ae_int32x2* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulafp32x2ras
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulafp32x2ras(<2 x i32> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulafp32x2ras(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulafp32x2rs.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulafp32x2rs(ae_int32x2* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulafp32x2rs
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulafp32x2rs(<2 x i32> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulafp32x2rs(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulafq32sp24s_h_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulafq32sp24s_h_s2(ae_int64* ae_mul_S2_q0,ae_int64 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulafq32sp24s_h_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulafq32sp24s.h.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <1 x i64> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulafq32sp24s_h_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulafq32sp24s_l_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulafq32sp24s_l_s2(ae_int64* ae_mul_S2_q0,ae_int64 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulafq32sp24s_l_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulafq32sp24s.l.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <1 x i64> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulafq32sp24s_l_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulap24x2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulap24x2(ae_int32x2* ae_mul_q0,ae_int32x2 ae_mul_d0,ae_int32x2 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulap24x2
// CHECK: %[[LD_AE_MUL_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulap24x2(<2 x i32> %[[LD_AE_MUL_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulap24x2(ae_mul_q0, ae_mul_d0, ae_mul_d1);
}

//--- ae_mulap24x2_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulap24x2_s2(ae_int32x2* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulap24x2_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulap24x2.s2(<2 x i32> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulap24x2_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulap32x16x2_h.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulap32x16x2_h(ae_int32x2* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulap32x16x2_h
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulap32x16x2.h(<2 x i32> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulap32x16x2_h(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulap32x16x2_l.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulap32x16x2_l(ae_int32x2* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulap32x16x2_l
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulap32x16x2.l(<2 x i32> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulap32x16x2_l(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulap32x2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulap32x2(ae_int32x2* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulap32x2
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulap32x2(<2 x i32> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulap32x2(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulaq32sp16s_l_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaq32sp16s_l_s2(ae_int64* ae_mul_S2_q0,ae_int64 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulaq32sp16s_l_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulaq32sp16s.l.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <1 x i64> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaq32sp16s_l_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulaq32sp16u_l_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulaq32sp16u_l_s2(ae_int64* ae_mul_S2_q0,ae_int64 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulaq32sp16u_l_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulaq32sp16u.l.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <1 x i64> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulaq32sp16u_l_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mularfq32sp24s_h_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mularfq32sp24s_h_s2(ae_int64* ae_mul_S2_q0,ae_int64 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mularfq32sp24s_h_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mularfq32sp24s.h.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <1 x i64> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mularfq32sp24s_h_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mularfq32sp24s_l_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mularfq32sp24s_l_s2(ae_int64* ae_mul_S2_q0,ae_int64 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mularfq32sp24s_l_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mularfq32sp24s.l.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <1 x i64> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mularfq32sp24s_l_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulas32f48p16s_hh.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulas32f48p16s_hh(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulas32f48p16s_hh
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulas32f48p16s.hh(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulas32f48p16s_hh(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulas32f48p16s_hh_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulas32f48p16s_hh_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulas32f48p16s_hh_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulas32f48p16s.hh.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulas32f48p16s_hh_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulas32f48p16s_lh.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulas32f48p16s_lh(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulas32f48p16s_lh
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulas32f48p16s.lh(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulas32f48p16s_lh(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulas32f48p16s_lh_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulas32f48p16s_lh_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulas32f48p16s_lh_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulas32f48p16s.lh.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulas32f48p16s_lh_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulas32f48p16s_ll.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulas32f48p16s_ll(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulas32f48p16s_ll
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulas32f48p16s.ll(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulas32f48p16s_ll(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulas32f48p16s_ll_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulas32f48p16s_ll_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulas32f48p16s_ll_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulas32f48p16s.ll.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulas32f48p16s_ll_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulasd24_hh_ll.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulasd24_hh_ll(ae_int64* ae_mul_q0,ae_int32x2 ae_mul_d0,ae_int32x2 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulasd24_hh_ll
// CHECK: %[[LD_AE_MUL_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulasd24.hh.ll(<1 x i64> %[[LD_AE_MUL_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulasd24_hh_ll(ae_mul_q0, ae_mul_d0, ae_mul_d1);
}

//--- ae_mulasd24_hh_ll_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulasd24_hh_ll_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulasd24_hh_ll_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulasd24.hh.ll.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulasd24_hh_ll_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulasd24_hl_lh.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulasd24_hl_lh(ae_int64* ae_mul_q0,ae_int32x2 ae_mul_d0,ae_int32x2 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulasd24_hl_lh
// CHECK: %[[LD_AE_MUL_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulasd24.hl.lh(<1 x i64> %[[LD_AE_MUL_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulasd24_hl_lh(ae_mul_q0, ae_mul_d0, ae_mul_d1);
}

//--- ae_mulasd24_hl_lh_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulasd24_hl_lh_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulasd24_hl_lh_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulasd24.hl.lh.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulasd24_hl_lh_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulasd32x16_h1_l0.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulasd32x16_h1_l0(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulasd32x16_h1_l0
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulasd32x16.h1.l0(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulasd32x16_h1_l0(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulasd32x16_h1_l0_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulasd32x16_h1_l0_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulasd32x16_h1_l0_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulasd32x16.h1.l0.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulasd32x16_h1_l0_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulasd32x16_h3_l2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulasd32x16_h3_l2(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulasd32x16_h3_l2
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulasd32x16.h3.l2(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulasd32x16_h3_l2(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulasd32x16_h3_l2_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulasd32x16_h3_l2_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulasd32x16_h3_l2_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulasd32x16.h3.l2.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulasd32x16_h3_l2_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulasfd24_hh_ll.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulasfd24_hh_ll(ae_int64* ae_mul_q0,ae_int32x2 ae_mul_d0,ae_int32x2 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulasfd24_hh_ll
// CHECK: %[[LD_AE_MUL_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulasfd24.hh.ll(<1 x i64> %[[LD_AE_MUL_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulasfd24_hh_ll(ae_mul_q0, ae_mul_d0, ae_mul_d1);
}

//--- ae_mulasfd24_hh_ll_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulasfd24_hh_ll_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulasfd24_hh_ll_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulasfd24.hh.ll.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulasfd24_hh_ll_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulasfd24_hl_lh.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulasfd24_hl_lh(ae_int64* ae_mul_q0,ae_int32x2 ae_mul_d0,ae_int32x2 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulasfd24_hl_lh
// CHECK: %[[LD_AE_MUL_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulasfd24.hl.lh(<1 x i64> %[[LD_AE_MUL_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulasfd24_hl_lh(ae_mul_q0, ae_mul_d0, ae_mul_d1);
}

//--- ae_mulasfd24_hl_lh_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulasfd24_hl_lh_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulasfd24_hl_lh_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulasfd24.hl.lh.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulasfd24_hl_lh_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulasfd32x16_h1_l0.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulasfd32x16_h1_l0(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulasfd32x16_h1_l0
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulasfd32x16.h1.l0(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulasfd32x16_h1_l0(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulasfd32x16_h1_l0_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulasfd32x16_h1_l0_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulasfd32x16_h1_l0_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulasfd32x16.h1.l0.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulasfd32x16_h1_l0_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulasfd32x16_h3_l2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulasfd32x16_h3_l2(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulasfd32x16_h3_l2
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulasfd32x16.h3.l2(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulasfd32x16_h3_l2(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulasfd32x16_h3_l2_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulasfd32x16_h3_l2_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulasfd32x16_h3_l2_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulasfd32x16.h3.l2.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulasfd32x16_h3_l2_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulc24.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_mulc24(ae_int32x2 ae_mul_d0,ae_int32x2 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulc24
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulc24(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_mulc24(ae_mul_d0, ae_mul_d1);
}

//--- ae_mulc32x16_h.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_mulc32x16_h(ae_int32x2 opnd_ae_sem_mul_x4_d0,ae_int16x4 opnd_ae_sem_mul_x4_d1) {
// CHECK-LABEL: test_ae_mulc32x16_h
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulc32x16.h(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_mulc32x16_h(opnd_ae_sem_mul_x4_d0, opnd_ae_sem_mul_x4_d1);
}

//--- ae_mulc32x16_l.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_mulc32x16_l(ae_int32x2 opnd_ae_sem_mul_x4_d0,ae_int16x4 opnd_ae_sem_mul_x4_d1) {
// CHECK-LABEL: test_ae_mulc32x16_l
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulc32x16.l(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_mulc32x16_l(opnd_ae_sem_mul_x4_d0, opnd_ae_sem_mul_x4_d1);
}

//--- ae_mulf16ss_00.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_mulf16ss_00(ae_int16x4 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulf16ss_00
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulf16ss.00(<4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_mulf16ss_00(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulf16ss_00_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_mulf16ss_00_s2(ae_int16x4 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulf16ss_00_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulf16ss.00.s2(<4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_mulf16ss_00_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulf16ss_10.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_mulf16ss_10(ae_int16x4 ae_mul_d0,ae_int16x4 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulf16ss_10
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulf16ss.10(<4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_mulf16ss_10(ae_mul_d0, ae_mul_d1);
}

//--- ae_mulf16ss_11.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_mulf16ss_11(ae_int16x4 ae_mul_d0,ae_int16x4 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulf16ss_11
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulf16ss.11(<4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_mulf16ss_11(ae_mul_d0, ae_mul_d1);
}

//--- ae_mulf16ss_20.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_mulf16ss_20(ae_int16x4 ae_mul_d0,ae_int16x4 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulf16ss_20
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulf16ss.20(<4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_mulf16ss_20(ae_mul_d0, ae_mul_d1);
}

//--- ae_mulf16ss_21.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_mulf16ss_21(ae_int16x4 ae_mul_d0,ae_int16x4 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulf16ss_21
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulf16ss.21(<4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_mulf16ss_21(ae_mul_d0, ae_mul_d1);
}

//--- ae_mulf16ss_22.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_mulf16ss_22(ae_int16x4 ae_mul_d0,ae_int16x4 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulf16ss_22
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulf16ss.22(<4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_mulf16ss_22(ae_mul_d0, ae_mul_d1);
}

//--- ae_mulf16ss_30.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_mulf16ss_30(ae_int16x4 ae_mul_d0,ae_int16x4 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulf16ss_30
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulf16ss.30(<4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_mulf16ss_30(ae_mul_d0, ae_mul_d1);
}

//--- ae_mulf16ss_31.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_mulf16ss_31(ae_int16x4 ae_mul_d0,ae_int16x4 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulf16ss_31
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulf16ss.31(<4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_mulf16ss_31(ae_mul_d0, ae_mul_d1);
}

//--- ae_mulf16ss_32.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_mulf16ss_32(ae_int16x4 ae_mul_d0,ae_int16x4 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulf16ss_32
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulf16ss.32(<4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_mulf16ss_32(ae_mul_d0, ae_mul_d1);
}

//--- ae_mulf16ss_33.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_mulf16ss_33(ae_int16x4 ae_mul_d0,ae_int16x4 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulf16ss_33
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulf16ss.33(<4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_mulf16ss_33(ae_mul_d0, ae_mul_d1);
}

//--- ae_mulf16x4ss.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulf16x4ss(ae_int32x2* ae_mul_q1,ae_int32x2* ae_mul_q0,ae_int16x4 ae_mul_d1,ae_int16x4 ae_mul_d0) {
// CHECK-LABEL: test_ae_mulf16x4ss
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <2 x i32>, <2 x i32> } @llvm.xtensa.ae.mulf16x4ss(<4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: %[[EV:.*]] = extractvalue { <2 x i32>, <2 x i32> } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_mulf16x4ss(ae_mul_q1, ae_mul_q0, ae_mul_d1, ae_mul_d0);
}

//--- ae_mulf32r_hh.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulf32r_hh(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulf32r_hh
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulf32r.hh(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulf32r_hh(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulf32r_lh.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulf32r_lh(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulf32r_lh
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulf32r.lh(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulf32r_lh(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulf32r_ll.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulf32r_ll(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulf32r_ll
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulf32r.ll(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulf32r_ll(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulf32r_ll_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulf32r_ll_s2(ae_int32x2 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulf32r_ll_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulf32r.ll.s2(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulf32r_ll_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulf32s_hh.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulf32s_hh(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulf32s_hh
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulf32s.hh(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulf32s_hh(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulf32s_lh.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulf32s_lh(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulf32s_lh
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulf32s.lh(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulf32s_lh(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulf32s_ll.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulf32s_ll(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulf32s_ll
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulf32s.ll(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulf32s_ll(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulf32s_ll_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulf32s_ll_s2(ae_int32x2 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulf32s_ll_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulf32s.ll.s2(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulf32s_ll_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulf32x16_h0.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulf32x16_h0(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulf32x16_h0
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulf32x16.h0(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulf32x16_h0(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulf32x16_h0_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulf32x16_h0_s2(ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulf32x16_h0_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulf32x16.h0.s2(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulf32x16_h0_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulf32x16_h1.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulf32x16_h1(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulf32x16_h1
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulf32x16.h1(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulf32x16_h1(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulf32x16_h1_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulf32x16_h1_s2(ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulf32x16_h1_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulf32x16.h1.s2(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulf32x16_h1_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulf32x16_h2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulf32x16_h2(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulf32x16_h2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulf32x16.h2(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulf32x16_h2(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulf32x16_h2_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulf32x16_h2_s2(ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulf32x16_h2_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulf32x16.h2.s2(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulf32x16_h2_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulf32x16_h3.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulf32x16_h3(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulf32x16_h3
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulf32x16.h3(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulf32x16_h3(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulf32x16_h3_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulf32x16_h3_s2(ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulf32x16_h3_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulf32x16.h3.s2(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulf32x16_h3_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulf32x16_l0.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulf32x16_l0(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulf32x16_l0
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulf32x16.l0(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulf32x16_l0(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulf32x16_l0_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulf32x16_l0_s2(ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulf32x16_l0_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulf32x16.l0.s2(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulf32x16_l0_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulf32x16_l1.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulf32x16_l1(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulf32x16_l1
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulf32x16.l1(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulf32x16_l1(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulf32x16_l1_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulf32x16_l1_s2(ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulf32x16_l1_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulf32x16.l1.s2(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulf32x16_l1_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulf32x16_l2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulf32x16_l2(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulf32x16_l2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulf32x16.l2(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulf32x16_l2(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulf32x16_l2_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulf32x16_l2_s2(ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulf32x16_l2_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulf32x16.l2.s2(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulf32x16_l2_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulf32x16_l3.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulf32x16_l3(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulf32x16_l3
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulf32x16.l3(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulf32x16_l3(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulf32x16_l3_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulf32x16_l3_s2(ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulf32x16_l3_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulf32x16.l3.s2(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulf32x16_l3_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulf48q32sp16s_l.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulf48q32sp16s_l(ae_int64 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulf48q32sp16s_l
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulf48q32sp16s.l(<1 x i64> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulf48q32sp16s_l(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulf48q32sp16s_l_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulf48q32sp16s_l_s2(ae_int64 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulf48q32sp16s_l_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulf48q32sp16s.l.s2(<1 x i64> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulf48q32sp16s_l_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulf48q32sp16u_l.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulf48q32sp16u_l(ae_int64 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulf48q32sp16u_l
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulf48q32sp16u.l(<1 x i64> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulf48q32sp16u_l(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulf48q32sp16u_l_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulf48q32sp16u_l_s2(ae_int64 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulf48q32sp16u_l_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulf48q32sp16u.l.s2(<1 x i64> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulf48q32sp16u_l_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulfc24ra.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_mulfc24ra(ae_int32x2 opnd_ae_sem_mul_x4_d0,ae_int32x2 opnd_ae_sem_mul_x4_d1) {
// CHECK-LABEL: test_ae_mulfc24ra
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulfc24ra(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_mulfc24ra(opnd_ae_sem_mul_x4_d0, opnd_ae_sem_mul_x4_d1);
}

//--- ae_mulfc32x16ras_h.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_mulfc32x16ras_h(ae_int32x2 opnd_ae_sem_mul_x4_d0,ae_int16x4 opnd_ae_sem_mul_x4_d1) {
// CHECK-LABEL: test_ae_mulfc32x16ras_h
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulfc32x16ras.h(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_mulfc32x16ras_h(opnd_ae_sem_mul_x4_d0, opnd_ae_sem_mul_x4_d1);
}

//--- ae_mulfc32x16ras_l.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_mulfc32x16ras_l(ae_int32x2 opnd_ae_sem_mul_x4_d0,ae_int16x4 opnd_ae_sem_mul_x4_d1) {
// CHECK-LABEL: test_ae_mulfc32x16ras_l
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulfc32x16ras.l(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_mulfc32x16ras_l(opnd_ae_sem_mul_x4_d0, opnd_ae_sem_mul_x4_d1);
}

//--- ae_mulfd24x2_fir_h.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulfd24x2_fir_h(ae_int64* ae_mul_q0,ae_int64* ae_mul_q1,ae_int32x2 ae_mul_d0,ae_int32x2 ae_mul_d1,ae_int32x2 ae_mul_d2) {
// CHECK-LABEL: test_ae_mulfd24x2_fir_h
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <1 x i64>, <1 x i64> } @llvm.xtensa.ae.mulfd24x2.fir.h(<2 x i32> {{.*}}, <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: %[[EV:.*]] = extractvalue { <1 x i64>, <1 x i64> } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_mulfd24x2_fir_h(ae_mul_q0, ae_mul_q1, ae_mul_d0, ae_mul_d1, ae_mul_d2);
}

//--- ae_mulfd24x2_fir_l.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulfd24x2_fir_l(ae_int64* ae_mul_q0,ae_int64* ae_mul_q1,ae_int32x2 ae_mul_d0,ae_int32x2 ae_mul_d1,ae_int32x2 ae_mul_d2) {
// CHECK-LABEL: test_ae_mulfd24x2_fir_l
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <1 x i64>, <1 x i64> } @llvm.xtensa.ae.mulfd24x2.fir.l(<2 x i32> {{.*}}, <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: %[[EV:.*]] = extractvalue { <1 x i64>, <1 x i64> } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_mulfd24x2_fir_l(ae_mul_q0, ae_mul_q1, ae_mul_d0, ae_mul_d1, ae_mul_d2);
}

//--- ae_mulfd32x16x2_fir_hh.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulfd32x16x2_fir_hh(ae_int64* ae_mul_q0,ae_int64* ae_mul_q1,ae_int32x2 ae_mul_d0,ae_int32x2 ae_mul_d1,ae_int16x4 ae_mul_d2) {
// CHECK-LABEL: test_ae_mulfd32x16x2_fir_hh
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <1 x i64>, <1 x i64> } @llvm.xtensa.ae.mulfd32x16x2.fir.hh(<2 x i32> {{.*}}, <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: %[[EV:.*]] = extractvalue { <1 x i64>, <1 x i64> } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_mulfd32x16x2_fir_hh(ae_mul_q0, ae_mul_q1, ae_mul_d0, ae_mul_d1, ae_mul_d2);
}

//--- ae_mulfd32x16x2_fir_hl.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulfd32x16x2_fir_hl(ae_int64* ae_mul_q0,ae_int64* ae_mul_q1,ae_int32x2 ae_mul_d0,ae_int32x2 ae_mul_d1,ae_int16x4 ae_mul_d2) {
// CHECK-LABEL: test_ae_mulfd32x16x2_fir_hl
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <1 x i64>, <1 x i64> } @llvm.xtensa.ae.mulfd32x16x2.fir.hl(<2 x i32> {{.*}}, <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: %[[EV:.*]] = extractvalue { <1 x i64>, <1 x i64> } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_mulfd32x16x2_fir_hl(ae_mul_q0, ae_mul_q1, ae_mul_d0, ae_mul_d1, ae_mul_d2);
}

//--- ae_mulfd32x16x2_fir_lh.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulfd32x16x2_fir_lh(ae_int64* ae_mul_q0,ae_int64* ae_mul_q1,ae_int32x2 ae_mul_d0,ae_int32x2 ae_mul_d1,ae_int16x4 ae_mul_d2) {
// CHECK-LABEL: test_ae_mulfd32x16x2_fir_lh
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <1 x i64>, <1 x i64> } @llvm.xtensa.ae.mulfd32x16x2.fir.lh(<2 x i32> {{.*}}, <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: %[[EV:.*]] = extractvalue { <1 x i64>, <1 x i64> } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_mulfd32x16x2_fir_lh(ae_mul_q0, ae_mul_q1, ae_mul_d0, ae_mul_d1, ae_mul_d2);
}

//--- ae_mulfd32x16x2_fir_ll.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulfd32x16x2_fir_ll(ae_int64* ae_mul_q0,ae_int64* ae_mul_q1,ae_int32x2 ae_mul_d0,ae_int32x2 ae_mul_d1,ae_int16x4 ae_mul_d2) {
// CHECK-LABEL: test_ae_mulfd32x16x2_fir_ll
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <1 x i64>, <1 x i64> } @llvm.xtensa.ae.mulfd32x16x2.fir.ll(<2 x i32> {{.*}}, <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: %[[EV:.*]] = extractvalue { <1 x i64>, <1 x i64> } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_mulfd32x16x2_fir_ll(ae_mul_q0, ae_mul_q1, ae_mul_d0, ae_mul_d1, ae_mul_d2);
}

//--- ae_mulfp16x4ras.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int16x4 test_ae_mulfp16x4ras(ae_int16x4 ae_mul_d0,ae_int16x4 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulfp16x4ras
// CHECK: %[[RET:.*]] = {{(tail)?}} call <4 x i16> @llvm.xtensa.ae.mulfp16x4ras(<4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <4 x i16> %[[RET]]
return __builtin_xtensa_ae_mulfp16x4ras(ae_mul_d0, ae_mul_d1);
}

//--- ae_mulfp16x4s.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int16x4 test_ae_mulfp16x4s(ae_int16x4 ae_mul_d0,ae_int16x4 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulfp16x4s
// CHECK: %[[RET:.*]] = {{(tail)?}} call <4 x i16> @llvm.xtensa.ae.mulfp16x4s(<4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <4 x i16> %[[RET]]
return __builtin_xtensa_ae_mulfp16x4s(ae_mul_d0, ae_mul_d1);
}

//--- ae_mulfp24x2r.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_mulfp24x2r(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulfp24x2r
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulfp24x2r(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_mulfp24x2r(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulfp24x2r_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_mulfp24x2r_s2(ae_int32x2 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulfp24x2r_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulfp24x2r.s2(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_mulfp24x2r_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulfp24x2ra.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_mulfp24x2ra(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulfp24x2ra
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulfp24x2ra(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_mulfp24x2ra(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulfp24x2ra_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_mulfp24x2ra_s2(ae_int32x2 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulfp24x2ra_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulfp24x2ra.s2(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_mulfp24x2ra_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulfp32x16x2ras_h.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_mulfp32x16x2ras_h(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulfp32x16x2ras_h
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulfp32x16x2ras.h(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_mulfp32x16x2ras_h(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulfp32x16x2ras_h_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_mulfp32x16x2ras_h_s2(ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulfp32x16x2ras_h_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulfp32x16x2ras.h.s2(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_mulfp32x16x2ras_h_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulfp32x16x2ras_l.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_mulfp32x16x2ras_l(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulfp32x16x2ras_l
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulfp32x16x2ras.l(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_mulfp32x16x2ras_l(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulfp32x16x2ras_l_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_mulfp32x16x2ras_l_s2(ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulfp32x16x2ras_l_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulfp32x16x2ras.l.s2(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_mulfp32x16x2ras_l_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulfp32x16x2rs_h.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_mulfp32x16x2rs_h(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulfp32x16x2rs_h
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulfp32x16x2rs.h(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_mulfp32x16x2rs_h(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulfp32x16x2rs_h_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_mulfp32x16x2rs_h_s2(ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulfp32x16x2rs_h_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulfp32x16x2rs.h.s2(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_mulfp32x16x2rs_h_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulfp32x16x2rs_l.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_mulfp32x16x2rs_l(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulfp32x16x2rs_l
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulfp32x16x2rs.l(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_mulfp32x16x2rs_l(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulfp32x16x2rs_l_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_mulfp32x16x2rs_l_s2(ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulfp32x16x2rs_l_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulfp32x16x2rs.l.s2(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_mulfp32x16x2rs_l_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulfp32x2ras.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_mulfp32x2ras(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulfp32x2ras
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulfp32x2ras(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_mulfp32x2ras(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulfp32x2rs.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_mulfp32x2rs(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulfp32x2rs
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulfp32x2rs(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_mulfp32x2rs(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulfq32sp24s_h_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulfq32sp24s_h_s2(ae_int64 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulfq32sp24s_h_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulfq32sp24s.h.s2(<1 x i64> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulfq32sp24s_h_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulfq32sp24s_l_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulfq32sp24s_l_s2(ae_int64 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulfq32sp24s_l_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulfq32sp24s.l.s2(<1 x i64> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulfq32sp24s_l_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulp24x2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_mulp24x2(ae_int32x2 ae_mul_d0,ae_int32x2 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulp24x2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulp24x2(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_mulp24x2(ae_mul_d0, ae_mul_d1);
}

//--- ae_mulp24x2_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_mulp24x2_s2(ae_int32x2 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulp24x2_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulp24x2.s2(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_mulp24x2_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulp32x16x2_h.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_mulp32x16x2_h(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulp32x16x2_h
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulp32x16x2.h(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_mulp32x16x2_h(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulp32x16x2_l.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_mulp32x16x2_l(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulp32x16x2_l
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulp32x16x2.l(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_mulp32x16x2_l(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulp32x2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_mulp32x2(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulp32x2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulp32x2(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_mulp32x2(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulq32sp16s_l_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulq32sp16s_l_s2(ae_int64 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulq32sp16s_l_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulq32sp16s.l.s2(<1 x i64> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulq32sp16s_l_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulq32sp16u_l_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulq32sp16u_l_s2(ae_int64 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulq32sp16u_l_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulq32sp16u.l.s2(<1 x i64> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulq32sp16u_l_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulrfq32sp24s_h_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulrfq32sp24s_h_s2(ae_int64 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulrfq32sp24s_h_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulrfq32sp24s.h.s2(<1 x i64> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulrfq32sp24s_h_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulrfq32sp24s_l_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulrfq32sp24s_l_s2(ae_int64 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulrfq32sp24s_l_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulrfq32sp24s.l.s2(<1 x i64> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulrfq32sp24s_l_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_muls16x4.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_muls16x4(ae_int32x2* ae_mul_q1,ae_int32x2* ae_mul_q0,ae_int16x4 ae_mul_d1,ae_int16x4 ae_mul_d0) {
// CHECK-LABEL: test_ae_muls16x4
// CHECK: %[[LD_AE_MUL_Q1:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[LD_AE_MUL_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <2 x i32>, <2 x i32> } @llvm.xtensa.ae.muls16x4(<2 x i32> %[[LD_AE_MUL_Q1]], <2 x i32> %[[LD_AE_MUL_Q0]], <4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: %[[EV:.*]] = extractvalue { <2 x i32>, <2 x i32> } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_muls16x4(ae_mul_q1, ae_mul_q0, ae_mul_d1, ae_mul_d0);
}

//--- ae_muls32_hh.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_muls32_hh(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_muls32_hh
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.muls32.hh(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_muls32_hh(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_muls32_lh.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_muls32_lh(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_muls32_lh
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.muls32.lh(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_muls32_lh(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_muls32_ll.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_muls32_ll(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_muls32_ll
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.muls32.ll(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_muls32_ll(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_muls32f48p16s_hh.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_muls32f48p16s_hh(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_muls32f48p16s_hh
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.muls32f48p16s.hh(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_muls32f48p16s_hh(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_muls32f48p16s_hh_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_muls32f48p16s_hh_s2(ae_int32x2 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_muls32f48p16s_hh_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.muls32f48p16s.hh.s2(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_muls32f48p16s_hh_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_muls32f48p16s_lh.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_muls32f48p16s_lh(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_muls32f48p16s_lh
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.muls32f48p16s.lh(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_muls32f48p16s_lh(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_muls32f48p16s_lh_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_muls32f48p16s_lh_s2(ae_int32x2 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_muls32f48p16s_lh_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.muls32f48p16s.lh.s2(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_muls32f48p16s_lh_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_muls32f48p16s_ll.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_muls32f48p16s_ll(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_muls32f48p16s_ll
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.muls32f48p16s.ll(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_muls32f48p16s_ll(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_muls32f48p16s_ll_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_muls32f48p16s_ll_s2(ae_int32x2 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_muls32f48p16s_ll_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.muls32f48p16s.ll.s2(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_muls32f48p16s_ll_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_muls32u_ll.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_muls32u_ll(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_muls32u_ll
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.muls32u.ll(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_muls32u_ll(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_muls32x16_h0.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_muls32x16_h0(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_muls32x16_h0
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.muls32x16.h0(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_muls32x16_h0(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_muls32x16_h0_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_muls32x16_h0_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_muls32x16_h0_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.muls32x16.h0.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_muls32x16_h0_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_muls32x16_h1.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_muls32x16_h1(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_muls32x16_h1
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.muls32x16.h1(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_muls32x16_h1(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_muls32x16_h1_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_muls32x16_h1_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_muls32x16_h1_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.muls32x16.h1.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_muls32x16_h1_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_muls32x16_h2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_muls32x16_h2(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_muls32x16_h2
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.muls32x16.h2(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_muls32x16_h2(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_muls32x16_h2_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_muls32x16_h2_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_muls32x16_h2_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.muls32x16.h2.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_muls32x16_h2_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_muls32x16_h3.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_muls32x16_h3(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_muls32x16_h3
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.muls32x16.h3(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_muls32x16_h3(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_muls32x16_h3_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_muls32x16_h3_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_muls32x16_h3_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.muls32x16.h3.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_muls32x16_h3_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_muls32x16_l0.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_muls32x16_l0(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_muls32x16_l0
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.muls32x16.l0(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_muls32x16_l0(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_muls32x16_l0_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_muls32x16_l0_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_muls32x16_l0_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.muls32x16.l0.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_muls32x16_l0_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_muls32x16_l1.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_muls32x16_l1(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_muls32x16_l1
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.muls32x16.l1(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_muls32x16_l1(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_muls32x16_l1_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_muls32x16_l1_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_muls32x16_l1_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.muls32x16.l1.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_muls32x16_l1_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_muls32x16_l2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_muls32x16_l2(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_muls32x16_l2
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.muls32x16.l2(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_muls32x16_l2(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_muls32x16_l2_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_muls32x16_l2_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_muls32x16_l2_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.muls32x16.l2.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_muls32x16_l2_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_muls32x16_l3.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_muls32x16_l3(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_muls32x16_l3
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.muls32x16.l3(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_muls32x16_l3(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_muls32x16_l3_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_muls32x16_l3_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_muls32x16_l3_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.muls32x16.l3.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_muls32x16_l3_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulsad24_hh_ll.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsad24_hh_ll(ae_int64* ae_mul_q0,ae_int32x2 ae_mul_d0,ae_int32x2 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulsad24_hh_ll
// CHECK: %[[LD_AE_MUL_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulsad24.hh.ll(<1 x i64> %[[LD_AE_MUL_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsad24_hh_ll(ae_mul_q0, ae_mul_d0, ae_mul_d1);
}

//--- ae_mulsad24_hh_ll_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsad24_hh_ll_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulsad24_hh_ll_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulsad24.hh.ll.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsad24_hh_ll_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulsad32x16_h1_l0.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsad32x16_h1_l0(ae_int64* ae_mul_q0,ae_int32x2 ae_mul_d0,ae_int16x4 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulsad32x16_h1_l0
// CHECK: %[[LD_AE_MUL_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulsad32x16.h1.l0(<1 x i64> %[[LD_AE_MUL_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsad32x16_h1_l0(ae_mul_q0, ae_mul_d0, ae_mul_d1);
}

//--- ae_mulsad32x16_h1_l0_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsad32x16_h1_l0_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulsad32x16_h1_l0_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulsad32x16.h1.l0.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsad32x16_h1_l0_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulsad32x16_h3_l2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsad32x16_h3_l2(ae_int64* ae_mul_q0,ae_int32x2 ae_mul_d0,ae_int16x4 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulsad32x16_h3_l2
// CHECK: %[[LD_AE_MUL_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulsad32x16.h3.l2(<1 x i64> %[[LD_AE_MUL_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsad32x16_h3_l2(ae_mul_q0, ae_mul_d0, ae_mul_d1);
}

//--- ae_mulsad32x16_h3_l2_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsad32x16_h3_l2_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulsad32x16_h3_l2_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulsad32x16.h3.l2.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsad32x16_h3_l2_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulsafd24_hh_ll.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsafd24_hh_ll(ae_int64* ae_mul_q0,ae_int32x2 ae_mul_d0,ae_int32x2 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulsafd24_hh_ll
// CHECK: %[[LD_AE_MUL_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulsafd24.hh.ll(<1 x i64> %[[LD_AE_MUL_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsafd24_hh_ll(ae_mul_q0, ae_mul_d0, ae_mul_d1);
}

//--- ae_mulsafd24_hh_ll_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsafd24_hh_ll_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulsafd24_hh_ll_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulsafd24.hh.ll.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsafd24_hh_ll_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulsafd32x16_h1_l0.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsafd32x16_h1_l0(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulsafd32x16_h1_l0
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulsafd32x16.h1.l0(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsafd32x16_h1_l0(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulsafd32x16_h1_l0_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsafd32x16_h1_l0_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulsafd32x16_h1_l0_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulsafd32x16.h1.l0.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsafd32x16_h1_l0_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulsafd32x16_h3_l2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsafd32x16_h3_l2(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulsafd32x16_h3_l2
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulsafd32x16.h3.l2(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsafd32x16_h3_l2(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulsafd32x16_h3_l2_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsafd32x16_h3_l2_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulsafd32x16_h3_l2_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulsafd32x16.h3.l2.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsafd32x16_h3_l2_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulsf16ss_00.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsf16ss_00(ae_int32x2* opnd_ae_sem_mul_x2_S1_q0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulsf16ss_00
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulsf16ss.00(<2 x i32> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsf16ss_00(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulsf16ss_00_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsf16ss_00_s2(ae_int32x2* ae_mul_S2_q0,ae_int16x4 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulsf16ss_00_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulsf16ss.00.s2(<2 x i32> %[[LD_AE_MUL_S2_Q0]], <4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsf16ss_00_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulsf16ss_10.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsf16ss_10(ae_int32x2* ae_mul_q0,ae_int16x4 ae_mul_d0,ae_int16x4 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulsf16ss_10
// CHECK: %[[LD_AE_MUL_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulsf16ss.10(<2 x i32> %[[LD_AE_MUL_Q0]], <4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsf16ss_10(ae_mul_q0, ae_mul_d0, ae_mul_d1);
}

//--- ae_mulsf16ss_11.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsf16ss_11(ae_int32x2* ae_mul_q0,ae_int16x4 ae_mul_d0,ae_int16x4 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulsf16ss_11
// CHECK: %[[LD_AE_MUL_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulsf16ss.11(<2 x i32> %[[LD_AE_MUL_Q0]], <4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsf16ss_11(ae_mul_q0, ae_mul_d0, ae_mul_d1);
}

//--- ae_mulsf16ss_20.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsf16ss_20(ae_int32x2* ae_mul_q0,ae_int16x4 ae_mul_d0,ae_int16x4 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulsf16ss_20
// CHECK: %[[LD_AE_MUL_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulsf16ss.20(<2 x i32> %[[LD_AE_MUL_Q0]], <4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsf16ss_20(ae_mul_q0, ae_mul_d0, ae_mul_d1);
}

//--- ae_mulsf16ss_21.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsf16ss_21(ae_int32x2* ae_mul_q0,ae_int16x4 ae_mul_d0,ae_int16x4 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulsf16ss_21
// CHECK: %[[LD_AE_MUL_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulsf16ss.21(<2 x i32> %[[LD_AE_MUL_Q0]], <4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsf16ss_21(ae_mul_q0, ae_mul_d0, ae_mul_d1);
}

//--- ae_mulsf16ss_22.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsf16ss_22(ae_int32x2* ae_mul_q0,ae_int16x4 ae_mul_d0,ae_int16x4 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulsf16ss_22
// CHECK: %[[LD_AE_MUL_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulsf16ss.22(<2 x i32> %[[LD_AE_MUL_Q0]], <4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsf16ss_22(ae_mul_q0, ae_mul_d0, ae_mul_d1);
}

//--- ae_mulsf16ss_30.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsf16ss_30(ae_int32x2* ae_mul_q0,ae_int16x4 ae_mul_d0,ae_int16x4 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulsf16ss_30
// CHECK: %[[LD_AE_MUL_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulsf16ss.30(<2 x i32> %[[LD_AE_MUL_Q0]], <4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsf16ss_30(ae_mul_q0, ae_mul_d0, ae_mul_d1);
}

//--- ae_mulsf16ss_31.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsf16ss_31(ae_int32x2* ae_mul_q0,ae_int16x4 ae_mul_d0,ae_int16x4 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulsf16ss_31
// CHECK: %[[LD_AE_MUL_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulsf16ss.31(<2 x i32> %[[LD_AE_MUL_Q0]], <4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsf16ss_31(ae_mul_q0, ae_mul_d0, ae_mul_d1);
}

//--- ae_mulsf16ss_32.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsf16ss_32(ae_int32x2* ae_mul_q0,ae_int16x4 ae_mul_d0,ae_int16x4 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulsf16ss_32
// CHECK: %[[LD_AE_MUL_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulsf16ss.32(<2 x i32> %[[LD_AE_MUL_Q0]], <4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsf16ss_32(ae_mul_q0, ae_mul_d0, ae_mul_d1);
}

//--- ae_mulsf16ss_33.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsf16ss_33(ae_int32x2* ae_mul_q0,ae_int16x4 ae_mul_d0,ae_int16x4 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulsf16ss_33
// CHECK: %[[LD_AE_MUL_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulsf16ss.33(<2 x i32> %[[LD_AE_MUL_Q0]], <4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsf16ss_33(ae_mul_q0, ae_mul_d0, ae_mul_d1);
}

//--- ae_mulsf16x4ss.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsf16x4ss(ae_int32x2* ae_mul_q1,ae_int32x2* ae_mul_q0,ae_int16x4 ae_mul_d1,ae_int16x4 ae_mul_d0) {
// CHECK-LABEL: test_ae_mulsf16x4ss
// CHECK: %[[LD_AE_MUL_Q1:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[LD_AE_MUL_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <2 x i32>, <2 x i32> } @llvm.xtensa.ae.mulsf16x4ss(<2 x i32> %[[LD_AE_MUL_Q1]], <2 x i32> %[[LD_AE_MUL_Q0]], <4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: %[[EV:.*]] = extractvalue { <2 x i32>, <2 x i32> } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_mulsf16x4ss(ae_mul_q1, ae_mul_q0, ae_mul_d1, ae_mul_d0);
}

//--- ae_mulsf32r_hh.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsf32r_hh(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulsf32r_hh
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulsf32r.hh(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsf32r_hh(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulsf32r_lh.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsf32r_lh(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulsf32r_lh
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulsf32r.lh(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsf32r_lh(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulsf32r_ll.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsf32r_ll(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulsf32r_ll
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulsf32r.ll(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsf32r_ll(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulsf32r_ll_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsf32r_ll_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulsf32r_ll_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulsf32r.ll.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsf32r_ll_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulsf32s_hh.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsf32s_hh(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulsf32s_hh
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulsf32s.hh(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsf32s_hh(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulsf32s_lh.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsf32s_lh(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulsf32s_lh
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulsf32s.lh(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsf32s_lh(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulsf32s_ll.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsf32s_ll(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulsf32s_ll
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulsf32s.ll(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsf32s_ll(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulsf32x16_h0.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsf32x16_h0(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulsf32x16_h0
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulsf32x16.h0(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsf32x16_h0(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulsf32x16_h0_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsf32x16_h0_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulsf32x16_h0_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulsf32x16.h0.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsf32x16_h0_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulsf32x16_h1.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsf32x16_h1(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulsf32x16_h1
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulsf32x16.h1(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsf32x16_h1(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulsf32x16_h1_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsf32x16_h1_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulsf32x16_h1_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulsf32x16.h1.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsf32x16_h1_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulsf32x16_h2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsf32x16_h2(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulsf32x16_h2
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulsf32x16.h2(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsf32x16_h2(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulsf32x16_h2_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsf32x16_h2_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulsf32x16_h2_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulsf32x16.h2.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsf32x16_h2_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulsf32x16_h3.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsf32x16_h3(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulsf32x16_h3
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulsf32x16.h3(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsf32x16_h3(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulsf32x16_h3_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsf32x16_h3_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulsf32x16_h3_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulsf32x16.h3.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsf32x16_h3_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulsf32x16_l0.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsf32x16_l0(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulsf32x16_l0
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulsf32x16.l0(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsf32x16_l0(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulsf32x16_l0_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsf32x16_l0_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulsf32x16_l0_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulsf32x16.l0.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsf32x16_l0_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulsf32x16_l1.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsf32x16_l1(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulsf32x16_l1
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulsf32x16.l1(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsf32x16_l1(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulsf32x16_l1_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsf32x16_l1_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulsf32x16_l1_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulsf32x16.l1.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsf32x16_l1_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulsf32x16_l2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsf32x16_l2(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulsf32x16_l2
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulsf32x16.l2(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsf32x16_l2(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulsf32x16_l2_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsf32x16_l2_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulsf32x16_l2_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulsf32x16.l2.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsf32x16_l2_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulsf32x16_l3.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsf32x16_l3(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulsf32x16_l3
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulsf32x16.l3(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsf32x16_l3(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulsf32x16_l3_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsf32x16_l3_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulsf32x16_l3_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulsf32x16.l3.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsf32x16_l3_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulsf48q32sp16s_l.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsf48q32sp16s_l(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int64 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulsf48q32sp16s_l
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulsf48q32sp16s.l(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <1 x i64> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsf48q32sp16s_l(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulsf48q32sp16s_l_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsf48q32sp16s_l_s2(ae_int64* ae_mul_S2_q0,ae_int64 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulsf48q32sp16s_l_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulsf48q32sp16s.l.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <1 x i64> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsf48q32sp16s_l_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulsf48q32sp16u_l.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsf48q32sp16u_l(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int64 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulsf48q32sp16u_l
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulsf48q32sp16u.l(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <1 x i64> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsf48q32sp16u_l(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulsf48q32sp16u_l_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsf48q32sp16u_l_s2(ae_int64* ae_mul_S2_q0,ae_int64 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulsf48q32sp16u_l_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulsf48q32sp16u.l.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <1 x i64> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsf48q32sp16u_l_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulsfp24x2r.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsfp24x2r(ae_int32x2* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulsfp24x2r
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulsfp24x2r(<2 x i32> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsfp24x2r(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulsfp24x2r_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsfp24x2r_s2(ae_int32x2* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulsfp24x2r_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulsfp24x2r.s2(<2 x i32> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsfp24x2r_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulsfp24x2ra.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsfp24x2ra(ae_int32x2* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulsfp24x2ra
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulsfp24x2ra(<2 x i32> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsfp24x2ra(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulsfp24x2ra_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsfp24x2ra_s2(ae_int32x2* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulsfp24x2ra_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulsfp24x2ra.s2(<2 x i32> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsfp24x2ra_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulsfp32x16x2ras_h.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsfp32x16x2ras_h(ae_int32x2* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulsfp32x16x2ras_h
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulsfp32x16x2ras.h(<2 x i32> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsfp32x16x2ras_h(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulsfp32x16x2ras_h_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsfp32x16x2ras_h_s2(ae_int32x2* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulsfp32x16x2ras_h_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulsfp32x16x2ras.h.s2(<2 x i32> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsfp32x16x2ras_h_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulsfp32x16x2ras_l.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsfp32x16x2ras_l(ae_int32x2* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulsfp32x16x2ras_l
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulsfp32x16x2ras.l(<2 x i32> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsfp32x16x2ras_l(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulsfp32x16x2ras_l_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsfp32x16x2ras_l_s2(ae_int32x2* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulsfp32x16x2ras_l_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulsfp32x16x2ras.l.s2(<2 x i32> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsfp32x16x2ras_l_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulsfp32x16x2rs_h.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsfp32x16x2rs_h(ae_int32x2* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulsfp32x16x2rs_h
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulsfp32x16x2rs.h(<2 x i32> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsfp32x16x2rs_h(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulsfp32x16x2rs_h_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsfp32x16x2rs_h_s2(ae_int32x2* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulsfp32x16x2rs_h_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulsfp32x16x2rs.h.s2(<2 x i32> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsfp32x16x2rs_h_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulsfp32x16x2rs_l.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsfp32x16x2rs_l(ae_int32x2* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulsfp32x16x2rs_l
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulsfp32x16x2rs.l(<2 x i32> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsfp32x16x2rs_l(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulsfp32x16x2rs_l_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsfp32x16x2rs_l_s2(ae_int32x2* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulsfp32x16x2rs_l_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulsfp32x16x2rs.l.s2(<2 x i32> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsfp32x16x2rs_l_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulsfp32x2ras.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsfp32x2ras(ae_int32x2* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulsfp32x2ras
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulsfp32x2ras(<2 x i32> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsfp32x2ras(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulsfp32x2rs.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsfp32x2rs(ae_int32x2* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulsfp32x2rs
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulsfp32x2rs(<2 x i32> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsfp32x2rs(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulsfq32sp24s_h_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsfq32sp24s_h_s2(ae_int64* ae_mul_S2_q0,ae_int64 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulsfq32sp24s_h_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulsfq32sp24s.h.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <1 x i64> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsfq32sp24s_h_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulsfq32sp24s_l_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsfq32sp24s_l_s2(ae_int64* ae_mul_S2_q0,ae_int64 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulsfq32sp24s_l_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulsfq32sp24s.l.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <1 x i64> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsfq32sp24s_l_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulsp24x2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsp24x2(ae_int32x2* ae_mul_q0,ae_int32x2 ae_mul_d0,ae_int32x2 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulsp24x2
// CHECK: %[[LD_AE_MUL_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulsp24x2(<2 x i32> %[[LD_AE_MUL_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsp24x2(ae_mul_q0, ae_mul_d0, ae_mul_d1);
}

//--- ae_mulsp24x2_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsp24x2_s2(ae_int32x2* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulsp24x2_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulsp24x2.s2(<2 x i32> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsp24x2_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulsp32x16x2_h.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsp32x16x2_h(ae_int32x2* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulsp32x16x2_h
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulsp32x16x2.h(<2 x i32> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsp32x16x2_h(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulsp32x16x2_l.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsp32x16x2_l(ae_int32x2* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulsp32x16x2_l
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulsp32x16x2.l(<2 x i32> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsp32x16x2_l(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulsp32x2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsp32x2(ae_int32x2* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulsp32x2
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulsp32x2(<2 x i32> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsp32x2(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulsq32sp16s_l_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsq32sp16s_l_s2(ae_int64* ae_mul_S2_q0,ae_int64 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulsq32sp16s_l_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulsq32sp16s.l.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <1 x i64> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsq32sp16s_l_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulsq32sp16u_l_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsq32sp16u_l_s2(ae_int64* ae_mul_S2_q0,ae_int64 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulsq32sp16u_l_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulsq32sp16u.l.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <1 x i64> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsq32sp16u_l_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulsrfq32sp24s_h_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsrfq32sp24s_h_s2(ae_int64* ae_mul_S2_q0,ae_int64 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulsrfq32sp24s_h_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulsrfq32sp24s.h.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <1 x i64> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsrfq32sp24s_h_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulsrfq32sp24s_l_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulsrfq32sp24s_l_s2(ae_int64* ae_mul_S2_q0,ae_int64 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulsrfq32sp24s_l_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulsrfq32sp24s.l.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <1 x i64> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulsrfq32sp24s_l_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulss32f48p16s_hh.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulss32f48p16s_hh(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulss32f48p16s_hh
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulss32f48p16s.hh(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulss32f48p16s_hh(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulss32f48p16s_hh_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulss32f48p16s_hh_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulss32f48p16s_hh_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulss32f48p16s.hh.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulss32f48p16s_hh_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulss32f48p16s_lh.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulss32f48p16s_lh(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulss32f48p16s_lh
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulss32f48p16s.lh(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulss32f48p16s_lh(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulss32f48p16s_lh_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulss32f48p16s_lh_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulss32f48p16s_lh_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulss32f48p16s.lh.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulss32f48p16s_lh_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulss32f48p16s_ll.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulss32f48p16s_ll(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulss32f48p16s_ll
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulss32f48p16s.ll(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulss32f48p16s_ll(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulss32f48p16s_ll_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulss32f48p16s_ll_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulss32f48p16s_ll_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulss32f48p16s.ll.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulss32f48p16s_ll_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulssd24_hh_ll.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulssd24_hh_ll(ae_int64* ae_mul_q0,ae_int32x2 ae_mul_d0,ae_int32x2 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulssd24_hh_ll
// CHECK: %[[LD_AE_MUL_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulssd24.hh.ll(<1 x i64> %[[LD_AE_MUL_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulssd24_hh_ll(ae_mul_q0, ae_mul_d0, ae_mul_d1);
}

//--- ae_mulssd24_hh_ll_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulssd24_hh_ll_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulssd24_hh_ll_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulssd24.hh.ll.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulssd24_hh_ll_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulssd24_hl_lh.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulssd24_hl_lh(ae_int64* ae_mul_q0,ae_int32x2 ae_mul_d0,ae_int32x2 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulssd24_hl_lh
// CHECK: %[[LD_AE_MUL_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulssd24.hl.lh(<1 x i64> %[[LD_AE_MUL_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulssd24_hl_lh(ae_mul_q0, ae_mul_d0, ae_mul_d1);
}

//--- ae_mulssd24_hl_lh_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulssd24_hl_lh_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulssd24_hl_lh_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulssd24.hl.lh.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulssd24_hl_lh_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulssd32x16_h1_l0.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulssd32x16_h1_l0(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulssd32x16_h1_l0
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulssd32x16.h1.l0(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulssd32x16_h1_l0(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulssd32x16_h1_l0_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulssd32x16_h1_l0_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulssd32x16_h1_l0_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulssd32x16.h1.l0.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulssd32x16_h1_l0_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulssd32x16_h3_l2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulssd32x16_h3_l2(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulssd32x16_h3_l2
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulssd32x16.h3.l2(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulssd32x16_h3_l2(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulssd32x16_h3_l2_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulssd32x16_h3_l2_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulssd32x16_h3_l2_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulssd32x16.h3.l2.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulssd32x16_h3_l2_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulssfd16ss_11_00.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulssfd16ss_11_00(ae_int32x2* opnd_ae_sem_mul_x2_S1_q0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulssfd16ss_11_00
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulssfd16ss.11.00(<2 x i32> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulssfd16ss_11_00(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulssfd16ss_11_00_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulssfd16ss_11_00_s2(ae_int32x2* ae_mul_S2_q0,ae_int16x4 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulssfd16ss_11_00_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulssfd16ss.11.00.s2(<2 x i32> %[[LD_AE_MUL_S2_Q0]], <4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulssfd16ss_11_00_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulssfd16ss_13_02.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulssfd16ss_13_02(ae_int32x2* opnd_ae_sem_mul_x2_S1_q0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulssfd16ss_13_02
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulssfd16ss.13.02(<2 x i32> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulssfd16ss_13_02(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulssfd16ss_13_02_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulssfd16ss_13_02_s2(ae_int32x2* ae_mul_S2_q0,ae_int16x4 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulssfd16ss_13_02_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulssfd16ss.13.02.s2(<2 x i32> %[[LD_AE_MUL_S2_Q0]], <4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulssfd16ss_13_02_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulssfd16ss_33_22.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulssfd16ss_33_22(ae_int32x2* opnd_ae_sem_mul_x2_S1_q0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulssfd16ss_33_22
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulssfd16ss.33.22(<2 x i32> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulssfd16ss_33_22(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulssfd16ss_33_22_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulssfd16ss_33_22_s2(ae_int32x2* ae_mul_S2_q0,ae_int16x4 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulssfd16ss_33_22_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulssfd16ss.33.22.s2(<2 x i32> %[[LD_AE_MUL_S2_Q0]], <4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulssfd16ss_33_22_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulssfd24_hh_ll.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulssfd24_hh_ll(ae_int64* ae_mul_q0,ae_int32x2 ae_mul_d0,ae_int32x2 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulssfd24_hh_ll
// CHECK: %[[LD_AE_MUL_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulssfd24.hh.ll(<1 x i64> %[[LD_AE_MUL_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulssfd24_hh_ll(ae_mul_q0, ae_mul_d0, ae_mul_d1);
}

//--- ae_mulssfd24_hh_ll_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulssfd24_hh_ll_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulssfd24_hh_ll_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulssfd24.hh.ll.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulssfd24_hh_ll_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulssfd24_hl_lh.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulssfd24_hl_lh(ae_int64* ae_mul_q0,ae_int32x2 ae_mul_d0,ae_int32x2 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulssfd24_hl_lh
// CHECK: %[[LD_AE_MUL_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulssfd24.hl.lh(<1 x i64> %[[LD_AE_MUL_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulssfd24_hl_lh(ae_mul_q0, ae_mul_d0, ae_mul_d1);
}

//--- ae_mulssfd24_hl_lh_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulssfd24_hl_lh_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulssfd24_hl_lh_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulssfd24.hl.lh.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulssfd24_hl_lh_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulssfd32x16_h1_l0.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulssfd32x16_h1_l0(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulssfd32x16_h1_l0
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulssfd32x16.h1.l0(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulssfd32x16_h1_l0(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulssfd32x16_h1_l0_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulssfd32x16_h1_l0_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulssfd32x16_h1_l0_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulssfd32x16.h1.l0.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulssfd32x16_h1_l0_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulssfd32x16_h3_l2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulssfd32x16_h3_l2(ae_int64* opnd_ae_sem_mul_x2_S1_q0,ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulssfd32x16_h3_l2
// CHECK: %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulssfd32x16.h3.l2(<1 x i64> %[[LD_OPND_AE_SEM_MUL_X2_S1_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulssfd32x16_h3_l2(opnd_ae_sem_mul_x2_S1_q0, opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulssfd32x16_h3_l2_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_mulssfd32x16_h3_l2_s2(ae_int64* ae_mul_S2_q0,ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulssfd32x16_h3_l2_s2
// CHECK: %[[LD_AE_MUL_S2_Q0:.*]] = load <1 x i64>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulssfd32x16.h3.l2.s2(<1 x i64> %[[LD_AE_MUL_S2_Q0]], <2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_mulssfd32x16_h3_l2_s2(ae_mul_S2_q0, ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulzaad24_hh_ll.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzaad24_hh_ll(ae_int32x2 ae_mul_d0,ae_int32x2 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulzaad24_hh_ll
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzaad24.hh.ll(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzaad24_hh_ll(ae_mul_d0, ae_mul_d1);
}

//--- ae_mulzaad24_hh_ll_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzaad24_hh_ll_s2(ae_int32x2 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulzaad24_hh_ll_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzaad24.hh.ll.s2(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzaad24_hh_ll_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulzaad24_hl_lh.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzaad24_hl_lh(ae_int32x2 ae_mul_d0,ae_int32x2 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulzaad24_hl_lh
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzaad24.hl.lh(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzaad24_hl_lh(ae_mul_d0, ae_mul_d1);
}

//--- ae_mulzaad24_hl_lh_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzaad24_hl_lh_s2(ae_int32x2 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulzaad24_hl_lh_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzaad24.hl.lh.s2(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzaad24_hl_lh_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulzaad32x16_h0_l1.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzaad32x16_h0_l1(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulzaad32x16_h0_l1
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzaad32x16.h0.l1(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzaad32x16_h0_l1(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulzaad32x16_h0_l1_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzaad32x16_h0_l1_s2(ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulzaad32x16_h0_l1_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzaad32x16.h0.l1.s2(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzaad32x16_h0_l1_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulzaad32x16_h1_l0.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzaad32x16_h1_l0(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulzaad32x16_h1_l0
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzaad32x16.h1.l0(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzaad32x16_h1_l0(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulzaad32x16_h1_l0_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzaad32x16_h1_l0_s2(ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulzaad32x16_h1_l0_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzaad32x16.h1.l0.s2(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzaad32x16_h1_l0_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulzaad32x16_h2_l3.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzaad32x16_h2_l3(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulzaad32x16_h2_l3
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzaad32x16.h2.l3(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzaad32x16_h2_l3(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulzaad32x16_h2_l3_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzaad32x16_h2_l3_s2(ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulzaad32x16_h2_l3_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzaad32x16.h2.l3.s2(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzaad32x16_h2_l3_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulzaad32x16_h3_l2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzaad32x16_h3_l2(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulzaad32x16_h3_l2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzaad32x16.h3.l2(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzaad32x16_h3_l2(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulzaad32x16_h3_l2_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzaad32x16_h3_l2_s2(ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulzaad32x16_h3_l2_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzaad32x16.h3.l2.s2(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzaad32x16_h3_l2_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulzaafd16ss_11_00.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_mulzaafd16ss_11_00(ae_int16x4 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulzaafd16ss_11_00
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulzaafd16ss.11.00(<4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_mulzaafd16ss_11_00(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulzaafd16ss_11_00_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_mulzaafd16ss_11_00_s2(ae_int16x4 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulzaafd16ss_11_00_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulzaafd16ss.11.00.s2(<4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_mulzaafd16ss_11_00_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulzaafd16ss_13_02.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_mulzaafd16ss_13_02(ae_int16x4 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulzaafd16ss_13_02
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulzaafd16ss.13.02(<4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_mulzaafd16ss_13_02(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulzaafd16ss_13_02_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_mulzaafd16ss_13_02_s2(ae_int16x4 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulzaafd16ss_13_02_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulzaafd16ss.13.02.s2(<4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_mulzaafd16ss_13_02_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulzaafd16ss_33_22.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_mulzaafd16ss_33_22(ae_int16x4 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulzaafd16ss_33_22
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulzaafd16ss.33.22(<4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_mulzaafd16ss_33_22(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulzaafd16ss_33_22_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_mulzaafd16ss_33_22_s2(ae_int16x4 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulzaafd16ss_33_22_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulzaafd16ss.33.22.s2(<4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_mulzaafd16ss_33_22_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulzaafd24_hh_ll.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzaafd24_hh_ll(ae_int32x2 ae_mul_d0,ae_int32x2 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulzaafd24_hh_ll
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzaafd24.hh.ll(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzaafd24_hh_ll(ae_mul_d0, ae_mul_d1);
}

//--- ae_mulzaafd24_hh_ll_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzaafd24_hh_ll_s2(ae_int32x2 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulzaafd24_hh_ll_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzaafd24.hh.ll.s2(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzaafd24_hh_ll_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulzaafd24_hl_lh.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzaafd24_hl_lh(ae_int32x2 ae_mul_d0,ae_int32x2 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulzaafd24_hl_lh
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzaafd24.hl.lh(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzaafd24_hl_lh(ae_mul_d0, ae_mul_d1);
}

//--- ae_mulzaafd24_hl_lh_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzaafd24_hl_lh_s2(ae_int32x2 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulzaafd24_hl_lh_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzaafd24.hl.lh.s2(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzaafd24_hl_lh_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulzaafd32x16_h0_l1.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzaafd32x16_h0_l1(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulzaafd32x16_h0_l1
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzaafd32x16.h0.l1(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzaafd32x16_h0_l1(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulzaafd32x16_h0_l1_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzaafd32x16_h0_l1_s2(ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulzaafd32x16_h0_l1_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzaafd32x16.h0.l1.s2(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzaafd32x16_h0_l1_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulzaafd32x16_h1_l0.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzaafd32x16_h1_l0(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulzaafd32x16_h1_l0
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzaafd32x16.h1.l0(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzaafd32x16_h1_l0(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulzaafd32x16_h1_l0_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzaafd32x16_h1_l0_s2(ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulzaafd32x16_h1_l0_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzaafd32x16.h1.l0.s2(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzaafd32x16_h1_l0_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulzaafd32x16_h2_l3.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzaafd32x16_h2_l3(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulzaafd32x16_h2_l3
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzaafd32x16.h2.l3(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzaafd32x16_h2_l3(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulzaafd32x16_h2_l3_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzaafd32x16_h2_l3_s2(ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulzaafd32x16_h2_l3_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzaafd32x16.h2.l3.s2(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzaafd32x16_h2_l3_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulzaafd32x16_h3_l2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzaafd32x16_h3_l2(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulzaafd32x16_h3_l2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzaafd32x16.h3.l2(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzaafd32x16_h3_l2(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulzaafd32x16_h3_l2_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzaafd32x16_h3_l2_s2(ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulzaafd32x16_h3_l2_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzaafd32x16.h3.l2.s2(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzaafd32x16_h3_l2_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulzasd24_hh_ll.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzasd24_hh_ll(ae_int32x2 ae_mul_d0,ae_int32x2 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulzasd24_hh_ll
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzasd24.hh.ll(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzasd24_hh_ll(ae_mul_d0, ae_mul_d1);
}

//--- ae_mulzasd24_hh_ll_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzasd24_hh_ll_s2(ae_int32x2 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulzasd24_hh_ll_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzasd24.hh.ll.s2(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzasd24_hh_ll_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulzasd24_hl_lh.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzasd24_hl_lh(ae_int32x2 ae_mul_d0,ae_int32x2 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulzasd24_hl_lh
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzasd24.hl.lh(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzasd24_hl_lh(ae_mul_d0, ae_mul_d1);
}

//--- ae_mulzasd24_hl_lh_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzasd24_hl_lh_s2(ae_int32x2 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulzasd24_hl_lh_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzasd24.hl.lh.s2(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzasd24_hl_lh_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulzasd32x16_h1_l0.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzasd32x16_h1_l0(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulzasd32x16_h1_l0
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzasd32x16.h1.l0(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzasd32x16_h1_l0(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulzasd32x16_h1_l0_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzasd32x16_h1_l0_s2(ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulzasd32x16_h1_l0_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzasd32x16.h1.l0.s2(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzasd32x16_h1_l0_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulzasd32x16_h3_l2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzasd32x16_h3_l2(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulzasd32x16_h3_l2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzasd32x16.h3.l2(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzasd32x16_h3_l2(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulzasd32x16_h3_l2_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzasd32x16_h3_l2_s2(ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulzasd32x16_h3_l2_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzasd32x16.h3.l2.s2(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzasd32x16_h3_l2_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulzasfd24_hh_ll.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzasfd24_hh_ll(ae_int32x2 ae_mul_d0,ae_int32x2 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulzasfd24_hh_ll
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzasfd24.hh.ll(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzasfd24_hh_ll(ae_mul_d0, ae_mul_d1);
}

//--- ae_mulzasfd24_hh_ll_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzasfd24_hh_ll_s2(ae_int32x2 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulzasfd24_hh_ll_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzasfd24.hh.ll.s2(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzasfd24_hh_ll_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulzasfd24_hl_lh.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzasfd24_hl_lh(ae_int32x2 ae_mul_d0,ae_int32x2 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulzasfd24_hl_lh
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzasfd24.hl.lh(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzasfd24_hl_lh(ae_mul_d0, ae_mul_d1);
}

//--- ae_mulzasfd24_hl_lh_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzasfd24_hl_lh_s2(ae_int32x2 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulzasfd24_hl_lh_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzasfd24.hl.lh.s2(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzasfd24_hl_lh_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulzasfd32x16_h1_l0.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzasfd32x16_h1_l0(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulzasfd32x16_h1_l0
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzasfd32x16.h1.l0(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzasfd32x16_h1_l0(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulzasfd32x16_h1_l0_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzasfd32x16_h1_l0_s2(ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulzasfd32x16_h1_l0_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzasfd32x16.h1.l0.s2(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzasfd32x16_h1_l0_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulzasfd32x16_h3_l2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzasfd32x16_h3_l2(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulzasfd32x16_h3_l2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzasfd32x16.h3.l2(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzasfd32x16_h3_l2(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulzasfd32x16_h3_l2_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzasfd32x16_h3_l2_s2(ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulzasfd32x16_h3_l2_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzasfd32x16.h3.l2.s2(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzasfd32x16_h3_l2_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulzsad24_hh_ll.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzsad24_hh_ll(ae_int32x2 ae_mul_d0,ae_int32x2 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulzsad24_hh_ll
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzsad24.hh.ll(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzsad24_hh_ll(ae_mul_d0, ae_mul_d1);
}

//--- ae_mulzsad24_hh_ll_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzsad24_hh_ll_s2(ae_int32x2 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulzsad24_hh_ll_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzsad24.hh.ll.s2(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzsad24_hh_ll_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulzsad32x16_h1_l0.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzsad32x16_h1_l0(ae_int32x2 ae_mul_d0,ae_int16x4 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulzsad32x16_h1_l0
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzsad32x16.h1.l0(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzsad32x16_h1_l0(ae_mul_d0, ae_mul_d1);
}

//--- ae_mulzsad32x16_h1_l0_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzsad32x16_h1_l0_s2(ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulzsad32x16_h1_l0_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzsad32x16.h1.l0.s2(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzsad32x16_h1_l0_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulzsad32x16_h3_l2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzsad32x16_h3_l2(ae_int32x2 ae_mul_d0,ae_int16x4 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulzsad32x16_h3_l2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzsad32x16.h3.l2(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzsad32x16_h3_l2(ae_mul_d0, ae_mul_d1);
}

//--- ae_mulzsad32x16_h3_l2_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzsad32x16_h3_l2_s2(ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulzsad32x16_h3_l2_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzsad32x16.h3.l2.s2(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzsad32x16_h3_l2_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulzsafd24_hh_ll.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzsafd24_hh_ll(ae_int32x2 ae_mul_d0,ae_int32x2 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulzsafd24_hh_ll
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzsafd24.hh.ll(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzsafd24_hh_ll(ae_mul_d0, ae_mul_d1);
}

//--- ae_mulzsafd24_hh_ll_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzsafd24_hh_ll_s2(ae_int32x2 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulzsafd24_hh_ll_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzsafd24.hh.ll.s2(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzsafd24_hh_ll_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulzsafd32x16_h1_l0.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzsafd32x16_h1_l0(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulzsafd32x16_h1_l0
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzsafd32x16.h1.l0(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzsafd32x16_h1_l0(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulzsafd32x16_h1_l0_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzsafd32x16_h1_l0_s2(ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulzsafd32x16_h1_l0_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzsafd32x16.h1.l0.s2(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzsafd32x16_h1_l0_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulzsafd32x16_h3_l2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzsafd32x16_h3_l2(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulzsafd32x16_h3_l2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzsafd32x16.h3.l2(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzsafd32x16_h3_l2(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulzsafd32x16_h3_l2_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzsafd32x16_h3_l2_s2(ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulzsafd32x16_h3_l2_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzsafd32x16.h3.l2.s2(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzsafd32x16_h3_l2_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulzssd24_hh_ll.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzssd24_hh_ll(ae_int32x2 ae_mul_d0,ae_int32x2 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulzssd24_hh_ll
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzssd24.hh.ll(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzssd24_hh_ll(ae_mul_d0, ae_mul_d1);
}

//--- ae_mulzssd24_hh_ll_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzssd24_hh_ll_s2(ae_int32x2 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulzssd24_hh_ll_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzssd24.hh.ll.s2(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzssd24_hh_ll_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulzssd24_hl_lh.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzssd24_hl_lh(ae_int32x2 ae_mul_d0,ae_int32x2 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulzssd24_hl_lh
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzssd24.hl.lh(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzssd24_hl_lh(ae_mul_d0, ae_mul_d1);
}

//--- ae_mulzssd24_hl_lh_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzssd24_hl_lh_s2(ae_int32x2 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulzssd24_hl_lh_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzssd24.hl.lh.s2(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzssd24_hl_lh_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulzssd32x16_h1_l0.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzssd32x16_h1_l0(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulzssd32x16_h1_l0
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzssd32x16.h1.l0(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzssd32x16_h1_l0(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulzssd32x16_h1_l0_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzssd32x16_h1_l0_s2(ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulzssd32x16_h1_l0_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzssd32x16.h1.l0.s2(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzssd32x16_h1_l0_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulzssd32x16_h3_l2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzssd32x16_h3_l2(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulzssd32x16_h3_l2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzssd32x16.h3.l2(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzssd32x16_h3_l2(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulzssd32x16_h3_l2_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzssd32x16_h3_l2_s2(ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulzssd32x16_h3_l2_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzssd32x16.h3.l2.s2(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzssd32x16_h3_l2_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulzssfd16ss_11_00.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_mulzssfd16ss_11_00(ae_int16x4 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulzssfd16ss_11_00
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulzssfd16ss.11.00(<4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_mulzssfd16ss_11_00(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulzssfd16ss_11_00_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_mulzssfd16ss_11_00_s2(ae_int16x4 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulzssfd16ss_11_00_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulzssfd16ss.11.00.s2(<4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_mulzssfd16ss_11_00_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulzssfd16ss_13_02.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_mulzssfd16ss_13_02(ae_int16x4 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulzssfd16ss_13_02
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulzssfd16ss.13.02(<4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_mulzssfd16ss_13_02(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulzssfd16ss_13_02_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_mulzssfd16ss_13_02_s2(ae_int16x4 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulzssfd16ss_13_02_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulzssfd16ss.13.02.s2(<4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_mulzssfd16ss_13_02_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulzssfd16ss_33_22.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_mulzssfd16ss_33_22(ae_int16x4 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulzssfd16ss_33_22
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulzssfd16ss.33.22(<4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_mulzssfd16ss_33_22(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulzssfd16ss_33_22_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_mulzssfd16ss_33_22_s2(ae_int16x4 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulzssfd16ss_33_22_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.mulzssfd16ss.33.22.s2(<4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_mulzssfd16ss_33_22_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulzssfd24_hh_ll.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzssfd24_hh_ll(ae_int32x2 ae_mul_d0,ae_int32x2 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulzssfd24_hh_ll
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzssfd24.hh.ll(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzssfd24_hh_ll(ae_mul_d0, ae_mul_d1);
}

//--- ae_mulzssfd24_hh_ll_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzssfd24_hh_ll_s2(ae_int32x2 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulzssfd24_hh_ll_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzssfd24.hh.ll.s2(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzssfd24_hh_ll_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulzssfd24_hl_lh.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzssfd24_hl_lh(ae_int32x2 ae_mul_d0,ae_int32x2 ae_mul_d1) {
// CHECK-LABEL: test_ae_mulzssfd24_hl_lh
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzssfd24.hl.lh(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzssfd24_hl_lh(ae_mul_d0, ae_mul_d1);
}

//--- ae_mulzssfd24_hl_lh_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzssfd24_hl_lh_s2(ae_int32x2 ae_mul_S2_d0,ae_int32x2 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulzssfd24_hl_lh_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzssfd24.hl.lh.s2(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzssfd24_hl_lh_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulzssfd32x16_h1_l0.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzssfd32x16_h1_l0(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulzssfd32x16_h1_l0
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzssfd32x16.h1.l0(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzssfd32x16_h1_l0(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulzssfd32x16_h1_l0_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzssfd32x16_h1_l0_s2(ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulzssfd32x16_h1_l0_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzssfd32x16.h1.l0.s2(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzssfd32x16_h1_l0_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_mulzssfd32x16_h3_l2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzssfd32x16_h3_l2(ae_int32x2 opnd_ae_sem_mul_x2_S1_d0,ae_int16x4 opnd_ae_sem_mul_x2_S1_d1) {
// CHECK-LABEL: test_ae_mulzssfd32x16_h3_l2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzssfd32x16.h3.l2(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzssfd32x16_h3_l2(opnd_ae_sem_mul_x2_S1_d0, opnd_ae_sem_mul_x2_S1_d1);
}

//--- ae_mulzssfd32x16_h3_l2_s2.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_mulzssfd32x16_h3_l2_s2(ae_int32x2 ae_mul_S2_d0,ae_int16x4 ae_mul_S2_d1) {
// CHECK-LABEL: test_ae_mulzssfd32x16_h3_l2_s2
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.mulzssfd32x16.h3.l2.s2(<2 x i32> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_mulzssfd32x16_h3_l2_s2(ae_mul_S2_d0, ae_mul_S2_d1);
}

//--- ae_nand.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_nand(ae_int64 ae_dr_to_dr_v0,ae_int64 ae_dr_to_dr_v1) {
// CHECK-LABEL: test_ae_nand
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.nand(<1 x i64> {{.*}}, <1 x i64> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_nand(ae_dr_to_dr_v0, ae_dr_to_dr_v1);
}

//--- ae_neg16s.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int16x4 test_ae_neg16s(ae_int16x4 ae_arth_v1) {
// CHECK-LABEL: test_ae_neg16s
// CHECK: %[[RET:.*]] = {{(tail)?}} call <4 x i16> @llvm.xtensa.ae.neg16s(<4 x i16> {{.*}})
// CHECK: ret <4 x i16> %[[RET]]
return __builtin_xtensa_ae_neg16s(ae_arth_v1);
}

//--- ae_neg24s.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_neg24s(ae_int32x2 ae_arth_v1) {
// CHECK-LABEL: test_ae_neg24s
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.neg24s(<2 x i32> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_neg24s(ae_arth_v1);
}

//--- ae_neg32.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_neg32(ae_int32x2 ae_arth_v1) {
// CHECK-LABEL: test_ae_neg32
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.neg32(<2 x i32> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_neg32(ae_arth_v1);
}

//--- ae_neg32s.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_neg32s(ae_int32x2 ae_arth_v1) {
// CHECK-LABEL: test_ae_neg32s
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.neg32s(<2 x i32> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_neg32s(ae_arth_v1);
}

//--- ae_neg64.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_neg64(ae_int64 ae_arth_v1) {
// CHECK-LABEL: test_ae_neg64
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.neg64(<1 x i64> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_neg64(ae_arth_v1);
}

//--- ae_neg64s.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_neg64s(ae_int64 ae_arth_v1) {
// CHECK-LABEL: test_ae_neg64s
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.neg64s(<1 x i64> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_neg64s(ae_arth_v1);
}

//--- ae_nsa64.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

int test_ae_nsa64(ae_int64 ae_dr_to_ar_v0) {
// CHECK-LABEL: test_ae_nsa64
// CHECK: %[[RET:.*]] = {{(tail)?}} call i32 @llvm.xtensa.ae.nsa64(<1 x i64> {{.*}})
// CHECK: ret i32 %[[RET]]
return __builtin_xtensa_ae_nsa64(ae_dr_to_ar_v0);
}

//--- ae_nsaz16_0.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

int test_ae_nsaz16_0(ae_int16x4 ae_dr_to_ar_v0) {
// CHECK-LABEL: test_ae_nsaz16_0
// CHECK: %[[RET:.*]] = {{(tail)?}} call i32 @llvm.xtensa.ae.nsaz16.0(<4 x i16> {{.*}})
// CHECK: ret i32 %[[RET]]
return __builtin_xtensa_ae_nsaz16_0(ae_dr_to_ar_v0);
}

//--- ae_nsaz32_l.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

int test_ae_nsaz32_l(ae_int32x2 ae_dr_to_ar_v0) {
// CHECK-LABEL: test_ae_nsaz32_l
// CHECK: %[[RET:.*]] = {{(tail)?}} call i32 @llvm.xtensa.ae.nsaz32.l(<2 x i32> {{.*}})
// CHECK: ret i32 %[[RET]]
return __builtin_xtensa_ae_nsaz32_l(ae_dr_to_ar_v0);
}

//--- ae_or.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_or(ae_int64 ae_dr_to_dr_v0,ae_int64 ae_dr_to_dr_v1) {
// CHECK-LABEL: test_ae_or
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.or(<1 x i64> {{.*}}, <1 x i64> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_or(ae_dr_to_dr_v0, ae_dr_to_dr_v1);
}

//--- ae_pksr24.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_pksr24(ae_int32x2* ae_pks_d,ae_int64 ae_pks_s,immediate ae_imm2) {
// CHECK-LABEL: test_ae_pksr24
// CHECK: %[[LD_AE_PKS_D:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.pksr24(<2 x i32> %[[LD_AE_PKS_D]], <1 x i64> {{.*}}, i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_pksr24(ae_pks_d, ae_pks_s, 0);
}

//--- ae_pksr32.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_pksr32(ae_int32x2* ae_pks_d,ae_int64 ae_pks_s,immediate ae_imm2) {
// CHECK-LABEL: test_ae_pksr32
// CHECK: %[[LD_AE_PKS_D:.*]] = load <2 x i32>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.pksr32(<2 x i32> %[[LD_AE_PKS_D]], <1 x i64> {{.*}}, i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_pksr32(ae_pks_d, ae_pks_s, 0);
}

//--- ae_round16x4f32sasym.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int16x4 test_ae_round16x4f32sasym(ae_int32x2 ae_arth_v1,ae_int32x2 ae_arth_v0) {
// CHECK-LABEL: test_ae_round16x4f32sasym
// CHECK: %[[RET:.*]] = {{(tail)?}} call <4 x i16> @llvm.xtensa.ae.round16x4f32sasym(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <4 x i16> %[[RET]]
return __builtin_xtensa_ae_round16x4f32sasym(ae_arth_v1, ae_arth_v0);
}

//--- ae_round16x4f32ssym.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int16x4 test_ae_round16x4f32ssym(ae_int32x2 ae_arth_v1,ae_int32x2 ae_arth_v0) {
// CHECK-LABEL: test_ae_round16x4f32ssym
// CHECK: %[[RET:.*]] = {{(tail)?}} call <4 x i16> @llvm.xtensa.ae.round16x4f32ssym(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <4 x i16> %[[RET]]
return __builtin_xtensa_ae_round16x4f32ssym(ae_arth_v1, ae_arth_v0);
}

//--- ae_round24x2f48sasym.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_round24x2f48sasym(ae_int64 ae_arth_v0,ae_int64 ae_arth_v1) {
// CHECK-LABEL: test_ae_round24x2f48sasym
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.round24x2f48sasym(<1 x i64> {{.*}}, <1 x i64> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_round24x2f48sasym(ae_arth_v0, ae_arth_v1);
}

//--- ae_round24x2f48ssym.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_round24x2f48ssym(ae_int64 ae_arth_v0,ae_int64 ae_arth_v1) {
// CHECK-LABEL: test_ae_round24x2f48ssym
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.round24x2f48ssym(<1 x i64> {{.*}}, <1 x i64> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_round24x2f48ssym(ae_arth_v0, ae_arth_v1);
}

//--- ae_round32x2f48sasym.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_round32x2f48sasym(ae_int64 ae_arth_v0,ae_int64 ae_arth_v1) {
// CHECK-LABEL: test_ae_round32x2f48sasym
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.round32x2f48sasym(<1 x i64> {{.*}}, <1 x i64> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_round32x2f48sasym(ae_arth_v0, ae_arth_v1);
}

//--- ae_round32x2f48ssym.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_round32x2f48ssym(ae_int64 ae_arth_v0,ae_int64 ae_arth_v1) {
// CHECK-LABEL: test_ae_round32x2f48ssym
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.round32x2f48ssym(<1 x i64> {{.*}}, <1 x i64> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_round32x2f48ssym(ae_arth_v0, ae_arth_v1);
}

//--- ae_round32x2f64sasym.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_round32x2f64sasym(ae_int64 ae_arth_v0,ae_int64 ae_arth_v1) {
// CHECK-LABEL: test_ae_round32x2f64sasym
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.round32x2f64sasym(<1 x i64> {{.*}}, <1 x i64> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_round32x2f64sasym(ae_arth_v0, ae_arth_v1);
}

//--- ae_round32x2f64ssym.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_round32x2f64ssym(ae_int64 ae_arth_v0,ae_int64 ae_arth_v1) {
// CHECK-LABEL: test_ae_round32x2f64ssym
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.round32x2f64ssym(<1 x i64> {{.*}}, <1 x i64> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_round32x2f64ssym(ae_arth_v0, ae_arth_v1);
}

//--- ae_roundsp16f24asym.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_roundsp16f24asym(ae_int32x2 ae_arth_v0) {
// CHECK-LABEL: test_ae_roundsp16f24asym
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.roundsp16f24asym(<2 x i32> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_roundsp16f24asym(ae_arth_v0);
}

//--- ae_roundsp16f24sym.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_roundsp16f24sym(ae_int32x2 ae_arth_v0) {
// CHECK-LABEL: test_ae_roundsp16f24sym
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.roundsp16f24sym(<2 x i32> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_roundsp16f24sym(ae_arth_v0);
}

//--- ae_roundsp16q48x2asym.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_roundsp16q48x2asym(ae_int64 ae_arth_v0,ae_int64 ae_arth_v1) {
// CHECK-LABEL: test_ae_roundsp16q48x2asym
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.roundsp16q48x2asym(<1 x i64> {{.*}}, <1 x i64> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_roundsp16q48x2asym(ae_arth_v0, ae_arth_v1);
}

//--- ae_roundsp16q48x2sym.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_roundsp16q48x2sym(ae_int64 ae_arth_v0,ae_int64 ae_arth_v1) {
// CHECK-LABEL: test_ae_roundsp16q48x2sym
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.roundsp16q48x2sym(<1 x i64> {{.*}}, <1 x i64> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_roundsp16q48x2sym(ae_arth_v0, ae_arth_v1);
}

//--- ae_roundsq32f48asym.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_roundsq32f48asym(ae_int64 ae_arth_v1) {
// CHECK-LABEL: test_ae_roundsq32f48asym
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.roundsq32f48asym(<1 x i64> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_roundsq32f48asym(ae_arth_v1);
}

//--- ae_roundsq32f48sym.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_roundsq32f48sym(ae_int64 ae_arth_v1) {
// CHECK-LABEL: test_ae_roundsq32f48sym
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.roundsq32f48sym(<1 x i64> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_roundsq32f48sym(ae_arth_v1);
}

//--- ae_s16_0_i.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s16_0_i(ae_int16x4 ae_ls_v,ae_int16* ars,immediate ae_immls16) {
// CHECK-LABEL: test_ae_s16_0_i
// CHECK:  {{(tail)?}} call void @llvm.xtensa.ae.s16.0.i(<4 x i16> {{.*}}, ptr {{.*}}, i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s16_0_i(ae_ls_v, ars, -16);
}

//--- ae_s16_0_ip.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s16_0_ip(ae_int16x4 ae_ls_v,ae_int16** ars,immediate ae_immls16) {
// CHECK-LABEL: test_ae_s16_0_ip
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.s16.0.ip(<4 x i16> {{.*}}, ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s16_0_ip(ae_ls_v, ars, -16);
}

//--- ae_s16_0_x.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s16_0_x(ae_int16x4 ae_ls_v,ae_int16* ars,int art) {
// CHECK-LABEL: test_ae_s16_0_x
// CHECK:  {{(tail)?}} call void @llvm.xtensa.ae.s16.0.x(<4 x i16> {{.*}}, ptr {{.*}}, i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s16_0_x(ae_ls_v, ars, art);
}

//--- ae_s16_0_xc.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s16_0_xc(ae_int16x4 ae_ls_v,ae_int16** ars,int art) {
// CHECK-LABEL: test_ae_s16_0_xc
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.s16.0.xc(<4 x i16> {{.*}}, ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s16_0_xc(ae_ls_v, ars, art);
}

//--- ae_s16_0_xp.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s16_0_xp(ae_int16x4 ae_ls_v,ae_int16** ars,int art) {
// CHECK-LABEL: test_ae_s16_0_xp
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.s16.0.xp(<4 x i16> {{.*}}, ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s16_0_xp(ae_ls_v, ars, art);
}

//--- ae_s16m_l_i.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s16m_l_i(ae_int32x2 ae_ls_v,ae_int16* ars,immediate ae_immls16) {
// CHECK-LABEL: test_ae_s16m_l_i
// CHECK:  {{(tail)?}} call void @llvm.xtensa.ae.s16m.l.i(<2 x i32> {{.*}}, ptr {{.*}}, i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s16m_l_i(ae_ls_v, ars, -16);
}

//--- ae_s16m_l_iu.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s16m_l_iu(ae_int32x2 ae_ls_v,ae_int16** ars,immediate ae_immls16) {
// CHECK-LABEL: test_ae_s16m_l_iu
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.s16m.l.iu(<2 x i32> {{.*}}, ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s16m_l_iu(ae_ls_v, ars, -16);
}

//--- ae_s16m_l_x.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s16m_l_x(ae_int32x2 ae_ls_v,ae_int16* ars,int art) {
// CHECK-LABEL: test_ae_s16m_l_x
// CHECK:  {{(tail)?}} call void @llvm.xtensa.ae.s16m.l.x(<2 x i32> {{.*}}, ptr {{.*}}, i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s16m_l_x(ae_ls_v, ars, art);
}

//--- ae_s16m_l_xc.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s16m_l_xc(ae_int32x2 ae_ls_v,ae_int16** ars,int art) {
// CHECK-LABEL: test_ae_s16m_l_xc
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.s16m.l.xc(<2 x i32> {{.*}}, ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s16m_l_xc(ae_ls_v, ars, art);
}

//--- ae_s16m_l_xu.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s16m_l_xu(ae_int32x2 ae_ls_v,ae_int16** ars,int art) {
// CHECK-LABEL: test_ae_s16m_l_xu
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.s16m.l.xu(<2 x i32> {{.*}}, ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s16m_l_xu(ae_ls_v, ars, art);
}

//--- ae_s16x2m_i.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s16x2m_i(ae_int32x2 ae_ls_v,ae_int16x2* ars,immediate ae_immls32) {
// CHECK-LABEL: test_ae_s16x2m_i
// CHECK:  {{(tail)?}} call void @llvm.xtensa.ae.s16x2m.i(<2 x i32> {{.*}}, ptr {{.*}}, i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s16x2m_i(ae_ls_v, ars, -32);
}

//--- ae_s16x2m_iu.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s16x2m_iu(ae_int32x2 ae_ls_v,ae_int16x2** ars,immediate ae_immls32) {
// CHECK-LABEL: test_ae_s16x2m_iu
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.s16x2m.iu(<2 x i32> {{.*}}, ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s16x2m_iu(ae_ls_v, ars, -32);
}

//--- ae_s16x2m_x.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s16x2m_x(ae_int32x2 ae_ls_v,ae_int16x2* ars,int art) {
// CHECK-LABEL: test_ae_s16x2m_x
// CHECK:  {{(tail)?}} call void @llvm.xtensa.ae.s16x2m.x(<2 x i32> {{.*}}, ptr {{.*}}, i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s16x2m_x(ae_ls_v, ars, art);
}

//--- ae_s16x2m_xc.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s16x2m_xc(ae_int32x2 ae_ls_v,ae_int16x2** ars,int art) {
// CHECK-LABEL: test_ae_s16x2m_xc
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.s16x2m.xc(<2 x i32> {{.*}}, ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s16x2m_xc(ae_ls_v, ars, art);
}

//--- ae_s16x2m_xu.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s16x2m_xu(ae_int32x2 ae_ls_v,ae_int16x2** ars,int art) {
// CHECK-LABEL: test_ae_s16x2m_xu
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.s16x2m.xu(<2 x i32> {{.*}}, ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s16x2m_xu(ae_ls_v, ars, art);
}

//--- ae_s16x4_i.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s16x4_i(ae_int16x4 ae_ls_v,ae_int16x4* ars,immediate ae_immls64) {
// CHECK-LABEL: test_ae_s16x4_i
// CHECK:  {{(tail)?}} call void @llvm.xtensa.ae.s16x4.i(<4 x i16> {{.*}}, ptr {{.*}}, i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s16x4_i(ae_ls_v, ars, -64);
}

//--- ae_s16x4_ip.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s16x4_ip(ae_int16x4 ae_ls_v,ae_int16x4** ars,immediate ae_immls64pos) {
// CHECK-LABEL: test_ae_s16x4_ip
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.s16x4.ip(<4 x i16> {{.*}}, ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s16x4_ip(ae_ls_v, ars, 0);
}

//--- ae_s16x4_ric.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s16x4_ric(ae_int16x4 ae_ls_v,ae_int16x4** ars) {
// CHECK-LABEL: test_ae_s16x4_ric
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.s16x4.ric(<4 x i16> {{.*}}, ptr %[[LD_ARS]])
// CHECK: ret void
 __builtin_xtensa_ae_s16x4_ric(ae_ls_v, ars);
}

//--- ae_s16x4_rip.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s16x4_rip(ae_int16x4 ae_ls_v,ae_int16x4** ars) {
// CHECK-LABEL: test_ae_s16x4_rip
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.s16x4.rip(<4 x i16> {{.*}}, ptr %[[LD_ARS]])
// CHECK: ret void
 __builtin_xtensa_ae_s16x4_rip(ae_ls_v, ars);
}

//--- ae_s16x4_x.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s16x4_x(ae_int16x4 ae_ls_v,ae_int16x4* ars,int art) {
// CHECK-LABEL: test_ae_s16x4_x
// CHECK:  {{(tail)?}} call void @llvm.xtensa.ae.s16x4.x(<4 x i16> {{.*}}, ptr {{.*}}, i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s16x4_x(ae_ls_v, ars, art);
}

//--- ae_s16x4_xc.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s16x4_xc(ae_int16x4 ae_ls_v,ae_int16x4** ars,int art) {
// CHECK-LABEL: test_ae_s16x4_xc
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.s16x4.xc(<4 x i16> {{.*}}, ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s16x4_xc(ae_ls_v, ars, art);
}

//--- ae_s16x4_xp.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s16x4_xp(ae_int16x4 ae_ls_v,ae_int16x4** ars,int art) {
// CHECK-LABEL: test_ae_s16x4_xp
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.s16x4.xp(<4 x i16> {{.*}}, ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s16x4_xp(ae_ls_v, ars, art);
}

//--- ae_s24ra64s_i.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s24ra64s_i(ae_int64 ae_ls_v1,ae_int32* ars,immediate ae_immls32) {
// CHECK-LABEL: test_ae_s24ra64s_i
// CHECK:  {{(tail)?}} call void @llvm.xtensa.ae.s24ra64s.i(<1 x i64> {{.*}}, ptr {{.*}}, i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s24ra64s_i(ae_ls_v1, ars, -32);
}

//--- ae_s24ra64s_ip.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s24ra64s_ip(ae_int64 ae_ls_v1,ae_int32** ars,immediate ae_immls32) {
// CHECK-LABEL: test_ae_s24ra64s_ip
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.s24ra64s.ip(<1 x i64> {{.*}}, ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s24ra64s_ip(ae_ls_v1, ars, -32);
}

//--- ae_s24ra64s_x.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s24ra64s_x(ae_int64 ae_ls_v1,ae_int32* ars,int art) {
// CHECK-LABEL: test_ae_s24ra64s_x
// CHECK:  {{(tail)?}} call void @llvm.xtensa.ae.s24ra64s.x(<1 x i64> {{.*}}, ptr {{.*}}, i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s24ra64s_x(ae_ls_v1, ars, art);
}

//--- ae_s24ra64s_xc.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s24ra64s_xc(ae_int64 ae_ls_v1,ae_int32** ars,int art) {
// CHECK-LABEL: test_ae_s24ra64s_xc
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.s24ra64s.xc(<1 x i64> {{.*}}, ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s24ra64s_xc(ae_ls_v1, ars, art);
}

//--- ae_s24ra64s_xp.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s24ra64s_xp(ae_int64 ae_ls_v1,ae_int32** ars,int art) {
// CHECK-LABEL: test_ae_s24ra64s_xp
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.s24ra64s.xp(<1 x i64> {{.*}}, ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s24ra64s_xp(ae_ls_v1, ars, art);
}

//--- ae_s24x2ra64s_ip.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s24x2ra64s_ip(ae_int64 ae_ls_v2,ae_int64 ae_ls_v1,ae_int32x2** ars) {
// CHECK-LABEL: test_ae_s24x2ra64s_ip
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.s24x2ra64s.ip(<1 x i64> {{.*}}, <1 x i64> {{.*}}, ptr %[[LD_ARS]])
// CHECK: ret void
 __builtin_xtensa_ae_s24x2ra64s_ip(ae_ls_v2, ae_ls_v1, ars);
}

//--- ae_s32_l_i.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s32_l_i(ae_int32x2 ae_ls_v,ae_int32* ars,immediate ae_immls32) {
// CHECK-LABEL: test_ae_s32_l_i
// CHECK:  {{(tail)?}} call void @llvm.xtensa.ae.s32.l.i(<2 x i32> {{.*}}, ptr {{.*}}, i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s32_l_i(ae_ls_v, ars, -32);
}

//--- ae_s32_l_ip.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s32_l_ip(ae_int32x2 ae_ls_v,ae_int32** ars,immediate ae_immls32) {
// CHECK-LABEL: test_ae_s32_l_ip
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.s32.l.ip(<2 x i32> {{.*}}, ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s32_l_ip(ae_ls_v, ars, -32);
}

//--- ae_s32_l_x.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s32_l_x(ae_int32x2 ae_ls_v,ae_int32* ars,int art) {
// CHECK-LABEL: test_ae_s32_l_x
// CHECK:  {{(tail)?}} call void @llvm.xtensa.ae.s32.l.x(<2 x i32> {{.*}}, ptr {{.*}}, i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s32_l_x(ae_ls_v, ars, art);
}

//--- ae_s32_l_xc.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s32_l_xc(ae_int32x2 ae_ls_v,ae_int32** ars,int art) {
// CHECK-LABEL: test_ae_s32_l_xc
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.s32.l.xc(<2 x i32> {{.*}}, ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s32_l_xc(ae_ls_v, ars, art);
}

//--- ae_s32_l_xp.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s32_l_xp(ae_int32x2 ae_ls_v,ae_int32** ars,int art) {
// CHECK-LABEL: test_ae_s32_l_xp
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.s32.l.xp(<2 x i32> {{.*}}, ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s32_l_xp(ae_ls_v, ars, art);
}

//--- ae_s32f24_l_i.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s32f24_l_i(ae_int32x2 ae_ls_v,ae_int32* ars,immediate ae_immls32) {
// CHECK-LABEL: test_ae_s32f24_l_i
// CHECK:  {{(tail)?}} call void @llvm.xtensa.ae.s32f24.l.i(<2 x i32> {{.*}}, ptr {{.*}}, i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s32f24_l_i(ae_ls_v, ars, -32);
}

//--- ae_s32f24_l_ip.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s32f24_l_ip(ae_int32x2 ae_ls_v,ae_int32** ars,immediate ae_immls32) {
// CHECK-LABEL: test_ae_s32f24_l_ip
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.s32f24.l.ip(<2 x i32> {{.*}}, ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s32f24_l_ip(ae_ls_v, ars, -32);
}

//--- ae_s32f24_l_x.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s32f24_l_x(ae_int32x2 ae_ls_v,ae_int32* ars,int art) {
// CHECK-LABEL: test_ae_s32f24_l_x
// CHECK:  {{(tail)?}} call void @llvm.xtensa.ae.s32f24.l.x(<2 x i32> {{.*}}, ptr {{.*}}, i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s32f24_l_x(ae_ls_v, ars, art);
}

//--- ae_s32f24_l_xc.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s32f24_l_xc(ae_int32x2 ae_ls_v,ae_int32** ars,int art) {
// CHECK-LABEL: test_ae_s32f24_l_xc
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.s32f24.l.xc(<2 x i32> {{.*}}, ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s32f24_l_xc(ae_ls_v, ars, art);
}

//--- ae_s32f24_l_xp.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s32f24_l_xp(ae_int32x2 ae_ls_v,ae_int32** ars,int art) {
// CHECK-LABEL: test_ae_s32f24_l_xp
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.s32f24.l.xp(<2 x i32> {{.*}}, ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s32f24_l_xp(ae_ls_v, ars, art);
}

//--- ae_s32m_i.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s32m_i(ae_int64 ae_ls_v,ae_int32* ars,immediate ae_immls32) {
// CHECK-LABEL: test_ae_s32m_i
// CHECK:  {{(tail)?}} call void @llvm.xtensa.ae.s32m.i(<1 x i64> {{.*}}, ptr {{.*}}, i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s32m_i(ae_ls_v, ars, -32);
}

//--- ae_s32m_iu.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s32m_iu(ae_int64 ae_ls_v,ae_int32** ars,immediate ae_immls32) {
// CHECK-LABEL: test_ae_s32m_iu
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.s32m.iu(<1 x i64> {{.*}}, ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s32m_iu(ae_ls_v, ars, -32);
}

//--- ae_s32m_x.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s32m_x(ae_int64 ae_ls_v,ae_int32* ars,int art) {
// CHECK-LABEL: test_ae_s32m_x
// CHECK:  {{(tail)?}} call void @llvm.xtensa.ae.s32m.x(<1 x i64> {{.*}}, ptr {{.*}}, i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s32m_x(ae_ls_v, ars, art);
}

//--- ae_s32m_xc.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s32m_xc(ae_int64 ae_ls_v,ae_int32** ars,int art) {
// CHECK-LABEL: test_ae_s32m_xc
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.s32m.xc(<1 x i64> {{.*}}, ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s32m_xc(ae_ls_v, ars, art);
}

//--- ae_s32m_xu.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s32m_xu(ae_int64 ae_ls_v,ae_int32** ars,int art) {
// CHECK-LABEL: test_ae_s32m_xu
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.s32m.xu(<1 x i64> {{.*}}, ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s32m_xu(ae_ls_v, ars, art);
}

//--- ae_s32ra64s_i.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s32ra64s_i(ae_int64 ae_ls_v1,ae_int32* ars,immediate ae_immls32) {
// CHECK-LABEL: test_ae_s32ra64s_i
// CHECK:  {{(tail)?}} call void @llvm.xtensa.ae.s32ra64s.i(<1 x i64> {{.*}}, ptr {{.*}}, i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s32ra64s_i(ae_ls_v1, ars, -32);
}

//--- ae_s32ra64s_ip.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s32ra64s_ip(ae_int64 ae_ls_v1,ae_int32** ars,immediate ae_immls32) {
// CHECK-LABEL: test_ae_s32ra64s_ip
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.s32ra64s.ip(<1 x i64> {{.*}}, ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s32ra64s_ip(ae_ls_v1, ars, -32);
}

//--- ae_s32ra64s_x.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s32ra64s_x(ae_int64 ae_ls_v1,ae_int32* ars,int art) {
// CHECK-LABEL: test_ae_s32ra64s_x
// CHECK:  {{(tail)?}} call void @llvm.xtensa.ae.s32ra64s.x(<1 x i64> {{.*}}, ptr {{.*}}, i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s32ra64s_x(ae_ls_v1, ars, art);
}

//--- ae_s32ra64s_xc.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s32ra64s_xc(ae_int64 ae_ls_v1,ae_int32** ars,int art) {
// CHECK-LABEL: test_ae_s32ra64s_xc
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.s32ra64s.xc(<1 x i64> {{.*}}, ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s32ra64s_xc(ae_ls_v1, ars, art);
}

//--- ae_s32ra64s_xp.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s32ra64s_xp(ae_int64 ae_ls_v1,ae_int32** ars,int art) {
// CHECK-LABEL: test_ae_s32ra64s_xp
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.s32ra64s.xp(<1 x i64> {{.*}}, ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s32ra64s_xp(ae_ls_v1, ars, art);
}

//--- ae_s32x2_i.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s32x2_i(ae_int32x2 ae_ls_v,ae_int32x2* ars,immediate ae_immls64) {
// CHECK-LABEL: test_ae_s32x2_i
// CHECK:  {{(tail)?}} call void @llvm.xtensa.ae.s32x2.i(<2 x i32> {{.*}}, ptr {{.*}}, i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s32x2_i(ae_ls_v, ars, -64);
}

//--- ae_s32x2_ip.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s32x2_ip(ae_int32x2 ae_ls_v,ae_int32x2** ars,immediate ae_immls64pos) {
// CHECK-LABEL: test_ae_s32x2_ip
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.s32x2.ip(<2 x i32> {{.*}}, ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s32x2_ip(ae_ls_v, ars, 0);
}

//--- ae_s32x2_ric.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s32x2_ric(ae_int32x2 ae_ls_v,ae_int32x2** ars) {
// CHECK-LABEL: test_ae_s32x2_ric
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.s32x2.ric(<2 x i32> {{.*}}, ptr %[[LD_ARS]])
// CHECK: ret void
 __builtin_xtensa_ae_s32x2_ric(ae_ls_v, ars);
}

//--- ae_s32x2_rip.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s32x2_rip(ae_int32x2 ae_ls_v,ae_int32x2** ars) {
// CHECK-LABEL: test_ae_s32x2_rip
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.s32x2.rip(<2 x i32> {{.*}}, ptr %[[LD_ARS]])
// CHECK: ret void
 __builtin_xtensa_ae_s32x2_rip(ae_ls_v, ars);
}

//--- ae_s32x2_x.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s32x2_x(ae_int32x2 ae_ls_v,ae_int32x2* ars,int art) {
// CHECK-LABEL: test_ae_s32x2_x
// CHECK:  {{(tail)?}} call void @llvm.xtensa.ae.s32x2.x(<2 x i32> {{.*}}, ptr {{.*}}, i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s32x2_x(ae_ls_v, ars, art);
}

//--- ae_s32x2_xc.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s32x2_xc(ae_int32x2 ae_ls_v,ae_int32x2** ars,int art) {
// CHECK-LABEL: test_ae_s32x2_xc
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.s32x2.xc(<2 x i32> {{.*}}, ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s32x2_xc(ae_ls_v, ars, art);
}

//--- ae_s32x2_xp.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s32x2_xp(ae_int32x2 ae_ls_v,ae_int32x2** ars,int art) {
// CHECK-LABEL: test_ae_s32x2_xp
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.s32x2.xp(<2 x i32> {{.*}}, ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s32x2_xp(ae_ls_v, ars, art);
}

//--- ae_s32x2f24_i.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s32x2f24_i(ae_int32x2 ae_ls_v,ae_int32x2* ars,immediate ae_immls64) {
// CHECK-LABEL: test_ae_s32x2f24_i
// CHECK:  {{(tail)?}} call void @llvm.xtensa.ae.s32x2f24.i(<2 x i32> {{.*}}, ptr {{.*}}, i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s32x2f24_i(ae_ls_v, ars, -64);
}

//--- ae_s32x2f24_ip.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s32x2f24_ip(ae_int32x2 ae_ls_v,ae_int32x2** ars,immediate ae_immls64pos) {
// CHECK-LABEL: test_ae_s32x2f24_ip
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.s32x2f24.ip(<2 x i32> {{.*}}, ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s32x2f24_ip(ae_ls_v, ars, 0);
}

//--- ae_s32x2f24_ric.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s32x2f24_ric(ae_int32x2 ae_ls_v,ae_int32x2** ars) {
// CHECK-LABEL: test_ae_s32x2f24_ric
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.s32x2f24.ric(<2 x i32> {{.*}}, ptr %[[LD_ARS]])
// CHECK: ret void
 __builtin_xtensa_ae_s32x2f24_ric(ae_ls_v, ars);
}

//--- ae_s32x2f24_rip.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s32x2f24_rip(ae_int32x2 ae_ls_v,ae_int32x2** ars) {
// CHECK-LABEL: test_ae_s32x2f24_rip
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.s32x2f24.rip(<2 x i32> {{.*}}, ptr %[[LD_ARS]])
// CHECK: ret void
 __builtin_xtensa_ae_s32x2f24_rip(ae_ls_v, ars);
}

//--- ae_s32x2f24_x.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s32x2f24_x(ae_int32x2 ae_ls_v,ae_int32x2* ars,int art) {
// CHECK-LABEL: test_ae_s32x2f24_x
// CHECK:  {{(tail)?}} call void @llvm.xtensa.ae.s32x2f24.x(<2 x i32> {{.*}}, ptr {{.*}}, i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s32x2f24_x(ae_ls_v, ars, art);
}

//--- ae_s32x2f24_xc.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s32x2f24_xc(ae_int32x2 ae_ls_v,ae_int32x2** ars,int art) {
// CHECK-LABEL: test_ae_s32x2f24_xc
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.s32x2f24.xc(<2 x i32> {{.*}}, ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s32x2f24_xc(ae_ls_v, ars, art);
}

//--- ae_s32x2f24_xp.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s32x2f24_xp(ae_int32x2 ae_ls_v,ae_int32x2** ars,int art) {
// CHECK-LABEL: test_ae_s32x2f24_xp
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.s32x2f24.xp(<2 x i32> {{.*}}, ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s32x2f24_xp(ae_ls_v, ars, art);
}

//--- ae_s32x2ra64s_ip.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s32x2ra64s_ip(ae_int64 ae_ls_v2,ae_int64 ae_ls_v1,ae_int32x2** ars) {
// CHECK-LABEL: test_ae_s32x2ra64s_ip
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.s32x2ra64s.ip(<1 x i64> {{.*}}, <1 x i64> {{.*}}, ptr %[[LD_ARS]])
// CHECK: ret void
 __builtin_xtensa_ae_s32x2ra64s_ip(ae_ls_v2, ae_ls_v1, ars);
}

//--- ae_s64_i.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s64_i(ae_int64 ae_ls_v,ae_int64* ars,immediate ae_immls64) {
// CHECK-LABEL: test_ae_s64_i
// CHECK:  {{(tail)?}} call void @llvm.xtensa.ae.s64.i(<1 x i64> {{.*}}, ptr {{.*}}, i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s64_i(ae_ls_v, ars, -64);
}

//--- ae_s64_ip.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s64_ip(ae_int64 ae_ls_v,ae_int64** ars,immediate ae_immls64) {
// CHECK-LABEL: test_ae_s64_ip
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.s64.ip(<1 x i64> {{.*}}, ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s64_ip(ae_ls_v, ars, -64);
}

//--- ae_s64_x.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s64_x(ae_int64 ae_ls_v,ae_int64* ars,int art) {
// CHECK-LABEL: test_ae_s64_x
// CHECK:  {{(tail)?}} call void @llvm.xtensa.ae.s64.x(<1 x i64> {{.*}}, ptr {{.*}}, i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s64_x(ae_ls_v, ars, art);
}

//--- ae_s64_xc.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s64_xc(ae_int64 ae_ls_v,ae_int64** ars,int art) {
// CHECK-LABEL: test_ae_s64_xc
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.s64.xc(<1 x i64> {{.*}}, ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s64_xc(ae_ls_v, ars, art);
}

//--- ae_s64_xp.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_s64_xp(ae_int64 ae_ls_v,ae_int64** ars,int art) {
// CHECK-LABEL: test_ae_s64_xp
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.s64.xp(<1 x i64> {{.*}}, ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_s64_xp(ae_ls_v, ars, art);
}

//--- ae_sa16x4_ic.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_sa16x4_ic(ae_int16x4 ae_ls_v,ae_valign* ae_ls_su,ae_int16x4** ars) {
// CHECK-LABEL: test_ae_sa16x4_ic
// CHECK: %[[LD_AE_LS_SU:.*]] = load <8 x i8>, ptr {{.*}}
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <8 x i8>, ptr } @llvm.xtensa.ae.sa16x4.ic(<4 x i16> {{.*}}, <8 x i8> %[[LD_AE_LS_SU]], ptr %[[LD_ARS]])
// CHECK: %[[EV:.*]] = extractvalue { <8 x i8>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_sa16x4_ic(ae_ls_v, ae_ls_su, ars);
}

//--- ae_sa16x4_ip.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_sa16x4_ip(ae_int16x4 ae_ls_v,ae_valign* ae_ls_su,ae_int16x4** ars) {
// CHECK-LABEL: test_ae_sa16x4_ip
// CHECK: %[[LD_AE_LS_SU:.*]] = load <8 x i8>, ptr {{.*}}
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <8 x i8>, ptr } @llvm.xtensa.ae.sa16x4.ip(<4 x i16> {{.*}}, <8 x i8> %[[LD_AE_LS_SU]], ptr %[[LD_ARS]])
// CHECK: %[[EV:.*]] = extractvalue { <8 x i8>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_sa16x4_ip(ae_ls_v, ae_ls_su, ars);
}

//--- ae_sa16x4_ric.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_sa16x4_ric(ae_int16x4 ae_ls_v,ae_valign* ae_ls_su,ae_int16x4** ars) {
// CHECK-LABEL: test_ae_sa16x4_ric
// CHECK: %[[LD_AE_LS_SU:.*]] = load <8 x i8>, ptr {{.*}}
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <8 x i8>, ptr } @llvm.xtensa.ae.sa16x4.ric(<4 x i16> {{.*}}, <8 x i8> %[[LD_AE_LS_SU]], ptr %[[LD_ARS]])
// CHECK: %[[EV:.*]] = extractvalue { <8 x i8>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_sa16x4_ric(ae_ls_v, ae_ls_su, ars);
}

//--- ae_sa16x4_rip.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_sa16x4_rip(ae_int16x4 ae_ls_v,ae_valign* ae_ls_su,ae_int16x4** ars) {
// CHECK-LABEL: test_ae_sa16x4_rip
// CHECK: %[[LD_AE_LS_SU:.*]] = load <8 x i8>, ptr {{.*}}
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <8 x i8>, ptr } @llvm.xtensa.ae.sa16x4.rip(<4 x i16> {{.*}}, <8 x i8> %[[LD_AE_LS_SU]], ptr %[[LD_ARS]])
// CHECK: %[[EV:.*]] = extractvalue { <8 x i8>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_sa16x4_rip(ae_ls_v, ae_ls_su, ars);
}

//--- ae_sa24_l_ic.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_sa24_l_ic(ae_int32x2 ae_ls_v,ae_valign* ae_ls_su,void** ars) {
// CHECK-LABEL: test_ae_sa24_l_ic
// CHECK: %[[LD_AE_LS_SU:.*]] = load <8 x i8>, ptr {{.*}}
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <8 x i8>, ptr } @llvm.xtensa.ae.sa24.l.ic(<2 x i32> {{.*}}, <8 x i8> %[[LD_AE_LS_SU]], ptr %[[LD_ARS]])
// CHECK: %[[EV:.*]] = extractvalue { <8 x i8>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_sa24_l_ic(ae_ls_v, ae_ls_su, ars);
}

//--- ae_sa24_l_ip.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_sa24_l_ip(ae_int32x2 ae_ls_v,ae_valign* ae_ls_su,void** ars) {
// CHECK-LABEL: test_ae_sa24_l_ip
// CHECK: %[[LD_AE_LS_SU:.*]] = load <8 x i8>, ptr {{.*}}
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <8 x i8>, ptr } @llvm.xtensa.ae.sa24.l.ip(<2 x i32> {{.*}}, <8 x i8> %[[LD_AE_LS_SU]], ptr %[[LD_ARS]])
// CHECK: %[[EV:.*]] = extractvalue { <8 x i8>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_sa24_l_ip(ae_ls_v, ae_ls_su, ars);
}

//--- ae_sa24_l_ric.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_sa24_l_ric(ae_int32x2 ae_ls_v,ae_valign* ae_ls_su,void** ars) {
// CHECK-LABEL: test_ae_sa24_l_ric
// CHECK: %[[LD_AE_LS_SU:.*]] = load <8 x i8>, ptr {{.*}}
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <8 x i8>, ptr } @llvm.xtensa.ae.sa24.l.ric(<2 x i32> {{.*}}, <8 x i8> %[[LD_AE_LS_SU]], ptr %[[LD_ARS]])
// CHECK: %[[EV:.*]] = extractvalue { <8 x i8>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_sa24_l_ric(ae_ls_v, ae_ls_su, ars);
}

//--- ae_sa24_l_rip.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_sa24_l_rip(ae_int32x2 ae_ls_v,ae_valign* ae_ls_su,void** ars) {
// CHECK-LABEL: test_ae_sa24_l_rip
// CHECK: %[[LD_AE_LS_SU:.*]] = load <8 x i8>, ptr {{.*}}
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <8 x i8>, ptr } @llvm.xtensa.ae.sa24.l.rip(<2 x i32> {{.*}}, <8 x i8> %[[LD_AE_LS_SU]], ptr %[[LD_ARS]])
// CHECK: %[[EV:.*]] = extractvalue { <8 x i8>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_sa24_l_rip(ae_ls_v, ae_ls_su, ars);
}

//--- ae_sa24x2_ic.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_sa24x2_ic(ae_int32x2 ae_ls_v,ae_valign* ae_ls_su,void** ars) {
// CHECK-LABEL: test_ae_sa24x2_ic
// CHECK: %[[LD_AE_LS_SU:.*]] = load <8 x i8>, ptr {{.*}}
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <8 x i8>, ptr } @llvm.xtensa.ae.sa24x2.ic(<2 x i32> {{.*}}, <8 x i8> %[[LD_AE_LS_SU]], ptr %[[LD_ARS]])
// CHECK: %[[EV:.*]] = extractvalue { <8 x i8>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_sa24x2_ic(ae_ls_v, ae_ls_su, ars);
}

//--- ae_sa24x2_ip.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_sa24x2_ip(ae_int32x2 ae_ls_v,ae_valign* ae_ls_su,void** ars) {
// CHECK-LABEL: test_ae_sa24x2_ip
// CHECK: %[[LD_AE_LS_SU:.*]] = load <8 x i8>, ptr {{.*}}
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <8 x i8>, ptr } @llvm.xtensa.ae.sa24x2.ip(<2 x i32> {{.*}}, <8 x i8> %[[LD_AE_LS_SU]], ptr %[[LD_ARS]])
// CHECK: %[[EV:.*]] = extractvalue { <8 x i8>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_sa24x2_ip(ae_ls_v, ae_ls_su, ars);
}

//--- ae_sa24x2_ric.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_sa24x2_ric(ae_int32x2 ae_ls_v,ae_valign* ae_ls_su,void** ars) {
// CHECK-LABEL: test_ae_sa24x2_ric
// CHECK: %[[LD_AE_LS_SU:.*]] = load <8 x i8>, ptr {{.*}}
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <8 x i8>, ptr } @llvm.xtensa.ae.sa24x2.ric(<2 x i32> {{.*}}, <8 x i8> %[[LD_AE_LS_SU]], ptr %[[LD_ARS]])
// CHECK: %[[EV:.*]] = extractvalue { <8 x i8>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_sa24x2_ric(ae_ls_v, ae_ls_su, ars);
}

//--- ae_sa24x2_rip.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_sa24x2_rip(ae_int32x2 ae_ls_v,ae_valign* ae_ls_su,void** ars) {
// CHECK-LABEL: test_ae_sa24x2_rip
// CHECK: %[[LD_AE_LS_SU:.*]] = load <8 x i8>, ptr {{.*}}
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <8 x i8>, ptr } @llvm.xtensa.ae.sa24x2.rip(<2 x i32> {{.*}}, <8 x i8> %[[LD_AE_LS_SU]], ptr %[[LD_ARS]])
// CHECK: %[[EV:.*]] = extractvalue { <8 x i8>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_sa24x2_rip(ae_ls_v, ae_ls_su, ars);
}

//--- ae_sa32x2_ic.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_sa32x2_ic(ae_int32x2 ae_ls_v,ae_valign* ae_ls_su,ae_int32x2** ars) {
// CHECK-LABEL: test_ae_sa32x2_ic
// CHECK: %[[LD_AE_LS_SU:.*]] = load <8 x i8>, ptr {{.*}}
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <8 x i8>, ptr } @llvm.xtensa.ae.sa32x2.ic(<2 x i32> {{.*}}, <8 x i8> %[[LD_AE_LS_SU]], ptr %[[LD_ARS]])
// CHECK: %[[EV:.*]] = extractvalue { <8 x i8>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_sa32x2_ic(ae_ls_v, ae_ls_su, ars);
}

//--- ae_sa32x2_ip.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_sa32x2_ip(ae_int32x2 ae_ls_v,ae_valign* ae_ls_su,ae_int32x2** ars) {
// CHECK-LABEL: test_ae_sa32x2_ip
// CHECK: %[[LD_AE_LS_SU:.*]] = load <8 x i8>, ptr {{.*}}
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <8 x i8>, ptr } @llvm.xtensa.ae.sa32x2.ip(<2 x i32> {{.*}}, <8 x i8> %[[LD_AE_LS_SU]], ptr %[[LD_ARS]])
// CHECK: %[[EV:.*]] = extractvalue { <8 x i8>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_sa32x2_ip(ae_ls_v, ae_ls_su, ars);
}

//--- ae_sa32x2_ric.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_sa32x2_ric(ae_int32x2 ae_ls_v,ae_valign* ae_ls_su,ae_int32x2** ars) {
// CHECK-LABEL: test_ae_sa32x2_ric
// CHECK: %[[LD_AE_LS_SU:.*]] = load <8 x i8>, ptr {{.*}}
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <8 x i8>, ptr } @llvm.xtensa.ae.sa32x2.ric(<2 x i32> {{.*}}, <8 x i8> %[[LD_AE_LS_SU]], ptr %[[LD_ARS]])
// CHECK: %[[EV:.*]] = extractvalue { <8 x i8>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_sa32x2_ric(ae_ls_v, ae_ls_su, ars);
}

//--- ae_sa32x2_rip.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_sa32x2_rip(ae_int32x2 ae_ls_v,ae_valign* ae_ls_su,ae_int32x2** ars) {
// CHECK-LABEL: test_ae_sa32x2_rip
// CHECK: %[[LD_AE_LS_SU:.*]] = load <8 x i8>, ptr {{.*}}
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <8 x i8>, ptr } @llvm.xtensa.ae.sa32x2.rip(<2 x i32> {{.*}}, <8 x i8> %[[LD_AE_LS_SU]], ptr %[[LD_ARS]])
// CHECK: %[[EV:.*]] = extractvalue { <8 x i8>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_sa32x2_rip(ae_ls_v, ae_ls_su, ars);
}

//--- ae_sa32x2f24_ic.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_sa32x2f24_ic(ae_int32x2 ae_ls_v,ae_valign* ae_ls_su,ae_int32x2** ars) {
// CHECK-LABEL: test_ae_sa32x2f24_ic
// CHECK: %[[LD_AE_LS_SU:.*]] = load <8 x i8>, ptr {{.*}}
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <8 x i8>, ptr } @llvm.xtensa.ae.sa32x2f24.ic(<2 x i32> {{.*}}, <8 x i8> %[[LD_AE_LS_SU]], ptr %[[LD_ARS]])
// CHECK: %[[EV:.*]] = extractvalue { <8 x i8>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_sa32x2f24_ic(ae_ls_v, ae_ls_su, ars);
}

//--- ae_sa32x2f24_ip.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_sa32x2f24_ip(ae_int32x2 ae_ls_v,ae_valign* ae_ls_su,ae_int32x2** ars) {
// CHECK-LABEL: test_ae_sa32x2f24_ip
// CHECK: %[[LD_AE_LS_SU:.*]] = load <8 x i8>, ptr {{.*}}
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <8 x i8>, ptr } @llvm.xtensa.ae.sa32x2f24.ip(<2 x i32> {{.*}}, <8 x i8> %[[LD_AE_LS_SU]], ptr %[[LD_ARS]])
// CHECK: %[[EV:.*]] = extractvalue { <8 x i8>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_sa32x2f24_ip(ae_ls_v, ae_ls_su, ars);
}

//--- ae_sa32x2f24_ric.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_sa32x2f24_ric(ae_int32x2 ae_ls_v,ae_valign* ae_ls_su,ae_int32x2** ars) {
// CHECK-LABEL: test_ae_sa32x2f24_ric
// CHECK: %[[LD_AE_LS_SU:.*]] = load <8 x i8>, ptr {{.*}}
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <8 x i8>, ptr } @llvm.xtensa.ae.sa32x2f24.ric(<2 x i32> {{.*}}, <8 x i8> %[[LD_AE_LS_SU]], ptr %[[LD_ARS]])
// CHECK: %[[EV:.*]] = extractvalue { <8 x i8>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_sa32x2f24_ric(ae_ls_v, ae_ls_su, ars);
}

//--- ae_sa32x2f24_rip.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_sa32x2f24_rip(ae_int32x2 ae_ls_v,ae_valign* ae_ls_su,ae_int32x2** ars) {
// CHECK-LABEL: test_ae_sa32x2f24_rip
// CHECK: %[[LD_AE_LS_SU:.*]] = load <8 x i8>, ptr {{.*}}
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <8 x i8>, ptr } @llvm.xtensa.ae.sa32x2f24.rip(<2 x i32> {{.*}}, <8 x i8> %[[LD_AE_LS_SU]], ptr %[[LD_ARS]])
// CHECK: %[[EV:.*]] = extractvalue { <8 x i8>, ptr } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_sa32x2f24_rip(ae_ls_v, ae_ls_su, ars);
}

//--- ae_sa64neg_fp.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_sa64neg_fp(ae_valign* ae_ls_su,void* ars) {
// CHECK-LABEL: test_ae_sa64neg_fp
// CHECK: %[[LD_AE_LS_SU:.*]] = load <8 x i8>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <8 x i8> @llvm.xtensa.ae.sa64neg.fp(<8 x i8> %[[LD_AE_LS_SU]], ptr {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_sa64neg_fp(ae_ls_su, ars);
}

//--- ae_sa64pos_fp.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_sa64pos_fp(ae_valign* ae_ls_su,void* ars) {
// CHECK-LABEL: test_ae_sa64pos_fp
// CHECK: %[[LD_AE_LS_SU:.*]] = load <8 x i8>, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call <8 x i8> @llvm.xtensa.ae.sa64pos.fp(<8 x i8> %[[LD_AE_LS_SU]], ptr {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_sa64pos_fp(ae_ls_su, ars);
}

//--- ae_salign64_i.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_salign64_i(ae_valign ae_ls_su,ae_valign* ars,immediate ae_immls64) {
// CHECK-LABEL: test_ae_salign64_i
// CHECK:  {{(tail)?}} call void @llvm.xtensa.ae.salign64.i(<8 x i8> {{.*}}, ptr {{.*}}, i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_salign64_i(ae_ls_su, ars, -64);
}

//--- ae_sat16x4.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int16x4 test_ae_sat16x4(ae_int32x2 ae_arth_v0,ae_int32x2 ae_arth_v1) {
// CHECK-LABEL: test_ae_sat16x4
// CHECK: %[[RET:.*]] = {{(tail)?}} call <4 x i16> @llvm.xtensa.ae.sat16x4(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <4 x i16> %[[RET]]
return __builtin_xtensa_ae_sat16x4(ae_arth_v0, ae_arth_v1);
}

//--- ae_sat24s.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_sat24s(ae_int32x2 ae_arth_v1) {
// CHECK-LABEL: test_ae_sat24s
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.sat24s(<2 x i32> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_sat24s(ae_arth_v1);
}

//--- ae_sat48s.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_sat48s(ae_int64 ae_arth_v1) {
// CHECK-LABEL: test_ae_sat48s
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.sat48s(<1 x i64> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_sat48s(ae_arth_v1);
}

//--- ae_satq56s.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_satq56s(ae_int64 ae_arth_v1) {
// CHECK-LABEL: test_ae_satq56s
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.satq56s(<1 x i64> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_satq56s(ae_arth_v1);
}

//--- ae_sb.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_sb(short** ars,int art) {
// CHECK-LABEL: test_ae_sb
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.sb(ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_sb(ars, art);
}

//--- ae_sb_ic.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_sb_ic(short** ars,int art) {
// CHECK-LABEL: test_ae_sb_ic
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.sb.ic(ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_sb_ic(ars, art);
}

//--- ae_sb_ip.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_sb_ip(short** ars,int art) {
// CHECK-LABEL: test_ae_sb_ip
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.sb.ip(ptr %[[LD_ARS]], i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_sb_ip(ars, art);
}

//--- ae_sbf.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_sbf(short** ars) {
// CHECK-LABEL: test_ae_sbf
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.sbf(ptr %[[LD_ARS]])
// CHECK: ret void
 __builtin_xtensa_ae_sbf(ars);
}

//--- ae_sbf_ic.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_sbf_ic(short** ars) {
// CHECK-LABEL: test_ae_sbf_ic
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.sbf.ic(ptr %[[LD_ARS]])
// CHECK: ret void
 __builtin_xtensa_ae_sbf_ic(ars);
}

//--- ae_sbf_ip.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_sbf_ip(short** ars) {
// CHECK-LABEL: test_ae_sbf_ip
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.sbf.ip(ptr %[[LD_ARS]])
// CHECK: ret void
 __builtin_xtensa_ae_sbf_ip(ars);
}

//--- ae_sbi.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_sbi(short** ars,int art,immediate ae_ohba2) {
// CHECK-LABEL: test_ae_sbi
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.sbi(ptr %[[LD_ARS]], i32 {{.*}}, i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_sbi(ars, art, 1);
}

//--- ae_sbi_ic.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_sbi_ic(short** ars,int art,immediate ae_ohba2) {
// CHECK-LABEL: test_ae_sbi_ic
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.sbi.ic(ptr %[[LD_ARS]], i32 {{.*}}, i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_sbi_ic(ars, art, 1);
}

//--- ae_sbi_ip.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_sbi_ip(short** ars,int art,immediate ae_ohba2) {
// CHECK-LABEL: test_ae_sbi_ip
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.sbi.ip(ptr %[[LD_ARS]], i32 {{.*}}, i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_sbi_ip(ars, art, 1);
}

//--- ae_sel16i.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int16x4 test_ae_sel16i(ae_int16x4 ae_dr_to_dr_v0,ae_int16x4 ae_dr_to_dr_v1,immediate ae_selimm) {
// CHECK-LABEL: test_ae_sel16i
// CHECK: %[[RET:.*]] = {{(tail)?}} call <4 x i16> @llvm.xtensa.ae.sel16i(<4 x i16> {{.*}}, <4 x i16> {{.*}}, i32 {{.*}})
// CHECK: ret <4 x i16> %[[RET]]
return __builtin_xtensa_ae_sel16i(ae_dr_to_dr_v0, ae_dr_to_dr_v1, 0);
}

//--- ae_sel16i_n.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int16x4 test_ae_sel16i_n(ae_int16x4 ae_dr_to_dr_v0,ae_int16x4 ae_dr_to_dr_v1,immediate ae_selimm_N) {
// CHECK-LABEL: test_ae_sel16i_n
// CHECK: %[[RET:.*]] = {{(tail)?}} call <4 x i16> @llvm.xtensa.ae.sel16i.n(<4 x i16> {{.*}}, <4 x i16> {{.*}}, i32 {{.*}})
// CHECK: ret <4 x i16> %[[RET]]
return __builtin_xtensa_ae_sel16i_n(ae_dr_to_dr_v0, ae_dr_to_dr_v1, 0);
}

//--- ae_sext32.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_sext32(ae_int32x2 ae_dr_to_dr_v0,immediate ae_opnd_tp7) {
// CHECK-LABEL: test_ae_sext32
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.sext32(<2 x i32> {{.*}}, i32 {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_sext32(ae_dr_to_dr_v0, 7);
}

//--- ae_sext32x2d16_10.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_sext32x2d16_10(ae_int16x4 ae_to_dr_v0) {
// CHECK-LABEL: test_ae_sext32x2d16_10
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.sext32x2d16.10(<4 x i16> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_sext32x2d16_10(ae_to_dr_v0);
}

//--- ae_sext32x2d16_32.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_sext32x2d16_32(ae_int16x4 ae_to_dr_v0) {
// CHECK-LABEL: test_ae_sext32x2d16_32
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.sext32x2d16.32(<4 x i16> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_sext32x2d16_32(ae_to_dr_v0);
}

//--- ae_sha32.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

int test_ae_sha32(int ars) {
// CHECK-LABEL: test_ae_sha32
// CHECK: %[[RET:.*]] = {{(tail)?}} call i32 @llvm.xtensa.ae.sha32(i32 {{.*}})
// CHECK: ret i32 %[[RET]]
return __builtin_xtensa_ae_sha32(ars);
}

//--- ae_shortswap.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int16x4 test_ae_shortswap(ae_int16x4 ae_to_dr_v0) {
// CHECK-LABEL: test_ae_shortswap
// CHECK: %[[RET:.*]] = {{(tail)?}} call <4 x i16> @llvm.xtensa.ae.shortswap(<4 x i16> {{.*}})
// CHECK: ret <4 x i16> %[[RET]]
return __builtin_xtensa_ae_shortswap(ae_to_dr_v0);
}

//--- ae_slaa16s.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int16x4 test_ae_slaa16s(ae_int16x4 ae_shift_d0,int ars) {
// CHECK-LABEL: test_ae_slaa16s
// CHECK: %[[RET:.*]] = {{(tail)?}} call <4 x i16> @llvm.xtensa.ae.slaa16s(<4 x i16> {{.*}}, i32 {{.*}})
// CHECK: ret <4 x i16> %[[RET]]
return __builtin_xtensa_ae_slaa16s(ae_shift_d0, ars);
}

//--- ae_slaa32.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_slaa32(ae_int32x2 ae_shift_d0,int ars) {
// CHECK-LABEL: test_ae_slaa32
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.slaa32(<2 x i32> {{.*}}, i32 {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_slaa32(ae_shift_d0, ars);
}

//--- ae_slaa32s.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_slaa32s(ae_int32x2 ae_shift_d0,int ars) {
// CHECK-LABEL: test_ae_slaa32s
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.slaa32s(<2 x i32> {{.*}}, i32 {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_slaa32s(ae_shift_d0, ars);
}

//--- ae_slaa64.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_slaa64(ae_int64 ae_shift_d0,int ars) {
// CHECK-LABEL: test_ae_slaa64
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.slaa64(<1 x i64> {{.*}}, i32 {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_slaa64(ae_shift_d0, ars);
}

//--- ae_slaa64s.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_slaa64s(ae_int64 ae_shift_d0,int ars) {
// CHECK-LABEL: test_ae_slaa64s
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.slaa64s(<1 x i64> {{.*}}, i32 {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_slaa64s(ae_shift_d0, ars);
}

//--- ae_slaaq56.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_slaaq56(ae_int64 ae_shift_d0,int ars) {
// CHECK-LABEL: test_ae_slaaq56
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.slaaq56(<1 x i64> {{.*}}, i32 {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_slaaq56(ae_shift_d0, ars);
}

//--- ae_slai16s.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int16x4 test_ae_slai16s(ae_int16x4 ae_shift_d0,immediate ae_osa16) {
// CHECK-LABEL: test_ae_slai16s
// CHECK: %[[RET:.*]] = {{(tail)?}} call <4 x i16> @llvm.xtensa.ae.slai16s(<4 x i16> {{.*}}, i32 {{.*}})
// CHECK: ret <4 x i16> %[[RET]]
return __builtin_xtensa_ae_slai16s(ae_shift_d0, 0);
}

//--- ae_slai24.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_slai24(ae_int32x2 ae_shift_d0,immediate ae_osa32) {
// CHECK-LABEL: test_ae_slai24
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.slai24(<2 x i32> {{.*}}, i32 {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_slai24(ae_shift_d0, 0);
}

//--- ae_slai24s.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_slai24s(ae_int32x2 ae_shift_d0,immediate ae_osa32) {
// CHECK-LABEL: test_ae_slai24s
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.slai24s(<2 x i32> {{.*}}, i32 {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_slai24s(ae_shift_d0, 0);
}

//--- ae_slai32.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_slai32(ae_int32x2 ae_shift_d0,immediate ae_osa32) {
// CHECK-LABEL: test_ae_slai32
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.slai32(<2 x i32> {{.*}}, i32 {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_slai32(ae_shift_d0, 0);
}

//--- ae_slai32s.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_slai32s(ae_int32x2 ae_shift_d0,immediate ae_osa32) {
// CHECK-LABEL: test_ae_slai32s
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.slai32s(<2 x i32> {{.*}}, i32 {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_slai32s(ae_shift_d0, 0);
}

//--- ae_slai64.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_slai64(ae_int64 ae_shift_d0,immediate ae_osa64) {
// CHECK-LABEL: test_ae_slai64
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.slai64(<1 x i64> {{.*}}, i32 {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_slai64(ae_shift_d0, 0);
}

//--- ae_slai64s.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_slai64s(ae_int64 ae_shift_d0,immediate ae_osa64) {
// CHECK-LABEL: test_ae_slai64s
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.slai64s(<1 x i64> {{.*}}, i32 {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_slai64s(ae_shift_d0, 0);
}

//--- ae_slaisq56s.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_slaisq56s(ae_int64 ae_shift_d0,immediate ae_osa64) {
// CHECK-LABEL: test_ae_slaisq56s
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.slaisq56s(<1 x i64> {{.*}}, i32 {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_slaisq56s(ae_shift_d0, 0);
}

//--- ae_slas24.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_slas24(ae_int32x2 ae_shift_d0) {
// CHECK-LABEL: test_ae_slas24
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.slas24(<2 x i32> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_slas24(ae_shift_d0);
}

//--- ae_slas24s.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_slas24s(ae_int32x2 ae_shift_d0) {
// CHECK-LABEL: test_ae_slas24s
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.slas24s(<2 x i32> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_slas24s(ae_shift_d0);
}

//--- ae_slas32.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_slas32(ae_int32x2 ae_shift_d0) {
// CHECK-LABEL: test_ae_slas32
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.slas32(<2 x i32> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_slas32(ae_shift_d0);
}

//--- ae_slas32s.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_slas32s(ae_int32x2 ae_shift_d0) {
// CHECK-LABEL: test_ae_slas32s
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.slas32s(<2 x i32> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_slas32s(ae_shift_d0);
}

//--- ae_slas64.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_slas64(ae_int64 ae_shift_d0) {
// CHECK-LABEL: test_ae_slas64
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.slas64(<1 x i64> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_slas64(ae_shift_d0);
}

//--- ae_slas64s.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_slas64s(ae_int64 ae_shift_d0) {
// CHECK-LABEL: test_ae_slas64s
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.slas64s(<1 x i64> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_slas64s(ae_shift_d0);
}

//--- ae_slasq56.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_slasq56(ae_int64 ae_shift_d0) {
// CHECK-LABEL: test_ae_slasq56
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.slasq56(<1 x i64> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_slasq56(ae_shift_d0);
}

//--- ae_slassq56s.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_slassq56s(ae_int64 ae_shift_d0) {
// CHECK-LABEL: test_ae_slassq56s
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.slassq56s(<1 x i64> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_slassq56s(ae_shift_d0);
}

//--- ae_sra64_32.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_sra64_32(ae_int32x2 ae_shift_d0,int ars) {
// CHECK-LABEL: test_ae_sra64_32
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.sra64.32(<2 x i32> {{.*}}, i32 {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_sra64_32(ae_shift_d0, ars);
}

//--- ae_sraa16rs.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int16x4 test_ae_sraa16rs(ae_int16x4 ae_shift_d0,int ars) {
// CHECK-LABEL: test_ae_sraa16rs
// CHECK: %[[RET:.*]] = {{(tail)?}} call <4 x i16> @llvm.xtensa.ae.sraa16rs(<4 x i16> {{.*}}, i32 {{.*}})
// CHECK: ret <4 x i16> %[[RET]]
return __builtin_xtensa_ae_sraa16rs(ae_shift_d0, ars);
}

//--- ae_sraa16s.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int16x4 test_ae_sraa16s(ae_int16x4 ae_shift_d0,int ars) {
// CHECK-LABEL: test_ae_sraa16s
// CHECK: %[[RET:.*]] = {{(tail)?}} call <4 x i16> @llvm.xtensa.ae.sraa16s(<4 x i16> {{.*}}, i32 {{.*}})
// CHECK: ret <4 x i16> %[[RET]]
return __builtin_xtensa_ae_sraa16s(ae_shift_d0, ars);
}

//--- ae_sraa32.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_sraa32(ae_int32x2 ae_shift_d0,int ars) {
// CHECK-LABEL: test_ae_sraa32
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.sraa32(<2 x i32> {{.*}}, i32 {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_sraa32(ae_shift_d0, ars);
}

//--- ae_sraa32rs.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_sraa32rs(ae_int32x2 ae_shift_d0,int ars) {
// CHECK-LABEL: test_ae_sraa32rs
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.sraa32rs(<2 x i32> {{.*}}, i32 {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_sraa32rs(ae_shift_d0, ars);
}

//--- ae_sraa32s.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_sraa32s(ae_int32x2 ae_shift_d0,int ars) {
// CHECK-LABEL: test_ae_sraa32s
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.sraa32s(<2 x i32> {{.*}}, i32 {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_sraa32s(ae_shift_d0, ars);
}

//--- ae_sraa64.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_sraa64(ae_int64 ae_shift_d0,int ars) {
// CHECK-LABEL: test_ae_sraa64
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.sraa64(<1 x i64> {{.*}}, i32 {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_sraa64(ae_shift_d0, ars);
}

//--- ae_srai16.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int16x4 test_ae_srai16(ae_int16x4 ae_shift_d0,immediate ae_osa16) {
// CHECK-LABEL: test_ae_srai16
// CHECK: %[[RET:.*]] = {{(tail)?}} call <4 x i16> @llvm.xtensa.ae.srai16(<4 x i16> {{.*}}, i32 {{.*}})
// CHECK: ret <4 x i16> %[[RET]]
return __builtin_xtensa_ae_srai16(ae_shift_d0, 0);
}

//--- ae_srai16r.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int16x4 test_ae_srai16r(ae_int16x4 ae_shift_d0,immediate ae_osa16) {
// CHECK-LABEL: test_ae_srai16r
// CHECK: %[[RET:.*]] = {{(tail)?}} call <4 x i16> @llvm.xtensa.ae.srai16r(<4 x i16> {{.*}}, i32 {{.*}})
// CHECK: ret <4 x i16> %[[RET]]
return __builtin_xtensa_ae_srai16r(ae_shift_d0, 0);
}

//--- ae_srai24.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_srai24(ae_int32x2 ae_shift_d0,immediate ae_osa32) {
// CHECK-LABEL: test_ae_srai24
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.srai24(<2 x i32> {{.*}}, i32 {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_srai24(ae_shift_d0, 0);
}

//--- ae_srai32.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_srai32(ae_int32x2 ae_shift_d0,immediate ae_osa32) {
// CHECK-LABEL: test_ae_srai32
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.srai32(<2 x i32> {{.*}}, i32 {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_srai32(ae_shift_d0, 0);
}

//--- ae_srai32r.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_srai32r(ae_int32x2 ae_shift_d0,immediate ae_osa32) {
// CHECK-LABEL: test_ae_srai32r
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.srai32r(<2 x i32> {{.*}}, i32 {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_srai32r(ae_shift_d0, 0);
}

//--- ae_srai64.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_srai64(ae_int64 ae_shift_d0,immediate ae_osa64) {
// CHECK-LABEL: test_ae_srai64
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.srai64(<1 x i64> {{.*}}, i32 {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_srai64(ae_shift_d0, 0);
}

//--- ae_sras24.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_sras24(ae_int32x2 ae_shift_d0) {
// CHECK-LABEL: test_ae_sras24
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.sras24(<2 x i32> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_sras24(ae_shift_d0);
}

//--- ae_sras32.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_sras32(ae_int32x2 ae_shift_d0) {
// CHECK-LABEL: test_ae_sras32
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.sras32(<2 x i32> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_sras32(ae_shift_d0);
}

//--- ae_sras64.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_sras64(ae_int64 ae_shift_d0) {
// CHECK-LABEL: test_ae_sras64
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.sras64(<1 x i64> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_sras64(ae_shift_d0);
}

//--- ae_srla32.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_srla32(ae_int32x2 ae_shift_d0,int ars) {
// CHECK-LABEL: test_ae_srla32
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.srla32(<2 x i32> {{.*}}, i32 {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_srla32(ae_shift_d0, ars);
}

//--- ae_srla64.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_srla64(ae_int64 ae_shift_d0,int ars) {
// CHECK-LABEL: test_ae_srla64
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.srla64(<1 x i64> {{.*}}, i32 {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_srla64(ae_shift_d0, ars);
}

//--- ae_srli24.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_srli24(ae_int32x2 ae_shift_d0,immediate ae_osa32) {
// CHECK-LABEL: test_ae_srli24
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.srli24(<2 x i32> {{.*}}, i32 {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_srli24(ae_shift_d0, 0);
}

//--- ae_srli32.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_srli32(ae_int32x2 ae_shift_d0,immediate ae_osa32) {
// CHECK-LABEL: test_ae_srli32
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.srli32(<2 x i32> {{.*}}, i32 {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_srli32(ae_shift_d0, 0);
}

//--- ae_srli64.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_srli64(ae_int64 ae_shift_d0,immediate ae_osa64) {
// CHECK-LABEL: test_ae_srli64
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.srli64(<1 x i64> {{.*}}, i32 {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_srli64(ae_shift_d0, 0);
}

//--- ae_srls24.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_srls24(ae_int32x2 ae_shift_d0) {
// CHECK-LABEL: test_ae_srls24
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.srls24(<2 x i32> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_srls24(ae_shift_d0);
}

//--- ae_srls32.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_srls32(ae_int32x2 ae_shift_d0) {
// CHECK-LABEL: test_ae_srls32
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.srls32(<2 x i32> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_srls32(ae_shift_d0);
}

//--- ae_srls64.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_srls64(ae_int64 ae_shift_d0) {
// CHECK-LABEL: test_ae_srls64
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.srls64(<1 x i64> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_srls64(ae_shift_d0);
}

//--- ae_sub16.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int16x4 test_ae_sub16(ae_int16x4 ae_arth_v0,ae_int16x4 ae_arth_v1) {
// CHECK-LABEL: test_ae_sub16
// CHECK: %[[RET:.*]] = {{(tail)?}} call <4 x i16> @llvm.xtensa.ae.sub16(<4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <4 x i16> %[[RET]]
return __builtin_xtensa_ae_sub16(ae_arth_v0, ae_arth_v1);
}

//--- ae_sub16s.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int16x4 test_ae_sub16s(ae_int16x4 ae_arth_v0,ae_int16x4 ae_arth_v1) {
// CHECK-LABEL: test_ae_sub16s
// CHECK: %[[RET:.*]] = {{(tail)?}} call <4 x i16> @llvm.xtensa.ae.sub16s(<4 x i16> {{.*}}, <4 x i16> {{.*}})
// CHECK: ret <4 x i16> %[[RET]]
return __builtin_xtensa_ae_sub16s(ae_arth_v0, ae_arth_v1);
}

//--- ae_sub24s.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_sub24s(ae_int32x2 ae_arth_v0,ae_int32x2 ae_arth_v1) {
// CHECK-LABEL: test_ae_sub24s
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.sub24s(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_sub24s(ae_arth_v0, ae_arth_v1);
}

//--- ae_sub32.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_sub32(ae_int32x2 ae_arth_v0,ae_int32x2 ae_arth_v1) {
// CHECK-LABEL: test_ae_sub32
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.sub32(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_sub32(ae_arth_v0, ae_arth_v1);
}

//--- ae_sub32s.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_sub32s(ae_int32x2 ae_arth_v0,ae_int32x2 ae_arth_v1) {
// CHECK-LABEL: test_ae_sub32s
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.sub32s(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_sub32s(ae_arth_v0, ae_arth_v1);
}

//--- ae_sub64.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_sub64(ae_int64 ae_arth_v0,ae_int64 ae_arth_v1) {
// CHECK-LABEL: test_ae_sub64
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.sub64(<1 x i64> {{.*}}, <1 x i64> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_sub64(ae_arth_v0, ae_arth_v1);
}

//--- ae_sub64s.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_sub64s(ae_int64 ae_arth_v0,ae_int64 ae_arth_v1) {
// CHECK-LABEL: test_ae_sub64s
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.sub64s(<1 x i64> {{.*}}, <1 x i64> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_sub64s(ae_arth_v0, ae_arth_v1);
}

//--- ae_subadd32.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_subadd32(ae_int32x2 ae_arth_v0,ae_int32x2 ae_arth_v1) {
// CHECK-LABEL: test_ae_subadd32
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.subadd32(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_subadd32(ae_arth_v0, ae_arth_v1);
}

//--- ae_subadd32s.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_subadd32s(ae_int32x2 ae_arth_v0,ae_int32x2 ae_arth_v1) {
// CHECK-LABEL: test_ae_subadd32s
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.subadd32s(<2 x i32> {{.*}}, <2 x i32> {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_subadd32s(ae_arth_v0, ae_arth_v1);
}

//--- ae_trunca32f64s_l.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_trunca32f64s_l(ae_int32x2 ae_shift_d0,ae_int64 ae_shift_sd,int ars) {
// CHECK-LABEL: test_ae_trunca32f64s_l
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.trunca32f64s.l(<2 x i32> {{.*}}, <1 x i64> {{.*}}, i32 {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_trunca32f64s_l(ae_shift_d0, ae_shift_sd, ars);
}

//--- ae_trunca32x2f64s.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_trunca32x2f64s(ae_int64 ae_shift_d0,ae_int64 ae_shift_sd,int ars) {
// CHECK-LABEL: test_ae_trunca32x2f64s
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.trunca32x2f64s(<1 x i64> {{.*}}, <1 x i64> {{.*}}, i32 {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_trunca32x2f64s(ae_shift_d0, ae_shift_sd, ars);
}

//--- ae_trunci32f64s_l.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_trunci32f64s_l(ae_int32x2 ae_shift_d0,ae_int64 ae_shift_sd,immediate ae_osa16) {
// CHECK-LABEL: test_ae_trunci32f64s_l
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.trunci32f64s.l(<2 x i32> {{.*}}, <1 x i64> {{.*}}, i32 {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_trunci32f64s_l(ae_shift_d0, ae_shift_sd, 0);
}

//--- ae_trunci32x2f64s.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int32x2 test_ae_trunci32x2f64s(ae_int64 ae_shift_d0,ae_int64 ae_shift_sd,immediate ae_osa16) {
// CHECK-LABEL: test_ae_trunci32x2f64s
// CHECK: %[[RET:.*]] = {{(tail)?}} call <2 x i32> @llvm.xtensa.ae.trunci32x2f64s(<1 x i64> {{.*}}, <1 x i64> {{.*}}, i32 {{.*}})
// CHECK: ret <2 x i32> %[[RET]]
return __builtin_xtensa_ae_trunci32x2f64s(ae_shift_d0, ae_shift_sd, 0);
}

//--- ae_vldl16c.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_vldl16c(const short** ars) {
// CHECK-LABEL: test_ae_vldl16c
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.vldl16c(ptr %[[LD_ARS]])
// CHECK: ret void
 __builtin_xtensa_ae_vldl16c(ars);
}

//--- ae_vldl16c_ic.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_vldl16c_ic(const short** ars) {
// CHECK-LABEL: test_ae_vldl16c_ic
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.vldl16c.ic(ptr %[[LD_ARS]])
// CHECK: ret void
 __builtin_xtensa_ae_vldl16c_ic(ars);
}

//--- ae_vldl16c_ip.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_vldl16c_ip(const short** ars) {
// CHECK-LABEL: test_ae_vldl16c_ip
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.vldl16c.ip(ptr %[[LD_ARS]])
// CHECK: ret void
 __builtin_xtensa_ae_vldl16c_ip(ars);
}

//--- ae_vldl16t.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_vldl16t(xtbool* br,int* art,const short* ars) {
// CHECK-LABEL: test_ae_vldl16t
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <1 x i1>, i32 } @llvm.xtensa.ae.vldl16t(ptr {{.*}})
// CHECK: %[[EV:.*]] = extractvalue { <1 x i1>, i32 } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_vldl16t(br, art, ars);
}

//--- ae_vldl32t.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_vldl32t(xtbool* br,int* art,const int* ars) {
// CHECK-LABEL: test_ae_vldl32t
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <1 x i1>, i32 } @llvm.xtensa.ae.vldl32t(ptr {{.*}})
// CHECK: %[[EV:.*]] = extractvalue { <1 x i1>, i32 } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_vldl32t(br, art, ars);
}

//--- ae_vldsht.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_vldsht(int art) {
// CHECK-LABEL: test_ae_vldsht
// CHECK:  {{(tail)?}} call void @llvm.xtensa.ae.vldsht(i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_ae_vldsht(art);
}

//--- ae_vlel16t.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_vlel16t(xtbool* br,int* art,const short* ars) {
// CHECK-LABEL: test_ae_vlel16t
// CHECK: %[[LD_ART:.*]] = load i32, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <1 x i1>, i32 } @llvm.xtensa.ae.vlel16t(i32 %[[LD_ART]], ptr {{.*}})
// CHECK: %[[EV:.*]] = extractvalue { <1 x i1>, i32 } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_vlel16t(br, art, ars);
}

//--- ae_vlel32t.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_vlel32t(xtbool* br,int* art,const int* ars) {
// CHECK-LABEL: test_ae_vlel32t
// CHECK: %[[LD_ART:.*]] = load i32, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call { <1 x i1>, i32 } @llvm.xtensa.ae.vlel32t(i32 %[[LD_ART]], ptr {{.*}})
// CHECK: %[[EV:.*]] = extractvalue { <1 x i1>, i32 } %[[RET]], 0
// CHECK: ret void
 __builtin_xtensa_ae_vlel32t(br, art, ars);
}

//--- ae_vles16c.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_vles16c(short** ars) {
// CHECK-LABEL: test_ae_vles16c
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.vles16c(ptr %[[LD_ARS]])
// CHECK: ret void
 __builtin_xtensa_ae_vles16c(ars);
}

//--- ae_vles16c_ic.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_vles16c_ic(short** ars) {
// CHECK-LABEL: test_ae_vles16c_ic
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.vles16c.ic(ptr %[[LD_ARS]])
// CHECK: ret void
 __builtin_xtensa_ae_vles16c_ic(ars);
}

//--- ae_vles16c_ip.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_ae_vles16c_ip(short** ars) {
// CHECK-LABEL: test_ae_vles16c_ip
// CHECK: %[[LD_ARS:.*]] = load ptr, ptr {{.*}}
// CHECK: %[[RET:.*]] = {{(tail)?}} call ptr @llvm.xtensa.ae.vles16c.ip(ptr %[[LD_ARS]])
// CHECK: ret void
 __builtin_xtensa_ae_vles16c_ip(ars);
}

//--- ae_xor.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_int64 test_ae_xor(ae_int64 ae_dr_to_dr_v0,ae_int64 ae_dr_to_dr_v1) {
// CHECK-LABEL: test_ae_xor
// CHECK: %[[RET:.*]] = {{(tail)?}} call <1 x i64> @llvm.xtensa.ae.xor(<1 x i64> {{.*}}, <1 x i64> {{.*}})
// CHECK: ret <1 x i64> %[[RET]]
return __builtin_xtensa_ae_xor(ae_dr_to_dr_v0, ae_dr_to_dr_v1);
}

//--- ae_zalign64.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

ae_valign test_ae_zalign64() {
// CHECK-LABEL: test_ae_zalign64
// CHECK: %[[RET:.*]] = {{(tail)?}} call <8 x i8> @llvm.xtensa.ae.zalign64()
// CHECK: %[[CAST:.*]] = bitcast <8 x i8> %[[RET]] to i64
// CHECK: ret i64 %[[CAST]]
return __builtin_xtensa_ae_zalign64();
}

//--- rur_ae_bithead.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

int test_rur_ae_bithead() {
// CHECK-LABEL: test_rur_ae_bithead
// CHECK: %[[RET:.*]] = {{(tail)?}} call i32 @llvm.xtensa.rur.ae.bithead()
// CHECK: ret i32 %[[RET]]
return __builtin_xtensa_rur_ae_bithead();
}

//--- rur_ae_bitptr.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

int test_rur_ae_bitptr() {
// CHECK-LABEL: test_rur_ae_bitptr
// CHECK: %[[RET:.*]] = {{(tail)?}} call i32 @llvm.xtensa.rur.ae.bitptr()
// CHECK: ret i32 %[[RET]]
return __builtin_xtensa_rur_ae_bitptr();
}

//--- rur_ae_bitsused.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

int test_rur_ae_bitsused() {
// CHECK-LABEL: test_rur_ae_bitsused
// CHECK: %[[RET:.*]] = {{(tail)?}} call i32 @llvm.xtensa.rur.ae.bitsused()
// CHECK: ret i32 %[[RET]]
return __builtin_xtensa_rur_ae_bitsused();
}

//--- rur_ae_cbegin0.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

int test_rur_ae_cbegin0() {
// CHECK-LABEL: test_rur_ae_cbegin0
// CHECK: %[[RET:.*]] = {{(tail)?}} call i32 @llvm.xtensa.rur.ae.cbegin0()
// CHECK: ret i32 %[[RET]]
return __builtin_xtensa_rur_ae_cbegin0();
}

//--- rur_ae_cend0.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

int test_rur_ae_cend0() {
// CHECK-LABEL: test_rur_ae_cend0
// CHECK: %[[RET:.*]] = {{(tail)?}} call i32 @llvm.xtensa.rur.ae.cend0()
// CHECK: ret i32 %[[RET]]
return __builtin_xtensa_rur_ae_cend0();
}

//--- rur_ae_cw_sd_no.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

int test_rur_ae_cw_sd_no() {
// CHECK-LABEL: test_rur_ae_cw_sd_no
// CHECK: %[[RET:.*]] = {{(tail)?}} call i32 @llvm.xtensa.rur.ae.cw.sd.no()
// CHECK: ret i32 %[[RET]]
return __builtin_xtensa_rur_ae_cw_sd_no();
}

//--- rur_ae_cwrap.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

int test_rur_ae_cwrap() {
// CHECK-LABEL: test_rur_ae_cwrap
// CHECK: %[[RET:.*]] = {{(tail)?}} call i32 @llvm.xtensa.rur.ae.cwrap()
// CHECK: ret i32 %[[RET]]
return __builtin_xtensa_rur_ae_cwrap();
}

//--- rur_ae_first_ts.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

int test_rur_ae_first_ts() {
// CHECK-LABEL: test_rur_ae_first_ts
// CHECK: %[[RET:.*]] = {{(tail)?}} call i32 @llvm.xtensa.rur.ae.first.ts()
// CHECK: ret i32 %[[RET]]
return __builtin_xtensa_rur_ae_first_ts();
}

//--- rur_ae_nextoffset.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

int test_rur_ae_nextoffset() {
// CHECK-LABEL: test_rur_ae_nextoffset
// CHECK: %[[RET:.*]] = {{(tail)?}} call i32 @llvm.xtensa.rur.ae.nextoffset()
// CHECK: ret i32 %[[RET]]
return __builtin_xtensa_rur_ae_nextoffset();
}

//--- rur_ae_overflow.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

int test_rur_ae_overflow() {
// CHECK-LABEL: test_rur_ae_overflow
// CHECK: %[[RET:.*]] = {{(tail)?}} call i32 @llvm.xtensa.rur.ae.overflow()
// CHECK: ret i32 %[[RET]]
return __builtin_xtensa_rur_ae_overflow();
}

//--- rur_ae_ovf_sar.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

int test_rur_ae_ovf_sar() {
// CHECK-LABEL: test_rur_ae_ovf_sar
// CHECK: %[[RET:.*]] = {{(tail)?}} call i32 @llvm.xtensa.rur.ae.ovf.sar()
// CHECK: ret i32 %[[RET]]
return __builtin_xtensa_rur_ae_ovf_sar();
}

//--- rur_ae_sar.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

int test_rur_ae_sar() {
// CHECK-LABEL: test_rur_ae_sar
// CHECK: %[[RET:.*]] = {{(tail)?}} call i32 @llvm.xtensa.rur.ae.sar()
// CHECK: ret i32 %[[RET]]
return __builtin_xtensa_rur_ae_sar();
}

//--- rur_ae_searchdone.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

int test_rur_ae_searchdone() {
// CHECK-LABEL: test_rur_ae_searchdone
// CHECK: %[[RET:.*]] = {{(tail)?}} call i32 @llvm.xtensa.rur.ae.searchdone()
// CHECK: ret i32 %[[RET]]
return __builtin_xtensa_rur_ae_searchdone();
}

//--- rur_ae_tablesize.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

int test_rur_ae_tablesize() {
// CHECK-LABEL: test_rur_ae_tablesize
// CHECK: %[[RET:.*]] = {{(tail)?}} call i32 @llvm.xtensa.rur.ae.tablesize()
// CHECK: ret i32 %[[RET]]
return __builtin_xtensa_rur_ae_tablesize();
}

//--- rur_ae_ts_fts_bu_bp.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

int test_rur_ae_ts_fts_bu_bp() {
// CHECK-LABEL: test_rur_ae_ts_fts_bu_bp
// CHECK: %[[RET:.*]] = {{(tail)?}} call i32 @llvm.xtensa.rur.ae.ts.fts.bu.bp()
// CHECK: ret i32 %[[RET]]
return __builtin_xtensa_rur_ae_ts_fts_bu_bp();
}

//--- wur_ae_bithead.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_wur_ae_bithead(int art) {
// CHECK-LABEL: test_wur_ae_bithead
// CHECK:  {{(tail)?}} call void @llvm.xtensa.wur.ae.bithead(i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_wur_ae_bithead(art);
}

//--- wur_ae_bitptr.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_wur_ae_bitptr(int art) {
// CHECK-LABEL: test_wur_ae_bitptr
// CHECK:  {{(tail)?}} call void @llvm.xtensa.wur.ae.bitptr(i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_wur_ae_bitptr(art);
}

//--- wur_ae_bitsused.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_wur_ae_bitsused(int art) {
// CHECK-LABEL: test_wur_ae_bitsused
// CHECK:  {{(tail)?}} call void @llvm.xtensa.wur.ae.bitsused(i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_wur_ae_bitsused(art);
}

//--- wur_ae_cbegin0.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_wur_ae_cbegin0(int art) {
// CHECK-LABEL: test_wur_ae_cbegin0
// CHECK:  {{(tail)?}} call void @llvm.xtensa.wur.ae.cbegin0(i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_wur_ae_cbegin0(art);
}

//--- wur_ae_cend0.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_wur_ae_cend0(int art) {
// CHECK-LABEL: test_wur_ae_cend0
// CHECK:  {{(tail)?}} call void @llvm.xtensa.wur.ae.cend0(i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_wur_ae_cend0(art);
}

//--- wur_ae_cw_sd_no.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_wur_ae_cw_sd_no(int art) {
// CHECK-LABEL: test_wur_ae_cw_sd_no
// CHECK:  {{(tail)?}} call void @llvm.xtensa.wur.ae.cw.sd.no(i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_wur_ae_cw_sd_no(art);
}

//--- wur_ae_cwrap.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_wur_ae_cwrap(int art) {
// CHECK-LABEL: test_wur_ae_cwrap
// CHECK:  {{(tail)?}} call void @llvm.xtensa.wur.ae.cwrap(i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_wur_ae_cwrap(art);
}

//--- wur_ae_first_ts.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_wur_ae_first_ts(int art) {
// CHECK-LABEL: test_wur_ae_first_ts
// CHECK:  {{(tail)?}} call void @llvm.xtensa.wur.ae.first.ts(i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_wur_ae_first_ts(art);
}

//--- wur_ae_nextoffset.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_wur_ae_nextoffset(int art) {
// CHECK-LABEL: test_wur_ae_nextoffset
// CHECK:  {{(tail)?}} call void @llvm.xtensa.wur.ae.nextoffset(i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_wur_ae_nextoffset(art);
}

//--- wur_ae_overflow.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_wur_ae_overflow(int art) {
// CHECK-LABEL: test_wur_ae_overflow
// CHECK:  {{(tail)?}} call void @llvm.xtensa.wur.ae.overflow(i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_wur_ae_overflow(art);
}

//--- wur_ae_ovf_sar.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_wur_ae_ovf_sar(int art) {
// CHECK-LABEL: test_wur_ae_ovf_sar
// CHECK:  {{(tail)?}} call void @llvm.xtensa.wur.ae.ovf.sar(i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_wur_ae_ovf_sar(art);
}

//--- wur_ae_sar.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_wur_ae_sar(int art) {
// CHECK-LABEL: test_wur_ae_sar
// CHECK:  {{(tail)?}} call void @llvm.xtensa.wur.ae.sar(i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_wur_ae_sar(art);
}

//--- wur_ae_searchdone.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_wur_ae_searchdone(int art) {
// CHECK-LABEL: test_wur_ae_searchdone
// CHECK:  {{(tail)?}} call void @llvm.xtensa.wur.ae.searchdone(i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_wur_ae_searchdone(art);
}

//--- wur_ae_tablesize.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_wur_ae_tablesize(int art) {
// CHECK-LABEL: test_wur_ae_tablesize
// CHECK:  {{(tail)?}} call void @llvm.xtensa.wur.ae.tablesize(i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_wur_ae_tablesize(art);
}

//--- wur_ae_ts_fts_bu_bp.c

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;
typedef __attribute__((ext_vector_type(2))) _Bool xtbool2;
typedef __attribute__((ext_vector_type(4))) _Bool xtbool4;

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));
typedef short ae_int16 __attribute__(( vector_size(2)));
typedef short ae_int16x2 __attribute__(( vector_size(4)));
typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef long long ae_int64 __attribute__(( vector_size(8)));
typedef unsigned char ae_valign __attribute__(( vector_size(8)));

typedef int immediate;

void test_wur_ae_ts_fts_bu_bp(int art) {
// CHECK-LABEL: test_wur_ae_ts_fts_bu_bp
// CHECK:  {{(tail)?}} call void @llvm.xtensa.wur.ae.ts.fts.bu.bp(i32 {{.*}})
// CHECK: ret void
 __builtin_xtensa_wur_ae_ts_fts_bu_bp(art);
}

