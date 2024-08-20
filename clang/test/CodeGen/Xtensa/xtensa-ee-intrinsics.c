// RUN: %clang_cc1 -no-opaque-pointers -triple xtensa -S -emit-llvm -O0 -o - %s \
// RUN: | FileCheck %s

#include <stdint.h>

void test() {
  // CHECK: void @test()
  uint32_t data = 10;
  float a, b, c, d;
  __builtin_xtensa_ee_andq(0, 4, 3);
  // CHECK: call void @llvm.xtensa.ee.andq
  __builtin_xtensa_ee_bitrev(5, data);
  // CHECK: call void @llvm.xtensa.ee.bitrev
  __builtin_xtensa_ee_cmul_s16(6, 1, 1, 1);
  // CHECK: call void @llvm.xtensa.ee.cmul.s16
  __builtin_xtensa_ee_cmul_s16_ld_incp(7, data, 3, 4, 7, 1);
  // CHECK: call void @llvm.xtensa.ee.cmul.s16.ld.incp
  __builtin_xtensa_ee_cmul_s16_st_incp(0, data, 1, 7, 3, 3);
  // CHECK: call void @llvm.xtensa.ee.cmul.s16.st.incp
  __builtin_xtensa_ee_fft_ams_s16_ld_incp(0, data, 4, 7, 7, 5, 5, 1);
  // CHECK: call void @llvm.xtensa.ee.fft.ams.s16.ld.incp
  __builtin_xtensa_ee_fft_ams_s16_ld_incp_uaup(1, data, 5, 2, 7, 3, 4, 0);
  // CHECK: call void @llvm.xtensa.ee.fft.ams.s16.ld.incp.uaup
  __builtin_xtensa_ee_fft_ams_s16_ld_r32_decp(4, data, 2, 4, 5, 0, 0, 1);
  // CHECK: call void @llvm.xtensa.ee.fft.ams.s16.ld.r32.decp
  __builtin_xtensa_ee_fft_ams_s16_st_incp(4, 5, data, data, 5, 6, 1, 1);
  // CHECK: call void @llvm.xtensa.ee.fft.ams.s16.st.incp
  __builtin_xtensa_ee_fft_cmul_s16_ld_xp(0, data, data, 4, 4, 0, 2);
  // CHECK: call void @llvm.xtensa.ee.fft.cmul.s16.ld.xp
  __builtin_xtensa_ee_fft_cmul_s16_st_xp(1, 1, 5, data, data, 6, 0, 0);
  // CHECK: call void @llvm.xtensa.ee.fft.cmul.s16.st.xp
  __builtin_xtensa_ee_fft_r2bf_s16(6, 0, 7, 1, 1);
  // CHECK: call void @llvm.xtensa.ee.fft.r2bf.s16
  __builtin_xtensa_ee_fft_r2bf_s16_st_incp(4, 6, 3, data, 3);
  // CHECK: call void @llvm.xtensa.ee.fft.r2bf.s16.st.incp
  __builtin_xtensa_ee_fft_vst_r32_decp(6, data, 0);
  // CHECK: call void @llvm.xtensa.ee.fft.vst.r32.decp
  __builtin_xtensa_ee_ldf_128_ip(a, b, d, d, data, 46);
  // CHECK: call void @llvm.xtensa.ee.ldf.128.ip
  __builtin_xtensa_ee_ldf_128_xp(b, b, d, b, data, data);
  // CHECK: call void @llvm.xtensa.ee.ldf.128.xp
  __builtin_xtensa_ee_ldf_64_ip(b, d, data, -962);
  // CHECK: call void @llvm.xtensa.ee.ldf.64.ip
  __builtin_xtensa_ee_ldf_64_xp(d, c, data, data);
  // CHECK: call void @llvm.xtensa.ee.ldf.64.xp
  __builtin_xtensa_ee_ldqa_s16_128_ip(data, 1790);
  // CHECK: call void @llvm.xtensa.ee.ldqa.s16.128.ip
  __builtin_xtensa_ee_ldqa_s16_128_xp(data, data);
  // CHECK: call void @llvm.xtensa.ee.ldqa.s16.128.xp
  __builtin_xtensa_ee_ldqa_s8_128_ip(data, -331);
  // CHECK: call void @llvm.xtensa.ee.ldqa.s8.128.ip
  __builtin_xtensa_ee_ldqa_s8_128_xp(data, data);
  // CHECK: call void @llvm.xtensa.ee.ldqa.s8.128.xp
  __builtin_xtensa_ee_ldqa_u16_128_ip(data, -1341);
  // CHECK: call void @llvm.xtensa.ee.ldqa.u16.128.ip
  __builtin_xtensa_ee_ldqa_u16_128_xp(data, data);
  // CHECK: call void @llvm.xtensa.ee.ldqa.u16.128.xp
  __builtin_xtensa_ee_ldqa_u8_128_ip(data, -1372);
  // CHECK: call void @llvm.xtensa.ee.ldqa.u8.128.ip
  __builtin_xtensa_ee_ldqa_u8_128_xp(data, data);
  // CHECK: call void @llvm.xtensa.ee.ldqa.u8.128.xp
  __builtin_xtensa_ee_ldxq_32(3, 3, data, 2, 2);
  // CHECK: call void @llvm.xtensa.ee.ldxq.32
  __builtin_xtensa_ee_ld_128_usar_ip(0, data, 344);
  // CHECK: call void @llvm.xtensa.ee.ld.128.usar.ip
  __builtin_xtensa_ee_ld_128_usar_xp(7, data, data);
  // CHECK: call void @llvm.xtensa.ee.ld.128.usar.xp
  __builtin_xtensa_ee_ld_accx_ip(data, 144);
  // CHECK: call void @llvm.xtensa.ee.ld.accx.ip
  __builtin_xtensa_ee_ld_qacc_h_h_32_ip(data, 355);
  // CHECK: call void @llvm.xtensa.ee.ld.qacc.h.h.32.ip
  __builtin_xtensa_ee_ld_qacc_h_l_128_ip(data, 1267);
  // CHECK: call void @llvm.xtensa.ee.ld.qacc.h.l.128.ip
  __builtin_xtensa_ee_ld_qacc_l_h_32_ip(data, 220);
  // CHECK: call void @llvm.xtensa.ee.ld.qacc.l.h.32.ip
  __builtin_xtensa_ee_ld_qacc_l_l_128_ip(data, 1453);
  // CHECK: call void @llvm.xtensa.ee.ld.qacc.l.l.128.ip
  __builtin_xtensa_ee_ld_ua_state_ip(data, -1314);
  // CHECK: call void @llvm.xtensa.ee.ld.ua.state.ip
  __builtin_xtensa_ee_movi_32_a(1, data, 0);
  // CHECK: call void @llvm.xtensa.ee.movi.32.a
  __builtin_xtensa_ee_movi_32_q(4, data, 3);
  // CHECK: call void @llvm.xtensa.ee.movi.32.q
  __builtin_xtensa_ee_mov_s16_qacc(0);
  // CHECK: call void @llvm.xtensa.ee.mov.s16.qacc
  __builtin_xtensa_ee_mov_s8_qacc(7);
  // CHECK: call void @llvm.xtensa.ee.mov.s8.qacc
  __builtin_xtensa_ee_mov_u16_qacc(0);
  // CHECK: call void @llvm.xtensa.ee.mov.u16.qacc
  __builtin_xtensa_ee_mov_u8_qacc(1);
  // CHECK: call void @llvm.xtensa.ee.mov.u8.qacc
  __builtin_xtensa_ee_notq(0, 2);
  // CHECK: call void @llvm.xtensa.ee.notq
  __builtin_xtensa_ee_orq(6, 2, 7);
  // CHECK: call void @llvm.xtensa.ee.orq
  __builtin_xtensa_ee_slci_2q(4, 3, 8);
  // CHECK: call void @llvm.xtensa.ee.slci.2q
  __builtin_xtensa_ee_slcxxp_2q(2, 5, data, data);
  // CHECK: call void @llvm.xtensa.ee.slcxxp.2q
  __builtin_xtensa_ee_srci_2q(0, 6, 13);
  // CHECK: call void @llvm.xtensa.ee.srci.2q
  __builtin_xtensa_ee_srcmb_s16_qacc(3, data, 1);
  // CHECK: call void @llvm.xtensa.ee.srcmb.s16.qacc
  __builtin_xtensa_ee_srcmb_s8_qacc(3, data, 1);
  // CHECK: call void @llvm.xtensa.ee.srcmb.s8.qacc
  __builtin_xtensa_ee_srcq_128_st_incp(0, 7, data);
  // CHECK: call void @llvm.xtensa.ee.srcq.128.st.incp
  __builtin_xtensa_ee_srcxxp_2q(1, 3, data, data);
  // CHECK: call void @llvm.xtensa.ee.srcxxp.2q
  __builtin_xtensa_ee_src_q(3, 4, 7);
  // CHECK: call void @llvm.xtensa.ee.src.q
  __builtin_xtensa_ee_src_q_ld_ip(4, data, -1149, 4, 7);
  // CHECK: call void @llvm.xtensa.ee.src.q.ld.ip
  __builtin_xtensa_ee_src_q_ld_xp(4, data, data, 7, 2);
  // CHECK: call void @llvm.xtensa.ee.src.q.ld.xp
  __builtin_xtensa_ee_src_q_qup(4, 5, 7);
  // CHECK: call void @llvm.xtensa.ee.src.q.qup
  __builtin_xtensa_ee_srs_accx(data, data, 0);
  // CHECK: call void @llvm.xtensa.ee.srs.accx
  __builtin_xtensa_ee_stf_128_ip(c, a, b, c, data, 14);
  // CHECK: call void @llvm.xtensa.ee.stf.128.ip
  __builtin_xtensa_ee_stf_128_xp(b, b, d, d, data, data);
  // CHECK: call void @llvm.xtensa.ee.stf.128.xp
  __builtin_xtensa_ee_stf_64_ip(d, c, data, 984);
  // CHECK: call void @llvm.xtensa.ee.stf.64.ip
  __builtin_xtensa_ee_stf_64_xp(d, b, data, data);
  // CHECK: call void @llvm.xtensa.ee.stf.64.xp
  __builtin_xtensa_ee_stxq_32(1, 0, data, 2, 6);
  // CHECK: call void @llvm.xtensa.ee.stxq.32
  __builtin_xtensa_ee_st_accx_ip(data, 842);
  // CHECK: call void @llvm.xtensa.ee.st.accx.ip
  __builtin_xtensa_ee_st_qacc_h_h_32_ip(data, -243);
  // CHECK: call void @llvm.xtensa.ee.st.qacc.h.h.32.ip
  __builtin_xtensa_ee_st_qacc_h_l_128_ip(data, 663);
  // CHECK: call void @llvm.xtensa.ee.st.qacc.h.l.128.ip
  __builtin_xtensa_ee_st_qacc_l_h_32_ip(data, 97);
  // CHECK: call void @llvm.xtensa.ee.st.qacc.l.h.32.ip
  __builtin_xtensa_ee_st_qacc_l_l_128_ip(data, -864);
  // CHECK: call void @llvm.xtensa.ee.st.qacc.l.l.128.ip
  __builtin_xtensa_ee_st_ua_state_ip(data, -709);
  // CHECK: call void @llvm.xtensa.ee.st.ua.state.ip
  __builtin_xtensa_ee_vadds_s16(7, 5, 2);
  // CHECK: call void @llvm.xtensa.ee.vadds.s16
  __builtin_xtensa_ee_vadds_s16_ld_incp(4, data, 6, 1, 6);
  // CHECK: call void @llvm.xtensa.ee.vadds.s16.ld.incp
  __builtin_xtensa_ee_vadds_s16_st_incp(4, data, 1, 3, 2);
  // CHECK: call void @llvm.xtensa.ee.vadds.s16.st.incp
  __builtin_xtensa_ee_vadds_s32(7, 0, 6);
  // CHECK: call void @llvm.xtensa.ee.vadds.s32
  __builtin_xtensa_ee_vadds_s32_ld_incp(4, data, 4, 7, 1);
  // CHECK: call void @llvm.xtensa.ee.vadds.s32.ld.incp
  __builtin_xtensa_ee_vadds_s32_st_incp(1, data, 7, 4, 4);
  // CHECK: call void @llvm.xtensa.ee.vadds.s32.st.incp
  __builtin_xtensa_ee_vadds_s8(0, 3, 6);
  // CHECK: call void @llvm.xtensa.ee.vadds.s8
  __builtin_xtensa_ee_vadds_s8_ld_incp(3, data, 2, 4, 2);
  // CHECK: call void @llvm.xtensa.ee.vadds.s8.ld.incp
  __builtin_xtensa_ee_vadds_s8_st_incp(2, data, 1, 7, 2);
  // CHECK: call void @llvm.xtensa.ee.vadds.s8.st.incp
  __builtin_xtensa_ee_vcmp_eq_s16(2, 1, 0);
  // CHECK: call void @llvm.xtensa.ee.vcmp.eq.s16
  __builtin_xtensa_ee_vcmp_eq_s32(6, 1, 2);
  // CHECK: call void @llvm.xtensa.ee.vcmp.eq.s32
  __builtin_xtensa_ee_vcmp_eq_s8(0, 2, 1);
  // CHECK: call void @llvm.xtensa.ee.vcmp.eq.s8
  __builtin_xtensa_ee_vcmp_gt_s16(1, 2, 5);
  // CHECK: call void @llvm.xtensa.ee.vcmp.gt.s16
  __builtin_xtensa_ee_vcmp_gt_s32(0, 0, 0);
  // CHECK: call void @llvm.xtensa.ee.vcmp.gt.s32
  __builtin_xtensa_ee_vcmp_gt_s8(4, 1, 4);
  // CHECK: call void @llvm.xtensa.ee.vcmp.gt.s8
  __builtin_xtensa_ee_vcmp_lt_s16(7, 4, 0);
  // CHECK: call void @llvm.xtensa.ee.vcmp.lt.s16
  __builtin_xtensa_ee_vcmp_lt_s32(0, 1, 0);
  // CHECK: call void @llvm.xtensa.ee.vcmp.lt.s32
  __builtin_xtensa_ee_vcmp_lt_s8(3, 3, 3);
  // CHECK: call void @llvm.xtensa.ee.vcmp.lt.s8
  __builtin_xtensa_ee_vldbc_16(6, data);
  // CHECK: call void @llvm.xtensa.ee.vldbc.16
  __builtin_xtensa_ee_vldbc_16_ip(3, data, 115);
  // CHECK: call void @llvm.xtensa.ee.vldbc.16.ip
  __builtin_xtensa_ee_vldbc_16_xp(0, data, data);
  // CHECK: call void @llvm.xtensa.ee.vldbc.16.xp
  __builtin_xtensa_ee_vldbc_32(4, data);
  // CHECK: call void @llvm.xtensa.ee.vldbc.32
  __builtin_xtensa_ee_vldbc_32_ip(5, data, -213);
  // CHECK: call void @llvm.xtensa.ee.vldbc.32.ip
  __builtin_xtensa_ee_vldbc_32_xp(1, data, data);
  // CHECK: call void @llvm.xtensa.ee.vldbc.32.xp
  __builtin_xtensa_ee_vldbc_8(5, data);
  // CHECK: call void @llvm.xtensa.ee.vldbc.8
  __builtin_xtensa_ee_vldbc_8_ip(7, data, 32);
  // CHECK: call void @llvm.xtensa.ee.vldbc.8.ip
  __builtin_xtensa_ee_vldbc_8_xp(6, data, data);
  // CHECK: call void @llvm.xtensa.ee.vldbc.8.xp
  __builtin_xtensa_ee_vldhbc_16_incp(1, 1, data);
  // CHECK: call void @llvm.xtensa.ee.vldhbc.16.incp
  __builtin_xtensa_ee_vld_128_ip(6, data, 1284);
  // CHECK: call void @llvm.xtensa.ee.vld.128.ip
  __builtin_xtensa_ee_vld_128_xp(4, data, data);
  // CHECK: call void @llvm.xtensa.ee.vld.128.xp
  __builtin_xtensa_ee_vld_h_64_ip(1, data, 400);
  // CHECK: call void @llvm.xtensa.ee.vld.h.64.ip
  __builtin_xtensa_ee_vld_h_64_xp(1, data, data);
  // CHECK: call void @llvm.xtensa.ee.vld.h.64.xp
  __builtin_xtensa_ee_vld_l_64_ip(0, data, 907);
  // CHECK: call void @llvm.xtensa.ee.vld.l.64.ip
  __builtin_xtensa_ee_vld_l_64_xp(0, data, data);
  // CHECK: call void @llvm.xtensa.ee.vld.l.64.xp
  __builtin_xtensa_ee_vmax_s16(4, 1, 0);
  // CHECK: call void @llvm.xtensa.ee.vmax.s16
  __builtin_xtensa_ee_vmax_s16_ld_incp(3, data, 0, 3, 6);
  // CHECK: call void @llvm.xtensa.ee.vmax.s16.ld.incp
  __builtin_xtensa_ee_vmax_s16_st_incp(7, data, 2, 0, 2);
  // CHECK: call void @llvm.xtensa.ee.vmax.s16.st.incp
  __builtin_xtensa_ee_vmax_s32(1, 4, 0);
  // CHECK: call void @llvm.xtensa.ee.vmax.s32
  __builtin_xtensa_ee_vmax_s32_ld_incp(6, data, 1, 7, 5);
  // CHECK: call void @llvm.xtensa.ee.vmax.s32.ld.incp
  __builtin_xtensa_ee_vmax_s32_st_incp(5, data, 4, 6, 4);
  // CHECK: call void @llvm.xtensa.ee.vmax.s32.st.incp
  __builtin_xtensa_ee_vmax_s8(2, 4, 6);
  // CHECK: call void @llvm.xtensa.ee.vmax.s8
  __builtin_xtensa_ee_vmax_s8_ld_incp(7, data, 7, 3, 3);
  // CHECK: call void @llvm.xtensa.ee.vmax.s8.ld.incp
  __builtin_xtensa_ee_vmax_s8_st_incp(0, data, 3, 5, 1);
  // CHECK: call void @llvm.xtensa.ee.vmax.s8.st.incp
  __builtin_xtensa_ee_vmin_s16(7, 4, 3);
  // CHECK: call void @llvm.xtensa.ee.vmin.s16
  __builtin_xtensa_ee_vmin_s16_ld_incp(2, data, 5, 2, 1);
  // CHECK: call void @llvm.xtensa.ee.vmin.s16.ld.incp
  __builtin_xtensa_ee_vmin_s16_st_incp(4, data, 4, 2, 0);
  // CHECK: call void @llvm.xtensa.ee.vmin.s16.st.incp
  __builtin_xtensa_ee_vmin_s32(4, 6, 5);
  // CHECK: call void @llvm.xtensa.ee.vmin.s32
  __builtin_xtensa_ee_vmin_s32_ld_incp(1, data, 3, 3, 5);
  // CHECK: call void @llvm.xtensa.ee.vmin.s32.ld.incp
  __builtin_xtensa_ee_vmin_s32_st_incp(3, data, 1, 0, 6);
  // CHECK: call void @llvm.xtensa.ee.vmin.s32.st.incp
  __builtin_xtensa_ee_vmin_s8(5, 1, 3);
  // CHECK: call void @llvm.xtensa.ee.vmin.s8
  __builtin_xtensa_ee_vmin_s8_ld_incp(0, data, 7, 2, 5);
  // CHECK: call void @llvm.xtensa.ee.vmin.s8.ld.incp
  __builtin_xtensa_ee_vmin_s8_st_incp(0, data, 5, 3, 7);
  // CHECK: call void @llvm.xtensa.ee.vmin.s8.st.incp
  __builtin_xtensa_ee_vmulas_s16_accx(4, 1);
  // CHECK: call void @llvm.xtensa.ee.vmulas.s16.accx
  __builtin_xtensa_ee_vmulas_s16_accx_ld_ip(7, data, 26, 1, 0);
  // CHECK: call void @llvm.xtensa.ee.vmulas.s16.accx.ld.ip
  __builtin_xtensa_ee_vmulas_s16_accx_ld_ip_qup(0, data, 7, 2, 3, 3, 7);
  // CHECK: call void @llvm.xtensa.ee.vmulas.s16.accx.ld.ip.qup
  __builtin_xtensa_ee_vmulas_s16_accx_ld_xp(6, data, data, 0, 3);
  // CHECK: call void @llvm.xtensa.ee.vmulas.s16.accx.ld.xp
  __builtin_xtensa_ee_vmulas_s16_accx_ld_xp_qup(7, data, data, 2, 4, 7, 2);
  // CHECK: call void @llvm.xtensa.ee.vmulas.s16.accx.ld.xp.qup
  __builtin_xtensa_ee_vmulas_s16_qacc(7, 7);
  // CHECK: call void @llvm.xtensa.ee.vmulas.s16.qacc
  __builtin_xtensa_ee_vmulas_s16_qacc_ldbc_incp(6, data, 1, 0);
  // CHECK: call void @llvm.xtensa.ee.vmulas.s16.qacc.ldbc.incp
  __builtin_xtensa_ee_vmulas_s16_qacc_ldbc_incp_qup(5, data, 4, 5, 3, 1);
  // CHECK: call void @llvm.xtensa.ee.vmulas.s16.qacc.ldbc.incp.qup
  __builtin_xtensa_ee_vmulas_s16_qacc_ld_ip(1, data, 35, 2, 2);
  // CHECK: call void @llvm.xtensa.ee.vmulas.s16.qacc.ld.ip
  __builtin_xtensa_ee_vmulas_s16_qacc_ld_ip_qup(1, data, 24, 0, 1, 1, 7);
  // CHECK: call void @llvm.xtensa.ee.vmulas.s16.qacc.ld.ip.qup
  __builtin_xtensa_ee_vmulas_s16_qacc_ld_xp(4, data, data, 5, 3);
  // CHECK: call void @llvm.xtensa.ee.vmulas.s16.qacc.ld.xp
  __builtin_xtensa_ee_vmulas_s16_qacc_ld_xp_qup(2, data, data, 3, 3, 1, 2);
  // CHECK: call void @llvm.xtensa.ee.vmulas.s16.qacc.ld.xp.qup
  __builtin_xtensa_ee_vmulas_s8_accx(5, 5);
  // CHECK: call void @llvm.xtensa.ee.vmulas.s8.accx
  __builtin_xtensa_ee_vmulas_s8_accx_ld_ip(0, data, -30, 1, 5);
  // CHECK: call void @llvm.xtensa.ee.vmulas.s8.accx.ld.ip
  __builtin_xtensa_ee_vmulas_s8_accx_ld_ip_qup(2, data, -1, 5, 0, 4, 0);
  // CHECK: call void @llvm.xtensa.ee.vmulas.s8.accx.ld.ip.qup
  __builtin_xtensa_ee_vmulas_s8_accx_ld_xp(1, data, data, 4, 5);
  // CHECK: call void @llvm.xtensa.ee.vmulas.s8.accx.ld.xp
  __builtin_xtensa_ee_vmulas_s8_accx_ld_xp_qup(4, data, data, 5, 7, 7, 0);
  // CHECK: call void @llvm.xtensa.ee.vmulas.s8.accx.ld.xp.qup
  __builtin_xtensa_ee_vmulas_s8_qacc(6, 7);
  // CHECK: call void @llvm.xtensa.ee.vmulas.s8.qacc
  __builtin_xtensa_ee_vmulas_s8_qacc_ldbc_incp(3, data, 6, 4);
  // CHECK: call void @llvm.xtensa.ee.vmulas.s8.qacc.ldbc.incp
  __builtin_xtensa_ee_vmulas_s8_qacc_ldbc_incp_qup(7, data, 2, 3, 4, 1);
  // CHECK: call void @llvm.xtensa.ee.vmulas.s8.qacc.ldbc.incp.qup
  __builtin_xtensa_ee_vmulas_s8_qacc_ld_ip(4, data, -99, 2, 4);
  // CHECK: call void @llvm.xtensa.ee.vmulas.s8.qacc.ld.ip
  __builtin_xtensa_ee_vmulas_s8_qacc_ld_ip_qup(4, data, -87, 5, 4, 2, 3);
  // CHECK: call void @llvm.xtensa.ee.vmulas.s8.qacc.ld.ip.qup
  __builtin_xtensa_ee_vmulas_s8_qacc_ld_xp(7, data, data, 1, 1);
  // CHECK: call void @llvm.xtensa.ee.vmulas.s8.qacc.ld.xp
  __builtin_xtensa_ee_vmulas_s8_qacc_ld_xp_qup(2, data, data, 1, 6, 3, 2);
  // CHECK: call void @llvm.xtensa.ee.vmulas.s8.qacc.ld.xp.qup
  __builtin_xtensa_ee_vmulas_u16_accx(2, 5);
  // CHECK: call void @llvm.xtensa.ee.vmulas.u16.accx
  __builtin_xtensa_ee_vmulas_u16_accx_ld_ip(0, data, -28, 6, 7);
  // CHECK: call void @llvm.xtensa.ee.vmulas.u16.accx.ld.ip
  __builtin_xtensa_ee_vmulas_u16_accx_ld_ip_qup(5, data, -41, 2, 1, 6, 6);
  // CHECK: call void @llvm.xtensa.ee.vmulas.u16.accx.ld.ip.qup
  __builtin_xtensa_ee_vmulas_u16_accx_ld_xp(1, data, data, 6, 7);
  // CHECK: call void @llvm.xtensa.ee.vmulas.u16.accx.ld.xp
  __builtin_xtensa_ee_vmulas_u16_accx_ld_xp_qup(7, data, data, 0, 0, 7, 5);
  // CHECK: call void @llvm.xtensa.ee.vmulas.u16.accx.ld.xp.qup
  __builtin_xtensa_ee_vmulas_u16_qacc(4, 3);
  // CHECK: call void @llvm.xtensa.ee.vmulas.u16.qacc
  __builtin_xtensa_ee_vmulas_u16_qacc_ldbc_incp(3, data, 5, 6);
  // CHECK: call void @llvm.xtensa.ee.vmulas.u16.qacc.ldbc.incp
  __builtin_xtensa_ee_vmulas_u16_qacc_ldbc_incp_qup(3, data, 4, 5, 4, 0);
  // CHECK: call void @llvm.xtensa.ee.vmulas.u16.qacc.ldbc.incp.qup
  __builtin_xtensa_ee_vmulas_u16_qacc_ld_ip(4, data, -94, 5, 3);
  // CHECK: call void @llvm.xtensa.ee.vmulas.u16.qacc.ld.ip
  __builtin_xtensa_ee_vmulas_u16_qacc_ld_ip_qup(7, data, -116, 5, 6, 1, 4);
  // CHECK: call void @llvm.xtensa.ee.vmulas.u16.qacc.ld.ip.qup
  __builtin_xtensa_ee_vmulas_u16_qacc_ld_xp(3, data, data, 0, 6);
  // CHECK: call void @llvm.xtensa.ee.vmulas.u16.qacc.ld.xp
  __builtin_xtensa_ee_vmulas_u16_qacc_ld_xp_qup(0, data, data, 5, 3, 3, 7);
  // CHECK: call void @llvm.xtensa.ee.vmulas.u16.qacc.ld.xp.qup
  __builtin_xtensa_ee_vmulas_u8_accx(3, 0);
  // CHECK: call void @llvm.xtensa.ee.vmulas.u8.accx
  __builtin_xtensa_ee_vmulas_u8_accx_ld_ip(1, data, -112, 1, 5);
  // CHECK: call void @llvm.xtensa.ee.vmulas.u8.accx.ld.ip
  __builtin_xtensa_ee_vmulas_u8_accx_ld_ip_qup(4, data, -68, 3, 1, 5, 4);
  // CHECK: call void @llvm.xtensa.ee.vmulas.u8.accx.ld.ip.qup
  __builtin_xtensa_ee_vmulas_u8_accx_ld_xp(2, data, data, 0, 5);
  // CHECK: call void @llvm.xtensa.ee.vmulas.u8.accx.ld.xp
  __builtin_xtensa_ee_vmulas_u8_accx_ld_xp_qup(0, data, data, 3, 5, 3, 3);
  // CHECK: call void @llvm.xtensa.ee.vmulas.u8.accx.ld.xp.qup
  __builtin_xtensa_ee_vmulas_u8_qacc(4, 2);
  // CHECK: call void @llvm.xtensa.ee.vmulas.u8.qacc
  __builtin_xtensa_ee_vmulas_u8_qacc_ldbc_incp(6, data, 4, 7);
  // CHECK: call void @llvm.xtensa.ee.vmulas.u8.qacc.ldbc.incp
  __builtin_xtensa_ee_vmulas_u8_qacc_ldbc_incp_qup(3, data, 3, 4, 4, 3);
  // CHECK: call void @llvm.xtensa.ee.vmulas.u8.qacc.ldbc.incp.qup
  __builtin_xtensa_ee_vmulas_u8_qacc_ld_ip(4, data, 46, 2, 0);
  // CHECK: call void @llvm.xtensa.ee.vmulas.u8.qacc.ld.ip
  __builtin_xtensa_ee_vmulas_u8_qacc_ld_ip_qup(4, data, -51, 6, 2, 2, 0);
  // CHECK: call void @llvm.xtensa.ee.vmulas.u8.qacc.ld.ip.qup
  __builtin_xtensa_ee_vmulas_u8_qacc_ld_xp(7, data, data, 1, 2);
  // CHECK: call void @llvm.xtensa.ee.vmulas.u8.qacc.ld.xp
  __builtin_xtensa_ee_vmulas_u8_qacc_ld_xp_qup(5, data, data, 0, 4, 5, 0);
  // CHECK: call void @llvm.xtensa.ee.vmulas.u8.qacc.ld.xp.qup
  __builtin_xtensa_ee_vmul_s16(4, 2, 5);
  // CHECK: call void @llvm.xtensa.ee.vmul.s16
  __builtin_xtensa_ee_vmul_s16_ld_incp(2, data, 7, 0, 3);
  // CHECK: call void @llvm.xtensa.ee.vmul.s16.ld.incp
  __builtin_xtensa_ee_vmul_s16_st_incp(3, data, 5, 7, 6);
  // CHECK: call void @llvm.xtensa.ee.vmul.s16.st.incp
  __builtin_xtensa_ee_vmul_s8(1, 3, 6);
  // CHECK: call void @llvm.xtensa.ee.vmul.s8
  __builtin_xtensa_ee_vmul_s8_ld_incp(3, data, 2, 2, 6);
  // CHECK: call void @llvm.xtensa.ee.vmul.s8.ld.incp
  __builtin_xtensa_ee_vmul_s8_st_incp(1, data, 2, 1, 4);
  // CHECK: call void @llvm.xtensa.ee.vmul.s8.st.incp
  __builtin_xtensa_ee_vmul_u16(6, 6, 2);
  // CHECK: call void @llvm.xtensa.ee.vmul.u16
  __builtin_xtensa_ee_vmul_u16_ld_incp(0, data, 7, 0, 5);
  // CHECK: call void @llvm.xtensa.ee.vmul.u16.ld.incp
  __builtin_xtensa_ee_vmul_u16_st_incp(7, data, 4, 0, 2);
  // CHECK: call void @llvm.xtensa.ee.vmul.u16.st.incp
  __builtin_xtensa_ee_vmul_u8(1, 5, 7);
  // CHECK: call void @llvm.xtensa.ee.vmul.u8
  __builtin_xtensa_ee_vmul_u8_ld_incp(1, data, 7, 4, 0);
  // CHECK: call void @llvm.xtensa.ee.vmul.u8.ld.incp
  __builtin_xtensa_ee_vmul_u8_st_incp(7, data, 0, 3, 4);
  // CHECK: call void @llvm.xtensa.ee.vmul.u8.st.incp
  __builtin_xtensa_ee_vprelu_s16(0, 0, 6, data);
  // CHECK: call void @llvm.xtensa.ee.vprelu.s16
  __builtin_xtensa_ee_vprelu_s8(4, 7, 1, data);
  // CHECK: call void @llvm.xtensa.ee.vprelu.s8
  __builtin_xtensa_ee_vrelu_s16(6, data, data);
  // CHECK: call void @llvm.xtensa.ee.vrelu.s16
  __builtin_xtensa_ee_vrelu_s8(1, data, data);
  // CHECK: call void @llvm.xtensa.ee.vrelu.s8
  __builtin_xtensa_ee_vsl_32(5, 1);
  // CHECK: call void @llvm.xtensa.ee.vsl.32
  __builtin_xtensa_ee_vsmulas_s16_qacc(4, 6, 4);
  // CHECK: call void @llvm.xtensa.ee.vsmulas.s16.qacc
  __builtin_xtensa_ee_vsmulas_s16_qacc_ld_incp(4, data, 6, 6, 4);
  // CHECK: call void @llvm.xtensa.ee.vsmulas.s16.qacc.ld.incp
  __builtin_xtensa_ee_vsmulas_s8_qacc(3, 2, 9);
  // CHECK: call void @llvm.xtensa.ee.vsmulas.s8.qacc
  __builtin_xtensa_ee_vsmulas_s8_qacc_ld_incp(3, data, 2, 2, 5);
  // CHECK: call void @llvm.xtensa.ee.vsmulas.s8.qacc.ld.incp
  __builtin_xtensa_ee_vsr_32(3, 1);
  // CHECK: call void @llvm.xtensa.ee.vsr.32
  __builtin_xtensa_ee_vst_128_ip(6, data, 68);
  // CHECK: call void @llvm.xtensa.ee.vst.128.ip
  __builtin_xtensa_ee_vst_128_xp(7, data, data);
  // CHECK: call void @llvm.xtensa.ee.vst.128.xp
  __builtin_xtensa_ee_vst_h_64_ip(2, data, 843);
  // CHECK: call void @llvm.xtensa.ee.vst.h.64.ip
  __builtin_xtensa_ee_vst_h_64_xp(5, data, data);
  // CHECK: call void @llvm.xtensa.ee.vst.h.64.xp
  __builtin_xtensa_ee_vst_l_64_ip(7, data, -658);
  // CHECK: call void @llvm.xtensa.ee.vst.l.64.ip
  __builtin_xtensa_ee_vst_l_64_xp(7, data, data);
  // CHECK: call void @llvm.xtensa.ee.vst.l.64.xp
  __builtin_xtensa_ee_vsubs_s16(7, 5, 3);
  // CHECK: call void @llvm.xtensa.ee.vsubs.s16
  __builtin_xtensa_ee_vsubs_s16_ld_incp(7, data, 5, 1, 2);
  // CHECK: call void @llvm.xtensa.ee.vsubs.s16.ld.incp
  __builtin_xtensa_ee_vsubs_s16_st_incp(3, data, 5, 2, 3);
  // CHECK: call void @llvm.xtensa.ee.vsubs.s16.st.incp
  __builtin_xtensa_ee_vsubs_s32(0, 7, 3);
  // CHECK: call void @llvm.xtensa.ee.vsubs.s32
  __builtin_xtensa_ee_vsubs_s32_ld_incp(0, data, 4, 1, 0);
  // CHECK: call void @llvm.xtensa.ee.vsubs.s32.ld.incp
  __builtin_xtensa_ee_vsubs_s32_st_incp(4, data, 6, 7, 6);
  // CHECK: call void @llvm.xtensa.ee.vsubs.s32.st.incp
  __builtin_xtensa_ee_vsubs_s8(4, 4, 0);
  // CHECK: call void @llvm.xtensa.ee.vsubs.s8
  __builtin_xtensa_ee_vsubs_s8_ld_incp(1, data, 1, 4, 4);
  // CHECK: call void @llvm.xtensa.ee.vsubs.s8.ld.incp
  __builtin_xtensa_ee_vsubs_s8_st_incp(2, data, 4, 4, 1);
  // CHECK: call void @llvm.xtensa.ee.vsubs.s8.st.incp
  __builtin_xtensa_ee_vunzip_16(7, 6);
  // CHECK: call void @llvm.xtensa.ee.vunzip.16
  __builtin_xtensa_ee_vunzip_32(7, 3);
  // CHECK: call void @llvm.xtensa.ee.vunzip.32
  __builtin_xtensa_ee_vunzip_8(0, 2);
  // CHECK: call void @llvm.xtensa.ee.vunzip.8
  __builtin_xtensa_ee_vzip_16(1, 1);
  // CHECK: call void @llvm.xtensa.ee.vzip.16
  __builtin_xtensa_ee_vzip_32(0, 1);
  // CHECK: call void @llvm.xtensa.ee.vzip.32
  __builtin_xtensa_ee_vzip_8(0, 5);
  // CHECK: call void @llvm.xtensa.ee.vzip.8
  __builtin_xtensa_ee_xorq(4, 3, 3);
  // CHECK: call void @llvm.xtensa.ee.xorq
  __builtin_xtensa_ee_zero_accx();
  // CHECK: call void @llvm.xtensa.ee.zero.accx
  __builtin_xtensa_ee_zero_q(0);
  // CHECK: call void @llvm.xtensa.ee.zero.q
  __builtin_xtensa_ee_zero_qacc();
  // CHECK: call void @llvm.xtensa.ee.zero.qacc
  data = __builtin_xtensa_rur_accx_0();
  // CHECK: call i32 @llvm.xtensa.rur.accx.0
  data = __builtin_xtensa_rur_accx_1();
  // CHECK: call i32 @llvm.xtensa.rur.accx.1
  data = __builtin_xtensa_rur_fft_bit_width();
  // CHECK: call i32 @llvm.xtensa.rur.fft.bit.width
  data = __builtin_xtensa_rur_gpio_out();
  // CHECK: call i32 @llvm.xtensa.rur.gpio.out
  data = __builtin_xtensa_rur_qacc_h_0();
  // CHECK: call i32 @llvm.xtensa.rur.qacc.h.0
  data = __builtin_xtensa_rur_qacc_h_1();
  // CHECK: call i32 @llvm.xtensa.rur.qacc.h.1
  data = __builtin_xtensa_rur_qacc_h_2();
  // CHECK: call i32 @llvm.xtensa.rur.qacc.h.2
  data = __builtin_xtensa_rur_qacc_h_3();
  // CHECK: call i32 @llvm.xtensa.rur.qacc.h.3
  data = __builtin_xtensa_rur_qacc_h_4();
  // CHECK: call i32 @llvm.xtensa.rur.qacc.h.4
  data = __builtin_xtensa_rur_qacc_l_0();
  // CHECK: call i32 @llvm.xtensa.rur.qacc.l.0
  data = __builtin_xtensa_rur_qacc_l_1();
  // CHECK: call i32 @llvm.xtensa.rur.qacc.l.1
  data = __builtin_xtensa_rur_qacc_l_2();
  // CHECK: call i32 @llvm.xtensa.rur.qacc.l.2
  data = __builtin_xtensa_rur_qacc_l_3();
  // CHECK: call i32 @llvm.xtensa.rur.qacc.l.3
  data = __builtin_xtensa_rur_qacc_l_4();
  // CHECK: call i32 @llvm.xtensa.rur.qacc.l.4
  data = __builtin_xtensa_rur_sar_byte();
  // CHECK: call i32 @llvm.xtensa.rur.sar.byte
  data = __builtin_xtensa_rur_ua_state_0();
  // CHECK: call i32 @llvm.xtensa.rur.ua.state.0
  data = __builtin_xtensa_rur_ua_state_1();
  // CHECK: call i32 @llvm.xtensa.rur.ua.state.1
  data = __builtin_xtensa_rur_ua_state_2();
  // CHECK: call i32 @llvm.xtensa.rur.ua.state.2
  data = __builtin_xtensa_rur_ua_state_3();
  // CHECK: call i32 @llvm.xtensa.rur.ua.state.3
  __builtin_xtensa_wur_accx_0(data);
  // CHECK: call void @llvm.xtensa.wur.accx.0
  __builtin_xtensa_wur_accx_1(data);
  // CHECK: call void @llvm.xtensa.wur.accx.1
  __builtin_xtensa_wur_fft_bit_width(data);
  // CHECK: call void @llvm.xtensa.wur.fft.bit.width
  __builtin_xtensa_wur_gpio_out(data);
  // CHECK: call void @llvm.xtensa.wur.gpio.out
  __builtin_xtensa_wur_qacc_h_0(data);
  // CHECK: call void @llvm.xtensa.wur.qacc.h.0
  __builtin_xtensa_wur_qacc_h_1(data);
  // CHECK: call void @llvm.xtensa.wur.qacc.h.1
  __builtin_xtensa_wur_qacc_h_2(data);
  // CHECK: call void @llvm.xtensa.wur.qacc.h.2
  __builtin_xtensa_wur_qacc_h_3(data);
  // CHECK: call void @llvm.xtensa.wur.qacc.h.3
  __builtin_xtensa_wur_qacc_h_4(data);
  // CHECK: call void @llvm.xtensa.wur.qacc.h.4
  __builtin_xtensa_wur_qacc_l_0(data);
  // CHECK: call void @llvm.xtensa.wur.qacc.l.0
  __builtin_xtensa_wur_qacc_l_1(data);
  // CHECK: call void @llvm.xtensa.wur.qacc.l.1
  __builtin_xtensa_wur_qacc_l_2(data);
  // CHECK: call void @llvm.xtensa.wur.qacc.l.2
  __builtin_xtensa_wur_qacc_l_3(data);
  // CHECK: call void @llvm.xtensa.wur.qacc.l.3
  __builtin_xtensa_wur_qacc_l_4(data);
  // CHECK: call void @llvm.xtensa.wur.qacc.l.4
  __builtin_xtensa_wur_sar_byte(data);
  // CHECK: call void @llvm.xtensa.wur.sar.byte
  __builtin_xtensa_wur_ua_state_0(data);
  // CHECK: call void @llvm.xtensa.wur.ua.state.0
  __builtin_xtensa_wur_ua_state_1(data);
  // CHECK: call void @llvm.xtensa.wur.ua.state.1
  __builtin_xtensa_wur_ua_state_2(data);
  // CHECK: call void @llvm.xtensa.wur.ua.state.2
  __builtin_xtensa_wur_ua_state_3(data);
  // CHECK: call void @llvm.xtensa.wur.ua.state.3
  __builtin_xtensa_mv_qr(4, 5);
  // CHECK: call void @llvm.xtensa.mv.qr
}
