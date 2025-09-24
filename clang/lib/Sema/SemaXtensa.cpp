//===------ SemaXtensa.cpp ------- Xtensa target-specific routines --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements semantic analysis functions specific to Xtensa.
//
//===----------------------------------------------------------------------===//

#include "clang/Sema/SemaXtensa.h"
#include "clang/Basic/TargetBuiltins.h"
#include "clang/Sema/Sema.h"

namespace clang {

SemaXtensa::SemaXtensa(Sema &S) : SemaBase(S) {}

bool SemaXtensa::CheckXtensaBuiltinFunctionCall(const TargetInfo &TI,
                                                unsigned BuiltinID,
                                                CallExpr *TheCall) {
  unsigned i = 0, l = 0, u = 0;

  switch (BuiltinID) {
  default:
    return false;
  case Xtensa::BI__builtin_xtensa_mul_ad_ll:
  case Xtensa::BI__builtin_xtensa_mul_ad_lh:
  case Xtensa::BI__builtin_xtensa_mul_ad_hl:
  case Xtensa::BI__builtin_xtensa_mul_ad_hh:
  case Xtensa::BI__builtin_xtensa_mula_ad_ll:
  case Xtensa::BI__builtin_xtensa_mula_ad_lh:
  case Xtensa::BI__builtin_xtensa_mula_ad_hl:
  case Xtensa::BI__builtin_xtensa_mula_ad_hh:
  case Xtensa::BI__builtin_xtensa_muls_ad_ll:
  case Xtensa::BI__builtin_xtensa_muls_ad_lh:
  case Xtensa::BI__builtin_xtensa_muls_ad_hl:
  case Xtensa::BI__builtin_xtensa_muls_ad_hh:
    i = 1;
    l = 2;
    u = 3;
    break;
  case Xtensa::BI__builtin_xtensa_mul_da_ll:
  case Xtensa::BI__builtin_xtensa_mul_da_lh:
  case Xtensa::BI__builtin_xtensa_mul_da_hl:
  case Xtensa::BI__builtin_xtensa_mul_da_hh:
  case Xtensa::BI__builtin_xtensa_mula_da_ll:
  case Xtensa::BI__builtin_xtensa_mula_da_lh:
  case Xtensa::BI__builtin_xtensa_mula_da_hl:
  case Xtensa::BI__builtin_xtensa_mula_da_hh:
  case Xtensa::BI__builtin_xtensa_muls_da_ll:
  case Xtensa::BI__builtin_xtensa_muls_da_lh:
  case Xtensa::BI__builtin_xtensa_muls_da_hl:
  case Xtensa::BI__builtin_xtensa_muls_da_hh:
    i = 0;
    l = 0;
    u = 1;
    break;
  case Xtensa::BI__builtin_xtensa_mul_dd_ll:
  case Xtensa::BI__builtin_xtensa_mul_dd_lh:
  case Xtensa::BI__builtin_xtensa_mul_dd_hl:
  case Xtensa::BI__builtin_xtensa_mul_dd_hh:
  case Xtensa::BI__builtin_xtensa_mula_dd_ll:
  case Xtensa::BI__builtin_xtensa_mula_dd_lh:
  case Xtensa::BI__builtin_xtensa_mula_dd_hl:
  case Xtensa::BI__builtin_xtensa_mula_dd_hh:
  case Xtensa::BI__builtin_xtensa_muls_dd_ll:
  case Xtensa::BI__builtin_xtensa_muls_dd_lh:
  case Xtensa::BI__builtin_xtensa_muls_dd_hl:
  case Xtensa::BI__builtin_xtensa_muls_dd_hh:
    return SemaRef.BuiltinConstantArgRange(TheCall, 0, 0, 1) ||
           SemaRef.BuiltinConstantArgRange(TheCall, 1, 2, 3);
  case Xtensa::BI__builtin_xtensa_mula_da_ll_lddec:
  case Xtensa::BI__builtin_xtensa_mula_da_lh_lddec:
  case Xtensa::BI__builtin_xtensa_mula_da_hl_lddec:
  case Xtensa::BI__builtin_xtensa_mula_da_hh_lddec:
  case Xtensa::BI__builtin_xtensa_mula_da_ll_ldinc:
  case Xtensa::BI__builtin_xtensa_mula_da_lh_ldinc:
  case Xtensa::BI__builtin_xtensa_mula_da_hl_ldinc:
  case Xtensa::BI__builtin_xtensa_mula_da_hh_ldinc:
    return SemaRef.BuiltinConstantArgRange(TheCall, 0, 0, 3) ||
           SemaRef.BuiltinConstantArgRange(TheCall, 2, 0, 1);
  case Xtensa::BI__builtin_xtensa_mula_dd_ll_lddec:
  case Xtensa::BI__builtin_xtensa_mula_dd_lh_lddec:
  case Xtensa::BI__builtin_xtensa_mula_dd_hl_lddec:
  case Xtensa::BI__builtin_xtensa_mula_dd_hh_lddec:
  case Xtensa::BI__builtin_xtensa_mula_dd_ll_ldinc:
  case Xtensa::BI__builtin_xtensa_mula_dd_lh_ldinc:
  case Xtensa::BI__builtin_xtensa_mula_dd_hl_ldinc:
  case Xtensa::BI__builtin_xtensa_mula_dd_hh_ldinc:
    return SemaRef.BuiltinConstantArgRange(TheCall, 0, 0, 3) ||
           SemaRef.BuiltinConstantArgRange(TheCall, 2, 0, 1) ||
           SemaRef.BuiltinConstantArgRange(TheCall, 3, 2, 3);
  case Xtensa::BI__builtin_xtensa_xt_trunc_s:
  case Xtensa::BI__builtin_xtensa_xt_utrunc_s:
  case Xtensa::BI__builtin_xtensa_xt_float_s:
  case Xtensa::BI__builtin_xtensa_xt_ufloat_s:
  case Xtensa::BI__builtin_xtensa_xt_ceil_s:
  case Xtensa::BI__builtin_xtensa_xt_floor_s:
  case Xtensa::BI__builtin_xtensa_xt_round_s:
    i = 1;
    l = 0;
    u = 15;
    break;
  case Xtensa::BI__builtin_xtensa_xt_lsi:
  case Xtensa::BI__builtin_xtensa_xt_lsip:
    return SemaRef.BuiltinConstantArgRange(TheCall, 1, 0, 1020) ||
           SemaRef.BuiltinConstantArgMultiple(TheCall, 1, 4);
  case Xtensa::BI__builtin_xtensa_xt_ssi:
  case Xtensa::BI__builtin_xtensa_xt_ssip:
    return SemaRef.BuiltinConstantArgRange(TheCall, 2, 0, 1020) ||
           SemaRef.BuiltinConstantArgMultiple(TheCall, 2, 4);
  case Xtensa::BI__builtin_xtensa_ee_andq:
  case Xtensa::BI__builtin_xtensa_ee_cmul_s16:
  case Xtensa::BI__builtin_xtensa_ee_fft_cmul_s16_st_xp:
  case Xtensa::BI__builtin_xtensa_ee_fft_r2bf_s16_st_incp:
  case Xtensa::BI__builtin_xtensa_ee_orq:
  case Xtensa::BI__builtin_xtensa_ee_src_q:
  case Xtensa::BI__builtin_xtensa_ee_src_q_qup:
  case Xtensa::BI__builtin_xtensa_ee_vadds_s16:
  case Xtensa::BI__builtin_xtensa_ee_vadds_s32:
  case Xtensa::BI__builtin_xtensa_ee_vadds_s8:
  case Xtensa::BI__builtin_xtensa_ee_vcmp_eq_s16:
  case Xtensa::BI__builtin_xtensa_ee_vcmp_eq_s32:
  case Xtensa::BI__builtin_xtensa_ee_vcmp_eq_s8:
  case Xtensa::BI__builtin_xtensa_ee_vcmp_gt_s16:
  case Xtensa::BI__builtin_xtensa_ee_vcmp_gt_s32:
  case Xtensa::BI__builtin_xtensa_ee_vcmp_gt_s8:
  case Xtensa::BI__builtin_xtensa_ee_vcmp_lt_s16:
  case Xtensa::BI__builtin_xtensa_ee_vcmp_lt_s32:
  case Xtensa::BI__builtin_xtensa_ee_vcmp_lt_s8:
  case Xtensa::BI__builtin_xtensa_ee_vmax_s16:
  case Xtensa::BI__builtin_xtensa_ee_vmax_s32:
  case Xtensa::BI__builtin_xtensa_ee_vmax_s8:
  case Xtensa::BI__builtin_xtensa_ee_vmin_s16:
  case Xtensa::BI__builtin_xtensa_ee_vmin_s32:
  case Xtensa::BI__builtin_xtensa_ee_vmin_s8:
  case Xtensa::BI__builtin_xtensa_ee_vmul_s16:
  case Xtensa::BI__builtin_xtensa_ee_vmul_s8:
  case Xtensa::BI__builtin_xtensa_ee_vmul_u16:
  case Xtensa::BI__builtin_xtensa_ee_vmul_u8:
  case Xtensa::BI__builtin_xtensa_ee_vprelu_s16:
  case Xtensa::BI__builtin_xtensa_ee_vprelu_s8:
  case Xtensa::BI__builtin_xtensa_ee_vsubs_s16:
  case Xtensa::BI__builtin_xtensa_ee_vsubs_s32:
  case Xtensa::BI__builtin_xtensa_ee_vsubs_s8:
  case Xtensa::BI__builtin_xtensa_ee_xorq:
    return SemaRef.BuiltinConstantArgRange(TheCall, 0, 0, 7) &&
           SemaRef.BuiltinConstantArgRange(TheCall, 1, 0, 7) &&
           SemaRef.BuiltinConstantArgRange(TheCall, 2, 0, 7);
  case Xtensa::BI__builtin_xtensa_ee_bitrev:
  case Xtensa::BI__builtin_xtensa_ee_fft_vst_r32_decp:
  case Xtensa::BI__builtin_xtensa_ee_ld_128_usar_ip:
  case Xtensa::BI__builtin_xtensa_ee_ld_128_usar_xp:
  case Xtensa::BI__builtin_xtensa_ee_movi_32_a:
  case Xtensa::BI__builtin_xtensa_ee_movi_32_q:
  case Xtensa::BI__builtin_xtensa_ee_mov_s16_qacc:
  case Xtensa::BI__builtin_xtensa_ee_mov_s8_qacc:
  case Xtensa::BI__builtin_xtensa_ee_mov_u16_qacc:
  case Xtensa::BI__builtin_xtensa_ee_mov_u8_qacc:
  case Xtensa::BI__builtin_xtensa_ee_srcmb_s16_qacc:
  case Xtensa::BI__builtin_xtensa_ee_srcmb_s8_qacc:
  case Xtensa::BI__builtin_xtensa_ee_vldbc_16:
  case Xtensa::BI__builtin_xtensa_ee_vldbc_16_ip:
  case Xtensa::BI__builtin_xtensa_ee_vldbc_16_xp:
  case Xtensa::BI__builtin_xtensa_ee_vldbc_32:
  case Xtensa::BI__builtin_xtensa_ee_vldbc_32_ip:
  case Xtensa::BI__builtin_xtensa_ee_vldbc_32_xp:
  case Xtensa::BI__builtin_xtensa_ee_vldbc_8:
  case Xtensa::BI__builtin_xtensa_ee_vldbc_8_ip:
  case Xtensa::BI__builtin_xtensa_ee_vldbc_8_xp:
  case Xtensa::BI__builtin_xtensa_ee_vld_128_ip:
  case Xtensa::BI__builtin_xtensa_ee_vld_128_xp:
  case Xtensa::BI__builtin_xtensa_ee_vld_h_64_ip:
  case Xtensa::BI__builtin_xtensa_ee_vld_h_64_xp:
  case Xtensa::BI__builtin_xtensa_ee_vld_l_64_ip:
  case Xtensa::BI__builtin_xtensa_ee_vld_l_64_xp:
  case Xtensa::BI__builtin_xtensa_ee_vrelu_s16:
  case Xtensa::BI__builtin_xtensa_ee_vrelu_s8:
  case Xtensa::BI__builtin_xtensa_ee_vst_128_ip:
  case Xtensa::BI__builtin_xtensa_ee_vst_128_xp:
  case Xtensa::BI__builtin_xtensa_ee_vst_h_64_ip:
  case Xtensa::BI__builtin_xtensa_ee_vst_h_64_xp:
  case Xtensa::BI__builtin_xtensa_ee_vst_l_64_ip:
  case Xtensa::BI__builtin_xtensa_ee_vst_l_64_xp:
  case Xtensa::BI__builtin_xtensa_ee_zero_q:
    return SemaRef.BuiltinConstantArgRange(TheCall, 0, 0, 7);
  case Xtensa::BI__builtin_xtensa_ee_cmul_s16_ld_incp:
  case Xtensa::BI__builtin_xtensa_ee_cmul_s16_st_incp:
  case Xtensa::BI__builtin_xtensa_ee_vadds_s16_ld_incp:
  case Xtensa::BI__builtin_xtensa_ee_vadds_s16_st_incp:
  case Xtensa::BI__builtin_xtensa_ee_vadds_s32_ld_incp:
  case Xtensa::BI__builtin_xtensa_ee_vadds_s32_st_incp:
  case Xtensa::BI__builtin_xtensa_ee_vadds_s8_ld_incp:
  case Xtensa::BI__builtin_xtensa_ee_vadds_s8_st_incp:
  case Xtensa::BI__builtin_xtensa_ee_vmax_s16_ld_incp:
  case Xtensa::BI__builtin_xtensa_ee_vmax_s16_st_incp:
  case Xtensa::BI__builtin_xtensa_ee_vmax_s32_ld_incp:
  case Xtensa::BI__builtin_xtensa_ee_vmax_s32_st_incp:
  case Xtensa::BI__builtin_xtensa_ee_vmax_s8_ld_incp:
  case Xtensa::BI__builtin_xtensa_ee_vmax_s8_st_incp:
  case Xtensa::BI__builtin_xtensa_ee_vmin_s16_ld_incp:
  case Xtensa::BI__builtin_xtensa_ee_vmin_s16_st_incp:
  case Xtensa::BI__builtin_xtensa_ee_vmin_s32_ld_incp:
  case Xtensa::BI__builtin_xtensa_ee_vmin_s32_st_incp:
  case Xtensa::BI__builtin_xtensa_ee_vmin_s8_ld_incp:
  case Xtensa::BI__builtin_xtensa_ee_vmin_s8_st_incp:
  case Xtensa::BI__builtin_xtensa_ee_vmul_s16_ld_incp:
  case Xtensa::BI__builtin_xtensa_ee_vmul_s16_st_incp:
  case Xtensa::BI__builtin_xtensa_ee_vmul_s8_ld_incp:
  case Xtensa::BI__builtin_xtensa_ee_vmul_s8_st_incp:
  case Xtensa::BI__builtin_xtensa_ee_vmul_u16_ld_incp:
  case Xtensa::BI__builtin_xtensa_ee_vmul_u16_st_incp:
  case Xtensa::BI__builtin_xtensa_ee_vmul_u8_ld_incp:
  case Xtensa::BI__builtin_xtensa_ee_vmul_u8_st_incp:
  case Xtensa::BI__builtin_xtensa_ee_vsubs_s16_ld_incp:
  case Xtensa::BI__builtin_xtensa_ee_vsubs_s16_st_incp:
  case Xtensa::BI__builtin_xtensa_ee_vsubs_s32_ld_incp:
  case Xtensa::BI__builtin_xtensa_ee_vsubs_s32_st_incp:
  case Xtensa::BI__builtin_xtensa_ee_vsubs_s8_ld_incp:
  case Xtensa::BI__builtin_xtensa_ee_vsubs_s8_st_incp:
    return SemaRef.BuiltinConstantArgRange(TheCall, 0, 0, 7) &&
           SemaRef.BuiltinConstantArgRange(TheCall, 2, 0, 7) &&
           SemaRef.BuiltinConstantArgRange(TheCall, 3, 0, 7) &&
           SemaRef.BuiltinConstantArgRange(TheCall, 4, 0, 7);
  case Xtensa::BI__builtin_xtensa_ee_fft_ams_s16_ld_incp:
  case Xtensa::BI__builtin_xtensa_ee_fft_ams_s16_ld_incp_uaup:
  case Xtensa::BI__builtin_xtensa_ee_fft_ams_s16_ld_r32_decp:
    return SemaRef.BuiltinConstantArgRange(TheCall, 0, 0, 7) &&
           SemaRef.BuiltinConstantArgRange(TheCall, 2, 0, 7) &&
           SemaRef.BuiltinConstantArgRange(TheCall, 3, 0, 7) &&
           SemaRef.BuiltinConstantArgRange(TheCall, 4, 0, 7) &&
           SemaRef.BuiltinConstantArgRange(TheCall, 5, 0, 7) &&
           SemaRef.BuiltinConstantArgRange(TheCall, 6, 0, 7);
  case Xtensa::BI__builtin_xtensa_ee_fft_ams_s16_st_incp:
    return SemaRef.BuiltinConstantArgRange(TheCall, 0, 0, 7) &&
           SemaRef.BuiltinConstantArgRange(TheCall, 1, 0, 7) &&
           SemaRef.BuiltinConstantArgRange(TheCall, 4, 0, 7) &&
           SemaRef.BuiltinConstantArgRange(TheCall, 5, 0, 7) &&
           SemaRef.BuiltinConstantArgRange(TheCall, 6, 0, 7);
  case Xtensa::BI__builtin_xtensa_ee_fft_cmul_s16_ld_xp:
    return SemaRef.BuiltinConstantArgRange(TheCall, 0, 0, 7) &&
           SemaRef.BuiltinConstantArgRange(TheCall, 3, 0, 7) &&
           SemaRef.BuiltinConstantArgRange(TheCall, 4, 0, 7) &&
           SemaRef.BuiltinConstantArgRange(TheCall, 5, 0, 7);
  case Xtensa::BI__builtin_xtensa_ee_fft_r2bf_s16:
    return SemaRef.BuiltinConstantArgRange(TheCall, 0, 0, 7) &&
           SemaRef.BuiltinConstantArgRange(TheCall, 1, 0, 7) &&
           SemaRef.BuiltinConstantArgRange(TheCall, 2, 0, 7) &&
           SemaRef.BuiltinConstantArgRange(TheCall, 3, 0, 7);
  case Xtensa::BI__builtin_xtensa_ee_ldxq_32:
  case Xtensa::BI__builtin_xtensa_ee_notq:
  case Xtensa::BI__builtin_xtensa_ee_slci_2q:
  case Xtensa::BI__builtin_xtensa_ee_slcxxp_2q:
  case Xtensa::BI__builtin_xtensa_ee_srci_2q:
  case Xtensa::BI__builtin_xtensa_ee_srcq_128_st_incp:
  case Xtensa::BI__builtin_xtensa_ee_srcxxp_2q:
  case Xtensa::BI__builtin_xtensa_ee_stxq_32:
  case Xtensa::BI__builtin_xtensa_ee_vldhbc_16_incp:
  case Xtensa::BI__builtin_xtensa_ee_vmulas_s16_accx:
  case Xtensa::BI__builtin_xtensa_ee_vmulas_s16_qacc:
  case Xtensa::BI__builtin_xtensa_ee_vmulas_s8_accx:
  case Xtensa::BI__builtin_xtensa_ee_vmulas_s8_qacc:
  case Xtensa::BI__builtin_xtensa_ee_vmulas_u16_accx:
  case Xtensa::BI__builtin_xtensa_ee_vmulas_u16_qacc:
  case Xtensa::BI__builtin_xtensa_ee_vmulas_u8_accx:
  case Xtensa::BI__builtin_xtensa_ee_vmulas_u8_qacc:
  case Xtensa::BI__builtin_xtensa_ee_vsl_32:
  case Xtensa::BI__builtin_xtensa_ee_vsmulas_s16_qacc:
  case Xtensa::BI__builtin_xtensa_ee_vsmulas_s8_qacc:
  case Xtensa::BI__builtin_xtensa_ee_vsr_32:
  case Xtensa::BI__builtin_xtensa_ee_vunzip_16:
  case Xtensa::BI__builtin_xtensa_ee_vunzip_32:
  case Xtensa::BI__builtin_xtensa_ee_vunzip_8:
  case Xtensa::BI__builtin_xtensa_ee_vzip_16:
  case Xtensa::BI__builtin_xtensa_ee_vzip_32:
  case Xtensa::BI__builtin_xtensa_ee_vzip_8:
  case Xtensa::BI__builtin_xtensa_mv_qr:
    return SemaRef.BuiltinConstantArgRange(TheCall, 0, 0, 7) &&
           SemaRef.BuiltinConstantArgRange(TheCall, 1, 0, 7);
  case Xtensa::BI__builtin_xtensa_ee_src_q_ld_ip:
  case Xtensa::BI__builtin_xtensa_ee_src_q_ld_xp:
  case Xtensa::BI__builtin_xtensa_ee_vmulas_s16_accx_ld_ip:
  case Xtensa::BI__builtin_xtensa_ee_vmulas_s16_accx_ld_xp:
  case Xtensa::BI__builtin_xtensa_ee_vmulas_s16_qacc_ld_ip:
  case Xtensa::BI__builtin_xtensa_ee_vmulas_s16_qacc_ld_xp:
  case Xtensa::BI__builtin_xtensa_ee_vmulas_s8_accx_ld_ip:
  case Xtensa::BI__builtin_xtensa_ee_vmulas_s8_accx_ld_xp:
  case Xtensa::BI__builtin_xtensa_ee_vmulas_s8_qacc_ld_ip:
  case Xtensa::BI__builtin_xtensa_ee_vmulas_s8_qacc_ld_xp:
  case Xtensa::BI__builtin_xtensa_ee_vmulas_u16_accx_ld_ip:
  case Xtensa::BI__builtin_xtensa_ee_vmulas_u16_accx_ld_xp:
  case Xtensa::BI__builtin_xtensa_ee_vmulas_u16_qacc_ld_ip:
  case Xtensa::BI__builtin_xtensa_ee_vmulas_u16_qacc_ld_xp:
  case Xtensa::BI__builtin_xtensa_ee_vmulas_u8_accx_ld_ip:
  case Xtensa::BI__builtin_xtensa_ee_vmulas_u8_accx_ld_xp:
  case Xtensa::BI__builtin_xtensa_ee_vmulas_u8_qacc_ld_ip:
  case Xtensa::BI__builtin_xtensa_ee_vmulas_u8_qacc_ld_xp:
    return SemaRef.BuiltinConstantArgRange(TheCall, 0, 0, 7) &&
           SemaRef.BuiltinConstantArgRange(TheCall, 3, 0, 7) &&
           SemaRef.BuiltinConstantArgRange(TheCall, 4, 0, 7);
  case Xtensa::BI__builtin_xtensa_ee_vmulas_s16_accx_ld_ip_qup:
  case Xtensa::BI__builtin_xtensa_ee_vmulas_s16_accx_ld_xp_qup:
  case Xtensa::BI__builtin_xtensa_ee_vmulas_s16_qacc_ld_ip_qup:
  case Xtensa::BI__builtin_xtensa_ee_vmulas_s16_qacc_ld_xp_qup:
  case Xtensa::BI__builtin_xtensa_ee_vmulas_s8_accx_ld_ip_qup:
  case Xtensa::BI__builtin_xtensa_ee_vmulas_s8_accx_ld_xp_qup:
  case Xtensa::BI__builtin_xtensa_ee_vmulas_s8_qacc_ld_ip_qup:
  case Xtensa::BI__builtin_xtensa_ee_vmulas_s8_qacc_ld_xp_qup:
  case Xtensa::BI__builtin_xtensa_ee_vmulas_u16_accx_ld_ip_qup:
  case Xtensa::BI__builtin_xtensa_ee_vmulas_u16_accx_ld_xp_qup:
  case Xtensa::BI__builtin_xtensa_ee_vmulas_u16_qacc_ld_ip_qup:
  case Xtensa::BI__builtin_xtensa_ee_vmulas_u16_qacc_ld_xp_qup:
  case Xtensa::BI__builtin_xtensa_ee_vmulas_u8_accx_ld_ip_qup:
  case Xtensa::BI__builtin_xtensa_ee_vmulas_u8_accx_ld_xp_qup:
  case Xtensa::BI__builtin_xtensa_ee_vmulas_u8_qacc_ld_ip_qup:
  case Xtensa::BI__builtin_xtensa_ee_vmulas_u8_qacc_ld_xp_qup:
    return SemaRef.BuiltinConstantArgRange(TheCall, 0, 0, 7) &&
           SemaRef.BuiltinConstantArgRange(TheCall, 3, 0, 7) &&
           SemaRef.BuiltinConstantArgRange(TheCall, 4, 0, 7) &&
           SemaRef.BuiltinConstantArgRange(TheCall, 5, 0, 7) &&
           SemaRef.BuiltinConstantArgRange(TheCall, 6, 0, 7);
  case Xtensa::BI__builtin_xtensa_ee_vmulas_s16_qacc_ldbc_incp:
  case Xtensa::BI__builtin_xtensa_ee_vmulas_s8_qacc_ldbc_incp:
  case Xtensa::BI__builtin_xtensa_ee_vmulas_u16_qacc_ldbc_incp:
  case Xtensa::BI__builtin_xtensa_ee_vmulas_u8_qacc_ldbc_incp:
  case Xtensa::BI__builtin_xtensa_ee_vsmulas_s16_qacc_ld_incp:
  case Xtensa::BI__builtin_xtensa_ee_vsmulas_s8_qacc_ld_incp:
    return SemaRef.BuiltinConstantArgRange(TheCall, 0, 0, 7) &&
           SemaRef.BuiltinConstantArgRange(TheCall, 2, 0, 7) &&
           SemaRef.BuiltinConstantArgRange(TheCall, 3, 0, 7);
  case Xtensa::BI__builtin_xtensa_ee_vmulas_s16_qacc_ldbc_incp_qup:
  case Xtensa::BI__builtin_xtensa_ee_vmulas_s8_qacc_ldbc_incp_qup:
  case Xtensa::BI__builtin_xtensa_ee_vmulas_u16_qacc_ldbc_incp_qup:
  case Xtensa::BI__builtin_xtensa_ee_vmulas_u8_qacc_ldbc_incp_qup:
    return SemaRef.BuiltinConstantArgRange(TheCall, 0, 0, 7) &&
           SemaRef.BuiltinConstantArgRange(TheCall, 2, 0, 7) &&
           SemaRef.BuiltinConstantArgRange(TheCall, 3, 0, 7) &&
           SemaRef.BuiltinConstantArgRange(TheCall, 4, 0, 7) &&
           SemaRef.BuiltinConstantArgRange(TheCall, 5, 0, 7);
  }
  return SemaRef.BuiltinConstantArgRange(TheCall, i, l, u);
}

} // namespace clang
