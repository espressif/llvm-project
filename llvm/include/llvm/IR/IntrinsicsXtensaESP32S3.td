//===- IntrinsicsXtensa.td - Defines Xtensa intrinsics -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the Xtensa-specific intrinsics.
// Automatically generated file, do not edit!
//===----------------------------------------------------------------------===//

def int_xtensa_ee_andq: ClangBuiltin<"__builtin_xtensa_ee_andq">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_xtensa_ee_bitrev: ClangBuiltin<"__builtin_xtensa_ee_bitrev">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;

def int_xtensa_ee_cmul_s16: ClangBuiltin<"__builtin_xtensa_ee_cmul_s16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_xtensa_ee_cmul_s16_ld_incp: ClangBuiltin<"__builtin_xtensa_ee_cmul_s16_ld_incp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>]>;

def int_xtensa_ee_cmul_s16_st_incp: ClangBuiltin<"__builtin_xtensa_ee_cmul_s16_st_incp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>]>;

def int_xtensa_ee_fft_ams_s16_ld_incp: ClangBuiltin<"__builtin_xtensa_ee_fft_ams_s16_ld_incp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>, ImmArg<ArgIndex<7>>]>;

def int_xtensa_ee_fft_ams_s16_ld_incp_uaup: ClangBuiltin<"__builtin_xtensa_ee_fft_ams_s16_ld_incp_uaup">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>, ImmArg<ArgIndex<7>>]>;

def int_xtensa_ee_fft_ams_s16_ld_r32_decp: ClangBuiltin<"__builtin_xtensa_ee_fft_ams_s16_ld_r32_decp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>, ImmArg<ArgIndex<7>>]>;

def int_xtensa_ee_fft_ams_s16_st_incp: ClangBuiltin<"__builtin_xtensa_ee_fft_ams_s16_st_incp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>, ImmArg<ArgIndex<7>>]>;

def int_xtensa_ee_fft_cmul_s16_ld_xp: ClangBuiltin<"__builtin_xtensa_ee_fft_cmul_s16_ld_xp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>]>;

def int_xtensa_ee_fft_cmul_s16_st_xp: ClangBuiltin<"__builtin_xtensa_ee_fft_cmul_s16_st_xp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>, ImmArg<ArgIndex<7>>]>;

def int_xtensa_ee_fft_r2bf_s16: ClangBuiltin<"__builtin_xtensa_ee_fft_r2bf_s16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_fft_r2bf_s16_st_incp: ClangBuiltin<"__builtin_xtensa_ee_fft_r2bf_s16_st_incp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_fft_vst_r32_decp: ClangBuiltin<"__builtin_xtensa_ee_fft_vst_r32_decp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>]>;

def int_xtensa_ee_ldf_128_ip: ClangBuiltin<"__builtin_xtensa_ee_ldf_128_ip">,
	Intrinsic<[], [llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<5>>]>;

def int_xtensa_ee_ldf_128_xp: ClangBuiltin<"__builtin_xtensa_ee_ldf_128_xp">,
	Intrinsic<[], [llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_i32_ty, llvm_i32_ty], []>;

def int_xtensa_ee_ldf_64_ip: ClangBuiltin<"__builtin_xtensa_ee_ldf_64_ip">,
	Intrinsic<[], [llvm_float_ty, llvm_float_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<3>>]>;

def int_xtensa_ee_ldf_64_xp: ClangBuiltin<"__builtin_xtensa_ee_ldf_64_xp">,
	Intrinsic<[], [llvm_float_ty, llvm_float_ty, llvm_i32_ty, llvm_i32_ty], []>;

def int_xtensa_ee_ldqa_s16_128_ip: ClangBuiltin<"__builtin_xtensa_ee_ldqa_s16_128_ip">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_xtensa_ee_ldqa_s16_128_xp: ClangBuiltin<"__builtin_xtensa_ee_ldqa_s16_128_xp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], []>;

def int_xtensa_ee_ldqa_s8_128_ip: ClangBuiltin<"__builtin_xtensa_ee_ldqa_s8_128_ip">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_xtensa_ee_ldqa_s8_128_xp: ClangBuiltin<"__builtin_xtensa_ee_ldqa_s8_128_xp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], []>;

def int_xtensa_ee_ldqa_u16_128_ip: ClangBuiltin<"__builtin_xtensa_ee_ldqa_u16_128_ip">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_xtensa_ee_ldqa_u16_128_xp: ClangBuiltin<"__builtin_xtensa_ee_ldqa_u16_128_xp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], []>;

def int_xtensa_ee_ldqa_u8_128_ip: ClangBuiltin<"__builtin_xtensa_ee_ldqa_u8_128_ip">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_xtensa_ee_ldqa_u8_128_xp: ClangBuiltin<"__builtin_xtensa_ee_ldqa_u8_128_xp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], []>;

def int_xtensa_ee_ldxq_32: ClangBuiltin<"__builtin_xtensa_ee_ldxq_32">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_ld_128_usar_ip: ClangBuiltin<"__builtin_xtensa_ee_ld_128_usar_ip">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>]>;

def int_xtensa_ee_ld_128_usar_xp: ClangBuiltin<"__builtin_xtensa_ee_ld_128_usar_xp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;

def int_xtensa_ee_ld_accx_ip: ClangBuiltin<"__builtin_xtensa_ee_ld_accx_ip">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_xtensa_ee_ld_qacc_h_h_32_ip: ClangBuiltin<"__builtin_xtensa_ee_ld_qacc_h_h_32_ip">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_xtensa_ee_ld_qacc_h_l_128_ip: ClangBuiltin<"__builtin_xtensa_ee_ld_qacc_h_l_128_ip">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_xtensa_ee_ld_qacc_l_h_32_ip: ClangBuiltin<"__builtin_xtensa_ee_ld_qacc_l_h_32_ip">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_xtensa_ee_ld_qacc_l_l_128_ip: ClangBuiltin<"__builtin_xtensa_ee_ld_qacc_l_l_128_ip">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_xtensa_ee_ld_ua_state_ip: ClangBuiltin<"__builtin_xtensa_ee_ld_ua_state_ip">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_xtensa_ee_movi_32_a: ClangBuiltin<"__builtin_xtensa_ee_movi_32_a">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>]>;

def int_xtensa_ee_movi_32_q: ClangBuiltin<"__builtin_xtensa_ee_movi_32_q">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>]>;

def int_xtensa_ee_mov_s16_qacc: ClangBuiltin<"__builtin_xtensa_ee_mov_s16_qacc">,
	Intrinsic<[], [llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;

def int_xtensa_ee_mov_s8_qacc: ClangBuiltin<"__builtin_xtensa_ee_mov_s8_qacc">,
	Intrinsic<[], [llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;

def int_xtensa_ee_mov_u16_qacc: ClangBuiltin<"__builtin_xtensa_ee_mov_u16_qacc">,
	Intrinsic<[], [llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;

def int_xtensa_ee_mov_u8_qacc: ClangBuiltin<"__builtin_xtensa_ee_mov_u8_qacc">,
	Intrinsic<[], [llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;

def int_xtensa_ee_notq: ClangBuiltin<"__builtin_xtensa_ee_notq">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

def int_xtensa_ee_orq: ClangBuiltin<"__builtin_xtensa_ee_orq">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_xtensa_ee_slci_2q: ClangBuiltin<"__builtin_xtensa_ee_slci_2q">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_xtensa_ee_slcxxp_2q: ClangBuiltin<"__builtin_xtensa_ee_slcxxp_2q">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

def int_xtensa_ee_srci_2q: ClangBuiltin<"__builtin_xtensa_ee_srci_2q">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_xtensa_ee_srcmb_s16_qacc: ClangBuiltin<"__builtin_xtensa_ee_srcmb_s16_qacc">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>]>;

def int_xtensa_ee_srcmb_s8_qacc: ClangBuiltin<"__builtin_xtensa_ee_srcmb_s8_qacc">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>]>;

def int_xtensa_ee_srcq_128_st_incp: ClangBuiltin<"__builtin_xtensa_ee_srcq_128_st_incp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

def int_xtensa_ee_srcxxp_2q: ClangBuiltin<"__builtin_xtensa_ee_srcxxp_2q">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

def int_xtensa_ee_src_q: ClangBuiltin<"__builtin_xtensa_ee_src_q">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_xtensa_ee_src_q_ld_ip: ClangBuiltin<"__builtin_xtensa_ee_src_q_ld_ip">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_src_q_ld_xp: ClangBuiltin<"__builtin_xtensa_ee_src_q_ld_xp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_src_q_qup: ClangBuiltin<"__builtin_xtensa_ee_src_q_qup">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_xtensa_ee_srs_accx: ClangBuiltin<"__builtin_xtensa_ee_srs_accx">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>]>;

def int_xtensa_ee_stf_128_ip: ClangBuiltin<"__builtin_xtensa_ee_stf_128_ip">,
	Intrinsic<[], [llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<5>>]>;

def int_xtensa_ee_stf_128_xp: ClangBuiltin<"__builtin_xtensa_ee_stf_128_xp">,
	Intrinsic<[], [llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_float_ty, llvm_i32_ty, llvm_i32_ty], []>;

def int_xtensa_ee_stf_64_ip: ClangBuiltin<"__builtin_xtensa_ee_stf_64_ip">,
	Intrinsic<[], [llvm_float_ty, llvm_float_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<3>>]>;

def int_xtensa_ee_stf_64_xp: ClangBuiltin<"__builtin_xtensa_ee_stf_64_xp">,
	Intrinsic<[], [llvm_float_ty, llvm_float_ty, llvm_i32_ty, llvm_i32_ty], []>;

def int_xtensa_ee_stxq_32: ClangBuiltin<"__builtin_xtensa_ee_stxq_32">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_st_accx_ip: ClangBuiltin<"__builtin_xtensa_ee_st_accx_ip">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_xtensa_ee_st_qacc_h_h_32_ip: ClangBuiltin<"__builtin_xtensa_ee_st_qacc_h_h_32_ip">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_xtensa_ee_st_qacc_h_l_128_ip: ClangBuiltin<"__builtin_xtensa_ee_st_qacc_h_l_128_ip">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_xtensa_ee_st_qacc_l_h_32_ip: ClangBuiltin<"__builtin_xtensa_ee_st_qacc_l_h_32_ip">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_xtensa_ee_st_qacc_l_l_128_ip: ClangBuiltin<"__builtin_xtensa_ee_st_qacc_l_l_128_ip">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_xtensa_ee_st_ua_state_ip: ClangBuiltin<"__builtin_xtensa_ee_st_ua_state_ip">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_xtensa_ee_vadds_s16: ClangBuiltin<"__builtin_xtensa_ee_vadds_s16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_xtensa_ee_vadds_s16_ld_incp: ClangBuiltin<"__builtin_xtensa_ee_vadds_s16_ld_incp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_vadds_s16_st_incp: ClangBuiltin<"__builtin_xtensa_ee_vadds_s16_st_incp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_vadds_s32: ClangBuiltin<"__builtin_xtensa_ee_vadds_s32">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_xtensa_ee_vadds_s32_ld_incp: ClangBuiltin<"__builtin_xtensa_ee_vadds_s32_ld_incp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_vadds_s32_st_incp: ClangBuiltin<"__builtin_xtensa_ee_vadds_s32_st_incp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_vadds_s8: ClangBuiltin<"__builtin_xtensa_ee_vadds_s8">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_xtensa_ee_vadds_s8_ld_incp: ClangBuiltin<"__builtin_xtensa_ee_vadds_s8_ld_incp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_vadds_s8_st_incp: ClangBuiltin<"__builtin_xtensa_ee_vadds_s8_st_incp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_vcmp_eq_s16: ClangBuiltin<"__builtin_xtensa_ee_vcmp_eq_s16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_xtensa_ee_vcmp_eq_s32: ClangBuiltin<"__builtin_xtensa_ee_vcmp_eq_s32">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_xtensa_ee_vcmp_eq_s8: ClangBuiltin<"__builtin_xtensa_ee_vcmp_eq_s8">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_xtensa_ee_vcmp_gt_s16: ClangBuiltin<"__builtin_xtensa_ee_vcmp_gt_s16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_xtensa_ee_vcmp_gt_s32: ClangBuiltin<"__builtin_xtensa_ee_vcmp_gt_s32">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_xtensa_ee_vcmp_gt_s8: ClangBuiltin<"__builtin_xtensa_ee_vcmp_gt_s8">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_xtensa_ee_vcmp_lt_s16: ClangBuiltin<"__builtin_xtensa_ee_vcmp_lt_s16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_xtensa_ee_vcmp_lt_s32: ClangBuiltin<"__builtin_xtensa_ee_vcmp_lt_s32">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_xtensa_ee_vcmp_lt_s8: ClangBuiltin<"__builtin_xtensa_ee_vcmp_lt_s8">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_xtensa_ee_vldbc_16: ClangBuiltin<"__builtin_xtensa_ee_vldbc_16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;

def int_xtensa_ee_vldbc_16_ip: ClangBuiltin<"__builtin_xtensa_ee_vldbc_16_ip">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>]>;

def int_xtensa_ee_vldbc_16_xp: ClangBuiltin<"__builtin_xtensa_ee_vldbc_16_xp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;

def int_xtensa_ee_vldbc_32: ClangBuiltin<"__builtin_xtensa_ee_vldbc_32">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;

def int_xtensa_ee_vldbc_32_ip: ClangBuiltin<"__builtin_xtensa_ee_vldbc_32_ip">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>]>;

def int_xtensa_ee_vldbc_32_xp: ClangBuiltin<"__builtin_xtensa_ee_vldbc_32_xp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;

def int_xtensa_ee_vldbc_8: ClangBuiltin<"__builtin_xtensa_ee_vldbc_8">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;

def int_xtensa_ee_vldbc_8_ip: ClangBuiltin<"__builtin_xtensa_ee_vldbc_8_ip">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>]>;

def int_xtensa_ee_vldbc_8_xp: ClangBuiltin<"__builtin_xtensa_ee_vldbc_8_xp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;

def int_xtensa_ee_vldhbc_16_incp: ClangBuiltin<"__builtin_xtensa_ee_vldhbc_16_incp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

def int_xtensa_ee_vld_128_ip: ClangBuiltin<"__builtin_xtensa_ee_vld_128_ip">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>]>;

def int_xtensa_ee_vld_128_xp: ClangBuiltin<"__builtin_xtensa_ee_vld_128_xp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;

def int_xtensa_ee_vld_h_64_ip: ClangBuiltin<"__builtin_xtensa_ee_vld_h_64_ip">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>]>;

def int_xtensa_ee_vld_h_64_xp: ClangBuiltin<"__builtin_xtensa_ee_vld_h_64_xp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;

def int_xtensa_ee_vld_l_64_ip: ClangBuiltin<"__builtin_xtensa_ee_vld_l_64_ip">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>]>;

def int_xtensa_ee_vld_l_64_xp: ClangBuiltin<"__builtin_xtensa_ee_vld_l_64_xp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;

def int_xtensa_ee_vmax_s16: ClangBuiltin<"__builtin_xtensa_ee_vmax_s16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_xtensa_ee_vmax_s16_ld_incp: ClangBuiltin<"__builtin_xtensa_ee_vmax_s16_ld_incp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_vmax_s16_st_incp: ClangBuiltin<"__builtin_xtensa_ee_vmax_s16_st_incp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_vmax_s32: ClangBuiltin<"__builtin_xtensa_ee_vmax_s32">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_xtensa_ee_vmax_s32_ld_incp: ClangBuiltin<"__builtin_xtensa_ee_vmax_s32_ld_incp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_vmax_s32_st_incp: ClangBuiltin<"__builtin_xtensa_ee_vmax_s32_st_incp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_vmax_s8: ClangBuiltin<"__builtin_xtensa_ee_vmax_s8">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_xtensa_ee_vmax_s8_ld_incp: ClangBuiltin<"__builtin_xtensa_ee_vmax_s8_ld_incp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_vmax_s8_st_incp: ClangBuiltin<"__builtin_xtensa_ee_vmax_s8_st_incp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_vmin_s16: ClangBuiltin<"__builtin_xtensa_ee_vmin_s16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_xtensa_ee_vmin_s16_ld_incp: ClangBuiltin<"__builtin_xtensa_ee_vmin_s16_ld_incp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_vmin_s16_st_incp: ClangBuiltin<"__builtin_xtensa_ee_vmin_s16_st_incp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_vmin_s32: ClangBuiltin<"__builtin_xtensa_ee_vmin_s32">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_xtensa_ee_vmin_s32_ld_incp: ClangBuiltin<"__builtin_xtensa_ee_vmin_s32_ld_incp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_vmin_s32_st_incp: ClangBuiltin<"__builtin_xtensa_ee_vmin_s32_st_incp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_vmin_s8: ClangBuiltin<"__builtin_xtensa_ee_vmin_s8">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_xtensa_ee_vmin_s8_ld_incp: ClangBuiltin<"__builtin_xtensa_ee_vmin_s8_ld_incp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_vmin_s8_st_incp: ClangBuiltin<"__builtin_xtensa_ee_vmin_s8_st_incp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_vmulas_s16_accx: ClangBuiltin<"__builtin_xtensa_ee_vmulas_s16_accx">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

def int_xtensa_ee_vmulas_s16_accx_ld_ip: ClangBuiltin<"__builtin_xtensa_ee_vmulas_s16_accx_ld_ip">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_vmulas_s16_accx_ld_ip_qup: ClangBuiltin<"__builtin_xtensa_ee_vmulas_s16_accx_ld_ip_qup">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>]>;

def int_xtensa_ee_vmulas_s16_accx_ld_xp: ClangBuiltin<"__builtin_xtensa_ee_vmulas_s16_accx_ld_xp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_vmulas_s16_accx_ld_xp_qup: ClangBuiltin<"__builtin_xtensa_ee_vmulas_s16_accx_ld_xp_qup">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>]>;

def int_xtensa_ee_vmulas_s16_qacc: ClangBuiltin<"__builtin_xtensa_ee_vmulas_s16_qacc">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

def int_xtensa_ee_vmulas_s16_qacc_ldbc_incp: ClangBuiltin<"__builtin_xtensa_ee_vmulas_s16_qacc_ldbc_incp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_xtensa_ee_vmulas_s16_qacc_ldbc_incp_qup: ClangBuiltin<"__builtin_xtensa_ee_vmulas_s16_qacc_ldbc_incp_qup">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>]>;

def int_xtensa_ee_vmulas_s16_qacc_ld_ip: ClangBuiltin<"__builtin_xtensa_ee_vmulas_s16_qacc_ld_ip">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_vmulas_s16_qacc_ld_ip_qup: ClangBuiltin<"__builtin_xtensa_ee_vmulas_s16_qacc_ld_ip_qup">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>]>;

def int_xtensa_ee_vmulas_s16_qacc_ld_xp: ClangBuiltin<"__builtin_xtensa_ee_vmulas_s16_qacc_ld_xp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_vmulas_s16_qacc_ld_xp_qup: ClangBuiltin<"__builtin_xtensa_ee_vmulas_s16_qacc_ld_xp_qup">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>]>;

def int_xtensa_ee_vmulas_s8_accx: ClangBuiltin<"__builtin_xtensa_ee_vmulas_s8_accx">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

def int_xtensa_ee_vmulas_s8_accx_ld_ip: ClangBuiltin<"__builtin_xtensa_ee_vmulas_s8_accx_ld_ip">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_vmulas_s8_accx_ld_ip_qup: ClangBuiltin<"__builtin_xtensa_ee_vmulas_s8_accx_ld_ip_qup">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>]>;

def int_xtensa_ee_vmulas_s8_accx_ld_xp: ClangBuiltin<"__builtin_xtensa_ee_vmulas_s8_accx_ld_xp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_vmulas_s8_accx_ld_xp_qup: ClangBuiltin<"__builtin_xtensa_ee_vmulas_s8_accx_ld_xp_qup">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>]>;

def int_xtensa_ee_vmulas_s8_qacc: ClangBuiltin<"__builtin_xtensa_ee_vmulas_s8_qacc">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

def int_xtensa_ee_vmulas_s8_qacc_ldbc_incp: ClangBuiltin<"__builtin_xtensa_ee_vmulas_s8_qacc_ldbc_incp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_xtensa_ee_vmulas_s8_qacc_ldbc_incp_qup: ClangBuiltin<"__builtin_xtensa_ee_vmulas_s8_qacc_ldbc_incp_qup">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>]>;

def int_xtensa_ee_vmulas_s8_qacc_ld_ip: ClangBuiltin<"__builtin_xtensa_ee_vmulas_s8_qacc_ld_ip">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_vmulas_s8_qacc_ld_ip_qup: ClangBuiltin<"__builtin_xtensa_ee_vmulas_s8_qacc_ld_ip_qup">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>]>;

def int_xtensa_ee_vmulas_s8_qacc_ld_xp: ClangBuiltin<"__builtin_xtensa_ee_vmulas_s8_qacc_ld_xp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_vmulas_s8_qacc_ld_xp_qup: ClangBuiltin<"__builtin_xtensa_ee_vmulas_s8_qacc_ld_xp_qup">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>]>;

def int_xtensa_ee_vmulas_u16_accx: ClangBuiltin<"__builtin_xtensa_ee_vmulas_u16_accx">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

def int_xtensa_ee_vmulas_u16_accx_ld_ip: ClangBuiltin<"__builtin_xtensa_ee_vmulas_u16_accx_ld_ip">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_vmulas_u16_accx_ld_ip_qup: ClangBuiltin<"__builtin_xtensa_ee_vmulas_u16_accx_ld_ip_qup">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>]>;

def int_xtensa_ee_vmulas_u16_accx_ld_xp: ClangBuiltin<"__builtin_xtensa_ee_vmulas_u16_accx_ld_xp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_vmulas_u16_accx_ld_xp_qup: ClangBuiltin<"__builtin_xtensa_ee_vmulas_u16_accx_ld_xp_qup">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>]>;

def int_xtensa_ee_vmulas_u16_qacc: ClangBuiltin<"__builtin_xtensa_ee_vmulas_u16_qacc">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

def int_xtensa_ee_vmulas_u16_qacc_ldbc_incp: ClangBuiltin<"__builtin_xtensa_ee_vmulas_u16_qacc_ldbc_incp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_xtensa_ee_vmulas_u16_qacc_ldbc_incp_qup: ClangBuiltin<"__builtin_xtensa_ee_vmulas_u16_qacc_ldbc_incp_qup">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>]>;

def int_xtensa_ee_vmulas_u16_qacc_ld_ip: ClangBuiltin<"__builtin_xtensa_ee_vmulas_u16_qacc_ld_ip">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_vmulas_u16_qacc_ld_ip_qup: ClangBuiltin<"__builtin_xtensa_ee_vmulas_u16_qacc_ld_ip_qup">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>]>;

def int_xtensa_ee_vmulas_u16_qacc_ld_xp: ClangBuiltin<"__builtin_xtensa_ee_vmulas_u16_qacc_ld_xp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_vmulas_u16_qacc_ld_xp_qup: ClangBuiltin<"__builtin_xtensa_ee_vmulas_u16_qacc_ld_xp_qup">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>]>;

def int_xtensa_ee_vmulas_u8_accx: ClangBuiltin<"__builtin_xtensa_ee_vmulas_u8_accx">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

def int_xtensa_ee_vmulas_u8_accx_ld_ip: ClangBuiltin<"__builtin_xtensa_ee_vmulas_u8_accx_ld_ip">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_vmulas_u8_accx_ld_ip_qup: ClangBuiltin<"__builtin_xtensa_ee_vmulas_u8_accx_ld_ip_qup">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>]>;

def int_xtensa_ee_vmulas_u8_accx_ld_xp: ClangBuiltin<"__builtin_xtensa_ee_vmulas_u8_accx_ld_xp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_vmulas_u8_accx_ld_xp_qup: ClangBuiltin<"__builtin_xtensa_ee_vmulas_u8_accx_ld_xp_qup">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>]>;

def int_xtensa_ee_vmulas_u8_qacc: ClangBuiltin<"__builtin_xtensa_ee_vmulas_u8_qacc">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

def int_xtensa_ee_vmulas_u8_qacc_ldbc_incp: ClangBuiltin<"__builtin_xtensa_ee_vmulas_u8_qacc_ldbc_incp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_xtensa_ee_vmulas_u8_qacc_ldbc_incp_qup: ClangBuiltin<"__builtin_xtensa_ee_vmulas_u8_qacc_ldbc_incp_qup">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>]>;

def int_xtensa_ee_vmulas_u8_qacc_ld_ip: ClangBuiltin<"__builtin_xtensa_ee_vmulas_u8_qacc_ld_ip">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_vmulas_u8_qacc_ld_ip_qup: ClangBuiltin<"__builtin_xtensa_ee_vmulas_u8_qacc_ld_ip_qup">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>]>;

def int_xtensa_ee_vmulas_u8_qacc_ld_xp: ClangBuiltin<"__builtin_xtensa_ee_vmulas_u8_qacc_ld_xp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_vmulas_u8_qacc_ld_xp_qup: ClangBuiltin<"__builtin_xtensa_ee_vmulas_u8_qacc_ld_xp_qup">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>]>;

def int_xtensa_ee_vmul_s16: ClangBuiltin<"__builtin_xtensa_ee_vmul_s16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_xtensa_ee_vmul_s16_ld_incp: ClangBuiltin<"__builtin_xtensa_ee_vmul_s16_ld_incp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_vmul_s16_st_incp: ClangBuiltin<"__builtin_xtensa_ee_vmul_s16_st_incp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_vmul_s8: ClangBuiltin<"__builtin_xtensa_ee_vmul_s8">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_xtensa_ee_vmul_s8_ld_incp: ClangBuiltin<"__builtin_xtensa_ee_vmul_s8_ld_incp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_vmul_s8_st_incp: ClangBuiltin<"__builtin_xtensa_ee_vmul_s8_st_incp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_vmul_u16: ClangBuiltin<"__builtin_xtensa_ee_vmul_u16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_xtensa_ee_vmul_u16_ld_incp: ClangBuiltin<"__builtin_xtensa_ee_vmul_u16_ld_incp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_vmul_u16_st_incp: ClangBuiltin<"__builtin_xtensa_ee_vmul_u16_st_incp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_vmul_u8: ClangBuiltin<"__builtin_xtensa_ee_vmul_u8">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_xtensa_ee_vmul_u8_ld_incp: ClangBuiltin<"__builtin_xtensa_ee_vmul_u8_ld_incp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_vmul_u8_st_incp: ClangBuiltin<"__builtin_xtensa_ee_vmul_u8_st_incp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_vprelu_s16: ClangBuiltin<"__builtin_xtensa_ee_vprelu_s16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_xtensa_ee_vprelu_s8: ClangBuiltin<"__builtin_xtensa_ee_vprelu_s8">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_xtensa_ee_vrelu_s16: ClangBuiltin<"__builtin_xtensa_ee_vrelu_s16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;

def int_xtensa_ee_vrelu_s8: ClangBuiltin<"__builtin_xtensa_ee_vrelu_s8">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;

def int_xtensa_ee_vsl_32: ClangBuiltin<"__builtin_xtensa_ee_vsl_32">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

def int_xtensa_ee_vsmulas_s16_qacc: ClangBuiltin<"__builtin_xtensa_ee_vsmulas_s16_qacc">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_xtensa_ee_vsmulas_s16_qacc_ld_incp: ClangBuiltin<"__builtin_xtensa_ee_vsmulas_s16_qacc_ld_incp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_vsmulas_s8_qacc: ClangBuiltin<"__builtin_xtensa_ee_vsmulas_s8_qacc">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_xtensa_ee_vsmulas_s8_qacc_ld_incp: ClangBuiltin<"__builtin_xtensa_ee_vsmulas_s8_qacc_ld_incp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_vsr_32: ClangBuiltin<"__builtin_xtensa_ee_vsr_32">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

def int_xtensa_ee_vst_128_ip: ClangBuiltin<"__builtin_xtensa_ee_vst_128_ip">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>]>;

def int_xtensa_ee_vst_128_xp: ClangBuiltin<"__builtin_xtensa_ee_vst_128_xp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;

def int_xtensa_ee_vst_h_64_ip: ClangBuiltin<"__builtin_xtensa_ee_vst_h_64_ip">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>]>;

def int_xtensa_ee_vst_h_64_xp: ClangBuiltin<"__builtin_xtensa_ee_vst_h_64_xp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;

def int_xtensa_ee_vst_l_64_ip: ClangBuiltin<"__builtin_xtensa_ee_vst_l_64_ip">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>]>;

def int_xtensa_ee_vst_l_64_xp: ClangBuiltin<"__builtin_xtensa_ee_vst_l_64_xp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;

def int_xtensa_ee_vsubs_s16: ClangBuiltin<"__builtin_xtensa_ee_vsubs_s16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_xtensa_ee_vsubs_s16_ld_incp: ClangBuiltin<"__builtin_xtensa_ee_vsubs_s16_ld_incp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_vsubs_s16_st_incp: ClangBuiltin<"__builtin_xtensa_ee_vsubs_s16_st_incp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_vsubs_s32: ClangBuiltin<"__builtin_xtensa_ee_vsubs_s32">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_xtensa_ee_vsubs_s32_ld_incp: ClangBuiltin<"__builtin_xtensa_ee_vsubs_s32_ld_incp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_vsubs_s32_st_incp: ClangBuiltin<"__builtin_xtensa_ee_vsubs_s32_st_incp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_vsubs_s8: ClangBuiltin<"__builtin_xtensa_ee_vsubs_s8">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_xtensa_ee_vsubs_s8_ld_incp: ClangBuiltin<"__builtin_xtensa_ee_vsubs_s8_ld_incp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_vsubs_s8_st_incp: ClangBuiltin<"__builtin_xtensa_ee_vsubs_s8_st_incp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_xtensa_ee_vunzip_16: ClangBuiltin<"__builtin_xtensa_ee_vunzip_16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

def int_xtensa_ee_vunzip_32: ClangBuiltin<"__builtin_xtensa_ee_vunzip_32">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

def int_xtensa_ee_vunzip_8: ClangBuiltin<"__builtin_xtensa_ee_vunzip_8">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

def int_xtensa_ee_vzip_16: ClangBuiltin<"__builtin_xtensa_ee_vzip_16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

def int_xtensa_ee_vzip_32: ClangBuiltin<"__builtin_xtensa_ee_vzip_32">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

def int_xtensa_ee_vzip_8: ClangBuiltin<"__builtin_xtensa_ee_vzip_8">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

def int_xtensa_ee_xorq: ClangBuiltin<"__builtin_xtensa_ee_xorq">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_xtensa_ee_zero_accx: ClangBuiltin<"__builtin_xtensa_ee_zero_accx">,
	Intrinsic<[], [], []>;

def int_xtensa_ee_zero_q: ClangBuiltin<"__builtin_xtensa_ee_zero_q">,
	Intrinsic<[], [llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;

def int_xtensa_ee_zero_qacc: ClangBuiltin<"__builtin_xtensa_ee_zero_qacc">,
	Intrinsic<[], [], []>;

def int_xtensa_rur_accx_0: ClangBuiltin<"__builtin_xtensa_rur_accx_0">,
	Intrinsic<[llvm_i32_ty], [], []>;

def int_xtensa_rur_accx_1: ClangBuiltin<"__builtin_xtensa_rur_accx_1">,
	Intrinsic<[llvm_i32_ty], [], []>;

def int_xtensa_rur_fft_bit_width: ClangBuiltin<"__builtin_xtensa_rur_fft_bit_width">,
	Intrinsic<[llvm_i32_ty], [], []>;

def int_xtensa_rur_gpio_out: ClangBuiltin<"__builtin_xtensa_rur_gpio_out">,
	Intrinsic<[llvm_i32_ty], [], []>;

def int_xtensa_rur_qacc_h_0: ClangBuiltin<"__builtin_xtensa_rur_qacc_h_0">,
	Intrinsic<[llvm_i32_ty], [], []>;

def int_xtensa_rur_qacc_h_1: ClangBuiltin<"__builtin_xtensa_rur_qacc_h_1">,
	Intrinsic<[llvm_i32_ty], [], []>;

def int_xtensa_rur_qacc_h_2: ClangBuiltin<"__builtin_xtensa_rur_qacc_h_2">,
	Intrinsic<[llvm_i32_ty], [], []>;

def int_xtensa_rur_qacc_h_3: ClangBuiltin<"__builtin_xtensa_rur_qacc_h_3">,
	Intrinsic<[llvm_i32_ty], [], []>;

def int_xtensa_rur_qacc_h_4: ClangBuiltin<"__builtin_xtensa_rur_qacc_h_4">,
	Intrinsic<[llvm_i32_ty], [], []>;

def int_xtensa_rur_qacc_l_0: ClangBuiltin<"__builtin_xtensa_rur_qacc_l_0">,
	Intrinsic<[llvm_i32_ty], [], []>;

def int_xtensa_rur_qacc_l_1: ClangBuiltin<"__builtin_xtensa_rur_qacc_l_1">,
	Intrinsic<[llvm_i32_ty], [], []>;

def int_xtensa_rur_qacc_l_2: ClangBuiltin<"__builtin_xtensa_rur_qacc_l_2">,
	Intrinsic<[llvm_i32_ty], [], []>;

def int_xtensa_rur_qacc_l_3: ClangBuiltin<"__builtin_xtensa_rur_qacc_l_3">,
	Intrinsic<[llvm_i32_ty], [], []>;

def int_xtensa_rur_qacc_l_4: ClangBuiltin<"__builtin_xtensa_rur_qacc_l_4">,
	Intrinsic<[llvm_i32_ty], [], []>;

def int_xtensa_rur_sar_byte: ClangBuiltin<"__builtin_xtensa_rur_sar_byte">,
	Intrinsic<[llvm_i32_ty], [], []>;

def int_xtensa_rur_ua_state_0: ClangBuiltin<"__builtin_xtensa_rur_ua_state_0">,
	Intrinsic<[llvm_i32_ty], [], []>;

def int_xtensa_rur_ua_state_1: ClangBuiltin<"__builtin_xtensa_rur_ua_state_1">,
	Intrinsic<[llvm_i32_ty], [], []>;

def int_xtensa_rur_ua_state_2: ClangBuiltin<"__builtin_xtensa_rur_ua_state_2">,
	Intrinsic<[llvm_i32_ty], [], []>;

def int_xtensa_rur_ua_state_3: ClangBuiltin<"__builtin_xtensa_rur_ua_state_3">,
	Intrinsic<[llvm_i32_ty], [], []>;

def int_xtensa_wur_accx_0: ClangBuiltin<"__builtin_xtensa_wur_accx_0">,
	Intrinsic<[], [llvm_i32_ty], []>;

def int_xtensa_wur_accx_1: ClangBuiltin<"__builtin_xtensa_wur_accx_1">,
	Intrinsic<[], [llvm_i32_ty], []>;

def int_xtensa_wur_fft_bit_width: ClangBuiltin<"__builtin_xtensa_wur_fft_bit_width">,
	Intrinsic<[], [llvm_i32_ty], []>;

def int_xtensa_wur_gpio_out: ClangBuiltin<"__builtin_xtensa_wur_gpio_out">,
	Intrinsic<[], [llvm_i32_ty], []>;

def int_xtensa_wur_qacc_h_0: ClangBuiltin<"__builtin_xtensa_wur_qacc_h_0">,
	Intrinsic<[], [llvm_i32_ty], []>;

def int_xtensa_wur_qacc_h_1: ClangBuiltin<"__builtin_xtensa_wur_qacc_h_1">,
	Intrinsic<[], [llvm_i32_ty], []>;

def int_xtensa_wur_qacc_h_2: ClangBuiltin<"__builtin_xtensa_wur_qacc_h_2">,
	Intrinsic<[], [llvm_i32_ty], []>;

def int_xtensa_wur_qacc_h_3: ClangBuiltin<"__builtin_xtensa_wur_qacc_h_3">,
	Intrinsic<[], [llvm_i32_ty], []>;

def int_xtensa_wur_qacc_h_4: ClangBuiltin<"__builtin_xtensa_wur_qacc_h_4">,
	Intrinsic<[], [llvm_i32_ty], []>;

def int_xtensa_wur_qacc_l_0: ClangBuiltin<"__builtin_xtensa_wur_qacc_l_0">,
	Intrinsic<[], [llvm_i32_ty], []>;

def int_xtensa_wur_qacc_l_1: ClangBuiltin<"__builtin_xtensa_wur_qacc_l_1">,
	Intrinsic<[], [llvm_i32_ty], []>;

def int_xtensa_wur_qacc_l_2: ClangBuiltin<"__builtin_xtensa_wur_qacc_l_2">,
	Intrinsic<[], [llvm_i32_ty], []>;

def int_xtensa_wur_qacc_l_3: ClangBuiltin<"__builtin_xtensa_wur_qacc_l_3">,
	Intrinsic<[], [llvm_i32_ty], []>;

def int_xtensa_wur_qacc_l_4: ClangBuiltin<"__builtin_xtensa_wur_qacc_l_4">,
	Intrinsic<[], [llvm_i32_ty], []>;

def int_xtensa_wur_sar_byte: ClangBuiltin<"__builtin_xtensa_wur_sar_byte">,
	Intrinsic<[], [llvm_i32_ty], []>;

def int_xtensa_wur_ua_state_0: ClangBuiltin<"__builtin_xtensa_wur_ua_state_0">,
	Intrinsic<[], [llvm_i32_ty], []>;

def int_xtensa_wur_ua_state_1: ClangBuiltin<"__builtin_xtensa_wur_ua_state_1">,
	Intrinsic<[], [llvm_i32_ty], []>;

def int_xtensa_wur_ua_state_2: ClangBuiltin<"__builtin_xtensa_wur_ua_state_2">,
	Intrinsic<[], [llvm_i32_ty], []>;

def int_xtensa_wur_ua_state_3: ClangBuiltin<"__builtin_xtensa_wur_ua_state_3">,
	Intrinsic<[], [llvm_i32_ty], []>;

def int_xtensa_mv_qr: ClangBuiltin<"__builtin_xtensa_mv_qr">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;