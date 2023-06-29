//===- IntrinsicsXtensaHIFI.td - Defines Xtensa HIFI intrinsics -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the intrinsics for Xtensa HIFI extension.
//
//===----------------------------------------------------------------------===//

def int_xtensa_ae_abs16s: ClangBuiltin<"__builtin_xtensa_ae_abs16s">,
  Intrinsic<[llvm_v4i16_ty], [llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_abs24s: ClangBuiltin<"__builtin_xtensa_ae_abs24s">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_abs32: ClangBuiltin<"__builtin_xtensa_ae_abs32">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_abs32s: ClangBuiltin<"__builtin_xtensa_ae_abs32s">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_abs64: ClangBuiltin<"__builtin_xtensa_ae_abs64">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty], [IntrNoMem]>;

def int_xtensa_ae_abs64s: ClangBuiltin<"__builtin_xtensa_ae_abs64s">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty], [IntrNoMem]>;

def int_xtensa_ae_add16: ClangBuiltin<"__builtin_xtensa_ae_add16">,
  Intrinsic<[llvm_v4i16_ty], [llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_add16s: ClangBuiltin<"__builtin_xtensa_ae_add16s">,
  Intrinsic<[llvm_v4i16_ty], [llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_add24s: ClangBuiltin<"__builtin_xtensa_ae_add24s">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_add32: ClangBuiltin<"__builtin_xtensa_ae_add32">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_add32_hl_lh: ClangBuiltin<"__builtin_xtensa_ae_add32_hl_lh">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_add32s: ClangBuiltin<"__builtin_xtensa_ae_add32s">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_add64: ClangBuiltin<"__builtin_xtensa_ae_add64">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v1i64_ty], [IntrNoMem]>;

def int_xtensa_ae_add64s: ClangBuiltin<"__builtin_xtensa_ae_add64s">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v1i64_ty], [IntrNoMem]>;

def int_xtensa_ae_addbrba32: ClangBuiltin<"__builtin_xtensa_ae_addbrba32">,
  Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;

def int_xtensa_ae_addsub32: ClangBuiltin<"__builtin_xtensa_ae_addsub32">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_addsub32s: ClangBuiltin<"__builtin_xtensa_ae_addsub32s">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_and: ClangBuiltin<"__builtin_xtensa_ae_and">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v1i64_ty], [IntrNoMem]>;

def int_xtensa_ae_cvt32x2f16_10: ClangBuiltin<"__builtin_xtensa_ae_cvt32x2f16_10">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_cvt32x2f16_32: ClangBuiltin<"__builtin_xtensa_ae_cvt32x2f16_32">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_cvt48a32: ClangBuiltin<"__builtin_xtensa_ae_cvt48a32">,
  Intrinsic<[llvm_v1i64_ty], [llvm_i32_ty], [IntrNoMem]>;

def int_xtensa_ae_cvt64a32: ClangBuiltin<"__builtin_xtensa_ae_cvt64a32">,
  Intrinsic<[llvm_v1i64_ty], [llvm_i32_ty], [IntrNoMem]>;

def int_xtensa_ae_cvt64f32_h: ClangBuiltin<"__builtin_xtensa_ae_cvt64f32_h">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_cvta32f24s_h: ClangBuiltin<"__builtin_xtensa_ae_cvta32f24s_h">,
  Intrinsic<[llvm_i32_ty], [llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_cvta32f24s_l: ClangBuiltin<"__builtin_xtensa_ae_cvta32f24s_l">,
  Intrinsic<[llvm_i32_ty], [llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_cvtq56a32s: ClangBuiltin<"__builtin_xtensa_ae_cvtq56a32s">,
  Intrinsic<[llvm_v1i64_ty], [llvm_i32_ty], [IntrNoMem]>;

def int_xtensa_ae_cvtq56p32s_h: ClangBuiltin<"__builtin_xtensa_ae_cvtq56p32s_h">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_cvtq56p32s_l: ClangBuiltin<"__builtin_xtensa_ae_cvtq56p32s_l">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_db: 
  Intrinsic<[llvm_ptr_ty], [llvm_ptr_ty, llvm_i32_ty], [IntrNoMem]>;

def int_xtensa_ae_db_ic: 
  Intrinsic<[llvm_ptr_ty], [llvm_ptr_ty, llvm_i32_ty], [IntrNoMem]>;

def int_xtensa_ae_db_ip: 
  Intrinsic<[llvm_ptr_ty], [llvm_ptr_ty, llvm_i32_ty], [IntrNoMem]>;

def int_xtensa_ae_dbi: 
  Intrinsic<[llvm_ptr_ty], [llvm_ptr_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, IntrNoMem]>;

def int_xtensa_ae_dbi_ic: 
  Intrinsic<[llvm_ptr_ty], [llvm_ptr_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, IntrNoMem]>;

def int_xtensa_ae_dbi_ip: 
  Intrinsic<[llvm_ptr_ty], [llvm_ptr_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, IntrNoMem]>;

def int_xtensa_ae_div64d32_h: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_div64d32_l: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_eq16: 
  Intrinsic<[llvm_v4i1_ty], [llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_eq32: 
  Intrinsic<[llvm_v2i1_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_eq64: 
  Intrinsic<[llvm_v1i1_ty], [llvm_v1i64_ty, llvm_v1i64_ty], [IntrNoMem]>;

def int_xtensa_ae_l16_i: ClangBuiltin<"__builtin_xtensa_ae_l16_i">,
  Intrinsic<[llvm_v4i16_ty], [llvm_ptr_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_xtensa_ae_l16_ip: 
  Intrinsic<[llvm_v4i16_ty, llvm_ptr_ty], [llvm_ptr_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_xtensa_ae_l16_x: ClangBuiltin<"__builtin_xtensa_ae_l16_x">,
  Intrinsic<[llvm_v4i16_ty], [llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_l16_xc: 
  Intrinsic<[llvm_v4i16_ty, llvm_ptr_ty], [llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_l16_xp: 
  Intrinsic<[llvm_v4i16_ty, llvm_ptr_ty], [llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_l16m_i: ClangBuiltin<"__builtin_xtensa_ae_l16m_i">,
  Intrinsic<[llvm_v2i32_ty], [llvm_ptr_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_xtensa_ae_l16m_iu: 
  Intrinsic<[llvm_v2i32_ty, llvm_ptr_ty], [llvm_ptr_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_xtensa_ae_l16m_x: ClangBuiltin<"__builtin_xtensa_ae_l16m_x">,
  Intrinsic<[llvm_v2i32_ty], [llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_l16m_xc: 
  Intrinsic<[llvm_v2i32_ty, llvm_ptr_ty], [llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_l16m_xu: 
  Intrinsic<[llvm_v2i32_ty, llvm_ptr_ty], [llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_l16x2m_i: ClangBuiltin<"__builtin_xtensa_ae_l16x2m_i">,
  Intrinsic<[llvm_v2i32_ty], [llvm_ptr_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_xtensa_ae_l16x2m_iu: 
  Intrinsic<[llvm_v2i32_ty, llvm_ptr_ty], [llvm_ptr_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_xtensa_ae_l16x2m_x: ClangBuiltin<"__builtin_xtensa_ae_l16x2m_x">,
  Intrinsic<[llvm_v2i32_ty], [llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_l16x2m_xc: 
  Intrinsic<[llvm_v2i32_ty, llvm_ptr_ty], [llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_l16x2m_xu: 
  Intrinsic<[llvm_v2i32_ty, llvm_ptr_ty], [llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_l16x4_i: ClangBuiltin<"__builtin_xtensa_ae_l16x4_i">,
  Intrinsic<[llvm_v4i16_ty], [llvm_ptr_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_xtensa_ae_l16x4_ip: 
  Intrinsic<[llvm_v4i16_ty, llvm_ptr_ty], [llvm_ptr_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_xtensa_ae_l16x4_ric: 
  Intrinsic<[llvm_v4i16_ty, llvm_ptr_ty], [llvm_ptr_ty], []>;

def int_xtensa_ae_l16x4_rip: 
  Intrinsic<[llvm_v4i16_ty, llvm_ptr_ty], [llvm_ptr_ty], []>;

def int_xtensa_ae_l16x4_x: ClangBuiltin<"__builtin_xtensa_ae_l16x4_x">,
  Intrinsic<[llvm_v4i16_ty], [llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_l16x4_xc: 
  Intrinsic<[llvm_v4i16_ty, llvm_ptr_ty], [llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_l16x4_xp: 
  Intrinsic<[llvm_v4i16_ty, llvm_ptr_ty], [llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_l32_i: ClangBuiltin<"__builtin_xtensa_ae_l32_i">,
  Intrinsic<[llvm_v2i32_ty], [llvm_ptr_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_xtensa_ae_l32_ip: 
  Intrinsic<[llvm_v2i32_ty, llvm_ptr_ty], [llvm_ptr_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_xtensa_ae_l32_x: ClangBuiltin<"__builtin_xtensa_ae_l32_x">,
  Intrinsic<[llvm_v2i32_ty], [llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_l32_xc: 
  Intrinsic<[llvm_v2i32_ty, llvm_ptr_ty], [llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_l32_xp: 
  Intrinsic<[llvm_v2i32_ty, llvm_ptr_ty], [llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_l32f24_i: ClangBuiltin<"__builtin_xtensa_ae_l32f24_i">,
  Intrinsic<[llvm_v2i32_ty], [llvm_ptr_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_xtensa_ae_l32f24_ip: 
  Intrinsic<[llvm_v2i32_ty, llvm_ptr_ty], [llvm_ptr_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_xtensa_ae_l32f24_x: ClangBuiltin<"__builtin_xtensa_ae_l32f24_x">,
  Intrinsic<[llvm_v2i32_ty], [llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_l32f24_xc: 
  Intrinsic<[llvm_v2i32_ty, llvm_ptr_ty], [llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_l32f24_xp: 
  Intrinsic<[llvm_v2i32_ty, llvm_ptr_ty], [llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_l32m_i: ClangBuiltin<"__builtin_xtensa_ae_l32m_i">,
  Intrinsic<[llvm_v1i64_ty], [llvm_ptr_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_xtensa_ae_l32m_iu: 
  Intrinsic<[llvm_v1i64_ty, llvm_ptr_ty], [llvm_ptr_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_xtensa_ae_l32m_x: ClangBuiltin<"__builtin_xtensa_ae_l32m_x">,
  Intrinsic<[llvm_v1i64_ty], [llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_l32m_xc: 
  Intrinsic<[llvm_v1i64_ty, llvm_ptr_ty], [llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_l32m_xu: 
  Intrinsic<[llvm_v1i64_ty, llvm_ptr_ty], [llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_l32x2_i: ClangBuiltin<"__builtin_xtensa_ae_l32x2_i">,
  Intrinsic<[llvm_v2i32_ty], [llvm_ptr_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_xtensa_ae_l32x2_ip: 
  Intrinsic<[llvm_v2i32_ty, llvm_ptr_ty], [llvm_ptr_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_xtensa_ae_l32x2_ric: 
  Intrinsic<[llvm_v2i32_ty, llvm_ptr_ty], [llvm_ptr_ty], []>;

def int_xtensa_ae_l32x2_rip: 
  Intrinsic<[llvm_v2i32_ty, llvm_ptr_ty], [llvm_ptr_ty], []>;

def int_xtensa_ae_l32x2_x: ClangBuiltin<"__builtin_xtensa_ae_l32x2_x">,
  Intrinsic<[llvm_v2i32_ty], [llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_l32x2_xc: 
  Intrinsic<[llvm_v2i32_ty, llvm_ptr_ty], [llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_l32x2_xp: 
  Intrinsic<[llvm_v2i32_ty, llvm_ptr_ty], [llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_l32x2f24_i: ClangBuiltin<"__builtin_xtensa_ae_l32x2f24_i">,
  Intrinsic<[llvm_v2i32_ty], [llvm_ptr_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_xtensa_ae_l32x2f24_ip: 
  Intrinsic<[llvm_v2i32_ty, llvm_ptr_ty], [llvm_ptr_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_xtensa_ae_l32x2f24_ric: 
  Intrinsic<[llvm_v2i32_ty, llvm_ptr_ty], [llvm_ptr_ty], []>;

def int_xtensa_ae_l32x2f24_rip: 
  Intrinsic<[llvm_v2i32_ty, llvm_ptr_ty], [llvm_ptr_ty], []>;

def int_xtensa_ae_l32x2f24_x: ClangBuiltin<"__builtin_xtensa_ae_l32x2f24_x">,
  Intrinsic<[llvm_v2i32_ty], [llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_l32x2f24_xc: 
  Intrinsic<[llvm_v2i32_ty, llvm_ptr_ty], [llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_l32x2f24_xp: 
  Intrinsic<[llvm_v2i32_ty, llvm_ptr_ty], [llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_l64_i: ClangBuiltin<"__builtin_xtensa_ae_l64_i">,
  Intrinsic<[llvm_v1i64_ty], [llvm_ptr_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_xtensa_ae_l64_ip: 
  Intrinsic<[llvm_v1i64_ty, llvm_ptr_ty], [llvm_ptr_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_xtensa_ae_l64_x: ClangBuiltin<"__builtin_xtensa_ae_l64_x">,
  Intrinsic<[llvm_v1i64_ty], [llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_l64_xc: 
  Intrinsic<[llvm_v1i64_ty, llvm_ptr_ty], [llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_l64_xp: 
  Intrinsic<[llvm_v1i64_ty, llvm_ptr_ty], [llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_la16x4_ic: 
  Intrinsic<[llvm_v4i16_ty, llvm_v8i8_ty, llvm_ptr_ty], [llvm_v8i8_ty, llvm_ptr_ty], []>;

def int_xtensa_ae_la16x4_ip: 
  Intrinsic<[llvm_v4i16_ty, llvm_v8i8_ty, llvm_ptr_ty], [llvm_v8i8_ty, llvm_ptr_ty], []>;

def int_xtensa_ae_la16x4_ric: 
  Intrinsic<[llvm_v4i16_ty, llvm_v8i8_ty, llvm_ptr_ty], [llvm_v8i8_ty, llvm_ptr_ty], []>;

def int_xtensa_ae_la16x4_rip: 
  Intrinsic<[llvm_v4i16_ty, llvm_v8i8_ty, llvm_ptr_ty], [llvm_v8i8_ty, llvm_ptr_ty], []>;

def int_xtensa_ae_la16x4neg_pc: 
  Intrinsic<[llvm_v8i8_ty, llvm_ptr_ty], [llvm_ptr_ty], []>;

def int_xtensa_ae_la16x4pos_pc: 
  Intrinsic<[llvm_v8i8_ty, llvm_ptr_ty], [llvm_ptr_ty], []>;

def int_xtensa_ae_la24_ic: 
  Intrinsic<[llvm_v2i32_ty, llvm_v8i8_ty, llvm_ptr_ty], [llvm_v8i8_ty, llvm_ptr_ty], []>;

def int_xtensa_ae_la24_ip: 
  Intrinsic<[llvm_v2i32_ty, llvm_v8i8_ty, llvm_ptr_ty], [llvm_v8i8_ty, llvm_ptr_ty], []>;

def int_xtensa_ae_la24_ric: 
  Intrinsic<[llvm_v2i32_ty, llvm_v8i8_ty, llvm_ptr_ty], [llvm_v8i8_ty, llvm_ptr_ty], []>;

def int_xtensa_ae_la24_rip: 
  Intrinsic<[llvm_v2i32_ty, llvm_v8i8_ty, llvm_ptr_ty], [llvm_v8i8_ty, llvm_ptr_ty], []>;

def int_xtensa_ae_la24neg_pc: 
  Intrinsic<[llvm_v8i8_ty, llvm_ptr_ty], [llvm_ptr_ty], []>;

def int_xtensa_ae_la24pos_pc: 
  Intrinsic<[llvm_v8i8_ty, llvm_ptr_ty], [llvm_ptr_ty], []>;

def int_xtensa_ae_la24x2_ic: 
  Intrinsic<[llvm_v2i32_ty, llvm_v8i8_ty, llvm_ptr_ty], [llvm_v8i8_ty, llvm_ptr_ty], []>;

def int_xtensa_ae_la24x2_ip: 
  Intrinsic<[llvm_v2i32_ty, llvm_v8i8_ty, llvm_ptr_ty], [llvm_v8i8_ty, llvm_ptr_ty], []>;

def int_xtensa_ae_la24x2_ric: 
  Intrinsic<[llvm_v2i32_ty, llvm_v8i8_ty, llvm_ptr_ty], [llvm_v8i8_ty, llvm_ptr_ty], []>;

def int_xtensa_ae_la24x2_rip: 
  Intrinsic<[llvm_v2i32_ty, llvm_v8i8_ty, llvm_ptr_ty], [llvm_v8i8_ty, llvm_ptr_ty], []>;

def int_xtensa_ae_la24x2neg_pc: 
  Intrinsic<[llvm_v8i8_ty, llvm_ptr_ty], [llvm_ptr_ty], []>;

def int_xtensa_ae_la24x2pos_pc: 
  Intrinsic<[llvm_v8i8_ty, llvm_ptr_ty], [llvm_ptr_ty], []>;

def int_xtensa_ae_la32x2_ic: 
  Intrinsic<[llvm_v2i32_ty, llvm_v8i8_ty, llvm_ptr_ty], [llvm_v8i8_ty, llvm_ptr_ty], []>;

def int_xtensa_ae_la32x2_ip: 
  Intrinsic<[llvm_v2i32_ty, llvm_v8i8_ty, llvm_ptr_ty], [llvm_v8i8_ty, llvm_ptr_ty], []>;

def int_xtensa_ae_la32x2_ric: 
  Intrinsic<[llvm_v2i32_ty, llvm_v8i8_ty, llvm_ptr_ty], [llvm_v8i8_ty, llvm_ptr_ty], []>;

def int_xtensa_ae_la32x2_rip: 
  Intrinsic<[llvm_v2i32_ty, llvm_v8i8_ty, llvm_ptr_ty], [llvm_v8i8_ty, llvm_ptr_ty], []>;

def int_xtensa_ae_la32x2f24_ic: 
  Intrinsic<[llvm_v2i32_ty, llvm_v8i8_ty, llvm_ptr_ty], [llvm_v8i8_ty, llvm_ptr_ty], []>;

def int_xtensa_ae_la32x2f24_ip: 
  Intrinsic<[llvm_v2i32_ty, llvm_v8i8_ty, llvm_ptr_ty], [llvm_v8i8_ty, llvm_ptr_ty], []>;

def int_xtensa_ae_la32x2f24_ric: 
  Intrinsic<[llvm_v2i32_ty, llvm_v8i8_ty, llvm_ptr_ty], [llvm_v8i8_ty, llvm_ptr_ty], []>;

def int_xtensa_ae_la32x2f24_rip: 
  Intrinsic<[llvm_v2i32_ty, llvm_v8i8_ty, llvm_ptr_ty], [llvm_v8i8_ty, llvm_ptr_ty], []>;

def int_xtensa_ae_la32x2neg_pc: 
  Intrinsic<[llvm_v8i8_ty, llvm_ptr_ty], [llvm_ptr_ty], []>;

def int_xtensa_ae_la32x2pos_pc: 
  Intrinsic<[llvm_v8i8_ty, llvm_ptr_ty], [llvm_ptr_ty], []>;

def int_xtensa_ae_la64_pp: ClangBuiltin<"__builtin_xtensa_ae_la64_pp">,
  Intrinsic<[llvm_v8i8_ty], [llvm_ptr_ty], []>;

def int_xtensa_ae_lalign64_i: ClangBuiltin<"__builtin_xtensa_ae_lalign64_i">,
  Intrinsic<[llvm_v8i8_ty], [llvm_ptr_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_xtensa_ae_lb: ClangBuiltin<"__builtin_xtensa_ae_lb">,
  Intrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem]>;

def int_xtensa_ae_lbi: ClangBuiltin<"__builtin_xtensa_ae_lbi">,
  Intrinsic<[llvm_i32_ty], [llvm_i32_ty], [ImmArg<ArgIndex<0>>, IntrNoMem]>;

def int_xtensa_ae_lbk: ClangBuiltin<"__builtin_xtensa_ae_lbk">,
  Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;

def int_xtensa_ae_lbki: ClangBuiltin<"__builtin_xtensa_ae_lbki">,
  Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, IntrNoMem]>;

def int_xtensa_ae_lbs: ClangBuiltin<"__builtin_xtensa_ae_lbs">,
  Intrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem]>;

def int_xtensa_ae_lbsi: ClangBuiltin<"__builtin_xtensa_ae_lbsi">,
  Intrinsic<[llvm_i32_ty], [llvm_i32_ty], [ImmArg<ArgIndex<0>>, IntrNoMem]>;

def int_xtensa_ae_le16: 
  Intrinsic<[llvm_v4i1_ty], [llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_le32: 
  Intrinsic<[llvm_v2i1_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_le64: 
  Intrinsic<[llvm_v1i1_ty], [llvm_v1i64_ty, llvm_v1i64_ty], [IntrNoMem]>;

def int_xtensa_ae_lt16: 
  Intrinsic<[llvm_v4i1_ty], [llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_lt32: 
  Intrinsic<[llvm_v2i1_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_lt64: 
  Intrinsic<[llvm_v1i1_ty], [llvm_v1i64_ty, llvm_v1i64_ty], [IntrNoMem]>;

def int_xtensa_ae_max32: ClangBuiltin<"__builtin_xtensa_ae_max32">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_max64: ClangBuiltin<"__builtin_xtensa_ae_max64">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v1i64_ty], [IntrNoMem]>;

def int_xtensa_ae_maxabs32s: ClangBuiltin<"__builtin_xtensa_ae_maxabs32s">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_maxabs64s: ClangBuiltin<"__builtin_xtensa_ae_maxabs64s">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v1i64_ty], [IntrNoMem]>;

def int_xtensa_ae_min32: ClangBuiltin<"__builtin_xtensa_ae_min32">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_min64: ClangBuiltin<"__builtin_xtensa_ae_min64">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v1i64_ty], [IntrNoMem]>;

def int_xtensa_ae_minabs32s: ClangBuiltin<"__builtin_xtensa_ae_minabs32s">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_minabs64s: ClangBuiltin<"__builtin_xtensa_ae_minabs64s">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v1i64_ty], [IntrNoMem]>;

def int_xtensa_ae_mov: ClangBuiltin<"__builtin_xtensa_ae_mov">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty], [IntrNoMem]>;

def int_xtensa_ae_movad16_0: ClangBuiltin<"__builtin_xtensa_ae_movad16_0">,
  Intrinsic<[llvm_i32_ty], [llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_movad16_1: ClangBuiltin<"__builtin_xtensa_ae_movad16_1">,
  Intrinsic<[llvm_i32_ty], [llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_movad16_2: ClangBuiltin<"__builtin_xtensa_ae_movad16_2">,
  Intrinsic<[llvm_i32_ty], [llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_movad16_3: ClangBuiltin<"__builtin_xtensa_ae_movad16_3">,
  Intrinsic<[llvm_i32_ty], [llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_movad32_h: ClangBuiltin<"__builtin_xtensa_ae_movad32_h">,
  Intrinsic<[llvm_i32_ty], [llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_movad32_l: ClangBuiltin<"__builtin_xtensa_ae_movad32_l">,
  Intrinsic<[llvm_i32_ty], [llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_movalign: ClangBuiltin<"__builtin_xtensa_ae_movalign">,
  Intrinsic<[llvm_v8i8_ty], [llvm_v8i8_ty], [IntrNoMem]>;

def int_xtensa_ae_movda16: ClangBuiltin<"__builtin_xtensa_ae_movda16">,
  Intrinsic<[llvm_v4i16_ty], [llvm_i32_ty], [IntrNoMem]>;

def int_xtensa_ae_movda16x2: ClangBuiltin<"__builtin_xtensa_ae_movda16x2">,
  Intrinsic<[llvm_v4i16_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;

def int_xtensa_ae_movda32: ClangBuiltin<"__builtin_xtensa_ae_movda32">,
  Intrinsic<[llvm_v1i32_ty], [llvm_i32_ty], [IntrNoMem]>;

def int_xtensa_ae_movda32x2: ClangBuiltin<"__builtin_xtensa_ae_movda32x2">,
  Intrinsic<[llvm_v2i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;

def int_xtensa_ae_movf16x4: 
  Intrinsic<[llvm_v4i16_ty], [llvm_v4i16_ty, llvm_v4i16_ty, llvm_v4i1_ty], [IntrNoMem]>;

def int_xtensa_ae_movf32x2: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v2i1_ty], [IntrNoMem]>;

def int_xtensa_ae_movf64: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v1i64_ty, llvm_v1i1_ty], [IntrNoMem]>;

def int_xtensa_ae_movi: ClangBuiltin<"__builtin_xtensa_ae_movi">,
  Intrinsic<[llvm_v2i32_ty], [llvm_i32_ty], [ImmArg<ArgIndex<0>>, IntrNoMem]>;

def int_xtensa_ae_movt16x4: 
  Intrinsic<[llvm_v4i16_ty], [llvm_v4i16_ty, llvm_v4i16_ty, llvm_v4i1_ty], [IntrNoMem]>;

def int_xtensa_ae_movt32x2: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v2i1_ty], [IntrNoMem]>;

def int_xtensa_ae_movt64: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v1i64_ty, llvm_v1i1_ty], [IntrNoMem]>;

def int_xtensa_ae_mul16x4: 
  Intrinsic<[llvm_v2i32_ty, llvm_v2i32_ty], [llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mul32_hh: ClangBuiltin<"__builtin_xtensa_ae_mul32_hh">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mul32_lh: ClangBuiltin<"__builtin_xtensa_ae_mul32_lh">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mul32_ll: ClangBuiltin<"__builtin_xtensa_ae_mul32_ll">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mul32_ll_s2: ClangBuiltin<"__builtin_xtensa_ae_mul32_ll_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mul32u_ll: ClangBuiltin<"__builtin_xtensa_ae_mul32u_ll">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mul32x16_h0: ClangBuiltin<"__builtin_xtensa_ae_mul32x16_h0">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mul32x16_h0_s2: ClangBuiltin<"__builtin_xtensa_ae_mul32x16_h0_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mul32x16_h1: ClangBuiltin<"__builtin_xtensa_ae_mul32x16_h1">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mul32x16_h1_s2: ClangBuiltin<"__builtin_xtensa_ae_mul32x16_h1_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mul32x16_h2: ClangBuiltin<"__builtin_xtensa_ae_mul32x16_h2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mul32x16_h2_s2: ClangBuiltin<"__builtin_xtensa_ae_mul32x16_h2_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mul32x16_h3: ClangBuiltin<"__builtin_xtensa_ae_mul32x16_h3">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mul32x16_h3_s2: ClangBuiltin<"__builtin_xtensa_ae_mul32x16_h3_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mul32x16_l0: ClangBuiltin<"__builtin_xtensa_ae_mul32x16_l0">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mul32x16_l0_s2: ClangBuiltin<"__builtin_xtensa_ae_mul32x16_l0_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mul32x16_l1: ClangBuiltin<"__builtin_xtensa_ae_mul32x16_l1">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mul32x16_l1_s2: ClangBuiltin<"__builtin_xtensa_ae_mul32x16_l1_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mul32x16_l2: ClangBuiltin<"__builtin_xtensa_ae_mul32x16_l2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mul32x16_l2_s2: ClangBuiltin<"__builtin_xtensa_ae_mul32x16_l2_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mul32x16_l3: ClangBuiltin<"__builtin_xtensa_ae_mul32x16_l3">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mul32x16_l3_s2: ClangBuiltin<"__builtin_xtensa_ae_mul32x16_l3_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mula16x4: 
  Intrinsic<[llvm_v2i32_ty, llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mula32_hh: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mula32_lh: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mula32_ll: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mula32_ll_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mula32u_ll: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mula32x16_h0: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mula32x16_h0_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mula32x16_h1: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mula32x16_h1_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mula32x16_h2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mula32x16_h2_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mula32x16_h3: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mula32x16_h3_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mula32x16_l0: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mula32x16_l0_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mula32x16_l1: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mula32x16_l1_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mula32x16_l2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mula32x16_l2_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mula32x16_l3: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mula32x16_l3_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaad24_hh_ll: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaad24_hh_ll_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaad24_hl_lh: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaad24_hl_lh_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaad32x16_h0_l1: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaad32x16_h0_l1_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaad32x16_h1_l0: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaad32x16_h1_l0_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaad32x16_h2_l3: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaad32x16_h2_l3_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaad32x16_h3_l2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaad32x16_h3_l2_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaafd16ss_11_00: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaafd16ss_11_00_s2: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaafd16ss_13_02: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaafd16ss_13_02_s2: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaafd16ss_33_22: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaafd16ss_33_22_s2: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaafd24_hh_ll: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaafd24_hh_ll_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaafd24_hl_lh: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaafd24_hl_lh_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaafd32x16_h0_l1: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaafd32x16_h0_l1_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaafd32x16_h1_l0: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaafd32x16_h1_l0_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaafd32x16_h2_l3: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaafd32x16_h2_l3_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaafd32x16_h3_l2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaafd32x16_h3_l2_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulac24: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulac32x16_h: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulac32x16_l: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaf16ss_00: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaf16ss_00_s2: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaf16ss_10: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaf16ss_11: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaf16ss_20: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaf16ss_21: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaf16ss_22: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaf16ss_30: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaf16ss_31: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaf16ss_32: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaf16ss_33: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaf16x4ss: 
  Intrinsic<[llvm_v2i32_ty, llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaf32r_hh: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaf32r_lh: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaf32r_ll: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaf32r_ll_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaf32s_hh: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaf32s_lh: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaf32s_ll: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaf32s_ll_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaf32x16_h0: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaf32x16_h0_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaf32x16_h1: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaf32x16_h1_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaf32x16_h2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaf32x16_h2_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaf32x16_h3: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaf32x16_h3_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaf32x16_l0: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaf32x16_l0_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaf32x16_l1: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaf32x16_l1_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaf32x16_l2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaf32x16_l2_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaf32x16_l3: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaf32x16_l3_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaf48q32sp16s_l: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v1i64_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaf48q32sp16s_l_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v1i64_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaf48q32sp16u_l: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v1i64_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaf48q32sp16u_l_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v1i64_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulafc24ra: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulafc32x16ras_h: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulafc32x16ras_l: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulafd24x2_fir_h: 
  Intrinsic<[llvm_v1i64_ty, llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulafd24x2_fir_l: 
  Intrinsic<[llvm_v1i64_ty, llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulafd32x16x2_fir_hh: 
  Intrinsic<[llvm_v1i64_ty, llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulafd32x16x2_fir_hl: 
  Intrinsic<[llvm_v1i64_ty, llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulafd32x16x2_fir_lh: 
  Intrinsic<[llvm_v1i64_ty, llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulafd32x16x2_fir_ll: 
  Intrinsic<[llvm_v1i64_ty, llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulafp24x2r: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulafp24x2r_s2: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulafp24x2ra: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulafp24x2ra_s2: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulafp32x16x2ras_h: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulafp32x16x2ras_h_s2: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulafp32x16x2ras_l: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulafp32x16x2ras_l_s2: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulafp32x16x2rs_h: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulafp32x16x2rs_h_s2: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulafp32x16x2rs_l: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulafp32x16x2rs_l_s2: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulafp32x2ras: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulafp32x2rs: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulafq32sp24s_h_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v1i64_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulafq32sp24s_l_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v1i64_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulap24x2: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulap24x2_s2: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulap32x16x2_h: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulap32x16x2_l: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulap32x2: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaq32sp16s_l_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v1i64_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulaq32sp16u_l_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v1i64_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mularfq32sp24s_h_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v1i64_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mularfq32sp24s_l_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v1i64_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulas32f48p16s_hh: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulas32f48p16s_hh_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulas32f48p16s_lh: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulas32f48p16s_lh_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulas32f48p16s_ll: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulas32f48p16s_ll_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulasd24_hh_ll: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulasd24_hh_ll_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulasd24_hl_lh: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulasd24_hl_lh_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulasd32x16_h1_l0: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulasd32x16_h1_l0_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulasd32x16_h3_l2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulasd32x16_h3_l2_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulasfd24_hh_ll: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulasfd24_hh_ll_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulasfd24_hl_lh: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulasfd24_hl_lh_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulasfd32x16_h1_l0: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulasfd32x16_h1_l0_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulasfd32x16_h3_l2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulasfd32x16_h3_l2_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulc24: ClangBuiltin<"__builtin_xtensa_ae_mulc24">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulc32x16_h: ClangBuiltin<"__builtin_xtensa_ae_mulc32x16_h">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulc32x16_l: ClangBuiltin<"__builtin_xtensa_ae_mulc32x16_l">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulf16ss_00: ClangBuiltin<"__builtin_xtensa_ae_mulf16ss_00">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulf16ss_00_s2: ClangBuiltin<"__builtin_xtensa_ae_mulf16ss_00_s2">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulf16ss_10: ClangBuiltin<"__builtin_xtensa_ae_mulf16ss_10">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulf16ss_11: ClangBuiltin<"__builtin_xtensa_ae_mulf16ss_11">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulf16ss_20: ClangBuiltin<"__builtin_xtensa_ae_mulf16ss_20">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulf16ss_21: ClangBuiltin<"__builtin_xtensa_ae_mulf16ss_21">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulf16ss_22: ClangBuiltin<"__builtin_xtensa_ae_mulf16ss_22">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulf16ss_30: ClangBuiltin<"__builtin_xtensa_ae_mulf16ss_30">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulf16ss_31: ClangBuiltin<"__builtin_xtensa_ae_mulf16ss_31">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulf16ss_32: ClangBuiltin<"__builtin_xtensa_ae_mulf16ss_32">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulf16ss_33: ClangBuiltin<"__builtin_xtensa_ae_mulf16ss_33">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulf16x4ss: 
  Intrinsic<[llvm_v2i32_ty, llvm_v2i32_ty], [llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulf32r_hh: ClangBuiltin<"__builtin_xtensa_ae_mulf32r_hh">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulf32r_lh: ClangBuiltin<"__builtin_xtensa_ae_mulf32r_lh">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulf32r_ll: ClangBuiltin<"__builtin_xtensa_ae_mulf32r_ll">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulf32r_ll_s2: ClangBuiltin<"__builtin_xtensa_ae_mulf32r_ll_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulf32s_hh: ClangBuiltin<"__builtin_xtensa_ae_mulf32s_hh">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulf32s_lh: ClangBuiltin<"__builtin_xtensa_ae_mulf32s_lh">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulf32s_ll: ClangBuiltin<"__builtin_xtensa_ae_mulf32s_ll">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulf32s_ll_s2: ClangBuiltin<"__builtin_xtensa_ae_mulf32s_ll_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulf32x16_h0: ClangBuiltin<"__builtin_xtensa_ae_mulf32x16_h0">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulf32x16_h0_s2: ClangBuiltin<"__builtin_xtensa_ae_mulf32x16_h0_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulf32x16_h1: ClangBuiltin<"__builtin_xtensa_ae_mulf32x16_h1">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulf32x16_h1_s2: ClangBuiltin<"__builtin_xtensa_ae_mulf32x16_h1_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulf32x16_h2: ClangBuiltin<"__builtin_xtensa_ae_mulf32x16_h2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulf32x16_h2_s2: ClangBuiltin<"__builtin_xtensa_ae_mulf32x16_h2_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulf32x16_h3: ClangBuiltin<"__builtin_xtensa_ae_mulf32x16_h3">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulf32x16_h3_s2: ClangBuiltin<"__builtin_xtensa_ae_mulf32x16_h3_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulf32x16_l0: ClangBuiltin<"__builtin_xtensa_ae_mulf32x16_l0">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulf32x16_l0_s2: ClangBuiltin<"__builtin_xtensa_ae_mulf32x16_l0_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulf32x16_l1: ClangBuiltin<"__builtin_xtensa_ae_mulf32x16_l1">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulf32x16_l1_s2: ClangBuiltin<"__builtin_xtensa_ae_mulf32x16_l1_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulf32x16_l2: ClangBuiltin<"__builtin_xtensa_ae_mulf32x16_l2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulf32x16_l2_s2: ClangBuiltin<"__builtin_xtensa_ae_mulf32x16_l2_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulf32x16_l3: ClangBuiltin<"__builtin_xtensa_ae_mulf32x16_l3">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulf32x16_l3_s2: ClangBuiltin<"__builtin_xtensa_ae_mulf32x16_l3_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulf48q32sp16s_l: ClangBuiltin<"__builtin_xtensa_ae_mulf48q32sp16s_l">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulf48q32sp16s_l_s2: ClangBuiltin<"__builtin_xtensa_ae_mulf48q32sp16s_l_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulf48q32sp16u_l: ClangBuiltin<"__builtin_xtensa_ae_mulf48q32sp16u_l">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulf48q32sp16u_l_s2: ClangBuiltin<"__builtin_xtensa_ae_mulf48q32sp16u_l_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulfc24ra: ClangBuiltin<"__builtin_xtensa_ae_mulfc24ra">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulfc32x16ras_h: ClangBuiltin<"__builtin_xtensa_ae_mulfc32x16ras_h">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulfc32x16ras_l: ClangBuiltin<"__builtin_xtensa_ae_mulfc32x16ras_l">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulfd24x2_fir_h: 
  Intrinsic<[llvm_v1i64_ty, llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulfd24x2_fir_l: 
  Intrinsic<[llvm_v1i64_ty, llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulfd32x16x2_fir_hh: 
  Intrinsic<[llvm_v1i64_ty, llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulfd32x16x2_fir_hl: 
  Intrinsic<[llvm_v1i64_ty, llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulfd32x16x2_fir_lh: 
  Intrinsic<[llvm_v1i64_ty, llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulfd32x16x2_fir_ll: 
  Intrinsic<[llvm_v1i64_ty, llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulfp16x4ras: ClangBuiltin<"__builtin_xtensa_ae_mulfp16x4ras">,
  Intrinsic<[llvm_v4i16_ty], [llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulfp16x4s: ClangBuiltin<"__builtin_xtensa_ae_mulfp16x4s">,
  Intrinsic<[llvm_v4i16_ty], [llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulfp24x2r: ClangBuiltin<"__builtin_xtensa_ae_mulfp24x2r">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulfp24x2r_s2: ClangBuiltin<"__builtin_xtensa_ae_mulfp24x2r_s2">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulfp24x2ra: ClangBuiltin<"__builtin_xtensa_ae_mulfp24x2ra">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulfp24x2ra_s2: ClangBuiltin<"__builtin_xtensa_ae_mulfp24x2ra_s2">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulfp32x16x2ras_h: ClangBuiltin<"__builtin_xtensa_ae_mulfp32x16x2ras_h">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulfp32x16x2ras_h_s2: ClangBuiltin<"__builtin_xtensa_ae_mulfp32x16x2ras_h_s2">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulfp32x16x2ras_l: ClangBuiltin<"__builtin_xtensa_ae_mulfp32x16x2ras_l">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulfp32x16x2ras_l_s2: ClangBuiltin<"__builtin_xtensa_ae_mulfp32x16x2ras_l_s2">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulfp32x16x2rs_h: ClangBuiltin<"__builtin_xtensa_ae_mulfp32x16x2rs_h">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulfp32x16x2rs_h_s2: ClangBuiltin<"__builtin_xtensa_ae_mulfp32x16x2rs_h_s2">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulfp32x16x2rs_l: ClangBuiltin<"__builtin_xtensa_ae_mulfp32x16x2rs_l">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulfp32x16x2rs_l_s2: ClangBuiltin<"__builtin_xtensa_ae_mulfp32x16x2rs_l_s2">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulfp32x2ras: ClangBuiltin<"__builtin_xtensa_ae_mulfp32x2ras">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulfp32x2rs: ClangBuiltin<"__builtin_xtensa_ae_mulfp32x2rs">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulfq32sp24s_h_s2: ClangBuiltin<"__builtin_xtensa_ae_mulfq32sp24s_h_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulfq32sp24s_l_s2: ClangBuiltin<"__builtin_xtensa_ae_mulfq32sp24s_l_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulp24x2: ClangBuiltin<"__builtin_xtensa_ae_mulp24x2">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulp24x2_s2: ClangBuiltin<"__builtin_xtensa_ae_mulp24x2_s2">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulp32x16x2_h: ClangBuiltin<"__builtin_xtensa_ae_mulp32x16x2_h">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulp32x16x2_l: ClangBuiltin<"__builtin_xtensa_ae_mulp32x16x2_l">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulp32x2: ClangBuiltin<"__builtin_xtensa_ae_mulp32x2">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulq32sp16s_l_s2: ClangBuiltin<"__builtin_xtensa_ae_mulq32sp16s_l_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulq32sp16u_l_s2: ClangBuiltin<"__builtin_xtensa_ae_mulq32sp16u_l_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulrfq32sp24s_h_s2: ClangBuiltin<"__builtin_xtensa_ae_mulrfq32sp24s_h_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulrfq32sp24s_l_s2: ClangBuiltin<"__builtin_xtensa_ae_mulrfq32sp24s_l_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_muls16x4: 
  Intrinsic<[llvm_v2i32_ty, llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_muls32_hh: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_muls32_lh: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_muls32_ll: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_muls32f48p16s_hh: ClangBuiltin<"__builtin_xtensa_ae_muls32f48p16s_hh">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_muls32f48p16s_hh_s2: ClangBuiltin<"__builtin_xtensa_ae_muls32f48p16s_hh_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_muls32f48p16s_lh: ClangBuiltin<"__builtin_xtensa_ae_muls32f48p16s_lh">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_muls32f48p16s_lh_s2: ClangBuiltin<"__builtin_xtensa_ae_muls32f48p16s_lh_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_muls32f48p16s_ll: ClangBuiltin<"__builtin_xtensa_ae_muls32f48p16s_ll">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_muls32f48p16s_ll_s2: ClangBuiltin<"__builtin_xtensa_ae_muls32f48p16s_ll_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_muls32u_ll: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_muls32x16_h0: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_muls32x16_h0_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_muls32x16_h1: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_muls32x16_h1_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_muls32x16_h2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_muls32x16_h2_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_muls32x16_h3: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_muls32x16_h3_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_muls32x16_l0: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_muls32x16_l0_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_muls32x16_l1: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_muls32x16_l1_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_muls32x16_l2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_muls32x16_l2_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_muls32x16_l3: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_muls32x16_l3_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsad24_hh_ll: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsad24_hh_ll_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsad32x16_h1_l0: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsad32x16_h1_l0_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsad32x16_h3_l2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsad32x16_h3_l2_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsafd24_hh_ll: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsafd24_hh_ll_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsafd32x16_h1_l0: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsafd32x16_h1_l0_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsafd32x16_h3_l2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsafd32x16_h3_l2_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsf16ss_00: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsf16ss_00_s2: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsf16ss_10: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsf16ss_11: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsf16ss_20: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsf16ss_21: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsf16ss_22: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsf16ss_30: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsf16ss_31: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsf16ss_32: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsf16ss_33: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsf16x4ss: 
  Intrinsic<[llvm_v2i32_ty, llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsf32r_hh: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsf32r_lh: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsf32r_ll: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsf32r_ll_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsf32s_hh: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsf32s_lh: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsf32s_ll: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsf32x16_h0: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsf32x16_h0_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsf32x16_h1: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsf32x16_h1_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsf32x16_h2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsf32x16_h2_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsf32x16_h3: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsf32x16_h3_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsf32x16_l0: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsf32x16_l0_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsf32x16_l1: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsf32x16_l1_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsf32x16_l2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsf32x16_l2_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsf32x16_l3: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsf32x16_l3_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsf48q32sp16s_l: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v1i64_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsf48q32sp16s_l_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v1i64_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsf48q32sp16u_l: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v1i64_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsf48q32sp16u_l_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v1i64_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsfp24x2r: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsfp24x2r_s2: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsfp24x2ra: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsfp24x2ra_s2: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsfp32x16x2ras_h: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsfp32x16x2ras_h_s2: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsfp32x16x2ras_l: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsfp32x16x2ras_l_s2: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsfp32x16x2rs_h: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsfp32x16x2rs_h_s2: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsfp32x16x2rs_l: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsfp32x16x2rs_l_s2: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsfp32x2ras: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsfp32x2rs: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsfq32sp24s_h_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v1i64_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsfq32sp24s_l_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v1i64_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsp24x2: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsp24x2_s2: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsp32x16x2_h: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsp32x16x2_l: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsp32x2: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsq32sp16s_l_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v1i64_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsq32sp16u_l_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v1i64_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsrfq32sp24s_h_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v1i64_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulsrfq32sp24s_l_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v1i64_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulss32f48p16s_hh: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulss32f48p16s_hh_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulss32f48p16s_lh: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulss32f48p16s_lh_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulss32f48p16s_ll: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulss32f48p16s_ll_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulssd24_hh_ll: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulssd24_hh_ll_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulssd24_hl_lh: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulssd24_hl_lh_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulssd32x16_h1_l0: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulssd32x16_h1_l0_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulssd32x16_h3_l2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulssd32x16_h3_l2_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulssfd16ss_11_00: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulssfd16ss_11_00_s2: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulssfd16ss_13_02: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulssfd16ss_13_02_s2: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulssfd16ss_33_22: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulssfd16ss_33_22_s2: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulssfd24_hh_ll: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulssfd24_hh_ll_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulssfd24_hl_lh: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulssfd24_hl_lh_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulssfd32x16_h1_l0: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulssfd32x16_h1_l0_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulssfd32x16_h3_l2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulssfd32x16_h3_l2_s2: 
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzaad24_hh_ll: ClangBuiltin<"__builtin_xtensa_ae_mulzaad24_hh_ll">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzaad24_hh_ll_s2: ClangBuiltin<"__builtin_xtensa_ae_mulzaad24_hh_ll_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzaad24_hl_lh: ClangBuiltin<"__builtin_xtensa_ae_mulzaad24_hl_lh">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzaad24_hl_lh_s2: ClangBuiltin<"__builtin_xtensa_ae_mulzaad24_hl_lh_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzaad32x16_h0_l1: ClangBuiltin<"__builtin_xtensa_ae_mulzaad32x16_h0_l1">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzaad32x16_h0_l1_s2: ClangBuiltin<"__builtin_xtensa_ae_mulzaad32x16_h0_l1_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzaad32x16_h1_l0: ClangBuiltin<"__builtin_xtensa_ae_mulzaad32x16_h1_l0">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzaad32x16_h1_l0_s2: ClangBuiltin<"__builtin_xtensa_ae_mulzaad32x16_h1_l0_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzaad32x16_h2_l3: ClangBuiltin<"__builtin_xtensa_ae_mulzaad32x16_h2_l3">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzaad32x16_h2_l3_s2: ClangBuiltin<"__builtin_xtensa_ae_mulzaad32x16_h2_l3_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzaad32x16_h3_l2: ClangBuiltin<"__builtin_xtensa_ae_mulzaad32x16_h3_l2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzaad32x16_h3_l2_s2: ClangBuiltin<"__builtin_xtensa_ae_mulzaad32x16_h3_l2_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzaafd16ss_11_00: ClangBuiltin<"__builtin_xtensa_ae_mulzaafd16ss_11_00">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzaafd16ss_11_00_s2: ClangBuiltin<"__builtin_xtensa_ae_mulzaafd16ss_11_00_s2">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzaafd16ss_13_02: ClangBuiltin<"__builtin_xtensa_ae_mulzaafd16ss_13_02">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzaafd16ss_13_02_s2: ClangBuiltin<"__builtin_xtensa_ae_mulzaafd16ss_13_02_s2">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzaafd16ss_33_22: ClangBuiltin<"__builtin_xtensa_ae_mulzaafd16ss_33_22">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzaafd16ss_33_22_s2: ClangBuiltin<"__builtin_xtensa_ae_mulzaafd16ss_33_22_s2">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzaafd24_hh_ll: ClangBuiltin<"__builtin_xtensa_ae_mulzaafd24_hh_ll">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzaafd24_hh_ll_s2: ClangBuiltin<"__builtin_xtensa_ae_mulzaafd24_hh_ll_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzaafd24_hl_lh: ClangBuiltin<"__builtin_xtensa_ae_mulzaafd24_hl_lh">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzaafd24_hl_lh_s2: ClangBuiltin<"__builtin_xtensa_ae_mulzaafd24_hl_lh_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzaafd32x16_h0_l1: ClangBuiltin<"__builtin_xtensa_ae_mulzaafd32x16_h0_l1">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzaafd32x16_h0_l1_s2: ClangBuiltin<"__builtin_xtensa_ae_mulzaafd32x16_h0_l1_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzaafd32x16_h1_l0: ClangBuiltin<"__builtin_xtensa_ae_mulzaafd32x16_h1_l0">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzaafd32x16_h1_l0_s2: ClangBuiltin<"__builtin_xtensa_ae_mulzaafd32x16_h1_l0_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzaafd32x16_h2_l3: ClangBuiltin<"__builtin_xtensa_ae_mulzaafd32x16_h2_l3">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzaafd32x16_h2_l3_s2: ClangBuiltin<"__builtin_xtensa_ae_mulzaafd32x16_h2_l3_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzaafd32x16_h3_l2: ClangBuiltin<"__builtin_xtensa_ae_mulzaafd32x16_h3_l2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzaafd32x16_h3_l2_s2: ClangBuiltin<"__builtin_xtensa_ae_mulzaafd32x16_h3_l2_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzasd24_hh_ll: ClangBuiltin<"__builtin_xtensa_ae_mulzasd24_hh_ll">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzasd24_hh_ll_s2: ClangBuiltin<"__builtin_xtensa_ae_mulzasd24_hh_ll_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzasd24_hl_lh: ClangBuiltin<"__builtin_xtensa_ae_mulzasd24_hl_lh">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzasd24_hl_lh_s2: ClangBuiltin<"__builtin_xtensa_ae_mulzasd24_hl_lh_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzasd32x16_h1_l0: ClangBuiltin<"__builtin_xtensa_ae_mulzasd32x16_h1_l0">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzasd32x16_h1_l0_s2: ClangBuiltin<"__builtin_xtensa_ae_mulzasd32x16_h1_l0_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzasd32x16_h3_l2: ClangBuiltin<"__builtin_xtensa_ae_mulzasd32x16_h3_l2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzasd32x16_h3_l2_s2: ClangBuiltin<"__builtin_xtensa_ae_mulzasd32x16_h3_l2_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzasfd24_hh_ll: ClangBuiltin<"__builtin_xtensa_ae_mulzasfd24_hh_ll">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzasfd24_hh_ll_s2: ClangBuiltin<"__builtin_xtensa_ae_mulzasfd24_hh_ll_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzasfd24_hl_lh: ClangBuiltin<"__builtin_xtensa_ae_mulzasfd24_hl_lh">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzasfd24_hl_lh_s2: ClangBuiltin<"__builtin_xtensa_ae_mulzasfd24_hl_lh_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzasfd32x16_h1_l0: ClangBuiltin<"__builtin_xtensa_ae_mulzasfd32x16_h1_l0">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzasfd32x16_h1_l0_s2: ClangBuiltin<"__builtin_xtensa_ae_mulzasfd32x16_h1_l0_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzasfd32x16_h3_l2: ClangBuiltin<"__builtin_xtensa_ae_mulzasfd32x16_h3_l2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzasfd32x16_h3_l2_s2: ClangBuiltin<"__builtin_xtensa_ae_mulzasfd32x16_h3_l2_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzsad24_hh_ll: ClangBuiltin<"__builtin_xtensa_ae_mulzsad24_hh_ll">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzsad24_hh_ll_s2: ClangBuiltin<"__builtin_xtensa_ae_mulzsad24_hh_ll_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzsad32x16_h1_l0: ClangBuiltin<"__builtin_xtensa_ae_mulzsad32x16_h1_l0">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzsad32x16_h1_l0_s2: ClangBuiltin<"__builtin_xtensa_ae_mulzsad32x16_h1_l0_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzsad32x16_h3_l2: ClangBuiltin<"__builtin_xtensa_ae_mulzsad32x16_h3_l2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzsad32x16_h3_l2_s2: ClangBuiltin<"__builtin_xtensa_ae_mulzsad32x16_h3_l2_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzsafd24_hh_ll: ClangBuiltin<"__builtin_xtensa_ae_mulzsafd24_hh_ll">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzsafd24_hh_ll_s2: ClangBuiltin<"__builtin_xtensa_ae_mulzsafd24_hh_ll_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzsafd32x16_h1_l0: ClangBuiltin<"__builtin_xtensa_ae_mulzsafd32x16_h1_l0">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzsafd32x16_h1_l0_s2: ClangBuiltin<"__builtin_xtensa_ae_mulzsafd32x16_h1_l0_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzsafd32x16_h3_l2: ClangBuiltin<"__builtin_xtensa_ae_mulzsafd32x16_h3_l2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzsafd32x16_h3_l2_s2: ClangBuiltin<"__builtin_xtensa_ae_mulzsafd32x16_h3_l2_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzssd24_hh_ll: ClangBuiltin<"__builtin_xtensa_ae_mulzssd24_hh_ll">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzssd24_hh_ll_s2: ClangBuiltin<"__builtin_xtensa_ae_mulzssd24_hh_ll_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzssd24_hl_lh: ClangBuiltin<"__builtin_xtensa_ae_mulzssd24_hl_lh">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzssd24_hl_lh_s2: ClangBuiltin<"__builtin_xtensa_ae_mulzssd24_hl_lh_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzssd32x16_h1_l0: ClangBuiltin<"__builtin_xtensa_ae_mulzssd32x16_h1_l0">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzssd32x16_h1_l0_s2: ClangBuiltin<"__builtin_xtensa_ae_mulzssd32x16_h1_l0_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzssd32x16_h3_l2: ClangBuiltin<"__builtin_xtensa_ae_mulzssd32x16_h3_l2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzssd32x16_h3_l2_s2: ClangBuiltin<"__builtin_xtensa_ae_mulzssd32x16_h3_l2_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzssfd16ss_11_00: ClangBuiltin<"__builtin_xtensa_ae_mulzssfd16ss_11_00">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzssfd16ss_11_00_s2: ClangBuiltin<"__builtin_xtensa_ae_mulzssfd16ss_11_00_s2">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzssfd16ss_13_02: ClangBuiltin<"__builtin_xtensa_ae_mulzssfd16ss_13_02">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzssfd16ss_13_02_s2: ClangBuiltin<"__builtin_xtensa_ae_mulzssfd16ss_13_02_s2">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzssfd16ss_33_22: ClangBuiltin<"__builtin_xtensa_ae_mulzssfd16ss_33_22">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzssfd16ss_33_22_s2: ClangBuiltin<"__builtin_xtensa_ae_mulzssfd16ss_33_22_s2">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzssfd24_hh_ll: ClangBuiltin<"__builtin_xtensa_ae_mulzssfd24_hh_ll">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzssfd24_hh_ll_s2: ClangBuiltin<"__builtin_xtensa_ae_mulzssfd24_hh_ll_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzssfd24_hl_lh: ClangBuiltin<"__builtin_xtensa_ae_mulzssfd24_hl_lh">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzssfd24_hl_lh_s2: ClangBuiltin<"__builtin_xtensa_ae_mulzssfd24_hl_lh_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzssfd32x16_h1_l0: ClangBuiltin<"__builtin_xtensa_ae_mulzssfd32x16_h1_l0">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzssfd32x16_h1_l0_s2: ClangBuiltin<"__builtin_xtensa_ae_mulzssfd32x16_h1_l0_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzssfd32x16_h3_l2: ClangBuiltin<"__builtin_xtensa_ae_mulzssfd32x16_h3_l2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_mulzssfd32x16_h3_l2_s2: ClangBuiltin<"__builtin_xtensa_ae_mulzssfd32x16_h3_l2_s2">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_nand: ClangBuiltin<"__builtin_xtensa_ae_nand">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v1i64_ty], [IntrNoMem]>;

def int_xtensa_ae_neg16s: ClangBuiltin<"__builtin_xtensa_ae_neg16s">,
  Intrinsic<[llvm_v4i16_ty], [llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_neg24s: ClangBuiltin<"__builtin_xtensa_ae_neg24s">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_neg32: ClangBuiltin<"__builtin_xtensa_ae_neg32">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_neg32s: ClangBuiltin<"__builtin_xtensa_ae_neg32s">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_neg64: ClangBuiltin<"__builtin_xtensa_ae_neg64">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty], [IntrNoMem]>;

def int_xtensa_ae_neg64s: ClangBuiltin<"__builtin_xtensa_ae_neg64s">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty], [IntrNoMem]>;

def int_xtensa_ae_nsa64: ClangBuiltin<"__builtin_xtensa_ae_nsa64">,
  Intrinsic<[llvm_i32_ty], [llvm_v1i64_ty], [IntrNoMem]>;

def int_xtensa_ae_nsaz16_0: ClangBuiltin<"__builtin_xtensa_ae_nsaz16_0">,
  Intrinsic<[llvm_i32_ty], [llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_nsaz32_l: ClangBuiltin<"__builtin_xtensa_ae_nsaz32_l">,
  Intrinsic<[llvm_i32_ty], [llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_or: ClangBuiltin<"__builtin_xtensa_ae_or">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v1i64_ty], [IntrNoMem]>;

def int_xtensa_ae_pksr24: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v1i64_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>, IntrNoMem]>;

def int_xtensa_ae_pksr32: 
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v1i64_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>, IntrNoMem]>;

def int_xtensa_ae_round16x4f32sasym: ClangBuiltin<"__builtin_xtensa_ae_round16x4f32sasym">,
  Intrinsic<[llvm_v4i16_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_round16x4f32ssym: ClangBuiltin<"__builtin_xtensa_ae_round16x4f32ssym">,
  Intrinsic<[llvm_v4i16_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_round24x2f48sasym: ClangBuiltin<"__builtin_xtensa_ae_round24x2f48sasym">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v1i64_ty, llvm_v1i64_ty], [IntrNoMem]>;

def int_xtensa_ae_round24x2f48ssym: ClangBuiltin<"__builtin_xtensa_ae_round24x2f48ssym">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v1i64_ty, llvm_v1i64_ty], [IntrNoMem]>;

def int_xtensa_ae_round32x2f48sasym: ClangBuiltin<"__builtin_xtensa_ae_round32x2f48sasym">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v1i64_ty, llvm_v1i64_ty], [IntrNoMem]>;

def int_xtensa_ae_round32x2f48ssym: ClangBuiltin<"__builtin_xtensa_ae_round32x2f48ssym">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v1i64_ty, llvm_v1i64_ty], [IntrNoMem]>;

def int_xtensa_ae_round32x2f64sasym: ClangBuiltin<"__builtin_xtensa_ae_round32x2f64sasym">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v1i64_ty, llvm_v1i64_ty], [IntrNoMem]>;

def int_xtensa_ae_round32x2f64ssym: ClangBuiltin<"__builtin_xtensa_ae_round32x2f64ssym">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v1i64_ty, llvm_v1i64_ty], [IntrNoMem]>;

def int_xtensa_ae_roundsp16f24asym: ClangBuiltin<"__builtin_xtensa_ae_roundsp16f24asym">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_roundsp16f24sym: ClangBuiltin<"__builtin_xtensa_ae_roundsp16f24sym">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_roundsp16q48x2asym: ClangBuiltin<"__builtin_xtensa_ae_roundsp16q48x2asym">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v1i64_ty, llvm_v1i64_ty], [IntrNoMem]>;

def int_xtensa_ae_roundsp16q48x2sym: ClangBuiltin<"__builtin_xtensa_ae_roundsp16q48x2sym">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v1i64_ty, llvm_v1i64_ty], [IntrNoMem]>;

def int_xtensa_ae_roundsq32f48asym: ClangBuiltin<"__builtin_xtensa_ae_roundsq32f48asym">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty], [IntrNoMem]>;

def int_xtensa_ae_roundsq32f48sym: ClangBuiltin<"__builtin_xtensa_ae_roundsq32f48sym">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty], [IntrNoMem]>;

def int_xtensa_ae_s16_0_i: ClangBuiltin<"__builtin_xtensa_ae_s16_0_i">,
  Intrinsic<[], [llvm_v4i16_ty, llvm_ptr_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>]>;

def int_xtensa_ae_s16_0_ip: 
  Intrinsic<[llvm_ptr_ty], [llvm_v4i16_ty, llvm_ptr_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>]>;

def int_xtensa_ae_s16_0_x: ClangBuiltin<"__builtin_xtensa_ae_s16_0_x">,
  Intrinsic<[], [llvm_v4i16_ty, llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_s16_0_xc: 
  Intrinsic<[llvm_ptr_ty], [llvm_v4i16_ty, llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_s16_0_xp: 
  Intrinsic<[llvm_ptr_ty], [llvm_v4i16_ty, llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_s16m_l_i: ClangBuiltin<"__builtin_xtensa_ae_s16m_l_i">,
  Intrinsic<[], [llvm_v2i32_ty, llvm_ptr_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>]>;

def int_xtensa_ae_s16m_l_iu: 
  Intrinsic<[llvm_ptr_ty], [llvm_v2i32_ty, llvm_ptr_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>]>;

def int_xtensa_ae_s16m_l_x: ClangBuiltin<"__builtin_xtensa_ae_s16m_l_x">,
  Intrinsic<[], [llvm_v2i32_ty, llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_s16m_l_xc: 
  Intrinsic<[llvm_ptr_ty], [llvm_v2i32_ty, llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_s16m_l_xu: 
  Intrinsic<[llvm_ptr_ty], [llvm_v2i32_ty, llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_s16x2m_i: ClangBuiltin<"__builtin_xtensa_ae_s16x2m_i">,
  Intrinsic<[], [llvm_v2i32_ty, llvm_ptr_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>]>;

def int_xtensa_ae_s16x2m_iu: 
  Intrinsic<[llvm_ptr_ty], [llvm_v2i32_ty, llvm_ptr_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>]>;

def int_xtensa_ae_s16x2m_x: ClangBuiltin<"__builtin_xtensa_ae_s16x2m_x">,
  Intrinsic<[], [llvm_v2i32_ty, llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_s16x2m_xc: 
  Intrinsic<[llvm_ptr_ty], [llvm_v2i32_ty, llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_s16x2m_xu: 
  Intrinsic<[llvm_ptr_ty], [llvm_v2i32_ty, llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_s16x4_i: ClangBuiltin<"__builtin_xtensa_ae_s16x4_i">,
  Intrinsic<[], [llvm_v4i16_ty, llvm_ptr_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>]>;

def int_xtensa_ae_s16x4_ip: 
  Intrinsic<[llvm_ptr_ty], [llvm_v4i16_ty, llvm_ptr_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>]>;

def int_xtensa_ae_s16x4_ric: 
  Intrinsic<[llvm_ptr_ty], [llvm_v4i16_ty, llvm_ptr_ty], []>;

def int_xtensa_ae_s16x4_rip: 
  Intrinsic<[llvm_ptr_ty], [llvm_v4i16_ty, llvm_ptr_ty], []>;

def int_xtensa_ae_s16x4_x: ClangBuiltin<"__builtin_xtensa_ae_s16x4_x">,
  Intrinsic<[], [llvm_v4i16_ty, llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_s16x4_xc: 
  Intrinsic<[llvm_ptr_ty], [llvm_v4i16_ty, llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_s16x4_xp: 
  Intrinsic<[llvm_ptr_ty], [llvm_v4i16_ty, llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_s24ra64s_i: ClangBuiltin<"__builtin_xtensa_ae_s24ra64s_i">,
  Intrinsic<[], [llvm_v1i64_ty, llvm_ptr_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>]>;

def int_xtensa_ae_s24ra64s_ip: 
  Intrinsic<[llvm_ptr_ty], [llvm_v1i64_ty, llvm_ptr_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>]>;

def int_xtensa_ae_s24ra64s_x: ClangBuiltin<"__builtin_xtensa_ae_s24ra64s_x">,
  Intrinsic<[], [llvm_v1i64_ty, llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_s24ra64s_xc: 
  Intrinsic<[llvm_ptr_ty], [llvm_v1i64_ty, llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_s24ra64s_xp: 
  Intrinsic<[llvm_ptr_ty], [llvm_v1i64_ty, llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_s24x2ra64s_ip: 
  Intrinsic<[llvm_ptr_ty], [llvm_v1i64_ty, llvm_v1i64_ty, llvm_ptr_ty], []>;

def int_xtensa_ae_s32_l_i: ClangBuiltin<"__builtin_xtensa_ae_s32_l_i">,
  Intrinsic<[], [llvm_v2i32_ty, llvm_ptr_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>]>;

def int_xtensa_ae_s32_l_ip: 
  Intrinsic<[llvm_ptr_ty], [llvm_v2i32_ty, llvm_ptr_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>]>;

def int_xtensa_ae_s32_l_x: ClangBuiltin<"__builtin_xtensa_ae_s32_l_x">,
  Intrinsic<[], [llvm_v2i32_ty, llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_s32_l_xc: 
  Intrinsic<[llvm_ptr_ty], [llvm_v2i32_ty, llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_s32_l_xp: 
  Intrinsic<[llvm_ptr_ty], [llvm_v2i32_ty, llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_s32f24_l_i: ClangBuiltin<"__builtin_xtensa_ae_s32f24_l_i">,
  Intrinsic<[], [llvm_v2i32_ty, llvm_ptr_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>]>;

def int_xtensa_ae_s32f24_l_ip: 
  Intrinsic<[llvm_ptr_ty], [llvm_v2i32_ty, llvm_ptr_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>]>;

def int_xtensa_ae_s32f24_l_x: ClangBuiltin<"__builtin_xtensa_ae_s32f24_l_x">,
  Intrinsic<[], [llvm_v2i32_ty, llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_s32f24_l_xc: 
  Intrinsic<[llvm_ptr_ty], [llvm_v2i32_ty, llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_s32f24_l_xp: 
  Intrinsic<[llvm_ptr_ty], [llvm_v2i32_ty, llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_s32m_i: ClangBuiltin<"__builtin_xtensa_ae_s32m_i">,
  Intrinsic<[], [llvm_v1i64_ty, llvm_ptr_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>]>;

def int_xtensa_ae_s32m_iu: 
  Intrinsic<[llvm_ptr_ty], [llvm_v1i64_ty, llvm_ptr_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>]>;

def int_xtensa_ae_s32m_x: ClangBuiltin<"__builtin_xtensa_ae_s32m_x">,
  Intrinsic<[], [llvm_v1i64_ty, llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_s32m_xc: 
  Intrinsic<[llvm_ptr_ty], [llvm_v1i64_ty, llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_s32m_xu: 
  Intrinsic<[llvm_ptr_ty], [llvm_v1i64_ty, llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_s32ra64s_i: ClangBuiltin<"__builtin_xtensa_ae_s32ra64s_i">,
  Intrinsic<[], [llvm_v1i64_ty, llvm_ptr_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>]>;

def int_xtensa_ae_s32ra64s_ip: 
  Intrinsic<[llvm_ptr_ty], [llvm_v1i64_ty, llvm_ptr_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>]>;

def int_xtensa_ae_s32ra64s_x: ClangBuiltin<"__builtin_xtensa_ae_s32ra64s_x">,
  Intrinsic<[], [llvm_v1i64_ty, llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_s32ra64s_xc: 
  Intrinsic<[llvm_ptr_ty], [llvm_v1i64_ty, llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_s32ra64s_xp: 
  Intrinsic<[llvm_ptr_ty], [llvm_v1i64_ty, llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_s32x2_i: ClangBuiltin<"__builtin_xtensa_ae_s32x2_i">,
  Intrinsic<[], [llvm_v2i32_ty, llvm_ptr_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>]>;

def int_xtensa_ae_s32x2_ip: 
  Intrinsic<[llvm_ptr_ty], [llvm_v2i32_ty, llvm_ptr_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>]>;

def int_xtensa_ae_s32x2_ric: 
  Intrinsic<[llvm_ptr_ty], [llvm_v2i32_ty, llvm_ptr_ty], []>;

def int_xtensa_ae_s32x2_rip: 
  Intrinsic<[llvm_ptr_ty], [llvm_v2i32_ty, llvm_ptr_ty], []>;

def int_xtensa_ae_s32x2_x: ClangBuiltin<"__builtin_xtensa_ae_s32x2_x">,
  Intrinsic<[], [llvm_v2i32_ty, llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_s32x2_xc: 
  Intrinsic<[llvm_ptr_ty], [llvm_v2i32_ty, llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_s32x2_xp: 
  Intrinsic<[llvm_ptr_ty], [llvm_v2i32_ty, llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_s32x2f24_i: ClangBuiltin<"__builtin_xtensa_ae_s32x2f24_i">,
  Intrinsic<[], [llvm_v2i32_ty, llvm_ptr_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>]>;

def int_xtensa_ae_s32x2f24_ip: 
  Intrinsic<[llvm_ptr_ty], [llvm_v2i32_ty, llvm_ptr_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>]>;

def int_xtensa_ae_s32x2f24_ric: 
  Intrinsic<[llvm_ptr_ty], [llvm_v2i32_ty, llvm_ptr_ty], []>;

def int_xtensa_ae_s32x2f24_rip: 
  Intrinsic<[llvm_ptr_ty], [llvm_v2i32_ty, llvm_ptr_ty], []>;

def int_xtensa_ae_s32x2f24_x: ClangBuiltin<"__builtin_xtensa_ae_s32x2f24_x">,
  Intrinsic<[], [llvm_v2i32_ty, llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_s32x2f24_xc: 
  Intrinsic<[llvm_ptr_ty], [llvm_v2i32_ty, llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_s32x2f24_xp: 
  Intrinsic<[llvm_ptr_ty], [llvm_v2i32_ty, llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_s32x2ra64s_ip: 
  Intrinsic<[llvm_ptr_ty], [llvm_v1i64_ty, llvm_v1i64_ty, llvm_ptr_ty], []>;

def int_xtensa_ae_s64_i: ClangBuiltin<"__builtin_xtensa_ae_s64_i">,
  Intrinsic<[], [llvm_v1i64_ty, llvm_ptr_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>]>;

def int_xtensa_ae_s64_ip: 
  Intrinsic<[llvm_ptr_ty], [llvm_v1i64_ty, llvm_ptr_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>]>;

def int_xtensa_ae_s64_x: ClangBuiltin<"__builtin_xtensa_ae_s64_x">,
  Intrinsic<[], [llvm_v1i64_ty, llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_s64_xc: 
  Intrinsic<[llvm_ptr_ty], [llvm_v1i64_ty, llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_s64_xp: 
  Intrinsic<[llvm_ptr_ty], [llvm_v1i64_ty, llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_ae_sa16x4_ic: 
  Intrinsic<[llvm_v8i8_ty, llvm_ptr_ty], [llvm_v4i16_ty, llvm_v8i8_ty, llvm_ptr_ty], []>;

def int_xtensa_ae_sa16x4_ip: 
  Intrinsic<[llvm_v8i8_ty, llvm_ptr_ty], [llvm_v4i16_ty, llvm_v8i8_ty, llvm_ptr_ty], []>;

def int_xtensa_ae_sa16x4_ric: 
  Intrinsic<[llvm_v8i8_ty, llvm_ptr_ty], [llvm_v4i16_ty, llvm_v8i8_ty, llvm_ptr_ty], []>;

def int_xtensa_ae_sa16x4_rip: 
  Intrinsic<[llvm_v8i8_ty, llvm_ptr_ty], [llvm_v4i16_ty, llvm_v8i8_ty, llvm_ptr_ty], []>;

def int_xtensa_ae_sa24_l_ic: 
  Intrinsic<[llvm_v8i8_ty, llvm_ptr_ty], [llvm_v2i32_ty, llvm_v8i8_ty, llvm_ptr_ty], []>;

def int_xtensa_ae_sa24_l_ip: 
  Intrinsic<[llvm_v8i8_ty, llvm_ptr_ty], [llvm_v2i32_ty, llvm_v8i8_ty, llvm_ptr_ty], []>;

def int_xtensa_ae_sa24_l_ric: 
  Intrinsic<[llvm_v8i8_ty, llvm_ptr_ty], [llvm_v2i32_ty, llvm_v8i8_ty, llvm_ptr_ty], []>;

def int_xtensa_ae_sa24_l_rip: 
  Intrinsic<[llvm_v8i8_ty, llvm_ptr_ty], [llvm_v2i32_ty, llvm_v8i8_ty, llvm_ptr_ty], []>;

def int_xtensa_ae_sa24x2_ic: 
  Intrinsic<[llvm_v8i8_ty, llvm_ptr_ty], [llvm_v2i32_ty, llvm_v8i8_ty, llvm_ptr_ty], []>;

def int_xtensa_ae_sa24x2_ip: 
  Intrinsic<[llvm_v8i8_ty, llvm_ptr_ty], [llvm_v2i32_ty, llvm_v8i8_ty, llvm_ptr_ty], []>;

def int_xtensa_ae_sa24x2_ric: 
  Intrinsic<[llvm_v8i8_ty, llvm_ptr_ty], [llvm_v2i32_ty, llvm_v8i8_ty, llvm_ptr_ty], []>;

def int_xtensa_ae_sa24x2_rip: 
  Intrinsic<[llvm_v8i8_ty, llvm_ptr_ty], [llvm_v2i32_ty, llvm_v8i8_ty, llvm_ptr_ty], []>;

def int_xtensa_ae_sa32x2_ic: 
  Intrinsic<[llvm_v8i8_ty, llvm_ptr_ty], [llvm_v2i32_ty, llvm_v8i8_ty, llvm_ptr_ty], []>;

def int_xtensa_ae_sa32x2_ip: 
  Intrinsic<[llvm_v8i8_ty, llvm_ptr_ty], [llvm_v2i32_ty, llvm_v8i8_ty, llvm_ptr_ty], []>;

def int_xtensa_ae_sa32x2_ric: 
  Intrinsic<[llvm_v8i8_ty, llvm_ptr_ty], [llvm_v2i32_ty, llvm_v8i8_ty, llvm_ptr_ty], []>;

def int_xtensa_ae_sa32x2_rip: 
  Intrinsic<[llvm_v8i8_ty, llvm_ptr_ty], [llvm_v2i32_ty, llvm_v8i8_ty, llvm_ptr_ty], []>;

def int_xtensa_ae_sa32x2f24_ic: 
  Intrinsic<[llvm_v8i8_ty, llvm_ptr_ty], [llvm_v2i32_ty, llvm_v8i8_ty, llvm_ptr_ty], []>;

def int_xtensa_ae_sa32x2f24_ip: 
  Intrinsic<[llvm_v8i8_ty, llvm_ptr_ty], [llvm_v2i32_ty, llvm_v8i8_ty, llvm_ptr_ty], []>;

def int_xtensa_ae_sa32x2f24_ric: 
  Intrinsic<[llvm_v8i8_ty, llvm_ptr_ty], [llvm_v2i32_ty, llvm_v8i8_ty, llvm_ptr_ty], []>;

def int_xtensa_ae_sa32x2f24_rip: 
  Intrinsic<[llvm_v8i8_ty, llvm_ptr_ty], [llvm_v2i32_ty, llvm_v8i8_ty, llvm_ptr_ty], []>;

def int_xtensa_ae_sa64neg_fp: 
  Intrinsic<[llvm_v8i8_ty], [llvm_v8i8_ty, llvm_ptr_ty], []>;

def int_xtensa_ae_sa64pos_fp: 
  Intrinsic<[llvm_v8i8_ty], [llvm_v8i8_ty, llvm_ptr_ty], []>;

def int_xtensa_ae_salign64_i: ClangBuiltin<"__builtin_xtensa_ae_salign64_i">,
  Intrinsic<[], [llvm_v8i8_ty, llvm_ptr_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>]>;

def int_xtensa_ae_sat16x4: ClangBuiltin<"__builtin_xtensa_ae_sat16x4">,
  Intrinsic<[llvm_v4i16_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_sat24s: ClangBuiltin<"__builtin_xtensa_ae_sat24s">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_sat48s: ClangBuiltin<"__builtin_xtensa_ae_sat48s">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty], [IntrNoMem]>;

def int_xtensa_ae_satq56s: ClangBuiltin<"__builtin_xtensa_ae_satq56s">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty], [IntrNoMem]>;

def int_xtensa_ae_sb: 
  Intrinsic<[llvm_ptr_ty], [llvm_ptr_ty, llvm_i32_ty], [IntrNoMem]>;

def int_xtensa_ae_sb_ic: 
  Intrinsic<[llvm_ptr_ty], [llvm_ptr_ty, llvm_i32_ty], [IntrNoMem]>;

def int_xtensa_ae_sb_ip: 
  Intrinsic<[llvm_ptr_ty], [llvm_ptr_ty, llvm_i32_ty], [IntrNoMem]>;

def int_xtensa_ae_sbf: 
  Intrinsic<[llvm_ptr_ty], [llvm_ptr_ty], [IntrNoMem]>;

def int_xtensa_ae_sbf_ic: 
  Intrinsic<[llvm_ptr_ty], [llvm_ptr_ty], [IntrNoMem]>;

def int_xtensa_ae_sbf_ip: 
  Intrinsic<[llvm_ptr_ty], [llvm_ptr_ty], [IntrNoMem]>;

def int_xtensa_ae_sbi: 
  Intrinsic<[llvm_ptr_ty], [llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>, IntrNoMem]>;

def int_xtensa_ae_sbi_ic: 
  Intrinsic<[llvm_ptr_ty], [llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>, IntrNoMem]>;

def int_xtensa_ae_sbi_ip: 
  Intrinsic<[llvm_ptr_ty], [llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>, IntrNoMem]>;

def int_xtensa_ae_sel16i: ClangBuiltin<"__builtin_xtensa_ae_sel16i">,
  Intrinsic<[llvm_v4i16_ty], [llvm_v4i16_ty, llvm_v4i16_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>, IntrNoMem]>;

def int_xtensa_ae_sel16i_n: ClangBuiltin<"__builtin_xtensa_ae_sel16i_n">,
  Intrinsic<[llvm_v4i16_ty], [llvm_v4i16_ty, llvm_v4i16_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>, IntrNoMem]>;

def int_xtensa_ae_sext32: ClangBuiltin<"__builtin_xtensa_ae_sext32">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, IntrNoMem]>;

def int_xtensa_ae_sext32x2d16_10: ClangBuiltin<"__builtin_xtensa_ae_sext32x2d16_10">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_sext32x2d16_32: ClangBuiltin<"__builtin_xtensa_ae_sext32x2d16_32">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_sha32: ClangBuiltin<"__builtin_xtensa_ae_sha32">,
  Intrinsic<[llvm_i32_ty], [llvm_i32_ty], [IntrNoMem]>;

def int_xtensa_ae_shortswap: ClangBuiltin<"__builtin_xtensa_ae_shortswap">,
  Intrinsic<[llvm_v4i16_ty], [llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_slaa16s: ClangBuiltin<"__builtin_xtensa_ae_slaa16s">,
  Intrinsic<[llvm_v4i16_ty], [llvm_v4i16_ty, llvm_i32_ty], [IntrNoMem]>;

def int_xtensa_ae_slaa32: ClangBuiltin<"__builtin_xtensa_ae_slaa32">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_i32_ty], [IntrNoMem]>;

def int_xtensa_ae_slaa32s: ClangBuiltin<"__builtin_xtensa_ae_slaa32s">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_i32_ty], [IntrNoMem]>;

def int_xtensa_ae_slaa64: ClangBuiltin<"__builtin_xtensa_ae_slaa64">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_i32_ty], [IntrNoMem]>;

def int_xtensa_ae_slaa64s: ClangBuiltin<"__builtin_xtensa_ae_slaa64s">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_i32_ty], [IntrNoMem]>;

def int_xtensa_ae_slaaq56: ClangBuiltin<"__builtin_xtensa_ae_slaaq56">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_i32_ty], [IntrNoMem]>;

def int_xtensa_ae_slai16s: ClangBuiltin<"__builtin_xtensa_ae_slai16s">,
  Intrinsic<[llvm_v4i16_ty], [llvm_v4i16_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, IntrNoMem]>;

def int_xtensa_ae_slai24: ClangBuiltin<"__builtin_xtensa_ae_slai24">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, IntrNoMem]>;

def int_xtensa_ae_slai24s: ClangBuiltin<"__builtin_xtensa_ae_slai24s">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, IntrNoMem]>;

def int_xtensa_ae_slai32: ClangBuiltin<"__builtin_xtensa_ae_slai32">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, IntrNoMem]>;

def int_xtensa_ae_slai32s: ClangBuiltin<"__builtin_xtensa_ae_slai32s">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, IntrNoMem]>;

def int_xtensa_ae_slai64: ClangBuiltin<"__builtin_xtensa_ae_slai64">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, IntrNoMem]>;

def int_xtensa_ae_slai64s: ClangBuiltin<"__builtin_xtensa_ae_slai64s">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, IntrNoMem]>;

def int_xtensa_ae_slaisq56s: ClangBuiltin<"__builtin_xtensa_ae_slaisq56s">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, IntrNoMem]>;

def int_xtensa_ae_slas24: ClangBuiltin<"__builtin_xtensa_ae_slas24">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_slas24s: ClangBuiltin<"__builtin_xtensa_ae_slas24s">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_slas32: ClangBuiltin<"__builtin_xtensa_ae_slas32">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_slas32s: ClangBuiltin<"__builtin_xtensa_ae_slas32s">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_slas64: ClangBuiltin<"__builtin_xtensa_ae_slas64">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty], [IntrNoMem]>;

def int_xtensa_ae_slas64s: ClangBuiltin<"__builtin_xtensa_ae_slas64s">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty], [IntrNoMem]>;

def int_xtensa_ae_slasq56: ClangBuiltin<"__builtin_xtensa_ae_slasq56">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty], [IntrNoMem]>;

def int_xtensa_ae_slassq56s: ClangBuiltin<"__builtin_xtensa_ae_slassq56s">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty], [IntrNoMem]>;

def int_xtensa_ae_sra64_32: ClangBuiltin<"__builtin_xtensa_ae_sra64_32">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v2i32_ty, llvm_i32_ty], [IntrNoMem]>;

def int_xtensa_ae_sraa16rs: ClangBuiltin<"__builtin_xtensa_ae_sraa16rs">,
  Intrinsic<[llvm_v4i16_ty], [llvm_v4i16_ty, llvm_i32_ty], [IntrNoMem]>;

def int_xtensa_ae_sraa16s: ClangBuiltin<"__builtin_xtensa_ae_sraa16s">,
  Intrinsic<[llvm_v4i16_ty], [llvm_v4i16_ty, llvm_i32_ty], [IntrNoMem]>;

def int_xtensa_ae_sraa32: ClangBuiltin<"__builtin_xtensa_ae_sraa32">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_i32_ty], [IntrNoMem]>;

def int_xtensa_ae_sraa32rs: ClangBuiltin<"__builtin_xtensa_ae_sraa32rs">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_i32_ty], [IntrNoMem]>;

def int_xtensa_ae_sraa32s: ClangBuiltin<"__builtin_xtensa_ae_sraa32s">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_i32_ty], [IntrNoMem]>;

def int_xtensa_ae_sraa64: ClangBuiltin<"__builtin_xtensa_ae_sraa64">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_i32_ty], [IntrNoMem]>;

def int_xtensa_ae_srai16: ClangBuiltin<"__builtin_xtensa_ae_srai16">,
  Intrinsic<[llvm_v4i16_ty], [llvm_v4i16_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, IntrNoMem]>;

def int_xtensa_ae_srai16r: ClangBuiltin<"__builtin_xtensa_ae_srai16r">,
  Intrinsic<[llvm_v4i16_ty], [llvm_v4i16_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, IntrNoMem]>;

def int_xtensa_ae_srai24: ClangBuiltin<"__builtin_xtensa_ae_srai24">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, IntrNoMem]>;

def int_xtensa_ae_srai32: ClangBuiltin<"__builtin_xtensa_ae_srai32">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, IntrNoMem]>;

def int_xtensa_ae_srai32r: ClangBuiltin<"__builtin_xtensa_ae_srai32r">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, IntrNoMem]>;

def int_xtensa_ae_srai64: ClangBuiltin<"__builtin_xtensa_ae_srai64">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, IntrNoMem]>;

def int_xtensa_ae_sras24: ClangBuiltin<"__builtin_xtensa_ae_sras24">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_sras32: ClangBuiltin<"__builtin_xtensa_ae_sras32">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_sras64: ClangBuiltin<"__builtin_xtensa_ae_sras64">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty], [IntrNoMem]>;

def int_xtensa_ae_srla32: ClangBuiltin<"__builtin_xtensa_ae_srla32">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_i32_ty], [IntrNoMem]>;

def int_xtensa_ae_srla64: ClangBuiltin<"__builtin_xtensa_ae_srla64">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_i32_ty], [IntrNoMem]>;

def int_xtensa_ae_srli24: ClangBuiltin<"__builtin_xtensa_ae_srli24">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, IntrNoMem]>;

def int_xtensa_ae_srli32: ClangBuiltin<"__builtin_xtensa_ae_srli32">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, IntrNoMem]>;

def int_xtensa_ae_srli64: ClangBuiltin<"__builtin_xtensa_ae_srli64">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, IntrNoMem]>;

def int_xtensa_ae_srls24: ClangBuiltin<"__builtin_xtensa_ae_srls24">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_srls32: ClangBuiltin<"__builtin_xtensa_ae_srls32">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_srls64: ClangBuiltin<"__builtin_xtensa_ae_srls64">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty], [IntrNoMem]>;

def int_xtensa_ae_sub16: ClangBuiltin<"__builtin_xtensa_ae_sub16">,
  Intrinsic<[llvm_v4i16_ty], [llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_sub16s: ClangBuiltin<"__builtin_xtensa_ae_sub16s">,
  Intrinsic<[llvm_v4i16_ty], [llvm_v4i16_ty, llvm_v4i16_ty], [IntrNoMem]>;

def int_xtensa_ae_sub24s: ClangBuiltin<"__builtin_xtensa_ae_sub24s">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_sub32: ClangBuiltin<"__builtin_xtensa_ae_sub32">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_sub32s: ClangBuiltin<"__builtin_xtensa_ae_sub32s">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_sub64: ClangBuiltin<"__builtin_xtensa_ae_sub64">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v1i64_ty], [IntrNoMem]>;

def int_xtensa_ae_sub64s: ClangBuiltin<"__builtin_xtensa_ae_sub64s">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v1i64_ty], [IntrNoMem]>;

def int_xtensa_ae_subadd32: ClangBuiltin<"__builtin_xtensa_ae_subadd32">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_subadd32s: ClangBuiltin<"__builtin_xtensa_ae_subadd32s">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v2i32_ty], [IntrNoMem]>;

def int_xtensa_ae_trunca32f64s_l: ClangBuiltin<"__builtin_xtensa_ae_trunca32f64s_l">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v1i64_ty, llvm_i32_ty], [IntrNoMem]>;

def int_xtensa_ae_trunca32x2f64s: ClangBuiltin<"__builtin_xtensa_ae_trunca32x2f64s">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v1i64_ty, llvm_v1i64_ty, llvm_i32_ty], [IntrNoMem]>;

def int_xtensa_ae_trunci32f64s_l: ClangBuiltin<"__builtin_xtensa_ae_trunci32f64s_l">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v2i32_ty, llvm_v1i64_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>, IntrNoMem]>;

def int_xtensa_ae_trunci32x2f64s: ClangBuiltin<"__builtin_xtensa_ae_trunci32x2f64s">,
  Intrinsic<[llvm_v2i32_ty], [llvm_v1i64_ty, llvm_v1i64_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>, IntrNoMem]>;

def int_xtensa_ae_vldl16c: 
  Intrinsic<[llvm_ptr_ty], [llvm_ptr_ty], [IntrNoMem]>;

def int_xtensa_ae_vldl16c_ic: 
  Intrinsic<[llvm_ptr_ty], [llvm_ptr_ty], [IntrNoMem]>;

def int_xtensa_ae_vldl16c_ip: 
  Intrinsic<[llvm_ptr_ty], [llvm_ptr_ty], [IntrNoMem]>;

def int_xtensa_ae_vldl16t: 
  Intrinsic<[llvm_v1i1_ty, llvm_i32_ty], [llvm_ptr_ty], [IntrNoMem]>;

def int_xtensa_ae_vldl32t: 
  Intrinsic<[llvm_v1i1_ty, llvm_i32_ty], [llvm_ptr_ty], [IntrNoMem]>;

def int_xtensa_ae_vldsht: ClangBuiltin<"__builtin_xtensa_ae_vldsht">,
  Intrinsic<[], [llvm_i32_ty], []>;

def int_xtensa_ae_vlel16t: 
  Intrinsic<[llvm_v1i1_ty, llvm_i32_ty], [llvm_i32_ty, llvm_ptr_ty], [IntrNoMem]>;

def int_xtensa_ae_vlel32t: 
  Intrinsic<[llvm_v1i1_ty, llvm_i32_ty], [llvm_i32_ty, llvm_ptr_ty], [IntrNoMem]>;

def int_xtensa_ae_vles16c: 
  Intrinsic<[llvm_ptr_ty], [llvm_ptr_ty], [IntrNoMem]>;

def int_xtensa_ae_vles16c_ic: 
  Intrinsic<[llvm_ptr_ty], [llvm_ptr_ty], [IntrNoMem]>;

def int_xtensa_ae_vles16c_ip: 
  Intrinsic<[llvm_ptr_ty], [llvm_ptr_ty], [IntrNoMem]>;

def int_xtensa_ae_xor: ClangBuiltin<"__builtin_xtensa_ae_xor">,
  Intrinsic<[llvm_v1i64_ty], [llvm_v1i64_ty, llvm_v1i64_ty], [IntrNoMem]>;

def int_xtensa_ae_zalign64: ClangBuiltin<"__builtin_xtensa_ae_zalign64">,
  Intrinsic<[llvm_v8i8_ty], [], [IntrNoMem]>;

def int_xtensa_rur_ae_bithead: ClangBuiltin<"__builtin_xtensa_rur_ae_bithead">,
  Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>;

def int_xtensa_rur_ae_bitptr: ClangBuiltin<"__builtin_xtensa_rur_ae_bitptr">,
  Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>;

def int_xtensa_rur_ae_bitsused: ClangBuiltin<"__builtin_xtensa_rur_ae_bitsused">,
  Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>;

def int_xtensa_rur_ae_cbegin0: ClangBuiltin<"__builtin_xtensa_rur_ae_cbegin0">,
  Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>;

def int_xtensa_rur_ae_cend0: ClangBuiltin<"__builtin_xtensa_rur_ae_cend0">,
  Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>;

def int_xtensa_rur_ae_cw_sd_no: ClangBuiltin<"__builtin_xtensa_rur_ae_cw_sd_no">,
  Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>;

def int_xtensa_rur_ae_cwrap: ClangBuiltin<"__builtin_xtensa_rur_ae_cwrap">,
  Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>;

def int_xtensa_rur_ae_first_ts: ClangBuiltin<"__builtin_xtensa_rur_ae_first_ts">,
  Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>;

def int_xtensa_rur_ae_nextoffset: ClangBuiltin<"__builtin_xtensa_rur_ae_nextoffset">,
  Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>;

def int_xtensa_rur_ae_overflow: ClangBuiltin<"__builtin_xtensa_rur_ae_overflow">,
  Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>;

def int_xtensa_rur_ae_ovf_sar: ClangBuiltin<"__builtin_xtensa_rur_ae_ovf_sar">,
  Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>;

def int_xtensa_rur_ae_sar: ClangBuiltin<"__builtin_xtensa_rur_ae_sar">,
  Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>;

def int_xtensa_rur_ae_searchdone: ClangBuiltin<"__builtin_xtensa_rur_ae_searchdone">,
  Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>;

def int_xtensa_rur_ae_tablesize: ClangBuiltin<"__builtin_xtensa_rur_ae_tablesize">,
  Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>;

def int_xtensa_rur_ae_ts_fts_bu_bp: ClangBuiltin<"__builtin_xtensa_rur_ae_ts_fts_bu_bp">,
  Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>;

def int_xtensa_wur_ae_bithead: ClangBuiltin<"__builtin_xtensa_wur_ae_bithead">,
  Intrinsic<[], [llvm_i32_ty], []>;

def int_xtensa_wur_ae_bitptr: ClangBuiltin<"__builtin_xtensa_wur_ae_bitptr">,
  Intrinsic<[], [llvm_i32_ty], []>;

def int_xtensa_wur_ae_bitsused: ClangBuiltin<"__builtin_xtensa_wur_ae_bitsused">,
  Intrinsic<[], [llvm_i32_ty], []>;

def int_xtensa_wur_ae_cbegin0: ClangBuiltin<"__builtin_xtensa_wur_ae_cbegin0">,
  Intrinsic<[], [llvm_i32_ty], []>;

def int_xtensa_wur_ae_cend0: ClangBuiltin<"__builtin_xtensa_wur_ae_cend0">,
  Intrinsic<[], [llvm_i32_ty], []>;

def int_xtensa_wur_ae_cw_sd_no: ClangBuiltin<"__builtin_xtensa_wur_ae_cw_sd_no">,
  Intrinsic<[], [llvm_i32_ty], []>;

def int_xtensa_wur_ae_cwrap: ClangBuiltin<"__builtin_xtensa_wur_ae_cwrap">,
  Intrinsic<[], [llvm_i32_ty], []>;

def int_xtensa_wur_ae_first_ts: ClangBuiltin<"__builtin_xtensa_wur_ae_first_ts">,
  Intrinsic<[], [llvm_i32_ty], []>;

def int_xtensa_wur_ae_nextoffset: ClangBuiltin<"__builtin_xtensa_wur_ae_nextoffset">,
  Intrinsic<[], [llvm_i32_ty], []>;

def int_xtensa_wur_ae_overflow: ClangBuiltin<"__builtin_xtensa_wur_ae_overflow">,
  Intrinsic<[], [llvm_i32_ty], []>;

def int_xtensa_wur_ae_ovf_sar: ClangBuiltin<"__builtin_xtensa_wur_ae_ovf_sar">,
  Intrinsic<[], [llvm_i32_ty], []>;

def int_xtensa_wur_ae_sar: ClangBuiltin<"__builtin_xtensa_wur_ae_sar">,
  Intrinsic<[], [llvm_i32_ty], []>;

def int_xtensa_wur_ae_searchdone: ClangBuiltin<"__builtin_xtensa_wur_ae_searchdone">,
  Intrinsic<[], [llvm_i32_ty], []>;

def int_xtensa_wur_ae_tablesize: ClangBuiltin<"__builtin_xtensa_wur_ae_tablesize">,
  Intrinsic<[], [llvm_i32_ty], []>;

def int_xtensa_wur_ae_ts_fts_bu_bp: ClangBuiltin<"__builtin_xtensa_wur_ae_ts_fts_bu_bp">,
  Intrinsic<[], [llvm_i32_ty], []>;

