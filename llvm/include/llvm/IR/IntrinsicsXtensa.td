//===- IntrinsicsXtensa.td - Defines Xtensa intrinsics -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the Xtensa-specific intrinsics.
//
//===----------------------------------------------------------------------===//

let TargetPrefix = "xtensa" in {  // All intrinsics start with "llvm.xtensa.".

def int_xtensa_umul_aa_ll: ClangBuiltin<"__builtin_xtensa_umul_aa_ll">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], []>;
def int_xtensa_umul_aa_hl: ClangBuiltin<"__builtin_xtensa_umul_aa_hl">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], []>;
def int_xtensa_umul_aa_lh: ClangBuiltin<"__builtin_xtensa_umul_aa_lh">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], []>;
def int_xtensa_umul_aa_hh: ClangBuiltin<"__builtin_xtensa_umul_aa_hh">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], []>;

def int_xtensa_mul_aa_ll: ClangBuiltin<"__builtin_xtensa_mul_aa_ll">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], []>;
def int_xtensa_mul_aa_hl: ClangBuiltin<"__builtin_xtensa_mul_aa_hl">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], []>;
def int_xtensa_mul_aa_lh: ClangBuiltin<"__builtin_xtensa_mul_aa_lh">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], []>;
def int_xtensa_mul_aa_hh: ClangBuiltin<"__builtin_xtensa_mul_aa_hh">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], []>;

def int_xtensa_mul_ad_ll: ClangBuiltin<"__builtin_xtensa_mul_ad_ll">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;
def int_xtensa_mul_ad_hl: ClangBuiltin<"__builtin_xtensa_mul_ad_hl">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;
def int_xtensa_mul_ad_lh: ClangBuiltin<"__builtin_xtensa_mul_ad_lh">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;
def int_xtensa_mul_ad_hh: ClangBuiltin<"__builtin_xtensa_mul_ad_hh">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_xtensa_mul_da_ll: ClangBuiltin<"__builtin_xtensa_mul_da_ll">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;
def int_xtensa_mul_da_hl: ClangBuiltin<"__builtin_xtensa_mul_da_hl">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;
def int_xtensa_mul_da_lh: ClangBuiltin<"__builtin_xtensa_mul_da_lh">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;
def int_xtensa_mul_da_hh: ClangBuiltin<"__builtin_xtensa_mul_da_hh">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;

def int_xtensa_mul_dd_ll: ClangBuiltin<"__builtin_xtensa_mul_dd_ll">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;
def int_xtensa_mul_dd_hl: ClangBuiltin<"__builtin_xtensa_mul_dd_hl">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;
def int_xtensa_mul_dd_lh: ClangBuiltin<"__builtin_xtensa_mul_dd_lh">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;
def int_xtensa_mul_dd_hh: ClangBuiltin<"__builtin_xtensa_mul_dd_hh">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

def int_xtensa_mula_aa_ll: ClangBuiltin<"__builtin_xtensa_mula_aa_ll">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], []>;
def int_xtensa_mula_aa_hl: ClangBuiltin<"__builtin_xtensa_mula_aa_hl">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], []>;
def int_xtensa_mula_aa_lh: ClangBuiltin<"__builtin_xtensa_mula_aa_lh">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], []>;
def int_xtensa_mula_aa_hh: ClangBuiltin<"__builtin_xtensa_mula_aa_hh">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], []>;

def int_xtensa_mula_ad_ll: ClangBuiltin<"__builtin_xtensa_mula_ad_ll">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;
def int_xtensa_mula_ad_hl: ClangBuiltin<"__builtin_xtensa_mula_ad_hl">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;
def int_xtensa_mula_ad_lh: ClangBuiltin<"__builtin_xtensa_mula_ad_lh">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;
def int_xtensa_mula_ad_hh: ClangBuiltin<"__builtin_xtensa_mula_ad_hh">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_xtensa_mula_da_ll: ClangBuiltin<"__builtin_xtensa_mula_da_ll">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;
def int_xtensa_mula_da_hl: ClangBuiltin<"__builtin_xtensa_mula_da_hl">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;
def int_xtensa_mula_da_lh: ClangBuiltin<"__builtin_xtensa_mula_da_lh">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;
def int_xtensa_mula_da_hh: ClangBuiltin<"__builtin_xtensa_mula_da_hh">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;

def int_xtensa_mula_dd_ll: ClangBuiltin<"__builtin_xtensa_mula_dd_ll">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;
def int_xtensa_mula_dd_hl: ClangBuiltin<"__builtin_xtensa_mula_dd_hl">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;
def int_xtensa_mula_dd_lh: ClangBuiltin<"__builtin_xtensa_mula_dd_lh">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;
def int_xtensa_mula_dd_hh: ClangBuiltin<"__builtin_xtensa_mula_dd_hh">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

def int_xtensa_muls_aa_ll: ClangBuiltin<"__builtin_xtensa_muls_aa_ll">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], []>;
def int_xtensa_muls_aa_hl: ClangBuiltin<"__builtin_xtensa_muls_aa_hl">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], []>;
def int_xtensa_muls_aa_lh: ClangBuiltin<"__builtin_xtensa_muls_aa_lh">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], []>;
def int_xtensa_muls_aa_hh: ClangBuiltin<"__builtin_xtensa_muls_aa_hh">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], []>;

def int_xtensa_muls_ad_ll: ClangBuiltin<"__builtin_xtensa_muls_ad_ll">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;
def int_xtensa_muls_ad_hl: ClangBuiltin<"__builtin_xtensa_muls_ad_hl">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;
def int_xtensa_muls_ad_lh: ClangBuiltin<"__builtin_xtensa_muls_ad_lh">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;
def int_xtensa_muls_ad_hh: ClangBuiltin<"__builtin_xtensa_muls_ad_hh">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_xtensa_muls_da_ll: ClangBuiltin<"__builtin_xtensa_muls_da_ll">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;
def int_xtensa_muls_da_hl: ClangBuiltin<"__builtin_xtensa_muls_da_hl">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;
def int_xtensa_muls_da_lh: ClangBuiltin<"__builtin_xtensa_muls_da_lh">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;
def int_xtensa_muls_da_hh: ClangBuiltin<"__builtin_xtensa_muls_da_hh">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;

def int_xtensa_muls_dd_ll: ClangBuiltin<"__builtin_xtensa_muls_dd_ll">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;
def int_xtensa_muls_dd_hl: ClangBuiltin<"__builtin_xtensa_muls_dd_hl">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;
def int_xtensa_muls_dd_lh: ClangBuiltin<"__builtin_xtensa_muls_dd_lh">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;
def int_xtensa_muls_dd_hh: ClangBuiltin<"__builtin_xtensa_muls_dd_hh">,
  Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;


def int_xtensa_mula_da_ll_lddec: ClangBuiltin<"__builtin_xtensa_mula_da_ll_lddec">,
  Intrinsic<[], [llvm_i32_ty, llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
            [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>]>;
def int_xtensa_mula_da_lh_lddec: ClangBuiltin<"__builtin_xtensa_mula_da_lh_lddec">,
  Intrinsic<[], [llvm_i32_ty, llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
            [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>]>;
def int_xtensa_mula_da_hl_lddec: ClangBuiltin<"__builtin_xtensa_mula_da_hl_lddec">,
  Intrinsic<[], [llvm_i32_ty, llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
            [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>]>;
def int_xtensa_mula_da_hh_lddec: ClangBuiltin<"__builtin_xtensa_mula_da_hh_lddec">,
  Intrinsic<[], [llvm_i32_ty, llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
            [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>]>;

def int_xtensa_mula_da_ll_ldinc: ClangBuiltin<"__builtin_xtensa_mula_da_ll_ldinc">,
  Intrinsic<[], [llvm_i32_ty, llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
            [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>]>;
def int_xtensa_mula_da_lh_ldinc: ClangBuiltin<"__builtin_xtensa_mula_da_lh_ldinc">,
  Intrinsic<[], [llvm_i32_ty, llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
            [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>]>;
def int_xtensa_mula_da_hl_ldinc: ClangBuiltin<"__builtin_xtensa_mula_da_hl_ldinc">,
  Intrinsic<[], [llvm_i32_ty, llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
            [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>]>;
def int_xtensa_mula_da_hh_ldinc: ClangBuiltin<"__builtin_xtensa_mula_da_hh_ldinc">,
  Intrinsic<[], [llvm_i32_ty, llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
            [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>]>;

def int_xtensa_mula_dd_ll_lddec: ClangBuiltin<"__builtin_xtensa_mula_dd_ll_lddec">,
  Intrinsic<[], [llvm_i32_ty, llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
            [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;
def int_xtensa_mula_dd_lh_lddec: ClangBuiltin<"__builtin_xtensa_mula_dd_lh_lddec">,
  Intrinsic<[], [llvm_i32_ty, llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
            [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;
def int_xtensa_mula_dd_hl_lddec: ClangBuiltin<"__builtin_xtensa_mula_dd_hl_lddec">,
  Intrinsic<[], [llvm_i32_ty, llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
            [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;
def int_xtensa_mula_dd_hh_lddec: ClangBuiltin<"__builtin_xtensa_mula_dd_hh_lddec">,
  Intrinsic<[], [llvm_i32_ty, llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
            [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_xtensa_mula_dd_ll_ldinc: ClangBuiltin<"__builtin_xtensa_mula_dd_ll_ldinc">,
  Intrinsic<[], [llvm_i32_ty, llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
            [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;
def int_xtensa_mula_dd_lh_ldinc: ClangBuiltin<"__builtin_xtensa_mula_dd_lh_ldinc">,
  Intrinsic<[], [llvm_i32_ty, llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
            [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;
def int_xtensa_mula_dd_hl_ldinc: ClangBuiltin<"__builtin_xtensa_mula_dd_hl_ldinc">,
  Intrinsic<[], [llvm_i32_ty, llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
            [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;
def int_xtensa_mula_dd_hh_ldinc: ClangBuiltin<"__builtin_xtensa_mula_dd_hh_ldinc">,
  Intrinsic<[], [llvm_i32_ty, llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
            [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

//===----------------------------------------------------------------------===//
// Load operations

def int_xtensa_lddec: ClangBuiltin<"__builtin_xtensa_lddec">,
  Intrinsic<[], [llvm_i32_ty, llvm_ptr_ty], [ImmArg<ArgIndex<0>>]>;

def int_xtensa_ldinc: ClangBuiltin<"__builtin_xtensa_ldinc">,
  Intrinsic<[], [llvm_i32_ty, llvm_ptr_ty], [ImmArg<ArgIndex<0>>]>;

//===----------------------------------------------------------------------===//
// WSR/XSR/RSR

def int_xtensa_wsr_acclo: ClangBuiltin<"__builtin_xtensa_wsr_acclo">,
  Intrinsic<[], [llvm_i32_ty], []>;

def int_xtensa_rsr_acclo: ClangBuiltin<"__builtin_xtensa_rsr_acclo">,
  Intrinsic<[llvm_i32_ty], [], []>;

def int_xtensa_xsr_acclo: ClangBuiltin<"__builtin_xtensa_xsr_acclo">,
  Intrinsic<[], [llvm_ptr_ty], []>;

def int_xtensa_wsr_acchi: ClangBuiltin<"__builtin_xtensa_wsr_acchi">,
  Intrinsic<[], [llvm_i32_ty], []>;

def int_xtensa_rsr_acchi: ClangBuiltin<"__builtin_xtensa_rsr_acchi">,
  Intrinsic<[llvm_i32_ty], [], []>;

def int_xtensa_xsr_acchi: ClangBuiltin<"__builtin_xtensa_xsr_acchi">,
  Intrinsic<[], [llvm_ptr_ty], []>;

def int_xtensa_wsr_m0: ClangBuiltin<"__builtin_xtensa_wsr_m0">,
  Intrinsic<[], [llvm_i32_ty], []>;

def int_xtensa_rsr_m0: ClangBuiltin<"__builtin_xtensa_rsr_m0">,
  Intrinsic<[llvm_i32_ty]>;

def int_xtensa_xsr_m0: ClangBuiltin<"__builtin_xtensa_xsr_m0">,
  Intrinsic<[], [llvm_ptr_ty], []>;

def int_xtensa_wsr_m1: ClangBuiltin<"__builtin_xtensa_wsr_m1">,
  Intrinsic<[], [llvm_i32_ty], []>;

def int_xtensa_rsr_m1: ClangBuiltin<"__builtin_xtensa_rsr_m1">,
  Intrinsic<[llvm_i32_ty], [], []>;

def int_xtensa_xsr_m1: ClangBuiltin<"__builtin_xtensa_xsr_m1">,
  Intrinsic<[], [llvm_ptr_ty], []>;

def int_xtensa_wsr_m2: ClangBuiltin<"__builtin_xtensa_wsr_m2">,
  Intrinsic<[], [llvm_i32_ty], []>;

def int_xtensa_rsr_m2: ClangBuiltin<"__builtin_xtensa_rsr_m2">,
  Intrinsic<[llvm_i32_ty], [], []>;

def int_xtensa_xsr_m2: ClangBuiltin<"__builtin_xtensa_xsr_m2">,
  Intrinsic<[], [llvm_ptr_ty], []>;

def int_xtensa_wsr_m3: ClangBuiltin<"__builtin_xtensa_wsr_m3">,
  Intrinsic<[], [llvm_i32_ty], []>;

def int_xtensa_rsr_m3: ClangBuiltin<"__builtin_xtensa_rsr_m3">,
  Intrinsic<[llvm_i32_ty], [], []>;

def int_xtensa_xsr_m3: ClangBuiltin<"__builtin_xtensa_xsr_m3">,
  Intrinsic<[], [llvm_ptr_ty], []>;

//===----------------------------------------------------------------------===//
// Float operations

def int_xtensa_xt_movt_s: ClangBuiltin<"__builtin_xtensa_xt_movt_s">,
  Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty, llvm_v1i1_ty], [IntrNoMem]>;

def int_xtensa_xt_movf_s: ClangBuiltin<"__builtin_xtensa_xt_movf_s">,
  Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty, llvm_v1i1_ty], [IntrNoMem]>;

def int_xtensa_xt_oeq_s: ClangBuiltin<"__builtin_xtensa_xt_oeq_s">,
  Intrinsic<[llvm_v1i1_ty], [llvm_float_ty, llvm_float_ty], [IntrNoMem]>;

def int_xtensa_xt_ole_s: ClangBuiltin<"__builtin_xtensa_xt_ole_s">,
  Intrinsic<[llvm_v1i1_ty], [llvm_float_ty, llvm_float_ty], [IntrNoMem]>;

def int_xtensa_xt_olt_s: ClangBuiltin<"__builtin_xtensa_xt_olt_s">,
  Intrinsic<[llvm_v1i1_ty], [llvm_float_ty, llvm_float_ty], [IntrNoMem]>;

def int_xtensa_xt_ueq_s: ClangBuiltin<"__builtin_xtensa_xt_ueq_s">,
  Intrinsic<[llvm_v1i1_ty], [llvm_float_ty, llvm_float_ty], [IntrNoMem]>;

def int_xtensa_xt_ule_s: ClangBuiltin<"__builtin_xtensa_xt_ule_s">,
  Intrinsic<[llvm_v1i1_ty], [llvm_float_ty, llvm_float_ty], [IntrNoMem]>;

def int_xtensa_xt_ult_s: ClangBuiltin<"__builtin_xtensa_xt_ult_s">,
  Intrinsic<[llvm_v1i1_ty], [llvm_float_ty, llvm_float_ty], [IntrNoMem]>;

def int_xtensa_xt_un_s: ClangBuiltin<"__builtin_xtensa_xt_un_s">,
  Intrinsic<[llvm_v1i1_ty], [llvm_float_ty, llvm_float_ty], [IntrNoMem]>;

def int_xtensa_xt_add_s: ClangBuiltin<"__builtin_xtensa_xt_add_s">,
  Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty], [IntrNoMem]>;

def int_xtensa_xt_abs_s: ClangBuiltin<"__builtin_xtensa_xt_abs_s">,
  Intrinsic<[llvm_float_ty], [llvm_float_ty], [IntrNoMem]>;

def int_xtensa_xt_mul_s: ClangBuiltin<"__builtin_xtensa_xt_mul_s">,
  Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty], [IntrNoMem]>;

def int_xtensa_xt_sub_s: ClangBuiltin<"__builtin_xtensa_xt_sub_s">,
  Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty], [IntrNoMem]>;

def int_xtensa_xt_trunc_s: ClangBuiltin<"__builtin_xtensa_xt_trunc_s">,
  Intrinsic<[llvm_i32_ty], [llvm_float_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, IntrNoMem]>;

def int_xtensa_xt_utrunc_s: ClangBuiltin<"__builtin_xtensa_xt_utrunc_s">,
  Intrinsic<[llvm_i32_ty], [llvm_float_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, IntrNoMem]>;

def int_xtensa_xt_float_s: ClangBuiltin<"__builtin_xtensa_xt_float_s">,
  Intrinsic<[llvm_float_ty], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, IntrNoMem]>;

def int_xtensa_xt_ufloat_s: ClangBuiltin<"__builtin_xtensa_xt_ufloat_s">,
  Intrinsic<[llvm_float_ty], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, IntrNoMem]>;


def int_xtensa_xt_addexp_s: ClangBuiltin<"__builtin_xtensa_xt_addexp_s">,
  Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty], [IntrNoMem]>;

def int_xtensa_xt_addexpm_s: ClangBuiltin<"__builtin_xtensa_xt_addexpm_s">,
  Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty], [IntrNoMem]>;

def int_xtensa_xt_ceil_s: ClangBuiltin<"__builtin_xtensa_xt_ceil_s">,
  Intrinsic<[llvm_i32_ty], [llvm_float_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, IntrNoMem]>;

def int_xtensa_xt_conjc_s: ClangBuiltin<"__builtin_xtensa_xt_conjc_s">,
  Intrinsic<[llvm_v2f32_ty], [llvm_v2f32_ty], [IntrNoMem]>;

def int_xtensa_xt_div0_s: ClangBuiltin<"__builtin_xtensa_xt_div0_s">,
  Intrinsic<[llvm_float_ty], [llvm_float_ty], [IntrNoMem]>;

def int_xtensa_xt_divn_s: ClangBuiltin<"__builtin_xtensa_xt_divn_s">,
  Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty, llvm_float_ty], [IntrNoMem]>;

def int_xtensa_xt_floor_s: ClangBuiltin<"__builtin_xtensa_xt_floor_s">,
  Intrinsic<[llvm_i32_ty], [llvm_float_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, IntrNoMem]>;

def int_xtensa_xt_lsi: ClangBuiltin<"__builtin_xtensa_xt_lsi">,
  Intrinsic<[llvm_float_ty], [llvm_ptr_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_xtensa_xt_lsip: 
  Intrinsic<[llvm_float_ty, llvm_ptr_ty], [llvm_ptr_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_xtensa_xt_lsx: ClangBuiltin<"__builtin_xtensa_xt_lsx">,
  Intrinsic<[llvm_float_ty], [llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_xt_lsxp: 
  Intrinsic<[llvm_float_ty, llvm_ptr_ty], [llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_xt_madd_s: ClangBuiltin<"__builtin_xtensa_xt_madd_s">,
  Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty, llvm_float_ty], [IntrNoMem]>;

def int_xtensa_xt_maddn_s: ClangBuiltin<"__builtin_xtensa_xt_maddn_s">,
  Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty, llvm_float_ty], [IntrNoMem]>;

def int_xtensa_xt_mkdadj_s: ClangBuiltin<"__builtin_xtensa_xt_mkdadj_s">,
  Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty], [IntrNoMem]>;

def int_xtensa_xt_mksadj_s: ClangBuiltin<"__builtin_xtensa_xt_mksadj_s">,
  Intrinsic<[llvm_float_ty], [llvm_float_ty], [IntrNoMem]>;

def int_xtensa_xt_mov_s: ClangBuiltin<"__builtin_xtensa_xt_mov_s">,
  Intrinsic<[llvm_float_ty], [llvm_float_ty], [IntrNoMem]>;

def int_xtensa_xt_moveqz_s: ClangBuiltin<"__builtin_xtensa_xt_moveqz_s">,
  Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty, llvm_i32_ty], [IntrNoMem]>;

def int_xtensa_xt_movgez_s: ClangBuiltin<"__builtin_xtensa_xt_movgez_s">,
  Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty, llvm_i32_ty], [IntrNoMem]>;

def int_xtensa_xt_movltz_s: ClangBuiltin<"__builtin_xtensa_xt_movltz_s">,
  Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty, llvm_i32_ty], [IntrNoMem]>;

def int_xtensa_xt_movnez_s: ClangBuiltin<"__builtin_xtensa_xt_movnez_s">,
  Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty, llvm_i32_ty], [IntrNoMem]>;

def int_xtensa_xt_msub_s: ClangBuiltin<"__builtin_xtensa_xt_msub_s">,
  Intrinsic<[llvm_float_ty], [llvm_float_ty, llvm_float_ty, llvm_float_ty], [IntrNoMem]>;

def int_xtensa_xt_neg_s: ClangBuiltin<"__builtin_xtensa_xt_neg_s">,
  Intrinsic<[llvm_float_ty], [llvm_float_ty], [IntrNoMem]>;

def int_xtensa_xt_nexp01_s: ClangBuiltin<"__builtin_xtensa_xt_nexp01_s">,
  Intrinsic<[llvm_float_ty], [llvm_float_ty], [IntrNoMem]>;

def int_xtensa_xt_recip0_s: ClangBuiltin<"__builtin_xtensa_xt_recip0_s">,
  Intrinsic<[llvm_float_ty], [llvm_float_ty], [IntrNoMem]>;

def int_xtensa_xt_rfr: ClangBuiltin<"__builtin_xtensa_xt_rfr">,
  Intrinsic<[llvm_i32_ty], [llvm_float_ty], [IntrNoMem]>;

def int_xtensa_xt_round_s: ClangBuiltin<"__builtin_xtensa_xt_round_s">,
  Intrinsic<[llvm_i32_ty], [llvm_float_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, IntrNoMem]>;

def int_xtensa_xt_rsqrt0_s: ClangBuiltin<"__builtin_xtensa_xt_rsqrt0_s">,
  Intrinsic<[llvm_float_ty], [llvm_float_ty], [IntrNoMem]>;

def int_xtensa_xt_rur_fcr: ClangBuiltin<"__builtin_xtensa_xt_rur_fcr">,
  Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>;

def int_xtensa_xt_rur_fsr: ClangBuiltin<"__builtin_xtensa_xt_rur_fsr">,
  Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>;

def int_xtensa_xt_sqrt0_s: ClangBuiltin<"__builtin_xtensa_xt_sqrt0_s">,
  Intrinsic<[llvm_float_ty], [llvm_float_ty], [IntrNoMem]>;

def int_xtensa_xt_ssi: ClangBuiltin<"__builtin_xtensa_xt_ssi">,
  Intrinsic<[], [llvm_float_ty, llvm_ptr_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>]>;

def int_xtensa_xt_ssip: ClangBuiltin<"__builtin_xtensa_xt_ssip">,
  Intrinsic<[llvm_ptr_ty], [llvm_float_ty, llvm_ptr_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>]>;

def int_xtensa_xt_ssx: ClangBuiltin<"__builtin_xtensa_xt_ssx">,
  Intrinsic<[], [llvm_float_ty, llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_xt_ssxp: ClangBuiltin<"__builtin_xtensa_xt_ssxp">,
  Intrinsic<[llvm_ptr_ty], [llvm_float_ty, llvm_ptr_ty, llvm_i32_ty], []>;

def int_xtensa_xt_wfr: ClangBuiltin<"__builtin_xtensa_xt_wfr">,
  Intrinsic<[llvm_float_ty], [llvm_i32_ty], [IntrNoMem]>;

def int_xtensa_xt_wur_fcr: ClangBuiltin<"__builtin_xtensa_xt_wur_fcr">,
  Intrinsic<[], [llvm_i32_ty], []>;

def int_xtensa_xt_wur_fsr: ClangBuiltin<"__builtin_xtensa_xt_wur_fsr">,
  Intrinsic<[], [llvm_i32_ty], []>;

//===----------------------------------------------------------------------===//
// HiFi3 Intrinsics
//===----------------------------------------------------------------------===//

//Extended Access
def int_xtensa_xt_l32ex: ClangBuiltin<"__builtin_xtensa_xt_l32ex">,
  Intrinsic<[llvm_i32_ty], [LLVMPointerType<llvm_i32_ty>],[]>;

def int_xtensa_xt_s32ex:
  Intrinsic<[llvm_i32_ty], [llvm_i32_ty, LLVMPointerType<llvm_i32_ty>],[]>;

def int_xtensa_xt_getex:
  Intrinsic<[llvm_i32_ty], [llvm_i32_ty],[]>;

def int_xtensa_xt_clrex: ClangBuiltin<"__builtin_xtensa_xt_clrex">,
  Intrinsic<[], [],[]>;

include "llvm/IR/IntrinsicsXtensaHIFI.td"
}
