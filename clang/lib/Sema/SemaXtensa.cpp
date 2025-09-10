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
  }
  return SemaRef.BuiltinConstantArgRange(TheCall, i, l, u);
}

} // namespace clang
