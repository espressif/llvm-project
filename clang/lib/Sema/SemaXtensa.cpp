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
  return false;
}

} // namespace clang
