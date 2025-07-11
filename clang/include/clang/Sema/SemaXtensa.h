//===----- SemaXtensa.h ----- Xtensa target-specific routines --*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file declares semantic analysis functions specific to Xtensa.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SEMA_SEMAXTENSA_H
#define LLVM_CLANG_SEMA_SEMAXTENSA_H

#include "clang/AST/Expr.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Sema/Attr.h"
#include "clang/Sema/SemaBase.h"

namespace clang {
class SemaXtensa : public SemaBase {
public:
  SemaXtensa(Sema &S);

  bool CheckXtensaBuiltinFunctionCall(const TargetInfo &TI, unsigned BuiltinID,
                                     CallExpr *TheCall);

  bool SemaBuiltinXtensaConversion(unsigned BuiltinID, CallExpr *TheCall);

  void handleXtensaShortCall(Decl *D, const ParsedAttr &AL);
};
} // namespace clang

#endif // LLVM_CLANG_SEMA_SEMAXTENSA_H
