//===--------- Xtensa.cpp - Emit LLVM Code for builtins -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code to emit Builtin calls as LLVM code.
//
//===----------------------------------------------------------------------===//

#include "CodeGenFunction.h"
#include "clang/Basic/TargetBuiltins.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsXtensa.h"

using namespace clang;
using namespace CodeGen;
using namespace llvm;

llvm::Value *
CodeGenFunction::EmitXtensaBuiltinExpr(unsigned BuiltinID, const CallExpr *E,
                                      ReturnValueSlot ReturnValue,
                                      llvm::Triple::ArchType Arch) {

  unsigned IntrinsicID;
  switch (BuiltinID) {
  case Xtensa::BI__builtin_xtensa_xt_lsxp:
    IntrinsicID = Intrinsic::xtensa_xt_lsxp;
    break;
  case Xtensa::BI__builtin_xtensa_xt_lsip:
    IntrinsicID = Intrinsic::xtensa_xt_lsip;
    break;
  default:
    llvm_unreachable("unexpected builtin ID");
  }

  llvm::Function *F = CGM.getIntrinsic(IntrinsicID);
  // 1st argument is passed by pointer
  /* float lsip(float **a, int off) =>     float p = *a
                                           ret, p' = @int.xtensa.lsip(p, off)
                                           *a = p'
  */
  auto InoutPtrTy = F->getArg(0)->getType()->getPointerTo();
  Address InoutPtrAddr = EmitPointerWithAlignment(E->getArg(0))
    .withElementType(InoutPtrTy);

  unsigned NumArgs = E->getNumArgs();
  Value *InoutVal = Builder.CreateLoad(InoutPtrAddr);
  SmallVector<Value *, 8> Args;

  Args.push_back(InoutVal);
  for (unsigned i = 1; i < NumArgs; i++)
    Args.push_back(EmitScalarExpr(E->getArg(i)));

  Value *Val = Builder.CreateCall(F, Args, "retval");
  Value *Val0 = Builder.CreateExtractValue(Val, 0);
  Value *Val1 = Builder.CreateExtractValue(Val, 1);
  // ret store
  Builder.CreateStore(Val1, InoutPtrAddr);
  return Val0;
}
