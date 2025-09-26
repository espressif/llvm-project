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

struct XtensaIntrinsicInfo {
   unsigned IntrinsicID;
  unsigned Kind;
  unsigned Arg;
};

static XtensaIntrinsicInfo GetXtensaIntrinsic(unsigned BuiltinID) {
  switch (BuiltinID) {
  case Xtensa::BI__builtin_xtensa_xt_lsip:
    return {Intrinsic::xtensa_xt_lsip, 2, 0x20100};
  case Xtensa::BI__builtin_xtensa_xt_lsxp:
    return {Intrinsic::xtensa_xt_lsxp, 2, 0x20100};
#include "clang/Basic/XtensaBuiltins.inc"
   default:
     llvm_unreachable("unexpected builtin ID");
   }
}

llvm::Value *CodeGenFunction::ConvertXtensaToC(Value *val,
                                               llvm::Type *destType) {
  Value *argCast;
  llvm::Type *valType = val->getType();

  if (valType != destType) { // i32 to C short or char
    argCast = Builder.CreateTruncOrBitCast(val, destType, "cast");
    return argCast;
  } else {
    return val;
  }
}

llvm::Value *CodeGenFunction::ConvertXtensaToBc(const Expr *ArgExpr,
                                                llvm::Type *destType) {

  Value *ArgVal = EmitScalarExpr(ArgExpr);
  Value *ArgCast = ArgVal;
  llvm::Type *ArgType = ArgVal->getType();
  bool sign = ArgExpr->getType()->isSignedIntegerType();

  if (ArgType != destType) { // short,char
    if (sign)
      ArgCast = Builder.CreateSExtOrBitCast(ArgVal, destType, "cast");
    else
      ArgCast = Builder.CreateZExtOrBitCast(ArgVal, destType, "cast");
  }
  return ArgCast;
}

llvm::Value *
CodeGenFunction::EmitXtensaConversionExpr(unsigned BuiltinID, const CallExpr *E,
                                          ReturnValueSlot ReturnValue,
                                          llvm::Triple::ArchType Arch) {
  unsigned MaxElems;
  switch (BuiltinID) {
  case Xtensa::BI__builtin_xtensa_ae_int32x2:
    MaxElems = 2;
    break;
  case Xtensa::BI__builtin_xtensa_ae_int32:
    MaxElems = 1;
    break;
  default:
    llvm_unreachable("Unknown intrinsic ID");
  }

  Value *ArgVal = EmitScalarExpr(E->getArg(0));
  QualType QT = E->getArg(0)->getType();
  if (auto *VecTy = QT->getAs<VectorType>()) {
    unsigned NumEl = VecTy->getNumElements();
    llvm::Type *ElType = ConvertType(VecTy->getElementType());
    if (ElType != Int32Ty || NumEl > MaxElems) {
      CGM.Error(E->getExprLoc(), "Expected int32x1 or int32x2");
      return ArgVal;
    }
    if (NumEl == MaxElems)
      return ArgVal; // no-op
    int Mask[] = {0,0};
    Value *Result =
        Builder.CreateShuffleVector(ArgVal, ArgVal, ArrayRef(Mask, MaxElems));
    return Result;
  } else if (QT->isIntegerType()) {
    Value *Int32Val = (QT->isSignedIntegerType())
                          ? Builder.CreateSExtOrTrunc(ArgVal, Int32Ty, "cast")
                          : Builder.CreateZExtOrTrunc(ArgVal, Int32Ty, "cast");
    Value *VecOps[] = {Int32Val,Int32Val};
    Value *Result = BuildVector(ArrayRef(VecOps, MaxElems));
    return Result;
  }
  llvm_unreachable("Invalid Argument type");
}
 
llvm::Value *
CodeGenFunction::EmitXtensaBuiltinExpr(unsigned BuiltinID, const CallExpr *E,
                                      ReturnValueSlot ReturnValue,
                                      llvm::Triple::ArchType Arch) {
  switch (BuiltinID) {
  case Xtensa::BI__builtin_xtensa_ae_int32x2:
  case Xtensa::BI__builtin_xtensa_ae_int32:
    return EmitXtensaConversionExpr(BuiltinID, E, ReturnValue, Arch);
  default:
    break;
  };

  XtensaIntrinsicInfo Info = GetXtensaIntrinsic(BuiltinID);
  unsigned Intrinsic = Info.IntrinsicID;

  llvm::Function *F = CGM.getIntrinsic(Intrinsic);

  switch (Info.Kind) {
  case 0: {
    // void case
    //
    // void builtin(t1 *out /*out*/,..,t2 *inout, ..., t3 in, ..,) =>
    //  load t2 inout, ...
    //  {t1 out1, ..., t2 inout, ... ,} = func(t2 inout, ..., t3 in, ...)
    //  store (extractvalue 0) t1, ..

    SmallVector<uint8_t, 8> Out;
    SmallVector<uint8_t, 8> Inout;
    SmallVector<uint8_t, 8> In;
    SmallVector<Address, 8> OutAddr;

    unsigned Code = Info.Arg;
    unsigned CodeOut = Code & 0xff;
    unsigned CodeInout = (Code >> 8) & 0xff;
    unsigned CodeIn = (Code >> 16) & 0xff;

    for (unsigned i = 0; i < 8; ++i) {
      if (CodeOut & (1 << i))
        Out.push_back(i);
      if (CodeInout & (1 << i))
        Inout.push_back(i);
      if (CodeIn & (1 << i))
        In.push_back(i);
    }

    size_t asize = Inout.size() + In.size();
    SmallVector<Value *, 8> Args(asize, nullptr);
    assert(Args.size() == asize);

    for (uint8_t idx : In) {
      unsigned funArg = idx - Out.size();
      llvm::Type *destType = F->getArg(funArg)->getType();
      Args[funArg] = ConvertXtensaToBc(E->getArg(idx), destType);
    }

    for (unsigned i = 0; i < Out.size(); ++i) {
      unsigned idx = Out[i];
      Address AIn = EmitPointerWithAlignment(E->getArg(idx));
      Address AOut = AIn;
      OutAddr.push_back(AOut);
    }

    for (uint8_t idx : Inout) {
      uint8_t FIdx = idx - Out.size();
      Address AIn = EmitPointerWithAlignment(E->getArg(idx));
      Address AOut = AIn;
      OutAddr.push_back(AOut);
      Value *Ptr = Builder.CreateLoad(AOut);
      Args[FIdx] = Ptr;
    }

    for (auto a : Args)
      assert(a != nullptr);

    Value *Val = Builder.CreateCall(F, Args);
    Value *Val0 = nullptr;
    // check if out is a struct
    if ((OutAddr.size() > 1)) {
      for (unsigned i = 0; i < OutAddr.size(); ++i) {
        Value *Out = Builder.CreateExtractValue(Val, i);
        if (!Val0) // return the first value
        Val0 = Out;
        Address Addr = OutAddr[i];
        llvm::Type *DestType = Addr.getElementType();
        Value *OutConv = ConvertXtensaToC(Out, DestType);
        Builder.CreateStore(OutConv, Addr);
      }
    } else if (OutAddr.size() == 1) {
      Builder.CreateStore(Val, OutAddr[0]);
      Val0 = Val;
    }
    assert(Val0);
    return Val0;
  }
  case 1: {
    // t_out bultin(t1 in1, t2 in2, ...) =>
    //  t_out out1  = BcToXt( func(XtToBc(t1), XtToBc(t2), ...) )
    unsigned Code = Info.Arg;
    uint8_t CodeOut = Code & 0xff;
    uint8_t CodeInout = (Code >> 8) & 0xff;
    uint8_t CodeIn = (Code >> 16) & 0xff;

    SmallVector<uint8_t, 8> In;

    assert(CodeOut == 1 && CodeInout == 0 && "Invalid signature");
    for (unsigned i = 0; i < 8; ++i) {
      if (CodeIn & (1 << i))
        In.push_back(i);
    }
    SmallVector<Value *, 8> Args(In.size(), nullptr);
    for (uint8_t idx : In) {
      uint8_t aIdx = idx - 1;
      llvm::Type *destType = F->getArg(aIdx)->getType();
      Args[aIdx] = ConvertXtensaToBc(E->getArg(aIdx), destType);
    }
    Value *Val = Builder.CreateCall(F, Args, "retval");
    llvm::Type *ResultType = ConvertType(E->getType());
    Value *ValConv = ConvertXtensaToC(Val, ResultType);
    return ValConv;
  }
  case 2: {
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
  default:
    llvm_unreachable("unknown intrinsic kind");
  }
}
