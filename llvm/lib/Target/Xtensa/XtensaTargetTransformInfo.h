//===- XtensaTargetTransformInfo.h - Xtensa specific TTI --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file defines a TargetTransformInfo::Concept conforming object specific
/// to the Xtensa target machine. It uses the target's detailed information to
/// provide more precise answers to certain TTI queries, while letting the
/// target independent and default TTI implementations handle the rest.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_XTENSA_XTENSATARGETTRANSFORMINFO_H
#define LLVM_LIB_TARGET_XTENSA_XTENSATARGETTRANSFORMINFO_H

#include "XtensaSubtarget.h"
#include "XtensaTargetMachine.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/BasicTTIImpl.h"
#include "llvm/IR/Function.h"

namespace llvm {

class XtensaTTIImpl : public BasicTTIImplBase<XtensaTTIImpl> {
  using BaseT = BasicTTIImplBase<XtensaTTIImpl>;
  using TTI = TargetTransformInfo;

  friend BaseT;

  const XtensaSubtarget *ST;
  const XtensaTargetLowering *TLI;

  const XtensaSubtarget *getST() const { return ST; }
  const XtensaTargetLowering *getTLI() const { return TLI; }

public:
  explicit XtensaTTIImpl(const XtensaTargetMachine *TM, const Function &F)
      : BaseT(TM, F.getParent()->getDataLayout()), ST(TM->getSubtargetImpl(F)),
        TLI(ST->getTargetLowering()) {}

  bool isHardwareLoopProfitable(Loop *L, ScalarEvolution &SE,
                                AssumptionCache &AC, TargetLibraryInfo *LibInfo,
                                HardwareLoopInfo &HWLoopInfo);
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_XTENSA_XTENSATARGETTRANSFORMINFO_H
