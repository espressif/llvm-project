//===- XtensaTargetTransformInfo.cpp - Xtensa specific TTI ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "XtensaTargetTransformInfo.h"

using namespace llvm;

#define DEBUG_TYPE "xtensatti"

static cl::opt<bool> DisableLowOverheadLoops(
    "disable-xtensa-hwloops", cl::Hidden, cl::init(false),
    cl::desc("Disable the generation of hardware loops"));

bool XtensaTTIImpl::isHardwareLoopProfitable(Loop *L, ScalarEvolution &SE,
                                             AssumptionCache &AC,
                                             TargetLibraryInfo *LibInfo,
                                             HardwareLoopInfo &HWLoopInfo) {
  if (DisableLowOverheadLoops)
    return false;

  if (!ST->hasLoop())
    return false;

  LLVMContext &C = L->getHeader()->getContext();
  HWLoopInfo.CounterInReg = false;
  HWLoopInfo.IsNestingLegal = false;
  HWLoopInfo.CountType = Type::getInt32Ty(C);
  HWLoopInfo.LoopDecrement = ConstantInt::get(HWLoopInfo.CountType, 1);
  return true;
}
