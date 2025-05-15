//===- RISCVSplitLoopByLength.h - Function Entry/Exit Instrumentation ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RISCVSplitLoopByLength pass - Split loops into two parts:
// 1. Loops with length greater than 2
// 2. Loops for all other cases
// This pass aims to optimize loops of specific lengths to improve performance
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_RISCVSPLITLOOPBYLENGTH_H
#define LLVM_TRANSFORMS_UTILS_RISCVSPLITLOOPBYLENGTH_H

#include "llvm/Analysis/IVDescriptors.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/PassManager.h"

namespace llvm {
class RecurrenceDescriptor;
extern cl::opt<bool> EnableRISCVSplitLoopByLength;
class Function;

struct RISCVSplitLoopByLengthPass : public PassInfoMixin<RISCVSplitLoopByLengthPass> {
  RISCVSplitLoopByLengthPass() {}

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  static bool isRequired() { return true; }
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_RISCVSPLITLOOPBYLENGTH_H
