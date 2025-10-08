//===- RISCVLoopUnrollAndRemainder.h - Loop Unrolling and Remainder Handling
//------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RISCVLoopUnrollAndRemainder pass
//
// This pass performs loop unrolling and handles the remainder iterations.
// It aims to improve performance by:
// 1. Unrolling loops to reduce loop overhead and enable further optimizations
// 2. Generating efficient code for handling any remaining iterations
//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_RISCVLOOPUNROLLANDREMAINDER_H
#define LLVM_TRANSFORMS_UTILS_RISCVLOOPUNROLLANDREMAINDER_H

#include "llvm/Analysis/IVDescriptors.h"
#include "llvm/IR/PassManager.h"

namespace llvm {
class RecurrenceDescriptor;
extern cl::opt<bool> EnableRISCVLoopUnrollAndRemainder;
class Function;

struct RISCVLoopUnrollAndRemainderPass
    : public PassInfoMixin<RISCVLoopUnrollAndRemainderPass> {
  RISCVLoopUnrollAndRemainderPass() {}

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  static bool isRequired() { return true; }
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_RISCVLOOPUNROLLANDREMAINDER_H
