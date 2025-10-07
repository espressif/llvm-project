//===- RISCVcustomLICM.h - Function Entry/Exit Instrumentation ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RISCVCustomLICM pass - Custom Loop Invariant Code Motion
// This pass aims to optimize loops by moving invariant code out of loops
//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_RISCVCUSTOMLICM_H
#define LLVM_TRANSFORMS_UTILS_RISCVCUSTOMLICM_H

#include "llvm/Support/CommandLine.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/PassManager.h"

namespace llvm {
extern cl::opt<bool> EnableRISCVCustomLICM;
class Function;

struct RISCVCustomLICMPass : public PassInfoMixin<RISCVCustomLICMPass> {
  RISCVCustomLICMPass() {}

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  static bool isRequired() { return true; }
  // Implement logic to move fneg operations out of the loop
  void moveFnegOutOfLoop(BasicBlock *Preheader, BasicBlock &BB,
                         LLVMContext &Ctx);

  // Implement logic to adjust phi nodes
  void adjustPhiNodes(BasicBlock &BB);

  // Implement logic to create for.cond.cleanup basic block
  void createCleanupBlock(Function &F, BasicBlock &LoopBB);

  // Implement logic to move store operations after the loop ends
  void moveStoreOutOfLoop(BasicBlock &BB);

  bool optimizeLoop(Loop *L, BasicBlock *Preheader, Function &F);

  bool isMustTailCall(Instruction *I);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_RISCVCUSTOMLICM_H
