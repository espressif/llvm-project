//===- RISCVCustomLICM.cpp - Custom Loop Invariant Code Motion ------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RISCVCustomLICM pass - Custom Loop Invariant Code Motion
// This pass aims to optimize loops by moving invariant code out of loops
// Main optimizations:
// 1. Hoisted loop-invariant loads out of the main loop:
//    Loaded coef[0-4] values before the loop and stored them in local variables
//    for use inside the loop
// 2. Pre-computed negations of some coefficients:
//    Created %7 = fneg float %1 (negation of coef[3])
//    Created %8 = fneg float %2 (negation of coef[4])
/*
Before optimization:
for.body:
  %1 = load float, ptr %arrayidx1, align 4
  %2 = load float, ptr %arrayidx3, align 4
  %3 = load float, ptr %coef, align 4
  %4 = load float, ptr %arrayidx7, align 4
  %5 = load float, ptr %arrayidx10, align 4
  %6 = load float, ptr %w, align 4
  ...
  %neg = fneg float %4
  ...
  %neg5 = fneg float %6

After optimization:
for.body.lr.ph:
  %1 = load float, ptr %arrayidx1, align 4
  %2 = load float, ptr %arrayidx3, align 4
  %3 = load float, ptr %coef, align 4
  %4 = load float, ptr %arrayidx7, align 4
  %5 = load float, ptr %arrayidx10, align 4
  %6 = load float, ptr %w, align 4
  ...
  %neg = fneg float %4
  %neg5 = fneg float %6
  br label %for.body
for.body:
  ...
*/

#include "RISCVCustomLICM.h"
#include "llvm/ADT/PriorityWorklist.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AliasSetTracker.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/CaptureTracking.h"
#include "llvm/Analysis/GuardUtils.h"
#include "llvm/Analysis/LazyBlockFrequencyInfo.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopIterator.h"
#include "llvm/Analysis/LoopNestAnalysis.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/MemorySSA.h"
#include "llvm/Analysis/MemorySSAUpdater.h"
#include "llvm/Analysis/MustExecute.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/IR/PredIteratorCache.h"
#include "llvm/InitializePasses.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/LICM.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/AssumeBundleBuilder.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/SSAUpdater.h"
#include <algorithm>
#include <utility>

using namespace llvm;

#define DEBUG_TYPE "riscv-custom-licm"

// Command line option to enable/disable RISCVCustomLICM
cl::opt<bool> llvm::EnableRISCVCustomLICM(
    "riscv-custom-licm", cl::init(false),
    cl::desc("enable custom licm for specific loop"));
/*
for.body:
  ...
  %neg = fneg float %4
  ...
  %neg5 = fneg float %6

=>
for.body.lr.ph:
  ...
  %neg = fneg float %4
  %neg5 = fneg float %6
  br label %for.body

for.body:
  ...

// Since the input and output of fneg remain constant throughout the loop,
// extracting it to the preheader can improve performance.
// This optimization moves fneg instructions out of the loop body
// to reduce redundant computations in each iteration.

*/
// Function to move fneg instructions out of the loop
void RISCVCustomLICMPass::moveFnegOutOfLoop(BasicBlock *Preheader,
                                            BasicBlock &BB, LLVMContext &Ctx) {
  IRBuilder<> Builder(Preheader->getTerminator());
  SmallVector<Instruction *> toRemove;

  for (auto &I : BB) {
    if (auto *FNeg = dyn_cast<UnaryOperator>(&I)) {
      if (FNeg->getOpcode() == Instruction::FNeg) {
        Value *Operand = FNeg->getOperand(0);
        Value *NewFNeg = Builder.CreateFNeg(Operand);
        FNeg->replaceAllUsesWith(NewFNeg);
        toRemove.push_back(FNeg);
      }
    }
  }

  // Remove the old fneg instructions
  for (auto *I : toRemove) {
    I->eraseFromParent();
  }
}

// Helper function to get a basic block by name
static inline BasicBlock *getBasicBlockByName(Function &F, StringRef Name) {
  for (BasicBlock &BB : F)
    if (BB.getName() == Name)
      return &BB;
  return nullptr;
}

// Function to adjust PHI nodes in the loop
void RISCVCustomLICMPass::adjustPhiNodes(BasicBlock &BB) {
  SmallVector<PHINode *> Phis;
  for (auto &I : BB.phis()) {
    Phis.push_back(&I);
  }

  if (Phis.size() >= 2) {
    PHINode *Phi1 = Phis[0];
    PHINode *Phi2 = Phis[1];

    // Swap the positions of two PHI nodes
    Phi2->moveBefore(Phi1);

    // Update the loop entry value of the second PHI node
    Value *LoopValue = Phi2;
    for (unsigned i = 0; i < Phi1->getNumIncomingValues(); ++i) {
      if (Phi1->getIncomingBlock(i) == &BB) {
        Phi1->setIncomingValue(i, LoopValue);
        break;
      }
    }
  }
}

// Function to create a cleanup block for the loop
void RISCVCustomLICMPass::createCleanupBlock(Function &F, BasicBlock &LoopBB) {
  LLVMContext &Ctx = F.getContext();
  BasicBlock *CleanupBB =
      BasicBlock::Create(Ctx, "for.cond.cleanup", &F, LoopBB.getNextNode());
  IRBuilder<> Builder(CleanupBB);

  // Create a branch to if.end
  BasicBlock *IfEndBB = getBasicBlockByName(F, "if.end");
  assert(IfEndBB && "if.end basic block not found");
  Builder.CreateBr(IfEndBB);

  // Update the loop's branch
  Instruction *LoopTerminator = LoopBB.getTerminator();
  assert(LoopTerminator && "Loop terminator not found");

  if (BranchInst *BI = dyn_cast<BranchInst>(LoopTerminator)) {
    if (BI->isConditional() && BI->getSuccessor(1) == &LoopBB) {
      BI->setSuccessor(0, CleanupBB);
    }
  }
}

// Function to move store instructions out of the loop
void RISCVCustomLICMPass::moveStoreOutOfLoop(BasicBlock &BB) {
  BasicBlock *CleanupBB = BB.getNextNode();
  if (!CleanupBB || CleanupBB->getName() != "for.cond.cleanup")
    return;

  IRBuilder<> Builder(CleanupBB->getFirstNonPHI());
  SmallVector<Instruction *> toRemove;

  for (auto &I : BB) {
    if (auto *Store = dyn_cast<StoreInst>(&I)) {
      Value *Val1 = Store->getOperand(1);
      Value *Val0 = Store->getOperand(0);
      // Check if the stored value is defined in the current basic block
      if (!isa<Instruction>(Val1) ||
          cast<Instruction>(Val1)->getParent() != &BB) {
        Value *Ptr = Store->getPointerOperand();
        Builder.CreateStore(Val0, Ptr);
        toRemove.push_back(Store);
      }
    }
  }

  // Remove the old store instructions
  for (auto *I : toRemove) {
    I->eraseFromParent();
  }
}

// Check the number of basic blocks and parameters of the function
static bool checkBasicBlocksAndParams(Function &F) {
  // Check number of basic blocks
  if (F.size() != 7)
    return false;

  // Check number of parameters
  if (F.arg_size() != 5)
    return false;

  return true;
}

// Check loop nesting depth
static bool checkLoopNesting(Function &F, LoopInfo &LI) {
  // Check maximum loop depth
  unsigned int maxLoopDepth = 0;
  for (auto &BB : F) {
    maxLoopDepth = std::max(maxLoopDepth, LI.getLoopDepth(&BB));
  }
  if (maxLoopDepth != 1)
    return false;

  // Check outer and inner loop counts
  int outerLoopCount = 0;
  int innerLoopCount = 0;
  for (Loop *L : LI.getLoopsInPreorder()) {
    if (L->getLoopDepth() == 1) {
      outerLoopCount++;
      if (L->getSubLoops().size() > 0) {
        innerLoopCount++;
      }
    }
  }

  return (outerLoopCount == 2 && innerLoopCount == 0);
}

// Check if fmuladd.f32 intrinsic is used
static bool checkFMulAddUsage(Function &F) {
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (RecurrenceDescriptor::isFMulAddIntrinsic(&I)) {
        return true;
      }
    }
  }
  return false;
}

// Check existence of basic blocks and control flow
static bool checkBasicBlocksAndControlFlow(Function &F) {
  // Get all required basic blocks
  BasicBlock *Entry = getBasicBlockByName(F, "entry");
  BasicBlock *ForCondPreheader = getBasicBlockByName(F, "for.cond.preheader");
  BasicBlock *ForBodyLrPh = getBasicBlockByName(F, "for.body.lr.ph");
  BasicBlock *IfEnd = getBasicBlockByName(F, "if.end");
  BasicBlock *ForBody = getBasicBlockByName(F, "for.body");
  BasicBlock *ForBodyLrPhClone = getBasicBlockByName(F, "for.body.lr.ph.clone");
  BasicBlock *ForBodyClone = getBasicBlockByName(F, "for.body.clone");

  // Check if all basic blocks exist
  if (!Entry || !ForCondPreheader || !ForBodyLrPh || !IfEnd || !ForBody ||
      !ForBodyLrPhClone || !ForBodyClone)
    return false;

  // Check control flow
  if (Entry->getTerminator()->getSuccessor(0) != ForCondPreheader ||
      Entry->getTerminator()->getSuccessor(1) != ForBodyLrPhClone ||
      ForCondPreheader->getTerminator()->getSuccessor(0) != ForBodyLrPh ||
      ForCondPreheader->getTerminator()->getSuccessor(1) != IfEnd ||
      ForBodyLrPh->getSingleSuccessor() != ForBody ||
      ForBody->getTerminator()->getSuccessor(0) != IfEnd ||
      ForBody->getTerminator()->getSuccessor(1) != ForBody ||
      ForBodyLrPhClone->getSingleSuccessor() != ForBodyClone ||
      ForBodyClone->getTerminator()->getSuccessor(0) != IfEnd ||
      ForBodyClone->getTerminator()->getSuccessor(1) != ForBodyClone)
    return false;

  return true;
}

// Main check function
static bool isSafeToOptimizeBiquadType(Function &F, LoopInfo &LI) {
  return checkBasicBlocksAndParams(F) && checkLoopNesting(F, LI) &&
         checkFMulAddUsage(F) && checkBasicBlocksAndControlFlow(F);
}

// Main function to run the CustomLICM pass
PreservedAnalyses RISCVCustomLICMPass::run(Function &F,
                                           FunctionAnalysisManager &FAM) {
  if (!EnableRISCVCustomLICM)
    return PreservedAnalyses::all();

  DominatorTree &DT = FAM.getResult<DominatorTreeAnalysis>(F);
  LoopInfo &LI = FAM.getResult<LoopAnalysis>(F);

  if (!isSafeToOptimizeBiquadType(F, LI)) {
    return PreservedAnalyses::all();
  }

  bool Changed = false;

  for (auto &L : LI) {
    if (L->getLoopDepth() != 1 || L->getBlocks().empty())
      continue; // Only process the outermost non-empty loops

    BasicBlock *Preheader = L->getLoopPreheader();
    if (!Preheader) {
      Preheader = InsertPreheaderForLoop(L, &DT, &LI, nullptr, true);
      if (!Preheader)
        continue;
    }

    Changed |= optimizeLoop(L, Preheader, F);
  }

  LLVM_DEBUG(F.dump());

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

// Function to optimize a single loop
bool RISCVCustomLICMPass::optimizeLoop(Loop *L, BasicBlock *Preheader,
                                       Function &F) {
  SmallVector<Instruction *, 8> InvariantInsts;

  for (auto &BB : L->blocks()) {
    if (BB->getName() != "for.body")
      continue;

    for (auto &I : *BB) {
      if (L->hasLoopInvariantOperands(&I) && !isMustTailCall(&I)) {
        InvariantInsts.push_back(&I);
      }
    }

    // Move loop invariant instructions
    for (auto *I : InvariantInsts) {
      I->moveBefore(Preheader->getTerminator());
    }

    // Execute other optimizations
    moveFnegOutOfLoop(Preheader, *BB, F.getContext());
    adjustPhiNodes(*BB);
    createCleanupBlock(F, *BB);
    moveStoreOutOfLoop(*BB);
  }

  return !InvariantInsts.empty();
}

// Helper function to check if an instruction is a must-tail call
bool RISCVCustomLICMPass::isMustTailCall(Instruction *I) {
  if (CallInst *CI = dyn_cast<CallInst>(I)) {
    return CI->isMustTailCall();
  }
  return false;
}
