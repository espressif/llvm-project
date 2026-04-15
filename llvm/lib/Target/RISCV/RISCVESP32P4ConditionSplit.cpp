//===- RISCVESP32P4ConditionSplit.cpp - Condition Split Pass    -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the RISCVESP32P4ConditionSplit pass.
///
/// This pass splits right-shift branches in matrix multiplication functions
/// to create separate paths for SIMD-optimizable cases (k % 8 == 0) and
/// scalar fallback cases. This enables subsequent SIMD optimization passes
/// to target the aligned path specifically.
///
/// Transformation:
///   if (final_shift <= 0) { /* right shift */ }
/// =>
///   if (final_shift <= 0) {
///     if (k % 8 == 0) { /* SIMD path */ }
///     else { /* scalar path */ }
///   }
///
//===----------------------------------------------------------------------===//

#include "RISCVESP32P4ConditionSplit.h"

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "riscv-esp32p4-condition-split"

// Command line option to enable/disable RISCVESP32P4ConditionSplit
cl::opt<bool> llvm::EnableRISCVESP32P4ConditionSplit(
    "riscv-esp32p4-condition-split", cl::init(false),
    cl::desc("Enable RISC-V ESP32-P4 condition split for matrix functions"));

namespace {

/// Check if the function has a triple-nested loop structure.
/// This is used as additional validation for matrix multiplication patterns.
static bool hasTripleNestedLoopStructure(const Function &F, LoopInfo &LI) {
  LLVM_DEBUG(dbgs() << "Analyzing loop structure for function: " << F.getName()
                    << "\n");

  unsigned MaxDepth = 0;

  // Helper lambda to recursively find maximum loop depth
  std::function<void(const Loop *)> visitLoop = [&](const Loop *L) {
    MaxDepth = std::max(MaxDepth, L->getLoopDepth());
    for (const Loop *SubLoop : L->getSubLoops())
      visitLoop(SubLoop);
  };

  // Visit all top-level loops
  for (const Loop *L : LI)
    visitLoop(L);

  LLVM_DEBUG(dbgs() << "Maximum loop depth found: " << MaxDepth << "\n");
  return MaxDepth >= 3;
}

/// Find the alloca instruction for the 'k' variable in matrix multiplication.
/// Returns nullptr if not found.
static AllocaInst *findKVariableAlloca(Function &F) {
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (auto *AI = dyn_cast<AllocaInst>(&I)) {
        if (AI->getName().contains("k.addr")) {
          LLVM_DEBUG(dbgs() << "Found k variable alloca: " << *AI << "\n");
          return AI;
        }
      }
    }
  }

  LLVM_DEBUG(dbgs() << "No k variable alloca found\n");
  return nullptr;
}

/// Find the comparison instruction for 'final_shift > 0' pattern.
/// This identifies the branch we want to split for SIMD optimization.
static ICmpInst *findFinalShiftComparison(Function &F) {
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      auto *CmpI = dyn_cast<ICmpInst>(&I);
      if (!CmpI || CmpI->getPredicate() != ICmpInst::ICMP_SGT)
        continue;

      // Check if this compares final_shift against zero
      auto *LoadI = dyn_cast<LoadInst>(CmpI->getOperand(0));
      auto *ZeroConst = dyn_cast<ConstantInt>(CmpI->getOperand(1));

      if (!LoadI || !ZeroConst || !ZeroConst->isZero())
        continue;

      auto *AI = dyn_cast<AllocaInst>(LoadI->getPointerOperand());
      if (AI && AI->getName().contains("final_shift")) {
        LLVM_DEBUG(dbgs() << "Found final_shift comparison: " << *CmpI << "\n");
        return CmpI;
      }
    }
  }

  LLVM_DEBUG(dbgs() << "No final_shift comparison found\n");
  return nullptr;
}

/// Find the branch instruction that uses the given comparison.
static BranchInst *findConditionalBranch(ICmpInst *CmpInst) {
  for (User *U : CmpInst->users()) {
    if (auto *BI = dyn_cast<BranchInst>(U)) {
      if (BI->isConditional()) {
        LLVM_DEBUG(dbgs() << "Found conditional branch: " << *BI << "\n");
        return BI;
      }
    }
  }

  LLVM_DEBUG(dbgs() << "No conditional branch found for comparison\n");
  return nullptr;
}

/// Create the k % 8 == 0 alignment check in the given basic block.
static Value *createAlignmentCheck(IRBuilder<> &Builder, AllocaInst *KAddr) {
  // Load k value
  LoadInst *KLoad = Builder.CreateLoad(Builder.getInt32Ty(), KAddr, "k.val");

  // Calculate k % 8
  Value *EightConst = ConstantInt::get(Builder.getInt32Ty(), 8);
  Value *RemainderVal = Builder.CreateSRem(KLoad, EightConst, "k.rem8");

  // Check k % 8 == 0
  Value *ZeroConst = ConstantInt::get(Builder.getInt32Ty(), 0);
  Value *IsAligned =
      Builder.CreateICmpEQ(RemainderVal, ZeroConst, "k.is_aligned");

  LLVM_DEBUG(dbgs() << "Created alignment check: k % 8 == 0\n");
  return IsAligned;
}

/// Clone instructions from source block to target block, excluding terminators.
/// Updates the provided value map for use relationship fixing.
static void cloneInstructionsToBlock(BasicBlock *SourceBB, BasicBlock *TargetBB,
                                     ValueToValueMapTy &VMap) {
  IRBuilder<> Builder(TargetBB);

  for (Instruction &I : *SourceBB) {
    if (I.isTerminator())
      continue;

    Instruction *ClonedInst = I.clone();
    Builder.Insert(ClonedInst);
    VMap[&I] = ClonedInst;
  }
}

/// Fix operand references in the cloned instructions using the value map.
static void fixClonedInstructionOperands(BasicBlock *BB,
                                         const ValueToValueMapTy &VMap) {
  for (Instruction &I : *BB) {
    for (unsigned Idx = 0, E = I.getNumOperands(); Idx != E; ++Idx) {
      Value *Op = I.getOperand(Idx);
      auto It = VMap.find(Op);
      if (It != VMap.end())
        I.setOperand(Idx, It->second);
    }
  }
}

/// Update PHI nodes in the successor block to use the new predecessor.
static void updateSuccessorPhiNodes(BasicBlock *SuccessorBB,
                                    BasicBlock *OldPred, BasicBlock *NewPred) {
  for (Instruction &I : *SuccessorBB) {
    auto *PHI = dyn_cast<PHINode>(&I);
    if (!PHI)
      break; // PHI nodes are always at the beginning

    int Idx = PHI->getBasicBlockIndex(OldPred);
    if (Idx >= 0) {
      PHI->setIncomingBlock(Idx, NewPred);
      LLVM_DEBUG(dbgs() << "Updated PHI node predecessor from "
                        << OldPred->getName() << " to " << NewPred->getName()
                        << "\n");
    }
  }
}

/// Perform the condition splitting transformation.
/// Splits the false branch of final_shift comparison into SIMD and scalar
/// paths.
static bool performConditionSplit(Function &F, ICmpInst *FinalShiftCmp,
                                  AllocaInst *KAddr) {
  // Find the conditional branch that uses the comparison
  BranchInst *Branch = findConditionalBranch(FinalShiftCmp);
  if (!Branch)
    return false;

  BasicBlock *OriginalFalseBB = Branch->getSuccessor(1); // Right shift path
  LLVM_DEBUG(dbgs() << "Splitting false branch: " << OriginalFalseBB->getName()
                    << "\n");

  // Create new basic blocks for the transformation
  LLVMContext &Ctx = F.getContext();
  BasicBlock *CondCheckBB = BasicBlock::Create(Ctx, "k.align.check", &F);
  BasicBlock *SIMDPathBB = BasicBlock::Create(Ctx, "simd.path", &F);
  BasicBlock *ScalarPathBB = BasicBlock::Create(Ctx, "scalar.path", &F);
  BasicBlock *MergeBB = BasicBlock::Create(Ctx, "split.merge", &F);

  // Redirect original branch to new condition check
  Branch->setSuccessor(1, CondCheckBB);

  // Create k % 8 == 0 check
  IRBuilder<> CondBuilder(CondCheckBB);
  Value *IsAligned = createAlignmentCheck(CondBuilder, KAddr);
  CondBuilder.CreateCondBr(IsAligned, SIMDPathBB, ScalarPathBB);

  // Clone original block content to both paths
  ValueToValueMapTy SIMDMap, ScalarMap;
  cloneInstructionsToBlock(OriginalFalseBB, SIMDPathBB, SIMDMap);
  cloneInstructionsToBlock(OriginalFalseBB, ScalarPathBB, ScalarMap);

  // Fix operand references in cloned instructions
  fixClonedInstructionOperands(SIMDPathBB, SIMDMap);
  fixClonedInstructionOperands(ScalarPathBB, ScalarMap);

  // Add branches to merge block
  IRBuilder<> SIMDBuilder(SIMDPathBB);
  IRBuilder<> ScalarBuilder(ScalarPathBB);
  SIMDBuilder.CreateBr(MergeBB);
  ScalarBuilder.CreateBr(MergeBB);

  // Handle original successor
  IRBuilder<> MergeBuilder(MergeBB);
  BasicBlock *OriginalSuccessor = nullptr;

  if (auto *Term = OriginalFalseBB->getTerminator()) {
    if (auto *Br = dyn_cast<BranchInst>(Term)) {
      if (!Br->isConditional())
        OriginalSuccessor = Br->getSuccessor(0);
    }
  }

  if (OriginalSuccessor) {
    MergeBuilder.CreateBr(OriginalSuccessor);
    updateSuccessorPhiNodes(OriginalSuccessor, OriginalFalseBB, MergeBB);
  } else {
    MergeBuilder.CreateUnreachable();
  }

  // Clean up: remove the original block
  OriginalFalseBB->eraseFromParent();

  LLVM_DEBUG(
      dbgs() << "Successfully performed condition split transformation\n");
  return true;
}

} // anonymous namespace

namespace {

/// Check if the function matches the expected matrix multiplication pattern.
/// This validates that the function has the necessary components for
/// transformation.
static bool isMatrixMultiplicationCandidate(Function &F, LoopInfo &LI,
                                            AllocaInst **KAddr,
                                            ICmpInst **FinalShiftCmp) {
  // Find k variable alloca
  *KAddr = findKVariableAlloca(F);
  if (!*KAddr) {
    LLVM_DEBUG(
        dbgs()
        << "Cannot find k variable - not a matrix multiplication function\n");
    return false;
  }

  // Find final_shift comparison
  *FinalShiftCmp = findFinalShiftComparison(F);
  if (!*FinalShiftCmp) {
    LLVM_DEBUG(
        dbgs()
        << "Cannot find final_shift comparison - not the target pattern\n");
    return false;
  }

  // Check loop structure (optional validation)
  bool HasTripleNestedLoops = hasTripleNestedLoopStructure(F, LI);
  LLVM_DEBUG(dbgs() << "Triple nested loops check: "
                    << (HasTripleNestedLoops ? "PASS" : "FAIL") << "\n");

  if (!HasTripleNestedLoops) {
    LLVM_DEBUG(dbgs() << "Warning: Loop structure validation failed, "
                      << "but proceeding based on pattern match\n");
  }

  return true;
}

} // anonymous namespace

PreservedAnalyses
RISCVESP32P4ConditionSplitPass::run(Function &F, FunctionAnalysisManager &AM) {
  LLVM_DEBUG(dbgs() << "Running RISCVESP32P4ConditionSplitPass on function: "
                    << F.getName() << "\n");

  // Early exit if pass is disabled
  if (!EnableRISCVESP32P4ConditionSplit) {
    LLVM_DEBUG(dbgs() << "Pass is disabled via command line option\n");
    return PreservedAnalyses::all();
  }

  // Get required analysis
  LoopInfo &LI = AM.getResult<LoopAnalysis>(F);

  // Validate function matches target pattern
  AllocaInst *KAddr = nullptr;
  ICmpInst *FinalShiftCmp = nullptr;

  if (!isMatrixMultiplicationCandidate(F, LI, &KAddr, &FinalShiftCmp)) {
    LLVM_DEBUG(
        dbgs() << "Function does not match matrix multiplication pattern\n");
    return PreservedAnalyses::all();
  }

  // Perform the transformation
  bool Changed = performConditionSplit(F, FinalShiftCmp, KAddr);

  if (!Changed) {
    LLVM_DEBUG(dbgs() << "Failed to apply condition split transformation\n");
    return PreservedAnalyses::all();
  }

  LLVM_DEBUG(dbgs() << "Successfully applied condition split transformation to "
                    << F.getName() << "\n");

  // Return preserved analyses - we preserve LoopInfo as we don't modify loop
  // structure
  PreservedAnalyses PA;
  PA.preserve<LoopAnalysis>();
  return PA;
}
