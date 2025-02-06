//===-- RISCVSplitLoopByLength.cpp - Loop splitting optimization --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass splits loops into two parts: one for length > 2 and another for
/// length <= 2. It's designed to prepare for the esp.lp.setup instruction.
///
/// Main steps of this pass:
/// 1. Identify the function type
/// 2. Clone the original loop
/// 3. Insert an if-else structure to choose between the original and cloned
/// loop
/// 4. Update phi nodes and branch instructions accordingly
//
//===----------------------------------------------------------------------===//

#include "RISCVSplitLoopByLength.h"
#include "RISCVESP32P4OptUtils.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Plugins/PassPlugin.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Utils/LoopUtils.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-split-loop-by-length"

// Error codes for ESP DSP library
static constexpr int ESP_ERR_DSP_BASE = 0x70000;
static constexpr int ESP_ERR_DSP_PARAM_OUTOFRANGE = ESP_ERR_DSP_BASE + 3;

namespace {

/// Constants for better code readability
static constexpr int LoopLengthThreshold = 2;
static constexpr unsigned ExpectedDotprodBBCount = 3;
static constexpr unsigned ExpectedAddcBBCount = 4;
static constexpr unsigned ExpectedBiquadBBCount = 4;
static constexpr unsigned ExpectedFirBBCount = 10;
static constexpr unsigned ExpectedFloatPhiCountBiquad = 2;
static constexpr unsigned ExpectedI32PhiCountBiquad = 1;

/// Split strategy enumeration
enum class SplitStrategy { DOTPROD, ADDC, BIQUAD_FIR, UNKNOWN };

} // end anonymous namespace

// Command line option to enable/disable RISCVSplitLoopByLength optimization
cl::opt<bool> llvm::EnableRISCVSplitLoopByLength(
    "riscv-split-loop-by-length", cl::init(false),
    cl::desc("Enable loop splitting optimization"));

/// Find the length parameter of the function by looking for ICmp users
/// \param F The function to analyze
/// \returns The argument used as length parameter, or nullptr if not found
static Value *findLengthParameter(Function *F) {
  for (auto &Arg : F->args()) {
    for (const User *U : Arg.users()) {
      if (const ICmpInst *ICmp = dyn_cast<ICmpInst>(U)) {
        if (ICmp->getPredicate() == ICmpInst::ICMP_EQ &&
            ICmp->getOperand(1) == &Arg) {
          return &Arg;
        }
      }
    }
  }
  llvm_unreachable("Length parameter not found");
}

/// Move GetElementPtr instructions from source to target basic block
/// \param SourceBB The source basic block to move instructions from
/// \param TargetBB The target basic block to move instructions to
static void moveGetElementPtrInstructions(BasicBlock &SourceBB,
                                          BasicBlock *TargetBB) {
  for (auto I = SourceBB.begin(); I != SourceBB.end(); ++I) {
    if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(I)) {
      GEP->moveBefore(TargetBB->getTerminator());
    }
  }
}

/// Insert conditional branch structure with cloned preheader for BIQUAD/FIR
/// functions
/// \param F The function to modify
/// \param ClonedPreheaderBB The cloned preheader basic block
/// \param NewBlocks Vector of newly created basic blocks
static void insertConditionalBranchWithClonedPreheader(
    Function *F, BasicBlock *ClonedPreheaderBB,
    SmallVector<BasicBlock *, 8> &NewBlocks) {

  // Rename entry to for.cond.preheader
  BasicBlock &EntryBB = F->getEntryBlock();
  EntryBB.setName("for.cond.preheader");

  // Rename for.cond.cleanup to if.end
  for (BasicBlock &BB : *F) {
    if (BB.getName() == "for.cond.cleanup")
      BB.setName("if.end");
  }

  // Create new entry basic block
  LLVMContext &Context = F->getContext();
  IRBuilder<> Builder(Context);
  BasicBlock *NewEntryBB =
      BasicBlock::Create(Context, "entry", F, &F->getEntryBlock());

  Builder.SetInsertPoint(NewEntryBB);

  // Find icmp sgt instruction
  ICmpInst *LengthComparison = nullptr;
  for (Instruction &I : EntryBB) {
    if (ICmpInst *CI = dyn_cast<ICmpInst>(&I)) {
      if (CI->getPredicate() == ICmpInst::ICMP_SGT) {
        LengthComparison = CI;
        break;
      }
    }
  }

  assert(LengthComparison && "Length comparison instruction not found");

  // Create new comparison and conditional branch
  Value *Length = LengthComparison->getOperand(0);
  Value *LengthGreaterThanThreshold = Builder.CreateICmpSGT(
      Length, ConstantInt::get(Length->getType(), LoopLengthThreshold));
  Builder.CreateCondBr(LengthGreaterThanThreshold, &EntryBB, ClonedPreheaderBB);
  moveGetElementPtrInstructions(EntryBB, NewEntryBB);
}

/// Insert conditional branch structure for DOTPROD functions
/// \param F The function to modify
/// \param NewBlocks Vector of newly created basic blocks
static void
insertStandardConditionalBranch(Function *F,
                                SmallVector<BasicBlock *, 8> &NewBlocks) {

  BasicBlock &EntryBB = F->getEntryBlock();
  EntryBB.setName("for.cond.preheader");

  BasicBlock *ForBodyBB = getBasicBlockByName(*F, "for.body");
  for (BasicBlock &BB : *F) {
    if (BB.getName() == "for.cond.cleanup")
      BB.setName("if.end");
  }
  assert(ForBodyBB && "ForBody basic block must exist");

  BasicBlock *ClonedForBodyBB = nullptr;
  Instruction *FMulAddInstruction = nullptr;
  for (BasicBlock *BB : NewBlocks) {
    if (BB->getName() == "for.body.clone") {
      ClonedForBodyBB = BB;
      for (Instruction &I : *BB) {
        if (RecurrenceDescriptor::isFMulAddIntrinsic(&I)) {
          FMulAddInstruction = &I;
          break;
        }
      }
    }
  }
  assert(ClonedForBodyBB && "Cloned ForBody basic block must exist");
  assert(FMulAddInstruction && "FMulAdd instruction must exist");

  LLVMContext &Context = F->getContext();
  IRBuilder<> Builder(Context);
  BasicBlock *NewEntryBB =
      BasicBlock::Create(Context, "entry", F, &F->getEntryBlock());

  Builder.SetInsertPoint(NewEntryBB);
  Value *Length = findLengthParameter(F);
  Value *LengthGreaterThanThreshold = Builder.CreateICmpSGT(
      Length, ConstantInt::get(Length->getType(), LoopLengthThreshold));
  Builder.CreateCondBr(LengthGreaterThanThreshold, ForBodyBB, &EntryBB);

  // Update phi nodes and successors
  for (BasicBlock &BB : *F) {
    if (BB.getName() == "if.end") {
      if (PHINode *IfEndEntryPhiNode = dyn_cast<PHINode>(&BB.front())) {
        IfEndEntryPhiNode->addIncoming(FMulAddInstruction, ClonedForBodyBB);
      }
    }
    if (BB.getName() == "for.cond.preheader") {
      BB.getTerminator()->setSuccessor(0, ClonedForBodyBB);
    }
  }

  if (PHINode *ForBodyEntryPhiNode = dyn_cast<PHINode>(&ForBodyBB->front())) {
    ForBodyEntryPhiNode->setIncomingBlock(1, NewEntryBB);
    if (PHINode *ForBodyEntryPhiNode2 = dyn_cast<PHINode>(
            ForBodyEntryPhiNode->getNextNode())) {
      ForBodyEntryPhiNode2->setIncomingBlock(1, NewEntryBB);
    }
  }
  moveGetElementPtrInstructions(EntryBB, NewEntryBB);
}

/// Insert conditional branch structure for ADDC-like functions
/// \param F The function to modify
/// \param NewBlocks Vector of newly created basic blocks
static void
insertAddcConditionalBranch(Function *F,
                            SmallVector<BasicBlock *, 8> &NewBlocks) {

  LLVMContext &Context = F->getContext();
  BasicBlock &EntryBB = F->getEntryBlock();

  BasicBlock *OriginalForBodyBB = getBasicBlockByName(*F, "for.body");
  BasicBlock *ForConditionPreheaderBB =
      getBasicBlockByName(*F, "for.cond.preheader");
  BasicBlock *ClonedForBodyBB = getBasicBlockByName(*F, "for.body.clone");
  BasicBlock *FunctionReturnBB = getBasicBlockByName(*F, "return");

  assert(OriginalForBodyBB && ForConditionPreheaderBB && ClonedForBodyBB &&
         FunctionReturnBB && "Required basic blocks not found");

  BasicBlock *IfEndBB =
      BasicBlock::Create(Context, "if.end", F, OriginalForBodyBB);

  if (BranchInst *EntryBranch = dyn_cast<BranchInst>(EntryBB.getTerminator())) {
    EntryBranch->setSuccessor(1, IfEndBB);
  }

  IRBuilder<> Builder(IfEndBB);
  Value *Length = findLengthParameter(F);
  Value *LengthGreaterThanThreshold = Builder.CreateICmpSGT(
      Length, ConstantInt::get(Length->getType(), LoopLengthThreshold),
      "length.gt.threshold");
  Builder.CreateCondBr(LengthGreaterThanThreshold, OriginalForBodyBB,
                       ForConditionPreheaderBB);

  // Update cloned for.body's terminator
  if (BranchInst *ClonedBranch =
          dyn_cast<BranchInst>(ClonedForBodyBB->getTerminator())) {
    if (ClonedBranch->isConditional() &&
        ClonedBranch->getSuccessor(1) == OriginalForBodyBB) {
      ClonedBranch->setSuccessor(1, ForConditionPreheaderBB);
    }
  }

  // Update ForConditionPreheaderBB's terminator
  if (BranchInst *PreheaderBranch =
          dyn_cast<BranchInst>(ForConditionPreheaderBB->getTerminator())) {
    if (PreheaderBranch->isConditional() &&
        PreheaderBranch->getSuccessor(0) == OriginalForBodyBB) {
      PreheaderBranch->setSuccessor(0, ClonedForBodyBB);
    }
  }

  // Update PHI nodes
  for (PHINode &Phi : ForConditionPreheaderBB->phis()) {
    if (Phi.getIncomingBlock(0) == &EntryBB) {
      Phi.setIncomingBlock(0, IfEndBB);
    }
    for (unsigned i = 0; i < Phi.getNumIncomingValues(); ++i) {
      if (Phi.getIncomingBlock(i) == OriginalForBodyBB) {
        Phi.setIncomingBlock(i, ClonedForBodyBB);
      }
    }
  }

  // Update return instruction
  if (ReturnInst *ReturnInstruction =
          dyn_cast<ReturnInst>(FunctionReturnBB->getTerminator())) {
    if (PHINode *ReturnPhi =
            dyn_cast<PHINode>(ReturnInstruction->getReturnValue())) {
      ReturnPhi->addIncoming(ConstantInt::get(ReturnPhi->getType(), 0),
                             ClonedForBodyBB);
    } else {
      PHINode *NewReturnPhi =
          PHINode::Create(ReturnInstruction->getReturnValue()->getType(), 3,
                          "retval.0", ReturnInstruction);
      NewReturnPhi->addIncoming(
          ConstantInt::get(ReturnInstruction->getReturnValue()->getType(),
                           ESP_ERR_DSP_PARAM_OUTOFRANGE),
          &EntryBB);
      NewReturnPhi->addIncoming(
          ConstantInt::get(ReturnInstruction->getReturnValue()->getType(), 0),
          ForConditionPreheaderBB);
      NewReturnPhi->addIncoming(ConstantInt::get(NewReturnPhi->getType(), 0),
                                ClonedForBodyBB);
      ReturnInstruction->setOperand(0, NewReturnPhi);
    }
  }

  // Update OriginalForBodyBB's PHI nodes
  for (PHINode &Phi : OriginalForBodyBB->phis()) {
    if (Phi.getNumIncomingValues() > 1) {
      Phi.setIncomingBlock(1, IfEndBB);
    }
  }

  // Reorder basic blocks
  ClonedForBodyBB->moveBefore(FunctionReturnBB);
  IfEndBB->moveBefore(ForConditionPreheaderBB);

  LLVM_DEBUG(F->dump());
}

/// Check if the function is ADDC-like based on structure analysis
/// \param F The function to analyze
/// \returns true if the function matches ADDC pattern
static bool isAddcLikeFunction(Function *F) {
  if (F->size() != ExpectedAddcBBCount)
    return false;

  BasicBlock *EntryBB = getBasicBlockByName(*F, "entry");
  BasicBlock *ForCondPreheaderBB =
      getBasicBlockByName(*F, "for.cond.preheader");
  BasicBlock *ForBodyBB = getBasicBlockByName(*F, "for.body");
  BasicBlock *ReturnBB = getBasicBlockByName(*F, "return");

  if (!ForBodyBB || !ForCondPreheaderBB || !EntryBB || !ReturnBB)
    return false;

  // Check successors
  if (EntryBB->getTerminator()->getSuccessor(0) != ReturnBB ||
      EntryBB->getTerminator()->getSuccessor(1) != ForCondPreheaderBB)
    return false;

  if (ForCondPreheaderBB->getTerminator()->getSuccessor(0) != ForBodyBB ||
      ForCondPreheaderBB->getTerminator()->getSuccessor(1) != ReturnBB)
    return false;

  if (ForBodyBB->getTerminator()->getSuccessor(0) != ReturnBB ||
      ForBodyBB->getTerminator()->getSuccessor(1) != ForBodyBB)
    return false;

  return true;
}

/// Check if the function is DOTPROD-like based on structure analysis
/// \param F The function to analyze
/// \returns true if the function matches DOTPROD pattern
static bool isDotProdLikeFunction(Function *F) {
  if (F->size() != ExpectedDotprodBBCount)
    return false;

  BasicBlock *EntryBB = getBasicBlockByName(*F, "entry");
  BasicBlock *ForCondCleanupBB = getBasicBlockByName(*F, "for.cond.cleanup");
  BasicBlock *ForBodyBB = getBasicBlockByName(*F, "for.body");

  if (!ForBodyBB || !ForCondCleanupBB || !EntryBB)
    return false;

  // Check successors
  if (EntryBB->getTerminator()->getNumSuccessors() != 2 ||
      EntryBB->getTerminator()->getSuccessor(0) != ForBodyBB ||
      EntryBB->getTerminator()->getSuccessor(1) != ForCondCleanupBB)
    return false;

  if (ForBodyBB->getTerminator()->getNumSuccessors() != 2 ||
      ForBodyBB->getTerminator()->getSuccessor(0) != ForCondCleanupBB ||
      ForBodyBB->getTerminator()->getSuccessor(1) != ForBodyBB)
    return false;

  // Check for float PHI node in ForBodyBB
  for (PHINode &Phi : ForBodyBB->phis()) {
    if (Phi.getType()->isFloatTy())
      return true;
  }

  return false;
}

/// Check if the function should use ADDC strategy
/// \param F The function to analyze
/// \returns true if ADDC strategy should be used
static bool shouldUseAddcStrategy(Function *F) {
  if (F->empty() || F->arg_empty() || !isAddcLikeFunction(F))
    return false;

  BasicBlock &EntryBB = F->getEntryBlock();
  if (!isa<BranchInst>(EntryBB.getTerminator()) ||
      !cast<BranchInst>(EntryBB.getTerminator())->isConditional()) {
    return false;
  }

  BasicBlock *ReturnBB = nullptr;
  for (BasicBlock &BB : *F) {
    if (isa<ReturnInst>(BB.getTerminator())) {
      ReturnBB = &BB;
      break;
    }
  }
  if (!ReturnBB || !isa<PHINode>(ReturnBB->front())) {
    return false;
  }

  return true;
}

/// Check if the function is Biquad based on structure analysis
/// \param F The function to analyze
/// \returns true if the function matches Biquad pattern
static bool isBiquadFunction(Function *F) {
  if (F->size() != ExpectedBiquadBBCount)
    return false;

  BasicBlock *ForBodyBB = getBasicBlockByName(*F, "for.body");
  BasicBlock *ForBodyLrPhBB = getBasicBlockByName(*F, "for.body.lr.ph");
  BasicBlock *ForCondCleanupBB = getBasicBlockByName(*F, "for.cond.cleanup");
  BasicBlock *EntryBB = getBasicBlockByName(*F, "entry");

  if (!ForBodyBB || !ForBodyLrPhBB || !ForCondCleanupBB || !EntryBB)
    return false;

  // Check successors
  if (EntryBB->getTerminator()->getNumSuccessors() != 2 ||
      EntryBB->getTerminator()->getSuccessor(0) != ForBodyLrPhBB ||
      EntryBB->getTerminator()->getSuccessor(1) != ForCondCleanupBB)
    return false;

  if (ForBodyLrPhBB->getTerminator()->getNumSuccessors() != 1 ||
      ForBodyLrPhBB->getTerminator()->getSuccessor(0) != ForBodyBB)
    return false;

  if (ForBodyBB->getTerminator()->getNumSuccessors() != 2 ||
      ForBodyBB->getTerminator()->getSuccessor(0) != ForCondCleanupBB ||
      ForBodyBB->getTerminator()->getSuccessor(1) != ForBodyBB)
    return false;

  // Check PHI nodes in ForBodyBB
  int FloatPhiCount = 0;
  int I32PhiCount = 0;
  for (PHINode &Phi : ForBodyBB->phis()) {
    if (Phi.getType()->isFloatTy())
      FloatPhiCount++;
    else if (Phi.getType()->isIntegerTy(32))
      I32PhiCount++;
  }

  return (FloatPhiCount == ExpectedFloatPhiCountBiquad &&
          I32PhiCount == ExpectedI32PhiCountBiquad);
}

/// Check if the function is FIR based on structure analysis
/// \param F The function to analyze
/// \returns true if the function matches FIR pattern
static bool isFIRFunction(Function *F) {
  if (F->size() != ExpectedFirBBCount)
    return false;

  BasicBlock *ForBodyBB = getBasicBlockByName(*F, "for.body");
  BasicBlock *ForBodyLrPhBB = getBasicBlockByName(*F, "for.body.lr.ph");
  BasicBlock *ForCondCleanupBB = getBasicBlockByName(*F, "for.cond.cleanup");
  BasicBlock *EntryBB = getBasicBlockByName(*F, "entry");

  if (!ForBodyBB || !ForBodyLrPhBB || !ForCondCleanupBB || !EntryBB)
    return false;

  // Check successors
  if (EntryBB->getTerminator()->getNumSuccessors() != 2 ||
      EntryBB->getTerminator()->getSuccessor(0) != ForBodyLrPhBB ||
      EntryBB->getTerminator()->getSuccessor(1) != ForCondCleanupBB)
    return false;

  if (ForBodyLrPhBB->getTerminator()->getNumSuccessors() != 1 ||
      ForBodyLrPhBB->getTerminator()->getSuccessor(0) != ForBodyBB)
    return false;

  return true;
}

/// Check if the function should use BIQUAD/FIR strategy
/// \param F The function to analyze
/// \returns true if BIQUAD/FIR strategy should be used
static bool shouldUseBiquadFirStrategy(Function *F) {
  if (F->empty() || F->arg_empty() ||
      (!isBiquadFunction(F) && !isFIRFunction(F)))
    return false;

  BasicBlock &EntryBB = F->getEntryBlock();
  BranchInst *EntryBranch = dyn_cast<BranchInst>(EntryBB.getTerminator());
  if (!EntryBranch || !EntryBranch->isConditional())
    return false;

  // Check for icmp sgt i32 %len, 0
  if (EntryBB.empty())
    return false;

  ICmpInst *LengthComparison = dyn_cast<ICmpInst>(&EntryBB.front());
  if (!LengthComparison ||
      LengthComparison->getPredicate() != ICmpInst::ICMP_SGT)
    return false;

  Value *ComparisonOperand0 = LengthComparison->getOperand(0);
  Value *ComparisonOperand1 = LengthComparison->getOperand(1);
  if (!isa<Argument>(ComparisonOperand0) ||
      !isa<ConstantInt>(ComparisonOperand1))
    return false;

  return cast<ConstantInt>(ComparisonOperand1)->isZero();
}

/// Check if the function should use DOTPROD strategy
/// \param F The function to analyze
/// \returns true if DOTPROD strategy should be used
static bool shouldUseDotProdStrategy(Function *F) {
  if (F->empty() || F->arg_size() < 3 || !isDotProdLikeFunction(F))
    return false;

  if (F->size() != 3 && F->size() != 4)
    return false;

  bool HasEntryBB = getBasicBlockByName(*F, "entry") != nullptr;
  bool HasForBodyBB = getBasicBlockByName(*F, "for.body") != nullptr;
  bool HasForCondPreheaderBB =
      getBasicBlockByName(*F, "for.cond.cleanup") != nullptr;

  if (!HasForBodyBB || !HasForCondPreheaderBB || !HasEntryBB)
    return false;

  // Check for fmuladd instruction
  bool HasFMulAddInstruction = false;
  for (BasicBlock &BB : *F) {
    for (Instruction &I : BB) {
      if (RecurrenceDescriptor::isFMulAddIntrinsic(&I)) {
        HasFMulAddInstruction = true;
        // Check operands of fmuladd
        if (I.getOperand(0)->getType()->isFloatTy() &&
            I.getOperand(1)->getType()->isFloatTy()) {
          if (!isa<LoadInst>(I.getOperand(0)) ||
              !isa<LoadInst>(I.getOperand(1))) {
            return false;
          }
        }
        break;
      }
    }
    if (HasFMulAddInstruction)
      break;
  }
  if (!HasFMulAddInstruction)
    return false;

  // Check for icmp sgt i32 %len, 0
  BasicBlock &EntryBB = F->getEntryBlock();
  Instruction &FirstInstruction = EntryBB.front();

  ICmpInst *LengthComparison = dyn_cast<ICmpInst>(&FirstInstruction);
  if (!LengthComparison)
    return false;

  if (LengthComparison->getPredicate() != ICmpInst::ICMP_SGT)
    return false;

  ConstantInt *ComparisonConstant =
      dyn_cast<ConstantInt>(LengthComparison->getOperand(1));
  if (!ComparisonConstant)
    return false;

  return ComparisonConstant->isZero();
}

/// Determine the appropriate split strategy for the function
/// \param F The function to analyze
/// \returns The determined split strategy
static SplitStrategy determineSplitStrategy(Function *F) {
  if (shouldUseDotProdStrategy(F)) {
    return SplitStrategy::DOTPROD;
  }
  if (shouldUseAddcStrategy(F)) {
    return SplitStrategy::ADDC;
  }
  if (shouldUseBiquadFirStrategy(F)) {
    return SplitStrategy::BIQUAD_FIR;
  }
  return SplitStrategy::UNKNOWN;
}

/// Clone loop structure without modifying control flow
/// \param OriginalLoop The loop to clone
/// \param F The function containing the loop
/// \param LI Loop info for updating loop structure
/// \param ValueMap Mapping from original to cloned values
/// \param NewBlocks Vector to store newly created blocks
/// \returns The newly created loop
static Loop *cloneLoopStructure(Loop *OriginalLoop, Function *F, LoopInfo &LI,
                                ValueToValueMapTy &ValueMap,
                                SmallVector<BasicBlock *, 8> &NewBlocks) {
  Loop *NewLoop = LI.AllocateLoop();

  // Clone LoopPreHeader if it exists
  BasicBlock *OriginalPreheaderBB = OriginalLoop->getLoopPreheader();
  BasicBlock *ClonedPreheaderBB = nullptr;
  if (OriginalPreheaderBB) {
    ClonedPreheaderBB =
        CloneBasicBlock(OriginalPreheaderBB, ValueMap, ".clone", F);
    ValueMap[OriginalPreheaderBB] = ClonedPreheaderBB;
    NewBlocks.push_back(ClonedPreheaderBB);
  }

  // Clone all blocks in the loop
  for (BasicBlock *BB : OriginalLoop->blocks()) {
    BasicBlock *ClonedBB = CloneBasicBlock(BB, ValueMap, ".clone", F);
    ValueMap[BB] = ClonedBB;
    NewBlocks.push_back(ClonedBB);
  }

  // Add new blocks to the new loop and update LoopInfo
  for (BasicBlock *BB : NewBlocks) {
    if (LI.getLoopFor(BB->getUniquePredecessor()) == OriginalLoop) {
      NewLoop->addBasicBlockToLoop(BB, LI);
    }
  }

  // Remap instructions and PHI nodes in the new loop
  remapInstructionsInBlocks(NewBlocks, ValueMap);
  LI.addTopLevelLoop(NewLoop);

  return NewLoop;
}

/// Apply the appropriate control flow transformation based on strategy
/// \param F The function to transform
/// \param ClonedPreheaderBB The cloned preheader basic block
/// \param NewBlocks Vector of newly created basic blocks
/// \param Strategy The transformation strategy to apply
static void
applyControlFlowTransformation(Function *F, BasicBlock *ClonedPreheaderBB,
                               SmallVector<BasicBlock *, 8> &NewBlocks,
                               SplitStrategy Strategy) {

  switch (Strategy) {
  case SplitStrategy::BIQUAD_FIR:
    LLVM_DEBUG(dbgs() << "Applying loop splitting for FIR/Biquad functions\n");
    insertConditionalBranchWithClonedPreheader(F, ClonedPreheaderBB, NewBlocks);
    break;
  case SplitStrategy::ADDC:
    LLVM_DEBUG(dbgs() << "Applying loop splitting for Addc-like functions\n");
    insertAddcConditionalBranch(F, NewBlocks);
    break;
  case SplitStrategy::DOTPROD:
    LLVM_DEBUG(dbgs() << "Applying loop splitting for Dotprod functions\n");
    insertStandardConditionalBranch(F, NewBlocks);
    break;
  case SplitStrategy::UNKNOWN:
    LLVM_DEBUG(dbgs() << "Skipping loop splitting - unknown function type\n");
    break;
  default:
    llvm_unreachable("Unsupported split strategy");
  }
}

/// Clone loop and apply control flow transformation
/// \param OriginalLoop The loop to process
/// \param F The function containing the loop
/// \param LI Loop info for analysis and updates
/// \param Strategy The transformation strategy to apply
/// \returns The newly created loop
static Loop *cloneLoopWithControlFlow(Loop *OriginalLoop, Function *F,
                                      LoopInfo &LI, SplitStrategy Strategy) {
  ValueToValueMapTy ValueMap;
  SmallVector<BasicBlock *, 8> NewBlocks;

  Loop *NewLoop = cloneLoopStructure(OriginalLoop, F, LI, ValueMap, NewBlocks);

  BasicBlock *ClonedPreheaderBB = nullptr;
  if (BasicBlock *OriginalPreheader = OriginalLoop->getLoopPreheader()) {
    ClonedPreheaderBB = cast<BasicBlock>(ValueMap[OriginalPreheader]);
  }

  applyControlFlowTransformation(F, ClonedPreheaderBB, NewBlocks, Strategy);

  LLVM_DEBUG(F->dump());
  verifyFunction(*F, &llvm::errs());
  return NewLoop;
}

/// Main function to run the RISCVSplitLoopByLength pass
PreservedAnalyses RISCVSplitLoopByLengthPass::run(Function &F,
                                                  FunctionAnalysisManager &AM) {
  if (!EnableRISCVSplitLoopByLength)
    return PreservedAnalyses::all();

  if (verifyFunction(F, &llvm::errs())) {
    LLVM_DEBUG(dbgs() << "Function verification failed!\n");
  }

  // Early exit if already processed
  if (getBasicBlockByName(F, "for.body.clone"))
    return PreservedAnalyses::all();

  auto &LI = AM.getResult<LoopAnalysis>(F);
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);

  SplitStrategy Strategy = determineSplitStrategy(&F);

  if (Strategy == SplitStrategy::UNKNOWN) {
    return PreservedAnalyses::all();
  }

  // Process the first top-level loop
  for (Loop *TopLevelLoop : LI) {
    if (TopLevelLoop->getLoopDepth() == 1) {
      cloneLoopWithControlFlow(TopLevelLoop, &F, LI, Strategy);
      break;
    }
  }

  LLVM_DEBUG(F.dump());

  PreservedAnalyses PA;
  PA.preserve<LoopAnalysis>();
  PA.preserve<DominatorTreeAnalysis>();
  PA.preserveSet<CFGAnalyses>();
  return PA;
}