//===-- RISCVSplitLoopByLength.cpp - Loop splitting optimization
//---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass splits loops into two parts: one for length > 2 and another for
// length <= 2. It's designed to prepare for the esp.lp.setup instruction.
//
// The pass handles several types of functions:
// - Arithmetic operations: add, addc, mulc, sub, mul
// - Dot product operations: dotprod, dotprode
// - Square root calculation: sqrt
// - Biquadratic filter: biquad
// - Finite Impulse Response filter: fir
//
// Main steps of this pass:
// 1. Identify the function type
// 2. Clone the original loop
// 3. Insert an if-else structure to choose between the original and cloned loop
// 4. Update phi nodes and branch instructions accordingly
//
//===----------------------------------------------------------------------===//

#include "RISCVSplitLoopByLength.h"
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
#include "llvm/Passes/PassPlugin.h"
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
#define ESP_ERR_DSP_BASE 0x70000
#define ESP_ERR_DSP_PARAM_OUTOFRANGE (ESP_ERR_DSP_BASE + 3)

enum class SplitType { DOTPROD, ADDC, BIQUAD_FIR, UNKNOWN };

static SplitType currentSplitType = SplitType::UNKNOWN;
// Command line option to enable/disable RISCVSplitLoopByLength optimization
cl::opt<bool> llvm::EnableRISCVSplitLoopByLength(
    "riscv-split-loop-by-length", cl::init(false),
    cl::desc("Enable loop splitting optimization"));

// Get the length parameter of the function
static Value *getLength(Function *F) {
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

// Get a basic block by its name
static inline BasicBlock *getBasicBlockByName(Function &F, StringRef Name) {
  for (BasicBlock &BB : F)
    if (BB.getName() == Name)
      return &BB;
  return nullptr;
}

// Move getelementptr instructions from Entry to NewBB
static void moveGEPInstructions(BasicBlock &Entry, BasicBlock *NewBB) {
  for (auto I = Entry.begin(); I != Entry.end();) {
    if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(I)) {
      I++;
      GEP->moveBefore(NewBB->getTerminator());
    } else {
      ++I;
    }
  }
}

// Insert if-else structure with cloned basic block (for cases with
// LoopPreHeader)
static void insertIfElseWithClonedBB(Function *F, BasicBlock *ClonedPhBB,
                                     SmallVector<BasicBlock *, 8> &NewBlocks) {
  // Rename entry to for.cond.preheader
  BasicBlock &Entry = F->getEntryBlock();
  Entry.setName("for.cond.preheader");

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
  ICmpInst *ICmp = nullptr;
  for (Instruction &I : Entry) {
    if (ICmpInst *CI = dyn_cast<ICmpInst>(&I)) {
      if (CI->getPredicate() == ICmpInst::ICMP_SGT) {
        ICmp = CI;
        break;
      }
    }
  }

  assert(ICmp && "icmp sgt instruction not found");

  // Create new comparison and conditional branch
  Value *Length = ICmp->getOperand(0);
  Value *Cmp =
      Builder.CreateICmpSGT(Length, ConstantInt::get(Length->getType(), 2));
  Builder.CreateCondBr(Cmp, &Entry, ClonedPhBB);
  moveGEPInstructions(Entry, NewEntryBB);
}

// Insert if-else structure (for cases without LoopPreHeader)
static void insertIfElse(Function *F, SmallVector<BasicBlock *, 8> &NewBlocks) {
  BasicBlock &Entry = F->getEntryBlock();
  Entry.setName("for.cond.preheader");

  BasicBlock *ForBodyBB = getBasicBlockByName(*F, "for.body");
  for (BasicBlock &BB : *F) {
    if (BB.getName() == "for.cond.cleanup")
      BB.setName("if.end");
  }
  assert(ForBodyBB && "ForBody must exist");

  BasicBlock *ForBodyCloneBB = nullptr;
  Instruction *FMulAdd = nullptr;
  for (BasicBlock *BB : NewBlocks) {
    if (BB->getName() == "for.body.clone") {
      ForBodyCloneBB = BB;
      for (Instruction &I : *BB) {
        if (RecurrenceDescriptor::isFMulAddIntrinsic(&I)) {
          FMulAdd = &I;
          break;
        }
      }
    }
  }
  assert(ForBodyCloneBB && "ForBodyCloneBB must exist");
  assert(FMulAdd && "FMulAdd must exist");

  LLVMContext &Context = F->getContext();
  IRBuilder<> Builder(Context);
  BasicBlock *NewEntryBB =
      BasicBlock::Create(Context, "entry", F, &F->getEntryBlock());

  Builder.SetInsertPoint(NewEntryBB);
  Value *Length = getLength(F);
  Value *Cmp =
      Builder.CreateICmpSGT(Length, ConstantInt::get(Length->getType(), 2));
  Builder.CreateCondBr(Cmp, ForBodyBB, &Entry);

  // Update phi nodes and successors
  for (BasicBlock &BB : *F) {
    if (BB.getName() == "if.end") {
      if (PHINode *IfEndEntryPhiNode = dyn_cast<PHINode>(&BB.front())) {
        IfEndEntryPhiNode->addIncoming(FMulAdd, ForBodyCloneBB);
      }
    }
    if (BB.getName() == "for.cond.preheader") {
      BB.getTerminator()->setSuccessor(0, ForBodyCloneBB);
    }
  }

  if (PHINode *ForBodyEntryPhiNode = dyn_cast<PHINode>(&ForBodyBB->front())) {
    ForBodyEntryPhiNode->setIncomingBlock(1, NewEntryBB);
    if (PHINode *ForBodyEntryPhiNode2 = dyn_cast<PHINode>(
            ForBodyEntryPhiNode->getNextNonDebugInstruction())) {
      ForBodyEntryPhiNode2->setIncomingBlock(1, NewEntryBB);
    }
  }
  moveGEPInstructions(Entry, NewEntryBB);
}

// Insert if-else structure for addc-like functions
static void insertAddcIfElse(Function *F,
                             SmallVector<BasicBlock *, 8> &NewBlocks) {
  LLVMContext &Context = F->getContext();
  BasicBlock &EntryBB = F->getEntryBlock();

  BasicBlock *OrigForBB = getBasicBlockByName(*F, "for.body");
  BasicBlock *ForCondBB = getBasicBlockByName(*F, "for.cond.preheader");
  BasicBlock *ForBodyCloneBB = getBasicBlockByName(*F, "for.body.clone");
  BasicBlock *ReturnBB = getBasicBlockByName(*F, "return");

  assert(OrigForBB && ForCondBB && ForBodyCloneBB && ReturnBB &&
         "Necessary basic blocks not found");

  BasicBlock *IfEndBB = BasicBlock::Create(Context, "if.end", F, OrigForBB);

  if (BranchInst *EntryBr = dyn_cast<BranchInst>(EntryBB.getTerminator())) {
    EntryBr->setSuccessor(1, IfEndBB);
  }

  IRBuilder<> Builder(IfEndBB);
  Value *Length = getLength(F);
  Value *Cmp = Builder.CreateICmpSGT(
      Length, ConstantInt::get(Length->getType(), 2), "cmp4");
  Builder.CreateCondBr(Cmp, OrigForBB, ForCondBB);

  // Update cloned for.body's terminator
  if (BranchInst *Br = dyn_cast<BranchInst>(ForBodyCloneBB->getTerminator())) {
    if (Br->isConditional() && Br->getSuccessor(1) == OrigForBB) {
      Br->setSuccessor(1, ForCondBB);
    }
  }

  // Update ForCondBB's terminator
  if (BranchInst *Br = dyn_cast<BranchInst>(ForCondBB->getTerminator())) {
    if (Br->isConditional() && Br->getSuccessor(0) == OrigForBB) {
      Br->setSuccessor(0, ForBodyCloneBB);
    }
  }

  // Update PHI nodes
  for (PHINode &Phi : ForCondBB->phis()) {
    if (Phi.getIncomingBlock(0) == &EntryBB) {
      Phi.setIncomingBlock(0, IfEndBB);
    }
    for (unsigned i = 0; i < Phi.getNumIncomingValues(); ++i) {
      if (Phi.getIncomingBlock(i) == OrigForBB) {
        Phi.setIncomingBlock(i, ForBodyCloneBB);
      }
    }
  }

  // Update return instruction
  if (ReturnInst *Ret = dyn_cast<ReturnInst>(ReturnBB->getTerminator())) {
    if (PHINode *RetPhi = dyn_cast<PHINode>(Ret->getReturnValue())) {
      RetPhi->addIncoming(ConstantInt::get(RetPhi->getType(), 0),
                          ForBodyCloneBB);
    } else {
      PHINode *NewRetPhi =
          PHINode::Create(Ret->getReturnValue()->getType(), 3, "retval.0", Ret);
      NewRetPhi->addIncoming(ConstantInt::get(Ret->getReturnValue()->getType(),
                                              ESP_ERR_DSP_PARAM_OUTOFRANGE),
                             &EntryBB);
      NewRetPhi->addIncoming(
          ConstantInt::get(Ret->getReturnValue()->getType(), 0), ForCondBB);
      NewRetPhi->addIncoming(ConstantInt::get(NewRetPhi->getType(), 0),
                             ForBodyCloneBB);
      Ret->setOperand(0, NewRetPhi);
    }
  }

  // Update OrigForBB's PHI nodes
  for (PHINode &Phi : OrigForBB->phis()) {
    if (Phi.getNumIncomingValues() > 1) {
      Phi.setIncomingBlock(1, IfEndBB);
    }
  }

  // Reorder basic blocks
  ForBodyCloneBB->moveBefore(ReturnBB);
  IfEndBB->moveBefore(ForCondBB);

  LLVM_DEBUG(F->dump());
}

// Check if the function is addc-like
static bool isAddcLike(Function *F) {

  if (F->size() != 4)
    return false;

  BasicBlock *Entry = getBasicBlockByName(*F, "entry");
  BasicBlock *ForCondPreheader = getBasicBlockByName(*F, "for.cond.preheader");
  BasicBlock *ForBody = getBasicBlockByName(*F, "for.body");
  BasicBlock *Return = getBasicBlockByName(*F, "return");

  if (!ForBody || !ForCondPreheader || !Entry || !Return)
    return false;

  // Check successors
  if (Entry->getTerminator()->getSuccessor(0) != Return ||
      Entry->getTerminator()->getSuccessor(1) != ForCondPreheader)
    return false;

  if (ForCondPreheader->getTerminator()->getSuccessor(0) != ForBody ||
      ForCondPreheader->getTerminator()->getSuccessor(1) != Return)
    return false;

  if (ForBody->getTerminator()->getSuccessor(0) != Return ||
      ForBody->getTerminator()->getSuccessor(1) != ForBody)
    return false;

  return true;
}

// Check if the function is dotprod-like
static bool isDotProdLike(Function *F) {

  if (F->size() != 3)
    return false;

  BasicBlock *Entry = getBasicBlockByName(*F, "entry");
  BasicBlock *ForCondCleanup = getBasicBlockByName(*F, "for.cond.cleanup");
  BasicBlock *ForBody = getBasicBlockByName(*F, "for.body");

  if (!ForBody || !ForCondCleanup || !Entry)
    return false;

  // Check successors
  if (Entry->getTerminator()->getNumSuccessors() != 2 ||
      Entry->getTerminator()->getSuccessor(0) != ForBody ||
      Entry->getTerminator()->getSuccessor(1) != ForCondCleanup)
    return false;

  if (ForBody->getTerminator()->getNumSuccessors() != 2 ||
      ForBody->getTerminator()->getSuccessor(0) != ForCondCleanup ||
      ForBody->getTerminator()->getSuccessor(1) != ForBody)
    return false;

  // Check for float PHI node in ForBody
  for (PHINode &Phi : ForBody->phis()) {
    if (Phi.getType()->isFloatTy())
      return true;
  }

  return false;
}

// Check if the function should use insertAddcIfElse
static bool shouldInsertAddcIfElse(Function *F) {
  if (F->empty() || F->arg_empty() || !isAddcLike(F))
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

// Check if the function is biquad
static bool isBiquad(Function *F) {

  if (F->size() != 4)
    return false;

  BasicBlock *ForBody = getBasicBlockByName(*F, "for.body");
  BasicBlock *ForBodyLrPh = getBasicBlockByName(*F, "for.body.lr.ph");
  BasicBlock *ForCondCleanup = getBasicBlockByName(*F, "for.cond.cleanup");
  BasicBlock *Entry = getBasicBlockByName(*F, "entry");

  if (!ForBody || !ForBodyLrPh || !ForCondCleanup || !Entry)
    return false;

  // Check successors
  if (Entry->getTerminator()->getNumSuccessors() != 2 ||
      Entry->getTerminator()->getSuccessor(0) != ForBodyLrPh ||
      Entry->getTerminator()->getSuccessor(1) != ForCondCleanup)
    return false;

  if (ForBodyLrPh->getTerminator()->getNumSuccessors() != 1 ||
      ForBodyLrPh->getTerminator()->getSuccessor(0) != ForBody)
    return false;

  if (ForBody->getTerminator()->getNumSuccessors() != 2 ||
      ForBody->getTerminator()->getSuccessor(0) != ForCondCleanup ||
      ForBody->getTerminator()->getSuccessor(1) != ForBody)
    return false;

  // Check PHI nodes in ForBody
  int floatPhiCount = 0;
  int i32PhiCount = 0;
  for (PHINode &Phi : ForBody->phis()) {
    if (Phi.getType()->isFloatTy())
      floatPhiCount++;
    else if (Phi.getType()->isIntegerTy(32))
      i32PhiCount++;
  }

  return (floatPhiCount == 2 && i32PhiCount == 1);
}

// Check if the function is FIR
static bool isFIR(Function *F) {

  if (F->size() != 10)
    return false;

  BasicBlock *ForBody = getBasicBlockByName(*F, "for.body");
  BasicBlock *ForBodyLrPh = getBasicBlockByName(*F, "for.body.lr.ph");
  BasicBlock *ForCondCleanup = getBasicBlockByName(*F, "for.cond.cleanup");
  BasicBlock *Entry = getBasicBlockByName(*F, "entry");

  if (!ForBody || !ForBodyLrPh || !ForCondCleanup || !Entry)
    return false;

  // Check successors
  if (Entry->getTerminator()->getNumSuccessors() != 2 ||
      Entry->getTerminator()->getSuccessor(0) != ForBodyLrPh ||
      Entry->getTerminator()->getSuccessor(1) != ForCondCleanup)
    return false;

  if (ForBodyLrPh->getTerminator()->getNumSuccessors() != 1 ||
      ForBodyLrPh->getTerminator()->getSuccessor(0) != ForBody)
    return false;

  return true;
}

// Check if the function should use insertIfElseWithClonedBB
static bool shouldInsertIfElseWithClonedBB(Function *F) {
  if (F->empty() || F->arg_empty() || (!isBiquad(F) && !isFIR(F)))
    return false;

  BasicBlock &EntryBB = F->getEntryBlock();
  BranchInst *EntryBr = dyn_cast<BranchInst>(EntryBB.getTerminator());
  if (!EntryBr || !EntryBr->isConditional())
    return false;

  // Check for icmp sgt i32 %len, 0
  if (EntryBB.empty())
    return false;

  ICmpInst *ICmp = dyn_cast<ICmpInst>(&EntryBB.front());
  if (!ICmp || ICmp->getPredicate() != ICmpInst::ICMP_SGT)
    return false;

  Value *Op0 = ICmp->getOperand(0);
  Value *Op1 = ICmp->getOperand(1);
  if (!isa<Argument>(Op0) || !isa<ConstantInt>(Op1))
    return false;

  return cast<ConstantInt>(Op1)->isZero();

  return false;
}

// Check if the function should use insertIfElse for dotprod
static bool shouldDotprodInsertIfElse(Function *F) {
  if (F->empty() || F->arg_size() < 3 || !isDotProdLike(F))
    return false;

  if (F->size() != 3 && F->size() != 4)
    return false;

  bool hasEntry = getBasicBlockByName(*F, "entry") != nullptr;
  bool hasForBody = getBasicBlockByName(*F, "for.body") != nullptr;
  bool hasForCondPreheader =
      getBasicBlockByName(*F, "for.cond.cleanup") != nullptr;

  if (!hasForBody || !hasForCondPreheader || !hasEntry)
    return false;

  // Check for fmuladd instruction
  bool hasFMulAdd = false;
  for (BasicBlock &BB : *F) {
    for (Instruction &I : BB) {
      if (RecurrenceDescriptor::isFMulAddIntrinsic(&I)) {
        hasFMulAdd = true;
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
    if (hasFMulAdd)
      break;
  }
  if (!hasFMulAdd)
    return false;

  // Check for icmp sgt i32 %len, 0
  BasicBlock &EntryBB = F->getEntryBlock();
  Instruction &FirstInst = EntryBB.front();
  
  ICmpInst *ICmp = dyn_cast<ICmpInst>(&FirstInst);
  if (!ICmp)
    return false;
    
  if (ICmp->getPredicate() != ICmpInst::ICMP_SGT)
    return false;
    
  ConstantInt *CI = dyn_cast<ConstantInt>(ICmp->getOperand(1));
  if (!CI)
    return false;
    
  return CI->isZero();
}

// Clone the loop
static Loop *CloneLoop(Loop *L, Function *F, LoopInfo &LI) {
  ValueToValueMapTy VMap;
  SmallVector<BasicBlock *, 8> NewBlocks;
  Loop *NewLoop = LI.AllocateLoop();

  // Clone LoopPreHeader if it exists
  BasicBlock *PhBB = L->getLoopPreheader();
  BasicBlock *ClonedPhBB = nullptr;
  if (PhBB) {
    ClonedPhBB = CloneBasicBlock(PhBB, VMap, ".clone", F);
    VMap[PhBB] = ClonedPhBB;
    NewBlocks.push_back(ClonedPhBB);
  }

  // Clone all blocks in the loop
  for (BasicBlock *BB : L->blocks()) {
    BasicBlock *NewBB = CloneBasicBlock(BB, VMap, ".clone", F);
    VMap[BB] = NewBB;
    NewBlocks.push_back(NewBB);
  }

  // Add new blocks to the new loop and update LoopInfo
  for (BasicBlock *BB : NewBlocks) {
    if (LI.getLoopFor(BB->getUniquePredecessor()) == L) {
      NewLoop->addBasicBlockToLoop(BB, LI);
    }
  }

  // Remap instructions and PHI nodes in the new loop
  remapInstructionsInBlocks(NewBlocks, VMap);

  // Add the new loop to the top level
  LI.addTopLevelLoop(NewLoop);

  // Apply the appropriate transformation based on the function type
  if (PhBB && currentSplitType == SplitType::BIQUAD_FIR) {
    LLVM_DEBUG(errs() << "Applying RISCVSplitLoopByLength for FIR/Biquad\n");
    insertIfElseWithClonedBB(F, ClonedPhBB, NewBlocks);
  } else if (currentSplitType == SplitType::ADDC) {
    LLVM_DEBUG(errs() << "Applying RISCVSplitLoopByLength for Addc-like\n");
    insertAddcIfElse(F, NewBlocks);
  } else if (currentSplitType == SplitType::DOTPROD) {
    LLVM_DEBUG(errs() << "Applying RISCVSplitLoopByLength for Dotprod\n");
    insertIfElse(F, NewBlocks);
  } else if (currentSplitType == SplitType::UNKNOWN) {
    LLVM_DEBUG(errs() << "Skipping RISCVSplitLoopByLength\n");
  } else {
    llvm_unreachable("Unknown function type");
  }

  LLVM_DEBUG(F->dump());
  verifyFunction(*F, &llvm::errs());
  return NewLoop;
}

// Main function to run the RISCVSplitLoopByLength pass
PreservedAnalyses
llvm::RISCVSplitLoopByLengthPass::run(Function &F,
                                      FunctionAnalysisManager &AM) {
  if (!EnableRISCVSplitLoopByLength)
    return PreservedAnalyses::all();

  if (verifyFunction(F, &llvm::errs())) {
    LLVM_DEBUG(errs() << "Function verification failed!\n");
  }

  // Skip if the function already has a for.body.clone basic block
  if (getBasicBlockByName(F, "for.body.clone"))
    return PreservedAnalyses::all();

  auto &LI = AM.getResult<LoopAnalysis>(F);
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);

  if (shouldDotprodInsertIfElse(&F)) {
    currentSplitType = SplitType::DOTPROD;
  } else if (shouldInsertAddcIfElse(&F)) {
    currentSplitType = SplitType::ADDC;
  } else if (shouldInsertIfElseWithClonedBB(&F)) {
    currentSplitType = SplitType::BIQUAD_FIR;
  } else {
    currentSplitType = SplitType::UNKNOWN;
    return PreservedAnalyses::all();
  }
  // Clone the first top-level loop
  for (Loop *L : LI) {
    if (L->getLoopDepth() == 1) {
      CloneLoop(L, &F, LI);
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