//===-- RISCVIntLoopUnrollAndRemainder.cpp - Loop Unrolling Pass --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This pass implements specialized loop unrolling optimizations for RISC-V DSP
/// operations. It targets several types of DSP functions including:
/// - INT16/INT8 dot product operations
/// - INT16 add/multiply/FIR filter operations
///
/// The pass performs:
/// - Pattern-specific loop unrolling (factors of 8 or 16)
/// - Remainder loop handling
/// - Post-unroll optimizations and instruction reordering
///
/// This optimization aims to improve performance of DSP operations on RISC-V
/// processors while preserving correct semantics.
///
//===----------------------------------------------------------------------===//

#include "RISCVIntLoopUnrollAndRemainder.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-int-loop-unroll-and-remainder"

// Command line option to enable the RISCVIntLoopUnrollAndRemainder pass
cl::opt<bool> llvm::EnableRISCVIntLoopUnrollAndRemainder(
    "riscv-int-loop-unroll-and-remainder", cl::init(false),
    cl::desc("Enable integer loop unrolling and remainder specific loop"));

void DspiInt16DotprodHandler::handlePhiNodes(BasicBlock *BB,
                                             PHINode *IndexLcssa) {
  for (PHINode &Phi : BB->phis()) {
    if (Phi.getType()->isIntegerTy(32)) {
      Phi.setIncomingValue(1, IndexLcssa);
      Phi.setIncomingValue(0, getLastAddInst<int32_t>(BB));
    } else if (Phi.getType()->isIntegerTy(64)) {
      Phi.setIncomingValue(0, getFirstAddInst<int64_t>(BB));
    }
  }
}

void DspiInt8DotprodHandler::handlePhiNodes(BasicBlock *BB,
                                            PHINode *IndexLcssa) {
  PHINode *IndexPhiClone = getFirstI32Phi(BB);
  Value *IncClone = nullptr;
  for (User *U : IndexPhiClone->users()) {
    if (auto *Add = dyn_cast<BinaryOperator>(U)) {
      if (Add->getOpcode() == Instruction::Add) {
        IncClone = Add;
        break;
      }
    }
  }
  IndexPhiClone->setIncomingValue(0, IncClone);
  IndexPhiClone->setIncomingValue(1, IndexLcssa);

  PHINode *AccPhiClone = getLastI32Phi(BB);
  Value *AddClone = nullptr;
  for (User *U : AccPhiClone->users()) {
    if (auto *Add = dyn_cast<BinaryOperator>(U)) {
      if (Add->getOpcode() == Instruction::Add) {
        AddClone = Add;
        break;
      }
    }
  }
  AccPhiClone->setIncomingValue(0, AddClone);
}

/// Check basic block properties
static bool hasValidSuccessors(BasicBlock *BB, unsigned NumSuccessors,
                               BasicBlock *Succ1, BasicBlock *Succ2 = nullptr) {
  if (!BB)
    return false;
  return checkSuccessors(BB, NumSuccessors, Succ1, Succ2);
}

/// Check return block
static bool isValidReturnBlock(BasicBlock *ReturnBB) {
  return ReturnBB && succ_empty(ReturnBB) &&
         isa<ReturnInst>(ReturnBB->getTerminator());
}

// Check DSPS Int16 dot product type
bool DspsInt16DotprodHandler::checkDspsInt16DotprodType(Function &F) {
  if (F.size() != 6 || F.arg_size() != 5)
    return false;

  BasicBlock *EntryBB = &F.getEntryBlock();
  BasicBlock *ForCondCleanupBB = getBasicBlockByName(F, "for.cond.cleanup");
  BasicBlock *ForBodyBB = getBasicBlockByName(F, "for.body");
  BasicBlock *IfThenBB = getBasicBlockByName(F, "if.then");
  BasicBlock *IfElseBB = getBasicBlockByName(F, "if.else");
  BasicBlock *IfEndBB = getBasicBlockByName(F, "if.end");

  if (!EntryBB || !ForCondCleanupBB || !ForBodyBB || !IfThenBB || !IfElseBB ||
      !IfEndBB)
    return false;

  return hasValidSuccessors(EntryBB, 2, ForBodyBB, ForCondCleanupBB) &&
         hasValidSuccessors(ForCondCleanupBB, 2, IfThenBB, IfElseBB) &&
         hasValidSuccessors(ForBodyBB, 2, ForCondCleanupBB, ForBodyBB) &&
         hasValidSuccessors(IfThenBB, 1, IfEndBB) &&
         hasValidSuccessors(IfElseBB, 1, IfEndBB) &&
         isValidReturnBlock(IfEndBB);
}

// Check DSPS Int16 add type
bool DspsInt16AddHandler::checkDspsInt16AddType(Function &F) {
  if (F.size() != 4 || F.arg_size() != 8)
    return false;

  BasicBlock *EntryBB = &F.getEntryBlock();
  BasicBlock *ForCondPreheaderBB = getBasicBlockByName(F, "for.cond.preheader");
  BasicBlock *ForBodyBB = getBasicBlockByName(F, "for.body");
  BasicBlock *ReturnBB = getBasicBlockByName(F, "return");

  if (!EntryBB || !ForCondPreheaderBB || !ForBodyBB || !ReturnBB)
    return false;

  return hasValidSuccessors(EntryBB, 2, ReturnBB, ForCondPreheaderBB) &&
         hasValidSuccessors(ForCondPreheaderBB, 2, ForBodyBB, ReturnBB) &&
         hasValidSuccessors(ForBodyBB, 2, ReturnBB, ForBodyBB) &&
         isValidReturnBlock(ReturnBB);
}

// Check DSPS Int16 multiply constant type
bool DspsInt16MulCHandler::checkDspsInt16MulCType(Function &F) {
  if (F.size() != 5 || F.arg_size() != 6)
    return false;

  BasicBlock *EntryBB = &F.getEntryBlock();
  BasicBlock *ForCondPreheaderBB = getBasicBlockByName(F, "for.cond.preheader");
  BasicBlock *ForBodyLrPhBB = getBasicBlockByName(F, "for.body.lr.ph");
  BasicBlock *ForBodyBB = getBasicBlockByName(F, "for.body");
  BasicBlock *ReturnBB = getBasicBlockByName(F, "return");

  if (!EntryBB || !ForCondPreheaderBB || !ForBodyLrPhBB || !ForBodyBB ||
      !ReturnBB)
    return false;

  return hasValidSuccessors(EntryBB, 2, ReturnBB, ForCondPreheaderBB) &&
         hasValidSuccessors(ForCondPreheaderBB, 2, ForBodyLrPhBB, ReturnBB) &&
         hasValidSuccessors(ForBodyLrPhBB, 1, ForBodyBB) &&
         hasValidSuccessors(ForBodyBB, 2, ReturnBB, ForBodyBB) &&
         isValidReturnBlock(ReturnBB);
}

// Check DSPS Int16 FIRD Phi nodes
bool DspsInt16FirdHandler::checkDspsInt16FirdPhiNodes(BasicBlock *ForBody) {
  int I16Count = 0, I32Count = 0, I64Count = 0, TotalPhiCount = 0;

  for (auto &Phi : ForBody->phis()) {
    TotalPhiCount++;
    if (Phi.getType()->isIntegerTy(16))
      I16Count++;
    else if (Phi.getType()->isIntegerTy(32))
      I32Count++;
    else if (Phi.getType()->isIntegerTy(64))
      I64Count++;
  }

  return TotalPhiCount == 3 && I16Count == 1 && I32Count == 1 && I64Count == 1;
}

// Check DSPS Int16 FIRD type
bool DspsInt16FirdHandler::checkDspsInt16FIRDType(Function &F, LoopInfo &LI) {
  int LoopCount = 0;
  for (Loop *L : LI) {
    for (Loop *SubL : L->getSubLoops()) {
      LoopCount++;
      if (!checkDspsInt16FirdUnrollPattern(*SubL))
        continue;

      BasicBlock *ForBodyLrPh = SubL->getLoopPreheader();
      BasicBlock *ForBody = SubL->getHeader();
      if (!ForBodyLrPh || !ForBody)
        return false;

      LoadInst *LI = getFirstInst<LoadInst>(ForBodyLrPh);
      if (!LI || !checkDspsInt16FirdPhiNodes(ForBody))
        return false;
    }
  }
  return LoopCount == 3;
}

// Check DSPI dot product common type
static bool validateDspiDotprodCommonStructure(Function &F, LoopInfo &LI,
                                               BasicBlock *&OuterLoopHeader,
                                               BasicBlock *&InnerLoopHeader) {
  for (Loop *OuterLoop : LI) {
    OuterLoopHeader = OuterLoop->getHeader();
    BasicBlock *OuterLoopPreheader = OuterLoop->getLoopPreheader();

    if (!OuterLoopHeader || !OuterLoopPreheader ||
        OuterLoopPreheader->getSingleSuccessor() != OuterLoopHeader)
      return false;

    for (Loop *InnerLoop : OuterLoop->getSubLoops()) {
      InnerLoopHeader = InnerLoop->getHeader();
      if (!InnerLoopHeader ||
          InnerLoop->getLoopPredecessor() != OuterLoopHeader)
        return false;
      if (succ_size(OuterLoopHeader) != 2 ||
          OuterLoopHeader->getTerminator()->getSuccessor(0) != InnerLoopHeader)
        return false;
      if (succ_size(InnerLoopHeader) != 2 ||
          InnerLoopHeader->getTerminator()->getSuccessor(1) != InnerLoopHeader)
        return false;

      if (InnerLoopHeader->getTerminator()->getSuccessor(0) !=
          OuterLoopHeader->getTerminator()->getSuccessor(1))
        return false;

      return true;
    }
  }
  return false;
}

// Check DSPI Int16 dot product type
bool DspiInt16DotprodHandler::checkDspiInt16DotprodType(Function &F,
                                                        LoopInfo &LI) {
  BasicBlock *OuterLoopHeader = nullptr;
  BasicBlock *InnerLoopHeader = nullptr;
  return validateDspiDotprodCommonStructure(F, LI, OuterLoopHeader,
                                            InnerLoopHeader) &&
         getFirstI64Phi(OuterLoopHeader) != nullptr;
}

// Check DSPI Int8 dot product type
bool DspiInt8DotprodHandler::checkDspiInt8DotprodType(Function &F,
                                                      LoopInfo &LI) {
  BasicBlock *OuterLoopHeader = nullptr;
  BasicBlock *InnerLoopHeader = nullptr;
  return validateDspiDotprodCommonStructure(F, LI, OuterLoopHeader,
                                            InnerLoopHeader) &&
         !getFirstI64Phi(OuterLoopHeader) &&
         !getFirstFMulAddInst(InnerLoopHeader);
}

// Check DSPM Int16 multiply type
bool DspmInt16MultHandler::checkDspmInt16MultType(Function &F, LoopInfo &LI) {
  for (Loop *L : LI) {
    // Check first level loop
    BasicBlock *ForCond1PreheaderLrPh = L->getLoopPreheader();
    BasicBlock *ForCond1Preheader = L->getHeader();
    if (!ForCond1PreheaderLrPh || !ForCond1Preheader) {
      return false;
    }
    SmallVector<BasicBlock *, 8> ExitBlocks;
    L->getExitBlocks(ExitBlocks);
    if (ExitBlocks.size() != 1) {
      return false;
    }
    BasicBlock *ForCondCleanup = ExitBlocks[0];
    BasicBlock *ForCondCleanup3 = L->getExitingBlock();

    if (!ForCond1PreheaderLrPh || !ForCond1Preheader || !ForCondCleanup ||
        !ForCondCleanup3 || ForCondCleanup3 != L->getLoopLatch() ||
        L->getSubLoops().size() != 1)
      return false;

    // Check second level loop
    Loop *SubL = L->getSubLoops().front();
    if (!SubL)
      return false;

    BasicBlock *ForBody4LrPh = SubL->getLoopPreheader();
    BasicBlock *ForBody4 = SubL->getHeader();
    BasicBlock *ForCondCleanup8 = SubL->getExitingBlock();

    if (!ForBody4LrPh || !ForBody4 || SubL->getExitBlock() != ForCondCleanup3 ||
        SubL->getLoopLatch() != ForCondCleanup8 ||
        SubL->getSubLoops().size() != 1)
      return false;

    // Check third level loop
    Loop *SubsubL = SubL->getSubLoops().front();
    if (!SubsubL || !SubsubL->getSubLoops().empty() ||
        ForBody4 != SubsubL->getLoopPredecessor())
      return false;

    BasicBlock *ForBody9 = SubsubL->getHeader();
    if (!ForBody9 || ForBody9 != SubsubL->getLoopLatch() ||
        ForBody9 != SubsubL->getExitingBlock() ||
        ForCondCleanup8 != SubsubL->getExitBlock())
      return false;

    return getFirstI64Phi(ForBody9) != nullptr;
  }
  return false;
}

static std::pair<Value *, Value *>
createEntryBlockModifications(BasicBlock &EntryBB, int UnrollCount) {
  ICmpInst *CompareInst = getLastInst<ICmpInst>(&EntryBB);
  assert(CompareInst && "Icmp not found");
  Value *StartIndex = CompareInst->getOperand(1);
  Value *EndIndex = CompareInst->getOperand(0);
  // Insert new instructions before Icmp
  IRBuilder<> Builder(CompareInst);
  Value *AndValue = Builder.CreateAnd(
      EndIndex, ConstantInt::getSigned(EndIndex->getType(), -UnrollCount), "and");
  CompareInst->setOperand(0, AndValue);
  CompareInst->setOperand(1, StartIndex);
  return std::make_pair(AndValue, EndIndex);
}

void DspsInt16DotprodHandler::groupSameInstForDotprod(
    BasicBlock *ForBodyMerged) {
  // Collect different types of instructions
  SmallVector<PHINode *> PhiNodes;
  SmallVector<Instruction *> OrInsts, GepInsts, LoadInsts, SextI16Insts,
      MulNswInsts, SextI32Insts;

  // Categorize instructions by type
  for (Instruction &Inst : *ForBodyMerged) {
    if (auto *PhiNode = dyn_cast<PHINode>(&Inst)) {
      PhiNodes.push_back(PhiNode);
    } else if (Inst.getOpcode() == Instruction::Or) {
      OrInsts.push_back(&Inst);
    } else if (isa<GetElementPtrInst>(&Inst)) {
      GepInsts.push_back(&Inst);
    } else if (isa<LoadInst>(&Inst)) {
      LoadInsts.push_back(&Inst);
    } else if (auto *SextInst = dyn_cast<SExtInst>(&Inst)) {
      if (SextInst->getSrcTy()->isIntegerTy(16)) {
        SextI16Insts.push_back(SextInst);
      } else if (SextInst->getSrcTy()->isIntegerTy(32)) {
        SextI32Insts.push_back(SextInst);
      }
    } else if (auto *MulInst = dyn_cast<BinaryOperator>(&Inst)) {
      if (MulInst->getOpcode() == Instruction::Mul &&
          MulInst->hasNoSignedWrap()) {
        MulNswInsts.push_back(MulInst);
      }
    }
  }

  // If no PHI nodes are found, return
  if (PhiNodes.empty()) {
    return;
  }

  // Reorder instructions
  Instruction *InsertPoint = PhiNodes.back()->getNextNode();

  // Move instructions in the desired order
  moveInstructionsBeforePoint(OrInsts, InsertPoint);
  moveInstructionsBeforePoint(GepInsts, InsertPoint);
  moveInstructionsBeforePoint(LoadInsts, InsertPoint);
  moveInstructionsBeforePoint(SextI16Insts, InsertPoint);
  moveInstructionsBeforePoint(MulNswInsts, InsertPoint);
  moveInstructionsBeforePoint(SextI32Insts, InsertPoint);
}

void DspsInt16FirdHandler::groupSameInstForFird(BasicBlock *ForBodyMerged) {
  // Collect different types of instructions
  SmallVector<PHINode *> PhiNodes;
  SmallVector<Instruction *> GepInsts, LoadInsts, SextI16Insts, MulNswInsts,
      SextI32Insts, AddNswI64Insts;

  // Categorize instructions by type
  for (Instruction &I : *ForBodyMerged) {
    if (auto *Phi = dyn_cast<PHINode>(&I)) {
      PhiNodes.push_back(Phi);
    } else if (isa<GetElementPtrInst>(&I)) {
      GepInsts.push_back(&I);
    } else if (isa<LoadInst>(&I)) {
      LoadInsts.push_back(&I);
    } else if (auto *SextInst = dyn_cast<SExtInst>(&I)) {
      if (SextInst->getSrcTy()->isIntegerTy(16)) {
        SextI16Insts.push_back(SextInst);
      } else if (SextInst->getSrcTy()->isIntegerTy(32)) {
        SextI32Insts.push_back(SextInst);
      }
    } else if (auto *MulInst = dyn_cast<BinaryOperator>(&I)) {
      if (MulInst->getOpcode() == Instruction::Mul &&
          MulInst->hasNoSignedWrap()) {
        MulNswInsts.push_back(MulInst);
      }
    } else if (auto *AddInst = dyn_cast<BinaryOperator>(&I)) {
      if (AddInst->getOpcode() == Instruction::Add &&
          AddInst->hasNoSignedWrap() && AddInst->getType()->isIntegerTy(64)) {
        AddNswI64Insts.push_back(AddInst);
      }
    }
  }

  // If no PHI nodes are found, return
  if (PhiNodes.empty()) {
    return;
  }

  // Reorder instructions
  Instruction *InsertPoint = PhiNodes.back()->getNextNode();

  // Move instructions in the desired order
  moveInstructionsBeforePoint(GepInsts, InsertPoint);
  moveInstructionsBeforePoint(LoadInsts, InsertPoint);
  moveInstructionsBeforePoint(SextI16Insts, InsertPoint);
  moveInstructionsBeforePoint(MulNswInsts, InsertPoint);
  moveInstructionsBeforePoint(SextI32Insts, InsertPoint);
  moveInstructionsBeforePoint(AddNswI64Insts, InsertPoint);
  // Check definition and use of each instruction
  for (Instruction &I : *ForBodyMerged) {
    // Skip PHI nodes since they must be at the beginning of the block
    if (isa<PHINode>(&I)) {
      continue;
    }

    // Get all uses of this instruction
    for (Use &U : I.uses()) {
      Instruction *User = dyn_cast<Instruction>(U.getUser());
      if (!User || User->getParent() != ForBodyMerged) {
        continue;
      }

      // If use comes before definition, move definition before use
      if (User->comesBefore(&I)) {
        I.moveBefore(User);
        return;
      }
    }
  }
}

void DspsInt16MulCHandler::preTransformDspsInt16MulC(Function &F) {
  BasicBlock *ForCondPreheader = getBasicBlockByName(F, "for.cond.preheader");
  BasicBlock *ForBodyLrPh = getBasicBlockByName(F, "for.body.lr.ph");

  // Get all instructions in for.body.lr.ph except the terminator
  SmallVector<Instruction *, 8> InstToMove;
  for (auto &I : *ForBodyLrPh) {
    if (!I.isTerminator()) {
      InstToMove.push_back(&I);
    }
  }

  Instruction *InsertPoint = &*ForCondPreheader->getFirstInsertionPt();
  for (auto *I : InstToMove) {
    I->moveBefore(InsertPoint);
  }

  // Get the Successor of for.body.lr.ph
  BasicBlock *Successor = ForBodyLrPh->getTerminator()->getSuccessor(0);

  // Update all PHI nodes that use for.body.lr.ph
  for (BasicBlock *Succ : successors(ForBodyLrPh)) {
    for (PHINode &PN : Succ->phis()) {
      int Idx = PN.getBasicBlockIndex(ForBodyLrPh);
      if (Idx != -1) {
        PN.setIncomingBlock(Idx, ForCondPreheader);
      }
    }
  }

  // Update the terminator of for.cond.preheader
  ForCondPreheader->getTerminator()->setSuccessor(0, Successor);

  // Erase for.body.lr.ph
  ForBodyLrPh->eraseFromParent();
}

static void mergeAndMoveLoadInst(BasicBlock *Forbody46LrPh,
                                 BasicBlock *Forbody64LrPh, BasicBlock *Entry) {
  // Get two load instructions
  LoadInst *Load46 = getFirstInst<LoadInst>(Forbody46LrPh);
  LoadInst *Load64 = getFirstInst<LoadInst>(Forbody64LrPh);

  // Move the first load instruction to the beginning of the Entry block
  if (Load46) {
    Load46->moveBefore(&*Entry->getFirstInsertionPt());
    // Replace all uses of the second load instruction with the first load
    if (Load64) {
      Load64->replaceAllUsesWith(Load46);
      Load64->eraseFromParent();
    }
  }
}

static void transformUnrollGetElementPtr(BasicBlock *BB) {
  // Collect all getelementptr instructions
  SmallVector<GetElementPtrInst *, 16> GEPs;
  for (auto &I : *BB) {
    if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
      GEPs.push_back(GEP);
    }
  }

  if (GEPs.size() < 2) {
    return;
  }

  // Get the first and second GEP as the base
  GetElementPtrInst *FirstGEP = GEPs[0];
  GetElementPtrInst *SecondGEP = GEPs[1];

  // Process from the third GEP onwards
  for (unsigned I = 2; I < GEPs.size(); I++) {
    GetElementPtrInst *GEP = GEPs[I];

    if (I % 2 == 0) {
      // GEP with even index
      GEP->setOperand(0, FirstGEP);
      int64_t Offset = -(I / 2);
      GEP->setOperand(1,
                      ConstantInt::get(GEP->getOperand(1)->getType(), Offset));
    } else {
      // GEP with odd index
      GEP->setOperand(0, SecondGEP);
      int64_t Offset = (I - 1) / 2;
      GEP->setOperand(1,
                      ConstantInt::get(GEP->getOperand(1)->getType(), Offset));
    }
  }
}

// both support dspi int16 dotprod and dotprod_offset
void DspsInt16DotprodHandler::postUnrollDspsInt16Dotprod(Function &F, Loop &L,
                                                         int Unroll_count) {
  // Get basic blocks
  BasicBlock *EntryBB = &F.getEntryBlock();
  BasicBlock *LoopPreheader = L.getLoopPreheader();

  BasicBlock *ForEnd = getBasicBlockByName(F, "for.cond.cleanup");
  BasicBlock *ForCondCleanupLoopExit =
      getBasicBlockByName(F, "for.cond.cleanup.loopexit");
  assert(ForEnd && "basic block not found");
  ForEnd->setName("for.end85");

  auto [LoopHeaderClone, ForBodyMerged] =
      cloneAndMergeLoop(&L, F, Unroll_count);
  // Create and move basic blocks
  BasicBlock *ForCond73Preheader =
      BasicBlock::Create(F.getContext(), "for.cond73.preheader", &F, ForEnd);
  ForCondCleanupLoopExit->getTerminator()->setSuccessor(0, ForCond73Preheader);
  ForEnd->moveAfter(ForCondCleanupLoopExit);

  LoopHeaderClone->moveAfter(ForBodyMerged);
  ForEnd->moveAfter(LoopHeaderClone);

  // Modify the Entry basic block
  auto [AndValue, EndIndex] =
      createEntryBlockModifications(*EntryBB, Unroll_count);
  ICmpInst *LastICmp = getLastInst<ICmpInst>(ForBodyMerged);
  assert(LastICmp && "Icmp not found");
  LastICmp->setOperand(1, AndValue);
  LastICmp->setPredicate(ICmpInst::ICMP_SLT);
  EntryBB->getTerminator()->setSuccessor(1, ForCond73Preheader);

  // Insert loop preheader instructions
  IRBuilder<> PreheaderBuilder(LoopPreheader->getTerminator());
  Value *MinusOne = PreheaderBuilder.CreateAdd(
      AndValue, ConstantInt::getSigned(AndValue->getType(), -1));
  Value *AndEight = PreheaderBuilder.CreateAnd(
      MinusOne, ConstantInt::getSigned(AndValue->getType(), -Unroll_count));

  // Insert loop exit instructions
  IRBuilder<> ExitBuilder(ForCondCleanupLoopExit->getTerminator());
  Value *AddEight = ExitBuilder.CreateAdd(
      AndEight, ConstantInt::get(AndValue->getType(), Unroll_count));

  // Process PHI nodes
  PHINode *IndexLcssa = PHINode::Create(Type::getInt32Ty(F.getContext()), 2,
                                        "I.0.lcssa", ForCond73Preheader);
  IndexLcssa->addIncoming(ConstantInt::get(Type::getInt32Ty(F.getContext()), 0),
                          EntryBB);
  IndexLcssa->addIncoming(AddEight, ForCondCleanupLoopExit);

  ICmpInst *RemainderLoopCondCmp =
      new ICmpInst(ICmpInst::ICMP_SLT, IndexLcssa, EndIndex, "Cmp74172");
  RemainderLoopCondCmp->insertAfter(IndexLcssa);

  BranchInst *Br =
      BranchInst::Create(LoopHeaderClone, ForEnd, RemainderLoopCondCmp);
  Br->insertAfter(RemainderLoopCondCmp);

  // Clone and process PHI nodes
  PHINode *FirstI64Phi = getFirstI64Phi(ForBodyMerged);
  assert(FirstI64Phi && "first i64 Phi not found");

  SmallVector<Instruction *> AddNswI64Insts;
  for (Instruction &I : *ForBodyMerged) {
    if (auto *BinOp = dyn_cast<BinaryOperator>(&I)) {
      if (BinOp->getOpcode() == Instruction::Add && BinOp->hasNoSignedWrap() &&
          BinOp->getType()->isIntegerTy(64)) {
        AddNswI64Insts.push_back(BinOp);
      }
    }
  }
  assert(!AddNswI64Insts.empty() && "add.nsw i64 instruction not found");

  PHINode *ClonedPhi = cast<PHINode>(FirstI64Phi->clone());
  ClonedPhi->setName("result0.0");
  ClonedPhi->insertAfter(FirstI64Phi);
  auto *Temp = AddNswI64Insts[0];
  ClonedPhi->setIncomingValue(1, Temp);
  Temp->setOperand(0, ClonedPhi);

  // Process return value
  Value *InitialAccValue = nullptr;
  for (PHINode &Phi : ForBodyMerged->phis()) {
    Phi.setIncomingBlock(0, EntryBB);
    auto *Temp = Phi.clone();
    Temp->setName("result0.0.lcssa");
    PHINode *TempPhi = cast<PHINode>(Temp);
    if (!InitialAccValue) {
      InitialAccValue = TempPhi->getIncomingValue(0);
      TempPhi->setIncomingValue(0, ConstantInt::get(Temp->getType(), 0));
      TempPhi->setIncomingBlock(1, ForCondCleanupLoopExit);
      Temp->insertBefore(ForCond73Preheader->getTerminator());
      break;
    }
  }

  movePHINodesToTop(*ForCond73Preheader);

  for (PHINode &Phi : ForBodyMerged->phis()) {
    Phi.setIncomingBlock(0, LoopPreheader);
    Phi.setIncomingValue(0, ConstantInt::get(Phi.getType(), 0));
  }

  swapTerminatorSuccessors(ForBodyMerged);

  // Process final return value
  PHINode *OriginalRetValue = getFirstI64Phi(ForEnd);

  PHINode *ResultLcssa = getLastI64Phi(ForCond73Preheader);
  ResultLcssa->setIncomingValue(0, InitialAccValue);

  Instruction *LastAddNswI64Inst = getLastAddNswI64Inst(LoopHeaderClone);
  OriginalRetValue->setIncomingValue(0, ResultLcssa);
  OriginalRetValue->setIncomingBlock(0, ForCond73Preheader);
  OriginalRetValue->setIncomingValue(1, LastAddNswI64Inst);
  OriginalRetValue->setIncomingBlock(1, LoopHeaderClone);
  Instruction *OriginalRetValueClone = OriginalRetValue->clone();
  PHINode *LoopHeaderCloneFirstI64Phi = getFirstI64Phi(LoopHeaderClone);
  OriginalRetValueClone->insertBefore(LoopHeaderCloneFirstI64Phi);
  LoopHeaderCloneFirstI64Phi->replaceAllUsesWith(OriginalRetValueClone);

  // Final cleanup
  movePHINodesToTop(*ForBodyMerged);

  PHINode *FirstI32PhiLoopHeaderClone = getFirstI32Phi(LoopHeaderClone);
  PHINode *FirstI32PhiForCond73Preheader = getFirstI32Phi(ForCond73Preheader);
  FirstI32PhiLoopHeaderClone->setIncomingValue(1,
                                               FirstI32PhiForCond73Preheader);
  FirstI32PhiLoopHeaderClone->setIncomingBlock(1, ForCond73Preheader);
  FirstI32PhiLoopHeaderClone->setIncomingBlock(0, LoopHeaderClone);

  Instruction *FirstAddNuwNswI32Inst =
      getFirstAddNuwNswI32Inst(LoopHeaderClone);
  FirstI32PhiLoopHeaderClone->setIncomingValue(0, FirstAddNuwNswI32Inst);

  LoopHeaderClone->getTerminator()->setSuccessor(0, ForEnd);
  LoopHeaderClone->getTerminator()->setSuccessor(1, LoopHeaderClone);

  getLastI64Phi(ForBodyMerged)->setIncomingValue(0, InitialAccValue);
  getLastI64Phi(ForBodyMerged)
      ->setIncomingValue(1, getLastAddNswI64Inst(ForBodyMerged));
  runPostPass(F);
  groupSameInstForDotprod(ForBodyMerged);
  runSimplifyDcePasses(F);
}

// unroll factor is 16
void DspsInt16MathHandler::postUnrollInt16MathType(Function &F, Loop &L,
                                                   int Unroll_count) {
  BasicBlock *ForBodyPreheader = L.getLoopPreheader();

  auto [LoopHeaderClone, ForBodyMerged] =
      cloneAndMergeLoop(&L, F, Unroll_count);
  LoopHeaderClone->moveAfter(ForBodyMerged);

  // modify for.cond.preheader
  BasicBlock *ForCondPreheader = getBasicBlockByName(F, "for.cond.preheader");

  ICmpInst *FirstICmp = getFirstInst<ICmpInst>(ForCondPreheader);
  assert(FirstICmp && "Icmp not found");
  Value *Lenarg = FirstICmp->getOperand(0);
  IRBuilder<> Builder(FirstICmp);
  Value *AndValue = Builder.CreateAnd(
      Lenarg, ConstantInt::getSigned(Lenarg->getType(), 1 - Unroll_count), "and");
  FirstICmp->setOperand(0, AndValue);
  ForCondPreheader->getTerminator()->setSuccessor(1, ForBodyMerged);
  swapTerminatorSuccessors(ForCondPreheader);

  getLastInst<ICmpInst>(ForBodyMerged)->setOperand(1, AndValue);
  getLastInst<ICmpInst>(ForBodyMerged)->setPredicate(ICmpInst::ICMP_SLT);

  // modify for.body.preheader
  PHINode *FirstI32Phi = getFirstI32Phi(ForBodyMerged);
  FirstI32Phi->setIncomingBlock(0, ForCondPreheader);
  // Clone PHI node to the beginning of ForBodyPreheader
  PHINode *ClonedPhi = cast<PHINode>(FirstI32Phi->clone());
  // Insert the cloned PHI node at the beginning of ForBodyPreheader
  ClonedPhi->setName("I.0.lcssa");
  ClonedPhi->insertBefore(&*ForBodyPreheader->begin());

  // Create comparison instruction
  Builder.SetInsertPoint(ForBodyPreheader->getTerminator());
  Value *RemainderLoopCondCmp =
      Builder.CreateICmpSLT(ClonedPhi, Lenarg, "cmp226415");

  // Replace original unconditional jump with conditional jump
  BranchInst *OldBr = cast<BranchInst>(ForBodyPreheader->getTerminator());
  BasicBlock *ReturnBB = getBasicBlockByName(F, "return");
  BranchInst *NewBr =
      BranchInst::Create(LoopHeaderClone, ReturnBB, RemainderLoopCondCmp);
  ReplaceInstWithInst(OldBr, NewBr);

  // modify for.body.merged
  ForBodyMerged->getTerminator()->setSuccessor(0, ForBodyMerged);
  ForBodyMerged->getTerminator()->setSuccessor(1, ForBodyPreheader);

  // modify for.body.clone
  getFirstI32Phi(LoopHeaderClone)->setIncomingValue(0, ClonedPhi);
  getFirstI32Phi(LoopHeaderClone)
      ->setIncomingValue(1,
                         getLastInst<ICmpInst>(LoopHeaderClone)->getOperand(0));
  getFirstI32Phi(LoopHeaderClone)->setIncomingBlock(1, LoopHeaderClone);
  LoopHeaderClone->getTerminator()->setSuccessor(1, LoopHeaderClone);

  // modify return
  getFirstI32Phi(ReturnBB)->setIncomingBlock(1, ForBodyPreheader);

  runPostPass(F);

  runSimplifyDcePasses(F);
}

void DspsInt16FirdHandler::postUnrollDspsInt16FIRD(
    Function &F, SmallVector<Loop *, 2> &FIRDWillTransformLoops,
    unsigned Unroll_count, LoopInfo &LI) {

  // ===================== BEGIN FIX V2 =====================

  // 1. Find the target location for hoisting
  BasicBlock *HoistTargetBB =
      getBasicBlockByName(F, "for.cond14.preheader.lr.ph");
  assert(HoistTargetBB && "Could not find the main loop preheader");

  // 2. Find the GEP instruction for the delay pointer
  GetElementPtrInst *DelayGep = nullptr;
  for (Instruction &I : *HoistTargetBB) {
    if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
      if (GEP->getName().starts_with("delay")) {
        DelayGep = GEP;
        break;
      }
    }
  }
  // assert(DelayGep && "Could not find the GEP for the delay pointer");
  if (!DelayGep) {
    return;
  }
  // 3. Find all the load instructions from this address
  SmallVector<LoadInst *, 4> LoadsToHoist;
  for (User *U : DelayGep->users()) {
    if (auto *LI = dyn_cast<LoadInst>(U)) {
      LoadsToHoist.push_back(LI);
    }
  }

  // If no load is found, or only one, no hoisting is needed
  if (LoadsToHoist.size() <= 1) {
    // Can add some logging or just return here
  } else {
    // 4. Create A new load at the hoisting point using the first found load as
    // template
    LoadInst *FirstLoad = LoadsToHoist[0];
    LoadInst *HoistedDelayLoad =
        new LoadInst(FirstLoad->getType(), // Use the type of the template!
                     DelayGep, "delay_ptr.hoisted",
                     false, // isVolatile
                     FirstLoad->getAlign(), FirstLoad->getOrdering(),
                     FirstLoad->getSyncScopeID());
    HoistedDelayLoad->setMetadata(LLVMContext::MD_tbaa,
                                  FirstLoad->getMetadata(LLVMContext::MD_tbaa));
    HoistedDelayLoad->insertBefore(HoistTargetBB->getTerminator());

    // 5. Replace the uses of all old loads and delete them
    for (LoadInst *OldLoad : LoadsToHoist) {
      OldLoad->replaceAllUsesWith(HoistedDelayLoad);
      OldLoad->eraseFromParent();
    }
  }

  runSimplifyDcePasses(F);

  // merge load fir into one and move it into Entry block
  Loop *FirstL = FIRDWillTransformLoops.front();
  Loop *SecondL = FIRDWillTransformLoops.back();
  BasicBlock *ForBody46LrPh = FirstL->getLoopPreheader();
  BasicBlock *ForBody64LrPh = SecondL->getLoopPreheader();
  BasicBlock *Entry = &F.getEntryBlock();
  mergeAndMoveLoadInst(ForBody46LrPh, ForBody64LrPh, Entry);

  BasicBlock *ForCondCleanup20 = ForBody46LrPh->getSinglePredecessor();
  SExtInst *PosSExt = getFirstInst<SExtInst>(ForCondCleanup20);
  SExtInst *LenSExt = getLastInst<SExtInst>(ForCondCleanup20);
  ICmpInst *ExitCondition = getLastInst<ICmpInst>(ForCondCleanup20);
  Value *LenMinus15 = BinaryOperator::CreateNSWAdd(
      LenSExt, ConstantInt::getSigned(LenSExt->getType(), -15), "sub45",
      ExitCondition);
  ExitCondition->setOperand(0, PosSExt);
  ExitCondition->setOperand(1, LenMinus15);

  // Clone and merge loop
  auto [ForBody46Clone, ForBody46Merged] =
      cloneAndMergeLoop(FirstL, F, Unroll_count);
  auto [ForBody64Clone, ForBody64Merged] =
      cloneAndMergeLoop(SecondL, F, Unroll_count);
  ForBody46Clone->moveAfter(ForBody46Merged);
  ForBody64Clone->moveAfter(ForBody64Merged);

  BasicBlock *ForCond58Preheader = ForBody64LrPh->getSinglePredecessor();
  ForBody64LrPh->getTerminator()->setSuccessor(0, ForBody46Clone);
  PHINode *Acc_0_lcssa = getFirstInst<PHINode>(ForCond58Preheader);
  PHINode *Coeff_pos_0_lcssa = getLastInst<PHINode>(ForCond58Preheader);

  BasicBlock *ForCond58PreheaderLoopExit = Acc_0_lcssa->getIncomingBlock(1);
  PHINode *Add_lcssa = cast<PHINode>(Acc_0_lcssa->getIncomingValue(1));
  Acc_0_lcssa->setIncomingValue(
      1, Add_lcssa->getIncomingValue(Add_lcssa->getNumIncomingValues() - 1));

  PHINode *Coeff_pos_0145 = getFirstInst<PHINode>(ForBody46Merged);
  Value *Coeff_pos_0_15 = Coeff_pos_0145->getIncomingValue(1);
  Coeff_pos_0_lcssa->setIncomingValue(1, Coeff_pos_0_15);

  ICmpInst *ExitCond_not_15 = getLastInst<ICmpInst>(ForBody46Merged);
  Value *Inc15_15 = ExitCond_not_15->getOperand(0);

  ICmpInst *LoopCondCmp = getLastInst<ICmpInst>(ForCond58Preheader);
  PHINode *N_0_lcssa =
      PHINode::Create(PosSExt->getType(), 2, "N.0.lcssa", LoopCondCmp);
  N_0_lcssa->addIncoming(PosSExt, ForCondCleanup20);
  N_0_lcssa->addIncoming(Inc15_15, ForCond58PreheaderLoopExit);
  for (auto &Phi : ForCond58Preheader->phis()) {
    Phi.setIncomingBlock(1, ForBody46Merged);
  }

  IRBuilder<> Builder(LoopCondCmp);
  Value *N_0_lcssa_trunc =
      Builder.CreateTrunc(N_0_lcssa, Type::getInt16Ty(F.getContext()));
  LoopCondCmp->setOperand(0, LenSExt);
  LoopCondCmp->setOperand(1, N_0_lcssa);

  ExitCond_not_15->setOperand(1, LenMinus15);
  ExitCond_not_15->setPredicate(ICmpInst::ICMP_SLT);
  ForBody46Merged->getTerminator()->setSuccessor(0, ForCond58Preheader);
  swapTerminatorSuccessors(ForBody46Merged);

  Value *Load_coeffs_len = LenSExt->getOperand(0);
  Value *Load_pos = PosSExt->getOperand(0);
  BasicBlock *ForCond227PreheaderLoopExit = BasicBlock::Create(
      F.getContext(), "for.cond227.preheader.loopexit", &F, ForBody46Clone);
  Builder.SetInsertPoint(ForCond227PreheaderLoopExit);
  Value *Num = Builder.CreateSub(Coeff_pos_0_lcssa, Load_coeffs_len);
  Value *CoeffPosOffset = Builder.CreateAdd(Num, N_0_lcssa_trunc);
  auto *Add_clone = getFirstAddInst<int64_t>(ForBody46Clone);
  for (auto &Phi : ForBody46Clone->phis()) {
    if (Phi.getType()->isIntegerTy(16)) {
      Phi.setIncomingValue(0, Coeff_pos_0_lcssa);
      Phi.setIncomingValue(1, getFirstAddInst<int16_t>(ForBody46Clone));
    } else if (Phi.getType()->isIntegerTy(32)) {
      Phi.setIncomingValue(0, N_0_lcssa);
      Phi.setIncomingValue(1, getFirstAddInst<int32_t>(ForBody46Clone));
    } else if (Phi.getType()->isIntegerTy(64)) {
      Phi.setIncomingValue(0, Acc_0_lcssa);
      Phi.setIncomingValue(1, Add_clone);
    }
  }
  setPHINodesBlock(ForBody46Clone, ForBody64LrPh, ForBody46Clone);

  ForBody46Clone->getTerminator()->setSuccessor(0, ForCond227PreheaderLoopExit);
  ForBody46Clone->getTerminator()->setSuccessor(1, ForBody46Clone);

  BasicBlock *ForCond227Preheader = BasicBlock::Create(
      F.getContext(), "for.cond227.preheader", &F, ForBody46Clone);
  BranchInst::Create(ForCond227Preheader, ForCond227PreheaderLoopExit);

  PHINode *Acc_1_lcssa = PHINode::Create(Type::getInt64Ty(F.getContext()), 2,
                                         "acc.1.lcssa", ForCond227Preheader);
  Acc_1_lcssa->addIncoming(Acc_0_lcssa, ForCond58Preheader);
  Acc_1_lcssa->addIncoming(Add_clone, ForCond227PreheaderLoopExit);

  PHINode *Coeff_pos_1_lcssa =
      PHINode::Create(Type::getInt16Ty(F.getContext()), 2, "coeff_pos.1.lcssa",
                      ForCond227Preheader);
  Coeff_pos_1_lcssa->addIncoming(Coeff_pos_0_lcssa, ForCond58Preheader);
  Coeff_pos_1_lcssa->addIncoming(CoeffPosOffset, ForCond227PreheaderLoopExit);

  Builder.SetInsertPoint(ForCond227Preheader);
  Value *PosMinus15 = Builder.CreateAdd(
      PosSExt, ConstantInt::getSigned(Type::getInt32Ty(F.getContext()), -15),
      "Sub230", true);
  Value *PosGt15Cmp = Builder.CreateICmpSGT(
      Load_pos, ConstantInt::get(Type::getInt16Ty(F.getContext()), 15),
      "Cmp231659");

  BasicBlock *ForCondCleanup63 =
      ForCond58Preheader->getTerminator()->getSuccessor(1);
  ForCond58Preheader->getTerminator()->setSuccessor(1, ForCond227Preheader);

  BasicBlock *ForBody233Preheader = BasicBlock::Create(
      F.getContext(), "for.body233.preheader", &F, ForBody46Clone);

  BasicBlock *ForCond398Preheader =
      BasicBlock::Create(F.getContext(), "for.cond398.preheader", &F,
                         ForBody46Clone->getNextNode());
  Builder.CreateCondBr(PosGt15Cmp, ForBody233Preheader, ForCond398Preheader);

  Builder.SetInsertPoint(ForBody233Preheader);
  Value *PosMasked = Builder.CreateAnd(
      PosSExt, ConstantInt::get(Type::getInt32Ty(F.getContext()), 32752),
      "M47");
  BranchInst::Create(ForBody64Merged, ForBody233Preheader);

  auto CloneAndInsertPhi = [&](PHINode &Phi) {
    PHINode *Phi_clone = cast<PHINode>(Phi.clone());
    Phi_clone->setIncomingBlock(0, ForCond227Preheader);
    Phi_clone->insertInto(ForCond398Preheader, ForCond398Preheader->begin());

    assert(Phi_clone->getParent() == ForCond398Preheader &&
           "PHI node inserted into wrong basic block");
    return Phi_clone;
  };

  DenseMap<unsigned, PHINode *> PhiCloneMap;
  for (auto &Phi : ForBody64Merged->phis()) {
    Phi.setIncomingBlock(0, ForBody233Preheader);
    if (Phi.getType()->isIntegerTy(16)) {
      Phi.setIncomingValue(0, Coeff_pos_1_lcssa);
      PhiCloneMap[16] = CloneAndInsertPhi(Phi);
    } else if (Phi.getType()->isIntegerTy(32)) {
      PHINode *Phi_clone = CloneAndInsertPhi(Phi);
      PhiCloneMap[32] = Phi_clone;
      Phi_clone->setIncomingValue(1, PosMasked);
      Builder.SetInsertPoint(ForCond398Preheader);
      Value *PhiLtPosCmp =
          Builder.CreateICmpSLT(Phi_clone, PosSExt, "Cmp401666");
      Builder.CreateCondBr(PhiLtPosCmp, ForBody64Clone, ForCondCleanup63);
    } else if (Phi.getType()->isIntegerTy(64)) {
      Phi.setIncomingValue(0, Acc_1_lcssa);
      PhiCloneMap[64] = CloneAndInsertPhi(Phi);
    }
  }
  ICmpInst *ExitCond159_not_64 = getLastInst<ICmpInst>(ForBody64Merged);
  ExitCond159_not_64->setPredicate(ICmpInst::ICMP_SLT);
  ExitCond159_not_64->setOperand(1, PosMinus15);
  ForBody64Merged->getTerminator()->setSuccessor(0, ForCond398Preheader);
  swapTerminatorSuccessors(ForBody64Merged);

  auto *AccAddClone = getFirstAddInst<int64_t>(ForBody64Clone);
  for (auto &Phi : ForBody64Clone->phis()) {
    if (Phi.getType()->isIntegerTy(16)) {
      Phi.setIncomingValue(0, PhiCloneMap[16]);
      Phi.setIncomingValue(1, getFirstAddInst<int16_t>(ForBody64Clone));
    } else if (Phi.getType()->isIntegerTy(32)) {
      Phi.setIncomingValue(0, PhiCloneMap[32]);
      Phi.setIncomingValue(1, getFirstAddInst<int32_t>(ForBody64Clone));
    } else if (Phi.getType()->isIntegerTy(64)) {
      Phi.setIncomingValue(0, PhiCloneMap[64]);
      Phi.setIncomingValue(1, AccAddClone);
    }
  }
  setPHINodesBlock(ForBody64Clone, ForCond398Preheader, ForBody64Clone);
  ForBody64Clone->getTerminator()->setSuccessor(1, ForBody64Clone);

  ForCondCleanup63->moveAfter(ForBody64Clone);

  PHINode *OldAccLcssa = getFirstInst<PHINode>(ForCondCleanup63);
  PHINode *FinalAccLcssa = PHINode::Create(Type::getInt64Ty(F.getContext()), 2,
                                           "acc.3.lcssa", OldAccLcssa);
  FinalAccLcssa->addIncoming(PhiCloneMap[64], ForCond398Preheader);
  FinalAccLcssa->addIncoming(AccAddClone, ForBody64Clone);

  // Get the first PHINode and replace all uses
  OldAccLcssa->replaceAllUsesWith(FinalAccLcssa);
  ForCond58PreheaderLoopExit->eraseFromParent();

  transformUnrollGetElementPtr(ForBody46Merged);
  groupSameInstForFird(ForBody46Merged);
  transformUnrollGetElementPtr(ForBody64Merged);
  groupSameInstForFird(ForBody64Merged);
  runSimplifyDcePasses(F);
  runPostPass(F);
}

void DspiIntDotprodHandler::postUnrollDspiIntDotprod(Function &F, Loop *L,
                                                     unsigned Unroll_count,
                                                     LoopInfo &LI) {
  runSimplifyDcePasses(F);
  BasicBlock *ForCond25Preheader = L->getLoopPredecessor();
  auto [LoopHeaderClone, ForBodyMerged] = cloneAndMergeLoop(L, F, Unroll_count);
  LoopHeaderClone->moveAfter(ForBodyMerged);

  BasicBlock *ForCond25PreheaderLrPh = nullptr;
  for (BasicBlock *Pred : predecessors(ForCond25Preheader)) {
    if (Pred->getSingleSuccessor() == ForCond25Preheader) {
      ForCond25PreheaderLrPh = Pred;
      break;
    }
  }
  // Get the original comparison instruction
  ICmpInst *EntryCmp = getFirstInst<ICmpInst>(ForCond25PreheaderLrPh);

  // Get CountValue operand
  Value *CountValue = EntryCmp->getOperand(0);
  EntryCmp->setOperand(1, ConstantInt::get(CountValue->getType(), 7));
  // Create new instruction
  IRBuilder<> Builder(EntryCmp);
  Value *CountMinus7 = Builder.CreateNSWAdd(
      CountValue, ConstantInt::getSigned(CountValue->getType(), -7), "Sub1");
  Value *And_val = Builder.CreateAnd(
      CountValue, ConstantInt::getSigned(CountValue->getType(), -8));

  BasicBlock *ForCondCleanup27 = nullptr;
  for (BasicBlock *Succ : successors(ForCond25Preheader)) {
    if (Succ != ForBodyMerged) {
      ForCondCleanup27 = Succ;
      break;
    }
  }
  ForCondCleanup27->moveAfter(LoopHeaderClone);

  // Create new for.cond128.preheader basic block
  BasicBlock *ForCond128Preheader = BasicBlock::Create(
      F.getContext(), "for.cond128.preheader", &F, ForBodyMerged);
  // Create X.0.lcssa Phi node
  Builder.SetInsertPoint(ForCond128Preheader);
  // Create PHI node
  PHINode *IndexLcssa =
      Builder.CreatePHI(Type::getInt32Ty(F.getContext()), 2, "X.0.lcssa");
  IndexLcssa->addIncoming(ConstantInt::get(Type::getInt32Ty(F.getContext()), 0),
                          ForCond25Preheader);
  IndexLcssa->addIncoming(And_val, ForBodyMerged);

  // Create comparison instruction
  Value *RemainderLoopCondCmp =
      Builder.CreateICmpSLT(IndexLcssa, CountValue, "Cmp129268");

  // Create conditional branch instruction
  Builder.CreateCondBr(RemainderLoopCondCmp, LoopHeaderClone, ForCondCleanup27);

  ForCond25Preheader->getTerminator()->setSuccessor(1, ForCond128Preheader);

  ICmpInst *LoopExitCmp = getLastInst<ICmpInst>(ForBodyMerged);
  LoopExitCmp->setPredicate(ICmpInst::ICMP_SLT);
  LoopExitCmp->setOperand(1, CountMinus7);
  ForBodyMerged->getTerminator()->setSuccessor(0, ForBodyMerged);
  ForBodyMerged->getTerminator()->setSuccessor(1, ForCond128Preheader);

  handlePhiNodes(LoopHeaderClone, IndexLcssa);
  setPHINodesBlock(LoopHeaderClone, LoopHeaderClone, ForCond128Preheader);
  LoopHeaderClone->getTerminator()->setSuccessor(1, LoopHeaderClone);

  // Get different types of PHI nodes
  AccPhi = getAccPhi(ForBodyMerged);
  Acc_1_lcssa = getAccLcssaPhi(ForCondCleanup27);
  AccPhiClone = getAccPhi(LoopHeaderClone);

  PHINode *ClonedPhi2 = nullptr;
  if (AccPhi) {
    ClonedPhi2 = cast<PHINode>(AccPhi->clone());
    ClonedPhi2->insertBefore(ForCond128Preheader->getFirstNonPHI());
  }

  PHINode *ClonedPhi = nullptr;
  if (AccPhiClone) {
    AccPhiClone->setIncomingValue(1, ClonedPhi2);
    ClonedPhi = cast<PHINode>(AccPhiClone->clone());
    ClonedPhi->insertBefore(ForCondCleanup27->getFirstNonPHI());
  }
  Acc_1_lcssa->replaceAllUsesWith(ClonedPhi);
  Acc_1_lcssa->eraseFromParent();

  runSimplifyDcePasses(F);
  runPostPass(F);
}

void DspmInt16MultHandler::postUnrollDspmInt16Mult(Function &F, Loop *L,
                                                   int Unroll_count) {
  runSimplifyDcePasses(F);
  BasicBlock *ForBody4 = L->getLoopPredecessor();
  BasicBlock *ForCondCleanup8 = ForBody4->getTerminator()->getSuccessor(1);
  auto [ForBody113, ForBodyMerged] = cloneAndMergeLoop(L, F, Unroll_count);
  ForBody113->moveAfter(ForBodyMerged);
  ForCondCleanup8->moveAfter(ForBody113);

  ICmpInst *Exitcond_not_7 = getLastInst<ICmpInst>(ForBodyMerged);
  Value *N = Exitcond_not_7->getOperand(1);

  BasicBlock *ForCond1PreheaderLrPh =
      F.getEntryBlock().getTerminator()->getSuccessor(0);
  // Get Icmp instruction in ForCond1PreheaderLrPh
  ICmpInst *EntryCmp = nullptr;
  for (auto &I : *ForCond1PreheaderLrPh) {
    if (auto *Icmp = dyn_cast<ICmpInst>(&I)) {
      if (Icmp->getPredicate() == ICmpInst::ICMP_SGT &&
          Icmp->getOperand(0) == N) {
        EntryCmp = Icmp;
        break;
      }
    }
  }
  assert(EntryCmp && "EntryCmp not found");
  EntryCmp->setOperand(1, ConstantInt::get(N->getType(), 7));

  IRBuilder<> Builder(EntryCmp);
  Value *CountMinus7 =
      Builder.CreateNSWAdd(N, ConstantInt::getSigned(N->getType(), -7), "Sub6");
  Value *And_val = Builder.CreateAnd(N, ConstantInt::getSigned(N->getType(), -8));

  // Create new for.cond110.preheader basic block
  BasicBlock *ForCond110Preheader = BasicBlock::Create(
      F.getContext(), "for.cond110.preheader", &F, ForBodyMerged);

  // Clone PHI nodes in forBodyMerged to for.cond110.preheader
  Builder.SetInsertPoint(ForCond110Preheader);

  PHINode *S_0_lcssa = nullptr;
  PHINode *Acc_0_lcssa = nullptr;
  // Iterate over PHI nodes in forBodyMerged and clone
  for (auto &Phi : ForBodyMerged->phis()) {
    PHINode *NewPhi = cast<PHINode>(Phi.clone());
    NewPhi->insertInto(ForCond110Preheader, ForCond110Preheader->begin());
    if (Phi.getType()->isIntegerTy(32)) {
      NewPhi->setName("S.0.lcssa");
      NewPhi->setIncomingValue(0, And_val);
      S_0_lcssa = NewPhi;
      // Create comparison instruction
      Value *RemainderLoopCondCmp =
          Builder.CreateICmpSLT(NewPhi, N, "Cmp111262");
      // Create conditional branch instruction
      Builder.CreateCondBr(RemainderLoopCondCmp, ForBody113, ForCondCleanup8);
    } else if (Phi.getType()->isIntegerTy(64)) {
      NewPhi->setName("acc.0.lcssa");
      Acc_0_lcssa = NewPhi;
    } else {
      llvm_unreachable("Unsupported type");
    }
  }

  // Update terminator of predecessor basic block
  ForBody4->getTerminator()->setSuccessor(1, ForCond110Preheader);
  ForBodyMerged->getTerminator()->setSuccessor(0, ForCond110Preheader);

  Exitcond_not_7->setOperand(1, CountMinus7);
  Exitcond_not_7->setPredicate(ICmpInst::ICMP_SLT);
  swapTerminatorSuccessors(ForBodyMerged);

  PHINode *AccClone = nullptr;
  for (auto &Phi : ForBody113->phis()) {
    Phi.setIncomingBlock(0, ForBody113);
    Phi.setIncomingBlock(1, ForCond110Preheader);
    if (Phi.getType()->isIntegerTy(32)) {
      Phi.setIncomingValue(0, Phi.user_back());
      Phi.setIncomingValue(1, S_0_lcssa);
    } else if (Phi.getType()->isIntegerTy(64)) {
      Phi.setIncomingValue(0, Phi.user_back());
      Phi.setIncomingValue(1, Acc_0_lcssa);
      AccClone = cast<PHINode>(Phi.clone());
      //
      for (auto &Phi : ForCondCleanup8->phis()) {
        if (Phi.getType()->isIntegerTy(64)) {
          Phi.replaceAllUsesWith(AccClone);
          Phi.eraseFromParent();
          break;
        }
      }
      AccClone->insertInto(ForCondCleanup8, ForCondCleanup8->begin());
      AccClone->setName("acc.1.lcssa");
    } else {
      llvm_unreachable("Unsupported type");
    }
  }
  ForBody113->getTerminator()->setSuccessor(1, ForBody113);
}

static PreservedAnalyses runLoopUnrollPass(Function &F,
                                           FunctionAnalysisManager &AM) {
  auto &LI = AM.getResult<LoopAnalysis>(F);
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  // There are no loops in the function. Return before computing other expensive
  // analyses.
  if (LI.empty())
    return PreservedAnalyses::all();

  // Create processor
  auto Handler = UnrollHandlerFactory::createHandler(F, LI, DT);
  if (!Handler)
    return PreservedAnalyses::all();

  // Preprocess
  Handler->preTransform();
  // SE must be Analysis after preTransform because F is changed
  auto &SE = AM.getResult<ScalarEvolutionAnalysis>(F);
  auto &TTI = AM.getResult<TargetIRAnalysis>(F);
  auto &AC = AM.getResult<AssumptionAnalysis>(F);

  auto &ORE = AM.getResult<OptimizationRemarkEmitterAnalysis>(F);

  SmallVector<Loop *, 2> FIRDWillTransformLoops;
  LoopAnalysisManager *LAM = nullptr;
  if (auto *LAMProxy = AM.getCachedResult<LoopAnalysisManagerFunctionProxy>(F))
    LAM = &LAMProxy->getManager();

  bool Changed = false;

  // The unroller requires loops to be in simplified form, and also needs LCSSA.
  // Since simplification may add new inner loops, it has to run before the
  // legality and profitability checks. This means running the loop unroller
  // will simplify all loops, regardless of whether anything end up being
  // unrolled.
  for (const auto &L : LI) {
    Changed |=
        simplifyLoop(L, &DT, &LI, &SE, &AC, nullptr, false /* PreserveLCSSA */);
    Changed |= formLCSSARecursively(*L, DT, &LI, &SE);
  }

  // Add the loop nests in the reverse order of LoopInfo. See method
  // declaration.
  SmallPriorityWorklist<Loop *, 4> Worklist;
  appendLoopsToWorklist(LI, Worklist);

  while (!Worklist.empty()) {
    // Because the LoopInfo stores the loops in RPO, we walk the worklist
    // from back to front so that we work forward across the CFG, which
    // for unrolling is only needed to get optimization remarks emitted in
    // A forward order.
    Loop &L = *Worklist.pop_back_val();

    if (!Handler->shouldUnroll(L)) {
      continue;
    }

#ifndef NDEBUG
    Loop *ParentL = L.getParentLoop();
#endif

    // Check if the profile summary indicates that the profiled application
    // has A huge working set size, in which case we disable peeling to avoid
    // bloating it further.

    std::string LoopName = std::string(L.getName());
    // The API here is quite complex to call and we allow to select some
    // flavors of unrolling during construction time (by setting UnrollOpts).

    LoopUnrollResult Result = UnrollLoop(
        &L,
        {/*Count*/ Handler->getUnrollCount(), /*Force*/ true, /*Runtime*/ false,
         /*AllowExpensiveTripCount*/ true,
         /*UnrollRemainder*/ true, true},
        &LI, &SE, &DT, &AC, &TTI, /*ORE*/ &ORE, true);
    Changed |= Result != LoopUnrollResult::Unmodified;

    // The parent must not be damaged by unrolling!
#ifndef NDEBUG
    if (Result != LoopUnrollResult::Unmodified && ParentL)
      ParentL->verifyLoop();
#endif

    // Clear any cached analysis results for L if we removed it completely.
    if (LAM && Result == LoopUnrollResult::FullyUnrolled)
      LAM->clear(L, LoopName);

    if (Result != LoopUnrollResult::Unmodified) {
      Handler->postTransform(L);
    }
  }

  // Final post-processing
  Handler->postUnroll();
  if (!Changed)
    return PreservedAnalyses::all();
  return getLoopPassPreservedAnalyses();
}

PreservedAnalyses
RISCVIntLoopUnrollAndRemainderPass::run(Function &F,
                                        FunctionAnalysisManager &AM) {
  if (!EnableRISCVIntLoopUnrollAndRemainder)
    return PreservedAnalyses::all();

  // Force recomputation of LoopInfo to avoid using stale analysis results
  AM.invalidate(F, PreservedAnalyses::none());
  if (verifyFunction(F, &errs())) {
    LLVM_DEBUG(dbgs() << "Function verification failed before "
                         "RISCVIntLoopUnrollAndRemainderPass\n");
    return PreservedAnalyses::all();
  }

  addnoalias(F);
  PreservedAnalyses PA = runLoopUnrollPass(F, AM);

  return PreservedAnalyses::all();
}
