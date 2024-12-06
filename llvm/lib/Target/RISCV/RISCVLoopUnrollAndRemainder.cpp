//===-- RISCVLoopUnrollAndRemainder.cpp - Loop Unrolling Pass
//------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a loop unrolling optimization pass specifically designed
// for Digital Signal Processing (DSP) algorithms. The pass targets common
// computational patterns found in various DSP operations including:
// - FIR and IIR filters
// - Convolution and correlation
// - Vector operations
// - Dot product calculations
// - Mathematical functions
//
// The pass performs the following main operations:
// 1. Identifies loops in DSP algorithm implementations
// 2. Unrolls the main computational loops, typically by a factor of 8
// 3. Efficiently handles remainder iterations
// 4. Optimizes memory access patterns for improved cache utilization
// 5. Adjusts control flow and PHI nodes to support the unrolled structure
// 6. Performs cleanup and further optimization after unrolling
//
// This transformation can significantly improve performance for DSP algorithms
// by:
// - Increasing instruction-level parallelism
// - Improving cache utilization for data and coefficient access
// - Reducing loop overhead
// - Enabling better vectorization opportunities
//
// The pass is particularly effective for algorithms with intensive loop-based
// computations, where the main computational loop dominates the execution time.
// It aims to optimize both the main loop body and the handling of edge cases,
// providing a balance between performance and code size.
//
//===----------------------------------------------------------------------===//
#include "RISCVLoopUnrollAndRemainder.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/CodeMetrics.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/LoopUnrollAnalyzer.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Transforms/Scalar/DeadStoreElimination.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/LoopStrengthReduce.h"
#include "llvm/Transforms/Scalar/Reassociate.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LoopPeel.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/SimplifyCFGOptions.h"
#include "llvm/Transforms/Utils/UnrollLoop.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

using namespace llvm;

#define DEBUG_TYPE "riscv-loop-unroll-and-remainder"

// Enumeration to represent different types of unrolling
enum class UnrollType {
  DOTPROD,
  ADD_ADDC_SUB_MUL_MULC_SQRT,
  CONV_CCORR,
  FIRD,
  FIR,
  CORR,
  UNKNOWN
};

// Global variable to store the current unroll type
static UnrollType currentUnrollType = UnrollType::UNKNOWN;

// Command line option to enable the RISCVLoopUnrollAndRemainder pass
cl::opt<bool> llvm::EnableRISCVLoopUnrollAndRemainder(
    "riscv-loop-unroll-and-remainder", cl::init(false),
    cl::desc("Enable loop unrolling and remainder specific loop"));

// Helper function to get a basic block by name from a function
static BasicBlock *getBasicBlockByName(Function &F, StringRef Name) {
  for (BasicBlock &BB : F)
    if (BB.getName() == Name)
      return &BB;
  return nullptr;
}

// Helper function to get the first ICmp instruction with a specific predicate
// in a basic block
static ICmpInst *getFirstICmpInstWithPredicate(BasicBlock *BB,
                                               ICmpInst::Predicate Predicate) {
  for (Instruction &I : *BB) {
    if (auto *CI = dyn_cast<ICmpInst>(&I)) {
      if (CI->getPredicate() == Predicate) {
        return CI;
      }
    }
  }
  return nullptr;
}

// Helper function to get the last ICmp instruction with a specific predicate in
// a basic block
static ICmpInst *getLastICmpInstWithPredicate(BasicBlock *BB,
                                              ICmpInst::Predicate Predicate) {
  ICmpInst *lastICmp = nullptr;
  for (Instruction &I : *BB) {
    if (auto *CI = dyn_cast<ICmpInst>(&I)) {
      if (CI->getPredicate() == Predicate) {
        lastICmp = CI;
      }
    }
  }
  return lastICmp;
}

template <typename T> static T *getFirstInst(BasicBlock *BB) {
  for (Instruction &I : *BB) {
    if (T *Inst = dyn_cast<T>(&I)) {
      return Inst;
    }
  }
  return nullptr;
}

template <typename T> static T *getLastInst(BasicBlock *BB) {
  for (Instruction &I : reverse(*BB)) {
    if (T *Inst = dyn_cast<T>(&I)) {
      return Inst;
    }
  }
  return nullptr;
}

// Helper function to get the first float PHI node in a basic block
static PHINode *getFirstFloatPhi(BasicBlock *BB) {
  for (auto &Inst : *BB) {
    if (auto *Phi = dyn_cast<PHINode>(&Inst)) {
      if (Phi->getType()->isFloatTy()) {
        return Phi;
      }
    }
  }
  return nullptr;
}

// Helper function to get the last float PHI node in a basic block
static PHINode *getLastFloatPhi(BasicBlock *BB) {
  for (auto it = BB->rbegin(); it != BB->rend(); ++it) {
    if (auto *Phi = dyn_cast<PHINode>(&*it)) {
      if (Phi->getType()->isFloatTy()) {
        return Phi;
      }
    }
  }
  return nullptr;
}

// Helper function to get the first 32-bit integer PHI node in a basic block
static PHINode *getFirstI32Phi(BasicBlock *BB) {
  for (auto &Inst : *BB) {
    if (auto *Phi = dyn_cast<PHINode>(&Inst)) {
      if (Phi->getType()->isIntegerTy(32)) {
        return Phi;
      }
    }
  }
  return nullptr;
}

// Helper function to get the last 32-bit integer PHI node in a basic block
static PHINode *getLastI32Phi(BasicBlock *BB) {
  for (auto it = BB->rbegin(); it != BB->rend(); ++it) {
    if (auto *Phi = dyn_cast<PHINode>(&*it)) {
      if (Phi->getType()->isIntegerTy(32)) {
        return Phi;
      }
    }
  }
  return nullptr;
}

// Helper function to get the first CallInst with a specific name in a basic
// block
static CallInst *getFirstCallInstWithName(BasicBlock *BB, StringRef Name) {
  for (Instruction &I : *BB) {
    if (auto *Call = dyn_cast<CallInst>(&I)) {
      if (Call->getCalledFunction() &&
          Call->getCalledFunction()->getName() == Name) {
        return Call;
      }
    }
  }
  return nullptr;
}

// Helper function to update operands of new instructions
static void updateOperands(SmallVector<Instruction *, 8> &NewInsts,
                           ValueToValueMapTy &ValueMap) {
  for (Instruction *inst : NewInsts) {
    for (unsigned i = 0; i < inst->getNumOperands(); i++) {
      Value *op = inst->getOperand(i);
      if (ValueMap.count(op)) {
        inst->setOperand(i, ValueMap[op]);
      }
    }
  }
}

// Helper function to swap the successors of a terminator instruction
static void swapTerminatorSuccessors(BasicBlock *BB) {
  if (auto *BI = dyn_cast<BranchInst>(BB->getTerminator())) {
    if (BI->isConditional() && BI->getNumSuccessors() == 2) {
      BasicBlock *TrueSuccessor = BI->getSuccessor(0);
      BasicBlock *FalseSuccessor = BI->getSuccessor(1);
      BI->setSuccessor(0, FalseSuccessor);
      BI->setSuccessor(1, TrueSuccessor);
    } else {
      llvm_unreachable("BB's terminator is not a conditional branch or doesn't "
                       "have two successors");
    }
  } else {
    llvm_unreachable("BB's terminator is not a branch instruction");
  }
}

// Helper function to clone a basic block and update its relations
static BasicBlock *cloneBasicBlockWithRelations(BasicBlock *BB,
                                                const std::string &NameSuffix,
                                                Function *F) {
  ValueToValueMapTy VMap;
  BasicBlock *NewBB = CloneBasicBlock(BB, VMap, NameSuffix, F);

  // Update instruction references in the new block
  for (Instruction &I : *NewBB) {
    // Update operands
    for (Use &U : I.operands()) {
      Value *V = U.get();
      Value *NewV = VMap[V];
      if (NewV) {
        U.set(NewV);
      }
    }

    // Update PHI node basic block references
    if (PHINode *PN = dyn_cast<PHINode>(&I)) {
      for (unsigned i = 0, e = PN->getNumIncomingValues(); i != e; ++i) {
        BasicBlock *IncomingBB = PN->getIncomingBlock(i);
        if (IncomingBB == BB) {
          PN->setIncomingBlock(i, NewBB);
        } else if (VMap.count(IncomingBB)) {
          PN->setIncomingBlock(i, cast<BasicBlock>(VMap[IncomingBB]));
        }
      }
    }
  }

  return NewBB;
}

// Helper function to unroll and duplicate a loop iteration
static Instruction *unrollAndDuplicateLoopIteration(LLVMContext &Ctx,
                                                    BasicBlock *BB,
                                                    IRBuilder<> &Builder,
                                                    unsigned int i) {
  PHINode *IPhi = dyn_cast<PHINode>(&BB->front());
  BasicBlock::iterator BeginIt, EndIt, ToIt;
  SmallVector<Instruction *, 8> newInsts;
  ValueToValueMapTy ValueMap;
  Instruction *Add = nullptr;
  Instruction *tailcallfmuladd = nullptr;
  Instruction *duplicatedPhiNode = nullptr;

  // Find the range of instructions to duplicate
  for (Instruction &I : *BB) {
    if (auto *phi = dyn_cast<PHINode>(&I)) {
      if (phi->getType()->isFloatTy()) {
        BeginIt = I.getIterator();
      }
    } else if (RecurrenceDescriptor::isFMulAddIntrinsic(&I)) {
      EndIt = std::next(I.getIterator());
      tailcallfmuladd = &I;
      ToIt = std::next(EndIt);
      break;
    }
  }

  assert(&*BeginIt && &*EndIt && "Failed to find instruction range");

  // Clone and modify instructions
  int arrayidx = 0;
  for (auto it = BeginIt; it != EndIt; ++it) {
    Instruction *newInst = it->clone();
    if (newInst->getOpcode() == Instruction::PHI)
      newInst->setName("acc" + Twine(i));

    if (auto *GEP = dyn_cast<GetElementPtrInst>(newInst)) {
      if (!Add)
        Add = BinaryOperator::CreateDisjoint(
            Instruction::Or, IPhi, ConstantInt::get(Type::getInt32Ty(Ctx), i),
            "add" + Twine(i), BB);

      newInst->setName("arrayidx" + Twine(i) + "_" + Twine(arrayidx));
      newInst->setOperand(1, Add);
      arrayidx++;
    }
    newInsts.push_back(newInst);
    ValueMap[&*it] = newInst;
  }

  // Update operands and insert new instructions
  updateOperands(newInsts, ValueMap);
  for (Instruction *newInst : newInsts) {
    if (newInst->getOpcode() == Instruction::PHI)
      duplicatedPhiNode = newInst->clone();
    newInst->insertInto(BB, BB->end());
  }

  return duplicatedPhiNode;
}

// Helper function to move PHI nodes to the top of a basic block
static void movePHINodesToTop(BasicBlock &BB,
                              BasicBlock *ForBodyPreheaderBB = nullptr) {
  SmallVector<PHINode *, 8> PHIs;
  for (Instruction &I : BB) {
    if (PHINode *PHI = dyn_cast<PHINode>(&I)) {
      if (ForBodyPreheaderBB)
        PHI->setIncomingBlock(1, ForBodyPreheaderBB);
      PHIs.push_back(PHI);
    }
  }

  // Move PHI nodes in reverse order
  for (auto it = PHIs.rbegin(); it != PHIs.rend(); ++it) {
    (*it)->moveBefore(&BB.front());
  }
}

static void modifyFirdAddToOr(BasicBlock *ClonedForBody) {
  SmallVector<BinaryOperator *> addInsts;

  // Collect all add instructions that meet the criteria
  for (auto &I : *ClonedForBody) {
    if (auto *binOp = dyn_cast<BinaryOperator>(&I)) {
      if (binOp->getOpcode() == Instruction::Add && binOp->hasNoSignedWrap() &&
          binOp->hasNoUnsignedWrap()) {
        addInsts.push_back(binOp);
      }
    }
  }
  if (addInsts.empty()) {
    return;
  }
  // Replace each add instruction with an or disjoint instruction
  for (auto it = addInsts.begin(); it != std::prev(addInsts.end()); ++it) {
    auto *addInst = *it;
    // Create a new or disjoint instruction
    Instruction *orInst =
        BinaryOperator::CreateDisjoint(Instruction::Or, addInst->getOperand(0),
                                       addInst->getOperand(1), "add", addInst);

    // Replace all uses of the add instruction
    addInst->replaceAllUsesWith(orInst);

    // Delete the original add instruction
    addInst->eraseFromParent();
    orInst->setName("add");
  }
}

// Helper function to update predecessors to point to a new preheader
static void updatePredecessorsToPreheader(BasicBlock *ForBody,
                                          BasicBlock *ForBodyPreheader) {
  SmallVector<BasicBlock *, 4> predecessors_bb;
  for (auto *Pred : predecessors(ForBody)) {
    if (Pred != ForBody)
      predecessors_bb.push_back(Pred);
  }

  for (BasicBlock *Pred : predecessors_bb) {
    Instruction *TI = Pred->getTerminator();
    for (unsigned i = 0; i < TI->getNumSuccessors(); ++i) {
      if (TI->getSuccessor(i) == ForBody) {
        TI->setSuccessor(i, ForBodyPreheader);
      }
    }
  }

  if (!ForBodyPreheader->getTerminator()) {
    BranchInst::Create(ForBody, ForBodyPreheader);
  }
}

// Helper function to get the 'len' value from the entry block
static Value *getLenFromEntryBlock(Function &F) {
  ICmpInst *ICmp = nullptr;
  for (BasicBlock &BB : F) {
    ICmp = getFirstICmpInstWithPredicate(&BB, ICmpInst::ICMP_SGT);
    if (ICmp)
      break;
  }

  assert(ICmp && "icmp sgt instruction not found");
  return ICmp->getOperand(0);
}

// Helper function to find specific instructions in a basic block
static std::tuple<PHINode *, CallInst *, BinaryOperator *>
findKeyInstructions(BasicBlock *ForBody) {
  PHINode *ThirdPHI = nullptr;
  CallInst *callInst = nullptr;
  BinaryOperator *addInst = nullptr;
  int PHICount = 0;

  for (Instruction &I : *ForBody) {
    if (auto *PHI = dyn_cast<PHINode>(&I)) {
      PHICount++;
      if (PHICount == 3) {
        ThirdPHI = PHI;
      }
    } else if (auto *ci = dyn_cast<CallInst>(&I)) {
      callInst = ci;
    } else if (auto *BinOp = dyn_cast<BinaryOperator>(&I)) {
      if (BinOp->getOpcode() == Instruction::Add) {
        addInst = BinOp;
      }
    }
  }

  return std::make_tuple(ThirdPHI, callInst, addInst);
}

// Helper function to rename instructions
static void renameInstruction(Instruction *inst) {
  if (inst->getOpcode() == Instruction::PHI) {
    inst->setName("acc");
  } else if (inst->getOpcode() == Instruction::GetElementPtr) {
    inst->setName("arrayidx");
  }
}

// Helper function to set add instruction in for body
static void setAddInForBody(Instruction *inst, Instruction *Add,
                            Instruction *InsertBefore) {
  if (inst->getOpcode() == Instruction::PHI) {
    Add->moveBefore(InsertBefore);
  } else if (inst->getOpcode() == Instruction::GetElementPtr) {
    inst->setOperand(1, Add);
  }
}

// Helper function to copy and remap instructions
static void copyAndRemapInstructions(Instruction *StartInst,
                                     Instruction *EndInst,
                                     Instruction *InsertBefore,
                                     Instruction *Add) {
  ValueToValueMapTy ValueMap;
  SmallVector<Instruction *, 8> NewInsts;

  for (auto it = StartInst->getIterator(); &*it != EndInst; ++it) {
    Instruction *newInst = it->clone();
    if (auto *BinOp = dyn_cast<BinaryOperator>(newInst)) {
      if (BinOp->getOpcode() == Instruction::Add) {
        continue;
      }
    }
    NewInsts.push_back(newInst);
    ValueMap[&*it] = newInst;
  }

  updateOperands(NewInsts, ValueMap);

  for (Instruction *newInst : NewInsts) {
    renameInstruction(newInst);
    newInst->insertBefore(InsertBefore);
    setAddInForBody(newInst, Add, InsertBefore);
  }
}

// Helper function to preprocess the cloned for body
static void preProcessClonedForBody(BasicBlock *ClonedForBody, Value *sub) {
  Instruction *addInst = nullptr;
  for (Instruction &I : *ClonedForBody) {
    if (auto *BinOp = dyn_cast<BinaryOperator>(&I)) {
      if (BinOp->getOpcode() == Instruction::Add) {
        BinOp->setOperand(1, ConstantInt::get(BinOp->getType(), 8));
        addInst = BinOp;
      }
    }
    if (auto *icmp = dyn_cast<ICmpInst>(&I)) {
      icmp->setPredicate(CmpInst::Predicate::ICMP_SLT);
      icmp->setOperand(0, addInst);
      icmp->setOperand(1, sub);
      icmp->setName("cmp11");
    }
  }
  LLVM_DEBUG(ClonedForBody->dump());
}

// Helper function to modify getelementptr instructions
static void modifyGetElementPtr(BasicBlock *BB) {
  SmallVector<GetElementPtrInst *, 8> gepInsts;
  Value *firstGEPOperand0 = nullptr;
  Value *secondGEPOperand1 = nullptr;

  for (Instruction &I : *BB) {
    if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
      gepInsts.push_back(GEP);
    }
  }

  if (gepInsts.size() < 8 || gepInsts.size() % 2 != 0) {
    return;
  }

  firstGEPOperand0 = gepInsts[0];
  secondGEPOperand1 = gepInsts[1];

  for (size_t i = 2; i < gepInsts.size(); ++i) {
    if (i % 2 == 0) {
      if (i < gepInsts.size() - 2) {
        gepInsts[i]->setOperand(0, firstGEPOperand0);
      }
    } else {
      gepInsts[i]->setOperand(0, secondGEPOperand1);
    }

    if (i == 14)
      continue;

    Instruction *operand1 = dyn_cast<Instruction>(gepInsts[i]->getOperand(1));
    gepInsts[i]->setOperand(
        1, ConstantInt::get(Type::getInt32Ty(BB->getContext()), i / 2));
    if (operand1 && operand1->use_empty()) {
      operand1->eraseFromParent();
    }
  }
}

// Helper function to check if a PHI node has an incoming value of zero
static bool isIncomingValueZeroOfPhi(PHINode *phi) {
  return phi->getType()->isIntegerTy(32) &&
         isa<ConstantInt>(phi->getIncomingValue(0)) &&
         cast<ConstantInt>(phi->getIncomingValue(0))->isZero();
}

// Helper function to find and set add instructions
static std::pair<Instruction *, Instruction *>
findAndSetAddInstructions(BasicBlock *ClonedForBody) {
  Instruction *FirstAdd = nullptr;
  Instruction *SecondAdd = nullptr;

  for (Instruction &I : *ClonedForBody) {
    if (BinaryOperator *BinOp = dyn_cast<BinaryOperator>(&I)) {
      if (BinOp->getOpcode() == Instruction::Add) {
        if (!FirstAdd) {
          FirstAdd = &I;
          FirstAdd->setHasNoSignedWrap(true);
        } else if (!SecondAdd) {
          SecondAdd = &I;
          break;
        }
      }
    }
  }
  assert(FirstAdd && SecondAdd && "Failed to find matching add instructions");
  return std::make_pair(FirstAdd, SecondAdd);
}

// Helper functions for PHI node manipulation

static PHINode *findZeroInitializedPHI(BasicBlock *block) {
  for (Instruction &I : *block) {
    if (PHINode *phi = dyn_cast<PHINode>(&I)) {
      if (isIncomingValueZeroOfPhi(phi)) {
        return phi;
      }
    }
  }
  return nullptr;
}

static PHINode *findIntegerPHI(BasicBlock *block) {
  for (Instruction &I : *block) {
    if (PHINode *phi = dyn_cast<PHINode>(&I)) {
      if (phi->getType()->isIntegerTy(32) && !isIncomingValueZeroOfPhi(phi)) {
        return phi;
      }
    }
  }
  return nullptr;
}

// Helper function to unroll loop body
static void unrollLoopBody(BasicBlock *block, PHINode *thirdPHI,
                           Instruction *callInst, Instruction *addInst,
                           PHINode *zeroInitializedPHI, LLVMContext &context) {
  for (int i = 1; i < 8; i++) {
    Instruction *add = BinaryOperator::CreateDisjoint(
        Instruction::Or, zeroInitializedPHI,
        ConstantInt::get(Type::getInt32Ty(context), i), "add" + Twine(i),
        block);
    copyAndRemapInstructions(thirdPHI, callInst->getNextNode(), addInst, add);
  }
}

// Helper function to update add instruction
static void updateAddInstruction(Instruction *addInst, PHINode *integerPHI,
                                 LLVMContext &context) {
  if (addInst) {
    addInst->setOperand(1, ConstantInt::get(Type::getInt32Ty(context), 8));
    addInst->setOperand(0, integerPHI);
  }
}

// Helper function to update block terminator
static void updateBlockTerminator(BasicBlock *block, BasicBlock *successor) {
  Instruction *terminator = block->getTerminator();
  terminator->setSuccessor(0, block);
  terminator->setSuccessor(1, successor);
}

// Helper function to modify getelementptr for unrolling
static void modifyGetElementPtrForUnrolling(BasicBlock *block) {
  SmallVector<GetElementPtrInst *, 8> gepInsts;
  for (Instruction &I : *block) {
    if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
      gepInsts.push_back(GEP);
    }
  }

  for (size_t i = 2; i < gepInsts.size(); i += 2) {
    gepInsts[i]->setOperand(0, gepInsts[0]);
    gepInsts[i]->setOperand(
        1, ConstantInt::get(Type::getInt32Ty(block->getContext()), i / 2));
  }
}

// Helper function to handle add instructions
static void handleAddInstructions(BasicBlock *block, unsigned int unrollFactor,
                                  PHINode *zeroInitializedPHI,
                                  LLVMContext &context) {
  auto [firstAdd, secondAdd] = findAndSetAddInstructions(block);

  if (firstAdd && secondAdd) {
    firstAdd->moveBefore(secondAdd);

    if (unrollFactor == 1) {
      firstAdd->setOperand(1, ConstantInt::get(Type::getInt32Ty(context), 8));
      secondAdd->setOperand(0, zeroInitializedPHI);
    }
  }
}

// Function to unroll the cloned for loop body
static void unrollClonedForBody(BasicBlock *clonedForBody,
                                BasicBlock *forCondPreheader,
                                unsigned int unrollFactor = 0) {
  Function *function = clonedForBody->getParent();
  LLVMContext &context = function->getContext();

  // Find key instructions in the cloned for body
  auto [thirdPHI, callInst, addInst] = findKeyInstructions(clonedForBody);
  PHINode *zeroInitializedPHI = findZeroInitializedPHI(clonedForBody);
  PHINode *integerPHI = findIntegerPHI(clonedForBody);

  assert(zeroInitializedPHI && "No matching zero-initialized PHI node found");

  // Unroll the loop body if key instructions are found
  if (thirdPHI && callInst) {
    unrollLoopBody(clonedForBody, thirdPHI, callInst, addInst,
                   zeroInitializedPHI, context);
  }

  // Update the add instruction
  updateAddInstruction(addInst, integerPHI, context);

  // Update the basic block terminator
  updateBlockTerminator(clonedForBody, forCondPreheader);

  // Move PHI nodes to the top of the basic block
  movePHINodesToTop(*clonedForBody);

  // Modify getelementptr instructions based on the unroll factor
  if (unrollFactor == 0) {
    modifyGetElementPtr(clonedForBody);
  } else {
    modifyGetElementPtrForUnrolling(clonedForBody);
  }

  // Handle add instructions
  handleAddInstructions(clonedForBody, unrollFactor, zeroInitializedPHI,
                        context);
}

// Function to check if a call instruction can be moved
static bool canMoveCallInstruction(CallInst *callInst,
                                   Instruction *insertPoint) {
  for (unsigned i = 0; i < callInst->getNumOperands(); ++i) {
    if (auto *operandInst = dyn_cast<Instruction>(callInst->getOperand(i))) {
      if (operandInst->getParent() == callInst->getParent() &&
          insertPoint->comesBefore(operandInst)) {
        return false;
      }
    }
  }
  return true;
}

// Function to group and reorder instructions in a basic block
static void groupAndReorderInstructions(BasicBlock *clonedForBody) {
  // Collect different types of instructions
  SmallVector<PHINode *> phiNodes;
  SmallVector<Instruction *> orInsts, gepInsts, loadInsts, storeInsts, mulInsts,
      addInsts, subInsts, callInsts, ashrInsts, faddInsts, fmulInsts, fsubInsts;

  // Categorize instructions by type
  for (Instruction &I : *clonedForBody) {
    if (auto *phi = dyn_cast<PHINode>(&I)) {
      phiNodes.push_back(phi);
    } else if (I.getOpcode() == Instruction::Or) {
      orInsts.push_back(&I);
    } else if (isa<GetElementPtrInst>(&I)) {
      gepInsts.push_back(&I);
    } else if (isa<LoadInst>(&I)) {
      loadInsts.push_back(&I);
    } else if (isa<StoreInst>(&I)) {
      storeInsts.push_back(&I);
    } else if (I.getOpcode() == Instruction::Mul) {
      mulInsts.push_back(&I);
    } else if (isa<CallInst>(&I)) {
      callInsts.push_back(&I);
    } else if (I.getOpcode() == Instruction::Add) {
      addInsts.push_back(&I);
    } else if (I.getOpcode() == Instruction::Sub) {
      subInsts.push_back(&I);
    } else if (I.getOpcode() == Instruction::FAdd) {
      faddInsts.push_back(&I);
    } else if (I.getOpcode() == Instruction::FMul) {
      fmulInsts.push_back(&I);
    } else if (I.getOpcode() == Instruction::FSub) {
      fsubInsts.push_back(&I);
    } else if (I.getOpcode() == Instruction::AShr) {
      return;
    }
  }

  // If no PHI nodes are found, return
  if (phiNodes.empty()) {
    return;
  }

  // Reorder instructions
  Instruction *insertPoint = phiNodes.back()->getNextNode();
  bool canMoveCallInst =
      callInsts.empty() ||
      canMoveCallInstruction(dyn_cast<CallInst>(callInsts[0]), insertPoint);

  auto moveInstructions = [&insertPoint](SmallVector<Instruction *> &insts) {
    for (auto *inst : insts) {
      inst->moveBefore(insertPoint);
      insertPoint = inst->getNextNode();
    }
  };

  // Move instructions in the desired order
  moveInstructions(mulInsts);
  moveInstructions(addInsts);
  moveInstructions(orInsts);
  moveInstructions(subInsts);
  moveInstructions(gepInsts);
  moveInstructions(loadInsts);
  moveInstructions(faddInsts);
  moveInstructions(fmulInsts);
  moveInstructions(fsubInsts);
  if (canMoveCallInst) {
    moveInstructions(callInsts);
  }
}

// Function to transform a single loop depth (currently suitable for
// dotprod/dotprode example)
static bool transformOneLoopDepth(Function &F) {
  LLVMContext &ctx = F.getContext();
  bool changed = false;

  // Get necessary basic blocks and values
  Value *len = getLenFromEntryBlock(F);
  BasicBlock *entryBB = &F.getEntryBlock();
  BasicBlock *forBodyBB = getBasicBlockByName(F, "for.body");
  BasicBlock *forBodyNewBB = getBasicBlockByName(F, "for.body.clone");
  BasicBlock *ifEnd = getBasicBlockByName(F, "if.end");
  BasicBlock *forCond46PreheaderBB =
      getBasicBlockByName(F, "for.cond.preheader");

  assert(forBodyBB && "Expected to find for.body!");
  assert(forBodyNewBB && "Expected to find for.body.clone!");
  assert(ifEnd && "Expected to find if.end!");
  assert(forCond46PreheaderBB && "Expected to find for.cond.preheader!");

  // Create new basic blocks
  BasicBlock *forCondPreheaderBB =
      BasicBlock::Create(F.getContext(), "for.cond.preheader", &F, forBodyBB);
  BasicBlock *forBodyPreheaderBB =
      BasicBlock::Create(F.getContext(), "for.body.preheader", &F, forBodyBB);
  BasicBlock *forCond31PreheaderBB =
      BasicBlock::Create(F.getContext(), "for.cond31.preheader", &F, forBodyBB);
  BasicBlock *forBody33BB = cloneBasicBlockWithRelations(forBodyBB, "33", &F);
  forBody33BB->setName("for.body33");
  forBody33BB->moveAfter(forBodyBB);
  BasicBlock *forEnd37BB =
      BasicBlock::Create(F.getContext(), "for.end37", &F, forBodyNewBB);

  // Add instructions to forCondPreheaderBB
  IRBuilder<> builder(forCondPreheaderBB);
  Value *negativeSeven = ConstantInt::get(Type::getInt32Ty(F.getContext()), -7);
  Value *sub = builder.CreateNSWAdd(len, negativeSeven, "sub");
  Value *seven = ConstantInt::get(Type::getInt32Ty(F.getContext()), 7);
  Value *cmp1113 = builder.CreateICmpUGT(len, seven, "cmp1113");
  builder.CreateCondBr(cmp1113, forBodyPreheaderBB, forCond31PreheaderBB);

  // Add instructions to forBodyPreheaderBB
  builder.SetInsertPoint(forBodyPreheaderBB);
  Value *mask = ConstantInt::get(Type::getInt32Ty(F.getContext()), 2147483640);
  Value *andValue = builder.CreateAnd(len, mask, "");
  builder.CreateBr(forBodyBB);

  // Modify for.body
  PHINode *iPhi = dyn_cast<PHINode>(&forBodyBB->front());
  iPhi->setName("i.0122");

  // copy first float phinode from forBodyBB to forCond31PreheaderBB
  PHINode *firstFloatPhi = getFirstFloatPhi(forBodyBB);
  PHINode *acc00Lcssa = PHINode::Create(firstFloatPhi->getType(), 2,
                                        "acc0.0.lcssa", forCond31PreheaderBB);
  acc00Lcssa->addIncoming(firstFloatPhi->getIncomingValue(0),
                          firstFloatPhi->getIncomingBlock(0));
  acc00Lcssa->addIncoming(firstFloatPhi->getIncomingValue(1),
                          forCondPreheaderBB);
  // Unroll and duplicate loop iterations
  SmallVector<Instruction *> instructions;
  for (int i = 0; i < 7; i++) {
    Instruction *copyedPhiNode =
        unrollAndDuplicateLoopIteration(ctx, forBodyBB, builder, i + 1);
    if (PHINode *phi = dyn_cast<PHINode>(copyedPhiNode)) {
      phi->setName("acc" + Twine(i + 1) + ".0.lcssa");
      phi->setIncomingBlock(1, forCondPreheaderBB);
      phi->insertInto(forCond31PreheaderBB, forCond31PreheaderBB->end());
      instructions.push_back(phi);
    }
  }

  // Update for.body terminator
  Instruction *incInst = nullptr;
  MDNode *loopMD = nullptr;
  for (auto &I : *forBodyBB) {
    if (I.getOpcode() == Instruction::Add) {
      incInst = &I;
      Instruction *icmp = I.getNextNode();
      Instruction *br = icmp->getNextNode();
      assert(icmp->getOpcode() == Instruction::ICmp &&
             br->getOpcode() == Instruction::Br &&
             "Unexpected instruction sequence");
      I.moveAfter(&forBodyBB->back());
      loopMD = br->getMetadata(LLVMContext::MD_loop);
      br->eraseFromParent();
      icmp->eraseFromParent();
      break;
    }
  }

  // Modify add instruction
  incInst->setOperand(1, ConstantInt::get(Type::getInt32Ty(F.getContext()), 8));
  incInst->setName("add30");

  builder.SetInsertPoint(forBodyBB);
  Value *cmp1 = builder.CreateICmpSLT(incInst, sub, "cmp1");
  BranchInst *newBr =
      builder.CreateCondBr(cmp1, forBodyBB, forCond31PreheaderBB);
  newBr->setMetadata(LLVMContext::MD_loop, loopMD);

  movePHINodesToTop(*forBodyBB, forBodyPreheaderBB);

  // Add instructions to forCond31PreheaderBB
  builder.SetInsertPoint(forCond31PreheaderBB);
  PHINode *i0Lcssa =
      builder.CreatePHI(Type::getInt32Ty(F.getContext()), 0, "i.0.lcssa");
  i0Lcssa->addIncoming(ConstantInt::get(Type::getInt32Ty(F.getContext()), 0),
                       forCondPreheaderBB);
  i0Lcssa->addIncoming(andValue, forBodyBB);
  Value *cmp32132 = builder.CreateICmpSLT(i0Lcssa, len, "cmp32132");
  builder.CreateCondBr(cmp32132, forBody33BB, forEnd37BB);

  // Modify forBody33BB
  Instruction *tempInstr = nullptr;
  for (auto &I : *forBody33BB) {
    if (PHINode *phi = dyn_cast<PHINode>(&I)) {
      if (phi->getType()->isIntegerTy(32)) {
        phi->setIncomingValue(1, i0Lcssa);
        phi->setIncomingBlock(1, forCond31PreheaderBB);
      } else if (phi->getType()->isFloatTy()) {
        phi->setIncomingValue(1, acc00Lcssa);
        phi->setIncomingBlock(1, forCond31PreheaderBB);
        tempInstr = phi;
      }
    }
  }

  // Modify forEnd37BB
  Instruction *acc01Lcssa = tempInstr->clone();
  acc01Lcssa->setName("acc0.1.lcssa");
  acc01Lcssa->insertInto(forEnd37BB, forEnd37BB->end());
  builder.SetInsertPoint(forEnd37BB);

  // Create pairs of floating-point additions
  Value *sum01 = builder.CreateFAdd(acc01Lcssa, instructions[0], "sum01");
  Value *sum23 = builder.CreateFAdd(instructions[1], instructions[2], "sum23");
  Value *sum45 = builder.CreateFAdd(instructions[3], instructions[4], "sum45");
  Value *sum67 = builder.CreateFAdd(instructions[5], instructions[6], "sum67");

  // Combine pairs
  Value *sum0123 = builder.CreateFAdd(sum01, sum23, "sum0123");
  Value *sum4567 = builder.CreateFAdd(sum45, sum67, "sum4567");

  // Final addition
  Value *currentAdd = builder.CreateFAdd(sum0123, sum4567, "add44");
  builder.CreateBr(ifEnd);

  // Modify entry basic block
  BranchInst *entryBi = dyn_cast<BranchInst>(entryBB->getTerminator());
  entryBi->setSuccessor(0, forCondPreheaderBB);
  entryBi->setSuccessor(1, forCond46PreheaderBB);

  // Modify forCond46PreheaderBB
  forCond46PreheaderBB->getTerminator()->getPrevNode()->setName("cmp47110");

  // Modify for.body33
  BranchInst *forBody33Bi = dyn_cast<BranchInst>(forBody33BB->getTerminator());
  forBody33Bi->setSuccessor(0, forEnd37BB);
  forBody33Bi->setSuccessor(1, forBody33BB);

  // Modify if.end
  PHINode *ifEndPhi = dyn_cast<PHINode>(&ifEnd->front());
  ifEndPhi->setIncomingValue(1, currentAdd);
  ifEndPhi->setIncomingBlock(1, forEnd37BB);

  changed = true;
  return changed;
}

// Function to unroll the cloned for.cond.preheader
static void unrollClonedForCondPreheader(BasicBlock *clonedForBody,
                                         BasicBlock *clonedForCondPreheader,
                                         BasicBlock *forCondPreheader) {
  Function *F = clonedForBody->getParent();
  BasicBlock *forBody = getBasicBlockByName(*F, "for.body");
  assert(forBody && "Expected to find for.body!");

  // Find PHI instructions in clonedForBody
  SmallVector<PHINode *> phiNodes;
  for (Instruction &I : *clonedForBody) {
    if (PHINode *phi = dyn_cast<PHINode>(&I)) {
      phiNodes.push_back(phi);
    }
  }

  // Remove unused PHI nodes in clonedForCondPreheader
  SmallVector<PHINode *> unusedPhiNodes;
  for (Instruction &I : *clonedForCondPreheader) {
    if (PHINode *phi = dyn_cast<PHINode>(&I)) {
      if (phi->use_empty()) {
        unusedPhiNodes.push_back(phi);
      }
    }
  }
  for (PHINode *phi : unusedPhiNodes) {
    phi->eraseFromParent();
  }

  // Clone PHI instructions to the beginning of clonedForCondPreheader
  Instruction *insertPoint = &clonedForCondPreheader->front();
  SmallVector<PHINode *> clonedPhiNodes;
  for (PHINode *phi : phiNodes) {
    PHINode *clonedPhi = cast<PHINode>(phi->clone());
    clonedPhi->setName(phi->getName() + ".clone");
    clonedPhi->setIncomingBlock(0, forBody);
    clonedPhi->insertBefore(insertPoint);
    insertPoint = clonedPhi->getNextNode();
    clonedPhiNodes.push_back(clonedPhi);
  }

  // Find and clone the unique icmp instruction in forBody
  Value *specStoreSelect = nullptr;
  Instruction *cmpSlt = nullptr;
  for (Instruction &I : *forBody) {
    if (auto *icmp = dyn_cast<ICmpInst>(&I)) {
      specStoreSelect = icmp->getOperand(0);
      cmpSlt = icmp->clone();
      cmpSlt->setName("cmp_slt");
      cmpSlt->insertAfter(insertPoint);
      break;
    }
  }
  assert(specStoreSelect && "Failed to find icmp instruction in ForBody");

  // Replace the existing icmp in clonedForCondPreheader
  for (Instruction &I : *clonedForCondPreheader) {
    if (auto *icmp = dyn_cast<ICmpInst>(&I)) {
      icmp->replaceAllUsesWith(cmpSlt);
      icmp->eraseFromParent();
      break;
    }
  }

  // Set the operand of cmp_slt to the first cloned PHI node
  cmpSlt->setOperand(0, clonedPhiNodes[0]);

  // Update the successor of clonedForCondPreheader
  clonedForCondPreheader->getTerminator()->setSuccessor(1, forCondPreheader);
}

static std::tuple<Value *, Value *, Value *>
modifyForBodyPreheader(BasicBlock *ForBodyPreheader,
                       BasicBlock *ClonedForCondPreheader) {
  PHINode *TargetPHI = nullptr;
  PHINode *TargetPHI2 = nullptr;
  PHINode *TargetPHI3 = nullptr;
  for (Instruction &I : *ClonedForCondPreheader) {
    if (auto *phi = dyn_cast<PHINode>(&I)) {
      if (phi->getType()->isIntegerTy(32)) {
        if (isIncomingValueZeroOfPhi(phi)) {
          // Found the target PHI node
          TargetPHI = phi;
        } else {
          TargetPHI2 = phi;
        }
      } else if (phi->getType()->isFloatTy()) {
        if (TargetPHI3 == nullptr) {
          TargetPHI3 = phi;
          break;
        }
      }
    }
  }
  BinaryOperator *NewSub = nullptr;
  for (Instruction &I : *ForBodyPreheader) {
    if (auto *BinOp = dyn_cast<BinaryOperator>(&I)) {
      if (BinOp->getOpcode() == Instruction::Sub) {
        // Change to add
        NewSub = BinaryOperator::CreateAdd(BinOp->getOperand(0), TargetPHI,
                                           BinOp->getName(), BinOp);
        BinOp->replaceAllUsesWith(NewSub);
        BinOp->eraseFromParent();
        break;
      }
    }
  }

  ForBodyPreheader->moveAfter(ClonedForCondPreheader);
  assert(NewSub && "NewSub should not be nullptr");
  return std::make_tuple(NewSub, TargetPHI2, TargetPHI3);
}

static Value *expandForCondPreheader(
    BasicBlock *ForBody, BasicBlock *ForCondPreheader,
    BasicBlock *ClonedForCondPreheader,
    std::tuple<Value *, Value *, Value *> NewSubAndTargetPHI3) {
  Instruction *TargetInst =
      getFirstCallInstWithName(ForBody, "llvm.fmuladd.f32");
  assert(TargetInst && "TargetInst not found");
  Value *NewSub = std::get<0>(NewSubAndTargetPHI3);
  Value *TargetPHI2 = std::get<1>(NewSubAndTargetPHI3);
  Value *TargetPHI3 = std::get<2>(NewSubAndTargetPHI3);
  // Create new .loopexit basic block
  BasicBlock *LoopExit = BasicBlock::Create(
      ForCondPreheader->getContext(), ForCondPreheader->getName() + ".loopexit",
      ForCondPreheader->getParent(), ForCondPreheader);

  // Create new sub instruction in .loopexit block
  IRBuilder<> Builder(LoopExit);
  Value *NewSubInst = Builder.CreateSub(NewSub, TargetPHI2);

  // Add unconditional branch to ForCondPreheader
  Builder.CreateBr(ForCondPreheader);

  // Find the target PHI node in ClonedForCondPreheader
  PHINode *TargetPHI = nullptr;
  for (PHINode &Phi : ClonedForCondPreheader->phis()) {
    if (isIncomingValueZeroOfPhi(&Phi)) {
      TargetPHI = &Phi;
      break;
    }
  }

  // Ensure we found the target PHI node
  assert(TargetPHI &&
         "Failed to find target PHI node in ClonedForCondPreheader");

  // Update the incoming value of the PHI nodes in ForCondPreheader to the
  // result of the new sub instruction
  for (PHINode &Phi : ForCondPreheader->phis()) {
    if (Phi.getType()->isIntegerTy(32)) {
      Phi.setIncomingValue(0, TargetPHI);
      Phi.setIncomingBlock(0, ClonedForCondPreheader);
      Phi.setIncomingValue(1, NewSubInst);
      Phi.setIncomingBlock(1, LoopExit);
    } else if (Phi.getType()->isFloatTy()) {
      Phi.setIncomingValue(0, TargetPHI3);
      Phi.setIncomingBlock(0, ClonedForCondPreheader);
      // Phi.setIncomingValue(1, TargetInst);
      Phi.setIncomingBlock(1, LoopExit);
    }
  }

  // Get the icmp instruction in ForCondPreheader
  ICmpInst *icmpInst = getFirstInst<ICmpInst>(ForCondPreheader);

  // Ensure we found the icmp instruction
  assert(icmpInst && "Failed to find icmp instruction in ForCondPreheader");

  // Set the operand 1 of icmpInst to constant 7
  LLVMContext &Ctx = ForCondPreheader->getContext();
  Value *const7 = ConstantInt::get(Type::getInt32Ty(Ctx), 7);
  icmpInst->setOperand(1, const7);

  // Create a new add nsw instruction before icmpInst, with operand 0 the same
  // as icmpInst, and operand 1 as -7. This instruction will be used as the
  // return value of the function
  Value *constNeg7 = ConstantInt::get(Type::getInt32Ty(Ctx), -7);
  IRBuilder<> BuilderBeforeICmp(icmpInst);
  Value *AddInst =
      BuilderBeforeICmp.CreateNSWAdd(icmpInst->getOperand(0), constNeg7);

  ForBody->getTerminator()->setSuccessor(0, LoopExit);

  return AddInst;
}

static void updateRealForBody(Function &F, Value *sub) {
  BasicBlock *ForBody = getBasicBlockByName(F, "for.body");
  assert(ForBody && "Expected to find for.body!");
  ICmpInst *lastICmp =
      getLastICmpInstWithPredicate(ForBody, ICmpInst::ICMP_SLT);
  if (lastICmp) {
    lastICmp->setOperand(1, sub);
  }
}

static void modifyForBody(BasicBlock *ClonedForCondPreheader,
                          BasicBlock *ForBody) {
  // Find the unique float type PHI node in ForBody
  PHINode *FloatPhiInForBody = getFirstFloatPhi(ForBody);
  assert(FloatPhiInForBody && "Failed to find float type PHI node in ForBody");
  // Find the first float type PHI node in ClonedForCondPreheader
  PHINode *FirstFloatPhiInClonedForCondPreheader =
      getFirstFloatPhi(ClonedForCondPreheader);
  assert(FloatPhiInForBody && "Failed to find float type PHI node in ForBody");
  // Set the incoming value of the float type PHI node in ForBody to the float
  // type PHI node in ClonedForCondPreheader
  FloatPhiInForBody->setIncomingValue(0, FirstFloatPhiInClonedForCondPreheader);

  // Find the unique icmp eq instruction in ForBody
  ICmpInst *IcmpEq = getFirstICmpInstWithPredicate(ForBody, ICmpInst::ICMP_EQ);

  // Ensure we found the icmp eq instruction
  assert(IcmpEq && "Failed to find icmp eq instruction in ForBody");

  // Get the original operand 1
  Value *OriginalOperand1 = IcmpEq->getOperand(1);

  // Ensure the original operand 1 is an instruction
  if (Instruction *OriginalOperand1Inst =
          dyn_cast<Instruction>(OriginalOperand1)) {
    // Set operand 1 to the operand 0 of the original operand 1 instruction
    IcmpEq->setOperand(1, OriginalOperand1Inst->getOperand(0));
  } else {
    assert(false && "The original operand 1 is not an instruction, "
                    "cannot get its operand 0\n");
  }

  // Find the phi i32 incoming value that is a variable in
  // ClonedForCondPreheader
  PHINode *TargetPHI = nullptr;
  PHINode *TargetPHI2 = nullptr;
  for (Instruction &I : *ClonedForCondPreheader) {
    if (PHINode *Phi = dyn_cast<PHINode>(&I)) {
      if (isIncomingValueZeroOfPhi(Phi)) {
        TargetPHI = Phi;
      } else {
        TargetPHI2 = Phi;
      }
      if (TargetPHI && TargetPHI2)
        break;
    }
  }

  // Ensure we found the target PHI node
  assert(TargetPHI &&
         "Failed to find the target PHI node in ClonedForCondPreheader");

  // Find the phi i32 incoming value that is a variable in ForBody
  PHINode *TargetPHIInForBody = nullptr;
  PHINode *TargetPHIInForBody2 = nullptr;
  for (Instruction &I : *ForBody) {
    if (PHINode *Phi = dyn_cast<PHINode>(&I)) {
      if (isIncomingValueZeroOfPhi(Phi)) {
        TargetPHIInForBody = Phi;
      } else {
        TargetPHIInForBody2 = Phi;
      }
      if (TargetPHIInForBody && TargetPHIInForBody2)
        break;
    }
  }

  // Ensure that the target PHI nodes are found
  assert(TargetPHIInForBody && TargetPHIInForBody2 &&
         "Failed to find matching PHI nodes in ForBody");

  // Set the incoming value of the PHI nodes found in ForBody
  // to the PHI nodes found in ClonedForCondPreheader
  TargetPHIInForBody->setIncomingValue(0, TargetPHI);
  TargetPHIInForBody2->setIncomingValue(0, TargetPHI2);

  IcmpEq->setOperand(0, TargetPHIInForBody2->getIncomingValue(1));
}

static void insertUnusedInstructionsBeforeIcmp(PHINode *phiI32InClonedForBody,
                                               ICmpInst *lastIcmpEq) {
  for (Use &U : phiI32InClonedForBody->uses()) {
    if (Instruction *Used = dyn_cast<Instruction>(U.getUser())) {
      if (Used->getParent() == nullptr) {
        if (Used->use_empty()) {
          Used->insertBefore(lastIcmpEq);
        }
      }
    }
  }
}

static void modifyClonedForBody(BasicBlock *ClonedForBody) {

  ICmpInst *lastIcmpEq = getLastInst<ICmpInst>(ClonedForBody);
  assert(lastIcmpEq &&
         "Failed to find last icmp eq instruction in ClonedForBody");

  PHINode *phiI32InClonedForBody = nullptr;
  for (auto &Inst : *ClonedForBody) {
    if (PHINode *Phi = dyn_cast<PHINode>(&Inst)) {
      if (isIncomingValueZeroOfPhi(Phi)) {
        phiI32InClonedForBody = Phi;
        insertUnusedInstructionsBeforeIcmp(phiI32InClonedForBody, lastIcmpEq);
      }
    }
  }

  // Ensure that the phi i32 node is found
  assert(phiI32InClonedForBody && "phi i32 node not found in ClonedForBody");
}

static BasicBlock *getFirstSuccessorOfForBody(BasicBlock *ForBody) {
  BasicBlock *ForCondPreheader = nullptr;
  assert(succ_size(ForBody) == 2 && "ForBody should have 2 successors");
  for (auto *succ : successors(ForBody)) {
    ForCondPreheader = succ;
    break;
  }
  return ForCondPreheader;
}

static std::tuple<BasicBlock *, BasicBlock *, BasicBlock *>
cloneThreeBB(BasicBlock *ForBodyPreheader, BasicBlock *ForBody,
             BasicBlock *ForCondPreheader, Function &F) {
  ValueToValueMapTy VMap;
  SmallVector<BasicBlock *, 2> NewBlocks;

  BasicBlock *ClonedForBodyPreheader =
      CloneBasicBlock(ForBodyPreheader, VMap, ".modify", &F);
  BasicBlock *ClonedForBody = CloneBasicBlock(ForBody, VMap, ".modify", &F);
  BasicBlock *ClonedForCondPreheader =
      CloneBasicBlock(ForCondPreheader, VMap, ".modify", &F);

  VMap[ForBodyPreheader] = ClonedForBodyPreheader;
  VMap[ForBody] = ClonedForBody;
  VMap[ForCondPreheader] = ClonedForCondPreheader;

  // Remap instructions and PHI nodes in the new loop
  remapInstructionsInBlocks(
      {ClonedForBodyPreheader, ClonedForBody, ClonedForCondPreheader}, VMap);
  return std::make_tuple(ClonedForBodyPreheader, ClonedForBody,
                         ClonedForCondPreheader);
}

static std::tuple<BasicBlock *, BasicBlock *, Value *>
modifyFirstForBody(Loop *L, Function &F, BasicBlock *ForBody, Value *sub) {

  BasicBlock *ForBodyPreheader = L->getLoopPreheader();

  // Find the predecessor of ForBodyPreheader
  BasicBlock *PreForBody = nullptr;
  assert(pred_size(ForBodyPreheader) == 1 &&
         "ForBodyPreheader should have only one predecessor");
  for (auto *Pred : predecessors(ForBodyPreheader)) {
    PreForBody = Pred;
  }

  // Find the first successor of ForBody, it should have two
  BasicBlock *ForCondPreheader = getFirstSuccessorOfForBody(ForBody);

  std::tuple<BasicBlock *, BasicBlock *, BasicBlock *> ClonedBBs =
      cloneThreeBB(ForBodyPreheader, ForBody, ForCondPreheader, F);
  BasicBlock *ClonedForBodyPreheader = std::get<0>(ClonedBBs);
  BasicBlock *ClonedForBody = std::get<1>(ClonedBBs);
  BasicBlock *ClonedForCondPreheader = std::get<2>(ClonedBBs);

  /* insert 2 cloned blocks between PreForBody and ForBody */
  // for.body -> for.body12.lr.ph
  PreForBody->getTerminator()->setSuccessor(0, ClonedForBodyPreheader);
  ClonedForBodyPreheader->moveAfter(PreForBody);
  // for.body12.lr.ph -> for.body12
  ClonedForBodyPreheader->getTerminator()->setSuccessor(0, ClonedForBody);

  // for.body12 -> for.cond59.preheader
  ClonedForBody->moveAfter(ClonedForBodyPreheader);

  // for.cond59.preheader -> for.body62.lr.ph
  ClonedForCondPreheader->getTerminator()->setSuccessor(0, ForBodyPreheader);

  // for.cond59.preheader -> for.cond71.preheader
  ClonedForCondPreheader->getTerminator()->setSuccessor(1,
                                                        ClonedForCondPreheader);
  ClonedForCondPreheader->moveAfter(ClonedForBodyPreheader);
  // for.body -> for.cond71.preheader
  PreForBody->getTerminator()->setSuccessor(1, ClonedForCondPreheader);

  preProcessClonedForBody(ClonedForBody, sub);
  updateRealForBody(F, sub);
  unrollClonedForBody(ClonedForBody, ClonedForCondPreheader, 0);
  modifyClonedForBody(ClonedForBody);
  unrollClonedForCondPreheader(ClonedForBody, ClonedForCondPreheader,
                               ForCondPreheader);

  modifyForBody(ClonedForCondPreheader, ForBody);
  std::tuple<Value *, Value *, Value *> NewSubAndTargetPHI3 =
      modifyForBodyPreheader(ForBodyPreheader, ClonedForCondPreheader);

  Value *AddInst = expandForCondPreheader(
      ForBody, ForCondPreheader, ClonedForCondPreheader, NewSubAndTargetPHI3);

  ClonedForBodyPreheader->moveBefore(ClonedForBody);
  groupAndReorderInstructions(ClonedForBody);
  return std::make_tuple(ClonedForCondPreheader, ForCondPreheader, AddInst);
}

static bool moveIfEndToEnd(Function &F) {

  BasicBlock &lastBB = F.back();
  if (lastBB.getName() == "if.end") {
    return false;
  }

  BasicBlock *ifEndBB = getBasicBlockByName(F, "if.end");
  assert(ifEndBB && "Expected to find if.end!");
  if (ifEndBB) {
    ifEndBB->removeFromParent();
    ifEndBB->insertInto(&F);
  }
  return true;
}

static Value *modifyForCondPreheader(Function &F) {
  LLVMContext &Ctx = F.getContext();

  BasicBlock *forCondPreheader = getBasicBlockByName(F, "for.cond.preheader");
  BasicBlock *forBodyLrPh = getBasicBlockByName(F, "for.body.lr.ph");
  assert(forCondPreheader && "Expected to find for.cond.preheader!");
  assert(forBodyLrPh && "Expected to find for.body.lr.ph!");
  forCondPreheader->replaceAllUsesWith(forBodyLrPh);
  forCondPreheader->eraseFromParent();
  forBodyLrPh->setName("for.cond.preheader");

  unsigned int loadnum = 0;
  for (auto I = forBodyLrPh->begin(); I != forBodyLrPh->end(); ++I) {
    if (auto *loadinst = dyn_cast<LoadInst>(&*I)) {
      loadnum++;
      if (loadnum == 2) {
        IRBuilder<> Builder(loadinst->getNextNode());
        Value *NegSeven = ConstantInt::get(Type::getInt32Ty(Ctx), -7);
        Value *Sub = Builder.CreateNSWAdd(loadinst, NegSeven, "sub");
        return Sub; // Return the newly inserted instruction
      }
    }
  }
  assert(false && "it must not be here");
}

static void modifyForCondPreheader2(BasicBlock *ClonedForBody,
                                    BasicBlock *ClonedForCondPreheader,
                                    BasicBlock *ForCondPreheader,
                                    Value *andinst) {

  // Find phi instructions of float type in ClonedForBody
  SmallVector<PHINode *> PhiNodes;
  for (Instruction &I : *ClonedForBody) {
    if (PHINode *Phi = dyn_cast<PHINode>(&I)) {
      PhiNodes.push_back(Phi);
    }
  }

  // Clone the found phi instructions to the beginning of ClonedForCondPreheader
  // in order
  Instruction *InsertPoint = &ForCondPreheader->front();
  PHINode *phi = cast<PHINode>(InsertPoint);

  BasicBlock *lastForCondPreheader = phi->getIncomingBlock(0);
  SmallVector<PHINode *> ClonedPhiNodes;
  unsigned int floatphicount = 0;
  for (PHINode *Phi : PhiNodes) {
    PHINode *ClonedPhi = cast<PHINode>(Phi->clone());
    ClonedPhi->setName(Phi->getName() + ".clone");
    // Modify the operand 0 basicblock of each phi instruction to ForBody
    if (Phi->getType()->isFloatTy()) {
      if (floatphicount == 0) {
        ClonedPhi->setIncomingValue(0, phi->getIncomingValue(0));
        floatphicount++;
      }
    }
    ClonedPhi->setIncomingBlock(0, lastForCondPreheader);
    ClonedPhi->insertAfter(InsertPoint);
    // Update the insertion point to after the newly inserted PHI node
    InsertPoint = ClonedPhi;

    ClonedPhiNodes.push_back(ClonedPhi);
  }

  // Find operand 1 of the icmp instruction from ClonedForBody
  ICmpInst *firstIcmp = getFirstInst<ICmpInst>(ClonedForBody);
  assert(firstIcmp && "Unable to find icmp instruction in ClonedForBody");
  Value *IcmpOperand1 = firstIcmp->getOperand(1);

  // Set operand 0 of icmp in ForCondPreheader to ClonedPhiNodes[0], and operand
  // 1 to IcmpOperand1
  for (Instruction &I : *ForCondPreheader) {
    if (ICmpInst *Icmp = dyn_cast<ICmpInst>(&I)) {
      Icmp->setOperand(0, ClonedPhiNodes[0]);
      Icmp->setOperand(1, IcmpOperand1);
      Icmp->setName("cmp");
      break;
    }
  }

  ForCondPreheader->getTerminator()->setSuccessor(1, ClonedForCondPreheader);

  // // Delete redundant getelementptr, store and add instructions
  SmallVector<Instruction *> InstructionsToRemove;
  for (Instruction &I : *ForCondPreheader) {
    if (isa<GetElementPtrInst>(&I) || isa<StoreInst>(&I) ||
        isa<BinaryOperator>(&I)) {
      InstructionsToRemove.push_back(&I);
    }
  }
  for (auto Inst = InstructionsToRemove.rbegin();
       Inst != InstructionsToRemove.rend(); ++Inst) {
    if ((*Inst)->use_empty()) {
      (*Inst)->eraseFromParent();
    }
  }
  // Find the icmp instruction in ClonedForCondPreheader
  ICmpInst *IcmpInForCondPreheader =
      getFirstICmpInstWithPredicate(ForCondPreheader, ICmpInst::ICMP_EQ);

  // Ensure that the icmp instruction is found
  assert(IcmpInForCondPreheader &&
         "icmp instruction not found in ClonedForCondPreheader");

  // Get the original operand 1
  Value *OriginalOperand1 = IcmpInForCondPreheader->getOperand(1);

  // If the original operand 1 is an instruction, get its operand 0
  if (Instruction *OriginalOperand1Inst =
          dyn_cast<Instruction>(OriginalOperand1)) {
    Value *NewOperand1 = OriginalOperand1Inst->getOperand(0);

    // Set the new operand 1
    IcmpInForCondPreheader->setOperand(1, NewOperand1);
    // Change the original eq to slt

    IcmpInForCondPreheader->setPredicate(CmpInst::ICMP_SLT);

  } else {
    assert(false && "The original operand 1 is not an instruction, cannot get "
                    "its operand 0\n");
  }

  // Find phi i32 node in ForCondPreheader with incoming 0 value == 0
  PHINode *TargetPhi = nullptr;
  for (Instruction &I : *ForCondPreheader) {
    if (PHINode *Phi = dyn_cast<PHINode>(&I)) {
      if (isIncomingValueZeroOfPhi(Phi)) {
        TargetPhi = Phi;
        break;
      }
    }
  }

  // Ensure the target phi node is found
  assert(TargetPhi && "No matching phi i32 node found in ForCondPreheader");

  TargetPhi->setIncomingValue(1, andinst);
}

static Value *modifyClonedForBodyPreheader(BasicBlock *ClonedForBodyPreheader,
                                           BasicBlock *ForBody) {
  ICmpInst *firstIcmp = getFirstInst<ICmpInst>(ForBody);
  assert(firstIcmp && "Unable to find icmp instruction in ForBody");

  Value *IcmpOperand1 = firstIcmp->getOperand(1);

  IRBuilder<> Builder(ClonedForBodyPreheader->getTerminator());
  Value *AndInst =
      Builder.CreateAnd(IcmpOperand1, Builder.getInt32(2147483640));
  return AndInst;
}

static void modifyClonedForCondPreheader(BasicBlock *ClonedForCondPreheader,
                                         BasicBlock *ForBody,
                                         BasicBlock *ForCondPreheader) {

  // Find float type phi node in ForBody
  PHINode *FloatPhiInForBody = nullptr;
  for (Instruction &I : *ForBody) {
    if (PHINode *Phi = dyn_cast<PHINode>(&I)) {
      if (Phi->getType()->isFloatTy()) {
        FloatPhiInForBody = cast<PHINode>(I.clone());
        break;
      }
    }
  }

  // Find and replace float type phi node in ClonedForCondPreheader
  if (FloatPhiInForBody) {
    PHINode *phi = getFirstFloatPhi(ClonedForCondPreheader);
    assert(phi && "phi node not found");
    FloatPhiInForBody->insertBefore(phi);
    phi->replaceAllUsesWith(FloatPhiInForBody);
    phi->eraseFromParent();
  }

  // Set incomingblock 0 of FloatPhiInForBody to ForCondPreheader
  if (FloatPhiInForBody) {
    FloatPhiInForBody->setIncomingBlock(0, ForCondPreheader);
  }

  // Find float type phi nodes in ForCondPreheader
  SmallVector<PHINode *> FloatPhisInForCondPreheader;
  for (Instruction &I : *ForCondPreheader) {
    if (PHINode *Phi = dyn_cast<PHINode>(&I)) {
      if (Phi->getType()->isFloatTy()) {
        FloatPhisInForCondPreheader.push_back(Phi);
      }
    }
  }

  // Create 7 fadd instructions
  Value *LastFAdd = nullptr;
  if (FloatPhisInForCondPreheader.size() >= 8) {
    IRBuilder<> Builder(FloatPhiInForBody->getNextNode());

    Value *PrevAdd = getFirstFloatPhi(ClonedForCondPreheader);

    assert(PrevAdd &&
           "Unable to find float type PHI node in ClonedForCondPreheader");
    Value *Add139 =
        Builder.CreateFAdd(PrevAdd, FloatPhisInForCondPreheader[2], "add139");
    Value *Add140 =
        Builder.CreateFAdd(FloatPhisInForCondPreheader[3],
                           FloatPhisInForCondPreheader[4], "add140");
    Value *Add141 =
        Builder.CreateFAdd(FloatPhisInForCondPreheader[5],
                           FloatPhisInForCondPreheader[6], "add141");
    Value *Add142 =
        Builder.CreateFAdd(FloatPhisInForCondPreheader[7],
                           FloatPhisInForCondPreheader[8], "add142");
    Value *Add143 = Builder.CreateFAdd(Add139, Add140, "add143");
    Value *Add144 = Builder.CreateFAdd(Add141, Add142, "add144");
    Value *Add145 = Builder.CreateFAdd(Add143, Add144, "add145");
    LastFAdd = Add145;
  } else {
    llvm_unreachable("Unable to find float type PHI node in ForCondPreheader");
  }

  // Find store instruction in ForCondPreheader and update its operand
  if (LastFAdd) {
    for (auto &Inst : *ClonedForCondPreheader) {
      if (auto *si = dyn_cast<StoreInst>(&Inst)) {
        si->setOperand(0, LastFAdd);
        break;
      }
    }
  }

  Value *addinst = nullptr;
  // Iterate through instructions in ClonedForCondPreheader, looking for addnuw
  // instruction
  for (auto &Inst : *ClonedForCondPreheader) {
    if (auto *AddInst = dyn_cast<BinaryOperator>(&Inst)) {
      if (AddInst->getOpcode() == Instruction::Add &&
          AddInst->hasNoUnsignedWrap()) {
        addinst = AddInst;
        break;
      }
    }
  }
  // Get the second successor of ClonedForCondPreheader
  BasicBlock *SecondSuccessor = nullptr;
  int SuccCount = 0;
  for (auto *Succ : successors(ClonedForCondPreheader)) {
    if (SuccCount == 1) {
      SecondSuccessor = Succ;
      break;
    }
    SuccCount++;
  }

  if (SecondSuccessor && addinst) {
    // Iterate through all PHI nodes in SecondSuccessor
    int phiCount = 0;
    for (PHINode &Phi : SecondSuccessor->phis()) {
      if (phiCount == 1) { // Second phi node
        // Set the second predecessor to ClonedForCondPreheader and its value to
        // addinst
        Phi.setIncomingBlock(1, ClonedForCondPreheader);
        Phi.setIncomingValue(1, addinst);
      } else {
        // For other phi nodes, only update the predecessor basic block
        Phi.setIncomingBlock(1, ClonedForCondPreheader);
      }
      phiCount++;
    }
  }
}

static void modifyClonedForBody2(BasicBlock *ClonedForBody,
                                 BasicBlock *ClonedForCondPreheader,
                                 Value *AddInst, BasicBlock *ForCondPreheader) {
  SmallVector<PHINode *> floatPhiNodes;

  // Iterate through all instructions in ClonedForCondPreheader
  for (Instruction &I : *ClonedForCondPreheader) {
    if (PHINode *Phi = dyn_cast<PHINode>(&I)) {
      if (Phi->getType()->isFloatTy()) {
        floatPhiNodes.push_back(Phi);
        if (floatPhiNodes.size() == 8) {
          break; // Stop after finding 8 float type PHI nodes
        }
      }
    }
  }

  // Ensure we found 8 float type PHI nodes
  assert(floatPhiNodes.size() == 8 &&
         "Unable to find 8 float type PHI nodes in ClonedForCondPreheader");

  // Now floatPhiNodes contains 8 float type PHI nodes in order

  // Iterate through all PHI nodes in ClonedForBody
  int phiIndex = 0;
  for (PHINode &Phi : ClonedForBody->phis()) {
    if (Phi.getType()->isFloatTy()) {
      // Ensure we don't access floatPhiNodes out of bounds
      if (phiIndex < floatPhiNodes.size()) {
        // Set the 0th incoming value of the PHI node to the corresponding node
        // in floatPhiNodes
        if (phiIndex >
            0) { // Don't set the first phi node, as it's floatPhiInForBody
          Phi.setIncomingValue(0, floatPhiNodes[phiIndex]);
        }
        phiIndex++;
      } else {
        // If the number of float type PHI nodes in ClonedForBody exceeds the
        // size of floatPhiNodes, output a warning
        assert(false && "Warning: Number of float type PHI nodes in "
                        "ClonedForBody exceeds expectations\n");
        break;
      }
    }
  }

  // Ensure we processed all expected PHI nodes
  if (phiIndex < floatPhiNodes.size()) {
    assert(false && "Warning: Number of float type PHI nodes in ClonedForBody "
                    "is less than expected\n");
  }

  // Find the last icmp eq instruction in ClonedForBody
  ICmpInst *lastIcmpEq =
      getLastICmpInstWithPredicate(ClonedForBody, ICmpInst::ICMP_EQ);

  // Ensure we found the icmp eq instruction
  assert(lastIcmpEq && "Unable to find icmp eq instruction in ClonedForBody");

  // Set operand 1 to addInst
  lastIcmpEq->setOperand(1, AddInst);
  // Change the predicate of the icmp eq instruction to slt (signed less than)
  lastIcmpEq->setPredicate(ICmpInst::ICMP_SLT);
  // Change the name to cmp
  lastIcmpEq->setName("cmp");

  ClonedForBody->getTerminator()->setSuccessor(1, ForCondPreheader);

  // Find phi i32 node in ClonedForBody
  PHINode *phiI32InClonedForBody = nullptr;
  for (auto &Inst : *ClonedForBody) {
    if (PHINode *Phi = dyn_cast<PHINode>(&Inst)) {
      if (Phi->getType()->isIntegerTy(32)) {
        phiI32InClonedForBody = Phi;
        insertUnusedInstructionsBeforeIcmp(phiI32InClonedForBody, lastIcmpEq);
      }
    }
  }

  // Ensure we found the phi i32 node
  assert(phiI32InClonedForBody &&
         "Unable to find phi i32 node in ClonedForBody");
}

static std::pair<PHINode *, PHINode *> findTwoI32PhiInBB(BasicBlock *ForBody) {
  // Find the first i32 type PHI instruction in ForBody
  PHINode *firstI32PhiInBB = nullptr;
  PHINode *secondI32PhiInBB = nullptr;
  int i32PhiCount2 = 0;
  for (auto &Inst : *ForBody) {
    if (PHINode *Phi = dyn_cast<PHINode>(&Inst)) {
      if (Phi->getType()->isIntegerTy(32)) {
        if (i32PhiCount2 == 0) {
          firstI32PhiInBB = Phi;
          i32PhiCount2++;
        } else if (i32PhiCount2 == 1) {
          secondI32PhiInBB = Phi;
          break;
        }
      }
    }
  }

  // Ensure we found two i32 type PHI instructions in ForBody
  assert(firstI32PhiInBB && secondI32PhiInBB &&
         "Unable to find two i32 type PHI instructions in BB");

  return std::make_pair(firstI32PhiInBB, secondI32PhiInBB);
}
static void modifyForBody2(BasicBlock *ClonedForCondPreheader,
                           BasicBlock *ForBody, BasicBlock *ForCondPreheader) {
  // Find the first i32 type PHI instruction in ForCondPreheader
  auto [firstI32PhiInForCondPreheader, secondI32PhiInForCondPreheader] =
      findTwoI32PhiInBB(ForCondPreheader);

  // Find the first i32 type PHI instruction in ForBody
  auto [firstI32PhiInForBody, secondI32PhiInForBody] =
      findTwoI32PhiInBB(ForBody);

  // Set the incoming 0 value of the two i32 type PHI instructions found in
  // ForBody to the firstI32Phi found in ForCondPreheader
  firstI32PhiInForBody->setIncomingValue(0, firstI32PhiInForCondPreheader);
  secondI32PhiInForBody->setIncomingValue(0, secondI32PhiInForCondPreheader);

  ForBody->getTerminator()->setSuccessor(0, ClonedForCondPreheader);

  // Find the first float type PHI instruction in ForCondPreheader
  PHINode *SecondFloatPhiInForCondPreheader = nullptr;
  int floatPhiCount = 0;
  for (auto &Inst : *ForCondPreheader) {
    if (PHINode *Phi = dyn_cast<PHINode>(&Inst)) {
      if (Phi->getType()->isFloatTy()) {
        floatPhiCount++;
        if (floatPhiCount == 2) {
          SecondFloatPhiInForCondPreheader = Phi;
          break;
        }
      }
    }
  }

  // Ensure we found a float type PHI instruction in ForCondPreheader
  assert(SecondFloatPhiInForCondPreheader &&
         "Unable to find float type PHI instruction in ForCondPreheader");

  // Find the only float type PHI instruction in ForBody
  PHINode *FloatPhiInForBody = getFirstFloatPhi(ForBody);
  assert(FloatPhiInForBody && "Unable to find float type PHI instruction in "
                              "ForBody");

  // Set incoming value 0 of the float type PHI instruction in ForBody
  FloatPhiInForBody->setIncomingValue(0, SecondFloatPhiInForCondPreheader);

  // Find the unique float type PHI instruction in ClonedForCondPreheader
  PHINode *FloatPhiInClonedForCondPreheader =
      getFirstFloatPhi(ClonedForCondPreheader);
  assert(FloatPhiInClonedForCondPreheader &&
         "Float type PHI instruction not found in ClonedForCondPreheader");

  // Set incoming value 0 of the float type PHI instruction in
  // ClonedForCondPreheader
  FloatPhiInClonedForCondPreheader->setIncomingValue(
      0, SecondFloatPhiInForCondPreheader);
}

// Helper function to run dead code elimination
static void runDeadCodeElimination(Function &F) {
  legacy::FunctionPassManager FPM(F.getParent());
  FPM.add(createDeadCodeEliminationPass());
  FPM.run(F);
  LLVM_DEBUG(F.dump());
}

static bool modifySecondForBody(Loop *L, Function &F, BasicBlock *ForBody,
                                BasicBlock *FirstClonedForCondPreheader,
                                BasicBlock *FirstForCondPreheader,
                                Value *AddInst) {
  BasicBlock *ForBodyPreheader = L->getLoopPreheader();

  // Find the 0th successor of ForBody, it should have two
  BasicBlock *ForCondPreheader = getFirstSuccessorOfForBody(ForBody);

  std::tuple<BasicBlock *, BasicBlock *, BasicBlock *> ClonedBBs =
      cloneThreeBB(ForBodyPreheader, ForBody, ForCondPreheader, F);
  BasicBlock *ClonedForBodyPreheader = std::get<0>(ClonedBBs);
  BasicBlock *ClonedForBody = std::get<1>(ClonedBBs);
  BasicBlock *ClonedForCondPreheader = std::get<2>(ClonedBBs);

  ClonedForCondPreheader->setName("for.end");
  ClonedForBody->moveBefore(ForBody);
  ClonedForBodyPreheader->moveBefore(ClonedForBody);
  ForCondPreheader->moveBefore(ClonedForBodyPreheader);
  ClonedForCondPreheader->moveAfter(ForBody);
  ForCondPreheader->getTerminator()->setSuccessor(0, ForBodyPreheader);

  unrollClonedForBody(ClonedForBody, ClonedForCondPreheader, 1);
  modifyClonedForBody2(ClonedForBody, FirstClonedForCondPreheader, AddInst,
                       ForCondPreheader);

  Value *andinst =
      modifyClonedForBodyPreheader(ClonedForBodyPreheader, ForBody);
  modifyForCondPreheader2(ClonedForBody, ClonedForCondPreheader,
                          ForCondPreheader, andinst);
  modifyClonedForCondPreheader(ClonedForCondPreheader, ForBody,
                               ForCondPreheader);
  modifyForBody2(ClonedForCondPreheader, ForBody, ForCondPreheader);

  FirstForCondPreheader->getTerminator()->setSuccessor(0,
                                                       ClonedForBodyPreheader);

  // Run Dead Code Elimination optimization
  runDeadCodeElimination(F);

  groupAndReorderInstructions(ClonedForBody);

  return true;
}
static void insertDoublePreheader(Function &F) {
  BasicBlock *entry = &F.getEntryBlock();
  BasicBlock *ifend = &F.back();
  BasicBlock *entry_successor1 = entry->getTerminator()->getSuccessor(1);

  // Create a new basic block
  BasicBlock *newBB = BasicBlock::Create(
      F.getContext(), entry_successor1->getName() + ".preheader", &F,
      entry_successor1);

  Value *len = getLenFromEntryBlock(F);

  // Insert instructions in the new basic block
  IRBuilder<> builder(newBB);
  Value *cmp151349 = builder.CreateICmpSGT(
      len, ConstantInt::get(len->getType(), 0), "cmp151349");

  // Create a conditional branch
  builder.CreateCondBr(cmp151349, entry_successor1, ifend);

  // Modify the terminator of entry to jump to the new basic block
  entry->getTerminator()->setSuccessor(1, newBB);
}
static bool unrollFir(Function &F, Loop *L) {

  bool Changed = false;
  static BasicBlock *FirstClonedForCondPreheader = nullptr;
  static BasicBlock *FirstForCondPreheader = nullptr;
  static Value *AddInst = nullptr;

  for (auto *BB : L->blocks()) {

    assert(BB->getName().contains("for.body") && "BB must is for.body");
    Changed = moveIfEndToEnd(F);
    // Temporarily skip processing the second loop

    if (Changed) {
      insertDoublePreheader(F);
      Value *sub = modifyForCondPreheader(F);
      std::tuple<BasicBlock *, BasicBlock *, Value *> result =
          modifyFirstForBody(L, F, BB, sub);
      FirstClonedForCondPreheader = std::get<0>(result);
      FirstForCondPreheader = std::get<1>(result);
      AddInst = std::get<2>(result);
    } else {
      modifySecondForBody(L, F, BB, FirstClonedForCondPreheader,
                          FirstForCondPreheader, AddInst);
    }
  }
  LLVM_DEBUG(F.dump());

  return Changed;
}

// Preprocessing function
static PHINode *preprocessClonedForBody(BasicBlock *ClonedForBody) {
  // Find the unique PHI node
  PHINode *phiNode = nullptr;
  for (auto &I : *ClonedForBody) {
    if (auto *phi = dyn_cast<PHINode>(&I)) {
      phiNode = phi;
      break;
    }
  }

  // Ensure that the PHI node is found
  assert(phiNode && "PHI node not found");

  // Find two mul nsw instructions
  SmallVector<BinaryOperator *> mulInsts;
  for (auto &I : *ClonedForBody) {
    if (auto *binOp = dyn_cast<BinaryOperator>(&I)) {
      if (binOp->getOpcode() == Instruction::Mul && binOp->hasNoSignedWrap()) {
        mulInsts.push_back(binOp);
      }
    }
  }

  // Replace mul nsw instructions with the PHI node
  for (auto *mulInst : mulInsts) {
    mulInst->replaceAllUsesWith(phiNode);
    mulInst->eraseFromParent();
  }
  return phiNode;
}

static Instruction *modifyAddToOrInClonedForBody(BasicBlock *ClonedForBody) {
  // Find the unique add nuw nsw instruction
  Instruction *addInst = nullptr;
  for (auto &I : *ClonedForBody) {
    if (auto *binOp = dyn_cast<BinaryOperator>(&I)) {
      if (binOp->getOpcode() == Instruction::Add &&
          binOp->hasNoUnsignedWrap()) {
        addInst = binOp;
        break;
      }
    }
  }

  // Ensure that the add nuw nsw instruction is found
  assert(addInst && "add nuw nsw instruction not found");

  // Create a new or disjoint instruction
  Instruction *orInst = BinaryOperator::CreateDisjoint(
      Instruction::Or, addInst->getOperand(0),
      ConstantInt::get(addInst->getType(), 1), "add", addInst);

  // Replace all uses of the add instruction
  addInst->replaceAllUsesWith(orInst);

  // Delete the original add instruction
  addInst->eraseFromParent();
  orInst->setName("add");
  return orInst;
}

static void runInstCombinePass(Function &F) {
  // Create necessary analysis managers
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  // Create pass builder
  PassBuilder PB;

  // Register analyses
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  // Create function-level optimization pipeline
  FunctionPassManager FPM;
  FPM.addPass(InstCombinePass());
  FPM.run(F, FAM);
}

static Value *unrolladdcClonedForBody(BasicBlock *ClonedForBody,
                                      int unroll_factor) {

  // Call the preprocessing function
  PHINode *phiNode = preprocessClonedForBody(ClonedForBody);

  // Replace add instructions with or instructions
  Instruction *orInst = modifyAddToOrInClonedForBody(ClonedForBody);

  // Find the first non-PHI instruction and or instruction
  Instruction *firstNonPHI = ClonedForBody->getFirstNonPHI();

  // Ensure that the start and end instructions are found
  assert(firstNonPHI && orInst && "Start or end instruction not found");

  // Find the icmp instruction
  Instruction *icmpInst = getFirstInst<ICmpInst>(ClonedForBody);

  // Ensure that the icmp instruction is found
  assert(icmpInst && "icmp instruction not found");

  // Print information about the icmp instruction

  Instruction *newOrInst = orInst;
  // Copy instructions 15 times
  for (int i = 1; i <= (unroll_factor - 1); i++) {
    ValueToValueMapTy VMap;
    for (auto it = firstNonPHI->getIterator(); &*it != orInst; ++it) {
      Instruction *newInst = it->clone();
      // For getelementptr instructions, set the second operand to orInst
      if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(newInst)) {
        newInst->setOperand(1, newOrInst);
        newInst->setName("arrayidx");
      }
      // If it's a fadd instruction, change its name to add
      if (newInst->getOpcode() == Instruction::FAdd) {
        newInst->setName("add");
      }
      VMap[&*it] = newInst;
      newInst->insertBefore(icmpInst);
    }

    // Update operands of new instructions
    for (auto it = firstNonPHI->getIterator(); &*it != orInst; ++it) {
      Instruction *newInst = cast<Instruction>(VMap[&*it]);
      for (unsigned j = 0; j < newInst->getNumOperands(); j++) {
        Value *op = newInst->getOperand(j);
        if (VMap.count(op)) {
          newInst->setOperand(j, VMap[op]);
        }
      }
    }
    // Clone orInst and insert before icmpInst
    newOrInst = orInst->clone();
    // Set the second operand of newOrInst to i+1
    newOrInst->setOperand(1, ConstantInt::get(newOrInst->getType(), i + 1));
    newOrInst->setName("add");
    newOrInst->insertBefore(icmpInst);
    VMap[orInst] = newOrInst;
  }

  // Replace or instruction with add nuw nsw instruction
  IRBuilder<> Builder(newOrInst);
  Value *newAddInst =
      Builder.CreateNUWAdd(newOrInst->getOperand(0), newOrInst->getOperand(1));
  newOrInst->replaceAllUsesWith(newAddInst);
  newOrInst->eraseFromParent();

  // Create a new add instruction, subtracting 16 from len
  Builder.SetInsertPoint(icmpInst);
  Value *len = icmpInst->getOperand(1);
  Value *sub = Builder.CreateNSWAdd(
      len, ConstantInt::get(len->getType(), -unroll_factor), "sub");
  // Set the icmp instruction's predicate to sgt, and operands to newAddInst
  if (ICmpInst *icmp = dyn_cast<ICmpInst>(icmpInst)) {
    icmp->setPredicate(ICmpInst::ICMP_SGT);
    icmp->setOperand(0, newAddInst);
    icmp->setOperand(1, sub);
  }

  phiNode->setIncomingValue(0, newAddInst);
  return sub;
}

static void expandForCondPreheaderaddc(Function &F,
                                       BasicBlock *ForCondPreheader,
                                       BasicBlock *ClonedForBody,
                                       BasicBlock *ForBody, Value *sub,
                                       int unroll_factor) {
  // Create a new ForCondPreheader after the original ForCondPreheader
  BasicBlock *NewForCondPreheader = BasicBlock::Create(
      ForCondPreheader->getContext(), "for.cond.preheader.new",
      ForCondPreheader->getParent(), ForCondPreheader->getNextNode());
  // Create a new empty BasicBlock after NewForCondPreheader
  BasicBlock *NewForCondPreheader2 = BasicBlock::Create(
      NewForCondPreheader->getContext(), "for.cond.preheader.new2",
      NewForCondPreheader->getParent(), NewForCondPreheader->getNextNode());

  // Move sub to the new ForCondPreheader
  if (Instruction *SubInst = dyn_cast<Instruction>(sub)) {
    SubInst->removeFromParent();
    SubInst->insertInto(NewForCondPreheader, NewForCondPreheader->begin());
  }

  // Create new comparison instruction in NewForCondPreheader
  IRBuilder<> Builder(NewForCondPreheader);
  Value *len = getLenFromEntryBlock(F);

  assert(len && "Parameter named 'len' not found");

  Value *cmp6not207 = Builder.CreateICmpULT(
      len, ConstantInt::get(len->getType(), unroll_factor), "cmp6.not207");

  // Create conditional branch instruction
  Builder.CreateCondBr(cmp6not207, NewForCondPreheader2, ClonedForBody);

  // Find if.end basic block
  BasicBlock *ifEndBB = getBasicBlockByName(F, "if.end");
  BasicBlock *returnBB = getBasicBlockByName(F, "return");
  assert(ifEndBB && "Expected to find if.end!");
  assert(returnBB && "Expected to find return!");
  // Get the terminator instruction of if.end
  Instruction *terminator = ifEndBB->getTerminator();
  if (!terminator) {
    assert(false && "if.end basic block has no terminator instruction\n");
    return;
  }

  // Replace the first operand of the terminator instruction with
  // NewForCondPreheader
  terminator->setOperand(2, NewForCondPreheader);

  // Find the unique PHINode in clonedForBody
  PHINode *uniquePHI = nullptr;
  for (Instruction &I : *ClonedForBody) {
    if (auto *phi = dyn_cast<PHINode>(&I)) {
      if (uniquePHI) {
        // If we've already found a PHINode but find another, it's not unique

        uniquePHI = nullptr;
        break;
      }
      uniquePHI = phi;
    }
  }

  assert(uniquePHI && "No unique PHINode found in ForBody\n");

  uniquePHI->setIncomingBlock(1, NewForCondPreheader);
  auto *clonedphi = uniquePHI->clone();
  clonedphi->insertInto(NewForCondPreheader2, NewForCondPreheader2->begin());

  // Create comparison instruction
  ICmpInst *cmp85209 =
      new ICmpInst(ICmpInst::ICMP_SLT, clonedphi, len, "cmp85209");
  cmp85209->insertAfter(clonedphi);

  // Create conditional branch instruction
  BranchInst *br = BranchInst::Create(ForBody, returnBB, cmp85209);

  br->insertAfter(cmp85209);

  // Get the terminator instruction of ClonedForBody
  BranchInst *clonedTerminator =
      dyn_cast<BranchInst>(ClonedForBody->getTerminator());
  assert(clonedTerminator &&
         "ClonedForBody's terminator should be a BranchInst");
  if (!clonedTerminator) {
    assert(false && "ClonedForBody has no terminator instruction\n");
    return;
  }

  // Set the first operand of ClonedForBody's terminator to NewForCondPreheader2
  clonedTerminator->setOperand(2, NewForCondPreheader2);

  // Find the unique PHI node in ForBody
  PHINode *uniquePHI2 = nullptr;
  for (Instruction &I : *ForBody) {
    if (auto *phi = dyn_cast<PHINode>(&I)) {
      if (uniquePHI2) {
        // If we've already found a PHINode but find another, it's not unique

        uniquePHI = nullptr;
        break;
      }
      uniquePHI2 = phi;
    }
  }

  assert(uniquePHI2 && "No unique PHINode found in ForBody\n");

  uniquePHI2->setIncomingValue(1, clonedphi);
  uniquePHI2->setIncomingBlock(1, NewForCondPreheader2);

  // Find the unique PHI node in returnBB
  PHINode *returnBBPHI = nullptr;
  for (Instruction &I : *returnBB) {
    if (auto *phi = dyn_cast<PHINode>(&I)) {
      if (returnBBPHI) {
        // If we've already found a PHINode but find another, it's not unique
        returnBBPHI = nullptr;
        break;
      }
      returnBBPHI = phi;
    }
  }

  if (returnBBPHI) {
    // Add [0, NewForCondPreheader2]
    returnBBPHI->addIncoming(ConstantInt::get(returnBBPHI->getType(), 0),
                             NewForCondPreheader2);
  } else {
    assert(false && "No unique PHI node found in returnBB\n");
  }
}

static void addnoalias(Function &F) {
  for (Argument &Arg : F.args()) {
    if (Arg.getType()->isPointerTy()) {
      Arg.addAttr(Attribute::NoAlias);
    }
  }
}
static BasicBlock *cloneForBody(Function &F, BasicBlock *ForBody,
                                const std::string &Suffix) {
  ValueToValueMapTy VMap;
  BasicBlock *ClonedForBody = CloneBasicBlock(ForBody, VMap, Suffix, &F);
  VMap[ForBody] = ClonedForBody;
  remapInstructionsInBlocks({ClonedForBody}, VMap);
  return ClonedForBody;
}

static void unrollAddc(Function &F, ScalarEvolution &SE, Loop *L,
                       int unroll_factor) {

  // Get the basic block containing the function body from L
  BasicBlock *ForBody = L->getHeader();

  // Ensure that the basic block containing the function body is found
  if (!ForBody) {
    assert(ForBody && "ForBody not found");
    return;
  }

  // clone for body

  BasicBlock *ClonedForBody = cloneForBody(F, ForBody, ".modify");
  ClonedForBody->moveBefore(ForBody);

  Value *sub = unrolladdcClonedForBody(ClonedForBody, unroll_factor);

  // Find the ForCondPreheader basic block from F
  BasicBlock *ForCondPreheader = getBasicBlockByName(F, "for.cond.preheader");
  assert(ForCondPreheader && "Expected to find for.cond.preheader!");
  expandForCondPreheaderaddc(F, ForCondPreheader, ClonedForBody, ForBody, sub,
                             unroll_factor);
  runInstCombinePass(F);
  groupAndReorderInstructions(ClonedForBody);

  // Verify the function
  if (verifyFunction(F, &errs())) {
    LLVM_DEBUG(errs() << "Function verification failed\n");
    return;
  }
}

static void unrollCorr(Function &F, Loop *L, int unroll_factor) {

  // Get the basic block containing the function body from L
  BasicBlock *ForBody = L->getHeader();
  assert(ForBody && "ForBody not found");

  // clone for body
  BasicBlock *ClonedForBody = cloneForBody(F, ForBody, ".unroll");

  BasicBlock *returnBB = getBasicBlockByName(F, "return");
  assert(returnBB && "Expected to find return!");
  BasicBlock *ForCondPreheader = getBasicBlockByName(F, "for.cond.preheader");
  assert(ForCondPreheader && "Expected to find for.cond.preheader!");
  BasicBlock *ForCond11PreheaderUs = L->getLoopPreheader();
  assert(ForCond11PreheaderUs && "Expected to find for.cond.preheader!");

  ClonedForBody->moveBefore(returnBB);

  ForCondPreheader->setName("if.end");

  // Find the first instruction in ForCondPreheader
  Instruction *FirstInst = &*ForCondPreheader->begin();
  Instruction *SecondInst = FirstInst->getNextNode();
  // Ensure the first instruction is a sub nsw instruction
  if (BinaryOperator *SubInst = dyn_cast<BinaryOperator>(FirstInst)) {
    if (SubInst->getOpcode() == Instruction::Sub &&
        SubInst->hasNoSignedWrap()) {
      ;
    } else {
      assert(false && "The first instruction in ForCondPreheader is not a sub "
                      "nsw instruction\n");
    }
  } else {
    assert(false && "The first instruction in ForCondPreheader is not a binary "
                    "operation\n");
  }
  // Insert new instruction after FirstInst
  IRBuilder<> Builder(FirstInst->getNextNode());
  Value *Sub6 = Builder.CreateNSWAdd(
      FirstInst, ConstantInt::get(FirstInst->getType(), 1 - unroll_factor),
      "sub6");

  if (ICmpInst *CmpInst = dyn_cast<ICmpInst>(SecondInst)) {
    if (CmpInst->getPredicate() == ICmpInst::ICMP_EQ) {
      CmpInst->setOperand(0, FirstInst);
      CmpInst->setOperand(
          1, ConstantInt::get(FirstInst->getType(), unroll_factor - 1));
      CmpInst->setPredicate(ICmpInst::ICMP_SGT);
    }
  }
  // Create new basic blocks
  BasicBlock *ForCond11PreheaderPreheader = ForCondPreheader->getNextNode();
  BasicBlock *ForCond8PreheaderLrPh =
      BasicBlock::Create(F.getContext(), "for.cond8.preheader.lr.ph", &F,
                         ForCond11PreheaderPreheader);
  BasicBlock *ForCond8Preheader = BasicBlock::Create(
      F.getContext(), "for.cond8.preheader", &F, ForCond11PreheaderPreheader);
  BasicBlock *ForBody10LrPh = BasicBlock::Create(
      F.getContext(), "for.body10.lr.ph", &F, ForCond11PreheaderPreheader);
  BasicBlock *ForCond91Preheader = BasicBlock::Create(
      F.getContext(), "for.cond91.preheader", &F, ForCond11PreheaderPreheader);
  BasicBlock *ForCond95PreheaderLrPh =
      BasicBlock::Create(F.getContext(), "for.cond95.preheader.lr.ph", &F,
                         ForCond11PreheaderPreheader);

  // Set predecessors for the basic blocks
  ForCondPreheader->getTerminator()->setSuccessor(0, ForCond8PreheaderLrPh);
  ForCondPreheader->getTerminator()->setSuccessor(1, ForCond91Preheader);

  // Find the parameter named patlen from the function arguments
  Value *PatlenArg = F.getArg(3);
  Value *SignalArg = F.getArg(0);
  assert(PatlenArg && "Parameter named patlen not found\n");
  assert(SignalArg && "Parameter named signal not found\n");

  // Add instructions to the for.cond8.preheader.lr.ph basic block
  Builder.SetInsertPoint(ForCond8PreheaderLrPh);
  Value *Cmp9242 = Builder.CreateICmpSGT(
      PatlenArg, ConstantInt::get(PatlenArg->getType(), 0), "cmp9242");
  Builder.CreateBr(ForCond8Preheader);

  // Add instructions to the for.cond8.preheader basic block
  Builder.SetInsertPoint(ForCond8Preheader);
  PHINode *N0276 =
      Builder.CreatePHI(Type::getInt32Ty(F.getContext()), 2, "n.0276");
  N0276->addIncoming(ConstantInt::get(Type::getInt32Ty(F.getContext()), 0),
                     ForCond8PreheaderLrPh);

  // Create conditional branch instruction
  Builder.CreateCondBr(Cmp9242, ForBody10LrPh, nullptr);

  // Add instructions to the for.body10.lr.ph basic block
  Builder.SetInsertPoint(ForBody10LrPh);

  // Create getelementptr instruction
  Value *GEP =
      Builder.CreateGEP(Type::getFloatTy(F.getContext()), SignalArg, N0276, "");

  // Create unconditional branch instruction to ClonedForBody
  Builder.CreateBr(ClonedForBody);

  // Add instructions to the for.cond91.preheader basic block
  Builder.SetInsertPoint(ForCond91Preheader);

  // Create PHI node
  PHINode *N0Lcssa =
      Builder.CreatePHI(Type::getInt32Ty(F.getContext()), 2, "n.0.lcssa");
  N0Lcssa->addIncoming(ConstantInt::get(Type::getInt32Ty(F.getContext()), 0),
                       ForCondPreheader);
  // Note: [ %add89, %for.cond.cleanup ] part not added yet

  // Create comparison instruction
  Value *Cmp92Not282 =
      Builder.CreateICmpSGT(N0Lcssa, FirstInst, "cmp92.not282");

  // Create conditional branch instruction
  Builder.CreateCondBr(Cmp92Not282, returnBB, ForCond95PreheaderLrPh);

  // Add instructions to the for.cond95.preheader.lr.ph basic block
  Builder.SetInsertPoint(ForCond95PreheaderLrPh);

  Value *Cmp92678 = Builder.CreateICmpSGT(
      PatlenArg, ConstantInt::get(Type::getInt32Ty(F.getContext()), 0),
      "Cmp92678");
  // Insert Cmp92678
  Builder.CreateCondBr(Cmp92678, ForCond11PreheaderUs,
                       ForCond11PreheaderPreheader);

  Builder.SetInsertPoint(ForCond11PreheaderPreheader,
                         ForCond11PreheaderPreheader->begin());

  Instruction *ForCond11PreheaderPreheaderterminater =
      ForCond11PreheaderPreheader->getTerminator();
  Instruction *ForCond11PreheaderPreheaderFirstInst =
      &*ForCond11PreheaderPreheader->begin();
  Value *SiglenArg = ForCond11PreheaderPreheaderFirstInst->getOperand(0);
  // Calculate the result of n.0.lcssa left shifted by 2 bits
  Value *ShiftedN = Builder.CreateShl(
      N0Lcssa, ConstantInt::get(Type::getInt32Ty(F.getContext()), 2), "");

  // Create getelementptr instruction
  // Find memset function call
  CallInst *MemsetCall = getFirstCallInstWithName(ForCond11PreheaderPreheader,
                                                  "llvm.memset.p0.i32");

  // Ensure memset call is found
  assert(MemsetCall && "memset call not found");

  // Get DestArg
  Value *DestArg = MemsetCall->getArgOperand(0);

  // Create new GEP instruction
  Value *Scevgep = Builder.CreateGEP(Type::getInt8Ty(F.getContext()), DestArg,
                                     ShiftedN, "scevgep");
  MemsetCall->setOperand(0, Scevgep);
  // Calculate siglen + 1
  Value *SiglenPlus1 = Builder.CreateAdd(
      SiglenArg, ConstantInt::get(Type::getInt32Ty(F.getContext()), 1), "");

  // Calculate n.0.lcssa + patlen
  Value *NplusPatlen = Builder.CreateAdd(N0Lcssa, PatlenArg, "");

  // Calculate (siglen + 1) - (n.0.lcssa + patlen)
  Value *SubResult = Builder.CreateSub(SiglenPlus1, NplusPatlen, "");

  // Calculate the final memset length
  Value *MemsetLen = Builder.CreateShl(
      SubResult, ConstantInt::get(Type::getInt32Ty(F.getContext()), 2), "");
  Instruction *addinst = dyn_cast<Instruction>(MemsetCall->getOperand(2));
  MemsetCall->setOperand(2, MemsetLen);
  if (addinst && addinst->use_empty())
    addinst->eraseFromParent();
  if (ForCond11PreheaderPreheaderFirstInst->use_empty())
    ForCond11PreheaderPreheaderFirstInst->eraseFromParent();

  // Create a Preheader for ForCond11PreheaderUs
  BasicBlock *ForCond11PreheaderUsPreheader =
      BasicBlock::Create(F.getContext(), "for.cond11.preheader.us.preheader",
                         &F, ForCond11PreheaderUs);

  // Add an unconditional branch to ForCond11PreheaderUs in the new Preheader
  BranchInst::Create(ForCond11PreheaderUs, ForCond11PreheaderUsPreheader);

  // Insert new instructions in ForCond11PreheaderUsPreheader
  Builder.SetInsertPoint(ForCond11PreheaderUsPreheader->getTerminator());

  // Add %6 = add i32 %siglen, 1
  Value *SiglenPlus2 = Builder.CreateAdd(
      SiglenArg, ConstantInt::get(Type::getInt32Ty(F.getContext()), 1), "");

  // Add %7 = sub i32 %6, %patlen
  Value *SubResult2 = Builder.CreateSub(SiglenPlus2, PatlenArg, "");

  // Find PHI node
  PHINode *PhiNode = nullptr;
  for (PHINode &Phi : ForCond11PreheaderUs->phis()) {
    PhiNode = &Phi;
    break;
  }

  assert(PhiNode && "PHI node not found in for.cond11.preheader.us\n");

  // Modify incoming values of the PHI node
  PhiNode->setIncomingBlock(1, ForCond11PreheaderUsPreheader);
  PhiNode->setIncomingValue(1, N0Lcssa);

  BasicBlock *ForCond11ForCondCleanup13CritEdgeUs = ForBody->getNextNode();
  // Find icmp ult instruction in ForCond11ForCondCleanup13CritEdgeUs
  ICmpInst *IcmpUltInst = getLastICmpInstWithPredicate(
      ForCond11ForCondCleanup13CritEdgeUs, ICmpInst::ICMP_ULT);

  assert(IcmpUltInst && "icmp ult instruction not found in "
                        "ForCond11ForCondCleanup13CritEdgeUs\n");

  IcmpUltInst->setOperand(0, PhiNode->getIncomingValue(0));
  IcmpUltInst->setOperand(1, SubResult2);
  IcmpUltInst->setPredicate(ICmpInst::ICMP_EQ);

  swapTerminatorSuccessors(ForCond11ForCondCleanup13CritEdgeUs);

  // Find PHI nodes in ClonedForBody
  for (PHINode &Phi : ClonedForBody->phis()) {
    Phi.setIncomingBlock(0, ForBody10LrPh);
  }

  // Find phi float instruction in ClonedForBody
  PHINode *FloatPhi = getFirstFloatPhi(ClonedForBody);
  assert(FloatPhi && "phi float node not found");
  // Find getelementptr inbounds instructions in ClonedForBody
  GetElementPtrInst *GEPInst = nullptr;
  GetElementPtrInst *GEPInst2 = nullptr;
  for (auto &I : *ClonedForBody) {
    if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
      if (GEP->isInBounds()) {
        GEPInst = GEP;
      } else {
        GEPInst2 = GEP;
      }
    }
  }
  assert(GEPInst &&
         "getelementptr inbounds instruction not found in ClonedForBody\n");
  assert(GEPInst2 &&
         "getelementptr inbounds instruction not found in ClonedForBody\n");

  GEPInst2->setOperand(0, GEP);

  Instruction *loadinst = GEPInst->getNextNode();
  GEPInst->moveBefore(FloatPhi);
  loadinst->moveBefore(FloatPhi);

  if (FloatPhi) {
    // Find the llvm.fmuladd.f32 instruction
    Instruction *FMulAdd =
        getFirstCallInstWithName(ClonedForBody, "llvm.fmuladd.f32");
    assert(FMulAdd && "llvm.fmuladd.f32 instruction not found\n");
    Instruction *InsertPoint = FMulAdd->getNextNode();
    if (FMulAdd) {
      // Copy instructions unroll_factor-1 times
      for (int i = 0; i < (unroll_factor - 1); ++i) {
        ValueToValueMapTy VMap;
        for (auto It = FloatPhi->getIterator(); &*It != FMulAdd->getNextNode();
             ++It) {
          Instruction *NewInst = It->clone();
          VMap[&*It] = NewInst;
          NewInst->insertBefore(InsertPoint);
        }

        // Update operands of new instructions
        for (auto It = FloatPhi->getIterator(); &*It != FMulAdd->getNextNode();
             ++It) {
          Instruction *NewInst = cast<Instruction>(VMap[&*It]);
          for (unsigned j = 0; j < NewInst->getNumOperands(); j++) {
            Value *Op = NewInst->getOperand(j);
            if (VMap.count(Op)) {
              NewInst->setOperand(j, VMap[Op]);
            }
          }
          // If NewInst is a getelementptr instruction, set its operand 1 to i+1
          if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(NewInst)) {
            GEP->setOperand(0, GEPInst);
            GEP->setOperand(
                1, ConstantInt::get(GEP->getOperand(1)->getType(), i + 1));
            GEP->setName("arrayidx" + std::to_string(i + 1));
          }
        }
      }

    } else {
      assert(false && "llvm.fmuladd.f32 instruction not found\n");
    }
  } else {
    assert(false && "phi float instruction not found\n");
  }
  movePHINodesToTop(*ClonedForBody);
  groupAndReorderInstructions(ClonedForBody);

  // Create new basic block for.cond.cleanup
  BasicBlock *ForCondCleanup =
      BasicBlock::Create(F.getContext(), "for.cond.cleanup", &F, ClonedForBody);

  ForCond8Preheader->getTerminator()->setSuccessor(1, ForCondCleanup);
  // Create unconditional branch to ClonedForBody in for.cond.cleanup
  BranchInst::Create(ClonedForBody, ForCondCleanup);

  // Get the terminator instruction of ClonedForBody
  Instruction *Terminator = ClonedForBody->getTerminator();

  // Set the first successor of ClonedForBody to for.cond.cleanup
  if (Terminator->getNumSuccessors() > 0) {
    Terminator->setSuccessor(0, ForCondCleanup);
  }

  // Clone phi float nodes from ClonedForBody to ForCondCleanup
  int i = 0;
  for (PHINode &Phi : ClonedForBody->phis()) {
    if (Phi.getType()->isFloatTy()) {
      Instruction *newPhi = Phi.clone();
      cast<PHINode>(newPhi)->setIncomingBlock(0, ForCond8Preheader);
      newPhi->insertBefore(ForCondCleanup->getTerminator());
      if (i == 0) {
        GetElementPtrInst *arrayidx = GetElementPtrInst::Create(
            Type::getFloatTy(F.getContext()), DestArg, N0276, "arrayidx",
            ForCondCleanup->getTerminator());
        StoreInst *storeInst =
            new StoreInst(newPhi, arrayidx, ForCondCleanup->getTerminator());
      } else {
        Instruction *orInst = BinaryOperator::CreateDisjoint(
            Instruction::Or, N0276, ConstantInt::get(N0276->getType(), i),
            "add");
        orInst->insertBefore(ForCondCleanup->getTerminator());
        GetElementPtrInst *arrayidx = GetElementPtrInst::Create(
            Type::getFloatTy(F.getContext()), DestArg, orInst, "arrayidx",
            ForCondCleanup->getTerminator());

        StoreInst *storeInst =
            new StoreInst(newPhi, arrayidx, ForCondCleanup->getTerminator());
      }
      i++;
    }
  }

  // Insert new instructions at the end of ClonedForBody
  Builder.SetInsertPoint(ForCondCleanup->getTerminator());
  Value *add89 = Builder.CreateAdd(
      N0276, ConstantInt::get(N0276->getType(), unroll_factor), "add89", true,
      true);
  Value *cmp7 = Builder.CreateICmpSLT(add89, Sub6, "cmp7");

  // Get the original terminator instruction
  Instruction *OldTerminator = ForCondCleanup->getTerminator();

  // Create new conditional branch instruction
  BranchInst *NewBr =
      BranchInst::Create(ForCond8Preheader, ForCond91Preheader, cmp7);

  // Insert new branch instruction and delete the old terminator
  ReplaceInstWithInst(OldTerminator, NewBr);

  movePHINodesToTop(*ForCondCleanup);
  groupAndReorderInstructions(ForCondCleanup);

  // Update PHI nodes in for.cond8.preheader
  for (PHINode &Phi : ForCond8Preheader->phis()) {
    Phi.addIncoming(add89, ForCondCleanup);
  }

  // Update PHI nodes in for.cond91.preheader
  for (PHINode &Phi : ForCond91Preheader->phis()) {
    Phi.addIncoming(add89, ForCondCleanup);
  }

  // Iterate through all PHI nodes in returnBB
  for (PHINode &Phi : returnBB->phis()) {
    // Add new incoming value for each PHI node
    Phi.addIncoming(ConstantInt::get(Type::getInt32Ty(F.getContext()), 0),
                    ForCond91Preheader);
  }
  // for.cond95.preheader.lr.ph -> for.cond11.preheader.us.preheader
  ForCond95PreheaderLrPh->getTerminator()->setSuccessor(
      0, ForCond11PreheaderUsPreheader);
}

static bool checkIfDotProdSimplest(Function &F) {
  bool flag = false;

  if (F.size() == 3) {
    BasicBlock *entryBB = getBasicBlockByName(F, "entry");
    BasicBlock *forCondCleanup = getBasicBlockByName(F, "for.cond.cleanup");
    BasicBlock *forBody = getBasicBlockByName(F, "for.body");
    if (entryBB && forCondCleanup && forBody) {
      CallInst *fmuladd = getFirstCallInstWithName(forBody, "llvm.fmuladd.f32");
      if (fmuladd) {
        if (forBody->getTerminator()->getSuccessor(0) == forCondCleanup &&
            forBody->getTerminator()->getSuccessor(1) == forBody) {
          if (entryBB->getTerminator()->getSuccessor(0) == forBody) {
            flag = true;
          }
        }
      }
    }
  }
  return flag;
}
// for dotprod, llvm.fmuladd.f32 is in for.body
static bool checkIfDotProdComplicated(Function &F) {
  bool flag1 = false;
  bool flag2 = false;
  bool flag3 = false;
  if (F.size() == 3) {
    BasicBlock *entryBB = getBasicBlockByName(F, "entry");
    BasicBlock *forCondCleanup = getBasicBlockByName(F, "for.cond.cleanup");
    BasicBlock *forBody = getBasicBlockByName(F, "for.body");
    if (entryBB && forCondCleanup && forBody) {
      CallInst *fmuladd = getFirstCallInstWithName(forBody, "llvm.fmuladd.f32");
      if (fmuladd) {

        if (forBody->getTerminator()->getSuccessor(0) == forCondCleanup &&
            forBody->getTerminator()->getSuccessor(1) == forBody) {
          if (entryBB->getTerminator()->getSuccessor(0) == forBody) {
            flag1 = true;
          }
        }
      }
    }
    if (forBody) {
      for (Instruction &I : *forBody) {
        if (auto *BinOp = dyn_cast<BinaryOperator>(&I)) {
          if (BinOp->getOpcode() == Instruction::FAdd ||
              BinOp->getOpcode() == Instruction::FMul ||
              BinOp->getOpcode() == Instruction::FSub ||
              BinOp->getOpcode() == Instruction::FDiv) {
            flag2 = true;
          }
        }
      }

      // Check if forBody has exactly one float PHI node
      int floatPhiCount = 0;
      for (PHINode &Phi : forBody->phis()) {
        if (Phi.getType()->isFloatTy()) {
          floatPhiCount++;
        }
      }
      if (floatPhiCount == 1) {
        flag3 = true;
      }
    }
  }

  return flag1 && flag2 && flag3;
}
static bool shouldUnrollLoopWithCount(Function &F, Loop *L,
                                      ScalarEvolution &SE) {
  if (!checkIfDotProdSimplest(F)) {
    return false;
  }
  // Check if the loop is suitable for unrolling
  if (!L->getLoopLatch())
    return false;
  if (!L->getExitingBlock())
    return false;

  // Check if the loop count is fixed and appropriate, loop count is constant
  const SCEV *TripCount = SE.getBackedgeTakenCount(L);
  if (isa<SCEVConstant>(TripCount)) {
    // More condition checks can be added here
    return true;
  }
  return false;
}

static void
insertPhiNodesForFMulAdd(BasicBlock *LoopHeader, BasicBlock *LoopPreheader,
                         SmallVector<CallInst *, 16> &FMulAddCalls) {
  // Collect all tail call float @llvm.fmuladd.f32 in LoopHeader
  for (Instruction &I : *LoopHeader) {
    if (CallInst *CI = dyn_cast<CallInst>(&I)) {
      if (Function *F = CI->getCalledFunction()) {
        if (F->getName() == "llvm.fmuladd.f32" && CI->isTailCall()) {
          FMulAddCalls.push_back(CI);
        }
      }
    }
  }

  // Insert phi nodes for each FMulAdd call
  for (CallInst *CI : FMulAddCalls) {
    // Create new phi node
    PHINode *PHI =
        PHINode::Create(CI->getType(), 2, CI->getName() + ".phi", CI);

    // Set incoming values for phi node
    PHI->addIncoming(ConstantFP::get(CI->getType(), 0), LoopPreheader);
    PHI->addIncoming(CI, LoopHeader);

    CI->setOperand(2, PHI);
  }
}

static void postUnrollLoopWithCount(Function &F, Loop *L, int unroll_count) {
  BasicBlock *LoopHeader = L->getHeader();
  BasicBlock *LoopPreheader = L->getLoopPreheader();
  // Collect all tail call float @llvm.fmuladd.f32 in LoopHeader
  SmallVector<CallInst *, 16> FMulAddCalls;
  insertPhiNodesForFMulAdd(LoopHeader, LoopPreheader, FMulAddCalls);

  movePHINodesToTop(*LoopHeader);
  runInstCombinePass(F);
  groupAndReorderInstructions(LoopHeader);

  // Create for.end basic block after LoopHeader
  ICmpInst *LastICmp = getLastInst<ICmpInst>(LoopHeader);
  LastICmp->setPredicate(ICmpInst::ICMP_ULT);
  // Get the first operand of LastICmp
  Value *Operand1 = LastICmp->getOperand(1);

  // Directly set the first operand of LastICmp to a new constant value
  LastICmp->setOperand(
      1, ConstantInt::get(Operand1->getType(),
                          dyn_cast<ConstantInt>(Operand1)->getSExtValue() -
                              (2 * unroll_count - 1)));
  LastICmp->setName("cmp");

  swapTerminatorSuccessors(LoopHeader);

  // After swapping, succ 0 is LoopHeader, succ 1 is returnBB
  BasicBlock *ExitingBlock = L->getExitBlock();
  ExitingBlock->setName("for.end");

  // Get ret instruction in ExitingBlock
  ReturnInst *RetInst = dyn_cast<ReturnInst>(ExitingBlock->getTerminator());
  if (!RetInst) {
    assert(false && "ret instruction not found\n");
    return;
  }

  // Get the original return value
  Value *OriginalRetValue = RetInst->getOperand(0);

  // Create IRBuilder, set insertion point before ret instruction
  IRBuilder<> Builder(RetInst);

  // Create a series of fadd instructions
  Value *CurrentSum = OriginalRetValue;
  Value *add37 = Builder.CreateFAdd(FMulAddCalls[1], CurrentSum, "add37");
  Value *add38 = Builder.CreateFAdd(FMulAddCalls[2], FMulAddCalls[3], "add38");
  Value *add39 = Builder.CreateFAdd(FMulAddCalls[4], FMulAddCalls[5], "add39");
  Value *add40 = Builder.CreateFAdd(FMulAddCalls[6], FMulAddCalls[7], "add40");
  Value *add41 = Builder.CreateFAdd(add37, add38, "add41");
  Value *add42 = Builder.CreateFAdd(add39, add40, "add42");
  CurrentSum = Builder.CreateFAdd(add41, add42, "add43");

  // Replace the original ret instruction
  RetInst->setOperand(0, CurrentSum);

  // Verify function
  if (verifyFunction(F, &errs())) {
    LLVM_DEBUG(errs() << "Function verification failed\n");
    return;
  }
}

static bool shouldUnrollComplexLoop(Function &F, Loop *L, ScalarEvolution &SE,
                                    DominatorTree &DT, LoopInfo &LI) {
  if (!checkIfDotProdComplicated(F)) {
    return false;
  }
  // Check if the loop is suitable for unrolling
  if (!L->getLoopLatch())
    return false;
  if (!L->getExitingBlock())
    return false;

  if (L->getCanonicalInductionVariable())
    return false;
  // Check if the loop count is fixed and appropriate, loop count is constant
  BasicBlock *LoopPreheader = L->getLoopPreheader();
  // Get the start value of the loop
  if (LoopPreheader) {
    return false;
  }

  BasicBlock *LoopHeader = L->getHeader();
  BasicBlock *NewPreheader =
      BasicBlock::Create(LoopHeader->getContext(), "for.cond.preheader",
                         LoopHeader->getParent(), LoopHeader);
  // Redirect all external predecessors to the new preheader basic block
  for (BasicBlock *pred : predecessors(LoopHeader)) {
    if (!L->contains(pred)) {
      pred->getTerminator()->replaceUsesOfWith(LoopHeader, NewPreheader);
      // Update PHI nodes in the loop header to point to the new preheader basic
      // block
      for (PHINode &PN : LoopHeader->phis()) {
        int Index = PN.getBasicBlockIndex(pred);
        if (Index != -1) {
          PN.setIncomingBlock(Index, NewPreheader);
        }
      }
    }
  }
  // Jump from the new preheader to the loop header
  BranchInst::Create(LoopHeader, NewPreheader);
  return true;
}

static bool shouldUnrollAddcType(Function &F, LoopInfo *LI) {
  // Check the number of basic blocks
  if (F.size() != 6)
    return false;

  // Check the loop nesting level
  unsigned int maxLoopDepth = 0;
  for (auto &BB : F) {
    maxLoopDepth = std::max(maxLoopDepth, LI->getLoopDepth(&BB));
  }
  if (maxLoopDepth != 1) {
    return false;
  }

  BasicBlock *Entry = getBasicBlockByName(F, "entry");
  BasicBlock *IfEnd = getBasicBlockByName(F, "if.end");
  BasicBlock *ForCondPreheader = getBasicBlockByName(F, "for.cond.preheader");
  BasicBlock *ForBody = getBasicBlockByName(F, "for.body");
  BasicBlock *ForBodyClone = getBasicBlockByName(F, "for.body.clone");
  BasicBlock *Return = getBasicBlockByName(F, "return");

  if (!Entry || !IfEnd || !ForCondPreheader || !ForBody || !ForBodyClone ||
      !Return)
    return false;

  if (Entry->getTerminator()->getSuccessor(0) != Return ||
      Entry->getTerminator()->getSuccessor(1) != IfEnd ||
      IfEnd->getTerminator()->getSuccessor(0) != ForBody ||
      IfEnd->getTerminator()->getSuccessor(1) != ForCondPreheader ||
      ForCondPreheader->getTerminator()->getSuccessor(0) != ForBodyClone ||
      ForCondPreheader->getTerminator()->getSuccessor(1) != Return ||
      ForBody->getTerminator()->getSuccessor(0) != Return ||
      ForBody->getTerminator()->getSuccessor(1) != ForBody ||
      ForBodyClone->getTerminator()->getSuccessor(0) != Return ||
      ForBodyClone->getTerminator()->getSuccessor(1) != ForBodyClone)
    return false;

  // Check if there are three outer loops, each with one inner loop
  int outerLoopCount = 0;
  int innerLoopCount = 0;
  for (Loop *L : LI->getLoopsInPreorder()) {
    if (L->getLoopDepth() == 1) {
      outerLoopCount++;
      if (L->getSubLoops().size() == 1) {
        innerLoopCount++;
      }
    }
  }

  if (outerLoopCount != 2 || innerLoopCount != 0) {
    return false;
  }

  return true;
}

static bool shouldUnrollDotprodType(Function &F, LoopInfo *LI) {
  // Check the number of basic blocks
  if (F.size() != 5)
    return false;

  // Check the loop nesting level
  unsigned int maxLoopDepth = 0;
  for (auto &BB : F) {
    maxLoopDepth = std::max(maxLoopDepth, LI->getLoopDepth(&BB));
  }
  if (maxLoopDepth != 1) {
    return false;
  }

  BasicBlock *Entry = getBasicBlockByName(F, "entry");
  BasicBlock *ForCondPreheader = getBasicBlockByName(F, "for.cond.preheader");
  BasicBlock *IfEnd = getBasicBlockByName(F, "if.end");
  BasicBlock *ForBody = getBasicBlockByName(F, "for.body");
  BasicBlock *ForBodyClone = getBasicBlockByName(F, "for.body.clone");

  if (!Entry || !IfEnd || !ForCondPreheader || !ForBody || !ForBodyClone)
    return false;

  if (Entry->getTerminator()->getSuccessor(0) != ForBody ||
      Entry->getTerminator()->getSuccessor(1) != ForCondPreheader ||
      ForCondPreheader->getTerminator()->getSuccessor(0) != ForBodyClone ||
      ForCondPreheader->getTerminator()->getSuccessor(1) != IfEnd ||
      ForBody->getTerminator()->getSuccessor(0) != IfEnd ||
      ForBody->getTerminator()->getSuccessor(1) != ForBody ||
      ForBodyClone->getTerminator()->getSuccessor(0) != IfEnd ||
      ForBodyClone->getTerminator()->getSuccessor(1) != ForBodyClone)
    return false;

  // Check if there are three outer loops, each with one inner loop
  int outerLoopCount = 0;
  int innerLoopCount = 0;
  for (Loop *L : LI->getLoopsInPreorder()) {
    if (L->getLoopDepth() == 1) {
      outerLoopCount++;
      if (L->getSubLoops().size() == 1) {
        innerLoopCount++;
      }
    }
  }

  if (outerLoopCount != 2 || innerLoopCount != 0) {
    return false;
  }

  return true;
}

static std::pair<Value *, Value *> modifyEntryBB(BasicBlock &entryBB) {
  ICmpInst *icmp = getLastInst<ICmpInst>(&entryBB);
  assert(icmp && "icmp not found");
  Value *start_index = icmp->getOperand(0);
  Value *end_index = icmp->getOperand(1);
  // Insert new instructions before icmp
  IRBuilder<> Builder(icmp);
  Value *sub = Builder.CreateNSWAdd(
      end_index, ConstantInt::get(end_index->getType(), -8), "sub");
  icmp->setOperand(0, sub);
  icmp->setOperand(1, start_index);
  return std::make_pair(sub, end_index);
}

static void postUnrollLoopWithVariable(Function &F, Loop *L, int unroll_count) {
  BasicBlock *LoopPreheader = L->getLoopPreheader();
  // Get the basic blocks to merge
  SmallVector<BasicBlock *> BBsToMerge;
  BasicBlock *ForBody1 = getBasicBlockByName(F, "for.body.1");
  BasicBlock *ForBody2 = getBasicBlockByName(F, "for.body.2");
  BasicBlock *ForBody3 = getBasicBlockByName(F, "for.body.3");
  BasicBlock *ForBody4 = getBasicBlockByName(F, "for.body.4");
  BasicBlock *ForBody5 = getBasicBlockByName(F, "for.body.5");
  BasicBlock *ForBody6 = getBasicBlockByName(F, "for.body.6");
  BasicBlock *ForBody7 = getBasicBlockByName(F, "for.body.7");
  assert(ForBody1 && ForBody2 && ForBody3 && ForBody4 && ForBody5 && ForBody6 &&
         ForBody7 && "basic block not found");
  BBsToMerge.push_back(ForBody1);
  BBsToMerge.push_back(ForBody2);
  BBsToMerge.push_back(ForBody3);
  BBsToMerge.push_back(ForBody4);
  BBsToMerge.push_back(ForBody5);
  BBsToMerge.push_back(ForBody6);
  BBsToMerge.push_back(ForBody7);

  BasicBlock *LoopHeader = L->getHeader();
  BasicBlock *LoopHeaderClone =
      cloneBasicBlockWithRelations(LoopHeader, ".clone", &F);
  LoopHeaderClone->moveAfter(LoopHeader);
  // Create a new basic block as for.end
  BasicBlock *ForEnd = getBasicBlockByName(F, "for.cond.cleanup");
  assert(ForEnd && "basic block not found");
  ForEnd->setName("for.end");

  LoopHeaderClone->getTerminator()->setSuccessor(1, LoopHeaderClone);
  for (PHINode &Phi : LoopHeaderClone->phis()) {
    Phi.setIncomingBlock(1, LoopHeaderClone);
  }

  for (BasicBlock *BB : BBsToMerge) {
    MergeBasicBlockIntoOnlyPred(BB);
  }

  // Adjust positions
  LoopHeaderClone->moveAfter(getBasicBlockByName(F, "for.body.7"));
  assert(LoopHeaderClone && "basic block not found");
  ForEnd->moveAfter(LoopHeaderClone);

  BasicBlock &entryBB = F.getEntryBlock();
  auto [Sub, end_index] = modifyEntryBB(entryBB);
  entryBB.getTerminator()->setSuccessor(1, ForBody7);

  SmallVector<Instruction *> FAMSDInsts;
  for (Instruction &I : *ForBody7) {
    if (auto *BinOp = dyn_cast<BinaryOperator>(&I)) {
      if (BinOp->getOpcode() == Instruction::FAdd ||
          BinOp->getOpcode() == Instruction::FMul ||
          BinOp->getOpcode() == Instruction::FSub ||
          BinOp->getOpcode() == Instruction::FDiv) {
        FAMSDInsts.push_back(BinOp);
      }
    }
  }
  assert(!FAMSDInsts.empty() && "fadd/fmul/fsub/fdiv instruction not found");
  PHINode *firstFloatPhi = getFirstFloatPhi(ForBody7);
  assert(firstFloatPhi && "phi node not found");
  // Clone phi node 7 times
  for (int i = 0; i < 7; i++) {
    PHINode *clonedPhi = cast<PHINode>(firstFloatPhi->clone());
    clonedPhi->setName("result" + Twine(i));
    clonedPhi->insertAfter(firstFloatPhi);
    auto *temp = FAMSDInsts[i];
    clonedPhi->setIncomingValue(1, temp);
    temp->setOperand(0, clonedPhi);
  }

  for (PHINode &Phi : ForBody7->phis()) {
    Phi.setIncomingBlock(0, &entryBB);
    auto *temp = Phi.clone();
    temp->setName("result0.0.lcssa");
    temp->insertBefore(LoopPreheader->getTerminator());
  }

  ICmpInst *lastICmp = getLastInst<ICmpInst>(ForBody7);
  assert(lastICmp && "icmp not found");
  lastICmp->setOperand(1, Sub);
  lastICmp->setPredicate(ICmpInst::ICMP_SLT);

  ForBody7->getTerminator()->setSuccessor(0, LoopPreheader);
  ForBody7->getTerminator()->setSuccessor(1, ForBody7);

  PHINode *firstI32Phi = getFirstI32Phi(LoopPreheader);
  assert(firstI32Phi && "phi node not found");
  // Insert icmp slt instruction in LoopPreheader
  IRBuilder<> Builder(LoopPreheader->getTerminator());
  ICmpInst *NewICmp =
      cast<ICmpInst>(Builder.CreateICmpSLT(firstI32Phi, end_index, "cmp"));

  // Convert the original unconditional branch to a conditional branch
  BranchInst *OldBr = cast<BranchInst>(LoopPreheader->getTerminator());
  BranchInst *NewBr = BranchInst::Create(LoopHeaderClone, ForEnd, NewICmp);
  ReplaceInstWithInst(OldBr, NewBr);

  Instruction *faddInst = nullptr;
  Instruction *addNswInst = nullptr;

  for (auto &I : *LoopHeaderClone) {
    if (auto *BinOp = dyn_cast<BinaryOperator>(&I)) {
      if ((BinOp->getOpcode() == Instruction::FAdd ||
           BinOp->getOpcode() == Instruction::FMul ||
           BinOp->getOpcode() == Instruction::FSub ||
           BinOp->getOpcode() == Instruction::FDiv) &&
          BinOp->getType()->isFloatTy()) {
        faddInst = BinOp;
      } else if (BinOp->getOpcode() == Instruction::Add &&
                 BinOp->hasNoSignedWrap()) {
        addNswInst = BinOp;
      }
    }

    if (faddInst && addNswInst) {
      break;
    }
  }
  assert(faddInst && addNswInst &&
         "fadd/fmul/fsub/fdiv float and add nsw instructions not found");
  PHINode *firstI32PhiLoopHeaderClone = getFirstI32Phi(LoopHeaderClone);
  assert(firstI32PhiLoopHeaderClone && "phi node not found");
  firstI32PhiLoopHeaderClone->setIncomingValue(0, firstI32Phi);
  firstI32PhiLoopHeaderClone->setIncomingValue(1, addNswInst);

  PHINode *firstFloatPhiLoopHeaderClone = getFirstFloatPhi(LoopHeaderClone);
  assert(firstFloatPhiLoopHeaderClone && "phi node not found");
  PHINode *lastFloatPhiLoopPreheader = getLastFloatPhi(LoopPreheader);
  assert(lastFloatPhiLoopPreheader && "phi node not found");
  firstFloatPhiLoopHeaderClone->setIncomingValue(0, lastFloatPhiLoopPreheader);
  firstFloatPhiLoopHeaderClone->setIncomingValue(1, faddInst);

  // Collect all phi float instructions in LoopPreheader
  SmallVector<PHINode *> floatPhis;
  for (auto &I : *LoopPreheader) {
    if (auto *Phi = dyn_cast<PHINode>(&I)) {
      if (Phi->getType()->isFloatTy()) {
        floatPhis.push_back(Phi);
      }
    }
  }

  // Get the ret instruction in ExitingBlock
  ReturnInst *RetInst = dyn_cast<ReturnInst>(ForEnd->getTerminator());
  if (!RetInst) {
    assert(false && "ret instruction not found in ExitingBlock");
    return;
  }

  // Get the original return value
  Value *OriginalRetValue = RetInst->getOperand(0);

  // Create IRBuilder, set insertion point before the ret instruction

  Builder.SetInsertPoint(RetInst);
  // Create a series of fadd instructions
  assert(floatPhis.size() == 8 && "expected floatPhis has 8 phi node");
  Value *CurrentSum = nullptr;
  Value *add64 = Builder.CreateFAdd(floatPhis[0], OriginalRetValue, "add64");
  Value *add65 = Builder.CreateFAdd(floatPhis[1], floatPhis[2], "add65");
  Value *add66 = Builder.CreateFAdd(floatPhis[3], floatPhis[4], "add66");
  Value *add67 = Builder.CreateFAdd(floatPhis[5], floatPhis[6], "add67");
  Value *add68 = Builder.CreateFAdd(add64, add65, "add68");
  Value *add69 = Builder.CreateFAdd(add66, add67, "add69");
  CurrentSum = Builder.CreateFAdd(add68, add69, "add70");

  // Replace the original ret instruction
  RetInst->setOperand(0, CurrentSum);
  PHINode *firstFloatPhiForEnd = getFirstFloatPhi(ForEnd);
  assert(firstFloatPhiForEnd && "phi node not found");
  // Remove existing incoming values from firstFloatPhiForEnd
  while (firstFloatPhiForEnd->getNumIncomingValues() > 0) {
    firstFloatPhiForEnd->removeIncomingValue(0u, false);
  }
  // Add two incoming values to firstFloatPhiForEnd
  firstFloatPhiForEnd->addIncoming(faddInst, LoopHeaderClone);
  firstFloatPhiForEnd->addIncoming(lastFloatPhiLoopPreheader, LoopPreheader);

  runDeadCodeElimination(F);
}

static bool shouldUnrollCorr(Function &F, LoopInfo *LI) {
  if (F.size() != 7)
    return false;

  BasicBlock *Entry = getBasicBlockByName(F, "entry");
  BasicBlock *ForCondPreheader = getBasicBlockByName(F, "for.cond.preheader");
  BasicBlock *Return = getBasicBlockByName(F, "return");

  if (!Entry || !ForCondPreheader || !Return)
    return false;

  if (Entry->getTerminator()->getSuccessor(0) != Return ||
      Entry->getTerminator()->getSuccessor(1) != ForCondPreheader) {
    return false;
  }

  // Feature 2: Has 5 parameters
  if (F.arg_size() != 5) {
    return false;
  }

  unsigned int loopNestLevel = 0;
  for (auto &BB : F) {
    if (isa<BranchInst>(BB.getTerminator())) {
      loopNestLevel = std::max(loopNestLevel, LI->getLoopDepth(&BB));
    }
  }
  if (loopNestLevel != 2) {
    return false;
  }

  bool hasFMulAdd = false;
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (RecurrenceDescriptor::isFMulAddIntrinsic(&I)) {
        hasFMulAdd = true;
        break;
      }
    }
    if (hasFMulAdd)
      break;
  }
  if (!hasFMulAdd) {
    return false;
  }

  return true;
}

static bool shouldUnrollConvccorr(Function &F, LoopInfo *LI) {
  // Check the number of basic blocks
  if (F.size() != 17)
    return false;

  // Check the number of parameters
  if (F.arg_size() != 5) {
    return false;
  }

  // Check the loop nesting level
  unsigned int maxLoopDepth = 0;
  for (auto &BB : F) {
    maxLoopDepth = std::max(maxLoopDepth, LI->getLoopDepth(&BB));
  }
  if (maxLoopDepth != 2) {
    return false;
  }

  // Check if the fmuladd.f32 inline function is used
  bool hasFMulAdd = false;
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (RecurrenceDescriptor::isFMulAddIntrinsic(&I)) {
        hasFMulAdd = true;
        break;
      }
    }
    if (hasFMulAdd)
      break;
  }
  if (!hasFMulAdd) {
    return false;
  }

  BasicBlock *Entry = getBasicBlockByName(F, "entry");
  BasicBlock *ForBody = getBasicBlockByName(F, "for.body");
  BasicBlock *ForEnd = getBasicBlockByName(F, "for.end");
  BasicBlock *Return = getBasicBlockByName(F, "return");

  if (!Entry || !ForBody || !ForEnd || !Return)
    return false;

  if (Entry->getTerminator()->getSuccessor(0) != Return ||
      ForEnd->getTerminator()->getSuccessor(1) != ForBody)
    return false;

  // Check if there are three outer loops, each with one inner loop
  int outerLoopCount = 0;
  int innerLoopCount = 0;
  for (Loop *L : LI->getLoopsInPreorder()) {
    if (L->getLoopDepth() == 1) {
      outerLoopCount++;
      if (L->getSubLoops().size() == 1) {
        innerLoopCount++;
      }
    }
  }

  if (outerLoopCount != 3 || innerLoopCount != 3) {
    return false;
  }

  // Check if there are three icmp eq instructions in the entry basic block
  int icmpEqCount = 0;
  for (auto &I : *Entry) {
    if (auto *ICmp = dyn_cast<ICmpInst>(&I)) {
      if (ICmp->getPredicate() == ICmpInst::ICMP_EQ) {
        icmpEqCount++;
      }
    }
  }

  if (icmpEqCount != 3) {
    return false;
  }

  return true;
}

static bool shouldUnrollFird(Function &F, LoopInfo *LI) {

  // Check the number of basic blocks
  if (F.size() != 14)
    return false;

  // Check the number of parameters
  if (F.arg_size() != 4) {
    return false;
  }

  // Check the loop nesting level
  unsigned int maxLoopDepth = 0;
  for (auto &BB : F) {
    maxLoopDepth = std::max(maxLoopDepth, LI->getLoopDepth(&BB));
  }
  if (maxLoopDepth != 2) {
    return false;
  }

  // Check if the fmuladd.f32 inline function is used
  bool hasFMulAdd = false;
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (RecurrenceDescriptor::isFMulAddIntrinsic(&I)) {
        hasFMulAdd = true;
        break;
      }
    }
    if (hasFMulAdd)
      break;
  }
  if (!hasFMulAdd) {
    return false;
  }

  BasicBlock *Entry = getBasicBlockByName(F, "entry");
  BasicBlock *ForCondCleanup = getBasicBlockByName(F, "for.cond.cleanup");

  if (!Entry || !ForCondCleanup)
    return false;

  if (Entry->getTerminator()->getSuccessor(1) != ForCondCleanup)
    return false;

  // Check if there are three outer loops, each with one inner loop
  int outerLoopCount = 0;
  int innerLoopCount = 0;
  for (Loop *L : LI->getLoopsInPreorder()) {
    if (L->getLoopDepth() == 1) {
      outerLoopCount++;
    } else if (L->getLoopDepth() == 2) {
      innerLoopCount++;
    } else {
      return false;
    }
  }

  if (outerLoopCount != 1 || innerLoopCount != 3) {
    return false;
  }

  return true;
}

static bool shouldUnrollFirType(Function &F, LoopInfo *LI) {
  // Check the number of basic blocks
  if (F.size() != 19)
    return false;

  // Check the number of parameters
  if (F.arg_size() != 4) {
    return false;
  }

  // Check the loop nesting level
  unsigned int maxLoopDepth = 0;
  for (auto &BB : F) {
    maxLoopDepth = std::max(maxLoopDepth, LI->getLoopDepth(&BB));
  }
  if (maxLoopDepth != 2) {
    return false;
  }

  // Check if the fmuladd.f32 inline function is used
  bool hasFMulAdd = false;
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (RecurrenceDescriptor::isFMulAddIntrinsic(&I)) {
        hasFMulAdd = true;
        break;
      }
    }
    if (hasFMulAdd)
      break;
  }
  if (!hasFMulAdd) {
    return false;
  }

  BasicBlock *Entry = getBasicBlockByName(F, "entry");
  BasicBlock *ForCondPreheader = getBasicBlockByName(F, "for.cond.preheader");
  BasicBlock *ForBodyLrPh = getBasicBlockByName(F, "for.body.lr.ph");
  BasicBlock *IfEnd = getBasicBlockByName(F, "if.end");
  BasicBlock *ForBody = getBasicBlockByName(F, "for.body");
  BasicBlock *ForBodyClone = getBasicBlockByName(F, "for.body.clone");
  BasicBlock *ForBodyLrPhClone = getBasicBlockByName(F, "for.body.lr.ph.clone");

  if (!Entry || !ForCondPreheader || !ForBodyLrPh || !IfEnd || !ForBody ||
      !ForBodyClone || !ForBodyLrPhClone)
    return false;

  if (Entry->getTerminator()->getSuccessor(0) != ForCondPreheader ||
      Entry->getTerminator()->getSuccessor(1) != ForBodyLrPhClone ||
      ForCondPreheader->getTerminator()->getSuccessor(0) != ForBodyLrPh ||
      ForCondPreheader->getTerminator()->getSuccessor(1) != IfEnd ||
      ForBodyLrPh->getSingleSuccessor() != ForBody ||
      ForBodyLrPhClone->getSingleSuccessor() != ForBodyClone)
    return false;

  // Check if there are three outer loops, each with one inner loop
  int outerLoopCount = 0;
  int innerLoopCount = 0;
  for (Loop *L : LI->getLoopsInPreorder()) {
    if (L->getLoopDepth() == 1) {
      outerLoopCount++;
    } else if (L->getLoopDepth() == 2) {
      innerLoopCount++;
    } else {
      return false;
    }
  }
  // for opt is 4, for clang is 2.
  if (outerLoopCount != 2 || (innerLoopCount != 2 && innerLoopCount != 4)) {
    return false;
  }

  return true;
}

static void eraseAllStoreInstInBB(BasicBlock *BB) {
  assert(BB && "BasicBlock is nullptr");
  // Erase all store instructions in BB
  for (auto it = BB->begin(); it != BB->end();) {
    if (isa<StoreInst>(&*it)) {
      it = it->eraseFromParent();
    } else {
      ++it;
    }
  }
}

static GetElementPtrInst *getUniqueGetElementPtrInst(BasicBlock *BB) {
  assert(BB && "BasicBlock is nullptr");
  // Get the unique getelementptr instruction in BB
  GetElementPtrInst *GEP = nullptr;
  for (Instruction &I : *BB) {
    if (auto *GEPI = dyn_cast<GetElementPtrInst>(&I)) {
      if (!GEP) {
        GEP = GEPI;
      } else {
        // If multiple getelementptr instructions are found, set GEP to nullptr
        // and exit the loop
        GEP = nullptr;
        break;
      }
    }
  }
  assert(GEP && "getelementptr instruction not found");
  return GEP;
}

static void createCriticalEdgeAndMoveStoreInst(BasicBlock *CloneForBody,
                                               BasicBlock *ForEnd37) {
  CloneForBody->getTerminator()->setSuccessor(1, CloneForBody);
  // Create a new BasicBlock: for.cond.for.end_crit_edge
  BasicBlock *CriticalEdge = BasicBlock::Create(
      CloneForBody->getContext(), "for.cond.for.end_crit_edge",
      CloneForBody->getParent(), ForEnd37);

  // Update the terminator instruction of CloneForBody
  CloneForBody->getTerminator()->setSuccessor(0, CriticalEdge);

  // Create an unconditional branch instruction to jump to OldForEnd
  BranchInst::Create(ForEnd37, CriticalEdge);

  // Find and move the StoreInst in CloneForBody to CriticalEdge
  StoreInst *StoreToMove = nullptr;
  for (auto &Inst : *CloneForBody) {
    if (auto *Store = dyn_cast<StoreInst>(&Inst)) {
      StoreToMove = Store;
      break;
    }
  }

  if (StoreToMove) {
    StoreToMove->removeFromParent();
    StoreToMove->insertBefore(CriticalEdge->getTerminator());
  }
}
static std::tuple<Value *, GetElementPtrInst *, Value *>
modifyOuterLoop4(Loop *L, BasicBlock *ForBodyMerged,
                 BasicBlock *CloneForBodyPreheader) {
  BasicBlock *BB = L->getHeader();
  PHINode *phi = getLastInst<PHINode>(BB);
  // Add new instructions
  IRBuilder<> Builder(BB);
  Builder.SetInsertPoint(phi->getNextNode());

  // and i32 %n.0551, -8
  Value *Add2 = Builder.CreateAnd(phi, ConstantInt::get(phi->getType(), -8));

  // %sub = and i32 %n.0551, 2147483644
  Value *Sub =
      Builder.CreateAnd(phi, ConstantInt::get(phi->getType(), 2147483640));

  // %cmp12538.not = icmp eq i32 %sub, 0
  Value *Cmp = Builder.CreateICmpEQ(Sub, ConstantInt::get(phi->getType(), 0));

  // br i1 %cmp12538.not, label %for.cond.cleanup, label %for.body.preheader
  // Move the conditional branch instruction to the end of BB
  auto *newcondBr =
      Builder.CreateCondBr(Cmp, CloneForBodyPreheader, ForBodyMerged);

  // Erase the terminator instruction of BB
  Instruction *oldTerminator = BB->getTerminator();
  newcondBr->moveAfter(oldTerminator);
  oldTerminator->eraseFromParent();

  // Erase all store instructions in BB
  eraseAllStoreInstInBB(BB);
  for (PHINode &Phi : ForBodyMerged->phis()) {
    Phi.setIncomingBlock(1, CloneForBodyPreheader);
  }
  // Get the unique getelementptr instruction in BB
  GetElementPtrInst *GEP = getUniqueGetElementPtrInst(BB);
  return std::make_tuple(Sub, GEP, Add2);
}

static void modifyInnerLoop4(Loop *L, BasicBlock *ForBodyMerged, Value *Sub,
                             BasicBlock *CloneForBody, GetElementPtrInst *GEP,
                             Value *Add2, BasicBlock *CloneForBodyPreheader) {
  BasicBlock *OuterBB = L->getHeader();
  SmallVector<CallInst *, 16> FMulAddCalls;
  insertPhiNodesForFMulAdd(ForBodyMerged, OuterBB, FMulAddCalls);
  movePHINodesToTop(*ForBodyMerged);

  groupAndReorderInstructions(ForBodyMerged);
  ICmpInst *LastICmp = getLastInst<ICmpInst>(ForBodyMerged);
  LastICmp->setPredicate(ICmpInst::ICMP_ULT);
  LastICmp->setOperand(1, Sub);
  swapTerminatorSuccessors(ForBodyMerged);
  eraseAllStoreInstInBB(ForBodyMerged);

  Function *F = ForBodyMerged->getParent();

  BasicBlock *NewForEnd =
      BasicBlock::Create(F->getContext(), "for.end", F, ForBodyMerged);
  NewForEnd->moveAfter(ForBodyMerged);

  // Create an instruction to add the results of four FMulAdd calls
  assert(FMulAddCalls.size() == 8 && "Expected 8 FMulAdd calls");
  Value *Sum = nullptr;
  Value *sum = BinaryOperator::CreateFAdd(FMulAddCalls[0], FMulAddCalls[1],
                                          "sum", NewForEnd);
  Value *sum23 = BinaryOperator::CreateFAdd(FMulAddCalls[2], FMulAddCalls[3],
                                            "sum23", NewForEnd);
  Value *sum24 = BinaryOperator::CreateFAdd(FMulAddCalls[4], FMulAddCalls[5],
                                            "sum24", NewForEnd);
  Value *sum25 = BinaryOperator::CreateFAdd(FMulAddCalls[6], FMulAddCalls[7],
                                            "sum25", NewForEnd);
  Value *sum26 = BinaryOperator::CreateFAdd(sum, sum23, "sum26", NewForEnd);
  Value *sum27 = BinaryOperator::CreateFAdd(sum24, sum25, "sum27", NewForEnd);
  Sum = BinaryOperator::CreateFAdd(sum26, sum27, "sum28", NewForEnd);
  IRBuilder<> Builder(NewForEnd);
  Builder.SetInsertPoint(NewForEnd);
  // Create a new StoreInst instruction
  Builder.CreateStore(Sum, GEP);
  // Create a comparison instruction
  Value *Cmp = Builder.CreateICmpUGT(Add2, GEP->getOperand(1), "cmp37.not548");

  // Create a conditional branch instruction
  Builder.CreateCondBr(Cmp, ForBodyMerged->getTerminator()->getSuccessor(1),
                       CloneForBodyPreheader);
  ForBodyMerged->getTerminator()->setSuccessor(1, NewForEnd);
  CloneForBodyPreheader->moveAfter(NewForEnd);
  CloneForBody->moveAfter(CloneForBodyPreheader);

  // Create a PHI node in CloneForBodyPreheader
  PHINode *SumPHI = PHINode::Create(Sum->getType(), 2, "sum.phi",
                                    CloneForBodyPreheader->getFirstNonPHI());

  // Set the incoming values of the PHI node
  SumPHI->addIncoming(ConstantFP::get(Sum->getType(), 0.0), OuterBB);
  SumPHI->addIncoming(Sum, NewForEnd);

  // Create a PHI node in CloneForBodyPreheader
  PHINode *AddPHI = PHINode::Create(Add2->getType(), 2, "add.phi",
                                    CloneForBodyPreheader->getFirstNonPHI());

  // Set the incoming values of the PHI node
  AddPHI->addIncoming(ConstantInt::get(Add2->getType(), 0), OuterBB);
  AddPHI->addIncoming(Add2, NewForEnd);
  Value *phifloatincomingvalue0 =
      getFirstCallInstWithName(CloneForBody, "llvm.fmuladd.f32");
  Value *phii32incomingvalue0 =
      getLastInst<ICmpInst>(CloneForBody)->getOperand(0);
  for (PHINode &Phi : CloneForBody->phis()) {
    if (Phi.getType()->isIntegerTy(32)) {
      Phi.setIncomingValue(0, AddPHI);
      Phi.setIncomingBlock(0, CloneForBodyPreheader);
      Phi.setIncomingValue(1, phii32incomingvalue0);
      Phi.setIncomingBlock(1, CloneForBody);
    } else if (Phi.getType()->isFloatTy()) {
      Phi.setIncomingValue(0, SumPHI);
      Phi.setIncomingBlock(0, CloneForBodyPreheader);
      Phi.setIncomingValue(1, phifloatincomingvalue0);
      Phi.setIncomingBlock(1, CloneForBody);
    }
  }
  BasicBlock *OldForEnd = CloneForBody->getTerminator()->getSuccessor(0);
  createCriticalEdgeAndMoveStoreInst(CloneForBody, OldForEnd);

  getFirstI32Phi(ForBodyMerged)->setIncomingBlock(1, ForBodyMerged);
}

static std::tuple<Value *, Value *, GetElementPtrInst *>
modifyOuterLoop8(Loop *L) {
  BasicBlock *BB = L->getHeader();
  ICmpInst *LastICmp = getLastInst<ICmpInst>(BB);
  LastICmp->setPredicate(ICmpInst::ICMP_ULT);
  swapTerminatorSuccessors(BB);

  eraseAllStoreInstInBB(BB);
  Value *lsig_0 = getFirstI32Phi(BB)->getIncomingValue(0);
  Value *add207 = LastICmp->getOperand(0);
  Value *sub206 = cast<Instruction>(add207)->getOperand(0);
  // Add new instructions before LastICmp
  IRBuilder<> Builder(LastICmp);

  // %add207.neg = xor i32 %sub206, -1
  Value *Add207Neg = Builder.CreateXor(
      sub206, ConstantInt::get(sub206->getType(), -1), "add207.neg");

  // %add211 = add i32 %lsig.0, %add207.neg
  Value *Add211 = Builder.CreateAdd(lsig_0, Add207Neg, "add211");

  // %div212535 = and i32 %add211, -8
  Value *Div212535 = Builder.CreateAnd(
      Add211, ConstantInt::get(Add211->getType(), -8), "div212535");

  // %add214 = add i32 %div212535, %add207
  Value *Add214 = Builder.CreateAdd(Div212535, add207, "add214");

  // Set the second operand of LastICmp to Add214
  LastICmp->setOperand(1, Add214);

  // Get the unique getelementptr instruction in BB
  GetElementPtrInst *GEP = getUniqueGetElementPtrInst(BB);

  return std::make_tuple(Add214, add207, GEP);
}

static std::tuple<Value *, Value *, GetElementPtrInst *>
modifyOuterLoop16(Loop *L) {
  BasicBlock *BB = L->getHeader();
  BasicBlock *BBLoopPreHeader = L->getLoopPreheader();
  ICmpInst *LastICmp = getLastInst<ICmpInst>(BB);
  LastICmp->setPredicate(ICmpInst::ICMP_ULT);
  swapTerminatorSuccessors(BB);

  eraseAllStoreInstInBB(BB);
  Value *lkern_0 = getFirstI32Phi(BB)->getIncomingValue(1);
  // Insert an and instruction in BBLoopPreHeader
  IRBuilder<> Builder(BBLoopPreHeader->getTerminator());
  Value *Div536 = Builder.CreateAnd(lkern_0, -16, "div536");
  // Get the first operand of LastICmp
  Value *Add56 = LastICmp->getOperand(0);

  // Create an add instruction before LastICmp
  Builder.SetInsertPoint(LastICmp);
  Value *Add60 = Builder.CreateAdd(Div536, Add56, "add60");

  // Set the second operand of LastICmp to Add60
  LastICmp->setOperand(1, Add60);

  // Get the unique getelementptr instruction in BB
  GetElementPtrInst *GEP = getUniqueGetElementPtrInst(BB);

  return std::make_tuple(Add60, Add56, GEP);
}

static void modifyInnerLoop(Loop *L, BasicBlock *ForBodyMerged, Value *Add60,
                            BasicBlock *CloneForBody, Value *Add56,
                            GetElementPtrInst *GEP, uint32_t unroll_count) {
  assert((unroll_count == 8 || unroll_count == 16) &&
         "unroll_count must be 8 or 16");
  BasicBlock *OuterBB = L->getHeader();

  // Find the predecessor BasicBlock of ForBodyMergedPreheader
  BasicBlock *PredBB = ForBodyMerged->getSinglePredecessor();
  if (!PredBB) {
    // If there is no single predecessor, traverse all predecessors
    for (BasicBlock *Pred : predecessors(ForBodyMerged)) {
      PredBB = Pred;
      break; // Take the first predecessor
    }
  }
  assert(PredBB && "can't find predecessor of ForBodyMerged");

  SmallVector<CallInst *, 16> FMulAddCalls;
  insertPhiNodesForFMulAdd(ForBodyMerged, PredBB, FMulAddCalls);

  movePHINodesToTop(*ForBodyMerged);

  groupAndReorderInstructions(ForBodyMerged);
  ICmpInst *LastICmp = getLastInst<ICmpInst>(ForBodyMerged);
  LastICmp->setPredicate(ICmpInst::ICMP_ULT);
  LastICmp->setOperand(1, Add60);
  swapTerminatorSuccessors(ForBodyMerged);
  eraseAllStoreInstInBB(ForBodyMerged);

  BasicBlock *ForEndLoopExit = ForBodyMerged->getTerminator()->getSuccessor(1);
  // Create an instruction to add the results of four FMulAdd calls
  Value *Sum = nullptr;
  if (unroll_count == 16) {
    Value *sum45 =
        BinaryOperator::CreateFAdd(FMulAddCalls[0], FMulAddCalls[1], "sum45",
                                   ForEndLoopExit->getTerminator());
    Value *sum46 =
        BinaryOperator::CreateFAdd(FMulAddCalls[2], FMulAddCalls[3], "sum46",
                                   ForEndLoopExit->getTerminator());
    Value *sum47 =
        BinaryOperator::CreateFAdd(FMulAddCalls[4], FMulAddCalls[5], "sum47",
                                   ForEndLoopExit->getTerminator());
    Value *sum48 =
        BinaryOperator::CreateFAdd(FMulAddCalls[6], FMulAddCalls[7], "sum48",
                                   ForEndLoopExit->getTerminator());
    Value *sum49 =
        BinaryOperator::CreateFAdd(FMulAddCalls[8], FMulAddCalls[9], "sum49",
                                   ForEndLoopExit->getTerminator());
    Value *sum50 =
        BinaryOperator::CreateFAdd(FMulAddCalls[10], FMulAddCalls[11], "sum50",
                                   ForEndLoopExit->getTerminator());
    Value *sum51 =
        BinaryOperator::CreateFAdd(FMulAddCalls[12], FMulAddCalls[13], "sum51",
                                   ForEndLoopExit->getTerminator());
    Value *sum52 =
        BinaryOperator::CreateFAdd(FMulAddCalls[14], FMulAddCalls[15], "sum52",
                                   ForEndLoopExit->getTerminator());

    Value *sum53 = BinaryOperator::CreateFAdd(sum45, sum46, "sum53",
                                              ForEndLoopExit->getTerminator());
    Value *sum54 = BinaryOperator::CreateFAdd(sum47, sum48, "sum54",
                                              ForEndLoopExit->getTerminator());
    Value *sum55 = BinaryOperator::CreateFAdd(sum49, sum50, "sum55",
                                              ForEndLoopExit->getTerminator());
    Value *sum56 = BinaryOperator::CreateFAdd(sum51, sum52, "sum56",
                                              ForEndLoopExit->getTerminator());

    Value *sum57 = BinaryOperator::CreateFAdd(sum53, sum54, "sum57",
                                              ForEndLoopExit->getTerminator());
    Value *sum58 = BinaryOperator::CreateFAdd(sum55, sum56, "sum58",
                                              ForEndLoopExit->getTerminator());

    Sum = BinaryOperator::CreateFAdd(sum57, sum58, "sum59",
                                     ForEndLoopExit->getTerminator());
  } else if (unroll_count == 8) {
    Value *sum60 =
        BinaryOperator::CreateFAdd(FMulAddCalls[0], FMulAddCalls[1], "sum60",
                                   ForEndLoopExit->getTerminator());
    Value *sum61 =
        BinaryOperator::CreateFAdd(FMulAddCalls[2], FMulAddCalls[3], "sum61",
                                   ForEndLoopExit->getTerminator());
    Value *sum62 =
        BinaryOperator::CreateFAdd(FMulAddCalls[4], FMulAddCalls[5], "sum62",
                                   ForEndLoopExit->getTerminator());
    Value *sum63 =
        BinaryOperator::CreateFAdd(FMulAddCalls[6], FMulAddCalls[7], "sum63",
                                   ForEndLoopExit->getTerminator());

    Value *sum64 = BinaryOperator::CreateFAdd(sum60, sum61, "sum64",
                                              ForEndLoopExit->getTerminator());
    Value *sum65 = BinaryOperator::CreateFAdd(sum62, sum63, "sum65",
                                              ForEndLoopExit->getTerminator());
    Sum = BinaryOperator::CreateFAdd(sum64, sum65, "sum66",
                                     ForEndLoopExit->getTerminator());
  }

  // Create a new basic block for.end164
  BasicBlock *ForEnd164 = BasicBlock::Create(
      ForEndLoopExit->getContext(), "for.end164", ForEndLoopExit->getParent(),
      ForEndLoopExit->getNextNode());

  // Set the target of the terminator instruction of ForEndLoopExit to
  // for.end164
  Instruction *Terminator = ForEndLoopExit->getTerminator();
  BasicBlock *OldSuccessor = Terminator->getSuccessor(0);
  Terminator->setSuccessor(0, ForEnd164);

  // Create an unconditional branch instruction in for.end164, jumping to the
  // original successor basic block
  BranchInst::Create(OldSuccessor, ForEnd164);

  // Create a new phi node in for.end164
  PHINode *PhiSum = PHINode::Create(Type::getInt32Ty(ForEnd164->getContext()),
                                    2, "phi.sum", ForEnd164->getFirstNonPHI());

  // Set the incoming values of the phi node
  PhiSum->addIncoming(Add56, OuterBB);
  PhiSum->addIncoming(LastICmp->getOperand(0), ForEndLoopExit);

  // Create a new phi float node in for.end164
  PHINode *PhiFloat =
      PHINode::Create(Type::getFloatTy(ForEnd164->getContext()), 2, "phi.float",
                      ForEnd164->getFirstNonPHI());

  // Set the incoming values of the phi node
  PhiFloat->addIncoming(
      ConstantFP::get(Type::getFloatTy(ForEnd164->getContext()), 0.0), OuterBB);
  PhiFloat->addIncoming(Sum, ForEndLoopExit);
  // Create a new StoreInst instruction in for.end164
  new StoreInst(PhiFloat, GEP, ForEnd164->getTerminator());

  Value *operand1 = unroll_count == 16
                        ? getFirstI32Phi(OuterBB)
                        : getLastInst<ICmpInst>(CloneForBody)->getOperand(1);
  // Create a new comparison instruction
  ICmpInst *NewCmp =
      new ICmpInst(ICmpInst::ICMP_UGT, PhiSum, operand1, "cmp182.not587");
  NewCmp->insertBefore(ForEnd164->getTerminator());

  // Replace the original unconditional branch with a conditional branch
  BranchInst *OldBr = cast<BranchInst>(ForEnd164->getTerminator());
  BasicBlock *ForEnd37 = OldBr->getSuccessor(0);
  BranchInst *NewBr = BranchInst::Create(ForEnd37, CloneForBody, NewCmp);
  ReplaceInstWithInst(OldBr, NewBr);

  CloneForBody->moveAfter(ForEnd164);
  Instruction *TargetInst =
      getFirstCallInstWithName(CloneForBody, "llvm.fmuladd.f32");
  for (PHINode &Phi : CloneForBody->phis()) {
    if (Phi.getType()->isIntegerTy(32)) {
      Phi.setIncomingValue(0,
                           getLastInst<ICmpInst>(CloneForBody)->getOperand(0));
      Phi.setIncomingBlock(0, CloneForBody);
      Phi.setIncomingValue(1, PhiSum);
      Phi.setIncomingBlock(1, ForEnd164);
    } else if (Phi.getType()->isFloatTy()) {
      Phi.setIncomingValue(0, TargetInst);
      Phi.setIncomingBlock(0, CloneForBody);
      Phi.setIncomingValue(1, PhiFloat);
      Phi.setIncomingBlock(1, ForEnd164);
    }
  }

  createCriticalEdgeAndMoveStoreInst(CloneForBody, ForEnd37);

  OuterBB->getTerminator()->setSuccessor(1, ForEnd164);
}

static void PostUnrollConv(Function &F, Loop *L, int unroll_count,
                           int unroll_index) {
  BasicBlock *ForBody = L->getHeader();
  BasicBlock *CloneForBody =
      cloneBasicBlockWithRelations(ForBody, ".clone", &F);
  CloneForBody->moveAfter(ForBody);
  // Set the second branch of the terminator instruction of CloneForBody to
  // ForBody
  CloneForBody->getTerminator()->setSuccessor(1, ForBody);

  StringRef ForBodyName = ForBody->getName();
  // Get the basic blocks to merge
  std::vector<BasicBlock *> BBsToMerge;
  for (int i = 1; i < unroll_count; ++i) {
    std::string BBName = (ForBodyName + "." + std::to_string(i)).str();
    BasicBlock *ForBodyClone = getBasicBlockByName(F, BBName);
    if (ForBodyClone) {
      BBsToMerge.push_back(ForBodyClone);
    }
  }

  if (BBsToMerge.size() == static_cast<size_t>(unroll_count - 1)) {
    for (BasicBlock *BB : BBsToMerge) {
      MergeBasicBlockIntoOnlyPred(BB);
    }
  }
  // Get the outer loop of L
  Loop *OuterLoop = L->getParentLoop();
  if (unroll_count == 8 && unroll_index == 0) {
    BasicBlock *CloneForBodyPreheader = BasicBlock::Create(
        CloneForBody->getContext(), CloneForBody->getName() + ".preheader",
        CloneForBody->getParent(), CloneForBody);

    updatePredecessorsToPreheader(CloneForBody, CloneForBodyPreheader);
    auto [Sub, GEP, Add2] =
        modifyOuterLoop4(OuterLoop, BBsToMerge[6], CloneForBodyPreheader);
    modifyInnerLoop4(OuterLoop, BBsToMerge[6], Sub, CloneForBody, GEP, Add2,
                     CloneForBodyPreheader);
  } else if (unroll_count == 16) {
    auto [Add60, Add56, GEP] = modifyOuterLoop16(OuterLoop);
    modifyInnerLoop(OuterLoop, BBsToMerge[14], Add60, CloneForBody, Add56, GEP,
                    unroll_count);
  } else if (unroll_count == 8) {
    auto [Add214, Add207, GEP] = modifyOuterLoop8(OuterLoop);
    modifyInnerLoop(OuterLoop, BBsToMerge[6], Add214, CloneForBody, Add207, GEP,
                    unroll_count);
  }
  LLVM_DEBUG(F.dump());
}

static void modifyFirstCloneForBody(BasicBlock *CloneForBody,
                                    PHINode *N_0_lcssa,
                                    BasicBlock *ForBody27LrPh,
                                    PHINode *CoeffPosLcssa, Value *Operand1) {
  CloneForBody->getTerminator()->setSuccessor(1, CloneForBody);
  for (PHINode &Phi : CloneForBody->phis()) {
    Phi.setIncomingBlock(0, ForBody27LrPh);
    Phi.setIncomingBlock(1, CloneForBody);
  }
  PHINode *FirstI32Phi = getFirstI32Phi(CloneForBody);
  PHINode *LastI32Phi = getLastI32Phi(CloneForBody);
  FirstI32Phi->setIncomingValue(0, N_0_lcssa);
  FirstI32Phi->setIncomingBlock(0, ForBody27LrPh);

  Instruction *firstAddInst = nullptr;
  Instruction *lastAddInst = nullptr;
  for (Instruction &I : *CloneForBody) {
    if (I.getOpcode() == Instruction::Add) {
      if (!firstAddInst) {
        firstAddInst = &I;
      }
      lastAddInst = &I;
    }
  }
  ICmpInst *LastCmpInst = getLastInst<ICmpInst>(CloneForBody);
  LastCmpInst->setOperand(0, lastAddInst);
  LastCmpInst->setOperand(1, Operand1);
  FirstI32Phi->setIncomingValue(1, lastAddInst);

  LastI32Phi->setIncomingValue(0, CoeffPosLcssa);
  LastI32Phi->setIncomingBlock(0, ForBody27LrPh);

  LastI32Phi->setIncomingValue(1, firstAddInst);
}

static bool setBBFromOtherBB(Function &F, StringRef BBName,
                             BasicBlock *ForBodyMerged) {
  // Find the first and last load instructions in ForBody27LrPh
  LoadInst *FirstLoad = nullptr;
  LoadInst *LastLoad = nullptr;
  BasicBlock *ForBody27LrPh = getBasicBlockByName(F, BBName);
  for (Instruction &I : *ForBody27LrPh) {
    if (auto *LI = dyn_cast<LoadInst>(&I)) {
      if (!FirstLoad) {
        FirstLoad = LI;
      }
      LastLoad = LI;
    }
  }

  assert(FirstLoad && LastLoad && "Find  load instructions in ForBody27LrPh");

  // modify getelementptr
  // Traverse the GEP instructions in ForBodyMerged
  std::vector<GetElementPtrInst *> GEPInsts;
  for (Instruction &I : *ForBodyMerged) {
    if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
      GEPInsts.push_back(GEP);
    }
  }
  // Ensure there is at least one GEP instruction
  if (!GEPInsts.empty()) {
    for (size_t i = 0; i < GEPInsts.size(); ++i) {
      GetElementPtrInst *CurrentGEP = GEPInsts[i];

      if (i % 2 == 1) { // Odd
        CurrentGEP->setOperand(0, LastLoad);
      } else { // Even
        CurrentGEP->setOperand(0, FirstLoad);
      }
    }
  }
  return true;
}

// Function to modify the first loop in FIRD (Finite Impulse Response Design)
// transformation
static void modifyFirdFirstLoop(Function &F, Loop *L, BasicBlock *ForBodyMerged,
                                BasicBlock *CloneForBody) {
  BasicBlock *ForCond23Preheader =
      ForBodyMerged->getTerminator()->getSuccessor(0)->getSingleSuccessor();
  assert(ForCond23Preheader &&
         "ForCondPreheader should have single predecessor");

  BasicBlock *ForCondCleanup3 =
      getFirstI32Phi(ForCond23Preheader)->getIncomingBlock(0);
  Instruction *FirstI32Phi = getFirstI32Phi(ForCondCleanup3);

  ICmpInst *LastICmp = getLastInst<ICmpInst>(ForCondCleanup3);
  // Create new add instruction
  IRBuilder<> Builder(LastICmp);
  Value *Add269 = Builder.CreateNSWAdd(
      FirstI32Phi, ConstantInt::get(FirstI32Phi->getType(), 8), "add269");
  LastICmp->setOperand(0, Add269);
  LastICmp->setPredicate(ICmpInst::ICMP_SGT);
  swapTerminatorSuccessors(ForCondCleanup3);

  PHINode *N_069 = getFirstI32Phi(ForBodyMerged);
  Value *Inc20_7 = N_069->getIncomingValue(1);
  BasicBlock *ForBodyMergedLoopPreheader = N_069->getIncomingBlock(0);
  // Create new phi node at the beginning of ForBodyMerged
  PHINode *Add281 = PHINode::Create(Type::getInt32Ty(F.getContext()), 2,
                                    "add281", &ForBodyMerged->front());

  // Set incoming values for phi node
  Add281->addIncoming(Add269, ForBodyMergedLoopPreheader);
  Add281->addIncoming(Inc20_7, ForBodyMerged);

  N_069->setIncomingValue(1, Add281);

  ICmpInst *LastICmpInPreheader = getLastInst<ICmpInst>(ForCond23Preheader);
  // Create new phi node
  PHINode *N_0_lcssa = PHINode::Create(Type::getInt32Ty(F.getContext()), 2,
                                       "n.0.lcssa", LastICmpInPreheader);

  // Set incoming values for phi node
  N_0_lcssa->addIncoming(FirstI32Phi, ForCondCleanup3);
  N_0_lcssa->addIncoming(Add281, ForBodyMerged);

  // Replace operand of LastICmpInPreheader with new phi node
  LastICmpInPreheader->setOperand(0, N_0_lcssa);
  LastICmpInPreheader->setPredicate(ICmpInst::ICMP_SLT);

  Value *Operand1 = LastICmp->getOperand(1);
  LastICmpInPreheader->setOperand(1, Operand1);

  // Get %coeff_pos.0.lcssa
  PHINode *CoeffPosLcssa = getFirstI32Phi(ForCond23Preheader);

  // Insert new add instruction at the end of ForBodyMergedLoopPreheader
  BasicBlock *ForBody27LrPh =
      ForCond23Preheader->getTerminator()->getSuccessor(0);
  Builder.SetInsertPoint(ForBody27LrPh->getTerminator());
  Value *Add11 = Builder.CreateAdd(Operand1, CoeffPosLcssa);

  ForBody27LrPh->getTerminator()->setSuccessor(0, CloneForBody);
  ICmpInst *LastICmpInForBodyMerged = getLastInst<ICmpInst>(ForBodyMerged);
  LastICmpInForBodyMerged->setOperand(1, Operand1);
  LastICmpInForBodyMerged->setOperand(0, Inc20_7);

  modifyFirstCloneForBody(CloneForBody, N_0_lcssa, ForBody27LrPh, CoeffPosLcssa,
                          Operand1);

  PHINode *acc_0_lcssa = getFirstFloatPhi(ForCond23Preheader);
  BasicBlock *ForCond23PreheaderLoopExit = acc_0_lcssa->getIncomingBlock(1);
  PHINode *_lcssa = getFirstFloatPhi(ForCond23PreheaderLoopExit);
  acc_0_lcssa->setIncomingValue(1, _lcssa->getIncomingValue(0));
  acc_0_lcssa->setIncomingBlock(1, _lcssa->getIncomingBlock(0));

  Value *floatZero = acc_0_lcssa->getIncomingValue(0);

  // Get all incoming values and blocks for PHINode
  for (unsigned i = 1; i < _lcssa->getNumIncomingValues(); ++i) {
    Value *IncomingValue = _lcssa->getIncomingValue(i);
    BasicBlock *IncomingBlock = _lcssa->getIncomingBlock(i);

    // Create new phi node in ForCond23Preheader
    PHINode *NewPhi =
        PHINode::Create(floatZero->getType(), 2,
                        "acc." + std::to_string(i) + ".lcssa", CoeffPosLcssa);
    // Add incoming values
    NewPhi->addIncoming(floatZero, ForCondCleanup3);
    NewPhi->addIncoming(IncomingValue, IncomingBlock);
  }
  Value *coeff_pos_068 = getLastI32Phi(ForBodyMerged)->getIncomingValue(1);
  CoeffPosLcssa->setIncomingValue(1, coeff_pos_068);

  getLastFloatPhi(CloneForBody)->setIncomingValue(0, acc_0_lcssa);

  BasicBlock *PredBB = ForBodyMerged->getSinglePredecessor();
  if (!PredBB) {
    // If no single predecessor, iterate through all predecessors
    for (BasicBlock *Pred : predecessors(ForBodyMerged)) {
      PredBB = Pred;
      break; // Only take first predecessor
    }
  }
  SmallVector<CallInst *, 8> FMulAddCalls;
  // insertPhiNodesForFMulAdd(ForBodyMerged, ForCond23PreHeader, FMulAddCalls);
  // Collect all tail call float @llvm.fmuladd.f32 in LoopHeader
  for (Instruction &I : *ForBodyMerged) {
    if (CallInst *CI = dyn_cast<CallInst>(&I)) {
      if (Function *F = CI->getCalledFunction()) {
        if (F->getName() == "llvm.fmuladd.f32" && CI->isTailCall()) {
          FMulAddCalls.push_back(CI);
        }
      }
    }
  }

  // Insert phi nodes for each FMulAdd call
  for (CallInst *CI : FMulAddCalls) {
    // Create new phi node
    PHINode *PHI = PHINode::Create(CI->getType(), 2, CI->getName() + "acc", CI);

    // Set incoming values for phi node
    PHI->addIncoming(ConstantFP::get(CI->getType(), 0), PredBB);
    PHI->addIncoming(CI, ForBodyMerged);

    CI->setOperand(2, PHI);
  }
  movePHINodesToTop(*ForBodyMerged);
  modifyFirdAddToOr(ForBodyMerged);
  ICmpInst *LastICmpForBodyMerged = getLastInst<ICmpInst>(ForBodyMerged);
  LastICmpForBodyMerged->setPredicate(ICmpInst::ICMP_SGT);
  cast<Instruction>(LastICmpForBodyMerged->getOperand(0))
      ->setOperand(0, getFirstI32Phi(ForBodyMerged));

  // Find first and last load instructions in ForBody14LrPh
  LoadInst *FirstLoad = nullptr;
  LoadInst *LastLoad = nullptr;
  BasicBlock *ForBody14LrPh = getBasicBlockByName(F, "for.body14.lr.ph");
  for (Instruction &I : *ForBody14LrPh) {
    if (auto *LI = dyn_cast<LoadInst>(&I)) {
      if (!FirstLoad) {
        FirstLoad = LI;
      }
      LastLoad = LI;
    }
  }

  assert(FirstLoad && LastLoad &&
         "Failed to find load instructions in ForBody14LrPh");

  // modify getelementptr
  // Iterate through getelementptr instructions in ForBodyMerged
  std::vector<GetElementPtrInst *> GEPInsts;
  for (Instruction &I : *ForBodyMerged) {
    if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
      GEPInsts.push_back(GEP);
    }
  }
  // Ensure at least one getelementptr instruction exists
  if (!GEPInsts.empty()) {
    for (size_t i = 0; i < GEPInsts.size(); ++i) {
      GetElementPtrInst *CurrentGEP = GEPInsts[i];

      if (i % 2 == 1) { // Odd
        CurrentGEP->setOperand(0, LastLoad);
      } else { // Even
        CurrentGEP->setOperand(0, FirstLoad);
      }
    }
  }

  // Ensure at least one getelementptr instruction exists
  if (!GEPInsts.empty()) {
    // Get first getelementptr instruction
    GetElementPtrInst *SecondGEP = GEPInsts[1];

    // Starting from index 1, process every other getelementptr
    for (size_t i = 3; i < GEPInsts.size(); i += 2) {
      GetElementPtrInst *CurrentGEP = GEPInsts[i];

      // Set current getelementptr's operand 0 to first getelementptr's value
      CurrentGEP->setOperand(0, SecondGEP);

      // Set operand 1 to current index value
      // ConstantInt *IndexValue =
      // ConstantInt::get(CurrentGEP->getOperand(1)->getType(), i);
      CurrentGEP->setOperand(
          1, ConstantInt::get(CurrentGEP->getOperand(1)->getType(), (i) / 2));
    }
  }

  setBBFromOtherBB(F, "for.body27.lr.ph", CloneForBody);

  BasicBlock *ForCondCleanup26LoopExit = CloneForBody->getNextNode();
  BasicBlock *ForCondCleanup26 = ForCondCleanup26LoopExit->getSingleSuccessor();
  Instruction *tailcallInst =
      getFirstCallInstWithName(CloneForBody, "llvm.fmuladd.f32");

  // Find add instruction in ForBody27LrPh
  Instruction *AddInst = nullptr;
  for (Instruction &I : *ForBody27LrPh) {
    if (I.getOpcode() == Instruction::Add) {
      AddInst = &I;
      break;
    }
  }

  // Insert new instructions in ForCondCleanup26LoopExit
  Builder.SetInsertPoint(ForCondCleanup26LoopExit->getFirstNonPHI());
  Value *SubResult = Builder.CreateSub(AddInst, N_0_lcssa);
  PHINode *firstFloatPhi = getFirstFloatPhi(ForCondCleanup26);
  firstFloatPhi->setIncomingValue(1, tailcallInst);

  ForCond23Preheader->setName("for.cond63.preheader");
  // Create new PHI node in ForCondCleanup26
  PHINode *CoeffPosLcssaPhi =
      PHINode::Create(CoeffPosLcssa->getType(), 2, "coeff_pos.1.lcssa",
                      &ForCondCleanup26->front());

  // Set incoming values and blocks for PHI node
  CoeffPosLcssaPhi->addIncoming(CoeffPosLcssa, ForCond23Preheader);
  CoeffPosLcssaPhi->addIncoming(SubResult, ForCondCleanup26LoopExit);
  // eraseAllStoreInstInBB(ForCondCleanup26);

  ICmpInst *LastICmpForCondCleanup26 = getLastInst<ICmpInst>(ForCondCleanup26);

  LastICmpForCondCleanup26->setPredicate(ICmpInst::ICMP_SLT);
  PHINode *FirstI32ForCondCleanup3 = getFirstI32Phi(ForCondCleanup3);
  LastICmpForCondCleanup26->setOperand(0, FirstI32ForCondCleanup3);
  LastICmpForCondCleanup26->setOperand(
      1,
      ConstantInt::get(LastICmpForCondCleanup26->getOperand(1)->getType(), 8));

  BasicBlock *ForBody79LrPh =
      cloneBasicBlockWithRelations(ForBody27LrPh, ".clone", &F);
  ForBody79LrPh->setName("for.body79.lr.ph");
  ForBody79LrPh->moveBefore(CloneForBody);
  ForBody79LrPh->getTerminator()->setSuccessor(0, ForBodyMerged);
  ForCondCleanup26->getTerminator()->setSuccessor(1, ForBody79LrPh);
  // Create new and instruction in ForBody79LrPh
  Builder.SetInsertPoint(ForBody79LrPh->getTerminator());
  Value *AndResult = Builder.CreateAnd(
      FirstI32ForCondCleanup3,
      ConstantInt::get(FirstI32ForCondCleanup3->getType(), 2147483640));

  BasicBlock *ForCond130Preheader =
      cloneBasicBlockWithRelations(ForCond23Preheader, ".clone", &F);
  ForCond130Preheader->setName("for.cond130.preheader");
  ForCond130Preheader->moveAfter(CloneForBody);
  ForCondCleanup26->getTerminator()->setSuccessor(0, ForCond130Preheader);
  for (PHINode &Phi : ForCond130Preheader->phis()) {
    Phi.setIncomingBlock(0, ForCondCleanup26);
  }
  // Iterate through phi nodes in ForCond130Preheader and ForCond23Preheader
  // simultaneously
  auto it130 = ForCond130Preheader->begin();
  auto it23 = ForCond23Preheader->begin();

  while (it130 != ForCond130Preheader->end() &&
         it23 != ForCond23Preheader->end()) {
    if (auto *phi130 = dyn_cast<PHINode>(&*it130)) {
      if (auto *phi23 = dyn_cast<PHINode>(&*it23)) {
        if (phi130->getType()->isFloatTy() && phi23->getType()->isFloatTy()) {
          // Write phi float from ForCond23Preheader to incomingvalue 0 position
          // in ForCond130Preheader
          phi130->setIncomingValue(0, phi23);
        }
      }
      ++it23;
    }
    ++it130;
  }
  getFirstFloatPhi(ForCond130Preheader)->setIncomingValue(0, firstFloatPhi);

  getFirstI32Phi(ForCond130Preheader)
      ->setIncomingValue(0, getFirstI32Phi(ForCondCleanup26));

  PHINode *LastI32Phi130 = getLastI32Phi(ForCond130Preheader);
  LastI32Phi130->setIncomingValue(
      0, ConstantInt::get(getLastI32Phi(ForCond130Preheader)->getType(), 0));
  LastI32Phi130->setIncomingValue(1, AndResult);

  ICmpInst *LastICmp130 = getLastInst<ICmpInst>(ForCond130Preheader);
  LastICmp130->setOperand(1, FirstI32ForCondCleanup3);

  PHINode *LastI32PhiClone = getLastFloatPhi(CloneForBody);
  LastI32PhiClone->setIncomingValue(1, tailcallInst);

  // modify for.cond23.preheader.loopexit
  // modify for.cond63.preheader
  for (PHINode &Phi : ForCond23Preheader->phis()) {
    Phi.setIncomingBlock(1, ForBodyMerged);
  }
  ForBodyMerged->getTerminator()->setSuccessor(0, ForCond130Preheader);

  CloneForBody->getTerminator()->setSuccessor(0, ForCondCleanup26LoopExit);

  // Get for.cond.cleanup.loopexit basic block
  BasicBlock *ForCondCleanupLoopExit =
      getBasicBlockByName(F, "for.cond23.preheader.loopexit");

  // Check if for.cond.cleanup.loopexit exists
  if (ForCondCleanupLoopExit) {
    // Check if for.cond.cleanup.loopexit has no predecessors
    if (pred_empty(ForCondCleanupLoopExit)) {
      // Delete for.cond.cleanup.loopexit basic block
      ForCondCleanupLoopExit->eraseFromParent();
    }
  }

  ForBodyMerged->getTerminator()->setSuccessor(0, ForCond23Preheader);
}

static bool copyFloatPhiIncomingValue(int i, BasicBlock *srcBB,
                                      BasicBlock *tarBB) {
  assert(srcBB && tarBB && "srcBB or tarBB should not be nullptr");
  // Collect phi float nodes from ForCond130Preheader in reverse order into
  // vector
  SmallVector<Value *, 8> floatPhis;

  for (auto it = srcBB->rbegin(); it != srcBB->rend(); ++it) {
    if (PHINode *phi = dyn_cast<PHINode>(&*it)) {
      if (phi->getType()->isFloatTy()) {
        floatPhis.push_back(phi->getIncomingValue(i));
      }
    }
  }

  // Traverse phi float nodes in ForBodyMerged in reverse order and store values
  // from floatPhis into their incoming value 0
  auto floatPhiIt = floatPhis.begin();
  for (auto it = tarBB->rbegin();
       it != tarBB->rend() && floatPhiIt != floatPhis.end(); ++it) {
    if (PHINode *phi = dyn_cast<PHINode>(&*it)) {
      if (phi->getType()->isFloatTy()) {
        phi->setIncomingValue(i, *floatPhiIt);
        ++floatPhiIt;
      }
    }
  }
  return true;
}

static void modifyFirdSecondLoop(Function &F, Loop *L,
                                 BasicBlock *ForBodyMerged,
                                 BasicBlock *CloneForBody) {
  BasicBlock *ForBody = L->getHeader();

  BasicBlock *ForBody133LrPh =
      BasicBlock::Create(CloneForBody->getContext(), "for.body133.lr.ph",
                         CloneForBody->getParent(), CloneForBody);

  updatePredecessorsToPreheader(CloneForBody, ForBody133LrPh);

  BasicBlock *PredBB = ForBodyMerged->getSinglePredecessor();
  if (!PredBB) {
    // If there is no single predecessor, iterate through all predecessors
    for (BasicBlock *Pred : predecessors(ForBodyMerged)) {
      PredBB = Pred;
      break; // Only take the first predecessor
    }
  }
  SmallVector<CallInst *, 8> FMulAddCalls;
  // Collect all tail call float @llvm.fmuladd.f32 in LoopHeader
  for (Instruction &I : *ForBodyMerged) {
    if (CallInst *CI = dyn_cast<CallInst>(&I)) {
      if (Function *F = CI->getCalledFunction()) {
        if (F->getName() == "llvm.fmuladd.f32" && CI->isTailCall()) {
          FMulAddCalls.push_back(CI);
        }
      }
    }
  }

  // Insert phi nodes for each FMulAdd call
  for (CallInst *CI : FMulAddCalls) {
    // Create new phi node
    PHINode *PHI = PHINode::Create(CI->getType(), 2, CI->getName() + "acc", CI);

    // Set incoming values for phi node
    PHI->addIncoming(ConstantFP::get(CI->getType(), 0), PredBB);
    PHI->addIncoming(CI, ForBodyMerged);

    CI->setOperand(2, PHI);
  }
  PHINode *n22_075 = getFirstI32Phi(ForBodyMerged);
  // Create new phi node in ForBodyMerged
  PHINode *Add76310 = PHINode::Create(Type::getInt32Ty(F.getContext()), 2,
                                      "add76310", &ForBodyMerged->front());
  Add76310->addIncoming(ConstantInt::get(Type::getInt32Ty(F.getContext()), 8),
                        ForBody133LrPh);
  n22_075->setIncomingValue(1, Add76310);
  // Create new add instruction in ForBodyMerged
  IRBuilder<> Builder(ForBodyMerged->getTerminator());
  Value *Add76 = Builder.CreateAdd(
      Add76310, ConstantInt::get(Type::getInt32Ty(F.getContext()), 8), "add76",
      true, true);

  // Update phi node's loop edge
  Add76310->addIncoming(Add76, ForBodyMerged);

  movePHINodesToTop(*ForBodyMerged);
  modifyFirdAddToOr(ForBodyMerged);
  ICmpInst *LastICmp = getLastInst<ICmpInst>(ForBodyMerged);
  LastICmp->setPredicate(ICmpInst::ICMP_SGT);
  cast<Instruction>(Add76)->moveBefore(LastICmp);
  LastICmp->setOperand(0, Add76);
  for (PHINode &Phi : ForBodyMerged->phis()) {
    Phi.setIncomingBlock(0, PredBB);
  }

  BasicBlock *NewForEnd141 =
      BasicBlock::Create(F.getContext(), "for.end141", &F, CloneForBody);
  NewForEnd141->moveAfter(CloneForBody);

  BasicBlock *ForCond1Preheader = getBasicBlockByName(F, "for.cond1.preheader");
  for (PHINode &Phi : ForCond1Preheader->phis()) {
    Phi.setIncomingBlock(1, NewForEnd141);
  }
  PHINode *ForCond1PreheaderLastI32Phi = getLastI32Phi(ForCond1Preheader);
  // Insert new add instruction in NewForEnd141
  Builder.SetInsertPoint(NewForEnd141);
  Value *Inc152 =
      Builder.CreateAdd(ForCond1PreheaderLastI32Phi,
                        ConstantInt::get(Type::getInt32Ty(F.getContext()), 1),
                        "inc152", true, true);
  Inc152->setName("inc152");

  // Update PHI nodes in ForCond1Preheader
  ForCond1PreheaderLastI32Phi->setIncomingValue(1, Inc152);

  BasicBlock *ForCondCleanup = getBasicBlockByName(F, "for.cond.cleanup");
  getFirstI32Phi(ForCondCleanup)->setIncomingBlock(1, NewForEnd141);

  // Find len parameter in function F
  Value *LenArg = getLenFromEntryBlock(F);
  assert(LenArg && "LenArg should be");

  // Create comparison instruction
  Value *ExitCond350 = Builder.CreateICmpEQ(Inc152, LenArg, "exitcond350.not");

  // Create conditional branch instruction
  Builder.CreateCondBr(ExitCond350, ForCondCleanup, ForCond1Preheader);

  BasicBlock *ForCond130Preheader =
      getBasicBlockByName(F, "for.cond130.preheader");
  for (PHINode &phi : ForCond130Preheader->phis()) {
    phi.setIncomingBlock(1, ForBodyMerged);
  }
  ForCond130Preheader->getTerminator()->setSuccessor(0, ForBody133LrPh);
  ForCond130Preheader->getTerminator()->setSuccessor(1, NewForEnd141);

  // ForBody133LrPh
  // Create new instructions in ForBody133LrPh
  BasicBlock *ForBody79LrPh = getBasicBlockByName(F, "for.body79.lr.ph");
  ForBody79LrPh->getTerminator()->setSuccessor(0, ForBodyMerged);
  // Copy loadinst from ForBody79LrPh to ForBody133LrPh
  Builder.SetInsertPoint(ForBody133LrPh->getTerminator());
  for (Instruction &I : *ForBody79LrPh) {
    if (isa<LoadInst>(I)) {
      Instruction *ClonedInst = I.clone();
      ClonedInst->setName(I.getName());
      Builder.Insert(ClonedInst);
    }
  }

  // modify ForBodyMerged
  for (PHINode &Phi : ForBodyMerged->phis()) {
    Phi.setIncomingBlock(0, ForBody79LrPh);
  }

  PHINode *coeff_pos174 = getLastI32Phi(ForBodyMerged);
  PHINode *coeff_pos_0_lcssa_clone = getFirstI32Phi(ForCond130Preheader);
  coeff_pos_0_lcssa_clone->setIncomingValue(1,
                                            coeff_pos174->getIncomingValue(1));
  coeff_pos174->setIncomingValue(0,
                                 coeff_pos_0_lcssa_clone->getIncomingValue(0));

  bool res = copyFloatPhiIncomingValue(0, ForCond130Preheader, ForBodyMerged);
  assert(res && "copyFloatPhiIncomingZeroValue failed");

  bool res1 = copyFloatPhiIncomingValue(1, ForBodyMerged, ForCond130Preheader);
  assert(res1 && "copyFloatPhiIncomingValue failed");
  // Find first and last load instructions in ForBody79LrPh
  LoadInst *FirstLoad = nullptr;
  LoadInst *LastLoad = nullptr;

  for (Instruction &I : *ForBody79LrPh) {
    if (auto *LI = dyn_cast<LoadInst>(&I)) {
      if (!FirstLoad) {
        FirstLoad = LI;
      }
      LastLoad = LI;
    }
  }

  assert(FirstLoad && LastLoad &&
         "Could not find load instructions in ForBody79LrPh");
  // Iterate through GetElementPtrInst
  std::vector<GetElementPtrInst *> GEPInsts;
  for (Instruction &I : *ForBodyMerged) {
    if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
      GEPInsts.push_back(GEP);
    }
  }

  // Ensure there is at least one getelementptr instruction
  if (!GEPInsts.empty()) {
    for (size_t i = 0; i < GEPInsts.size(); ++i) {
      GetElementPtrInst *CurrentGEP = GEPInsts[i];

      if (i % 2 == 1) { // odd
        CurrentGEP->setOperand(0, LastLoad);
      } else { // even
        CurrentGEP->setOperand(0, FirstLoad);
      }
    }
  }

  // Ensure there is at least one getelementptr instruction
  if (!GEPInsts.empty()) {
    // Get first getelementptr instruction
    GetElementPtrInst *FirstGEP = GEPInsts[0];

    // Starting from index 1, process every other getelementptr
    for (size_t i = 2; i < GEPInsts.size(); i += 2) {
      GetElementPtrInst *CurrentGEP = GEPInsts[i];

      // Set current getelementptr's operand 0 to first getelementptr's value
      CurrentGEP->setOperand(0, FirstGEP);

      // Set operand 1 to current index value
      CurrentGEP->setOperand(
          1, ConstantInt::get(CurrentGEP->getOperand(1)->getType(), (i) / 2));
    }
  }

  ForBodyMerged->getTerminator()->setSuccessor(0, ForCond130Preheader);

  // modify for.body27.clone
  PHINode *n_0_lcssa_clone = getLastI32Phi(ForCond130Preheader);
  PHINode *acc_0_lcssa_clone = getFirstFloatPhi(ForCond130Preheader);
  Instruction *tailcallInst =
      getFirstCallInstWithName(CloneForBody, "llvm.fmuladd.f32");
  Instruction *firstAddInst = nullptr;
  Instruction *lastAddInst = nullptr;
  for (Instruction &I : *CloneForBody) {
    if (I.getOpcode() == Instruction::Add) {
      if (!firstAddInst) {
        firstAddInst = &I;
      }
      lastAddInst = &I;
    }
  }
  int index = 0;
  for (PHINode &Phi : CloneForBody->phis()) {
    Phi.setIncomingBlock(0, ForBody133LrPh);
    Phi.setIncomingBlock(1, CloneForBody);
    if (index == 0) {
      Phi.setIncomingValue(0, n_0_lcssa_clone);
      Phi.setIncomingValue(1, lastAddInst);
    } else if (index == 1) {
      Phi.setIncomingValue(0, coeff_pos_0_lcssa_clone);
      Phi.setIncomingValue(1, firstAddInst);
    } else if (index == 2) {
      Phi.setIncomingValue(0, acc_0_lcssa_clone);
      Phi.setIncomingValue(1, tailcallInst);
    }
    index++;
  }

  CloneForBody->getTerminator()->setSuccessor(0, NewForEnd141);
  CloneForBody->getTerminator()->setSuccessor(1, CloneForBody);

  // modify for.end141
  // Create phi float node in NewForEnd141
  PHINode *AccPhi = PHINode::Create(Type::getFloatTy(F.getContext()), 2,
                                    "acc0.3.lcssa", &NewForEnd141->front());
  AccPhi->addIncoming(acc_0_lcssa_clone, ForCond130Preheader);
  AccPhi->addIncoming(tailcallInst, CloneForBody);

  int i = 0;
  Value *Sum = nullptr;
  Instruction *insertPoint = AccPhi->getNextNode();
  // Count the number of float type phi nodes in ForCond130Preheader
  SmallVector<PHINode *, 8> floatPhis;
  for (PHINode &phi : ForCond130Preheader->phis()) {
    if (phi.getType()->isFloatTy()) {
      floatPhis.push_back(&phi);
    }
  }
  assert(floatPhis.size() == 8 &&
         "Expected 8 float phi nodes in ForCond130Preheader");
  // Create parallel add instructions for better performance
  Value *Add60 =
      BinaryOperator::CreateFAdd(floatPhis[1], AccPhi, "add60", insertPoint);
  Value *Add61 = BinaryOperator::CreateFAdd(floatPhis[2], floatPhis[3], "add61",
                                            insertPoint);
  Value *Add62 = BinaryOperator::CreateFAdd(floatPhis[4], floatPhis[5], "add62",
                                            insertPoint);
  Value *Add63 = BinaryOperator::CreateFAdd(floatPhis[6], floatPhis[7], "add63",
                                            insertPoint);
  Value *Add64 = BinaryOperator::CreateFAdd(Add60, Add61, "add64", insertPoint);
  Value *Add65 = BinaryOperator::CreateFAdd(Add62, Add63, "add65", insertPoint);
  Value *Add66 = BinaryOperator::CreateFAdd(Add64, Add65, "add66", insertPoint);
  Sum = Add66;

  // Move getelementptr and store instructions from for.cond.cleanup26 to
  // NewForEnd141
  BasicBlock *ForCondCleanup26 = getBasicBlockByName(F, "for.cond.cleanup26");

  SmallVector<Instruction *, 2> instructionsToMove;

  // Collect instructions to move
  for (Instruction &I : *ForCondCleanup26) {
    if (isa<GetElementPtrInst>(I) || isa<StoreInst>(I)) {
      instructionsToMove.push_back(&I);
    }
  }

  // Move instructions
  for (Instruction *I : instructionsToMove) {
    I->moveBefore(insertPoint);
    if (isa<StoreInst>(I)) {
      I->setOperand(0, Sum);
    }
  }

  // Update instructions that used moved instructions
  for (Instruction &I : *NewForEnd141) {
    I.replaceUsesOfWith(ForCondCleanup26, NewForEnd141);
  }

  // Get for.cond.cleanup.loopexit basic block
  BasicBlock *ForCondCleanupLoopExit =
      getBasicBlockByName(F, "for.cond.cleanup.loopexit");

  // Check if for.cond.cleanup.loopexit exists
  if (ForCondCleanupLoopExit) {
    // Check if for.cond.cleanup.loopexit has no predecessors
    if (pred_empty(ForCondCleanupLoopExit)) {
      // Delete for.cond.cleanup.loopexit basic block
      ForCondCleanupLoopExit->eraseFromParent();
    }
  }

  setBBFromOtherBB(F, "for.body133.lr.ph", CloneForBody);
}

// Main function to perform FIRD unrolling
static void PostUnrollFird(Function &F, Loop *L, int loop_index) {
  BasicBlock *ForBody = L->getHeader();
  BasicBlock *CloneForBody =
      cloneBasicBlockWithRelations(ForBody, ".clone", &F);
  CloneForBody->moveAfter(ForBody);
  CloneForBody->getTerminator()->setSuccessor(1, ForBody);

  // Merge basic blocks
  std::vector<BasicBlock *> BBsToMerge;
  for (int i = 1; i < 8; ++i) {
    std::string BBName = (ForBody->getName() + "." + std::to_string(i)).str();
    BasicBlock *ForBodyClone = getBasicBlockByName(F, BBName);
    if (ForBodyClone) {
      BBsToMerge.push_back(ForBodyClone);
    } else {
      llvm_unreachable("can't find ForBodyClone");
    }
  }
  if (BBsToMerge.size() == 7) {
    for (BasicBlock *BB : BBsToMerge) {
      MergeBasicBlockIntoOnlyPred(BB);
    }
  }
  BasicBlock *ForBodyMerged = BBsToMerge[6];
  CloneForBody->moveAfter(ForBodyMerged);

  // Perform loop-specific modifications
  if (loop_index == 1) {
    modifyFirdFirstLoop(F, L, ForBodyMerged, CloneForBody);
  } else if (loop_index == 2) {
    modifyFirdSecondLoop(F, L, ForBodyMerged, CloneForBody);
  }
}

// Helper function to check if a loop is simple (single-level, innermost, and
// outermost)
static bool isSimpleLoop(const Loop *L) {
  return L->getLoopDepth() == 1 && L->isInnermost() && L->isOutermost();
}

// Handle simple loops
static bool handleSimpleLoop(Function &F, Loop *L, ScalarEvolution &SE,
                             LoopInfo *LI, DominatorTree &DT,
                             AssumptionCache &AC,
                             const TargetTransformInfo &TTI,
                             OptimizationRemarkEmitter &ORE) {
  if (shouldUnrollLoopWithCount(F, L, SE)) {
    LLVM_DEBUG(errs() << "Unrolling loop with count\n");
    auto UnrollResult =
        UnrollLoop(L,
                   {/*Count*/ 8, /*Force*/ true, /*Runtime*/ false,
                    /*AllowExpensiveTripCount*/ true,
                    /*UnrollRemainder*/ true, true},
                   LI, &SE, &DT, &AC, &TTI, /*ORE*/ &ORE, true);
    postUnrollLoopWithCount(F, L, 8);
    return true;
  }

  if (shouldUnrollComplexLoop(F, L, SE, DT, *LI)) {
    LLVM_DEBUG(errs() << "Unrolling complex loop\n");
    auto UnrollResult =
        UnrollLoop(L,
                   {/*Count*/ 8, /*Force*/ true, /*Runtime*/ false,
                    /*AllowExpensiveTripCount*/ true,
                    /*UnrollRemainder*/ true, true},
                   LI, &SE, &DT, &AC, &TTI, /*ORE*/ &ORE, true);
    postUnrollLoopWithVariable(F, L, 8);
    return true;
  }

  if (shouldUnrollAddcType(F, LI)) {
    LLVM_DEBUG(errs() << "Unrolling ADDC type loop\n");
    unrollAddc(F, SE, L, 16);
    currentUnrollType = UnrollType::ADD_ADDC_SUB_MUL_MULC_SQRT;
    return true;
  }

  if (shouldUnrollDotprodType(F, LI)) {
    LLVM_DEBUG(errs() << "Transforming dot product type loop\n");
    currentUnrollType = UnrollType::DOTPROD;
    transformOneLoopDepth(F);
    return true;
  }

  LLVM_DEBUG(errs() << "No unrolling performed for this loop\n");
  return false;
}

// Helper function to simplify loop and form LCSSA
static void simplifyAndFormLCSSA(Loop *L, DominatorTree &DT, LoopInfo *LI,
                                 ScalarEvolution &SE, AssumptionCache &AC) {
  simplifyLoop(L, &DT, LI, &SE, &AC, nullptr, false);
  formLCSSARecursively(*L, DT, LI, &SE);
}

// Helper function to get CONV unroll factor
static unsigned int getConvUnrollFactor(uint32_t unrollCount) {
  static const unsigned int unrollFactors[] = {8, 16, 8};
  return unrollFactors[unrollCount % 3];
}

// Handle CONV type unrolling
static bool handleConvUnroll(Function &F, Loop *L, ScalarEvolution &SE,
                             LoopInfo *LI, DominatorTree &DT,
                             AssumptionCache &AC,
                             const TargetTransformInfo &TTI,
                             OptimizationRemarkEmitter &ORE,
                             uint32_t &unrollCount) {
  LLVM_DEBUG(errs() << "Unrolling CONV type loop\n");
  currentUnrollType = UnrollType::CONV_CCORR;

  unsigned int unrollFactor = getConvUnrollFactor(unrollCount);
  simplifyAndFormLCSSA(L, DT, LI, SE, AC);

  auto UnrollResult =
      UnrollLoop(L, {unrollFactor, true, false, true, true, true}, LI, &SE, &DT,
                 &AC, &TTI, &ORE, true);

  unrollCount++;
  return true;
}

// Handle FIRD type unrolling
static bool handleFirdUnroll(Function &F, Loop *L, ScalarEvolution &SE,
                             LoopInfo *LI, DominatorTree &DT,
                             AssumptionCache &AC,
                             const TargetTransformInfo &TTI,
                             OptimizationRemarkEmitter &ORE,
                             uint32_t &unroll_times) {
  LLVM_DEBUG(errs() << "Unrolling FIRD type loop\n");
  currentUnrollType = UnrollType::FIRD;

  if (unroll_times == 0) {
    unroll_times++;
    return false;
  }

  simplifyAndFormLCSSA(L, DT, LI, SE, AC);

  auto UnrollResult = UnrollLoop(L, {8, true, false, true, true, true}, LI, &SE,
                                 &DT, &AC, &TTI, &ORE, false);

  return true;
}

// Handle innermost loops
static bool handleInnermostLoop(Function &F, Loop *L, ScalarEvolution &SE,
                                LoopInfo *LI, DominatorTree &DT,
                                AssumptionCache &AC,
                                const TargetTransformInfo &TTI,
                                OptimizationRemarkEmitter &ORE,
                                uint32_t &unrollCount) {
  if (shouldUnrollCorr(F, LI)) {
    LLVM_DEBUG(errs() << "Unrolling correlation type loop\n");
    unrollCorr(F, L, 16);
    currentUnrollType = UnrollType::CORR;
    return true;
  }

  if (shouldUnrollFirType(F, LI) || currentUnrollType == UnrollType::FIR) {
    LLVM_DEBUG(errs() << "Transforming FIR type loop\n");
    unrollFir(F, L);
    currentUnrollType = UnrollType::FIR;
    return true;
  }

  if (shouldUnrollConvccorr(F, LI) ||
      currentUnrollType == UnrollType::CONV_CCORR) {
    return handleConvUnroll(F, L, SE, LI, DT, AC, TTI, ORE, unrollCount);
  }

  if (shouldUnrollFird(F, LI) || currentUnrollType == UnrollType::FIRD) {
    return handleFirdUnroll(F, L, SE, LI, DT, AC, TTI, ORE, unrollCount);
  }

  LLVM_DEBUG(errs() << "No unrolling performed for this innermost loop\n");
  return false;
}

static LoopUnrollResult
tryToUnrollLoop(Function &F, Loop *L, DominatorTree &DT, LoopInfo *LI,
                ScalarEvolution &SE, const TargetTransformInfo &TTI,
                AssumptionCache &AC, OptimizationRemarkEmitter &ORE,
                BlockFrequencyInfo *BFI, ProfileSummaryInfo *PSI) {
  // Initialize variables
  bool changed = false;
  static uint32_t unrollCount = 0;
  // Handle single-level loops
  if (isSimpleLoop(L)) {
    changed = handleSimpleLoop(F, L, SE, LI, DT, AC, TTI, ORE);
  }
  // Handle innermost loops
  else if (L->isInnermost()) {
    changed = handleInnermostLoop(F, L, SE, LI, DT, AC, TTI, ORE, unrollCount);
  }

  return changed ? LoopUnrollResult::PartiallyUnrolled
                 : LoopUnrollResult::Unmodified;
}

// Helper function to process CONV unroll type
void processConvUnroll(Function &F, const SmallVector<Loop *, 4> &InnerLoops) {
  static const int unroll_counts[] = {8, 16, 8};
  static int unroll_index = 0;
  for (auto *L : InnerLoops) {
    PostUnrollConv(F, L, unroll_counts[unroll_index], unroll_index);
    unroll_index = (unroll_index + 1) % 3;
  }
}

// Helper function to process FIRD unroll type
void processFirdUnroll(Function &F, const SmallVector<Loop *, 4> &InnerLoops) {
  static int loop_index = 0;
  for (auto &L : InnerLoops) {
    if (loop_index == 0) {
      loop_index++;
      continue;
    }
    PostUnrollFird(F, L, loop_index);
    loop_index++;
  }
}

static void addCommonOptimizationPasses(Function &F) {
  // Create necessary analysis managers
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;

  // Create pass builder
  PassBuilder PB;

  // Register analyses
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  // Create function-level optimization pipeline
  FunctionPassManager FPM;

  if (currentUnrollType == UnrollType::CORR ||
      currentUnrollType == UnrollType::FIRD)
    FPM.addPass(createFunctionToLoopPassAdaptor(LoopStrengthReducePass()));
  FPM.addPass(EarlyCSEPass(true));
  FPM.addPass(ReassociatePass());

  FPM.run(F, FAM);
}

static void addLegacyCommonOptimizationPasses(Function &F) {
  legacy::FunctionPassManager FPM(F.getParent());
  FPM.add(createLoopSimplifyPass());
  FPM.add(createLICMPass()); // Loop Invariant Code Motion

  // Add SimplifyCFG pass with common options
  FPM.add(createCFGSimplificationPass(
      SimplifyCFGOptions()
          .bonusInstThreshold(1) // Set instruction bonus threshold
          .forwardSwitchCondToPhi(
              true) // Allow forwarding switch conditions to phi
          .convertSwitchToLookupTable(
              true)                  // Allow converting switch to lookup table
          .needCanonicalLoops(false) // Don't require canonical loop form
          .hoistCommonInsts(true)    // Hoist common instructions
          .sinkCommonInsts(true)     // Sink common instructions
      ));

  // Initialize and run passes
  FPM.doInitialization();
  FPM.run(F);
  FPM.doFinalization();
}

PreservedAnalyses
RISCVLoopUnrollAndRemainderPass::run(Function &F, FunctionAnalysisManager &AM) {
  if (!EnableRISCVLoopUnrollAndRemainder || F.arg_empty())
    return PreservedAnalyses::all();

  addnoalias(F);
  auto &LI = AM.getResult<LoopAnalysis>(F);
  if (LI.empty())
    return PreservedAnalyses::all();

  // Retrieve necessary analysis results
  auto &SE = AM.getResult<ScalarEvolutionAnalysis>(F);
  auto &TTI = AM.getResult<TargetIRAnalysis>(F);
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  auto &AC = AM.getResult<AssumptionAnalysis>(F);
  auto &ORE = AM.getResult<OptimizationRemarkEmitterAnalysis>(F);

  LoopAnalysisManager *LAM = nullptr;
  if (auto *LAMProxy = AM.getCachedResult<LoopAnalysisManagerFunctionProxy>(F))
    LAM = &LAMProxy->getManager();

  auto &MAMProxy = AM.getResult<ModuleAnalysisManagerFunctionProxy>(F);
  ProfileSummaryInfo *PSI =
      MAMProxy.getCachedResult<ProfileSummaryAnalysis>(*F.getParent());
  auto *BFI = (PSI && PSI->hasProfileSummary())
                  ? &AM.getResult<BlockFrequencyAnalysis>(F)
                  : nullptr;

  bool Changed = false;

  // Process loops in reverse order of LoopInfo
  SmallPriorityWorklist<Loop *, 4> Worklist;
  appendLoopsToWorklist(LI, Worklist);
  SmallVector<Loop *, 4> InnerLoops;

  while (!Worklist.empty()) {
    Loop &L = *Worklist.pop_back_val();
    if (L.getBlocks().empty()) {
      LLVM_DEBUG(errs() << "Skipping empty loop\n");
      continue;
    }

    std::string LoopName = std::string(L.getName());
    if (L.getName().contains(".clone"))
      continue;

    if (L.isInnermost()) {
      InnerLoops.push_back(&L);
    }

    LoopUnrollResult Result =
        tryToUnrollLoop(F, &L, DT, &LI, SE, TTI, AC, ORE, BFI, PSI);
    Changed |= Result != LoopUnrollResult::Unmodified;

    // Clear cached analysis results if loop was fully unrolled
    if (LAM && Result == LoopUnrollResult::FullyUnrolled)
      LAM->clear(L, LoopName);
  }

  // Post-processing for specific unroll types
  if (currentUnrollType == UnrollType::CONV_CCORR) {
    processConvUnroll(F, InnerLoops);
  } else if (currentUnrollType == UnrollType::FIRD) {
    processFirdUnroll(F, InnerLoops);
  }

  // Run dead code elimination
  runDeadCodeElimination(F);
  if (currentUnrollType != UnrollType::FIR)
    addCommonOptimizationPasses(F);
  if (currentUnrollType == UnrollType::FIRD) {
    addLegacyCommonOptimizationPasses(F);
  }

  // Verify function
  if (verifyFunction(F, &errs())) {
    LLVM_DEBUG(errs() << "Function verification failed\n");
    report_fatal_error("Function verification failed");
  }

  return Changed ? getLoopPassPreservedAnalyses() : PreservedAnalyses::all();
}
