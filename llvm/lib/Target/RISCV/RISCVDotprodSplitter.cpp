//===-- RISCVDotprodSplitter.cpp - ESP32-P4 Dotprod Splitter --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the RISCVDotprodSplitterPass, which identifies specific
// patterns associated with dot product computation functions and creates
// specialized paths for common constant step values.
//
//===----------------------------------------------------------------------===//

#include "RISCVDotprodSplitter.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/IPO/LoopExtractor.h"
#include "llvm/Transforms/Utils/CodeExtractor.h"

using namespace llvm;
using namespace llvm::PatternMatch;

#define DEBUG_TYPE "riscv-dotprod-splitter"

// Command line option to enable the RISCVDotprodSplitter pass
cl::opt<bool>
    llvm::EnableRISCVDotprodSplitter("riscv-dotprod-splitter", cl::init(false),
                                     cl::desc("Enable dotprod splitter"));

STATISTIC(NumDotprodCallsSpecialized,
          "Number of inner calls specialized for step 1, 2, or 3");
STATISTIC(NumFunctionsProcessedBySplitter,
          "Number of functions processed by the dotprod splitter");
STATISTIC(NumFunctionsSkippedBySplitter,
          "Number of functions skipped (pattern not found or ambiguous)");

namespace {
// Configuration constants for argument indices in inner function calls
// Assume step_x for image is arg 1 and for filter is arg 3 in the inner func
const unsigned InnerFuncImageStepArgIdx = 1;
const unsigned InnerFuncFilterStepArgIdx = 3;

// Specialization constants
const unsigned NumSpecializedSteps = 3;
const unsigned SpecializedStepValues[NumSpecializedSteps] = {1, 2, 3};
const unsigned NumPHIIncomingValues = 4; // 3 specialized + 1 generic

// Helper struct to hold the identified pattern
struct TargetCallInfo {
  CallInst *InnerCall = nullptr;
  AllocaInst *ResultAlloca = nullptr;
  unsigned ResultAllocaArgIdx = 0;
  Instruction *LifetimeStart = nullptr;
  LoadInst *ResultLoad = nullptr;
  Instruction *LifetimeEnd = nullptr;
};

// Helper struct to hold results from CFG restructuring
struct CFGRestructureInfo {
  BasicBlock *MergeBB = nullptr;
  PHINode *AccPHI = nullptr;
  LoadInst *GenericResultLoad = nullptr; // Original load in generic path
  BasicBlock *CallGenericBB = nullptr;   // Original block, renamed
  BasicBlock *OrigSuccBB = nullptr;      // Original successor block
};

// Helper function to verify the start-call-load-end sequence in a block
static std::optional<std::tuple<Instruction *, LoadInst *, Instruction *>>
verifySequenceInBlock(BasicBlock &BB, CallInst *TheCall, Value *CallArg) {
  Instruction *FoundStart = nullptr;
  LoadInst *FoundLoad = nullptr;
  Instruction *FoundEnd = nullptr;
  bool FoundCall = false;

  for (Instruction &I : BB) {
    Instruction *Current = &I;
    if (auto *II = dyn_cast<IntrinsicInst>(Current)) {
      if (II->getIntrinsicID() == Intrinsic::lifetime_start ||
          II->getIntrinsicID() == Intrinsic::lifetime_end) {
        // In LLVM 22+, lifetime intrinsics have only one argument (ptr).
        // In older versions, they have two arguments (size, ptr).
        // Check the pointer argument which is at index 0 for new version,
        // or index 1 for old version.
        unsigned PtrArgIdx = (II->arg_size() == 1) ? 0 : 1;
        Value *LifetimePtr = II->getArgOperand(PtrArgIdx);

        if (II->getIntrinsicID() == Intrinsic::lifetime_start &&
            LifetimePtr == CallArg) {
          // Found start before the call?
          if (!FoundCall)
            FoundStart = II;
        } else if (II->getIntrinsicID() == Intrinsic::lifetime_end &&
                   LifetimePtr == CallArg) {
          // Found end after start, call, and load?
          if (FoundStart && FoundCall && FoundLoad) {
            FoundEnd = II;
            break; // Found the complete sequence
          }
        }
      }
    } else if (Current == TheCall) {
      // Found the call after the start?
      if (FoundStart)
        FoundCall = true;
    } else if (auto *LI = dyn_cast<LoadInst>(Current)) {
      // Found load after the call?
      if (LI->getPointerOperand() == CallArg && FoundCall) {
        FoundLoad = LI;
      }
    }
  }

  if (FoundStart && FoundCall && FoundLoad && FoundEnd) {
    return std::make_tuple(FoundStart, FoundLoad, FoundEnd);
  }
  return std::nullopt;
}

// Helper function to find the target call sequence
static std::optional<TargetCallInfo> findTargetCallSequence(Function &F) {
  TargetCallInfo Info;
  bool FoundPotential = false;

  BasicBlock &EntryBB = F.getEntryBlock();
  SmallVector<AllocaInst *, 4> EntryAllocas;
  for (Instruction &I : EntryBB) {
    if (auto *AI = dyn_cast<AllocaInst>(&I)) {
      EntryAllocas.push_back(AI);
    }
  }
  if (EntryAllocas.empty())
    return std::nullopt; // No allocas in entry block

  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      auto *CI = dyn_cast<CallInst>(&I);
      if (!CI || CI->isInlineAsm())
        continue; // Skip inline assembly

      // Check arguments for underlying entry block alloca
      for (unsigned i = 0; i < CI->arg_size(); ++i) {
        Value *Arg = CI->getArgOperand(i); // The direct argument to the call
        Value *UnderlyingObj = getUnderlyingObject(Arg);
        auto *ArgAlloca = dyn_cast_or_null<AllocaInst>(UnderlyingObj);

        // Check if it's one of the allocas from the entry block
        if (ArgAlloca && llvm::is_contained(EntryAllocas, ArgAlloca)) {
          LLVM_DEBUG(dbgs() << "Found potential call: " << *CI
                            << " using alloca: " << *ArgAlloca << " as arg "
                            << i << " (direct arg: " << *Arg << ")\n");

          // Verify the full sequence in this block for this argument
          auto SequenceOpt = verifySequenceInBlock(BB, CI, Arg);

          if (SequenceOpt) {
            auto [FoundStart, FoundLoad, FoundEnd] = *SequenceOpt;
            LLVM_DEBUG(dbgs() << "Found full sequence: Start=" << *FoundStart
                              << ", Call=" << *CI << ", Load=" << *FoundLoad
                              << ", End=" << *FoundEnd << "\n");
            // Check for ambiguity
            if (FoundPotential) {
              LLVM_DEBUG(dbgs() << "Ambiguous pattern: Found multiple "
                                   "calls/args with full sequence in function "
                                << F.getName() << ". Skipping.\n");
              return std::nullopt; // Found more than one, ambiguous
            }

            // Store the found information
            Info.InnerCall = CI;
            Info.ResultAlloca = ArgAlloca; // Store the underlying alloca
            Info.ResultAllocaArgIdx = i;
            Info.LifetimeStart = FoundStart;
            Info.ResultLoad = FoundLoad;
            Info.LifetimeEnd = FoundEnd;
            FoundPotential = true;
            // Don't break inner loops yet, need to check other args/calls for
            // ambiguity

          } else {
            LLVM_DEBUG(dbgs() << "Incomplete sequence for call " << *CI
                              << " using arg " << i << " (direct arg: " << *Arg
                              << "). Skipping this candidate.\n");
          }
        } // End if ArgAlloca check
      } // End loop through arguments
    } // End loop through instructions
  } // End loop through basic blocks

  return FoundPotential ? std::make_optional(Info) : std::nullopt;
}

// Helper function to validate CFG structure before restructuring
static bool validateCFGStructure(BasicBlock *CallBlock) {
  // Check for single predecessor
  BasicBlock *PredBB = CallBlock->getSinglePredecessor();
  if (!PredBB) {
    LLVM_DEBUG(
        dbgs() << "Call block does not have a single predecessor. Skipping.\n");
    return false;
  }

  // Check for single successor
  Instruction *Terminator = CallBlock->getTerminator();
  if (!Terminator || Terminator->getNumSuccessors() != 1) {
    LLVM_DEBUG(dbgs() << "Call block does not have a single successor "
                         "terminator. Skipping.\n");
    return false;
  }

  return true;
}

// Helper function to create check blocks for step values
static void createCheckBlocks(Function &F, LLVMContext &Ctx,
                              BasicBlock *CallBlock, BasicBlock *&CheckEntryBB,
                              BasicBlock *&CheckStep2BB,
                              BasicBlock *&CheckStep3BB) {
  CheckEntryBB = BasicBlock::Create(Ctx, "codeRepl.entry", &F, CallBlock);
  CheckStep2BB = BasicBlock::Create(Ctx, "check.step2", &F, CallBlock);
  CheckStep3BB = BasicBlock::Create(Ctx, "check.step3", &F, CallBlock);
}

// Helper function to create specialized call blocks
static void
createSpecializedCallBlocks(Function &F, LLVMContext &Ctx,
                            BasicBlock *CallBlock, BasicBlock *OrigSuccBB,
                            BasicBlock *&CallStep1BB, BasicBlock *&CallStep2BB,
                            BasicBlock *&CallStep3BB, BasicBlock *&MergeBB) {
  CallStep1BB = BasicBlock::Create(Ctx, "call.step1", &F, CallBlock);
  CallStep2BB = BasicBlock::Create(Ctx, "call.step2", &F, CallBlock);
  CallStep3BB = BasicBlock::Create(Ctx, "call.step3", &F, CallBlock);
  MergeBB = BasicBlock::Create(Ctx, "call.merge", &F, OrigSuccBB);
}

// Helper function to populate check blocks with conditional branches
static void
populateCheckBlocks(BasicBlock *CheckEntryBB, BasicBlock *CheckStep2BB,
                    BasicBlock *CheckStep3BB, BasicBlock *CallStep1BB,
                    BasicBlock *CallStep2BB, BasicBlock *CallStep3BB,
                    BasicBlock *CallGenericBB, Value *ImageStepVal,
                    Value *FilterStepVal) {
  // Populate CheckEntryBB
  IRBuilder<> BuilderCheckEntry(CheckEntryBB);
  Value *IsImgStep1 = BuilderCheckEntry.CreateICmpEQ(
      ImageStepVal, BuilderCheckEntry.getInt32(1), "cmp_img_step1");
  Value *IsFiltStep1 = BuilderCheckEntry.CreateICmpEQ(
      FilterStepVal, BuilderCheckEntry.getInt32(1), "cmp_filt_step1");
  Value *CondStep1 =
      BuilderCheckEntry.CreateAnd(IsImgStep1, IsFiltStep1, "cond_step1");
  BuilderCheckEntry.CreateCondBr(CondStep1, CallStep1BB, CheckStep2BB);

  // Populate CheckStep2BB
  IRBuilder<> BuilderCheckStep2(CheckStep2BB);
  Value *IsImgStep2 = BuilderCheckStep2.CreateICmpEQ(
      ImageStepVal, BuilderCheckStep2.getInt32(2), "cmp_img_step2");
  Value *IsFiltStep2 = BuilderCheckStep2.CreateICmpEQ(
      FilterStepVal, BuilderCheckStep2.getInt32(2), "cmp_filt_step2");
  Value *CondStep2 =
      BuilderCheckStep2.CreateAnd(IsImgStep2, IsFiltStep2, "cond_step2");
  BuilderCheckStep2.CreateCondBr(CondStep2, CallStep2BB, CheckStep3BB);

  // Populate CheckStep3BB
  IRBuilder<> BuilderCheckStep3(CheckStep3BB);
  Value *IsImgStep3 = BuilderCheckStep3.CreateICmpEQ(
      ImageStepVal, BuilderCheckStep3.getInt32(3), "cmp_img_step3");
  Value *IsFiltStep3 = BuilderCheckStep3.CreateICmpEQ(
      FilterStepVal, BuilderCheckStep3.getInt32(3), "cmp_filt_step3");
  Value *CondStep3 =
      BuilderCheckStep3.CreateAnd(IsImgStep3, IsFiltStep3, "cond_step3");
  BuilderCheckStep3.CreateCondBr(CondStep3, CallStep3BB, CallGenericBB);
}

// Helper function to create a specialized call block
static Instruction *createSpecializedCallBlock(
    BasicBlock *BB, CallInst *InnerCall, Instruction *LifetimeStart,
    LoadInst *ResultLoad, Instruction *LifetimeEnd, BasicBlock *MergeBB,
    int StepVal, const Twine &Suffix, unsigned InnerFuncImageStepArgIdx,
    unsigned InnerFuncFilterStepArgIdx) {
  IRBuilder<> Builder(BB);

  // Clone lifetime start
  Instruction *NewLifetimeStart = LifetimeStart->clone();
  Builder.Insert(NewLifetimeStart);

  // Create specialized call with constant step values
  SmallVector<Value *, 8> Args;
  for (unsigned i = 0; i < InnerCall->arg_size(); ++i) {
    if (i == InnerFuncImageStepArgIdx || i == InnerFuncFilterStepArgIdx) {
      Args.push_back(Builder.getInt32(StepVal));
    } else {
      Args.push_back(InnerCall->getArgOperand(i));
    }
  }
  CallInst *NewCall = Builder.CreateCall(InnerCall->getFunctionType(),
                                         InnerCall->getCalledOperand(), Args);
  NewCall->setCallingConv(InnerCall->getCallingConv());

  // Clone result load
  Instruction *NewLoad = ResultLoad->clone();
  NewLoad->setName(ResultLoad->getName() + Suffix);
  Builder.Insert(NewLoad);

  // Clone lifetime end
  Instruction *NewLifetimeEnd = LifetimeEnd->clone();
  Builder.Insert(NewLifetimeEnd);

  // Branch to merge block
  Builder.CreateBr(MergeBB);

  return NewLoad;
}

// Helper function to populate all specialized call blocks
static SmallVector<Instruction *, NumSpecializedSteps>
populateSpecializedCallBlocks(BasicBlock *CallStep1BB, BasicBlock *CallStep2BB,
                              BasicBlock *CallStep3BB, CallInst *InnerCall,
                              Instruction *LifetimeStart, LoadInst *ResultLoad,
                              Instruction *LifetimeEnd, BasicBlock *MergeBB,
                              unsigned InnerFuncImageStepArgIdx,
                              unsigned InnerFuncFilterStepArgIdx) {
  SmallVector<Instruction *, NumSpecializedSteps> SpecializedLoads;

  BasicBlock *CallBlocks[NumSpecializedSteps] = {CallStep1BB, CallStep2BB,
                                                 CallStep3BB};
  const char *Suffixes[NumSpecializedSteps] = {".step1", ".step2", ".step3"};

  for (unsigned i = 0; i < NumSpecializedSteps; ++i) {
    Instruction *Load = createSpecializedCallBlock(
        CallBlocks[i], InnerCall, LifetimeStart, ResultLoad, LifetimeEnd,
        MergeBB, SpecializedStepValues[i], Suffixes[i],
        InnerFuncImageStepArgIdx, InnerFuncFilterStepArgIdx);
    SpecializedLoads.push_back(Load);
  }

  return SpecializedLoads;
}

// Helper function to create and populate merge block
static PHINode *createMergeBlock(
    BasicBlock *MergeBB, BasicBlock *CallStep1BB, BasicBlock *CallStep2BB,
    BasicBlock *CallStep3BB, BasicBlock *CallGenericBB, BasicBlock *OrigSuccBB,
    LoadInst *ResultLoad,
    const SmallVector<Instruction *, NumSpecializedSteps> &SpecializedLoads) {
  IRBuilder<> BuilderMerge(MergeBB);
  PHINode *AccPHI = BuilderMerge.CreatePHI(
      ResultLoad->getType(), NumPHIIncomingValues, "add.us.reload");

  // Add incoming values from specialized blocks
  BasicBlock *SpecializedBlocks[NumSpecializedSteps] = {
      CallStep1BB, CallStep2BB, CallStep3BB};
  for (unsigned i = 0; i < NumSpecializedSteps; ++i) {
    AccPHI->addIncoming(SpecializedLoads[i], SpecializedBlocks[i]);
  }

  // Add incoming value from generic block
  AccPHI->addIncoming(ResultLoad, CallGenericBB);

  BuilderMerge.CreateBr(OrigSuccBB);
  return AccPHI;
}

// Main helper function to restructure CFG for specialization (now much cleaner)
static std::optional<CFGRestructureInfo>
restructureCFGForSpecialization(Function &F, TargetCallInfo &FoundInfo,
                                Value *ImageStepVal, Value *FilterStepVal,
                                unsigned InnerFuncImageStepArgIdx,
                                unsigned InnerFuncFilterStepArgIdx) {
  LLVMContext &Ctx = F.getContext();
  CallInst *InnerCall = FoundInfo.InnerCall;
  LoadInst *ResultLoad = FoundInfo.ResultLoad;
  Instruction *LifetimeStart = FoundInfo.LifetimeStart;
  Instruction *LifetimeEnd = FoundInfo.LifetimeEnd;
  BasicBlock *CallBlock = InnerCall->getParent();

  // 1. Validate CFG structure
  if (!validateCFGStructure(CallBlock))
    return std::nullopt;

  BasicBlock *OrigPredBB = CallBlock->getSinglePredecessor();
  BasicBlock *OrigSuccBB = CallBlock->getTerminator()->getSuccessor(0);

  // 2. Create new blocks
  BasicBlock *CheckEntryBB, *CheckStep2BB, *CheckStep3BB;
  createCheckBlocks(F, Ctx, CallBlock, CheckEntryBB, CheckStep2BB,
                    CheckStep3BB);

  BasicBlock *CallStep1BB, *CallStep2BB, *CallStep3BB, *MergeBB;
  createSpecializedCallBlocks(F, Ctx, CallBlock, OrigSuccBB, CallStep1BB,
                              CallStep2BB, CallStep3BB, MergeBB);

  BasicBlock *CallGenericBB = CallBlock;
  CallGenericBB->setName("call.generic");

  // 3. Rewire predecessor
  OrigPredBB->getTerminator()->replaceSuccessorWith(CallGenericBB,
                                                    CheckEntryBB);

  // 4. Populate check blocks
  populateCheckBlocks(CheckEntryBB, CheckStep2BB, CheckStep3BB, CallStep1BB,
                      CallStep2BB, CallStep3BB, CallGenericBB, ImageStepVal,
                      FilterStepVal);

  // 5. Populate specialized call blocks
  SmallVector<Instruction *, NumSpecializedSteps> SpecializedLoads =
      populateSpecializedCallBlocks(
          CallStep1BB, CallStep2BB, CallStep3BB, InnerCall, LifetimeStart,
          ResultLoad, LifetimeEnd, MergeBB, InnerFuncImageStepArgIdx,
          InnerFuncFilterStepArgIdx);

  // 6. Adjust generic call block
  ResultLoad->setName(ResultLoad->getName() + ".generic");
  Instruction *GenericTerminator = CallGenericBB->getTerminator();
  IRBuilder<> BuilderGeneric(GenericTerminator);
  BuilderGeneric.CreateBr(MergeBB);
  GenericTerminator->eraseFromParent();

  // 7. Create and populate merge block
  PHINode *AccPHI =
      createMergeBlock(MergeBB, CallStep1BB, CallStep2BB, CallStep3BB,
                       CallGenericBB, OrigSuccBB, ResultLoad, SpecializedLoads);

  // Prepare return structure
  CFGRestructureInfo ResultInfo;
  ResultInfo.MergeBB = MergeBB;
  ResultInfo.AccPHI = AccPHI;
  ResultInfo.GenericResultLoad = ResultLoad;
  ResultInfo.CallGenericBB = CallGenericBB;
  ResultInfo.OrigSuccBB = OrigSuccBB;

  return ResultInfo;
}

// Helper function to check if a value represents a load instruction (possibly
// through cast)
static LoadInst *getUnderlyingLoadInst(Value *V) {
  LLVM_DEBUG(dbgs() << "  Checking operand: " << *V << "\n");

  if (auto *CI = dyn_cast<CastInst>(V)) {
    LLVM_DEBUG(dbgs() << "    Is Cast instruction, checking source operand: "
                      << *CI->getOperand(0) << "\n");
    V = CI->getOperand(0);
  }

  if (auto *Load = dyn_cast<LoadInst>(V)) {
    LLVM_DEBUG(dbgs() << "    Found Load instruction: " << *Load << "\n");
    return Load;
  }

  LLVM_DEBUG(dbgs() << "    Not a Load instruction\n");
  return nullptr;
}

// Helper function to check if a multiply instruction has load+offset pattern
static bool hasLoadPlusOffsetPattern(Instruction &MulInst) {
  Value *Op1 = MulInst.getOperand(0);
  Value *Op2 = MulInst.getOperand(1);

  bool HasDirectLoad = false;
  bool HasLoadPlusOffset = false;

  // Check if Op1 is a direct Load
  if (auto *CI = dyn_cast<CastInst>(Op1))
    Op1 = CI->getOperand(0);
  if (isa<LoadInst>(Op1))
    HasDirectLoad = true;

  // Check if Op2 is Load+offset pattern
  if (auto *CI = dyn_cast<CastInst>(Op2))
    Op2 = CI->getOperand(0);
  if (auto *AddInst = dyn_cast<BinaryOperator>(Op2)) {
    if (AddInst->getOpcode() == Instruction::Add) {
      for (unsigned i = 0; i < 2; ++i) {
        Value *AddOp = AddInst->getOperand(i);
        if (auto *CI = dyn_cast<CastInst>(AddOp))
          AddOp = CI->getOperand(0);
        if (isa<LoadInst>(AddOp)) {
          HasLoadPlusOffset = true;
          break;
        }
      }
    }
  }

  return HasDirectLoad && HasLoadPlusOffset;
}

// Check for offset version dot product pattern
static bool hasOffsetDotProductPattern(Loop *L) {
  for (BasicBlock *BB : L->getBlocks()) {
    for (Instruction &I : *BB) {
      if (I.getOpcode() == Instruction::Mul) {
        if (hasLoadPlusOffsetPattern(I)) {
          return true; // Found offset version dot product pattern
        }
      }
    }
  }
  return false;
}

// Helper function to validate memory access pattern using ScalarEvolution
static bool isSimpleForwardAccess(LoadInst *Load, Loop *L,
                                  ScalarEvolution &SE) {
  const SCEV *PtrSCEV = SE.getSCEV(Load->getPointerOperand());
  LLVM_DEBUG(dbgs() << "    Checking Load SCEV: " << *PtrSCEV << "\n");

  // We expect the pointer address to be an affine expression with respect to
  // loop L
  const SCEVAddRecExpr *AddRec = dyn_cast<SCEVAddRecExpr>(PtrSCEV);
  if (!AddRec || AddRec->getLoop() != L) {
    LLVM_DEBUG(dbgs() << "    Not an AddRec expression for current loop\n");
    return false;
  }

  LLVM_DEBUG(dbgs() << "    Is AddRec expression, checking step\n");

  // Get step SCEV
  const SCEV *Step = AddRec->getStepRecurrence(SE);
  LLVM_DEBUG(dbgs() << "    Step SCEV: " << *Step << "\n");

  // Check if step is constant
  if (const SCEVConstant *StepC = dyn_cast<SCEVConstant>(Step)) {
    LLVM_DEBUG(dbgs() << "    Step is compile-time constant: "
                      << StepC->getValue()->getValue() << "\n");

    // Step must be strictly positive, excluding reverse and fixed address
    // access
    if (!StepC->getValue()->getValue().isStrictlyPositive()) {
      LLVM_DEBUG(dbgs() << "    Step is not strictly positive\n");
      return false;
    }
  } else {
    // Check if step is loop invariant
    if (!SE.isLoopInvariant(Step, L)) {
      LLVM_DEBUG(dbgs() << "    Step is not loop invariant\n");
      return false;
    }

    LLVM_DEBUG(dbgs() << "    Step is loop invariant (runtime constant)\n");

    // For loop invariants, we can further check their form
    // For example, check if it's a simple multiplication form (element_size *
    // step_variable)

    // Simple heuristic: if step involves multiplication and includes type size,
    // it might be reasonable
    if (isa<SCEVMulExpr>(Step)) {
      LLVM_DEBUG(dbgs() << "    Step is multiplication expression, might "
                           "include element size\n");
    }
  }

  LLVM_DEBUG(dbgs() << "    Access pattern check passed\n");
  return true;
}

// Helper function to check if multiply result is used in accumulation pattern
static bool hasAccumulationPattern(Instruction &MulInst) {
  LLVM_DEBUG(dbgs() << "Checking accumulation pattern\n");

  for (User *U : MulInst.users()) {
    Instruction *UserInst = cast<Instruction>(U);
    LLVM_DEBUG(dbgs() << "  Multiply user: " << *UserInst << "\n");

    // Check direct user: mul -> add -> phi
    if (auto *BinOp = dyn_cast<BinaryOperator>(UserInst)) {
      if (BinOp->getOpcode() == Instruction::Add) {
        LLVM_DEBUG(dbgs() << "    Found add instruction\n");
        for (User *AddU : BinOp->users()) {
          LLVM_DEBUG(dbgs() << "      Add user: " << *AddU << "\n");
          if (isa<PHINode>(AddU)) {
            LLVM_DEBUG(
                dbgs()
                << "        Found PHI node, accumulation pattern confirmed\n");
            return true;
          }
        }
      }
    }

    // Check indirect user: mul -> cast -> add -> phi
    if (isa<CastInst>(UserInst)) {
      LLVM_DEBUG(dbgs() << "    Multiply used through Cast instruction\n");
      for (User *CastU : UserInst->users()) {
        if (auto *BinOp = dyn_cast<BinaryOperator>(CastU)) {
          if (BinOp->getOpcode() == Instruction::Add) {
            LLVM_DEBUG(dbgs() << "      Found add instruction after Cast\n");
            for (User *AddU : BinOp->users()) {
              LLVM_DEBUG(dbgs() << "        Add user: " << *AddU << "\n");
              if (isa<PHINode>(AddU)) {
                LLVM_DEBUG(dbgs() << "          Found PHI node, accumulation "
                                     "pattern confirmed\n");
                return true;
              }
            }
          }
        }
      }
    }
  }

  return false;
}

// Helper function to validate multiply instruction for dot product pattern
static bool isValidDotProductMultiply(Instruction &MulInst, Loop *L,
                                      ScalarEvolution &SE) {
  LLVM_DEBUG(dbgs() << "Found multiply instruction: " << MulInst << "\n");

  Value *Op1 = MulInst.getOperand(0);
  Value *Op2 = MulInst.getOperand(1);

  LLVM_DEBUG(dbgs() << "Operand 1: " << *Op1 << "\n");
  LLVM_DEBUG(dbgs() << "Operand 2: " << *Op2 << "\n");

  LoadInst *Load1 = getUnderlyingLoadInst(Op1);
  LoadInst *Load2 = getUnderlyingLoadInst(Op2);

  if (!Load1 && !Load2) {
    LLVM_DEBUG(dbgs() << "Neither operand is a Load, skipping\n");
    return false;
  }

  if (!Load1 || !Load2) {
    LLVM_DEBUG(
        dbgs()
        << "Only one operand is a Load, checking if it's offset pattern\n");
    return false;
  }

  LLVM_DEBUG(dbgs() << "Both operands are Loads, checking access pattern\n");

  // Both Loads must be simple sequential access
  bool Load1Valid = isSimpleForwardAccess(Load1, L, SE);
  bool Load2Valid = isSimpleForwardAccess(Load2, L, SE);

  LLVM_DEBUG(dbgs() << "Load1 access pattern valid: "
                    << (Load1Valid ? "Yes" : "No") << "\n");
  LLVM_DEBUG(dbgs() << "Load2 access pattern valid: "
                    << (Load2Valid ? "Yes" : "No") << "\n");

  if (!Load1Valid || !Load2Valid) {
    LLVM_DEBUG(dbgs() << "Access pattern check failed, skipping\n");
    return false;
  }

  // Check if multiply result is used for addition accumulation
  if (!hasAccumulationPattern(MulInst)) {
    LLVM_DEBUG(dbgs() << "No accumulation pattern found\n");
    return false;
  }

  LLVM_DEBUG(dbgs() << "Found complete multiply-accumulate pattern!\n");
  return true;
}

// Check for multiply-accumulate pattern (refactored for better readability)
static bool hasMultiplyAccumulatePattern(Loop *L, ScalarEvolution &SE) {
  LLVM_DEBUG(
      dbgs() << "=== Checking multiply-accumulate pattern in loop ===\n");

  for (BasicBlock *BB : L->getBlocks()) {
    for (Instruction &I : *BB) {
      if (I.getOpcode() == Instruction::Mul) {
        if (isValidDotProductMultiply(I, L, SE)) {
          return true;
        }
      }
    }
  }

  LLVM_DEBUG(dbgs() << "No standard pattern found, checking offset pattern\n");
  return hasOffsetDotProductPattern(L);
}

// Conditional LoopExtractor implementation
// Inspired by LoopExtractor design, create selective Loop extractor
struct SelectiveDotprodLoopExtractor {
  explicit SelectiveDotprodLoopExtractor(
      function_ref<DominatorTree &(Function &)> LookupDomTree,
      function_ref<LoopInfo &(Function &)> LookupLoopInfo,
      function_ref<AssumptionCache *(Function &)> LookupAssumptionCache,
      function_ref<bool(Function &)> ShouldExtractFromFunction)
      : LookupDomTree(LookupDomTree), LookupLoopInfo(LookupLoopInfo),
        LookupAssumptionCache(LookupAssumptionCache),
        ShouldExtractFromFunction(ShouldExtractFromFunction) {}

  bool runOnModule(Module &M);

private:
  function_ref<DominatorTree &(Function &)> LookupDomTree;
  function_ref<LoopInfo &(Function &)> LookupLoopInfo;
  function_ref<AssumptionCache *(Function &)> LookupAssumptionCache;
  function_ref<bool(Function &)> ShouldExtractFromFunction;

  bool runOnFunction(Function &F);
  bool extractLoops(Loop::iterator From, Loop::iterator To, LoopInfo &LI,
                    DominatorTree &DT);
  bool extractLoop(Loop *L, LoopInfo &LI, DominatorTree &DT);
};

bool SelectiveDotprodLoopExtractor::runOnModule(Module &M) {
  if (M.empty())
    return false;

  bool Changed = false;

  // Inspired by LoopExtractor module traversal logic
  // The end of the function list may change (new functions will be added at the
  // end), so we run from the first to the current last.
  auto I = M.begin(), E = --M.end();
  while (true) {
    Function &F = *I;

    // Key change: only process functions that pass detection
    if (ShouldExtractFromFunction(F)) {
      LLVM_DEBUG(dbgs() << "Processing function " << F.getName()
                        << " for loop extraction\n");
      Changed |= runOnFunction(F);
    } else {
      LLVM_DEBUG(dbgs() << "Skipping function " << F.getName()
                        << " (no processable patterns)\n");
    }

    // If this is the last function.
    if (I == E)
      break;

    ++I;
  }
  return Changed;
}

bool SelectiveDotprodLoopExtractor::runOnFunction(Function &F) {
  // Directly borrow the complete logic from LoopExtractor::runOnFunction

  // Do not modify `optnone` functions.
  if (F.hasOptNone())
    return false;

  if (F.empty())
    return false;

  bool Changed = false;
  LoopInfo &LI = LookupLoopInfo(F);

  // If there are no loops in the function.
  if (LI.empty())
    return Changed;

  DominatorTree &DT = LookupDomTree(F);

  // If there is more than one top-level loop in this function, extract all of
  // the loops.
  if (std::next(LI.begin()) != LI.end())
    return Changed | extractLoops(LI.begin(), LI.end(), LI, DT);

  // Otherwise there is exactly one top-level loop.
  Loop *TLL = *LI.begin();

  // If the loop is in LoopSimplify form, then extract it only if this function
  // is more than a minimal wrapper around the loop.
  if (TLL->isLoopSimplifyForm()) {
    bool ShouldExtractLoop = false;

    // Extract the loop if the entry block doesn't branch to the loop header.
    Instruction *EntryTI = F.getEntryBlock().getTerminator();
    if (!isa<BranchInst>(EntryTI) ||
        !cast<BranchInst>(EntryTI)->isUnconditional() ||
        EntryTI->getSuccessor(0) != TLL->getHeader()) {
      ShouldExtractLoop = true;
    } else {
      // Check to see if any exits from the loop are more than just return
      // blocks.
      SmallVector<BasicBlock *, 8> ExitBlocks;
      TLL->getExitBlocks(ExitBlocks);
      for (auto *ExitBlock : ExitBlocks)
        if (!isa<ReturnInst>(ExitBlock->getTerminator())) {
          ShouldExtractLoop = true;
          break;
        }
    }

    if (ShouldExtractLoop)
      return Changed | extractLoop(TLL, LI, DT);
  }

  // Okay, this function is a minimal container around the specified loop.
  // If we extract the loop, we will continue to just keep extracting it
  // infinitely... so don't extract it. However, if the loop contains any
  // sub-loops, extract them.
  return Changed | extractLoops(TLL->begin(), TLL->end(), LI, DT);
}

bool SelectiveDotprodLoopExtractor::extractLoops(Loop::iterator From,
                                                 Loop::iterator To,
                                                 LoopInfo &LI,
                                                 DominatorTree &DT) {
  // Directly borrow from LoopExtractor::extractLoops
  bool Changed = false;
  SmallVector<Loop *, 8> Loops;

  // Save the list of loops, as it may change.
  Loops.assign(From, To);
  for (Loop *L : Loops) {
    // If LoopSimplify form is not available, stay out of trouble.
    if (!L->isLoopSimplifyForm())
      continue;

    Changed |= extractLoop(L, LI, DT);
  }
  return Changed;
}

bool SelectiveDotprodLoopExtractor::extractLoop(Loop *L, LoopInfo &LI,
                                                DominatorTree &DT) {
  // Directly borrow from LoopExtractor::extractLoop
  Function &Func = *L->getHeader()->getParent();
  AssumptionCache *AC = LookupAssumptionCache(Func);
  CodeExtractorAnalysisCache CEAC(Func);
  CodeExtractor Extractor(L->getBlocks(), &DT, false, nullptr, nullptr, AC);
  Function *ExtractedFunc = Extractor.extractCodeRegion(CEAC);
  if (ExtractedFunc) {
    // Set meaningful names for step arguments (indices 1 and 3)
    unsigned ArgIdx = 0;
    for (Argument &Arg : ExtractedFunc->args()) {
      if (ArgIdx == InnerFuncImageStepArgIdx) {
        Arg.setName("img_step");
      } else if (ArgIdx == InnerFuncFilterStepArgIdx) {
        Arg.setName("filt_step");
      }
      ++ArgIdx;
    }
    LI.erase(L);
    LLVM_DEBUG(dbgs() << "Successfully extracted loop from function "
                      << Func.getName() << "\n");
    return true;
  }
  return false;
}

} // end anonymous namespace

// Helper function to replace uses of the original load with PHI result
static void replaceLoadUsesWithPHI(LoadInst *GenericResultLoad,
                                   PHINode *AccPHI) {
  SmallVector<Use *, 16> UsesToReplace;

  // Collect all uses except the one within AccPHI itself
  for (Use &U : GenericResultLoad->uses()) {
    if (U.getUser() != AccPHI) {
      UsesToReplace.push_back(&U);
    }
  }

  // Replace all collected uses
  for (Use *U : UsesToReplace) {
    U->set(AccPHI);
  }
}

// Helper function to update PHI nodes in successor blocks
static void updateSuccessorPHINodes(BasicBlock *OrigSuccBB,
                                    BasicBlock *CallGenericBB,
                                    BasicBlock *MergeBB) {
  for (PHINode &PN : OrigSuccBB->phis()) {
    int Idx = PN.getBasicBlockIndex(CallGenericBB);
    if (Idx != -1) {
      PN.setIncomingBlock(Idx, MergeBB);
    }
  }
}

// Helper function to perform post-restructuring updates
static void
performPostRestructuringUpdates(const CFGRestructureInfo &ResultInfo) {
  // Replace uses of the original load instruction with the merge PHI result
  replaceLoadUsesWithPHI(ResultInfo.GenericResultLoad, ResultInfo.AccPHI);

  // Update PHI nodes in the original successor block
  updateSuccessorPHINodes(ResultInfo.OrigSuccBB, ResultInfo.CallGenericBB,
                          ResultInfo.MergeBB);
}

// Helper function to extract step values from inner call
static std::pair<Value *, Value *> extractStepValues(CallInst *InnerCall) {
  Value *ImageStepVal = InnerCall->getArgOperand(InnerFuncImageStepArgIdx);
  Value *FilterStepVal = InnerCall->getArgOperand(InnerFuncFilterStepArgIdx);
  return {ImageStepVal, FilterStepVal};
}

// Helper function to log target call information
static void logTargetCallInfo(const TargetCallInfo &FoundInfo) {
  LLVM_DEBUG(dbgs() << "Identified target call: " << *FoundInfo.InnerCall
                    << "\n"
                    << "  Result Alloca: " << *FoundInfo.ResultAlloca
                    << " (Arg Index: " << FoundInfo.ResultAllocaArgIdx << ")\n"
                    << "  Lifetime Start: " << *FoundInfo.LifetimeStart << "\n"
                    << "  Result Load: " << *FoundInfo.ResultLoad << "\n"
                    << "  Lifetime End: " << *FoundInfo.LifetimeEnd << "\n");
}

PreservedAnalyses RISCVDotprodSplitterPass::run(Function &F,
                                                FunctionAnalysisManager &AM) {
  if (!EnableRISCVDotprodSplitter || F.isDeclaration())
    return PreservedAnalyses::all();

  NumFunctionsProcessedBySplitter++;
  LLVM_DEBUG(dbgs() << "Running RISCVDotprodSplitterPass on: " << F.getName()
                    << "\n");

  // Step 1: Find the target call sequence
  std::optional<TargetCallInfo> FoundInfoOpt = findTargetCallSequence(F);
  if (!FoundInfoOpt) {
    LLVM_DEBUG(dbgs() << "No suitable call sequence found in " << F.getName()
                      << ". Skipping.\n");
    NumFunctionsSkippedBySplitter++;
    return PreservedAnalyses::all();
  }

  TargetCallInfo &FoundInfo = *FoundInfoOpt;
  logTargetCallInfo(FoundInfo);

  // Step 2: Extract step values from the inner call
  auto [ImageStepVal, FilterStepVal] = extractStepValues(FoundInfo.InnerCall);

  // Step 3: Restructure CFG for specialization
  std::optional<CFGRestructureInfo> RestructureResultOpt =
      restructureCFGForSpecialization(F, FoundInfo, ImageStepVal, FilterStepVal,
                                      InnerFuncImageStepArgIdx,
                                      InnerFuncFilterStepArgIdx);

  if (!RestructureResultOpt) {
    LLVM_DEBUG(dbgs() << "CFG restructuring failed for " << F.getName()
                      << ". Skipping.\n");
    NumFunctionsSkippedBySplitter++;
    return PreservedAnalyses::all();
  }

  // Step 4: Perform post-restructuring updates
  CFGRestructureInfo &ResultInfo = *RestructureResultOpt;
  performPostRestructuringUpdates(ResultInfo);

  LLVM_DEBUG(dbgs() << "Successfully specialized call in " << F.getName()
                    << "\n");
  NumDotprodCallsSpecialized++;
  return PreservedAnalyses::none();
}

// Helper function to check if a function has nested loops with processable
// patterns
static bool hasNestedLoopsWithProcessablePatterns(Function &F) {
  if (F.isDeclaration())
    return false;

  // Construct required analysis objects
  DominatorTree DT(F);
  LoopInfo LI(DT);
  AssumptionCache AC(F);

  // Create TargetLibraryInfoImpl using the module's target triple
  TargetLibraryInfoImpl TLII(F.getParent()->getTargetTriple());
  TargetLibraryInfo TLI(TLII);

  ScalarEvolution SE(F, TLI, AC, DT, LI);

  // Check nested loops for multiply-accumulate patterns
  for (Loop *L : LI.getLoopsInPreorder()) {
    if (!L->getSubLoops().empty()) {
      // Has nested loops, check if inner loops have multiply-accumulate pattern
      for (Loop *InnerL : L->getSubLoops()) {
        if (hasMultiplyAccumulatePattern(InnerL, SE)) {
          return true;
        }
      }
    }
  }

  return false;
}

bool RISCVDotprodSplitterPass::hasProcessablePattern(Function &F) {
  return hasNestedLoopsWithProcessablePatterns(F);
}

// Helper function to check if any function in module has processable patterns
static bool anyFunctionHasProcessablePatterns(Module &M) {
  for (Function &F : M) {
    if (RISCVDotprodSplitterPass::hasProcessablePattern(F)) {
      return true;
    }
  }
  return false;
}

bool RISCVConditionalLoopExtractorPass::moduleHasProcessablePatterns(
    Module &M) {
  return anyFunctionHasProcessablePatterns(M);
}

// Helper function to create analysis lookup functions
static auto createAnalysisLookupFunctions(FunctionAnalysisManager &FAM) {
  auto LookupDomTree = [&FAM](Function &F) -> DominatorTree & {
    return FAM.getResult<DominatorTreeAnalysis>(F);
  };

  auto LookupLoopInfo = [&FAM](Function &F) -> LoopInfo & {
    return FAM.getResult<LoopAnalysis>(F);
  };

  auto LookupAssumptionCache = [&FAM](Function &F) -> AssumptionCache * {
    return FAM.getCachedResult<AssumptionAnalysis>(F);
  };

  auto ShouldExtractFromFunction = [](Function &F) -> bool {
    return RISCVDotprodSplitterPass::hasProcessablePattern(F);
  };

  return std::make_tuple(LookupDomTree, LookupLoopInfo, LookupAssumptionCache,
                         ShouldExtractFromFunction);
}

PreservedAnalyses
RISCVConditionalLoopExtractorPass::run(Module &M, ModuleAnalysisManager &AM) {

  if (!EnableRISCVDotprodSplitter) {
    return PreservedAnalyses::all();
  }

  // Early exit if no processable patterns found
  if (!moduleHasProcessablePatterns(M)) {
    LLVM_DEBUG(dbgs() << "No processable dotprod patterns found in module "
                      << M.getName() << ". Skipping LoopExtractor.\n");
    return PreservedAnalyses::all();
  }

  LLVM_DEBUG(
      dbgs()
      << "Found processable patterns, running selective LoopExtractor\n");

  // Get function analysis manager and create lookup functions
  auto &FAM = AM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  auto [LookupDomTree, LookupLoopInfo, LookupAssumptionCache,
        ShouldExtractFromFunction] = createAnalysisLookupFunctions(FAM);

  // Run selective loop extractor
  SelectiveDotprodLoopExtractor Extractor(LookupDomTree, LookupLoopInfo,
                                          LookupAssumptionCache,
                                          ShouldExtractFromFunction);

  if (!Extractor.runOnModule(M)) {
    return PreservedAnalyses::all();
  }

  // Preserve loop analysis as required by LoopExtractor pattern
  PreservedAnalyses PA;
  PA.preserve<LoopAnalysis>();
  return PA;
}
