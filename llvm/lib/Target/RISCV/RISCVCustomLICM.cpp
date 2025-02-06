//===- RISCVCustomLICM.cpp - Custom Loop Invariant Code Motion -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file
// This file implements the RISCVCustomLICM pass, which performs custom loop
// invariant code motion optimizations specifically for ESP-DSP functions on
// RISC-V targets.
//
// **EXPERIMENTAL PASS**: This is an experimental implementation that explores
// domain-specific optimizations for Digital Signal Processing workloads. It
// serves as a research vehicle to investigate more aggressive LICM
// optimizations that can be safely applied when additional domain knowledge is
// available.
//
// This pass is designed as an experimental complement to the standard LICM
// pass, targeting domain-specific optimizations for Digital Signal Processing
// (DSP) workloads on ESP32-P4 and similar RISC-V processors.
//
// **Rationale for Custom LICM:**
//
// The standard LICM pass applies conservative analysis to ensure correctness
// across all possible code patterns. However, in well-defined DSP computation
// patterns, we can identify loop invariants with higher confidence through
// pattern recognition and domain knowledge. This allows us to:
//
// 1. **Relax Conservative Checks**: For recognized DSP patterns (biquad
// filters,
//    FIR filters, convolution, FFT, window functions), we can safely identify
//    and hoist instructions that standard LICM might consider unsafe due to
//    its general-purpose conservative analysis.
//
// 2. **Domain-Specific Pattern Recognition**: Unlike standard LICM which
// analyzes
//    individual instructions, this pass recognizes entire function structures
//    and applies specialized optimizations based on DSP computation patterns.
//
// 3. **ESP-DSP Specific Optimizations**: Targets specific code patterns
// generated
//    by ESP-DSP library functions, where we have additional guarantees about
//    data flow and loop behavior.
//
// **Implementation Status:**
// This is an experimental implementation that reimplements and extends some
// features of the standard LICM pass with domain-specific enhancements. The
// pass is designed to coexist with standard LICM and provides additional
// optimization opportunities through pattern-based analysis.
//
// **Main optimizations include:**
// 1. **Enhanced Load Hoisting**: More aggressive hoisting of loads in
// recognized
//    DSP patterns where data dependencies are well-understood
// 2. **Coefficient Pre-computation**: Moving negations and other coefficient
//    computations to loop preheaders, including algebraic rearrangements like
//    transforming `fneg(X) * Y` to `X * (-Y)` when Y is loop-invariant
// 3. **Specialized GEP Movement**: Advanced GEP instruction movement for
//    structured data access patterns common in DSP algorithms
// 4. **Window Function Optimization**: Complex transformations of
// trigonometric
//    computations in window functions (Blackman, Hann, etc.)
// 5. **DSP-Specific Instruction Reordering**: Grouping and reordering
// instructions
//    to improve pipeline efficiency on ESP32-P4 architecture
//
// **Coexistence with Standard LICM:**
// This experimental pass is designed to run alongside (typically after)
// standard LICM, providing additional optimization opportunities that standard
// LICM cannot safely perform due to its conservative nature. It does not
// replace standard LICM functionality but augments it with domain-specific
// knowledge.
//
// **Pattern Recognition Examples:**
// - Biquad filters: Recognizes 7 basic blocks with 5 parameters
// - FIRD filters: Identifies specific PHI node patterns (1 i16, 1 i32, 1 i64)
// - Convolution: Handles up to 4-level nested loops with specific GEP patterns
// - FFT: Recognizes complex number access patterns and fmuladd operations
// - Window functions: Optimizes cosine function call patterns
//
// **Example transformation:**
// \code
// Before optimization (conservative standard LICM):
// for.body:
//   %coeff = load float, ptr %coeffs_ptr, align 4  // Not hoisted due to
//   aliasing concerns %neg_coeff = fneg float %coeff                 // Not
//   moved due to dependency %result = fmul float %input, %neg_coeff
//
// After ESP-DSP Custom LICM (with pattern recognition):
// for.body.lr.ph:
//   %coeff = load float, ptr %coeffs_ptr, align 4  // Safely hoisted - coeffs
//   are read-only in DSP %neg_coeff = fneg float %coeff                 //
//   Pre-computed in preheader br label %for.body
// for.body:
//   %result = fmul float %input, %neg_coeff        // Uses pre-computed value
// \endcode
//
// **Safety Guarantees:**
// Pattern recognition ensures that optimizations are only applied to verified
// DSP computation structures where additional invariant properties can be
// safely assumed based on domain knowledge and function signatures.
//
// **Future Work:**
// This experimental implementation may inform future enhancements to the
// standard LICM pass or serve as a foundation for target-specific optimization
// frameworks in LLVM.
//
//===----------------------------------------------------------------------===//

#include "RISCVCustomLICM.h"
#include "RISCVESP32P4OptUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/AliasSetTracker.h"
#include "llvm/Analysis/Loads.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/ADCE.h"
#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/IndVarSimplify.h"
#include "llvm/Transforms/Scalar/LICM.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/SSAUpdater.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-custom-licm"

// Constant definitions - eliminate magic numbers
namespace {
// Biquad pattern constants
constexpr unsigned BIQUAD_EXPECTED_BB_COUNT = 7;
constexpr unsigned BIQUAD_EXPECTED_PARAM_COUNT = 5;
constexpr unsigned BIQUAD_MAX_LOOP_DEPTH = 1;
constexpr unsigned BIQUAD_EXPECTED_OUTER_LOOPS = 2;
constexpr unsigned BIQUAD_EXPECTED_INNER_LOOPS = 0;

// FIRD pattern constants
constexpr unsigned FIRD_EXPECTED_LOOP_COUNT = 3;
constexpr unsigned FIRD_PHI_I16_COUNT = 1;
constexpr unsigned FIRD_PHI_I32_COUNT = 1;
constexpr unsigned FIRD_PHI_I64_COUNT = 1;
constexpr unsigned FIRD_TOTAL_PHI_COUNT = 3;

// Convolution pattern constants
constexpr unsigned CONV_MAX_LOOP_DEPTH = 4;

// Window function constants
constexpr unsigned WINDOW_EXPECTED_PHI_COUNT = 1;
} // anonymous namespace

// Statistics
STATISTIC(NumBiquadFunctionsOptimized, "Number of biquad functions optimized");
STATISTIC(NumFIRDFunctionsOptimized, "Number of FIRD functions optimized");
STATISTIC(NumConvFunctionsOptimized,
          "Number of convolution functions optimized");
STATISTIC(NumFFTFunctionsOptimized, "Number of FFT functions optimized");
STATISTIC(NumWindowFunctionsOptimized, "Number of window functions optimized");
STATISTIC(NumInstructionsHoisted,
          "Number of instructions hoisted out of loops");
STATISTIC(NumFNegInstructionsMoved,
          "Number of FNeg instructions moved to preheader");

// Command line option to enable/disable RISCVCustomLICM.
cl::opt<bool> llvm::EnableRISCVCustomLICM(
    "riscv-custom-licm", cl::init(false),
    cl::desc(
        "Enable custom LICM optimizations for RISC-V DSP functions including "
        "biquad filters, FIR filters, convolution, FFT, and window functions"));

// Move fneg instructions out of the loop to reduce redundant computations.
// Since the input and output of fneg remain constant throughout the loop,
// extracting it to the preheader can improve performance.
// =>
// for.body.lr.ph:
//   ...
//   %neg = fneg float %4
//   %neg5 = fneg float %6
//   br label %for.body

// for.body:
//   ...

// Since the input and output of fneg remain constant throughout the loop,
// extracting it to the preheader can improve performance.
// This optimization moves fneg instructions out of the loop body
// to reduce redundant computations in each iteration.

// Function to move fneg instructions out of the loop
void BiquadCustomLICMOptimizer::moveFNegOutOfLoop(BasicBlock *Preheader,
                                                  BasicBlock &BB) {
  LLVM_DEBUG(dbgs() << "  Moving FNeg instructions from " << BB.getName()
                    << " to preheader\n");

  IRBuilder<> Builder(Preheader->getTerminator());
  SmallVector<Instruction *> InstructionsToRemove;
  unsigned NumMoved = 0;

  for (auto &I : BB) {
    if (auto *FNeg = dyn_cast<UnaryOperator>(&I)) {
      if (FNeg->getOpcode() == Instruction::FNeg) {
        Value *Operand = FNeg->getOperand(0);
        Value *NewFNeg =
            Builder.CreateFNeg(Operand, FNeg->getName() + ".hoisted");

        LLVM_DEBUG(dbgs() << "    Hoisting: " << *FNeg << "\n");
        LLVM_DEBUG(dbgs() << "    Created: " << *NewFNeg << "\n");

        FNeg->replaceAllUsesWith(NewFNeg);
        InstructionsToRemove.push_back(FNeg);
        ++NumMoved;
      }
    }
  }

  // Remove the old fneg instructions
  for (auto *I : InstructionsToRemove) {
    I->eraseFromParent();
  }

  NumFNegInstructionsMoved += NumMoved;
  LLVM_DEBUG(dbgs() << "  Moved " << NumMoved << " FNeg instructions\n");
}

// Adjust PHI nodes in the loop by swapping positions and updating values.
void BiquadCustomLICMOptimizer::adjustPhiNodes(BasicBlock &BB) {
  SmallVector<PHINode *> Phis;
  for (auto &I : BB.phis()) {
    Phis.push_back(&I);
  }

  if (Phis.size() >= 2) {
    PHINode *Phi1 = Phis[0];
    PHINode *Phi2 = Phis[1];

    // Use iterator instead of deprecated moveBefore
    Phi2->moveBefore(Phi1->getIterator());

    // Update the loop entry value of the second PHI node
    Value *LoopValue = Phi2;
    for (unsigned I = 0; I < Phi1->getNumIncomingValues(); ++I) {
      if (Phi1->getIncomingBlock(I) == &BB) {
        Phi1->setIncomingValue(I, LoopValue);
        break;
      }
    }
  }
}

// Create a cleanup block for proper loop exit handling.
void BiquadCustomLICMOptimizer::createCleanupBlock(Function &F,
                                                   BasicBlock &LoopBB) {
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

// Move store instructions out of the loop to the cleanup block.
void BiquadCustomLICMOptimizer::moveStoreOutOfLoop(BasicBlock &BB) {
  BasicBlock *CleanupBB = BB.getNextNode();
  if (!CleanupBB || CleanupBB->getName() != "for.cond.cleanup")
    return;

  // Use iterator instead of deprecated getFirstNonPHI()
  IRBuilder<> Builder(CleanupBB, CleanupBB->getFirstNonPHIIt());
  SmallVector<Instruction *> ToRemove;

  for (auto &I : BB) {
    if (auto *Store = dyn_cast<StoreInst>(&I)) {
      Value *Val1 = Store->getOperand(1);
      Value *Val0 = Store->getOperand(0);
      // Check if the stored value is defined in the current basic block
      if (!isa<Instruction>(Val1) ||
          cast<Instruction>(Val1)->getParent() != &BB) {
        Value *Ptr = Store->getPointerOperand();
        Builder.CreateStore(Val0, Ptr);
        ToRemove.push_back(Store);
      }
    }
  }

  // Remove the old store instructions
  for (auto *I : ToRemove) {
    I->eraseFromParent();
  }
}

// Check if an instruction is a must-tail call.
bool BiquadCustomLICMOptimizer::isMustTailCall(Instruction *I) {
  if (CallInst *CI = dyn_cast<CallInst>(I)) {
    return CI->isMustTailCall();
  }
  return false;
}

// Optimize a single loop by moving invariant instructions to preheader.
bool BiquadCustomLICMOptimizer::optimizeLoop(Loop *L, BasicBlock *Preheader,
                                             Function &F) {
  LLVM_DEBUG(dbgs() << "  Optimizing loop with header: "
                    << L->getHeader()->getName() << "\n");

  SmallVector<Instruction *, 8> InvariantInsts;
  unsigned NumHoisted = 0;

  for (auto &BB : L->blocks()) {
    if (BB->getName() != "for.body") {
      LLVM_DEBUG(dbgs() << "    Skipping block: " << BB->getName() << "\n");
      continue;
    }

    LLVM_DEBUG(dbgs() << "    Processing block: " << BB->getName() << "\n");

    for (auto &I : *BB) {
      if (L->hasLoopInvariantOperands(&I) && !isMustTailCall(&I)) {
        InvariantInsts.push_back(&I);
        LLVM_DEBUG(dbgs() << "      Found invariant: " << I << "\n");
      }
    }

    // Move loop invariant instructions
    for (auto *I : InvariantInsts) {
      I->moveBefore(Preheader->getTerminator()->getIterator());
      ++NumHoisted;
    }

    // Execute other optimizations
    moveFNegOutOfLoop(Preheader, *BB);
    adjustPhiNodes(*BB);
    createCleanupBlock(F, *BB);
    moveStoreOutOfLoop(*BB);
  }

  NumInstructionsHoisted += NumHoisted;
  LLVM_DEBUG(dbgs() << "  Hoisted " << NumHoisted
                    << " invariant instructions\n");

  return !InvariantInsts.empty();
}

// Check the number of basic blocks and parameters of the function.
bool BiquadCustomLICMOptimizer::checkBasicBlocksAndParameters(Function &F) {
  LLVM_DEBUG(dbgs() << "  Checking basic blocks and parameters\n");

  // Check number of basic blocks
  if (F.size() != BIQUAD_EXPECTED_BB_COUNT) {
    LLVM_DEBUG(dbgs() << "    Expected " << BIQUAD_EXPECTED_BB_COUNT
                      << " basic blocks, found " << F.size() << "\n");
    return false;
  }

  // Check number of parameters
  if (F.arg_size() != BIQUAD_EXPECTED_PARAM_COUNT) {
    LLVM_DEBUG(dbgs() << "    Expected " << BIQUAD_EXPECTED_PARAM_COUNT
                      << " parameters, found " << F.arg_size() << "\n");
    return false;
  }

  LLVM_DEBUG(dbgs() << "    Basic blocks and parameters check passed\n");
  return true;
}

// Check loop nesting depth constraints.
bool BiquadCustomLICMOptimizer::checkLoopNestingStructure(Function &F,
                                                          LoopInfo &LI) {
  LLVM_DEBUG(dbgs() << "  Checking loop nesting structure\n");

  // Check maximum loop depth
  unsigned MaxLoopDepth = 0;
  for (auto &BB : F) {
    MaxLoopDepth = std::max(MaxLoopDepth, LI.getLoopDepth(&BB));
  }

  if (MaxLoopDepth != BIQUAD_MAX_LOOP_DEPTH) {
    LLVM_DEBUG(dbgs() << "    Expected max loop depth " << BIQUAD_MAX_LOOP_DEPTH
                      << ", found " << MaxLoopDepth << "\n");
    return false;
  }

  // Check outer and inner loop counts
  int OuterLoopCount = 0;
  int InnerLoopCount = 0;
  for (Loop *L : LI.getLoopsInPreorder()) {
    if (L->getLoopDepth() == 1) {
      OuterLoopCount++;
      if (L->getSubLoops().size() > 0) {
        InnerLoopCount++;
      }
    }
  }

  bool IsValidNesting = (OuterLoopCount == BIQUAD_EXPECTED_OUTER_LOOPS &&
                         InnerLoopCount == BIQUAD_EXPECTED_INNER_LOOPS);

  if (!IsValidNesting) {
    LLVM_DEBUG(dbgs() << "    Expected " << BIQUAD_EXPECTED_OUTER_LOOPS
                      << " outer loops and " << BIQUAD_EXPECTED_INNER_LOOPS
                      << " inner loops, found " << OuterLoopCount << " and "
                      << InnerLoopCount << "\n");
    return false;
  }

  LLVM_DEBUG(dbgs() << "    Loop nesting check passed\n");
  return true;
}

// Check existence of required basic blocks and control flow structure.
bool BiquadCustomLICMOptimizer::checkBasicBlocksAndControlFlow(Function &F) {
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

// Main safety check function for biquad optimization.
bool BiquadCustomLICMOptimizer::isSafeToOptimizeBiquadType(Function &F,
                                                           LoopInfo &LI) {
  return checkBasicBlocksAndParameters(F) && checkLoopNestingStructure(F, LI) &&
         checkFMulAddUsage(F) && checkBasicBlocksAndControlFlow(F);
}

// Process a single biquad-type loop.
bool BiquadCustomLICMOptimizer::processBiquadTypeLoop(Loop *L,
                                                      DominatorTree &DT,
                                                      LoopInfo &LI,
                                                      Function &F) {
  if (L->getLoopDepth() != 1 || L->getBlocks().empty())
    return false;

  BasicBlock *Preheader = L->getLoopPreheader();
  if (!Preheader) {
    Preheader = InsertPreheaderForLoop(L, &DT, &LI, nullptr, true);
    if (!Preheader)
      return false;
  }

  return optimizeLoop(L, Preheader, F);
}

// Process all loops in the function.
bool BiquadCustomLICMOptimizer::processBiquadTypeAllLoops(Function &F,
                                                          DominatorTree &DT,
                                                          LoopInfo &LI) {
  bool Changed = false;
  for (auto &L : LI) {
    Changed |= processBiquadTypeLoop(L, DT, LI, F);
  }
  return Changed;
}

SmallVector<GetElementPtrInst *>
FIRDCustomLICMOptimizer::getGEPInstructions(BasicBlock *BB) {
  SmallVector<GetElementPtrInst *> GEPs;
  for (auto &I : *BB) {
    if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
      // Only process struct member access
      if (GEP->getNumIndices() == 2 &&
          GEP->getSourceElementType()->isStructTy()) {
        GEPs.push_back(GEP);
      }
    }
  }
  return GEPs;
}

void FIRDCustomLICMOptimizer::modifyGEPAndLoadInstruction(
    GetElementPtrInst *GEP, ICmpInst *LastICmp) {

  // Move GEP to the beginning of IfEnd
  GEP->moveBefore(LastICmp->getIterator());

  // Get the first LoadInst and move it after GEP
  LoadInst *FirstLoad = nullptr;
  SmallVector<LoadInst *> OtherLoads;
  for (auto *U : GEP->users()) {
    if (auto *LI = dyn_cast<LoadInst>(U)) {
      if (!FirstLoad) {
        FirstLoad = LI;
      } else {
        OtherLoads.push_back(LI);
      }
    }
  }

  // Use iterator
  FirstLoad->moveBefore(GEP->getNextNode()->getIterator());

  // Replace other LoadInsts with the first LoadInst
  for (auto *OtherLoad : OtherLoads) {
    OtherLoad->replaceAllUsesWith(FirstLoad);
    OtherLoad->eraseFromParent();
  }
}

bool FIRDCustomLICMOptimizer::transformLoadInstructions(
    SmallVector<GetElementPtrInst *> GEPs, ICmpInst *LastICmp) {
  bool Changed = false;
  for (auto *GEP : GEPs) {
    // Check if all users of GEP are load instructions
    bool AllLoads = true;
    for (auto *U : GEP->users()) {
      if (!isa<LoadInst>(U)) {
        AllLoads = false;
        break;
      }
    }

    if (AllLoads && !GEP->users().empty()) {
      modifyGEPAndLoadInstruction(GEP, LastICmp);
      Changed = true;
    }
  }
  return Changed;
}

// Refactor checkInt16FIRDPhiNodes function
bool FIRDCustomLICMOptimizer::checkInt16FIRDPhiNodes(BasicBlock *ForBody) {
  LLVM_DEBUG(dbgs() << "  Checking FIRD PHI node pattern\n");

  unsigned I16Count = 0;
  unsigned I32Count = 0;
  unsigned I64Count = 0;
  unsigned TotalPhiCount = 0;

  for (auto &Phi : ForBody->phis()) {
    TotalPhiCount++;
    if (Phi.getType()->isIntegerTy(16)) {
      I16Count++;
    } else if (Phi.getType()->isIntegerTy(32)) {
      I32Count++;
    } else if (Phi.getType()->isIntegerTy(64)) {
      I64Count++;
    }
  }

  bool IsValidPattern =
      (TotalPhiCount == FIRD_TOTAL_PHI_COUNT &&
       I16Count == FIRD_PHI_I16_COUNT && I32Count == FIRD_PHI_I32_COUNT &&
       I64Count == FIRD_PHI_I64_COUNT);

  if (!IsValidPattern) {
    LLVM_DEBUG(dbgs() << "    Expected PHI pattern: " << FIRD_TOTAL_PHI_COUNT
                      << " total (" << FIRD_PHI_I16_COUNT << " i16, "
                      << FIRD_PHI_I32_COUNT << " i32, " << FIRD_PHI_I64_COUNT
                      << " i64), found: " << TotalPhiCount << " total ("
                      << I16Count << " i16, " << I32Count << " i32, "
                      << I64Count << " i64)\n");
    return false;
  }

  LLVM_DEBUG(dbgs() << "    FIRD PHI node pattern check passed\n");
  return true;
}

bool FIRDCustomLICMOptimizer::checkInt16FIRDUnrollPattern(Loop &L) {
  if (L.getLoopDepth() == 1) {
    return false;
  }
  BasicBlock *loopHeader = L.getHeader();
  if (loopHeader->getTerminator()->getNumSuccessors() != 2) {
    return false;
  }
  if (loopHeader->getTerminator()->getSuccessor(1) != loopHeader) {
    return false;
  }
  return true;
}

// Refactor checkInt16FIRDType function
bool FIRDCustomLICMOptimizer::checkInt16FIRDType(Function &F, LoopInfo &LI) {
  LLVM_DEBUG(dbgs() << "Checking Int16 FIRD pattern for function: "
                    << F.getName() << "\n");

  unsigned LoopCount = 0;
  for (Loop *L : LI) {
    for (Loop *SubL : L->getSubLoops()) {
      LoopCount++;
      if (!checkInt16FIRDUnrollPattern(*SubL)) {
        LLVM_DEBUG(dbgs() << "  Unroll pattern check failed for subloop\n");
        continue;
      }

      BasicBlock *ForBodyLrPh = SubL->getLoopPreheader();
      BasicBlock *ForBody = SubL->getHeader();
      if (!ForBodyLrPh || !ForBody) {
        LLVM_DEBUG(dbgs() << "  Missing preheader or header\n");
        return false;
      }

      LoadInst *LI = getFirstInst<LoadInst>(ForBodyLrPh);
      if (!LI) {
        LLVM_DEBUG(dbgs() << "  No load instruction found in preheader\n");
        return false;
      }

      if (!checkInt16FIRDPhiNodes(ForBody)) {
        LLVM_DEBUG(dbgs() << "  PHI node pattern check failed\n");
        return false;
      }
    }
  }

  bool IsValidLoopCount = (LoopCount == FIRD_EXPECTED_LOOP_COUNT);
  if (!IsValidLoopCount) {
    LLVM_DEBUG(dbgs() << "  Expected " << FIRD_EXPECTED_LOOP_COUNT
                      << " loops, found " << LoopCount << "\n");
  }

  return IsValidLoopCount;
}

// Extract common window function check logic
template <unsigned N>
bool DspsWindCustomLICMOptimizer<N>::checkDspsWindCommonF32Type(
    Function &F, unsigned NumCosf) {
  // 1. Check basic block structure
  BasicBlock &EntryBB = F.getEntryBlock();
  // Remove unnecessary null checks because getEntryBlock() always returns a
  // valid reference

  Instruction *Terminator = EntryBB.getTerminator();
  if (Terminator->getNumSuccessors() == 0) {
    return false;
  }
  BasicBlock *ForBodyLrPh = EntryBB.getTerminator()->getSuccessor(0);
  if (!ForBodyLrPh)
    return false;

  BasicBlock *ForBody = ForBodyLrPh->getSingleSuccessor();
  if (!ForBody)
    return false;

  // 2. Check floating point multiplication instruction in for.body
  bool HasFMulInst = false;
  for (auto &I : *ForBody) {
    if (auto *FMul = dyn_cast<BinaryOperator>(&I)) {
      if (FMul->getOpcode() == Instruction::FMul &&
          FMul->getOperand(0)->getType()->isDoubleTy() &&
          isa<Constant>(FMul->getOperand(1))) {
        HasFMulInst = true;
        break;
      }
    }
  }
  if (!HasFMulInst)
    return false;

  // 3. Check conversion instruction in for.body.lr.ph
  if (!ForBodyLrPh->getTerminator()->getPrevNode())
    return false;

  // 4. Check phi nodes
  unsigned PhiCount = 0;
  for (auto &I : *ForBody) {
    if (auto *PN = dyn_cast<PHINode>(&I)) {
      if (PN->getType()->isIntegerTy(32)) {
        PhiCount++;
      }
    }
  }
  if (PhiCount != 1)
    return false;

  // 5. Check cosf calls - Fix symbol comparison warning
  unsigned CosfCount = 0;
  for (auto &I : *ForBody) {
    if (auto *Call = dyn_cast<CallInst>(&I)) {
      if (Call->getCalledFunction() &&
          Call->getCalledFunction()->getName() == "cosf") {
        CosfCount++;
      }
    }
  }
  return CosfCount == NumCosf; // Now both are unsigned types
}

bool FIRDCustomLICMOptimizer::transformFIRDTypeGEPLoad(Function &F,
                                                       DominatorTree &DT,
                                                       LoopInfo &LI) {
  bool Changed = false;

  for (auto *L : LI) {
    if (L->getLoopDepth() == 1) {
      BasicBlock *ForCondPreheaderLrPh = L->getLoopPreheader();
      if (!ForCondPreheaderLrPh)
        return false;
      BasicBlock *IfEnd = ForCondPreheaderLrPh->getSinglePredecessor();
      if (!IfEnd)
        return false;
      ICmpInst *LastICmp = getLastInst<ICmpInst>(IfEnd);
      SmallVector<GetElementPtrInst *> GEPs =
          getGEPInstructions(ForCondPreheaderLrPh);
      Changed = transformLoadInstructions(GEPs, LastICmp);
    }
  }

  return Changed;
}

// Refactor checkDspiConvF32Type function
bool ConvCustomLICMOptimizer::checkDSPIConvF32Type(Function &F, LoopInfo &LI) {
  LLVM_DEBUG(dbgs() << "Checking DSPI convolution F32 pattern for function: "
                    << F.getName() << "\n");

  // 1. Check loop nesting structure
  unsigned MaxDepth = 0;
  unsigned LoopCount = 0;
  for (auto *L : LI) {
    LoopCount++;
    unsigned Depth = 1;
    Loop *CurrLoop = L;
    while (!CurrLoop->getSubLoops().empty()) {
      Depth++;
      CurrLoop = CurrLoop->getSubLoops().front();
    }
    MaxDepth = std::max(MaxDepth, Depth);
  }

  // Require maximum nesting depth not exceed 4 levels
  if (MaxDepth != CONV_MAX_LOOP_DEPTH) {
    LLVM_DEBUG(dbgs() << "  Expected max loop depth " << CONV_MAX_LOOP_DEPTH
                      << ", found " << MaxDepth << "\n");
    return false;
  }

  // 2. Check instruction patterns in loops
  for (auto *L : LI) {
    bool HasLoadInst = false;
    bool HasGEPInst = false;

    for (auto &BB : L->blocks()) {
      for (auto &I : *BB) {
        // Fix: Use different variable names to avoid type name conflict
        if (auto *LI = dyn_cast<LoadInst>(&I)) {
          HasLoadInst = true;
          // Check if load instruction's operand is GEP
          if (dyn_cast<GetElementPtrInst>(LI->getPointerOperand())) {
            HasGEPInst = true;
          }
        }
      }
    }

    // Each loop must contain load and GEP instructions
    if (!HasLoadInst || !HasGEPInst) {
      LLVM_DEBUG(dbgs() << "  Loop missing required load/GEP instructions\n");
      return false;
    }
  }

  // 3. Check loop invariants
  for (auto *L : LI) {
    bool HasInvariantInst = false;
    for (auto &BB : L->blocks()) {
      for (auto &I : *BB) {
        if (L->hasLoopInvariantOperands(&I) && !isa<BranchInst>(&I)) {
          HasInvariantInst = true;
          break;
        }
      }
    }
    // Each loop must contain loop invariants
    if (!HasInvariantInst) {
      LLVM_DEBUG(dbgs() << "  Loop has no invariant instructions\n");
      return false;
    }
  }

  LLVM_DEBUG(dbgs() << "  DSPI convolution F32 pattern check passed\n");
  return true;
}

bool DspsFft2rFc32CustomLICMOptimizer::checkDspsFft2rFc32Type(Function &F,
                                                              LoopInfo &LI) {
  // Check if has loops
  if (LI.empty())
    return false;

  // Must have exactly one top-level loop
  if (LI.getTopLevelLoops().size() != 1)
    return false;

  Loop *L1 = LI.getTopLevelLoops()[0];

  // Must have exactly one second-level loop
  if (L1->getSubLoops().size() != 1)
    return false;

  Loop *L2 = L1->getSubLoops()[0];

  // Must have exactly one third-level loop
  if (L2->getSubLoops().size() != 1)
    return false;

  Loop *L3 = L2->getSubLoops()[0];

  // Third level loop should not have sub loops
  if (!L3->getSubLoops().empty())
    return false;

  return checkFFTComputationPattern(F, LI);
}

// Refactor analyzeFFTPatterns function
DspsFft2rFc32CustomLICMOptimizer::FFTPatternAnalysis
DspsFft2rFc32CustomLICMOptimizer::analyzeFFTPatterns(BasicBlock *Header) {
  FFTPatternAnalysis Analysis;

  for (Instruction &I : *Header) {
    // Check for fmuladd instructions
    if (RecurrenceDescriptor::isFMulAddIntrinsic(&I)) {
      Analysis.HasFMulAdd = true;
    }

    // Check for fneg + fmul pattern
    if (I.getOpcode() == Instruction::FNeg) {
      for (User *U : I.users()) {
        if (isa<BinaryOperator>(U) &&
            cast<BinaryOperator>(U)->getOpcode() == Instruction::FMul) {
          Analysis.HasFNegFMulPattern = true;
          break;
        }
      }
    }

    // Count floating point operations
    if (I.getOpcode() == Instruction::FAdd) {
      Analysis.FAddCount++;
    } else if (I.getOpcode() == Instruction::FSub) {
      Analysis.FSubCount++;
    }

    // Check for complex number access pattern
    if (auto *BinOp = dyn_cast<BinaryOperator>(&I)) {
      if (BinOp->getOpcode() == Instruction::Or) {
        if (auto *C = dyn_cast<ConstantInt>(BinOp->getOperand(1))) {
          if (C->getZExtValue() == 1) {
            Analysis.HasComplexAccess = true;
          }
        }
      }
    }
  }

  return Analysis;
}

// Extract FFT pattern validation logic
bool DspsFft2rFc32CustomLICMOptimizer::isValidFFTPattern(
    const FFTPatternAnalysis &Analysis) {

  bool HasRequiredOperations =
      Analysis.HasFMulAdd && Analysis.HasFNegFMulPattern;
  bool HasBalancedArithmetic =
      (Analysis.FAddCount > 0 && Analysis.FSubCount > 0);
  bool HasComplexDataAccess = Analysis.HasComplexAccess;

  LLVM_DEBUG(dbgs() << "FFT Pattern Analysis:\n");
  LLVM_DEBUG(dbgs() << "  FMulAdd: " << Analysis.HasFMulAdd << "\n");
  LLVM_DEBUG(dbgs() << "  FNeg+FMul: " << Analysis.HasFNegFMulPattern << "\n");
  LLVM_DEBUG(dbgs() << "  FAdd count: " << Analysis.FAddCount << "\n");
  LLVM_DEBUG(dbgs() << "  FSub count: " << Analysis.FSubCount << "\n");
  LLVM_DEBUG(dbgs() << "  Complex access: " << Analysis.HasComplexAccess
                    << "\n");

  return HasRequiredOperations && HasBalancedArithmetic && HasComplexDataAccess;
}

// FFT computation pattern check function
bool DspsFft2rFc32CustomLICMOptimizer::checkFFTComputationPattern(
    Function &F, LoopInfo &LI) {
  LLVM_DEBUG(dbgs() << "Checking FFT computation pattern\n");

  // First get the top level loop, then find the innermost loop
  Loop *InnermostLoop = nullptr;
  for (auto *L : LI) {
    InnermostLoop = findInnermostLoops(L);
    if (InnermostLoop)
      break;
  }

  if (!InnermostLoop) {
    LLVM_DEBUG(dbgs() << "No innermost loop found\n");
    return false;
  }

  BasicBlock *Header = InnermostLoop->getHeader();
  if (!Header) {
    LLVM_DEBUG(dbgs() << "No loop header found\n");
    return false;
  }

  // Analyze FFT pattern
  auto Analysis = analyzeFFTPatterns(Header);
  bool IsValid = isValidFFTPattern(Analysis);

  LLVM_DEBUG(dbgs() << "FFT pattern check " << (IsValid ? "passed" : "failed")
                    << "\n");
  return IsValid;
}

// Extract basic block get logic
template <unsigned N>
typename DspsWindCustomLICMOptimizer<N>::WindowFunctionBlocks
DspsWindCustomLICMOptimizer<N>::getWindowFunctionBlocks(Function &F) {
  WindowFunctionBlocks Blocks;

  Blocks.EntryBB = &F.getEntryBlock();

  Instruction *Terminator = Blocks.EntryBB->getTerminator();
  if (Terminator->getNumSuccessors() == 0) {
    return {}; // Return default constructed structure, all pointers are nullptr
  }

  Blocks.ForBodyLrPh = Terminator->getSuccessor(0);
  if (!Blocks.ForBodyLrPh) {
    return {};
  }

  Blocks.ForBody = Blocks.ForBodyLrPh->getSingleSuccessor();
  if (!Blocks.ForBody) {
    return {};
  }

  return Blocks;
}

// Extract FMul instruction find logic
template <unsigned N>
std::pair<BinaryOperator *, Value *>
DspsWindCustomLICMOptimizer<N>::findFMulInstructionAndConstant(
    BasicBlock *ForBody) {
  for (auto &I : *ForBody) {
    if (auto *FMul = dyn_cast<BinaryOperator>(&I)) {
      if (FMul->getOpcode() == Instruction::FMul &&
          FMul->getOperand(0)->getType()->isDoubleTy() &&
          isa<Constant>(FMul->getOperand(1))) {
        return {FMul, FMul->getOperand(1)};
      }
    }
  }
  return {nullptr, nullptr};
}

// Extract multiplication instruction create logic
template <unsigned N>
SmallVector<Value *, 4>
DspsWindCustomLICMOptimizer<N>::createMultiplicationInstructions(
    IRBuilder<> &Builder, Instruction *Conv1, Value *FMulConst, Function &F,
    unsigned NumCosf) {

  // Create doubled constant
  Value *DoubleFMulConst = ConstantFP::get(
      FMulConst->getType(),
      cast<ConstantFP>(FMulConst)->getValueAPF().convertToDouble() * 2.0);

  Value *Mul = Builder.CreateFMul(Conv1, DoubleFMulConst, "mul");
  Value *Conv2 =
      Builder.CreateFPTrunc(Mul, Type::getFloatTy(F.getContext()), "conv2");

  // Create mul3-mulN
  SmallVector<Value *, 4> Muls;
  Muls.push_back(Conv2);

  for (unsigned i = 2; i <= NumCosf; i++) {
    Value *MulN = Builder.CreateFMul(
        Conv2, ConstantFP::get(Type::getFloatTy(F.getContext()), i),
        "mul" + std::to_string(i + 1));
    Muls.push_back(MulN);
  }

  return Muls;
}

// Extract cosine function call update logic
template <unsigned N>
bool DspsWindCustomLICMOptimizer<N>::updateCosineFunctionCalls(
    BasicBlock *ForBody, const SmallVector<Value *, 4> &Muls,
    unsigned NumCosf) {

  // Collect all cosf calls
  SmallVector<CallInst *, 4> CosfCalls;
  for (auto &I : *ForBody) {
    if (auto *Call = dyn_cast<CallInst>(&I)) {
      if (Call->getCalledFunction() &&
          Call->getCalledFunction()->getName() == "cosf") {
        CosfCalls.push_back(Call);
      }
    }
  }

  // Check cosf call count - Fix type comparison
  if (CosfCalls.size() != NumCosf) {
    LLVM_DEBUG(dbgs() << "Expected " << NumCosf << " cosf calls, found "
                      << CosfCalls.size() << "\n");
    return false;
  }

  // Update cosf parameters in order
  IRBuilder<> Builder(ForBody->getContext());
  for (unsigned i = 0; i < NumCosf; i++) {
    Builder.SetInsertPoint(CosfCalls[i]);
    Value *MulN =
        Builder.CreateFMul(Muls[i], cast<Value>(&*ForBody->getFirstNonPHIIt()),
                           "mul" + std::to_string(3 * i + 6));
    CosfCalls[i]->setArgOperand(0, MulN);
  }

  return true;
}

template <unsigned N>
bool DspsWindCustomLICMOptimizer<N>::transformDspsWindCommonF32Type(
    Function &F, unsigned NumCosf) {

  LLVM_DEBUG(dbgs() << "Transforming window function with " << NumCosf
                    << " cosine calls\n");

  // Get basic blocks
  auto Blocks = getWindowFunctionBlocks(F);
  if (!Blocks.EntryBB || !Blocks.ForBodyLrPh || !Blocks.ForBody) {
    LLVM_DEBUG(dbgs() << "Failed to get required basic blocks\n");
    return false;
  }

  // Find floating point multiplication instruction and constant
  auto [FMulInst, FMulConst] = findFMulInstructionAndConstant(Blocks.ForBody);
  if (!FMulInst || !FMulConst) {
    LLVM_DEBUG(dbgs() << "Failed to find FMul instruction and constant\n");
    return false;
  }

  // Get conversion instruction
  Instruction *Conv1 = Blocks.ForBodyLrPh->getTerminator()->getPrevNode();
  if (!Conv1) {
    LLVM_DEBUG(dbgs() << "Failed to find conversion instruction\n");
    return false;
  }

  // Create new instructions in preheader
  IRBuilder<> Builder(Blocks.ForBodyLrPh->getTerminator());
  auto Muls =
      createMultiplicationInstructions(Builder, Conv1, FMulConst, F, NumCosf);

  // Get phi node and insert sitofp instruction
  PHINode *PhiNode = getFirstI32Phi(Blocks.ForBody);
  if (!PhiNode) {
    LLVM_DEBUG(dbgs() << "Failed to find i32 PHI node\n");
    return false;
  }

  Builder.SetInsertPoint(Blocks.ForBody, Blocks.ForBody->getFirstNonPHIIt());
  Builder.CreateSIToFP(PhiNode, Type::getFloatTy(F.getContext()), "conv5");

  // Update cosine function calls
  bool Success = updateCosineFunctionCalls(Blocks.ForBody, Muls, NumCosf);
  if (!Success) {
    return false;
  }

  // Clean up dead code
  runDeadCodeElimination(F);

  LLVM_DEBUG(
      dbgs() << "Window function transformation completed successfully\n");
  return true;
}

bool ConvCustomLICMOptimizer::hoistInstructionsFromLoops(LoopInfo &LI,
                                                         BasicBlock &entryBB,
                                                         Function &F) {
  bool Changed = false;
  // Store all instructions from preheader blocks
  SmallVector<Instruction *, 16> commonInstructions;

  // Iterate through all loops
  for (auto *L : LI) {
    BasicBlock *forBodyLrPh = L->getLoopPreheader();
    if (!forBodyLrPh)
      continue;
    // Iterate through instructions in preheader
    for (auto &I : *forBodyLrPh) {
      // Skip icmp and br instructions
      if (isa<ICmpInst>(I) || isa<BranchInst>(I))
        continue;
      // Check if instruction already exists in commonInstructions
      bool isCommon = true;
      for (auto *CI : commonInstructions) {
        if (!I.isIdenticalTo(CI)) {
          isCommon = false;
          break;
        }
      }

      // If instruction is common, add to commonInstructions
      if (isCommon) {
        commonInstructions.push_back(&I);
      }
    }
  }

  // Move common instructions to entryBB
  for (auto *CI : commonInstructions) {
    // Clone instruction to entryBB
    CI->moveBefore(entryBB.getTerminator()->getIterator());
    Changed = true;
    // Delete redundant instructions from preheader blocks
    for (auto *L : LI) {
      BasicBlock *forBodyLrPh = L->getLoopPreheader();
      if (!forBodyLrPh)
        continue;

      for (auto &I : *forBodyLrPh) {
        if (CI->isIdenticalTo(&I)) {
          break;
        }
      }
    }
  }

  return Changed;
}

// Fix hoistInstructionsFromSubLoop function deprecated API
bool ConvCustomLICMOptimizer::hoistInstructionsFromSubLoop(Loop *L, int Depth) {
  // Get loop header block
  BasicBlock *Header = L->getHeader();
  // Collect instructions from preheaders of all subloops
  SmallVector<Instruction *, 16> CommonInstructions;
  SmallVector<Instruction *, 16> CandidateInstructions;

  bool Changed = true;
  while (Changed) {
    Changed = false;
    CandidateInstructions.clear();

    // Iterate first subloop to collect candidate instructions
    auto SubLoops = L->getSubLoops();
    if (!SubLoops.empty()) {
      BasicBlock *FirstPreheader = SubLoops.front()->getLoopPreheader();
      if (FirstPreheader) {
        for (auto &I : *FirstPreheader) {
          if (!isa<BranchInst>(I)) {
            CandidateInstructions.push_back(&I);
          }
        }
      }
    }

    // Check other subloops to verify instructions exist in all
    for (auto it = std::next(SubLoops.begin()); it != SubLoops.end(); ++it) {
      BasicBlock *Preheader = (*it)->getLoopPreheader();
      if (!Preheader)
        continue;

      SmallVector<Instruction *, 16> MatchedInsts;
      for (auto *CandInst : CandidateInstructions) {
        for (auto &I : *Preheader) {
          if (CandInst->isIdenticalTo(&I)) {
            MatchedInsts.push_back(CandInst);
            break;
          }
        }
      }

      // Update candidate instructions to currently matched ones
      CandidateInstructions = MatchedInsts;
    }

    // Add all candidate instructions to common instructions set
    if (!CandidateInstructions.empty()) {
      CommonInstructions = CandidateInstructions;

      // Move common instructions to Header
      if (Header) {
        for (auto *CI : CommonInstructions) {
          // Use iterator instead of deprecated moveBefore
          CI->moveBefore(Header->getTerminator()->getIterator());

          // Delete redundant instructions from preheader blocks
          for (auto *SubL : L->getSubLoops()) {
            BasicBlock *Preheader = SubL->getLoopPreheader();
            if (!Preheader)
              continue;

            for (auto &I : *Preheader) {
              if (CI->isIdenticalTo(&I)) {
                I.replaceAllUsesWith(CI);
                I.eraseFromParent();
                Changed = true;
                break;
              }
            }
          }
        }
      }
    }
  }

  return Changed;
}

// Collect loop invariant instructions and related GEP instructions
void ConvCustomLICMOptimizer::collectLoopInvariantInstructions(
    Loop *L, SmallVectorImpl<Instruction *> &InvariantInsts) {
  // Iterate through all basic blocks in loop
  for (BasicBlock *BB : L->blocks()) {
    for (Instruction &I : *BB) {
      // Skip branch instructions
      if (isa<BranchInst>(I))
        continue;

      // Check if instruction is loop invariant
      if (L->hasLoopInvariantOperands(&I)) {
        InvariantInsts.push_back(&I);

        // For load instructions, check and collect GEP operands
        if (auto *LoadI = dyn_cast<LoadInst>(&I)) {
          if (auto *GEPI =
                  dyn_cast<GetElementPtrInst>(LoadI->getPointerOperand())) {
            LLVM_DEBUG(GEPI->dump());
            InvariantInsts.push_back(GEPI);
          }
        }
      }
    }
  }
}

bool ConvCustomLICMOptimizer::transformDSPIConvF32Type(Function &F,
                                                       LoopInfo &LI) {
  runDeadCodeElimination(F);
  BasicBlock &EntryBB = F.getEntryBlock();
  SmallVector<Instruction *, 8> InvariantInsts;
  for (Loop *L : LI) {
    collectLoopInvariantInstructions(L, InvariantInsts);
  }
  Instruction *InsertPoint = EntryBB.getTerminator()->getPrevNode();
  for (auto &I : InvariantInsts) {
    I->moveAfter(InsertPoint);
  }

  bool Changed = true;
  while (Changed) {
    Changed = hoistInstructionsFromLoops(LI, EntryBB, F);
  }

  runPostPass(F);
  for (auto *L : LI) {
    hoistInstructionsFromSubLoop(L, 1);
    for (auto *L2 : L->getSubLoops()) {
      hoistInstructionsFromSubLoop(L2, 2);
      for (auto *L3 : L2->getSubLoops()) {
        hoistInstructionsFromSubLoop(L3, 3);
        for (auto *L4 : L3->getSubLoops()) {
          (void)L4; // Avoid unused variable warning
        }
      }
    }
  }
  return true;
}

// Recursive function to find innermost loops
Loop *DspsFft2rFc32CustomLICMOptimizer::findInnermostLoops(Loop *L) {
  // If current loop has no subloops, it is innermost
  if (L->isInnermost()) {
    return L;
  } else {
    // Recursively check subloops
    for (Loop *SubLoop : L->getSubLoops()) {
      return findInnermostLoops(SubLoop);
    }
  }
  llvm_unreachable("Unhandled loop nesting in findInnermostLoops");
  return nullptr;
}

void DspsFft2rFc32CustomLICMOptimizer::groupSameInstructionDspsFft2rFc32(
    BasicBlock *forBody) {
  // Collect different types of instructions
  SmallVector<PHINode *> phiNodes;
  SmallVector<Instruction *> addInsts, shlInsts, orInsts, gepInsts, loadInsts,
      fmulInsts, fsubInsts, faddInsts, fmuladdInsts, fnegInsts;

  // Categorize instructions by type
  for (Instruction &I : *forBody) {
    if (auto *phi = dyn_cast<PHINode>(&I)) {
      phiNodes.push_back(phi);
    } else if (I.getOpcode() == Instruction::Or) {
      orInsts.push_back(&I);
    } else if (isa<GetElementPtrInst>(&I)) {
      gepInsts.push_back(&I);
    } else if (isa<LoadInst>(&I)) {
      loadInsts.push_back(&I);
    } else if (I.getOpcode() == Instruction::FAdd) {
      faddInsts.push_back(&I);
    } else if (I.getOpcode() == Instruction::FSub) {
      fsubInsts.push_back(&I);
    } else if (I.getOpcode() == Instruction::FMul) {
      fmulInsts.push_back(&I);
    } else if (I.getOpcode() == Instruction::FNeg) {
      fnegInsts.push_back(&I);
    } else if (auto *mulInst = dyn_cast<BinaryOperator>(&I)) {
      if (mulInst->getOpcode() == Instruction::Add) {
        addInsts.push_back(mulInst);
      } else if (mulInst->getOpcode() == Instruction::Shl) {
        shlInsts.push_back(mulInst);
      }
    } else if (RecurrenceDescriptor::isFMulAddIntrinsic(&I)) {
      fmuladdInsts.push_back(&I);
    }
  }

  // If no PHI nodes are found, return
  if (phiNodes.empty()) {
    return;
  }

  // Reorder instructions
  Instruction *insertPoint = phiNodes.back()->getNextNode();

  // Move instructions in the desired order
  moveInstructionsBeforePoint(addInsts, insertPoint);
  moveInstructionsBeforePoint(shlInsts, insertPoint);
  moveInstructionsBeforePoint(orInsts, insertPoint);
  moveInstructionsBeforePoint(gepInsts, insertPoint);
  moveInstructionsBeforePoint(loadInsts, insertPoint);
  moveInstructionsBeforePoint(fnegInsts, insertPoint);
  moveInstructionsBeforePoint(fmulInsts, insertPoint);
  moveInstructionsBeforePoint(fmuladdInsts, insertPoint);
  moveInstructionsBeforePoint(faddInsts, insertPoint);
  moveInstructionsBeforePoint(fsubInsts, insertPoint);
}

void DspsFft2rFc32CustomLICMOptimizer::hoistLoopInvariantFNeg(Loop *L,
                                                              DominatorTree &DT,
                                                              LoopInfo &LI) {
  BasicBlock *preheader = L->getLoopPreheader();

  if (!preheader) {
    preheader = InsertPreheaderForLoop(L, &DT, &LI, nullptr, false);
    if (!preheader)
      return;
  }

  BasicBlock *header = L->getHeader();

  // Find fneg + fmul pattern to rearrange
  for (Instruction &I : *header) {
    if (I.getOpcode() == Instruction::FNeg) {
      // Find fmul instruction using this fneg result
      for (User *U : I.users()) {
        if (auto *FMul = dyn_cast<BinaryOperator>(U)) {
          if (FMul->getOpcode() == Instruction::FMul) {
            // Check if the other operand of fmul is loop invariant
            Value *OtherOp = (FMul->getOperand(0) == &I) ? FMul->getOperand(1)
                                                         : FMul->getOperand(0);

            if (auto *OtherInst = dyn_cast<Instruction>(OtherOp)) {
              if (!L->contains(OtherInst)) {
                // Other operand is loop invariant, can do algebraic
                // rearrangement Create (-OtherOp) in preheader
                IRBuilder<> Builder(preheader->getTerminator());
                Value *NegOtherOp =
                    Builder.CreateFNeg(OtherOp, "neg_" + OtherOp->getName());

                // Replace original fneg(X) * Y with X * (-Y)
                Value *OrigX = I.getOperand(0); // fneg's operand
                IRBuilder<> LoopBuilder(FMul);
                Value *NewMul =
                    LoopBuilder.CreateFMul(OrigX, NegOtherOp, "rearranged_mul");

                FMul->replaceAllUsesWith(NewMul);
                FMul->eraseFromParent();

                // If fneg has no other users, also delete it
                if (I.use_empty()) {
                  I.eraseFromParent();
                }

                LLVM_DEBUG(dbgs() << "Applied algebraic rearrangement for "
                                     "fneg+fmul pattern\n");
                return; // Only process one pattern, avoid iterator invalidation
              }
            }
          }
        }
      }
    }
  }
}

bool DspsFft2rFc32CustomLICMOptimizer::transformDspsFft2rFc32Type(
    Function &F, DominatorTree &DT, LoopInfo &LI) {
  Loop *innermostLoop = nullptr;
  for (auto *L : LI) {
    innermostLoop = findInnermostLoops(L);
  }
  hoistLoopInvariantFNeg(innermostLoop, DT, LI);

  BasicBlock *forBody = innermostLoop->getHeader();

  groupSameInstructionDspsFft2rFc32(forBody);
  return true;
}

// Extract strategy create logic
std::vector<std::unique_ptr<RISCVCustomLICMOptimizationStrategy>>
RISCVCustomLICMPass::createOptimizationStrategies() {
  std::vector<std::unique_ptr<RISCVCustomLICMOptimizationStrategy>> Strategies;

  // Core DSP function optimizers
  Strategies.push_back(std::make_unique<BiquadCustomLICMOptimizer>());
  Strategies.push_back(std::make_unique<FIRDCustomLICMOptimizer>());
  Strategies.push_back(std::make_unique<ConvCustomLICMOptimizer>());
  Strategies.push_back(std::make_unique<DspsFft2rFc32CustomLICMOptimizer>());

  // Window function optimizers
  Strategies.push_back(std::make_unique<DspsWindCustomLICMBlackmanOptimizer>());
  Strategies.push_back(
      std::make_unique<DspsWindCustomLICMBlackmanHarrisOptimizer>());
  Strategies.push_back(std::make_unique<DspsWindCustomLICMFlatTopOptimizer>());
  Strategies.push_back(std::make_unique<DspsWindCustomLICMHannOptimizer>());

  return Strategies;
}

// Extract strategy name get logic
const char *RISCVCustomLICMPass::getStrategyName(size_t Index) {
  static const char *StrategyNames[] = {
      "BiquadCustomLICMOptimizer",
      "FIRDCustomLICMOptimizer",
      "ConvCustomLICMOptimizer",
      "DspsFft2rFc32CustomLICMOptimizer",
      "DspsWindCustomLICMBlackmanOptimizer",
      "DspsWindCustomLICMBlackmanHarrisOptimizer",
      "DspsWindCustomLICMFlatTopOptimizer",
      "DspsWindCustomLICMHannOptimizer"};

  return (Index < sizeof(StrategyNames) / sizeof(StrategyNames[0]))
             ? StrategyNames[Index]
             : "UnknownStrategy";
}

// Extract statistics update logic
void RISCVCustomLICMPass::updateStatistics(size_t StrategyIndex) {
  switch (StrategyIndex) {
  case 0:
    ++NumBiquadFunctionsOptimized;
    break;
  case 1:
    ++NumFIRDFunctionsOptimized;
    break;
  case 2:
    ++NumConvFunctionsOptimized;
    break;
  case 3:
    ++NumFFTFunctionsOptimized;
    break;
  default:
    ++NumWindowFunctionsOptimized;
    break;
  }
}

// Extract strategy apply logic
bool RISCVCustomLICMPass::applyOptimizationStrategies(
    const std::vector<std::unique_ptr<RISCVCustomLICMOptimizationStrategy>>
        &Strategies,
    Function &F, DominatorTree &DT, LoopInfo &LI) {

  for (size_t i = 0; i < Strategies.size(); ++i) {
    const auto &Strategy = Strategies[i];

    if (Strategy->isApplicable(F, LI)) {
      const char *StrategyName = getStrategyName(i);
      LLVM_DEBUG(dbgs() << "Applying strategy: " << StrategyName << "\n");

      bool Changed = Strategy->optimize(F, DT, LI);
      if (Changed) {
        LLVM_DEBUG(dbgs() << "Strategy " << StrategyName << " succeeded\n");
        updateStatistics(i);
        return true; // Only apply the first matching strategy
      } else {
        LLVM_DEBUG(dbgs() << "Strategy " << StrategyName
                          << " made no changes\n");
      }
    }
  }

  return false;
}

// Simplified main entry function
PreservedAnalyses RISCVCustomLICMPass::run(Function &F,
                                           FunctionAnalysisManager &FAM) {
  if (!EnableRISCVCustomLICM) {
    LLVM_DEBUG(dbgs() << "RISCVCustomLICM disabled, skipping function: "
                      << F.getName() << "\n");
    return PreservedAnalyses::all();
  }

  LLVM_DEBUG(dbgs() << "=== RISCVCustomLICM analyzing function: " << F.getName()
                    << " ===\n");
  LLVM_DEBUG(dbgs() << "Function has " << F.size() << " basic blocks and "
                    << F.arg_size() << " parameters\n");

  auto &DT = FAM.getResult<DominatorTreeAnalysis>(F);
  auto &LI = FAM.getResult<LoopAnalysis>(F);

  auto Strategies = createOptimizationStrategies();
  bool Changed = applyOptimizationStrategies(Strategies, F, DT, LI);

  LLVM_DEBUG(dbgs() << "=== RISCVCustomLICM "
                    << (Changed ? "modified" : "unchanged")
                    << " function: " << F.getName() << " ===\n");

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
