//===-- RISCVLoopUnrollAndRemainder.cpp - Loop Unrolling Pass
//-------------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a specialized loop unrolling optimization pass for
// RISC-V architecture. The pass targets Digital Signal Processing (DSP)
// algorithms and includes optimizations for:
//
// Supported DSP Algorithm Types:
// - FIR and IIR filters (DspsF32FirLoopUnroller, DspsF32FirdLoopUnroller)
// - Convolution and correlation (DspsF32ConvCcorrUnroller, DspiF32ConvUnroller)
// - Matrix operations (DspmF32MultLoopUnroller, DspmF32MultExLoopUnroller,
//                     DspmF32AddLoopUnroller)
// - Dot product calculations (DspsF32DotprodLoopUnroller and variants)
// - FFT transforms (DspsF32Fft2rUnroller)
// - Mathematical functions (DspsF32MathLoopUnroller)
// - Window functions (DspsF32WindBlackmanLoopUnroller)
//
// Key Optimization Features:
// 1. Loop Unrolling
//    - Supports various unroll factors (4, 8, 16, etc.)
//    - Optimizes unrolling strategy based on data set sizes
//    - Handles remainder iterations efficiently
//
// 2. Memory Access Optimization
//    - Optimizes array access patterns
//    - Improves cache utilization
//    - Enables vectorization opportunities
//
// 3. Control Flow Optimization
//    - Adjusts basic block layout
//    - Optimizes PHI nodes
//    - Handles nested loop structures
//
// 4. Post-processing Optimization
//    - Algorithm-specific optimizations
//    - Strength reduction
//    - Common subexpression elimination
//
// Configuration Options:
// - EnableRISCVLoopUnrollAndRemainder: Enable/disable the pass
// - DSPIMatrixSize: Configure matrix sizes (4x4 to 64x64)
// - DSPSFft2rFc32N: Configure FFT sizes (64 to 1024 points)
//
// The pass uses a factory pattern (LoopUnrollerFactory) to create specific
// Unroller instances. Each DSP algorithm type has its corresponding Unroller
// class that inherits from the base LoopUnroller class.
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
#include "llvm/Analysis/DomTreeUpdater.h"
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
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

using namespace llvm;

#define DEBUG_TYPE "riscv-loop-unroll-and-remainder"

// Command line option to enable the RISCVLoopUnrollAndRemainder pass
cl::opt<bool> llvm::EnableRISCVLoopUnrollAndRemainder(
    "riscv-loop-unroll-and-remainder", cl::init(false),
    cl::desc("Enable loop unrolling and remainder specific loop"));

// Define enum type to represent different DSPI matrix sizes
enum class DSPIMatrixSize {
  SIZE_4X4,
  SIZE_8X8,
  SIZE_16X16,
  SIZE_32X32,
  SIZE_64X64
};

// Command line option to control DSPI matrix size
static cl::opt<DSPIMatrixSize>
    DSPIMatrixSizeOpt("dspi-matrix-size",
                      cl::desc("Select DSPI matrix size for optimization:"),
                      cl::values(clEnumValN(DSPIMatrixSize::SIZE_4X4, "4x4",
                                            "Optimize 4x4 matrix"),
                                 clEnumValN(DSPIMatrixSize::SIZE_8X8, "8x8",
                                            "Optimize 8x8 matrix"),
                                 clEnumValN(DSPIMatrixSize::SIZE_16X16, "16x16",
                                            "Optimize 16x16 matrix"),
                                 clEnumValN(DSPIMatrixSize::SIZE_32X32, "32x32",
                                            "Optimize 32x32 matrix"),
                                 clEnumValN(DSPIMatrixSize::SIZE_64X64, "64x64",
                                            "Optimize 64x64 matrix")),
                      cl::init(DSPIMatrixSize::SIZE_8X8));

enum class DSPSFft2rFc32N { N_64, N_128, N_256, N_512, N_1024 };

// Command line option to control DSPS FFT2R FC32 matrix size
static cl::opt<DSPSFft2rFc32N> DSPSFft2rFc32NOpt(
    "dsps-fft2r-fc32-n",
    cl::desc("Select DSPS FFT2R FC32 matrix size for optimization:"),
    cl::values(clEnumValN(DSPSFft2rFc32N::N_64, "64", "Optimize 64N"),
               clEnumValN(DSPSFft2rFc32N::N_128, "128", "Optimize 128N"),
               clEnumValN(DSPSFft2rFc32N::N_256, "256", "Optimize 256N"),
               clEnumValN(DSPSFft2rFc32N::N_512, "512", "Optimize 512N"),
               clEnumValN(DSPSFft2rFc32N::N_1024, "1024", "Optimize 1024N")),
    cl::init(DSPSFft2rFc32N::N_64));

// Helper Function to simplify loop and form LCSSA
void LoopUnroller::simplifyAndFormLCSSA(Loop *L, DominatorTree &DT,
                                        LoopInfo *LI, ScalarEvolution &SE,
                                        AssumptionCache &AC) {
  simplifyLoop(L, &DT, LI, &SE, &AC, nullptr, false);
  formLCSSARecursively(*L, DT, LI, &SE);
}

LoopUnrollResult DspsF32BiquadLoopUnroller::unroll(Loop &L) {
  LoopUnrollResult Result =
      UnrollLoop(&L,
                 {/*Count*/ UnrollCount, /*Force*/ true, /*Runtime*/ false,
                  /*AllowExpensiveTripCount*/ true,
                  /*UnrollRemainder*/ true, true},
                 &LI, &SE, &DT, &AC, &TTI, /*ORE*/ &ORE, true);
  return Result;
}

LoopUnrollResult DspsF32DotprodSimpleLoopUnroller::unroll(Loop &L) {
  LoopUnrollResult Result =
      UnrollLoop(&L,
                 {/*Count*/ UnrollCount, /*Force*/ true, /*Runtime*/ false,
                  /*AllowExpensiveTripCount*/ true,
                  /*UnrollRemainder*/ true, true},
                 &LI, &SE, &DT, &AC, &TTI, /*ORE*/ &ORE, true);
  return Result;
}

LoopUnrollResult DspsF32DotprodLoopUnroller::unroll(Loop &L) {
  transformOneLoopDepth(F);
  return LoopUnrollResult::FullyUnrolled;
}

LoopUnrollResult DspsF32DotprodComplexLoopUnroller::unroll(Loop &L) {
  LoopUnrollResult Result =
      UnrollLoop(&L,
                 {/*Count*/ UnrollCount, /*Force*/ true, /*Runtime*/ false,
                  /*AllowExpensiveTripCount*/ true,
                  /*UnrollRemainder*/ true, true},
                 &LI, &SE, &DT, &AC, &TTI, /*ORE*/ &ORE, true);
  return Result;
}

// Factory Function to create DspiF32DotprodUnroller
std::unique_ptr<DspiF32DotprodUnroller>
createDspiF32DotprodUnroller(Function &F, LoopInfo &LI, ScalarEvolution &SE,
                             DominatorTree &DT, AssumptionCache &AC,
                             const TargetTransformInfo &TTI,
                             OptimizationRemarkEmitter &ORE, int UnrollCount) {
  if (DSPIMatrixSizeOpt == DSPIMatrixSize::SIZE_4X4 ||
      DSPIMatrixSizeOpt == DSPIMatrixSize::SIZE_8X8 ||
      DSPIMatrixSizeOpt == DSPIMatrixSize::SIZE_16X16) {
    return std::make_unique<DspiF32DotprodSmallUnroller>(F, LI, SE, DT, AC, TTI,
                                                         ORE, UnrollCount);
  } else if (DSPIMatrixSizeOpt == DSPIMatrixSize::SIZE_32X32 ||
             DSPIMatrixSizeOpt == DSPIMatrixSize::SIZE_64X64) {
    return std::make_unique<DspiF32DotprodLargeUnroller>(F, LI, SE, DT, AC, TTI,
                                                         ORE, UnrollCount);
  }
  llvm_unreachable("unsupported DSPIMatrixSize");
}

// Factory Function to create DspiF32ConvUnroller
std::unique_ptr<DspiF32ConvUnroller> createDspiF32ConvUnroller(
    Function &F, LoopInfo &LI, DominatorTree &DT, ScalarEvolution &SE,
    AssumptionCache &AC, const TargetTransformInfo &TTI,
    OptimizationRemarkEmitter &ORE, unsigned UnrollCount) {
  if (DSPIMatrixSizeOpt == DSPIMatrixSize::SIZE_4X4) {
    // 4x4 matrix uses factor 4, no post-processing
    return std::make_unique<DspiF32ConvSmallUnroller>(F, LI, DT, SE, AC, TTI,
                                                      ORE, 8);
  } else if (DSPIMatrixSizeOpt == DSPIMatrixSize::SIZE_8X8) {
    // 8x8 matrix uses factor 8, requires post-processing
    return std::make_unique<DspiF32ConvLargeUnroller>(F, LI, DT, SE, AC, TTI,
                                                      ORE, 8);
  } else if (DSPIMatrixSizeOpt == DSPIMatrixSize::SIZE_16X16) {
    // 16x16 matrix uses factor 16, requires post-processing
    return std::make_unique<DspiF32ConvLargeUnroller>(F, LI, DT, SE, AC, TTI,
                                                      ORE, 16);
  } else if (DSPIMatrixSizeOpt == DSPIMatrixSize::SIZE_32X32 ||
             DSPIMatrixSizeOpt == DSPIMatrixSize::SIZE_64X64) {
    // 32x32 matrix uses factor 32, requires post-processing
    return std::make_unique<DspiF32ConvLargeUnroller>(F, LI, DT, SE, AC, TTI,
                                                      ORE, 16);
  }
  llvm_unreachable("unsupported DSPIMatrixSize");
}

// Factory Function to create DspsF32Fft2rUnroller
std::unique_ptr<DspsF32Fft2rUnroller> createDspsF32Fft2rUnroller(
    Function &F, LoopInfo &LI, DominatorTree &DT, ScalarEvolution &SE,
    AssumptionCache &AC, const TargetTransformInfo &TTI,
    OptimizationRemarkEmitter &ORE, unsigned UnrollCount) {
  switch (DSPSFft2rFc32NOpt) {
  case DSPSFft2rFc32N::N_256:
  case DSPSFft2rFc32N::N_512:
  case DSPSFft2rFc32N::N_1024:
    // Large FFT uses factor 4, requires post-processing
    return std::make_unique<DspsF32Fft2rLargeUnroller>(F, LI, DT, SE, AC, TTI,
                                                       ORE, 4);
  case DSPSFft2rFc32N::N_64:
  case DSPSFft2rFc32N::N_128:
    // Small FFT uses factor 4, no post-processing
    return std::make_unique<DspsF32Fft2rSmallUnroller>(F, LI, DT, SE, AC, TTI,
                                                       ORE, 4);
  default:
    llvm_unreachable("unsupported FFT size");
  }
}

// Factory class to create specific unrollers
class LoopUnrollerFactory {
public:
  static std::unique_ptr<LoopUnroller>
  createLoopUnroller(Function &F, LoopInfo &LI, DominatorTree &DT,
                     ScalarEvolution &SE, AssumptionCache &AC,
                     const TargetTransformInfo &TTI,
                     OptimizationRemarkEmitter &ORE) {
    std::unique_ptr<LoopUnroller> Unroller;
    Unroller = std::make_unique<DspsF32CorrLoopUnroller>(F, LI, DT, SE, AC, TTI,
                                                         ORE, 16);
    if (Unroller->checkType()) {
      return Unroller;
    }

    Unroller = std::make_unique<DspsF32BiquadLoopUnroller>(F, LI, DT, SE, AC,
                                                           TTI, ORE, 8);
    if (Unroller->checkType()) {
      return Unroller;
    }
    Unroller = createDspiF32ConvUnroller(F, LI, DT, SE, AC, TTI, ORE, 8);
    if (Unroller->checkType()) {
      return Unroller;
    }
    Unroller = createDspiF32DotprodUnroller(F, LI, SE, DT, AC, TTI, ORE, 8);
    if (Unroller->checkType()) {
      return Unroller;
    }
    Unroller = std::make_unique<DspsF32WindBlackmanLoopUnroller>(
        F, LI, DT, SE, AC, TTI, ORE, 8);
    if (Unroller->checkType()) {
      return Unroller;
    }
    Unroller = std::make_unique<DspmF32MultLoopUnroller>(F, LI, DT, SE, AC, TTI,
                                                         ORE, 8);
    if (Unroller->checkType()) {
      return Unroller;
    }
    Unroller = std::make_unique<DspmF32MultExLoopUnroller>(F, LI, DT, SE, AC,
                                                           TTI, ORE, 8);
    if (Unroller->checkType()) {
      return Unroller;
    }
    Unroller = std::make_unique<DspsF32ConvCcorrUnroller>(F, LI, DT, SE, AC,
                                                          TTI, ORE, 8);
    if (Unroller->checkType()) {
      return Unroller;
    }
    Unroller = std::make_unique<DspmF32AddLoopUnroller>(F, LI, DT, SE, AC, TTI,
                                                        ORE, 8);
    if (Unroller->checkType()) {
      return Unroller;
    }
    Unroller = std::make_unique<DspsF32DotprodLoopUnroller>(F, LI, DT, SE, AC,
                                                            TTI, ORE, 8);
    if (Unroller->checkType()) {
      return Unroller;
    }
    Unroller = std::make_unique<DspsF32DotprodSimpleLoopUnroller>(
        F, LI, DT, SE, AC, TTI, ORE, 8);
    if (Unroller->checkType()) {
      return Unroller;
    }
    Unroller = std::make_unique<DspsF32DotprodComplexLoopUnroller>(
        F, LI, DT, SE, AC, TTI, ORE, 8);
    if (Unroller->checkType()) {
      return Unroller;
    }
    Unroller = std::make_unique<DspsF32MathLoopUnroller>(F, LI, DT, SE, AC, TTI,
                                                         ORE, 16);
    if (Unroller->checkType()) {
      return Unroller;
    }

    Unroller = std::make_unique<DspsF32FirLoopUnroller>(F, LI, DT, SE, AC, TTI,
                                                        ORE, 16);
    if (Unroller->checkType()) {
      return Unroller;
    }
    Unroller = std::make_unique<DspsF32FirdLoopUnroller>(F, LI, DT, SE, AC, TTI,
                                                         ORE, 8);

    if (Unroller->checkType()) {
      return Unroller;
    }
    Unroller = createDspsF32Fft2rUnroller(F, LI, DT, SE, AC, TTI, ORE, 8);
    if (Unroller->checkType()) {
      return Unroller;
    }

    return nullptr;
  }
};

// Check loop depth and Sub-loops
static bool checkLoopStructure(Function &F, LoopInfo &LI,
                               unsigned ExpectedDepth = 1,
                               unsigned ExpectedOuterLoops = 2,
                               unsigned ExpectedInnerLoops = 0) {
  // Check maximum loop depth
  unsigned MaxDepth = 0;
  for (auto &BB : F) {
    MaxDepth = std::max(MaxDepth, LI.getLoopDepth(&BB));
  }
  if (MaxDepth != ExpectedDepth) {
    return false;
  }

  // Check outer and inner loop counts
  int OuterLoopCount = 0;
  int InnerLoopCount = 0;
  for (Loop *L : LI.getLoopsInPreorder()) {
    if (L->getLoopDepth() == 1) {
      OuterLoopCount++;
    } else if (L->getLoopDepth() == 2) {
      InnerLoopCount++;
    } else {
      return false;
    }
  }

  return OuterLoopCount == ExpectedOuterLoops &&
         InnerLoopCount == ExpectedInnerLoops;
}

// Check DSPS F32 CORR type Function
bool DspsF32CorrLoopUnroller::checkDspsF32CorrType(Function &F, LoopInfo &LI) {
  // Check basic block count
  if (F.size() != 7 || F.arg_size() != 5)
    return false;

  // Get critical basic blocks
  BasicBlock *Entry = &F.getEntryBlock();
  BasicBlock *ForCondPreheader = getBasicBlockByName(F, "for.cond.preheader");
  BasicBlock *Return = getBasicBlockByName(F, "return");

  // Check basic block existence
  if (!Entry || !ForCondPreheader || !Return)
    return false;

  // Check basic block jump relation
  if (!checkSuccessors(Entry, 2, Return, ForCondPreheader))
    return false;

  if (!checkLoopStructure(F, LI, 2, 1, 1))
    return false;
  // Check floating point multiply-add instruction usage
  if (!checkFMulAddUsage(F))
    return false;

  return true;
}

/// Check convolution-specific loop structure patterns.
/// Convolution algorithms typically have 2-3 main loops with specific
/// characteristics including Fmuladd operations and complex bounds.
bool DspsF32ConvCcorrUnroller::checkConvolutionLoopStructure(Function &F,
                                                             LoopInfo &LI) {
  auto Loops = LI.getLoopsInPreorder();

  // Convolution algorithms typically have 2-3 main loops
  if (Loops.size() < 2)
    return false;

  int NestedLoopsWithFMulAdd = 0;
  int LoopsWithComplexBounds = 0;

  for (auto *L : Loops) {
    // Check if loop contains Fmuladd operations
    if (containsFMulAdd(L)) {
      NestedLoopsWithFMulAdd++;
    }

    // Check loop boundary complexity (convolution characteristic)
    // Use simpler but effective method to detect complex loop bounds
    BasicBlock *Header = L->getHeader();
    if (Header && !Header->hasNPredecessors(1)) {
      LoopsWithComplexBounds++;
    }

    // Check exit condition complexity
    SmallVector<BasicBlock *, 4> ExitBlocks;
    L->getExitBlocks(ExitBlocks);
    if (ExitBlocks.size() > 1) {
      LoopsWithComplexBounds++;
    }

    // Check for complex control flow within the loop (multiple branches)
    for (auto *BB : L->getBlocks()) {
      if (auto *BI = dyn_cast<BranchInst>(BB->getTerminator())) {
        if (BI->isConditional()) {
          LoopsWithComplexBounds++;
          break;
        }
      }
    }
  }

  // At least 2 loops with multiply-add operations and complex bounds
  return NestedLoopsWithFMulAdd >= 2 && LoopsWithComplexBounds >= 1;
}

/// Check if a loop contains Fmuladd operations.
/// This is a key characteristic of convolution algorithms.
bool DspsF32ConvCcorrUnroller::containsFMulAdd(Loop *L) {
  for (auto *BB : L->getBlocks()) {
    for (auto &I : *BB) {
      if (auto *Call = dyn_cast<CallInst>(&I)) {
        if (Call->getIntrinsicID() == Intrinsic::fmuladd) {
          return true;
        }
      }
    }
  }
  return false;
}

// Analyze GEP instruction Index patterns.
/// Supports complex patterns from Clang 20 including XOR optimizations.
void DspsF32ConvCcorrUnroller::analyzeIndexPattern(GetElementPtrInst *GEP,
                                                   int &SubCount,
                                                   int &AddCount) {
  // Check GEP direct operands
  for (auto &Op : GEP->operands()) {
    if (auto *BinOp = dyn_cast<BinaryOperator>(Op)) {
      switch (BinOp->getOpcode()) {
      case Instruction::Sub:
        SubCount++;
        break;
      case Instruction::Add:
        AddCount++;
        break;
      case Instruction::Xor:
        // XOR with -1 is equivalent to NOT, often used for subtraction
        // optimization
        if (auto *CI = dyn_cast<ConstantInt>(BinOp->getOperand(1))) {
          if (CI->isMinusOne()) {
            SubCount++; // xor x, -1 equivalent to subtraction operation
          }
        }
        break;
      default:
        break;
      }
    }
  }

  // Check GEP Index value definition instructions
  for (unsigned I = 1; I < GEP->getNumOperands(); ++I) {
    Value *Index = GEP->getOperand(I);
    if (auto *BinOp = dyn_cast<BinaryOperator>(Index)) {
      switch (BinOp->getOpcode()) {
      case Instruction::Sub:
        SubCount++;
        break;
      case Instruction::Add:
        AddCount++;
        break;
      case Instruction::Xor:
        if (auto *CI = dyn_cast<ConstantInt>(BinOp->getOperand(1))) {
          if (CI->isMinusOne()) {
            SubCount++;
          }
        }
        break;
      default:
        break;
      }
    }
  }

  // Special handling for chained GEP: check if GEP base address is also a GEP
  if (auto *BaseGEP = dyn_cast<GetElementPtrInst>(GEP->getPointerOperand())) {
    // Recursively analyze base address GEP
    analyzeIndexPattern(BaseGEP, SubCount, AddCount);
  }
}

// Updated memory pattern check Function - deep data flow analysis
bool DspsF32ConvCcorrUnroller::checkConvolutionMemoryPattern(Function &F) {
  int FMulAddCount = 0;
  int GEPWithSub = 0;
  int GEPWithAdd = 0;
  int ComplexIndexPatterns = 0;
  int XorPatterns = 0;

  // Collect all arithmetic instructions
  std::set<Value *> AddInstructions;
  std::set<Value *> SubInstructions;
  std::set<Value *> XorInstructions;

  for (auto &BB : F) {
    for (auto &I : BB) {
      // Count Fmuladd instructions
      if (auto *Call = dyn_cast<CallInst>(&I)) {
        if (Call->getIntrinsicID() == Intrinsic::fmuladd) {
          FMulAddCount++;
        }
      }

      // Collect all arithmetic instructions
      if (auto *BinOp = dyn_cast<BinaryOperator>(&I)) {
        switch (BinOp->getOpcode()) {
        case Instruction::Add:
          AddInstructions.insert(BinOp);
          break;
        case Instruction::Sub:
          SubInstructions.insert(BinOp);
          break;
        case Instruction::Xor:
          if (auto *CI = dyn_cast<ConstantInt>(BinOp->getOperand(1))) {
            if (CI->isMinusOne()) {
              XorInstructions.insert(BinOp);
            }
          }
          break;
        default:
          break;
        }
      }
    }
  }

  // Deep analysis of GEP instruction Index sources
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (auto *GEP = dyn_cast<GetElementPtrInst>(&I)) {
        // Analyze each Index operand
        for (unsigned I = 1; I < GEP->getNumOperands(); ++I) {
          Value *Index = GEP->getOperand(I);

          // Directly check if Index is an arithmetic instruction
          if (AddInstructions.count(Index)) {
            GEPWithAdd++;
            ComplexIndexPatterns++;
          }
          if (SubInstructions.count(Index)) {
            GEPWithSub++;
            ComplexIndexPatterns++;
          }
          if (XorInstructions.count(Index)) {
            GEPWithSub++;
            ComplexIndexPatterns++;
            XorPatterns++;
          }

          // Check PHI nodes: PHI nodes may connect arithmetic instructions and
          // GEP
          if (auto *PHI = dyn_cast<PHINode>(Index)) {
            for (unsigned J = 0; J < PHI->getNumIncomingValues(); ++J) {
              Value *IncomingValue = PHI->getIncomingValue(J);

              if (AddInstructions.count(IncomingValue)) {
                GEPWithAdd++;
                ComplexIndexPatterns++;
              }
              if (SubInstructions.count(IncomingValue)) {
                GEPWithSub++;
                ComplexIndexPatterns++;
              }
              if (XorInstructions.count(IncomingValue)) {
                GEPWithSub++;
                ComplexIndexPatterns++;
                XorPatterns++;
              }
            }
          }

          // Recursive check: if Index itself is the result of another
          // instruction
          if (auto *DefInst = dyn_cast<Instruction>(Index)) {
            // Check operands of the defining instruction
            for (auto &Op : DefInst->operands()) {
              if (AddInstructions.count(Op)) {
                GEPWithAdd++;
                ComplexIndexPatterns++;
              }
              if (SubInstructions.count(Op)) {
                GEPWithSub++;
                ComplexIndexPatterns++;
              }
              if (XorInstructions.count(Op)) {
                GEPWithSub++;
                ComplexIndexPatterns++;
                XorPatterns++;
              }
            }
          }
        }

        // Analyze chained GEP
        if (auto *BaseGEP =
                dyn_cast<GetElementPtrInst>(GEP->getPointerOperand())) {
          ComplexIndexPatterns++;
        }
      }
    }
  }

  // Essential characteristics of convolution algorithms:
  bool HasSubPattern = GEPWithSub > 0 || XorPatterns > 0;
  bool HasAddPattern = GEPWithAdd > 0;

  return FMulAddCount >= 3 && HasSubPattern && HasAddPattern &&
         ComplexIndexPatterns >= 2;
}

/// Check DSPS F32 CONV/CCORR type Function.
/// This Function validates the characteristics of convolution/cross-correlation
/// algorithms including parameter count, computation patterns, loop structure,
/// memory access patterns, and defensive programming practices.
bool DspsF32ConvCcorrUnroller::checkDspsF32ConvCcorr(Function &F,
                                                     LoopInfo &LI) {
  // 1. Parameter characteristics: 5 parameters (signal, signal_length, kernel,
  // kernel_length, output)
  if (F.arg_size() != 5)
    return false;

  // 2. Core computation pattern: must have Fmuladd instructions
  if (!checkFMulAddUsage(F))
    return false;

  // 3. Loop structure check (at least two main loops)
  if (!checkConvolutionLoopStructure(F, LI))
    return false;

  // 4. Memory access pattern: different indexing patterns for two input arrays
  if (!checkConvolutionMemoryPattern(F))
    return false;

  // 5. Null pointer validation pattern (defensive programming characteristic)
  if (!checkNullPointerValidation(F))
    return false;

  return true;
}

/// Check null pointer validation patterns.
/// This is a defensive programming characteristic commonly found in DSP
/// functions.
bool DspsF32ConvCcorrUnroller::checkNullPointerValidation(Function &F) {
  // Find null pointer checks in Entry basic block
  BasicBlock *Entry = &F.getEntryBlock();

  int NullChecks = 0;
  int EarlyReturns = 0;

  for (auto &I : *Entry) {
    // Check null pointer comparisons
    if (auto *ICmp = dyn_cast<ICmpInst>(&I)) {
      if (ICmp->getPredicate() == ICmpInst::ICMP_EQ) {
        // Check if comparing with null
        for (auto &Op : ICmp->operands()) {
          if (isa<ConstantPointerNull>(Op)) {
            NullChecks++;
            break;
          }
        }
      }
    }

    // Check early returns
    if (auto *Br = dyn_cast<BranchInst>(&I)) {
      if (Br->isConditional()) {
        // Check if there are direct jumps to return block
        for (unsigned I = 0; I < Br->getNumSuccessors(); ++I) {
          BasicBlock *Successor = Br->getSuccessor(I);
          if (Successor->getTerminator() &&
              isa<ReturnInst>(Successor->getTerminator())) {
            EarlyReturns++;
            break;
          }
        }
      }
    }
  }

  // DSP functions typically have multiple null pointer checks and early returns
  return NullChecks >= 2 && EarlyReturns >= 1;
}

// Check DSPS F32 FIRD type Function
bool DspsF32FirdLoopUnroller::checkDspsF32FirdType(Function &F, LoopInfo &LI) {
  // Check basic block and parameter count
  if (F.size() != 14 || F.arg_size() != 4)
    return false;

  // Check floating point multiply-add instruction usage
  if (!checkFMulAddUsage(F))
    return false;

  // Get critical basic blocks
  BasicBlock *Entry = &F.getEntryBlock();
  BasicBlock *ForCondCleanup = getBasicBlockByName(F, "for.cond.cleanup");

  if (!Entry || !ForCondCleanup)
    return false;

  if (Entry->getTerminator()->getSuccessor(1) != ForCondCleanup)
    return false;

  if (!checkLoopStructure(F, LI, 2, 1, 3))
    return false;
  return true;
}

// Check DSPI CONV F32 type Function
bool DspiF32ConvUnroller::checkDspiF32ConvType(Function &F, LoopInfo &LI) {
  // Check parameter count
  if (F.arg_size() != 3)
    return false;

  // Check parameter type
  for (const Argument &Arg : F.args()) {
    if (!Arg.getType()->isPointerTy()) {
      return false;
    }
  }

  // Check loop structure
  int OuterLoopCount = 0;
  int InnerLoopCount = 0;
  for (Loop *L : LI) {
    OuterLoopCount++;
    for (Loop *SubL : L->getSubLoops()) {
      for (Loop *SubsubL : SubL->getSubLoops()) {
        for (Loop *subsubsubL : SubsubL->getSubLoops()) {
          InnerLoopCount++;
        }
      }
    }
  }

  return OuterLoopCount == 4 && InnerLoopCount == 9;
}

// Check DSPS F32 FIR type Function
bool DspsF32FirLoopUnroller::checkDspsF32FirType(Function &F, LoopInfo &LI) {
  // Check basic block and parameter count
  if (F.size() != 19 || F.arg_size() != 4)
    return false;

  // Check floating point multiply-add instruction usage
  if (!checkFMulAddUsage(F))
    return false;

  // Get critical basic blocks
  BasicBlock *Entry = &F.getEntryBlock();
  BasicBlock *ForCondPreheader = getBasicBlockByName(F, "for.cond.preheader");
  BasicBlock *ForBodyLrPh = getBasicBlockByName(F, "for.body.lr.ph");
  BasicBlock *IfEnd = getBasicBlockByName(F, "if.end");
  BasicBlock *ForBody = getBasicBlockByName(F, "for.body");
  BasicBlock *ForBodyClone = getBasicBlockByName(F, "for.body.clone");
  BasicBlock *ForBodyLrPhClone = getBasicBlockByName(F, "for.body.lr.ph.clone");

  // Check basic block existence and jump relation
  if (!Entry || !ForCondPreheader || !ForBodyLrPh || !IfEnd || !ForBody ||
      !ForBodyClone || !ForBodyLrPhClone)
    return false;

  if (!checkSuccessors(Entry, 2, ForCondPreheader, ForBodyLrPhClone))
    return false;
  if (!checkSuccessors(ForCondPreheader, 2, ForBodyLrPh, IfEnd))
    return false;
  if (!checkSuccessors(ForBodyLrPh, 1, ForBody))
    return false;
  if (!checkSuccessors(ForBodyLrPhClone, 1, ForBodyClone))
    return false;

  // Check loop structure
  if (!checkLoopStructure(F, LI, 2, 2, 2) &&
      !checkLoopStructure(F, LI, 2, 2, 4))
    return false;
  return true;
}

// Check DSPS Wind Blackman F32 type Function
bool DspsF32WindBlackmanLoopUnroller::checkDspsWindBlackmanF32Type(
    Function &F) {
  // Check basic block and parameter count
  if (F.size() != 4 || F.arg_size() != 2)
    return false;

  // Get critical basic blocks
  BasicBlock *Entry = &F.getEntryBlock();
  BasicBlock *ForBodyLrPh = getBasicBlockByName(F, "for.body.lr.ph");
  BasicBlock *ForBody = getBasicBlockByName(F, "for.body");
  BasicBlock *ForCondCleanup = getBasicBlockByName(F, "for.cond.cleanup");

  // Check basic block existence and jump relation
  if (!Entry || !ForBodyLrPh || !ForBody || !ForCondCleanup)
    return false;

  if (!checkSuccessors(Entry, 2, ForBodyLrPh, ForCondCleanup))
    return false;
  if (ForBodyLrPh->getSingleSuccessor() != ForBody)
    return false;
  if (succ_size(ForCondCleanup) != 0)
    return false;
  if (!checkSuccessors(ForBody, 2, ForCondCleanup, ForBody))
    return false;

  return true;
}

// Check DSPM F32 Add type function，add/Sub/addc/mulc
bool DspmF32AddLoopUnroller::checkDspmF32AddType(Function &F, LoopInfo &LI) {
  // 1. Basic constraint check - support different versions with varying basic
  // block counts
  if (F.size() < 4 || F.size() > 7)
    return false;

  // 2. Parameter count check - support different Function modes
  // 11 parameters: dspm_add_f32_ansi, dspm_sub_f32_ansi (two input arrays)
  // 9 parameters: dspm_addc_f32_ansi, dspm_mulc_f32_ansi (array and constant
  // operations)
  if (!(F.arg_size() == 11 || F.arg_size() == 9))
    return false;

  // 3. Parameter type pattern validation - determine Function type based on
  // parameter count
  auto ArgIt = F.arg_begin();

  if (F.arg_size() == 11) {
    // Two array mode: input1, input2, output, rows, cols, padd1, padd2,
    // padd_out, step1, step2, step_out
    if (!ArgIt->getType()->isPointerTy())
      return false; // input1 ptr
    ++ArgIt;
    if (!ArgIt->getType()->isPointerTy())
      return false; // input2 ptr
    ++ArgIt;
    if (!ArgIt->getType()->isPointerTy())
      return false; // output ptr
    ++ArgIt;
    // The remaining 8 parameters should be i32
    for (int I = 0; I < 8; ++I) {
      if (!ArgIt->getType()->isIntegerTy(32))
        return false;
      ++ArgIt;
    }
  } else if (F.arg_size() == 9) {
    // Array and constant mode: input, output, C, rows, cols, padd_in, padd_out,
    // step_in, step_out
    if (!ArgIt->getType()->isPointerTy())
      return false; // input ptr
    ++ArgIt;
    if (!ArgIt->getType()->isPointerTy())
      return false; // output ptr
    ++ArgIt;
    if (!ArgIt->getType()->isFloatTy())
      return false; // C (float constant)
    ++ArgIt;
    // The remaining 6 parameters should be i32
    for (int I = 0; I < 6; ++I) {
      if (!ArgIt->getType()->isIntegerTy(32))
        return false;
      ++ArgIt;
    }
  }

  // 4. Loop structure analysis - this is the essence of control flow
  if (LI.empty())
    return false;

  // There should be nested loop structure: outer loop (row) + inner loop
  // (column)
  Loop *OuterLoop = nullptr;
  int TopLevelLoopCount = 0;

  for (Loop *L : LI) {
    if (L->getLoopDepth() == 1) {
      OuterLoop = L;
      TopLevelLoopCount++;
    }
  }

  // There should be only one top level loop
  if (TopLevelLoopCount != 1 || !OuterLoop)
    return false;

  // The outer loop should contain one inner loop
  if (OuterLoop->getSubLoops().size() != 1)
    return false;

  Loop *InnerLoop = OuterLoop->getSubLoops()[0];
  if (!InnerLoop || InnerLoop->getLoopDepth() != 2)
    return false;

  // The inner loop should not have any Sub loops
  if (!InnerLoop->getSubLoops().empty())
    return false;

  // 5. Key operation validation - check floating point operations
  bool HasFloatBinaryOp = false;
  bool HasLoad = false;
  bool HasStore = false;
  int LoadCount = 0;

  // Supported floating point operations
  bool HasFAdd = false, HasFSub = false, HasFMul = false;

  for (BasicBlock &BB : F) {
    for (Instruction &Inst : BB) {
      // Check floating point operations - core operations of DSPM Function
      if (auto *BinOp = dyn_cast<BinaryOperator>(&Inst)) {
        switch (BinOp->getOpcode()) {
        case Instruction::FAdd:
          HasFAdd = true;
          HasFloatBinaryOp = true;
          break;
        case Instruction::FSub:
          HasFSub = true;
          HasFloatBinaryOp = true;
          break;
        case Instruction::FMul:
          HasFMul = true;
          HasFloatBinaryOp = true;
          break;
        default:
          break;
        }
      }
      // Check memory access pattern
      else if (isa<LoadInst>(&Inst)) {
        HasLoad = true;
        LoadCount++;
      } else if (isa<StoreInst>(&Inst)) {
        HasStore = true;
      }
    }
  }

  // There must be floating point operations, memory loads and stores
  if (!HasFloatBinaryOp || !HasLoad || !HasStore)
    return false;

  // 6. Validate Function pattern based on parameter count and operation type
  if (F.arg_size() == 11) {
    // Two array mode requires at least 2 load instructions, and can only be
    // addition or subtraction
    if (LoadCount < 2 || (!HasFAdd && !HasFSub))
      return false;
  } else if (F.arg_size() == 9) {
    // Array and constant mode requires at least 1 load instruction, and
    // supports addition or multiplication
    if (LoadCount < 1 || (!HasFAdd && !HasFMul))
      return false;
  }

  // 7. Control flow topology validation - based on structure rather than name
  BasicBlock *Entry = &F.getEntryBlock();

  // Entry should have a conditional branch - one to return, one to loop
  if (!isa<BranchInst>(Entry->getTerminator()) ||
      Entry->getTerminator()->getNumSuccessors() != 2)
    return false;

  // Find return basic block
  BasicBlock *ReturnBB = nullptr;
  for (BasicBlock &BB : F) {
    if (isa<ReturnInst>(BB.getTerminator())) {
      ReturnBB = &BB;
      break;
    }
  }

  if (!ReturnBB)
    return false;

  // One successor of Entry should be ReturnBB (error handling path)
  bool HasReturnSuccessor = false;
  for (unsigned I = 0; I < Entry->getTerminator()->getNumSuccessors(); ++I) {
    if (Entry->getTerminator()->getSuccessor(I) == ReturnBB) {
      HasReturnSuccessor = true;
      break;
    }
  }

  if (!HasReturnSuccessor)
    return false;

  // 8. Loop structure topology validation
  BasicBlock *OuterHeader = OuterLoop->getHeader();
  BasicBlock *InnerHeader = InnerLoop->getHeader();

  if (!OuterHeader || !InnerHeader)
    return false;

  // Validate basic loop topology
  // The inner loop should have a self-loop edge (this is the essential
  // characteristic of loop)
  bool InnerHasSelfLoop = false;
  if (auto *BI = dyn_cast<BranchInst>(InnerHeader->getTerminator())) {
    for (unsigned I = 0; I < BI->getNumSuccessors(); ++I) {
      if (BI->getSuccessor(I) == InnerHeader) {
        InnerHasSelfLoop = true;
        break;
      }
    }
  }

  if (!InnerHasSelfLoop)
    return false;

  // The outer loop should be able to return to its head (outer loop
  // characteristic)
  BasicBlock *OuterLatch = OuterLoop->getLoopLatch();
  if (!OuterLatch)
    return false;

  bool OuterHasBackEdge = false;
  if (auto *BI = dyn_cast<BranchInst>(OuterLatch->getTerminator())) {
    for (unsigned I = 0; I < BI->getNumSuccessors(); ++I) {
      if (BI->getSuccessor(I) == OuterHeader) {
        OuterHasBackEdge = true;
        break;
      }
    }
  }

  if (!OuterHasBackEdge)
    return false;

  // 9. Validate loop exit structure
  // The outer loop should have a path to return
  SmallVector<BasicBlock *, 4> OuterExitBlocks;
  OuterLoop->getExitBlocks(OuterExitBlocks);

  bool HasExitToReturn = false;
  for (BasicBlock *ExitBB : OuterExitBlocks) {
    if (ExitBB == ReturnBB) {
      HasExitToReturn = true;
      break;
    }
    // Or through intermediate basic blocks to return
    if (auto *BI = dyn_cast<BranchInst>(ExitBB->getTerminator())) {
      for (unsigned I = 0; I < BI->getNumSuccessors(); ++I) {
        if (BI->getSuccessor(I) == ReturnBB) {
          HasExitToReturn = true;
          break;
        }
      }
    }
    if (HasExitToReturn)
      break;
  }

  if (!HasExitToReturn)
    return false;

  return true;
}

// Check DSPM F32 Mult type Function
bool DspmF32MultLoopUnroller::checkDspmF32MultType(Function &F, LoopInfo &LI) {
  for (Loop *L : LI) {
    // Check first layer loop
    BasicBlock *ForCond1PreheaderLrPh = L->getLoopPreheader();
    BasicBlock *ForCond1Preheader = L->getHeader();
    BasicBlock *ForCondCleanup = L->getExitBlock();
    BasicBlock *ForCondCleanup3 = L->getExitingBlock();

    // Check basic block existence and jump relation
    if (!ForCond1PreheaderLrPh || !ForCond1Preheader || !ForCondCleanup ||
        !ForCondCleanup3)
      return false;
    if (ForCondCleanup3 != L->getLoopLatch())
      return false;
    if (L->getSubLoops().size() != 1)
      return false;

    // Check second layer loop
    Loop *SubL = L->getSubLoops().front();
    if (!SubL)
      return false;

    BasicBlock *ForBody4LrPh = SubL->getLoopPreheader();
    BasicBlock *ForBody4 = SubL->getHeader();
    BasicBlock *ForCondCleanup8 = SubL->getExitingBlock();

    if (!ForBody4LrPh || !ForBody4)
      return false;
    if (SubL->getExitBlock() != ForCondCleanup3)
      return false;
    if (SubL->getLoopLatch() != ForCondCleanup8)
      return false;
    if (SubL->getSubLoops().size() != 1)
      return false;

    // Check third layer loop
    Loop *SubsubL = SubL->getSubLoops().front();
    if (!SubsubL || !SubsubL->getSubLoops().empty())
      return false;

    if (ForBody4 != SubsubL->getLoopPredecessor())
      return false;

    BasicBlock *ForBody9 = SubsubL->getHeader();
    if (!ForBody9)
      return false;

    if (ForBody9 != SubsubL->getLoopLatch() ||
        ForBody9 != SubsubL->getExitingBlock())
      return false;

    if (ForCondCleanup8 != SubsubL->getExitBlock())
      return false;

    // Check PHI node
    PHINode *FirstF32Phi = getFirstFloatPhi(ForBody9);
    if (!FirstF32Phi)
      return false;

    // Check compare instruction
    for (auto &I : *ForCond1PreheaderLrPh) {
      if (auto *Icmp = dyn_cast<ICmpInst>(&I)) {
        if (Icmp->getPredicate() == ICmpInst::ICMP_SGT) {
          return true;
        }
      }
    }
    return false;
  }
  return false;
}

// Check DSPM F32 Mult Ex type Function
bool DspmF32MultExLoopUnroller::checkDspmF32MultExType(Function &F,
                                                       LoopInfo &LI) {
  for (Loop *L : LI) {
    // Check first layer loop
    BasicBlock *ForCond1PreheaderLrPh = L->getLoopPreheader();
    BasicBlock *ForCond1Preheader = L->getHeader();
    BasicBlock *ForCondCleanup = L->getExitBlock();
    BasicBlock *ForCondCleanup3 = L->getExitingBlock();

    // Check basic block existence and jump relation
    if (!ForCond1PreheaderLrPh || !ForCond1Preheader || !ForCondCleanup ||
        !ForCondCleanup3)
      return false;
    if (ForCondCleanup3 != L->getLoopLatch())
      return false;
    if (L->getSubLoops().size() != 1)
      return false;

    // Check second layer loop
    Loop *SubL = L->getSubLoops().front();
    if (!SubL)
      return false;

    BasicBlock *ForBody4LrPh = SubL->getLoopPreheader();
    BasicBlock *ForBody4 = SubL->getHeader();
    BasicBlock *ForCondCleanup8 = SubL->getExitingBlock();

    if (!ForBody4LrPh || !ForBody4)
      return false;
    if (SubL->getExitBlock() != ForCondCleanup3)
      return false;
    if (SubL->getLoopLatch() != ForCondCleanup8)
      return false;
    if (SubL->getSubLoops().size() != 1)
      return false;

    // Check third layer loop
    Loop *SubsubL = SubL->getSubLoops().front();
    if (!SubsubL || !SubsubL->getSubLoops().empty())
      return false;

    if (ForBody4 != SubsubL->getLoopPredecessor())
      return false;

    BasicBlock *ForBody9 = SubsubL->getHeader();
    if (!ForBody9)
      return false;

    if (ForBody9 != SubsubL->getLoopLatch() ||
        ForBody9 != SubsubL->getExitingBlock())
      return false;

    if (ForCondCleanup8 != SubsubL->getExitBlock())
      return false;

    // Check PHI node
    PHINode *FirstF32Phi = getFirstFloatPhi(ForBody9);
    if (!FirstF32Phi)
      return false;

    // Check compare instruction
    for (auto &I : *ForCond1PreheaderLrPh) {
      if (auto *Icmp = dyn_cast<ICmpInst>(&I)) {
        if (Icmp->getPredicate() == ICmpInst::ICMP_UGT) {
          return true;
        }
      }
    }
    return false;
  }
  return false;
}

// Check DSPI F32 Dotprod type Function
bool DspiF32DotprodUnroller::checkDspiF32DotprodType(Function &F,
                                                     LoopInfo &LI) {
  for (Loop *L : LI) {
    // Get critical basic blocks
    BasicBlock *ForCond25Preheader = L->getHeader();
    BasicBlock *ForCond25PreheaderLrPh = L->getLoopPreheader();

    if (!ForCond25Preheader || !ForCond25PreheaderLrPh)
      return false;
    if (ForCond25PreheaderLrPh->getSingleSuccessor() != ForCond25Preheader)
      return false;

    // Check Sub loop
    for (Loop *SubL : L->getSubLoops()) {
      BasicBlock *ForBody28 = SubL->getHeader();
      if (!ForBody28)
        return false;
      if (SubL->getLoopPredecessor() != ForCond25Preheader)
        return false;

      // Check jump relation
      if (ForCond25Preheader->getTerminator()->getNumSuccessors() != 2 ||
          ForCond25Preheader->getTerminator()->getSuccessor(0) != ForBody28)
        return false;
      if (ForBody28->getTerminator()->getNumSuccessors() != 2 ||
          ForBody28->getTerminator()->getSuccessor(1) != ForBody28)
        return false;
      if (ForBody28->getTerminator()->getSuccessor(0) !=
          ForCond25Preheader->getTerminator()->getSuccessor(1))
        return false;

      // Check PHI node and floating point multiply-add instruction
      PHINode *FirstF32Phi = getFirstFloatPhi(ForCond25Preheader);
      if (!FirstF32Phi)
        return false;
      Instruction *FirstFMulAddInst = getFirstFMulAddInst(ForBody28);
      if (!FirstFMulAddInst)
        return false;

      return true;
    }
  }
  return false;
}

// Check DSPS FFT2R FC32 type Function
bool DspsF32Fft2rUnroller::checkDspsFft2rFc32Type(Function &F, LoopInfo &LI) {
  // Check parameter
  if (F.arg_size() != 3)
    return false;

  // Get Function parameter iterator
  Function::arg_iterator Args = F.arg_begin();

  // Check parameter type
  if (!Args->getType()->isPointerTy())
    return false;
  ++Args;

  if (!Args->getType()->isIntegerTy(32))
    return false;
  ++Args;

  if (!Args->getType()->isPointerTy())
    return false;

  return true;
}

// Check simplest dot product type Function
bool DspsF32DotprodSimpleLoopUnroller::checkIfDotProdSimplest(Function &F) {
  bool Flag = false;

  if (F.size() == 3) {
    // Get critical basic blocks
    BasicBlock *EntryBB = &F.getEntryBlock();
    BasicBlock *ForCondCleanup = getBasicBlockByName(F, "for.cond.cleanup");
    BasicBlock *ForBody = getBasicBlockByName(F, "for.body");

    if (EntryBB && ForCondCleanup && ForBody) {
      // Check floating point multiply-add instruction
      Instruction *Fmuladd = getFirstFMulAddInst(ForBody);
      if (Fmuladd) {
        // Check jump relation
        if (checkSuccessors(ForBody, 2, ForCondCleanup, ForBody)) {
          if (EntryBB->getTerminator()->getSuccessor(0) == ForBody) {
            Flag = true;
          }
        }
      }
    }
  }
  return Flag;
}

// Check complicated dot product type Function
bool DspsF32DotprodComplexLoopUnroller::checkIfDotProdComplicated(Function &F) {
  bool Flag1 = false;
  bool Flag2 = false;
  bool Flag3 = false;

  if (F.size() == 3) {
    // Get critical basic blocks
    BasicBlock *EntryBB = &F.getEntryBlock();
    BasicBlock *ForCondCleanup = getBasicBlockByName(F, "for.cond.cleanup");
    BasicBlock *ForBody = getBasicBlockByName(F, "for.body");

    if (EntryBB && ForCondCleanup && ForBody) {
      // Check floating point multiply-add instruction
      Instruction *Fmuladd = getFirstFMulAddInst(ForBody);
      if (Fmuladd) {
        // Check jump relation
        if (checkSuccessors(ForBody, 2, ForCondCleanup, ForBody)) {
          if (EntryBB->getTerminator()->getSuccessor(0) == ForBody) {
            Flag1 = true;
          }
        }
      }
    }

    if (ForBody) {
      // Check floating point operation instruction
      for (Instruction &Inst : *ForBody) {
        if (auto *BinOp = dyn_cast<BinaryOperator>(&Inst)) {
          if (BinOp->getOpcode() == Instruction::FAdd ||
              BinOp->getOpcode() == Instruction::FMul ||
              BinOp->getOpcode() == Instruction::FSub ||
              BinOp->getOpcode() == Instruction::FDiv) {
            Flag2 = true;
          }
        }
      }

      // Check floating point PHI node
      int FloatPhiCount = 0;
      for (PHINode &Phi : ForBody->phis()) {
        if (Phi.getType()->isFloatTy()) {
          FloatPhiCount++;
        }
      }
      if (FloatPhiCount == 1) {
        Flag3 = true;
      }
    }
  }

  return Flag1 && Flag2 && Flag3;
}

// Check DSPS F32 Biquad type Function
bool DspsF32BiquadLoopUnroller::checkDspsF32BiquadType(Function &F) {
  // Check basic block and parameter
  if (F.size() != 8 || F.arg_size() != 5)
    return false;

  // Get critical basic blocks
  BasicBlock *EntryBB = &F.getEntryBlock();
  BasicBlock *ForCondPreheader = getBasicBlockByName(F, "for.cond.preheader");
  BasicBlock *ForBodyLrPh = getBasicBlockByName(F, "for.body.lr.ph");
  BasicBlock *IfEnd = getBasicBlockByName(F, "if.end");
  BasicBlock *ForBody = getBasicBlockByName(F, "for.body");
  BasicBlock *ForCondCleanup = getBasicBlockByName(F, "for.cond.cleanup");

  // Check basic block existence
  if (!EntryBB || !ForCondPreheader || !ForBodyLrPh || !IfEnd || !ForBody ||
      !ForCondCleanup)
    return false;

  // Check basic block successor number
  if (succ_size(EntryBB) != 2 || succ_size(ForCondPreheader) != 2 ||
      succ_size(ForBodyLrPh) != 1 || succ_size(IfEnd) != 0 ||
      succ_size(ForBody) != 2 || succ_size(ForCondCleanup) != 1)
    return false;

  // Check jump relation
  if (EntryBB->getTerminator()->getSuccessor(0) != ForCondPreheader ||
      !checkSuccessors(ForCondPreheader, 2, ForBodyLrPh, IfEnd) ||
      !checkSuccessors(ForBodyLrPh, 1, ForBody) ||
      !checkSuccessors(ForBody, 2, ForCondCleanup, ForBody) ||
      !checkSuccessors(ForCondCleanup, 1, IfEnd))
    return false;

  return true;
}

// Check DSPS F32 Dotprod type Function with count
bool DspsF32DotprodSimpleLoopUnroller::checkDspsF32DotprodWithCount(
    Function &F, LoopInfo &LI, ScalarEvolution &SE) {
  // Get unique one-layer loop
  Loop *L = nullptr;
  for (Loop *TopLevelLoop : LI) {
    if (TopLevelLoop->getLoopDepth() == 1) {
      L = TopLevelLoop;
      break;
    }
  }
  if (!L || !L->getSubLoops().empty())
    return false;

  // Check if It is the simplest dot product type
  if (!checkIfDotProdSimplest(F))
    return false;

  // Check loop structure
  if (!L->getLoopLatch() || !L->getExitingBlock())
    return false;

  // Check loop count
  const SCEV *TripCount = SE.getBackedgeTakenCount(L);
  if (isa<SCEVConstant>(TripCount)) {
    return true;
  }
  return false;
}

// Check complicated DSPS F32 Dotprod type Function
bool DspsF32DotprodComplexLoopUnroller::checkDspsF32DotprodComplex(
    Function &F, LoopInfo &LI) {
  // Get unique one-layer loop
  Loop *L = nullptr;
  for (Loop *TopLevelLoop : LI) {
    if (TopLevelLoop->getLoopDepth() == 1) {
      L = TopLevelLoop;
      break;
    }
  }
  if (!L || !L->getSubLoops().empty())
    return false;

  // Check if It is the complicated dot product type
  if (!checkIfDotProdComplicated(F))
    return false;

  // Check loop structure
  if (!L->getLoopLatch() || !L->getExitingBlock())
    return false;

  if (L->getCanonicalInductionVariable())
    return false;

  // Check loop preheader
  BasicBlock *LoopPreheader = L->getLoopPreheader();
  if (LoopPreheader)
    return true;

  BasicBlock *LoopHeader = L->getHeader();
  BasicBlock *NewPreheader =
      BasicBlock::Create(LoopHeader->getContext(), "for.cond.preheader",
                         LoopHeader->getParent(), LoopHeader);

  DomTreeUpdater DTU(DT, DomTreeUpdater::UpdateStrategy::Eager);
  SmallVector<DominatorTree::UpdateType, 4> Updates;

  // Redirect all external predecessors to the new preheader basic block
  for (BasicBlock *Pred : predecessors(LoopHeader)) {
    if (!L->contains(Pred)) {
      Updates.push_back({DominatorTree::Delete, Pred, LoopHeader});
      Updates.push_back({DominatorTree::Insert, Pred, NewPreheader});

      Pred->getTerminator()->replaceUsesOfWith(LoopHeader, NewPreheader);
      // Update PHI nodes in the loop header to point to the new preheader basic
      // block
      for (PHINode &PN : LoopHeader->phis()) {
        int Index = PN.getBasicBlockIndex(Pred);
        if (Index != -1) {
          PN.setIncomingBlock(Index, NewPreheader);
        }
      }
    }
  }
  // Jump from the new preheader to the loop header
  BranchInst::Create(LoopHeader, NewPreheader);
  Updates.push_back({DominatorTree::Insert, NewPreheader, LoopHeader});

  DTU.applyUpdates(Updates);
  SE.forgetLoop(L);

  return true;
}

bool DspsF32MathLoopUnroller::checkDspsF32MathType(Function &F, LoopInfo &LI) {
  // Check the number of basic blocks
  if (F.size() != 6)
    return false;

  BasicBlock *Entry = &F.getEntryBlock();
  BasicBlock *IfEnd = getBasicBlockByName(F, "if.end");
  BasicBlock *ForCondPreheader = getBasicBlockByName(F, "for.cond.preheader");
  BasicBlock *ForBody = getBasicBlockByName(F, "for.body");
  BasicBlock *ForBodyClone = getBasicBlockByName(F, "for.body.clone");
  BasicBlock *Return = getBasicBlockByName(F, "return");

  if (!Entry || !IfEnd || !ForCondPreheader || !ForBody || !ForBodyClone ||
      !Return)
    return false;

  if (!checkSuccessors(Entry, 2, Return, IfEnd) ||
      !checkSuccessors(IfEnd, 2, ForBody, ForCondPreheader) ||
      !checkSuccessors(ForCondPreheader, 2, ForBodyClone, Return) ||
      !checkSuccessors(ForBody, 2, Return, ForBody) ||
      !checkSuccessors(ForBodyClone, 2, Return, ForBodyClone))
    return false;

  // Check if there are three outer loops, each with one inner loop
  int OuterLoopCount = 0;
  int InnerLoopCount = 0;
  for (Loop *L : LI.getLoopsInPreorder()) {
    if (L->getLoopDepth() == 1) {
      OuterLoopCount++;
      if (L->getSubLoops().size() == 1) {
        InnerLoopCount++;
      }
    }
  }

  if (OuterLoopCount != 2 || InnerLoopCount != 0) {
    return false;
  }

  return true;
}

bool DspsF32DotprodLoopUnroller::checkDspsF32DotprodType(Function &F,
                                                         LoopInfo &LI) {
  // Check the number of basic blocks
  if (F.size() != 5)
    return false;

  // Check the loop nesting level
  unsigned int MaxLoopDepth = 0;
  for (auto &BB : F) {
    MaxLoopDepth = std::max(MaxLoopDepth, LI.getLoopDepth(&BB));
  }
  if (MaxLoopDepth != 1) {
    return false;
  }

  BasicBlock *Entry = &F.getEntryBlock();
  BasicBlock *ForCondPreheader = getBasicBlockByName(F, "for.cond.preheader");
  BasicBlock *IfEnd = getBasicBlockByName(F, "if.end");
  BasicBlock *ForBody = getBasicBlockByName(F, "for.body");
  BasicBlock *ForBodyClone = getBasicBlockByName(F, "for.body.clone");

  if (!Entry || !IfEnd || !ForCondPreheader || !ForBody || !ForBodyClone)
    return false;

  if (!checkSuccessors(Entry, 2, ForBody, ForCondPreheader) ||
      !checkSuccessors(ForCondPreheader, 2, ForBodyClone, IfEnd) ||
      !checkSuccessors(ForBody, 2, IfEnd, ForBody) ||
      !checkSuccessors(ForBody, 2, IfEnd, ForBody) ||
      !checkSuccessors(ForBodyClone, 2, IfEnd, ForBodyClone))
    return false;

  // Check if there are three outer loops, each with one inner loop
  int OuterLoopCount = 0;
  int InnerLoopCount = 0;
  for (Loop *L : LI.getLoopsInPreorder()) {
    if (L->getLoopDepth() == 1) {
      OuterLoopCount++;
      if (L->getSubLoops().size() == 1) {
        InnerLoopCount++;
      }
    }
  }

  if (OuterLoopCount != 2 || InnerLoopCount != 0) {
    return false;
  }

  return true;
}

// Helper Function to unroll and duplicate a loop iteration
static Instruction *unrollAndDuplicateLoopIteration(LLVMContext &Ctx,
                                                    BasicBlock *BB,
                                                    IRBuilder<> &Builder,
                                                    unsigned int I) {
  PHINode *IPhi = dyn_cast<PHINode>(&BB->front());
  BasicBlock::iterator BeginIt, EndIt, ToIt;
  SmallVector<Instruction *, 8> NewInsts;
  ValueToValueMapTy ValueMap;
  Instruction *Add = nullptr;
  Instruction *DuplicatedPhiNode = nullptr;

  // Find the range of instructions to duplicate
  for (Instruction &Inst : *BB) {
    if (auto *Phi = dyn_cast<PHINode>(&Inst)) {
      if (Phi->getType()->isFloatTy()) {
        BeginIt = Inst.getIterator();
      }
    } else if (RecurrenceDescriptor::isFMulAddIntrinsic(&Inst)) {
      EndIt = std::next(Inst.getIterator());
      ToIt = std::next(EndIt);
      break;
    }
  }

  assert(&*BeginIt && &*EndIt && "Failed to find instruction range");

  // Clone and modify instructions
  int Arrayidx = 0;
  for (auto It = BeginIt; It != EndIt; ++It) {
    Instruction *NewInst = It->clone();
    if (NewInst->getOpcode() == Instruction::PHI)
      NewInst->setName("acc" + Twine(I));

    if (auto *GEP = dyn_cast<GetElementPtrInst>(NewInst)) {
      if (!Add)
        Add = BinaryOperator::CreateDisjoint(
            Instruction::Or, IPhi, ConstantInt::get(Type::getInt32Ty(Ctx), I),
            "add" + Twine(I), BB);

      NewInst->setName("Arrayidx" + Twine(I) + "_" + Twine(Arrayidx));
      NewInst->setOperand(1, Add);
      Arrayidx++;
    }
    NewInsts.push_back(NewInst);
    ValueMap[&*It] = NewInst;
  }

  // Update operands and insert new instructions
  updateOperands(NewInsts, ValueMap);
  for (Instruction *NewInst : NewInsts) {
    if (NewInst->getOpcode() == Instruction::PHI)
      DuplicatedPhiNode = NewInst->clone();
    NewInst->insertInto(BB, BB->end());
  }

  return DuplicatedPhiNode;
}

static void modifyFirdAddToOr(BasicBlock *ClonedForBody) {
  SmallVector<BinaryOperator *> AddInsts;

  // Collect all add instructions that meet the criteria
  for (auto &I : *ClonedForBody) {
    if (auto *BinOp = dyn_cast<BinaryOperator>(&I)) {
      if (BinOp->getOpcode() == Instruction::Add && BinOp->hasNoSignedWrap() &&
          BinOp->hasNoUnsignedWrap()) {
        AddInsts.push_back(BinOp);
      }
    }
  }
  if (AddInsts.empty()) {
    return;
  }
  // Replace each add instruction with an or disjoint instruction
  for (auto It = AddInsts.begin(); It != std::prev(AddInsts.end()); ++It) {
    auto *AddInst = *It;
    // Create a new or disjoint instruction
    Instruction *OrInst =
        BinaryOperator::CreateDisjoint(Instruction::Or, AddInst->getOperand(0),
                                       AddInst->getOperand(1), "add", AddInst);

    // Replace all uses of the add instruction
    AddInst->replaceAllUsesWith(OrInst);

    // Delete the original add instruction
    AddInst->eraseFromParent();
    OrInst->setName("add");
  }
}

// Helper Function to update predecessors to point to a new preheader
static void updatePredecessorsToPreheader(BasicBlock *ForBody,
                                          BasicBlock *ForBodyPreheader) {
  SmallVector<BasicBlock *, 4> PredecessorsBb;
  for (auto *Pred : predecessors(ForBody)) {
    if (Pred != ForBody)
      PredecessorsBb.push_back(Pred);
  }

  for (BasicBlock *Pred : PredecessorsBb) {
    Instruction *TI = Pred->getTerminator();
    for (unsigned I = 0; I < TI->getNumSuccessors(); ++I) {
      if (TI->getSuccessor(I) == ForBody) {
        TI->setSuccessor(I, ForBodyPreheader);
      }
    }
  }

  if (!ForBodyPreheader->getTerminator()) {
    BranchInst::Create(ForBody, ForBodyPreheader);
  }
}

// Helper Function to get the 'Len' value from the Entry block
static Value *getLenFromEntryBlock(Function &F) {
  ICmpInst *ICmp = nullptr;
  for (BasicBlock &BB : F) {
    ICmp = getFirstICmpInstWithPredicate(&BB, ICmpInst::ICMP_SGT);
    if (ICmp)
      break;
  }

  assert(ICmp && "Icmp sgt instruction not found");
  return ICmp->getOperand(0);
}

// Helper Function to find specific instructions in a basic block
static std::tuple<PHINode *, CallInst *, BinaryOperator *>
findKeyInstructions(BasicBlock *ForBody) {
  PHINode *ThirdPHI = nullptr;
  CallInst *CallInst2 = nullptr;
  BinaryOperator *AddInst = nullptr;
  int PHICount = 0;

  for (Instruction &Inst : *ForBody) {
    if (auto *PHI = dyn_cast<PHINode>(&Inst)) {
      PHICount++;
      if (PHICount == 3) {
        ThirdPHI = PHI;
      }
    } else if (auto *CI = dyn_cast<CallInst>(&Inst)) {
      CallInst2 = CI;
    } else if (auto *BinOp = dyn_cast<BinaryOperator>(&Inst)) {
      if (BinOp->getOpcode() == Instruction::Add) {
        AddInst = BinOp;
      }
    }
  }

  return std::make_tuple(ThirdPHI, CallInst2, AddInst);
}

// Helper Function to rename instructions
static void inline renameInstruction(Instruction *Inst) {
  if (Inst->getOpcode() == Instruction::PHI) {
    Inst->setName("acc");
  } else if (Inst->getOpcode() == Instruction::GetElementPtr) {
    Inst->setName("Arrayidx");
  }
}

// Helper Function to set add instruction in for body
static void inline setAddInForBody(Instruction *Inst, Instruction *Add,
                                   Instruction *InsertBefore) {
  if (Inst->getOpcode() == Instruction::PHI) {
    Add->moveBefore(InsertBefore);
  } else if (Inst->getOpcode() == Instruction::GetElementPtr) {
    Inst->setOperand(1, Add);
  }
}

// Helper Function to copy and remap instructions
static void copyAndRemapInstructions(Instruction *StartInst,
                                     Instruction *EndInst,
                                     Instruction *InsertBefore,
                                     Instruction *Add) {
  ValueToValueMapTy ValueMap;
  SmallVector<Instruction *, 8> NewInsts;

  for (auto It = StartInst->getIterator(); &*It != EndInst; ++It) {
    Instruction *NewInst = It->clone();
    if (auto *BinOp = dyn_cast<BinaryOperator>(NewInst)) {
      if (BinOp->getOpcode() == Instruction::Add) {
        continue;
      }
    }
    NewInsts.push_back(NewInst);
    ValueMap[&*It] = NewInst;
  }

  updateOperands(NewInsts, ValueMap);

  for (Instruction *NewInst : NewInsts) {
    renameInstruction(NewInst);
    NewInst->insertBefore(InsertBefore);
    setAddInForBody(NewInst, Add, InsertBefore);
  }
}

// Helper Function to preprocess the cloned for body
static void preProcessClonedForBody(BasicBlock *ClonedForBody, Value *Sub) {
  Instruction *AddInst = nullptr;
  for (Instruction &Inst : *ClonedForBody) {
    if (auto *BinOp = dyn_cast<BinaryOperator>(&Inst)) {
      if (BinOp->getOpcode() == Instruction::Add) {
        BinOp->setOperand(1, ConstantInt::get(BinOp->getType(), 8));
        AddInst = BinOp;
      }
    }
    if (auto *Icmp = dyn_cast<ICmpInst>(&Inst)) {
      Icmp->setPredicate(CmpInst::Predicate::ICMP_SLT);
      Icmp->setOperand(0, AddInst);
      Icmp->setOperand(1, Sub);
      Icmp->setName("cmp11");
    }
  }
}

// Helper Function to modify getelementptr instructions
static void modifyGetElementPtr(BasicBlock *BB) {
  SmallVector<GetElementPtrInst *, 8> GepInsts;
  Value *FirstGEPOperand0 = nullptr;
  Value *SecondGEPOperand1 = nullptr;

  for (Instruction &Inst : *BB) {
    if (auto *GEP = dyn_cast<GetElementPtrInst>(&Inst)) {
      GepInsts.push_back(GEP);
    }
  }

  if (GepInsts.size() < 8 || GepInsts.size() % 2 != 0) {
    return;
  }

  FirstGEPOperand0 = GepInsts[0];
  SecondGEPOperand1 = GepInsts[1];

  for (size_t I = 2; I < GepInsts.size(); ++I) {
    if (I % 2 == 0) {
      if (I < GepInsts.size() - 2) {
        GepInsts[I]->setOperand(0, FirstGEPOperand0);
      }
    } else {
      GepInsts[I]->setOperand(0, SecondGEPOperand1);
    }

    if (I == 14)
      continue;

    Instruction *Operand1 = dyn_cast<Instruction>(GepInsts[I]->getOperand(1));
    GepInsts[I]->setOperand(
        1, ConstantInt::get(Type::getInt32Ty(BB->getContext()), I / 2));
    if (Operand1 && Operand1->use_empty()) {
      Operand1->eraseFromParent();
    }
  }
}

// Helper Function to check if a PHI node has an incoming value of zero
static bool isIncomingValueZeroOfPhi(PHINode *Phi) {
  return Phi->getType()->isIntegerTy(32) &&
         isa<ConstantInt>(Phi->getIncomingValue(0)) &&
         cast<ConstantInt>(Phi->getIncomingValue(0))->isZero();
}

// Helper Function to find and set add instructions
static std::pair<Instruction *, Instruction *>
findAndSetAddInstructions(BasicBlock *ClonedForBody) {
  Instruction *FirstAdd = nullptr;
  Instruction *SecondAdd = nullptr;

  for (Instruction &Inst : *ClonedForBody) {
    if (BinaryOperator *BinOp = dyn_cast<BinaryOperator>(&Inst)) {
      if (BinOp->getOpcode() == Instruction::Add) {
        if (!FirstAdd) {
          FirstAdd = &Inst;
          FirstAdd->setHasNoSignedWrap(true);
        } else if (!SecondAdd) {
          SecondAdd = &Inst;
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
  for (PHINode &Phi : block->phis()) {
    if (isIncomingValueZeroOfPhi(&Phi)) {
      return &Phi;
    }
  }
  return nullptr;
}

static PHINode *findIntegerPHI(BasicBlock *block) {
  for (PHINode &Phi : block->phis()) {
    if (Phi.getType()->isIntegerTy(32) && !isIncomingValueZeroOfPhi(&Phi)) {
      return &Phi;
    }
  }
  return nullptr;
}

// Helper Function to unroll loop body
static void unrollLoopBody(BasicBlock *block, PHINode *ThirdPHI,
                           Instruction *CallInst, Instruction *AddInst,
                           PHINode *ZeroInitializedPHI, LLVMContext &Context) {
  for (int I = 1; I < 8; I++) {
    Instruction *Add = BinaryOperator::CreateDisjoint(
        Instruction::Or, ZeroInitializedPHI,
        ConstantInt::get(Type::getInt32Ty(Context), I), "add" + Twine(I),
        block);
    copyAndRemapInstructions(ThirdPHI, CallInst->getNextNode(), AddInst, Add);
  }
}

// Helper Function to modify getelementptr for unrolling
static void modifyGetElementPtrForUnrolling(BasicBlock *block) {
  SmallVector<GetElementPtrInst *, 8> GepInsts;
  for (Instruction &Inst : *block) {
    if (auto *GEP = dyn_cast<GetElementPtrInst>(&Inst)) {
      GepInsts.push_back(GEP);
    }
  }

  for (size_t I = 2; I < GepInsts.size(); I += 2) {
    GepInsts[I]->setOperand(0, GepInsts[0]);
    GepInsts[I]->setOperand(
        1, ConstantInt::get(Type::getInt32Ty(block->getContext()), I / 2));
  }
}

// Helper Function to handle add instructions
static void handleAddInstructions(BasicBlock *block, unsigned int unrollFactor,
                                  PHINode *ZeroInitializedPHI,
                                  LLVMContext &Context) {
  auto [FirstAdd, SecondAdd] = findAndSetAddInstructions(block);

  if (FirstAdd && SecondAdd) {
    FirstAdd->moveBefore(SecondAdd);

    if (unrollFactor == 1) {
      FirstAdd->setOperand(1, ConstantInt::get(Type::getInt32Ty(Context), 8));
      SecondAdd->setOperand(0, ZeroInitializedPHI);
    }
  }
}

// Function to unroll the cloned for loop body
static void unrollClonedForBody(BasicBlock *clonedForBody,
                                BasicBlock *ForCondPreheader,
                                unsigned int unrollFactor = 0) {
  Function *Function = clonedForBody->getParent();
  LLVMContext &Context = Function->getContext();

  // Find key instructions in the cloned for body
  auto [ThirdPHI, CallInst, AddInst] = findKeyInstructions(clonedForBody);
  PHINode *ZeroInitializedPHI = findZeroInitializedPHI(clonedForBody);
  PHINode *IntegerPHI = findIntegerPHI(clonedForBody);

  assert(ZeroInitializedPHI && "No matching zero-initialized PHI node found");

  // Unroll the loop body if key instructions are found
  if (ThirdPHI && CallInst) {
    unrollLoopBody(clonedForBody, ThirdPHI, CallInst, AddInst,
                   ZeroInitializedPHI, Context);
  }

  // Update the add instruction
  if (AddInst) {
    AddInst->setOperand(1, ConstantInt::get(Type::getInt32Ty(Context), 8));
    AddInst->setOperand(0, IntegerPHI);
  }

  // Update the basic block Terminator
  Instruction *Terminator = clonedForBody->getTerminator();
  Terminator->setSuccessor(0, clonedForBody);
  Terminator->setSuccessor(1, ForCondPreheader);

  // Move PHI nodes to the top of the basic block
  movePHINodesToTop(*clonedForBody);

  // Modify getelementptr instructions based on the unroll factor
  if (unrollFactor == 0) {
    modifyGetElementPtr(clonedForBody);
  } else {
    modifyGetElementPtrForUnrolling(clonedForBody);
  }

  // Handle add instructions
  handleAddInstructions(clonedForBody, unrollFactor, ZeroInitializedPHI,
                        Context);
}

// Function to check if a call instruction can be moved
static bool canMoveCallInstruction(CallInst *callInst,
                                   Instruction *InsertPoint) {
  for (unsigned I = 0; I < callInst->getNumOperands(); ++I) {
    if (auto *OperandInst = dyn_cast<Instruction>(callInst->getOperand(I))) {
      if (OperandInst->getParent() == callInst->getParent() &&
          InsertPoint->comesBefore(OperandInst)) {
        return false;
      }
    }
  }
  return true;
}

// Function to group and reorder instructions in a basic block
static void groupAndReorderInstructions(BasicBlock *clonedForBody) {
  // Collect different types of instructions
  SmallVector<PHINode *> PhiNodes;
  SmallVector<Instruction *> OrInsts, GepInsts, LoadInsts, StoreInsts, MulInsts,
      AddInsts, SubInsts, CallInsts, FaddInsts, FmulInsts, FsubInsts;

  // Categorize instructions by type
  for (Instruction &Inst : *clonedForBody) {
    if (auto *Phi = dyn_cast<PHINode>(&Inst)) {
      PhiNodes.push_back(Phi);
    } else if (Inst.getOpcode() == Instruction::Or) {
      OrInsts.push_back(&Inst);
    } else if (isa<GetElementPtrInst>(&Inst)) {
      GepInsts.push_back(&Inst);
    } else if (isa<LoadInst>(&Inst)) {
      LoadInsts.push_back(&Inst);
    } else if (isa<StoreInst>(&Inst)) {
      StoreInsts.push_back(&Inst);
    } else if (Inst.getOpcode() == Instruction::Mul) {
      MulInsts.push_back(&Inst);
    } else if (isa<CallInst>(&Inst)) {
      CallInsts.push_back(&Inst);
    } else if (Inst.getOpcode() == Instruction::Add) {
      AddInsts.push_back(&Inst);
    } else if (Inst.getOpcode() == Instruction::Sub) {
      SubInsts.push_back(&Inst);
    } else if (Inst.getOpcode() == Instruction::FAdd) {
      FaddInsts.push_back(&Inst);
    } else if (Inst.getOpcode() == Instruction::FMul) {
      FmulInsts.push_back(&Inst);
    } else if (Inst.getOpcode() == Instruction::FSub) {
      FsubInsts.push_back(&Inst);
    } else if (Inst.getOpcode() == Instruction::AShr) {
      return;
    }
  }

  // If no PHI nodes are found, return
  if (PhiNodes.empty()) {
    return;
  }

  // Reorder instructions
  Instruction *InsertPoint = PhiNodes.back()->getNextNode();
  bool CanMoveCallInst =
      CallInsts.empty() ||
      canMoveCallInstruction(dyn_cast<CallInst>(CallInsts[0]), InsertPoint);

  auto moveInstructions = [&InsertPoint](SmallVector<Instruction *> &Insts) {
    for (auto *Inst : Insts) {
      Inst->moveBefore(InsertPoint);
      InsertPoint = Inst->getNextNode();
    }
  };

  // Move instructions in the desired order
  moveInstructions(MulInsts);
  moveInstructions(AddInsts);
  moveInstructions(OrInsts);
  moveInstructions(SubInsts);
  moveInstructions(GepInsts);
  moveInstructions(LoadInsts);
  moveInstructions(FaddInsts);
  moveInstructions(FmulInsts);
  moveInstructions(FsubInsts);
  if (CanMoveCallInst) {
    moveInstructions(CallInsts);
  }
}

// Function to transform a single loop depth (currently suitable for
// dotprod/dotprode example)
bool DspsF32DotprodLoopUnroller::transformOneLoopDepth(Function &F) {
  LLVMContext &Ctx = F.getContext();
  bool Changed = false;

  // Get necessary basic blocks and values
  Value *Len = getLenFromEntryBlock(F);
  BasicBlock *EntryBB = &F.getEntryBlock();
  BasicBlock *ForBodyBB = getBasicBlockByName(F, "for.body");
  BasicBlock *ForBodyNewBB = getBasicBlockByName(F, "for.body.clone");
  BasicBlock *IfEnd = getBasicBlockByName(F, "if.end");
  BasicBlock *ForCond46PreheaderBB =
      getBasicBlockByName(F, "for.cond.preheader");

  assert(ForBodyBB && "Expected to find for.body!");
  assert(ForBodyNewBB && "Expected to find for.body.clone!");
  assert(IfEnd && "Expected to find if.end!");
  assert(ForCond46PreheaderBB && "Expected to find for.cond.preheader!");

  // Create new basic blocks
  BasicBlock *ForCondPreheaderBB =
      BasicBlock::Create(F.getContext(), "for.cond.preheader", &F, ForBodyBB);
  BasicBlock *ForBodyPreheaderBB =
      BasicBlock::Create(F.getContext(), "for.body.preheader", &F, ForBodyBB);
  BasicBlock *ForCond31PreheaderBB =
      BasicBlock::Create(F.getContext(), "for.cond31.preheader", &F, ForBodyBB);
  BasicBlock *ForBody33BB = cloneBasicBlockWithRelations(ForBodyBB, "33", &F);
  ForBody33BB->setName("for.body33");
  ForBody33BB->moveAfter(ForBodyBB);
  BasicBlock *ForEnd37BB =
      BasicBlock::Create(F.getContext(), "for.end37", &F, ForBodyNewBB);

  // Add instructions to ForCondPreheaderBB
  IRBuilder<> builder(ForCondPreheaderBB);
  Value *NegativeSeven = ConstantInt::getSigned(Type::getInt32Ty(F.getContext()), -7);
  Value *Sub = builder.CreateNSWAdd(Len, NegativeSeven, "Sub");
  Value *Seven = ConstantInt::get(Type::getInt32Ty(F.getContext()), 7);
  Value *Cmp1113 = builder.CreateICmpUGT(Len, Seven, "Cmp1113");
  builder.CreateCondBr(Cmp1113, ForBodyPreheaderBB, ForCond31PreheaderBB);

  // Add instructions to ForBodyPreheaderBB
  builder.SetInsertPoint(ForBodyPreheaderBB);
  Value *Mask = ConstantInt::get(Type::getInt32Ty(F.getContext()), 2147483640);
  Value *AndValue = builder.CreateAnd(Len, Mask, "");
  builder.CreateBr(ForBodyBB);

  // Modify for.body
  PHINode *IPhi = dyn_cast<PHINode>(&ForBodyBB->front());
  IPhi->setName("I.0122");

  // copy first float phinode from ForBodyBB to ForCond31PreheaderBB
  PHINode *FirstFloatPhi = getFirstFloatPhi(ForBodyBB);
  PHINode *Acc00Lcssa = PHINode::Create(FirstFloatPhi->getType(), 2,
                                        "acc0.0.lcssa", ForCond31PreheaderBB);
  Acc00Lcssa->addIncoming(FirstFloatPhi->getIncomingValue(0),
                          FirstFloatPhi->getIncomingBlock(0));
  Acc00Lcssa->addIncoming(FirstFloatPhi->getIncomingValue(1),
                          ForCondPreheaderBB);
  // Unroll and duplicate loop iterations
  SmallVector<Instruction *> instructions;
  for (int I = 0; I < 7; I++) {
    Instruction *CopyedPhiNode =
        unrollAndDuplicateLoopIteration(Ctx, ForBodyBB, builder, I + 1);
    if (PHINode *Phi = dyn_cast<PHINode>(CopyedPhiNode)) {
      Phi->setName("acc" + Twine(I + 1) + ".0.lcssa");
      Phi->setIncomingBlock(1, ForCondPreheaderBB);
      Phi->insertInto(ForCond31PreheaderBB, ForCond31PreheaderBB->end());
      instructions.push_back(Phi);
    }
  }

  // Update for.body Terminator
  Instruction *IncInst = nullptr;
  MDNode *LoopMD = nullptr;
  for (auto &I : *ForBodyBB) {
    if (I.getOpcode() == Instruction::Add) {
      IncInst = &I;
      Instruction *Icmp = I.getNextNode();
      Instruction *Br = Icmp->getNextNode();
      assert(Icmp->getOpcode() == Instruction::ICmp &&
             Br->getOpcode() == Instruction::Br &&
             "Unexpected instruction sequence");
      I.moveAfter(&ForBodyBB->back());
      LoopMD = Br->getMetadata(LLVMContext::MD_loop);
      Br->eraseFromParent();
      Icmp->eraseFromParent();
      break;
    }
  }

  // Modify add instruction
  IncInst->setOperand(1, ConstantInt::get(Type::getInt32Ty(F.getContext()), 8));
  IncInst->setName("add30");

  builder.SetInsertPoint(ForBodyBB);
  Value *Cmp1 = builder.CreateICmpSLT(IncInst, Sub, "Cmp1");
  BranchInst *newBr =
      builder.CreateCondBr(Cmp1, ForBodyBB, ForCond31PreheaderBB);
  newBr->setMetadata(LLVMContext::MD_loop, LoopMD);

  movePHINodesToTop(*ForBodyBB);
  for (PHINode &PHI : ForBodyBB->phis()) {
    PHI.setIncomingBlock(1, ForBodyPreheaderBB);
  }

  // Add instructions to ForCond31PreheaderBB
  builder.SetInsertPoint(ForCond31PreheaderBB);
  PHINode *I0Lcssa =
      builder.CreatePHI(Type::getInt32Ty(F.getContext()), 0, "I.0.lcssa");
  I0Lcssa->addIncoming(ConstantInt::get(Type::getInt32Ty(F.getContext()), 0),
                       ForCondPreheaderBB);
  I0Lcssa->addIncoming(AndValue, ForBodyBB);
  Value *Cmp32132 = builder.CreateICmpSLT(I0Lcssa, Len, "Cmp32132");
  builder.CreateCondBr(Cmp32132, ForBody33BB, ForEnd37BB);

  // Modify ForBody33BB
  Instruction *TempInstr = nullptr;
  for (PHINode &PHI : ForBody33BB->phis()) {
    if (PHI.getType()->isIntegerTy(32)) {
      PHI.setIncomingValue(1, I0Lcssa);
      PHI.setIncomingBlock(1, ForCond31PreheaderBB);
    } else if (PHI.getType()->isFloatTy()) {
      PHI.setIncomingValue(1, Acc00Lcssa);
      PHI.setIncomingBlock(1, ForCond31PreheaderBB);
      TempInstr = &PHI;
    }
  }

  // Modify ForEnd37BB
  Instruction *Acc01Lcssa = TempInstr->clone();
  Acc01Lcssa->setName("acc0.1.lcssa");
  Acc01Lcssa->insertInto(ForEnd37BB, ForEnd37BB->end());
  builder.SetInsertPoint(ForEnd37BB);

  // Create pairs of floating-point additions
  Value *Sum01 = builder.CreateFAdd(Acc01Lcssa, instructions[0], "Sum01");
  Value *Sum23 = builder.CreateFAdd(instructions[1], instructions[2], "Sum23");
  Value *Sum45 = builder.CreateFAdd(instructions[3], instructions[4], "Sum45");
  Value *Sum67 = builder.CreateFAdd(instructions[5], instructions[6], "Sum67");

  // Combine pairs
  Value *Sum0123 = builder.CreateFAdd(Sum01, Sum23, "Sum0123");
  Value *Sum4567 = builder.CreateFAdd(Sum45, Sum67, "Sum4567");

  // Final addition
  Value *CurrentAdd = builder.CreateFAdd(Sum0123, Sum4567, "add44");
  builder.CreateBr(IfEnd);

  // Modify Entry basic block
  BranchInst *EntryBi = dyn_cast<BranchInst>(EntryBB->getTerminator());
  EntryBi->setSuccessor(0, ForCondPreheaderBB);
  EntryBi->setSuccessor(1, ForCond46PreheaderBB);

  // Modify ForCond46PreheaderBB
  ForCond46PreheaderBB->getTerminator()->getPrevNode()->setName("cmp47110");

  // Modify for.body33
  BranchInst *ForBody33Bi = dyn_cast<BranchInst>(ForBody33BB->getTerminator());
  ForBody33Bi->setSuccessor(0, ForEnd37BB);
  ForBody33Bi->setSuccessor(1, ForBody33BB);

  // Modify if.end
  PHINode *IfEndPhi = dyn_cast<PHINode>(&IfEnd->front());
  IfEndPhi->setIncomingValue(1, CurrentAdd);
  IfEndPhi->setIncomingBlock(1, ForEnd37BB);

  Changed = true;
  return Changed;
}

// Function to unroll the cloned for.cond.preheader
static void unrollClonedForCondPreheader(BasicBlock *clonedForBody,
                                         BasicBlock *clonedForCondPreheader,
                                         BasicBlock *ForCondPreheader) {
  Function *F = clonedForBody->getParent();
  BasicBlock *ForBody = getBasicBlockByName(*F, "for.body");
  assert(ForBody && "Expected to find for.body!");

  DeleteDeadPHIs(clonedForCondPreheader);
  // Clone PHI instructions to the beginning of clonedForCondPreheader
  Instruction *InsertPoint = &clonedForCondPreheader->front();
  SmallVector<PHINode *> clonedPhiNodes;
  for (PHINode &Phi : clonedForBody->phis()) {
    PHINode *ClonedPhi = cast<PHINode>(Phi.clone());
    ClonedPhi->setName(Phi.getName() + ".clone");
    ClonedPhi->setIncomingBlock(0, ForBody);
    ClonedPhi->insertBefore(InsertPoint);
    InsertPoint = ClonedPhi->getNextNode();
    clonedPhiNodes.push_back(ClonedPhi);
  }
  // Find and clone the unique Icmp instruction in ForBody
  Value *SpecStoreSelect = nullptr;
  Instruction *CmpSlt = nullptr;
  for (Instruction &Inst : *ForBody) {
    if (auto *Icmp = dyn_cast<ICmpInst>(&Inst)) {
      SpecStoreSelect = Icmp->getOperand(0);
      CmpSlt = Icmp->clone();
      CmpSlt->setName("cmp_slt");
      CmpSlt->insertAfter(InsertPoint);
      break;
    }
  }
  assert(SpecStoreSelect && "Failed to find Icmp instruction in ForBody");

  // Replace the existing Icmp in clonedForCondPreheader
  for (Instruction &Inst : *clonedForCondPreheader) {
    if (auto *Icmp = dyn_cast<ICmpInst>(&Inst)) {
      Icmp->replaceAllUsesWith(CmpSlt);
      Icmp->eraseFromParent();
      break;
    }
  }

  // Set the operand of cmp_slt to the first cloned PHI node
  CmpSlt->setOperand(0, clonedPhiNodes[0]);

  // Update the successor of clonedForCondPreheader
  clonedForCondPreheader->getTerminator()->setSuccessor(1, ForCondPreheader);
}

static std::tuple<Value *, Value *, Value *>
modifyForBodyPreheader(BasicBlock *ForBodyPreheader,
                       BasicBlock *ClonedForCondPreheader) {
  PHINode *TargetPHI = nullptr;
  PHINode *TargetPHI2 = nullptr;
  PHINode *TargetPHI3 = nullptr;
  for (PHINode &Phi : ClonedForCondPreheader->phis()) {
    if (Phi.getType()->isIntegerTy(32)) {
      if (isIncomingValueZeroOfPhi(&Phi)) {
        // Found the target PHI node
        TargetPHI = &Phi;
      } else {
        TargetPHI2 = &Phi;
      }
    } else if (Phi.getType()->isFloatTy()) {
      if (TargetPHI3 == nullptr) {
        TargetPHI3 = &Phi;
        break;
      }
    }
  }

  BinaryOperator *NewSub = nullptr;
  for (Instruction &Inst : *ForBodyPreheader) {
    if (auto *BinOp = dyn_cast<BinaryOperator>(&Inst)) {
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
  Instruction *TargetInst = getFirstFMulAddInst(ForBody);
  assert(TargetInst && "TargetInst not found");
  Value *NewSub = std::get<0>(NewSubAndTargetPHI3);
  Value *TargetPHI2 = std::get<1>(NewSubAndTargetPHI3);
  Value *TargetPHI3 = std::get<2>(NewSubAndTargetPHI3);
  // Create new .loopexit basic block
  BasicBlock *LoopExit = BasicBlock::Create(
      ForCondPreheader->getContext(), ForCondPreheader->getName() + ".loopexit",
      ForCondPreheader->getParent(), ForCondPreheader);

  // Create new Sub instruction in .loopexit block
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
  // result of the new Sub instruction
  for (PHINode &Phi : ForCondPreheader->phis()) {
    if (Phi.getType()->isIntegerTy(32)) {
      Phi.setIncomingValue(0, TargetPHI);
      Phi.setIncomingValue(1, NewSubInst);
    } else if (Phi.getType()->isFloatTy()) {
      Phi.setIncomingValue(0, TargetPHI3);
      Phi.setIncomingValue(1, TargetInst);
    }
  }
  setPHINodesBlock(ForCondPreheader, ClonedForCondPreheader, LoopExit);
  // Get the Icmp instruction in ForCondPreheader
  ICmpInst *icmpInst = getFirstInst<ICmpInst>(ForCondPreheader);

  // Ensure we found the Icmp instruction
  assert(icmpInst && "Failed to find Icmp instruction in ForCondPreheader");

  // Set the operand 1 of icmpInst to constant 7
  LLVMContext &Ctx = ForCondPreheader->getContext();
  Value *const7 = ConstantInt::get(Type::getInt32Ty(Ctx), 7);
  icmpInst->setOperand(1, const7);

  // Create a new add nsw instruction before icmpInst, with operand 0 the same
  // as icmpInst, and operand 1 as -7. This instruction will be used as the
  // return value of the Function
  Value *constNeg7 = ConstantInt::getSigned(Type::getInt32Ty(Ctx), -7);
  IRBuilder<> BuilderBeforeICmp(icmpInst);
  Value *AddInst =
      BuilderBeforeICmp.CreateNSWAdd(icmpInst->getOperand(0), constNeg7);

  ForBody->getTerminator()->setSuccessor(0, LoopExit);

  return AddInst;
}

static void updateRealForBody(Function &F, Value *Sub) {
  BasicBlock *ForBody = getBasicBlockByName(F, "for.body");
  assert(ForBody && "Expected to find for.body!");
  ICmpInst *LastICmp =
      getLastICmpInstWithPredicate(ForBody, ICmpInst::ICMP_SLT);
  if (LastICmp) {
    LastICmp->setOperand(1, Sub);
  }
}

// Helper Function to find two i32 PHI nodes in a basic block
static std::pair<PHINode *, PHINode *> findTwoI32PhiInBB(BasicBlock *BB) {
  PHINode *FirstI32Phi = nullptr;
  PHINode *secondI32Phi = nullptr;

  for (PHINode &Phi : BB->phis()) {
    if (Phi.getType()->isIntegerTy(32)) {
      if (isIncomingValueZeroOfPhi(&Phi)) {
        FirstI32Phi = &Phi;
      } else {
        secondI32Phi = &Phi;
      }
      if (FirstI32Phi && secondI32Phi)
        break;
    }
  }

  assert(FirstI32Phi && secondI32Phi &&
         "Failed to find two i32 type PHI nodes in basic block");

  return std::make_pair(FirstI32Phi, secondI32Phi);
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

  // Find the unique Icmp eq instruction in ForBody
  ICmpInst *IcmpEq = getFirstICmpInstWithPredicate(ForBody, ICmpInst::ICMP_EQ);

  // Ensure we found the Icmp eq instruction
  assert(IcmpEq && "Failed to find Icmp eq instruction in ForBody");

  // Get the original operand 1
  Value *OriginalOperand1 = IcmpEq->getOperand(1);

  // Ensure the original operand 1 is an instruction
  if (Instruction *OriginalOperand1Inst =
          dyn_cast<Instruction>(OriginalOperand1)) {
    // Set operand 1 to the operand 0 of the original operand 1 instruction
    IcmpEq->setOperand(1, OriginalOperand1Inst->getOperand(0));
  } else {
    llvm_unreachable("The original operand 1 is not an instruction, "
                     "cannot get its operand 0\n");
  }

  // Original code can be simplified to:
  auto [TargetPHI, TargetPHI2] = findTwoI32PhiInBB(ClonedForCondPreheader);
  auto [TargetPHIInForBody, TargetPHIInForBody2] = findTwoI32PhiInBB(ForBody);

  // Set the incoming value of the PHI nodes found in ForBody
  // to the PHI nodes found in ClonedForCondPreheader
  TargetPHIInForBody->setIncomingValue(0, TargetPHI);
  TargetPHIInForBody2->setIncomingValue(0, TargetPHI2);

  IcmpEq->setOperand(0, TargetPHIInForBody2->getIncomingValue(1));
}

static void insertUnusedInstructionsBeforeIcmp(PHINode *phiI32InClonedForBody,
                                               ICmpInst *LastIcmpEq) {
  for (Use &U : phiI32InClonedForBody->uses()) {
    if (Instruction *Used = dyn_cast<Instruction>(U.getUser())) {
      if (Used->getParent() == nullptr) {
        if (Used->use_empty()) {
          Used->insertBefore(LastIcmpEq);
        }
      }
    }
  }
}

static void modifyClonedForBody(BasicBlock *ClonedForBody) {

  ICmpInst *LastIcmpEq = getLastInst<ICmpInst>(ClonedForBody);
  assert(LastIcmpEq &&
         "Failed to find last Icmp eq instruction in ClonedForBody");

  PHINode *phiI32InClonedForBody = nullptr;
  for (PHINode &Phi : ClonedForBody->phis()) {
    if (isIncomingValueZeroOfPhi(&Phi)) {
      phiI32InClonedForBody = &Phi;
      insertUnusedInstructionsBeforeIcmp(phiI32InClonedForBody, LastIcmpEq);
    }
  }

  // Ensure that the Phi i32 node is found
  assert(phiI32InClonedForBody && "Phi i32 node not found in ClonedForBody");
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
modifyFirstForBody(Loop *L, Function &F, BasicBlock *ForBody, Value *Sub) {

  BasicBlock *ForBodyPreheader = L->getLoopPreheader();

  // Find the predecessor of ForBodyPreheader
  BasicBlock *PreForBody = nullptr;
  assert(pred_size(ForBodyPreheader) == 1 &&
         "ForBodyPreheader should have only one predecessor");
  for (auto *Pred : predecessors(ForBodyPreheader)) {
    PreForBody = Pred;
  }

  // Find the first successor of ForBody, It should have two
  BasicBlock *ForCondPreheader = ForBody->getTerminator()->getSuccessor(0);

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

  preProcessClonedForBody(ClonedForBody, Sub);
  updateRealForBody(F, Sub);
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

  BasicBlock *ForCondPreheader = getBasicBlockByName(F, "for.cond.preheader");
  BasicBlock *ForBodyLrPh = getBasicBlockByName(F, "for.body.lr.ph");
  assert(ForCondPreheader && "Expected to find for.cond.preheader!");
  assert(ForBodyLrPh && "Expected to find for.body.lr.ph!");
  ForCondPreheader->replaceAllUsesWith(ForBodyLrPh);
  ForCondPreheader->eraseFromParent();
  ForBodyLrPh->setName("for.cond.preheader");

  unsigned int Loadnum = 0;
  for (auto I = ForBodyLrPh->begin(); I != ForBodyLrPh->end(); ++I) {
    if (auto *Loadinst = dyn_cast<LoadInst>(&*I)) {
      Loadnum++;
      if (Loadnum == 2) {
        IRBuilder<> Builder(Loadinst->getNextNode());
        Value *NegSeven = ConstantInt::getSigned(Type::getInt32Ty(Ctx), -7);
        Value *Sub = Builder.CreateNSWAdd(Loadinst, NegSeven, "Sub");
        return Sub; // Return the newly inserted instruction
      }
    }
  }
  llvm_unreachable("It must not be here");
}

static void modifyForCondPreheader2(BasicBlock *ClonedForBody,
                                    BasicBlock *ClonedForCondPreheader,
                                    BasicBlock *ForCondPreheader,
                                    Value *andinst) {

  // Clone the found Phi instructions to the beginning of ClonedForCondPreheader
  // in order
  Instruction *InsertPoint = &ForCondPreheader->front();
  PHINode *Phi = cast<PHINode>(InsertPoint);

  BasicBlock *LastForCondPreheader = Phi->getIncomingBlock(0);
  SmallVector<PHINode *> ClonedPhiNodes;
  unsigned int floatphicount = 0;
  for (PHINode &Phi2 : ClonedForBody->phis()) {
    PHINode *ClonedPhi = cast<PHINode>(Phi2.clone());
    ClonedPhi->setName(Phi2.getName() + ".clone");
    // Modify the operand 0 basicblock of each Phi instruction to ForBody
    if (Phi2.getType()->isFloatTy()) {
      if (floatphicount == 0) {
        ClonedPhi->setIncomingValue(0, Phi->getIncomingValue(0));
        floatphicount++;
      }
    }
    ClonedPhi->setIncomingBlock(0, LastForCondPreheader);
    ClonedPhi->insertAfter(InsertPoint);
    // Update the insertion point to after the newly inserted PHI node
    InsertPoint = ClonedPhi;

    ClonedPhiNodes.push_back(ClonedPhi);
  }

  // Find operand 1 of the Icmp instruction from ClonedForBody
  ICmpInst *firstIcmp = getFirstInst<ICmpInst>(ClonedForBody);
  assert(firstIcmp && "Unable to find Icmp instruction in ClonedForBody");
  Value *IcmpOperand1 = firstIcmp->getOperand(1);

  // Set operand 0 of Icmp in ForCondPreheader to ClonedPhiNodes[0], and operand
  // 1 to IcmpOperand1
  for (Instruction &Inst : *ForCondPreheader) {
    if (ICmpInst *Icmp = dyn_cast<ICmpInst>(&Inst)) {
      Icmp->setOperand(0, ClonedPhiNodes[0]);
      Icmp->setOperand(1, IcmpOperand1);
      Icmp->setName("Cmp");
      break;
    }
  }

  ForCondPreheader->getTerminator()->setSuccessor(1, ClonedForCondPreheader);

  // Delete redundant getelementptr, store and add instructions
  SmallVector<Instruction *> InstructionsToRemove;
  for (Instruction &Inst : *ForCondPreheader) {
    if (isa<GetElementPtrInst>(&Inst) || isa<StoreInst>(&Inst) ||
        isa<BinaryOperator>(&Inst)) {
      InstructionsToRemove.push_back(&Inst);
    }
  }
  for (auto Inst = InstructionsToRemove.rbegin();
       Inst != InstructionsToRemove.rend(); ++Inst) {
    if ((*Inst)->use_empty()) {
      (*Inst)->eraseFromParent();
    }
  }
  // Find the Icmp instruction in ClonedForCondPreheader
  ICmpInst *IcmpInForCondPreheader =
      getFirstICmpInstWithPredicate(ForCondPreheader, ICmpInst::ICMP_EQ);

  // Ensure that the Icmp instruction is found
  assert(IcmpInForCondPreheader &&
         "Icmp instruction not found in ClonedForCondPreheader");

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
    llvm_unreachable("The original operand 1 is not an instruction, cannot get "
                     "its operand 0\n");
  }

  // Find Phi i32 node in ForCondPreheader with incoming 0 value == 0
  PHINode *TargetPhi = nullptr;
  for (PHINode &Phi : ForCondPreheader->phis()) {
    if (isIncomingValueZeroOfPhi(&Phi)) {
      TargetPhi = &Phi;
      break;
    }
  }

  // Ensure the target Phi node is found
  assert(TargetPhi && "No matching Phi i32 node found in ForCondPreheader");

  TargetPhi->setIncomingValue(1, andinst);
}

static Value *modifyClonedForBodyPreheader(BasicBlock *ClonedForBodyPreheader,
                                           BasicBlock *ForBody) {
  ICmpInst *firstIcmp = getFirstInst<ICmpInst>(ForBody);
  assert(firstIcmp && "Unable to find Icmp instruction in ForBody");

  Value *IcmpOperand1 = firstIcmp->getOperand(1);

  IRBuilder<> Builder(ClonedForBodyPreheader->getTerminator());
  Value *AndInst =
      Builder.CreateAnd(IcmpOperand1, Builder.getInt32(2147483640));
  return AndInst;
}

static void modifyClonedForCondPreheader(BasicBlock *ClonedForCondPreheader,
                                         BasicBlock *ForBody,
                                         BasicBlock *ForCondPreheader) {

  PHINode *FloatPhiInForBody =
      cast<PHINode>(getFirstFloatPhi(ForBody)->clone());
  // Find and replace float type Phi node in ClonedForCondPreheader
  if (FloatPhiInForBody) {
    PHINode *Phi = getFirstFloatPhi(ClonedForCondPreheader);
    assert(Phi && "Phi node not found");
    FloatPhiInForBody->insertBefore(Phi);
    Phi->replaceAllUsesWith(FloatPhiInForBody);
    Phi->eraseFromParent();
  }

  // // Set incomingblock 0 of FloatPhiInForBody to ForCondPreheader
  if (FloatPhiInForBody) {
    FloatPhiInForBody->setIncomingBlock(0, ForCondPreheader);
  }

  // Find float type Phi nodes in ForCondPreheader
  SmallVector<PHINode *> FloatPhisInForCondPreheader;
  for (auto &Phi : ForCondPreheader->phis()) {
    if (Phi.getType()->isFloatTy()) {
      FloatPhisInForCondPreheader.push_back(&Phi);
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
    int PhiCount = 0;
    for (PHINode &Phi : SecondSuccessor->phis()) {
      if (PhiCount == 1) { // Second Phi node
        // Set the second predecessor to ClonedForCondPreheader and its value to
        // addinst
        Phi.setIncomingBlock(1, ClonedForCondPreheader);
        Phi.setIncomingValue(1, addinst);
      } else {
        // For other Phi nodes, only update the predecessor basic block
        Phi.setIncomingBlock(1, ClonedForCondPreheader);
      }
      PhiCount++;
    }
  }
}

static void modifyClonedForBody2(BasicBlock *ClonedForBody,
                                 BasicBlock *ClonedForCondPreheader,
                                 Value *AddInst, BasicBlock *ForCondPreheader) {
  SmallVector<PHINode *> FloatPhiNodes;

  // Iterate through all instructions in ClonedForCondPreheader
  for (auto &Phi : ClonedForCondPreheader->phis()) {
    if (Phi.getType()->isFloatTy()) {
      FloatPhiNodes.push_back(&Phi);
      if (FloatPhiNodes.size() == 8) {
        break; // Stop after finding 8 float type PHI nodes
      }
    }
  }

  // Ensure we found 8 float type PHI nodes
  assert(FloatPhiNodes.size() == 8 &&
         "Unable to find 8 float type PHI nodes in ClonedForCondPreheader");

  // Now FloatPhiNodes contains 8 float type PHI nodes in order

  // Iterate through all PHI nodes in ClonedForBody
  int PhiIndex = 0;
  for (PHINode &Phi : ClonedForBody->phis()) {
    if (Phi.getType()->isFloatTy()) {
      // Ensure we don't access FloatPhiNodes out of bounds
      if (PhiIndex < FloatPhiNodes.size()) {
        // Set the 0th incoming value of the PHI node to the corresponding node
        // in FloatPhiNodes
        if (PhiIndex >
            0) { // Don't set the first Phi node, as It's floatPhiInForBody
          Phi.setIncomingValue(0, FloatPhiNodes[PhiIndex]);
        }
        PhiIndex++;
      } else {
        // If the number of float type PHI nodes in ClonedForBody exceeds the
        // size of FloatPhiNodes, output a warning
        assert(false && "Warning: Number of float type PHI nodes in "
                        "ClonedForBody exceeds expectations\n");
      }
    }
  }

  // Ensure we processed all expected PHI nodes
  if (PhiIndex < FloatPhiNodes.size()) {
    assert(false && "Warning: Number of float type PHI nodes in ClonedForBody "
                    "is less than expected\n");
  }

  // Find the last Icmp eq instruction in ClonedForBody
  ICmpInst *LastIcmpEq =
      getLastICmpInstWithPredicate(ClonedForBody, ICmpInst::ICMP_EQ);

  // Ensure we found the Icmp eq instruction
  assert(LastIcmpEq && "Unable to find Icmp eq instruction in ClonedForBody");

  // Set operand 1 to AddInst
  LastIcmpEq->setOperand(1, AddInst);
  // Change the predicate of the Icmp eq instruction to slt (signed less than)
  LastIcmpEq->setPredicate(ICmpInst::ICMP_SLT);
  // Change the name to Cmp
  LastIcmpEq->setName("Cmp");

  ClonedForBody->getTerminator()->setSuccessor(1, ForCondPreheader);

  // Find Phi i32 node in ClonedForBody
  PHINode *phiI32InClonedForBody = nullptr;
  for (auto &Phi : ClonedForBody->phis()) {
    if (Phi.getType()->isIntegerTy(32)) {
      phiI32InClonedForBody = &Phi;
      insertUnusedInstructionsBeforeIcmp(phiI32InClonedForBody, LastIcmpEq);
    }
  }

  // Ensure we found the Phi i32 node
  assert(phiI32InClonedForBody &&
         "Unable to find Phi i32 node in ClonedForBody");
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
  // ForBody to the FirstI32Phi found in ForCondPreheader
  firstI32PhiInForBody->setIncomingValue(0, firstI32PhiInForCondPreheader);
  secondI32PhiInForBody->setIncomingValue(0, secondI32PhiInForCondPreheader);

  ForBody->getTerminator()->setSuccessor(0, ClonedForCondPreheader);

  // Find the first float type PHI instruction in ForCondPreheader
  PHINode *SecondFloatPhiInForCondPreheader = nullptr;
  int FloatPhiCount = 0;
  for (auto &Phi : ForCondPreheader->phis()) {
    if (Phi.getType()->isFloatTy()) {
      FloatPhiCount++;
      if (FloatPhiCount == 2) {
        SecondFloatPhiInForCondPreheader = &Phi;
        break;
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

static bool modifySecondForBody(Loop *L, Function &F, BasicBlock *ForBody,
                                BasicBlock *FirstClonedForCondPreheader,
                                BasicBlock *FirstForCondPreheader,
                                Value *AddInst) {
  BasicBlock *ForBodyPreheader = L->getLoopPreheader();

  // Find the 0th successor of ForBody, It should have two
  BasicBlock *ForCondPreheader = ForBody->getTerminator()->getSuccessor(0);

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
  BasicBlock *Entry = &F.getEntryBlock();
  BasicBlock *ifend = &F.back();
  BasicBlock *entry_successor1 = Entry->getTerminator()->getSuccessor(1);

  // Create a new basic block
  BasicBlock *newBB = BasicBlock::Create(
      F.getContext(), entry_successor1->getName() + ".preheader", &F,
      entry_successor1);

  Value *Len = getLenFromEntryBlock(F);

  // Insert instructions in the new basic block
  IRBuilder<> builder(newBB);
  Value *cmp151349 = builder.CreateICmpSGT(
      Len, ConstantInt::get(Len->getType(), 0), "cmp151349");

  // Create a conditional branch
  builder.CreateCondBr(cmp151349, entry_successor1, ifend);

  // Modify the Terminator of Entry to jump to the new basic block
  Entry->getTerminator()->setSuccessor(1, newBB);
}

bool DspsF32FirLoopUnroller::unrollFir(Function &F, Loop *L) {

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
      Value *Sub = modifyForCondPreheader(F);
      std::tuple<BasicBlock *, BasicBlock *, Value *> result =
          modifyFirstForBody(L, F, BB, Sub);
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

// Preprocessing Function
static PHINode *preprocessClonedForBody(BasicBlock *ClonedForBody) {
  // Find the unique PHI node
  PHINode *PhiNode = getFirstInst<PHINode>(ClonedForBody);
  // Ensure that the PHI node is found
  assert(PhiNode && "PHI node not found");

  // Find two mul nsw instructions
  SmallVector<BinaryOperator *> MulInsts;
  for (auto &I : *ClonedForBody) {
    if (auto *BinOp = dyn_cast<BinaryOperator>(&I)) {
      if (BinOp->getOpcode() == Instruction::Mul && BinOp->hasNoSignedWrap()) {
        MulInsts.push_back(BinOp);
      }
    }
  }

  // Replace mul nsw instructions with the PHI node
  for (auto *mulInst : MulInsts) {
    mulInst->replaceAllUsesWith(PhiNode);
    mulInst->eraseFromParent();
  }
  return PhiNode;
}

static Instruction *modifyAddToOrInClonedForBody(BasicBlock *ClonedForBody) {
  // Find the unique add nuw nsw instruction
  Instruction *AddInst = nullptr;
  for (auto &I : *ClonedForBody) {
    if (auto *BinOp = dyn_cast<BinaryOperator>(&I)) {
      if (BinOp->getOpcode() == Instruction::Add &&
          BinOp->hasNoUnsignedWrap()) {
        AddInst = BinOp;
        break;
      }
    }
  }

  // Ensure that the add nuw nsw instruction is found
  assert(AddInst && "add nuw nsw instruction not found");

  // Create a new or disjoint instruction
  Instruction *OrInst = BinaryOperator::CreateDisjoint(
      Instruction::Or, AddInst->getOperand(0),
      ConstantInt::get(AddInst->getType(), 1), "add", AddInst);

  // Replace all uses of the add instruction
  AddInst->replaceAllUsesWith(OrInst);

  // Delete the original add instruction
  AddInst->eraseFromParent();
  OrInst->setName("add");
  return OrInst;
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

  // Create Function-level optimization pipeline
  FunctionPassManager FPM;
  FPM.addPass(InstCombinePass());
  FPM.run(F, FAM);
}

static Value *unrolladdcClonedForBody(BasicBlock *ClonedForBody,
                                      int UnrollFactor) {

  // Call the preprocessing Function
  PHINode *PhiNode = preprocessClonedForBody(ClonedForBody);

  // Replace add instructions with or instructions
  Instruction *OrInst = modifyAddToOrInClonedForBody(ClonedForBody);

  // Find the first non-PHI instruction and or instruction
  Instruction *FirstNonPHI = ClonedForBody->getFirstNonPHI();

  // Ensure that the start and end instructions are found
  assert(FirstNonPHI && OrInst && "Start or end instruction not found");

  // Find the Icmp instruction
  Instruction *icmpInst = getFirstInst<ICmpInst>(ClonedForBody);

  // Ensure that the Icmp instruction is found
  assert(icmpInst && "Icmp instruction not found");

  // Print information about the Icmp instruction

  Instruction *NewOrInst = OrInst;
  // Copy instructions 15 times
  for (int I = 1; I <= (UnrollFactor - 1); I++) {
    ValueToValueMapTy VMap;
    for (auto It = FirstNonPHI->getIterator(); &*It != OrInst; ++It) {
      Instruction *NewInst = It->clone();
      // For getelementptr instructions, set the second operand to OrInst
      if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(NewInst)) {
        NewInst->setOperand(1, NewOrInst);
        NewInst->setName("Arrayidx");
      }
      // If It's a fadd instruction, change its name to add
      if (NewInst->getOpcode() == Instruction::FAdd) {
        NewInst->setName("add");
      }
      VMap[&*It] = NewInst;
      NewInst->insertBefore(icmpInst);
    }

    // Update operands of new instructions
    for (auto It = FirstNonPHI->getIterator(); &*It != OrInst; ++It) {
      Instruction *NewInst = cast<Instruction>(VMap[&*It]);
      for (unsigned J = 0; J < NewInst->getNumOperands(); J++) {
        Value *Op = NewInst->getOperand(J);
        if (VMap.count(Op)) {
          NewInst->setOperand(J, VMap[Op]);
        }
      }
    }
    // Clone OrInst and insert before icmpInst
    NewOrInst = OrInst->clone();
    // Set the second operand of NewOrInst to I+1
    NewOrInst->setOperand(1, ConstantInt::get(NewOrInst->getType(), I + 1));
    NewOrInst->setName("add");
    NewOrInst->insertBefore(icmpInst);
    VMap[OrInst] = NewOrInst;
  }

  // Replace or instruction with add nuw nsw instruction
  IRBuilder<> Builder(NewOrInst);
  Value *newAddInst =
      Builder.CreateNUWAdd(NewOrInst->getOperand(0), NewOrInst->getOperand(1));
  NewOrInst->replaceAllUsesWith(newAddInst);
  NewOrInst->eraseFromParent();

  // Create a new add instruction, subtracting 16 from Len
  Builder.SetInsertPoint(icmpInst);
  Value *Len = icmpInst->getOperand(1);
  Value *Sub = Builder.CreateNSWAdd(
      Len, ConstantInt::getSigned(Len->getType(), -UnrollFactor), "Sub");
  // Set the Icmp instruction's predicate to sgt, and operands to newAddInst
  if (ICmpInst *Icmp = dyn_cast<ICmpInst>(icmpInst)) {
    Icmp->setPredicate(ICmpInst::ICMP_SGT);
    Icmp->setOperand(0, newAddInst);
    Icmp->setOperand(1, Sub);
  }

  PhiNode->setIncomingValue(0, newAddInst);
  return Sub;
}

static void expandForCondPreheaderaddc(Function &F,
                                       BasicBlock *ForCondPreheader,
                                       BasicBlock *ClonedForBody,
                                       BasicBlock *ForBody, Value *Sub,
                                       int UnrollFactor) {
  // Create a new ForCondPreheader after the original ForCondPreheader
  BasicBlock *NewForCondPreheader = BasicBlock::Create(
      ForCondPreheader->getContext(), "for.cond.preheader.new",
      ForCondPreheader->getParent(), ForCondPreheader->getNextNode());
  // Create a new empty BasicBlock after NewForCondPreheader
  BasicBlock *NewForCondPreheader2 = BasicBlock::Create(
      NewForCondPreheader->getContext(), "for.cond.preheader.new2",
      NewForCondPreheader->getParent(), NewForCondPreheader->getNextNode());

  // Move Sub to the new ForCondPreheader
  if (Instruction *SubInst = dyn_cast<Instruction>(Sub)) {
    SubInst->removeFromParent();
    SubInst->insertInto(NewForCondPreheader, NewForCondPreheader->begin());
  }

  // Create new comparison instruction in NewForCondPreheader
  IRBuilder<> Builder(NewForCondPreheader);
  Value *Len = getLenFromEntryBlock(F);

  assert(Len && "Parameter named 'Len' not found");

  Value *cmp6not207 = Builder.CreateICmpULT(
      Len, ConstantInt::get(Len->getType(), UnrollFactor), "cmp6.not207");

  // Create conditional branch instruction
  Builder.CreateCondBr(cmp6not207, NewForCondPreheader2, ClonedForBody);

  // Find if.end basic block
  BasicBlock *ifEndBB = getBasicBlockByName(F, "if.end");
  BasicBlock *returnBB = getBasicBlockByName(F, "return");
  assert(ifEndBB && "Expected to find if.end!");
  assert(returnBB && "Expected to find return!");
  // Get the Terminator instruction of if.end
  Instruction *Terminator = ifEndBB->getTerminator();
  if (!Terminator) {
    assert(false && "if.end basic block has no Terminator instruction\n");
    return;
  }

  // Replace the first operand of the Terminator instruction with
  // NewForCondPreheader
  Terminator->setOperand(2, NewForCondPreheader);

  // Find the unique PHINode in clonedForBody
  PHINode *UniquePHI = nullptr;
  for (auto &Phi : ClonedForBody->phis()) {
    if (UniquePHI) {
      // If we've already found a PHINode but find another, It's not unique
      UniquePHI = nullptr;
      break;
    }
    UniquePHI = &Phi;
  }

  assert(UniquePHI && "No unique PHINode found in ForBody\n");

  UniquePHI->setIncomingBlock(1, NewForCondPreheader);
  auto *clonedphi = UniquePHI->clone();
  clonedphi->insertInto(NewForCondPreheader2, NewForCondPreheader2->begin());

  // Create comparison instruction
  ICmpInst *cmp85209 =
      new ICmpInst(ICmpInst::ICMP_SLT, clonedphi, Len, "cmp85209");
  cmp85209->insertAfter(clonedphi);

  // Create conditional branch instruction
  BranchInst *Br = BranchInst::Create(ForBody, returnBB, cmp85209);

  Br->insertAfter(cmp85209);

  // Get the Terminator instruction of ClonedForBody
  BranchInst *clonedTerminator =
      dyn_cast<BranchInst>(ClonedForBody->getTerminator());
  assert(clonedTerminator &&
         "ClonedForBody's Terminator should be a BranchInst");
  if (!clonedTerminator) {
    assert(false && "ClonedForBody has no Terminator instruction\n");
    return;
  }

  // Set the first operand of ClonedForBody's Terminator to NewForCondPreheader2
  clonedTerminator->setOperand(2, NewForCondPreheader2);

  // Find the unique PHI node in ForBody
  PHINode *uniquePHI2 = nullptr;
  for (auto &Phi : ForBody->phis()) {
    if (uniquePHI2) {
      // If we've already found a PHINode but find another, It's not unique

      UniquePHI = nullptr;
      break;
    }
    uniquePHI2 = &Phi;
  }

  assert(uniquePHI2 && "No unique PHINode found in ForBody\n");

  uniquePHI2->setIncomingValue(1, clonedphi);
  uniquePHI2->setIncomingBlock(1, NewForCondPreheader2);

  // Find the unique PHI node in returnBB
  PHINode *returnBBPHI = nullptr;
  for (auto &Phi : returnBB->phis()) {
    if (returnBBPHI) {
      // If we've already found a PHINode but find another, It's not unique
      returnBBPHI = nullptr;
      break;
    }
    returnBBPHI = &Phi;
  }

  if (returnBBPHI) {
    // Add [0, NewForCondPreheader2]
    returnBBPHI->addIncoming(ConstantInt::get(returnBBPHI->getType(), 0),
                             NewForCondPreheader2);
  } else {
    assert(false && "No unique PHI node found in returnBB\n");
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

void DspsF32MathLoopUnroller::unrollMath(Function &F, ScalarEvolution &SE,
                                         Loop *L, int UnrollFactor) {

  // Get the basic block containing the Function body from L
  BasicBlock *ForBody = L->getHeader();

  // Ensure that the basic block containing the Function body is found
  assert(ForBody && "ForBody not found");

  // clone for body
  BasicBlock *ClonedForBody = cloneForBody(F, ForBody, ".modify");
  ClonedForBody->moveBefore(ForBody);

  Value *Sub = unrolladdcClonedForBody(ClonedForBody, UnrollFactor);

  // Find the ForCondPreheader basic block from F
  BasicBlock *ForCondPreheader = getBasicBlockByName(F, "for.cond.preheader");
  assert(ForCondPreheader && "Expected to find for.cond.preheader!");
  expandForCondPreheaderaddc(F, ForCondPreheader, ClonedForBody, ForBody, Sub,
                             UnrollFactor);
  runInstCombinePass(F);
  groupAndReorderInstructions(ClonedForBody);
}

bool DspsF32CorrLoopUnroller::unrollCorr(Function &F, Loop *L,
                                         int UnrollFactor) {

  // Get critical basic blocks
  BasicBlock *ForBody = L->getHeader();
  BasicBlock *returnBB = getBasicBlockByName(F, "return");
  BasicBlock *ForCondPreheader = getBasicBlockByName(F, "for.cond.preheader");
  BasicBlock *ForCond11PreheaderUs = L->getLoopPreheader();

  // Verify that basic blocks exist
  if (!ForBody || !returnBB || !ForCondPreheader || !ForCond11PreheaderUs) {
    report_fatal_error("Required basic blocks not found");
  }

  // clone for body
  BasicBlock *ClonedForBody = cloneForBody(F, ForBody, ".unroll");

  ClonedForBody->moveBefore(returnBB);

  ForCondPreheader->setName("if.end");

  // Find the first instruction in ForCondPreheader
  Instruction *FirstInst = &*ForCondPreheader->begin();
  Instruction *SecondInst = FirstInst->getNextNode();
  // Ensure the first instruction is a Sub nsw instruction
  if (BinaryOperator *SubInst = dyn_cast<BinaryOperator>(FirstInst)) {
    if (SubInst->getOpcode() == Instruction::Sub &&
        SubInst->hasNoSignedWrap()) {
      ;
    } else {
      assert(false && "The first instruction in ForCondPreheader is not a Sub "
                      "nsw instruction\n");
    }
  } else {
    assert(false && "The first instruction in ForCondPreheader is not a binary "
                    "operation\n");
  }
  // Insert new instruction after FirstInst
  IRBuilder<> Builder(FirstInst->getNextNode());
  Value *Sub6 = Builder.CreateNSWAdd(
      FirstInst, ConstantInt::getSigned(FirstInst->getType(), 1 - UnrollFactor),
      "sub6");

  if (ICmpInst *CmpInst = dyn_cast<ICmpInst>(SecondInst)) {
    if (CmpInst->getPredicate() == ICmpInst::ICMP_EQ) {
      CmpInst->setOperand(0, FirstInst);
      CmpInst->setOperand(
          1, ConstantInt::get(FirstInst->getType(), UnrollFactor - 1));
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

  // Find the parameter named patlen from the Function arguments
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

  Instruction *ForCond11PreheaderPreheaderFirstInst =
      &*ForCond11PreheaderPreheader->begin();
  Value *SiglenArg = ForCond11PreheaderPreheaderFirstInst->getOperand(0);
  // Calculate the result of n.0.lcssa left shifted by 2 bits
  Value *ShiftedN = Builder.CreateShl(
      N0Lcssa, ConstantInt::get(Type::getInt32Ty(F.getContext()), 2), "");

  // Create getelementptr instruction
  // Find memset Function call
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

  // Add %7 = Sub i32 %6, %patlen
  Value *SubResult2 = Builder.CreateSub(SiglenPlus2, PatlenArg, "");

  PHINode *PhiNode = getFirstInst<PHINode>(ForCond11PreheaderUs);

  assert(PhiNode && "PHI node not found in for.cond11.preheader.us\n");

  // Modify incoming values of the PHI node
  PhiNode->setIncomingBlock(1, ForCond11PreheaderUsPreheader);
  PhiNode->setIncomingValue(1, N0Lcssa);

  BasicBlock *ForCond11ForCondCleanup13CritEdgeUs = ForBody->getNextNode();
  // Find Icmp ult instruction in ForCond11ForCondCleanup13CritEdgeUs
  ICmpInst *IcmpUltInst = getLastICmpInstWithPredicate(
      ForCond11ForCondCleanup13CritEdgeUs, ICmpInst::ICMP_ULT);

  assert(IcmpUltInst && "Icmp ult instruction not found in "
                        "ForCond11ForCondCleanup13CritEdgeUs\n");

  IcmpUltInst->setOperand(0, PhiNode->getIncomingValue(0));
  IcmpUltInst->setOperand(1, SubResult2);
  IcmpUltInst->setPredicate(ICmpInst::ICMP_EQ);

  swapTerminatorSuccessors(ForCond11ForCondCleanup13CritEdgeUs);

  // Find PHI nodes in ClonedForBody
  setPHIIndexIncomingBlock(ClonedForBody, 0, ForBody10LrPh);
  // Find Phi float instruction in ClonedForBody
  PHINode *FloatPhi = getFirstFloatPhi(ClonedForBody);
  assert(FloatPhi && "Phi float node not found");
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

  Instruction *Loadinst = GEPInst->getNextNode();
  GEPInst->moveBefore(FloatPhi);
  Loadinst->moveBefore(FloatPhi);

  if (FloatPhi) {
    // Find the llvm.fmuladd.f32 instruction
    Instruction *FMulAdd = getFirstFMulAddInst(ClonedForBody);
    assert(FMulAdd && "llvm.fmuladd.f32 instruction not found\n");
    Instruction *InsertPoint = FMulAdd->getNextNode();
    if (FMulAdd) {
      // Copy instructions UnrollFactor-1 times
      for (int I = 0; I < (UnrollFactor - 1); ++I) {
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
          for (unsigned J = 0; J < NewInst->getNumOperands(); J++) {
            Value *Op = NewInst->getOperand(J);
            if (VMap.count(Op)) {
              NewInst->setOperand(J, VMap[Op]);
            }
          }
          // If NewInst is a getelementptr instruction, set its operand 1 to I+1
          if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(NewInst)) {
            GEP->setOperand(0, GEPInst);
            GEP->setOperand(
                1, ConstantInt::get(GEP->getOperand(1)->getType(), I + 1));
            GEP->setName("Arrayidx" + std::to_string(I + 1));
          }
        }
      }

    } else {
      assert(false && "llvm.fmuladd.f32 instruction not found\n");
    }
  } else {
    assert(false && "Phi float instruction not found\n");
  }
  movePHINodesToTop(*ClonedForBody);
  groupAndReorderInstructions(ClonedForBody);

  // Create new basic block for.cond.cleanup
  BasicBlock *ForCondCleanup =
      BasicBlock::Create(F.getContext(), "for.cond.cleanup", &F, ClonedForBody);

  ForCond8Preheader->getTerminator()->setSuccessor(1, ForCondCleanup);
  // Create unconditional branch to ClonedForBody in for.cond.cleanup
  BranchInst::Create(ClonedForBody, ForCondCleanup);

  // Get the Terminator instruction of ClonedForBody
  Instruction *Terminator = ClonedForBody->getTerminator();

  // Set the first successor of ClonedForBody to for.cond.cleanup
  if (Terminator->getNumSuccessors() > 0) {
    Terminator->setSuccessor(0, ForCondCleanup);
  }

  // Clone Phi float nodes from ClonedForBody to ForCondCleanup
  int I = 0;
  for (PHINode &Phi : ClonedForBody->phis()) {
    if (Phi.getType()->isFloatTy()) {
      Instruction *newPhi = Phi.clone();
      cast<PHINode>(newPhi)->setIncomingBlock(0, ForCond8Preheader);
      newPhi->insertBefore(ForCondCleanup->getTerminator());
      if (I == 0) {
        GetElementPtrInst *Arrayidx = GetElementPtrInst::Create(
            Type::getFloatTy(F.getContext()), DestArg, N0276, "Arrayidx",
            ForCondCleanup->getTerminator());

        new StoreInst(newPhi, Arrayidx, ForCondCleanup->getTerminator());
      } else {
        Instruction *OrInst = BinaryOperator::CreateDisjoint(
            Instruction::Or, N0276, ConstantInt::get(N0276->getType(), I),
            "add");
        OrInst->insertBefore(ForCondCleanup->getTerminator());
        GetElementPtrInst *Arrayidx = GetElementPtrInst::Create(
            Type::getFloatTy(F.getContext()), DestArg, OrInst, "Arrayidx",
            ForCondCleanup->getTerminator());

        new StoreInst(newPhi, Arrayidx, ForCondCleanup->getTerminator());
      }
      I++;
    }
  }

  // Insert new instructions at the end of ClonedForBody
  Builder.SetInsertPoint(ForCondCleanup->getTerminator());
  Value *add89 =
      Builder.CreateAdd(N0276, ConstantInt::get(N0276->getType(), UnrollFactor),
                        "add89", true, true);
  Value *cmp7 = Builder.CreateICmpSLT(add89, Sub6, "cmp7");

  // Get the original Terminator instruction
  Instruction *OldTerminator = ForCondCleanup->getTerminator();

  // Create new conditional branch instruction
  BranchInst *NewBr =
      BranchInst::Create(ForCond8Preheader, ForCond91Preheader, cmp7);

  // Insert new branch instruction and delete the old Terminator
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

  return true;
}

static void
insertPhiNodesForFMulAdd(BasicBlock *LoopHeader, BasicBlock *LoopPreheader,
                         SmallVector<CallInst *, 16> &FMulAddCalls) {
  // Collect all tail call float @llvm.fmuladd.f32 in LoopHeader
  for (Instruction &Inst : *LoopHeader) {
    if (CallInst *CI = dyn_cast<CallInst>(&Inst)) {
      if (Function *F = CI->getCalledFunction()) {
        if (F->getName() == "llvm.fmuladd.f32" && CI->isTailCall()) {
          FMulAddCalls.push_back(CI);
        }
      }
    }
  }

  // Insert Phi nodes for each FMulAdd call
  for (CallInst *CI : FMulAddCalls) {
    // Create new Phi node
    PHINode *PHI =
        PHINode::Create(CI->getType(), 2, CI->getName() + ".Phi", CI);

    // Set incoming values for Phi node
    PHI->addIncoming(ConstantFP::get(CI->getType(), 0), LoopPreheader);
    PHI->addIncoming(CI, LoopHeader);

    CI->setOperand(2, PHI);
  }
}

void DspsF32DotprodSimpleLoopUnroller::postUnrollLoopWithCount(
    Function &F, Loop *L, int unrollCount) {
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
                              (2 * unrollCount - 1)));
  LastICmp->setName("Cmp");

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

  // Verify Function
  if (verifyFunction(F, &errs())) {
    LLVM_DEBUG(errs() << "Function verification failed\n");
    return;
  }
}

static std::pair<Value *, Value *> modifyEntryBB(BasicBlock &EntryBB,
                                                 int unrollCount) {
  ICmpInst *Icmp = getLastInst<ICmpInst>(&EntryBB);
  assert(Icmp && "Icmp not found");
  Value *start_index = Icmp->getOperand(0);
  Value *end_index = Icmp->getOperand(1);
  // Insert new instructions before Icmp
  IRBuilder<> Builder(Icmp);
  Value *Sub = Builder.CreateNSWAdd(
      end_index, ConstantInt::getSigned(end_index->getType(), -unrollCount), "Sub");
  Icmp->setOperand(0, Sub);
  Icmp->setOperand(1, start_index);
  return std::make_pair(Sub, end_index);
}

void DspsF32DotprodComplexLoopUnroller::postUnrollLoopWithVariable(
    Function &F, Loop *L, int unrollCount) {
  BasicBlock *LoopPreheader = L->getLoopPreheader();
  BasicBlock *ForEnd = getBasicBlockByName(F, "for.cond.cleanup");
  assert(ForEnd && "basic block not found");
  ForEnd->setName("for.end");

  auto [LoopHeaderClone, ForBody7] = cloneAndMergeLoop(L, F, unrollCount);
  LoopHeaderClone->getTerminator()->setSuccessor(1, LoopHeaderClone);
  for (PHINode &Phi : LoopHeaderClone->phis()) {
    Phi.setIncomingBlock(1, LoopHeaderClone);
  }
  // Adjust positions
  LoopHeaderClone->moveAfter(getBasicBlockByName(F, "for.body.7"));
  assert(LoopHeaderClone && "basic block not found");
  ForEnd->moveAfter(LoopHeaderClone);

  BasicBlock &EntryBB = F.getEntryBlock();
  auto [Sub, end_index] = modifyEntryBB(EntryBB, unrollCount);
  EntryBB.getTerminator()->setSuccessor(1, ForBody7);

  SmallVector<Instruction *> FAMSDInsts;
  for (Instruction &Inst : *ForBody7) {
    if (auto *BinOp = dyn_cast<BinaryOperator>(&Inst)) {
      if (BinOp->getOpcode() == Instruction::FAdd ||
          BinOp->getOpcode() == Instruction::FMul ||
          BinOp->getOpcode() == Instruction::FSub ||
          BinOp->getOpcode() == Instruction::FDiv) {
        FAMSDInsts.push_back(BinOp);
      }
    }
  }
  assert(!FAMSDInsts.empty() && "fadd/fmul/fsub/fdiv instruction not found");
  PHINode *FirstFloatPhi = getFirstFloatPhi(ForBody7);
  assert(FirstFloatPhi && "Phi node not found");
  // Clone Phi node 7 times
  for (int I = 0; I < 7; I++) {
    PHINode *ClonedPhi = cast<PHINode>(FirstFloatPhi->clone());
    ClonedPhi->setName("result" + Twine(I));
    ClonedPhi->insertAfter(FirstFloatPhi);
    auto *Temp = FAMSDInsts[I];
    ClonedPhi->setIncomingValue(1, Temp);
    Temp->setOperand(0, ClonedPhi);
  }

  for (PHINode &Phi : ForBody7->phis()) {
    Phi.setIncomingBlock(0, &EntryBB);
    auto *Temp = Phi.clone();
    Temp->setName("result0.0.lcssa");
    Temp->insertBefore(LoopPreheader->getTerminator());
  }

  ICmpInst *LastICmp = getLastInst<ICmpInst>(ForBody7);
  assert(LastICmp && "Icmp not found");
  LastICmp->setOperand(1, Sub);
  LastICmp->setPredicate(ICmpInst::ICMP_SLT);

  ForBody7->getTerminator()->setSuccessor(0, LoopPreheader);
  ForBody7->getTerminator()->setSuccessor(1, ForBody7);

  PHINode *FirstI32Phi = getFirstI32Phi(LoopPreheader);
  assert(FirstI32Phi && "Phi node not found");
  // Insert Icmp slt instruction in LoopPreheader
  IRBuilder<> Builder(LoopPreheader->getTerminator());
  ICmpInst *NewICmp =
      cast<ICmpInst>(Builder.CreateICmpSLT(FirstI32Phi, end_index, "Cmp"));

  // Convert the original unconditional branch to a conditional branch
  BranchInst *OldBr = cast<BranchInst>(LoopPreheader->getTerminator());
  BranchInst *NewBr = BranchInst::Create(LoopHeaderClone, ForEnd, NewICmp);
  ReplaceInstWithInst(OldBr, NewBr);

  Instruction *FaddInst = nullptr;
  Instruction *AddNswInst = nullptr;

  for (auto &I : *LoopHeaderClone) {
    if (auto *BinOp = dyn_cast<BinaryOperator>(&I)) {
      if ((BinOp->getOpcode() == Instruction::FAdd ||
           BinOp->getOpcode() == Instruction::FMul ||
           BinOp->getOpcode() == Instruction::FSub ||
           BinOp->getOpcode() == Instruction::FDiv) &&
          BinOp->getType()->isFloatTy()) {
        FaddInst = BinOp;
      } else if (BinOp->getOpcode() == Instruction::Add &&
                 BinOp->hasNoSignedWrap()) {
        AddNswInst = BinOp;
      }
    }

    if (FaddInst && AddNswInst) {
      break;
    }
  }
  assert(FaddInst && AddNswInst &&
         "fadd/fmul/fsub/fdiv float and add nsw instructions not found");
  PHINode *FirstI32PhiLoopHeaderClone = getFirstI32Phi(LoopHeaderClone);
  assert(FirstI32PhiLoopHeaderClone && "Phi node not found");
  FirstI32PhiLoopHeaderClone->setIncomingValue(0, FirstI32Phi);
  FirstI32PhiLoopHeaderClone->setIncomingValue(1, AddNswInst);

  PHINode *FirstFloatPhiLoopHeaderClone = getFirstFloatPhi(LoopHeaderClone);
  assert(FirstFloatPhiLoopHeaderClone && "Phi node not found");
  PHINode *LastFloatPhiLoopPreheader = getLastFloatPhi(LoopPreheader);
  assert(LastFloatPhiLoopPreheader && "Phi node not found");
  FirstFloatPhiLoopHeaderClone->setIncomingValue(0, LastFloatPhiLoopPreheader);
  FirstFloatPhiLoopHeaderClone->setIncomingValue(1, FaddInst);

  // Collect all Phi float instructions in LoopPreheader
  SmallVector<PHINode *> FloatPhis;
  for (auto &Phi : LoopPreheader->phis()) {
    if (Phi.getType()->isFloatTy()) {
      FloatPhis.push_back(&Phi);
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
  assert(FloatPhis.size() == 8 && "expected FloatPhis has 8 Phi node");
  Value *CurrentSum = nullptr;
  Value *Add64 = Builder.CreateFAdd(FloatPhis[0], OriginalRetValue, "Add64");
  Value *Add65 = Builder.CreateFAdd(FloatPhis[1], FloatPhis[2], "Add65");
  Value *Add66 = Builder.CreateFAdd(FloatPhis[3], FloatPhis[4], "Add66");
  Value *Add67 = Builder.CreateFAdd(FloatPhis[5], FloatPhis[6], "Add67");
  Value *Add68 = Builder.CreateFAdd(Add64, Add65, "Add68");
  Value *Add69 = Builder.CreateFAdd(Add66, Add67, "Add69");
  CurrentSum = Builder.CreateFAdd(Add68, Add69, "add70");

  // Replace the original ret instruction
  RetInst->setOperand(0, CurrentSum);
  PHINode *FirstFloatPhiForEnd = getFirstFloatPhi(ForEnd);
  assert(FirstFloatPhiForEnd && "Phi node not found");
  // Remove existing incoming values from FirstFloatPhiForEnd
  while (FirstFloatPhiForEnd->getNumIncomingValues() > 0) {
    FirstFloatPhiForEnd->removeIncomingValue(0u, false);
  }
  // Add two incoming values to FirstFloatPhiForEnd
  FirstFloatPhiForEnd->addIncoming(FaddInst, LoopHeaderClone);
  FirstFloatPhiForEnd->addIncoming(LastFloatPhiLoopPreheader, LoopPreheader);

  runDeadCodeElimination(F);
}

static void eraseAllStoreInstInBB(BasicBlock *BB) {
  assert(BB && "BasicBlock is nullptr");
  // Erase all store instructions in BB
  for (auto It = BB->begin(); It != BB->end();) {
    if (isa<StoreInst>(&*It)) {
      It = It->eraseFromParent();
    } else {
      ++It;
    }
  }
}

static GetElementPtrInst *getUniqueGetElementPtrInst(BasicBlock *BB) {
  assert(BB && "BasicBlock is nullptr");
  // Get the unique getelementptr instruction in BB
  GetElementPtrInst *GEP = nullptr;
  for (Instruction &Inst : *BB) {
    if (auto *GEPI = dyn_cast<GetElementPtrInst>(&Inst)) {
      if (!GEP) {
        GEP = GEPI;
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

  // Update the Terminator instruction of CloneForBody
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
  PHINode *Phi = getLastInst<PHINode>(BB);
  // Add new instructions
  IRBuilder<> Builder(BB);
  Builder.SetInsertPoint(Phi->getNextNode());

  // and i32 %n.0551, -8
  Value *Add2 = Builder.CreateAnd(Phi, ConstantInt::getSigned(Phi->getType(), -8));

  // %Sub = and i32 %n.0551, 2147483644
  Value *Sub =
      Builder.CreateAnd(Phi, ConstantInt::get(Phi->getType(), 2147483640));

  // %cmp12538.not = Icmp eq i32 %Sub, 0
  Value *Cmp = Builder.CreateICmpEQ(Sub, ConstantInt::get(Phi->getType(), 0));

  // Br i1 %cmp12538.not, label %for.cond.cleanup, label %for.body.preheader
  // Move the conditional branch instruction to the end of BB
  auto *NewcondBr =
      Builder.CreateCondBr(Cmp, CloneForBodyPreheader, ForBodyMerged);

  // Erase the Terminator instruction of BB
  Instruction *OldTerminator = BB->getTerminator();
  NewcondBr->moveAfter(OldTerminator);
  OldTerminator->eraseFromParent();

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
  Value *Sum1 = BinaryOperator::CreateFAdd(FMulAddCalls[0], FMulAddCalls[1],
                                           "sum", NewForEnd);
  Value *Sum23 = BinaryOperator::CreateFAdd(FMulAddCalls[2], FMulAddCalls[3],
                                            "Sum23", NewForEnd);
  Value *sum24 = BinaryOperator::CreateFAdd(FMulAddCalls[4], FMulAddCalls[5],
                                            "sum24", NewForEnd);
  Value *sum25 = BinaryOperator::CreateFAdd(FMulAddCalls[6], FMulAddCalls[7],
                                            "sum25", NewForEnd);
  Value *sum26 = BinaryOperator::CreateFAdd(Sum1, Sum23, "sum26", NewForEnd);
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
  PHINode *SumPHI = PHINode::Create(Sum->getType(), 2, "sum.Phi",
                                    CloneForBodyPreheader->getFirstNonPHI());

  // Set the incoming values of the PHI node
  SumPHI->addIncoming(ConstantFP::get(Sum->getType(), 0.0), OuterBB);
  SumPHI->addIncoming(Sum, NewForEnd);

  // Create a PHI node in CloneForBodyPreheader
  PHINode *AddPHI = PHINode::Create(Add2->getType(), 2, "add.Phi",
                                    CloneForBodyPreheader->getFirstNonPHI());

  // Set the incoming values of the PHI node
  AddPHI->addIncoming(ConstantInt::get(Add2->getType(), 0), OuterBB);
  AddPHI->addIncoming(Add2, NewForEnd);
  Instruction *phifloatincomingvalue0 = getFirstFMulAddInst(CloneForBody);
  Value *phii32incomingvalue0 =
      getLastInst<ICmpInst>(CloneForBody)->getOperand(0);
  for (PHINode &Phi : CloneForBody->phis()) {
    if (Phi.getType()->isIntegerTy(32)) {
      Phi.setIncomingValue(0, AddPHI);
      Phi.setIncomingValue(1, phii32incomingvalue0);
    } else if (Phi.getType()->isFloatTy()) {
      Phi.setIncomingValue(0, SumPHI);
      Phi.setIncomingValue(1, phifloatincomingvalue0);
    }
  }
  setPHINodesBlock(CloneForBody, CloneForBodyPreheader, CloneForBody);
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
      sub206, ConstantInt::getSigned(sub206->getType(), -1), "add207.neg");

  // %add211 = add i32 %lsig.0, %add207.neg
  Value *Add211 = Builder.CreateAdd(lsig_0, Add207Neg, "add211");

  // %div212535 = and i32 %add211, -8
  Value *Div212535 = Builder.CreateAnd(
      Add211, ConstantInt::getSigned(Add211->getType(), -8), "div212535");

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
  ICmpInst *LastICmp = getLastInst<ICmpInst>(BB);
  LastICmp->setPredicate(ICmpInst::ICMP_ULT);
  swapTerminatorSuccessors(BB);

  eraseAllStoreInstInBB(BB);
  Value *lkern_0 = getFirstI32Phi(BB)->getIncomingValue(1);

  BasicBlock *BBLoopPreHeader = L->getLoopPreheader();
  // Insert an and instruction in BBLoopPreHeader
  IRBuilder<> Builder(BBLoopPreHeader->getTerminator());
  Value *Div536 = Builder.CreateAnd(lkern_0, ConstantInt::getSigned(lkern_0->getType(), -16), "div536");
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
                            GetElementPtrInst *GEP, uint32_t unrollCount) {
  assert((unrollCount == 8 || unrollCount == 16) &&
         "unrollCount must be 8 or 16");
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
  if (unrollCount == 16) {
    Value *Sum45 =
        BinaryOperator::CreateFAdd(FMulAddCalls[0], FMulAddCalls[1], "Sum45",
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

    Value *sum53 = BinaryOperator::CreateFAdd(Sum45, sum46, "sum53",
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
  } else if (unrollCount == 8) {
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

  // Set the target of the Terminator instruction of ForEndLoopExit to
  // for.end164
  Instruction *Terminator = ForEndLoopExit->getTerminator();
  BasicBlock *OldSuccessor = Terminator->getSuccessor(0);
  Terminator->setSuccessor(0, ForEnd164);

  // Create an unconditional branch instruction in for.end164, jumping to the
  // original successor basic block
  BranchInst::Create(OldSuccessor, ForEnd164);

  // Create a new Phi node in for.end164
  PHINode *PhiSum = PHINode::Create(Type::getInt32Ty(ForEnd164->getContext()),
                                    2, "Phi.sum", ForEnd164->getFirstNonPHI());

  // Set the incoming values of the Phi node
  PhiSum->addIncoming(Add56, OuterBB);
  PhiSum->addIncoming(LastICmp->getOperand(0), ForEndLoopExit);

  // Create a new Phi float node in for.end164
  PHINode *PhiFloat =
      PHINode::Create(Type::getFloatTy(ForEnd164->getContext()), 2, "Phi.float",
                      ForEnd164->getFirstNonPHI());

  // Set the incoming values of the Phi node
  PhiFloat->addIncoming(
      ConstantFP::get(Type::getFloatTy(ForEnd164->getContext()), 0.0), OuterBB);
  PhiFloat->addIncoming(Sum, ForEndLoopExit);
  // Create a new StoreInst instruction in for.end164
  new StoreInst(PhiFloat, GEP, ForEnd164->getTerminator());

  Value *Operand1 = unrollCount == 16
                        ? getFirstI32Phi(OuterBB)
                        : getLastInst<ICmpInst>(CloneForBody)->getOperand(1);
  // Create a new comparison instruction
  ICmpInst *NewCmp =
      new ICmpInst(ICmpInst::ICMP_UGT, PhiSum, Operand1, "cmp182.not587");
  NewCmp->insertBefore(ForEnd164->getTerminator());

  // Replace the original unconditional branch with a conditional branch
  BranchInst *OldBr = cast<BranchInst>(ForEnd164->getTerminator());
  BasicBlock *ForEnd37 = OldBr->getSuccessor(0);
  BranchInst *NewBr = BranchInst::Create(ForEnd37, CloneForBody, NewCmp);
  ReplaceInstWithInst(OldBr, NewBr);

  CloneForBody->moveAfter(ForEnd164);
  Instruction *TargetInst = getFirstFMulAddInst(CloneForBody);
  for (PHINode &Phi : CloneForBody->phis()) {
    if (Phi.getType()->isIntegerTy(32)) {
      Phi.setIncomingValue(0,
                           getLastInst<ICmpInst>(CloneForBody)->getOperand(0));
      Phi.setIncomingValue(1, PhiSum);
    } else if (Phi.getType()->isFloatTy()) {
      Phi.setIncomingValue(0, TargetInst);
      Phi.setIncomingValue(1, PhiFloat);
    }
  }
  setPHINodesBlock(CloneForBody, CloneForBody, ForEnd164);
  createCriticalEdgeAndMoveStoreInst(CloneForBody, ForEnd37);

  OuterBB->getTerminator()->setSuccessor(1, ForEnd164);
}

void DspsF32ConvCcorrUnroller::PostUnrollConv(Function &F, Loop *L,
                                              int unrollCount,
                                              int unroll_index) {

  auto [CloneForBody, ForBodyMerged] = cloneAndMergeLoop(L, F, unrollCount);
  // Get the outer loop of L
  Loop *OuterLoop = L->getParentLoop();
  if (unrollCount == 8 && unroll_index == 0) {
    BasicBlock *CloneForBodyPreheader = BasicBlock::Create(
        CloneForBody->getContext(), CloneForBody->getName() + ".preheader",
        CloneForBody->getParent(), CloneForBody);

    updatePredecessorsToPreheader(CloneForBody, CloneForBodyPreheader);
    auto [Sub, GEP, Add2] =
        modifyOuterLoop4(OuterLoop, ForBodyMerged, CloneForBodyPreheader);
    modifyInnerLoop4(OuterLoop, ForBodyMerged, Sub, CloneForBody, GEP, Add2,
                     CloneForBodyPreheader);
  } else if (unrollCount == 16) {
    auto [Add60, Add56, GEP] = modifyOuterLoop16(OuterLoop);
    modifyInnerLoop(OuterLoop, ForBodyMerged, Add60, CloneForBody, Add56, GEP,
                    unrollCount);
  } else if (unrollCount == 8) {
    auto [Add214, Add207, GEP] = modifyOuterLoop8(OuterLoop);
    modifyInnerLoop(OuterLoop, ForBodyMerged, Add214, CloneForBody, Add207, GEP,
                    unrollCount);
  }
  LLVM_DEBUG(F.dump());
}

static void modifyFirstCloneForBody(BasicBlock *CloneForBody,
                                    PHINode *N_0_lcssa,
                                    BasicBlock *ForBody27LrPh,
                                    PHINode *CoeffPosLcssa, Value *Operand1) {
  CloneForBody->getTerminator()->setSuccessor(1, CloneForBody);
  setPHINodesBlock(CloneForBody, ForBody27LrPh, CloneForBody);
  PHINode *FirstI32Phi = getFirstI32Phi(CloneForBody);
  PHINode *LastI32Phi = getLastI32Phi(CloneForBody);
  FirstI32Phi->setIncomingValue(0, N_0_lcssa);
  FirstI32Phi->setIncomingBlock(0, ForBody27LrPh);

  Instruction *FirstAddInst = nullptr;
  Instruction *LastAddInst = nullptr;
  for (Instruction &Inst : *CloneForBody) {
    if (Inst.getOpcode() == Instruction::Add) {
      if (!FirstAddInst) {
        FirstAddInst = &Inst;
      }
      LastAddInst = &Inst;
    }
  }
  ICmpInst *LastCmpInst = getLastInst<ICmpInst>(CloneForBody);
  LastCmpInst->setOperand(0, LastAddInst);
  LastCmpInst->setOperand(1, Operand1);
  FirstI32Phi->setIncomingValue(1, LastAddInst);

  LastI32Phi->setIncomingValue(0, CoeffPosLcssa);
  LastI32Phi->setIncomingBlock(0, ForBody27LrPh);

  LastI32Phi->setIncomingValue(1, FirstAddInst);
}

static bool setBBFromOtherBB(Function &F, StringRef BBName,
                             BasicBlock *ForBodyMerged) {
  // Find the first and last load instructions in ForBody27LrPh
  LoadInst *FirstLoad = nullptr;
  LoadInst *LastLoad = nullptr;
  BasicBlock *ForBody27LrPh = getBasicBlockByName(F, BBName);
  for (Instruction &Inst : *ForBody27LrPh) {
    if (auto *LI = dyn_cast<LoadInst>(&Inst)) {
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
  for (Instruction &Inst : *ForBodyMerged) {
    if (auto *GEP = dyn_cast<GetElementPtrInst>(&Inst)) {
      GEPInsts.push_back(GEP);
    }
  }
  // Ensure there is at least one GEP instruction
  if (!GEPInsts.empty()) {
    for (size_t I = 0; I < GEPInsts.size(); ++I) {
      GetElementPtrInst *CurrentGEP = GEPInsts[I];

      if (I % 2 == 1) { // Odd
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
void DspsF32FirdLoopUnroller::modifyFirdFirstLoop(Function &F, Loop *L,
                                                  BasicBlock *ForBodyMerged,
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
  // Create new Phi node at the beginning of ForBodyMerged
  PHINode *Add281 = PHINode::Create(Type::getInt32Ty(F.getContext()), 2,
                                    "add281", &ForBodyMerged->front());

  // Set incoming values for Phi node
  Add281->addIncoming(Add269, ForBodyMergedLoopPreheader);
  Add281->addIncoming(Inc20_7, ForBodyMerged);

  N_069->setIncomingValue(1, Add281);

  ICmpInst *LastICmpInPreheader = getLastInst<ICmpInst>(ForCond23Preheader);
  // Create new Phi node
  PHINode *N_0_lcssa = PHINode::Create(Type::getInt32Ty(F.getContext()), 2,
                                       "n.0.lcssa", LastICmpInPreheader);

  // Set incoming values for Phi node
  N_0_lcssa->addIncoming(FirstI32Phi, ForCondCleanup3);
  N_0_lcssa->addIncoming(Add281, ForBodyMerged);

  // Replace operand of LastICmpInPreheader with new Phi node
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
  Builder.CreateAdd(Operand1, CoeffPosLcssa);

  ForBody27LrPh->getTerminator()->setSuccessor(0, CloneForBody);
  ICmpInst *LastICmpInForBodyMerged = getLastInst<ICmpInst>(ForBodyMerged);
  LastICmpInForBodyMerged->setOperand(1, Operand1);
  LastICmpInForBodyMerged->setOperand(0, Inc20_7);

  modifyFirstCloneForBody(CloneForBody, N_0_lcssa, ForBody27LrPh, CoeffPosLcssa,
                          Operand1);

  PHINode *Acc_0_lcssa = getFirstFloatPhi(ForCond23Preheader);
  BasicBlock *ForCond23PreheaderLoopExit = Acc_0_lcssa->getIncomingBlock(1);
  PHINode *_lcssa = getFirstFloatPhi(ForCond23PreheaderLoopExit);
  Acc_0_lcssa->setIncomingValue(1, _lcssa->getIncomingValue(0));
  Acc_0_lcssa->setIncomingBlock(1, _lcssa->getIncomingBlock(0));

  Value *floatZero = Acc_0_lcssa->getIncomingValue(0);

  // Get all incoming values and blocks for PHINode
  for (unsigned I = 1; I < _lcssa->getNumIncomingValues(); ++I) {
    Value *IncomingValue = _lcssa->getIncomingValue(I);
    BasicBlock *IncomingBlock = _lcssa->getIncomingBlock(I);

    // Create new Phi node in ForCond23Preheader
    PHINode *NewPhi =
        PHINode::Create(floatZero->getType(), 2,
                        "acc." + std::to_string(I) + ".lcssa", CoeffPosLcssa);
    // Add incoming values
    NewPhi->addIncoming(floatZero, ForCondCleanup3);
    NewPhi->addIncoming(IncomingValue, IncomingBlock);
  }
  Value *coeff_pos_068 = getLastI32Phi(ForBodyMerged)->getIncomingValue(1);
  CoeffPosLcssa->setIncomingValue(1, coeff_pos_068);

  getLastFloatPhi(CloneForBody)->setIncomingValue(0, Acc_0_lcssa);

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
  for (Instruction &Inst : *ForBodyMerged) {
    if (CallInst *CI = dyn_cast<CallInst>(&Inst)) {
      if (Function *F = CI->getCalledFunction()) {
        if (F->getName() == "llvm.fmuladd.f32" && CI->isTailCall()) {
          FMulAddCalls.push_back(CI);
        }
      }
    }
  }

  // Insert Phi nodes for each FMulAdd call
  for (CallInst *CI : FMulAddCalls) {
    // Create new Phi node
    PHINode *PHI = PHINode::Create(CI->getType(), 2, CI->getName() + "acc", CI);

    // Set incoming values for Phi node
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
  for (Instruction &Inst : *ForBody14LrPh) {
    if (auto *LI = dyn_cast<LoadInst>(&Inst)) {
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
  for (Instruction &Inst : *ForBodyMerged) {
    if (auto *GEP = dyn_cast<GetElementPtrInst>(&Inst)) {
      GEPInsts.push_back(GEP);
    }
  }
  // Ensure at least one getelementptr instruction exists
  if (!GEPInsts.empty()) {
    for (size_t I = 0; I < GEPInsts.size(); ++I) {
      GetElementPtrInst *CurrentGEP = GEPInsts[I];

      if (I % 2 == 1) { // Odd
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

    // Starting from Index 1, process every other getelementptr
    for (size_t I = 3; I < GEPInsts.size(); I += 2) {
      GetElementPtrInst *CurrentGEP = GEPInsts[I];

      // Set current getelementptr's operand 0 to first getelementptr's value
      CurrentGEP->setOperand(0, SecondGEP);

      // Set operand 1 to current Index value
      // ConstantInt *IndexValue =
      // ConstantInt::get(CurrentGEP->getOperand(1)->getType(), I);
      CurrentGEP->setOperand(
          1, ConstantInt::get(CurrentGEP->getOperand(1)->getType(), (I) / 2));
    }
  }

  setBBFromOtherBB(F, "for.body27.lr.ph", CloneForBody);

  BasicBlock *ForCondCleanup26LoopExit = CloneForBody->getNextNode();
  BasicBlock *ForCondCleanup26 = ForCondCleanup26LoopExit->getSingleSuccessor();
  Instruction *tailcallInst = getFirstFMulAddInst(CloneForBody);

  // Find add instruction in ForBody27LrPh
  Instruction *AddInst = nullptr;
  for (Instruction &Inst : *ForBody27LrPh) {
    if (Inst.getOpcode() == Instruction::Add) {
      AddInst = &Inst;
      break;
    }
  }

  // Insert new instructions in ForCondCleanup26LoopExit
  Builder.SetInsertPoint(ForCondCleanup26LoopExit->getFirstNonPHI());
  Value *SubResult = Builder.CreateSub(AddInst, N_0_lcssa);
  PHINode *FirstFloatPhi = getFirstFloatPhi(ForCondCleanup26);
  FirstFloatPhi->setIncomingValue(1, tailcallInst);

  ForCond23Preheader->setName("for.cond63.preheader");
  // Create new PHI node in ForCondCleanup26
  PHINode *CoeffPosLcssaPhi =
      PHINode::Create(CoeffPosLcssa->getType(), 2, "coeff_pos.1.lcssa",
                      &ForCondCleanup26->front());

  // Set incoming values and blocks for PHI node
  CoeffPosLcssaPhi->addIncoming(CoeffPosLcssa, ForCond23Preheader);
  CoeffPosLcssaPhi->addIncoming(SubResult, ForCondCleanup26LoopExit);

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

  setPHIIndexIncomingBlock(ForCond130Preheader, 0, ForCondCleanup26);
  // Iterate through Phi nodes in ForCond130Preheader and ForCond23Preheader
  // simultaneously
  auto It130 = ForCond130Preheader->begin();
  auto It23 = ForCond23Preheader->begin();

  while (It130 != ForCond130Preheader->end() &&
         It23 != ForCond23Preheader->end()) {
    if (auto *Phi130 = dyn_cast<PHINode>(&*It130)) {
      if (auto *Phi23 = dyn_cast<PHINode>(&*It23)) {
        if (Phi130->getType()->isFloatTy() && Phi23->getType()->isFloatTy()) {
          // Write Phi float from ForCond23Preheader to incomingvalue 0 position
          // in ForCond130Preheader
          Phi130->setIncomingValue(0, Phi23);
        }
      }
      ++It23;
    }
    ++It130;
  }
  getFirstFloatPhi(ForCond130Preheader)->setIncomingValue(0, FirstFloatPhi);

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

static bool copyFloatPhiIncomingValue(int I, BasicBlock *srcBB,
                                      BasicBlock *tarBB) {
  assert(srcBB && tarBB && "srcBB or tarBB should not be nullptr");
  // Collect Phi float nodes from ForCond130Preheader in reverse order into
  // vector
  SmallVector<Value *, 8> FloatPhis;

  for (auto It = srcBB->rbegin(); It != srcBB->rend(); ++It) {
    if (PHINode *Phi = dyn_cast<PHINode>(&*It)) {
      if (Phi->getType()->isFloatTy()) {
        FloatPhis.push_back(Phi->getIncomingValue(I));
      }
    }
  }

  // Traverse Phi float nodes in ForBodyMerged in reverse order and store values
  // from FloatPhis into their incoming value 0
  auto FloatPhiIt = FloatPhis.begin();
  for (auto It = tarBB->rbegin();
       It != tarBB->rend() && FloatPhiIt != FloatPhis.end(); ++It) {
    if (PHINode *Phi = dyn_cast<PHINode>(&*It)) {
      if (Phi->getType()->isFloatTy()) {
        Phi->setIncomingValue(I, *FloatPhiIt);
        ++FloatPhiIt;
      }
    }
  }
  return true;
}

void DspsF32FirdLoopUnroller::modifyFirdSecondLoop(Function &F, Loop *L,
                                                   BasicBlock *ForBodyMerged,
                                                   BasicBlock *CloneForBody) {

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
  for (Instruction &Inst : *ForBodyMerged) {
    if (CallInst *CI = dyn_cast<CallInst>(&Inst)) {
      if (Function *F = CI->getCalledFunction()) {
        if (F->getName() == "llvm.fmuladd.f32" && CI->isTailCall()) {
          FMulAddCalls.push_back(CI);
        }
      }
    }
  }

  // Insert Phi nodes for each FMulAdd call
  for (CallInst *CI : FMulAddCalls) {
    // Create new Phi node
    PHINode *PHI = PHINode::Create(CI->getType(), 2, CI->getName() + "acc", CI);

    // Set incoming values for Phi node
    PHI->addIncoming(ConstantFP::get(CI->getType(), 0), PredBB);
    PHI->addIncoming(CI, ForBodyMerged);

    CI->setOperand(2, PHI);
  }
  PHINode *N22_075 = getFirstI32Phi(ForBodyMerged);
  // Create new Phi node in ForBodyMerged
  PHINode *Add76310 = PHINode::Create(Type::getInt32Ty(F.getContext()), 2,
                                      "add76310", &ForBodyMerged->front());
  Add76310->addIncoming(ConstantInt::get(Type::getInt32Ty(F.getContext()), 8),
                        ForBody133LrPh);
  N22_075->setIncomingValue(1, Add76310);
  // Create new add instruction in ForBodyMerged
  IRBuilder<> Builder(ForBodyMerged->getTerminator());
  Value *Add76 = Builder.CreateAdd(
      Add76310, ConstantInt::get(Type::getInt32Ty(F.getContext()), 8), "add76",
      true, true);

  // Update Phi node's loop edge
  Add76310->addIncoming(Add76, ForBodyMerged);

  movePHINodesToTop(*ForBodyMerged);
  modifyFirdAddToOr(ForBodyMerged);
  ICmpInst *LastICmp = getLastInst<ICmpInst>(ForBodyMerged);
  LastICmp->setPredicate(ICmpInst::ICMP_SGT);
  cast<Instruction>(Add76)->moveBefore(LastICmp);
  LastICmp->setOperand(0, Add76);

  setPHIIndexIncomingBlock(ForBodyMerged, 0, PredBB);

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

  // Find Len parameter in Function F
  Value *LenArg = getLenFromEntryBlock(F);
  assert(LenArg && "LenArg should be");

  // Create comparison instruction
  Value *ExitCond350 = Builder.CreateICmpEQ(Inc152, LenArg, "exitcond350.not");

  // Create conditional branch instruction
  Builder.CreateCondBr(ExitCond350, ForCondCleanup, ForCond1Preheader);

  BasicBlock *ForCond130Preheader =
      getBasicBlockByName(F, "for.cond130.preheader");
  for (PHINode &Phi : ForCond130Preheader->phis()) {
    Phi.setIncomingBlock(1, ForBodyMerged);
  }
  ForCond130Preheader->getTerminator()->setSuccessor(0, ForBody133LrPh);
  ForCond130Preheader->getTerminator()->setSuccessor(1, NewForEnd141);

  // ForBody133LrPh
  // Create new instructions in ForBody133LrPh
  BasicBlock *ForBody79LrPh = getBasicBlockByName(F, "for.body79.lr.ph");
  ForBody79LrPh->getTerminator()->setSuccessor(0, ForBodyMerged);
  // Copy Loadinst from ForBody79LrPh to ForBody133LrPh
  Builder.SetInsertPoint(ForBody133LrPh->getTerminator());
  for (Instruction &Inst : *ForBody79LrPh) {
    if (isa<LoadInst>(Inst)) {
      Instruction *ClonedInst = Inst.clone();
      ClonedInst->setName(Inst.getName());
      Builder.Insert(ClonedInst);
    }
  }

  // modify ForBodyMerged
  setPHIIndexIncomingBlock(ForBodyMerged, 0, ForBody79LrPh);

  PHINode *Coeff_pos174 = getLastI32Phi(ForBodyMerged);
  PHINode *Coeff_pos_0_lcssa_clone = getFirstI32Phi(ForCond130Preheader);
  Coeff_pos_0_lcssa_clone->setIncomingValue(1,
                                            Coeff_pos174->getIncomingValue(1));
  Coeff_pos174->setIncomingValue(0,
                                 Coeff_pos_0_lcssa_clone->getIncomingValue(0));

  bool Res = copyFloatPhiIncomingValue(0, ForCond130Preheader, ForBodyMerged);
  assert(Res && "copyFloatPhiIncomingZeroValue failed");

  bool Res1 = copyFloatPhiIncomingValue(1, ForBodyMerged, ForCond130Preheader);
  assert(Res1 && "copyFloatPhiIncomingValue failed");
  // Find first and last load instructions in ForBody79LrPh
  LoadInst *FirstLoad = nullptr;
  LoadInst *LastLoad = nullptr;

  for (Instruction &Inst : *ForBody79LrPh) {
    if (auto *LI = dyn_cast<LoadInst>(&Inst)) {
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
  for (Instruction &Inst : *ForBodyMerged) {
    if (auto *GEP = dyn_cast<GetElementPtrInst>(&Inst)) {
      GEPInsts.push_back(GEP);
    }
  }

  // Ensure there is at least one getelementptr instruction
  if (!GEPInsts.empty()) {
    for (size_t I = 0; I < GEPInsts.size(); ++I) {
      GetElementPtrInst *CurrentGEP = GEPInsts[I];

      if (I % 2 == 1) { // odd
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

    // Starting from Index 1, process every other getelementptr
    for (size_t I = 2; I < GEPInsts.size(); I += 2) {
      GetElementPtrInst *CurrentGEP = GEPInsts[I];

      // Set current getelementptr's operand 0 to first getelementptr's value
      CurrentGEP->setOperand(0, FirstGEP);

      // Set operand 1 to current Index value
      CurrentGEP->setOperand(
          1, ConstantInt::get(CurrentGEP->getOperand(1)->getType(), (I) / 2));
    }
  }

  ForBodyMerged->getTerminator()->setSuccessor(0, ForCond130Preheader);

  // modify for.body27.clone
  PHINode *N_0_lcssa_clone = getLastI32Phi(ForCond130Preheader);
  PHINode *acc_0_lcssa_clone = getFirstFloatPhi(ForCond130Preheader);
  Instruction *tailcallInst = getFirstFMulAddInst(CloneForBody);
  Instruction *FirstAddInst = nullptr;
  Instruction *LastAddInst = nullptr;
  for (Instruction &Inst : *CloneForBody) {
    if (Inst.getOpcode() == Instruction::Add) {
      if (!FirstAddInst) {
        FirstAddInst = &Inst;
      }
      LastAddInst = &Inst;
    }
  }
  int Index = 0;
  for (PHINode &Phi : CloneForBody->phis()) {
    if (Index == 0) {
      Phi.setIncomingValue(0, N_0_lcssa_clone);
      Phi.setIncomingValue(1, LastAddInst);
    } else if (Index == 1) {
      Phi.setIncomingValue(0, Coeff_pos_0_lcssa_clone);
      Phi.setIncomingValue(1, FirstAddInst);
    } else if (Index == 2) {
      Phi.setIncomingValue(0, acc_0_lcssa_clone);
      Phi.setIncomingValue(1, tailcallInst);
    }
    Index++;
  }
  setPHINodesBlock(CloneForBody, ForBody133LrPh, CloneForBody);

  CloneForBody->getTerminator()->setSuccessor(0, NewForEnd141);
  CloneForBody->getTerminator()->setSuccessor(1, CloneForBody);

  // modify for.end141
  // Create Phi float node in NewForEnd141
  PHINode *AccPhi = PHINode::Create(Type::getFloatTy(F.getContext()), 2,
                                    "acc0.3.lcssa", &NewForEnd141->front());
  AccPhi->addIncoming(acc_0_lcssa_clone, ForCond130Preheader);
  AccPhi->addIncoming(tailcallInst, CloneForBody);

  Value *Sum = nullptr;
  Instruction *InsertPoint = AccPhi->getNextNode();
  // Count the number of float type Phi nodes in ForCond130Preheader
  SmallVector<PHINode *, 8> FloatPhis;
  for (PHINode &Phi : ForCond130Preheader->phis()) {
    if (Phi.getType()->isFloatTy()) {
      FloatPhis.push_back(&Phi);
    }
  }
  assert(FloatPhis.size() == 8 &&
         "Expected 8 float Phi nodes in ForCond130Preheader");
  // Create parallel add instructions for better performance
  Value *Add60 =
      BinaryOperator::CreateFAdd(FloatPhis[1], AccPhi, "add60", InsertPoint);
  Value *Add61 = BinaryOperator::CreateFAdd(FloatPhis[2], FloatPhis[3], "add61",
                                            InsertPoint);
  Value *Add62 = BinaryOperator::CreateFAdd(FloatPhis[4], FloatPhis[5], "add62",
                                            InsertPoint);
  Value *Add63 = BinaryOperator::CreateFAdd(FloatPhis[6], FloatPhis[7], "add63",
                                            InsertPoint);
  Value *Add64 = BinaryOperator::CreateFAdd(Add60, Add61, "Add64", InsertPoint);
  Value *Add65 = BinaryOperator::CreateFAdd(Add62, Add63, "Add65", InsertPoint);
  Value *Add66 = BinaryOperator::CreateFAdd(Add64, Add65, "Add66", InsertPoint);
  Sum = Add66;

  // Move getelementptr and store instructions from for.cond.cleanup26 to
  // NewForEnd141
  BasicBlock *ForCondCleanup26 = getBasicBlockByName(F, "for.cond.cleanup26");

  SmallVector<Instruction *, 2> instructionsToMove;

  // Collect instructions to move
  for (Instruction &Inst : *ForCondCleanup26) {
    if (isa<GetElementPtrInst>(Inst) || isa<StoreInst>(Inst)) {
      instructionsToMove.push_back(&Inst);
    }
  }

  // Move instructions
  for (Instruction *Inst : instructionsToMove) {
    Inst->moveBefore(InsertPoint);
    if (isa<StoreInst>(Inst)) {
      Inst->setOperand(0, Sum);
    }
  }

  // Update instructions that used moved instructions
  for (Instruction &Inst : *NewForEnd141) {
    Inst.replaceUsesOfWith(ForCondCleanup26, NewForEnd141);
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

// Main Function to perform FIRD unrolling
void DspsF32FirdLoopUnroller::PostUnrollFird(Function &F, Loop *L,
                                             int Loop_index) {
  auto [CloneForBody, ForBodyMerged] = cloneAndMergeLoop(L, F, 8);
  CloneForBody->moveAfter(ForBodyMerged);

  // Perform loop-specific modifications
  if (Loop_index == 1) {
    modifyFirdFirstLoop(F, L, ForBodyMerged, CloneForBody);
  } else if (Loop_index == 2) {
    modifyFirdSecondLoop(F, L, ForBodyMerged, CloneForBody);
  }
}

void DspsF32WindBlackmanLoopUnroller::postUnrollDspsWindBlackmanF32(
    Function &F, Loop *L, int unrollCount) {

  BasicBlock *ForBodyLrPh = L->getLoopPreheader();
  runSimplifyDcePasses(F);

  auto [ForBodyclone, ForBodyMerged] = cloneAndMergeLoop(L, F, unrollCount);
  ForBodyclone->moveAfter(ForBodyMerged);

  BasicBlock *ForCondCleanup = ForBodyMerged->getTerminator()->getSuccessor(0);
  ICmpInst *Exitcond_not_7 = getLastInst<ICmpInst>(ForBodyMerged);
  Value *Len = Exitcond_not_7->getOperand(1);

  IRBuilder<> Builder(ForBodyLrPh->getTerminator());
  Value *Sub4 =
      Builder.CreateNSWAdd(Len, ConstantInt::getSigned(Len->getType(), -7), "Sub4");
  BasicBlock *ForCond97Preheader = BasicBlock::Create(
      F.getContext(), "for.cond97.preheader", &F, ForBodyMerged);

  // BasicBlock *ForBodyLrPh = L->getLoopPreheader();
  Value *Cmp169 =
      Builder.CreateICmpSGT(Len, ConstantInt::get(Len->getType(), 7), "Cmp169");

  // Modify the original branch
  ForBodyLrPh->getTerminator()->eraseFromParent();
  BranchInst::Create(ForBodyMerged, ForCond97Preheader, Cmp169, ForBodyLrPh);
  Exitcond_not_7->setOperand(1, Sub4);
  Exitcond_not_7->setPredicate(ICmpInst::ICMP_SLT);

  // Clone PHI nodes from ForBodyMerged to ForCond97Preheader
  Builder.SetInsertPoint(ForCond97Preheader);

  PHINode *I_0_lcssa = nullptr;
  // Iterate through PHI nodes in ForBodyMerged and clone
  for (auto &Phi : ForBodyMerged->phis()) {
    if (Phi.getType()->isIntegerTy(32)) {
      PHINode *NewPhi = cast<PHINode>(Phi.clone());
      NewPhi->insertInto(ForCond97Preheader, ForCond97Preheader->begin());
      NewPhi->setName("I.0.lcssa");
      I_0_lcssa = NewPhi;
      // Create comparison instruction
      Value *Cmp98171 = Builder.CreateICmpSLT(NewPhi, Len, "Cmp98171");
      // Create conditional branch instruction
      Builder.CreateCondBr(Cmp98171, ForBodyclone, ForCondCleanup);
    } else {
      llvm_unreachable("Unsupported type");
    }
  }

  ForBodyMerged->getTerminator()->setSuccessor(0, ForCond97Preheader);
  swapTerminatorSuccessors(ForBodyMerged);

  ForBodyclone->getTerminator()->setSuccessor(1, ForBodyclone);

  for (auto &Phi : ForBodyclone->phis()) {
    if (Phi.getType()->isIntegerTy(32)) {
      Phi.setIncomingValue(0, Phi.user_back());
      Phi.setIncomingValue(1, I_0_lcssa);
    } else {
      llvm_unreachable("Unsupported type");
    }
  }
  setPHINodesBlock(ForBodyclone, ForBodyclone, ForCond97Preheader);
  runSimplifyDcePasses(F);
  runPostPass(F);
}

LoopUnrollResult DspsF32FirdLoopUnroller::unroll(Loop &L) {
  simplifyAndFormLCSSA(&L, DT, &LI, SE, AC);
  LoopUnrollResult Result =
      UnrollLoop(&L,
                 {/*Count*/ UnrollCount, /*Force*/ true, /*Runtime*/ false,
                  /*AllowExpensiveTripCount*/ true,
                  /*UnrollRemainder*/ true, true},
                 &LI, &SE, &DT, &AC, &TTI, /*ORE*/ &ORE,
                 false); // must set false, true cause error
  return Result;
}

void DspiF32DotprodSmallUnroller::postUnrollDspiF32Dotprod(Function &F, Loop *L,
                                                           int unrollCount,
                                                           LoopInfo &LI) {
  runSimplifyDcePasses(F);
  BasicBlock *ForCond25Preheader = L->getLoopPredecessor();
  BasicBlock *ForCondCleanup27 =
      ForCond25Preheader->getTerminator()->getSuccessor(1);
  auto [LoopHeaderClone, ForBodyMerged] = cloneAndMergeLoop(L, F, unrollCount);
  LoopHeaderClone->moveAfter(ForBodyMerged);
  ForCondCleanup27->moveAfter(LoopHeaderClone);

  BasicBlock *ForCond25PreheaderLrPh = nullptr;
  for (BasicBlock *Pred : predecessors(ForCond25Preheader)) {
    if (Pred->getSingleSuccessor() == ForCond25Preheader) {
      ForCond25PreheaderLrPh = Pred;
      break;
    }
  }
  // Get original comparison instruction
  ICmpInst *Cmp2673 = getFirstInst<ICmpInst>(ForCond25PreheaderLrPh);

  // Get Count_x operand
  Value *Count_x = Cmp2673->getOperand(0);
  Cmp2673->setOperand(1, ConstantInt::get(Count_x->getType(), 7));
  // Create new instruction
  IRBuilder<> Builder(Cmp2673);
  Value *Sub = Builder.CreateNSWAdd(
      Count_x, ConstantInt::getSigned(Count_x->getType(), -7), "Sub");
  Value *And_val =
      Builder.CreateAnd(Count_x, ConstantInt::getSigned(Count_x->getType(), -8));

  // Create new for.cond128.preheader basic block
  BasicBlock *ForCond128Preheader = BasicBlock::Create(
      F.getContext(), "for.cond128.preheader", &F, ForBodyMerged);
  // Create x.0.lcssa Phi node
  Builder.SetInsertPoint(ForCond128Preheader);
  // Create PHI node
  PHINode *X0Lcssa =
      Builder.CreatePHI(Type::getInt32Ty(F.getContext()), 2, "x.0.lcssa");
  X0Lcssa->addIncoming(ConstantInt::get(Type::getInt32Ty(F.getContext()), 0),
                       ForCond25Preheader);
  X0Lcssa->addIncoming(And_val, ForBodyMerged);

  // Create comparison instruction
  Value *Cmp129268 = Builder.CreateICmpSLT(X0Lcssa, Count_x, "Cmp129268");

  // Create conditional branch instruction
  Builder.CreateCondBr(Cmp129268, LoopHeaderClone, ForCondCleanup27);

  ForCond25Preheader->getTerminator()->setSuccessor(1, ForCond128Preheader);

  ICmpInst *Cmp26 = getLastInst<ICmpInst>(ForBodyMerged);
  Cmp26->setPredicate(ICmpInst::ICMP_SLT);
  Cmp26->setOperand(1, Sub);
  ForBodyMerged->getTerminator()->setSuccessor(0, ForBodyMerged);
  ForBodyMerged->getTerminator()->setSuccessor(1, ForCond128Preheader);

  for (auto &Phi : LoopHeaderClone->phis()) {
    if (Phi.getType()->isIntegerTy(32)) {
      Phi.setIncomingValue(1, X0Lcssa);
      Phi.setIncomingValue(0, getLastAddInst<int32_t>(LoopHeaderClone));
    } else if (Phi.getType()->isFloatTy()) {
      Phi.setIncomingValue(0, getFirstFMulAddInst(LoopHeaderClone));
    }
  }
  setPHINodesBlock(LoopHeaderClone, LoopHeaderClone, ForCond128Preheader);
  LoopHeaderClone->getTerminator()->setSuccessor(1, LoopHeaderClone);

  PHINode *Acc174 = getFirstFloatPhi(ForBodyMerged);
  PHINode *ClonedPhi2 = nullptr;
  if (Acc174) {
    ClonedPhi2 = cast<PHINode>(Acc174->clone());
    ClonedPhi2->insertBefore(ForCond128Preheader->getFirstNonPHI());
  }

  PHINode *Acc_1_lcssa = getFirstFloatPhi(ForCondCleanup27);
  PHINode *Acc174clone = getFirstFloatPhi(LoopHeaderClone);
  PHINode *ClonedPhi = nullptr;
  if (Acc174clone) {
    Acc174clone->setIncomingValue(1, ClonedPhi2);
    ClonedPhi = cast<PHINode>(Acc174clone->clone());
    ClonedPhi->insertBefore(ForCondCleanup27->getFirstNonPHI());
  }
  Acc_1_lcssa->replaceAllUsesWith(ClonedPhi);
  Acc_1_lcssa->eraseFromParent();

  runSimplifyDcePasses(F);
  runPostPass(F);
}

void DspiF32DotprodLargeUnroller::postUnrollDspiF32DotprodVariables(
    Function &F, Loop *L, int unrollCount, LoopInfo &LI) {
  runSimplifyDcePasses(F);
  BasicBlock *ForCond25Preheader = L->getLoopPredecessor();
  BasicBlock *ForCondCleanup27 =
      ForCond25Preheader->getTerminator()->getSuccessor(1);
  auto [LoopHeaderClone, ForBodyMerged] = cloneAndMergeLoop(L, F, unrollCount);
  LoopHeaderClone->moveAfter(ForBodyMerged);
  ForCondCleanup27->moveAfter(LoopHeaderClone);

  BasicBlock *ForCond25PreheaderLrPh = nullptr;
  for (BasicBlock *Pred : predecessors(ForCond25Preheader)) {
    if (Pred->getSingleSuccessor() == ForCond25Preheader) {
      ForCond25PreheaderLrPh = Pred;
      break;
    }
  }

  // Get original comparison instruction
  ICmpInst *Cmp2673 = getFirstInst<ICmpInst>(ForCond25PreheaderLrPh);

  // Get Count_x operand
  Value *Count_x = Cmp2673->getOperand(0);
  Cmp2673->setOperand(1, ConstantInt::get(Count_x->getType(), 7));
  // Create new instruction
  IRBuilder<> Builder(Cmp2673);
  Value *Sub = Builder.CreateNSWAdd(
      Count_x, ConstantInt::getSigned(Count_x->getType(), -7), "Sub");
  Value *And_val =
      Builder.CreateAnd(Count_x, ConstantInt::getSigned(Count_x->getType(), -8));

  // Create for.end.loopexit basic block
  BasicBlock *ForEndLoopexit = BasicBlock::Create(
      F.getContext(), "for.end.loopexit", &F, LoopHeaderClone);

  // Create for.end basic block
  BasicBlock *ForEnd =
      BasicBlock::Create(F.getContext(), "for.end", &F, LoopHeaderClone);

  // Create unconditional jump from for.end.loopexit to for.end
  BranchInst::Create(ForEnd, ForEndLoopexit);

  // Create PHI node in for.end
  PHINode *X0Lcssa =
      PHINode::Create(Count_x->getType(), 2, "x.0.lcssa", ForEnd);
  X0Lcssa->addIncoming(ConstantInt::get(Count_x->getType(), 0),
                       ForCond25Preheader);
  X0Lcssa->addIncoming(And_val, ForEndLoopexit);

  // Create comparison instruction
  ICmpInst *Cmp106225 =
      new ICmpInst(ForEnd->getTerminator(), ICmpInst::ICMP_SLT, X0Lcssa,
                   Count_x, "Cmp106225");

  // Create conditional branch instruction
  BranchInst::Create(LoopHeaderClone, ForCondCleanup27, Cmp106225, ForEnd);

  // Iterate through instructions in ForBodyMerged, find Fmuladd instructions
  SmallVector<Instruction *, 8> FmuladdInsts;
  for (auto &I : *ForBodyMerged) {
    if (RecurrenceDescriptor::isFMulAddIntrinsic(&I)) {
      FmuladdInsts.push_back(&I);
    }
  }

  for (Instruction *I : FmuladdInsts) {
    // Create new Phi float instruction
    PHINode *PhiFloat = PHINode::Create(Type::getFloatTy(F.getContext()), 2, "",
                                        &*ForBodyMerged->begin());
    PhiFloat->addIncoming(cast<CallInst>(I), ForBodyMerged);
    PhiFloat->addIncoming(
        ConstantFP::get(Type::getFloatTy(F.getContext()), 0.0),
        ForCond25Preheader);
    PhiFloat->setName("acc");
    // Set last operand of Fmuladd instruction to new Phi instruction
    cast<CallInst>(I)->setArgOperand(2, PhiFloat);
  }

  Builder.SetInsertPoint(ForEndLoopexit->getFirstNonPHI());

  // Add pairs until only one value remains
  SmallVector<Value *, 8> CurrentValues(FmuladdInsts.begin(),
                                        FmuladdInsts.end());
  while (CurrentValues.size() > 1) {
    SmallVector<Value *, 8> NextValues;

    // Pairwise addition
    for (size_t I = 0; I < CurrentValues.size() - 1; I += 2) {
      Value *Sum = Builder.CreateFAdd(CurrentValues[I], CurrentValues[I + 1],
                                      "pairwise.sum");
      NextValues.push_back(Sum);
    }

    // If there's an odd value, add It to the next round
    if (CurrentValues.size() % 2 == 1) {
      NextValues.push_back(CurrentValues.back());
    }

    CurrentValues = NextValues;
  }

  PHINode *Acc_071 = getFirstFloatPhi(ForCond25Preheader);
  // Create Phi node in ForEnd
  PHINode *Add103 = PHINode::Create(Type::getFloatTy(F.getContext()), 2,
                                    "Add103", ForEnd->getFirstNonPHI());
  Add103->addIncoming(ConstantFP::get(Type::getFloatTy(F.getContext()), 0.0),
                      ForCond25Preheader);
  Add103->addIncoming(CurrentValues[0], ForEndLoopexit);

  Builder.SetInsertPoint(Add103->getNextNode());
  Value *Add104 = Builder.CreateFAdd(Acc_071, Add103, "Add104");

  ICmpInst *ExitCondNot7 = getLastInst<ICmpInst>(ForBodyMerged);
  ExitCondNot7->setPredicate(ICmpInst::ICMP_SLT);
  ExitCondNot7->setOperand(1, Sub);
  ForBodyMerged->getTerminator()->setSuccessor(0, ForBodyMerged);
  ForBodyMerged->getTerminator()->setSuccessor(1, ForEndLoopexit);

  for (auto &Phi : LoopHeaderClone->phis()) {
    if (Phi.getType()->isIntegerTy(32)) {
      Phi.setIncomingValue(1, X0Lcssa);
      Phi.setIncomingValue(0, getLastAddInst<int32_t>(LoopHeaderClone));
    } else if (Phi.getType()->isFloatTy()) {
      Phi.setIncomingValue(1, Add104);
      Phi.setIncomingValue(0, getFirstFMulAddInst(LoopHeaderClone));
    }
  }
  setPHINodesBlock(LoopHeaderClone, LoopHeaderClone, ForEnd);
  LoopHeaderClone->getTerminator()->setSuccessor(1, LoopHeaderClone);

  PHINode *Acc_1_lcssa = getFirstFloatPhi(ForCondCleanup27);
  PHINode *ClonedPhi =
      PHINode::Create(Type::getFloatTy(F.getContext()), 2, "acc.1.lcssa.clone",
                      ForCondCleanup27->getFirstNonPHI());
  ClonedPhi->addIncoming(Add104, ForEnd);
  ClonedPhi->addIncoming(getFirstFMulAddInst(LoopHeaderClone), LoopHeaderClone);
  Acc_1_lcssa->replaceAllUsesWith(ClonedPhi);
  Acc_1_lcssa->eraseFromParent();

  ForCond25Preheader->getTerminator()->setSuccessor(1, ForEnd);
}

static void preprocessUnrolledBBs(BasicBlock *BB) {
  // Get last two instructions of BB
  Instruction *Br = BB->getTerminator();
  ICmpInst *Cmp = getLastInst<ICmpInst>(BB);
  BasicBlock *NextBB = nullptr;
  BasicBlock *ForCondCleanup26Loopexit = nullptr;
  // Set successor block based on comparison instruction type
  if (Cmp->getPredicate() == ICmpInst::ICMP_EQ) {
    // If eq, swap successor block order
    NextBB = Br->getSuccessor(1);
    ForCondCleanup26Loopexit = Br->getSuccessor(0);
  } else if (Cmp->getPredicate() == ICmpInst::ICMP_SLT) {
    // If slt, keep original order
    NextBB = Br->getSuccessor(0);
    ForCondCleanup26Loopexit = Br->getSuccessor(1);
  } else {
    llvm_unreachable("Unsupported comparison predicate");
  }

  // Delete original conditional branch and comparison instruction
  Br->eraseFromParent();
  Cmp->eraseFromParent();

  // Create unconditional jump from BB to NextBB
  BranchInst::Create(NextBB, BB);

  // Delete incoming value and block from Phi node
  if (ForCondCleanup26Loopexit) {
    for (PHINode &Phi : ForCondCleanup26Loopexit->phis()) {
      int Idx = Phi.getBasicBlockIndex(BB);
      if (Idx != -1) {
        Phi.removeIncomingValue(Idx);
      }
    }
  }
}

// General Function to hoist Index into Entry block
static void hoistIndexIntoEntryBlock(BasicBlock *ForBodyMerged, Function &F,
                                     int unrollFactor) {
  BasicBlock *EntryBlock = &F.getEntryBlock();
  Instruction *EntryBlockTerminator = EntryBlock->getTerminator();
  PHINode *N_0894 = getFirstI32Phi(ForBodyMerged);

  // Static array to store multiplication instructions
  static std::vector<Value *> FilterMuls;
  static std::vector<Value *> ImageMuls;
  if (FilterMuls.empty()) {
    FilterMuls.resize(unrollFactor - 1);
    ImageMuls.resize(unrollFactor - 1);
  }

  // Get all referencing instructions of N_0894
  SmallVector<Instruction *, 2> MulUsers;
  SmallVector<Instruction *, 2> AddUsers;
  for (User *U : N_0894->users()) {
    if (Instruction *I = dyn_cast<Instruction>(U)) {
      if (I->getOpcode() == Instruction::Mul) {
        MulUsers.push_back(I);
      } else if (I->getOpcode() == Instruction::Add) {
        AddUsers.push_back(I);
      }
    }
  }

  // Get Filter_step_x and In_image_step_x
  Value *Filter_step_x = nullptr;
  Value *Mul30_ = nullptr;
  Value *In_image_step_x = nullptr;
  Value *Mul28_ = nullptr;
  for (Instruction *I : reverse(MulUsers)) {
    if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(I->user_back())) {
      if (GEP->isInBounds()) {
        Filter_step_x = I->getOperand(0);
        Mul30_ = I;
      } else {
        In_image_step_x = I->getOperand(0);
        Mul28_ = I;
      }
    }
  }
  assert(Mul30_ && "Mul30_ is nullptr");
  assert(Filter_step_x && "Filter_step_x is nullptr");

  // Process addition instructions
  for (Instruction *I : reverse(AddUsers)) {
    if (I->getOpcode() == Instruction::Add) {
      if (auto *ConstOp = dyn_cast<ConstantInt>(I->getOperand(1))) {
        int64_t ConstVal = ConstOp->getSExtValue();
        if (ConstVal != unrollFactor)
          I->setOperand(0, Mul30_);

        if (ConstVal == unrollFactor) {
          // Skip for unrollFactor case
        } else if (ConstVal == 1) {
          I->setOperand(1, Filter_step_x);
        } else if (ConstVal == 2) {
          if (FilterMuls[0] == nullptr) {
            FilterMuls[0] = BinaryOperator::CreateNSWShl(
                Filter_step_x, ConstantInt::get(Filter_step_x->getType(), 1),
                "mul28", EntryBlockTerminator);
          }
          I->setOperand(1, FilterMuls[0]);
        } else {
          int MulIdx = ConstVal - 2;
          if (FilterMuls[MulIdx] == nullptr) {
            FilterMuls[MulIdx] = BinaryOperator::CreateNSWMul(
                Filter_step_x,
                ConstantInt::get(Filter_step_x->getType(), ConstVal),
                "mul" + std::to_string(28 + MulIdx), EntryBlockTerminator);
          }
          I->setOperand(1, FilterMuls[MulIdx]);
        }

        Instruction *Inst = I->user_back();
        if (Inst->getOpcode() == Instruction::Mul) {
          Inst->replaceAllUsesWith(I);
          Inst->eraseFromParent();
        }
      }
    }
  }

  // Process GEP instruction
  Instruction *Arrayidx = dyn_cast<Instruction>(Mul28_->user_back());
  Value *Arrayidx_0_op = Arrayidx->getOperand(0);

  SmallVector<Instruction *, 16> Get_element_ptr_users;
  for (User *U : Arrayidx_0_op->users()) {
    if (Instruction *I = dyn_cast<Instruction>(U)) {
      Get_element_ptr_users.push_back(I);
    }
  }

  int Idx = 1;
  for (Instruction *I : reverse(Get_element_ptr_users)) {
    if (I != Arrayidx) {
      I->setOperand(0, Arrayidx);

      if (Idx == unrollFactor) {
        // Skip for unrollFactor case
      } else if (Idx == 1) {
        I->setOperand(1, In_image_step_x);
      } else if (Idx == 2) {
        if (ImageMuls[0] == nullptr) {
          ImageMuls[0] = BinaryOperator::CreateNSWShl(
              In_image_step_x, ConstantInt::get(In_image_step_x->getType(), 1),
              "mul" + std::to_string(34), EntryBlockTerminator);
        }
        I->setOperand(1, ImageMuls[0]);
      } else {
        int MulIdx = Idx - 2;
        if (ImageMuls[MulIdx] == nullptr) {
          ImageMuls[MulIdx] = BinaryOperator::CreateNSWMul(
              In_image_step_x,
              ConstantInt::get(In_image_step_x->getType(), Idx),
              "mul" + std::to_string(34 + MulIdx), EntryBlockTerminator);
        }
        I->setOperand(1, ImageMuls[MulIdx]);
      }
      Idx++;
    }
  }
}

// Wrapper Function, call general Function
static void hoistIndexIntoEntryBlock4(BasicBlock *ForBodyMerged, Function &F) {
  hoistIndexIntoEntryBlock(ForBodyMerged, F, 4);
}

static void hoistIndexIntoEntryBlock8(BasicBlock *ForBodyMerged, Function &F) {
  hoistIndexIntoEntryBlock(ForBodyMerged, F, 8);
}

static void hoistIndexIntoEntryBlock16(BasicBlock *ForBodyMerged, Function &F) {
  hoistIndexIntoEntryBlock(ForBodyMerged, F, 16);
}

void DspiF32ConvLargeUnroller::postUnrollDspiF32Conv(
    Function &F, SmallVector<Loop *, 2> &FIRDWillTransformLoops,
    int unrollCount, LoopInfo &LI) {
  runSimplifyDcePasses(F);
  BasicBlock &EntryBB = F.getEntryBlock();
  int Loop_idx = 0;
  for (auto L : FIRDWillTransformLoops) {

    BasicBlock *LoopHeader = L->getHeader();

    ValueToValueMapTy VMap;
    BasicBlock *LoopHeaderClone =
        CloneBasicBlock(LoopHeader, VMap, ".clone", &F);

    for (Instruction &Inst : *LoopHeaderClone) {
      for (unsigned I = 0; I < Inst.getNumOperands(); ++I) {
        Value *Op = Inst.getOperand(I);
        if (VMap.count(Op)) {
          Inst.setOperand(I, VMap[Op]); // Update operand to new value
        }
      }
    }

    LoopHeaderClone->moveAfter(LoopHeader);
    Instruction *LoopHeaderCloneTerminator = LoopHeaderClone->getTerminator();
    LoopHeaderCloneTerminator->eraseFromParent();

    preprocessUnrolledBBs(LoopHeader);

    std::vector<BasicBlock *> BBsToMerge;
    StringRef ForBodyName = LoopHeader->getName();
    for (int I = 1; I < unrollCount; ++I) {
      std::string BBName = (ForBodyName + "." + std::to_string(I)).str();
      BasicBlock *ClonedBB = getBasicBlockByName(F, BBName);
      if (I < unrollCount - 1) {
        preprocessUnrolledBBs(ClonedBB);
      }
      if (ClonedBB) {
        BBsToMerge.push_back(ClonedBB);
      } else {
        llvm_unreachable("Basic block not found");
      }
    }

    if (BBsToMerge.size() == static_cast<size_t>(unrollCount - 1)) {
      for (BasicBlock *BB : BBsToMerge) {
        MergeBasicBlockIntoOnlyPred(BB);
      }
    }

    BasicBlock *ForBodyMerged = BBsToMerge.back();
    if (unrollCount == 4) {
      hoistIndexIntoEntryBlock4(ForBodyMerged, F);
    } else if (unrollCount == 8) {
      hoistIndexIntoEntryBlock8(ForBodyMerged, F);
    } else if (unrollCount == 16) {
      hoistIndexIntoEntryBlock16(ForBodyMerged, F);
    }

    BasicBlock *ForBody = nullptr;
    for (BasicBlock *Pred : predecessors(ForBodyMerged)) {
      if (Pred != ForBodyMerged) {
        ForBody = Pred;
        break;
      }
    }
    assert(ForBody != nullptr && "ForBody not found");

    LoopHeaderClone->moveAfter(ForBodyMerged);
    for (auto &Phi : LoopHeaderClone->phis()) {
      Phi.setIncomingBlock(1, LoopHeaderClone);
    }

    BasicBlock *ForCondCleanup26 = nullptr;
    for (BasicBlock *succ : successors(ForBodyMerged)) {
      if (succ != ForBodyMerged) {
        ForCondCleanup26 = succ;
        break;
      }
    }
    ForCondCleanup26->moveAfter(LoopHeaderClone);

    // Create new loop preheader basic block
    BasicBlock *loopHeaderCloneLrPh = BasicBlock::Create(
        F.getContext(), LoopHeaderClone->getName() + ".lr.ph", &F,
        LoopHeaderClone);

    // Create unconditional branch instruction to LoopHeaderClone
    IRBuilder<> Builder(loopHeaderCloneLrPh);
    Builder.CreateBr(LoopHeaderClone);

    // Update incoming block of Phi nodes in LoopHeaderClone
    setPHIIndexIncomingBlock(LoopHeaderClone, 0, loopHeaderCloneLrPh);

    ICmpInst *cmp25_clone = getLastInst<ICmpInst>(LoopHeaderClone);
    Value *inc_clone = cmp25_clone->getOperand(0);
    Value *op1 = cmp25_clone->getOperand(1);
    cmp25_clone->setPredicate(ICmpInst::ICMP_EQ);
    Value *first_fadd = getFirstFMulAddInst(LoopHeaderClone);
    for (auto &PN : LoopHeaderClone->phis()) {
      if (PN.getType()->isIntegerTy(32)) {
        PN.setIncomingValue(1, inc_clone);
      } else if (PN.getType()->isFloatingPointTy()) {
        PN.setIncomingValue(1, first_fadd);
      }
    }

    // Create for.cond.preheader basic block
    BasicBlock *ForCondPreheader = BasicBlock::Create(
        F.getContext(), "for.cond.preheader", &F, loopHeaderCloneLrPh);

    // Clone Phi nodes from ForBodyMerged to ForCondPreheader
    IRBuilder<> BuilderPreheader(ForCondPreheader);
    for (auto &I : *ForBodyMerged) {
      PHINode *PN = dyn_cast<PHINode>(&I);
      if (!PN)
        break;

      Instruction *In = I.clone();

      if (PN->getType()->isIntegerTy(32)) {
        Value *cond = BuilderPreheader.CreateICmpSLT(In, op1);
        BuilderPreheader.CreateCondBr(cond, loopHeaderCloneLrPh,
                                      ForCondCleanup26);
      }
      In->insertBefore(&*ForCondPreheader->begin());
    }

    // Create conditional branch instruction in LoopHeaderClone
    BranchInst *BI =
        BranchInst::Create(ForCondCleanup26, LoopHeaderClone, cmp25_clone);
    BI->insertAfter(cmp25_clone);

    ForBody->getTerminator()->setSuccessor(1, ForCondPreheader);
    ForBodyMerged->getTerminator()->setSuccessor(1, ForCondPreheader);

    PHINode *Acc_1_lcssa = getLastInst<PHINode>(ForCondCleanup26);
    Acc_1_lcssa->setIncomingBlock(0, ForCondPreheader);
    Acc_1_lcssa->setIncomingValue(0, getFirstFloatPhi(ForCondPreheader));
    Acc_1_lcssa->setIncomingValue(1, first_fadd);
    Acc_1_lcssa->setIncomingBlock(1, LoopHeaderClone);

    for (auto &Phi : ForCondPreheader->phis()) {
      if (Phi.getType()->isFloatingPointTy() ||
          Phi.getType()->isIntegerTy(32)) {
        for (auto &HeaderPN : LoopHeaderClone->phis()) {
          if (HeaderPN.getType() == Phi.getType()) {
            HeaderPN.setIncomingValue(0, &Phi);
          }
        }
      }
    }

    GetElementPtrInst *gep = getFirstInst<GetElementPtrInst>(LoopHeaderClone);
    Instruction *gep_op0 = dyn_cast<Instruction>(gep->getOperand(0));
    if (gep_op0->getParent() != LoopHeaderClone) {
      Instruction *gep_op0_op0_clone =
          dyn_cast<Instruction>(gep_op0->getOperand(0))->clone();
      gep_op0_op0_clone->insertBefore(&*loopHeaderCloneLrPh->begin());
      gep->setOperand(0, gep_op0_op0_clone);
    }

    LoopHeaderClone->getTerminator()->setSuccessor(0, ForCondCleanup26);

    ICmpInst *exitcond1057_not_7 = getLastInst<ICmpInst>(ForBodyMerged);
    exitcond1057_not_7->setPredicate(ICmpInst::ICMP_SLT);

    Value *Operand_1 = exitcond1057_not_7->getOperand(1);
    if (dyn_cast<Instruction>(Operand_1)->getParent() == &EntryBB) {
      IRBuilder<> BuilderEntry(EntryBB.getTerminator());
      Value *sub22 = BuilderEntry.CreateNSWAdd(
          Operand_1, ConstantInt::getSigned(Operand_1->getType(), 1 - unrollCount),
          "sub22");
      exitcond1057_not_7->setOperand(1, sub22);

      if (BranchInst *forBodyTerm =
              dyn_cast<BranchInst>(ForBody->getTerminator())) {
        if (ICmpInst *forBodyTermCond =
                dyn_cast<ICmpInst>(forBodyTerm->getCondition())) {
          if (forBodyTermCond->getOperand(1) == Operand_1) {
            forBodyTermCond->setOperand(1, sub22);
          } else {
            // llvm_unreachable("Unexpected operand in ForBody Terminator
            // condition");
            ;
          }
        }
      }
    } else if (Instruction *Sub135 = dyn_cast<Instruction>(Operand_1)) {
      IRBuilder<> Builder(Sub135->getParent()->getTerminator());
      Value *Sub216 = Builder.CreateNSWAdd(
          Operand_1, ConstantInt::getSigned(Operand_1->getType(), 1 - unrollCount),
          "Sub216");
      exitcond1057_not_7->setOperand(1, Sub216);
    }

    Instruction *Sub135 = dyn_cast<Instruction>(Operand_1);
    for (User *U : Sub135->users()) {
      if (ICmpInst *ICI = dyn_cast<ICmpInst>(U)) {
        if (ICI->getOperand(0) == Sub135) {
          if (ConstantInt *CI = dyn_cast<ConstantInt>(ICI->getOperand(1))) {
            if (CI->isZero()) {
              ICI->setOperand(1,
                              ConstantInt::get(CI->getType(), unrollCount - 1));
            }
          }
        }
      }
    }
    ForBodyMerged->getTerminator()->setSuccessor(0, ForBodyMerged);

    Loop_idx++;
  }

  SmallVector<Instruction *, 512> InstsToMove;
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (isa<PHINode>(I) || isa<BranchInst>(I))
        continue;
      InstsToMove.push_back(&I); // Store pointer instead of object
    }
  }

  for (auto *I : InstsToMove) { // Use pointer
    if (I->getNumOperands() > 0) {
      Value *FirstOp = I->getOperand(0);
      if (Instruction *FirstOpInst = dyn_cast<Instruction>(FirstOp)) {
        BasicBlock *BaseParent = FirstOpInst->getParent();
        BasicBlock *InstParent = I->getParent();

        bool AllSameParent = true;
        for (unsigned Idx = 1; Idx < I->getNumOperands(); Idx++) {
          Value *Op = I->getOperand(Idx);
          if (Instruction *OpInst = dyn_cast<Instruction>(Op)) {
            if (OpInst->getParent() != BaseParent) {
              AllSameParent = false;
              break;
            }
          }
        }

        if (AllSameParent && BaseParent != InstParent &&
            BaseParent != nullptr) {
          I->moveBefore(BaseParent->getTerminator());
        }
      }
    }
  }

  runPostPass(F);
  runSimplifyDcePasses(F);
}

static void groupSameInstForAdd(BasicBlock *ForBodyMerged) {
  // Collect different types of instructions
  SmallVector<PHINode *> PhiNodes;
  SmallVector<Instruction *> OrInsts, GepInsts, LoadInsts, MulnswInsts,
      FaddInsts, FmulInsts, FsubInsts, StoreInsts;

  // Categorize instructions by type
  for (Instruction &Inst : *ForBodyMerged) {
    if (auto *Phi = dyn_cast<PHINode>(&Inst)) {
      PhiNodes.push_back(Phi);
    } else if (Inst.getOpcode() == Instruction::Or) {
      OrInsts.push_back(&Inst);
    } else if (isa<GetElementPtrInst>(&Inst)) {
      GepInsts.push_back(&Inst);
    } else if (isa<LoadInst>(&Inst)) {
      LoadInsts.push_back(&Inst);
    } else if (auto *binInst = dyn_cast<BinaryOperator>(&Inst)) {
      if (binInst->getOpcode() == Instruction::Mul &&
          binInst->hasNoSignedWrap()) {
        MulnswInsts.push_back(binInst);
      } else if (binInst->getOpcode() == Instruction::FAdd) {
        FaddInsts.push_back(binInst);
      } else if (binInst->getOpcode() == Instruction::FMul) {
        FmulInsts.push_back(binInst);
      } else if (binInst->getOpcode() == Instruction::FSub) {
        FsubInsts.push_back(binInst);
      }
    } else if (isa<StoreInst>(&Inst)) {
      StoreInsts.push_back(&Inst);
    }
  }

  // If no PHI nodes are found, return
  if (PhiNodes.empty()) {
    return;
  }

  // Reorder instructions
  Instruction *InsertPoint = PhiNodes.back()->getNextNode();

  auto moveInstructions = [&InsertPoint](SmallVector<Instruction *> &Insts) {
    for (auto *Inst : Insts) {
      Inst->moveBefore(InsertPoint);
      InsertPoint = Inst->getNextNode();
    }
  };

  // Move instructions in the desired order
  moveInstructions(OrInsts);
  moveInstructions(MulnswInsts);
  moveInstructions(GepInsts);
  moveInstructions(LoadInsts);
  moveInstructions(FaddInsts);
  moveInstructions(FsubInsts);
  moveInstructions(FmulInsts);
  moveInstructions(StoreInsts);
}

void DspmF32AddLoopUnroller::postUnrollDspmF32Add(Function &F, Loop *L,
                                                  int unrollCount) {
  runSimplifyDcePasses(F);
  BasicBlock *forCond34Preheader = L->getLoopPreheader();
  auto [ForBodyCloned, ForBodyMerged] = cloneAndMergeLoop(L, F, unrollCount);
  ForBodyCloned->moveAfter(ForBodyMerged);
  BasicBlock *forCondCleanup36 =
      ForBodyMerged->getTerminator()->getSuccessor(0);
  forCondCleanup36->moveAfter(ForBodyCloned);

  BasicBlock *forCond34PreheaderLrPh =
      F.getEntryBlock().getTerminator()->getSuccessor(1);

  ICmpInst *exitCond_Not_7 = getLastInst<ICmpInst>(ForBodyMerged);
  Value *cols = exitCond_Not_7->getOperand(1);
  exitCond_Not_7->setPredicate(ICmpInst::ICMP_SLT);

  IRBuilder<> Builder(forCond34PreheaderLrPh->getTerminator());
  Value *Sub = Builder.CreateAdd(cols, ConstantInt::getSigned(cols->getType(), -7),
                                 "Sub", true);
  Value *cmp35236 =
      Builder.CreateICmp(ICmpInst::ICMP_UGT, cols,
                         ConstantInt::get(cols->getType(), 7), "cmp35236");
  exitCond_Not_7->setOperand(1, Sub);

  BasicBlock *forCond113Preheader = BasicBlock::Create(
      F.getContext(), "forCond113Preheader", &F, ForBodyMerged);
  PHINode *col_0_lcssa = cast<PHINode>(ForBodyMerged->begin()->clone());
  col_0_lcssa->setName("col.0.lcssa");
  col_0_lcssa->insertInto(forCond113Preheader, forCond113Preheader->begin());

  Builder.SetInsertPoint(forCond113Preheader);
  Value *cmp114238 =
      Builder.CreateICmp(ICmpInst::ICMP_SLT, col_0_lcssa, cols, "cmp114238");
  Builder.CreateCondBr(cmp114238, ForBodyCloned, forCondCleanup36);

  // Modify jump target of forCond34Preheader to forCond113Preheader
  forCond34Preheader->getTerminator()->eraseFromParent();
  BranchInst::Create(ForBodyMerged, forCond113Preheader, cmp35236,
                     forCond34Preheader);

  PHINode *acc_080_clone = getFirstI32Phi(ForBodyCloned);
  acc_080_clone->setIncomingValue(0, getLastAddInst<int32_t>(ForBodyCloned));
  acc_080_clone->setIncomingBlock(0, ForBodyCloned);
  acc_080_clone->setIncomingValue(1, col_0_lcssa);
  acc_080_clone->setIncomingBlock(1, forCond113Preheader);

  ForBodyCloned->getTerminator()->setSuccessor(1, ForBodyCloned);
  ForBodyMerged->getTerminator()->setSuccessor(0, forCond113Preheader);
  swapTerminatorSuccessors(ForBodyMerged);

  runSimplifyDcePasses(F);
  runPostPass(F);
  groupSameInstForAdd(ForBodyMerged);
}

void DspmF32MultLoopUnroller::postUnrollDspmF32Mult(Function &F, Loop *L,
                                                    int unrollCount) {
  runSimplifyDcePasses(F);
  BasicBlock *ForBody4 = L->getLoopPredecessor();
  BasicBlock *ForCondCleanup8 = ForBody4->getTerminator()->getSuccessor(1);

  auto [forBody113, ForBodyMerged] = cloneAndMergeLoop(L, F, unrollCount);
  forBody113->moveAfter(ForBodyMerged);
  ForCondCleanup8->moveAfter(forBody113);

  ICmpInst *Exitcond_not_7 = getLastInst<ICmpInst>(ForBodyMerged);
  Value *n = Exitcond_not_7->getOperand(1);

  BasicBlock *ForCond1PreheaderLrPh =
      F.getEntryBlock().getTerminator()->getSuccessor(0);
  // Get Icmp instruction in ForCond1PreheaderLrPh
  ICmpInst *Cmp658 = nullptr;
  for (auto &I : *ForCond1PreheaderLrPh) {
    if (auto *Icmp = dyn_cast<ICmpInst>(&I)) {
      if (Icmp->getPredicate() == ICmpInst::ICMP_SGT &&
          Icmp->getOperand(0) == n) {
        Cmp658 = Icmp;
        break;
      }
    }
  }
  assert(Cmp658 && "Cmp658 not found");
  Cmp658->setOperand(1, ConstantInt::get(n->getType(), 8));

  IRBuilder<> Builder(Cmp658);
  Value *sub6 =
      Builder.CreateNSWAdd(n, ConstantInt::getSigned(n->getType(), -7), "sub6");

  // Create new for.cond110.preheader basic block
  BasicBlock *forCond110Preheader = BasicBlock::Create(
      F.getContext(), "for.cond110.preheader", &F, ForBodyMerged);

  // Clone PHI nodes from ForBodyMerged to for.cond110.preheader
  Builder.SetInsertPoint(forCond110Preheader);

  PHINode *S_0_lcssa = nullptr;
  PHINode *Acc_0_lcssa = nullptr;
  // Iterate over PHI nodes in ForBodyMerged and clone
  for (auto &Phi : ForBodyMerged->phis()) {
    PHINode *NewPhi = cast<PHINode>(Phi.clone());
    NewPhi->insertInto(forCond110Preheader, forCond110Preheader->begin());
    if (Phi.getType()->isIntegerTy(32)) {
      NewPhi->setName("s.0.lcssa");
      S_0_lcssa = NewPhi;
      // Create comparison instruction
      Value *cmp111262 = Builder.CreateICmpSLT(NewPhi, n, "cmp111262");
      // Create conditional branch instruction
      Builder.CreateCondBr(cmp111262, forBody113, ForCondCleanup8);
    } else if (Phi.getType()->isFloatTy()) {
      NewPhi->setName("acc.0.lcssa");
      Acc_0_lcssa = NewPhi;
    } else {
      llvm_unreachable("Unsupported type");
    }
  }

  // Update predecessor basic block Terminator
  ForBody4->getTerminator()->setSuccessor(1, forCond110Preheader);
  ForBodyMerged->getTerminator()->setSuccessor(0, forCond110Preheader);

  Exitcond_not_7->setOperand(1, sub6);
  Exitcond_not_7->setPredicate(ICmpInst::ICMP_SLT);
  swapTerminatorSuccessors(ForBodyMerged);

  for (auto &Phi : forBody113->phis()) {
    if (Phi.getType()->isIntegerTy(32)) {
      Phi.setIncomingValue(0, Phi.user_back());
      Phi.setIncomingValue(1, S_0_lcssa);
    } else if (Phi.getType()->isFloatTy()) {
      Phi.setIncomingValue(0, Phi.user_back());
      Phi.setIncomingValue(1, Acc_0_lcssa);
    } else {
      llvm_unreachable("Unsupported type");
    }
  }
  setPHINodesBlock(forBody113, forBody113, forCond110Preheader);
  forBody113->getTerminator()->setSuccessor(1, forBody113);

  runSimplifyDcePasses(F);
  runPostPass(F);
}

void DspmF32MultExLoopUnroller::postUnrollDspmF32MultEx(Function &F, Loop *L,
                                                        int unrollCount) {
  runSimplifyDcePasses(F);
  BasicBlock *ForBody4 = L->getLoopPredecessor();
  BasicBlock *ForCondCleanup8 = ForBody4->getTerminator()->getSuccessor(1);

  auto [forBody113, ForBodyMerged] = cloneAndMergeLoop(L, F, unrollCount);
  forBody113->moveAfter(ForBodyMerged);
  ForCondCleanup8->moveAfter(forBody113);

  ICmpInst *Exitcond_not_7 = getLastInst<ICmpInst>(ForBodyMerged);
  Value *n = Exitcond_not_7->getOperand(1);

  BasicBlock *ForCond1PreheaderLrPh =
      F.getEntryBlock().getTerminator()->getSuccessor(1);
  // Get Icmp instruction in ForCond1PreheaderLrPh
  ICmpInst *Cmp658 = nullptr;
  for (auto &I : *ForCond1PreheaderLrPh) {
    if (auto *Icmp = dyn_cast<ICmpInst>(&I)) {
      if (Icmp->getPredicate() == ICmpInst::ICMP_UGT) {
        Cmp658 = Icmp;
        break;
      }
    }
  }
  assert(Cmp658 && "Cmp658 not found");
  Cmp658->setOperand(1, ConstantInt::get(n->getType(), 8));

  IRBuilder<> Builder(Cmp658);
  Value *sub6 =
      Builder.CreateNSWAdd(n, ConstantInt::getSigned(n->getType(), -7), "sub6");

  // Create new for.cond110.preheader basic block
  BasicBlock *forCond110Preheader = BasicBlock::Create(
      F.getContext(), "for.cond110.preheader", &F, ForBodyMerged);

  // Clone PHI nodes from ForBodyMerged to for.cond110.preheader
  Builder.SetInsertPoint(forCond110Preheader);

  PHINode *S_0_lcssa = nullptr;
  PHINode *Acc_0_lcssa = nullptr;
  // Iterate over PHI nodes in ForBodyMerged and clone
  for (auto &Phi : ForBodyMerged->phis()) {
    PHINode *NewPhi = cast<PHINode>(Phi.clone());
    NewPhi->insertInto(forCond110Preheader, forCond110Preheader->begin());
    if (Phi.getType()->isIntegerTy(32)) {
      NewPhi->setName("s.0.lcssa");
      S_0_lcssa = NewPhi;
      // Create comparison instruction
      Value *cmp111262 = Builder.CreateICmpSLT(NewPhi, n, "cmp111262");
      // Create conditional branch instruction
      Builder.CreateCondBr(cmp111262, forBody113, ForCondCleanup8);
    } else if (Phi.getType()->isFloatTy()) {
      NewPhi->setName("acc.0.lcssa");
      Acc_0_lcssa = NewPhi;
    } else {
      llvm_unreachable("Unsupported type");
    }
  }

  // Update predecessor basic block Terminator
  ForBody4->getTerminator()->setSuccessor(1, forCond110Preheader);
  ForBodyMerged->getTerminator()->setSuccessor(0, forCond110Preheader);

  Exitcond_not_7->setOperand(1, sub6);
  Exitcond_not_7->setPredicate(ICmpInst::ICMP_SLT);
  swapTerminatorSuccessors(ForBodyMerged);

  for (auto &Phi : forBody113->phis()) {
    if (Phi.getType()->isIntegerTy(32)) {
      Phi.setIncomingValue(0, Phi.user_back());
      Phi.setIncomingValue(1, S_0_lcssa);
    } else if (Phi.getType()->isFloatTy()) {
      Phi.setIncomingValue(0, Phi.user_back());
      Phi.setIncomingValue(1, Acc_0_lcssa);
    } else {
      llvm_unreachable("Unsupported type");
    }
  }
  setPHINodesBlock(forBody113, forBody113, forCond110Preheader);
  forBody113->getTerminator()->setSuccessor(1, forBody113);

  runSimplifyDcePasses(F);
  runPostPass(F);
}

LoopUnrollResult DspsF32ConvCcorrUnroller::unroll(Loop &L) {
  simplifyAndFormLCSSA(&L, DT, &LI, SE, AC);
  // Get unroll factor of the loop
  unsigned factor = UnrollFactors[&L];
  return UnrollLoop(&L,
                    {/*Count*/ factor,
                     /*Force*/ true,
                     /*Runtime*/ false,
                     /*AllowExpensiveTripCount*/ true,
                     /*UnrollRemainder*/ true,
                     /*UnrollRemainder*/ true},
                    &LI, &SE, &DT, &AC, &TTI, /*ORE*/ &ORE, true);
}

// Helper Function to process CONV unroll type
void DspsF32ConvCcorrUnroller::processConvUnroll(
    Function &F, SmallVector<Loop *, 2> &InnerLoops) {
  // static const int unroll_counts[] = {8, 16, 8};
  static int unroll_index = 0;
  for (auto *L : InnerLoops) {
    PostUnrollConv(F, L, UnrollFactors[L], unroll_index);
    unroll_index = (unroll_index + 1) % 3;
  }
}

// Helper Function to process FIRD unroll type
void DspsF32FirdLoopUnroller::processFirdUnroll(
    Function &F, SmallVector<Loop *, 2> &InnerLoops) {
  static int Loop_index = 0;
  for (auto &L : InnerLoops) {
    if (Loop_index == 0) {
      Loop_index++;
      continue;
    }
    PostUnrollFird(F, L, Loop_index);
    Loop_index++;
  }
}

void LoopUnroller::addCommonOptimizationPasses(Function &F) {
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

  // Create Function-level optimization pipeline

  addPasses();
  FPM.addPass(EarlyCSEPass(true));
  FPM.addPass(ReassociatePass());

  FPM.run(F, FAM);
}

void DspsF32FirdLoopUnroller::addLegacyCommonOptimizationPasses(Function &F) {
  legacy::FunctionPassManager FPM(F.getParent());
  FPM.add(createLoopSimplifyPass());
  FPM.add(createLICMPass()); // Loop Invariant Code Motion

  // Add SimplifyCFG pass with common options
  FPM.add(createCFGSimplificationPass(
      SimplifyCFGOptions()
          .bonusInstThreshold(1) // Set instruction bonus threshold
          .forwardSwitchCondToPhi(
              true) // Allow forwarding switch conditions to Phi
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

void DspsF32BiquadLoopUnroller::postUnrollBiquad(
    Function &F, SmallVector<Loop *, 2> &InnerLoops, uint32_t unrollCount) {
  Loop *L = InnerLoops[0];

  BasicBlock *LoopHeader = L->getHeader();
  ValueToValueMapTy ForBodyClonedVMap;
  BasicBlock *ForBodyCloned =
      CloneBasicBlock(LoopHeader, ForBodyClonedVMap, ".clone", &F);

  ForBodyCloned->moveAfter(LoopHeader);
  ForBodyCloned->getTerminator()->setSuccessor(1, LoopHeader);

  std::vector<BasicBlock *> BBsToMerge;
  StringRef ForBodyName = LoopHeader->getName();
  for (int I = 1; I < unrollCount; ++I) {
    std::string BBName = (ForBodyName + "." + std::to_string(I)).str();
    BasicBlock *ClonedBB = getBasicBlockByName(F, BBName);
    if (ClonedBB) {
      BBsToMerge.push_back(ClonedBB);
    } else {
      llvm_unreachable("Basic block not found");
    }
  }

  if (BBsToMerge.size() == static_cast<size_t>(unrollCount - 1)) {
    for (BasicBlock *BB : BBsToMerge) {
      MergeBasicBlockIntoOnlyPred(BB);
    }
  }

  BasicBlock *ForBodyMerged = BBsToMerge.back();
  ForBodyCloned->moveAfter(ForBodyMerged);
  BasicBlock *ForCondCleanup = ForBodyMerged->getTerminator()->getSuccessor(0);
  ForCondCleanup->moveAfter(ForBodyCloned);

  ICmpInst *Exitcond_not_7 = getLastInst<ICmpInst>(ForBodyMerged);
  Value *Len = Exitcond_not_7->getOperand(1);
  Exitcond_not_7->setPredicate(ICmpInst::ICMP_SLT);

  BasicBlock *ForCondPreheader =
      F.getEntryBlock().getTerminator()->getSuccessor(0);

  IRBuilder<> Builder(ForCondPreheader->getTerminator());
  Value *Sub =
      Builder.CreateAdd(Len, ConstantInt::getSigned(Len->getType(), -7), "Sub", true);
  Exitcond_not_7->setOperand(1, Sub);

  BasicBlock *forCond150Preheader = BasicBlock::Create(
      F.getContext(), "for.cond150.preheader", &F, ForBodyMerged);
  PHINode *I_0_lcssa =
      cast<PHINode>(cast<Instruction>(getFirstI32Phi(ForBodyMerged))->clone());
  I_0_lcssa->setIncomingBlock(0, ForCondPreheader);
  I_0_lcssa->setName("I.0.lcssa");
  I_0_lcssa->insertInto(forCond150Preheader, forCond150Preheader->begin());

  Builder.SetInsertPoint(forCond150Preheader);
  Value *cmp1514 =
      Builder.CreateICmp(ICmpInst::ICMP_SLT, I_0_lcssa, Len, "cmp151376");
  Builder.CreateCondBr(cmp1514, ForBodyCloned, ForCondCleanup);

  ForCondPreheader->getTerminator()->setSuccessor(1, forCond150Preheader);

  ForBodyMerged->getTerminator()->setSuccessor(0, forCond150Preheader);
  swapTerminatorSuccessors(ForBodyMerged);

  BasicBlock *ForBodyLrPh = ForCondPreheader->getTerminator()->getSuccessor(0);
  // Clone ForBodyLrPh and move It before ForBodyCloned
  ValueToValueMapTy VMap;
  BasicBlock *forBodyLrPhClone =
      CloneBasicBlock(ForBodyLrPh, VMap, ".clone", &F);

  forBodyLrPhClone->moveBefore(ForBodyCloned);
  forBodyLrPhClone->getTerminator()->setSuccessor(0, ForBodyCloned);
  // Update PHI nodes in ForBodyCloned to use forBodyLrPhClone
  updatePhiNodes(ForBodyCloned, ForBodyLrPh, forBodyLrPhClone);

  for (Instruction &Inst : *forBodyLrPhClone) {
    for (unsigned I = 0; I < Inst.getNumOperands(); ++I) {
      Value *Op = Inst.getOperand(I);
      if (VMap.count(Op)) {
        Inst.setOperand(I, VMap[Op]); // Update operand to mapped new value
      }
    }
  }
  // Update branch instruction in forCond150Preheader to target forBodyLrPhClone
  forCond150Preheader->getTerminator()->setSuccessor(0, forBodyLrPhClone);

  // Collect llvm.fmuladd.f32 instructions in ForBodyCloned
  for (Instruction &Inst : *ForBodyCloned) {
    if (RecurrenceDescriptor::isFMulAddIntrinsic(&Inst)) {
      CallInst *CI = cast<CallInst>(&Inst);
      for (unsigned I = 0; I < CI->getNumOperands(); I++) {
        Value *Op = CI->getOperand(I);
        if (VMap.count(Op)) {
          CI->setOperand(I, VMap[Op]);
        } else if (ForBodyClonedVMap.count(Op)) {
          CI->setOperand(I, ForBodyClonedVMap[Op]);
        }
      }
    } else {
      for (unsigned I = 0; I < Inst.getNumOperands(); ++I) {
        Value *Op = Inst.getOperand(I);
        if (VMap.count(Op)) {
          Inst.setOperand(I, VMap[Op]);
        } else if (ForBodyClonedVMap.count(Op)) {
          Inst.setOperand(I, ForBodyClonedVMap[Op]);
        }
      }
    }
  }

  for (PHINode &Phi : ForBodyCloned->phis()) {
    if (Phi.getType()->isIntegerTy(32)) {
      Phi.setIncomingValue(0, I_0_lcssa);
    }
    Phi.setIncomingBlock(1, ForBodyCloned);
  }

  // Clone ForCondCleanup and move It after ForBodyCloned
  ValueToValueMapTy CleanupVMap;
  BasicBlock *forCondCleanupClone =
      CloneBasicBlock(ForCondCleanup, CleanupVMap, ".clone", &F);

  forCondCleanupClone->moveBefore(ForBodyCloned->getNextNode());
  ForBodyCloned->getTerminator()->setSuccessor(0, forCondCleanupClone);
  ForBodyCloned->getTerminator()->setSuccessor(1, ForBodyCloned);

  for (Instruction &Inst : *forCondCleanupClone) {
    for (unsigned I = 0; I < Inst.getNumOperands(); ++I) {
      Value *Op = Inst.getOperand(I);
      if (VMap.count(Op)) {
        Inst.setOperand(I, VMap[Op]);
      } else if (ForBodyClonedVMap.count(Op)) {
        Inst.setOperand(I, ForBodyClonedVMap[Op]);
      }
    }
  }

  PHINode *phi59 = getFirstFloatPhi(ForBodyCloned);
  Value *pre_clone14 = phi59->getIncomingValue(0);
  phi59->replaceAllUsesWith(pre_clone14);
  phi59->eraseFromParent();

  PHINode *Phi60 = getFirstFloatPhi(ForBodyCloned);
  Value *Phi62 = nullptr;
  for (User *U : Phi60->users()) {
    Phi62 = U;
  }
  Phi60->setIncomingValue(1, Phi62);

  // Find last store instruction in ForBodyCloned
  StoreInst *LastStore = nullptr;
  for (Instruction &BI : *ForBodyMerged) {
    if (auto *Store = dyn_cast<StoreInst>(&BI)) {
      LastStore = Store;
    }
  }

  // Iterate over store instructions in ForCondCleanup
  // First, collect all store instructions to be moved
  SmallVector<StoreInst *, 4> StoresToMove;
  for (Instruction &Inst : *ForCondCleanup) {
    if (auto *SI = dyn_cast<StoreInst>(&Inst)) {
      StoresToMove.push_back(SI);
    }
  }

  // Move collected store instructions together
  if (LastStore) {
    Instruction *InsertPoint = LastStore->getNextNode();
    for (StoreInst *SI : StoresToMove) {
      SI->moveBefore(InsertPoint);
    }
  }

  getFirstI32Phi(ForBodyCloned)
      ->setIncomingValue(1, getFirstAddInst<int32_t>(ForBodyCloned));
  runSimplifyDcePasses(F);
  runPostPass(F);
}

static void groupSameInstructionDspsFft2rFc32(BasicBlock *ForBody) {
  // Collect different types of instructions
  SmallVector<PHINode *> PhiNodes;
  SmallVector<Instruction *> AddInsts, ShlInsts, OrInsts, GepInsts, LoadInsts,
      FmulInsts, FsubInsts, FaddInsts, FmuladdInsts;

  // Categorize instructions by type
  for (Instruction &Inst : *ForBody) {
    if (auto *Phi = dyn_cast<PHINode>(&Inst)) {
      PhiNodes.push_back(Phi);
    } else if (Inst.getOpcode() == Instruction::Or) {
      OrInsts.push_back(&Inst);
    } else if (isa<GetElementPtrInst>(&Inst)) {
      GepInsts.push_back(&Inst);
    } else if (isa<LoadInst>(&Inst)) {
      LoadInsts.push_back(&Inst);
    } else if (Inst.getOpcode() == Instruction::FAdd) {
      FaddInsts.push_back(&Inst);
    } else if (Inst.getOpcode() == Instruction::FSub) {
      FsubInsts.push_back(&Inst);
    } else if (Inst.getOpcode() == Instruction::FMul) {
      FmulInsts.push_back(&Inst);
    } else if (auto *mulInst = dyn_cast<BinaryOperator>(&Inst)) {
      if (mulInst->getOpcode() == Instruction::Add) {
        AddInsts.push_back(mulInst);
      } else if (mulInst->getOpcode() == Instruction::Shl) {
        ShlInsts.push_back(mulInst);
      }
    } else if (RecurrenceDescriptor::isFMulAddIntrinsic(&Inst)) {
      FmuladdInsts.push_back(&Inst);
    }
  }

  // If no PHI nodes are found, return
  if (PhiNodes.empty()) {
    return;
  }

  // Reorder instructions
  Instruction *InsertPoint = PhiNodes.back()->getNextNode();

  auto moveInstructions = [&InsertPoint](SmallVector<Instruction *> &Insts) {
    for (auto *Inst : Insts) {
      Inst->moveBefore(InsertPoint);
      InsertPoint = Inst->getNextNode();
    }
  };

  // Move instructions in the desired order
  moveInstructions(AddInsts);
  moveInstructions(ShlInsts);
  moveInstructions(OrInsts);
  moveInstructions(GepInsts);
  moveInstructions(LoadInsts);
  moveInstructions(FmulInsts);
  moveInstructions(FmuladdInsts);
  moveInstructions(FaddInsts);
  moveInstructions(FsubInsts);
}

static void copyPHINodes(BasicBlock *sourceBB, BasicBlock *targetBB) {
  for (PHINode &Phi : sourceBB->phis()) {
    // Create a new PHI node
    PHINode *newPhi = PHINode::Create(Phi.getType(), Phi.getNumIncomingValues(),
                                      Phi.getName(), targetBB);

    // Copy each operand
    for (unsigned I = 0; I < Phi.getNumIncomingValues(); ++I) {
      Value *incomingValue = Phi.getIncomingValue(I);
      BasicBlock *incomingBlock = Phi.getIncomingBlock(I);
      newPhi->addIncoming(incomingValue, incomingBlock);
    }
  }
}

void DspsF32Fft2rLargeUnroller::postUnrollDspsFft2rFc32(
    Function &F, SmallVector<Loop *, 2> &FIRDWillTransformLoops,
    int unrollCount, LoopInfo &LI) {
  runSimplifyDcePasses(F);

  assert(FIRDWillTransformLoops.size() == 1);
  Loop *L = FIRDWillTransformLoops[0];

  BasicBlock *LoopHeader = L->getHeader();

  ValueToValueMapTy VMap;
  BasicBlock *LoopHeaderClone = CloneBasicBlock(LoopHeader, VMap, ".clone", &F);

  for (Instruction &Inst : *LoopHeaderClone) {
    for (unsigned I = 0; I < Inst.getNumOperands(); ++I) {
      Value *Op = Inst.getOperand(I);
      if (VMap.count(Op)) {
        Inst.setOperand(I, VMap[Op]); // Update operand to mapped new value
      }
    }
  }

  LoopHeaderClone->moveAfter(LoopHeader);
  Instruction *LoopHeaderCloneTerminator = LoopHeaderClone->getTerminator();
  LoopHeaderCloneTerminator->eraseFromParent();

  preprocessUnrolledBBs(LoopHeader);

  std::vector<BasicBlock *> BBsToMerge;
  StringRef ForBodyName = LoopHeader->getName();
  for (int I = 1; I < unrollCount; ++I) {
    std::string BBName = (ForBodyName + "." + std::to_string(I)).str();
    BasicBlock *ClonedBB = getBasicBlockByName(F, BBName);
    if (I < unrollCount - 1) {
      preprocessUnrolledBBs(ClonedBB);
    }
    if (ClonedBB) {
      BBsToMerge.push_back(ClonedBB);
    } else {
      llvm_unreachable("Basic block not found");
    }
  }

  if (BBsToMerge.size() == static_cast<size_t>(unrollCount - 1)) {
    for (BasicBlock *BB : BBsToMerge) {
      MergeBasicBlockIntoOnlyPred(BB);
    }
  }

  BasicBlock *ForBodyMerged = BBsToMerge.back();
  BasicBlock *ForBody = nullptr;
  for (BasicBlock *Pred : predecessors(ForBodyMerged)) {
    if (Pred != ForBodyMerged) {
      ForBody = Pred;
      break;
    }
  }
  assert(ForBody != nullptr && "ForBody not found");

  LoopHeaderClone->moveAfter(ForBodyMerged);
  for (auto &Phi : LoopHeaderClone->phis()) {
    Phi.setIncomingBlock(1, LoopHeaderClone);
  }

  BasicBlock *ForCondCleanup26 = nullptr;
  for (BasicBlock *succ : successors(ForBodyMerged)) {
    if (succ != ForBodyMerged) {
      ForCondCleanup26 = succ;
      break;
    }
  }
  ForCondCleanup26->moveAfter(LoopHeaderClone);

  // Create new loop preheader basic block
  BasicBlock *loopHeaderCloneLrPh =
      BasicBlock::Create(F.getContext(), LoopHeaderClone->getName() + ".lr.ph",
                         &F, LoopHeaderClone);

  // Create unconditional branch instruction to LoopHeaderClone
  IRBuilder<> Builder(loopHeaderCloneLrPh);
  Builder.CreateBr(LoopHeaderClone);

  // Update incoming block of PHI nodes in LoopHeaderClone
  setPHIIndexIncomingBlock(LoopHeaderClone, 0, loopHeaderCloneLrPh);

  for (auto &PN : LoopHeaderClone->phis()) {
    for (auto &I : *LoopHeaderClone) {
      if (auto *Add = dyn_cast<BinaryOperator>(&I)) {
        if (Add->getOpcode() == Instruction::Add && Add->hasNoSignedWrap()) {
          if (Add->getOperand(0) == &PN &&
              isa<ConstantInt>(Add->getOperand(1)) &&
              cast<ConstantInt>(Add->getOperand(1))->equalsInt(1)) {
            PN.setIncomingValue(1, Add);
            break;
          }
        }
      }
    }
  }

  ICmpInst *cmp25_clone = getLastInst<ICmpInst>(LoopHeaderClone);

  Value *N2_0105 = cmp25_clone->getOperand(1);
  // Create for.cond.preheader basic block
  BasicBlock *ForCondPreheader = BasicBlock::Create(
      F.getContext(), "for.cond.preheader", &F, loopHeaderCloneLrPh);

  // Clone PHI nodes from ForBodyMerged to ForCondPreheader
  IRBuilder<> BuilderPreheader(ForCondPreheader);

  copyPHINodes(ForBodyMerged, ForCondPreheader);
  PHINode *Temp = getFirstI32Phi(ForCondPreheader);
  Value *Cond = BuilderPreheader.CreateICmpULT(Temp, N2_0105);
  BuilderPreheader.CreateCondBr(Cond, loopHeaderCloneLrPh, ForCondCleanup26);

  // Create conditional branch instruction in LoopHeaderClone
  BranchInst *BI =
      BranchInst::Create(ForCondCleanup26, LoopHeaderClone, cmp25_clone);
  BI->insertAfter(cmp25_clone);

  ForCondPreheader->moveBefore(ForBodyMerged);
  BasicBlock *ForBody6LrPh = nullptr;
  BasicBlock *ForBody6 = nullptr;
  for (auto &L : LI) {
    for (auto &SubL : *L) {
      ForBody6LrPh = SubL->getLoopPredecessor();
      ForBody6 = SubL->getHeader();
      break;
    }
  }
  assert(ForBody6LrPh != nullptr && "ForBody6LrPh not found");
  assert(ForBody6 != nullptr && "ForBody6 not found");

  ICmpInst *cmp1097_not = getLastInst<ICmpInst>(ForBody6LrPh);

  IRBuilder<> BuilderForBody6LrPh(cmp1097_not);
  Value *Sub = BuilderForBody6LrPh.CreateAdd(
      N2_0105, ConstantInt::getSigned(N2_0105->getType(), -3), "sub",
      /*HasNUW=*/false, /*HasNSW=*/true);

  cmp1097_not->setOperand(1, ConstantInt::get(N2_0105->getType(), 7));
  cmp1097_not->setPredicate(ICmpInst::ICMP_UGT);
  Value *And_val = BuilderForBody6LrPh.CreateAnd(
      N2_0105, ConstantInt::get(N2_0105->getType(), 1073741820), "and");
  ForBody6->getTerminator()->setSuccessor(0, ForCondPreheader);
  swapTerminatorSuccessors(ForBody6);

  setPHIIndexIncomingBlock(ForCondPreheader, 0, ForBody6);

  PHINode *I_0_lcssa = getFirstI32Phi(ForCondPreheader);
  I_0_lcssa->setIncomingValue(1, And_val);

  BasicBlock *forBody12LrPh = ForBody6->getTerminator()->getSuccessor(0);
  // Move non-branch instructions from forBody12LrPh to ForBody6
  std::vector<Instruction *> instsToMove;
  for (auto &I : *forBody12LrPh) {
    if (!isa<BranchInst>(I)) {
      instsToMove.push_back(&I);
    }
  }

  // Move instructions to before ForBody6 Terminator
  Instruction *forBody6Term = ForBody6->getTerminator();
  for (auto *I : instsToMove) {
    I->moveBefore(forBody6Term);
  }

  ICmpInst *exitCond_not_3 = getLastInst<ICmpInst>(ForBodyMerged);
  exitCond_not_3->setOperand(1, Sub);
  exitCond_not_3->setPredicate(ICmpInst::ICMP_SLT);
  ForBodyMerged->getTerminator()->setSuccessor(0, ForCondPreheader);
  swapTerminatorSuccessors(ForBodyMerged);

  for (auto &phi1 : LoopHeaderClone->phis()) {
    for (auto &phi2 : ForCondPreheader->phis()) {
      if (phi1.getIncomingValue(0) == phi2.getIncomingValue(0)) {
        phi1.setIncomingValue(0, &phi2);
      }
    }
  }
  PHINode *Ia_198_clone = getLastI32Phi(LoopHeaderClone);
  PHINode *Ia_198_clone_clone = cast<PHINode>(Ia_198_clone->clone());
  Ia_198_clone_clone->insertBefore(ForCondCleanup26->getFirstNonPHI());
  Ia_198_clone_clone->setIncomingBlock(0, ForCondPreheader);
  PHINode *Ia_1_lcssa = getFirstI32Phi(ForCondCleanup26);
  Ia_1_lcssa->replaceAllUsesWith(Ia_198_clone_clone);
  Ia_1_lcssa->eraseFromParent();
  groupSameInstructionDspsFft2rFc32(ForBodyMerged);

  runPostPass(F);
  runSimplifyDcePasses(F);
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

  auto Handler =
      LoopUnrollerFactory::createLoopUnroller(F, LI, DT, SE, AC, TTI, ORE);
  if (!Handler)
    return PreservedAnalyses::all();

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
    if (!Handler->shouldUnroll(L)) {
      continue;
    }

    LoopUnrollResult Result = Handler->unroll(L);
    Changed |= Result != LoopUnrollResult::Unmodified;
    Handler->postTransform(L);
    // Clear cached analysis results if loop was fully unrolled
    if (LAM && Result == LoopUnrollResult::FullyUnrolled)
      LAM->clear(L, LoopName);
  }

  Handler->postUnrolledLoops();
  // Run dead code elimination
  runDeadCodeElimination(F);
  Handler->addCommonOptimizationPasses(F);
  Handler->addLegacyCommonOptimizationPasses(F);

  // Verify Function
  if (verifyFunction(F, &errs())) {
    LLVM_DEBUG(errs() << "Function verification failed\n");
    report_fatal_error("Function verification failed");
  }

  return Changed ? getLoopPassPreservedAnalyses() : PreservedAnalyses::all();
}
