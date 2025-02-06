//===- RISCVCustomLICM.h - Custom Loop Invariant Code Motion ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file declares the RISCVCustomLICM pass, which implements custom loop
/// invariant code motion optimizations for RISC-V targets.
///
/// The pass performs specialized optimizations for different types of DSP
/// functions including biquad filters, FIR filters, convolution operations,
/// FFT transforms, and window functions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_RISCVCUSTOMLICM_H
#define LLVM_LIB_TARGET_RISCV_RISCVCUSTOMLICM_H

#include "RISCVESP32P4OptUtils.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

extern cl::opt<bool> EnableRISCVCustomLICM;
class Function;

/// Abstract base class for RISC-V custom LICM optimization strategies.
/// Note: This class must be declared before RISCVCustomLICMPass
class RISCVCustomLICMOptimizationStrategy {
public:
  virtual ~RISCVCustomLICMOptimizationStrategy() = default;

  /// Check if this optimization can be applied to the given function.
  virtual bool isApplicable(Function &F, LoopInfo &LI) = 0;

  /// Apply the optimization to the given function.
  virtual bool optimize(Function &F, DominatorTree &DT, LoopInfo &LI) = 0;
};

/// Pass that performs custom loop invariant code motion for RISC-V targets.
struct RISCVCustomLICMPass : public PassInfoMixin<RISCVCustomLICMPass> {
  RISCVCustomLICMPass() {}

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  static bool isRequired() { return true; }

private:
  /// Create all optimization strategies.
  std::vector<std::unique_ptr<RISCVCustomLICMOptimizationStrategy>>
  createOptimizationStrategies();

  /// Apply optimization strategies to the function.
  bool applyOptimizationStrategies(
      const std::vector<std::unique_ptr<RISCVCustomLICMOptimizationStrategy>>
          &Strategies,
      Function &F, DominatorTree &DT, LoopInfo &LI);

  /// Update statistics based on the applied strategy.
  void updateStatistics(size_t StrategyIndex);

  /// Get strategy name by index.
  const char *getStrategyName(size_t Index);
};

/// Optimizer for biquad filter functions.
class BiquadCustomLICMOptimizer : public RISCVCustomLICMOptimizationStrategy {
public:
  bool isApplicable(Function &F, LoopInfo &LI) override {
    return isSafeToOptimizeBiquadType(F, LI);
  }

  bool optimize(Function &F, DominatorTree &DT, LoopInfo &LI) override {
    return processBiquadTypeAllLoops(F, DT, LI);
  }

  /// Check if it's safe to optimize the given function as a biquad type.
  bool isSafeToOptimizeBiquadType(Function &F, LoopInfo &LI);

  /// Check basic blocks and function parameters.
  bool checkBasicBlocksAndParameters(Function &F);

  /// Check loop nesting structure.
  bool checkLoopNestingStructure(Function &F, LoopInfo &LI);

  /// Check basic blocks and control flow.
  bool checkBasicBlocksAndControlFlow(Function &F);

  /// Process a single biquad-type loop.
  bool processBiquadTypeLoop(Loop *L, DominatorTree &DT, LoopInfo &LI,
                             Function &F);

  /// Optimize a single loop by moving invariant code.
  bool optimizeLoop(Loop *L, BasicBlock *Preheader, Function &F);

  /// Move fneg instructions out of the loop.
  void moveFNegOutOfLoop(BasicBlock *Preheader, BasicBlock &BB);

  /// Adjust PHI nodes in the loop.
  void adjustPhiNodes(BasicBlock &BB);

  /// Create a cleanup block for the loop.
  void createCleanupBlock(Function &F, BasicBlock &LoopBB);

  /// Move store instructions out of the loop.
  void moveStoreOutOfLoop(BasicBlock &BB);

  /// Process all loops in the function.
  bool processBiquadTypeAllLoops(Function &F, DominatorTree &DT, LoopInfo &LI);

  /// Check if an instruction is a must-tail call.
  bool isMustTailCall(Instruction *I);
};

/// Optimizer for FIRD (Finite Impulse Response Decimator) functions.
class FIRDCustomLICMOptimizer : public RISCVCustomLICMOptimizationStrategy {
public:
  bool isApplicable(Function &F, LoopInfo &LI) override {
    return checkInt16FIRDType(F, LI);
  }

  bool optimize(Function &F, DominatorTree &DT, LoopInfo &LI) override {
    return transformFIRDTypeGEPLoad(F, DT, LI);
  }

  /// Check if the function is an Int16 FIRD type.
  bool checkInt16FIRDType(Function &F, LoopInfo &LI);

  /// Check FIRD unroll pattern.
  bool checkInt16FIRDUnrollPattern(Loop &L);

  /// Check PHI nodes in FIRD function.
  bool checkInt16FIRDPhiNodes(BasicBlock *ForBody);

  /// Transform GEP and load instructions for FIRD type.
  bool transformFIRDTypeGEPLoad(Function &F, DominatorTree &DT, LoopInfo &LI);

  /// Modify GEP and load instructions.
  void modifyGEPAndLoadInstruction(GetElementPtrInst *GEP, ICmpInst *LastICmp);

  /// Transform load instructions.
  bool transformLoadInstructions(SmallVector<GetElementPtrInst *> GEPs,
                                 ICmpInst *LastICmp);

  /// Get GEP instructions from a basic block.
  SmallVector<GetElementPtrInst *> getGEPInstructions(BasicBlock *BB);
};

/// Optimizer for convolution functions.
class ConvCustomLICMOptimizer : public RISCVCustomLICMOptimizationStrategy {
public:
  bool isApplicable(Function &F, LoopInfo &LI) override {
    return checkDSPIConvF32Type(F, LI);
  }

  bool optimize(Function &F, DominatorTree &DT, LoopInfo &LI) override {
    return transformDSPIConvF32Type(F, LI);
  }

  /// Check if the function is a DSPI convolution F32 type.
  bool checkDSPIConvF32Type(Function &F, LoopInfo &LI);

  /// Transform DSPI convolution F32 type function.
  bool transformDSPIConvF32Type(Function &F, LoopInfo &LI);

  /// Collect loop invariant instructions.
  void collectLoopInvariantInstructions(
      Loop *L, SmallVectorImpl<Instruction *> &InvariantInsts);

  /// Hoist instructions from loops.
  bool hoistInstructionsFromLoops(LoopInfo &LI, BasicBlock &EntryBB,
                                  Function &F);

  /// Hoist instructions from sub-loops.
  bool hoistInstructionsFromSubLoop(Loop *L, int Depth);
};

/// Optimizer for DSP FFT2R FC32 functions.
class DspsFft2rFc32CustomLICMOptimizer
    : public RISCVCustomLICMOptimizationStrategy {
public:
  bool isApplicable(Function &F, LoopInfo &LI) override {
    return checkDspsFft2rFc32Type(F, LI);
  }

  bool optimize(Function &F, DominatorTree &DT, LoopInfo &LI) override {
    return transformDspsFft2rFc32Type(F, DT, LI);
  }

  /// Check if the function is a DSPS FFT2R FC32 type.
  bool checkDspsFft2rFc32Type(Function &F, LoopInfo &LI);

  /// Transform DSPS FFT2R FC32 type function.
  bool transformDspsFft2rFc32Type(Function &F, DominatorTree &DT, LoopInfo &LI);

  /// Group same instructions for DSPS FFT2R FC32.
  void groupSameInstructionDspsFft2rFc32(BasicBlock *ForBody);

  /// Find innermost loops.
  Loop *findInnermostLoops(Loop *L);

  /// Check FFT computation pattern.
  bool checkFFTComputationPattern(Function &F, LoopInfo &LI);

  /// Hoist loop invariant fneg.
  void hoistLoopInvariantFNeg(Loop *L, DominatorTree &DT, LoopInfo &LI);

private:
  /// FFT pattern analysis results.
  struct FFTPatternAnalysis {
    bool HasFMulAdd = false;
    bool HasFNegFMulPattern = false;
    bool HasComplexAccess = false;
    unsigned FAddCount = 0;
    unsigned FSubCount = 0;
  };

  /// Analyze FFT computation patterns in a basic block.
  FFTPatternAnalysis analyzeFFTPatterns(BasicBlock *Header);

  /// Check if the analysis results indicate a valid FFT pattern.
  bool isValidFFTPattern(const FFTPatternAnalysis &Analysis);
};

/// Template-based optimizer for DSP window functions.
template <unsigned N>
class DspsWindCustomLICMOptimizer : public RISCVCustomLICMOptimizationStrategy {
public:
  bool isApplicable(Function &F, LoopInfo &LI) override {
    return checkDspsWindCommonF32Type(F, N);
  }

  bool optimize(Function &F, DominatorTree &DT, LoopInfo &LI) override {
    return transformDspsWindCommonF32Type(F, N);
  }

  /// Check if the function is a DSPS window function F32 type.
  bool checkDspsWindCommonF32Type(Function &F, unsigned NumCosf);

  /// Transform DSPS window function F32 type.
  bool transformDspsWindCommonF32Type(Function &F, unsigned NumCosf);

private:
  /// Extract basic blocks for window function transformation.
  struct WindowFunctionBlocks {
    BasicBlock *EntryBB = nullptr;
    BasicBlock *ForBodyLrPh = nullptr;
    BasicBlock *ForBody = nullptr;
  };

  /// Get basic blocks needed for transformation.
  WindowFunctionBlocks getWindowFunctionBlocks(Function &F);

  /// Find floating point multiplication instruction and constant.
  std::pair<BinaryOperator *, Value *>
  findFMulInstructionAndConstant(BasicBlock *ForBody);

  /// Create multiplication instructions in preheader.
  SmallVector<Value *, 4> createMultiplicationInstructions(IRBuilder<> &Builder,
                                                           Instruction *Conv1,
                                                           Value *FMulConst,
                                                           Function &F,
                                                           unsigned NumCosf);

  /// Update cosine function calls with new parameters.
  bool updateCosineFunctionCalls(BasicBlock *ForBody,
                                 const SmallVector<Value *, 4> &Muls,
                                 unsigned NumCosf);
};

/// Type aliases for specific window function optimizers.
using DspsWindCustomLICMHannOptimizer = DspsWindCustomLICMOptimizer<1>;
using DspsWindCustomLICMBlackmanOptimizer = DspsWindCustomLICMOptimizer<2>;
using DspsWindCustomLICMBlackmanHarrisOptimizer =
    DspsWindCustomLICMOptimizer<3>;
using DspsWindCustomLICMFlatTopOptimizer = DspsWindCustomLICMOptimizer<4>;

} // namespace llvm

#endif // LLVM_LIB_TARGET_RISCV_RISCVCUSTOMLICM_H