//===- RISCVLoopUnrollAndRemainder.h - Loop Unrolling and Remainder Handling
//------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_RISCVLOOPUNROLLANDREMAINDER_H
#define LLVM_TRANSFORMS_UTILS_RISCVLOOPUNROLLANDREMAINDER_H

#include "RISCVESP32P4OptUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/IVDescriptors.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include <cassert>
#include <tuple>

namespace llvm {
class RecurrenceDescriptor;
extern cl::opt<bool> EnableRISCVLoopUnrollAndRemainder;
class Function;

struct RISCVLoopUnrollAndRemainderPass
    : public PassInfoMixin<RISCVLoopUnrollAndRemainderPass> {
  RISCVLoopUnrollAndRemainderPass() {}

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  static bool isRequired() { return true; }
};

class LoopUnroller {
public:
  LoopUnroller(Function &F, LoopInfo &LI, DominatorTree &DT,
               ScalarEvolution &SE, AssumptionCache &AC,
               const TargetTransformInfo &TTI, OptimizationRemarkEmitter &ORE,
               unsigned UnrollCount)
      : F(F), LI(LI), DT(DT), SE(SE), AC(AC), TTI(TTI), ORE(ORE),
        UnrollCount(UnrollCount) {}
  Function &F;
  LoopInfo &LI;
  DominatorTree &DT;
  ScalarEvolution &SE;
  AssumptionCache &AC;
  const TargetTransformInfo &TTI;
  OptimizationRemarkEmitter &ORE;
  unsigned UnrollCount;
  SmallVector<Loop *, 2> willTransformLoops;
  Loop *willTransformLoop;
  FunctionPassManager FPM;
  virtual ~LoopUnroller() = default;

  virtual LoopUnrollResult unroll(Loop &L) {
    simplifyAndFormLCSSA(&L, DT, &LI, SE, AC);
    LoopUnrollResult Result =
        UnrollLoop(&L,
                   {/*Count*/ UnrollCount, /*Force*/ true, /*Runtime*/ false,
                    /*AllowExpensiveTripCount*/ true,
                    /*UnrollRemainder*/ true, true},
                   &LI, &SE, &DT, &AC, &TTI, /*ORE*/ &ORE, true);
    return Result;
  }

  bool isSimpleLoop(const Loop *L) {
    return L->getLoopDepth() == 1 && L->isInnermost() && L->isOutermost();
  }
  // Check type
  virtual bool checkType() = 0;
  // Post transform for single loop
  virtual void postTransform(Loop &L) {}
  // Post transform for multiple loops
  virtual void postUnrolledLoops() {}
  // Check if loop should be unrolled
  virtual bool shouldUnroll(Loop &L) {}
  // Get unroll count
  unsigned getUnrollCount() const { return UnrollCount; }

  virtual void addCommonOptimizationPasses(Function &F);
  virtual void addLegacyCommonOptimizationPasses(Function &F) {}
  virtual void addPasses() {}
  void simplifyAndFormLCSSA(Loop *L, DominatorTree &DT, LoopInfo *LI,
                            ScalarEvolution &SE, AssumptionCache &AC);
};

class DspsF32BiquadLoopUnroller : public LoopUnroller {
public:
  DspsF32BiquadLoopUnroller(Function &F, LoopInfo &LI, DominatorTree &DT,
                            ScalarEvolution &SE, AssumptionCache &AC,
                            const TargetTransformInfo &TTI,
                            OptimizationRemarkEmitter &ORE,
                            unsigned UnrollCount)
      : LoopUnroller(F, LI, DT, SE, AC, TTI, ORE, UnrollCount) {}

  LoopUnrollResult unroll(Loop &L) override;
  bool checkType() override { return checkDspsF32BiquadType(F); }
  void postUnrolledLoops() override {
    postUnrollBiquad(F, willTransformLoops, UnrollCount);
  }
  bool shouldUnroll(Loop &L) override {
    if (L.isInnermost()) {
      willTransformLoops.push_back(&L);
      return true;
    }
    return false;
  }
  bool checkDspsF32BiquadType(Function &F);
  void postUnrollBiquad(Function &F, SmallVector<Loop *, 2> &willTransformLoops,
                        unsigned UnrollCount);
};

class DspsF32WindBlackmanLoopUnroller : public LoopUnroller {
public:
  DspsF32WindBlackmanLoopUnroller(Function &F, LoopInfo &LI, DominatorTree &DT,
                                  ScalarEvolution &SE, AssumptionCache &AC,
                                  const TargetTransformInfo &TTI,
                                  OptimizationRemarkEmitter &ORE,
                                  unsigned UnrollCount)
      : LoopUnroller(F, LI, DT, SE, AC, TTI, ORE, UnrollCount) {}
  bool checkType() override { return checkDspsWindBlackmanF32Type(F); }
  bool shouldUnroll(Loop &L) override {
    if (L.isInnermost()) {
      return true;
    }
    return false;
  }
  bool checkDspsWindBlackmanF32Type(Function &F);
  void postTransform(Loop &L) override {
    postUnrollDspsWindBlackmanF32(F, &L, UnrollCount);
  }
  void postUnrollDspsWindBlackmanF32(Function &F, Loop *L, int unrollCount);
};

class DspsF32DotprodSimpleLoopUnroller : public LoopUnroller {
public:
  DspsF32DotprodSimpleLoopUnroller(Function &F, LoopInfo &LI, DominatorTree &DT,
                                   ScalarEvolution &SE, AssumptionCache &AC,
                                   const TargetTransformInfo &TTI,
                                   OptimizationRemarkEmitter &ORE,
                                   unsigned UnrollCount)
      : LoopUnroller(F, LI, DT, SE, AC, TTI, ORE, UnrollCount) {}
  LoopUnrollResult unroll(Loop &L) override;
  bool checkType() override { return checkDspsF32DotprodWithCount(F, LI, SE); }
  bool shouldUnroll(Loop &L) override {
    if (L.isInnermost()) {
      return true;
    }
    return false;
  }
  bool checkDspsF32DotprodWithCount(Function &F, LoopInfo &LI,
                                    ScalarEvolution &SE);
  void postTransform(Loop &L) override {
    postUnrollLoopWithCount(F, &L, UnrollCount);
  }
  void postUnrollLoopWithCount(Function &F, Loop *L, int unrollCount);

  bool checkIfDotProdSimplest(Function &F);
};

class DspsF32DotprodComplexLoopUnroller : public LoopUnroller {
public:
  DspsF32DotprodComplexLoopUnroller(Function &F, LoopInfo &LI,
                                    DominatorTree &DT, ScalarEvolution &SE,
                                    AssumptionCache &AC,
                                    const TargetTransformInfo &TTI,
                                    OptimizationRemarkEmitter &ORE,
                                    unsigned UnrollCount)
      : LoopUnroller(F, LI, DT, SE, AC, TTI, ORE, UnrollCount) {}
  LoopUnrollResult unroll(Loop &L) override;
  bool checkType() override { return checkDspsF32DotprodComplex(F, LI); }
  bool shouldUnroll(Loop &L) override {
    if (L.isInnermost()) {
      return true;
    }
    return false;
  }
  bool checkDspsF32DotprodComplex(Function &F, LoopInfo &LI);
  void postTransform(Loop &L) override {
    postUnrollLoopWithVariable(F, &L, UnrollCount);
  }
  void postUnrollLoopWithVariable(Function &F, Loop *L, int unrollCount);

  bool checkIfDotProdComplicated(Function &F);
};

class DspsF32MathLoopUnroller : public LoopUnroller {
public:
  DspsF32MathLoopUnroller(Function &F, LoopInfo &LI, DominatorTree &DT,
                          ScalarEvolution &SE, AssumptionCache &AC,
                          const TargetTransformInfo &TTI,
                          OptimizationRemarkEmitter &ORE, unsigned UnrollCount)
      : LoopUnroller(F, LI, DT, SE, AC, TTI, ORE, UnrollCount) {}
  LoopUnrollResult unroll(Loop &L) override {
    unrollMath(F, SE, &L, UnrollCount);
    return LoopUnrollResult::FullyUnrolled;
  }
  bool checkType() override { return checkDspsF32MathType(F, LI); }
  bool shouldUnroll(Loop &L) override {
    if (L.isInnermost()) {
      return true;
    }
    return false;
  }
  bool checkDspsF32MathType(Function &F, LoopInfo &LI);
  void unrollMath(Function &F, ScalarEvolution &SE, Loop *L, int unrollCount);
};

class DspmF32MultLoopUnroller : public LoopUnroller {
public:
  DspmF32MultLoopUnroller(Function &F, LoopInfo &LI, DominatorTree &DT,
                          ScalarEvolution &SE, AssumptionCache &AC,
                          const TargetTransformInfo &TTI,
                          OptimizationRemarkEmitter &ORE, unsigned UnrollCount)
      : LoopUnroller(F, LI, DT, SE, AC, TTI, ORE, UnrollCount) {}

  bool checkType() override { return checkDspmF32MultType(F, LI); }
  bool shouldUnroll(Loop &L) override {
    if (L.isInnermost()) {
      return true;
    }
    return false;
  }
  bool checkDspmF32MultType(Function &F, LoopInfo &LI);
  void postTransform(Loop &L) override {
    postUnrollDspmF32Mult(F, &L, UnrollCount);
  }
  void postUnrollDspmF32Mult(Function &F, Loop *L, int unrollCount);
};

class DspmF32MultExLoopUnroller : public LoopUnroller {
public:
  DspmF32MultExLoopUnroller(Function &F, LoopInfo &LI, DominatorTree &DT,
                            ScalarEvolution &SE, AssumptionCache &AC,
                            const TargetTransformInfo &TTI,
                            OptimizationRemarkEmitter &ORE,
                            unsigned UnrollCount)
      : LoopUnroller(F, LI, DT, SE, AC, TTI, ORE, UnrollCount) {}

  bool checkType() override { return checkDspmF32MultExType(F, LI); }
  bool shouldUnroll(Loop &L) override {
    if (L.isInnermost()) {
      return true;
    }
    return false;
  }
  bool checkDspmF32MultExType(Function &F, LoopInfo &LI);
  void postTransform(Loop &L) override {
    postUnrollDspmF32MultEx(F, &L, UnrollCount);
  }
  void postUnrollDspmF32MultEx(Function &F, Loop *L, int unrollCount);
};

class DspmF32AddLoopUnroller : public LoopUnroller {
public:
  DspmF32AddLoopUnroller(Function &F, LoopInfo &LI, DominatorTree &DT,
                         ScalarEvolution &SE, AssumptionCache &AC,
                         const TargetTransformInfo &TTI,
                         OptimizationRemarkEmitter &ORE, unsigned UnrollCount)
      : LoopUnroller(F, LI, DT, SE, AC, TTI, ORE, UnrollCount) {}

  bool checkType() override { return checkDspmF32AddType(F, LI); }
  bool shouldUnroll(Loop &L) override {
    if (L.isInnermost()) {
      return true;
    }
    return false;
  }
  bool checkDspmF32AddType(Function &F, LoopInfo &LI);
  void postTransform(Loop &L) override {
    postUnrollDspmF32Add(F, &L, UnrollCount);
  }
  void postUnrollDspmF32Add(Function &F, Loop *L, int unrollCount);
};

class DspsF32FirLoopUnroller : public LoopUnroller {
public:
  DspsF32FirLoopUnroller(Function &F, LoopInfo &LI, DominatorTree &DT,
                         ScalarEvolution &SE, AssumptionCache &AC,
                         const TargetTransformInfo &TTI,
                         OptimizationRemarkEmitter &ORE, unsigned UnrollCount)
      : LoopUnroller(F, LI, DT, SE, AC, TTI, ORE, UnrollCount) {}

  LoopUnrollResult unroll(Loop &L) override {
    unrollFir(F, &L);
    return LoopUnrollResult::FullyUnrolled;
  }
  bool checkType() override { return checkDspsF32FirType(F, LI); }
  bool shouldUnroll(Loop &L) override {
    if (L.isInnermost()) {
      return true;
    }
    return false;
  }
  bool unrollFir(Function &F, Loop *L);
  bool checkDspsF32FirType(Function &F, LoopInfo &LI);

  void addCommonOptimizationPasses(Function &F) override {}
};

class DspsF32CorrLoopUnroller : public LoopUnroller {
public:
  DspsF32CorrLoopUnroller(Function &F, LoopInfo &LI, DominatorTree &DT,
                          ScalarEvolution &SE, AssumptionCache &AC,
                          const TargetTransformInfo &TTI,
                          OptimizationRemarkEmitter &ORE, unsigned UnrollCount)
      : LoopUnroller(F, LI, DT, SE, AC, TTI, ORE, UnrollCount) {}

  LoopUnrollResult unroll(Loop &L) override {
    bool result = unrollCorr(F, &L, UnrollCount);
    return result ? LoopUnrollResult::FullyUnrolled
                  : LoopUnrollResult::Unmodified;
  }
  bool checkType() override { return checkDspsF32CorrType(F, LI); }
  bool shouldUnroll(Loop &L) override {
    if (L.isInnermost()) {
      return true;
    }
    return false;
  }
  bool unrollCorr(Function &F, Loop *L, int UnrollFactor);
  bool checkDspsF32CorrType(Function &F, LoopInfo &LI);
  void addPasses() override {
    FPM.addPass(createFunctionToLoopPassAdaptor(LoopStrengthReducePass()));
  }
};

class DspsF32FirdLoopUnroller : public LoopUnroller {
public:
  DspsF32FirdLoopUnroller(Function &F, LoopInfo &LI, DominatorTree &DT,
                          ScalarEvolution &SE, AssumptionCache &AC,
                          const TargetTransformInfo &TTI,
                          OptimizationRemarkEmitter &ORE, unsigned UnrollCount)
      : LoopUnroller(F, LI, DT, SE, AC, TTI, ORE, UnrollCount) {}

  LoopUnrollResult unroll(Loop &L) override;

  bool checkType() override { return checkDspsF32FirdType(F, LI); }

  bool shouldUnroll(Loop &L) override {
    if (L.isInnermost()) {
      willTransformLoops.push_back(&L);
      // Skip first loop (index 0)
      if (willTransformLoops.size() == 1) {
        return false;
      }
      return true;
    }
    return false;
  }

  void postUnrolledLoops() override {
    processFirdUnroll(F, willTransformLoops);
  }

  bool checkDspsF32FirdType(Function &F, LoopInfo &LI);
  void PostUnrollFird(Function &F, Loop *L, int loop_index);
  void addLegacyCommonOptimizationPasses(Function &F) override;
  void addPasses() override {
    FPM.addPass(createFunctionToLoopPassAdaptor(LoopStrengthReducePass()));
  }
  void processFirdUnroll(Function &F, SmallVector<Loop *, 2> &InnerLoops);

  void modifyFirdSecondLoop(Function &F, Loop *L, BasicBlock *ForBodyMerged,
                            BasicBlock *CloneForBody);

  void modifyFirdFirstLoop(Function &F, Loop *L, BasicBlock *ForBodyMerged,
                           BasicBlock *CloneForBody);
};

class DspsF32DotprodLoopUnroller : public LoopUnroller {
public:
  DspsF32DotprodLoopUnroller(Function &F, LoopInfo &LI, DominatorTree &DT,
                             ScalarEvolution &SE, AssumptionCache &AC,
                             const TargetTransformInfo &TTI,
                             OptimizationRemarkEmitter &ORE,
                             unsigned UnrollCount)
      : LoopUnroller(F, LI, DT, SE, AC, TTI, ORE, UnrollCount) {}

  LoopUnrollResult unroll(Loop &L) override;
  bool checkType() override { return checkDspsF32DotprodType(F, LI); }
  bool shouldUnroll(Loop &L) override {
    if (L.isInnermost()) {
      return true;
    }
    return false;
  }
  bool checkDspsF32DotprodType(Function &F, LoopInfo &LI);
  bool transformOneLoopDepth(Function &F);
};

class DspiF32DotprodUnroller : public LoopUnroller {

public:
  DspiF32DotprodUnroller(Function &F, LoopInfo &LI, ScalarEvolution &SE,
                         DominatorTree &DT, AssumptionCache &AC,
                         const TargetTransformInfo &TTI,
                         OptimizationRemarkEmitter &ORE, int UnrollCount)
      : LoopUnroller(F, LI, DT, SE, AC, TTI, ORE, UnrollCount) {}

  virtual ~DspiF32DotprodUnroller() = default;

  bool checkType() override { return checkDspiF32DotprodType(F, LI); }
  bool shouldUnroll(Loop &L) override {
    if (L.isInnermost()) {
      return true;
    }
    return false;
  }

protected:
  bool checkDspiF32DotprodType(Function &F, LoopInfo &LI);
};

class DspiF32DotprodSmallUnroller : public DspiF32DotprodUnroller {
public:
  using DspiF32DotprodUnroller::DspiF32DotprodUnroller;

  void postTransform(Loop &L) override {
    postUnrollDspiF32Dotprod(F, &L, UnrollCount, LI);
  }
  void postUnrollDspiF32Dotprod(Function &F, Loop *L, int unrollCount,
                                LoopInfo &LI);
};

class DspiF32DotprodLargeUnroller : public DspiF32DotprodUnroller {
public:
  using DspiF32DotprodUnroller::DspiF32DotprodUnroller;

  void postTransform(Loop &L) override {
    postUnrollDspiF32DotprodVariables(F, &L, UnrollCount, LI);
  }
  void postUnrollDspiF32DotprodVariables(Function &F, Loop *L, int unrollCount,
                                         LoopInfo &LI);
};

class DspiF32ConvUnroller : public LoopUnroller {
public:
  DspiF32ConvUnroller(Function &F, LoopInfo &LI, DominatorTree &DT,
                      ScalarEvolution &SE, AssumptionCache &AC,
                      const TargetTransformInfo &TTI,
                      OptimizationRemarkEmitter &ORE, int UnrollCount)
      : LoopUnroller(F, LI, DT, SE, AC, TTI, ORE, UnrollCount) {}

  bool checkType() override { return checkDspiF32ConvType(F, LI); }

  bool shouldUnroll(Loop &L) override {
    if (L.isInnermost()) {
      willTransformLoops.push_back(&L);
      return true;
    }
    return false;
  }
  bool checkDspiF32ConvType(Function &F, LoopInfo &LI);
};

// 4x4 and 8x8
class DspiF32ConvSmallUnroller : public DspiF32ConvUnroller {
public:
  using DspiF32ConvUnroller::DspiF32ConvUnroller;

  unsigned getUnrollCount() const { return UnrollCount; }

  void postUnrolledLoops() override {
    // 4x4 and 8x8 do not need post unroll processing
  }
};

// 16x16 implementation
class DspiF32ConvLargeUnroller : public DspiF32ConvUnroller {
public:
  using DspiF32ConvUnroller::DspiF32ConvUnroller;

  unsigned getUnrollCount() const { return UnrollCount; }

  void postUnrolledLoops() override {
    postUnrollDspiF32Conv(F, willTransformLoops, getUnrollCount(), LI);
  }

protected:
  void postUnrollDspiF32Conv(Function &F,
                             SmallVector<Loop *, 2> &FIRDWillTransformLoops,
                             int unrollCount, LoopInfo &LI);
};

class DspsF32Fft2rUnroller : public LoopUnroller {
public:
  DspsF32Fft2rUnroller(Function &F, LoopInfo &LI, DominatorTree &DT,
                       ScalarEvolution &SE, AssumptionCache &AC,
                       const TargetTransformInfo &TTI,
                       OptimizationRemarkEmitter &ORE, int UnrollCount)
      : LoopUnroller(F, LI, DT, SE, AC, TTI, ORE, UnrollCount) {}

  bool checkType() override { return checkDspsFft2rFc32Type(F, LI); }

  bool shouldUnroll(Loop &L) override {
    if (L.isInnermost()) {
      willTransformLoops.push_back(&L);
      return true;
    }
    return false;
  }
  bool checkDspsFft2rFc32Type(Function &F, LoopInfo &LI);
};

// 4x4 and 8x8 implementation
class DspsF32Fft2rSmallUnroller : public DspsF32Fft2rUnroller {
public:
  using DspsF32Fft2rUnroller::DspsF32Fft2rUnroller;

  unsigned getUnrollCount() const { return UnrollCount; }

  void postUnrolledLoops() override {
    // 4x4 and 8x8 do not need post unroll processing
  }
};

// 16x16 implementation
class DspsF32Fft2rLargeUnroller : public DspsF32Fft2rUnroller {
public:
  using DspsF32Fft2rUnroller::DspsF32Fft2rUnroller;

  unsigned getUnrollCount() const { return UnrollCount; }

  void postUnrolledLoops() override {
    postUnrollDspsFft2rFc32(F, willTransformLoops, getUnrollCount(), LI);
  }

protected:
  void postUnrollDspsFft2rFc32(Function &F,
                               SmallVector<Loop *, 2> &FIRDWillTransformLoops,
                               int unrollCount, LoopInfo &LI);
};

class DspsF32ConvCcorrUnroller : public LoopUnroller {

private:
  // Store the unroll factor for each loop
  DenseMap<Loop *, unsigned> UnrollFactors;

public:
  DspsF32ConvCcorrUnroller(Function &F, LoopInfo &LI, DominatorTree &DT,
                           ScalarEvolution &SE, AssumptionCache &AC,
                           const TargetTransformInfo &TTI,
                           OptimizationRemarkEmitter &ORE, int UnrollCount)
      : LoopUnroller(F, LI, DT, SE, AC, TTI, ORE, UnrollCount) {}

  LoopUnrollResult unroll(Loop &L) override;
  bool checkType() override { return checkDspsF32ConvCcorr(F, LI); }

  bool shouldUnroll(Loop &L) override {
    if (L.isInnermost()) {
      // Set the unroll factor based on the loop index
      unsigned factor = (willTransformLoops.size() == 1) ? 16 : 8;
      UnrollFactors[&L] = factor;
      willTransformLoops.push_back(&L);
      return true;
    }
    return false;
  }

  bool checkDspsF32ConvCcorr(Function &F, LoopInfo &LI);

  void postUnrolledLoops() override {
    processConvUnroll(F, willTransformLoops);
  }
  void processConvUnroll(Function &F,
                         SmallVector<Loop *, 2> &WillTransformLoops);

  void PostUnrollConv(Function &F, Loop *L, int unrollCount, int unroll_index);

  bool checkDspsF32ConvCcorrEssential(Function &F, LoopInfo &LI);

  bool checkConvolutionMemoryPattern(Function &F);
  bool checkConvolutionLoopStructure(Function &F, LoopInfo &LI);
  bool checkNullPointerValidation(Function &F);
  void analyzeIndexPattern(GetElementPtrInst *GEP, int &subCount,
                           int &addCount);
  bool containsFMulAdd(Loop *L);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_RISCVLOOPUNROLLANDREMAINDER_H