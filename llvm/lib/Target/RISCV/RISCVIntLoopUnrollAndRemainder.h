//===- RISCVIntLoopUnrollAndRemainder.h - Int Loop Unrolling and Remainder
// Handling
//------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RISCVIntLoopUnrollAndRemainder pass
//
// This pass performs loop unrolling and handles the remainder iterations.
// It aims to improve performance by:
// 1. Unrolling loops to reduce loop overhead and enable further optimizations
// 2. Generating efficient code for handling any remaining iterations
//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_RISCVINTLOOPUNROLLANDREMAINDER_H
#define LLVM_TRANSFORMS_UTILS_RISCVINTLOOPUNROLLANDREMAINDER_H

#include "RISCVESP32P4OptUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/IVDescriptors.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Scalar/LoopUnrollPass.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/UnrollLoop.h"
#include <cassert>

namespace llvm {
class RecurrenceDescriptor;
extern cl::opt<bool> EnableRISCVIntLoopUnrollAndRemainder;
class Function;

struct RISCVIntLoopUnrollAndRemainderPass
    : public PassInfoMixin<RISCVIntLoopUnrollAndRemainderPass> {
  RISCVIntLoopUnrollAndRemainderPass() {}

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  static bool isRequired() { return true; }
};

// Base class definition
class UnrollHandler {
protected:
  Function &F;
  LoopInfo &LI;
  DominatorTree &DT;
  unsigned UnrollCount;

public:
  SmallVector<Loop *, 2> FIRDWillTransformLoops;
  Loop *willTransformLoop;
  UnrollHandler(Function &F, LoopInfo &LI, DominatorTree &DT, unsigned Count)
      : F(F), LI(LI), DT(DT), UnrollCount(Count) {}
  virtual ~UnrollHandler() = default;

  virtual bool checkType() = 0;
  virtual void preTransform() {}
  virtual void postTransform(Loop &L) {}
  virtual void postUnroll() {}
  virtual bool shouldUnroll(Loop &L) {}
  unsigned getUnrollCount() const { return UnrollCount; }
};

// DSPS Int16 Math Base Class
class DspsInt16MathHandler : public UnrollHandler {
public:
  DspsInt16MathHandler(Function &F, LoopInfo &LI, DominatorTree &DT)
      : UnrollHandler(F, LI, DT, /*Count=*/16) {}

  bool shouldUnroll(Loop &L) override { return true; }
  void postTransform(Loop &L) override {
    postUnrollInt16MathType(F, L, UnrollCount);
  }
  void postUnrollInt16MathType(Function &F, Loop &L, int Count);
};

// DSPS Int16 Add Handler
class DspsInt16AddHandler : public DspsInt16MathHandler {
public:
  using DspsInt16MathHandler::DspsInt16MathHandler;
  bool checkType() override { return checkDspsInt16AddType(F); }
  bool checkDspsInt16AddType(Function &F);
};

// DSPS Int16 Multiply Constant Handler
class DspsInt16MulCHandler : public DspsInt16MathHandler {
public:
  using DspsInt16MathHandler::DspsInt16MathHandler;
  bool checkType() override { return checkDspsInt16MulCType(F); }
  void preTransform() override {
    preTransformDspsInt16MulC(F);
    runSimplifyDcePasses(F);
  }
  // Add these function declarations
  bool checkDspsInt16MulCType(Function &F);
  void preTransformDspsInt16MulC(Function &F);
};

// Check DSPS Int16 FIRD Unroll Pattern
static bool checkDspsInt16FirdUnrollPattern(Loop &L) {
  if (L.getLoopDepth() == 1)
    return false;

  BasicBlock *loopHeader = L.getHeader();
  return loopHeader && succ_size(loopHeader) == 2 &&
         loopHeader->getTerminator()->getSuccessor(1) == loopHeader;
}

// DSPS Int16 FIRD Handler
class DspsInt16FirdHandler : public UnrollHandler {
private:
  SmallVector<Loop *, 2> FIRDWillTransformLoops;

public:
  DspsInt16FirdHandler(Function &F, LoopInfo &LI, DominatorTree &DT)
      : UnrollHandler(F, LI, DT, /*Count=*/16) {}

  bool shouldUnroll(Loop &L) override {
    if (!checkDspsInt16FirdUnrollPattern(L)) {
      return false;
    }
    FIRDWillTransformLoops.push_back(&L);
    return true;
  }
  bool checkType() override { return checkDspsInt16FIRDType(F, LI); }
  void postUnroll() override {
    postUnrollDspsInt16FIRD(F, FIRDWillTransformLoops, UnrollCount, LI);
  }
  bool checkDspsInt16FirdPhiNodes(BasicBlock *forBody);
  bool checkDspsInt16FIRDType(Function &F, LoopInfo &LI);
  void postUnrollDspsInt16FIRD(Function &F,
                               SmallVector<Loop *, 2> &FIRDWillTransformLoops,
                               unsigned Count, LoopInfo &LI);
  void groupSameInstForFird(BasicBlock *forBodyMerged);
};

// DSPI Int Dot Product Base Class
class DspiIntDotprodHandler : public UnrollHandler {
protected:
  Loop *InnermostLoop = nullptr;
  PHINode *AccPhi = nullptr;
  PHINode *Acc_1_lcssa = nullptr;
  PHINode *AccPhiClone = nullptr;

public:
  DspiIntDotprodHandler(Function &F, LoopInfo &LI, DominatorTree &DT,
                        unsigned Count)
      : UnrollHandler(F, LI, DT, Count) {}

  bool shouldUnroll(Loop &L) override {
    if (L.isInnermost()) {
      return true;
    }
    return false;
  }
  void postTransform(Loop &L) override {
    postUnrollDspiIntDotprod(F, &L, UnrollCount, LI);
  }

  virtual void handlePhiNodes(BasicBlock *BB, PHINode *IndexLcssa) = 0;
  virtual PHINode *getAccPhi(BasicBlock *BB) = 0;
  virtual PHINode *getAccLcssaPhi(BasicBlock *BB) = 0;
  void postUnrollDspiIntDotprod(Function &F, Loop *L, unsigned Count,
                                LoopInfo &LI);
};

// DSPI Int16 Dot Product Handler
class DspiInt16DotprodHandler : public DspiIntDotprodHandler {
public:
  DspiInt16DotprodHandler(Function &F, LoopInfo &LI, DominatorTree &DT)
      : DspiIntDotprodHandler(F, LI, DT, /*Count=*/8) {}

  bool checkType() override { return checkDspiInt16DotprodType(F, LI); }
  bool checkDspiInt16DotprodType(Function &F, LoopInfo &LI);
  void handlePhiNodes(BasicBlock *BB, PHINode *x0Lcssa) override;

  PHINode *getAccPhi(BasicBlock *BB) override { return getFirstI64Phi(BB); }

  PHINode *getAccLcssaPhi(BasicBlock *BB) override {
    return getFirstI64Phi(BB);
  }
};

// DSPI Int8 Dot Product Handler
class DspiInt8DotprodHandler : public DspiIntDotprodHandler {
public:
  DspiInt8DotprodHandler(Function &F, LoopInfo &LI, DominatorTree &DT)
      : DspiIntDotprodHandler(F, LI, DT, /*Count=*/8) {}

  bool checkType() override { return checkDspiInt8DotprodType(F, LI); }
  bool checkDspiInt8DotprodType(Function &F, LoopInfo &LI);
  void handlePhiNodes(BasicBlock *BB, PHINode *x0Lcssa) override;

  PHINode *getAccPhi(BasicBlock *BB) override { return getLastI32Phi(BB); }

  PHINode *getAccLcssaPhi(BasicBlock *BB) override {
    return getFirstI32Phi(BB);
  }
};

// DSPM Int16 Multiply Handler
class DspmInt16MultHandler : public UnrollHandler {
private:
  Loop *InnermostLoop = nullptr;

public:
  DspmInt16MultHandler(Function &F, LoopInfo &LI, DominatorTree &DT)
      : UnrollHandler(F, LI, DT, /*Count=*/8) {}

  // Add function declarations
  bool checkDspmInt16MultType(Function &F, LoopInfo &LI);
  void postUnrollDspmInt16Mult(Function &F, Loop *L, int Count);

  bool shouldUnroll(Loop &L) override {
    if (L.isInnermost()) {
      return true;
    }
    return false;
  }
  bool checkType() override { return checkDspmInt16MultType(F, LI); }
  void postTransform(Loop &L) override {
    postUnrollDspmInt16Mult(F, &L, UnrollCount);
  }
};

// DSPS Int16 Dot Product Handler
class DspsInt16DotprodHandler : public UnrollHandler {
public:
  DspsInt16DotprodHandler(Function &F, LoopInfo &LI, DominatorTree &DT)
      : UnrollHandler(F, LI, DT, /*Count=*/8) {}

  bool checkType() override { return checkDspsInt16DotprodType(F); }
  bool checkDspsInt16DotprodType(Function &F);
  void postTransform(Loop &L) override {
    postUnrollDspsInt16Dotprod(F, L, UnrollCount);
  }
  bool shouldUnroll(Loop &L) override { return true; }
  void postUnrollDspsInt16Dotprod(Function &F, Loop &L, int Count);
  void groupSameInstForDotprod(BasicBlock *forBodyMerged);
};

// Factory Class
class UnrollHandlerFactory {
public:
  static std::unique_ptr<UnrollHandler> createHandler(Function &F, LoopInfo &LI,
                                                      DominatorTree &DT) {
    std::unique_ptr<UnrollHandler> handler; // Change to base class pointer

    // Check each type in priority order
    handler = std::make_unique<DspsInt16DotprodHandler>(F, LI, DT);
    if (handler->checkType()) {
      return handler;
    }

    // DSPS Int16 Add
    handler = std::make_unique<DspsInt16AddHandler>(F, LI, DT);
    if (handler->checkType())
      return handler;

    // DSPS Int16 Multiply Constant
    handler = std::make_unique<DspsInt16MulCHandler>(F, LI, DT);
    if (handler->checkType()) {
      return handler;
    }

    // DSPS Int16 FIRD
    handler = std::make_unique<DspsInt16FirdHandler>(F, LI, DT);
    if (handler->checkType())
      return handler;

    // DSPI Int16 Dot Product
    handler = std::make_unique<DspiInt16DotprodHandler>(F, LI, DT);
    if (handler->checkType())
      return handler;

    // DSPI Int8 Dot Product
    handler = std::make_unique<DspiInt8DotprodHandler>(F, LI, DT);
    if (handler->checkType())
      return handler;

    // DSPM Int16 Multiply
    handler = std::make_unique<DspmInt16MultHandler>(F, LI, DT);
    if (handler->checkType())
      return handler;

    return nullptr;
  }
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_RISCVINTLOOPUNROLLANDREMAINDER_H
