//===- RISCVESP32P4LoopVectorizeExtractor.h - ESP32-P4 Loop Vectorizer -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the RISCVESP32P4LoopVectorizeExtractorPass which prepares
// loops for ESP32-P4 specific vectorization.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_RISCVESP32P4LOOPVECTORIZEEXTRACTOR_H
#define LLVM_LIB_TARGET_RISCV_RISCVESP32P4LOOPVECTORIZEEXTRACTOR_H

#include "llvm/IR/PassManager.h"
#include "llvm/Support/CommandLine.h"

namespace llvm {

extern cl::opt<bool> EnableRISCVESP32P4LoopVectorizeExtractor;

class Function;
class Module;

/// Pass that prepares loops for ESP32-P4 specific vectorization by setting
/// appropriate loop metadata and running optimized vectorization passes.
struct RISCVESP32P4LoopVectorizeExtractorPass
    : public PassInfoMixin<RISCVESP32P4LoopVectorizeExtractorPass> {

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  static bool isRequired() { return true; }

  /// Check if the function has loops that can be processed by this pass
  static bool hasProcessableLoops(Function &F, FunctionAnalysisManager &AM);

private:
  /// Prepare loops for vectorization by setting appropriate metadata
  bool prepareLoopForVectorization(Function &F, FunctionAnalysisManager &AM,
                                   unsigned InterleaveCount);

  /// Run the actual vectorization passes on the function
  bool runVectorizationPass(Function &F, FunctionAnalysisManager &AM,
                            unsigned InterleaveCount);
};

/// Conditional wrapper that only runs LoopExtractor passes when the module
/// contains loops that have been processed by
/// RISCVESP32P4LoopVectorizeExtractorPass. This follows the same pattern as
/// CoroConditionalWrapper in LLVM.
struct RISCVESP32P4LoopExtractorConditionalWrapper
    : PassInfoMixin<RISCVESP32P4LoopExtractorConditionalWrapper> {

  explicit RISCVESP32P4LoopExtractorConditionalWrapper(ModulePassManager &&PM);

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

  static bool isRequired() { return true; }

private:
  ModulePassManager PM;

  /// Check if the module contains loops that need extraction
  bool hasLoopsNeedingExtraction(Module &M, ModuleAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_RISCV_RISCVESP32P4LOOPVECTORIZEEXTRACTOR_H
