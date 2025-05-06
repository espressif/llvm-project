//===-- RISCVDotprodSplitter.h - RISC-V Dotprod Splitter Pass -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the RISCVDotprodSplitterPass class.
// This pass identifies a specific pattern often associated with calls to inner
// dot product computation functions, where the result is passed via a pointer
// argument (typically an alloca in the caller). The pattern involves a
// sequence of lifetime start, the call instruction, a load from the result
// pointer, and lifetime end, all within the same basic block.
//
// If this unique pattern is found, the pass restructures the control flow
// graph (CFG) to create specialized paths for common constant "step" or
// "stride" values (specifically 1, 2, and 3) passed as arguments to the
// inner call.
//
// It introduces conditional branches based on the runtime values of the step
// arguments. If the steps match one of the specialized constant pairs (e.g.,
// image_step=1 and filter_step=1), control flows to a duplicated version of
// the call sequence where the step arguments are replaced with constants.
// Otherwise, control flows to the original call sequence (generic path).
//
// A PHI node merges the results from the specialized and generic paths.
// This transformation aims to enable further optimizations like constant
// propagation and function specialization for the common step values within
// the called function, potentially improving performance on targets like
// RISC-V with specific dot product acceleration capabilities.
// The pass is controlled by the `-riscv-dotprod-splitter` command-line option.
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_RISCVDOTPRODSPLITTER_H
#define LLVM_LIB_TARGET_RISCV_RISCVDOTPRODSPLITTER_H

#include "llvm/IR/PassManager.h"
#include "llvm/Support/CommandLine.h"

namespace llvm {

class Function;
class Module;

extern cl::opt<bool> EnableRISCVDotprodSplitter;

/// Pass that specializes dot product calls for common step values.
///
/// This pass identifies patterns where inner dot product functions are called
/// with runtime step parameters and creates specialized versions for common
/// constant step values (1, 2, 3) to enable better optimization.
struct RISCVDotprodSplitterPass
    : public PassInfoMixin<RISCVDotprodSplitterPass> {

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  static bool isRequired() { return true; }

  /// Check if the function contains patterns that can be processed by this
  /// pass.
  static bool hasProcessablePattern(Function &F);
};

/// Conditional LoopExtractor Pass that only runs when dotprod patterns exist.
///
/// This pass runs LoopExtractor only on modules that contain processable
/// dot product patterns, avoiding unnecessary loop extraction on modules
/// that won't benefit from the dotprod splitter optimization.
struct RISCVConditionalLoopExtractorPass
    : public PassInfoMixin<RISCVConditionalLoopExtractorPass> {

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }

private:
  bool moduleHasProcessablePatterns(Module &M);
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_RISCV_RISCVDOTPRODSPLITTER_H