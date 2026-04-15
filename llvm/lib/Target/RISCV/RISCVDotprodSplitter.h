//===-- RISCVDotprodSplitter.h - RISC-V Dotprod Splitter Pass -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the RISCVDotprodSplitterPass class.
// This pass identifies calls to inner dot-product helpers that pass the
// running accumulator through a pointer (usually an entry-block alloca). The
// expected caller shape is: optional @llvm.lifetime.start, the call, a single
// reload from that slot, optional @llvm.lifetime.end, in one basic block; if
// the reload was sunk to the unique successor, the pass can hoist it back.
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

  /// True if the function has a nested loop that looks like a dot-product inner
  /// loop (legacy two-load \c mul, offset variant, or Clang-style MAC with
  /// multiple affine loads). Used by the conditional loop extractor gate and
  /// related tooling.
  static bool hasProcessablePattern(Function &F);
};

/// Conditional LoopExtractor Pass that only runs when dotprod patterns exist.
///
/// This pass runs selective CodeExtractor-based loop extraction only on
/// modules that contain processable dot product patterns (same heuristic as
/// \c RISCVDotprodSplitterPass::hasProcessablePattern), avoiding work on other
/// modules. Enable with the same \c -riscv-dotprod-splitter flag as the
/// splitter pass.
///
/// Pipeline name (RISC-V target): \c riscv-dotprod-conditional-loop-extractor
/// (module pass). Run before \c riscv-dotprod-splitter when extracting inner
/// loops from Clang output is required.
struct RISCVConditionalLoopExtractorPass
    : public PassInfoMixin<RISCVConditionalLoopExtractorPass> {

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }

private:
  bool moduleHasProcessablePatterns(Module &M);
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_RISCV_RISCVDOTPRODSPLITTER_H