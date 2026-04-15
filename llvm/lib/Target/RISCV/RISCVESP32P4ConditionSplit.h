//===- RISCVESP32P4ConditionSplit.h - Condition Split Pass   ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the RISCVESP32P4ConditionSplit pass which splits
// condition branches in matrix multiplication functions to enable SIMD
// optimization.
//
// The pass identifies matrix multiplication functions by looking for:
// - A 'k' variable (loop iteration count)
// - A 'final_shift > 0' comparison pattern
// - Triple-nested loop structure (optional validation)
//
// It then splits the false branch (right-shift path) into two paths:
// - SIMD path: when k % 8 == 0 (aligned for vectorization)
// - Scalar path: when k % 8 != 0 (fallback implementation)
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_RISCVESP32P4CONDITIONSPLIT_H
#define LLVM_LIB_TARGET_RISCV_RISCVESP32P4CONDITIONSPLIT_H

#include "llvm/IR/PassManager.h"
#include "llvm/Support/CommandLine.h"

namespace llvm {

/// Command line option to enable/disable the condition split pass
extern cl::opt<bool> EnableRISCVESP32P4ConditionSplit;

class Function;

/// Pass that splits condition branches in matrix multiplication functions
/// to enable subsequent SIMD optimization.
///
/// This pass transforms:
///   if (final_shift <= 0) { /* right shift */ }
/// Into:
///   if (final_shift <= 0) {
///     if (k % 8 == 0) { /* SIMD path */ }
///     else { /* scalar path */ }
///   }
struct RISCVESP32P4ConditionSplitPass
    : public PassInfoMixin<RISCVESP32P4ConditionSplitPass> {

  /// Run the condition split transformation on the given function
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  /// This pass is always required when enabled
  static bool isRequired() { return true; }
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_RISCV_RISCVESP32P4CONDITIONSPLIT_H
