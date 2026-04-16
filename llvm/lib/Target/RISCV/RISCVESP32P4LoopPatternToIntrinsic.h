//===- RISCVESP32P4LoopPatternToIntrinsic.h - Loop Pattern Transform Pass ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This header defines the RISCV ESP32P4LoopPatternToIntrinsic transformation
/// pass which identifies matrix multiplication patterns in loops and replaces
/// them with optimized SIMD intrinsic calls for ESP32-P4 architecture.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_RISCVESP32P4LOOPPATTERNTOINTRINSIC_H
#define LLVM_LIB_TARGET_RISCV_RISCVESP32P4LOOPPATTERNTOINTRINSIC_H

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"

namespace llvm {

/// Command line option to enable/disable the transformation.
extern cl::opt<bool> EnableRISCVESP32P4LoopPatternToIntrinsic;

/// ESP32-P4 Loop Pattern to Intrinsic Transformation Pass.
///
/// This pass identifies specific matrix multiplication patterns in nested loops
/// and transforms them to use ESP32-P4 SIMD intrinsics when beneficial.
/// The transformation is only applied when:
/// 1. Function name matches "dspm_mult_s16*" pattern
/// 2. Function has proper nested loop structure (3 levels)
/// 3. Loop contains shift operations and rounding constants
/// 4. Vectorization conditions are met (k % 8 == 0 && shift < 15)
struct RISCVESP32P4LoopPatternToIntrinsicPass
    : public PassInfoMixin<RISCVESP32P4LoopPatternToIntrinsicPass> {

  /// Run the transformation pass on a function.
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  /// This pass is always required when enabled.
  static bool isRequired() { return true; }
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_RISCV_RISCVESP32P4LOOPPATTERNTOINTRINSIC_H