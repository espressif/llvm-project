//===- RISCVSplitLoopByLength.h - RISCV Loop Splitting Pass ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file declares the RISCVSplitLoopByLength pass.
///
/// This pass splits loops into two parts: one for length > 2 and another for
/// length <= 2. It's designed to prepare for the esp.lp.setup instruction.
///
/// The pass handles several types of functions:
/// - Arithmetic operations: add, addc, mulc, sub, mul
/// - Dot product operations: dotprod, dotprode
/// - Square root calculation: sqrt
/// - Biquadratic filter: biquad
/// - Finite Impulse Response filter: fir
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_RISCVSPLITLOOPBYLENGTH_H
#define LLVM_LIB_TARGET_RISCV_RISCVSPLITLOOPBYLENGTH_H

#include "llvm/Analysis/IVDescriptors.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/PassManager.h"

namespace llvm {

class RecurrenceDescriptor;
class Function;

/// Command line option to enable/disable RISCVSplitLoopByLength optimization
extern cl::opt<bool> EnableRISCVSplitLoopByLength;

/// Pass that splits loops based on their length for RISCV targets.
///
/// This pass optimizes loops by creating separate code paths for loops with
/// length greater than 2 versus those with length <= 2, enabling better
/// utilization of ESP32-P4 specific loop setup instructions.
struct RISCVSplitLoopByLengthPass
    : public PassInfoMixin<RISCVSplitLoopByLengthPass> {

  /// Default constructor
  RISCVSplitLoopByLengthPass() = default;

  /// Run the pass on the given function
  /// \param F The function to transform
  /// \param AM The analysis manager providing required analyses
  /// \returns Preserved analyses after transformation
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  /// Indicates this pass is always required to run
  static bool isRequired() { return true; }
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_RISCV_RISCVSPLITLOOPBYLENGTH_H
