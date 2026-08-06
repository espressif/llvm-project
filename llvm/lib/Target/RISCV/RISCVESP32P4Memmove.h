//===- RISCVESP32P4Memmove.h - ESP32-P4 memmove opt pass -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Slice s0: pass registration shell. Later stacked MRs add optimization paths.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_RISCVESP32P4MEMMOVE_H
#define LLVM_LIB_TARGET_RISCV_RISCVESP32P4MEMMOVE_H

#include "llvm/IR/PassManager.h"
#include "llvm/Support/CommandLine.h"

namespace llvm {

extern cl::opt<bool> EnableRISCVESP32P4Memmove;

struct RISCVESP32P4MemmovePass : PassInfoMixin<RISCVESP32P4MemmovePass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);
  static bool isRequired() { return true; }
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_RISCV_RISCVESP32P4MEMMOVE_H
