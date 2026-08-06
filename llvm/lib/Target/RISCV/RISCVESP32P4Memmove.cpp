//===-- RISCVESP32P4Memmove.cpp - ESP32-P4 memmove opt (s0 shell) ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Slice s0: enable flag + no-op run(). Behavior lands in later stacked MRs.
//
//===----------------------------------------------------------------------===//

#include "RISCVESP32P4Memmove.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/PassManager.h"

using namespace llvm;

cl::opt<bool> llvm::EnableRISCVESP32P4Memmove(
    "riscv-esp32-p4-memmove", cl::init(false),
    cl::desc("Enable ESP32-P4 memmove intrinsics optimization"));

PreservedAnalyses RISCVESP32P4MemmovePass::run(Function &F,
                                               FunctionAnalysisManager &AM) {
  (void)F;
  (void)AM;
  // s0: registration only. Overlap / SIMD paths arrive in child MRs.
  if (!EnableRISCVESP32P4Memmove)
    return PreservedAnalyses::all();
  return PreservedAnalyses::all();
}
