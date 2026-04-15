//===- RISCVESP32P4FunctionSpecialization.h - Function Specialization -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass that specializes the dsps_firmr_s16_ansi
// function based on the value of the 'shift' field in its context struct.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_RISCV_RISCVESP32P4FUNCTIONSPECIALIZATION_H
#define LLVM_LIB_TARGET_RISCV_RISCVESP32P4FUNCTIONSPECIALIZATION_H

#include "llvm/IR/PassManager.h"
#include "llvm/Support/CommandLine.h"

namespace llvm {

class Function;
class Module;

extern cl::opt<bool> EnableRISCVESP32P4FunctionSpecialization;

/// Pass that performs function specialization for ESP32P4 RISC-V targets.
/// This pass specializes functions based on constant values in struct fields
/// to enable better optimization opportunities.
struct RISCVESP32P4FunctionSpecializationPass
    : public PassInfoMixin<RISCVESP32P4FunctionSpecializationPass> {

  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM);

  static bool isRequired() { return true; }

private:
  void runLoopExtractor(Function &F);
};

/// A wrapper pass to programmatically enable '-force-specialization'
/// for a single run of the IPSCCP pass.
struct ForceSpecializationWrapperPass
    : PassInfoMixin<ForceSpecializationWrapperPass> {
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

private:
  /// RAII helper to temporarily set and restore a command line option.
  class ScopedFlagSetter {
  public:
    ScopedFlagSetter(cl::opt<bool> *Opt);
    ~ScopedFlagSetter();

    ScopedFlagSetter(const ScopedFlagSetter &) = delete;
    ScopedFlagSetter &operator=(const ScopedFlagSetter &) = delete;

  private:
    cl::opt<bool> *Opt;
    bool OldValue;
    bool WasSet;
  };
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_RISCV_RISCVESP32P4FUNCTIONSPECIALIZATION_H
