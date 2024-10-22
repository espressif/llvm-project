//===--- Xtensa.cpp - Xtensa Helpers for Tools -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Xtensa.h"
#include "ToolChains/CommonArgs.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "llvm/TargetParser/XtensaTargetParser.h"

using namespace clang::driver;
using namespace clang::driver::tools;
// using namespace clang;
using namespace llvm::opt;

// Get features by CPU name
static void getXtensaFeaturesFromMcpu(const Driver &D,
                                      const llvm::opt::ArgList &Args,
                                      const llvm::opt::Arg *A, llvm::StringRef Mcpu,
                                      std::vector<llvm::StringRef> &Features) {
  if (llvm::Xtensa::parseCPUKind(Mcpu) == llvm::Xtensa::CK_INVALID) {
    D.Diag(clang::diag::err_drv_clang_unsupported) << A->getAsString(Args);
  } else {
    llvm::SmallVector<llvm::StringRef, 16> CPUFeatures;
    llvm::Xtensa::getCPUFeatures(Mcpu, CPUFeatures);
    for (auto &F : CPUFeatures) {
      Features.push_back(F);
    }
  }
}

// Xtensa target features.
void xtensa::getXtensaTargetFeatures(const Driver &D, const llvm::Triple &Triple,
                                     const llvm::opt::ArgList &Args,
                                     std::vector<llvm::StringRef> &Features) {
  if (Arg *A = Args.getLastArg(options::OPT_mcpu_EQ))
    getXtensaFeaturesFromMcpu(D, Args, A, A->getValue(), Features);
}
