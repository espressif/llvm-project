//===--- Xtensa.cpp - Implement Xtensa target feature support -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements Xtensa TargetInfo objects.
//
//===----------------------------------------------------------------------===//

#include "Xtensa.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/MacroBuilder.h"
#include "clang/Basic/TargetBuiltins.h"

using namespace clang;
using namespace clang::targets;

const Builtin::Info XtensaTargetInfo::BuiltinInfo[] = {
#define BUILTIN(ID, TYPE, ATTRS)                                               \
  {#ID, TYPE, ATTRS, nullptr, ALL_LANGUAGES, nullptr},
#include "clang/Basic/BuiltinsXtensa.def"
};

void XtensaTargetInfo::getTargetDefines(const LangOptions &Opts,
                                        MacroBuilder &Builder) const {
  Builder.defineMacro("__ELF__");
  Builder.defineMacro("__xtensa__");
  Builder.defineMacro("__XTENSA__");
  Builder.defineMacro("__XTENSA_EL__");
  if (HasWindowed)
    Builder.defineMacro("__XTENSA_WINDOWED_ABI__");
  else
    Builder.defineMacro("__XTENSA_CALL0_ABI__");
  if (!HasFP)
    Builder.defineMacro("__XTENSA_SOFT_FLOAT__");
}

ArrayRef<Builtin::Info> XtensaTargetInfo::getTargetBuiltins() const {
  return llvm::makeArrayRef(BuiltinInfo, clang::Xtensa::LastTSBuiltin -
                                             Builtin::FirstTSBuiltin);
}

void XtensaTargetInfo::fillValidCPUList(
    SmallVectorImpl<StringRef> &Values) const {
  llvm::Xtensa::fillValidCPUList(Values);
}

bool XtensaTargetInfo::initFeatureMap(
    llvm::StringMap<bool> &Features, DiagnosticsEngine &Diags, StringRef CPU,
    const std::vector<std::string> &FeaturesVec) const {

  // Assume that by default cpu is esp32
  if (CPU.empty())
    CPU = "esp32";

  CPU = llvm::Xtensa::getBaseName(CPU);

  SmallVector<StringRef, 16> CPUFeatures;
  llvm::Xtensa::getCPUFeatures(CPU, CPUFeatures);

  for (auto Feature : CPUFeatures)
    if (Feature[0] == '+')
      Features[Feature.drop_front(1)] = true;

  return TargetInfo::initFeatureMap(Features, Diags, CPU, FeaturesVec);
}

/// Return true if has this feature, need to sync with handleTargetFeatures.
bool XtensaTargetInfo::hasFeature(StringRef Feature) const {
  return llvm::StringSwitch<bool>(Feature)
      .Case("fp", HasFP)
      .Case("windowed", HasWindowed)
      .Case("bool", HasBoolean)
      .Case("hifi3", HasHIFI3)
      .Default(false);
}

/// Perform initialization based on the user configured set of features.
bool XtensaTargetInfo::handleTargetFeatures(std::vector<std::string> &Features,
                                            DiagnosticsEngine &Diags) {
  for (const auto &Feature : Features) {
    if (Feature == "+fp")
      HasFP = true;
    else if (Feature == "+bool")
      HasBoolean = true;
    else if (Feature == "+windowed")
      HasWindowed = true;
    else if (Feature == "+hifi3")
      HasHIFI3 = true;
  }

  return true;
}
