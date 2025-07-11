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

static constexpr Builtin::Info BuiltinInfo[] = {
#define BUILTIN(ID, TYPE, ATTRS)                                               \
  {#ID, TYPE, ATTRS, nullptr, HeaderDesc::NO_HEADER, ALL_LANGUAGES},
#include "clang/Basic/BuiltinsXtensa.def"
#include "clang/Basic/BuiltinsXtensaHIFI.def"
#undef BUILTIN
};

ArrayRef<Builtin::Info> XtensaTargetInfo::getTargetBuiltins() const {
  return llvm::ArrayRef(BuiltinInfo,
                        clang::Xtensa::LastTSBuiltin - Builtin::FirstTSBuiltin);
}

void XtensaTargetInfo::getTargetDefines(const LangOptions &Opts,
                                        MacroBuilder &Builder) const {
  Builder.defineMacro("__ELF__");
  Builder.defineMacro("__xtensa__");
  Builder.defineMacro("__XTENSA__");
  if (BigEndian)
    Builder.defineMacro("__XTENSA_EB__");
  else
    Builder.defineMacro("__XTENSA_EL__");
  if (HasWindowed)
    Builder.defineMacro("__XTENSA_WINDOWED_ABI__");
  else
    Builder.defineMacro("__XTENSA_CALL0_ABI__");
  if (!HasFP)
    Builder.defineMacro("__XTENSA_SOFT_FLOAT__");
  Builder.defineMacro("__XCHAL_HAVE_BE", BigEndian ? "1" : "0");
  Builder.defineMacro("__XCHAL_HAVE_DENSITY", HasDensity ? "1" : "0");
  Builder.defineMacro("__XCHAL_HAVE_MAC16", HasMAC16 ? "1" : "0");
  Builder.defineMacro("__XCHAL_HAVE_MUL32", HasMul32 ? "1" : "0");
  Builder.defineMacro("__XCHAL_HAVE_MUL32_HIGH", HasMul32High ? "1" : "0");
  Builder.defineMacro("__XCHAL_HAVE_DIV32", HasDiv32 ? "1" : "0");
  Builder.defineMacro("__XCHAL_HAVE_NSA", HasNSA ? "1" : "0");
  Builder.defineMacro("__XCHAL_HAVE_MINMAX", HasMINMAX ? "1" : "0");
  Builder.defineMacro("__XCHAL_HAVE_SEXT", HasSEXT ? "1" : "0");
  Builder.defineMacro("__XCHAL_HAVE_LOOPS", HasLoop ? "1" : "0");
  Builder.defineMacro("__XCHAL_HAVE_THREADPTR", HasTHREADPTR ? "1" : "0");
  Builder.defineMacro("__XCHAL_HAVE_S32C1I", HasS32C1I ? "1" : "0");
  Builder.defineMacro("__XCHAL_HAVE_BOOLEANS", HasBoolean ? "1" : "0");
  Builder.defineMacro("__XCHAL_HAVE_FP", HasFP ? "1" : "0");
  Builder.defineMacro("__XCHAL_HAVE_DFP_ACCEL", HasDFP ? "1" : "0");
  Builder.defineMacro("__XCHAL_HAVE_WINDOWED", HasWindowed ? "1" : "0");
  Builder.defineMacro("__XCHAL_HAVE_DEBUG", HasDebug ? "1" : "0");
  // XSHAL_ABI
  // XTHAL_ABI_WINDOWED
  // XTHAL_ABI_CALL0
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
      .Case("+density", HasDensity)
      .Case("+loop", HasLoop)
      .Case("+sext", HasSEXT)
      .Case("+nsa", HasNSA)
      .Case("+clamps", HasCLAPMS)
      .Case("+minmax", HasMINMAX)
      .Case("+mul32", HasMul32)
      .Case("+mul32high", HasMul32High)
      .Case("+div32", HasDiv32)
      .Case("+mac16", HasMAC16)
      .Case("+dfpaccel", HasDFP)
      .Case("+s32c1i", HasS32C1I)
      .Case("+threadptr", HasTHREADPTR)
      .Case("+extendedl32r", HasExtendedL32R)
      .Case("+atomctl", HasATOMCTL)
      .Case("+memctl", HasMEMCTL)
      .Case("+debug", HasDebug)
      .Case("+exception", HasException)
      .Case("+highpriinterrupts", HasHighPriInterrupts)
      .Case("+coprocessor", HasCoprocessor)
      .Case("+interrupt", HasInterrupt)
      .Case("+rvector", HasRelocatableVector)
      .Case("+timerint", HasTimerInt)
      .Case("+prid", HasPRID)
      .Case("+regprotect", HasRegionProtection)
      .Case("+miscsr", HasMiscSR)
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
    else if (Feature == "+density")
      HasDensity = true;
    else if (Feature == "+loop")
      HasLoop = true;
    else if (Feature == "+sext")
      HasSEXT = true;
    else if (Feature == "+nsa")
      HasNSA = true;
    else if (Feature == "+clamps")
      HasCLAPMS = true;
    else if (Feature == "+minmax")
      HasMINMAX = true;
    else if (Feature == "+mul32")
      HasMul32 = true;
    else if (Feature == "+mul32high")
      HasMul32High = true;
    else if (Feature == "+div32")
      HasDiv32 = true;
    else if (Feature == "+mac16")
      HasMAC16 = true;
    else if (Feature == "+dfpaccel")
      HasDFP = true;
    else if (Feature == "+s32c1i")
      HasS32C1I = true;
    else if (Feature == "+threadptr")
      HasTHREADPTR = true;
    else if (Feature == "+extendedl32r")
      HasExtendedL32R = true;
    else if (Feature == "+atomctl")
      HasATOMCTL = true;
    else if (Feature == "+memctl")
      HasMEMCTL = true;
    else if (Feature == "+debug")
      HasDebug = true;
    else if (Feature == "+exception")
      HasException = true;
    else if (Feature == "+highpriinterrupts")
      HasHighPriInterrupts = true;
    else if (Feature == "+coprocessor")
      HasCoprocessor = true;
    else if (Feature == "+interrupt")
      HasInterrupt = true;
    else if (Feature == "+rvector")
      HasRelocatableVector = true;
    else if (Feature == "+timerint")
      HasTimerInt = true;
    else if (Feature == "+prid")
      HasPRID = true;
    else if (Feature == "+regprotect")
      HasRegionProtection = true;
    else if (Feature == "+miscsr")
      HasMiscSR = true;
  }

  return true;
}
