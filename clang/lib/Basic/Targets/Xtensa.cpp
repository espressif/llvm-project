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

using namespace clang;
using namespace clang::targets;

static constexpr int NumBuiltins =
    Xtensa::LastTSBuiltin - Builtin::FirstTSBuiltin;

static constexpr llvm::StringTable BuiltinStrings =
    CLANG_BUILTIN_STR_TABLE_START
#define BUILTIN CLANG_BUILTIN_STR_TABLE
#include "clang/Basic/BuiltinsXtensa.def"
#include "clang/Basic/BuiltinsXtensaHIFI.def"
    ;

static constexpr auto BuiltinInfos = Builtin::MakeInfos<NumBuiltins>({
#define BUILTIN CLANG_BUILTIN_ENTRY
#define LIBBUILTIN CLANG_LIBBUILTIN_ENTRY
#include "clang/Basic/BuiltinsXtensa.def"
#include "clang/Basic/BuiltinsXtensaHIFI.def"
});

llvm::SmallVector<Builtin::InfosShard>
XtensaTargetInfo::getTargetBuiltins() const {
  return {{&BuiltinStrings, BuiltinInfos}};
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
  Builder.defineMacro("__XCHAL_HAVE_BE", BigEndian ? "1" : "0");
  Builder.defineMacro("__XCHAL_HAVE_ABS");  // core arch
  Builder.defineMacro("__XCHAL_HAVE_ADDX"); // core arch
  Builder.defineMacro("__XCHAL_HAVE_L32R"); // core arch
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
      .Case("+clamps", HasCLAMPS)
      .Case("+minmax", HasMINMAX)
      .Case("+mul16", HasMul16)
      .Case("+mul32", HasMul32)
      .Case("+mul32high", HasMul32High)
      .Case("+div32", HasDiv32)
      .Case("+mac16", HasMAC16)
      .Case("+dfpaccel", HasDFP)
      .Case("+s32c1i", HasS32C1I)
      .Case("+threadptr", HasTHREADPTR)
      .Case("+extendedl32r", HasExtendedL32R)
      .Case("+debug", HasDebug)
      .Case("+exception", HasException)
      .Case("+highpriinterrupts", HasHighPriInterrupts)
      .Case("+coprocessor", HasCoprocessor)
      .Case("+interrupt", HasInterrupt)
      .Case("+rvector", HasRelocatableVector)
      .Case("+timers1", HasTimers1)
      .Case("+timers2", HasTimers1)
      .Case("+timers3", HasTimers1)
      .Case("+prid", HasPRID)
      .Case("+regprotect", HasRegionProtection)
      .Case("+miscsr", HasMiscSR)
      .Case("+dcache", HasDataCache)
      .Case("+highpriinterrupts-level3", HasHighPriInterruptsLevel3)
      .Case("+highpriinterrupts-level4", HasHighPriInterruptsLevel4)
      .Case("+highpriinterrupts-level5", HasHighPriInterruptsLevel5)
      .Case("+highpriinterrupts-level6", HasHighPriInterruptsLevel6)
      .Case("+highpriinterrupts-level7", HasHighPriInterruptsLevel7)
      .Case("+esp32s2ops", HasESP32S2Ops)
      .Case("+esp32s3ops", HasESP32S3Ops)
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
      HasCLAMPS = true;
    else if (Feature == "+minmax")
      HasMINMAX = true;
    else if (Feature == "+mul16")
      HasMul16 = true;
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
    else if (Feature == "+timers1")
      HasTimers1 = true;
    else if (Feature == "+timers2")
      HasTimers2 = true;
    else if (Feature == "+timers3")
      HasTimers3 = true;
    else if (Feature == "+prid")
      HasPRID = true;
    else if (Feature == "+regprotect")
      HasRegionProtection = true;
    else if (Feature == "+miscsr")
      HasMiscSR = true;
    else if (Feature == "+highpriinterrupts-level3")
      HasHighPriInterruptsLevel3 = true;
    else if (Feature == "+highpriinterrupts-level4")
      HasHighPriInterruptsLevel4 = true;
    else if (Feature == "+highpriinterrupts-level5")
      HasHighPriInterruptsLevel5 = true;
    else if (Feature == "+highpriinterrupts-level6")
      HasHighPriInterruptsLevel6 = true;
    else if (Feature == "+highpriinterrupts-level7")
      HasHighPriInterruptsLevel7 = true;
    else if (Feature == "+esp32s2ops")
      HasESP32S2Ops = true;
    else if (Feature == "+esp32s3ops")
      HasESP32S3Ops = true;
  }

  return true;
}
