//===--- Xtensa.h - Declare Xtensa target feature support -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares Xtensa TargetInfo objects.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_BASIC_TARGETS_XTENSA_H
#define LLVM_CLANG_LIB_BASIC_TARGETS_XTENSA_H

#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Support/Compiler.h"
#include "llvm/TargetParser/XtensaTargetParser.h"

#include "clang/Basic/Builtins.h"
#include "clang/Basic/MacroBuilder.h"
#include "clang/Basic/TargetBuiltins.h"

namespace clang {
namespace targets {

class LLVM_LIBRARY_VISIBILITY XtensaTargetInfo : public TargetInfo {
protected:
  std::string CPU;
  bool HasFP = false;
  bool HasWindowed = false;
  bool HasBoolean = false;
  bool HasHIFI3 = false;
  bool HasDensity = false;
  bool HasLoop = false;
  bool HasSEXT = false;
  bool HasNSA = false;
  bool HasCLAPMS = false;
  bool HasMINMAX = false;
  bool HasMul32 = false;
  bool HasMul32High = false;
  bool HasDiv32 = false;
  bool HasMAC16 = false;
  bool HasDFP = false;
  bool HasS32C1I = false;
  bool HasTHREADPTR = false;
  bool HasExtendedL32R = false;
  bool HasATOMCTL = false;
  bool HasMEMCTL = false;
  bool HasDebug = false;
  bool HasException = false;
  bool HasHighPriInterrupts = false;
  bool HasCoprocessor = false;
  bool HasInterrupt = false;
  bool HasRelocatableVector = false;
  bool HasTimerInt = false;
  bool HasPRID = false;
  bool HasRegionProtection = false;
  bool HasMiscSR = false;

public:
  XtensaTargetInfo(const llvm::Triple &Triple, const TargetOptions &)
      : TargetInfo(Triple) {
    BigEndian = false;
    NoAsmVariants = true;
    LongLongAlign = 64;
    SuitableAlign = 32;
    DoubleAlign = LongDoubleAlign = 64;
    SizeType = UnsignedInt;
    PtrDiffType = SignedInt;
    IntPtrType = SignedInt;
    WCharType = SignedInt;
    WIntType = UnsignedInt;
    UseZeroLengthBitfieldAlignment = true;
    MaxAtomicPromoteWidth = MaxAtomicInlineWidth = 32;
    resetDataLayout("e-m:e-p:32:32-v1:8:8-i64:64-i128:128-n32");
  }

  virtual void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override;

  ArrayRef<Builtin::Info> getTargetBuiltins() const override;

  BuiltinVaListKind getBuiltinVaListKind() const override {

    return TargetInfo::XtensaABIBuiltinVaList;
  }

  std::string_view getClobbers() const override { return ""; }

  ArrayRef<const char *> getGCCRegNames() const override {
    static const char *const GCCRegNames[] = {
        // General register name
        "a0", "sp", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "a10",
        "a11", "a12", "a13", "a14", "a15",
        // Special register name
        "sar"};
    return llvm::ArrayRef(GCCRegNames);
  }

  ArrayRef<TargetInfo::GCCRegAlias> getGCCRegAliases() const override {
    return std::nullopt;
  }

  bool validateAsmConstraint(const char *&Name,
                             TargetInfo::ConstraintInfo &Info) const override {
    switch (*Name) {
    default:
      return false;
    case 'a':
    case 'f':
      Info.setAllowsRegister();
      return true;
    }
    return false;
  }

  int getEHDataRegisterNumber(unsigned RegNo) const override {
    return (RegNo < 2) ? RegNo : -1;
  }

  bool isValidCPUName(StringRef Name) const override {
    return llvm::Xtensa::parseCPUKind(Name) != llvm::Xtensa::CK_INVALID;
  }

  bool setCPU(const std::string &Name) override {
    CPU = Name;
    return isValidCPUName(Name);
  }

  void fillValidCPUList(SmallVectorImpl<StringRef> &Values) const override;

  bool
  initFeatureMap(llvm::StringMap<bool> &Features, DiagnosticsEngine &Diags,
                 StringRef CPU,
                 const std::vector<std::string> &FeaturesVec) const override;

  bool hasFeature(StringRef Feature) const override;

  bool handleTargetFeatures(std::vector<std::string> &Features,
                            DiagnosticsEngine &Diags) override;
};

class LLVM_LIBRARY_VISIBILITY EspXtensaTargetInfo : public XtensaTargetInfo {

public:
  EspXtensaTargetInfo(const llvm::Triple &Triple, const TargetOptions &opts)
      : XtensaTargetInfo(Triple, opts) {
  }

  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override {
    XtensaTargetInfo::getTargetDefines(Opts, Builder);
    Builder.defineMacro("__XCHAL_HAVE_CONST16", "0");
    Builder.defineMacro("__XCHAL_HAVE_ABS");
    Builder.defineMacro("__XCHAL_HAVE_ADDX");
    Builder.defineMacro("__XCHAL_HAVE_L32R");
    Builder.defineMacro("__XCHAL_HAVE_MUL16");
    // FIXME: should we enable FP options below for all Xtensa CPUs if __XCHAL_HAVE_FP is 1
    Builder.defineMacro("__XCHAL_HAVE_FP_DIV", HasFP ? "1" : "0");
    Builder.defineMacro("__XCHAL_HAVE_FP_RECIP", HasFP ? "1" : "0");
    Builder.defineMacro("__XCHAL_HAVE_FP_SQRT", HasFP ? "1" : "0");
    Builder.defineMacro("__XCHAL_HAVE_FP_RSQRT", HasFP ? "1" : "0");
    Builder.defineMacro("__XCHAL_HAVE_FP_POSTINC", HasFP ? "1" : "0");
    // FIXME: should we enable DFP options below for all Xtensa CPUs if __XCHAL_HAVE_DFP_ACCEL is 1
    Builder.defineMacro("__XCHAL_HAVE_DFP_DIV", HasDFP ? "1" : "0");
    Builder.defineMacro("__XCHAL_HAVE_DFP_RECIP", HasDFP ? "1" : "0");
    Builder.defineMacro("__XCHAL_HAVE_DFP_SQRT", HasDFP ? "1" : "0");
    Builder.defineMacro("__XCHAL_HAVE_DFP_RSQRT", HasDFP ? "1" : "0");
    Builder.defineMacro("__XCHAL_HAVE_RELEASE_SYNC", "1");
    // XSHAL_USE_ABSOLUTE_LITERALS
    // XSHAL_HAVE_TEXT_SECTION_LITERALS
    Builder.defineMacro("__XCHAL_NUM_AREGS", "64");
    Builder.defineMacro("__XCHAL_HAVE_WIDE_BRANCHES", "0");
    Builder.defineMacro("__XCHAL_HAVE_PREDICTED_BRANCHES", "0");
    Builder.defineMacro("__XCHAL_ICACHE_SIZE", "0");
    Builder.defineMacro("__XCHAL_DCACHE_SIZE", "0");
    Builder.defineMacro("__XCHAL_ICACHE_LINESIZE", "16");
    Builder.defineMacro("__XCHAL_DCACHE_LINESIZE", "16");
    Builder.defineMacro("__XCHAL_ICACHE_LINEWIDTH", "4");
    Builder.defineMacro("__XCHAL_DCACHE_LINEWIDTH", "4");
    Builder.defineMacro("__XCHAL_DCACHE_IS_WRITEBACK", "0");
    Builder.defineMacro("__XCHAL_HAVE_MMU", "0");
    Builder.defineMacro("__XCHAL_NUM_IBREAK", "2");
    Builder.defineMacro("__XCHAL_NUM_DBREAK", "2");
    Builder.defineMacro("__XCHAL_DEBUGLEVEL", "6");
    if (CPU == "esp32")
        Builder.defineMacro("__XCHAL_MAX_INSTRUCTION_SIZE", "3");
    else if (CPU == "esp32s2")
        Builder.defineMacro("__XCHAL_MAX_INSTRUCTION_SIZE", "3");
    else if (CPU == "esp32s3")
        Builder.defineMacro("__XCHAL_MAX_INSTRUCTION_SIZE", "4");
    Builder.defineMacro("__XCHAL_INST_FETCH_WIDTH", "4");
  }
};

} // namespace targets
} // namespace clang
#endif // LLVM_CLANG_LIB_BASIC_TARGETS_XTENSA_H
