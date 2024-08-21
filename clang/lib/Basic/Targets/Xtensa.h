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
  std::string CPU;
  bool HasFP = false;
  bool HasWindowed = false;

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
    resetDataLayout("e-m:e-p:32:32-i64:64-i128:128-n32");
  }

  void getTargetDefines(const LangOptions &Opts,
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
} // namespace targets
} // namespace clang
#endif // LLVM_CLANG_LIB_BASIC_TARGETS_XTENSA_H
