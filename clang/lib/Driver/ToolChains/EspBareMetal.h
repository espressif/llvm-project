//===--- EspBareMetal.h - Espressif Bare Metal Tool and ToolChain ------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_ESPBAREMETAL_H
#define LLVM_CLANG_LIB_DRIVER_TOOLCHAINS_ESPBAREMETAL_H

#include "BareMetal.h"

namespace clang {
namespace driver {

namespace toolchains {

class LLVM_LIBRARY_VISIBILITY EspBareMetal : public BareMetal {
public:
  EspBareMetal(const Driver &D, const llvm::Triple &Triple,
            const llvm::opt::ArgList &Args);

  ~EspBareMetal() override = default;

protected:
  Tool *buildLinker() const override;
  Tool *buildAssembler() const override;

public:
  // BareMetal forces to use integrated ASM, we wnat to be more flexible 
  // and allow users to to switch to GNU ASM using command line
  bool useIntegratedAs() const override { return ToolChain::useIntegratedAs(); }
  bool IsIntegratedAssemblerDefault() const override { return IsIntegratedAsm; }

  RuntimeLibType GetDefaultRuntimeLibType() const override {
    return ToolChain::RLT_CompilerRT;
  }
  CXXStdlibType GetDefaultCXXStdlibType() const override {
    return ToolChain::CST_Libstdcxx;
  }

  void AddClangCXXStdlibIncludeArgs(
      const llvm::opt::ArgList &DriverArgs,
      llvm::opt::ArgStringList &CC1Args) const override;

  const char *getDefaultLinker() const override { return "ld.lld"; }

  void AddCXXStdlibLibArgs(const llvm::opt::ArgList &Args,
                           llvm::opt::ArgStringList &CmdArgs) const override;

  virtual Multilib::flags_list getMultilibFlags(const llvm::opt::ArgList &) const override;

private:
  bool IsIntegratedAsm = true;
};

} // namespace toolchains

namespace tools {
namespace baremetal {
namespace esp {

class LLVM_LIBRARY_VISIBILITY Linker : public Tool {
public:
  Linker(const ToolChain &TC) : Tool("baremetal::esp::Linker", "ld", TC) {}
  bool isLinkJob() const override { return true; }
  bool hasIntegratedCPP() const override { return false; }
  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};

class LLVM_LIBRARY_VISIBILITY Assembler : public Tool {
public:
  Assembler(const ToolChain &TC)
      : Tool("baremetal::esp::Assembler", "as", TC) {}

  bool hasIntegratedCPP() const override { return false; }
  void ConstructJob(Compilation &C, const JobAction &JA,
                    const InputInfo &Output, const InputInfoList &Inputs,
                    const llvm::opt::ArgList &TCArgs,
                    const char *LinkingOutput) const override;
};

void getXtensaTargetFeatures(const Driver &D, const llvm::Triple &Triple,
                             const llvm::opt::ArgList &Args,
                             std::vector<llvm::StringRef> &Features);

} // namespace esp
} // namespace baremetal
} // namespace tools

} // namespace driver
} // namespace clang

#endif
