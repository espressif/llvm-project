//===--- Xtensa.cpp - Xtensa ToolChain Implementations ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Xtensa.h"
#include "CommonArgs.h"
#include "clang/Basic/Cuda.h"
#include "clang/Config/config.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Distro.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/MultilibBuilder.h"
#include "clang/Driver/Options.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/TargetParser/XtensaTargetParser.h"
#include <system_error>

using namespace clang::driver;
using namespace clang::driver::tools;
using namespace clang::driver::toolchains;
using namespace clang;
using namespace llvm::opt;

using tools::addMultilibFlag;

/// Xtensa Toolchain
XtensaToolChain::XtensaToolChain(const Driver &D, const llvm::Triple &Triple,
                                 const ArgList &Args)
    : Generic_ELF(D, Triple, Args) {

  std::vector<std::string> ExtraAliases;

  if (Triple.getVendor() == llvm::Triple::Espressif) {
    std::string ESPCpuName = "esp32";
    if (Arg *A = Args.getLastArg(options::OPT_mcpu_EQ)) {
      ESPCpuName = A->getValue();
    }
    ExtraAliases = {std::string("xtensa-") + ESPCpuName + "-elf"};
    if (Args.hasArg(options::OPT_v)) {
      llvm::errs() << "Use GCC target extra alias: " << ExtraAliases[0] << "\n";
    }
  }

  GCCInstallation.init(Triple, Args);
  if (!GCCInstallation.isValid()) {
    llvm_unreachable("Unexpected Xtensa GCC toolchain version");
  }

  Multilibs = GCCInstallation.getMultilibs();
  SelectedMultilibs.assign({GCCInstallation.getMultilib()});

  GCCLibAndIncVersion = GCCInstallation.getVersion().Text;
  GCCToolchainName = GCCInstallation.getTriple().str();
  SmallString<128> Path(GCCInstallation.getParentLibPath());
  llvm::sys::path::append(Path, "..");
  GCCToolchainDir = Path.c_str();

  for (auto *A : Args) {
    std::string Str = A->getAsString(Args);
    if (!Str.compare("-mlongcalls"))
      A->claim();
    if (!Str.compare("-fno-tree-switch-conversion"))
      A->claim();

    // Currently don't use integrated assembler for assembler input files
    if ((IsIntegratedAsm) && (Str.length() > 2)) {
      std::string ExtSubStr = Str.substr(Str.length() - 2);
      if (!ExtSubStr.compare(".s"))
        IsIntegratedAsm = false;
      if (!ExtSubStr.compare(".S"))
        IsIntegratedAsm = false;
    }
  }

  // Currently don't use integrated assembler for assembler input files
  if (IsIntegratedAsm) {
    if (Args.getLastArgValue(options::OPT_x) == "assembler")
      IsIntegratedAsm = false;

    if (Args.getLastArgValue(options::OPT_x) == "assembler-with-cpp")
      IsIntegratedAsm = false;
  }

  SmallString<128> Libs1(GCCToolchainDir);
  llvm::sys::path::append(Libs1, "lib", "gcc", GCCToolchainName,
                          GCCLibAndIncVersion);
  if (!SelectedMultilibs.back().gccSuffix().empty())
    llvm::sys::path::append(Libs1, SelectedMultilibs.back().gccSuffix());
  getFilePaths().push_back(Libs1.c_str());

  SmallString<128> Libs2(GCCToolchainDir);
  llvm::sys::path::append(Libs2, GCCToolchainName, "lib");
  if (!SelectedMultilibs.back().gccSuffix().empty())
    llvm::sys::path::append(Libs2, SelectedMultilibs.back().gccSuffix());
  getFilePaths().push_back(Libs2.c_str());
}


Tool *XtensaToolChain::buildLinker() const {
  return new tools::xtensa::Linker(*this);
}

Tool *XtensaToolChain::buildAssembler() const {
  return new tools::xtensa::Assembler(*this);
}

void XtensaToolChain::AddClangSystemIncludeArgs(const ArgList &DriverArgs,
                                                ArgStringList &CC1Args) const {
  if (DriverArgs.hasArg(clang::driver::options::OPT_nostdinc) ||
      DriverArgs.hasArg(options::OPT_nostdlibinc))
    return;

  if (!GCCInstallation.isValid())
    return;

  SmallString<128> Path1(getDriver().ResourceDir);
  llvm::sys::path::append(Path1, "include");
  SmallString<128> Path2(GCCToolchainDir);
  llvm::sys::path::append(Path2, GCCToolchainName, "sys-include");
  SmallString<128> Path3(GCCToolchainDir);
  llvm::sys::path::append(Path3, GCCToolchainName, "include");

  const StringRef Paths[] = {Path1, Path2, Path3};
  addSystemIncludes(DriverArgs, CC1Args, Paths);
}

void XtensaToolChain::addLibStdCxxIncludePaths(
    const llvm::opt::ArgList &DriverArgs,
    llvm::opt::ArgStringList &CC1Args) const {
  if (!GCCInstallation.isValid())
    return;

  SmallString<128> BaseDir(GCCToolchainDir);
  llvm::sys::path::append(BaseDir, GCCToolchainName, "include", "c++",
                          GCCLibAndIncVersion);
  SmallString<128> TargetDir(BaseDir);
  llvm::sys::path::append(TargetDir, GCCToolchainName);
  SmallString<128> TargetDirBackward(BaseDir);
  llvm::sys::path::append(TargetDirBackward, "backward");

  addLibStdCXXIncludePaths(BaseDir, "", "", DriverArgs, CC1Args);
  addLibStdCXXIncludePaths(TargetDir, "", "", DriverArgs, CC1Args);
  addLibStdCXXIncludePaths(TargetDirBackward, "", "", DriverArgs, CC1Args);
}

ToolChain::CXXStdlibType
XtensaToolChain::GetCXXStdlibType(const ArgList &Args) const {
  Arg *A = Args.getLastArg(options::OPT_stdlib_EQ);
  if (!A)
    return ToolChain::CST_Libstdcxx;

  StringRef Value = A->getValue();
  if (Value != "libstdc++")
    getDriver().Diag(diag::err_drv_invalid_stdlib_name) << A->getAsString(Args);

  return ToolChain::CST_Libstdcxx;
}

ToolChain::UnwindLibType
XtensaToolChain::GetUnwindLibType(const llvm::opt::ArgList &Args) const {
  return ToolChain::UNW_None;
}

const StringRef XtensaToolChain::GetTargetCPUVersion(const ArgList &Args) {
  if (Arg *A = Args.getLastArg(clang::driver::options::OPT_mcpu_EQ)) {
    StringRef CPUName = A->getValue();
    return CPUName;
  }
  return "esp32";
}

void tools::xtensa::Assembler::ConstructJob(Compilation &C, const JobAction &JA,
                                            const InputInfo &Output,
                                            const InputInfoList &Inputs,
                                            const ArgList &Args,
                                            const char *LinkingOutput) const {
  const auto &TC =
      static_cast<const toolchains::XtensaToolChain &>(getToolChain());

  if (TC.GCCToolchainName == "")
    llvm_unreachable("Unable to find Xtensa GCC assembler");

  claimNoWarnArgs(Args);
  ArgStringList CmdArgs;

  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  CmdArgs.push_back("-c");

  if (Args.hasArg(options::OPT_v))
    CmdArgs.push_back("-v");

  if (Arg *A = Args.getLastArg(options::OPT_g_Group))
    if (!A->getOption().matches(options::OPT_g0))
      CmdArgs.push_back("-g");

  if (Args.hasFlag(options::OPT_fverbose_asm, options::OPT_fno_verbose_asm,
                   false))
    CmdArgs.push_back("-fverbose-asm");

  Args.AddAllArgValues(CmdArgs, options::OPT_Wa_COMMA, options::OPT_Xassembler);

  for (const auto &II : Inputs)
    CmdArgs.push_back(II.getFilename());

  SmallString<128> Asm(TC.GCCToolchainDir);
  llvm::sys::path::append(Asm, "bin",
                          TC.GCCToolchainName + "-" + getShortName());

  C.addCommand(
      std::make_unique<Command>(JA, *this, ResponseFileSupport::AtFileCurCP(),
                                Args.MakeArgString(Asm), CmdArgs, Inputs));
}

void xtensa::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                  const InputInfo &Output,
                                  const InputInfoList &Inputs,
                                  const ArgList &Args,
                                  const char *LinkingOutput) const {
  ArgStringList CmdArgs;
  SmallString<128> Linker;
  bool WantCRTs =
      !Args.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles);
  const auto &ToolChain =
      static_cast<const toolchains::XtensaToolChain &>(getToolChain());  

  if (ToolChain.GCCToolchainName == "")
    llvm_unreachable("Unable to find Xtensa GCC linker");

  if (!Args.getLastArgValue(options::OPT_fuse_ld_EQ).empty()) {
    Linker.assign(ToolChain.GetLinkerPath());
  } else {
    Linker.assign(ToolChain.GCCToolchainDir);
    llvm::sys::path::append(Linker, "bin",
                            ToolChain.GCCToolchainName + "-" + getShortName());
  }

  const char *crtbegin, *crtend;
  auto RuntimeLib = ToolChain.GetRuntimeLibType(Args);
  if (RuntimeLib == ToolChain::RLT_Libgcc) {
    crtbegin = "crtbegin.o";
    crtend = "crtend.o";
  } else {
    assert (RuntimeLib == ToolChain::RLT_CompilerRT);
    crtbegin = ToolChain.getCompilerRTArgString(Args, "crtbegin",
                                                ToolChain::FT_Object);
    crtend = ToolChain.getCompilerRTArgString(Args, "crtend",
                                              ToolChain::FT_Object);
  }

  if (WantCRTs) {
    // TODO: The crt0.o is not used for esp targets, but maybe used in
    // future for other vendors
    //CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath("crt0.o")));
    CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath(crtbegin)));
  }

  AddLinkerInputs(getToolChain(), Inputs, Args, CmdArgs, JA);

  Args.AddAllArgs(CmdArgs, options::OPT_L);
  ToolChain.AddFilePathLibArgs(Args, CmdArgs);
  Args.addAllArgs(CmdArgs,
                  {options::OPT_T_Group, options::OPT_e, options::OPT_s,
                   options::OPT_t, options::OPT_u_Group});
  
  if (!Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nodefaultlibs)) {
    if (ToolChain.ShouldLinkCXXStdlib(Args))
      ToolChain.AddCXXStdlibLibArgs(Args, CmdArgs);
    AddRunTimeLibs(ToolChain, ToolChain.getDriver(), CmdArgs, Args);
  }

  if (WantCRTs)
    CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath(crtend)));

  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());
  C.addCommand(std::make_unique<Command>(
      JA, *this, ResponseFileSupport::AtFileCurCP(), Args.MakeArgString(Linker), 
      CmdArgs, Inputs));
}

// Get features by CPU name
static void getXtensaFeaturesFromMcpu(const Driver &D,
                                      const llvm::opt::ArgList &Args,
                                      const llvm::opt::Arg *A, StringRef Mcpu,
                                      std::vector<StringRef> &Features) {
  if (llvm::Xtensa::parseCPUKind(Mcpu) == llvm::Xtensa::CK_INVALID) {
    D.Diag(clang::diag::err_drv_clang_unsupported) << A->getAsString(Args);
  } else {
    SmallVector<StringRef, 16> CPUFeatures;
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
