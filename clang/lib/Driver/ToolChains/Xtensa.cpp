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
#include "clang/Driver/Options.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/TargetParser.h"
#include "llvm/Support/VirtualFileSystem.h"
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

  if (Triple.getVendor() == llvm::Triple::Espressif ||
      Triple.getVendor() == llvm::Triple::UnknownVendor) {
    Arg *mcpuArg = Args.getLastArg(options::OPT_mcpu_EQ);
    SmallString<128> CpuName;
    if (mcpuArg)
      CpuName = mcpuArg->getValue();
    else if (Triple.getVendor() == llvm::Triple::Espressif)
      // 'esp32' is default for 'xtensa-esp-xxx' targets,
      // for generic 'xtensa' target CPU should be always specified explicitly with '-mcpu'
      CpuName = "esp32";
    // Xtensa GCC toolchain uses shorten triple "xtensa-<cpu>-elf", so add it as
    // an alias to help Clang detect GCC installation properly
    ExtraAliases = {std::string("xtensa-") + CpuName.c_str() + "-elf"};
    if (Args.hasArg(options::OPT_v)) {
      llvm::errs() << "Use GCC target extra alias: " << ExtraAliases[0] << "\n";
    }
  }

  GCCInstallation.init(Triple, Args, ExtraAliases);
  if (GCCInstallation.isValid()) {
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
      if (Args.getLastArgValue(options::OPT_x).equals("assembler"))
        IsIntegratedAsm = false;

      if (Args.getLastArgValue(options::OPT_x).equals("assembler-with-cpp"))
        IsIntegratedAsm = false;
    }

    Multilibs = GCCInstallation.getMultilibs();
    SelectedMultilib = GCCInstallation.getMultilib();

    GCCLibAndIncVersion = GCCInstallation.getVersion().Text;
    GCCToolchainName = GCCInstallation.getTriple().str();
    SmallString<128> Path(GCCInstallation.getParentLibPath());
    llvm::sys::path::append(Path, "..");
    GCCToolchainDir = Path.c_str();

    SmallString<128> Libs1(GCCToolchainDir);
    llvm::sys::path::append(Libs1, "lib", "gcc", GCCToolchainName,
                            GCCLibAndIncVersion);
    if (!SelectedMultilib.gccSuffix().empty())
      llvm::sys::path::append(Libs1, SelectedMultilib.gccSuffix());
    getFilePaths().push_back(Libs1.c_str());

    SmallString<128> Libs2(GCCToolchainDir);
    llvm::sys::path::append(Libs2, GCCToolchainName, "lib");
    if (!SelectedMultilib.gccSuffix().empty())
      llvm::sys::path::append(Libs2, SelectedMultilib.gccSuffix());
    getFilePaths().push_back(Libs2.c_str());

    ToolChain::path_list &PPaths = getProgramPaths();
    // Multilib cross-compiler GCC installations put ld in a triple-prefixed
    // directory of the GCC installation parent dir.
    StringRef ParentDir = llvm::sys::path::parent_path(GCCInstallation.getParentLibPath());

    SmallString<128> PathTripleBin(ParentDir);
    llvm::sys::path::append(PathTripleBin, GCCInstallation.getTriple().str());
    llvm::sys::path::append(PathTripleBin, "bin");
    PPaths.push_back(PathTripleBin.c_str());

    SmallString<128> PathBin(ParentDir);
    llvm::sys::path::append(PathBin, "bin");
    PPaths.push_back(PathBin.c_str());

    if (!getDriver().SysRoot.empty()) {
      SmallString<128> SysRoot(computeSysRoot());
      llvm::sys::path::append(SysRoot, "lib");
      getFilePaths().push_back(SysRoot.c_str());
    }
  } else {
    getProgramPaths().push_back(D.Dir);
    SmallString<128> SysRoot(computeSysRoot());
    llvm::sys::path::append(SysRoot, "lib");
    getFilePaths().push_back(SysRoot.c_str());
  }
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

  if (!getDriver().SysRoot.empty()) {
    SmallString<128> Dir(getDriver().SysRoot);
    llvm::sys::path::append(Dir, "include");
    addSystemInclude(DriverArgs, CC1Args, Dir.str());
  } else if (GCCInstallation.isValid()) {
    SmallString<128> Path1(getDriver().ResourceDir);
    llvm::sys::path::append(Path1, "include");
    SmallString<128> Path2(GCCToolchainDir);
    llvm::sys::path::append(Path2, GCCToolchainName, "sys-include");
    SmallString<128> Path3(GCCToolchainDir);
    llvm::sys::path::append(Path3, GCCToolchainName, "include");

    const StringRef Paths[] = {Path1, Path2, Path3};
    addSystemIncludes(DriverArgs, CC1Args, Paths);
  } else {
    SmallString<128> Dir(computeSysRoot());
    llvm::sys::path::append(Dir, "include");
    addSystemInclude(DriverArgs, CC1Args, Dir.str());
  }
}

void XtensaToolChain::addLibStdCxxIncludePaths(
    const llvm::opt::ArgList &DriverArgs,
    llvm::opt::ArgStringList &CC1Args) const {
  if (!GCCInstallation.isValid())
    return;

  const GCCVersion &Version = GCCInstallation.getVersion();
  StringRef TripleStr = GCCInstallation.getTriple().str();
  addLibStdCXXIncludePaths(computeSysRoot() + "/include/c++/" + Version.Text,
                           TripleStr, "", DriverArgs, CC1Args);
}

std::string XtensaToolChain::computeSysRoot() const {
  if (!getDriver().SysRoot.empty())
    return getDriver().SysRoot;

  SmallString<128> SysRootDir;
  if (GCCInstallation.isValid()) {
    StringRef LibDir = GCCInstallation.getParentLibPath();
    StringRef TripleStr = GCCInstallation.getTriple().str();
    llvm::sys::path::append(SysRootDir, LibDir, "..", TripleStr);
  } else {
    // Use the triple as provided to the driver. Unlike the parsed triple
    // this has not been normalized to always contain every field.
    llvm::sys::path::append(SysRootDir, getDriver().Dir, "..",
                            getDriver().getTargetTriple());
  }

  if (!llvm::sys::fs::exists(SysRootDir))
    return std::string();

  return std::string(SysRootDir.str());
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

  if (Args.getLastArg(options::OPT_mtext_section_literals))
    CmdArgs.push_back("--text-section-literals");

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
  const Driver &D = ToolChain.getDriver();

  if (!D.SysRoot.empty())
    CmdArgs.push_back(Args.MakeArgString("--sysroot=" + D.SysRoot));

  bool LinkerIsLLD;
  std::string LinkerPath = ToolChain.GetLinkerPath(&LinkerIsLLD);
  if (ToolChain.GCCToolchainName != "") {
    if (!LinkerIsLLD) {
      Linker.assign(ToolChain.GCCToolchainDir);
      llvm::sys::path::append(
          Linker, "bin", ToolChain.GCCToolchainName + "-" + getShortName());
    } else {
      Linker.assign(LinkerPath);
    }
  } else {
    Linker.assign(LinkerPath);
  }

  const char *crtbegin, *crtend;
  auto RuntimeLib = ToolChain.GetRuntimeLibType(Args);
  if (RuntimeLib == ToolChain::RLT_Libgcc) {
    crtbegin = "crtbegin.o";
    crtend = "crtend.o";
  } else {
    assert(RuntimeLib == ToolChain::RLT_CompilerRT);
    crtbegin = ToolChain.getCompilerRTArgString(Args, "crtbegin",
                                                ToolChain::FT_Object);
    crtend =
        ToolChain.getCompilerRTArgString(Args, "crtend", ToolChain::FT_Object);
  }

  if (WantCRTs) {
    // TODO: The crt0.o is not used for esp targets, but maybe used in
    // future for other vendors
    // CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath("crt0.o")));
    CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath(crtbegin)));
  }

  AddLinkerInputs(getToolChain(), Inputs, Args, CmdArgs, JA);

  Args.AddAllArgs(CmdArgs, options::OPT_L);
  ToolChain.AddFilePathLibArgs(Args, CmdArgs);
  Args.AddAllArgs(CmdArgs,
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
  C.addCommand(
      std::make_unique<Command>(JA, *this, ResponseFileSupport::AtFileCurCP(),
                                Args.MakeArgString(Linker), CmdArgs, Inputs));
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
void xtensa::getXtensaTargetFeatures(const Driver &D, const ArgList &Args,
                                     std::vector<StringRef> &Features) {
  if (Arg *A = Args.getLastArg(options::OPT_mcpu_EQ))
    getXtensaFeaturesFromMcpu(D, Args, A, A->getValue(), Features);
}
