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
#include "clang/Driver/InputInfo.h"
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
#include <system_error>

using namespace clang::driver;
using namespace clang::driver::tools;
using namespace clang::driver::toolchains;
using namespace clang;
using namespace llvm::opt;

using tools::addMultilibFlag;

XtensaGCCToolchainDetector::XtensaGCCToolchainDetector(
    const Driver &D, const llvm::Triple &HostTriple,
    const llvm::opt::ArgList &Args) {
  std::string InstalledDir;
  InstalledDir = D.Dir;
  StringRef CPUName = XtensaToolChain::GetTargetCPUVersion(Args);
  std::string Dir;
  std::string ToolchainName;
  std::string ToolchainDir;

  if (CPUName == "esp32")
    ToolchainName = "xtensa-esp32-elf";
  else if (CPUName == "esp32-s2")
    ToolchainName = "xtensa-esp32s2-elf";
  else if (CPUName == "esp8266")
    ToolchainName = "xtensa-lx106-elf";

  Slash = llvm::sys::path::get_separator().str();

  ToolchainDir = InstalledDir + Slash + "..";
  Dir = ToolchainDir + Slash + "lib" + Slash + "gcc" + Slash + ToolchainName +
        Slash;
  GCCLibAndIncVersion = "";

  if (D.getVFS().exists(Dir)) {
    std::error_code EC;
    for (llvm::vfs::directory_iterator LI = D.getVFS().dir_begin(Dir, EC), LE;
         !EC && LI != LE; LI = LI.increment(EC)) {
      StringRef VersionText = llvm::sys::path::filename(LI->path());
      auto GCCVersion = Generic_GCC::GCCVersion::Parse(VersionText);
      if (GCCVersion.Major == -1)
        continue;
      GCCLibAndIncVersion = GCCVersion.Text;
    }
    if (GCCLibAndIncVersion == "")
      llvm_unreachable("Unexpected Xtensa GCC toolchain version");

  } else {
    // Unable to find Xtensa GCC toolchain;
    GCCToolchainName = "";
    return;
  }
  GCCToolchainDir = ToolchainDir;
  GCCToolchainName = ToolchainName;
}

/// Xtensa Toolchain
XtensaToolChain::XtensaToolChain(const Driver &D, const llvm::Triple &Triple,
                                 const ArgList &Args)
    : Generic_ELF(D, Triple, Args), XtensaGCCToolchain(D, getTriple(), Args) {
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

  bool IsESP32 = XtensaToolChain::GetTargetCPUVersion(Args) == "esp32";
  Multilibs.push_back(Multilib());

  if (IsESP32)
    Multilibs.push_back(MultilibBuilder("esp32-psram", {}, {})
                            .flag("-mfix-esp32-psram-cache-issue")
                            .makeMultilib());

  Multilibs.push_back(MultilibBuilder("no-rtti", {}, {})
                          .flag("-frtti", /*Disallow=*/true)
                          .flag("-fno-rtti")
                          .makeMultilib());

  if (IsESP32)
    Multilibs.push_back(MultilibBuilder("esp32-psram/no-rtti", {}, {})
                            .flag("-fno-rtti")
                            .flag("-frtti", /*Disallow=*/true)
                            .flag("-mfix-esp32-psram-cache-issue")
                            .makeMultilib());

  Multilib::flags_list Flags;
  addMultilibFlag(
      Args.hasFlag(options::OPT_frtti, options::OPT_fno_rtti, false), "frtti",
      Flags);

  if (IsESP32)
    addMultilibFlag(Args.hasFlag(options::OPT_mfix_esp32_psram_cache_issue,
                                 options::OPT_mfix_esp32_psram_cache_issue,
                                 false),
                    "mfix-esp32-psram-cache-issue", Flags);

  Multilibs.select(Flags, SelectedMultilibs);

  const std::string Slash = XtensaGCCToolchain.Slash;
  std::string Libs =
      XtensaGCCToolchain.GCCToolchainDir + Slash + "lib" + Slash + "gcc" +
      Slash + XtensaGCCToolchain.GCCToolchainName + Slash +
      XtensaGCCToolchain.GCCLibAndIncVersion + SelectedMultilibs.back().gccSuffix();
  getFilePaths().push_back(Libs);

  Libs = XtensaGCCToolchain.GCCToolchainDir + Slash +
         XtensaGCCToolchain.GCCToolchainName + Slash + "lib" +
         SelectedMultilibs.back().gccSuffix();
  getFilePaths().push_back(Libs);
}

Tool *XtensaToolChain::buildLinker() const {
  return new tools::Xtensa::Linker(*this);
}

Tool *XtensaToolChain::buildAssembler() const {
  return new tools::Xtensa::Assembler(*this);
}

void XtensaToolChain::AddClangSystemIncludeArgs(const ArgList &DriverArgs,
                                                ArgStringList &CC1Args) const {
  if (DriverArgs.hasArg(clang::driver::options::OPT_nostdinc) ||
      DriverArgs.hasArg(options::OPT_nostdlibinc))
    return;

  if (!XtensaGCCToolchain.IsValid())
    return;

  std::string Slash = XtensaGCCToolchain.Slash;

  std::string Path1 = getDriver().ResourceDir.c_str() + Slash + "include";
  std::string Path2 = XtensaGCCToolchain.GCCToolchainDir + Slash +
                      XtensaGCCToolchain.GCCToolchainName + Slash +
                      "sys-include";
  std::string Path3 = XtensaGCCToolchain.GCCToolchainDir + Slash +
                      XtensaGCCToolchain.GCCToolchainName + Slash + "include";

  const StringRef Paths[] = {Path1, Path2, Path3};
  addSystemIncludes(DriverArgs, CC1Args, Paths);
}

void XtensaToolChain::addLibStdCxxIncludePaths(
    const llvm::opt::ArgList &DriverArgs,
    llvm::opt::ArgStringList &CC1Args) const {
  if (!XtensaGCCToolchain.IsValid())
    return;

  std::string Slash = XtensaGCCToolchain.Slash;

  std::string BaseDir = XtensaGCCToolchain.GCCToolchainDir + Slash +
                        XtensaGCCToolchain.GCCToolchainName + Slash +
                        "include" + Slash + "c++" + Slash +
                        XtensaGCCToolchain.GCCLibAndIncVersion;
  std::string TargetDir = BaseDir + Slash + XtensaGCCToolchain.GCCToolchainName;
  addLibStdCXXIncludePaths(BaseDir, "", "", DriverArgs, CC1Args);
  addLibStdCXXIncludePaths(TargetDir, "", "", DriverArgs, CC1Args);
  TargetDir = BaseDir + Slash + "backward";
  addLibStdCXXIncludePaths(TargetDir, "", "", DriverArgs, CC1Args);
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

const StringRef XtensaToolChain::GetTargetCPUVersion(const ArgList &Args) {
  if (Arg *A = Args.getLastArg(clang::driver::options::OPT_mcpu_EQ)) {
    StringRef CPUName = A->getValue();
    return CPUName;
  }
  return "esp32";
}

void tools::Xtensa::Assembler::ConstructJob(Compilation &C, const JobAction &JA,
                                            const InputInfo &Output,
                                            const InputInfoList &Inputs,
                                            const ArgList &Args,
                                            const char *LinkingOutput) const {
  const auto &TC =
      static_cast<const toolchains::XtensaToolChain &>(getToolChain());

  if (!TC.XtensaGCCToolchain.IsValid())
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

  std::string Slash = TC.XtensaGCCToolchain.Slash;

  const char *Asm =
      Args.MakeArgString(getToolChain().getDriver().Dir + Slash +
                         TC.XtensaGCCToolchain.GCCToolchainName + "-as");
  C.addCommand(std::make_unique<Command>(
      JA, *this, ResponseFileSupport::AtFileCurCP(), Asm, CmdArgs, Inputs));
}

void Xtensa::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                  const InputInfo &Output,
                                  const InputInfoList &Inputs,
                                  const ArgList &Args,
                                  const char *LinkingOutput) const {
  const auto &TC =
      static_cast<const toolchains::XtensaToolChain &>(getToolChain());
  std::string Slash = TC.XtensaGCCToolchain.Slash;

  if (!TC.XtensaGCCToolchain.IsValid())
    llvm_unreachable("Unable to find Xtensa GCC linker");

  std::string Linker = getToolChain().getDriver().Dir + Slash +
                       TC.XtensaGCCToolchain.GCCToolchainName + "-ld";
  ArgStringList CmdArgs;

  Args.AddAllArgs(CmdArgs, options::OPT_L);
  TC.AddFilePathLibArgs(Args, CmdArgs);

  Args.addAllArgs(CmdArgs,
                  {options::OPT_T_Group, options::OPT_e, options::OPT_s,
                   options::OPT_t, options::OPT_u_Group});

  AddLinkerInputs(getToolChain(), Inputs, Args, CmdArgs, JA);

  CmdArgs.push_back("-lgcc");

  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());
  C.addCommand(
      std::make_unique<Command>(JA, *this, ResponseFileSupport::AtFileCurCP(),
                                Args.MakeArgString(Linker), CmdArgs, Inputs));
}
