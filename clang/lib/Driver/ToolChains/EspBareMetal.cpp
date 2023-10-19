//===-- BareMetal.cpp - Bare Metal ToolChain --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "EspBareMetal.h"

#include "CommonArgs.h"
#include "Gnu.h"
#include "clang/Driver/InputInfo.h"

#include "Arch/RISCV.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/MultilibBuilder.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::opt;
using namespace clang;
using namespace clang::driver;
using namespace clang::driver::tools;
using namespace clang::driver::toolchains;


EspBareMetal::EspBareMetal(const Driver &D, const llvm::Triple &Triple,
                     const ArgList &Args)
    : BareMetal(D, Triple, Args, false) {

  findMultilibs(D, Triple, Args);
  SmallString<128> SysRoot(computeSysRoot());
  if (!SysRoot.empty()) {
    for (const Multilib &M : getOrderedMultilibs()) {
      SmallString<128> Dir(SysRoot);
      llvm::sys::path::append(Dir, M.osSuffix(), "lib");
      getFilePaths().push_back(std::string(Dir));
      getLibraryPaths().push_back(std::string(Dir));
    }
  }

  // TODO: Add full support for Xtensa to integrated asm
  // LLVM-290, LLVM-291
  if (Triple.getArch() == llvm::Triple::xtensa) {
    for (auto *A : Args) {
      std::string Str = A->getAsString(Args);
      // Currently don't use integrated assembler for assembler input files
      if ((IsIntegratedAsm) && (Str.length() > 2)) {
        std::string ExtSubStr = Str.substr(Str.length() - 2);
        if (!ExtSubStr.compare(".s"))
          IsIntegratedAsm = false;
        if (!ExtSubStr.compare(".S"))
          IsIntegratedAsm = false;
      }
    }
    if (IsIntegratedAsm) {
      if (Args.getLastArgValue(options::OPT_x).equals("assembler"))
        IsIntegratedAsm = false;

      if (Args.getLastArgValue(options::OPT_x).equals("assembler-with-cpp"))
        IsIntegratedAsm = false;
    }
  }
}

void EspBareMetal::AddClangCXXStdlibIncludeArgs(const ArgList &DriverArgs,
                                             ArgStringList &CC1Args) const {
  if (DriverArgs.hasArg(options::OPT_nostdinc) ||
      DriverArgs.hasArg(options::OPT_nostdlibinc) ||
      DriverArgs.hasArg(options::OPT_nostdincxx))
    return;

  const Driver &D = getDriver();
  std::string SysRoot(computeSysRoot());
  if (SysRoot.empty())
    return;

  BareMetal::AddClangCXXStdlibIncludeArgs(DriverArgs, CC1Args);
  
  // add include path to non-multilib libsdtcxx headers
  if(GetCXXStdlibType(DriverArgs) == ToolChain::CST_Libstdcxx) {
    // computed SysRoot points to clang-runtimes
    SmallString<128> Dir(SysRoot);
    llvm::sys::path::append(Dir, getTriple().str(), "include", "c++");
    DetectAndAppendGCCVersion(D, Dir);
    addSystemInclude(DriverArgs, CC1Args, Dir.str());
  }
}

void EspBareMetal::AddCXXStdlibLibArgs(const ArgList &Args,
                                    ArgStringList &CmdArgs) const {
  switch (GetCXXStdlibType(Args)) {
  case ToolChain::CST_Libcxx:
    CmdArgs.push_back("-lc++");
    if (Args.hasArg(options::OPT_fexperimental_library))
      CmdArgs.push_back("-lc++experimental");
    CmdArgs.push_back("-lc++abi");
    CmdArgs.push_back("-lunwind");
    break;
  case ToolChain::CST_Libstdcxx:
    CmdArgs.push_back("-lstdc++");
    break;
  }
}

static void getRISCV32MultilibFlags(const llvm::Triple &Triple,
                                          const llvm::opt::ArgList &Args,
                                          Multilib::flags_list &Result) {

  Result.push_back("-march=" + riscv::getRISCVArch(Args, Triple).str());
  Result.push_back("-mabi=" + riscv::getRISCVABI(Args, Triple).str());
}

static void getXtensaMultilibFlags(const llvm::Triple &Triple,
                                          const llvm::opt::ArgList &Args,
                                          Multilib::flags_list &Result) {

  if (Arg *A = Args.getLastArg(options::OPT_mcpu_EQ))
    Result.push_back(A->getAsString(Args));
  else
    Result.push_back("-mcpu=esp32");

  if (Args.hasArg(options::OPT_mfix_esp32_psram_cache_issue))
    Result.push_back("-mfix-esp32-psram-cache-issue");
}

Multilib::flags_list
EspBareMetal::getMultilibFlags(const llvm::opt::ArgList &Args) const {

  std::vector<std::string> Result;
  const llvm::Triple Triple(ComputeEffectiveClangTriple(Args));

  if (Triple.getVendor() != llvm::Triple::Espressif)
    return BareMetal::getMultilibFlags(Args);

  Result.push_back("--target=" + Triple.str());

  if (Args.hasArg(options::OPT_fno_rtti))
    Result.push_back("-fno-rtti");

  switch (Triple.getArch()) {
  case llvm::Triple::riscv32:
    getRISCV32MultilibFlags(Triple, Args, Result);
    break;
  case llvm::Triple::xtensa:
    getXtensaMultilibFlags(Triple, Args, Result);
    break;
  default:
    break;
  }

  // Sort and remove duplicates.
  std::sort(Result.begin(), Result.end());
  Result.erase(std::unique(Result.begin(), Result.end()), Result.end());
  return Result;
}

Tool *EspBareMetal::buildLinker() const {
  return new tools::baremetal::esp::Linker(*this);
}

Tool *EspBareMetal::buildAssembler() const {
  return new tools::baremetal::esp::Assembler(*this);
}

void baremetal::esp::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                     const InputInfo &Output,
                                     const InputInfoList &Inputs,
                                     const ArgList &Args,
                                     const char *LinkingOutput) const {
  auto &ToolChain = static_cast<const toolchains::EspBareMetal &>(getToolChain());
  const Driver &D = ToolChain.getDriver();
  ArgStringList CmdArgs;

  if (!D.SysRoot.empty())
    CmdArgs.push_back(Args.MakeArgString("--sysroot=" + D.SysRoot));

  if (ToolChain.getArch() == llvm::Triple::riscv32) {
    CmdArgs.push_back("-m");
    CmdArgs.push_back("elf32lriscv");
  }

  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  CmdArgs.push_back("-X");

  std::string Linker = ToolChain.GetLinkerPath();

  bool WantCRTs =
      !Args.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles);

  if (WantCRTs) {
    /* Espressif toolcahin uses newlib. crt0.o from it refers to 'main' symbol.
       In 'freestanding' mode 'main' is not marked as special symbol by clang,
       so when compiling C++ program with 'clang++' 'main' gets mmangled
       (if not decalred as 'extern "C"' ) and linker can not resolve it.
       The problem can happen, for example, when cmake checks C++ compiler by buiding simple C++ code,
       unfortunately 'main' function in that code is not decalred as 'extern "C"'. */
    bool Freestanding =
        Args.hasFlag(options::OPT_ffreestanding, options::OPT_fhosted, false);
    if (!Freestanding) {
      CmdArgs.push_back(Args.MakeArgString(ToolChain.GetFilePath("crt0.o")));
    }
  }

  AddLinkerInputs(ToolChain, Inputs, Args, CmdArgs, JA);

  Args.AddAllArgs(CmdArgs, options::OPT_L);
  Args.AddAllArgs(CmdArgs, options::OPT_u);
  ToolChain.AddFilePathLibArgs(Args, CmdArgs);
  Args.AddAllArgs(CmdArgs,
                  {options::OPT_T_Group, options::OPT_e, options::OPT_s,
                   options::OPT_t, options::OPT_Z_Flag, options::OPT_r});

  // TODO: add C++ includes and libs if compiling C++.

  if (!Args.hasArg(options::OPT_nostdlib) &&
      !Args.hasArg(options::OPT_nodefaultlibs)) {
    if (ToolChain.ShouldLinkCXXStdlib(Args))
      ToolChain.AddCXXStdlibLibArgs(Args, CmdArgs);
    CmdArgs.push_back("-lm");
    CmdArgs.push_back("--start-group");
    CmdArgs.push_back("-lc");
    CmdArgs.push_back("-lgloss");
    CmdArgs.push_back("-lnosys");
    CmdArgs.push_back("--end-group");
    ToolChain.AddLinkRuntimeLib(Args, CmdArgs);
  }

  C.addCommand(std::make_unique<Command>(
      JA, *this, ResponseFileSupport::AtFileCurCP(), Args.MakeArgString(Linker),
      CmdArgs, Inputs, Output));
}

void baremetal::esp::Assembler::ConstructJob(Compilation &C, const JobAction &JA,
                                            const InputInfo &Output,
                                            const InputInfoList &Inputs,
                                            const ArgList &Args,
                                            const char *LinkingOutput) const {
  const auto &TC = static_cast<const toolchains::EspBareMetal &>(getToolChain());

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

  std::string AsmPrefix;
  if (TC.getTriple().getArch() == llvm::Triple::xtensa) {
    StringRef cpu = Args.getLastArgValue(options::OPT_mcpu_EQ, "esp32");
    // xtensa-esp32-elf
    AsmPrefix = TC.getTriple().getArchName().str() + "-" + cpu.str() + "-" + 
              TC.getTriple().getEnvironmentName().str();
  } else {
    // riscv32-esp-elf
    if (Args.hasArg(options::OPT_march_EQ))
      Args.AddAllArgs(CmdArgs, options::OPT_march_EQ);
    else
      CmdArgs.push_back("-march=rv32imac");
    if (Args.hasArg(options::OPT_mabi_EQ))
      Args.AddAllArgs(CmdArgs, options::OPT_mabi_EQ);
    else
      CmdArgs.push_back("-mabi=ilp32");
    AsmPrefix = TC.getTriple().getArchName().str() + "-" + 
              TC.getTriple().getVendorName().str() + "-" + 
              TC.getTriple().getEnvironmentName().str();
  }
  SmallString<128> Asm(AsmPrefix + "-" + getShortName());
  C.addCommand(
      std::make_unique<Command>(JA, *this, ResponseFileSupport::AtFileCurCP(),
                                Args.MakeArgString(TC.GetProgramPath(Asm.str().str().c_str())), CmdArgs, Inputs));
}
