//===- XtensaTargetMachine.cpp - Define TargetMachine for Xtensa ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements the info about Xtensa target spec.
//
//===----------------------------------------------------------------------===//

#include "XtensaMachineFunctionInfo.h"
#include "XtensaTargetMachine.h"
#include "XtensaTargetObjectFile.h"
#include "XtensaTargetTransformInfo.h"
#include "TargetInfo/XtensaTargetInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Transforms/Scalar.h"
#include <optional>

using namespace llvm;

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeXtensaTarget() {
  // Register the target.
  RegisterTargetMachine<XtensaTargetMachine> A(getTheXtensaTarget());
}

static std::string computeDataLayout(const Triple &TT, StringRef CPU,
                                     const TargetOptions &Options,
                                     bool IsLittle) {
  std::string Ret = "e-m:e-p:32:32-v1:8:8-i64:64-i128:128-n32";
  return Ret;
}

static Reloc::Model getEffectiveRelocModel(bool JIT,
                                           std::optional<Reloc::Model> RM) {
  if (!RM || JIT)
     return Reloc::Static;
  return *RM;
}

static std::unique_ptr<TargetLoweringObjectFile> createTLOF() {
  return std::make_unique<XtensaElfTargetObjectFile>();
}

static StringRef getCPUName(StringRef CPU) {
  if (CPU.empty())
    CPU = "esp32";
  else if (CPU == "esp32-s2")
    CPU = "esp32s2";
  else if (CPU == "esp32-s3")
    CPU = "esp32s3";
  return CPU;
}

XtensaTargetMachine::XtensaTargetMachine(const Target &T, const Triple &TT,
                                         StringRef CPU, StringRef FS,
                                         const TargetOptions &Options,
                                         std::optional<Reloc::Model> RM,
                                         std::optional<CodeModel::Model> CM,
                                         CodeGenOptLevel OL, bool JIT,
                                         bool IsLittle)
    : LLVMTargetMachine(T, computeDataLayout(TT, CPU, Options, IsLittle), TT,
                        CPU, FS, Options, getEffectiveRelocModel(JIT, RM),
                        getEffectiveCodeModel(CM, CodeModel::Small), OL),
      TLOF(createTLOF()),
      Subtarget(TT, std::string(CPU), std::string(FS), *this) {
  initAsmInfo();
}

XtensaTargetMachine::XtensaTargetMachine(const Target &T, const Triple &TT,
                                         StringRef CPU, StringRef FS,
                                         const TargetOptions &Options,
                                         std::optional<Reloc::Model> RM,
                                         std::optional<CodeModel::Model> CM,
                                         CodeGenOptLevel OL, bool JIT)
    : XtensaTargetMachine(T, TT, getCPUName(CPU), FS, Options, RM, CM, OL, JIT, true) {}

const XtensaSubtarget *
XtensaTargetMachine::getSubtargetImpl(const Function &F) const {
  return &Subtarget;
}

TargetTransformInfo
XtensaTargetMachine::getTargetTransformInfo(const Function &F) const {
  return TargetTransformInfo(XtensaTTIImpl(this, F));
}

MachineFunctionInfo *XtensaTargetMachine::createMachineFunctionInfo(
    BumpPtrAllocator &Allocator, const Function &F,
    const TargetSubtargetInfo *STI) const {
  return XtensaFunctionInfo::create<XtensaFunctionInfo>(Allocator, F, STI);
}

namespace {
/// Xtensa Code Generator Pass Configuration Options.
class XtensaPassConfig : public TargetPassConfig {
public:
  XtensaPassConfig(XtensaTargetMachine &TM, PassManagerBase &PM)
      : TargetPassConfig(TM, PM) {}

  XtensaTargetMachine &getXtensaTargetMachine() const {
    return getTM<XtensaTargetMachine>();
  }

  void addIRPasses() override;
  bool addPreISel() override;
  bool addInstSelector() override;
  void addPreRegAlloc() override;
  void addPreEmitPass() override;
};
} // end anonymous namespace

bool XtensaPassConfig::addPreISel() {
  if (TM->getOptLevel() != CodeGenOptLevel::None) {
    addPass(createHardwareLoopsLegacyPass());
  }

  return false;
}

bool XtensaPassConfig::addInstSelector() {
  addPass(createXtensaISelDag(getXtensaTargetMachine(), getOptLevel()));
  return false;
}

void XtensaPassConfig::addIRPasses() {
    addPass(createAtomicExpandPass());
    TargetPassConfig::addIRPasses();
}

void XtensaPassConfig::addPreRegAlloc() {
  addPass(createXtensaHardwareLoops());
}

void XtensaPassConfig::addPreEmitPass() {
  addPass(createXtensaPSRAMCacheFixPass());
  addPass(createXtensaBRegFixupPass());
  addPass(createXtensaSizeReductionPass());
  addPass(createXtensaFixupHwLoops());
  addPass(&BranchRelaxationPassID);
  addPass(createXtensaConstantIslandPass());
}

TargetPassConfig *XtensaTargetMachine::createPassConfig(PassManagerBase &PM) {
  return new XtensaPassConfig(*this, PM);
}
