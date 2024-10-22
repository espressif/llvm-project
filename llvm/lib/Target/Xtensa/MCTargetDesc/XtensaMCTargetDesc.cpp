//===-- XtensaMCTargetDesc.cpp - Xtensa target descriptions ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "XtensaMCTargetDesc.h"
#include "XtensaInstPrinter.h"
#include "XtensaMCAsmInfo.h"
#include "XtensaTargetStreamer.h"
#include "TargetInfo/XtensaTargetInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/ErrorHandling.h"

#define GET_INSTRINFO_MC_DESC
#include "XtensaGenInstrInfo.inc"

#define GET_REGINFO_MC_DESC
#include "XtensaGenRegisterInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "XtensaGenSubtargetInfo.inc"

using namespace llvm;

static MCAsmInfo *createXtensaMCAsmInfo(const MCRegisterInfo &MRI,
                                        const Triple &TT,
                                        const MCTargetOptions &Options) {
  MCAsmInfo *MAI = new XtensaMCAsmInfo(TT);
  MCCFIInstruction Inst = MCCFIInstruction::cfiDefCfa(
      nullptr, MRI.getDwarfRegNum(Xtensa::SP, true), 0);
  MAI->addInitialFrameState(Inst);
  return MAI;
}

static MCInstrInfo *createXtensaMCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitXtensaMCInstrInfo(X);
  return X;
}

static MCInstPrinter *createXtensaMCInstPrinter(const Triple &TT,
                                                unsigned SyntaxVariant,
                                                const MCAsmInfo &MAI,
                                                const MCInstrInfo &MII,
                                                const MCRegisterInfo &MRI) {
  return new XtensaInstPrinter(MAI, MII, MRI);
}

static MCRegisterInfo *createXtensaMCRegisterInfo(const Triple &TT) {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitXtensaMCRegisterInfo(X, Xtensa::SP);
  return X;
}

static MCSubtargetInfo *
createXtensaMCSubtargetInfo(const Triple &TT, StringRef CPU, StringRef FS) {
  if (CPU.empty())
    CPU = "esp32";
  else if (CPU == "esp32-s2")
    CPU = "esp32s2";
  else if (CPU == "esp32-s3")
    CPU = "esp32s3";
  return createXtensaMCSubtargetInfoImpl(TT, CPU, CPU, FS);
}
 
static MCTargetStreamer *
createXtensaAsmTargetStreamer(MCStreamer &S, formatted_raw_ostream &OS,
                              MCInstPrinter *InstPrint, bool isVerboseAsm) {
  return new XtensaTargetAsmStreamer(S, OS);
}

static MCTargetStreamer *
createXtensaObjectTargetStreamer(MCStreamer &S, const MCSubtargetInfo &STI) {
  return new XtensaTargetELFStreamer(S);
}

extern "C" LLVM_EXTERNAL_VISIBILITY void LLVMInitializeXtensaTargetMC() {
  // Register the MCAsmInfo.
  TargetRegistry::RegisterMCAsmInfo(getTheXtensaTarget(), createXtensaMCAsmInfo);

  // Register the MCCodeEmitter.
  TargetRegistry::RegisterMCCodeEmitter(getTheXtensaTarget(),
                                        createXtensaMCCodeEmitter);

  // Register the MCInstrInfo.
  TargetRegistry::RegisterMCInstrInfo(getTheXtensaTarget(), createXtensaMCInstrInfo);

  // Register the MCInstPrinter.
  TargetRegistry::RegisterMCInstPrinter(getTheXtensaTarget(),
                                        createXtensaMCInstPrinter);

  // Register the MCRegisterInfo.
  TargetRegistry::RegisterMCRegInfo(getTheXtensaTarget(),
                                    createXtensaMCRegisterInfo);

  // Register the MCSubtargetInfo.
  TargetRegistry::RegisterMCSubtargetInfo(getTheXtensaTarget(),
                                          createXtensaMCSubtargetInfo);

  // Register the MCAsmBackend.
  TargetRegistry::RegisterMCAsmBackend(getTheXtensaTarget(),
                                       createXtensaMCAsmBackend);

  // Register the asm target streamer.
  TargetRegistry::RegisterAsmTargetStreamer(getTheXtensaTarget(),
                                            createXtensaAsmTargetStreamer);

  // Register the ELF target streamer.
  TargetRegistry::RegisterObjectTargetStreamer(
      getTheXtensaTarget(), createXtensaObjectTargetStreamer);
}
