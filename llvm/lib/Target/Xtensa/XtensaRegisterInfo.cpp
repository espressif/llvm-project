//===- XtensaRegisterInfo.cpp - Xtensa Register Information ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the Xtensa implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#include "XtensaRegisterInfo.h"
#include "MCTargetDesc/XtensaMCTargetDesc.h"
#include "XtensaInstrInfo.h"
#include "XtensaMachineFunctionInfo.h"
#include "XtensaSubtarget.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "xtensa-reg-info"

#define GET_REGINFO_TARGET_DESC
#include "XtensaGenRegisterInfo.inc"

using namespace llvm;

XtensaRegisterInfo::XtensaRegisterInfo(const XtensaSubtarget &STI)
    : XtensaGenRegisterInfo(Xtensa::A0), Subtarget(STI) {}

const uint16_t *
XtensaRegisterInfo::getCalleeSavedRegs(const MachineFunction *MF) const {
  if (Subtarget.isWinABI())
    return CSRWE_Xtensa_SaveList;
  else
    return CSR_Xtensa_SaveList;
}

const uint32_t *
XtensaRegisterInfo::getCallPreservedMask(const MachineFunction &MF,
                                         CallingConv::ID) const {
  if (Subtarget.isWinABI())
    return CSRWE_Xtensa_RegMask;
  else
    return CSR_Xtensa_RegMask;
}

BitVector XtensaRegisterInfo::getReservedRegs(const MachineFunction &MF) const {
  BitVector Reserved(getNumRegs());
  const TargetFrameLowering *TFI = MF.getSubtarget().getFrameLowering();

  Reserved.set(Xtensa::A0);
  if (TFI->hasFP(MF)) {
    // Reserve frame pointer.
    Reserved.set(getFrameRegister(MF));
  }

  // Reserve stack pointer.
  Reserved.set(Xtensa::SP);
  //Reserve QR regs
  Reserved.set(Xtensa::Q0);
  Reserved.set(Xtensa::Q1);
  Reserved.set(Xtensa::Q2);
  Reserved.set(Xtensa::Q3);
  Reserved.set(Xtensa::Q4);
  Reserved.set(Xtensa::Q5);
  Reserved.set(Xtensa::Q6);
  Reserved.set(Xtensa::Q7);
  Reserved.set(Xtensa::BREG);
  return Reserved;
}

bool XtensaRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                             int SPAdj, unsigned FIOperandNum,
                                             RegScavenger *RS) const {
  MachineInstr &MI = *II;
  MachineFunction &MF = *MI.getParent()->getParent();
  int FrameIndex = MI.getOperand(FIOperandNum).getIndex();
  uint64_t StackSize = MF.getFrameInfo().getStackSize();
  int64_t SPOffset = MF.getFrameInfo().getObjectOffset(FrameIndex);
  MachineFrameInfo &MFI = MF.getFrameInfo();
  const std::vector<CalleeSavedInfo> &CSI = MFI.getCalleeSavedInfo();
  int MinCSFI = 0;
  int MaxCSFI = -1;

  if (CSI.size()) {
    MinCSFI = CSI[0].getFrameIdx();
    MaxCSFI = CSI[CSI.size() - 1].getFrameIdx();
  }
  // The following stack frame objects are always referenced relative to $sp:
  //  1. Outgoing arguments.
  //  2. Pointer to dynamically allocated stack space.
  //  3. Locations for callee-saved registers.
  //  4. Locations for eh data registers.
  // Everything else is referenced relative to whatever register
  // getFrameRegister() returns.
  unsigned FrameReg;
  if ((FrameIndex >= MinCSFI && FrameIndex <= MaxCSFI))
    FrameReg = Xtensa::SP;
  else
    FrameReg = getFrameRegister(MF);

  // Calculate final offset.
  // - There is no need to change the offset if the frame object is one of the
  //   following: an outgoing argument, pointer to a dynamically allocated
  //   stack space or a $gp restore location,
  // - If the frame object is any of the following, its offset must be adjusted
  //   by adding the size of the stack:
  //   incoming argument, callee-saved register location or local variable.
  bool IsKill = false;
  int64_t Offset =
      SPOffset + (int64_t)StackSize + MI.getOperand(FIOperandNum + 1).getImm();

  bool Valid = Xtensa::isValidAddrOffsetForOpcode(MI.getOpcode(), Offset);

  // If MI is not a debug value, make sure Offset fits in the 16-bit immediate
  // field.
  if (!MI.isDebugValue() && !Valid) {
    MachineBasicBlock &MBB = *MI.getParent();
    DebugLoc DL = II->getDebugLoc();
    unsigned ADD = Xtensa::ADD;
    unsigned Reg;
    const XtensaInstrInfo &TII = *static_cast<const XtensaInstrInfo *>(
        MBB.getParent()->getSubtarget().getInstrInfo());

    TII.loadImmediate(MBB, II, &Reg, Offset);
    BuildMI(MBB, II, DL, TII.get(ADD), Reg)
        .addReg(FrameReg)
        .addReg(Reg, RegState::Kill);

    FrameReg = Reg;
    Offset = 0;
    IsKill = true;
  }

  MachineBasicBlock &MBB = *MI.getParent();
  DebugLoc DL = II->getDebugLoc();
  const XtensaInstrInfo &TII = *static_cast<const XtensaInstrInfo *>(
      MBB.getParent()->getSubtarget().getInstrInfo());
  unsigned BRegBase = Xtensa::B0;

  switch (MI.getOpcode()) {
  case Xtensa::SPILL_BOOL: {
    Register TempAR =
        RS->scavengeRegisterBackwards(Xtensa::ARRegClass, II, false, 0);
    RS->setRegUsed(TempAR);

    BuildMI(MBB, II, DL, TII.get(Xtensa::RSR), TempAR).addReg(Xtensa::BREG);
    MachineOperand &Breg = MI.getOperand(0);
    unsigned RegNo = Breg.getReg().id() - BRegBase;

    BuildMI(MBB, II, DL, TII.get(Xtensa::EXTUI), TempAR)
        .addReg(TempAR)
        .addImm(RegNo)
        .addImm(1);

    BuildMI(MBB, II, DL, TII.get(Xtensa::S8I))
        .addReg(TempAR, RegState::Kill)
        .addReg(FrameReg, getKillRegState(IsKill))
        .addImm(Offset);

    MI.eraseFromParent();
    return true;
  }
  case Xtensa::RESTORE_BOOL: {

    Register SrcAR =
        RS->scavengeRegisterBackwards(Xtensa::ARRegClass, II, false, 0);
    RS->setRegUsed(SrcAR);
    Register MaskAR =
        RS->scavengeRegisterBackwards(Xtensa::ARRegClass, II, false, 0);
    RS->setRegUsed(MaskAR);
    Register BRegAR =
        RS->scavengeRegisterBackwards(Xtensa::ARRegClass, II, false, 0);
    RS->setRegUsed(BRegAR);

    MachineOperand &Breg = MI.getOperand(0);
    unsigned RegNo = Breg.getReg().id() - BRegBase;

    BuildMI(MBB, II, DL, TII.get(Xtensa::L8UI), SrcAR)
        .addReg(FrameReg, getKillRegState(IsKill))
        .addImm(Offset);

    BuildMI(MBB, II, DL, TII.get(Xtensa::EXTUI), SrcAR)
        .addReg(SrcAR)
        .addImm(0)
        .addImm(1);

    if (RegNo != 0) {
      BuildMI(MBB, II, DL, TII.get(Xtensa::SLLI), SrcAR)
          .addReg(SrcAR)
          .addImm(RegNo);
    }

    BuildMI(MBB, II, DL, TII.get(Xtensa::RSR), BRegAR).addReg(Xtensa::BREG);

    unsigned Mask = ~(1 << RegNo) & 0x3ff;
    BuildMI(MBB, II, DL, TII.get(Xtensa::MOVI), MaskAR)
        .addImm(RegNo < 12 ? Mask : 1);
    if (RegNo >= 12) {
      BuildMI(MBB, II, DL, TII.get(Xtensa::SLLI), MaskAR)
          .addReg(MaskAR)
          .addImm(RegNo);
    }
    BuildMI(MBB, II, DL, TII.get(Xtensa::AND), BRegAR)
        .addReg(BRegAR)
        .addReg(MaskAR);

    BuildMI(MBB, II, DL, TII.get(Xtensa::OR), BRegAR)
        .addReg(SrcAR)
        .addReg(BRegAR);

    BuildMI(MBB, II, DL, TII.get(Xtensa::WSR))
        .addReg(Xtensa::BREG, RegState::Define)
        .addReg(BRegAR)
        .addDef(Breg.getReg(), RegState::Implicit);

    MI.eraseFromParent();
    return true;
  }
  default:
    break;
  }

  MI.getOperand(FIOperandNum).ChangeToRegister(FrameReg, false, false, IsKill);
  MI.getOperand(FIOperandNum + 1).ChangeToImmediate(Offset);

  return false;
}

Register XtensaRegisterInfo::getFrameRegister(const MachineFunction &MF) const {
  const TargetFrameLowering *TFI = MF.getSubtarget().getFrameLowering();
  return TFI->hasFP(MF) ? (Subtarget.isWinABI() ? Xtensa::A7 : Xtensa::A15)
                        : Xtensa::SP;
}

bool XtensaRegisterInfo::requiresFrameIndexReplacementScavenging(
    const MachineFunction &MF) const {
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  return MFI.hasStackObjects();
}
