//===- XtensaRegisterInfo.cpp - Xtensa Register Information ---------------===//
//
//                     The LLVM Compiler Infrastructure
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
#include "XtensaInstrInfo.h"
#include "XtensaMachineFunctionInfo.h"
#include "XtensaSubtarget.h"
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
  Reserved.set(Xtensa::BREG);
  return Reserved;
}

void XtensaRegisterInfo::eliminateFI(MachineBasicBlock::iterator II,
                                     unsigned OpNo, int FrameIndex,
                                     uint64_t StackSize, int64_t SPOffset,
                                     RegScavenger *RS) const {
  MachineInstr &MI = *II;
  MachineFunction &MF = *MI.getParent()->getParent();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  MachineBasicBlock &MBB = *MI.getParent();
  DebugLoc DL = II->getDebugLoc();
  const XtensaInstrInfo &TII = *static_cast<const XtensaInstrInfo *>(
      MBB.getParent()->getSubtarget().getInstrInfo());

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
  int64_t Offset;

  Offset = SPOffset + (int64_t)StackSize;
  Offset += MI.getOperand(OpNo + 1).getImm();

  LLVM_DEBUG(errs() << "Offset     : " << Offset << "\n"
                    << "<--------->\n");

  bool Valid = false;
  switch (MI.getOpcode()) {
  case Xtensa::L8I_P:
  case Xtensa::L8UI:
  case Xtensa::S8I:
  case Xtensa::SPILL_BOOL:
  case Xtensa::RESTORE_BOOL:
    Valid = (Offset >= 0 && Offset <= 255);
    break;
  case Xtensa::L16SI:
  case Xtensa::L16UI:
  case Xtensa::S16I:
    Valid = (Offset >= 0 && Offset <= 510);
    break;
  case Xtensa::LEA_ADD:
    Valid = (Offset >= -128 && Offset <= 127);
    break;
  case Xtensa::AE_L64_I:
  case Xtensa::AE_S64_I:
  case Xtensa::AE_S32X2_I:
  case Xtensa::AE_L32X2_I:
  case Xtensa::AE_S16X4_I:
  case Xtensa::AE_L16X4_I:
  case Xtensa::AE_LALIGN64_I:
  case Xtensa::AE_SALIGN64_I:
    Valid = (Offset >= -64 && Offset <= 56);
    break;
  case Xtensa::AE_S64_IP:
  case Xtensa::AE_L64_IP:
  case Xtensa::AE_S32X2_IP:
  case Xtensa::AE_L32X2_IP:
  case Xtensa::AE_S16X4_IP:
  case Xtensa::AE_L16X4_IP:
    Valid = (Offset >= 0 && Offset <= 56);
    break;
  case Xtensa::AE_L16X2M_I:
  case Xtensa::AE_L16X2M_IU:
  case Xtensa::AE_L32F24_I:
  case Xtensa::AE_L32F24_IP:
  case Xtensa::AE_L32M_I:
  case Xtensa::AE_L32M_IU:
  case Xtensa::AE_L32_I:
  case Xtensa::AE_L32_IP:
  case Xtensa::AE_S16X2M_I:
  case Xtensa::AE_S16X2M_IU:
  case Xtensa::AE_S24RA64S_I:
  case Xtensa::AE_S24RA64S_IP:
  case Xtensa::AE_S32F24_L_I:
  case Xtensa::AE_S32F24_L_IP:
  case Xtensa::AE_S32M_I:
  case Xtensa::AE_S32M_IU:
  case Xtensa::AE_S32RA64S_I:
  case Xtensa::AE_S32RA64S_IP:
  case Xtensa::AE_S32_L_I:
  case Xtensa::AE_S32_L_IP:
    Valid = (Offset >= -32 && Offset <= 28);
    break;
  default:
    Valid = (Offset >= 0 && Offset <= 1020);
    break;
  }

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

  unsigned BRegBase = Xtensa::B0;
  switch (MI.getOpcode()) {
  case Xtensa::SPILL_BOOL: {
    Register TempAR = RS->scavengeRegister(&Xtensa::ARRegClass, II, 0);
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
    return;
  }
  case Xtensa::RESTORE_BOOL: {

    Register SrcAR = RS->scavengeRegister(&Xtensa::ARRegClass, II, 0);
    RS->setRegUsed(SrcAR);
    Register MaskAR = RS->scavengeRegister(&Xtensa::ARRegClass, II, 0);
    RS->setRegUsed(MaskAR);
    Register BRegAR = RS->scavengeRegister(&Xtensa::ARRegClass, II, 0);
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
    return;
  }
  default:
    break;
  }
  MI.getOperand(OpNo).ChangeToRegister(FrameReg, false, false, IsKill);
  MI.getOperand(OpNo + 1).ChangeToImmediate(Offset);
}

void XtensaRegisterInfo::eliminateFrameIndex(MachineBasicBlock::iterator II,
                                             int SPAdj, unsigned FIOperandNum,
                                             RegScavenger *RS) const {
  MachineInstr &MI = *II;
  MachineFunction &MF = *MI.getParent()->getParent();

  assert(RS && "Need register scavenger");

  LLVM_DEBUG(errs() << "\nFunction : " << MF.getName() << "\n";
             errs() << "<--------->\n"
                    << MI);

  int FrameIndex = MI.getOperand(FIOperandNum).getIndex();
  uint64_t stackSize = MF.getFrameInfo().getStackSize();
  int64_t spOffset = MF.getFrameInfo().getObjectOffset(FrameIndex);

  LLVM_DEBUG(errs() << "FrameIndex : " << FrameIndex << "\n"
                    << "spOffset   : " << spOffset << "\n"
                    << "stackSize  : " << stackSize << "\n");

  eliminateFI(MI, FIOperandNum, FrameIndex, stackSize, spOffset, RS);
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