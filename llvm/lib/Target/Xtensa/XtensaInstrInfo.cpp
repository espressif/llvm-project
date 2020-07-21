//===- XtensaInstrInfo.cpp - Xtensa Instruction Information ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the Xtensa implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "XtensaInstrInfo.h"
#include "XtensaTargetMachine.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

#define GET_INSTRINFO_CTOR_DTOR
#include "XtensaGenInstrInfo.inc"

using namespace llvm;

static inline const MachineInstrBuilder &
addFrameReference(const MachineInstrBuilder &MIB, int FI) {
  MachineInstr *MI = MIB;
  MachineFunction &MF = *MI->getParent()->getParent();
  MachineFrameInfo &MFFrame = MF.getFrameInfo();
  const MCInstrDesc &MCID = MI->getDesc();
  MachineMemOperand::Flags Flags = MachineMemOperand::MONone;
  if (MCID.mayLoad())
    Flags |= MachineMemOperand::MOLoad;
  if (MCID.mayStore())
    Flags |= MachineMemOperand::MOStore;
  int64_t Offset = 0;
  unsigned Align = MFFrame.getObjectAlignment(FI);

  MachineMemOperand *MMO =
      MF.getMachineMemOperand(MachinePointerInfo::getFixedStack(MF, FI, Offset),
                              Flags, MFFrame.getObjectSize(FI), Align);
  return MIB.addFrameIndex(FI).addImm(Offset).addMemOperand(MMO);
}

XtensaInstrInfo::XtensaInstrInfo(XtensaSubtarget &sti)
    : XtensaGenInstrInfo(Xtensa::ADJCALLSTACKDOWN, Xtensa::ADJCALLSTACKUP), RI(sti), STI(sti) {}

/// Adjust SP by Amount bytes.
void XtensaInstrInfo::adjustStackPtr(unsigned SP, int64_t Amount,
                                     MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator I) const {
  DebugLoc DL = I != MBB.end() ? I->getDebugLoc() : DebugLoc();

  if (Amount == 0)
    return;

  MachineRegisterInfo &RegInfo = MBB.getParent()->getRegInfo();
  const TargetRegisterClass *RC = &Xtensa::ARRegClass;

  // create virtual reg to store immediate
  unsigned Reg = RegInfo.createVirtualRegister(RC);

  if (isInt<8>(Amount)) // addi sp, sp, amount
    BuildMI(MBB, I, DL, get(Xtensa::ADDI), Reg).addReg(SP).addImm(Amount);
  else { // Expand immediate that doesn't fit in 12-bit.
    unsigned Reg1;
    loadImmediate(MBB, I, &Reg1, Amount);
    BuildMI(MBB, I, DL, get(Xtensa::ADD), Reg)
        .addReg(SP)
        .addReg(Reg1, RegState::Kill);
  }

  BuildMI(MBB, I, DL, get(Xtensa::OR), SP).addReg(Reg, RegState::Kill).addReg(Reg, RegState::Kill);
}

void XtensaInstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                                  MachineBasicBlock::iterator MBBI,
                                  const DebugLoc &DL, MCRegister DestReg,
                                  MCRegister SrcReg, bool KillSrc) const {
  // when we are copying a phys reg we want the bits for fp
  if (Xtensa::ARRegClass.contains(DestReg, SrcReg))
    BuildMI(MBB, MBBI, DL, get(Xtensa::OR), DestReg)
        .addReg(SrcReg, getKillRegState(KillSrc))
        .addReg(SrcReg, getKillRegState(KillSrc));
  else
    llvm_unreachable("Impossible reg-to-reg copy");
}

void XtensaInstrInfo::storeRegToStackSlot(MachineBasicBlock &MBB,
                                          MachineBasicBlock::iterator MBBI,
                                          unsigned SrcReg, bool isKill,
                                          int FrameIdx,
                                          const TargetRegisterClass *RC,
                                          const TargetRegisterInfo *TRI) const {
  DebugLoc DL = MBBI != MBB.end() ? MBBI->getDebugLoc() : DebugLoc();
  unsigned LoadOpcode, StoreOpcode;
  getLoadStoreOpcodes(RC, LoadOpcode, StoreOpcode, FrameIdx);
  addFrameReference(BuildMI(MBB, MBBI, DL, get(StoreOpcode))
                        .addReg(SrcReg, getKillRegState(isKill)),
                    FrameIdx);
}

void XtensaInstrInfo::loadRegFromStackSlot(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI, unsigned DestReg,
    int FrameIdx, const TargetRegisterClass *RC,
    const TargetRegisterInfo *TRI) const {
  DebugLoc DL = MBBI != MBB.end() ? MBBI->getDebugLoc() : DebugLoc();
  unsigned LoadOpcode, StoreOpcode;
  getLoadStoreOpcodes(RC, LoadOpcode, StoreOpcode, FrameIdx);
  addFrameReference(BuildMI(MBB, MBBI, DL, get(LoadOpcode), DestReg), FrameIdx);
}

void XtensaInstrInfo::getLoadStoreOpcodes(const TargetRegisterClass *RC,
                                          unsigned &LoadOpcode,
                                          unsigned &StoreOpcode,
                                          int64_t offset) const {
  if (RC == &Xtensa::ARRegClass) {
    LoadOpcode = Xtensa::L32I;
    StoreOpcode = Xtensa::S32I;
  } else
    llvm_unreachable("Unsupported regclass to load or store");
}

void XtensaInstrInfo::loadImmediate(MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator MBBI,
                                    unsigned *Reg, int64_t Value) const {
  DebugLoc DL = MBBI != MBB.end() ? MBBI->getDebugLoc() : DebugLoc();
  MachineRegisterInfo &RegInfo = MBB.getParent()->getRegInfo();
  const TargetRegisterClass *RC = &Xtensa::ARRegClass;

  // create virtual reg to store immediate
  *Reg = RegInfo.createVirtualRegister(RC);
  if (Value >= -2048 && Value <= 2047) {
    BuildMI(MBB, MBBI, DL, get(Xtensa::MOVI), *Reg).addImm(Value);
  } else if (Value >= -32768 && Value <= 32767) {
    int Low = Value & 0xFF;
    int High = Value & ~0xFF;

    BuildMI(MBB, MBBI, DL, get(Xtensa::MOVI), *Reg).addImm(Low);
    BuildMI(MBB, MBBI, DL, get(Xtensa::ADDMI), *Reg).addReg(*Reg).addImm(High);
  } else if (Value >= -4294967296LL && Value <= 4294967295LL) {
    // 32 bit arbirary constant
    MachineConstantPool *MCP = MBB.getParent()->getConstantPool();
    uint64_t UVal = ((uint64_t)Value) & 0xFFFFFFFFLL;
    const Constant *CVal = ConstantInt::get(
        Type::getInt32Ty(MBB.getParent()->getFunction().getContext()), UVal,
        false);
    unsigned Idx = MCP->getConstantPoolIndex(CVal, 2U);
    //	MCSymbol MSym
    BuildMI(MBB, MBBI, DL, get(Xtensa::L32R), *Reg).addConstantPoolIndex(Idx);
  } else {
    // use L32R to let assembler load immediate best
    // TODO replace to L32R
    llvm_unreachable("Unsupported load immediate value");
  }
}
