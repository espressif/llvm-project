//===- XtensaBRegFixup.cpp - Xtensa boolean register fixup ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Xtensa.h"
#include "XtensaInstrInfo.h"
#include "XtensaSubtarget.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen//MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

#define DEBUG_TYPE "xtensa-breg-fixup-pass"

class XtensaBRegFixup : public MachineFunctionPass {
public:
  static char ID;
  XtensaBRegFixup() : MachineFunctionPass(ID) {}

  const XtensaSubtarget *Subtarget;
  static const XtensaInstrInfo *XtensaII;

  bool runOnMachineFunction(MachineFunction &MF) override;

  llvm::StringRef getPassName() const override {
    return "Xtensa bool reg fixup pass";
  }

private:
  bool VisitMBB(MachineBasicBlock &MBB);
  bool VisitInstruction(const MachineBasicBlock::instr_iterator &MII);
};

char XtensaBRegFixup::ID = 0;
const XtensaInstrInfo *XtensaBRegFixup::XtensaII;

bool XtensaBRegFixup::runOnMachineFunction(MachineFunction &MF) {

  Subtarget = &static_cast<const XtensaSubtarget &>(MF.getSubtarget());
  XtensaII = static_cast<const XtensaInstrInfo *>(Subtarget->getInstrInfo());
  bool Modified = false;

  MachineFunction::iterator I = MF.begin(), E = MF.end();

  LLVM_DEBUG(dbgs() << "********* " << getPassName() << " *********\n");

  for (; I != E; ++I)
    Modified |= VisitMBB(*I);
  return Modified;
}

FunctionPass *llvm::createXtensaBRegFixupPass() {
  return new XtensaBRegFixup();
}

bool XtensaBRegFixup::VisitMBB(MachineBasicBlock &MBB) {
  bool Modified = false;
  MachineBasicBlock::instr_iterator MII = MBB.instr_begin(),
                                    E = MBB.instr_end();
  MachineBasicBlock::instr_iterator NextMII;

  // Iterate through the instructions in the basic block
  for (; MII != E; MII = NextMII) {
    NextMII = std::next(MII);
    MachineInstr *MI = &*MII;

    // Don't reduce bundled instructions or pseudo operations
    if (MI->isBundle() || MI->isTransient())
      continue;

    Modified |= VisitInstruction(MII);
  }

  return Modified;
}

bool XtensaBRegFixup::VisitInstruction(
    const MachineBasicBlock::instr_iterator &MII) {
  MachineInstr *MI = &*MII;
  MachineBasicBlock &MBB = *MI->getParent();
  MachineFunction *MF = MBB.getParent();
  const XtensaInstrInfo &TII =
      *static_cast<const XtensaInstrInfo *>(MF->getSubtarget().getInstrInfo());
  unsigned Opcode = MI->getOpcode();
  unsigned RegBase = Xtensa::B0;

  switch (Opcode) {
  case Xtensa::MOVBA_P2: {
    /*
      MOVBA_P2 Breg, Dst1, Dst2, Src
       |
       V
      RSR       Dst1, BREG
      LoadImmediate Dst2, BregMask
      AND       Dst2, Dst2, Dst1
      SLLI      Dst1, Src, BregShift
      OR        Dst2, Dst2, Dst1
      WSR       BREG, Dst2
      */
    MachineOperand Breg = MI->getOperand(0);
    MachineOperand Dst1 = MI->getOperand(1);
    MachineOperand Dst2 = MI->getOperand(2);
    MachineOperand Src = MI->getOperand(3);
    DebugLoc DL = MI->getDebugLoc();
    unsigned RegNo = Breg.getReg().id() - RegBase;
    int64_t Mask = 0xffff & (~(1 << RegNo));

    MachineInstrBuilder MIB = BuildMI(MBB, MI, DL, TII.get(Xtensa::RSR)).add(Dst1).addReg(Xtensa::BREG);

    TII.buildLoadImmediate(MBB, MI, Dst2.getReg(), Mask);

    BuildMI(MBB, MI, DL, TII.get(Xtensa::AND)).add(Dst2).add(Dst2).add(Dst1);

    if (RegNo > 0) {
      BuildMI(MBB, MI, DL, TII.get(Xtensa::SLLI))
          .add(Dst1)
          .add(Src)
          .addImm(RegNo);
    } else {
      Dst1.setReg(Src.getReg());
    }

    BuildMI(MBB, MI, DL, TII.get(Xtensa::OR)).add(Dst2).add(Dst2).add(Dst1);

    BuildMI(MBB, MI, DL, TII.get(Xtensa::WSR)).addReg(Xtensa::BREG).add(Dst2);

    LLVM_DEBUG(dbgs() << " Fixed MOVBA_P2: " << *MIB);
    MBB.erase_instr(MI);
    return true;
  } break;
  case Xtensa::EXTUI_BR_P: {

    MachineOperand Breg = MI->getOperand(2);
    DebugLoc dl = MI->getDebugLoc();
    const MCInstrDesc &NewMCID = XtensaII->get(Xtensa::EXTUI);
    MachineInstrBuilder MIB = BuildMI(MBB, MI, dl, NewMCID);
    // Transfer MI flags.
    MIB.setMIFlags(MI->getFlags());
    MIB.add(MI->getOperand(0));
    MIB.add(MI->getOperand(1));
    unsigned RegNo = Breg.getReg().id() - RegBase;
    MIB.addImm(RegNo);
    MIB.addImm(1);

    LLVM_DEBUG(dbgs() << " Fixed EXTUI: " << *MIB);
    MBB.erase_instr(MI);
    return true;

  } break;

  case Xtensa::SLLI_BR_P: {

    MachineOperand Breg = MI->getOperand(2);
    unsigned RegNo = Breg.getReg().id() - RegBase;
    if (RegNo != 0) {
      DebugLoc dl = MI->getDebugLoc();
      const MCInstrDesc &NewMCID = XtensaII->get(Xtensa::SLLI);
      MachineInstrBuilder MIB = BuildMI(MBB, MI, dl, NewMCID);
      // Transfer MI flags.
      MIB.setMIFlags(MI->getFlags());
      MIB.add(MI->getOperand(0));
      MIB.add(MI->getOperand(1));
      MIB.addImm(RegNo);

      LLVM_DEBUG(dbgs() << " Fixed SLLI: " << *MIB);
    } else {
      LLVM_DEBUG(dbgs() << " Fixed SLLI: SLLI 0 => NOP");
    }
    MBB.erase_instr(MI);
    return true;

  } break;
  default:
    break;
  }

  return false;
}