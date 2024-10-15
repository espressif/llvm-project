//===- XtensaHardwareLoops.cpp - Idenify and generate hardware Loops ------===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains implementation of the pass which optimizes loops .
//
//===----------------------------------------------------------------------===//

#include "XtensaInstrInfo.h"
#include "XtensaSubtarget.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "xtensa-hwloops"
#define MAX_LOOP_SIZE 256

namespace llvm {

FunctionPass *createXtensaHardwareLoops();
void initializeXtensaHardwareLoopsPass(PassRegistry &);

} // end namespace llvm

namespace {

struct XtensaHardwareLoops : public MachineFunctionPass {
  MachineLoopInfo *MLI;
  MachineRegisterInfo *MRI;
  MachineDominatorTree *MDT;
  const XtensaInstrInfo *TII;
  const TargetRegisterInfo *TRI;
  const XtensaSubtarget *STI;
  SmallPtrSet<MachineBasicBlock *, 1> VisitedMBBs;

public:
  static char ID;

  XtensaHardwareLoops() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return "Xtensa Hardware Loops"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineLoopInfo>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
  // Return true if the instruction is not valid within a hardware
  // loop.
  bool isInvalidLoopOperation(const MachineInstr *MI) const;

  // Return true if the loop contains an instruction that inhibits
  // using the hardware loop.
  bool containsInvalidInstruction(MachineLoop *L) const;

  // Given a loop, check if we can convert it to a hardware loop.
  // If so, then perform the conversion and return true.
  bool processLoop(MachineLoop *L);

  bool checkLoopSize(MachineLoop *L);

  bool checkLoopEndDisplacement(MachineFunction &MF, MachineBasicBlock *LH,
                                MachineBasicBlock *LE);

  void revertNonLoops(MachineFunction &M);
};

char XtensaHardwareLoops::ID = 0;

} // end anonymous namespace

INITIALIZE_PASS(XtensaHardwareLoops, "hwloops", "Xtensa Hardware Loops", false,
                false)

FunctionPass *llvm::createXtensaHardwareLoops() {
  return new XtensaHardwareLoops();
}

bool XtensaHardwareLoops::runOnMachineFunction(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "********* Xtensa Hardware Loops *********\n");
  if (skipFunction(MF.getFunction()))
    return false;

  bool Changed = false;
  MLI = &getAnalysis<MachineLoopInfo>();
  MRI = &MF.getRegInfo();
  STI = &MF.getSubtarget<XtensaSubtarget>();
  TII = STI->getInstrInfo();
  TRI = STI->getRegisterInfo();

  if (!STI->hasLoop())
    return false;

  VisitedMBBs.clear();

  for (auto &L : *MLI)
    if (!L->getParentLoop()) {
      Changed |= processLoop(L);
    }

  revertNonLoops(MF);

  return Changed;
}

// Return true if the operation is invalid within hardware loop.
bool XtensaHardwareLoops::isInvalidLoopOperation(const MachineInstr *MI) const {

  // Call is not allowed because the callee may use a hardware loop
  if (MI->getDesc().isCall())
    return true;

  if ((MI->getOpcode() == Xtensa::LOOP) ||
      (MI->getOpcode() == Xtensa::LOOPGTZ) ||
      (MI->getOpcode() == Xtensa::LOOPNEZ))
    return true;

  if (MI->isInlineAsm())
    return true;

  return false;
}

// Return true if the loop contains an instruction that inhibits
// the use of the hardware loop instruction.
bool XtensaHardwareLoops::containsInvalidInstruction(MachineLoop *L) const {
  LLVM_DEBUG(dbgs() << "\nhw_loop head, "
                    << printMBBReference(**L->block_begin()));
  for (MachineBasicBlock *MBB : L->getBlocks()) {
    for (MachineBasicBlock::iterator MII = MBB->begin(), E = MBB->end();
         MII != E; ++MII) {
      const MachineInstr *MI = &*MII;
      if (isInvalidLoopOperation(MI)) {
        LLVM_DEBUG(dbgs() << "\nCannot convert to hw_loop due to:";
                   MI->dump(););
        return true;
      }
    }
  }
  return false;
}

// Check if this loop is suitable for converting to a hardware loop
bool XtensaHardwareLoops::processLoop(MachineLoop *L) {
  // This is just for sanity.
  assert(L->getHeader() && "Loop without a header?");

  bool Changed = false;

  // Process nested loops first.
  for (MachineLoop::iterator I = L->begin(), E = L->end(); I != E; ++I) {
    Changed |= processLoop(*I);
  }

  if (Changed)
    return true;

  using instr_iterator = MachineBasicBlock::instr_iterator;
  MachineInstr *LII = nullptr;   // LOOPINIT instruction
  MachineInstr *LDECI = nullptr; // LOOPDEC instruction
  MachineInstr *LBRI = nullptr;  // LOOPBR instruction
  MachineBasicBlock *LDECMBB = nullptr;
  MachineBasicBlock *LBRMBB = nullptr;
  MachineBasicBlock *LH = L->getHeader();
  MachineBasicBlock *LastMBB = L->getLoopLatch();
  std::map<MachineBasicBlock *, MachineInstr *> LoopInitMap;

  // Try to find LOOPENDDEC instruction in the loop latch
  for (auto MBI = L->block_begin(), MBIE = L->block_end(); MBI != MBIE; ++MBI) {
    if (VisitedMBBs.count(*MBI))
      continue;
    for (auto MII = (*MBI)->begin(), MIE = (*MBI)->end(); MII != MIE; ++MII) {
      MachineInstr *LMI = &*MII;
      if (LMI->getOpcode() == Xtensa::LOOPDEC) {
        LDECI = LMI;
        LDECMBB = *MBI;
      }

      if (LMI->getOpcode() == Xtensa::LOOPBR) {
        LBRI = LMI;
        LBRMBB = *MBI;
      }

      // Collect LOOPINIT instructions inside the loop
      if (LMI->getOpcode() == Xtensa::LOOPINIT) {
        MachineBasicBlock *SB = LMI->getParent();
        while (!SB->isSuccessor(LH)) {
          for (auto SBI : SB->successors()) {
            if (!L->contains(SBI))
              continue;
            SB = SBI;
            break;
          }
          if (!L->contains(SB))
            llvm_unreachable("Wrong hardware loop");
        }
        LoopInitMap[SB] = LMI;
      }
    }
    VisitedMBBs.insert(*MBI);
  }

  if ((LBRI != nullptr) && (LDECI != nullptr)) {
    MachineBasicBlock *LIMBB = nullptr;

    for (const auto &Use : MRI->use_operands(LDECI->getOperand(0).getReg())) {
      const MachineInstr *UseMI = Use.getParent();
      if ((UseMI != LBRI) && (UseMI->getOpcode() != TargetOpcode::PHI)) {
        LLVM_DEBUG(dbgs() << "Xtensa Loops: Unable to remove LoopDec.\n");
        return false;
      }
    }

    // Collect LOOPINIT instructions in predecessors from outter loop
    for (auto PBI : LH->predecessors()) {
      if (L->contains(PBI))
        continue;
      LIMBB = PBI;
      LII = nullptr;
      // Try to find LOOPINIT instructions in predecessor
      while ((LII == nullptr) && (LIMBB != nullptr) &&
             ((L->getParentLoop() == nullptr) ||
              (L->getParentLoop()->contains(LIMBB)))) {
        for (instr_iterator I = LIMBB->instr_begin(), E = LIMBB->instr_end();
             I != E; ++I) {
          MachineInstr *MI = &*I;
          if (MI->getOpcode() == Xtensa::LOOPINIT) {
            LII = MI;
            break;
          }
        }
        if (LII == nullptr)
          LIMBB = *LIMBB->pred_begin();
      }
      if (LII == nullptr) {
        llvm_unreachable("Hardware loop init instruction not found");
        return false;
      }
      LoopInitMap[PBI] = LII;
    }

    DebugLoc DL = LII->getDebugLoc();

    // If loop is too large or have wrong configuration
    // then restore branch instruction
    // sub a, a, 1
    // bnez a, LH
    if (!checkLoopSize(L) || containsInvalidInstruction(L) ||
        (LBRMBB != LastMBB) ||
        (!checkLoopEndDisplacement(*LH->getParent(), LH, LBRMBB))) {

      for (auto PB : LH->predecessors()) {
        if (LoopInitMap.find(PB) != LoopInitMap.end()) {
          Register Elts = LoopInitMap[PB]->getOperand(1).getReg();
          Register Def = LoopInitMap[PB]->getOperand(0).getReg();

          for (auto &Use : make_early_inc_range(MRI->use_operands(Def))) {
            Use.setReg(Elts);
          }
          LoopInitMap[PB]->getParent()->erase(LoopInitMap[PB]);
        }
      }

      Register IndR = LDECI->getOperand(0).getReg();
      Register PR = LDECI->getOperand(1).getReg();

      BuildMI(*LDECMBB, LDECI, LDECI->getDebugLoc(), TII->get(Xtensa::ADDI),
              IndR)
          .addReg(PR)
          .addImm(-1);
      BuildMI(*LBRMBB, LBRI, LBRI->getDebugLoc(), TII->get(Xtensa::BNEZ))
          .addReg(IndR)
          .addMBB(LBRI->getOperand(1).getMBB());
      LDECMBB->erase(LDECI);
      LBRMBB->erase(LBRI);
      return false;
    }

    MachineInstr *PN = nullptr;

    for (auto &Use : MRI->use_operands(LDECI->getOperand(0).getReg())) {
      MachineInstr *UseMI = Use.getParent();
      if (UseMI->getOpcode() == TargetOpcode::PHI) {
        PN = UseMI;
      }
    }

    assert(((PN != nullptr) && (PN->getParent() == LH)) &&
           "Expected PHI node successor of the LOOPEND instruction in loop "
           "header");
    LII = PN;

    Register EltsDec = LDECI->getOperand(0).getReg();
    Register Elts = LDECI->getOperand(1).getReg();

    for (MachineOperand &MO : PN->operands()) {
      if (!MO.isReg() || MO.getReg() != EltsDec)
        continue;
      MO.substVirtReg(Elts, 0, *TRI);
    }
    LDECMBB->erase(LDECI);

    MachineBasicBlock::iterator LHI = LH->getFirstNonPHI();

    BuildMI(*LH, LHI, DL, TII->get(Xtensa::LOOPSTART))
        .addReg(LII->getOperand(0).getReg())
        .addMBB(LBRMBB);

    if (LII->getOpcode() == Xtensa::LOOPINIT)
      LII->getParent()->erase(LII);

    BuildMI(*LBRMBB, LBRI, DL, TII->get(Xtensa::LOOPEND)).addMBB(LH);
    LBRMBB->erase(LBRI);

    return true;
  }

  return false;
}

bool XtensaHardwareLoops::checkLoopSize(MachineLoop *L) {
  uint64_t LoopSize = 3; //Reserve space for possible NOP

  for (auto *MBB : L->getBlocks()) {
    uint64_t BlockSize = 0;
    for (const MachineInstr &MI : *MBB) {
      uint64_t InstSize = TII->getInstSizeInBytes(MI);
      if (MI.isPHI())
        InstSize = 3;
      BlockSize += InstSize;
    }
    LoopSize += BlockSize;
  }

  if (LoopSize > MAX_LOOP_SIZE)
    return false;

  return true;
}

bool XtensaHardwareLoops::checkLoopEndDisplacement(MachineFunction &MF,
                                                   MachineBasicBlock *LH,
                                                   MachineBasicBlock *LE) {
  bool isLHVisited = false;

  if (LH == LE)
    return true;

  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I) {
    MachineBasicBlock *MBB = &*I;
    if (MBB == LH)
      isLHVisited = true;
    else if (MBB == LE) {
      if (isLHVisited)
        return true;
      else
        return false;
    }
  }
  llvm_unreachable("Wrong hardware loop");
}

void XtensaHardwareLoops::revertNonLoops(MachineFunction &MF) {
  for (MachineFunction::iterator I = MF.begin(); I != MF.end(); ++I) {
    MachineBasicBlock &MBB = *I;

    for (MachineBasicBlock::iterator MII = MBB.begin(), E = MBB.end(); MII != E;
         ++MII) {
      MachineInstr *MI = &*MII;
      if (MI->getOpcode() == Xtensa::LOOPINIT) {
        MachineInstr *LI = MI;
        MachineBasicBlock *LIMBB = LI->getParent();
        Register Elts = LI->getOperand(1).getReg();
        Register Def = LI->getOperand(0).getReg();
        for (auto &Use : make_early_inc_range(MRI->use_operands(Def))) {
          Use.setReg(Elts);
        }
        --MII;
        LIMBB->erase(LI);
      } else if (MI->getOpcode() == Xtensa::LOOPDEC) {
        MachineInstr *LEI = MI;
        MachineBasicBlock *LEMBB = LEI->getParent();
        Register IndR = LEI->getOperand(0).getReg();
        Register PR = LEI->getOperand(1).getReg();

        BuildMI(*LEMBB, LEI, LEI->getDebugLoc(), TII->get(Xtensa::ADDI), IndR)
            .addReg(PR)
            .addImm(-1);
        --MII;
        LEMBB->erase(LEI);
      } else if (MI->getOpcode() == Xtensa::LOOPBR) {
        MachineInstr *LBRI = MI;
        MachineBasicBlock *LBRMBB = LBRI->getParent();

        BuildMI(*LBRMBB, LBRI, LBRI->getDebugLoc(), TII->get(Xtensa::BNEZ))
            .addReg(LBRI->getOperand(0).getReg())
            .addMBB(LBRI->getOperand(1).getMBB());
        --MII;
        LBRMBB->erase(LBRI);
      }
    }
  }
}
