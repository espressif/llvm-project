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

  bool checkLoopEndDisplacement(MachineFunction &MF, MachineBasicBlock *LH, MachineBasicBlock* LE);
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

  if (!STI->hasLoop())
    return false;

  VisitedMBBs.clear();

  for (auto &L : *MLI)
    if (!L->getParentLoop()) {
      Changed |= processLoop(L);
    }

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
  MachineInstr *LII = nullptr; // LOOPINIT instruction
  MachineInstr *LEI = nullptr; // LOOPEND instruction
  MachineBasicBlock *LEMBB = nullptr;
  MachineBasicBlock *LH = L->getHeader();
  MachineBasicBlock *LastMBB = L->getLoopLatch();
  std::vector<MachineInstr *> LoopInitInsts;
  std::map<MachineBasicBlock *, MachineInstr *> LoopInitMap;

  // Try to find LOOPEND instruction in the loop latch
  for (auto MBI = L->block_begin(), MBIE = L->block_end(); MBI != MBIE; ++MBI) {
    if (VisitedMBBs.count(*MBI))
      continue;
    for (auto MII = (*MBI)->begin(), MIE = (*MBI)->end(); MII != MIE; ++MII) {
      MachineInstr *LMI = &*MII;
      if (LMI->getOpcode() == Xtensa::LOOPEND) {
        LEI = LMI;
        LEMBB = *MBI;
      }
      // Collect LOOPINIT instructions inside the loop
      if (LMI->getOpcode() == Xtensa::LOOPINIT) {
        LoopInitInsts.push_back(LMI);
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

  if (LEI != nullptr) {
    MachineBasicBlock::iterator LHI = LH->getFirstNonPHI();
    MachineBasicBlock *LIMBB = nullptr;

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
        (LEMBB != LastMBB) ||
        (!checkLoopEndDisplacement(*LH->getParent(), LH, LEMBB))) {
      const MCInstrDesc &PD = TII->get(TargetOpcode::PHI);
      MachineInstr *NewPN = LH->getParent()->CreateMachineInstr(PD, DL);
      LH->insert(LH->begin(), NewPN);
      Register PR = MRI->createVirtualRegister(&Xtensa::ARRegClass);
      NewPN->addOperand(MachineOperand::CreateReg(PR, true));

      Register IndR = MRI->createVirtualRegister(&Xtensa::ARRegClass);

      for (auto PB : LH->predecessors()) {

        if (LoopInitMap.find(PB) != LoopInitMap.end()) {
          MachineOperand MO = MachineOperand::CreateReg(
              LoopInitMap[PB]->getOperand(0).getReg(), false);
          NewPN->addOperand(MO);
          NewPN->addOperand(MachineOperand::CreateMBB(PB));
          LoopInitMap[PB]->getParent()->erase(LoopInitMap[PB]);
        } else {
          MachineOperand MO = MachineOperand::CreateReg(IndR, false);
          NewPN->addOperand(MO);
          NewPN->addOperand(MachineOperand::CreateMBB(PB));
        }
      }

      MachineInstrBuilder MIB =
          BuildMI(*LEMBB, LEI, LEI->getDebugLoc(), TII->get(Xtensa::ADDI), IndR)
              .addReg(PR)
              .addImm(-1);

      MIB = BuildMI(*LEMBB, LEI, LEI->getDebugLoc(), TII->get(Xtensa::BNEZ))
                .addReg(IndR)
                .addMBB(LEI->getOperand(0).getMBB());
      LEMBB->erase(LEI);
      return false;
    }

    // If several LOOPINIT instructions are dicovered then create PHI
    // function
    if (LoopInitMap.size() > 1) {
      const MCInstrDesc &PD = TII->get(TargetOpcode::PHI);
      MachineInstr *NewPN = LH->getParent()->CreateMachineInstr(PD, DL);
      LH->insert(LH->begin(), NewPN);
      Register PR = MRI->createVirtualRegister(&Xtensa::ARRegClass);
      NewPN->addOperand(MachineOperand::CreateReg(PR, true));

      for (auto PB : LH->predecessors()) {

        if (LoopInitMap.find(PB) != LoopInitMap.end()) {
          MachineOperand MO = MachineOperand::CreateReg(
              LoopInitMap[PB]->getOperand(0).getReg(), false);
          NewPN->addOperand(MO);
          NewPN->addOperand(MachineOperand::CreateMBB(PB));
          LoopInitMap[PB]->getParent()->erase(LoopInitMap[PB]);
        } else {
          MachineOperand MO = MachineOperand::CreateReg(PR, false);
          NewPN->addOperand(MO);
          NewPN->addOperand(MachineOperand::CreateMBB(PB));
        }
      }
      LII = NewPN;
    }

    BuildMI(*LH, LHI, DL, TII->get(Xtensa::LOOPSTART))
        .addReg(LII->getOperand(0).getReg())
        .addMBB(LEMBB);

    if (LII->getOpcode() == Xtensa::LOOPINIT)
      LII->getParent()->erase(LII);

    return true;
  }

  return false;
}

bool XtensaHardwareLoops::checkLoopSize(MachineLoop *L) {
  uint64_t LoopSize = 0;

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

