//===- XtensaPSRAMFIx.cpp - Fixup PSRAM Cache issues   --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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

#define DEBUG_TYPE "xtensa-fix-esp32-psram-cache-pass"

enum PSRAMFixChoice {
  ESP32_PSRAM_FIX_MEMW,
  ESP32_PSRAM_FIX_NOPS
};

static cl::opt<bool> AlwaysMembarrier("malways-memw", cl::init(false),
                                      cl::Hidden);

static cl::opt<bool> FixESP32PSRAMCacheIssue("mfix-esp32-psram-cache-issue",
                                             cl::init(false), cl::Hidden);

static cl::opt<PSRAMFixChoice> ESP32PSRAMFixStrat(
    "mfix-esp32-psram-cache-strategy", cl::init(ESP32_PSRAM_FIX_MEMW),
    cl::desc(""),
    cl::values(clEnumValN(ESP32_PSRAM_FIX_MEMW, "memw", ""),
               clEnumValN(ESP32_PSRAM_FIX_NOPS, "nops", "")));

STATISTIC(NumAdded, "Number of instructions added");

class createXtensaPSRAMCacheFix : public MachineFunctionPass {
public:
  static char ID;
  createXtensaPSRAMCacheFix() : MachineFunctionPass(ID) {}

  const XtensaSubtarget *Subtarget;
  static const XtensaInstrInfo *XtensaII;

  bool runOnMachineFunction(MachineFunction &MF) override;

  llvm::StringRef getPassName() const override {
    return "Xtensa fix PSRAM cache issue in the ESP32 chips";
  }

private:
  bool xtensaPSRAMCacheFixNopReorg(MachineFunction &MF);
  /*
  Alternative fix to xtensaPSRAMCacheFixNopReorg. Tries to solve the 32-bit
  load/store inversion by explicitly inserting a memory barrier instead of nops.
  Slower than nops, but faster than just adding memws everywhere.
  */
  bool xtensaPSRAMCacheFixMemwReorg(MachineFunction &MF);
  // Emits a memw before every load/store instruction. Hard-handed approach to
  // get rid of any pipeline/memory issues...
  bool xtensaInsertMemwReorg(MachineFunction &MF);
};

char createXtensaPSRAMCacheFix::ID = 0;
const XtensaInstrInfo *createXtensaPSRAMCacheFix::XtensaII;

// Affected piece of pipeline is 5 entries long; the load/store itself fills
// one.
#define LOAD_STORE_OFF 4

bool createXtensaPSRAMCacheFix::xtensaPSRAMCacheFixNopReorg(
    MachineFunction &MF) {
  MachineFunction::iterator I = MF.begin(), E = MF.end();
  MachineInstr *LastHIQIStore = nullptr;
  MachineInstr *StoreInsn = nullptr;
  int InsnsSinceStore = 0;
  bool Modified = false;

  for (; I != E; ++I) {
    MachineBasicBlock &MBB = *I;
    MachineBasicBlock::instr_iterator MII = MBB.instr_begin(),
                                      MIE = MBB.instr_end();
    MachineBasicBlock::instr_iterator NextMII;

    // Iterate through the instructions in the basic block
    for (; MII != MIE; MII = NextMII) {
      MachineInstr *MI = &*MII;
      unsigned Opcode = MI->getOpcode();
      NextMII = std::next(MII);

      if (MI->isCall() || MI->isBranch() || MI->isReturn()) {
        if (LastHIQIStore) {
          DebugLoc dl = LastHIQIStore->getDebugLoc();
          const MCInstrDesc &NewMCID = XtensaII->get(Xtensa::MEMW);
          BuildMI(*LastHIQIStore->getParent(), LastHIQIStore, dl, NewMCID);
          LastHIQIStore = nullptr;
          Modified = true;
          NumAdded++;
        }
        if (!(MI->isBranch() && (MI->getOpcode() != Xtensa::J) &&
              (MI->getOpcode() != Xtensa::JX))) {
          StoreInsn = nullptr;
        }
        continue;
      }

      switch (Opcode) {
      case Xtensa::LSI:
      case Xtensa::L32I_N:
      case Xtensa::L32I:
      case Xtensa::L16SI:
      case Xtensa::L16UI:
      case Xtensa::L8UI:
        if (StoreInsn) {
          while (InsnsSinceStore++ < LOAD_STORE_OFF) {
            DebugLoc dl = MII->getDebugLoc();
            const MCInstrDesc &NewMCID = XtensaII->get(Xtensa::NOP);
            BuildMI(MBB, MII, dl, NewMCID);
            Modified = true;
            NumAdded++;
          }
        }
        if (LastHIQIStore) {
          DebugLoc dl = LastHIQIStore->getDebugLoc();
          const MCInstrDesc &NewMCID = XtensaII->get(Xtensa::MEMW);
          BuildMI(*LastHIQIStore->getParent(),
                  std::next(LastHIQIStore->getIterator()), dl, NewMCID);
          LastHIQIStore = nullptr;
          Modified = true;
          NumAdded++;
        }
        break;
      case Xtensa::SSI:
      case Xtensa::S32I_N:
      case Xtensa::S32I: {
        LastHIQIStore = nullptr;
        InsnsSinceStore = 0;
        StoreInsn = MI;
      } break;
      case Xtensa::S16I:
      case Xtensa::S8I: {
        LastHIQIStore = MI;
        InsnsSinceStore = 0;
        StoreInsn = MI;
      } break;
      default:
        InsnsSinceStore++;
        break;
      }
    }
  }
  return Modified;
}

bool createXtensaPSRAMCacheFix::xtensaPSRAMCacheFixMemwReorg(
    MachineFunction &MF) {
  MachineFunction::iterator I = MF.begin(), E = MF.end();
  MachineInstr *LastHIQIStore = nullptr;
  MachineInstr *StoreInsn = nullptr;
  bool Modified = false;

  for (; I != E; ++I) {
    MachineBasicBlock &MBB = *I;

    MachineBasicBlock::instr_iterator MII = MBB.instr_begin(),
                                      MIE = MBB.instr_end();
    MachineBasicBlock::instr_iterator NextMII;

    // Iterate through the instructions in the basic block
    for (; MII != MIE; MII = NextMII) {
      NextMII = std::next(MII);
      MachineInstr *MI = &*MII;
      unsigned Opcode = MI->getOpcode();

      // Don't process bundled instructions or pseudo operations
      if (MI->isBundle() || MI->isTransient())
        continue;

      if (MI->isCall() || MI->isBranch() || MI->isReturn()) {
        if (StoreInsn) {
          if (!(MI->isBranch() && (MI->getOpcode() != Xtensa::J) &&
                (MI->getOpcode() != Xtensa::JX))) {
            DebugLoc dl = MI->getDebugLoc();
            const MCInstrDesc &NewMCID = XtensaII->get(Xtensa::MEMW);
            MachineBasicBlock::instr_iterator BranchI = MI->getIterator();
            while (((*BranchI).isBranch() || (*BranchI).isCall() ||
                    (*BranchI).isReturn()) &&
                   (BranchI != MBB.instr_begin()))
              BranchI = std::prev(BranchI);

            if (BranchI != MBB.instr_begin())
              BranchI = std::next(BranchI);

            BuildMI(MBB, BranchI, dl, NewMCID);
            Modified = true;
            StoreInsn = nullptr;
            NumAdded++;
          }
        }
        if (LastHIQIStore) {
          DebugLoc dl = LastHIQIStore->getDebugLoc();
          const MCInstrDesc &NewMCID = XtensaII->get(Xtensa::MEMW);
          BuildMI(*LastHIQIStore->getParent(),
                  std::next(LastHIQIStore->getIterator()), dl, NewMCID);
          LastHIQIStore = nullptr;
          Modified = true;
          NumAdded++;
        }
        continue;
      }

      switch (Opcode) {
      case Xtensa::LSI:
      case Xtensa::L32I_N:
      case Xtensa::L32I:
      case Xtensa::L16SI:
      case Xtensa::L16UI:
      case Xtensa::L8UI:
        if (StoreInsn) {
          if (!MII->memoperands_empty()) {
            MachineMemOperand *MMO = *MII->memoperands_begin();
            if (!MMO->isVolatile()) {
              DebugLoc dl = MII->getDebugLoc();
              const MCInstrDesc &NewMCID = XtensaII->get(Xtensa::MEMW);
              BuildMI(MBB, MII, dl, NewMCID);
              Modified = true;
              StoreInsn = nullptr;
              NumAdded++;
            }
          }
        }
        if (LastHIQIStore) {
          DebugLoc dl = LastHIQIStore->getDebugLoc();
          const MCInstrDesc &NewMCID = XtensaII->get(Xtensa::MEMW);
          BuildMI(*LastHIQIStore->getParent(),
                  std::next(LastHIQIStore->getIterator()), dl, NewMCID);
          LastHIQIStore = nullptr;
          Modified = true;
          NumAdded++;
        }
        break;
      case Xtensa::SSI:
      case Xtensa::S32I_N:
      case Xtensa::S32I: {
        LastHIQIStore = nullptr;
        StoreInsn = MI;
      } break;
      case Xtensa::S16I:
      case Xtensa::S8I: {
        if (!MII->memoperands_empty()) {
          MachineMemOperand *MMO = *MII->memoperands_begin();
          if (!MMO->isVolatile()) {
            LastHIQIStore = MI;
          }
        }
        StoreInsn = MI;
      } break;
      }
    }
  }
  return Modified;
}

bool createXtensaPSRAMCacheFix::xtensaInsertMemwReorg(MachineFunction &MF) {
  MachineFunction::iterator I = MF.begin(), E = MF.end();
  bool Modified = false;
  bool HadMemw = false;

  for (; I != E; ++I) {
    MachineBasicBlock &MBB = *I;

    MachineBasicBlock::instr_iterator MII = MBB.instr_begin(),
                                      MIE = MBB.instr_end();
    MachineBasicBlock::instr_iterator NextMII;

    // Iterate through the instructions in the basic block
    for (; MII != MIE; MII = NextMII) {
      NextMII = std::next(MII);
      MachineInstr *MI = &*MII;
      unsigned Opcode = MI->getOpcode();

      // Don't process bundled instructions or pseudo operations
      if (MI->isBundle() || MI->isTransient())
        continue;

      switch (Opcode) {
      case Xtensa::LSI:
      case Xtensa::L32I_N:
      case Xtensa::L32I:
      case Xtensa::L16SI:
      case Xtensa::L16UI:
      case Xtensa::L8UI: {
        if (!MII->memoperands_empty()) {
          MachineMemOperand *MMO = *MII->memoperands_begin();
          if (!MMO->isVolatile() && (!HadMemw)) {
            DebugLoc dl = MII->getDebugLoc();
            const MCInstrDesc &NewMCID = XtensaII->get(Xtensa::MEMW);
            BuildMI(MBB, MII, dl, NewMCID);
            Modified = true;
            NumAdded++;
          }
        }
        HadMemw = false;
      } break;
      case Xtensa::SSI:
      case Xtensa::S32I_N:
      case Xtensa::S32I:
      case Xtensa::S16I:
      case Xtensa::S8I: {
        if (!MII->memoperands_empty()) {
          MachineMemOperand *MMO = *MII->memoperands_begin();
          if (!MMO->isVolatile()) {
            DebugLoc dl = MII->getDebugLoc();
            const MCInstrDesc &NewMCID = XtensaII->get(Xtensa::MEMW);
            BuildMI(MBB, NextMII, dl, NewMCID);
            Modified = true;
            NumAdded++;
          }
        }
        HadMemw = true;
      } break;
      default:
        HadMemw = false;
        break;
      }
    }
  }
  return Modified;
}

bool createXtensaPSRAMCacheFix::runOnMachineFunction(MachineFunction &MF) {

  Subtarget = &static_cast<const XtensaSubtarget &>(MF.getSubtarget());
  XtensaII = static_cast<const XtensaInstrInfo *>(Subtarget->getInstrInfo());
  bool Modified = false;

  if (AlwaysMembarrier)
    return xtensaInsertMemwReorg(MF);

  if (!FixESP32PSRAMCacheIssue)
    return false;

  if (ESP32PSRAMFixStrat == ESP32_PSRAM_FIX_MEMW) {
    Modified = xtensaPSRAMCacheFixMemwReorg(MF);
  } else if (ESP32PSRAMFixStrat == ESP32_PSRAM_FIX_NOPS) {
    Modified = xtensaPSRAMCacheFixNopReorg(MF);
  }

  return Modified;
}

FunctionPass *llvm::createXtensaPSRAMCacheFixPass() {
  return new createXtensaPSRAMCacheFix();
}

