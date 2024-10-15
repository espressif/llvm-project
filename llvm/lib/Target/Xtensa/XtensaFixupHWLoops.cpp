//===---- XtensaFixupHWLoops.cpp - Fixup HW loops -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include "Xtensa.h"
#include "XtensaTargetMachine.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/Pass.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;

namespace llvm {
FunctionPass *createXtensaFixupHwLoops();
void initializeXtensaFixupHwLoopsPass(PassRegistry &);
} // namespace llvm

namespace {
class XtensaFixupHwLoops : public MachineFunctionPass {
  // BasicBlockInfo - Information about the offset and size of a single
  // basic block.
  struct BasicBlockInfo {
    // Offset - Distance from the beginning of the function to the beginning
    // of this basic block.
    //
    // The offset is always aligned as required by the basic block.
    unsigned Offset = 0;

    // Size - Size of the basic block in bytes.  If the block contains
    // inline assembly, this is a worst case estimate.
    //
    // The size does not include any alignment padding whether from the
    // beginning of the block, or from an aligned jump table at the end.
    unsigned Size = 0;

    BasicBlockInfo() = default;

    // Compute the offset immediately following this block. \p MBB is the next
    // block.
    unsigned postOffset(const MachineBasicBlock &MBB) const {
      const unsigned PO = Offset + Size;
      const Align Alignment = MBB.getAlignment();
      if (Alignment == 1)
        return PO;

      const Align ParentAlign = MBB.getParent()->getAlignment();
      if (Alignment <= ParentAlign)
        return PO + offsetToAlignment(PO, Alignment);

      // The alignment of this MBB is larger than the function's alignment, so
      // we can't tell whether or not it will insert nops. Assume that it will.
      return PO + Alignment.value() + offsetToAlignment(PO, Alignment);
    }
  };

  SmallVector<BasicBlockInfo, 16> BlockInfo;
  SmallPtrSet<MachineBasicBlock *, 1> AnalyzedMBBs;

  MachineFunction *MF;
  MachineLoopInfo *MLI;
  const TargetRegisterInfo *TRI;
  const TargetInstrInfo *TII;

  bool processLoop(MachineLoop *L);

  bool fixupLoopInstrs(MachineLoop *L);

  void scanFunction();

  uint64_t computeBlockSize(const MachineBasicBlock &MBB) const;

  void adjustBlockOffsets(MachineBasicBlock &Start);

public:
  static char ID;

  XtensaFixupHwLoops() : MachineFunctionPass(ID) {
    initializeXtensaFixupHwLoopsPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::NoVRegs);
  }

  StringRef getPassName() const override {
    return "Xtensa Hardware Loop Fixup";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addRequired<MachineLoopInfo>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
};

char XtensaFixupHwLoops::ID = 0;
} // namespace

INITIALIZE_PASS(XtensaFixupHwLoops, "hwloopsfixup",
                "Xtensa Hardware Loops Fixup", false, false)

FunctionPass *llvm::createXtensaFixupHwLoops() {
  return new XtensaFixupHwLoops();
}

// Returns true if the instruction is a hardware loop instruction.
static bool isHardwareLoop(const MachineInstr &MI) {
  return (MI.getOpcode() == Xtensa::LOOPSTART);
}

bool XtensaFixupHwLoops::runOnMachineFunction(MachineFunction &mf) {
  if (skipFunction(mf.getFunction()))
    return false;

  MF = &mf;
  MLI = &getAnalysis<MachineLoopInfo>();
  const TargetSubtargetInfo &ST = mf.getSubtarget();
  TII = ST.getInstrInfo();
  TRI = ST.getRegisterInfo();

  // Renumber all of the machine basic blocks in the function, guaranteeing that
  // the numbers agree with the position of the block in the function.
  mf.RenumberBlocks();

  // Do the initial scan of the function, building up information about the
  // sizes of each block.
  scanFunction();

  AnalyzedMBBs.clear();

  bool Changed = false;

  for (auto &L : *MLI)
    if (!L->getParentLoop()) {
      Changed |= processLoop(L);
    }

  return Changed;
}

// Scan loop and find hardware loop pseudo instructions LOOPSTART and LOOPEND.
// Transform LOOPSTART to Xtensa instructions and remove LOOPEND.
bool XtensaFixupHwLoops::fixupLoopInstrs(MachineLoop *L) {
  MachineBasicBlock &MBB = *(L->getHeader());
  const TargetInstrInfo *TII = MBB.getParent()->getSubtarget().getInstrInfo();
  bool Changed = false;
  unsigned Num = MBB.getNumber();
  unsigned Offset = BlockInfo[Num].Offset;
  MachineBasicBlock *LastBlock = nullptr;
  unsigned LHOffset = Offset;
  unsigned LastBlockOffset = 0;

  // Loop over all the instructions.
  MachineBasicBlock::iterator MII = MBB.begin();
  MachineBasicBlock::iterator MIE = MBB.end();
  MachineInstr *PredI1 = nullptr;
  MachineInstr *FirstMI = nullptr;

  // Find appropriate place for the LOOPEND block
  for (auto MBI = L->block_begin(), MBIE = L->block_end(); MBI != MBIE; ++MBI) {
    MachineBasicBlock *TBB = nullptr, *FBB = nullptr;
    SmallVector<MachineOperand, 4> Cond;
    if (!TII->analyzeBranch(*(*MBI), TBB, FBB, Cond)) {
      if (FBB && TBB) {
        if (LastBlockOffset < BlockInfo[(*MBI)->getNumber()].Offset) {
          LastBlockOffset = BlockInfo[(*MBI)->getNumber()].Offset;
          LastBlock = (*MBI);
        }
      }
    }
  }

  while (MII != MIE) {
    if (MII->isMetaInstruction()) {
      ++MII;
      continue;
    }

    MachineInstr &MI = *MII;

    if (FirstMI == nullptr)
      FirstMI = &MI;

    if (isHardwareLoop(*MII)) {
      MachineBasicBlock *LoopEnd = nullptr;

      MII->getNextNode();

      MachineBasicBlock::iterator NextMII = std::next(MII);

      // Check whether loop is empty and remove if true
      if (NextMII != MIE) {
        if ((*NextMII).getOpcode() == Xtensa::LOOPEND) {
          MBB.erase(*NextMII);
          MBB.erase(*MII);
          MBB.removeSuccessor(&MBB, true);
          return true;
        }
      }

      for (MachineBasicBlock::pred_iterator PI = MBB.pred_begin(),
                                            PE = MBB.pred_end();
           PI != PE; ++PI) {
        MachineBasicBlock *PMBB = *PI;
        MachineBasicBlock::iterator PIB = PMBB->begin();
        MachineBasicBlock::iterator PII = PMBB->end();

        do {
          --PII;
          if (PII->isMetaInstruction()) {
            continue;
          }

          if ((*PII).getOpcode() == Xtensa::LOOPEND) {
            DebugLoc DL = PII->getDebugLoc();
            unsigned OffsetLE = BlockInfo[PMBB->getNumber()].Offset;

            // In most cases we expect that blocks in loop are ordered by such manner that block
            // with LOOPSTART instruction preceeds block with LOOPEND instruction.
            // But in some cases after transformations loop block which contains LOOPEND instruction
            // maybe placed before LOOPSTART block during code generaion. We must handle such situation
            // because "loop" instruction placed instead of LOOPSTART must have positive offset in the target
            // field to the LOOPEND block.
            // So, in such situation we add new LOOPEND block after the LOOPSTART block and create jump from old
            // LOOPEND block to the new LOOPEND block adn set new LOOPEND block then as target for "loop" instruction
            if (OffsetLE < LHOffset) {
              LoopEnd = MF->CreateMachineBasicBlock();

              // If last block in the loop is whithin 256 byte offset from loop instruction
              // then just place LOOPEND block after the last block.
              if ((LastBlockOffset - LHOffset) < 256) {
                //Insert after appropriate block
                MF->insert(++LastBlock->getIterator(), LoopEnd);
              } else {
                // If loop is to large for hardware loop instructuin offset then
                // place LoopEnd block just after loop header
                MF->insert(++MBB.getIterator(), LoopEnd);
                MachineBasicBlock *TBB = nullptr, *FBB = nullptr;
                SmallVector<MachineOperand, 4> Cond;
                if (!TII->analyzeBranch(MBB, TBB, FBB, Cond)) {
                  if (!FBB) {
                    // LH block just falls through to its succ
                    for (auto I = MBB.succ_begin(), E = MBB.succ_end(); I != E;
                         ++I) {
                      MachineBasicBlock *Succ = *I;
                      if (Succ != TBB) {
                        BuildMI(MBB, MBB.end(), DL, TII->get(Xtensa::J))
                            .addMBB(Succ);
                      }
                    }
                  }
                }
              }

              LoopEnd->transferSuccessors(PMBB);
              LoopEnd->splice(LoopEnd->end(), PMBB, PII, PMBB->end());

              MachineBasicBlock::iterator LEI = LoopEnd->end();
              --LEI;

              // Expect jump instruction
              assert((LEI->getOpcode() == Xtensa::J) && "Broken hardware loop");

              // Create block and insert it before loop end address as
              // target for jump instruction to avoid premature exit from loop
              MachineBasicBlock *BlockForJump = MF->CreateMachineBasicBlock();
              MF->insert(LoopEnd->getIterator(), BlockForJump);
              BlockForJump->addSuccessor(LoopEnd);
              BuildMI(*BlockForJump, BlockForJump->end(), DL,
                      TII->get(Xtensa::NOP));
              BuildMI(*PMBB, PMBB->end(), DL, TII->get(Xtensa::J))
                  .addMBB(BlockForJump);
              PMBB->addSuccessor(BlockForJump);

              BuildMI(*LoopEnd, LoopEnd->begin(), DL, TII->get(Xtensa::LOOPEND))
                  .addMBB(LoopEnd);
              LoopEnd->addSuccessor(LoopEnd);
              Changed = true;
              break;
            }

            if (PII != PIB) {
              LoopEnd = MF->CreateMachineBasicBlock();
              MF->insert(++(PMBB->getIterator()), LoopEnd);
              LoopEnd->transferSuccessors(PMBB);
              LoopEnd->splice(LoopEnd->end(), PMBB, PII, PMBB->end());
              PMBB->addSuccessor(LoopEnd);

              BuildMI(*LoopEnd, LoopEnd->begin(), DL, TII->get(Xtensa::LOOPEND))
                  .addMBB(LoopEnd);
              LoopEnd->addSuccessor(LoopEnd);
            } else {
              bool NeedBlockForJump = false;
              // Check for branches to the loop end basic block from
              // predecessors
              for (auto I = PMBB->pred_begin(), E = PMBB->pred_end(); I != E;
                   ++I) {
                MachineBasicBlock *PLEMBB = *I;
                MachineBasicBlock *TBB = nullptr, *FBB = nullptr;
                SmallVector<MachineOperand, 4> Cond;
                if (!TII->analyzeBranch(*PLEMBB, TBB, FBB, Cond)) {
                  if (TBB == PMBB) {
                    NeedBlockForJump = true;
                    break;
                  }
                } else {
                  NeedBlockForJump = true;
                  break;
                }
              }
              // Create block and insert it before loop end address as
              // target for jump/branch instruction to avoid premature exit from
              // loop
              if (NeedBlockForJump) {
                LoopEnd = MF->CreateMachineBasicBlock();
                MF->insert(++(PMBB->getIterator()), LoopEnd);
                LoopEnd->transferSuccessors(PMBB);
                LoopEnd->splice(LoopEnd->end(), PMBB, PII, PMBB->end());
                PMBB->addSuccessor(LoopEnd);
                BuildMI(*PMBB, PMBB->end(), DL, TII->get(Xtensa::NOP));

                BuildMI(*LoopEnd, LoopEnd->begin(), DL,
                        TII->get(Xtensa::LOOPEND))
                    .addMBB(LoopEnd);
                LoopEnd->addSuccessor(LoopEnd);
              } else {
                BuildMI(*PMBB, PII, DL, TII->get(Xtensa::LOOPEND)).addMBB(PMBB);
                PMBB->addSuccessor(PMBB);
                BuildMI(*PMBB, PII, DL, TII->get(Xtensa::NOP));
                LoopEnd = PMBB;
              }
            }

            Changed = true;
            break;
          }
        } while (PII != PIB);
        if (Changed)
          break;
      }

      assert((Changed) && "Broken hardware loop");

      if (MII != FirstMI) {
        MBB.splice(FirstMI->getIterator(), &MBB, MII);
        Offset = BlockInfo[Num].Offset;
        switch (PredI1->getOpcode()) {
        case Xtensa::L32I_N:
          if (PredI1->getOperand(0).getReg() == MII->getOperand(0).getReg()) {
            MBB.splice(MII, &MBB, PredI1);
            Offset += 2;
          }
          break;
        case Xtensa::L32I:
          if (PredI1->getOperand(0).getReg() == MII->getOperand(0).getReg()) {
            MBB.splice(MII, &MBB, PredI1);
            Offset += 3;
          }
          break;
        }
      }

      DebugLoc DL = MII->getDebugLoc();

      // Fixup Loop alignment
      switch (Offset & 0x3) {
      case 0x0:
        BuildMI(MBB, MII, DL, TII->get(Xtensa::NOP));
        BuildMI(MBB, MII, DL, TII->get(Xtensa::NOP));
        break;
      case 0x3:
        BuildMI(MBB, MII, DL, TII->get(Xtensa::NOP));
        break;
      }

      BuildMI(MBB, MII, DL, TII->get(Xtensa::LOOP))
          .addReg(MII->getOperand(0).getReg())
          .addMBB(LoopEnd);
      MBB.erase(MII);

      MF->RenumberBlocks();
      scanFunction();
      AnalyzedMBBs.insert(&MBB);
      return true;
    } else {
      Offset += TII->getInstSizeInBytes(MI);
      PredI1 = &MI;
      ++MII;
    }
  }

  return Changed;
}

bool XtensaFixupHwLoops::processLoop(MachineLoop *L) {
  bool Changed = false;

  // Process nested loops first.
  for (MachineLoop::iterator I = L->begin(), E = L->end(); I != E; ++I) {
    Changed |= processLoop(*I);
  }

  if (Changed)
    return true;

  return fixupLoopInstrs(L);
}

// scanFunction - Do the initial scan of the function, building up
// information about each block.
void XtensaFixupHwLoops::scanFunction() {
  BlockInfo.clear();
  BlockInfo.resize(MF->getNumBlockIDs());

  // First thing, compute the size of all basic blocks, and see if the function
  // has any inline assembly in it. If so, we have to be conservative about
  // alignment assumptions, as we don't know for sure the size of any
  // instructions in the inline assembly.
  for (MachineBasicBlock &MBB : *MF)
    BlockInfo[MBB.getNumber()].Size = computeBlockSize(MBB);

  // Compute block offsets and known bits.
  adjustBlockOffsets(*MF->begin());
}

// computeBlockSize - Compute the size for MBB.
uint64_t
XtensaFixupHwLoops::computeBlockSize(const MachineBasicBlock &MBB) const {
  uint64_t Size = 0;
  for (const MachineInstr &MI : MBB)
    if (MI.getOpcode() != Xtensa::LOOPEND)
      Size += TII->getInstSizeInBytes(MI);
  return Size;
}

void XtensaFixupHwLoops::adjustBlockOffsets(MachineBasicBlock &Start) {
  unsigned PrevNum = Start.getNumber();
  for (auto &MBB : make_range(MachineFunction::iterator(Start), MF->end())) {
    unsigned Num = MBB.getNumber();
    if (!Num) // block zero is never changed from offset zero.
      continue;
    // Get the offset and known bits at the end of the layout predecessor.
    // Include the alignment of the current block.
    BlockInfo[Num].Offset = BlockInfo[PrevNum].postOffset(MBB);

    PrevNum = Num;
  }
}

