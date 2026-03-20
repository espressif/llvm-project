//===-- RISCVExpandPseudoInsts.cpp - Expand pseudo instructions -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass that expands pseudo instructions into target
// instructions. This pass should be run after register allocation but before
// the post-regalloc scheduling pass.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVInstrInfo.h"
#include "RISCVTargetMachine.h"

#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/MC/MCContext.h"

using namespace llvm;

#define RISCV_EXPAND_PSEUDO_NAME "RISC-V pseudo instruction expansion pass"
#define RISCV_PRERA_EXPAND_PSEUDO_NAME "RISC-V Pre-RA pseudo instruction expansion pass"

namespace {

class RISCVExpandPseudo : public MachineFunctionPass {
public:
  const RISCVSubtarget *STI;
  const RISCVInstrInfo *TII;
  static char ID;

  RISCVExpandPseudo() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return RISCV_EXPAND_PSEUDO_NAME; }

private:
  bool expandMBB(MachineBasicBlock &MBB);
  bool expandMI(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                MachineBasicBlock::iterator &NextMBBI);
  bool expandCCOp(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                  MachineBasicBlock::iterator &NextMBBI);
  bool expandCCOpToCMov(MachineBasicBlock &MBB,
                        MachineBasicBlock::iterator MBBI);
  bool expandVMSET_VMCLR(MachineBasicBlock &MBB,
                         MachineBasicBlock::iterator MBBI, unsigned Opcode);
  bool expandMV_FPR16INX(MachineBasicBlock &MBB,
                         MachineBasicBlock::iterator MBBI);
  bool expandMV_FPR32INX(MachineBasicBlock &MBB,
                         MachineBasicBlock::iterator MBBI);
  bool expandRV32ZdinxStore(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator MBBI);
  bool expandRV32ZdinxLoad(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MBBI);
  bool expandPseudoReadVLENBViaVSETVLIX0(MachineBasicBlock &MBB,
                                         MachineBasicBlock::iterator MBBI);
  bool expandTHMatrixPseudo(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator MBBI);
#ifndef NDEBUG
  unsigned getInstSizeInBytes(const MachineFunction &MF) const {
    unsigned Size = 0;
    for (auto &MBB : MF)
      for (auto &MI : MBB)
        Size += TII->getInstSizeInBytes(MI);
    return Size;
  }
#endif
};

char RISCVExpandPseudo::ID = 0;

bool RISCVExpandPseudo::runOnMachineFunction(MachineFunction &MF) {
  STI = &MF.getSubtarget<RISCVSubtarget>();
  TII = STI->getInstrInfo();

#ifndef NDEBUG
  const unsigned OldSize = getInstSizeInBytes(MF);
#endif

  bool Modified = false;
  for (auto &MBB : MF)
    Modified |= expandMBB(MBB);

#ifndef NDEBUG
  const unsigned NewSize = getInstSizeInBytes(MF);
  assert(OldSize >= NewSize);
#endif
  return Modified;
}

bool RISCVExpandPseudo::expandMBB(MachineBasicBlock &MBB) {
  bool Modified = false;

  MachineBasicBlock::iterator MBBI = MBB.begin(), E = MBB.end();
  while (MBBI != E) {
    MachineBasicBlock::iterator NMBBI = std::next(MBBI);
    Modified |= expandMI(MBB, MBBI, NMBBI);
    MBBI = NMBBI;
  }

  return Modified;
}

// Forward declaration for XTHeadMatrix pseudo lookup (defined below).
namespace {
struct THMatrixPseudoEntry;
} // namespace
static const THMatrixPseudoEntry *lookupTHMatrixPseudo(unsigned Opc);

bool RISCVExpandPseudo::expandMI(MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator MBBI,
                                 MachineBasicBlock::iterator &NextMBBI) {
  // RISCVInstrInfo::getInstSizeInBytes expects that the total size of the
  // expanded instructions for each pseudo is correct in the Size field of the
  // tablegen definition for the pseudo.
  switch (MBBI->getOpcode()) {
  case RISCV::PseudoMV_FPR16INX:
    return expandMV_FPR16INX(MBB, MBBI);
  case RISCV::PseudoMV_FPR32INX:
    return expandMV_FPR32INX(MBB, MBBI);
  case RISCV::PseudoRV32ZdinxSD:
    return expandRV32ZdinxStore(MBB, MBBI);
  case RISCV::PseudoRV32ZdinxLD:
    return expandRV32ZdinxLoad(MBB, MBBI);
  case RISCV::PseudoCCMOVGPRNoX0:
  case RISCV::PseudoCCMOVGPR:
  case RISCV::PseudoCCADD:
  case RISCV::PseudoCCSUB:
  case RISCV::PseudoCCAND:
  case RISCV::PseudoCCOR:
  case RISCV::PseudoCCXOR:
  case RISCV::PseudoCCMAX:
  case RISCV::PseudoCCMAXU:
  case RISCV::PseudoCCMIN:
  case RISCV::PseudoCCMINU:
  case RISCV::PseudoCCMUL:
  case RISCV::PseudoCCLUI:
  case RISCV::PseudoCCQC_E_LB:
  case RISCV::PseudoCCQC_E_LH:
  case RISCV::PseudoCCQC_E_LW:
  case RISCV::PseudoCCQC_E_LHU:
  case RISCV::PseudoCCQC_E_LBU:
  case RISCV::PseudoCCLB:
  case RISCV::PseudoCCLH:
  case RISCV::PseudoCCLW:
  case RISCV::PseudoCCLHU:
  case RISCV::PseudoCCLBU:
  case RISCV::PseudoCCLWU:
  case RISCV::PseudoCCLD:
  case RISCV::PseudoCCQC_LI:
  case RISCV::PseudoCCQC_E_LI:
  case RISCV::PseudoCCADDW:
  case RISCV::PseudoCCSUBW:
  case RISCV::PseudoCCSLL:
  case RISCV::PseudoCCSRL:
  case RISCV::PseudoCCSRA:
  case RISCV::PseudoCCADDI:
  case RISCV::PseudoCCSLLI:
  case RISCV::PseudoCCSRLI:
  case RISCV::PseudoCCSRAI:
  case RISCV::PseudoCCANDI:
  case RISCV::PseudoCCORI:
  case RISCV::PseudoCCXORI:
  case RISCV::PseudoCCSLLW:
  case RISCV::PseudoCCSRLW:
  case RISCV::PseudoCCSRAW:
  case RISCV::PseudoCCADDIW:
  case RISCV::PseudoCCSLLIW:
  case RISCV::PseudoCCSRLIW:
  case RISCV::PseudoCCSRAIW:
  case RISCV::PseudoCCANDN:
  case RISCV::PseudoCCORN:
  case RISCV::PseudoCCXNOR:
  case RISCV::PseudoCCNDS_BFOS:
  case RISCV::PseudoCCNDS_BFOZ:
    return expandCCOp(MBB, MBBI, NextMBBI);
  case RISCV::PseudoVMCLR_M_B1:
  case RISCV::PseudoVMCLR_M_B2:
  case RISCV::PseudoVMCLR_M_B4:
  case RISCV::PseudoVMCLR_M_B8:
  case RISCV::PseudoVMCLR_M_B16:
  case RISCV::PseudoVMCLR_M_B32:
  case RISCV::PseudoVMCLR_M_B64:
    // vmclr.m vd => vmxor.mm vd, vd, vd
    return expandVMSET_VMCLR(MBB, MBBI, RISCV::VMXOR_MM);
  case RISCV::PseudoVMSET_M_B1:
  case RISCV::PseudoVMSET_M_B2:
  case RISCV::PseudoVMSET_M_B4:
  case RISCV::PseudoVMSET_M_B8:
  case RISCV::PseudoVMSET_M_B16:
  case RISCV::PseudoVMSET_M_B32:
  case RISCV::PseudoVMSET_M_B64:
    // vmset.m vd => vmxnor.mm vd, vd, vd
    return expandVMSET_VMCLR(MBB, MBBI, RISCV::VMXNOR_MM);
  case RISCV::PseudoReadVLENBViaVSETVLIX0:
    return expandPseudoReadVLENBViaVSETVLIX0(MBB, MBBI);
  case RISCV::PTH_MATRIX_SPILL: {
    // Expand spill: PTH_MATRIX_SPILL $src, $base
    // → TH_MSME_E8 $src, $base, x0  (stride=0)
    // eliminateFrameIndex has already materialized the full address into $base.
    MachineInstr &MI = *MBBI;
    DebugLoc DL = MI.getDebugLoc();
    Register SrcReg = MI.getOperand(0).getReg();
    Register BaseReg = MI.getOperand(1).getReg();
    BuildMI(MBB, MBBI, DL, TII->get(RISCV::TH_MSME_E8))
        .addReg(SrcReg, getKillRegState(MI.getOperand(0).isKill()))
        .addReg(BaseReg)
        .addReg(RISCV::X0);
    MI.eraseFromParent();
    return true;
  }
  case RISCV::PTH_MATRIX_RELOAD: {
    // Expand reload: PTH_MATRIX_RELOAD $dst, $base
    // → TH_MLME_E8 $dst, $base, x0  (stride=0)
    MachineInstr &MI = *MBBI;
    DebugLoc DL = MI.getDebugLoc();
    Register DstReg = MI.getOperand(0).getReg();
    Register BaseReg = MI.getOperand(1).getReg();
    auto NewMI = BuildMI(MBB, MBBI, DL, TII->get(RISCV::TH_MLME_E8))
        .addReg(DstReg)
        .addReg(BaseReg)
        .addReg(RISCV::X0);
    // TH_MLME_E8 has $md in (ins) not (outs), but it semantically defines
    // the register. Add an implicit def for correct liveness tracking.
    NewMI.addReg(DstReg, RegState::Implicit | RegState::Define);
    MI.eraseFromParent();
    return true;
  }
  default:
    // Try expanding XTHeadMatrix pseudos (PTH_* → TH_*).
    if (lookupTHMatrixPseudo(MBBI->getOpcode()))
      return expandTHMatrixPseudo(MBB, MBBI);
    break;
  }

  return false;
}

bool RISCVExpandPseudo::expandCCOp(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator MBBI,
                                   MachineBasicBlock::iterator &NextMBBI) {
  // First try expanding to a Conditional Move rather than a branch+mv
  if (expandCCOpToCMov(MBB, MBBI))
    return true;

  MachineFunction *MF = MBB.getParent();
  MachineInstr &MI = *MBBI;
  DebugLoc DL = MI.getDebugLoc();

  MachineBasicBlock *TrueBB = MF->CreateMachineBasicBlock(MBB.getBasicBlock());
  MachineBasicBlock *MergeBB = MF->CreateMachineBasicBlock(MBB.getBasicBlock());

  MF->insert(++MBB.getIterator(), TrueBB);
  MF->insert(++TrueBB->getIterator(), MergeBB);

  // We want to copy the "true" value when the condition is true which means
  // we need to invert the branch condition to jump over TrueBB when the
  // condition is false.
  auto CC = static_cast<RISCVCC::CondCode>(MI.getOperand(3).getImm());
  CC = RISCVCC::getInverseBranchCondition(CC);

  // Insert branch instruction.
  BuildMI(MBB, MBBI, DL, TII->get(RISCVCC::getBrCond(CC)))
      .addReg(MI.getOperand(1).getReg())
      .addReg(MI.getOperand(2).getReg())
      .addMBB(MergeBB);

  Register DestReg = MI.getOperand(0).getReg();
  assert(MI.getOperand(4).getReg() == DestReg);

  if (MI.getOpcode() == RISCV::PseudoCCMOVGPR ||
      MI.getOpcode() == RISCV::PseudoCCMOVGPRNoX0) {
    // Add MV.
    BuildMI(TrueBB, DL, TII->get(RISCV::ADDI), DestReg)
        .add(MI.getOperand(5))
        .addImm(0);
  } else {
    unsigned NewOpc;
    // clang-format off
    switch (MI.getOpcode()) {
    default:
      llvm_unreachable("Unexpected opcode!");
    case RISCV::PseudoCCADD:   NewOpc = RISCV::ADD;   break;
    case RISCV::PseudoCCSUB:   NewOpc = RISCV::SUB;   break;
    case RISCV::PseudoCCSLL:   NewOpc = RISCV::SLL;   break;
    case RISCV::PseudoCCSRL:   NewOpc = RISCV::SRL;   break;
    case RISCV::PseudoCCSRA:   NewOpc = RISCV::SRA;   break;
    case RISCV::PseudoCCAND:   NewOpc = RISCV::AND;   break;
    case RISCV::PseudoCCOR:    NewOpc = RISCV::OR;    break;
    case RISCV::PseudoCCXOR:   NewOpc = RISCV::XOR;   break;
    case RISCV::PseudoCCMAX:   NewOpc = RISCV::MAX;   break;
    case RISCV::PseudoCCMIN:   NewOpc = RISCV::MIN;   break;
    case RISCV::PseudoCCMAXU:  NewOpc = RISCV::MAXU;  break;
    case RISCV::PseudoCCMINU:  NewOpc = RISCV::MINU;  break;
    case RISCV::PseudoCCMUL:   NewOpc = RISCV::MUL;   break;
    case RISCV::PseudoCCLUI:   NewOpc = RISCV::LUI;   break;
    case RISCV::PseudoCCQC_E_LB:  NewOpc = RISCV::QC_E_LB;    break;
    case RISCV::PseudoCCQC_E_LH:  NewOpc = RISCV::QC_E_LH;    break;
    case RISCV::PseudoCCQC_E_LW:  NewOpc = RISCV::QC_E_LW;    break;
    case RISCV::PseudoCCQC_E_LHU: NewOpc = RISCV::QC_E_LHU;   break;
    case RISCV::PseudoCCQC_E_LBU: NewOpc = RISCV::QC_E_LBU;   break;
    case RISCV::PseudoCCLB:    NewOpc = RISCV::LB;    break;
    case RISCV::PseudoCCLH:    NewOpc = RISCV::LH;    break;
    case RISCV::PseudoCCLW:    NewOpc = RISCV::LW;    break;
    case RISCV::PseudoCCLHU:   NewOpc = RISCV::LHU;   break;
    case RISCV::PseudoCCLBU:   NewOpc = RISCV::LBU;   break;
    case RISCV::PseudoCCLWU:   NewOpc = RISCV::LWU;   break;
    case RISCV::PseudoCCLD:    NewOpc = RISCV::LD;    break;
    case RISCV::PseudoCCQC_LI:  NewOpc = RISCV::QC_LI;   break;
    case RISCV::PseudoCCQC_E_LI: NewOpc = RISCV::QC_E_LI;   break;
    case RISCV::PseudoCCADDI:  NewOpc = RISCV::ADDI;  break;
    case RISCV::PseudoCCSLLI:  NewOpc = RISCV::SLLI;  break;
    case RISCV::PseudoCCSRLI:  NewOpc = RISCV::SRLI;  break;
    case RISCV::PseudoCCSRAI:  NewOpc = RISCV::SRAI;  break;
    case RISCV::PseudoCCANDI:  NewOpc = RISCV::ANDI;  break;
    case RISCV::PseudoCCORI:   NewOpc = RISCV::ORI;   break;
    case RISCV::PseudoCCXORI:  NewOpc = RISCV::XORI;  break;
    case RISCV::PseudoCCADDW:  NewOpc = RISCV::ADDW;  break;
    case RISCV::PseudoCCSUBW:  NewOpc = RISCV::SUBW;  break;
    case RISCV::PseudoCCSLLW:  NewOpc = RISCV::SLLW;  break;
    case RISCV::PseudoCCSRLW:  NewOpc = RISCV::SRLW;  break;
    case RISCV::PseudoCCSRAW:  NewOpc = RISCV::SRAW;  break;
    case RISCV::PseudoCCADDIW: NewOpc = RISCV::ADDIW; break;
    case RISCV::PseudoCCSLLIW: NewOpc = RISCV::SLLIW; break;
    case RISCV::PseudoCCSRLIW: NewOpc = RISCV::SRLIW; break;
    case RISCV::PseudoCCSRAIW: NewOpc = RISCV::SRAIW; break;
    case RISCV::PseudoCCANDN:  NewOpc = RISCV::ANDN;  break;
    case RISCV::PseudoCCORN:   NewOpc = RISCV::ORN;   break;
    case RISCV::PseudoCCXNOR:  NewOpc = RISCV::XNOR;  break;
    case RISCV::PseudoCCNDS_BFOS: NewOpc = RISCV::NDS_BFOS; break;
    case RISCV::PseudoCCNDS_BFOZ: NewOpc = RISCV::NDS_BFOZ; break;
    }
    // clang-format on

    if (NewOpc == RISCV::NDS_BFOZ || NewOpc == RISCV::NDS_BFOS) {
      BuildMI(TrueBB, DL, TII->get(NewOpc), DestReg)
          .add(MI.getOperand(5))
          .add(MI.getOperand(6))
          .add(MI.getOperand(7));
    } else if (NewOpc == RISCV::LUI || NewOpc == RISCV::QC_LI ||
               NewOpc == RISCV::QC_E_LI) {
      BuildMI(TrueBB, DL, TII->get(NewOpc), DestReg).add(MI.getOperand(5));
    } else {
      BuildMI(TrueBB, DL, TII->get(NewOpc), DestReg)
          .add(MI.getOperand(5))
          .add(MI.getOperand(6));
    }
  }

  TrueBB->addSuccessor(MergeBB);

  MergeBB->splice(MergeBB->end(), &MBB, MI, MBB.end());
  MergeBB->transferSuccessors(&MBB);

  MBB.addSuccessor(TrueBB);
  MBB.addSuccessor(MergeBB);

  NextMBBI = MBB.end();
  MI.eraseFromParent();

  // Make sure live-ins are correctly attached to this new basic block.
  LivePhysRegs LiveRegs;
  computeAndAddLiveIns(LiveRegs, *TrueBB);
  computeAndAddLiveIns(LiveRegs, *MergeBB);

  return true;
}

bool RISCVExpandPseudo::expandCCOpToCMov(MachineBasicBlock &MBB,
                                         MachineBasicBlock::iterator MBBI) {
  MachineInstr &MI = *MBBI;
  DebugLoc DL = MI.getDebugLoc();

  if (MI.getOpcode() != RISCV::PseudoCCMOVGPR &&
      MI.getOpcode() != RISCV::PseudoCCMOVGPRNoX0)
    return false;

  if (!STI->hasVendorXqcicm())
    return false;

  // FIXME: Would be wonderful to support LHS=X0, but not very easy.
  if (MI.getOperand(1).getReg() == RISCV::X0 ||
      MI.getOperand(4).getReg() == RISCV::X0 ||
      MI.getOperand(5).getReg() == RISCV::X0)
    return false;

  auto CC = static_cast<RISCVCC::CondCode>(MI.getOperand(3).getImm());

  unsigned CMovOpcode, CMovIOpcode;
  switch (CC) {
  default:
    llvm_unreachable("Unhandled CC");
  case RISCVCC::COND_EQ:
    CMovOpcode = RISCV::QC_MVEQ;
    CMovIOpcode = RISCV::QC_MVEQI;
    break;
  case RISCVCC::COND_NE:
    CMovOpcode = RISCV::QC_MVNE;
    CMovIOpcode = RISCV::QC_MVNEI;
    break;
  case RISCVCC::COND_LT:
    CMovOpcode = RISCV::QC_MVLT;
    CMovIOpcode = RISCV::QC_MVLTI;
    break;
  case RISCVCC::COND_GE:
    CMovOpcode = RISCV::QC_MVGE;
    CMovIOpcode = RISCV::QC_MVGEI;
    break;
  case RISCVCC::COND_LTU:
    CMovOpcode = RISCV::QC_MVLTU;
    CMovIOpcode = RISCV::QC_MVLTUI;
    break;
  case RISCVCC::COND_GEU:
    CMovOpcode = RISCV::QC_MVGEU;
    CMovIOpcode = RISCV::QC_MVGEUI;
    break;
  }

  if (MI.getOperand(2).getReg() == RISCV::X0) {
    // $dst = PseudoCCMOVGPR $lhs, X0, $cc, $falsev (=$dst), $truev
    // $dst = PseudoCCMOVGPRNoX0 $lhs, X0, $cc, $falsev (=$dst), $truev
    // =>
    // $dst = QC_MVccI $falsev (=$dst), $lhs, 0, $truev
    BuildMI(MBB, MBBI, DL, TII->get(CMovIOpcode))
        .addDef(MI.getOperand(0).getReg())
        .addReg(MI.getOperand(4).getReg())
        .addReg(MI.getOperand(1).getReg())
        .addImm(0)
        .addReg(MI.getOperand(5).getReg());

    MI.eraseFromParent();
    return true;
  }

  // $dst = PseudoCCMOVGPR $lhs, $rhs, $cc, $falsev (=$dst), $truev
  // $dst = PseudoCCMOVGPRNoX0 $lhs, $rhs, $cc, $falsev (=$dst), $truev
  // =>
  // $dst = QC_MVcc $falsev (=$dst), $lhs, $rhs, $truev
  BuildMI(MBB, MBBI, DL, TII->get(CMovOpcode))
      .addDef(MI.getOperand(0).getReg())
      .addReg(MI.getOperand(4).getReg())
      .addReg(MI.getOperand(1).getReg())
      .addReg(MI.getOperand(2).getReg())
      .addReg(MI.getOperand(5).getReg());
  MI.eraseFromParent();
  return true;
}

bool RISCVExpandPseudo::expandVMSET_VMCLR(MachineBasicBlock &MBB,
                                          MachineBasicBlock::iterator MBBI,
                                          unsigned Opcode) {
  DebugLoc DL = MBBI->getDebugLoc();
  Register DstReg = MBBI->getOperand(0).getReg();
  const MCInstrDesc &Desc = TII->get(Opcode);
  BuildMI(MBB, MBBI, DL, Desc, DstReg)
      .addReg(DstReg, RegState::Undef)
      .addReg(DstReg, RegState::Undef);
  MBBI->eraseFromParent(); // The pseudo instruction is gone now.
  return true;
}

bool RISCVExpandPseudo::expandMV_FPR16INX(MachineBasicBlock &MBB,
                                          MachineBasicBlock::iterator MBBI) {
  DebugLoc DL = MBBI->getDebugLoc();
  const TargetRegisterInfo *TRI = STI->getRegisterInfo();
  Register DstReg = TRI->getMatchingSuperReg(
      MBBI->getOperand(0).getReg(), RISCV::sub_16, &RISCV::GPRRegClass);
  Register SrcReg = TRI->getMatchingSuperReg(
      MBBI->getOperand(1).getReg(), RISCV::sub_16, &RISCV::GPRRegClass);

  BuildMI(MBB, MBBI, DL, TII->get(RISCV::ADDI), DstReg)
      .addReg(SrcReg, getKillRegState(MBBI->getOperand(1).isKill()))
      .addImm(0);

  MBBI->eraseFromParent(); // The pseudo instruction is gone now.
  return true;
}

bool RISCVExpandPseudo::expandMV_FPR32INX(MachineBasicBlock &MBB,
                                          MachineBasicBlock::iterator MBBI) {
  DebugLoc DL = MBBI->getDebugLoc();
  const TargetRegisterInfo *TRI = STI->getRegisterInfo();
  Register DstReg = TRI->getMatchingSuperReg(
      MBBI->getOperand(0).getReg(), RISCV::sub_32, &RISCV::GPRRegClass);
  Register SrcReg = TRI->getMatchingSuperReg(
      MBBI->getOperand(1).getReg(), RISCV::sub_32, &RISCV::GPRRegClass);

  BuildMI(MBB, MBBI, DL, TII->get(RISCV::ADDI), DstReg)
      .addReg(SrcReg, getKillRegState(MBBI->getOperand(1).isKill()))
      .addImm(0);

  MBBI->eraseFromParent(); // The pseudo instruction is gone now.
  return true;
}

// This function expands the PseudoRV32ZdinxSD for storing a double-precision
// floating-point value into memory by generating an equivalent instruction
// sequence for RV32.
bool RISCVExpandPseudo::expandRV32ZdinxStore(MachineBasicBlock &MBB,
                                             MachineBasicBlock::iterator MBBI) {
  DebugLoc DL = MBBI->getDebugLoc();
  const TargetRegisterInfo *TRI = STI->getRegisterInfo();
  Register Lo =
      TRI->getSubReg(MBBI->getOperand(0).getReg(), RISCV::sub_gpr_even);
  Register Hi =
      TRI->getSubReg(MBBI->getOperand(0).getReg(), RISCV::sub_gpr_odd);
  if (Hi == RISCV::DUMMY_REG_PAIR_WITH_X0)
    Hi = RISCV::X0;

  auto MIBLo = BuildMI(MBB, MBBI, DL, TII->get(RISCV::SW))
                   .addReg(Lo, getKillRegState(MBBI->getOperand(0).isKill()))
                   .addReg(MBBI->getOperand(1).getReg())
                   .add(MBBI->getOperand(2));

  MachineInstrBuilder MIBHi;
  if (MBBI->getOperand(2).isGlobal() || MBBI->getOperand(2).isCPI()) {
    assert(MBBI->getOperand(2).getOffset() % 8 == 0);
    MBBI->getOperand(2).setOffset(MBBI->getOperand(2).getOffset() + 4);
    MIBHi = BuildMI(MBB, MBBI, DL, TII->get(RISCV::SW))
                .addReg(Hi, getKillRegState(MBBI->getOperand(0).isKill()))
                .add(MBBI->getOperand(1))
                .add(MBBI->getOperand(2));
  } else {
    assert(isInt<12>(MBBI->getOperand(2).getImm() + 4));
    MIBHi = BuildMI(MBB, MBBI, DL, TII->get(RISCV::SW))
                .addReg(Hi, getKillRegState(MBBI->getOperand(0).isKill()))
                .add(MBBI->getOperand(1))
                .addImm(MBBI->getOperand(2).getImm() + 4);
  }

  MachineFunction *MF = MBB.getParent();
  SmallVector<MachineMemOperand *> NewLoMMOs;
  SmallVector<MachineMemOperand *> NewHiMMOs;
  for (const MachineMemOperand *MMO : MBBI->memoperands()) {
    NewLoMMOs.push_back(MF->getMachineMemOperand(MMO, 0, 4));
    NewHiMMOs.push_back(MF->getMachineMemOperand(MMO, 4, 4));
  }
  MIBLo.setMemRefs(NewLoMMOs);
  MIBHi.setMemRefs(NewHiMMOs);

  MBBI->eraseFromParent();
  return true;
}

// This function expands PseudoRV32ZdinxLoad for loading a double-precision
// floating-point value from memory into an equivalent instruction sequence for
// RV32.
bool RISCVExpandPseudo::expandRV32ZdinxLoad(MachineBasicBlock &MBB,
                                            MachineBasicBlock::iterator MBBI) {
  DebugLoc DL = MBBI->getDebugLoc();
  const TargetRegisterInfo *TRI = STI->getRegisterInfo();
  Register Lo =
      TRI->getSubReg(MBBI->getOperand(0).getReg(), RISCV::sub_gpr_even);
  Register Hi =
      TRI->getSubReg(MBBI->getOperand(0).getReg(), RISCV::sub_gpr_odd);
  assert(Hi != RISCV::DUMMY_REG_PAIR_WITH_X0 && "Cannot write to X0_Pair");

  MachineInstrBuilder MIBLo, MIBHi;

  // If the register of operand 1 is equal to the Lo register, then swap the
  // order of loading the Lo and Hi statements.
  bool IsOp1EqualToLo = Lo == MBBI->getOperand(1).getReg();
  // Order: Lo, Hi
  if (!IsOp1EqualToLo) {
    MIBLo = BuildMI(MBB, MBBI, DL, TII->get(RISCV::LW), Lo)
                .addReg(MBBI->getOperand(1).getReg())
                .add(MBBI->getOperand(2));
  }

  if (MBBI->getOperand(2).isGlobal() || MBBI->getOperand(2).isCPI()) {
    auto Offset = MBBI->getOperand(2).getOffset();
    assert(Offset % 8 == 0);
    MBBI->getOperand(2).setOffset(Offset + 4);
    MIBHi = BuildMI(MBB, MBBI, DL, TII->get(RISCV::LW), Hi)
                .addReg(MBBI->getOperand(1).getReg())
                .add(MBBI->getOperand(2));
    MBBI->getOperand(2).setOffset(Offset);
  } else {
    assert(isInt<12>(MBBI->getOperand(2).getImm() + 4));
    MIBHi = BuildMI(MBB, MBBI, DL, TII->get(RISCV::LW), Hi)
                .addReg(MBBI->getOperand(1).getReg())
                .addImm(MBBI->getOperand(2).getImm() + 4);
  }

  // Order: Hi, Lo
  if (IsOp1EqualToLo) {
    MIBLo = BuildMI(MBB, MBBI, DL, TII->get(RISCV::LW), Lo)
                .addReg(MBBI->getOperand(1).getReg())
                .add(MBBI->getOperand(2));
  }

  MachineFunction *MF = MBB.getParent();
  SmallVector<MachineMemOperand *> NewLoMMOs;
  SmallVector<MachineMemOperand *> NewHiMMOs;
  for (const MachineMemOperand *MMO : MBBI->memoperands()) {
    NewLoMMOs.push_back(MF->getMachineMemOperand(MMO, 0, 4));
    NewHiMMOs.push_back(MF->getMachineMemOperand(MMO, 4, 4));
  }
  MIBLo.setMemRefs(NewLoMMOs);
  MIBHi.setMemRefs(NewHiMMOs);

  MBBI->eraseFromParent();
  return true;
}

bool RISCVExpandPseudo::expandPseudoReadVLENBViaVSETVLIX0(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI) {
  DebugLoc DL = MBBI->getDebugLoc();
  Register Dst = MBBI->getOperand(0).getReg();
  unsigned Mul = MBBI->getOperand(1).getImm();
  RISCVVType::VLMUL VLMUL = RISCVVType::encodeLMUL(Mul, /*Fractional=*/false);
  unsigned VTypeImm = RISCVVType::encodeVTYPE(
      VLMUL, /*SEW=*/8, /*TailAgnostic=*/true, /*MaskAgnostic=*/true);

  BuildMI(MBB, MBBI, DL, TII->get(RISCV::PseudoVSETVLIX0))
      .addReg(Dst, RegState::Define)
      .addReg(RISCV::X0, RegState::Kill)
      .addImm(VTypeImm);

  MBBI->eraseFromParent();
  return true;
}

// XTHeadMatrix pseudo-to-real instruction mapping.
// SkipTiedInput: if true, the first (ins) operand is a tied input that must
// be dropped when expanding to the real instruction.
namespace {
struct THMatrixPseudoEntry {
  unsigned PseudoOpc;
  unsigned RealOpc;
  bool SkipTiedInput;
};
} // namespace

// clang-format off
static const THMatrixPseudoEntry THMatrixPseudoTable[] = {
    // Loads (no tied)
    {RISCV::PTH_MLAE_E8V,   RISCV::TH_MLAE_E8,   false},
    {RISCV::PTH_MLAE_E16V,  RISCV::TH_MLAE_E16,  false},
    {RISCV::PTH_MLAE_E32V,  RISCV::TH_MLAE_E32,  false},
    {RISCV::PTH_MLAE_E64V,  RISCV::TH_MLAE_E64,  false},
    {RISCV::PTH_MLATE_E8V,  RISCV::TH_MLATE_E8,  false},
    {RISCV::PTH_MLATE_E16V, RISCV::TH_MLATE_E16, false},
    {RISCV::PTH_MLATE_E32V, RISCV::TH_MLATE_E32, false},
    {RISCV::PTH_MLATE_E64V, RISCV::TH_MLATE_E64, false},
    {RISCV::PTH_MLBE_E8V,   RISCV::TH_MLBE_E8,   false},
    {RISCV::PTH_MLBE_E16V,  RISCV::TH_MLBE_E16,  false},
    {RISCV::PTH_MLBE_E32V,  RISCV::TH_MLBE_E32,  false},
    {RISCV::PTH_MLBE_E64V,  RISCV::TH_MLBE_E64,  false},
    {RISCV::PTH_MLBTE_E8V,  RISCV::TH_MLBTE_E8,  false},
    {RISCV::PTH_MLBTE_E16V, RISCV::TH_MLBTE_E16, false},
    {RISCV::PTH_MLBTE_E32V, RISCV::TH_MLBTE_E32, false},
    {RISCV::PTH_MLBTE_E64V, RISCV::TH_MLBTE_E64, false},
    {RISCV::PTH_MLCE_E8V,   RISCV::TH_MLCE_E8,   false},
    {RISCV::PTH_MLCE_E16V,  RISCV::TH_MLCE_E16,  false},
    {RISCV::PTH_MLCE_E32V,  RISCV::TH_MLCE_E32,  false},
    {RISCV::PTH_MLCE_E64V,  RISCV::TH_MLCE_E64,  false},
    {RISCV::PTH_MLCTE_E8V,  RISCV::TH_MLCTE_E8,  false},
    {RISCV::PTH_MLCTE_E16V, RISCV::TH_MLCTE_E16, false},
    {RISCV::PTH_MLCTE_E32V, RISCV::TH_MLCTE_E32, false},
    {RISCV::PTH_MLCTE_E64V, RISCV::TH_MLCTE_E64, false},
    {RISCV::PTH_MLME_E8V,   RISCV::TH_MLME_E8,   false},
    {RISCV::PTH_MLME_E16V,  RISCV::TH_MLME_E16,  false},
    {RISCV::PTH_MLME_E32V,  RISCV::TH_MLME_E32,  false},
    {RISCV::PTH_MLME_E64V,  RISCV::TH_MLME_E64,  false},
    // Stores (no tied)
    {RISCV::PTH_MSAE_E8V,   RISCV::TH_MSAE_E8,   false},
    {RISCV::PTH_MSAE_E16V,  RISCV::TH_MSAE_E16,  false},
    {RISCV::PTH_MSAE_E32V,  RISCV::TH_MSAE_E32,  false},
    {RISCV::PTH_MSAE_E64V,  RISCV::TH_MSAE_E64,  false},
    {RISCV::PTH_MSATE_E8V,  RISCV::TH_MSATE_E8,  false},
    {RISCV::PTH_MSATE_E16V, RISCV::TH_MSATE_E16, false},
    {RISCV::PTH_MSATE_E32V, RISCV::TH_MSATE_E32, false},
    {RISCV::PTH_MSATE_E64V, RISCV::TH_MSATE_E64, false},
    {RISCV::PTH_MSBE_E8V,   RISCV::TH_MSBE_E8,   false},
    {RISCV::PTH_MSBE_E16V,  RISCV::TH_MSBE_E16,  false},
    {RISCV::PTH_MSBE_E32V,  RISCV::TH_MSBE_E32,  false},
    {RISCV::PTH_MSBE_E64V,  RISCV::TH_MSBE_E64,  false},
    {RISCV::PTH_MSBTE_E8V,  RISCV::TH_MSBTE_E8,  false},
    {RISCV::PTH_MSBTE_E16V, RISCV::TH_MSBTE_E16, false},
    {RISCV::PTH_MSBTE_E32V, RISCV::TH_MSBTE_E32, false},
    {RISCV::PTH_MSBTE_E64V, RISCV::TH_MSBTE_E64, false},
    {RISCV::PTH_MSCE_E8V,   RISCV::TH_MSCE_E8,   false},
    {RISCV::PTH_MSCE_E16V,  RISCV::TH_MSCE_E16,  false},
    {RISCV::PTH_MSCE_E32V,  RISCV::TH_MSCE_E32,  false},
    {RISCV::PTH_MSCE_E64V,  RISCV::TH_MSCE_E64,  false},
    {RISCV::PTH_MSCTE_E8V,  RISCV::TH_MSCTE_E8,  false},
    {RISCV::PTH_MSCTE_E16V, RISCV::TH_MSCTE_E16, false},
    {RISCV::PTH_MSCTE_E32V, RISCV::TH_MSCTE_E32, false},
    {RISCV::PTH_MSCTE_E64V, RISCV::TH_MSCTE_E64, false},
    {RISCV::PTH_MSME_E8V,   RISCV::TH_MSME_E8,   false},
    {RISCV::PTH_MSME_E16V,  RISCV::TH_MSME_E16,  false},
    {RISCV::PTH_MSME_E32V,  RISCV::TH_MSME_E32,  false},
    {RISCV::PTH_MSME_E64V,  RISCV::TH_MSME_E64,  false},
    // Matmul (tied acc)
    {RISCV::PTH_MFMACC_H_V,       RISCV::TH_MFMACC_H,       true},
    {RISCV::PTH_MFMACC_S_V,       RISCV::TH_MFMACC_S,       true},
    {RISCV::PTH_MFMACC_D_V,       RISCV::TH_MFMACC_D,       true},
    {RISCV::PTH_MFMACC_H_E4_V,    RISCV::TH_MFMACC_H_E4,    true},
    {RISCV::PTH_MFMACC_H_E5_V,    RISCV::TH_MFMACC_H_E5,    true},
    {RISCV::PTH_MFMACC_BF16_E4_V, RISCV::TH_MFMACC_BF16_E4, true},
    {RISCV::PTH_MFMACC_BF16_E5_V, RISCV::TH_MFMACC_BF16_E5, true},
    {RISCV::PTH_MFMACC_S_H_V,     RISCV::TH_MFMACC_S_H,     true},
    {RISCV::PTH_MFMACC_S_BF16_V,  RISCV::TH_MFMACC_S_BF16,  true},
    {RISCV::PTH_MFMACC_S_E4_V,    RISCV::TH_MFMACC_S_E4,    true},
    {RISCV::PTH_MFMACC_S_E5_V,    RISCV::TH_MFMACC_S_E5,    true},
    {RISCV::PTH_MFMACC_S_TF32_V,  RISCV::TH_MFMACC_S_TF32,  true},
    {RISCV::PTH_MFMACC_D_S_V,     RISCV::TH_MFMACC_D_S,     true},
    {RISCV::PTH_MMACC_W_B_V,      RISCV::TH_MMACC_W_B,      true},
    {RISCV::PTH_MMACCU_W_B_V,     RISCV::TH_MMACCU_W_B,     true},
    {RISCV::PTH_MMACCUS_W_B_V,    RISCV::TH_MMACCUS_W_B,    true},
    {RISCV::PTH_MMACCSU_W_B_V,    RISCV::TH_MMACCSU_W_B,    true},
    {RISCV::PTH_MMACC_D_H_V,      RISCV::TH_MMACC_D_H,      true},
    {RISCV::PTH_MMACCU_D_H_V,     RISCV::TH_MMACCU_D_H,     true},
    {RISCV::PTH_MMACCUS_D_H_V,    RISCV::TH_MMACCUS_D_H,    true},
    {RISCV::PTH_MMACCSU_D_H_V,    RISCV::TH_MMACCSU_D_H,    true},
    {RISCV::PTH_PMMACC_W_B_V,     RISCV::TH_PMMACC_W_B,     true},
    {RISCV::PTH_PMMACCU_W_B_V,    RISCV::TH_PMMACCU_W_B,    true},
    {RISCV::PTH_PMMACCUS_W_B_V,   RISCV::TH_PMMACCUS_W_B,   true},
    {RISCV::PTH_PMMACCSU_W_B_V,   RISCV::TH_PMMACCSU_W_B,   true},
    {RISCV::PTH_MMACC_W_BP_V,     RISCV::TH_MMACC_W_BP,     true},
    {RISCV::PTH_MMACCU_W_BP_V,    RISCV::TH_MMACCU_W_BP,    true},
    // Zero (no tied)
    {RISCV::PTH_MZERO_V,   RISCV::TH_MZERO,   false},
    {RISCV::PTH_MZERO2R_V, RISCV::TH_MZERO2R, false},
    {RISCV::PTH_MZERO4R_V, RISCV::TH_MZERO4R, false},
    {RISCV::PTH_MZERO8R_V, RISCV::TH_MZERO8R, false},
    // Move, conversions (no tied)
    {RISCV::PTH_MMOV_MM_V, RISCV::TH_MMOV_MM, false},
    {RISCV::PTH_MFCVTL_H_E4_V, RISCV::TH_MFCVTL_H_E4, false},
    {RISCV::PTH_MFCVTH_H_E4_V, RISCV::TH_MFCVTH_H_E4, false},
    {RISCV::PTH_MFCVTL_H_E5_V, RISCV::TH_MFCVTL_H_E5, false},
    {RISCV::PTH_MFCVTH_H_E5_V, RISCV::TH_MFCVTH_H_E5, false},
    {RISCV::PTH_MFCVTL_E4_H_V, RISCV::TH_MFCVTL_E4_H, false},
    {RISCV::PTH_MFCVTH_E4_H_V, RISCV::TH_MFCVTH_E4_H, false},
    {RISCV::PTH_MFCVTL_E5_H_V, RISCV::TH_MFCVTL_E5_H, false},
    {RISCV::PTH_MFCVTH_E5_H_V, RISCV::TH_MFCVTH_E5_H, false},
    {RISCV::PTH_MFCVTL_S_H_V,    RISCV::TH_MFCVTL_S_H,    false},
    {RISCV::PTH_MFCVTH_S_H_V,    RISCV::TH_MFCVTH_S_H,    false},
    {RISCV::PTH_MFCVTL_H_S_V,    RISCV::TH_MFCVTL_H_S,    false},
    {RISCV::PTH_MFCVTH_H_S_V,    RISCV::TH_MFCVTH_H_S,    false},
    {RISCV::PTH_MFCVTL_S_BF16_V, RISCV::TH_MFCVTL_S_BF16, false},
    {RISCV::PTH_MFCVTH_S_BF16_V, RISCV::TH_MFCVTH_S_BF16, false},
    {RISCV::PTH_MFCVTL_BF16_S_V, RISCV::TH_MFCVTL_BF16_S, false},
    {RISCV::PTH_MFCVTH_BF16_S_V, RISCV::TH_MFCVTH_BF16_S, false},
    {RISCV::PTH_MFCVTL_E4_S_V, RISCV::TH_MFCVTL_E4_S, false},
    {RISCV::PTH_MFCVTH_E4_S_V, RISCV::TH_MFCVTH_E4_S, false},
    {RISCV::PTH_MFCVTL_E5_S_V, RISCV::TH_MFCVTL_E5_S, false},
    {RISCV::PTH_MFCVTH_E5_S_V, RISCV::TH_MFCVTH_E5_S, false},
    {RISCV::PTH_MFCVTL_D_S_V, RISCV::TH_MFCVTL_D_S, false},
    {RISCV::PTH_MFCVTH_D_S_V, RISCV::TH_MFCVTH_D_S, false},
    {RISCV::PTH_MFCVTL_S_D_V, RISCV::TH_MFCVTL_S_D, false},
    {RISCV::PTH_MFCVTH_S_D_V, RISCV::TH_MFCVTH_S_D, false},
    {RISCV::PTH_MFCVT_S_TF32_V, RISCV::TH_MFCVT_S_TF32, false},
    {RISCV::PTH_MFCVT_TF32_S_V, RISCV::TH_MFCVT_TF32_S, false},
    {RISCV::PTH_MUFCVTL_H_B_V, RISCV::TH_MUFCVTL_H_B, false},
    {RISCV::PTH_MUFCVTH_H_B_V, RISCV::TH_MUFCVTH_H_B, false},
    {RISCV::PTH_MSFCVTL_H_B_V, RISCV::TH_MSFCVTL_H_B, false},
    {RISCV::PTH_MSFCVTH_H_B_V, RISCV::TH_MSFCVTH_H_B, false},
    {RISCV::PTH_MFUCVTL_B_H_V, RISCV::TH_MFUCVTL_B_H, false},
    {RISCV::PTH_MFUCVTH_B_H_V, RISCV::TH_MFUCVTH_B_H, false},
    {RISCV::PTH_MFSCVTL_B_H_V, RISCV::TH_MFSCVTL_B_H, false},
    {RISCV::PTH_MFSCVTH_B_H_V, RISCV::TH_MFSCVTH_B_H, false},
    {RISCV::PTH_MSFCVT_S_W_V, RISCV::TH_MSFCVT_S_W, false},
    {RISCV::PTH_MUFCVT_S_W_V, RISCV::TH_MUFCVT_S_W, false},
    {RISCV::PTH_MFSCVT_W_S_V, RISCV::TH_MFSCVT_W_S, false},
    {RISCV::PTH_MFUCVT_W_S_V, RISCV::TH_MFUCVT_W_S, false},
    {RISCV::PTH_MUCVTL_B_P_V, RISCV::TH_MUCVTL_B_P, false},
    {RISCV::PTH_MSCVTL_B_P_V, RISCV::TH_MSCVTL_B_P, false},
    {RISCV::PTH_MUCVTH_B_P_V, RISCV::TH_MUCVTH_B_P, false},
    {RISCV::PTH_MSCVTH_B_P_V, RISCV::TH_MSCVTH_B_P, false},
    // ToGPR (no tied)
    {RISCV::PTH_MMOVB_X_M_V, RISCV::TH_MMOVB_X_M, false},
    {RISCV::PTH_MMOVH_X_M_V, RISCV::TH_MMOVH_X_M, false},
    {RISCV::PTH_MMOVW_X_M_V, RISCV::TH_MMOVW_X_M, false},
    {RISCV::PTH_MMOVD_X_M_V, RISCV::TH_MMOVD_X_M, false},
    // FromGPR2 (tied: md_in=dst, skip tied)
    {RISCV::PTH_MMOVB_M_X_V, RISCV::TH_MMOVB_M_X, true},
    {RISCV::PTH_MMOVH_M_X_V, RISCV::TH_MMOVH_M_X, true},
    {RISCV::PTH_MMOVW_M_X_V, RISCV::TH_MMOVW_M_X, true},
    {RISCV::PTH_MMOVD_M_X_V, RISCV::TH_MMOVD_M_X, true},
    // FromGPR/Dup (tied: md_in=dst, skip tied)
    {RISCV::PTH_MDUPB_M_X_V, RISCV::TH_MDUPB_M_X, true},
    {RISCV::PTH_MDUPH_M_X_V, RISCV::TH_MDUPH_M_X, true},
    {RISCV::PTH_MDUPW_M_X_V, RISCV::TH_MDUPW_M_X, true},
    {RISCV::PTH_MDUPD_M_X_V, RISCV::TH_MDUPD_M_X, true},
    // Pack (no tied)
    {RISCV::PTH_MPACK_V,   RISCV::TH_MPACK,   false},
    {RISCV::PTH_MPACKHL_V, RISCV::TH_MPACKHL, false},
    {RISCV::PTH_MPACKHH_V, RISCV::TH_MPACKHH, false},
    // Slides/broadcasts (no tied)
    {RISCV::PTH_MRSLIDEDOWN_V,  RISCV::TH_MRSLIDEDOWN,  false},
    {RISCV::PTH_MRSLIDEUP_V,    RISCV::TH_MRSLIDEUP,    false},
    {RISCV::PTH_MCSLIDEDOWN_B_V, RISCV::TH_MCSLIDEDOWN_B, false},
    {RISCV::PTH_MCSLIDEDOWN_H_V, RISCV::TH_MCSLIDEDOWN_H, false},
    {RISCV::PTH_MCSLIDEDOWN_W_V, RISCV::TH_MCSLIDEDOWN_W, false},
    {RISCV::PTH_MCSLIDEDOWN_D_V, RISCV::TH_MCSLIDEDOWN_D, false},
    {RISCV::PTH_MCSLIDEUP_B_V,  RISCV::TH_MCSLIDEUP_B,  false},
    {RISCV::PTH_MCSLIDEUP_H_V,  RISCV::TH_MCSLIDEUP_H,  false},
    {RISCV::PTH_MCSLIDEUP_W_V,  RISCV::TH_MCSLIDEUP_W,  false},
    {RISCV::PTH_MCSLIDEUP_D_V,  RISCV::TH_MCSLIDEUP_D,  false},
    {RISCV::PTH_MRBCA_MV_I_V,   RISCV::TH_MRBCA_MV_I,   false},
    {RISCV::PTH_MCBCAB_MV_I_V,  RISCV::TH_MCBCAB_MV_I,  false},
    {RISCV::PTH_MCBCAH_MV_I_V,  RISCV::TH_MCBCAH_MV_I,  false},
    {RISCV::PTH_MCBCAW_MV_I_V,  RISCV::TH_MCBCAW_MV_I,  false},
    {RISCV::PTH_MCBCAD_MV_I_V,  RISCV::TH_MCBCAD_MV_I,  false},
    // N4clip .mm (tied)
    {RISCV::PTH_MN4CLIPL_W_MM_V,  RISCV::TH_MN4CLIPL_W_MM,  true},
    {RISCV::PTH_MN4CLIPH_W_MM_V,  RISCV::TH_MN4CLIPH_W_MM,  true},
    {RISCV::PTH_MN4CLIPLU_W_MM_V, RISCV::TH_MN4CLIPLU_W_MM, true},
    {RISCV::PTH_MN4CLIPHU_W_MM_V, RISCV::TH_MN4CLIPHU_W_MM, true},
    // N4clip .mv.i (tied)
    {RISCV::PTH_MN4CLIPL_W_MV_I_V,  RISCV::TH_MN4CLIPL_W_MV_I,  true},
    {RISCV::PTH_MN4CLIPH_W_MV_I_V,  RISCV::TH_MN4CLIPH_W_MV_I,  true},
    {RISCV::PTH_MN4CLIPLU_W_MV_I_V, RISCV::TH_MN4CLIPLU_W_MV_I, true},
    {RISCV::PTH_MN4CLIPHU_W_MV_I_V, RISCV::TH_MN4CLIPHU_W_MV_I, true},
    // Int EW .mm (tied)
    {RISCV::PTH_MADD_W_MM_V,   RISCV::TH_MADD_W_MM,   true},
    {RISCV::PTH_MSUB_W_MM_V,   RISCV::TH_MSUB_W_MM,   true},
    {RISCV::PTH_MMUL_W_MM_V,   RISCV::TH_MMUL_W_MM,   true},
    {RISCV::PTH_MMULH_W_MM_V,  RISCV::TH_MMULH_W_MM,  true},
    {RISCV::PTH_MMAX_W_MM_V,   RISCV::TH_MMAX_W_MM,   true},
    {RISCV::PTH_MUMAX_W_MM_V,  RISCV::TH_MUMAX_W_MM,  true},
    {RISCV::PTH_MMIN_W_MM_V,   RISCV::TH_MMIN_W_MM,   true},
    {RISCV::PTH_MUMIN_W_MM_V,  RISCV::TH_MUMIN_W_MM,  true},
    {RISCV::PTH_MSRL_W_MM_V,   RISCV::TH_MSRL_W_MM,   true},
    {RISCV::PTH_MSLL_W_MM_V,   RISCV::TH_MSLL_W_MM,   true},
    {RISCV::PTH_MSRA_W_MM_V,   RISCV::TH_MSRA_W_MM,   true},
    // Int EW .mv.i (tied)
    {RISCV::PTH_MADD_W_MV_I_V,   RISCV::TH_MADD_W_MV_I,   true},
    {RISCV::PTH_MSUB_W_MV_I_V,   RISCV::TH_MSUB_W_MV_I,   true},
    {RISCV::PTH_MMUL_W_MV_I_V,   RISCV::TH_MMUL_W_MV_I,   true},
    {RISCV::PTH_MMULH_W_MV_I_V,  RISCV::TH_MMULH_W_MV_I,  true},
    {RISCV::PTH_MMAX_W_MV_I_V,   RISCV::TH_MMAX_W_MV_I,   true},
    {RISCV::PTH_MUMAX_W_MV_I_V,  RISCV::TH_MUMAX_W_MV_I,  true},
    {RISCV::PTH_MMIN_W_MV_I_V,   RISCV::TH_MMIN_W_MV_I,   true},
    {RISCV::PTH_MUMIN_W_MV_I_V,  RISCV::TH_MUMIN_W_MV_I,  true},
    {RISCV::PTH_MSRL_W_MV_I_V,   RISCV::TH_MSRL_W_MV_I,   true},
    {RISCV::PTH_MSLL_W_MV_I_V,   RISCV::TH_MSLL_W_MV_I,   true},
    {RISCV::PTH_MSRA_W_MV_I_V,   RISCV::TH_MSRA_W_MV_I,   true},
    // FP EW .mm (tied)
    {RISCV::PTH_MFADD_H_MM_V, RISCV::TH_MFADD_H_MM, true},
    {RISCV::PTH_MFADD_S_MM_V, RISCV::TH_MFADD_S_MM, true},
    {RISCV::PTH_MFADD_D_MM_V, RISCV::TH_MFADD_D_MM, true},
    {RISCV::PTH_MFSUB_H_MM_V, RISCV::TH_MFSUB_H_MM, true},
    {RISCV::PTH_MFSUB_S_MM_V, RISCV::TH_MFSUB_S_MM, true},
    {RISCV::PTH_MFSUB_D_MM_V, RISCV::TH_MFSUB_D_MM, true},
    {RISCV::PTH_MFMUL_H_MM_V, RISCV::TH_MFMUL_H_MM, true},
    {RISCV::PTH_MFMUL_S_MM_V, RISCV::TH_MFMUL_S_MM, true},
    {RISCV::PTH_MFMUL_D_MM_V, RISCV::TH_MFMUL_D_MM, true},
    {RISCV::PTH_MFMAX_H_MM_V, RISCV::TH_MFMAX_H_MM, true},
    {RISCV::PTH_MFMAX_S_MM_V, RISCV::TH_MFMAX_S_MM, true},
    {RISCV::PTH_MFMAX_D_MM_V, RISCV::TH_MFMAX_D_MM, true},
    {RISCV::PTH_MFMIN_H_MM_V, RISCV::TH_MFMIN_H_MM, true},
    {RISCV::PTH_MFMIN_S_MM_V, RISCV::TH_MFMIN_S_MM, true},
    {RISCV::PTH_MFMIN_D_MM_V, RISCV::TH_MFMIN_D_MM, true},
    // FP EW .mv.i (tied)
    {RISCV::PTH_MFADD_H_MV_I_V, RISCV::TH_MFADD_H_MV_I, true},
    {RISCV::PTH_MFADD_S_MV_I_V, RISCV::TH_MFADD_S_MV_I, true},
    {RISCV::PTH_MFADD_D_MV_I_V, RISCV::TH_MFADD_D_MV_I, true},
    {RISCV::PTH_MFSUB_H_MV_I_V, RISCV::TH_MFSUB_H_MV_I, true},
    {RISCV::PTH_MFSUB_S_MV_I_V, RISCV::TH_MFSUB_S_MV_I, true},
    {RISCV::PTH_MFSUB_D_MV_I_V, RISCV::TH_MFSUB_D_MV_I, true},
    {RISCV::PTH_MFMUL_H_MV_I_V, RISCV::TH_MFMUL_H_MV_I, true},
    {RISCV::PTH_MFMUL_S_MV_I_V, RISCV::TH_MFMUL_S_MV_I, true},
    {RISCV::PTH_MFMUL_D_MV_I_V, RISCV::TH_MFMUL_D_MV_I, true},
    {RISCV::PTH_MFMAX_H_MV_I_V, RISCV::TH_MFMAX_H_MV_I, true},
    {RISCV::PTH_MFMAX_S_MV_I_V, RISCV::TH_MFMAX_S_MV_I, true},
    {RISCV::PTH_MFMAX_D_MV_I_V, RISCV::TH_MFMAX_D_MV_I, true},
    {RISCV::PTH_MFMIN_H_MV_I_V, RISCV::TH_MFMIN_H_MV_I, true},
    {RISCV::PTH_MFMIN_S_MV_I_V, RISCV::TH_MFMIN_S_MV_I, true},
    {RISCV::PTH_MFMIN_D_MV_I_V, RISCV::TH_MFMIN_D_MV_I, true},
};
// clang-format on

static const THMatrixPseudoEntry *lookupTHMatrixPseudo(unsigned Opc) {
  for (const auto &E : THMatrixPseudoTable)
    if (E.PseudoOpc == Opc)
      return &E;
  return nullptr;
}

bool RISCVExpandPseudo::expandTHMatrixPseudo(MachineBasicBlock &MBB,
                                             MachineBasicBlock::iterator MBBI) {
  MachineInstr &MI = *MBBI;
  const THMatrixPseudoEntry *Entry = lookupTHMatrixPseudo(MI.getOpcode());
  if (!Entry)
    return false;

  DebugLoc DL = MI.getDebugLoc();
  MachineInstrBuilder MIB = BuildMI(MBB, MBBI, DL, TII->get(Entry->RealOpc));

  // Copy def operands (outputs).
  for (unsigned I = 0, E = MI.getNumDefs(); I < E; ++I) {
    MachineOperand &MO = MI.getOperand(I);
    MIB.add(MO);
  }

  // Copy use operands (inputs), optionally skipping the first input
  // (tied operand that duplicates the output).
  bool SkippedTied = false;
  for (unsigned I = MI.getNumDefs(), E = MI.getNumOperands(); I < E; ++I) {
    MachineOperand &MO = MI.getOperand(I);
    // Skip implicit operands.
    if (MO.isImplicit())
      continue;
    // Skip the first tied input if requested.
    if (Entry->SkipTiedInput && !SkippedTied && MO.isReg() && MO.isTied()) {
      SkippedTied = true;
      continue;
    }
    MIB.add(MO);
  }

  MI.eraseFromParent();
  return true;
}

class RISCVPreRAExpandPseudo : public MachineFunctionPass {
public:
  const RISCVSubtarget *STI;
  const RISCVInstrInfo *TII;
  static char ID;

  RISCVPreRAExpandPseudo() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }
  StringRef getPassName() const override {
    return RISCV_PRERA_EXPAND_PSEUDO_NAME;
  }

private:
  bool expandMBB(MachineBasicBlock &MBB);
  bool expandMI(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                MachineBasicBlock::iterator &NextMBBI);
  bool expandAuipcInstPair(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MBBI,
                           MachineBasicBlock::iterator &NextMBBI,
                           unsigned FlagsHi, unsigned SecondOpcode);
  bool expandLoadLocalAddress(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator MBBI,
                              MachineBasicBlock::iterator &NextMBBI);
  bool expandLoadGlobalAddress(MachineBasicBlock &MBB,
                               MachineBasicBlock::iterator MBBI,
                               MachineBasicBlock::iterator &NextMBBI);
  bool expandLoadTLSIEAddress(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator MBBI,
                              MachineBasicBlock::iterator &NextMBBI);
  bool expandLoadTLSGDAddress(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator MBBI,
                              MachineBasicBlock::iterator &NextMBBI);
  bool expandLoadTLSDescAddress(MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator MBBI,
                                MachineBasicBlock::iterator &NextMBBI);

#ifndef NDEBUG
  unsigned getInstSizeInBytes(const MachineFunction &MF) const {
    unsigned Size = 0;
    for (auto &MBB : MF)
      for (auto &MI : MBB)
        Size += TII->getInstSizeInBytes(MI);
    return Size;
  }
#endif
};

char RISCVPreRAExpandPseudo::ID = 0;

bool RISCVPreRAExpandPseudo::runOnMachineFunction(MachineFunction &MF) {
  STI = &MF.getSubtarget<RISCVSubtarget>();
  TII = STI->getInstrInfo();

#ifndef NDEBUG
  const unsigned OldSize = getInstSizeInBytes(MF);
#endif

  bool Modified = false;
  for (auto &MBB : MF)
    Modified |= expandMBB(MBB);

#ifndef NDEBUG
  const unsigned NewSize = getInstSizeInBytes(MF);
  assert(OldSize >= NewSize);
#endif
  return Modified;
}

bool RISCVPreRAExpandPseudo::expandMBB(MachineBasicBlock &MBB) {
  bool Modified = false;

  MachineBasicBlock::iterator MBBI = MBB.begin(), E = MBB.end();
  while (MBBI != E) {
    MachineBasicBlock::iterator NMBBI = std::next(MBBI);
    Modified |= expandMI(MBB, MBBI, NMBBI);
    MBBI = NMBBI;
  }

  return Modified;
}

bool RISCVPreRAExpandPseudo::expandMI(MachineBasicBlock &MBB,
                                      MachineBasicBlock::iterator MBBI,
                                      MachineBasicBlock::iterator &NextMBBI) {

  switch (MBBI->getOpcode()) {
  case RISCV::PseudoLLA:
    return expandLoadLocalAddress(MBB, MBBI, NextMBBI);
  case RISCV::PseudoLGA:
    return expandLoadGlobalAddress(MBB, MBBI, NextMBBI);
  case RISCV::PseudoLA_TLS_IE:
    return expandLoadTLSIEAddress(MBB, MBBI, NextMBBI);
  case RISCV::PseudoLA_TLS_GD:
    return expandLoadTLSGDAddress(MBB, MBBI, NextMBBI);
  case RISCV::PseudoLA_TLSDESC:
    return expandLoadTLSDescAddress(MBB, MBBI, NextMBBI);
  }
  return false;
}

bool RISCVPreRAExpandPseudo::expandAuipcInstPair(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    MachineBasicBlock::iterator &NextMBBI, unsigned FlagsHi,
    unsigned SecondOpcode) {
  MachineFunction *MF = MBB.getParent();
  MachineInstr &MI = *MBBI;
  DebugLoc DL = MI.getDebugLoc();

  Register DestReg = MI.getOperand(0).getReg();
  Register ScratchReg =
      MF->getRegInfo().createVirtualRegister(&RISCV::GPRRegClass);

  MachineOperand &Symbol = MI.getOperand(1);
  Symbol.setTargetFlags(FlagsHi);
  MCSymbol *AUIPCSymbol = MF->getContext().createNamedTempSymbol("pcrel_hi");

  MachineInstr *MIAUIPC =
      BuildMI(MBB, MBBI, DL, TII->get(RISCV::AUIPC), ScratchReg).add(Symbol);
  MIAUIPC->setPreInstrSymbol(*MF, AUIPCSymbol);

  MachineInstr *SecondMI =
      BuildMI(MBB, MBBI, DL, TII->get(SecondOpcode), DestReg)
          .addReg(ScratchReg)
          .addSym(AUIPCSymbol, RISCVII::MO_PCREL_LO);

  if (MI.hasOneMemOperand())
    SecondMI->addMemOperand(*MF, *MI.memoperands_begin());

  MI.eraseFromParent();
  return true;
}

bool RISCVPreRAExpandPseudo::expandLoadLocalAddress(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    MachineBasicBlock::iterator &NextMBBI) {
  return expandAuipcInstPair(MBB, MBBI, NextMBBI, RISCVII::MO_PCREL_HI,
                             RISCV::ADDI);
}

bool RISCVPreRAExpandPseudo::expandLoadGlobalAddress(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    MachineBasicBlock::iterator &NextMBBI) {
  unsigned SecondOpcode = STI->is64Bit() ? RISCV::LD : RISCV::LW;
  return expandAuipcInstPair(MBB, MBBI, NextMBBI, RISCVII::MO_GOT_HI,
                             SecondOpcode);
}

bool RISCVPreRAExpandPseudo::expandLoadTLSIEAddress(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    MachineBasicBlock::iterator &NextMBBI) {
  unsigned SecondOpcode = STI->is64Bit() ? RISCV::LD : RISCV::LW;
  return expandAuipcInstPair(MBB, MBBI, NextMBBI, RISCVII::MO_TLS_GOT_HI,
                             SecondOpcode);
}

bool RISCVPreRAExpandPseudo::expandLoadTLSGDAddress(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    MachineBasicBlock::iterator &NextMBBI) {
  return expandAuipcInstPair(MBB, MBBI, NextMBBI, RISCVII::MO_TLS_GD_HI,
                             RISCV::ADDI);
}

bool RISCVPreRAExpandPseudo::expandLoadTLSDescAddress(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
    MachineBasicBlock::iterator &NextMBBI) {
  MachineFunction *MF = MBB.getParent();
  MachineInstr &MI = *MBBI;
  DebugLoc DL = MI.getDebugLoc();

  const auto &STI = MF->getSubtarget<RISCVSubtarget>();
  unsigned SecondOpcode = STI.is64Bit() ? RISCV::LD : RISCV::LW;

  Register FinalReg = MI.getOperand(0).getReg();
  Register DestReg =
      MF->getRegInfo().createVirtualRegister(&RISCV::GPRRegClass);
  Register ScratchReg =
      MF->getRegInfo().createVirtualRegister(&RISCV::GPRRegClass);

  MachineOperand &Symbol = MI.getOperand(1);
  Symbol.setTargetFlags(RISCVII::MO_TLSDESC_HI);
  MCSymbol *AUIPCSymbol = MF->getContext().createNamedTempSymbol("tlsdesc_hi");

  MachineInstr *MIAUIPC =
      BuildMI(MBB, MBBI, DL, TII->get(RISCV::AUIPC), ScratchReg).add(Symbol);
  MIAUIPC->setPreInstrSymbol(*MF, AUIPCSymbol);

  BuildMI(MBB, MBBI, DL, TII->get(SecondOpcode), DestReg)
      .addReg(ScratchReg)
      .addSym(AUIPCSymbol, RISCVII::MO_TLSDESC_LOAD_LO);

  BuildMI(MBB, MBBI, DL, TII->get(RISCV::ADDI), RISCV::X10)
      .addReg(ScratchReg)
      .addSym(AUIPCSymbol, RISCVII::MO_TLSDESC_ADD_LO);

  BuildMI(MBB, MBBI, DL, TII->get(RISCV::PseudoTLSDESCCall), RISCV::X5)
      .addReg(DestReg)
      .addImm(0)
      .addSym(AUIPCSymbol, RISCVII::MO_TLSDESC_CALL);

  BuildMI(MBB, MBBI, DL, TII->get(RISCV::ADD), FinalReg)
      .addReg(RISCV::X10)
      .addReg(RISCV::X4);

  MI.eraseFromParent();
  return true;
}

} // end of anonymous namespace

INITIALIZE_PASS(RISCVExpandPseudo, "riscv-expand-pseudo",
                RISCV_EXPAND_PSEUDO_NAME, false, false)

INITIALIZE_PASS(RISCVPreRAExpandPseudo, "riscv-prera-expand-pseudo",
                RISCV_PRERA_EXPAND_PSEUDO_NAME, false, false)

namespace llvm {

FunctionPass *createRISCVExpandPseudoPass() { return new RISCVExpandPseudo(); }
FunctionPass *createRISCVPreRAExpandPseudoPass() { return new RISCVPreRAExpandPseudo(); }

} // end of namespace llvm
