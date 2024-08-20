//===- XtensaS3ISelLowering.cpp - Xtensa S3 DAG Lowering Implementation ---===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that Xtensa uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#include "XtensaISelLowering.h"
#include "XtensaSubtarget.h"

using namespace llvm;

MachineBasicBlock *XtensaTargetLowering::EmitDSPInstrWithCustomInserter(
    MachineInstr &MI, MachineBasicBlock *MBB, const TargetInstrInfo &TII,
    MachineFunction *MF, MachineRegisterInfo &MRI, DebugLoc DL) const {
  switch (MI.getOpcode()) {
  default:
    llvm_unreachable("Unexpected instr type to insert");
  case Xtensa::EE_ANDQ_P: {
    unsigned Opc = Xtensa::EE_ANDQ;
    MachineOperand &QA = MI.getOperand(0);
    unsigned QAVal = QA.getImm();
    assert(QAVal < 8 && "Unexpected value of ee_andq first argument, it must "
                        "be in range [0,7]");
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_andq first argument, it must "
                        "be in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_andq first argument, it must "
                        "be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QAVal)
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_BITREV_P: {
    unsigned Opc = Xtensa::EE_BITREV;
    MachineOperand &QA = MI.getOperand(0);
    unsigned QAVal = QA.getImm();
    assert(QAVal < 8 && "Unexpected value of ee_bitrev first argument, it must "
                        "be in range [0,7]");
    MachineOperand &AX = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QAVal)
        .addReg(R1, RegState::Undef)
        .addReg(AX.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_CMUL_S16_P: {
    unsigned Opc = Xtensa::EE_CMUL_S16;
    MachineOperand &QZ = MI.getOperand(0);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of ee_cmul_s16 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_cmul_s16 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_cmul_s16 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &SEL4 = MI.getOperand(3);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QZVal)
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal)
        .addImm(SEL4.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_CMUL_S16_LD_INCP_P: {
    unsigned Opc = Xtensa::EE_CMUL_S16_LD_INCP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_cmul_s16_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &QZ = MI.getOperand(2);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of ee_cmul_s16_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_cmul_s16_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_cmul_s16_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &SEL4 = MI.getOperand(5);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QZVal)
        .addReg(AS.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal)
        .addImm(SEL4.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_CMUL_S16_ST_INCP_P: {
    unsigned Opc = Xtensa::EE_CMUL_S16_ST_INCP;
    MachineOperand &QV = MI.getOperand(0);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of ee_cmul_s16_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &QZ = MI.getOperand(2);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of ee_cmul_s16_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_cmul_s16_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_cmul_s16_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &SEL4 = MI.getOperand(5);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QZVal)
        .addReg(Xtensa::Q0 + QVVal)
        .addReg(AS.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal)
        .addImm(SEL4.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_FFT_AMS_S16_LD_INCP_P: {
    unsigned Opc = Xtensa::EE_FFT_AMS_S16_LD_INCP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_fft_ams_s16_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &QZ = MI.getOperand(2);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of ee_fft_ams_s16_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QZ1 = MI.getOperand(3);
    unsigned QZ1Val = QZ1.getImm();
    assert(QZ1Val < 8 && "Unexpected value of ee_fft_ams_s16_ld_incp first "
                         "argument, it must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(4);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_fft_ams_s16_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(5);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_fft_ams_s16_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QM = MI.getOperand(6);
    unsigned QMVal = QM.getImm();
    assert(QMVal < 8 && "Unexpected value of ee_fft_ams_s16_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &SEL2 = MI.getOperand(7);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QZVal)
        .addReg(Xtensa::Q0 + QZ1Val)
        .addReg(AS.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal)
        .addReg(Xtensa::Q0 + QMVal)
        .addImm(SEL2.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_FFT_AMS_S16_LD_INCP_UAUP_P: {
    unsigned Opc = Xtensa::EE_FFT_AMS_S16_LD_INCP_UAUP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_fft_ams_s16_ld_incp_uaup first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &QZ = MI.getOperand(2);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of ee_fft_ams_s16_ld_incp_uaup first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QZ1 = MI.getOperand(3);
    unsigned QZ1Val = QZ1.getImm();
    assert(QZ1Val < 8 && "Unexpected value of ee_fft_ams_s16_ld_incp_uaup "
                         "first argument, it must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(4);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_fft_ams_s16_ld_incp_uaup first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(5);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_fft_ams_s16_ld_incp_uaup first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QM = MI.getOperand(6);
    unsigned QMVal = QM.getImm();
    assert(QMVal < 8 && "Unexpected value of ee_fft_ams_s16_ld_incp_uaup first "
                        "argument, it must be in range [0,7]");
    MachineOperand &SEL2 = MI.getOperand(7);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QZVal)
        .addReg(Xtensa::Q0 + QZ1Val)
        .addReg(AS.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal)
        .addReg(Xtensa::Q0 + QMVal)
        .addImm(SEL2.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_FFT_AMS_S16_LD_R32_DECP_P: {
    unsigned Opc = Xtensa::EE_FFT_AMS_S16_LD_R32_DECP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_fft_ams_s16_ld_r32_decp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &QZ = MI.getOperand(2);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of ee_fft_ams_s16_ld_r32_decp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QZ1 = MI.getOperand(3);
    unsigned QZ1Val = QZ1.getImm();
    assert(QZ1Val < 8 && "Unexpected value of ee_fft_ams_s16_ld_r32_decp first "
                         "argument, it must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(4);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_fft_ams_s16_ld_r32_decp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(5);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_fft_ams_s16_ld_r32_decp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QM = MI.getOperand(6);
    unsigned QMVal = QM.getImm();
    assert(QMVal < 8 && "Unexpected value of ee_fft_ams_s16_ld_r32_decp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &SEL2 = MI.getOperand(7);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QZVal)
        .addReg(Xtensa::Q0 + QZ1Val)
        .addReg(AS.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal)
        .addReg(Xtensa::Q0 + QMVal)
        .addImm(SEL2.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_FFT_AMS_S16_ST_INCP_P: {
    unsigned Opc = Xtensa::EE_FFT_AMS_S16_ST_INCP;
    MachineOperand &QV = MI.getOperand(0);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of ee_fft_ams_s16_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QZ1 = MI.getOperand(1);
    unsigned QZ1Val = QZ1.getImm();
    assert(QZ1Val < 8 && "Unexpected value of ee_fft_ams_s16_st_incp first "
                         "argument, it must be in range [0,7]");
    MachineOperand &AS0 = MI.getOperand(2);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &AS = MI.getOperand(3);
    unsigned R2 = MRI.createVirtualRegister(RC);
    MachineOperand &QX = MI.getOperand(4);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_fft_ams_s16_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(5);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_fft_ams_s16_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QM = MI.getOperand(6);
    unsigned QMVal = QM.getImm();
    assert(QMVal < 8 && "Unexpected value of ee_fft_ams_s16_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &SEL2 = MI.getOperand(7);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QZ1Val)
        .addReg(R1, RegState::Undef)
        .addReg(R2, RegState::Undef)
        .addReg(Xtensa::Q0 + QVVal)
        .addReg(AS0.getReg())
        .addReg(AS.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal)
        .addReg(Xtensa::Q0 + QMVal)
        .addImm(SEL2.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_FFT_CMUL_S16_LD_XP_P: {
    unsigned Opc = Xtensa::EE_FFT_CMUL_S16_LD_XP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_fft_cmul_s16_ld_xp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &AD = MI.getOperand(2);
    MachineOperand &QZ = MI.getOperand(3);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of ee_fft_cmul_s16_ld_xp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(4);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_fft_cmul_s16_ld_xp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(5);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_fft_cmul_s16_ld_xp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &SEL8 = MI.getOperand(6);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QZVal)
        .addReg(AS.getReg())
        .addReg(AD.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal)
        .addImm(SEL8.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_FFT_CMUL_S16_ST_XP_P: {
    unsigned Opc = Xtensa::EE_FFT_CMUL_S16_ST_XP;
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_fft_cmul_s16_st_xp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_fft_cmul_s16_st_xp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QV = MI.getOperand(2);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of ee_fft_cmul_s16_st_xp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(3);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &AD = MI.getOperand(4);
    MachineOperand &SEL8 = MI.getOperand(5);
    MachineOperand &UPD4 = MI.getOperand(6);
    MachineOperand &SAR4 = MI.getOperand(7);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal)
        .addReg(Xtensa::Q0 + QVVal)
        .addReg(AS.getReg())
        .addReg(AD.getReg())
        .addImm(SEL8.getImm())
        .addImm(UPD4.getImm())
        .addImm(SAR4.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_FFT_R2BF_S16_P: {
    unsigned Opc = Xtensa::EE_FFT_R2BF_S16;
    MachineOperand &QA0 = MI.getOperand(0);
    unsigned QA0Val = QA0.getImm();
    assert(QA0Val < 8 && "Unexpected value of ee_fft_r2bf_s16 first argument, "
                         "it must be in range [0,7]");
    MachineOperand &QA1 = MI.getOperand(1);
    unsigned QA1Val = QA1.getImm();
    assert(QA1Val < 8 && "Unexpected value of ee_fft_r2bf_s16 first argument, "
                         "it must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(2);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_fft_r2bf_s16 first argument, "
                        "it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(3);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_fft_r2bf_s16 first argument, "
                        "it must be in range [0,7]");
    MachineOperand &SEL2 = MI.getOperand(4);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QA0Val)
        .addReg(Xtensa::Q0 + QA1Val)
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal)
        .addImm(SEL2.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_FFT_R2BF_S16_ST_INCP_P: {
    unsigned Opc = Xtensa::EE_FFT_R2BF_S16_ST_INCP;
    MachineOperand &QA0 = MI.getOperand(0);
    unsigned QA0Val = QA0.getImm();
    assert(QA0Val < 8 && "Unexpected value of ee_fft_r2bf_s16_st_incp first "
                         "argument, it must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_fft_r2bf_s16_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_fft_r2bf_s16_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(3);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &SAR4 = MI.getOperand(4);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QA0Val)
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal)
        .addReg(AS.getReg())
        .addImm(SAR4.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_FFT_VST_R32_DECP_P: {
    unsigned Opc = Xtensa::EE_FFT_VST_R32_DECP;
    MachineOperand &QV = MI.getOperand(0);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of ee_fft_vst_r32_decp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &SAR2 = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QVVal)
        .addReg(AS.getReg())
        .addImm(SAR2.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_LDF_128_IP_P: {
    unsigned Opc = Xtensa::EE_LDF_128_IP;
    MachineOperand &FU3 = MI.getOperand(0);
    MachineOperand &FU2 = MI.getOperand(1);
    MachineOperand &FU1 = MI.getOperand(2);
    MachineOperand &FU0 = MI.getOperand(3);
    MachineOperand &AS = MI.getOperand(4);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &IMM16F = MI.getOperand(5);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(FU3.getReg())
        .addReg(FU2.getReg())
        .addReg(FU1.getReg())
        .addReg(FU0.getReg())
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addImm(IMM16F.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_LDF_128_XP_P: {
    unsigned Opc = Xtensa::EE_LDF_128_XP;
    MachineOperand &FU3 = MI.getOperand(0);
    MachineOperand &FU2 = MI.getOperand(1);
    MachineOperand &FU1 = MI.getOperand(2);
    MachineOperand &FU0 = MI.getOperand(3);
    MachineOperand &AS = MI.getOperand(4);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &AD = MI.getOperand(5);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(FU3.getReg())
        .addReg(FU2.getReg())
        .addReg(FU1.getReg())
        .addReg(FU0.getReg())
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addReg(AD.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_LDF_64_IP_P: {
    unsigned Opc = Xtensa::EE_LDF_64_IP;
    MachineOperand &FU1 = MI.getOperand(0);
    MachineOperand &FU0 = MI.getOperand(1);
    MachineOperand &AS = MI.getOperand(2);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &IMM8 = MI.getOperand(3);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(FU1.getReg())
        .addReg(FU0.getReg())
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addImm(IMM8.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_LDF_64_XP_P: {
    unsigned Opc = Xtensa::EE_LDF_64_XP;
    MachineOperand &FU1 = MI.getOperand(0);
    MachineOperand &FU0 = MI.getOperand(1);
    MachineOperand &AS = MI.getOperand(2);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &AD = MI.getOperand(3);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(FU1.getReg())
        .addReg(FU0.getReg())
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addReg(AD.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_LDQA_S16_128_IP_P: {
    unsigned Opc = Xtensa::EE_LDQA_S16_128_IP;
    MachineOperand &AS = MI.getOperand(0);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &IMM16 = MI.getOperand(1);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addImm(IMM16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_LDQA_S16_128_XP_P: {
    unsigned Opc = Xtensa::EE_LDQA_S16_128_XP;
    MachineOperand &AS = MI.getOperand(0);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &AD = MI.getOperand(1);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addReg(AD.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_LDQA_S8_128_IP_P: {
    unsigned Opc = Xtensa::EE_LDQA_S8_128_IP;
    MachineOperand &AS = MI.getOperand(0);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &IMM16 = MI.getOperand(1);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addImm(IMM16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_LDQA_S8_128_XP_P: {
    unsigned Opc = Xtensa::EE_LDQA_S8_128_XP;
    MachineOperand &AS = MI.getOperand(0);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &AD = MI.getOperand(1);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addReg(AD.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_LDQA_U16_128_IP_P: {
    unsigned Opc = Xtensa::EE_LDQA_U16_128_IP;
    MachineOperand &AS = MI.getOperand(0);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &IMM16 = MI.getOperand(1);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addImm(IMM16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_LDQA_U16_128_XP_P: {
    unsigned Opc = Xtensa::EE_LDQA_U16_128_XP;
    MachineOperand &AS = MI.getOperand(0);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &AD = MI.getOperand(1);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addReg(AD.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_LDQA_U8_128_IP_P: {
    unsigned Opc = Xtensa::EE_LDQA_U8_128_IP;
    MachineOperand &AS = MI.getOperand(0);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &IMM16 = MI.getOperand(1);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addImm(IMM16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_LDQA_U8_128_XP_P: {
    unsigned Opc = Xtensa::EE_LDQA_U8_128_XP;
    MachineOperand &AS = MI.getOperand(0);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &AD = MI.getOperand(1);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addReg(AD.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_LDXQ_32_P: {
    unsigned Opc = Xtensa::EE_LDXQ_32;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_ldxq_32 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QS = MI.getOperand(1);
    unsigned QSVal = QS.getImm();
    assert(QSVal < 8 && "Unexpected value of ee_ldxq_32 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(2);
    MachineOperand &SEL4 = MI.getOperand(3);
    MachineOperand &SEL8 = MI.getOperand(4);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(Xtensa::Q0 + QSVal)
        .addReg(AS.getReg())
        .addImm(SEL4.getImm())
        .addImm(SEL8.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_LD_128_USAR_IP_P: {
    unsigned Opc = Xtensa::EE_LD_128_USAR_IP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_ld_128_usar_ip first argument, "
                        "it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &IMM16 = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addImm(IMM16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_LD_128_USAR_XP_P: {
    unsigned Opc = Xtensa::EE_LD_128_USAR_XP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_ld_128_usar_xp first argument, "
                        "it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &AD = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addReg(AD.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_LD_ACCX_IP_P: {
    unsigned Opc = Xtensa::EE_LD_ACCX_IP;
    MachineOperand &AS = MI.getOperand(0);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &IMM8 = MI.getOperand(1);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addImm(IMM8.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_LD_QACC_H_H_32_IP_P: {
    unsigned Opc = Xtensa::EE_LD_QACC_H_H_32_IP;
    MachineOperand &AS = MI.getOperand(0);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &IMM4 = MI.getOperand(1);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addImm(IMM4.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_LD_QACC_H_L_128_IP_P: {
    unsigned Opc = Xtensa::EE_LD_QACC_H_L_128_IP;
    MachineOperand &AS = MI.getOperand(0);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &IMM16 = MI.getOperand(1);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addImm(IMM16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_LD_QACC_L_H_32_IP_P: {
    unsigned Opc = Xtensa::EE_LD_QACC_L_H_32_IP;
    MachineOperand &AS = MI.getOperand(0);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &IMM4 = MI.getOperand(1);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addImm(IMM4.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_LD_QACC_L_L_128_IP_P: {
    unsigned Opc = Xtensa::EE_LD_QACC_L_L_128_IP;
    MachineOperand &AS = MI.getOperand(0);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &IMM16 = MI.getOperand(1);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addImm(IMM16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_LD_UA_STATE_IP_P: {
    unsigned Opc = Xtensa::EE_LD_UA_STATE_IP;
    MachineOperand &AS = MI.getOperand(0);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &IMM16 = MI.getOperand(1);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addImm(IMM16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_MOVI_32_A_P: {
    unsigned Opc = Xtensa::EE_MOVI_32_A;
    MachineOperand &QS = MI.getOperand(0);
    unsigned QSVal = QS.getImm();
    assert(QSVal < 8 && "Unexpected value of ee_movi_32_a first argument, it "
                        "must be in range [0,7]");
    MachineOperand &AU = MI.getOperand(1);
    MachineOperand &SEL4 = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(AU.getReg())
        .addReg(Xtensa::Q0 + QSVal)
        .addImm(SEL4.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_MOVI_32_Q_P: {
    unsigned Opc = Xtensa::EE_MOVI_32_Q;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_movi_32_q first argument, it "
                        "must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    MachineOperand &SEL4 = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(AS.getReg())
        .addImm(SEL4.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_MOV_S16_QACC_P: {
    unsigned Opc = Xtensa::EE_MOV_S16_QACC;
    MachineOperand &QS = MI.getOperand(0);
    unsigned QSVal = QS.getImm();
    assert(QSVal < 8 && "Unexpected value of ee_mov_s16_qacc first argument, "
                        "it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc)).addReg(Xtensa::Q0 + QSVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_MOV_S8_QACC_P: {
    unsigned Opc = Xtensa::EE_MOV_S8_QACC;
    MachineOperand &QS = MI.getOperand(0);
    unsigned QSVal = QS.getImm();
    assert(QSVal < 8 && "Unexpected value of ee_mov_s8_qacc first argument, it "
                        "must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc)).addReg(Xtensa::Q0 + QSVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_MOV_U16_QACC_P: {
    unsigned Opc = Xtensa::EE_MOV_U16_QACC;
    MachineOperand &QS = MI.getOperand(0);
    unsigned QSVal = QS.getImm();
    assert(QSVal < 8 && "Unexpected value of ee_mov_u16_qacc first argument, "
                        "it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc)).addReg(Xtensa::Q0 + QSVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_MOV_U8_QACC_P: {
    unsigned Opc = Xtensa::EE_MOV_U8_QACC;
    MachineOperand &QS = MI.getOperand(0);
    unsigned QSVal = QS.getImm();
    assert(QSVal < 8 && "Unexpected value of ee_mov_u8_qacc first argument, it "
                        "must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc)).addReg(Xtensa::Q0 + QSVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_NOTQ_P: {
    unsigned Opc = Xtensa::EE_NOTQ;
    MachineOperand &QA = MI.getOperand(0);
    unsigned QAVal = QA.getImm();
    assert(QAVal < 8 && "Unexpected value of ee_notq first argument, it must "
                        "be in range [0,7]");
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_notq first argument, it must "
                        "be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QAVal)
        .addReg(Xtensa::Q0 + QXVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_ORQ_P: {
    unsigned Opc = Xtensa::EE_ORQ;
    MachineOperand &QA = MI.getOperand(0);
    unsigned QAVal = QA.getImm();
    assert(
        QAVal < 8 &&
        "Unexpected value of ee_orq first argument, it must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(
        QXVal < 8 &&
        "Unexpected value of ee_orq first argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(
        QYVal < 8 &&
        "Unexpected value of ee_orq first argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QAVal)
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_SLCI_2Q_P: {
    unsigned Opc = Xtensa::EE_SLCI_2Q;
    MachineOperand &QS1 = MI.getOperand(0);
    unsigned QS1Val = QS1.getImm();
    assert(QS1Val < 8 && "Unexpected value of ee_slci_2q first argument, it "
                         "must be in range [0,7]");
    MachineOperand &QS0 = MI.getOperand(1);
    unsigned QS0Val = QS0.getImm();
    assert(QS0Val < 8 && "Unexpected value of ee_slci_2q first argument, it "
                         "must be in range [0,7]");
    MachineOperand &SAR16 = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QS1Val)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(Xtensa::Q0 + QS1Val)
        .addReg(Xtensa::Q0 + QS0Val)
        .addImm(SAR16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_SLCXXP_2Q_P: {
    unsigned Opc = Xtensa::EE_SLCXXP_2Q;
    MachineOperand &QS1 = MI.getOperand(0);
    unsigned QS1Val = QS1.getImm();
    assert(QS1Val < 8 && "Unexpected value of ee_slcxxp_2q first argument, it "
                         "must be in range [0,7]");
    MachineOperand &QS0 = MI.getOperand(1);
    unsigned QS0Val = QS0.getImm();
    assert(QS0Val < 8 && "Unexpected value of ee_slcxxp_2q first argument, it "
                         "must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(2);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &AD = MI.getOperand(3);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QS1Val)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QS1Val)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(AS.getReg())
        .addReg(AD.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_SRCI_2Q_P: {
    unsigned Opc = Xtensa::EE_SRCI_2Q;
    MachineOperand &QS1 = MI.getOperand(0);
    unsigned QS1Val = QS1.getImm();
    assert(QS1Val < 8 && "Unexpected value of ee_srci_2q first argument, it "
                         "must be in range [0,7]");
    MachineOperand &QS0 = MI.getOperand(1);
    unsigned QS0Val = QS0.getImm();
    assert(QS0Val < 8 && "Unexpected value of ee_srci_2q first argument, it "
                         "must be in range [0,7]");
    MachineOperand &SAR16 = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QS1Val)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(Xtensa::Q0 + QS1Val)
        .addReg(Xtensa::Q0 + QS0Val)
        .addImm(SAR16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_SRCMB_S16_QACC_P: {
    unsigned Opc = Xtensa::EE_SRCMB_S16_QACC;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_srcmb_s16_qacc first argument, "
                        "it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    MachineOperand &SEL2 = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(AS.getReg())
        .addImm(SEL2.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_SRCMB_S8_QACC_P: {
    unsigned Opc = Xtensa::EE_SRCMB_S8_QACC;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_srcmb_s8_qacc first argument, "
                        "it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    MachineOperand &SEL2 = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(AS.getReg())
        .addImm(SEL2.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_SRCQ_128_ST_INCP_P: {
    unsigned Opc = Xtensa::EE_SRCQ_128_ST_INCP;
    MachineOperand &QS0 = MI.getOperand(0);
    unsigned QS0Val = QS0.getImm();
    assert(QS0Val < 8 && "Unexpected value of ee_srcq_128_st_incp first "
                         "argument, it must be in range [0,7]");
    MachineOperand &QS1 = MI.getOperand(1);
    unsigned QS1Val = QS1.getImm();
    assert(QS1Val < 8 && "Unexpected value of ee_srcq_128_st_incp first "
                         "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(2);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(Xtensa::Q0 + QS1Val)
        .addReg(AS.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_SRCXXP_2Q_P: {
    unsigned Opc = Xtensa::EE_SRCXXP_2Q;
    MachineOperand &QS1 = MI.getOperand(0);
    unsigned QS1Val = QS1.getImm();
    assert(QS1Val < 8 && "Unexpected value of ee_srcxxp_2q first argument, it "
                         "must be in range [0,7]");
    MachineOperand &QS0 = MI.getOperand(1);
    unsigned QS0Val = QS0.getImm();
    assert(QS0Val < 8 && "Unexpected value of ee_srcxxp_2q first argument, it "
                         "must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(2);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &AD = MI.getOperand(3);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QS1Val)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QS1Val)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(AS.getReg())
        .addReg(AD.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_SRC_Q_P: {
    unsigned Opc = Xtensa::EE_SRC_Q;
    MachineOperand &QA = MI.getOperand(0);
    unsigned QAVal = QA.getImm();
    assert(QAVal < 8 && "Unexpected value of ee_src_q first argument, it must "
                        "be in range [0,7]");
    MachineOperand &QS0 = MI.getOperand(1);
    unsigned QS0Val = QS0.getImm();
    assert(QS0Val < 8 && "Unexpected value of ee_src_q first argument, it must "
                         "be in range [0,7]");
    MachineOperand &QS1 = MI.getOperand(2);
    unsigned QS1Val = QS1.getImm();
    assert(QS1Val < 8 && "Unexpected value of ee_src_q first argument, it must "
                         "be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QAVal)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(Xtensa::Q0 + QS1Val);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_SRC_Q_LD_IP_P: {
    unsigned Opc = Xtensa::EE_SRC_Q_LD_IP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_src_q_ld_ip first argument, it "
                        "must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &IMM16 = MI.getOperand(2);
    MachineOperand &QS0 = MI.getOperand(3);
    unsigned QS0Val = QS0.getImm();
    assert(QS0Val < 8 && "Unexpected value of ee_src_q_ld_ip first argument, "
                         "it must be in range [0,7]");
    MachineOperand &QS1 = MI.getOperand(4);
    unsigned QS1Val = QS1.getImm();
    assert(QS1Val < 8 && "Unexpected value of ee_src_q_ld_ip first argument, "
                         "it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(AS.getReg())
        .addImm(IMM16.getImm())
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(Xtensa::Q0 + QS1Val);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_SRC_Q_LD_XP_P: {
    unsigned Opc = Xtensa::EE_SRC_Q_LD_XP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_src_q_ld_xp first argument, it "
                        "must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &AD = MI.getOperand(2);
    MachineOperand &QS0 = MI.getOperand(3);
    unsigned QS0Val = QS0.getImm();
    assert(QS0Val < 8 && "Unexpected value of ee_src_q_ld_xp first argument, "
                         "it must be in range [0,7]");
    MachineOperand &QS1 = MI.getOperand(4);
    unsigned QS1Val = QS1.getImm();
    assert(QS1Val < 8 && "Unexpected value of ee_src_q_ld_xp first argument, "
                         "it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(AS.getReg())
        .addReg(AD.getReg())
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(Xtensa::Q0 + QS1Val);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_SRC_Q_QUP_P: {
    unsigned Opc = Xtensa::EE_SRC_Q_QUP;
    MachineOperand &QA = MI.getOperand(0);
    unsigned QAVal = QA.getImm();
    assert(QAVal < 8 && "Unexpected value of ee_src_q_qup first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QS0 = MI.getOperand(1);
    unsigned QS0Val = QS0.getImm();
    assert(QS0Val < 8 && "Unexpected value of ee_src_q_qup first argument, it "
                         "must be in range [0,7]");
    MachineOperand &QS1 = MI.getOperand(2);
    unsigned QS1Val = QS1.getImm();
    assert(QS1Val < 8 && "Unexpected value of ee_src_q_qup first argument, it "
                         "must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QAVal)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(Xtensa::Q0 + QS1Val);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_SRS_ACCX_P: {
    unsigned Opc = Xtensa::EE_SRS_ACCX;
    MachineOperand &AU = MI.getOperand(0);
    MachineOperand &AS = MI.getOperand(1);
    MachineOperand &SEL2 = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(AU.getReg())
        .addReg(AS.getReg())
        .addImm(SEL2.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_STF_128_IP_P: {
    unsigned Opc = Xtensa::EE_STF_128_IP;
    MachineOperand &FV3 = MI.getOperand(0);
    MachineOperand &FV2 = MI.getOperand(1);
    MachineOperand &FV1 = MI.getOperand(2);
    MachineOperand &FV0 = MI.getOperand(3);
    MachineOperand &AS = MI.getOperand(4);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &IMM16F = MI.getOperand(5);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Undef)
        .addReg(FV3.getReg())
        .addReg(FV2.getReg())
        .addReg(FV1.getReg())
        .addReg(FV0.getReg())
        .addReg(AS.getReg())
        .addImm(IMM16F.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_STF_128_XP_P: {
    unsigned Opc = Xtensa::EE_STF_128_XP;
    MachineOperand &FV3 = MI.getOperand(0);
    MachineOperand &FV2 = MI.getOperand(1);
    MachineOperand &FV1 = MI.getOperand(2);
    MachineOperand &FV0 = MI.getOperand(3);
    MachineOperand &AS = MI.getOperand(4);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &AD = MI.getOperand(5);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Undef)
        .addReg(FV3.getReg())
        .addReg(FV2.getReg())
        .addReg(FV1.getReg())
        .addReg(FV0.getReg())
        .addReg(AS.getReg())
        .addReg(AD.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_STF_64_IP_P: {
    unsigned Opc = Xtensa::EE_STF_64_IP;
    MachineOperand &FV1 = MI.getOperand(0);
    MachineOperand &FV0 = MI.getOperand(1);
    MachineOperand &AS = MI.getOperand(2);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &IMM8 = MI.getOperand(3);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Undef)
        .addReg(FV1.getReg())
        .addReg(FV0.getReg())
        .addReg(AS.getReg())
        .addImm(IMM8.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_STF_64_XP_P: {
    unsigned Opc = Xtensa::EE_STF_64_XP;
    MachineOperand &FV1 = MI.getOperand(0);
    MachineOperand &FV0 = MI.getOperand(1);
    MachineOperand &AS = MI.getOperand(2);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &AD = MI.getOperand(3);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Undef)
        .addReg(FV1.getReg())
        .addReg(FV0.getReg())
        .addReg(AS.getReg())
        .addReg(AD.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_STXQ_32_P: {
    unsigned Opc = Xtensa::EE_STXQ_32;
    MachineOperand &QV = MI.getOperand(0);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of ee_stxq_32 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QS = MI.getOperand(1);
    unsigned QSVal = QS.getImm();
    assert(QSVal < 8 && "Unexpected value of ee_stxq_32 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(2);
    MachineOperand &SEL4 = MI.getOperand(3);
    MachineOperand &SEL8 = MI.getOperand(4);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QVVal)
        .addReg(Xtensa::Q0 + QSVal)
        .addReg(AS.getReg())
        .addImm(SEL4.getImm())
        .addImm(SEL8.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_ST_ACCX_IP_P: {
    unsigned Opc = Xtensa::EE_ST_ACCX_IP;
    MachineOperand &AS = MI.getOperand(0);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &IMM8 = MI.getOperand(1);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addImm(IMM8.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_ST_QACC_H_H_32_IP_P: {
    unsigned Opc = Xtensa::EE_ST_QACC_H_H_32_IP;
    MachineOperand &AS = MI.getOperand(0);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &IMM4 = MI.getOperand(1);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addImm(IMM4.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_ST_QACC_H_L_128_IP_P: {
    unsigned Opc = Xtensa::EE_ST_QACC_H_L_128_IP;
    MachineOperand &AS = MI.getOperand(0);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &IMM16 = MI.getOperand(1);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addImm(IMM16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_ST_QACC_L_H_32_IP_P: {
    unsigned Opc = Xtensa::EE_ST_QACC_L_H_32_IP;
    MachineOperand &AS = MI.getOperand(0);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &IMM4 = MI.getOperand(1);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addImm(IMM4.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_ST_QACC_L_L_128_IP_P: {
    unsigned Opc = Xtensa::EE_ST_QACC_L_L_128_IP;
    MachineOperand &AS = MI.getOperand(0);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &IMM16 = MI.getOperand(1);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addImm(IMM16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_ST_UA_STATE_IP_P: {
    unsigned Opc = Xtensa::EE_ST_UA_STATE_IP;
    MachineOperand &AS = MI.getOperand(0);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &IMM16 = MI.getOperand(1);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addImm(IMM16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VADDS_S16_P: {
    unsigned Opc = Xtensa::EE_VADDS_S16;
    MachineOperand &QA = MI.getOperand(0);
    unsigned QAVal = QA.getImm();
    assert(QAVal < 8 && "Unexpected value of ee_vadds_s16 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vadds_s16 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vadds_s16 first argument, it "
                        "must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QAVal)
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VADDS_S16_LD_INCP_P: {
    unsigned Opc = Xtensa::EE_VADDS_S16_LD_INCP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vadds_s16_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &QA = MI.getOperand(2);
    unsigned QAVal = QA.getImm();
    assert(QAVal < 8 && "Unexpected value of ee_vadds_s16_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vadds_s16_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vadds_s16_ld_incp first "
                        "argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QAVal)
        .addReg(AS.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VADDS_S16_ST_INCP_P: {
    unsigned Opc = Xtensa::EE_VADDS_S16_ST_INCP;
    MachineOperand &QV = MI.getOperand(0);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of ee_vadds_s16_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &QA = MI.getOperand(2);
    unsigned QAVal = QA.getImm();
    assert(QAVal < 8 && "Unexpected value of ee_vadds_s16_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vadds_s16_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vadds_s16_st_incp first "
                        "argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QAVal)
        .addReg(Xtensa::Q0 + QVVal)
        .addReg(AS.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VADDS_S32_P: {
    unsigned Opc = Xtensa::EE_VADDS_S32;
    MachineOperand &QA = MI.getOperand(0);
    unsigned QAVal = QA.getImm();
    assert(QAVal < 8 && "Unexpected value of ee_vadds_s32 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vadds_s32 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vadds_s32 first argument, it "
                        "must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QAVal)
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VADDS_S32_LD_INCP_P: {
    unsigned Opc = Xtensa::EE_VADDS_S32_LD_INCP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vadds_s32_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &QA = MI.getOperand(2);
    unsigned QAVal = QA.getImm();
    assert(QAVal < 8 && "Unexpected value of ee_vadds_s32_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vadds_s32_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vadds_s32_ld_incp first "
                        "argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QAVal)
        .addReg(AS.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VADDS_S32_ST_INCP_P: {
    unsigned Opc = Xtensa::EE_VADDS_S32_ST_INCP;
    MachineOperand &QV = MI.getOperand(0);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of ee_vadds_s32_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &QA = MI.getOperand(2);
    unsigned QAVal = QA.getImm();
    assert(QAVal < 8 && "Unexpected value of ee_vadds_s32_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vadds_s32_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vadds_s32_st_incp first "
                        "argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QAVal)
        .addReg(Xtensa::Q0 + QVVal)
        .addReg(AS.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VADDS_S8_P: {
    unsigned Opc = Xtensa::EE_VADDS_S8;
    MachineOperand &QA = MI.getOperand(0);
    unsigned QAVal = QA.getImm();
    assert(QAVal < 8 && "Unexpected value of ee_vadds_s8 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vadds_s8 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vadds_s8 first argument, it "
                        "must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QAVal)
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VADDS_S8_LD_INCP_P: {
    unsigned Opc = Xtensa::EE_VADDS_S8_LD_INCP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vadds_s8_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &QA = MI.getOperand(2);
    unsigned QAVal = QA.getImm();
    assert(QAVal < 8 && "Unexpected value of ee_vadds_s8_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vadds_s8_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vadds_s8_ld_incp first "
                        "argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QAVal)
        .addReg(AS.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VADDS_S8_ST_INCP_P: {
    unsigned Opc = Xtensa::EE_VADDS_S8_ST_INCP;
    MachineOperand &QV = MI.getOperand(0);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of ee_vadds_s8_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &QA = MI.getOperand(2);
    unsigned QAVal = QA.getImm();
    assert(QAVal < 8 && "Unexpected value of ee_vadds_s8_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vadds_s8_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vadds_s8_st_incp first "
                        "argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QAVal)
        .addReg(Xtensa::Q0 + QVVal)
        .addReg(AS.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VCMP_EQ_S16_P: {
    unsigned Opc = Xtensa::EE_VCMP_EQ_S16;
    MachineOperand &QA = MI.getOperand(0);
    unsigned QAVal = QA.getImm();
    assert(QAVal < 8 && "Unexpected value of ee_vcmp_eq_s16 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vcmp_eq_s16 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vcmp_eq_s16 first argument, it "
                        "must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QAVal)
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VCMP_EQ_S32_P: {
    unsigned Opc = Xtensa::EE_VCMP_EQ_S32;
    MachineOperand &QA = MI.getOperand(0);
    unsigned QAVal = QA.getImm();
    assert(QAVal < 8 && "Unexpected value of ee_vcmp_eq_s32 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vcmp_eq_s32 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vcmp_eq_s32 first argument, it "
                        "must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QAVal)
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VCMP_EQ_S8_P: {
    unsigned Opc = Xtensa::EE_VCMP_EQ_S8;
    MachineOperand &QA = MI.getOperand(0);
    unsigned QAVal = QA.getImm();
    assert(QAVal < 8 && "Unexpected value of ee_vcmp_eq_s8 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vcmp_eq_s8 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vcmp_eq_s8 first argument, it "
                        "must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QAVal)
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VCMP_GT_S16_P: {
    unsigned Opc = Xtensa::EE_VCMP_GT_S16;
    MachineOperand &QA = MI.getOperand(0);
    unsigned QAVal = QA.getImm();
    assert(QAVal < 8 && "Unexpected value of ee_vcmp_gt_s16 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vcmp_gt_s16 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vcmp_gt_s16 first argument, it "
                        "must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QAVal)
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VCMP_GT_S32_P: {
    unsigned Opc = Xtensa::EE_VCMP_GT_S32;
    MachineOperand &QA = MI.getOperand(0);
    unsigned QAVal = QA.getImm();
    assert(QAVal < 8 && "Unexpected value of ee_vcmp_gt_s32 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vcmp_gt_s32 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vcmp_gt_s32 first argument, it "
                        "must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QAVal)
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VCMP_GT_S8_P: {
    unsigned Opc = Xtensa::EE_VCMP_GT_S8;
    MachineOperand &QA = MI.getOperand(0);
    unsigned QAVal = QA.getImm();
    assert(QAVal < 8 && "Unexpected value of ee_vcmp_gt_s8 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vcmp_gt_s8 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vcmp_gt_s8 first argument, it "
                        "must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QAVal)
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VCMP_LT_S16_P: {
    unsigned Opc = Xtensa::EE_VCMP_LT_S16;
    MachineOperand &QA = MI.getOperand(0);
    unsigned QAVal = QA.getImm();
    assert(QAVal < 8 && "Unexpected value of ee_vcmp_lt_s16 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vcmp_lt_s16 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vcmp_lt_s16 first argument, it "
                        "must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QAVal)
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VCMP_LT_S32_P: {
    unsigned Opc = Xtensa::EE_VCMP_LT_S32;
    MachineOperand &QA = MI.getOperand(0);
    unsigned QAVal = QA.getImm();
    assert(QAVal < 8 && "Unexpected value of ee_vcmp_lt_s32 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vcmp_lt_s32 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vcmp_lt_s32 first argument, it "
                        "must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QAVal)
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VCMP_LT_S8_P: {
    unsigned Opc = Xtensa::EE_VCMP_LT_S8;
    MachineOperand &QA = MI.getOperand(0);
    unsigned QAVal = QA.getImm();
    assert(QAVal < 8 && "Unexpected value of ee_vcmp_lt_s8 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vcmp_lt_s8 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vcmp_lt_s8 first argument, it "
                        "must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QAVal)
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VLDBC_16_P: {
    unsigned Opc = Xtensa::EE_VLDBC_16;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vldbc_16 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(AS.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VLDBC_16_IP_P: {
    unsigned Opc = Xtensa::EE_VLDBC_16_IP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vldbc_16_ip first argument, it "
                        "must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &IMM2 = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addImm(IMM2.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VLDBC_16_XP_P: {
    unsigned Opc = Xtensa::EE_VLDBC_16_XP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vldbc_16_xp first argument, it "
                        "must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &AD = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addReg(AD.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VLDBC_32_P: {
    unsigned Opc = Xtensa::EE_VLDBC_32;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vldbc_32 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(AS.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VLDBC_32_IP_P: {
    unsigned Opc = Xtensa::EE_VLDBC_32_IP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vldbc_32_ip first argument, it "
                        "must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &IMM4 = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addImm(IMM4.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VLDBC_32_XP_P: {
    unsigned Opc = Xtensa::EE_VLDBC_32_XP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vldbc_32_xp first argument, it "
                        "must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &AD = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addReg(AD.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VLDBC_8_P: {
    unsigned Opc = Xtensa::EE_VLDBC_8;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vldbc_8 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(AS.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VLDBC_8_IP_P: {
    unsigned Opc = Xtensa::EE_VLDBC_8_IP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vldbc_8_ip first argument, it "
                        "must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &IMM1 = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addImm(IMM1.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VLDBC_8_XP_P: {
    unsigned Opc = Xtensa::EE_VLDBC_8_XP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vldbc_8_xp first argument, it "
                        "must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &AD = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addReg(AD.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VLDHBC_16_INCP_P: {
    unsigned Opc = Xtensa::EE_VLDHBC_16_INCP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vldhbc_16_incp first argument, "
                        "it must be in range [0,7]");
    MachineOperand &QU1 = MI.getOperand(1);
    unsigned QU1Val = QU1.getImm();
    assert(QU1Val < 8 && "Unexpected value of ee_vldhbc_16_incp first "
                         "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(2);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(Xtensa::Q0 + QU1Val)
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VLD_128_IP_P: {
    unsigned Opc = Xtensa::EE_VLD_128_IP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vld_128_ip first argument, it "
                        "must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &IMM16 = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addImm(IMM16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VLD_128_XP_P: {
    unsigned Opc = Xtensa::EE_VLD_128_XP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vld_128_xp first argument, it "
                        "must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &AD = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addReg(AD.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VLD_H_64_IP_P: {
    unsigned Opc = Xtensa::EE_VLD_H_64_IP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vld_h_64_ip first argument, it "
                        "must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &IMM8 = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addImm(IMM8.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VLD_H_64_XP_P: {
    unsigned Opc = Xtensa::EE_VLD_H_64_XP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vld_h_64_xp first argument, it "
                        "must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &AD = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addReg(AD.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VLD_L_64_IP_P: {
    unsigned Opc = Xtensa::EE_VLD_L_64_IP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vld_l_64_ip first argument, it "
                        "must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &IMM8 = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addImm(IMM8.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VLD_L_64_XP_P: {
    unsigned Opc = Xtensa::EE_VLD_L_64_XP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vld_l_64_xp first argument, it "
                        "must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &AD = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addReg(AD.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMAX_S16_P: {
    unsigned Opc = Xtensa::EE_VMAX_S16;
    MachineOperand &QA = MI.getOperand(0);
    unsigned QAVal = QA.getImm();
    assert(QAVal < 8 && "Unexpected value of ee_vmax_s16 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmax_s16 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmax_s16 first argument, it "
                        "must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QAVal)
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMAX_S16_LD_INCP_P: {
    unsigned Opc = Xtensa::EE_VMAX_S16_LD_INCP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vmax_s16_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &QA = MI.getOperand(2);
    unsigned QAVal = QA.getImm();
    assert(QAVal < 8 && "Unexpected value of ee_vmax_s16_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmax_s16_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmax_s16_ld_incp first "
                        "argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QAVal)
        .addReg(AS.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMAX_S16_ST_INCP_P: {
    unsigned Opc = Xtensa::EE_VMAX_S16_ST_INCP;
    MachineOperand &QV = MI.getOperand(0);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of ee_vmax_s16_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &QA = MI.getOperand(2);
    unsigned QAVal = QA.getImm();
    assert(QAVal < 8 && "Unexpected value of ee_vmax_s16_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmax_s16_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmax_s16_st_incp first "
                        "argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QAVal)
        .addReg(Xtensa::Q0 + QVVal)
        .addReg(AS.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMAX_S32_P: {
    unsigned Opc = Xtensa::EE_VMAX_S32;
    MachineOperand &QA = MI.getOperand(0);
    unsigned QAVal = QA.getImm();
    assert(QAVal < 8 && "Unexpected value of ee_vmax_s32 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmax_s32 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmax_s32 first argument, it "
                        "must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QAVal)
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMAX_S32_LD_INCP_P: {
    unsigned Opc = Xtensa::EE_VMAX_S32_LD_INCP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vmax_s32_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &QA = MI.getOperand(2);
    unsigned QAVal = QA.getImm();
    assert(QAVal < 8 && "Unexpected value of ee_vmax_s32_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmax_s32_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmax_s32_ld_incp first "
                        "argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QAVal)
        .addReg(AS.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMAX_S32_ST_INCP_P: {
    unsigned Opc = Xtensa::EE_VMAX_S32_ST_INCP;
    MachineOperand &QV = MI.getOperand(0);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of ee_vmax_s32_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &QA = MI.getOperand(2);
    unsigned QAVal = QA.getImm();
    assert(QAVal < 8 && "Unexpected value of ee_vmax_s32_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmax_s32_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmax_s32_st_incp first "
                        "argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QAVal)
        .addReg(Xtensa::Q0 + QVVal)
        .addReg(AS.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMAX_S8_P: {
    unsigned Opc = Xtensa::EE_VMAX_S8;
    MachineOperand &QA = MI.getOperand(0);
    unsigned QAVal = QA.getImm();
    assert(QAVal < 8 && "Unexpected value of ee_vmax_s8 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmax_s8 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmax_s8 first argument, it "
                        "must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QAVal)
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMAX_S8_LD_INCP_P: {
    unsigned Opc = Xtensa::EE_VMAX_S8_LD_INCP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vmax_s8_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &QA = MI.getOperand(2);
    unsigned QAVal = QA.getImm();
    assert(QAVal < 8 && "Unexpected value of ee_vmax_s8_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmax_s8_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmax_s8_ld_incp first "
                        "argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QAVal)
        .addReg(AS.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMAX_S8_ST_INCP_P: {
    unsigned Opc = Xtensa::EE_VMAX_S8_ST_INCP;
    MachineOperand &QV = MI.getOperand(0);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of ee_vmax_s8_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &QA = MI.getOperand(2);
    unsigned QAVal = QA.getImm();
    assert(QAVal < 8 && "Unexpected value of ee_vmax_s8_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmax_s8_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmax_s8_st_incp first "
                        "argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QAVal)
        .addReg(Xtensa::Q0 + QVVal)
        .addReg(AS.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMIN_S16_P: {
    unsigned Opc = Xtensa::EE_VMIN_S16;
    MachineOperand &QA = MI.getOperand(0);
    unsigned QAVal = QA.getImm();
    assert(QAVal < 8 && "Unexpected value of ee_vmin_s16 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmin_s16 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmin_s16 first argument, it "
                        "must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QAVal)
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMIN_S16_LD_INCP_P: {
    unsigned Opc = Xtensa::EE_VMIN_S16_LD_INCP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vmin_s16_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &QA = MI.getOperand(2);
    unsigned QAVal = QA.getImm();
    assert(QAVal < 8 && "Unexpected value of ee_vmin_s16_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmin_s16_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmin_s16_ld_incp first "
                        "argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QAVal)
        .addReg(AS.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMIN_S16_ST_INCP_P: {
    unsigned Opc = Xtensa::EE_VMIN_S16_ST_INCP;
    MachineOperand &QV = MI.getOperand(0);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of ee_vmin_s16_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &QA = MI.getOperand(2);
    unsigned QAVal = QA.getImm();
    assert(QAVal < 8 && "Unexpected value of ee_vmin_s16_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmin_s16_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmin_s16_st_incp first "
                        "argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QAVal)
        .addReg(Xtensa::Q0 + QVVal)
        .addReg(AS.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMIN_S32_P: {
    unsigned Opc = Xtensa::EE_VMIN_S32;
    MachineOperand &QA = MI.getOperand(0);
    unsigned QAVal = QA.getImm();
    assert(QAVal < 8 && "Unexpected value of ee_vmin_s32 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmin_s32 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmin_s32 first argument, it "
                        "must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QAVal)
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMIN_S32_LD_INCP_P: {
    unsigned Opc = Xtensa::EE_VMIN_S32_LD_INCP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vmin_s32_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &QA = MI.getOperand(2);
    unsigned QAVal = QA.getImm();
    assert(QAVal < 8 && "Unexpected value of ee_vmin_s32_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmin_s32_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmin_s32_ld_incp first "
                        "argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QAVal)
        .addReg(AS.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMIN_S32_ST_INCP_P: {
    unsigned Opc = Xtensa::EE_VMIN_S32_ST_INCP;
    MachineOperand &QV = MI.getOperand(0);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of ee_vmin_s32_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &QA = MI.getOperand(2);
    unsigned QAVal = QA.getImm();
    assert(QAVal < 8 && "Unexpected value of ee_vmin_s32_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmin_s32_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmin_s32_st_incp first "
                        "argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QAVal)
        .addReg(Xtensa::Q0 + QVVal)
        .addReg(AS.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMIN_S8_P: {
    unsigned Opc = Xtensa::EE_VMIN_S8;
    MachineOperand &QA = MI.getOperand(0);
    unsigned QAVal = QA.getImm();
    assert(QAVal < 8 && "Unexpected value of ee_vmin_s8 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmin_s8 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmin_s8 first argument, it "
                        "must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QAVal)
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMIN_S8_LD_INCP_P: {
    unsigned Opc = Xtensa::EE_VMIN_S8_LD_INCP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vmin_s8_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &QA = MI.getOperand(2);
    unsigned QAVal = QA.getImm();
    assert(QAVal < 8 && "Unexpected value of ee_vmin_s8_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmin_s8_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmin_s8_ld_incp first "
                        "argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QAVal)
        .addReg(AS.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMIN_S8_ST_INCP_P: {
    unsigned Opc = Xtensa::EE_VMIN_S8_ST_INCP;
    MachineOperand &QV = MI.getOperand(0);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of ee_vmin_s8_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &QA = MI.getOperand(2);
    unsigned QAVal = QA.getImm();
    assert(QAVal < 8 && "Unexpected value of ee_vmin_s8_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmin_s8_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmin_s8_st_incp first "
                        "argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QAVal)
        .addReg(Xtensa::Q0 + QVVal)
        .addReg(AS.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMULAS_S16_ACCX_P: {
    unsigned Opc = Xtensa::EE_VMULAS_S16_ACCX;
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmulas_s16_accx first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmulas_s16_accx first "
                        "argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMULAS_S16_ACCX_LD_IP_P: {
    unsigned Opc = Xtensa::EE_VMULAS_S16_ACCX_LD_IP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vmulas_s16_accx_ld_ip first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &IMM16 = MI.getOperand(2);
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmulas_s16_accx_ld_ip first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmulas_s16_accx_ld_ip first "
                        "argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addImm(IMM16.getImm())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMULAS_S16_ACCX_LD_IP_QUP_P: {
    unsigned Opc = Xtensa::EE_VMULAS_S16_ACCX_LD_IP_QUP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vmulas_s16_accx_ld_ip_qup "
                        "first argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &IMM16 = MI.getOperand(2);
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmulas_s16_accx_ld_ip_qup "
                        "first argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmulas_s16_accx_ld_ip_qup "
                        "first argument, it must be in range [0,7]");
    MachineOperand &QS0 = MI.getOperand(5);
    unsigned QS0Val = QS0.getImm();
    assert(QS0Val < 8 && "Unexpected value of ee_vmulas_s16_accx_ld_ip_qup "
                         "first argument, it must be in range [0,7]");
    MachineOperand &QS1 = MI.getOperand(6);
    unsigned QS1Val = QS1.getImm();
    assert(QS1Val < 8 && "Unexpected value of ee_vmulas_s16_accx_ld_ip_qup "
                         "first argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(AS.getReg())
        .addImm(IMM16.getImm())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(Xtensa::Q0 + QS1Val);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMULAS_S16_ACCX_LD_XP_P: {
    unsigned Opc = Xtensa::EE_VMULAS_S16_ACCX_LD_XP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vmulas_s16_accx_ld_xp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &AD = MI.getOperand(2);
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmulas_s16_accx_ld_xp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmulas_s16_accx_ld_xp first "
                        "argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addReg(AD.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMULAS_S16_ACCX_LD_XP_QUP_P: {
    unsigned Opc = Xtensa::EE_VMULAS_S16_ACCX_LD_XP_QUP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vmulas_s16_accx_ld_xp_qup "
                        "first argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &AD = MI.getOperand(2);
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmulas_s16_accx_ld_xp_qup "
                        "first argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmulas_s16_accx_ld_xp_qup "
                        "first argument, it must be in range [0,7]");
    MachineOperand &QS0 = MI.getOperand(5);
    unsigned QS0Val = QS0.getImm();
    assert(QS0Val < 8 && "Unexpected value of ee_vmulas_s16_accx_ld_xp_qup "
                         "first argument, it must be in range [0,7]");
    MachineOperand &QS1 = MI.getOperand(6);
    unsigned QS1Val = QS1.getImm();
    assert(QS1Val < 8 && "Unexpected value of ee_vmulas_s16_accx_ld_xp_qup "
                         "first argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(AS.getReg())
        .addReg(AD.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(Xtensa::Q0 + QS1Val);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMULAS_S16_QACC_P: {
    unsigned Opc = Xtensa::EE_VMULAS_S16_QACC;
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmulas_s16_qacc first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmulas_s16_qacc first "
                        "argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMULAS_S16_QACC_LDBC_INCP_P: {
    unsigned Opc = Xtensa::EE_VMULAS_S16_QACC_LDBC_INCP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vmulas_s16_qacc_ldbc_incp "
                        "first argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &QX = MI.getOperand(2);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmulas_s16_qacc_ldbc_incp "
                        "first argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(3);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmulas_s16_qacc_ldbc_incp "
                        "first argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMULAS_S16_QACC_LDBC_INCP_QUP_P: {
    unsigned Opc = Xtensa::EE_VMULAS_S16_QACC_LDBC_INCP_QUP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vmulas_s16_qacc_ldbc_incp_qup "
                        "first argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &QX = MI.getOperand(2);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmulas_s16_qacc_ldbc_incp_qup "
                        "first argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(3);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmulas_s16_qacc_ldbc_incp_qup "
                        "first argument, it must be in range [0,7]");
    MachineOperand &QS0 = MI.getOperand(4);
    unsigned QS0Val = QS0.getImm();
    assert(QS0Val < 8 && "Unexpected value of ee_vmulas_s16_qacc_ldbc_incp_qup "
                         "first argument, it must be in range [0,7]");
    MachineOperand &QS1 = MI.getOperand(5);
    unsigned QS1Val = QS1.getImm();
    assert(QS1Val < 8 && "Unexpected value of ee_vmulas_s16_qacc_ldbc_incp_qup "
                         "first argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(AS.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(Xtensa::Q0 + QS1Val);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMULAS_S16_QACC_LD_IP_P: {
    unsigned Opc = Xtensa::EE_VMULAS_S16_QACC_LD_IP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vmulas_s16_qacc_ld_ip first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &IMM16 = MI.getOperand(2);
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmulas_s16_qacc_ld_ip first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmulas_s16_qacc_ld_ip first "
                        "argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addImm(IMM16.getImm())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMULAS_S16_QACC_LD_IP_QUP_P: {
    unsigned Opc = Xtensa::EE_VMULAS_S16_QACC_LD_IP_QUP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vmulas_s16_qacc_ld_ip_qup "
                        "first argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &IMM16 = MI.getOperand(2);
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmulas_s16_qacc_ld_ip_qup "
                        "first argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmulas_s16_qacc_ld_ip_qup "
                        "first argument, it must be in range [0,7]");
    MachineOperand &QS0 = MI.getOperand(5);
    unsigned QS0Val = QS0.getImm();
    assert(QS0Val < 8 && "Unexpected value of ee_vmulas_s16_qacc_ld_ip_qup "
                         "first argument, it must be in range [0,7]");
    MachineOperand &QS1 = MI.getOperand(6);
    unsigned QS1Val = QS1.getImm();
    assert(QS1Val < 8 && "Unexpected value of ee_vmulas_s16_qacc_ld_ip_qup "
                         "first argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(AS.getReg())
        .addImm(IMM16.getImm())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(Xtensa::Q0 + QS1Val);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMULAS_S16_QACC_LD_XP_P: {
    unsigned Opc = Xtensa::EE_VMULAS_S16_QACC_LD_XP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vmulas_s16_qacc_ld_xp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &AD = MI.getOperand(2);
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmulas_s16_qacc_ld_xp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmulas_s16_qacc_ld_xp first "
                        "argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addReg(AD.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMULAS_S16_QACC_LD_XP_QUP_P: {
    unsigned Opc = Xtensa::EE_VMULAS_S16_QACC_LD_XP_QUP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vmulas_s16_qacc_ld_xp_qup "
                        "first argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &AD = MI.getOperand(2);
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmulas_s16_qacc_ld_xp_qup "
                        "first argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmulas_s16_qacc_ld_xp_qup "
                        "first argument, it must be in range [0,7]");
    MachineOperand &QS0 = MI.getOperand(5);
    unsigned QS0Val = QS0.getImm();
    assert(QS0Val < 8 && "Unexpected value of ee_vmulas_s16_qacc_ld_xp_qup "
                         "first argument, it must be in range [0,7]");
    MachineOperand &QS1 = MI.getOperand(6);
    unsigned QS1Val = QS1.getImm();
    assert(QS1Val < 8 && "Unexpected value of ee_vmulas_s16_qacc_ld_xp_qup "
                         "first argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(AS.getReg())
        .addReg(AD.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(Xtensa::Q0 + QS1Val);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMULAS_S8_ACCX_P: {
    unsigned Opc = Xtensa::EE_VMULAS_S8_ACCX;
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmulas_s8_accx first argument, "
                        "it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmulas_s8_accx first argument, "
                        "it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMULAS_S8_ACCX_LD_IP_P: {
    unsigned Opc = Xtensa::EE_VMULAS_S8_ACCX_LD_IP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vmulas_s8_accx_ld_ip first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &IMM16 = MI.getOperand(2);
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmulas_s8_accx_ld_ip first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmulas_s8_accx_ld_ip first "
                        "argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addImm(IMM16.getImm())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMULAS_S8_ACCX_LD_IP_QUP_P: {
    unsigned Opc = Xtensa::EE_VMULAS_S8_ACCX_LD_IP_QUP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vmulas_s8_accx_ld_ip_qup first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &IMM16 = MI.getOperand(2);
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmulas_s8_accx_ld_ip_qup first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmulas_s8_accx_ld_ip_qup first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QS0 = MI.getOperand(5);
    unsigned QS0Val = QS0.getImm();
    assert(QS0Val < 8 && "Unexpected value of ee_vmulas_s8_accx_ld_ip_qup "
                         "first argument, it must be in range [0,7]");
    MachineOperand &QS1 = MI.getOperand(6);
    unsigned QS1Val = QS1.getImm();
    assert(QS1Val < 8 && "Unexpected value of ee_vmulas_s8_accx_ld_ip_qup "
                         "first argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(AS.getReg())
        .addImm(IMM16.getImm())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(Xtensa::Q0 + QS1Val);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMULAS_S8_ACCX_LD_XP_P: {
    unsigned Opc = Xtensa::EE_VMULAS_S8_ACCX_LD_XP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vmulas_s8_accx_ld_xp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &AD = MI.getOperand(2);
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmulas_s8_accx_ld_xp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmulas_s8_accx_ld_xp first "
                        "argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addReg(AD.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMULAS_S8_ACCX_LD_XP_QUP_P: {
    unsigned Opc = Xtensa::EE_VMULAS_S8_ACCX_LD_XP_QUP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vmulas_s8_accx_ld_xp_qup first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &AD = MI.getOperand(2);
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmulas_s8_accx_ld_xp_qup first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmulas_s8_accx_ld_xp_qup first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QS0 = MI.getOperand(5);
    unsigned QS0Val = QS0.getImm();
    assert(QS0Val < 8 && "Unexpected value of ee_vmulas_s8_accx_ld_xp_qup "
                         "first argument, it must be in range [0,7]");
    MachineOperand &QS1 = MI.getOperand(6);
    unsigned QS1Val = QS1.getImm();
    assert(QS1Val < 8 && "Unexpected value of ee_vmulas_s8_accx_ld_xp_qup "
                         "first argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(AS.getReg())
        .addReg(AD.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(Xtensa::Q0 + QS1Val);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMULAS_S8_QACC_P: {
    unsigned Opc = Xtensa::EE_VMULAS_S8_QACC;
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmulas_s8_qacc first argument, "
                        "it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmulas_s8_qacc first argument, "
                        "it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMULAS_S8_QACC_LDBC_INCP_P: {
    unsigned Opc = Xtensa::EE_VMULAS_S8_QACC_LDBC_INCP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vmulas_s8_qacc_ldbc_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &QX = MI.getOperand(2);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmulas_s8_qacc_ldbc_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(3);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmulas_s8_qacc_ldbc_incp first "
                        "argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMULAS_S8_QACC_LDBC_INCP_QUP_P: {
    unsigned Opc = Xtensa::EE_VMULAS_S8_QACC_LDBC_INCP_QUP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vmulas_s8_qacc_ldbc_incp_qup "
                        "first argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &QX = MI.getOperand(2);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmulas_s8_qacc_ldbc_incp_qup "
                        "first argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(3);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmulas_s8_qacc_ldbc_incp_qup "
                        "first argument, it must be in range [0,7]");
    MachineOperand &QS0 = MI.getOperand(4);
    unsigned QS0Val = QS0.getImm();
    assert(QS0Val < 8 && "Unexpected value of ee_vmulas_s8_qacc_ldbc_incp_qup "
                         "first argument, it must be in range [0,7]");
    MachineOperand &QS1 = MI.getOperand(5);
    unsigned QS1Val = QS1.getImm();
    assert(QS1Val < 8 && "Unexpected value of ee_vmulas_s8_qacc_ldbc_incp_qup "
                         "first argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(AS.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(Xtensa::Q0 + QS1Val);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMULAS_S8_QACC_LD_IP_P: {
    unsigned Opc = Xtensa::EE_VMULAS_S8_QACC_LD_IP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vmulas_s8_qacc_ld_ip first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &IMM16 = MI.getOperand(2);
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmulas_s8_qacc_ld_ip first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmulas_s8_qacc_ld_ip first "
                        "argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addImm(IMM16.getImm())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMULAS_S8_QACC_LD_IP_QUP_P: {
    unsigned Opc = Xtensa::EE_VMULAS_S8_QACC_LD_IP_QUP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vmulas_s8_qacc_ld_ip_qup first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &IMM16 = MI.getOperand(2);
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmulas_s8_qacc_ld_ip_qup first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmulas_s8_qacc_ld_ip_qup first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QS0 = MI.getOperand(5);
    unsigned QS0Val = QS0.getImm();
    assert(QS0Val < 8 && "Unexpected value of ee_vmulas_s8_qacc_ld_ip_qup "
                         "first argument, it must be in range [0,7]");
    MachineOperand &QS1 = MI.getOperand(6);
    unsigned QS1Val = QS1.getImm();
    assert(QS1Val < 8 && "Unexpected value of ee_vmulas_s8_qacc_ld_ip_qup "
                         "first argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(AS.getReg())
        .addImm(IMM16.getImm())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(Xtensa::Q0 + QS1Val);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMULAS_S8_QACC_LD_XP_P: {
    unsigned Opc = Xtensa::EE_VMULAS_S8_QACC_LD_XP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vmulas_s8_qacc_ld_xp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &AD = MI.getOperand(2);
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmulas_s8_qacc_ld_xp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmulas_s8_qacc_ld_xp first "
                        "argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addReg(AD.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMULAS_S8_QACC_LD_XP_QUP_P: {
    unsigned Opc = Xtensa::EE_VMULAS_S8_QACC_LD_XP_QUP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vmulas_s8_qacc_ld_xp_qup first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &AD = MI.getOperand(2);
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmulas_s8_qacc_ld_xp_qup first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmulas_s8_qacc_ld_xp_qup first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QS0 = MI.getOperand(5);
    unsigned QS0Val = QS0.getImm();
    assert(QS0Val < 8 && "Unexpected value of ee_vmulas_s8_qacc_ld_xp_qup "
                         "first argument, it must be in range [0,7]");
    MachineOperand &QS1 = MI.getOperand(6);
    unsigned QS1Val = QS1.getImm();
    assert(QS1Val < 8 && "Unexpected value of ee_vmulas_s8_qacc_ld_xp_qup "
                         "first argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(AS.getReg())
        .addReg(AD.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(Xtensa::Q0 + QS1Val);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMULAS_U16_ACCX_P: {
    unsigned Opc = Xtensa::EE_VMULAS_U16_ACCX;
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmulas_u16_accx first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmulas_u16_accx first "
                        "argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMULAS_U16_ACCX_LD_IP_P: {
    unsigned Opc = Xtensa::EE_VMULAS_U16_ACCX_LD_IP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vmulas_u16_accx_ld_ip first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &IMM16 = MI.getOperand(2);
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmulas_u16_accx_ld_ip first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmulas_u16_accx_ld_ip first "
                        "argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addImm(IMM16.getImm())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMULAS_U16_ACCX_LD_IP_QUP_P: {
    unsigned Opc = Xtensa::EE_VMULAS_U16_ACCX_LD_IP_QUP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vmulas_u16_accx_ld_ip_qup "
                        "first argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &IMM16 = MI.getOperand(2);
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmulas_u16_accx_ld_ip_qup "
                        "first argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmulas_u16_accx_ld_ip_qup "
                        "first argument, it must be in range [0,7]");
    MachineOperand &QS0 = MI.getOperand(5);
    unsigned QS0Val = QS0.getImm();
    assert(QS0Val < 8 && "Unexpected value of ee_vmulas_u16_accx_ld_ip_qup "
                         "first argument, it must be in range [0,7]");
    MachineOperand &QS1 = MI.getOperand(6);
    unsigned QS1Val = QS1.getImm();
    assert(QS1Val < 8 && "Unexpected value of ee_vmulas_u16_accx_ld_ip_qup "
                         "first argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(AS.getReg())
        .addImm(IMM16.getImm())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(Xtensa::Q0 + QS1Val);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMULAS_U16_ACCX_LD_XP_P: {
    unsigned Opc = Xtensa::EE_VMULAS_U16_ACCX_LD_XP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vmulas_u16_accx_ld_xp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &AD = MI.getOperand(2);
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmulas_u16_accx_ld_xp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmulas_u16_accx_ld_xp first "
                        "argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addReg(AD.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMULAS_U16_ACCX_LD_XP_QUP_P: {
    unsigned Opc = Xtensa::EE_VMULAS_U16_ACCX_LD_XP_QUP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vmulas_u16_accx_ld_xp_qup "
                        "first argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &AD = MI.getOperand(2);
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmulas_u16_accx_ld_xp_qup "
                        "first argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmulas_u16_accx_ld_xp_qup "
                        "first argument, it must be in range [0,7]");
    MachineOperand &QS0 = MI.getOperand(5);
    unsigned QS0Val = QS0.getImm();
    assert(QS0Val < 8 && "Unexpected value of ee_vmulas_u16_accx_ld_xp_qup "
                         "first argument, it must be in range [0,7]");
    MachineOperand &QS1 = MI.getOperand(6);
    unsigned QS1Val = QS1.getImm();
    assert(QS1Val < 8 && "Unexpected value of ee_vmulas_u16_accx_ld_xp_qup "
                         "first argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(AS.getReg())
        .addReg(AD.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(Xtensa::Q0 + QS1Val);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMULAS_U16_QACC_P: {
    unsigned Opc = Xtensa::EE_VMULAS_U16_QACC;
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmulas_u16_qacc first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmulas_u16_qacc first "
                        "argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMULAS_U16_QACC_LDBC_INCP_P: {
    unsigned Opc = Xtensa::EE_VMULAS_U16_QACC_LDBC_INCP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vmulas_u16_qacc_ldbc_incp "
                        "first argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &QX = MI.getOperand(2);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmulas_u16_qacc_ldbc_incp "
                        "first argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(3);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmulas_u16_qacc_ldbc_incp "
                        "first argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMULAS_U16_QACC_LDBC_INCP_QUP_P: {
    unsigned Opc = Xtensa::EE_VMULAS_U16_QACC_LDBC_INCP_QUP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vmulas_u16_qacc_ldbc_incp_qup "
                        "first argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &QX = MI.getOperand(2);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmulas_u16_qacc_ldbc_incp_qup "
                        "first argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(3);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmulas_u16_qacc_ldbc_incp_qup "
                        "first argument, it must be in range [0,7]");
    MachineOperand &QS0 = MI.getOperand(4);
    unsigned QS0Val = QS0.getImm();
    assert(QS0Val < 8 && "Unexpected value of ee_vmulas_u16_qacc_ldbc_incp_qup "
                         "first argument, it must be in range [0,7]");
    MachineOperand &QS1 = MI.getOperand(5);
    unsigned QS1Val = QS1.getImm();
    assert(QS1Val < 8 && "Unexpected value of ee_vmulas_u16_qacc_ldbc_incp_qup "
                         "first argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(AS.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(Xtensa::Q0 + QS1Val);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMULAS_U16_QACC_LD_IP_P: {
    unsigned Opc = Xtensa::EE_VMULAS_U16_QACC_LD_IP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vmulas_u16_qacc_ld_ip first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &IMM16 = MI.getOperand(2);
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmulas_u16_qacc_ld_ip first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmulas_u16_qacc_ld_ip first "
                        "argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addImm(IMM16.getImm())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMULAS_U16_QACC_LD_IP_QUP_P: {
    unsigned Opc = Xtensa::EE_VMULAS_U16_QACC_LD_IP_QUP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vmulas_u16_qacc_ld_ip_qup "
                        "first argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &IMM16 = MI.getOperand(2);
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmulas_u16_qacc_ld_ip_qup "
                        "first argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmulas_u16_qacc_ld_ip_qup "
                        "first argument, it must be in range [0,7]");
    MachineOperand &QS0 = MI.getOperand(5);
    unsigned QS0Val = QS0.getImm();
    assert(QS0Val < 8 && "Unexpected value of ee_vmulas_u16_qacc_ld_ip_qup "
                         "first argument, it must be in range [0,7]");
    MachineOperand &QS1 = MI.getOperand(6);
    unsigned QS1Val = QS1.getImm();
    assert(QS1Val < 8 && "Unexpected value of ee_vmulas_u16_qacc_ld_ip_qup "
                         "first argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(AS.getReg())
        .addImm(IMM16.getImm())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(Xtensa::Q0 + QS1Val);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMULAS_U16_QACC_LD_XP_P: {
    unsigned Opc = Xtensa::EE_VMULAS_U16_QACC_LD_XP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vmulas_u16_qacc_ld_xp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &AD = MI.getOperand(2);
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmulas_u16_qacc_ld_xp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmulas_u16_qacc_ld_xp first "
                        "argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addReg(AD.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMULAS_U16_QACC_LD_XP_QUP_P: {
    unsigned Opc = Xtensa::EE_VMULAS_U16_QACC_LD_XP_QUP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vmulas_u16_qacc_ld_xp_qup "
                        "first argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &AD = MI.getOperand(2);
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmulas_u16_qacc_ld_xp_qup "
                        "first argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmulas_u16_qacc_ld_xp_qup "
                        "first argument, it must be in range [0,7]");
    MachineOperand &QS0 = MI.getOperand(5);
    unsigned QS0Val = QS0.getImm();
    assert(QS0Val < 8 && "Unexpected value of ee_vmulas_u16_qacc_ld_xp_qup "
                         "first argument, it must be in range [0,7]");
    MachineOperand &QS1 = MI.getOperand(6);
    unsigned QS1Val = QS1.getImm();
    assert(QS1Val < 8 && "Unexpected value of ee_vmulas_u16_qacc_ld_xp_qup "
                         "first argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(AS.getReg())
        .addReg(AD.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(Xtensa::Q0 + QS1Val);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMULAS_U8_ACCX_P: {
    unsigned Opc = Xtensa::EE_VMULAS_U8_ACCX;
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmulas_u8_accx first argument, "
                        "it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmulas_u8_accx first argument, "
                        "it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMULAS_U8_ACCX_LD_IP_P: {
    unsigned Opc = Xtensa::EE_VMULAS_U8_ACCX_LD_IP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vmulas_u8_accx_ld_ip first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &IMM16 = MI.getOperand(2);
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmulas_u8_accx_ld_ip first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmulas_u8_accx_ld_ip first "
                        "argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addImm(IMM16.getImm())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMULAS_U8_ACCX_LD_IP_QUP_P: {
    unsigned Opc = Xtensa::EE_VMULAS_U8_ACCX_LD_IP_QUP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vmulas_u8_accx_ld_ip_qup first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &IMM16 = MI.getOperand(2);
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmulas_u8_accx_ld_ip_qup first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmulas_u8_accx_ld_ip_qup first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QS0 = MI.getOperand(5);
    unsigned QS0Val = QS0.getImm();
    assert(QS0Val < 8 && "Unexpected value of ee_vmulas_u8_accx_ld_ip_qup "
                         "first argument, it must be in range [0,7]");
    MachineOperand &QS1 = MI.getOperand(6);
    unsigned QS1Val = QS1.getImm();
    assert(QS1Val < 8 && "Unexpected value of ee_vmulas_u8_accx_ld_ip_qup "
                         "first argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(AS.getReg())
        .addImm(IMM16.getImm())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(Xtensa::Q0 + QS1Val);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMULAS_U8_ACCX_LD_XP_P: {
    unsigned Opc = Xtensa::EE_VMULAS_U8_ACCX_LD_XP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vmulas_u8_accx_ld_xp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &AD = MI.getOperand(2);
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmulas_u8_accx_ld_xp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmulas_u8_accx_ld_xp first "
                        "argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addReg(AD.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMULAS_U8_ACCX_LD_XP_QUP_P: {
    unsigned Opc = Xtensa::EE_VMULAS_U8_ACCX_LD_XP_QUP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vmulas_u8_accx_ld_xp_qup first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &AD = MI.getOperand(2);
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmulas_u8_accx_ld_xp_qup first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmulas_u8_accx_ld_xp_qup first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QS0 = MI.getOperand(5);
    unsigned QS0Val = QS0.getImm();
    assert(QS0Val < 8 && "Unexpected value of ee_vmulas_u8_accx_ld_xp_qup "
                         "first argument, it must be in range [0,7]");
    MachineOperand &QS1 = MI.getOperand(6);
    unsigned QS1Val = QS1.getImm();
    assert(QS1Val < 8 && "Unexpected value of ee_vmulas_u8_accx_ld_xp_qup "
                         "first argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(AS.getReg())
        .addReg(AD.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(Xtensa::Q0 + QS1Val);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMULAS_U8_QACC_P: {
    unsigned Opc = Xtensa::EE_VMULAS_U8_QACC;
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmulas_u8_qacc first argument, "
                        "it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmulas_u8_qacc first argument, "
                        "it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMULAS_U8_QACC_LDBC_INCP_P: {
    unsigned Opc = Xtensa::EE_VMULAS_U8_QACC_LDBC_INCP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vmulas_u8_qacc_ldbc_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &QX = MI.getOperand(2);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmulas_u8_qacc_ldbc_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(3);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmulas_u8_qacc_ldbc_incp first "
                        "argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMULAS_U8_QACC_LDBC_INCP_QUP_P: {
    unsigned Opc = Xtensa::EE_VMULAS_U8_QACC_LDBC_INCP_QUP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vmulas_u8_qacc_ldbc_incp_qup "
                        "first argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &QX = MI.getOperand(2);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmulas_u8_qacc_ldbc_incp_qup "
                        "first argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(3);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmulas_u8_qacc_ldbc_incp_qup "
                        "first argument, it must be in range [0,7]");
    MachineOperand &QS0 = MI.getOperand(4);
    unsigned QS0Val = QS0.getImm();
    assert(QS0Val < 8 && "Unexpected value of ee_vmulas_u8_qacc_ldbc_incp_qup "
                         "first argument, it must be in range [0,7]");
    MachineOperand &QS1 = MI.getOperand(5);
    unsigned QS1Val = QS1.getImm();
    assert(QS1Val < 8 && "Unexpected value of ee_vmulas_u8_qacc_ldbc_incp_qup "
                         "first argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(AS.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(Xtensa::Q0 + QS1Val);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMULAS_U8_QACC_LD_IP_P: {
    unsigned Opc = Xtensa::EE_VMULAS_U8_QACC_LD_IP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vmulas_u8_qacc_ld_ip first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &IMM16 = MI.getOperand(2);
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmulas_u8_qacc_ld_ip first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmulas_u8_qacc_ld_ip first "
                        "argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addImm(IMM16.getImm())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMULAS_U8_QACC_LD_IP_QUP_P: {
    unsigned Opc = Xtensa::EE_VMULAS_U8_QACC_LD_IP_QUP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vmulas_u8_qacc_ld_ip_qup first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &IMM16 = MI.getOperand(2);
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmulas_u8_qacc_ld_ip_qup first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmulas_u8_qacc_ld_ip_qup first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QS0 = MI.getOperand(5);
    unsigned QS0Val = QS0.getImm();
    assert(QS0Val < 8 && "Unexpected value of ee_vmulas_u8_qacc_ld_ip_qup "
                         "first argument, it must be in range [0,7]");
    MachineOperand &QS1 = MI.getOperand(6);
    unsigned QS1Val = QS1.getImm();
    assert(QS1Val < 8 && "Unexpected value of ee_vmulas_u8_qacc_ld_ip_qup "
                         "first argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(AS.getReg())
        .addImm(IMM16.getImm())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(Xtensa::Q0 + QS1Val);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMULAS_U8_QACC_LD_XP_P: {
    unsigned Opc = Xtensa::EE_VMULAS_U8_QACC_LD_XP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vmulas_u8_qacc_ld_xp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &AD = MI.getOperand(2);
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmulas_u8_qacc_ld_xp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmulas_u8_qacc_ld_xp first "
                        "argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addReg(AD.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMULAS_U8_QACC_LD_XP_QUP_P: {
    unsigned Opc = Xtensa::EE_VMULAS_U8_QACC_LD_XP_QUP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vmulas_u8_qacc_ld_xp_qup first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &AD = MI.getOperand(2);
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmulas_u8_qacc_ld_xp_qup first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmulas_u8_qacc_ld_xp_qup first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QS0 = MI.getOperand(5);
    unsigned QS0Val = QS0.getImm();
    assert(QS0Val < 8 && "Unexpected value of ee_vmulas_u8_qacc_ld_xp_qup "
                         "first argument, it must be in range [0,7]");
    MachineOperand &QS1 = MI.getOperand(6);
    unsigned QS1Val = QS1.getImm();
    assert(QS1Val < 8 && "Unexpected value of ee_vmulas_u8_qacc_ld_xp_qup "
                         "first argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(AS.getReg())
        .addReg(AD.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(Xtensa::Q0 + QS1Val);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMUL_S16_P: {
    unsigned Opc = Xtensa::EE_VMUL_S16;
    MachineOperand &QZ = MI.getOperand(0);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of ee_vmul_s16 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmul_s16 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmul_s16 first argument, it "
                        "must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QZVal)
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMUL_S16_LD_INCP_P: {
    unsigned Opc = Xtensa::EE_VMUL_S16_LD_INCP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vmul_s16_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &QZ = MI.getOperand(2);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of ee_vmul_s16_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmul_s16_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmul_s16_ld_incp first "
                        "argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QZVal)
        .addReg(AS.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMUL_S16_ST_INCP_P: {
    unsigned Opc = Xtensa::EE_VMUL_S16_ST_INCP;
    MachineOperand &QV = MI.getOperand(0);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of ee_vmul_s16_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &QZ = MI.getOperand(2);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of ee_vmul_s16_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmul_s16_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmul_s16_st_incp first "
                        "argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QZVal)
        .addReg(Xtensa::Q0 + QVVal)
        .addReg(AS.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMUL_S8_P: {
    unsigned Opc = Xtensa::EE_VMUL_S8;
    MachineOperand &QZ = MI.getOperand(0);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of ee_vmul_s8 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmul_s8 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmul_s8 first argument, it "
                        "must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QZVal)
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMUL_S8_LD_INCP_P: {
    unsigned Opc = Xtensa::EE_VMUL_S8_LD_INCP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vmul_s8_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &QZ = MI.getOperand(2);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of ee_vmul_s8_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmul_s8_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmul_s8_ld_incp first "
                        "argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QZVal)
        .addReg(AS.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMUL_S8_ST_INCP_P: {
    unsigned Opc = Xtensa::EE_VMUL_S8_ST_INCP;
    MachineOperand &QV = MI.getOperand(0);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of ee_vmul_s8_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &QZ = MI.getOperand(2);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of ee_vmul_s8_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmul_s8_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmul_s8_st_incp first "
                        "argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QZVal)
        .addReg(Xtensa::Q0 + QVVal)
        .addReg(AS.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMUL_U16_P: {
    unsigned Opc = Xtensa::EE_VMUL_U16;
    MachineOperand &QZ = MI.getOperand(0);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of ee_vmul_u16 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmul_u16 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmul_u16 first argument, it "
                        "must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QZVal)
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMUL_U16_LD_INCP_P: {
    unsigned Opc = Xtensa::EE_VMUL_U16_LD_INCP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vmul_u16_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &QZ = MI.getOperand(2);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of ee_vmul_u16_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmul_u16_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmul_u16_ld_incp first "
                        "argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QZVal)
        .addReg(AS.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMUL_U16_ST_INCP_P: {
    unsigned Opc = Xtensa::EE_VMUL_U16_ST_INCP;
    MachineOperand &QV = MI.getOperand(0);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of ee_vmul_u16_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &QZ = MI.getOperand(2);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of ee_vmul_u16_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmul_u16_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmul_u16_st_incp first "
                        "argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QZVal)
        .addReg(Xtensa::Q0 + QVVal)
        .addReg(AS.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMUL_U8_P: {
    unsigned Opc = Xtensa::EE_VMUL_U8;
    MachineOperand &QZ = MI.getOperand(0);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of ee_vmul_u8 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmul_u8 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmul_u8 first argument, it "
                        "must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QZVal)
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMUL_U8_LD_INCP_P: {
    unsigned Opc = Xtensa::EE_VMUL_U8_LD_INCP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vmul_u8_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &QZ = MI.getOperand(2);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of ee_vmul_u8_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmul_u8_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmul_u8_ld_incp first "
                        "argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QZVal)
        .addReg(AS.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VMUL_U8_ST_INCP_P: {
    unsigned Opc = Xtensa::EE_VMUL_U8_ST_INCP;
    MachineOperand &QV = MI.getOperand(0);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of ee_vmul_u8_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &QZ = MI.getOperand(2);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of ee_vmul_u8_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vmul_u8_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vmul_u8_st_incp first "
                        "argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QZVal)
        .addReg(Xtensa::Q0 + QVVal)
        .addReg(AS.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VPRELU_S16_P: {
    unsigned Opc = Xtensa::EE_VPRELU_S16;
    MachineOperand &QZ = MI.getOperand(0);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of ee_vprelu_s16 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vprelu_s16 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vprelu_s16 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &AY = MI.getOperand(3);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QZVal)
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal)
        .addReg(AY.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VPRELU_S8_P: {
    unsigned Opc = Xtensa::EE_VPRELU_S8;
    MachineOperand &QZ = MI.getOperand(0);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of ee_vprelu_s8 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vprelu_s8 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vprelu_s8 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &AY = MI.getOperand(3);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QZVal)
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal)
        .addReg(AY.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VRELU_S16_P: {
    unsigned Opc = Xtensa::EE_VRELU_S16;
    MachineOperand &QS = MI.getOperand(0);
    unsigned QSVal = QS.getImm();
    assert(QSVal < 8 && "Unexpected value of ee_vrelu_s16 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &AX = MI.getOperand(1);
    MachineOperand &AY = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QSVal)
        .addReg(Xtensa::Q0 + QSVal)
        .addReg(AX.getReg())
        .addReg(AY.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VRELU_S8_P: {
    unsigned Opc = Xtensa::EE_VRELU_S8;
    MachineOperand &QS = MI.getOperand(0);
    unsigned QSVal = QS.getImm();
    assert(QSVal < 8 && "Unexpected value of ee_vrelu_s8 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &AX = MI.getOperand(1);
    MachineOperand &AY = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QSVal)
        .addReg(Xtensa::Q0 + QSVal)
        .addReg(AX.getReg())
        .addReg(AY.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VSL_32_P: {
    unsigned Opc = Xtensa::EE_VSL_32;
    MachineOperand &QA = MI.getOperand(0);
    unsigned QAVal = QA.getImm();
    assert(QAVal < 8 && "Unexpected value of ee_vsl_32 first argument, it must "
                        "be in range [0,7]");
    MachineOperand &QS = MI.getOperand(1);
    unsigned QSVal = QS.getImm();
    assert(QSVal < 8 && "Unexpected value of ee_vsl_32 first argument, it must "
                        "be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QAVal)
        .addReg(Xtensa::Q0 + QSVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VSMULAS_S16_QACC_P: {
    unsigned Opc = Xtensa::EE_VSMULAS_S16_QACC;
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vsmulas_s16_qacc first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vsmulas_s16_qacc first "
                        "argument, it must be in range [0,7]");
    MachineOperand &SEL8 = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal)
        .addImm(SEL8.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VSMULAS_S16_QACC_LD_INCP_P: {
    unsigned Opc = Xtensa::EE_VSMULAS_S16_QACC_LD_INCP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vsmulas_s16_qacc_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &QX = MI.getOperand(2);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vsmulas_s16_qacc_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(3);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vsmulas_s16_qacc_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &SEL8 = MI.getOperand(4);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal)
        .addImm(SEL8.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VSMULAS_S8_QACC_P: {
    unsigned Opc = Xtensa::EE_VSMULAS_S8_QACC;
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vsmulas_s8_qacc first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vsmulas_s8_qacc first "
                        "argument, it must be in range [0,7]");
    MachineOperand &SEL16 = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal)
        .addImm(SEL16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VSMULAS_S8_QACC_LD_INCP_P: {
    unsigned Opc = Xtensa::EE_VSMULAS_S8_QACC_LD_INCP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vsmulas_s8_qacc_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &QX = MI.getOperand(2);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vsmulas_s8_qacc_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(3);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vsmulas_s8_qacc_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &SEL16 = MI.getOperand(4);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(AS.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal)
        .addImm(SEL16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VSR_32_P: {
    unsigned Opc = Xtensa::EE_VSR_32;
    MachineOperand &QA = MI.getOperand(0);
    unsigned QAVal = QA.getImm();
    assert(QAVal < 8 && "Unexpected value of ee_vsr_32 first argument, it must "
                        "be in range [0,7]");
    MachineOperand &QS = MI.getOperand(1);
    unsigned QSVal = QS.getImm();
    assert(QSVal < 8 && "Unexpected value of ee_vsr_32 first argument, it must "
                        "be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QAVal)
        .addReg(Xtensa::Q0 + QSVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VST_128_IP_P: {
    unsigned Opc = Xtensa::EE_VST_128_IP;
    MachineOperand &QV = MI.getOperand(0);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of ee_vst_128_ip first argument, it "
                        "must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &IMM16 = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QVVal)
        .addReg(AS.getReg())
        .addImm(IMM16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VST_128_XP_P: {
    unsigned Opc = Xtensa::EE_VST_128_XP;
    MachineOperand &QV = MI.getOperand(0);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of ee_vst_128_xp first argument, it "
                        "must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &AD = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QVVal)
        .addReg(AS.getReg())
        .addReg(AD.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VST_H_64_IP_P: {
    unsigned Opc = Xtensa::EE_VST_H_64_IP;
    MachineOperand &QV = MI.getOperand(0);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of ee_vst_h_64_ip first argument, it "
                        "must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &IMM8 = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QVVal)
        .addReg(AS.getReg())
        .addImm(IMM8.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VST_H_64_XP_P: {
    unsigned Opc = Xtensa::EE_VST_H_64_XP;
    MachineOperand &QV = MI.getOperand(0);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of ee_vst_h_64_xp first argument, it "
                        "must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &AD = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QVVal)
        .addReg(AS.getReg())
        .addReg(AD.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VST_L_64_IP_P: {
    unsigned Opc = Xtensa::EE_VST_L_64_IP;
    MachineOperand &QV = MI.getOperand(0);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of ee_vst_l_64_ip first argument, it "
                        "must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &IMM8 = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QVVal)
        .addReg(AS.getReg())
        .addImm(IMM8.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VST_L_64_XP_P: {
    unsigned Opc = Xtensa::EE_VST_L_64_XP;
    MachineOperand &QV = MI.getOperand(0);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of ee_vst_l_64_xp first argument, it "
                        "must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &AD = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QVVal)
        .addReg(AS.getReg())
        .addReg(AD.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VSUBS_S16_P: {
    unsigned Opc = Xtensa::EE_VSUBS_S16;
    MachineOperand &QA = MI.getOperand(0);
    unsigned QAVal = QA.getImm();
    assert(QAVal < 8 && "Unexpected value of ee_vsubs_s16 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vsubs_s16 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vsubs_s16 first argument, it "
                        "must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QAVal)
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VSUBS_S16_LD_INCP_P: {
    unsigned Opc = Xtensa::EE_VSUBS_S16_LD_INCP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vsubs_s16_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &QA = MI.getOperand(2);
    unsigned QAVal = QA.getImm();
    assert(QAVal < 8 && "Unexpected value of ee_vsubs_s16_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vsubs_s16_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vsubs_s16_ld_incp first "
                        "argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QAVal)
        .addReg(AS.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VSUBS_S16_ST_INCP_P: {
    unsigned Opc = Xtensa::EE_VSUBS_S16_ST_INCP;
    MachineOperand &QV = MI.getOperand(0);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of ee_vsubs_s16_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &QA = MI.getOperand(2);
    unsigned QAVal = QA.getImm();
    assert(QAVal < 8 && "Unexpected value of ee_vsubs_s16_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vsubs_s16_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vsubs_s16_st_incp first "
                        "argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QAVal)
        .addReg(Xtensa::Q0 + QVVal)
        .addReg(AS.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VSUBS_S32_P: {
    unsigned Opc = Xtensa::EE_VSUBS_S32;
    MachineOperand &QA = MI.getOperand(0);
    unsigned QAVal = QA.getImm();
    assert(QAVal < 8 && "Unexpected value of ee_vsubs_s32 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vsubs_s32 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vsubs_s32 first argument, it "
                        "must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QAVal)
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VSUBS_S32_LD_INCP_P: {
    unsigned Opc = Xtensa::EE_VSUBS_S32_LD_INCP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vsubs_s32_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &QA = MI.getOperand(2);
    unsigned QAVal = QA.getImm();
    assert(QAVal < 8 && "Unexpected value of ee_vsubs_s32_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vsubs_s32_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vsubs_s32_ld_incp first "
                        "argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QAVal)
        .addReg(AS.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VSUBS_S32_ST_INCP_P: {
    unsigned Opc = Xtensa::EE_VSUBS_S32_ST_INCP;
    MachineOperand &QV = MI.getOperand(0);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of ee_vsubs_s32_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &QA = MI.getOperand(2);
    unsigned QAVal = QA.getImm();
    assert(QAVal < 8 && "Unexpected value of ee_vsubs_s32_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vsubs_s32_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vsubs_s32_st_incp first "
                        "argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QAVal)
        .addReg(Xtensa::Q0 + QVVal)
        .addReg(AS.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VSUBS_S8_P: {
    unsigned Opc = Xtensa::EE_VSUBS_S8;
    MachineOperand &QA = MI.getOperand(0);
    unsigned QAVal = QA.getImm();
    assert(QAVal < 8 && "Unexpected value of ee_vsubs_s8 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vsubs_s8 first argument, it "
                        "must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vsubs_s8 first argument, it "
                        "must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QAVal)
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VSUBS_S8_LD_INCP_P: {
    unsigned Opc = Xtensa::EE_VSUBS_S8_LD_INCP;
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of ee_vsubs_s8_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &QA = MI.getOperand(2);
    unsigned QAVal = QA.getImm();
    assert(QAVal < 8 && "Unexpected value of ee_vsubs_s8_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vsubs_s8_ld_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vsubs_s8_ld_incp first "
                        "argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QUVal)
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QAVal)
        .addReg(AS.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VSUBS_S8_ST_INCP_P: {
    unsigned Opc = Xtensa::EE_VSUBS_S8_ST_INCP;
    MachineOperand &QV = MI.getOperand(0);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of ee_vsubs_s8_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &AS = MI.getOperand(1);
    const TargetRegisterClass *RC = getRegClassFor(MVT::i32);
    unsigned R1 = MRI.createVirtualRegister(RC);
    MachineOperand &QA = MI.getOperand(2);
    unsigned QAVal = QA.getImm();
    assert(QAVal < 8 && "Unexpected value of ee_vsubs_s8_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QX = MI.getOperand(3);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_vsubs_s8_st_incp first "
                        "argument, it must be in range [0,7]");
    MachineOperand &QY = MI.getOperand(4);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_vsubs_s8_st_incp first "
                        "argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Undef)
        .addReg(Xtensa::Q0 + QAVal)
        .addReg(Xtensa::Q0 + QVVal)
        .addReg(AS.getReg())
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VUNZIP_16_P: {
    unsigned Opc = Xtensa::EE_VUNZIP_16;
    MachineOperand &QS0 = MI.getOperand(0);
    unsigned QS0Val = QS0.getImm();
    assert(QS0Val < 8 && "Unexpected value of ee_vunzip_16 first argument, it "
                         "must be in range [0,7]");
    MachineOperand &QS1 = MI.getOperand(1);
    unsigned QS1Val = QS1.getImm();
    assert(QS1Val < 8 && "Unexpected value of ee_vunzip_16 first argument, it "
                         "must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(Xtensa::Q0 + QS1Val)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(Xtensa::Q0 + QS1Val);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VUNZIP_32_P: {
    unsigned Opc = Xtensa::EE_VUNZIP_32;
    MachineOperand &QS0 = MI.getOperand(0);
    unsigned QS0Val = QS0.getImm();
    assert(QS0Val < 8 && "Unexpected value of ee_vunzip_32 first argument, it "
                         "must be in range [0,7]");
    MachineOperand &QS1 = MI.getOperand(1);
    unsigned QS1Val = QS1.getImm();
    assert(QS1Val < 8 && "Unexpected value of ee_vunzip_32 first argument, it "
                         "must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(Xtensa::Q0 + QS1Val)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(Xtensa::Q0 + QS1Val);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VUNZIP_8_P: {
    unsigned Opc = Xtensa::EE_VUNZIP_8;
    MachineOperand &QS0 = MI.getOperand(0);
    unsigned QS0Val = QS0.getImm();
    assert(QS0Val < 8 && "Unexpected value of ee_vunzip_8 first argument, it "
                         "must be in range [0,7]");
    MachineOperand &QS1 = MI.getOperand(1);
    unsigned QS1Val = QS1.getImm();
    assert(QS1Val < 8 && "Unexpected value of ee_vunzip_8 first argument, it "
                         "must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(Xtensa::Q0 + QS1Val)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(Xtensa::Q0 + QS1Val);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VZIP_16_P: {
    unsigned Opc = Xtensa::EE_VZIP_16;
    MachineOperand &QS0 = MI.getOperand(0);
    unsigned QS0Val = QS0.getImm();
    assert(QS0Val < 8 && "Unexpected value of ee_vzip_16 first argument, it "
                         "must be in range [0,7]");
    MachineOperand &QS1 = MI.getOperand(1);
    unsigned QS1Val = QS1.getImm();
    assert(QS1Val < 8 && "Unexpected value of ee_vzip_16 first argument, it "
                         "must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(Xtensa::Q0 + QS1Val)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(Xtensa::Q0 + QS1Val);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VZIP_32_P: {
    unsigned Opc = Xtensa::EE_VZIP_32;
    MachineOperand &QS0 = MI.getOperand(0);
    unsigned QS0Val = QS0.getImm();
    assert(QS0Val < 8 && "Unexpected value of ee_vzip_32 first argument, it "
                         "must be in range [0,7]");
    MachineOperand &QS1 = MI.getOperand(1);
    unsigned QS1Val = QS1.getImm();
    assert(QS1Val < 8 && "Unexpected value of ee_vzip_32 first argument, it "
                         "must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(Xtensa::Q0 + QS1Val)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(Xtensa::Q0 + QS1Val);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_VZIP_8_P: {
    unsigned Opc = Xtensa::EE_VZIP_8;
    MachineOperand &QS0 = MI.getOperand(0);
    unsigned QS0Val = QS0.getImm();
    assert(QS0Val < 8 && "Unexpected value of ee_vzip_8 first argument, it "
                         "must be in range [0,7]");
    MachineOperand &QS1 = MI.getOperand(1);
    unsigned QS1Val = QS1.getImm();
    assert(QS1Val < 8 && "Unexpected value of ee_vzip_8 first argument, it "
                         "must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(Xtensa::Q0 + QS1Val)
        .addReg(Xtensa::Q0 + QS0Val)
        .addReg(Xtensa::Q0 + QS1Val);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_XORQ_P: {
    unsigned Opc = Xtensa::EE_XORQ;
    MachineOperand &QA = MI.getOperand(0);
    unsigned QAVal = QA.getImm();
    assert(QAVal < 8 && "Unexpected value of ee_xorq first argument, it must "
                        "be in range [0,7]");
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of ee_xorq first argument, it must "
                        "be in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of ee_xorq first argument, it must "
                        "be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + QAVal)
        .addReg(Xtensa::Q0 + QXVal)
        .addReg(Xtensa::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_ZERO_ACCX_P: {
    unsigned Opc = Xtensa::EE_ZERO_ACCX;
    BuildMI(*MBB, MI, DL, TII.get(Opc));

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_ZERO_Q_P: {
    unsigned Opc = Xtensa::EE_ZERO_Q;
    MachineOperand &QA = MI.getOperand(0);
    unsigned QAVal = QA.getImm();
    assert(QAVal < 8 && "Unexpected value of ee_zero_q first argument, it must "
                        "be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc)).addReg(Xtensa::Q0 + QAVal);

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::EE_ZERO_QACC_P: {
    unsigned Opc = Xtensa::EE_ZERO_QACC;
    BuildMI(*MBB, MI, DL, TII.get(Opc));

    MI.eraseFromParent();
    return MBB;
  }
  case Xtensa::mv_QR_P: {
    unsigned Opc = Xtensa::mv_QR;
    MachineOperand &A = MI.getOperand(0);
    unsigned AVal = A.getImm();
    assert(
        AVal < 8 &&
        "Unexpected value of mv_qr first argument, it must be in range [0,7]");
    MachineOperand &B = MI.getOperand(1);
    unsigned BVal = B.getImm();
    assert(
        BVal < 8 &&
        "Unexpected value of mv_qr first argument, it must be in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(Xtensa::Q0 + AVal)
        .addReg(Xtensa::Q0 + BVal);

    MI.eraseFromParent();
    return MBB;
  }
  }
}
