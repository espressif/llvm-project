//==- RISCVESPISelLowering.cpp - ESP32 P4 DAG Lowering Implementation -===//
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

#include "RISCVISelLowering.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

using namespace llvm;

MachineBasicBlock *RISCVTargetLowering::emitDSPInstrWithCustomInserter(
    MachineInstr &MI, MachineBasicBlock *MBB, const TargetInstrInfo &TII,
    MachineFunction *MF, MachineRegisterInfo &MRI, DebugLoc DL) const {
  switch (MI.getOpcode()) {
  default:
    llvm_unreachable("Unexpected instr type to insert");
  case RISCV::ESP_CMUL_U16_P: {
    unsigned Opc = RISCV::ESP_CMUL_U16;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_cmul_u16 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_cmul_u16 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &SELECT_4 = MI.getOperand(2);
    MachineOperand &QZ = MI.getOperand(3);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_cmul_u16 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addImm(SELECT_4.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_CMUL_U16_LD_INCP_P: {
    unsigned Opc = RISCV::ESP_CMUL_U16_LD_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_cmul_u16_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_cmul_u16_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &SELECT_4 = MI.getOperand(4);
    MachineOperand &QZ = MI.getOperand(5);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_cmul_u16_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(6);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_cmul_u16_ld_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg())
        .addImm(SELECT_4.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_CMUL_U16_ST_INCP_P: {
    unsigned Opc = RISCV::ESP_CMUL_U16_ST_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_cmul_u16_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_cmul_u16_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_cmul_u16_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &SELECT_4 = MI.getOperand(5);
    MachineOperand &QZ = MI.getOperand(6);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_cmul_u16_st_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg())
        .addImm(SELECT_4.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_CMUL_U8_P: {
    unsigned Opc = RISCV::ESP_CMUL_U8;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_cmul_u8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_cmul_u8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &SELECT_4 = MI.getOperand(2);
    MachineOperand &QZ = MI.getOperand(3);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_cmul_u8 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addImm(SELECT_4.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_CMUL_U8_LD_INCP_P: {
    unsigned Opc = RISCV::ESP_CMUL_U8_LD_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_cmul_u8_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_cmul_u8_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &SELECT_4 = MI.getOperand(4);
    MachineOperand &QZ = MI.getOperand(5);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_cmul_u8_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(6);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_cmul_u8_ld_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg())
        .addImm(SELECT_4.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_CMUL_U8_ST_INCP_P: {
    unsigned Opc = RISCV::ESP_CMUL_U8_ST_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_cmul_u8_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_cmul_u8_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_cmul_u8_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &SELECT_4 = MI.getOperand(5);
    MachineOperand &QZ = MI.getOperand(6);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_cmul_u8_st_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg())
        .addImm(SELECT_4.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VCMP_EQ_U16_P: {
    unsigned Opc = RISCV::ESP_VCMP_EQ_U16;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vcmp_eq_u16 first argument, "
                        "it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vcmp_eq_u16 first argument, "
                        "it must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(2);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vcmp_eq_u16 first argument, "
                        "it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VCMP_EQ_U32_P: {
    unsigned Opc = RISCV::ESP_VCMP_EQ_U32;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vcmp_eq_u32 first argument, "
                        "it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vcmp_eq_u32 first argument, "
                        "it must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(2);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vcmp_eq_u32 first argument, "
                        "it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VCMP_EQ_U8_P: {
    unsigned Opc = RISCV::ESP_VCMP_EQ_U8;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vcmp_eq_u8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vcmp_eq_u8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(2);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vcmp_eq_u8 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VCMP_GT_U16_P: {
    unsigned Opc = RISCV::ESP_VCMP_GT_U16;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vcmp_gt_u16 first argument, "
                        "it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vcmp_gt_u16 first argument, "
                        "it must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(2);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vcmp_gt_u16 first argument, "
                        "it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VCMP_GT_U32_P: {
    unsigned Opc = RISCV::ESP_VCMP_GT_U32;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vcmp_gt_u32 first argument, "
                        "it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vcmp_gt_u32 first argument, "
                        "it must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(2);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vcmp_gt_u32 first argument, "
                        "it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VCMP_GT_U8_P: {
    unsigned Opc = RISCV::ESP_VCMP_GT_U8;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vcmp_gt_u8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vcmp_gt_u8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(2);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vcmp_gt_u8 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VLDBC_16_IP_P: {
    unsigned Opc = RISCV::ESP_VLDBC_16_IP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(1);
    MachineOperand &OFFSET_256_4 = MI.getOperand(2);
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vldbc_16_ip first argument, "
                        "it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS1.getReg())
        .addImm(OFFSET_256_4.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VLDBC_16_XP_P: {
    unsigned Opc = RISCV::ESP_VLDBC_16_XP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &RS1 = MI.getOperand(2);
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vldbc_16_xp first argument, "
                        "it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS2.getReg())
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VLDBC_32_IP_P: {
    unsigned Opc = RISCV::ESP_VLDBC_32_IP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(1);
    MachineOperand &OFFSET_256_4 = MI.getOperand(2);
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vldbc_32_ip first argument, "
                        "it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS1.getReg())
        .addImm(OFFSET_256_4.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VLDBC_32_XP_P: {
    unsigned Opc = RISCV::ESP_VLDBC_32_XP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &RS1 = MI.getOperand(2);
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vldbc_32_xp first argument, "
                        "it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS2.getReg())
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VLDBC_8_IP_P: {
    unsigned Opc = RISCV::ESP_VLDBC_8_IP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(1);
    MachineOperand &OFFSET_256_4 = MI.getOperand(2);
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vldbc_8_ip first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS1.getReg())
        .addImm(OFFSET_256_4.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VLDBC_8_XP_P: {
    unsigned Opc = RISCV::ESP_VLDBC_8_XP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &RS1 = MI.getOperand(2);
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vldbc_8_xp first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS2.getReg())
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VLDEXT_S16_IP_P: {
    unsigned Opc = RISCV::ESP_VLDEXT_S16_IP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(1);
    MachineOperand &OFFSET_16_16 = MI.getOperand(2);
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vldext_s16_ip first argument, "
                        "it must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(4);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vldext_s16_ip first argument, "
                        "it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS1.getReg())
        .addImm(OFFSET_16_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VLDEXT_S16_XP_P: {
    unsigned Opc = RISCV::ESP_VLDEXT_S16_XP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &RS1 = MI.getOperand(2);
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vldext_s16_xp first argument, "
                        "it must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(4);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vldext_s16_xp first argument, "
                        "it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS2.getReg())
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VLDEXT_S8_IP_P: {
    unsigned Opc = RISCV::ESP_VLDEXT_S8_IP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(1);
    MachineOperand &OFFSET_16_16 = MI.getOperand(2);
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vldext_s8_ip first argument, "
                        "it must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(4);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vldext_s8_ip first argument, "
                        "it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS1.getReg())
        .addImm(OFFSET_16_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VLDEXT_S8_XP_P: {
    unsigned Opc = RISCV::ESP_VLDEXT_S8_XP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &RS1 = MI.getOperand(2);
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vldext_s8_xp first argument, "
                        "it must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(4);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vldext_s8_xp first argument, "
                        "it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS2.getReg())
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VLDEXT_U16_IP_P: {
    unsigned Opc = RISCV::ESP_VLDEXT_U16_IP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(1);
    MachineOperand &OFFSET_16_16 = MI.getOperand(2);
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vldext_u16_ip first argument, "
                        "it must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(4);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vldext_u16_ip first argument, "
                        "it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS1.getReg())
        .addImm(OFFSET_16_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VLDEXT_U16_XP_P: {
    unsigned Opc = RISCV::ESP_VLDEXT_U16_XP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &RS1 = MI.getOperand(2);
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vldext_u16_xp first argument, "
                        "it must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(4);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vldext_u16_xp first argument, "
                        "it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS2.getReg())
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VLDEXT_U8_IP_P: {
    unsigned Opc = RISCV::ESP_VLDEXT_U8_IP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(1);
    MachineOperand &OFFSET_16_16 = MI.getOperand(2);
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vldext_u8_ip first argument, "
                        "it must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(4);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vldext_u8_ip first argument, "
                        "it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS1.getReg())
        .addImm(OFFSET_16_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VLDEXT_U8_XP_P: {
    unsigned Opc = RISCV::ESP_VLDEXT_U8_XP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &RS1 = MI.getOperand(2);
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vldext_u8_xp first argument, "
                        "it must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(4);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vldext_u8_xp first argument, "
                        "it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS2.getReg())
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VLDHBC_16_INCP_P: {
    unsigned Opc = RISCV::ESP_VLDHBC_16_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(1);
    MachineOperand &QU = MI.getOperand(2);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vldhbc_16_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(3);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vldhbc_16_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_MULS16IX2_P: {
    unsigned Opc = RISCV::ESP_MULS16IX2;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(0);
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &SHAMT = MI.getOperand(2);
    const TargetRegisterClass *RC = &RISCV::GPRPIERegClass;
    unsigned R1 = MRI.createVirtualRegister(RC);
    unsigned R2 = MRI.createVirtualRegister(RC);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Define)
        .addReg(R2, RegState::Define)
        .addReg(RS1.getReg())
        .addReg(RS2.getReg())
        .addImm(SHAMT.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_MULS16X2_P: {
    unsigned Opc = RISCV::ESP_MULS16X2;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(0);
    MachineOperand &RS2 = MI.getOperand(1);
    const TargetRegisterClass *RC = &RISCV::GPRPIERegClass;
    unsigned R1 = MRI.createVirtualRegister(RC);
    unsigned R2 = MRI.createVirtualRegister(RC);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Define)
        .addReg(R2, RegState::Define)
        .addReg(RS1.getReg())
        .addReg(RS2.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_CMUL_U16_2P2_P: {
    unsigned Opc = RISCV::ESP_CMUL_U16_2P2;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_cmul_u16_2p2 first argument, "
                        "it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_cmul_u16_2p2 first argument, "
                        "it must bi in range [0,7]");
    MachineOperand &SAT = MI.getOperand(2);
    MachineOperand &SELECT_4 = MI.getOperand(3);
    MachineOperand &RM = MI.getOperand(4);
    MachineOperand &QZ = MI.getOperand(5);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_cmul_u16_2p2 first argument, "
                        "it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addImm(SAT.getImm())
        .addImm(SELECT_4.getImm())
        .addImm(RM.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_CMUL_U16_LD_INCP_2P2_P: {
    unsigned Opc = RISCV::ESP_CMUL_U16_LD_INCP_2P2;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_cmul_u16_ld_incp_2p2 first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_cmul_u16_ld_incp_2p2 first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &SAT = MI.getOperand(4);
    MachineOperand &SELECT_4 = MI.getOperand(5);
    MachineOperand &RM = MI.getOperand(6);
    MachineOperand &QZ = MI.getOperand(7);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_cmul_u16_ld_incp_2p2 first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(8);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_cmul_u16_ld_incp_2p2 first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg())
        .addImm(SAT.getImm())
        .addImm(SELECT_4.getImm())
        .addImm(RM.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_CMUL_U16_ST_INCP_2P2_P: {
    unsigned Opc = RISCV::ESP_CMUL_U16_ST_INCP_2P2;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_cmul_u16_st_incp_2p2 first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_cmul_u16_st_incp_2p2 first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_cmul_u16_st_incp_2p2 first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &SAT = MI.getOperand(5);
    MachineOperand &SELECT_4 = MI.getOperand(6);
    MachineOperand &RM = MI.getOperand(7);
    MachineOperand &QZ = MI.getOperand(8);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_cmul_u16_st_incp_2p2 first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg())
        .addImm(SAT.getImm())
        .addImm(SELECT_4.getImm())
        .addImm(RM.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_CMUL_U8_2P2_P: {
    unsigned Opc = RISCV::ESP_CMUL_U8_2P2;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_cmul_u8_2p2 first argument, "
                        "it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_cmul_u8_2p2 first argument, "
                        "it must bi in range [0,7]");
    MachineOperand &SAT = MI.getOperand(2);
    MachineOperand &SELECT_4 = MI.getOperand(3);
    MachineOperand &RM = MI.getOperand(4);
    MachineOperand &QZ = MI.getOperand(5);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_cmul_u8_2p2 first argument, "
                        "it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addImm(SAT.getImm())
        .addImm(SELECT_4.getImm())
        .addImm(RM.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_CMUL_U8_LD_INCP_2P2_P: {
    unsigned Opc = RISCV::ESP_CMUL_U8_LD_INCP_2P2;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_cmul_u8_ld_incp_2p2 first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_cmul_u8_ld_incp_2p2 first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &SAT = MI.getOperand(4);
    MachineOperand &SELECT_4 = MI.getOperand(5);
    MachineOperand &RM = MI.getOperand(6);
    MachineOperand &QZ = MI.getOperand(7);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_cmul_u8_ld_incp_2p2 first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(8);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_cmul_u8_ld_incp_2p2 first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg())
        .addImm(SAT.getImm())
        .addImm(SELECT_4.getImm())
        .addImm(RM.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_CMUL_U8_ST_INCP_2P2_P: {
    unsigned Opc = RISCV::ESP_CMUL_U8_ST_INCP_2P2;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_cmul_u8_st_incp_2p2 first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_cmul_u8_st_incp_2p2 first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_cmul_u8_st_incp_2p2 first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &SAT = MI.getOperand(5);
    MachineOperand &SELECT_4 = MI.getOperand(6);
    MachineOperand &RM = MI.getOperand(7);
    MachineOperand &QZ = MI.getOperand(8);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_cmul_u8_st_incp_2p2 first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg())
        .addImm(SAT.getImm())
        .addImm(SELECT_4.getImm())
        .addImm(RM.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_ADDX2_2P2_P: {
    unsigned Opc = RISCV::ESP_ADDX2_2P2;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(0);
    MachineOperand &RS2 = MI.getOperand(1);
    const TargetRegisterClass *RC = &RISCV::GPRPIERegClass;
    unsigned R1 = MRI.createVirtualRegister(RC);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Define)
        .addReg(RS1.getReg())
        .addReg(RS2.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_ADDX4_2P2_P: {
    unsigned Opc = RISCV::ESP_ADDX4_2P2;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(0);
    MachineOperand &RS2 = MI.getOperand(1);
    const TargetRegisterClass *RC = &RISCV::GPRPIERegClass;
    unsigned R1 = MRI.createVirtualRegister(RC);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Define)
        .addReg(RS1.getReg())
        .addReg(RS2.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_SAT_2P2_P: {
    unsigned Opc = RISCV::ESP_SAT_2P2;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RD = MI.getOperand(1);
    MachineOperand &RS1 = MI.getOperand(2);
    MachineOperand &RS2 = MI.getOperand(3);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RD.getReg())
        .addReg(RS1.getReg())
        .addReg(RS2.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_SUBX2_2P2_P: {
    unsigned Opc = RISCV::ESP_SUBX2_2P2;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(0);
    MachineOperand &RS2 = MI.getOperand(1);
    const TargetRegisterClass *RC = &RISCV::GPRPIERegClass;
    unsigned R1 = MRI.createVirtualRegister(RC);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Define)
        .addReg(RS1.getReg())
        .addReg(RS2.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_SUBX4_2P2_P: {
    unsigned Opc = RISCV::ESP_SUBX4_2P2;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(0);
    MachineOperand &RS2 = MI.getOperand(1);
    const TargetRegisterClass *RC = &RISCV::GPRPIERegClass;
    unsigned R1 = MRI.createVirtualRegister(RC);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Define)
        .addReg(RS1.getReg())
        .addReg(RS2.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_MOVX_W_FFT_BIT_WIDTH_M_2P2_P: {
    unsigned Opc = RISCV::ESP_MOVX_W_FFT_BIT_WIDTH_2P2;
    MachineBasicBlock *MBB = MI.getParent();
    Register DstReg = MI.getOperand(0).getReg();
    Register SrcReg = MI.getOperand(1).getReg();
    BuildMI(*MBB, MI, DL, TII.get(Opc)).addReg(SrcReg);
    // $rd = $rs1 tie: regalloc may coalesce; COPY only if vregs differ here.
    if (DstReg != SrcReg)
      BuildMI(*MBB, MI, DL, TII.get(RISCV::COPY), DstReg).addReg(SrcReg);
    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VLDBC_16_IP_2P2_P: {
    unsigned Opc = RISCV::ESP_VLDBC_16_IP_2P2;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(1);
    MachineOperand &OFFSET_256_2 = MI.getOperand(2);
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vldbc_16_ip_2p2 first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS1.getReg())
        .addImm(OFFSET_256_2.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VLDBC_16_XP_2P2_P: {
    unsigned Opc = RISCV::ESP_VLDBC_16_XP_2P2;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &RS1 = MI.getOperand(2);
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vldbc_16_xp_2p2 first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS2.getReg())
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VLDBC_32_IP_2P2_P: {
    unsigned Opc = RISCV::ESP_VLDBC_32_IP_2P2;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(1);
    MachineOperand &OFFSET_256_4 = MI.getOperand(2);
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vldbc_32_ip_2p2 first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS1.getReg())
        .addImm(OFFSET_256_4.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VLDBC_32_XP_2P2_P: {
    unsigned Opc = RISCV::ESP_VLDBC_32_XP_2P2;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &RS1 = MI.getOperand(2);
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vldbc_32_xp_2p2 first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS2.getReg())
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VLDBC_8_IP_2P2_P: {
    unsigned Opc = RISCV::ESP_VLDBC_8_IP_2P2;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(1);
    MachineOperand &OFFSET_256_1 = MI.getOperand(2);
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vldbc_8_ip_2p2 first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS1.getReg())
        .addImm(OFFSET_256_1.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VLDBC_8_XP_2P2_P: {
    unsigned Opc = RISCV::ESP_VLDBC_8_XP_2P2;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &RS1 = MI.getOperand(2);
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vldbc_8_xp_2p2 first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS2.getReg())
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VLDEXT_S16_IP_2P2_P: {
    unsigned Opc = RISCV::ESP_VLDEXT_S16_IP_2P2;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(1);
    MachineOperand &OFFSET_16_16 = MI.getOperand(2);
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vldext_s16_ip_2p2 first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(4);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vldext_s16_ip_2p2 first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS1.getReg())
        .addImm(OFFSET_16_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VLDEXT_S16_XP_2P2_P: {
    unsigned Opc = RISCV::ESP_VLDEXT_S16_XP_2P2;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &RS1 = MI.getOperand(2);
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vldext_s16_xp_2p2 first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(4);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vldext_s16_xp_2p2 first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS2.getReg())
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VLDEXT_S8_IP_2P2_P: {
    unsigned Opc = RISCV::ESP_VLDEXT_S8_IP_2P2;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(1);
    MachineOperand &OFFSET_16_16 = MI.getOperand(2);
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vldext_s8_ip_2p2 first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(4);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vldext_s8_ip_2p2 first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS1.getReg())
        .addImm(OFFSET_16_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VLDEXT_S8_XP_2P2_P: {
    unsigned Opc = RISCV::ESP_VLDEXT_S8_XP_2P2;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &RS1 = MI.getOperand(2);
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vldext_s8_xp_2p2 first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(4);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vldext_s8_xp_2p2 first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS2.getReg())
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VLDEXT_U16_IP_2P2_P: {
    unsigned Opc = RISCV::ESP_VLDEXT_U16_IP_2P2;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(1);
    MachineOperand &OFFSET_16_16 = MI.getOperand(2);
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vldext_u16_ip_2p2 first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(4);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vldext_u16_ip_2p2 first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS1.getReg())
        .addImm(OFFSET_16_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VLDEXT_U16_XP_2P2_P: {
    unsigned Opc = RISCV::ESP_VLDEXT_U16_XP_2P2;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &RS1 = MI.getOperand(2);
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vldext_u16_xp_2p2 first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(4);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vldext_u16_xp_2p2 first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS2.getReg())
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VLDEXT_U8_IP_2P2_P: {
    unsigned Opc = RISCV::ESP_VLDEXT_U8_IP_2P2;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(1);
    MachineOperand &OFFSET_16_16 = MI.getOperand(2);
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vldext_u8_ip_2p2 first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(4);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vldext_u8_ip_2p2 first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS1.getReg())
        .addImm(OFFSET_16_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VLDEXT_U8_XP_2P2_P: {
    unsigned Opc = RISCV::ESP_VLDEXT_U8_XP_2P2;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &RS1 = MI.getOperand(2);
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vldext_u8_xp_2p2 first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(4);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vldext_u8_xp_2p2 first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS2.getReg())
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VLDHBC_16_INCP_2P2_P: {
    unsigned Opc = RISCV::ESP_VLDHBC_16_INCP_2P2;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(1);
    MachineOperand &QU = MI.getOperand(2);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vldhbc_16_incp_2p2 first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(3);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vldhbc_16_incp_2p2 first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VSL_U32_2P2_P: {
    unsigned Opc = RISCV::ESP_VSL_U32_2P2;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QY = MI.getOperand(0);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vsl_u32_2p2 first argument, "
                        "it must bi in range [0,7]");
    MachineOperand &SAT = MI.getOperand(1);
    MachineOperand &QU = MI.getOperand(2);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vsl_u32_2p2 first argument, "
                        "it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(RISCV::Q0 + QYVal)
        .addImm(SAT.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  }
}
