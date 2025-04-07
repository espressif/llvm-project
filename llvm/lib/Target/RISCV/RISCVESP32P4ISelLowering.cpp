//==- RISCVESP32P4ISelLowering.cpp - ESP32 P4 DAG Lowering Implementation -===//
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
  case RISCV::ESP_VCMULAS_S16_QACC_H_P: {
    unsigned Opc = RISCV::ESP_VCMULAS_S16_QACC_H;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vcmulas_s16_qacc_h first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vcmulas_s16_qacc_h first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VCMULAS_S16_QACC_H_LD_IP_P: {
    unsigned Opc = RISCV::ESP_VCMULAS_S16_QACC_H_LD_IP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vcmulas_s16_qacc_h_ld_ip "
                        "first argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vcmulas_s16_qacc_h_ld_ip "
                        "first argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &OFFSET_16_16 = MI.getOperand(4);
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vcmulas_s16_qacc_h_ld_ip "
                        "first argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg())
        .addImm(OFFSET_16_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VCMULAS_S16_QACC_H_LD_XP_P: {
    unsigned Opc = RISCV::ESP_VCMULAS_S16_QACC_H_LD_XP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &QX = MI.getOperand(2);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vcmulas_s16_qacc_h_ld_xp "
                        "first argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(3);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vcmulas_s16_qacc_h_ld_xp "
                        "first argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vcmulas_s16_qacc_h_ld_xp "
                        "first argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS2.getReg())
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VCMULAS_S16_QACC_L_P: {
    unsigned Opc = RISCV::ESP_VCMULAS_S16_QACC_L;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vcmulas_s16_qacc_l first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vcmulas_s16_qacc_l first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VCMULAS_S16_QACC_L_LD_IP_P: {
    unsigned Opc = RISCV::ESP_VCMULAS_S16_QACC_L_LD_IP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vcmulas_s16_qacc_l_ld_ip "
                        "first argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vcmulas_s16_qacc_l_ld_ip "
                        "first argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &OFFSET_16_16 = MI.getOperand(4);
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vcmulas_s16_qacc_l_ld_ip "
                        "first argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg())
        .addImm(OFFSET_16_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VCMULAS_S16_QACC_L_LD_XP_P: {
    unsigned Opc = RISCV::ESP_VCMULAS_S16_QACC_L_LD_XP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &QX = MI.getOperand(2);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vcmulas_s16_qacc_l_ld_xp "
                        "first argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(3);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vcmulas_s16_qacc_l_ld_xp "
                        "first argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vcmulas_s16_qacc_l_ld_xp "
                        "first argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS2.getReg())
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VCMULAS_S8_QACC_H_P: {
    unsigned Opc = RISCV::ESP_VCMULAS_S8_QACC_H;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vcmulas_s8_qacc_h first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vcmulas_s8_qacc_h first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VCMULAS_S8_QACC_H_LD_IP_P: {
    unsigned Opc = RISCV::ESP_VCMULAS_S8_QACC_H_LD_IP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vcmulas_s8_qacc_h_ld_ip first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vcmulas_s8_qacc_h_ld_ip first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &OFFSET_16_16 = MI.getOperand(4);
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vcmulas_s8_qacc_h_ld_ip first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg())
        .addImm(OFFSET_16_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VCMULAS_S8_QACC_H_LD_XP_P: {
    unsigned Opc = RISCV::ESP_VCMULAS_S8_QACC_H_LD_XP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &QX = MI.getOperand(2);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vcmulas_s8_qacc_h_ld_xp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(3);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vcmulas_s8_qacc_h_ld_xp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vcmulas_s8_qacc_h_ld_xp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS2.getReg())
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VCMULAS_S8_QACC_L_P: {
    unsigned Opc = RISCV::ESP_VCMULAS_S8_QACC_L;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vcmulas_s8_qacc_l first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vcmulas_s8_qacc_l first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VCMULAS_S8_QACC_L_LD_IP_P: {
    unsigned Opc = RISCV::ESP_VCMULAS_S8_QACC_L_LD_IP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vcmulas_s8_qacc_l_ld_ip first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vcmulas_s8_qacc_l_ld_ip first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &OFFSET_16_16 = MI.getOperand(4);
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vcmulas_s8_qacc_l_ld_ip first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg())
        .addImm(OFFSET_16_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VCMULAS_S8_QACC_L_LD_XP_P: {
    unsigned Opc = RISCV::ESP_VCMULAS_S8_QACC_L_LD_XP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &QX = MI.getOperand(2);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vcmulas_s8_qacc_l_ld_xp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(3);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vcmulas_s8_qacc_l_ld_xp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vcmulas_s8_qacc_l_ld_xp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS2.getReg())
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMULAS_S16_QACC_P: {
    unsigned Opc = RISCV::ESP_VMULAS_S16_QACC;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmulas_s16_qacc first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmulas_s16_qacc first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMULAS_S16_QACC_LD_IP_P: {
    unsigned Opc = RISCV::ESP_VMULAS_S16_QACC_LD_IP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmulas_s16_qacc_ld_ip first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmulas_s16_qacc_ld_ip first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &OFFSET_16_16 = MI.getOperand(4);
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmulas_s16_qacc_ld_ip first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg())
        .addImm(OFFSET_16_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMULAS_S16_QACC_LD_XP_P: {
    unsigned Opc = RISCV::ESP_VMULAS_S16_QACC_LD_XP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &QX = MI.getOperand(2);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmulas_s16_qacc_ld_xp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(3);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmulas_s16_qacc_ld_xp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmulas_s16_qacc_ld_xp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS2.getReg())
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMULAS_S16_QACC_ST_IP_P: {
    unsigned Opc = RISCV::ESP_VMULAS_S16_QACC_ST_IP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmulas_s16_qacc_st_ip first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmulas_s16_qacc_st_ip first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmulas_s16_qacc_st_ip first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &OFFSET_16_16 = MI.getOperand(5);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg())
        .addImm(OFFSET_16_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMULAS_S16_QACC_ST_XP_P: {
    unsigned Opc = RISCV::ESP_VMULAS_S16_QACC_ST_XP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &QX = MI.getOperand(2);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmulas_s16_qacc_st_xp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(3);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmulas_s16_qacc_st_xp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(4);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmulas_s16_qacc_st_xp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(5);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS2.getReg())
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMULAS_S16_XACC_P: {
    unsigned Opc = RISCV::ESP_VMULAS_S16_XACC;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmulas_s16_xacc first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmulas_s16_xacc first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMULAS_S16_XACC_LD_IP_P: {
    unsigned Opc = RISCV::ESP_VMULAS_S16_XACC_LD_IP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmulas_s16_xacc_ld_ip first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmulas_s16_xacc_ld_ip first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &OFFSET_16_16 = MI.getOperand(4);
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmulas_s16_xacc_ld_ip first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg())
        .addImm(OFFSET_16_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMULAS_S16_XACC_LD_XP_P: {
    unsigned Opc = RISCV::ESP_VMULAS_S16_XACC_LD_XP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &QX = MI.getOperand(2);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmulas_s16_xacc_ld_xp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(3);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmulas_s16_xacc_ld_xp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmulas_s16_xacc_ld_xp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS2.getReg())
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMULAS_S16_XACC_ST_IP_P: {
    unsigned Opc = RISCV::ESP_VMULAS_S16_XACC_ST_IP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmulas_s16_xacc_st_ip first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmulas_s16_xacc_st_ip first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmulas_s16_xacc_st_ip first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &OFFSET_16_16 = MI.getOperand(5);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg())
        .addImm(OFFSET_16_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMULAS_S16_XACC_ST_XP_P: {
    unsigned Opc = RISCV::ESP_VMULAS_S16_XACC_ST_XP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &QX = MI.getOperand(2);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmulas_s16_xacc_st_xp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(3);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmulas_s16_xacc_st_xp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(4);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmulas_s16_xacc_st_xp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(5);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS2.getReg())
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMULAS_S8_QACC_P: {
    unsigned Opc = RISCV::ESP_VMULAS_S8_QACC;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmulas_s8_qacc first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmulas_s8_qacc first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMULAS_S8_QACC_LD_IP_P: {
    unsigned Opc = RISCV::ESP_VMULAS_S8_QACC_LD_IP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmulas_s8_qacc_ld_ip first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmulas_s8_qacc_ld_ip first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &OFFSET_16_16 = MI.getOperand(4);
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmulas_s8_qacc_ld_ip first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg())
        .addImm(OFFSET_16_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMULAS_S8_QACC_LD_XP_P: {
    unsigned Opc = RISCV::ESP_VMULAS_S8_QACC_LD_XP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &QX = MI.getOperand(2);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmulas_s8_qacc_ld_xp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(3);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmulas_s8_qacc_ld_xp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmulas_s8_qacc_ld_xp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS2.getReg())
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMULAS_S8_QACC_ST_IP_P: {
    unsigned Opc = RISCV::ESP_VMULAS_S8_QACC_ST_IP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmulas_s8_qacc_st_ip first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmulas_s8_qacc_st_ip first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmulas_s8_qacc_st_ip first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &OFFSET_16_16 = MI.getOperand(5);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg())
        .addImm(OFFSET_16_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMULAS_S8_QACC_ST_XP_P: {
    unsigned Opc = RISCV::ESP_VMULAS_S8_QACC_ST_XP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &QX = MI.getOperand(2);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmulas_s8_qacc_st_xp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(3);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmulas_s8_qacc_st_xp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(4);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmulas_s8_qacc_st_xp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(5);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS2.getReg())
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMULAS_S8_XACC_P: {
    unsigned Opc = RISCV::ESP_VMULAS_S8_XACC;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmulas_s8_xacc first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmulas_s8_xacc first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMULAS_S8_XACC_LD_IP_P: {
    unsigned Opc = RISCV::ESP_VMULAS_S8_XACC_LD_IP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmulas_s8_xacc_ld_ip first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmulas_s8_xacc_ld_ip first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &OFFSET_16_16 = MI.getOperand(4);
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmulas_s8_xacc_ld_ip first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg())
        .addImm(OFFSET_16_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMULAS_S8_XACC_LD_XP_P: {
    unsigned Opc = RISCV::ESP_VMULAS_S8_XACC_LD_XP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &QX = MI.getOperand(2);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmulas_s8_xacc_ld_xp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(3);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmulas_s8_xacc_ld_xp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmulas_s8_xacc_ld_xp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS2.getReg())
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMULAS_S8_XACC_ST_IP_P: {
    unsigned Opc = RISCV::ESP_VMULAS_S8_XACC_ST_IP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmulas_s8_xacc_st_ip first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmulas_s8_xacc_st_ip first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmulas_s8_xacc_st_ip first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &OFFSET_16_16 = MI.getOperand(5);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg())
        .addImm(OFFSET_16_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMULAS_S8_XACC_ST_XP_P: {
    unsigned Opc = RISCV::ESP_VMULAS_S8_XACC_ST_XP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &QX = MI.getOperand(2);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmulas_s8_xacc_st_xp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(3);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmulas_s8_xacc_st_xp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(4);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmulas_s8_xacc_st_xp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(5);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS2.getReg())
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMULAS_U16_QACC_P: {
    unsigned Opc = RISCV::ESP_VMULAS_U16_QACC;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmulas_u16_qacc first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmulas_u16_qacc first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMULAS_U16_QACC_LD_IP_P: {
    unsigned Opc = RISCV::ESP_VMULAS_U16_QACC_LD_IP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmulas_u16_qacc_ld_ip first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmulas_u16_qacc_ld_ip first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &OFFSET_16_16 = MI.getOperand(4);
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmulas_u16_qacc_ld_ip first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg())
        .addImm(OFFSET_16_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMULAS_U16_QACC_LD_XP_P: {
    unsigned Opc = RISCV::ESP_VMULAS_U16_QACC_LD_XP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &QX = MI.getOperand(2);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmulas_u16_qacc_ld_xp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(3);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmulas_u16_qacc_ld_xp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmulas_u16_qacc_ld_xp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS2.getReg())
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMULAS_U16_QACC_ST_IP_P: {
    unsigned Opc = RISCV::ESP_VMULAS_U16_QACC_ST_IP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmulas_u16_qacc_st_ip first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmulas_u16_qacc_st_ip first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmulas_u16_qacc_st_ip first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &OFFSET_16_16 = MI.getOperand(5);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg())
        .addImm(OFFSET_16_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMULAS_U16_QACC_ST_XP_P: {
    unsigned Opc = RISCV::ESP_VMULAS_U16_QACC_ST_XP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &QX = MI.getOperand(2);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmulas_u16_qacc_st_xp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(3);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmulas_u16_qacc_st_xp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(4);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmulas_u16_qacc_st_xp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(5);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS2.getReg())
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMULAS_U16_XACC_P: {
    unsigned Opc = RISCV::ESP_VMULAS_U16_XACC;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmulas_u16_xacc first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmulas_u16_xacc first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMULAS_U16_XACC_LD_IP_P: {
    unsigned Opc = RISCV::ESP_VMULAS_U16_XACC_LD_IP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmulas_u16_xacc_ld_ip first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmulas_u16_xacc_ld_ip first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &OFFSET_16_16 = MI.getOperand(4);
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmulas_u16_xacc_ld_ip first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg())
        .addImm(OFFSET_16_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMULAS_U16_XACC_LD_XP_P: {
    unsigned Opc = RISCV::ESP_VMULAS_U16_XACC_LD_XP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &QX = MI.getOperand(2);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmulas_u16_xacc_ld_xp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(3);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmulas_u16_xacc_ld_xp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmulas_u16_xacc_ld_xp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS2.getReg())
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMULAS_U16_XACC_ST_IP_P: {
    unsigned Opc = RISCV::ESP_VMULAS_U16_XACC_ST_IP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmulas_u16_xacc_st_ip first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmulas_u16_xacc_st_ip first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmulas_u16_xacc_st_ip first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &OFFSET_16_16 = MI.getOperand(5);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg())
        .addImm(OFFSET_16_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMULAS_U16_XACC_ST_XP_P: {
    unsigned Opc = RISCV::ESP_VMULAS_U16_XACC_ST_XP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &QX = MI.getOperand(2);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmulas_u16_xacc_st_xp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(3);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmulas_u16_xacc_st_xp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(4);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmulas_u16_xacc_st_xp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(5);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS2.getReg())
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMULAS_U8_QACC_P: {
    unsigned Opc = RISCV::ESP_VMULAS_U8_QACC;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmulas_u8_qacc first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmulas_u8_qacc first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMULAS_U8_QACC_LD_IP_P: {
    unsigned Opc = RISCV::ESP_VMULAS_U8_QACC_LD_IP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmulas_u8_qacc_ld_ip first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmulas_u8_qacc_ld_ip first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &OFFSET_16_16 = MI.getOperand(4);
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmulas_u8_qacc_ld_ip first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg())
        .addImm(OFFSET_16_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMULAS_U8_QACC_LD_XP_P: {
    unsigned Opc = RISCV::ESP_VMULAS_U8_QACC_LD_XP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &QX = MI.getOperand(2);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmulas_u8_qacc_ld_xp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(3);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmulas_u8_qacc_ld_xp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmulas_u8_qacc_ld_xp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS2.getReg())
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMULAS_U8_QACC_ST_IP_P: {
    unsigned Opc = RISCV::ESP_VMULAS_U8_QACC_ST_IP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmulas_u8_qacc_st_ip first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmulas_u8_qacc_st_ip first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmulas_u8_qacc_st_ip first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &OFFSET_16_16 = MI.getOperand(5);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg())
        .addImm(OFFSET_16_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMULAS_U8_QACC_ST_XP_P: {
    unsigned Opc = RISCV::ESP_VMULAS_U8_QACC_ST_XP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &QX = MI.getOperand(2);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmulas_u8_qacc_st_xp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(3);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmulas_u8_qacc_st_xp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(4);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmulas_u8_qacc_st_xp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(5);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS2.getReg())
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMULAS_U8_XACC_P: {
    unsigned Opc = RISCV::ESP_VMULAS_U8_XACC;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmulas_u8_xacc first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmulas_u8_xacc first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMULAS_U8_XACC_LD_IP_P: {
    unsigned Opc = RISCV::ESP_VMULAS_U8_XACC_LD_IP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmulas_u8_xacc_ld_ip first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmulas_u8_xacc_ld_ip first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &OFFSET_16_16 = MI.getOperand(4);
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmulas_u8_xacc_ld_ip first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg())
        .addImm(OFFSET_16_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMULAS_U8_XACC_LD_XP_P: {
    unsigned Opc = RISCV::ESP_VMULAS_U8_XACC_LD_XP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &QX = MI.getOperand(2);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmulas_u8_xacc_ld_xp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(3);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmulas_u8_xacc_ld_xp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmulas_u8_xacc_ld_xp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS2.getReg())
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMULAS_U8_XACC_ST_IP_P: {
    unsigned Opc = RISCV::ESP_VMULAS_U8_XACC_ST_IP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmulas_u8_xacc_st_ip first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmulas_u8_xacc_st_ip first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmulas_u8_xacc_st_ip first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &OFFSET_16_16 = MI.getOperand(5);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg())
        .addImm(OFFSET_16_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMULAS_U8_XACC_ST_XP_P: {
    unsigned Opc = RISCV::ESP_VMULAS_U8_XACC_ST_XP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &QX = MI.getOperand(2);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmulas_u8_xacc_st_xp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(3);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmulas_u8_xacc_st_xp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(4);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmulas_u8_xacc_st_xp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(5);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS2.getReg())
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMULAS_S16_QACC_LDBC_INCP_P: {
    unsigned Opc = RISCV::ESP_VMULAS_S16_QACC_LDBC_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmulas_s16_qacc_ldbc_incp "
                        "first argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmulas_s16_qacc_ldbc_incp "
                        "first argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &QU = MI.getOperand(4);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmulas_s16_qacc_ldbc_incp "
                        "first argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMULAS_S8_QACC_LDBC_INCP_P: {
    unsigned Opc = RISCV::ESP_VMULAS_S8_QACC_LDBC_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmulas_s8_qacc_ldbc_incp "
                        "first argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmulas_s8_qacc_ldbc_incp "
                        "first argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &QU = MI.getOperand(4);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmulas_s8_qacc_ldbc_incp "
                        "first argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMULAS_U16_QACC_LDBC_INCP_P: {
    unsigned Opc = RISCV::ESP_VMULAS_U16_QACC_LDBC_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmulas_u16_qacc_ldbc_incp "
                        "first argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmulas_u16_qacc_ldbc_incp "
                        "first argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &QU = MI.getOperand(4);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmulas_u16_qacc_ldbc_incp "
                        "first argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMULAS_U8_QACC_LDBC_INCP_P: {
    unsigned Opc = RISCV::ESP_VMULAS_U8_QACC_LDBC_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmulas_u8_qacc_ldbc_incp "
                        "first argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmulas_u8_qacc_ldbc_incp "
                        "first argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &QU = MI.getOperand(4);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmulas_u8_qacc_ldbc_incp "
                        "first argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VSMULAS_S16_QACC_P: {
    unsigned Opc = RISCV::ESP_VSMULAS_S16_QACC;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vsmulas_s16_qacc first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vsmulas_s16_qacc first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &SELECT_16 = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addImm(SELECT_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VSMULAS_S16_QACC_LD_INCP_P: {
    unsigned Opc = RISCV::ESP_VSMULAS_S16_QACC_LD_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vsmulas_s16_qacc_ld_incp "
                        "first argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vsmulas_s16_qacc_ld_incp "
                        "first argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &SELECT_16 = MI.getOperand(4);
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vsmulas_s16_qacc_ld_incp "
                        "first argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg())
        .addImm(SELECT_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VSMULAS_S8_QACC_P: {
    unsigned Opc = RISCV::ESP_VSMULAS_S8_QACC;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vsmulas_s8_qacc first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vsmulas_s8_qacc first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &SELECT_16 = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addImm(SELECT_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VSMULAS_S8_QACC_LD_INCP_P: {
    unsigned Opc = RISCV::ESP_VSMULAS_S8_QACC_LD_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vsmulas_s8_qacc_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vsmulas_s8_qacc_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &SELECT_16 = MI.getOperand(4);
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vsmulas_s8_qacc_ld_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg())
        .addImm(SELECT_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VSMULAS_U16_QACC_P: {
    unsigned Opc = RISCV::ESP_VSMULAS_U16_QACC;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vsmulas_u16_qacc first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vsmulas_u16_qacc first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &SELECT_16 = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addImm(SELECT_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VSMULAS_U16_QACC_LD_INCP_P: {
    unsigned Opc = RISCV::ESP_VSMULAS_U16_QACC_LD_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vsmulas_u16_qacc_ld_incp "
                        "first argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vsmulas_u16_qacc_ld_incp "
                        "first argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &SELECT_16 = MI.getOperand(4);
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vsmulas_u16_qacc_ld_incp "
                        "first argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg())
        .addImm(SELECT_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VSMULAS_U8_QACC_P: {
    unsigned Opc = RISCV::ESP_VSMULAS_U8_QACC;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vsmulas_u8_qacc first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vsmulas_u8_qacc first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &SELECT_16 = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addImm(SELECT_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VSMULAS_U8_QACC_LD_INCP_P: {
    unsigned Opc = RISCV::ESP_VSMULAS_U8_QACC_LD_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vsmulas_u8_qacc_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vsmulas_u8_qacc_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &SELECT_16 = MI.getOperand(4);
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vsmulas_u8_qacc_ld_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg())
        .addImm(SELECT_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_CMUL_S16_P: {
    unsigned Opc = RISCV::ESP_CMUL_S16;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_cmul_s16 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_cmul_s16 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &SELECT_4 = MI.getOperand(2);
    MachineOperand &QZ = MI.getOperand(3);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_cmul_s16 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addImm(SELECT_4.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_CMUL_S16_LD_INCP_P: {
    unsigned Opc = RISCV::ESP_CMUL_S16_LD_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_cmul_s16_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_cmul_s16_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &SELECT_4 = MI.getOperand(4);
    MachineOperand &QZ = MI.getOperand(5);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_cmul_s16_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(6);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_cmul_s16_ld_incp first "
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
  case RISCV::ESP_CMUL_S16_ST_INCP_P: {
    unsigned Opc = RISCV::ESP_CMUL_S16_ST_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_cmul_s16_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_cmul_s16_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_cmul_s16_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &SELECT_4 = MI.getOperand(5);
    MachineOperand &QZ = MI.getOperand(6);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_cmul_s16_st_incp first "
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
  case RISCV::ESP_CMUL_S8_P: {
    unsigned Opc = RISCV::ESP_CMUL_S8;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_cmul_s8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_cmul_s8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &SELECT_4 = MI.getOperand(2);
    MachineOperand &QZ = MI.getOperand(3);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_cmul_s8 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addImm(SELECT_4.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_CMUL_S8_LD_INCP_P: {
    unsigned Opc = RISCV::ESP_CMUL_S8_LD_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_cmul_s8_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_cmul_s8_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &SELECT_4 = MI.getOperand(4);
    MachineOperand &QZ = MI.getOperand(5);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_cmul_s8_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(6);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_cmul_s8_ld_incp first "
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
  case RISCV::ESP_CMUL_S8_ST_INCP_P: {
    unsigned Opc = RISCV::ESP_CMUL_S8_ST_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_cmul_s8_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_cmul_s8_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_cmul_s8_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &SELECT_4 = MI.getOperand(5);
    MachineOperand &QZ = MI.getOperand(6);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_cmul_s8_st_incp first "
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
  case RISCV::ESP_MAX_S16_A_P: {
    unsigned Opc = RISCV::ESP_MAX_S16_A;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QW = MI.getOperand(0);
    unsigned QWVal = QW.getImm();
    assert(QWVal < 8 && "Unexpected value of esp_max_s16_a first argument, it "
                        "must bi in range [0,7]");
    const TargetRegisterClass *RC = &RISCV::GPRPIERegClass;
    unsigned R1 = MRI.createVirtualRegister(RC);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Define)
        .addReg(RISCV::Q0 + QWVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_MAX_S32_A_P: {
    unsigned Opc = RISCV::ESP_MAX_S32_A;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QW = MI.getOperand(0);
    unsigned QWVal = QW.getImm();
    assert(QWVal < 8 && "Unexpected value of esp_max_s32_a first argument, it "
                        "must bi in range [0,7]");
    const TargetRegisterClass *RC = &RISCV::GPRPIERegClass;
    unsigned R1 = MRI.createVirtualRegister(RC);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Define)
        .addReg(RISCV::Q0 + QWVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_MAX_S8_A_P: {
    unsigned Opc = RISCV::ESP_MAX_S8_A;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QW = MI.getOperand(0);
    unsigned QWVal = QW.getImm();
    assert(QWVal < 8 && "Unexpected value of esp_max_s8_a first argument, it "
                        "must bi in range [0,7]");
    const TargetRegisterClass *RC = &RISCV::GPRPIERegClass;
    unsigned R1 = MRI.createVirtualRegister(RC);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Define)
        .addReg(RISCV::Q0 + QWVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_MAX_U16_A_P: {
    unsigned Opc = RISCV::ESP_MAX_U16_A;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QW = MI.getOperand(0);
    unsigned QWVal = QW.getImm();
    assert(QWVal < 8 && "Unexpected value of esp_max_u16_a first argument, it "
                        "must bi in range [0,7]");
    const TargetRegisterClass *RC = &RISCV::GPRPIERegClass;
    unsigned R1 = MRI.createVirtualRegister(RC);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Define)
        .addReg(RISCV::Q0 + QWVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_MAX_U32_A_P: {
    unsigned Opc = RISCV::ESP_MAX_U32_A;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QW = MI.getOperand(0);
    unsigned QWVal = QW.getImm();
    assert(QWVal < 8 && "Unexpected value of esp_max_u32_a first argument, it "
                        "must bi in range [0,7]");
    const TargetRegisterClass *RC = &RISCV::GPRPIERegClass;
    unsigned R1 = MRI.createVirtualRegister(RC);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Define)
        .addReg(RISCV::Q0 + QWVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_MAX_U8_A_P: {
    unsigned Opc = RISCV::ESP_MAX_U8_A;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QW = MI.getOperand(0);
    unsigned QWVal = QW.getImm();
    assert(QWVal < 8 && "Unexpected value of esp_max_u8_a first argument, it "
                        "must bi in range [0,7]");
    const TargetRegisterClass *RC = &RISCV::GPRPIERegClass;
    unsigned R1 = MRI.createVirtualRegister(RC);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Define)
        .addReg(RISCV::Q0 + QWVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_MIN_S16_A_P: {
    unsigned Opc = RISCV::ESP_MIN_S16_A;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QW = MI.getOperand(0);
    unsigned QWVal = QW.getImm();
    assert(QWVal < 8 && "Unexpected value of esp_min_s16_a first argument, it "
                        "must bi in range [0,7]");
    const TargetRegisterClass *RC = &RISCV::GPRPIERegClass;
    unsigned R1 = MRI.createVirtualRegister(RC);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Define)
        .addReg(RISCV::Q0 + QWVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_MIN_S32_A_P: {
    unsigned Opc = RISCV::ESP_MIN_S32_A;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QW = MI.getOperand(0);
    unsigned QWVal = QW.getImm();
    assert(QWVal < 8 && "Unexpected value of esp_min_s32_a first argument, it "
                        "must bi in range [0,7]");
    const TargetRegisterClass *RC = &RISCV::GPRPIERegClass;
    unsigned R1 = MRI.createVirtualRegister(RC);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Define)
        .addReg(RISCV::Q0 + QWVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_MIN_S8_A_P: {
    unsigned Opc = RISCV::ESP_MIN_S8_A;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QW = MI.getOperand(0);
    unsigned QWVal = QW.getImm();
    assert(QWVal < 8 && "Unexpected value of esp_min_s8_a first argument, it "
                        "must bi in range [0,7]");
    const TargetRegisterClass *RC = &RISCV::GPRPIERegClass;
    unsigned R1 = MRI.createVirtualRegister(RC);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Define)
        .addReg(RISCV::Q0 + QWVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_MIN_U16_A_P: {
    unsigned Opc = RISCV::ESP_MIN_U16_A;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QW = MI.getOperand(0);
    unsigned QWVal = QW.getImm();
    assert(QWVal < 8 && "Unexpected value of esp_min_u16_a first argument, it "
                        "must bi in range [0,7]");
    const TargetRegisterClass *RC = &RISCV::GPRPIERegClass;
    unsigned R1 = MRI.createVirtualRegister(RC);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Define)
        .addReg(RISCV::Q0 + QWVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_MIN_U32_A_P: {
    unsigned Opc = RISCV::ESP_MIN_U32_A;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QW = MI.getOperand(0);
    unsigned QWVal = QW.getImm();
    assert(QWVal < 8 && "Unexpected value of esp_min_u32_a first argument, it "
                        "must bi in range [0,7]");
    const TargetRegisterClass *RC = &RISCV::GPRPIERegClass;
    unsigned R1 = MRI.createVirtualRegister(RC);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Define)
        .addReg(RISCV::Q0 + QWVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_MIN_U8_A_P: {
    unsigned Opc = RISCV::ESP_MIN_U8_A;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QW = MI.getOperand(0);
    unsigned QWVal = QW.getImm();
    assert(QWVal < 8 && "Unexpected value of esp_min_u8_a first argument, it "
                        "must bi in range [0,7]");
    const TargetRegisterClass *RC = &RISCV::GPRPIERegClass;
    unsigned R1 = MRI.createVirtualRegister(RC);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Define)
        .addReg(RISCV::Q0 + QWVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VABS_16_P: {
    unsigned Opc = RISCV::ESP_VABS_16;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QY = MI.getOperand(0);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vabs_16 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QV = MI.getOperand(1);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_vabs_16 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VABS_32_P: {
    unsigned Opc = RISCV::ESP_VABS_32;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QY = MI.getOperand(0);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vabs_32 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QV = MI.getOperand(1);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_vabs_32 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VABS_8_P: {
    unsigned Opc = RISCV::ESP_VABS_8;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QY = MI.getOperand(0);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vabs_8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QV = MI.getOperand(1);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_vabs_8 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VADD_S16_P: {
    unsigned Opc = RISCV::ESP_VADD_S16;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vadd_s16 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vadd_s16 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QV = MI.getOperand(2);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_vadd_s16 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VADD_S16_LD_INCP_P: {
    unsigned Opc = RISCV::ESP_VADD_S16_LD_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vadd_s16_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vadd_s16_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &QV = MI.getOperand(4);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_vadd_s16_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vadd_s16_ld_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VADD_S16_ST_INCP_P: {
    unsigned Opc = RISCV::ESP_VADD_S16_ST_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vadd_s16_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vadd_s16_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vadd_s16_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &QV = MI.getOperand(5);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_vadd_s16_st_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VADD_S32_P: {
    unsigned Opc = RISCV::ESP_VADD_S32;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vadd_s32 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vadd_s32 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QV = MI.getOperand(2);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_vadd_s32 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VADD_S32_LD_INCP_P: {
    unsigned Opc = RISCV::ESP_VADD_S32_LD_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vadd_s32_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vadd_s32_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &QV = MI.getOperand(4);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_vadd_s32_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vadd_s32_ld_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VADD_S32_ST_INCP_P: {
    unsigned Opc = RISCV::ESP_VADD_S32_ST_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vadd_s32_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vadd_s32_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vadd_s32_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &QV = MI.getOperand(5);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_vadd_s32_st_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VADD_S8_P: {
    unsigned Opc = RISCV::ESP_VADD_S8;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vadd_s8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vadd_s8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QV = MI.getOperand(2);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_vadd_s8 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VADD_S8_LD_INCP_P: {
    unsigned Opc = RISCV::ESP_VADD_S8_LD_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vadd_s8_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vadd_s8_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &QV = MI.getOperand(4);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_vadd_s8_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vadd_s8_ld_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VADD_S8_ST_INCP_P: {
    unsigned Opc = RISCV::ESP_VADD_S8_ST_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vadd_s8_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vadd_s8_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vadd_s8_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &QV = MI.getOperand(5);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_vadd_s8_st_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VADD_U16_P: {
    unsigned Opc = RISCV::ESP_VADD_U16;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vadd_u16 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vadd_u16 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QV = MI.getOperand(2);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_vadd_u16 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VADD_U16_LD_INCP_P: {
    unsigned Opc = RISCV::ESP_VADD_U16_LD_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vadd_u16_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vadd_u16_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &QV = MI.getOperand(4);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_vadd_u16_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vadd_u16_ld_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VADD_U16_ST_INCP_P: {
    unsigned Opc = RISCV::ESP_VADD_U16_ST_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vadd_u16_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vadd_u16_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vadd_u16_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &QV = MI.getOperand(5);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_vadd_u16_st_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VADD_U32_P: {
    unsigned Opc = RISCV::ESP_VADD_U32;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vadd_u32 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vadd_u32 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QV = MI.getOperand(2);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_vadd_u32 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VADD_U32_LD_INCP_P: {
    unsigned Opc = RISCV::ESP_VADD_U32_LD_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vadd_u32_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vadd_u32_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &QV = MI.getOperand(4);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_vadd_u32_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vadd_u32_ld_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VADD_U32_ST_INCP_P: {
    unsigned Opc = RISCV::ESP_VADD_U32_ST_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vadd_u32_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vadd_u32_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vadd_u32_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &QV = MI.getOperand(5);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_vadd_u32_st_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VADD_U8_P: {
    unsigned Opc = RISCV::ESP_VADD_U8;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vadd_u8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vadd_u8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QV = MI.getOperand(2);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_vadd_u8 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VADD_U8_LD_INCP_P: {
    unsigned Opc = RISCV::ESP_VADD_U8_LD_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vadd_u8_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vadd_u8_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &QV = MI.getOperand(4);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_vadd_u8_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vadd_u8_ld_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VADD_U8_ST_INCP_P: {
    unsigned Opc = RISCV::ESP_VADD_U8_ST_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vadd_u8_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vadd_u8_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vadd_u8_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &QV = MI.getOperand(5);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_vadd_u8_st_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VCLAMP_S16_P: {
    unsigned Opc = RISCV::ESP_VCLAMP_S16;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vclamp_s16 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &SELECT_16 = MI.getOperand(1);
    MachineOperand &QZ = MI.getOperand(2);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vclamp_s16 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addImm(SELECT_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMAX_S16_P: {
    unsigned Opc = RISCV::ESP_VMAX_S16;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmax_s16 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmax_s16 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(2);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vmax_s16 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMAX_S16_LD_INCP_P: {
    unsigned Opc = RISCV::ESP_VMAX_S16_LD_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmax_s16_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmax_s16_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &QZ = MI.getOperand(4);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vmax_s16_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmax_s16_ld_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMAX_S16_ST_INCP_P: {
    unsigned Opc = RISCV::ESP_VMAX_S16_ST_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmax_s16_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmax_s16_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmax_s16_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &QZ = MI.getOperand(5);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vmax_s16_st_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMAX_S32_P: {
    unsigned Opc = RISCV::ESP_VMAX_S32;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmax_s32 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmax_s32 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(2);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vmax_s32 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMAX_S32_LD_INCP_P: {
    unsigned Opc = RISCV::ESP_VMAX_S32_LD_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmax_s32_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmax_s32_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &QZ = MI.getOperand(4);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vmax_s32_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmax_s32_ld_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMAX_S32_ST_INCP_P: {
    unsigned Opc = RISCV::ESP_VMAX_S32_ST_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmax_s32_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmax_s32_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmax_s32_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &QZ = MI.getOperand(5);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vmax_s32_st_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMAX_S8_P: {
    unsigned Opc = RISCV::ESP_VMAX_S8;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmax_s8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmax_s8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(2);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vmax_s8 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMAX_S8_LD_INCP_P: {
    unsigned Opc = RISCV::ESP_VMAX_S8_LD_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmax_s8_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmax_s8_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &QZ = MI.getOperand(4);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vmax_s8_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmax_s8_ld_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMAX_S8_ST_INCP_P: {
    unsigned Opc = RISCV::ESP_VMAX_S8_ST_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmax_s8_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmax_s8_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmax_s8_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &QZ = MI.getOperand(5);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vmax_s8_st_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMAX_U16_P: {
    unsigned Opc = RISCV::ESP_VMAX_U16;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmax_u16 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmax_u16 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(2);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vmax_u16 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMAX_U16_LD_INCP_P: {
    unsigned Opc = RISCV::ESP_VMAX_U16_LD_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmax_u16_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmax_u16_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &QZ = MI.getOperand(4);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vmax_u16_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmax_u16_ld_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMAX_U16_ST_INCP_P: {
    unsigned Opc = RISCV::ESP_VMAX_U16_ST_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmax_u16_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmax_u16_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmax_u16_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &QZ = MI.getOperand(5);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vmax_u16_st_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMAX_U32_P: {
    unsigned Opc = RISCV::ESP_VMAX_U32;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmax_u32 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmax_u32 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(2);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vmax_u32 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMAX_U32_LD_INCP_P: {
    unsigned Opc = RISCV::ESP_VMAX_U32_LD_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmax_u32_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmax_u32_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &QZ = MI.getOperand(4);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vmax_u32_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmax_u32_ld_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMAX_U32_ST_INCP_P: {
    unsigned Opc = RISCV::ESP_VMAX_U32_ST_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmax_u32_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmax_u32_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmax_u32_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &QZ = MI.getOperand(5);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vmax_u32_st_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMAX_U8_P: {
    unsigned Opc = RISCV::ESP_VMAX_U8;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmax_u8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmax_u8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(2);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vmax_u8 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMAX_U8_LD_INCP_P: {
    unsigned Opc = RISCV::ESP_VMAX_U8_LD_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmax_u8_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmax_u8_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &QZ = MI.getOperand(4);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vmax_u8_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmax_u8_ld_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMAX_U8_ST_INCP_P: {
    unsigned Opc = RISCV::ESP_VMAX_U8_ST_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmax_u8_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmax_u8_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmax_u8_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &QZ = MI.getOperand(5);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vmax_u8_st_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMIN_S16_P: {
    unsigned Opc = RISCV::ESP_VMIN_S16;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmin_s16 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmin_s16 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(2);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vmin_s16 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMIN_S16_LD_INCP_P: {
    unsigned Opc = RISCV::ESP_VMIN_S16_LD_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmin_s16_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmin_s16_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &QZ = MI.getOperand(4);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vmin_s16_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmin_s16_ld_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMIN_S16_ST_INCP_P: {
    unsigned Opc = RISCV::ESP_VMIN_S16_ST_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmin_s16_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmin_s16_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmin_s16_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &QZ = MI.getOperand(5);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vmin_s16_st_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMIN_S32_P: {
    unsigned Opc = RISCV::ESP_VMIN_S32;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmin_s32 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmin_s32 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(2);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vmin_s32 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMIN_S32_LD_INCP_P: {
    unsigned Opc = RISCV::ESP_VMIN_S32_LD_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmin_s32_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmin_s32_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &QZ = MI.getOperand(4);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vmin_s32_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmin_s32_ld_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMIN_S32_ST_INCP_P: {
    unsigned Opc = RISCV::ESP_VMIN_S32_ST_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmin_s32_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmin_s32_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmin_s32_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &QZ = MI.getOperand(5);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vmin_s32_st_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMIN_S8_P: {
    unsigned Opc = RISCV::ESP_VMIN_S8;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmin_s8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmin_s8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(2);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vmin_s8 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMIN_S8_LD_INCP_P: {
    unsigned Opc = RISCV::ESP_VMIN_S8_LD_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmin_s8_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmin_s8_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &QZ = MI.getOperand(4);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vmin_s8_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmin_s8_ld_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMIN_S8_ST_INCP_P: {
    unsigned Opc = RISCV::ESP_VMIN_S8_ST_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmin_s8_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmin_s8_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmin_s8_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &QZ = MI.getOperand(5);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vmin_s8_st_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMIN_U16_P: {
    unsigned Opc = RISCV::ESP_VMIN_U16;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmin_u16 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmin_u16 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(2);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vmin_u16 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMIN_U16_LD_INCP_P: {
    unsigned Opc = RISCV::ESP_VMIN_U16_LD_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmin_u16_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmin_u16_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &QZ = MI.getOperand(4);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vmin_u16_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmin_u16_ld_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMIN_U16_ST_INCP_P: {
    unsigned Opc = RISCV::ESP_VMIN_U16_ST_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmin_u16_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmin_u16_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmin_u16_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &QZ = MI.getOperand(5);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vmin_u16_st_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMIN_U32_P: {
    unsigned Opc = RISCV::ESP_VMIN_U32;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmin_u32 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmin_u32 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(2);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vmin_u32 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMIN_U32_LD_INCP_P: {
    unsigned Opc = RISCV::ESP_VMIN_U32_LD_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmin_u32_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmin_u32_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &QZ = MI.getOperand(4);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vmin_u32_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmin_u32_ld_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMIN_U32_ST_INCP_P: {
    unsigned Opc = RISCV::ESP_VMIN_U32_ST_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmin_u32_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmin_u32_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmin_u32_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &QZ = MI.getOperand(5);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vmin_u32_st_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMIN_U8_P: {
    unsigned Opc = RISCV::ESP_VMIN_U8;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmin_u8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmin_u8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(2);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vmin_u8 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMIN_U8_LD_INCP_P: {
    unsigned Opc = RISCV::ESP_VMIN_U8_LD_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmin_u8_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmin_u8_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &QZ = MI.getOperand(4);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vmin_u8_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmin_u8_ld_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMIN_U8_ST_INCP_P: {
    unsigned Opc = RISCV::ESP_VMIN_U8_ST_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmin_u8_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmin_u8_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmin_u8_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &QZ = MI.getOperand(5);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vmin_u8_st_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMUL_S16_P: {
    unsigned Opc = RISCV::ESP_VMUL_S16;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmul_s16 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmul_s16 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(2);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vmul_s16 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMUL_S16_LD_INCP_P: {
    unsigned Opc = RISCV::ESP_VMUL_S16_LD_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmul_s16_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmul_s16_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &QZ = MI.getOperand(4);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vmul_s16_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmul_s16_ld_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMUL_S16_S8XS8_P: {
    unsigned Opc = RISCV::ESP_VMUL_S16_S8XS8;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmul_s16_s8xs8 first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmul_s16_s8xs8 first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(2);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vmul_s16_s8xs8 first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QV = MI.getOperand(3);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_vmul_s16_s8xs8 first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMUL_S16_ST_INCP_P: {
    unsigned Opc = RISCV::ESP_VMUL_S16_ST_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmul_s16_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmul_s16_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmul_s16_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &QZ = MI.getOperand(5);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vmul_s16_st_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMUL_S32_S16XS16_P: {
    unsigned Opc = RISCV::ESP_VMUL_S32_S16XS16;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmul_s32_s16xs16 first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmul_s32_s16xs16 first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(2);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vmul_s32_s16xs16 first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QV = MI.getOperand(3);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_vmul_s32_s16xs16 first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMUL_S8_P: {
    unsigned Opc = RISCV::ESP_VMUL_S8;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmul_s8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmul_s8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(2);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vmul_s8 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMUL_S8_LD_INCP_P: {
    unsigned Opc = RISCV::ESP_VMUL_S8_LD_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmul_s8_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmul_s8_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &QZ = MI.getOperand(4);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vmul_s8_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmul_s8_ld_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMUL_S8_ST_INCP_P: {
    unsigned Opc = RISCV::ESP_VMUL_S8_ST_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmul_s8_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmul_s8_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmul_s8_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &QZ = MI.getOperand(5);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vmul_s8_st_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMUL_U16_P: {
    unsigned Opc = RISCV::ESP_VMUL_U16;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmul_u16 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmul_u16 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(2);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vmul_u16 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMUL_U16_LD_INCP_P: {
    unsigned Opc = RISCV::ESP_VMUL_U16_LD_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmul_u16_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmul_u16_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &QZ = MI.getOperand(4);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vmul_u16_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmul_u16_ld_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMUL_U16_ST_INCP_P: {
    unsigned Opc = RISCV::ESP_VMUL_U16_ST_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmul_u16_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmul_u16_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmul_u16_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &QZ = MI.getOperand(5);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vmul_u16_st_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMUL_U8_P: {
    unsigned Opc = RISCV::ESP_VMUL_U8;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmul_u8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmul_u8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(2);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vmul_u8 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMUL_U8_LD_INCP_P: {
    unsigned Opc = RISCV::ESP_VMUL_U8_LD_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmul_u8_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmul_u8_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &QZ = MI.getOperand(4);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vmul_u8_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmul_u8_ld_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VMUL_U8_ST_INCP_P: {
    unsigned Opc = RISCV::ESP_VMUL_U8_ST_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vmul_u8_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vmul_u8_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vmul_u8_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &QZ = MI.getOperand(5);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vmul_u8_st_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VPRELU_S16_P: {
    unsigned Opc = RISCV::ESP_VPRELU_S16;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(0);
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vprelu_s16 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vprelu_s16 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(3);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vprelu_s16 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RS1.getReg())
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VPRELU_S8_P: {
    unsigned Opc = RISCV::ESP_VPRELU_S8;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(0);
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vprelu_s8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vprelu_s8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(3);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vprelu_s8 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RS1.getReg())
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VRELU_S16_P: {
    unsigned Opc = RISCV::ESP_VRELU_S16;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(0);
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vrelu_s16 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QYVal, RegState::Define)
        .addReg(RS1.getReg())
        .addReg(RS2.getReg())
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VRELU_S8_P: {
    unsigned Opc = RISCV::ESP_VRELU_S8;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(0);
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vrelu_s8 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QYVal, RegState::Define)
        .addReg(RS1.getReg())
        .addReg(RS2.getReg())
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VSADDS_S16_P: {
    unsigned Opc = RISCV::ESP_VSADDS_S16;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(0);
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vsadds_s16 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QV = MI.getOperand(2);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_vsadds_s16 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(RS1.getReg())
        .addReg(RISCV::Q0 + QXVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VSADDS_S8_P: {
    unsigned Opc = RISCV::ESP_VSADDS_S8;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(0);
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vsadds_s8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QV = MI.getOperand(2);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_vsadds_s8 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(RS1.getReg())
        .addReg(RISCV::Q0 + QXVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VSADDS_U16_P: {
    unsigned Opc = RISCV::ESP_VSADDS_U16;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(0);
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vsadds_u16 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QV = MI.getOperand(2);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_vsadds_u16 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(RS1.getReg())
        .addReg(RISCV::Q0 + QXVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VSADDS_U8_P: {
    unsigned Opc = RISCV::ESP_VSADDS_U8;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(0);
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vsadds_u8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QV = MI.getOperand(2);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_vsadds_u8 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(RS1.getReg())
        .addReg(RISCV::Q0 + QXVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VSAT_S16_P: {
    unsigned Opc = RISCV::ESP_VSAT_S16;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(0);
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &QX = MI.getOperand(2);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vsat_s16 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(3);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vsat_s16 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RS1.getReg())
        .addReg(RS2.getReg())
        .addReg(RISCV::Q0 + QXVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VSAT_S32_P: {
    unsigned Opc = RISCV::ESP_VSAT_S32;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(0);
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &QX = MI.getOperand(2);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vsat_s32 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(3);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vsat_s32 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RS1.getReg())
        .addReg(RS2.getReg())
        .addReg(RISCV::Q0 + QXVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VSAT_S8_P: {
    unsigned Opc = RISCV::ESP_VSAT_S8;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(0);
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &QX = MI.getOperand(2);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vsat_s8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(3);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vsat_s8 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RS1.getReg())
        .addReg(RS2.getReg())
        .addReg(RISCV::Q0 + QXVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VSAT_U16_P: {
    unsigned Opc = RISCV::ESP_VSAT_U16;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(0);
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &QX = MI.getOperand(2);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vsat_u16 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(3);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vsat_u16 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RS1.getReg())
        .addReg(RS2.getReg())
        .addReg(RISCV::Q0 + QXVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VSAT_U32_P: {
    unsigned Opc = RISCV::ESP_VSAT_U32;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(0);
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &QX = MI.getOperand(2);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vsat_u32 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(3);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vsat_u32 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RS1.getReg())
        .addReg(RS2.getReg())
        .addReg(RISCV::Q0 + QXVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VSAT_U8_P: {
    unsigned Opc = RISCV::ESP_VSAT_U8;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(0);
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &QX = MI.getOperand(2);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vsat_u8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(3);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vsat_u8 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RS1.getReg())
        .addReg(RS2.getReg())
        .addReg(RISCV::Q0 + QXVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VSSUBS_S16_P: {
    unsigned Opc = RISCV::ESP_VSSUBS_S16;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(0);
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vssubs_s16 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QV = MI.getOperand(2);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_vssubs_s16 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(RS1.getReg())
        .addReg(RISCV::Q0 + QXVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VSSUBS_S8_P: {
    unsigned Opc = RISCV::ESP_VSSUBS_S8;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(0);
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vssubs_s8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QV = MI.getOperand(2);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_vssubs_s8 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(RS1.getReg())
        .addReg(RISCV::Q0 + QXVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VSSUBS_U16_P: {
    unsigned Opc = RISCV::ESP_VSSUBS_U16;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(0);
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vssubs_u16 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QV = MI.getOperand(2);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_vssubs_u16 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(RS1.getReg())
        .addReg(RISCV::Q0 + QXVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VSSUBS_U8_P: {
    unsigned Opc = RISCV::ESP_VSSUBS_U8;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(0);
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vssubs_u8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QV = MI.getOperand(2);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_vssubs_u8 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(RS1.getReg())
        .addReg(RISCV::Q0 + QXVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VSUB_S16_P: {
    unsigned Opc = RISCV::ESP_VSUB_S16;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vsub_s16 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vsub_s16 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QV = MI.getOperand(2);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_vsub_s16 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VSUB_S16_LD_INCP_P: {
    unsigned Opc = RISCV::ESP_VSUB_S16_LD_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vsub_s16_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vsub_s16_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &QV = MI.getOperand(4);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_vsub_s16_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vsub_s16_ld_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VSUB_S16_ST_INCP_P: {
    unsigned Opc = RISCV::ESP_VSUB_S16_ST_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vsub_s16_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vsub_s16_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vsub_s16_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &QV = MI.getOperand(5);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_vsub_s16_st_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VSUB_S32_P: {
    unsigned Opc = RISCV::ESP_VSUB_S32;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vsub_s32 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vsub_s32 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QV = MI.getOperand(2);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_vsub_s32 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VSUB_S32_LD_INCP_P: {
    unsigned Opc = RISCV::ESP_VSUB_S32_LD_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vsub_s32_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vsub_s32_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &QV = MI.getOperand(4);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_vsub_s32_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vsub_s32_ld_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VSUB_S32_ST_INCP_P: {
    unsigned Opc = RISCV::ESP_VSUB_S32_ST_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vsub_s32_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vsub_s32_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vsub_s32_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &QV = MI.getOperand(5);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_vsub_s32_st_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VSUB_S8_P: {
    unsigned Opc = RISCV::ESP_VSUB_S8;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vsub_s8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vsub_s8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QV = MI.getOperand(2);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_vsub_s8 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VSUB_S8_LD_INCP_P: {
    unsigned Opc = RISCV::ESP_VSUB_S8_LD_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vsub_s8_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vsub_s8_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &QV = MI.getOperand(4);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_vsub_s8_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vsub_s8_ld_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VSUB_S8_ST_INCP_P: {
    unsigned Opc = RISCV::ESP_VSUB_S8_ST_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vsub_s8_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vsub_s8_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vsub_s8_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &QV = MI.getOperand(5);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_vsub_s8_st_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VSUB_U16_P: {
    unsigned Opc = RISCV::ESP_VSUB_U16;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vsub_u16 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vsub_u16 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QV = MI.getOperand(2);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_vsub_u16 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VSUB_U16_LD_INCP_P: {
    unsigned Opc = RISCV::ESP_VSUB_U16_LD_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vsub_u16_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vsub_u16_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &QV = MI.getOperand(4);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_vsub_u16_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vsub_u16_ld_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VSUB_U16_ST_INCP_P: {
    unsigned Opc = RISCV::ESP_VSUB_U16_ST_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vsub_u16_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vsub_u16_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vsub_u16_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &QV = MI.getOperand(5);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_vsub_u16_st_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VSUB_U32_P: {
    unsigned Opc = RISCV::ESP_VSUB_U32;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vsub_u32 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vsub_u32 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QV = MI.getOperand(2);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_vsub_u32 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VSUB_U32_LD_INCP_P: {
    unsigned Opc = RISCV::ESP_VSUB_U32_LD_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vsub_u32_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vsub_u32_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &QV = MI.getOperand(4);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_vsub_u32_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vsub_u32_ld_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VSUB_U32_ST_INCP_P: {
    unsigned Opc = RISCV::ESP_VSUB_U32_ST_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vsub_u32_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vsub_u32_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vsub_u32_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &QV = MI.getOperand(5);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_vsub_u32_st_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VSUB_U8_P: {
    unsigned Opc = RISCV::ESP_VSUB_U8;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vsub_u8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vsub_u8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QV = MI.getOperand(2);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_vsub_u8 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VSUB_U8_LD_INCP_P: {
    unsigned Opc = RISCV::ESP_VSUB_U8_LD_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vsub_u8_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vsub_u8_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &QV = MI.getOperand(4);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_vsub_u8_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vsub_u8_ld_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VSUB_U8_ST_INCP_P: {
    unsigned Opc = RISCV::ESP_VSUB_U8_ST_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vsub_u8_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vsub_u8_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vsub_u8_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &QV = MI.getOperand(5);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_vsub_u8_st_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_ADDX2_P: {
    unsigned Opc = RISCV::ESP_ADDX2;
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
  case RISCV::ESP_ADDX4_P: {
    unsigned Opc = RISCV::ESP_ADDX4;
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
  case RISCV::ESP_SAT_P: {
    unsigned Opc = RISCV::ESP_SAT;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS0 = MI.getOperand(1);
    MachineOperand &RS1 = MI.getOperand(2);
    MachineOperand &RSD = MI.getOperand(3);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS0.getReg())
        .addReg(RS1.getReg())
        .addReg(RSD.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_SUBX2_P: {
    unsigned Opc = RISCV::ESP_SUBX2;
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
  case RISCV::ESP_SUBX4_P: {
    unsigned Opc = RISCV::ESP_SUBX4;
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
  case RISCV::ESP_ANDQ_P: {
    unsigned Opc = RISCV::ESP_ANDQ;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_andq first argument, it must "
                        "bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_andq first argument, it must "
                        "bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(2);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_andq first argument, it must "
                        "bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_NOTQ_P: {
    unsigned Opc = RISCV::ESP_NOTQ;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_notq first argument, it must "
                        "bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(1);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_notq first argument, it must "
                        "bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_ORQ_P: {
    unsigned Opc = RISCV::ESP_ORQ;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_orq first argument, it must "
                        "bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_orq first argument, it must "
                        "bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(2);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_orq first argument, it must "
                        "bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_XORQ_P: {
    unsigned Opc = RISCV::ESP_XORQ;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_xorq first argument, it must "
                        "bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_xorq first argument, it must "
                        "bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(2);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_xorq first argument, it must "
                        "bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VCMP_EQ_S16_P: {
    unsigned Opc = RISCV::ESP_VCMP_EQ_S16;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vcmp_eq_s16 first argument, "
                        "it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vcmp_eq_s16 first argument, "
                        "it must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(2);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vcmp_eq_s16 first argument, "
                        "it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VCMP_EQ_S32_P: {
    unsigned Opc = RISCV::ESP_VCMP_EQ_S32;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vcmp_eq_s32 first argument, "
                        "it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vcmp_eq_s32 first argument, "
                        "it must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(2);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vcmp_eq_s32 first argument, "
                        "it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VCMP_EQ_S8_P: {
    unsigned Opc = RISCV::ESP_VCMP_EQ_S8;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vcmp_eq_s8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vcmp_eq_s8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(2);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vcmp_eq_s8 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

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
  case RISCV::ESP_VCMP_GT_S16_P: {
    unsigned Opc = RISCV::ESP_VCMP_GT_S16;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vcmp_gt_s16 first argument, "
                        "it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vcmp_gt_s16 first argument, "
                        "it must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(2);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vcmp_gt_s16 first argument, "
                        "it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VCMP_GT_S32_P: {
    unsigned Opc = RISCV::ESP_VCMP_GT_S32;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vcmp_gt_s32 first argument, "
                        "it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vcmp_gt_s32 first argument, "
                        "it must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(2);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vcmp_gt_s32 first argument, "
                        "it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VCMP_GT_S8_P: {
    unsigned Opc = RISCV::ESP_VCMP_GT_S8;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vcmp_gt_s8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vcmp_gt_s8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(2);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vcmp_gt_s8 first argument, it "
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
  case RISCV::ESP_VCMP_LT_S16_P: {
    unsigned Opc = RISCV::ESP_VCMP_LT_S16;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vcmp_lt_s16 first argument, "
                        "it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vcmp_lt_s16 first argument, "
                        "it must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(2);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vcmp_lt_s16 first argument, "
                        "it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VCMP_LT_S32_P: {
    unsigned Opc = RISCV::ESP_VCMP_LT_S32;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vcmp_lt_s32 first argument, "
                        "it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vcmp_lt_s32 first argument, "
                        "it must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(2);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vcmp_lt_s32 first argument, "
                        "it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VCMP_LT_S8_P: {
    unsigned Opc = RISCV::ESP_VCMP_LT_S8;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vcmp_lt_s8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vcmp_lt_s8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(2);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vcmp_lt_s8 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VCMP_LT_U16_P: {
    unsigned Opc = RISCV::ESP_VCMP_LT_U16;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vcmp_lt_u16 first argument, "
                        "it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vcmp_lt_u16 first argument, "
                        "it must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(2);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vcmp_lt_u16 first argument, "
                        "it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VCMP_LT_U32_P: {
    unsigned Opc = RISCV::ESP_VCMP_LT_U32;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vcmp_lt_u32 first argument, "
                        "it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vcmp_lt_u32 first argument, "
                        "it must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(2);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vcmp_lt_u32 first argument, "
                        "it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VCMP_LT_U8_P: {
    unsigned Opc = RISCV::ESP_VCMP_LT_U8;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vcmp_lt_u8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vcmp_lt_u8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(2);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vcmp_lt_u8 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_MOV_S16_QACC_P: {
    unsigned Opc = RISCV::ESP_MOV_S16_QACC;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_mov_s16_qacc first argument, "
                        "it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc)).addReg(RISCV::Q0 + QUVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_MOV_S8_QACC_P: {
    unsigned Opc = RISCV::ESP_MOV_S8_QACC;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_mov_s8_qacc first argument, "
                        "it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc)).addReg(RISCV::Q0 + QUVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_MOV_U16_QACC_P: {
    unsigned Opc = RISCV::ESP_MOV_U16_QACC;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_mov_u16_qacc first argument, "
                        "it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc)).addReg(RISCV::Q0 + QUVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_MOV_U8_QACC_P: {
    unsigned Opc = RISCV::ESP_MOV_U8_QACC;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QU = MI.getOperand(0);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_mov_u8_qacc first argument, "
                        "it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc)).addReg(RISCV::Q0 + QUVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_MOVI_16_A_P: {
    unsigned Opc = RISCV::ESP_MOVI_16_A;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QY = MI.getOperand(0);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_movi_16_a first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &SELECT_16 = MI.getOperand(1);
    const TargetRegisterClass *RC = &RISCV::GPRPIERegClass;
    unsigned R1 = MRI.createVirtualRegister(RC);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Define)
        .addReg(RISCV::Q0 + QYVal)
        .addImm(SELECT_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_MOVI_16_Q_P: {
    unsigned Opc = RISCV::ESP_MOVI_16_Q;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(0);
    MachineOperand &SELECT_16 = MI.getOperand(1);
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_movi_16_q first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QYVal, RegState::Define)
        .addReg(RS1.getReg())
        .addImm(SELECT_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_MOVI_32_A_P: {
    unsigned Opc = RISCV::ESP_MOVI_32_A;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QY = MI.getOperand(0);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_movi_32_a first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &SELECT_4 = MI.getOperand(1);
    const TargetRegisterClass *RC = &RISCV::GPRPIERegClass;
    unsigned R1 = MRI.createVirtualRegister(RC);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Define)
        .addReg(RISCV::Q0 + QYVal)
        .addImm(SELECT_4.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_MOVI_32_Q_P: {
    unsigned Opc = RISCV::ESP_MOVI_32_Q;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(0);
    MachineOperand &SELECT_4 = MI.getOperand(1);
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_movi_32_q first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QYVal, RegState::Define)
        .addReg(RS1.getReg())
        .addImm(SELECT_4.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_MOVI_8_A_P: {
    unsigned Opc = RISCV::ESP_MOVI_8_A;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QY = MI.getOperand(0);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_movi_8_a first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &SELECT_16 = MI.getOperand(1);
    const TargetRegisterClass *RC = &RISCV::GPRPIERegClass;
    unsigned R1 = MRI.createVirtualRegister(RC);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Define)
        .addReg(RISCV::Q0 + QYVal)
        .addImm(SELECT_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_MOVI_8_Q_P: {
    unsigned Opc = RISCV::ESP_MOVI_8_Q;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(0);
    MachineOperand &SELECT_16 = MI.getOperand(1);
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_movi_8_q first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QYVal, RegState::Define)
        .addReg(RS1.getReg())
        .addImm(SELECT_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_MOVX_R_CFG_P: {
    unsigned Opc = RISCV::ESP_MOVX_R_CFG;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(0);
    BuildMI(*MBB, MI, DL, TII.get(Opc)).addReg(RS1.getReg(), RegState::Define);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_MOVX_R_FFT_BIT_WIDTH_P: {
    unsigned Opc = RISCV::ESP_MOVX_R_FFT_BIT_WIDTH;
    MachineBasicBlock *MBB = MI.getParent();
    const TargetRegisterClass *RC = &RISCV::GPRPIERegClass;
    unsigned R1 = MRI.createVirtualRegister(RC);
    BuildMI(*MBB, MI, DL, TII.get(Opc)).addReg(R1, RegState::Define);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_MOVX_R_PERF_P: {
    unsigned Opc = RISCV::ESP_MOVX_R_PERF;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(0);
    const TargetRegisterClass *RC = &RISCV::GPRPIERegClass;
    unsigned R1 = MRI.createVirtualRegister(RC);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Define)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_MOVX_R_SAR_P: {
    unsigned Opc = RISCV::ESP_MOVX_R_SAR;
    MachineBasicBlock *MBB = MI.getParent();
    const TargetRegisterClass *RC = &RISCV::GPRPIERegClass;
    unsigned R1 = MRI.createVirtualRegister(RC);
    BuildMI(*MBB, MI, DL, TII.get(Opc)).addReg(R1, RegState::Define);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_MOVX_R_SAR_BYTES_P: {
    unsigned Opc = RISCV::ESP_MOVX_R_SAR_BYTES;
    MachineBasicBlock *MBB = MI.getParent();
    const TargetRegisterClass *RC = &RISCV::GPRPIERegClass;
    unsigned R1 = MRI.createVirtualRegister(RC);
    BuildMI(*MBB, MI, DL, TII.get(Opc)).addReg(R1, RegState::Define);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_MOVX_R_XACC_H_P: {
    unsigned Opc = RISCV::ESP_MOVX_R_XACC_H;
    MachineBasicBlock *MBB = MI.getParent();
    const TargetRegisterClass *RC = &RISCV::GPRPIERegClass;
    unsigned R1 = MRI.createVirtualRegister(RC);
    BuildMI(*MBB, MI, DL, TII.get(Opc)).addReg(R1, RegState::Define);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_MOVX_R_XACC_L_P: {
    unsigned Opc = RISCV::ESP_MOVX_R_XACC_L;
    MachineBasicBlock *MBB = MI.getParent();
    const TargetRegisterClass *RC = &RISCV::GPRPIERegClass;
    unsigned R1 = MRI.createVirtualRegister(RC);
    BuildMI(*MBB, MI, DL, TII.get(Opc)).addReg(R1, RegState::Define);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_MOVX_W_CFG_P: {
    unsigned Opc = RISCV::ESP_MOVX_W_CFG;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(0);
    BuildMI(*MBB, MI, DL, TII.get(Opc)).addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_MOVX_W_FFT_BIT_WIDTH_P: {
    unsigned Opc = RISCV::ESP_MOVX_W_FFT_BIT_WIDTH;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(0);
    BuildMI(*MBB, MI, DL, TII.get(Opc)).addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_MOVX_W_PERF_P: {
    unsigned Opc = RISCV::ESP_MOVX_W_PERF;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(0);
    BuildMI(*MBB, MI, DL, TII.get(Opc)).addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_MOVX_W_SAR_P: {
    unsigned Opc = RISCV::ESP_MOVX_W_SAR;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(0);
    BuildMI(*MBB, MI, DL, TII.get(Opc)).addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_MOVX_W_SAR_BYTES_P: {
    unsigned Opc = RISCV::ESP_MOVX_W_SAR_BYTES;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(0);
    BuildMI(*MBB, MI, DL, TII.get(Opc)).addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_MOVX_W_XACC_H_P: {
    unsigned Opc = RISCV::ESP_MOVX_W_XACC_H;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(0);
    BuildMI(*MBB, MI, DL, TII.get(Opc)).addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_MOVX_W_XACC_L_P: {
    unsigned Opc = RISCV::ESP_MOVX_W_XACC_L;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(0);
    BuildMI(*MBB, MI, DL, TII.get(Opc)).addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VEXT_S16_P: {
    unsigned Opc = RISCV::ESP_VEXT_S16;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QW = MI.getOperand(0);
    unsigned QWVal = QW.getImm();
    assert(QWVal < 8 && "Unexpected value of esp_vext_s16 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(1);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vext_s16 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QV = MI.getOperand(2);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_vext_s16 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(RISCV::Q0 + QWVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VEXT_S8_P: {
    unsigned Opc = RISCV::ESP_VEXT_S8;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QW = MI.getOperand(0);
    unsigned QWVal = QW.getImm();
    assert(QWVal < 8 && "Unexpected value of esp_vext_s8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(1);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vext_s8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QV = MI.getOperand(2);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_vext_s8 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(RISCV::Q0 + QWVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VEXT_U16_P: {
    unsigned Opc = RISCV::ESP_VEXT_U16;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QW = MI.getOperand(0);
    unsigned QWVal = QW.getImm();
    assert(QWVal < 8 && "Unexpected value of esp_vext_u16 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(1);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vext_u16 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QV = MI.getOperand(2);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_vext_u16 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(RISCV::Q0 + QWVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VEXT_U8_P: {
    unsigned Opc = RISCV::ESP_VEXT_U8;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QW = MI.getOperand(0);
    unsigned QWVal = QW.getImm();
    assert(QWVal < 8 && "Unexpected value of esp_vext_u8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(1);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_vext_u8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QV = MI.getOperand(2);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_vext_u8 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(RISCV::Q0 + QWVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VUNZIP_16_P: {
    unsigned Opc = RISCV::ESP_VUNZIP_16;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vunzip_16 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vunzip_16 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QXVal, RegState::Define)
        .addReg(RISCV::Q0 + QYVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VUNZIP_32_P: {
    unsigned Opc = RISCV::ESP_VUNZIP_32;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vunzip_32 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vunzip_32 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QXVal, RegState::Define)
        .addReg(RISCV::Q0 + QYVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VUNZIP_8_P: {
    unsigned Opc = RISCV::ESP_VUNZIP_8;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vunzip_8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vunzip_8 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QXVal, RegState::Define)
        .addReg(RISCV::Q0 + QYVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VUNZIPT_16_P: {
    unsigned Opc = RISCV::ESP_VUNZIPT_16;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vunzipt_16 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vunzipt_16 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QW = MI.getOperand(2);
    unsigned QWVal = QW.getImm();
    assert(QWVal < 8 && "Unexpected value of esp_vunzipt_16 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QXVal, RegState::Define)
        .addReg(RISCV::Q0 + QYVal, RegState::Define)
        .addReg(RISCV::Q0 + QWVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QWVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VUNZIPT_8_P: {
    unsigned Opc = RISCV::ESP_VUNZIPT_8;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vunzipt_8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vunzipt_8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QW = MI.getOperand(2);
    unsigned QWVal = QW.getImm();
    assert(QWVal < 8 && "Unexpected value of esp_vunzipt_8 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QXVal, RegState::Define)
        .addReg(RISCV::Q0 + QYVal, RegState::Define)
        .addReg(RISCV::Q0 + QWVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QWVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VZIP_16_P: {
    unsigned Opc = RISCV::ESP_VZIP_16;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vzip_16 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vzip_16 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QXVal, RegState::Define)
        .addReg(RISCV::Q0 + QYVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VZIP_32_P: {
    unsigned Opc = RISCV::ESP_VZIP_32;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vzip_32 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vzip_32 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QXVal, RegState::Define)
        .addReg(RISCV::Q0 + QYVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VZIP_8_P: {
    unsigned Opc = RISCV::ESP_VZIP_8;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vzip_8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vzip_8 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QXVal, RegState::Define)
        .addReg(RISCV::Q0 + QYVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VZIPT_16_P: {
    unsigned Opc = RISCV::ESP_VZIPT_16;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vzipt_16 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vzipt_16 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QW = MI.getOperand(2);
    unsigned QWVal = QW.getImm();
    assert(QWVal < 8 && "Unexpected value of esp_vzipt_16 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QXVal, RegState::Define)
        .addReg(RISCV::Q0 + QYVal, RegState::Define)
        .addReg(RISCV::Q0 + QWVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QWVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VZIPT_8_P: {
    unsigned Opc = RISCV::ESP_VZIPT_8;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_vzipt_8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vzipt_8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QW = MI.getOperand(2);
    unsigned QWVal = QW.getImm();
    assert(QWVal < 8 && "Unexpected value of esp_vzipt_8 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QXVal, RegState::Define)
        .addReg(RISCV::Q0 + QYVal, RegState::Define)
        .addReg(RISCV::Q0 + QWVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QWVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_ZERO_Q_P: {
    unsigned Opc = RISCV::ESP_ZERO_Q;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QZ = MI.getOperand(0);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_zero_q first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_ZERO_QACC_P: {
    unsigned Opc = RISCV::ESP_ZERO_QACC;
    MachineBasicBlock *MBB = MI.getParent();
    BuildMI(*MBB, MI, DL, TII.get(Opc));

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_ZERO_XACC_P: {
    unsigned Opc = RISCV::ESP_ZERO_XACC;
    MachineBasicBlock *MBB = MI.getParent();
    BuildMI(*MBB, MI, DL, TII.get(Opc));

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_FFT_AMS_S16_LD_INCP_P: {
    unsigned Opc = RISCV::ESP_FFT_AMS_S16_LD_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_fft_ams_s16_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_fft_ams_s16_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QW = MI.getOperand(3);
    unsigned QWVal = QW.getImm();
    assert(QWVal < 8 && "Unexpected value of esp_fft_ams_s16_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &SELECT_2 = MI.getOperand(5);
    MachineOperand &QU = MI.getOperand(6);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_fft_ams_s16_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(7);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_fft_ams_s16_ld_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QV = MI.getOperand(8);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_fft_ams_s16_ld_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QWVal)
        .addReg(RS1.getReg())
        .addImm(SELECT_2.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_FFT_AMS_S16_LD_INCP_UAUP_P: {
    unsigned Opc = RISCV::ESP_FFT_AMS_S16_LD_INCP_UAUP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_fft_ams_s16_ld_incp_uaup "
                        "first argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_fft_ams_s16_ld_incp_uaup "
                        "first argument, it must bi in range [0,7]");
    MachineOperand &QW = MI.getOperand(3);
    unsigned QWVal = QW.getImm();
    assert(QWVal < 8 && "Unexpected value of esp_fft_ams_s16_ld_incp_uaup "
                        "first argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &SELECT_2 = MI.getOperand(5);
    MachineOperand &QU = MI.getOperand(6);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_fft_ams_s16_ld_incp_uaup "
                        "first argument, it must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(7);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_fft_ams_s16_ld_incp_uaup "
                        "first argument, it must bi in range [0,7]");
    MachineOperand &QV = MI.getOperand(8);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_fft_ams_s16_ld_incp_uaup "
                        "first argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QWVal)
        .addReg(RS1.getReg())
        .addImm(SELECT_2.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_FFT_AMS_S16_LD_R32_DECP_P: {
    unsigned Opc = RISCV::ESP_FFT_AMS_S16_LD_R32_DECP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_fft_ams_s16_ld_r32_decp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_fft_ams_s16_ld_r32_decp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QW = MI.getOperand(3);
    unsigned QWVal = QW.getImm();
    assert(QWVal < 8 && "Unexpected value of esp_fft_ams_s16_ld_r32_decp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &SELECT_2 = MI.getOperand(5);
    MachineOperand &QU = MI.getOperand(6);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_fft_ams_s16_ld_r32_decp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(7);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_fft_ams_s16_ld_r32_decp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QV = MI.getOperand(8);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_fft_ams_s16_ld_r32_decp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QWVal)
        .addReg(RS1.getReg())
        .addImm(SELECT_2.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_FFT_AMS_S16_ST_INCP_P: {
    unsigned Opc = RISCV::ESP_FFT_AMS_S16_ST_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_fft_ams_s16_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_fft_ams_s16_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QW = MI.getOperand(2);
    unsigned QWVal = QW.getImm();
    assert(QWVal < 8 && "Unexpected value of esp_fft_ams_s16_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_fft_ams_s16_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &RS2 = MI.getOperand(5);
    MachineOperand &SELECT_2 = MI.getOperand(6);
    MachineOperand &QZ = MI.getOperand(7);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_fft_ams_s16_st_incp first "
                        "argument, it must bi in range [0,7]");
    const TargetRegisterClass *RC = &RISCV::GPRPIERegClass;
    unsigned R1 = MRI.createVirtualRegister(RC);
    unsigned R2 = MRI.createVirtualRegister(RC);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(R1, RegState::Define)
        .addReg(R2, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QWVal)
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg())
        .addReg(RS2.getReg())
        .addImm(SELECT_2.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_FFT_BITREV_P: {
    unsigned Opc = RISCV::ESP_FFT_BITREV;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(1);
    MachineOperand &QV = MI.getOperand(2);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_fft_bitrev first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(RS1.getReg())
        .addReg(RISCV::Q0 + QVVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_FFT_CMUL_S16_LD_XP_P: {
    unsigned Opc = RISCV::ESP_FFT_CMUL_S16_LD_XP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &QX = MI.getOperand(2);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_fft_cmul_s16_ld_xp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(3);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_fft_cmul_s16_ld_xp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(4);
    MachineOperand &SELECT_8 = MI.getOperand(5);
    MachineOperand &QZ = MI.getOperand(6);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_fft_cmul_s16_ld_xp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(7);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_fft_cmul_s16_ld_xp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS2.getReg())
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg())
        .addImm(SELECT_8.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_FFT_CMUL_S16_ST_XP_P: {
    unsigned Opc = RISCV::ESP_FFT_CMUL_S16_ST_XP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &QX = MI.getOperand(2);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_fft_cmul_s16_st_xp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(3);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_fft_cmul_s16_st_xp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(4);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_fft_cmul_s16_st_xp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(5);
    MachineOperand &SELECT_4 = MI.getOperand(6);
    MachineOperand &UPD_4 = MI.getOperand(7);
    MachineOperand &SELECT_8 = MI.getOperand(8);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS2.getReg())
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg())
        .addImm(SELECT_4.getImm())
        .addImm(UPD_4.getImm())
        .addImm(SELECT_8.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_FFT_R2BF_S16_P: {
    unsigned Opc = RISCV::ESP_FFT_R2BF_S16;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(0);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_fft_r2bf_s16 first argument, "
                        "it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_fft_r2bf_s16 first argument, "
                        "it must bi in range [0,7]");
    MachineOperand &SELECT_2 = MI.getOperand(2);
    MachineOperand &QZ = MI.getOperand(3);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_fft_r2bf_s16 first argument, "
                        "it must bi in range [0,7]");
    MachineOperand &QV = MI.getOperand(4);
    unsigned QVVal = QV.getImm();
    assert(QVVal < 8 && "Unexpected value of esp_fft_r2bf_s16 first argument, "
                        "it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QVVal, RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addImm(SELECT_2.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_FFT_R2BF_S16_ST_INCP_P: {
    unsigned Opc = RISCV::ESP_FFT_R2BF_S16_ST_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QX = MI.getOperand(1);
    unsigned QXVal = QX.getImm();
    assert(QXVal < 8 && "Unexpected value of esp_fft_r2bf_s16_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_fft_r2bf_s16_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &SELECT_4 = MI.getOperand(4);
    MachineOperand &QZ = MI.getOperand(5);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_fft_r2bf_s16_st_incp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QXVal)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg())
        .addImm(SELECT_4.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_FFT_VST_R32_DECP_P: {
    unsigned Opc = RISCV::ESP_FFT_VST_R32_DECP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QU = MI.getOperand(1);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_fft_vst_r32_decp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(2);
    MachineOperand &SELECT_2 = MI.getOperand(3);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg())
        .addImm(SELECT_2.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_LD_128_USAR_IP_P: {
    unsigned Opc = RISCV::ESP_LD_128_USAR_IP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(1);
    MachineOperand &OFFSET_256_16 = MI.getOperand(2);
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_ld_128_usar_ip first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS1.getReg())
        .addImm(OFFSET_256_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_LD_128_USAR_XP_P: {
    unsigned Opc = RISCV::ESP_LD_128_USAR_XP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &RS1 = MI.getOperand(2);
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_ld_128_usar_xp first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS2.getReg())
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_LD_XACC_IP_P: {
    unsigned Opc = RISCV::ESP_LD_XACC_IP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(1);
    MachineOperand &OFFSET_256_8 = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS1.getReg())
        .addImm(OFFSET_256_8.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_LDQA_S16_128_IP_P: {
    unsigned Opc = RISCV::ESP_LDQA_S16_128_IP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(1);
    MachineOperand &OFFSET_256_16 = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS1.getReg())
        .addImm(OFFSET_256_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_LDQA_S16_128_XP_P: {
    unsigned Opc = RISCV::ESP_LDQA_S16_128_XP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &RS1 = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS2.getReg())
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_LDQA_S8_128_IP_P: {
    unsigned Opc = RISCV::ESP_LDQA_S8_128_IP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(1);
    MachineOperand &OFFSET_256_16 = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS1.getReg())
        .addImm(OFFSET_256_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_LDQA_S8_128_XP_P: {
    unsigned Opc = RISCV::ESP_LDQA_S8_128_XP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &RS1 = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS2.getReg())
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_LDQA_U16_128_IP_P: {
    unsigned Opc = RISCV::ESP_LDQA_U16_128_IP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(1);
    MachineOperand &OFFSET_256_16 = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS1.getReg())
        .addImm(OFFSET_256_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_LDQA_U16_128_XP_P: {
    unsigned Opc = RISCV::ESP_LDQA_U16_128_XP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &RS1 = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS2.getReg())
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_LDQA_U8_128_IP_P: {
    unsigned Opc = RISCV::ESP_LDQA_U8_128_IP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(1);
    MachineOperand &OFFSET_256_16 = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS1.getReg())
        .addImm(OFFSET_256_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_LDQA_U8_128_XP_P: {
    unsigned Opc = RISCV::ESP_LDQA_U8_128_XP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &RS1 = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS2.getReg())
        .addReg(RS1.getReg());

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
  case RISCV::ESP_LD_QACC_H_H_128_IP_P: {
    unsigned Opc = RISCV::ESP_LD_QACC_H_H_128_IP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(1);
    MachineOperand &OFFSET_256_16 = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS1.getReg())
        .addImm(OFFSET_256_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_LD_QACC_H_L_128_IP_P: {
    unsigned Opc = RISCV::ESP_LD_QACC_H_L_128_IP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(1);
    MachineOperand &OFFSET_256_16 = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS1.getReg())
        .addImm(OFFSET_256_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_LD_QACC_L_H_128_IP_P: {
    unsigned Opc = RISCV::ESP_LD_QACC_L_H_128_IP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(1);
    MachineOperand &OFFSET_256_16 = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS1.getReg())
        .addImm(OFFSET_256_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_LD_QACC_L_L_128_IP_P: {
    unsigned Opc = RISCV::ESP_LD_QACC_L_L_128_IP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(1);
    MachineOperand &OFFSET_256_16 = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS1.getReg())
        .addImm(OFFSET_256_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_LD_UA_STATE_IP_P: {
    unsigned Opc = RISCV::ESP_LD_UA_STATE_IP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(1);
    MachineOperand &OFFSET_256_16 = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS1.getReg())
        .addImm(OFFSET_256_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_LDXQ_32_P: {
    unsigned Opc = RISCV::ESP_LDXQ_32;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(0);
    MachineOperand &QW = MI.getOperand(1);
    unsigned QWVal = QW.getImm();
    assert(QWVal < 8 && "Unexpected value of esp_ldxq_32 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &SELECT_4 = MI.getOperand(2);
    MachineOperand &SELECT_8 = MI.getOperand(3);
    MachineOperand &QU = MI.getOperand(4);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_ldxq_32 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(RS1.getReg())
        .addReg(RISCV::Q0 + QWVal)
        .addImm(SELECT_4.getImm())
        .addImm(SELECT_8.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_ST_QACC_H_H_128_IP_P: {
    unsigned Opc = RISCV::ESP_ST_QACC_H_H_128_IP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(1);
    MachineOperand &OFFSET_256_16 = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS1.getReg())
        .addImm(OFFSET_256_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_ST_QACC_H_L_128_IP_P: {
    unsigned Opc = RISCV::ESP_ST_QACC_H_L_128_IP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(1);
    MachineOperand &OFFSET_256_16 = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS1.getReg())
        .addImm(OFFSET_256_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_ST_QACC_L_H_128_IP_P: {
    unsigned Opc = RISCV::ESP_ST_QACC_L_H_128_IP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(1);
    MachineOperand &OFFSET_256_16 = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS1.getReg())
        .addImm(OFFSET_256_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_ST_QACC_L_L_128_IP_P: {
    unsigned Opc = RISCV::ESP_ST_QACC_L_L_128_IP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(1);
    MachineOperand &OFFSET_256_16 = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS1.getReg())
        .addImm(OFFSET_256_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_ST_UA_STATE_IP_P: {
    unsigned Opc = RISCV::ESP_ST_UA_STATE_IP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(1);
    MachineOperand &OFFSET_256_16 = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS1.getReg())
        .addImm(OFFSET_256_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_STXQ_32_P: {
    unsigned Opc = RISCV::ESP_STXQ_32;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(0);
    MachineOperand &QW = MI.getOperand(1);
    unsigned QWVal = QW.getImm();
    assert(QWVal < 8 && "Unexpected value of esp_stxq_32 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(2);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_stxq_32 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &SELECT_4 = MI.getOperand(3);
    MachineOperand &SELECT_8 = MI.getOperand(4);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RS1.getReg())
        .addReg(RISCV::Q0 + QWVal)
        .addReg(RISCV::Q0 + QUVal)
        .addImm(SELECT_4.getImm())
        .addImm(SELECT_8.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VLD_128_IP_P: {
    unsigned Opc = RISCV::ESP_VLD_128_IP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(1);
    MachineOperand &OFFSET_256_16 = MI.getOperand(2);
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vld_128_ip first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS1.getReg())
        .addImm(OFFSET_256_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VLD_128_XP_P: {
    unsigned Opc = RISCV::ESP_VLD_128_XP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &RS1 = MI.getOperand(2);
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vld_128_xp first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS2.getReg())
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VLD_H_64_IP_P: {
    unsigned Opc = RISCV::ESP_VLD_H_64_IP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(1);
    MachineOperand &OFFSET_256_8 = MI.getOperand(2);
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vld_h_64_ip first argument, "
                        "it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS1.getReg())
        .addImm(OFFSET_256_8.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VLD_H_64_XP_P: {
    unsigned Opc = RISCV::ESP_VLD_H_64_XP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &RS1 = MI.getOperand(2);
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vld_h_64_xp first argument, "
                        "it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS2.getReg())
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VLD_L_64_IP_P: {
    unsigned Opc = RISCV::ESP_VLD_L_64_IP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(1);
    MachineOperand &OFFSET_256_8 = MI.getOperand(2);
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vld_l_64_ip first argument, "
                        "it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS1.getReg())
        .addImm(OFFSET_256_8.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VLD_L_64_XP_P: {
    unsigned Opc = RISCV::ESP_VLD_L_64_XP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &RS1 = MI.getOperand(2);
    MachineOperand &QU = MI.getOperand(3);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vld_l_64_xp first argument, "
                        "it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS2.getReg())
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VST_128_IP_P: {
    unsigned Opc = RISCV::ESP_VST_128_IP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QU = MI.getOperand(1);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vst_128_ip first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(2);
    MachineOperand &OFFSET_256_16 = MI.getOperand(3);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg())
        .addImm(OFFSET_256_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VST_128_XP_P: {
    unsigned Opc = RISCV::ESP_VST_128_XP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &QU = MI.getOperand(2);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vst_128_xp first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS2.getReg())
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VST_H_64_IP_P: {
    unsigned Opc = RISCV::ESP_VST_H_64_IP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QU = MI.getOperand(1);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vst_h_64_ip first argument, "
                        "it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(2);
    MachineOperand &OFFSET_256_8 = MI.getOperand(3);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg())
        .addImm(OFFSET_256_8.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VST_H_64_XP_P: {
    unsigned Opc = RISCV::ESP_VST_H_64_XP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &QU = MI.getOperand(2);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vst_h_64_xp first argument, "
                        "it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS2.getReg())
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VST_L_64_IP_P: {
    unsigned Opc = RISCV::ESP_VST_L_64_IP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QU = MI.getOperand(1);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vst_l_64_ip first argument, "
                        "it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(2);
    MachineOperand &OFFSET_256_8 = MI.getOperand(3);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg())
        .addImm(OFFSET_256_8.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VST_L_64_XP_P: {
    unsigned Opc = RISCV::ESP_VST_L_64_XP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &QU = MI.getOperand(2);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vst_l_64_xp first argument, "
                        "it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS2.getReg())
        .addReg(RISCV::Q0 + QUVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_SLCI_2Q_P: {
    unsigned Opc = RISCV::ESP_SLCI_2Q;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QY = MI.getOperand(0);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_slci_2q first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QW = MI.getOperand(1);
    unsigned QWVal = QW.getImm();
    assert(QWVal < 8 && "Unexpected value of esp_slci_2q first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &SELECT_16 = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QYVal, RegState::Define)
        .addReg(RISCV::Q0 + QWVal, RegState::Define)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QWVal)
        .addImm(SELECT_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_SLCXXP_2Q_P: {
    unsigned Opc = RISCV::ESP_SLCXXP_2Q;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(0);
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_slcxxp_2q first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QW = MI.getOperand(3);
    unsigned QWVal = QW.getImm();
    assert(QWVal < 8 && "Unexpected value of esp_slcxxp_2q first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QYVal, RegState::Define)
        .addReg(RISCV::Q0 + QWVal, RegState::Define)
        .addReg(RS1.getReg())
        .addReg(RS2.getReg())
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QWVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_SRC_Q_P: {
    unsigned Opc = RISCV::ESP_SRC_Q;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QY = MI.getOperand(0);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_src_q first argument, it must "
                        "bi in range [0,7]");
    MachineOperand &QW = MI.getOperand(1);
    unsigned QWVal = QW.getImm();
    assert(QWVal < 8 && "Unexpected value of esp_src_q first argument, it must "
                        "bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(2);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_src_q first argument, it must "
                        "bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QWVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_SRC_Q_LD_IP_P: {
    unsigned Opc = RISCV::ESP_SRC_Q_LD_IP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_src_q_ld_ip first argument, "
                        "it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(2);
    MachineOperand &QW = MI.getOperand(3);
    unsigned QWVal = QW.getImm();
    assert(QWVal < 8 && "Unexpected value of esp_src_q_ld_ip first argument, "
                        "it must bi in range [0,7]");
    MachineOperand &OFFSET_256_16 = MI.getOperand(4);
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_src_q_ld_ip first argument, "
                        "it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QWVal, RegState::Define)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg())
        .addReg(RISCV::Q0 + QWVal)
        .addImm(OFFSET_256_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_SRC_Q_LD_XP_P: {
    unsigned Opc = RISCV::ESP_SRC_Q_LD_XP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_src_q_ld_xp first argument, "
                        "it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    MachineOperand &QW = MI.getOperand(4);
    unsigned QWVal = QW.getImm();
    assert(QWVal < 8 && "Unexpected value of esp_src_q_ld_xp first argument, "
                        "it must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(5);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_src_q_ld_xp first argument, "
                        "it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QWVal, RegState::Define)
        .addReg(RS2.getReg())
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RS1.getReg())
        .addReg(RISCV::Q0 + QWVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_SRC_Q_QUP_P: {
    unsigned Opc = RISCV::ESP_SRC_Q_QUP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QY = MI.getOperand(0);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_src_q_qup first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QW = MI.getOperand(1);
    unsigned QWVal = QW.getImm();
    assert(QWVal < 8 && "Unexpected value of esp_src_q_qup first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QZ = MI.getOperand(2);
    unsigned QZVal = QZ.getImm();
    assert(QZVal < 8 && "Unexpected value of esp_src_q_qup first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QZVal, RegState::Define)
        .addReg(RISCV::Q0 + QWVal, RegState::Define)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QWVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_SRCI_2Q_P: {
    unsigned Opc = RISCV::ESP_SRCI_2Q;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QY = MI.getOperand(0);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_srci_2q first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QW = MI.getOperand(1);
    unsigned QWVal = QW.getImm();
    assert(QWVal < 8 && "Unexpected value of esp_srci_2q first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &SELECT_16 = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QYVal, RegState::Define)
        .addReg(RISCV::Q0 + QWVal, RegState::Define)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QWVal)
        .addImm(SELECT_16.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_SRCMB_S16_Q_QACC_P: {
    unsigned Opc = RISCV::ESP_SRCMB_S16_Q_QACC;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QW = MI.getOperand(0);
    unsigned QWVal = QW.getImm();
    assert(QWVal < 8 && "Unexpected value of esp_srcmb_s16_q_qacc first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &SELECT_2 = MI.getOperand(1);
    MachineOperand &QU = MI.getOperand(2);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_srcmb_s16_q_qacc first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(RISCV::Q0 + QWVal)
        .addImm(SELECT_2.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_SRCMB_S16_QACC_P: {
    unsigned Opc = RISCV::ESP_SRCMB_S16_QACC;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(0);
    MachineOperand &SELECT_2 = MI.getOperand(1);
    MachineOperand &QU = MI.getOperand(2);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_srcmb_s16_qacc first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(RS1.getReg())
        .addImm(SELECT_2.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_SRCMB_S8_Q_QACC_P: {
    unsigned Opc = RISCV::ESP_SRCMB_S8_Q_QACC;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QW = MI.getOperand(0);
    unsigned QWVal = QW.getImm();
    assert(QWVal < 8 && "Unexpected value of esp_srcmb_s8_q_qacc first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &SELECT_2 = MI.getOperand(1);
    MachineOperand &QU = MI.getOperand(2);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_srcmb_s8_q_qacc first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(RISCV::Q0 + QWVal)
        .addImm(SELECT_2.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_SRCMB_S8_QACC_P: {
    unsigned Opc = RISCV::ESP_SRCMB_S8_QACC;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(0);
    MachineOperand &SELECT_2 = MI.getOperand(1);
    MachineOperand &QU = MI.getOperand(2);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_srcmb_s8_qacc first argument, "
                        "it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(RS1.getReg())
        .addImm(SELECT_2.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_SRCMB_U16_Q_QACC_P: {
    unsigned Opc = RISCV::ESP_SRCMB_U16_Q_QACC;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QW = MI.getOperand(0);
    unsigned QWVal = QW.getImm();
    assert(QWVal < 8 && "Unexpected value of esp_srcmb_u16_q_qacc first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &SELECT_2 = MI.getOperand(1);
    MachineOperand &QU = MI.getOperand(2);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_srcmb_u16_q_qacc first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(RISCV::Q0 + QWVal)
        .addImm(SELECT_2.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_SRCMB_U16_QACC_P: {
    unsigned Opc = RISCV::ESP_SRCMB_U16_QACC;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(0);
    MachineOperand &SELECT_2 = MI.getOperand(1);
    MachineOperand &QU = MI.getOperand(2);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_srcmb_u16_qacc first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(RS1.getReg())
        .addImm(SELECT_2.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_SRCMB_U8_Q_QACC_P: {
    unsigned Opc = RISCV::ESP_SRCMB_U8_Q_QACC;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QW = MI.getOperand(0);
    unsigned QWVal = QW.getImm();
    assert(QWVal < 8 && "Unexpected value of esp_srcmb_u8_q_qacc first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &SELECT_2 = MI.getOperand(1);
    MachineOperand &QU = MI.getOperand(2);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_srcmb_u8_q_qacc first "
                        "argument, it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(RISCV::Q0 + QWVal)
        .addImm(SELECT_2.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_SRCMB_U8_QACC_P: {
    unsigned Opc = RISCV::ESP_SRCMB_U8_QACC;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(0);
    MachineOperand &SELECT_2 = MI.getOperand(1);
    MachineOperand &QU = MI.getOperand(2);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_srcmb_u8_qacc first argument, "
                        "it must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(RS1.getReg())
        .addImm(SELECT_2.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_SRCQ_128_ST_INCP_P: {
    unsigned Opc = RISCV::ESP_SRCQ_128_ST_INCP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QY = MI.getOperand(1);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_srcq_128_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &QW = MI.getOperand(2);
    unsigned QWVal = QW.getImm();
    assert(QWVal < 8 && "Unexpected value of esp_srcq_128_st_incp first "
                        "argument, it must bi in range [0,7]");
    MachineOperand &RS1 = MI.getOperand(3);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QWVal)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_SRCXXP_2Q_P: {
    unsigned Opc = RISCV::ESP_SRCXXP_2Q;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(0);
    MachineOperand &RS2 = MI.getOperand(1);
    MachineOperand &QY = MI.getOperand(2);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_srcxxp_2q first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QW = MI.getOperand(3);
    unsigned QWVal = QW.getImm();
    assert(QWVal < 8 && "Unexpected value of esp_srcxxp_2q first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QYVal, RegState::Define)
        .addReg(RISCV::Q0 + QWVal, RegState::Define)
        .addReg(RS1.getReg())
        .addReg(RS2.getReg())
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QWVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_SRS_S_XACC_P: {
    unsigned Opc = RISCV::ESP_SRS_S_XACC;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(0);
    const TargetRegisterClass *RC = &RISCV::GPRPIERegClass;
    unsigned R1 = MRI.createVirtualRegister(RC);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Define)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_SRS_U_XACC_P: {
    unsigned Opc = RISCV::ESP_SRS_U_XACC;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(0);
    const TargetRegisterClass *RC = &RISCV::GPRPIERegClass;
    unsigned R1 = MRI.createVirtualRegister(RC);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(R1, RegState::Define)
        .addReg(RS1.getReg());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VSL_32_P: {
    unsigned Opc = RISCV::ESP_VSL_32;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QY = MI.getOperand(0);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vsl_32 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(1);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vsl_32 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VSLD_16_P: {
    unsigned Opc = RISCV::ESP_VSLD_16;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QY = MI.getOperand(0);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vsld_16 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QW = MI.getOperand(1);
    unsigned QWVal = QW.getImm();
    assert(QWVal < 8 && "Unexpected value of esp_vsld_16 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(2);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vsld_16 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QWVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VSLD_32_P: {
    unsigned Opc = RISCV::ESP_VSLD_32;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QY = MI.getOperand(0);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vsld_32 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QW = MI.getOperand(1);
    unsigned QWVal = QW.getImm();
    assert(QWVal < 8 && "Unexpected value of esp_vsld_32 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(2);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vsld_32 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QWVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VSLD_8_P: {
    unsigned Opc = RISCV::ESP_VSLD_8;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QY = MI.getOperand(0);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vsld_8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QW = MI.getOperand(1);
    unsigned QWVal = QW.getImm();
    assert(QWVal < 8 && "Unexpected value of esp_vsld_8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(2);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vsld_8 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QWVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VSR_S32_P: {
    unsigned Opc = RISCV::ESP_VSR_S32;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QY = MI.getOperand(0);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vsr_s32 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(1);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vsr_s32 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VSR_U32_P: {
    unsigned Opc = RISCV::ESP_VSR_U32;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QY = MI.getOperand(0);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vsr_u32 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(1);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vsr_u32 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(RISCV::Q0 + QYVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VSRD_16_P: {
    unsigned Opc = RISCV::ESP_VSRD_16;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QY = MI.getOperand(0);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vsrd_16 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QW = MI.getOperand(1);
    unsigned QWVal = QW.getImm();
    assert(QWVal < 8 && "Unexpected value of esp_vsrd_16 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(2);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vsrd_16 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QWVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VSRD_32_P: {
    unsigned Opc = RISCV::ESP_VSRD_32;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QY = MI.getOperand(0);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vsrd_32 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QW = MI.getOperand(1);
    unsigned QWVal = QW.getImm();
    assert(QWVal < 8 && "Unexpected value of esp_vsrd_32 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(2);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vsrd_32 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QWVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_VSRD_8_P: {
    unsigned Opc = RISCV::ESP_VSRD_8;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &QY = MI.getOperand(0);
    unsigned QYVal = QY.getImm();
    assert(QYVal < 8 && "Unexpected value of esp_vsrd_8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QW = MI.getOperand(1);
    unsigned QWVal = QW.getImm();
    assert(QWVal < 8 && "Unexpected value of esp_vsrd_8 first argument, it "
                        "must bi in range [0,7]");
    MachineOperand &QU = MI.getOperand(2);
    unsigned QUVal = QU.getImm();
    assert(QUVal < 8 && "Unexpected value of esp_vsrd_8 first argument, it "
                        "must bi in range [0,7]");
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(RISCV::Q0 + QUVal, RegState::Define)
        .addReg(RISCV::Q0 + QYVal)
        .addReg(RISCV::Q0 + QWVal);

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_ST_S_XACC_IP_P: {
    unsigned Opc = RISCV::ESP_ST_S_XACC_IP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(1);
    MachineOperand &OFFSET_256_8 = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS1.getReg())
        .addImm(OFFSET_256_8.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  case RISCV::ESP_ST_U_XACC_IP_P: {
    unsigned Opc = RISCV::ESP_ST_U_XACC_IP;
    MachineBasicBlock *MBB = MI.getParent();
    MachineOperand &RS1 = MI.getOperand(1);
    MachineOperand &OFFSET_256_8 = MI.getOperand(2);
    BuildMI(*MBB, MI, DL, TII.get(Opc))
        .addReg(MI.getOperand(0).getReg(), RegState::Define)
        .addReg(RS1.getReg())
        .addImm(OFFSET_256_8.getImm());

    MI.eraseFromParent();
    return MBB;
  }
  }
}
