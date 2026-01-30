//===-- RISCVESPVISelLowering.cpp - ESPV DAG Lowering Implementation -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements ESPV-specific lowering functions for RISC-V.
//
//===----------------------------------------------------------------------===//

#include "RISCVESPVISelLowering.h"
#include "RISCV.h"
#include "RISCVISelLowering.h"
#include "RISCVSelectionDAGInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>

using namespace llvm;
using namespace llvm::ISD;

namespace llvm {
namespace RISCV {

static SDValue LowerSTXACCIP(SDValue Op, SelectionDAG &DAG, unsigned ISDOpcode);
static SDValue LowerLDQAIP(SDValue Op, SelectionDAG &DAG, unsigned ISDOpcode);
static SDValue LowerLDQAXP(SDValue Op, SelectionDAG &DAG, unsigned ISDOpcode);
static SDValue LowerLDUASTATEIP(SDValue Op, SelectionDAG &DAG,
                                unsigned ISDOpcode);
static SDValue LowerSTUASTATEIP(SDValue Op, SelectionDAG &DAG,
                                unsigned ISDOpcode);
static SDValue LowerVMULASQACCLDIP(SDValue Op, SelectionDAG &DAG,
                                   unsigned ISDOpcode);
static SDValue LowerVMULASQACCLDXP(SDValue Op, SelectionDAG &DAG,
                                   unsigned ISDOpcode);
static SDValue LowerVMULASQACCSTIP(SDValue Op, SelectionDAG &DAG,
                                   unsigned ISDOpcode);
static SDValue LowerVMULASQACCSTXP(SDValue Op, SelectionDAG &DAG,
                                   unsigned ISDOpcode);
static SDValue LowerVMULASQACCLDBCINCP(SDValue Op, SelectionDAG &DAG,
                                       unsigned ISDOpcode);

bool getESPVTgtMemIntrinsic(TargetLowering::IntrinsicInfo &Info,
                            const CallInst &I, unsigned Intrinsic) {
  switch (Intrinsic) {
  default:
    return false;
  case Intrinsic::riscv_esp_vld_128_ip_m:
  case Intrinsic::riscv_esp_vld_128_xp_m:
  case Intrinsic::riscv_esp_ld_128_usar_ip_m:
  case Intrinsic::riscv_esp_ld_128_usar_xp_m: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(0);
    Info.memVT = MVT::v16i8;
    Info.align = Align(16);
    Info.size = 16;
    Info.flags |= MachineMemOperand::MOLoad;
    return true;
  }
  case Intrinsic::riscv_esp_ld_qacc_h_h_128_ip_m:
  case Intrinsic::riscv_esp_ld_qacc_h_l_128_ip_m:
  case Intrinsic::riscv_esp_ld_qacc_l_h_128_ip_m:
  case Intrinsic::riscv_esp_ld_qacc_l_l_128_ip_m: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(0);
    Info.memVT = MVT::v16i8;
    Info.align = Align(16);
    Info.size = 16;
    Info.flags |= MachineMemOperand::MOLoad;
    return true;
  }
  case Intrinsic::riscv_esp_ldqa_s16_128_ip_m:
  case Intrinsic::riscv_esp_ldqa_s16_128_xp_m:
  case Intrinsic::riscv_esp_ldqa_s8_128_ip_m:
  case Intrinsic::riscv_esp_ldqa_s8_128_xp_m:
  case Intrinsic::riscv_esp_ldqa_u16_128_ip_m:
  case Intrinsic::riscv_esp_ldqa_u16_128_xp_m:
  case Intrinsic::riscv_esp_ldqa_u8_128_ip_m:
  case Intrinsic::riscv_esp_ldqa_u8_128_xp_m: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(1);
    Info.memVT = MVT::v16i8;
    Info.align = Align(16);
    Info.size = 16;
    Info.flags |= MachineMemOperand::MOLoad;
    return true;
  }
  case Intrinsic::riscv_esp_vadd_s8_ld_incp_m:
  case Intrinsic::riscv_esp_vadd_u8_ld_incp_m:
  case Intrinsic::riscv_esp_vadd_s16_ld_incp_m:
  case Intrinsic::riscv_esp_vadd_u16_ld_incp_m: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(2);
    Info.memVT = MVT::v16i8;
    Info.align = Align(16);
    Info.size = 16;
    Info.flags |= MachineMemOperand::MOLoad;
    return true;
  }
  case Intrinsic::riscv_esp_vadd_s8_st_incp_m:
  case Intrinsic::riscv_esp_vadd_u8_st_incp_m:
  case Intrinsic::riscv_esp_vadd_s16_st_incp_m:
  case Intrinsic::riscv_esp_vadd_u16_st_incp_m: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(3);
    Info.memVT = MVT::v16i8;
    Info.align = Align(16);
    Info.size = 16;
    Info.flags |= MachineMemOperand::MOStore;
    return true;
  }
  case Intrinsic::riscv_esp_vmulas_s8_qacc_ldbc_incp_m:
  case Intrinsic::riscv_esp_vmulas_s16_qacc_ldbc_incp_m:
  case Intrinsic::riscv_esp_vmulas_u8_qacc_ldbc_incp_m:
  case Intrinsic::riscv_esp_vmulas_u16_qacc_ldbc_incp_m: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(6);
    Info.memVT = MVT::v16i8;
    Info.align = Align(16);
    Info.size = 16;
    Info.flags |= MachineMemOperand::MOLoad;
    return true;
  }
  case Intrinsic::riscv_esp_vmulas_s8_qacc_ld_ip_m:
  case Intrinsic::riscv_esp_vmulas_s16_qacc_ld_ip_m:
  case Intrinsic::riscv_esp_vmulas_u8_qacc_ld_ip_m:
  case Intrinsic::riscv_esp_vmulas_u16_qacc_ld_ip_m: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(6);
    Info.memVT = MVT::v16i8;
    Info.align = Align(16);
    Info.size = 16;
    Info.flags |= MachineMemOperand::MOLoad;
    return true;
  }
  case Intrinsic::riscv_esp_vmulas_s8_qacc_ld_xp_m:
  case Intrinsic::riscv_esp_vmulas_s16_qacc_ld_xp_m:
  case Intrinsic::riscv_esp_vmulas_u8_qacc_ld_xp_m:
  case Intrinsic::riscv_esp_vmulas_u16_qacc_ld_xp_m: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(6);
    Info.memVT = MVT::v16i8;
    Info.align = Align(16);
    Info.size = 16;
    Info.flags |= MachineMemOperand::MOLoad;
    return true;
  }
  case Intrinsic::riscv_esp_vmulas_s8_qacc_st_ip_m:
  case Intrinsic::riscv_esp_vmulas_s16_qacc_st_ip_m:
  case Intrinsic::riscv_esp_vmulas_u8_qacc_st_ip_m:
  case Intrinsic::riscv_esp_vmulas_u16_qacc_st_ip_m: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(7);
    Info.memVT = MVT::v16i8;
    Info.align = Align(16);
    Info.size = 16;
    Info.flags |= MachineMemOperand::MOStore;
    return true;
  }
  case Intrinsic::riscv_esp_vmulas_s8_qacc_st_xp_m:
  case Intrinsic::riscv_esp_vmulas_s16_qacc_st_xp_m:
  case Intrinsic::riscv_esp_vmulas_u8_qacc_st_xp_m:
  case Intrinsic::riscv_esp_vmulas_u16_qacc_st_xp_m: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(7);
    Info.memVT = MVT::v16i8;
    Info.align = Align(16);
    Info.size = 16;
    Info.flags |= MachineMemOperand::MOStore;
    return true;
  }
  case Intrinsic::riscv_esp_st_s_xacc_ip_m: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(2);
    Info.memVT = MVT::i64;
    Info.align = Align(8);
    Info.size = 8;
    Info.flags |= MachineMemOperand::MOStore;
    return true;
  }
  case Intrinsic::riscv_esp_vcmulas_s8_qacc_h_ld_ip_m:
  case Intrinsic::riscv_esp_vcmulas_s8_qacc_l_ld_ip_m:
  case Intrinsic::riscv_esp_vcmulas_s16_qacc_h_ld_ip_m:
  case Intrinsic::riscv_esp_vcmulas_s16_qacc_l_ld_ip_m:
  case Intrinsic::riscv_esp_vcmulas_s8_qacc_h_ld_xp_m:
  case Intrinsic::riscv_esp_vcmulas_s8_qacc_l_ld_xp_m:
  case Intrinsic::riscv_esp_vcmulas_s16_qacc_h_ld_xp_m:
  case Intrinsic::riscv_esp_vcmulas_s16_qacc_l_ld_xp_m: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(4);
    Info.memVT = MVT::v16i8;
    Info.align = Align(16);
    Info.size = 16;
    Info.flags |= MachineMemOperand::MOLoad;
    return true;
  }
  case Intrinsic::riscv_esp_vst_128_ip_m:
  case Intrinsic::riscv_esp_vst_128_xp_m:
  case Intrinsic::riscv_esp_st_qacc_h_h_128_ip_m:
  case Intrinsic::riscv_esp_st_qacc_h_l_128_ip_m:
  case Intrinsic::riscv_esp_st_qacc_l_h_128_ip_m:
  case Intrinsic::riscv_esp_st_qacc_l_l_128_ip_m: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(1);
    Info.memVT = MVT::v16i8;
    Info.align = Align(16);
    Info.size = 16;
    Info.flags |= MachineMemOperand::MOStore;
    return true;
  }
  case Intrinsic::riscv_esp_cmul_s8_ld_incp_m:
  case Intrinsic::riscv_esp_cmul_s16_ld_incp_m:
  case Intrinsic::riscv_esp_cmul_u8_ld_incp_m:
  case Intrinsic::riscv_esp_cmul_u16_ld_incp_m: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(3);
    Info.memVT = MVT::v16i8;
    Info.align = Align(16);
    Info.size = 16;
    Info.flags |= MachineMemOperand::MOLoad;
    return true;
  }
  case Intrinsic::riscv_esp_fft_ams_s16_ld_incp_m:
  case Intrinsic::riscv_esp_fft_ams_s16_ld_incp_uaup_m:
  case Intrinsic::riscv_esp_fft_ams_s16_ld_r32_decp_m: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(3);
    Info.memVT = MVT::v16i8;
    Info.align = Align(16);
    Info.size = 16;
    Info.flags |= MachineMemOperand::MOLoad;
    return true;
  }
  case Intrinsic::riscv_esp_fft_r2bf_s16_st_incp_m: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(2);
    Info.memVT = MVT::v16i8;
    Info.align = Align(16);
    Info.size = 16;
    Info.flags |= MachineMemOperand::MOStore;
    return true;
  }
  case Intrinsic::riscv_esp_fft_ams_s16_st_incp_m: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(4);
    Info.memVT = MVT::v16i8;
    Info.align = Align(16);
    Info.size = 16;
    Info.flags |= MachineMemOperand::MOStore;
    return true;
  }
  case Intrinsic::riscv_esp_fft_cmul_s16_ld_xp_m: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(2);
    Info.memVT = MVT::v16i8;
    Info.align = Align(16);
    Info.size = 16;
    Info.flags |= MachineMemOperand::MOLoad;
    return true;
  }
  case Intrinsic::riscv_esp_fft_cmul_s16_st_xp_m: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(3);
    Info.memVT = MVT::v16i8;
    Info.align = Align(16);
    Info.size = 16;
    Info.flags |= MachineMemOperand::MOStore;
    return true;
  }
  case Intrinsic::riscv_esp_fft_vst_r32_decp_m: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(1);
    Info.memVT = MVT::v16i8;
    Info.align = Align(16);
    Info.size = 16;
    Info.flags |= MachineMemOperand::MOStore;
    return true;
  }
  case Intrinsic::riscv_esp_cmul_s8_st_incp_m:
  case Intrinsic::riscv_esp_cmul_s16_st_incp_m:
  case Intrinsic::riscv_esp_cmul_u8_st_incp_m:
  case Intrinsic::riscv_esp_cmul_u16_st_incp_m: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(4);
    Info.memVT = MVT::v16i8;
    Info.align = Align(16);
    Info.size = 16;
    Info.flags |= MachineMemOperand::MOStore;
    return true;
  }
  case Intrinsic::riscv_esp_vld_h_64_ip_m:
  case Intrinsic::riscv_esp_vld_h_64_xp_m:
  case Intrinsic::riscv_esp_vld_l_64_ip_m:
  case Intrinsic::riscv_esp_vld_l_64_xp_m: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(0);
    Info.memVT = MVT::v8i8;
    Info.align = Align(8);
    Info.size = 8;
    Info.flags |= MachineMemOperand::MOLoad;
    return true;
  }
  case Intrinsic::riscv_esp_vst_h_64_ip_m:
  case Intrinsic::riscv_esp_vst_h_64_xp_m:
  case Intrinsic::riscv_esp_vst_l_64_ip_m:
  case Intrinsic::riscv_esp_vst_l_64_xp_m: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(1);
    Info.memVT = MVT::v8i8;
    Info.align = Align(8);
    Info.size = 8;
    Info.flags |= MachineMemOperand::MOStore;
    return true;
  }
  case Intrinsic::riscv_esp_vldbc_8_ip_m:
  case Intrinsic::riscv_esp_vldbc_8_xp_m: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(0);
    Info.memVT = MVT::i8;
    Info.align = Align(1);
    Info.size = 1;
    Info.flags |= MachineMemOperand::MOLoad;
    return true;
  }
  case Intrinsic::riscv_esp_vldbc_16_ip_m:
  case Intrinsic::riscv_esp_vldbc_16_xp_m: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(0);
    Info.memVT = MVT::i16;
    Info.align = Align(2);
    Info.size = 2;
    Info.flags |= MachineMemOperand::MOLoad;
    return true;
  }
  case Intrinsic::riscv_esp_vldbc_32_ip_m:
  case Intrinsic::riscv_esp_vldbc_32_xp_m: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(0);
    Info.memVT = MVT::i32;
    Info.align = Align(4);
    Info.size = 4;
    Info.flags |= MachineMemOperand::MOLoad;
    return true;
  }
  case Intrinsic::riscv_esp_vldext_s8_ip_m:
  case Intrinsic::riscv_esp_vldext_s8_xp_m:
  case Intrinsic::riscv_esp_vldext_u8_ip_m:
  case Intrinsic::riscv_esp_vldext_u8_xp_m: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(0);
    Info.memVT = MVT::v8i8;
    Info.align = Align(8);
    Info.size = 8;
    Info.flags |= MachineMemOperand::MOLoad;
    return true;
  }
  case Intrinsic::riscv_esp_vldext_s16_ip_m:
  case Intrinsic::riscv_esp_vldext_s16_xp_m:
  case Intrinsic::riscv_esp_vldext_u16_ip_m:
  case Intrinsic::riscv_esp_vldext_u16_xp_m: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(0);
    Info.memVT = MVT::v4i16;
    Info.align = Align(8);
    Info.size = 8;
    Info.flags |= MachineMemOperand::MOLoad;
    return true;
  }
  }
}

// ESPV intrinsic lowering for INTRINSIC_W_CHAIN
SDValue lowerESPVIntrinsicWChain(SDValue Op, SelectionDAG &DAG,
                                 const RISCVSubtarget &Subtarget) {
  if (!Subtarget.hasVendorXespv2p1())
    return SDValue();

  unsigned IntNo = Op.getConstantOperandVal(1);
  SDLoc DL(Op);

  switch (IntNo) {
  case Intrinsic::riscv_esp_vld_128_ip_m: {
    // Lower intrinsic to custom SDNode that will be matched to ESP_VLD_128_IP
    // Intrinsic: (chain, int_id, ptr, imm)

    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue Ptr = Op.getOperand(2);
    SDValue Imm = Op.getOperand(3);

    EVT VecVT = MVT::v16i8;
    EVT PtrVT = Ptr.getValueType();
    SDVTList VTs = DAG.getVTList(VecVT, PtrVT, MVT::Other);

    SDValue Ops[] = {Chain, Ptr, Imm};

    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_VLD_128_IP_M, DL, VTs,
                                           Ops, VecVT, MMO);
    return DAG.getMergeValues(
        {Node.getValue(0), Node.getValue(1), Node.getValue(2)}, DL);
  }
  case Intrinsic::riscv_esp_vld_128_xp_m: {
    // Lower intrinsic to custom SDNode that will be matched to
    // ESP_VLD_128_XP_M_P Intrinsic: (chain, int_id, ptr, offset_reg) Note: This
    // intrinsic always arrives as MemIntrinsicSDNode because
    //       getTgtMemIntrinsic returns true for it.
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue Ptr = Op.getOperand(2);
    SDValue Offset = Op.getOperand(3); // Register offset

    EVT VecVT = MVT::v16i8;
    EVT PtrVT = Ptr.getValueType();
    SDVTList VTs = DAG.getVTList(VecVT, PtrVT, MVT::Other);

    SDValue Ops[] = {Chain, Ptr, Offset};

    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_VLD_128_XP_M, DL, VTs,
                                           Ops, VecVT, MMO);
    return DAG.getMergeValues(
        {Node.getValue(0), Node.getValue(1), Node.getValue(2)}, DL);
  }
  case Intrinsic::riscv_esp_vst_128_ip_m: {
    // Lower intrinsic to custom SDNode that will be matched to
    // ESP_VST_128_IP_M_P Intrinsic: (chain, int_id, vec, ptr, imm) Note: This
    // intrinsic always arrives as MemIntrinsicSDNode because
    //       getTgtMemIntrinsic returns true for it.
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue Vec = Op.getOperand(2);
    SDValue Ptr = Op.getOperand(3);
    SDValue Imm = Op.getOperand(4);

    EVT VecVT = MVT::v16i8;
    EVT PtrVT = Ptr.getValueType();
    SDVTList VTs = DAG.getVTList(PtrVT, MVT::Other);

    SDValue Ops[] = {Chain, Vec, Ptr, Imm};

    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_VST_128_IP_M, DL, VTs,
                                           Ops, VecVT, MMO);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1)}, DL);
  }
  // LD/ST UA_STATE IP
  case Intrinsic::riscv_esp_ld_ua_state_ip_m:
    return LowerLDUASTATEIP(Op, DAG, RISCVISD::ESP_LD_UA_STATE_IP_M);
  case Intrinsic::riscv_esp_st_ua_state_ip_m:
    return LowerSTUASTATEIP(Op, DAG, RISCVISD::ESP_ST_UA_STATE_IP_M);
  case Intrinsic::riscv_esp_ld_128_usar_ip_m: {
    // Lower intrinsic to custom SDNode that will be matched to
    // ESP_LD_128_USAR_IP Intrinsic: (chain, int_id, ptr, imm) Returns: vector,
    // updated pointer, SAR_BYTES (32-bit, only low 4 bits used)
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue Ptr = Op.getOperand(2);
    SDValue Imm = Op.getOperand(3);

    EVT VecVT = MVT::v16i8;
    EVT PtrVT = Ptr.getValueType();
    EVT SarBytesVT = MVT::i32;
    SDVTList VTs = DAG.getVTList(VecVT, PtrVT, SarBytesVT, MVT::Other);

    SDValue Ops[] = {Chain, Ptr, Imm};
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_LD_128_USAR_IP_M, DL,
                                           VTs, Ops, VecVT, MMO);

    // Return: vector, updated pointer, SAR_BYTES, chain
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1),
                               Node.getValue(2), Node.getValue(3)},
                              DL);
  }
  case Intrinsic::riscv_esp_ld_128_usar_xp_m: {
    // Lower intrinsic to custom SDNode that will be matched to
    // ESP_LD_128_USAR_XP Intrinsic: (chain, int_id, ptr, offset_reg) Returns:
    // vector, updated pointer, SAR_BYTES (32-bit, only low 4 bits used)
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue Ptr = Op.getOperand(2);
    SDValue Offset = Op.getOperand(3);

    EVT VecVT = MVT::v16i8;
    EVT PtrVT = Ptr.getValueType();
    EVT SarBytesVT = MVT::i32;
    SDVTList VTs = DAG.getVTList(VecVT, PtrVT, SarBytesVT, MVT::Other);

    SDValue Ops[] = {Chain, Ptr, Offset};
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_LD_128_USAR_XP_M, DL,
                                           VTs, Ops, VecVT, MMO);

    // Return: vector, updated pointer, SAR_BYTES, chain
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1),
                               Node.getValue(2), Node.getValue(3)},
                              DL);
  }
  case Intrinsic::riscv_esp_vst_128_xp_m: {
    // Lower intrinsic to custom SDNode that will be matched to
    // ESP_VST_128_XP_M_P Intrinsic: (chain, int_id, vec, ptr, offset_reg) Note:
    // This intrinsic always arrives as MemIntrinsicSDNode because
    //       getTgtMemIntrinsic returns true for it.
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue Vec = Op.getOperand(2);
    SDValue Ptr = Op.getOperand(3);
    SDValue Offset = Op.getOperand(4); // Register offset

    EVT VecVT = MVT::v16i8;
    EVT PtrVT = Ptr.getValueType();
    SDVTList VTs = DAG.getVTList(PtrVT, MVT::Other);

    SDValue Ops[] = {Chain, Vec, Ptr, Offset};

    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_VST_128_XP_M, DL, VTs,
                                           Ops, VecVT, MMO);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1)}, DL);
  }
  case Intrinsic::riscv_esp_vld_h_64_ip_m: {

    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue Ptr = Op.getOperand(2);
    SDValue Imm = Op.getOperand(3);

    EVT VecVT = MVT::v8i8;
    EVT PtrVT = Ptr.getValueType();
    SDVTList VTs = DAG.getVTList(VecVT, PtrVT, MVT::Other);

    SDValue Ops[] = {Chain, Ptr, Imm};
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_VLD_H_64_IP_M, DL, VTs,
                                           Ops, VecVT, MMO);
    return DAG.getMergeValues(
        {Node.getValue(0), Node.getValue(1), Node.getValue(2)}, DL);
  }
  case Intrinsic::riscv_esp_vld_h_64_xp_m: {

    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue Ptr = Op.getOperand(2);
    SDValue Offset = Op.getOperand(3);

    EVT VecVT = MVT::v8i8;
    EVT PtrVT = Ptr.getValueType();
    SDVTList VTs = DAG.getVTList(VecVT, PtrVT, MVT::Other);

    SDValue Ops[] = {Chain, Ptr, Offset};
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_VLD_H_64_XP_M, DL, VTs,
                                           Ops, VecVT, MMO);
    return DAG.getMergeValues(
        {Node.getValue(0), Node.getValue(1), Node.getValue(2)}, DL);
  }
  case Intrinsic::riscv_esp_vld_l_64_ip_m: {

    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue Ptr = Op.getOperand(2);
    SDValue Imm = Op.getOperand(3);

    EVT VecVT = MVT::v8i8;
    EVT PtrVT = Ptr.getValueType();
    SDVTList VTs = DAG.getVTList(VecVT, PtrVT, MVT::Other);

    SDValue Ops[] = {Chain, Ptr, Imm};
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_VLD_L_64_IP_M, DL, VTs,
                                           Ops, VecVT, MMO);
    return DAG.getMergeValues(
        {Node.getValue(0), Node.getValue(1), Node.getValue(2)}, DL);
  }
  case Intrinsic::riscv_esp_vld_l_64_xp_m: {

    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue Ptr = Op.getOperand(2);
    SDValue Offset = Op.getOperand(3);

    EVT VecVT = MVT::v8i8;
    EVT PtrVT = Ptr.getValueType();
    SDVTList VTs = DAG.getVTList(VecVT, PtrVT, MVT::Other);

    SDValue Ops[] = {Chain, Ptr, Offset};
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_VLD_L_64_XP_M, DL, VTs,
                                           Ops, VecVT, MMO);
    return DAG.getMergeValues(
        {Node.getValue(0), Node.getValue(1), Node.getValue(2)}, DL);
  }
  case Intrinsic::riscv_esp_vst_h_64_ip_m: {
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue Vec = Op.getOperand(2);
    SDValue Ptr = Op.getOperand(3);
    SDValue Imm = Op.getOperand(4);

    // Extract high 64 bits (v8i8) from 128-bit vector (v16i8)
    // High 64 bits are at index 8 (second half of v16i8)
    EVT VecVT = Vec.getValueType();
    if (VecVT == MVT::v16i8) {
      Vec = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, MVT::v8i8, Vec,
                        DAG.getConstant(8, DL, MVT::i32));
    }

    EVT PtrVT = Ptr.getValueType();
    SDVTList VTs = DAG.getVTList(PtrVT, MVT::Other);

    SDValue Ops[] = {Chain, Vec, Ptr, Imm};
    VecVT = MVT::v8i8;
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_VST_H_64_IP_M, DL, VTs,
                                           Ops, VecVT, MMO);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1)}, DL);
  }
  case Intrinsic::riscv_esp_vst_h_64_xp_m: {
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue Vec = Op.getOperand(2);
    SDValue Ptr = Op.getOperand(3);
    SDValue Offset = Op.getOperand(4);

    // Extract high 64 bits (v8i8) from 128-bit vector (v16i8)
    // High 64 bits are at index 8 (second half of v16i8)
    EVT VecVT = Vec.getValueType();
    if (VecVT == MVT::v16i8) {
      Vec = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, MVT::v8i8, Vec,
                        DAG.getConstant(8, DL, MVT::i32));
    }

    EVT PtrVT = Ptr.getValueType();
    SDVTList VTs = DAG.getVTList(PtrVT, MVT::Other);

    SDValue Ops[] = {Chain, Vec, Ptr, Offset};
    VecVT = MVT::v8i8;
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_VST_H_64_XP_M, DL, VTs,
                                           Ops, VecVT, MMO);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1)}, DL);
  }
  case Intrinsic::riscv_esp_vst_l_64_ip_m: {
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue Vec = Op.getOperand(2);
    SDValue Ptr = Op.getOperand(3);
    SDValue Imm = Op.getOperand(4);

    // Extract low 64 bits (v8i8) from 128-bit vector (v16i8)
    // Low 64 bits are at index 0 (first half of v16i8)
    EVT VecVT = Vec.getValueType();
    if (VecVT == MVT::v16i8) {
      Vec = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, MVT::v8i8, Vec,
                        DAG.getConstant(0, DL, MVT::i32));
    }

    EVT PtrVT = Ptr.getValueType();
    SDVTList VTs = DAG.getVTList(PtrVT, MVT::Other);

    SDValue Ops[] = {Chain, Vec, Ptr, Imm};
    VecVT = MVT::v8i8;
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_VST_L_64_IP_M, DL, VTs,
                                           Ops, VecVT, MMO);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1)}, DL);
  }
  case Intrinsic::riscv_esp_vst_l_64_xp_m: {
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue Vec = Op.getOperand(2);
    SDValue Ptr = Op.getOperand(3);
    SDValue Offset = Op.getOperand(4);

    // Extract low 64 bits (v8i8) from 128-bit vector (v16i8)
    // Low 64 bits are at index 0 (first half of v16i8)
    EVT VecVT = Vec.getValueType();
    if (VecVT == MVT::v16i8) {
      Vec = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, MVT::v8i8, Vec,
                        DAG.getConstant(0, DL, MVT::i32));
    }

    EVT PtrVT = Ptr.getValueType();
    SDVTList VTs = DAG.getVTList(PtrVT, MVT::Other);

    SDValue Ops[] = {Chain, Vec, Ptr, Offset};
    VecVT = MVT::v8i8;
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_VST_L_64_XP_M, DL, VTs,
                                           Ops, VecVT, MMO);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1)}, DL);
  }
  case Intrinsic::riscv_esp_vldbc_8_ip_m: {

    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue Ptr = Op.getOperand(2);
    SDValue Imm = Op.getOperand(3);

    EVT ResultVT = MVT::v16i8; // Result vector type
    EVT MemVT = MVT::i8;       // Memory access type (1 byte)
    EVT PtrVT = Ptr.getValueType();
    SDVTList VTs = DAG.getVTList(ResultVT, PtrVT, MVT::Other);

    SDValue Ops[] = {Chain, Ptr, Imm};
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_VLDBC_8_IP_M, DL, VTs,
                                           Ops, MemVT, MMO);
    return DAG.getMergeValues(
        {Node.getValue(0), Node.getValue(1), Node.getValue(2)}, DL);
  }
  case Intrinsic::riscv_esp_vldbc_8_xp_m: {

    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue Ptr = Op.getOperand(2);
    SDValue Offset = Op.getOperand(3);

    EVT ResultVT = MVT::v16i8; // Result vector type
    EVT MemVT = MVT::i8;       // Memory access type (1 byte)
    EVT PtrVT = Ptr.getValueType();
    SDVTList VTs = DAG.getVTList(ResultVT, PtrVT, MVT::Other);

    SDValue Ops[] = {Chain, Ptr, Offset};
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_VLDBC_8_XP_M, DL, VTs,
                                           Ops, MemVT, MMO);
    return DAG.getMergeValues(
        {Node.getValue(0), Node.getValue(1), Node.getValue(2)}, DL);
  }
  case Intrinsic::riscv_esp_vldbc_16_ip_m: {

    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue Ptr = Op.getOperand(2);
    SDValue Imm = Op.getOperand(3);

    EVT ResultVT = MVT::v8i16; // Result vector type
    EVT MemVT = MVT::i16;      // Memory access type (2 bytes)
    EVT PtrVT = Ptr.getValueType();
    SDVTList VTs = DAG.getVTList(ResultVT, PtrVT, MVT::Other);

    SDValue Ops[] = {Chain, Ptr, Imm};
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_VLDBC_16_IP_M, DL, VTs,
                                           Ops, MemVT, MMO);
    return DAG.getMergeValues(
        {Node.getValue(0), Node.getValue(1), Node.getValue(2)}, DL);
  }
  case Intrinsic::riscv_esp_vldbc_16_xp_m: {

    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue Ptr = Op.getOperand(2);
    SDValue Offset = Op.getOperand(3);

    EVT ResultVT = MVT::v8i16; // Result vector type
    EVT MemVT = MVT::i16;      // Memory access type (2 bytes)
    EVT PtrVT = Ptr.getValueType();
    SDVTList VTs = DAG.getVTList(ResultVT, PtrVT, MVT::Other);

    SDValue Ops[] = {Chain, Ptr, Offset};
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_VLDBC_16_XP_M, DL, VTs,
                                           Ops, MemVT, MMO);
    return DAG.getMergeValues(
        {Node.getValue(0), Node.getValue(1), Node.getValue(2)}, DL);
  }
  case Intrinsic::riscv_esp_vldbc_32_ip_m: {

    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue Ptr = Op.getOperand(2);
    SDValue Imm = Op.getOperand(3);

    EVT ResultVT = MVT::v4i32; // Result vector type
    EVT MemVT = MVT::i32;      // Memory access type (4 bytes)
    EVT PtrVT = Ptr.getValueType();
    SDVTList VTs = DAG.getVTList(ResultVT, PtrVT, MVT::Other);

    SDValue Ops[] = {Chain, Ptr, Imm};
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_VLDBC_32_IP_M, DL, VTs,
                                           Ops, MemVT, MMO);
    return DAG.getMergeValues(
        {Node.getValue(0), Node.getValue(1), Node.getValue(2)}, DL);
  }
  case Intrinsic::riscv_esp_vldbc_32_xp_m: {

    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue Ptr = Op.getOperand(2);
    SDValue Offset = Op.getOperand(3);

    EVT ResultVT = MVT::v4i32; // Result vector type
    EVT MemVT = MVT::i32;      // Memory access type (4 bytes)
    EVT PtrVT = Ptr.getValueType();
    SDVTList VTs = DAG.getVTList(ResultVT, PtrVT, MVT::Other);

    SDValue Ops[] = {Chain, Ptr, Offset};
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_VLDBC_32_XP_M, DL, VTs,
                                           Ops, MemVT, MMO);
    return DAG.getMergeValues(
        {Node.getValue(0), Node.getValue(1), Node.getValue(2)}, DL);
  }
  case Intrinsic::riscv_esp_vldext_s8_ip_m: {
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue Ptr = Op.getOperand(2);
    SDValue Imm = Op.getOperand(3);

    EVT VecVT = MVT::v8i16;
    EVT MemVT = MVT::v8i8; // Memory type: 8 bytes (8 x i8)
    EVT PtrVT = Ptr.getValueType();
    SDVTList VTs = DAG.getVTList(VecVT, VecVT, PtrVT, MVT::Other);

    SDValue Ops[] = {Chain, Ptr, Imm};
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_VLDEXT_S8_IP_M, DL,
                                           VTs, Ops, MemVT, MMO);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1),
                               Node.getValue(2), Node.getValue(3)},
                              DL);
  }
  case Intrinsic::riscv_esp_vldext_s8_xp_m: {
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue Ptr = Op.getOperand(2);
    SDValue Offset = Op.getOperand(3);

    EVT VecVT = MVT::v8i16;
    EVT MemVT = MVT::v8i8; // Memory type: 8 bytes (8 x i8)
    EVT PtrVT = Ptr.getValueType();
    SDVTList VTs = DAG.getVTList(VecVT, VecVT, PtrVT, MVT::Other);

    SDValue Ops[] = {Chain, Ptr, Offset};
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_VLDEXT_S8_XP_M, DL,
                                           VTs, Ops, MemVT, MMO);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1),
                               Node.getValue(2), Node.getValue(3)},
                              DL);
  }
  case Intrinsic::riscv_esp_vldext_s16_ip_m: {
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue Ptr = Op.getOperand(2);
    SDValue Imm = Op.getOperand(3);

    EVT VecVT = MVT::v4i32;
    EVT MemVT = MVT::v4i16; // Memory type: 8 bytes (4 x i16)
    EVT PtrVT = Ptr.getValueType();
    SDVTList VTs = DAG.getVTList(VecVT, VecVT, PtrVT, MVT::Other);

    SDValue Ops[] = {Chain, Ptr, Imm};
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_VLDEXT_S16_IP_M, DL,
                                           VTs, Ops, MemVT, MMO);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1),
                               Node.getValue(2), Node.getValue(3)},
                              DL);
  }
  case Intrinsic::riscv_esp_vldext_s16_xp_m: {
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue Ptr = Op.getOperand(2);
    SDValue Offset = Op.getOperand(3);

    EVT VecVT = MVT::v4i32;
    EVT MemVT = MVT::v4i16; // Memory type: 8 bytes (4 x i16)
    EVT PtrVT = Ptr.getValueType();
    SDVTList VTs = DAG.getVTList(VecVT, VecVT, PtrVT, MVT::Other);

    SDValue Ops[] = {Chain, Ptr, Offset};
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_VLDEXT_S16_XP_M, DL,
                                           VTs, Ops, MemVT, MMO);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1),
                               Node.getValue(2), Node.getValue(3)},
                              DL);
  }
  case Intrinsic::riscv_esp_vldext_u8_ip_m: {
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue Ptr = Op.getOperand(2);
    SDValue Imm = Op.getOperand(3);

    EVT VecVT = MVT::v8i16;
    EVT MemVT = MVT::v8i8; // Memory type: 8 bytes (8 x i8)
    EVT PtrVT = Ptr.getValueType();
    SDVTList VTs = DAG.getVTList(VecVT, VecVT, PtrVT, MVT::Other);

    SDValue Ops[] = {Chain, Ptr, Imm};
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_VLDEXT_U8_IP_M, DL,
                                           VTs, Ops, MemVT, MMO);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1),
                               Node.getValue(2), Node.getValue(3)},
                              DL);
  }
  case Intrinsic::riscv_esp_vldext_u8_xp_m: {
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue Ptr = Op.getOperand(2);
    SDValue Offset = Op.getOperand(3);

    EVT VecVT = MVT::v8i16;
    EVT MemVT = MVT::v8i8; // Memory type: 8 bytes (8 x i8)
    EVT PtrVT = Ptr.getValueType();
    SDVTList VTs = DAG.getVTList(VecVT, VecVT, PtrVT, MVT::Other);

    SDValue Ops[] = {Chain, Ptr, Offset};
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_VLDEXT_U8_XP_M, DL,
                                           VTs, Ops, MemVT, MMO);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1),
                               Node.getValue(2), Node.getValue(3)},
                              DL);
  }
  case Intrinsic::riscv_esp_vldext_u16_ip_m: {
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue Ptr = Op.getOperand(2);
    SDValue Imm = Op.getOperand(3);

    EVT VecVT = MVT::v4i32;
    EVT MemVT = MVT::v4i16; // Memory type: 8 bytes (4 x i16)
    EVT PtrVT = Ptr.getValueType();
    SDVTList VTs = DAG.getVTList(VecVT, VecVT, PtrVT, MVT::Other);

    SDValue Ops[] = {Chain, Ptr, Imm};
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_VLDEXT_U16_IP_M, DL,
                                           VTs, Ops, MemVT, MMO);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1),
                               Node.getValue(2), Node.getValue(3)},
                              DL);
  }
  case Intrinsic::riscv_esp_vldext_u16_xp_m: {
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue Ptr = Op.getOperand(2);
    SDValue Offset = Op.getOperand(3);

    EVT VecVT = MVT::v4i32;
    EVT MemVT = MVT::v4i16; // Memory type: 8 bytes (4 x i16)
    EVT PtrVT = Ptr.getValueType();
    SDVTList VTs = DAG.getVTList(VecVT, VecVT, PtrVT, MVT::Other);

    SDValue Ops[] = {Chain, Ptr, Offset};
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_VLDEXT_U16_XP_M, DL,
                                           VTs, Ops, MemVT, MMO);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1),
                               Node.getValue(2), Node.getValue(3)},
                              DL);
  }
  case Intrinsic::riscv_esp_cmul_s16_ld_incp_m: {
    // Lower CMUL S16 LD INCP intrinsic to custom SDNode with explicit SAR state
    // passing Intrinsic: (chain, int_id, qz_in, qx, qy, rs1, sel4, sar)
    // Returns: {qz, qu, ptr}
    // SDNode: (chain, qz_in, qx, qy, rs1, sel4, sar) -> (qz, qu, rs1r, chain)
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue QZ_IN = Op.getOperand(2);
    SDValue QX = Op.getOperand(3);
    SDValue QY = Op.getOperand(4);
    SDValue RS1 = Op.getOperand(5);
    SDValue SEL4 = Op.getOperand(6);
    SDValue Sar = Op.getOperand(7); // SAR parameter (explicit state passing)

    EVT PtrVT = RS1.getValueType();
    SDVTList VTs = DAG.getVTList(MVT::v8i16, MVT::v16i8, PtrVT, MVT::Other);
    SDValue Ops[] = {Chain, QZ_IN, QX, QY, RS1, SEL4, Sar};
    EVT MemVT = MVT::v16i8;

    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_CMUL_S16_LD_INCP_M, DL,
                                           VTs, Ops, MemVT, MMO);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1),
                               Node.getValue(2), Node.getValue(3)},
                              DL);
  }
  case Intrinsic::riscv_esp_cmul_s8_ld_incp_m: {
    // Lower CMUL S8 LD INCP intrinsic to custom SDNode with explicit SAR state
    // passing Intrinsic: (chain, int_id, qz_in, qx, qy, rs1, sel4, sar)
    // Returns: {qz, qu, ptr}
    // SDNode: (chain, qz_in, qx, qy, rs1, sel4, sar) -> (qz, qu, rs1r, chain)
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue QZ_IN = Op.getOperand(2);
    SDValue QX = Op.getOperand(3);
    SDValue QY = Op.getOperand(4);
    SDValue RS1 = Op.getOperand(5);
    SDValue SEL4 = Op.getOperand(6);
    SDValue Sar = Op.getOperand(7); // SAR parameter (explicit state passing)

    EVT PtrVT = RS1.getValueType();
    SDVTList VTs = DAG.getVTList(MVT::v16i8, MVT::v16i8, PtrVT, MVT::Other);
    SDValue Ops[] = {Chain, QZ_IN, QX, QY, RS1, SEL4, Sar};
    EVT MemVT = MVT::v16i8;

    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_CMUL_S8_LD_INCP_M, DL,
                                           VTs, Ops, MemVT, MMO);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1),
                               Node.getValue(2), Node.getValue(3)},
                              DL);
  }
  case Intrinsic::riscv_esp_cmul_u16_ld_incp_m: {
    // Lower CMUL U16 LD INCP intrinsic to custom SDNode with explicit SAR state
    // passing Intrinsic: (chain, int_id, qz_in, qx, qy, rs1, sel4, sar)
    // Returns: {qz, qu, ptr}
    // SDNode: (chain, qz_in, qx, qy, rs1, sel4, sar) -> (qz, qu, rs1r, chain)
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue QZ_IN = Op.getOperand(2);
    SDValue QX = Op.getOperand(3);
    SDValue QY = Op.getOperand(4);
    SDValue RS1 = Op.getOperand(5);
    SDValue SEL4 = Op.getOperand(6);
    SDValue Sar = Op.getOperand(7); // SAR parameter (explicit state passing)

    EVT PtrVT = RS1.getValueType();
    SDVTList VTs = DAG.getVTList(MVT::v8i16, MVT::v16i8, PtrVT, MVT::Other);
    SDValue Ops[] = {Chain, QZ_IN, QX, QY, RS1, SEL4, Sar};
    EVT MemVT = MVT::v16i8;

    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_CMUL_U16_LD_INCP_M, DL,
                                           VTs, Ops, MemVT, MMO);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1),
                               Node.getValue(2), Node.getValue(3)},
                              DL);
  }
  case Intrinsic::riscv_esp_cmul_u8_ld_incp_m: {
    // Lower CMUL U8 LD INCP intrinsic to custom SDNode with explicit SAR state
    // passing Intrinsic: (chain, int_id, qz_in, qx, qy, rs1, sel4, sar)
    // Returns: {qz, qu, ptr}
    // SDNode: (chain, qz_in, qx, qy, rs1, sel4, sar) -> (qz, qu, rs1r, chain)
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue QZ_IN = Op.getOperand(2);
    SDValue QX = Op.getOperand(3);
    SDValue QY = Op.getOperand(4);
    SDValue RS1 = Op.getOperand(5);
    SDValue SEL4 = Op.getOperand(6);
    SDValue Sar = Op.getOperand(7); // SAR parameter (explicit state passing)

    EVT PtrVT = RS1.getValueType();
    SDVTList VTs = DAG.getVTList(MVT::v16i8, MVT::v16i8, PtrVT, MVT::Other);
    SDValue Ops[] = {Chain, QZ_IN, QX, QY, RS1, SEL4, Sar};
    EVT MemVT = MVT::v16i8;

    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_CMUL_U8_LD_INCP_M, DL,
                                           VTs, Ops, MemVT, MMO);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1),
                               Node.getValue(2), Node.getValue(3)},
                              DL);
  }
  case Intrinsic::riscv_esp_cmul_s16_st_incp_m: {
    // Lower CMUL S16 ST INCP intrinsic to custom SDNode with explicit SAR state
    // passing Intrinsic: (chain, int_id, qz_in, qx, qy, qu, rs1, sel4, sar)
    // Returns: {qz, ptr}
    // SDNode: (chain, qz_in, qx, qy, qu, rs1, sel4, sar) -> (qz, rs1r, chain)
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue QZ_IN = Op.getOperand(2);
    SDValue QX = Op.getOperand(3);
    SDValue QY = Op.getOperand(4);
    SDValue QU = Op.getOperand(5);
    SDValue RS1 = Op.getOperand(6);
    SDValue SEL4 = Op.getOperand(7);
    SDValue Sar = Op.getOperand(8); // SAR parameter (explicit state passing)

    EVT PtrVT = RS1.getValueType();
    SDVTList VTs = DAG.getVTList(MVT::v8i16, PtrVT, MVT::Other);
    SDValue Ops[] = {Chain, QZ_IN, QX, QY, QU, RS1, SEL4, Sar};
    EVT MemVT = MVT::v16i8;

    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_CMUL_S16_ST_INCP_M, DL,
                                           VTs, Ops, MemVT, MMO);
    return DAG.getMergeValues(
        {Node.getValue(0), Node.getValue(1), Node.getValue(2)}, DL);
  }
  case Intrinsic::riscv_esp_cmul_s8_st_incp_m: {
    // Lower CMUL S8 ST INCP intrinsic to custom SDNode with explicit SAR state
    // passing Intrinsic: (chain, int_id, qz_in, qx, qy, qu, rs1, sel4, sar)
    // Returns: {qz, ptr}
    // SDNode: (chain, qz_in, qx, qy, qu, rs1, sel4, sar) -> (qz, rs1r, chain)
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue QZ_IN = Op.getOperand(2);
    SDValue QX = Op.getOperand(3);
    SDValue QY = Op.getOperand(4);
    SDValue QU = Op.getOperand(5);
    SDValue RS1 = Op.getOperand(6);
    SDValue SEL4 = Op.getOperand(7);
    SDValue Sar = Op.getOperand(8); // SAR parameter (explicit state passing)

    EVT PtrVT = RS1.getValueType();
    SDVTList VTs = DAG.getVTList(MVT::v16i8, PtrVT, MVT::Other);
    SDValue Ops[] = {Chain, QZ_IN, QX, QY, QU, RS1, SEL4, Sar};
    EVT MemVT = MVT::v16i8;

    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_CMUL_S8_ST_INCP_M, DL,
                                           VTs, Ops, MemVT, MMO);
    return DAG.getMergeValues(
        {Node.getValue(0), Node.getValue(1), Node.getValue(2)}, DL);
  }
  case Intrinsic::riscv_esp_cmul_u16_st_incp_m: {
    // Lower CMUL U16 ST INCP intrinsic to custom SDNode with explicit SAR state
    // passing Intrinsic: (chain, int_id, qz_in, qx, qy, qu, rs1, sel4, sar)
    // Returns: {qz, ptr}
    // SDNode: (chain, qz_in, qx, qy, qu, rs1, sel4, sar) -> (qz, rs1r, chain)
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue QZ_IN = Op.getOperand(2);
    SDValue QX = Op.getOperand(3);
    SDValue QY = Op.getOperand(4);
    SDValue QU = Op.getOperand(5);
    SDValue RS1 = Op.getOperand(6);
    SDValue SEL4 = Op.getOperand(7);
    SDValue Sar = Op.getOperand(8); // SAR parameter (explicit state passing)

    EVT PtrVT = RS1.getValueType();
    SDVTList VTs = DAG.getVTList(MVT::v8i16, PtrVT, MVT::Other);
    SDValue Ops[] = {Chain, QZ_IN, QX, QY, QU, RS1, SEL4, Sar};
    EVT MemVT = MVT::v16i8;

    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_CMUL_U16_ST_INCP_M, DL,
                                           VTs, Ops, MemVT, MMO);
    return DAG.getMergeValues(
        {Node.getValue(0), Node.getValue(1), Node.getValue(2)}, DL);
  }
  case Intrinsic::riscv_esp_cmul_u8_st_incp_m: {
    // Lower CMUL U8 ST INCP intrinsic to custom SDNode with explicit SAR state
    // passing Intrinsic: (chain, int_id, qz_in, qx, qy, qu, rs1, sel4, sar)
    // Returns: {qz, ptr}
    // SDNode: (chain, qz_in, qx, qy, qu, rs1, sel4, sar) -> (qz, rs1r, chain)
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue QZ_IN = Op.getOperand(2);
    SDValue QX = Op.getOperand(3);
    SDValue QY = Op.getOperand(4);
    SDValue QU = Op.getOperand(5);
    SDValue RS1 = Op.getOperand(6);
    SDValue SEL4 = Op.getOperand(7);
    SDValue Sar = Op.getOperand(8); // SAR parameter (explicit state passing)

    EVT PtrVT = RS1.getValueType();
    SDVTList VTs = DAG.getVTList(MVT::v16i8, PtrVT, MVT::Other);
    SDValue Ops[] = {Chain, QZ_IN, QX, QY, QU, RS1, SEL4, Sar};
    EVT MemVT = MVT::v16i8;

    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_CMUL_U8_ST_INCP_M, DL,
                                           VTs, Ops, MemVT, MMO);
    return DAG.getMergeValues(
        {Node.getValue(0), Node.getValue(1), Node.getValue(2)}, DL);
  }
  case Intrinsic::riscv_esp_ld_qacc_h_h_128_ip_m: {
    // Lower intrinsic to custom SDNode
    // Intrinsic: (chain, int_id, ptr, imm) -> (v16i8, ptr, chain)
    // Subregister model: returns loaded 128-bit data (QACC_H[255:128])
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue Ptr = Op.getOperand(2);
    SDValue Imm = Op.getOperand(3);

    EVT VecVT = MVT::v16i8; // 128-bit subregister
    EVT PtrVT = Ptr.getValueType();
    SDVTList VTs = DAG.getVTList(VecVT, PtrVT, MVT::Other);

    SDValue Ops[] = {Chain, Ptr, Imm};
    // Note: This intrinsic always arrives as MemIntrinsicSDNode because
    //       getTgtMemIntrinsic returns true for it.
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_LD_QACC_H_H_128_IP_M,
                                           DL, VTs, Ops, VecVT, MMO);
    return DAG.getMergeValues(
        {Node.getValue(0), Node.getValue(1), Node.getValue(2)}, DL);
  }
  case Intrinsic::riscv_esp_ld_qacc_h_l_128_ip_m: {
    // Lower intrinsic to custom SDNode
    // Intrinsic: (chain, int_id, ptr, imm) -> (v16i8, ptr, chain)
    // Subregister model: returns loaded 128-bit data (QACC_H[127:0])
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue Ptr = Op.getOperand(2);
    SDValue Imm = Op.getOperand(3);

    EVT VecVT = MVT::v16i8; // 128-bit subregister
    EVT PtrVT = Ptr.getValueType();
    SDVTList VTs = DAG.getVTList(VecVT, PtrVT, MVT::Other);

    SDValue Ops[] = {Chain, Ptr, Imm};
    // Note: This intrinsic always arrives as MemIntrinsicSDNode because
    //       getTgtMemIntrinsic returns true for it.
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_LD_QACC_H_L_128_IP_M,
                                           DL, VTs, Ops, VecVT, MMO);
    return DAG.getMergeValues(
        {Node.getValue(0), Node.getValue(1), Node.getValue(2)}, DL);
  }
  case Intrinsic::riscv_esp_ld_qacc_l_h_128_ip_m: {
    // Lower intrinsic to custom SDNode
    // Intrinsic: (chain, int_id, ptr, imm) -> (v16i8, ptr, chain)
    // Subregister model: returns loaded 128-bit data (QACC_L[255:128])
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue Ptr = Op.getOperand(2);
    SDValue Imm = Op.getOperand(3);

    EVT VecVT = MVT::v16i8; // 128-bit subregister
    EVT PtrVT = Ptr.getValueType();
    SDVTList VTs = DAG.getVTList(VecVT, PtrVT, MVT::Other);

    SDValue Ops[] = {Chain, Ptr, Imm};
    // Note: This intrinsic always arrives as MemIntrinsicSDNode because
    //       getTgtMemIntrinsic returns true for it.
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_LD_QACC_L_H_128_IP_M,
                                           DL, VTs, Ops, VecVT, MMO);
    return DAG.getMergeValues(
        {Node.getValue(0), Node.getValue(1), Node.getValue(2)}, DL);
  }
  case Intrinsic::riscv_esp_ld_qacc_l_l_128_ip_m: {
    // Lower intrinsic to custom SDNode
    // Intrinsic: (chain, int_id, ptr, imm) -> (v16i8, ptr, chain)
    // Subregister model: returns loaded 128-bit data (QACC_L[127:0])
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue Ptr = Op.getOperand(2);
    SDValue Imm = Op.getOperand(3);

    EVT VecVT = MVT::v16i8; // 128-bit subregister
    EVT PtrVT = Ptr.getValueType();
    SDVTList VTs = DAG.getVTList(VecVT, PtrVT, MVT::Other);

    SDValue Ops[] = {Chain, Ptr, Imm};
    // Note: This intrinsic always arrives as MemIntrinsicSDNode because
    //       getTgtMemIntrinsic returns true for it.
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_LD_QACC_L_L_128_IP_M,
                                           DL, VTs, Ops, VecVT, MMO);
    return DAG.getMergeValues(
        {Node.getValue(0), Node.getValue(1), Node.getValue(2)}, DL);
  }
  // LDQA IP
  case Intrinsic::riscv_esp_ldqa_s16_128_ip_m:
    return LowerLDQAIP(Op, DAG, RISCVISD::ESP_LDQA_S16_128_IP_M);
  case Intrinsic::riscv_esp_ldqa_s8_128_ip_m:
    return LowerLDQAIP(Op, DAG, RISCVISD::ESP_LDQA_S8_128_IP_M);
  case Intrinsic::riscv_esp_ldqa_u16_128_ip_m:
    return LowerLDQAIP(Op, DAG, RISCVISD::ESP_LDQA_U16_128_IP_M);
  case Intrinsic::riscv_esp_ldqa_u8_128_ip_m:
    return LowerLDQAIP(Op, DAG, RISCVISD::ESP_LDQA_U8_128_IP_M);
  // LDQA XP
  case Intrinsic::riscv_esp_ldqa_s16_128_xp_m:
    return LowerLDQAXP(Op, DAG, RISCVISD::ESP_LDQA_S16_128_XP_M);
  case Intrinsic::riscv_esp_ldqa_s8_128_xp_m:
    return LowerLDQAXP(Op, DAG, RISCVISD::ESP_LDQA_S8_128_XP_M);
  case Intrinsic::riscv_esp_ldqa_u16_128_xp_m:
    return LowerLDQAXP(Op, DAG, RISCVISD::ESP_LDQA_U16_128_XP_M);
  case Intrinsic::riscv_esp_ldqa_u8_128_xp_m:
    return LowerLDQAXP(Op, DAG, RISCVISD::ESP_LDQA_U8_128_XP_M);
  // ST QACC_H/QACC_L IP
  case Intrinsic::riscv_esp_st_qacc_h_h_128_ip_m: {
    // Lower intrinsic to custom SDNode
    // Intrinsic: (chain, int_id, qacc_h_high (v16i8, 128-bit), ptr, imm)
    // First principle: directly accept 128-bit value, matching hardware
    // operation
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue QACCH_High = Op.getOperand(2); // QACC_H[255:128] (v16i8, 128-bit)
    SDValue Ptr = Op.getOperand(3);
    SDValue Imm = Op.getOperand(4);

    EVT VecVT = MVT::v16i8; // 128-bit
    EVT PtrVT = Ptr.getValueType();
    SDVTList VTs = DAG.getVTList(PtrVT, MVT::Other);

    SDValue Ops[] = {Chain, QACCH_High, Ptr, Imm};
    // Note: This intrinsic always arrives as MemIntrinsicSDNode because
    //       getTgtMemIntrinsic returns true for it.
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_ST_QACC_H_H_128_IP_M,
                                           DL, VTs, Ops, VecVT, MMO);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1)}, DL);
  }
  case Intrinsic::riscv_esp_st_qacc_h_l_128_ip_m: {
    // Lower intrinsic to custom SDNode
    // Intrinsic: (chain, int_id, qacc_h_low (v16i8, 128-bit), ptr, imm)
    // First principle: directly accept 128-bit value, matching hardware
    // operation
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue QACCH_Low = Op.getOperand(2); // QACC_H[127:0] (v16i8, 128-bit)
    SDValue Ptr = Op.getOperand(3);
    SDValue Imm = Op.getOperand(4);

    EVT VecVT = MVT::v16i8; // 128-bit
    EVT PtrVT = Ptr.getValueType();
    SDVTList VTs = DAG.getVTList(PtrVT, MVT::Other);

    SDValue Ops[] = {Chain, QACCH_Low, Ptr, Imm};
    // Note: This intrinsic always arrives as MemIntrinsicSDNode because
    //       getTgtMemIntrinsic returns true for it.
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_ST_QACC_H_L_128_IP_M,
                                           DL, VTs, Ops, VecVT, MMO);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1)}, DL);
  }
  case Intrinsic::riscv_esp_st_qacc_l_h_128_ip_m: {
    // Lower intrinsic to custom SDNode
    // Intrinsic: (chain, int_id, qacc_l_high (v16i8, 128-bit), ptr, imm)
    // First principle: directly accept 128-bit value, matching hardware
    // operation
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue QACCL_High =
        Op.getOperand(2); // QACC_L_HIGH[255:128] (v16i8, 128-bit)
    SDValue Ptr = Op.getOperand(3);
    SDValue Imm = Op.getOperand(4);

    EVT VecVT = MVT::v16i8; // 128-bit
    EVT PtrVT = Ptr.getValueType();
    SDVTList VTs = DAG.getVTList(PtrVT, MVT::Other);

    SDValue Ops[] = {Chain, QACCL_High, Ptr, Imm};
    // Note: This intrinsic always arrives as MemIntrinsicSDNode because
    //       getTgtMemIntrinsic returns true for it.
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_ST_QACC_L_H_128_IP_M,
                                           DL, VTs, Ops, VecVT, MMO);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1)}, DL);
  }
  case Intrinsic::riscv_esp_st_qacc_l_l_128_ip_m: {
    // Lower intrinsic to custom SDNode
    // Intrinsic: (chain, int_id, qacc_l_low (v16i8, 128-bit), ptr, imm)
    // First principle: directly accept 128-bit value, matching hardware
    // operation
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue QACCL_Low = Op.getOperand(2); // QACC_L_LOW (v16i8, 128-bit)
    SDValue Ptr = Op.getOperand(3);
    SDValue Imm = Op.getOperand(4);

    EVT VecVT = MVT::v16i8; // 128-bit
    EVT PtrVT = Ptr.getValueType();
    SDVTList VTs = DAG.getVTList(PtrVT, MVT::Other);

    SDValue Ops[] = {Chain, QACCL_Low, Ptr, Imm};

    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_ST_QACC_L_L_128_IP_M,
                                           DL, VTs, Ops, VecVT, MMO);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1)}, DL);
  }
  case Intrinsic::riscv_esp_vadd_s8_ld_incp_m: {
    // Lower VADD S8 LD INCP intrinsic to custom SDNode that will be matched
    // directly to instruction Intrinsic: (chain, int_id, qx, qy, rs1) Returns:
    // {qv, qu, ptr} SDNode: (chain, qx, qy, rs1) -> (qv, qu, rs1r, chain)
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue QX = Op.getOperand(2);
    SDValue QY = Op.getOperand(3);
    SDValue RS1 = Op.getOperand(4);

    EVT PtrVT = RS1.getValueType();
    SDVTList VTs = DAG.getVTList(MVT::v16i8, MVT::v16i8, PtrVT, MVT::Other);
    SDValue Ops[] = {Chain, QX, QY, RS1};
    EVT MemVT = MVT::v16i8;
    // Note: This intrinsic always arrives as MemIntrinsicSDNode because
    //       getTgtMemIntrinsic returns true for it.
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_VADD_S8_LD_INCP_M, DL,
                                           VTs, Ops, MemVT, MMO);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1),
                               Node.getValue(2), Node.getValue(3)},
                              DL);
  }
  // VADD LD.INCP lowering (s16, u16, u8)
  case Intrinsic::riscv_esp_vadd_s16_ld_incp_m: {
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue QX = Op.getOperand(2);
    SDValue QY = Op.getOperand(3);
    SDValue RS1 = Op.getOperand(4);
    EVT PtrVT = RS1.getValueType();
    SDVTList VTs = DAG.getVTList(MVT::v8i16, MVT::v16i8, PtrVT, MVT::Other);
    SDValue Ops[] = {Chain, QX, QY, RS1};
    EVT MemVT = MVT::v16i8;
    // Note: This intrinsic always arrives as MemIntrinsicSDNode because
    //       getTgtMemIntrinsic returns true for it.
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_VADD_S16_LD_INCP_M, DL,
                                           VTs, Ops, MemVT, MMO);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1),
                               Node.getValue(2), Node.getValue(3)},
                              DL);
  }
  case Intrinsic::riscv_esp_vadd_u16_ld_incp_m: {
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue QX = Op.getOperand(2);
    SDValue QY = Op.getOperand(3);
    SDValue RS1 = Op.getOperand(4);
    EVT PtrVT = RS1.getValueType();
    SDVTList VTs = DAG.getVTList(MVT::v8i16, MVT::v16i8, PtrVT, MVT::Other);
    SDValue Ops[] = {Chain, QX, QY, RS1};
    EVT MemVT = MVT::v16i8;
    // Note: This intrinsic always arrives as MemIntrinsicSDNode because
    //       getTgtMemIntrinsic returns true for it.
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_VADD_U16_LD_INCP_M, DL,
                                           VTs, Ops, MemVT, MMO);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1),
                               Node.getValue(2), Node.getValue(3)},
                              DL);
  }
  case Intrinsic::riscv_esp_vadd_u8_ld_incp_m: {
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue QX = Op.getOperand(2);
    SDValue QY = Op.getOperand(3);
    SDValue RS1 = Op.getOperand(4);
    EVT PtrVT = RS1.getValueType();
    SDVTList VTs = DAG.getVTList(MVT::v16i8, MVT::v16i8, PtrVT, MVT::Other);
    SDValue Ops[] = {Chain, QX, QY, RS1};
    EVT MemVT = MVT::v16i8;
    // Note: This intrinsic always arrives as MemIntrinsicSDNode because
    //       getTgtMemIntrinsic returns true for it.
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_VADD_U8_LD_INCP_M, DL,
                                           VTs, Ops, MemVT, MMO);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1),
                               Node.getValue(2), Node.getValue(3)},
                              DL);
  }
  // VADD ST.INCP lowering
  case Intrinsic::riscv_esp_vadd_s8_st_incp_m: {
    // Lower VADD S8 ST INCP intrinsic to custom SDNode
    // Intrinsic: (chain, int_id, qx, qy, qu, rs1, qv)
    // Returns: {qv, ptr}
    // SDNode: (chain, qx, qy, qu, rs1) -> (qv, rs1r, chain)
    // Note: qv is input to intrinsic (for register selection) but output from
    // instruction
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue QX = Op.getOperand(2);
    SDValue QY = Op.getOperand(3);
    SDValue QU = Op.getOperand(4);
    SDValue RS1 = Op.getOperand(5);
    // QV is passed as input to intrinsic but is output from instruction
    // The instruction will compute qv, so we don't pass it to SDNode
    EVT PtrVT = RS1.getValueType();
    SDVTList VTs = DAG.getVTList(MVT::v16i8, PtrVT, MVT::Other);
    SDValue Ops[] = {Chain, QX, QY, QU, RS1};
    EVT MemVT = MVT::v16i8;
    // Note: This intrinsic always arrives as MemIntrinsicSDNode because
    //       getTgtMemIntrinsic returns true for it.
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_VADD_S8_ST_INCP_M, DL,
                                           VTs, Ops, MemVT, MMO);
    return DAG.getMergeValues(
        {Node.getValue(0), Node.getValue(1), Node.getValue(2)}, DL);
  }
  case Intrinsic::riscv_esp_vadd_s16_st_incp_m: {
    // Lower VADD S16 ST INCP intrinsic to custom SDNode
    // Intrinsic: (chain, int_id, qx, qy, qu, rs1, qv)
    // Returns: {qv, ptr}
    // SDNode: (chain, qx, qy, qu, rs1) -> (qv, rs1r, chain)
    // Note: qv is input to intrinsic (for register selection) but output from
    // instruction
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue QX = Op.getOperand(2);
    SDValue QY = Op.getOperand(3);
    SDValue QU = Op.getOperand(4);
    SDValue RS1 = Op.getOperand(5);
    // QV is passed as input to intrinsic but is output from instruction
    // The instruction will compute qv, so we don't pass it to SDNode
    EVT PtrVT = RS1.getValueType();
    SDVTList VTs = DAG.getVTList(MVT::v8i16, PtrVT, MVT::Other);
    SDValue Ops[] = {Chain, QX, QY, QU, RS1};
    EVT MemVT = MVT::v16i8;
    // Note: This intrinsic always arrives as MemIntrinsicSDNode because
    //       getTgtMemIntrinsic returns true for it.
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_VADD_S16_ST_INCP_M, DL,
                                           VTs, Ops, MemVT, MMO);
    return DAG.getMergeValues(
        {Node.getValue(0), Node.getValue(1), Node.getValue(2)}, DL);
  }
  case Intrinsic::riscv_esp_vadd_u16_st_incp_m: {
    // Lower VADD U16 ST INCP intrinsic to custom SDNode
    // Intrinsic: (chain, int_id, qx, qy, qu, rs1, qv)
    // Returns: {qv, ptr}
    // SDNode: (chain, qx, qy, qu, rs1) -> (qv, rs1r, chain)
    // Note: qv is input to intrinsic (for register selection) but output from
    // instruction
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue QX = Op.getOperand(2);
    SDValue QY = Op.getOperand(3);
    SDValue QU = Op.getOperand(4);
    SDValue RS1 = Op.getOperand(5);
    // QV is passed as input to intrinsic but is output from instruction
    // The instruction will compute qv, so we don't pass it to SDNode
    EVT PtrVT = RS1.getValueType();
    SDVTList VTs = DAG.getVTList(MVT::v8i16, PtrVT, MVT::Other);
    SDValue Ops[] = {Chain, QX, QY, QU, RS1};
    EVT MemVT = MVT::v16i8;
    // Note: This intrinsic always arrives as MemIntrinsicSDNode because
    //       getTgtMemIntrinsic returns true for it.
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_VADD_U16_ST_INCP_M, DL,
                                           VTs, Ops, MemVT, MMO);
    return DAG.getMergeValues(
        {Node.getValue(0), Node.getValue(1), Node.getValue(2)}, DL);
  }
  case Intrinsic::riscv_esp_vadd_u8_st_incp_m: {
    // Lower VADD U8 ST INCP intrinsic to custom SDNode
    // Intrinsic: (chain, int_id, qx, qy, qu, rs1, qv)
    // Returns: {qv, ptr}
    // SDNode: (chain, qx, qy, qu, rs1) -> (qv, rs1r, chain)
    // Note: qv is input to intrinsic (for register selection) but output from
    // instruction
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue QX = Op.getOperand(2);
    SDValue QY = Op.getOperand(3);
    SDValue QU = Op.getOperand(4);
    SDValue RS1 = Op.getOperand(5);
    // QV is passed as input to intrinsic but is output from instruction
    // The instruction will compute qv, so we don't pass it to SDNode
    EVT PtrVT = RS1.getValueType();
    SDVTList VTs = DAG.getVTList(MVT::v16i8, PtrVT, MVT::Other);
    SDValue Ops[] = {Chain, QX, QY, QU, RS1};
    EVT MemVT = MVT::v16i8;
    // Note: This intrinsic always arrives as MemIntrinsicSDNode because
    //       getTgtMemIntrinsic returns true for it.
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_VADD_U8_ST_INCP_M, DL,
                                           VTs, Ops, MemVT, MMO);
    return DAG.getMergeValues(
        {Node.getValue(0), Node.getValue(1), Node.getValue(2)}, DL);
  }

  case Intrinsic::riscv_esp_fft_ams_s16_ld_incp_uaup_m: {
    // Lower FFT AMS S16 LD INCP UAUP intrinsic to custom SDNode with explicit
    // phantom operands Intrinsic: (chain, int_id, qx, qy, qw, rs1, sel2,
    // ua_state_in, sar_bytes_in, sar_in) Returns: {qu, qz, qv, ptr,
    // ua_state_out} SDNode: (chain, qx, qy, qw, rs1, sel2, ua_state_in,
    // sar_bytes_in, sar_in) -> (qu, qz, qv, rs1r, ua_state_out, chain) Note:
    // UA_STATE is input/output (phantom operand), SAR_BYTES and SAR are
    // input-only (phantom operands)
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue QX = Op.getOperand(2);
    SDValue QY = Op.getOperand(3);
    SDValue QW = Op.getOperand(4);
    SDValue RS1 = Op.getOperand(5);
    SDValue SEL2 = Op.getOperand(6);
    SDValue UAStateIn = Op.getOperand(7); // UA_STATE input (phantom operand)
    SDValue SarBytesIn =
        Op.getOperand(8); // SAR_BYTES input (phantom operand, read-only)
    SDValue SarIn = Op.getOperand(9); // SAR input (phantom operand, read-only)

    EVT PtrVT = RS1.getValueType();
    SmallVector<EVT, 6> VTs = {MVT::v16i8, MVT::v8i16, MVT::v8i16,
                               PtrVT,      MVT::v16i8, MVT::Other};
    SDVTList VTList = DAG.getVTList(VTs);
    SDValue Ops[] = {Chain, QX,        QY,         QW,   RS1,
                     SEL2,  UAStateIn, SarBytesIn, SarIn};
    EVT MemVT = MVT::v16i8;
    // Note: This intrinsic always arrives as MemIntrinsicSDNode because
    //       getTgtMemIntrinsic returns true for it.
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(
        RISCVISD::ESP_FFT_AMS_S16_LD_INCP_UAUP_M, DL, VTList, Ops, MemVT, MMO);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1),
                               Node.getValue(2), Node.getValue(3),
                               Node.getValue(4), Node.getValue(5)},
                              DL);
  }

  case Intrinsic::riscv_esp_fft_r2bf_s16_st_incp_m: {
    // Lower FFT R2BF S16 ST INCP intrinsic to custom SDNode
    // Intrinsic: (chain, int_id, qx, qy, rs1, sel4)
    // Returns: {qz, ptr}
    // SDNode: (chain, qx, qy, rs1, sel4) -> (qz, rs1r, chain)
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue QX = Op.getOperand(2);
    SDValue QY = Op.getOperand(3);
    SDValue RS1 = Op.getOperand(4);
    SDValue SEL4 = Op.getOperand(5);

    EVT PtrVT = RS1.getValueType();
    SmallVector<EVT, 3> VTs = {MVT::v8i16, PtrVT, MVT::Other};
    SDVTList VTList = DAG.getVTList(VTs);
    SDValue Ops[] = {Chain, QX, QY, RS1, SEL4};
    EVT MemVT = MVT::v16i8;
    // Note: This intrinsic always arrives as MemIntrinsicSDNode because
    //       getTgtMemIntrinsic returns true for it.
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_FFT_R2BF_S16_ST_INCP_M,
                                           DL, VTList, Ops, MemVT, MMO);
    return DAG.getMergeValues(
        {Node.getValue(0), Node.getValue(1), Node.getValue(2)}, DL);
  }
  case Intrinsic::riscv_esp_fft_ams_s16_ld_incp_m: {
    // Lower FFT AMS S16 LD INCP intrinsic to custom SDNode with explicit SAR
    // state passing Intrinsic: (chain, int_id, qx, qy, qw, rs1, sel2, sar)
    // Returns: {qu, qz, qv, ptr}
    // SDNode: (chain, qx, qy, qw, rs1, sel2, sar) -> (qu, qz, qv, rs1r, chain)
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue QX = Op.getOperand(2);
    SDValue QY = Op.getOperand(3);
    SDValue QW = Op.getOperand(4);
    SDValue RS1 = Op.getOperand(5);
    SDValue SEL2 = Op.getOperand(6);
    SDValue Sar = Op.getOperand(7); // SAR parameter (explicit state passing)

    EVT PtrVT = RS1.getValueType();
    SmallVector<EVT, 5> VTs = {MVT::v16i8, MVT::v8i16, MVT::v8i16, PtrVT,
                               MVT::Other};
    SDVTList VTList = DAG.getVTList(VTs);
    SDValue Ops[] = {Chain, QX, QY, QW, RS1, SEL2, Sar};
    EVT MemVT = MVT::v16i8;
    // Note: This intrinsic always arrives as MemIntrinsicSDNode because
    //       getTgtMemIntrinsic returns true for it.
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_FFT_AMS_S16_LD_INCP_M,
                                           DL, VTList, Ops, MemVT, MMO);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1),
                               Node.getValue(2), Node.getValue(3),
                               Node.getValue(4)},
                              DL);
  }

  case Intrinsic::riscv_esp_fft_ams_s16_ld_r32_decp_m: {
    // Lower FFT AMS S16 LD R32 DECP intrinsic to custom SDNode with explicit
    // SAR state passing Intrinsic: (chain, int_id, qx, qy, qw, rs1, sel2, sar)
    // Returns: {qu, qz, qv, ptr}
    // SDNode: (chain, qx, qy, qw, rs1, sel2, sar) -> (qu, qz, qv, rs1r, chain)
    // Note: Same as LD.INCP but decrements pointer instead of incrementing
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue QX = Op.getOperand(2);
    SDValue QY = Op.getOperand(3);
    SDValue QW = Op.getOperand(4);
    SDValue RS1 = Op.getOperand(5);
    SDValue SEL2 = Op.getOperand(6);
    SDValue Sar = Op.getOperand(7); // SAR parameter (explicit state passing)

    EVT PtrVT = RS1.getValueType();
    SmallVector<EVT, 5> VTs = {MVT::v16i8, MVT::v8i16, MVT::v8i16, PtrVT,
                               MVT::Other};
    SDVTList VTList = DAG.getVTList(VTs);
    SDValue Ops[] = {Chain, QX, QY, QW, RS1, SEL2, Sar};
    EVT MemVT = MVT::v16i8;
    // Note: This intrinsic always arrives as MemIntrinsicSDNode because
    //       getTgtMemIntrinsic returns true for it.
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(
        RISCVISD::ESP_FFT_AMS_S16_LD_R32_DECP_M, DL, VTList, Ops, MemVT, MMO);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1),
                               Node.getValue(2), Node.getValue(3),
                               Node.getValue(4)},
                              DL);
  }

  case Intrinsic::riscv_esp_fft_ams_s16_st_incp_m: {
    // Lower FFT AMS S16 ST INCP intrinsic to custom SDNode with explicit SAR
    // state passing Intrinsic: (chain, int_id, qx, qy, qw, qu, rs1, rs2, sel2,
    // sar) Returns: {qz, ptr} SDNode: (chain, qx, qy, qw, qu, rs1, rs2, sel2,
    // sar) -> (qz, rs1r, chain) Note: rs2 is updated with computation result,
    // handled at instruction level
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue QX = Op.getOperand(2);
    SDValue QY = Op.getOperand(3);
    SDValue QW = Op.getOperand(4);
    SDValue QU = Op.getOperand(5);
    SDValue RS1 = Op.getOperand(6);
    SDValue RS2 = Op.getOperand(7);
    SDValue SEL2 = Op.getOperand(8);
    SDValue Sar = Op.getOperand(9); // SAR parameter (explicit state passing)

    EVT PtrVT = RS1.getValueType();
    SmallVector<EVT, 3> VTs = {MVT::v8i16, PtrVT, MVT::Other};
    SDVTList VTList = DAG.getVTList(VTs);
    SDValue Ops[] = {Chain, QX, QY, QW, QU, RS1, RS2, SEL2, Sar};
    EVT MemVT = MVT::v16i8;
    // Note: This intrinsic always arrives as MemIntrinsicSDNode because
    //       getTgtMemIntrinsic returns true for it.
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_FFT_AMS_S16_ST_INCP_M,
                                           DL, VTList, Ops, MemVT, MMO);
    return DAG.getMergeValues(
        {Node.getValue(0), Node.getValue(1), Node.getValue(2)}, DL);
  }

  case Intrinsic::riscv_esp_fft_cmul_s16_ld_xp_m: {
    // Lower FFT CMUL S16 LD XP intrinsic to custom SDNode with explicit SAR
    // state passing Intrinsic: (chain, int_id, qx, qy, rs1, rs2, sel8, sar)
    // Returns: {qz, qu, ptr}
    // SDNode: (chain, qx, qy, rs1, rs2, sel8, sar) -> (qz, qu, rs1r, chain)
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue QX = Op.getOperand(2);
    SDValue QY = Op.getOperand(3);
    SDValue RS1 = Op.getOperand(4);
    SDValue RS2 = Op.getOperand(5);
    SDValue SEL8 = Op.getOperand(6);
    SDValue Sar = Op.getOperand(7); // SAR parameter (explicit state passing)

    EVT PtrVT = RS1.getValueType();
    SmallVector<EVT, 4> VTs = {MVT::v8i16, MVT::v16i8, PtrVT, MVT::Other};
    SDVTList VTList = DAG.getVTList(VTs);
    SDValue Ops[] = {Chain, QX, QY, RS1, RS2, SEL8, Sar};
    EVT MemVT = MVT::v16i8;
    // Note: This intrinsic always arrives as MemIntrinsicSDNode because
    //       getTgtMemIntrinsic returns true for it.
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_FFT_CMUL_S16_LD_XP_M,
                                           DL, VTList, Ops, MemVT, MMO);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1),
                               Node.getValue(2), Node.getValue(3)},
                              DL);
  }

  case Intrinsic::riscv_esp_fft_cmul_s16_st_xp_m: {
    // Lower FFT CMUL S16 ST XP intrinsic to custom SDNode with explicit SAR
    // state passing Intrinsic: (chain, int_id, qx, qy, qu, rs1, rs2, sel8,
    // upd4, sel4, sar) Returns: {ptr} SDNode: (chain, qx, qy, qu, rs1, rs2,
    // sel8, upd4, sel4, sar) -> (rs1r, chain)
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue QX = Op.getOperand(2);
    SDValue QY = Op.getOperand(3);
    SDValue QU = Op.getOperand(4);
    SDValue RS1 = Op.getOperand(5);
    SDValue RS2 = Op.getOperand(6);
    SDValue SEL8 = Op.getOperand(7);
    SDValue UPD4 = Op.getOperand(8);
    SDValue SEL4 = Op.getOperand(9);
    SDValue Sar = Op.getOperand(10); // SAR parameter (explicit state passing)

    EVT PtrVT = RS1.getValueType();
    SmallVector<EVT, 2> VTs = {PtrVT, MVT::Other};
    SDVTList VTList = DAG.getVTList(VTs);
    SDValue Ops[] = {Chain, QX, QY, QU, RS1, RS2, SEL8, UPD4, SEL4, Sar};
    EVT MemVT = MVT::v16i8;
    // Note: This intrinsic always arrives as MemIntrinsicSDNode because
    //       getTgtMemIntrinsic returns true for it.
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_FFT_CMUL_S16_ST_XP_M,
                                           DL, VTList, Ops, MemVT, MMO);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1)}, DL);
  }

  case Intrinsic::riscv_esp_fft_vst_r32_decp_m: {
    // Lower FFT VST R32 DECP intrinsic to custom SDNode
    // Intrinsic: (chain, int_id, qu, rs1, sel2)
    // Returns: {ptr}
    // SDNode: (chain, qu, rs1, sel2) -> (rs1r, chain)
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue QU = Op.getOperand(2);
    SDValue RS1 = Op.getOperand(3);
    SDValue SEL2 = Op.getOperand(4);

    EVT PtrVT = RS1.getValueType();
    SmallVector<EVT, 2> VTs = {PtrVT, MVT::Other};
    SDVTList VTList = DAG.getVTList(VTs);
    SDValue Ops[] = {Chain, QU, RS1, SEL2};
    EVT MemVT = MVT::v16i8;
    // Note: This intrinsic always arrives as MemIntrinsicSDNode because
    //       getTgtMemIntrinsic returns true for it.
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_FFT_VST_R32_DECP_M, DL,
                                           VTList, Ops, MemVT, MMO);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1)}, DL);
  }
  // VMULAS QACC LD IP
  case Intrinsic::riscv_esp_vmulas_s16_qacc_ld_ip_m:
    return LowerVMULASQACCLDIP(Op, DAG, RISCVISD::ESP_VMULAS_S16_QACC_LD_IP_M);
  case Intrinsic::riscv_esp_vmulas_s8_qacc_ld_ip_m:
    return LowerVMULASQACCLDIP(Op, DAG, RISCVISD::ESP_VMULAS_S8_QACC_LD_IP_M);
  case Intrinsic::riscv_esp_vmulas_u16_qacc_ld_ip_m:
    return LowerVMULASQACCLDIP(Op, DAG, RISCVISD::ESP_VMULAS_U16_QACC_LD_IP_M);
  case Intrinsic::riscv_esp_vmulas_u8_qacc_ld_ip_m:
    return LowerVMULASQACCLDIP(Op, DAG, RISCVISD::ESP_VMULAS_U8_QACC_LD_IP_M);
  // VMULAS QACC LD XP
  case Intrinsic::riscv_esp_vmulas_s16_qacc_ld_xp_m:
    return LowerVMULASQACCLDXP(Op, DAG, RISCVISD::ESP_VMULAS_S16_QACC_LD_XP_M);
  case Intrinsic::riscv_esp_vmulas_s8_qacc_ld_xp_m:
    return LowerVMULASQACCLDXP(Op, DAG, RISCVISD::ESP_VMULAS_S8_QACC_LD_XP_M);
  case Intrinsic::riscv_esp_vmulas_u16_qacc_ld_xp_m:
    return LowerVMULASQACCLDXP(Op, DAG, RISCVISD::ESP_VMULAS_U16_QACC_LD_XP_M);
  case Intrinsic::riscv_esp_vmulas_u8_qacc_ld_xp_m:
    return LowerVMULASQACCLDXP(Op, DAG, RISCVISD::ESP_VMULAS_U8_QACC_LD_XP_M);
  // VMULAS QACC ST IP
  case Intrinsic::riscv_esp_vmulas_s16_qacc_st_ip_m:
    return LowerVMULASQACCSTIP(Op, DAG, RISCVISD::ESP_VMULAS_S16_QACC_ST_IP_M);
  case Intrinsic::riscv_esp_vmulas_s8_qacc_st_ip_m:
    return LowerVMULASQACCSTIP(Op, DAG, RISCVISD::ESP_VMULAS_S8_QACC_ST_IP_M);
  case Intrinsic::riscv_esp_vmulas_u16_qacc_st_ip_m:
    return LowerVMULASQACCSTIP(Op, DAG, RISCVISD::ESP_VMULAS_U16_QACC_ST_IP_M);
  case Intrinsic::riscv_esp_vmulas_u8_qacc_st_ip_m:
    return LowerVMULASQACCSTIP(Op, DAG, RISCVISD::ESP_VMULAS_U8_QACC_ST_IP_M);
  // VMULAS QACC ST XP
  case Intrinsic::riscv_esp_vmulas_s16_qacc_st_xp_m:
    return LowerVMULASQACCSTXP(Op, DAG, RISCVISD::ESP_VMULAS_S16_QACC_ST_XP_M);
  case Intrinsic::riscv_esp_vmulas_s8_qacc_st_xp_m:
    return LowerVMULASQACCSTXP(Op, DAG, RISCVISD::ESP_VMULAS_S8_QACC_ST_XP_M);
  case Intrinsic::riscv_esp_vmulas_u16_qacc_st_xp_m:
    return LowerVMULASQACCSTXP(Op, DAG, RISCVISD::ESP_VMULAS_U16_QACC_ST_XP_M);
  case Intrinsic::riscv_esp_vmulas_u8_qacc_st_xp_m:
    return LowerVMULASQACCSTXP(Op, DAG, RISCVISD::ESP_VMULAS_U8_QACC_ST_XP_M);
  // VMULAS QACC LDBC INCP
  case Intrinsic::riscv_esp_vmulas_s16_qacc_ldbc_incp_m:
    return LowerVMULASQACCLDBCINCP(Op, DAG,
                                   RISCVISD::ESP_VMULAS_S16_QACC_LDBC_INCP_M);
  case Intrinsic::riscv_esp_vmulas_s8_qacc_ldbc_incp_m:
    return LowerVMULASQACCLDBCINCP(Op, DAG,
                                   RISCVISD::ESP_VMULAS_S8_QACC_LDBC_INCP_M);
  case Intrinsic::riscv_esp_vmulas_u16_qacc_ldbc_incp_m:
    return LowerVMULASQACCLDBCINCP(Op, DAG,
                                   RISCVISD::ESP_VMULAS_U16_QACC_LDBC_INCP_M);
  case Intrinsic::riscv_esp_vmulas_u8_qacc_ldbc_incp_m:
    return LowerVMULASQACCLDBCINCP(Op, DAG,
                                   RISCVISD::ESP_VMULAS_U8_QACC_LDBC_INCP_M);
  case Intrinsic::riscv_esp_vcmulas_s8_qacc_h_ld_ip_m: {
    // Lower VCMULAS S8 QACC H LD IP intrinsic to custom SDNode
    // Intrinsic: (chain, int_id, v2_in, v3_in, qx, qy, ptr, offset) -> {ptr,
    // qu, v2, v3, chain} SDNode returns: (qu, ptr, v2, v3, chain) - 5 outputs
    // (Glue removed) SDNode operands: (chain, v2_in, v3_in, qx, qy, ptr,
    // offset) - 7 operands (Glue removed)
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue V2In = Op.getOperand(2); // QACC_H[127:0] passthru (v16i8)
    SDValue V3In = Op.getOperand(3); // QACC_H[255:128] passthru (v16i8)
    SDValue QX = Op.getOperand(4);
    SDValue QY = Op.getOperand(5);
    SDValue Ptr = Op.getOperand(6);
    SDValue Offset = Op.getOperand(7);

    EVT PtrVT = Ptr.getValueType();
    EVT MemVT = MVT::v16i8;

    // SDNode returns: (qu, ptr, v2, v3, chain) - 5 outputs (Glue removed)
    SmallVector<EVT, 5> VTList = {
        MVT::v16i8, PtrVT, MVT::v16i8,
        MVT::v16i8, // qu + ptr + 2x128-bit QACC_H
        MVT::Other  // Chain only, no Glue
    };
    SDVTList VTs = DAG.getVTList(VTList);

    // SDNode operands: (chain, v2_in, v3_in, qx, qy, ptr, offset) - 7 operands
    // (Glue removed)
    SDValue Ops[] = {Chain, V2In, V3In, QX, QY, Ptr, Offset};

    // Note: This intrinsic always arrives as MemIntrinsicSDNode because
    //       getTgtMemIntrinsic returns true for it.
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(
        RISCVISD::ESP_VCMULAS_S8_QACC_H_LD_IP_M, DL, VTs, Ops, MemVT, MMO);
    SDValue Qu = Node.getValue(0);
    SDValue PtrOut = Node.getValue(1);
    SDValue V2 = Node.getValue(2);
    SDValue V3 = Node.getValue(3);
    Chain = Node.getValue(4);

    return DAG.getMergeValues({PtrOut, Qu, V2, V3, Chain}, DL);
  }
  case Intrinsic::riscv_esp_vcmulas_s8_qacc_l_ld_ip_m: {
    // Lower VCMULAS S8 QACC L LD IP intrinsic to custom SDNode
    // Intrinsic: (chain, int_id, v0_in, v1_in, qx, qy, ptr, offset) -> {ptr,
    // qu, v0, v1, chain} SDNode returns: (qu, ptr, v0, v1, chain) - 5 outputs
    // (Glue removed) SDNode operands: (chain, v0_in, v1_in, qx, qy, ptr,
    // offset) - 7 operands (Glue removed)
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue V0In = Op.getOperand(2); // QACC_L[127:0] passthru (v16i8)
    SDValue V1In = Op.getOperand(3); // QACC_L[255:128] passthru (v16i8)
    SDValue QX = Op.getOperand(4);
    SDValue QY = Op.getOperand(5);
    SDValue Ptr = Op.getOperand(6);
    SDValue Offset = Op.getOperand(7);

    EVT PtrVT = Ptr.getValueType();
    EVT MemVT = MVT::v16i8;

    // SDNode returns: (qu, ptr, v0, v1, chain) - 5 outputs (Glue removed)
    SmallVector<EVT, 5> VTList = {
        MVT::v16i8, PtrVT, MVT::v16i8,
        MVT::v16i8, // qu + ptr + 2x128-bit QACC_L
        MVT::Other  // Chain only, no Glue
    };
    SDVTList VTs = DAG.getVTList(VTList);

    // SDNode operands: (chain, v0_in, v1_in, qx, qy, ptr, offset) - 7 operands
    // (Glue removed)
    SDValue Ops[] = {Chain, V0In, V1In, QX, QY, Ptr, Offset};

    // Note: This intrinsic always arrives as MemIntrinsicSDNode because
    //       getTgtMemIntrinsic returns true for it.
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(
        RISCVISD::ESP_VCMULAS_S8_QACC_L_LD_IP_M, DL, VTs, Ops, MemVT, MMO);
    SDValue Qu = Node.getValue(0);
    SDValue PtrOut = Node.getValue(1);
    SDValue V0 = Node.getValue(2);
    SDValue V1 = Node.getValue(3);
    Chain = Node.getValue(4);

    return DAG.getMergeValues({PtrOut, Qu, V0, V1, Chain}, DL);
  }
  case Intrinsic::riscv_esp_vcmulas_s16_qacc_h_ld_ip_m: {
    // Lower VCMULAS S16 QACC H LD IP intrinsic to custom SDNode
    // Intrinsic: (chain, int_id, v2, v3, qx, qy, ptr, offset) -> {ptr, qu, v2,
    // v3, chain} SDNode returns: (qu, ptr, v2, v3, chain) - 5 outputs SDNode
    // operands: (chain, v2_in, v3_in, qx, qy, ptr, offset) - 7 operands total
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue V2In = Op.getOperand(2); // QACC_H[127:0] passthru (v16i8)
    SDValue V3In = Op.getOperand(3); // QACC_H[255:128] passthru (v16i8)
    SDValue QX = Op.getOperand(4);
    SDValue QY = Op.getOperand(5);
    SDValue Ptr = Op.getOperand(6);
    SDValue Offset = Op.getOperand(7);

    // Simplified: Remove CopyToReg, QACC passthru directly as explicit phantom
    // operands Reference: esp.vld.128.ip and esp.mov.s16.qacc implementation
    // pattern

    EVT PtrVT = Ptr.getValueType();
    EVT MemVT = MVT::v16i8;

    // SDNode returns: (qu, ptr, v2, v3, chain) - 5 outputs (remove Glue)
    SmallVector<EVT, 5> VTList = {
        MVT::v16i8, PtrVT, MVT::v16i8,
        MVT::v16i8, // qu + ptr + 2x128-bit QACC_H
        MVT::Other  // Chain only, no Glue
    };
    SDVTList VTs = DAG.getVTList(VTList);

    // SDNode operands: (chain, v2_in, v3_in, qx, qy, ptr, offset) - 7 operands
    // (remove Glue)
    SDValue Ops[] = {Chain, V2In, V3In, QX, QY, Ptr, Offset};

    // Note: This intrinsic always arrives as MemIntrinsicSDNode because
    //       getTgtMemIntrinsic returns true for it.
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(
        RISCVISD::ESP_VCMULAS_S16_QACC_H_LD_IP_M, DL, VTs, Ops, MemVT, MMO);
    SDValue Qu = Node.getValue(0);     // qu (Result 0) - v16i8
    SDValue PtrOut = Node.getValue(1); // Updated pointer (Result 1)
    SDValue V2 = Node.getValue(2); // QACC_H[127:0] output (Result 2) - v16i8
    SDValue V3 = Node.getValue(3); // QACC_H[255:128] output (Result 3) - v16i8
    Chain = Node.getValue(4);      // Chain (Result 4)

    return DAG.getMergeValues({PtrOut, Qu, V2, V3, Chain}, DL);
  }
  case Intrinsic::riscv_esp_vcmulas_s16_qacc_l_ld_ip_m: {
    // Lower VCMULAS S16 QACC L LD IP intrinsic to custom SDNode
    // Intrinsic: (chain, int_id, v0_in, v1_in, qx, qy, ptr, offset) -> {ptr,
    // qu, v0, v1, chain} SDNode returns: (qu, ptr, v0, v1, chain) - 5 outputs
    // (Glue removed) SDNode operands: (chain, v0_in, v1_in, qx, qy, ptr,
    // offset) - 7 operands (Glue removed)
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue V0In = Op.getOperand(2); // QACC_L[127:0] passthru (v16i8)
    SDValue V1In = Op.getOperand(3); // QACC_L[255:128] passthru (v16i8)
    SDValue QX = Op.getOperand(4);
    SDValue QY = Op.getOperand(5);
    SDValue Ptr = Op.getOperand(6);
    SDValue Offset = Op.getOperand(7);

    EVT PtrVT = Ptr.getValueType();
    EVT MemVT = MVT::v16i8;

    // SDNode returns: (qu, ptr, v0, v1, chain) - 5 outputs (Glue removed)
    SmallVector<EVT, 5> VTList = {
        MVT::v16i8, PtrVT, MVT::v16i8,
        MVT::v16i8, // qu + ptr + 2x128-bit QACC_L
        MVT::Other  // Chain only, no Glue
    };
    SDVTList VTs = DAG.getVTList(VTList);

    // SDNode operands: (chain, v0_in, v1_in, qx, qy, ptr, offset) - 7 operands
    // (Glue removed)
    SDValue Ops[] = {Chain, V0In, V1In, QX, QY, Ptr, Offset};

    // Note: This intrinsic always arrives as MemIntrinsicSDNode because
    //       getTgtMemIntrinsic returns true for it.
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(
        RISCVISD::ESP_VCMULAS_S16_QACC_L_LD_IP_M, DL, VTs, Ops, MemVT, MMO);
    SDValue Qu = Node.getValue(0);
    SDValue PtrOut = Node.getValue(1);
    SDValue V0 = Node.getValue(2);
    SDValue V1 = Node.getValue(3);
    Chain = Node.getValue(4);

    return DAG.getMergeValues({PtrOut, Qu, V0, V1, Chain}, DL);
  }
  case Intrinsic::riscv_esp_vcmulas_s8_qacc_h_ld_xp_m: {
    // Lower VCMULAS S8 QACC H LD XP intrinsic to custom SDNode
    // Intrinsic: (chain, int_id, v2_in, v3_in, qx, qy, ptr, rs2) -> {ptr, qu,
    // v2, v3, chain} SDNode returns: (qu, ptr, v2, v3, chain) - 5 outputs (Glue
    // removed) SDNode operands: (chain, v2_in, v3_in, qx, qy, ptr, rs2) - 7
    // operands (Glue removed)
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue V2In = Op.getOperand(2); // QACC_H[127:0] passthru (v16i8)
    SDValue V3In = Op.getOperand(3); // QACC_H[255:128] passthru (v16i8)
    SDValue QX = Op.getOperand(4);
    SDValue QY = Op.getOperand(5);
    SDValue Ptr = Op.getOperand(6);
    SDValue Rs2 = Op.getOperand(7);

    EVT PtrVT = Ptr.getValueType();
    EVT MemVT = MVT::v16i8;

    // SDNode returns: (qu, ptr, v2, v3, chain) - 5 outputs (Glue removed)
    SmallVector<EVT, 5> VTList = {
        MVT::v16i8, PtrVT, MVT::v16i8,
        MVT::v16i8, // qu + ptr + 2x128-bit QACC_H
        MVT::Other  // Chain only, no Glue
    };
    SDVTList VTs = DAG.getVTList(VTList);

    // SDNode operands: (chain, v2_in, v3_in, qx, qy, ptr, rs2) - 7 operands
    // (Glue removed)
    SDValue Ops[] = {Chain, V2In, V3In, QX, QY, Ptr, Rs2};

    // Note: This intrinsic always arrives as MemIntrinsicSDNode because
    //       getTgtMemIntrinsic returns true for it.
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(
        RISCVISD::ESP_VCMULAS_S8_QACC_H_LD_XP_M, DL, VTs, Ops, MemVT, MMO);
    SDValue Qu = Node.getValue(0);
    SDValue PtrOut = Node.getValue(1);
    SDValue V2 = Node.getValue(2);
    SDValue V3 = Node.getValue(3);
    Chain = Node.getValue(4);

    return DAG.getMergeValues({PtrOut, Qu, V2, V3, Chain}, DL);
  }
  case Intrinsic::riscv_esp_vcmulas_s8_qacc_l_ld_xp_m: {
    // Lower VCMULAS S8 QACC L LD XP intrinsic to custom SDNode
    // Intrinsic: (chain, int_id, v0_in, v1_in, qx, qy, ptr, rs2) -> {ptr, qu,
    // v0, v1, chain} SDNode returns: (qu, ptr, v0, v1, chain) - 5 outputs (Glue
    // removed) SDNode operands: (chain, v0_in, v1_in, qx, qy, ptr, rs2) - 7
    // operands (Glue removed)
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue V0In = Op.getOperand(2); // QACC_L[127:0] passthru (v16i8)
    SDValue V1In = Op.getOperand(3); // QACC_L[255:128] passthru (v16i8)
    SDValue QX = Op.getOperand(4);
    SDValue QY = Op.getOperand(5);
    SDValue Ptr = Op.getOperand(6);
    SDValue Rs2 = Op.getOperand(7);

    EVT PtrVT = Ptr.getValueType();
    EVT MemVT = MVT::v16i8;

    // SDNode returns: (qu, ptr, v0, v1, chain) - 5 outputs (Glue removed)
    SmallVector<EVT, 5> VTList = {
        MVT::v16i8, PtrVT, MVT::v16i8,
        MVT::v16i8, // qu + ptr + 2x128-bit QACC_L
        MVT::Other  // Chain only, no Glue
    };
    SDVTList VTs = DAG.getVTList(VTList);

    // SDNode operands: (chain, v0_in, v1_in, qx, qy, ptr, rs2) - 7 operands
    // (Glue removed)
    SDValue Ops[] = {Chain, V0In, V1In, QX, QY, Ptr, Rs2};

    // Note: This intrinsic always arrives as MemIntrinsicSDNode because
    //       getTgtMemIntrinsic returns true for it.
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(
        RISCVISD::ESP_VCMULAS_S8_QACC_L_LD_XP_M, DL, VTs, Ops, MemVT, MMO);
    SDValue Qu = Node.getValue(0);
    SDValue PtrOut = Node.getValue(1);
    SDValue V0 = Node.getValue(2);
    SDValue V1 = Node.getValue(3);
    Chain = Node.getValue(4);

    return DAG.getMergeValues({PtrOut, Qu, V0, V1, Chain}, DL);
  }
  case Intrinsic::riscv_esp_vcmulas_s16_qacc_h_ld_xp_m: {
    // Lower VCMULAS S16 QACC H LD XP intrinsic to custom SDNode
    // Intrinsic: (chain, int_id, v2_in, v3_in, qx, qy, ptr, rs2) -> {ptr, qu,
    // v2, v3, chain} SDNode returns: (qu, ptr, v2, v3, chain) - 5 outputs (Glue
    // removed) SDNode operands: (chain, v2_in, v3_in, qx, qy, ptr, rs2) - 7
    // operands (Glue removed)
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue V2In = Op.getOperand(2); // QACC_H[127:0] passthru (v16i8)
    SDValue V3In = Op.getOperand(3); // QACC_H[255:128] passthru (v16i8)
    SDValue QX = Op.getOperand(4);
    SDValue QY = Op.getOperand(5);
    SDValue Ptr = Op.getOperand(6);
    SDValue Rs2 = Op.getOperand(7);

    EVT PtrVT = Ptr.getValueType();
    EVT MemVT = MVT::v16i8;

    // SDNode returns: (qu, ptr, v2, v3, chain) - 5 outputs (Glue removed)
    SmallVector<EVT, 5> VTList = {
        MVT::v16i8, PtrVT, MVT::v16i8,
        MVT::v16i8, // qu + ptr + 2x128-bit QACC_H
        MVT::Other  // Chain only, no Glue
    };
    SDVTList VTs = DAG.getVTList(VTList);

    // SDNode operands: (chain, v2_in, v3_in, qx, qy, ptr, rs2) - 7 operands
    // (Glue removed)
    SDValue Ops[] = {Chain, V2In, V3In, QX, QY, Ptr, Rs2};

    // Note: This intrinsic always arrives as MemIntrinsicSDNode because
    //       getTgtMemIntrinsic returns true for it.
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(
        RISCVISD::ESP_VCMULAS_S16_QACC_H_LD_XP_M, DL, VTs, Ops, MemVT, MMO);
    SDValue Qu = Node.getValue(0);
    SDValue PtrOut = Node.getValue(1);
    SDValue V2 = Node.getValue(2);
    SDValue V3 = Node.getValue(3);
    Chain = Node.getValue(4);

    return DAG.getMergeValues({PtrOut, Qu, V2, V3, Chain}, DL);
  }
  case Intrinsic::riscv_esp_vcmulas_s16_qacc_l_ld_xp_m: {
    // Lower VCMULAS S16 QACC L LD XP intrinsic to custom SDNode
    // Intrinsic: (chain, int_id, v0_in, v1_in, qx, qy, ptr, rs2) -> {ptr, qu,
    // v0, v1, chain} SDNode returns: (qu, ptr, v0, v1, chain) - 5 outputs (Glue
    // removed) SDNode operands: (chain, v0_in, v1_in, qx, qy, ptr, rs2) - 7
    // operands (Glue removed)
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue V0In = Op.getOperand(2); // QACC_L[127:0] passthru (v16i8)
    SDValue V1In = Op.getOperand(3); // QACC_L[255:128] passthru (v16i8)
    SDValue QX = Op.getOperand(4);
    SDValue QY = Op.getOperand(5);
    SDValue Ptr = Op.getOperand(6);
    SDValue Rs2 = Op.getOperand(7);

    EVT PtrVT = Ptr.getValueType();
    EVT MemVT = MVT::v16i8;

    // SDNode returns: (qu, ptr, v0, v1, chain) - 5 outputs (Glue removed)
    SmallVector<EVT, 5> VTList = {
        MVT::v16i8, PtrVT, MVT::v16i8,
        MVT::v16i8, // qu + ptr + 2x128-bit QACC_L
        MVT::Other  // Chain only, no Glue
    };
    SDVTList VTs = DAG.getVTList(VTList);

    // SDNode operands: (chain, v0_in, v1_in, qx, qy, ptr, rs2) - 7 operands
    // (Glue removed)
    SDValue Ops[] = {Chain, V0In, V1In, QX, QY, Ptr, Rs2};

    // Note: This intrinsic always arrives as MemIntrinsicSDNode because
    //       getTgtMemIntrinsic returns true for it.
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(
        RISCVISD::ESP_VCMULAS_S16_QACC_L_LD_XP_M, DL, VTs, Ops, MemVT, MMO);
    SDValue Qu = Node.getValue(0);
    SDValue PtrOut = Node.getValue(1);
    SDValue V0 = Node.getValue(2);
    SDValue V1 = Node.getValue(3);
    Chain = Node.getValue(4);

    return DAG.getMergeValues({PtrOut, Qu, V0, V1, Chain}, DL);
  }

  default:
    return SDValue(); // Not an ESPV intrinsic handled here
  }
}

static SDValue LowerSTXACCIP(SDValue Op, SelectionDAG &DAG,
                             unsigned ISDOpcode) {
  // Intrinsic: (chain, int_id, xacc_low_in, xacc_high_in, ptr, offset) -> {ptr,
  // xacc_low_unchanged, xacc_high_unchanged, chain} Mixed model: XACC as {i32
  // low, i32 high} SDNode: (xacc_low_in, xacc_high_in, chain, ptr, offset,
  // glue) -> {ptr, xacc_low_unchanged, xacc_high_unchanged, chain, glue} Direct
  // Real Instruction Approach: Intrinsic -> SDNode -> Real MachineInstr (with
  // phantom operand)
  //
  // Lowering Stage: Generate SDNode with passthru operands
  // - Passthru establishes explicit data dependency, preventing esp.zero.xacc
  // from being optimized away
  // - Select stage will choose real instruction ESP_ST_S_XACC_IP /
  // ESP_ST_U_XACC_IP directly
  // - Real instruction has XACC parts as phantom operands (in (ins) but not
  // printed in assembly)
  // - No pseudo instruction expansion needed, avoiding Pre-RA expansion issues
  SDLoc DL(Op);
  SDValue Chain = Op.getOperand(0);
  SDValue XACCLowIn = Op.getOperand(2); // i32 passthru (XACC[31:0])
  SDValue XACCHighIn =
      Op.getOperand(3); // i32 passthru (XACC[39:32], only low 8 bits valid)
  SDValue Ptr = Op.getOperand(4);
  SDValue Offset = Op.getOperand(5);

  EVT PtrVT = Ptr.getValueType();
  EVT MemVT = MVT::i64; // Store 64-bit, use low 40 bits

  // SDNode with SDNPHasChain and SDNPOutGlue: Chain and Glue are added
  // automatically SDTypeProfile defines 3 explicit results (ptr,
  // xacc_low_unchanged, xacc_high_unchanged), plus Chain and Glue = 5 values
  // total
  SmallVector<EVT, 5> VTs = {PtrVT, MVT::i32, MVT::i32, MVT::Other, MVT::Glue};
  SDVTList VTList = DAG.getVTList(VTs);
  // Operands: Chain (SDNPHasChain requires it as first operand), XACC low, XACC
  // high, Ptr, Offset SDTypeProfile defines 4 operands (XACC low, XACC high,
  // Ptr, Offset), SDNPHasChain adds Chain as first operand = 5 total
  // SDNPOptInGlue means Glue is optional and doesn't need to be explicitly
  // passed Passthru operands XACCLowIn and XACCHighIn establish data dependency
  // (phantom operands for data flow) No need for CopyToReg - passthru operands
  // directly establish data dependency
  SDValue Ops[] = {Chain, XACCLowIn, XACCHighIn, Ptr, Offset};

  // Create the SDNode - it returns 5 values: (ptr, xacc_low_unchanged,
  // xacc_high_unchanged, chain, glue) Use getMemIntrinsicNode to preserve
  // memory information (similar to esp.vst.128.ip) Note: This intrinsic always
  // arrives as MemIntrinsicSDNode because
  //       getTgtMemIntrinsic returns true for it.
  auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
  MachineMemOperand *MMO = MemIntr->getMemOperand();
  SDValue Node =
      DAG.getMemIntrinsicNode(ISDOpcode, DL, VTList, Ops, MemVT, MMO);

  // SDNode returns (ptr, xacc_low_unchanged, xacc_high_unchanged, chain, glue)
  // - 5 values total Instruction outputs XACC_LOW and XACC_HIGH virtual
  // registers (unchanged, equals input) Use instruction outputs directly for
  // consistency
  SDValue PtrOut = Node.getValue(0);
  SDValue XACCLowOut = Node.getValue(
      1); // XACC_LOW virtual register from instruction output (unchanged)
  SDValue XACCHighOut = Node.getValue(
      2); // XACC_HIGH virtual register from instruction output (unchanged)
  Chain = Node.getValue(3);

  return DAG.getMergeValues({PtrOut, XACCLowOut, XACCHighOut, Chain}, DL);
}

// LD/ST UA_STATE Lowering
static SDValue LowerLDUASTATEIP(SDValue Op, SelectionDAG &DAG,
                                unsigned ISDOpcode) {
  // Lower intrinsic to custom SDNode that will be matched to ESP_LD_UA_STATE_IP
  // Intrinsic: (chain, int_id, ua_state_passthru, ptr, offset)
  // Returns: {new_ua_state, ptr, chain}
  // SDNode: (chain, ptr, offset, ua_state_passthru) -> (new_ua_state, ptr,
  // chain)
  SDLoc DL(Op);
  SDValue Chain = Op.getOperand(0);
  SDValue Passthru =
      Op.getOperand(2); // v16i8 passthru (phantom operand for data flow)
  SDValue Ptr = Op.getOperand(3);
  SDValue Offset = Op.getOperand(4);

  EVT VecVT = MVT::v16i8;
  EVT PtrVT = Ptr.getValueType();
  SDVTList VTs = DAG.getVTList(VecVT, PtrVT, MVT::Other);

  SDValue Ops[] = {Chain, Ptr, Offset, Passthru};
  SDValue Node = DAG.getNode(ISDOpcode, DL, VTs, Ops);

  return DAG.getMergeValues(
      {Node.getValue(0), Node.getValue(1), Node.getValue(2)}, DL);
}

static SDValue LowerSTUASTATEIP(SDValue Op, SelectionDAG &DAG,
                                unsigned ISDOpcode) {
  // Lower intrinsic to custom SDNode that will be matched to ESP_ST_UA_STATE_IP
  // Intrinsic: (chain, int_id, ua_state_passthru, ptr, offset)
  // Returns: {new_ua_state, ptr, chain}
  // SDNode: (chain, ua_state_passthru, ptr, offset) -> (new_ua_state, ptr,
  // chain)
  SDLoc DL(Op);
  SDValue Chain = Op.getOperand(0);
  SDValue Passthru =
      Op.getOperand(2); // v16i8 passthru (phantom operand for data flow)
  SDValue Ptr = Op.getOperand(3);
  SDValue Offset = Op.getOperand(4);

  EVT VecVT = MVT::v16i8;
  EVT PtrVT = Ptr.getValueType();
  SDVTList VTs = DAG.getVTList(VecVT, PtrVT, MVT::Other);

  SDValue Ops[] = {Chain, Passthru, Ptr, Offset};
  SDValue Node = DAG.getNode(ISDOpcode, DL, VTs, Ops);

  return DAG.getMergeValues(
      {Node.getValue(0), Node.getValue(1), Node.getValue(2)}, DL);
}

static SDValue LowerLDQAIP(SDValue Op, SelectionDAG &DAG, unsigned ISDOpcode) {
  // Intrinsic: (chain, int_id, qacc_passthru, ptr, offset) -> {ptr, v16i8,
  // v16i8, v16i8, v16i8, chain} SDNode returns: (v16i8, v16i8, v16i8, v16i8,
  // ptr, chain, glue) - explicit outputs
  SDLoc DL(Op);
  SDValue Chain = Op.getOperand(0);
  SDValue Passthru = Op.getOperand(2); // v64i8 passthru
  SDValue Ptr = Op.getOperand(3);
  SDValue Offset = Op.getOperand(4);

  // Split v64i8 passthru into 4x128-bit for passthru handling
  // Extract 4x128-bit from passthru: [0:15], [16:31], [32:47], [48:63]
  SDValue PassthruV0 = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, MVT::v16i8,
                                   Passthru, DAG.getConstant(0, DL, MVT::i32));
  SDValue PassthruV1 = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, MVT::v16i8,
                                   Passthru, DAG.getConstant(16, DL, MVT::i32));
  SDValue PassthruV2 = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, MVT::v16i8,
                                   Passthru, DAG.getConstant(32, DL, MVT::i32));
  SDValue PassthruV3 = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, MVT::v16i8,
                                   Passthru, DAG.getConstant(48, DL, MVT::i32));

  // Combine 4x128-bit into 2x256-bit for register passthru
  SDValue PassthruL =
      DAG.getNode(ISD::CONCAT_VECTORS, DL, MVT::v32i8, PassthruV0, PassthruV1);
  SDValue PassthruH =
      DAG.getNode(ISD::CONCAT_VECTORS, DL, MVT::v32i8, PassthruV2, PassthruV3);

  SDValue Glue;
  Chain = DAG.getCopyToReg(Chain, DL, RISCV::QACC_H_REG, PassthruH, Glue);
  Glue = Chain.getValue(1);
  Chain = DAG.getCopyToReg(Chain, DL, RISCV::QACC_L_REG, PassthruL, Glue);
  Glue = Chain.getValue(1);

  EVT PtrVT = Ptr.getValueType();
  EVT MemVT = MVT::v16i8;
  // SDNode returns: (v16i8, v16i8, v16i8, v16i8, ptr, chain, glue) - 7 outputs
  // (4x128-bit + ptr + chain + glue)
  SmallVector<EVT, 7> VTList = {MVT::v16i8, MVT::v16i8, MVT::v16i8, MVT::v16i8,
                                PtrVT,      MVT::Other, MVT::Glue};
  SDVTList VTs = DAG.getVTList(VTList);
  SDValue Ops[] = {Chain, Ptr, Offset, Glue};

  // Note: This intrinsic always arrives as MemIntrinsicSDNode because
  //       getTgtMemIntrinsic returns true for it.
  auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
  MachineMemOperand *MMO = MemIntr->getMemOperand();
  SDValue Node = DAG.getMemIntrinsicNode(ISDOpcode, DL, VTs, Ops, MemVT, MMO);
  SDValue V0 = Node.getValue(0); // QACC_L[127:0] output (Result 0) - v16i8
  SDValue V1 = Node.getValue(1); // QACC_L[255:128] output (Result 1) - v16i8
  SDValue V2 = Node.getValue(2); // QACC_H[127:0] output (Result 2) - v16i8
  SDValue V3 = Node.getValue(3); // QACC_H[255:128] output (Result 3) - v16i8
  SDValue PtrOut = Node.getValue(4); // Updated pointer (Result 4)
  Chain = Node.getValue(5);          // Chain (Result 5)

  return DAG.getMergeValues({PtrOut, V0, V1, V2, V3, Chain}, DL);
}

static SDValue LowerLDQAXP(SDValue Op, SelectionDAG &DAG, unsigned ISDOpcode) {
  // Intrinsic: (chain, int_id, qacc_passthru, ptr, rs2) -> {ptr, v16i8, v16i8,
  // v16i8, v16i8, chain} SDNode returns: (QACC_L, QACC_H, ptr, chain) -
  // explicit outputs
  SDLoc DL(Op);
  SDValue Chain = Op.getOperand(0);
  SDValue Passthru = Op.getOperand(2); // v64i8 passthru
  SDValue Ptr = Op.getOperand(3);
  SDValue Rs2 = Op.getOperand(4);

  // Split v64i8 passthru into 4x128-bit for passthru handling
  SDValue PassthruV0 = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, MVT::v16i8,
                                   Passthru, DAG.getConstant(0, DL, MVT::i32));
  SDValue PassthruV1 = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, MVT::v16i8,
                                   Passthru, DAG.getConstant(16, DL, MVT::i32));
  SDValue PassthruV2 = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, MVT::v16i8,
                                   Passthru, DAG.getConstant(32, DL, MVT::i32));
  SDValue PassthruV3 = DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, MVT::v16i8,
                                   Passthru, DAG.getConstant(48, DL, MVT::i32));

  // Combine 4x128-bit into 2x256-bit for register passthru
  SDValue PassthruL =
      DAG.getNode(ISD::CONCAT_VECTORS, DL, MVT::v32i8, PassthruV0, PassthruV1);
  SDValue PassthruH =
      DAG.getNode(ISD::CONCAT_VECTORS, DL, MVT::v32i8, PassthruV2, PassthruV3);

  SDValue Glue;
  Chain = DAG.getCopyToReg(Chain, DL, RISCV::QACC_H_REG, PassthruH, Glue);
  Glue = Chain.getValue(1);
  Chain = DAG.getCopyToReg(Chain, DL, RISCV::QACC_L_REG, PassthruL, Glue);
  Glue = Chain.getValue(1);

  EVT PtrVT = Ptr.getValueType();
  EVT MemVT = MVT::v16i8;
  // SDNode returns: (v16i8, v16i8, v16i8, v16i8, ptr, chain, glue) - 7 outputs
  // (4x128-bit + ptr + chain + glue)
  SmallVector<EVT, 7> VTList = {MVT::v16i8, MVT::v16i8, MVT::v16i8, MVT::v16i8,
                                PtrVT,      MVT::Other, MVT::Glue};
  SDVTList VTs = DAG.getVTList(VTList);
  SDValue Ops[] = {Chain, Ptr, Rs2, Glue};

  // Note: This intrinsic always arrives as MemIntrinsicSDNode because
  //       getTgtMemIntrinsic returns true for it.
  auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
  MachineMemOperand *MMO = MemIntr->getMemOperand();
  SDValue Node = DAG.getMemIntrinsicNode(ISDOpcode, DL, VTs, Ops, MemVT, MMO);
  SDValue V0 = Node.getValue(0); // QACC_L[127:0] output (Result 0) - v16i8
  SDValue V1 = Node.getValue(1); // QACC_L[255:128] output (Result 1) - v16i8
  SDValue V2 = Node.getValue(2); // QACC_H[127:0] output (Result 2) - v16i8
  SDValue V3 = Node.getValue(3); // QACC_H[255:128] output (Result 3) - v16i8
  SDValue PtrOut = Node.getValue(4); // Updated pointer (Result 4)
  Chain = Node.getValue(5);          // Chain (Result 5)

  return DAG.getMergeValues({PtrOut, V0, V1, V2, V3, Chain}, DL);
}

// ESPV intrinsic lowering for INTRINSIC_WO_CHAIN
SDValue lowerESPVIntrinsicWOChain(SDValue Op, SelectionDAG &DAG,
                                  const RISCVSubtarget &Subtarget) {
  if (!Subtarget.hasVendorXespv())
    return SDValue();

  unsigned IntNo = Op.getConstantOperandVal(0);
  SDLoc DL(Op);

  switch (IntNo) {
  // ESP MAX/MIN reduction intrinsics - lower to vecreduce nodes for pattern
  // matching
  case Intrinsic::riscv_esp_max_s8_a_m:
  case Intrinsic::riscv_esp_max_s16_a_m:
  case Intrinsic::riscv_esp_max_s32_a_m:
    return DAG.getNode(ISD::VECREDUCE_SMAX, DL, Op.getValueType(),
                       Op.getOperand(1));
  case Intrinsic::riscv_esp_max_u8_a_m:
  case Intrinsic::riscv_esp_max_u16_a_m:
  case Intrinsic::riscv_esp_max_u32_a_m:
    return DAG.getNode(ISD::VECREDUCE_UMAX, DL, Op.getValueType(),
                       Op.getOperand(1));
  case Intrinsic::riscv_esp_min_s8_a_m:
  case Intrinsic::riscv_esp_min_s16_a_m:
  case Intrinsic::riscv_esp_min_s32_a_m:
    return DAG.getNode(ISD::VECREDUCE_SMIN, DL, Op.getValueType(),
                       Op.getOperand(1));
  case Intrinsic::riscv_esp_min_u8_a_m:
  case Intrinsic::riscv_esp_min_u16_a_m:
  case Intrinsic::riscv_esp_min_u32_a_m:
    return DAG.getNode(ISD::VECREDUCE_UMIN, DL, Op.getValueType(),
                       Op.getOperand(1));
  case Intrinsic::riscv_esp_zero_qacc_m: {
    // ESP.ZERO.QACC - Zero QACC accumulator with explicit state passing
    // Intrinsic: () -> {v16i8, v16i8, v16i8, v16i8} - 4x128-bit QACC directly
    // SDNode returns: (v16i8, v16i8, v16i8, v16i8) - 4x128-bit QACC directly
    // Consistent with ESP_VMULAS_S16_QACC_M and ESP_MOV_S16_QACC_M
    SDValue Chain = DAG.getEntryNode();

    // 1. Generate ESP_ZERO_QACC instruction with explicit 4x128-bit outputs
    // Instruction outputs: (QACC_L_LOW, QACC_L_HIGH, QACC_H_LOW, QACC_H_HIGH,
    // Chain, Glue)
    SmallVector<EVT, 6> VTs = {MVT::v16i8, MVT::v16i8, MVT::v16i8,
                               MVT::v16i8, MVT::Other, MVT::Glue};
    SDVTList VTList = DAG.getVTList(VTs);
    SDValue Ops[] = {Chain};
    SDValue ZeroCmd = DAG.getNode(RISCVISD::ESP_ZERO_QACC_M, DL, VTList, Ops);

    // 2. Return structure with 4x128-bit QACC directly
    return DAG.getMergeValues({ZeroCmd.getValue(0), ZeroCmd.getValue(1),
                               ZeroCmd.getValue(2), ZeroCmd.getValue(3)},
                              DL);
  }
  case Intrinsic::riscv_esp_mov_s16_qacc_m: {
    // ESP.MOV.S16.QACC - Sign extend 8x16-bit to 64-bit, store to QACC_H and
    // QACC_L Intrinsic: (v8i16) -> {v16i8, v16i8, v16i8, v16i8} - 4x128-bit
    // QACC directly SDNode returns: (v16i8, v16i8, v16i8, v16i8) - 4x128-bit
    // QACC directly
    SDValue QU = Op.getOperand(1); // v8i16 input vector

    SmallVector<EVT, 4> VTList = {MVT::v16i8, MVT::v16i8, MVT::v16i8,
                                  MVT::v16i8};
    SDVTList VTs = DAG.getVTList(VTList);
    SDValue Ops[] = {QU};
    SDValue Node = DAG.getNode(RISCVISD::ESP_MOV_S16_QACC_M, DL, VTs, Ops);

    // Return structure with 4x128-bit QACC directly
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1),
                               Node.getValue(2), Node.getValue(3)},
                              DL);
  }
  case Intrinsic::riscv_esp_mov_s8_qacc_m: {
    // ESP.MOV.S8.QACC - Sign extend 16x8-bit to 32-bit, store to QACC_H and
    // QACC_L Intrinsic: (v16i8) -> {v16i8, v16i8, v16i8, v16i8} - 4x128-bit
    // QACC directly SDNode returns: (v16i8, v16i8, v16i8, v16i8) - 4x128-bit
    // QACC directly
    SDValue QU = Op.getOperand(1); // v16i8 input vector

    SmallVector<EVT, 4> VTList = {MVT::v16i8, MVT::v16i8, MVT::v16i8,
                                  MVT::v16i8};
    SDVTList VTs = DAG.getVTList(VTList);
    SDValue Ops[] = {QU};
    SDValue Node = DAG.getNode(RISCVISD::ESP_MOV_S8_QACC_M, DL, VTs, Ops);

    // Return structure with 4x128-bit QACC directly
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1),
                               Node.getValue(2), Node.getValue(3)},
                              DL);
  }
  case Intrinsic::riscv_esp_mov_u16_qacc_m: {
    // ESP.MOV.U16.QACC - Zero extend 8x16-bit to 64-bit, store to QACC_H and
    // QACC_L Intrinsic: (v8i16) -> {v16i8, v16i8, v16i8, v16i8} - 4x128-bit
    // QACC directly SDNode returns: (v16i8, v16i8, v16i8, v16i8) - 4x128-bit
    // QACC directly
    SDValue QU = Op.getOperand(1); // v8i16 input vector

    SmallVector<EVT, 4> VTList = {MVT::v16i8, MVT::v16i8, MVT::v16i8,
                                  MVT::v16i8};
    SDVTList VTs = DAG.getVTList(VTList);
    SDValue Ops[] = {QU};
    SDValue Node = DAG.getNode(RISCVISD::ESP_MOV_U16_QACC_M, DL, VTs, Ops);

    // Return structure with 4x128-bit QACC directly
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1),
                               Node.getValue(2), Node.getValue(3)},
                              DL);
  }
  case Intrinsic::riscv_esp_mov_u8_qacc_m: {
    // ESP.MOV.U8.QACC - Zero extend 16x8-bit to 32-bit, store to QACC_H and
    // QACC_L Intrinsic: (v16i8) -> {v16i8, v16i8, v16i8, v16i8} - 4x128-bit
    // QACC directly SDNode returns: (v16i8, v16i8, v16i8, v16i8) - 4x128-bit
    // QACC directly
    SDValue QU = Op.getOperand(1); // v16i8 input vector

    SmallVector<EVT, 4> VTList = {MVT::v16i8, MVT::v16i8, MVT::v16i8,
                                  MVT::v16i8};
    SDVTList VTs = DAG.getVTList(VTList);
    SDValue Ops[] = {QU};
    SDValue Node = DAG.getNode(RISCVISD::ESP_MOV_U8_QACC_M, DL, VTs, Ops);

    // Return structure with 4x128-bit QACC directly
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1),
                               Node.getValue(2), Node.getValue(3)},
                              DL);
  }
  case Intrinsic::riscv_esp_zero_xacc_m: {
    // ESP.ZERO.XACC - Mixed model: XACC as {i32 low, i32 high}
    // Intrinsic: () -> {i32, i32} (both set to 0)
    // Create SDNode with Chain and Glue to prevent optimization
    SDValue Chain = DAG.getEntryNode();

    SDVTList VTs = DAG.getVTList(MVT::i32, MVT::i32, MVT::Other, MVT::Glue);
    SDValue Ops[] = {Chain};
    SDValue Node = DAG.getNode(RISCVISD::ESP_ZERO_XACC_M, DL, VTs, Ops);

    // Return {i32 xacc_low=0, i32 xacc_high=0}
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1)}, DL);
  }
  case Intrinsic::riscv_esp_vcmulas_s16_qacc_l_m: {
    // Lower VCMULAS S16 QACC L pure compute intrinsic
    // Intrinsic: (int_id, v0, v1, qx, qy) -> {v16i8, v16i8}
    // SDNode returns: (v16i8, v16i8) - 2x128-bit QACC_L directly
    // Passthru is passed as 2x128-bit explicit phantom operands
    SDLoc DL(Op);
    SDValue V0In = Op.getOperand(1); // QACC_L[127:0] passthru (v16i8)
    SDValue V1In = Op.getOperand(2); // QACC_L[255:128] passthru (v16i8)
    SDValue QX = Op.getOperand(3);
    SDValue QY = Op.getOperand(4);

    // SDNode returns: (v16i8, v16i8) - 2x128-bit QACC_L directly
    // Operands: (v0, v1, qx, qy) - 2x128-bit passthru as explicit phantom
    // operands
    SmallVector<EVT, 2> VTList = {MVT::v16i8, MVT::v16i8};
    SDVTList VTs = DAG.getVTList(VTList);
    SDValue Ops[] = {V0In, V1In, QX, QY};
    SDValue Node =
        DAG.getNode(RISCVISD::ESP_VCMULAS_S16_QACC_L_M, DL, VTs, Ops);

    // Return structure with 2x128-bit QACC_L directly
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1)}, DL);
  }
  case Intrinsic::riscv_esp_vcmulas_s16_qacc_h_m: {
    // Lower VCMULAS S16 QACC H pure compute intrinsic
    // Intrinsic: (int_id, v2, v3, qx, qy) -> {v16i8, v16i8}
    // SDNode returns: (v16i8, v16i8) - 2x128-bit QACC_H directly
    // Passthru is passed as 2x128-bit explicit phantom operands
    SDLoc DL(Op);
    SDValue V2In = Op.getOperand(1); // QACC_H[127:0] passthru (v16i8)
    SDValue V3In = Op.getOperand(2); // QACC_H[255:128] passthru (v16i8)
    SDValue QX = Op.getOperand(3);
    SDValue QY = Op.getOperand(4);

    // SDNode returns: (v16i8, v16i8) - 2x128-bit QACC_H directly
    // Operands: (v2, v3, qx, qy) - 2x128-bit passthru as explicit phantom
    // operands
    SmallVector<EVT, 2> VTList = {MVT::v16i8, MVT::v16i8};
    SDVTList VTs = DAG.getVTList(VTList);
    SDValue Ops[] = {V2In, V3In, QX, QY};
    SDValue Node =
        DAG.getNode(RISCVISD::ESP_VCMULAS_S16_QACC_H_M, DL, VTs, Ops);

    // Return structure with 2x128-bit QACC_H directly
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1)}, DL);
  }
  case Intrinsic::riscv_esp_vcmulas_s8_qacc_h_m: {
    // Lower VCMULAS S8 QACC H pure compute intrinsic
    // Intrinsic: (int_id, v2, v3, qx, qy) -> {v16i8, v16i8}
    // SDNode returns: (v16i8, v16i8) - 2x128-bit QACC_H directly
    // Passthru is passed as 2x128-bit explicit phantom operands
    SDLoc DL(Op);
    SDValue V2In = Op.getOperand(1); // QACC_H[127:0] passthru (v16i8)
    SDValue V3In = Op.getOperand(2); // QACC_H[255:128] passthru (v16i8)
    SDValue QX = Op.getOperand(3);
    SDValue QY = Op.getOperand(4);

    // SDNode returns: (v16i8, v16i8) - 2x128-bit QACC_H directly
    // Operands: (v2, v3, qx, qy) - 2x128-bit passthru as explicit phantom
    // operands
    SmallVector<EVT, 2> VTList = {MVT::v16i8, MVT::v16i8};
    SDVTList VTs = DAG.getVTList(VTList);
    SDValue Ops[] = {V2In, V3In, QX, QY};
    SDValue Node = DAG.getNode(RISCVISD::ESP_VCMULAS_S8_QACC_H_M, DL, VTs, Ops);

    // Return structure with 2x128-bit QACC_H directly
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1)}, DL);
  }
  case Intrinsic::riscv_esp_vcmulas_s8_qacc_l_m: {
    // Lower VCMULAS S8 QACC L pure compute intrinsic
    // Intrinsic: (int_id, v0, v1, qx, qy) -> {v16i8, v16i8}
    // SDNode returns: (v16i8, v16i8) - 2x128-bit QACC_L directly
    // Passthru is passed as 2x128-bit explicit phantom operands
    SDLoc DL(Op);
    SDValue V0In = Op.getOperand(1); // QACC_L[127:0] passthru (v16i8)
    SDValue V1In = Op.getOperand(2); // QACC_L[255:128] passthru (v16i8)
    SDValue QX = Op.getOperand(3);
    SDValue QY = Op.getOperand(4);

    // SDNode returns: (v16i8, v16i8) - 2x128-bit QACC_L directly
    // Operands: (v0, v1, qx, qy) - 2x128-bit passthru as explicit phantom
    // operands
    SmallVector<EVT, 2> VTList = {MVT::v16i8, MVT::v16i8};
    SDVTList VTs = DAG.getVTList(VTList);
    SDValue Ops[] = {V0In, V1In, QX, QY};
    SDValue Node = DAG.getNode(RISCVISD::ESP_VCMULAS_S8_QACC_L_M, DL, VTs, Ops);

    // Return structure with 2x128-bit QACC_L directly
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1)}, DL);
  }
  case Intrinsic::riscv_esp_fft_bitrev_m: {
    // Lower FFT BITREV intrinsic to custom SDNode with explicit FFT_BIT_WIDTH
    // state passing Intrinsic: (int_id, rs1, fft_bit_width) - IntrNoMem, so no
    // Chain Returns: {ptr, qv} SDNode: (rs1, fft_bit_width) -> (rs1r, qv) Note:
    // FFT_BIT_WIDTH is passed explicitly as i32 for explicit state passing
    // Note: No Chain because this is a computation-only instruction that
    // doesn't access memory
    SDLoc DL(Op);
    SDValue RS1 =
        Op.getOperand(1); // WO_CHAIN: operand 0 is int_id, operand 1 is rs1
    SDValue FftBitWidth =
        Op.getOperand(2); // FFT_BIT_WIDTH (i32, only low 4 bits used)

    EVT PtrVT = RS1.getValueType();
    SmallVector<EVT, 2> VTs = {PtrVT, MVT::v8i16};
    SDVTList VTList = DAG.getVTList(VTs);
    SDValue Ops[] = {RS1, FftBitWidth};
    SDValue Node = DAG.getNode(RISCVISD::ESP_FFT_BITREV_M, DL, VTList, Ops);

    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1)}, DL);
  }
  case Intrinsic::riscv_esp_fft_r2bf_s16_m: {
    // Lower FFT R2BF S16 intrinsic to custom SDNode
    // Intrinsic: (int_id, qx, qy, sel2)
    // Returns: {qz, qv}
    // SDNode: (qx, qy, sel2) -> (qz, qv)
    SDValue QX = Op.getOperand(1);
    SDValue QY = Op.getOperand(2);
    SDValue SEL2 = Op.getOperand(3);

    SmallVector<EVT, 2> VTs = {MVT::v8i16, MVT::v8i16};
    SDVTList VTList = DAG.getVTList(VTs);
    SDValue Ops[] = {QX, QY, SEL2};
    SDValue Node = DAG.getNode(RISCVISD::ESP_FFT_R2BF_S16_M, DL, VTList, Ops);

    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1)}, DL);
  }
  case Intrinsic::riscv_esp_srcmb_s16_qacc_m: {
    // Lower SRCMB S16 QACC intrinsic
    // Intrinsic: (v0, v1, v2, v3, rs1, sel2) -> v8i16
    // v0-v3: 4x128-bit QACC (QACC_L[127:0], QACC_L[255:128], QACC_H[127:0],
    // QACC_H[255:128]) SDNode: (v0, v1, v2, v3, rs1, sel2) -> v8i16 QACC is
    // passed as explicit phantom operands (4x128-bit) for proper data flow
    // tracking
    SDLoc DL(Op);
    SDValue V0 = Op.getOperand(1);   // QACC_L[127:0]
    SDValue V1 = Op.getOperand(2);   // QACC_L[255:128]
    SDValue V2 = Op.getOperand(3);   // QACC_H[127:0]
    SDValue V3 = Op.getOperand(4);   // QACC_H[255:128]
    SDValue RS1 = Op.getOperand(5);  // Shift amount
    SDValue Sel2 = Op.getOperand(6); // Saturation select

    // Create SDNode with QACC as explicit phantom operands (4x128-bit)
    // SDNode returns: v8i16 (qu)
    // Operands: (v0, v1, v2, v3, rs1, sel2) - QACC as 4x128-bit phantom
    // operands
    SDVTList VTs = DAG.getVTList(MVT::v8i16);
    SDValue Ops[] = {V0, V1, V2, V3, RS1, Sel2};
    SDValue Node = DAG.getNode(RISCVISD::ESP_SRCMB_S16_QACC_M, DL, VTs, Ops);

    return Node;
  }
  case Intrinsic::riscv_esp_srcmb_s8_qacc_m: {
    // Lower SRCMB S8 QACC intrinsic
    // Intrinsic: (v0, v1, v2, v3, rs1, sel2) -> v16i8
    // v0-v3: 4x128-bit QACC (QACC_L[127:0], QACC_L[255:128], QACC_H[127:0],
    // QACC_H[255:128]) SDNode: (v0, v1, v2, v3, rs1, sel2) -> v16i8 QACC is
    // passed as explicit phantom operands (4x128-bit) for proper data flow
    // tracking
    SDLoc DL(Op);
    SDValue V0 = Op.getOperand(1);   // QACC_L[127:0]
    SDValue V1 = Op.getOperand(2);   // QACC_L[255:128]
    SDValue V2 = Op.getOperand(3);   // QACC_H[127:0]
    SDValue V3 = Op.getOperand(4);   // QACC_H[255:128]
    SDValue RS1 = Op.getOperand(5);  // Shift amount
    SDValue Sel2 = Op.getOperand(6); // Saturation select

    // Create SDNode with QACC as explicit phantom operands (4x128-bit)
    // SDNode returns: v16i8 (qu)
    // Operands: (v0, v1, v2, v3, rs1, sel2) - QACC as 4x128-bit phantom
    // operands
    SDVTList VTs = DAG.getVTList(MVT::v16i8);
    SDValue Ops[] = {V0, V1, V2, V3, RS1, Sel2};
    SDValue Node = DAG.getNode(RISCVISD::ESP_SRCMB_S8_QACC_M, DL, VTs, Ops);

    return Node;
  }
  case Intrinsic::riscv_esp_srcmb_u16_qacc_m: {
    // Lower SRCMB U16 QACC intrinsic
    // Intrinsic: (v0, v1, v2, v3, rs1, sel2) -> v8i16
    // v0-v3: 4x128-bit QACC (QACC_L[127:0], QACC_L[255:128], QACC_H[127:0],
    // QACC_H[255:128]) SDNode: (v0, v1, v2, v3, rs1, sel2) -> v8i16 QACC is
    // passed as explicit phantom operands (4x128-bit) for proper data flow
    // tracking
    SDLoc DL(Op);
    SDValue V0 = Op.getOperand(1);   // QACC_L[127:0]
    SDValue V1 = Op.getOperand(2);   // QACC_L[255:128]
    SDValue V2 = Op.getOperand(3);   // QACC_H[127:0]
    SDValue V3 = Op.getOperand(4);   // QACC_H[255:128]
    SDValue RS1 = Op.getOperand(5);  // Shift amount
    SDValue Sel2 = Op.getOperand(6); // Saturation select

    // Create SDNode with QACC as explicit phantom operands (4x128-bit)
    // SDNode returns: v8i16 (qu)
    // Operands: (v0, v1, v2, v3, rs1, sel2) - QACC as 4x128-bit phantom
    // operands
    SDVTList VTs = DAG.getVTList(MVT::v8i16);
    SDValue Ops[] = {V0, V1, V2, V3, RS1, Sel2};
    SDValue Node = DAG.getNode(RISCVISD::ESP_SRCMB_U16_QACC_M, DL, VTs, Ops);

    return Node;
  }
  case Intrinsic::riscv_esp_srcmb_u8_qacc_m: {
    // Lower SRCMB U8 QACC intrinsic
    // Intrinsic: (v0, v1, v2, v3, rs1, sel2) -> v16i8
    // v0-v3: 4x128-bit QACC (QACC_L[127:0], QACC_L[255:128], QACC_H[127:0],
    // QACC_H[255:128]) SDNode: (v0, v1, v2, v3, rs1, sel2) -> v16i8 QACC is
    // passed as explicit phantom operands (4x128-bit) for proper data flow
    // tracking
    SDLoc DL(Op);
    SDValue V0 = Op.getOperand(1);   // QACC_L[127:0]
    SDValue V1 = Op.getOperand(2);   // QACC_L[255:128]
    SDValue V2 = Op.getOperand(3);   // QACC_H[127:0]
    SDValue V3 = Op.getOperand(4);   // QACC_H[255:128]
    SDValue RS1 = Op.getOperand(5);  // Shift amount
    SDValue Sel2 = Op.getOperand(6); // Saturation select

    // Create SDNode with QACC as explicit phantom operands (4x128-bit)
    // SDNode returns: v16i8 (qu)
    // Operands: (v0, v1, v2, v3, rs1, sel2) - QACC as 4x128-bit phantom
    // operands
    SDVTList VTs = DAG.getVTList(MVT::v16i8);
    SDValue Ops[] = {V0, V1, V2, V3, RS1, Sel2};
    SDValue Node = DAG.getNode(RISCVISD::ESP_SRCMB_U8_QACC_M, DL, VTs, Ops);

    return Node;
  }
  case Intrinsic::riscv_esp_vmulas_s16_qacc_m: {
    // Lower VMULAS S16 QACC pure compute intrinsic
    // Intrinsic: (int_id, v0, v1, v2, v3, qx, qy) -> {v16i8, v16i8, v16i8,
    // v16i8} SDNode returns: (v16i8, v16i8, v16i8, v16i8) - 4x128-bit QACC
    // directly Passthru is passed as 4x128-bit explicit phantom operands
    SDLoc DL(Op);
    SDValue V0In = Op.getOperand(1); // QACC_L[127:0] passthru (v16i8)
    SDValue V1In = Op.getOperand(2); // QACC_L[255:128] passthru (v16i8)
    SDValue V2In = Op.getOperand(3); // QACC_H[127:0] passthru (v16i8)
    SDValue V3In = Op.getOperand(4); // QACC_H[255:128] passthru (v16i8)
    SDValue QX = Op.getOperand(5);
    SDValue QY = Op.getOperand(6);

    // SDNode returns: (v16i8, v16i8, v16i8, v16i8) - 4x128-bit QACC directly
    // Operands: (v0, v1, v2, v3, qx, qy) - 4x128-bit passthru as explicit
    // phantom operands
    SmallVector<EVT, 4> VTList = {MVT::v16i8, MVT::v16i8, MVT::v16i8,
                                  MVT::v16i8};
    SDVTList VTs = DAG.getVTList(VTList);
    SDValue Ops[] = {V0In, V1In, V2In, V3In, QX, QY};
    SDValue Node = DAG.getNode(RISCVISD::ESP_VMULAS_S16_QACC_M, DL, VTs, Ops);

    // Return structure with 4x128-bit QACC directly
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1),
                               Node.getValue(2), Node.getValue(3)},
                              DL);
  }
  case Intrinsic::riscv_esp_vmulas_s8_qacc_m: {
    // Lower VMULAS S8 QACC pure compute intrinsic
    // Intrinsic: (int_id, v0, v1, v2, v3, qx, qy) -> {v16i8, v16i8, v16i8,
    // v16i8} SDNode returns: (v16i8, v16i8, v16i8, v16i8) - 4x128-bit QACC
    // directly Passthru is passed as 4x128-bit explicit phantom operands
    SDLoc DL(Op);
    SDValue V0In = Op.getOperand(1); // QACC_L[127:0] passthru (v16i8)
    SDValue V1In = Op.getOperand(2); // QACC_L[255:128] passthru (v16i8)
    SDValue V2In = Op.getOperand(3); // QACC_H[127:0] passthru (v16i8)
    SDValue V3In = Op.getOperand(4); // QACC_H[255:128] passthru (v16i8)
    SDValue QX = Op.getOperand(5);
    SDValue QY = Op.getOperand(6);

    // SDNode returns: (v16i8, v16i8, v16i8, v16i8) - 4x128-bit QACC directly
    // Operands: (v0, v1, v2, v3, qx, qy) - 4x128-bit passthru as explicit
    // phantom operands
    SmallVector<EVT, 4> VTList = {MVT::v16i8, MVT::v16i8, MVT::v16i8,
                                  MVT::v16i8};
    SDVTList VTs = DAG.getVTList(VTList);
    SDValue Ops[] = {V0In, V1In, V2In, V3In, QX, QY};
    SDValue Node = DAG.getNode(RISCVISD::ESP_VMULAS_S8_QACC_M, DL, VTs, Ops);

    // Return structure with 4x128-bit QACC directly
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1),
                               Node.getValue(2), Node.getValue(3)},
                              DL);
  }
  case Intrinsic::riscv_esp_vmulas_u16_qacc_m: {
    // Lower VMULAS U16 QACC pure compute intrinsic
    // Intrinsic: (int_id, v0, v1, v2, v3, qx, qy) -> {v16i8, v16i8, v16i8,
    // v16i8} SDNode returns: (v16i8, v16i8, v16i8, v16i8) - 4x128-bit QACC
    // directly Passthru is passed as 4x128-bit explicit phantom operands
    SDLoc DL(Op);
    SDValue V0In = Op.getOperand(1); // QACC_L[127:0] passthru (v16i8)
    SDValue V1In = Op.getOperand(2); // QACC_L[255:128] passthru (v16i8)
    SDValue V2In = Op.getOperand(3); // QACC_H[127:0] passthru (v16i8)
    SDValue V3In = Op.getOperand(4); // QACC_H[255:128] passthru (v16i8)
    SDValue QX = Op.getOperand(5);
    SDValue QY = Op.getOperand(6);

    // SDNode returns: (v16i8, v16i8, v16i8, v16i8) - 4x128-bit QACC directly
    // Operands: (v0, v1, v2, v3, qx, qy) - 4x128-bit passthru as explicit
    // phantom operands
    SmallVector<EVT, 4> VTList = {MVT::v16i8, MVT::v16i8, MVT::v16i8,
                                  MVT::v16i8};
    SDVTList VTs = DAG.getVTList(VTList);
    SDValue Ops[] = {V0In, V1In, V2In, V3In, QX, QY};
    SDValue Node = DAG.getNode(RISCVISD::ESP_VMULAS_U16_QACC_M, DL, VTs, Ops);

    // Return structure with 4x128-bit QACC directly
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1),
                               Node.getValue(2), Node.getValue(3)},
                              DL);
  }
  case Intrinsic::riscv_esp_vmulas_u8_qacc_m: {
    // Lower VMULAS U8 QACC pure compute intrinsic
    // Intrinsic: (int_id, v0, v1, v2, v3, qx, qy) -> {v16i8, v16i8, v16i8,
    // v16i8} SDNode returns: (v16i8, v16i8, v16i8, v16i8) - 4x128-bit QACC
    // directly Passthru is passed as 4x128-bit explicit phantom operands
    SDLoc DL(Op);
    SDValue V0In = Op.getOperand(1); // QACC_L[127:0] passthru (v16i8)
    SDValue V1In = Op.getOperand(2); // QACC_L[255:128] passthru (v16i8)
    SDValue V2In = Op.getOperand(3); // QACC_H[127:0] passthru (v16i8)
    SDValue V3In = Op.getOperand(4); // QACC_H[255:128] passthru (v16i8)
    SDValue QX = Op.getOperand(5);
    SDValue QY = Op.getOperand(6);

    // SDNode returns: (v16i8, v16i8, v16i8, v16i8) - 4x128-bit QACC directly
    // Operands: (v0, v1, v2, v3, qx, qy) - 4x128-bit passthru as explicit
    // phantom operands
    SmallVector<EVT, 4> VTList = {MVT::v16i8, MVT::v16i8, MVT::v16i8,
                                  MVT::v16i8};
    SDVTList VTs = DAG.getVTList(VTList);
    SDValue Ops[] = {V0In, V1In, V2In, V3In, QX, QY};
    SDValue Node = DAG.getNode(RISCVISD::ESP_VMULAS_U8_QACC_M, DL, VTs, Ops);

    // Return structure with 4x128-bit QACC directly
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1),
                               Node.getValue(2), Node.getValue(3)},
                              DL);
  }
  default:
    return SDValue(); // Not an ESPV intrinsic handled here
  }
}

// VMULAS QACC LD IP Lowering
static SDValue LowerVMULASQACCLDIP(SDValue Op, SelectionDAG &DAG,
                                   unsigned ISDOpcode) {
  // Intrinsic: (chain, int_id, v0, v1, v2, v3, qx, qy, ptr, offset) -> {ptr,
  // qu, v0, v1, v2, v3, chain} SDNode returns: (qu, ptr, v16i8, v16i8, v16i8,
  // v16i8, chain) - qu + ptr + 4x128-bit QACC + chain SDNode operands: (chain,
  // v0, v1, v2, v3, qx, qy, ptr, offset) - 4x128-bit passthru as explicit
  // phantom operands
  SDLoc DL(Op);
  SDValue Chain = Op.getOperand(0);
  SDValue V0In = Op.getOperand(2); // QACC_L[127:0] passthru (v16i8)
  SDValue V1In = Op.getOperand(3); // QACC_L[255:128] passthru (v16i8)
  SDValue V2In = Op.getOperand(4); // QACC_H[127:0] passthru (v16i8)
  SDValue V3In = Op.getOperand(5); // QACC_H[255:128] passthru (v16i8)
  SDValue QX = Op.getOperand(6);
  SDValue QY = Op.getOperand(7);
  SDValue Ptr = Op.getOperand(8);
  SDValue Offset = Op.getOperand(9);

  EVT PtrVT = Ptr.getValueType();
  EVT MemVT = MVT::v16i8;
  // SDNode returns: (qu, ptr, v16i8, v16i8, v16i8, v16i8, chain) - 7 outputs
  // (Glue removed)
  SmallVector<EVT, 7> VTList = {
      MVT::v16i8, PtrVT,      MVT::v16i8,
      MVT::v16i8, MVT::v16i8, MVT::v16i8, // qu + ptr + 4x128-bit QACC
      MVT::Other                          // Chain only, no Glue
  };
  SDVTList VTs = DAG.getVTList(VTList);
  // SDNode operands: (chain, v0, v1, v2, v3, qx, qy, ptr, offset) - 9 operands
  // (Glue removed)
  SDValue Ops[] = {Chain, V0In, V1In, V2In, V3In, QX, QY, Ptr, Offset};

  // Note: This intrinsic always arrives as MemIntrinsicSDNode because
  //       getTgtMemIntrinsic returns true for it.
  auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
  MachineMemOperand *MMO = MemIntr->getMemOperand();
  SDValue Node = DAG.getMemIntrinsicNode(ISDOpcode, DL, VTs, Ops, MemVT, MMO);
  SDValue Qu = Node.getValue(0);     // qu (Result 0) - v16i8
  SDValue PtrOut = Node.getValue(1); // Updated pointer (Result 1)
  SDValue V0 = Node.getValue(2);     // QACC_L[127:0] output (Result 2) - v16i8
  SDValue V1 = Node.getValue(3); // QACC_L[255:128] output (Result 3) - v16i8
  SDValue V2 = Node.getValue(4); // QACC_H[127:0] output (Result 4) - v16i8
  SDValue V3 = Node.getValue(5); // QACC_H[255:128] output (Result 5) - v16i8
  Chain = Node.getValue(6);      // Chain (Result 6)

  return DAG.getMergeValues({PtrOut, Qu, V0, V1, V2, V3, Chain}, DL);
}

// VMULAS QACC LD XP Lowering
static SDValue LowerVMULASQACCLDXP(SDValue Op, SelectionDAG &DAG,
                                   unsigned ISDOpcode) {
  // Intrinsic: (chain, int_id, v0, v1, v2, v3, qx, qy, ptr, rs2) -> {ptr, qu,
  // v0, v1, v2, v3, chain} SDNode returns: (qu, ptr, v16i8, v16i8, v16i8,
  // v16i8, chain) - qu + ptr + 4x128-bit QACC + chain SDNode operands: (chain,
  // v0, v1, v2, v3, qx, qy, ptr, rs2) - 4x128-bit passthru as explicit phantom
  // operands
  SDLoc DL(Op);
  SDValue Chain = Op.getOperand(0);
  SDValue V0In = Op.getOperand(2); // QACC_L[127:0] passthru (v16i8)
  SDValue V1In = Op.getOperand(3); // QACC_L[255:128] passthru (v16i8)
  SDValue V2In = Op.getOperand(4); // QACC_H[127:0] passthru (v16i8)
  SDValue V3In = Op.getOperand(5); // QACC_H[255:128] passthru (v16i8)
  SDValue QX = Op.getOperand(6);
  SDValue QY = Op.getOperand(7);
  SDValue Ptr = Op.getOperand(8);
  SDValue Rs2 = Op.getOperand(9);

  EVT PtrVT = Ptr.getValueType();
  EVT MemVT = MVT::v16i8;
  // SDNode returns: (qu, ptr, v16i8, v16i8, v16i8, v16i8, chain) - 7 outputs
  // (Glue removed)
  SmallVector<EVT, 7> VTList = {
      MVT::v16i8, PtrVT,      MVT::v16i8,
      MVT::v16i8, MVT::v16i8, MVT::v16i8, // qu + ptr + 4x128-bit QACC
      MVT::Other                          // Chain only, no Glue
  };
  SDVTList VTs = DAG.getVTList(VTList);
  // SDNode operands: (chain, v0, v1, v2, v3, qx, qy, ptr, rs2) - 9 operands
  // (Glue removed)
  SDValue Ops[] = {Chain, V0In, V1In, V2In, V3In, QX, QY, Ptr, Rs2};

  // Note: This intrinsic always arrives as MemIntrinsicSDNode because
  //       getTgtMemIntrinsic returns true for it.
  auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
  MachineMemOperand *MMO = MemIntr->getMemOperand();
  SDValue Node = DAG.getMemIntrinsicNode(ISDOpcode, DL, VTs, Ops, MemVT, MMO);
  SDValue Qu = Node.getValue(0);     // qu (Result 0) - v16i8
  SDValue PtrOut = Node.getValue(1); // Updated pointer (Result 1)
  SDValue V0 = Node.getValue(2);     // QACC_L[127:0] output (Result 2) - v16i8
  SDValue V1 = Node.getValue(3); // QACC_L[255:128] output (Result 3) - v16i8
  SDValue V2 = Node.getValue(4); // QACC_H[127:0] output (Result 4) - v16i8
  SDValue V3 = Node.getValue(5); // QACC_H[255:128] output (Result 5) - v16i8
  Chain = Node.getValue(6);      // Chain (Result 6)

  return DAG.getMergeValues({PtrOut, Qu, V0, V1, V2, V3, Chain}, DL);
}

// VMULAS QACC ST IP Lowering
static SDValue LowerVMULASQACCSTIP(SDValue Op, SelectionDAG &DAG,
                                   unsigned ISDOpcode) {
  // Intrinsic: (chain, int_id, v0, v1, v2, v3, qu, qx, qy, ptr, offset) ->
  // {ptr, v0, v1, v2, v3, chain}
  SDLoc DL(Op);
  SDValue Chain = Op.getOperand(0);
  SDValue V0In = Op.getOperand(2); // QACC_L[127:0] passthru (v16i8)
  SDValue V1In = Op.getOperand(3); // QACC_L[255:128] passthru (v16i8)
  SDValue V2In = Op.getOperand(4); // QACC_H[127:0] passthru (v16i8)
  SDValue V3In = Op.getOperand(5); // QACC_H[255:128] passthru (v16i8)
  SDValue QU = Op.getOperand(6);
  SDValue QX = Op.getOperand(7);
  SDValue QY = Op.getOperand(8);
  SDValue Ptr = Op.getOperand(9);
  SDValue Offset = Op.getOperand(10);

  EVT PtrVT = Ptr.getValueType();
  EVT MemVT = MVT::v16i8;
  // SDNode returns: (ptr, v16i8, v16i8, v16i8, v16i8, chain) - 6 outputs (no
  // glue)
  SmallVector<EVT, 6> VTList = {
      PtrVT,      MVT::v16i8, MVT::v16i8,
      MVT::v16i8, MVT::v16i8, // ptr + 4x128-bit QACC
      MVT::Other              // Chain
  };
  SDVTList VTs = DAG.getVTList(VTList);
  // SDNode operands: (chain, v0, v1, v2, v3, qu, qx, qy, ptr, offset) - 10
  // operands total Note: SDNPHasChain doesn't automatically add Chain, we must
  // pass it explicitly
  SDValue Ops[] = {Chain, V0In, V1In, V2In, V3In, QU, QX, QY, Ptr, Offset};

  // Note: This intrinsic always arrives as MemIntrinsicSDNode because
  //       getTgtMemIntrinsic returns true for it.
  auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
  MachineMemOperand *MMO = MemIntr->getMemOperand();
  SDValue Node = DAG.getMemIntrinsicNode(ISDOpcode, DL, VTs, Ops, MemVT, MMO);
  SDValue PtrOut = Node.getValue(0); // Updated pointer (Result 0)
  SDValue V0 = Node.getValue(1);     // QACC_L[127:0] output (Result 1) - v16i8
  SDValue V1 = Node.getValue(2); // QACC_L[255:128] output (Result 2) - v16i8
  SDValue V2 = Node.getValue(3); // QACC_H[127:0] output (Result 3) - v16i8
  SDValue V3 = Node.getValue(4); // QACC_H[255:128] output (Result 4) - v16i8
  Chain = Node.getValue(5);      // Chain (Result 5)
  return DAG.getMergeValues({PtrOut, V0, V1, V2, V3, Chain}, DL);
}

// VMULAS QACC ST XP Lowering
static SDValue LowerVMULASQACCSTXP(SDValue Op, SelectionDAG &DAG,
                                   unsigned ISDOpcode) {
  // Intrinsic: (chain, int_id, v0, v1, v2, v3, qu, qx, qy, ptr, rs2) -> {ptr,
  // v0, v1, v2, v3, chain}
  SDLoc DL(Op);
  SDValue Chain = Op.getOperand(0);
  SDValue V0In = Op.getOperand(2); // QACC_L[127:0] passthru (v16i8)
  SDValue V1In = Op.getOperand(3); // QACC_L[255:128] passthru (v16i8)
  SDValue V2In = Op.getOperand(4); // QACC_H[127:0] passthru (v16i8)
  SDValue V3In = Op.getOperand(5); // QACC_H[255:128] passthru (v16i8)
  SDValue QU = Op.getOperand(6);
  SDValue QX = Op.getOperand(7);
  SDValue QY = Op.getOperand(8);
  SDValue Ptr = Op.getOperand(9);
  SDValue Rs2 = Op.getOperand(10);

  EVT PtrVT = Ptr.getValueType();
  EVT MemVT = MVT::v16i8;
  // SDNode returns: (ptr, v16i8, v16i8, v16i8, v16i8, chain) - 6 outputs (no
  // glue)
  SmallVector<EVT, 6> VTList = {
      PtrVT,      MVT::v16i8, MVT::v16i8,
      MVT::v16i8, MVT::v16i8, // ptr + 4x128-bit QACC
      MVT::Other              // Chain
  };
  SDVTList VTs = DAG.getVTList(VTList);
  // SDNode operands: (chain, v0, v1, v2, v3, qu, qx, qy, ptr, rs2) - 10
  // operands total Note: SDNPHasChain doesn't automatically add Chain, we must
  // pass it explicitly
  SDValue Ops[] = {Chain, V0In, V1In, V2In, V3In, QU, QX, QY, Ptr, Rs2};

  // Note: This intrinsic always arrives as MemIntrinsicSDNode because
  //       getTgtMemIntrinsic returns true for it.
  auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
  MachineMemOperand *MMO = MemIntr->getMemOperand();
  SDValue Node = DAG.getMemIntrinsicNode(ISDOpcode, DL, VTs, Ops, MemVT, MMO);
  SDValue PtrOut = Node.getValue(0); // Updated pointer (Result 0)
  SDValue V0 = Node.getValue(1);     // QACC_L[127:0] output (Result 1) - v16i8
  SDValue V1 = Node.getValue(2); // QACC_L[255:128] output (Result 2) - v16i8
  SDValue V2 = Node.getValue(3); // QACC_H[127:0] output (Result 3) - v16i8
  SDValue V3 = Node.getValue(4); // QACC_H[255:128] output (Result 4) - v16i8
  Chain = Node.getValue(5);      // Chain (Result 5)
  return DAG.getMergeValues({PtrOut, V0, V1, V2, V3, Chain}, DL);
}

// VMULAS QACC LDBC INCP Lowering
static SDValue LowerVMULASQACCLDBCINCP(SDValue Op, SelectionDAG &DAG,
                                       unsigned ISDOpcode) {
  // Intrinsic: (chain, int_id, v0, v1, v2, v3, qx, qy, ptr) -> {qu, ptr, v0,
  // v1, v2, v3, chain} SDNode returns: (qu, ptr, v16i8, v16i8, v16i8, v16i8,
  // chain) - qu + ptr + 4x128-bit QACC + chain SDNode operands: (chain, v0, v1,
  // v2, v3, qx, qy, ptr) - 4x128-bit passthru as explicit phantom operands
  // Note: LDBC.INCP doesn't need offset (fixed increment by 2)
  SDLoc DL(Op);
  SDValue Chain = Op.getOperand(0);
  SDValue V0In = Op.getOperand(2); // QACC_L[127:0] passthru (v16i8)
  SDValue V1In = Op.getOperand(3); // QACC_L[255:128] passthru (v16i8)
  SDValue V2In = Op.getOperand(4); // QACC_H[127:0] passthru (v16i8)
  SDValue V3In = Op.getOperand(5); // QACC_H[255:128] passthru (v16i8)
  SDValue QX = Op.getOperand(6);
  SDValue QY = Op.getOperand(7);
  SDValue Ptr = Op.getOperand(8);

  EVT PtrVT = Ptr.getValueType();
  EVT MemVT = MVT::v16i8;
  // SDNode returns: (qu, ptr, v16i8, v16i8, v16i8, v16i8, chain) - 7 outputs
  // (Glue removed)
  SmallVector<EVT, 7> VTList = {
      MVT::v16i8, PtrVT,      MVT::v16i8,
      MVT::v16i8, MVT::v16i8, MVT::v16i8, // qu + ptr + 4x128-bit QACC
      MVT::Other                          // Chain only, no Glue
  };
  SDVTList VTs = DAG.getVTList(VTList);
  // SDNode operands: (chain, v0, v1, v2, v3, qx, qy, ptr) - 8 operands (Glue
  // removed, offset removed)
  SDValue Ops[] = {Chain, V0In, V1In, V2In, V3In, QX, QY, Ptr};

  // Note: This intrinsic always arrives as MemIntrinsicSDNode because
  //       getTgtMemIntrinsic returns true for it.
  auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
  MachineMemOperand *MMO = MemIntr->getMemOperand();
  SDValue Node = DAG.getMemIntrinsicNode(ISDOpcode, DL, VTs, Ops, MemVT, MMO);
  SDValue Qu = Node.getValue(0);     // qu (Result 0) - v16i8
  SDValue PtrOut = Node.getValue(1); // Updated pointer (Result 1)
  SDValue V0 = Node.getValue(2);     // QACC_L[127:0] output (Result 2) - v16i8
  SDValue V1 = Node.getValue(3); // QACC_L[255:128] output (Result 3) - v16i8
  SDValue V2 = Node.getValue(4); // QACC_H[127:0] output (Result 4) - v16i8
  SDValue V3 = Node.getValue(5); // QACC_H[255:128] output (Result 5) - v16i8
  Chain = Node.getValue(6);      // Chain (Result 6)

  return DAG.getMergeValues({Qu, PtrOut, V0, V1, V2, V3, Chain}, DL);
}

} // namespace RISCV
} // namespace llvm
