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
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>

using namespace llvm;
using namespace llvm::ISD;

namespace llvm {
namespace RISCV {

static SDValue LowerSTXACCIP(SDValue Op, SelectionDAG &DAG, unsigned ISDOpcode);

// ESPV intrinsic lowering for INTRINSIC_W_CHAIN
SDValue lowerESPVIntrinsicWChain(SDValue Op, SelectionDAG &DAG,
                                 const RISCVSubtarget &Subtarget) {
  if (!Subtarget.hasVendorXespv())
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

    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_ST_QACC_H_H_128_IP_M,
                                           DL, VTs, Ops, VecVT, MMO);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1)}, DL);
  }
  case Intrinsic::riscv_esp_st_s_xacc_ip_m:
    return LowerSTXACCIP(Op, DAG, RISCVISD::ESP_ST_S_XACC_IP_M);

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
  // xacc_high_unchanged, chain, glue) Note: This intrinsic always arrives as
  // MemIntrinsicSDNode because
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
  default:
    return SDValue(); // Not an ESPV intrinsic handled here
  }
}

} // namespace RISCV
} // namespace llvm
