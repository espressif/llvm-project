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
    // Get MMO from the original intrinsic node to preserve memory information
    // Note: This intrinsic always arrives as MemIntrinsicSDNode because
    //       getTgtMemIntrinsic returns true for it.
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_VLD_128_IP_M, DL, VTs,
                                           Ops, VecVT, MMO);
    return DAG.getMergeValues(
        {Node.getValue(0), Node.getValue(1), Node.getValue(2)}, DL);
  }
  case Intrinsic::riscv_esp_vld_128_xp_m: {
    // Lower intrinsic to custom SDNode that will be matched to
    // ESP_VLD_128_XP_M_P Intrinsic: (chain, int_id, ptr, offset_reg)
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue Ptr = Op.getOperand(2);
    SDValue Offset = Op.getOperand(3); // Register offset

    EVT VecVT = MVT::v16i8;
    EVT PtrVT = Ptr.getValueType();
    SDVTList VTs = DAG.getVTList(VecVT, PtrVT, MVT::Other);

    SDValue Ops[] = {Chain, Ptr, Offset};
    // Get MMO from the original intrinsic node to preserve memory information
    // Note: This intrinsic always arrives as MemIntrinsicSDNode because
    //       getTgtMemIntrinsic returns true for it.
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_VLD_128_XP_M, DL, VTs,
                                           Ops, VecVT, MMO);
    return DAG.getMergeValues(
        {Node.getValue(0), Node.getValue(1), Node.getValue(2)}, DL);
  }
  case Intrinsic::riscv_esp_vst_128_ip_m: {
    // Lower intrinsic to custom SDNode that will be matched to
    // ESP_VST_128_IP_M_P Intrinsic: (chain, int_id, vec, ptr, imm)
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue Vec = Op.getOperand(2);
    SDValue Ptr = Op.getOperand(3);
    SDValue Imm = Op.getOperand(4);

    EVT VecVT = MVT::v16i8;
    EVT PtrVT = Ptr.getValueType();
    SDVTList VTs = DAG.getVTList(PtrVT, MVT::Other);

    SDValue Ops[] = {Chain, Vec, Ptr, Imm};
    // Get MMO from the original intrinsic node to preserve memory information
    // Note: This intrinsic always arrives as MemIntrinsicSDNode because
    //       getTgtMemIntrinsic returns true for it.
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
    // ESP_VST_128_XP_M_P Intrinsic: (chain, int_id, vec, ptr, offset_reg)
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue Vec = Op.getOperand(2);
    SDValue Ptr = Op.getOperand(3);
    SDValue Offset = Op.getOperand(4); // Register offset

    EVT VecVT = MVT::v16i8;
    EVT PtrVT = Ptr.getValueType();
    SDVTList VTs = DAG.getVTList(PtrVT, MVT::Other);

    SDValue Ops[] = {Chain, Vec, Ptr, Offset};
    // Get MMO from the original intrinsic node to preserve memory information
    // Note: This intrinsic always arrives as MemIntrinsicSDNode because
    //       getTgtMemIntrinsic returns true for it.
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

  default:
    return SDValue(); // Not an ESPV intrinsic handled here
  }
}

} // namespace RISCV
} // namespace llvm
