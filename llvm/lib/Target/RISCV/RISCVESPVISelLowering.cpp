//===-- RISCVESPVISelLowering.cpp - ESPV DAG Lowering Implementation ------===//
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
#include "RISCVRegisterInfo.h"
#include "RISCVSelectionDAGInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>

using namespace llvm;
using namespace llvm::ISD;

namespace {
constexpr unsigned ESPV_RM_DYNAMIC = 7;

static SDValue lowerCmulTargetImm(SelectionDAG &DAG, const SDLoc &DL,
                                  SDValue V) {
  if (V.getOpcode() == ISD::TargetConstant)
    return V;
  if (auto *C = dyn_cast<ConstantSDNode>(V.getNode()))
    return DAG.getTargetConstant(C->getZExtValue(), DL, MVT::i32);
  return V;
}

static void diagnoseESPV21SatRm(SelectionDAG &DAG, SDValue Sat, SDValue Rm) {
  const Function &F = DAG.getMachineFunction().getFunction();
  auto Diagnose = [&](const char *Msg) {
    F.getContext().diagnose(DiagnosticInfoUnsupported{F, Msg});
  };
  if (auto *C = dyn_cast<ConstantSDNode>(Sat.getNode())) {
    if (C->getZExtValue() != 0)
      Diagnose("sat must be 0 for PIE 2.1 (+xespv2p1)");
  } else {
    Diagnose("sat must be a constant immediate for PIE 2.1 (+xespv2p1)");
  }
  if (auto *C = dyn_cast<ConstantSDNode>(Rm.getNode())) {
    if (C->getZExtValue() != ESPV_RM_DYNAMIC)
      Diagnose("rm must be RM_DYNAMIC (7) for PIE 2.1 (+xespv2p1)");
  } else {
    Diagnose("rm must be a constant immediate for PIE 2.1 (+xespv2p1)");
  }
}

static void diagnoseESPV21CmulSatRm(SelectionDAG &DAG, SDValue Sat,
                                    SDValue Rm) {
  const Function &F = DAG.getMachineFunction().getFunction();
  auto Diagnose = [&](const char *Msg) {
    F.getContext().diagnose(DiagnosticInfoUnsupported{F, Msg});
  };
  if (auto *C = dyn_cast<ConstantSDNode>(Sat.getNode())) {
    if (C->getZExtValue() != 0)
      Diagnose("cmul sat must be 0 for PIE 2.1 (+xespv2p1)");
  } else {
    Diagnose("cmul sat must be a constant immediate for PIE 2.1 (+xespv2p1)");
  }
  if (auto *C = dyn_cast<ConstantSDNode>(Rm.getNode())) {
    if (C->getZExtValue() != ESPV_RM_DYNAMIC)
      Diagnose("cmul rm must be RM_DYNAMIC (7) for PIE 2.1 (+xespv2p1)");
  } else {
    Diagnose("cmul rm must be a constant immediate for PIE 2.1 (+xespv2p1)");
  }
}

static SDValue lowerCmulBasic(SDValue Op, SelectionDAG &DAG,
                              const RISCVSubtarget &Subtarget, MVT RetVT,
                              unsigned ISD21, unsigned ISD22) {
  SDLoc DL(Op);
  SDValue QX = Op.getOperand(2);
  SDValue QY = Op.getOperand(3);
  SDValue SEL4 = Op.getOperand(4);
  SDValue SAT = Op.getOperand(5);
  SDValue RM = Op.getOperand(6);
  SDValue Sar = Op.getOperand(7);
  if (Subtarget.hasVendorXespv2p1())
    diagnoseESPV21CmulSatRm(DAG, SAT, RM);
  if (Subtarget.useESPV2P2Instructions()) {
    SDVTList VTs = DAG.getVTList(RetVT);
    SDValue Ops[] = {QX, QY, lowerCmulTargetImm(DAG, DL, SEL4),
                     lowerCmulTargetImm(DAG, DL, SAT),
                     lowerCmulTargetImm(DAG, DL, RM)};
    return DAG.getNode(ISD22, DL, VTs, Ops);
  }
  return SDValue();
}

static void diagnoseESPV21Rm(SelectionDAG &DAG, SDValue Rm) {
  const Function &F = DAG.getMachineFunction().getFunction();
  auto Diagnose = [&](const char *Msg) {
    F.getContext().diagnose(DiagnosticInfoUnsupported{F, Msg});
  };
  if (auto *C = dyn_cast<ConstantSDNode>(Rm.getNode())) {
    if (C->getZExtValue() != ESPV_RM_DYNAMIC)
      Diagnose("rm must be RM_DYNAMIC (7) for PIE 2.1 (+xespv2p1)");
  } else {
    Diagnose("rm must be a constant immediate for PIE 2.1 (+xespv2p1)");
  }
}

// PIE 2.2 CFG writable: mis_st[0], mis_ld[1], vxrm[6:4]. vxsat_en is
// PIE 2.1-only (see pie-merge-buckets/03-pie21-only.csv); masked on +xespv
// writes.
static constexpr unsigned ESPPie22CfgWritableMask = 0x73;

static SDValue lowerEspMovxCfgWriteValue(SDValue Val, SelectionDAG &DAG,
                                         SDLoc DL,
                                         const RISCVSubtarget &Subtarget) {
  if (!Subtarget.useESPV2P2Instructions())
    return Val;
  return DAG.getNode(ISD::AND, DL, MVT::i32, Val,
                     DAG.getConstant(ESPPie22CfgWritableMask, DL, MVT::i32));
}

static void diagnoseESPV21Sat(SelectionDAG &DAG, SDValue Sat) {
  const Function &F = DAG.getMachineFunction().getFunction();
  auto Diagnose = [&](const char *Msg) {
    F.getContext().diagnose(DiagnosticInfoUnsupported{F, Msg});
  };
  if (auto *C = dyn_cast<ConstantSDNode>(Sat.getNode())) {
    if (C->getZExtValue() != 0)
      Diagnose("sat must be 0 for PIE 2.1 (+xespv2p1)");
  } else {
    Diagnose("sat must be a constant immediate for PIE 2.1 (+xespv2p1)");
  }
}

static SDValue lowerVaddVsubSatBasic(SDValue Op, SelectionDAG &DAG,
                                     const RISCVSubtarget &Subtarget, MVT RetVT,
                                     unsigned ISD22) {
  SDLoc DL(Op);
  SDValue QX = Op.getOperand(1);
  SDValue QY = Op.getOperand(2);
  SDValue SAT = Op.getOperand(3);
  if (Subtarget.hasVendorXespv2p1())
    diagnoseESPV21Sat(DAG, SAT);
  if (Subtarget.useESPV2P2Instructions()) {
    SDVTList VTs = DAG.getVTList(RetVT);
    SDValue Ops[] = {QX, QY, lowerCmulTargetImm(DAG, DL, SAT)};
    return DAG.getNode(ISD22, DL, VTs, Ops);
  }
  return SDValue();
}

static SDValue lowerVsaddsVssubsSatBasic(SDValue Op, SelectionDAG &DAG,
                                         const RISCVSubtarget &Subtarget,
                                         MVT RetVT, unsigned ISD22) {
  SDLoc DL(Op);
  SDValue QX = Op.getOperand(1);
  SDValue RS1 = Op.getOperand(2);
  SDValue SAT = Op.getOperand(3);
  if (Subtarget.hasVendorXespv2p1())
    diagnoseESPV21Sat(DAG, SAT);
  if (Subtarget.useESPV2P2Instructions()) {
    SDVTList VTs = DAG.getVTList(RetVT);
    SDValue Ops[] = {QX, RS1, lowerCmulTargetImm(DAG, DL, SAT)};
    return DAG.getNode(ISD22, DL, VTs, Ops);
  }
  return SDValue();
}

static SDValue lowerVaddVsubLdIncpSat(SDValue Op, SelectionDAG &DAG,
                                      const RISCVSubtarget &Subtarget, MVT QvVT,
                                      unsigned ISD22) {
  SDLoc DL(Op);
  SDValue Chain = Op.getOperand(0);
  SDValue QX = Op.getOperand(2);
  SDValue QY = Op.getOperand(3);
  SDValue RS1 = Op.getOperand(4);
  SDValue SAT = Op.getOperand(5);
  if (Subtarget.hasVendorXespv2p1())
    diagnoseESPV21Sat(DAG, SAT);
  if (Subtarget.useESPV2P2Instructions()) {
    EVT PtrVT = RS1.getValueType();
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDVTList VTs = DAG.getVTList(QvVT, MVT::v16i8, PtrVT, MVT::Other);
    SDValue Ops[] = {Chain, QX, QY, RS1, lowerCmulTargetImm(DAG, DL, SAT)};
    SDValue Node =
        DAG.getMemIntrinsicNode(ISD22, DL, VTs, Ops, MVT::v16i8, MMO);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1),
                               Node.getValue(2), Node.getValue(3)},
                              DL);
  }
  return SDValue();
}

static SDValue lowerVaddVsubStIncpSat(SDValue Op, SelectionDAG &DAG,
                                      const RISCVSubtarget &Subtarget, MVT QvVT,
                                      unsigned ISD22) {
  SDLoc DL(Op);
  SDValue Chain = Op.getOperand(0);
  SDValue QX = Op.getOperand(2);
  SDValue QY = Op.getOperand(3);
  SDValue QU = Op.getOperand(4);
  SDValue RS1 = Op.getOperand(5);
  SDValue QVIn = Op.getOperand(6);
  SDValue SAT = Op.getOperand(7);
  if (Subtarget.hasVendorXespv2p1())
    diagnoseESPV21Sat(DAG, SAT);
  if (Subtarget.useESPV2P2Instructions()) {
    EVT PtrVT = RS1.getValueType();
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDVTList VTs = DAG.getVTList(QvVT, PtrVT, MVT::Other);
    SDValue Ops[] = {
        Chain, QX, QY, QU, RS1, QVIn, lowerCmulTargetImm(DAG, DL, SAT)};
    SDValue Node =
        DAG.getMemIntrinsicNode(ISD22, DL, VTs, Ops, MVT::v16i8, MMO);
    return DAG.getMergeValues(
        {Node.getValue(0), Node.getValue(1), Node.getValue(2)}, DL);
  }
  return SDValue();
}
static SDValue lowerVsldVsrdBasic(SDValue Op, SelectionDAG &DAG,
                                  const RISCVSubtarget &Subtarget, MVT RetVT,
                                  unsigned ISD22) {
  SDLoc DL(Op);
  SDValue QY = Op.getOperand(1);
  SDValue QW = Op.getOperand(2);
  SDValue SAT = Op.getOperand(3);
  SDValue RM = Op.getOperand(4);
  if (Subtarget.hasVendorXespv2p1())
    diagnoseESPV21CmulSatRm(DAG, SAT, RM);
  if (Subtarget.useESPV2P2Instructions()) {
    SDVTList VTs = DAG.getVTList(RetVT);
    SDValue Ops[] = {QY, QW, lowerCmulTargetImm(DAG, DL, SAT),
                     lowerCmulTargetImm(DAG, DL, RM)};
    return DAG.getNode(ISD22, DL, VTs, Ops);
  }
  return SDValue();
}

static SDValue lowerVsrBasic(SDValue Op, SelectionDAG &DAG,
                             const RISCVSubtarget &Subtarget, MVT RetVT,
                             unsigned ISD22) {
  SDLoc DL(Op);
  SDValue QY = Op.getOperand(1);
  SDValue RM = Op.getOperand(3);
  if (Subtarget.hasVendorXespv2p1())
    diagnoseESPV21Rm(DAG, RM);
  if (Subtarget.useESPV2P2Instructions()) {
    SDVTList VTs = DAG.getVTList(RetVT);
    SDValue Ops[] = {QY, lowerCmulTargetImm(DAG, DL, RM)};
    return DAG.getNode(ISD22, DL, VTs, Ops);
  }
  return SDValue();
}

static SDValue lowerVsl32Basic(SDValue Op, SelectionDAG &DAG,
                               const RISCVSubtarget &Subtarget, MVT RetVT,
                               unsigned ISD22) {
  SDLoc DL(Op);
  SDValue QY = Op.getOperand(1);
  SDValue SAT = Op.getOperand(2);
  if (Subtarget.hasVendorXespv2p1())
    diagnoseESPV21Sat(DAG, SAT);
  if (Subtarget.useESPV2P2Instructions()) {
    SDVTList VTs = DAG.getVTList(RetVT);
    SDValue Ops[] = {QY, lowerCmulTargetImm(DAG, DL, SAT)};
    return DAG.getNode(ISD22, DL, VTs, Ops);
  }
  return SDValue();
}

static SDValue lowerVpreluBasic(SDValue Op, SelectionDAG &DAG,
                                const RISCVSubtarget &Subtarget, MVT RetVT,
                                unsigned ISD22) {
  SDLoc DL(Op);
  SDValue QX = Op.getOperand(1);
  SDValue QY = Op.getOperand(2);
  SDValue RS1 = Op.getOperand(3);
  SDValue SAT = Op.getOperand(4);
  SDValue RM = Op.getOperand(5);
  if (Subtarget.hasVendorXespv2p1())
    diagnoseESPV21CmulSatRm(DAG, SAT, RM);
  if (Subtarget.useESPV2P2Instructions()) {
    SDVTList VTs = DAG.getVTList(RetVT);
    SDValue Ops[] = {QX, QY, RS1, lowerCmulTargetImm(DAG, DL, SAT),
                     lowerCmulTargetImm(DAG, DL, RM)};
    return DAG.getNode(ISD22, DL, VTs, Ops);
  }
  return SDValue();
}

static SDValue lowerVreluBasic(SDValue Op, SelectionDAG &DAG,
                               const RISCVSubtarget &Subtarget, MVT RetVT,
                               unsigned ISD22) {
  SDLoc DL(Op);
  SDValue QY = Op.getOperand(1);
  SDValue RS1 = Op.getOperand(2);
  SDValue RS2 = Op.getOperand(3);
  SDValue SAT = Op.getOperand(4);
  SDValue RM = Op.getOperand(5);
  if (Subtarget.hasVendorXespv2p1())
    diagnoseESPV21CmulSatRm(DAG, SAT, RM);
  if (Subtarget.useESPV2P2Instructions()) {
    SDVTList VTs = DAG.getVTList(RetVT);
    SDValue Ops[] = {QY, RS1, RS2, lowerCmulTargetImm(DAG, DL, SAT),
                     lowerCmulTargetImm(DAG, DL, RM)};
    return DAG.getNode(ISD22, DL, VTs, Ops);
  }
  return SDValue();
}

static SDValue lowerSrcmbSQacc(SDValue Op, SelectionDAG &DAG,
                               const RISCVSubtarget &Subtarget, MVT RetVT,
                               unsigned ISD21, unsigned ISD22,
                               unsigned ShiftOpIdx) {
  SDLoc DL(Op);
  SDValue V0 = Op.getOperand(1);
  SDValue V1 = Op.getOperand(2);
  SDValue V2 = Op.getOperand(3);
  SDValue V3 = Op.getOperand(4);
  SDValue RS1 = Op.getOperand(ShiftOpIdx);
  SDValue SAT = Op.getOperand(ShiftOpIdx + 1);
  SDValue RM = Op.getOperand(ShiftOpIdx + 2);
  if (Subtarget.hasVendorXespv2p1()) {
    diagnoseESPV21Rm(DAG, RM);
    SDVTList VTs = DAG.getVTList(RetVT);
    SDValue Ops[] = {V0, V1, V2, V3, RS1, SAT};
    return DAG.getNode(ISD21, DL, VTs, Ops);
  }
  if (Subtarget.useESPV2P2Instructions()) {
    SDVTList VTs = DAG.getVTList(RetVT);
    SDValue Ops[] = {RS1, lowerCmulTargetImm(DAG, DL, SAT),
                     lowerCmulTargetImm(DAG, DL, RM)};
    return DAG.getNode(ISD22, DL, VTs, Ops);
  }
  return SDValue();
}

static SDValue lowerSrcmbUQacc(SDValue Op, SelectionDAG &DAG,
                               const RISCVSubtarget &Subtarget, MVT RetVT,
                               unsigned ISD21, unsigned ISD22) {
  SDLoc DL(Op);
  SDValue V0 = Op.getOperand(1);
  SDValue V1 = Op.getOperand(2);
  SDValue V2 = Op.getOperand(3);
  SDValue V3 = Op.getOperand(4);
  SDValue RS1 = Op.getOperand(5);
  SDValue SEL2 = Op.getOperand(6);
  SDValue SAT = Op.getOperand(7);
  SDValue RM = Op.getOperand(8);
  if (Subtarget.hasVendorXespv2p1()) {
    diagnoseESPV21CmulSatRm(DAG, SAT, RM);
    SDVTList VTs = DAG.getVTList(RetVT);
    SDValue Ops[] = {V0, V1, V2, V3, RS1, SEL2};
    return DAG.getNode(ISD21, DL, VTs, Ops);
  }
  if (Subtarget.useESPV2P2Instructions()) {
    SDVTList VTs = DAG.getVTList(RetVT);
    SDValue Ops[] = {RS1, lowerCmulTargetImm(DAG, DL, SAT),
                     lowerCmulTargetImm(DAG, DL, RM)};
    return DAG.getNode(ISD22, DL, VTs, Ops);
  }
  return SDValue();
}

static SDValue lowerSrcmbUQQacc(SDValue Op, SelectionDAG &DAG,
                                const RISCVSubtarget &Subtarget, MVT RetVT,
                                unsigned ISD21, unsigned ISD22) {
  SDLoc DL(Op);
  SDValue V0 = Op.getOperand(1);
  SDValue V1 = Op.getOperand(2);
  SDValue V2 = Op.getOperand(3);
  SDValue V3 = Op.getOperand(4);
  SDValue QW = Op.getOperand(5);
  SDValue SEL2 = Op.getOperand(6);
  SDValue SAT = Op.getOperand(7);
  SDValue RM = Op.getOperand(8);
  if (Subtarget.hasVendorXespv2p1()) {
    diagnoseESPV21SatRm(DAG, SAT, RM);
    SDVTList VTs = DAG.getVTList(RetVT);
    SDValue Ops[] = {V0, V1, V2, V3, QW, SEL2};
    return DAG.getNode(ISD21, DL, VTs, Ops);
  }
  if (Subtarget.useESPV2P2Instructions()) {
    SDVTList VTs = DAG.getVTList(RetVT);
    SDValue Ops[] = {QW, lowerCmulTargetImm(DAG, DL, SAT),
                     lowerCmulTargetImm(DAG, DL, RM)};
    return DAG.getNode(ISD22, DL, VTs, Ops);
  }
  return SDValue();
}

static SDValue lowerVcmulasCompute(SDValue Op, SelectionDAG &DAG,
                                   const RISCVSubtarget &Subtarget,
                                   unsigned ISD21, unsigned ISD22) {
  SDLoc DL(Op);
  SDValue V0 = Op.getOperand(1);
  SDValue V1 = Op.getOperand(2);
  SDValue QX = Op.getOperand(3);
  SDValue QY = Op.getOperand(4);
  SDValue SAT = Op.getOperand(5);
  SDVTList VTs = DAG.getVTList(MVT::v16i8, MVT::v16i8);
  if (Subtarget.useESPV2P2Instructions()) {
    SDValue Ops[] = {V0, V1, QX, QY, lowerCmulTargetImm(DAG, DL, SAT)};
    SDValue Node = DAG.getNode(ISD22, DL, VTs, Ops);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1)}, DL);
  }
  if (Subtarget.hasVendorXespv2p1()) {
    diagnoseESPV21Sat(DAG, SAT);
    SDValue Ops[] = {V0, V1, QX, QY};
    SDValue Node = DAG.getNode(ISD21, DL, VTs, Ops);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1)}, DL);
  }
  return SDValue();
}

static SDValue lowerVcmulasLdIp(SDValue Op, SelectionDAG &DAG,
                                const RISCVSubtarget &Subtarget, unsigned ISD21,
                                unsigned ISD22) {
  SDLoc DL(Op);
  SDValue Chain = Op.getOperand(0);
  SDValue V0 = Op.getOperand(2);
  SDValue V1 = Op.getOperand(3);
  SDValue QX = Op.getOperand(4);
  SDValue QY = Op.getOperand(5);
  SDValue Ptr = Op.getOperand(6);
  SDValue Offset = Op.getOperand(7);
  SDValue SAT = Op.getOperand(8);
  if (Subtarget.hasVendorXespv2p1() && !Subtarget.useESPV2P2Instructions())
    diagnoseESPV21Sat(DAG, SAT);
  EVT PtrVT = Ptr.getValueType();
  auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
  MachineMemOperand *MMO = MemIntr->getMemOperand();
  SmallVector<EVT, 5> VTList = {MVT::v16i8, PtrVT, MVT::v16i8, MVT::v16i8,
                                MVT::Other};
  SDVTList VTs = DAG.getVTList(VTList);
  if (Subtarget.useESPV2P2Instructions()) {
    SDValue Ops[] = {Chain, V0,  V1,     QX,
                     QY,    Ptr, Offset, lowerCmulTargetImm(DAG, DL, SAT)};
    SDValue Node =
        DAG.getMemIntrinsicNode(ISD22, DL, VTs, Ops, MVT::v16i8, MMO);
    return DAG.getMergeValues({Node.getValue(1), Node.getValue(0),
                               Node.getValue(2), Node.getValue(3),
                               Node.getValue(4)},
                              DL);
  }
  if (Subtarget.hasVendorXespv2p1()) {
    SDValue Ops[] = {Chain, V0, V1, QX, QY, Ptr, Offset};
    SDValue Node =
        DAG.getMemIntrinsicNode(ISD21, DL, VTs, Ops, MVT::v16i8, MMO);
    return DAG.getMergeValues({Node.getValue(1), Node.getValue(0),
                               Node.getValue(2), Node.getValue(3),
                               Node.getValue(4)},
                              DL);
  }
  return SDValue();
}

static SDValue lowerVcmulasLdXp(SDValue Op, SelectionDAG &DAG,
                                const RISCVSubtarget &Subtarget, unsigned ISD21,
                                unsigned ISD22) {
  SDLoc DL(Op);
  SDValue Chain = Op.getOperand(0);
  SDValue V0 = Op.getOperand(2);
  SDValue V1 = Op.getOperand(3);
  SDValue QX = Op.getOperand(4);
  SDValue QY = Op.getOperand(5);
  SDValue Ptr = Op.getOperand(6);
  SDValue Rs2 = Op.getOperand(7);
  SDValue SAT = Op.getOperand(8);
  if (Subtarget.hasVendorXespv2p1() && !Subtarget.useESPV2P2Instructions())
    diagnoseESPV21Sat(DAG, SAT);
  EVT PtrVT = Ptr.getValueType();
  auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
  MachineMemOperand *MMO = MemIntr->getMemOperand();
  SmallVector<EVT, 5> VTList = {MVT::v16i8, PtrVT, MVT::v16i8, MVT::v16i8,
                                MVT::Other};
  SDVTList VTs = DAG.getVTList(VTList);
  if (Subtarget.useESPV2P2Instructions()) {
    SDValue Ops[] = {Chain, V0,  V1,  QX,
                     QY,    Ptr, Rs2, lowerCmulTargetImm(DAG, DL, SAT)};
    SDValue Node =
        DAG.getMemIntrinsicNode(ISD22, DL, VTs, Ops, MVT::v16i8, MMO);
    return DAG.getMergeValues({Node.getValue(1), Node.getValue(0),
                               Node.getValue(2), Node.getValue(3),
                               Node.getValue(4)},
                              DL);
  }
  if (Subtarget.hasVendorXespv2p1()) {
    SDValue Ops[] = {Chain, V0, V1, QX, QY, Ptr, Rs2};
    SDValue Node =
        DAG.getMemIntrinsicNode(ISD21, DL, VTs, Ops, MVT::v16i8, MMO);
    return DAG.getMergeValues({Node.getValue(1), Node.getValue(0),
                               Node.getValue(2), Node.getValue(3),
                               Node.getValue(4)},
                              DL);
  }
  return SDValue();
}

static SDValue lowerSrsXacc(SDValue Op, SelectionDAG &DAG,
                            const RISCVSubtarget &Subtarget, unsigned ISD21,
                            unsigned ISD22) {
  SDLoc DL(Op);
  SDValue XACCHighPassthru = Op.getOperand(1);
  SDValue XACCLowPassthru = Op.getOperand(2);
  SDValue RS1 = Op.getOperand(3);
  SDValue SAT = Op.getOperand(4);
  SDValue RM = Op.getOperand(5);
  SDVTList VTs = DAG.getVTList(MVT::i32, MVT::i32, MVT::i32);
  if (Subtarget.hasVendorXespv2p1()) {
    diagnoseESPV21CmulSatRm(DAG, SAT, RM);
    SDValue Ops[] = {XACCHighPassthru, XACCLowPassthru, RS1};
    SDValue Node = DAG.getNode(ISD21, DL, VTs, Ops);
    return DAG.getMergeValues(
        {Node.getValue(0), Node.getValue(1), Node.getValue(2)}, DL);
  }
  if (Subtarget.useESPV2P2Instructions()) {
    SDValue Ops[] = {XACCHighPassthru, XACCLowPassthru, RS1,
                     lowerCmulTargetImm(DAG, DL, SAT),
                     lowerCmulTargetImm(DAG, DL, RM)};
    SDValue Node = DAG.getNode(ISD22, DL, VTs, Ops);
    return DAG.getMergeValues(
        {Node.getValue(0), Node.getValue(1), Node.getValue(2)}, DL);
  }
  return SDValue();
}

static SDValue lowerCmulLdIncp(SDValue Op, SelectionDAG &DAG,
                               const RISCVSubtarget &Subtarget, MVT QzVT,
                               unsigned ISD21, unsigned ISD22) {
  SDLoc DL(Op);
  SDValue Chain = Op.getOperand(0);
  SDValue QZ_IN = Op.getOperand(2);
  SDValue QX = Op.getOperand(3);
  SDValue QY = Op.getOperand(4);
  SDValue RS1 = Op.getOperand(5);
  SDValue SEL4 = Op.getOperand(6);
  SDValue SAT = Op.getOperand(7);
  SDValue RM = Op.getOperand(8);
  SDValue Sar = Op.getOperand(9);
  if (Subtarget.hasVendorXespv2p1())
    diagnoseESPV21CmulSatRm(DAG, SAT, RM);
  EVT PtrVT = RS1.getValueType();
  EVT MemVT = MVT::v16i8;
  auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
  MachineMemOperand *MMO = MemIntr->getMemOperand();
  if (Subtarget.useESPV2P2Instructions()) {
    SDVTList VTs = DAG.getVTList(QzVT, MVT::v16i8, PtrVT, MVT::Other);
    SDValue Ops[] = {Chain,
                     QX,
                     QY,
                     RS1,
                     lowerCmulTargetImm(DAG, DL, SAT),
                     lowerCmulTargetImm(DAG, DL, SEL4),
                     lowerCmulTargetImm(DAG, DL, RM)};
    SDValue Node = DAG.getMemIntrinsicNode(ISD22, DL, VTs, Ops, MemVT, MMO);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1),
                               Node.getValue(2), Node.getValue(3)},
                              DL);
  }
  return SDValue();
}

static SDValue lowerCmulStIncp(SDValue Op, SelectionDAG &DAG,
                               const RISCVSubtarget &Subtarget, MVT QzVT,
                               unsigned ISD21, unsigned ISD22) {
  SDLoc DL(Op);
  SDValue Chain = Op.getOperand(0);
  SDValue QZ_IN = Op.getOperand(2);
  SDValue QX = Op.getOperand(3);
  SDValue QY = Op.getOperand(4);
  SDValue QU = Op.getOperand(5);
  SDValue RS1 = Op.getOperand(6);
  SDValue SEL4 = Op.getOperand(7);
  SDValue SAT = Op.getOperand(8);
  SDValue RM = Op.getOperand(9);
  SDValue Sar = Op.getOperand(10);
  if (Subtarget.hasVendorXespv2p1())
    diagnoseESPV21CmulSatRm(DAG, SAT, RM);
  EVT PtrVT = RS1.getValueType();
  EVT MemVT = MVT::v16i8;
  auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
  MachineMemOperand *MMO = MemIntr->getMemOperand();
  if (Subtarget.useESPV2P2Instructions()) {
    SDVTList VTs = DAG.getVTList(QzVT, PtrVT, MVT::Other);
    SDValue Ops[] = {Chain,
                     QX,
                     QY,
                     QU,
                     RS1,
                     lowerCmulTargetImm(DAG, DL, SAT),
                     lowerCmulTargetImm(DAG, DL, SEL4),
                     lowerCmulTargetImm(DAG, DL, RM)};
    SDValue Node = DAG.getMemIntrinsicNode(ISD22, DL, VTs, Ops, MemVT, MMO);
    return DAG.getMergeValues(
        {Node.getValue(0), Node.getValue(1), Node.getValue(2)}, DL);
  }
  return SDValue();
}

static SDValue lowerVmulBasic(SDValue Op, SelectionDAG &DAG,
                              const RISCVSubtarget &Subtarget, MVT RetVT,
                              unsigned ISD22) {
  SDLoc DL(Op);
  SDValue QX = Op.getOperand(1);
  SDValue QY = Op.getOperand(2);
  SDValue SAT = Op.getOperand(3);
  SDValue RM = Op.getOperand(4);
  if (Subtarget.hasVendorXespv2p1())
    diagnoseESPV21SatRm(DAG, SAT, RM);
  if (Subtarget.useESPV2P2Instructions()) {
    SDVTList VTs = DAG.getVTList(RetVT);
    SDValue Ops[] = {QX, QY, lowerCmulTargetImm(DAG, DL, SAT),
                     lowerCmulTargetImm(DAG, DL, RM)};
    return DAG.getNode(ISD22, DL, VTs, Ops);
  }
  return SDValue();
}

static SDValue lowerVmulLdIncp(SDValue Op, SelectionDAG &DAG,
                               const RISCVSubtarget &Subtarget, MVT QvVT,
                               unsigned ISD22) {
  SDLoc DL(Op);
  SDValue Chain = Op.getOperand(0);
  SDValue QX = Op.getOperand(2);
  SDValue QY = Op.getOperand(3);
  SDValue RS1 = Op.getOperand(4);
  SDValue SAT = Op.getOperand(5);
  SDValue RM = Op.getOperand(6);
  if (Subtarget.hasVendorXespv2p1())
    diagnoseESPV21SatRm(DAG, SAT, RM);
  EVT PtrVT = RS1.getValueType();
  EVT MemVT = MVT::v16i8;
  auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
  MachineMemOperand *MMO = MemIntr->getMemOperand();
  if (Subtarget.useESPV2P2Instructions()) {
    SDVTList VTs = DAG.getVTList(QvVT, MemVT, PtrVT, MVT::Other);
    SDValue Ops[] = {Chain,
                     QX,
                     QY,
                     RS1,
                     lowerCmulTargetImm(DAG, DL, SAT),
                     lowerCmulTargetImm(DAG, DL, RM)};
    SDValue Node = DAG.getMemIntrinsicNode(ISD22, DL, VTs, Ops, MemVT, MMO);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1),
                               Node.getValue(2), Node.getValue(3)},
                              DL);
  }
  return SDValue();
}

static SDValue lowerVmulStIncp(SDValue Op, SelectionDAG &DAG,
                               const RISCVSubtarget &Subtarget, MVT QvVT,
                               unsigned ISD21, unsigned ISD22) {
  SDLoc DL(Op);
  SDValue Chain = Op.getOperand(0);
  SDValue QX = Op.getOperand(2);
  SDValue QY = Op.getOperand(3);
  SDValue QU = Op.getOperand(4);
  SDValue RS1 = Op.getOperand(5);
  // Op6 = qz passthrough (unused for 2.1 SDNode; tied at MI via outs).
  SDValue SAT = Op.getOperand(7);
  SDValue RM = Op.getOperand(8);
  SDValue Sar = Op.getOperand(9);
  if (Subtarget.hasVendorXespv2p1())
    diagnoseESPV21SatRm(DAG, SAT, RM);
  EVT PtrVT = RS1.getValueType();
  EVT MemVT = MVT::v16i8;
  auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
  MachineMemOperand *MMO = MemIntr->getMemOperand();
  if (Subtarget.useESPV2P2Instructions()) {
    SDVTList VTs = DAG.getVTList(QvVT, PtrVT, MVT::Other);
    SDValue Ops[] = {Chain,
                     QX,
                     QY,
                     QU,
                     RS1,
                     lowerCmulTargetImm(DAG, DL, SAT),
                     lowerCmulTargetImm(DAG, DL, RM)};
    SDValue Node = DAG.getMemIntrinsicNode(ISD22, DL, VTs, Ops, MemVT, MMO);
    return DAG.getMergeValues(
        {Node.getValue(0), Node.getValue(1), Node.getValue(2)}, DL);
  }
  // +xespv2p1: SAR state-passing SDNode (matches ESP_VMUL_*_ST_INCP Pats).
  SDVTList VTs = DAG.getVTList(QvVT, PtrVT, MVT::Other);
  SDValue Ops[] = {Chain, QX, QY, QU, RS1, Sar};
  SDValue Node = DAG.getMemIntrinsicNode(ISD21, DL, VTs, Ops, MemVT, MMO);
  return DAG.getMergeValues(
      {Node.getValue(0), Node.getValue(1), Node.getValue(2)}, DL);
}

static SDValue lowerVmulS8xS8(SDValue Op, SelectionDAG &DAG,
                              const RISCVSubtarget &Subtarget, unsigned ISD22) {
  SDLoc DL(Op);
  SDValue QX = Op.getOperand(1);
  SDValue QY = Op.getOperand(2);
  SDValue SAT = Op.getOperand(3);
  SDValue RM = Op.getOperand(4);
  if (Subtarget.hasVendorXespv2p1())
    diagnoseESPV21SatRm(DAG, SAT, RM);
  if (Subtarget.useESPV2P2Instructions()) {
    SDVTList VTs = DAG.getVTList(MVT::v8i16, MVT::v8i16);
    SDValue Ops[] = {QX, QY, lowerCmulTargetImm(DAG, DL, SAT),
                     lowerCmulTargetImm(DAG, DL, RM)};
    SDValue Node = DAG.getNode(ISD22, DL, VTs, Ops);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1)}, DL);
  }
  return SDValue();
}

static SDValue lowerVmulS16xS16(SDValue Op, SelectionDAG &DAG,
                                const RISCVSubtarget &Subtarget,
                                unsigned ISD22) {
  SDLoc DL(Op);
  SDValue QX = Op.getOperand(1);
  SDValue QY = Op.getOperand(2);
  SDValue SAT = Op.getOperand(3);
  SDValue RM = Op.getOperand(4);
  if (Subtarget.hasVendorXespv2p1())
    diagnoseESPV21SatRm(DAG, SAT, RM);
  if (Subtarget.useESPV2P2Instructions()) {
    SDVTList VTs = DAG.getVTList(MVT::v4i32, MVT::v4i32);
    SDValue Ops[] = {QX, QY, lowerCmulTargetImm(DAG, DL, SAT),
                     lowerCmulTargetImm(DAG, DL, RM)};
    SDValue Node = DAG.getNode(ISD22, DL, VTs, Ops);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1)}, DL);
  }
  return SDValue();
}

static SDValue lowerFftTargetImm(SelectionDAG &DAG, const SDLoc &DL,
                                 SDValue V) {
  if (V.getOpcode() == ISD::TargetConstant)
    return V;
  if (auto *C = dyn_cast<ConstantSDNode>(V.getNode()))
    return DAG.getTargetConstant(C->getZExtValue(), DL, MVT::i32);
  return V;
}

static void diagnoseESPV21FftSat(SelectionDAG &DAG, SDValue Sat) {
  const Function &F = DAG.getMachineFunction().getFunction();
  auto Diagnose = [&](const char *Msg) {
    F.getContext().diagnose(DiagnosticInfoUnsupported{F, Msg});
  };
  if (auto *C = dyn_cast<ConstantSDNode>(Sat.getNode())) {
    if (C->getZExtValue() != 0)
      Diagnose("fft sat must be 0 for PIE 2.1 (+xespv2p1)");
  } else {
    Diagnose("fft sat must be a constant immediate for PIE 2.1 (+xespv2p1)");
  }
}

static SDValue lowerFftR2bf(SDValue Op, SelectionDAG &DAG,
                            const RISCVSubtarget &Subtarget, unsigned ISD22) {
  SDLoc DL(Op);
  SDValue QX = Op.getOperand(1);
  SDValue QY = Op.getOperand(2);
  SDValue SEL2 = Op.getOperand(3);
  SDValue SAT = Op.getOperand(4);
  if (Subtarget.hasVendorXespv2p1())
    diagnoseESPV21FftSat(DAG, SAT);
  if (Subtarget.useESPV2P2Instructions()) {
    SDVTList VTs = DAG.getVTList(MVT::v8i16, MVT::v8i16);
    SDValue Ops[] = {QX, QY, lowerFftTargetImm(DAG, DL, SEL2),
                     lowerFftTargetImm(DAG, DL, SAT)};
    return DAG.getNode(ISD22, DL, VTs, Ops);
  }
  return SDValue();
}

static SDValue lowerFftR2bfStIncp(SDValue Op, SelectionDAG &DAG,
                                  const RISCVSubtarget &Subtarget,
                                  unsigned ISD22) {
  SDLoc DL(Op);
  SDValue Chain = Op.getOperand(0);
  SDValue QX = Op.getOperand(2);
  SDValue QY = Op.getOperand(3);
  SDValue RS1 = Op.getOperand(4);
  SDValue SEL4 = Op.getOperand(5);
  SDValue SAT = Op.getOperand(6);
  if (Subtarget.hasVendorXespv2p1())
    diagnoseESPV21FftSat(DAG, SAT);
  if (Subtarget.useESPV2P2Instructions()) {
    EVT PtrVT = RS1.getValueType();
    SDVTList VTs = DAG.getVTList(MVT::v8i16, PtrVT, MVT::Other);
    SDValue Ops[] = {Chain,
                     QX,
                     QY,
                     RS1,
                     lowerFftTargetImm(DAG, DL, SAT),
                     lowerFftTargetImm(DAG, DL, SEL4)};
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node =
        DAG.getMemIntrinsicNode(ISD22, DL, VTs, Ops, MVT::v16i8, MMO);
    return DAG.getMergeValues(
        {Node.getValue(0), Node.getValue(1), Node.getValue(2)}, DL);
  }
  return SDValue();
}

static SDValue lowerFftAmsLdIncp(SDValue Op, SelectionDAG &DAG,
                                 const RISCVSubtarget &Subtarget,
                                 unsigned ISD22) {
  SDLoc DL(Op);
  SDValue Chain = Op.getOperand(0);
  SDValue QX = Op.getOperand(2);
  SDValue QY = Op.getOperand(3);
  SDValue QW = Op.getOperand(4);
  SDValue RS1 = Op.getOperand(5);
  SDValue SEL2 = Op.getOperand(6);
  SDValue SAT = Op.getOperand(7);
  if (Subtarget.hasVendorXespv2p1())
    diagnoseESPV21FftSat(DAG, SAT);
  if (Subtarget.useESPV2P2Instructions()) {
    EVT PtrVT = RS1.getValueType();
    SmallVector<EVT, 5> VTList = {MVT::v16i8, MVT::v8i16, MVT::v8i16, PtrVT,
                                  MVT::Other};
    SDVTList VTs = DAG.getVTList(VTList);
    SDValue Ops[] = {Chain,
                     QX,
                     QY,
                     QW,
                     RS1,
                     lowerFftTargetImm(DAG, DL, SEL2),
                     lowerFftTargetImm(DAG, DL, SAT)};
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node =
        DAG.getMemIntrinsicNode(ISD22, DL, VTs, Ops, MVT::v16i8, MMO);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1),
                               Node.getValue(2), Node.getValue(3),
                               Node.getValue(4)},
                              DL);
  }
  return SDValue();
}

static SDValue lowerFftAmsLdIncpUaup(SDValue Op, SelectionDAG &DAG,
                                     const RISCVSubtarget &Subtarget,
                                     unsigned ISD22) {
  SDLoc DL(Op);
  SDValue Chain = Op.getOperand(0);
  SDValue QX = Op.getOperand(2);
  SDValue QY = Op.getOperand(3);
  SDValue QW = Op.getOperand(4);
  SDValue RS1 = Op.getOperand(5);
  SDValue SEL2 = Op.getOperand(6);
  SDValue SAT = Op.getOperand(7);
  SDValue UAStateIn = Op.getOperand(8);
  if (Subtarget.hasVendorXespv2p1())
    diagnoseESPV21FftSat(DAG, SAT);
  if (Subtarget.useESPV2P2Instructions()) {
    EVT PtrVT = RS1.getValueType();
    SmallVector<EVT, 5> VTList = {MVT::v16i8, MVT::v8i16, MVT::v8i16, PtrVT,
                                  MVT::Other};
    SDVTList VTs = DAG.getVTList(VTList);
    SDValue Ops[] = {Chain,
                     QX,
                     QY,
                     QW,
                     RS1,
                     lowerFftTargetImm(DAG, DL, SEL2),
                     lowerFftTargetImm(DAG, DL, SAT)};
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node =
        DAG.getMemIntrinsicNode(ISD22, DL, VTs, Ops, MVT::v16i8, MMO);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1),
                               Node.getValue(2), Node.getValue(3), UAStateIn,
                               Node.getValue(4)},
                              DL);
  }
  return SDValue();
}

static SDValue lowerFftAmsLdR32Decp(SDValue Op, SelectionDAG &DAG,
                                    const RISCVSubtarget &Subtarget,
                                    unsigned ISD22) {
  return lowerFftAmsLdIncp(Op, DAG, Subtarget, ISD22);
}

static SDValue lowerFftAmsStIncp(SDValue Op, SelectionDAG &DAG,
                                 const RISCVSubtarget &Subtarget,
                                 unsigned ISD22) {
  SDLoc DL(Op);
  SDValue Chain = Op.getOperand(0);
  SDValue QX = Op.getOperand(2);
  SDValue QY = Op.getOperand(3);
  SDValue QW = Op.getOperand(4);
  SDValue QU = Op.getOperand(5);
  SDValue RS1 = Op.getOperand(6);
  SDValue RS2 = Op.getOperand(7);
  SDValue SEL2 = Op.getOperand(8);
  SDValue SAT = Op.getOperand(9);
  if (Subtarget.hasVendorXespv2p1())
    diagnoseESPV21FftSat(DAG, SAT);
  if (Subtarget.useESPV2P2Instructions()) {
    EVT PtrVT = RS1.getValueType();
    SDVTList VTs = DAG.getVTList(MVT::v8i16, PtrVT, PtrVT, MVT::Other);
    SDValue Ops[] = {Chain,
                     QX,
                     QY,
                     QW,
                     QU,
                     RS1,
                     RS2,
                     lowerFftTargetImm(DAG, DL, SEL2),
                     lowerFftTargetImm(DAG, DL, SAT)};
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node =
        DAG.getMemIntrinsicNode(ISD22, DL, VTs, Ops, MVT::v16i8, MMO);
    return DAG.getMergeValues(
        {Node.getValue(0), Node.getValue(1), Node.getValue(3)}, DL);
  }
  return SDValue();
}

static SDValue lowerFftCmulLdXp(SDValue Op, SelectionDAG &DAG,
                                const RISCVSubtarget &Subtarget,
                                unsigned ISD22) {
  SDLoc DL(Op);
  SDValue Chain = Op.getOperand(0);
  SDValue QX = Op.getOperand(2);
  SDValue QY = Op.getOperand(3);
  SDValue RS1 = Op.getOperand(4);
  SDValue RS2 = Op.getOperand(5);
  SDValue SEL8 = Op.getOperand(6);
  SDValue SAT = Op.getOperand(7);
  if (Subtarget.hasVendorXespv2p1())
    diagnoseESPV21FftSat(DAG, SAT);
  if (Subtarget.useESPV2P2Instructions()) {
    EVT PtrVT = RS1.getValueType();
    SDVTList VTs = DAG.getVTList(MVT::v8i16, MVT::v16i8, PtrVT, MVT::Other);
    SDValue Ops[] = {Chain,
                     QX,
                     QY,
                     RS2,
                     RS1,
                     lowerFftTargetImm(DAG, DL, SAT),
                     lowerFftTargetImm(DAG, DL, SEL8)};
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node =
        DAG.getMemIntrinsicNode(ISD22, DL, VTs, Ops, MVT::v16i8, MMO);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1),
                               Node.getValue(2), Node.getValue(3)},
                              DL);
  }
  return SDValue();
}

static SDValue lowerFftCmulStXp(SDValue Op, SelectionDAG &DAG,
                                const RISCVSubtarget &Subtarget,
                                unsigned ISD22) {
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
  SDValue SAT = Op.getOperand(10);
  if (Subtarget.hasVendorXespv2p1())
    diagnoseESPV21FftSat(DAG, SAT);
  if (Subtarget.useESPV2P2Instructions()) {
    EVT PtrVT = RS1.getValueType();
    SDVTList VTs = DAG.getVTList(PtrVT, MVT::Other);
    SDValue Ops[] = {Chain,
                     QX,
                     QY,
                     QU,
                     RS1,
                     RS2,
                     lowerFftTargetImm(DAG, DL, SAT),
                     lowerFftTargetImm(DAG, DL, SEL8),
                     lowerFftTargetImm(DAG, DL, UPD4),
                     lowerFftTargetImm(DAG, DL, SEL4)};
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node =
        DAG.getMemIntrinsicNode(ISD22, DL, VTs, Ops, MVT::v16i8, MMO);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1)}, DL);
  }
  return SDValue();
}
} // namespace

namespace llvm {
namespace RISCV {

static SDValue LowerLDXACCIP(SDValue Op, SelectionDAG &DAG, unsigned ISDOpcode);
static SDValue LowerSTXACCIP(SDValue Op, SelectionDAG &DAG, unsigned ISDOpcode);
static SDValue LowerLDQAIP(SDValue Op, SelectionDAG &DAG, unsigned ISDOpcode);
static SDValue LowerLDQAXP(SDValue Op, SelectionDAG &DAG, unsigned ISDOpcode);
static SDValue LowerESPLdIncpM(SDValue Op, SelectionDAG &DAG,
                               unsigned ISDOpcode, MVT ResVT);
static SDValue LowerLDUASTATEIP(SDValue Op, SelectionDAG &DAG,
                                unsigned ISDOpcode);
static SDValue LowerSTUASTATEIP(SDValue Op, SelectionDAG &DAG,
                                unsigned ISDOpcode);
static SDValue LowerVMULASQACCLDIPLegacy(SDValue Op, SelectionDAG &DAG,
                                         unsigned ISDOpcode);
static SDValue lowerVmulasQaccCompute(SDValue Op, SelectionDAG &DAG,
                                      const RISCVSubtarget &Subtarget,
                                      unsigned ISD21, unsigned ISD22);
static SDValue lowerVmulasXaccCompute(SDValue Op, SelectionDAG &DAG,
                                      const RISCVSubtarget &Subtarget,
                                      unsigned ISD21, unsigned ISD22);
static SDValue LowerVMULASQACCLDIP(SDValue Op, SelectionDAG &DAG,
                                   const RISCVSubtarget &Subtarget,
                                   unsigned ISD21, unsigned ISD22);
static SDValue LowerVMULASQACCLDXP(SDValue Op, SelectionDAG &DAG,
                                   const RISCVSubtarget &Subtarget,
                                   unsigned ISD21, unsigned ISD22);
static SDValue LowerVMULASQACCSTIP(SDValue Op, SelectionDAG &DAG,
                                   const RISCVSubtarget &Subtarget,
                                   unsigned ISD21, unsigned ISD22);
static SDValue LowerVMULASQACCSTXP(SDValue Op, SelectionDAG &DAG,
                                   const RISCVSubtarget &Subtarget,
                                   unsigned ISD21, unsigned ISD22);
static SDValue LowerVMULASQACCLDBCINCP(SDValue Op, SelectionDAG &DAG,
                                       const RISCVSubtarget &Subtarget,
                                       unsigned ISD21, unsigned ISD22);
static SDValue LowerVMULASXACCLDIP(SDValue Op, SelectionDAG &DAG,
                                   const RISCVSubtarget &Subtarget,
                                   unsigned ISD21, unsigned ISD22);
static SDValue LowerVMULASXACCLDXP(SDValue Op, SelectionDAG &DAG,
                                   const RISCVSubtarget &Subtarget,
                                   unsigned ISD21, unsigned ISD22);
static SDValue LowerVMULASXACCSTIP(SDValue Op, SelectionDAG &DAG,
                                   const RISCVSubtarget &Subtarget,
                                   unsigned ISD21, unsigned ISD22);
static SDValue LowerVMULASXACCSTXP(SDValue Op, SelectionDAG &DAG,
                                   const RISCVSubtarget &Subtarget,
                                   unsigned ISD21, unsigned ISD22);
static SDValue LowerVSMULASQACCLDIP(SDValue Op, SelectionDAG &DAG,
                                    const RISCVSubtarget &Subtarget,
                                    unsigned ISD21, unsigned ISD22);
static SDValue lowerVsmulasQaccCompute(SDValue Op, SelectionDAG &DAG,
                                       const RISCVSubtarget &Subtarget,
                                       unsigned ISD21, unsigned ISD22);
bool getESPVTgtMemIntrinsic(TargetLowering::IntrinsicInfo &Info,
                            const CallBase &I, unsigned Intrinsic) {
  switch (Intrinsic) {
  default:
    return false;
  case Intrinsic::riscv_esp_vld_128_ip:
  case Intrinsic::riscv_esp_vld_128_xp:
  case Intrinsic::riscv_esp_ld_128_usar_ip:
  case Intrinsic::riscv_esp_ld_128_usar_xp: {
    // Load intrinsics: (ptr, ...) -> { <16 x i8>, ptr }
    // Pointer is the first argument (operand 0)
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(0);
    Info.memVT = MVT::v16i8;
    Info.align = Align(16);
    Info.size = 16;
    Info.flags |= MachineMemOperand::MOLoad;
    return true;
  }
  case Intrinsic::riscv_esp_ldxq_32: {
    // (ptr, qw, sel4, sel8) -> v4i32
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(0);
    Info.memVT = MVT::v4i32;
    Info.align = Align(16);
    Info.size = 16;
    Info.flags |= MachineMemOperand::MOLoad;
    return true;
  }
  case Intrinsic::riscv_esp_ld_ua_state_ip: {
    // Load intrinsic: (ua_state_passthru, ptr, offset) -> { <16 x i8>, ptr }
    // Pointer is the second argument (operand 1)
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(1);
    Info.memVT = MVT::v16i8;
    Info.align = Align(16);
    Info.size = 16;
    Info.flags |= MachineMemOperand::MOLoad;
    return true;
  }
  case Intrinsic::riscv_esp_ld_qacc_h_h_128_ip:
  case Intrinsic::riscv_esp_ld_qacc_h_l_128_ip:
  case Intrinsic::riscv_esp_ld_qacc_l_h_128_ip:
  case Intrinsic::riscv_esp_ld_qacc_l_l_128_ip: {
    // LD QACC intrinsics: (ptr, offset) -> { v16i8, ptr }
    // Pointer is the first argument (operand 0)
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(0);
    Info.memVT = MVT::v16i8;
    Info.align = Align(16);
    Info.size = 16;
    Info.flags |= MachineMemOperand::MOLoad;
    return true;
  }
  case Intrinsic::riscv_esp_ldqa_s16_128_ip:
  case Intrinsic::riscv_esp_ldqa_s16_128_xp:
  case Intrinsic::riscv_esp_ldqa_s8_128_ip:
  case Intrinsic::riscv_esp_ldqa_s8_128_xp:
  case Intrinsic::riscv_esp_ldqa_u16_128_ip:
  case Intrinsic::riscv_esp_ldqa_u16_128_xp:
  case Intrinsic::riscv_esp_ldqa_u8_128_ip:
  case Intrinsic::riscv_esp_ldqa_u8_128_xp: {
    // LDQA intrinsics: (qacc_passthru, ptr, offset) -> { ptr, v16i8, v16i8,
    // v16i8, v16i8 } Pointer is the second argument (operand 1)
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(1);
    Info.memVT = MVT::v16i8;
    Info.align = Align(16);
    Info.size = 16;
    Info.flags |= MachineMemOperand::MOLoad;
    return true;
  }
  case Intrinsic::riscv_esp_vadd_s8_ld_incp:
  case Intrinsic::riscv_esp_vadd_u8_ld_incp:
  case Intrinsic::riscv_esp_vadd_s16_ld_incp:
  case Intrinsic::riscv_esp_vadd_u16_ld_incp:
  case Intrinsic::riscv_esp_vadd_s32_ld_incp:
  case Intrinsic::riscv_esp_vadd_u32_ld_incp:
  case Intrinsic::riscv_esp_vmax_s8_ld_incp:
  case Intrinsic::riscv_esp_vmax_s16_ld_incp:
  case Intrinsic::riscv_esp_vmax_s32_ld_incp:
  case Intrinsic::riscv_esp_vmax_u8_ld_incp:
  case Intrinsic::riscv_esp_vmax_u16_ld_incp:
  case Intrinsic::riscv_esp_vmax_u32_ld_incp:
  case Intrinsic::riscv_esp_vmin_s8_ld_incp:
  case Intrinsic::riscv_esp_vmin_s16_ld_incp:
  case Intrinsic::riscv_esp_vmin_s32_ld_incp:
  case Intrinsic::riscv_esp_vmin_u8_ld_incp:
  case Intrinsic::riscv_esp_vmin_u16_ld_incp:
  case Intrinsic::riscv_esp_vmin_u32_ld_incp:
  case Intrinsic::riscv_esp_vsub_s8_ld_incp:
  case Intrinsic::riscv_esp_vsub_s16_ld_incp:
  case Intrinsic::riscv_esp_vsub_s32_ld_incp:
  case Intrinsic::riscv_esp_vsub_u8_ld_incp:
  case Intrinsic::riscv_esp_vsub_u16_ld_incp:
  case Intrinsic::riscv_esp_vsub_u32_ld_incp: {
    // LD.INCP: (qx, qy, ptr[, sat]) -> { ..., ptr }; memory at ptr
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(2);
    Info.memVT = MVT::v16i8;
    Info.align = Align(16);
    Info.size = 16;
    Info.flags |= MachineMemOperand::MOLoad;
    return true;
  }
  case Intrinsic::riscv_esp_vadd_s8_st_incp:
  case Intrinsic::riscv_esp_vadd_u8_st_incp:
  case Intrinsic::riscv_esp_vadd_s16_st_incp:
  case Intrinsic::riscv_esp_vadd_u16_st_incp:
  case Intrinsic::riscv_esp_vadd_s32_st_incp:
  case Intrinsic::riscv_esp_vadd_u32_st_incp:
  case Intrinsic::riscv_esp_vmax_s8_st_incp:
  case Intrinsic::riscv_esp_vmax_s16_st_incp:
  case Intrinsic::riscv_esp_vmax_s32_st_incp:
  case Intrinsic::riscv_esp_vmax_u8_st_incp:
  case Intrinsic::riscv_esp_vmax_u16_st_incp:
  case Intrinsic::riscv_esp_vmax_u32_st_incp:
  case Intrinsic::riscv_esp_vmin_s8_st_incp:
  case Intrinsic::riscv_esp_vmin_s16_st_incp:
  case Intrinsic::riscv_esp_vmin_s32_st_incp:
  case Intrinsic::riscv_esp_vmin_u8_st_incp:
  case Intrinsic::riscv_esp_vmin_u16_st_incp:
  case Intrinsic::riscv_esp_vmin_u32_st_incp:
  case Intrinsic::riscv_esp_vsub_s8_st_incp:
  case Intrinsic::riscv_esp_vsub_s16_st_incp:
  case Intrinsic::riscv_esp_vsub_s32_st_incp:
  case Intrinsic::riscv_esp_vsub_u8_st_incp:
  case Intrinsic::riscv_esp_vsub_u16_st_incp:
  case Intrinsic::riscv_esp_vsub_u32_st_incp: {
    // ST.INCP: (qx, qy, qu, ptr, qv[, sat]) -> { qv, ptr }
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(3);
    Info.memVT = MVT::v16i8;
    Info.align = Align(16);
    Info.size = 16;
    Info.flags |= MachineMemOperand::MOStore;
    return true;
  }
  // ESP vector multiply-accumulate broadcast load intrinsics (VMULAS QACC
  // LDBC.INCP) Parameters: (qacc_l_l_in, qacc_l_h_in, qacc_h_l_in, qacc_h_h_in,
  // qx, qy, ptr)
  case Intrinsic::riscv_esp_vmulas_s8_qacc_ldbc_incp:
  case Intrinsic::riscv_esp_vmulas_s16_qacc_ldbc_incp:
  case Intrinsic::riscv_esp_vmulas_u8_qacc_ldbc_incp:
  case Intrinsic::riscv_esp_vmulas_u16_qacc_ldbc_incp: {
    // Pointer is the seventh argument (operand 6)
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(6);
    Info.memVT = MVT::v16i8;
    Info.align = Align(16);
    Info.size = 16;
    Info.flags |= MachineMemOperand::MOLoad;
    return true;
  }
  // ESP vector multiply-accumulate load intrinsics (VMULAS QACC LD.IP)
  // Parameters: (v0, v1, v2, v3, qx, qy, ptr, offset) where ptr is the pointer
  case Intrinsic::riscv_esp_vmulas_s8_qacc_ld_ip:
  case Intrinsic::riscv_esp_vmulas_s16_qacc_ld_ip:
  case Intrinsic::riscv_esp_vmulas_u8_qacc_ld_ip:
  case Intrinsic::riscv_esp_vmulas_u16_qacc_ld_ip:
  case Intrinsic::riscv_esp_vsmulas_s8_qacc_ld_incp:
  case Intrinsic::riscv_esp_vsmulas_s16_qacc_ld_incp:
  case Intrinsic::riscv_esp_vsmulas_u8_qacc_ld_incp:
  case Intrinsic::riscv_esp_vsmulas_u16_qacc_ld_incp: {
    // Pointer is the seventh argument (operand 6)
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(6);
    Info.memVT = MVT::v16i8;
    Info.align = Align(16);
    Info.size = 16;
    Info.flags |= MachineMemOperand::MOLoad;
    return true;
  }
  // ESP vector multiply-accumulate load intrinsics (VMULAS QACC LD.XP)
  // Parameters: (v0, v1, v2, v3, qx, qy, ptr, rs2) where ptr is the pointer
  case Intrinsic::riscv_esp_vmulas_s8_qacc_ld_xp:
  case Intrinsic::riscv_esp_vmulas_s16_qacc_ld_xp:
  case Intrinsic::riscv_esp_vmulas_u8_qacc_ld_xp:
  case Intrinsic::riscv_esp_vmulas_u16_qacc_ld_xp: {
    // Pointer is the seventh argument (operand 6)
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(6);
    Info.memVT = MVT::v16i8;
    Info.align = Align(16);
    Info.size = 16;
    Info.flags |= MachineMemOperand::MOLoad;
    return true;
  }
  // ESP vector multiply-accumulate store intrinsics (VMULAS QACC ST.IP)
  // Parameters: (v0, v1, v2, v3, qu, qx, qy, ptr, offset) where ptr is the
  // pointer
  case Intrinsic::riscv_esp_vmulas_s8_qacc_st_ip:
  case Intrinsic::riscv_esp_vmulas_s16_qacc_st_ip:
  case Intrinsic::riscv_esp_vmulas_u8_qacc_st_ip:
  case Intrinsic::riscv_esp_vmulas_u16_qacc_st_ip: {
    // Pointer is the eighth argument (operand 7)
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(7);
    Info.memVT = MVT::v16i8;
    Info.align = Align(16);
    Info.size = 16;
    Info.flags |= MachineMemOperand::MOStore;
    return true;
  }
  // ESP vector multiply-accumulate store intrinsics (VMULAS QACC ST.XP)
  // Parameters: (v0, v1, v2, v3, qu, qx, qy, ptr, rs2) where ptr is the pointer
  case Intrinsic::riscv_esp_vmulas_s8_qacc_st_xp:
  case Intrinsic::riscv_esp_vmulas_s16_qacc_st_xp:
  case Intrinsic::riscv_esp_vmulas_u8_qacc_st_xp:
  case Intrinsic::riscv_esp_vmulas_u16_qacc_st_xp: {
    // Pointer is the eighth argument (operand 7)
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(7);
    Info.memVT = MVT::v16i8;
    Info.align = Align(16);
    Info.size = 16;
    Info.flags |= MachineMemOperand::MOStore;
    return true;
  }
  // ESP vector multiply-accumulate load intrinsics (VMULAS XACC LD.IP)
  // Parameters: (xacc_low_in, xacc_high_in, qx, qy, ptr, offset)
  case Intrinsic::riscv_esp_vmulas_s16_xacc_ld_ip:
  case Intrinsic::riscv_esp_vmulas_s8_xacc_ld_ip:
  case Intrinsic::riscv_esp_vmulas_u16_xacc_ld_ip:
  case Intrinsic::riscv_esp_vmulas_u8_xacc_ld_ip: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(4);
    Info.memVT = MVT::v16i8;
    Info.align = Align(16);
    Info.size = 16;
    Info.flags |= MachineMemOperand::MOLoad;
    return true;
  }
  // ESP vector multiply-accumulate load intrinsics (VMULAS XACC LD.XP)
  // Parameters: (xacc_low_in, xacc_high_in, qx, qy, ptr, rs2)
  case Intrinsic::riscv_esp_vmulas_s16_xacc_ld_xp:
  case Intrinsic::riscv_esp_vmulas_s8_xacc_ld_xp:
  case Intrinsic::riscv_esp_vmulas_u16_xacc_ld_xp:
  case Intrinsic::riscv_esp_vmulas_u8_xacc_ld_xp: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(4);
    Info.memVT = MVT::v16i8;
    Info.align = Align(16);
    Info.size = 16;
    Info.flags |= MachineMemOperand::MOLoad;
    return true;
  }
  // ESP vector multiply-accumulate store intrinsics (VMULAS XACC ST.IP)
  // Parameters: (xacc_low_in, xacc_high_in, qu, qx, qy, ptr, offset)
  case Intrinsic::riscv_esp_vmulas_s16_xacc_st_ip:
  case Intrinsic::riscv_esp_vmulas_s8_xacc_st_ip:
  case Intrinsic::riscv_esp_vmulas_u16_xacc_st_ip:
  case Intrinsic::riscv_esp_vmulas_u8_xacc_st_ip: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(5);
    Info.memVT = MVT::v16i8;
    Info.align = Align(16);
    Info.size = 16;
    Info.flags |= MachineMemOperand::MOStore;
    return true;
  }
  // ESP vector multiply-accumulate store intrinsics (VMULAS XACC ST.XP)
  // Parameters: (xacc_low_in, xacc_high_in, qu, qx, qy, ptr, rs2)
  case Intrinsic::riscv_esp_vmulas_s16_xacc_st_xp:
  case Intrinsic::riscv_esp_vmulas_s8_xacc_st_xp:
  case Intrinsic::riscv_esp_vmulas_u16_xacc_st_xp:
  case Intrinsic::riscv_esp_vmulas_u8_xacc_st_xp: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(5);
    Info.memVT = MVT::v16i8;
    Info.align = Align(16);
    Info.size = 16;
    Info.flags |= MachineMemOperand::MOStore;
    return true;
  }

  // ESP store XACC intrinsics (ST.S.XACC.IP and ST.U.XACC.IP)
  // Parameters: (xacc_low_in, xacc_high_in, ptr, offset) where ptr is the
  // pointer
  case Intrinsic::riscv_esp_st_s_xacc_ip:
  case Intrinsic::riscv_esp_st_u_xacc_ip: {
    // Pointer is the third argument (operand 2)
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(2);
    Info.memVT = MVT::i64;
    Info.align = Align(8);
    Info.size = 8;
    Info.flags |= MachineMemOperand::MOStore;
    return true;
  }
  // ESP load XACC intrinsics (LD.XACC.IP)
  case Intrinsic::riscv_esp_ld_xacc_ip: {
    // Pointer is the third argument (operand 2)
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(2);
    Info.memVT = MVT::i64;
    Info.align = Align(8);
    Info.size = 8;
    Info.flags |= MachineMemOperand::MOLoad;
    return true;
  }
  // ESP vector complex multiply-accumulate load intrinsics (VCMULAS QACC
  // LD.IP/LD.XP) Parameters: (qacc_passthru_2x128bit, qx, qy, ptr, offset/rs2,
  // sat)
  case Intrinsic::riscv_esp_vcmulas_s8_qacc_h_ld_ip:
  case Intrinsic::riscv_esp_vcmulas_s8_qacc_l_ld_ip:
  case Intrinsic::riscv_esp_vcmulas_s16_qacc_h_ld_ip:
  case Intrinsic::riscv_esp_vcmulas_s16_qacc_l_ld_ip:
  case Intrinsic::riscv_esp_vcmulas_s8_qacc_h_ld_xp:
  case Intrinsic::riscv_esp_vcmulas_s8_qacc_l_ld_xp:
  case Intrinsic::riscv_esp_vcmulas_s16_qacc_h_ld_xp:
  case Intrinsic::riscv_esp_vcmulas_s16_qacc_l_ld_xp: {
    // Fused load: ptr is arg 4 (after 2x passthru + qx + qy)
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(4);
    Info.memVT = MVT::v16i8;
    Info.align = Align(16);
    Info.size = 16;
    Info.flags |= MachineMemOperand::MOLoad;
    return true;
  }
  case Intrinsic::riscv_esp_vst_128_ip:
  case Intrinsic::riscv_esp_vst_128_xp:
  case Intrinsic::riscv_esp_st_qacc_h_h_128_ip:
  case Intrinsic::riscv_esp_st_qacc_h_l_128_ip:
  case Intrinsic::riscv_esp_st_qacc_l_h_128_ip:
  case Intrinsic::riscv_esp_st_qacc_l_l_128_ip:
  case Intrinsic::riscv_esp_st_ua_state_ip: {
    // Store intrinsics: (ua_state_or_vec, ptr, ...) -> ptr
    // Pointer is the second argument (operand 1)
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(1);
    Info.memVT = MVT::v16i8;
    Info.align = Align(16);
    Info.size = 16;
    Info.flags |= MachineMemOperand::MOStore;
    return true;
  }
  // ESP complex multiply fused load intrinsics (CMUL LD.INCP)
  // Parameters: (qz_in, qx, qy, ptr, offset, SAR) where ptr is the pointer
  case Intrinsic::riscv_esp_cmul_s8_ld_incp:
  case Intrinsic::riscv_esp_cmul_s16_ld_incp:
  case Intrinsic::riscv_esp_cmul_u8_ld_incp_m:
  case Intrinsic::riscv_esp_cmul_u16_ld_incp_m: {
    // Fused load intrinsics: (qz_in, qx, qy, ptr, offset, SAR) -> {qz, qu, ptr}
    // Pointer is the fourth argument (operand 3)
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(3);
    Info.memVT = MVT::v16i8;
    Info.align = Align(16);
    Info.size = 16;
    Info.flags |= MachineMemOperand::MOLoad;
    return true;
  }
  // ESP vector multiply fused load/store intrinsics (VMUL LD/ST.INCP)
  case Intrinsic::riscv_esp_vmul_s8_ld_incp:
  case Intrinsic::riscv_esp_vmul_u8_ld_incp:
  case Intrinsic::riscv_esp_vmul_s16_ld_incp:
  case Intrinsic::riscv_esp_vmul_u16_ld_incp: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(2);
    Info.memVT = MVT::v16i8;
    Info.align = Align(16);
    Info.size = 16;
    Info.flags |= MachineMemOperand::MOLoad;
    return true;
  }
  case Intrinsic::riscv_esp_vmul_s8_st_incp:
  case Intrinsic::riscv_esp_vmul_u8_st_incp:
  case Intrinsic::riscv_esp_vmul_s16_st_incp:
  case Intrinsic::riscv_esp_vmul_u16_st_incp: {
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(3);
    Info.memVT = MVT::v16i8;
    Info.align = Align(16);
    Info.size = 16;
    Info.flags |= MachineMemOperand::MOStore;
    return true;
  }
  case Intrinsic::riscv_esp_fft_ams_s16_ld_incp:
  case Intrinsic::riscv_esp_fft_ams_s16_ld_incp_uaup:
  case Intrinsic::riscv_esp_fft_ams_s16_ld_r32_decp: {
    // Pointer is the fourth argument (operand 3)
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(3);
    Info.memVT = MVT::v16i8;
    Info.align = Align(16);
    Info.size = 16;
    Info.flags |= MachineMemOperand::MOLoad;
    return true;
  }
  // FFT.R2BF.S16.ST.INCP: (qx, qy, ptr, sel4)
  case Intrinsic::riscv_esp_fft_r2bf_s16_st_incp: {
    // Pointer is the third argument (operand 2)
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(2);
    Info.memVT = MVT::v16i8;
    Info.align = Align(16);
    Info.size = 16;
    Info.flags |= MachineMemOperand::MOStore;
    return true;
  }
  // FFT.AMS.S16.ST.INCP: (qx, qy, qw, qu, ptr1, ptr2, sel2, upd4)
  case Intrinsic::riscv_esp_fft_ams_s16_st_incp: {
    // Primary pointer is the fifth argument (operand 4)
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(4);
    Info.memVT = MVT::v16i8;
    Info.align = Align(16);
    Info.size = 16;
    Info.flags |= MachineMemOperand::MOStore;
    return true;
  }
  // FFT.CMUL.S16.LD.XP: (qx, qy, ptr1, ptr2, sel8, upd4)
  case Intrinsic::riscv_esp_fft_cmul_s16_ld_xp: {
    // Primary pointer is the third argument (operand 2)
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(2);
    Info.memVT = MVT::v16i8;
    Info.align = Align(16);
    Info.size = 16;
    Info.flags |= MachineMemOperand::MOLoad;
    return true;
  }
  // FFT.CMUL.S16.ST.XP: (qx, qy, qu, ptr1, ptr2, sel8, upd4, sel4, sar)
  case Intrinsic::riscv_esp_fft_cmul_s16_st_xp: {
    // Primary pointer is the fourth argument (operand 3)
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(3);
    Info.memVT = MVT::v16i8;
    Info.align = Align(16);
    Info.size = 16;
    Info.flags |= MachineMemOperand::MOStore;
    return true;
  }
  // ESP shift right concatenated fused load intrinsics (SRC.Q LD.IP/XP)
  // Parameters: (sar_bytes, qy, qw, ptr, offset) where ptr is the pointer
  case Intrinsic::riscv_esp_src_q_ld_ip:
  case Intrinsic::riscv_esp_src_q_ld_xp: {
    // Fused load intrinsics: (sar_bytes, qy, qw, ptr, offset) -> {qw_out,
    // qu_out, ptr} Pointer is the fourth argument (operand 3)
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(3);
    Info.memVT = MVT::v16i8;
    Info.align = Align(16);
    Info.size = 16;
    Info.flags |= MachineMemOperand::MOLoad;
    return true;
  }
  // ESP shift right concatenated fused store intrinsic (SRCQ.128.ST.INCP)
  // Parameters: (sar_bytes, qy, qw, ptr) where ptr is the pointer
  case Intrinsic::riscv_esp_srcq_128_st_incp: {
    // Fused store intrinsic: (sar_bytes, qy, qw, ptr) -> ptr
    // Pointer is the fourth argument (operand 3)
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(3);
    Info.memVT = MVT::v16i8;
    Info.align = Align(16);
    Info.size = 16;
    Info.flags |= MachineMemOperand::MOStore;
    return true;
  }
  // FFT.VST.R32.DECP: (qu, ptr, sel2)
  case Intrinsic::riscv_esp_fft_vst_r32_decp_m: {
    // Pointer is the second argument (operand 1)
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(1);
    Info.memVT = MVT::v16i8;
    Info.align = Align(16);
    Info.size = 16;
    Info.flags |= MachineMemOperand::MOStore;
    return true;
  }
  // ESP complex multiply fused store intrinsics (CMUL ST.INCP)
  // Parameters: (qz_in, qx, qy, qu, ptr, offset, SAR) where ptr is the pointer
  case Intrinsic::riscv_esp_cmul_s8_st_incp:
  case Intrinsic::riscv_esp_cmul_s16_st_incp:
  case Intrinsic::riscv_esp_cmul_u8_st_incp_m:
  case Intrinsic::riscv_esp_cmul_u16_st_incp_m: {
    // Fused store intrinsics: (qz_in, qx, qy, qu, ptr, offset, SAR) -> {qz,
    // ptr} Pointer is the fifth argument (operand 4)
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(4);
    Info.memVT = MVT::v16i8;
    Info.align = Align(16);
    Info.size = 16;
    Info.flags |= MachineMemOperand::MOStore;
    return true;
  }
  case Intrinsic::riscv_esp_vld_h_64_ip:
  case Intrinsic::riscv_esp_vld_h_64_xp:
  case Intrinsic::riscv_esp_vld_l_64_ip:
  case Intrinsic::riscv_esp_vld_l_64_xp: {
    // Load intrinsics: (ptr, ...) -> { <8 x i8>, ptr }
    Info.opc = ISD::INTRINSIC_W_CHAIN;
    Info.ptrVal = I.getArgOperand(0);
    Info.memVT = MVT::v8i8;
    Info.align = Align(8);
    Info.size = 8;
    Info.flags |= MachineMemOperand::MOLoad;
    return true;
  }
  case Intrinsic::riscv_esp_vst_h_64_ip:
  case Intrinsic::riscv_esp_vst_h_64_xp:
  case Intrinsic::riscv_esp_vst_l_64_ip:
  case Intrinsic::riscv_esp_vst_l_64_xp: {
    // Store intrinsics: (<8 x i8>, ptr, ...) -> ptr
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
    // Load broadcast intrinsics: (ptr, ...) -> { <16 x i8>, ptr }
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
    // Load broadcast intrinsics: (ptr, ...) -> { <8 x i16>, ptr }
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
    // Load broadcast intrinsics: (ptr, ...) -> { <4 x i32>, ptr }
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
    // Load extend intrinsics: (ptr, ...) -> { <8 x i16>, <8 x i16>, ptr }
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
    // Load extend intrinsics: (ptr, ...) -> { <4 x i32>, <4 x i32>, ptr }
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
  if (!Subtarget.hasESPVTargetLowering())
    return SDValue();

  unsigned IntNo = Op.getConstantOperandVal(1);
  SDLoc DL(Op);

  switch (IntNo) {
  case Intrinsic::riscv_esp_vld_128_ip: {
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
  case Intrinsic::riscv_esp_vld_128_xp: {
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
  case Intrinsic::riscv_esp_vst_128_ip: {
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
  case Intrinsic::riscv_esp_ldxq_32: {
    // (chain, int_id, ptr, qw, sel4, sel8) -> (v4i32 qu, chain)
    SDValue Chain = Op.getOperand(0);
    SDValue Ptr = Op.getOperand(2);
    SDValue Qw = Op.getOperand(3);
    SDValue Sel4 = Op.getOperand(4);
    SDValue Sel8 = Op.getOperand(5);
    EVT MemVT = MVT::v4i32;
    SDVTList VTs = DAG.getVTList(MemVT, MVT::Other);
    SDValue Ops[] = {Chain, Ptr, Qw, Sel4, Sel8};
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_LDXQ_32_M, DL, VTs,
                                           Ops, MemVT, MMO);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1)}, DL);
  }
  // LD/ST XACC IP
  case Intrinsic::riscv_esp_ld_xacc_ip:
    return LowerLDXACCIP(Op, DAG, RISCVISD::ESP_LD_XACC_IP_M);
  case Intrinsic::riscv_esp_st_s_xacc_ip:
    return LowerSTXACCIP(Op, DAG, RISCVISD::ESP_ST_S_XACC_IP_M);
  case Intrinsic::riscv_esp_st_u_xacc_ip:
    return LowerSTXACCIP(Op, DAG, RISCVISD::ESP_ST_U_XACC_IP_M);
  // LD/ST UA_STATE IP
  case Intrinsic::riscv_esp_ld_ua_state_ip:
    return LowerLDUASTATEIP(Op, DAG, RISCVISD::ESP_LD_UA_STATE_IP_M);
  case Intrinsic::riscv_esp_st_ua_state_ip:
    return LowerSTUASTATEIP(Op, DAG, RISCVISD::ESP_ST_UA_STATE_IP_M);
  case Intrinsic::riscv_esp_ld_128_usar_ip: {
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
  case Intrinsic::riscv_esp_ld_128_usar_xp: {
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
  case Intrinsic::riscv_esp_vst_128_xp: {
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
  case Intrinsic::riscv_esp_vld_h_64_ip: {

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
  case Intrinsic::riscv_esp_vld_h_64_xp: {

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
  case Intrinsic::riscv_esp_vld_l_64_ip: {

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
  case Intrinsic::riscv_esp_vld_l_64_xp: {

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
  case Intrinsic::riscv_esp_vst_h_64_ip: {
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
    // Note: This intrinsic always arrives as MemIntrinsicSDNode because
    //       getTgtMemIntrinsic returns true for it.
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_VST_H_64_IP_M, DL, VTs,
                                           Ops, VecVT, MMO);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1)}, DL);
  }
  case Intrinsic::riscv_esp_vst_h_64_xp: {
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
    // Note: This intrinsic always arrives as MemIntrinsicSDNode because
    //       getTgtMemIntrinsic returns true for it.
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_VST_H_64_XP_M, DL, VTs,
                                           Ops, VecVT, MMO);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1)}, DL);
  }
  case Intrinsic::riscv_esp_vst_l_64_ip: {
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
    // Note: This intrinsic always arrives as MemIntrinsicSDNode because
    //       getTgtMemIntrinsic returns true for it.
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_VST_L_64_IP_M, DL, VTs,
                                           Ops, VecVT, MMO);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1)}, DL);
  }
  case Intrinsic::riscv_esp_vst_l_64_xp: {
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
    // Note: This intrinsic always arrives as MemIntrinsicSDNode because
    //       getTgtMemIntrinsic returns true for it.
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
    // Note: This intrinsic always arrives as MemIntrinsicSDNode because
    //       getTgtMemIntrinsic returns true for it.
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
    // Note: This intrinsic always arrives as MemIntrinsicSDNode because
    //       getTgtMemIntrinsic returns true for it.
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
    // Note: This intrinsic always arrives as MemIntrinsicSDNode because
    //       getTgtMemIntrinsic returns true for it.
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
    // Note: This intrinsic always arrives as MemIntrinsicSDNode because
    //       getTgtMemIntrinsic returns true for it.
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
    // Note: This intrinsic always arrives as MemIntrinsicSDNode because
    //       getTgtMemIntrinsic returns true for it.
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
    // Note: This intrinsic always arrives as MemIntrinsicSDNode because
    //       getTgtMemIntrinsic returns true for it.
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
    // Note: This intrinsic always arrives as MemIntrinsicSDNode because
    //       getTgtMemIntrinsic returns true for it.
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
    // Note: This intrinsic always arrives as MemIntrinsicSDNode because
    //       getTgtMemIntrinsic returns true for it.
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_VLDEXT_U16_XP_M, DL,
                                           VTs, Ops, MemVT, MMO);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1),
                               Node.getValue(2), Node.getValue(3)},
                              DL);
  }
  case Intrinsic::riscv_esp_cmul_s16_ld_incp:
    return lowerCmulLdIncp(Op, DAG, Subtarget, MVT::v8i16,
                           RISCVISD::ESP_CMUL_S16_LD_INCP_M,
                           RISCVISD::ESP_CMUL_S16_LD_INCP_PIE22_M);
  case Intrinsic::riscv_esp_cmul_s8_ld_incp:
    return lowerCmulLdIncp(Op, DAG, Subtarget, MVT::v16i8,
                           RISCVISD::ESP_CMUL_S8_LD_INCP_M,
                           RISCVISD::ESP_CMUL_S8_LD_INCP_PIE22_M);
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
  case Intrinsic::riscv_esp_cmul_s16_st_incp:
    return lowerCmulStIncp(Op, DAG, Subtarget, MVT::v8i16,
                           RISCVISD::ESP_CMUL_S16_ST_INCP_M,
                           RISCVISD::ESP_CMUL_S16_ST_INCP_PIE22_M);
  case Intrinsic::riscv_esp_cmul_s8_st_incp:
    return lowerCmulStIncp(Op, DAG, Subtarget, MVT::v16i8,
                           RISCVISD::ESP_CMUL_S8_ST_INCP_M,
                           RISCVISD::ESP_CMUL_S8_ST_INCP_PIE22_M);
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
  case Intrinsic::riscv_esp_ld_qacc_h_h_128_ip: {
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
  case Intrinsic::riscv_esp_ld_qacc_h_l_128_ip: {
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
  case Intrinsic::riscv_esp_ld_qacc_l_h_128_ip: {
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
  case Intrinsic::riscv_esp_ld_qacc_l_l_128_ip: {
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
  case Intrinsic::riscv_esp_ldqa_s16_128_ip:
    return LowerLDQAIP(Op, DAG, RISCVISD::ESP_LDQA_S16_128_IP_M);
  case Intrinsic::riscv_esp_ldqa_s8_128_ip:
    return LowerLDQAIP(Op, DAG, RISCVISD::ESP_LDQA_S8_128_IP_M);
  case Intrinsic::riscv_esp_ldqa_u16_128_ip:
    return LowerLDQAIP(Op, DAG, RISCVISD::ESP_LDQA_U16_128_IP_M);
  case Intrinsic::riscv_esp_ldqa_u8_128_ip:
    return LowerLDQAIP(Op, DAG, RISCVISD::ESP_LDQA_U8_128_IP_M);
  // LDQA XP
  case Intrinsic::riscv_esp_ldqa_s16_128_xp:
    return LowerLDQAXP(Op, DAG, RISCVISD::ESP_LDQA_S16_128_XP_M);
  case Intrinsic::riscv_esp_ldqa_s8_128_xp:
    return LowerLDQAXP(Op, DAG, RISCVISD::ESP_LDQA_S8_128_XP_M);
  case Intrinsic::riscv_esp_ldqa_u16_128_xp:
    return LowerLDQAXP(Op, DAG, RISCVISD::ESP_LDQA_U16_128_XP_M);
  case Intrinsic::riscv_esp_ldqa_u8_128_xp:
    return LowerLDQAXP(Op, DAG, RISCVISD::ESP_LDQA_U8_128_XP_M);
  case Intrinsic::riscv_esp_st_qacc_h_h_128_ip: {
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
  case Intrinsic::riscv_esp_st_qacc_h_l_128_ip: {
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
  case Intrinsic::riscv_esp_st_qacc_l_h_128_ip: {
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
  case Intrinsic::riscv_esp_st_qacc_l_l_128_ip: {
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
  case Intrinsic::riscv_esp_vadd_s8_ld_incp: {
    if (SDValue V =
            lowerVaddVsubLdIncpSat(Op, DAG, Subtarget, MVT::v16i8,
                                   RISCVISD::ESP_VADD_S8_LD_INCP_PIE22_M))
      return V;
    return LowerESPLdIncpM(Op, DAG, RISCVISD::ESP_VADD_S8_LD_INCP_M,
                           MVT::v16i8);
  }
  case Intrinsic::riscv_esp_vadd_s16_ld_incp: {
    if (SDValue V =
            lowerVaddVsubLdIncpSat(Op, DAG, Subtarget, MVT::v8i16,
                                   RISCVISD::ESP_VADD_S16_LD_INCP_PIE22_M))
      return V;
    return LowerESPLdIncpM(Op, DAG, RISCVISD::ESP_VADD_S16_LD_INCP_M,
                           MVT::v8i16);
  }
  case Intrinsic::riscv_esp_vadd_s32_ld_incp: {
    if (SDValue V =
            lowerVaddVsubLdIncpSat(Op, DAG, Subtarget, MVT::v4i32,
                                   RISCVISD::ESP_VADD_S32_LD_INCP_PIE22_M))
      return V;
    return LowerESPLdIncpM(Op, DAG, RISCVISD::ESP_VADD_S32_LD_INCP_M,
                           MVT::v4i32);
  }
  case Intrinsic::riscv_esp_vadd_u8_ld_incp: {
    if (SDValue V =
            lowerVaddVsubLdIncpSat(Op, DAG, Subtarget, MVT::v16i8,
                                   RISCVISD::ESP_VADD_U8_LD_INCP_PIE22_M))
      return V;
    return LowerESPLdIncpM(Op, DAG, RISCVISD::ESP_VADD_U8_LD_INCP_M,
                           MVT::v16i8);
  }
  case Intrinsic::riscv_esp_vadd_u16_ld_incp: {
    if (SDValue V =
            lowerVaddVsubLdIncpSat(Op, DAG, Subtarget, MVT::v8i16,
                                   RISCVISD::ESP_VADD_U16_LD_INCP_PIE22_M))
      return V;
    return LowerESPLdIncpM(Op, DAG, RISCVISD::ESP_VADD_U16_LD_INCP_M,
                           MVT::v8i16);
  }
  case Intrinsic::riscv_esp_vadd_u32_ld_incp: {
    if (SDValue V =
            lowerVaddVsubLdIncpSat(Op, DAG, Subtarget, MVT::v4i32,
                                   RISCVISD::ESP_VADD_U32_LD_INCP_PIE22_M))
      return V;
    return LowerESPLdIncpM(Op, DAG, RISCVISD::ESP_VADD_U32_LD_INCP_M,
                           MVT::v4i32);
  }
  case Intrinsic::riscv_esp_vsub_s8_ld_incp: {
    if (SDValue V =
            lowerVaddVsubLdIncpSat(Op, DAG, Subtarget, MVT::v16i8,
                                   RISCVISD::ESP_VSUB_S8_LD_INCP_PIE22_M))
      return V;
    return LowerESPLdIncpM(Op, DAG, RISCVISD::ESP_VSUB_S8_LD_INCP_M,
                           MVT::v16i8);
  }
  case Intrinsic::riscv_esp_vsub_s16_ld_incp: {
    if (SDValue V =
            lowerVaddVsubLdIncpSat(Op, DAG, Subtarget, MVT::v8i16,
                                   RISCVISD::ESP_VSUB_S16_LD_INCP_PIE22_M))
      return V;
    return LowerESPLdIncpM(Op, DAG, RISCVISD::ESP_VSUB_S16_LD_INCP_M,
                           MVT::v8i16);
  }
  case Intrinsic::riscv_esp_vsub_s32_ld_incp: {
    if (SDValue V =
            lowerVaddVsubLdIncpSat(Op, DAG, Subtarget, MVT::v4i32,
                                   RISCVISD::ESP_VSUB_S32_LD_INCP_PIE22_M))
      return V;
    return LowerESPLdIncpM(Op, DAG, RISCVISD::ESP_VSUB_S32_LD_INCP_M,
                           MVT::v4i32);
  }
  case Intrinsic::riscv_esp_vsub_u8_ld_incp: {
    if (SDValue V =
            lowerVaddVsubLdIncpSat(Op, DAG, Subtarget, MVT::v16i8,
                                   RISCVISD::ESP_VSUB_U8_LD_INCP_PIE22_M))
      return V;
    return LowerESPLdIncpM(Op, DAG, RISCVISD::ESP_VSUB_U8_LD_INCP_M,
                           MVT::v16i8);
  }
  case Intrinsic::riscv_esp_vsub_u16_ld_incp: {
    if (SDValue V =
            lowerVaddVsubLdIncpSat(Op, DAG, Subtarget, MVT::v8i16,
                                   RISCVISD::ESP_VSUB_U16_LD_INCP_PIE22_M))
      return V;
    return LowerESPLdIncpM(Op, DAG, RISCVISD::ESP_VSUB_U16_LD_INCP_M,
                           MVT::v8i16);
  }
  case Intrinsic::riscv_esp_vsub_u32_ld_incp: {
    if (SDValue V =
            lowerVaddVsubLdIncpSat(Op, DAG, Subtarget, MVT::v4i32,
                                   RISCVISD::ESP_VSUB_U32_LD_INCP_PIE22_M))
      return V;
    return LowerESPLdIncpM(Op, DAG, RISCVISD::ESP_VSUB_U32_LD_INCP_M,
                           MVT::v4i32);
  }
#define LOWER_ESPV_VADD_ST_INCP(OPC21, OPC22, INTR, VT)                        \
  case Intrinsic::INTR: {                                                      \
    if (SDValue V =                                                            \
            lowerVaddVsubStIncpSat(Op, DAG, Subtarget, VT, RISCVISD::OPC22))   \
      return V;                                                                \
    SDLoc DL(Op);                                                              \
    SDValue Chain = Op.getOperand(0);                                          \
    SDValue QX = Op.getOperand(2);                                             \
    SDValue QY = Op.getOperand(3);                                             \
    SDValue QU = Op.getOperand(4);                                             \
    SDValue RS1 = Op.getOperand(5);                                            \
    EVT PtrVT = RS1.getValueType();                                            \
    SDVTList VTs = DAG.getVTList(VT, PtrVT, MVT::Other);                       \
    SDValue Ops[] = {Chain, QX, QY, QU, RS1};                                  \
    EVT MemVT = MVT::v16i8;                                                    \
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());                    \
    MachineMemOperand *MMO = MemIntr->getMemOperand();                         \
    SDValue Node =                                                             \
        DAG.getMemIntrinsicNode(RISCVISD::OPC21, DL, VTs, Ops, MemVT, MMO);    \
    return DAG.getMergeValues(                                                 \
        {Node.getValue(0), Node.getValue(1), Node.getValue(2)}, DL);           \
  }
    LOWER_ESPV_VADD_ST_INCP(ESP_VADD_S8_ST_INCP_M, ESP_VADD_S8_ST_INCP_PIE22_M,
                            riscv_esp_vadd_s8_st_incp, MVT::v16i8)
    LOWER_ESPV_VADD_ST_INCP(ESP_VADD_S16_ST_INCP_M,
                            ESP_VADD_S16_ST_INCP_PIE22_M,
                            riscv_esp_vadd_s16_st_incp, MVT::v8i16)
    LOWER_ESPV_VADD_ST_INCP(ESP_VADD_S32_ST_INCP_M,
                            ESP_VADD_S32_ST_INCP_PIE22_M,
                            riscv_esp_vadd_s32_st_incp, MVT::v4i32)
    LOWER_ESPV_VADD_ST_INCP(ESP_VADD_U8_ST_INCP_M, ESP_VADD_U8_ST_INCP_PIE22_M,
                            riscv_esp_vadd_u8_st_incp, MVT::v16i8)
    LOWER_ESPV_VADD_ST_INCP(ESP_VADD_U16_ST_INCP_M,
                            ESP_VADD_U16_ST_INCP_PIE22_M,
                            riscv_esp_vadd_u16_st_incp, MVT::v8i16)
    LOWER_ESPV_VADD_ST_INCP(ESP_VADD_U32_ST_INCP_M,
                            ESP_VADD_U32_ST_INCP_PIE22_M,
                            riscv_esp_vadd_u32_st_incp, MVT::v4i32)
    LOWER_ESPV_VADD_ST_INCP(ESP_VSUB_S8_ST_INCP_M, ESP_VSUB_S8_ST_INCP_PIE22_M,
                            riscv_esp_vsub_s8_st_incp, MVT::v16i8)
    LOWER_ESPV_VADD_ST_INCP(ESP_VSUB_S16_ST_INCP_M,
                            ESP_VSUB_S16_ST_INCP_PIE22_M,
                            riscv_esp_vsub_s16_st_incp, MVT::v8i16)
    LOWER_ESPV_VADD_ST_INCP(ESP_VSUB_S32_ST_INCP_M,
                            ESP_VSUB_S32_ST_INCP_PIE22_M,
                            riscv_esp_vsub_s32_st_incp, MVT::v4i32)
    LOWER_ESPV_VADD_ST_INCP(ESP_VSUB_U8_ST_INCP_M, ESP_VSUB_U8_ST_INCP_PIE22_M,
                            riscv_esp_vsub_u8_st_incp, MVT::v16i8)
    LOWER_ESPV_VADD_ST_INCP(ESP_VSUB_U16_ST_INCP_M,
                            ESP_VSUB_U16_ST_INCP_PIE22_M,
                            riscv_esp_vsub_u16_st_incp, MVT::v8i16)
    LOWER_ESPV_VADD_ST_INCP(ESP_VSUB_U32_ST_INCP_M,
                            ESP_VSUB_U32_ST_INCP_PIE22_M,
                            riscv_esp_vsub_u32_st_incp, MVT::v4i32)
#undef LOWER_ESPV_VADD_ST_INCP
  case Intrinsic::riscv_esp_vmax_s8_ld_incp:
    return LowerESPLdIncpM(Op, DAG, RISCVISD::ESP_VMAX_S8_LD_INCP_M,
                           MVT::v16i8);
  case Intrinsic::riscv_esp_vmax_s16_ld_incp:
    return LowerESPLdIncpM(Op, DAG, RISCVISD::ESP_VMAX_S16_LD_INCP_M,
                           MVT::v8i16);
  case Intrinsic::riscv_esp_vmax_s32_ld_incp:
    return LowerESPLdIncpM(Op, DAG, RISCVISD::ESP_VMAX_S32_LD_INCP_M,
                           MVT::v4i32);
  case Intrinsic::riscv_esp_vmax_u8_ld_incp:
    return LowerESPLdIncpM(Op, DAG, RISCVISD::ESP_VMAX_U8_LD_INCP_M,
                           MVT::v16i8);
  case Intrinsic::riscv_esp_vmax_u16_ld_incp:
    return LowerESPLdIncpM(Op, DAG, RISCVISD::ESP_VMAX_U16_LD_INCP_M,
                           MVT::v8i16);
  case Intrinsic::riscv_esp_vmax_u32_ld_incp:
    return LowerESPLdIncpM(Op, DAG, RISCVISD::ESP_VMAX_U32_LD_INCP_M,
                           MVT::v4i32);
  case Intrinsic::riscv_esp_vmin_s8_ld_incp:
    return LowerESPLdIncpM(Op, DAG, RISCVISD::ESP_VMIN_S8_LD_INCP_M,
                           MVT::v16i8);
  case Intrinsic::riscv_esp_vmin_s16_ld_incp:
    return LowerESPLdIncpM(Op, DAG, RISCVISD::ESP_VMIN_S16_LD_INCP_M,
                           MVT::v8i16);
  case Intrinsic::riscv_esp_vmin_s32_ld_incp:
    return LowerESPLdIncpM(Op, DAG, RISCVISD::ESP_VMIN_S32_LD_INCP_M,
                           MVT::v4i32);
  case Intrinsic::riscv_esp_vmin_u8_ld_incp:
    return LowerESPLdIncpM(Op, DAG, RISCVISD::ESP_VMIN_U8_LD_INCP_M,
                           MVT::v16i8);
  case Intrinsic::riscv_esp_vmin_u16_ld_incp:
    return LowerESPLdIncpM(Op, DAG, RISCVISD::ESP_VMIN_U16_LD_INCP_M,
                           MVT::v8i16);
  case Intrinsic::riscv_esp_vmin_u32_ld_incp:
    return LowerESPLdIncpM(Op, DAG, RISCVISD::ESP_VMIN_U32_LD_INCP_M,
                           MVT::v4i32);
    // VMAX/VMIN ST.INCP lowering (vadd/vsub ST covered above)
#define LOWER_ESPV_ST_INCP_M_V16(OPC, INTR)                                    \
  case Intrinsic::INTR: {                                                      \
    SDLoc DL(Op);                                                              \
    SDValue Chain = Op.getOperand(0);                                          \
    SDValue QX = Op.getOperand(2);                                             \
    SDValue QY = Op.getOperand(3);                                             \
    SDValue QU = Op.getOperand(4);                                             \
    SDValue RS1 = Op.getOperand(5);                                            \
    EVT PtrVT = RS1.getValueType();                                            \
    SDVTList VTs = DAG.getVTList(MVT::v16i8, PtrVT, MVT::Other);               \
    SDValue Ops[] = {Chain, QX, QY, QU, RS1};                                  \
    EVT MemVT = MVT::v16i8;                                                    \
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());                    \
    MachineMemOperand *MMO = MemIntr->getMemOperand();                         \
    SDValue Node =                                                             \
        DAG.getMemIntrinsicNode(RISCVISD::OPC, DL, VTs, Ops, MemVT, MMO);      \
    return DAG.getMergeValues(                                                 \
        {Node.getValue(0), Node.getValue(1), Node.getValue(2)}, DL);           \
  }
#define LOWER_ESPV_ST_INCP_M_V8I16(OPC, INTR)                                  \
  case Intrinsic::INTR: {                                                      \
    SDLoc DL(Op);                                                              \
    SDValue Chain = Op.getOperand(0);                                          \
    SDValue QX = Op.getOperand(2);                                             \
    SDValue QY = Op.getOperand(3);                                             \
    SDValue QU = Op.getOperand(4);                                             \
    SDValue RS1 = Op.getOperand(5);                                            \
    EVT PtrVT = RS1.getValueType();                                            \
    SDVTList VTs = DAG.getVTList(MVT::v8i16, PtrVT, MVT::Other);               \
    SDValue Ops[] = {Chain, QX, QY, QU, RS1};                                  \
    EVT MemVT = MVT::v16i8;                                                    \
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());                    \
    MachineMemOperand *MMO = MemIntr->getMemOperand();                         \
    SDValue Node =                                                             \
        DAG.getMemIntrinsicNode(RISCVISD::OPC, DL, VTs, Ops, MemVT, MMO);      \
    return DAG.getMergeValues(                                                 \
        {Node.getValue(0), Node.getValue(1), Node.getValue(2)}, DL);           \
  }
#define LOWER_ESPV_ST_INCP_M_V4I32(OPC, INTR)                                  \
  case Intrinsic::INTR: {                                                      \
    SDLoc DL(Op);                                                              \
    SDValue Chain = Op.getOperand(0);                                          \
    SDValue QX = Op.getOperand(2);                                             \
    SDValue QY = Op.getOperand(3);                                             \
    SDValue QU = Op.getOperand(4);                                             \
    SDValue RS1 = Op.getOperand(5);                                            \
    EVT PtrVT = RS1.getValueType();                                            \
    SDVTList VTs = DAG.getVTList(MVT::v4i32, PtrVT, MVT::Other);               \
    SDValue Ops[] = {Chain, QX, QY, QU, RS1};                                  \
    EVT MemVT = MVT::v16i8;                                                    \
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());                    \
    MachineMemOperand *MMO = MemIntr->getMemOperand();                         \
    SDValue Node =                                                             \
        DAG.getMemIntrinsicNode(RISCVISD::OPC, DL, VTs, Ops, MemVT, MMO);      \
    return DAG.getMergeValues(                                                 \
        {Node.getValue(0), Node.getValue(1), Node.getValue(2)}, DL);           \
  }
    LOWER_ESPV_ST_INCP_M_V16(ESP_VMAX_S8_ST_INCP_M, riscv_esp_vmax_s8_st_incp)
    LOWER_ESPV_ST_INCP_M_V8I16(ESP_VMAX_S16_ST_INCP_M,
                               riscv_esp_vmax_s16_st_incp)
    LOWER_ESPV_ST_INCP_M_V4I32(ESP_VMAX_S32_ST_INCP_M,
                               riscv_esp_vmax_s32_st_incp)
    LOWER_ESPV_ST_INCP_M_V16(ESP_VMAX_U8_ST_INCP_M, riscv_esp_vmax_u8_st_incp)
    LOWER_ESPV_ST_INCP_M_V8I16(ESP_VMAX_U16_ST_INCP_M,
                               riscv_esp_vmax_u16_st_incp)
    LOWER_ESPV_ST_INCP_M_V4I32(ESP_VMAX_U32_ST_INCP_M,
                               riscv_esp_vmax_u32_st_incp)
    LOWER_ESPV_ST_INCP_M_V16(ESP_VMIN_S8_ST_INCP_M, riscv_esp_vmin_s8_st_incp)
    LOWER_ESPV_ST_INCP_M_V8I16(ESP_VMIN_S16_ST_INCP_M,
                               riscv_esp_vmin_s16_st_incp)
    LOWER_ESPV_ST_INCP_M_V4I32(ESP_VMIN_S32_ST_INCP_M,
                               riscv_esp_vmin_s32_st_incp)
    LOWER_ESPV_ST_INCP_M_V16(ESP_VMIN_U8_ST_INCP_M, riscv_esp_vmin_u8_st_incp)
    LOWER_ESPV_ST_INCP_M_V8I16(ESP_VMIN_U16_ST_INCP_M,
                               riscv_esp_vmin_u16_st_incp)
    LOWER_ESPV_ST_INCP_M_V4I32(ESP_VMIN_U32_ST_INCP_M,
                               riscv_esp_vmin_u32_st_incp)
#undef LOWER_ESPV_ST_INCP_M_V4I32
#undef LOWER_ESPV_ST_INCP_M_V8I16
#undef LOWER_ESPV_ST_INCP_M_V16
  // VMUL LD/ST.INCP lowering (unified 2.1/2.2 API)
  case Intrinsic::riscv_esp_vmul_s8_ld_incp:
    return lowerVmulLdIncp(Op, DAG, Subtarget, MVT::v16i8,
                           RISCVISD::ESP_VMUL_S8_LD_INCP_PIE22_M);
  case Intrinsic::riscv_esp_vmul_u8_ld_incp:
    return lowerVmulLdIncp(Op, DAG, Subtarget, MVT::v16i8,
                           RISCVISD::ESP_VMUL_U8_LD_INCP_PIE22_M);
  case Intrinsic::riscv_esp_vmul_s16_ld_incp:
    return lowerVmulLdIncp(Op, DAG, Subtarget, MVT::v8i16,
                           RISCVISD::ESP_VMUL_S16_LD_INCP_PIE22_M);
  case Intrinsic::riscv_esp_vmul_u16_ld_incp:
    return lowerVmulLdIncp(Op, DAG, Subtarget, MVT::v8i16,
                           RISCVISD::ESP_VMUL_U16_LD_INCP_PIE22_M);
  case Intrinsic::riscv_esp_vmul_s8_st_incp:
    return lowerVmulStIncp(Op, DAG, Subtarget, MVT::v16i8,
                           RISCVISD::ESP_VMUL_S8_ST_INCP_M,
                           RISCVISD::ESP_VMUL_S8_ST_INCP_PIE22_M);
  case Intrinsic::riscv_esp_vmul_u8_st_incp:
    return lowerVmulStIncp(Op, DAG, Subtarget, MVT::v16i8,
                           RISCVISD::ESP_VMUL_U8_ST_INCP_M,
                           RISCVISD::ESP_VMUL_U8_ST_INCP_PIE22_M);
  case Intrinsic::riscv_esp_vmul_s16_st_incp:
    return lowerVmulStIncp(Op, DAG, Subtarget, MVT::v8i16,
                           RISCVISD::ESP_VMUL_S16_ST_INCP_M,
                           RISCVISD::ESP_VMUL_S16_ST_INCP_PIE22_M);
  case Intrinsic::riscv_esp_vmul_u16_st_incp:
    return lowerVmulStIncp(Op, DAG, Subtarget, MVT::v8i16,
                           RISCVISD::ESP_VMUL_U16_ST_INCP_M,
                           RISCVISD::ESP_VMUL_U16_ST_INCP_PIE22_M);

  case Intrinsic::riscv_esp_fft_ams_s16_ld_incp_uaup: {
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue QX = Op.getOperand(2);
    SDValue QY = Op.getOperand(3);
    SDValue QW = Op.getOperand(4);
    SDValue RS1 = Op.getOperand(5);
    SDValue SEL2 = Op.getOperand(6);
    SDValue SAT = Op.getOperand(7);
    SDValue UAStateIn = Op.getOperand(8);
    SDValue SarBytesIn = Op.getOperand(9);
    SDValue SarIn = Op.getOperand(10);
    if (Subtarget.hasVendorXespv2p1()) {
      diagnoseESPV21FftSat(DAG, SAT);
      return SDValue();
    }
    if (Subtarget.useESPV2P2Instructions())
      return lowerFftAmsLdIncpUaup(
          Op, DAG, Subtarget, RISCVISD::ESP_FFT_AMS_S16_LD_INCP_UAUP_PIE22_M);
    (void)Chain;
    (void)QX;
    (void)QY;
    (void)QW;
    (void)RS1;
    (void)SEL2;
    (void)UAStateIn;
    (void)SarBytesIn;
    (void)SarIn;
    return SDValue();
  }
  case Intrinsic::riscv_esp_fft_r2bf_s16_st_incp:
    return lowerFftR2bfStIncp(Op, DAG, Subtarget,
                              RISCVISD::ESP_FFT_R2BF_S16_ST_INCP_PIE22_M);
  case Intrinsic::riscv_esp_fft_ams_s16_ld_incp:
    return lowerFftAmsLdIncp(Op, DAG, Subtarget,
                             RISCVISD::ESP_FFT_AMS_S16_LD_INCP_PIE22_M);
  case Intrinsic::riscv_esp_fft_ams_s16_ld_r32_decp:
    return lowerFftAmsLdR32Decp(Op, DAG, Subtarget,
                                RISCVISD::ESP_FFT_AMS_S16_LD_R32_DECP_PIE22_M);
  case Intrinsic::riscv_esp_fft_ams_s16_st_incp:
    return lowerFftAmsStIncp(Op, DAG, Subtarget,
                             RISCVISD::ESP_FFT_AMS_S16_ST_INCP_PIE22_M);
  case Intrinsic::riscv_esp_fft_cmul_s16_ld_xp:
    return lowerFftCmulLdXp(Op, DAG, Subtarget,
                            RISCVISD::ESP_FFT_CMUL_S16_LD_XP_PIE22_M);
  case Intrinsic::riscv_esp_fft_cmul_s16_st_xp:
    return lowerFftCmulStXp(Op, DAG, Subtarget,
                            RISCVISD::ESP_FFT_CMUL_S16_ST_XP_PIE22_M);

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
  case Intrinsic::riscv_esp_vmulas_s16_qacc_ld_ip:
    return LowerVMULASQACCLDIP(Op, DAG, Subtarget,
                               RISCVISD::ESP_VMULAS_S16_QACC_LD_IP_M,
                               RISCVISD::ESP_VMULAS_S16_QACC_LD_IP_PIE22_M);
  case Intrinsic::riscv_esp_vmulas_s8_qacc_ld_ip:
    return LowerVMULASQACCLDIP(Op, DAG, Subtarget,
                               RISCVISD::ESP_VMULAS_S8_QACC_LD_IP_M,
                               RISCVISD::ESP_VMULAS_S8_QACC_LD_IP_PIE22_M);
  case Intrinsic::riscv_esp_vmulas_u16_qacc_ld_ip:
    return LowerVMULASQACCLDIP(Op, DAG, Subtarget,
                               RISCVISD::ESP_VMULAS_U16_QACC_LD_IP_M,
                               RISCVISD::ESP_VMULAS_U16_QACC_LD_IP_PIE22_M);
  case Intrinsic::riscv_esp_vmulas_u8_qacc_ld_ip:
    return LowerVMULASQACCLDIP(Op, DAG, Subtarget,
                               RISCVISD::ESP_VMULAS_U8_QACC_LD_IP_M,
                               RISCVISD::ESP_VMULAS_U8_QACC_LD_IP_PIE22_M);
  case Intrinsic::riscv_esp_vsmulas_s16_qacc_ld_incp:
    return LowerVSMULASQACCLDIP(Op, DAG, Subtarget,
                                RISCVISD::ESP_VSMULAS_S16_QACC_LD_INCP_M,
                                RISCVISD::ESP_VSMULAS_S16_QACC_LD_INCP_PIE22_M);
  case Intrinsic::riscv_esp_vsmulas_s8_qacc_ld_incp:
    return LowerVSMULASQACCLDIP(Op, DAG, Subtarget,
                                RISCVISD::ESP_VSMULAS_S8_QACC_LD_INCP_M,
                                RISCVISD::ESP_VSMULAS_S8_QACC_LD_INCP_PIE22_M);
  case Intrinsic::riscv_esp_vsmulas_u16_qacc_ld_incp:
    return LowerVSMULASQACCLDIP(Op, DAG, Subtarget,
                                RISCVISD::ESP_VSMULAS_U16_QACC_LD_INCP_M,
                                RISCVISD::ESP_VSMULAS_U16_QACC_LD_INCP_PIE22_M);
  case Intrinsic::riscv_esp_vsmulas_u8_qacc_ld_incp:
    return LowerVSMULASQACCLDIP(Op, DAG, Subtarget,
                                RISCVISD::ESP_VSMULAS_U8_QACC_LD_INCP_M,
                                RISCVISD::ESP_VSMULAS_U8_QACC_LD_INCP_PIE22_M);
  // VMULAS QACC LD XP
  case Intrinsic::riscv_esp_vmulas_s16_qacc_ld_xp:
    return LowerVMULASQACCLDXP(Op, DAG, Subtarget,
                               RISCVISD::ESP_VMULAS_S16_QACC_LD_XP_M,
                               RISCVISD::ESP_VMULAS_S16_QACC_LD_XP_PIE22_M);
  case Intrinsic::riscv_esp_vmulas_s8_qacc_ld_xp:
    return LowerVMULASQACCLDXP(Op, DAG, Subtarget,
                               RISCVISD::ESP_VMULAS_S8_QACC_LD_XP_M,
                               RISCVISD::ESP_VMULAS_S8_QACC_LD_XP_PIE22_M);
  case Intrinsic::riscv_esp_vmulas_u16_qacc_ld_xp:
    return LowerVMULASQACCLDXP(Op, DAG, Subtarget,
                               RISCVISD::ESP_VMULAS_U16_QACC_LD_XP_M,
                               RISCVISD::ESP_VMULAS_U16_QACC_LD_XP_PIE22_M);
  case Intrinsic::riscv_esp_vmulas_u8_qacc_ld_xp:
    return LowerVMULASQACCLDXP(Op, DAG, Subtarget,
                               RISCVISD::ESP_VMULAS_U8_QACC_LD_XP_M,
                               RISCVISD::ESP_VMULAS_U8_QACC_LD_XP_PIE22_M);
  // VMULAS QACC ST IP
  case Intrinsic::riscv_esp_vmulas_s16_qacc_st_ip:
    return LowerVMULASQACCSTIP(Op, DAG, Subtarget,
                               RISCVISD::ESP_VMULAS_S16_QACC_ST_IP_M,
                               RISCVISD::ESP_VMULAS_S16_QACC_ST_IP_PIE22_M);
  case Intrinsic::riscv_esp_vmulas_s8_qacc_st_ip:
    return LowerVMULASQACCSTIP(Op, DAG, Subtarget,
                               RISCVISD::ESP_VMULAS_S8_QACC_ST_IP_M,
                               RISCVISD::ESP_VMULAS_S8_QACC_ST_IP_PIE22_M);
  case Intrinsic::riscv_esp_vmulas_u16_qacc_st_ip:
    return LowerVMULASQACCSTIP(Op, DAG, Subtarget,
                               RISCVISD::ESP_VMULAS_U16_QACC_ST_IP_M,
                               RISCVISD::ESP_VMULAS_U16_QACC_ST_IP_PIE22_M);
  case Intrinsic::riscv_esp_vmulas_u8_qacc_st_ip:
    return LowerVMULASQACCSTIP(Op, DAG, Subtarget,
                               RISCVISD::ESP_VMULAS_U8_QACC_ST_IP_M,
                               RISCVISD::ESP_VMULAS_U8_QACC_ST_IP_PIE22_M);
  // VMULAS QACC ST XP
  case Intrinsic::riscv_esp_vmulas_s16_qacc_st_xp:
    return LowerVMULASQACCSTXP(Op, DAG, Subtarget,
                               RISCVISD::ESP_VMULAS_S16_QACC_ST_XP_M,
                               RISCVISD::ESP_VMULAS_S16_QACC_ST_XP_PIE22_M);
  case Intrinsic::riscv_esp_vmulas_s8_qacc_st_xp:
    return LowerVMULASQACCSTXP(Op, DAG, Subtarget,
                               RISCVISD::ESP_VMULAS_S8_QACC_ST_XP_M,
                               RISCVISD::ESP_VMULAS_S8_QACC_ST_XP_PIE22_M);
  case Intrinsic::riscv_esp_vmulas_u16_qacc_st_xp:
    return LowerVMULASQACCSTXP(Op, DAG, Subtarget,
                               RISCVISD::ESP_VMULAS_U16_QACC_ST_XP_M,
                               RISCVISD::ESP_VMULAS_U16_QACC_ST_XP_PIE22_M);
  case Intrinsic::riscv_esp_vmulas_u8_qacc_st_xp:
    return LowerVMULASQACCSTXP(Op, DAG, Subtarget,
                               RISCVISD::ESP_VMULAS_U8_QACC_ST_XP_M,
                               RISCVISD::ESP_VMULAS_U8_QACC_ST_XP_PIE22_M);
  // VMULAS QACC LDBC INCP
  case Intrinsic::riscv_esp_vmulas_s16_qacc_ldbc_incp:
    return LowerVMULASQACCLDBCINCP(
        Op, DAG, Subtarget, RISCVISD::ESP_VMULAS_S16_QACC_LDBC_INCP_M,
        RISCVISD::ESP_VMULAS_S16_QACC_LDBC_INCP_PIE22_M);
  case Intrinsic::riscv_esp_vmulas_s8_qacc_ldbc_incp:
    return LowerVMULASQACCLDBCINCP(
        Op, DAG, Subtarget, RISCVISD::ESP_VMULAS_S8_QACC_LDBC_INCP_M,
        RISCVISD::ESP_VMULAS_S8_QACC_LDBC_INCP_PIE22_M);
  case Intrinsic::riscv_esp_vmulas_u16_qacc_ldbc_incp:
    return LowerVMULASQACCLDBCINCP(
        Op, DAG, Subtarget, RISCVISD::ESP_VMULAS_U16_QACC_LDBC_INCP_M,
        RISCVISD::ESP_VMULAS_U16_QACC_LDBC_INCP_PIE22_M);
  // VMULAS XACC LD IP
  case Intrinsic::riscv_esp_vmulas_s16_xacc_ld_ip:
    return LowerVMULASXACCLDIP(Op, DAG, Subtarget,
                               RISCVISD::ESP_VMULAS_S16_XACC_LD_IP_M,
                               RISCVISD::ESP_VMULAS_S16_XACC_LD_IP_PIE22_M);
  case Intrinsic::riscv_esp_vmulas_s8_xacc_ld_ip:
    return LowerVMULASXACCLDIP(Op, DAG, Subtarget,
                               RISCVISD::ESP_VMULAS_S8_XACC_LD_IP_M,
                               RISCVISD::ESP_VMULAS_S8_XACC_LD_IP_PIE22_M);
  case Intrinsic::riscv_esp_vmulas_u16_xacc_ld_ip:
    return LowerVMULASXACCLDIP(Op, DAG, Subtarget,
                               RISCVISD::ESP_VMULAS_U16_XACC_LD_IP_M,
                               RISCVISD::ESP_VMULAS_U16_XACC_LD_IP_PIE22_M);
  case Intrinsic::riscv_esp_vmulas_u8_xacc_ld_ip:
    return LowerVMULASXACCLDIP(Op, DAG, Subtarget,
                               RISCVISD::ESP_VMULAS_U8_XACC_LD_IP_M,
                               RISCVISD::ESP_VMULAS_U8_XACC_LD_IP_PIE22_M);
  // VMULAS XACC LD XP
  case Intrinsic::riscv_esp_vmulas_s16_xacc_ld_xp:
    return LowerVMULASXACCLDXP(Op, DAG, Subtarget,
                               RISCVISD::ESP_VMULAS_S16_XACC_LD_XP_M,
                               RISCVISD::ESP_VMULAS_S16_XACC_LD_XP_PIE22_M);
  case Intrinsic::riscv_esp_vmulas_s8_xacc_ld_xp:
    return LowerVMULASXACCLDXP(Op, DAG, Subtarget,
                               RISCVISD::ESP_VMULAS_S8_XACC_LD_XP_M,
                               RISCVISD::ESP_VMULAS_S8_XACC_LD_XP_PIE22_M);
  case Intrinsic::riscv_esp_vmulas_u16_xacc_ld_xp:
    return LowerVMULASXACCLDXP(Op, DAG, Subtarget,
                               RISCVISD::ESP_VMULAS_U16_XACC_LD_XP_M,
                               RISCVISD::ESP_VMULAS_U16_XACC_LD_XP_PIE22_M);
  case Intrinsic::riscv_esp_vmulas_u8_xacc_ld_xp:
    return LowerVMULASXACCLDXP(Op, DAG, Subtarget,
                               RISCVISD::ESP_VMULAS_U8_XACC_LD_XP_M,
                               RISCVISD::ESP_VMULAS_U8_XACC_LD_XP_PIE22_M);
  // VMULAS XACC ST IP
  case Intrinsic::riscv_esp_vmulas_s16_xacc_st_ip:
    return LowerVMULASXACCSTIP(Op, DAG, Subtarget,
                               RISCVISD::ESP_VMULAS_S16_XACC_ST_IP_M,
                               RISCVISD::ESP_VMULAS_S16_XACC_ST_IP_PIE22_M);
  case Intrinsic::riscv_esp_vmulas_s8_xacc_st_ip:
    return LowerVMULASXACCSTIP(Op, DAG, Subtarget,
                               RISCVISD::ESP_VMULAS_S8_XACC_ST_IP_M,
                               RISCVISD::ESP_VMULAS_S8_XACC_ST_IP_PIE22_M);
  case Intrinsic::riscv_esp_vmulas_u16_xacc_st_ip:
    return LowerVMULASXACCSTIP(Op, DAG, Subtarget,
                               RISCVISD::ESP_VMULAS_U16_XACC_ST_IP_M,
                               RISCVISD::ESP_VMULAS_U16_XACC_ST_IP_PIE22_M);
  case Intrinsic::riscv_esp_vmulas_u8_xacc_st_ip:
    return LowerVMULASXACCSTIP(Op, DAG, Subtarget,
                               RISCVISD::ESP_VMULAS_U8_XACC_ST_IP_M,
                               RISCVISD::ESP_VMULAS_U8_XACC_ST_IP_PIE22_M);
  // VMULAS XACC ST XP
  case Intrinsic::riscv_esp_vmulas_s16_xacc_st_xp:
    return LowerVMULASXACCSTXP(Op, DAG, Subtarget,
                               RISCVISD::ESP_VMULAS_S16_XACC_ST_XP_M,
                               RISCVISD::ESP_VMULAS_S16_XACC_ST_XP_PIE22_M);
  case Intrinsic::riscv_esp_vmulas_s8_xacc_st_xp:
    return LowerVMULASXACCSTXP(Op, DAG, Subtarget,
                               RISCVISD::ESP_VMULAS_S8_XACC_ST_XP_M,
                               RISCVISD::ESP_VMULAS_S8_XACC_ST_XP_PIE22_M);
  case Intrinsic::riscv_esp_vmulas_u16_xacc_st_xp:
    return LowerVMULASXACCSTXP(Op, DAG, Subtarget,
                               RISCVISD::ESP_VMULAS_U16_XACC_ST_XP_M,
                               RISCVISD::ESP_VMULAS_U16_XACC_ST_XP_PIE22_M);
  case Intrinsic::riscv_esp_vmulas_u8_xacc_st_xp:
    return LowerVMULASXACCSTXP(Op, DAG, Subtarget,
                               RISCVISD::ESP_VMULAS_U8_XACC_ST_XP_M,
                               RISCVISD::ESP_VMULAS_U8_XACC_ST_XP_PIE22_M);
  case Intrinsic::riscv_esp_vmulas_u8_qacc_ldbc_incp:
    return LowerVMULASQACCLDBCINCP(
        Op, DAG, Subtarget, RISCVISD::ESP_VMULAS_U8_QACC_LDBC_INCP_M,
        RISCVISD::ESP_VMULAS_U8_QACC_LDBC_INCP_PIE22_M);

  case Intrinsic::riscv_esp_srcq_128_st_incp: {
    // Lower intrinsic to custom SDNode that will be matched to
    // ESP_SRCQ_128_ST_INCP Intrinsic: (chain, int_id, SAR_BYTES, qy, qw, ptr)
    // Returns: ptr (updated pointer)
    // SDNode: (SAR_BYTES, qy, qw, ptr) -> (ptr)
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue SarBytes =
        Op.getOperand(2); // SAR_BYTES (32-bit, only low 4 bits used)
    SDValue QY = Op.getOperand(3);
    SDValue QW = Op.getOperand(4);
    SDValue Ptr = Op.getOperand(5);

    EVT PtrVT = Ptr.getValueType();
    SDVTList VTs = DAG.getVTList(PtrVT, MVT::Other);

    SDValue Ops[] = {Chain, SarBytes, QY, QW, Ptr};
    EVT MemVT = MVT::v16i8;
    // Note: This intrinsic always arrives as MemIntrinsicSDNode because
    //       getTgtMemIntrinsic returns true for it.
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_SRCQ_128_ST_INCP_M, DL,
                                           VTs, Ops, MemVT, MMO);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1)}, DL);
  }
  case Intrinsic::riscv_esp_src_q_ld_ip: {
    // Lower intrinsic to custom SDNode that will be matched to ESP_SRC_Q_LD_IP
    // Intrinsic: (chain, int_id, SAR_BYTES, qy, qw, ptr, imm)
    // Returns: qw (updated), qu (loaded), ptr (updated)
    // SDNode outputs: qu (0), ptr (1), qw (2) - matches instruction output
    // order
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue SarBytes =
        Op.getOperand(2); // SAR_BYTES (32-bit, only low 4 bits used)
    SDValue QY = Op.getOperand(3);
    SDValue QW = Op.getOperand(4);
    SDValue Ptr = Op.getOperand(5);
    SDValue Imm = Op.getOperand(6);

    EVT VecVT = MVT::v16i8;
    EVT PtrVT = Ptr.getValueType();
    // Adjust output order: qu, ptr, qw (matches instruction definition)
    SDVTList VTs = DAG.getVTList(VecVT, PtrVT, VecVT, MVT::Other);

    SDValue Ops[] = {Chain, SarBytes, QY, QW, Ptr, Imm};
    EVT MemVT = MVT::v16i8;
    // Note: This intrinsic always arrives as MemIntrinsicSDNode because
    //       getTgtMemIntrinsic returns true for it.
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_SRC_Q_LD_IP_M, DL, VTs,
                                           Ops, MemVT, MMO);
    // Intrinsic return value order: qw, qu, ptr
    // Node output order: qu (0), ptr (1), qw (2)
    // Need to reorder to match intrinsic return value order
    return DAG.getMergeValues(
        {
            Node.getValue(2), // qw (from Node result 2)
            Node.getValue(0), // qu (from Node result 0)
            Node.getValue(1), // ptr (from Node result 1)
            Node.getValue(3)  // chain
        },
        DL);
  }
  case Intrinsic::riscv_esp_src_q_ld_xp: {
    // Lower intrinsic to custom SDNode that will be matched to ESP_SRC_Q_LD_XP
    // Intrinsic: (chain, int_id, SAR_BYTES, qy, qw, ptr, rs2)
    // Returns: qw (updated), qu (loaded), ptr (updated)
    // SDNode outputs: qu (0), ptr (1), qw (2) - matches instruction output
    // order
    SDLoc DL(Op);
    SDValue Chain = Op.getOperand(0);
    SDValue SarBytes =
        Op.getOperand(2); // SAR_BYTES (32-bit, only low 4 bits used)
    SDValue QY = Op.getOperand(3);
    SDValue QW = Op.getOperand(4);
    SDValue Ptr = Op.getOperand(5);
    SDValue Rs2 = Op.getOperand(6);

    EVT VecVT = MVT::v16i8;
    EVT PtrVT = Ptr.getValueType();
    // Adjust output order: qu, ptr, qw (matches instruction definition)
    SDVTList VTs = DAG.getVTList(VecVT, PtrVT, VecVT, MVT::Other);

    // SDNode operand order: SAR_BYTES, qy, qw, ptr, offset (register)
    SDValue Ops[] = {Chain, SarBytes, QY, QW, Ptr, Rs2};
    EVT MemVT = MVT::v16i8;
    // Note: This intrinsic always arrives as MemIntrinsicSDNode because
    //       getTgtMemIntrinsic returns true for it.
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(RISCVISD::ESP_SRC_Q_LD_XP_M, DL, VTs,
                                           Ops, MemVT, MMO);
    // Intrinsic return value order: qw, qu, ptr
    // Node output order: qu (0), ptr (1), qw (2)
    // Need to reorder to match intrinsic return value order
    return DAG.getMergeValues(
        {
            Node.getValue(2), // qw (from Node result 2)
            Node.getValue(0), // qu (from Node result 0)
            Node.getValue(1), // ptr (from Node result 1)
            Node.getValue(3)  // chain
        },
        DL);
  }

  case Intrinsic::riscv_esp_vcmulas_s8_qacc_h_ld_ip:
    return lowerVcmulasLdIp(Op, DAG, Subtarget,
                            RISCVISD::ESP_VCMULAS_S8_QACC_H_LD_IP_M,
                            RISCVISD::ESP_VCMULAS_S8_QACC_H_LD_IP_PIE22_M);
  case Intrinsic::riscv_esp_vcmulas_s8_qacc_h_ld_xp:
    return lowerVcmulasLdXp(Op, DAG, Subtarget,
                            RISCVISD::ESP_VCMULAS_S8_QACC_H_LD_XP_M,
                            RISCVISD::ESP_VCMULAS_S8_QACC_H_LD_XP_PIE22_M);
  case Intrinsic::riscv_esp_vcmulas_s8_qacc_l_ld_ip:
    return lowerVcmulasLdIp(Op, DAG, Subtarget,
                            RISCVISD::ESP_VCMULAS_S8_QACC_L_LD_IP_M,
                            RISCVISD::ESP_VCMULAS_S8_QACC_L_LD_IP_PIE22_M);
  case Intrinsic::riscv_esp_vcmulas_s8_qacc_l_ld_xp:
    return lowerVcmulasLdXp(Op, DAG, Subtarget,
                            RISCVISD::ESP_VCMULAS_S8_QACC_L_LD_XP_M,
                            RISCVISD::ESP_VCMULAS_S8_QACC_L_LD_XP_PIE22_M);
  case Intrinsic::riscv_esp_vcmulas_s16_qacc_h_ld_ip:
    return lowerVcmulasLdIp(Op, DAG, Subtarget,
                            RISCVISD::ESP_VCMULAS_S16_QACC_H_LD_IP_M,
                            RISCVISD::ESP_VCMULAS_S16_QACC_H_LD_IP_PIE22_M);
  case Intrinsic::riscv_esp_vcmulas_s16_qacc_h_ld_xp:
    return lowerVcmulasLdXp(Op, DAG, Subtarget,
                            RISCVISD::ESP_VCMULAS_S16_QACC_H_LD_XP_M,
                            RISCVISD::ESP_VCMULAS_S16_QACC_H_LD_XP_PIE22_M);
  case Intrinsic::riscv_esp_vcmulas_s16_qacc_l_ld_ip:
    return lowerVcmulasLdIp(Op, DAG, Subtarget,
                            RISCVISD::ESP_VCMULAS_S16_QACC_L_LD_IP_M,
                            RISCVISD::ESP_VCMULAS_S16_QACC_L_LD_IP_PIE22_M);
  case Intrinsic::riscv_esp_vcmulas_s16_qacc_l_ld_xp:
    return lowerVcmulasLdXp(Op, DAG, Subtarget,
                            RISCVISD::ESP_VCMULAS_S16_QACC_L_LD_XP_M,
                            RISCVISD::ESP_VCMULAS_S16_QACC_L_LD_XP_PIE22_M);
  default:
    return SDValue(); // Not an ESPV intrinsic handled here
  }
}

// LD.INCP.M: (chain, int_id, qx, qy, rs1) -> {vec_result, loaded v16i8, ptr,
// chain}
static SDValue LowerESPLdIncpM(SDValue Op, SelectionDAG &DAG,
                               unsigned ISDOpcode, MVT ResVT) {
  SDLoc DL(Op);
  SDValue Chain = Op.getOperand(0);
  SDValue QX = Op.getOperand(2);
  SDValue QY = Op.getOperand(3);
  SDValue RS1 = Op.getOperand(4);
  EVT PtrVT = RS1.getValueType();
  SDVTList VTs = DAG.getVTList(ResVT, MVT::v16i8, PtrVT, MVT::Other);
  SDValue Ops[] = {Chain, QX, QY, RS1};
  EVT MemVT = MVT::v16i8;
  auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
  MachineMemOperand *MMO = MemIntr->getMemOperand();
  SDValue Node = DAG.getMemIntrinsicNode(ISDOpcode, DL, VTs, Ops, MemVT, MMO);
  return DAG.getMergeValues(
      {Node.getValue(0), Node.getValue(1), Node.getValue(2), Node.getValue(3)},
      DL);
}

static SDValue LowerLDXACCIP(SDValue Op, SelectionDAG &DAG,
                             unsigned ISDOpcode) {
  // Intrinsic: (chain, int_id, xacc_low_in, xacc_high_in, ptr, offset) -> {ptr,
  // new_xacc_low, new_xacc_high, chain} Mixed model: XACC as {i32 low, i32
  // high}
  SDLoc DL(Op);
  SDValue Chain = Op.getOperand(0);
  SDValue XACCLowIn = Op.getOperand(2); // i32 passthru (XACC[31:0])
  SDValue XACCHighIn =
      Op.getOperand(3); // i32 passthru (XACC[39:32], only low 8 bits valid)
  SDValue Ptr = Op.getOperand(4);
  SDValue Offset = Op.getOperand(5);

  EVT PtrVT = Ptr.getValueType();
  EVT MemVT = MVT::i64; // Load 64-bit, use low 40 bits
  // SDNode with SDNPHasChain and SDNPOutGlue: Chain and Glue are added
  // automatically SDTypeProfile defines 3 explicit results (ptr, new_xacc_low,
  // new_xacc_high), plus Chain and Glue = 5 values total
  SmallVector<EVT, 5> VTs = {PtrVT, MVT::i32, MVT::i32, MVT::Other, MVT::Glue};
  SDVTList VTList = DAG.getVTList(VTs);
  // Operands: Chain (SDNPHasChain requires it as first operand), XACC low, XACC
  // high, Ptr, Offset SDTypeProfile defines 4 operands, SDNPHasChain adds Chain
  // as first operand = 5 total SDNPOptInGlue means Glue is optional and doesn't
  // need to be explicitly passed Passthru operands XACCLowIn and XACCHighIn
  // establish data dependency (phantom operands for data flow) No need for
  // CopyToReg - passthru operands directly establish data dependency
  SDValue Ops[] = {Chain, XACCLowIn, XACCHighIn, Ptr, Offset};

  // This intrinsic always arrives as MemIntrinsicSDNode because
  // getTgtMemIntrinsic returns true for it.
  auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
  MachineMemOperand *MMO = MemIntr->getMemOperand();
  SDValue Node =
      DAG.getMemIntrinsicNode(ISDOpcode, DL, VTList, Ops, MemVT, MMO);
  // SDNode returns (ptr, new_xacc_low, new_xacc_high, chain, glue)
  SDValue PtrOut = Node.getValue(0);
  SDValue NewXACCLow =
      Node.getValue(1); // XACC_LOW virtual register from instruction output
  SDValue NewXACCHigh =
      Node.getValue(2); // XACC_HIGH virtual register from instruction output
  Chain = Node.getValue(3);
  return DAG.getMergeValues({PtrOut, NewXACCLow, NewXACCHigh, Chain}, DL);
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
  // xacc_high_unchanged, chain, glue). This intrinsic always arrives as
  // MemIntrinsicSDNode because getTgtMemIntrinsic returns true for it.
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

// VMULAS XACC LD IP Lowering
// VMULAS XACC LD XP Lowering
// VMULAS XACC ST IP Lowering
// VMULAS XACC ST XP Lowering
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
  // v16i8, v16i8, v16i8, chain} SDNode returns: (QACC_L, QACC_H, ptr, chain) -
  // explicit outputs
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
  if (!Subtarget.hasESPVTargetLowering())
    return SDValue();

  unsigned IntNo = Op.getConstantOperandVal(0);
  SDLoc DL(Op);

  switch (IntNo) {
  // ESP MAX/MIN reduction intrinsics - lower to vecreduce nodes for pattern
  // matching
  case Intrinsic::riscv_esp_max_s8_a:
  case Intrinsic::riscv_esp_max_s16_a:
  case Intrinsic::riscv_esp_max_s32_a:
    return DAG.getNode(ISD::VECREDUCE_SMAX, DL, Op.getValueType(),
                       Op.getOperand(1));
  case Intrinsic::riscv_esp_max_u8_a:
  case Intrinsic::riscv_esp_max_u16_a:
  case Intrinsic::riscv_esp_max_u32_a:
    return DAG.getNode(ISD::VECREDUCE_UMAX, DL, Op.getValueType(),
                       Op.getOperand(1));
  case Intrinsic::riscv_esp_min_s8_a:
  case Intrinsic::riscv_esp_min_s16_a:
  case Intrinsic::riscv_esp_min_s32_a:
    return DAG.getNode(ISD::VECREDUCE_SMIN, DL, Op.getValueType(),
                       Op.getOperand(1));
  case Intrinsic::riscv_esp_min_u8_a:
  case Intrinsic::riscv_esp_min_u16_a:
  case Intrinsic::riscv_esp_min_u32_a:
    return DAG.getNode(ISD::VECREDUCE_UMIN, DL, Op.getValueType(),
                       Op.getOperand(1));
  case Intrinsic::riscv_esp_zero_qacc: {
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
  // MOVX.R/W.XACC.H/L - Read/Write XACC subregisters with explicit state
  // passing
  case Intrinsic::riscv_esp_movx_r_xacc_l: {
    // ESP.MOVX.R.XACC.L - Read XACC[31:0] (low 32 bits)
    // Intrinsic: (i32 xacc_l) -> i32
    // Instruction: ESP_MOVX_R_XACC_L outputs GPRPIE (i32)
    // Instruction operation: rd[31:0] = XACC[31:0]
    // Note: xacc_l parameter is for explicit state passing (data flow),
    // hardware reads directly from XACC We pass xacc_l as input operand to
    // maintain data flow dependency in DAG
    SDLoc DL(Op);
    SDValue XACCLowIn =
        Op.getOperand(1); // i32 input (passthru for explicit state passing)
    // Generate machine instruction with passthru operand to maintain data flow
    // Hardware ignores this operand but it ensures compiler tracks the
    // dependency
    SDVTList VTs = DAG.getVTList(MVT::i32);
    SmallVector<SDValue, 1> Ops = {
        XACCLowIn}; // Pass passthru to maintain data flow
    MachineSDNode *Inst =
        DAG.getMachineNode(RISCV::ESP_MOVX_R_XACC_L, DL, VTs, Ops);
    return SDValue(Inst, 0); // Returns i32
  }
  case Intrinsic::riscv_esp_movx_w_xacc_l: {
    // ESP.MOVX.W.XACC.L - Write XACC[31:0] (low 32 bits)
    // This intrinsic can be directly matched by TableGen patterns (i32 types
    // match)
    return SDValue();
  }
  case Intrinsic::riscv_esp_movx_r_xacc_h: {
    // ESP.MOVX.R.XACC.H - Read XACC[39:32] (high 8 bits)
    // Intrinsic: (i32 xacc_h) -> i32 (xacc_h is i32 but only low 8 bits valid)
    // Instruction: ESP_MOVX_R_XACC_H outputs GPRPIE (i32)
    // Instruction operation: rd[31:0] = {24'b0, XACC[39:32]} - zero-extends
    // 8-bit to 32-bit Note: xacc_h parameter is for explicit state passing
    // (data flow), hardware reads directly from XACC We pass xacc_h as input
    // operand to maintain data flow dependency in DAG
    SDLoc DL(Op);
    SDValue XACCHigh =
        Op.getOperand(1); // i32 xacc_h (passthru, only low 8 bits valid)

    // Generate machine instruction with passthru operand to maintain data flow
    // Hardware ignores this operand but it ensures compiler tracks the
    // dependency
    SDVTList VTs = DAG.getVTList(MVT::i32);
    SmallVector<SDValue, 1> Ops = {
        XACCHigh}; // Pass passthru to maintain data flow
    MachineSDNode *Inst =
        DAG.getMachineNode(RISCV::ESP_MOVX_R_XACC_H, DL, VTs, Ops);
    SDValue Result32 = SDValue(Inst, 0);

    // Return i32 directly (instruction zero-extends 8-bit value to 32-bit, only
    // low 8 bits valid) XACCHigh operand maintains data flow dependency even
    // though hardware doesn't use it
    return Result32;
  }
  case Intrinsic::riscv_esp_movx_w_xacc_h: {
    // ESP.MOVX.W.XACC.H - Write XACC[39:32] (high 8 bits)
    // Intrinsic: (i32 value) -> i32 (input is i32 to avoid type promotion
    // issues in RV32) Instruction: ESP_MOVX_W_XACC_H outputs XACC_HIGH register
    // type, hardware uses only low 8 bits Type legalizer will handle conversion
    // from XACC_HIGH to i32 if needed
    SDLoc DL(Op);
    SDValue Val = Op.getOperand(1); // i32 input (only low 8 bits used)

    // Val is already i32, use directly (instruction uses only low 8 bits)
    // Generate machine instruction directly - outputs XACC_HIGH register type
    SDVTList VTs = DAG.getVTList(MVT::i32); // Output is i32 (will be converted
                                            // from XACC_HIGH by type legalizer)
    SmallVector<SDValue, 1> Ops = {Val};
    MachineSDNode *Inst =
        DAG.getMachineNode(RISCV::ESP_MOVX_W_XACC_H, DL, VTs, Ops);
    return SDValue(
        Inst,
        0); // Returns i32 (type legalizer handles XACC_HIGH -> i32 conversion)
  }
  case Intrinsic::riscv_esp_vmul_s8:
    return lowerVmulBasic(Op, DAG, Subtarget, MVT::v16i8,
                          RISCVISD::ESP_VMUL_S8_M_PIE22);
  case Intrinsic::riscv_esp_vmul_u8:
    return lowerVmulBasic(Op, DAG, Subtarget, MVT::v16i8,
                          RISCVISD::ESP_VMUL_U8_M_PIE22);
  case Intrinsic::riscv_esp_vmul_s16:
    return lowerVmulBasic(Op, DAG, Subtarget, MVT::v8i16,
                          RISCVISD::ESP_VMUL_S16_M_PIE22);
  case Intrinsic::riscv_esp_vmul_u16:
    return lowerVmulBasic(Op, DAG, Subtarget, MVT::v8i16,
                          RISCVISD::ESP_VMUL_U16_M_PIE22);
  case Intrinsic::riscv_esp_vmul_s16_s8xs8:
    return lowerVmulS8xS8(Op, DAG, Subtarget,
                          RISCVISD::ESP_VMUL_S16_S8XS8_PIE22_M);
  case Intrinsic::riscv_esp_vmul_s32_s16xs16:
    return lowerVmulS16xS16(Op, DAG, Subtarget,
                            RISCVISD::ESP_VMUL_S32_S16XS16_PIE22_M);
  case Intrinsic::riscv_esp_mov_s16_qacc: {
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
  case Intrinsic::riscv_esp_mov_s8_qacc: {
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
  case Intrinsic::riscv_esp_mov_u16_qacc: {
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
  case Intrinsic::riscv_esp_mov_u8_qacc: {
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
  case Intrinsic::riscv_esp_zero_xacc: {
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
  case Intrinsic::riscv_esp_vsmulas_s16_qacc:
    return lowerVsmulasQaccCompute(Op, DAG, Subtarget,
                                   RISCVISD::ESP_VSMULAS_S16_QACC_M,
                                   RISCVISD::ESP_VSMULAS_S16_QACC_PIE22);
  case Intrinsic::riscv_esp_vsmulas_s8_qacc:
    return lowerVsmulasQaccCompute(Op, DAG, Subtarget,
                                   RISCVISD::ESP_VSMULAS_S8_QACC_M,
                                   RISCVISD::ESP_VSMULAS_S8_QACC_PIE22);
  case Intrinsic::riscv_esp_vsmulas_u16_qacc:
    return lowerVsmulasQaccCompute(Op, DAG, Subtarget,
                                   RISCVISD::ESP_VSMULAS_U16_QACC_M,
                                   RISCVISD::ESP_VSMULAS_U16_QACC_PIE22);
  case Intrinsic::riscv_esp_vsmulas_u8_qacc:
    return lowerVsmulasQaccCompute(Op, DAG, Subtarget,
                                   RISCVISD::ESP_VSMULAS_U8_QACC_M,
                                   RISCVISD::ESP_VSMULAS_U8_QACC_PIE22);
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
  case Intrinsic::riscv_esp_fft_r2bf_s16:
    return lowerFftR2bf(Op, DAG, Subtarget, RISCVISD::ESP_FFT_R2BF_S16_M_PIE22);
  // mr-vcmulas unified API (+xespv PIE22 path). ld_ip/ld_xp stay on WChain
  // only.
  case Intrinsic::riscv_esp_vcmulas_s8_qacc_h:
    return lowerVcmulasCompute(Op, DAG, Subtarget,
                               RISCVISD::ESP_VCMULAS_S8_QACC_H_M,
                               RISCVISD::ESP_VCMULAS_S8_QACC_H_PIE22);
  case Intrinsic::riscv_esp_vcmulas_s8_qacc_l:
    return lowerVcmulasCompute(Op, DAG, Subtarget,
                               RISCVISD::ESP_VCMULAS_S8_QACC_L_M,
                               RISCVISD::ESP_VCMULAS_S8_QACC_L_PIE22);
  case Intrinsic::riscv_esp_vcmulas_s16_qacc_h:
    return lowerVcmulasCompute(Op, DAG, Subtarget,
                               RISCVISD::ESP_VCMULAS_S16_QACC_H_M,
                               RISCVISD::ESP_VCMULAS_S16_QACC_H_PIE22);
  case Intrinsic::riscv_esp_vcmulas_s16_qacc_l:
    return lowerVcmulasCompute(Op, DAG, Subtarget,
                               RISCVISD::ESP_VCMULAS_S16_QACC_L_M,
                               RISCVISD::ESP_VCMULAS_S16_QACC_L_PIE22);
  case Intrinsic::riscv_esp_vmulas_s16_qacc:
    return lowerVmulasQaccCompute(Op, DAG, Subtarget,
                                  RISCVISD::ESP_VMULAS_S16_QACC_M,
                                  RISCVISD::ESP_VMULAS_S16_QACC_PIE22);
  case Intrinsic::riscv_esp_vmulas_s8_qacc:
    return lowerVmulasQaccCompute(Op, DAG, Subtarget,
                                  RISCVISD::ESP_VMULAS_S8_QACC_M,
                                  RISCVISD::ESP_VMULAS_S8_QACC_PIE22);
  case Intrinsic::riscv_esp_vmulas_u16_qacc:
    return lowerVmulasQaccCompute(Op, DAG, Subtarget,
                                  RISCVISD::ESP_VMULAS_U16_QACC_M,
                                  RISCVISD::ESP_VMULAS_U16_QACC_PIE22);
  case Intrinsic::riscv_esp_vmulas_u8_qacc:
    return lowerVmulasQaccCompute(Op, DAG, Subtarget,
                                  RISCVISD::ESP_VMULAS_U8_QACC_M,
                                  RISCVISD::ESP_VMULAS_U8_QACC_PIE22);
  case Intrinsic::riscv_esp_vmulas_s16_xacc:
    return lowerVmulasXaccCompute(Op, DAG, Subtarget,
                                  RISCVISD::ESP_VMULAS_S16_XACC_M,
                                  RISCVISD::ESP_VMULAS_S16_XACC_PIE22);
  case Intrinsic::riscv_esp_vmulas_s8_xacc:
    return lowerVmulasXaccCompute(Op, DAG, Subtarget,
                                  RISCVISD::ESP_VMULAS_S8_XACC_M,
                                  RISCVISD::ESP_VMULAS_S8_XACC_PIE22);
  case Intrinsic::riscv_esp_vmulas_u16_xacc:
    return lowerVmulasXaccCompute(Op, DAG, Subtarget,
                                  RISCVISD::ESP_VMULAS_U16_XACC_M,
                                  RISCVISD::ESP_VMULAS_U16_XACC_PIE22);
  case Intrinsic::riscv_esp_vmulas_u8_xacc:
    return lowerVmulasXaccCompute(Op, DAG, Subtarget,
                                  RISCVISD::ESP_VMULAS_U8_XACC_M,
                                  RISCVISD::ESP_VMULAS_U8_XACC_PIE22);

  case Intrinsic::riscv_esp_srs_s_xacc:
    return lowerSrsXacc(Op, DAG, Subtarget, RISCVISD::ESP_SRS_S_XACC_M,
                        RISCVISD::ESP_SRS_S_XACC_PIE22);
  // mr-vadds-vsubs unified API (+xespv PIE22 path)
  case Intrinsic::riscv_esp_vadd_s8:
    return lowerVaddVsubSatBasic(Op, DAG, Subtarget, MVT::v16i8,
                                 RISCVISD::ESP_VADD_S8_PIE22);
  case Intrinsic::riscv_esp_vadd_s8_ld_incp:
    return lowerVaddVsubLdIncpSat(Op, DAG, Subtarget, MVT::v16i8,
                                  RISCVISD::ESP_VADD_S8_LD_INCP_PIE22_M);
  case Intrinsic::riscv_esp_vadd_s8_st_incp:
    return lowerVaddVsubStIncpSat(Op, DAG, Subtarget, MVT::v16i8,
                                  RISCVISD::ESP_VADD_S8_ST_INCP_PIE22_M);
  case Intrinsic::riscv_esp_vadd_s16:
    return lowerVaddVsubSatBasic(Op, DAG, Subtarget, MVT::v8i16,
                                 RISCVISD::ESP_VADD_S16_PIE22);
  case Intrinsic::riscv_esp_vadd_s16_ld_incp:
    return lowerVaddVsubLdIncpSat(Op, DAG, Subtarget, MVT::v8i16,
                                  RISCVISD::ESP_VADD_S16_LD_INCP_PIE22_M);
  case Intrinsic::riscv_esp_vadd_s16_st_incp:
    return lowerVaddVsubStIncpSat(Op, DAG, Subtarget, MVT::v8i16,
                                  RISCVISD::ESP_VADD_S16_ST_INCP_PIE22_M);
  case Intrinsic::riscv_esp_vadd_s32:
    return lowerVaddVsubSatBasic(Op, DAG, Subtarget, MVT::v4i32,
                                 RISCVISD::ESP_VADD_S32_PIE22);
  case Intrinsic::riscv_esp_vadd_s32_ld_incp:
    return lowerVaddVsubLdIncpSat(Op, DAG, Subtarget, MVT::v4i32,
                                  RISCVISD::ESP_VADD_S32_LD_INCP_PIE22_M);
  case Intrinsic::riscv_esp_vadd_s32_st_incp:
    return lowerVaddVsubStIncpSat(Op, DAG, Subtarget, MVT::v4i32,
                                  RISCVISD::ESP_VADD_S32_ST_INCP_PIE22_M);
  case Intrinsic::riscv_esp_vadd_u8:
    return lowerVaddVsubSatBasic(Op, DAG, Subtarget, MVT::v16i8,
                                 RISCVISD::ESP_VADD_U8_PIE22);
  case Intrinsic::riscv_esp_vadd_u8_ld_incp:
    return lowerVaddVsubLdIncpSat(Op, DAG, Subtarget, MVT::v16i8,
                                  RISCVISD::ESP_VADD_U8_LD_INCP_PIE22_M);
  case Intrinsic::riscv_esp_vadd_u8_st_incp:
    return lowerVaddVsubStIncpSat(Op, DAG, Subtarget, MVT::v16i8,
                                  RISCVISD::ESP_VADD_U8_ST_INCP_PIE22_M);
  case Intrinsic::riscv_esp_vadd_u16:
    return lowerVaddVsubSatBasic(Op, DAG, Subtarget, MVT::v8i16,
                                 RISCVISD::ESP_VADD_U16_PIE22);
  case Intrinsic::riscv_esp_vadd_u16_ld_incp:
    return lowerVaddVsubLdIncpSat(Op, DAG, Subtarget, MVT::v8i16,
                                  RISCVISD::ESP_VADD_U16_LD_INCP_PIE22_M);
  case Intrinsic::riscv_esp_vadd_u16_st_incp:
    return lowerVaddVsubStIncpSat(Op, DAG, Subtarget, MVT::v8i16,
                                  RISCVISD::ESP_VADD_U16_ST_INCP_PIE22_M);
  case Intrinsic::riscv_esp_vadd_u32:
    return lowerVaddVsubSatBasic(Op, DAG, Subtarget, MVT::v4i32,
                                 RISCVISD::ESP_VADD_U32_PIE22);
  case Intrinsic::riscv_esp_vadd_u32_ld_incp:
    return lowerVaddVsubLdIncpSat(Op, DAG, Subtarget, MVT::v4i32,
                                  RISCVISD::ESP_VADD_U32_LD_INCP_PIE22_M);
  case Intrinsic::riscv_esp_vadd_u32_st_incp:
    return lowerVaddVsubStIncpSat(Op, DAG, Subtarget, MVT::v4i32,
                                  RISCVISD::ESP_VADD_U32_ST_INCP_PIE22_M);
  case Intrinsic::riscv_esp_vsub_s8:
    return lowerVaddVsubSatBasic(Op, DAG, Subtarget, MVT::v16i8,
                                 RISCVISD::ESP_VSUB_S8_PIE22);
  case Intrinsic::riscv_esp_vsub_s8_ld_incp:
    return lowerVaddVsubLdIncpSat(Op, DAG, Subtarget, MVT::v16i8,
                                  RISCVISD::ESP_VSUB_S8_LD_INCP_PIE22_M);
  case Intrinsic::riscv_esp_vsub_s8_st_incp:
    return lowerVaddVsubStIncpSat(Op, DAG, Subtarget, MVT::v16i8,
                                  RISCVISD::ESP_VSUB_S8_ST_INCP_PIE22_M);
  case Intrinsic::riscv_esp_vsub_s16:
    return lowerVaddVsubSatBasic(Op, DAG, Subtarget, MVT::v8i16,
                                 RISCVISD::ESP_VSUB_S16_PIE22);
  case Intrinsic::riscv_esp_vsub_s16_ld_incp:
    return lowerVaddVsubLdIncpSat(Op, DAG, Subtarget, MVT::v8i16,
                                  RISCVISD::ESP_VSUB_S16_LD_INCP_PIE22_M);
  case Intrinsic::riscv_esp_vsub_s16_st_incp:
    return lowerVaddVsubStIncpSat(Op, DAG, Subtarget, MVT::v8i16,
                                  RISCVISD::ESP_VSUB_S16_ST_INCP_PIE22_M);
  case Intrinsic::riscv_esp_vsub_s32:
    return lowerVaddVsubSatBasic(Op, DAG, Subtarget, MVT::v4i32,
                                 RISCVISD::ESP_VSUB_S32_PIE22);
  case Intrinsic::riscv_esp_vsub_s32_ld_incp:
    return lowerVaddVsubLdIncpSat(Op, DAG, Subtarget, MVT::v4i32,
                                  RISCVISD::ESP_VSUB_S32_LD_INCP_PIE22_M);
  case Intrinsic::riscv_esp_vsub_s32_st_incp:
    return lowerVaddVsubStIncpSat(Op, DAG, Subtarget, MVT::v4i32,
                                  RISCVISD::ESP_VSUB_S32_ST_INCP_PIE22_M);
  case Intrinsic::riscv_esp_vsub_u8:
    return lowerVaddVsubSatBasic(Op, DAG, Subtarget, MVT::v16i8,
                                 RISCVISD::ESP_VSUB_U8_PIE22);
  case Intrinsic::riscv_esp_vsub_u8_ld_incp:
    return lowerVaddVsubLdIncpSat(Op, DAG, Subtarget, MVT::v16i8,
                                  RISCVISD::ESP_VSUB_U8_LD_INCP_PIE22_M);
  case Intrinsic::riscv_esp_vsub_u8_st_incp:
    return lowerVaddVsubStIncpSat(Op, DAG, Subtarget, MVT::v16i8,
                                  RISCVISD::ESP_VSUB_U8_ST_INCP_PIE22_M);
  case Intrinsic::riscv_esp_vsub_u16:
    return lowerVaddVsubSatBasic(Op, DAG, Subtarget, MVT::v8i16,
                                 RISCVISD::ESP_VSUB_U16_PIE22);
  case Intrinsic::riscv_esp_vsub_u16_ld_incp:
    return lowerVaddVsubLdIncpSat(Op, DAG, Subtarget, MVT::v8i16,
                                  RISCVISD::ESP_VSUB_U16_LD_INCP_PIE22_M);
  case Intrinsic::riscv_esp_vsub_u16_st_incp:
    return lowerVaddVsubStIncpSat(Op, DAG, Subtarget, MVT::v8i16,
                                  RISCVISD::ESP_VSUB_U16_ST_INCP_PIE22_M);
  case Intrinsic::riscv_esp_vsub_u32:
    return lowerVaddVsubSatBasic(Op, DAG, Subtarget, MVT::v4i32,
                                 RISCVISD::ESP_VSUB_U32_PIE22);
  case Intrinsic::riscv_esp_vsub_u32_ld_incp:
    return lowerVaddVsubLdIncpSat(Op, DAG, Subtarget, MVT::v4i32,
                                  RISCVISD::ESP_VSUB_U32_LD_INCP_PIE22_M);
  case Intrinsic::riscv_esp_vsub_u32_st_incp:
    return lowerVaddVsubStIncpSat(Op, DAG, Subtarget, MVT::v4i32,
                                  RISCVISD::ESP_VSUB_U32_ST_INCP_PIE22_M);
  case Intrinsic::riscv_esp_vsadds_s8:
    return lowerVsaddsVssubsSatBasic(Op, DAG, Subtarget, MVT::v16i8,
                                     RISCVISD::ESP_VSADDS_S8_PIE22);
  case Intrinsic::riscv_esp_vsadds_s16:
    return lowerVsaddsVssubsSatBasic(Op, DAG, Subtarget, MVT::v8i16,
                                     RISCVISD::ESP_VSADDS_S16_PIE22);
  case Intrinsic::riscv_esp_vsadds_u8:
    return lowerVsaddsVssubsSatBasic(Op, DAG, Subtarget, MVT::v16i8,
                                     RISCVISD::ESP_VSADDS_U8_PIE22);
  case Intrinsic::riscv_esp_vsadds_u16:
    return lowerVsaddsVssubsSatBasic(Op, DAG, Subtarget, MVT::v8i16,
                                     RISCVISD::ESP_VSADDS_U16_PIE22);
  case Intrinsic::riscv_esp_vssubs_s8:
    return lowerVsaddsVssubsSatBasic(Op, DAG, Subtarget, MVT::v16i8,
                                     RISCVISD::ESP_VSSUBS_S8_PIE22);
  case Intrinsic::riscv_esp_vssubs_s16:
    return lowerVsaddsVssubsSatBasic(Op, DAG, Subtarget, MVT::v8i16,
                                     RISCVISD::ESP_VSSUBS_S16_PIE22);
  case Intrinsic::riscv_esp_vssubs_u8:
    return lowerVsaddsVssubsSatBasic(Op, DAG, Subtarget, MVT::v16i8,
                                     RISCVISD::ESP_VSSUBS_U8_PIE22);
  case Intrinsic::riscv_esp_vssubs_u16:
    return lowerVsaddsVssubsSatBasic(Op, DAG, Subtarget, MVT::v8i16,
                                     RISCVISD::ESP_VSSUBS_U16_PIE22);
  case Intrinsic::riscv_esp_srs_u_xacc:
    return lowerSrsXacc(Op, DAG, Subtarget, RISCVISD::ESP_SRS_U_XACC_M,
                        RISCVISD::ESP_SRS_U_XACC_PIE22);
  case Intrinsic::riscv_esp_srcxxp_2q: {
    // ESP.SRCXXP.2Q - Shift Right Concatenated with pointer update
    // Intrinsic: (qy, qw, ptr, offset) -> {qy_new, qw_new, ptr_new}
    // SDNode: ESP_SRCXXP_2Q_M (qy, qw, rs1, rs2) -> (qyr, qwr, rs1r)
    // Explicit state passing: All register updates are visible in IR through
    // return values This allows optimization without IntrHasSideEffects while
    // preventing dead code elimination
    SDLoc DL(Op);
    SDValue QY = Op.getOperand(1);     // v16i8
    SDValue QW = Op.getOperand(2);     // v16i8
    SDValue Ptr = Op.getOperand(3);    // i32 pointer
    SDValue Offset = Op.getOperand(4); // i32 offset

    // Create ESP_SRCXXP_2Q_M SDNode
    // SDNode outputs: (v16i8, v16i8, ptr) - updated qy, qw, and pointer
    // SDNode inputs: (v16i8, v16i8, ptr, i32) - qy, qw, rs1, rs2
    EVT VecVT = MVT::v16i8;
    EVT PtrVT = Ptr.getValueType();
    SDVTList VTList = DAG.getVTList(VecVT, VecVT, PtrVT);
    SDValue Ops[] = {QY, QW, Ptr, Offset};
    SDValue Inst = DAG.getNode(RISCVISD::ESP_SRCXXP_2Q_M, DL, VTList, Ops);

    // Return merge values: {qy_new, qw_new, ptr_new}
    // Order matches intrinsic return type: [llvm_v16i8_ty, llvm_v16i8_ty,
    // llvm_ptr_ty]
    return DAG.getMergeValues(
        {Inst.getValue(0), Inst.getValue(1), Inst.getValue(2)}, DL);
  }
  case Intrinsic::riscv_esp_slcxxp_2q: {
    // ESP.SLCXXP.2Q - Shift Left Concatenated with pointer update
    // Intrinsic: (qy, qw, ptr, offset) -> {qy_new, qw_new, ptr_new}
    // SDNode: ESP_SLCXXP_2Q_M (qy, qw, rs1, rs2) -> (qyr, qwr, rs1r)
    // Explicit state passing: All register updates are visible in IR through
    // return values This allows optimization without IntrHasSideEffects while
    // preventing dead code elimination
    SDLoc DL(Op);
    SDValue QY = Op.getOperand(1);     // v16i8
    SDValue QW = Op.getOperand(2);     // v16i8
    SDValue Ptr = Op.getOperand(3);    // i32 pointer
    SDValue Offset = Op.getOperand(4); // i32 offset

    // Create ESP_SLCXXP_2Q_M SDNode
    // SDNode outputs: (v16i8, v16i8, ptr) - updated qy, qw, and pointer
    // SDNode inputs: (v16i8, v16i8, ptr, i32) - qy, qw, rs1, rs2
    EVT VecVT = MVT::v16i8;
    EVT PtrVT = Ptr.getValueType();
    SDVTList VTList = DAG.getVTList(VecVT, VecVT, PtrVT);
    SDValue Ops[] = {QY, QW, Ptr, Offset};
    SDValue Inst = DAG.getNode(RISCVISD::ESP_SLCXXP_2Q_M, DL, VTList, Ops);

    // Return merge values: {qy_new, qw_new, ptr_new}
    // Order matches intrinsic return type: [llvm_v16i8_ty, llvm_v16i8_ty,
    // llvm_ptr_ty]
    return DAG.getMergeValues(
        {Inst.getValue(0), Inst.getValue(1), Inst.getValue(2)}, DL);
  }
  case Intrinsic::riscv_esp_cmul_s16:
    return lowerCmulBasic(Op, DAG, Subtarget, MVT::v8i16,
                          RISCVISD::ESP_CMUL_S16_M,
                          RISCVISD::ESP_CMUL_S16_M_PIE22);
  case Intrinsic::riscv_esp_cmul_s8:
    return lowerCmulBasic(Op, DAG, Subtarget, MVT::v16i8,
                          RISCVISD::ESP_CMUL_S8_M,
                          RISCVISD::ESP_CMUL_S8_M_PIE22);
  case Intrinsic::riscv_esp_vprelu_s16:
    return lowerVpreluBasic(Op, DAG, Subtarget, MVT::v8i16,
                            RISCVISD::ESP_VPRELU_S16_M_PIE22);
  case Intrinsic::riscv_esp_vprelu_s8:
    return lowerVpreluBasic(Op, DAG, Subtarget, MVT::v16i8,
                            RISCVISD::ESP_VPRELU_S8_M_PIE22);
  case Intrinsic::riscv_esp_vrelu_s16:
    return lowerVreluBasic(Op, DAG, Subtarget, MVT::v8i16,
                           RISCVISD::ESP_VRELU_S16_M_PIE22);
  case Intrinsic::riscv_esp_vrelu_s8:
    return lowerVreluBasic(Op, DAG, Subtarget, MVT::v16i8,
                           RISCVISD::ESP_VRELU_S8_M_PIE22);
  case Intrinsic::riscv_esp_vsld_8:
    return lowerVsldVsrdBasic(Op, DAG, Subtarget, MVT::v16i8,
                              RISCVISD::ESP_VSLD_8_M_PIE22);
  case Intrinsic::riscv_esp_vsld_16:
    return lowerVsldVsrdBasic(Op, DAG, Subtarget, MVT::v8i16,
                              RISCVISD::ESP_VSLD_16_M_PIE22);
  case Intrinsic::riscv_esp_vsld_32:
    return lowerVsldVsrdBasic(Op, DAG, Subtarget, MVT::v4i32,
                              RISCVISD::ESP_VSLD_32_M_PIE22);
  case Intrinsic::riscv_esp_vsrd_8:
    return lowerVsldVsrdBasic(Op, DAG, Subtarget, MVT::v16i8,
                              RISCVISD::ESP_VSRD_8_M_PIE22);
  case Intrinsic::riscv_esp_vsrd_16:
    return lowerVsldVsrdBasic(Op, DAG, Subtarget, MVT::v8i16,
                              RISCVISD::ESP_VSRD_16_M_PIE22);
  case Intrinsic::riscv_esp_vsrd_32:
    return lowerVsldVsrdBasic(Op, DAG, Subtarget, MVT::v4i32,
                              RISCVISD::ESP_VSRD_32_M_PIE22);
  case Intrinsic::riscv_esp_vsr_s32:
    return lowerVsrBasic(Op, DAG, Subtarget, MVT::v4i32,
                         RISCVISD::ESP_VSR_S32_M_PIE22);
  case Intrinsic::riscv_esp_vsr_u32:
    return lowerVsrBasic(Op, DAG, Subtarget, MVT::v4i32,
                         RISCVISD::ESP_VSR_U32_M_PIE22);
  case Intrinsic::riscv_esp_vsl_32:
    return lowerVsl32Basic(Op, DAG, Subtarget, MVT::v4i32,
                           RISCVISD::ESP_VSL_32_M_PIE22);
  case Intrinsic::riscv_esp_srcmb_s16_qacc:
    return lowerSrcmbSQacc(Op, DAG, Subtarget, MVT::v8i16,
                           RISCVISD::ESP_SRCMB_S16_QACC_M,
                           RISCVISD::ESP_SRCMB_S16_QACC_PIE22, 5);
  case Intrinsic::riscv_esp_srcmb_s8_qacc:
    return lowerSrcmbSQacc(Op, DAG, Subtarget, MVT::v16i8,
                           RISCVISD::ESP_SRCMB_S8_QACC_M,
                           RISCVISD::ESP_SRCMB_S8_QACC_PIE22, 5);
  case Intrinsic::riscv_esp_srcmb_s16_q_qacc:
    return lowerSrcmbSQacc(Op, DAG, Subtarget, MVT::v8i16,
                           RISCVISD::ESP_SRCMB_S16_Q_QACC_M,
                           RISCVISD::ESP_SRCMB_S16_Q_QACC_PIE22, 5);
  case Intrinsic::riscv_esp_srcmb_s8_q_qacc:
    return lowerSrcmbSQacc(Op, DAG, Subtarget, MVT::v16i8,
                           RISCVISD::ESP_SRCMB_S8_Q_QACC_M,
                           RISCVISD::ESP_SRCMB_S8_Q_QACC_PIE22, 5);
  case Intrinsic::riscv_esp_srcmb_u16_qacc:
    return lowerSrcmbUQacc(Op, DAG, Subtarget, MVT::v8i16,
                           RISCVISD::ESP_SRCMB_U16_QACC_M,
                           RISCVISD::ESP_SRCMB_U16_QACC_PIE22);
  case Intrinsic::riscv_esp_srcmb_u8_qacc:
    return lowerSrcmbUQacc(Op, DAG, Subtarget, MVT::v16i8,
                           RISCVISD::ESP_SRCMB_U8_QACC_M,
                           RISCVISD::ESP_SRCMB_U8_QACC_PIE22);
  case Intrinsic::riscv_esp_srcmb_u16_q_qacc:
    return lowerSrcmbUQQacc(Op, DAG, Subtarget, MVT::v8i16,
                            RISCVISD::ESP_SRCMB_U16_Q_QACC_M,
                            RISCVISD::ESP_SRCMB_U16_Q_QACC_PIE22);
  case Intrinsic::riscv_esp_srcmb_u8_q_qacc:
    return lowerSrcmbUQQacc(Op, DAG, Subtarget, MVT::v16i8,
                            RISCVISD::ESP_SRCMB_U8_Q_QACC_M,
                            RISCVISD::ESP_SRCMB_U8_Q_QACC_PIE22);
  default:
    return SDValue();
  }
}

// VMULAS QACC LD IP Lowering
// VMULAS QACC LD XP Lowering
// VMULAS QACC ST IP Lowering
// VMULAS QACC ST XP Lowering
// VMULAS QACC LDBC INCP Lowering
// Combine two 64-bit QR halves into one 128-bit QR via INSERT_SUBREG.
static SDValue combineQR64Halves(SDLoc DL, MVT VT, SDValue Lo, SDValue Hi,
                                 SelectionDAG &DAG) {
  SDValue Undef = DAG.getUNDEF(VT);
  SDValue Vec = DAG.getTargetInsertSubreg(RISCV::sub_qr_64, DL, VT, Undef, Lo);
  return DAG.getTargetInsertSubreg(RISCV::sub_qr_64_hi, DL, VT, Vec, Hi);
}

static bool isQR64ConcatShuffleMask(ArrayRef<int> Mask, unsigned HalfSize,
                                    unsigned V1Size) {
  for (unsigned I = 0; I < HalfSize; ++I) {
    if (Mask[I] != (int)I && Mask[I] != -1)
      return false;
  }
  for (unsigned I = HalfSize; I < HalfSize * 2; ++I) {
    int MaskIdx = Mask[I];
    if (MaskIdx == -1)
      continue;
    if (MaskIdx != (int)(V1Size + (I - HalfSize)))
      return false;
  }
  return true;
}

SDValue lowerESPVConcatVectors(SDValue Op, SelectionDAG &DAG,
                               const RISCVSubtarget &Subtarget) {
  if (!Subtarget.hasESPVTargetLowering())
    return SDValue();
  if (!Op.getSimpleValueType().isFixedLengthVector() ||
      Op.getNumOperands() != 2)
    return SDValue();

  MVT VT = Op.getSimpleValueType();
  SDValue Lo = Op.getOperand(0);
  SDValue Hi = Op.getOperand(1);
  MVT LoVT = Lo.getSimpleValueType();
  MVT HiVT = Hi.getSimpleValueType();
  if (LoVT != HiVT)
    return SDValue();
  if (VT != MVT::getVectorVT(LoVT.getVectorElementType(),
                             LoVT.getVectorNumElements() * 2))
    return SDValue();

  SDLoc DL(Op);
  return combineQR64Halves(DL, VT, Lo, Hi, DAG);
}

SDValue lowerESPVExtractSubvector(SDValue Op, SelectionDAG &DAG,
                                  const RISCVSubtarget &Subtarget) {
  if (!Subtarget.hasESPVTargetLowering())
    return SDValue();

  SDValue Vec = Op.getOperand(0);
  MVT SubVecVT = Op.getSimpleValueType();
  MVT VecVT = Vec.getSimpleValueType();
  if (!VecVT.isFixedLengthVector() || !SubVecVT.isFixedLengthVector())
    return SDValue();

  unsigned OrigIdx = Op.getConstantOperandVal(1);
  SDLoc DL(Op);

  // Defer wide QACC extractions to type legalizer + instruction selection.
  if ((VecVT == MVT::v64i8 && SubVecVT == MVT::v32i8) ||
      (VecVT == MVT::v32i8 && SubVecVT == MVT::v16i8))
    return Op;

  if (VecVT.getVectorElementType() == SubVecVT.getVectorElementType() &&
      VecVT.getVectorNumElements() == SubVecVT.getVectorNumElements() * 2) {
    if (auto SubIdx = getQR64SubRegIdxForExtractIndex(
            OrigIdx, VecVT.getVectorNumElements()))
      return DAG.getTargetExtractSubreg(*SubIdx, DL, SubVecVT, Vec);
  }
  return SDValue();
}

// ESPV: scalarize mask ext/trunc lane-wise (no RVV vector length).
SDValue lowerESPVVectorMaskExt(SDValue Op, SelectionDAG &DAG,
                               const RISCVSubtarget &Subtarget,
                               int64_t ExtTrueVal) {
  SDLoc DL(Op);
  MVT VecVT = Op.getSimpleValueType();
  SDValue Src = Op.getOperand(0);
  assert(Subtarget.hasESPVTargetLowering() && VecVT.isFixedLengthVector() &&
         "Unexpected scalable mask ext on ESPV");
  MVT DstEltVT = VecVT.getVectorElementType();
  unsigned NumElts = VecVT.getVectorNumElements();
  SmallVector<SDValue, 16> Elts;
  Elts.reserve(NumElts);
  SDValue TrueVal = DAG.getSignedConstant(ExtTrueVal, DL, DstEltVT);
  SDValue ZeroVal = DAG.getConstant(0, DL, DstEltVT);
  for (unsigned I = 0; I != NumElts; ++I) {
    SDValue EltI1 = DAG.getExtractVectorElt(DL, MVT::i1, Src, I);
    SDValue Elt = DAG.getSelect(DL, DstEltVT, EltI1, TrueVal, ZeroVal);
    Elts.push_back(Elt);
  }
  return DAG.getBuildVector(VecVT, DL, Elts);
}

SDValue lowerESPVVectorMaskTrunc(SDValue Op, SelectionDAG &DAG,
                                 const RISCVSubtarget &Subtarget) {
  SDLoc DL(Op);
  EVT MaskVT = Op.getValueType();
  SDValue Src = Op.getOperand(0);
  MVT VecVT = Src.getSimpleValueType();
  assert(Subtarget.hasESPVTargetLowering() &&
         Op.getOpcode() != ISD::VP_TRUNCATE &&
         "Unexpected VP truncate for ESPV mask lowering");
  MVT SrcEltVT = VecVT.getVectorElementType();
  unsigned NumElts = MaskVT.getVectorNumElements();
  SmallVector<SDValue, 16> Elts;
  Elts.reserve(NumElts);
  for (unsigned I = 0; I != NumElts; ++I) {
    SDValue Elt = DAG.getExtractVectorElt(DL, SrcEltVT, Src, I);
    Elt = DAG.getNode(ISD::AND, DL, SrcEltVT, Elt,
                      DAG.getConstant(1, DL, SrcEltVT));
    Elts.push_back(DAG.getNode(ISD::TRUNCATE, DL, MVT::i1, Elt));
  }
  return DAG.getBuildVector(MaskVT.getSimpleVT(), DL, Elts);
}

// Main ESP vector shuffle lowering function

SDValue lowerESPVIntrinsicVoid(SDValue Op, SelectionDAG &DAG,
                               const RISCVSubtarget &Subtarget) {
  if (!Subtarget.hasESPVTargetLowering())
    return SDValue();

  unsigned IntNo = Op.getConstantOperandVal(1);
  SDLoc DL(Op);

  switch (IntNo) {
  case Intrinsic::riscv_esp_movx_w_cfg: {
    // PIE 2.1: TableGen Pat on ESP_MOVX_W_CFG. PIE 2.2: mask then MI here
    // (Pat cannot fold andi for arbitrary non-constant operands).
    if (!Subtarget.useESPV2P2Instructions())
      return SDValue();
    SDValue Chain = Op.getOperand(0);
    SDValue Val =
        lowerEspMovxCfgWriteValue(Op.getOperand(2), DAG, DL, Subtarget);
    SDVTList VTs = DAG.getVTList(MVT::Other);
    SmallVector<SDValue, 2> Ops = {Val, Chain};
    MachineSDNode *Inst =
        DAG.getMachineNode(RISCV::ESP_MOVX_W_CFG_2P2, DL, VTs, Ops);
    return SDValue(Inst, 0);
  }
  default:
    return SDValue();
  }
}

SDValue lowerESPVectorShuffle(SDValue Op, SelectionDAG &DAG,
                              const RISCVSubtarget &Subtarget) {
  if (!Subtarget.hasESPVTargetLowering())
    return SDValue();

  SDValue V1 = Op.getOperand(0);
  SDValue V2 = Op.getOperand(1);
  SDLoc DL(Op);
  MVT VT = Op.getSimpleValueType();
  ShuffleVectorSDNode *SVN = cast<ShuffleVectorSDNode>(Op.getNode());
  ArrayRef<int> Mask = SVN->getMask();

  // Handle direct concatenation: two 64-bit QR halves -> 128-bit QR.
  MVT V1VT = V1.getSimpleValueType();
  MVT V2VT = V2.getSimpleValueType();
  if (V1VT == V2VT && VT == MVT::getVectorVT(V1VT.getVectorElementType(),
                                             V1VT.getVectorNumElements() * 2)) {
    unsigned HalfSize = V1VT.getVectorNumElements();
    if (isQR64ConcatShuffleMask(Mask, HalfSize, HalfSize))
      return combineQR64Halves(DL, VT, V1, V2, DAG);
  }

  // Handle simple extract patterns: extract contiguous elements from a vector
  // This converts shufflevector to EXTRACT_SUBVECTOR for better type
  // legalization
  if (V2.isUndef() || (V2.getOpcode() == ISD::UNDEF)) {
    MVT InVT = V1.getSimpleValueType();
    unsigned InNumElts = InVT.getVectorNumElements();
    unsigned OutNumElts = VT.getVectorNumElements();

    // Check if this is a simple extract: contiguous elements from the input
    // Handle cases where OutNumElts divides InNumElts (e.g., v64i8 -> v16i8,
    // v32i8 -> v16i8)
    if (InNumElts % OutNumElts == 0 && InNumElts > OutNumElts) {
      // Check if mask is [N, N+1, N+2, ...] where N is a valid start index
      bool IsValidExtract = true;
      unsigned StartIdx = Mask[0];

      // Verify all mask indices are contiguous starting from StartIdx
      for (unsigned I = 0; I < OutNumElts; ++I) {
        if (Mask[I] != (int)(StartIdx + I) || Mask[I] >= (int)InNumElts) {
          IsValidExtract = false;
          break;
        }
      }

      if (IsValidExtract) {
        // Convert shufflevector to EXTRACT_SUBVECTOR for better type
        // legalization This handles:
        // - v64i8 -> v16i8 extraction (QACC -> QACC_L/QACC_H subregisters)
        // - v32i8 -> v16i8 extraction (QACC_L/QACC_H subregisters)
        return DAG.getNode(ISD::EXTRACT_SUBVECTOR, DL, VT, V1,
                           DAG.getVectorIdxConstant(StartIdx, DL));
      }
    }
  }

  // For other patterns, return SDValue() to fall back to default handling
  return SDValue();
}

static SDValue LowerVMULASQACCLDIPLegacy(SDValue Op, SelectionDAG &DAG,
                                         unsigned ISDOpcode) {
  SDLoc DL(Op);
  SDValue Chain = Op.getOperand(0);
  SDValue V0In = Op.getOperand(2);
  SDValue V1In = Op.getOperand(3);
  SDValue V2In = Op.getOperand(4);
  SDValue V3In = Op.getOperand(5);
  SDValue QX = Op.getOperand(6);
  SDValue QY = Op.getOperand(7);
  SDValue Ptr = Op.getOperand(8);
  SDValue Offset = Op.getOperand(9);

  EVT PtrVT = Ptr.getValueType();
  EVT MemVT = MVT::v16i8;
  SmallVector<EVT, 7> VTList = {MVT::v16i8, PtrVT,      MVT::v16i8, MVT::v16i8,
                                MVT::v16i8, MVT::v16i8, MVT::Other};
  SDVTList VTs = DAG.getVTList(VTList);
  SDValue Ops[] = {Chain, V0In, V1In, V2In, V3In, QX, QY, Ptr, Offset};
  auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
  MachineMemOperand *MMO = MemIntr->getMemOperand();
  SDValue Node = DAG.getMemIntrinsicNode(ISDOpcode, DL, VTs, Ops, MemVT, MMO);
  SDValue Qu = Node.getValue(0);
  SDValue PtrOut = Node.getValue(1);
  SDValue V0 = Node.getValue(2);
  SDValue V1 = Node.getValue(3);
  SDValue V2 = Node.getValue(4);
  SDValue V3 = Node.getValue(5);
  Chain = Node.getValue(6);
  return DAG.getMergeValues({PtrOut, Qu, V0, V1, V2, V3, Chain}, DL);
}

static SDValue lowerVmulasQaccCompute(SDValue Op, SelectionDAG &DAG,
                                      const RISCVSubtarget &Subtarget,
                                      unsigned ISD21, unsigned ISD22) {
  SDLoc DL(Op);
  SDValue V0In = Op.getOperand(1);
  SDValue V1In = Op.getOperand(2);
  SDValue V2In = Op.getOperand(3);
  SDValue V3In = Op.getOperand(4);
  SDValue QX = Op.getOperand(5);
  SDValue QY = Op.getOperand(6);
  SDValue SAT = Op.getOperand(7);
  SmallVector<EVT, 4> VTList = {MVT::v16i8, MVT::v16i8, MVT::v16i8, MVT::v16i8};
  SDVTList VTs = DAG.getVTList(VTList);
  if (Subtarget.useESPV2P2Instructions()) {
    SDValue Ops[] = {
        V0In, V1In, V2In, V3In, QX, QY, lowerCmulTargetImm(DAG, DL, SAT)};
    SDValue Node = DAG.getNode(ISD22, DL, VTs, Ops);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1),
                               Node.getValue(2), Node.getValue(3)},
                              DL);
  }
  if (Subtarget.hasVendorXespv2p1()) {
    diagnoseESPV21Sat(DAG, SAT);
    SDValue Ops[] = {V0In, V1In, V2In, V3In, QX, QY};
    SDValue Node = DAG.getNode(ISD21, DL, VTs, Ops);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1),
                               Node.getValue(2), Node.getValue(3)},
                              DL);
  }
  return SDValue();
}

static SDValue lowerVmulasXaccCompute(SDValue Op, SelectionDAG &DAG,
                                      const RISCVSubtarget &Subtarget,
                                      unsigned ISD21, unsigned ISD22) {
  SDLoc DL(Op);
  SDValue XLow = Op.getOperand(1);
  SDValue XHigh = Op.getOperand(2);
  SDValue QX = Op.getOperand(3);
  SDValue QY = Op.getOperand(4);
  SDValue SAT = Op.getOperand(5);
  SDVTList VTs = DAG.getVTList(MVT::i32, MVT::i32);
  if (Subtarget.useESPV2P2Instructions()) {
    SDValue Ops[] = {XLow, XHigh, QX, QY, lowerCmulTargetImm(DAG, DL, SAT)};
    SDValue Node = DAG.getNode(ISD22, DL, VTs, Ops);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1)}, DL);
  }
  if (Subtarget.hasVendorXespv2p1()) {
    diagnoseESPV21Sat(DAG, SAT);
    SDValue Ops[] = {XLow, XHigh, QX, QY};
    SDValue Node = DAG.getNode(ISD21, DL, VTs, Ops);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1)}, DL);
  }
  return SDValue();
}

static SDValue LowerVMULASQACCLDIP(SDValue Op, SelectionDAG &DAG,
                                   const RISCVSubtarget &Subtarget,
                                   unsigned ISD21, unsigned ISD22) {
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
  SDValue SAT = Op.getOperand(10);

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
  if (Subtarget.useESPV2P2Instructions()) {
    SDValue Ops[] = {
        Chain, V0In, V1In, V2In,   V3In,
        QX,    QY,   Ptr,  Offset, lowerCmulTargetImm(DAG, DL, SAT)};
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(ISD22, DL, VTs, Ops, MemVT, MMO);
    SDValue Qu = Node.getValue(0);
    SDValue PtrOut = Node.getValue(1);
    SDValue V0 = Node.getValue(2);
    SDValue V1 = Node.getValue(3);
    SDValue V2 = Node.getValue(4);
    SDValue V3 = Node.getValue(5);
    Chain = Node.getValue(6);
    return DAG.getMergeValues({PtrOut, Qu, V0, V1, V2, V3, Chain}, DL);
  }
  if (Subtarget.hasVendorXespv2p1()) {
    diagnoseESPV21Sat(DAG, SAT);
    SDValue Ops[] = {Chain, V0In, V1In, V2In, V3In, QX, QY, Ptr, Offset};
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(ISD21, DL, VTs, Ops, MemVT, MMO);
    SDValue Qu = Node.getValue(0);
    SDValue PtrOut = Node.getValue(1);
    SDValue V0 = Node.getValue(2);
    SDValue V1 = Node.getValue(3);
    SDValue V2 = Node.getValue(4);
    SDValue V3 = Node.getValue(5);
    Chain = Node.getValue(6);
    return DAG.getMergeValues({PtrOut, Qu, V0, V1, V2, V3, Chain}, DL);
  }
  return SDValue();
}

static SDValue LowerVMULASQACCLDXP(SDValue Op, SelectionDAG &DAG,
                                   const RISCVSubtarget &Subtarget,
                                   unsigned ISD21, unsigned ISD22) {
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
  SDValue SAT = Op.getOperand(10);

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
  if (Subtarget.useESPV2P2Instructions()) {
    SDValue Ops[] = {Chain, V0In, V1In, V2In, V3In,
                     QX,    QY,   Ptr,  Rs2,  lowerCmulTargetImm(DAG, DL, SAT)};
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(ISD22, DL, VTs, Ops, MemVT, MMO);
    SDValue Qu = Node.getValue(0);
    SDValue PtrOut = Node.getValue(1);
    SDValue V0 = Node.getValue(2);
    SDValue V1 = Node.getValue(3);
    SDValue V2 = Node.getValue(4);
    SDValue V3 = Node.getValue(5);
    Chain = Node.getValue(6);
    return DAG.getMergeValues({PtrOut, Qu, V0, V1, V2, V3, Chain}, DL);
  }
  if (Subtarget.hasVendorXespv2p1()) {
    diagnoseESPV21Sat(DAG, SAT);
    SDValue Ops[] = {Chain, V0In, V1In, V2In, V3In, QX, QY, Ptr, Rs2};
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(ISD21, DL, VTs, Ops, MemVT, MMO);
    SDValue Qu = Node.getValue(0);
    SDValue PtrOut = Node.getValue(1);
    SDValue V0 = Node.getValue(2);
    SDValue V1 = Node.getValue(3);
    SDValue V2 = Node.getValue(4);
    SDValue V3 = Node.getValue(5);
    Chain = Node.getValue(6);
    return DAG.getMergeValues({PtrOut, Qu, V0, V1, V2, V3, Chain}, DL);
  }
  return SDValue();
}

static SDValue LowerVMULASQACCSTIP(SDValue Op, SelectionDAG &DAG,
                                   const RISCVSubtarget &Subtarget,
                                   unsigned ISD21, unsigned ISD22) {
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
  SDValue SAT = Op.getOperand(11);

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
  if (Subtarget.useESPV2P2Instructions()) {
    SDValue Ops[] = {Chain,
                     V0In,
                     V1In,
                     V2In,
                     V3In,
                     QU,
                     QX,
                     QY,
                     Ptr,
                     Offset,
                     lowerCmulTargetImm(DAG, DL, SAT)};
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(ISD22, DL, VTs, Ops, MemVT, MMO);
    SDValue PtrOut = Node.getValue(0);
    SDValue V0 = Node.getValue(1);
    SDValue V1 = Node.getValue(2);
    SDValue V2 = Node.getValue(3);
    SDValue V3 = Node.getValue(4);
    Chain = Node.getValue(5);
    return DAG.getMergeValues({PtrOut, V0, V1, V2, V3, Chain}, DL);
  }
  if (Subtarget.hasVendorXespv2p1()) {
    diagnoseESPV21Sat(DAG, SAT);
    SDValue Ops[] = {Chain, V0In, V1In, V2In, V3In, QU, QX, QY, Ptr, Offset};
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(ISD21, DL, VTs, Ops, MemVT, MMO);
    SDValue PtrOut = Node.getValue(0);
    SDValue V0 = Node.getValue(1);
    SDValue V1 = Node.getValue(2);
    SDValue V2 = Node.getValue(3);
    SDValue V3 = Node.getValue(4);
    Chain = Node.getValue(5);
    return DAG.getMergeValues({PtrOut, V0, V1, V2, V3, Chain}, DL);
  }
  return SDValue();
}

static SDValue LowerVMULASQACCSTXP(SDValue Op, SelectionDAG &DAG,
                                   const RISCVSubtarget &Subtarget,
                                   unsigned ISD21, unsigned ISD22) {
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
  SDValue SAT = Op.getOperand(11);

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
  if (Subtarget.useESPV2P2Instructions()) {
    SDValue Ops[] = {Chain,
                     V0In,
                     V1In,
                     V2In,
                     V3In,
                     QU,
                     QX,
                     QY,
                     Ptr,
                     Rs2,
                     lowerCmulTargetImm(DAG, DL, SAT)};
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(ISD22, DL, VTs, Ops, MemVT, MMO);
    SDValue PtrOut = Node.getValue(0);
    SDValue V0 = Node.getValue(1);
    SDValue V1 = Node.getValue(2);
    SDValue V2 = Node.getValue(3);
    SDValue V3 = Node.getValue(4);
    Chain = Node.getValue(5);
    return DAG.getMergeValues({PtrOut, V0, V1, V2, V3, Chain}, DL);
  }
  if (Subtarget.hasVendorXespv2p1()) {
    diagnoseESPV21Sat(DAG, SAT);
    SDValue Ops[] = {Chain, V0In, V1In, V2In, V3In, QU, QX, QY, Ptr, Rs2};
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(ISD21, DL, VTs, Ops, MemVT, MMO);
    SDValue PtrOut = Node.getValue(0);
    SDValue V0 = Node.getValue(1);
    SDValue V1 = Node.getValue(2);
    SDValue V2 = Node.getValue(3);
    SDValue V3 = Node.getValue(4);
    Chain = Node.getValue(5);
    return DAG.getMergeValues({PtrOut, V0, V1, V2, V3, Chain}, DL);
  }
  return SDValue();
}

static SDValue LowerVMULASQACCLDBCINCP(SDValue Op, SelectionDAG &DAG,
                                       const RISCVSubtarget &Subtarget,
                                       unsigned ISD21, unsigned ISD22) {
  SDLoc DL(Op);
  SDValue Chain = Op.getOperand(0);
  SDValue V0In = Op.getOperand(2);
  SDValue V1In = Op.getOperand(3);
  SDValue V2In = Op.getOperand(4);
  SDValue V3In = Op.getOperand(5);
  SDValue QX = Op.getOperand(6);
  SDValue QY = Op.getOperand(7);
  SDValue Ptr = Op.getOperand(8);
  SDValue SAT = Op.getOperand(9);

  EVT PtrVT = Ptr.getValueType();
  EVT MemVT = MVT::v16i8;
  SmallVector<EVT, 7> VTList = {MVT::v16i8, PtrVT,      MVT::v16i8, MVT::v16i8,
                                MVT::v16i8, MVT::v16i8, MVT::Other};
  SDVTList VTs = DAG.getVTList(VTList);
  if (Subtarget.useESPV2P2Instructions()) {
    SDValue Ops[] = {Chain, V0In, V1In,
                     V2In,  V3In, QX,
                     QY,    Ptr,  lowerCmulTargetImm(DAG, DL, SAT)};
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(ISD22, DL, VTs, Ops, MemVT, MMO);
    return DAG.getMergeValues(
        {Node.getValue(0), Node.getValue(1), Node.getValue(2), Node.getValue(3),
         Node.getValue(4), Node.getValue(5), Node.getValue(6)},
        DL);
  }
  if (Subtarget.hasVendorXespv2p1()) {
    diagnoseESPV21Sat(DAG, SAT);
    SDValue Ops[] = {Chain, V0In, V1In, V2In, V3In, QX, QY, Ptr};
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(ISD21, DL, VTs, Ops, MemVT, MMO);
    return DAG.getMergeValues(
        {Node.getValue(0), Node.getValue(1), Node.getValue(2), Node.getValue(3),
         Node.getValue(4), Node.getValue(5), Node.getValue(6)},
        DL);
  }
  return SDValue();
}

static SDValue LowerVMULASXACCLDIP(SDValue Op, SelectionDAG &DAG,
                                   const RISCVSubtarget &Subtarget,
                                   unsigned ISD21, unsigned ISD22) {
  SDLoc DL(Op);
  SDValue Chain = Op.getOperand(0);
  SDValue XACCLowIn = Op.getOperand(2);
  SDValue XACCHighIn = Op.getOperand(3);
  SDValue QX = Op.getOperand(4);
  SDValue QY = Op.getOperand(5);
  SDValue Ptr = Op.getOperand(6);
  SDValue Offset = Op.getOperand(7);
  SDValue SAT = Op.getOperand(8);
  EVT PtrVT = Ptr.getValueType();
  EVT MemVT = MVT::v16i8;
  EVT VTsArray[] = {MVT::v16i8, PtrVT,      MVT::i32,
                    MVT::i32,   MVT::Other, MVT::Glue};
  SDVTList VTs = DAG.getVTList(VTsArray);
  if (Subtarget.useESPV2P2Instructions()) {
    SDValue Ops[] = {
        Chain, XACCLowIn, XACCHighIn, QX,
        QY,    Ptr,       Offset,     lowerCmulTargetImm(DAG, DL, SAT)};
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(ISD22, DL, VTs, Ops, MemVT, MMO);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1),
                               Node.getValue(2), Node.getValue(3),
                               Node.getValue(4)},
                              DL);
  }
  if (Subtarget.hasVendorXespv2p1()) {
    diagnoseESPV21Sat(DAG, SAT);
    SDValue Ops[] = {Chain, XACCLowIn, XACCHighIn, QX, QY, Ptr, Offset};
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(ISD21, DL, VTs, Ops, MemVT, MMO);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1),
                               Node.getValue(2), Node.getValue(3),
                               Node.getValue(4)},
                              DL);
  }
  return SDValue();
}

static SDValue LowerVMULASXACCLDXP(SDValue Op, SelectionDAG &DAG,
                                   const RISCVSubtarget &Subtarget,
                                   unsigned ISD21, unsigned ISD22) {
  SDLoc DL(Op);
  SDValue Chain = Op.getOperand(0);
  SDValue XACCLowIn = Op.getOperand(2);
  SDValue XACCHighIn = Op.getOperand(3);
  SDValue QX = Op.getOperand(4);
  SDValue QY = Op.getOperand(5);
  SDValue Ptr = Op.getOperand(6);
  SDValue Rs2 = Op.getOperand(7);
  SDValue SAT = Op.getOperand(8);
  EVT PtrVT = Ptr.getValueType();
  EVT MemVT = MVT::v16i8;
  EVT VTsArray[] = {MVT::v16i8, PtrVT,      MVT::i32,
                    MVT::i32,   MVT::Other, MVT::Glue};
  SDVTList VTs = DAG.getVTList(VTsArray);
  if (Subtarget.useESPV2P2Instructions()) {
    SDValue Ops[] = {
        Chain, XACCLowIn, XACCHighIn, QX,
        QY,    Ptr,       Rs2,        lowerCmulTargetImm(DAG, DL, SAT)};
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(ISD22, DL, VTs, Ops, MemVT, MMO);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1),
                               Node.getValue(2), Node.getValue(3),
                               Node.getValue(4)},
                              DL);
  }
  if (Subtarget.hasVendorXespv2p1()) {
    diagnoseESPV21Sat(DAG, SAT);
    SDValue Ops[] = {Chain, XACCLowIn, XACCHighIn, QX, QY, Ptr, Rs2};
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(ISD21, DL, VTs, Ops, MemVT, MMO);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1),
                               Node.getValue(2), Node.getValue(3),
                               Node.getValue(4)},
                              DL);
  }
  return SDValue();
}

static SDValue LowerVMULASXACCSTIP(SDValue Op, SelectionDAG &DAG,
                                   const RISCVSubtarget &Subtarget,
                                   unsigned ISD21, unsigned ISD22) {
  SDLoc DL(Op);
  SDValue Chain = Op.getOperand(0);
  SDValue XACCLowIn = Op.getOperand(2);
  SDValue XACCHighIn = Op.getOperand(3);
  SDValue QU = Op.getOperand(4);
  SDValue QX = Op.getOperand(5);
  SDValue QY = Op.getOperand(6);
  SDValue Ptr = Op.getOperand(7);
  SDValue Offset = Op.getOperand(8);
  SDValue SAT = Op.getOperand(9);
  EVT PtrVT = Ptr.getValueType();
  EVT MemVT = MVT::v16i8;
  EVT VTsArray[] = {PtrVT, MVT::i32, MVT::i32, MVT::Other, MVT::Glue};
  SDVTList VTs = DAG.getVTList(VTsArray);
  if (Subtarget.useESPV2P2Instructions()) {
    SDValue Ops[] = {Chain, XACCLowIn, XACCHighIn,
                     QU,    QX,        QY,
                     Ptr,   Offset,    lowerCmulTargetImm(DAG, DL, SAT)};
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(ISD22, DL, VTs, Ops, MemVT, MMO);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1),
                               Node.getValue(2), Node.getValue(3)},
                              DL);
  }
  if (Subtarget.hasVendorXespv2p1()) {
    diagnoseESPV21Sat(DAG, SAT);
    SDValue Ops[] = {Chain, XACCLowIn, XACCHighIn, QU, QX, QY, Ptr, Offset};
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(ISD21, DL, VTs, Ops, MemVT, MMO);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1),
                               Node.getValue(2), Node.getValue(3)},
                              DL);
  }
  return SDValue();
}

static SDValue LowerVMULASXACCSTXP(SDValue Op, SelectionDAG &DAG,
                                   const RISCVSubtarget &Subtarget,
                                   unsigned ISD21, unsigned ISD22) {
  SDLoc DL(Op);
  SDValue Chain = Op.getOperand(0);
  SDValue XACCLowIn = Op.getOperand(2);
  SDValue XACCHighIn = Op.getOperand(3);
  SDValue QU = Op.getOperand(4);
  SDValue QX = Op.getOperand(5);
  SDValue QY = Op.getOperand(6);
  SDValue Ptr = Op.getOperand(7);
  SDValue Rs2 = Op.getOperand(8);
  SDValue SAT = Op.getOperand(9);
  EVT PtrVT = Ptr.getValueType();
  EVT MemVT = MVT::v16i8;
  EVT VTsArray[] = {PtrVT, MVT::i32, MVT::i32, MVT::Other, MVT::Glue};
  SDVTList VTs = DAG.getVTList(VTsArray);
  if (Subtarget.useESPV2P2Instructions()) {
    SDValue Ops[] = {Chain, XACCLowIn, XACCHighIn,
                     QU,    QX,        QY,
                     Ptr,   Rs2,       lowerCmulTargetImm(DAG, DL, SAT)};
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(ISD22, DL, VTs, Ops, MemVT, MMO);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1),
                               Node.getValue(2), Node.getValue(3)},
                              DL);
  }
  if (Subtarget.hasVendorXespv2p1()) {
    diagnoseESPV21Sat(DAG, SAT);
    SDValue Ops[] = {Chain, XACCLowIn, XACCHighIn, QU, QX, QY, Ptr, Rs2};
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(ISD21, DL, VTs, Ops, MemVT, MMO);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1),
                               Node.getValue(2), Node.getValue(3)},
                              DL);
  }
  return SDValue();
}

static SDValue LowerVSMULASQACCLDIP(SDValue Op, SelectionDAG &DAG,
                                    const RISCVSubtarget &Subtarget,
                                    unsigned ISD21, unsigned ISD22) {
  SDLoc DL(Op);
  SDValue Chain = Op.getOperand(0);
  SDValue V0In = Op.getOperand(2);
  SDValue V1In = Op.getOperand(3);
  SDValue V2In = Op.getOperand(4);
  SDValue V3In = Op.getOperand(5);
  SDValue QX = Op.getOperand(6);
  SDValue QY = Op.getOperand(7);
  SDValue Ptr = Op.getOperand(8);
  SDValue SEL16 = Op.getOperand(9);
  SDValue SAT = Op.getOperand(10);

  EVT PtrVT = Ptr.getValueType();
  EVT MemVT = MVT::v16i8;
  SmallVector<EVT, 7> VTList = {MVT::v16i8, PtrVT,      MVT::v16i8, MVT::v16i8,
                                MVT::v16i8, MVT::v16i8, MVT::Other};
  SDVTList VTs = DAG.getVTList(VTList);
  if (Subtarget.useESPV2P2Instructions()) {
    SDValue Ops[] = {
        Chain, V0In, V1In, V2In,  V3In,
        QX,    QY,   Ptr,  SEL16, lowerCmulTargetImm(DAG, DL, SAT)};
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(ISD22, DL, VTs, Ops, MemVT, MMO);
    SDValue Qu = Node.getValue(0);
    SDValue PtrOut = Node.getValue(1);
    SDValue V0 = Node.getValue(2);
    SDValue V1 = Node.getValue(3);
    SDValue V2 = Node.getValue(4);
    SDValue V3 = Node.getValue(5);
    Chain = Node.getValue(6);
    return DAG.getMergeValues({PtrOut, Qu, V0, V1, V2, V3, Chain}, DL);
  }
  if (Subtarget.hasVendorXespv2p1()) {
    diagnoseESPV21Sat(DAG, SAT);
    SDValue Ops[] = {Chain, V0In, V1In, V2In, V3In, QX, QY, Ptr, SEL16};
    auto *MemIntr = cast<MemIntrinsicSDNode>(Op.getNode());
    MachineMemOperand *MMO = MemIntr->getMemOperand();
    SDValue Node = DAG.getMemIntrinsicNode(ISD21, DL, VTs, Ops, MemVT, MMO);
    SDValue Qu = Node.getValue(0);
    SDValue PtrOut = Node.getValue(1);
    SDValue V0 = Node.getValue(2);
    SDValue V1 = Node.getValue(3);
    SDValue V2 = Node.getValue(4);
    SDValue V3 = Node.getValue(5);
    Chain = Node.getValue(6);
    return DAG.getMergeValues({PtrOut, Qu, V0, V1, V2, V3, Chain}, DL);
  }
  return SDValue();
}

static SDValue lowerVsmulasQaccCompute(SDValue Op, SelectionDAG &DAG,
                                       const RISCVSubtarget &Subtarget,
                                       unsigned ISD21, unsigned ISD22) {
  SDLoc DL(Op);
  SDValue V0In = Op.getOperand(1);
  SDValue V1In = Op.getOperand(2);
  SDValue V2In = Op.getOperand(3);
  SDValue V3In = Op.getOperand(4);
  SDValue QX = Op.getOperand(5);
  SDValue QY = Op.getOperand(6);
  SDValue SEL16 = Op.getOperand(7);
  SDValue SAT = Op.getOperand(8);
  SmallVector<EVT, 4> VTList = {MVT::v16i8, MVT::v16i8, MVT::v16i8, MVT::v16i8};
  SDVTList VTs = DAG.getVTList(VTList);
  if (Subtarget.useESPV2P2Instructions()) {
    SDValue Ops[] = {V0In, V1In, V2In,  V3In,
                     QX,   QY,   SEL16, lowerCmulTargetImm(DAG, DL, SAT)};
    SDValue Node = DAG.getNode(ISD22, DL, VTs, Ops);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1),
                               Node.getValue(2), Node.getValue(3)},
                              DL);
  }
  if (Subtarget.hasVendorXespv2p1()) {
    diagnoseESPV21Sat(DAG, SAT);
    SDValue Ops[] = {V0In, V1In, V2In, V3In, QX, QY, SEL16};
    SDValue Node = DAG.getNode(ISD21, DL, VTs, Ops);
    return DAG.getMergeValues({Node.getValue(0), Node.getValue(1),
                               Node.getValue(2), Node.getValue(3)},
                              DL);
  }
  return SDValue();
}
} // namespace RISCV
} // namespace llvm
