//===- XtensaCallingConv.td - Xtensa Calling Conventions -*- tablegen ---*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This describes the calling conventions for the Xtensa ABI.
//===----------------------------------------------------------------------===//

class CCIfFeature<string Feature, CCAction A>:
  CCIf<!strconcat("State.getMachineFunction().getSubtarget<XtensaSubtarget>().has", Feature, "()"), A>;

//===----------------------------------------------------------------------===//
// Xtensa return value calling convention
//===----------------------------------------------------------------------===//
def RetCC_Xtensa : CallingConv<[
  CCIfFeature<"Boolean",CCIfType<[v1i1], CCAssignToReg<[B0]>>>,
  CCIfFeature<"Boolean",CCIfType<[v2i1], CCAssignToReg<[B0_B1]>>>,
  CCIfFeature<"Boolean",CCIfType<[v4i1], CCAssignToReg<[B0_B1_B2_B3]>>>,

  // First two return values go in a2, a3, a4, a5
  CCIfType<[i32], CCAssignToReg<[A2, A3, A4, A5]>>,
  CCIfType<[f32], CCAssignToReg<[A2, A3, A4, A5]>>,
  CCIfType<[i64], CCAssignToRegWithShadow<[A2, A4], [A3, A5]>>,
  CCIfFeature<"HIFI3",
    CCIfType<[v4i16, v2i32, v1i64, v1i32], 
    CCAssignToReg<[AED0, AED1, AED2, AED3]>>>
]>;

//===----------------------------------------------------------------------===//
// Callee-saved register lists.
//===----------------------------------------------------------------------===//

def CSR_Xtensa : CalleeSavedRegs<(add A0, A12, A13, A14, A15)>;
def CSRWE_Xtensa : CalleeSavedRegs<(add)> {
  let OtherPreserved = (add A0, SP, A2, A3, A4, A5, A6, A7);
}
//===----------------------------------------------------------------------===//

def RetCCW_Xtensa : CallingConv<[
  CCIfType<[i1, i8, i16], CCPromoteToType<i32>>,
  CCIfFeature<"Boolean",CCIfType<[v1i1], CCAssignToReg<[B0]>>>,
  CCIfType<[f32], CCBitConvertToType<i32>>,

  //First two return values go in a10, a11, a12, a13
  CCIfType<[i32], CCAssignToReg<[A10, A11, A12, A13]>>,
  CCIfType<[f32], CCAssignToReg<[A10, A11, A12, A13]>>,
  CCIfType<[i64], CCAssignToRegWithShadow<[A10, A12], [A11, A13]>>,
  CCIfFeature<"HIFI3",
    CCIfType<[v4i16, v2i32, v1i64, v1i32], 
    CCAssignToReg<[AED0, AED1, AED2, AED3]>>>
]>;
