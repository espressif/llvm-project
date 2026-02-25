//====-- RISCVInstrFormatsESP32P4.td - ESP32P4 Instr Formats -*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file describes the RISC-V ESP32-P4 extension instruction formats.
//
//===----------------------------------------------------------------------===//

class Esp32P4Inst<dag outs, dag ins, string asmstr, list<dag> pattern,
                 InstrItinClass itin = NoItinerary> 
  : Instruction, Requires<[HasVendorXesppie]> {
  let Namespace = "RISCV";
  let DecoderNamespace = "ESP32P4";
  field bits<32> Inst;
  field bits<32> SoftFail = 0;

  let TSFlags{4-0} = 23;
  let Size = 4;

  let OutOperandList = outs;
  let InOperandList  = ins;

  let AsmString   = asmstr;
  let Pattern     = pattern;

  let Itinerary   = itin;
}


// Pseudo instructions
class PseudoESP32P4<dag outs, dag ins, string asmstr, list<dag> pattern>
  : Esp32P4Inst<outs, ins, asmstr, pattern> {
  let isPseudo = 1;
  let isCodeGenOnly = 1;
  let mayLoad = 1;
  let mayStore = 1;
  let hasSideEffects = 1;
}