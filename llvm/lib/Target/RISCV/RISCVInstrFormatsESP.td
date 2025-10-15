//====-- RISCVInstrFormatsESP.td - ESP32P4 Instr Formats -*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file describes the RISC-V ESP extensions instruction formats.
//
//===----------------------------------------------------------------------===//

class EspBase<dag outs, dag ins, string asmstr, list<dag> pattern, string DecoderNS,
                 InstrItinClass itin = NoItinerary> 
  : Instruction {
  let Namespace = "RISCV";
  let DecoderNamespace = DecoderNS;
  field bits<32> Inst;
  field bits<32> SoftFail = 0;

  let TSFlags{4-0} = 28;
  let Size = 4;

  let OutOperandList = outs;
  let InOperandList  = ins;

  let AsmString   = asmstr;
  let Pattern     = pattern;

  let Itinerary   = itin;
}

class Esp32DSPInst<dag outs, dag ins, string asmstr, list<dag> pattern,
                   InstrItinClass itin = NoItinerary> 
  : EspBase<outs, ins, asmstr, pattern, "ESPDSP", itin>, Requires<[HasVendorXespdsp]>;

class EspVInst<dag outs, dag ins, string asmstr, list<dag> pattern,
                InstrItinClass itin = NoItinerary>
  : EspBase<outs, ins, asmstr, pattern, "ESPAI", itin>, Requires<[HasVendorXespv]>;

class EspLoopInst<dag outs, dag ins, string asmstr, list<dag> pattern,
                InstrItinClass itin = NoItinerary>
  : EspBase<outs, ins, asmstr, pattern, "ESPLOOP", itin>, Requires<[HasVendorXesploop]>;

class EspV_DSPInst<dag outs, dag ins, string asmstr, list<dag> pattern,
                InstrItinClass itin = NoItinerary>
  : EspBase<outs, ins, asmstr, pattern, "ESPAI", itin>, Requires<[HasVendorEspSAT]>;

// Pseudo instructions
class PseudoEsp<dag outs, dag ins, string asmstr, list<dag> pattern>
  : EspBase<outs, ins, asmstr, pattern, ""> {
  let isPseudo = 1;
  let isCodeGenOnly = 1;
  let mayLoad = 1;
  let mayStore = 1;
  let hasSideEffects = 1;
}