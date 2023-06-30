//===- XtensaHIFIInstrFormats.td - Instruction formats for Xtensa HIFI -*- tablegen -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains Tablegen instruction formats  for Xtensa HIFI extension
//
//===----------------------------------------------------------------------===//

// Base class for Xtensa 64 bit Format
class XtensaInst64<dag outs, dag ins, string asmstr, list<dag> pattern,
                   InstrItinClass itin = NoItinerary>
      : XtensaInst<8, outs, ins, asmstr, pattern, itin>
{
  bits<64> Inst;
  bits<64> SoftFail = 0;
}

class XtensaInst48<dag outs, dag ins, string asmstr, list<dag> pattern,
                   InstrItinClass itin = NoItinerary>
      : XtensaInst<6, outs, ins, asmstr, pattern, itin>
{
  bits<48> Inst;
  bits<48> SoftFail = 0;
}

class XtensaInst88<dag outs, dag ins, string asmstr, list<dag> pattern,
                   InstrItinClass itin = NoItinerary>
      : XtensaInst<11, outs, ins, asmstr, pattern, itin>
{
  bits<88> Inst;
  bits<88> SoftFail = 0;
}
