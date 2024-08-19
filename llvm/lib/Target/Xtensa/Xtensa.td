//===- Xtensa.td - Describe the Xtensa Target Machine ------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Target-independent interfaces
//===----------------------------------------------------------------------===//

include "llvm/Target/Target.td"

//===----------------------------------------------------------------------===//
// Subtarget Features.
//===----------------------------------------------------------------------===//
def FeatureDensity : SubtargetFeature<"density", "HasDensity", "true",
                    "Enable Density instructions">;
def HasDensity : Predicate<"Subtarget->hasDensity()">,
                     AssemblerPredicate<(all_of FeatureDensity)>;

def FeatureSingleFloat      : SubtargetFeature<"fp", "HasSingleFloat", "true",
                                               "Enable Xtensa Single FP instructions">;
def HasSingleFloat          : Predicate<"Subtarget->hasSingleFloat()">,
                                        AssemblerPredicate<(all_of FeatureSingleFloat)>;

def FeatureWindowed         : SubtargetFeature<"windowed", "HasWindowed", "true",
                                               "Enable Xtensa Windowed Register option">;
def HasWindowed             : Predicate<"Subtarget->hasWindowed()">,
                                         AssemblerPredicate<(all_of FeatureWindowed)>;

def FeatureBoolean          : SubtargetFeature<"bool", "HasBoolean", "true",
                                               "Enable Xtensa Boolean extension">;
def HasBoolean              : Predicate<"Subtarget->hasBoolean()">,
                                         AssemblerPredicate<(all_of FeatureBoolean)>;

def FeatureLoop             : SubtargetFeature<"loop", "HasLoop", "true",
                                               "Enable Xtensa Loop extension">;
def HasLoop                 : Predicate<"Subtarget->hasLoop()">,
                                         AssemblerPredicate<(all_of FeatureLoop)>;

def FeatureSEXT             : SubtargetFeature<"sext", "HasSEXT", "true",
                                              "Enable Xtensa Sign Extend option">;
def HasSEXT                 : Predicate<"Subtarget->hasSEXT()">,
                                         AssemblerPredicate<(all_of FeatureSEXT)>;

def FeatureNSA              : SubtargetFeature<"nsa", "HasNSA", "true",
                                               "Enable Xtensa NSA option">;
def HasNSA                  : Predicate<"Subtarget->hasNSA()">,
                                         AssemblerPredicate<(all_of FeatureNSA)>;

def FeatureMul32            : SubtargetFeature<"mul32", "HasMul32", "true",
                                               "Enable Xtensa Mul32 option">;
def HasMul32                : Predicate<"Subtarget->hasMul32()">,
                                         AssemblerPredicate<(all_of FeatureMul32)>;

def FeatureMul32High        : SubtargetFeature<"mul32high", "HasMul32High", "true",
                                               "Enable Xtensa Mul32High option">;
def HasMul32High            : Predicate<"Subtarget->hasMul32High()">,
                                         AssemblerPredicate<(all_of FeatureMul32High)>;

def FeatureDiv32            : SubtargetFeature<"div32", "HasDiv32", "true",
                                               "Enable Xtensa Div32 option">;
def HasDiv32                : Predicate<"Subtarget->hasDiv32()">,
                                         AssemblerPredicate<(all_of FeatureDiv32)>;

def FeatureMAC16            : SubtargetFeature<"mac16", "HasMAC16", "true",
                                               "Enable Xtensa MAC16 instructions">;
def HasMAC16                : Predicate<"Subtarget->hasMAC16()">,
                                         AssemblerPredicate<(all_of FeatureMAC16)>;

def FeatureDFPAccel         : SubtargetFeature<"dfpaccel", "HasDFPAccel", "true",
                                               "Enable Xtensa Double Precision FP acceleration">;
def HasDFPAccel             : Predicate<"Subtarget->hasDFPAccel()">,
                                        AssemblerPredicate<(all_of FeatureDFPAccel)>;

def FeatureS32C1I           : SubtargetFeature<"s32c1i", "HasS32C1I", "true",
                                               "Enable Xtensa S32C1I option">;
def HasS32C1I               : Predicate<"Subtarget->hasS32C1I()">,
                                         AssemblerPredicate<(all_of FeatureS32C1I)>;

def FeatureTHREADPTR         : SubtargetFeature<"threadptr", "HasTHREADPTR", "true",
                                                "Enable Xtensa THREADPTR option">;
def HasTHREADPTR             : Predicate<"Subtarget->hasTHREADPTR()">,
                                          AssemblerPredicate<(all_of FeatureTHREADPTR)>;

def FeatureExtendedL32R      : SubtargetFeature<"extendedl32r", "HasExtendedL32R", "true",
                                                "Enable Xtensa Extended L32R option">;
def HasExtendedL32R          : Predicate<"Subtarget->hasExtendedL32R()">,
                                          AssemblerPredicate<(all_of FeatureExtendedL32R)>;

def FeatureATOMCTL           : SubtargetFeature<"atomctl", "HasATOMCTL", "true",
                                                "Enable Xtensa ATOMCTL option">;
def HasATOMCTL               : Predicate<"Subtarget->hasATOMCTL()">,
                                          AssemblerPredicate<(all_of FeatureATOMCTL)>;

def FeatureMEMCTL           : SubtargetFeature<"memctl", "HasMEMCTL", "true",
                                                "Enable Xtensa MEMCTL option">;
def HasMEMCTL               : Predicate<"Subtarget->hasMEMCTL()">,
                                          AssemblerPredicate<(all_of FeatureMEMCTL)>;

def FeatureDebug             : SubtargetFeature<"debug", "HasDebug", "true",
                                                "Enable Xtensa Debug option">;
def HasDebug                 : Predicate<"Subtarget->hasDebug()">,
                                          AssemblerPredicate<(all_of FeatureDebug)>;

def FeatureException         : SubtargetFeature<"exception", "HasException", "true",
                                                "Enable Xtensa Exception option">;
def HasException             : Predicate<"Subtarget->hasException()">,
                                          AssemblerPredicate<(all_of FeatureException)>;

def FeatureHighPriInterrupts : SubtargetFeature<"highpriinterrupts",
                                                "HasHighPriInterrupts", "true",
                                                "Enable Xtensa HighPriInterrupts option">;
def HasHighPriInterrupts     : Predicate<"Subtarget->hasHighPriInterrupts()">,
                                          AssemblerPredicate<(all_of FeatureHighPriInterrupts)>;

def FeatureCoprocessor       : SubtargetFeature<"coprocessor", "HasCoprocessor", "true",
                                                "Enable Xtensa Coprocessor option">;
def HasCoprocessor           : Predicate<"Subtarget->hasCoprocessor()">,
                                          AssemblerPredicate<(all_of FeatureCoprocessor)>;

def FeatureInterrupt         : SubtargetFeature<"interrupt", "HasInterrupt", "true",
                                                "Enable Xtensa Interrupt option">;
def HasInterrupt             : Predicate<"Subtarget->hasInterrupt()">,
                                          AssemblerPredicate<(all_of FeatureInterrupt)>;

def FeatureRelocatableVector : SubtargetFeature<"rvector", "HasRelocatableVector", "true",
                                                "Enable Xtensa Relocatable Vector option">;
def HasRelocatableVector     : Predicate<"Subtarget->hasRelocatableVector()">,
                                          AssemblerPredicate<(all_of FeatureRelocatableVector)>;

def FeatureTimerInt          : SubtargetFeature<"timerint", "HasTimerInt", "true",
                                                "Enable Xtensa Timer Interrupt option">;
def HasTimerInt              : Predicate<"Subtarget->hasTimerInt()">,
                                          AssemblerPredicate<(all_of FeatureTimerInt)>;

def FeaturePRID              : SubtargetFeature<"prid", "HasPRID", "true",
                                                "Enable Xtensa Processor ID option">;
def HasPRID                  : Predicate<"Subtarget->hasPRID()">,
                                          AssemblerPredicate<(all_of FeaturePRID)>;

def FeatureRegionProtection  : SubtargetFeature<"regprotect", "HasRegionProtection", "true",
                                                "Enable Xtensa Region Protection option">;
def HasRegionProtection      : Predicate<"Subtarget->hasRegionProtection()">,
                                          AssemblerPredicate<(all_of FeatureRegionProtection)>;

def FeatureMiscSR            : SubtargetFeature<"miscsr", "HasMiscSR", "true",
                                                "Enable Xtensa Miscellaneous SR option">;
def HasMiscSR                : Predicate<"Subtarget->hasMiscSR()">,
                                          AssemblerPredicate<(all_of FeatureMiscSR)>;

//===----------------------------------------------------------------------===//
// Xtensa supported processors.
//===----------------------------------------------------------------------===//
class Proc<string Name, list<SubtargetFeature> Features>
    : Processor<Name, NoItineraries, Features>;

def : Proc<"generic", []>;

//===----------------------------------------------------------------------===//
// Register File Description
//===----------------------------------------------------------------------===//

include "XtensaRegisterInfo.td"

//===----------------------------------------------------------------------===//
// Calling Convention Description
//===----------------------------------------------------------------------===//

include "XtensaCallingConv.td"

//===----------------------------------------------------------------------===//
// Instruction Descriptions
//===----------------------------------------------------------------------===//

include "XtensaInstrInfo.td"

def XtensaInstrInfo : InstrInfo;

//===----------------------------------------------------------------------===//
// Target Declaration
//===----------------------------------------------------------------------===//

def XtensaAsmParser : AsmParser {
  let ShouldEmitMatchRegisterAltName = 1;
}

def XtensaInstPrinter : AsmWriter {
  string AsmWriterClassName  = "InstPrinter";
}

def Xtensa : Target {
  let InstructionSet = XtensaInstrInfo;
  let AssemblyWriters = [XtensaInstPrinter];
  let AssemblyParsers = [XtensaAsmParser];
}

