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

def FeatureMul16            : SubtargetFeature<"mul16", "HasMul16", "true",
                                               "Enable Xtensa Mul16 option">;
def HasMul16                : Predicate<"Subtarget->hasMul16()">,
                                         AssemblerPredicate<(all_of FeatureMul16)>;

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

def FeatureESP32S2Ops        : SubtargetFeature<"esp32s2", "HasESP32S2Ops", "true",
                                                "Support Xtensa esp32-s2 ISA extension">;
def HasESP32S2Ops            : Predicate<"Subtarget->hasESP32S2Ops()">,
                                          AssemblerPredicate<(all_of FeatureESP32S2Ops)>;

def FeatureESP32S3Ops        : SubtargetFeature<"esp32s3", "HasESP32S3Ops", "true",
                                                "Support Xtensa esp32-s3 ISA extension">;
def HasESP32S3Ops            : Predicate<"Subtarget->hasESP32S3Ops()">,
                                          AssemblerPredicate<(all_of FeatureESP32S3Ops)>;

def FeatureHIFI3            : SubtargetFeature<"hifi3", "HasHIFI3", "true",
                                               "Enable Xtensa HIFI3 instructions">;
def HasHIFI3                : Predicate<"Subtarget->hasHIFI3()">,
                                         AssemblerPredicate<(all_of FeatureHIFI3)>;

//===----------------------------------------------------------------------===//
// Xtensa supported processors.
//===----------------------------------------------------------------------===//
class Proc<string Name, list<SubtargetFeature> Features>
    : Processor<Name, NoItineraries, Features>;

def : Proc<"generic", []>;

def : Proc<"esp32", [FeatureDensity, FeatureSingleFloat, FeatureLoop, FeatureMAC16, FeatureWindowed, FeatureBoolean, FeatureSEXT,
                     FeatureNSA, FeatureMul16, FeatureMul32, FeatureMul32High, FeatureDFPAccel, FeatureS32C1I, FeatureTHREADPTR, FeatureDiv32,
                     FeatureATOMCTL, FeatureMEMCTL, FeatureDebug, FeatureException, FeatureHighPriInterrupts, FeatureCoprocessor,
                     FeatureInterrupt, FeatureRelocatableVector, FeatureTimerInt, FeaturePRID, FeatureRegionProtection, FeatureMiscSR]>;

def : Proc<"esp8266", [FeatureDensity, FeatureNSA, FeatureMul16, FeatureMul32, FeatureExtendedL32R, FeatureDebug, FeatureException,
                       FeatureHighPriInterrupts, FeatureInterrupt, FeatureRelocatableVector, FeatureTimerInt, FeatureRegionProtection, FeaturePRID]>;

def : Proc<"esp32s2", [FeatureDensity, FeatureWindowed, FeatureSEXT, FeatureNSA,  FeatureMul16, FeatureMul32, FeatureMul32High, FeatureTHREADPTR,
                       FeatureDiv32, FeatureMEMCTL, FeatureDebug, FeatureException, FeatureHighPriInterrupts, FeatureCoprocessor, FeatureInterrupt,
                       FeatureRelocatableVector,	FeatureTimerInt, FeaturePRID, FeatureRegionProtection, FeatureMiscSR, FeatureESP32S2Ops]>;

def : Proc<"esp32s3", [FeatureDensity, FeatureSingleFloat, FeatureLoop, FeatureMAC16, FeatureWindowed, FeatureBoolean, FeatureSEXT,
                       FeatureNSA, FeatureMul16, FeatureMul32, FeatureMul32High, FeatureDFPAccel, FeatureS32C1I, FeatureTHREADPTR, FeatureDiv32,
                       FeatureATOMCTL, FeatureMEMCTL, FeatureDebug, FeatureException, FeatureHighPriInterrupts, FeatureCoprocessor,
                       FeatureInterrupt, FeatureRelocatableVector, FeatureTimerInt, FeaturePRID, FeatureRegionProtection, FeatureMiscSR,
                       FeatureESP32S3Ops]>;

def : Proc<"cnl", [FeatureDensity, FeatureSingleFloat, FeatureLoop,  FeatureWindowed, FeatureBoolean, 
                    FeatureSEXT, FeatureNSA, FeatureMul32, FeatureMul32High,  FeatureS32C1I,
                    FeatureTHREADPTR, FeatureDiv32, FeatureATOMCTL, FeatureMEMCTL, FeatureDebug, 
                    FeatureException, FeatureHighPriInterrupts, FeatureCoprocessor,
                     FeatureInterrupt, FeatureRelocatableVector, FeatureTimerInt, FeaturePRID,
                     FeatureRegionProtection, FeatureMiscSR, FeatureHIFI3]>; 

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
  bit isMCAsmWriter = 1;
}

def Xtensa : Target {
  let InstructionSet = XtensaInstrInfo;
  let AssemblyWriters = [XtensaInstPrinter];
  let AssemblyParsers = [XtensaAsmParser];
}

