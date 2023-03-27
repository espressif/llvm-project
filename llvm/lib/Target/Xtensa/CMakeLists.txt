add_llvm_component_group(Xtensa)

set(LLVM_TARGET_DEFINITIONS Xtensa.td)

tablegen(LLVM XtensaGenAsmMatcher.inc -gen-asm-matcher)
tablegen(LLVM XtensaGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM XtensaGenCallingConv.inc -gen-callingconv)
tablegen(LLVM XtensaGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM XtensaGenDisassemblerTables.inc -gen-disassembler)
tablegen(LLVM XtensaGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM XtensaGenMCCodeEmitter.inc -gen-emitter)
tablegen(LLVM XtensaGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM XtensaGenSubtargetInfo.inc -gen-subtarget)

add_public_tablegen_target(XtensaCommonTableGen)

add_llvm_target(XtensaCodeGen
  XtensaAsmPrinter.cpp
  XtensaConstantPoolValue.cpp
  XtensaConstantIsland.cpp
  XtensaESP32PSRAMFix.cpp
  XtensaFixupHWLoops.cpp
  XtensaFrameLowering.cpp
  XtensaHardwareLoops.cpp
  XtensaInstrInfo.cpp
  XtensaISelDAGToDAG.cpp
  XtensaISelLowering.cpp
  XtensaS3ISelLowering.cpp
  XtensaMachineFunctionInfo.cpp
  XtensaMCInstLower.cpp
  XtensaRegisterInfo.cpp
  XtensaSizeReductionPass.cpp
  XtensaSubtarget.cpp
  XtensaTargetMachine.cpp
  XtensaTargetObjectFile.cpp
  XtensaTargetTransformInfo.cpp
  XtensaBRegFixupPass.cpp

  LINK_COMPONENTS
  AsmPrinter
  CodeGen
  Core
  MC
  SelectionDAG
  Support
  Target
  XtensaDesc
  XtensaInfo

  ADD_TO_COMPONENT
  Xtensa
  )

add_subdirectory(AsmParser)
add_subdirectory(Disassembler)
add_subdirectory(MCTargetDesc)
add_subdirectory(TargetInfo)

