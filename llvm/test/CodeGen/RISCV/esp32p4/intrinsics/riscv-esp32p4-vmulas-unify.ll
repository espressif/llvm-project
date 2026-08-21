; Surface (!629): DAGToDAG selectESP covers VMULAS XACC/QACC + PIE22 sat after restack onto !628;
; lit already present on parent — keep FileCheck ASM surface.
; RUN: llc -O2 -mattr=+xespv2p1,+espv-lowering -mtriple=riscv32 %s -o - | FileCheck %s --check-prefix=ASM

define dso_local void @test_vmulas_s8_qacc(ptr %src1, ptr %src2, ptr %dst) {
; ASM-LABEL: test_vmulas_s8_qacc:
; ASM:       esp.vmulas.s8.qacc
; ASM-NOT:   trunc
entry:
  %z = tail call { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.riscv.esp.zero.qacc()
  %v0 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %z, 0
  %v1 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %z, 1
  %v2 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %z, 2
  %v3 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %z, 3
  %ld1 = tail call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %src1, i32 16)
  %qx = extractvalue { <16 x i8>, ptr } %ld1, 0
  %ld2 = tail call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %src2, i32 16)
  %qy = extractvalue { <16 x i8>, ptr } %ld2, 0
  %acc = tail call { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.riscv.esp.vmulas.s8.qacc(<16 x i8> %v0, <16 x i8> %v1, <16 x i8> %v2, <16 x i8> %v3, <16 x i8> %qx, <16 x i8> %qy, i32 0)
  %o0 = extractvalue { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } %acc, 0
  tail call ptr @llvm.riscv.esp.st.qacc.l.l.128.ip(<16 x i8> %o0, ptr %dst, i32 16)
  ret void
}

declare { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.riscv.esp.zero.qacc()
declare { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr, i32)
declare { <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8> } @llvm.riscv.esp.vmulas.s8.qacc(<16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, <16 x i8>, i32)
declare ptr @llvm.riscv.esp.st.qacc.l.l.128.ip(<16 x i8>, ptr, i32)
