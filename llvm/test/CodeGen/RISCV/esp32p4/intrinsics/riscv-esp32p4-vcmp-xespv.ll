; RUN: llc -O2 -mattr=+xespv,+espv-lowering -mtriple=riscv32 %s -o - | FileCheck %s --check-prefix=ASM
; RUN: llc -O2 -mattr=+xespv,+espv-lowering -mtriple=riscv32 -stop-after=finalize-isel %s -o - | FileCheck %s --check-prefix=MIR

; Same espvm vcmp intrinsic: +xespv selects ESP_VCMP_*_2P2.
; VCMP.EQ.U / VCMP.GT.U are PIE 2.1-only (riscv-esp32p4-vcmp-u21-only.ll).

; MIR-DAG: ESP_VCMP_EQ_S8_2P2
; MIR-DAG: ESP_VCMP_GT_S8_2P2
; MIR-DAG: ESP_VCMP_LT_S8_2P2

define void @test_vcmp_eq_s8_xespv(ptr %a, ptr %b, ptr %dst) {
; ASM-LABEL: test_vcmp_eq_s8_xespv:
; ASM:       esp.vcmp.eq.s8
entry:
  %va = call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %a, i32 16)
  %ea = extractvalue { <16 x i8>, ptr } %va, 0
  %vb = call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %b, i32 16)
  %eb = extractvalue { <16 x i8>, ptr } %vb, 0
  %r = call <16 x i8> @llvm.riscv.esp.vcmp.eq.s8(<16 x i8> %ea, <16 x i8> %eb)
  %p = call ptr @llvm.riscv.esp.vst.128.ip(<16 x i8> %r, ptr %dst, i32 16)
  ret void
}

define void @test_vcmp_gt_s8_xespv(ptr %a, ptr %b, ptr %dst) {
; ASM-LABEL: test_vcmp_gt_s8_xespv:
; ASM:       esp.vcmp.gt.s8
entry:
  %va = call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %a, i32 16)
  %ea = extractvalue { <16 x i8>, ptr } %va, 0
  %vb = call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %b, i32 16)
  %eb = extractvalue { <16 x i8>, ptr } %vb, 0
  %r = call <16 x i8> @llvm.riscv.esp.vcmp.gt.s8(<16 x i8> %ea, <16 x i8> %eb)
  %p = call ptr @llvm.riscv.esp.vst.128.ip(<16 x i8> %r, ptr %dst, i32 16)
  ret void
}

define void @test_vcmp_lt_s8_xespv(ptr %a, ptr %b, ptr %dst) {
; ASM-LABEL: test_vcmp_lt_s8_xespv:
; ASM:       esp.vcmp.lt.s8
entry:
  %va = call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %a, i32 16)
  %ea = extractvalue { <16 x i8>, ptr } %va, 0
  %vb = call { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr %b, i32 16)
  %eb = extractvalue { <16 x i8>, ptr } %vb, 0
  %r = call <16 x i8> @llvm.riscv.esp.vcmp.lt.s8(<16 x i8> %ea, <16 x i8> %eb)
  %p = call ptr @llvm.riscv.esp.vst.128.ip(<16 x i8> %r, ptr %dst, i32 16)
  ret void
}

declare { <16 x i8>, ptr } @llvm.riscv.esp.vld.128.ip(ptr, i32)
declare ptr @llvm.riscv.esp.vst.128.ip(<16 x i8>, ptr, i32)
declare <16 x i8> @llvm.riscv.esp.vcmp.eq.s8(<16 x i8>, <16 x i8>)
declare <16 x i8> @llvm.riscv.esp.vcmp.gt.s8(<16 x i8>, <16 x i8>)
declare <16 x i8> @llvm.riscv.esp.vcmp.lt.s8(<16 x i8>, <16 x i8>)
