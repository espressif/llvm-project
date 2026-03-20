; RUN: llc -mtriple=riscv64 -mattr=+experimental-xtheadmatrix < %s \
; RUN:   | FileCheck %s
;
; Test that register class constraints are enforced:
; - Load A/B produces tile registers (tr0-tr3)
; - Load C produces accumulator registers (acc0-acc3)
; - Matmul takes acc + tile + tile, produces acc

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "riscv64-unknown-unknown"

;; ============================================================================
;; Test: Register class enforcement for matmul
;; Load-A should use tr*, Load-C should use acc*, matmul dst should be acc*
;; ============================================================================
define void @test_regclass_matmul(ptr %a_ptr, ptr %c_ptr, i64 %stride) {
; CHECK-LABEL: test_regclass_matmul:
; CHECK:        th.mlae32 tr{{[0-3]}}, ({{.*}}), {{.*}}
; CHECK:        th.mlce32 acc{{[0-3]}}, ({{.*}}), {{.*}}
; CHECK:        th.mfmacc.s acc{{[0-3]}}, tr{{[0-3]}}, tr{{[0-3]}}
; CHECK:        th.msce32 acc{{[0-3]}}, ({{.*}}), {{.*}}
; CHECK:        ret
  call void @llvm.riscv.th.msettilem.i64(i64 4)
  call void @llvm.riscv.th.msettilek.i64(i64 4)
  call void @llvm.riscv.th.msettilen.i64(i64 4)
  ; Load A into tile register
  %a = call target("riscv.matrix") @llvm.riscv.th.mlae.internal32.triscv.matrixt.i64(ptr %a_ptr, i64 %stride)
  ; Load C into accumulator register
  %c = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %c_ptr, i64 %stride)
  ; Matmul: acc = acc + tile * tile (note: a used for both ms2 and ms1 here)
  %result = call target("riscv.matrix") @llvm.riscv.th.mfmacc.s.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(
      target("riscv.matrix") %c, target("riscv.matrix") %a, target("riscv.matrix") %a)
  ; Store C from accumulator register
  call void @llvm.riscv.th.msce.internal32.triscv.matrixt.i64(
      target("riscv.matrix") %result, ptr %c_ptr, i64 %stride)
  ret void
}

;; ============================================================================
;; Test: Whole-register load/store uses any matrix register
;; ============================================================================
define void @test_whole_reg(ptr %src, ptr %dst) {
; CHECK-LABEL: test_whole_reg:
; CHECK:        th.mlme32 {{(tr|acc)[0-3]}}, ({{.*}}), {{.*}}
; CHECK:        th.msme32 {{(tr|acc)[0-3]}}, ({{.*}}), {{.*}}
; CHECK:        ret
  %v = call target("riscv.matrix") @llvm.riscv.th.mlme.internal32.triscv.matrixt.i64(ptr %src, i64 0)
  call void @llvm.riscv.th.msme.internal32.triscv.matrixt.i64(
      target("riscv.matrix") %v, ptr %dst, i64 0)
  ret void
}

;; ============================================================================
;; Test: Multiple matmul iterations (chained accumulate)
;; ============================================================================
define void @test_chained_matmul(ptr %a_ptr, ptr %b_ptr, ptr %c_ptr, i64 %stride) {
; CHECK-LABEL: test_chained_matmul:
; CHECK:        th.mlae32
; CHECK:        th.mlbe32
; CHECK:        th.mlce32
; CHECK:        th.mfmacc.s
; CHECK:        th.mfmacc.s
; CHECK:        th.msce32
; CHECK:        ret
  call void @llvm.riscv.th.msettilem.i64(i64 4)
  call void @llvm.riscv.th.msettilek.i64(i64 4)
  call void @llvm.riscv.th.msettilen.i64(i64 4)
  %a = call target("riscv.matrix") @llvm.riscv.th.mlae.internal32.triscv.matrixt.i64(ptr %a_ptr, i64 %stride)
  %b = call target("riscv.matrix") @llvm.riscv.th.mlbe.internal32.triscv.matrixt.i64(ptr %b_ptr, i64 %stride)
  %c = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %c_ptr, i64 %stride)
  ; First matmul
  %r1 = call target("riscv.matrix") @llvm.riscv.th.mfmacc.s.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(
      target("riscv.matrix") %c, target("riscv.matrix") %a, target("riscv.matrix") %b)
  ; Second matmul (chain: accumulates on top of first result)
  %r2 = call target("riscv.matrix") @llvm.riscv.th.mfmacc.s.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(
      target("riscv.matrix") %r1, target("riscv.matrix") %a, target("riscv.matrix") %b)
  call void @llvm.riscv.th.msce.internal32.triscv.matrixt.i64(
      target("riscv.matrix") %r2, ptr %c_ptr, i64 %stride)
  ret void
}

; Declarations
declare void @llvm.riscv.th.msettilem.i64(i64)
declare void @llvm.riscv.th.msettilek.i64(i64)
declare void @llvm.riscv.th.msettilen.i64(i64)
declare target("riscv.matrix") @llvm.riscv.th.mlae.internal32.triscv.matrixt.i64(ptr, i64)
declare target("riscv.matrix") @llvm.riscv.th.mlbe.internal32.triscv.matrixt.i64(ptr, i64)
declare target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr, i64)
declare target("riscv.matrix") @llvm.riscv.th.mlme.internal32.triscv.matrixt.i64(ptr, i64)
declare void @llvm.riscv.th.msce.internal32.triscv.matrixt.i64(target("riscv.matrix"), ptr, i64)
declare void @llvm.riscv.th.msme.internal32.triscv.matrixt.i64(target("riscv.matrix"), ptr, i64)
declare target("riscv.matrix") @llvm.riscv.th.mfmacc.s.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix"), target("riscv.matrix"), target("riscv.matrix"))
