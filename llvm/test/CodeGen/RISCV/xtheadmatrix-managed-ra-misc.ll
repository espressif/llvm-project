; RUN: llc -mtriple=riscv64 -mattr=+experimental-xtheadmatrix < %s \
; RUN:   | FileCheck %s
;
; Tests for ManagedRA miscellaneous operations: slides, broadcasts, move-to-GPR,
; move-from-GPR, dup, whole-register load/store, n4clip.

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "riscv64-unknown-unknown"

;; ============================================================================
;; Test: Row slide down
;; ============================================================================
define void @test_row_slide(ptr %src, ptr %dst) {
; CHECK-LABEL: test_row_slide:
; CHECK:        th.mlme32
; CHECK:        th.mrslidedown
; CHECK:        th.msme32
; CHECK:        ret
  %v = call target("riscv.matrix") @llvm.riscv.th.mlme.internal32.triscv.matrixt.i64(ptr %src, i64 0)
  %slid = call target("riscv.matrix") @llvm.riscv.th.mrslidedown.internal.triscv.matrixt.triscv.matrixt.i64(
      target("riscv.matrix") %v, i64 2)
  call void @llvm.riscv.th.msme.internal32.triscv.matrixt.i64(target("riscv.matrix") %slid, ptr %dst, i64 0)
  ret void
}

;; ============================================================================
;; Test: Row broadcast
;; ============================================================================
define void @test_row_broadcast(ptr %src, ptr %dst) {
; CHECK-LABEL: test_row_broadcast:
; CHECK:        th.mlme32
; CHECK:        th.mrbca.mv.i
; CHECK:        th.msme32
; CHECK:        ret
  %v = call target("riscv.matrix") @llvm.riscv.th.mlme.internal32.triscv.matrixt.i64(ptr %src, i64 0)
  %bcast = call target("riscv.matrix") @llvm.riscv.th.mrbca.mv.i.internal.triscv.matrixt.triscv.matrixt.i64(
      target("riscv.matrix") %v, i64 1)
  call void @llvm.riscv.th.msme.internal32.triscv.matrixt.i64(target("riscv.matrix") %bcast, ptr %dst, i64 0)
  ret void
}

;; ============================================================================
;; Test: Move matrix-element-to-GPR
;; ============================================================================
define i64 @test_mmov_to_gpr(ptr %src) {
; CHECK-LABEL: test_mmov_to_gpr:
; CHECK:        th.mlme32
; CHECK:        th.mmovw.x.m
; CHECK:        ret
  %v = call target("riscv.matrix") @llvm.riscv.th.mlme.internal32.triscv.matrixt.i64(ptr %src, i64 0)
  %elem = call i64 @llvm.riscv.th.mmovw.x.m.internal.i64.triscv.matrixt(
      target("riscv.matrix") %v, i64 0)
  ret i64 %elem
}

;; ============================================================================
;; Test: Move GPR-to-matrix-element
;; ============================================================================
define void @test_mmov_from_gpr(ptr %src, ptr %dst, i64 %val) {
; CHECK-LABEL: test_mmov_from_gpr:
; CHECK:        th.mlme32
; CHECK:        th.mmovw.m.x
; CHECK:        th.msme32
; CHECK:        ret
  %v = call target("riscv.matrix") @llvm.riscv.th.mlme.internal32.triscv.matrixt.i64(ptr %src, i64 0)
  %modified = call target("riscv.matrix") @llvm.riscv.th.mmovw.m.x.internal.triscv.matrixt.i64(
      target("riscv.matrix") %v, i64 %val, i64 0)
  call void @llvm.riscv.th.msme.internal32.triscv.matrixt.i64(target("riscv.matrix") %modified, ptr %dst, i64 0)
  ret void
}

;; ============================================================================
;; Test: Dup GPR to matrix column
;; ============================================================================
define void @test_dup(ptr %src, ptr %dst, i64 %val) {
; CHECK-LABEL: test_dup:
; CHECK:        th.mlme32
; CHECK:        th.mdupw.m.x
; CHECK:        th.msme32
; CHECK:        ret
  %v = call target("riscv.matrix") @llvm.riscv.th.mlme.internal32.triscv.matrixt.i64(ptr %src, i64 0)
  %duped = call target("riscv.matrix") @llvm.riscv.th.mdupw.m.x.internal.triscv.matrixt.i64(
      target("riscv.matrix") %v, i64 %val)
  call void @llvm.riscv.th.msme.internal32.triscv.matrixt.i64(target("riscv.matrix") %duped, ptr %dst, i64 0)
  ret void
}

;; ============================================================================
;; Test: Integer element-wise multiply
;; ============================================================================
define void @test_ew_mul(ptr %a_ptr, ptr %b_ptr, ptr %c_ptr, i64 %stride) {
; CHECK-LABEL: test_ew_mul:
; CHECK:        th.mlce32
; CHECK:        th.mlce32
; CHECK:        th.mlce32
; CHECK:        th.mmul.w.mm
; CHECK:        th.msce32
; CHECK:        ret
  call void @llvm.riscv.th.msettilem.i64(i64 4)
  call void @llvm.riscv.th.msettilen.i64(i64 4)
  %a = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %a_ptr, i64 %stride)
  %b = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %b_ptr, i64 %stride)
  %c = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %c_ptr, i64 %stride)
  %r = call target("riscv.matrix") @llvm.riscv.th.mmul.w.mm.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(
      target("riscv.matrix") %c, target("riscv.matrix") %a, target("riscv.matrix") %b)
  call void @llvm.riscv.th.msce.internal32.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %c_ptr, i64 %stride)
  ret void
}

;; ============================================================================
;; Test: Transpose load (tile-stride)
;; ============================================================================
define void @test_transpose_load(ptr %ptr, ptr %dst, i64 %stride) {
; CHECK-LABEL: test_transpose_load:
; CHECK:        th.mlate32
; CHECK:        th.msae32
; CHECK:        ret
  call void @llvm.riscv.th.msettilem.i64(i64 4)
  call void @llvm.riscv.th.msettilek.i64(i64 4)
  %t = call target("riscv.matrix") @llvm.riscv.th.mlate.internal32.triscv.matrixt.i64(ptr %ptr, i64 %stride)
  call void @llvm.riscv.th.msae.internal32.triscv.matrixt.i64(target("riscv.matrix") %t, ptr %dst, i64 %stride)
  ret void
}

;; ============================================================================
;; Intrinsic declarations
;; ============================================================================
declare void @llvm.riscv.th.msettilem.i64(i64)
declare void @llvm.riscv.th.msettilek.i64(i64)
declare void @llvm.riscv.th.msettilen.i64(i64)

; Whole-register load/store
declare target("riscv.matrix") @llvm.riscv.th.mlme.internal32.triscv.matrixt.i64(ptr, i64)
declare void @llvm.riscv.th.msme.internal32.triscv.matrixt.i64(target("riscv.matrix"), ptr, i64)

; Strided load/store
declare target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr, i64)
declare target("riscv.matrix") @llvm.riscv.th.mlate.internal32.triscv.matrixt.i64(ptr, i64)
declare void @llvm.riscv.th.msce.internal32.triscv.matrixt.i64(target("riscv.matrix"), ptr, i64)
declare void @llvm.riscv.th.msae.internal32.triscv.matrixt.i64(target("riscv.matrix"), ptr, i64)

; Slide
declare target("riscv.matrix") @llvm.riscv.th.mrslidedown.internal.triscv.matrixt.triscv.matrixt.i64(target("riscv.matrix"), i64)

; Broadcast
declare target("riscv.matrix") @llvm.riscv.th.mrbca.mv.i.internal.triscv.matrixt.triscv.matrixt.i64(target("riscv.matrix"), i64)

; Move to/from GPR
declare i64 @llvm.riscv.th.mmovw.x.m.internal.i64.triscv.matrixt(target("riscv.matrix"), i64)
declare target("riscv.matrix") @llvm.riscv.th.mmovw.m.x.internal.triscv.matrixt.i64(target("riscv.matrix"), i64, i64)

; Dup
declare target("riscv.matrix") @llvm.riscv.th.mdupw.m.x.internal.triscv.matrixt.i64(target("riscv.matrix"), i64)

; EW arithmetic
declare target("riscv.matrix") @llvm.riscv.th.mmul.w.mm.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix"), target("riscv.matrix"), target("riscv.matrix"))
