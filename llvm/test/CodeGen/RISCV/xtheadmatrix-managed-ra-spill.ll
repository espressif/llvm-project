; RUN: llc -mtriple=riscv64 -mattr=+experimental-xtheadmatrix < %s \
; RUN:   | FileCheck %s
;
; Test that register pressure causes spills/reloads via th.msme/th.mlme.
; With only 4 tile registers (tr0-tr3) and 4 accumulator registers (acc0-acc3),
; using 5+ matrix values should force at least one spill.

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "riscv64-unknown-unknown"

;; ============================================================================
;; Test: 5 tile loads forces a spill (only 4 THRVMTR registers available)
;; ============================================================================
define void @test_tile_spill(ptr %p0, ptr %p1, ptr %p2, ptr %p3, ptr %p4,
                             ptr %dst, i64 %stride) {
; CHECK-LABEL: test_tile_spill:
; CHECK:        th.mlae32
; CHECK:        th.mlae32
; CHECK:        th.mlae32
; CHECK:        th.mlae32
; The 5th tile load forces a spill (only 4 THRVMTR registers).
; Verify spill via whole-register store.
; CHECK:        th.msme
; CHECK:        th.mlae32
; Verify reload and stores.
; CHECK:        th.msae32
; CHECK:        th.msae32
; CHECK:        th.msae32
; CHECK:        th.mlme
; CHECK:        th.msae32
; CHECK:        th.mlme
; CHECK:        th.msae32
; CHECK:        ret
  call void @llvm.riscv.th.msettilem.i64(i64 4)
  call void @llvm.riscv.th.msettilek.i64(i64 4)
  %t0 = call target("riscv.matrix") @llvm.riscv.th.mlae.internal32.triscv.matrixt.i64(ptr %p0, i64 %stride)
  %t1 = call target("riscv.matrix") @llvm.riscv.th.mlae.internal32.triscv.matrixt.i64(ptr %p1, i64 %stride)
  %t2 = call target("riscv.matrix") @llvm.riscv.th.mlae.internal32.triscv.matrixt.i64(ptr %p2, i64 %stride)
  %t3 = call target("riscv.matrix") @llvm.riscv.th.mlae.internal32.triscv.matrixt.i64(ptr %p3, i64 %stride)
  %t4 = call target("riscv.matrix") @llvm.riscv.th.mlae.internal32.triscv.matrixt.i64(ptr %p4, i64 %stride)
  ; Store all 5 — forces all to be live simultaneously
  call void @llvm.riscv.th.msae.internal32.triscv.matrixt.i64(target("riscv.matrix") %t0, ptr %dst, i64 %stride)
  %dst1 = getelementptr i8, ptr %dst, i64 1024
  call void @llvm.riscv.th.msae.internal32.triscv.matrixt.i64(target("riscv.matrix") %t1, ptr %dst1, i64 %stride)
  %dst2 = getelementptr i8, ptr %dst, i64 2048
  call void @llvm.riscv.th.msae.internal32.triscv.matrixt.i64(target("riscv.matrix") %t2, ptr %dst2, i64 %stride)
  %dst3 = getelementptr i8, ptr %dst, i64 3072
  call void @llvm.riscv.th.msae.internal32.triscv.matrixt.i64(target("riscv.matrix") %t3, ptr %dst3, i64 %stride)
  %dst4 = getelementptr i8, ptr %dst, i64 4096
  call void @llvm.riscv.th.msae.internal32.triscv.matrixt.i64(target("riscv.matrix") %t4, ptr %dst4, i64 %stride)
  ret void
}

; Declarations
declare void @llvm.riscv.th.msettilem.i64(i64)
declare void @llvm.riscv.th.msettilek.i64(i64)
declare target("riscv.matrix") @llvm.riscv.th.mlae.internal32.triscv.matrixt.i64(ptr, i64)
declare void @llvm.riscv.th.msae.internal32.triscv.matrixt.i64(target("riscv.matrix"), ptr, i64)
