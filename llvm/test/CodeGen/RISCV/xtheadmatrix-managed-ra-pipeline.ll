; RUN: llc -mtriple=riscv64 -mattr=+experimental-xtheadmatrix < %s \
; RUN:   | FileCheck %s
;
; Register allocation and dependency chain tests for XTHeadMatrix ManagedRA.
; These test register class correctness (tile vs accumulator), spill behavior
; under pressure, dependency chain preservation, and multi-operation pipelines
; at the assembly output level.

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "riscv64-unknown-unknown"

;; ============================================================================
;; Test 1: Register class — A-tile in tr*, B-tile in tr*, C-acc in acc*,
;; matmul output in acc*, store from acc*. Full pipeline.
;; ============================================================================
define void @test_full_pipeline_regclass(ptr %a, ptr %b, ptr %c, i64 %stride) {
; CHECK-LABEL: test_full_pipeline_regclass:
; CHECK:        th.mlae8 tr{{[0-3]}}
; CHECK:        th.mlbe8 tr{{[0-3]}}
; CHECK:        th.mzero acc{{[0-3]}}
; CHECK:        th.mmacc.w.b acc{{[0-3]}}, tr{{[0-3]}}, tr{{[0-3]}}
; CHECK:        th.msce32 acc{{[0-3]}}
; CHECK:        ret
  call void @llvm.riscv.th.msettilem.i64(i64 4)
  call void @llvm.riscv.th.msettilek.i64(i64 16)
  call void @llvm.riscv.th.msettilen.i64(i64 4)
  %ta = call target("riscv.matrix") @llvm.riscv.th.mlae.internal8.triscv.matrixt.i64(ptr %a, i64 %stride)
  %tb = call target("riscv.matrix") @llvm.riscv.th.mlbe.internal8.triscv.matrixt.i64(ptr %b, i64 %stride)
  %z = call target("riscv.matrix") @llvm.riscv.th.mzero.internal.triscv.matrixt()
  %r = call target("riscv.matrix") @llvm.riscv.th.mmacc.w.b.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(
      target("riscv.matrix") %z, target("riscv.matrix") %ta, target("riscv.matrix") %tb)
  call void @llvm.riscv.th.msce.internal32.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %c, i64 %stride)
  ret void
}

;; ============================================================================
;; Test 2: EW operations — all operands must be acc registers
;; ============================================================================
define void @test_ew_regclass(ptr %a, ptr %b, ptr %c, i64 %stride) {
; CHECK-LABEL: test_ew_regclass:
; CHECK:        th.mlce32 acc{{[0-3]}}
; CHECK:        th.mlce32 acc{{[0-3]}}
; CHECK:        th.madd.w.mm acc{{[0-3]}}, acc{{[0-3]}}, acc{{[0-3]}}
; CHECK:        th.msce32 acc{{[0-3]}}
; CHECK:        ret
  call void @llvm.riscv.th.msettilem.i64(i64 4)
  call void @llvm.riscv.th.msettilen.i64(i64 4)
  %va = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %a, i64 %stride)
  %vb = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %b, i64 %stride)
  %r = call target("riscv.matrix") @llvm.riscv.th.madd.w.mm.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(
      target("riscv.matrix") %va, target("riscv.matrix") %va, target("riscv.matrix") %vb)
  call void @llvm.riscv.th.msce.internal32.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %c, i64 %stride)
  ret void
}

;; ============================================================================
;; Test 3: Conversion — both input and output must be acc registers
;; ============================================================================
define void @test_conversion_regclass(ptr %src, ptr %dst, i64 %stride) {
; CHECK-LABEL: test_conversion_regclass:
; CHECK:        th.mlce16 acc{{[0-3]}}
; CHECK:        th.mfcvtl.s.h acc{{[0-3]}}, acc{{[0-3]}}
; CHECK:        th.msce32 acc{{[0-3]}}
; CHECK:        ret
  call void @llvm.riscv.th.msettilem.i64(i64 4)
  call void @llvm.riscv.th.msettilen.i64(i64 4)
  %src_val = call target("riscv.matrix") @llvm.riscv.th.mlce.internal16.triscv.matrixt.i64(ptr %src, i64 %stride)
  %cvt = call target("riscv.matrix") @llvm.riscv.th.mfcvtl.s.h.internal.triscv.matrixt.triscv.matrixt(
      target("riscv.matrix") %src_val)
  call void @llvm.riscv.th.msce.internal32.triscv.matrixt.i64(target("riscv.matrix") %cvt, ptr %dst, i64 %stride)
  ret void
}

;; ============================================================================
;; Test 4: Dependency chain — matmul → EW add → store
;; The EW result must feed from the matmul result, not be reordered.
;; ============================================================================
define void @test_dependency_chain(ptr %a, ptr %b, ptr %bias, ptr %c, i64 %stride) {
; CHECK-LABEL: test_dependency_chain:
; CHECK:        th.mlae8
; CHECK:        th.mlbe8
; CHECK:        th.mzero
; CHECK:        th.mmacc.w.b
;   EW add must come after matmul:
; CHECK:        th.mlce32
; CHECK:        th.madd.w.mm
; CHECK:        th.msce32
; CHECK:        ret
  call void @llvm.riscv.th.msettilem.i64(i64 4)
  call void @llvm.riscv.th.msettilek.i64(i64 16)
  call void @llvm.riscv.th.msettilen.i64(i64 4)
  %ta = call target("riscv.matrix") @llvm.riscv.th.mlae.internal8.triscv.matrixt.i64(ptr %a, i64 %stride)
  %tb = call target("riscv.matrix") @llvm.riscv.th.mlbe.internal8.triscv.matrixt.i64(ptr %b, i64 %stride)
  %z = call target("riscv.matrix") @llvm.riscv.th.mzero.internal.triscv.matrixt()
  %matmul_r = call target("riscv.matrix") @llvm.riscv.th.mmacc.w.b.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(
      target("riscv.matrix") %z, target("riscv.matrix") %ta, target("riscv.matrix") %tb)
  ; Load bias into acc register
  %bias_v = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %bias, i64 %stride)
  ; EW add: matmul_r + bias (uses matmul result as dependency)
  %add_r = call target("riscv.matrix") @llvm.riscv.th.madd.w.mm.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(
      target("riscv.matrix") %matmul_r, target("riscv.matrix") %matmul_r, target("riscv.matrix") %bias_v)
  call void @llvm.riscv.th.msce.internal32.triscv.matrixt.i64(target("riscv.matrix") %add_r, ptr %c, i64 %stride)
  ret void
}

;; ============================================================================
;; Test 5: Register pressure — 3 ACC values live simultaneously
;; With only 4 acc registers, 3 live values should still fit without spill.
;; ============================================================================
define void @test_3_acc_no_spill(ptr %p1, ptr %p2, ptr %p3, ptr %dst, i64 %stride) {
; CHECK-LABEL: test_3_acc_no_spill:
; CHECK-NOT:    th.msme
; CHECK:        th.mlce32 acc{{[0-3]}}
; CHECK:        th.mlce32 acc{{[0-3]}}
; CHECK:        th.mlce32 acc{{[0-3]}}
; CHECK:        th.madd.w.mm
; CHECK:        th.madd.w.mm
; CHECK:        th.msce32
; CHECK:        ret
  call void @llvm.riscv.th.msettilem.i64(i64 4)
  call void @llvm.riscv.th.msettilen.i64(i64 4)
  %v1 = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %p1, i64 %stride)
  %v2 = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %p2, i64 %stride)
  %v3 = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %p3, i64 %stride)
  ; v1 + v2
  %sum12 = call target("riscv.matrix") @llvm.riscv.th.madd.w.mm.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(
      target("riscv.matrix") %v1, target("riscv.matrix") %v1, target("riscv.matrix") %v2)
  ; (v1+v2) + v3
  %total = call target("riscv.matrix") @llvm.riscv.th.madd.w.mm.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(
      target("riscv.matrix") %sum12, target("riscv.matrix") %sum12, target("riscv.matrix") %v3)
  call void @llvm.riscv.th.msce.internal32.triscv.matrixt.i64(target("riscv.matrix") %total, ptr %dst, i64 %stride)
  ret void
}

;; ============================================================================
;; Test 6: Register pressure — 5 ACC values force spill
;; With only 4 acc registers, 5 live values MUST cause a spill.
;; ============================================================================
define void @test_5_acc_spill(ptr %p1, ptr %p2, ptr %p3, ptr %p4, ptr %p5, ptr %dst, i64 %stride) {
; CHECK-LABEL: test_5_acc_spill:
;   At least one spill (msme = whole-register store) must occur:
; CHECK:        th.msme
;   And at least one reload (mlme = whole-register load):
; CHECK:        th.mlme
; CHECK:        ret
  call void @llvm.riscv.th.msettilem.i64(i64 4)
  call void @llvm.riscv.th.msettilen.i64(i64 4)
  %v1 = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %p1, i64 %stride)
  %v2 = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %p2, i64 %stride)
  %v3 = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %p3, i64 %stride)
  %v4 = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %p4, i64 %stride)
  %v5 = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %p5, i64 %stride)
  ; Use all 5 to force them all live
  %s12 = call target("riscv.matrix") @llvm.riscv.th.madd.w.mm.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(
      target("riscv.matrix") %v1, target("riscv.matrix") %v1, target("riscv.matrix") %v2)
  %s34 = call target("riscv.matrix") @llvm.riscv.th.madd.w.mm.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(
      target("riscv.matrix") %v3, target("riscv.matrix") %v3, target("riscv.matrix") %v4)
  %s1234 = call target("riscv.matrix") @llvm.riscv.th.madd.w.mm.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(
      target("riscv.matrix") %s12, target("riscv.matrix") %s12, target("riscv.matrix") %s34)
  %total = call target("riscv.matrix") @llvm.riscv.th.madd.w.mm.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(
      target("riscv.matrix") %s1234, target("riscv.matrix") %s1234, target("riscv.matrix") %v5)
  call void @llvm.riscv.th.msce.internal32.triscv.matrixt.i64(target("riscv.matrix") %total, ptr %dst, i64 %stride)
  ret void
}

;; ============================================================================
;; Test 7: Chained FP matmul accumulate (2 iterations)
;; Second matmul's acc input must be first matmul's output (not reordered).
;; ============================================================================
define void @test_chained_fp_matmul(ptr %a, ptr %b, ptr %c, i64 %stride) {
; CHECK-LABEL: test_chained_fp_matmul:
;   Two mfmacc.s in order, same acc register:
; CHECK:        th.mfmacc.s [[ACC:acc[0-3]]], tr{{[0-3]}}, tr{{[0-3]}}
; CHECK:        th.mfmacc.s [[ACC]], tr{{[0-3]}}, tr{{[0-3]}}
; CHECK:        th.msce32 [[ACC]]
; CHECK:        ret
  call void @llvm.riscv.th.msettilem.i64(i64 4)
  call void @llvm.riscv.th.msettilek.i64(i64 4)
  call void @llvm.riscv.th.msettilen.i64(i64 4)
  %ta = call target("riscv.matrix") @llvm.riscv.th.mlae.internal32.triscv.matrixt.i64(ptr %a, i64 %stride)
  %tb = call target("riscv.matrix") @llvm.riscv.th.mlbe.internal32.triscv.matrixt.i64(ptr %b, i64 %stride)
  %tc = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %c, i64 %stride)
  %r1 = call target("riscv.matrix") @llvm.riscv.th.mfmacc.s.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(
      target("riscv.matrix") %tc, target("riscv.matrix") %ta, target("riscv.matrix") %tb)
  %r2 = call target("riscv.matrix") @llvm.riscv.th.mfmacc.s.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(
      target("riscv.matrix") %r1, target("riscv.matrix") %ta, target("riscv.matrix") %tb)
  call void @llvm.riscv.th.msce.internal32.triscv.matrixt.i64(target("riscv.matrix") %r2, ptr %c, i64 %stride)
  ret void
}

;; ============================================================================
;; Test 8: N4clip uses acc register class for all operands
;; ============================================================================
define void @test_n4clip_regclass(ptr %a, ptr %b, ptr %c, i64 %stride) {
; CHECK-LABEL: test_n4clip_regclass:
; CHECK:        th.mlce32 acc{{[0-3]}}
; CHECK:        th.mlce32 acc{{[0-3]}}
; CHECK:        th.mn4clipl.w.mm acc{{[0-3]}}, acc{{[0-3]}}, acc{{[0-3]}}
; CHECK:        th.msce32 acc{{[0-3]}}
; CHECK:        ret
  call void @llvm.riscv.th.msettilem.i64(i64 4)
  call void @llvm.riscv.th.msettilen.i64(i64 4)
  %va = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %a, i64 %stride)
  %vb = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %b, i64 %stride)
  %clip = call target("riscv.matrix") @llvm.riscv.th.mn4clipl.w.mm.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(
      target("riscv.matrix") %va, target("riscv.matrix") %vb, target("riscv.matrix") %va)
  call void @llvm.riscv.th.msce.internal32.triscv.matrixt.i64(target("riscv.matrix") %clip, ptr %c, i64 %stride)
  ret void
}

;; ============================================================================
;; Test 9: FP EW uses acc registers
;; ============================================================================
define void @test_fp_ew_regclass(ptr %a, ptr %b, ptr %c, i64 %stride) {
; CHECK-LABEL: test_fp_ew_regclass:
; CHECK:        th.mlce32 acc{{[0-3]}}
; CHECK:        th.mlce32 acc{{[0-3]}}
; CHECK:        th.mfadd.s.mm acc{{[0-3]}}, acc{{[0-3]}}, acc{{[0-3]}}
; CHECK:        th.mfmul.s.mm acc{{[0-3]}}, acc{{[0-3]}}, acc{{[0-3]}}
; CHECK:        th.msce32 acc{{[0-3]}}
; CHECK:        ret
  call void @llvm.riscv.th.msettilem.i64(i64 4)
  call void @llvm.riscv.th.msettilen.i64(i64 4)
  %va = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %a, i64 %stride)
  %vb = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %b, i64 %stride)
  %add_r = call target("riscv.matrix") @llvm.riscv.th.mfadd.s.mm.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(
      target("riscv.matrix") %va, target("riscv.matrix") %va, target("riscv.matrix") %vb)
  %mul_r = call target("riscv.matrix") @llvm.riscv.th.mfmul.s.mm.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(
      target("riscv.matrix") %add_r, target("riscv.matrix") %add_r, target("riscv.matrix") %vb)
  call void @llvm.riscv.th.msce.internal32.triscv.matrixt.i64(target("riscv.matrix") %mul_r, ptr %c, i64 %stride)
  ret void
}

;; ============================================================================
;; Test 10: Dual matmul — two independent matmuls, both results stored
;; Both acc values must be live at the store point.
;; ============================================================================
define void @test_dual_matmul(ptr %a, ptr %b1, ptr %b2, ptr %c1, ptr %c2, i64 %stride) {
; CHECK-LABEL: test_dual_matmul:
; CHECK:        th.mlae32 tr{{[0-3]}}
; CHECK:        th.mlbe32 tr{{[0-3]}}
; CHECK:        th.mzero acc{{[0-3]}}
; CHECK:        th.mfmacc.s
; CHECK:        th.mlbe32
; CHECK:        th.mzero
; CHECK:        th.mfmacc.s
; CHECK:        th.msce32
; CHECK:        th.msce32
; CHECK:        ret
  call void @llvm.riscv.th.msettilem.i64(i64 4)
  call void @llvm.riscv.th.msettilek.i64(i64 4)
  call void @llvm.riscv.th.msettilen.i64(i64 4)
  %ta = call target("riscv.matrix") @llvm.riscv.th.mlae.internal32.triscv.matrixt.i64(ptr %a, i64 %stride)
  ; First matmul
  %tb1 = call target("riscv.matrix") @llvm.riscv.th.mlbe.internal32.triscv.matrixt.i64(ptr %b1, i64 %stride)
  %z1 = call target("riscv.matrix") @llvm.riscv.th.mzero.internal.triscv.matrixt()
  %r1 = call target("riscv.matrix") @llvm.riscv.th.mfmacc.s.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(
      target("riscv.matrix") %z1, target("riscv.matrix") %ta, target("riscv.matrix") %tb1)
  ; Second matmul
  %tb2 = call target("riscv.matrix") @llvm.riscv.th.mlbe.internal32.triscv.matrixt.i64(ptr %b2, i64 %stride)
  %z2 = call target("riscv.matrix") @llvm.riscv.th.mzero.internal.triscv.matrixt()
  %r2 = call target("riscv.matrix") @llvm.riscv.th.mfmacc.s.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(
      target("riscv.matrix") %z2, target("riscv.matrix") %ta, target("riscv.matrix") %tb2)
  ; Store both results (r1 and r2 must both be live here)
  call void @llvm.riscv.th.msce.internal32.triscv.matrixt.i64(target("riscv.matrix") %r1, ptr %c1, i64 %stride)
  call void @llvm.riscv.th.msce.internal32.triscv.matrixt.i64(target("riscv.matrix") %r2, ptr %c2, i64 %stride)
  ret void
}

; Declarations
declare void @llvm.riscv.th.msettilem.i64(i64)
declare void @llvm.riscv.th.msettilek.i64(i64)
declare void @llvm.riscv.th.msettilen.i64(i64)
declare target("riscv.matrix") @llvm.riscv.th.mlae.internal8.triscv.matrixt.i64(ptr, i64)
declare target("riscv.matrix") @llvm.riscv.th.mlae.internal16.triscv.matrixt.i64(ptr, i64)
declare target("riscv.matrix") @llvm.riscv.th.mlae.internal32.triscv.matrixt.i64(ptr, i64)
declare target("riscv.matrix") @llvm.riscv.th.mlbe.internal8.triscv.matrixt.i64(ptr, i64)
declare target("riscv.matrix") @llvm.riscv.th.mlbe.internal32.triscv.matrixt.i64(ptr, i64)
declare target("riscv.matrix") @llvm.riscv.th.mlce.internal16.triscv.matrixt.i64(ptr, i64)
declare target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr, i64)
declare target("riscv.matrix") @llvm.riscv.th.mlme.internal32.triscv.matrixt.i64(ptr, i64)
declare void @llvm.riscv.th.msce.internal32.triscv.matrixt.i64(target("riscv.matrix"), ptr, i64)
declare void @llvm.riscv.th.msme.internal32.triscv.matrixt.i64(target("riscv.matrix"), ptr, i64)
declare target("riscv.matrix") @llvm.riscv.th.mzero.internal.triscv.matrixt()
declare target("riscv.matrix") @llvm.riscv.th.mfmacc.s.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix"), target("riscv.matrix"), target("riscv.matrix"))
declare target("riscv.matrix") @llvm.riscv.th.mmacc.w.b.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix"), target("riscv.matrix"), target("riscv.matrix"))
declare target("riscv.matrix") @llvm.riscv.th.madd.w.mm.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix"), target("riscv.matrix"), target("riscv.matrix"))
declare target("riscv.matrix") @llvm.riscv.th.mfadd.s.mm.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix"), target("riscv.matrix"), target("riscv.matrix"))
declare target("riscv.matrix") @llvm.riscv.th.mfmul.s.mm.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix"), target("riscv.matrix"), target("riscv.matrix"))
declare target("riscv.matrix") @llvm.riscv.th.mn4clipl.w.mm.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix"), target("riscv.matrix"), target("riscv.matrix"))
declare target("riscv.matrix") @llvm.riscv.th.mfcvtl.s.h.internal.triscv.matrixt.triscv.matrixt(target("riscv.matrix"))
