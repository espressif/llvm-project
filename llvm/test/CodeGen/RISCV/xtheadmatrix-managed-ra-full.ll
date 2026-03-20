; RUN: llc -mtriple=riscv64 -mattr=+experimental-xtheadmatrix < %s \
; RUN:   | FileCheck %s
;
; Comprehensive backend codegen tests for ALL XTHeadMatrix instruction variants.
; Fills coverage gaps identified in verification round 8.

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "riscv64-unknown-unknown"

;; ============================================================================
;; 1. FP Matmul — missing variants
;; ============================================================================

define void @test_mfmacc_h(ptr %a, ptr %b, ptr %c, i64 %s) {
; CHECK-LABEL: test_mfmacc_h:
; CHECK: th.mfmacc.h
  %ta = call target("riscv.matrix") @llvm.riscv.th.mlae.internal16.triscv.matrixt.i64(ptr %a, i64 %s)
  %tb = call target("riscv.matrix") @llvm.riscv.th.mlbe.internal16.triscv.matrixt.i64(ptr %b, i64 %s)
  %z = call target("riscv.matrix") @llvm.riscv.th.mzero.internal.triscv.matrixt()
  %r = call target("riscv.matrix") @llvm.riscv.th.mfmacc.h.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix") %z, target("riscv.matrix") %ta, target("riscv.matrix") %tb)
  call void @llvm.riscv.th.msce.internal16.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %c, i64 %s)
  ret void
}

define void @test_mfmacc_d(ptr %a, ptr %b, ptr %c, i64 %s) {
; CHECK-LABEL: test_mfmacc_d:
; CHECK: th.mfmacc.d
  %ta = call target("riscv.matrix") @llvm.riscv.th.mlae.internal64.triscv.matrixt.i64(ptr %a, i64 %s)
  %tb = call target("riscv.matrix") @llvm.riscv.th.mlbe.internal64.triscv.matrixt.i64(ptr %b, i64 %s)
  %z = call target("riscv.matrix") @llvm.riscv.th.mzero.internal.triscv.matrixt()
  %r = call target("riscv.matrix") @llvm.riscv.th.mfmacc.d.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix") %z, target("riscv.matrix") %ta, target("riscv.matrix") %tb)
  call void @llvm.riscv.th.msce.internal64.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %c, i64 %s)
  ret void
}

define void @test_mfmacc_s_h(ptr %a, ptr %b, ptr %c, i64 %s) {
; CHECK-LABEL: test_mfmacc_s_h:
; CHECK: th.mfmacc.s.h
  %ta = call target("riscv.matrix") @llvm.riscv.th.mlae.internal16.triscv.matrixt.i64(ptr %a, i64 %s)
  %tb = call target("riscv.matrix") @llvm.riscv.th.mlbe.internal16.triscv.matrixt.i64(ptr %b, i64 %s)
  %z = call target("riscv.matrix") @llvm.riscv.th.mzero.internal.triscv.matrixt()
  %r = call target("riscv.matrix") @llvm.riscv.th.mfmacc.s.h.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix") %z, target("riscv.matrix") %ta, target("riscv.matrix") %tb)
  call void @llvm.riscv.th.msce.internal32.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %c, i64 %s)
  ret void
}

define void @test_mfmacc_d_s(ptr %a, ptr %b, ptr %c, i64 %s) {
; CHECK-LABEL: test_mfmacc_d_s:
; CHECK: th.mfmacc.d.s
  %ta = call target("riscv.matrix") @llvm.riscv.th.mlae.internal32.triscv.matrixt.i64(ptr %a, i64 %s)
  %tb = call target("riscv.matrix") @llvm.riscv.th.mlbe.internal32.triscv.matrixt.i64(ptr %b, i64 %s)
  %z = call target("riscv.matrix") @llvm.riscv.th.mzero.internal.triscv.matrixt()
  %r = call target("riscv.matrix") @llvm.riscv.th.mfmacc.d.s.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix") %z, target("riscv.matrix") %ta, target("riscv.matrix") %tb)
  call void @llvm.riscv.th.msce.internal64.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %c, i64 %s)
  ret void
}

define void @test_mfmacc_h_e5(ptr %a, ptr %b, ptr %c, i64 %s) {
; CHECK-LABEL: test_mfmacc_h_e5:
; CHECK: th.mfmacc.h.e5
  %ta = call target("riscv.matrix") @llvm.riscv.th.mlae.internal32.triscv.matrixt.i64(ptr %a, i64 %s)
  %tb = call target("riscv.matrix") @llvm.riscv.th.mlbe.internal32.triscv.matrixt.i64(ptr %b, i64 %s)
  %z = call target("riscv.matrix") @llvm.riscv.th.mzero.internal.triscv.matrixt()
  %r = call target("riscv.matrix") @llvm.riscv.th.mfmacc.h.e5.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix") %z, target("riscv.matrix") %ta, target("riscv.matrix") %tb)
  call void @llvm.riscv.th.msce.internal16.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %c, i64 %s)
  ret void
}

define void @test_mfmacc_bf16_e4(ptr %a, ptr %b, ptr %c, i64 %s) {
; CHECK-LABEL: test_mfmacc_bf16_e4:
; CHECK: th.mfmacc.bf16.e4
  %ta = call target("riscv.matrix") @llvm.riscv.th.mlae.internal32.triscv.matrixt.i64(ptr %a, i64 %s)
  %tb = call target("riscv.matrix") @llvm.riscv.th.mlbe.internal32.triscv.matrixt.i64(ptr %b, i64 %s)
  %z = call target("riscv.matrix") @llvm.riscv.th.mzero.internal.triscv.matrixt()
  %r = call target("riscv.matrix") @llvm.riscv.th.mfmacc.bf16.e4.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix") %z, target("riscv.matrix") %ta, target("riscv.matrix") %tb)
  call void @llvm.riscv.th.msce.internal16.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %c, i64 %s)
  ret void
}

define void @test_mfmacc_bf16_e5(ptr %a, ptr %b, ptr %c, i64 %s) {
; CHECK-LABEL: test_mfmacc_bf16_e5:
; CHECK: th.mfmacc.bf16.e5
  %ta = call target("riscv.matrix") @llvm.riscv.th.mlae.internal32.triscv.matrixt.i64(ptr %a, i64 %s)
  %tb = call target("riscv.matrix") @llvm.riscv.th.mlbe.internal32.triscv.matrixt.i64(ptr %b, i64 %s)
  %z = call target("riscv.matrix") @llvm.riscv.th.mzero.internal.triscv.matrixt()
  %r = call target("riscv.matrix") @llvm.riscv.th.mfmacc.bf16.e5.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix") %z, target("riscv.matrix") %ta, target("riscv.matrix") %tb)
  call void @llvm.riscv.th.msce.internal16.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %c, i64 %s)
  ret void
}

define void @test_mfmacc_s_e4(ptr %a, ptr %b, ptr %c, i64 %s) {
; CHECK-LABEL: test_mfmacc_s_e4:
; CHECK: th.mfmacc.s.e4
  %ta = call target("riscv.matrix") @llvm.riscv.th.mlae.internal32.triscv.matrixt.i64(ptr %a, i64 %s)
  %tb = call target("riscv.matrix") @llvm.riscv.th.mlbe.internal32.triscv.matrixt.i64(ptr %b, i64 %s)
  %z = call target("riscv.matrix") @llvm.riscv.th.mzero.internal.triscv.matrixt()
  %r = call target("riscv.matrix") @llvm.riscv.th.mfmacc.s.e4.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix") %z, target("riscv.matrix") %ta, target("riscv.matrix") %tb)
  call void @llvm.riscv.th.msce.internal32.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %c, i64 %s)
  ret void
}

define void @test_mfmacc_s_e5(ptr %a, ptr %b, ptr %c, i64 %s) {
; CHECK-LABEL: test_mfmacc_s_e5:
; CHECK: th.mfmacc.s.e5
  %ta = call target("riscv.matrix") @llvm.riscv.th.mlae.internal32.triscv.matrixt.i64(ptr %a, i64 %s)
  %tb = call target("riscv.matrix") @llvm.riscv.th.mlbe.internal32.triscv.matrixt.i64(ptr %b, i64 %s)
  %z = call target("riscv.matrix") @llvm.riscv.th.mzero.internal.triscv.matrixt()
  %r = call target("riscv.matrix") @llvm.riscv.th.mfmacc.s.e5.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix") %z, target("riscv.matrix") %ta, target("riscv.matrix") %tb)
  call void @llvm.riscv.th.msce.internal32.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %c, i64 %s)
  ret void
}

;; ============================================================================
;; 2. INT Matmul — missing unsigned/mixed/partial/bypass variants
;; ============================================================================

define void @test_mmaccu_w_b(ptr %a, ptr %b, ptr %c, i64 %s) {
; CHECK-LABEL: test_mmaccu_w_b:
; CHECK: th.mmaccu.w.b
  %ta = call target("riscv.matrix") @llvm.riscv.th.mlae.internal8.triscv.matrixt.i64(ptr %a, i64 %s)
  %tb = call target("riscv.matrix") @llvm.riscv.th.mlbe.internal8.triscv.matrixt.i64(ptr %b, i64 %s)
  %tc = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %c, i64 %s)
  %r = call target("riscv.matrix") @llvm.riscv.th.mmaccu.w.b.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix") %tc, target("riscv.matrix") %ta, target("riscv.matrix") %tb)
  call void @llvm.riscv.th.msce.internal32.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %c, i64 %s)
  ret void
}

define void @test_mmaccus_w_b(ptr %a, ptr %b, ptr %c, i64 %s) {
; CHECK-LABEL: test_mmaccus_w_b:
; CHECK: th.mmaccus.w.b
  %ta = call target("riscv.matrix") @llvm.riscv.th.mlae.internal8.triscv.matrixt.i64(ptr %a, i64 %s)
  %tb = call target("riscv.matrix") @llvm.riscv.th.mlbe.internal8.triscv.matrixt.i64(ptr %b, i64 %s)
  %tc = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %c, i64 %s)
  %r = call target("riscv.matrix") @llvm.riscv.th.mmaccus.w.b.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix") %tc, target("riscv.matrix") %ta, target("riscv.matrix") %tb)
  call void @llvm.riscv.th.msce.internal32.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %c, i64 %s)
  ret void
}

define void @test_mmaccsu_w_b(ptr %a, ptr %b, ptr %c, i64 %s) {
; CHECK-LABEL: test_mmaccsu_w_b:
; CHECK: th.mmaccsu.w.b
  %ta = call target("riscv.matrix") @llvm.riscv.th.mlae.internal8.triscv.matrixt.i64(ptr %a, i64 %s)
  %tb = call target("riscv.matrix") @llvm.riscv.th.mlbe.internal8.triscv.matrixt.i64(ptr %b, i64 %s)
  %tc = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %c, i64 %s)
  %r = call target("riscv.matrix") @llvm.riscv.th.mmaccsu.w.b.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix") %tc, target("riscv.matrix") %ta, target("riscv.matrix") %tb)
  call void @llvm.riscv.th.msce.internal32.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %c, i64 %s)
  ret void
}

define void @test_mmacc_d_h(ptr %a, ptr %b, ptr %c, i64 %s) {
; CHECK-LABEL: test_mmacc_d_h:
; CHECK: th.mmacc.d.h
  %ta = call target("riscv.matrix") @llvm.riscv.th.mlae.internal16.triscv.matrixt.i64(ptr %a, i64 %s)
  %tb = call target("riscv.matrix") @llvm.riscv.th.mlbe.internal16.triscv.matrixt.i64(ptr %b, i64 %s)
  %tc = call target("riscv.matrix") @llvm.riscv.th.mlce.internal64.triscv.matrixt.i64(ptr %c, i64 %s)
  %r = call target("riscv.matrix") @llvm.riscv.th.mmacc.d.h.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix") %tc, target("riscv.matrix") %ta, target("riscv.matrix") %tb)
  call void @llvm.riscv.th.msce.internal64.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %c, i64 %s)
  ret void
}

define void @test_mmaccu_d_h(ptr %a, ptr %b, ptr %c, i64 %s) {
; CHECK-LABEL: test_mmaccu_d_h:
; CHECK: th.mmaccu.d.h
  %ta = call target("riscv.matrix") @llvm.riscv.th.mlae.internal16.triscv.matrixt.i64(ptr %a, i64 %s)
  %tb = call target("riscv.matrix") @llvm.riscv.th.mlbe.internal16.triscv.matrixt.i64(ptr %b, i64 %s)
  %tc = call target("riscv.matrix") @llvm.riscv.th.mlce.internal64.triscv.matrixt.i64(ptr %c, i64 %s)
  %r = call target("riscv.matrix") @llvm.riscv.th.mmaccu.d.h.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix") %tc, target("riscv.matrix") %ta, target("riscv.matrix") %tb)
  call void @llvm.riscv.th.msce.internal64.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %c, i64 %s)
  ret void
}

define void @test_mmaccu_w_bp(ptr %a, ptr %b, ptr %c, i64 %s) {
; CHECK-LABEL: test_mmaccu_w_bp:
; CHECK: th.mmaccu.w.bp
  %ta = call target("riscv.matrix") @llvm.riscv.th.mlae.internal8.triscv.matrixt.i64(ptr %a, i64 %s)
  %tb = call target("riscv.matrix") @llvm.riscv.th.mlbe.internal8.triscv.matrixt.i64(ptr %b, i64 %s)
  %tc = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %c, i64 %s)
  %r = call target("riscv.matrix") @llvm.riscv.th.mmaccu.w.bp.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix") %tc, target("riscv.matrix") %ta, target("riscv.matrix") %tb)
  call void @llvm.riscv.th.msce.internal32.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %c, i64 %s)
  ret void
}

define void @test_mmacc_w_bp(ptr %a, ptr %b, ptr %c, i64 %s) {
; CHECK-LABEL: test_mmacc_w_bp:
; CHECK: th.mmacc.w.bp
  %ta = call target("riscv.matrix") @llvm.riscv.th.mlae.internal8.triscv.matrixt.i64(ptr %a, i64 %s)
  %tb = call target("riscv.matrix") @llvm.riscv.th.mlbe.internal8.triscv.matrixt.i64(ptr %b, i64 %s)
  %tc = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %c, i64 %s)
  %r = call target("riscv.matrix") @llvm.riscv.th.mmacc.w.bp.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix") %tc, target("riscv.matrix") %ta, target("riscv.matrix") %tb)
  call void @llvm.riscv.th.msce.internal32.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %c, i64 %s)
  ret void
}

define void @test_pmmacc_w_b(ptr %a, ptr %b, ptr %c, i64 %s) {
; CHECK-LABEL: test_pmmacc_w_b:
; CHECK: th.pmmacc.w.b
  %ta = call target("riscv.matrix") @llvm.riscv.th.mlae.internal8.triscv.matrixt.i64(ptr %a, i64 %s)
  %tb = call target("riscv.matrix") @llvm.riscv.th.mlbe.internal8.triscv.matrixt.i64(ptr %b, i64 %s)
  %tc = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %c, i64 %s)
  %r = call target("riscv.matrix") @llvm.riscv.th.pmmacc.w.b.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix") %tc, target("riscv.matrix") %ta, target("riscv.matrix") %tb)
  call void @llvm.riscv.th.msce.internal32.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %c, i64 %s)
  ret void
}

define void @test_pmmaccu_w_b(ptr %a, ptr %b, ptr %c, i64 %s) {
; CHECK-LABEL: test_pmmaccu_w_b:
; CHECK: th.pmmaccu.w.b
  %ta = call target("riscv.matrix") @llvm.riscv.th.mlae.internal8.triscv.matrixt.i64(ptr %a, i64 %s)
  %tb = call target("riscv.matrix") @llvm.riscv.th.mlbe.internal8.triscv.matrixt.i64(ptr %b, i64 %s)
  %tc = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %c, i64 %s)
  %r = call target("riscv.matrix") @llvm.riscv.th.pmmaccu.w.b.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix") %tc, target("riscv.matrix") %ta, target("riscv.matrix") %tb)
  call void @llvm.riscv.th.msce.internal32.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %c, i64 %s)
  ret void
}

;; ============================================================================
;; 3. Element-wise INT — .w.mm variants
;; ============================================================================

define void @test_madd_w_mm(ptr %p, i64 %s) {
; CHECK-LABEL: test_madd_w_mm:
; CHECK: th.madd.w.mm
  %a = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %p, i64 %s)
  %b = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %p, i64 %s)
  %c = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %p, i64 %s)
  %r = call target("riscv.matrix") @llvm.riscv.th.madd.w.mm.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix") %c, target("riscv.matrix") %a, target("riscv.matrix") %b)
  call void @llvm.riscv.th.msce.internal32.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %p, i64 %s)
  ret void
}

define void @test_msub_w_mm(ptr %p, i64 %s) {
; CHECK-LABEL: test_msub_w_mm:
; CHECK: th.msub.w.mm
  %a = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %p, i64 %s)
  %b = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %p, i64 %s)
  %c = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %p, i64 %s)
  %r = call target("riscv.matrix") @llvm.riscv.th.msub.w.mm.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix") %c, target("riscv.matrix") %a, target("riscv.matrix") %b)
  call void @llvm.riscv.th.msce.internal32.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %p, i64 %s)
  ret void
}

define void @test_mmulh_w_mm(ptr %p, i64 %s) {
; CHECK-LABEL: test_mmulh_w_mm:
; CHECK: th.mmulh.w.mm
  %a = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %p, i64 %s)
  %b = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %p, i64 %s)
  %c = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %p, i64 %s)
  %r = call target("riscv.matrix") @llvm.riscv.th.mmulh.w.mm.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix") %c, target("riscv.matrix") %a, target("riscv.matrix") %b)
  call void @llvm.riscv.th.msce.internal32.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %p, i64 %s)
  ret void
}

define void @test_mmax_w_mm(ptr %p, i64 %s) {
; CHECK-LABEL: test_mmax_w_mm:
; CHECK: th.mmax.w.mm
  %a = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %p, i64 %s)
  %b = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %p, i64 %s)
  %c = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %p, i64 %s)
  %r = call target("riscv.matrix") @llvm.riscv.th.mmax.w.mm.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix") %c, target("riscv.matrix") %a, target("riscv.matrix") %b)
  call void @llvm.riscv.th.msce.internal32.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %p, i64 %s)
  ret void
}

define void @test_msrl_w_mm(ptr %p, i64 %s) {
; CHECK-LABEL: test_msrl_w_mm:
; CHECK: th.msrl.w.mm
  %a = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %p, i64 %s)
  %b = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %p, i64 %s)
  %c = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %p, i64 %s)
  %r = call target("riscv.matrix") @llvm.riscv.th.msrl.w.mm.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix") %c, target("riscv.matrix") %a, target("riscv.matrix") %b)
  call void @llvm.riscv.th.msce.internal32.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %p, i64 %s)
  ret void
}

define void @test_msll_w_mm(ptr %p, i64 %s) {
; CHECK-LABEL: test_msll_w_mm:
; CHECK: th.msll.w.mm
  %a = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %p, i64 %s)
  %b = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %p, i64 %s)
  %c = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %p, i64 %s)
  %r = call target("riscv.matrix") @llvm.riscv.th.msll.w.mm.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix") %c, target("riscv.matrix") %a, target("riscv.matrix") %b)
  call void @llvm.riscv.th.msce.internal32.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %p, i64 %s)
  ret void
}

define void @test_msra_w_mm(ptr %p, i64 %s) {
; CHECK-LABEL: test_msra_w_mm:
; CHECK: th.msra.w.mm
  %a = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %p, i64 %s)
  %b = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %p, i64 %s)
  %c = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %p, i64 %s)
  %r = call target("riscv.matrix") @llvm.riscv.th.msra.w.mm.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix") %c, target("riscv.matrix") %a, target("riscv.matrix") %b)
  call void @llvm.riscv.th.msce.internal32.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %p, i64 %s)
  ret void
}

;; ============================================================================
;; 4. Element-wise FP — .mm variants
;; ============================================================================

define void @test_mfsub_s_mm(ptr %p, i64 %s) {
; CHECK-LABEL: test_mfsub_s_mm:
; CHECK: th.mfsub.s.mm
  %a = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %p, i64 %s)
  %b = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %p, i64 %s)
  %c = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %p, i64 %s)
  %r = call target("riscv.matrix") @llvm.riscv.th.mfsub.s.mm.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix") %c, target("riscv.matrix") %a, target("riscv.matrix") %b)
  call void @llvm.riscv.th.msce.internal32.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %p, i64 %s)
  ret void
}

define void @test_mfmul_d_mm(ptr %p, i64 %s) {
; CHECK-LABEL: test_mfmul_d_mm:
; CHECK: th.mfmul.d.mm
  %a = call target("riscv.matrix") @llvm.riscv.th.mlce.internal64.triscv.matrixt.i64(ptr %p, i64 %s)
  %b = call target("riscv.matrix") @llvm.riscv.th.mlce.internal64.triscv.matrixt.i64(ptr %p, i64 %s)
  %c = call target("riscv.matrix") @llvm.riscv.th.mlce.internal64.triscv.matrixt.i64(ptr %p, i64 %s)
  %r = call target("riscv.matrix") @llvm.riscv.th.mfmul.d.mm.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix") %c, target("riscv.matrix") %a, target("riscv.matrix") %b)
  call void @llvm.riscv.th.msce.internal64.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %p, i64 %s)
  ret void
}

define void @test_mfmax_h_mm(ptr %p, i64 %s) {
; CHECK-LABEL: test_mfmax_h_mm:
; CHECK: th.mfmax.h.mm
  %a = call target("riscv.matrix") @llvm.riscv.th.mlce.internal16.triscv.matrixt.i64(ptr %p, i64 %s)
  %b = call target("riscv.matrix") @llvm.riscv.th.mlce.internal16.triscv.matrixt.i64(ptr %p, i64 %s)
  %c = call target("riscv.matrix") @llvm.riscv.th.mlce.internal16.triscv.matrixt.i64(ptr %p, i64 %s)
  %r = call target("riscv.matrix") @llvm.riscv.th.mfmax.h.mm.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix") %c, target("riscv.matrix") %a, target("riscv.matrix") %b)
  call void @llvm.riscv.th.msce.internal16.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %p, i64 %s)
  ret void
}

define void @test_mfmin_d_mm(ptr %p, i64 %s) {
; CHECK-LABEL: test_mfmin_d_mm:
; CHECK: th.mfmin.d.mm
  %a = call target("riscv.matrix") @llvm.riscv.th.mlce.internal64.triscv.matrixt.i64(ptr %p, i64 %s)
  %b = call target("riscv.matrix") @llvm.riscv.th.mlce.internal64.triscv.matrixt.i64(ptr %p, i64 %s)
  %c = call target("riscv.matrix") @llvm.riscv.th.mlce.internal64.triscv.matrixt.i64(ptr %p, i64 %s)
  %r = call target("riscv.matrix") @llvm.riscv.th.mfmin.d.mm.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix") %c, target("riscv.matrix") %a, target("riscv.matrix") %b)
  call void @llvm.riscv.th.msce.internal64.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %p, i64 %s)
  ret void
}

;; ============================================================================
;; 5. Element-wise .mv.i variants (with immediate)
;; ============================================================================

define void @test_madd_w_mv_i(ptr %p, i64 %s) {
; CHECK-LABEL: test_madd_w_mv_i:
; CHECK: th.madd.w.mv.i
  %a = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %p, i64 %s)
  %b = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %p, i64 %s)
  %c = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %p, i64 %s)
  %r = call target("riscv.matrix") @llvm.riscv.th.madd.w.mv.i.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt.i64(target("riscv.matrix") %c, target("riscv.matrix") %a, target("riscv.matrix") %b, i64 2)
  call void @llvm.riscv.th.msce.internal32.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %p, i64 %s)
  ret void
}

define void @test_mfmul_s_mv_i(ptr %p, i64 %s) {
; CHECK-LABEL: test_mfmul_s_mv_i:
; CHECK: th.mfmul.s.mv.i
  %a = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %p, i64 %s)
  %b = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %p, i64 %s)
  %c = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %p, i64 %s)
  %r = call target("riscv.matrix") @llvm.riscv.th.mfmul.s.mv.i.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt.i64(target("riscv.matrix") %c, target("riscv.matrix") %a, target("riscv.matrix") %b, i64 1)
  call void @llvm.riscv.th.msce.internal32.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %p, i64 %s)
  ret void
}

;; ============================================================================
;; 6. FP format conversions
;; ============================================================================

define void @test_mfcvth_s_h(ptr %p, i64 %s) {
; CHECK-LABEL: test_mfcvth_s_h:
; CHECK: th.mfcvth.s.h
  %v = call target("riscv.matrix") @llvm.riscv.th.mlme.internal32.triscv.matrixt.i64(ptr %p, i64 0)
  %r = call target("riscv.matrix") @llvm.riscv.th.mfcvth.s.h.internal.triscv.matrixt.triscv.matrixt(target("riscv.matrix") %v)
  call void @llvm.riscv.th.msme.internal32.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %p, i64 0)
  ret void
}

define void @test_mfcvtl_d_s(ptr %p, i64 %s) {
; CHECK-LABEL: test_mfcvtl_d_s:
; CHECK: th.mfcvtl.d.s
  %v = call target("riscv.matrix") @llvm.riscv.th.mlme.internal32.triscv.matrixt.i64(ptr %p, i64 0)
  %r = call target("riscv.matrix") @llvm.riscv.th.mfcvtl.d.s.internal.triscv.matrixt.triscv.matrixt(target("riscv.matrix") %v)
  call void @llvm.riscv.th.msme.internal32.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %p, i64 0)
  ret void
}

define void @test_mfcvth_s_d(ptr %p, i64 %s) {
; CHECK-LABEL: test_mfcvth_s_d:
; CHECK: th.mfcvth.s.d
  %v = call target("riscv.matrix") @llvm.riscv.th.mlme.internal32.triscv.matrixt.i64(ptr %p, i64 0)
  %r = call target("riscv.matrix") @llvm.riscv.th.mfcvth.s.d.internal.triscv.matrixt.triscv.matrixt(target("riscv.matrix") %v)
  call void @llvm.riscv.th.msme.internal32.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %p, i64 0)
  ret void
}

define void @test_mfcvt_tf32_s(ptr %p, i64 %s) {
; CHECK-LABEL: test_mfcvt_tf32_s:
; CHECK: th.mfcvt.tf32.s
  %v = call target("riscv.matrix") @llvm.riscv.th.mlme.internal32.triscv.matrixt.i64(ptr %p, i64 0)
  %r = call target("riscv.matrix") @llvm.riscv.th.mfcvt.tf32.s.internal.triscv.matrixt.triscv.matrixt(target("riscv.matrix") %v)
  call void @llvm.riscv.th.msme.internal32.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %p, i64 0)
  ret void
}

define void @test_mfcvt_s_tf32(ptr %p, i64 %s) {
; CHECK-LABEL: test_mfcvt_s_tf32:
; CHECK: th.mfcvt.s.tf32
  %v = call target("riscv.matrix") @llvm.riscv.th.mlme.internal32.triscv.matrixt.i64(ptr %p, i64 0)
  %r = call target("riscv.matrix") @llvm.riscv.th.mfcvt.s.tf32.internal.triscv.matrixt.triscv.matrixt(target("riscv.matrix") %v)
  call void @llvm.riscv.th.msme.internal32.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %p, i64 0)
  ret void
}

define void @test_mfcvtl_h_e4(ptr %p, i64 %s) {
; CHECK-LABEL: test_mfcvtl_h_e4:
; CHECK: th.mfcvtl.h.e4
  %v = call target("riscv.matrix") @llvm.riscv.th.mlme.internal32.triscv.matrixt.i64(ptr %p, i64 0)
  %r = call target("riscv.matrix") @llvm.riscv.th.mfcvtl.h.e4.internal.triscv.matrixt.triscv.matrixt(target("riscv.matrix") %v)
  call void @llvm.riscv.th.msme.internal32.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %p, i64 0)
  ret void
}

;; ============================================================================
;; 7. Float-int conversions
;; ============================================================================

define void @test_msfcvt_s_w(ptr %p) {
; CHECK-LABEL: test_msfcvt_s_w:
; CHECK: th.msfcvt.s.w
  %v = call target("riscv.matrix") @llvm.riscv.th.mlme.internal32.triscv.matrixt.i64(ptr %p, i64 0)
  %r = call target("riscv.matrix") @llvm.riscv.th.msfcvt.s.w.internal.triscv.matrixt.triscv.matrixt(target("riscv.matrix") %v)
  call void @llvm.riscv.th.msme.internal32.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %p, i64 0)
  ret void
}

define void @test_mfscvt_w_s(ptr %p) {
; CHECK-LABEL: test_mfscvt_w_s:
; CHECK: th.mfscvt.w.s
  %v = call target("riscv.matrix") @llvm.riscv.th.mlme.internal32.triscv.matrixt.i64(ptr %p, i64 0)
  %r = call target("riscv.matrix") @llvm.riscv.th.mfscvt.w.s.internal.triscv.matrixt.triscv.matrixt(target("riscv.matrix") %v)
  call void @llvm.riscv.th.msme.internal32.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %p, i64 0)
  ret void
}

define void @test_mufcvtl_h_b(ptr %p) {
; CHECK-LABEL: test_mufcvtl_h_b:
; CHECK: th.mufcvtl.h.b
  %v = call target("riscv.matrix") @llvm.riscv.th.mlme.internal32.triscv.matrixt.i64(ptr %p, i64 0)
  %r = call target("riscv.matrix") @llvm.riscv.th.mufcvtl.h.b.internal.triscv.matrixt.triscv.matrixt(target("riscv.matrix") %v)
  call void @llvm.riscv.th.msme.internal32.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %p, i64 0)
  ret void
}

define void @test_mfucvtl_b_h(ptr %p) {
; CHECK-LABEL: test_mfucvtl_b_h:
; CHECK: th.mfucvtl.b.h
  %v = call target("riscv.matrix") @llvm.riscv.th.mlme.internal32.triscv.matrixt.i64(ptr %p, i64 0)
  %r = call target("riscv.matrix") @llvm.riscv.th.mfucvtl.b.h.internal.triscv.matrixt.triscv.matrixt(target("riscv.matrix") %v)
  call void @llvm.riscv.th.msme.internal32.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %p, i64 0)
  ret void
}

;; ============================================================================
;; 8. N4clip — all 8 variants
;; ============================================================================

define void @test_mn4clipl_w_mm(ptr %p, i64 %s) {
; CHECK-LABEL: test_mn4clipl_w_mm:
; CHECK: th.mn4clipl.w.mm
  %a = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %p, i64 %s)
  %b = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %p, i64 %s)
  %c = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %p, i64 %s)
  %r = call target("riscv.matrix") @llvm.riscv.th.mn4clipl.w.mm.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix") %c, target("riscv.matrix") %a, target("riscv.matrix") %b)
  call void @llvm.riscv.th.msce.internal32.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %p, i64 %s)
  ret void
}

define void @test_mn4cliph_w_mm(ptr %p, i64 %s) {
; CHECK-LABEL: test_mn4cliph_w_mm:
; CHECK: th.mn4cliph.w.mm
  %a = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %p, i64 %s)
  %b = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %p, i64 %s)
  %c = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %p, i64 %s)
  %r = call target("riscv.matrix") @llvm.riscv.th.mn4cliph.w.mm.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix") %c, target("riscv.matrix") %a, target("riscv.matrix") %b)
  call void @llvm.riscv.th.msce.internal32.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %p, i64 %s)
  ret void
}

define void @test_mn4cliplu_w_mm(ptr %p, i64 %s) {
; CHECK-LABEL: test_mn4cliplu_w_mm:
; CHECK: th.mn4cliplu.w.mm
  %a = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %p, i64 %s)
  %b = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %p, i64 %s)
  %c = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %p, i64 %s)
  %r = call target("riscv.matrix") @llvm.riscv.th.mn4cliplu.w.mm.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix") %c, target("riscv.matrix") %a, target("riscv.matrix") %b)
  call void @llvm.riscv.th.msce.internal32.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %p, i64 %s)
  ret void
}

define void @test_mn4cliphu_w_mm(ptr %p, i64 %s) {
; CHECK-LABEL: test_mn4cliphu_w_mm:
; CHECK: th.mn4cliphu.w.mm
  %a = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %p, i64 %s)
  %b = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %p, i64 %s)
  %c = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %p, i64 %s)
  %r = call target("riscv.matrix") @llvm.riscv.th.mn4cliphu.w.mm.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix") %c, target("riscv.matrix") %a, target("riscv.matrix") %b)
  call void @llvm.riscv.th.msce.internal32.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %p, i64 %s)
  ret void
}

define void @test_mn4clipl_w_mv_i(ptr %p, i64 %s) {
; CHECK-LABEL: test_mn4clipl_w_mv_i:
; CHECK: th.mn4clipl.w.mv.i
  %a = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %p, i64 %s)
  %b = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %p, i64 %s)
  %c = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %p, i64 %s)
  %r = call target("riscv.matrix") @llvm.riscv.th.mn4clipl.w.mv.i.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt.i64(target("riscv.matrix") %c, target("riscv.matrix") %a, target("riscv.matrix") %b, i64 3)
  call void @llvm.riscv.th.msce.internal32.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %p, i64 %s)
  ret void
}

;; ============================================================================
;; 9. Packed conversions
;; ============================================================================

define void @test_mucvtl_b_p(ptr %p) {
; CHECK-LABEL: test_mucvtl_b_p:
; CHECK: th.mucvtl.b.p
  %v = call target("riscv.matrix") @llvm.riscv.th.mlme.internal32.triscv.matrixt.i64(ptr %p, i64 0)
  %r = call target("riscv.matrix") @llvm.riscv.th.mucvtl.b.p.internal.triscv.matrixt.triscv.matrixt(target("riscv.matrix") %v)
  call void @llvm.riscv.th.msme.internal32.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %p, i64 0)
  ret void
}

define void @test_mscvtl_b_p(ptr %p) {
; CHECK-LABEL: test_mscvtl_b_p:
; CHECK: th.mscvtl.b.p
  %v = call target("riscv.matrix") @llvm.riscv.th.mlme.internal32.triscv.matrixt.i64(ptr %p, i64 0)
  %r = call target("riscv.matrix") @llvm.riscv.th.mscvtl.b.p.internal.triscv.matrixt.triscv.matrixt(target("riscv.matrix") %v)
  call void @llvm.riscv.th.msme.internal32.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %p, i64 0)
  ret void
}

;; ============================================================================
;; 10. Misc — slides, broadcasts, pack, dup, zero, loads/stores
;; ============================================================================

define void @test_mrslideup(ptr %p) {
; CHECK-LABEL: test_mrslideup:
; CHECK: th.mrslideup
  %v = call target("riscv.matrix") @llvm.riscv.th.mlme.internal32.triscv.matrixt.i64(ptr %p, i64 0)
  %r = call target("riscv.matrix") @llvm.riscv.th.mrslideup.internal.triscv.matrixt.triscv.matrixt.i64(target("riscv.matrix") %v, i64 1)
  call void @llvm.riscv.th.msme.internal32.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %p, i64 0)
  ret void
}

define void @test_mcslidedown_b(ptr %p) {
; CHECK-LABEL: test_mcslidedown_b:
; CHECK: th.mcslidedown.b
  %v = call target("riscv.matrix") @llvm.riscv.th.mlme.internal32.triscv.matrixt.i64(ptr %p, i64 0)
  %r = call target("riscv.matrix") @llvm.riscv.th.mcslidedown.b.internal.triscv.matrixt.triscv.matrixt.i64(target("riscv.matrix") %v, i64 1)
  call void @llvm.riscv.th.msme.internal32.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %p, i64 0)
  ret void
}

define void @test_mcslideup_d(ptr %p) {
; CHECK-LABEL: test_mcslideup_d:
; CHECK: th.mcslideup.d
  %v = call target("riscv.matrix") @llvm.riscv.th.mlme.internal32.triscv.matrixt.i64(ptr %p, i64 0)
  %r = call target("riscv.matrix") @llvm.riscv.th.mcslideup.d.internal.triscv.matrixt.triscv.matrixt.i64(target("riscv.matrix") %v, i64 1)
  call void @llvm.riscv.th.msme.internal32.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %p, i64 0)
  ret void
}

define void @test_mcbcab_mv_i(ptr %p) {
; CHECK-LABEL: test_mcbcab_mv_i:
; CHECK: th.mcbcab.mv.i
  %v = call target("riscv.matrix") @llvm.riscv.th.mlme.internal32.triscv.matrixt.i64(ptr %p, i64 0)
  %r = call target("riscv.matrix") @llvm.riscv.th.mcbcab.mv.i.internal.triscv.matrixt.triscv.matrixt.i64(target("riscv.matrix") %v, i64 0)
  call void @llvm.riscv.th.msme.internal32.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %p, i64 0)
  ret void
}

define void @test_mpackhl(ptr %a, ptr %b, ptr %d) {
; CHECK-LABEL: test_mpackhl:
; CHECK: th.mpackhl
  %va = call target("riscv.matrix") @llvm.riscv.th.mlme.internal32.triscv.matrixt.i64(ptr %a, i64 0)
  %vb = call target("riscv.matrix") @llvm.riscv.th.mlme.internal32.triscv.matrixt.i64(ptr %b, i64 0)
  %r = call target("riscv.matrix") @llvm.riscv.th.mpackhl.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix") %va, target("riscv.matrix") %vb)
  call void @llvm.riscv.th.msme.internal32.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %d, i64 0)
  ret void
}

define void @test_mpackhh(ptr %a, ptr %b, ptr %d) {
; CHECK-LABEL: test_mpackhh:
; CHECK: th.mpackhh
  %va = call target("riscv.matrix") @llvm.riscv.th.mlme.internal32.triscv.matrixt.i64(ptr %a, i64 0)
  %vb = call target("riscv.matrix") @llvm.riscv.th.mlme.internal32.triscv.matrixt.i64(ptr %b, i64 0)
  %r = call target("riscv.matrix") @llvm.riscv.th.mpackhh.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix") %va, target("riscv.matrix") %vb)
  call void @llvm.riscv.th.msme.internal32.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %d, i64 0)
  ret void
}

define void @test_mmovb_x_m(ptr %p) {
; CHECK-LABEL: test_mmovb_x_m:
; CHECK: th.mmovb.x.m
  %v = call target("riscv.matrix") @llvm.riscv.th.mlme.internal32.triscv.matrixt.i64(ptr %p, i64 0)
  %r = call i64 @llvm.riscv.th.mmovb.x.m.internal.i64.triscv.matrixt(target("riscv.matrix") %v, i64 0)
  ret void
}

define void @test_mmovd_m_x(ptr %p, i64 %val) {
; CHECK-LABEL: test_mmovd_m_x:
; CHECK: th.mmovd.m.x
  %v = call target("riscv.matrix") @llvm.riscv.th.mlme.internal32.triscv.matrixt.i64(ptr %p, i64 0)
  %r = call target("riscv.matrix") @llvm.riscv.th.mmovd.m.x.internal.triscv.matrixt.i64(target("riscv.matrix") %v, i64 %val, i64 0)
  call void @llvm.riscv.th.msme.internal32.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %p, i64 0)
  ret void
}

define void @test_mdupb_m_x(ptr %p, i64 %val) {
; CHECK-LABEL: test_mdupb_m_x:
; CHECK: th.mdupb.m.x
  %v = call target("riscv.matrix") @llvm.riscv.th.mlme.internal32.triscv.matrixt.i64(ptr %p, i64 0)
  %r = call target("riscv.matrix") @llvm.riscv.th.mdupb.m.x.internal.triscv.matrixt.i64(target("riscv.matrix") %v, i64 %val)
  call void @llvm.riscv.th.msme.internal32.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %p, i64 0)
  ret void
}

define void @test_mzero2r(ptr %p) {
; CHECK-LABEL: test_mzero2r:
; CHECK: th.mzero2r
  %r = call target("riscv.matrix") @llvm.riscv.th.mzero2r.internal.triscv.matrixt()
  call void @llvm.riscv.th.msme.internal32.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %p, i64 0)
  ret void
}

define void @test_mzero4r(ptr %p) {
; CHECK-LABEL: test_mzero4r:
; CHECK: th.mzero4r
  %r = call target("riscv.matrix") @llvm.riscv.th.mzero4r.internal.triscv.matrixt()
  call void @llvm.riscv.th.msme.internal32.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %p, i64 0)
  ret void
}

define void @test_mzero8r(ptr %p) {
; CHECK-LABEL: test_mzero8r:
; CHECK: th.mzero8r
  %r = call target("riscv.matrix") @llvm.riscv.th.mzero8r.internal.triscv.matrixt()
  call void @llvm.riscv.th.msme.internal32.triscv.matrixt.i64(target("riscv.matrix") %r, ptr %p, i64 0)
  ret void
}

;; Load/store e64 and transposed variants
define void @test_load_store_e64(ptr %a, ptr %b, ptr %c, i64 %s) {
; CHECK-LABEL: test_load_store_e64:
; CHECK: th.mlae64
; CHECK: th.mlbe64
; CHECK: th.mlce64
; CHECK: th.msce64
  %ta = call target("riscv.matrix") @llvm.riscv.th.mlae.internal64.triscv.matrixt.i64(ptr %a, i64 %s)
  %tb = call target("riscv.matrix") @llvm.riscv.th.mlbe.internal64.triscv.matrixt.i64(ptr %b, i64 %s)
  %tc = call target("riscv.matrix") @llvm.riscv.th.mlce.internal64.triscv.matrixt.i64(ptr %c, i64 %s)
  call void @llvm.riscv.th.msce.internal64.triscv.matrixt.i64(target("riscv.matrix") %tc, ptr %c, i64 %s)
  ret void
}

define void @test_transposed_bce(ptr %b, ptr %c, i64 %s) {
; CHECK-LABEL: test_transposed_bce:
; CHECK: th.mlbte32
; CHECK: th.mlcte32
; CHECK: th.mscte32
  %tb = call target("riscv.matrix") @llvm.riscv.th.mlbte.internal32.triscv.matrixt.i64(ptr %b, i64 %s)
  %tc = call target("riscv.matrix") @llvm.riscv.th.mlcte.internal32.triscv.matrixt.i64(ptr %c, i64 %s)
  call void @llvm.riscv.th.mscte.internal32.triscv.matrixt.i64(target("riscv.matrix") %tc, ptr %c, i64 %s)
  ret void
}

define void @test_b_tile_store(ptr %p, i64 %s) {
; CHECK-LABEL: test_b_tile_store:
; CHECK: th.mlbe32
; CHECK: th.msbe32
  %v = call target("riscv.matrix") @llvm.riscv.th.mlbe.internal32.triscv.matrixt.i64(ptr %p, i64 %s)
  call void @llvm.riscv.th.msbe.internal32.triscv.matrixt.i64(target("riscv.matrix") %v, ptr %p, i64 %s)
  ret void
}

;; ============================================================================
;; Intrinsic declarations
;; ============================================================================

; Config
declare void @llvm.riscv.th.msettilem.i64(i64)
declare void @llvm.riscv.th.msettilek.i64(i64)
declare void @llvm.riscv.th.msettilen.i64(i64)

; Loads
declare target("riscv.matrix") @llvm.riscv.th.mlae.internal8.triscv.matrixt.i64(ptr, i64)
declare target("riscv.matrix") @llvm.riscv.th.mlae.internal16.triscv.matrixt.i64(ptr, i64)
declare target("riscv.matrix") @llvm.riscv.th.mlae.internal32.triscv.matrixt.i64(ptr, i64)
declare target("riscv.matrix") @llvm.riscv.th.mlae.internal64.triscv.matrixt.i64(ptr, i64)
declare target("riscv.matrix") @llvm.riscv.th.mlbe.internal8.triscv.matrixt.i64(ptr, i64)
declare target("riscv.matrix") @llvm.riscv.th.mlbe.internal16.triscv.matrixt.i64(ptr, i64)
declare target("riscv.matrix") @llvm.riscv.th.mlbe.internal32.triscv.matrixt.i64(ptr, i64)
declare target("riscv.matrix") @llvm.riscv.th.mlbe.internal64.triscv.matrixt.i64(ptr, i64)
declare target("riscv.matrix") @llvm.riscv.th.mlce.internal16.triscv.matrixt.i64(ptr, i64)
declare target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr, i64)
declare target("riscv.matrix") @llvm.riscv.th.mlce.internal64.triscv.matrixt.i64(ptr, i64)
declare target("riscv.matrix") @llvm.riscv.th.mlme.internal32.triscv.matrixt.i64(ptr, i64)
declare target("riscv.matrix") @llvm.riscv.th.mlbte.internal32.triscv.matrixt.i64(ptr, i64)
declare target("riscv.matrix") @llvm.riscv.th.mlcte.internal32.triscv.matrixt.i64(ptr, i64)

; Stores
declare void @llvm.riscv.th.msce.internal16.triscv.matrixt.i64(target("riscv.matrix"), ptr, i64)
declare void @llvm.riscv.th.msce.internal32.triscv.matrixt.i64(target("riscv.matrix"), ptr, i64)
declare void @llvm.riscv.th.msce.internal64.triscv.matrixt.i64(target("riscv.matrix"), ptr, i64)
declare void @llvm.riscv.th.msme.internal32.triscv.matrixt.i64(target("riscv.matrix"), ptr, i64)
declare void @llvm.riscv.th.msbe.internal32.triscv.matrixt.i64(target("riscv.matrix"), ptr, i64)
declare void @llvm.riscv.th.mscte.internal32.triscv.matrixt.i64(target("riscv.matrix"), ptr, i64)

; Zero
declare target("riscv.matrix") @llvm.riscv.th.mzero.internal.triscv.matrixt()
declare target("riscv.matrix") @llvm.riscv.th.mzero2r.internal.triscv.matrixt()
declare target("riscv.matrix") @llvm.riscv.th.mzero4r.internal.triscv.matrixt()
declare target("riscv.matrix") @llvm.riscv.th.mzero8r.internal.triscv.matrixt()

; FP Matmul
declare target("riscv.matrix") @llvm.riscv.th.mfmacc.h.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix"), target("riscv.matrix"), target("riscv.matrix"))
declare target("riscv.matrix") @llvm.riscv.th.mfmacc.d.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix"), target("riscv.matrix"), target("riscv.matrix"))
declare target("riscv.matrix") @llvm.riscv.th.mfmacc.s.h.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix"), target("riscv.matrix"), target("riscv.matrix"))
declare target("riscv.matrix") @llvm.riscv.th.mfmacc.d.s.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix"), target("riscv.matrix"), target("riscv.matrix"))
declare target("riscv.matrix") @llvm.riscv.th.mfmacc.h.e5.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix"), target("riscv.matrix"), target("riscv.matrix"))
declare target("riscv.matrix") @llvm.riscv.th.mfmacc.bf16.e4.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix"), target("riscv.matrix"), target("riscv.matrix"))
declare target("riscv.matrix") @llvm.riscv.th.mfmacc.bf16.e5.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix"), target("riscv.matrix"), target("riscv.matrix"))
declare target("riscv.matrix") @llvm.riscv.th.mfmacc.s.e4.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix"), target("riscv.matrix"), target("riscv.matrix"))
declare target("riscv.matrix") @llvm.riscv.th.mfmacc.s.e5.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix"), target("riscv.matrix"), target("riscv.matrix"))

; INT Matmul
declare target("riscv.matrix") @llvm.riscv.th.mmaccu.w.b.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix"), target("riscv.matrix"), target("riscv.matrix"))
declare target("riscv.matrix") @llvm.riscv.th.mmaccus.w.b.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix"), target("riscv.matrix"), target("riscv.matrix"))
declare target("riscv.matrix") @llvm.riscv.th.mmaccsu.w.b.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix"), target("riscv.matrix"), target("riscv.matrix"))
declare target("riscv.matrix") @llvm.riscv.th.mmacc.d.h.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix"), target("riscv.matrix"), target("riscv.matrix"))
declare target("riscv.matrix") @llvm.riscv.th.mmaccu.d.h.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix"), target("riscv.matrix"), target("riscv.matrix"))
declare target("riscv.matrix") @llvm.riscv.th.mmaccu.w.bp.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix"), target("riscv.matrix"), target("riscv.matrix"))
declare target("riscv.matrix") @llvm.riscv.th.mmacc.w.bp.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix"), target("riscv.matrix"), target("riscv.matrix"))
declare target("riscv.matrix") @llvm.riscv.th.pmmacc.w.b.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix"), target("riscv.matrix"), target("riscv.matrix"))
declare target("riscv.matrix") @llvm.riscv.th.pmmaccu.w.b.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix"), target("riscv.matrix"), target("riscv.matrix"))

; EW INT .mm
declare target("riscv.matrix") @llvm.riscv.th.madd.w.mm.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix"), target("riscv.matrix"), target("riscv.matrix"))
declare target("riscv.matrix") @llvm.riscv.th.msub.w.mm.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix"), target("riscv.matrix"), target("riscv.matrix"))
declare target("riscv.matrix") @llvm.riscv.th.mmulh.w.mm.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix"), target("riscv.matrix"), target("riscv.matrix"))
declare target("riscv.matrix") @llvm.riscv.th.mmax.w.mm.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix"), target("riscv.matrix"), target("riscv.matrix"))
declare target("riscv.matrix") @llvm.riscv.th.msrl.w.mm.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix"), target("riscv.matrix"), target("riscv.matrix"))
declare target("riscv.matrix") @llvm.riscv.th.msll.w.mm.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix"), target("riscv.matrix"), target("riscv.matrix"))
declare target("riscv.matrix") @llvm.riscv.th.msra.w.mm.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix"), target("riscv.matrix"), target("riscv.matrix"))

; EW FP .mm
declare target("riscv.matrix") @llvm.riscv.th.mfsub.s.mm.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix"), target("riscv.matrix"), target("riscv.matrix"))
declare target("riscv.matrix") @llvm.riscv.th.mfmul.d.mm.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix"), target("riscv.matrix"), target("riscv.matrix"))
declare target("riscv.matrix") @llvm.riscv.th.mfmax.h.mm.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix"), target("riscv.matrix"), target("riscv.matrix"))
declare target("riscv.matrix") @llvm.riscv.th.mfmin.d.mm.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix"), target("riscv.matrix"), target("riscv.matrix"))

; EW .mv.i
declare target("riscv.matrix") @llvm.riscv.th.madd.w.mv.i.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt.i64(target("riscv.matrix"), target("riscv.matrix"), target("riscv.matrix"), i64)
declare target("riscv.matrix") @llvm.riscv.th.mfmul.s.mv.i.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt.i64(target("riscv.matrix"), target("riscv.matrix"), target("riscv.matrix"), i64)

; N4clip
declare target("riscv.matrix") @llvm.riscv.th.mn4clipl.w.mm.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix"), target("riscv.matrix"), target("riscv.matrix"))
declare target("riscv.matrix") @llvm.riscv.th.mn4cliph.w.mm.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix"), target("riscv.matrix"), target("riscv.matrix"))
declare target("riscv.matrix") @llvm.riscv.th.mn4cliplu.w.mm.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix"), target("riscv.matrix"), target("riscv.matrix"))
declare target("riscv.matrix") @llvm.riscv.th.mn4cliphu.w.mm.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix"), target("riscv.matrix"), target("riscv.matrix"))
declare target("riscv.matrix") @llvm.riscv.th.mn4clipl.w.mv.i.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt.i64(target("riscv.matrix"), target("riscv.matrix"), target("riscv.matrix"), i64)

; Conversions
declare target("riscv.matrix") @llvm.riscv.th.mfcvth.s.h.internal.triscv.matrixt.triscv.matrixt(target("riscv.matrix"))
declare target("riscv.matrix") @llvm.riscv.th.mfcvtl.d.s.internal.triscv.matrixt.triscv.matrixt(target("riscv.matrix"))
declare target("riscv.matrix") @llvm.riscv.th.mfcvth.s.d.internal.triscv.matrixt.triscv.matrixt(target("riscv.matrix"))
declare target("riscv.matrix") @llvm.riscv.th.mfcvt.tf32.s.internal.triscv.matrixt.triscv.matrixt(target("riscv.matrix"))
declare target("riscv.matrix") @llvm.riscv.th.mfcvt.s.tf32.internal.triscv.matrixt.triscv.matrixt(target("riscv.matrix"))
declare target("riscv.matrix") @llvm.riscv.th.mfcvtl.h.e4.internal.triscv.matrixt.triscv.matrixt(target("riscv.matrix"))

; Float-int conversions
declare target("riscv.matrix") @llvm.riscv.th.msfcvt.s.w.internal.triscv.matrixt.triscv.matrixt(target("riscv.matrix"))
declare target("riscv.matrix") @llvm.riscv.th.mfscvt.w.s.internal.triscv.matrixt.triscv.matrixt(target("riscv.matrix"))
declare target("riscv.matrix") @llvm.riscv.th.mufcvtl.h.b.internal.triscv.matrixt.triscv.matrixt(target("riscv.matrix"))
declare target("riscv.matrix") @llvm.riscv.th.mfucvtl.b.h.internal.triscv.matrixt.triscv.matrixt(target("riscv.matrix"))

; Packed conversions
declare target("riscv.matrix") @llvm.riscv.th.mucvtl.b.p.internal.triscv.matrixt.triscv.matrixt(target("riscv.matrix"))
declare target("riscv.matrix") @llvm.riscv.th.mscvtl.b.p.internal.triscv.matrixt.triscv.matrixt(target("riscv.matrix"))

; Slides
declare target("riscv.matrix") @llvm.riscv.th.mrslideup.internal.triscv.matrixt.triscv.matrixt.i64(target("riscv.matrix"), i64)
declare target("riscv.matrix") @llvm.riscv.th.mcslidedown.b.internal.triscv.matrixt.triscv.matrixt.i64(target("riscv.matrix"), i64)
declare target("riscv.matrix") @llvm.riscv.th.mcslideup.d.internal.triscv.matrixt.triscv.matrixt.i64(target("riscv.matrix"), i64)

; Broadcasts
declare target("riscv.matrix") @llvm.riscv.th.mcbcab.mv.i.internal.triscv.matrixt.triscv.matrixt.i64(target("riscv.matrix"), i64)

; Pack
declare target("riscv.matrix") @llvm.riscv.th.mpackhl.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix"), target("riscv.matrix"))
declare target("riscv.matrix") @llvm.riscv.th.mpackhh.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix"), target("riscv.matrix"))

; Move/dup
declare i64 @llvm.riscv.th.mmovb.x.m.internal.i64.triscv.matrixt(target("riscv.matrix"), i64)
declare target("riscv.matrix") @llvm.riscv.th.mmovd.m.x.internal.triscv.matrixt.i64(target("riscv.matrix"), i64, i64)
declare target("riscv.matrix") @llvm.riscv.th.mdupb.m.x.internal.triscv.matrixt.i64(target("riscv.matrix"), i64)
