; RUN: llc -mtriple=riscv64 -mattr=+experimental-xtheadmatrix < %s \
; RUN:   | FileCheck %s
;
; Tests for the ManagedRA (register-allocator-managed) programming model.
; These use _internal intrinsics that return/consume target("riscv.matrix")
; SSA values, and the register allocator assigns tr0-tr3/acc0-acc3.

target datalayout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
target triple = "riscv64-unknown-unknown"

;; ============================================================================
;; Test 1: Basic load-matmul-store (the core kernel pattern)
;; ============================================================================
define void @test_matmul_basic(ptr %a_ptr, ptr %b_ptr, ptr %c_ptr, i64 %stride) {
; CHECK-LABEL: test_matmul_basic:
; CHECK:       # %bb.0:
; CHECK-DAG:    th.msettilem
; CHECK-DAG:    th.msettilek
; CHECK-DAG:    th.msettilen
; CHECK:        th.mlae32
; CHECK:        th.mlbe32
; CHECK:        th.mlce32
; CHECK:        th.mfmacc.s
; CHECK:        th.msce32
; CHECK:        ret
  ; Configure tile dimensions
  call void @llvm.riscv.th.msettilem.i64(i64 4)
  call void @llvm.riscv.th.msettilek.i64(i64 4)
  call void @llvm.riscv.th.msettilen.i64(i64 4)
  ; Load tile A (tile register)
  %a = call target("riscv.matrix") @llvm.riscv.th.mlae.internal32.triscv.matrixt.i64(ptr %a_ptr, i64 %stride)
  ; Load tile B (tile register)
  %b = call target("riscv.matrix") @llvm.riscv.th.mlbe.internal32.triscv.matrixt.i64(ptr %b_ptr, i64 %stride)
  ; Load accumulator C
  %c = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %c_ptr, i64 %stride)
  ; Matmul: c = c + a * b
  %result = call target("riscv.matrix") @llvm.riscv.th.mfmacc.s.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(
      target("riscv.matrix") %c, target("riscv.matrix") %a, target("riscv.matrix") %b)
  ; Store result
  call void @llvm.riscv.th.msce.internal32.triscv.matrixt.i64(
      target("riscv.matrix") %result, ptr %c_ptr, i64 %stride)
  ret void
}

;; ============================================================================
;; Test 2: Zero initialization
;; ============================================================================
define void @test_zero(ptr %c_ptr, i64 %stride) {
; CHECK-LABEL: test_zero:
; CHECK:        th.mzero
; CHECK:        th.msce32
; CHECK:        ret
  call void @llvm.riscv.th.msettilen.i64(i64 4)
  call void @llvm.riscv.th.msettilem.i64(i64 4)
  %z = call target("riscv.matrix") @llvm.riscv.th.mzero.internal.triscv.matrixt()
  call void @llvm.riscv.th.msce.internal32.triscv.matrixt.i64(
      target("riscv.matrix") %z, ptr %c_ptr, i64 %stride)
  ret void
}

;; ============================================================================
;; Test 3: INT8 matmul (signed*signed)
;; ============================================================================
define void @test_int_matmul(ptr %a_ptr, ptr %b_ptr, ptr %c_ptr, i64 %stride) {
; CHECK-LABEL: test_int_matmul:
; CHECK:        th.mlae8
; CHECK:        th.mlbe8
; CHECK:        th.mzero
; CHECK:        th.mmacc.w.b
; CHECK:        th.msce32
; CHECK:        ret
  call void @llvm.riscv.th.msettilem.i64(i64 4)
  call void @llvm.riscv.th.msettilek.i64(i64 16)
  call void @llvm.riscv.th.msettilen.i64(i64 4)
  %a = call target("riscv.matrix") @llvm.riscv.th.mlae.internal8.triscv.matrixt.i64(ptr %a_ptr, i64 %stride)
  %b = call target("riscv.matrix") @llvm.riscv.th.mlbe.internal8.triscv.matrixt.i64(ptr %b_ptr, i64 %stride)
  %zero = call target("riscv.matrix") @llvm.riscv.th.mzero.internal.triscv.matrixt()
  %result = call target("riscv.matrix") @llvm.riscv.th.mmacc.w.b.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(
      target("riscv.matrix") %zero, target("riscv.matrix") %a, target("riscv.matrix") %b)
  call void @llvm.riscv.th.msce.internal32.triscv.matrixt.i64(
      target("riscv.matrix") %result, ptr %c_ptr, i64 %stride)
  ret void
}

;; ============================================================================
;; Test 4: Conversion (FP format)
;; ============================================================================
define void @test_conversion(ptr %src_ptr, ptr %dst_ptr, i64 %stride) {
; CHECK-LABEL: test_conversion:
; CHECK:        th.mlae16
; CHECK:        th.mfcvtl.s.h
; CHECK:        th.msce32
; CHECK:        ret
  call void @llvm.riscv.th.msettilem.i64(i64 4)
  call void @llvm.riscv.th.msettilek.i64(i64 4)
  call void @llvm.riscv.th.msettilen.i64(i64 4)
  %src = call target("riscv.matrix") @llvm.riscv.th.mlae.internal16.triscv.matrixt.i64(ptr %src_ptr, i64 %stride)
  %cvt = call target("riscv.matrix") @llvm.riscv.th.mfcvtl.s.h.internal.triscv.matrixt.triscv.matrixt(
      target("riscv.matrix") %src)
  call void @llvm.riscv.th.msce.internal32.triscv.matrixt.i64(
      target("riscv.matrix") %cvt, ptr %dst_ptr, i64 %stride)
  ret void
}

;; ============================================================================
;; Test 5: Matrix move
;; ============================================================================
define void @test_mmov(ptr %src_ptr, ptr %dst_ptr, i64 %stride) {
; CHECK-LABEL: test_mmov:
; CHECK:        th.mlme32
; CHECK:        th.mmov.mm
; CHECK:        th.msme32
; CHECK:        ret
  %src = call target("riscv.matrix") @llvm.riscv.th.mlme.internal32.triscv.matrixt.i64(ptr %src_ptr, i64 0)
  %copy = call target("riscv.matrix") @llvm.riscv.th.mmov.mm.internal.triscv.matrixt.triscv.matrixt(
      target("riscv.matrix") %src)
  call void @llvm.riscv.th.msme.internal32.triscv.matrixt.i64(
      target("riscv.matrix") %copy, ptr %dst_ptr, i64 0)
  ret void
}

;; ============================================================================
;; Test 6: Element-wise addition (FP32)
;; ============================================================================
define void @test_ew_add(ptr %a_ptr, ptr %b_ptr, ptr %c_ptr, i64 %stride) {
; CHECK-LABEL: test_ew_add:
; CHECK:        th.mlce32
; CHECK:        th.mlce32
; CHECK:        th.mlce32
; CHECK:        th.mfadd.s.mm
; CHECK:        th.msce32
; CHECK:        ret
  call void @llvm.riscv.th.msettilem.i64(i64 4)
  call void @llvm.riscv.th.msettilen.i64(i64 4)
  %a = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %a_ptr, i64 %stride)
  %b = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %b_ptr, i64 %stride)
  %c = call target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr %c_ptr, i64 %stride)
  %result = call target("riscv.matrix") @llvm.riscv.th.mfadd.s.mm.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(
      target("riscv.matrix") %c, target("riscv.matrix") %a, target("riscv.matrix") %b)
  call void @llvm.riscv.th.msce.internal32.triscv.matrixt.i64(
      target("riscv.matrix") %result, ptr %c_ptr, i64 %stride)
  ret void
}

;; ============================================================================
;; Test 7: Pack operation
;; ============================================================================
define void @test_pack(ptr %s1_ptr, ptr %s2_ptr, ptr %dst_ptr, i64 %stride) {
; CHECK-LABEL: test_pack:
; CHECK:        th.mlme32
; CHECK:        th.mlme32
; CHECK:        th.mpack
; CHECK:        th.msme32
; CHECK:        ret
  %s1 = call target("riscv.matrix") @llvm.riscv.th.mlme.internal32.triscv.matrixt.i64(ptr %s1_ptr, i64 0)
  %s2 = call target("riscv.matrix") @llvm.riscv.th.mlme.internal32.triscv.matrixt.i64(ptr %s2_ptr, i64 0)
  %packed = call target("riscv.matrix") @llvm.riscv.th.mpack.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(
      target("riscv.matrix") %s1, target("riscv.matrix") %s2)
  call void @llvm.riscv.th.msme.internal32.triscv.matrixt.i64(
      target("riscv.matrix") %packed, ptr %dst_ptr, i64 0)
  ret void
}

;; ============================================================================
;; Test 8: Widening FP matmul — FP8 -> FP16 (h_e4)
;; Verifies proper register class allocation: acc in THRVMACC, a/b in THRVMTR
;; ============================================================================
define void @test_widen_h_e4(ptr %a_ptr, ptr %b_ptr, ptr %c_ptr, i64 %stride) {
; CHECK-LABEL: test_widen_h_e4:
; CHECK:        th.mlae32
; CHECK:        th.mlbe32
; CHECK:        th.mzero
; CHECK:        th.mfmacc.h.e4
; CHECK:        th.msce16
; CHECK:        ret
  call void @llvm.riscv.th.msettilem.i64(i64 4)
  call void @llvm.riscv.th.msettilek.i64(i64 8)
  call void @llvm.riscv.th.msettilen.i64(i64 4)
  %a = call target("riscv.matrix") @llvm.riscv.th.mlae.internal32.triscv.matrixt.i64(ptr %a_ptr, i64 %stride)
  %b = call target("riscv.matrix") @llvm.riscv.th.mlbe.internal32.triscv.matrixt.i64(ptr %b_ptr, i64 %stride)
  %zero = call target("riscv.matrix") @llvm.riscv.th.mzero.internal.triscv.matrixt()
  %result = call target("riscv.matrix") @llvm.riscv.th.mfmacc.h.e4.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(
      target("riscv.matrix") %zero, target("riscv.matrix") %a, target("riscv.matrix") %b)
  call void @llvm.riscv.th.msce.internal16.triscv.matrixt.i64(
      target("riscv.matrix") %result, ptr %c_ptr, i64 %stride)
  ret void
}

;; ============================================================================
;; Test 9: Widening FP matmul — BF16 -> FP32 (s_bf16)
;; ============================================================================
define void @test_widen_s_bf16(ptr %a_ptr, ptr %b_ptr, ptr %c_ptr, i64 %stride) {
; CHECK-LABEL: test_widen_s_bf16:
; CHECK:        th.mlae32
; CHECK:        th.mlbe32
; CHECK:        th.mzero
; CHECK:        th.mfmacc.s.bf16
; CHECK:        th.msce32
; CHECK:        ret
  call void @llvm.riscv.th.msettilem.i64(i64 4)
  call void @llvm.riscv.th.msettilek.i64(i64 8)
  call void @llvm.riscv.th.msettilen.i64(i64 4)
  %a = call target("riscv.matrix") @llvm.riscv.th.mlae.internal32.triscv.matrixt.i64(ptr %a_ptr, i64 %stride)
  %b = call target("riscv.matrix") @llvm.riscv.th.mlbe.internal32.triscv.matrixt.i64(ptr %b_ptr, i64 %stride)
  %zero = call target("riscv.matrix") @llvm.riscv.th.mzero.internal.triscv.matrixt()
  %result = call target("riscv.matrix") @llvm.riscv.th.mfmacc.s.bf16.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(
      target("riscv.matrix") %zero, target("riscv.matrix") %a, target("riscv.matrix") %b)
  call void @llvm.riscv.th.msce.internal32.triscv.matrixt.i64(
      target("riscv.matrix") %result, ptr %c_ptr, i64 %stride)
  ret void
}

;; ============================================================================
;; Test 10: Widening FP matmul — TF32 -> FP32 (s_tf32)
;; ============================================================================
define void @test_widen_s_tf32(ptr %a_ptr, ptr %b_ptr, ptr %c_ptr, i64 %stride) {
; CHECK-LABEL: test_widen_s_tf32:
; CHECK:        th.mlae32
; CHECK:        th.mlbe32
; CHECK:        th.mzero
; CHECK:        th.mfmacc.s.tf32
; CHECK:        th.msce32
; CHECK:        ret
  call void @llvm.riscv.th.msettilem.i64(i64 4)
  call void @llvm.riscv.th.msettilek.i64(i64 4)
  call void @llvm.riscv.th.msettilen.i64(i64 4)
  %a = call target("riscv.matrix") @llvm.riscv.th.mlae.internal32.triscv.matrixt.i64(ptr %a_ptr, i64 %stride)
  %b = call target("riscv.matrix") @llvm.riscv.th.mlbe.internal32.triscv.matrixt.i64(ptr %b_ptr, i64 %stride)
  %zero = call target("riscv.matrix") @llvm.riscv.th.mzero.internal.triscv.matrixt()
  %result = call target("riscv.matrix") @llvm.riscv.th.mfmacc.s.tf32.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(
      target("riscv.matrix") %zero, target("riscv.matrix") %a, target("riscv.matrix") %b)
  call void @llvm.riscv.th.msce.internal32.triscv.matrixt.i64(
      target("riscv.matrix") %result, ptr %c_ptr, i64 %stride)
  ret void
}

;; ============================================================================
;; Intrinsic declarations
;; ============================================================================

; Config
declare void @llvm.riscv.th.msettilem.i64(i64)
declare void @llvm.riscv.th.msettilek.i64(i64)
declare void @llvm.riscv.th.msettilen.i64(i64)

; Loads (_internal)
declare target("riscv.matrix") @llvm.riscv.th.mlae.internal8.triscv.matrixt.i64(ptr, i64)
declare target("riscv.matrix") @llvm.riscv.th.mlae.internal16.triscv.matrixt.i64(ptr, i64)
declare target("riscv.matrix") @llvm.riscv.th.mlae.internal32.triscv.matrixt.i64(ptr, i64)
declare target("riscv.matrix") @llvm.riscv.th.mlbe.internal8.triscv.matrixt.i64(ptr, i64)
declare target("riscv.matrix") @llvm.riscv.th.mlbe.internal32.triscv.matrixt.i64(ptr, i64)
declare target("riscv.matrix") @llvm.riscv.th.mlce.internal32.triscv.matrixt.i64(ptr, i64)
declare target("riscv.matrix") @llvm.riscv.th.mlme.internal32.triscv.matrixt.i64(ptr, i64)

; Stores (_internal)
declare void @llvm.riscv.th.msce.internal16.triscv.matrixt.i64(target("riscv.matrix"), ptr, i64)
declare void @llvm.riscv.th.msce.internal32.triscv.matrixt.i64(target("riscv.matrix"), ptr, i64)
declare void @llvm.riscv.th.msme.internal32.triscv.matrixt.i64(target("riscv.matrix"), ptr, i64)

; Matmul (_internal)
declare target("riscv.matrix") @llvm.riscv.th.mfmacc.s.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix"), target("riscv.matrix"), target("riscv.matrix"))
declare target("riscv.matrix") @llvm.riscv.th.mmacc.w.b.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix"), target("riscv.matrix"), target("riscv.matrix"))
declare target("riscv.matrix") @llvm.riscv.th.mfmacc.h.e4.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix"), target("riscv.matrix"), target("riscv.matrix"))
declare target("riscv.matrix") @llvm.riscv.th.mfmacc.s.bf16.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix"), target("riscv.matrix"), target("riscv.matrix"))
declare target("riscv.matrix") @llvm.riscv.th.mfmacc.s.tf32.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix"), target("riscv.matrix"), target("riscv.matrix"))

; Zero
declare target("riscv.matrix") @llvm.riscv.th.mzero.internal.triscv.matrixt()

; Move
declare target("riscv.matrix") @llvm.riscv.th.mmov.mm.internal.triscv.matrixt.triscv.matrixt(target("riscv.matrix"))

; Conversion
declare target("riscv.matrix") @llvm.riscv.th.mfcvtl.s.h.internal.triscv.matrixt.triscv.matrixt(target("riscv.matrix"))

; EW arithmetic
declare target("riscv.matrix") @llvm.riscv.th.mfadd.s.mm.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix"), target("riscv.matrix"), target("riscv.matrix"))

; Pack
declare target("riscv.matrix") @llvm.riscv.th.mpack.internal.triscv.matrixt.triscv.matrixt.triscv.matrixt(target("riscv.matrix"), target("riscv.matrix"))

attributes #0 = { nounwind }
