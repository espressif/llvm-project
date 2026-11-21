; RUN: llc -mtriple=xtensa -mcpu=esp32s3 -verify-machineinstrs < %s | FileCheck %s

define void @qr_inline_asm() {
; CHECK-LABEL: qr_inline_asm:
; CHECK: ee.vadds.s8 q{{[0-7]}}, q{{[0-7]}}, q{{[0-7]}}
entry:
  %a = tail call <16 x i8> asm sideeffect "ee.zero.q $0", "=q"()
  %b = tail call <16 x i8> asm sideeffect "ee.zero.q $0", "=q"()
  %r = tail call <16 x i8> asm sideeffect "ee.vadds.s8 $0, $1, $2", "=q,q,q"(<16 x i8> %a, <16 x i8> %b)
  ret void
}

define void @qr_vector_memory(i32 %ptr) {
; CHECK-LABEL: qr_vector_memory:
; CHECK: ee.vld.128.ip
; CHECK: ee.vst.128.ip
entry:
  %p = inttoptr i32 %ptr to ptr
  %v = load <16 x i8>, ptr %p, align 16
  %r = tail call <16 x i8> asm sideeffect "ee.vadds.s8 $0, $1, $2", "=q,q,q"(<16 x i8> %v, <16 x i8> %v)
  store <16 x i8> %r, ptr %p, align 16
  ret void
}

define void @qr_vector_constant() {
; CHECK-LABEL: qr_vector_constant:
; CHECK: ee.vld.128.ip
; CHECK: ee.vadds.s8 q{{[0-7]}}, q{{[0-7]}}, q{{[0-7]}}
entry:
  %z = tail call <16 x i8> asm sideeffect "ee.zero.q $0", "=q"()
  %r = tail call <16 x i8> asm sideeffect "ee.vadds.s8 $0, $1, $2", "=q,q,q"(<16 x i8> <i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15, i8 16>, <16 x i8> %z)
  ret void
}

define void @qr_vector_constant_i32x4() {
; CHECK-LABEL: qr_vector_constant_i32x4:
; CHECK: ee.vld.128.ip
; CHECK: ee.vadds.s8 q{{[0-7]}}, q{{[0-7]}}, q{{[0-7]}}
entry:
  %z = tail call <4 x i32> asm sideeffect "ee.zero.q $0", "=q"()
  %r = tail call <4 x i32> asm sideeffect "ee.vadds.s8 $0, $1, $2", "=q,q,q"(<4 x i32> <i32 1, i32 2, i32 3, i32 4>, <4 x i32> %z)
  ret void
}

define void @qr_vector_constant_i16x8() {
; CHECK-LABEL: qr_vector_constant_i16x8:
; CHECK: ee.vadds.s16 q{{[0-7]}}, q{{[0-7]}}, q{{[0-7]}}
entry:
  %z = tail call <8 x i16> asm sideeffect "ee.zero.q $0", "=q"()
  %r = tail call <8 x i16> asm sideeffect "ee.vadds.s16 $0, $1, $2", "=q,q,q"(<8 x i16> <i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7, i16 8>, <8 x i16> %z)
  ret void
}

define void @qr_spill_pressure() {
; CHECK-LABEL: qr_spill_pressure:
; CHECK: ee.movi.32.a
; CHECK: s32i
; CHECK: l32i
; CHECK: ee.movi.32.q
entry:
  %v0 = tail call <16 x i8> asm sideeffect "ee.zero.q $0", "=q"()
  %v1 = tail call <16 x i8> asm sideeffect "ee.zero.q $0", "=q"()
  %v2 = tail call <16 x i8> asm sideeffect "ee.zero.q $0", "=q"()
  %v3 = tail call <16 x i8> asm sideeffect "ee.zero.q $0", "=q"()
  %v4 = tail call <16 x i8> asm sideeffect "ee.zero.q $0", "=q"()
  %v5 = tail call <16 x i8> asm sideeffect "ee.zero.q $0", "=q"()
  %v6 = tail call <16 x i8> asm sideeffect "ee.zero.q $0", "=q"()
  %v7 = tail call <16 x i8> asm sideeffect "ee.zero.q $0", "=q"()
  %v8 = tail call <16 x i8> asm sideeffect "ee.zero.q $0", "=q"()
  %r0 = tail call <16 x i8> asm sideeffect "ee.vadds.s8 $0, $1, $2", "=q,q,q"(<16 x i8> %v0, <16 x i8> %v1)
  %r1 = tail call <16 x i8> asm sideeffect "ee.vadds.s8 $0, $1, $2", "=q,q,q"(<16 x i8> %r0, <16 x i8> %v2)
  %r2 = tail call <16 x i8> asm sideeffect "ee.vadds.s8 $0, $1, $2", "=q,q,q"(<16 x i8> %r1, <16 x i8> %v3)
  %r3 = tail call <16 x i8> asm sideeffect "ee.vadds.s8 $0, $1, $2", "=q,q,q"(<16 x i8> %r2, <16 x i8> %v4)
  %r4 = tail call <16 x i8> asm sideeffect "ee.vadds.s8 $0, $1, $2", "=q,q,q"(<16 x i8> %r3, <16 x i8> %v5)
  %r5 = tail call <16 x i8> asm sideeffect "ee.vadds.s8 $0, $1, $2", "=q,q,q"(<16 x i8> %r4, <16 x i8> %v6)
  %r6 = tail call <16 x i8> asm sideeffect "ee.vadds.s8 $0, $1, $2", "=q,q,q"(<16 x i8> %r5, <16 x i8> %v7)
  %r7 = tail call <16 x i8> asm sideeffect "ee.vadds.s8 $0, $1, $2", "=q,q,q"(<16 x i8> %r6, <16 x i8> %v8)
  ret void
}
