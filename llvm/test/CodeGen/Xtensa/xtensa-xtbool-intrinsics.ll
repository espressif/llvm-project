; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py UTC_ARGS: --version 5
; RUN: llc -mtriple=xtensa -mcpu=esp32 %s -o - | FileCheck %s

define float @test_xtbool_movt(float %a, float %b, <1 x i1> %c)  {
; CHECK-LABEL: test_xtbool_movt:
; CHECK:       # %bb.0:
; CHECK-NEXT:    entry a1, 32
; CHECK-NEXT:    wfr f8, a3
; CHECK-NEXT:    wfr f9, a2
; CHECK-NEXT:    movt.s f9, f8, b0
; CHECK-NEXT:    rfr a2, f9
; CHECK-NEXT:    retw.n
  %f = call float @llvm.xtensa.xt.movt.s(float %a, float %b, <1 x i1> %c)
  ret float %f
}

define float @test_xtbool_movf(float %a, float %b, <1 x i1> %c)  {
; CHECK-LABEL: test_xtbool_movf:
; CHECK:       # %bb.0:
; CHECK-NEXT:    entry a1, 32
; CHECK-NEXT:    wfr f8, a3
; CHECK-NEXT:    wfr f9, a2
; CHECK-NEXT:    movf.s f9, f8, b0
; CHECK-NEXT:    rfr a2, f9
; CHECK-NEXT:    retw.n
  %f = call float @llvm.xtensa.xt.movf.s(float %a, float %b, <1 x i1> %c)
  ret float %f
}

define <1 x i1> @test_xtbool_oeq_s(float %a, float %b)  {
; CHECK-LABEL: test_xtbool_oeq_s:
; CHECK:       # %bb.0:
; CHECK-NEXT:    entry a1, 32
; CHECK-NEXT:    wfr f8, a3
; CHECK-NEXT:    wfr f9, a2
; CHECK-NEXT:    oeq.s b0, f9, f8
; CHECK-NEXT:    retw.n
  %r = call <1 x i1> @llvm.xtensa.xt.oeq.s(float %a, float %b)
  ret <1 x i1> %r
}

define <1 x i1> @test_xtbool_ueq_s(float %a, float %b)  {
; CHECK-LABEL: test_xtbool_ueq_s:
; CHECK:       # %bb.0:
; CHECK-NEXT:    entry a1, 32
; CHECK-NEXT:    wfr f8, a3
; CHECK-NEXT:    wfr f9, a2
; CHECK-NEXT:    ueq.s b0, f9, f8
; CHECK-NEXT:    retw.n
  %r = call <1 x i1> @llvm.xtensa.xt.ueq.s(float %a, float %b)
  ret <1 x i1> %r
}

define <1 x i1> @test_xtbool_ole_s(float %a, float %b)  {
; CHECK-LABEL: test_xtbool_ole_s:
; CHECK:       # %bb.0:
; CHECK-NEXT:    entry a1, 32
; CHECK-NEXT:    wfr f8, a3
; CHECK-NEXT:    wfr f9, a2
; CHECK-NEXT:    ole.s b0, f9, f8
; CHECK-NEXT:    retw.n
  %r = call <1 x i1> @llvm.xtensa.xt.ole.s(float %a, float %b)
  ret <1 x i1> %r
}

define <1 x i1> @test_xtbool_ule_s(float %a, float %b)  {
; CHECK-LABEL: test_xtbool_ule_s:
; CHECK:       # %bb.0:
; CHECK-NEXT:    entry a1, 32
; CHECK-NEXT:    wfr f8, a3
; CHECK-NEXT:    wfr f9, a2
; CHECK-NEXT:    ule.s b0, f9, f8
; CHECK-NEXT:    retw.n
  %r = call <1 x i1> @llvm.xtensa.xt.ule.s(float %a, float %b)
  ret <1 x i1> %r
}

define <1 x i1> @test_xtbool_olt_s(float %a, float %b)  {
; CHECK-LABEL: test_xtbool_olt_s:
; CHECK:       # %bb.0:
; CHECK-NEXT:    entry a1, 32
; CHECK-NEXT:    wfr f8, a3
; CHECK-NEXT:    wfr f9, a2
; CHECK-NEXT:    olt.s b0, f9, f8
; CHECK-NEXT:    retw.n
  %r = call <1 x i1> @llvm.xtensa.xt.olt.s(float %a, float %b)
  ret <1 x i1> %r
}

define <1 x i1> @test_xtbool_ult_s(float %a, float %b)  {
; CHECK-LABEL: test_xtbool_ult_s:
; CHECK:       # %bb.0:
; CHECK-NEXT:    entry a1, 32
; CHECK-NEXT:    wfr f8, a3
; CHECK-NEXT:    wfr f9, a2
; CHECK-NEXT:    ult.s b0, f9, f8
; CHECK-NEXT:    retw.n
  %r = call <1 x i1> @llvm.xtensa.xt.ult.s(float %a, float %b)
  ret <1 x i1> %r
}

define <1 x i1> @test_un_s(float %a, float %b)  {
; CHECK-LABEL: test_un_s:
; CHECK:       # %bb.0:
; CHECK-NEXT:    entry a1, 32
; CHECK-NEXT:    wfr f8, a3
; CHECK-NEXT:    wfr f9, a2
; CHECK-NEXT:    un.s b0, f9, f8
; CHECK-NEXT:    retw.n
  %r = call <1 x i1> @llvm.xtensa.xt.un.s(float %a, float %b)
  ret <1 x i1> %r
}

declare float @llvm.xtensa.xt.movt.s(float, float, <1 x i1>);
declare float @llvm.xtensa.xt.movf.s(float, float, <1 x i1>);
declare <1 x i1> @llvm.xtensa.xt.oeq.s(float, float);
declare <1 x i1> @llvm.xtensa.xt.ueq.s(float, float);
declare <1 x i1> @llvm.xtensa.xt.ole.s(float, float);
declare <1 x i1> @llvm.xtensa.xt.ule.s(float, float);
declare <1 x i1> @llvm.xtensa.xt.olt.s(float, float);
declare <1 x i1> @llvm.xtensa.xt.ult.s(float, float);
declare <1 x i1> @llvm.xtensa.xt.un.s(float %a, float %b);
