; RUN: llc -mtriple=xtensa -mattr=+fp -O1 -verify-machineinstrs < %s \
; RUN:   | FileCheck %s

define float @select_fp_consts(float %x) {
; CHECK: .literal_position
; CHECK-NEXT: .literal .LCPI0_0, 0
; CHECK-NEXT: .literal .LCPI0_1, .LCP0_0
; CHECK-LABEL: select_fp_consts:
; CHECK: # %bb.0:                                # %entry
; CHECK-NEXT: l32r	a8, .LCPI0_0
; CHECK-NEXT: wfr	f8, a8
; CHECK-NEXT: wfr	f9, a2
; CHECK-NEXT: ule.s	b0, f9, f8
; CHECK-NEXT: bf	b0, .LBB0_2
; CHECK-NEXT: # %bb.1:                                # %entry
; CHECK-NEXT: movi	a8, 0
; CHECK-NEXT: j	.LBB0_3
; CHECK-NEXT: .LBB0_2:
; CHECK-NEXT: movi	a8, 4
; CHECK-NEXT: .LBB0_3:                                # %entry
; CHECK-NEXT: l32r	a9, .LCPI0_1
; CHECK-NEXT: add	a8, a9, a8
; CHECK-NEXT: l32i	a2, a8, 0
; CHECK-NEXT: ret
entry:
  %c = fcmp ogt float %x, 0.000000e+00
  %r = select i1 %c, float -1.000000e+00, float 1.000000e+00
  ret float %r
}
