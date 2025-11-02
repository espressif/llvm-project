; RUN: llc -O0 -mtriple=xtensa -mcpu=esp32 %s -o - | FileCheck %s

define <1 x i1> @test_spill(<1 x i1> %b0, <1 x i1> %b1)
; CHECK-LABEL: test_spill:
; CHECK:       # %bb.0:
; CHECK-NEXT:    entry a1, 80
; CHECK-NEXT:    movi.n a9, 34
; CHECK-NEXT:    add.n a9, a1, a9
; CHECK-NEXT:    rsr a8, br
; CHECK-NEXT:    extui a8, a8, 1, 1
; CHECK-NEXT:    s8i a8, a9, 0
; CHECK-NEXT:    movi.n a9, 35
; CHECK-NEXT:    add.n a9, a1, a9
; CHECK-NEXT:    rsr a8, br
; CHECK-NEXT:    extui a8, a8, 0, 1
; CHECK-NEXT:    s8i a8, a9, 0
; CHECK-NEXT:    l32r a8, .LCPI0_0
; CHECK-NEXT:    callx8 a8
; CHECK-NEXT:    movi.n a8, 34
; CHECK-NEXT:    add.n a8, a1, a8
; CHECK-NEXT:    l8ui a8, a8, 0
; CHECK-NEXT:    extui a8, a8, 0, 1
; CHECK-NEXT:    slli a8, a8, 1
; CHECK-NEXT:    rsr a10, br
; CHECK-NEXT:    movi a9, 1021
; CHECK-NEXT:    and a10, a10, a9
; CHECK-NEXT:    or a10, a8, a10
; CHECK-NEXT:    wsr a10, br
; CHECK-NEXT:    # kill: def $b2 killed $b0
; CHECK-NEXT:    movi.n a8, 35
; CHECK-NEXT:    add.n a8, a1, a8
; CHECK-NEXT:    l8ui a8, a8, 0
; CHECK-NEXT:    extui a8, a8, 0, 1
; CHECK-NEXT:    rsr a10, br
; CHECK-NEXT:    movi a9, 1022
; CHECK-NEXT:    and a10, a10, a9
; CHECK-NEXT:    or a10, a8, a10
; CHECK-NEXT:    wsr a10, br
; CHECK-NEXT:    orb b0, b0, b1
; CHECK-NEXT:    retw.n
{
  %b2 = call <1 x i1> @get_xtbool()
  
  %r0 = or <1 x i1> %b0, %b1
  ret <1 x i1> %r0
}

declare <1 x i1> @get_xtbool()

define <1 x i1> @test_xtbool_load(i32 %addr)  {
; CHECK-LABEL: test_xtbool_load:
; CHECK:       # %bb.0:
; CHECK-NEXT:    entry a1, 32
; CHECK-NEXT:    l8ui a10, a2, 0
; CHECK-NEXT:    rsr a8, br
; CHECK-NEXT:    l32r a9, .LCPI1_0
; CHECK-NEXT:    and a9, a9, a8
; CHECK-NEXT:    or a9, a9, a10
; CHECK-NEXT:    wsr a9, br
; CHECK-NEXT:    retw.n
  %ptr = inttoptr i32 %addr to ptr
  %load_bits = load <8 x i1>, ptr %ptr, align 1
  %extractvec = shufflevector <8 x i1> %load_bits, <8 x i1> poison, <1 x i32> zeroinitializer
  ret <1 x i1> %extractvec
}

define void @test_xtbool_store(i32 %addr, <1 x i1> %b) {
entry:
; CHECK-LABEL: test_xtbool_store:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    entry a1, 32
; CHECK-NEXT:    rsr a8, br
; CHECK-NEXT:    extui a8, a8, 0, 1
; CHECK-NEXT:    s8i a8, a2, 0
; CHECK-NEXT:    retw.n
  %ptr = inttoptr i32 %addr to ptr
  %insertvec = shufflevector <1 x i1> %b, <1 x i1> poison, <8 x i32> <i32 0, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  store <8 x i1> %insertvec, ptr %ptr, align 1
  ret void
}
