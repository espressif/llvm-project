; RUN: llc -O0 -mtriple=xtensa -mcpu=esp32 %s -o - | FileCheck %s

define <1 x i1> @test_spill(<1 x i1> %b0, <1 x i1> %b1)
{
  ; CHECK-LABEL: test_spill
  ; CHECK: rsr {{a[0-9]+}}, br
  ; CHECK: s8i {{a[0-9]+}}
  ; CHECK: callx8 a8
  %b2 = call <1 x i1> @get_xtbool()
  
  ; CHECK: l8ui {{a[0-9]+}}
  ; CHECK: rsr {{a[0-9]+}}, br
  ; CHECK: wsr {{a[0-9]+}}, br
  ; CHECK: orb {{b[0-9]+}}
  %r0 = or <1 x i1> %b0, %b1
  ret <1 x i1> %r0
}

declare <1 x i1> @get_xtbool()

define <1 x i1> @test_xtbool_load(i32 %addr)  {
  ; CHECK-LABEL: test_xtbool_load
  ; CHECK: l8ui {{a[0-9]+}}
  ; CHECK: rsr [[BREG:a[0-9]+]], br
  ; CHECK: and [[AND:a[0-9]+]], {{a[0-9]+}}, [[BREG]]
  ; CHECK: or  [[OR:a[0-9]+]], [[AND]], {{a[0-9]+}}
  ; CHECK: wsr [[OR]], br
  %ptr = inttoptr i32 %addr to ptr
  %load_bits = load <8 x i1>, ptr %ptr, align 1
  %extractvec = shufflevector <8 x i1> %load_bits, <8 x i1> poison, <1 x i32> zeroinitializer
  ret <1 x i1> %extractvec
}

define void @test_xtbool_store(i32 %addr, <1 x i1> %b) {
  ; CHECK-LABEL: test_xtbool_store:
  ; CHECK: rsr [[BREG:a[0-9]+]], br
  ; CHECK: extui [[DST:a[0-9]+]], [[BREG]], 0, 1
  ; CHECK: s8i [[DST]], {{a[0-9]+}}, {{[0-9]+}}
  %ptr = inttoptr i32 %addr to ptr
  %insertvec = shufflevector <1 x i1> %b, <1 x i1> poison, <8 x i32> <i32 0, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  store <8 x i1> %insertvec, ptr %ptr, align 1
  ret void
}
define <2 x i1> @test_xtbool2_load(i32 %addr)  {
  ; CHECK-LABEL: test_xtbool2_load:
  ; CHECK: l8ui {{a[0-9]+}}
  ; CHECK: rsr [[BREG:a[0-9]+]], br
  ; CHECK: and [[AND:a[0-9]+]], {{a[0-9]+}}, [[BREG]]
  ; CHECK: or  [[OR:a[0-9]+]], [[AND]], {{a[0-9]+}}
  ; CHECK: wsr [[OR]], br
  %ptr = inttoptr i32 %addr to ptr
  %load_bits = load <8 x i1>, ptr %ptr, align 1
  %extractvec = shufflevector <8 x i1> %load_bits, <8 x i1> poison,<2 x i32> <i32 0, i32 1>
  ret <2 x i1> %extractvec
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn writeonly
define void @test_xtbool2_store(i32 %p, <2 x i1> %v) {
  ; CHECK-LABEL: test_xtbool2_store:
  ; CHECK: rsr [[BREG:a[0-9]+]], br
  ; CHECK: extui [[DST:a[0-9]+]], [[BREG]], 0, 2
  ; CHECK: s8i [[DST]], {{a[0-9]+}}, {{[0-9]+}}
  %ptr = inttoptr i32 %p to ptr
  %insertvec = shufflevector <2 x i1> %v, <2 x i1> poison, <8 x i32> <i32 0, i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  store <8 x i1> %insertvec, ptr %ptr, align 1
  ret void
}
