; RUN: llc -mtriple=xtensa -mcpu=esp32 %s -o - | FileCheck %s

define <1 x i1> @test_xtbool_trunc(i32 %a)  {
  ; CHECK-LABEL: test_xtbool_trunc
  ; CHECK: rsr [[BREG:a[0-9]+]], br
  ; CHECK: and [[AND:a[0-9]+]], {{a[0-9]+}}, {{a[0-9]+}}
  ; CHECK: or  [[OR:a[0-9]+]], [[AND]], a2
  ; CHECK: wsr [[OR]], br
  %trunc = trunc i32 %a to i1
  %vec = insertelement <1 x i1> poison, i1 %trunc, i64 0
  ret <1 x i1> %vec
}

define i32 @test_xtbool_zext(<1 x i1> %b) {
  ; CHECK-LABEL: test_xtbool_zext
  ; CHECK: rsr [[BREG:a[0-9]+]], br
  ; CHECK: extui {{a[0-9]+}}, [[BREG]], 0, 1
  %bit = extractelement <1 x i1> %b, i64 0
  %int = zext i1 %bit to i32
  ret i32 %int
}


define <2 x i1> @test_xtbool2_build(i32 %a, i32 %b) {
  ; CHECK-LABEL: test_xtbool2_build:
  ; CHECK: slli {{a[0-9]+}}, {{a[0-9]+}}, 1
  ; CHECK: or {{a[0-9]+}}, {{a[0-9]+}}, {{a[0-9]+}}
  ; CHECK: rsr [[BREG:a[0-9]+]], br
  ; CHECK: and [[AND:a[0-9]+]], {{a[0-9]+}}, {{a[0-9]+}}
  ; CHECK: or  [[OR:a[0-9]+]], [[AND]], {{a[0-9]+}}
  ; CHECK: wsr [[OR]], br
  %tobool = icmp ne i32 %a, 0
  %vecinit = insertelement <2 x i1> undef, i1 %tobool, i64 0
  %tobool1 = icmp ne i32 %b, 0
  %vecinit2 = insertelement <2 x i1> %vecinit, i1 %tobool1, i64 1
  ret <2 x i1> %vecinit2
}
