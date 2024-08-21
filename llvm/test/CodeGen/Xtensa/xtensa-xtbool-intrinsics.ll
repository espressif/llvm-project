; RUN: llc -mtriple=xtensa -mcpu=esp32 %s -o - | FileCheck %s

define float @test_xtbool_movt(float %a, float %b, <1 x i1> %c)  {
  ; CHECK-LABEL: test_xtbool_movt
  ; CHECK: movt.s {{f[0-9]+}}, {{f[0-9]+}}, b0
  %f = call float @llvm.xtensa.xt.movt.s(float %a, float %b, <1 x i1> %c)
  ret float %f
}

define float @test_xtbool_movf(float %a, float %b, <1 x i1> %c)  {
  ; CHECK-LABEL: test_xtbool_movf
  ; CHECK: movf.s {{f[0-9]+}}, {{f[0-9]+}}, b0
  %f = call float @llvm.xtensa.xt.movf.s(float %a, float %b, <1 x i1> %c)
  ret float %f
}


declare float @llvm.xtensa.xt.movt.s(float, float, <1 x i1>);
declare float @llvm.xtensa.xt.movf.s(float, float, <1 x i1>);
