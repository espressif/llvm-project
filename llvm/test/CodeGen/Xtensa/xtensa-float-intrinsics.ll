; RUN: llc -mtriple=xtensa -mcpu=esp32 %s -o - | FileCheck %s



define float @test_float_s(i32 %a)  {
  ; CHECK-LABEL: test_float_s
  ; CHECK: float.s {{f[0-9]+}}, a2, 1
  %r = call float @llvm.xtensa.xt.float.s(i32 %a, i32 1)
  ret float %r
}

define float @test_ufloat_s(i32 %a)  {
  ; CHECK-LABEL: test_ufloat_s
  ; CHECK: ufloat.s {{f[0-9]+}}, a2, 1
  %r = call float @llvm.xtensa.xt.ufloat.s(i32 %a, i32 1)
  ret float %r
}

define i32 @test_trunc_s(float %a)  {
  ; CHECK-LABEL: test_trunc_s
  ; CHECK: trunc.s a2, {{f[0-9]+}}, 1
  %r = call i32 @llvm.xtensa.xt.trunc.s(float %a, i32 1)
  ret i32 %r
}

define i32 @test_utrunc_s(float %a)  {
  ; CHECK-LABEL: test_utrunc_s
  ; CHECK: trunc.s a2, {{f[0-9]+}}, 1
  %r = call i32 @llvm.xtensa.xt.utrunc.s(float %a, i32 1)
  ret i32 %r
}

define float @test_add_s(float %a, float %b)  {
  ; CHECK-LABEL: test_add_s
  ; CHECK: add.s {{f[0-9]+}}, {{f[0-9]+}}, {{f[0-9]+}}
  %r = call float @llvm.xtensa.xt.add.s(float %a, float %b)
  ret float %r
}

define float @test_sub_s(float %a, float %b)  {
  ; CHECK-LABEL: test_sub_s
  ; CHECK: sub.s {{f[0-9]+}}, {{f[0-9]+}}, {{f[0-9]+}}
  %r = call float @llvm.xtensa.xt.sub.s(float %a, float %b)
  ret float %r
}

define float @test_mul_s(float %a, float %b)  {
  ; CHECK-LABEL: test_mul_s
  ; CHECK: mul.s {{f[0-9]+}}, {{f[0-9]+}}, {{f[0-9]+}}
  %r = call float @llvm.xtensa.xt.mul.s(float %a, float %b)
  ret float %r
}

define float @test_lsxp(ptr %a, i32 %b)  {
  ; CHECK-LABEL: test_lsxp
  ; CHECK: lsxp {{f[0-9]+}}, {{a[0-9]+}}, {{a[0-9]+}}
  %s = call {float,ptr} @llvm.xtensa.xt.lsxp(ptr %a, i32 %b)
  %r = extractvalue {float,ptr} %s , 0
  ret float %r
}

define float @test_lsip(ptr %a)  {
  ; CHECK-LABEL: test_lsip
  ; CHECK: lsip {{f[0-9]+}}, {{a[0-9]+}}, 4
  %s = call {float,ptr} @llvm.xtensa.xt.lsip(ptr %a, i32 4)
  %r = extractvalue {float,ptr} %s , 0
  ret float %r
}

declare {float,ptr} @llvm.xtensa.xt.lsip(ptr, i32);
declare {float,ptr} @llvm.xtensa.xt.lsxp(ptr, i32);
declare float @llvm.xtensa.xt.float.s(i32, i32);
declare float @llvm.xtensa.xt.ufloat.s(i32, i32);
declare i32 @llvm.xtensa.xt.trunc.s(float , i32);
declare i32 @llvm.xtensa.xt.utrunc.s(float, i32);
declare float @llvm.xtensa.xt.add.s(float %a, float %b);
declare float @llvm.xtensa.xt.sub.s(float %a, float %b);
declare float @llvm.xtensa.xt.mul.s(float %a, float %b);
