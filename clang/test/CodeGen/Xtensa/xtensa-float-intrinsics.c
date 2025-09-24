// RUN: %clang_cc1  -triple xtensa -O0 -emit-llvm %s -o - | FileCheck %s

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;

float test_float_s(int a) {
  // CHECK: %{{.*}} = call float @llvm.xtensa.xt.float.s(i32 %{{.*}}, i32 1)
  return __builtin_xtensa_xt_float_s(a, 1);
}

float test_ufloat_s(int a) {
  // CHECK: %{{.*}} = call float @llvm.xtensa.xt.ufloat.s(i32 %{{.*}}, i32 1)
  return __builtin_xtensa_xt_ufloat_s(a, 1);
}

int test_trunc_s(float a) {
  // CHECK: %{{.*}} = call i32 @llvm.xtensa.xt.trunc.s(float %{{.*}}, i32 1)
  return __builtin_xtensa_xt_trunc_s(a, 1);
}

int test_utrunc_s(float a) {
  // CHECK: %{{.*}} = call i32 @llvm.xtensa.xt.utrunc.s(float %{{.*}}, i32 1)
  return __builtin_xtensa_xt_utrunc_s(a, 1);
}

float test_add_s(float a, float b) {
  // CHECK: %{{.*}} = call float @llvm.xtensa.xt.add.s(float %{{.*}}, float %{{.*}})
  return __builtin_xtensa_xt_add_s(a, b);
}

float test_sub_s(float a, float b) {
  // CHECK: %{{.*}} = call float @llvm.xtensa.xt.sub.s(float %{{.*}}, float %{{.*}})
  return __builtin_xtensa_xt_sub_s(a, b);
}

float test_mul_s(float a, float b) {
  // CHECK: %{{.*}} = call float @llvm.xtensa.xt.mul.s(float %{{.*}}, float %{{.*}})
  return __builtin_xtensa_xt_mul_s(a, b);
}

float test_xt_lsip(float * a0) {
  // CHECK: %{{.*}} = call { float, ptr } @llvm.xtensa.xt.lsip(ptr %{{.*}}, i32 0)
  return __builtin_xtensa_xt_lsip(&a0, 0);
}

float test_xt_lsxp(float * a0, int a1) {
  // CHECK: %{{.*}} = call { float, ptr } @llvm.xtensa.xt.lsxp(ptr %{{.*}}, i32 %{{.*}})
  return __builtin_xtensa_xt_lsxp(&a0, a1);
}
