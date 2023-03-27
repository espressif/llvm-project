// RUN: %clang_cc1 -no-opaque-pointers -triple xtensa -O0 -emit-llvm %s -o - | FileCheck %s

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;

xtbool test_xtbool_movt(float a, float b, xtbool c) {
  // CHECK: %{{.*}} = call float @llvm.xtensa.xt.movt.s(float %{{.*}}, float {{.*}}, <1 x i1> {{.*}})
  return __builtin_xtensa_xt_movt_s(a, b, c);
}

xtbool test_xtbool_movf(float a, float b, xtbool c) {
  // CHECK: %{{.*}} = call float @llvm.xtensa.xt.movf.s(float %{{.*}}, float {{.*}}, <1 x i1> {{.*}})
  return __builtin_xtensa_xt_movf_s(a, b, c);
}