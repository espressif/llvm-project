// RUN: %clang_cc1 -triple xtensa -O0 -emit-llvm %s -o - | FileCheck %s

typedef __attribute__((ext_vector_type(1))) _Bool xtbool;

xtbool test_xtbool_movt(float a, float b, xtbool c) {
  // CHECK: %{{.*}} = call float @llvm.xtensa.xt.movt.s(float %{{.*}}, float {{.*}}, <1 x i1> {{.*}})
  return __builtin_xtensa_xt_movt_s(a, b, c);
}

xtbool test_xtbool_movf(float a, float b, xtbool c) {
  // CHECK: %{{.*}} = call float @llvm.xtensa.xt.movf.s(float %{{.*}}, float {{.*}}, <1 x i1> {{.*}})
  return __builtin_xtensa_xt_movf_s(a, b, c);
}

xtbool test_xtbool_oeq_s(float a, float b) {
  // CHECK: %{{.*}} = call <1 x i1> @llvm.xtensa.xt.oeq.s(float %{{.*}}, float {{.*}})
  return __builtin_xtensa_xt_oeq_s(a, b);
}

xtbool test_xtbool_ueq_s(float a, float b) {
  // CHECK: %{{.*}} = call <1 x i1> @llvm.xtensa.xt.ueq.s(float %{{.*}}, float {{.*}})
  return __builtin_xtensa_xt_ueq_s(a, b);
}

xtbool test_xtbool_olt_s(float a, float b) {
  // CHECK: %{{.*}} = call <1 x i1> @llvm.xtensa.xt.olt.s(float %{{.*}}, float {{.*}})
  return __builtin_xtensa_xt_olt_s(a, b);
}

xtbool test_xtbool_ult_s(float a, float b) {
  // CHECK: %{{.*}} = call <1 x i1> @llvm.xtensa.xt.ult.s(float %{{.*}}, float {{.*}})
  return __builtin_xtensa_xt_ult_s(a, b);
}

xtbool test_xtbool_ole_s(float a, float b) {
  // CHECK: %{{.*}} = call <1 x i1> @llvm.xtensa.xt.ole.s(float %{{.*}}, float {{.*}})
  return __builtin_xtensa_xt_ole_s(a, b);
}

xtbool test_xtbool_ule_s(float a, float b) {
  // CHECK: %{{.*}} = call <1 x i1> @llvm.xtensa.xt.ule.s(float %{{.*}}, float {{.*}})
  return __builtin_xtensa_xt_ule_s(a, b);
}

xtbool test_xtbool_un_s(float a, float b) {
  // CHECK: %{{.*}} = call <1 x i1> @llvm.xtensa.xt.un.s(float %{{.*}}, float {{.*}})
  return __builtin_xtensa_xt_un_s(a, b);
}