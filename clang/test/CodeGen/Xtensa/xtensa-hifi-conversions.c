// RUN: split-file %s %t
// RUN: %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/correct.c | FileCheck %t/correct.c
// RUN: not %clang -target xtensa -mcpu=cnl -S -emit-llvm -O1 -o - %t/bad_vec.c 2>&1 | FileCheck %t/bad_vec.c

//--- correct.c

typedef int ae_int32 __attribute__(( vector_size(4)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));

ae_int32x2 test_ae_int32x2_from_int(int a) {
// CHECK-LABEL: @test_ae_int32x2_from_int(i
// CHECK:       %[[INS:.*]] = insertelement <2 x i32> poison, i32 %{{.*}}, i64 0
// CHECK:       %[[SHUF:.*]] = shufflevector <2 x i32> %[[INS]], <2 x i32> poison, <2 x i32> zeroinitializer
// CHECK:       ret <2 x i32> %[[SHUF]]
return __builtin_xtensa_ae_int32x2(a);
}

ae_int32x2 test_ae_int32x2_from_ae_int32(ae_int32 a) {
// CHECK-LABEL: @test_ae_int32x2_from_ae_int32(
// CHECK:       %[[SHUF:.*]] = shufflevector <1 x i32> %{{.*}}, <1 x i32> poison, <2 x i32> zeroinitializer
// CHECK:       ret <2 x i32> %[[SHUF]]
return __builtin_xtensa_ae_int32x2(a);
}

ae_int32x2 test_ae_int32x2_from_ae_int32x2(ae_int32x2 a) {
// CHECK: {{.*}}<2 x i32> @test_ae_int32x2_from_ae_int32x2(<2 x i32>{{.*}} %[[A:.*]])
// CHECK:       ret <2 x i32> %[[A]]
return __builtin_xtensa_ae_int32x2(a);
}

ae_int32x2 test_ae_int32x2_from_short(short a) {
// CHECK-LABEL: @test_ae_int32x2_from_short(
// CHECK:       %[[SEXT:.*]] = sext i16 %{{.*}} to i32
// CHECK:       %[[INS:.*]] = insertelement <2 x i32> poison, i32 %[[SEXT]], i64 0
// CHECK:       %[[SHUF:.*]] = shufflevector <2 x i32> %[[INS]], <2 x i32> poison, <2 x i32> zeroinitializer
// CHECK:       ret <2 x i32> %[[SHUF]]
return __builtin_xtensa_ae_int32x2(a);
}

//--- bad_vec.c

typedef short ae_int16x4 __attribute__(( vector_size(8)));
typedef int ae_int32x2 __attribute__(( vector_size(8)));

ae_int32x2 test_ae_int32x2_from_bad_vec(ae_int16x4 a) {
// CHECK: error: passing 'ae_int16x4' {{.*}} to parameter of incompatible type
return __builtin_xtensa_ae_int32x2(a);
}
