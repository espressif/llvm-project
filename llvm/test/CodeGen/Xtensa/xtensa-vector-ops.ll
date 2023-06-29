; RUN: llc -mtriple=xtensa -mcpu=cnl %s -o - | FileCheck %s

define i32 @test_2xi32toi32(<2 x i32> %a)  {
  ; CHECK-LABEL: test_2xi32toi32
  ; CHECK: ae_movad32.l a2, aed0
  %r = extractelement <2 x i32> %a, i32 0
  ret i32 %r
}

define <2 x i32> @test_i32to2xi32(i32 %a)  {
  ; CHECK-LABEL: test_i32to2xi32
  ; CHECK: ae_movda32x2 aed0, a2, a2
  %vecinit = insertelement <2 x i32> undef, i32 %a, i64 0
  %vecinit1 = shufflevector <2 x i32> %vecinit, <2 x i32> poison, <2 x i32> zeroinitializer
  ret <2 x i32> %vecinit1
}

define void @test_store_2xi32(i32 %a, <2 x i32> %v) {
  ; CHECK-LABEL: test_store_2xi32
  ; CHECK: ae_s32x2.i aed0, a2, 0
  %p = inttoptr i32 %a to ptr
  store <2 x i32> %v, ptr %p, align 8
  ret void
}

define void @test_store_1xi64(i32 %a, <1 x i64> %v) {
  ; CHECK-LABEL: test_store_1xi64
  %p = inttoptr i32 %a to ptr
  ; CHECK: ae_s64.i aed0, a2, 0
  store <1 x i64> %v, ptr %p, align 8
  ret void
}
  
define <1 x i64> @test_build_1xi64(i64 %v) {
  ; CHECK-LABEL: test_build_1xi64
  ; CHECK: ae_movda32x2 aed0, a3, a2
  %vec = insertelement <1 x i64> undef, i64 %v, i64 0
  ret <1 x i64> %vec
}

define void @test_store_4xi16(i32 %a, <4 x i16> %v) {
  ; CHECK-LABEL: test_store_4xi16
  ; CHECK: ae_s16x4.i aed0, a2, 0
  %p = inttoptr i32 %a to ptr
  store <4 x i16> %v, ptr %p, align 8
  ret void
}

define <2 x i32> @test_load_2xi32(i32 %a) {
  ; CHECK-LABEL: test_load_2xi32
  ; CHECK: ae_l32x2.i aed0, a2, 0
  %p = inttoptr i32 %a to ptr
  %v = load <2 x i32>, ptr %p, align 8
  ret <2 x i32> %v
}

define <1 x i64> @test_load_1xi64(i32 %a) {
  ; CHECK-LABEL: test_load_1xi64
  ; CHECK: ae_l64.i aed0, a2, 0
  %p = inttoptr i32 %a to ptr
  %v = load <1 x i64>, ptr %p, align 8
  ret <1 x i64> %v
}

define <4 x i16> @test_load_4xi16(i32 %a) {
  ; CHECK-LABEL: test_load_4xi16
  ; CHECK: ae_l16x4.i aed0, a2, 0
  %p = inttoptr i32 %a to ptr
  %v = load <4 x i16>, ptr %p, align 8
  ret <4 x i16> %v
}

define void @test_build_store_1xi32(i32 %a, i32 %v) {
  ; CHECK-LABEL: test_build_store_1xi32
  ; CHECK: ae_movda32 [[A:aed[0-9]+]], a3
  %vec = insertelement <1 x i32> undef, i32 %v, i64 0
  %p = inttoptr i32 %a to ptr
  ; CHECK: ae_s32.l.i [[A]], a2, 0
  store <1 x i32> %vec, ptr %p, align 8
  ret void
}

define i32 @test_load_extract_1xi32(i32 %a) {
  ; CHECK-LABEL: test_load_extract_1xi32
  %p = inttoptr i32 %a to ptr
  ; CHECK: ae_l32.i [[A:aed[0-9]+]], a2, 0
  %vec = load <1 x i32>, ptr %p, align 8
  ; CHECK: ae_movad32.l a2, [[A]]
  %r = extractelement <1 x i32> %vec, i32 0
  ret i32 %r
}

define <4 x i16> @test_build_4xi16_2(i16 %a, i16 %b) {
  ; CHECK-LABEL: test_build_4xi16_2
  ; CHECK: ae_movda16x2 aed0, a2, a3
  %vecinit = insertelement <4 x i16> undef, i16 %a, i64 0
  %vecinit1 = insertelement <4 x i16> %vecinit, i16 %b, i64 1
  %vecinit2 = insertelement <4 x i16> %vecinit1, i16 %a, i64 2
  %vecinit3 = insertelement <4 x i16> %vecinit2, i16 %b, i64 3
  ret <4 x i16> %vecinit3
}

define <4 x i16> @test_build_4xi16_1(i16 %a) {
  ; CHECK-LABEL: test_build_4xi16_1
  ; CHECK: ae_movda16 aed0, a2
  %vecinit = insertelement <4 x i16> undef, i16 %a, i64 0
  %vecinit1 = shufflevector <4 x i16> %vecinit, <4 x i16> poison, <4 x i32> zeroinitializer
  ret <4 x i16> %vecinit1
}

define i32 @test_extract(<2 x i32> %v2i, <1 x i32> %v1i, <4 x i16> %v4s, <1 x i64> %v1l) {
  ; CHECK-LABEL: test_extract
  ; CHECK-DAG: ae_movad32.h {{a[0-9]+}}, aed0
  %v2i0 = extractelement <2 x i32> %v2i, i64 0
  ; CHECK-DAG: ae_movad32.l {{a[0-9]+}}, aed0
  %v2i1 = extractelement <2 x i32> %v2i, i64 1
  %sum1 = add i32 %v2i0, %v2i1
  ; CHECK-DAG: ae_movad32.l {{a[0-9]+}}, aed1
  %v1i0 = extractelement <1 x i32> %v1i, i64 0
  %sum2 = add i32 %sum1, %v1i0
  ; CHECK-DAG: ae_movad16.0 {{a[0-9]+}}, aed2
  %v4s0 = extractelement <4 x i16> %v4s, i64 0
  %v4s0i = zext i16 %v4s0 to i32
  %sum3 = add i32 %v4s0i, %sum2
  ; CHECK-DAG: ae_movad16.1 {{a[0-9]+}}, aed2
  %v4s1 = extractelement <4 x i16> %v4s, i64 1
  %v4s1i = zext i16 %v4s1 to i32
  %sum4 = add i32 %v4s1i, %sum3
  ; CHECK-DAG: ae_movad16.2 {{a[0-9]+}}, aed2
  %v4s2 = extractelement <4 x i16> %v4s, i64 2
  %v4s2i = zext i16 %v4s2 to i32
  %sum5 = add i32 %v4s2i, %sum4
  ; CHECK-DAG: ae_movad16.3 {{a[0-9]+}}, aed2
  %v4s3 = extractelement <4 x i16> %v4s, i64 3
  %v4s3i = zext i16 %v4s3 to i32
  %sum6 = add i32 %v4s3i, %sum5
  ; CHECK-DAG: ae_movad32.l {{a[0-9]+}}, aed3
  %v1l0 = extractelement <1 x i64> %v1l, i64 0
  %v1l0l = trunc i64 %v1l0 to i32
  %sum7 = add i32 %v1l0l, %sum6

  ret i32 %sum7
}

define <1 x i32> @test_extract_subvec_1x32(<2 x i32> %v) {
  ; CHECK-LABEL: test_extract_subvec_1x32
  ; CHECK: ae_movad32.l {{a[0-9]+}}, aed0
  ; CHECK: ae_movda32 aed0, {{a[0-9]+}}
  %shuffle = shufflevector <2 x i32> %v, <2 x i32> poison, <1 x i32> zeroinitializer
  ret <1 x i32> %shuffle
}


define <4 x i16> @rlshift4(<4 x i16> %a, i16 signext %b) {
  ; CHECK-LABEL: rlshift4:
  ; CHECK: ssr {{a[0-9]+}}
  %v = insertelement <4 x i16> undef, i16 %b, i64 0
  %sh_prom = shufflevector <4 x i16> %v, <4 x i16> poison, <4 x i32> zeroinitializer
  %shr = lshr <4 x i16> %a, %sh_prom
  ret <4 x i16> %shr
}


define <4 x i16> @rlshift4_imm(<4 x i16> %a) {
  ; CHECK-LABEL: rlshift4_imm:
  ; CHECK: srli {{a[0-9]+}}, {{a[0-9]+}}, 1
  %shr = lshr <4 x i16> %a, <i16 1, i16 1, i16 1, i16 1>
  ret <4 x i16> %shr
}


define <2 x i32> @rlshift2(<2 x i32> %a, i32 %b) {
  ; CHECK-LABEL: rlshift2:
  ; CHECK: ssr {{a[0-9]+}}
  %splat.splatinsert = insertelement <2 x i32> poison, i32 %b, i64 0
  %splat.splat = shufflevector <2 x i32> %splat.splatinsert, <2 x i32> poison, <2 x i32> zeroinitializer
  %shr = lshr <2 x i32> %a, %splat.splat
  ret <2 x i32> %shr
}


define <2 x i32> @rlshift2_imm(<2 x i32> %a) {
  ; CHECK-LABEL: rlshift2_imm:
  ; CHECK: srli {{a[0-9]+}}, {{a[0-9]+}}, 1
  %shr = lshr <2 x i32> %a, <i32 1, i32 1>
  ret <2 x i32> %shr
}

define <1 x i64> @rlshift1(<1 x i64> %a, i32 %b) {
  ; CHECK-LABEL: rlshift1:
  ; CHECK: ssr {{a[0-9]+}}
  ; CHECK: src {{a[0-9]+}}
  ; CHECK: srl {{a[0-9]+}}
   %splat.splatinsert = insertelement <1 x i32> poison, i32 %b, i64 0
  %sh_prom = zext <1 x i32> %splat.splatinsert to <1 x i64>
  %shr = lshr <1 x i64> %a, %sh_prom
  ret <1 x i64> %shr
}

define <1 x i64> @rlshift1_imm(<1 x i64> %a) {
  ; CHECK-LABEL: rlshift1_imm:
  ; CHECK: ssr {{a[0-9]+}}
  ; CHECK: src {{a[0-9]+}}
  ; CHECK: srl {{a[0-9]+}}
  %shr = lshr <1 x i64> %a, <i64 1>
  ret <1 x i64> %shr
}

define <4 x i16> @rashift4(<4 x i16> %a, i16 signext %b) {
  ; CHECK-LABEL: rashift4:
  ; CHECK: ssr {{a[0-9]+}}
  ; CHECK: sra {{a[0-9]+}}
  %v = insertelement <4 x i16> undef, i16 %b, i64 0
  %sh_prom = shufflevector <4 x i16> %v, <4 x i16> poison, <4 x i32> zeroinitializer
  %shr = ashr <4 x i16> %a, %sh_prom
  ret <4 x i16> %shr
}

define <4 x i16> @rashift4_imm(<4 x i16> %a) {
  ; CHECK-LABEL: rashift4_imm:
  ; CHECK: srai {{a[0-9]+}}, {{a[0-9]+}}, 1
  %shr = ashr <4 x i16> %a, <i16 1, i16 1, i16 1, i16 1>
  ret <4 x i16> %shr
}

define <2 x i32> @rashift2(<2 x i32> %a, i32 %b) {
  ; CHECK-LABEL: rashift2:
  ; CHECK: ssr {{a[0-9]+}}
  ; CHECK: sra {{a[0-9]+}}
  %splat.splatinsert = insertelement <2 x i32> poison, i32 %b, i64 0
  %splat.splat = shufflevector <2 x i32> %splat.splatinsert, <2 x i32> poison, <2 x i32> zeroinitializer
  %shr = ashr <2 x i32> %a, %splat.splat
  ret <2 x i32> %shr
}


define <2 x i32> @rashift2_imm(<2 x i32> %a) {
  ; CHECK-LABEL: rashift2_imm:
  ; CHECK: srai {{a[0-9]+}}, {{a[0-9]+}}, 1
  %shr = ashr <2 x i32> %a, <i32 1, i32 1>
  ret <2 x i32> %shr
}


define <1 x i64> @rashift1(<1 x i64> %a, i64 %b) {
  ; CHECK-LABEL: rashift1:
  ; CHECK: ssr {{a[0-9]+}}
  ; CHECK: sra {{a[0-9]+}}
  %splat.splatinsert = insertelement <1 x i64> poison, i64 %b, i64 0
  %shr = ashr <1 x i64> %a, %splat.splatinsert
  ret <1 x i64> %shr
}


define <1 x i64> @rashift1_imm(<1 x i64> %a) {
  ; CHECK-LABEL: rashift1_imm:
  ; CHECK: ssr {{a[0-9]+}}
  ; CHECK: sra {{a[0-9]+}}
  %shr = ashr <1 x i64> %a, <i64 1>
  ret <1 x i64> %shr
}


define <4 x i16> @lshift4(<4 x i16> %a, i16 signext %b) {
  ; CHECK-LABEL: lshift4:
  ; CHECK: ssl {{a[0-9]+}}
  ; CHECK: sll {{a[0-9]+}}
  %v = insertelement <4 x i16> undef, i16 %b, i64 0
  %sh_prom = shufflevector <4 x i16> %v, <4 x i16> poison, <4 x i32> zeroinitializer
  %shl = shl <4 x i16> %a, %sh_prom
  ret <4 x i16> %shl
}


define <4 x i16> @lshift4_imm(<4 x i16> %a) {
  ; CHECK-LABEL: lshift4_imm:
  ; CHECK: slli {{a[0-9]+}}, {{a[0-9]+}}, 1
  %shl = shl <4 x i16> %a, <i16 1, i16 1, i16 1, i16 1>
  ret <4 x i16> %shl
}


define <2 x i32> @lshift2(<2 x i32> %a, i32 %b) {
  ; CHECK-LABEL: lshift2:
  ; CHECK: ssl {{a[0-9]+}}
  ; CHECK: sll {{a[0-9]+}}
  %splat.splatinsert = insertelement <2 x i32> poison, i32 %b, i64 0
  %splat.splat = shufflevector <2 x i32> %splat.splatinsert, <2 x i32> poison, <2 x i32> zeroinitializer
  %shl = shl <2 x i32> %a, %splat.splat
  ret <2 x i32> %shl
}


define <2 x i32> @lshift2_imm(<2 x i32> %a) {
  ; CHECK-LABEL: lshift2_imm:
  ; CHECK: slli {{a[0-9]+}}, {{a[0-9]+}}, 1
  %shl = shl <2 x i32> %a, <i32 1, i32 1>
  ret <2 x i32> %shl
}


define <1 x i64> @lshift1(<1 x i64> %a, i64 %b) {
  ; CHECK-LABEL: lshift1:
  ; CHECK: ssl {{a[0-9]+}}
  ; CHECK: sll {{a[0-9]+}}
  %splat.splatinsert = insertelement <1 x i64> poison, i64 %b, i64 0
  %shl = shl <1 x i64> %a, %splat.splatinsert
  ret <1 x i64> %shl
}


define <1 x i64> @lshift1_imm(<1 x i64> %a) {
  ; CHECK-LABEL: lshift1_imm:
  ; CHECK: ssl {{a[0-9]+}}
  ; CHECK: sll {{a[0-9]+}}
  %shl = shl <1 x i64> %a, <i64 1>
  ret <1 x i64> %shl
}

define void @test_valign_load_store(i32 %p1, i32 %p2) {
  ; CHECK-LABEL: test_valign_load_store:
  %ptr1 = inttoptr i32 %p1 to ptr
  %ptr2 = inttoptr i32 %p2 to ptr
  ; CHECK: ae_lalign64.i [[V:u[0-3]]], {{a[0-9]+}}, 0
  %v = load <8 x i8>, ptr %ptr1, align 8
  ; CHECK: ae_salign64.i [[V]], {{a[0-9]+}}, 0
  store <8 x i8> %v, ptr %ptr2, align 8
  ret void
}
