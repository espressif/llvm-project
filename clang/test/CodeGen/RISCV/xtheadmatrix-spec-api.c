// NOTE: XTHeadMatrix spec-API (ManagedRA) Clang CodeGen test.
// Tests that spec-API builtins emit _internal intrinsics with proper SSA values.
//
// RUN: %clang_cc1 -O2 -triple riscv64 -target-feature +experimental-xtheadmatrix \
// RUN:   -emit-llvm -o - %s | FileCheck %s

#include <stdint.h>
#include <stddef.h>

// Declare the spec-API builtins directly (normally via thead_matrix.h).
// Load A-tile: (base, stride, m, k) -> matrix
__rvm_int32_t __builtin_riscv_th_mld_spec_i32(void *, size_t, size_t, size_t);
__rvm_int8_t __builtin_riscv_th_mld_spec_i8(void *, size_t, size_t, size_t);

// Load B-tile: (base, stride, k, n) -> matrix
__rvm_int8_t __builtin_riscv_th_mld_b_spec_i8(void *, size_t, size_t, size_t);

// Load accumulator (C-role): (base, stride, m, n) -> matrix
__rvm_int32_t __builtin_riscv_th_mld_acc_spec_i32(void *, size_t, size_t, size_t);

// Store (C-role): (base, stride, val, m, n) -> void
void __builtin_riscv_th_mst_spec_i32(void *, size_t, __rvm_int32_t, size_t, size_t);

// Matmul INT8: (acc, a, b, m, k, n) -> acc
__rvm_int32_t __builtin_riscv_th_mmaqa_spec_ss_w_b(
    __rvm_int32_t, __rvm_int8_t, __rvm_int8_t, size_t, size_t, size_t);

// Zero: (m, n) -> matrix
__rvm_int32_t __builtin_riscv_th_mzero_spec_i32(size_t, size_t);

// CHECK-LABEL: @test_spec_load_store
// CHECK: call void @llvm.riscv.th.msettilem
// CHECK: call void @llvm.riscv.th.msettilek
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mlae.internal32
// CHECK: call void @llvm.riscv.th.msettilem
// CHECK: call void @llvm.riscv.th.msettilen
// CHECK: call void @llvm.riscv.th.msce.internal32
void test_spec_load_store(int32_t *base, long stride) {
    __rvm_int32_t tile = __builtin_riscv_th_mld_spec_i32(base, stride, 4, 4);
    __builtin_riscv_th_mst_spec_i32(base, stride, tile, 4, 4);
}

// CHECK-LABEL: @test_spec_matmul
// CHECK: call void @llvm.riscv.th.msettilem
// CHECK: call void @llvm.riscv.th.msettilek
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mlae.internal8
// CHECK: call void @llvm.riscv.th.msettilek
// CHECK: call void @llvm.riscv.th.msettilen
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mlbe.internal8
// CHECK: call void @llvm.riscv.th.msettilem
// CHECK: call void @llvm.riscv.th.msettilen
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mlce.internal32
// CHECK: call void @llvm.riscv.th.msettilem
// CHECK: call void @llvm.riscv.th.msettilek
// CHECK: call void @llvm.riscv.th.msettilen
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mmacc.w.b.internal
// CHECK: call void @llvm.riscv.th.msettilem
// CHECK: call void @llvm.riscv.th.msettilen
// CHECK: call void @llvm.riscv.th.msce.internal32
void test_spec_matmul(int8_t *a, int8_t *b, int32_t *c, long stride,
                      size_t m, size_t k, size_t n) {
    __rvm_int8_t ta = __builtin_riscv_th_mld_spec_i8(a, stride, m, k);
    __rvm_int8_t tb = __builtin_riscv_th_mld_b_spec_i8(b, stride, k, n);
    __rvm_int32_t tc = __builtin_riscv_th_mld_acc_spec_i32(c, stride, m, n);
    __rvm_int32_t result = __builtin_riscv_th_mmaqa_spec_ss_w_b(tc, ta, tb, m, k, n);
    __builtin_riscv_th_mst_spec_i32(c, stride, result, m, n);
}

// CHECK-LABEL: @test_spec_zero
// CHECK: call void @llvm.riscv.th.msettilem
// CHECK: call void @llvm.riscv.th.msettilen
// CHECK: call target("riscv.matrix") @llvm.riscv.th.mzero.internal
void test_spec_zero(int32_t *base, long stride) {
    __rvm_int32_t z = __builtin_riscv_th_mzero_spec_i32(4, 4);
    __builtin_riscv_th_mst_spec_i32(base, stride, z, 4, 4);
}
