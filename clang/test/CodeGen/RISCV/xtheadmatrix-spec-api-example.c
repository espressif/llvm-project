// End-to-end example: FP8->FP16 widening matmul using the Spec-API.
//
// This test verifies the full pipeline from C source through Clang codegen:
//   config -> load A/B tiles -> zero acc -> widening matmul -> store
//
// The key invariant is that the matmul intrinsic receives 3 *distinct* SSA
// values (acc, b, a) rather than repeating the accumulator for all three.
//
// RUN: %clang_cc1 -O2 -triple riscv64 -target-feature +experimental-xtheadmatrix \
// RUN:   -emit-llvm -o - %s | FileCheck %s

#include <thead_matrix.h>

// CHECK-LABEL: @fp8_to_fp16_matmul
//
// Step 1: load A tile (opaque FP8 data loaded as i32)
// CHECK: call void @llvm.riscv.th.msettilem
// CHECK: call void @llvm.riscv.th.msettilek
// CHECK: %[[A:.+]] = {{.*}}call target("riscv.matrix") @llvm.riscv.th.mlae.internal32
//
// Step 2: load B tile
// CHECK: call void @llvm.riscv.th.msettilek
// CHECK: call void @llvm.riscv.th.msettilen
// CHECK: %[[B:.+]] = {{.*}}call target("riscv.matrix") @llvm.riscv.th.mlbe.internal32
//
// Step 3: zero the accumulator
// CHECK: call void @llvm.riscv.th.msettilem
// CHECK: call void @llvm.riscv.th.msettilen
// CHECK: %[[Z:.+]] = {{.*}}call target("riscv.matrix") @llvm.riscv.th.mzero.internal
//
// Step 4: widening matmul — acc, b, a are distinct values (operand swap: b->ms2, a->ms1)
// CHECK: call void @llvm.riscv.th.msettilem
// CHECK: call void @llvm.riscv.th.msettilek
// CHECK: call void @llvm.riscv.th.msettilen
// CHECK: %[[R:.+]] = {{.*}}call target("riscv.matrix") @llvm.riscv.th.mfmacc.h.e4.internal{{.*}}(target("riscv.matrix") %[[Z]], target("riscv.matrix") %[[B]], target("riscv.matrix") %[[A]])
//
// Step 5: store result
// CHECK: call void @llvm.riscv.th.msettilem
// CHECK: call void @llvm.riscv.th.msettilen
// CHECK: call void @llvm.riscv.th.msce.internal16
void fp8_to_fp16_matmul(void *a_ptr, long a_stride,
                         void *b_ptr, long b_stride,
                         uint16_t *c_ptr, long c_stride,
                         mrow_t m, mcol_t k, mcol_t n) {
    // Load FP8 A-tile (opaque, use mint32_t)
    mint32_t a = __riscv_th_mld_a_i32(a_ptr, a_stride, m, k);

    // Load FP8 B-tile (opaque, use mint32_t)
    mint32_t b = __riscv_th_mld_b_i32(b_ptr, b_stride, k, n);

    // Zero the FP16 accumulator
    mfloat16_t acc = __riscv_th_mzeros_f16(m, n);

    // Widening matmul: FP8 * FP8 -> FP16 accumulate
    mfloat16_t result = __riscv_th_mfmacc_h_e4(acc, a, b, m, k, n);

    // Store the FP16 result
    __riscv_th_mst_f16(c_ptr, c_stride, result, m, n);
}
