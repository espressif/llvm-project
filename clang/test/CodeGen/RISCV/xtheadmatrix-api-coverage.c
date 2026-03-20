// XTHeadMatrix C API comprehensive coverage test.
//
// Tests public C API functions from <thead_matrix.h> that have insufficient
// test coverage in other test files. Covers:
//   - CSR read/write macros
//   - All mundefined constructors (single + x2)
//   - All mreinterpret casts (x2 variants)
//   - All mzero constructors for every type
//   - All tuple mget/mset for every type
//   - Load/store type variants not covered elsewhere
//   - Store A/B for all element types
//   - Transposed load/store for all element types
//   - Whole-register load/store for all element types
//   - Immediate config macros
//
// RUN: %clang_cc1 -O0 -triple riscv64 -target-feature +experimental-xtheadmatrix \
// RUN:   -emit-llvm -o - %s | FileCheck --check-prefix=O0 %s
// RUN: %clang_cc1 -O2 -triple riscv64 -target-feature +experimental-xtheadmatrix \
// RUN:   -emit-llvm -o - %s | FileCheck --check-prefix=O2 %s

#include <thead_matrix.h>

// ========================================================================
// 1. CSR Read/Write
// ========================================================================

// O0-LABEL: @test_csr_read_xmcsr
// O0: call i64 asm sideeffect "csrr $0, th.xmcsr"
unsigned long test_csr_read_xmcsr(void) {
    return __riscv_th_mread_csr(RVM_CSR_XMCSR);
}

// O0-LABEL: @test_csr_read_mtilem
// O0: call i64 asm sideeffect "csrr $0, th.mtilem"
unsigned long test_csr_read_mtilem(void) {
    return __riscv_th_mread_csr(RVM_CSR_MTILEM);
}

// O0-LABEL: @test_csr_read_mtilen
// O0: call i64 asm sideeffect "csrr $0, th.mtilen"
unsigned long test_csr_read_mtilen(void) {
    return __riscv_th_mread_csr(RVM_CSR_MTILEN);
}

// O0-LABEL: @test_csr_read_mtilek
// O0: call i64 asm sideeffect "csrr $0, th.mtilek"
unsigned long test_csr_read_mtilek(void) {
    return __riscv_th_mread_csr(RVM_CSR_MTILEK);
}

// O0-LABEL: @test_csr_read_xmxrm
// O0: call i64 asm sideeffect "csrr $0, th.xmxrm"
unsigned long test_csr_read_xmxrm(void) {
    return __riscv_th_mread_csr(RVM_CSR_XMXRM);
}

// O0-LABEL: @test_csr_read_xmsat
// O0: call i64 asm sideeffect "csrr $0, th.xmsat"
unsigned long test_csr_read_xmsat(void) {
    return __riscv_th_mread_csr(RVM_CSR_XMSAT);
}

// O0-LABEL: @test_csr_read_xmfflags
// O0: call i64 asm sideeffect "csrr $0, th.xmfflags"
unsigned long test_csr_read_xmfflags(void) {
    return __riscv_th_mread_csr(RVM_CSR_XMFFLAGS);
}

// O0-LABEL: @test_csr_read_xmfrm
// O0: call i64 asm sideeffect "csrr $0, th.xmfrm"
unsigned long test_csr_read_xmfrm(void) {
    return __riscv_th_mread_csr(RVM_CSR_XMFRM);
}

// O0-LABEL: @test_csr_read_xmsaten
// O0: call i64 asm sideeffect "csrr $0, th.xmsaten"
unsigned long test_csr_read_xmsaten(void) {
    return __riscv_th_mread_csr(RVM_CSR_XMSATEN);
}

// O0-LABEL: @test_csr_read_xalenb
// O0: call i64 asm sideeffect "csrr $0, th.xalenb"
unsigned long test_csr_read_xalenb(void) {
    return __riscv_th_mread_csr(RVM_CSR_XALENB);
}

// O0-LABEL: @test_csr_write_xmcsr
// O0: call void asm sideeffect "csrw th.xmcsr, $0"
void test_csr_write_xmcsr(unsigned long val) {
    __riscv_th_mwrite_csr(RVM_CSR_XMCSR, val);
}

// O0-LABEL: @test_csr_write_xmxrm
// O0: call void asm sideeffect "csrw th.xmxrm, $0"
void test_csr_write_xmxrm(unsigned long val) {
    __riscv_th_mwrite_csr(RVM_CSR_XMXRM, val);
}

// O0-LABEL: @test_csr_write_xmfrm
// O0: call void asm sideeffect "csrw th.xmfrm, $0"
void test_csr_write_xmfrm(unsigned long val) {
    __riscv_th_mwrite_csr(RVM_CSR_XMFRM, val);
}

// ========================================================================
// 2. Undefined Value Constructors - All single types
// ========================================================================

// O0-LABEL: @test_mundefined_i8
// O0: ret target("riscv.matrix") poison
mint8_t test_mundefined_i8(void) { return __riscv_th_mundefined_i8(); }

// O0-LABEL: @test_mundefined_i16
// O0: ret target("riscv.matrix") poison
mint16_t test_mundefined_i16(void) { return __riscv_th_mundefined_i16(); }

// O0-LABEL: @test_mundefined_i64
// O0: ret target("riscv.matrix") poison
mint64_t test_mundefined_i64(void) { return __riscv_th_mundefined_i64(); }

// O0-LABEL: @test_mundefined_u8
// O0: ret target("riscv.matrix") poison
muint8_t test_mundefined_u8(void) { return __riscv_th_mundefined_u8(); }

// O0-LABEL: @test_mundefined_u16
// O0: ret target("riscv.matrix") poison
muint16_t test_mundefined_u16(void) { return __riscv_th_mundefined_u16(); }

// O0-LABEL: @test_mundefined_u32
// O0: ret target("riscv.matrix") poison
muint32_t test_mundefined_u32(void) { return __riscv_th_mundefined_u32(); }

// O0-LABEL: @test_mundefined_u64
// O0: ret target("riscv.matrix") poison
muint64_t test_mundefined_u64(void) { return __riscv_th_mundefined_u64(); }

// O0-LABEL: @test_mundefined_f16
// O0: ret target("riscv.matrix") poison
mfloat16_t test_mundefined_f16(void) { return __riscv_th_mundefined_f16(); }

// O0-LABEL: @test_mundefined_f32
// O0: ret target("riscv.matrix") poison
mfloat32_t test_mundefined_f32(void) { return __riscv_th_mundefined_f32(); }

// O0-LABEL: @test_mundefined_f64
// O0: ret target("riscv.matrix") poison
mfloat64_t test_mundefined_f64(void) { return __riscv_th_mundefined_f64(); }

// ========================================================================
// 3. Undefined Value Constructors - All x2 types
// ========================================================================

// O0-LABEL: @test_mundefined_i8x2
// O0: ret { target("riscv.matrix"), target("riscv.matrix") } poison
mint8x2_t test_mundefined_i8x2(void) { return __riscv_th_mundefined_i8x2(); }

// O0-LABEL: @test_mundefined_i16x2
// O0: ret { target("riscv.matrix"), target("riscv.matrix") } poison
mint16x2_t test_mundefined_i16x2(void) { return __riscv_th_mundefined_i16x2(); }

// O0-LABEL: @test_mundefined_i32x2
// O0: ret { target("riscv.matrix"), target("riscv.matrix") } poison
mint32x2_t test_mundefined_i32x2(void) { return __riscv_th_mundefined_i32x2(); }

// O0-LABEL: @test_mundefined_u8x2
// O0: ret { target("riscv.matrix"), target("riscv.matrix") } poison
muint8x2_t test_mundefined_u8x2(void) { return __riscv_th_mundefined_u8x2(); }

// O0-LABEL: @test_mundefined_u32x2
// O0: ret { target("riscv.matrix"), target("riscv.matrix") } poison
muint32x2_t test_mundefined_u32x2(void) { return __riscv_th_mundefined_u32x2(); }

// O0-LABEL: @test_mundefined_f32x2
// O0: ret { target("riscv.matrix"), target("riscv.matrix") } poison
mfloat32x2_t test_mundefined_f32x2(void) { return __riscv_th_mundefined_f32x2(); }

// ========================================================================
// 4. Reinterpret Casts - x2 variants (single variants covered elsewhere)
// ========================================================================

// O0-LABEL: @test_mreinterpret_x2_i8_to_u16
// O0: call target("riscv.matrix") asm "", "=^tr,0"
muint16x2_t test_mreinterpret_x2_i8_to_u16(mint8x2_t src) {
    return __riscv_th_mreinterpret_u16x2(src);
}

// O0-LABEL: @test_mreinterpret_x2_f32_to_i64
// O0: call target("riscv.matrix") asm "", "=^tr,0"
mint64x2_t test_mreinterpret_x2_f32_to_i64(mfloat32x2_t src) {
    return __riscv_th_mreinterpret_i64x2(src);
}

// O0-LABEL: @test_mreinterpret_x2_u64_to_f16
// O0: call target("riscv.matrix") asm "", "=^tr,0"
mfloat16x2_t test_mreinterpret_x2_u64_to_f16(muint64x2_t src) {
    return __riscv_th_mreinterpret_f16x2(src);
}

// O0-LABEL: @test_mreinterpret_x2_i32_to_f64
// O0: call target("riscv.matrix") asm "", "=^tr,0"
mfloat64x2_t test_mreinterpret_x2_i32_to_f64(mint32x2_t src) {
    return __riscv_th_mreinterpret_f64x2(src);
}

// O0-LABEL: @test_mreinterpret_x2_u8_to_i32
// O0: call target("riscv.matrix") asm "", "=^tr,0"
mint32x2_t test_mreinterpret_x2_u8_to_i32(muint8x2_t src) {
    return __riscv_th_mreinterpret_i32x2(src);
}

// ========================================================================
// 5. Zero Constructors - All types
// ========================================================================

// O2-LABEL: @test_mzero_i8
// O2: call void @llvm.riscv.th.msettilem
// O2: call void @llvm.riscv.th.msettilen
// O2: call target("riscv.matrix") @llvm.riscv.th.mzero.internal
void test_mzero_i8(int8_t *p, long s) {
    mint8_t z = __riscv_th_mzeros_i8(4, 4);
    __riscv_th_mst_i8(p, s, z, 4, 4);
}

// O2-LABEL: @test_mzero_i16
// O2: call target("riscv.matrix") @llvm.riscv.th.mzero.internal
void test_mzero_i16(int16_t *p, long s) {
    mint16_t z = __riscv_th_mzeros_i16(4, 4);
    __riscv_th_mst_i16(p, s, z, 4, 4);
}

// O2-LABEL: @test_mzero_i64
// O2: call target("riscv.matrix") @llvm.riscv.th.mzero.internal
void test_mzero_i64(int64_t *p, long s) {
    mint64_t z = __riscv_th_mzeros_i64(4, 4);
    __riscv_th_mst_i64(p, s, z, 4, 4);
}

// O2-LABEL: @test_mzero_u8
// O2: call target("riscv.matrix") @llvm.riscv.th.mzero.internal
void test_mzero_u8(uint8_t *p, long s) {
    muint8_t z = __riscv_th_mzeros_u8(4, 4);
    __riscv_th_mst_u8(p, s, z, 4, 4);
}

// O2-LABEL: @test_mzero_u16
// O2: call target("riscv.matrix") @llvm.riscv.th.mzero.internal
void test_mzero_u16(uint16_t *p, long s) {
    muint16_t z = __riscv_th_mzeros_u16(4, 4);
    __riscv_th_mst_u16(p, s, z, 4, 4);
}

// O2-LABEL: @test_mzero_u32
// O2: call target("riscv.matrix") @llvm.riscv.th.mzero.internal
void test_mzero_u32(uint32_t *p, long s) {
    muint32_t z = __riscv_th_mzeros_u32(4, 4);
    __riscv_th_mst_u32(p, s, z, 4, 4);
}

// O2-LABEL: @test_mzero_u64
// O2: call target("riscv.matrix") @llvm.riscv.th.mzero.internal
void test_mzero_u64(uint64_t *p, long s) {
    muint64_t z = __riscv_th_mzeros_u64(4, 4);
    __riscv_th_mst_u64(p, s, z, 4, 4);
}

// O2-LABEL: @test_mzero_f16
// O2: call target("riscv.matrix") @llvm.riscv.th.mzero.internal
void test_mzero_f16(_Float16 *p, long s) {
    mfloat16_t z = __riscv_th_mzeros_f16(4, 4);
    __riscv_th_mst_f16(p, s, z, 4, 4);
}

// O2-LABEL: @test_mzero_f32
// O2: call target("riscv.matrix") @llvm.riscv.th.mzero.internal
void test_mzero_f32(float *p, long s) {
    mfloat32_t z = __riscv_th_mzeros_f32(4, 4);
    __riscv_th_mst_f32(p, s, z, 4, 4);
}

// O2-LABEL: @test_mzero_f64
// O2: call target("riscv.matrix") @llvm.riscv.th.mzero.internal
void test_mzero_f64(double *p, long s) {
    mfloat64_t z = __riscv_th_mzeros_f64(4, 4);
    __riscv_th_mst_f64(p, s, z, 4, 4);
}

// Test mzero alias (not mzeros)
// O2-LABEL: @test_mzero_alias_i32
// O2: call target("riscv.matrix") @llvm.riscv.th.mzero.internal
void test_mzero_alias_i32(int32_t *p, long s) {
    mint32_t z = __riscv_th_mzero_i32(4, 4);
    __riscv_th_mst_i32(p, s, z, 4, 4);
}

// ========================================================================
// 6. Tuple Get/Set - All types
// ========================================================================

// O0-LABEL: @test_mget_i8
// O0: extractvalue
mint8_t test_mget_i8(mint8x2_t pair) {
    return __riscv_th_mget_i8(pair, 0);
}

// O0-LABEL: @test_mget_i16
// O0: extractvalue
mint16_t test_mget_i16(mint16x2_t pair) {
    return __riscv_th_mget_i16(pair, 1);
}

// O0-LABEL: @test_mget_u8
// O0: extractvalue
muint8_t test_mget_u8(muint8x2_t pair) {
    return __riscv_th_mget_u8(pair, 0);
}

// O0-LABEL: @test_mget_u16
// O0: extractvalue
muint16_t test_mget_u16(muint16x2_t pair) {
    return __riscv_th_mget_u16(pair, 0);
}

// O0-LABEL: @test_mget_u32
// O0: extractvalue
muint32_t test_mget_u32(muint32x2_t pair) {
    return __riscv_th_mget_u32(pair, 1);
}

// O0-LABEL: @test_mget_f32
// O0: extractvalue
mfloat32_t test_mget_f32(mfloat32x2_t pair) {
    return __riscv_th_mget_f32(pair, 0);
}

// O0-LABEL: @test_mset_i8
// O0: insertvalue
mint8x2_t test_mset_i8(mint8x2_t pair, mint8_t val) {
    return __riscv_th_mset_i8(pair, 0, val);
}

// O0-LABEL: @test_mset_i16
// O0: insertvalue
mint16x2_t test_mset_i16(mint16x2_t pair, mint16_t val) {
    return __riscv_th_mset_i16(pair, 1, val);
}

// O0-LABEL: @test_mset_u8
// O0: insertvalue
muint8x2_t test_mset_u8(muint8x2_t pair, muint8_t val) {
    return __riscv_th_mset_u8(pair, 0, val);
}

// O0-LABEL: @test_mset_u16
// O0: insertvalue
muint16x2_t test_mset_u16(muint16x2_t pair, muint16_t val) {
    return __riscv_th_mset_u16(pair, 1, val);
}

// O0-LABEL: @test_mset_u32
// O0: insertvalue
muint32x2_t test_mset_u32(muint32x2_t pair, muint32_t val) {
    return __riscv_th_mset_u32(pair, 0, val);
}

// O0-LABEL: @test_mset_f32
// O0: insertvalue
mfloat32x2_t test_mset_f32(mfloat32x2_t pair, mfloat32_t val) {
    return __riscv_th_mset_f32(pair, 1, val);
}

// ========================================================================
// 7. Load - unsigned and FP types for all element widths
// ========================================================================

// O2-LABEL: @test_load_a_u8
// O2: call target("riscv.matrix") @llvm.riscv.th.mlae.internal8
void test_load_a_u8(uint8_t *base, long stride) {
    muint8_t t = __riscv_th_mld_a_u8(base, stride, 4, 8);
    __riscv_th_mst_u8(base, stride, t, 4, 8);
}

// O2-LABEL: @test_load_a_u16
// O2: call target("riscv.matrix") @llvm.riscv.th.mlae.internal16
void test_load_a_u16(uint16_t *base, long stride) {
    muint16_t t = __riscv_th_mld_a_u16(base, stride, 4, 4);
    __riscv_th_mst_u16(base, stride, t, 4, 4);
}

// O2-LABEL: @test_load_a_u32
// O2: call target("riscv.matrix") @llvm.riscv.th.mlae.internal32
void test_load_a_u32(uint32_t *base, long stride) {
    muint32_t t = __riscv_th_mld_a_u32(base, stride, 4, 4);
    __riscv_th_mst_u32(base, stride, t, 4, 4);
}

// O2-LABEL: @test_load_a_u64
// O2: call target("riscv.matrix") @llvm.riscv.th.mlae.internal64
void test_load_a_u64(uint64_t *base, long stride) {
    muint64_t t = __riscv_th_mld_a_u64(base, stride, 4, 2);
    __riscv_th_mst_u64(base, stride, t, 4, 2);
}

// O2-LABEL: @test_load_b_f16
// O2: call target("riscv.matrix") @llvm.riscv.th.mlbe.internal16
void test_load_b_f16(_Float16 *base, long stride) {
    mfloat16_t t = __riscv_th_mld_b_f16(base, stride, 4, 4);
    __riscv_th_mst_f16(base, stride, t, 4, 4);
}

// O2-LABEL: @test_load_b_f64
// O2: call target("riscv.matrix") @llvm.riscv.th.mlbe.internal64
void test_load_b_f64(double *base, long stride) {
    mfloat64_t t = __riscv_th_mld_b_f64(base, stride, 2, 4);
    __riscv_th_mst_f64(base, stride, t, 4, 4);
}

// O2-LABEL: @test_load_acc_u8
// O2: call target("riscv.matrix") @llvm.riscv.th.mlce.internal8
void test_load_acc_u8(uint8_t *base, long stride) {
    muint8_t t = __riscv_th_mld_acc_u8(base, stride, 4, 4);
    __riscv_th_mst_u8(base, stride, t, 4, 4);
}

// O2-LABEL: @test_load_acc_u16
// O2: call target("riscv.matrix") @llvm.riscv.th.mlce.internal16
void test_load_acc_u16(uint16_t *base, long stride) {
    muint16_t t = __riscv_th_mld_acc_u16(base, stride, 4, 4);
    __riscv_th_mst_u16(base, stride, t, 4, 4);
}

// O2-LABEL: @test_load_acc_f32
// O2: call target("riscv.matrix") @llvm.riscv.th.mlce.internal32
void test_load_acc_f32(float *base, long stride) {
    mfloat32_t t = __riscv_th_mld_acc_f32(base, stride, 4, 4);
    __riscv_th_mst_f32(base, stride, t, 4, 4);
}

// ========================================================================
// 8. Transposed Load/Store - all element types
// ========================================================================

// O2-LABEL: @test_load_at_u8
// O2: call target("riscv.matrix") @llvm.riscv.th.mlate.internal8
void test_load_at_u8(uint8_t *base, long stride) {
    muint8_t t = __riscv_th_mld_at_u8(base, stride, 4, 8);
    __riscv_th_mst_at_u8(base, stride, t, 4, 8);
}

// O2-LABEL: @test_load_bt_f32
// O2: call target("riscv.matrix") @llvm.riscv.th.mlbte.internal32
void test_load_bt_f32(float *base, long stride) {
    mfloat32_t t = __riscv_th_mld_bt_f32(base, stride, 4, 4);
    __riscv_th_mst_bt_f32(base, stride, t, 4, 4);
}

// O2-LABEL: @test_load_ct_u32
// O2: call target("riscv.matrix") @llvm.riscv.th.mlcte.internal32
void test_load_ct_u32(uint32_t *base, long stride) {
    muint32_t t = __riscv_th_mld_ct_u32(base, stride, 4, 4);
    __riscv_th_mst_ct_u32(base, stride, t, 4, 4);
}

// O2-LABEL: @test_store_at_f64
// O2: call void @llvm.riscv.th.msate.internal64
void test_store_at_f64(double *base, long stride) {
    mfloat64_t t = __riscv_th_mld_at_f64(base, stride, 4, 2);
    __riscv_th_mst_at_f64(base, stride, t, 4, 2);
}

// O2-LABEL: @test_store_bt_i16
// O2: call void @llvm.riscv.th.msbte.internal16
void test_store_bt_i16(int16_t *base, long stride) {
    mint16_t t = __riscv_th_mld_bt_i16(base, stride, 4, 4);
    __riscv_th_mst_bt_i16(base, stride, t, 4, 4);
}

// O2-LABEL: @test_store_ct_f16
// O2: call void @llvm.riscv.th.mscte.internal16
void test_store_ct_f16(_Float16 *base, long stride) {
    mfloat16_t t = __riscv_th_mld_ct_f16(base, stride, 4, 4);
    __riscv_th_mst_ct_f16(base, stride, t, 4, 4);
}

// ========================================================================
// 9. Whole-Register Load/Store - all element types
// ========================================================================

// O2-LABEL: @test_whole_i8
// O2: call target("riscv.matrix") @llvm.riscv.th.mlme.internal8
// O2: call void @llvm.riscv.th.msme.internal8
void test_whole_i8(int8_t *base, long stride) {
    mint8_t t = __riscv_th_mld_m_i8(base, stride);
    __riscv_th_mst_m_i8(base, stride, t);
}

// O2-LABEL: @test_whole_i16
// O2: call target("riscv.matrix") @llvm.riscv.th.mlme.internal16
// O2: call void @llvm.riscv.th.msme.internal16
void test_whole_i16(int16_t *base, long stride) {
    mint16_t t = __riscv_th_mld_m_i16(base, stride);
    __riscv_th_mst_m_i16(base, stride, t);
}

// O2-LABEL: @test_whole_f16
// O2: call target("riscv.matrix") @llvm.riscv.th.mlme.internal16
// O2: call void @llvm.riscv.th.msme.internal16
void test_whole_f16(_Float16 *base, long stride) {
    mfloat16_t t = __riscv_th_mld_m_f16(base, stride);
    __riscv_th_mst_m_f16(base, stride, t);
}

// O2-LABEL: @test_whole_u32
// O2: call target("riscv.matrix") @llvm.riscv.th.mlme.internal32
// O2: call void @llvm.riscv.th.msme.internal32
void test_whole_u32(uint32_t *base, long stride) {
    muint32_t t = __riscv_th_mld_m_u32(base, stride);
    __riscv_th_mst_m_u32(base, stride, t);
}

// O2-LABEL: @test_whole_f64
// O2: call target("riscv.matrix") @llvm.riscv.th.mlme.internal64
// O2: call void @llvm.riscv.th.msme.internal64
void test_whole_f64(double *base, long stride) {
    mfloat64_t t = __riscv_th_mld_m_f64(base, stride);
    __riscv_th_mst_m_f64(base, stride, t);
}

// ========================================================================
// 10. Store A/B - all element types
// ========================================================================

// O2-LABEL: @test_store_a_i8
// O2: call void @llvm.riscv.th.msae.internal8
void test_store_a_i8(int8_t *base, long stride) {
    mint8_t t = __riscv_th_mld_a_i8(base, stride, 4, 8);
    __riscv_th_mst_a_i8(base, stride, t, 4, 8);
}

// O2-LABEL: @test_store_a_f16
// O2: call void @llvm.riscv.th.msae.internal16
void test_store_a_f16(_Float16 *base, long stride) {
    mfloat16_t t = __riscv_th_mld_a_f16(base, stride, 4, 4);
    __riscv_th_mst_a_f16(base, stride, t, 4, 4);
}

// O2-LABEL: @test_store_a_u64
// O2: call void @llvm.riscv.th.msae.internal64
void test_store_a_u64(uint64_t *base, long stride) {
    muint64_t t = __riscv_th_mld_a_u64(base, stride, 4, 2);
    __riscv_th_mst_a_u64(base, stride, t, 4, 2);
}

// O2-LABEL: @test_store_b_u8
// O2: call void @llvm.riscv.th.msbe.internal8
void test_store_b_u8(uint8_t *base, long stride) {
    muint8_t t = __riscv_th_mld_b_u8(base, stride, 8, 4);
    __riscv_th_mst_b_u8(base, stride, t, 8, 4);
}

// O2-LABEL: @test_store_b_f32
// O2: call void @llvm.riscv.th.msbe.internal32
void test_store_b_f32(float *base, long stride) {
    mfloat32_t t = __riscv_th_mld_b_f32(base, stride, 4, 4);
    __riscv_th_mst_b_f32(base, stride, t, 4, 4);
}

// O2-LABEL: @test_store_b_i64
// O2: call void @llvm.riscv.th.msbe.internal64
void test_store_b_i64(int64_t *base, long stride) {
    mint64_t t = __riscv_th_mld_b_i64(base, stride, 2, 4);
    __riscv_th_mst_b_i64(base, stride, t, 2, 4);
}

// ========================================================================
// 11. Immediate config macros
// ========================================================================

// O2-LABEL: @test_msettilemi
// O2: call void @llvm.riscv.th.msettilemi(i64 16)
void test_msettilemi(void) {
    __riscv_th_msettilemi(16);
}

// O2-LABEL: @test_msettileki
// O2: call void @llvm.riscv.th.msettileki(i64 8)
void test_msettileki(void) {
    __riscv_th_msettileki(8);
}

// O2-LABEL: @test_msettileni
// O2: call void @llvm.riscv.th.msettileni(i64 4)
void test_msettileni(void) {
    __riscv_th_msettileni(4);
}

// ========================================================================
// 12. Config functions - msetmrow_m and msetmrow_n return value
// ========================================================================

// O2-LABEL: @test_msetmrow_m_return
// O2: call void @llvm.riscv.th.msettilem(i64 %
// O2: ret i64 %
mrow_t test_msetmrow_m_return(mrow_t m) {
    return __riscv_th_msetmrow_m(m);
}

// O2-LABEL: @test_msetmrow_n_return
// O2: call void @llvm.riscv.th.msettilen(i64 %
// O2: ret i64 %
mrow_t test_msetmrow_n_return(mrow_t n) {
    return __riscv_th_msetmrow_n(n);
}

// O2-LABEL: @test_msetmcol_e8_return
// O2: call void @llvm.riscv.th.msettilek(i64 %
// O2: ret i64 %
mcol_t test_msetmcol_e8_return(mcol_t c) {
    return __riscv_th_msetmcol_e8(c);
}

// O2-LABEL: @test_msetmcol_e16_return
// O2: call void @llvm.riscv.th.msettilek(i64 %
// O2: ret i64 %
mcol_t test_msetmcol_e16_return(mcol_t c) {
    return __riscv_th_msetmcol_e16(c);
}

// O2-LABEL: @test_msetmcol_e64_return
// O2: call void @llvm.riscv.th.msettilek(i64 %
// O2: ret i64 %
mcol_t test_msetmcol_e64_return(mcol_t c) {
    return __riscv_th_msetmcol_e64(c);
}

// ========================================================================
// 13. CSR helper functions
// ========================================================================

// O2-LABEL: @test_xmlenb
// O2: call i64 asm sideeffect "csrr $0, th.xtlenb"
unsigned long test_xmlenb(void) {
    return __riscv_th_xmlenb();
}

// O2-LABEL: @test_xrlenb
// O2: call i64 asm sideeffect "csrr $0, th.xtrlenb"
unsigned long test_xrlenb(void) {
    return __riscv_th_xrlenb();
}

// O2-LABEL: @test_xmisa
// O2: call i64 asm sideeffect "csrr $0, th.xmisa"
unsigned long test_xmisa(void) {
    return __riscv_th_xmisa();
}

// O2-LABEL: @test_xmsize_alias
// O2: call i64 asm sideeffect "csrr $0, th.xmisa"
unsigned long test_xmsize_alias(void) {
    return __riscv_th_xmsize();
}

// ========================================================================
// 14. End-to-end: unsigned INT8 matmul pipeline
// ========================================================================

// O2-LABEL: @test_uint8_matmul_e2e
// O2: call target("riscv.matrix") @llvm.riscv.th.mlae.internal8
// O2: call target("riscv.matrix") @llvm.riscv.th.mlbe.internal8
// O2: call target("riscv.matrix") @llvm.riscv.th.mzero.internal
// O2: call target("riscv.matrix") @llvm.riscv.th.mmaccu.w.b.internal
// O2: call void @llvm.riscv.th.msce.internal32
void test_uint8_matmul_e2e(uint8_t *a, uint8_t *b, uint32_t *c, long stride,
                           mrow_t m, mcol_t k, mcol_t n) {
    muint8_t ta = __riscv_th_mld_a_u8(a, stride, m, k);
    muint8_t tb = __riscv_th_mld_b_u8(b, stride, k, n);
    muint32_t acc = __riscv_th_mzeros_u32(m, n);
    muint32_t result = __riscv_th_mmaccu_w_b(acc, ta, tb, m, k, n);
    __riscv_th_mst_u32(c, stride, result, m, n);
}

// ========================================================================
// 15. End-to-end: FP64 matmul pipeline
// ========================================================================

// O2-LABEL: @test_fp64_matmul_e2e
// O2: call target("riscv.matrix") @llvm.riscv.th.mlae.internal64
// O2: call target("riscv.matrix") @llvm.riscv.th.mlbe.internal64
// O2: call target("riscv.matrix") @llvm.riscv.th.mlce.internal64
// O2: call target("riscv.matrix") @llvm.riscv.th.mfmacc.d.internal
// O2: call void @llvm.riscv.th.msce.internal64
void test_fp64_matmul_e2e(double *a, double *b, double *c, long stride,
                          mrow_t m, mcol_t k, mcol_t n) {
    mfloat64_t ta = __riscv_th_mld_a_f64(a, stride, m, k);
    mfloat64_t tb = __riscv_th_mld_b_f64(b, stride, k, n);
    mfloat64_t tc = __riscv_th_mld_acc_f64(c, stride, m, n);
    mfloat64_t result = __riscv_th_mfmacc_d(tc, ta, tb, m, k, n);
    __riscv_th_mst_f64(c, stride, result, m, n);
}

// ========================================================================
// 16. End-to-end: FP16->FP32 widening matmul
// ========================================================================

// O2-LABEL: @test_fp16_widen_matmul_e2e
// O2: call target("riscv.matrix") @llvm.riscv.th.mlae.internal16
// O2: call target("riscv.matrix") @llvm.riscv.th.mlbe.internal16
// O2: call target("riscv.matrix") @llvm.riscv.th.mzero.internal
// O2: call target("riscv.matrix") @llvm.riscv.th.mfmacc.s.h.internal
// O2: call void @llvm.riscv.th.msce.internal32
void test_fp16_widen_matmul_e2e(_Float16 *a, _Float16 *b, float *c,
                                long stride, mrow_t m, mcol_t k, mcol_t n) {
    mfloat16_t ta = __riscv_th_mld_a_f16(a, stride, m, k);
    mfloat16_t tb = __riscv_th_mld_b_f16(b, stride, k, n);
    mfloat32_t acc = __riscv_th_mzeros_f32(m, n);
    mfloat32_t result = __riscv_th_mfmacc_s_h(acc, ta, tb, m, k, n);
    __riscv_th_mst_f32(c, stride, result, m, n);
}

// ========================================================================
// 17. End-to-end: transposed load + matmul
// ========================================================================

// O2-LABEL: @test_transposed_load_matmul
// O2: call target("riscv.matrix") @llvm.riscv.th.mlate.internal32
// O2: call target("riscv.matrix") @llvm.riscv.th.mlbte.internal32
// O2: call target("riscv.matrix") @llvm.riscv.th.mzero.internal
// O2: call target("riscv.matrix") @llvm.riscv.th.mfmacc.s.internal
// O2: call void @llvm.riscv.th.mscte.internal32
void test_transposed_load_matmul(float *a, float *b, float *c, long stride,
                                 mrow_t m, mcol_t k, mcol_t n) {
    mfloat32_t ta = __riscv_th_mld_at_f32(a, stride, m, k);
    mfloat32_t tb = __riscv_th_mld_bt_f32(b, stride, k, n);
    mfloat32_t acc = __riscv_th_mzeros_f32(m, n);
    mfloat32_t result = __riscv_th_mfmacc_s(acc, ta, tb, m, k, n);
    __riscv_th_mst_ct_f32(c, stride, result, m, n);
}

// ========================================================================
// 18. End-to-end: mixed-sign INT8 matmul
// ========================================================================

// O2-LABEL: @test_mixed_sign_matmul
// O2: call target("riscv.matrix") @llvm.riscv.th.mmaccus.w.b.internal
void test_mixed_sign_matmul(uint8_t *a, int8_t *b, int32_t *c, long stride,
                            mrow_t m, mcol_t k, mcol_t n) {
    muint8_t ta = __riscv_th_mld_a_u8(a, stride, m, k);
    mint8_t tb = __riscv_th_mld_b_i8(b, stride, k, n);
    mint32_t acc = __riscv_th_mzeros_i32(m, n);
    mint32_t result = __riscv_th_mmaccus_w_b(acc, ta, tb, m, k, n);
    __riscv_th_mst_i32(c, stride, result, m, n);
}

// ========================================================================
// 19. End-to-end: partial matmul
// ========================================================================

// O2-LABEL: @test_partial_matmul
// O2: call target("riscv.matrix") @llvm.riscv.th.pmmacc.w.b.internal
void test_partial_matmul(int8_t *a, int8_t *b, int32_t *c, long stride,
                         mrow_t m, mcol_t k, mcol_t n) {
    mint8_t ta = __riscv_th_mld_a_i8(a, stride, m, k);
    mint8_t tb = __riscv_th_mld_b_i8(b, stride, k, n);
    mint32_t acc = __riscv_th_mzeros_i32(m, n);
    mint32_t result = __riscv_th_pmmacc_w_b(acc, ta, tb, m, k, n);
    __riscv_th_mst_i32(c, stride, result, m, n);
}

// ========================================================================
// 20. End-to-end: whole-register context save/restore
// ========================================================================

// O2-LABEL: @test_context_save_restore
// O2: call target("riscv.matrix") @llvm.riscv.th.mlae.internal32
// O2: call void @llvm.riscv.th.msme.internal32
// O2: call target("riscv.matrix") @llvm.riscv.th.mlme.internal32
// O2: call void @llvm.riscv.th.msce.internal32
void test_context_save_restore(float *base, long stride, float *save_buf,
                               long save_stride) {
    // Load tile, save to scratch, restore, store as accumulator
    mfloat32_t tile = __riscv_th_mld_a_f32(base, stride, 4, 4);
    __riscv_th_mst_m_f32(save_buf, save_stride, tile);
    mfloat32_t restored = __riscv_th_mld_m_f32(save_buf, save_stride);
    __riscv_th_mst_f32(base, stride, restored, 4, 4);
}
