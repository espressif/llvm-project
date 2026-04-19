let TargetPrefix = "riscv" in {
// Load with Post-Increment (Immediate offset)
// Returns: { vector data, updated pointer }
// Parameters: [ original pointer, offset (immediate) ]
def int_riscv_esp_vld_128_ip_m : Intrinsic<[llvm_v16i8_ty, llvm_ptr_ty], 
                                         [llvm_ptr_ty, llvm_i32_ty], 
                                         [IntrReadMem, IntrArgMemOnly]>;

// ESP.LD.128.USAR.IP (_m version) - Load 128-bit data and write SAR_BYTES register
// Returns: vector + updated pointer + SAR_BYTES (32-bit, only low 4 bits used)
// Parameters: [pointer, immediate offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
// Note: SAR_BYTES[3:0] = rs1[3:0] (hardware automatically extracts only low 4 bits from rs1)
def int_riscv_esp_ld_128_usar_ip_m : Intrinsic<[llvm_v16i8_ty, llvm_ptr_ty, llvm_i32_ty], 
                                         [llvm_ptr_ty, llvm_i32_ty], 
                                         [IntrReadMem, IntrArgMemOnly]>;

// Load with Post-Increment (Register offset)
// Returns: { vector data, updated pointer }
// Parameters: [ original pointer, offset (register) ]
def int_riscv_esp_vld_128_xp_m : Intrinsic<[llvm_v16i8_ty, llvm_ptr_ty], 
                                         [llvm_ptr_ty, llvm_i32_ty], 
                                         [IntrReadMem, IntrArgMemOnly]>;
// Load 128-bit with SAR_BYTES update (Register offset)
// ESP.LD.128.USAR.XP (_m version) - Load 128-bit data and write SAR_BYTES register
// Returns: vector + updated pointer + SAR_BYTES (32-bit, only low 4 bits used)
// Parameters: [pointer, register offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
// Note: SAR_BYTES[3:0] = rs1[3:0] (hardware automatically extracts only low 4 bits from rs1)
def int_riscv_esp_ld_128_usar_xp_m : Intrinsic<[llvm_v16i8_ty, llvm_ptr_ty, llvm_i32_ty],
                                         [llvm_ptr_ty, llvm_i32_ty],
                                         [IntrReadMem, IntrArgMemOnly]>;

// Store with Post-Increment (Immediate offset)
// Returns: updated pointer
// Parameters: [ vector data, original pointer, offset (immediate) ]
def int_riscv_esp_vst_128_ip_m : Intrinsic<[llvm_ptr_ty], 
                                         [llvm_v16i8_ty, llvm_ptr_ty, llvm_i32_ty], 
                                         [IntrWriteMem, IntrArgMemOnly]>;

// Store with Post-Increment (Register offset)
// Returns: updated pointer
// Parameters: [ vector data, original pointer, offset (register) ]
def int_riscv_esp_vst_128_xp_m : Intrinsic<[llvm_ptr_ty], 
                                         [llvm_v16i8_ty, llvm_ptr_ty, llvm_i32_ty], 
                                         [IntrWriteMem, IntrArgMemOnly]>;

// Load H.64 with Post-Increment (Immediate offset)
// Returns: { 64-bit data, updated pointer }
// Parameters: [ original pointer, offset (immediate) ]
// H version: load to upper 64-bit segment of qu (qu[127:64])
def int_riscv_esp_vld_h_64_ip_m : Intrinsic<[llvm_v8i8_ty, llvm_ptr_ty], 
                                            [llvm_ptr_ty, llvm_i32_ty], 
                                            [IntrReadMem, IntrArgMemOnly]>;

// Load H.64 with Post-Increment (Register offset)
// Returns: { 64-bit data, updated pointer }
// Parameters: [ original pointer, offset (register) ]
def int_riscv_esp_vld_h_64_xp_m : Intrinsic<[llvm_v8i8_ty, llvm_ptr_ty], 
                                            [llvm_ptr_ty, llvm_i32_ty], 
                                            [IntrReadMem, IntrArgMemOnly]>;

// Load L.64 with Post-Increment (Immediate offset)
// Returns: { 64-bit data, updated pointer }
// Parameters: [ original pointer, offset (immediate) ]
// L version: load to lower 64-bit segment of qu (qu[63:0])
def int_riscv_esp_vld_l_64_ip_m : Intrinsic<[llvm_v8i8_ty, llvm_ptr_ty], 
                                            [llvm_ptr_ty, llvm_i32_ty], 
                                            [IntrReadMem, IntrArgMemOnly]>;

// Load L.64 with Post-Increment (Register offset)
// Returns: { 64-bit data, updated pointer }
// Parameters: [ original pointer, offset (register) ]
def int_riscv_esp_vld_l_64_xp_m : Intrinsic<[llvm_v8i8_ty, llvm_ptr_ty], 
                                            [llvm_ptr_ty, llvm_i32_ty], 
                                            [IntrReadMem, IntrArgMemOnly]>;

// Store H.64 with Post-Increment (Immediate offset)
// Returns: updated pointer
// Parameters: [ 64-bit data, original pointer, offset (immediate) ]
// H version: store upper 64-bit segment of qu (qu[127:64])
def int_riscv_esp_vst_h_64_ip_m : Intrinsic<[llvm_ptr_ty], 
                                             [llvm_v8i8_ty, llvm_ptr_ty, llvm_i32_ty], 
                                             [IntrWriteMem, IntrArgMemOnly]>;

// Store H.64 with Post-Increment (Register offset)
// Returns: updated pointer
// Parameters: [ 64-bit data, original pointer, offset (register) ]
def int_riscv_esp_vst_h_64_xp_m : Intrinsic<[llvm_ptr_ty], 
                                             [llvm_v8i8_ty, llvm_ptr_ty, llvm_i32_ty], 
                                             [IntrWriteMem, IntrArgMemOnly]>;

// Store L.64 with Post-Increment (Immediate offset)
// Returns: updated pointer
// Parameters: [ 64-bit data, original pointer, offset (immediate) ]
// L version: store lower 64-bit segment of qu (qu[63:0])
def int_riscv_esp_vst_l_64_ip_m : Intrinsic<[llvm_ptr_ty], 
                                             [llvm_v8i8_ty, llvm_ptr_ty, llvm_i32_ty], 
                                             [IntrWriteMem, IntrArgMemOnly]>;

// Store L.64 with Post-Increment (Register offset)
// Returns: updated pointer
// Parameters: [ 64-bit data, original pointer, offset (register) ]
def int_riscv_esp_vst_l_64_xp_m : Intrinsic<[llvm_ptr_ty], 
                                             [llvm_v8i8_ty, llvm_ptr_ty, llvm_i32_ty], 
                                             [IntrWriteMem, IntrArgMemOnly]>;

// VLDBC.8.IP - Broadcast 8-bit value to v16i8
def int_riscv_esp_vldbc_8_ip_m : Intrinsic<[llvm_v16i8_ty, llvm_ptr_ty], 
                                            [llvm_ptr_ty, llvm_i32_ty], 
                                            [IntrReadMem, IntrArgMemOnly]>;
// VLDBC.8.XP
def int_riscv_esp_vldbc_8_xp_m : Intrinsic<[llvm_v16i8_ty, llvm_ptr_ty], 
                                            [llvm_ptr_ty, llvm_i32_ty], 
                                            [IntrReadMem, IntrArgMemOnly]>;

def int_riscv_esp_vldbc_16_ip_m : Intrinsic<[llvm_v8i16_ty, llvm_ptr_ty], 
                                             [llvm_ptr_ty, llvm_i32_ty], 
                                             [IntrReadMem, IntrArgMemOnly]>;
// VLDBC.16.XP
def int_riscv_esp_vldbc_16_xp_m : Intrinsic<[llvm_v8i16_ty, llvm_ptr_ty], 
                                             [llvm_ptr_ty, llvm_i32_ty], 
                                             [IntrReadMem, IntrArgMemOnly]>;
// VLDBC.32.IP - Broadcast 32-bit value to v4i32
def int_riscv_esp_vldbc_32_ip_m : Intrinsic<[llvm_v4i32_ty, llvm_ptr_ty], 
                                             [llvm_ptr_ty, llvm_i32_ty], 
                                             [IntrReadMem, IntrArgMemOnly]>;
// VLDBC.32.XP
def int_riscv_esp_vldbc_32_xp_m : Intrinsic<[llvm_v4i32_ty, llvm_ptr_ty], 
                                             [llvm_ptr_ty, llvm_i32_ty], 
                                             [IntrReadMem, IntrArgMemOnly]>;

// VLDEXT - Vector Load and Extend
// Loads data and extends it (sign or zero extend)
// Returns struct containing two vectors and updated pointer
// VLDEXT.S16.IP - Sign extend 16-bit to 32-bit
def int_riscv_esp_vldext_s16_ip_m : Intrinsic<[llvm_v4i32_ty, llvm_v4i32_ty, llvm_ptr_ty], 
                                               [llvm_ptr_ty, llvm_i32_ty], 
                                               [IntrReadMem, IntrArgMemOnly]>;
// VLDEXT.S16.XP
def int_riscv_esp_vldext_s16_xp_m : Intrinsic<[llvm_v4i32_ty, llvm_v4i32_ty, llvm_ptr_ty], 
                                               [llvm_ptr_ty, llvm_i32_ty], 
                                               [IntrReadMem, IntrArgMemOnly]>;
// VLDEXT.S8.IP - Sign extend 8-bit to 16-bit
def int_riscv_esp_vldext_s8_ip_m : Intrinsic<[llvm_v8i16_ty, llvm_v8i16_ty, llvm_ptr_ty], 
                                              [llvm_ptr_ty, llvm_i32_ty], 
                                              [IntrReadMem, IntrArgMemOnly]>;
// VLDEXT.S8.XP
def int_riscv_esp_vldext_s8_xp_m : Intrinsic<[llvm_v8i16_ty, llvm_v8i16_ty, llvm_ptr_ty], 
                                              [llvm_ptr_ty, llvm_i32_ty], 
                                              [IntrReadMem, IntrArgMemOnly]>;
// VLDEXT.U16.IP - Zero extend 16-bit to 32-bit
def int_riscv_esp_vldext_u16_ip_m : Intrinsic<[llvm_v4i32_ty, llvm_v4i32_ty, llvm_ptr_ty], 
                                               [llvm_ptr_ty, llvm_i32_ty], 
                                               [IntrReadMem, IntrArgMemOnly]>;
// VLDEXT.U16.XP
def int_riscv_esp_vldext_u16_xp_m : Intrinsic<[llvm_v4i32_ty, llvm_v4i32_ty, llvm_ptr_ty], 
                                               [llvm_ptr_ty, llvm_i32_ty], 
                                               [IntrReadMem, IntrArgMemOnly]>;
// VLDEXT.U8.IP - Zero extend 8-bit to 16-bit
def int_riscv_esp_vldext_u8_ip_m : Intrinsic<[llvm_v8i16_ty, llvm_v8i16_ty, llvm_ptr_ty], 
                                              [llvm_ptr_ty, llvm_i32_ty], 
                                              [IntrReadMem, IntrArgMemOnly]>;
// VLDEXT.U8.XP
def int_riscv_esp_vldext_u8_xp_m : Intrinsic<[llvm_v8i16_ty, llvm_v8i16_ty, llvm_ptr_ty], 
                                              [llvm_ptr_ty, llvm_i32_ty], 
                                              [IntrReadMem, IntrArgMemOnly]>;
}
