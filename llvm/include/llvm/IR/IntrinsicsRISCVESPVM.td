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

// ESP.LD.UA.STATE.IP (_m version) - Load 128-bit data to UA_STATE[127:0]
// Returns: updated pointer + new UA_STATE (128-bit)
// Parameters: [current UA_STATE (passthru), pointer, immediate offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_ld_ua_state_ip_m : Intrinsic<
    [llvm_v16i8_ty, llvm_ptr_ty],        // Returns: new UA_STATE (128-bit) + pointer
    [llvm_v16i8_ty,                       // Current UA_STATE (passthru)
     llvm_ptr_ty,                         // pointer
     llvm_i32_ty],                        // offset
    [IntrReadMem, IntrArgMemOnly]>;       // No IntrHasSideEffects

// ESP.ST.UA.STATE.IP (_m version) - Store UA_STATE register to memory
// Returns: updated pointer + new UA_STATE (explicit state passing)
// Parameters: [current UA_STATE (passthru), pointer, immediate offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_st_ua_state_ip_m : Intrinsic<
    [llvm_v16i8_ty,                       // Returns: new UA_STATE (explicit state passing)
     llvm_ptr_ty],                        // Returns: pointer
    [llvm_v16i8_ty,                       // Current UA_STATE (passthru)
     llvm_ptr_ty,                         // pointer
     llvm_i32_ty],                        // offset
    [IntrWriteMem, IntrArgMemOnly]>;       // No IntrHasSideEffects

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

// VSADDS - Signed 16-bit
// Returns: v8i16 (vector saturating addition on 16-bit signed data)
// Parameters: [ v8i16 vector qx, i32 rs1 ]
def int_riscv_esp_vsadds_s16_m : Intrinsic<[llvm_v8i16_ty], 
                                           [llvm_v8i16_ty, llvm_i32_ty], 
                                           [IntrNoMem]>;

// VSADDS - Signed 8-bit
// Returns: v16i8 (vector saturating addition on 8-bit signed data)
// Parameters: [ v16i8 vector qx, i32 rs1 ]
def int_riscv_esp_vsadds_s8_m : Intrinsic<[llvm_v16i8_ty], 
                                          [llvm_v16i8_ty, llvm_i32_ty], 
                                          [IntrNoMem]>;

// VSADDS - Unsigned 16-bit
// Returns: v8i16 (vector saturating addition on 16-bit unsigned data)
// Parameters: [ v8i16 vector qx, i32 rs1 ]
def int_riscv_esp_vsadds_u16_m : Intrinsic<[llvm_v8i16_ty], 
                                           [llvm_v8i16_ty, llvm_i32_ty], 
                                           [IntrNoMem]>;

// VSADDS - Unsigned 8-bit
// Returns: v16i8 (vector saturating addition on 8-bit unsigned data)
// Parameters: [ v16i8 vector qx, i32 rs1 ]
def int_riscv_esp_vsadds_u8_m : Intrinsic<[llvm_v16i8_ty], 
                                         [llvm_v16i8_ty, llvm_i32_ty], 
                                         [IntrNoMem]>;

// VSSUBS - Signed 16-bit
// Returns: v8i16 (vector saturating subtraction on 16-bit signed data)
// Parameters: [ v8i16 vector qx, i32 rs1 ]
def int_riscv_esp_vssubs_s16_m : Intrinsic<[llvm_v8i16_ty], 
                                           [llvm_v8i16_ty, llvm_i32_ty], 
                                           [IntrNoMem]>;

// VSSUBS - Signed 8-bit
// Returns: v16i8 (vector saturating subtraction on 8-bit signed data)
// Parameters: [ v16i8 vector qx, i32 rs1 ]
def int_riscv_esp_vssubs_s8_m : Intrinsic<[llvm_v16i8_ty], 
                                          [llvm_v16i8_ty, llvm_i32_ty], 
                                          [IntrNoMem]>;

// VSSUBS - Unsigned 16-bit
// Returns: v8i16 (vector saturating subtraction on 16-bit unsigned data)
// Parameters: [ v8i16 vector qx, i32 rs1 ]
def int_riscv_esp_vssubs_u16_m : Intrinsic<[llvm_v8i16_ty], 
                                           [llvm_v8i16_ty, llvm_i32_ty], 
                                           [IntrNoMem]>;

// VSSUBS - Unsigned 8-bit
// Returns: v16i8 (vector saturating subtraction on 8-bit unsigned data)
// Parameters: [ v16i8 vector qx, i32 rs1 ]
def int_riscv_esp_vssubs_u8_m : Intrinsic<[llvm_v16i8_ty], 
                                          [llvm_v16i8_ty, llvm_i32_ty], 
                                          [IntrNoMem]>;

// VCLAMP - Signed 16-bit
// Returns: v8i16 (vector clamp on 16-bit signed data)
// Parameters: [ v8i16 vector qx, i32 sel16 (immediate) ]
def int_riscv_esp_vclamp_s16_m : Intrinsic<[llvm_v8i16_ty], 
                                           [llvm_v8i16_ty, llvm_i32_ty], 
                                           [IntrNoMem, ImmArg<ArgIndex<1>>]>;

// VPRELU - Signed 16-bit
// Returns: v8i16 (if qx segment value <= 0, then qz = (qx * qy) >> rs1[4:0], else qz = qx)
// Parameters: [ v8i16 vector qx, v8i16 vector qy, i32 rs1 ]
def int_riscv_esp_vprelu_s16_m : Intrinsic<[llvm_v8i16_ty], 
                                            [llvm_v8i16_ty, llvm_v8i16_ty, llvm_i32_ty], 
                                            [IntrNoMem]>;

// VPRELU - Signed 8-bit
// Returns: v16i8 (if qx segment value <= 0, then qz = (qx * qy) >> rs1[4:0], else qz = qx)
// Parameters: [ v16i8 vector qx, v16i8 vector qy, i32 rs1 ]
def int_riscv_esp_vprelu_s8_m : Intrinsic<[llvm_v16i8_ty], 
                                          [llvm_v16i8_ty, llvm_v16i8_ty, llvm_i32_ty], 
                                          [IntrNoMem]>;

// VZIP.8 - Vector zip (interleave) 8-bit elements
// Returns: {v16i8, v16i8} (modified qx and qy)
// Parameters: [ v16i8 vector qx, v16i8 vector qy ]
def int_riscv_esp_vzip_8_m : Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty], 
                                       [llvm_v16i8_ty, llvm_v16i8_ty], 
                                       [IntrNoMem]>;

// VZIP.16 - Vector zip (interleave) 16-bit elements
// Returns: {v8i16, v8i16} (modified qx and qy)
// Parameters: [ v8i16 vector qx, v8i16 vector qy ]
def int_riscv_esp_vzip_16_m : Intrinsic<[llvm_v8i16_ty, llvm_v8i16_ty], 
                                        [llvm_v8i16_ty, llvm_v8i16_ty], 
                                        [IntrNoMem]>;

// VZIP.32 - Vector zip (interleave) 32-bit elements
// Returns: {v4i32, v4i32} (modified qx and qy)
// Parameters: [ v4i32 vector qx, v4i32 vector qy ]
def int_riscv_esp_vzip_32_m : Intrinsic<[llvm_v4i32_ty, llvm_v4i32_ty], 
                                        [llvm_v4i32_ty, llvm_v4i32_ty], 
                                        [IntrNoMem]>;

// VUNZIP.8 - Vector unzip (deinterleave) 8-bit elements
// Returns: {v16i8, v16i8} (modified qx and qy)
// Parameters: [ v16i8 vector qx, v16i8 vector qy ]
def int_riscv_esp_vunzip_8_m : Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty], 
                                         [llvm_v16i8_ty, llvm_v16i8_ty], 
                                         [IntrNoMem]>;

// VZIPT.8 - Vector zip transpose (3 vectors) 8-bit elements
// Returns: {v16i8, v16i8, v16i8} (modified qx, qy, qw)
// Parameters: [ v16i8 vector qx, v16i8 vector qy, v16i8 vector qw ]
def int_riscv_esp_vzipt_8_m : Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty], 
                                        [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty], 
                                        [IntrNoMem]>;

// VEXT.S8 - Sign extend 8-bit to 16-bit
// Returns: {v8i16, v8i16} (two 8x16-bit vectors)
// Parameters: [ v16i8 vector qw ]
def int_riscv_esp_vext_s8_m : Intrinsic<[llvm_v8i16_ty, llvm_v8i16_ty], 
                                        [llvm_v16i8_ty], 
                                        [IntrNoMem]>;

// VEXT.S16 - Sign extend 16-bit to 32-bit
// Returns: {v4i32, v4i32} (two 4x32-bit vectors)
// Parameters: [ v8i16 vector qw ]
def int_riscv_esp_vext_s16_m : Intrinsic<[llvm_v4i32_ty, llvm_v4i32_ty], 
                                         [llvm_v8i16_ty], 
                                         [IntrNoMem]>;

// VEXT.U8 - Zero extend 8-bit to 16-bit
// Returns: {v8i16, v8i16} (two 8x16-bit vectors)
// Parameters: [ v16i8 vector qw ]
def int_riscv_esp_vext_u8_m : Intrinsic<[llvm_v8i16_ty, llvm_v8i16_ty], 
                                        [llvm_v16i8_ty], 
                                        [IntrNoMem]>;

// VEXT.U16 - Zero extend 16-bit to 32-bit
// Returns: {v4i32, v4i32} (two 4x32-bit vectors)
// Parameters: [ v8i16 vector qw ]
def int_riscv_esp_vext_u16_m : Intrinsic<[llvm_v4i32_ty, llvm_v4i32_ty], 
                                         [llvm_v8i16_ty], 
                                         [IntrNoMem]>;

// VSAT - Signed 8-bit
// Returns: v16i8 (vector saturation on 8-bit signed segments)
// Parameters: [ v16i8 vector qx, i32 rs1, i32 rs2 ]
def int_riscv_esp_vsat_s8_m : Intrinsic<[llvm_v16i8_ty], 
                                         [llvm_v16i8_ty, llvm_i32_ty, llvm_i32_ty], 
                                         [IntrNoMem]>;

// VSAT - Unsigned 8-bit
// Returns: v16i8 (vector saturation on 8-bit unsigned segments)
// Parameters: [ v16i8 vector qx, i32 rs1, i32 rs2 ]
def int_riscv_esp_vsat_u8_m : Intrinsic<[llvm_v16i8_ty], 
                                         [llvm_v16i8_ty, llvm_i32_ty, llvm_i32_ty], 
                                         [IntrNoMem]>;

// MAX - Signed 8-bit reduction
// Returns: i32 (maximum value from vector)
// Parameters: [ v16i8 vector qw ]
def int_riscv_esp_max_s8_a_m : Intrinsic<[llvm_i32_ty], 
                                         [llvm_v16i8_ty], 
                                         [IntrNoMem]>;

// MAX - Signed 16-bit reduction
// Returns: i32 (maximum value from vector)
// Parameters: [ v8i16 vector qw ]
def int_riscv_esp_max_s16_a_m : Intrinsic<[llvm_i32_ty], 
                                          [llvm_v8i16_ty], 
                                          [IntrNoMem]>;

// MAX - Unsigned 8-bit reduction
// Returns: i32 (maximum value from vector)
// Parameters: [ v16i8 vector qw ]
def int_riscv_esp_max_u8_a_m : Intrinsic<[llvm_i32_ty], 
                                         [llvm_v16i8_ty], 
                                         [IntrNoMem]>;

// MAX - Unsigned 16-bit reduction
// Returns: i32 (maximum value from vector)
// Parameters: [ v8i16 vector qw ]
def int_riscv_esp_max_u16_a_m : Intrinsic<[llvm_i32_ty], 
                                          [llvm_v8i16_ty], 
                                          [IntrNoMem]>;

// MIN - Signed 8-bit reduction
// Returns: i32 (minimum value from vector)
// Parameters: [ v16i8 vector qw ]
def int_riscv_esp_min_s8_a_m : Intrinsic<[llvm_i32_ty], 
                                         [llvm_v16i8_ty], 
                                         [IntrNoMem]>;

// MIN - Signed 16-bit reduction
// Returns: i32 (minimum value from vector)
// Parameters: [ v8i16 vector qw ]
def int_riscv_esp_min_s16_a_m : Intrinsic<[llvm_i32_ty], 
                                          [llvm_v8i16_ty], 
                                          [IntrNoMem]>;

// MIN - Unsigned 8-bit reduction
// Returns: i32 (minimum value from vector)
// Parameters: [ v16i8 vector qw ]
def int_riscv_esp_min_u8_a_m : Intrinsic<[llvm_i32_ty], 
                                         [llvm_v16i8_ty], 
                                         [IntrNoMem]>;

// MIN - Unsigned 16-bit reduction
// Returns: i32 (minimum value from vector)
// Parameters: [ v8i16 vector qw ]
def int_riscv_esp_min_u16_a_m : Intrinsic<[llvm_i32_ty], 
                                          [llvm_v8i16_ty], 
                                          [IntrNoMem]>;

// Vector Add - Signed 32-bit
// Returns: v4i32 (result vector)
// Parameters: [ v4i32 vector qx, v4i32 vector qy ]
def int_riscv_esp_vadd_s32_m : Intrinsic<[llvm_v4i32_ty], 
                                         [llvm_v4i32_ty, llvm_v4i32_ty], 
                                         [IntrNoMem]>;

// Vector Add - Unsigned 32-bit
// Returns: v4i32 (result vector)
// Parameters: [ v4i32 vector qx, v4i32 vector qy ]
def int_riscv_esp_vadd_u32_m : Intrinsic<[llvm_v4i32_ty], 
                                         [llvm_v4i32_ty, llvm_v4i32_ty], 
                                         [IntrNoMem]>;

// Vector Subtract - Signed 32-bit
// Returns: v4i32 (result vector)
// Parameters: [ v4i32 vector qx, v4i32 vector qy ]
def int_riscv_esp_vsub_s32_m : Intrinsic<[llvm_v4i32_ty], 
                                         [llvm_v4i32_ty, llvm_v4i32_ty], 
                                         [IntrNoMem]>;

// Vector Max - Signed 32-bit
// Returns: v4i32 (result vector)
// Parameters: [ v4i32 vector qx, v4i32 vector qy ]
def int_riscv_esp_vmax_s32_m : Intrinsic<[llvm_v4i32_ty], 
                                         [llvm_v4i32_ty, llvm_v4i32_ty], 
                                         [IntrNoMem]>;

// Vector Absolute Value - 32-bit
// Returns: v4i32 (result vector)
// Parameters: [ v4i32 vector qy ]
def int_riscv_esp_vabs_32_m : Intrinsic<[llvm_v4i32_ty], 
                                        [llvm_v4i32_ty], 
                                        [IntrNoMem]>;

// MAX - Signed 32-bit reduction
// Returns: i32 (maximum value from vector)
// Parameters: [ v4i32 vector qw ]
def int_riscv_esp_max_s32_a_m : Intrinsic<[llvm_i32_ty], 
                                         [llvm_v4i32_ty], 
                                         [IntrNoMem]>;

// MAX - Unsigned 32-bit reduction
// Returns: i32 (maximum value from vector)
// Parameters: [ v4i32 vector qw ]
def int_riscv_esp_max_u32_a_m : Intrinsic<[llvm_i32_ty], 
                                          [llvm_v4i32_ty], 
                                          [IntrNoMem]>;

// MIN - Signed 32-bit reduction
// Returns: i32 (minimum value from vector)
// Parameters: [ v4i32 vector qw ]
def int_riscv_esp_min_s32_a_m : Intrinsic<[llvm_i32_ty], 
                                         [llvm_v4i32_ty], 
                                         [IntrNoMem]>;

// MIN - Unsigned 32-bit reduction
// Returns: i32 (minimum value from vector)
// Parameters: [ v4i32 vector qw ]
def int_riscv_esp_min_u32_a_m : Intrinsic<[llvm_i32_ty], 
                                          [llvm_v4i32_ty], 
                                          [IntrNoMem]>;

// MOVX.W.FFT.BIT.WIDTH - Write FFT bit width register
// Returns: i32 (FFT_BIT_WIDTH value, 32-bit unsigned, only low 4 bits used)
// Parameters: [ i32 value to write (rs1, 32-bit unsigned) ]
def int_riscv_esp_movx_w_fft_bit_width_m : Intrinsic<[llvm_i32_ty], 
                                                     [llvm_i32_ty], 
                                                     [IntrNoMem]>;

// MOVX.R.FFT.BIT.WIDTH - Read FFT bit width register
// Returns: i32 (rd[31:0] = {28'b0, FFT_BIT_WIDTH[3:0]}, 32-bit unsigned)
// Parameters: [ i32 FFT_BIT_WIDTH (input, 32-bit unsigned, only low 4 bits used) ]
def int_riscv_esp_movx_r_fft_bit_width_m : Intrinsic<[llvm_i32_ty], 
                                                     [llvm_i32_ty], 
                                                     [IntrNoMem]>;

// FFT.R2BF.S16 - FFT Radix-2 Butterfly operation
// Has 2 outputs (qz, qv) and 3 inputs (qx, qy, sel2)
// Returns a struct containing both output vectors
def int_riscv_esp_fft_r2bf_s16_m : Intrinsic<[llvm_v8i16_ty, llvm_v8i16_ty], 
                                                 [llvm_v8i16_ty, llvm_v8i16_ty, llvm_i32_ty], 
                                                 [IntrNoMem]>;

// FFT.AMS.S16.LD.INCP - FFT Add-Multiply-Subtract with load and increment
// Returns struct {v16i8 qu, v8i16 qz, v8i16 qv, i8* ptr}
// Inputs: qx (v8i16), qy (v8i16), qw (v8i16), rs1 (ptr), sel2 (i32)
// FFT.AMS.S16.LD.INCP (_m version) - FFT AMS with explicit SAR state passing
// Returns struct {v16i8 qu, v8i16 qz, v8i16 qv, i8* ptr}
// Inputs: qx (v8i16), qy (v8i16), qw (v8i16), rs1 (ptr), sel2 (i32), sar (i64)
def int_riscv_esp_fft_ams_s16_ld_incp_m : Intrinsic<[llvm_v16i8_ty, llvm_v8i16_ty, llvm_v8i16_ty, llvm_ptr_ty],
                                                      [llvm_v8i16_ty, llvm_v8i16_ty, llvm_v8i16_ty, llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
                                                      [IntrReadMem, IntrArgMemOnly]>;

// FFT.R2BF.S16.ST.INCP - Returns struct {v8i16 qz, i8* ptr}
// Inputs: qx (v8i16), qy (v8i16), rs1 (ptr), sel4 (i32)
def int_riscv_esp_fft_r2bf_s16_st_incp_m : Intrinsic<[llvm_v8i16_ty, llvm_ptr_ty],
                                                      [llvm_v8i16_ty, llvm_v8i16_ty, llvm_ptr_ty, llvm_i32_ty],
                                                      [IntrWriteMem, IntrArgMemOnly]>;

// FFT.AMS.S16.LD.R32.DECP - Returns struct {v16i8 qu, v8i16 qz, v8i16 qv, i8* ptr}
// Inputs: qx (v8i16), qy (v8i16), qw (v8i16), rs1 (ptr), sel2 (i32)
// FFT.AMS.S16.LD.R32.DECP (_m version) - FFT AMS with load, big-endian word order, decrement pointer, and explicit SAR state passing
// Returns struct {v16i8 qu, v8i16 qz, v8i16 qv, i8* ptr}
// Operands: qx (v8i16), qy (v8i16), qw (v8i16), rs1 (pointer), sel2 (i32 immediate), sar (i32)
// Note: Same as LD.INCP but decrements pointer instead of incrementing
def int_riscv_esp_fft_ams_s16_ld_r32_decp_m : Intrinsic<[llvm_v16i8_ty, llvm_v8i16_ty, llvm_v8i16_ty, llvm_ptr_ty],
                                                         [llvm_v8i16_ty, llvm_v8i16_ty, llvm_v8i16_ty, llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
                                                         [IntrReadMem, IntrArgMemOnly]>;

// FFT.AMS.S16.ST.INCP (_m version) - FFT AMS with store and increment and explicit SAR state passing
// Returns struct {v8i16 qz, i8* ptr}
// Inputs: qx (v8i16), qy (v8i16), qw (v8i16), qu (v16i8), rs1 (ptr), rs2 (ptr), sel2 (i32), sar (i64)
// Note: rs2 is updated with computation result
// SAR is passed explicitly as i64 for explicit state passing
def int_riscv_esp_fft_ams_s16_st_incp_m : Intrinsic<[llvm_v8i16_ty, llvm_ptr_ty],
                                                      [llvm_v8i16_ty, llvm_v8i16_ty, llvm_v8i16_ty, llvm_v16i8_ty, llvm_ptr_ty, llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
                                                      [IntrWriteMem, IntrArgMemOnly]>;

// FFT.BITREV - FFT bit reverse operation
// Returns: { ptr (updated pointer), v8i16 (result vector) }
// Parameters: [ ptr (source pointer), i32 (FFT_BIT_WIDTH, 32-bit unsigned) ]
def int_riscv_esp_fft_bitrev_m : Intrinsic<[llvm_ptr_ty, llvm_v8i16_ty], 
                                            [llvm_ptr_ty, llvm_i32_ty], 
                                            [IntrNoMem]>;

// FFT.CMUL.S16.LD.XP (_m version) - Complex multiply with load and register offset and explicit SAR state passing
// Returns struct {v8i16 qz, v16i8 qu, i8* ptr}
// Inputs: qx (v8i16), qy (v8i16), rs1 (ptr), rs2 (ptr), sel8 (i32), sar (i64)
// SAR is passed explicitly as i64 for explicit state passing
def int_riscv_esp_fft_cmul_s16_ld_xp_m : Intrinsic<[llvm_v8i16_ty, llvm_v16i8_ty, llvm_ptr_ty],
                                                     [llvm_v8i16_ty, llvm_v8i16_ty, llvm_ptr_ty, llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
                                                     [IntrReadMem, IntrArgMemOnly]>;

// FFT.CMUL.S16.ST.XP (_m version) - Complex multiply with store and register offset and explicit SAR state passing
// Returns struct {i8* ptr}
// Inputs: qx (v8i16), qy (v8i16), qu (v16i8), rs1 (ptr), rs2 (ptr), sel8 (i32), upd4 (i32), sel4 (i32), sar (i64)
// SAR is passed explicitly as i64 for explicit state passing
def int_riscv_esp_fft_cmul_s16_st_xp_m : Intrinsic<[llvm_ptr_ty],
                                                     [llvm_v8i16_ty, llvm_v8i16_ty, llvm_v16i8_ty, llvm_ptr_ty, llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                                                     [IntrWriteMem, IntrArgMemOnly]>;

// FFT.VST.R32.DECP - Returns struct {i8* ptr}
// Inputs: qu (v16i8), rs1 (ptr), sel2 (i32)
def int_riscv_esp_fft_vst_r32_decp_m : Intrinsic<[llvm_ptr_ty],
                                                   [llvm_v16i8_ty, llvm_ptr_ty, llvm_i32_ty],
                                                   [IntrWriteMem, IntrArgMemOnly]>;

// FFT.AMS.S16.LD.INCP.UAUP - FFT AMS S16 Load with Post-Increment and UA_STATE update
// Returns: { v16i8 (qu), v8i16 (qz), v8i16 (qv), ptr (updated pointer), v16i8 (ua_state_out) }
// Parameters: [ v8i16 (qx), v8i16 (qy), v8i16 (qw), ptr (rs1), i32 (sel2), v16i8 (ua_state_in), i32 (sar_bytes_in), i32 (sar_in) ]
def int_riscv_esp_fft_ams_s16_ld_incp_uaup_m : Intrinsic<[llvm_v16i8_ty, llvm_v8i16_ty, llvm_v8i16_ty, llvm_ptr_ty, llvm_v16i8_ty],
                                                         [llvm_v8i16_ty, llvm_v8i16_ty, llvm_v8i16_ty, llvm_ptr_ty, llvm_i32_ty, llvm_v16i8_ty, llvm_i32_ty, llvm_i32_ty],
                                                         [IntrReadMem, IntrArgMemOnly]>;

// VSLD - Vector Shift Left Dynamic
// Performs vector shift left/right based on shift amounts from qw segments
// VSLD.8 - 8-bit vector shift
def int_riscv_esp_vsld_8_m : Intrinsic<[llvm_v16i8_ty], 
                                       [llvm_v16i8_ty, llvm_v16i8_ty], 
                                       [IntrNoMem]>;

// VSLD.16 - 16-bit vector shift
def int_riscv_esp_vsld_16_m : Intrinsic<[llvm_v8i16_ty], 
                                        [llvm_v8i16_ty, llvm_v8i16_ty], 
                                        [IntrNoMem]>;

// VSRD - Vector Shift Right Dynamic
// Performs vector shift right based on shift amounts from qw segments
// VSRD.8 - 8-bit vector shift
def int_riscv_esp_vsrd_8_m : Intrinsic<[llvm_v16i8_ty], 
                                       [llvm_v16i8_ty, llvm_v16i8_ty], 
                                       [IntrNoMem]>;

// MOVX.W.SAR - Write SAR register
// Returns: i32 (SAR value, 32-bit unsigned, only low 6 bits used)
// Parameters: [ i32 value to write (rs1, 32-bit unsigned) ]
def int_riscv_esp_movx_w_sar_m : Intrinsic<[llvm_i32_ty], 
                                          [llvm_i32_ty], 
                                          [IntrNoMem]>;

// MOVX.W.SAR_BYTES - Write SAR_BYTES register
// Returns: i32 (SAR_BYTES value, 32-bit unsigned, only low 4 bits used)
// Parameters: [ i32 value to write (rs1, 32-bit unsigned) ]
def int_riscv_esp_movx_w_sar_bytes_m : Intrinsic<[llvm_i32_ty], 
                                                 [llvm_i32_ty], 
                                                 [IntrNoMem]>;

// VSR - Vector Shift Right (using SAR register)
// VSR.S32 - Signed arithmetic right shift
// SAR is passed explicitly as i32 for explicit state passing
def int_riscv_esp_vsr_s32_m : Intrinsic<[llvm_v4i32_ty], 
                                        [llvm_v4i32_ty, llvm_i32_ty], 
                                        [IntrNoMem]>;

// VSR.U32 - Unsigned logical right shift
// SAR is passed explicitly as i32 for explicit state passing
def int_riscv_esp_vsr_u32_m : Intrinsic<[llvm_v4i32_ty], 
                                        [llvm_v4i32_ty, llvm_i32_ty], 
                                        [IntrNoMem]>;

// VSL.32 - Vector Shift Left (using SAR register)
// Performs left shift on 32-bit segments using SAR register
// SAR is passed explicitly as i32 for explicit state passing
def int_riscv_esp_vsl_32_m : Intrinsic<[llvm_v4i32_ty], 
                                       [llvm_v4i32_ty, llvm_i32_ty], 
                                       [IntrNoMem]>;

// VABS - Vector Absolute Value
// VABS.8 - 8-bit vector absolute value
// Returns: v16i8 (absolute values of 16 8-bit data segments)
// Parameters: [ v16i8 vector ]
def int_riscv_esp_vabs_8_m : Intrinsic<[llvm_v16i8_ty], 
                                       [llvm_v16i8_ty], 
                                       [IntrNoMem]>;

// VADD - Vector Add
// VADD.S8 - Signed 8-bit vector addition
// Returns: v16i8 (16 8-bit signed addition results)
// Parameters: [ v16i8 vector qx, v16i8 vector qy ]
def int_riscv_esp_vadd_s8_m : Intrinsic<[llvm_v16i8_ty], 
                                        [llvm_v16i8_ty, llvm_v16i8_ty], 
                                        [IntrNoMem]>;

// VADD.U8 - Unsigned 8-bit vector addition
// Returns: v16i8 (16 8-bit unsigned addition results)
// Parameters: [ v16i8 vector qx, v16i8 vector qy ]
def int_riscv_esp_vadd_u8_m : Intrinsic<[llvm_v16i8_ty], 
                                        [llvm_v16i8_ty, llvm_v16i8_ty], 
                                        [IntrNoMem]>;

// VADD.S16 - Signed 16-bit vector addition
// Returns: v8i16 (8 16-bit signed addition results)
// Parameters: [ v8i16 vector qx, v8i16 vector qy ]
def int_riscv_esp_vadd_s16_m : Intrinsic<[llvm_v8i16_ty], 
                                         [llvm_v8i16_ty, llvm_v8i16_ty], 
                                         [IntrNoMem]>;

// VADD.U16 - Unsigned 16-bit vector addition
// Returns: v8i16 (8 16-bit unsigned addition results)
// Parameters: [ v8i16 vector qx, v8i16 vector qy ]
def int_riscv_esp_vadd_u16_m : Intrinsic<[llvm_v8i16_ty], 
                                         [llvm_v8i16_ty, llvm_v8i16_ty], 
                                         [IntrNoMem]>;

// VSUB - Vector Subtract
// VSUB.S8 - Signed 8-bit vector subtraction
// Returns: v16i8 (16 8-bit signed subtraction results)
// Parameters: [ v16i8 vector qx, v16i8 vector qy ]
def int_riscv_esp_vsub_s8_m : Intrinsic<[llvm_v16i8_ty], 
                                        [llvm_v16i8_ty, llvm_v16i8_ty], 
                                        [IntrNoMem]>;

// VSUB.U8 - Unsigned 8-bit vector subtraction
// Returns: v16i8 (16 8-bit unsigned subtraction results)
// Parameters: [ v16i8 vector qx, v16i8 vector qy ]
def int_riscv_esp_vsub_u8_m : Intrinsic<[llvm_v16i8_ty], 
                                        [llvm_v16i8_ty, llvm_v16i8_ty], 
                                        [IntrNoMem]>;

// VSUB.S16 - Signed 16-bit vector subtraction
// Returns: v8i16 (8 16-bit signed subtraction results)
// Parameters: [ v8i16 vector qx, v8i16 vector qy ]
def int_riscv_esp_vsub_s16_m : Intrinsic<[llvm_v8i16_ty], 
                                         [llvm_v8i16_ty, llvm_v8i16_ty], 
                                         [IntrNoMem]>;

// VABS.16 - Vector Absolute Value - 16-bit
// Returns: v8i16 (absolute values of 8 16-bit data segments)
// Parameters: [ v8i16 vector qy ]
def int_riscv_esp_vabs_16_m : ClangBuiltin<"__builtin_riscv_esp_vabs_16_m">,
	Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty], [IntrNoMem]>;

// VMAX.S16 - Signed 16-bit vector maximum
// Returns: v8i16 (maximum values of 8 16-bit signed elements)
// Parameters: [ v8i16 vector qx, v8i16 vector qy ]
def int_riscv_esp_vmax_s16_m : ClangBuiltin<"__builtin_riscv_esp_vmax_s16_m">,
	Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty, llvm_v8i16_ty], [IntrNoMem]>;

// VCMP - Vector Compare
// VCMP.EQ.S8 - Signed 8-bit vector equal comparison
// Returns: v16i8 (comparison results)
// Parameters: [ v16i8 vector qx, v16i8 vector qy ]
def int_riscv_esp_vcmp_eq_s8_m : Intrinsic<[llvm_v16i8_ty], 
                                            [llvm_v16i8_ty, llvm_v16i8_ty], 
                                            [IntrNoMem]>;

// VCMP.EQ.U8 - Unsigned 8-bit vector equal comparison
// Returns: v16i8 (comparison results)
// Parameters: [ v16i8 vector qx, v16i8 vector qy ]
def int_riscv_esp_vcmp_eq_u8_m : Intrinsic<[llvm_v16i8_ty], 
                                            [llvm_v16i8_ty, llvm_v16i8_ty], 
                                            [IntrNoMem]>;

// VCMP.GT.S8 - Signed 8-bit vector greater than comparison
// Returns: v16i8 (comparison results)
// Parameters: [ v16i8 vector qx, v16i8 vector qy ]
def int_riscv_esp_vcmp_gt_s8_m : Intrinsic<[llvm_v16i8_ty], 
                                            [llvm_v16i8_ty, llvm_v16i8_ty], 
                                            [IntrNoMem]>;

// VCMP.GT.U8 - Unsigned 8-bit vector greater than comparison
// Returns: v16i8 (comparison results)
// Parameters: [ v16i8 vector qx, v16i8 vector qy ]
def int_riscv_esp_vcmp_gt_u8_m : Intrinsic<[llvm_v16i8_ty], 
                                            [llvm_v16i8_ty, llvm_v16i8_ty], 
                                            [IntrNoMem]>;

// VCMP.LT.S8 - Signed 8-bit vector less than comparison
// Returns: v16i8 (comparison results)
// Parameters: [ v16i8 vector qx, v16i8 vector qy ]
def int_riscv_esp_vcmp_lt_s8_m : Intrinsic<[llvm_v16i8_ty], 
                                            [llvm_v16i8_ty, llvm_v16i8_ty], 
                                            [IntrNoMem]>;

// VCMP.LT.U8 - Unsigned 8-bit vector less than comparison
// Returns: v16i8 (comparison results)
// Parameters: [ v16i8 vector qx, v16i8 vector qy ]
def int_riscv_esp_vcmp_lt_u8_m : Intrinsic<[llvm_v16i8_ty], 
                                            [llvm_v16i8_ty, llvm_v16i8_ty], 
                                            [IntrNoMem]>;

// VMAX - Vector Maximum
// VMAX.S8 - Signed 8-bit vector maximum
// Returns: v16i8 (maximum values)
// Parameters: [ v16i8 vector qx, v16i8 vector qy ]
def int_riscv_esp_vmax_s8_m : Intrinsic<[llvm_v16i8_ty], 
                                         [llvm_v16i8_ty, llvm_v16i8_ty], 
                                         [IntrNoMem]>;

// VMAX.U8 - Unsigned 8-bit vector maximum
// Returns: v16i8 (maximum values)
// Parameters: [ v16i8 vector qx, v16i8 vector qy ]
def int_riscv_esp_vmax_u8_m : Intrinsic<[llvm_v16i8_ty], 
                                         [llvm_v16i8_ty, llvm_v16i8_ty], 
                                         [IntrNoMem]>;

// VMIN - Vector Minimum
// VMIN.S8 - Signed 8-bit vector minimum
// Returns: v16i8 (minimum values)
// Parameters: [ v16i8 vector qx, v16i8 vector qy ]
def int_riscv_esp_vmin_s8_m : Intrinsic<[llvm_v16i8_ty], 
                                         [llvm_v16i8_ty, llvm_v16i8_ty], 
                                         [IntrNoMem]>;

// VMIN.U8 - Unsigned 8-bit vector minimum
// Returns: v16i8 (minimum values)
// Parameters: [ v16i8 vector qx, v16i8 vector qy ]
def int_riscv_esp_vmin_u8_m : Intrinsic<[llvm_v16i8_ty], 
                                         [llvm_v16i8_ty, llvm_v16i8_ty], 
                                         [IntrNoMem]>;

// VMUL - Vector Multiplication
// VMUL.S8 - Signed 8-bit vector multiplication
// Returns: v16i8 (multiplication results)
// Parameters: [ v16i8 vector qx, v16i8 vector qy, i32 sar (SAR register value) ]
def int_riscv_esp_vmul_s8_m : Intrinsic<[llvm_v16i8_ty], 
                                         [llvm_v16i8_ty, llvm_v16i8_ty, llvm_i32_ty], 
                                         [IntrNoMem]>;

// VMUL.U8 - Unsigned 8-bit vector multiplication
// Returns: v16i8 (multiplication results)
// Parameters: [ v16i8 vector qx, v16i8 vector qy, i32 sar (SAR register value) ]
def int_riscv_esp_vmul_u8_m : Intrinsic<[llvm_v16i8_ty], 
                                         [llvm_v16i8_ty, llvm_v16i8_ty, llvm_i32_ty], 
                                         [IntrNoMem]>;
// VMUL.S16 - Signed 16-bit vector multiplication
// Returns: v8i16 (multiplication results)
// Parameters: [ v8i16 vector qx, v8i16 vector qy, i32 sar (SAR register value) ]
// SAR is passed explicitly as i32 for explicit state passing
// SAR stores unsigned shift amount (0-63), only low 6 bits are used
def int_riscv_esp_vmul_s16_m : Intrinsic<[llvm_v8i16_ty], 
                                         [llvm_v8i16_ty, llvm_v8i16_ty, llvm_i32_ty], 
                                         [IntrNoMem]>;

// Bitwise Operations
// ANDQ - Vector bitwise AND
// Returns: v16i8 (bitwise AND results)
// Parameters: [ v16i8 vector qx, v16i8 vector qy ]
def int_riscv_esp_andq_m : Intrinsic<[llvm_v16i8_ty], 
                                      [llvm_v16i8_ty, llvm_v16i8_ty], 
                                      [IntrNoMem]>;

// ORQ - Vector bitwise OR
// Returns: v16i8 (bitwise OR results)
// Parameters: [ v16i8 vector qx, v16i8 vector qy ]
def int_riscv_esp_orq_m : Intrinsic<[llvm_v16i8_ty], 
                                     [llvm_v16i8_ty, llvm_v16i8_ty], 
                                     [IntrNoMem]>;

// XORQ - Vector bitwise XOR
// Returns: v16i8 (bitwise XOR results)
// Parameters: [ v16i8 vector qx, v16i8 vector qy ]
def int_riscv_esp_xorq_m : Intrinsic<[llvm_v16i8_ty], 
                                      [llvm_v16i8_ty, llvm_v16i8_ty], 
                                      [IntrNoMem]>;

// NOTQ - Vector bitwise NOT
// Returns: v16i8 (bitwise NOT results)
// Parameters: [ v16i8 vector qx ]
def int_riscv_esp_notq_m : Intrinsic<[llvm_v16i8_ty], 
                                      [llvm_v16i8_ty], 
                                      [IntrNoMem]>;

// ZERO.Q - Zero vector register
// Returns: v16i8 (zero vector)
// Parameters: []
def int_riscv_esp_zero_q_m : Intrinsic<[llvm_v16i8_ty], 
                                        [], 
                                        [IntrNoMem]>;

// ZERO.QACC - Zero QACC accumulator
// Returns: {v16i8, v16i8, v16i8, v16i8} (4x128-bit QACC, all zero)
// Parameters: []
def int_riscv_esp_zero_qacc_m : Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty], 
                                           [], 
                                           [IntrNoMem]>;

// MOV.S16.QACC - Sign extend 8x16-bit to 64-bit, store to QACC_H and QACC_L
// Returns: {v16i8, v16i8, v16i8, v16i8} (4x128-bit QACC)
// Parameters: [v8i16] (input vector)
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_mov_s16_qacc_m : Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],
                                              [llvm_v8i16_ty],
                                              [IntrNoMem]>;

// MOV.S8.QACC - Sign extend 16x8-bit to 32-bit, store to QACC_H and QACC_L
// Returns: {v16i8, v16i8, v16i8, v16i8} (4x128-bit QACC)
// Parameters: [v16i8] (input vector)
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_mov_s8_qacc_m : Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],
                                             [llvm_v16i8_ty],
                                             [IntrNoMem]>;

// MOV.U16.QACC - Zero extend 8x16-bit to 64-bit, store to QACC_H and QACC_L
// Returns: {v16i8, v16i8, v16i8, v16i8} (4x128-bit QACC)
// Parameters: [v8i16] (input vector)
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_mov_u16_qacc_m : Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],
                                              [llvm_v8i16_ty],
                                              [IntrNoMem]>;

// MOV.U8.QACC - Zero extend 16x8-bit to 32-bit, store to QACC_H and QACC_L
// Returns: {v16i8, v16i8, v16i8, v16i8} (4x128-bit QACC)
// Parameters: [v16i8] (input vector)
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_mov_u8_qacc_m : Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],
                                             [llvm_v16i8_ty],
                                             [IntrNoMem]>;

// ZERO.XACC - Zero XACC accumulator
// Returns: {i32, i32} (XACC set to zero {i32 low=0, i32 high=0})
// Parameters: []
def int_riscv_esp_zero_xacc_m : Intrinsic<[llvm_i32_ty, llvm_i32_ty], 
                                          [], 
                                          [IntrNoMem]>;

// ESP.ST.S.XACC.IP (_m version) - Mixed model: XACC as {i32 low, i32 high}
// Returns: updated pointer + XACC unchanged {i32 low, i32 high}
// Parameters: [xacc_low_in (i32), xacc_high_in (i32, only low 8 bits valid), ptr, offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
// Note: XACC value is unchanged after store, so output XACC equals input XACC
def int_riscv_esp_st_s_xacc_ip_m : Intrinsic<
    [llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],  // Returns: pointer + XACC unchanged {i32 low, i32 high}
    [llvm_i32_ty,                              // xacc_low_in (i32)
     llvm_i32_ty,                              // xacc_high_in (i32, only low 8 bits valid)
     llvm_ptr_ty,                              // pointer
     llvm_i32_ty],                             // offset
    [IntrWriteMem, IntrArgMemOnly]>;           // No IntrHasSideEffects

// ESP.VCMULAS.S16.QACC.L (_m version)
// Returns: {v16i8, v16i8} - 2x128-bit QACC_L directly
// Parameters: [v16i8, v16i8 (2x128-bit QACC_L passthru), qx, qy]
// Migration: Explicit state passing (no IntrHasSideEffects)
// QACC_L is split into 2x128-bit: QACC_L_LOW (v0) and QACC_L_HIGH (v1)
// Note: No ClangBuiltin - builtin signature (void with 6 params) doesn't match intrinsic signature ({v16i8, v16i8} with 4 params)
def int_riscv_esp_vcmulas_s16_qacc_l_m :
	Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty],     // Returns: 2x128-bit QACC_L (qacc_l_l, qacc_l_h)
              [llvm_v16i8_ty,                     // QACC_L[127:0] passthru (v16i8)
               llvm_v16i8_ty,                     // QACC_L[255:128] passthru (v16i8)
               llvm_v8i16_ty,                     // qx
               llvm_v8i16_ty],                    // qy
              [IntrNoMem]>;                       // No IntrHasSideEffects

// ESP.VCMULAS.S16.QACC.H (_m version)
// Returns: {v16i8, v16i8} - 2x128-bit QACC_H directly
// Parameters: [v16i8, v16i8 (2x128-bit QACC_H passthru), qx, qy]
// Migration: Explicit state passing (no IntrHasSideEffects)
// QACC_H is split into 2x128-bit: QACC_H_LOW (v2) and QACC_H_HIGH (v3)
// Note: No ClangBuiltin - builtin signature (void with 6 params) doesn't match intrinsic signature ({v16i8, v16i8} with 4 params)
def int_riscv_esp_vcmulas_s16_qacc_h_m :
	Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty],     // Returns: 2x128-bit QACC_H (qacc_h_l, qacc_h_h)
              [llvm_v16i8_ty,                     // QACC_H[127:0] passthru (v16i8)
               llvm_v16i8_ty,                     // QACC_H[255:128] passthru (v16i8)
               llvm_v8i16_ty,                     // qx
               llvm_v8i16_ty],                    // qy
              [IntrNoMem]>;                       // No IntrHasSideEffects

// ESP.VCMULAS.S16.QACC.H.LD.IP (_m version)
// Returns: updated pointer + qu vector + new QACC_H (2x128-bit)
// Parameters: [current QACC_H (2x128-bit passthru), qx, qy, pointer, immediate offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_vcmulas_s16_qacc_h_ld_ip_m : Intrinsic<
    [llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],  // Returns: ptr + qu + 2x128-bit QACC_H (4 values)
    [llvm_v16i8_ty, llvm_v16i8_ty,                               // Current QACC_H (2x128-bit passthru)
     llvm_v8i16_ty,                                               // qx
     llvm_v8i16_ty,                                               // qy
     llvm_ptr_ty,                                                 // pointer
     llvm_i32_ty],                                                // offset
    [IntrReadMem, IntrArgMemOnly]>;                               // No IntrHasSideEffects

// ESP.VCMULAS.S16.QACC.H.LD.XP (_m version)
// Returns: updated pointer + qu vector + new QACC_H (2x128-bit)
// Parameters: [current QACC_H (2x128-bit passthru), qx, qy, pointer, register offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_vcmulas_s16_qacc_h_ld_xp_m : Intrinsic<
    [llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],  // Returns: ptr + qu + 2x128-bit QACC_H (4 values)
    [llvm_v16i8_ty, llvm_v16i8_ty,                               // Current QACC_H (2x128-bit passthru)
     llvm_v8i16_ty,                                               // qx
     llvm_v8i16_ty,                                               // qy
     llvm_ptr_ty,                                                 // pointer
     llvm_i32_ty],                                                // register offset
    [IntrReadMem, IntrArgMemOnly]>;                               // No IntrHasSideEffects

// ESP.VCMULAS.S16.QACC.L.LD.IP (_m version)
// Returns: updated pointer + qu vector + new QACC_L (2x128-bit)
// Parameters: [current QACC_L (2x128-bit passthru), qx, qy, pointer, immediate offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_vcmulas_s16_qacc_l_ld_ip_m : Intrinsic<
    [llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],  // Returns: ptr + qu + 2x128-bit QACC_L (4 values)
    [llvm_v16i8_ty, llvm_v16i8_ty,                               // Current QACC_L (2x128-bit passthru)
     llvm_v8i16_ty,                                               // qx
     llvm_v8i16_ty,                                               // qy
     llvm_ptr_ty,                                                 // pointer
     llvm_i32_ty],                                                // offset
    [IntrReadMem, IntrArgMemOnly]>;                               // No IntrHasSideEffects

// ESP.VCMULAS.S16.QACC.L.LD.XP (_m version)
// Returns: updated pointer + qu vector + new QACC_L (2x128-bit)
// Parameters: [current QACC_L (2x128-bit passthru), qx, qy, pointer, register offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_vcmulas_s16_qacc_l_ld_xp_m : Intrinsic<
    [llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],  // Returns: ptr + qu + 2x128-bit QACC_L (4 values)
    [llvm_v16i8_ty, llvm_v16i8_ty,                               // Current QACC_L (2x128-bit passthru)
     llvm_v8i16_ty,                                               // qx
     llvm_v8i16_ty,                                               // qy
     llvm_ptr_ty,                                                 // pointer
     llvm_i32_ty],                                                // register offset
    [IntrReadMem, IntrArgMemOnly]>;                               // No IntrHasSideEffects

// ESP.VCMULAS.S8.QACC.H (_m version)
// Returns: {v16i8, v16i8} - 2x128-bit QACC_H directly
// Parameters: [v16i8, v16i8 (2x128-bit QACC_H passthru), qx, qy]
// Migration: Explicit state passing (no IntrHasSideEffects)
// QACC_H is split into 2x128-bit: QACC_H_LOW (v2) and QACC_H_HIGH (v3)
// Note: No ClangBuiltin - builtin signature (void with 6 params) doesn't match intrinsic signature ({v16i8, v16i8} with 4 params)
def int_riscv_esp_vcmulas_s8_qacc_h_m :
	Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty],     // Returns: 2x128-bit QACC_H (qacc_h_l, qacc_h_h)
              [llvm_v16i8_ty,                     // QACC_H[127:0] passthru (v16i8)
               llvm_v16i8_ty,                     // QACC_H[255:128] passthru (v16i8)
               llvm_v16i8_ty,                     // qx
               llvm_v16i8_ty],                    // qy
              [IntrNoMem]>;                       // No IntrHasSideEffects

// ESP.VCMULAS.S8.QACC.H.LD.IP (_m version)
// Returns: updated pointer + qu vector + new QACC_H (2x128-bit)
// Parameters: [current QACC_H (2x128-bit passthru), qx, qy, pointer, immediate offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_vcmulas_s8_qacc_h_ld_ip_m : Intrinsic<
    [llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],  // Returns: ptr + qu + 2x128-bit QACC_H (4 values)
    [llvm_v16i8_ty, llvm_v16i8_ty,                               // Current QACC_H (2x128-bit passthru)
     llvm_v16i8_ty,                                               // qx
     llvm_v16i8_ty,                                               // qy
     llvm_ptr_ty,                                                 // pointer
     llvm_i32_ty],                                                // offset
    [IntrReadMem, IntrArgMemOnly]>;                               // No IntrHasSideEffects

// ESP.VCMULAS.S8.QACC.H.LD.XP (_m version)
// Returns: updated pointer + qu vector + new QACC_H (2x128-bit)
// Parameters: [current QACC_H (2x128-bit passthru), qx, qy, pointer, register offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_vcmulas_s8_qacc_h_ld_xp_m : Intrinsic<
    [llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],  // Returns: ptr + qu + 2x128-bit QACC_H (4 values)
    [llvm_v16i8_ty, llvm_v16i8_ty,                               // Current QACC_H (2x128-bit passthru)
     llvm_v16i8_ty,                                               // qx
     llvm_v16i8_ty,                                               // qy
     llvm_ptr_ty,                                                 // pointer
     llvm_i32_ty],                                                // register offset
    [IntrReadMem, IntrArgMemOnly]>;                               // No IntrHasSideEffects

// ESP.VCMULAS.S8.QACC.L (_m version)
// Returns: {v16i8, v16i8} - 2x128-bit QACC_L directly
// Parameters: [v16i8, v16i8 (2x128-bit QACC_L passthru), qx, qy]
// Migration: Explicit state passing (no IntrHasSideEffects)
// QACC_L is split into 2x128-bit: QACC_L_LOW (v0) and QACC_L_HIGH (v1)
// Note: No ClangBuiltin - builtin signature (void with 6 params) doesn't match intrinsic signature ({v16i8, v16i8} with 4 params)
def int_riscv_esp_vcmulas_s8_qacc_l_m :
	Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty],     // Returns: 2x128-bit QACC_L (qacc_l_l, qacc_l_h)
              [llvm_v16i8_ty,                     // QACC_L[127:0] passthru (v16i8)
               llvm_v16i8_ty,                     // QACC_L[255:128] passthru (v16i8)
               llvm_v16i8_ty,                     // qx
               llvm_v16i8_ty],                    // qy
              [IntrNoMem]>;                       // No IntrHasSideEffects

// ESP.VCMULAS.S8.QACC.L.LD.IP (_m version)
// Returns: updated pointer + qu vector + new QACC_L (2x128-bit)
// Parameters: [current QACC_L (2x128-bit passthru), qx, qy, pointer, immediate offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_vcmulas_s8_qacc_l_ld_ip_m : Intrinsic<
    [llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],  // Returns: ptr + qu + 2x128-bit QACC_L (4 values)
    [llvm_v16i8_ty, llvm_v16i8_ty,                               // Current QACC_L (2x128-bit passthru)
     llvm_v16i8_ty,                                               // qx
     llvm_v16i8_ty,                                               // qy
     llvm_ptr_ty,                                                 // pointer
     llvm_i32_ty],                                                // offset
    [IntrReadMem, IntrArgMemOnly]>;                               // No IntrHasSideEffects

// ESP.VCMULAS.S8.QACC.L.LD.XP (_m version)
// Returns: updated pointer + qu vector + new QACC_L (2x128-bit)
// Parameters: [current QACC_L (2x128-bit passthru), qx, qy, pointer, register offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_vcmulas_s8_qacc_l_ld_xp_m : Intrinsic<
    [llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],  // Returns: ptr + qu + 2x128-bit QACC_L (4 values)
    [llvm_v16i8_ty, llvm_v16i8_ty,                               // Current QACC_L (2x128-bit passthru)
     llvm_v16i8_ty,                                               // qx
     llvm_v16i8_ty,                                               // qy
     llvm_ptr_ty,                                                 // pointer
     llvm_i32_ty],                                                // register offset
    [IntrReadMem, IntrArgMemOnly]>;                               // No IntrHasSideEffects

// ESP.LD.QACC.H.H.128.IP (_m version)
// Load 128-bit data to QACC_H[255:128]
// Subregister model: returns loaded 128-bit data (v16i8) + updated pointer
// Parameters: [pointer, immediate offset]
// Note: Hardware loads 128-bit data to QACC_H[255:128], QACC_H[127:0] remains unchanged
def int_riscv_esp_ld_qacc_h_h_128_ip_m : Intrinsic<
    [llvm_v16i8_ty, llvm_ptr_ty],        // Returns: loaded 128-bit data + pointer
    [llvm_ptr_ty,                         // pointer
     llvm_i32_ty],                        // offset
    [IntrReadMem, IntrArgMemOnly]>;       // No IntrHasSideEffects

// ESP.LD.QACC.H.L.128.IP (_m version)
// Load 128-bit data to QACC_H[127:0]
// Subregister model: returns loaded 128-bit data (v16i8) + updated pointer
// Parameters: [pointer, immediate offset]
// Note: Hardware loads 128-bit data to QACC_H[127:0], QACC_H[255:128] remains unchanged
def int_riscv_esp_ld_qacc_h_l_128_ip_m : Intrinsic<
    [llvm_v16i8_ty, llvm_ptr_ty],        // Returns: loaded 128-bit data + pointer
    [llvm_ptr_ty,                         // pointer
     llvm_i32_ty],                        // offset
    [IntrReadMem, IntrArgMemOnly]>;       // No IntrHasSideEffects

// ESP.LD.QACC.L.H.128.IP (_m version)
// Load 128-bit data to QACC_L[255:128]
// Subregister model: returns loaded 128-bit data (v16i8) + updated pointer
// Parameters: [pointer, immediate offset]
// Note: Hardware loads 128-bit data to QACC_L[255:128], QACC_L[127:0] remains unchanged
def int_riscv_esp_ld_qacc_l_h_128_ip_m : Intrinsic<
    [llvm_v16i8_ty, llvm_ptr_ty],        // Returns: loaded 128-bit data + pointer
    [llvm_ptr_ty,                         // pointer
     llvm_i32_ty],                        // offset
    [IntrReadMem, IntrArgMemOnly]>;       // No IntrHasSideEffects

// ESP.LD.QACC.L.L.128.IP (_m version)
// Load 128-bit data to QACC_L[127:0]
// Subregister model: returns loaded 128-bit data (v16i8) + updated pointer
// Parameters: [pointer, immediate offset]
// Note: Hardware loads 128-bit data to QACC_L[127:0], QACC_L[255:128] remains unchanged
def int_riscv_esp_ld_qacc_l_l_128_ip_m : Intrinsic<
    [llvm_v16i8_ty, llvm_ptr_ty],        // Returns: loaded 128-bit data + pointer
    [llvm_ptr_ty,                         // pointer
     llvm_i32_ty],                        // offset
    [IntrReadMem, IntrArgMemOnly]>;       // No IntrHasSideEffects

// ESP.ST.QACC.H.H.128.IP (_m version)
// Store QACC_H[255:128] to memory
// Returns: updated pointer
// Parameters: [QACC_H[255:128] (128-bit), pointer, immediate offset]
// Accept v16i8 (128-bit) to match hardware operation and simplify usage.
// First principle: directly accept 128-bit value, matching hardware behavior.
def int_riscv_esp_st_qacc_h_h_128_ip_m : Intrinsic<
    [llvm_ptr_ty],                        // Returns: updated pointer
    [llvm_v16i8_ty,                       // QACC_H[255:128] (128-bit) - Direct match hardware operation
     llvm_ptr_ty,                         // pointer
     llvm_i32_ty],                        // offset
    [IntrWriteMem, IntrArgMemOnly]>;      //  No IntrHasSideEffects

// ESP.ST.QACC.H.L.128.IP (_m version)
// Store QACC_H[127:0] to memory
// Returns: updated pointer
// Parameters: [QACC_H[127:0] (128-bit), pointer, immediate offset]
// Accept v16i8 (128-bit) to match hardware operation and simplify usage.
// First principle: directly accept 128-bit value, matching hardware behavior.
def int_riscv_esp_st_qacc_h_l_128_ip_m : Intrinsic<
    [llvm_ptr_ty],                        // Returns: updated pointer
    [llvm_v16i8_ty,                       // QACC_H[127:0] (128-bit) - Direct match hardware operation
     llvm_ptr_ty,                         // pointer
     llvm_i32_ty],                        // offset
    [IntrWriteMem, IntrArgMemOnly]>;      //  No IntrHasSideEffects

// ESP.ST.QACC.L.H.128.IP (_m version)
// Store QACC_L[255:128] to memory
// Returns: updated pointer
// Parameters: [QACC_L[255:128] (128-bit), pointer, immediate offset]
// Hardware directly reads QACC_L[255:128] from full QACC_L register.
// Accept v16i8 (128-bit) to match hardware operation and simplify usage.
// First principle: directly accept 128-bit value, matching hardware behavior.
def int_riscv_esp_st_qacc_l_h_128_ip_m : Intrinsic<
    [llvm_ptr_ty],                        // Returns: updated pointer
    [llvm_v16i8_ty,                       // QACC_L_HIGH (128-bit) - matches hardware operation
     llvm_ptr_ty,                         // pointer
     llvm_i32_ty],                        // offset
    [IntrWriteMem, IntrArgMemOnly]>;      //  No IntrHasSideEffects

// ESP.ST.QACC.L.L.128.IP (_m version)
// Store QACC_L[127:0] to memory
// Returns: updated pointer
// Parameters: [QACC_L[127:0] (128-bit), pointer, immediate offset]
// Hardware directly reads QACC_L[127:0] from full QACC_L register.
// Accept v16i8 (128-bit) to match hardware operation and simplify usage.
// First principle: directly accept 128-bit value, matching hardware behavior.
def int_riscv_esp_st_qacc_l_l_128_ip_m : Intrinsic<
    [llvm_ptr_ty],                        // Returns: updated pointer
    [llvm_v16i8_ty,                       // QACC_L_LOW (128-bit) - matches hardware operation
     llvm_ptr_ty,                         // pointer
     llvm_i32_ty],                        // offset
    [IntrWriteMem, IntrArgMemOnly]>;      //  No IntrHasSideEffects

// ESP.LDQA.S16.128.IP (_m version)
// Load 128-bit data, sign-extend 8x16-bit to 64-bit, store to QACC_H and QACC_L
// Returns: updated pointer + 4x128-bit QACC (QACC_L_LOW, QACC_L_HIGH, QACC_H_LOW, QACC_H_HIGH)
// Parameters: [current QACC (passthru, v64i8), pointer, immediate offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_ldqa_s16_128_ip_m : Intrinsic<
    [llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],  // Returns: pointer + 4x128-bit QACC
    [llvm_v64i8_ty,                       // Current QACC (passthru)
     llvm_ptr_ty,                         // pointer
     llvm_i32_ty],                        // offset
    [IntrReadMem, IntrArgMemOnly]>;       // No IntrHasSideEffects

// ESP.LDQA.S16.128.XP (_m version)
// Load 128-bit data, sign-extend 8x16-bit to 64-bit, store to QACC_H and QACC_L
// Returns: updated pointer + new QACC (512-bit: {QACC_H, QACC_L})
// Parameters: [current QACC (passthru), pointer, register offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_ldqa_s16_128_xp_m : Intrinsic<
    [llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],  // Returns: pointer + 4x128-bit QACC
    [llvm_v64i8_ty,                       // Current QACC (passthru)
     llvm_ptr_ty,                         // pointer
     llvm_i32_ty],                        // register offset
    [IntrReadMem, IntrArgMemOnly]>;       // No IntrHasSideEffects

// ESP.LDQA.S8.128.IP (_m version)
// Load 128-bit data, sign-extend 16x8-bit to 32-bit, store to QACC_H and QACC_L
// Returns: updated pointer + new QACC (512-bit: {QACC_H, QACC_L})
// Parameters: [current QACC (passthru), pointer, immediate offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_ldqa_s8_128_ip_m : Intrinsic<
    [llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],  // Returns: pointer + 4x128-bit QACC
    [llvm_v64i8_ty,                       // Current QACC (passthru)
     llvm_ptr_ty,                         // pointer
     llvm_i32_ty],                        // offset
    [IntrReadMem, IntrArgMemOnly]>;       // No IntrHasSideEffects

// ESP.LDQA.S8.128.XP (_m version)
// Load 128-bit data, sign-extend 16x8-bit to 32-bit, store to QACC_H and QACC_L
// Returns: updated pointer + new QACC (512-bit: {QACC_H, QACC_L})
// Parameters: [current QACC (passthru), pointer, register offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_ldqa_s8_128_xp_m : Intrinsic<
    [llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],  // Returns: pointer + 4x128-bit QACC
    [llvm_v64i8_ty,                       // Current QACC (passthru)
     llvm_ptr_ty,                         // pointer
     llvm_i32_ty],                        // register offset
    [IntrReadMem, IntrArgMemOnly]>;       // No IntrHasSideEffects

// ESP.LDQA.U16.128.IP (_m version)
// Load 128-bit data, zero-extend 8x16-bit to 64-bit, store to QACC_H and QACC_L
// Returns: updated pointer + new QACC (512-bit: {QACC_H, QACC_L})
// Parameters: [current QACC (passthru), pointer, immediate offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_ldqa_u16_128_ip_m : Intrinsic<
    [llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],  // Returns: pointer + 4x128-bit QACC
    [llvm_v64i8_ty,                       // Current QACC (passthru)
     llvm_ptr_ty,                         // pointer
     llvm_i32_ty],                        // offset
    [IntrReadMem, IntrArgMemOnly]>;       // No IntrHasSideEffects

// ESP.LDQA.U16.128.XP (_m version)
// Load 128-bit data, zero-extend 8x16-bit to 64-bit, store to QACC_H and QACC_L
// Returns: updated pointer + new QACC (512-bit: {QACC_H, QACC_L})
// Parameters: [current QACC (passthru), pointer, register offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_ldqa_u16_128_xp_m : Intrinsic<
    [llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],  // Returns: pointer + 4x128-bit QACC
    [llvm_v64i8_ty,                       // Current QACC (passthru)
     llvm_ptr_ty,                         // pointer
     llvm_i32_ty],                        // register offset
    [IntrReadMem, IntrArgMemOnly]>;       // No IntrHasSideEffects

// ESP.LDQA.U8.128.IP (_m version)
// Load 128-bit data, zero-extend 16x8-bit to 32-bit, store to QACC_H and QACC_L
// Returns: updated pointer + new QACC (512-bit: {QACC_H, QACC_L})
// Parameters: [current QACC (passthru), pointer, immediate offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_ldqa_u8_128_ip_m : Intrinsic<
    [llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],  // Returns: pointer + 4x128-bit QACC
    [llvm_v64i8_ty,                       // Current QACC (passthru)
     llvm_ptr_ty,                         // pointer
     llvm_i32_ty],                        // offset
    [IntrReadMem, IntrArgMemOnly]>;       // No IntrHasSideEffects

// ESP.LDQA.U8.128.XP (_m version)
// Load 128-bit data, zero-extend 16x8-bit to 32-bit, store to QACC_H and QACC_L
// Returns: updated pointer + new QACC (512-bit: {QACC_H, QACC_L})
// Parameters: [current QACC (passthru), pointer, register offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_ldqa_u8_128_xp_m : Intrinsic<
    [llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],  // Returns: pointer + 4x128-bit QACC
    [llvm_v64i8_ty,                       // Current QACC (passthru)
     llvm_ptr_ty,                         // pointer
     llvm_i32_ty],                        // register offset
    [IntrReadMem, IntrArgMemOnly]>;       // No IntrHasSideEffects

// CMUL - Complex Multiplication
// CMUL.S16 - Signed 16-bit complex multiplication
// Returns: v8i16 (complex multiplication result)
// Parameters: [ v8i16 qz_in (passthru), v8i16 qx, v8i16 qy, i32 sel4 (immediate), i32 sar (SAR register value) ]
def int_riscv_esp_cmul_s16_m : Intrinsic<[llvm_v8i16_ty], 
                                          [llvm_v8i16_ty, llvm_v8i16_ty, llvm_v8i16_ty, llvm_i32_ty, llvm_i32_ty], 
                                          [IntrNoMem, ImmArg<ArgIndex<3>>]>;

// CMUL.S16.LD.INCP - Signed 16-bit complex multiplication with load and pointer increment
// Returns: {v8i16 qz, v16i8 qu, ptr} (complex multiplication result, loaded vector, updated pointer)
// Parameters: [ v8i16 qz_in (passthru), v8i16 qx, v8i16 qy, ptr, i32 offset, i32 sar (SAR register value) ]
def int_riscv_esp_cmul_s16_ld_incp_m : Intrinsic<[llvm_v8i16_ty, llvm_v16i8_ty, llvm_ptr_ty], 
                                                   [llvm_v8i16_ty, llvm_v8i16_ty, llvm_v8i16_ty, llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty], 
                                                   [IntrReadMem, IntrArgMemOnly]>;

// CMUL.S16.ST.INCP - Signed 16-bit complex multiplication with store and pointer increment
// Returns: {v8i16 qz, ptr} (complex multiplication result, updated pointer)
// Parameters: [ v8i16 qz_in (passthru), v8i16 qx, v8i16 qy, v16i8 qu, ptr, i32 offset, i32 sar (SAR register value) ]
def int_riscv_esp_cmul_s16_st_incp_m : Intrinsic<[llvm_v8i16_ty, llvm_ptr_ty], 
                                                   [llvm_v8i16_ty, llvm_v8i16_ty, llvm_v8i16_ty, llvm_v16i8_ty, llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty], 
                                                   [IntrWriteMem, IntrArgMemOnly]>;

// CMUL.S8 - Signed 8-bit complex multiplication
// Returns: v16i8 (complex multiplication result)
// Parameters: [ v16i8 qz_in (passthru), v16i8 qx, v16i8 qy, i32 sel4 (immediate), i32 sar (SAR register value) ]
def int_riscv_esp_cmul_s8_m : Intrinsic<[llvm_v16i8_ty], 
                                         [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_i32_ty, llvm_i32_ty], 
                                         [IntrNoMem, ImmArg<ArgIndex<3>>]>;

// CMUL.S8.LD.INCP - Signed 8-bit complex multiplication with load and pointer increment
// Returns: {v16i8 qz, v16i8 qu, ptr} (complex multiplication result, loaded vector, updated pointer)
// Parameters: [ v16i8 qz_in (passthru), v16i8 qx, v16i8 qy, ptr, i32 offset, i32 sar (SAR register value) ]
def int_riscv_esp_cmul_s8_ld_incp_m : Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty, llvm_ptr_ty], 
                                                 [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty], 
                                                 [IntrReadMem, IntrArgMemOnly]>;

// CMUL.S8.ST.INCP - Signed 8-bit complex multiplication with store and pointer increment
// Returns: {v16i8 qz, ptr} (complex multiplication result, updated pointer)
// Parameters: [ v16i8 qz_in (passthru), v16i8 qx, v16i8 qy, v16i8 qu, ptr, i32 offset, i32 sar (SAR register value) ]
def int_riscv_esp_cmul_s8_st_incp_m : Intrinsic<[llvm_v16i8_ty, llvm_ptr_ty], 
                                                 [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty], 
                                                 [IntrWriteMem, IntrArgMemOnly]>;

// CMUL.U16 - Unsigned 16-bit complex multiplication
// Returns: v8i16 (complex multiplication result)
// Parameters: [ v8i16 qz_in (passthru), v8i16 qx, v8i16 qy, i32 sel4 (immediate), i32 sar (SAR register value) ]
def int_riscv_esp_cmul_u16_m : Intrinsic<[llvm_v8i16_ty], 
                                          [llvm_v8i16_ty, llvm_v8i16_ty, llvm_v8i16_ty, llvm_i32_ty, llvm_i32_ty], 
                                          [IntrNoMem, ImmArg<ArgIndex<3>>]>;

// CMUL.U16.LD.INCP - Unsigned 16-bit complex multiplication with load and pointer increment
// Returns: {v8i16 qz, v16i8 qu, ptr} (complex multiplication result, loaded vector, updated pointer)
// Parameters: [ v8i16 qz_in (passthru), v8i16 qx, v8i16 qy, ptr, i32 offset, i32 sar (SAR register value) ]
def int_riscv_esp_cmul_u16_ld_incp_m : Intrinsic<[llvm_v8i16_ty, llvm_v16i8_ty, llvm_ptr_ty], 
                                                   [llvm_v8i16_ty, llvm_v8i16_ty, llvm_v8i16_ty, llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty], 
                                                   [IntrReadMem, IntrArgMemOnly]>;

// CMUL.U16.ST.INCP - Unsigned 16-bit complex multiplication with store and pointer increment
// Returns: {v8i16 qz, ptr} (complex multiplication result, updated pointer)
// Parameters: [ v8i16 qz_in (passthru), v8i16 qx, v8i16 qy, v16i8 qu, ptr, i32 offset, i32 sar (SAR register value) ]
def int_riscv_esp_cmul_u16_st_incp_m : Intrinsic<[llvm_v8i16_ty, llvm_ptr_ty], 
                                                   [llvm_v8i16_ty, llvm_v8i16_ty, llvm_v8i16_ty, llvm_v16i8_ty, llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty], 
                                                   [IntrWriteMem, IntrArgMemOnly]>;

// CMUL.U8 - Unsigned 8-bit complex multiplication
// Returns: v16i8 (complex multiplication result)
// Parameters: [ v16i8 qz_in (passthru), v16i8 qx, v16i8 qy, i32 sel4 (immediate), i32 sar (SAR register value) ]
def int_riscv_esp_cmul_u8_m : Intrinsic<[llvm_v16i8_ty], 
                                         [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_i32_ty, llvm_i32_ty], 
                                         [IntrNoMem, ImmArg<ArgIndex<3>>]>;

// CMUL.U8.LD.INCP - Unsigned 8-bit complex multiplication with load and pointer increment
// Returns: {v16i8 qz, v16i8 qu, ptr} (complex multiplication result, loaded vector, updated pointer)
// Parameters: [ v16i8 qz_in (passthru), v16i8 qx, v16i8 qy, ptr, i32 offset, i32 sar (SAR register value) ]
def int_riscv_esp_cmul_u8_ld_incp_m : Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty, llvm_ptr_ty], 
                                                 [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty], 
                                                 [IntrReadMem, IntrArgMemOnly]>;

// CMUL.U8.ST.INCP - Unsigned 8-bit complex multiplication with store and pointer increment
// Returns: {v16i8 qz, ptr} (complex multiplication result, updated pointer)
// Parameters: [ v16i8 qz_in (passthru), v16i8 qx, v16i8 qy, v16i8 qu, ptr, i32 offset, i32 sar (SAR register value) ]
def int_riscv_esp_cmul_u8_st_incp_m : Intrinsic<[llvm_v16i8_ty, llvm_ptr_ty], 
                                                 [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty], 
                                                 [IntrWriteMem, IntrArgMemOnly]>;

// MOVX.R.SAR - Read SAR register
// Returns: i32 (SAR register value, low 6 bits)
// Parameters: [ i32 sar (SAR register value, passthru for explicit state passing) ]
def int_riscv_esp_movx_r_sar_m : Intrinsic<[llvm_i32_ty], 
                                            [llvm_i32_ty], 
                                            [IntrNoMem]>;

// MOVX.R.SAR_BYTES - Read SAR_BYTES register
// Returns: i32 (SAR_BYTES register value, low 4 bits)
// Parameters: [ i32 sar_bytes (SAR_BYTES register value, passthru for explicit state passing) ]
def int_riscv_esp_movx_r_sar_bytes_m : Intrinsic<[llvm_i32_ty], 
                                                   [llvm_i32_ty], 
                                                   [IntrNoMem]>;

// ESP.VADD.S16.LD.INCP (_m version)
// Returns struct {v8i16 qv, v16i8 qu, i8* ptr}
def int_riscv_esp_vadd_s16_ld_incp_m : Intrinsic<
	[llvm_v8i16_ty, llvm_v16i8_ty, llvm_ptr_ty],  // Returns: qv, qu, ptr
	[llvm_v8i16_ty, llvm_v8i16_ty, llvm_ptr_ty],  // Parameters: qx, qy, rs1
	[IntrReadMem, IntrArgMemOnly]
>;

// ESP.VADD.S16.ST.INCP (_m version)
// Returns struct {v8i16 qv, i8* ptr}
// ESP.VADD.S16.ST.INCP (_m version)
// Returns struct {v8i16 qv, i8* ptr}
// Note: qv is input parameter (for register selection) but output from instruction
def int_riscv_esp_vadd_s16_st_incp_m : Intrinsic<
	[llvm_v8i16_ty, llvm_ptr_ty],  // Returns: qv, ptr
	[llvm_v8i16_ty, llvm_v8i16_ty, llvm_v16i8_ty, llvm_ptr_ty, llvm_v8i16_ty],  // Parameters: qx, qy, qu, rs1, qv
	[IntrWriteMem, IntrArgMemOnly]
>;

// ESP.VADD.S8.LD.INCP (_m version)
// Returns struct {v16i8 qv, v16i8 qu, i8* ptr}
// Key: Remove ClangBuiltin, use manual CodeGen
def int_riscv_esp_vadd_s8_ld_incp_m : Intrinsic<
	[llvm_v16i8_ty, llvm_v16i8_ty, llvm_ptr_ty],  // Returns: qv, qu, ptr
	[llvm_v16i8_ty, llvm_v16i8_ty, llvm_ptr_ty],  // Parameters: qx, qy, rs1
	[IntrReadMem, IntrArgMemOnly]  // Attributes: read memory
>;

// ESP.VADD.S8.ST.INCP (_m version)
// Returns struct {v16i8 qv, i8* ptr}
// Note: qv is input parameter (for register selection) but output from instruction
def int_riscv_esp_vadd_s8_st_incp_m : Intrinsic<
	[llvm_v16i8_ty, llvm_ptr_ty],  // Returns: qv, ptr
	[llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_ptr_ty, llvm_v16i8_ty],  // Parameters: qx, qy, qu, rs1, qv
	[IntrWriteMem, IntrArgMemOnly]
>;

// ESP.VADD.U16.LD.INCP (_m version)
// Returns struct {v8i16 qv, v16i8 qu, i8* ptr}
def int_riscv_esp_vadd_u16_ld_incp_m : Intrinsic<
	[llvm_v8i16_ty, llvm_v16i8_ty, llvm_ptr_ty],  // Returns: qv, qu, ptr
	[llvm_v8i16_ty, llvm_v8i16_ty, llvm_ptr_ty],  // Parameters: qx, qy, rs1
	[IntrReadMem, IntrArgMemOnly]
>;

// ESP.VADD.U16.ST.INCP (_m version)
// Returns struct {v8i16 qv, i8* ptr}
// Note: qv is input parameter (for register selection) but output from instruction
def int_riscv_esp_vadd_u16_st_incp_m : Intrinsic<
	[llvm_v8i16_ty, llvm_ptr_ty],  // Returns: qv, ptr
	[llvm_v8i16_ty, llvm_v8i16_ty, llvm_v16i8_ty, llvm_ptr_ty, llvm_v8i16_ty],  // Parameters: qx, qy, qu, rs1, qv
	[IntrWriteMem, IntrArgMemOnly]
>;

// ESP.VADD.U8.LD.INCP (_m version)
// Returns struct {v16i8 qv, v16i8 qu, i8* ptr}
def int_riscv_esp_vadd_u8_ld_incp_m : Intrinsic<
	[llvm_v16i8_ty, llvm_v16i8_ty, llvm_ptr_ty],  // Returns: qv, qu, ptr
	[llvm_v16i8_ty, llvm_v16i8_ty, llvm_ptr_ty],  // Parameters: qx, qy, rs1
	[IntrReadMem, IntrArgMemOnly]
>;

// ESP.VADD.U8.ST.INCP (_m version)
// Returns struct {v16i8 qv, i8* ptr}
// Note: qv is input parameter (for register selection) but output from instruction
def int_riscv_esp_vadd_u8_st_incp_m : Intrinsic<
	[llvm_v16i8_ty, llvm_ptr_ty],  // Returns: qv, ptr
	[llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_ptr_ty, llvm_v16i8_ty],  // Parameters: qx, qy, qu, rs1, qv
	[IntrWriteMem, IntrArgMemOnly]
>;

// SRCMB.S16.QACC - Shift amount from rs1 register (rs1[5:0])
// Returns v8i16 (saturated 16-bit signed values)
// QACC is passed as explicit phantom operand (4x128-bit) for proper data flow tracking
def int_riscv_esp_srcmb_s16_qacc_m : Intrinsic<[llvm_v8i16_ty], 
                                                 [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_i32_ty, llvm_i32_ty], 
                                                 [IntrNoMem]>;

// SRCMB.S8.QACC - Shift amount from rs1 register (rs1[4:0])
// Returns v16i8 (saturated 8-bit signed values)
// QACC is passed as explicit phantom operand (4x128-bit) for proper data flow tracking
def int_riscv_esp_srcmb_s8_qacc_m : Intrinsic<[llvm_v16i8_ty], 
                                                 [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_i32_ty, llvm_i32_ty], 
                                                 [IntrNoMem]>;

// SRCMB.U16.QACC - Shift amount from rs1 register (rs1[5:0])
// Returns v8i16 (saturated 16-bit unsigned values)
// QACC is passed as explicit phantom operand (4x128-bit) for proper data flow tracking
def int_riscv_esp_srcmb_u16_qacc_m : Intrinsic<[llvm_v8i16_ty], 
                                                  [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_i32_ty, llvm_i32_ty], 
                                                  [IntrNoMem]>;

// SRCMB.U8.QACC - Shift amount from rs1 register (rs1[4:0])
// Returns v16i8 (saturated 8-bit unsigned values)
// QACC is passed as explicit phantom operand (4x128-bit) for proper data flow tracking
def int_riscv_esp_srcmb_u8_qacc_m : Intrinsic<[llvm_v16i8_ty], 
                                                 [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_i32_ty, llvm_i32_ty], 
                                                 [IntrNoMem]>;

// ESP.VMULAS.S16.QACC (_m version) - Multiply-accumulate to QACC_H and QACC_L
// Returns: {v16i8, v16i8, v16i8, v16i8} - 4x128-bit QACC directly
// Parameters: v16i8, v16i8, v16i8, v16i8 (4x128-bit QACC passthru), v8i16, v8i16 (qx, qy)
def int_riscv_esp_vmulas_s16_qacc_m :
	Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],  // Returns: 4x128-bit QACC
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,   // Current QACC (4x128-bit passthru)
               llvm_v8i16_ty,                                                   // qx
               llvm_v8i16_ty],                                                  // qy
              [IntrNoMem]>;

// ESP.VMULAS.S16.QACC.LD.IP (_m version) - Load 128-bit, multiply-accumulate to QACC_H and QACC_L
// Returns: {ptr, v16i8 qu, v16i8, v16i8, v16i8, v16i8} - ptr + qu + 4x128-bit QACC directly
// Parameters: v16i8, v16i8, v16i8, v16i8 (4x128-bit QACC passthru), v8i16, v8i16 (qx, qy), ptr, i32 (offset)
def int_riscv_esp_vmulas_s16_qacc_ld_ip_m :
	Intrinsic<[llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],  // Returns: ptr + qu + 4x128-bit QACC (6 values)
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,               // Current QACC (4x128-bit passthru)
               llvm_v8i16_ty,                                                               // qx
               llvm_v8i16_ty,                                                               // qy
               llvm_ptr_ty,                                 // pointer
               llvm_i32_ty],                                // offset
              [IntrReadMem, IntrArgMemOnly]>;

// ESP.VMULAS.S16.QACC.LD.XP (_m version)
def int_riscv_esp_vmulas_s16_qacc_ld_xp_m :
	Intrinsic<[llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],  // Returns: ptr + qu + 4x128-bit QACC (6 values)
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,               // Current QACC (4x128-bit passthru)
               llvm_v8i16_ty,                                                               // qx
               llvm_v8i16_ty,                                                               // qy
               llvm_ptr_ty,                                 // pointer
               llvm_i32_ty],                                // register offset
              [IntrReadMem, IntrArgMemOnly]>;

// ESP.VMULAS.S16.QACC.ST.IP (_m version)
def int_riscv_esp_vmulas_s16_qacc_st_ip_m :
	Intrinsic<[llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],  // Returns: ptr + 4x128-bit QACC
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,               // Current QACC (4x128-bit passthru)
               llvm_v16i8_ty,                                                               // qu
               llvm_v8i16_ty,                                                               // qx
               llvm_v8i16_ty,                                                               // qy
               llvm_ptr_ty,                                 // pointer
               llvm_i32_ty],                                // offset
              [IntrWriteMem, IntrArgMemOnly]>;

// ESP.VMULAS.S16.QACC.ST.XP (_m version)
def int_riscv_esp_vmulas_s16_qacc_st_xp_m :
	Intrinsic<[llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],  // Returns: ptr + 4x128-bit QACC
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,               // Current QACC (4x128-bit passthru)
               llvm_v16i8_ty,                                                               // qu
               llvm_v8i16_ty,                                                               // qx
               llvm_v8i16_ty,                                                               // qy
               llvm_ptr_ty,                                 // pointer
               llvm_i32_ty],                                // rs2 register offset
              [IntrWriteMem, IntrArgMemOnly]>;

// ESP.VMULAS.S8.QACC (_m version) - Multiply-accumulate to QACC_H and QACC_L
// Returns: {v16i8, v16i8, v16i8, v16i8} - 4x128-bit QACC directly
// Parameters: v16i8, v16i8, v16i8, v16i8 (4x128-bit QACC passthru), v16i8, v16i8 (qx, qy)
def int_riscv_esp_vmulas_s8_qacc_m :
	Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],  // Returns: 4x128-bit QACC
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,   // Current QACC (4x128-bit passthru)
               llvm_v16i8_ty,                                                   // qx
               llvm_v16i8_ty],                                                  // qy
              [IntrNoMem]>;

// ESP.VMULAS.S8.QACC.LD.IP (_m version)
def int_riscv_esp_vmulas_s8_qacc_ld_ip_m :
	Intrinsic<[llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],  // Returns: ptr + qu + 4x128-bit QACC
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,               // Current QACC (4x128-bit passthru)
               llvm_v16i8_ty,                                                               // qx
               llvm_v16i8_ty,                                                               // qy
               llvm_ptr_ty,                                 // pointer
               llvm_i32_ty],                                // offset
              [IntrReadMem, IntrArgMemOnly]>;

// ESP.VMULAS.S8.QACC.LD.XP (_m version)
def int_riscv_esp_vmulas_s8_qacc_ld_xp_m :
	Intrinsic<[llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],  // Returns: ptr + qu + 4x128-bit QACC (6 values)
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,               // Current QACC (4x128-bit passthru)
               llvm_v16i8_ty,                                                               // qx
               llvm_v16i8_ty,                                                               // qy
               llvm_ptr_ty,                                 // pointer
               llvm_i32_ty],                                // register offset
              [IntrReadMem, IntrArgMemOnly]>;

// ESP.VMULAS.S8.QACC.ST.IP (_m version)
def int_riscv_esp_vmulas_s8_qacc_st_ip_m :
	Intrinsic<[llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],  // Returns: ptr + 4x128-bit QACC
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,               // Current QACC (4x128-bit passthru)
               llvm_v16i8_ty,                                                               // qu
               llvm_v16i8_ty,                                                               // qx
               llvm_v16i8_ty,                                                               // qy
               llvm_ptr_ty,                                 // pointer
               llvm_i32_ty],                                // offset
              [IntrWriteMem, IntrArgMemOnly]>;

// ESP.VMULAS.S8.QACC.ST.XP (_m version)
def int_riscv_esp_vmulas_s8_qacc_st_xp_m :
	Intrinsic<[llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],  // Returns: ptr + 4x128-bit QACC
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,               // Current QACC (4x128-bit passthru)
               llvm_v16i8_ty,                                                               // qu
               llvm_v16i8_ty,                                                               // qx
               llvm_v16i8_ty,                                                               // qy
               llvm_ptr_ty,                                 // pointer
               llvm_i32_ty],                                // rs2 register offset
              [IntrWriteMem, IntrArgMemOnly]>;

// ESP.VMULAS.U16.QACC (_m version) - Multiply-accumulate to QACC_H and QACC_L
// Returns: {v16i8, v16i8, v16i8, v16i8} - 4x128-bit QACC directly
// Parameters: v16i8, v16i8, v16i8, v16i8 (4x128-bit QACC passthru), v8i16, v8i16 (qx, qy)
def int_riscv_esp_vmulas_u16_qacc_m :
	Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],  // Returns: 4x128-bit QACC
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,   // Current QACC (4x128-bit passthru)
               llvm_v8i16_ty,                                                   // qx
               llvm_v8i16_ty],                                                  // qy
              [IntrNoMem]>;

// ESP.VMULAS.U16.QACC.LD.IP (_m version)
def int_riscv_esp_vmulas_u16_qacc_ld_ip_m :
	Intrinsic<[llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],  // Returns: ptr + qu + 4x128-bit QACC
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,               // Current QACC (4x128-bit passthru)
               llvm_v8i16_ty,                                                               // qx
               llvm_v8i16_ty,                                                               // qy
               llvm_ptr_ty,                                 // pointer
               llvm_i32_ty],                                // offset
              [IntrReadMem, IntrArgMemOnly]>;

// ESP.VMULAS.U16.QACC.LD.XP (_m version)
def int_riscv_esp_vmulas_u16_qacc_ld_xp_m :
	Intrinsic<[llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],  // Returns: ptr + qu + 4x128-bit QACC (6 values)
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,               // Current QACC (4x128-bit passthru)
               llvm_v8i16_ty,                                                               // qx
               llvm_v8i16_ty,                                                               // qy
               llvm_ptr_ty,                                 // pointer
               llvm_i32_ty],                                // register offset
              [IntrReadMem, IntrArgMemOnly]>;

// ESP.VMULAS.U16.QACC.ST.IP (_m version)
def int_riscv_esp_vmulas_u16_qacc_st_ip_m :
	Intrinsic<[llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],  // Returns: ptr + 4x128-bit QACC
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,               // Current QACC (4x128-bit passthru)
               llvm_v16i8_ty,                                                               // qu
               llvm_v8i16_ty,                                                               // qx
               llvm_v8i16_ty,                                                               // qy
               llvm_ptr_ty,                                 // pointer
               llvm_i32_ty],                                // offset
              [IntrWriteMem, IntrArgMemOnly]>;

// ESP.VMULAS.U16.QACC.ST.XP (_m version)
def int_riscv_esp_vmulas_u16_qacc_st_xp_m :
	Intrinsic<[llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],  // Returns: ptr + 4x128-bit QACC
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,               // Current QACC (4x128-bit passthru)
               llvm_v16i8_ty,                                                               // qu
               llvm_v8i16_ty,                                                               // qx
               llvm_v8i16_ty,                                                               // qy
               llvm_ptr_ty,                                 // pointer
               llvm_i32_ty],                                // rs2 register offset
              [IntrWriteMem, IntrArgMemOnly]>;

// ESP.VMULAS.U8.QACC (_m version) - Multiply-accumulate to QACC_H and QACC_L
// Returns: {v16i8, v16i8, v16i8, v16i8} - 4x128-bit QACC directly
// Parameters: v16i8, v16i8, v16i8, v16i8 (4x128-bit QACC passthru), v16i8, v16i8 (qx, qy)
def int_riscv_esp_vmulas_u8_qacc_m :
	Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],  // Returns: 4x128-bit QACC
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,   // Current QACC (4x128-bit passthru)
               llvm_v16i8_ty,                                                   // qx
               llvm_v16i8_ty],                                                  // qy
              [IntrNoMem]>;

// ESP.VMULAS.U8.QACC.LD.IP (_m version)
def int_riscv_esp_vmulas_u8_qacc_ld_ip_m :
	Intrinsic<[llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],  // Returns: ptr + qu + 4x128-bit QACC
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,               // Current QACC (4x128-bit passthru)
               llvm_v16i8_ty,                                                               // qx
               llvm_v16i8_ty,                                                               // qy
               llvm_ptr_ty,                                 // pointer
               llvm_i32_ty],                                // offset
              [IntrReadMem, IntrArgMemOnly]>;

// ESP.VMULAS.U8.QACC.LD.XP (_m version)
def int_riscv_esp_vmulas_u8_qacc_ld_xp_m :
	Intrinsic<[llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],  // Returns: ptr + qu + 4x128-bit QACC (6 values)
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,               // Current QACC (4x128-bit passthru)
               llvm_v16i8_ty,                                                               // qx
               llvm_v16i8_ty,                                                               // qy
               llvm_ptr_ty,                                 // pointer
               llvm_i32_ty],                                // register offset
              [IntrReadMem, IntrArgMemOnly]>;

// ESP.VMULAS.U8.QACC.ST.IP (_m version)
def int_riscv_esp_vmulas_u8_qacc_st_ip_m :
	Intrinsic<[llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],  // Returns: ptr + 4x128-bit QACC
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,               // Current QACC (4x128-bit passthru)
               llvm_v16i8_ty,                                                               // qu
               llvm_v16i8_ty,                                                               // qx
               llvm_v16i8_ty,                                                               // qy
               llvm_ptr_ty,                                 // pointer
               llvm_i32_ty],                                // offset
              [IntrWriteMem, IntrArgMemOnly]>;

// ESP.VMULAS.U8.QACC.ST.XP (_m version)
def int_riscv_esp_vmulas_u8_qacc_st_xp_m :
	Intrinsic<[llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],  // Returns: ptr + 4x128-bit QACC
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,               // Current QACC (4x128-bit passthru)
               llvm_v16i8_ty,                                                               // qu
               llvm_v16i8_ty,                                                               // qx
               llvm_v16i8_ty,                                                               // qy
               llvm_ptr_ty,                                 // pointer
               llvm_i32_ty],                                // rs2 register offset
              [IntrWriteMem, IntrArgMemOnly]>;

// ESP.VMULAS.S16.QACC.LDBC.INCP (_m version)
def int_riscv_esp_vmulas_s16_qacc_ldbc_incp_m :
	Intrinsic<[llvm_v16i8_ty, llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],  // Returns: qu + ptr + 4x128-bit QACC (6 values)
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,               // Current QACC (4x128-bit passthru)
               llvm_v8i16_ty,                                                               // qx
               llvm_v8i16_ty,                                                               // qy
               llvm_ptr_ty],                                 // pointer
              [IntrReadMem, IntrArgMemOnly]>;

// ESP.VMULAS.S8.QACC.LDBC.INCP (_m version)
def int_riscv_esp_vmulas_s8_qacc_ldbc_incp_m :
	Intrinsic<[llvm_v16i8_ty, llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],  // Returns: qu + ptr + 4x128-bit QACC (6 values)
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,               // Current QACC (4x128-bit passthru)
               llvm_v16i8_ty,                                                               // qx
               llvm_v16i8_ty,                                                               // qy
               llvm_ptr_ty],                                 // pointer
              [IntrReadMem, IntrArgMemOnly]>;

// ESP.VMULAS.U16.QACC.LDBC.INCP (_m version)
def int_riscv_esp_vmulas_u16_qacc_ldbc_incp_m :
	Intrinsic<[llvm_v16i8_ty, llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],  // Returns: qu + ptr + 4x128-bit QACC (6 values)
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,               // Current QACC (4x128-bit passthru)
               llvm_v8i16_ty,                                                               // qx
               llvm_v8i16_ty,                                                               // qy
               llvm_ptr_ty],                                 // pointer
              [IntrReadMem, IntrArgMemOnly]>;

// ESP.VMULAS.U8.QACC.LDBC.INCP (_m version)
def int_riscv_esp_vmulas_u8_qacc_ldbc_incp_m :
	Intrinsic<[llvm_v16i8_ty, llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],  // Returns: qu + ptr + 4x128-bit QACC (6 values)
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,               // Current QACC (4x128-bit passthru)
               llvm_v16i8_ty,                                                               // qx
               llvm_v16i8_ty,                                                               // qy
               llvm_ptr_ty],                                 // pointer
              [IntrReadMem, IntrArgMemOnly]>;
}
