let TargetPrefix = "riscv" in {
// Load with Post-Increment (Immediate offset)
// Returns: { vector data, updated pointer }
// Parameters: [ original pointer, offset (immediate) ]
def int_riscv_esp_vld_128_ip : Intrinsic<[llvm_v16i8_ty, llvm_ptr_ty],
                                         [llvm_ptr_ty, llvm_i32_ty], 
                                         [IntrReadMem, IntrArgMemOnly]>;

// ESP.LD.128.USAR.IP (_m version) - Load 128-bit data and write SAR_BYTES register
// Returns: vector + updated pointer + SAR_BYTES (32-bit, only low 4 bits used)
// Parameters: [pointer, immediate offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
// Note: SAR_BYTES[3:0] = rs1[3:0] (hardware automatically extracts only low 4 bits from rs1)
def int_riscv_esp_ld_128_usar_ip : Intrinsic<[llvm_v16i8_ty, llvm_ptr_ty, llvm_i32_ty], 
                                         [llvm_ptr_ty, llvm_i32_ty], 
                                         [IntrReadMem, IntrArgMemOnly]>;

// ESP.LD.UA.STATE.IP (_m version) - Load 128-bit data to UA_STATE[127:0]
// Returns: updated pointer + new UA_STATE (128-bit)
// Parameters: [current UA_STATE (passthru), pointer, immediate offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_ld_ua_state_ip : Intrinsic<
    [llvm_v16i8_ty, llvm_ptr_ty],       
    [llvm_v16i8_ty,                      
     llvm_ptr_ty,                        
     llvm_i32_ty],                       
    [IntrReadMem, IntrArgMemOnly]>;      

// ESP.ST.UA.STATE.IP (_m version) - Store UA_STATE register to memory
// Returns: updated pointer + new UA_STATE (explicit state passing)
// Parameters: [current UA_STATE (passthru), pointer, immediate offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_st_ua_state_ip : Intrinsic<
    [llvm_v16i8_ty,                      
     llvm_ptr_ty],                       
    [llvm_v16i8_ty,                      
     llvm_ptr_ty,                        
     llvm_i32_ty],                       
    [IntrWriteMem, IntrArgMemOnly]>;      

// ESP.LD.XACC.IP (_m version) - Load 64-bit data and store low 40 bits to XACC
// Returns: updated pointer + new XACC as {i32 low, i32 high}
// Parameters: [xacc_low_in (i32), xacc_high_in (i32, only low 8 bits valid), ptr, offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_ld_xacc_ip : Intrinsic<
    [llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty], 
    [llvm_i32_ty,                             
     llvm_i32_ty,                             
     llvm_ptr_ty,                             
     llvm_i32_ty],                            
    [IntrReadMem, IntrArgMemOnly]>;           

// Load with Post-Increment (Register offset)
// Returns: { vector data, updated pointer }
// Parameters: [ original pointer, offset (register) ]
def int_riscv_esp_vld_128_xp : Intrinsic<[llvm_v16i8_ty, llvm_ptr_ty],
                                         [llvm_ptr_ty, llvm_i32_ty], 
                                         [IntrReadMem, IntrArgMemOnly]>;
// Load 128-bit with SAR_BYTES update (Register offset)
// ESP.LD.128.USAR.XP (_m version) - Load 128-bit data and write SAR_BYTES register
// Returns: vector + updated pointer + SAR_BYTES (32-bit, only low 4 bits used)
// Parameters: [pointer, register offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
// Note: SAR_BYTES[3:0] = rs1[3:0] (hardware automatically extracts only low 4 bits from rs1)
def int_riscv_esp_ld_128_usar_xp : Intrinsic<[llvm_v16i8_ty, llvm_ptr_ty, llvm_i32_ty],
                                         [llvm_ptr_ty, llvm_i32_ty],
                                         [IntrReadMem, IntrArgMemOnly]>;

// Store with Post-Increment (Immediate offset)
// Returns: updated pointer
// Parameters: [ vector data, original pointer, offset (immediate) ]
def int_riscv_esp_vst_128_ip : Intrinsic<[llvm_ptr_ty],
                                         [llvm_v16i8_ty, llvm_ptr_ty, llvm_i32_ty], 
                                         [IntrWriteMem, IntrArgMemOnly]>;

// Store with Post-Increment (Register offset)
// Returns: updated pointer
// Parameters: [ vector data, original pointer, offset (register) ]
def int_riscv_esp_vst_128_xp : Intrinsic<[llvm_ptr_ty],
                                         [llvm_v16i8_ty, llvm_ptr_ty, llvm_i32_ty], 
                                         [IntrWriteMem, IntrArgMemOnly]>;

// STXQ.32 - (ptr rs1, qw v4i32, qu v8i16, sel4, sel8)
def int_riscv_esp_stxq_32 : Intrinsic<[],
                                        [llvm_ptr_ty, llvm_v4i32_ty, llvm_v8i16_ty, llvm_i32_ty, llvm_i32_ty],
                                        [IntrWriteMem, IntrArgMemOnly,
                                         ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

// Load H.64 with Post-Increment (Immediate offset)
// Returns: { 64-bit data, updated pointer }
// Parameters: [ original pointer, offset (immediate) ]
// H version: load to upper 64-bit segment of qu (qu[127:64])
def int_riscv_esp_vld_h_64_ip : Intrinsic<[llvm_v8i8_ty, llvm_ptr_ty],
                                            [llvm_ptr_ty, llvm_i32_ty], 
                                            [IntrReadMem, IntrArgMemOnly]>;

// Load H.64 with Post-Increment (Register offset)
// Returns: { 64-bit data, updated pointer }
// Parameters: [ original pointer, offset (register) ]
def int_riscv_esp_vld_h_64_xp : Intrinsic<[llvm_v8i8_ty, llvm_ptr_ty],
                                            [llvm_ptr_ty, llvm_i32_ty], 
                                            [IntrReadMem, IntrArgMemOnly]>;

// Load L.64 with Post-Increment (Immediate offset)
// Returns: { 64-bit data, updated pointer }
// Parameters: [ original pointer, offset (immediate) ]
// L version: load to lower 64-bit segment of qu (qu[63:0])
def int_riscv_esp_vld_l_64_ip : Intrinsic<[llvm_v8i8_ty, llvm_ptr_ty],
                                            [llvm_ptr_ty, llvm_i32_ty], 
                                            [IntrReadMem, IntrArgMemOnly]>;

// Load L.64 with Post-Increment (Register offset)
// Returns: { 64-bit data, updated pointer }
// Parameters: [ original pointer, offset (register) ]
def int_riscv_esp_vld_l_64_xp : Intrinsic<[llvm_v8i8_ty, llvm_ptr_ty],
                                            [llvm_ptr_ty, llvm_i32_ty], 
                                            [IntrReadMem, IntrArgMemOnly]>;

// Store H.64 with Post-Increment (Immediate offset)
// Returns: updated pointer
// Parameters: [ 64-bit data, original pointer, offset (immediate) ]
// H version: store upper 64-bit segment of qu (qu[127:64])
def int_riscv_esp_vst_h_64_ip : Intrinsic<[llvm_ptr_ty],
                                             [llvm_v8i8_ty, llvm_ptr_ty, llvm_i32_ty], 
                                             [IntrWriteMem, IntrArgMemOnly]>;

// Store H.64 with Post-Increment (Register offset)
// Returns: updated pointer
// Parameters: [ 64-bit data, original pointer, offset (register) ]
def int_riscv_esp_vst_h_64_xp : Intrinsic<[llvm_ptr_ty],
                                             [llvm_v8i8_ty, llvm_ptr_ty, llvm_i32_ty], 
                                             [IntrWriteMem, IntrArgMemOnly]>;

// Store L.64 with Post-Increment (Immediate offset)
// Returns: updated pointer
// Parameters: [ 64-bit data, original pointer, offset (immediate) ]
// L version: store lower 64-bit segment of qu (qu[63:0])
def int_riscv_esp_vst_l_64_ip : Intrinsic<[llvm_ptr_ty],
                                             [llvm_v8i8_ty, llvm_ptr_ty, llvm_i32_ty], 
                                             [IntrWriteMem, IntrArgMemOnly]>;

// Store L.64 with Post-Increment (Register offset)
// Returns: updated pointer
// Parameters: [ 64-bit data, original pointer, offset (register) ]
def int_riscv_esp_vst_l_64_xp : Intrinsic<[llvm_ptr_ty],
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

// VLDHBC.16.INCP - Load and broadcast 16-bit halves; updated pointer in result
def int_riscv_esp_vldhbc_16_incp_m : Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty, llvm_ptr_ty],
                                                [llvm_ptr_ty],
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
def int_riscv_esp_vsadds_s16 : Intrinsic<[llvm_v8i16_ty],
    [llvm_v8i16_ty, llvm_i32_ty, llvm_i32_ty],
    [IntrNoMem, ImmArg<ArgIndex<2>>]>;

// VSADDS - Signed 8-bit
// Returns: v16i8 (vector saturating addition on 8-bit signed data)
// Parameters: [ v16i8 vector qx, i32 rs1, i32 sat ]
def int_riscv_esp_vsadds_s8 : Intrinsic<[llvm_v16i8_ty],
    [llvm_v16i8_ty, llvm_i32_ty, llvm_i32_ty],
    [IntrNoMem, ImmArg<ArgIndex<2>>]>;

// VSADDS - Unsigned 16-bit
// Returns: v8i16 (vector saturating addition on 16-bit unsigned data)
// Parameters: [ v8i16 vector qx, i32 rs1, i32 sat ]
def int_riscv_esp_vsadds_u16 : Intrinsic<[llvm_v8i16_ty],
    [llvm_v8i16_ty, llvm_i32_ty, llvm_i32_ty],
    [IntrNoMem, ImmArg<ArgIndex<2>>]>;

// VSADDS - Unsigned 8-bit
// Returns: v16i8 (vector saturating addition on 8-bit unsigned data)
// Parameters: [ v16i8 vector qx, i32 rs1, i32 sat ]
def int_riscv_esp_vsadds_u8 : Intrinsic<[llvm_v16i8_ty],
    [llvm_v16i8_ty, llvm_i32_ty, llvm_i32_ty],
    [IntrNoMem, ImmArg<ArgIndex<2>>]>;

// VSSUBS - Signed 16-bit
// Returns: v8i16 (vector saturating subtraction on 16-bit signed data)
// Parameters: [ v8i16 vector qx, i32 rs1, i32 sat ]
def int_riscv_esp_vssubs_s16 : Intrinsic<[llvm_v8i16_ty],
    [llvm_v8i16_ty, llvm_i32_ty, llvm_i32_ty],
    [IntrNoMem, ImmArg<ArgIndex<2>>]>;

// VSSUBS - Signed 8-bit
// Returns: v16i8 (vector saturating subtraction on 8-bit signed data)
// Parameters: [ v16i8 vector qx, i32 rs1, i32 sat ]
def int_riscv_esp_vssubs_s8 : Intrinsic<[llvm_v16i8_ty],
    [llvm_v16i8_ty, llvm_i32_ty, llvm_i32_ty],
    [IntrNoMem, ImmArg<ArgIndex<2>>]>;

// VSSUBS - Unsigned 16-bit
// Returns: v8i16 (vector saturating subtraction on 16-bit unsigned data)
// Parameters: [ v8i16 vector qx, i32 rs1, i32 sat ]
def int_riscv_esp_vssubs_u16 : Intrinsic<[llvm_v8i16_ty],
    [llvm_v8i16_ty, llvm_i32_ty, llvm_i32_ty],
    [IntrNoMem, ImmArg<ArgIndex<2>>]>;

// VSSUBS - Unsigned 8-bit
// Returns: v16i8 (vector saturating subtraction on 8-bit unsigned data)
// Parameters: [ v16i8 vector qx, i32 rs1, i32 sat ]
def int_riscv_esp_vssubs_u8 : Intrinsic<[llvm_v16i8_ty],
    [llvm_v16i8_ty, llvm_i32_ty, llvm_i32_ty],
    [IntrNoMem, ImmArg<ArgIndex<2>>]>;

// VSUB.S32 (unified 2.1/2.2 API)
def int_riscv_esp_vsub_s32 : Intrinsic<[llvm_v4i32_ty],
    [llvm_v4i32_ty, llvm_v4i32_ty, llvm_i32_ty],
    [IntrNoMem, ImmArg<ArgIndex<2>>]>;

// VSUB.U16 (unified 2.1/2.2 API)
def int_riscv_esp_vsub_u16 : Intrinsic<[llvm_v8i16_ty],
    [llvm_v8i16_ty, llvm_v8i16_ty, llvm_i32_ty],
    [IntrNoMem, ImmArg<ArgIndex<2>>]>;

// VSUB.U32 (unified 2.1/2.2 API)
def int_riscv_esp_vsub_u32 : Intrinsic<[llvm_v4i32_ty],
    [llvm_v4i32_ty, llvm_v4i32_ty, llvm_i32_ty],
    [IntrNoMem, ImmArg<ArgIndex<2>>]>;


// VCMP.EQ - Equal comparison
// VCMP.EQ.S16
def int_riscv_esp_vcmp_eq_s16 : Intrinsic<[llvm_v8i16_ty], 
                                               [llvm_v8i16_ty, llvm_v8i16_ty], 
                                               [IntrNoMem]>;


// VCMP.EQ.U16
def int_riscv_esp_vcmp_eq_u16_m : Intrinsic<[llvm_v8i16_ty], 
                                               [llvm_v8i16_ty, llvm_v8i16_ty], 
                                               [IntrNoMem]>;
// VCMP.EQ.S32
def int_riscv_esp_vcmp_eq_s32 : Intrinsic<[llvm_v4i32_ty], 
                                               [llvm_v4i32_ty, llvm_v4i32_ty], 
                                               [IntrNoMem]>;
// VCMP.EQ.U32
def int_riscv_esp_vcmp_eq_u32_m : Intrinsic<[llvm_v4i32_ty], 
                                               [llvm_v4i32_ty, llvm_v4i32_ty], 
                                               [IntrNoMem]>;


// VCMP.GT - Greater than comparison
// VCMP.GT.S16
def int_riscv_esp_vcmp_gt_s16 : Intrinsic<[llvm_v8i16_ty], 
                                               [llvm_v8i16_ty, llvm_v8i16_ty], 
                                               [IntrNoMem]>;

// VCMP.GT.U16
def int_riscv_esp_vcmp_gt_u16_m : Intrinsic<[llvm_v8i16_ty], 
                                               [llvm_v8i16_ty, llvm_v8i16_ty], 
                                               [IntrNoMem]>;
// VCMP.GT.S32
def int_riscv_esp_vcmp_gt_s32 : Intrinsic<[llvm_v4i32_ty], 
                                               [llvm_v4i32_ty, llvm_v4i32_ty], 
                                               [IntrNoMem]>;
// VCMP.GT.U32
def int_riscv_esp_vcmp_gt_u32_m : Intrinsic<[llvm_v4i32_ty], 
                                               [llvm_v4i32_ty, llvm_v4i32_ty], 
                                               [IntrNoMem]>;


// VCMP.LT - Less than comparison
// VCMP.LT.S16
def int_riscv_esp_vcmp_lt_s16 : Intrinsic<[llvm_v8i16_ty], 
                                               [llvm_v8i16_ty, llvm_v8i16_ty], 
                                               [IntrNoMem]>;

// VCMP.LT.U16
def int_riscv_esp_vcmp_lt_u16 : Intrinsic<[llvm_v8i16_ty], 
                                               [llvm_v8i16_ty, llvm_v8i16_ty], 
                                               [IntrNoMem]>;
// VCMP.LT.S32
def int_riscv_esp_vcmp_lt_s32 : Intrinsic<[llvm_v4i32_ty], 
                                               [llvm_v4i32_ty, llvm_v4i32_ty], 
                                               [IntrNoMem]>;
// VCMP.LT.U32
def int_riscv_esp_vcmp_lt_u32 : Intrinsic<[llvm_v4i32_ty], 
                                               [llvm_v4i32_ty, llvm_v4i32_ty], 
                                               [IntrNoMem]>;


// VCLAMP - Signed 16-bit
// Returns: v8i16 (vector clamp on 16-bit signed data)
// Parameters: [ v8i16 vector qx, i32 sel16 (immediate) ]
def int_riscv_esp_vclamp_s16 : Intrinsic<[llvm_v8i16_ty], 
                                           [llvm_v8i16_ty, llvm_i32_ty], 
                                           [IntrNoMem, ImmArg<ArgIndex<1>>]>;

// VPRELU.S16 (unified 2.2 API)
// Returns: v8i16
// Parameters: [qx, qy, rs1, sat, rm]
def int_riscv_esp_vprelu_s16 : Intrinsic<[llvm_v8i16_ty],
                                          [llvm_v8i16_ty, llvm_v8i16_ty, llvm_i32_ty,
                                           llvm_i32_ty, llvm_i32_ty],
                                          [IntrNoMem, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

// VPRELU.S8 (unified 2.2 API)
def int_riscv_esp_vprelu_s8 : Intrinsic<[llvm_v16i8_ty],
                                         [llvm_v16i8_ty, llvm_v16i8_ty, llvm_i32_ty,
                                          llvm_i32_ty, llvm_i32_ty],
                                         [IntrNoMem, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

// VRELU.S16 (unified 2.2 API; in-place on qy)
def int_riscv_esp_vrelu_s16 : Intrinsic<[llvm_v8i16_ty],
                                          [llvm_v8i16_ty, llvm_i32_ty, llvm_i32_ty,
                                           llvm_i32_ty, llvm_i32_ty],
                                          [IntrNoMem, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

// VRELU.S8 (unified 2.2 API; in-place on qy)
def int_riscv_esp_vrelu_s8 : Intrinsic<[llvm_v16i8_ty],
                                        [llvm_v16i8_ty, llvm_i32_ty, llvm_i32_ty,
                                         llvm_i32_ty, llvm_i32_ty],
                                        [IntrNoMem, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

// VUNZIP - Vector unzip (deinterleave)
// Returns modified vectors (Read-Modify-Write operation)
// VUNZIP.16 - Returns {v8i16, v8i16}
def int_riscv_esp_vunzip_16 : Intrinsic<[llvm_v8i16_ty, llvm_v8i16_ty], 
                                             [llvm_v8i16_ty, llvm_v8i16_ty], 
                                             [IntrNoMem]>;
// VUNZIP.32 - Returns {v4i32, v4i32}
def int_riscv_esp_vunzip_32 : Intrinsic<[llvm_v4i32_ty, llvm_v4i32_ty], 
                                             [llvm_v4i32_ty, llvm_v4i32_ty], 
                                             [IntrNoMem]>;
// VUNZIP.8 - Returns {v16i8, v16i8}
def int_riscv_esp_vunzip_8 : Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty], 
                                            [llvm_v16i8_ty, llvm_v16i8_ty], 
                                            [IntrNoMem]>;

// VUNZIPT - Vector unzip transpose (3 vectors)
// Returns modified vectors {v8i16, v8i16, v8i16}
def int_riscv_esp_vunzipt_16 : Intrinsic<[llvm_v8i16_ty, llvm_v8i16_ty, llvm_v8i16_ty], 
                                              [llvm_v8i16_ty, llvm_v8i16_ty, llvm_v8i16_ty], 
                                              [IntrNoMem]>;
// Returns modified vectors {v16i8, v16i8, v16i8}
def int_riscv_esp_vunzipt_8 : Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty], 
                                             [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty], 
                                             [IntrNoMem]>;

// VZIP - Vector zip (interleave)
// Returns modified vectors (Read-Modify-Write operation)
// VZIP.16 - Returns {v8i16, v8i16}
def int_riscv_esp_vzip_16 : Intrinsic<[llvm_v8i16_ty, llvm_v8i16_ty], 
                                           [llvm_v8i16_ty, llvm_v8i16_ty], 
                                           [IntrNoMem]>;
// VZIP.32 - Returns {v4i32, v4i32}
def int_riscv_esp_vzip_32 : Intrinsic<[llvm_v4i32_ty, llvm_v4i32_ty], 
                                           [llvm_v4i32_ty, llvm_v4i32_ty], 
                                           [IntrNoMem]>;
// VZIP.8 - Returns {v16i8, v16i8}
def int_riscv_esp_vzip_8 : Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty], 
                                          [llvm_v16i8_ty, llvm_v16i8_ty], 
                                          [IntrNoMem]>;

// VZIPT - Vector zip transpose (3 vectors)
// Returns modified vectors {v8i16, v8i16, v8i16}
def int_riscv_esp_vzipt_16 : Intrinsic<[llvm_v8i16_ty, llvm_v8i16_ty, llvm_v8i16_ty], 
                                            [llvm_v8i16_ty, llvm_v8i16_ty, llvm_v8i16_ty], 
                                            [IntrNoMem]>;
// Returns modified vectors {v16i8, v16i8, v16i8}
def int_riscv_esp_vzipt_8 : Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty], 
                                           [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty], 
                                           [IntrNoMem]>;

// VEXT.S8 - Sign extend 8-bit to 16-bit
// Returns: {v8i16, v8i16} (two 8x16-bit vectors)
// Parameters: [ v16i8 vector qw ]
def int_riscv_esp_vext_s8 : Intrinsic<[llvm_v8i16_ty, llvm_v8i16_ty], 
                                        [llvm_v16i8_ty], 
                                        [IntrNoMem]>;

// VEXT.S16 - Sign extend 16-bit to 32-bit
// Returns: {v4i32, v4i32} (two 4x32-bit vectors)
// Parameters: [ v8i16 vector qw ]
def int_riscv_esp_vext_s16 : Intrinsic<[llvm_v4i32_ty, llvm_v4i32_ty], 
                                         [llvm_v8i16_ty], 
                                         [IntrNoMem]>;

// VEXT.U8 - Zero extend 8-bit to 16-bit
// Returns: {v8i16, v8i16} (two 8x16-bit vectors)
// Parameters: [ v16i8 vector qw ]
def int_riscv_esp_vext_u8 : Intrinsic<[llvm_v8i16_ty, llvm_v8i16_ty], 
                                        [llvm_v16i8_ty], 
                                        [IntrNoMem]>;

// VEXT.U16 - Zero extend 16-bit to 32-bit
// Returns: {v4i32, v4i32} (two 4x32-bit vectors)
// Parameters: [ v8i16 vector qw ]
def int_riscv_esp_vext_u16 : Intrinsic<[llvm_v4i32_ty, llvm_v4i32_ty], 
                                         [llvm_v8i16_ty], 
                                         [IntrNoMem]>;

// VSAT - Signed 8-bit
// Returns: v16i8 (vector saturation on 8-bit signed segments)
// Parameters: [ v16i8 vector qx, i32 rs1, i32 rs2 ]
def int_riscv_esp_vsat_s8 : Intrinsic<[llvm_v16i8_ty], 
                                         [llvm_v16i8_ty, llvm_i32_ty, llvm_i32_ty], 
                                         [IntrNoMem]>;

// VSAT - Unsigned 8-bit
// Returns: v16i8 (vector saturation on 8-bit unsigned segments)
// Parameters: [ v16i8 vector qx, i32 rs1, i32 rs2 ]
def int_riscv_esp_vsat_u8 : Intrinsic<[llvm_v16i8_ty], 
                                         [llvm_v16i8_ty, llvm_i32_ty, llvm_i32_ty], 
                                         [IntrNoMem]>;


// VSAT - Signed 16-bit
// Returns: v8i16 (vector saturation on 16-bit signed segments)
// Parameters: [ v8i16 vector qx, i32 rs1, i32 rs2 ]
def int_riscv_esp_vsat_s16 : Intrinsic<[llvm_v8i16_ty], 
                                            [llvm_v8i16_ty, llvm_i32_ty, llvm_i32_ty], 
                                            [IntrNoMem]>;

// VSAT - Signed 32-bit
// Returns: v4i32 (vector saturation on 32-bit signed segments)
// Parameters: [ v4i32 vector qx, i32 rs1, i32 rs2 ]
def int_riscv_esp_vsat_s32 : Intrinsic<[llvm_v4i32_ty], 
                                            [llvm_v4i32_ty, llvm_i32_ty, llvm_i32_ty], 
                                            [IntrNoMem]>;


// VSAT - Unsigned 16-bit
// Returns: v8i16 (vector saturation on 16-bit unsigned segments)
// Parameters: [ v8i16 vector qx, i32 rs1, i32 rs2 ]
def int_riscv_esp_vsat_u16 : Intrinsic<[llvm_v8i16_ty], 
                                            [llvm_v8i16_ty, llvm_i32_ty, llvm_i32_ty], 
                                            [IntrNoMem]>;

// VSAT - Unsigned 32-bit
// Returns: v4i32 (vector saturation on 32-bit unsigned segments)
// Parameters: [ v4i32 vector qx, i32 rs1, i32 rs2 ]
def int_riscv_esp_vsat_u32 : Intrinsic<[llvm_v4i32_ty], 
                                            [llvm_v4i32_ty, llvm_i32_ty, llvm_i32_ty], 
                                            [IntrNoMem]>;


// MAX - Signed 8-bit reduction
// Returns: i32 (maximum value from vector)
// Parameters: [ v16i8 vector qw ]
def int_riscv_esp_max_s8_a : Intrinsic<[llvm_i32_ty], 
                                         [llvm_v16i8_ty], 
                                         [IntrNoMem]>;

// MAX - Signed 16-bit reduction
// Returns: i32 (maximum value from vector)
// Parameters: [ v8i16 vector qw ]
def int_riscv_esp_max_s16_a : Intrinsic<[llvm_i32_ty], 
                                          [llvm_v8i16_ty], 
                                          [IntrNoMem]>;

// MAX - Unsigned 8-bit reduction
// Returns: i32 (maximum value from vector)
// Parameters: [ v16i8 vector qw ]
def int_riscv_esp_max_u8_a : Intrinsic<[llvm_i32_ty], 
                                         [llvm_v16i8_ty], 
                                         [IntrNoMem]>;

// MAX - Unsigned 16-bit reduction
// Returns: i32 (maximum value from vector)
// Parameters: [ v8i16 vector qw ]
def int_riscv_esp_max_u16_a : Intrinsic<[llvm_i32_ty], 
                                          [llvm_v8i16_ty], 
                                          [IntrNoMem]>;

// MIN - Signed 8-bit reduction
// Returns: i32 (minimum value from vector)
// Parameters: [ v16i8 vector qw ]
def int_riscv_esp_min_s8_a : Intrinsic<[llvm_i32_ty], 
                                         [llvm_v16i8_ty], 
                                         [IntrNoMem]>;

// MIN - Signed 16-bit reduction
// Returns: i32 (minimum value from vector)
// Parameters: [ v8i16 vector qw ]
def int_riscv_esp_min_s16_a : Intrinsic<[llvm_i32_ty], 
                                          [llvm_v8i16_ty], 
                                          [IntrNoMem]>;

// MIN - Unsigned 8-bit reduction
// Returns: i32 (minimum value from vector)
// Parameters: [ v16i8 vector qw ]
def int_riscv_esp_min_u8_a : Intrinsic<[llvm_i32_ty], 
                                         [llvm_v16i8_ty], 
                                         [IntrNoMem]>;

// MIN - Unsigned 16-bit reduction
// Returns: i32 (minimum value from vector)
// Parameters: [ v8i16 vector qw ]
def int_riscv_esp_min_u16_a : Intrinsic<[llvm_i32_ty], 
                                          [llvm_v8i16_ty], 
                                          [IntrNoMem]>;

// Vector Add - Signed 32-bit
// Returns: v4i32 (result vector)
// Parameters: [ v4i32 vector qx, v4i32 vector qy ]
def int_riscv_esp_vadd_s32 : Intrinsic<[llvm_v4i32_ty],
    [llvm_v4i32_ty, llvm_v4i32_ty, llvm_i32_ty],
    [IntrNoMem, ImmArg<ArgIndex<2>>]>;

// Vector Add - Unsigned 32-bit
// Returns: v4i32 (result vector)
// Parameters: [ v4i32 vector qx, v4i32 vector qy, i32 sat ]
def int_riscv_esp_vadd_u32 : Intrinsic<[llvm_v4i32_ty],
    [llvm_v4i32_ty, llvm_v4i32_ty, llvm_i32_ty],
    [IntrNoMem, ImmArg<ArgIndex<2>>]>;

// Vector Max - Signed 32-bit
// Returns: v4i32 (result vector)
// Parameters: [ v4i32 vector qx, v4i32 vector qy ]
def int_riscv_esp_vmax_s32 : Intrinsic<[llvm_v4i32_ty], 
                                         [llvm_v4i32_ty, llvm_v4i32_ty], 
                                         [IntrNoMem]>;

// Vector Absolute Value - 32-bit
// Returns: v4i32 (result vector)
// Parameters: [ v4i32 vector qy ]
def int_riscv_esp_vabs_32 : Intrinsic<[llvm_v4i32_ty], 
                                        [llvm_v4i32_ty], 
                                        [IntrNoMem]>;

// MAX - Signed 32-bit reduction
// Returns: i32 (maximum value from vector)
// Parameters: [ v4i32 vector qw ]
def int_riscv_esp_max_s32_a : Intrinsic<[llvm_i32_ty], 
                                         [llvm_v4i32_ty], 
                                         [IntrNoMem]>;

// MAX - Unsigned 32-bit reduction
// Returns: i32 (maximum value from vector)
// Parameters: [ v4i32 vector qw ]
def int_riscv_esp_max_u32_a : Intrinsic<[llvm_i32_ty], 
                                          [llvm_v4i32_ty], 
                                          [IntrNoMem]>;

// MIN - Signed 32-bit reduction
// Returns: i32 (minimum value from vector)
// Parameters: [ v4i32 vector qw ]
def int_riscv_esp_min_s32_a : Intrinsic<[llvm_i32_ty], 
                                         [llvm_v4i32_ty], 
                                         [IntrNoMem]>;

// MIN - Unsigned 32-bit reduction
// Returns: i32 (minimum value from vector)
// Parameters: [ v4i32 vector qw ]
def int_riscv_esp_min_u32_a : Intrinsic<[llvm_i32_ty], 
                                          [llvm_v4i32_ty], 
                                          [IntrNoMem]>;

// MOVX.W.FFT.BIT.WIDTH - Write FFT bit width register
// Returns: i32 (FFT_BIT_WIDTH value, 32-bit unsigned, only low 4 bits used)
// Parameters: [ i32 value to write (rs1, 32-bit unsigned) ]
def int_riscv_esp_movx_w_fft_bit_width : Intrinsic<[llvm_i32_ty], 
                                                     [llvm_i32_ty], 
                                                     [IntrNoMem]>;

// MOVX.R.FFT.BIT.WIDTH - Read FFT bit width register
// Returns: i32 (rd[31:0] = {28'b0, FFT_BIT_WIDTH[3:0]}, 32-bit unsigned)
// Parameters: [ i32 FFT_BIT_WIDTH (input, 32-bit unsigned, only low 4 bits used) ]
def int_riscv_esp_movx_r_fft_bit_width : Intrinsic<[llvm_i32_ty], 
                                                     [llvm_i32_ty], 
                                                     [IntrNoMem]>;

// MOVX.W.PERF / MOVX.R.PERF - PERF explicit state (i32 token) and GPR rs1
def int_riscv_esp_movx_w_perf : Intrinsic<[llvm_i32_ty],
                                          [llvm_i32_ty, llvm_i32_ty],
                                          [IntrNoMem]>;
def int_riscv_esp_movx_r_perf : Intrinsic<[llvm_i32_ty],
                                           [llvm_i32_ty, llvm_i32_ty],
                                           [IntrNoMem]>;

// FFT.R2BF.S16 (unified 2.2 API; sat only)
// Returns: {v8i16 qz, v8i16 qv}
// Parameters: [qx, qy, sel2, sat]
def int_riscv_esp_fft_r2bf_s16 : Intrinsic<[llvm_v8i16_ty, llvm_v8i16_ty],
                                              [llvm_v8i16_ty, llvm_v8i16_ty, llvm_i32_ty, llvm_i32_ty],
                                              [IntrNoMem, ImmArg<ArgIndex<2>>]>;

// FFT.AMS.S16.LD.INCP (unified 2.2 API)
// Returns: {v16i8 qu, v8i16 qz, v8i16 qv, ptr}
// Parameters: [qx, qy, qw, ptr, sel2, sat, sar]
def int_riscv_esp_fft_ams_s16_ld_incp : Intrinsic<[llvm_v16i8_ty, llvm_v8i16_ty, llvm_v8i16_ty, llvm_ptr_ty],
                                                     [llvm_v8i16_ty, llvm_v8i16_ty, llvm_v8i16_ty, llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                                                     [IntrReadMem, IntrArgMemOnly, ImmArg<ArgIndex<4>>]>;

// FFT.R2BF.S16.ST.INCP (unified 2.2 API)
// Returns: {v8i16 qz, ptr}
// Parameters: [qx, qy, ptr, sel4, sat]
def int_riscv_esp_fft_r2bf_s16_st_incp : Intrinsic<[llvm_v8i16_ty, llvm_ptr_ty],
                                                      [llvm_v8i16_ty, llvm_v8i16_ty, llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
                                                      [IntrWriteMem, IntrArgMemOnly, ImmArg<ArgIndex<3>>]>;

// FFT.AMS.S16.LD.R32.DECP (unified 2.2 API)
// Returns: {v16i8 qu, v8i16 qz, v8i16 qv, ptr}
// Parameters: [qx, qy, qw, ptr, sel2, sat, sar]
def int_riscv_esp_fft_ams_s16_ld_r32_decp : Intrinsic<[llvm_v16i8_ty, llvm_v8i16_ty, llvm_v8i16_ty, llvm_ptr_ty],
                                                         [llvm_v8i16_ty, llvm_v8i16_ty, llvm_v8i16_ty, llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                                                         [IntrReadMem, IntrArgMemOnly, ImmArg<ArgIndex<4>>]>;

// FFT.AMS.S16.ST.INCP (unified 2.2 API)
// Returns: {v8i16 qz, ptr}
// Parameters: [qx, qy, qw, qu, ptr, ptr, sel2, sat, sar]
def int_riscv_esp_fft_ams_s16_st_incp : Intrinsic<[llvm_v8i16_ty, llvm_ptr_ty],
                                                     [llvm_v8i16_ty, llvm_v8i16_ty, llvm_v8i16_ty, llvm_v16i8_ty, llvm_ptr_ty, llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                                                     [IntrWriteMem, IntrArgMemOnly, ImmArg<ArgIndex<6>>]>;

// FFT.BITREV - FFT bit reverse operation
// Returns: { ptr (updated pointer), v8i16 (result vector) }
// Parameters: [ ptr (source pointer), i32 (FFT_BIT_WIDTH, 32-bit unsigned) ]
def int_riscv_esp_fft_bitrev_m : Intrinsic<[llvm_ptr_ty, llvm_v8i16_ty], 
                                            [llvm_ptr_ty, llvm_i32_ty], 
                                            [IntrNoMem]>;

// FFT.CMUL.S16.LD.XP (unified 2.2 API)
// Returns: {v8i16 qz, v16i8 qu, ptr}
// Parameters: [qx, qy, ptr, ptr, sel8, sat, sar]
def int_riscv_esp_fft_cmul_s16_ld_xp : Intrinsic<[llvm_v8i16_ty, llvm_v16i8_ty, llvm_ptr_ty],
                                                    [llvm_v8i16_ty, llvm_v8i16_ty, llvm_ptr_ty, llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                                                    [IntrReadMem, IntrArgMemOnly, ImmArg<ArgIndex<4>>]>;

// FFT.CMUL.S16.ST.XP (unified 2.2 API)
// Returns: ptr
// Parameters: [qx, qy, qu, ptr, ptr, sel8, upd4, sel4, sat, sar]
def int_riscv_esp_fft_cmul_s16_st_xp : Intrinsic<[llvm_ptr_ty],
                                                    [llvm_v8i16_ty, llvm_v8i16_ty, llvm_v16i8_ty, llvm_ptr_ty, llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                                                    // sel8/upd4/sel4/sat are immediates; final i32 is SAR (not ImmArg).
                                                    [IntrWriteMem, IntrArgMemOnly,
                                                     ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>,
                                                     ImmArg<ArgIndex<7>>, ImmArg<ArgIndex<8>>]>;

// FFT.VST.R32.DECP - Returns struct {i8* ptr}
// Inputs: qu (v16i8), rs1 (ptr), sel2 (i32)
def int_riscv_esp_fft_vst_r32_decp_m : Intrinsic<[llvm_ptr_ty],
                                                   [llvm_v16i8_ty, llvm_ptr_ty, llvm_i32_ty],
                                                   [IntrWriteMem, IntrArgMemOnly]>;

// FFT.AMS.S16.LD.INCP.UAUP (unified 2.2 API)
// Returns: {v16i8 qu, v8i16 qz, v8i16 qv, ptr, v16i8 ua_state_out}
// Parameters: [qx, qy, qw, ptr, sel2, sat, ua_state_in, sar_bytes_in, sar_in]
def int_riscv_esp_fft_ams_s16_ld_incp_uaup : Intrinsic<[llvm_v16i8_ty, llvm_v8i16_ty, llvm_v8i16_ty, llvm_ptr_ty, llvm_v16i8_ty],
                                                          [llvm_v8i16_ty, llvm_v8i16_ty, llvm_v8i16_ty, llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty, llvm_v16i8_ty, llvm_i32_ty, llvm_i32_ty],
                                                          [IntrReadMem, IntrArgMemOnly, ImmArg<ArgIndex<4>>]>;

// VSLD - Vector Shift Left Dynamic (unified 2.1/2.2 API)
def int_riscv_esp_vsld_8 : Intrinsic<[llvm_v16i8_ty],
    [llvm_v16i8_ty, llvm_v16i8_ty, llvm_i32_ty, llvm_i32_ty],
    [IntrNoMem, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;
def int_riscv_esp_vsld_16 : Intrinsic<[llvm_v8i16_ty],
    [llvm_v8i16_ty, llvm_v8i16_ty, llvm_i32_ty, llvm_i32_ty],
    [IntrNoMem, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;
def int_riscv_esp_vsld_32 : Intrinsic<[llvm_v4i32_ty],
    [llvm_v4i32_ty, llvm_v4i32_ty, llvm_i32_ty, llvm_i32_ty],
    [IntrNoMem, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

// VSRD - Vector Shift Right Dynamic (unified 2.1/2.2 API)
def int_riscv_esp_vsrd_8 : Intrinsic<[llvm_v16i8_ty],
    [llvm_v16i8_ty, llvm_v16i8_ty, llvm_i32_ty, llvm_i32_ty],
    [IntrNoMem, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;
def int_riscv_esp_vsrd_16 : Intrinsic<[llvm_v8i16_ty],
    [llvm_v8i16_ty, llvm_v8i16_ty, llvm_i32_ty, llvm_i32_ty],
    [IntrNoMem, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;
def int_riscv_esp_vsrd_32 : Intrinsic<[llvm_v4i32_ty],
    [llvm_v4i32_ty, llvm_v4i32_ty, llvm_i32_ty, llvm_i32_ty],
    [IntrNoMem, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

// MOVX.W.SAR - Write SAR register
// Returns: i32 (SAR value, 32-bit unsigned, only low 6 bits used)
// Parameters: [ i32 value to write (rs1, 32-bit unsigned) ]
def int_riscv_esp_movx_w_sar : Intrinsic<[llvm_i32_ty], 
                                          [llvm_i32_ty], 
                                          [IntrNoMem]>;

// MOVX.W.SAR_BYTES - Write SAR_BYTES register
// Returns: i32 (SAR_BYTES value, 32-bit unsigned, only low 4 bits used)
// Parameters: [ i32 value to write (rs1, 32-bit unsigned) ]
// MOVX.W - Write implicit registers (all return void)
def int_riscv_esp_movx_w_cfg : Intrinsic<[], [llvm_i32_ty],
                                            [IntrNoMem,IntrHasSideEffects]>;

def int_riscv_esp_movx_w_sar_bytes : Intrinsic<[llvm_i32_ty], 
                                                 [llvm_i32_ty], 
                                                 [IntrNoMem]>;

def int_riscv_esp_movx_w_xacc_h : Intrinsic<
    [llvm_i32_ty],                         
    [llvm_i32_ty],                         
    [IntrNoMem]>;                          

// ESP.MOVX.R.XACC_H (_m version) - Read XACC[39:32] with explicit state passing
// Returns: XACC[39:32] as i32 (only low 8 bits valid)
// Parameters: [xacc_h (passthru, i32, only low 8 bits valid)]
def int_riscv_esp_movx_r_xacc_h : Intrinsic<
    [llvm_i32_ty],                         
    [llvm_i32_ty],                         
    [IntrNoMem]>;                          
// ESP.MOVX.R.XACC_L (_m version) - Subregister model: reads XACC[31:0] only
// Returns: XACC[31:0] value as i32 (unsigned int)
// Parameters: [xacc_l (passthru for explicit state passing)]
def int_riscv_esp_movx_r_xacc_l : Intrinsic<
    [llvm_i32_ty],                         
    [llvm_i32_ty],                         
    [IntrNoMem]>;                          

// ESP.MOVX.W.XACC_H (_m version) - Write XACC[39:32] with explicit state passing
// Returns: new XACC[31:0] value as i32 (unsigned int)
// Parameters: [value to write (i32, unsigned int)]
def int_riscv_esp_movx_w_xacc_l : Intrinsic<
    [llvm_i32_ty],                         
    [llvm_i32_ty],                         
    [IntrNoMem]>;                          

// VSR - Vector Shift Right (using SAR register; unified 2.1/2.2 API)
def int_riscv_esp_vsr_s32 : Intrinsic<[llvm_v4i32_ty],
    [llvm_v4i32_ty, llvm_i32_ty, llvm_i32_ty],
    [IntrNoMem, ImmArg<ArgIndex<2>>]>;
def int_riscv_esp_vsr_u32 : Intrinsic<[llvm_v4i32_ty],
    [llvm_v4i32_ty, llvm_i32_ty, llvm_i32_ty],
    [IntrNoMem, ImmArg<ArgIndex<2>>]>;

// VSL.32 - Vector Shift Left (using SAR register; unified 2.1/2.2 API)
def int_riscv_esp_vsl_32 : Intrinsic<[llvm_v4i32_ty],
    [llvm_v4i32_ty, llvm_i32_ty, llvm_i32_ty],
    [IntrNoMem, ImmArg<ArgIndex<1>>]>;

// VABS - Vector Absolute Value
// VABS.8 - 8-bit vector absolute value
// Returns: v16i8 (absolute values of 16 8-bit data segments)
// Parameters: [ v16i8 vector ]
def int_riscv_esp_vabs_8 : Intrinsic<[llvm_v16i8_ty], 
                                       [llvm_v16i8_ty], 
                                       [IntrNoMem]>;

// VADD - Vector Add
// VADD.S8 - Signed 8-bit vector addition
// Returns: v16i8 (16 8-bit signed addition results)
// Parameters: [ v16i8 vector qx, v16i8 vector qy ]
def int_riscv_esp_vadd_s8 : Intrinsic<[llvm_v16i8_ty],
    [llvm_v16i8_ty, llvm_v16i8_ty, llvm_i32_ty],
    [IntrNoMem, ImmArg<ArgIndex<2>>]>;

// VADD.U8 - Unsigned 8-bit vector addition
// Returns: v16i8 (16 8-bit unsigned addition results)
// Parameters: [ v16i8 vector qx, v16i8 vector qy ]
def int_riscv_esp_vadd_u8 : Intrinsic<[llvm_v16i8_ty],
    [llvm_v16i8_ty, llvm_v16i8_ty, llvm_i32_ty],
    [IntrNoMem, ImmArg<ArgIndex<2>>]>;

// VADD.S16 - Signed 16-bit vector addition
// Returns: v8i16 (8 16-bit signed addition results)
// Parameters: [ v8i16 vector qx, v8i16 vector qy ]
def int_riscv_esp_vadd_s16 : Intrinsic<[llvm_v8i16_ty],
    [llvm_v8i16_ty, llvm_v8i16_ty, llvm_i32_ty],
    [IntrNoMem, ImmArg<ArgIndex<2>>]>;

// VADD.U16 - Unsigned 16-bit vector addition
// Returns: v8i16 (8 16-bit unsigned addition results)
// Parameters: [ v8i16 vector qx, v8i16 vector qy, i32 sat ]
def int_riscv_esp_vadd_u16 : Intrinsic<[llvm_v8i16_ty],
    [llvm_v8i16_ty, llvm_v8i16_ty, llvm_i32_ty],
    [IntrNoMem, ImmArg<ArgIndex<2>>]>;

// VSUB - Vector Subtract
// VSUB.S8 - Signed 8-bit vector subtraction
// Returns: v16i8 (16 8-bit signed subtraction results)
// Parameters: [ v16i8 vector qx, v16i8 vector qy ]
def int_riscv_esp_vsub_s8 : Intrinsic<[llvm_v16i8_ty],
    [llvm_v16i8_ty, llvm_v16i8_ty, llvm_i32_ty],
    [IntrNoMem, ImmArg<ArgIndex<2>>]>;

// VSUB.U8 - Unsigned 8-bit vector subtraction
// Returns: v16i8 (16 8-bit unsigned subtraction results)
// Parameters: [ v16i8 vector qx, v16i8 vector qy ]
def int_riscv_esp_vsub_u8 : Intrinsic<[llvm_v16i8_ty],
    [llvm_v16i8_ty, llvm_v16i8_ty, llvm_i32_ty],
    [IntrNoMem, ImmArg<ArgIndex<2>>]>;

// VSUB.S16 - Signed 16-bit vector subtraction
// Returns: v8i16 (8 16-bit signed subtraction results)
// Parameters: [ v8i16 vector qx, v8i16 vector qy ]
def int_riscv_esp_vsub_s16 : Intrinsic<[llvm_v8i16_ty],
    [llvm_v8i16_ty, llvm_v8i16_ty, llvm_i32_ty],
    [IntrNoMem, ImmArg<ArgIndex<2>>]>;

// VABS.16 - Vector Absolute Value - 16-bit
// Returns: v8i16 (absolute values of 8 16-bit data segments)
// Parameters: [ v8i16 vector qy ]
def int_riscv_esp_vabs_16 : ClangBuiltin<"__builtin_riscv_esp_vabs_16">,
	Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty], [IntrNoMem]>;

// VMAX.S16 - Signed 16-bit vector maximum
// Returns: v8i16 (maximum values of 8 16-bit signed elements)
// Parameters: [ v8i16 vector qx, v8i16 vector qy ]
def int_riscv_esp_vmax_s16 : ClangBuiltin<"__builtin_riscv_esp_vmax_s16">,
	Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty, llvm_v8i16_ty], [IntrNoMem]>;

// VCMP - Vector Compare
// VCMP.EQ.S8 - Signed 8-bit vector equal comparison
// Returns: v16i8 (comparison results)
// Parameters: [ v16i8 vector qx, v16i8 vector qy ]
def int_riscv_esp_vcmp_eq_s8 : Intrinsic<[llvm_v16i8_ty], 
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
def int_riscv_esp_vcmp_gt_s8 : Intrinsic<[llvm_v16i8_ty], 
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
def int_riscv_esp_vcmp_lt_s8 : Intrinsic<[llvm_v16i8_ty], 
                                            [llvm_v16i8_ty, llvm_v16i8_ty], 
                                            [IntrNoMem]>;

// VCMP.LT.U8 - Unsigned 8-bit vector less than comparison
// Returns: v16i8 (comparison results)
// Parameters: [ v16i8 vector qx, v16i8 vector qy ]
def int_riscv_esp_vcmp_lt_u8 : Intrinsic<[llvm_v16i8_ty], 
                                            [llvm_v16i8_ty, llvm_v16i8_ty], 
                                            [IntrNoMem]>;

// VMAX - Vector Maximum
// VMAX.S8 - Signed 8-bit vector maximum
// Returns: v16i8 (maximum values)
// Parameters: [ v16i8 vector qx, v16i8 vector qy ]
def int_riscv_esp_vmax_s8 : Intrinsic<[llvm_v16i8_ty], 
                                         [llvm_v16i8_ty, llvm_v16i8_ty], 
                                         [IntrNoMem]>;

// VMAX.U8 - Unsigned 8-bit vector maximum
// Returns: v16i8 (maximum values)
// Parameters: [ v16i8 vector qx, v16i8 vector qy ]
def int_riscv_esp_vmax_u8 : Intrinsic<[llvm_v16i8_ty], 
                                         [llvm_v16i8_ty, llvm_v16i8_ty], 
                                         [IntrNoMem]>;

// ESP.VMAX.U16 (_m version)
// Returns: v8i16 (maximum values of 8 16-bit unsigned elements)
// Parameters: [ v8i16 vector qx, v8i16 vector qy ]
def int_riscv_esp_vmax_u16 : ClangBuiltin<"__builtin_riscv_esp_vmax_u16">,
	Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty, llvm_v8i16_ty], [IntrNoMem]>;

// ESP.VMAX.U32 (_m version)
// Returns: v4i32 (maximum values of 4 32-bit unsigned elements)
// Parameters: [ v4i32 vector qx, v4i32 vector qy ]
def int_riscv_esp_vmax_u32 : ClangBuiltin<"__builtin_riscv_esp_vmax_u32">,
	Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty], [IntrNoMem]>;

// VMIN - Vector Minimum
// VMIN.S8 - Signed 8-bit vector minimum
// Returns: v16i8 (minimum values)
// Parameters: [ v16i8 vector qx, v16i8 vector qy ]
def int_riscv_esp_vmin_s8 : Intrinsic<[llvm_v16i8_ty], 
                                         [llvm_v16i8_ty, llvm_v16i8_ty], 
                                         [IntrNoMem]>;

// ESP.VMIN.S16 (_m version)
// Returns: v8i16 (minimum values of 8 16-bit signed elements)
// Parameters: [ v8i16 vector qx, v8i16 vector qy ]
def int_riscv_esp_vmin_s16 : ClangBuiltin<"__builtin_riscv_esp_vmin_s16">,
	Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty, llvm_v8i16_ty], [IntrNoMem]>;

// ESP.VMIN.S32 (_m version)
// Returns: v4i32 (minimum values of 4 32-bit signed elements)
// Parameters: [ v4i32 vector qx, v4i32 vector qy ]
def int_riscv_esp_vmin_s32 : ClangBuiltin<"__builtin_riscv_esp_vmin_s32">,
	Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty], [IntrNoMem]>;

// VMIN.U8 - Unsigned 8-bit vector minimum
// Returns: v16i8 (minimum values)
// Parameters: [ v16i8 vector qx, v16i8 vector qy ]
def int_riscv_esp_vmin_u8 : Intrinsic<[llvm_v16i8_ty], 
                                         [llvm_v16i8_ty, llvm_v16i8_ty], 
                                         [IntrNoMem]>;

// ESP.VMIN.U16 (_m version)
// Returns: v8i16 (minimum values of 8 16-bit unsigned elements)
// Parameters: [ v8i16 vector qx, v8i16 vector qy ]
def int_riscv_esp_vmin_u16 : ClangBuiltin<"__builtin_riscv_esp_vmin_u16">,
	Intrinsic<[llvm_v8i16_ty], [llvm_v8i16_ty, llvm_v8i16_ty], [IntrNoMem]>;

// ESP.VMIN.U32 (_m version)
// Returns: v4i32 (minimum values of 4 32-bit unsigned elements)
// Parameters: [ v4i32 vector qx, v4i32 vector qy ]
def int_riscv_esp_vmin_u32 : ClangBuiltin<"__builtin_riscv_esp_vmin_u32">,
	Intrinsic<[llvm_v4i32_ty], [llvm_v4i32_ty, llvm_v4i32_ty], [IntrNoMem]>;

// VMUL - Vector Multiplication (unified 2.2 API)
// VMUL.S8 - Signed 8-bit vector multiplication
// Returns: v16i8
// Parameters: [qx, qy, sat, rm, sar]
def int_riscv_esp_vmul_s8 : Intrinsic<[llvm_v16i8_ty],
                                         [llvm_v16i8_ty, llvm_v16i8_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                                         [IntrNoMem, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

// VMUL.U8 - Unsigned 8-bit vector multiplication
// Returns: v16i8
// Parameters: [qx, qy, sat, rm, sar]
def int_riscv_esp_vmul_u8 : Intrinsic<[llvm_v16i8_ty],
                                         [llvm_v16i8_ty, llvm_v16i8_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                                         [IntrNoMem, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;
// VMUL.S16 - Signed 16-bit vector multiplication
// Returns: v8i16
// Parameters: [qx, qy, sat, rm, sar]
def int_riscv_esp_vmul_s16 : Intrinsic<[llvm_v8i16_ty],
                                         [llvm_v8i16_ty, llvm_v8i16_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                                         [IntrNoMem, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

// ESP.VMUL.S16.LD.INCP (unified 2.2 API)
// Returns: {qv, qu, ptr}
// Parameters: [qx, qy, ptr, sat, rm, sar]
def int_riscv_esp_vmul_s16_ld_incp : Intrinsic<
	[llvm_v8i16_ty, llvm_v16i8_ty, llvm_ptr_ty],
	[llvm_v8i16_ty, llvm_v8i16_ty, llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
	[IntrReadMem, IntrArgMemOnly, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]
>;

// ESP.VMUL.S16.S8XS8 (unified 2.2 API)
// Returns: {qz_v8i16, qv_v8i16}
// Parameters: [qx_v16i8, qy_v16i8, sat, rm, sar]
def int_riscv_esp_vmul_s16_s8xs8 : Intrinsic<[llvm_v8i16_ty, llvm_v8i16_ty],
	[llvm_v16i8_ty, llvm_v16i8_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
	[IntrNoMem, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

// ESP.VMUL.S16.ST.INCP (unified 2.2 API)
// Returns: {qv, ptr}
// Parameters: [qx, qy, qu, ptr, qv_in, sat, rm, sar]
def int_riscv_esp_vmul_s16_st_incp : Intrinsic<
	[llvm_v8i16_ty, llvm_ptr_ty],
	[llvm_v8i16_ty, llvm_v8i16_ty, llvm_v16i8_ty, llvm_ptr_ty, llvm_v8i16_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
	[IntrWriteMem, IntrArgMemOnly, ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>]
>;

// ESP.VMUL.S32.S16XS16 (unified 2.2 API)
// Returns: {qz_v4i32, qv_v4i32}
// Parameters: [qx_v8i16, qy_v8i16, sat, rm, sar]
def int_riscv_esp_vmul_s32_s16xs16 : Intrinsic<[llvm_v4i32_ty, llvm_v4i32_ty],
	[llvm_v8i16_ty, llvm_v8i16_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
	[IntrNoMem, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

// ESP.VMUL.U16 (unified 2.2 API)
// Returns: v8i16
// Parameters: [qx, qy, sat, rm, sar]
def int_riscv_esp_vmul_u16 : Intrinsic<[llvm_v8i16_ty],
                                         [llvm_v8i16_ty, llvm_v8i16_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                                         [IntrNoMem, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

// Bitwise Operations
// ANDQ - Vector bitwise AND
// Returns: v16i8 (bitwise AND results)
// Parameters: [ v16i8 vector qx, v16i8 vector qy ]
def int_riscv_esp_andq : Intrinsic<[llvm_v16i8_ty], 
                                      [llvm_v16i8_ty, llvm_v16i8_ty], 
                                      [IntrNoMem]>;

// ORQ - Vector bitwise OR
// Returns: v16i8 (bitwise OR results)
// Parameters: [ v16i8 vector qx, v16i8 vector qy ]
def int_riscv_esp_orq : Intrinsic<[llvm_v16i8_ty], 
                                     [llvm_v16i8_ty, llvm_v16i8_ty], 
                                     [IntrNoMem]>;

// XORQ - Vector bitwise XOR
// Returns: v16i8 (bitwise XOR results)
// Parameters: [ v16i8 vector qx, v16i8 vector qy ]
def int_riscv_esp_xorq : Intrinsic<[llvm_v16i8_ty], 
                                      [llvm_v16i8_ty, llvm_v16i8_ty], 
                                      [IntrNoMem]>;

// NOTQ - Vector bitwise NOT
// Returns: v16i8 (bitwise NOT results)
// Parameters: [ v16i8 vector qx ]
def int_riscv_esp_notq : Intrinsic<[llvm_v16i8_ty], 
                                      [llvm_v16i8_ty], 
                                      [IntrNoMem]>;

// ZERO.Q - Zero vector register
// Returns: v16i8 (zero vector)
// Parameters: []
def int_riscv_esp_zero_q : Intrinsic<[llvm_v16i8_ty], 
                                        [], 
                                        [IntrNoMem]>;

// ZERO.QACC - Zero QACC accumulator
// Returns: {v16i8, v16i8, v16i8, v16i8} (4x128-bit QACC, all zero)
// Parameters: []
def int_riscv_esp_zero_qacc : Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty], 
                                           [], 
                                           [IntrNoMem]>;

// MOV.S16.QACC - Sign extend 8x16-bit to 64-bit, store to QACC_H and QACC_L
// Returns: {v16i8, v16i8, v16i8, v16i8} (4x128-bit QACC)
// Parameters: [v8i16] (input vector)
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_mov_s16_qacc : Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],
                                              [llvm_v8i16_ty],
                                              [IntrNoMem]>;

// MOV.S8.QACC - Sign extend 16x8-bit to 32-bit, store to QACC_H and QACC_L
// Returns: {v16i8, v16i8, v16i8, v16i8} (4x128-bit QACC)
// Parameters: [v16i8] (input vector)
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_mov_s8_qacc : Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],
                                             [llvm_v16i8_ty],
                                             [IntrNoMem]>;

// MOV.U16.QACC - Zero extend 8x16-bit to 64-bit, store to QACC_H and QACC_L
// Returns: {v16i8, v16i8, v16i8, v16i8} (4x128-bit QACC)
// Parameters: [v8i16] (input vector)
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_mov_u16_qacc : Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],
                                              [llvm_v8i16_ty],
                                              [IntrNoMem]>;

// MOV.U8.QACC - Zero extend 16x8-bit to 32-bit, store to QACC_H and QACC_L
// Returns: {v16i8, v16i8, v16i8, v16i8} (4x128-bit QACC)
// Parameters: [v16i8] (input vector)
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_mov_u8_qacc : Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],
                                             [llvm_v16i8_ty],
                                             [IntrNoMem]>;

// ZERO.XACC - Zero XACC accumulator
// Returns: {i32, i32} (XACC set to zero {i32 low=0, i32 high=0})
// Parameters: []
def int_riscv_esp_zero_xacc : Intrinsic<[llvm_i32_ty, llvm_i32_ty], 
                                          [], 
                                          [IntrNoMem]>;

// ESP.ST.S.XACC.IP (_m version) - Mixed model: XACC as {i32 low, i32 high}
// Returns: updated pointer + XACC unchanged {i32 low, i32 high}
// Parameters: [xacc_low_in (i32), xacc_high_in (i32, only low 8 bits valid), ptr, offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
// Note: XACC value is unchanged after store, so output XACC equals input XACC
def int_riscv_esp_st_s_xacc_ip : Intrinsic<
    [llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty], 
    [llvm_i32_ty,                             
     llvm_i32_ty,                             
     llvm_ptr_ty,                             
     llvm_i32_ty],                            
    [IntrWriteMem, IntrArgMemOnly]>;          

// ESP.ST.U.XACC.IP (_m version) - Store XACC zero-extended to 64-bit memory
// Returns: updated pointer + XACC unchanged {i32 low, i32 high}
// Parameters: [xacc_low_in (i32), xacc_high_in (i32, only low 8 bits valid), ptr, offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_st_u_xacc_ip : Intrinsic<
    [llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty], 
    [llvm_i32_ty,                             
     llvm_i32_ty,                             
     llvm_ptr_ty,                             
     llvm_i32_ty],                            
    [IntrWriteMem, IntrArgMemOnly]>;          

// ESP.VCMULAS.S16.QACC.L (_m version)
// Returns: {v16i8, v16i8} - 2x128-bit QACC_L directly
// Parameters: [v16i8, v16i8 (2x128-bit QACC_L passthru), qx, qy]
// Migration: Explicit state passing (no IntrHasSideEffects)
// QACC_L is split into 2x128-bit: QACC_L_LOW (v0) and QACC_L_HIGH (v1)
// Note: No ClangBuiltin - builtin signature (void with 6 params) doesn't match intrinsic signature ({v16i8, v16i8} with 4 params)
def int_riscv_esp_vcmulas_s16_qacc_l :
	Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty],
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v8i16_ty, llvm_v8i16_ty, llvm_i32_ty],
              [IntrNoMem, ImmArg<ArgIndex<4>>]>;                      

// ESP.VCMULAS.S16.QACC.H (_m version)
// Returns: {v16i8, v16i8} - 2x128-bit QACC_H directly
// Parameters: [v16i8, v16i8 (2x128-bit QACC_H passthru), qx, qy]
// Migration: Explicit state passing (no IntrHasSideEffects)
// QACC_H is split into 2x128-bit: QACC_H_LOW (v2) and QACC_H_HIGH (v3)
// Note: No ClangBuiltin - builtin signature (void with 6 params) doesn't match intrinsic signature ({v16i8, v16i8} with 4 params)
def int_riscv_esp_vcmulas_s16_qacc_h :
	Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty],
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v8i16_ty, llvm_v8i16_ty, llvm_i32_ty],
              [IntrNoMem, ImmArg<ArgIndex<4>>]>;                      

// ESP.VCMULAS.S16.QACC.H.LD.IP (_m version)
// Returns: updated pointer + qu vector + new QACC_H (2x128-bit)
// Parameters: [current QACC_H (2x128-bit passthru), qx, qy, pointer, immediate offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_vcmulas_s16_qacc_h_ld_ip : Intrinsic<
    [llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty], 
    [llvm_v16i8_ty, llvm_v16i8_ty,                              
     llvm_v8i16_ty,                                              
     llvm_v8i16_ty,                                              
     llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
    [IntrReadMem, IntrArgMemOnly, ImmArg<ArgIndex<6>>]>;                              

// ESP.VCMULAS.S16.QACC.H.LD.XP (_m version)
// Returns: updated pointer + qu vector + new QACC_H (2x128-bit)
// Parameters: [current QACC_H (2x128-bit passthru), qx, qy, pointer, register offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_vcmulas_s16_qacc_h_ld_xp : Intrinsic<
    [llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty], 
    [llvm_v16i8_ty, llvm_v16i8_ty,                              
     llvm_v8i16_ty,                                              
     llvm_v8i16_ty,                                              
     llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
    [IntrReadMem, IntrArgMemOnly, ImmArg<ArgIndex<6>>]>;                              

// ESP.VCMULAS.S16.QACC.L.LD.IP (_m version)
// Returns: updated pointer + qu vector + new QACC_L (2x128-bit)
// Parameters: [current QACC_L (2x128-bit passthru), qx, qy, pointer, immediate offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_vcmulas_s16_qacc_l_ld_ip : Intrinsic<
    [llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty], 
    [llvm_v16i8_ty, llvm_v16i8_ty,                              
     llvm_v8i16_ty,                                              
     llvm_v8i16_ty,                                              
     llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
    [IntrReadMem, IntrArgMemOnly, ImmArg<ArgIndex<6>>]>;                              

// ESP.VCMULAS.S16.QACC.L.LD.XP (_m version)
// Returns: updated pointer + qu vector + new QACC_L (2x128-bit)
// Parameters: [current QACC_L (2x128-bit passthru), qx, qy, pointer, register offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_vcmulas_s16_qacc_l_ld_xp : Intrinsic<
    [llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty], 
    [llvm_v16i8_ty, llvm_v16i8_ty,                              
     llvm_v8i16_ty,                                              
     llvm_v8i16_ty,                                              
     llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
    [IntrReadMem, IntrArgMemOnly, ImmArg<ArgIndex<6>>]>;                              

// ESP.VCMULAS.S8.QACC.H (_m version)
// Returns: {v16i8, v16i8} - 2x128-bit QACC_H directly
// Parameters: [v16i8, v16i8 (2x128-bit QACC_H passthru), qx, qy]
// Migration: Explicit state passing (no IntrHasSideEffects)
// QACC_H is split into 2x128-bit: QACC_H_LOW (v2) and QACC_H_HIGH (v3)
// Note: No ClangBuiltin - builtin signature (void with 6 params) doesn't match intrinsic signature ({v16i8, v16i8} with 4 params)
def int_riscv_esp_vcmulas_s8_qacc_h :
	Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty],
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_i32_ty],
              [IntrNoMem, ImmArg<ArgIndex<4>>]>;                      

// ESP.VCMULAS.S8.QACC.H.LD.IP (_m version)
// Returns: updated pointer + qu vector + new QACC_H (2x128-bit)
// Parameters: [current QACC_H (2x128-bit passthru), qx, qy, pointer, immediate offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_vcmulas_s8_qacc_h_ld_ip : Intrinsic<
    [llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty], 
    [llvm_v16i8_ty, llvm_v16i8_ty,                              
     llvm_v16i8_ty,                                              
     llvm_v16i8_ty,                                              
     llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
    [IntrReadMem, IntrArgMemOnly, ImmArg<ArgIndex<6>>]>;                              

// ESP.VCMULAS.S8.QACC.H.LD.XP (_m version)
// Returns: updated pointer + qu vector + new QACC_H (2x128-bit)
// Parameters: [current QACC_H (2x128-bit passthru), qx, qy, pointer, register offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_vcmulas_s8_qacc_h_ld_xp : Intrinsic<
    [llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty], 
    [llvm_v16i8_ty, llvm_v16i8_ty,                              
     llvm_v16i8_ty,                                              
     llvm_v16i8_ty,                                              
     llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
    [IntrReadMem, IntrArgMemOnly, ImmArg<ArgIndex<6>>]>;                              

// ESP.VCMULAS.S8.QACC.L (_m version)
// Returns: {v16i8, v16i8} - 2x128-bit QACC_L directly
// Parameters: [v16i8, v16i8 (2x128-bit QACC_L passthru), qx, qy]
// Migration: Explicit state passing (no IntrHasSideEffects)
// QACC_L is split into 2x128-bit: QACC_L_LOW (v0) and QACC_L_HIGH (v1)
// Note: No ClangBuiltin - builtin signature (void with 6 params) doesn't match intrinsic signature ({v16i8, v16i8} with 4 params)
def int_riscv_esp_vcmulas_s8_qacc_l :
	Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty],
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_i32_ty],
              [IntrNoMem, ImmArg<ArgIndex<4>>]>;                      

// ESP.VCMULAS.S8.QACC.L.LD.IP (_m version)
// Returns: updated pointer + qu vector + new QACC_L (2x128-bit)
// Parameters: [current QACC_L (2x128-bit passthru), qx, qy, pointer, immediate offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_vcmulas_s8_qacc_l_ld_ip : Intrinsic<
    [llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty], 
    [llvm_v16i8_ty, llvm_v16i8_ty,                              
     llvm_v16i8_ty,                                              
     llvm_v16i8_ty,                                              
     llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
    [IntrReadMem, IntrArgMemOnly, ImmArg<ArgIndex<6>>]>;                              

// ESP.VCMULAS.S8.QACC.L.LD.XP (_m version)
// Returns: updated pointer + qu vector + new QACC_L (2x128-bit)
// Parameters: [current QACC_L (2x128-bit passthru), qx, qy, pointer, register offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_vcmulas_s8_qacc_l_ld_xp : Intrinsic<
    [llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty], 
    [llvm_v16i8_ty, llvm_v16i8_ty,                              
     llvm_v16i8_ty,                                              
     llvm_v16i8_ty,                                              
     llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
    [IntrReadMem, IntrArgMemOnly, ImmArg<ArgIndex<6>>]>;                              

// ESP.LD.QACC.H.H.128.IP (_m version)
// Load 128-bit data to QACC_H[255:128]
// Subregister model: returns loaded 128-bit data (v16i8) + updated pointer
// Parameters: [pointer, immediate offset]
// Note: Hardware loads 128-bit data to QACC_H[255:128], QACC_H[127:0] remains unchanged
def int_riscv_esp_ld_qacc_h_h_128_ip : Intrinsic<
    [llvm_v16i8_ty, llvm_ptr_ty],       
    [llvm_ptr_ty,                        
     llvm_i32_ty],                       
    [IntrReadMem, IntrArgMemOnly]>;      

// ESP.LD.QACC.H.L.128.IP (_m version)
// Load 128-bit data to QACC_H[127:0]
// Subregister model: returns loaded 128-bit data (v16i8) + updated pointer
// Parameters: [pointer, immediate offset]
// Note: Hardware loads 128-bit data to QACC_H[127:0], QACC_H[255:128] remains unchanged
def int_riscv_esp_ld_qacc_h_l_128_ip : Intrinsic<
    [llvm_v16i8_ty, llvm_ptr_ty],       
    [llvm_ptr_ty,                        
     llvm_i32_ty],                       
    [IntrReadMem, IntrArgMemOnly]>;      

// ESP.LD.QACC.L.H.128.IP (_m version)
// Load 128-bit data to QACC_L[255:128]
// Subregister model: returns loaded 128-bit data (v16i8) + updated pointer
// Parameters: [pointer, immediate offset]
// Note: Hardware loads 128-bit data to QACC_L[255:128], QACC_L[127:0] remains unchanged
def int_riscv_esp_ld_qacc_l_h_128_ip : Intrinsic<
    [llvm_v16i8_ty, llvm_ptr_ty],       
    [llvm_ptr_ty,                        
     llvm_i32_ty],                       
    [IntrReadMem, IntrArgMemOnly]>;      

// ESP.LD.QACC.L.L.128.IP (_m version)
// Load 128-bit data to QACC_L[127:0]
// Subregister model: returns loaded 128-bit data (v16i8) + updated pointer
// Parameters: [pointer, immediate offset]
// Note: Hardware loads 128-bit data to QACC_L[127:0], QACC_L[255:128] remains unchanged
def int_riscv_esp_ld_qacc_l_l_128_ip : Intrinsic<
    [llvm_v16i8_ty, llvm_ptr_ty],       
    [llvm_ptr_ty,                        
     llvm_i32_ty],                       
    [IntrReadMem, IntrArgMemOnly]>;      

// ESP.ST.QACC.H.H.128.IP (_m version)
// Store QACC_H[255:128] to memory
// Returns: updated pointer
// Parameters: [QACC_H[255:128] (128-bit), pointer, immediate offset]
// Accept v16i8 (128-bit) to match hardware operation and simplify usage.
// First principle: directly accept 128-bit value, matching hardware behavior.
def int_riscv_esp_st_qacc_h_h_128_ip : Intrinsic<
    [llvm_ptr_ty],                       
    [llvm_v16i8_ty,                      
     llvm_ptr_ty,                        
     llvm_i32_ty],                       
    [IntrWriteMem, IntrArgMemOnly]>;     

// ESP.ST.QACC.H.L.128.IP (_m version)
// Store QACC_H[127:0] to memory
// Returns: updated pointer
// Parameters: [QACC_H[127:0] (128-bit), pointer, immediate offset]
// Accept v16i8 (128-bit) to match hardware operation and simplify usage.
// First principle: directly accept 128-bit value, matching hardware behavior.
def int_riscv_esp_st_qacc_h_l_128_ip : Intrinsic<
    [llvm_ptr_ty],                       
    [llvm_v16i8_ty,                      
     llvm_ptr_ty,                        
     llvm_i32_ty],                       
    [IntrWriteMem, IntrArgMemOnly]>;     

// ESP.ST.QACC.L.H.128.IP (_m version)
// Store QACC_L[255:128] to memory
// Returns: updated pointer
// Parameters: [QACC_L[255:128] (128-bit), pointer, immediate offset]
// Hardware directly reads QACC_L[255:128] from full QACC_L register.
// Accept v16i8 (128-bit) to match hardware operation and simplify usage.
// First principle: directly accept 128-bit value, matching hardware behavior.
def int_riscv_esp_st_qacc_l_h_128_ip : Intrinsic<
    [llvm_ptr_ty],                       
    [llvm_v16i8_ty,                      
     llvm_ptr_ty,                        
     llvm_i32_ty],                       
    [IntrWriteMem, IntrArgMemOnly]>;     

// ESP.ST.QACC.L.L.128.IP (_m version)
// Store QACC_L[127:0] to memory
// Returns: updated pointer
// Parameters: [QACC_L[127:0] (128-bit), pointer, immediate offset]
// Hardware directly reads QACC_L[127:0] from full QACC_L register.
// Accept v16i8 (128-bit) to match hardware operation and simplify usage.
// First principle: directly accept 128-bit value, matching hardware behavior.
def int_riscv_esp_st_qacc_l_l_128_ip : Intrinsic<
    [llvm_ptr_ty],                       
    [llvm_v16i8_ty,                      
     llvm_ptr_ty,                        
     llvm_i32_ty],                       
    [IntrWriteMem, IntrArgMemOnly]>;     

// ESP.LDXQ.32 (_m version)
// Load 32-bit data from memory address calculated using qw vector and sel8, store to qu vector using sel4
// Returns: qu vector
// Parameters: [pointer, qw vector, sel4 immediate, sel8 immediate]
// Note: Not marked as ClangBuiltin to avoid generic intrinsic path
// Builtin has 4 parameters (ptr + vector + 2 immediates + output pointer) but intrinsic has 4 parameters
def int_riscv_esp_ldxq_32 : Intrinsic<[llvm_v4i32_ty], 
                                         [llvm_ptr_ty, llvm_v8i16_ty, llvm_i32_ty, llvm_i32_ty], 
                                         [IntrReadMem, IntrArgMemOnly]>;

// ESP.LDQA.S16.128.IP (_m version)
// Load 128-bit data, sign-extend 8x16-bit to 64-bit, store to QACC_H and QACC_L
// Returns: updated pointer + 4x128-bit QACC (QACC_L_LOW, QACC_L_HIGH, QACC_H_LOW, QACC_H_HIGH)
// Parameters: [current QACC (passthru, v64i8), pointer, immediate offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_ldqa_s16_128_ip : Intrinsic<
    [llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty], 
    [llvm_v64i8_ty,                      
     llvm_ptr_ty,                        
     llvm_i32_ty],                       
    [IntrReadMem, IntrArgMemOnly]>;      

// ESP.LDQA.S16.128.XP (_m version)
// Load 128-bit data, sign-extend 8x16-bit to 64-bit, store to QACC_H and QACC_L
// Returns: updated pointer + new QACC (512-bit: {QACC_H, QACC_L})
// Parameters: [current QACC (passthru), pointer, register offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_ldqa_s16_128_xp : Intrinsic<
    [llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty], 
    [llvm_v64i8_ty,                      
     llvm_ptr_ty,                        
     llvm_i32_ty],                       
    [IntrReadMem, IntrArgMemOnly]>;      

// ESP.LDQA.S8.128.IP (_m version)
// Load 128-bit data, sign-extend 16x8-bit to 32-bit, store to QACC_H and QACC_L
// Returns: updated pointer + new QACC (512-bit: {QACC_H, QACC_L})
// Parameters: [current QACC (passthru), pointer, immediate offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_ldqa_s8_128_ip : Intrinsic<
    [llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty], 
    [llvm_v64i8_ty,                      
     llvm_ptr_ty,                        
     llvm_i32_ty],                       
    [IntrReadMem, IntrArgMemOnly]>;      

// ESP.LDQA.S8.128.XP (_m version)
// Load 128-bit data, sign-extend 16x8-bit to 32-bit, store to QACC_H and QACC_L
// Returns: updated pointer + new QACC (512-bit: {QACC_H, QACC_L})
// Parameters: [current QACC (passthru), pointer, register offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_ldqa_s8_128_xp : Intrinsic<
    [llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty], 
    [llvm_v64i8_ty,                      
     llvm_ptr_ty,                        
     llvm_i32_ty],                       
    [IntrReadMem, IntrArgMemOnly]>;      

// ESP.LDQA.U16.128.IP (_m version)
// Load 128-bit data, zero-extend 8x16-bit to 64-bit, store to QACC_H and QACC_L
// Returns: updated pointer + new QACC (512-bit: {QACC_H, QACC_L})
// Parameters: [current QACC (passthru), pointer, immediate offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_ldqa_u16_128_ip : Intrinsic<
    [llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty], 
    [llvm_v64i8_ty,                      
     llvm_ptr_ty,                        
     llvm_i32_ty],                       
    [IntrReadMem, IntrArgMemOnly]>;      

// ESP.LDQA.U16.128.XP (_m version)
// Load 128-bit data, zero-extend 8x16-bit to 64-bit, store to QACC_H and QACC_L
// Returns: updated pointer + new QACC (512-bit: {QACC_H, QACC_L})
// Parameters: [current QACC (passthru), pointer, register offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_ldqa_u16_128_xp : Intrinsic<
    [llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty], 
    [llvm_v64i8_ty,                      
     llvm_ptr_ty,                        
     llvm_i32_ty],                       
    [IntrReadMem, IntrArgMemOnly]>;      

// ESP.LDQA.U8.128.IP (_m version)
// Load 128-bit data, zero-extend 16x8-bit to 32-bit, store to QACC_H and QACC_L
// Returns: updated pointer + new QACC (512-bit: {QACC_H, QACC_L})
// Parameters: [current QACC (passthru), pointer, immediate offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_ldqa_u8_128_ip : Intrinsic<
    [llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty], 
    [llvm_v64i8_ty,                      
     llvm_ptr_ty,                        
     llvm_i32_ty],                       
    [IntrReadMem, IntrArgMemOnly]>;      

// ESP.LDQA.U8.128.XP (_m version)
// Load 128-bit data, zero-extend 16x8-bit to 32-bit, store to QACC_H and QACC_L
// Returns: updated pointer + new QACC (512-bit: {QACC_H, QACC_L})
// Parameters: [current QACC (passthru), pointer, register offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_ldqa_u8_128_xp : Intrinsic<
    [llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty], 
    [llvm_v64i8_ty,                      
     llvm_ptr_ty,                        
     llvm_i32_ty],                       
    [IntrReadMem, IntrArgMemOnly]>;      





// ESP.VSMULAS.S8.QACC.LD.INCP (_m version)
// Returns: (ptr, qu, v0, v1, v2, v3) - ptr + qu + 4x128-bit QACC (6 values)
// Parameters: (v0, v1, v2, v3, qx, qy, ptr, sel16) - 4x128-bit QACC passthru + qx + qy + ptr + sel16
def int_riscv_esp_vsmulas_s8_qacc_ld_incp :
	Intrinsic<[llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty], 
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,              
               llvm_v16i8_ty,                                                              
               llvm_v16i8_ty,                                                              
               llvm_ptr_ty,                                
               llvm_i32_ty, llvm_i32_ty],                               
              [IntrReadMem, IntrArgMemOnly, ImmArg<ArgIndex<8>>]>;

// ESP.VSMULAS.U16.QACC (_m version) - Multiply-accumulate to QACC_H and QACC_L
// Returns: {v16i8, v16i8, v16i8, v16i8} - 4x128-bit QACC directly
// Parameters: v16i8, v16i8, v16i8, v16i8 (4x128-bit QACC passthru), v8i16, v8i16 (qx, qy), i32 (sel16)
// Note: No ClangBuiltin - builtin signature doesn't match intrinsic signature (builtin uses output pointers)
def int_riscv_esp_vsmulas_u16_qacc :
	Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty], 
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,  
               llvm_v8i16_ty,                                                 
               llvm_v8i16_ty,                                                 
               llvm_i32_ty, llvm_i32_ty],                                                  
              [IntrNoMem, ImmArg<ArgIndex<7>>]>;                      

// ESP.VSMULAS.U16.QACC.LD.INCP (_m version)
// Returns: (ptr, qu, v0, v1, v2, v3) - ptr + qu + 4x128-bit QACC (6 values)
// Parameters: (v0, v1, v2, v3, qx, qy, ptr, sel16) - 4x128-bit QACC passthru + qx + qy + ptr + sel16
def int_riscv_esp_vsmulas_u16_qacc_ld_incp :
	Intrinsic<[llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty], 
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,              
               llvm_v8i16_ty,                                                              
               llvm_v8i16_ty,                                                              
               llvm_ptr_ty,                                
               llvm_i32_ty, llvm_i32_ty],                               
              [IntrReadMem, IntrArgMemOnly, ImmArg<ArgIndex<8>>]>;

// ESP.VSMULAS.U8.QACC (_m version) - Multiply-accumulate to QACC_H and QACC_L
// Returns: {v16i8, v16i8, v16i8, v16i8} - 4x128-bit QACC directly
// Parameters: v16i8, v16i8, v16i8, v16i8 (4x128-bit QACC passthru), v16i8, v16i8 (qx, qy), i32 (sel16)
// Note: No ClangBuiltin - builtin signature doesn't match intrinsic signature (builtin uses output pointers)
def int_riscv_esp_vsmulas_u8_qacc :
	Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty], 
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,  
               llvm_v16i8_ty,                                                 
               llvm_v16i8_ty,                                                 
               llvm_i32_ty, llvm_i32_ty],                                                  
              [IntrNoMem, ImmArg<ArgIndex<7>>]>;                      

// ESP.VSMULAS.U8.QACC.LD.INCP (_m version)
// Returns: (ptr, qu, v0, v1, v2, v3) - ptr + qu + 4x128-bit QACC (6 values)
// Parameters: (v0, v1, v2, v3, qx, qy, ptr, sel16) - 4x128-bit QACC passthru + qx + qy + ptr + sel16
def int_riscv_esp_vsmulas_u8_qacc_ld_incp :
	Intrinsic<[llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty], 
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,              
               llvm_v16i8_ty,                                                              
               llvm_v16i8_ty,                                                              
               llvm_ptr_ty,                                
               llvm_i32_ty, llvm_i32_ty],                               
              [IntrReadMem, IntrArgMemOnly, ImmArg<ArgIndex<8>>]>;

// CMUL - Complex Multiplication
// CMUL.S16 - Signed 16-bit complex multiplication (unified 2.2 API)
// Returns: v8i16
// Parameters: [qz_in, qx, qy, sel4, sat, rm, sar]
def int_riscv_esp_cmul_s16 : Intrinsic<[llvm_v8i16_ty], 
                                          [llvm_v8i16_ty, llvm_v8i16_ty, llvm_v8i16_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], 
                                          [IntrNoMem, ImmArg<ArgIndex<3>>]>;

// CMUL.S16.LD.INCP (unified 2.2 API)
// Returns: {v8i16 qz, v16i8 qu, ptr}
// Parameters: [qz_in, qx, qy, ptr, sel4, sat, rm, sar]
def int_riscv_esp_cmul_s16_ld_incp : Intrinsic<[llvm_v8i16_ty, llvm_v16i8_ty, llvm_ptr_ty], 
                                                   [llvm_v8i16_ty, llvm_v8i16_ty, llvm_v8i16_ty, llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], 
                                                   [IntrReadMem, IntrArgMemOnly, ImmArg<ArgIndex<4>>]>;

// CMUL.S16.ST.INCP (unified 2.2 API)
// Returns: {v8i16 qz, ptr}
// Parameters: [qz_in, qx, qy, qu, ptr, sel4, sat, rm, sar]
def int_riscv_esp_cmul_s16_st_incp : Intrinsic<[llvm_v8i16_ty, llvm_ptr_ty], 
                                                   [llvm_v8i16_ty, llvm_v8i16_ty, llvm_v8i16_ty, llvm_v16i8_ty, llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], 
                                                   [IntrWriteMem, IntrArgMemOnly, ImmArg<ArgIndex<5>>]>;

// CMUL.S8 - Signed 8-bit complex multiplication (unified 2.2 API)
// Returns: v16i8
// Parameters: [qz_in, qx, qy, sel4, sat, rm, sar]
def int_riscv_esp_cmul_s8 : Intrinsic<[llvm_v16i8_ty], 
                                         [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], 
                                         [IntrNoMem, ImmArg<ArgIndex<3>>]>;

// CMUL.S8.LD.INCP (unified 2.2 API)
// Returns: {v16i8 qz, v16i8 qu, ptr}
// Parameters: [qz_in, qx, qy, ptr, sel4, sat, rm, sar]
def int_riscv_esp_cmul_s8_ld_incp : Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty, llvm_ptr_ty], 
                                                  [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], 
                                                  [IntrReadMem, IntrArgMemOnly, ImmArg<ArgIndex<4>>]>;

// CMUL.S8.ST.INCP (unified 2.2 API)
// Returns: {v16i8 qz, ptr}
// Parameters: [qz_in, qx, qy, qu, ptr, sel4, sat, rm, sar]
def int_riscv_esp_cmul_s8_st_incp : Intrinsic<[llvm_v16i8_ty, llvm_ptr_ty], 
                                                  [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], 
                                                  [IntrWriteMem, IntrArgMemOnly, ImmArg<ArgIndex<5>>]>;

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
def int_riscv_esp_movx_r_sar : Intrinsic<[llvm_i32_ty], 
                                            [llvm_i32_ty], 
                                            [IntrNoMem]>;

// MOVX - Read/Write implicit registers
// MOVX.R - Read implicit registers (all return i32)
def int_riscv_esp_movx_r_cfg : Intrinsic<[llvm_i32_ty], [],
                                            [IntrNoMem,IntrHasSideEffects]>;

// MOVX.R.SAR_BYTES - Read SAR_BYTES register
// Returns: i32 (SAR_BYTES register value, low 4 bits)
// Parameters: [ i32 sar_bytes (SAR_BYTES register value, passthru for explicit state passing) ]
def int_riscv_esp_movx_r_sar_bytes : Intrinsic<[llvm_i32_ty], 
                                                   [llvm_i32_ty], 
                                                   [IntrNoMem]>;

// ESP.VADD.*.LD/ST.INCP (unified 2.2 API)
def int_riscv_esp_vadd_s16_ld_incp : Intrinsic<
	[llvm_v8i16_ty, llvm_v16i8_ty, llvm_ptr_ty],
	[llvm_v8i16_ty, llvm_v8i16_ty, llvm_ptr_ty, llvm_i32_ty],
	[IntrReadMem, IntrArgMemOnly, ImmArg<ArgIndex<3>>]
>;
def int_riscv_esp_vadd_s16_st_incp : Intrinsic<
	[llvm_v8i16_ty, llvm_ptr_ty],
	[llvm_v8i16_ty, llvm_v8i16_ty, llvm_v16i8_ty, llvm_ptr_ty, llvm_v8i16_ty, llvm_i32_ty],
	[IntrWriteMem, IntrArgMemOnly, ImmArg<ArgIndex<5>>]
>;
def int_riscv_esp_vadd_s8_ld_incp : Intrinsic<
	[llvm_v16i8_ty, llvm_v16i8_ty, llvm_ptr_ty],
	[llvm_v16i8_ty, llvm_v16i8_ty, llvm_ptr_ty, llvm_i32_ty],
	[IntrReadMem, IntrArgMemOnly, ImmArg<ArgIndex<3>>]
>;
def int_riscv_esp_vadd_s8_st_incp : Intrinsic<
	[llvm_v16i8_ty, llvm_ptr_ty],
	[llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_ptr_ty, llvm_v16i8_ty, llvm_i32_ty],
	[IntrWriteMem, IntrArgMemOnly, ImmArg<ArgIndex<5>>]
>;
def int_riscv_esp_vadd_u16_ld_incp : Intrinsic<
	[llvm_v8i16_ty, llvm_v16i8_ty, llvm_ptr_ty],
	[llvm_v8i16_ty, llvm_v8i16_ty, llvm_ptr_ty, llvm_i32_ty],
	[IntrReadMem, IntrArgMemOnly, ImmArg<ArgIndex<3>>]
>;
def int_riscv_esp_vadd_u16_st_incp : Intrinsic<
	[llvm_v8i16_ty, llvm_ptr_ty],
	[llvm_v8i16_ty, llvm_v8i16_ty, llvm_v16i8_ty, llvm_ptr_ty, llvm_v8i16_ty, llvm_i32_ty],
	[IntrWriteMem, IntrArgMemOnly, ImmArg<ArgIndex<5>>]
>;
def int_riscv_esp_vadd_u8_ld_incp : Intrinsic<
	[llvm_v16i8_ty, llvm_v16i8_ty, llvm_ptr_ty],
	[llvm_v16i8_ty, llvm_v16i8_ty, llvm_ptr_ty, llvm_i32_ty],
	[IntrReadMem, IntrArgMemOnly, ImmArg<ArgIndex<3>>]
>;
def int_riscv_esp_vadd_u8_st_incp : Intrinsic<
	[llvm_v16i8_ty, llvm_ptr_ty],
	[llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_ptr_ty, llvm_v16i8_ty, llvm_i32_ty],
	[IntrWriteMem, IntrArgMemOnly, ImmArg<ArgIndex<5>>]
>;
def int_riscv_esp_vadd_s32_ld_incp : Intrinsic<
	[llvm_v4i32_ty, llvm_v16i8_ty, llvm_ptr_ty],
	[llvm_v4i32_ty, llvm_v4i32_ty, llvm_ptr_ty, llvm_i32_ty],
	[IntrReadMem, IntrArgMemOnly, ImmArg<ArgIndex<3>>]
>;
def int_riscv_esp_vadd_s32_st_incp : Intrinsic<
	[llvm_v4i32_ty, llvm_ptr_ty],
	[llvm_v4i32_ty, llvm_v4i32_ty, llvm_v16i8_ty, llvm_ptr_ty, llvm_v4i32_ty, llvm_i32_ty],
	[IntrWriteMem, IntrArgMemOnly, ImmArg<ArgIndex<5>>]
>;
def int_riscv_esp_vadd_u32_ld_incp : Intrinsic<
	[llvm_v4i32_ty, llvm_v16i8_ty, llvm_ptr_ty],
	[llvm_v4i32_ty, llvm_v4i32_ty, llvm_ptr_ty, llvm_i32_ty],
	[IntrReadMem, IntrArgMemOnly, ImmArg<ArgIndex<3>>]
>;
def int_riscv_esp_vadd_u32_st_incp : Intrinsic<
	[llvm_v4i32_ty, llvm_ptr_ty],
	[llvm_v4i32_ty, llvm_v4i32_ty, llvm_v16i8_ty, llvm_ptr_ty, llvm_v4i32_ty, llvm_i32_ty],
	[IntrWriteMem, IntrArgMemOnly, ImmArg<ArgIndex<5>>]
>;

// ESP.VMAX.*.LD/ST.INCP (_m)
def int_riscv_esp_vmax_s8_ld_incp : Intrinsic<
	[llvm_v16i8_ty, llvm_v16i8_ty, llvm_ptr_ty],
	[llvm_v16i8_ty, llvm_v16i8_ty, llvm_ptr_ty],
	[IntrReadMem, IntrArgMemOnly]>;
def int_riscv_esp_vmax_s8_st_incp : Intrinsic<
	[llvm_v16i8_ty, llvm_ptr_ty],
	[llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_ptr_ty, llvm_v16i8_ty],
	[IntrWriteMem, IntrArgMemOnly]>;
def int_riscv_esp_vmax_s16_ld_incp : Intrinsic<
	[llvm_v8i16_ty, llvm_v16i8_ty, llvm_ptr_ty],
	[llvm_v8i16_ty, llvm_v8i16_ty, llvm_ptr_ty],
	[IntrReadMem, IntrArgMemOnly]>;
def int_riscv_esp_vmax_s16_st_incp : Intrinsic<
	[llvm_v8i16_ty, llvm_ptr_ty],
	[llvm_v8i16_ty, llvm_v8i16_ty, llvm_v16i8_ty, llvm_ptr_ty, llvm_v8i16_ty],
	[IntrWriteMem, IntrArgMemOnly]>;
def int_riscv_esp_vmax_s32_ld_incp : Intrinsic<
	[llvm_v4i32_ty, llvm_v16i8_ty, llvm_ptr_ty],
	[llvm_v4i32_ty, llvm_v4i32_ty, llvm_ptr_ty],
	[IntrReadMem, IntrArgMemOnly]>;
def int_riscv_esp_vmax_s32_st_incp : Intrinsic<
	[llvm_v4i32_ty, llvm_ptr_ty],
	[llvm_v4i32_ty, llvm_v4i32_ty, llvm_v16i8_ty, llvm_ptr_ty, llvm_v4i32_ty],
	[IntrWriteMem, IntrArgMemOnly]>;
def int_riscv_esp_vmax_u8_ld_incp : Intrinsic<
	[llvm_v16i8_ty, llvm_v16i8_ty, llvm_ptr_ty],
	[llvm_v16i8_ty, llvm_v16i8_ty, llvm_ptr_ty],
	[IntrReadMem, IntrArgMemOnly]>;
def int_riscv_esp_vmax_u8_st_incp : Intrinsic<
	[llvm_v16i8_ty, llvm_ptr_ty],
	[llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_ptr_ty, llvm_v16i8_ty],
	[IntrWriteMem, IntrArgMemOnly]>;
def int_riscv_esp_vmax_u16_ld_incp : Intrinsic<
	[llvm_v8i16_ty, llvm_v16i8_ty, llvm_ptr_ty],
	[llvm_v8i16_ty, llvm_v8i16_ty, llvm_ptr_ty],
	[IntrReadMem, IntrArgMemOnly]>;
def int_riscv_esp_vmax_u16_st_incp : Intrinsic<
	[llvm_v8i16_ty, llvm_ptr_ty],
	[llvm_v8i16_ty, llvm_v8i16_ty, llvm_v16i8_ty, llvm_ptr_ty, llvm_v8i16_ty],
	[IntrWriteMem, IntrArgMemOnly]>;
def int_riscv_esp_vmax_u32_ld_incp : Intrinsic<
	[llvm_v4i32_ty, llvm_v16i8_ty, llvm_ptr_ty],
	[llvm_v4i32_ty, llvm_v4i32_ty, llvm_ptr_ty],
	[IntrReadMem, IntrArgMemOnly]>;
def int_riscv_esp_vmax_u32_st_incp : Intrinsic<
	[llvm_v4i32_ty, llvm_ptr_ty],
	[llvm_v4i32_ty, llvm_v4i32_ty, llvm_v16i8_ty, llvm_ptr_ty, llvm_v4i32_ty],
	[IntrWriteMem, IntrArgMemOnly]>;

// ESP.VMIN.*.LD/ST.INCP (_m)
def int_riscv_esp_vmin_s8_ld_incp : Intrinsic<
	[llvm_v16i8_ty, llvm_v16i8_ty, llvm_ptr_ty],
	[llvm_v16i8_ty, llvm_v16i8_ty, llvm_ptr_ty],
	[IntrReadMem, IntrArgMemOnly]>;
def int_riscv_esp_vmin_s8_st_incp : Intrinsic<
	[llvm_v16i8_ty, llvm_ptr_ty],
	[llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_ptr_ty, llvm_v16i8_ty],
	[IntrWriteMem, IntrArgMemOnly]>;
def int_riscv_esp_vmin_s16_ld_incp : Intrinsic<
	[llvm_v8i16_ty, llvm_v16i8_ty, llvm_ptr_ty],
	[llvm_v8i16_ty, llvm_v8i16_ty, llvm_ptr_ty],
	[IntrReadMem, IntrArgMemOnly]>;
def int_riscv_esp_vmin_s16_st_incp : Intrinsic<
	[llvm_v8i16_ty, llvm_ptr_ty],
	[llvm_v8i16_ty, llvm_v8i16_ty, llvm_v16i8_ty, llvm_ptr_ty, llvm_v8i16_ty],
	[IntrWriteMem, IntrArgMemOnly]>;
def int_riscv_esp_vmin_s32_ld_incp : Intrinsic<
	[llvm_v4i32_ty, llvm_v16i8_ty, llvm_ptr_ty],
	[llvm_v4i32_ty, llvm_v4i32_ty, llvm_ptr_ty],
	[IntrReadMem, IntrArgMemOnly]>;
def int_riscv_esp_vmin_s32_st_incp : Intrinsic<
	[llvm_v4i32_ty, llvm_ptr_ty],
	[llvm_v4i32_ty, llvm_v4i32_ty, llvm_v16i8_ty, llvm_ptr_ty, llvm_v4i32_ty],
	[IntrWriteMem, IntrArgMemOnly]>;
def int_riscv_esp_vmin_u8_ld_incp : Intrinsic<
	[llvm_v16i8_ty, llvm_v16i8_ty, llvm_ptr_ty],
	[llvm_v16i8_ty, llvm_v16i8_ty, llvm_ptr_ty],
	[IntrReadMem, IntrArgMemOnly]>;
def int_riscv_esp_vmin_u8_st_incp : Intrinsic<
	[llvm_v16i8_ty, llvm_ptr_ty],
	[llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_ptr_ty, llvm_v16i8_ty],
	[IntrWriteMem, IntrArgMemOnly]>;
def int_riscv_esp_vmin_u16_ld_incp : Intrinsic<
	[llvm_v8i16_ty, llvm_v16i8_ty, llvm_ptr_ty],
	[llvm_v8i16_ty, llvm_v8i16_ty, llvm_ptr_ty],
	[IntrReadMem, IntrArgMemOnly]>;
def int_riscv_esp_vmin_u16_st_incp : Intrinsic<
	[llvm_v8i16_ty, llvm_ptr_ty],
	[llvm_v8i16_ty, llvm_v8i16_ty, llvm_v16i8_ty, llvm_ptr_ty, llvm_v8i16_ty],
	[IntrWriteMem, IntrArgMemOnly]>;
def int_riscv_esp_vmin_u32_ld_incp : Intrinsic<
	[llvm_v4i32_ty, llvm_v16i8_ty, llvm_ptr_ty],
	[llvm_v4i32_ty, llvm_v4i32_ty, llvm_ptr_ty],
	[IntrReadMem, IntrArgMemOnly]>;
def int_riscv_esp_vmin_u32_st_incp : Intrinsic<
	[llvm_v4i32_ty, llvm_ptr_ty],
	[llvm_v4i32_ty, llvm_v4i32_ty, llvm_v16i8_ty, llvm_ptr_ty, llvm_v4i32_ty],
	[IntrWriteMem, IntrArgMemOnly]>;

// ESP.VSUB.*.LD/ST.INCP (unified 2.2 API)
def int_riscv_esp_vsub_s8_ld_incp : Intrinsic<
	[llvm_v16i8_ty, llvm_v16i8_ty, llvm_ptr_ty],
	[llvm_v16i8_ty, llvm_v16i8_ty, llvm_ptr_ty, llvm_i32_ty],
	[IntrReadMem, IntrArgMemOnly, ImmArg<ArgIndex<3>>]
>;
def int_riscv_esp_vsub_s8_st_incp : Intrinsic<
	[llvm_v16i8_ty, llvm_ptr_ty],
	[llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_ptr_ty, llvm_v16i8_ty, llvm_i32_ty],
	[IntrWriteMem, IntrArgMemOnly, ImmArg<ArgIndex<5>>]
>;
def int_riscv_esp_vsub_s16_ld_incp : Intrinsic<
	[llvm_v8i16_ty, llvm_v16i8_ty, llvm_ptr_ty],
	[llvm_v8i16_ty, llvm_v8i16_ty, llvm_ptr_ty, llvm_i32_ty],
	[IntrReadMem, IntrArgMemOnly, ImmArg<ArgIndex<3>>]
>;
def int_riscv_esp_vsub_s16_st_incp : Intrinsic<
	[llvm_v8i16_ty, llvm_ptr_ty],
	[llvm_v8i16_ty, llvm_v8i16_ty, llvm_v16i8_ty, llvm_ptr_ty, llvm_v8i16_ty, llvm_i32_ty],
	[IntrWriteMem, IntrArgMemOnly, ImmArg<ArgIndex<5>>]
>;
def int_riscv_esp_vsub_s32_ld_incp : Intrinsic<
	[llvm_v4i32_ty, llvm_v16i8_ty, llvm_ptr_ty],
	[llvm_v4i32_ty, llvm_v4i32_ty, llvm_ptr_ty, llvm_i32_ty],
	[IntrReadMem, IntrArgMemOnly, ImmArg<ArgIndex<3>>]
>;
def int_riscv_esp_vsub_s32_st_incp : Intrinsic<
	[llvm_v4i32_ty, llvm_ptr_ty],
	[llvm_v4i32_ty, llvm_v4i32_ty, llvm_v16i8_ty, llvm_ptr_ty, llvm_v4i32_ty, llvm_i32_ty],
	[IntrWriteMem, IntrArgMemOnly, ImmArg<ArgIndex<5>>]
>;
def int_riscv_esp_vsub_u8_ld_incp : Intrinsic<
	[llvm_v16i8_ty, llvm_v16i8_ty, llvm_ptr_ty],
	[llvm_v16i8_ty, llvm_v16i8_ty, llvm_ptr_ty, llvm_i32_ty],
	[IntrReadMem, IntrArgMemOnly, ImmArg<ArgIndex<3>>]
>;
def int_riscv_esp_vsub_u8_st_incp : Intrinsic<
	[llvm_v16i8_ty, llvm_ptr_ty],
	[llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_ptr_ty, llvm_v16i8_ty, llvm_i32_ty],
	[IntrWriteMem, IntrArgMemOnly, ImmArg<ArgIndex<5>>]
>;
def int_riscv_esp_vsub_u16_ld_incp : Intrinsic<
	[llvm_v8i16_ty, llvm_v16i8_ty, llvm_ptr_ty],
	[llvm_v8i16_ty, llvm_v8i16_ty, llvm_ptr_ty, llvm_i32_ty],
	[IntrReadMem, IntrArgMemOnly, ImmArg<ArgIndex<3>>]
>;
def int_riscv_esp_vsub_u16_st_incp : Intrinsic<
	[llvm_v8i16_ty, llvm_ptr_ty],
	[llvm_v8i16_ty, llvm_v8i16_ty, llvm_v16i8_ty, llvm_ptr_ty, llvm_v8i16_ty, llvm_i32_ty],
	[IntrWriteMem, IntrArgMemOnly, ImmArg<ArgIndex<5>>]
>;
def int_riscv_esp_vsub_u32_ld_incp : Intrinsic<
	[llvm_v4i32_ty, llvm_v16i8_ty, llvm_ptr_ty],
	[llvm_v4i32_ty, llvm_v4i32_ty, llvm_ptr_ty, llvm_i32_ty],
	[IntrReadMem, IntrArgMemOnly, ImmArg<ArgIndex<3>>]
>;
def int_riscv_esp_vsub_u32_st_incp : Intrinsic<
	[llvm_v4i32_ty, llvm_ptr_ty],
	[llvm_v4i32_ty, llvm_v4i32_ty, llvm_v16i8_ty, llvm_ptr_ty, llvm_v4i32_ty, llvm_i32_ty],
	[IntrWriteMem, IntrArgMemOnly, ImmArg<ArgIndex<5>>]
>;

// ESP.VMUL.S8 / U8 / U16.LD.INCP, .ST.INCP (unified 2.2 API)
def int_riscv_esp_vmul_s8_ld_incp : Intrinsic<
	[llvm_v16i8_ty, llvm_v16i8_ty, llvm_ptr_ty],
	[llvm_v16i8_ty, llvm_v16i8_ty, llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
	[IntrReadMem, IntrArgMemOnly, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;
def int_riscv_esp_vmul_s8_st_incp : Intrinsic<
	[llvm_v16i8_ty, llvm_ptr_ty],
	[llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_ptr_ty, llvm_v16i8_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
	[IntrWriteMem, IntrArgMemOnly, ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>]>;
def int_riscv_esp_vmul_u8_ld_incp : Intrinsic<
	[llvm_v16i8_ty, llvm_v16i8_ty, llvm_ptr_ty],
	[llvm_v16i8_ty, llvm_v16i8_ty, llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
	[IntrReadMem, IntrArgMemOnly, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;
def int_riscv_esp_vmul_u8_st_incp : Intrinsic<
	[llvm_v16i8_ty, llvm_ptr_ty],
	[llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_ptr_ty, llvm_v16i8_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
	[IntrWriteMem, IntrArgMemOnly, ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>]>;
def int_riscv_esp_vmul_u16_ld_incp : Intrinsic<
	[llvm_v8i16_ty, llvm_v16i8_ty, llvm_ptr_ty],
	[llvm_v8i16_ty, llvm_v8i16_ty, llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
	[IntrReadMem, IntrArgMemOnly, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;
def int_riscv_esp_vmul_u16_st_incp : Intrinsic<
	[llvm_v8i16_ty, llvm_ptr_ty],
	[llvm_v8i16_ty, llvm_v8i16_ty, llvm_v16i8_ty, llvm_ptr_ty, llvm_v8i16_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
	[IntrWriteMem, IntrArgMemOnly, ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>]>;

// GPR ALU (.m) — SSA model: rd = op(rs1, rs2); maps to esp.{addx2,addx4,subx2,subx4} rd, rs1, rs2.
def int_riscv_esp_addx2_m : ClangBuiltin<"__builtin_riscv_esp_addx2_m">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
def int_riscv_esp_addx4_m : ClangBuiltin<"__builtin_riscv_esp_addx4_m">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
def int_riscv_esp_subx2_m : ClangBuiltin<"__builtin_riscv_esp_subx2_m">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
def int_riscv_esp_subx4_m : ClangBuiltin<"__builtin_riscv_esp_subx4_m">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;
def int_riscv_esp_sat_m : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrNoMem]>;

// SRC.Q - Shift Right Concatenated Quad registers (_m version)
// Performs arithmetic right shift on concatenated qy and qw using SAR_BYTES register
// Returns: shifted result in qz
// Parameters: [SAR_BYTES (64-bit, only low 4 bits used), qy, qw]
// Migration: Explicit state passing (no IntrHasSideEffects)
// Note: Shift amount is SAR_BYTES[3:0] << 3 (hardware automatically uses only low 4 bits)
def int_riscv_esp_src_q : Intrinsic<[llvm_v16i8_ty], 
                                         [llvm_i32_ty, llvm_v16i8_ty, llvm_v16i8_ty], 
                                         [IntrNoMem]>;

// SRC.Q.LD.IP - Shift Right Concatenated Quad registers with Load and Immediate Post-increment (_m version)
// Performs arithmetic right shift on concatenated qy and qw, loads data to qu, updates pointer
// Returns: qw (updated), qu (loaded), ptr (updated)
// Parameters: [SAR_BYTES (32-bit, only low 4 bits used, range 0-15), qy, qw, ptr, offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
// Note: Shift amount is SAR_BYTES[3:0] << 3 (hardware automatically uses only low 4 bits)
def int_riscv_esp_src_q_ld_ip : Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty, llvm_ptr_ty], 
                                              [llvm_i32_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_ptr_ty, llvm_i32_ty], 
                                              [IntrReadMem, IntrArgMemOnly]>;

// SRC.Q.LD.XP - Shift Right Concatenated Quad registers with Load and Register Post-increment (_m version)
// Performs arithmetic right shift on concatenated qy and qw, loads data to qu, updates pointer
// Returns: qw (updated), qu (loaded), ptr (updated)
// Parameters: [SAR_BYTES (64-bit, only low 4 bits used), qy, qw, ptr, offset_reg]
// Migration: Explicit state passing (no IntrHasSideEffects)
// Note: Shift amount is SAR_BYTES[3:0] << 3 (hardware automatically uses only low 4 bits)
def int_riscv_esp_src_q_ld_xp : Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty, llvm_ptr_ty], 
                                              [llvm_i32_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_ptr_ty, llvm_i32_ty], 
                                              [IntrReadMem, IntrArgMemOnly]>;

// SRC.Q.qup - Shift Right Concatenated Quad registers with Update (_m version)
// Performs arithmetic right shift on concatenated qw and qy, updates qw = qy
// Returns: qz (new), qw (updated)
// Parameters: [SAR_BYTES (64-bit, only low 4 bits used), qy, qw]
// Migration: Explicit state passing (no IntrHasSideEffects)
// Note: Shift amount is SAR_BYTES[3:0] << 3 (hardware automatically uses only low 4 bits)
def int_riscv_esp_src_q_qup : Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty], 
                                             [llvm_i32_ty, llvm_v16i8_ty, llvm_v16i8_ty], 
                                             [IntrNoMem]>;

// SRCQ.128.ST.INCP - Shift Right Concatenated Quad registers with Store and Increment Pointer (_m version)
// Performs arithmetic right shift on concatenated qy and qw, stores result, updates pointer
// Returns: ptr (updated pointer)
// Parameters: [SAR_BYTES (32-bit, only low 4 bits used, range 0-15), qy, qw, ptr]
// Migration: Explicit state passing (no IntrHasSideEffects)
// Note: Shift amount is SAR_BYTES[3:0] << 3 (hardware automatically uses only low 4 bits)
// SAR_BYTES is i32 to match SAR_BYTESReg register class definition
def int_riscv_esp_srcq_128_st_incp : Intrinsic<[llvm_ptr_ty], 
                                                   [llvm_i32_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_ptr_ty], 
                                                   [IntrWriteMem, IntrArgMemOnly]>;

// SRCXXP.2Q - Shift Right Concatenated with Register offset and Post-increment, 2 Quad registers
// Performs logical right shift on concatenated qw and qy, updates both registers and pointer
// Returns: { <16 x i8> new_qy, <16 x i8> new_qw, ptr new_ptr }
// Explicit state passing: All register updates are visible in IR through return values
// This allows optimization without IntrHasSideEffects while preventing dead code elimination
def int_riscv_esp_srcxxp_2q : Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty, llvm_ptr_ty], 
                                            [llvm_v16i8_ty, llvm_v16i8_ty, llvm_ptr_ty, llvm_i32_ty], 
                                            [IntrNoMem]>;

// SLCXXP.2Q - Shift Left with Register offset and Post-increment, 2 Quad registers
// Shifts concatenated qy and qw left, updates both registers and pointer
// Returns modified values and updated pointer, input and output registers are the same due to constraints
def int_riscv_esp_slcxxp_2q : Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty, llvm_ptr_ty], 
                                             [llvm_v16i8_ty, llvm_v16i8_ty, llvm_ptr_ty, llvm_i32_ty], 
                                             [IntrNoMem]>;

// SRCMB - Shift Right and Saturate from QACC (unified 2.1/2.2 API)
// S*: sat maps to sel2 on PIE 2.1; rm must be RM_DYNAMIC (7) on +xespv2p1.
// U*: sel2 used on PIE 2.1; sat=0 and rm=7 on +xespv2p1; PIE 2.2 uses sat/rm.
def int_riscv_esp_srcmb_s16_qacc : Intrinsic<[llvm_v8i16_ty],
    [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
    [IntrNoMem, ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>]>;
def int_riscv_esp_srcmb_s8_qacc : Intrinsic<[llvm_v16i8_ty],
    [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
    [IntrNoMem, ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>]>;
def int_riscv_esp_srcmb_s16_q_qacc : Intrinsic<[llvm_v8i16_ty],
    [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v8i16_ty, llvm_i32_ty, llvm_i32_ty],
    [IntrNoMem, ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>]>;
def int_riscv_esp_srcmb_s8_q_qacc : Intrinsic<[llvm_v16i8_ty],
    [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_i32_ty, llvm_i32_ty],
    [IntrNoMem, ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>]>;
def int_riscv_esp_srcmb_u16_qacc : Intrinsic<[llvm_v8i16_ty],
    [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
    [IntrNoMem, ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>, ImmArg<ArgIndex<7>>]>;
def int_riscv_esp_srcmb_u8_qacc : Intrinsic<[llvm_v16i8_ty],
    [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
    [IntrNoMem, ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>, ImmArg<ArgIndex<7>>]>;
def int_riscv_esp_srcmb_u16_q_qacc : Intrinsic<[llvm_v8i16_ty],
    [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v8i16_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
    [IntrNoMem, ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>, ImmArg<ArgIndex<7>>]>;
def int_riscv_esp_srcmb_u8_q_qacc : Intrinsic<[llvm_v16i8_ty],
    [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
    [IntrNoMem, ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>, ImmArg<ArgIndex<7>>]>;

// ESP.VMULAS.S16.QACC (_m version) - Multiply-accumulate to QACC_H and QACC_L
// Returns: {v16i8, v16i8, v16i8, v16i8} - 4x128-bit QACC directly
// Parameters: v16i8, v16i8, v16i8, v16i8 (4x128-bit QACC passthru), v8i16, v8i16 (qx, qy)
def int_riscv_esp_vmulas_s16_qacc :
	Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,  
               llvm_v8i16_ty,                                                  
               llvm_v8i16_ty, llvm_i32_ty],
              [IntrNoMem, ImmArg<ArgIndex<6>>]>;

// ESP.VMULAS.S16.QACC.LD.IP (_m version) - Load 128-bit, multiply-accumulate to QACC_H and QACC_L
// Returns: {ptr, v16i8 qu, v16i8, v16i8, v16i8, v16i8} - ptr + qu + 4x128-bit QACC directly
// Parameters: v16i8, v16i8, v16i8, v16i8 (4x128-bit QACC passthru), v8i16, v8i16 (qx, qy), ptr, i32 (offset)
def int_riscv_esp_vmulas_s16_qacc_ld_ip :
	Intrinsic<[llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,              
               llvm_v8i16_ty,                                                              
               llvm_v8i16_ty,                                                              
               llvm_ptr_ty,                                
               llvm_i32_ty, llvm_i32_ty],
              [IntrReadMem, IntrArgMemOnly, ImmArg<ArgIndex<8>>]>;

// ESP.VMULAS.S16.QACC.LD.XP (_m version)
def int_riscv_esp_vmulas_s16_qacc_ld_xp :
	Intrinsic<[llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,              
               llvm_v8i16_ty,                                                              
               llvm_v8i16_ty,                                                              
               llvm_ptr_ty,                                
               llvm_i32_ty, llvm_i32_ty],
              [IntrReadMem, IntrArgMemOnly, ImmArg<ArgIndex<8>>]>;

// ESP.VMULAS.S16.QACC.ST.IP (_m version)
def int_riscv_esp_vmulas_s16_qacc_st_ip :
	Intrinsic<[llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,              
               llvm_v16i8_ty,                                                              
               llvm_v8i16_ty,                                                              
               llvm_v8i16_ty,                                                              
               llvm_ptr_ty,                                
               llvm_i32_ty, llvm_i32_ty],
              [IntrWriteMem, IntrArgMemOnly, ImmArg<ArgIndex<9>>]>;

// ESP.VMULAS.S16.QACC.ST.XP (_m version)
def int_riscv_esp_vmulas_s16_qacc_st_xp :
	Intrinsic<[llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,              
               llvm_v16i8_ty,                                                              
               llvm_v8i16_ty,                                                              
               llvm_v8i16_ty,                                                              
               llvm_ptr_ty,                                
               llvm_i32_ty, llvm_i32_ty],
              [IntrWriteMem, IntrArgMemOnly, ImmArg<ArgIndex<9>>]>;

// ESP.VMULAS.S8.QACC (_m version) - Multiply-accumulate to QACC_H and QACC_L
// Returns: {v16i8, v16i8, v16i8, v16i8} - 4x128-bit QACC directly
// Parameters: v16i8, v16i8, v16i8, v16i8 (4x128-bit QACC passthru), v16i8, v16i8 (qx, qy)
def int_riscv_esp_vmulas_s8_qacc :
	Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,  
               llvm_v16i8_ty,                                                  
               llvm_v16i8_ty, llvm_i32_ty],
              [IntrNoMem, ImmArg<ArgIndex<6>>]>;

// ESP.VSMULAS.S16.QACC (_m version) - Multiply-accumulate to QACC_H and QACC_L
// Returns: {v16i8, v16i8, v16i8, v16i8} - 4x128-bit QACC directly
// Parameters: v16i8, v16i8, v16i8, v16i8 (4x128-bit QACC passthru), v8i16, v8i16 (qx, qy), i32 (sel16)
// Note: No ClangBuiltin - builtin signature doesn't match intrinsic signature (builtin uses output pointers)
def int_riscv_esp_vsmulas_s16_qacc :
	Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty], 
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,  
               llvm_v8i16_ty,                                                  
               llvm_v8i16_ty,                                                 
               llvm_i32_ty, llvm_i32_ty],                                                  
              [IntrNoMem, ImmArg<ArgIndex<7>>]>;                      

// ESP.VSMULAS.S16.QACC.LD.INCP (_m version)
// Returns: (ptr, qu, v0, v1, v2, v3) - ptr + qu + 4x128-bit QACC (6 values)
// Parameters: (v0, v1, v2, v3, qx, qy, ptr, sel16) - 4x128-bit QACC passthru + qx + qy + ptr + sel16
def int_riscv_esp_vsmulas_s16_qacc_ld_incp :
	Intrinsic<[llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty], 
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,              
               llvm_v8i16_ty,                                                              
               llvm_v8i16_ty,                                                              
               llvm_ptr_ty,                                
               llvm_i32_ty, llvm_i32_ty],                               
              [IntrReadMem, IntrArgMemOnly, ImmArg<ArgIndex<8>>]>;

// ESP.VSMULAS.S8.QACC (_m version) - Multiply-accumulate to QACC_H and QACC_L
// Returns: {v16i8, v16i8, v16i8, v16i8} - 4x128-bit QACC directly
// Parameters: v16i8, v16i8, v16i8, v16i8 (4x128-bit QACC passthru), v16i8, v16i8 (qx, qy), i32 (sel16)
// Note: No ClangBuiltin - builtin signature doesn't match intrinsic signature (builtin uses output pointers)
def int_riscv_esp_vsmulas_s8_qacc :
	Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty], 
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,  
               llvm_v16i8_ty,                                                 
               llvm_v16i8_ty,                                                 
               llvm_i32_ty, llvm_i32_ty],                                                  
              [IntrNoMem, ImmArg<ArgIndex<7>>]>;                      

// ESP.VMULAS.S8.QACC.LD.IP (_m version)
def int_riscv_esp_vmulas_s8_qacc_ld_ip :
	Intrinsic<[llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,              
               llvm_v16i8_ty,                                                              
               llvm_v16i8_ty,                                                              
               llvm_ptr_ty,                                
               llvm_i32_ty, llvm_i32_ty],
              [IntrReadMem, IntrArgMemOnly, ImmArg<ArgIndex<8>>]>;

// ESP.VMULAS.S8.QACC.LD.XP (_m version)
def int_riscv_esp_vmulas_s8_qacc_ld_xp :
	Intrinsic<[llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,              
               llvm_v16i8_ty,                                                              
               llvm_v16i8_ty,                                                              
               llvm_ptr_ty,                                
               llvm_i32_ty, llvm_i32_ty],
              [IntrReadMem, IntrArgMemOnly, ImmArg<ArgIndex<8>>]>;

// ESP.VMULAS.S8.QACC.ST.IP (_m version)
def int_riscv_esp_vmulas_s8_qacc_st_ip :
	Intrinsic<[llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,              
               llvm_v16i8_ty,                                                              
               llvm_v16i8_ty,                                                              
               llvm_v16i8_ty,                                                              
               llvm_ptr_ty,                                
               llvm_i32_ty, llvm_i32_ty],
              [IntrWriteMem, IntrArgMemOnly, ImmArg<ArgIndex<9>>]>;

// ESP.VMULAS.S8.QACC.ST.XP (_m version)
def int_riscv_esp_vmulas_s8_qacc_st_xp :
	Intrinsic<[llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,              
               llvm_v16i8_ty,                                                              
               llvm_v16i8_ty,                                                              
               llvm_v16i8_ty,                                                              
               llvm_ptr_ty,                                
               llvm_i32_ty, llvm_i32_ty],
              [IntrWriteMem, IntrArgMemOnly, ImmArg<ArgIndex<9>>]>;

// ESP.VMULAS.U16.QACC (_m version) - Multiply-accumulate to QACC_H and QACC_L
// Returns: {v16i8, v16i8, v16i8, v16i8} - 4x128-bit QACC directly
// Parameters: v16i8, v16i8, v16i8, v16i8 (4x128-bit QACC passthru), v8i16, v8i16 (qx, qy)
def int_riscv_esp_vmulas_u16_qacc :
	Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,  
               llvm_v8i16_ty,                                                  
               llvm_v8i16_ty, llvm_i32_ty],
              [IntrNoMem, ImmArg<ArgIndex<6>>]>;

// ESP.VMULAS.U16.QACC.LD.IP (_m version)
def int_riscv_esp_vmulas_u16_qacc_ld_ip :
	Intrinsic<[llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,              
               llvm_v8i16_ty,                                                              
               llvm_v8i16_ty,                                                              
               llvm_ptr_ty,                                
               llvm_i32_ty, llvm_i32_ty],
              [IntrReadMem, IntrArgMemOnly, ImmArg<ArgIndex<8>>]>;

// ESP.VMULAS.U16.QACC.LD.XP (_m version)
def int_riscv_esp_vmulas_u16_qacc_ld_xp :
	Intrinsic<[llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,              
               llvm_v8i16_ty,                                                              
               llvm_v8i16_ty,                                                              
               llvm_ptr_ty,                                
               llvm_i32_ty, llvm_i32_ty],
              [IntrReadMem, IntrArgMemOnly, ImmArg<ArgIndex<8>>]>;

// ESP.VMULAS.U16.QACC.ST.IP (_m version)
def int_riscv_esp_vmulas_u16_qacc_st_ip :
	Intrinsic<[llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,              
               llvm_v16i8_ty,                                                              
               llvm_v8i16_ty,                                                              
               llvm_v8i16_ty,                                                              
               llvm_ptr_ty,                                
               llvm_i32_ty, llvm_i32_ty],
              [IntrWriteMem, IntrArgMemOnly, ImmArg<ArgIndex<9>>]>;

// ESP.VMULAS.U16.QACC.ST.XP (_m version)
def int_riscv_esp_vmulas_u16_qacc_st_xp :
	Intrinsic<[llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,              
               llvm_v16i8_ty,                                                              
               llvm_v8i16_ty,                                                              
               llvm_v8i16_ty,                                                              
               llvm_ptr_ty,                                
               llvm_i32_ty, llvm_i32_ty],
              [IntrWriteMem, IntrArgMemOnly, ImmArg<ArgIndex<9>>]>;

// ESP.VMULAS.U8.QACC (_m version) - Multiply-accumulate to QACC_H and QACC_L
// Returns: {v16i8, v16i8, v16i8, v16i8} - 4x128-bit QACC directly
// Parameters: v16i8, v16i8, v16i8, v16i8 (4x128-bit QACC passthru), v16i8, v16i8 (qx, qy)
def int_riscv_esp_vmulas_u8_qacc :
	Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,  
               llvm_v16i8_ty,                                                  
               llvm_v16i8_ty, llvm_i32_ty],
              [IntrNoMem, ImmArg<ArgIndex<6>>]>;

// ESP.VMULAS.U8.QACC.LD.IP (_m version)
def int_riscv_esp_vmulas_u8_qacc_ld_ip :
	Intrinsic<[llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,              
               llvm_v16i8_ty,                                                              
               llvm_v16i8_ty,                                                              
               llvm_ptr_ty,                                
               llvm_i32_ty, llvm_i32_ty],
              [IntrReadMem, IntrArgMemOnly, ImmArg<ArgIndex<8>>]>;

// ESP.VMULAS.U8.QACC.LD.XP (_m version)
def int_riscv_esp_vmulas_u8_qacc_ld_xp :
	Intrinsic<[llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,              
               llvm_v16i8_ty,                                                              
               llvm_v16i8_ty,                                                              
               llvm_ptr_ty,                                
               llvm_i32_ty, llvm_i32_ty],
              [IntrReadMem, IntrArgMemOnly, ImmArg<ArgIndex<8>>]>;

// ESP.VMULAS.U8.QACC.ST.IP (_m version)
def int_riscv_esp_vmulas_u8_qacc_st_ip :
	Intrinsic<[llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,              
               llvm_v16i8_ty,                                                              
               llvm_v16i8_ty,                                                              
               llvm_v16i8_ty,                                                              
               llvm_ptr_ty,                                
               llvm_i32_ty, llvm_i32_ty],
              [IntrWriteMem, IntrArgMemOnly, ImmArg<ArgIndex<9>>]>;

// ESP.VMULAS.U8.QACC.ST.XP (_m version)
def int_riscv_esp_vmulas_u8_qacc_st_xp :
	Intrinsic<[llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,              
               llvm_v16i8_ty,                                                              
               llvm_v16i8_ty,                                                              
               llvm_v16i8_ty,                                                              
               llvm_ptr_ty,                                
               llvm_i32_ty, llvm_i32_ty],
              [IntrWriteMem, IntrArgMemOnly, ImmArg<ArgIndex<9>>]>;

// ESP.VMULAS.S16.XACC (_m version)
// Returns: new XACC (only low 40 bits used)
// Parameters: [current XACC (passthru, only low 40 bits used), qx, qy]
// Migration: Explicit state passing (no IntrHasSideEffects)
// Note: No ClangBuiltin - builtin signature (void with 6 params including pointers) doesn't match intrinsic signature ({i32, i8} with 4 params)
// CodeGen is handled manually in riscv_espv_builtin_cg.inc
// ESP.VMULAS.S16.XACC (_m version) - Mixed model: xacc_h and xacc_l are both i32
// Returns: struct {i32 (XACC[31:0]), i32 (XACC[39:32], only low 8 bits valid)}
def int_riscv_esp_vmulas_s16_xacc :
	Intrinsic<[llvm_i32_ty, llvm_i32_ty],
              [llvm_i32_ty,                         
               llvm_i32_ty,                         
               llvm_v8i16_ty,                       
               llvm_v8i16_ty, llvm_i32_ty],
              [IntrNoMem, ImmArg<ArgIndex<4>>]>;                         

// ESP.VMULAS.S16.XACC.LD.IP (_m version)
// Returns: qu vector + updated pointer + new XACC (only low 40 bits used)
// Parameters: [current XACC (passthru, only low 40 bits used), qx, qy, pointer, immediate offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
// ESP.VMULAS.S16.XACC.LD.IP (_m version) - Mixed model: XACC as {i32 low, i32 high}
// Returns: qu vector + updated pointer + new XACC as {i32 low, i32 high}
// Parameters: [xacc_low_in (i32), xacc_high_in (i32, only low 8 bits valid), qx, qy, pointer, offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_vmulas_s16_xacc_ld_ip :
	Intrinsic<[llvm_v16i8_ty, llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i32_ty,                                            
               llvm_i32_ty,                                            
               llvm_v8i16_ty,                                          
               llvm_v8i16_ty,                                          
               llvm_ptr_ty,                                            
               llvm_i32_ty, llvm_i32_ty],
              [IntrReadMem, IntrArgMemOnly, ImmArg<ArgIndex<6>>]>;                          

// ESP.VMULAS.S16.XACC.LD.XP (_m version) - Mixed model: XACC as {i32 low, i32 high}
// Returns: qu vector + updated pointer + new XACC as {i32 low, i32 high}
// Parameters: [xacc_low_in (i32), xacc_high_in (i32, only low 8 bits valid), qx, qy, pointer, register offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_vmulas_s16_xacc_ld_xp :
	Intrinsic<[llvm_v16i8_ty, llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i32_ty,                                            
               llvm_i32_ty,                                            
               llvm_v8i16_ty,                                          
               llvm_v8i16_ty,                                          
               llvm_ptr_ty,                                            
               llvm_i32_ty, llvm_i32_ty],
              [IntrReadMem, IntrArgMemOnly, ImmArg<ArgIndex<6>>]>;                          

// ESP.VMULAS.S16.XACC.ST.IP (_m version) - Mixed model: XACC as {i32 low, i32 high}
// Returns: updated pointer + new XACC as {i32 low, i32 high}
// Parameters: [xacc_low_in (i32), xacc_high_in (i32, only low 8 bits valid), qu, qx, qy, pointer, immediate offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_vmulas_s16_xacc_st_ip :
	Intrinsic<[llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i32_ty,                             
               llvm_i32_ty,                             
               llvm_v16i8_ty,                           
               llvm_v8i16_ty,                           
               llvm_v8i16_ty,                           
               llvm_ptr_ty,                             
               llvm_i32_ty, llvm_i32_ty],
              [IntrWriteMem, IntrArgMemOnly, ImmArg<ArgIndex<7>>]>;          

// ESP.VMULAS.S16.XACC.ST.XP (_m version) - Mixed model: XACC as {i32 low, i32 high}
// Returns: updated pointer + new XACC as {i32 low, i32 high}
// Parameters: [xacc_low_in (i32), xacc_high_in (i32, only low 8 bits valid), qu, qx, qy, pointer, register offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_vmulas_s16_xacc_st_xp :
	Intrinsic<[llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i32_ty,                             
               llvm_i32_ty,                             
               llvm_v16i8_ty,                           
               llvm_v8i16_ty,                           
               llvm_v8i16_ty,                           
               llvm_ptr_ty,                             
               llvm_i32_ty, llvm_i32_ty],
              [IntrWriteMem, IntrArgMemOnly, ImmArg<ArgIndex<7>>]>;          

// ESP.VMULAS.S8.XACC (_m version)
// Returns: new XACC (only low 40 bits used)
// Parameters: [current XACC (passthru, only low 40 bits used), qx, qy]
// Migration: Explicit state passing (no IntrHasSideEffects)
// Note: No ClangBuiltin - builtin signature (void with 6 params including pointers) doesn't match intrinsic signature ({i32, i8} with 4 params)
// CodeGen is handled manually in riscv_espv_builtin_cg.inc
// ESP.VMULAS.S8.XACC (_m version) - Mixed model: xacc_h and xacc_l are both i32
// Returns: struct {i32 (XACC[31:0]), i32 (XACC[39:32], only low 8 bits valid)}
def int_riscv_esp_vmulas_s8_xacc :
	Intrinsic<[llvm_i32_ty, llvm_i32_ty],
              [llvm_i32_ty,                         
               llvm_i32_ty,                         
               llvm_v16i8_ty,                       
               llvm_v16i8_ty, llvm_i32_ty],
              [IntrNoMem, ImmArg<ArgIndex<4>>]>;                         

// ESP.VMULAS.S8.XACC.LD.IP (_m version) - Mixed model: XACC as {i32 low, i32 high}
// Returns: qu vector + updated pointer + new XACC as {i32 low, i32 high}
// Parameters: [xacc_low_in (i32), xacc_high_in (i32, only low 8 bits valid), qx, qy, pointer, offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_vmulas_s8_xacc_ld_ip :
	Intrinsic<[llvm_v16i8_ty, llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i32_ty,                                            
               llvm_i32_ty,                                            
               llvm_v16i8_ty,                                          
               llvm_v16i8_ty,                                          
               llvm_ptr_ty,                                            
               llvm_i32_ty, llvm_i32_ty],
              [IntrReadMem, IntrArgMemOnly, ImmArg<ArgIndex<6>>]>;                          

// ESP.VMULAS.S8.XACC.LD.XP (_m version) - Mixed model: XACC as {i32 low, i32 high}
// Returns: qu vector + updated pointer + new XACC as {i32 low, i32 high}
// Parameters: [xacc_low_in (i32), xacc_high_in (i32, only low 8 bits valid), qx, qy, pointer, register offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_vmulas_s8_xacc_ld_xp :
	Intrinsic<[llvm_v16i8_ty, llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i32_ty,                                            
               llvm_i32_ty,                                            
               llvm_v16i8_ty,                                          
               llvm_v16i8_ty,                                          
               llvm_ptr_ty,                                            
               llvm_i32_ty, llvm_i32_ty],
              [IntrReadMem, IntrArgMemOnly, ImmArg<ArgIndex<6>>]>;                          

// ESP.VMULAS.S8.XACC.ST.IP (_m version) - Mixed model: XACC as {i32 low, i32 high}
// Returns: updated pointer + new XACC as {i32 low, i32 high}
// Parameters: [xacc_low_in (i32), xacc_high_in (i32, only low 8 bits valid), qu, qx, qy, pointer, immediate offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_vmulas_s8_xacc_st_ip :
	Intrinsic<[llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i32_ty,                             
               llvm_i32_ty,                             
               llvm_v16i8_ty,                           
               llvm_v16i8_ty,                           
               llvm_v16i8_ty,                           
               llvm_ptr_ty,                             
               llvm_i32_ty, llvm_i32_ty],
              [IntrWriteMem, IntrArgMemOnly, ImmArg<ArgIndex<7>>]>;          

// ESP.VMULAS.S8.XACC.ST.XP (_m version) - Mixed model: XACC as {i32 low, i32 high}
// Returns: updated pointer + new XACC as {i32 low, i32 high}
// Parameters: [xacc_low_in (i32), xacc_high_in (i32, only low 8 bits valid), qu, qx, qy, pointer, register offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_vmulas_s8_xacc_st_xp :
	Intrinsic<[llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i32_ty,                             
               llvm_i32_ty,                             
               llvm_v16i8_ty,                           
               llvm_v16i8_ty,                           
               llvm_v16i8_ty,                           
               llvm_ptr_ty,                             
               llvm_i32_ty, llvm_i32_ty],
              [IntrWriteMem, IntrArgMemOnly, ImmArg<ArgIndex<7>>]>;          

// ESP.VMULAS.U16.XACC (_m version)
// Returns: new XACC (only low 40 bits used)
// Parameters: [current XACC (passthru, only low 40 bits used), qx, qy]
// Migration: Explicit state passing (no IntrHasSideEffects)
// Note: No ClangBuiltin - builtin signature (void with 6 params including pointers) doesn't match intrinsic signature ({i32, i8} with 4 params)
// CodeGen is handled manually in riscv_espv_builtin_cg.inc
// ESP.VMULAS.U16.XACC (_m version) - Mixed model: xacc_h and xacc_l are both i32
// Returns: struct {i32 (XACC[31:0]), i32 (XACC[39:32], only low 8 bits valid)}
def int_riscv_esp_vmulas_u16_xacc :
	Intrinsic<[llvm_i32_ty, llvm_i32_ty],
              [llvm_i32_ty,                         
               llvm_i32_ty,                         
               llvm_v8i16_ty,                       
               llvm_v8i16_ty, llvm_i32_ty],
              [IntrNoMem, ImmArg<ArgIndex<4>>]>;                         

// ESP.VMULAS.U16.XACC.LD.IP (_m version)
// Returns: qu vector + updated pointer + new XACC (only low 40 bits used)
// Parameters: [current XACC (passthru, only low 40 bits used), qx, qy, pointer, immediate offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
// ESP.VMULAS.U16.XACC.LD.IP (_m version) - Mixed model: XACC as {i32 low, i32 high}
// Returns: qu vector + updated pointer + new XACC as {i32 low, i32 high}
// Parameters: [xacc_low_in (i32), xacc_high_in (i32, only low 8 bits valid), qx, qy, pointer, offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_vmulas_u16_xacc_ld_ip :
	Intrinsic<[llvm_v16i8_ty, llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i32_ty,                                            
               llvm_i32_ty,                                            
               llvm_v8i16_ty,                                          
               llvm_v8i16_ty,                                          
               llvm_ptr_ty,                                            
               llvm_i32_ty, llvm_i32_ty],
              [IntrReadMem, IntrArgMemOnly, ImmArg<ArgIndex<6>>]>;                          

// ESP.VMULAS.U16.XACC.LD.XP (_m version) - Mixed model: XACC as {i32 low, i32 high}
// Returns: qu vector + updated pointer + new XACC as {i32 low, i32 high}
// Parameters: [xacc_low_in (i32), xacc_high_in (i32, only low 8 bits valid), qx, qy, pointer, register offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_vmulas_u16_xacc_ld_xp :
	Intrinsic<[llvm_v16i8_ty, llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i32_ty,                                            
               llvm_i32_ty,                                            
               llvm_v8i16_ty,                                          
               llvm_v8i16_ty,                                          
               llvm_ptr_ty,                                            
               llvm_i32_ty, llvm_i32_ty],
              [IntrReadMem, IntrArgMemOnly, ImmArg<ArgIndex<6>>]>;                          

// ESP.VMULAS.U16.XACC.ST.IP (_m version) - Mixed model: XACC as {i32 low, i32 high}
// Returns: updated pointer + new XACC as {i32 low, i32 high}
// Parameters: [xacc_low_in (i32), xacc_high_in (i32, only low 8 bits valid), qu, qx, qy, pointer, immediate offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_vmulas_u16_xacc_st_ip :
	Intrinsic<[llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i32_ty,                             
               llvm_i32_ty,                             
               llvm_v16i8_ty,                           
               llvm_v8i16_ty,                           
               llvm_v8i16_ty,                           
               llvm_ptr_ty,                             
               llvm_i32_ty, llvm_i32_ty],
              [IntrWriteMem, IntrArgMemOnly, ImmArg<ArgIndex<7>>]>;          

// ESP.VMULAS.U16.XACC.ST.XP (_m version) - Mixed model: XACC as {i32 low, i32 high}
// Returns: updated pointer + new XACC as {i32 low, i32 high}
// Parameters: [xacc_low_in (i32), xacc_high_in (i32, only low 8 bits valid), qu, qx, qy, pointer, register offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_vmulas_u16_xacc_st_xp :
	Intrinsic<[llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i32_ty,                             
               llvm_i32_ty,                             
               llvm_v16i8_ty,                           
               llvm_v8i16_ty,                           
               llvm_v8i16_ty,                           
               llvm_ptr_ty,                             
               llvm_i32_ty, llvm_i32_ty],
              [IntrWriteMem, IntrArgMemOnly, ImmArg<ArgIndex<7>>]>;          

// ESP.VMULAS.U8.XACC (_m version)
// Returns: new XACC (only low 40 bits used)
// Parameters: [current XACC (passthru, only low 40 bits used), qx, qy]
// Migration: Explicit state passing (no IntrHasSideEffects)
// Note: No ClangBuiltin - builtin signature (void with 6 params including pointers) doesn't match intrinsic signature ({i32, i8} with 4 params)
// CodeGen is handled manually in riscv_espv_builtin_cg.inc
// ESP.VMULAS.U8.XACC (_m version) - Mixed model: xacc_h and xacc_l are both i32
// Returns: struct {i32 (XACC[31:0]), i32 (XACC[39:32], only low 8 bits valid)}
def int_riscv_esp_vmulas_u8_xacc :
	Intrinsic<[llvm_i32_ty, llvm_i32_ty],
              [llvm_i32_ty,                         
               llvm_i32_ty,                         
               llvm_v16i8_ty,                       
               llvm_v16i8_ty, llvm_i32_ty],
              [IntrNoMem, ImmArg<ArgIndex<4>>]>;                         

// ESP.VMULAS.U8.XACC.LD.IP (_m version) - Mixed model: XACC as {i32 low, i32 high}
// Returns: qu vector + updated pointer + new XACC as {i32 low, i32 high}
// Parameters: [xacc_low_in (i32), xacc_high_in (i32, only low 8 bits valid), qx, qy, pointer, offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_vmulas_u8_xacc_ld_ip :
	Intrinsic<[llvm_v16i8_ty, llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i32_ty,                                            
               llvm_i32_ty,                                            
               llvm_v16i8_ty,                                          
               llvm_v16i8_ty,                                          
               llvm_ptr_ty,                                            
               llvm_i32_ty, llvm_i32_ty],
              [IntrReadMem, IntrArgMemOnly, ImmArg<ArgIndex<6>>]>;                          

// ESP.VMULAS.U8.XACC.LD.XP (_m version) - Mixed model: XACC as {i32 low, i32 high}
// Returns: qu vector + updated pointer + new XACC as {i32 low, i32 high}
// Parameters: [xacc_low_in (i32), xacc_high_in (i32, only low 8 bits valid), qx, qy, pointer, register offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_vmulas_u8_xacc_ld_xp :
	Intrinsic<[llvm_v16i8_ty, llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i32_ty,                                            
               llvm_i32_ty,                                            
               llvm_v16i8_ty,                                          
               llvm_v16i8_ty,                                          
               llvm_ptr_ty,                                            
               llvm_i32_ty, llvm_i32_ty],
              [IntrReadMem, IntrArgMemOnly, ImmArg<ArgIndex<6>>]>;                          

// ESP.VMULAS.U8.XACC.ST.IP (_m version) - Mixed model: XACC as {i32 low, i32 high}
// Returns: updated pointer + new XACC as {i32 low, i32 high}
// Parameters: [xacc_low_in (i32), xacc_high_in (i32, only low 8 bits valid), qu, qx, qy, pointer, immediate offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_vmulas_u8_xacc_st_ip :
	Intrinsic<[llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i32_ty,                             
               llvm_i32_ty,                             
               llvm_v16i8_ty,                           
               llvm_v16i8_ty,                           
               llvm_v16i8_ty,                           
               llvm_ptr_ty,                             
               llvm_i32_ty, llvm_i32_ty],
              [IntrWriteMem, IntrArgMemOnly, ImmArg<ArgIndex<7>>]>;         

// ESP.VMULAS.U8.XACC.ST.XP (_m version) - Mixed model: XACC as {i32 low, i32 high}
// Returns: updated pointer + new XACC as {i32 low, i32 high}
// Parameters: [xacc_low_in (i32), xacc_high_in (i32, only low 8 bits valid), qu, qx, qy, pointer, register offset]
// Migration: Explicit state passing (no IntrHasSideEffects)
def int_riscv_esp_vmulas_u8_xacc_st_xp :
	Intrinsic<[llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty],
              [llvm_i32_ty,                             
               llvm_i32_ty,                             
               llvm_v16i8_ty,                           
               llvm_v16i8_ty,                           
               llvm_v16i8_ty,                           
               llvm_ptr_ty,                             
               llvm_i32_ty, llvm_i32_ty],
              [IntrWriteMem, IntrArgMemOnly, ImmArg<ArgIndex<7>>]>;          

// ESP.SRS.S.XACC (unified 2.2 API)
// Returns: saturated value + new XACC {i32 xacc_h (only low 8 bits valid), i32 xacc_l}
def int_riscv_esp_srs_s_xacc : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                                          [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                                           llvm_i32_ty, llvm_i32_ty],
                                          [IntrNoMem, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

// ESP.SRS.U.XACC (unified 2.2 API)
def int_riscv_esp_srs_u_xacc : Intrinsic<[llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                                          [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty,
                                           llvm_i32_ty, llvm_i32_ty],
                                          [IntrNoMem, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

// ESP.VMULAS.S16.QACC.LDBC.INCP (_m version)
def int_riscv_esp_vmulas_s16_qacc_ldbc_incp :
	Intrinsic<[llvm_v16i8_ty, llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,              
               llvm_v8i16_ty,                                                              
               llvm_v8i16_ty,                                                              
               llvm_ptr_ty, llvm_i32_ty],
              [IntrReadMem, IntrArgMemOnly, ImmArg<ArgIndex<7>>]>;

// ESP.VMULAS.S8.QACC.LDBC.INCP (_m version)
def int_riscv_esp_vmulas_s8_qacc_ldbc_incp :
	Intrinsic<[llvm_v16i8_ty, llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,              
               llvm_v16i8_ty,                                                              
               llvm_v16i8_ty,                                                              
               llvm_ptr_ty, llvm_i32_ty],
              [IntrReadMem, IntrArgMemOnly, ImmArg<ArgIndex<7>>]>;

// ESP.VMULAS.U16.QACC.LDBC.INCP (_m version)
def int_riscv_esp_vmulas_u16_qacc_ldbc_incp :
	Intrinsic<[llvm_v16i8_ty, llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,              
               llvm_v8i16_ty,                                                              
               llvm_v8i16_ty,                                                              
               llvm_ptr_ty, llvm_i32_ty],
              [IntrReadMem, IntrArgMemOnly, ImmArg<ArgIndex<7>>]>;

// ESP.VMULAS.U8.QACC.LDBC.INCP (_m version)
def int_riscv_esp_vmulas_u8_qacc_ldbc_incp :
	Intrinsic<[llvm_v16i8_ty, llvm_ptr_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty],
              [llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty, llvm_v16i8_ty,              
               llvm_v16i8_ty,                                                              
               llvm_v16i8_ty,                                                              
               llvm_ptr_ty, llvm_i32_ty],
              [IntrReadMem, IntrArgMemOnly, ImmArg<ArgIndex<7>>]>;


// SLCI.2Q - Shift Left with Immediate, 2 Quad registers
// Shifts concatenated qy and qw left, updates both registers in-place
// Returns modified values, input and output registers are the same due to constraints
def int_riscv_esp_slci_2q : Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty], 
                                           [llvm_v16i8_ty, llvm_v16i8_ty, llvm_i32_ty], 
                                           [IntrNoMem, ImmArg<ArgIndex<2>>]>;

// SRCI.2Q - Shift Right with Immediate, 2 Quad registers
// Shifts concatenated qy and qw right, updates both registers in-place
// Returns modified values, input and output registers are the same due to constraints
def int_riscv_esp_srci_2q : Intrinsic<[llvm_v16i8_ty, llvm_v16i8_ty], 
                                           [llvm_v16i8_ty, llvm_v16i8_ty, llvm_i32_ty], 
                                           [IntrNoMem, ImmArg<ArgIndex<2>>]>;

}
