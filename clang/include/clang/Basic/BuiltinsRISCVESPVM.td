
let Features = "xespv", Attributes = [NoThrow, RequiredVectorWidth<128>] in
{
  // Load function (Internal) - Immediate offset
  // 1. Must use C language full names (void, int, char)
  // 2. Must use _Vector<count, type> to represent vectors
  // 3. Important: Must write "void const *" (East Const), cannot write "const void *"
  // 4. Output parameters: Use void * to point to vector data (16-byte aligned)
  def esp_vld_128_ip_m : RISCVBuiltin<"void*(void const *, int, void *)">;

  def esp_ld_128_usar_ip_m : RISCVBuiltin<"void*(void const *, int, void *, unsigned int *)">;

  // ESP.LD.UA.STATE.IP (_m version) - Load 128-bit data to UA_STATE[127:0]
  // ESP.LD.UA.STATE.IP (_m version) - Load 128-bit data to UA_STATE register
  // Builtin signature: void*(void const *, int, void *)
  // Parameters: (ptr, offset, ua_state_out)
  // Returns: void* (updated pointer)
  // Note: UA_STATE value is returned through output parameter, similar to esp_vld_128_ip_m
  def esp_ld_ua_state_ip_m : RISCVBuiltin<"void*(void const *, int, void *)">;

  // Load function (Internal) - Register offset
  def esp_vld_128_xp_m : RISCVBuiltin<"void*(void const *, int, void *)">;
  
  def esp_ld_128_usar_xp_m : RISCVBuiltin<"void*(void const *, int, void *, unsigned int *)">;
  
  // Store function - Immediate offset
  def esp_vst_128_ip_m : RISCVBuiltin<"void*(_Vector<16, char>, void *, int)">;
  
  // Store function - Register offset
  def esp_vst_128_xp_m : RISCVBuiltin<"void*(_Vector<16, char>, void *, int)">;

  // Load H.64 function (Internal) - Immediate offset
  // Load 64 bits to upper 64-bit segment of qu (qu[127:64])
  def esp_vld_h_64_ip_m : RISCVBuiltin<"void*(void const *, int, void *)">;
  
  // Load H.64 function (Internal) - Register offset
  def esp_vld_h_64_xp_m : RISCVBuiltin<"void*(void const *, int, void *)">;
  
  // Load L.64 function (Internal) - Immediate offset
  // Load 64 bits to lower 64-bit segment of qu (qu[63:0])
  def esp_vld_l_64_ip_m : RISCVBuiltin<"void*(void const *, int, void *)">;
  
  // Load L.64 function (Internal) - Register offset
  def esp_vld_l_64_xp_m : RISCVBuiltin<"void*(void const *, int, void *)">;
  
  // Store H.64 function - Immediate offset
  // Store upper 64-bit segment of qu (qu[127:64])
  def esp_vst_h_64_ip_m : RISCVBuiltin<"void*(_Vector<8, char>, void *, int)">;
  
  // Store H.64 function - Register offset
  def esp_vst_h_64_xp_m : RISCVBuiltin<"void*(_Vector<8, char>, void *, int)">;
  
  // Store L.64 function - Immediate offset
  // Store lower 64-bit segment of qu (qu[63:0])
  def esp_vst_l_64_ip_m : RISCVBuiltin<"void*(_Vector<8, char>, void *, int)">;
  
  // Store L.64 function - Register offset
  def esp_vst_l_64_xp_m : RISCVBuiltin<"void*(_Vector<8, char>, void *, int)">;

  // VLDBC - Vector Load Broadcast
  // Broadcasts a single value to fill the entire vector register
  // Returns updated pointer, vector passed via pointer parameter
  def esp_vldbc_16_ip_m : RISCVBuiltin<"void*(void const *, int, void *)">;
  def esp_vldbc_16_xp_m : RISCVBuiltin<"void*(void const *, int, void *)">;
  def esp_vldbc_32_ip_m : RISCVBuiltin<"void*(void const *, int, void *)">;
  def esp_vldbc_32_xp_m : RISCVBuiltin<"void*(void const *, int, void *)">;
  def esp_vldbc_8_ip_m : RISCVBuiltin<"void*(void const *, int, void *)">;
  def esp_vldbc_8_xp_m : RISCVBuiltin<"void*(void const *, int, void *)">;

  // VLDEXT - Vector Load and Extend
  // Loads data and extends it (sign or zero extend)
  // Has 2 outputs (qu, qz) and returns updated pointer
  def esp_vldext_s16_ip_m : RISCVBuiltin<"void*(void const *, int, void *, void *)">;
  def esp_vldext_s16_xp_m : RISCVBuiltin<"void*(void const *, int, void *, void *)">;
  def esp_vldext_s8_ip_m : RISCVBuiltin<"void*(void const *, int, void *, void *)">;
  def esp_vldext_s8_xp_m : RISCVBuiltin<"void*(void const *, int, void *, void *)">;
  def esp_vldext_u16_ip_m : RISCVBuiltin<"void*(void const *, int, void *, void *)">;
  def esp_vldext_u16_xp_m : RISCVBuiltin<"void*(void const *, int, void *, void *)">;
  def esp_vldext_u8_ip_m : RISCVBuiltin<"void*(void const *, int, void *, void *)">;
  def esp_vldext_u8_xp_m : RISCVBuiltin<"void*(void const *, int, void *, void *)">;

  // VSADDS function - Signed 16-bit
  // Performs vector saturated addition on 16-bit signed data
  def esp_vsadds_s16_m : RISCVBuiltin<"_Vector<8, short>(_Vector<8, short>, int)">;

  // VSADDS function - Signed 8-bit
  // Performs vector saturated addition on 8-bit signed data
  def esp_vsadds_s8_m : RISCVBuiltin<"_Vector<16, char>(_Vector<16, char>, int)">;

  // VSADDS function - Unsigned 16-bit
  // Performs vector saturated addition on 16-bit unsigned data
  def esp_vsadds_u16_m : RISCVBuiltin<"_Vector<8, short>(_Vector<8, short>, int)">;

  // VSADDS function - Unsigned 8-bit
  // Performs vector saturated addition on 8-bit unsigned data
  def esp_vsadds_u8_m : RISCVBuiltin<"_Vector<16, char>(_Vector<16, char>, int)">;

  // VSSUBS function - Signed 16-bit
  // Performs vector saturated subtraction on 16-bit signed data
  def esp_vssubs_s16_m : RISCVBuiltin<"_Vector<8, short>(_Vector<8, short>, int)">;

  // VSSUBS function - Signed 8-bit
  // Performs vector saturated subtraction on 8-bit signed data
  def esp_vssubs_s8_m : RISCVBuiltin<"_Vector<16, char>(_Vector<16, char>, int)">;

  // VSSUBS function - Unsigned 16-bit
  // Performs vector saturated subtraction on 16-bit unsigned data
  def esp_vssubs_u16_m : RISCVBuiltin<"_Vector<8, short>(_Vector<8, short>, int)">;

  // VSSUBS function - Unsigned 8-bit
  // Performs vector saturated subtraction on 8-bit unsigned data
  def esp_vssubs_u8_m : RISCVBuiltin<"_Vector<16, char>(_Vector<16, char>, int)">;

  // ESP.VADD.S16.LD.INCP (_m version)
  // ESP.VADD.S16.LD.INCP (_m version)
  // Output parameters: void *qv_out, void *qu_out
  def esp_vadd_s16_ld_incp_m : RISCVBuiltin<"void*(_Vector<8, short>, _Vector<8, short>, void const *, void *, void *)">;
  
  // ESP.VADD.S16.ST.INCP (_m version)
  def esp_vadd_s16_st_incp_m : RISCVBuiltin<"void*(_Vector<8, short>, _Vector<8, short>, _Vector<16, char>, void *, _Vector<8, short>)">;
  
  // ESP.VADD.S8.LD.INCP (_m version)
  // Output parameters: void *qv_out, void *qu_out (similar to esp_cmul and esp_vld_128_ip_m)
  def esp_vadd_s8_ld_incp_m : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, void const *, void *, void *)">;
  
  // ESP.VADD.S8.ST.INCP (_m version)
  def esp_vadd_s8_st_incp_m : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, void *, _Vector<16, char>)">;
  
  // ESP.VADD.U16.LD.INCP (_m version)
  // Output parameters: void *qv_out, void *qu_out
  def esp_vadd_u16_ld_incp_m : RISCVBuiltin<"void*(_Vector<8, short>, _Vector<8, short>, void const *, void *, void *)">;
  
  // ESP.VADD.U16.ST.INCP (_m version)
  def esp_vadd_u16_st_incp_m : RISCVBuiltin<"void*(_Vector<8, short>, _Vector<8, short>, _Vector<16, char>, void *, _Vector<8, short>)">;
  
  // ESP.VADD.U8.LD.INCP (_m version)
  // Output parameters: void *qv_out, void *qu_out
  def esp_vadd_u8_ld_incp_m : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, void const *, void *, void *)">;
  
  // ESP.VADD.U8.ST.INCP (_m version)
  def esp_vadd_u8_st_incp_m : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, void *, _Vector<16, char>)">;

  // ANDQ function - Bitwise AND operation
  // Performs bitwise AND operation on 128-bit vector
  def esp_andq_m : RISCVBuiltin<"_Vector<16, char>(_Vector<16, char>, _Vector<16, char>)">;

  // NOTQ function - Bitwise NOT operation
  // Performs bitwise NOT operation on 128-bit vector
  def esp_notq_m : RISCVBuiltin<"_Vector<16, char>(_Vector<16, char>)">;

  // ORQ function - Bitwise OR operation
  // Performs bitwise OR operation on 128-bit vector
  def esp_orq_m : RISCVBuiltin<"_Vector<16, char>(_Vector<16, char>, _Vector<16, char>)">;

  // XORQ function - Bitwise XOR operation
  // Performs bitwise XOR operation on 128-bit vector
  def esp_xorq_m : RISCVBuiltin<"_Vector<16, char>(_Vector<16, char>, _Vector<16, char>)">;

  // ZERO.Q - Zero vector register
  // Sets the vector register to zero
  def esp_zero_q_m : RISCVBuiltin<"_Vector<16, char>()">;

  // ZERO.QACC - Zero QACC accumulator
  // Explicit state passing: Returns v64i8 (QACC_H + QACC_L, both zero)
  // ESP.ZERO.QACC (_m version) - Zero QACC_H and QACC_L with explicit state passing
  // Builtin signature: void(void *, void *, void *, void *) - Output 4x128-bit QACC via pointers
  def esp_zero_qacc_m : RISCVBuiltin<"void(void *, void *, void *, void *)">;

  // ESP.VCMULAS.S16.QACC.H (_m version)
  // Explicit state passing: Input QACC_H (2x128-bit), Output QACC_H (2x128-bit) via pointers
  // Builtin signature: void(_Vector<16, char>, _Vector<16, char>, _Vector<8, short>, _Vector<8, short>, void *, void *)
  def esp_vcmulas_s16_qacc_h_m : RISCVBuiltin<"void(_Vector<16, char>, _Vector<16, char>, _Vector<8, short>, _Vector<8, short>, void *, void *)">;
  
  // ESP.VCMULAS.S16.QACC.L (_m version)
  // Explicit state passing: Input QACC_L (2x128-bit), Output QACC_L (2x128-bit) via pointers
  // Builtin signature: void(_Vector<16, char>, _Vector<16, char>, _Vector<8, short>, _Vector<8, short>, void *, void *)
  def esp_vcmulas_s16_qacc_l_m : RISCVBuiltin<"void(_Vector<16, char>, _Vector<16, char>, _Vector<8, short>, _Vector<8, short>, void *, void *)">;

  // ESP.ST.QACC.H.H.128.IP (_m version) - Store QACC_H[255:128] to memory
  // Accept v16i8 (128-bit) to match hardware operation and simplify usage.
  def esp_st_qacc_h_h_128_ip_m : RISCVBuiltin<"void*(_Vector<16, char>, void *, int)">;

  // ESP.ST.QACC.H.L.128.IP (_m version) - Store QACC_H[127:0] to memory
  // Accept v16i8 (128-bit) to match hardware operation and simplify usage.
  def esp_st_qacc_h_l_128_ip_m : RISCVBuiltin<"void*(_Vector<16, char>, void *, int)">;

  // ESP.ST.QACC.L.H.128.IP (_m version) - Store QACC_L[255:128] to memory
  // Accept v16i8 (128-bit) to match hardware operation and simplify usage.
  def esp_st_qacc_l_h_128_ip_m : RISCVBuiltin<"void*(_Vector<16, char>, void *, int)">;

  // ESP.ST.QACC.L.L.128.IP (_m version) - Store QACC_L[127:0] to memory
  // Accept v16i8 (128-bit) to match hardware operation and simplify usage.
  def esp_st_qacc_l_l_128_ip_m : RISCVBuiltin<"void*(_Vector<16, char>, void *, int)">;

  // VCLAMP function - Signed 16-bit
  // Performs clamping operation on 16-bit signed data, sel16 specifies the range
  def esp_vclamp_s16_m : RISCVBuiltin<"_Vector<8, short>(_Vector<8, short>, int)">;

  // VPRELU function - Signed 16-bit
  // If segment value of qx <= 0, then qz = (qx * qy) >> rs1[5:0], otherwise qz = qx
  def esp_vprelu_s16_m : RISCVBuiltin<"_Vector<8, short>(_Vector<8, short>, _Vector<8, short>, int)">;

  // VPRELU function - Signed 8-bit
  // If segment value of qx <= 0, then qz = (qx * qy) >> rs1[4:0], otherwise qz = qx
  def esp_vprelu_s8_m : RISCVBuiltin<"_Vector<16, char>(_Vector<16, char>, _Vector<16, char>, int)">;

  // ST.S.XACC.IP - Store Signed XACC with Immediate Post-increment
  // Sign-extends XACC to 64-bit and stores to memory, updates pointer
  // Returns updated pointer
  // ESP.ST.S.XACC.IP (_m version) - Mixed model: XACC as {unsigned int low, unsigned int high}
  // Explicit state passing: Input XACC as two i32, Output {ptr, XACC unchanged as two i32}
  // Builtin signature: void*(unsigned int xacc_low_in, unsigned int xacc_high_in, void *ptr, int offset, void *ptr_out, unsigned int *xacc_low_out, unsigned int *xacc_high_out)
  // Returns pointer, stores XACC (unchanged) to output parameters
  // Note: xacc_high_in and xacc_high_out are i32 but only low 8 bits are valid (XACC[39:32])
  def esp_st_s_xacc_ip_m : RISCVBuiltin<"void*(unsigned int, unsigned int, void *, int, void *, unsigned int *, unsigned int *)">;

  // ESP.CMUL.S16 (_m version)
  // Note: qz_in is a passthru operand for preserving unmodified bits (RMW semantics).
  // SAR is passed explicitly as unsigned int (i32) for explicit state passing
  def esp_cmul_s16_m : RISCVBuiltin<"_Vector<8, short>(_Vector<8, short>, _Vector<8, short>, _Vector<8, short>, int, unsigned int)">;
  
  // ESP.CMUL.S16.LD.INCP (_m version)
  // Note: qz_in is a passthru operand for preserving unmodified bits (RMW semantics).
  // Output parameters: void *qz_out, void *qu_out (similar to esp_vld_128_ip_m)
  // SAR is passed explicitly as unsigned int (i32) for explicit state passing
  def esp_cmul_s16_ld_incp_m : RISCVBuiltin<"void*(_Vector<8, short>, _Vector<8, short>, _Vector<8, short>, void const *, int, void *, void *, unsigned int)">;
  
  // ESP.CMUL.S16.ST.INCP (_m version)
  // Note: qz_in is a passthru operand for preserving unmodified bits (RMW semantics).
  // SAR is passed explicitly as unsigned int (i32) for explicit state passing
  def esp_cmul_s16_st_incp_m : RISCVBuiltin<"void*(_Vector<8, short>, _Vector<8, short>, _Vector<8, short>, _Vector<16, char>, void *, int, unsigned int)">;
  
  // ESP.CMUL.S8.LD.INCP (_m version)
  // Note: qz_in is a passthru operand for preserving unmodified bits (RMW semantics).
  // Output parameters: void *qz_out, void *qu_out (similar to esp_vld_128_ip_m)
  // SAR is passed explicitly as unsigned int (i32) for explicit state passing
  def esp_cmul_s8_ld_incp_m : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, void const *, int, void *, void *, unsigned int)">;
  
  // ESP.CMUL.S8.ST.INCP (_m version)
  // Note: qz_in is a passthru operand for preserving unmodified bits (RMW semantics).
  // SAR is passed explicitly as unsigned int (i32) for explicit state passing
  def esp_cmul_s8_st_incp_m : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, void *, int, unsigned int)">;
  
  // ESP.CMUL.U16.LD.INCP (_m version)
  // Note: qz_in is a passthru operand for preserving unmodified bits (RMW semantics).
  // Output parameters: void *qz_out, void *qu_out (similar to esp_vld_128_ip_m)
  // SAR is passed explicitly as unsigned int (i32) for explicit state passing
  def esp_cmul_u16_ld_incp_m : RISCVBuiltin<"void*(_Vector<8, short>, _Vector<8, short>, _Vector<8, short>, void const *, int, void *, void *, unsigned int)">;
  
  // ESP.CMUL.U16.ST.INCP (_m version)
  // Note: qz_in is a passthru operand for preserving unmodified bits (RMW semantics).
  // SAR is passed explicitly as unsigned int (i32) for explicit state passing
  def esp_cmul_u16_st_incp_m : RISCVBuiltin<"void*(_Vector<8, short>, _Vector<8, short>, _Vector<8, short>, _Vector<16, char>, void *, int, unsigned int)">;
  
  // ESP.CMUL.U8.LD.INCP (_m version)
  // Note: qz_in is a passthru operand for preserving unmodified bits (RMW semantics).
  // Output parameters: void *qz_out, void *qu_out (similar to esp_vld_128_ip_m)
  // SAR is passed explicitly as unsigned int (i32) for explicit state passing
  def esp_cmul_u8_ld_incp_m : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, void const *, int, void *, void *, unsigned int)">;
  
  // ESP.CMUL.U8.ST.INCP (_m version)
  // Note: qz_in is a passthru operand for preserving unmodified bits (RMW semantics).
  // SAR is passed explicitly as unsigned int (i32) for explicit state passing
  def esp_cmul_u8_st_incp_m : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, void *, int, unsigned int)">;

  // ESP.MOVX.W.SAR (_m version)
  // Returns SAR (32-bit unsigned, only low 6 bits used, range 0-63) for explicit state passing (chaining)
  def esp_movx_w_sar_m : RISCVBuiltin<"unsigned int(unsigned int)">;
  
  // ESP.MOVX.W.SAR_BYTES (_m version)
  // Returns SAR_BYTES (32-bit unsigned, only low 4 bits used, range 0-15) for explicit state passing (chaining)
  def esp_movx_w_sar_bytes_m : RISCVBuiltin<"unsigned int(unsigned int)">;
  
  // ESP.MOVX.R.SAR (_m version)
  // Takes SAR (32-bit unsigned, only low 6 bits used, range 0-63) as input, returns 32-bit unsigned
  def esp_movx_r_sar_m : RISCVBuiltin<"unsigned int(unsigned int)">;

  // FFT.AMS.S16.LD.INCP.UAUP (_m version) - FFT AMS with UA update and explicit state passing
  // Has 3 inputs (qx, qy, qw) and 3 outputs (qu, qz, qv)
  // UA_STATE is input/output (phantom operand), SAR_BYTES and SAR are input-only (phantom operands)
  // Returns updated pointer, outputs passed via pointer parameters
  // Parameters: (qx, qy, qw, rs1, sel2, qu_out, qz_out, qv_out, ua_state_in, ua_state_out, sar_bytes_in, sar_in)
  def esp_fft_ams_s16_ld_incp_uaup_m : RISCVBuiltin<"void*(_Vector<8, short>, _Vector<8, short>, _Vector<8, short>, void const *, int, void *, void *, void *, _Vector<16, char>, void *, unsigned int, unsigned int)">;

  // FFT.R2BF.S16 - FFT Radix-2 Butterfly operation
  // Has 2 outputs (qz, qv) and 3 inputs (qx, qy, sel2)
  // For _m version, return a struct containing both output vectors
  def esp_fft_r2bf_s16_m : RISCVBuiltin<"void(_Vector<8, short>, _Vector<8, short>, int, void *, void *)">;

  // FFT.AMS.S16.LD.INCP - FFT Add-Multiply-Subtract with load and increment
  // Has 3 inputs (qx, qy, qw) and 3 outputs (qu, qz, qv)
  // Returns updated pointer, outputs passed via pointer parameters
  // FFT.AMS.S16.LD.INCP (_m version) - FFT AMS with explicit SAR state passing
  // SAR is passed explicitly as unsigned int (i32) for explicit state passing
  def esp_fft_ams_s16_ld_incp_m : RISCVBuiltin<"void*(_Vector<8, short>, _Vector<8, short>, _Vector<8, short>, void const *, int, void *, void *, void *, unsigned int)">;

  // FFT.R2BF.S16.ST.INCP - FFT Radix-2 Butterfly with store and increment
  // Has 2 inputs (qx, qy) and 1 output (qz)
  // Returns updated pointer, qz passed via pointer parameter
  def esp_fft_r2bf_s16_st_incp_m : RISCVBuiltin<"void*(_Vector<8, short>, _Vector<8, short>, void *, int, void *)">;

  // FFT.AMS.S16.LD.R32.DECP - FFT AMS with reverse 32-bit load and decrement
  // Has 3 inputs (qx, qy, qw) and 3 outputs (qu, qz, qv)
  // Returns updated pointer (rs1 - 16), outputs passed via pointer parameters
  // FFT.AMS.S16.LD.R32.DECP (_m version) - FFT AMS with explicit SAR state passing
  // SAR is passed explicitly as unsigned int (i32) for explicit state passing
  def esp_fft_ams_s16_ld_r32_decp_m : RISCVBuiltin<"void*(_Vector<8, short>, _Vector<8, short>, _Vector<8, short>, void const *, int, void *, void *, void *, unsigned int)">;

  // FFT.AMS.S16.ST.INCP (_m version) - FFT AMS with store and increment and explicit SAR state passing
  // Has 4 inputs (qx, qy, qw, qu) and 1 output (qz)
  // rs2 is updated with computation result
  // Returns updated pointer (rs1 + 16), qz passed via pointer parameter
  // SAR is passed explicitly as unsigned int (i32) for explicit state passing
  def esp_fft_ams_s16_st_incp_m : RISCVBuiltin<"void*(_Vector<8, short>, _Vector<8, short>, _Vector<8, short>, _Vector<16, char>, void *, void *, void *, int, unsigned int)">;

  // FFT.BITREV - Bit reversal operation
  // Has 0 inputs and 1 output (qv)
  // Returns updated pointer (rs1 + 8), qv passed via pointer parameter
  // ESP.FFT.BITREV (_m version) - Bit reversal operation with explicit FFT_BIT_WIDTH state passing
  // Parameters: (rs1, qv_out, fft_bit_width)
  // rs1: pointer to source data
  // qv_out: output pointer for qv vector
  // fft_bit_width: FFT_BIT_WIDTH value (unsigned int, i32, only low 4 bits used, range 0-15)
  // Returns: updated pointer (rs1 + 8)
  // Note: FFT_BIT_WIDTH is passed explicitly for explicit state passing
  def esp_fft_bitrev_m : RISCVBuiltin<"void*(void *, void *, unsigned int)">;

  // FFT.CMUL.S16.LD.XP (_m version) - Complex multiply with load and register offset and explicit SAR state passing
  // Has 2 inputs (qx, qy) and 2 outputs (qz, qu)
  // Returns updated pointer (rs1 + rs2), outputs passed via pointer parameters
  // SAR is passed explicitly as unsigned int (i32) for explicit state passing
  def esp_fft_cmul_s16_ld_xp_m : RISCVBuiltin<"void*(_Vector<8, short>, _Vector<8, short>, void const *, void *, int, void *, void *, unsigned int)">;

  // FFT.CMUL.S16.ST.XP (_m version) - Complex multiply with store and register offset and explicit SAR state passing
  // Has 3 inputs (qx, qy, qu) and 0 outputs
  // Returns updated pointer (rs1 + rs2)
  // SAR is passed explicitly as unsigned int (i32) for explicit state passing
  def esp_fft_cmul_s16_st_xp_m : RISCVBuiltin<"void*(_Vector<8, short>, _Vector<8, short>, _Vector<16, char>, void *, void *, int, int, int, unsigned int)">;

  // FFT.VST.R32.DECP - Vector store with reverse 32-bit order and decrement
  // Has 1 input (qu) and 0 outputs
  // Returns updated pointer (rs1 - 16)
  def esp_fft_vst_r32_decp_m : RISCVBuiltin<"void*(_Vector<16, char>, void *, int)">;

}
