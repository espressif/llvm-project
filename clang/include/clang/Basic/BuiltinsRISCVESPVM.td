
let Features = "xespv", Attributes = [NoThrow, RequiredVectorWidth<128>] in
{
  // Load function (Internal) - Immediate offset
  // 1. Must use C language full names (void, int, char)
  // 2. Must use _Vector<count, type> to represent vectors
  // 3. Important: Must write "void const *" (East Const), cannot write "const void *"
  // 4. Output parameters: Use void * to point to vector data (16-byte aligned)
  def esp_vld_128_ip_m : RISCVBuiltin<"void*(void const *, int, void *)">;

  def esp_ld_128_usar_ip_m : RISCVBuiltin<"void*(void const *, int, void *, unsigned int *)">;

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

}
