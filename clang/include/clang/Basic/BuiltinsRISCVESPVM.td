
let Features = "xespv2p1|xespv", Attributes = [NoThrow, RequiredVectorWidth<128>] in
{
  // ESP.VCMULAS.S16.QACC.H (_m version)
  // Explicit state passing: Input QACC_H (2x128-bit), Output QACC_H (2x128-bit) via pointers
  // Input: v2 (QACC_H[127:0]), v3 (QACC_H[255:128]), qx, qy
  // Output: 2x128-bit QACC_H via pointers
  def esp_vcmulas_s16_qacc_h : RISCVBuiltin<"void(_Vector<16, char>, _Vector<16, char>, _Vector<8, short>, _Vector<8, short>, void *, void *, int)">;
  
  // ESP.VCMULAS.S16.QACC.L (_m version)
  // Explicit state passing: Input QACC_L (2x128-bit), Output QACC_L (2x128-bit) via pointers
  // Input: v0 (QACC_L[127:0]), v1 (QACC_L[255:128]), qx, qy
  // Output: 2x128-bit QACC_L via pointers
  def esp_vcmulas_s16_qacc_l : RISCVBuiltin<"void(_Vector<16, char>, _Vector<16, char>, _Vector<8, short>, _Vector<8, short>, void *, void *, int)">;
  
  // ESP.VCMULAS.S16.QACC.H.LD.IP (_m version)
  // Explicit state passing: Input QACC_H (2x128-bit), Output {void*, qu, new QACC_H (2x128-bit)}
  // Input: v2 (QACC_H[127:0]), v3 (QACC_H[255:128]), qx, qy, ptr, imm
  // Output: ptr (return value), qu (via pointer), 2x128-bit QACC_H via pointers (3 pointers total)
  def esp_vcmulas_s16_qacc_h_ld_ip : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, _Vector<8, short>, _Vector<8, short>, void const *, int, void *, void *, void *)">;
  
  // ESP.VCMULAS.S16.QACC.H.LD.XP (_m version)
  // Explicit state passing: Input QACC_H (2x128-bit), Output {void*, qu, new QACC_H (2x128-bit)}
  // Input: v2 (QACC_H[127:0]), v3 (QACC_H[255:128]), qx, qy, ptr, rs2
  // Output: ptr (return value), qu (via pointer), 2x128-bit QACC_H via pointers (3 pointers total)
  def esp_vcmulas_s16_qacc_h_ld_xp : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, _Vector<8, short>, _Vector<8, short>, void const *, int, void *, void *, void *)">;
  
  // ESP.VCMULAS.S16.QACC.L.LD.IP (_m version)
  // Explicit state passing: Input QACC_L (2x128-bit), Output {void*, qu, new QACC_L (2x128-bit)}
  // Input: v0 (QACC_L[127:0]), v1 (QACC_L[255:128]), qx, qy, ptr, imm
  // Output: ptr (return value), qu (via pointer), 2x128-bit QACC_L via pointers (3 pointers total)
  def esp_vcmulas_s16_qacc_l_ld_ip : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, _Vector<8, short>, _Vector<8, short>, void const *, int, void *, void *, void *)">;
  
  // ESP.VCMULAS.S16.QACC.L.LD.XP (_m version)
  // Explicit state passing: Input QACC_L (2x128-bit), Output {void*, qu, new QACC_L (2x128-bit)}
  // Input: v0 (QACC_L[127:0]), v1 (QACC_L[255:128]), qx, qy, ptr, rs2
  // Output: ptr (return value), qu (via pointer), 2x128-bit QACC_L via pointers (3 pointers total)
  def esp_vcmulas_s16_qacc_l_ld_xp : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, _Vector<8, short>, _Vector<8, short>, void const *, int, void *, void *, void *)">;
  
  // ESP.VCMULAS.S8.QACC.H (_m version)
  // Explicit state passing: Input QACC_H (2x128-bit), Output QACC_H (2x128-bit) via pointers
  // Input: v2 (QACC_H[127:0]), v3 (QACC_H[255:128]), qx, qy
  // Output: 2x128-bit QACC_H via pointers
  def esp_vcmulas_s8_qacc_h : RISCVBuiltin<"void(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, void *, void *, int)">;
  
  // ESP.VCMULAS.S8.QACC.H.LD.IP (_m version)
  // Explicit state passing: Input QACC_H (2x128-bit), Output {void*, qu, new QACC_H (2x128-bit)}
  // Input: v2 (QACC_H[127:0]), v3 (QACC_H[255:128]), qx, qy, ptr, imm
  // Output: ptr (return value), qu (via pointer), 2x128-bit QACC_H via pointers (3 pointers total)
  def esp_vcmulas_s8_qacc_h_ld_ip : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, void const *, int, void *, void *, void *)">;
  
  // ESP.VCMULAS.S8.QACC.H.LD.XP (_m version)
  // Explicit state passing: Input QACC_H (2x128-bit), Output {void*, qu, new QACC_H (2x128-bit)}
  // Input: v2 (QACC_H[127:0]), v3 (QACC_H[255:128]), qx, qy, ptr, rs2
  // Output: ptr (return value), qu (via pointer), 2x128-bit QACC_H via pointers (3 pointers total)
  def esp_vcmulas_s8_qacc_h_ld_xp : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, void const *, int, void *, void *, void *)">;
  
  // ESP.VCMULAS.S8.QACC.L (_m version)
  // Explicit state passing: Input QACC_L (2x128-bit), Output QACC_L (2x128-bit) via pointers
  // Input: v0 (QACC_L[127:0]), v1 (QACC_L[255:128]), qx, qy
  // Output: 2x128-bit QACC_L via pointers
  def esp_vcmulas_s8_qacc_l : RISCVBuiltin<"void(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, void *, void *, int)">;
  
  // ESP.VCMULAS.S8.QACC.L.LD.IP (_m version)
  // Explicit state passing: Input QACC_L (2x128-bit), Output {void*, qu, new QACC_L (2x128-bit)}
  // Input: v0 (QACC_L[127:0]), v1 (QACC_L[255:128]), qx, qy, ptr, imm
  // Output: ptr (return value), qu (via pointer), 2x128-bit QACC_L via pointers (3 pointers total)
  def esp_vcmulas_s8_qacc_l_ld_ip : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, void const *, int, void *, void *, void *)">;
  
  // ESP.VCMULAS.S8.QACC.L.LD.XP (_m version)
  // Explicit state passing: Input QACC_L (2x128-bit), Output {void*, qu, new QACC_L (2x128-bit)}
  // Input: v0 (QACC_L[127:0]), v1 (QACC_L[255:128]), qx, qy, ptr, rs2
  // Output: ptr (return value), qu (via pointer), 2x128-bit QACC_L via pointers (3 pointers total)
  def esp_vcmulas_s8_qacc_l_ld_xp : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, void const *, int, void *, void *, void *)">;
  
  // ESP.VMULAS.S16.QACC (_m version)
  // Explicit state passing: Input QACC (H+L), Output new QACC (H+L) via two pointers
  // ESP.VMULAS.S16.QACC (_m version)
  // Explicit state passing: Input QACC as 4x128-bit, Output new QACC as 4x128-bit via four pointers
  // Input: v0 (QACC_L[127:0]), v1 (QACC_L[255:128]), v2 (QACC_H[127:0]), v3 (QACC_H[255:128]), qx, qy
  // Output: 4x128-bit QACC via pointers
  def esp_vmulas_s16_qacc : RISCVBuiltin<"void(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<8, short>, _Vector<8, short>, void *, void *, void *, void *)">;
  
  // ESP.VMULAS.S16.QACC.LD.IP (_m version)
  // Explicit state passing: Input QACC as 4x128-bit, Output {void*, v16i8 qu, 4x128-bit QACC} via pointers
  // Input: v0, v1, v2, v3 (4x128-bit QACC passthru), qx, qy, ptr, imm
  // Output: ptr (return value), qu (via pointer), 4x128-bit QACC via pointers (5 pointers total)
  def esp_vmulas_s16_qacc_ld_ip : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<8, short>, _Vector<8, short>, void const *, int, void *, void *, void *, void *, void *)">;
  
  // ESP.VMULAS.S16.QACC.LD.XP (_m version)
  // Explicit state passing: Input QACC as 4x128-bit, Output {void*, v16i8 qu, 4x128-bit QACC} via pointers
  // Input: v0, v1, v2, v3 (4x128-bit QACC passthru), qx, qy, ptr, rs2
  // Output: ptr (return value), qu (via pointer), 4x128-bit QACC via pointers (5 pointers total)
  def esp_vmulas_s16_qacc_ld_xp : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<8, short>, _Vector<8, short>, void const *, int, void *, void *, void *, void *, void *)">;
  
  // ESP.VMULAS.S16.QACC.ST.IP (_m version)
  // Explicit state passing: Input QACC (H+L), Output {void*, new QACC (H+L)} via two pointers
  // Explicit state passing: Input QACC as 4x128-bit passthru, Output {void*, 4x128-bit QACC} via pointers
  def esp_vmulas_s16_qacc_st_ip : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<8, short>, _Vector<8, short>, void *, int, void *, void *, void *, void *)">;
  
  // ESP.VMULAS.S16.QACC.ST.XP (_m version)
  // Explicit state passing: Input QACC as 4x128-bit passthru, Output {void*, 4x128-bit QACC} via pointers
  def esp_vmulas_s16_qacc_st_xp : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<8, short>, _Vector<8, short>, void *, int, void *, void *, void *, void *)">;
  
  // ESP.VMULAS.S16.XACC (_m version)
  // Mixed model: Input XACC as struct {unsigned int low, unsigned char high}, Output new XACC as struct
  // Note: TableGen doesn't support anonymous struct, so we use void and build struct in CodeGen
  def esp_vmulas_s16_xacc : RISCVBuiltin<"void(unsigned int, unsigned int, _Vector<8, short>, _Vector<8, short>, unsigned int *, unsigned int *)">;  // xacc_h and xacc_l are both unsigned int (i32)
  
  // ESP.VMULAS.S16.XACC.LD.IP (_m version)
  // Explicit state passing: Input XACC, Output {qu (128-bit), ptr, new XACC}
  // Returns pointer, stores qu and xacc to output parameters
  // ESP.VMULAS.S16.XACC.LD.IP (_m version) - Mixed model: XACC as {unsigned int low, unsigned int high}
  // Explicit state passing: Input XACC as two i32, Output {qu (128-bit), ptr, new XACC as two i32}
  // Returns pointer, stores qu and xacc to output parameters
  // Note: xacc_high_in and xacc_high_out are i32 but only low 8 bits are valid (XACC[39:32])
  def esp_vmulas_s16_xacc_ld_ip : RISCVBuiltin<"void*(unsigned int, unsigned int, _Vector<8, short>, _Vector<8, short>, void const *, int, void *, unsigned int *, unsigned int *)">;
  
  // ESP.VMULAS.S16.XACC.LD.XP (_m version) - Mixed model: XACC as {unsigned int low, unsigned int high}
  // Explicit state passing: Input XACC as two i32, Output {qu (128-bit), ptr, new XACC as two i32}
  // Returns pointer, stores qu and xacc to output parameters
  // Note: xacc_high_in and xacc_high_out are i32 but only low 8 bits are valid (XACC[39:32])
  def esp_vmulas_s16_xacc_ld_xp : RISCVBuiltin<"void*(unsigned int, unsigned int, _Vector<8, short>, _Vector<8, short>, void const *, int, void *, unsigned int *, unsigned int *)">;
  
  // ESP.VMULAS.S16.XACC.ST.IP (_m version) - Mixed model: XACC as {unsigned int low, unsigned int high}
  // Explicit state passing: Input XACC as two i32, Output {void*, new XACC as two i32}
  // Note: xacc_high_in and xacc_high_out are i32 but only low 8 bits are valid (XACC[39:32])
  def esp_vmulas_s16_xacc_st_ip : RISCVBuiltin<"void*(unsigned int, unsigned int, _Vector<16, char>, _Vector<8, short>, _Vector<8, short>, void *, int, unsigned int *, unsigned int *)">;
  
  // ESP.VMULAS.S16.XACC.ST.XP (_m version) - Mixed model: XACC as {unsigned int low, unsigned int high}
  // Explicit state passing: Input XACC as two i32, Output {void*, new XACC as two i32}
  // Note: xacc_high_in and xacc_high_out are i32 but only low 8 bits are valid (XACC[39:32])
  def esp_vmulas_s16_xacc_st_xp : RISCVBuiltin<"void*(unsigned int, unsigned int, _Vector<16, char>, _Vector<8, short>, _Vector<8, short>, void *, int, unsigned int *, unsigned int *)">;
  
  // ESP.VMULAS.S8.QACC (_m version)
  // Explicit state passing: Input QACC (4x128-bit), Output QACC (4x128-bit) via pointers
  // Input: v0 (QACC_L[127:0]), v1 (QACC_L[255:128]), v2 (QACC_H[127:0]), v3 (QACC_H[255:128]), qx, qy
  // Output: 4x128-bit QACC via pointers
  def esp_vmulas_s8_qacc : RISCVBuiltin<"void(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, void *, void *, void *, void *)">;
  
  // ESP.VMULAS.S8.QACC.LD.IP (_m version)
  // Explicit state passing: Input QACC as 4x128-bit passthru, Output {void*, qu, 4x128-bit QACC} via pointers
  def esp_vmulas_s8_qacc_ld_ip : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, void const *, int, void *, void *, void *, void *, void *)">;
  
  // ESP.VMULAS.S8.QACC.LD.XP (_m version)
  // Explicit state passing: Input QACC as 4x128-bit, Output {void*, v16i8 qu, 4x128-bit QACC} via pointers
  // Input: v0, v1, v2, v3 (4x128-bit QACC passthru), qx, qy, ptr, rs2
  // Output: ptr (return value), qu (via pointer), 4x128-bit QACC via pointers (5 pointers total)
  def esp_vmulas_s8_qacc_ld_xp : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, void const *, int, void *, void *, void *, void *, void *)">;
  
  // ESP.VMULAS.S8.QACC.ST.IP (_m version)
  // Explicit state passing: Input QACC as 4x128-bit passthru, Output {void*, 4x128-bit QACC} via pointers
  def esp_vmulas_s8_qacc_st_ip : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, void *, int, void *, void *, void *, void *)">;
  
  // ESP.VMULAS.S8.QACC.ST.XP (_m version)
  // Explicit state passing: Input QACC as 4x128-bit passthru, Output {void*, 4x128-bit QACC} via pointers
  def esp_vmulas_s8_qacc_st_xp : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, void *, int, void *, void *, void *, void *)">;
  
  // ESP.VMULAS.S8.XACC (_m version)
  // Explicit state passing: Input XACC, Output new XACC
  def esp_vmulas_s8_xacc : RISCVBuiltin<"void(unsigned int, unsigned int, _Vector<16, char>, _Vector<16, char>, unsigned int *, unsigned int *)">;  // xacc_h and xacc_l are both unsigned int (i32)
  
  // ESP.VMULAS.S8.XACC.LD.IP (_m version)
  // Explicit state passing: Input XACC, Output {qu (128-bit), ptr, new XACC}
  // Returns pointer, stores qu and xacc to output parameters
  // ESP.VMULAS.S8.XACC.LD.IP (_m version) - Mixed model: XACC as {unsigned int low, unsigned int high}
  // Explicit state passing: Input XACC as two i32, Output {qu (128-bit), ptr, new XACC as two i32}
  // Note: xacc_high_in and xacc_high_out are i32 but only low 8 bits are valid (XACC[39:32])
  def esp_vmulas_s8_xacc_ld_ip : RISCVBuiltin<"void*(unsigned int, unsigned int, _Vector<16, char>, _Vector<16, char>, void const *, int, void *, unsigned int *, unsigned int *)">;
  
  // ESP.VMULAS.S8.XACC.LD.XP (_m version) - Mixed model: XACC as {unsigned int low, unsigned int high}
  // Explicit state passing: Input XACC as two i32, Output {qu (128-bit), ptr, new XACC as two i32}
  // Note: xacc_high_in and xacc_high_out are i32 but only low 8 bits are valid (XACC[39:32])
  def esp_vmulas_s8_xacc_ld_xp : RISCVBuiltin<"void*(unsigned int, unsigned int, _Vector<16, char>, _Vector<16, char>, void const *, int, void *, unsigned int *, unsigned int *)">;
  
  // ESP.VMULAS.S8.XACC.ST.IP (_m version) - Mixed model: XACC as {unsigned int low, unsigned int high}
  // Explicit state passing: Input XACC as two i32, Output {void*, new XACC as two i32}
  // Note: xacc_high_in and xacc_high_out are i32 but only low 8 bits are valid (XACC[39:32])
  def esp_vmulas_s8_xacc_st_ip : RISCVBuiltin<"void*(unsigned int, unsigned int, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, void *, int, unsigned int *, unsigned int *)">;
  
  // ESP.VMULAS.S8.XACC.ST.XP (_m version) - Mixed model: XACC as {unsigned int low, unsigned int high}
  // Explicit state passing: Input XACC as two i32, Output {void*, new XACC as two i32}
  // Note: xacc_high_in and xacc_high_out are i32 but only low 8 bits are valid (XACC[39:32])
  def esp_vmulas_s8_xacc_st_xp : RISCVBuiltin<"void*(unsigned int, unsigned int, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, void *, int, unsigned int *, unsigned int *)">;
  
  // ESP.VMULAS.U16.QACC (_m version)
  // Explicit state passing: Input QACC (4x128-bit), Output QACC (4x128-bit) via pointers
  // Input: v0 (QACC_L[127:0]), v1 (QACC_L[255:128]), v2 (QACC_H[127:0]), v3 (QACC_H[255:128]), qx, qy
  // Output: 4x128-bit QACC via pointers
  def esp_vmulas_u16_qacc : RISCVBuiltin<"void(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<8, short>, _Vector<8, short>, void *, void *, void *, void *)">;
  
  // ESP.VMULAS.U16.QACC.LD.IP (_m version)
  // Explicit state passing: Input QACC as 4x128-bit passthru, Output {void*, qu, 4x128-bit QACC} via pointers
  def esp_vmulas_u16_qacc_ld_ip : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<8, short>, _Vector<8, short>, void const *, int, void *, void *, void *, void *, void *)">;
  
  // ESP.VMULAS.U16.QACC.LD.XP (_m version)
  // Explicit state passing: Input QACC as 4x128-bit, Output {void*, v16i8 qu, 4x128-bit QACC} via pointers
  // Input: v0, v1, v2, v3 (4x128-bit QACC passthru), qx, qy, ptr, rs2
  // Output: ptr (return value), qu (via pointer), 4x128-bit QACC via pointers (5 pointers total)
  def esp_vmulas_u16_qacc_ld_xp : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<8, short>, _Vector<8, short>, void const *, int, void *, void *, void *, void *, void *)">;
  
  // ESP.VMULAS.U16.QACC.ST.IP (_m version)
  // Explicit state passing: Input QACC as 4x128-bit passthru, Output {void*, 4x128-bit QACC} via pointers
  def esp_vmulas_u16_qacc_st_ip : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<8, short>, _Vector<8, short>, void *, int, void *, void *, void *, void *)">;
  
  // ESP.VMULAS.U16.QACC.ST.XP (_m version)
  // Explicit state passing: Input QACC as 4x128-bit passthru, Output {void*, 4x128-bit QACC} via pointers
  def esp_vmulas_u16_qacc_st_xp : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<8, short>, _Vector<8, short>, void *, int, void *, void *, void *, void *)">;
  
  // ESP.VMULAS.U16.XACC (_m version)
  // Explicit state passing: Input XACC, Output new XACC
  def esp_vmulas_u16_xacc : RISCVBuiltin<"void(unsigned int, unsigned int, _Vector<8, short>, _Vector<8, short>, unsigned int *, unsigned int *)">;  // xacc_h and xacc_l are both unsigned int (i32)
  
  // ESP.VMULAS.U16.XACC.LD.IP (_m version)
  // Explicit state passing: Input XACC, Output {qu (128-bit), ptr, new XACC}
  // Returns pointer, stores qu and xacc to output parameters
  // ESP.VMULAS.U16.XACC.LD.IP (_m version) - Mixed model: XACC as {unsigned int low, unsigned int high}
  // Explicit state passing: Input XACC as two i32, Output {qu (128-bit), ptr, new XACC as two i32}
  // Note: xacc_high_in and xacc_high_out are i32 but only low 8 bits are valid (XACC[39:32])
  def esp_vmulas_u16_xacc_ld_ip : RISCVBuiltin<"void*(unsigned int, unsigned int, _Vector<8, short>, _Vector<8, short>, void const *, int, void *, unsigned int *, unsigned int *)">;
  
  // ESP.VMULAS.U16.XACC.LD.XP (_m version) - Mixed model: XACC as {unsigned int low, unsigned int high}
  // Explicit state passing: Input XACC as two i32, Output {qu (128-bit), ptr, new XACC as two i32}
  // Note: xacc_high_in and xacc_high_out are i32 but only low 8 bits are valid (XACC[39:32])
  def esp_vmulas_u16_xacc_ld_xp : RISCVBuiltin<"void*(unsigned int, unsigned int, _Vector<8, short>, _Vector<8, short>, void const *, int, void *, unsigned int *, unsigned int *)">;
  
  // ESP.VMULAS.U16.XACC.ST.IP (_m version) - Mixed model: XACC as {unsigned int low, unsigned int high}
  // Explicit state passing: Input XACC as two i32, Output {void*, new XACC as two i32}
  // Note: xacc_high_in and xacc_high_out are i32 but only low 8 bits are valid (XACC[39:32])
  def esp_vmulas_u16_xacc_st_ip : RISCVBuiltin<"void*(unsigned int, unsigned int, _Vector<16, char>, _Vector<8, short>, _Vector<8, short>, void *, int, unsigned int *, unsigned int *)">;
  
  // ESP.VMULAS.U16.XACC.ST.XP (_m version) - Mixed model: XACC as {unsigned int low, unsigned int high}
  // Explicit state passing: Input XACC as two i32, Output {void*, new XACC as two i32}
  // Note: xacc_high_in and xacc_high_out are i32 but only low 8 bits are valid (XACC[39:32])
  def esp_vmulas_u16_xacc_st_xp : RISCVBuiltin<"void*(unsigned int, unsigned int, _Vector<16, char>, _Vector<8, short>, _Vector<8, short>, void *, int, unsigned int *, unsigned int *)">;
  
  // ESP.VMULAS.U8.QACC (_m version)
  // Explicit state passing: Input QACC (4x128-bit), Output QACC (4x128-bit) via pointers
  // Input: v0 (QACC_L[127:0]), v1 (QACC_L[255:128]), v2 (QACC_H[127:0]), v3 (QACC_H[255:128]), qx, qy
  // Output: 4x128-bit QACC via pointers
  def esp_vmulas_u8_qacc : RISCVBuiltin<"void(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, void *, void *, void *, void *)">;
  
  // ESP.VMULAS.U8.QACC.LD.IP (_m version)
  // Explicit state passing: Input QACC as 4x128-bit passthru, Output {void*, qu, 4x128-bit QACC} via pointers
  def esp_vmulas_u8_qacc_ld_ip : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, void const *, int, void *, void *, void *, void *, void *)">;
  
  // ESP.VMULAS.U8.QACC.LD.XP (_m version)
  // Explicit state passing: Input QACC as 4x128-bit, Output {void*, v16i8 qu, 4x128-bit QACC} via pointers
  // Input: v0, v1, v2, v3 (4x128-bit QACC passthru), qx, qy, ptr, rs2
  // Output: ptr (return value), qu (via pointer), 4x128-bit QACC via pointers (5 pointers total)
  def esp_vmulas_u8_qacc_ld_xp : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, void const *, int, void *, void *, void *, void *, void *)">;
  
  // ESP.VMULAS.U8.QACC.ST.IP (_m version)
  // Explicit state passing: Input QACC as 4x128-bit passthru, Output {void*, 4x128-bit QACC} via pointers
  def esp_vmulas_u8_qacc_st_ip : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, void *, int, void *, void *, void *, void *)">;
  
  // ESP.VMULAS.U8.QACC.ST.XP (_m version)
  // Explicit state passing: Input QACC as 4x128-bit passthru, Output {void*, 4x128-bit QACC} via pointers
  def esp_vmulas_u8_qacc_st_xp : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, void *, int, void *, void *, void *, void *)">;
  
  // ESP.VMULAS.U8.XACC (_m version)
  // Explicit state passing: Input XACC, Output new XACC
  def esp_vmulas_u8_xacc : RISCVBuiltin<"void(unsigned int, unsigned int, _Vector<16, char>, _Vector<16, char>, unsigned int *, unsigned int *)">;  // xacc_h and xacc_l are both unsigned int (i32)
  
  // ESP.VMULAS.U8.XACC.LD.IP (_m version)
  // Explicit state passing: Input XACC, Output {qu (128-bit), ptr, new XACC}
  // Returns pointer, stores qu and xacc to output parameters
  // ESP.VMULAS.U8.XACC.LD.IP (_m version) - Mixed model: XACC as {unsigned int low, unsigned int high}
  // Explicit state passing: Input XACC as two i32, Output {qu (128-bit), ptr, new XACC as two i32}
  // Note: xacc_high_in and xacc_high_out are i32 but only low 8 bits are valid (XACC[39:32])
  def esp_vmulas_u8_xacc_ld_ip : RISCVBuiltin<"void*(unsigned int, unsigned int, _Vector<16, char>, _Vector<16, char>, void const *, int, void *, unsigned int *, unsigned int *)">;
  
  // ESP.VMULAS.U8.XACC.LD.XP (_m version) - Mixed model: XACC as {unsigned int low, unsigned int high}
  // Explicit state passing: Input XACC as two i32, Output {qu (128-bit), ptr, new XACC as two i32}
  // Note: xacc_high_in and xacc_high_out are i32 but only low 8 bits are valid (XACC[39:32])
  def esp_vmulas_u8_xacc_ld_xp : RISCVBuiltin<"void*(unsigned int, unsigned int, _Vector<16, char>, _Vector<16, char>, void const *, int, void *, unsigned int *, unsigned int *)">;
  
  // ESP.VMULAS.U8.XACC.ST.IP (_m version) - Mixed model: XACC as {unsigned int low, unsigned int high}
  // Explicit state passing: Input XACC as two i32, Output {void*, new XACC as two i32}
  // Note: xacc_high_in and xacc_high_out are i32 but only low 8 bits are valid (XACC[39:32])
  def esp_vmulas_u8_xacc_st_ip : RISCVBuiltin<"void*(unsigned int, unsigned int, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, void *, int, unsigned int *, unsigned int *)">;
  
  // ESP.VMULAS.U8.XACC.ST.XP (_m version) - Mixed model: XACC as {unsigned int low, unsigned int high}
  // Explicit state passing: Input XACC as two i32, Output {void*, new XACC as two i32}
  // Note: xacc_high_in and xacc_high_out are i32 but only low 8 bits are valid (XACC[39:32])
  def esp_vmulas_u8_xacc_st_xp : RISCVBuiltin<"void*(unsigned int, unsigned int, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, void *, int, unsigned int *, unsigned int *)">;
  
  // ESP.VMULAS.S16.QACC.LDBC.INCP (_m version)
  // Explicit state passing: Input QACC as 4x128-bit, Output {void*, v16i8 qu, 4x128-bit QACC} via pointers
  // Input: v0, v1, v2, v3 (4x128-bit QACC passthru), qx, qy, ptr
  // Output: ptr (return value), qu (via pointer), 4x128-bit QACC via pointers (5 pointers total)
  def esp_vmulas_s16_qacc_ldbc_incp : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<8, short>, _Vector<8, short>, void const *, void *, void *, void *, void *, void *)">;
  
  // ESP.VMULAS.S8.QACC.LDBC.INCP (_m version)
  // Explicit state passing: Input QACC as 4x128-bit, Output {void*, v16i8 qu, 4x128-bit QACC} via pointers
  def esp_vmulas_s8_qacc_ldbc_incp : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, void const *, void *, void *, void *, void *, void *)">;
  
  // ESP.VMULAS.U16.QACC.LDBC.INCP (_m version)
  // Explicit state passing: Input QACC as 4x128-bit, Output {void*, v16i8 qu, 4x128-bit QACC} via pointers
  def esp_vmulas_u16_qacc_ldbc_incp : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<8, short>, _Vector<8, short>, void const *, void *, void *, void *, void *, void *)">;
  
  // ESP.VMULAS.U8.QACC.LDBC.INCP (_m version)
  // Explicit state passing: Input QACC as 4x128-bit, Output {void*, v16i8 qu, 4x128-bit QACC} via pointers
  def esp_vmulas_u8_qacc_ldbc_incp : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, void const *, void *, void *, void *, void *, void *)">;
  
  // ESP.VSMULAS.S16.QACC (_m version)
  // Explicit state passing: Input QACC (4x128-bit), Output QACC (4x128-bit) via pointers
  // Input: v0 (QACC_L[127:0]), v1 (QACC_L[255:128]), v2 (QACC_H[127:0]), v3 (QACC_H[255:128]), qx, qy, sel16
  // Output: 4x128-bit QACC via pointers
  def esp_vsmulas_s16_qacc : RISCVBuiltin<"void(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<8, short>, _Vector<8, short>, int, void *, void *, void *, void *)">;
  
  // ESP.VSMULAS.S16.QACC.LD.INCP (_m version)
  // Explicit state passing: Input QACC (4x128-bit), Output QACC (4x128-bit) via pointers
  // Input: v0 (QACC_L[127:0]), v1 (QACC_L[255:128]), v2 (QACC_H[127:0]), v3 (QACC_H[255:128]), qx, qy, ptr, sel16
  // Output: qu + 4x128-bit QACC via pointers
  def esp_vsmulas_s16_qacc_ld_incp : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<8, short>, _Vector<8, short>, void const *, int, void *, void *, void *, void *, void *)">;
  
  // ESP.VSMULAS.S8.QACC (_m version)
  // Explicit state passing: Input QACC (4x128-bit), Output QACC (4x128-bit) via pointers
  // Input: v0 (QACC_L[127:0]), v1 (QACC_L[255:128]), v2 (QACC_H[127:0]), v3 (QACC_H[255:128]), qx, qy, sel16
  // Output: 4x128-bit QACC via pointers
  def esp_vsmulas_s8_qacc : RISCVBuiltin<"void(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, int, void *, void *, void *, void *)">;
  
  // ESP.VSMULAS.S8.QACC.LD.INCP (_m version)
  // Explicit state passing: Input QACC (4x128-bit), Output QACC (4x128-bit) via pointers
  // Input: v0 (QACC_L[127:0]), v1 (QACC_L[255:128]), v2 (QACC_H[127:0]), v3 (QACC_H[255:128]), qx, qy, ptr, sel16
  // Output: qu + 4x128-bit QACC via pointers
  def esp_vsmulas_s8_qacc_ld_incp : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, void const *, int, void *, void *, void *, void *, void *)">;
  
  // ESP.VSMULAS.U16.QACC (_m version)
  // Explicit state passing: Input QACC (4x128-bit), Output QACC (4x128-bit) via pointers
  // Input: v0 (QACC_L[127:0]), v1 (QACC_L[255:128]), v2 (QACC_H[127:0]), v3 (QACC_H[255:128]), qx, qy, sel16
  // Output: 4x128-bit QACC via pointers
  def esp_vsmulas_u16_qacc : RISCVBuiltin<"void(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<8, short>, _Vector<8, short>, int, void *, void *, void *, void *)">;
  
  // ESP.VSMULAS.U16.QACC.LD.INCP (_m version)
  // Explicit state passing: Input QACC (4x128-bit), Output QACC (4x128-bit) via pointers
  // Input: v0 (QACC_L[127:0]), v1 (QACC_L[255:128]), v2 (QACC_H[127:0]), v3 (QACC_H[255:128]), qx, qy, ptr, sel16
  // Output: qu + 4x128-bit QACC via pointers
  def esp_vsmulas_u16_qacc_ld_incp : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<8, short>, _Vector<8, short>, void const *, int, void *, void *, void *, void *, void *)">;
  
  // ESP.VSMULAS.U8.QACC (_m version)
  // Explicit state passing: Input QACC (4x128-bit), Output QACC (4x128-bit) via pointers
  // Input: v0 (QACC_L[127:0]), v1 (QACC_L[255:128]), v2 (QACC_H[127:0]), v3 (QACC_H[255:128]), qx, qy, sel16
  // Output: 4x128-bit QACC via pointers
  def esp_vsmulas_u8_qacc : RISCVBuiltin<"void(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, int, void *, void *, void *, void *)">;
  
  // ESP.VSMULAS.U8.QACC.LD.INCP (_m version)
  // Explicit state passing: Input QACC (4x128-bit), Output QACC (4x128-bit) via pointers
  // Input: v0 (QACC_L[127:0]), v1 (QACC_L[255:128]), v2 (QACC_H[127:0]), v3 (QACC_H[255:128]), qx, qy, ptr, sel16
  // Output: qu + 4x128-bit QACC via pointers
  def esp_vsmulas_u8_qacc_ld_incp : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, void const *, int, void *, void *, void *, void *, void *)">;
  
  // ESP.CMUL.S16 (unified 2.2 API; defaults sat=1 rm=7 for +xespv)
  def esp_cmul_s16 : RISCVBuiltin<"_Vector<8, short>(_Vector<8, short>, _Vector<8, short>, _Vector<8, short>, int, unsigned int, unsigned int, unsigned int)">;
  
  // ESP.CMUL.S16.LD.INCP (unified 2.2 API)
  def esp_cmul_s16_ld_incp : RISCVBuiltin<"void*(_Vector<8, short>, _Vector<8, short>, _Vector<8, short>, void const *, int, unsigned int, unsigned int, void *, void *, unsigned int)">;
  
  // ESP.CMUL.S16.ST.INCP (unified 2.2 API)
  def esp_cmul_s16_st_incp : RISCVBuiltin<"void*(_Vector<8, short>, _Vector<8, short>, _Vector<8, short>, _Vector<16, char>, void *, int, unsigned int, unsigned int, unsigned int)">;
  
  // ESP.CMUL.S8 (unified 2.2 API; defaults sat=1 rm=7 for +xespv)
  def esp_cmul_s8 : RISCVBuiltin<"_Vector<16, char>(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, int, unsigned int, unsigned int, unsigned int)">;
  
  // ESP.CMUL.S8.LD.INCP (unified 2.2 API)
  def esp_cmul_s8_ld_incp : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, void const *, int, unsigned int, unsigned int, void *, void *, unsigned int)">;
  
  // ESP.CMUL.S8.ST.INCP (unified 2.2 API)
  def esp_cmul_s8_st_incp : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, void *, int, unsigned int, unsigned int, unsigned int)">;
  
  // ESP.CMUL.U16 (_m version)
  // Note: qz_in is a passthru operand for preserving unmodified bits (RMW semantics).
  // Hardware doesn't read qz, but compiler needs it for correct register allocation.
  // SAR is passed explicitly as unsigned int (i32) for explicit state passing
  def esp_cmul_u16_m : RISCVBuiltin<"_Vector<8, short>(_Vector<8, short>, _Vector<8, short>, _Vector<8, short>, int, unsigned int)">;
  
  // ESP.CMUL.U16.LD.INCP (_m version)
  // Note: qz_in is a passthru operand for preserving unmodified bits (RMW semantics).
  // Output parameters: void *qz_out, void *qu_out (similar to esp_vld_128_ip)
  // SAR is passed explicitly as unsigned int (i32) for explicit state passing
  def esp_cmul_u16_ld_incp_m : RISCVBuiltin<"void*(_Vector<8, short>, _Vector<8, short>, _Vector<8, short>, void const *, int, void *, void *, unsigned int)">;
  
  // ESP.CMUL.U16.ST.INCP (_m version)
  // Note: qz_in is a passthru operand for preserving unmodified bits (RMW semantics).
  // SAR is passed explicitly as unsigned int (i32) for explicit state passing
  def esp_cmul_u16_st_incp_m : RISCVBuiltin<"void*(_Vector<8, short>, _Vector<8, short>, _Vector<8, short>, _Vector<16, char>, void *, int, unsigned int)">;
  
  // ESP.CMUL.U8 (_m version)
  // Note: qz_in is a passthru operand for preserving unmodified bits (RMW semantics).
  // Hardware doesn't read qz, but compiler needs it for correct register allocation.
  // SAR is passed explicitly as unsigned int (i32) for explicit state passing
  def esp_cmul_u8_m : RISCVBuiltin<"_Vector<16, char>(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, int, unsigned int)">;
  
  // ESP.CMUL.U8.LD.INCP (_m version)
  // Note: qz_in is a passthru operand for preserving unmodified bits (RMW semantics).
  // Output parameters: void *qz_out, void *qu_out (similar to esp_vld_128_ip)
  // SAR is passed explicitly as unsigned int (i32) for explicit state passing
  def esp_cmul_u8_ld_incp_m : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, void const *, int, void *, void *, unsigned int)">;
  
  // ESP.CMUL.U8.ST.INCP (_m version)
  // Note: qz_in is a passthru operand for preserving unmodified bits (RMW semantics).
  // SAR is passed explicitly as unsigned int (i32) for explicit state passing
  def esp_cmul_u8_st_incp_m : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, void *, int, unsigned int)">;

  // Load function (Internal) - Immediate offset
  // 1. Must use C language full names (void, int, char)
  // 2. Must use _Vector<count, type> to represent vectors
  // 3. Important: Must write "void const *" (East Const), cannot write "const void *"
  // 4. Output parameters: Use void * to point to vector data (16-byte aligned)
  def esp_vld_128_ip : RISCVBuiltin<"void*(void const *, int, void *)">;

  def esp_ld_128_usar_ip : RISCVBuiltin<"void*(void const *, int, void *, unsigned int *)">;

  // ESP.LD.XACC.IP (_m version) - Load 64-bit data and store low 40 bits to XACC
  // ESP.LD.XACC.IP (_m version) - Mixed model: XACC as {unsigned int low, unsigned int high}
  // Explicit state passing: Input XACC as two i32, Output {ptr, new XACC as two i32}
  // Returns pointer, stores new XACC to output parameters
  // Note: xacc_high_in and xacc_high_out are i32 but only low 8 bits are valid (XACC[39:32])
  def esp_ld_xacc_ip : RISCVBuiltin<"void*(unsigned int, unsigned int, void const *, int, void *, unsigned int *, unsigned int *)">;

  // ESP.LDQA.S16.128.IP (_m version) - Load 128-bit, sign-extend 8x16-bit to 64-bit, store to QACC
  // ESP.LDQA.S16.128.IP (_m version) - Returns 4x128-bit QACC directly to avoid shufflevector
  def esp_ldqa_s16_128_ip : RISCVBuiltin<"void*(void const *, int, void *, void *, void *, void *)">;

  // ESP.LDQA.S16.128.XP (_m version) - Load 128-bit, sign-extend 8x16-bit to 64-bit, store to QACC
  // Returns 4x128-bit QACC directly to avoid shufflevector
  def esp_ldqa_s16_128_xp : RISCVBuiltin<"void*(void const *, int, void *, void *, void *, void *)">;

  // ESP.LDQA.S8.128.IP (_m version) - Load 128-bit, sign-extend 16x8-bit to 32-bit, store to QACC
  // Returns 4x128-bit QACC directly to avoid shufflevector
  def esp_ldqa_s8_128_ip : RISCVBuiltin<"void*(void const *, int, void *, void *, void *, void *)">;

  // ESP.LDQA.S8.128.XP (_m version) - Load 128-bit, sign-extend 16x8-bit to 32-bit, store to QACC
  // Returns 4x128-bit QACC directly to avoid shufflevector
  def esp_ldqa_s8_128_xp : RISCVBuiltin<"void*(void const *, int, void *, void *, void *, void *)">;

  // ESP.LDQA.U16.128.IP (_m version) - Load 128-bit, zero-extend 8x16-bit to 64-bit, store to QACC
  // Returns 4x128-bit QACC directly to avoid shufflevector
  def esp_ldqa_u16_128_ip : RISCVBuiltin<"void*(void const *, int, void *, void *, void *, void *)">;

  // ESP.LDQA.U16.128.XP (_m version) - Load 128-bit, zero-extend 8x16-bit to 64-bit, store to QACC
  // Returns 4x128-bit QACC directly to avoid shufflevector
  def esp_ldqa_u16_128_xp : RISCVBuiltin<"void*(void const *, int, void *, void *, void *, void *)">;

  // ESP.LDQA.U8.128.IP (_m version) - Load 128-bit, zero-extend 16x8-bit to 32-bit, store to QACC
  // Returns 4x128-bit QACC directly to avoid shufflevector
  def esp_ldqa_u8_128_ip : RISCVBuiltin<"void*(void const *, int, void *, void *, void *, void *)">;

  // ESP.LDQA.U8.128.XP (_m version) - Load 128-bit, zero-extend 16x8-bit to 32-bit, store to QACC
  // Returns 4x128-bit QACC directly to avoid shufflevector
  def esp_ldqa_u8_128_xp : RISCVBuiltin<"void*(void const *, int, void *, void *, void *, void *)">;

  // ESP.LD.QACC.H.H.128.IP (_m version) - Load 128-bit data to QACC_H[255:128]
  // Subregister model: returns updated pointer, outputs QACC_H[255:128] (128-bit) via pointer parameter
  def esp_ld_qacc_h_h_128_ip : RISCVBuiltin<"void*(void const *, int, void *)">;

  // ESP.LD.QACC.H.L.128.IP (_m version) - Load 128-bit data to QACC_H[127:0]
  // Subregister model: returns updated pointer, outputs QACC_H[127:0] (128-bit) via pointer parameter
  def esp_ld_qacc_h_l_128_ip : RISCVBuiltin<"void*(void const *, int, void *)">;

  // ESP.LD.QACC.L.H.128.IP (_m version) - Load 128-bit data to QACC_L[255:128]
  // Subregister model: returns updated pointer, outputs QACC_L[255:128] (128-bit) via pointer parameter
  def esp_ld_qacc_l_h_128_ip : RISCVBuiltin<"void*(void const *, int, void *)">;

  // ESP.LD.QACC.L.L.128.IP (_m version) - Load 128-bit data to QACC_L[127:0]
  // Subregister model: returns updated pointer, outputs QACC_L[127:0] (128-bit) via pointer parameter
  def esp_ld_qacc_l_l_128_ip : RISCVBuiltin<"void*(void const *, int, void *)">;

  // ESP.LD.UA.STATE.IP (_m version) - Load 128-bit data to UA_STATE[127:0]
  // ESP.LD.UA.STATE.IP (_m version) - Load 128-bit data to UA_STATE register
  // Parameters: (ptr, offset, ua_state_out)
  // Returns: void* (updated pointer)
  // Note: UA_STATE value is returned through output parameter, similar to esp_vld_128_ip
  def esp_ld_ua_state_ip : RISCVBuiltin<"void*(void const *, int, void *)">;

  // ESP.LDXQ.32 (_m version) - Load 32-bit data using qw vector and sel8, store to qu using sel4
  // Returns void, output passed via pointer
  def esp_ldxq_32 : RISCVBuiltin<"void(_Vector<8, short>, void const *, int, int, void *)">;

  // ESP.ST.QACC.H.H.128.IP (_m version) - Store QACC_H[255:128] to memory
  // Parameters: [QACC_H[255:128] (128-bit), pointer, offset]
  // Accept v16i8 (128-bit) to match hardware operation and simplify usage.
  // First principle: directly accept 128-bit value, matching hardware behavior.
  def esp_st_qacc_h_h_128_ip : RISCVBuiltin<"void*(_Vector<16, char>, void *, int)">;

  // ESP.ST.QACC.H.L.128.IP (_m version) - Store QACC_H[127:0] to memory
  // Parameters: [QACC_H[127:0] (128-bit), pointer, offset]
  // Accept v16i8 (128-bit) to match hardware operation and simplify usage.
  // First principle: directly accept 128-bit value, matching hardware behavior.
  def esp_st_qacc_h_l_128_ip : RISCVBuiltin<"void*(_Vector<16, char>, void *, int)">;

  // ESP.ST.QACC.L.H.128.IP (_m version) - Store QACC_L[255:128] to memory
  // Accept v16i8 (128-bit) to match hardware operation and simplify usage.
  // First principle: directly accept 128-bit value, matching hardware behavior.
  def esp_st_qacc_l_h_128_ip : RISCVBuiltin<"void*(_Vector<16, char>, void *, int)">;

  // ESP.ST.QACC.L.L.128.IP (_m version) - Store QACC_L[127:0] to memory
  // Parameters: [QACC_L (256-bit), pointer, offset]
  // ESP.ST.QACC.L.L.128.IP (_m version) - Store QACC_L[127:0] to memory
  // Accept v16i8 (128-bit) to match hardware operation and simplify usage.
  // First principle: directly accept 128-bit value, matching hardware behavior.
  def esp_st_qacc_l_l_128_ip : RISCVBuiltin<"void*(_Vector<16, char>, void *, int)">;

  // ESP.ST.UA.STATE.IP (_m version) - Store UA_STATE[127:0] to memory
  // ESP.ST.UA.STATE.IP (_m version) - Store UA_STATE register to memory
  // Parameters: (ua_state_passthru, ptr, offset)
  // Returns: void* (updated pointer)
  // Note: UA_STATE is passed as explicit parameter for explicit state passing
  def esp_st_ua_state_ip : RISCVBuiltin<"void*(_Vector<16, char>, void *, int)">;

  // ESP.STXQ.32 (_m version) - Store 32-bit data from qu vector using sel4 to memory address calculated using qw vector and sel8
  // Returns void
  def esp_stxq_32 : RISCVBuiltin<"void(void *, _Vector<4, int>, _Vector<8, short>, int, int)">;

  // ESP.VLDHBC.16.INCP (_m version) - Load 128-bit data and broadcast 16-bit halves to two 128-bit vectors
  // Returns updated pointer, outputs passed via pointers
  def esp_vldhbc_16_incp_m : RISCVBuiltin<"void*(void const *, void *, void *)">;

  // Load function (Internal) - Register offset
  def esp_vld_128_xp : RISCVBuiltin<"void*(void const *, int, void *)">;
  
  def esp_ld_128_usar_xp : RISCVBuiltin<"void*(void const *, int, void *, unsigned int *)">;
  
  // Store function - Immediate offset
  def esp_vst_128_ip : RISCVBuiltin<"void*(_Vector<16, char>, void *, int)">;
  
  // Store function - Register offset
  def esp_vst_128_xp : RISCVBuiltin<"void*(_Vector<16, char>, void *, int)">;

  // Load H.64 function (Internal) - Immediate offset
  // Load 64 bits to upper 64-bit segment of qu (qu[127:64])
  def esp_vld_h_64_ip : RISCVBuiltin<"void*(void const *, int, void *)">;
  
  // Load H.64 function (Internal) - Register offset
  def esp_vld_h_64_xp : RISCVBuiltin<"void*(void const *, int, void *)">;
  
  // Load L.64 function (Internal) - Immediate offset
  // Load 64 bits to lower 64-bit segment of qu (qu[63:0])
  def esp_vld_l_64_ip : RISCVBuiltin<"void*(void const *, int, void *)">;
  
  // Load L.64 function (Internal) - Register offset
  def esp_vld_l_64_xp : RISCVBuiltin<"void*(void const *, int, void *)">;
  
  // Store H.64 function - Immediate offset
  // Store upper 64-bit segment of qu (qu[127:64])
  def esp_vst_h_64_ip : RISCVBuiltin<"void*(_Vector<8, char>, void *, int)">;
  
  // Store H.64 function - Register offset
  def esp_vst_h_64_xp : RISCVBuiltin<"void*(_Vector<8, char>, void *, int)">;
  
  // Store L.64 function - Immediate offset
  // Store lower 64-bit segment of qu (qu[63:0])
  def esp_vst_l_64_ip : RISCVBuiltin<"void*(_Vector<8, char>, void *, int)">;
  
  // Store L.64 function - Register offset
  def esp_vst_l_64_xp : RISCVBuiltin<"void*(_Vector<8, char>, void *, int)">;

  // MAX function - Signed 8-bit
  // Returns the maximum value of 8-bit signed elements in the vector
  def esp_max_s8_a : RISCVBuiltin<"int(_Vector<16, char>)">;
  
  // MAX function - Signed 16-bit
  def esp_max_s16_a : RISCVBuiltin<"int(_Vector<8, short>)">;
  
  // MAX function - Signed 32-bit
  def esp_max_s32_a : RISCVBuiltin<"int(_Vector<4, int>)">;
  
  // MAX function - Unsigned 8-bit
  def esp_max_u8_a : RISCVBuiltin<"int(_Vector<16, char>)">;
  
  // MAX function - Unsigned 16-bit
  def esp_max_u16_a : RISCVBuiltin<"int(_Vector<8, short>)">;
  
  // MAX function - Unsigned 32-bit
  def esp_max_u32_a : RISCVBuiltin<"int(_Vector<4, int>)">;

  // MIN function - Signed 8-bit
  // Returns the minimum value of 8-bit signed elements in the vector
  def esp_min_s8_a : RISCVBuiltin<"int(_Vector<16, char>)">;
  
  // MIN function - Signed 16-bit
  def esp_min_s16_a : RISCVBuiltin<"int(_Vector<8, short>)">;
  
  // MIN function - Signed 32-bit
  def esp_min_s32_a : RISCVBuiltin<"int(_Vector<4, int>)">;
  
  // MIN function - Unsigned 8-bit
  def esp_min_u8_a : RISCVBuiltin<"int(_Vector<16, char>)">;
  
  // MIN function - Unsigned 16-bit
  def esp_min_u16_a : RISCVBuiltin<"int(_Vector<8, short>)">;
  
  // MIN function - Unsigned 32-bit
  def esp_min_u32_a : RISCVBuiltin<"int(_Vector<4, int>)">;

  // VABS function - 8-bit
  // Returns absolute values of 16 8-bit data segments
  def esp_vabs_8 : RISCVBuiltin<"_Vector<16, char>(_Vector<16, char>)">;
  
  // VABS function - 16-bit
  // Returns absolute values of 8 16-bit data segments
  def esp_vabs_16 : RISCVBuiltin<"_Vector<8, short>(_Vector<8, short>)">;
  
  // VABS function - 32-bit
  // Returns absolute values of 4 32-bit data segments
  def esp_vabs_32 : RISCVBuiltin<"_Vector<4, int>(_Vector<4, int>)">;

  // VADD function - Signed 8-bit
  // Adds two 8-bit signed vectors with saturated result
  def esp_vadd_s8 : RISCVBuiltin<"_Vector<16, char>(_Vector<16, char>, _Vector<16, char>, int)">;
  
  // VADD function - Signed 16-bit
  def esp_vadd_s16 : RISCVBuiltin<"_Vector<8, short>(_Vector<8, short>, _Vector<8, short>, int)">;
  
  // VADD function - Signed 32-bit
  def esp_vadd_s32 : RISCVBuiltin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>, int)">;
  
  // VADD function - Unsigned 8-bit
  // Adds two 8-bit unsigned vectors with saturated result
  def esp_vadd_u8 : RISCVBuiltin<"_Vector<16, char>(_Vector<16, char>, _Vector<16, char>, int)">;
  
  // VADD function - Unsigned 16-bit
  def esp_vadd_u16 : RISCVBuiltin<"_Vector<8, short>(_Vector<8, short>, _Vector<8, short>, int)">;
  
  // VADD function - Unsigned 32-bit
  def esp_vadd_u32 : RISCVBuiltin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>, int)">;

  // VCLAMP function - Signed 16-bit
  // Performs clamping operation on 16-bit signed data, sel16 specifies the range
  def esp_vclamp_s16 : RISCVBuiltin<"_Vector<8, short>(_Vector<8, short>, int)">;

  // VMAX function - Signed 8-bit
  // Compares two 8-bit signed vectors and returns the larger values
  def esp_vmax_s8 : RISCVBuiltin<"_Vector<16, char>(_Vector<16, char>, _Vector<16, char>)">;
  
  // VMAX function - Signed 16-bit
  def esp_vmax_s16 : RISCVBuiltin<"_Vector<8, short>(_Vector<8, short>, _Vector<8, short>)">;
  
  // VMAX function - Signed 32-bit
  def esp_vmax_s32 : RISCVBuiltin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>)">;
  
  // VMAX function - Unsigned 8-bit
  def esp_vmax_u8 : RISCVBuiltin<"_Vector<16, char>(_Vector<16, char>, _Vector<16, char>)">;
  
  // VMAX function - Unsigned 16-bit
  def esp_vmax_u16 : RISCVBuiltin<"_Vector<8, short>(_Vector<8, short>, _Vector<8, short>)">;
  
  // VMAX function - Unsigned 32-bit
  def esp_vmax_u32 : RISCVBuiltin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>)">;

  // VMIN function - Signed 8-bit
  // Compares two 8-bit signed vectors and returns the smaller values
  def esp_vmin_s8 : RISCVBuiltin<"_Vector<16, char>(_Vector<16, char>, _Vector<16, char>)">;
  
  // VMIN function - Signed 16-bit
  def esp_vmin_s16 : RISCVBuiltin<"_Vector<8, short>(_Vector<8, short>, _Vector<8, short>)">;
  
  // VMIN function - Signed 32-bit
  def esp_vmin_s32 : RISCVBuiltin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>)">;
  
  // VMIN function - Unsigned 8-bit
  def esp_vmin_u8 : RISCVBuiltin<"_Vector<16, char>(_Vector<16, char>, _Vector<16, char>)">;
  
  // VMIN function - Unsigned 16-bit
  def esp_vmin_u16 : RISCVBuiltin<"_Vector<8, short>(_Vector<8, short>, _Vector<8, short>)">;
  
  // VMIN function - Unsigned 32-bit
  def esp_vmin_u32 : RISCVBuiltin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>)">;

  // VPRELU.S16 (unified 2.2 API; defaults sat=1 rm=7 for +xespv)
  def esp_vprelu_s16 : RISCVBuiltin<"_Vector<8, short>(_Vector<8, short>, _Vector<8, short>, int, unsigned int, unsigned int)">;

  // VPRELU.S8 (unified 2.2 API; defaults sat=1 rm=7 for +xespv)
  def esp_vprelu_s8 : RISCVBuiltin<"_Vector<16, char>(_Vector<16, char>, _Vector<16, char>, int, unsigned int, unsigned int)">;

  // VRELU.S16 (unified 2.2 API; defaults sat=1 rm=7 for +xespv)
  def esp_vrelu_s16 : RISCVBuiltin<"_Vector<8, short>(_Vector<8, short>, int, int, unsigned int, unsigned int)">;

  // VRELU.S8 (unified 2.2 API; defaults sat=1 rm=7 for +xespv)
  def esp_vrelu_s8 : RISCVBuiltin<"_Vector<16, char>(_Vector<16, char>, int, int, unsigned int, unsigned int)">;

  // VSADDS function - Signed 16-bit
  // Performs vector saturated addition on 16-bit signed data
  def esp_vsadds_s16 : RISCVBuiltin<"_Vector<8, short>(_Vector<8, short>, int, int)">;

  // VSADDS function - Signed 8-bit
  // Performs vector saturated addition on 8-bit signed data
  def esp_vsadds_s8 : RISCVBuiltin<"_Vector<16, char>(_Vector<16, char>, int, int)">;

  // VSADDS function - Unsigned 16-bit
  // Performs vector saturated addition on 16-bit unsigned data
  def esp_vsadds_u16 : RISCVBuiltin<"_Vector<8, short>(_Vector<8, short>, int, int)">;

  // VSADDS function - Unsigned 8-bit
  // Performs vector saturated addition on 8-bit unsigned data
  def esp_vsadds_u8 : RISCVBuiltin<"_Vector<16, char>(_Vector<16, char>, int, int)">;

  // VSAT function - Signed 16-bit
  // Performs vector saturation on 16-bit signed segments
  def esp_vsat_s16 : RISCVBuiltin<"_Vector<8, short>(_Vector<8, short>, int, int)">;

  // VSAT function - Signed 32-bit
  // Performs vector saturation on 32-bit signed segments
  def esp_vsat_s32 : RISCVBuiltin<"_Vector<4, int>(_Vector<4, int>, int, int)">;

  // VSAT function - Signed 8-bit
  // Performs vector saturation on 8-bit signed segments
  def esp_vsat_s8 : RISCVBuiltin<"_Vector<16, char>(_Vector<16, char>, int, int)">;

  // VSAT function - Unsigned 16-bit
  // Performs vector saturation on 16-bit unsigned segments
  def esp_vsat_u16 : RISCVBuiltin<"_Vector<8, short>(_Vector<8, short>, int, int)">;

  // VSAT function - Unsigned 32-bit
  // Performs vector saturation on 32-bit unsigned segments
  def esp_vsat_u32 : RISCVBuiltin<"_Vector<4, int>(_Vector<4, int>, int, int)">;

  // VSAT function - Unsigned 8-bit
  // Performs vector saturation on 8-bit unsigned segments
  def esp_vsat_u8 : RISCVBuiltin<"_Vector<16, char>(_Vector<16, char>, int, int)">;

  // VSSUBS function - Signed 16-bit
  // Performs vector saturated subtraction on 16-bit signed data
  def esp_vssubs_s16 : RISCVBuiltin<"_Vector<8, short>(_Vector<8, short>, int, int)">;

  // VSSUBS function - Signed 8-bit
  // Performs vector saturated subtraction on 8-bit signed data
  def esp_vssubs_s8 : RISCVBuiltin<"_Vector<16, char>(_Vector<16, char>, int, int)">;

  // VSSUBS function - Unsigned 16-bit
  // Performs vector saturated subtraction on 16-bit unsigned data
  def esp_vssubs_u16 : RISCVBuiltin<"_Vector<8, short>(_Vector<8, short>, int, int)">;

  // VSSUBS function - Unsigned 8-bit
  // Performs vector saturated subtraction on 8-bit unsigned data
  def esp_vssubs_u8 : RISCVBuiltin<"_Vector<16, char>(_Vector<16, char>, int, int)">;

  // VSUB function - Signed 16-bit
  // Performs vector subtraction on 16-bit data with saturated result
  def esp_vsub_s16 : RISCVBuiltin<"_Vector<8, short>(_Vector<8, short>, _Vector<8, short>, int)">;

  // VSUB function - Signed 32-bit
  // Performs vector subtraction on 32-bit data with saturated result
  def esp_vsub_s32 : RISCVBuiltin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>, int)">;

  // VSUB function - Signed 8-bit
  // Performs vector subtraction on 8-bit data with saturated result
  def esp_vsub_s8 : RISCVBuiltin<"_Vector<16, char>(_Vector<16, char>, _Vector<16, char>, int)">;

  // VSUB function - Unsigned 16-bit
  // Performs vector subtraction on 16-bit unsigned data with saturated result
  def esp_vsub_u16 : RISCVBuiltin<"_Vector<8, short>(_Vector<8, short>, _Vector<8, short>, int)">;

  // VSUB function - Unsigned 32-bit
  // Performs vector subtraction on 32-bit unsigned data with saturated result
  def esp_vsub_u32 : RISCVBuiltin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>, int)">;

  // VSUB function - Unsigned 8-bit
  // Performs vector subtraction on 8-bit unsigned data with saturated result
  def esp_vsub_u8 : RISCVBuiltin<"_Vector<16, char>(_Vector<16, char>, _Vector<16, char>, int)">;

  // ANDQ function - Bitwise AND operation
  // Performs bitwise AND operation on 128-bit vector
  def esp_andq : RISCVBuiltin<"_Vector<16, char>(_Vector<16, char>, _Vector<16, char>)">;

  // NOTQ function - Bitwise NOT operation
  // Performs bitwise NOT operation on 128-bit vector
  def esp_notq : RISCVBuiltin<"_Vector<16, char>(_Vector<16, char>)">;

  // ORQ function - Bitwise OR operation
  // Performs bitwise OR operation on 128-bit vector
  def esp_orq : RISCVBuiltin<"_Vector<16, char>(_Vector<16, char>, _Vector<16, char>)">;

  // XORQ function - Bitwise XOR operation
  // Performs bitwise XOR operation on 128-bit vector
  def esp_xorq : RISCVBuiltin<"_Vector<16, char>(_Vector<16, char>, _Vector<16, char>)">;

  // VCMP.EQ function - Equality comparison
  def esp_vcmp_eq_s16 : RISCVBuiltin<"_Vector<8, short>(_Vector<8, short>, _Vector<8, short>)">;
  def esp_vcmp_eq_s32 : RISCVBuiltin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>)">;
  def esp_vcmp_eq_s8 : RISCVBuiltin<"_Vector<16, char>(_Vector<16, char>, _Vector<16, char>)">;
  // VCMP.GT function - Greater than comparison
  def esp_vcmp_gt_s16 : RISCVBuiltin<"_Vector<8, short>(_Vector<8, short>, _Vector<8, short>)">;
  def esp_vcmp_gt_s32 : RISCVBuiltin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>)">;
  def esp_vcmp_gt_s8 : RISCVBuiltin<"_Vector<16, char>(_Vector<16, char>, _Vector<16, char>)">;

  // VCMP.LT function - Less than comparison
  def esp_vcmp_lt_s16 : RISCVBuiltin<"_Vector<8, short>(_Vector<8, short>, _Vector<8, short>)">;
  def esp_vcmp_lt_s32 : RISCVBuiltin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>)">;
  def esp_vcmp_lt_s8 : RISCVBuiltin<"_Vector<16, char>(_Vector<16, char>, _Vector<16, char>)">;
  def esp_vcmp_lt_u16 : RISCVBuiltin<"_Vector<8, short>(_Vector<8, short>, _Vector<8, short>)">;
  def esp_vcmp_lt_u32 : RISCVBuiltin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>)">;
  def esp_vcmp_lt_u8 : RISCVBuiltin<"_Vector<16, char>(_Vector<16, char>, _Vector<16, char>)">;

  // MOVI - Extract from vector to scalar (A variants)
  // For explicit state passing consistency, SAR should be passed as input parameter.
  // MOVI.A extracts elements from vector to scalar (no SAR parameter needed)
  def esp_movi_16_a : RISCVBuiltin<"int(_Vector<8, short>, int)">;  // (vector, index)
  def esp_movi_32_a : RISCVBuiltin<"int(_Vector<4, int>, int)">;  // (vector, index)
  def esp_movi_8_a : RISCVBuiltin<"int(_Vector<16, char>, int)">;  // (vector, index)

  // MOVI - Insert from scalar to vector (Q variants)
  def esp_movi_16_q : RISCVBuiltin<"_Vector<8, short>(_Vector<8, short>, int, int)">;  // (qy, sel16, rs1) -> qy
  def esp_movi_32_q : RISCVBuiltin<"_Vector<4, int>(_Vector<4, int>, int, int)">;  // (qy, sel4, rs1) -> qy
  def esp_movi_8_q : RISCVBuiltin<"_Vector<16, char>(_Vector<16, char>, int, int)">;  // (qy, sel16, rs1) -> qy

  // VEXT - Vector extension (sign/zero extend)
  // VEXT has 2 outputs (qz, qv) and 1 input (qw)
  // For _m version, use void return with pointer parameters for output vectors
  def esp_vext_s16 : RISCVBuiltin<"void(_Vector<8, short>, void *, void *)">;
  def esp_vext_s8 : RISCVBuiltin<"void(_Vector<16, char>, void *, void *)">;
  def esp_vext_u16 : RISCVBuiltin<"void(_Vector<8, short>, void *, void *)">;
  def esp_vext_u8 : RISCVBuiltin<"void(_Vector<16, char>, void *, void *)">;

  // VUNZIP - Vector unzip (deinterleave)
  // Similar to VMUL.S16.S8XS8: void return with pointer parameters for outputs
  def esp_vunzip_16 : RISCVBuiltin<"void(_Vector<8, short>, _Vector<8, short>, void *, void *)">;
  def esp_vunzip_32 : RISCVBuiltin<"void(_Vector<4, int>, _Vector<4, int>, void *, void *)">;
  def esp_vunzip_8 : RISCVBuiltin<"void(_Vector<16, char>, _Vector<16, char>, void *, void *)">;

  // VUNZIPT - Vector unzip transpose (3 vectors)
  def esp_vunzipt_16 : RISCVBuiltin<"void(_Vector<8, short>, _Vector<8, short>, _Vector<8, short>, void *, void *, void *)">;
  def esp_vunzipt_8 : RISCVBuiltin<"void(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, void *, void *, void *)">;

  // VZIP - Vector zip (interleave)
  def esp_vzip_16 : RISCVBuiltin<"void(_Vector<8, short>, _Vector<8, short>, void *, void *)">;
  def esp_vzip_32 : RISCVBuiltin<"void(_Vector<4, int>, _Vector<4, int>, void *, void *)">;
  def esp_vzip_8 : RISCVBuiltin<"void(_Vector<16, char>, _Vector<16, char>, void *, void *)">;

  // VZIPT - Vector zip transpose (3 vectors)
  def esp_vzipt_16 : RISCVBuiltin<"void(_Vector<8, short>, _Vector<8, short>, _Vector<8, short>, void *, void *, void *)">;
  def esp_vzipt_8 : RISCVBuiltin<"void(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, void *, void *, void *)">;

  // ZERO.Q - Zero vector register
  // Sets the vector register to zero
  def esp_zero_q : RISCVBuiltin<"_Vector<16, char>()">;

  // ZERO.QACC - Zero QACC accumulator
  // Explicit state passing: Returns v64i8 (QACC_H + QACC_L, both zero)
  // Note: Although QACC_H and QACC_L are independent registers in hardware,
  // we model them as a single 512-bit aggregate value in the compiler to avoid
  // memory operations (alloca/store/load) and maintain register-level data flow.
  // ESP.ZERO.QACC (_m version) - Zero QACC_H and QACC_L with explicit state passing
  // Returns: _Vector<64, char>() - v64i8 (QACC_H + QACC_L, both zero)
  // ESP.ZERO.QACC (_m version) - Zero QACC_H and QACC_L with explicit state passing
  // Consistent with ESP_MOV_S16_QACC_M pattern
  def esp_zero_qacc : RISCVBuiltin<"void(void *, void *, void *, void *)">;

  // ESP.QACC.GET.L - Extract QACC_L (low 256 bits) from QACC (512 bits)
  // Extract Low 256 bits (QACC_L) from 512-bit QACC vector
  // Returns: v32i8 (QACC_L, low 256 bits)
  // This generates extract_subvector IR directly, avoiding memory operations
  def esp_qacc_get_l : RISCVBuiltin<"_Vector<32, char>(_Vector<64, char>)">;

  // ESP.QACC.GET.H - Extract QACC_H (high 256 bits) from QACC (512 bits)
  // Extract High 256 bits (QACC_H) from 512-bit QACC vector
  // Returns: v32i8 (QACC_H, high 256 bits)
  // This generates extract_subvector IR directly, avoiding memory operations
  def esp_qacc_get_h : RISCVBuiltin<"_Vector<32, char>(_Vector<64, char>)">;

  // ESP.QACC.L.GET.LOW - Extract QACC_L[127:0] (low 128 bits) from QACC_L (256 bits)
  // Extract Low 128 bits from 256-bit QACC_L vector
  // Returns: v16i8 (QACC_L[127:0], low 128 bits)
  // This generates extract_subvector IR directly, avoiding memory operations
  def esp_qacc_l_get_low : RISCVBuiltin<"_Vector<16, char>(_Vector<32, char>)">;

  // ESP.QACC.L.GET.HIGH - Extract QACC_L[255:128] (high 128 bits) from QACC_L (256 bits)
  // Extract High 128 bits from 256-bit QACC_L vector
  // Returns: v16i8 (QACC_L[255:128], high 128 bits)
  // This generates extract_subvector IR directly, avoiding memory operations
  def esp_qacc_l_get_high : RISCVBuiltin<"_Vector<16, char>(_Vector<32, char>)">;

  // ESP.QACC.H.GET.LOW - Extract QACC_H[127:0] (low 128 bits) from QACC_H (256 bits)
  // Extract Low 128 bits from 256-bit QACC_H vector
  // Returns: v16i8 (QACC_H[127:0], low 128 bits)
  // This generates extract_subvector IR directly, avoiding memory operations
  def esp_qacc_h_get_low : RISCVBuiltin<"_Vector<16, char>(_Vector<32, char>)">;

  // ESP.QACC.H.GET.HIGH - Extract QACC_H[255:128] (high 128 bits) from QACC_H (256 bits)
  // Extract High 128 bits from 256-bit QACC_H vector
  // Returns: v16i8 (QACC_H[255:128], high 128 bits)
  // This generates extract_subvector IR directly, avoiding memory operations
  def esp_qacc_h_get_high : RISCVBuiltin<"_Vector<16, char>(_Vector<32, char>)">;

  // FFT.R2BF.S16 (unified 2.2 API; defaults sat=1 for +xespv)
  def esp_fft_r2bf_s16 : RISCVBuiltin<"void(_Vector<8, short>, _Vector<8, short>, int, unsigned int, void *, void *)">;

  // FFT.AMS.S16.LD.INCP (unified 2.2 API)
  def esp_fft_ams_s16_ld_incp : RISCVBuiltin<"void*(_Vector<8, short>, _Vector<8, short>, _Vector<8, short>, void const *, int, unsigned int, void *, void *, void *, unsigned int)">;

  // FFT.AMS.S16.LD.INCP.UAUP (unified 2.2 API)
  def esp_fft_ams_s16_ld_incp_uaup : RISCVBuiltin<"void*(_Vector<8, short>, _Vector<8, short>, _Vector<8, short>, void const *, int, unsigned int, void *, void *, void *, _Vector<16, char>, void *, unsigned int, unsigned int)">;

  // FFT.R2BF.S16.ST.INCP (unified 2.2 API)
  def esp_fft_r2bf_s16_st_incp : RISCVBuiltin<"void*(_Vector<8, short>, _Vector<8, short>, void *, int, unsigned int, void *)">;

  // FFT.AMS.S16.LD.R32.DECP (unified 2.2 API)
  def esp_fft_ams_s16_ld_r32_decp : RISCVBuiltin<"void*(_Vector<8, short>, _Vector<8, short>, _Vector<8, short>, void const *, int, unsigned int, void *, void *, void *, unsigned int)">;

  // FFT.AMS.S16.ST.INCP (unified 2.2 API)
  def esp_fft_ams_s16_st_incp : RISCVBuiltin<"void*(_Vector<8, short>, _Vector<8, short>, _Vector<8, short>, _Vector<16, char>, void *, void *, void *, int, unsigned int, unsigned int)">;

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

  // FFT.CMUL.S16.LD.XP (unified 2.2 API)
  def esp_fft_cmul_s16_ld_xp : RISCVBuiltin<"void*(_Vector<8, short>, _Vector<8, short>, void const *, void *, int, unsigned int, void *, void *, unsigned int)">;

  // FFT.CMUL.S16.ST.XP (unified 2.2 API)
  def esp_fft_cmul_s16_st_xp : RISCVBuiltin<"void*(_Vector<8, short>, _Vector<8, short>, _Vector<16, char>, void *, void *, int, int, int, unsigned int, unsigned int)">;

  // FFT.VST.R32.DECP - Vector store with reverse 32-bit order and decrement
  // Has 1 input (qu) and 0 outputs
  // Returns updated pointer (rs1 - 16)
  def esp_fft_vst_r32_decp_m : RISCVBuiltin<"void*(_Vector<16, char>, void *, int)">;

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

  // SLCI.2Q - Shift Left with Immediate, 2 Quad registers
  // Shifts concatenated qy and qw left, updates both registers in-place
  // Input and output registers are the same due to constraints
  // Note: Uses void* for vector pointers as TableGen doesn't support _Vector<16, char>*
  def esp_slci_2q : RISCVBuiltin<"void(void *, void *, int)">;

  // SLCXXP.2Q - Shift Left with Register offset and Post-increment, 2 Quad registers
  // Shifts concatenated qy and qw left, updates both registers and pointer
  // Returns updated pointer, input and output registers are the same due to constraints
  // Explicit state passing: uses value parameters instead of pointers
  def esp_slcxxp_2q : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, void *, int)">;

  // SRCI.2Q - Shift Right with Immediate, 2 Quad registers
  // Shifts concatenated qy and qw right, updates both registers in-place
  // Input and output registers are the same due to constraints
  // Note: Uses void* for vector pointers as TableGen doesn't support _Vector<16, char>*
  def esp_srci_2q : RISCVBuiltin<"void(void *, void *, int)">;

  // SRC.Q - Shift Right Concatenated Quad registers
  // Performs arithmetic right shift on concatenated qy and qw using SAR_BYTE register
  // SAR_BYTES is passed explicitly as last parameter for explicit state passing
  // Returns shifted result in qz
  def esp_src_q : RISCVBuiltin<"_Vector<16, char>(_Vector<16, char>, _Vector<16, char>, unsigned int)">;

  // SRC.Q.LD.IP - Shift Right Concatenated Quad registers with Load and Immediate Post-increment
  // Performs arithmetic right shift on concatenated qy and qw, loads data to qu, updates pointer
  // SAR_BYTES is passed explicitly as last parameter for explicit state passing
  // Output parameters: void *qw_out, void *qu_out (similar to esp_cmul_ld_incp_m)
  def esp_src_q_ld_ip : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, void const *, int, void *, void *, unsigned int)">;

  // SRC.Q.LD.XP - Shift Right Concatenated Quad registers with Load and Register Post-increment
  // Performs arithmetic right shift on concatenated qy and qw, loads data to qu, updates pointer
  // SAR_BYTES is passed explicitly as last parameter for explicit state passing
  // Output parameters: void *qw_out, void *qu_out (similar to esp_cmul_ld_incp_m)
  def esp_src_q_ld_xp : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, void const *, int, void *, void *, unsigned int)">;

  // SRC.Q.qup - Shift Right Concatenated Quad registers with Update
  // Performs arithmetic right shift on concatenated qw and qy, updates qw = qy
  // SAR_BYTES is passed explicitly as last parameter for explicit state passing
  // Output parameters: void *qz_out, void *qw_out
  def esp_src_q_qup : RISCVBuiltin<"void(_Vector<16, char>, _Vector<16, char>, void *, void *, unsigned int)">;

  // SRCMB - Shift Right and Saturate from QACC
  // These instructions extract data segments from QACC_H and QACC_L, perform arithmetic right shift,
  // saturate the results, write back to QACC_H/QACC_L, and output saturated values to qu
  // Uses QACC_H and QACC_L as implicit registers (both input and output)

  // SRCMB (unified 2.1/2.2 API)
  def esp_srcmb_s16_q_qacc : RISCVBuiltin<"_Vector<8, short>(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<8, short>, int, int)">;
  def esp_srcmb_s16_qacc : RISCVBuiltin<"_Vector<8, short>(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, int, int, int)">;
  def esp_srcmb_s8_q_qacc : RISCVBuiltin<"_Vector<16, char>(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, int, int)">;
  def esp_srcmb_s8_qacc : RISCVBuiltin<"_Vector<16, char>(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, int, int, int)">;
  def esp_srcmb_u16_q_qacc : RISCVBuiltin<"_Vector<8, short>(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<8, short>, int, int, int)">;
  def esp_srcmb_u16_qacc : RISCVBuiltin<"_Vector<8, short>(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, int, int, int, int)">;
  def esp_srcmb_u8_q_qacc : RISCVBuiltin<"_Vector<16, char>(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, int, int, int)">;
  def esp_srcmb_u8_qacc : RISCVBuiltin<"_Vector<16, char>(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, _Vector<16, char>, int, int, int, int)">;

  // VSLD - Vector Shift Left Dynamic (unified 2.1/2.2 API)
  def esp_vsld_16 : RISCVBuiltin<"_Vector<8, short>(_Vector<8, short>, _Vector<8, short>, int, int)">;
  def esp_vsld_32 : RISCVBuiltin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>, int, int)">;
  def esp_vsld_8 : RISCVBuiltin<"_Vector<16, char>(_Vector<16, char>, _Vector<16, char>, int, int)">;

  // VSR - Vector Shift Right (using SAR register; unified 2.1/2.2 API)
  def esp_vsr_s32 : RISCVBuiltin<"_Vector<4, int>(_Vector<4, int>, unsigned int, int)">;
  def esp_vsr_u32 : RISCVBuiltin<"_Vector<4, int>(_Vector<4, int>, unsigned int, int)">;

  // VSL.32 - Vector Shift Left (using SAR register; unified 2.1/2.2 API)
  def esp_vsl_32 : RISCVBuiltin<"_Vector<4, int>(_Vector<4, int>, int, unsigned int)">;

  // SRCQ.128.ST.INCP - Shift Right Concatenated Quad registers and Store with Immediate Post-increment
  // Performs arithmetic right shift on concatenated qy and qw, stores low 128 bits to memory
  // SAR_BYTES is passed explicitly as last parameter for explicit state passing
  // Returns updated pointer
  def esp_srcq_128_st_incp : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, void *, unsigned int)">;

  // SRCXXP.2Q - Shift Right Concatenated with Register offset and Post-increment, 2 Quad registers
  // Performs logical right shift on concatenated qw and qy, updates both registers and pointer
  // Returns updated pointer
  def esp_srcxxp_2q : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, void *, int)">;

  // ESP.SRS.S.XACC (_m version)
  // Explicit state passing: Input XACC, Output {saturated value, new XACC}
  // Note: Instruction performs arithmetic right shift on XACC, saturates to 32-bit signed, writes back to XACC
  // ESP.SRS.S.XACC (_m version)
  // Explicit state passing: Input XACC (struct {xacc_h, xacc_l}), Output {saturated value, new XACC}
  // Note: Instruction performs arithmetic right shift on XACC, saturates to 32-bit signed, writes back to XACC
  // XACC is represented as struct {unsigned char xacc_h; unsigned int xacc_l;} via two parameters
  def esp_srs_s_xacc : RISCVBuiltin<"int(unsigned int, unsigned int, int, unsigned int, unsigned int, unsigned int *, unsigned int *)">;

  def esp_srs_u_xacc : RISCVBuiltin<"unsigned int(unsigned int, unsigned int, int, unsigned int, unsigned int, unsigned int *, unsigned int *)">;

  // ST.S.XACC.IP - Store Signed XACC with Immediate Post-increment
  // Sign-extends XACC to 64-bit and stores to memory, updates pointer
  // Returns updated pointer
  // Parameters: [current XACC (passthru), pointer, immediate offset]
  // ESP.ST.S.XACC.IP (_m version) - Mixed model: XACC as {unsigned int low, unsigned int high}
  // Explicit state passing: Input XACC as two i32, Output {ptr, XACC unchanged as two i32}
  // Returns pointer, stores XACC (unchanged) to output parameters
  // Note: xacc_high_in and xacc_high_out are i32 but only low 8 bits are valid (XACC[39:32])
  def esp_st_s_xacc_ip : RISCVBuiltin<"void*(unsigned int, unsigned int, void *, int, void *, unsigned int *, unsigned int *)">;

  // ESP.ST.U.XACC.IP (_m version) - Mixed model: XACC as {unsigned int low, unsigned int high}
  // Explicit state passing: Input XACC as two i32, Output {ptr, XACC unchanged as two i32}
  // Returns pointer, stores XACC (unchanged) to output parameters
  // Note: xacc_high_in and xacc_high_out are i32 but only low 8 bits are valid (XACC[39:32])
  def esp_st_u_xacc_ip : RISCVBuiltin<"void*(unsigned int, unsigned int, void *, int, void *, unsigned int *, unsigned int *)">;

  // VSRD - Vector Shift Right Dynamic (unified 2.1/2.2 API)
  def esp_vsrd_16 : RISCVBuiltin<"_Vector<8, short>(_Vector<8, short>, _Vector<8, short>, int, int)">;
  def esp_vsrd_32 : RISCVBuiltin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>, int, int)">;
  def esp_vsrd_8 : RISCVBuiltin<"_Vector<16, char>(_Vector<16, char>, _Vector<16, char>, int, int)">;
  
  // ESP.VADD.S16.LD.INCP (_m version)
  // ESP.VADD.S16.LD.INCP (_m version)
  // Output parameters: void *qv_out, void *qu_out
  def esp_vadd_s16_ld_incp : RISCVBuiltin<"void*(_Vector<8, short>, _Vector<8, short>, void const *, void *, void *, int)">;
  
  // ESP.VADD.S16.ST.INCP (_m version)
  def esp_vadd_s16_st_incp : RISCVBuiltin<"void*(_Vector<8, short>, _Vector<8, short>, _Vector<16, char>, void *, _Vector<8, short>, int)">;
  
  // ESP.VADD.S32.LD.INCP (_m version)
  // Output parameters: void *qv_out, void *qu_out
  def esp_vadd_s32_ld_incp : RISCVBuiltin<"void*(_Vector<4, int>, _Vector<4, int>, void const *, void *, void *, int)">;
  
  // ESP.VADD.S32.ST.INCP (_m version)
  def esp_vadd_s32_st_incp : RISCVBuiltin<"void*(_Vector<4, int>, _Vector<4, int>, _Vector<16, char>, void *, _Vector<4, int>, int)">;
  
  // ESP.VADD.S8.LD.INCP (_m version)
  // Output parameters: void *qv_out, void *qu_out (similar to esp_cmul and esp_vld_128_ip)
  def esp_vadd_s8_ld_incp : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, void const *, void *, void *, int)">;
  
  // ESP.VADD.S8.ST.INCP (_m version)
  def esp_vadd_s8_st_incp : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, void *, _Vector<16, char>, int)">;
  
  // ESP.VADD.U16.LD.INCP (_m version)
  // Output parameters: void *qv_out, void *qu_out
  def esp_vadd_u16_ld_incp : RISCVBuiltin<"void*(_Vector<8, short>, _Vector<8, short>, void const *, void *, void *, int)">;
  
  // ESP.VADD.U16.ST.INCP (_m version)
  def esp_vadd_u16_st_incp : RISCVBuiltin<"void*(_Vector<8, short>, _Vector<8, short>, _Vector<16, char>, void *, _Vector<8, short>, int)">;
  
  // ESP.VADD.U32.LD.INCP (_m version)
  // Output parameters: void *qv_out, void *qu_out
  def esp_vadd_u32_ld_incp : RISCVBuiltin<"void*(_Vector<4, int>, _Vector<4, int>, void const *, void *, void *, int)">;
  
  // ESP.VADD.U32.ST.INCP (_m version)
  def esp_vadd_u32_st_incp : RISCVBuiltin<"void*(_Vector<4, int>, _Vector<4, int>, _Vector<16, char>, void *, _Vector<4, int>, int)">;
  
  // ESP.VADD.U8.LD.INCP (_m version)
  // Output parameters: void *qv_out, void *qu_out
  def esp_vadd_u8_ld_incp : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, void const *, void *, void *, int)">;
  
  // ESP.VADD.U8.ST.INCP (_m version)
  def esp_vadd_u8_st_incp : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, void *, _Vector<16, char>, int)">;
  
  // ESP.VMAX.S16.LD.INCP (_m version)
  // ESP.VMAX.S16.LD.INCP (_m version)
  // Output parameters: void *qv_out, void *qu_out
  def esp_vmax_s16_ld_incp : RISCVBuiltin<"void*(_Vector<8, short>, _Vector<8, short>, void const *, void *, void *)">;
  
  // ESP.VMAX.S16.ST.INCP (_m version)
  def esp_vmax_s16_st_incp : RISCVBuiltin<"void*(_Vector<8, short>, _Vector<8, short>, _Vector<16, char>, void *, _Vector<8, short>)">;
  
  // ESP.VMAX.S32.LD.INCP (_m version)
  // ESP.VMAX.S32.LD.INCP (_m version)
  // Output parameters: void *qv_out, void *qu_out
  def esp_vmax_s32_ld_incp : RISCVBuiltin<"void*(_Vector<4, int>, _Vector<4, int>, void const *, void *, void *)">;
  
  // ESP.VMAX.S32.ST.INCP (_m version)
  def esp_vmax_s32_st_incp : RISCVBuiltin<"void*(_Vector<4, int>, _Vector<4, int>, _Vector<16, char>, void *, _Vector<4, int>)">;
  
  // ESP.VMAX.S8.LD.INCP (_m version)
  // ESP.VMAX.S8.LD.INCP (_m version)
  // Output parameters: void *qv_out, void *qu_out
  def esp_vmax_s8_ld_incp : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, void const *, void *, void *)">;
  
  // ESP.VMAX.S8.ST.INCP (_m version)
  def esp_vmax_s8_st_incp : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, void *, _Vector<16, char>)">;
  
  // ESP.VMAX.U16.LD.INCP (_m version)
  // ESP.VMAX.U16.LD.INCP (_m version)
  // Output parameters: void *qv_out, void *qu_out
  def esp_vmax_u16_ld_incp : RISCVBuiltin<"void*(_Vector<8, short>, _Vector<8, short>, void const *, void *, void *)">;
  
  // ESP.VMAX.U16.ST.INCP (_m version)
  def esp_vmax_u16_st_incp : RISCVBuiltin<"void*(_Vector<8, short>, _Vector<8, short>, _Vector<16, char>, void *, _Vector<8, short>)">;
  
  // ESP.VMAX.U32.LD.INCP (_m version)
  // ESP.VMAX.U32.LD.INCP (_m version)
  // Output parameters: void *qv_out, void *qu_out
  def esp_vmax_u32_ld_incp : RISCVBuiltin<"void*(_Vector<4, int>, _Vector<4, int>, void const *, void *, void *)">;
  
  // ESP.VMAX.U32.ST.INCP (_m version)
  def esp_vmax_u32_st_incp : RISCVBuiltin<"void*(_Vector<4, int>, _Vector<4, int>, _Vector<16, char>, void *, _Vector<4, int>)">;
  
  // ESP.VMAX.U8.LD.INCP (_m version)
  // ESP.VMAX.U8.LD.INCP (_m version)
  // Output parameters: void *qv_out, void *qu_out
  def esp_vmax_u8_ld_incp : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, void const *, void *, void *)">;
  
  // ESP.VMAX.U8.ST.INCP (_m version)
  def esp_vmax_u8_st_incp : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, void *, _Vector<16, char>)">;
  
  // ESP.VMIN.S16.LD.INCP (_m version)
  // ESP.VMIN.S16.LD.INCP (_m version)
  // Output parameters: void *qv_out, void *qu_out
  def esp_vmin_s16_ld_incp : RISCVBuiltin<"void*(_Vector<8, short>, _Vector<8, short>, void const *, void *, void *)">;
  
  // ESP.VMIN.S16.ST.INCP (_m version)
  def esp_vmin_s16_st_incp : RISCVBuiltin<"void*(_Vector<8, short>, _Vector<8, short>, _Vector<16, char>, void *, _Vector<8, short>)">;
  
  // ESP.VMIN.S32.LD.INCP (_m version)
  // ESP.VMIN.S32.LD.INCP (_m version)
  // Output parameters: void *qv_out, void *qu_out
  def esp_vmin_s32_ld_incp : RISCVBuiltin<"void*(_Vector<4, int>, _Vector<4, int>, void const *, void *, void *)">;
  
  // ESP.VMIN.S32.ST.INCP (_m version)
  def esp_vmin_s32_st_incp : RISCVBuiltin<"void*(_Vector<4, int>, _Vector<4, int>, _Vector<16, char>, void *, _Vector<4, int>)">;
  
  // ESP.VMIN.S8.LD.INCP (_m version)
  // ESP.VMIN.S8.LD.INCP (_m version)
  // Output parameters: void *qv_out, void *qu_out
  def esp_vmin_s8_ld_incp : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, void const *, void *, void *)">;
  
  // ESP.VMIN.S8.ST.INCP (_m version)
  def esp_vmin_s8_st_incp : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, void *, _Vector<16, char>)">;
  
  // ESP.VMIN.U16.LD.INCP (_m version)
  // ESP.VMIN.U16.LD.INCP (_m version)
  // Output parameters: void *qv_out, void *qu_out
  def esp_vmin_u16_ld_incp : RISCVBuiltin<"void*(_Vector<8, short>, _Vector<8, short>, void const *, void *, void *)">;
  
  // ESP.VMIN.U16.ST.INCP (_m version)
  def esp_vmin_u16_st_incp : RISCVBuiltin<"void*(_Vector<8, short>, _Vector<8, short>, _Vector<16, char>, void *, _Vector<8, short>)">;
  
  // ESP.VMIN.U32.LD.INCP (_m version)
  // ESP.VMIN.U32.LD.INCP (_m version)
  // Output parameters: void *qv_out, void *qu_out
  def esp_vmin_u32_ld_incp : RISCVBuiltin<"void*(_Vector<4, int>, _Vector<4, int>, void const *, void *, void *)">;
  
  // ESP.VMIN.U32.ST.INCP (_m version)
  def esp_vmin_u32_st_incp : RISCVBuiltin<"void*(_Vector<4, int>, _Vector<4, int>, _Vector<16, char>, void *, _Vector<4, int>)">;
  
  // ESP.VMIN.U8.LD.INCP (_m version)
  // ESP.VMIN.U8.LD.INCP (_m version)
  // Output parameters: void *qv_out, void *qu_out
  def esp_vmin_u8_ld_incp : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, void const *, void *, void *)">;
  
  // ESP.VMIN.U8.ST.INCP (_m version)
  def esp_vmin_u8_st_incp : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, void *, _Vector<16, char>)">;
  
  // ESP.VMUL.S16 (unified 2.2 API)
  def esp_vmul_s16 : RISCVBuiltin<"_Vector<8, short>(_Vector<8, short>, _Vector<8, short>, unsigned int, unsigned int, unsigned int)">;

  // ESP.VMUL.S16.LD.INCP (unified 2.2 API): QX,QY,RS1,SAT,RM,QV*,QU*,Sar
  def esp_vmul_s16_ld_incp : RISCVBuiltin<"void*(_Vector<8, short>, _Vector<8, short>, void const *, unsigned int, unsigned int, void *, void *, unsigned int)">;

  // ESP.VMUL.S16.S8XS8 (unified 2.2 API)
  def esp_vmul_s16_s8xs8 : RISCVBuiltin<"void(_Vector<16, char>, _Vector<16, char>, void *, void *, unsigned int, unsigned int, unsigned int)">;

  // ESP.VMUL.S16.ST.INCP (unified 2.2 API)
  def esp_vmul_s16_st_incp : RISCVBuiltin<"void*(_Vector<8, short>, _Vector<8, short>, _Vector<16, char>, void *, _Vector<8, short>, unsigned int, unsigned int, unsigned int)">;

  // ESP.VMUL.S32.S16XS16 (unified 2.2 API)
  def esp_vmul_s32_s16xs16 : RISCVBuiltin<"void(_Vector<8, short>, _Vector<8, short>, void *, void *, unsigned int, unsigned int, unsigned int)">;

  // ESP.VMUL.S8 (unified 2.2 API)
  def esp_vmul_s8 : RISCVBuiltin<"_Vector<16, char>(_Vector<16, char>, _Vector<16, char>, unsigned int, unsigned int, unsigned int)">;

  // ESP.VMUL.S8.LD.INCP (unified 2.2 API): QX,QY,RS1,SAT,RM,QV*,QU*,Sar
  def esp_vmul_s8_ld_incp : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, void const *, unsigned int, unsigned int, void *, void *, unsigned int)">;

  // ESP.VMUL.S8.ST.INCP (unified 2.2 API)
  def esp_vmul_s8_st_incp : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, void *, _Vector<16, char>, unsigned int, unsigned int, unsigned int)">;

  // ESP.VMUL.U16 (unified 2.2 API)
  def esp_vmul_u16 : RISCVBuiltin<"_Vector<8, short>(_Vector<8, short>, _Vector<8, short>, unsigned int, unsigned int, unsigned int)">;

  // ESP.VMUL.U16.LD.INCP (unified 2.2 API): QX,QY,RS1,SAT,RM,QV*,QU*,Sar
  def esp_vmul_u16_ld_incp : RISCVBuiltin<"void*(_Vector<8, short>, _Vector<8, short>, void const *, unsigned int, unsigned int, void *, void *, unsigned int)">;

  // ESP.VMUL.U16.ST.INCP (unified 2.2 API)
  def esp_vmul_u16_st_incp : RISCVBuiltin<"void*(_Vector<8, short>, _Vector<8, short>, _Vector<16, char>, void *, _Vector<8, short>, unsigned int, unsigned int, unsigned int)">;

  // ESP.VMUL.U8 (unified 2.2 API)
  def esp_vmul_u8 : RISCVBuiltin<"_Vector<16, char>(_Vector<16, char>, _Vector<16, char>, unsigned int, unsigned int, unsigned int)">;

  // ESP.VMUL.U8.LD.INCP (unified 2.2 API): QX,QY,RS1,SAT,RM,QV*,QU*,Sar
  def esp_vmul_u8_ld_incp : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, void const *, unsigned int, unsigned int, void *, void *, unsigned int)">;

  // ESP.VMUL.U8.ST.INCP (unified 2.2 API)
  def esp_vmul_u8_st_incp : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, void *, _Vector<16, char>, unsigned int, unsigned int, unsigned int)">;
  
  // ESP.VSUB.S16.LD.INCP (_m version)
  // ESP.VSUB.S16.LD.INCP (_m version)
  // Output parameters: void *qv_out, void *qu_out
  def esp_vsub_s16_ld_incp : RISCVBuiltin<"void*(_Vector<8, short>, _Vector<8, short>, void const *, void *, void *, int)">;
  
  // ESP.VSUB.S16.ST.INCP (_m version)
  def esp_vsub_s16_st_incp : RISCVBuiltin<"void*(_Vector<8, short>, _Vector<8, short>, _Vector<16, char>, void *, _Vector<8, short>, int)">;
  
  // ESP.VSUB.S32.LD.INCP (_m version)
  // ESP.VSUB.S32.LD.INCP (_m version)
  // Output parameters: void *qv_out, void *qu_out
  def esp_vsub_s32_ld_incp : RISCVBuiltin<"void*(_Vector<4, int>, _Vector<4, int>, void const *, void *, void *, int)">;
  
  // ESP.VSUB.S32.ST.INCP (_m version)
  def esp_vsub_s32_st_incp : RISCVBuiltin<"void*(_Vector<4, int>, _Vector<4, int>, _Vector<16, char>, void *, _Vector<4, int>, int)">;
  
  // ESP.VSUB.S8.LD.INCP (_m version)
  // ESP.VSUB.S8.LD.INCP (_m version)
  // Output parameters: void *qv_out, void *qu_out
  def esp_vsub_s8_ld_incp : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, void const *, void *, void *, int)">;
  
  // ESP.VSUB.S8.ST.INCP (_m version)
  def esp_vsub_s8_st_incp : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, void *, _Vector<16, char>, int)">;
  
  // ESP.VSUB.U16.LD.INCP (_m version)
  // ESP.VSUB.U16.LD.INCP (_m version)
  // Output parameters: void *qv_out, void *qu_out
  def esp_vsub_u16_ld_incp : RISCVBuiltin<"void*(_Vector<8, short>, _Vector<8, short>, void const *, void *, void *, int)">;
  
  // ESP.VSUB.U16.ST.INCP (_m version)
  def esp_vsub_u16_st_incp : RISCVBuiltin<"void*(_Vector<8, short>, _Vector<8, short>, _Vector<16, char>, void *, _Vector<8, short>, int)">;
  
  // ESP.VSUB.U32.LD.INCP (_m version)
  // ESP.VSUB.U32.LD.INCP (_m version)
  // Output parameters: void *qv_out, void *qu_out
  def esp_vsub_u32_ld_incp : RISCVBuiltin<"void*(_Vector<4, int>, _Vector<4, int>, void const *, void *, void *, int)">;
  
  // ESP.VSUB.U32.ST.INCP (_m version)
  def esp_vsub_u32_st_incp : RISCVBuiltin<"void*(_Vector<4, int>, _Vector<4, int>, _Vector<16, char>, void *, _Vector<4, int>, int)">;
  
  // ESP.VSUB.U8.LD.INCP (_m version)
  // ESP.VSUB.U8.LD.INCP (_m version)
  // Output parameters: void *qv_out, void *qu_out
  def esp_vsub_u8_ld_incp : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, void const *, void *, void *, int)">;
  
  // ESP.VSUB.U8.ST.INCP (_m version)
  def esp_vsub_u8_st_incp : RISCVBuiltin<"void*(_Vector<16, char>, _Vector<16, char>, _Vector<16, char>, void *, _Vector<16, char>, int)">;

  // ESP.MOV.S16.QACC (_m version) - Sign extend 8x16-bit to 64-bit, store to QACC_H and QACC_L
  def esp_mov_s16_qacc : RISCVBuiltin<"void(_Vector<8, short>, void *, void *, void *, void *)">;

  // ESP.MOV.S8.QACC (_m version) - Sign extend 16x8-bit to 32-bit, store to QACC_H and QACC_L
  def esp_mov_s8_qacc : RISCVBuiltin<"void(_Vector<16, char>, void *, void *, void *, void *)">;

  // ESP.MOV.U16.QACC (_m version) - Zero extend 8x16-bit to 64-bit, store to QACC_H and QACC_L
  def esp_mov_u16_qacc : RISCVBuiltin<"void(_Vector<8, short>, void *, void *, void *, void *)">;

  // ESP.MOV.U8.QACC (_m version) - Zero extend 16x8-bit to 32-bit, store to QACC_H and QACC_L
  def esp_mov_u8_qacc : RISCVBuiltin<"void(_Vector<16, char>, void *, void *, void *, void *)">;

  // Note: Scalar instructions (ADDX2, ADDX4, SUBX2, SUBX4, SAT) are already
  // defined in BuiltinsRISCVESP32P4.td, so we don't redefine them here.
}

// GPR ALU (.m): SSA result = op(rs1, rs2); lowers to esp.{addx2,addx4,subx2,subx4}.
let Features = "xespv2p1|xespv", Attributes = [NoThrow] in {
  def esp_addx2_m : RISCVBuiltin<"unsigned int(unsigned int, unsigned int)">;
  def esp_addx4_m : RISCVBuiltin<"unsigned int(unsigned int, unsigned int)">;
  def esp_subx2_m : RISCVBuiltin<"unsigned int(unsigned int, unsigned int)">;
  def esp_subx4_m : RISCVBuiltin<"unsigned int(unsigned int, unsigned int)">;
  def esp_sat_m : RISCVBuiltin<"unsigned int(unsigned int, unsigned int, unsigned int)">;
}

// MOVX - Read/Write implicit registers (_m version)
// These are scalar instructions, so they don't need RequiredVectorWidth<128>
let Features = "xespv2p1|xespv", Attributes = [NoThrow] in {
  // ZERO.XACC - Zero XACC accumulator
  // Returns: i64 (XACC value, low 40 bits are zero)
  // ESP.ZERO.XACC (_m version) - Mixed model: XACC as {unsigned int low, unsigned int high}
  // Explicit state passing: Output XACC as two i32 (both set to 0)
  // Sets XACC to zero, stores result to output parameters
  // Note: xacc_high_out is i32 but only low 8 bits are valid (XACC[39:32])
  // Note: Using void * instead of unsigned int * because TableGen may not support unsigned int * in all cases
  def esp_zero_xacc : RISCVBuiltin<"void(void *, void *)">;

  // MOVX.R - Read implicit registers (all return int)
  // CFG: PIE 2.1 exposes 32-bit CFG including vxsat_en; PIE 2.2 uses 7-bit CFG
  // without vxsat_en (per-instruction sat instead). No unified sat-enable API.
  def esp_movx_r_cfg : RISCVBuiltin<"unsigned int()">;
  // ESP.MOVX.R.FFT.BIT.WIDTH reads from FFT_BIT_WIDTH register (FFT_BIT_WIDTH is input source).
  // For explicit state passing, it takes FFT_BIT_WIDTH (32-bit unsigned) as input parameter.
  // Returns: rd[31:0] = {28'b0, FFT_BIT_WIDTH[3:0]} (32-bit unsigned, hardware extracts only low 4 bits)
  // FFT_BIT_WIDTH stores unsigned bit width mode (0-15), so both input and output are unsigned
  // Note: Using i32 instead of i64 because hardware register only has 4 bits valid, and input/output are both 32-bit
  def esp_movx_r_fft_bit_width : RISCVBuiltin<"unsigned int(unsigned int)">;  // Takes FFT_BIT_WIDTH (32-bit unsigned) as input, returns 32-bit unsigned
  def esp_movx_r_perf : RISCVBuiltin<"unsigned int(unsigned int, int)">;
  // Note: ESP.MOVX.R.SAR reads from SAR register (SAR is input source).
  // For explicit state passing consistency, it takes SAR (64-bit unsigned) as input parameter.
  // Returns: rd[31:0] = {26'b0, SAR[5:0]} (32-bit unsigned, hardware extracts only low 6 bits)
  // SAR stores unsigned shift amount (0-63), so both input and output are unsigned
  def esp_movx_r_sar : RISCVBuiltin<"unsigned int(unsigned int)">;  // Takes SAR (32-bit unsigned, only low 6 bits used, range 0-63) as input, returns 32-bit unsigned
  // ESP.MOVX.R.SAR_BYTES reads from SAR_BYTES register (SAR_BYTES is input source).
  // For explicit state passing consistency, it takes SAR_BYTES (64-bit unsigned) as input parameter.
  // Returns: rd[31:0] = {28'b0, SAR_BYTES[3:0]} (32-bit unsigned, hardware extracts only low 4 bits)
  // SAR_BYTES stores unsigned byte shift amount (0-15), so both input and output are unsigned
  def esp_movx_r_sar_bytes : RISCVBuiltin<"unsigned int(unsigned int)">;  // Takes SAR_BYTES (32-bit unsigned, only low 4 bits used, range 0-15) as input, returns 32-bit unsigned
  // ESP.MOVX.R.XACC_H reads from XACC[39:32] register (complete XACC is input source).
  // For explicit state passing consistency, it takes complete XACC as input parameter.
  // Returns: XACC[39:32] (i32, zero-extended from 8-bit)
  // ESP.MOVX.R.XACC_H reads from XACC[39:32] register (subregister model)
  // Returns: XACC[39:32] as unsigned char (i8)
  def esp_movx_r_xacc_h : RISCVBuiltin<"unsigned int(unsigned int)">;  // Subregister model: takes xacc_h (passthru, i32), returns XACC[39:32] as unsigned int (i32, only low 8 bits valid)
  // ESP.MOVX.R.XACC_L reads from XACC[31:0] register (subregister model)
  // Returns: XACC[31:0] as unsigned int (i32)
  def esp_movx_r_xacc_l : RISCVBuiltin<"unsigned int(unsigned int)">;  // Subregister model: takes xacc_l (passthru), returns XACC[31:0] (unsigned int)

  // MOVX.W - Write implicit registers
  // Note: esp_movx_w_sar returns unsigned int (SAR is 32-bit unsigned, only low 6 bits used, range 0-63) for explicit state passing (chaining)
  // SAR stores unsigned shift amount (0-63), so return type is unsigned
  // Raw i32 CFG writes on +xespv are masked to PIE 2.2 writable bits in codegen.
  def esp_movx_w_cfg : RISCVBuiltin<"void(unsigned int)">;
  // ESP.MOVX.W.FFT.BIT.WIDTH returns unsigned int (FFT_BIT_WIDTH is 32-bit unsigned) for explicit state passing (chaining)
  // Note: FFT_BIT_WIDTH is 32-bit register, but only low 4 bits are used (FFT_BIT_WIDTH[3:0] = rs1[3:0])
  // FFT_BIT_WIDTH stores unsigned bit width mode (0-15), so return type is unsigned
  // Note: Using i32 instead of i64 because hardware register only has 4 bits valid, and input/output are both 32-bit
  def esp_movx_w_fft_bit_width : RISCVBuiltin<"unsigned int(unsigned int)">;  // Returns FFT_BIT_WIDTH (32-bit unsigned, only low 4 bits used, range 0-15)
  def esp_movx_w_perf : RISCVBuiltin<"unsigned int(unsigned int, int)">;
  def esp_movx_w_sar : RISCVBuiltin<"unsigned int(unsigned int)">;  // Returns SAR (32-bit unsigned, only low 6 bits used, range 0-63)
  // ESP.MOVX.W.SAR_BYTES returns unsigned int (SAR_BYTES is 32-bit unsigned, only low 4 bits used, range 0-15) for explicit state passing (chaining)
  // Note: SAR_BYTES is 32-bit register, but only low 4 bits are used (SAR_BYTES[3:0] = rs1[3:0])
  // SAR_BYTES stores unsigned byte shift amount (0-15), so return type is unsigned
  def esp_movx_w_sar_bytes : RISCVBuiltin<"unsigned int(unsigned int)">;  // Returns SAR_BYTES (32-bit unsigned, only low 4 bits used, range 0-15)
  // ESP.MOVX.W.XACC_H writes XACC[39:32] (subregister model)
  // Writes XACC[39:32] = value[7:0], returns new XACC[39:32] as unsigned char
  def esp_movx_w_xacc_h : RISCVBuiltin<"unsigned int(unsigned int)">;  // Subregister model: value (unsigned int, only low 8 bits used), returns new XACC[39:32] as unsigned int (i32, only low 8 bits valid)
  // ESP.MOVX.W.XACC_L writes XACC[31:0] (subregister model)
  // Writes XACC[31:0] = value, returns new XACC[31:0] as unsigned int
  def esp_movx_w_xacc_l : RISCVBuiltin<"unsigned int(unsigned int)">;  // Subregister model: value (unsigned int), returns new XACC[31:0] (unsigned int)
}

// PIE 2.1-only: VCMP.EQ.U / VCMP.GT.U deleted in PIE 2.2 (see pie-merge-buckets/03-pie21-only.csv).
let Features = "xespv2p1", Attributes = [NoThrow, RequiredVectorWidth<128>] in {
  def esp_vcmp_eq_u16_m : RISCVBuiltin<"_Vector<8, short>(_Vector<8, short>, _Vector<8, short>)">;
  def esp_vcmp_eq_u32_m : RISCVBuiltin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>)">;
  def esp_vcmp_eq_u8_m : RISCVBuiltin<"_Vector<16, char>(_Vector<16, char>, _Vector<16, char>)">;
  def esp_vcmp_gt_u16_m : RISCVBuiltin<"_Vector<8, short>(_Vector<8, short>, _Vector<8, short>)">;
  def esp_vcmp_gt_u32_m : RISCVBuiltin<"_Vector<4, int>(_Vector<4, int>, _Vector<4, int>)">;
  def esp_vcmp_gt_u8_m : RISCVBuiltin<"_Vector<16, char>(_Vector<16, char>, _Vector<16, char>)">;
}
