// ESP32P4 multi-return load/store intrinsics
// SDTypeProfile<num_results, num_operands, [constraints]>
// Constraints: first N constraints for results, then constraints for operands
def SDT_ESPVLD128IPM : SDTypeProfile<2, 2, [
  SDTCisVT<0, v16i8>,      // Result 0: vector
  SDTCisPtrTy<1>,          // Result 1: updated pointer
  SDTCisPtrTy<2>,          // Operand 0: pointer
  SDTCisVT<3, i32>         // Operand 1: offset
]>;
def SDT_ESPLD128USARIPM : SDTypeProfile<3, 2, [
  SDTCisVT<0, v16i8>,      // Result 0: vector
  SDTCisPtrTy<1>,          // Result 1: updated pointer
  SDTCisVT<2, i32>,        // Result 2: SAR_BYTES (32-bit, only low 4 bits used, range 0-15)
  SDTCisPtrTy<3>,          // Operand 0: pointer
  SDTCisVT<4, i32>         // Operand 1: offset
]>;

def SDT_ESPVST128IPM : SDTypeProfile<1, 3, [
  SDTCisPtrTy<0>,          // Result 0: updated pointer
  SDTCisVT<1, v16i8>,      // Operand 0: vector
  SDTCisPtrTy<2>,          // Operand 1: pointer
  SDTCisVT<3, i32>         // Operand 2: offset
]>;
// Register offset versions use the same type profile
def SDT_ESPVLD128XPM : SDTypeProfile<2, 2, [
  SDTCisVT<0, v16i8>,      // Result 0: vector
  SDTCisPtrTy<1>,          // Result 1: updated pointer
  SDTCisPtrTy<2>,          // Operand 0: pointer
  SDTCisVT<3, i32>         // Operand 1: offset (register)
]>;
// USAR XP version has SAR_BYTES output
def SDT_ESPLD128USARXPM : SDTypeProfile<3, 2, [
  SDTCisVT<0, v16i8>,      // Result 0: vector
  SDTCisPtrTy<1>,          // Result 1: updated pointer
  SDTCisVT<2, i32>,        // Result 2: SAR_BYTES (32-bit, only low 4 bits used, range 0-15)
  SDTCisPtrTy<3>,          // Operand 0: pointer
  SDTCisVT<4, i32>         // Operand 1: offset (register)
]>;
def SDT_ESPVST128XPM : SDTypeProfile<1, 3, [
  SDTCisPtrTy<0>,          // Result 0: updated pointer
  SDTCisVT<1, v16i8>,      // Operand 0: vector
  SDTCisPtrTy<2>,          // Operand 1: pointer
  SDTCisVT<3, i32>         // Operand 2: offset (register)
]>;
def riscv_esp_vld_128_ip_m : RVSDNode<"ESP_VLD_128_IP_M", SDT_ESPVLD128IPM,
                                      [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def riscv_esp_ld_128_usar_ip_m : RVSDNode<"ESP_LD_128_USAR_IP_M", SDT_ESPLD128USARIPM,
                                      [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def riscv_esp_vst_128_ip_m : RVSDNode<"ESP_VST_128_IP_M", SDT_ESPVST128IPM,
                                      [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;
def riscv_esp_vld_128_xp_m : RVSDNode<"ESP_VLD_128_XP_M", SDT_ESPVLD128XPM,
                                      [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def riscv_esp_ld_128_usar_xp_m : RVSDNode<"ESP_LD_128_USAR_XP_M", SDT_ESPLD128USARXPM,
                                      [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;

def riscv_esp_vst_128_xp_m : RVSDNode<"ESP_VST_128_XP_M", SDT_ESPVST128XPM,
                                      [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;

// VLD/VST H.64/L.64 SDNodes
def SDT_ESPVLDH64IPM : SDTypeProfile<2, 2, [
  SDTCisVT<0, v8i8>,       // Result 0: vector
  SDTCisPtrTy<1>,          // Result 1: updated pointer
  SDTCisPtrTy<2>,          // Operand 0: pointer
  SDTCisVT<3, i32>         // Operand 1: offset
]>;
def SDT_ESPVLDH64XPM : SDTypeProfile<2, 2, [
  SDTCisVT<0, v8i8>,       // Result 0: vector
  SDTCisPtrTy<1>,          // Result 1: updated pointer
  SDTCisPtrTy<2>,          // Operand 0: pointer
  SDTCisVT<3, i32>         // Operand 1: offset (register)
]>;
def SDT_ESPVLDL64IPM : SDTypeProfile<2, 2, [
  SDTCisVT<0, v8i8>,       // Result 0: vector
  SDTCisPtrTy<1>,          // Result 1: updated pointer
  SDTCisPtrTy<2>,          // Operand 0: pointer
  SDTCisVT<3, i32>         // Operand 1: offset
]>;
def SDT_ESPVLDL64XPM : SDTypeProfile<2, 2, [
  SDTCisVT<0, v8i8>,       // Result 0: vector
  SDTCisPtrTy<1>,          // Result 1: updated pointer
  SDTCisPtrTy<2>,          // Operand 0: pointer
  SDTCisVT<3, i32>         // Operand 1: offset (register)
]>;
def SDT_ESPVSTH64IPM : SDTypeProfile<1, 3, [
  SDTCisPtrTy<0>,          // Result 0: updated pointer
  SDTCisVT<1, v8i8>,       // Operand 0: vector
  SDTCisPtrTy<2>,          // Operand 1: pointer
  SDTCisVT<3, i32>         // Operand 2: offset
]>;
def SDT_ESPVSTH64XPM : SDTypeProfile<1, 3, [
  SDTCisPtrTy<0>,          // Result 0: updated pointer
  SDTCisVT<1, v8i8>,       // Operand 0: vector
  SDTCisPtrTy<2>,          // Operand 1: pointer
  SDTCisVT<3, i32>         // Operand 2: offset (register)
]>;
def SDT_ESPVSTL64IPM : SDTypeProfile<1, 3, [
  SDTCisPtrTy<0>,          // Result 0: updated pointer
  SDTCisVT<1, v8i8>,       // Operand 0: vector
  SDTCisPtrTy<2>,          // Operand 1: pointer
  SDTCisVT<3, i32>         // Operand 2: offset
]>;
def SDT_ESPVSTL64XPM : SDTypeProfile<1, 3, [
  SDTCisPtrTy<0>,          // Result 0: updated pointer
  SDTCisVT<1, v8i8>,       // Operand 0: vector
  SDTCisPtrTy<2>,          // Operand 1: pointer
  SDTCisVT<3, i32>         // Operand 2: offset (register)
]>;
def riscv_esp_vld_h_64_ip_m : RVSDNode<"ESP_VLD_H_64_IP_M", SDT_ESPVLDH64IPM,
                                       [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def riscv_esp_vld_h_64_xp_m : RVSDNode<"ESP_VLD_H_64_XP_M", SDT_ESPVLDH64XPM,
                                       [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def riscv_esp_vld_l_64_ip_m : RVSDNode<"ESP_VLD_L_64_IP_M", SDT_ESPVLDL64IPM,
                                       [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def riscv_esp_vld_l_64_xp_m : RVSDNode<"ESP_VLD_L_64_XP_M", SDT_ESPVLDL64XPM,
                                       [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def riscv_esp_vst_h_64_ip_m : RVSDNode<"ESP_VST_H_64_IP_M", SDT_ESPVSTH64IPM,
                                       [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;
def riscv_esp_vst_h_64_xp_m : RVSDNode<"ESP_VST_H_64_XP_M", SDT_ESPVSTH64XPM,
                                       [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;
def riscv_esp_vst_l_64_ip_m : RVSDNode<"ESP_VST_L_64_IP_M", SDT_ESPVSTL64IPM,
                                       [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;
def riscv_esp_vst_l_64_xp_m : RVSDNode<"ESP_VST_L_64_XP_M", SDT_ESPVSTL64XPM,
                                       [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;

// ESP.ZERO.QACC SDNode
// Returns: 4x128-bit QACC (QACC_L_LOW, QACC_L_HIGH, QACC_H_LOW, QACC_H_HIGH)
// Operands: Chain (automatically added by SDNPHasChain)
def SDT_ESPZEROQACCM : SDTypeProfile<4, 0, [
  SDTCisVT<0, v16i8>,      // Result 0: QACC_L_LOW
  SDTCisVT<1, v16i8>,      // Result 1: QACC_L_HIGH
  SDTCisVT<2, v16i8>,      // Result 2: QACC_H_LOW
  SDTCisVT<3, v16i8>       // Result 3: QACC_H_HIGH
]>;
def riscv_esp_zero_qacc_m : RVSDNode<"ESP_ZERO_QACC_M", SDT_ESPZEROQACCM,
                                     [SDNPHasChain, SDNPOutGlue]>;
// Mixed model: XACC as {i32 low, i32 high}
def SDT_ESPZEROXACCM : SDTypeProfile<2, 0, [
  SDTCisVT<0, i32>,  // Result 0: XACC low (i32, set to 0)
  SDTCisVT<1, i32>   // Result 1: XACC high (i32, set to 0, only low 8 bits valid)
]>;
def riscv_esp_zero_xacc_m : RVSDNode<"ESP_ZERO_XACC_M", SDT_ESPZEROXACCM,
                                     [SDNPHasChain, SDNPOutGlue]>;

// CMUL LD.INCP SDNodes
// Results: qz (vector), qu (vector), rs1r (pointer)
// Operands: qz_in (passthru), qx, qy, rs1 (pointer), sel4 (i32 immediate), sar (i32)
// Chain is automatically added by SDNPHasChain
def SDT_ESPCMULS16LDINCPM : SDTypeProfile<3, 6, [
  SDTCisVT<0, v8i16>,      // Result 0: qz vector
  SDTCisVT<1, v16i8>,      // Result 1: qu vector (16 bytes loaded from memory)
  SDTCisPtrTy<2>,          // Result 2: updated pointer
  SDTCisVT<3, v8i16>,      // Operand 0: qz_in (passthru operand)
  SDTCisVT<4, v8i16>,      // Operand 1: qx vector
  SDTCisVT<5, v8i16>,      // Operand 2: qy vector
  SDTCisPtrTy<6>,          // Operand 3: rs1 pointer
  SDTCisVT<7, i32>,        // Operand 4: sel4 immediate
  SDTCisVT<8, i32>         // Operand 5: sar (SAR register value, explicit state passing, i32, only low 6 bits used)
]>;
def SDT_ESPCMULS8LDINCPM : SDTypeProfile<3, 6, [
  SDTCisVT<0, v16i8>,      // Result 0: qz vector
  SDTCisVT<1, v16i8>,      // Result 1: qu vector (16 bytes loaded from memory)
  SDTCisPtrTy<2>,          // Result 2: updated pointer
  SDTCisVT<3, v16i8>,      // Operand 0: qz_in (passthru operand)
  SDTCisVT<4, v16i8>,      // Operand 1: qx vector
  SDTCisVT<5, v16i8>,      // Operand 2: qy vector
  SDTCisPtrTy<6>,          // Operand 3: rs1 pointer
  SDTCisVT<7, i32>,        // Operand 4: sel4 immediate
  SDTCisVT<8, i32>         // Operand 5: sar (SAR register value, explicit state passing, i32, only low 6 bits used)
]>;
def SDT_ESPCMULU16LDINCPM : SDTypeProfile<3, 6, [
  SDTCisVT<0, v8i16>,      // Result 0: qz vector
  SDTCisVT<1, v16i8>,      // Result 1: qu vector (16 bytes loaded from memory)
  SDTCisPtrTy<2>,          // Result 2: updated pointer
  SDTCisVT<3, v8i16>,      // Operand 0: qz_in (passthru operand)
  SDTCisVT<4, v8i16>,      // Operand 1: qx vector
  SDTCisVT<5, v8i16>,      // Operand 2: qy vector
  SDTCisPtrTy<6>,          // Operand 3: rs1 pointer
  SDTCisVT<7, i32>,        // Operand 4: sel4 immediate
  SDTCisVT<8, i32>         // Operand 5: sar (SAR register value, explicit state passing, i32, only low 6 bits used)
]>;
def SDT_ESPCMULU8LDINCPM : SDTypeProfile<3, 6, [
  SDTCisVT<0, v16i8>,      // Result 0: qz vector
  SDTCisVT<1, v16i8>,      // Result 1: qu vector (16 bytes loaded from memory)
  SDTCisPtrTy<2>,          // Result 2: updated pointer
  SDTCisVT<3, v16i8>,      // Operand 0: qz_in (passthru operand)
  SDTCisVT<4, v16i8>,      // Operand 1: qx vector
  SDTCisVT<5, v16i8>,      // Operand 2: qy vector
  SDTCisPtrTy<6>,          // Operand 3: rs1 pointer
  SDTCisVT<7, i32>,        // Operand 4: sel4 immediate
  SDTCisVT<8, i32>         // Operand 5: sar (SAR register value, explicit state passing, i32, only low 6 bits used)
]>;

def riscv_esp_cmul_s16_ld_incp_m : RVSDNode<"ESP_CMUL_S16_LD_INCP_M", SDT_ESPCMULS16LDINCPM,
                                            [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def riscv_esp_cmul_s8_ld_incp_m : RVSDNode<"ESP_CMUL_S8_LD_INCP_M", SDT_ESPCMULS8LDINCPM,
                                           [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def riscv_esp_cmul_u16_ld_incp_m : RVSDNode<"ESP_CMUL_U16_LD_INCP_M", SDT_ESPCMULU16LDINCPM,
                                            [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def riscv_esp_cmul_u8_ld_incp_m : RVSDNode<"ESP_CMUL_U8_LD_INCP_M", SDT_ESPCMULU8LDINCPM,
                                           [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;

// CMUL ST.INCP SDNodes
// Results: qz (vector), rs1r (pointer)
// Operands: qz_in (passthru), qx, qy, qu, rs1 (pointer), sel4 (i32 immediate), sar (i32)
// Chain is automatically added by SDNPHasChain
def SDT_ESPCMULS16STINCPM : SDTypeProfile<2, 7, [
  SDTCisVT<0, v8i16>,      // Result 0: qz vector
  SDTCisPtrTy<1>,          // Result 1: updated pointer
  SDTCisVT<2, v8i16>,      // Operand 0: qz_in (passthru operand)
  SDTCisVT<3, v8i16>,      // Operand 1: qx vector
  SDTCisVT<4, v8i16>,      // Operand 2: qy vector
  SDTCisVT<5, v16i8>,      // Operand 3: qu vector
  SDTCisPtrTy<6>,          // Operand 4: rs1 pointer
  SDTCisVT<7, i32>,        // Operand 5: sel4 immediate
  SDTCisVT<8, i32>         // Operand 6: sar (SAR register value, explicit state passing, i32, only low 6 bits used)
]>;
def SDT_ESPCMULS8STINCPM : SDTypeProfile<2, 7, [
  SDTCisVT<0, v16i8>,      // Result 0: qz vector
  SDTCisPtrTy<1>,          // Result 1: updated pointer
  SDTCisVT<2, v16i8>,      // Operand 0: qz_in (passthru operand)
  SDTCisVT<3, v16i8>,      // Operand 1: qx vector
  SDTCisVT<4, v16i8>,      // Operand 2: qy vector
  SDTCisVT<5, v16i8>,      // Operand 3: qu vector
  SDTCisPtrTy<6>,          // Operand 4: rs1 pointer
  SDTCisVT<7, i32>,        // Operand 5: sel4 immediate
  SDTCisVT<8, i32>         // Operand 6: sar (SAR register value, explicit state passing, i32, only low 6 bits used)
]>;
def SDT_ESPCMULU16STINCPM : SDTypeProfile<2, 7, [
  SDTCisVT<0, v8i16>,      // Result 0: qz vector
  SDTCisPtrTy<1>,          // Result 1: updated pointer
  SDTCisVT<2, v8i16>,      // Operand 0: qz_in (passthru operand)
  SDTCisVT<3, v8i16>,      // Operand 1: qx vector
  SDTCisVT<4, v8i16>,      // Operand 2: qy vector
  SDTCisVT<5, v16i8>,      // Operand 3: qu vector
  SDTCisPtrTy<6>,          // Operand 4: rs1 pointer
  SDTCisVT<7, i32>,        // Operand 5: sel4 immediate
  SDTCisVT<8, i32>         // Operand 6: sar (SAR register value, explicit state passing, i32, only low 6 bits used)
]>;
def SDT_ESPCMULU8STINCPM : SDTypeProfile<2, 7, [
  SDTCisVT<0, v16i8>,      // Result 0: qz vector
  SDTCisPtrTy<1>,          // Result 1: updated pointer
  SDTCisVT<2, v16i8>,      // Operand 0: qz_in (passthru operand)
  SDTCisVT<3, v16i8>,      // Operand 1: qx vector
  SDTCisVT<4, v16i8>,      // Operand 2: qy vector
  SDTCisVT<5, v16i8>,      // Operand 3: qu vector
  SDTCisPtrTy<6>,          // Operand 4: rs1 pointer
  SDTCisVT<7, i32>,        // Operand 5: sel4 immediate
  SDTCisVT<8, i32>         // Operand 6: sar (SAR register value, explicit state passing, i32, only low 6 bits used)
]>;

def riscv_esp_cmul_s16_st_incp_m : RVSDNode<"ESP_CMUL_S16_ST_INCP_M", SDT_ESPCMULS16STINCPM,
                                             [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;
def riscv_esp_cmul_s8_st_incp_m : RVSDNode<"ESP_CMUL_S8_ST_INCP_M", SDT_ESPCMULS8STINCPM,
                                            [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;
def riscv_esp_cmul_u16_st_incp_m : RVSDNode<"ESP_CMUL_U16_ST_INCP_M", SDT_ESPCMULU16STINCPM,
                                             [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;
def riscv_esp_cmul_u8_st_incp_m : RVSDNode<"ESP_CMUL_U8_ST_INCP_M", SDT_ESPCMULU8STINCPM,
                                            [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;

// VLDBC SDNodes
def SDT_ESPVLDBC8IPM : SDTypeProfile<2, 2, [
  SDTCisVT<0, v16i8>,      // Result 0: vector
  SDTCisPtrTy<1>,          // Result 1: updated pointer
  SDTCisPtrTy<2>,          // Operand 0: pointer
  SDTCisVT<3, i32>         // Operand 1: offset
]>;
def SDT_ESPVLDBC8XPM : SDTypeProfile<2, 2, [
  SDTCisVT<0, v16i8>,      // Result 0: vector
  SDTCisPtrTy<1>,          // Result 1: updated pointer
  SDTCisPtrTy<2>,          // Operand 0: pointer
  SDTCisVT<3, i32>         // Operand 1: offset (register)
]>;
def SDT_ESPVLDBC16IPM : SDTypeProfile<2, 2, [
  SDTCisVT<0, v8i16>,      // Result 0: vector
  SDTCisPtrTy<1>,          // Result 1: updated pointer
  SDTCisPtrTy<2>,          // Operand 0: pointer
  SDTCisVT<3, i32>         // Operand 1: offset
]>;
def SDT_ESPVLDBC16XPM : SDTypeProfile<2, 2, [
  SDTCisVT<0, v8i16>,      // Result 0: vector
  SDTCisPtrTy<1>,          // Result 1: updated pointer
  SDTCisPtrTy<2>,          // Operand 0: pointer
  SDTCisVT<3, i32>         // Operand 1: offset (register)
]>;
def SDT_ESPVLDBC32IPM : SDTypeProfile<2, 2, [
  SDTCisVT<0, v4i32>,      // Result 0: vector
  SDTCisPtrTy<1>,          // Result 1: updated pointer
  SDTCisPtrTy<2>,          // Operand 0: pointer
  SDTCisVT<3, i32>         // Operand 1: offset
]>;
def SDT_ESPVLDBC32XPM : SDTypeProfile<2, 2, [
  SDTCisVT<0, v4i32>,      // Result 0: vector
  SDTCisPtrTy<1>,          // Result 1: updated pointer
  SDTCisPtrTy<2>,          // Operand 0: pointer
  SDTCisVT<3, i32>         // Operand 1: offset (register)
]>;
def riscv_esp_vldbc_8_ip_m : RVSDNode<"ESP_VLDBC_8_IP_M", SDT_ESPVLDBC8IPM,
                                      [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def riscv_esp_vldbc_8_xp_m : RVSDNode<"ESP_VLDBC_8_XP_M", SDT_ESPVLDBC8XPM,
                                      [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def riscv_esp_vldbc_16_ip_m : RVSDNode<"ESP_VLDBC_16_IP_M", SDT_ESPVLDBC16IPM,
                                       [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def riscv_esp_vldbc_16_xp_m : RVSDNode<"ESP_VLDBC_16_XP_M", SDT_ESPVLDBC16XPM,
                                       [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def riscv_esp_vldbc_32_ip_m : RVSDNode<"ESP_VLDBC_32_IP_M", SDT_ESPVLDBC32IPM,
                                       [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def riscv_esp_vldbc_32_xp_m : RVSDNode<"ESP_VLDBC_32_XP_M", SDT_ESPVLDBC32XPM,
                                       [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;

// VLDEXT SDNodes
def SDT_ESPVLDEXTS8IPM : SDTypeProfile<3, 2, [
  SDTCisVT<0, v8i16>,      // Result 0: vector (first)
  SDTCisVT<1, v8i16>,      // Result 1: vector (second)
  SDTCisPtrTy<2>,          // Result 2: updated pointer
  SDTCisPtrTy<3>,          // Operand 0: pointer
  SDTCisVT<4, i32>         // Operand 1: offset
]>;
def SDT_ESPVLDEXTS8XPM : SDTypeProfile<3, 2, [
  SDTCisVT<0, v8i16>,      // Result 0: vector (first)
  SDTCisVT<1, v8i16>,      // Result 1: vector (second)
  SDTCisPtrTy<2>,          // Result 2: updated pointer
  SDTCisPtrTy<3>,          // Operand 0: pointer
  SDTCisVT<4, i32>         // Operand 1: offset (register)
]>;
def SDT_ESPVLDEXTS16IPM : SDTypeProfile<3, 2, [
  SDTCisVT<0, v4i32>,      // Result 0: vector (first)
  SDTCisVT<1, v4i32>,      // Result 1: vector (second)
  SDTCisPtrTy<2>,          // Result 2: updated pointer
  SDTCisPtrTy<3>,          // Operand 0: pointer
  SDTCisVT<4, i32>         // Operand 1: offset
]>;
def SDT_ESPVLDEXTS16XPM : SDTypeProfile<3, 2, [
  SDTCisVT<0, v4i32>,      // Result 0: vector (first)
  SDTCisVT<1, v4i32>,      // Result 1: vector (second)
  SDTCisPtrTy<2>,          // Result 2: updated pointer
  SDTCisPtrTy<3>,          // Operand 0: pointer
  SDTCisVT<4, i32>         // Operand 1: offset (register)
]>;
def SDT_ESPVLDEXTU8IPM : SDTypeProfile<3, 2, [
  SDTCisVT<0, v8i16>,      // Result 0: vector (first)
  SDTCisVT<1, v8i16>,      // Result 1: vector (second)
  SDTCisPtrTy<2>,          // Result 2: updated pointer
  SDTCisPtrTy<3>,          // Operand 0: pointer
  SDTCisVT<4, i32>         // Operand 1: offset
]>;
def SDT_ESPVLDEXTU8XPM : SDTypeProfile<3, 2, [
  SDTCisVT<0, v8i16>,      // Result 0: vector (first)
  SDTCisVT<1, v8i16>,      // Result 1: vector (second)
  SDTCisPtrTy<2>,          // Result 2: updated pointer
  SDTCisPtrTy<3>,          // Operand 0: pointer
  SDTCisVT<4, i32>         // Operand 1: offset (register)
]>;
def SDT_ESPVLDEXTU16IPM : SDTypeProfile<3, 2, [
  SDTCisVT<0, v4i32>,      // Result 0: vector (first)
  SDTCisVT<1, v4i32>,      // Result 1: vector (second)
  SDTCisPtrTy<2>,          // Result 2: updated pointer
  SDTCisPtrTy<3>,          // Operand 0: pointer
  SDTCisVT<4, i32>         // Operand 1: offset
]>;
def SDT_ESPVLDEXTU16XPM : SDTypeProfile<3, 2, [
  SDTCisVT<0, v4i32>,      // Result 0: vector (first)
  SDTCisVT<1, v4i32>,      // Result 1: vector (second)
  SDTCisPtrTy<2>,          // Result 2: updated pointer
  SDTCisPtrTy<3>,          // Operand 0: pointer
  SDTCisVT<4, i32>         // Operand 1: offset (register)
]>;
def riscv_esp_vldext_s8_ip_m : RVSDNode<"ESP_VLDEXT_S8_IP_M", SDT_ESPVLDEXTS8IPM,
                                        [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def riscv_esp_vldext_s8_xp_m : RVSDNode<"ESP_VLDEXT_S8_XP_M", SDT_ESPVLDEXTS8XPM,
                                        [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def riscv_esp_vldext_s16_ip_m : RVSDNode<"ESP_VLDEXT_S16_IP_M", SDT_ESPVLDEXTS16IPM,
                                         [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def riscv_esp_vldext_s16_xp_m : RVSDNode<"ESP_VLDEXT_S16_XP_M", SDT_ESPVLDEXTS16XPM,
                                         [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def riscv_esp_vldext_u8_ip_m : RVSDNode<"ESP_VLDEXT_U8_IP_M", SDT_ESPVLDEXTU8IPM,
                                        [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def riscv_esp_vldext_u8_xp_m : RVSDNode<"ESP_VLDEXT_U8_XP_M", SDT_ESPVLDEXTU8XPM,
                                        [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def riscv_esp_vldext_u16_ip_m : RVSDNode<"ESP_VLDEXT_U16_IP_M", SDT_ESPVLDEXTU16IPM,
                                         [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def riscv_esp_vldext_u16_xp_m : RVSDNode<"ESP_VLDEXT_U16_XP_M", SDT_ESPVLDEXTU16XPM,
                                         [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def SDT_ESPSTQACCHH128IPM : SDTypeProfile<1, 3, [
  SDTCisPtrTy<2>,          // Operand 1: rs1 pointer
  SDTCisVT<3, i32>         // Operand 2: offset immediate
]>;
def SDT_ESPSTQACCLL128IPM : SDTypeProfile<1, 3, [
  SDTCisPtrTy<0>,          // Result 0: updated pointer
  SDTCisVT<1, v16i8>,      // Operand 0: QACC_L_LOW (128-bit) - extracted from QACC_L in Lowering
  SDTCisPtrTy<2>,          // Operand 1: rs1 pointer
  SDTCisVT<3, i32>         // Operand 2: offset immediate
]>;
def riscv_esp_st_qacc_h_h_128_ip_m : RVSDNode<"ESP_ST_QACC_H_H_128_IP_M", SDT_ESPSTQACCHH128IPM,
                                               [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;
def riscv_esp_st_qacc_l_l_128_ip_m : RVSDNode<"ESP_ST_QACC_L_L_128_IP_M", SDT_ESPSTQACCLL128IPM,
                                               [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;
// LD/ST XACC SDNodes
// Mixed model: XACC as {i32 low, i32 high}
def SDT_ESPSTXACCIPM : SDTypeProfile<3, 4, [
  SDTCisPtrTy<0>,          // Result 0: updated pointer
  SDTCisVT<1, i32>,        // Result 1: XACC low unchanged (i32, equals input)
  SDTCisVT<2, i32>,        // Result 2: XACC high unchanged (i32, equals input, only low 8 bits valid)
  SDTCisVT<3, i32>,        // Operand 0: XACC low passthru (i32)
  SDTCisVT<4, i32>,        // Operand 1: XACC high passthru (i32, only low 8 bits valid)
  SDTCisPtrTy<5>,          // Operand 2: rs1 pointer
  SDTCisVT<6, i32>         // Operand 3: offset immediate
]>;
def riscv_esp_st_s_xacc_ip_m : RVSDNode<"ESP_ST_S_XACC_IP_M", SDT_ESPSTXACCIPM,
                                         [SDNPHasChain, SDNPMayStore, SDNPMemOperand,
                                          SDNPOptInGlue, SDNPOutGlue]>;
def riscv_esp_st_u_xacc_ip_m : RVSDNode<"ESP_ST_U_XACC_IP_M", SDT_ESPSTXACCIPM,
                                         [SDNPHasChain, SDNPMayStore, SDNPMemOperand,
                                          SDNPOptInGlue, SDNPOutGlue]>;
def SDT_ESPVCMULASS16QACCLM : SDTypeProfile<2, 4, [
  SDTCisVT<0, v16i8>,        // Result 0: QACC_L[127:0] (128-bit)
  SDTCisVT<1, v16i8>,        // Result 1: QACC_L[255:128] (128-bit)
  SDTCisVT<2, v16i8>,        // Operand 0: QACC_L[127:0] passthru (phantom operand)
  SDTCisVT<3, v16i8>,        // Operand 1: QACC_L[255:128] passthru (phantom operand)
  SDTCisVT<4, v8i16>,        // Operand 2: qx vector (8x i16)
  SDTCisVT<5, v8i16>         // Operand 3: qy vector (8x i16)
]>;
def riscv_esp_vcmulas_s16_qacc_l_m : RVSDNode<"ESP_VCMULAS_S16_QACC_L_M", SDT_ESPVCMULASS16QACCLM>;

def SDT_ESPVCMULASS16QACCHM : SDTypeProfile<2, 4, [
  SDTCisVT<0, v16i8>,        // Result 0: QACC_H[127:0] (128-bit)
  SDTCisVT<1, v16i8>,        // Result 1: QACC_H[255:128] (128-bit)
  SDTCisVT<2, v16i8>,        // Operand 0: QACC_H[127:0] passthru (phantom operand)
  SDTCisVT<3, v16i8>,        // Operand 1: QACC_H[255:128] passthru (phantom operand)
  SDTCisVT<4, v8i16>,        // Operand 2: qx vector (8x i16)
  SDTCisVT<5, v8i16>         // Operand 3: qy vector (8x i16)
]>;
def riscv_esp_vcmulas_s16_qacc_h_m : RVSDNode<"ESP_VCMULAS_S16_QACC_H_M", SDT_ESPVCMULASS16QACCHM>;

def SDT_ESPVCMULASS8QACCLM : SDTypeProfile<2, 4, [
  SDTCisVT<0, v16i8>,        // Result 0: QACC_L[127:0] (128-bit)
  SDTCisVT<1, v16i8>,        // Result 1: QACC_L[255:128] (128-bit)
  SDTCisVT<2, v16i8>,        // Operand 0: QACC_L[127:0] passthru (phantom operand)
  SDTCisVT<3, v16i8>,        // Operand 1: QACC_L[255:128] passthru (phantom operand)
  SDTCisVT<4, v16i8>,        // Operand 2: qx vector (16x i8)
  SDTCisVT<5, v16i8>         // Operand 3: qy vector (16x i8)
]>;
def riscv_esp_vcmulas_s8_qacc_l_m : RVSDNode<"ESP_VCMULAS_S8_QACC_L_M", SDT_ESPVCMULASS8QACCLM>;

def SDT_ESPVCMULASS8QACCHM : SDTypeProfile<2, 4, [
  SDTCisVT<0, v16i8>,        // Result 0: QACC_H[127:0] (128-bit)
  SDTCisVT<1, v16i8>,        // Result 1: QACC_H[255:128] (128-bit)
  SDTCisVT<2, v16i8>,        // Operand 0: QACC_H[127:0] passthru (phantom operand)
  SDTCisVT<3, v16i8>,        // Operand 1: QACC_H[255:128] passthru (phantom operand)
  SDTCisVT<4, v16i8>,        // Operand 2: qx vector (16x i8)
  SDTCisVT<5, v16i8>         // Operand 3: qy vector (16x i8)
]>;
def riscv_esp_vcmulas_s8_qacc_h_m : RVSDNode<"ESP_VCMULAS_S8_QACC_H_M", SDT_ESPVCMULASS8QACCHM>;

// FFT.BITREV SDNode with explicit FFT_BIT_WIDTH state passing
// Results: rs1r (pointer), qv (v8i16)
// Operands: rs1 (pointer), fft_bit_width (i32)
// Note: FFT_BIT_WIDTH is passed explicitly as FFT_BIT_WIDTHReg (i32) for explicit state passing
// Note: Return order matches instruction definition: (rs1r, qvr)
// Note: This is a computation-only instruction, does not access memory
// Note: No SDNPHasChain because it doesn't access memory
def SDT_ESPFFTBITREVM : SDTypeProfile<2, 2, [
  SDTCisPtrTy<0>,          // Result 0: updated pointer
  SDTCisVT<1, v8i16>,      // Result 1: qv vector
  SDTCisPtrTy<2>,          // Operand 0: rs1 pointer
  SDTCisVT<3, i32>         // Operand 1: fft_bit_width (i32, only low 4 bits used)
]>;
def riscv_esp_fft_bitrev_m : RVSDNode<"ESP_FFT_BITREV_M", SDT_ESPFFTBITREVM, []>;

