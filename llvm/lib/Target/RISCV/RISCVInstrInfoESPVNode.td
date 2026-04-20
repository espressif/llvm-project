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
// ESP.MOV.S16.QACC SDNode
// Returns: 4x128-bit QACC (QACC_L_LOW, QACC_L_HIGH, QACC_H_LOW, QACC_H_HIGH)
// Operands: v8i16 (input vector)
def SDT_ESPMOVS16QACCM : SDTypeProfile<4, 1, [
  SDTCisVT<0, v16i8>,      // Result 0: QACC_L_LOW
  SDTCisVT<1, v16i8>,      // Result 1: QACC_L_HIGH
  SDTCisVT<2, v16i8>,      // Result 2: QACC_H_LOW
  SDTCisVT<3, v16i8>,      // Result 3: QACC_H_HIGH
  SDTCisVT<4, v8i16>       // Operand 0: v8i16 input vector
]>;
def riscv_esp_mov_s16_qacc_m : RVSDNode<"ESP_MOV_S16_QACC_M", SDT_ESPMOVS16QACCM>;
// ESP.MOV.S8.QACC SDNode
// Returns: 4x128-bit QACC (QACC_L_LOW, QACC_L_HIGH, QACC_H_LOW, QACC_H_HIGH)
// Operands: v16i8 (input vector)
def SDT_ESPMOVS8QACCM : SDTypeProfile<4, 1, [
  SDTCisVT<0, v16i8>,      // Result 0: QACC_L_LOW
  SDTCisVT<1, v16i8>,      // Result 1: QACC_L_HIGH
  SDTCisVT<2, v16i8>,      // Result 2: QACC_H_LOW
  SDTCisVT<3, v16i8>,      // Result 3: QACC_H_HIGH
  SDTCisVT<4, v16i8>       // Operand 0: v16i8 input vector
]>;
def riscv_esp_mov_s8_qacc_m : RVSDNode<"ESP_MOV_S8_QACC_M", SDT_ESPMOVS8QACCM>;
// ESP.MOV.U16.QACC SDNode
// Returns: 4x128-bit QACC (QACC_L_LOW, QACC_L_HIGH, QACC_H_LOW, QACC_H_HIGH)
// Operands: v8i16 (input vector)
def SDT_ESPMOVU16QACCM : SDTypeProfile<4, 1, [
  SDTCisVT<0, v16i8>,      // Result 0: QACC_L_LOW
  SDTCisVT<1, v16i8>,      // Result 1: QACC_L_HIGH
  SDTCisVT<2, v16i8>,      // Result 2: QACC_H_LOW
  SDTCisVT<3, v16i8>,      // Result 3: QACC_H_HIGH
  SDTCisVT<4, v8i16>       // Operand 0: v8i16 input vector
]>;
def riscv_esp_mov_u16_qacc_m : RVSDNode<"ESP_MOV_U16_QACC_M", SDT_ESPMOVU16QACCM>;
// ESP.MOV.U8.QACC SDNode
// Returns: 4x128-bit QACC (QACC_L_LOW, QACC_L_HIGH, QACC_H_LOW, QACC_H_HIGH)
// Operands: v16i8 (input vector)
def SDT_ESPMOVU8QACCM : SDTypeProfile<4, 1, [
  SDTCisVT<0, v16i8>,      // Result 0: QACC_L_LOW
  SDTCisVT<1, v16i8>,      // Result 1: QACC_L_HIGH
  SDTCisVT<2, v16i8>,      // Result 2: QACC_H_LOW
  SDTCisVT<3, v16i8>,      // Result 3: QACC_H_HIGH
  SDTCisVT<4, v16i8>       // Operand 0: v16i8 input vector
]>;
def riscv_esp_mov_u8_qacc_m : RVSDNode<"ESP_MOV_U8_QACC_M", SDT_ESPMOVU8QACCM>;
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
def SDT_ESPLDQACCHH128IPM : SDTypeProfile<2, 2, [
  SDTCisVT<0, v16i8>,      // Result 0: QACC_H[255:128] value (128-bit subregister)
  SDTCisPtrTy<1>,          // Result 1: updated pointer
  SDTCisPtrTy<2>,          // Operand 0: rs1 pointer
  SDTCisVT<3, i32>         // Operand 1: offset immediate
]>;
def SDT_ESPLDQACCHL128IPM : SDTypeProfile<2, 2, [
  SDTCisVT<0, v16i8>,      // Result 0: QACC_H[127:0] value (128-bit subregister)
  SDTCisPtrTy<1>,          // Result 1: updated pointer
  SDTCisPtrTy<2>,          // Operand 0: rs1 pointer
  SDTCisVT<3, i32>         // Operand 1: offset immediate
]>;
def SDT_ESPLDQACCLH128IPM : SDTypeProfile<2, 2, [
  SDTCisVT<0, v16i8>,      // Result 0: QACC_L[255:128] value (128-bit subregister)
  SDTCisPtrTy<1>,          // Result 1: updated pointer
  SDTCisPtrTy<2>,          // Operand 0: rs1 pointer
  SDTCisVT<3, i32>         // Operand 1: offset immediate
]>;
def SDT_ESPLDQACCLL128IPM : SDTypeProfile<2, 2, [
  SDTCisVT<0, v16i8>,      // Result 0: QACC_L[127:0] value (128-bit subregister)
  SDTCisPtrTy<1>,          // Result 1: updated pointer
  SDTCisPtrTy<2>,          // Operand 0: rs1 pointer
  SDTCisVT<3, i32>         // Operand 1: offset immediate
]>;
def SDT_ESPSTQACCHH128IPM : SDTypeProfile<1, 3, [
  SDTCisPtrTy<0>,          // Result 0: updated pointer
  SDTCisVT<1, v16i8>,      // Operand 0: QACC_H[255:128] (128-bit) - Subregister modeling
  SDTCisPtrTy<2>,          // Operand 1: rs1 pointer
  SDTCisVT<3, i32>         // Operand 2: offset immediate
]>;
def SDT_ESPSTQACCHL128IPM : SDTypeProfile<1, 3, [
  SDTCisPtrTy<0>,          // Result 0: updated pointer
  SDTCisVT<1, v16i8>,      // Operand 0: QACC_H[127:0] (128-bit) - Subregister modeling
  SDTCisPtrTy<2>,          // Operand 1: rs1 pointer
  SDTCisVT<3, i32>         // Operand 2: offset immediate
]>;
def SDT_ESPSTQACCLH128IPM : SDTypeProfile<1, 3, [
  SDTCisPtrTy<0>,          // Result 0: updated pointer
  SDTCisVT<1, v16i8>,      // Operand 0: QACC_L_HIGH (128-bit) - extracted from QACC_L in Lowering
  SDTCisPtrTy<2>,          // Operand 1: rs1 pointer
  SDTCisVT<3, i32>         // Operand 2: offset immediate
]>;
def SDT_ESPSTQACCLL128IPM : SDTypeProfile<1, 3, [
  SDTCisPtrTy<0>,          // Result 0: updated pointer
  SDTCisVT<1, v16i8>,      // Operand 0: QACC_L_LOW (128-bit) - extracted from QACC_L in Lowering
  SDTCisPtrTy<2>,          // Operand 1: rs1 pointer
  SDTCisVT<3, i32>         // Operand 2: offset immediate
]>;

def riscv_esp_ld_qacc_h_h_128_ip_m : RVSDNode<"ESP_LD_QACC_H_H_128_IP_M", SDT_ESPLDQACCHH128IPM,
                                               [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def riscv_esp_ld_qacc_h_l_128_ip_m : RVSDNode<"ESP_LD_QACC_H_L_128_IP_M", SDT_ESPLDQACCHL128IPM,
                                               [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def riscv_esp_ld_qacc_l_h_128_ip_m : RVSDNode<"ESP_LD_QACC_L_H_128_IP_M", SDT_ESPLDQACCLH128IPM,
                                               [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def riscv_esp_ld_qacc_l_l_128_ip_m : RVSDNode<"ESP_LD_QACC_L_L_128_IP_M", SDT_ESPLDQACCLL128IPM,
                                               [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def riscv_esp_st_qacc_h_h_128_ip_m : RVSDNode<"ESP_ST_QACC_H_H_128_IP_M", SDT_ESPSTQACCHH128IPM,
                                               [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;
def riscv_esp_st_qacc_h_l_128_ip_m : RVSDNode<"ESP_ST_QACC_H_L_128_IP_M", SDT_ESPSTQACCHL128IPM,
                                               [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;
def riscv_esp_st_qacc_l_h_128_ip_m : RVSDNode<"ESP_ST_QACC_L_H_128_IP_M", SDT_ESPSTQACCLH128IPM,
                                               [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;
def riscv_esp_st_qacc_l_l_128_ip_m : RVSDNode<"ESP_ST_QACC_L_L_128_IP_M", SDT_ESPSTQACCLL128IPM,
                                               [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;
// LD/ST XACC SDNodes

// FFT.AMS.S16.LD.INCP SDNode with explicit SAR state passing
// Results: qu (v16i8), qz (v8i16), qv (v8i16), rs1r (pointer)
// Operands: qx (v8i16), qy (v8i16), qw (v8i16), rs1 (pointer), sel2 (i32 immediate), sar (i32)
def SDT_ESPFFTAMSS16LDINCPM : SDTypeProfile<4, 6, [
  SDTCisVT<0, v16i8>,      // Result 0: qu vector (loaded from memory)
  SDTCisVT<1, v8i16>,      // Result 1: qz vector (computation result)
  SDTCisVT<2, v8i16>,      // Result 2: qv vector (computation result)
  SDTCisPtrTy<3>,          // Result 3: updated pointer
  SDTCisVT<4, v8i16>,      // Operand 0: qx vector
  SDTCisVT<5, v8i16>,      // Operand 1: qy vector
  SDTCisVT<6, v8i16>,      // Operand 2: qw vector
  SDTCisPtrTy<7>,          // Operand 3: rs1 pointer
  SDTCisVT<8, i32>,        // Operand 4: sel2 immediate
  SDTCisVT<9, i32>         // Operand 5: sar (SAR register value, explicit state passing, i32, only low 6 bits used)
]>;
def riscv_esp_fft_ams_s16_ld_incp_m : RVSDNode<"ESP_FFT_AMS_S16_LD_INCP_M", SDT_ESPFFTAMSS16LDINCPM,
                                               [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;

// FFT.R2BF.S16 SDNode
// Results: qz (v8i16), qv (v8i16)
// Operands: qx (v8i16), qy (v8i16), sel2 (i32 immediate)
// Note: This is a computation-only instruction, does not access memory
// Note: No chain needed since it doesn't access memory
def SDT_ESPFFTR2BFS16M : SDTypeProfile<2, 3, [
  SDTCisVT<0, v8i16>,      // Result 0: qz vector
  SDTCisVT<1, v8i16>,      // Result 1: qv vector
  SDTCisVT<2, v8i16>,      // Operand 0: qx vector
  SDTCisVT<3, v8i16>,      // Operand 1: qy vector
  SDTCisVT<4, i32>         // Operand 2: sel2 immediate
]>;
def riscv_esp_fft_r2bf_s16_m : RVSDNode<"ESP_FFT_R2BF_S16_M", SDT_ESPFFTR2BFS16M, []>;

// FFT.R2BF.S16.ST.INCP SDNode
// Results: qz (v8i16), rs1r (pointer)
// Operands: qx (v8i16), qy (v8i16), rs1 (pointer), sel4 (i32 immediate)
def SDT_ESPFFTR2BFS16STINCPM : SDTypeProfile<2, 4, [
  SDTCisVT<0, v8i16>,      // Result 0: qz vector
  SDTCisPtrTy<1>,          // Result 1: updated pointer
  SDTCisVT<2, v8i16>,      // Operand 0: qx vector
  SDTCisVT<3, v8i16>,      // Operand 1: qy vector
  SDTCisPtrTy<4>,          // Operand 2: rs1 pointer
  SDTCisVT<5, i32>         // Operand 3: sel4 immediate
]>;
def riscv_esp_fft_r2bf_s16_st_incp_m : RVSDNode<"ESP_FFT_R2BF_S16_ST_INCP_M", SDT_ESPFFTR2BFS16STINCPM,
                                                 [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;

// FFT.AMS.S16.LD.R32.DECP SDNode (without explicit SAR - uses implicit SAR)
// Results: qu (v16i8), qz (v8i16), qv (v8i16), rs1r (pointer)
// Operands: qx (v8i16), qy (v8i16), qw (v8i16), rs1 (pointer), sel2 (i32 immediate)
// Note: Uses implicit SAR register (not explicit state passing)
// Note: Same type profile as LD.INCP but decrements pointer instead of incrementing
// SAR is passed explicitly as phantom operand for explicit state passing
def SDT_ESPFFTAMSS16LDR32DECPM : SDTypeProfile<4, 6, [
  SDTCisVT<0, v16i8>,      // Result 0: qu vector (loaded from memory)
  SDTCisVT<1, v8i16>,      // Result 1: qz vector (computation result)
  SDTCisVT<2, v8i16>,      // Result 2: qv vector (computation result)
  SDTCisPtrTy<3>,          // Result 3: updated pointer
  SDTCisVT<4, v8i16>,      // Operand 0: qx vector
  SDTCisVT<5, v8i16>,      // Operand 1: qy vector
  SDTCisVT<6, v8i16>,      // Operand 2: qw vector
  SDTCisPtrTy<7>,          // Operand 3: rs1 pointer
  SDTCisVT<8, i32>,        // Operand 4: sel2 immediate
  SDTCisVT<9, i32>         // Operand 5: sar (SAR register value, explicit state passing, i32, only low 6 bits used)
]>;
def riscv_esp_fft_ams_s16_ld_r32_decp_m : RVSDNode<"ESP_FFT_AMS_S16_LD_R32_DECP_M", SDT_ESPFFTAMSS16LDR32DECPM,
                                                    [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;

// FFT.AMS.S16.ST.INCP SDNode with explicit SAR state passing
// Results: qz (v8i16), rs1r (pointer)
// Operands: qx (v8i16), qy (v8i16), qw (v8i16), qu (v16i8), rs1 (pointer), rs2 (pointer), sel2 (i32 immediate), sar (i32)
// Note: rs2 is updated with computation result, but it's handled at instruction level
def SDT_ESPFFTAMSS16STINCPM : SDTypeProfile<2, 8, [
  SDTCisVT<0, v8i16>,      // Result 0: qz vector
  SDTCisPtrTy<1>,          // Result 1: updated pointer
  SDTCisVT<2, v8i16>,      // Operand 0: qx vector
  SDTCisVT<3, v8i16>,      // Operand 1: qy vector
  SDTCisVT<4, v8i16>,      // Operand 2: qw vector
  SDTCisVT<5, v16i8>,      // Operand 3: qu vector
  SDTCisPtrTy<6>,          // Operand 4: rs1 pointer
  SDTCisPtrTy<7>,          // Operand 5: rs2 pointer
  SDTCisVT<8, i32>,        // Operand 6: sel2 immediate
  SDTCisVT<9, i32>         // Operand 7: sar (SAR register value, explicit state passing, i32, only low 6 bits used)
]>;
def riscv_esp_fft_ams_s16_st_incp_m : RVSDNode<"ESP_FFT_AMS_S16_ST_INCP_M", SDT_ESPFFTAMSS16STINCPM,
                                                [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;

// FFT.AMS.S16.LD.INCP.UAUP SDNode
// Results: qu (v16i8), qz (v8i16), qv (v8i16), rs1r (ptr), ua_state_out (v16i8)
// Operands: qx (v8i16), qy (v8i16), qw (v8i16), rs1 (ptr), sel2 (i32), ua_state_in (v16i8), sar_bytes_in (i32), sar_in (i32)
// Note: UA_STATE is input/output (phantom operand), SAR_BYTES and SAR are input-only (phantom operands)
def SDT_ESPFFTAMSS16LDINCPUAUP : SDTypeProfile<5, 8, [
  SDTCisVT<0, v16i8>,      // Result 0: qu vector (loaded from memory)
  SDTCisVT<1, v8i16>,      // Result 1: qz vector (computation result)
  SDTCisVT<2, v8i16>,      // Result 2: qv vector (computation result)
  SDTCisPtrTy<3>,          // Result 3: updated pointer
  SDTCisVT<4, v16i8>,      // Result 4: ua_state_out (phantom operand output)
  SDTCisVT<5, v8i16>,      // Operand 0: qx vector
  SDTCisVT<6, v8i16>,      // Operand 1: qy vector
  SDTCisVT<7, v8i16>,      // Operand 2: qw vector
  SDTCisPtrTy<8>,          // Operand 3: rs1 pointer
  SDTCisVT<9, i32>,        // Operand 4: sel2 immediate
  SDTCisVT<10, v16i8>,     // Operand 5: ua_state_in (phantom operand input)
  SDTCisVT<11, i32>,       // Operand 6: sar_bytes_in (phantom operand input, read-only, i32, only low 4 bits used, range 0-15)
  SDTCisVT<12, i32>        // Operand 7: sar_in (phantom operand input, read-only, i32, only low 6 bits used)
]>;
def riscv_esp_fft_ams_s16_ld_incp_uaup_m : RVSDNode<"ESP_FFT_AMS_S16_LD_INCP_UAUP_M", SDT_ESPFFTAMSS16LDINCPUAUP,
                                                     [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;

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


// FFT.CMUL.S16.LD.XP SDNode with explicit SAR state passing
// Results: qz (v8i16), qu (v16i8), rs1r (pointer)
// Operands: qx (v8i16), qy (v8i16), rs1 (pointer), rs2 (pointer), sel8 (i32 immediate), sar (i32)
def SDT_ESPFFTCMULS16LDXPM : SDTypeProfile<3, 6, [
  SDTCisVT<0, v8i16>,      // Result 0: qz vector
  SDTCisVT<1, v16i8>,      // Result 1: qu vector (loaded from memory)
  SDTCisPtrTy<2>,          // Result 2: updated pointer
  SDTCisVT<3, v8i16>,      // Operand 0: qx vector
  SDTCisVT<4, v8i16>,      // Operand 1: qy vector
  SDTCisPtrTy<5>,          // Operand 2: rs1 pointer
  SDTCisPtrTy<6>,          // Operand 3: rs2 pointer
  SDTCisVT<7, i32>,        // Operand 4: sel8 immediate
  SDTCisVT<8, i32>         // Operand 5: sar (SAR register value, explicit state passing, i32, only low 6 bits used)
]>;
def riscv_esp_fft_cmul_s16_ld_xp_m : RVSDNode<"ESP_FFT_CMUL_S16_LD_XP_M", SDT_ESPFFTCMULS16LDXPM,
                                               [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;

// FFT.CMUL.S16.ST.XP SDNode with explicit SAR state passing
// Results: rs1r (pointer)
// Operands: qx (v8i16), qy (v8i16), qu (v16i8), rs1 (pointer), rs2 (pointer), sel8 (i32), upd4 (i32), sel4 (i32), sar (i32)
def SDT_ESPFFTCMULS16STXPM : SDTypeProfile<1, 9, [
  SDTCisPtrTy<0>,          // Result 0: updated pointer
  SDTCisVT<1, v8i16>,      // Operand 0: qx vector
  SDTCisVT<2, v8i16>,      // Operand 1: qy vector
  SDTCisVT<3, v16i8>,      // Operand 2: qu vector
  SDTCisPtrTy<4>,          // Operand 3: rs1 pointer
  SDTCisPtrTy<5>,          // Operand 4: rs2 pointer
  SDTCisVT<6, i32>,        // Operand 5: sel8 immediate
  SDTCisVT<7, i32>,        // Operand 6: upd4 immediate
  SDTCisVT<8, i32>,        // Operand 7: sel4 immediate
  SDTCisVT<9, i32>         // Operand 8: sar (SAR register value, explicit state passing, i32, only low 6 bits used)
]>;
def riscv_esp_fft_cmul_s16_st_xp_m : RVSDNode<"ESP_FFT_CMUL_S16_ST_XP_M", SDT_ESPFFTCMULS16STXPM,
                                               [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;

// FFT.VST.R32.DECP SDNode
// Results: rs1r (pointer)
// Operands: qu (v16i8), rs1 (pointer), sel2 (i32 immediate)
def SDT_ESPFFTVSTR32DECPM : SDTypeProfile<1, 3, [
  SDTCisPtrTy<0>,          // Result 0: updated pointer
  SDTCisVT<1, v16i8>,      // Operand 0: qu vector
  SDTCisPtrTy<2>,          // Operand 1: rs1 pointer
  SDTCisVT<3, i32>         // Operand 2: sel2 immediate
]>;
def riscv_esp_fft_vst_r32_decp_m : RVSDNode<"ESP_FFT_VST_R32_DECP_M", SDT_ESPFFTVSTR32DECPM,
                                             [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;
// VADD LD INCP SDNodes
// Results: qv (vector), qu (vector), rs1r (pointer)
// Operands: qx (vector), qy (vector), rs1 (pointer)
def SDT_ESPVADDS8LDINCPM : SDTypeProfile<3, 3, [
  SDTCisVT<0, v16i8>,      // Result 0: qv vector
  SDTCisVT<1, v16i8>,      // Result 1: qu vector (16 bytes loaded from memory)
  SDTCisPtrTy<2>,          // Result 2: updated pointer
  SDTCisVT<3, v16i8>,      // Operand 0: qx vector
  SDTCisVT<4, v16i8>,      // Operand 1: qy vector
  SDTCisPtrTy<5>           // Operand 2: rs1 pointer
]>;
def riscv_esp_vadd_s8_ld_incp_m : RVSDNode<"ESP_VADD_S8_LD_INCP_M", SDT_ESPVADDS8LDINCPM,
                                           [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
// VADD LD INCP SDNodes (s16, u16, u8)
def SDT_ESPVADDS16LDINCPM : SDTypeProfile<3, 3, [
  SDTCisVT<0, v8i16>,      // Result 0: qv vector
  SDTCisVT<1, v16i8>,      // Result 1: qu vector
  SDTCisPtrTy<2>,          // Result 2: updated pointer
  SDTCisVT<3, v8i16>,      // Operand 0: qx vector
  SDTCisVT<4, v8i16>,      // Operand 1: qy vector
  SDTCisPtrTy<5>           // Operand 2: rs1 pointer
]>;
def SDT_ESPVADDU16LDINCPM : SDTypeProfile<3, 3, [
  SDTCisVT<0, v8i16>,      // Result 0: qv vector
  SDTCisVT<1, v16i8>,      // Result 1: qu vector
  SDTCisPtrTy<2>,          // Result 2: updated pointer
  SDTCisVT<3, v8i16>,      // Operand 0: qx vector
  SDTCisVT<4, v8i16>,      // Operand 1: qy vector
  SDTCisPtrTy<5>           // Operand 2: rs1 pointer
]>;
def SDT_ESPVADDU8LDINCPM : SDTypeProfile<3, 3, [
  SDTCisVT<0, v16i8>,      // Result 0: qv vector
  SDTCisVT<1, v16i8>,      // Result 1: qu vector
  SDTCisPtrTy<2>,          // Result 2: updated pointer
  SDTCisVT<3, v16i8>,      // Operand 0: qx vector
  SDTCisVT<4, v16i8>,      // Operand 1: qy vector
  SDTCisPtrTy<5>           // Operand 2: rs1 pointer
]>;
def riscv_esp_vadd_s16_ld_incp_m : RVSDNode<"ESP_VADD_S16_LD_INCP_M", SDT_ESPVADDS16LDINCPM,
                                           [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def riscv_esp_vadd_u16_ld_incp_m : RVSDNode<"ESP_VADD_U16_LD_INCP_M", SDT_ESPVADDU16LDINCPM,
                                           [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def riscv_esp_vadd_u8_ld_incp_m : RVSDNode<"ESP_VADD_U8_LD_INCP_M", SDT_ESPVADDU8LDINCPM,
                                           [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
// VADD ST INCP SDNodes
// Results: qv (vector), rs1r (pointer)
// Operands: qx (vector), qy (vector), qu (vector), rs1 (pointer)
def SDT_ESPVADDS8STINCPM : SDTypeProfile<2, 4, [
  SDTCisVT<0, v16i8>,      // Result 0: qv vector
  SDTCisPtrTy<1>,          // Result 1: updated pointer
  SDTCisVT<2, v16i8>,      // Operand 0: qx vector
  SDTCisVT<3, v16i8>,      // Operand 1: qy vector
  SDTCisVT<4, v16i8>,      // Operand 2: qu vector
  SDTCisPtrTy<5>           // Operand 3: rs1 pointer
]>;
def SDT_ESPVADDS16STINCPM : SDTypeProfile<2, 4, [
  SDTCisVT<0, v8i16>,      // Result 0: qv vector
  SDTCisPtrTy<1>,          // Result 1: updated pointer
  SDTCisVT<2, v8i16>,      // Operand 0: qx vector
  SDTCisVT<3, v8i16>,      // Operand 1: qy vector
  SDTCisVT<4, v16i8>,      // Operand 2: qu vector
  SDTCisPtrTy<5>           // Operand 3: rs1 pointer
]>;
def SDT_ESPVADDU16STINCPM : SDTypeProfile<2, 4, [
  SDTCisVT<0, v8i16>,      // Result 0: qv vector
  SDTCisPtrTy<1>,          // Result 1: updated pointer
  SDTCisVT<2, v8i16>,      // Operand 0: qx vector
  SDTCisVT<3, v8i16>,      // Operand 1: qy vector
  SDTCisVT<4, v16i8>,      // Operand 2: qu vector
  SDTCisPtrTy<5>           // Operand 3: rs1 pointer
]>;
def SDT_ESPVADDU8STINCPM : SDTypeProfile<2, 4, [
  SDTCisVT<0, v16i8>,      // Result 0: qv vector
  SDTCisPtrTy<1>,          // Result 1: updated pointer
  SDTCisVT<2, v16i8>,      // Operand 0: qx vector
  SDTCisVT<3, v16i8>,      // Operand 1: qy vector
  SDTCisVT<4, v16i8>,      // Operand 2: qu vector
  SDTCisPtrTy<5>           // Operand 3: rs1 pointer
]>;
def riscv_esp_vadd_s8_st_incp_m : RVSDNode<"ESP_VADD_S8_ST_INCP_M", SDT_ESPVADDS8STINCPM,
                                           [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;
def riscv_esp_vadd_s16_st_incp_m : RVSDNode<"ESP_VADD_S16_ST_INCP_M", SDT_ESPVADDS16STINCPM,
                                           [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;
def riscv_esp_vadd_u16_st_incp_m : RVSDNode<"ESP_VADD_U16_ST_INCP_M", SDT_ESPVADDU16STINCPM,
                                           [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;
def riscv_esp_vadd_u8_st_incp_m : RVSDNode<"ESP_VADD_U8_ST_INCP_M", SDT_ESPVADDU8STINCPM,
                                           [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;

// VCMULAS pure compute SDNodes (no memory access, explicit passthru)
// Results: v16i8, v16i8 (2x128-bit QACC directly)
// Operands: v16i8, v16i8 (2x128-bit QACC passthru), qx (vector), qy (vector)
def SDT_ESPVCMULASS8QACCHM : SDTypeProfile<2, 4, [
  SDTCisVT<0, v16i8>,        // Result 0: QACC_H[127:0] (128-bit)
  SDTCisVT<1, v16i8>,        // Result 1: QACC_H[255:128] (128-bit)
  SDTCisVT<2, v16i8>,        // Operand 0: QACC_H[127:0] passthru (phantom operand)
  SDTCisVT<3, v16i8>,        // Operand 1: QACC_H[255:128] passthru (phantom operand)
  SDTCisVT<4, v16i8>,        // Operand 2: qx vector
  SDTCisVT<5, v16i8>         // Operand 3: qy vector
]>;
def SDT_ESPVCMULASS8QACCLM : SDTypeProfile<2, 4, [
  SDTCisVT<0, v16i8>,        // Result 0: QACC_L[127:0] (128-bit)
  SDTCisVT<1, v16i8>,        // Result 1: QACC_L[255:128] (128-bit)
  SDTCisVT<2, v16i8>,        // Operand 0: QACC_L[127:0] passthru (phantom operand)
  SDTCisVT<3, v16i8>,        // Operand 1: QACC_L[255:128] passthru (phantom operand)
  SDTCisVT<4, v16i8>,        // Operand 2: qx vector
  SDTCisVT<5, v16i8>         // Operand 3: qy vector
]>;
def SDT_ESPVCMULASS16QACCHM : SDTypeProfile<2, 4, [
  SDTCisVT<0, v16i8>,        // Result 0: QACC_H[127:0] (128-bit)
  SDTCisVT<1, v16i8>,        // Result 1: QACC_H[255:128] (128-bit)
  SDTCisVT<2, v16i8>,        // Operand 0: QACC_H[127:0] passthru (phantom operand)
  SDTCisVT<3, v16i8>,        // Operand 1: QACC_H[255:128] passthru (phantom operand)
  SDTCisVT<4, v8i16>,        // Operand 2: qx vector (8x i16)
  SDTCisVT<5, v8i16>         // Operand 3: qy vector (8x i16)
]>;
def SDT_ESPVCMULASS16QACCLM : SDTypeProfile<2, 4, [
  SDTCisVT<0, v16i8>,        // Result 0: QACC_L[127:0] (128-bit)
  SDTCisVT<1, v16i8>,        // Result 1: QACC_L[255:128] (128-bit)
  SDTCisVT<2, v16i8>,        // Operand 0: QACC_L[127:0] passthru (phantom operand)
  SDTCisVT<3, v16i8>,        // Operand 1: QACC_L[255:128] passthru (phantom operand)
  SDTCisVT<4, v8i16>,        // Operand 2: qx vector (8x i16)
  SDTCisVT<5, v8i16>         // Operand 3: qy vector (8x i16)
]>;

def riscv_esp_vcmulas_s8_qacc_h_m : RVSDNode<"ESP_VCMULAS_S8_QACC_H_M", SDT_ESPVCMULASS8QACCHM>;
def riscv_esp_vcmulas_s8_qacc_l_m : RVSDNode<"ESP_VCMULAS_S8_QACC_L_M", SDT_ESPVCMULASS8QACCLM>;
def riscv_esp_vcmulas_s16_qacc_h_m : RVSDNode<"ESP_VCMULAS_S16_QACC_H_M", SDT_ESPVCMULASS16QACCHM>;
def riscv_esp_vcmulas_s16_qacc_l_m : RVSDNode<"ESP_VCMULAS_S16_QACC_L_M", SDT_ESPVCMULASS16QACCLM>;

// VCMULAS S8 QACC H LD IP SDNodes
// Results: qu (vector), ptr (updated pointer), v2 (QACC_H[127:0]), v3 (QACC_H[255:128])
// Operands: v2_in (QACC_H[127:0] passthru), v3_in (QACC_H[255:128] passthru), qx (v16i8 vector), qy (v16i8 vector), rs1 (pointer), offset (i32 immediate)
def SDT_ESPVCMULASS8QACCHLDIPM : SDTypeProfile<4, 6, [
  SDTCisVT<0, v16i8>,      // Result 0: qu vector (matches instruction def: outs QR:$qu)
  SDTCisPtrTy<1>,          // Result 1: updated pointer (matches instruction def: GPRPIE:$rs1r)
  SDTCisVT<2, v16i8>,      // Result 2: QACC_H[127:0] (matches instruction def: QACC_H_LOW:$qacc_h_l)
  SDTCisVT<3, v16i8>,      // Result 3: QACC_H[255:128] (matches instruction def: QACC_H_HIGH:$qacc_h_h)
  SDTCisVT<4, v16i8>,      // Operand 0: QACC_H[127:0] passthru (phantom operand)
  SDTCisVT<5, v16i8>,      // Operand 1: QACC_H[255:128] passthru (phantom operand)
  SDTCisVT<6, v16i8>,      // Operand 2: qx vector (16x i8)
  SDTCisVT<7, v16i8>,      // Operand 3: qy vector (16x i8)
  SDTCisPtrTy<8>,          // Operand 4: rs1 pointer
  SDTCisVT<9, i32>         // Operand 5: offset immediate
]>;
def riscv_esp_vcmulas_s8_qacc_h_ld_ip_m : RVSDNode<"ESP_VCMULAS_S8_QACC_H_LD_IP_M", SDT_ESPVCMULASS8QACCHLDIPM,
                                                    [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
// VCMULAS S8 QACC L LD IP SDNodes
// Results: qu (vector), ptr (updated pointer), v0 (QACC_L[127:0]), v1 (QACC_L[255:128])
// Operands: v0_in (QACC_L[127:0] passthru), v1_in (QACC_L[255:128] passthru), qx (v16i8 vector), qy (v16i8 vector), rs1 (pointer), offset (i32 immediate)
def SDT_ESPVCMULASS8QACCLDIPM : SDTypeProfile<4, 6, [
  SDTCisVT<0, v16i8>,      // Result 0: qu vector (matches instruction def: outs QR:$qu)
  SDTCisPtrTy<1>,          // Result 1: updated pointer (matches instruction def: GPRPIE:$rs1r)
  SDTCisVT<2, v16i8>,      // Result 2: QACC_L[127:0] (matches instruction def: QACC_L_LOW:$qacc_l_l)
  SDTCisVT<3, v16i8>,      // Result 3: QACC_L[255:128] (matches instruction def: QACC_L_HIGH:$qacc_l_h)
  SDTCisVT<4, v16i8>,      // Operand 0: QACC_L[127:0] passthru (phantom operand)
  SDTCisVT<5, v16i8>,      // Operand 1: QACC_L[255:128] passthru (phantom operand)
  SDTCisVT<6, v16i8>,      // Operand 2: qx vector (16x i8)
  SDTCisVT<7, v16i8>,      // Operand 3: qy vector (16x i8)
  SDTCisPtrTy<8>,          // Operand 4: rs1 pointer
  SDTCisVT<9, i32>         // Operand 5: offset immediate
]>;
def riscv_esp_vcmulas_s8_qacc_l_ld_ip_m : RVSDNode<"ESP_VCMULAS_S8_QACC_L_LD_IP_M", SDT_ESPVCMULASS8QACCLDIPM,
                                                    [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
// VCMULAS S16 QACC H LD IP SDNodes
// Results: qu (vector), ptr (updated pointer), v2 (QACC_H[127:0]), v3 (QACC_H[255:128])
// Operands: v2_in (QACC_H[127:0] passthru), v3_in (QACC_H[255:128] passthru), qx (v8i16 vector), qy (v8i16 vector), rs1 (pointer), offset (i32 immediate)
def SDT_ESPVCMULASS16QACCHLDIPM : SDTypeProfile<4, 6, [
  SDTCisVT<0, v16i8>,      // Result 0: qu vector (matches instruction def: outs QR:$qu)
  SDTCisPtrTy<1>,          // Result 1: updated pointer (matches instruction def: GPRPIE:$rs1r)
  SDTCisVT<2, v16i8>,      // Result 2: QACC_H[127:0] (matches instruction def: QACC_H_LOW:$qacc_h_l)
  SDTCisVT<3, v16i8>,      // Result 3: QACC_H[255:128] (matches instruction def: QACC_H_HIGH:$qacc_h_h)
  SDTCisVT<4, v16i8>,      // Operand 0: QACC_H[127:0] passthru (phantom operand)
  SDTCisVT<5, v16i8>,      // Operand 1: QACC_H[255:128] passthru (phantom operand)
  SDTCisVT<6, v8i16>,      // Operand 2: qx vector (8x i16)
  SDTCisVT<7, v8i16>,      // Operand 3: qy vector (8x i16)
  SDTCisPtrTy<8>,          // Operand 4: rs1 pointer
  SDTCisVT<9, i32>         // Operand 5: offset immediate
]>;
def riscv_esp_vcmulas_s16_qacc_h_ld_ip_m : RVSDNode<"ESP_VCMULAS_S16_QACC_H_LD_IP_M", SDT_ESPVCMULASS16QACCHLDIPM,
                                                     [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
// VCMULAS S16 QACC L LD IP SDNodes
// Results: qu (vector), ptr (updated pointer), v0 (QACC_L[127:0]), v1 (QACC_L[255:128])
// Operands: v0_in (QACC_L[127:0] passthru), v1_in (QACC_L[255:128] passthru), qx (v8i16 vector), qy (v8i16 vector), rs1 (pointer), offset (i32 immediate)
def SDT_ESPVCMULASS16QACCLDIPM : SDTypeProfile<4, 6, [
  SDTCisVT<0, v16i8>,      // Result 0: qu vector (matches instruction def: outs QR:$qu)
  SDTCisPtrTy<1>,          // Result 1: updated pointer (matches instruction def: GPRPIE:$rs1r)
  SDTCisVT<2, v16i8>,      // Result 2: QACC_L[127:0] (matches instruction def: QACC_L_LOW:$qacc_l_l)
  SDTCisVT<3, v16i8>,      // Result 3: QACC_L[255:128] (matches instruction def: QACC_L_HIGH:$qacc_l_h)
  SDTCisVT<4, v16i8>,      // Operand 0: QACC_L[127:0] passthru (phantom operand)
  SDTCisVT<5, v16i8>,      // Operand 1: QACC_L[255:128] passthru (phantom operand)
  SDTCisVT<6, v8i16>,      // Operand 2: qx vector (8x i16)
  SDTCisVT<7, v8i16>,      // Operand 3: qy vector (8x i16)
  SDTCisPtrTy<8>,          // Operand 4: rs1 pointer
  SDTCisVT<9, i32>         // Operand 5: offset immediate
]>;
def riscv_esp_vcmulas_s16_qacc_l_ld_ip_m : RVSDNode<"ESP_VCMULAS_S16_QACC_L_LD_IP_M", SDT_ESPVCMULASS16QACCLDIPM,
                                                     [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
// VCMULAS S8 QACC H LD XP SDNodes (register offset)
// Results: qu (vector), ptr (updated pointer), v2 (QACC_H[127:0]), v3 (QACC_H[255:128])
// Operands: v2_in (QACC_H[127:0] passthru), v3_in (QACC_H[255:128] passthru), qx (v16i8 vector), qy (v16i8 vector), rs1 (pointer), rs2 (i32 register offset)
def SDT_ESPVCMULASS8QACCHLDXPM : SDTypeProfile<4, 6, [
  SDTCisVT<0, v16i8>,      // Result 0: qu vector (matches instruction def: outs QR:$qu)
  SDTCisPtrTy<1>,          // Result 1: updated pointer (matches instruction def: GPRPIE:$rs1r)
  SDTCisVT<2, v16i8>,      // Result 2: QACC_H[127:0] (matches instruction def: QACC_H_LOW:$qacc_h_l)
  SDTCisVT<3, v16i8>,      // Result 3: QACC_H[255:128] (matches instruction def: QACC_H_HIGH:$qacc_h_h)
  SDTCisVT<4, v16i8>,      // Operand 0: QACC_H[127:0] passthru (phantom operand)
  SDTCisVT<5, v16i8>,      // Operand 1: QACC_H[255:128] passthru (phantom operand)
  SDTCisVT<6, v16i8>,      // Operand 2: qx vector (16x i8)
  SDTCisVT<7, v16i8>,      // Operand 3: qy vector (16x i8)
  SDTCisPtrTy<8>,          // Operand 4: rs1 pointer
  SDTCisVT<9, i32>         // Operand 5: rs2 register offset
]>;
def riscv_esp_vcmulas_s8_qacc_h_ld_xp_m : RVSDNode<"ESP_VCMULAS_S8_QACC_H_LD_XP_M", SDT_ESPVCMULASS8QACCHLDXPM,
                                                    [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
// VCMULAS S8 QACC L LD XP SDNodes (register offset)
// Results: qu (vector), ptr (updated pointer), v0 (QACC_L[127:0]), v1 (QACC_L[255:128])
// Operands: v0_in (QACC_L[127:0] passthru), v1_in (QACC_L[255:128] passthru), qx (v16i8 vector), qy (v16i8 vector), rs1 (pointer), rs2 (i32 register offset)
def SDT_ESPVCMULASS8QACCLDXPM : SDTypeProfile<4, 6, [
  SDTCisVT<0, v16i8>,      // Result 0: qu vector (matches instruction def: outs QR:$qu)
  SDTCisPtrTy<1>,          // Result 1: updated pointer (matches instruction def: GPRPIE:$rs1r)
  SDTCisVT<2, v16i8>,      // Result 2: QACC_L[127:0] (matches instruction def: QACC_L_LOW:$qacc_l_l)
  SDTCisVT<3, v16i8>,      // Result 3: QACC_L[255:128] (matches instruction def: QACC_L_HIGH:$qacc_l_h)
  SDTCisVT<4, v16i8>,      // Operand 0: QACC_L[127:0] passthru (phantom operand)
  SDTCisVT<5, v16i8>,      // Operand 1: QACC_L[255:128] passthru (phantom operand)
  SDTCisVT<6, v16i8>,      // Operand 2: qx vector (16x i8)
  SDTCisVT<7, v16i8>,      // Operand 3: qy vector (16x i8)
  SDTCisPtrTy<8>,          // Operand 4: rs1 pointer
  SDTCisVT<9, i32>         // Operand 5: rs2 register offset
]>;
def riscv_esp_vcmulas_s8_qacc_l_ld_xp_m : RVSDNode<"ESP_VCMULAS_S8_QACC_L_LD_XP_M", SDT_ESPVCMULASS8QACCLDXPM,
                                                    [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
// VCMULAS S16 QACC H LD XP SDNodes (register offset)
// Results: qu (vector), ptr (updated pointer), v2 (QACC_H[127:0]), v3 (QACC_H[255:128])
// Operands: v2_in (QACC_H[127:0] passthru), v3_in (QACC_H[255:128] passthru), qx (v8i16 vector), qy (v8i16 vector), rs1 (pointer), rs2 (i32 register offset)
def SDT_ESPVCMULASS16QACCHLDXPM : SDTypeProfile<4, 6, [
  SDTCisVT<0, v16i8>,      // Result 0: qu vector (matches instruction def: outs QR:$qu)
  SDTCisPtrTy<1>,          // Result 1: updated pointer (matches instruction def: GPRPIE:$rs1r)
  SDTCisVT<2, v16i8>,      // Result 2: QACC_H[127:0] (matches instruction def: QACC_H_LOW:$qacc_h_l)
  SDTCisVT<3, v16i8>,      // Result 3: QACC_H[255:128] (matches instruction def: QACC_H_HIGH:$qacc_h_h)
  SDTCisVT<4, v16i8>,      // Operand 0: QACC_H[127:0] passthru (phantom operand)
  SDTCisVT<5, v16i8>,      // Operand 1: QACC_H[255:128] passthru (phantom operand)
  SDTCisVT<6, v8i16>,      // Operand 2: qx vector (8x i16)
  SDTCisVT<7, v8i16>,      // Operand 3: qy vector (8x i16)
  SDTCisPtrTy<8>,          // Operand 4: rs1 pointer
  SDTCisVT<9, i32>         // Operand 5: rs2 register offset
]>;
def riscv_esp_vcmulas_s16_qacc_h_ld_xp_m : RVSDNode<"ESP_VCMULAS_S16_QACC_H_LD_XP_M", SDT_ESPVCMULASS16QACCHLDXPM,
                                                     [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
// VCMULAS S16 QACC L LD XP SDNodes (register offset)
// Results: qu (vector), ptr (updated pointer), v0 (QACC_L[127:0]), v1 (QACC_L[255:128])
// Operands: v0_in (QACC_L[127:0] passthru), v1_in (QACC_L[255:128] passthru), qx (v8i16 vector), qy (v8i16 vector), rs1 (pointer), rs2 (i32 register offset)
def SDT_ESPVCMULASS16QACCLDXPM : SDTypeProfile<4, 6, [
  SDTCisVT<0, v16i8>,      // Result 0: qu vector (matches instruction def: outs QR:$qu)
  SDTCisPtrTy<1>,          // Result 1: updated pointer (matches instruction def: GPRPIE:$rs1r)
  SDTCisVT<2, v16i8>,      // Result 2: QACC_L[127:0] (matches instruction def: QACC_L_LOW:$qacc_l_l)
  SDTCisVT<3, v16i8>,      // Result 3: QACC_L[255:128] (matches instruction def: QACC_L_HIGH:$qacc_l_h)
  SDTCisVT<4, v16i8>,      // Operand 0: QACC_L[127:0] passthru (phantom operand)
  SDTCisVT<5, v16i8>,      // Operand 1: QACC_L[255:128] passthru (phantom operand)
  SDTCisVT<6, v8i16>,      // Operand 2: qx vector (8x i16)
  SDTCisVT<7, v8i16>,      // Operand 3: qy vector (8x i16)
  SDTCisPtrTy<8>,          // Operand 4: rs1 pointer
  SDTCisVT<9, i32>         // Operand 5: rs2 register offset
]>;
def riscv_esp_vcmulas_s16_qacc_l_ld_xp_m : RVSDNode<"ESP_VCMULAS_S16_QACC_L_LD_XP_M", SDT_ESPVCMULASS16QACCLDXPM,
                                                     [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;

// LD/ST XACC IP SDNodes
// Mixed model: XACC as {i32 low, i32 high}
def SDT_ESPLDXACCIPM : SDTypeProfile<3, 4, [
  SDTCisPtrTy<0>,          // Result 0: updated pointer
  SDTCisVT<1, i32>,        // Result 1: new XACC low (i32)
  SDTCisVT<2, i32>,        // Result 2: new XACC high (i32, only low 8 bits valid)
  SDTCisVT<3, i32>,        // Operand 0: XACC low passthru (i32)
  SDTCisVT<4, i32>,        // Operand 1: XACC high passthru (i32, only low 8 bits valid)
  SDTCisPtrTy<5>,          // Operand 2: rs1 pointer
  SDTCisVT<6, i32>         // Operand 3: offset immediate
]>;
def SDT_ESPSTXACCIPM : SDTypeProfile<3, 4, [
  SDTCisPtrTy<0>,          // Result 0: updated pointer
  SDTCisVT<1, i32>,        // Result 1: XACC low unchanged (i32, equals input)
  SDTCisVT<2, i32>,        // Result 2: XACC high unchanged (i32, equals input, only low 8 bits valid)
  SDTCisVT<3, i32>,        // Operand 0: XACC low passthru (i32)
  SDTCisVT<4, i32>,        // Operand 1: XACC high passthru (i32, only low 8 bits valid)
  SDTCisPtrTy<5>,          // Operand 2: rs1 pointer
  SDTCisVT<6, i32>         // Operand 3: offset immediate
]>;

def riscv_esp_ld_xacc_ip_m : RVSDNode<"ESP_LD_XACC_IP_M", SDT_ESPLDXACCIPM,
                                      [SDNPHasChain, SDNPMayLoad, SDNPMemOperand,
                                       SDNPOptInGlue, SDNPOutGlue]>;
def riscv_esp_st_s_xacc_ip_m : RVSDNode<"ESP_ST_S_XACC_IP_M", SDT_ESPSTXACCIPM,
                                         [SDNPHasChain, SDNPMayStore, SDNPMemOperand,
                                          SDNPOptInGlue, SDNPOutGlue]>;
def riscv_esp_st_u_xacc_ip_m : RVSDNode<"ESP_ST_U_XACC_IP_M", SDT_ESPSTXACCIPM,
                                         [SDNPHasChain, SDNPMayStore, SDNPMemOperand,
                                          SDNPOptInGlue, SDNPOutGlue]>;

// LDQA SDNodes (Load QACC from memory)
// Results: 4x128-bit QACC + updated pointer
// Operands: rs1 (pointer), offset (i32 immediate or register)
// Note: These instructions load 128-bit data and store to QACC_H and QACC_L
def SDT_ESPLDQAS16128IPM : SDTypeProfile<5, 2, [
  SDTCisVT<0, v16i8>,      // Result 0: QACC_L[127:0] (128-bit)
  SDTCisVT<1, v16i8>,      // Result 1: QACC_L[255:128] (128-bit)
  SDTCisVT<2, v16i8>,      // Result 2: QACC_H[127:0] (128-bit)
  SDTCisVT<3, v16i8>,      // Result 3: QACC_H[255:128] (128-bit)
  SDTCisPtrTy<4>,          // Result 4: updated pointer
  SDTCisPtrTy<5>,          // Operand 0: rs1 pointer
  SDTCisVT<6, i32>         // Operand 1: offset immediate
]>;
def riscv_esp_ldqa_s16_128_ip_m : RVSDNode<"ESP_LDQA_S16_128_IP_M", SDT_ESPLDQAS16128IPM,
                                           [SDNPHasChain, SDNPMayLoad, SDNPMemOperand,
                                            SDNPOptInGlue, SDNPOutGlue]>;
def SDT_ESPLDQAS16128XPM : SDTypeProfile<5, 2, [
  SDTCisVT<0, v16i8>,      // Result 0: QACC_L[127:0] (128-bit)
  SDTCisVT<1, v16i8>,      // Result 1: QACC_L[255:128] (128-bit)
  SDTCisVT<2, v16i8>,      // Result 2: QACC_H[127:0] (128-bit)
  SDTCisVT<3, v16i8>,      // Result 3: QACC_H[255:128] (128-bit)
  SDTCisPtrTy<4>,          // Result 4: updated pointer
  SDTCisPtrTy<5>,          // Operand 0: rs1 pointer
  SDTCisVT<6, i32>         // Operand 1: rs2 register offset
]>;
def SDT_ESPLDQAS8128IPM : SDTypeProfile<5, 2, [
  SDTCisVT<0, v16i8>,      // Result 0: QACC_L[127:0] (128-bit)
  SDTCisVT<1, v16i8>,      // Result 1: QACC_L[255:128] (128-bit)
  SDTCisVT<2, v16i8>,      // Result 2: QACC_H[127:0] (128-bit)
  SDTCisVT<3, v16i8>,      // Result 3: QACC_H[255:128] (128-bit)
  SDTCisPtrTy<4>,          // Result 4: updated pointer
  SDTCisPtrTy<5>,          // Operand 0: rs1 pointer
  SDTCisVT<6, i32>         // Operand 1: offset immediate
]>;
def SDT_ESPLDQAS8128XPM : SDTypeProfile<5, 2, [
  SDTCisVT<0, v16i8>,      // Result 0: QACC_L[127:0] (128-bit)
  SDTCisVT<1, v16i8>,      // Result 1: QACC_L[255:128] (128-bit)
  SDTCisVT<2, v16i8>,      // Result 2: QACC_H[127:0] (128-bit)
  SDTCisVT<3, v16i8>,      // Result 3: QACC_H[255:128] (128-bit)
  SDTCisPtrTy<4>,          // Result 4: updated pointer
  SDTCisPtrTy<5>,          // Operand 0: rs1 pointer
  SDTCisVT<6, i32>         // Operand 1: rs2 register offset
]>;
def SDT_ESPLDQAU16128IPM : SDTypeProfile<5, 2, [
  SDTCisVT<0, v16i8>,      // Result 0: QACC_L[127:0] (128-bit)
  SDTCisVT<1, v16i8>,      // Result 1: QACC_L[255:128] (128-bit)
  SDTCisVT<2, v16i8>,      // Result 2: QACC_H[127:0] (128-bit)
  SDTCisVT<3, v16i8>,      // Result 3: QACC_H[255:128] (128-bit)
  SDTCisPtrTy<4>,          // Result 4: updated pointer
  SDTCisPtrTy<5>,          // Operand 0: rs1 pointer
  SDTCisVT<6, i32>         // Operand 1: offset immediate
]>;
def SDT_ESPLDQAU16128XPM : SDTypeProfile<5, 2, [
  SDTCisVT<0, v16i8>,      // Result 0: QACC_L[127:0] (128-bit)
  SDTCisVT<1, v16i8>,      // Result 1: QACC_L[255:128] (128-bit)
  SDTCisVT<2, v16i8>,      // Result 2: QACC_H[127:0] (128-bit)
  SDTCisVT<3, v16i8>,      // Result 3: QACC_H[255:128] (128-bit)
  SDTCisPtrTy<4>,          // Result 4: updated pointer
  SDTCisPtrTy<5>,          // Operand 0: rs1 pointer
  SDTCisVT<6, i32>         // Operand 1: rs2 register offset
]>;
def SDT_ESPLDQAU8128IPM : SDTypeProfile<5, 2, [
  SDTCisVT<0, v16i8>,      // Result 0: QACC_L[127:0] (128-bit)
  SDTCisVT<1, v16i8>,      // Result 1: QACC_L[255:128] (128-bit)
  SDTCisVT<2, v16i8>,      // Result 2: QACC_H[127:0] (128-bit)
  SDTCisVT<3, v16i8>,      // Result 3: QACC_H[255:128] (128-bit)
  SDTCisPtrTy<4>,          // Result 4: updated pointer
  SDTCisPtrTy<5>,          // Operand 0: rs1 pointer
  SDTCisVT<6, i32>         // Operand 1: offset immediate
]>;
def SDT_ESPLDQAU8128XPM : SDTypeProfile<5, 2, [
  SDTCisVT<0, v16i8>,      // Result 0: QACC_L[127:0] (128-bit)
  SDTCisVT<1, v16i8>,      // Result 1: QACC_L[255:128] (128-bit)
  SDTCisVT<2, v16i8>,      // Result 2: QACC_H[127:0] (128-bit)
  SDTCisVT<3, v16i8>,      // Result 3: QACC_H[255:128] (128-bit)
  SDTCisPtrTy<4>,          // Result 4: updated pointer
  SDTCisPtrTy<5>,          // Operand 0: rs1 pointer
  SDTCisVT<6, i32>         // Operand 1: rs2 register offset
]>;
def riscv_esp_ldqa_s16_128_xp_m : RVSDNode<"ESP_LDQA_S16_128_XP_M", SDT_ESPLDQAS16128XPM,
                                           [SDNPHasChain, SDNPMayLoad, SDNPMemOperand,
                                            SDNPOptInGlue, SDNPOutGlue]>;
def riscv_esp_ldqa_s8_128_ip_m : RVSDNode<"ESP_LDQA_S8_128_IP_M", SDT_ESPLDQAS8128IPM,
                                          [SDNPHasChain, SDNPMayLoad, SDNPMemOperand,
                                           SDNPOptInGlue, SDNPOutGlue]>;
def riscv_esp_ldqa_s8_128_xp_m : RVSDNode<"ESP_LDQA_S8_128_XP_M", SDT_ESPLDQAS8128XPM,
                                          [SDNPHasChain, SDNPMayLoad, SDNPMemOperand,
                                           SDNPOptInGlue, SDNPOutGlue]>;
def riscv_esp_ldqa_u16_128_ip_m : RVSDNode<"ESP_LDQA_U16_128_IP_M", SDT_ESPLDQAU16128IPM,
                                           [SDNPHasChain, SDNPMayLoad, SDNPMemOperand,
                                            SDNPOptInGlue, SDNPOutGlue]>;
def riscv_esp_ldqa_u16_128_xp_m : RVSDNode<"ESP_LDQA_U16_128_XP_M", SDT_ESPLDQAU16128XPM,
                                           [SDNPHasChain, SDNPMayLoad, SDNPMemOperand,
                                            SDNPOptInGlue, SDNPOutGlue]>;
def riscv_esp_ldqa_u8_128_ip_m : RVSDNode<"ESP_LDQA_U8_128_IP_M", SDT_ESPLDQAU8128IPM,
                                          [SDNPHasChain, SDNPMayLoad, SDNPMemOperand,
                                           SDNPOptInGlue, SDNPOutGlue]>;
def riscv_esp_ldqa_u8_128_xp_m : RVSDNode<"ESP_LDQA_U8_128_XP_M", SDT_ESPLDQAU8128XPM,
                                          [SDNPHasChain, SDNPMayLoad, SDNPMemOperand,
                                           SDNPOptInGlue, SDNPOutGlue]>;

// LD/ST UA_STATE SDNodes
// Results: ua_state (v16i8), rs1r (pointer)
// Operands: rs1 (pointer), offset (i32 immediate), ua_state_passthru (v16i8)
def SDT_ESPLDUASTATEIPM : SDTypeProfile<2, 3, [
  SDTCisVT<0, v16i8>,      // Result 0: new UA_STATE (128-bit)
  SDTCisPtrTy<1>,          // Result 1: updated pointer
  SDTCisPtrTy<2>,          // Operand 0: rs1 pointer
  SDTCisVT<3, i32>,        // Operand 1: offset immediate
  SDTCisVT<4, v16i8>      // Operand 2: UA_STATE passthru (phantom operand for data flow)
]>;
def SDT_ESPSTUASTATEIPM : SDTypeProfile<2, 3, [
  SDTCisVT<0, v16i8>,      // Result 0: new UA_STATE (explicit state passing)
  SDTCisPtrTy<1>,          // Result 1: updated pointer
  SDTCisVT<2, v16i8>,      // Operand 0: UA_STATE passthru (phantom operand for data flow)
  SDTCisPtrTy<3>,          // Operand 1: rs1 pointer
  SDTCisVT<4, i32>         // Operand 2: offset immediate
]>;

def riscv_esp_ld_ua_state_ip_m : RVSDNode<"ESP_LD_UA_STATE_IP_M", SDT_ESPLDUASTATEIPM,
                                           [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def riscv_esp_st_ua_state_ip_m : RVSDNode<"ESP_ST_UA_STATE_IP_M", SDT_ESPSTUASTATEIPM,
                                           [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;

def SDT_ESPSRCMBS16QACCM : SDTypeProfile<1, 6, [
  SDTCisVT<0, v8i16>,      // Result 0: qu vector
  SDTCisVT<1, v16i8>,      // Operand 0: QACC_L[127:0] (phantom operand)
  SDTCisVT<2, v16i8>,      // Operand 1: QACC_L[255:128] (phantom operand)
  SDTCisVT<3, v16i8>,      // Operand 2: QACC_H[127:0] (phantom operand)
  SDTCisVT<4, v16i8>,      // Operand 3: QACC_H[255:128] (phantom operand)
  SDTCisVT<5, i32>,        // Operand 4: rs1 (shift amount)
  SDTCisVT<6, i32>         // Operand 5: sel2 immediate
]>;
def riscv_esp_srcmb_s16_qacc_m : RVSDNode<"ESP_SRCMB_S16_QACC_M", SDT_ESPSRCMBS16QACCM,
                                             [SDNPSideEffect]>;

def SDT_ESPSRCMBS8QACCM : SDTypeProfile<1, 6, [
  SDTCisVT<0, v16i8>,      // Result 0: qu vector
  SDTCisVT<1, v16i8>,      // Operand 0: QACC_L[127:0] (phantom operand)
  SDTCisVT<2, v16i8>,      // Operand 1: QACC_L[255:128] (phantom operand)
  SDTCisVT<3, v16i8>,      // Operand 2: QACC_H[127:0] (phantom operand)
  SDTCisVT<4, v16i8>,      // Operand 3: QACC_H[255:128] (phantom operand)
  SDTCisVT<5, i32>,        // Operand 4: rs1 (shift amount)
  SDTCisVT<6, i32>         // Operand 5: sel2 immediate
]>;
def riscv_esp_srcmb_s8_qacc_m : RVSDNode<"ESP_SRCMB_S8_QACC_M", SDT_ESPSRCMBS8QACCM>;

def SDT_ESPSRCMBU16QACCM : SDTypeProfile<1, 6, [
  SDTCisVT<0, v8i16>,      // Result 0: qu vector
  SDTCisVT<1, v16i8>,      // Operand 0: QACC_L[127:0] (phantom operand)
  SDTCisVT<2, v16i8>,      // Operand 1: QACC_L[255:128] (phantom operand)
  SDTCisVT<3, v16i8>,      // Operand 2: QACC_H[127:0] (phantom operand)
  SDTCisVT<4, v16i8>,      // Operand 3: QACC_H[255:128] (phantom operand)
  SDTCisVT<5, i32>,        // Operand 4: rs1 (shift amount)
  SDTCisVT<6, i32>         // Operand 5: sel2 immediate
]>;
def riscv_esp_srcmb_u16_qacc_m : RVSDNode<"ESP_SRCMB_U16_QACC_M", SDT_ESPSRCMBU16QACCM,
                                             [SDNPSideEffect]>;

def SDT_ESPSRCMBU8QACCM : SDTypeProfile<1, 6, [
  SDTCisVT<0, v16i8>,      // Result 0: qu vector
  SDTCisVT<1, v16i8>,      // Operand 0: QACC_L[127:0] (phantom operand)
  SDTCisVT<2, v16i8>,      // Operand 1: QACC_L[255:128] (phantom operand)
  SDTCisVT<3, v16i8>,      // Operand 2: QACC_H[127:0] (phantom operand)
  SDTCisVT<4, v16i8>,      // Operand 3: QACC_H[255:128] (phantom operand)
  SDTCisVT<5, i32>,        // Operand 4: rs1 (shift amount)
  SDTCisVT<6, i32>         // Operand 5: sel2 immediate
]>;
def riscv_esp_srcmb_u8_qacc_m : RVSDNode<"ESP_SRCMB_U8_QACC_M", SDT_ESPSRCMBU8QACCM,
                                             [SDNPSideEffect]>;

// VMULAS QACC SDNodes - Vector Multiply-Accumulate to QACC with explicit state passing
// Results: 4x128-bit QACC (v16i8, v16i8, v16i8, v16i8)
// Operands: 4x128-bit QACC passthru (v16i8, v16i8, v16i8, v16i8) + qx + qy
def SDT_ESPVMULASS16QACCM : SDTypeProfile<4, 6, [
  SDTCisVT<0, v16i8>,        // Result 0: QACC_L[127:0] (128-bit)
  SDTCisVT<1, v16i8>,        // Result 1: QACC_L[255:128] (128-bit)
  SDTCisVT<2, v16i8>,        // Result 2: QACC_H[127:0] (128-bit)
  SDTCisVT<3, v16i8>,        // Result 3: QACC_H[255:128] (128-bit)
  SDTCisVT<4, v16i8>,        // Operand 0: QACC_L[127:0] passthru (phantom operand)
  SDTCisVT<5, v16i8>,        // Operand 1: QACC_L[255:128] passthru (phantom operand)
  SDTCisVT<6, v16i8>,        // Operand 2: QACC_H[127:0] passthru (phantom operand)
  SDTCisVT<7, v16i8>,        // Operand 3: QACC_H[255:128] passthru (phantom operand)
  SDTCisVT<8, v8i16>,        // Operand 4: qx vector (8x i16)
  SDTCisVT<9, v8i16>         // Operand 5: qy vector (8x i16)
]>;
def SDT_ESPVMULASS8QACCM : SDTypeProfile<4, 6, [
  SDTCisVT<0, v16i8>,        // Result 0: QACC_L[127:0] (128-bit)
  SDTCisVT<1, v16i8>,        // Result 1: QACC_L[255:128] (128-bit)
  SDTCisVT<2, v16i8>,        // Result 2: QACC_H[127:0] (128-bit)
  SDTCisVT<3, v16i8>,        // Result 3: QACC_H[255:128] (128-bit)
  SDTCisVT<4, v16i8>,        // Operand 0: QACC_L[127:0] passthru (phantom operand)
  SDTCisVT<5, v16i8>,        // Operand 1: QACC_L[255:128] passthru (phantom operand)
  SDTCisVT<6, v16i8>,        // Operand 2: QACC_H[127:0] passthru (phantom operand)
  SDTCisVT<7, v16i8>,        // Operand 3: QACC_H[255:128] passthru (phantom operand)
  SDTCisVT<8, v16i8>,        // Operand 4: qx vector
  SDTCisVT<9, v16i8>         // Operand 5: qy vector
]>;
def SDT_ESPVMULASU16QACCM : SDTypeProfile<4, 6, [
  SDTCisVT<0, v16i8>,        // Result 0: QACC_L[127:0] (128-bit)
  SDTCisVT<1, v16i8>,        // Result 1: QACC_L[255:128] (128-bit)
  SDTCisVT<2, v16i8>,        // Result 2: QACC_H[127:0] (128-bit)
  SDTCisVT<3, v16i8>,        // Result 3: QACC_H[255:128] (128-bit)
  SDTCisVT<4, v16i8>,        // Operand 0: QACC_L[127:0] passthru (phantom operand)
  SDTCisVT<5, v16i8>,        // Operand 1: QACC_L[255:128] passthru (phantom operand)
  SDTCisVT<6, v16i8>,        // Operand 2: QACC_H[127:0] passthru (phantom operand)
  SDTCisVT<7, v16i8>,        // Operand 3: QACC_H[255:128] passthru (phantom operand)
  SDTCisVT<8, v8i16>,        // Operand 4: qx vector (8x i16)
  SDTCisVT<9, v8i16>         // Operand 5: qy vector (8x i16)
]>;
def SDT_ESPVMULASU8QACCM : SDTypeProfile<4, 6, [
  SDTCisVT<0, v16i8>,        // Result 0: QACC_L[127:0] (128-bit)
  SDTCisVT<1, v16i8>,        // Result 1: QACC_L[255:128] (128-bit)
  SDTCisVT<2, v16i8>,        // Result 2: QACC_H[127:0] (128-bit)
  SDTCisVT<3, v16i8>,        // Result 3: QACC_H[255:128] (128-bit)
  SDTCisVT<4, v16i8>,        // Operand 0: QACC_L[127:0] passthru (phantom operand)
  SDTCisVT<5, v16i8>,        // Operand 1: QACC_L[255:128] passthru (phantom operand)
  SDTCisVT<6, v16i8>,        // Operand 2: QACC_H[127:0] passthru (phantom operand)
  SDTCisVT<7, v16i8>,        // Operand 3: QACC_H[255:128] passthru (phantom operand)
  SDTCisVT<8, v16i8>,        // Operand 4: qx vector
  SDTCisVT<9, v16i8>         // Operand 5: qy vector
]>;

def riscv_esp_vmulas_s16_qacc_m : RVSDNode<"ESP_VMULAS_S16_QACC_M", SDT_ESPVMULASS16QACCM>;
def riscv_esp_vmulas_s8_qacc_m : RVSDNode<"ESP_VMULAS_S8_QACC_M", SDT_ESPVMULASS8QACCM>;
def riscv_esp_vmulas_u16_qacc_m : RVSDNode<"ESP_VMULAS_U16_QACC_M", SDT_ESPVMULASU16QACCM>;
def riscv_esp_vmulas_u8_qacc_m : RVSDNode<"ESP_VMULAS_U8_QACC_M", SDT_ESPVMULASU8QACCM>;

// VMULAS QACC LD IP SDNodes
def SDT_ESPVMULASS16QACCLDIPM : SDTypeProfile<6, 8, [
  SDTCisVT<0, v16i8>,      // Result 0: qu vector
  SDTCisPtrTy<1>,          // Result 1: updated pointer
  SDTCisVT<2, v16i8>,      // Result 2: QACC_L[127:0]
  SDTCisVT<3, v16i8>,      // Result 3: QACC_L[255:128]
  SDTCisVT<4, v16i8>,      // Result 4: QACC_H[127:0]
  SDTCisVT<5, v16i8>,      // Result 5: QACC_H[255:128]
  SDTCisVT<6, v16i8>,      // Operand 0: QACC_L[127:0] passthru
  SDTCisVT<7, v16i8>,      // Operand 1: QACC_L[255:128] passthru
  SDTCisVT<8, v16i8>,      // Operand 2: QACC_H[127:0] passthru
  SDTCisVT<9, v16i8>,      // Operand 3: QACC_H[255:128] passthru
  SDTCisVT<10, v8i16>,     // Operand 4: qx vector
  SDTCisVT<11, v8i16>,     // Operand 5: qy vector
  SDTCisPtrTy<12>,         // Operand 6: rs1 pointer
  SDTCisVT<13, i32>        // Operand 7: offset immediate
]>;
def SDT_ESPVMULASS8QACCLDIPM : SDTypeProfile<6, 8, [
  SDTCisVT<0, v16i8>,      // Result 0: qu vector
  SDTCisPtrTy<1>,          // Result 1: updated pointer
  SDTCisVT<2, v16i8>,      // Result 2: QACC_L[127:0]
  SDTCisVT<3, v16i8>,      // Result 3: QACC_L[255:128]
  SDTCisVT<4, v16i8>,      // Result 4: QACC_H[127:0]
  SDTCisVT<5, v16i8>,      // Result 5: QACC_H[255:128]
  SDTCisVT<6, v16i8>,      // Operand 0: QACC_L[127:0] passthru
  SDTCisVT<7, v16i8>,      // Operand 1: QACC_L[255:128] passthru
  SDTCisVT<8, v16i8>,      // Operand 2: QACC_H[127:0] passthru
  SDTCisVT<9, v16i8>,      // Operand 3: QACC_H[255:128] passthru
  SDTCisVT<10, v16i8>,     // Operand 4: qx vector
  SDTCisVT<11, v16i8>,     // Operand 5: qy vector
  SDTCisPtrTy<12>,         // Operand 6: rs1 pointer
  SDTCisVT<13, i32>        // Operand 7: offset immediate
]>;
def SDT_ESPVMULASU16QACCLDIPM : SDTypeProfile<6, 8, [
  SDTCisVT<0, v16i8>,      // Result 0: qu vector
  SDTCisPtrTy<1>,          // Result 1: updated pointer
  SDTCisVT<2, v16i8>,      // Result 2: QACC_L[127:0]
  SDTCisVT<3, v16i8>,      // Result 3: QACC_L[255:128]
  SDTCisVT<4, v16i8>,      // Result 4: QACC_H[127:0]
  SDTCisVT<5, v16i8>,      // Result 5: QACC_H[255:128]
  SDTCisVT<6, v16i8>,      // Operand 0: QACC_L[127:0] passthru
  SDTCisVT<7, v16i8>,      // Operand 1: QACC_L[255:128] passthru
  SDTCisVT<8, v16i8>,      // Operand 2: QACC_H[127:0] passthru
  SDTCisVT<9, v16i8>,      // Operand 3: QACC_H[255:128] passthru
  SDTCisVT<10, v8i16>,     // Operand 4: qx vector
  SDTCisVT<11, v8i16>,     // Operand 5: qy vector
  SDTCisPtrTy<12>,         // Operand 6: rs1 pointer
  SDTCisVT<13, i32>        // Operand 7: offset immediate
]>;
def SDT_ESPVMULASU8QACCLDIPM : SDTypeProfile<6, 8, [
  SDTCisVT<0, v16i8>,      // Result 0: qu vector
  SDTCisPtrTy<1>,          // Result 1: updated pointer
  SDTCisVT<2, v16i8>,      // Result 2: QACC_L[127:0]
  SDTCisVT<3, v16i8>,      // Result 3: QACC_L[255:128]
  SDTCisVT<4, v16i8>,      // Result 4: QACC_H[127:0]
  SDTCisVT<5, v16i8>,      // Result 5: QACC_H[255:128]
  SDTCisVT<6, v16i8>,      // Operand 0: QACC_L[127:0] passthru
  SDTCisVT<7, v16i8>,      // Operand 1: QACC_L[255:128] passthru
  SDTCisVT<8, v16i8>,      // Operand 2: QACC_H[127:0] passthru
  SDTCisVT<9, v16i8>,      // Operand 3: QACC_H[255:128] passthru
  SDTCisVT<10, v16i8>,     // Operand 4: qx vector
  SDTCisVT<11, v16i8>,     // Operand 5: qy vector
  SDTCisPtrTy<12>,         // Operand 6: rs1 pointer
  SDTCisVT<13, i32>        // Operand 7: offset immediate
]>;

def riscv_esp_vmulas_s16_qacc_ld_ip_m : RVSDNode<"ESP_VMULAS_S16_QACC_LD_IP_M", SDT_ESPVMULASS16QACCLDIPM,
                                                  [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def riscv_esp_vmulas_s8_qacc_ld_ip_m : RVSDNode<"ESP_VMULAS_S8_QACC_LD_IP_M", SDT_ESPVMULASS8QACCLDIPM,
                                                 [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def riscv_esp_vmulas_u16_qacc_ld_ip_m : RVSDNode<"ESP_VMULAS_U16_QACC_LD_IP_M", SDT_ESPVMULASU16QACCLDIPM,
                                                  [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def riscv_esp_vmulas_u8_qacc_ld_ip_m : RVSDNode<"ESP_VMULAS_U8_QACC_LD_IP_M", SDT_ESPVMULASU8QACCLDIPM,
                                                 [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;

// VMULAS QACC LD XP SDNodes (register offset)
def SDT_ESPVMULASS16QACCLDXPM : SDTypeProfile<6, 8, [
  SDTCisVT<0, v16i8>,      // Result 0: qu vector
  SDTCisPtrTy<1>,          // Result 1: updated pointer
  SDTCisVT<2, v16i8>,      // Result 2: QACC_L[127:0]
  SDTCisVT<3, v16i8>,      // Result 3: QACC_L[255:128]
  SDTCisVT<4, v16i8>,      // Result 4: QACC_H[127:0]
  SDTCisVT<5, v16i8>,      // Result 5: QACC_H[255:128]
  SDTCisVT<6, v16i8>,      // Operand 0: QACC_L[127:0] passthru
  SDTCisVT<7, v16i8>,      // Operand 1: QACC_L[255:128] passthru
  SDTCisVT<8, v16i8>,      // Operand 2: QACC_H[127:0] passthru
  SDTCisVT<9, v16i8>,      // Operand 3: QACC_H[255:128] passthru
  SDTCisVT<10, v8i16>,     // Operand 4: qx vector
  SDTCisVT<11, v8i16>,     // Operand 5: qy vector
  SDTCisPtrTy<12>,         // Operand 6: rs1 pointer
  SDTCisVT<13, i32>        // Operand 7: rs2 register offset
]>;
def SDT_ESPVMULASS8QACCLDXPM : SDTypeProfile<6, 8, [
  SDTCisVT<0, v16i8>,      // Result 0: qu vector
  SDTCisPtrTy<1>,          // Result 1: updated pointer
  SDTCisVT<2, v16i8>,      // Result 2: QACC_L[127:0]
  SDTCisVT<3, v16i8>,      // Result 3: QACC_L[255:128]
  SDTCisVT<4, v16i8>,      // Result 4: QACC_H[127:0]
  SDTCisVT<5, v16i8>,      // Result 5: QACC_H[255:128]
  SDTCisVT<6, v16i8>,      // Operand 0: QACC_L[127:0] passthru
  SDTCisVT<7, v16i8>,      // Operand 1: QACC_L[255:128] passthru
  SDTCisVT<8, v16i8>,      // Operand 2: QACC_H[127:0] passthru
  SDTCisVT<9, v16i8>,      // Operand 3: QACC_H[255:128] passthru
  SDTCisVT<10, v16i8>,     // Operand 4: qx vector
  SDTCisVT<11, v16i8>,     // Operand 5: qy vector
  SDTCisPtrTy<12>,         // Operand 6: rs1 pointer
  SDTCisVT<13, i32>        // Operand 7: rs2 register offset
]>;
def SDT_ESPVMULASU16QACCLDXPM : SDTypeProfile<6, 8, [
  SDTCisVT<0, v16i8>,      // Result 0: qu vector
  SDTCisPtrTy<1>,          // Result 1: updated pointer
  SDTCisVT<2, v16i8>,      // Result 2: QACC_L[127:0]
  SDTCisVT<3, v16i8>,      // Result 3: QACC_L[255:128]
  SDTCisVT<4, v16i8>,      // Result 4: QACC_H[127:0]
  SDTCisVT<5, v16i8>,      // Result 5: QACC_H[255:128]
  SDTCisVT<6, v16i8>,      // Operand 0: QACC_L[127:0] passthru
  SDTCisVT<7, v16i8>,      // Operand 1: QACC_L[255:128] passthru
  SDTCisVT<8, v16i8>,      // Operand 2: QACC_H[127:0] passthru
  SDTCisVT<9, v16i8>,      // Operand 3: QACC_H[255:128] passthru
  SDTCisVT<10, v8i16>,     // Operand 4: qx vector
  SDTCisVT<11, v8i16>,     // Operand 5: qy vector
  SDTCisPtrTy<12>,         // Operand 6: rs1 pointer
  SDTCisVT<13, i32>        // Operand 7: rs2 register offset
]>;
def SDT_ESPVMULASU8QACCLDXPM : SDTypeProfile<6, 8, [
  SDTCisVT<0, v16i8>,      // Result 0: qu vector
  SDTCisPtrTy<1>,          // Result 1: updated pointer
  SDTCisVT<2, v16i8>,      // Result 2: QACC_L[127:0]
  SDTCisVT<3, v16i8>,      // Result 3: QACC_L[255:128]
  SDTCisVT<4, v16i8>,      // Result 4: QACC_H[127:0]
  SDTCisVT<5, v16i8>,      // Result 5: QACC_H[255:128]
  SDTCisVT<6, v16i8>,      // Operand 0: QACC_L[127:0] passthru
  SDTCisVT<7, v16i8>,      // Operand 1: QACC_L[255:128] passthru
  SDTCisVT<8, v16i8>,      // Operand 2: QACC_H[127:0] passthru
  SDTCisVT<9, v16i8>,      // Operand 3: QACC_H[255:128] passthru
  SDTCisVT<10, v16i8>,     // Operand 4: qx vector
  SDTCisVT<11, v16i8>,     // Operand 5: qy vector
  SDTCisPtrTy<12>,         // Operand 6: rs1 pointer
  SDTCisVT<13, i32>        // Operand 7: rs2 register offset
]>;

def riscv_esp_vmulas_s16_qacc_ld_xp_m : RVSDNode<"ESP_VMULAS_S16_QACC_LD_XP_M", SDT_ESPVMULASS16QACCLDXPM,
                                                  [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def riscv_esp_vmulas_s8_qacc_ld_xp_m : RVSDNode<"ESP_VMULAS_S8_QACC_LD_XP_M", SDT_ESPVMULASS8QACCLDXPM,
                                                 [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def riscv_esp_vmulas_u16_qacc_ld_xp_m : RVSDNode<"ESP_VMULAS_U16_QACC_LD_XP_M", SDT_ESPVMULASU16QACCLDXPM,
                                                  [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def riscv_esp_vmulas_u8_qacc_ld_xp_m : RVSDNode<"ESP_VMULAS_U8_QACC_LD_XP_M", SDT_ESPVMULASU8QACCLDXPM,
                                                 [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;

// VMULAS QACC ST IP SDNodes
def SDT_ESPVMULASS16QACCSTIPM : SDTypeProfile<5, 9, [
  SDTCisPtrTy<0>,          // Result 0: updated pointer
  SDTCisVT<1, v16i8>,      // Result 1: QACC_L[127:0]
  SDTCisVT<2, v16i8>,      // Result 2: QACC_L[255:128]
  SDTCisVT<3, v16i8>,      // Result 3: QACC_H[127:0]
  SDTCisVT<4, v16i8>,      // Result 4: QACC_H[255:128]
  SDTCisVT<5, v16i8>,      // Operand 0: QACC_L[127:0] passthru
  SDTCisVT<6, v16i8>,      // Operand 1: QACC_L[255:128] passthru
  SDTCisVT<7, v16i8>,      // Operand 2: QACC_H[127:0] passthru
  SDTCisVT<8, v16i8>,      // Operand 3: QACC_H[255:128] passthru
  SDTCisVT<9, v16i8>,      // Operand 4: qu vector
  SDTCisVT<10, v8i16>,     // Operand 5: qx vector
  SDTCisVT<11, v8i16>,     // Operand 6: qy vector
  SDTCisPtrTy<12>,         // Operand 7: rs1 pointer
  SDTCisVT<13, i32>        // Operand 8: offset immediate
]>;
def SDT_ESPVMULASS8QACCSTIPM : SDTypeProfile<5, 9, [
  SDTCisPtrTy<0>,          // Result 0: updated pointer
  SDTCisVT<1, v16i8>,      // Result 1: QACC_L[127:0]
  SDTCisVT<2, v16i8>,      // Result 2: QACC_L[255:128]
  SDTCisVT<3, v16i8>,      // Result 3: QACC_H[127:0]
  SDTCisVT<4, v16i8>,      // Result 4: QACC_H[255:128]
  SDTCisVT<5, v16i8>,      // Operand 0: QACC_L[127:0] passthru
  SDTCisVT<6, v16i8>,      // Operand 1: QACC_L[255:128] passthru
  SDTCisVT<7, v16i8>,      // Operand 2: QACC_H[127:0] passthru
  SDTCisVT<8, v16i8>,      // Operand 3: QACC_H[255:128] passthru
  SDTCisVT<9, v16i8>,      // Operand 4: qu vector
  SDTCisVT<10, v16i8>,     // Operand 5: qx vector
  SDTCisVT<11, v16i8>,     // Operand 6: qy vector
  SDTCisPtrTy<12>,         // Operand 7: rs1 pointer
  SDTCisVT<13, i32>        // Operand 8: offset immediate
]>;
def SDT_ESPVMULASU16QACCSTIPM : SDTypeProfile<5, 9, [
  SDTCisPtrTy<0>,          // Result 0: updated pointer
  SDTCisVT<1, v16i8>,      // Result 1: QACC_L[127:0]
  SDTCisVT<2, v16i8>,      // Result 2: QACC_L[255:128]
  SDTCisVT<3, v16i8>,      // Result 3: QACC_H[127:0]
  SDTCisVT<4, v16i8>,      // Result 4: QACC_H[255:128]
  SDTCisVT<5, v16i8>,      // Operand 0: QACC_L[127:0] passthru
  SDTCisVT<6, v16i8>,      // Operand 1: QACC_L[255:128] passthru
  SDTCisVT<7, v16i8>,      // Operand 2: QACC_H[127:0] passthru
  SDTCisVT<8, v16i8>,      // Operand 3: QACC_H[255:128] passthru
  SDTCisVT<9, v16i8>,      // Operand 4: qu vector
  SDTCisVT<10, v8i16>,     // Operand 5: qx vector
  SDTCisVT<11, v8i16>,     // Operand 6: qy vector
  SDTCisPtrTy<12>,         // Operand 7: rs1 pointer
  SDTCisVT<13, i32>        // Operand 8: offset immediate
]>;
def SDT_ESPVMULASU8QACCSTIPM : SDTypeProfile<5, 9, [
  SDTCisPtrTy<0>,          // Result 0: updated pointer
  SDTCisVT<1, v16i8>,      // Result 1: QACC_L[127:0]
  SDTCisVT<2, v16i8>,      // Result 2: QACC_L[255:128]
  SDTCisVT<3, v16i8>,      // Result 3: QACC_H[127:0]
  SDTCisVT<4, v16i8>,      // Result 4: QACC_H[255:128]
  SDTCisVT<5, v16i8>,      // Operand 0: QACC_L[127:0] passthru
  SDTCisVT<6, v16i8>,      // Operand 1: QACC_L[255:128] passthru
  SDTCisVT<7, v16i8>,      // Operand 2: QACC_H[127:0] passthru
  SDTCisVT<8, v16i8>,      // Operand 3: QACC_H[255:128] passthru
  SDTCisVT<9, v16i8>,      // Operand 4: qu vector
  SDTCisVT<10, v16i8>,     // Operand 5: qx vector
  SDTCisVT<11, v16i8>,     // Operand 6: qy vector
  SDTCisPtrTy<12>,         // Operand 7: rs1 pointer
  SDTCisVT<13, i32>        // Operand 8: offset immediate
]>;

def riscv_esp_vmulas_s16_qacc_st_ip_m : RVSDNode<"ESP_VMULAS_S16_QACC_ST_IP_M", SDT_ESPVMULASS16QACCSTIPM,
                                                  [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;
def riscv_esp_vmulas_s8_qacc_st_ip_m : RVSDNode<"ESP_VMULAS_S8_QACC_ST_IP_M", SDT_ESPVMULASS8QACCSTIPM,
                                                 [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;
def riscv_esp_vmulas_u16_qacc_st_ip_m : RVSDNode<"ESP_VMULAS_U16_QACC_ST_IP_M", SDT_ESPVMULASU16QACCSTIPM,
                                                   [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;
def riscv_esp_vmulas_u8_qacc_st_ip_m : RVSDNode<"ESP_VMULAS_U8_QACC_ST_IP_M", SDT_ESPVMULASU8QACCSTIPM,
                                                 [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;

// VMULAS QACC ST XP SDNodes (register offset)
def SDT_ESPVMULASS16QACCSTXPM : SDTypeProfile<5, 9, [
  SDTCisPtrTy<0>,          // Result 0: updated pointer
  SDTCisVT<1, v16i8>,      // Result 1: QACC_L[127:0]
  SDTCisVT<2, v16i8>,      // Result 2: QACC_L[255:128]
  SDTCisVT<3, v16i8>,      // Result 3: QACC_H[127:0]
  SDTCisVT<4, v16i8>,      // Result 4: QACC_H[255:128]
  SDTCisVT<5, v16i8>,      // Operand 0: QACC_L[127:0] passthru
  SDTCisVT<6, v16i8>,      // Operand 1: QACC_L[255:128] passthru
  SDTCisVT<7, v16i8>,      // Operand 2: QACC_H[127:0] passthru
  SDTCisVT<8, v16i8>,      // Operand 3: QACC_H[255:128] passthru
  SDTCisVT<9, v16i8>,      // Operand 4: qu vector
  SDTCisVT<10, v8i16>,     // Operand 5: qx vector
  SDTCisVT<11, v8i16>,     // Operand 6: qy vector
  SDTCisPtrTy<12>,         // Operand 7: rs1 pointer
  SDTCisVT<13, i32>        // Operand 8: rs2 register offset
]>;
def SDT_ESPVMULASS8QACCSTXPM : SDTypeProfile<5, 9, [
  SDTCisPtrTy<0>,          // Result 0: updated pointer
  SDTCisVT<1, v16i8>,      // Result 1: QACC_L[127:0]
  SDTCisVT<2, v16i8>,      // Result 2: QACC_L[255:128]
  SDTCisVT<3, v16i8>,      // Result 3: QACC_H[127:0]
  SDTCisVT<4, v16i8>,      // Result 4: QACC_H[255:128]
  SDTCisVT<5, v16i8>,      // Operand 0: QACC_L[127:0] passthru
  SDTCisVT<6, v16i8>,      // Operand 1: QACC_L[255:128] passthru
  SDTCisVT<7, v16i8>,      // Operand 2: QACC_H[127:0] passthru
  SDTCisVT<8, v16i8>,      // Operand 3: QACC_H[255:128] passthru
  SDTCisVT<9, v16i8>,      // Operand 4: qu vector
  SDTCisVT<10, v16i8>,     // Operand 5: qx vector
  SDTCisVT<11, v16i8>,     // Operand 6: qy vector
  SDTCisPtrTy<12>,         // Operand 7: rs1 pointer
  SDTCisVT<13, i32>        // Operand 8: rs2 register offset
]>;
def SDT_ESPVMULASU16QACCSTXPM : SDTypeProfile<5, 9, [
  SDTCisPtrTy<0>,          // Result 0: updated pointer
  SDTCisVT<1, v16i8>,      // Result 1: QACC_L[127:0]
  SDTCisVT<2, v16i8>,      // Result 2: QACC_L[255:128]
  SDTCisVT<3, v16i8>,      // Result 3: QACC_H[127:0]
  SDTCisVT<4, v16i8>,      // Result 4: QACC_H[255:128]
  SDTCisVT<5, v16i8>,      // Operand 0: QACC_L[127:0] passthru
  SDTCisVT<6, v16i8>,      // Operand 1: QACC_L[255:128] passthru
  SDTCisVT<7, v16i8>,      // Operand 2: QACC_H[127:0] passthru
  SDTCisVT<8, v16i8>,      // Operand 3: QACC_H[255:128] passthru
  SDTCisVT<9, v16i8>,      // Operand 4: qu vector
  SDTCisVT<10, v8i16>,     // Operand 5: qx vector
  SDTCisVT<11, v8i16>,     // Operand 6: qy vector
  SDTCisPtrTy<12>,         // Operand 7: rs1 pointer
  SDTCisVT<13, i32>        // Operand 8: rs2 register offset
]>;
def SDT_ESPVMULASU8QACCSTXPM : SDTypeProfile<5, 9, [
  SDTCisPtrTy<0>,          // Result 0: updated pointer
  SDTCisVT<1, v16i8>,      // Result 1: QACC_L[127:0]
  SDTCisVT<2, v16i8>,      // Result 2: QACC_L[255:128]
  SDTCisVT<3, v16i8>,      // Result 3: QACC_H[127:0]
  SDTCisVT<4, v16i8>,      // Result 4: QACC_H[255:128]
  SDTCisVT<5, v16i8>,      // Operand 0: QACC_L[127:0] passthru
  SDTCisVT<6, v16i8>,      // Operand 1: QACC_L[255:128] passthru
  SDTCisVT<7, v16i8>,      // Operand 2: QACC_H[127:0] passthru
  SDTCisVT<8, v16i8>,      // Operand 3: QACC_H[255:128] passthru
  SDTCisVT<9, v16i8>,      // Operand 4: qu vector
  SDTCisVT<10, v16i8>,     // Operand 5: qx vector
  SDTCisVT<11, v16i8>,     // Operand 6: qy vector
  SDTCisPtrTy<12>,         // Operand 7: rs1 pointer
  SDTCisVT<13, i32>        // Operand 8: rs2 register offset
]>;

def riscv_esp_vmulas_s16_qacc_st_xp_m : RVSDNode<"ESP_VMULAS_S16_QACC_ST_XP_M", SDT_ESPVMULASS16QACCSTXPM,
                                                  [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;
def riscv_esp_vmulas_s8_qacc_st_xp_m : RVSDNode<"ESP_VMULAS_S8_QACC_ST_XP_M", SDT_ESPVMULASS8QACCSTXPM,
                                                 [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;
def riscv_esp_vmulas_u16_qacc_st_xp_m : RVSDNode<"ESP_VMULAS_U16_QACC_ST_XP_M", SDT_ESPVMULASU16QACCSTXPM,
                                                   [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;
def riscv_esp_vmulas_u8_qacc_st_xp_m : RVSDNode<"ESP_VMULAS_U8_QACC_ST_XP_M", SDT_ESPVMULASU8QACCSTXPM,
                                                 [SDNPHasChain, SDNPMayStore, SDNPMemOperand]>;

// VMULAS QACC LDBC INCP SDNodes
def SDT_ESPVMULASS16QACCLDBCINCPM : SDTypeProfile<6, 7, [
  SDTCisVT<0, v16i8>,      // Result 0: qu vector
  SDTCisPtrTy<1>,          // Result 1: updated pointer
  SDTCisVT<2, v16i8>,      // Result 2: QACC_L[127:0]
  SDTCisVT<3, v16i8>,      // Result 3: QACC_L[255:128]
  SDTCisVT<4, v16i8>,      // Result 4: QACC_H[127:0]
  SDTCisVT<5, v16i8>,      // Result 5: QACC_H[255:128]
  SDTCisVT<6, v16i8>,      // Operand 0: QACC_L[127:0] passthru
  SDTCisVT<7, v16i8>,      // Operand 1: QACC_L[255:128] passthru
  SDTCisVT<8, v16i8>,      // Operand 2: QACC_H[127:0] passthru
  SDTCisVT<9, v16i8>,      // Operand 3: QACC_H[255:128] passthru
  SDTCisVT<10, v8i16>,     // Operand 4: qx vector
  SDTCisVT<11, v8i16>,     // Operand 5: qy vector
  SDTCisPtrTy<12>          // Operand 6: rs1 pointer
]>;
def SDT_ESPVMULASS8QACCLDBCINCPM : SDTypeProfile<6, 7, [
  SDTCisVT<0, v16i8>,      // Result 0: qu vector
  SDTCisPtrTy<1>,          // Result 1: updated pointer
  SDTCisVT<2, v16i8>,      // Result 2: QACC_L[127:0]
  SDTCisVT<3, v16i8>,      // Result 3: QACC_L[255:128]
  SDTCisVT<4, v16i8>,      // Result 4: QACC_H[127:0]
  SDTCisVT<5, v16i8>,      // Result 5: QACC_H[255:128]
  SDTCisVT<6, v16i8>,      // Operand 0: QACC_L[127:0] passthru
  SDTCisVT<7, v16i8>,      // Operand 1: QACC_L[255:128] passthru
  SDTCisVT<8, v16i8>,      // Operand 2: QACC_H[127:0] passthru
  SDTCisVT<9, v16i8>,      // Operand 3: QACC_H[255:128] passthru
  SDTCisVT<10, v16i8>,     // Operand 4: qx vector
  SDTCisVT<11, v16i8>,     // Operand 5: qy vector
  SDTCisPtrTy<12>          // Operand 6: rs1 pointer
]>;
def SDT_ESPVMULASU16QACCLDBCINCPM : SDTypeProfile<6, 7, [
  SDTCisVT<0, v16i8>,      // Result 0: qu vector
  SDTCisPtrTy<1>,          // Result 1: updated pointer
  SDTCisVT<2, v16i8>,      // Result 2: QACC_L[127:0]
  SDTCisVT<3, v16i8>,      // Result 3: QACC_L[255:128]
  SDTCisVT<4, v16i8>,      // Result 4: QACC_H[127:0]
  SDTCisVT<5, v16i8>,      // Result 5: QACC_H[255:128]
  SDTCisVT<6, v16i8>,      // Operand 0: QACC_L[127:0] passthru
  SDTCisVT<7, v16i8>,      // Operand 1: QACC_L[255:128] passthru
  SDTCisVT<8, v16i8>,      // Operand 2: QACC_H[127:0] passthru
  SDTCisVT<9, v16i8>,      // Operand 3: QACC_H[255:128] passthru
  SDTCisVT<10, v8i16>,     // Operand 4: qx vector
  SDTCisVT<11, v8i16>,     // Operand 5: qy vector
  SDTCisPtrTy<12>          // Operand 6: rs1 pointer
]>;
def SDT_ESPVMULASU8QACCLDBCINCPM : SDTypeProfile<6, 7, [
  SDTCisVT<0, v16i8>,      // Result 0: qu vector
  SDTCisPtrTy<1>,          // Result 1: updated pointer
  SDTCisVT<2, v16i8>,      // Result 2: QACC_L[127:0]
  SDTCisVT<3, v16i8>,      // Result 3: QACC_L[255:128]
  SDTCisVT<4, v16i8>,      // Result 4: QACC_H[127:0]
  SDTCisVT<5, v16i8>,      // Result 5: QACC_H[255:128]
  SDTCisVT<6, v16i8>,      // Operand 0: QACC_L[127:0] passthru
  SDTCisVT<7, v16i8>,      // Operand 1: QACC_L[255:128] passthru
  SDTCisVT<8, v16i8>,      // Operand 2: QACC_H[127:0] passthru
  SDTCisVT<9, v16i8>,      // Operand 3: QACC_H[255:128] passthru
  SDTCisVT<10, v16i8>,     // Operand 4: qx vector
  SDTCisVT<11, v16i8>,     // Operand 5: qy vector
  SDTCisPtrTy<12>          // Operand 6: rs1 pointer
]>;

def riscv_esp_vmulas_s16_qacc_ldbc_incp_m : RVSDNode<"ESP_VMULAS_S16_QACC_LDBC_INCP_M", SDT_ESPVMULASS16QACCLDBCINCPM,
                                                      [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def riscv_esp_vmulas_s8_qacc_ldbc_incp_m : RVSDNode<"ESP_VMULAS_S8_QACC_LDBC_INCP_M", SDT_ESPVMULASS8QACCLDBCINCPM,
                                                     [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def riscv_esp_vmulas_u16_qacc_ldbc_incp_m : RVSDNode<"ESP_VMULAS_U16_QACC_LDBC_INCP_M", SDT_ESPVMULASU16QACCLDBCINCPM,
                                                       [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
def riscv_esp_vmulas_u8_qacc_ldbc_incp_m : RVSDNode<"ESP_VMULAS_U8_QACC_LDBC_INCP_M", SDT_ESPVMULASU8QACCLDBCINCPM,
                                                     [SDNPHasChain, SDNPMayLoad, SDNPMemOperand]>;
