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
