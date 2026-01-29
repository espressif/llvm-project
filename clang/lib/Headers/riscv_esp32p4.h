/*===---- riscv_esp32p4.h - RISC-V ESP32P4 intrinsics -----------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __RISCV_ESP32P4_H
#define __RISCV_ESP32P4_H

#include <stdint.h>

// Ensure stdint types are available even if stdint.h didn't define them
#ifndef int8_t
typedef signed char int8_t;
typedef unsigned char uint8_t;
#endif

#ifndef int32_t
typedef int int32_t;
typedef unsigned int uint32_t;
#endif

#if defined(__cplusplus)
extern "C" {
#endif

// Type definitions
typedef __attribute__((vector_size(16))) char esp_vec128_t;
typedef __attribute__((vector_size(16))) short esp_vec128_16_t;
typedef __attribute__((vector_size(16))) int esp_vec128_32_t;
typedef __attribute__((vector_size(8))) char esp_vec64_t;
typedef __attribute__((vector_size(64))) char esp_vec512_t;

typedef struct {
  union {
    esp_vec128_t V8;
    esp_vec128_16_t V16;
    esp_vec128_32_t V32;
  } Val;
  void *Ptr;
} esp_vld_res_t;

// 64-bit load result structure - for esp_vld_h_64_ip_m and esp_vld_l_64_ip_m
typedef struct {
  esp_vec64_t Val; // 64-bit vector
  void *Ptr;
} esp_vld_64_res_t;

// ESP.LD.UA.STATE.IP result structure - Load 128-bit Data to UA_STATE register
typedef struct {
  esp_vec128_t UaState; // UA_STATE value (128-bit)
  void *Ptr;            // Updated pointer
} esp_ua_state_res_t;

typedef struct {
  esp_vec128_t Val1;
  esp_vec128_t Val2;
  void *Ptr;
} esp_vldext_res_t;

// ESP.SRC.Q result structures
typedef struct {
  esp_vec128_t Qw;
  esp_vec128_t Qu;
  void *Ptr;
} esp_src_q_ld_res_t;

typedef struct {
  esp_vec128_t Qz;
  esp_vec128_t Qw;
} esp_src_q_qup_res_t;

// ESP.LD.128.USAR.IP/XP result structure with explicit SAR_BYTES
typedef struct {
  union {
    esp_vec128_t V8;
    esp_vec128_16_t V16;
    esp_vec128_32_t V32;
  } Val;
  void *Ptr;
  unsigned int SarBytes; // SAR_BYTES[3:0] = Rs1[3:0] (32-bit, only low 4 bits
                         // used, range 0-15)
} esp_vld_usar_res_t;

// ESP.ZERO.XACC result structure - Zero XACC with explicit state passing
// Mixed model: XACC as {unsigned int low, unsigned int high}
typedef struct {
  unsigned int xacc_low;  // XACC[31:0] (i32, set to 0)
  unsigned int xacc_high; // XACC[39:32] (i32, only low 8 bits valid, set to 0)
} esp_xacc_zero_res_t;

// ESP.LD.XACC.IP result structure - Load 64-bit Data and store low 40 bits to
// XACC Mixed model: XACC as {unsigned int low, unsigned int high}
typedef struct {
  unsigned int xacc_low;  // XACC[31:0] (i32)
  unsigned int xacc_high; // XACC[39:32] (i32, only low 8 bits valid)
  void *Ptr;              // Updated pointer
} esp_xacc_res_t;

// ESP.CMUL.*.LD.INCP result structure
typedef struct {
  esp_vec128_t Qz; // For u16/s16 is v8i16, for u8/s8 is v16i8
  esp_vec128_t Qu; // Always v16i8
  void *Ptr;
} esp_cmul_ld_incp_res_t;

// ESP.VOP.LD.INCP result structure - Vector operation with load and increment
// pointer Used for VADD, VSUB, VMUL, etc. LD.INCP instructions
typedef struct {
  union {
    esp_vec128_t V8;
    esp_vec128_16_t V16;
    esp_vec128_32_t V32;
  } Qv;
  esp_vec128_t Qu;
  void *Ptr;
} esp_vop_ld_incp_res_t;

// QACC as 4x128-bit structure for explicit phantom operand passing
typedef struct {
  esp_vec128_t v0; // QACC_L[127:0]: First 128 bits
  esp_vec128_t v1; // QACC_L[255:128]: Second 128 bits
  esp_vec128_t v2; // QACC_H[127:0]: Third 128 bits
  esp_vec128_t v3; // QACC_H[255:128]: Fourth 128 bits
} esp_qacc_4x128_t;
// ESP.VLD.128.IP.M / ESP.VST.128.IP.M - using immediate increment
static inline __attribute__((always_inline)) esp_vld_res_t
esp_vld_128_ip_m(void const *Ptr, int Imm) {
  esp_vld_res_t Res;
  Res.Ptr = __builtin_riscv_esp_vld_128_ip_m(Ptr, Imm, &Res.Val.V8);
  return Res;
}

static inline __attribute__((always_inline)) void *
esp_vst_128_ip_m(esp_vec128_t Data, void *Ptr, int Imm) {
  return __builtin_riscv_esp_vst_128_ip_m(Data, Ptr, Imm);
}

// ESP.LD.UA.STATE.IP.M / ESP.ST.UA.STATE.IP.M - Load/Store UA_STATE register
static inline __attribute__((always_inline)) esp_ua_state_res_t
esp_ld_ua_state_ip_m(void const *Ptr, int Imm) {
  esp_ua_state_res_t Res;
  // Builtin signature: void*(void const *, int, void *)
  // Parameters: (Ptr, offset, ua_state_out)
  // Returns: void* (updated pointer)
  // UA_STATE value is returned through output parameter
  Res.Ptr = __builtin_riscv_esp_ld_ua_state_ip_m(Ptr, Imm, &Res.UaState);
  return Res;
}

static inline __attribute__((always_inline)) void *
esp_st_ua_state_ip_m(esp_vec128_t UaState, void *Ptr, int Imm) {
  return __builtin_riscv_esp_st_ua_state_ip_m(UaState, Ptr, Imm);
}

// ESP.FFT.AMS.S16.LD.INCP.UAUP result structure with explicit phantom operands
typedef struct {
  esp_vec128_t Qu;      // v16i8 - loaded Data
  esp_vec128_16_t Qz;   // v8i16 - computation result
  esp_vec128_16_t Qv;   // v8i16 - computation result
  void *Ptr;            // Updated pointer
  esp_vec128_t UaState; // UA_STATE value (phantom operand output)
} esp_fft_ams_s16_ld_incp_uaup_res_t;

// ESP.FFT.AMS.S16.LD.INCP.UAUP wrapper function - FFT AMS with UA update and
// explicit state passing UA_STATE is input/output (phantom operand), SAR_BYTES
// and SAR are input-only (phantom operands) Input: ua_state_in, sar_bytes_in,
// sar_in - current state values (SAR_BYTES and SAR are read-only) Output:
// Res.UaState - new UA_STATE value after operation Returns structure with Qu,
// Qz, Qv, updated pointer, and new UA_STATE value
static inline __attribute__((always_inline)) esp_fft_ams_s16_ld_incp_uaup_res_t
esp_fft_ams_s16_ld_incp_uaup_m(esp_vec128_16_t Qx, esp_vec128_16_t Qy,
                               esp_vec128_16_t Qw, void const *Ptr, int Sel2,
                               esp_vec128_t ua_state_in,
                               unsigned int sar_bytes_in, unsigned int sar_in) {
  esp_fft_ams_s16_ld_incp_uaup_res_t Res;
  // Call builtin with explicit phantom operands
  Res.Ptr = __builtin_riscv_esp_fft_ams_s16_ld_incp_uaup_m(
      Qx, Qy, Qw, Ptr, Sel2, &Res.Qu, &Res.Qz, &Res.Qv, ua_state_in,
      &Res.UaState, sar_bytes_in, sar_in);
  return Res;
}

// ESP.VLD.128.XP.M / ESP.VST.128.XP.M - using register increment
static inline __attribute__((always_inline)) esp_vld_res_t
esp_vld_128_xp_m(void const *Ptr, int Reg) {
  esp_vld_res_t Res;
  Res.Ptr = __builtin_riscv_esp_vld_128_xp_m(Ptr, Reg, &Res.Val.V8);
  return Res;
}

static inline __attribute__((always_inline)) void *
esp_vst_128_xp_m(esp_vec128_t Data, void *Ptr, int Reg) {
  return __builtin_riscv_esp_vst_128_xp_m(Data, Ptr, Reg);
}

// ESP.LD.128.USAR.IP.M / ESP.LD.128.USAR.XP.M - Load 128-bit with SAR_BYTES
// update SAR_BYTE = Rs1[3:0] (hardware extracts low 4 bits from Ptr) Builtin
// returns SAR_BYTES as output parameter
static inline __attribute__((always_inline)) esp_vld_usar_res_t
esp_ld_128_usar_ip_m(void const *Ptr, int Imm) {
  esp_vld_usar_res_t Res;
  Res.Ptr = __builtin_riscv_esp_ld_128_usar_ip_m(Ptr, Imm, &Res.Val.V8,
                                                 &Res.SarBytes);
  // SAR_BYTES is filled by builtin as output parameter
  return Res;
}

static inline __attribute__((always_inline)) esp_vld_usar_res_t
esp_ld_128_usar_xp_m(void const *Ptr, int Reg) {
  esp_vld_usar_res_t Res;
  Res.Ptr = __builtin_riscv_esp_ld_128_usar_xp_m(Ptr, Reg, &Res.Val.V8,
                                                 &Res.SarBytes);
  // SAR_BYTES is filled by builtin as output parameter
  return Res;
}

// ESP.VLD.H.64.IP.M / ESP.VLD.L.64.IP.M - Load 64-bit (high/low)
// Return 64-bit structure to match actual Data size (similar to
// esp_vst_h_64_ip_m pattern) Users can combine two 64-bit results into 128-bit
// using union (see test_64_to_128)
static inline __attribute__((always_inline)) esp_vld_64_res_t
esp_vld_h_64_ip_m(void const *Ptr, int Imm) {
  esp_vld_64_res_t Res;
  // Create temporary 128-bit vector for builtin storage
  // Builtin stores 64-bit Data to offset 8 (high 64 bits)
  esp_vec128_t TempVec;
  Res.Ptr = __builtin_riscv_esp_vld_h_64_ip_m(Ptr, Imm, &TempVec);
  // Extract high 64 bits from offset 8 using union (similar to
  // esp_vst_h_64_ip_m)
  union {
    esp_vec128_t V128;
    struct {
      esp_vec64_t Low;
      esp_vec64_t High;
    } Parts;
  } U;
  U.V128 = TempVec;
  Res.Val = U.Parts.High;
  return Res;
}

static inline __attribute__((always_inline)) esp_vld_64_res_t
esp_vld_l_64_ip_m(void const *Ptr, int Imm) {
  esp_vld_64_res_t Res;
  // Create temporary 128-bit vector for builtin storage
  // Builtin stores 64-bit Data to offset 0 (low 64 bits)
  esp_vec128_t TempVec;
  Res.Ptr = __builtin_riscv_esp_vld_l_64_ip_m(Ptr, Imm, &TempVec);
  // Extract low 64 bits from offset 0 using union (similar to
  // esp_vst_l_64_ip_m)
  union {
    esp_vec128_t V128;
    struct {
      esp_vec64_t Low;
      esp_vec64_t High;
    } Parts;
  } U;
  U.V128 = TempVec;
  Res.Val = U.Parts.Low;
  return Res;
}

static inline __attribute__((always_inline)) void *
esp_vst_h_64_ip_m(esp_vec128_t Data, void *Ptr, int Imm) {
  union {
    esp_vec128_t V128;
    struct {
      esp_vec64_t Low;
      esp_vec64_t High;
    } Parts;
  } U;
  U.V128 = Data;
  return __builtin_riscv_esp_vst_h_64_ip_m(U.Parts.High, Ptr, Imm);
}

static inline __attribute__((always_inline)) void *
esp_vst_l_64_ip_m(esp_vec128_t Data, void *Ptr, int Imm) {
  union {
    esp_vec128_t V128;
    struct {
      esp_vec64_t Low;
      esp_vec64_t High;
    } Parts;
  } U;
  U.V128 = Data;
  return __builtin_riscv_esp_vst_l_64_ip_m(U.Parts.Low, Ptr, Imm);
}

// ESP.VLD.H.64.XP.M / ESP.VLD.L.64.XP.M - Load 64-bit with register offset
// Return 64-bit structure (consistent with esp_vld_h_64_ip_m)
static inline __attribute__((always_inline)) esp_vld_64_res_t
esp_vld_h_64_xp_m(void const *Ptr, int Reg) {
  esp_vld_64_res_t Res;
  esp_vec128_t TempVec;
  Res.Ptr = __builtin_riscv_esp_vld_h_64_xp_m(Ptr, Reg, &TempVec);
  union {
    esp_vec128_t V128;
    struct {
      esp_vec64_t Low;
      esp_vec64_t High;
    } Parts;
  } U;
  U.V128 = TempVec;
  Res.Val = U.Parts.High;
  return Res;
}

static inline __attribute__((always_inline)) esp_vld_64_res_t
esp_vld_l_64_xp_m(void const *Ptr, int Reg) {
  esp_vld_64_res_t Res;
  esp_vec128_t TempVec;
  Res.Ptr = __builtin_riscv_esp_vld_l_64_xp_m(Ptr, Reg, &TempVec);
  union {
    esp_vec128_t V128;
    struct {
      esp_vec64_t Low;
      esp_vec64_t High;
    } Parts;
  } U;
  U.V128 = TempVec;
  Res.Val = U.Parts.Low;
  return Res;
}

static inline __attribute__((always_inline)) void *
esp_vst_h_64_xp_m(esp_vec128_t Data, void *Ptr, int Reg) {
  union {
    esp_vec128_t V128;
    struct {
      esp_vec64_t Low;
      esp_vec64_t High;
    } Parts;
  } U;
  U.V128 = Data;
  return __builtin_riscv_esp_vst_h_64_xp_m(U.Parts.High, Ptr, Reg);
}

static inline __attribute__((always_inline)) void *
esp_vst_l_64_xp_m(esp_vec128_t Data, void *Ptr, int Reg) {
  union {
    esp_vec128_t V128;
    struct {
      esp_vec64_t Low;
      esp_vec64_t High;
    } Parts;
  } U;
  U.V128 = Data;
  return __builtin_riscv_esp_vst_l_64_xp_m(U.Parts.Low, Ptr, Reg);
}

// ESP.VLDBC - Vector Load Broadcast
static inline __attribute__((always_inline)) esp_vld_res_t
esp_vldbc_8_ip_m(void const *Ptr, int Imm) {
  esp_vld_res_t Res;
  Res.Ptr = __builtin_riscv_esp_vldbc_8_ip_m(Ptr, Imm, &Res.Val.V8);
  return Res;
}

static inline __attribute__((always_inline)) esp_vld_res_t
esp_vldbc_16_ip_m(void const *Ptr, int Imm) {
  esp_vld_res_t Res;
  Res.Ptr = __builtin_riscv_esp_vldbc_16_ip_m(Ptr, Imm, &Res.Val.V8);
  return Res;
}

static inline __attribute__((always_inline)) esp_vld_res_t
esp_vldbc_32_ip_m(void const *Ptr, int Imm) {
  esp_vld_res_t Res;
  Res.Ptr = __builtin_riscv_esp_vldbc_32_ip_m(Ptr, Imm, &Res.Val.V8);
  return Res;
}

static inline __attribute__((always_inline)) esp_vld_res_t
esp_vldbc_8_xp_m(void const *Ptr, int Reg) {
  esp_vld_res_t Res;
  Res.Ptr = __builtin_riscv_esp_vldbc_8_xp_m(Ptr, Reg, &Res.Val.V8);
  return Res;
}

static inline __attribute__((always_inline)) esp_vld_res_t
esp_vldbc_16_xp_m(void const *Ptr, int Reg) {
  esp_vld_res_t Res;
  Res.Ptr = __builtin_riscv_esp_vldbc_16_xp_m(Ptr, Reg, &Res.Val.V8);
  return Res;
}

static inline __attribute__((always_inline)) esp_vld_res_t
esp_vldbc_32_xp_m(void const *Ptr, int Reg) {
  esp_vld_res_t Res;
  Res.Ptr = __builtin_riscv_esp_vldbc_32_xp_m(Ptr, Reg, &Res.Val.V8);
  return Res;
}

// ESP.VLDEXT - Vector Load and Extend
static inline __attribute__((always_inline)) esp_vldext_res_t
esp_vldext_s8_ip_m(void const *Ptr, int Imm) {
  esp_vldext_res_t Res;
  Res.Ptr = __builtin_riscv_esp_vldext_s8_ip_m(Ptr, Imm, &Res.Val1, &Res.Val2);
  return Res;
}

static inline __attribute__((always_inline)) esp_vldext_res_t
esp_vldext_s16_ip_m(void const *Ptr, int Imm) {
  esp_vldext_res_t Res;
  Res.Ptr = __builtin_riscv_esp_vldext_s16_ip_m(Ptr, Imm, &Res.Val1, &Res.Val2);
  return Res;
}

static inline __attribute__((always_inline)) esp_vldext_res_t
esp_vldext_u16_ip_m(void const *Ptr, int Imm) {
  esp_vldext_res_t Res;
  Res.Ptr = __builtin_riscv_esp_vldext_u16_ip_m(Ptr, Imm, &Res.Val1, &Res.Val2);
  return Res;
}

static inline __attribute__((always_inline)) esp_vldext_res_t
esp_vldext_s8_xp_m(void const *Ptr, int Reg) {
  esp_vldext_res_t Res;
  Res.Ptr = __builtin_riscv_esp_vldext_s8_xp_m(Ptr, Reg, &Res.Val1, &Res.Val2);
  return Res;
}

static inline __attribute__((always_inline)) esp_vldext_res_t
esp_vldext_s16_xp_m(void const *Ptr, int Reg) {
  esp_vldext_res_t Res;
  Res.Ptr = __builtin_riscv_esp_vldext_s16_xp_m(Ptr, Reg, &Res.Val1, &Res.Val2);
  return Res;
}

static inline __attribute__((always_inline)) esp_vldext_res_t
esp_vldext_u8_xp_m(void const *Ptr, int Reg) {
  esp_vldext_res_t Res;
  Res.Ptr = __builtin_riscv_esp_vldext_u8_xp_m(Ptr, Reg, &Res.Val1, &Res.Val2);
  return Res;
}

static inline __attribute__((always_inline)) esp_vldext_res_t
esp_vldext_u16_xp_m(void const *Ptr, int Reg) {
  esp_vldext_res_t Res;
  Res.Ptr = __builtin_riscv_esp_vldext_u16_xp_m(Ptr, Reg, &Res.Val1, &Res.Val2);
  return Res;
}

// ESP.VADD.*.LD.INCP wrapper functions - Vector add with load and increment
// pointer
static inline __attribute__((always_inline)) esp_vop_ld_incp_res_t
esp_vadd_s8_ld_incp_m(esp_vec128_t Qx, esp_vec128_t Qy, void const *Rs1) {
  esp_vop_ld_incp_res_t Res;
  Res.Ptr =
      __builtin_riscv_esp_vadd_s8_ld_incp_m(Qx, Qy, Rs1, &Res.Qv.V8, &Res.Qu);
  return Res;
}

static inline __attribute__((always_inline)) esp_vop_ld_incp_res_t
esp_vadd_u8_ld_incp_m(esp_vec128_t Qx, esp_vec128_t Qy, void const *Rs1) {
  esp_vop_ld_incp_res_t Res;
  Res.Ptr =
      __builtin_riscv_esp_vadd_u8_ld_incp_m(Qx, Qy, Rs1, &Res.Qv.V8, &Res.Qu);
  return Res;
}

static inline __attribute__((always_inline)) esp_vop_ld_incp_res_t
esp_vadd_s16_ld_incp_m(esp_vec128_16_t Qx, esp_vec128_16_t Qy,
                       void const *Rs1) {
  esp_vop_ld_incp_res_t Res;
  // qv_out is esp_vec128_t*; Res.Qv.V8 and Res.Qv.V16 alias the same 128 bits.
  Res.Ptr =
      __builtin_riscv_esp_vadd_s16_ld_incp_m(Qx, Qy, Rs1, &Res.Qv.V8, &Res.Qu);
  return Res;
}

static inline __attribute__((always_inline)) esp_vop_ld_incp_res_t
esp_vadd_u16_ld_incp_m(esp_vec128_16_t Qx, esp_vec128_16_t Qy,
                       void const *Rs1) {
  esp_vop_ld_incp_res_t Res;
  Res.Ptr =
      __builtin_riscv_esp_vadd_u16_ld_incp_m(Qx, Qy, Rs1, &Res.Qv.V8, &Res.Qu);
  return Res;
}

// ESP.ST.S.XACC.IP (_m version) - Store Signed XACC with Immediate
// Post-increment Returns: updated pointer Mixed model: XACC as {unsigned int
// low, unsigned int high} This wrapper calls the builtin and uses explicit
// state passing for XACC Note: XACC is passthru (unchanged) for ST instructions
// For simplified API, we use 0 as default XACC value (caller should initialize
// XACC before calling)
static inline __attribute__((always_inline)) void *esp_st_s_xacc_ip_m(void *Ptr,
                                                                      int Imm) {
  void *PtrOut;
  unsigned int XaccLowOut;
  unsigned int XaccHighOut;
  // Use 0 as default XACC value (caller should have initialized XACC via
  // __builtin_riscv_esp_zero_xacc() or similar) For explicit state passing, we
  // pass current XACC state (passthru)
  unsigned int XaccLowIn = 0U;  // Default: XACC[31:0] = 0
  unsigned int XaccHighIn = 0U; // Default: XACC[39:32] = 0
  // Call builtin to get updated pointer
  // Builtin signature: void*(unsigned int, unsigned int, void *, int, void *,
  // unsigned int *, unsigned int *) XaccLowIn, XaccHighIn: current XACC state
  // (input, passthru) &PtrOut: pointer to store updated pointer (output)
  // &XaccLowOut, &XaccHighOut: pointers to store XACC state (output, unchanged
  // for ST)
  return __builtin_riscv_esp_st_s_xacc_ip_m(XaccLowIn, XaccHighIn, Ptr, Imm,
                                            &PtrOut, &XaccLowOut, &XaccHighOut);
}

#if defined(__cplusplus)
}
#endif

#endif // __RISCV_ESP32P4_H
