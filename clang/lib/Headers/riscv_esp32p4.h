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
    esp_vec128_t v8;
    esp_vec128_16_t v16;
    esp_vec128_32_t v32;
  } Qv;
  esp_vec128_t Qu;
  void *Ptr;
} esp_vop_ld_incp_res_t;

// ESP.VMULAS.*.XACC.LD.IP/XP result structure with explicit XACC
// Mixed model: XACC as {unsigned int low, unsigned int high}
typedef struct {
  union {
    esp_vec128_t v8;
    esp_vec128_16_t v16;
  } Qu;                   // Loaded 128-bit Data
  void *Ptr;              // Updated pointer
  unsigned int xacc_low;  // XACC[31:0] (i32)
  unsigned int xacc_high; // XACC[39:32] (i32, only low 8 bits valid)
} esp_vmulas_xacc_ld_res_t;

// ESP.VMULAS.*.XACC.ST.IP/XP result structure with explicit XACC
// Mixed model: XACC as {unsigned int low, unsigned int high}
typedef struct {
  void *Ptr;              // Updated pointer
  unsigned int xacc_low;  // XACC[31:0] (i32)
  unsigned int xacc_high; // XACC[39:32] (i32, only low 8 bits valid)
} esp_vmulas_xacc_st_res_t;

// Partial QACC state: QACC_H or QACC_L half (same layout for MOV and VCMULAS).
typedef struct {
  esp_vec128_t v2; // QACC_H[127:0]
  esp_vec128_t v3; // QACC_H[255:128]
} esp_mov_qacc_h_res_t;

typedef struct {
  esp_vec128_t v0; // QACC_L[127:0]
  esp_vec128_t v1; // QACC_L[255:128]
} esp_mov_qacc_l_res_t;

typedef esp_mov_qacc_h_res_t esp_vcmulas_qacc_h_res_t;
typedef esp_mov_qacc_l_res_t esp_vcmulas_qacc_l_res_t;

// QACC as 4x128-bit structure for explicit phantom operand passing
typedef struct {
  esp_vec128_t v0; // QACC_L[127:0]: First 128 bits
  esp_vec128_t v1; // QACC_L[255:128]: Second 128 bits
  esp_vec128_t v2; // QACC_H[127:0]: Third 128 bits
  esp_vec128_t v3; // QACC_H[255:128]: Fourth 128 bits
} esp_qacc_4x128_t;

// Reinterpret 128-bit vector between element-width views (same underlying
// bits).
static inline __attribute__((always_inline)) esp_vec128_t
esp_vec128_16_to_8(esp_vec128_16_t v) {
  union {
    esp_vec128_16_t V16;
    esp_vec128_t V8;
  } U;
  U.V16 = v;
  return U.V8;
}

// ESP.VLD.128.IP.M / ESP.VST.128.IP.M - using immediate increment

// ESP.LD.UA.STATE.IP.M / ESP.ST.UA.STATE.IP.M - Load/Store UA_STATE register

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

// ESP.VLD.128.XP.M / ESP.VST.128.XP.M - using register increment

// ESP.LD.128.USAR.IP.M / ESP.LD.128.USAR.XP.M - Load 128-bit with SAR_BYTES
// update SAR_BYTE = Rs1[3:0] (hardware extracts low 4 bits from Ptr) Builtin
// returns SAR_BYTES as output parameter

// ESP.VLD.H.64.IP.M / ESP.VLD.L.64.IP.M - Load 64-bit (high/low)
// Return 64-bit structure to match actual Data size (similar to
// esp_vst_h_64_ip_m pattern) Users can combine two 64-bit results into 128-bit
// using union (see test_64_to_128)

// ESP.VLD.H.64.XP.M / ESP.VLD.L.64.XP.M - Load 64-bit with register offset
// Return 64-bit structure (consistent with esp_vld_h_64_ip_m)

// ESP.VLDBC - Vector Load Broadcast

// ESP.VLDEXT - Vector Load and Extend

// ESP.VADD.*.LD.INCP wrapper functions - Vector add with load and increment
// pointer

// ESP.ST.S.XACC.IP (_m version) - Store Signed XACC with Immediate
// Post-increment Returns: updated pointer Mixed model: XACC as {unsigned int
// low, unsigned int high} This wrapper calls the builtin and uses explicit
// state passing for XACC Note: XACC is passthru (unchanged) for ST instructions
// For simplified API, we use 0 as default XACC value (caller should initialize
// XACC before calling)

// QACC register types for explicit state passing
typedef __attribute__((
    vector_size(32))) int8_t esp_qacc_h_t; // QACC_H: v32i8 (256-bit)
typedef __attribute__((
    vector_size(32))) int8_t esp_qacc_l_t; // QACC_L: v32i8 (256-bit)

// ESP.LD.QACC result structures
typedef struct {
  esp_vec128_t qacc_128; // QACC value (128-bit)
  void *Ptr;
} esp_qacc_128_res_t;

// ESP.LDQA result structure - Load 128-bit Data, extend and store to QACC
typedef struct {
  esp_vec128_t v0; // QACC_L[127:0]
  esp_vec128_t v1; // QACC_L[255:128]
  esp_vec128_t v2; // QACC_H[127:0]
  esp_vec128_t v3; // QACC_H[255:128]
  void *Ptr;
} esp_ldqa_res_t;

// ESP.MOV result structure - Move vector to QACC
typedef struct {
  esp_vec128_t v0; // QACC_L[127:0]
  esp_vec128_t v1; // QACC_L[255:128]
  esp_vec128_t v2; // QACC_H[127:0]
  esp_vec128_t v3; // QACC_H[255:128]
} esp_mov_qacc_res_t;

// ESP.VMULAS result structure
typedef esp_mov_qacc_res_t esp_vmulas_qacc_res_t;

// ESP.VMULAS result structure with pointer
typedef struct {
  esp_vec128_t v0; // QACC_L[127:0]
  esp_vec128_t v1; // QACC_L[255:128]
  esp_vec128_t v2; // QACC_H[127:0]
  esp_vec128_t v3; // QACC_H[255:128]
  esp_vec128_t Qu; // Qu vector
  void *Ptr;
} esp_vmulas_qacc_ld_res_t;

// ESP.VSMULAS result structure
typedef esp_mov_qacc_res_t esp_vsmulas_qacc_res_t;

// VCMULAS QACC H LD result structure
typedef struct {
  esp_vec128_t v2; // QACC_H[127:0]
  esp_vec128_t v3; // QACC_H[255:128]
  esp_vec128_t Qu;
  void *Ptr;
} esp_vcmulas_qacc_h_ld_res_t;

// VCMULAS QACC L LD result structure
typedef struct {
  esp_vec128_t v0; // QACC_L[127:0]
  esp_vec128_t v1; // QACC_L[255:128]
  esp_vec128_t Qu;
  void *Ptr;
} esp_vcmulas_qacc_l_ld_res_t;

// ESP.ZERO.QACC (_m version)

// ESP.MOV.S8.QACC (_m version)

// ESP.MOV.S16.QACC (_m version)

// ESP.MOV.U8.QACC (_m version)

// ESP.MOV.U16.QACC (_m version)

// ESP.VMULAS.S8.QACC (_m version)

// ESP.VMULAS.S8.QACC.ST.IP (_m version)

// ESP.VCMULAS.S8.QACC.H (_m version)

// ESP.VCMULAS.S8.QACC.L (_m version)

// ESP.VCMULAS.S16.QACC.H (_m version)

// ESP.VCMULAS.S16.QACC.L (_m version)

// ESP.VCMULAS.S8.QACC.H.LD.IP (_m version)

// ESP.VCMULAS.S8.QACC.L.LD.IP (_m version)

// ESP.LD.QACC.H.H.128.IP (_m version)

// ESP.LD.QACC.H.L.128.IP (_m version)

// ESP.LD.QACC.L.H.128.IP (_m version)

// ESP.LD.QACC.L.L.128.IP (_m version)

// ESP.ST.QACC.H.H.128.IP (_m version)

// ESP.ST.QACC.H.L.128.IP (_m version)

// ESP.ST.QACC.L.H.128.IP (_m version)

// ESP.ST.QACC.L.L.128.IP (_m version)

// ESP.LDQA.S16.128.IP (_m version)

// ESP.LDQA.S16.128.XP (_m version)

// ESP.LDQA.S8.128.IP (_m version)

// ESP.LDQA.S8.128.XP (_m version)

// ESP.LDQA.U16.128.IP (_m version)

// ESP.LDQA.U16.128.XP (_m version)

// ESP.LDQA.U8.128.IP (_m version)

// ESP.LDQA.U8.128.XP (_m version)

// ESP.SRCMB.S16.QACC (_m version)

// ESP.SRCMB.S8.QACC (_m version)

// ESP.SRCMB.U16.QACC (_m version)

// ESP.SRCMB.U8.QACC (_m version)

// ESP.VMULAS.S16.QACC (_m version)

// ESP.VMULAS.U16.QACC (_m version)

// ESP.VMULAS.U8.QACC (_m version)

// ESP.VMULAS.S16.QACC.LD.IP (_m version)

// ESP.VMULAS.S16.QACC.LD.XP (_m version)

// ESP.VMULAS.S8.QACC.LD.IP (_m version)

// ESP.VMULAS.S8.QACC.LD.XP (_m version)

// ESP.VMULAS.U16.QACC.LD.IP (_m version)

// ESP.VMULAS.U16.QACC.LD.XP (_m version)

// ESP.VMULAS.U8.QACC.LD.IP (_m version)

// ESP.VMULAS.U8.QACC.LD.XP (_m version)

// ESP.VMULAS.S16.QACC.ST.IP (_m version)

// ESP.VMULAS.S16.QACC.ST.XP (_m version)

// ESP.VMULAS.S8.QACC.ST.XP (_m version)

// ESP.VMULAS.U16.QACC.ST.IP (_m version)

// ESP.VMULAS.U16.QACC.ST.XP (_m version)

// ESP.VMULAS.U8.QACC.ST.IP (_m version)

// ESP.VMULAS.U8.QACC.ST.XP (_m version)

// ESP.VMULAS.S16.QACC.LDBC.INCP (_m version)

// ESP.VMULAS.S8.QACC.LDBC.INCP (_m version)

// ESP.VMULAS.U16.QACC.LDBC.INCP (_m version)

// ESP.VMULAS.U8.QACC.LDBC.INCP (_m version)

// ESP.VCMULAS.S16.QACC.H.LD.IP (_m version)

// ESP.VCMULAS.S16.QACC.L.LD.IP (_m version)

// ESP.VCMULAS.S16.QACC.H.LD.XP (_m version)

// ESP.VCMULAS.S16.QACC.L.LD.XP (_m version)

// ESP.VCMULAS.S8.QACC.H.LD.XP (_m version)

// ESP.VCMULAS.S8.QACC.L.LD.XP (_m version)

// ESP.VMULAS.*.XACC.LD.IP/XP wrapper functions - Load Data and perform
// multiply-accumulate XACC is both input and output (explicit state passing for
// chaining) Mixed model: XACC as {unsigned int low, unsigned int high} Input:
// XaccLowIn, XaccHighIn - current XACC state (input) Output: Res.xacc_low,
// Res.xacc_high - new XACC value after multiply-accumulate operation Returns
// structure with Qu (128-bit loaded Data), updated pointer, and new XACC

// ESP.VMULAS.*.XACC.ST.IP/XP wrapper functions - Store Data and perform
// multiply-accumulate Mixed model: XACC as {unsigned int low, unsigned int
// high} XACC explicit state passing: XaccLowIn, XaccHighIn (input) ->
// instruction -> Res.xacc_low, Res.xacc_high (output) This enables chaining:
// xacc_old -> VMULAS -> xacc_new -> next instruction -> ... Returns structure
// with updated pointer and new XACC

// QACC_H: v32i8 (256-bit)  // QACC_L: v32i8 (256-bit)

// 128-bit types (16 bytes) - Subregisters
typedef __attribute__((vector_size(
    16))) int8_t esp_qacc_h_high_t; // QACC_H[255:128]: v16i8 (128-bit)
typedef __attribute__((
    vector_size(16))) int8_t esp_qacc_h_low_t; // QACC_H[127:0]: v16i8 (128-bit)
typedef __attribute__((vector_size(
    16))) int8_t esp_qacc_l_high_t; // QACC_L[255:128]: v16i8 (128-bit)
typedef __attribute__((
    vector_size(16))) int8_t esp_qacc_l_low_t; // QACC_L[127:0]: v16i8 (128-bit)

// Union structure for QACC (512-bit) with multiple views:
// - 512-bit full view (for instructions that use full QACC)
// - 256-bit L/H view (for instructions that use QACC_L or QACC_H separately)
// - 128-bit subregister view (for instructions that use 128-bit parts)
//
// Usage examples:
//   1. 512-bit view (full QACC):
//      esp_qacc_union_t qacc;
//      qacc.full = __builtin_riscv_esp_zero_qacc_m();
//
//   2. 256-bit view (QACC_L and QACC_H):
//      esp_qacc_l_t qacc_l = qacc.parts_256.l;  // Access QACC_L (256-bit)
//      esp_qacc_h_t qacc_h = qacc.parts_256.h;  // Access QACC_H (256-bit)
//
//   3. 128-bit view (subregisters):
//      esp_qacc_l_low_t l_low = qacc.parts_128.l_low;    // QACC_L[127:0]
//      esp_qacc_l_high_t l_high = qacc.parts_128.l_high;  // QACC_L[255:128]
//      esp_qacc_h_low_t h_low = qacc.parts_128.h_low;    // QACC_H[127:0]
//      esp_qacc_h_high_t h_high = qacc.parts_128.h_high;  // QACC_H[255:128]
//
// Note: All views share the same memory, so modifying one view affects all
// others.
typedef union {
  esp_vec512_t full; // Full 512-bit QACC view
  struct {
    esp_qacc_l_t l; // QACC_L: Low 256 bits (v32i8)
    esp_qacc_h_t h; // QACC_H: High 256 bits (v32i8)
  } parts_256;      // 256-bit view: L and H
  struct {
    esp_qacc_l_low_t l_low;   // QACC_L[127:0]: Low 128 bits of QACC_L
    esp_qacc_l_high_t l_high; // QACC_L[255:128]: High 128 bits of QACC_L
    esp_qacc_h_low_t h_low;   // QACC_H[127:0]: Low 128 bits of QACC_H
    esp_qacc_h_high_t h_high; // QACC_H[255:128]: High 128 bits of QACC_H
  } parts_128;                // 128-bit view: Four subregisters
} esp_qacc_union_t;

// QACC result structure (similar to esp_vld_res_t pattern)
// Provides multiple views: 1x512-bit, 2x256-bit, 4x128-bit
typedef struct {
  union {
    esp_vec512_t v512; // 1x 512-bit (full QACC)
    struct {
      esp_qacc_l_t l; // QACC_L: Low 256 bits
      esp_qacc_h_t h; // QACC_H: High 256 bits
    } v256;           // 2x 256-bit
    struct {
      esp_vec128_t v0; // QACC_L[127:0]: First 128 bits
      esp_vec128_t v1; // QACC_L[255:128]: Second 128 bits
      esp_vec128_t v2; // QACC_H[127:0]: Third 128 bits
      esp_vec128_t v3; // QACC_H[255:128]: Fourth 128 bits
    } v128;            // 4x 128-bit
  } Val;
  void *Ptr;
} esp_qacc_res_t;

// QACC pair structure (similar to esp_vld_res_t pattern)
// Uses union to provide multiple access methods
typedef struct {
  esp_qacc_h_t h; // QACC_H: offset 0, 256-bit
  esp_qacc_l_t l; // QACC_L: offset 32, 256-bit
} esp_qacc_pair_t;

// Helper functions to extract 128-bit subvectors from 256-bit QACC registers
// ESP.ST.QACC.*.128.IP instructions only store 128 bits, so we need to extract
// the low or high 128-bit part from the 256-bit QACC_L or QACC_H
// These use builtin functions (similar to esp_qacc_get_l/h) for zero-overhead
// extraction

// Extract QACC_L[127:0] (low 128 bits) from QACC_L (256 bits)

// Extract QACC_L[255:128] (high 128 bits) from QACC_L (256 bits)

// Extract QACC_H[127:0] (low 128 bits) from QACC_H (256 bits)

// Extract QACC_H[255:128] (high 128 bits) from QACC_H (256 bits)

// Convert esp_qacc_pair_t to 4x128-bit representation
// This is useful for passing QACC as explicit phantom operand to instructions

// ESP.ZERO.XACC (_m version) - Zero XACC with explicit state passing
// Returns: {i32 low=0, i32 high=0} for explicit state passing
// Mixed model: XACC as {unsigned int low, unsigned int high}
// This wrapper calls the intrinsic and returns XACC state as {i32, i32}

// ESP.LD.XACC.IP (_m version) - Load 64-bit Data and store low 40 bits to XACC
// Returns: structure with XACC value (mixed model: {i32 low, i32 high}) and
// updated pointer Mixed model: XACC as {unsigned int low, unsigned int high}
// This wrapper calls the builtin and uses explicit state passing for XACC
// QACC_H: v32i8 (256-bit)  // QACC_L: v32i8 (256-bit)  // QACC_H[255:128]:
// v16i8 (128-bit)   // QACC_H[127:0]: v16i8 (128-bit)  // QACC_L[255:128]:
// v16i8 (128-bit)   // QACC_L[127:0]: v16i8 (128-bit)

// ESP.ST.U.XACC.IP (_m version) - Store Unsigned XACC with Immediate
// Post-increment Returns: updated pointer Mixed model: XACC as {unsigned int
// low, unsigned int high} This wrapper calls the builtin and uses explicit
// state passing for XACC Note: XACC is passthru (unchanged) for ST instructions
// For simplified API, we use 0 as default XACC value (caller should initialize
// XACC before calling)
// QACC_H: v32i8 (256-bit)  // QACC_L: v32i8 (256-bit)  // QACC_H[255:128]:
// v16i8 (128-bit)   // QACC_H[127:0]: v16i8 (128-bit)  // QACC_L[255:128]:
// v16i8 (128-bit)   // QACC_L[127:0]: v16i8 (128-bit)

typedef struct {
  esp_qacc_l_t qacc_l; // QACC_L value (256-bit)
  void *Ptr;           // Updated pointer
} esp_qacc_l_res_t; // QACC_H: v32i8 (256-bit)  // QACC_L: v32i8 (256-bit)  //
                    // QACC_H[255:128]: v16i8 (128-bit)   // QACC_H[127:0]:
                    // v16i8 (128-bit)  // QACC_L[255:128]: v16i8 (128-bit)   //
                    // QACC_L[127:0]: v16i8 (128-bit)  // QACC_H: v32i8
                    // (256-bit)  // QACC_L: v32i8 (256-bit)  //
                    // QACC_H[255:128]: v16i8 (128-bit)   // QACC_H[127:0]:
                    // v16i8 (128-bit)  // QACC_L[255:128]: v16i8 (128-bit)   //
                    // QACC_L[127:0]: v16i8 (128-bit)  // QACC_H: v32i8
                    // (256-bit)  // QACC_L: v32i8 (256-bit)  //
                    // QACC_H[255:128]: v16i8 (128-bit)   // QACC_H[127:0]:
                    // v16i8 (128-bit)  // QACC_L[255:128]: v16i8 (128-bit)   //
                    // QACC_L[127:0]: v16i8 (128-bit)  // QACC_H: v32i8
                    // (256-bit)  // QACC_L: v32i8 (256-bit)  //
                    // QACC_H[255:128]: v16i8 (128-bit)   // QACC_H[127:0]:
                    // v16i8 (128-bit)  // QACC_L[255:128]: v16i8 (128-bit)   //
                    // QACC_L[127:0]: v16i8 (128-bit)  // QACC_H: v32i8
                    // (256-bit)  // QACC_L: v32i8 (256-bit)  //
                    // QACC_H[255:128]: v16i8 (128-bit)   // QACC_H[127:0]:
                    // v16i8 (128-bit)  // QACC_L[255:128]: v16i8 (128-bit)   //
                    // QACC_L[127:0]: v16i8 (128-bit)  // QACC_H: v32i8
                    // (256-bit)  // QACC_L: v32i8 (256-bit)  //
                    // QACC_H[255:128]: v16i8 (128-bit)   // QACC_H[127:0]:
                    // v16i8 (128-bit)  // QACC_L[255:128]: v16i8 (128-bit)   //
                    // QACC_L[127:0]: v16i8 (128-bit)  // QACC_H: v32i8
                    // (256-bit)  // QACC_L: v32i8 (256-bit)  //
                    // QACC_H[255:128]: v16i8 (128-bit)   // QACC_H[127:0]:
                    // v16i8 (128-bit)  // QACC_L[255:128]: v16i8 (128-bit)   //
                    // QACC_L[127:0]: v16i8 (128-bit)

// ESP.SRS.S.XACC (_m version) - Shift Right and Saturate Signed from XACC
// Returns: saturated 32-bit signed value
// Mixed model: XACC as {unsigned int low, unsigned int high}
// This wrapper calls the builtin and uses explicit state passing for XACC
// For simplified API, we use 0 as default XACC value (caller should initialize
// XACC before calling)

// ESP.SRS.U.XACC (_m version) - Shift Right and Saturate Unsigned from XACC
// Returns: saturated 32-bit unsigned value
// Mixed model: XACC as {unsigned int low, unsigned int high}
// This wrapper calls the builtin and uses explicit state passing for XACC
// For simplified API, we use 0 as default XACC value (caller should initialize
// XACC before calling)

// ESP.SRC.Q wrapper functions with explicit SAR_BYTES (SAR_BYTES is last
// parameter)

// Bit-cast v8i16 to v16i8 (same 128 bits). Avoids GNU statement-expressions so
// nested uses in one expression do not trip Clang IR generation.
#define esp_vec128_16_to_8(V16) ((esp_vec128_t)(V16))

// All former static inline wrappers have been removed. Call
// __builtin_riscv_esp_* directly; use the esp_*_t / esp_*_res_t typedefs for
// result layouts where applicable.

// ESP.SRCMB builtin declarations - Shift Right and Saturate from QACC
// All SRCMB variants use explicit QACC phantom operands (4x128-bit) for proper
// Data flow tracking SRCMB.S16.Q.QACC - QACC is passed as explicit phantom
// operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_s16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.S16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_s16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.S8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.S8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_qacc_m(esp_vec128_t v0,
                                                 esp_vec128_t v1,
                                                 esp_vec128_t v2,
                                                 esp_vec128_t v3, int Rs1,
                                                 int Sel2);
// SRCMB.U16.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_u16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.U16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_u16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.U8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.U8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_qacc_m(esp_vec128_t v0,
                                                 esp_vec128_t v1,
                                                 esp_vec128_t v2,
                                                 esp_vec128_t v3, int Rs1,
                                                 int Sel2);

// ESP.VMULAS result structure - Multiply-accumulate with QACC_H and QACC_L
// Returns 4x128-bit QACC directly (same as esp_mov_qacc_res_t)
typedef esp_mov_qacc_res_t esp_vmulas_qacc_res_t;

// Helper function to convert esp_mov_qacc_res_t (4x128-bit) to esp_vec512_t
// (512-bit) Used for compatibility with old builtin signatures that expect
// esp_vec512_t

// ESP.VSMULAS result structure - Scalar Multiply-accumulate with QACC_H and
// QACC_L Returns 4x128-bit QACC directly (same as esp_mov_qacc_res_t)
typedef esp_mov_qacc_res_t esp_vsmulas_qacc_res_t;

// ESP.VSMULAS.S16.QACC (_m version) - Scalar Multiply-accumulate to QACC_H and
// QACC_L Returns: structure with QACC as 4x128-bit (v0, v1, v2, v3)

// ESP.VSMULAS.S8.QACC (_m version) - Scalar Multiply-accumulate to QACC_H and
// QACC_L Returns: structure with QACC as 4x128-bit (v0, v1, v2, v3)

// ESP.VSMULAS.U16.QACC (_m version) - Scalar Multiply-accumulate to QACC_H and
// QACC_L Returns: structure with QACC as 4x128-bit (v0, v1, v2, v3)

// ESP.VSMULAS.U8.QACC (_m version) - Scalar Multiply-accumulate to QACC_H and
// QACC_L Returns: structure with QACC as 4x128-bit (v0, v1, v2, v3)

// ESP.VSMULAS.S16.QACC.LD.INCP (_m version) - Similar to VMULAS LD.INCP but
// with sel16 parameter

// ESP.VSMULAS.S8.QACC.LD.INCP (_m version)

// ESP.VSMULAS.U16.QACC.LD.INCP (_m version)

// ESP.VSMULAS.U8.QACC.LD.INCP (_m version)

// ESP.MOVI.8.A.M / ESP.MOVI.16.A.M / ESP.MOVI.32.A.M - Extract element from
// vector to scalar MOVI.A instructions extract elements from QR register
// (128-bit vector) to GPR register Note: SAR parameter is not used by the
// instruction (always 0), but required by intrinsic signature Parameters:
// (vector, index)
// - vector: input vector (v16i8, v8i16, or v4i32)
// - index: element index (i32, sel16 for 8/16-bit, sel4 for 32-bit)
// Returns: extracted element (i32, zero-extended for 8/16-bit, direct for
// 32-bit)

// QACC_H: v32i8 (256-bit)  // QACC_L: v32i8 (256-bit)  // QACC_H[255:128]:
// v16i8 (128-bit)   // QACC_H[127:0]: v16i8 (128-bit)  // QACC_L[255:128]:
// v16i8 (128-bit)   // QACC_L[127:0]: v16i8 (128-bit)

// ESP.SRCMB builtin declarations - Shift Right and Saturate from QACC
// All SRCMB variants use explicit QACC phantom operands (4x128-bit) for proper
// Data flow tracking SRCMB.S16.Q.QACC - QACC is passed as explicit phantom
// operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_s16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.S16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_s16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.S8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.S8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_qacc_m(esp_vec128_t v0,
                                                 esp_vec128_t v1,
                                                 esp_vec128_t v2,
                                                 esp_vec128_t v3, int Rs1,
                                                 int Sel2);
// SRCMB.U16.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_u16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.U16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_u16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.U8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.U8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_qacc_m(esp_vec128_t v0,
                                                 esp_vec128_t v1,
                                                 esp_vec128_t v2,
                                                 esp_vec128_t v3, int Rs1,
                                                 int Sel2);

// ESP.VMULAS result structure - Multiply-accumulate with QACC_H and QACC_L
// Returns 4x128-bit QACC directly (same as esp_mov_qacc_res_t)
typedef esp_mov_qacc_res_t esp_vmulas_qacc_res_t;

// ESP.VSMULAS result structure - Scalar Multiply-accumulate with QACC_H and
// QACC_L Returns 4x128-bit QACC directly (same as esp_mov_qacc_res_t)
typedef esp_mov_qacc_res_t esp_vsmulas_qacc_res_t;

// QACC_H: v32i8 (256-bit)  // QACC_L: v32i8 (256-bit)  // QACC_H[255:128]:
// v16i8 (128-bit)   // QACC_H[127:0]: v16i8 (128-bit)  // QACC_L[255:128]:
// v16i8 (128-bit)   // QACC_L[127:0]: v16i8 (128-bit)

// ESP.SRCMB builtin declarations - Shift Right and Saturate from QACC
// All SRCMB variants use explicit QACC phantom operands (4x128-bit) for proper
// Data flow tracking SRCMB.S16.Q.QACC - QACC is passed as explicit phantom
// operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_s16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.S16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_s16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.S8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.S8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_qacc_m(esp_vec128_t v0,
                                                 esp_vec128_t v1,
                                                 esp_vec128_t v2,
                                                 esp_vec128_t v3, int Rs1,
                                                 int Sel2);
// SRCMB.U16.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_u16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.U16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_u16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.U8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.U8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_qacc_m(esp_vec128_t v0,
                                                 esp_vec128_t v1,
                                                 esp_vec128_t v2,
                                                 esp_vec128_t v3, int Rs1,
                                                 int Sel2);

// ESP.VMULAS result structure - Multiply-accumulate with QACC_H and QACC_L
// Returns 4x128-bit QACC directly (same as esp_mov_qacc_res_t)
typedef esp_mov_qacc_res_t esp_vmulas_qacc_res_t;

// ESP.VSMULAS result structure - Scalar Multiply-accumulate with QACC_H and
// QACC_L Returns 4x128-bit QACC directly (same as esp_mov_qacc_res_t)
typedef esp_mov_qacc_res_t
    esp_vsmulas_qacc_res_t; // QACC_H: v32i8 (256-bit)  // QACC_L: v32i8
                            // (256-bit)  // QACC_H[255:128]: v16i8 (128-bit) //
                            // QACC_H[127:0]: v16i8 (128-bit)  //
                            // QACC_L[255:128]: v16i8 (128-bit)   //
                            // QACC_L[127:0]: v16i8 (128-bit)

// ESP.SRCMB builtin declarations - Shift Right and Saturate from QACC
// All SRCMB variants use explicit QACC phantom operands (4x128-bit) for proper
// Data flow tracking SRCMB.S16.Q.QACC - QACC is passed as explicit phantom
// operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_s16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.S16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_s16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.S8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.S8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_qacc_m(esp_vec128_t v0,
                                                 esp_vec128_t v1,
                                                 esp_vec128_t v2,
                                                 esp_vec128_t v3, int Rs1,
                                                 int Sel2);
// SRCMB.U16.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_u16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.U16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_u16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.U8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.U8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_qacc_m(
    esp_vec128_t v0, esp_vec128_t v1, esp_vec128_t v2, esp_vec128_t v3, int Rs1,
    int Sel2); // QACC_H: v32i8 (256-bit)  // QACC_L: v32i8 (256-bit)  //
               // QACC_H[255:128]: v16i8 (128-bit)   // QACC_H[127:0]: v16i8
               // (128-bit)  // QACC_L[255:128]: v16i8 (128-bit)   //
               // QACC_L[127:0]: v16i8 (128-bit)

// ESP.SRCMB builtin declarations - Shift Right and Saturate from QACC
// All SRCMB variants use explicit QACC phantom operands (4x128-bit) for proper
// Data flow tracking SRCMB.S16.Q.QACC - QACC is passed as explicit phantom
// operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_s16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.S16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_s16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.S8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.S8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_qacc_m(esp_vec128_t v0,
                                                 esp_vec128_t v1,
                                                 esp_vec128_t v2,
                                                 esp_vec128_t v3, int Rs1,
                                                 int Sel2);
// SRCMB.U16.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_u16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.U16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_u16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.U8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.U8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_qacc_m(
    esp_vec128_t v0, esp_vec128_t v1, esp_vec128_t v2, esp_vec128_t v3, int Rs1,
    int Sel2); // QACC_H: v32i8 (256-bit)  // QACC_L: v32i8 (256-bit)  //
               // QACC_H[255:128]: v16i8 (128-bit)   // QACC_H[127:0]: v16i8
               // (128-bit)  // QACC_L[255:128]: v16i8 (128-bit)   //
               // QACC_L[127:0]: v16i8 (128-bit)

// ESP.SRCMB builtin declarations - Shift Right and Saturate from QACC
// All SRCMB variants use explicit QACC phantom operands (4x128-bit) for proper
// Data flow tracking SRCMB.S16.Q.QACC - QACC is passed as explicit phantom
// operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_s16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.S16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_s16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.S8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.S8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_qacc_m(esp_vec128_t v0,
                                                 esp_vec128_t v1,
                                                 esp_vec128_t v2,
                                                 esp_vec128_t v3, int Rs1,
                                                 int Sel2);
// SRCMB.U16.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_u16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.U16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_u16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.U8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.U8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_qacc_m(
    esp_vec128_t v0, esp_vec128_t v1, esp_vec128_t v2, esp_vec128_t v3, int Rs1,
    int Sel2); // QACC_H: v32i8 (256-bit)  // QACC_L: v32i8 (256-bit)  //
               // QACC_H[255:128]: v16i8 (128-bit)   // QACC_H[127:0]: v16i8
               // (128-bit)  // QACC_L[255:128]: v16i8 (128-bit)   //
               // QACC_L[127:0]: v16i8 (128-bit)

// ESP.SRCMB builtin declarations - Shift Right and Saturate from QACC
// All SRCMB variants use explicit QACC phantom operands (4x128-bit) for proper
// Data flow tracking SRCMB.S16.Q.QACC - QACC is passed as explicit phantom
// operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_s16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.S16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_s16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.S8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.S8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_qacc_m(esp_vec128_t v0,
                                                 esp_vec128_t v1,
                                                 esp_vec128_t v2,
                                                 esp_vec128_t v3, int Rs1,
                                                 int Sel2);
// SRCMB.U16.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_u16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.U16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_u16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.U8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.U8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_qacc_m(
    esp_vec128_t v0, esp_vec128_t v1, esp_vec128_t v2, esp_vec128_t v3, int Rs1,
    int Sel2); // QACC_H: v32i8 (256-bit)  // QACC_L: v32i8 (256-bit)  //
               // QACC_H[255:128]: v16i8 (128-bit)   // QACC_H[127:0]: v16i8
               // (128-bit)  // QACC_L[255:128]: v16i8 (128-bit)   //
               // QACC_L[127:0]: v16i8 (128-bit)

// ESP.SRCMB builtin declarations - Shift Right and Saturate from QACC
// All SRCMB variants use explicit QACC phantom operands (4x128-bit) for proper
// Data flow tracking SRCMB.S16.Q.QACC - QACC is passed as explicit phantom
// operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_s16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.S16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_s16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.S8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.S8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_qacc_m(esp_vec128_t v0,
                                                 esp_vec128_t v1,
                                                 esp_vec128_t v2,
                                                 esp_vec128_t v3, int Rs1,
                                                 int Sel2);
// SRCMB.U16.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_u16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.U16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_u16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.U8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.U8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_qacc_m(esp_vec128_t v0,
                                                 esp_vec128_t v1,
                                                 esp_vec128_t v2,
                                                 esp_vec128_t v3, int Rs1,
                                                 int Sel2);

// ESP.VMULAS result structure - Multiply-accumulate with QACC_H and QACC_L
// Returns 4x128-bit QACC directly (same as esp_mov_qacc_res_t)
typedef esp_mov_qacc_res_t esp_vmulas_qacc_res_t;

// ESP.VSMULAS result structure - Scalar Multiply-accumulate with QACC_H and
// QACC_L Returns 4x128-bit QACC directly (same as esp_mov_qacc_res_t)
typedef esp_mov_qacc_res_t
    esp_vsmulas_qacc_res_t; // QACC_H: v32i8 (256-bit)  // QACC_L: v32i8
                            // (256-bit)  // QACC_H[255:128]: v16i8 (128-bit) //
                            // QACC_H[127:0]: v16i8 (128-bit)  //
                            // QACC_L[255:128]: v16i8 (128-bit)   //
                            // QACC_L[127:0]: v16i8 (128-bit)

// ESP.SRCMB builtin declarations - Shift Right and Saturate from QACC
// All SRCMB variants use explicit QACC phantom operands (4x128-bit) for proper
// Data flow tracking SRCMB.S16.Q.QACC - QACC is passed as explicit phantom
// operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_s16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.S16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_s16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.S8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.S8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_qacc_m(esp_vec128_t v0,
                                                 esp_vec128_t v1,
                                                 esp_vec128_t v2,
                                                 esp_vec128_t v3, int Rs1,
                                                 int Sel2);
// SRCMB.U16.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_u16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.U16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_u16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.U8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.U8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_qacc_m(esp_vec128_t v0,
                                                 esp_vec128_t v1,
                                                 esp_vec128_t v2,
                                                 esp_vec128_t v3, int Rs1,
                                                 int Sel2);

// ESP.VMULAS result structure - Multiply-accumulate with QACC_H and QACC_L
// Returns 4x128-bit QACC directly (same as esp_mov_qacc_res_t)
typedef esp_mov_qacc_res_t esp_vmulas_qacc_res_t;

// ESP.VSMULAS result structure - Scalar Multiply-accumulate with QACC_H and
// QACC_L Returns 4x128-bit QACC directly (same as esp_mov_qacc_res_t)
typedef esp_mov_qacc_res_t
    esp_vsmulas_qacc_res_t; // QACC_H: v32i8 (256-bit)  // QACC_L: v32i8
                            // (256-bit)  // QACC_H[255:128]: v16i8 (128-bit) //
                            // QACC_H[127:0]: v16i8 (128-bit)  //
                            // QACC_L[255:128]: v16i8 (128-bit)   //
                            // QACC_L[127:0]: v16i8 (128-bit)

// ESP.SRCMB builtin declarations - Shift Right and Saturate from QACC
// All SRCMB variants use explicit QACC phantom operands (4x128-bit) for proper
// Data flow tracking SRCMB.S16.Q.QACC - QACC is passed as explicit phantom
// operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_s16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.S16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_s16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.S8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.S8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_qacc_m(esp_vec128_t v0,
                                                 esp_vec128_t v1,
                                                 esp_vec128_t v2,
                                                 esp_vec128_t v3, int Rs1,
                                                 int Sel2);
// SRCMB.U16.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_u16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.U16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_u16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.U8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.U8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_qacc_m(esp_vec128_t v0,
                                                 esp_vec128_t v1,
                                                 esp_vec128_t v2,
                                                 esp_vec128_t v3, int Rs1,
                                                 int Sel2);

// ESP.VMULAS result structure - Multiply-accumulate with QACC_H and QACC_L
// Returns 4x128-bit QACC directly (same as esp_mov_qacc_res_t)
typedef esp_mov_qacc_res_t esp_vmulas_qacc_res_t;

// ESP.VSMULAS result structure - Scalar Multiply-accumulate with QACC_H and
// QACC_L Returns 4x128-bit QACC directly (same as esp_mov_qacc_res_t)
typedef esp_mov_qacc_res_t
    esp_vsmulas_qacc_res_t; // QACC_H: v32i8 (256-bit)  // QACC_L: v32i8
                            // (256-bit)  // QACC_H[255:128]: v16i8 (128-bit) //
                            // QACC_H[127:0]: v16i8 (128-bit)  //
                            // QACC_L[255:128]: v16i8 (128-bit)   //
                            // QACC_L[127:0]: v16i8 (128-bit)

// ESP.SRCMB builtin declarations - Shift Right and Saturate from QACC
// All SRCMB variants use explicit QACC phantom operands (4x128-bit) for proper
// Data flow tracking SRCMB.S16.Q.QACC - QACC is passed as explicit phantom
// operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_s16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.S16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_s16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.S8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.S8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_qacc_m(esp_vec128_t v0,
                                                 esp_vec128_t v1,
                                                 esp_vec128_t v2,
                                                 esp_vec128_t v3, int Rs1,
                                                 int Sel2);
// SRCMB.U16.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_u16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.U16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_u16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.U8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.U8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_qacc_m(
    esp_vec128_t v0, esp_vec128_t v1, esp_vec128_t v2, esp_vec128_t v3, int Rs1,
    int Sel2); // QACC_H: v32i8 (256-bit)  // QACC_L: v32i8 (256-bit)  //
               // QACC_H[255:128]: v16i8 (128-bit)   // QACC_H[127:0]: v16i8
               // (128-bit)  // QACC_L[255:128]: v16i8 (128-bit)   //
               // QACC_L[127:0]: v16i8 (128-bit)

// ESP.SRCMB builtin declarations - Shift Right and Saturate from QACC
// All SRCMB variants use explicit QACC phantom operands (4x128-bit) for proper
// Data flow tracking SRCMB.S16.Q.QACC - QACC is passed as explicit phantom
// operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_s16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.S16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_s16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.S8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.S8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_qacc_m(esp_vec128_t v0,
                                                 esp_vec128_t v1,
                                                 esp_vec128_t v2,
                                                 esp_vec128_t v3, int Rs1,
                                                 int Sel2);
// SRCMB.U16.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_u16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.U16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_u16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.U8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.U8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_qacc_m(
    esp_vec128_t v0, esp_vec128_t v1, esp_vec128_t v2, esp_vec128_t v3, int Rs1,
    int Sel2); // QACC_H: v32i8 (256-bit)  // QACC_L: v32i8 (256-bit)  //
               // QACC_H[255:128]: v16i8 (128-bit)   // QACC_H[127:0]: v16i8
               // (128-bit)  // QACC_L[255:128]: v16i8 (128-bit)   //
               // QACC_L[127:0]: v16i8 (128-bit)

// ESP.SRCMB builtin declarations - Shift Right and Saturate from QACC
// All SRCMB variants use explicit QACC phantom operands (4x128-bit) for proper
// Data flow tracking SRCMB.S16.Q.QACC - QACC is passed as explicit phantom
// operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_s16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.S16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_s16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.S8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.S8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_qacc_m(esp_vec128_t v0,
                                                 esp_vec128_t v1,
                                                 esp_vec128_t v2,
                                                 esp_vec128_t v3, int Rs1,
                                                 int Sel2);
// SRCMB.U16.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_u16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.U16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_u16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.U8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.U8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_qacc_m(
    esp_vec128_t v0, esp_vec128_t v1, esp_vec128_t v2, esp_vec128_t v3, int Rs1,
    int Sel2); // QACC_H: v32i8 (256-bit)  // QACC_L: v32i8 (256-bit)  //
               // QACC_H[255:128]: v16i8 (128-bit)   // QACC_H[127:0]: v16i8
               // (128-bit)  // QACC_L[255:128]: v16i8 (128-bit)   //
               // QACC_L[127:0]: v16i8 (128-bit)

// ESP.SRCMB builtin declarations - Shift Right and Saturate from QACC
// All SRCMB variants use explicit QACC phantom operands (4x128-bit) for proper
// Data flow tracking SRCMB.S16.Q.QACC - QACC is passed as explicit phantom
// operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_s16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.S16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_s16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.S8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.S8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_qacc_m(esp_vec128_t v0,
                                                 esp_vec128_t v1,
                                                 esp_vec128_t v2,
                                                 esp_vec128_t v3, int Rs1,
                                                 int Sel2);
// SRCMB.U16.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_u16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.U16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_u16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.U8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.U8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_qacc_m(
    esp_vec128_t v0, esp_vec128_t v1, esp_vec128_t v2, esp_vec128_t v3, int Rs1,
    int Sel2); // QACC_H: v32i8 (256-bit)  // QACC_L: v32i8 (256-bit)  //
               // QACC_H[255:128]: v16i8 (128-bit)   // QACC_H[127:0]: v16i8
               // (128-bit)  // QACC_L[255:128]: v16i8 (128-bit)   //
               // QACC_L[127:0]: v16i8 (128-bit)

// ESP.SRCMB builtin declarations - Shift Right and Saturate from QACC
// All SRCMB variants use explicit QACC phantom operands (4x128-bit) for proper
// Data flow tracking SRCMB.S16.Q.QACC - QACC is passed as explicit phantom
// operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_s16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.S16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_s16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.S8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.S8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_qacc_m(esp_vec128_t v0,
                                                 esp_vec128_t v1,
                                                 esp_vec128_t v2,
                                                 esp_vec128_t v3, int Rs1,
                                                 int Sel2);
// SRCMB.U16.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_u16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.U16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_u16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.U8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.U8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_qacc_m(esp_vec128_t v0,
                                                 esp_vec128_t v1,
                                                 esp_vec128_t v2,
                                                 esp_vec128_t v3, int Rs1,
                                                 int Sel2);

// ESP.VMULAS result structure - Multiply-accumulate with QACC_H and QACC_L
// Returns 4x128-bit QACC directly (same as esp_mov_qacc_res_t)
typedef esp_mov_qacc_res_t esp_vmulas_qacc_res_t;

// ESP.VSMULAS result structure - Scalar Multiply-accumulate with QACC_H and
// QACC_L Returns 4x128-bit QACC directly (same as esp_mov_qacc_res_t)
typedef esp_mov_qacc_res_t
    esp_vsmulas_qacc_res_t; // QACC_H: v32i8 (256-bit)  // QACC_L: v32i8
                            // (256-bit)  // QACC_H[255:128]: v16i8 (128-bit) //
                            // QACC_H[127:0]: v16i8 (128-bit)  //
                            // QACC_L[255:128]: v16i8 (128-bit)   //
                            // QACC_L[127:0]: v16i8 (128-bit)

// ESP.SRCMB builtin declarations - Shift Right and Saturate from QACC
// All SRCMB variants use explicit QACC phantom operands (4x128-bit) for proper
// Data flow tracking SRCMB.S16.Q.QACC - QACC is passed as explicit phantom
// operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_s16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.S16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_s16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.S8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.S8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_qacc_m(esp_vec128_t v0,
                                                 esp_vec128_t v1,
                                                 esp_vec128_t v2,
                                                 esp_vec128_t v3, int Rs1,
                                                 int Sel2);
// SRCMB.U16.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_u16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.U16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_u16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.U8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.U8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_qacc_m(esp_vec128_t v0,
                                                 esp_vec128_t v1,
                                                 esp_vec128_t v2,
                                                 esp_vec128_t v3, int Rs1,
                                                 int Sel2);

// ESP.VMULAS result structure - Multiply-accumulate with QACC_H and QACC_L
// Returns 4x128-bit QACC directly (same as esp_mov_qacc_res_t)
typedef esp_mov_qacc_res_t esp_vmulas_qacc_res_t;

// ESP.VSMULAS result structure - Scalar Multiply-accumulate with QACC_H and
// QACC_L Returns 4x128-bit QACC directly (same as esp_mov_qacc_res_t)
typedef esp_mov_qacc_res_t
    esp_vsmulas_qacc_res_t; // QACC_H: v32i8 (256-bit)  // QACC_L: v32i8
                            // (256-bit)  // QACC_H[255:128]: v16i8 (128-bit) //
                            // QACC_H[127:0]: v16i8 (128-bit)  //
                            // QACC_L[255:128]: v16i8 (128-bit)   //
                            // QACC_L[127:0]: v16i8 (128-bit)

// ESP.SRCMB builtin declarations - Shift Right and Saturate from QACC
// All SRCMB variants use explicit QACC phantom operands (4x128-bit) for proper
// Data flow tracking SRCMB.S16.Q.QACC - QACC is passed as explicit phantom
// operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_s16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.S16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_s16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.S8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.S8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_qacc_m(esp_vec128_t v0,
                                                 esp_vec128_t v1,
                                                 esp_vec128_t v2,
                                                 esp_vec128_t v3, int Rs1,
                                                 int Sel2);
// SRCMB.U16.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_u16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.U16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_u16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.U8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.U8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_qacc_m(esp_vec128_t v0,
                                                 esp_vec128_t v1,
                                                 esp_vec128_t v2,
                                                 esp_vec128_t v3, int Rs1,
                                                 int Sel2);

// ESP.VMULAS result structure - Multiply-accumulate with QACC_H and QACC_L
// Returns 4x128-bit QACC directly (same as esp_mov_qacc_res_t)
typedef esp_mov_qacc_res_t esp_vmulas_qacc_res_t;

// ESP.VSMULAS result structure - Scalar Multiply-accumulate with QACC_H and
// QACC_L Returns 4x128-bit QACC directly (same as esp_mov_qacc_res_t)
typedef esp_mov_qacc_res_t
    esp_vsmulas_qacc_res_t; // QACC_H: v32i8 (256-bit)  // QACC_L: v32i8
                            // (256-bit)  // QACC_H[255:128]: v16i8 (128-bit) //
                            // QACC_H[127:0]: v16i8 (128-bit)  //
                            // QACC_L[255:128]: v16i8 (128-bit)   //
                            // QACC_L[127:0]: v16i8 (128-bit)

// ESP.SRCMB builtin declarations - Shift Right and Saturate from QACC
// All SRCMB variants use explicit QACC phantom operands (4x128-bit) for proper
// Data flow tracking SRCMB.S16.Q.QACC - QACC is passed as explicit phantom
// operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_s16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.S16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_s16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.S8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.S8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_qacc_m(esp_vec128_t v0,
                                                 esp_vec128_t v1,
                                                 esp_vec128_t v2,
                                                 esp_vec128_t v3, int Rs1,
                                                 int Sel2);
// SRCMB.U16.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_u16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.U16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_u16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.U8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.U8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_qacc_m(
    esp_vec128_t v0, esp_vec128_t v1, esp_vec128_t v2, esp_vec128_t v3, int Rs1,
    int Sel2); // QACC_H: v32i8 (256-bit)  // QACC_L: v32i8 (256-bit)  //
               // QACC_H[255:128]: v16i8 (128-bit)   // QACC_H[127:0]: v16i8
               // (128-bit)  // QACC_L[255:128]: v16i8 (128-bit)   //
               // QACC_L[127:0]: v16i8 (128-bit)

// ESP.SRCMB builtin declarations - Shift Right and Saturate from QACC
// All SRCMB variants use explicit QACC phantom operands (4x128-bit) for proper
// Data flow tracking SRCMB.S16.Q.QACC - QACC is passed as explicit phantom
// operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_s16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.S16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_s16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.S8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.S8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_qacc_m(esp_vec128_t v0,
                                                 esp_vec128_t v1,
                                                 esp_vec128_t v2,
                                                 esp_vec128_t v3, int Rs1,
                                                 int Sel2);
// SRCMB.U16.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_u16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.U16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_u16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.U8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.U8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_qacc_m(
    esp_vec128_t v0, esp_vec128_t v1, esp_vec128_t v2, esp_vec128_t v3, int Rs1,
    int Sel2); // QACC_H: v32i8 (256-bit)  // QACC_L: v32i8 (256-bit)  //
               // QACC_H[255:128]: v16i8 (128-bit)   // QACC_H[127:0]: v16i8
               // (128-bit)  // QACC_L[255:128]: v16i8 (128-bit)   //
               // QACC_L[127:0]: v16i8 (128-bit)

// ESP.SRCMB builtin declarations - Shift Right and Saturate from QACC
// All SRCMB variants use explicit QACC phantom operands (4x128-bit) for proper
// Data flow tracking SRCMB.S16.Q.QACC - QACC is passed as explicit phantom
// operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_s16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.S16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_s16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.S8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.S8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_qacc_m(esp_vec128_t v0,
                                                 esp_vec128_t v1,
                                                 esp_vec128_t v2,
                                                 esp_vec128_t v3, int Rs1,
                                                 int Sel2);
// SRCMB.U16.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_u16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.U16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_u16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.U8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.U8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_qacc_m(
    esp_vec128_t v0, esp_vec128_t v1, esp_vec128_t v2, esp_vec128_t v3, int Rs1,
    int Sel2); // QACC_H: v32i8 (256-bit)  // QACC_L: v32i8 (256-bit)  //
               // QACC_H[255:128]: v16i8 (128-bit)   // QACC_H[127:0]: v16i8
               // (128-bit)  // QACC_L[255:128]: v16i8 (128-bit)   //
               // QACC_L[127:0]: v16i8 (128-bit)

// ESP.SRCMB builtin declarations - Shift Right and Saturate from QACC
// All SRCMB variants use explicit QACC phantom operands (4x128-bit) for proper
// Data flow tracking SRCMB.S16.Q.QACC - QACC is passed as explicit phantom
// operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_s16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.S16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_s16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.S8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.S8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_qacc_m(esp_vec128_t v0,
                                                 esp_vec128_t v1,
                                                 esp_vec128_t v2,
                                                 esp_vec128_t v3, int Rs1,
                                                 int Sel2);
// SRCMB.U16.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_u16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.U16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_u16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.U8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.U8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_qacc_m(
    esp_vec128_t v0, esp_vec128_t v1, esp_vec128_t v2, esp_vec128_t v3, int Rs1,
    int Sel2); // QACC_H: v32i8 (256-bit)  // QACC_L: v32i8 (256-bit)  //
               // QACC_H[255:128]: v16i8 (128-bit)   // QACC_H[127:0]: v16i8
               // (128-bit)  // QACC_L[255:128]: v16i8 (128-bit)   //
               // QACC_L[127:0]: v16i8 (128-bit)

// ESP.SRCMB builtin declarations - Shift Right and Saturate from QACC
// All SRCMB variants use explicit QACC phantom operands (4x128-bit) for proper
// Data flow tracking SRCMB.S16.Q.QACC - QACC is passed as explicit phantom
// operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_s16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.S16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_s16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.S8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.S8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_qacc_m(esp_vec128_t v0,
                                                 esp_vec128_t v1,
                                                 esp_vec128_t v2,
                                                 esp_vec128_t v3, int Rs1,
                                                 int Sel2);
// SRCMB.U16.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_u16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.U16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_u16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.U8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.U8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_qacc_m(esp_vec128_t v0,
                                                 esp_vec128_t v1,
                                                 esp_vec128_t v2,
                                                 esp_vec128_t v3, int Rs1,
                                                 int Sel2);

// ESP.VMULAS result structure - Multiply-accumulate with QACC_H and QACC_L
// Returns 4x128-bit QACC directly (same as esp_mov_qacc_res_t)
typedef esp_mov_qacc_res_t esp_vmulas_qacc_res_t;

// ESP.VSMULAS result structure - Scalar Multiply-accumulate with QACC_H and
// QACC_L Returns 4x128-bit QACC directly (same as esp_mov_qacc_res_t)
typedef esp_mov_qacc_res_t
    esp_vsmulas_qacc_res_t; // QACC_H: v32i8 (256-bit)  // QACC_L: v32i8
                            // (256-bit)  // QACC_H[255:128]: v16i8 (128-bit) //
                            // QACC_H[127:0]: v16i8 (128-bit)  //
                            // QACC_L[255:128]: v16i8 (128-bit)   //
                            // QACC_L[127:0]: v16i8 (128-bit)

// ESP.SRCMB builtin declarations - Shift Right and Saturate from QACC
// All SRCMB variants use explicit QACC phantom operands (4x128-bit) for proper
// Data flow tracking SRCMB.S16.Q.QACC - QACC is passed as explicit phantom
// operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_s16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.S16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_s16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.S8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.S8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_qacc_m(esp_vec128_t v0,
                                                 esp_vec128_t v1,
                                                 esp_vec128_t v2,
                                                 esp_vec128_t v3, int Rs1,
                                                 int Sel2);
// SRCMB.U16.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_u16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.U16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_u16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.U8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.U8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_qacc_m(esp_vec128_t v0,
                                                 esp_vec128_t v1,
                                                 esp_vec128_t v2,
                                                 esp_vec128_t v3, int Rs1,
                                                 int Sel2);

// ESP.VMULAS result structure - Multiply-accumulate with QACC_H and QACC_L
// Returns 4x128-bit QACC directly (same as esp_mov_qacc_res_t)
typedef esp_mov_qacc_res_t esp_vmulas_qacc_res_t;

// ESP.VSMULAS result structure - Scalar Multiply-accumulate with QACC_H and
// QACC_L Returns 4x128-bit QACC directly (same as esp_mov_qacc_res_t)
typedef esp_mov_qacc_res_t
    esp_vsmulas_qacc_res_t; // QACC_H: v32i8 (256-bit)  // QACC_L: v32i8
                            // (256-bit)  // QACC_H[255:128]: v16i8 (128-bit) //
                            // QACC_H[127:0]: v16i8 (128-bit)  //
                            // QACC_L[255:128]: v16i8 (128-bit)   //
                            // QACC_L[127:0]: v16i8 (128-bit)

// ESP.SRCMB builtin declarations - Shift Right and Saturate from QACC
// All SRCMB variants use explicit QACC phantom operands (4x128-bit) for proper
// Data flow tracking SRCMB.S16.Q.QACC - QACC is passed as explicit phantom
// operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_s16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.S16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_s16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.S8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.S8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_qacc_m(esp_vec128_t v0,
                                                 esp_vec128_t v1,
                                                 esp_vec128_t v2,
                                                 esp_vec128_t v3, int Rs1,
                                                 int Sel2);
// SRCMB.U16.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_u16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.U16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_u16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.U8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.U8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_qacc_m(esp_vec128_t v0,
                                                 esp_vec128_t v1,
                                                 esp_vec128_t v2,
                                                 esp_vec128_t v3, int Rs1,
                                                 int Sel2);

// ESP.VMULAS result structure - Multiply-accumulate with QACC_H and QACC_L
// Returns 4x128-bit QACC directly (same as esp_mov_qacc_res_t)
typedef esp_mov_qacc_res_t esp_vmulas_qacc_res_t;

// ESP.VSMULAS result structure - Scalar Multiply-accumulate with QACC_H and
// QACC_L Returns 4x128-bit QACC directly (same as esp_mov_qacc_res_t)
typedef esp_mov_qacc_res_t
    esp_vsmulas_qacc_res_t; // QACC_H: v32i8 (256-bit)  // QACC_L: v32i8
                            // (256-bit)  // QACC_H[255:128]: v16i8 (128-bit) //
                            // QACC_H[127:0]: v16i8 (128-bit)  //
                            // QACC_L[255:128]: v16i8 (128-bit)   //
                            // QACC_L[127:0]: v16i8 (128-bit)

// ESP.SRCMB builtin declarations - Shift Right and Saturate from QACC
// All SRCMB variants use explicit QACC phantom operands (4x128-bit) for proper
// Data flow tracking SRCMB.S16.Q.QACC - QACC is passed as explicit phantom
// operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_s16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.S16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_s16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.S8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.S8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_qacc_m(esp_vec128_t v0,
                                                 esp_vec128_t v1,
                                                 esp_vec128_t v2,
                                                 esp_vec128_t v3, int Rs1,
                                                 int Sel2);
// SRCMB.U16.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_u16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.U16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_u16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.U8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.U8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_qacc_m(esp_vec128_t v0,
                                                 esp_vec128_t v1,
                                                 esp_vec128_t v2,
                                                 esp_vec128_t v3, int Rs1,
                                                 int Sel2);

// ESP.VMULAS result structure - Multiply-accumulate with QACC_H and QACC_L
// Returns 4x128-bit QACC directly (same as esp_mov_qacc_res_t)
typedef esp_mov_qacc_res_t esp_vmulas_qacc_res_t;

// ESP.VSMULAS result structure - Scalar Multiply-accumulate with QACC_H and
// QACC_L Returns 4x128-bit QACC directly (same as esp_mov_qacc_res_t)
typedef esp_mov_qacc_res_t esp_vsmulas_qacc_res_t;

// Helper function definitions - always_inline wrappers

// QACC_H: v32i8 (256-bit)  // QACC_L: v32i8 (256-bit)  // QACC_H[255:128]:
// v16i8 (128-bit)   // QACC_H[127:0]: v16i8 (128-bit)  // QACC_L[255:128]:
// v16i8 (128-bit)   // QACC_L[127:0]: v16i8 (128-bit)

// ESP.SRCMB builtin declarations - Shift Right and Saturate from QACC
// All SRCMB variants use explicit QACC phantom operands (4x128-bit) for proper
// Data flow tracking SRCMB.S16.Q.QACC - QACC is passed as explicit phantom
// operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_s16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.S16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_s16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.S8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.S8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_qacc_m(esp_vec128_t v0,
                                                 esp_vec128_t v1,
                                                 esp_vec128_t v2,
                                                 esp_vec128_t v3, int Rs1,
                                                 int Sel2);
// SRCMB.U16.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_u16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.U16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_u16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.U8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.U8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_qacc_m(esp_vec128_t v0,
                                                 esp_vec128_t v1,
                                                 esp_vec128_t v2,
                                                 esp_vec128_t v3, int Rs1,
                                                 int Sel2);

// ESP.VMULAS result structure - Multiply-accumulate with QACC_H and QACC_L
// Returns 4x128-bit QACC directly (same as esp_mov_qacc_res_t)
typedef esp_mov_qacc_res_t esp_vmulas_qacc_res_t;

// ESP.VSMULAS result structure - Scalar Multiply-accumulate with QACC_H and
// QACC_L Returns 4x128-bit QACC directly (same as esp_mov_qacc_res_t)
typedef esp_mov_qacc_res_t esp_vsmulas_qacc_res_t;

// ESP.SRCMB builtin declarations - Shift Right and Saturate from QACC
// All SRCMB variants use explicit QACC phantom operands (4x128-bit) for proper
// Data flow tracking SRCMB.S16.Q.QACC - QACC is passed as explicit phantom
// operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_s16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.S16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_s16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.S8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.S8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_qacc_m(esp_vec128_t v0,
                                                 esp_vec128_t v1,
                                                 esp_vec128_t v2,
                                                 esp_vec128_t v3, int Rs1,
                                                 int Sel2);
// SRCMB.U16.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_u16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.U16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_u16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.U8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.U8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_qacc_m(esp_vec128_t v0,
                                                 esp_vec128_t v1,
                                                 esp_vec128_t v2,
                                                 esp_vec128_t v3, int Rs1,
                                                 int Sel2);

// ESP.VMULAS result structure - Multiply-accumulate with QACC_H and QACC_L
// Returns 4x128-bit QACC directly (same as esp_mov_qacc_res_t)
typedef esp_mov_qacc_res_t esp_vmulas_qacc_res_t;

// ESP.VSMULAS result structure - Scalar Multiply-accumulate with QACC_H and
// QACC_L Returns 4x128-bit QACC directly (same as esp_mov_qacc_res_t)
typedef esp_mov_qacc_res_t
    esp_vsmulas_qacc_res_t; // QACC_H: v32i8 (256-bit)  // QACC_L: v32i8
                            // (256-bit)  // QACC_H[255:128]: v16i8 (128-bit) //
                            // QACC_H[127:0]: v16i8 (128-bit)  //
                            // QACC_L[255:128]: v16i8 (128-bit)   //
                            // QACC_L[127:0]: v16i8 (128-bit)

// ESP.SRCMB builtin declarations - Shift Right and Saturate from QACC
// All SRCMB variants use explicit QACC phantom operands (4x128-bit) for proper
// Data flow tracking SRCMB.S16.Q.QACC - QACC is passed as explicit phantom
// operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_s16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.S16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_s16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.S8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.S8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_qacc_m(esp_vec128_t v0,
                                                 esp_vec128_t v1,
                                                 esp_vec128_t v2,
                                                 esp_vec128_t v3, int Rs1,
                                                 int Sel2);
// SRCMB.U16.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_u16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.U16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_u16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.U8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.U8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_qacc_m(esp_vec128_t v0,
                                                 esp_vec128_t v1,
                                                 esp_vec128_t v2,
                                                 esp_vec128_t v3, int Rs1,
                                                 int Sel2);

// ESP.VMULAS result structure - Multiply-accumulate with QACC_H and QACC_L
// Returns 4x128-bit QACC directly (same as esp_mov_qacc_res_t)
typedef esp_mov_qacc_res_t esp_vmulas_qacc_res_t;

// ESP.VSMULAS result structure - Scalar Multiply-accumulate with QACC_H and
// QACC_L Returns 4x128-bit QACC directly (same as esp_mov_qacc_res_t)
typedef esp_mov_qacc_res_t
    esp_vsmulas_qacc_res_t; // QACC_H: v32i8 (256-bit)  // QACC_L: v32i8
                            // (256-bit)  // QACC_H[255:128]: v16i8 (128-bit) //
                            // QACC_H[127:0]: v16i8 (128-bit)  //
                            // QACC_L[255:128]: v16i8 (128-bit)   //
                            // QACC_L[127:0]: v16i8 (128-bit)

// ESP.SRCMB builtin declarations - Shift Right and Saturate from QACC
// All SRCMB variants use explicit QACC phantom operands (4x128-bit) for proper
// Data flow tracking SRCMB.S16.Q.QACC - QACC is passed as explicit phantom
// operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_s16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.S16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_s16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.S8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.S8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_qacc_m(esp_vec128_t v0,
                                                 esp_vec128_t v1,
                                                 esp_vec128_t v2,
                                                 esp_vec128_t v3, int Rs1,
                                                 int Sel2);
// SRCMB.U16.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_u16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.U16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_u16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.U8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.U8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_qacc_m(esp_vec128_t v0,
                                                 esp_vec128_t v1,
                                                 esp_vec128_t v2,
                                                 esp_vec128_t v3, int Rs1,
                                                 int Sel2);

// ESP.VMULAS result structure - Multiply-accumulate with QACC_H and QACC_L
// Returns 4x128-bit QACC directly (same as esp_mov_qacc_res_t)
typedef esp_mov_qacc_res_t esp_vmulas_qacc_res_t;

// ESP.VSMULAS result structure - Scalar Multiply-accumulate with QACC_H and
// QACC_L Returns 4x128-bit QACC directly (same as esp_mov_qacc_res_t)
typedef esp_mov_qacc_res_t
    esp_vsmulas_qacc_res_t; // QACC_H: v32i8 (256-bit)  // QACC_L: v32i8
                            // (256-bit)  // QACC_H[255:128]: v16i8 (128-bit) //
                            // QACC_H[127:0]: v16i8 (128-bit)  //
                            // QACC_L[255:128]: v16i8 (128-bit)   //
                            // QACC_L[127:0]: v16i8 (128-bit)

// ESP.SRCMB builtin declarations - Shift Right and Saturate from QACC
// All SRCMB variants use explicit QACC phantom operands (4x128-bit) for proper
// Data flow tracking SRCMB.S16.Q.QACC - QACC is passed as explicit phantom
// operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_s16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.S16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_s16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.S8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.S8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_s8_qacc_m(esp_vec128_t v0,
                                                 esp_vec128_t v1,
                                                 esp_vec128_t v2,
                                                 esp_vec128_t v3, int Rs1,
                                                 int Sel2);
// SRCMB.U16.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t
__builtin_riscv_esp_srcmb_u16_q_qacc_m(esp_vec128_t v0, esp_vec128_t v1,
                                       esp_vec128_t v2, esp_vec128_t v3,
                                       esp_vec128_16_t Qw, int Sel2);
// SRCMB.U16.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_16_t __builtin_riscv_esp_srcmb_u16_qacc_m(esp_vec128_t v0,
                                                     esp_vec128_t v1,
                                                     esp_vec128_t v2,
                                                     esp_vec128_t v3, int Rs1,
                                                     int Sel2);
// SRCMB.U8.Q.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_q_qacc_m(esp_vec128_t v0,
                                                   esp_vec128_t v1,
                                                   esp_vec128_t v2,
                                                   esp_vec128_t v3,
                                                   esp_vec128_t Qw, int Sel2);
// SRCMB.U8.QACC - QACC is passed as explicit phantom operand (4x128-bit)
esp_vec128_t __builtin_riscv_esp_srcmb_u8_qacc_m(esp_vec128_t v0,
                                                 esp_vec128_t v1,
                                                 esp_vec128_t v2,
                                                 esp_vec128_t v3, int Rs1,
                                                 int Sel2);

// ESP.VMULAS result structure - Multiply-accumulate with QACC_H and QACC_L
// Returns 4x128-bit QACC directly (same as esp_mov_qacc_res_t)
typedef esp_mov_qacc_res_t esp_vmulas_qacc_res_t;

// ESP.VSMULAS result structure - Scalar Multiply-accumulate with QACC_H and
// QACC_L Returns 4x128-bit QACC directly (same as esp_mov_qacc_res_t)
typedef esp_mov_qacc_res_t
    esp_vsmulas_qacc_res_t; // QACC_H: v32i8 (256-bit)  // QACC_L: v32i8
                            // (256-bit)  // QACC_H[255:128]: v16i8 (128-bit) //
                            // QACC_H[127:0]: v16i8 (128-bit)  //
                            // QACC_L[255:128]: v16i8 (128-bit)   //
                            // QACC_L[127:0]: v16i8 (128-bit)  // QACC_H: v32i8
                            // (256-bit)  // QACC_L: v32i8 (256-bit)  //
                            // QACC_H[255:128]: v16i8 (128-bit)   //
                            // QACC_H[127:0]: v16i8 (128-bit)  //
                            // QACC_L[255:128]: v16i8 (128-bit)   //
                            // QACC_L[127:0]: v16i8 (128-bit)

// Helper function definitions - always_inline wrappers
// VZIP/VUNZIP - Vector Zip/Unzip operations

// VZIPT/VUNZIPT - Vector zip/unzip transpose (3 vectors)

#if defined(__cplusplus)
}
#endif

#endif // __RISCV_ESP32P4_H
