
// Loads/Stores
// Support vector load patterns - handling multiple output instructions
class EspVectorLdPat<PatFrag LoadOp, EspVInst Inst, ValueType vt>
    : Pat<(vt (LoadOp GPRPIE:$rs1)),
          (Inst GPRPIE:$rs1, 0)>;

// Support vector store patterns - handling multiple output instructions
class EspVectorStPat<PatFrag StoreOp, EspVInst Inst, RegisterClass StTy, ValueType vt>
    : Pat<(StoreOp (vt StTy:$qu), GPRPIE:$rs1),
          (Inst StTy:$qu, GPRPIE:$rs1, 0)>;

// Support vector load patterns with stride - handling stride instructions
class EspVectorLdStridePat<PatFrag LoadOp, EspVInst Inst, ValueType vt>
    : Pat<(vt (LoadOp (add GPRPIE:$rs1, offset_256_16:$stride))),
          (Inst GPRPIE:$rs1, offset_256_16:$stride)>;

// Support vector store patterns with stride - handling stride instructions
class EspVectorStStridePat<PatFrag StoreOp, EspVInst Inst, RegisterClass StTy, ValueType vt>
    : Pat<(StoreOp (vt StTy:$qu), (add GPRPIE:$rs1, offset_256_16:$stride)),
          (Inst StTy:$qu, GPRPIE:$rs1, offset_256_16:$stride)>;

// Support vector abs patterns - handling llvm.abs intrinsic to ESP32P4 instructions
class EspVectorAbsPat<ValueType vt, EspVInst Inst>
    : Pat<(vt (abs (vt QR:$qy))),
          (Inst QR:$qy)>;

// Support vector bitwise patterns - handling bitwise SDNode to ESP32P4 instructions
class EspVectorBitwisePat<ValueType vt, SDPatternOperator op, EspVInst Inst>
    : Pat<(vt (op (vt QR:$qx), (vt QR:$qy))),
          (Inst QR:$qx, QR:$qy)>;

// Support vector not patterns - handling vnot to ESP32P4 instructions
class EspVectorNotPat<ValueType vt, EspVInst Inst>
    : Pat<(vt (vnot (vt QR:$qx))),
          (Inst QR:$qx)>;

let usesCustomInserter = 1, Constraints = "$rs1r = $rs1" in 
def ESP_ZERO_VST_128_IP : PseudoEsp<(outs GPRPIE:$rs1r), (ins GPRPIE:$rs1, offset_256_16:$off25616),
                                        "!esp_zero_vst_128_ip $rs1, $off25616", []>;


class EspVectorSaddSatPat<ValueType vt, EspVInst Inst>
    : Pat<(vt (saddsat (vt QR:$qx), (vt QR:$qy))),
          (Inst QR:$qx, QR:$qy)>;

class EspVectorSsubSatPat<ValueType vt, EspVInst Inst>
    : Pat<(vt (ssubsat (vt QR:$qx), (vt QR:$qy))),
          (Inst QR:$qx, QR:$qy)>;

class EspVectorUaddSatPat<ValueType vt, EspVInst Inst>
    : Pat<(vt (uaddsat (vt QR:$qx), (vt QR:$qy))),
          (Inst QR:$qx, QR:$qy)>;

class EspVectorUsubSatPat<ValueType vt, EspVInst Inst>
    : Pat<(vt (usubsat (vt QR:$qx), (vt QR:$qy))),
          (Inst QR:$qx, QR:$qy)>;

// Support vector max patterns - handling smax/umax SDNode to ESP32P4 instructions
class EspVectorSmaxPat<ValueType vt, EspVInst Inst>
    : Pat<(vt (smax (vt QR:$qx), (vt QR:$qy))),
          (Inst QR:$qx, QR:$qy)>;

class EspVectorUmaxPat<ValueType vt, EspVInst Inst>
    : Pat<(vt (umax (vt QR:$qx), (vt QR:$qy))),
          (Inst QR:$qx, QR:$qy)>;

// Support vector min patterns - handling smin/umin SDNode to ESP32P4 instructions
class EspVectorSminPat<ValueType vt, EspVInst Inst>
    : Pat<(vt (smin (vt QR:$qx), (vt QR:$qy))),
          (Inst QR:$qx, QR:$qy)>;

class EspVectorUminPat<ValueType vt, EspVInst Inst>
    : Pat<(vt (umin (vt QR:$qx), (vt QR:$qy))),
          (Inst QR:$qx, QR:$qy)>;

// Support vector reduce max patterns - handling vector_reduce_smax/umax to ESP32P4 instructions
class EspVectorReduceSmaxPat<ValueType vt, ValueType result_vt, EspVInst Inst>
    : Pat<(result_vt (sext (i16 (int_vector_reduce_smax (vt QR:$qw))))),
          (Inst QR:$qw)>;

class EspVectorReduceUmaxPat<ValueType vt, ValueType result_vt, EspVInst Inst>
    : Pat<(result_vt (zext (i16 (int_vector_reduce_umax (vt QR:$qw))))),
          (Inst QR:$qw)>;

// Support vector reduce max patterns - handling vector_reduce_smax/umax to ESP32P4 instructions
class EspVectorReduceSmax32Pat<ValueType vt, EspVInst Inst>
    : Pat<(i32 (int_vector_reduce_smax (vt QR:$qw))),
          (Inst QR:$qw)>;

class EspVectorReduceUmax32Pat<ValueType vt, EspVInst Inst>
    : Pat<(i32 (int_vector_reduce_umax (vt QR:$qw))),
          (Inst QR:$qw)>;

// Support vector reduce min patterns - handling vector_reduce_smin/umin to ESP32P4 instructions
class EspVectorReduceSminPat<ValueType vt, ValueType result_vt, EspVInst Inst>
    : Pat<(result_vt (sext (i16 (int_vector_reduce_smin (vt QR:$qw))))),
          (Inst QR:$qw)>;

class EspVectorReduceUminPat<ValueType vt, ValueType result_vt, EspVInst Inst>
    : Pat<(result_vt (zext (i16 (int_vector_reduce_umin (vt QR:$qw))))),
          (Inst QR:$qw)>;

// Support vector reduce min patterns - handling vector_reduce_smin/umin to ESP32P4 instructions
class EspVectorReduceSmin32Pat<ValueType vt, EspVInst Inst>
    : Pat<(i32 (int_vector_reduce_smin (vt QR:$qw))),
          (Inst QR:$qw)>;

class EspVectorReduceUmin32Pat<ValueType vt, EspVInst Inst>
    : Pat<(i32 (int_vector_reduce_umin (vt QR:$qw))),
          (Inst QR:$qw)>;

multiclass ESP_VCMP_Pattern<string suffix, SDNode cmpNode> {
  def : Pat<(v16i8 (cmpNode v16i8:$qx, v16i8:$qy)),
            (!cast<Instruction>("ESP_VCMP_" # suffix # "_S8") QR:$qx, QR:$qy)>;
  def : Pat<(v8i16 (cmpNode v8i16:$qx, v8i16:$qy)),
            (!cast<Instruction>("ESP_VCMP_" # suffix # "_S16") QR:$qx, QR:$qy)>;
  def : Pat<(v4i32 (cmpNode v4i32:$qx, v4i32:$qy)),
            (!cast<Instruction>("ESP_VCMP_" # suffix # "_S32") QR:$qx, QR:$qy)>;
}

// Support vector compare patterns - equal comparison
def RISCVESP_VCMP_EQ_S8  : SDNode<"RISCVISD::ESP_VCMP_EQ_S8", SDTIntBinOp>;
def RISCVESP_VCMP_EQ_S16 : SDNode<"RISCVISD::ESP_VCMP_EQ_S16", SDTIntBinOp>;

def RISCVESP_VCMP_EQ_S32 : SDNode<"RISCVISD::ESP_VCMP_EQ_S32", SDTIntBinOp>;
def RISCVESP_VCMP_EQ_U8  : SDNode<"RISCVISD::ESP_VCMP_EQ_U8", SDTIntBinOp>;
def RISCVESP_VCMP_EQ_U16 : SDNode<"RISCVISD::ESP_VCMP_EQ_U16", SDTIntBinOp>;
def RISCVESP_VCMP_EQ_U32 : SDNode<"RISCVISD::ESP_VCMP_EQ_U32", SDTIntBinOp>;

// Support vector compare patterns - greater than comparison
def RISCVESP_VCMP_GT_S8  : SDNode<"RISCVISD::ESP_VCMP_GT_S8", SDTIntBinOp>;
def RISCVESP_VCMP_GT_S16 : SDNode<"RISCVISD::ESP_VCMP_GT_S16", SDTIntBinOp>;
def RISCVESP_VCMP_GT_S32 : SDNode<"RISCVISD::ESP_VCMP_GT_S32", SDTIntBinOp>;
def RISCVESP_VCMP_GT_U8  : SDNode<"RISCVISD::ESP_VCMP_GT_U8", SDTIntBinOp>;
def RISCVESP_VCMP_GT_U16 : SDNode<"RISCVISD::ESP_VCMP_GT_U16", SDTIntBinOp>;
def RISCVESP_VCMP_GT_U32 : SDNode<"RISCVISD::ESP_VCMP_GT_U32", SDTIntBinOp>;

// Support vector compare patterns - less than comparison
def RISCVESP_VCMP_LT_S8  : SDNode<"RISCVISD::ESP_VCMP_LT_S8", SDTIntBinOp>;
def RISCVESP_VCMP_LT_S16 : SDNode<"RISCVISD::ESP_VCMP_LT_S16", SDTIntBinOp>;
def RISCVESP_VCMP_LT_S32 : SDNode<"RISCVISD::ESP_VCMP_LT_S32", SDTIntBinOp>;
def RISCVESP_VCMP_LT_U8  : SDNode<"RISCVISD::ESP_VCMP_LT_U8", SDTIntBinOp>;
def RISCVESP_VCMP_LT_U16 : SDNode<"RISCVISD::ESP_VCMP_LT_U16", SDTIntBinOp>;
def RISCVESP_VCMP_LT_U32 : SDNode<"RISCVISD::ESP_VCMP_LT_U32", SDTIntBinOp>;

// Explicit VCMP selection patterns (each SDNode is type-specific; multiclass
// cannot map one SDNode to multiple types, so we add one Pat per (SDNode, Inst)).
// Only enable for ESPV (esp32p4 etc.) so ESP_VCMP_* instructions exist.
let Predicates = [HasVendorXespv1v] in {
def : Pat<(v16i8 (RISCVESP_VCMP_EQ_S8 v16i8:$qx, v16i8:$qy)), (ESP_VCMP_EQ_S8 QR:$qx, QR:$qy)>;
def : Pat<(v8i16 (RISCVESP_VCMP_EQ_S16 v8i16:$qx, v8i16:$qy)), (ESP_VCMP_EQ_S16 QR:$qx, QR:$qy)>;
def : Pat<(v4i32 (RISCVESP_VCMP_EQ_S32 v4i32:$qx, v4i32:$qy)), (ESP_VCMP_EQ_S32 QR:$qx, QR:$qy)>;
def : Pat<(v16i8 (RISCVESP_VCMP_EQ_U8 v16i8:$qx, v16i8:$qy)), (ESP_VCMP_EQ_U8 QR:$qx, QR:$qy)>;
def : Pat<(v8i16 (RISCVESP_VCMP_EQ_U16 v8i16:$qx, v8i16:$qy)), (ESP_VCMP_EQ_U16 QR:$qx, QR:$qy)>;
def : Pat<(v4i32 (RISCVESP_VCMP_EQ_U32 v4i32:$qx, v4i32:$qy)), (ESP_VCMP_EQ_U32 QR:$qx, QR:$qy)>;
def : Pat<(v16i8 (RISCVESP_VCMP_GT_S8 v16i8:$qx, v16i8:$qy)), (ESP_VCMP_GT_S8 QR:$qx, QR:$qy)>;
def : Pat<(v8i16 (RISCVESP_VCMP_GT_S16 v8i16:$qx, v8i16:$qy)), (ESP_VCMP_GT_S16 QR:$qx, QR:$qy)>;
def : Pat<(v4i32 (RISCVESP_VCMP_GT_S32 v4i32:$qx, v4i32:$qy)), (ESP_VCMP_GT_S32 QR:$qx, QR:$qy)>;
def : Pat<(v16i8 (RISCVESP_VCMP_GT_U8 v16i8:$qx, v16i8:$qy)), (ESP_VCMP_GT_U8 QR:$qx, QR:$qy)>;
def : Pat<(v8i16 (RISCVESP_VCMP_GT_U16 v8i16:$qx, v8i16:$qy)), (ESP_VCMP_GT_U16 QR:$qx, QR:$qy)>;
def : Pat<(v4i32 (RISCVESP_VCMP_GT_U32 v4i32:$qx, v4i32:$qy)), (ESP_VCMP_GT_U32 QR:$qx, QR:$qy)>;
def : Pat<(v16i8 (RISCVESP_VCMP_LT_S8 v16i8:$qx, v16i8:$qy)), (ESP_VCMP_LT_S8 QR:$qx, QR:$qy)>;
def : Pat<(v8i16 (RISCVESP_VCMP_LT_S16 v8i16:$qx, v8i16:$qy)), (ESP_VCMP_LT_S16 QR:$qx, QR:$qy)>;
def : Pat<(v4i32 (RISCVESP_VCMP_LT_S32 v4i32:$qx, v4i32:$qy)), (ESP_VCMP_LT_S32 QR:$qx, QR:$qy)>;
def : Pat<(v16i8 (RISCVESP_VCMP_LT_U8 v16i8:$qx, v16i8:$qy)), (ESP_VCMP_LT_U8 QR:$qx, QR:$qy)>;
def : Pat<(v8i16 (RISCVESP_VCMP_LT_U16 v8i16:$qx, v8i16:$qy)), (ESP_VCMP_LT_U16 QR:$qx, QR:$qy)>;
def : Pat<(v4i32 (RISCVESP_VCMP_LT_U32 v4i32:$qx, v4i32:$qy)), (ESP_VCMP_LT_U32 QR:$qx, QR:$qy)>;
}

def extract_high_v8i16 : ComplexPattern<v4i16, 1, "SelectExtractHigh", 
                                       [extract_subvector, bitconvert]>;

// Support vector extract patterns - S16 mode matching
multiclass ESP32P4VectorSExtLongBHS<string asm> {
  // Move usesCustomInserter here
  let mayStore = 0, mayLoad = 0, hasSideEffects = 0 in {
    def v8i16_lo : PseudoEsp<(outs QR:$Rd), (ins QR:$Rn), 
                                 asm # "_lo\t$Rd, $Rn",
                                 []>;
    
    def v8i16_hi : PseudoEsp<(outs QR:$Rd), (ins QR:$Rn),
                                 asm # "_hi\t$Rd, $Rn", 
                                 []>;
    // For S8: v16i8 -> v8i16  
    def v16i8_lo : PseudoEsp<(outs QR:$Rd), (ins QR:$Rn), 
                                 asm # "_lo\t$Rd, $Rn",
                                 []>;
    
    def v16i8_hi : PseudoEsp<(outs QR:$Rd), (ins QR:$Rn),
                                 asm # "_hi\t$Rd, $Rn", 
                                 []>;
  }
}


// Support vector extract patterns - U16 and U8 mode matching
multiclass ESP32P4VectorZExtLongBHS<string asm> {
  let mayStore = 0, mayLoad = 0, hasSideEffects = 0 in {
    // For U16: v8i16 -> v4i32
    def v8i16_lo : PseudoEsp<(outs QR:$Rd), (ins QR:$Rn), 
                                 asm # "_lo\t$Rd, $Rn",
                                 []>;
    
    def v8i16_hi : PseudoEsp<(outs QR:$Rd), (ins QR:$Rn),
                                 asm # "_hi\t$Rd, $Rn", 
                                 []>;
    
    // For U8: v16i8 -> v8i16  
    def v16i8_lo : PseudoEsp<(outs QR:$Rd), (ins QR:$Rn), 
                                 asm # "_lo\t$Rd, $Rn",
                                 []>;
    
    def v16i8_hi : PseudoEsp<(outs QR:$Rd), (ins QR:$Rn),
                                 asm # "_hi\t$Rd, $Rn", 
                                 []>;
  }
}


defm ESP_VEXT_S16 : ESP32P4VectorSExtLongBHS<"esp_vext_s16">;
defm ESP_VEXT_S8 : ESP32P4VectorSExtLongBHS<"esp_vext_s8">;
defm ESP_VEXT_U16 : ESP32P4VectorZExtLongBHS<"esp_vext_u16">;
defm ESP_VEXT_U8 : ESP32P4VectorZExtLongBHS<"esp_vext_u8">;

// Support vector load patterns with register offset - handling XP instructions
class EspVectorLdRegOffsetPat<PatFrag LoadOp, EspVInst Inst, ValueType vt>
    : Pat<(vt (LoadOp (add GPRPIE:$rs1, GPRPIE:$rs2))),
          (Inst GPRPIE:$rs2, GPRPIE:$rs1)>;

// Support vector store patterns with register offset - handling XP instructions
class EspVectorStRegOffsetPat<PatFrag StoreOp, EspVInst Inst, RegisterClass StTy, ValueType vt>
    : Pat<(StoreOp (vt StTy:$qu), (add GPRPIE:$rs1, GPRPIE:$rs2)),
          (Inst GPRPIE:$rs2, StTy:$qu, GPRPIE:$rs1)>;

// Zero-offset vector load/store: lower directly to ESP_VLD_128_IP / ESP_VST_128_IP with imm 0.
class EspVectorLdPatZero<PatFrag LoadOp, ValueType vt>
    : Pat<(vt (LoadOp GPRPIE:$rs1)),
          (ESP_VLD_128_IP GPRPIE:$rs1, 0)>;

class EspVectorStPatZero<PatFrag StoreOp, RegisterClass StTy, ValueType vt>
    : Pat<(StoreOp (vt StTy:$qu), GPRPIE:$rs1),
          (ESP_VST_128_IP StTy:$qu, GPRPIE:$rs1, 0)>;

// Support vector load patterns with zero offset for regular GPR addresses
// This handles cases where addresses are in regular GPR (e.g., function parameters)
// The address will be copied to GPRPIE during instruction selection
class EspVectorLdPatZeroGPR<PatFrag LoadOp, ValueType vt>
    : Pat<(vt (LoadOp GPR:$rs1)),
          (ESP_VLD_128_IP (COPY_TO_REGCLASS GPR:$rs1, GPRPIE), 0)>;

// Support vector store patterns with zero offset for regular GPR addresses
class EspVectorStPatZeroGPR<PatFrag StoreOp, RegisterClass StTy, ValueType vt>
    : Pat<(StoreOp (vt StTy:$qu), GPR:$rs1),
          (ESP_VST_128_IP StTy:$qu, (COPY_TO_REGCLASS GPR:$rs1, GPRPIE), 0)>;

// 64-bit vector (v8i8, v4i16, v2i32) zero-offset load/store -> esp.vld.l.64.ip / esp.vst.l.64.ip
class EspVectorLdPatZero64<PatFrag LoadOp, ValueType vt>
    : Pat<(vt (LoadOp GPRPIE:$rs1)),
          (ESP_VLD_L_64_IP GPRPIE:$rs1, 0)>;

class EspVectorStPatZero64<PatFrag StoreOp, RegisterClass StTy, ValueType vt>
    : Pat<(StoreOp (vt StTy:$qu), GPRPIE:$rs1),
          (ESP_VST_L_64_IP StTy:$qu, GPRPIE:$rs1, 0)>;

class EspVectorLdPatZero64GPR<PatFrag LoadOp, ValueType vt>
    : Pat<(vt (LoadOp GPR:$rs1)),
          (ESP_VLD_L_64_IP (COPY_TO_REGCLASS GPR:$rs1, GPRPIE), 0)>;

class EspVectorStPatZero64GPR<PatFrag StoreOp, RegisterClass StTy, ValueType vt>
    : Pat<(StoreOp (vt StTy:$qu), GPR:$rs1),
          (ESP_VST_L_64_IP StTy:$qu, (COPY_TO_REGCLASS GPR:$rs1, GPRPIE), 0)>;

// 64-bit vector load/store with offset (add rs1, offset_256_8) -> esp.vld.l.64.ip / esp.vst.l.64.ip
class EspVectorLdStridePat64<PatFrag LoadOp, ValueType vt>
    : Pat<(vt (LoadOp (add GPRPIE:$rs1, offset_256_8:$stride))),
          (ESP_VLD_L_64_IP GPRPIE:$rs1, offset_256_8:$stride)>;

class EspVectorStStridePat64<PatFrag StoreOp, RegisterClass StTy, ValueType vt>
    : Pat<(StoreOp (vt StTy:$qu), (add GPRPIE:$rs1, offset_256_8:$stride)),
          (ESP_VST_L_64_IP StTy:$qu, GPRPIE:$rs1, offset_256_8:$stride)>;

// 64-bit vector load/store with register offset -> esp.vld.l.64.xp / esp.vst.l.64.xp
class EspVectorLdRegOffsetPat64<PatFrag LoadOp, ValueType vt>
    : Pat<(vt (LoadOp (add GPRPIE:$rs1, GPRPIE:$rs2))),
          (ESP_VLD_L_64_XP GPRPIE:$rs2, GPRPIE:$rs1)>;

class EspVectorStRegOffsetPat64<PatFrag StoreOp, RegisterClass StTy, ValueType vt>
    : Pat<(StoreOp (vt StTy:$qu), (add GPRPIE:$rs1, GPRPIE:$rs2)),
          (ESP_VST_L_64_XP GPRPIE:$rs2, StTy:$qu, GPRPIE:$rs1)>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 1, isCodeGenOnly = 1 in {
def PseudoESP_VSPILL_128 : Pseudo<(outs), (ins QR:$qu, GPR:$rs1, simm12:$imm12), []>;
}

let hasSideEffects = 0, mayLoad = 1, mayStore = 0, isCodeGenOnly = 1 in {
def PseudoESP_VRELOAD_128 : Pseudo<(outs QR:$qu), (ins GPR:$rs1, simm12:$imm12), []>;
}

def riscv_esp_vunzip_16_uzp1 : SDNode<"RISCVISD::ESP_VUNZIP_16_UZP1", 
                                      SDTypeProfile<1, 2, [SDTCisVT<0, v8i16>, 
                                                           SDTCisSameAs<0, 1>,
                                                           SDTCisSameAs<0, 2>]>>;

def riscv_esp_vunzip_16_uzp2 : SDNode<"RISCVISD::ESP_VUNZIP_16_UZP2", 
                                      SDTypeProfile<1, 2, [SDTCisVT<0, v8i16>, 
                                                           SDTCisSameAs<0, 1>,
                                                           SDTCisSameAs<0, 2>]>>;

// Support vector unzip patterns - 32-bit version
def riscv_esp_vunzip_32_uzp1 : SDNode<"RISCVISD::ESP_VUNZIP_32_UZP1", 
                                      SDTypeProfile<1, 2, [SDTCisVT<0, v4i32>, 
                                                           SDTCisSameAs<0, 1>,
                                                           SDTCisSameAs<0, 2>]>>;

def riscv_esp_vunzip_32_uzp2 : SDNode<"RISCVISD::ESP_VUNZIP_32_UZP2", 
                                      SDTypeProfile<1, 2, [SDTCisVT<0, v4i32>, 
                                                           SDTCisSameAs<0, 1>,
                                                           SDTCisSameAs<0, 2>]>>;

def riscv_esp_vunzip_32_pair : SDNode<"RISCVISD::ESP_VUNZIP_32_PAIR", 
                                     SDTypeProfile<2, 2, [SDTCisVT<0, v4i32>, 
                                                          SDTCisSameAs<0, 1>,
                                                          SDTCisSameAs<0, 2>,
                                                          SDTCisSameAs<0, 3>]>>;

// Support vector unzip patterns - 8-bit version
def riscv_esp_vunzip_8_uzp1 : SDNode<"RISCVISD::ESP_VUNZIP_8_UZP1", 
                                     SDTypeProfile<1, 2, [SDTCisVT<0, v16i8>, 
                                                          SDTCisSameAs<0, 1>,
                                                          SDTCisSameAs<0, 2>]>>;

def riscv_esp_vunzip_8_uzp2 : SDNode<"RISCVISD::ESP_VUNZIP_8_UZP2", 
                                     SDTypeProfile<1, 2, [SDTCisVT<0, v16i8>, 
                                                          SDTCisSameAs<0, 1>,
                                                          SDTCisSameAs<0, 2>]>>;

def riscv_esp_vunzip_8_pair : SDNode<"RISCVISD::ESP_VUNZIP_8_PAIR", 
                                    SDTypeProfile<2, 2, [SDTCisVT<0, v16i8>, 
                                                         SDTCisSameAs<0, 1>,
                                                         SDTCisSameAs<0, 2>,
                                                         SDTCisSameAs<0, 3>]>>;

// Support vector unzip patterns - 16-bit version
def ESP_VUNZIP_16_UZP1 : PseudoEsp<(outs QR:$dst), (ins QR:$src1, QR:$src2),
                         "# ESP_VUNZIP_16_UZP1 $dst, $src1, $src2", 
                         [(set v8i16:$dst, (riscv_esp_vunzip_16_uzp1 v8i16:$src1, v8i16:$src2))]> {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
  
}

def ESP_VUNZIP_16_UZP2 : PseudoEsp<(outs QR:$dst), (ins QR:$src1, QR:$src2),
                         "# ESP_VUNZIP_16_UZP2 $dst, $src1, $src2",
                         [(set v8i16:$dst, (riscv_esp_vunzip_16_uzp2 v8i16:$src1, v8i16:$src2))]> {
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
}

def riscv_esp_vunzip_16_pair : SDNode<"RISCVISD::ESP_VUNZIP_16_PAIR", 
                                     SDTypeProfile<2, 2, [SDTCisVT<0, v8i16>, 
                                                          SDTCisSameAs<0, 1>,
                                                          SDTCisSameAs<0, 2>,
                                                          SDTCisSameAs<0, 3>]>>;

// Support vector unzip patterns - 32-bit version
def ESP_VUNZIP_32_UZP1 : PseudoEsp<(outs QR:$dst), (ins QR:$src1, QR:$src2),
                         "# ESP_VUNZIP_32_UZP1 $dst, $src1, $src2", 
                         [(set v4i32:$dst, (riscv_esp_vunzip_32_uzp1 v4i32:$src1, v4i32:$src2))]> {
  let hasSideEffects = 0;
  let mayLoad = 0;
  let mayStore = 0;
  let usesCustomInserter = 1;
}

def ESP_VUNZIP_32_UZP2 : PseudoEsp<(outs QR:$dst), (ins QR:$src1, QR:$src2),
                         "# ESP_VUNZIP_32_UZP2 $dst, $src1, $src2",
                         [(set v4i32:$dst, (riscv_esp_vunzip_32_uzp2 v4i32:$src1, v4i32:$src2))]> {
  let hasSideEffects = 0;
  let mayLoad = 0;
  let mayStore = 0;
  let usesCustomInserter = 1;
}

// Support vector unzip patterns - 8-bit version
def ESP_VUNZIP_8_UZP1 : PseudoEsp<(outs QR:$dst), (ins QR:$src1, QR:$src2),
                        "# ESP_VUNZIP_8_UZP1 $dst, $src1, $src2", 
                        [(set v16i8:$dst, (riscv_esp_vunzip_8_uzp1 v16i8:$src1, v16i8:$src2))]> {
  let hasSideEffects = 0;
  let mayLoad = 0;
  let mayStore = 0;
  let usesCustomInserter = 1;
}

def ESP_VUNZIP_8_UZP2 : PseudoEsp<(outs QR:$dst), (ins QR:$src1, QR:$src2),
                        "# ESP_VUNZIP_8_UZP2 $dst, $src1, $src2",
                        [(set v16i8:$dst, (riscv_esp_vunzip_8_uzp2 v16i8:$src1, v16i8:$src2))]> {
  let hasSideEffects = 0;
  let mayLoad = 0;
  let mayStore = 0;
  let usesCustomInserter = 1;
}

def : Pat<(riscv_esp_vunzip_16_pair v8i16:$qx, v8i16:$qy),
          (ESP_VUNZIP_16 QR:$qx, QR:$qy)>;

// Support vector unzip patterns - pair matching
def : Pat<(riscv_esp_vunzip_32_pair v4i32:$qx, v4i32:$qy),
          (ESP_VUNZIP_32 QR:$qx, QR:$qy)>;

def : Pat<(riscv_esp_vunzip_8_pair v16i8:$qx, v16i8:$qy),
          (ESP_VUNZIP_8 QR:$qx, QR:$qy)>;

// Support vector zip patterns - 16-bit version
def riscv_esp_vzip_16_zip1 : SDNode<"RISCVISD::ESP_VZIP_16_ZIP1", 
                                    SDTypeProfile<1, 2, [SDTCisVT<0, v8i16>, 
                                                         SDTCisSameAs<0, 1>,
                                                         SDTCisSameAs<0, 2>]>>;

def riscv_esp_vzip_16_zip2 : SDNode<"RISCVISD::ESP_VZIP_16_ZIP2", 
                                    SDTypeProfile<1, 2, [SDTCisVT<0, v8i16>, 
                                                         SDTCisSameAs<0, 1>,
                                                         SDTCisSameAs<0, 2>]>>;

def riscv_esp_vzip_16_pair : SDNode<"RISCVISD::ESP_VZIP_16_PAIR", 
                                   SDTypeProfile<2, 2, [SDTCisVT<0, v8i16>, 
                                                        SDTCisSameAs<0, 1>,
                                                        SDTCisSameAs<0, 2>,
                                                        SDTCisSameAs<0, 3>]>>;

// Support vector zip patterns - 32-bit version
def riscv_esp_vzip_32_zip1 : SDNode<"RISCVISD::ESP_VZIP_32_ZIP1", 
                                    SDTypeProfile<1, 2, [SDTCisVT<0, v4i32>, 
                                                         SDTCisSameAs<0, 1>,
                                                         SDTCisSameAs<0, 2>]>>;

def riscv_esp_vzip_32_zip2 : SDNode<"RISCVISD::ESP_VZIP_32_ZIP2", 
                                    SDTypeProfile<1, 2, [SDTCisVT<0, v4i32>, 
                                                         SDTCisSameAs<0, 1>,
                                                         SDTCisSameAs<0, 2>]>>;

def riscv_esp_vzip_32_pair : SDNode<"RISCVISD::ESP_VZIP_32_PAIR", 
                                   SDTypeProfile<2, 2, [SDTCisVT<0, v4i32>, 
                                                        SDTCisSameAs<0, 1>,
                                                        SDTCisSameAs<0, 2>,
                                                        SDTCisSameAs<0, 3>]>>;

// Support vector zip patterns - 8-bit version
def riscv_esp_vzip_8_zip1 : SDNode<"RISCVISD::ESP_VZIP_8_ZIP1", 
                                   SDTypeProfile<1, 2, [SDTCisVT<0, v16i8>, 
                                                        SDTCisSameAs<0, 1>,
                                                        SDTCisSameAs<0, 2>]>>;

def riscv_esp_vzip_8_zip2 : SDNode<"RISCVISD::ESP_VZIP_8_ZIP2", 
                                   SDTypeProfile<1, 2, [SDTCisVT<0, v16i8>, 
                                                        SDTCisSameAs<0, 1>,
                                                        SDTCisSameAs<0, 2>]>>;

def riscv_esp_vzip_8_pair : SDNode<"RISCVISD::ESP_VZIP_8_PAIR", 
                                  SDTypeProfile<2, 2, [SDTCisVT<0, v16i8>, 
                                                       SDTCisSameAs<0, 1>,
                                                       SDTCisSameAs<0, 2>,
                                                       SDTCisSameAs<0, 3>]>>;

// Support vector zip patterns - 16-bit version
def ESP_VZIP_16_ZIP1 : PseudoEsp<(outs QR:$dst), (ins QR:$src1, QR:$src2),
                       "# ESP_VZIP_16_ZIP1 $dst, $src1, $src2", 
                       [(set v8i16:$dst, (riscv_esp_vzip_16_zip1 v8i16:$src1, v8i16:$src2))]> {
  let hasSideEffects = 0;
  let mayLoad = 0;
  let mayStore = 0;
  let usesCustomInserter = 1;
}

def ESP_VZIP_16_ZIP2 : PseudoEsp<(outs QR:$dst), (ins QR:$src1, QR:$src2),
                       "# ESP_VZIP_16_ZIP2 $dst, $src1, $src2",
                       [(set v8i16:$dst, (riscv_esp_vzip_16_zip2 v8i16:$src1, v8i16:$src2))]> {
  let hasSideEffects = 0;
  let mayLoad = 0;
  let mayStore = 0;
  let usesCustomInserter = 1;
}

def ESP_VZIP_32_ZIP1 : PseudoEsp<(outs QR:$dst), (ins QR:$src1, QR:$src2),
                       "# ESP_VZIP_32_ZIP1 $dst, $src1, $src2", 
                       [(set v4i32:$dst, (riscv_esp_vzip_32_zip1 v4i32:$src1, v4i32:$src2))]> {
  let hasSideEffects = 0;
  let mayLoad = 0;
  let mayStore = 0;
  let usesCustomInserter = 1;
}

def ESP_VZIP_32_ZIP2 : PseudoEsp<(outs QR:$dst), (ins QR:$src1, QR:$src2),
                       "# ESP_VZIP_32_ZIP2 $dst, $src1, $src2",
                       [(set v4i32:$dst, (riscv_esp_vzip_32_zip2 v4i32:$src1, v4i32:$src2))]> {
  let hasSideEffects = 0;
  let mayLoad = 0;
  let mayStore = 0;
  let usesCustomInserter = 1;
}

def ESP_VZIP_8_ZIP1 : PseudoEsp<(outs QR:$dst), (ins QR:$src1, QR:$src2),
                      "# ESP_VZIP_8_ZIP1 $dst, $src1, $src2", 
                      [(set v16i8:$dst, (riscv_esp_vzip_8_zip1 v16i8:$src1, v16i8:$src2))]> {
  let hasSideEffects = 0;
  let mayLoad = 0;
  let mayStore = 0;
  let usesCustomInserter = 1;
}

def ESP_VZIP_8_ZIP2 : PseudoEsp<(outs QR:$dst), (ins QR:$src1, QR:$src2),
                      "# ESP_VZIP_8_ZIP2 $dst, $src1, $src2",
                      [(set v16i8:$dst, (riscv_esp_vzip_8_zip2 v16i8:$src1, v16i8:$src2))]> {
  let hasSideEffects = 0;
  let mayLoad = 0;
  let mayStore = 0;
  let usesCustomInserter = 1;
}

// Support vector zip patterns - pair matching
def : Pat<(riscv_esp_vzip_16_pair v8i16:$qx, v8i16:$qy),
          (ESP_VZIP_16 QR:$qx, QR:$qy)>;

def : Pat<(riscv_esp_vzip_32_pair v4i32:$qx, v4i32:$qy),
          (ESP_VZIP_32 QR:$qx, QR:$qy)>;

def : Pat<(riscv_esp_vzip_8_pair v16i8:$qx, v16i8:$qy),
          (ESP_VZIP_8 QR:$qx, QR:$qy)>;

let Predicates = [HasVendorXespv1v] in {
  // Use patterns for GPRPIE addresses (preferred, no copy needed)
  def : EspVectorLdPatZero<load, v16i8>;
  def : EspVectorLdPatZero<load, v8i16>;
  def : EspVectorLdPatZero<load, v4i32>;
  
  def : EspVectorStPatZero<store, QR, v16i8>;
  def : EspVectorStPatZero<store, QR, v8i16>;
  def : EspVectorStPatZero<store, QR, v4i32>;
  
  // Use patterns for regular GPR addresses (fallback, requires copy to GPRPIE)
  def : EspVectorLdPatZeroGPR<load, v16i8>;
  def : EspVectorLdPatZeroGPR<load, v8i16>;
  def : EspVectorLdPatZeroGPR<load, v4i32>;
  
  def : EspVectorStPatZeroGPR<store, QR, v16i8>;
  def : EspVectorStPatZeroGPR<store, QR, v8i16>;
  def : EspVectorStPatZeroGPR<store, QR, v4i32>;

  // 64-bit vector load/store (esp.vld.l.64.ip / esp.vst.l.64.ip) - zero offset
  def : EspVectorLdPatZero64<load, v8i8>;
  def : EspVectorLdPatZero64<load, v4i16>;
  def : EspVectorLdPatZero64<load, v2i32>;
  def : EspVectorStPatZero64<store, QR_64, v8i8>;
  def : EspVectorStPatZero64<store, QR_64, v4i16>;
  def : EspVectorStPatZero64<store, QR_64, v2i32>;
  def : EspVectorLdPatZero64GPR<load, v8i8>;
  def : EspVectorLdPatZero64GPR<load, v4i16>;
  def : EspVectorLdPatZero64GPR<load, v2i32>;
  def : EspVectorStPatZero64GPR<store, QR_64, v8i8>;
  def : EspVectorStPatZero64GPR<store, QR_64, v4i16>;
  def : EspVectorStPatZero64GPR<store, QR_64, v2i32>;
}

let Predicates = [HasVendorXespv1v] in {
  // Support vector load patterns with stride - unified handling of add and getelementptr
  def : EspVectorLdStridePat<load, ESP_VLD_128_IP, v16i8>;
  def : EspVectorLdStridePat<load, ESP_VLD_128_IP, v8i16>;
  def : EspVectorLdStridePat<load, ESP_VLD_128_IP, v4i32>;

  // Support vector store patterns with stride - unified handling of add and getelementptr
  def : EspVectorStStridePat<store, ESP_VST_128_IP, QR, v16i8>;
  def : EspVectorStStridePat<store, ESP_VST_128_IP, QR, v8i16>;
  def : EspVectorStStridePat<store, ESP_VST_128_IP, QR, v4i32>;

  // Support vector load patterns with register offset - unified handling of add and getelementptr
  def : EspVectorLdRegOffsetPat<load, ESP_VLD_128_XP, v16i8>;
  def : EspVectorLdRegOffsetPat<load, ESP_VLD_128_XP, v8i16>;
  def : EspVectorLdRegOffsetPat<load, ESP_VLD_128_XP, v4i32>;

  // Support vector store patterns with register offset - unified handling of add and getelementptr
  def : EspVectorStRegOffsetPat<store, ESP_VST_128_XP, QR, v16i8>;
  def : EspVectorStRegOffsetPat<store, ESP_VST_128_XP, QR, v8i16>;
  def : EspVectorStRegOffsetPat<store, ESP_VST_128_XP, QR, v4i32>;

  // 64-bit vector load/store with stride (add rs1, offset_256_8) -> esp.vld.l.64.ip / esp.vst.l.64.ip
  def : EspVectorLdStridePat64<load, v8i8>;
  def : EspVectorLdStridePat64<load, v4i16>;
  def : EspVectorLdStridePat64<load, v2i32>;
  def : EspVectorStStridePat64<store, QR_64, v8i8>;
  def : EspVectorStStridePat64<store, QR_64, v4i16>;
  def : EspVectorStStridePat64<store, QR_64, v2i32>;

  // 64-bit vector load/store with register offset -> esp.vld.l.64.xp / esp.vst.l.64.xp
  def : EspVectorLdRegOffsetPat64<load, v8i8>;
  def : EspVectorLdRegOffsetPat64<load, v4i16>;
  def : EspVectorLdRegOffsetPat64<load, v2i32>;
  def : EspVectorStRegOffsetPat64<store, QR_64, v8i8>;
  def : EspVectorStRegOffsetPat64<store, QR_64, v4i16>;
  def : EspVectorStRegOffsetPat64<store, QR_64, v2i32>;

   // Support vector abs patterns
  def : EspVectorAbsPat<v16i8, ESP_VABS_8>;
  def : EspVectorAbsPat<v8i16, ESP_VABS_16>;
  def : EspVectorAbsPat<v4i32, ESP_VABS_32>;

  // Support vector bitwise patterns - all vector types use the same 128-bit bitwise instructions
  def : EspVectorBitwisePat<v16i8, or, ESP_ORQ>;
  def : EspVectorBitwisePat<v8i16, or, ESP_ORQ>;
  def : EspVectorBitwisePat<v4i32, or, ESP_ORQ>;

  def : EspVectorBitwisePat<v16i8, and, ESP_ANDQ>;
  def : EspVectorBitwisePat<v8i16, and, ESP_ANDQ>;
  def : EspVectorBitwisePat<v4i32, and, ESP_ANDQ>;

  def : EspVectorBitwisePat<v16i8, xor, ESP_XORQ>;
  def : EspVectorBitwisePat<v8i16, xor, ESP_XORQ>;
  def : EspVectorBitwisePat<v4i32, xor, ESP_XORQ>;

  // Support vector not patterns - use vnot PatFrag
  def : EspVectorNotPat<v16i8, ESP_NOTQ>;
  def : EspVectorNotPat<v8i16, ESP_NOTQ>;
  def : EspVectorNotPat<v4i32, ESP_NOTQ>;

  def : Pat<(store (v4i32 immAllZerosV), GPRPIE:$rs1),
            (ESP_ZERO_VST_128_IP GPRPIE:$rs1, 0)>;
  
  def : Pat<(store (v8i16 immAllZerosV), GPRPIE:$rs1),
            (ESP_ZERO_VST_128_IP GPRPIE:$rs1, 0)>;
  
  def : Pat<(store (v16i8 immAllZerosV), GPRPIE:$rs1),
            (ESP_ZERO_VST_128_IP GPRPIE:$rs1, 0)>;
  // Support vector store patterns with zero offset - unified handling of add and getelementptr
  def : Pat<(store (v4i32 immAllZerosV), (add GPRPIE:$rs1, offset_256_16:$offset)),
            (ESP_ZERO_VST_128_IP GPRPIE:$rs1, offset_256_16:$offset)>;

  def : Pat<(store (v8i16 immAllZerosV), (add GPRPIE:$rs1, offset_256_16:$offset)),
            (ESP_ZERO_VST_128_IP GPRPIE:$rs1, offset_256_16:$offset)>;

  def : Pat<(store (v16i8 immAllZerosV), (add GPRPIE:$rs1, offset_256_16:$offset)),
            (ESP_ZERO_VST_128_IP GPRPIE:$rs1, offset_256_16:$offset)>;

  def : EspVectorSaddSatPat<v16i8, ESP_VADD_S8>;
  def : EspVectorSaddSatPat<v8i16, ESP_VADD_S16>;  
  def : EspVectorSaddSatPat<v4i32, ESP_VADD_S32>;

  def : EspVectorSsubSatPat<v16i8, ESP_VSUB_S8>;
  def : EspVectorSsubSatPat<v8i16, ESP_VSUB_S16>;  
  def : EspVectorSsubSatPat<v4i32, ESP_VSUB_S32>;

  def : EspVectorUaddSatPat<v16i8, ESP_VADD_U8>;
  def : EspVectorUaddSatPat<v8i16, ESP_VADD_U16>;  
  def : EspVectorUaddSatPat<v4i32, ESP_VADD_U32>;

  def : EspVectorUsubSatPat<v16i8, ESP_VSUB_U8>;
  def : EspVectorUsubSatPat<v8i16, ESP_VSUB_U16>;  
  def : EspVectorUsubSatPat<v4i32, ESP_VSUB_U32>;

  // Support vector signed max patterns
  def : EspVectorSmaxPat<v16i8, ESP_VMAX_S8>;
  def : EspVectorSmaxPat<v8i16, ESP_VMAX_S16>;
  def : EspVectorSmaxPat<v4i32, ESP_VMAX_S32>;

  // Support vector unsigned max patterns
  def : EspVectorUmaxPat<v16i8, ESP_VMAX_U8>;
  def : EspVectorUmaxPat<v8i16, ESP_VMAX_U16>;
  def : EspVectorUmaxPat<v4i32, ESP_VMAX_U32>;

  // Support vector signed min patterns
  def : EspVectorSminPat<v16i8, ESP_VMIN_S8>;
  def : EspVectorSminPat<v8i16, ESP_VMIN_S16>;
  def : EspVectorSminPat<v4i32, ESP_VMIN_S32>;

  // Support vector unsigned min patterns
  def : EspVectorUminPat<v16i8, ESP_VMIN_U8>;
  def : EspVectorUminPat<v8i16, ESP_VMIN_U16>;
  def : EspVectorUminPat<v4i32, ESP_VMIN_U32>;

  // Support vector reduce max patterns - note that the return type is i32 (matches hardware behavior)
  def : EspVectorReduceSmaxDirectPat<v16i8, i32, ESP_MAX_S8_A>;
  def : EspVectorReduceSmaxDirectPat<v8i16, i32, ESP_MAX_S16_A>;
  def : EspVectorReduceSmaxDirectPat<v4i32, i32, ESP_MAX_S32_A>;
  
  def : EspVectorReduceUmaxDirectPat<v16i8, i32, ESP_MAX_U8_A>;
  def : EspVectorReduceUmaxDirectPat<v8i16, i32, ESP_MAX_U16_A>;
  def : EspVectorReduceUmaxDirectPat<v4i32, i32, ESP_MAX_U32_A>;
  
  // Support vector reduce min patterns - note that the return type is i32 (matches hardware behavior)
  def : EspVectorReduceSminDirectPat<v16i8, i32, ESP_MIN_S8_A>;
  def : EspVectorReduceSminDirectPat<v8i16, i32, ESP_MIN_S16_A>;
  def : EspVectorReduceSminDirectPat<v4i32, i32, ESP_MIN_S32_A>;
  
  def : EspVectorReduceUminDirectPat<v16i8, i32, ESP_MIN_U8_A>;
  def : EspVectorReduceUminDirectPat<v8i16, i32, ESP_MIN_U16_A>;
  def : EspVectorReduceUminDirectPat<v4i32, i32, ESP_MIN_U32_A>;

  // GPR ALU *_m only (PIE/AR = GPRPIE). Non-_m intrinsics keep pseudo + custom inserter (unchanged in InstrInfo).
  def : Pat<(i32 (int_riscv_esp_addx2_m GPRPIE:$rs1, GPRPIE:$rs2)),
            (ESP_ADDX2 GPRPIE:$rs1, GPRPIE:$rs2)>;
  def : Pat<(i32 (int_riscv_esp_addx4_m GPRPIE:$rs1, GPRPIE:$rs2)),
            (ESP_ADDX4 GPRPIE:$rs1, GPRPIE:$rs2)>;
  def : Pat<(i32 (int_riscv_esp_subx2_m GPRPIE:$rs1, GPRPIE:$rs2)),
            (ESP_SUBX2 GPRPIE:$rs1, GPRPIE:$rs2)>;
  def : Pat<(i32 (int_riscv_esp_subx4_m GPRPIE:$rs1, GPRPIE:$rs2)),
            (ESP_SUBX4 GPRPIE:$rs1, GPRPIE:$rs2)>;
  // No (set ...) on rhs: GlobalISel Pat rejects (set dst, (inst...)); single-def inst like ADDX2.
  def : Pat<(i32 (int_riscv_esp_sat_m GPRPIE:$rs0, GPRPIE:$rs1, GPRPIE:$rsd)),
            (ESP_SAT GPRPIE:$rs0, GPRPIE:$rs1, GPRPIE:$rsd)>;
}