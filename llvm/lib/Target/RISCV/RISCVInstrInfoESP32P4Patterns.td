// Support vector load patterns with zero offset - directly match to ESP_VLD_128_IP with imm=0
// Note: Must match to instruction, not SDNode, because Pat output must be an instruction
class EspVectorLdPatZero<PatFrag LoadOp, ValueType vt>
    : Pat<(vt (LoadOp GPRPIE:$rs1)),
          (ESP_VLD_128_IP GPRPIE:$rs1, 0)>;

// Support vector store patterns with zero offset - directly match to ESP_VST_128_IP with imm=0
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

let Predicates = [HasVendorXespv] in {
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
}