let TargetPrefix = "riscv" in {


def int_riscv_esp_cmul_u16: ClangBuiltin<"__builtin_riscv_esp_cmul_u16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_esp_cmul_u16_ld_incp: ClangBuiltin<"__builtin_riscv_esp_cmul_u16_ld_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>]>;

def int_riscv_esp_cmul_u16_st_incp: ClangBuiltin<"__builtin_riscv_esp_cmul_u16_st_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>]>;

def int_riscv_esp_cmul_u8: ClangBuiltin<"__builtin_riscv_esp_cmul_u8">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_esp_cmul_u8_ld_incp: ClangBuiltin<"__builtin_riscv_esp_cmul_u8_ld_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>]>;

def int_riscv_esp_cmul_u8_st_incp: ClangBuiltin<"__builtin_riscv_esp_cmul_u8_st_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>]>;















































































































def int_riscv_esp_addx2: ClangBuiltin<"__builtin_riscv_esp_addx2">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], []>;

def int_riscv_esp_addx4: ClangBuiltin<"__builtin_riscv_esp_addx4">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], []>;

def int_riscv_esp_sat: ClangBuiltin<"__builtin_riscv_esp_sat">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], []>;

def int_riscv_esp_subx2: ClangBuiltin<"__builtin_riscv_esp_subx2">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], []>;

def int_riscv_esp_subx4: ClangBuiltin<"__builtin_riscv_esp_subx4">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], []>;

def int_riscv_esp_vcmp_eq_u16: ClangBuiltin<"__builtin_riscv_esp_vcmp_eq_u16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vcmp_eq_u32: ClangBuiltin<"__builtin_riscv_esp_vcmp_eq_u32">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vcmp_eq_u8: ClangBuiltin<"__builtin_riscv_esp_vcmp_eq_u8">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vcmp_gt_u16: ClangBuiltin<"__builtin_riscv_esp_vcmp_gt_u16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vcmp_gt_u32: ClangBuiltin<"__builtin_riscv_esp_vcmp_gt_u32">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vcmp_gt_u8: ClangBuiltin<"__builtin_riscv_esp_vcmp_gt_u8">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

// MOVI - Extract from vector to scalar (_m variants for machine instruction matching)
// MOVI.16.A - Extract 16-bit from vector to scalar
def int_riscv_esp_movi_16_a : Intrinsic<[llvm_i32_ty],
                                          [llvm_v8i16_ty, llvm_i32_ty],  // (vector, index)
                                          [IntrNoMem]>;
// MOVI.32.A - Extract 32-bit from vector to scalar
def int_riscv_esp_movi_32_a : Intrinsic<[llvm_i32_ty],
                                          [llvm_v4i32_ty, llvm_i32_ty],  // (vector, index)
                                          [IntrNoMem]>;
// MOVI.8.A - Extract 8-bit from vector to scalar
def int_riscv_esp_movi_8_a : Intrinsic<[llvm_i32_ty],
                                         [llvm_v16i8_ty, llvm_i32_ty],  // (vector, index)
                                         [IntrNoMem]>;

// MOVI - Insert from scalar to vector (_m variants for machine instruction matching)
// MOVI.16.Q - Insert 16-bit from scalar to vector
def int_riscv_esp_movi_16_q : Intrinsic<[llvm_v8i16_ty],
                                          [llvm_v8i16_ty, llvm_i32_ty, llvm_i32_ty],  // (vector, index, value)
                                          [IntrNoMem]>;
// MOVI.32.Q - Insert 32-bit from scalar to vector
def int_riscv_esp_movi_32_q : Intrinsic<[llvm_v4i32_ty],
                                          [llvm_v4i32_ty, llvm_i32_ty, llvm_i32_ty],  // (vector, index, value)
                                          [IntrNoMem]>;
// MOVI.8.Q - Insert 8-bit from scalar to vector
def int_riscv_esp_movi_8_q : Intrinsic<[llvm_v16i8_ty],
                                         [llvm_v16i8_ty, llvm_i32_ty, llvm_i32_ty],  // (vector, index, value)
                                         [IntrNoMem]>;

def int_riscv_esp_vldbc_16_ip: ClangBuiltin<"__builtin_riscv_esp_vldbc_16_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vldbc_16_xp: ClangBuiltin<"__builtin_riscv_esp_vldbc_16_xp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vldbc_32_ip: ClangBuiltin<"__builtin_riscv_esp_vldbc_32_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vldbc_32_xp: ClangBuiltin<"__builtin_riscv_esp_vldbc_32_xp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vldbc_8_ip: ClangBuiltin<"__builtin_riscv_esp_vldbc_8_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vldbc_8_xp: ClangBuiltin<"__builtin_riscv_esp_vldbc_8_xp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vldext_s16_ip: ClangBuiltin<"__builtin_riscv_esp_vldext_s16_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_esp_vldext_s16_xp: ClangBuiltin<"__builtin_riscv_esp_vldext_s16_xp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_esp_vldext_s8_ip: ClangBuiltin<"__builtin_riscv_esp_vldext_s8_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_esp_vldext_s8_xp: ClangBuiltin<"__builtin_riscv_esp_vldext_s8_xp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_esp_vldext_u16_ip: ClangBuiltin<"__builtin_riscv_esp_vldext_u16_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_esp_vldext_u16_xp: ClangBuiltin<"__builtin_riscv_esp_vldext_u16_xp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_esp_vldext_u8_ip: ClangBuiltin<"__builtin_riscv_esp_vldext_u8_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_esp_vldext_u8_xp: ClangBuiltin<"__builtin_riscv_esp_vldext_u8_xp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_esp_vldhbc_16_incp: ClangBuiltin<"__builtin_riscv_esp_vldhbc_16_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

// Legacy Index-form intrinsics retained for RISCVEsp32P4MemIntrin (no Clang builtin).
// Canonical ESPVM forms live in IntrinsicsRISCVESPVM.td under the non-_pie names.
def int_riscv_esp_src_q_pie :
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_src_q_ld_ip_pie :
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

// Legacy Index-form USAR for RISCVEsp32P4MemIntrin (no Clang builtin).
def int_riscv_esp_ld_128_usar_ip_pie :
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

// Legacy Index-form VLD for RISCVEsp32P4MemIntrin (no Clang builtin).
def int_riscv_esp_vld_128_ip_pie :
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vld_h_64_ip_pie :
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vld_l_64_ip_pie :
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

// Legacy Index-form VST for RISCVEsp32P4MemIntrin (no Clang builtin).
def int_riscv_esp_vst_128_ip_pie :
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vst_h_64_ip_pie :
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vst_l_64_ip_pie :
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>]>;


























































def int_riscv_esp_vsmulas_s16_qacc_2p2: ClangBuiltin<"__builtin_riscv_esp_vsmulas_s16_qacc_2p2">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_esp_vsmulas_s16_qacc_ld_incp_2p2: ClangBuiltin<"__builtin_riscv_esp_vsmulas_s16_qacc_ld_incp_2p2">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>]>;

def int_riscv_esp_vsmulas_s8_qacc_2p2: ClangBuiltin<"__builtin_riscv_esp_vsmulas_s8_qacc_2p2">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_esp_vsmulas_s8_qacc_ld_incp_2p2: ClangBuiltin<"__builtin_riscv_esp_vsmulas_s8_qacc_ld_incp_2p2">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>]>;

def int_riscv_esp_vsmulas_u16_qacc_2p2: ClangBuiltin<"__builtin_riscv_esp_vsmulas_u16_qacc_2p2">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_esp_vsmulas_u16_qacc_ld_incp_2p2: ClangBuiltin<"__builtin_riscv_esp_vsmulas_u16_qacc_ld_incp_2p2">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>]>;

def int_riscv_esp_vsmulas_u8_qacc_2p2: ClangBuiltin<"__builtin_riscv_esp_vsmulas_u8_qacc_2p2">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_esp_vsmulas_u8_qacc_ld_incp_2p2: ClangBuiltin<"__builtin_riscv_esp_vsmulas_u8_qacc_ld_incp_2p2">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>]>;

def int_riscv_esp_cmul_u16_2p2: ClangBuiltin<"__builtin_riscv_esp_cmul_u16_2p2">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>]>;

def int_riscv_esp_cmul_u16_ld_incp_2p2: ClangBuiltin<"__builtin_riscv_esp_cmul_u16_ld_incp_2p2">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>, ImmArg<ArgIndex<7>>]>;

def int_riscv_esp_cmul_u16_st_incp_2p2: ClangBuiltin<"__builtin_riscv_esp_cmul_u16_st_incp_2p2">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>, ImmArg<ArgIndex<7>>]>;

def int_riscv_esp_cmul_u8_2p2: ClangBuiltin<"__builtin_riscv_esp_cmul_u8_2p2">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>]>;

def int_riscv_esp_cmul_u8_ld_incp_2p2: ClangBuiltin<"__builtin_riscv_esp_cmul_u8_ld_incp_2p2">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>, ImmArg<ArgIndex<7>>]>;

def int_riscv_esp_cmul_u8_st_incp_2p2: ClangBuiltin<"__builtin_riscv_esp_cmul_u8_st_incp_2p2">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>, ImmArg<ArgIndex<7>>]>;















































































































def int_riscv_esp_addx2_2p2: ClangBuiltin<"__builtin_riscv_esp_addx2_2p2">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], []>;

def int_riscv_esp_addx4_2p2: ClangBuiltin<"__builtin_riscv_esp_addx4_2p2">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], []>;

def int_riscv_esp_sat_2p2: ClangBuiltin<"__builtin_riscv_esp_sat_2p2">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], []>;

def int_riscv_esp_subx2_2p2: ClangBuiltin<"__builtin_riscv_esp_subx2_2p2">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], []>;

def int_riscv_esp_subx4_2p2: ClangBuiltin<"__builtin_riscv_esp_subx4_2p2">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], []>;










def int_riscv_esp_vsl_u32_2p2: ClangBuiltin<"__builtin_riscv_esp_vsl_u32_2p2">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vldbc_16_ip_2p2: ClangBuiltin<"__builtin_riscv_esp_vldbc_16_ip_2p2">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vldbc_16_xp_2p2: ClangBuiltin<"__builtin_riscv_esp_vldbc_16_xp_2p2">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vldbc_32_ip_2p2: ClangBuiltin<"__builtin_riscv_esp_vldbc_32_ip_2p2">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vldbc_32_xp_2p2: ClangBuiltin<"__builtin_riscv_esp_vldbc_32_xp_2p2">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vldbc_8_ip_2p2: ClangBuiltin<"__builtin_riscv_esp_vldbc_8_ip_2p2">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vldbc_8_xp_2p2: ClangBuiltin<"__builtin_riscv_esp_vldbc_8_xp_2p2">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vldext_s16_ip_2p2: ClangBuiltin<"__builtin_riscv_esp_vldext_s16_ip_2p2">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_esp_vldext_s16_xp_2p2: ClangBuiltin<"__builtin_riscv_esp_vldext_s16_xp_2p2">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_esp_vldext_s8_ip_2p2: ClangBuiltin<"__builtin_riscv_esp_vldext_s8_ip_2p2">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_esp_vldext_s8_xp_2p2: ClangBuiltin<"__builtin_riscv_esp_vldext_s8_xp_2p2">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_esp_vldext_u16_ip_2p2: ClangBuiltin<"__builtin_riscv_esp_vldext_u16_ip_2p2">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_esp_vldext_u16_xp_2p2: ClangBuiltin<"__builtin_riscv_esp_vldext_u16_xp_2p2">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_esp_vldext_u8_ip_2p2: ClangBuiltin<"__builtin_riscv_esp_vldext_u8_ip_2p2">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_esp_vldext_u8_xp_2p2: ClangBuiltin<"__builtin_riscv_esp_vldext_u8_xp_2p2">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_esp_vldhbc_16_incp_2p2: ClangBuiltin<"__builtin_riscv_esp_vldhbc_16_incp_2p2">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

}
