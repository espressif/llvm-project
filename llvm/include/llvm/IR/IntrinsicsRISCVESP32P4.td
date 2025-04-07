let TargetPrefix = "riscv" in {
def int_riscv_esp_vcmulas_s16_qacc_h: ClangBuiltin<"__builtin_riscv_esp_vcmulas_s16_qacc_h">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

def int_riscv_esp_vcmulas_s16_qacc_h_ld_ip: ClangBuiltin<"__builtin_riscv_esp_vcmulas_s16_qacc_h_ld_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vcmulas_s16_qacc_h_ld_xp: ClangBuiltin<"__builtin_riscv_esp_vcmulas_s16_qacc_h_ld_xp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vcmulas_s16_qacc_l: ClangBuiltin<"__builtin_riscv_esp_vcmulas_s16_qacc_l">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

def int_riscv_esp_vcmulas_s16_qacc_l_ld_ip: ClangBuiltin<"__builtin_riscv_esp_vcmulas_s16_qacc_l_ld_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vcmulas_s16_qacc_l_ld_xp: ClangBuiltin<"__builtin_riscv_esp_vcmulas_s16_qacc_l_ld_xp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vcmulas_s8_qacc_h: ClangBuiltin<"__builtin_riscv_esp_vcmulas_s8_qacc_h">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

def int_riscv_esp_vcmulas_s8_qacc_h_ld_ip: ClangBuiltin<"__builtin_riscv_esp_vcmulas_s8_qacc_h_ld_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vcmulas_s8_qacc_h_ld_xp: ClangBuiltin<"__builtin_riscv_esp_vcmulas_s8_qacc_h_ld_xp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vcmulas_s8_qacc_l: ClangBuiltin<"__builtin_riscv_esp_vcmulas_s8_qacc_l">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

def int_riscv_esp_vcmulas_s8_qacc_l_ld_ip: ClangBuiltin<"__builtin_riscv_esp_vcmulas_s8_qacc_l_ld_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vcmulas_s8_qacc_l_ld_xp: ClangBuiltin<"__builtin_riscv_esp_vcmulas_s8_qacc_l_ld_xp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmulas_s16_qacc: ClangBuiltin<"__builtin_riscv_esp_vmulas_s16_qacc">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

def int_riscv_esp_vmulas_s16_qacc_ld_ip: ClangBuiltin<"__builtin_riscv_esp_vmulas_s16_qacc_ld_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmulas_s16_qacc_ld_xp: ClangBuiltin<"__builtin_riscv_esp_vmulas_s16_qacc_ld_xp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmulas_s16_qacc_st_ip: ClangBuiltin<"__builtin_riscv_esp_vmulas_s16_qacc_st_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmulas_s16_qacc_st_xp: ClangBuiltin<"__builtin_riscv_esp_vmulas_s16_qacc_st_xp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_esp_vmulas_s16_xacc: ClangBuiltin<"__builtin_riscv_esp_vmulas_s16_xacc">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

def int_riscv_esp_vmulas_s16_xacc_ld_ip: ClangBuiltin<"__builtin_riscv_esp_vmulas_s16_xacc_ld_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmulas_s16_xacc_ld_xp: ClangBuiltin<"__builtin_riscv_esp_vmulas_s16_xacc_ld_xp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmulas_s16_xacc_st_ip: ClangBuiltin<"__builtin_riscv_esp_vmulas_s16_xacc_st_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmulas_s16_xacc_st_xp: ClangBuiltin<"__builtin_riscv_esp_vmulas_s16_xacc_st_xp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_esp_vmulas_s8_qacc: ClangBuiltin<"__builtin_riscv_esp_vmulas_s8_qacc">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

def int_riscv_esp_vmulas_s8_qacc_ld_ip: ClangBuiltin<"__builtin_riscv_esp_vmulas_s8_qacc_ld_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmulas_s8_qacc_ld_xp: ClangBuiltin<"__builtin_riscv_esp_vmulas_s8_qacc_ld_xp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmulas_s8_qacc_st_ip: ClangBuiltin<"__builtin_riscv_esp_vmulas_s8_qacc_st_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmulas_s8_qacc_st_xp: ClangBuiltin<"__builtin_riscv_esp_vmulas_s8_qacc_st_xp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_esp_vmulas_s8_xacc: ClangBuiltin<"__builtin_riscv_esp_vmulas_s8_xacc">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

def int_riscv_esp_vmulas_s8_xacc_ld_ip: ClangBuiltin<"__builtin_riscv_esp_vmulas_s8_xacc_ld_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmulas_s8_xacc_ld_xp: ClangBuiltin<"__builtin_riscv_esp_vmulas_s8_xacc_ld_xp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmulas_s8_xacc_st_ip: ClangBuiltin<"__builtin_riscv_esp_vmulas_s8_xacc_st_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmulas_s8_xacc_st_xp: ClangBuiltin<"__builtin_riscv_esp_vmulas_s8_xacc_st_xp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_esp_vmulas_u16_qacc: ClangBuiltin<"__builtin_riscv_esp_vmulas_u16_qacc">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

def int_riscv_esp_vmulas_u16_qacc_ld_ip: ClangBuiltin<"__builtin_riscv_esp_vmulas_u16_qacc_ld_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmulas_u16_qacc_ld_xp: ClangBuiltin<"__builtin_riscv_esp_vmulas_u16_qacc_ld_xp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmulas_u16_qacc_st_ip: ClangBuiltin<"__builtin_riscv_esp_vmulas_u16_qacc_st_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmulas_u16_qacc_st_xp: ClangBuiltin<"__builtin_riscv_esp_vmulas_u16_qacc_st_xp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_esp_vmulas_u16_xacc: ClangBuiltin<"__builtin_riscv_esp_vmulas_u16_xacc">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

def int_riscv_esp_vmulas_u16_xacc_ld_ip: ClangBuiltin<"__builtin_riscv_esp_vmulas_u16_xacc_ld_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmulas_u16_xacc_ld_xp: ClangBuiltin<"__builtin_riscv_esp_vmulas_u16_xacc_ld_xp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmulas_u16_xacc_st_ip: ClangBuiltin<"__builtin_riscv_esp_vmulas_u16_xacc_st_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmulas_u16_xacc_st_xp: ClangBuiltin<"__builtin_riscv_esp_vmulas_u16_xacc_st_xp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_esp_vmulas_u8_qacc: ClangBuiltin<"__builtin_riscv_esp_vmulas_u8_qacc">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

def int_riscv_esp_vmulas_u8_qacc_ld_ip: ClangBuiltin<"__builtin_riscv_esp_vmulas_u8_qacc_ld_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmulas_u8_qacc_ld_xp: ClangBuiltin<"__builtin_riscv_esp_vmulas_u8_qacc_ld_xp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmulas_u8_qacc_st_ip: ClangBuiltin<"__builtin_riscv_esp_vmulas_u8_qacc_st_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmulas_u8_qacc_st_xp: ClangBuiltin<"__builtin_riscv_esp_vmulas_u8_qacc_st_xp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_esp_vmulas_u8_xacc: ClangBuiltin<"__builtin_riscv_esp_vmulas_u8_xacc">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

def int_riscv_esp_vmulas_u8_xacc_ld_ip: ClangBuiltin<"__builtin_riscv_esp_vmulas_u8_xacc_ld_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmulas_u8_xacc_ld_xp: ClangBuiltin<"__builtin_riscv_esp_vmulas_u8_xacc_ld_xp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmulas_u8_xacc_st_ip: ClangBuiltin<"__builtin_riscv_esp_vmulas_u8_xacc_st_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmulas_u8_xacc_st_xp: ClangBuiltin<"__builtin_riscv_esp_vmulas_u8_xacc_st_xp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_esp_vmulas_s16_qacc_ldbc_incp: ClangBuiltin<"__builtin_riscv_esp_vmulas_s16_qacc_ldbc_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_esp_vmulas_s8_qacc_ldbc_incp: ClangBuiltin<"__builtin_riscv_esp_vmulas_s8_qacc_ldbc_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_esp_vmulas_u16_qacc_ldbc_incp: ClangBuiltin<"__builtin_riscv_esp_vmulas_u16_qacc_ldbc_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_esp_vmulas_u8_qacc_ldbc_incp: ClangBuiltin<"__builtin_riscv_esp_vmulas_u8_qacc_ldbc_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_esp_vsmulas_s16_qacc: ClangBuiltin<"__builtin_riscv_esp_vsmulas_s16_qacc">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vsmulas_s16_qacc_ld_incp: ClangBuiltin<"__builtin_riscv_esp_vsmulas_s16_qacc_ld_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vsmulas_s8_qacc: ClangBuiltin<"__builtin_riscv_esp_vsmulas_s8_qacc">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vsmulas_s8_qacc_ld_incp: ClangBuiltin<"__builtin_riscv_esp_vsmulas_s8_qacc_ld_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vsmulas_u16_qacc: ClangBuiltin<"__builtin_riscv_esp_vsmulas_u16_qacc">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vsmulas_u16_qacc_ld_incp: ClangBuiltin<"__builtin_riscv_esp_vsmulas_u16_qacc_ld_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vsmulas_u8_qacc: ClangBuiltin<"__builtin_riscv_esp_vsmulas_u8_qacc">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vsmulas_u8_qacc_ld_incp: ClangBuiltin<"__builtin_riscv_esp_vsmulas_u8_qacc_ld_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_cmul_s16: ClangBuiltin<"__builtin_riscv_esp_cmul_s16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_esp_cmul_s16_ld_incp: ClangBuiltin<"__builtin_riscv_esp_cmul_s16_ld_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>]>;

def int_riscv_esp_cmul_s16_st_incp: ClangBuiltin<"__builtin_riscv_esp_cmul_s16_st_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>]>;

def int_riscv_esp_cmul_s8: ClangBuiltin<"__builtin_riscv_esp_cmul_s8">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_esp_cmul_s8_ld_incp: ClangBuiltin<"__builtin_riscv_esp_cmul_s8_ld_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>]>;

def int_riscv_esp_cmul_s8_st_incp: ClangBuiltin<"__builtin_riscv_esp_cmul_s8_st_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>]>;

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

def int_riscv_esp_max_s16_a: ClangBuiltin<"__builtin_riscv_esp_max_s16_a">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;

def int_riscv_esp_max_s32_a: ClangBuiltin<"__builtin_riscv_esp_max_s32_a">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;

def int_riscv_esp_max_s8_a: ClangBuiltin<"__builtin_riscv_esp_max_s8_a">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;

def int_riscv_esp_max_u16_a: ClangBuiltin<"__builtin_riscv_esp_max_u16_a">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;

def int_riscv_esp_max_u32_a: ClangBuiltin<"__builtin_riscv_esp_max_u32_a">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;

def int_riscv_esp_max_u8_a: ClangBuiltin<"__builtin_riscv_esp_max_u8_a">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;

def int_riscv_esp_min_s16_a: ClangBuiltin<"__builtin_riscv_esp_min_s16_a">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;

def int_riscv_esp_min_s32_a: ClangBuiltin<"__builtin_riscv_esp_min_s32_a">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;

def int_riscv_esp_min_s8_a: ClangBuiltin<"__builtin_riscv_esp_min_s8_a">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;

def int_riscv_esp_min_u16_a: ClangBuiltin<"__builtin_riscv_esp_min_u16_a">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;

def int_riscv_esp_min_u32_a: ClangBuiltin<"__builtin_riscv_esp_min_u32_a">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;

def int_riscv_esp_min_u8_a: ClangBuiltin<"__builtin_riscv_esp_min_u8_a">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;

def int_riscv_esp_vabs_16: ClangBuiltin<"__builtin_riscv_esp_vabs_16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

def int_riscv_esp_vabs_32: ClangBuiltin<"__builtin_riscv_esp_vabs_32">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

def int_riscv_esp_vabs_8: ClangBuiltin<"__builtin_riscv_esp_vabs_8">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

def int_riscv_esp_vadd_s16: ClangBuiltin<"__builtin_riscv_esp_vadd_s16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vadd_s16_ld_incp: ClangBuiltin<"__builtin_riscv_esp_vadd_s16_ld_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vadd_s16_st_incp: ClangBuiltin<"__builtin_riscv_esp_vadd_s16_st_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vadd_s32: ClangBuiltin<"__builtin_riscv_esp_vadd_s32">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vadd_s32_ld_incp: ClangBuiltin<"__builtin_riscv_esp_vadd_s32_ld_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vadd_s32_st_incp: ClangBuiltin<"__builtin_riscv_esp_vadd_s32_st_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vadd_s8: ClangBuiltin<"__builtin_riscv_esp_vadd_s8">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vadd_s8_ld_incp: ClangBuiltin<"__builtin_riscv_esp_vadd_s8_ld_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vadd_s8_st_incp: ClangBuiltin<"__builtin_riscv_esp_vadd_s8_st_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vadd_u16: ClangBuiltin<"__builtin_riscv_esp_vadd_u16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vadd_u16_ld_incp: ClangBuiltin<"__builtin_riscv_esp_vadd_u16_ld_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vadd_u16_st_incp: ClangBuiltin<"__builtin_riscv_esp_vadd_u16_st_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vadd_u32: ClangBuiltin<"__builtin_riscv_esp_vadd_u32">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vadd_u32_ld_incp: ClangBuiltin<"__builtin_riscv_esp_vadd_u32_ld_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vadd_u32_st_incp: ClangBuiltin<"__builtin_riscv_esp_vadd_u32_st_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vadd_u8: ClangBuiltin<"__builtin_riscv_esp_vadd_u8">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vadd_u8_ld_incp: ClangBuiltin<"__builtin_riscv_esp_vadd_u8_ld_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vadd_u8_st_incp: ClangBuiltin<"__builtin_riscv_esp_vadd_u8_st_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vclamp_s16: ClangBuiltin<"__builtin_riscv_esp_vclamp_s16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vmax_s16: ClangBuiltin<"__builtin_riscv_esp_vmax_s16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vmax_s16_ld_incp: ClangBuiltin<"__builtin_riscv_esp_vmax_s16_ld_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmax_s16_st_incp: ClangBuiltin<"__builtin_riscv_esp_vmax_s16_st_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmax_s32: ClangBuiltin<"__builtin_riscv_esp_vmax_s32">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vmax_s32_ld_incp: ClangBuiltin<"__builtin_riscv_esp_vmax_s32_ld_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmax_s32_st_incp: ClangBuiltin<"__builtin_riscv_esp_vmax_s32_st_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmax_s8: ClangBuiltin<"__builtin_riscv_esp_vmax_s8">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vmax_s8_ld_incp: ClangBuiltin<"__builtin_riscv_esp_vmax_s8_ld_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmax_s8_st_incp: ClangBuiltin<"__builtin_riscv_esp_vmax_s8_st_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmax_u16: ClangBuiltin<"__builtin_riscv_esp_vmax_u16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vmax_u16_ld_incp: ClangBuiltin<"__builtin_riscv_esp_vmax_u16_ld_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmax_u16_st_incp: ClangBuiltin<"__builtin_riscv_esp_vmax_u16_st_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmax_u32: ClangBuiltin<"__builtin_riscv_esp_vmax_u32">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vmax_u32_ld_incp: ClangBuiltin<"__builtin_riscv_esp_vmax_u32_ld_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmax_u32_st_incp: ClangBuiltin<"__builtin_riscv_esp_vmax_u32_st_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmax_u8: ClangBuiltin<"__builtin_riscv_esp_vmax_u8">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vmax_u8_ld_incp: ClangBuiltin<"__builtin_riscv_esp_vmax_u8_ld_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmax_u8_st_incp: ClangBuiltin<"__builtin_riscv_esp_vmax_u8_st_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmin_s16: ClangBuiltin<"__builtin_riscv_esp_vmin_s16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vmin_s16_ld_incp: ClangBuiltin<"__builtin_riscv_esp_vmin_s16_ld_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmin_s16_st_incp: ClangBuiltin<"__builtin_riscv_esp_vmin_s16_st_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmin_s32: ClangBuiltin<"__builtin_riscv_esp_vmin_s32">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vmin_s32_ld_incp: ClangBuiltin<"__builtin_riscv_esp_vmin_s32_ld_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmin_s32_st_incp: ClangBuiltin<"__builtin_riscv_esp_vmin_s32_st_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmin_s8: ClangBuiltin<"__builtin_riscv_esp_vmin_s8">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vmin_s8_ld_incp: ClangBuiltin<"__builtin_riscv_esp_vmin_s8_ld_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmin_s8_st_incp: ClangBuiltin<"__builtin_riscv_esp_vmin_s8_st_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmin_u16: ClangBuiltin<"__builtin_riscv_esp_vmin_u16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vmin_u16_ld_incp: ClangBuiltin<"__builtin_riscv_esp_vmin_u16_ld_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmin_u16_st_incp: ClangBuiltin<"__builtin_riscv_esp_vmin_u16_st_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmin_u32: ClangBuiltin<"__builtin_riscv_esp_vmin_u32">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vmin_u32_ld_incp: ClangBuiltin<"__builtin_riscv_esp_vmin_u32_ld_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmin_u32_st_incp: ClangBuiltin<"__builtin_riscv_esp_vmin_u32_st_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmin_u8: ClangBuiltin<"__builtin_riscv_esp_vmin_u8">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vmin_u8_ld_incp: ClangBuiltin<"__builtin_riscv_esp_vmin_u8_ld_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmin_u8_st_incp: ClangBuiltin<"__builtin_riscv_esp_vmin_u8_st_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmul_s16: ClangBuiltin<"__builtin_riscv_esp_vmul_s16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vmul_s16_ld_incp: ClangBuiltin<"__builtin_riscv_esp_vmul_s16_ld_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmul_s16_s8xs8: ClangBuiltin<"__builtin_riscv_esp_vmul_s16_s8xs8">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_esp_vmul_s16_st_incp: ClangBuiltin<"__builtin_riscv_esp_vmul_s16_st_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmul_s32_s16xs16: ClangBuiltin<"__builtin_riscv_esp_vmul_s32_s16xs16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_esp_vmul_s8: ClangBuiltin<"__builtin_riscv_esp_vmul_s8">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vmul_s8_ld_incp: ClangBuiltin<"__builtin_riscv_esp_vmul_s8_ld_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmul_s8_st_incp: ClangBuiltin<"__builtin_riscv_esp_vmul_s8_st_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmul_u16: ClangBuiltin<"__builtin_riscv_esp_vmul_u16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vmul_u16_ld_incp: ClangBuiltin<"__builtin_riscv_esp_vmul_u16_ld_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmul_u16_st_incp: ClangBuiltin<"__builtin_riscv_esp_vmul_u16_st_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmul_u8: ClangBuiltin<"__builtin_riscv_esp_vmul_u8">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vmul_u8_ld_incp: ClangBuiltin<"__builtin_riscv_esp_vmul_u8_ld_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vmul_u8_st_incp: ClangBuiltin<"__builtin_riscv_esp_vmul_u8_st_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vprelu_s16: ClangBuiltin<"__builtin_riscv_esp_vprelu_s16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_esp_vprelu_s8: ClangBuiltin<"__builtin_riscv_esp_vprelu_s8">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_esp_vrelu_s16: ClangBuiltin<"__builtin_riscv_esp_vrelu_s16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vrelu_s8: ClangBuiltin<"__builtin_riscv_esp_vrelu_s8">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vsadds_s16: ClangBuiltin<"__builtin_riscv_esp_vsadds_s16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vsadds_s8: ClangBuiltin<"__builtin_riscv_esp_vsadds_s8">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vsadds_u16: ClangBuiltin<"__builtin_riscv_esp_vsadds_u16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vsadds_u8: ClangBuiltin<"__builtin_riscv_esp_vsadds_u8">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vsat_s16: ClangBuiltin<"__builtin_riscv_esp_vsat_s16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_esp_vsat_s32: ClangBuiltin<"__builtin_riscv_esp_vsat_s32">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_esp_vsat_s8: ClangBuiltin<"__builtin_riscv_esp_vsat_s8">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_esp_vsat_u16: ClangBuiltin<"__builtin_riscv_esp_vsat_u16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_esp_vsat_u32: ClangBuiltin<"__builtin_riscv_esp_vsat_u32">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_esp_vsat_u8: ClangBuiltin<"__builtin_riscv_esp_vsat_u8">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_esp_vssubs_s16: ClangBuiltin<"__builtin_riscv_esp_vssubs_s16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vssubs_s8: ClangBuiltin<"__builtin_riscv_esp_vssubs_s8">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vssubs_u16: ClangBuiltin<"__builtin_riscv_esp_vssubs_u16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vssubs_u8: ClangBuiltin<"__builtin_riscv_esp_vssubs_u8">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vsub_s16: ClangBuiltin<"__builtin_riscv_esp_vsub_s16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vsub_s16_ld_incp: ClangBuiltin<"__builtin_riscv_esp_vsub_s16_ld_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vsub_s16_st_incp: ClangBuiltin<"__builtin_riscv_esp_vsub_s16_st_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vsub_s32: ClangBuiltin<"__builtin_riscv_esp_vsub_s32">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vsub_s32_ld_incp: ClangBuiltin<"__builtin_riscv_esp_vsub_s32_ld_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vsub_s32_st_incp: ClangBuiltin<"__builtin_riscv_esp_vsub_s32_st_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vsub_s8: ClangBuiltin<"__builtin_riscv_esp_vsub_s8">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vsub_s8_ld_incp: ClangBuiltin<"__builtin_riscv_esp_vsub_s8_ld_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vsub_s8_st_incp: ClangBuiltin<"__builtin_riscv_esp_vsub_s8_st_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vsub_u16: ClangBuiltin<"__builtin_riscv_esp_vsub_u16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vsub_u16_ld_incp: ClangBuiltin<"__builtin_riscv_esp_vsub_u16_ld_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vsub_u16_st_incp: ClangBuiltin<"__builtin_riscv_esp_vsub_u16_st_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vsub_u32: ClangBuiltin<"__builtin_riscv_esp_vsub_u32">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vsub_u32_ld_incp: ClangBuiltin<"__builtin_riscv_esp_vsub_u32_ld_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vsub_u32_st_incp: ClangBuiltin<"__builtin_riscv_esp_vsub_u32_st_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vsub_u8: ClangBuiltin<"__builtin_riscv_esp_vsub_u8">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vsub_u8_ld_incp: ClangBuiltin<"__builtin_riscv_esp_vsub_u8_ld_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vsub_u8_st_incp: ClangBuiltin<"__builtin_riscv_esp_vsub_u8_st_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>]>;

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

def int_riscv_esp_andq: ClangBuiltin<"__builtin_riscv_esp_andq">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_notq: ClangBuiltin<"__builtin_riscv_esp_notq">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

def int_riscv_esp_orq: ClangBuiltin<"__builtin_riscv_esp_orq">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_xorq: ClangBuiltin<"__builtin_riscv_esp_xorq">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vcmp_eq_s16: ClangBuiltin<"__builtin_riscv_esp_vcmp_eq_s16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vcmp_eq_s32: ClangBuiltin<"__builtin_riscv_esp_vcmp_eq_s32">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vcmp_eq_s8: ClangBuiltin<"__builtin_riscv_esp_vcmp_eq_s8">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vcmp_eq_u16: ClangBuiltin<"__builtin_riscv_esp_vcmp_eq_u16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vcmp_eq_u32: ClangBuiltin<"__builtin_riscv_esp_vcmp_eq_u32">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vcmp_eq_u8: ClangBuiltin<"__builtin_riscv_esp_vcmp_eq_u8">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vcmp_gt_s16: ClangBuiltin<"__builtin_riscv_esp_vcmp_gt_s16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vcmp_gt_s32: ClangBuiltin<"__builtin_riscv_esp_vcmp_gt_s32">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vcmp_gt_s8: ClangBuiltin<"__builtin_riscv_esp_vcmp_gt_s8">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vcmp_gt_u16: ClangBuiltin<"__builtin_riscv_esp_vcmp_gt_u16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vcmp_gt_u32: ClangBuiltin<"__builtin_riscv_esp_vcmp_gt_u32">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vcmp_gt_u8: ClangBuiltin<"__builtin_riscv_esp_vcmp_gt_u8">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vcmp_lt_s16: ClangBuiltin<"__builtin_riscv_esp_vcmp_lt_s16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vcmp_lt_s32: ClangBuiltin<"__builtin_riscv_esp_vcmp_lt_s32">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vcmp_lt_s8: ClangBuiltin<"__builtin_riscv_esp_vcmp_lt_s8">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vcmp_lt_u16: ClangBuiltin<"__builtin_riscv_esp_vcmp_lt_u16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vcmp_lt_u32: ClangBuiltin<"__builtin_riscv_esp_vcmp_lt_u32">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vcmp_lt_u8: ClangBuiltin<"__builtin_riscv_esp_vcmp_lt_u8">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_mov_s16_qacc: ClangBuiltin<"__builtin_riscv_esp_mov_s16_qacc">,
	Intrinsic<[], [llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;

def int_riscv_esp_mov_s8_qacc: ClangBuiltin<"__builtin_riscv_esp_mov_s8_qacc">,
	Intrinsic<[], [llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;

def int_riscv_esp_mov_u16_qacc: ClangBuiltin<"__builtin_riscv_esp_mov_u16_qacc">,
	Intrinsic<[], [llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;

def int_riscv_esp_mov_u8_qacc: ClangBuiltin<"__builtin_riscv_esp_mov_u8_qacc">,
	Intrinsic<[], [llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;

def int_riscv_esp_movi_16_a: ClangBuiltin<"__builtin_riscv_esp_movi_16_a">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

def int_riscv_esp_movi_16_q: ClangBuiltin<"__builtin_riscv_esp_movi_16_q">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_movi_32_a: ClangBuiltin<"__builtin_riscv_esp_movi_32_a">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

def int_riscv_esp_movi_32_q: ClangBuiltin<"__builtin_riscv_esp_movi_32_q">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_movi_8_a: ClangBuiltin<"__builtin_riscv_esp_movi_8_a">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

def int_riscv_esp_movi_8_q: ClangBuiltin<"__builtin_riscv_esp_movi_8_q">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_movx_r_cfg: ClangBuiltin<"__builtin_riscv_esp_movx_r_cfg">,
	Intrinsic<[llvm_i32_ty], [], []>;

def int_riscv_esp_movx_r_fft_bit_width: ClangBuiltin<"__builtin_riscv_esp_movx_r_fft_bit_width">,
	Intrinsic<[], [llvm_i32_ty], []>;

def int_riscv_esp_movx_r_perf: ClangBuiltin<"__builtin_riscv_esp_movx_r_perf">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], []>;

def int_riscv_esp_movx_r_sar: ClangBuiltin<"__builtin_riscv_esp_movx_r_sar">,
	Intrinsic<[], [llvm_i32_ty], []>;

def int_riscv_esp_movx_r_sar_bytes: ClangBuiltin<"__builtin_riscv_esp_movx_r_sar_bytes">,
	Intrinsic<[], [llvm_i32_ty], []>;

def int_riscv_esp_movx_r_xacc_h: ClangBuiltin<"__builtin_riscv_esp_movx_r_xacc_h">,
	Intrinsic<[], [llvm_i32_ty], []>;

def int_riscv_esp_movx_r_xacc_l: ClangBuiltin<"__builtin_riscv_esp_movx_r_xacc_l">,
	Intrinsic<[], [llvm_i32_ty], []>;

def int_riscv_esp_movx_w_cfg: ClangBuiltin<"__builtin_riscv_esp_movx_w_cfg">,
	Intrinsic<[], [llvm_i32_ty], []>;

def int_riscv_esp_movx_w_fft_bit_width: ClangBuiltin<"__builtin_riscv_esp_movx_w_fft_bit_width">,
	Intrinsic<[], [llvm_i32_ty], []>;

def int_riscv_esp_movx_w_perf: ClangBuiltin<"__builtin_riscv_esp_movx_w_perf">,
	Intrinsic<[], [llvm_i32_ty], []>;

def int_riscv_esp_movx_w_sar: ClangBuiltin<"__builtin_riscv_esp_movx_w_sar">,
	Intrinsic<[], [llvm_i32_ty], []>;

def int_riscv_esp_movx_w_sar_bytes: ClangBuiltin<"__builtin_riscv_esp_movx_w_sar_bytes">,
	Intrinsic<[], [llvm_i32_ty], []>;

def int_riscv_esp_movx_w_xacc_h: ClangBuiltin<"__builtin_riscv_esp_movx_w_xacc_h">,
	Intrinsic<[], [llvm_i32_ty], []>;

def int_riscv_esp_movx_w_xacc_l: ClangBuiltin<"__builtin_riscv_esp_movx_w_xacc_l">,
	Intrinsic<[], [llvm_i32_ty], []>;

def int_riscv_esp_vext_s16: ClangBuiltin<"__builtin_riscv_esp_vext_s16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vext_s8: ClangBuiltin<"__builtin_riscv_esp_vext_s8">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vext_u16: ClangBuiltin<"__builtin_riscv_esp_vext_u16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vext_u8: ClangBuiltin<"__builtin_riscv_esp_vext_u8">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vunzip_16: ClangBuiltin<"__builtin_riscv_esp_vunzip_16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

def int_riscv_esp_vunzip_32: ClangBuiltin<"__builtin_riscv_esp_vunzip_32">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

def int_riscv_esp_vunzip_8: ClangBuiltin<"__builtin_riscv_esp_vunzip_8">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

def int_riscv_esp_vunzipt_16: ClangBuiltin<"__builtin_riscv_esp_vunzipt_16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vunzipt_8: ClangBuiltin<"__builtin_riscv_esp_vunzipt_8">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vzip_16: ClangBuiltin<"__builtin_riscv_esp_vzip_16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

def int_riscv_esp_vzip_32: ClangBuiltin<"__builtin_riscv_esp_vzip_32">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

def int_riscv_esp_vzip_8: ClangBuiltin<"__builtin_riscv_esp_vzip_8">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

def int_riscv_esp_vzipt_16: ClangBuiltin<"__builtin_riscv_esp_vzipt_16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vzipt_8: ClangBuiltin<"__builtin_riscv_esp_vzipt_8">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_zero_q: ClangBuiltin<"__builtin_riscv_esp_zero_q">,
	Intrinsic<[], [llvm_i32_ty], [ImmArg<ArgIndex<0>>]>;

def int_riscv_esp_zero_qacc: ClangBuiltin<"__builtin_riscv_esp_zero_qacc">,
	Intrinsic<[], [], []>;

def int_riscv_esp_zero_xacc: ClangBuiltin<"__builtin_riscv_esp_zero_xacc">,
	Intrinsic<[], [], []>;

def int_riscv_esp_fft_ams_s16_ld_incp: ClangBuiltin<"__builtin_riscv_esp_fft_ams_s16_ld_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>, ImmArg<ArgIndex<7>>]>;

def int_riscv_esp_fft_ams_s16_ld_incp_uaup: ClangBuiltin<"__builtin_riscv_esp_fft_ams_s16_ld_incp_uaup">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>, ImmArg<ArgIndex<7>>]>;

def int_riscv_esp_fft_ams_s16_ld_r32_decp: ClangBuiltin<"__builtin_riscv_esp_fft_ams_s16_ld_r32_decp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>, ImmArg<ArgIndex<7>>]>;

def int_riscv_esp_fft_ams_s16_st_incp: ClangBuiltin<"__builtin_riscv_esp_fft_ams_s16_st_incp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<6>>, ImmArg<ArgIndex<7>>]>;

def int_riscv_esp_fft_bitrev: ClangBuiltin<"__builtin_riscv_esp_fft_bitrev">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_riscv_esp_fft_cmul_s16_ld_xp: ClangBuiltin<"__builtin_riscv_esp_fft_cmul_s16_ld_xp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<4>>, ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>]>;

def int_riscv_esp_fft_cmul_s16_st_xp: ClangBuiltin<"__builtin_riscv_esp_fft_cmul_s16_st_xp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<5>>, ImmArg<ArgIndex<6>>, ImmArg<ArgIndex<7>>]>;

def int_riscv_esp_fft_r2bf_s16: ClangBuiltin<"__builtin_riscv_esp_fft_r2bf_s16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_fft_r2bf_s16_st_incp: ClangBuiltin<"__builtin_riscv_esp_fft_r2bf_s16_st_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_fft_vst_r32_decp: ClangBuiltin<"__builtin_riscv_esp_fft_vst_r32_decp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_ld_128_usar_ip: ClangBuiltin<"__builtin_riscv_esp_ld_128_usar_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_ld_128_usar_xp: ClangBuiltin<"__builtin_riscv_esp_ld_128_usar_xp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_ld_xacc_ip: ClangBuiltin<"__builtin_riscv_esp_ld_xacc_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_riscv_esp_ldqa_s16_128_ip: ClangBuiltin<"__builtin_riscv_esp_ldqa_s16_128_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_riscv_esp_ldqa_s16_128_xp: ClangBuiltin<"__builtin_riscv_esp_ldqa_s16_128_xp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], []>;

def int_riscv_esp_ldqa_s8_128_ip: ClangBuiltin<"__builtin_riscv_esp_ldqa_s8_128_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_riscv_esp_ldqa_s8_128_xp: ClangBuiltin<"__builtin_riscv_esp_ldqa_s8_128_xp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], []>;

def int_riscv_esp_ldqa_u16_128_ip: ClangBuiltin<"__builtin_riscv_esp_ldqa_u16_128_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_riscv_esp_ldqa_u16_128_xp: ClangBuiltin<"__builtin_riscv_esp_ldqa_u16_128_xp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], []>;

def int_riscv_esp_ldqa_u8_128_ip: ClangBuiltin<"__builtin_riscv_esp_ldqa_u8_128_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_riscv_esp_ldqa_u8_128_xp: ClangBuiltin<"__builtin_riscv_esp_ldqa_u8_128_xp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], []>;

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

def int_riscv_esp_ld_qacc_h_h_128_ip: ClangBuiltin<"__builtin_riscv_esp_ld_qacc_h_h_128_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_riscv_esp_ld_qacc_h_l_128_ip: ClangBuiltin<"__builtin_riscv_esp_ld_qacc_h_l_128_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_riscv_esp_ld_qacc_l_h_128_ip: ClangBuiltin<"__builtin_riscv_esp_ld_qacc_l_h_128_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_riscv_esp_ld_qacc_l_l_128_ip: ClangBuiltin<"__builtin_riscv_esp_ld_qacc_l_l_128_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_riscv_esp_ld_ua_state_ip: ClangBuiltin<"__builtin_riscv_esp_ld_ua_state_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_riscv_esp_ldxq_32: ClangBuiltin<"__builtin_riscv_esp_ldxq_32">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_st_qacc_h_h_128_ip: ClangBuiltin<"__builtin_riscv_esp_st_qacc_h_h_128_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_riscv_esp_st_qacc_h_l_128_ip: ClangBuiltin<"__builtin_riscv_esp_st_qacc_h_l_128_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_riscv_esp_st_qacc_l_h_128_ip: ClangBuiltin<"__builtin_riscv_esp_st_qacc_l_h_128_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_riscv_esp_st_qacc_l_l_128_ip: ClangBuiltin<"__builtin_riscv_esp_st_qacc_l_l_128_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_riscv_esp_st_ua_state_ip: ClangBuiltin<"__builtin_riscv_esp_st_ua_state_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_riscv_esp_stxq_32: ClangBuiltin<"__builtin_riscv_esp_stxq_32">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_vld_128_ip: ClangBuiltin<"__builtin_riscv_esp_vld_128_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vld_128_xp: ClangBuiltin<"__builtin_riscv_esp_vld_128_xp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vld_h_64_ip: ClangBuiltin<"__builtin_riscv_esp_vld_h_64_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vld_h_64_xp: ClangBuiltin<"__builtin_riscv_esp_vld_h_64_xp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vld_l_64_ip: ClangBuiltin<"__builtin_riscv_esp_vld_l_64_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vld_l_64_xp: ClangBuiltin<"__builtin_riscv_esp_vld_l_64_xp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vst_128_ip: ClangBuiltin<"__builtin_riscv_esp_vst_128_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vst_128_xp: ClangBuiltin<"__builtin_riscv_esp_vst_128_xp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_riscv_esp_vst_h_64_ip: ClangBuiltin<"__builtin_riscv_esp_vst_h_64_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vst_h_64_xp: ClangBuiltin<"__builtin_riscv_esp_vst_h_64_xp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_riscv_esp_vst_l_64_ip: ClangBuiltin<"__builtin_riscv_esp_vst_l_64_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vst_l_64_xp: ClangBuiltin<"__builtin_riscv_esp_vst_l_64_xp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_riscv_esp_slci_2q: ClangBuiltin<"__builtin_riscv_esp_slci_2q">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_slcxxp_2q: ClangBuiltin<"__builtin_riscv_esp_slcxxp_2q">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_esp_src_q: ClangBuiltin<"__builtin_riscv_esp_src_q">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_src_q_ld_ip: ClangBuiltin<"__builtin_riscv_esp_src_q_ld_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_src_q_ld_xp: ClangBuiltin<"__builtin_riscv_esp_src_q_ld_xp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<3>>, ImmArg<ArgIndex<4>>]>;

def int_riscv_esp_src_q_qup: ClangBuiltin<"__builtin_riscv_esp_src_q_qup">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_srci_2q: ClangBuiltin<"__builtin_riscv_esp_srci_2q">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_srcmb_s16_q_qacc: ClangBuiltin<"__builtin_riscv_esp_srcmb_s16_q_qacc">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_srcmb_s16_qacc: ClangBuiltin<"__builtin_riscv_esp_srcmb_s16_qacc">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_srcmb_s8_q_qacc: ClangBuiltin<"__builtin_riscv_esp_srcmb_s8_q_qacc">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_srcmb_s8_qacc: ClangBuiltin<"__builtin_riscv_esp_srcmb_s8_qacc">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_srcmb_u16_q_qacc: ClangBuiltin<"__builtin_riscv_esp_srcmb_u16_q_qacc">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_srcmb_u16_qacc: ClangBuiltin<"__builtin_riscv_esp_srcmb_u16_qacc">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_srcmb_u8_q_qacc: ClangBuiltin<"__builtin_riscv_esp_srcmb_u8_q_qacc">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_srcmb_u8_qacc: ClangBuiltin<"__builtin_riscv_esp_srcmb_u8_qacc">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_srcq_128_st_incp: ClangBuiltin<"__builtin_riscv_esp_srcq_128_st_incp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

def int_riscv_esp_srcxxp_2q: ClangBuiltin<"__builtin_riscv_esp_srcxxp_2q">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_esp_srs_s_xacc: ClangBuiltin<"__builtin_riscv_esp_srs_s_xacc">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], []>;

def int_riscv_esp_srs_u_xacc: ClangBuiltin<"__builtin_riscv_esp_srs_u_xacc">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], []>;

def int_riscv_esp_vsl_32: ClangBuiltin<"__builtin_riscv_esp_vsl_32">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

def int_riscv_esp_vsld_16: ClangBuiltin<"__builtin_riscv_esp_vsld_16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vsld_32: ClangBuiltin<"__builtin_riscv_esp_vsld_32">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vsld_8: ClangBuiltin<"__builtin_riscv_esp_vsld_8">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vsr_s32: ClangBuiltin<"__builtin_riscv_esp_vsr_s32">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

def int_riscv_esp_vsr_u32: ClangBuiltin<"__builtin_riscv_esp_vsr_u32">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>]>;

def int_riscv_esp_vsrd_16: ClangBuiltin<"__builtin_riscv_esp_vsrd_16">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vsrd_32: ClangBuiltin<"__builtin_riscv_esp_vsrd_32">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_vsrd_8: ClangBuiltin<"__builtin_riscv_esp_vsrd_8">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<0>>, ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_st_s_xacc_ip: ClangBuiltin<"__builtin_riscv_esp_st_s_xacc_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;

def int_riscv_esp_st_u_xacc_ip: ClangBuiltin<"__builtin_riscv_esp_st_u_xacc_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>]>;


}
