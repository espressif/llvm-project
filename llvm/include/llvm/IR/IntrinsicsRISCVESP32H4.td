let TargetPrefix = "riscv" in {
def int_riscv_esp_muls16ix2: ClangBuiltin<"__builtin_riscv_esp_muls16ix2">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_muls16x2: ClangBuiltin<"__builtin_riscv_esp_muls16x2">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], []>;

def int_riscv_esp_muls32i: ClangBuiltin<"__builtin_riscv_esp_muls32i">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_muls32: ClangBuiltin<"__builtin_riscv_esp_muls32">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], []>;

def int_riscv_esp_shl: ClangBuiltin<"__builtin_riscv_esp_shl">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], []>;

def int_riscv_esp_shr: ClangBuiltin<"__builtin_riscv_esp_shr">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], []>;

def int_riscv_esp_abs: ClangBuiltin<"__builtin_riscv_esp_abs">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty], []>;

def int_riscv_esp_movr_xacc: ClangBuiltin<"__builtin_riscv_esp_movr_xacc">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty], []>;

def int_riscv_esp_min: ClangBuiltin<"__builtin_riscv_esp_min">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], []>;

def int_riscv_esp_max: ClangBuiltin<"__builtin_riscv_esp_max">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], []>;

def int_riscv_esp_clzs: ClangBuiltin<"__builtin_riscv_esp_clzs">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty], []>;

def int_riscv_esp_flw_ip: ClangBuiltin<"__builtin_riscv_esp_flw_ip">,
	Intrinsic<[llvm_float_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_flw_xp: ClangBuiltin<"__builtin_riscv_esp_flw_xp">,
	Intrinsic<[llvm_float_ty], [llvm_i32_ty, llvm_i32_ty], []>;

def int_riscv_esp_fsw_ip: ClangBuiltin<"__builtin_riscv_esp_fsw_ip">,
	Intrinsic<[], [llvm_i32_ty, llvm_float_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_esp_fsw_xp: ClangBuiltin<"__builtin_riscv_esp_fsw_xp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_float_ty], []>;

def int_riscv_esp_lw_ip: ClangBuiltin<"__builtin_riscv_esp_lw_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_lw_xp: ClangBuiltin<"__builtin_riscv_esp_lw_xp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], []>;

def int_riscv_esp_sw_ip: ClangBuiltin<"__builtin_riscv_esp_sw_ip">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_esp_sw_xp: ClangBuiltin<"__builtin_riscv_esp_sw_xp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], []>;

def int_riscv_esp_lh_ip: ClangBuiltin<"__builtin_riscv_esp_lh_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_lh_xp: ClangBuiltin<"__builtin_riscv_esp_lh_xp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], []>;

def int_riscv_esp_sh_ip: ClangBuiltin<"__builtin_riscv_esp_sh_ip">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_esp_sh_xp: ClangBuiltin<"__builtin_riscv_esp_sh_xp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], []>;

def int_riscv_esp_lb_ip: ClangBuiltin<"__builtin_riscv_esp_lb_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_lb_xp: ClangBuiltin<"__builtin_riscv_esp_lb_xp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], []>;

def int_riscv_esp_sb_ip: ClangBuiltin<"__builtin_riscv_esp_sb_ip">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>, ImmArg<ArgIndex<3>>]>;

def int_riscv_esp_sb_xp: ClangBuiltin<"__builtin_riscv_esp_sb_xp">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], []>;

def int_riscv_esp_lhu_ip: ClangBuiltin<"__builtin_riscv_esp_lhu_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_lhu_xp: ClangBuiltin<"__builtin_riscv_esp_lhu_xp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], []>;

def int_riscv_esp_lbu_ip: ClangBuiltin<"__builtin_riscv_esp_lbu_ip">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_lbu_xp: ClangBuiltin<"__builtin_riscv_esp_lbu_xp">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty], []>;

def int_riscv_esp_movw_xacc: ClangBuiltin<"__builtin_riscv_esp_movw_xacc">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], []>;

def int_riscv_esp_macs32i: ClangBuiltin<"__builtin_riscv_esp_macs32i">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_macs32: ClangBuiltin<"__builtin_riscv_esp_macs32">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], []>;

def int_riscv_esp_macs32_ld: ClangBuiltin<"__builtin_riscv_esp_macs32_ld">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], []>;

def int_riscv_esp_macs32_st: ClangBuiltin<"__builtin_riscv_esp_macs32_st">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], []>;

def int_riscv_esp_macs16ix2: ClangBuiltin<"__builtin_riscv_esp_macs16ix2">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_macs16x2: ClangBuiltin<"__builtin_riscv_esp_macs16x2">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], []>;

def int_riscv_esp_macs16x2_ld: ClangBuiltin<"__builtin_riscv_esp_macs16x2_ld">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], []>;

def int_riscv_esp_macs16x1_ld: ClangBuiltin<"__builtin_riscv_esp_macs16x1_ld">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], []>;

def int_riscv_esp_macs16x2_st: ClangBuiltin<"__builtin_riscv_esp_macs16x2_st">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], []>;

def int_riscv_esp_msus32i: ClangBuiltin<"__builtin_riscv_esp_msus32i">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_msus32: ClangBuiltin<"__builtin_riscv_esp_msus32">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], []>;

def int_riscv_esp_msus32_ld: ClangBuiltin<"__builtin_riscv_esp_msus32_ld">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], []>;

def int_riscv_esp_msus32_st: ClangBuiltin<"__builtin_riscv_esp_msus32_st">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], []>;

def int_riscv_esp_msus16ix2: ClangBuiltin<"__builtin_riscv_esp_msus16ix2">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [ImmArg<ArgIndex<2>>]>;

def int_riscv_esp_msus16x2: ClangBuiltin<"__builtin_riscv_esp_msus16x2">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty], []>;

def int_riscv_esp_msus16x2_ld: ClangBuiltin<"__builtin_riscv_esp_msus16x2_ld">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], []>;

def int_riscv_esp_msus16x1_ld: ClangBuiltin<"__builtin_riscv_esp_msus16x1_ld">,
	Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], []>;

def int_riscv_esp_msus16x2_st: ClangBuiltin<"__builtin_riscv_esp_msus16x2_st">,
	Intrinsic<[], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], []>;



}
