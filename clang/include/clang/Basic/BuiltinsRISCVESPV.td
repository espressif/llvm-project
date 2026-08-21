let Features = "xespv2p1" in {
def esp_cmul_u16 : RISCVBuiltin<"void(unsigned int, unsigned int, unsigned int, unsigned int)">;
def esp_cmul_u16_ld_incp : RISCVBuiltin<"int(unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int)">;
def esp_cmul_u16_st_incp : RISCVBuiltin<"int(unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int)">;
def esp_cmul_u8 : RISCVBuiltin<"void(unsigned int, unsigned int, unsigned int, unsigned int)">;
def esp_cmul_u8_ld_incp : RISCVBuiltin<"int(unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int)">;
def esp_cmul_u8_st_incp : RISCVBuiltin<"int(unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int)">;
def esp_addx2 : RISCVBuiltin<"void(unsigned int, unsigned int, unsigned int)">;
def esp_addx4 : RISCVBuiltin<"void(unsigned int, unsigned int, unsigned int)">;
def esp_sat : RISCVBuiltin<"int(unsigned int, unsigned int, unsigned int)">;
def esp_subx2 : RISCVBuiltin<"void(unsigned int, unsigned int, unsigned int)">;
def esp_subx4 : RISCVBuiltin<"void(unsigned int, unsigned int, unsigned int)">;
def esp_vcmp_eq_u16 : RISCVBuiltin<"void(unsigned int, unsigned int, unsigned int)">;
def esp_vcmp_eq_u32 : RISCVBuiltin<"void(unsigned int, unsigned int, unsigned int)">;
def esp_vcmp_eq_u8 : RISCVBuiltin<"void(unsigned int, unsigned int, unsigned int)">;
def esp_vcmp_gt_u16 : RISCVBuiltin<"void(unsigned int, unsigned int, unsigned int)">;
def esp_vcmp_gt_u32 : RISCVBuiltin<"void(unsigned int, unsigned int, unsigned int)">;
def esp_vcmp_gt_u8 : RISCVBuiltin<"void(unsigned int, unsigned int, unsigned int)">;
}
let Features = "xespv" in {
def esp_cmul_u16_2p2 : RISCVBuiltin<"void(unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int)">;
def esp_cmul_u16_ld_incp_2p2 : RISCVBuiltin<"int(unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int)">;
def esp_cmul_u16_st_incp_2p2 : RISCVBuiltin<"int(unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int)">;
def esp_cmul_u8_2p2 : RISCVBuiltin<"void(unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int)">;
def esp_cmul_u8_ld_incp_2p2 : RISCVBuiltin<"int(unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int)">;
def esp_cmul_u8_st_incp_2p2 : RISCVBuiltin<"int(unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int)">;
def esp_addx2_2p2 : RISCVBuiltin<"void(unsigned int, unsigned int, unsigned int)">;
def esp_addx4_2p2 : RISCVBuiltin<"void(unsigned int, unsigned int, unsigned int)">;
def esp_sat_2p2 : RISCVBuiltin<"int(unsigned int, unsigned int, unsigned int)">;
def esp_subx2_2p2 : RISCVBuiltin<"void(unsigned int, unsigned int, unsigned int)">;
def esp_subx4_2p2 : RISCVBuiltin<"void(unsigned int, unsigned int, unsigned int)">;
def esp_vsl_u32_2p2 : RISCVBuiltin<"void(unsigned int, unsigned int, unsigned int)">;
}
