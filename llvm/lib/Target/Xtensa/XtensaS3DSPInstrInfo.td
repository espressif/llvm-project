//===- XtensaS3DSPInstrInfo.td - Xtensa Target Description -*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the Xtensa S3 DSP instructions in TableGen format.
//
// These definitions are generated
// This file is generated
// AI_S6_V2.h
//
//===----------------------------------------------------------------------===//

// This file is generated


def EE_ANDQ:  EE_Inst24<(outs QR:$qa), (ins QR:$qx, QR:$qy),
                   "ee.andq\t $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;
  
  
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qa{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qa{0};
  let Inst{14-12} = 0x3;
  let Inst{11-10} = qy{2-1};
  let Inst{9-8} = 0x0;
  let Inst{7-6} = qx{2-1};
  let Inst{5} = qy{0};
  let Inst{4} = qx{0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_ANDQ_P : Pseudo<(outs), (ins imm8:$qa, imm8:$qx, imm8:$qy),
				"!xtensa_ee_andq_p, $qa, $qx, $qy",
			[(int_xtensa_ee_andq timm:$qa, timm:$qx, timm:$qy)]>;

def EE_BITREV:  EE_Inst24<(outs QR:$qa, AR:$axr), (ins AR:$ax),
                   "ee.bitrev\t $qa, $ax", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<4> ax;
  
  
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qa{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qa{0};
  let Inst{14-8} = 0x7b;
  let Inst{7-4} = ax{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_BITREV_P : Pseudo<(outs), (ins imm8:$qa, AR:$ax),
				"!xtensa_ee_bitrev_p, $qa, $ax",
			[(int_xtensa_ee_bitrev timm:$qa, AR:$ax)]>;

def EE_CMUL_S16:  EE_Inst24<(outs QR:$qz), (ins QR:$qx, QR:$qy, select_4:$sel4),
                   "ee.cmul.s16\t $qz, $qx, $qy, $sel4", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qz;
  bits<3> qx;
  bits<3> qy;
  bits<2> sel4;
  
  
  let Inst{23-22} = 0x2;
  let Inst{21-20} = qz{2-1};
  let Inst{19-16} = 0xe;
  let Inst{15} = qz{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x0;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-6} = 0x0;
  let Inst{5-4} = sel4{1-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_CMUL_S16_P : Pseudo<(outs), (ins imm8:$qz, imm8:$qx, imm8:$qy, select_4:$sel4),
				"!xtensa_ee_cmul_s16_p, $qz, $qx, $qy, $sel4",
			[(int_xtensa_ee_cmul_s16 timm:$qz, timm:$qx, timm:$qy, timm:$sel4)]>;

def EE_CMUL_S16_LD_INCP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qz), (ins AR:$as, QR:$qx, QR:$qy, select_4:$sel4),
                   "ee.cmul.s16.ld.incp\t $qu, $as, $qz, $qx, $qy, $sel4", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qz;
  bits<3> qx;
  bits<3> qy;
  bits<2> sel4;
  
  let mayLoad = 1;
  
  let Inst{28-23} = 0x38;
  let Inst{22-20} = qu{2-0};
  let Inst{19-17} = qz{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-6} = 0x3;
  let Inst{5-4} = sel4{1-0};
  let Inst{3-0} = as{3-0};
  
  let Constraints = "$asr = $as";
}

let usesCustomInserter = 1 in
def EE_CMUL_S16_LD_INCP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, imm8:$qz, imm8:$qx, imm8:$qy, select_4:$sel4),
				"!xtensa_ee_cmul_s16_ld_incp_p, $qu, $as, $qz, $qx, $qy, $sel4",
			[(int_xtensa_ee_cmul_s16_ld_incp timm:$qu, AR:$as, timm:$qz, timm:$qx, timm:$qy, timm:$sel4)]>;

def EE_CMUL_S16_ST_INCP:  EE_Inst32<(outs AR:$asr, QR:$qz), (ins QR:$qv, AR:$as, QR:$qx, QR:$qy, select_4:$sel4),
                   "ee.cmul.s16.st.incp\t $qv, $as, $qz, $qx, $qy, $sel4", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qv;
  bits<4> as;
  bits<3> qz;
  bits<3> qx;
  bits<3> qy;
  bits<2> sel4;
  
  let mayStore = 1;
  
  let Inst{28-20} = 0x1c8;
  let Inst{19-17} = qz{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-8} = qv{2-0};
  let Inst{7-6} = 0x0;
  let Inst{5-4} = sel4{1-0};
  let Inst{3-0} = as{3-0};

  let Constraints = "$asr = $as";
}

let usesCustomInserter = 1 in
def EE_CMUL_S16_ST_INCP_P : Pseudo<(outs), (ins imm8:$qv, AR:$as, imm8:$qz, imm8:$qx, imm8:$qy, select_4:$sel4),
				"!xtensa_ee_cmul_s16_st_incp_p, $qv, $as, $qz, $qx, $qy, $sel4",
			[(int_xtensa_ee_cmul_s16_st_incp timm:$qv, AR:$as, timm:$qz, timm:$qx, timm:$qy, timm:$sel4)]>;

def EE_FFT_AMS_S16_LD_INCP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qz, QR:$qz1), (ins AR:$as, QR:$qx, QR:$qy, QR:$qm, select_2:$sel2),
                   "ee.fft.ams.s16.ld.incp\t $qu, $as, $qz, $qz1, $qx, $qy, $qm, $sel2", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qz;
  bits<3> qz1;
  bits<3> qx;
  bits<3> qy;
  bits<3> qm;
  bits<1> sel2;
  
  let mayLoad = 1;
  
  let Inst{28-23} = 0x34;
  let Inst{22} = sel2{0};
  let Inst{21-20} = qz1{2-1};
  let Inst{19-17} = qm{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qz{2-0};
  let Inst{10-8} = qy{2-0};
  let Inst{7} = qz1{0};
  let Inst{6-4} = qu{2-0};
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_FFT_AMS_S16_LD_INCP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, imm8:$qz, imm8:$qz1, imm8:$qx, imm8:$qy, imm8:$qm, select_2:$sel2),
				"!xtensa_ee_fft_ams_s16_ld_incp_p, $qu, $as, $qz, $qz1, $qx, $qy, $qm, $sel2",
			[(int_xtensa_ee_fft_ams_s16_ld_incp timm:$qu, AR:$as, timm:$qz, timm:$qz1, timm:$qx, timm:$qy, timm:$qm, timm:$sel2)]>;

def EE_FFT_AMS_S16_LD_INCP_UAUP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qz, QR:$qz1), (ins AR:$as, QR:$qx, QR:$qy, QR:$qm, select_2:$sel2),
                   "ee.fft.ams.s16.ld.incp.uaup\t $qu, $as, $qz, $qz1, $qx, $qy, $qm, $sel2", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qz;
  bits<3> qz1;
  bits<3> qx;
  bits<3> qy;
  bits<3> qm;
  bits<1> sel2;
  
  let mayLoad = 1;
  
  let Inst{28-23} = 0x35;
  let Inst{22} = sel2{0};
  let Inst{21-20} = qz1{2-1};
  let Inst{19-17} = qm{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qz{2-0};
  let Inst{10-8} = qy{2-0};
  let Inst{7} = qz1{0};
  let Inst{6-4} = qu{2-0};
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_FFT_AMS_S16_LD_INCP_UAUP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, imm8:$qz, imm8:$qz1, imm8:$qx, imm8:$qy, imm8:$qm, select_2:$sel2),
				"!xtensa_ee_fft_ams_s16_ld_incp_uaup_p, $qu, $as, $qz, $qz1, $qx, $qy, $qm, $sel2",
			[(int_xtensa_ee_fft_ams_s16_ld_incp_uaup timm:$qu, AR:$as, timm:$qz, timm:$qz1, timm:$qx, timm:$qy, timm:$qm, timm:$sel2)]>;

def EE_FFT_AMS_S16_LD_R32_DECP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qz, QR:$qz1), (ins AR:$as, QR:$qx, QR:$qy, QR:$qm, select_2:$sel2),
                   "ee.fft.ams.s16.ld.r32.decp\t $qu, $as, $qz, $qz1, $qx, $qy, $qm, $sel2", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qz;
  bits<3> qz1;
  bits<3> qx;
  bits<3> qy;
  bits<3> qm;
  bits<1> sel2;
  
  let mayLoad = 1;
  
  let Inst{28-23} = 0x36;
  let Inst{22} = sel2{0};
  let Inst{21-20} = qz1{2-1};
  let Inst{19-17} = qm{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qz{2-0};
  let Inst{10-8} = qy{2-0};
  let Inst{7} = qz1{0};
  let Inst{6-4} = qu{2-0};
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_FFT_AMS_S16_LD_R32_DECP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, imm8:$qz, imm8:$qz1, imm8:$qx, imm8:$qy, imm8:$qm, select_2:$sel2),
				"!xtensa_ee_fft_ams_s16_ld_r32_decp_p, $qu, $as, $qz, $qz1, $qx, $qy, $qm, $sel2",
			[(int_xtensa_ee_fft_ams_s16_ld_r32_decp timm:$qu, AR:$as, timm:$qz, timm:$qz1, timm:$qx, timm:$qy, timm:$qm, timm:$sel2)]>;

def EE_FFT_AMS_S16_ST_INCP:  EE_Inst32<(outs QR:$qz1, AR:$as0r, AR:$asr), (ins QR:$qv, AR:$as0, AR:$as, QR:$qx, QR:$qy, QR:$qm, select_2:$sel2),
                   "ee.fft.ams.s16.st.incp\t $qv, $qz1, $as0, $as, $qx, $qy, $qm, $sel2", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qv;
  bits<3> qz1;
  bits<4> as0;
  bits<4> as;
  bits<3> qx;
  bits<3> qy;
  bits<3> qm;
  bits<1> sel2;
  
  let mayStore = 1;
  
  let Inst{28-24} = 0x14;
  let Inst{23} = sel2{0};
  let Inst{22-20} = qz1{2-0};
  let Inst{19-17} = qx{2-0};
  let Inst{16-14} = qy{2-0};
  let Inst{13-11} = qm{2-0};
  let Inst{10-8} = qv{2-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = as0{3-0};
}

let usesCustomInserter = 1 in
def EE_FFT_AMS_S16_ST_INCP_P : Pseudo<(outs), (ins imm8:$qv, imm8:$qz1, AR:$as0, AR:$as, imm8:$qx, imm8:$qy, imm8:$qm, select_2:$sel2),
				"!xtensa_ee_fft_ams_s16_st_incp_p, $qv, $qz1, $as0, $as, $qx, $qy, $qm, $sel2",
			[(int_xtensa_ee_fft_ams_s16_st_incp timm:$qv, timm:$qz1, AR:$as0, AR:$as, timm:$qx, timm:$qy, timm:$qm, timm:$sel2)]>;

def EE_FFT_CMUL_S16_LD_XP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qz), (ins AR:$as, AR:$ad, QR:$qx, QR:$qy, select_8:$sel8),
                   "ee.fft.cmul.s16.ld.xp\t $qu, $as, $ad, $qz, $qx, $qy, $sel8", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<4> ad;
  bits<3> qz;
  bits<3> qx;
  bits<3> qy;
  bits<3> sel8;
  
  let mayLoad = 1;
  
  let Inst{28-23} = 0x37;
  let Inst{22-20} = sel8{2-0};
  let Inst{19-17} = qz{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qu{2-0};
  let Inst{10-8} = qy{2-0};
  let Inst{7-4} = ad{3-0};
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_FFT_CMUL_S16_LD_XP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, AR:$ad, imm8:$qz, imm8:$qx, imm8:$qy, select_8:$sel8),
				"!xtensa_ee_fft_cmul_s16_ld_xp_p, $qu, $as, $ad, $qz, $qx, $qy, $sel8",
			[(int_xtensa_ee_fft_cmul_s16_ld_xp timm:$qu, AR:$as, AR:$ad, timm:$qz, timm:$qx, timm:$qy, timm:$sel8)]>;

def EE_FFT_CMUL_S16_ST_XP:  EE_Inst32<(outs AR:$asr), (ins QR:$qx, QR:$qy, QR:$qv, AR:$as, AR:$ad, select_8:$sel8, select_4:$upd4, select_4:$sar4),
                   "ee.fft.cmul.s16.st.xp\t $qx, $qy, $qv, $as, $ad, $sel8, $upd4, $sar4", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qx;
  bits<3> qy;
  bits<3> qv;
  bits<4> as;
  bits<4> ad;
  bits<3> sel8;
  bits<2> upd4;
  bits<2> sar4;
  
  let mayStore = 1;
  
  let Inst{28-24} = 0x15;
  let Inst{23-22} = sar4{1-0};
  let Inst{21-20} = upd4{1-0};
  let Inst{19-17} = qv{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = sel8{2-0};
  let Inst{10-8} = qy{2-0};
  let Inst{7-4} = ad{3-0};
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_FFT_CMUL_S16_ST_XP_P : Pseudo<(outs), (ins imm8:$qx, imm8:$qy, imm8:$qv, AR:$as, AR:$ad, select_8:$sel8, select_4:$upd4, select_4:$sar4),
				"!xtensa_ee_fft_cmul_s16_st_xp_p, $qx, $qy, $qv, $as, $ad, $sel8, $upd4, $sar4",
			[(int_xtensa_ee_fft_cmul_s16_st_xp timm:$qx, timm:$qy, timm:$qv, AR:$as, AR:$ad, timm:$sel8, timm:$upd4, timm:$sar4)]>;

def EE_FFT_R2BF_S16:  EE_Inst24<(outs QR:$qa0, QR:$qa1), (ins QR:$qx, QR:$qy, select_2:$sel2),
                   "ee.fft.r2bf.s16\t $qa0, $qa1, $qx, $qy, $sel2", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa0;
  bits<3> qa1;
  bits<3> qx;
  bits<3> qy;
  bits<1> sel2;
  
  
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qa0{2-1};
  let Inst{19-16} = 0xc;
  let Inst{15} = qa0{0};
  let Inst{14-12} = qa1{2-0};
  let Inst{11-10} = qy{2-1};
  let Inst{9} = 0x0;
  let Inst{8} = sel2{0};
  let Inst{7-6} = qx{2-1};
  let Inst{5} = qy{0};
  let Inst{4} = qx{0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_FFT_R2BF_S16_P : Pseudo<(outs), (ins imm8:$qa0, imm8:$qa1, imm8:$qx, imm8:$qy, select_2:$sel2),
				"!xtensa_ee_fft_r2bf_s16_p, $qa0, $qa1, $qx, $qy, $sel2",
			[(int_xtensa_ee_fft_r2bf_s16 timm:$qa0, timm:$qa1, timm:$qx, timm:$qy, timm:$sel2)]>;

def EE_FFT_R2BF_S16_ST_INCP:  EE_Inst32<(outs QR:$qa0, AR:$asr), (ins QR:$qx, QR:$qy, AR:$as, select_4:$sar4),
                   "ee.fft.r2bf.s16.st.incp\t $qa0, $qx, $qy, $as, $sar4", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa0;
  bits<3> qx;
  bits<3> qy;
  bits<4> as;
  bits<2> sar4;
  
  let mayStore = 1;
  
  let Inst{28-20} = 0x1d1;
  let Inst{19-17} = qa0{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13} = 0x0;
  let Inst{12-11} = sar4{1-0};
  let Inst{10-8} = qy{2-0};
  let Inst{7-4} = 0x4;
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_FFT_R2BF_S16_ST_INCP_P : Pseudo<(outs), (ins imm8:$qa0, imm8:$qx, imm8:$qy, AR:$as, select_4:$sar4),
				"!xtensa_ee_fft_r2bf_s16_st_incp_p, $qa0, $qx, $qy, $as, $sar4",
			[(int_xtensa_ee_fft_r2bf_s16_st_incp timm:$qa0, timm:$qx, timm:$qy, AR:$as, timm:$sar4)]>;

def EE_FFT_VST_R32_DECP:  EE_Inst24<(outs AR:$asr), (ins QR:$qv, AR:$as, select_2:$sar2),
                   "ee.fft.vst.r32.decp\t $qv, $as, $sar2", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qv;
  bits<4> as;
  bits<1> sar2;
  
  let mayStore = 1;
  
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qv{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qv{0};
  let Inst{14-11} = 0x6;
  let Inst{10} = sar2{0};
  let Inst{9-8} = 0x3;
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_FFT_VST_R32_DECP_P : Pseudo<(outs), (ins imm8:$qv, AR:$as, select_2:$sar2),
				"!xtensa_ee_fft_vst_r32_decp_p, $qv, $as, $sar2",
			[(int_xtensa_ee_fft_vst_r32_decp timm:$qv, AR:$as, timm:$sar2)]>;

def EE_LDF_128_IP:  EE_Inst32<(outs FPR:$fu3, FPR:$fu2, FPR:$fu1, FPR:$fu0, AR:$asr), (ins AR:$as, offset_16_16:$imm16f),
                   "ee.ldf.128.ip\t $fu3, $fu2, $fu1, $fu0, $as, $imm16f", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> fu3;
  bits<4> fu2;
  bits<4> fu1;
  bits<4> fu0;
  bits<4> as;
  bits<4> imm16f;
  
  let mayLoad = 1;
  
  let Inst{28-24} = 0x10;
  let Inst{23-20} = fu3{3-0};
  let Inst{19-16} = fu2{3-0};
  let Inst{15-12} = fu1{3-0};
  let Inst{11-8} = fu0{3-0};
  let Inst{7-4} = imm16f{3-0};
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_LDF_128_IP_P : Pseudo<(outs), (ins FPR:$fu3, FPR:$fu2, FPR:$fu1, FPR:$fu0, AR:$as, offset_16_16:$imm16f),
				"!xtensa_ee_ldf_128_ip_p, $fu3, $fu2, $fu1, $fu0, $as, $imm16f",
			[(int_xtensa_ee_ldf_128_ip FPR:$fu3, FPR:$fu2, FPR:$fu1, FPR:$fu0, AR:$as, timm:$imm16f)]>;

def EE_LDF_128_XP:  EE_Inst32<(outs FPR:$fu3, FPR:$fu2, FPR:$fu1, FPR:$fu0, AR:$asr), (ins AR:$as, AR:$ad),
                   "ee.ldf.128.xp\t $fu3, $fu2, $fu1, $fu0, $as, $ad", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> fu3;
  bits<4> fu2;
  bits<4> fu1;
  bits<4> fu0;
  bits<4> as;
  bits<4> ad;
  
  let mayLoad = 1;
  
  let Inst{28-24} = 0x11;
  let Inst{23-20} = fu3{3-0};
  let Inst{19-16} = fu2{3-0};
  let Inst{15-12} = fu1{3-0};
  let Inst{11-8} = fu0{3-0};
  let Inst{7-4} = ad{3-0};
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_LDF_128_XP_P : Pseudo<(outs), (ins FPR:$fu3, FPR:$fu2, FPR:$fu1, FPR:$fu0, AR:$as, AR:$ad),
				"!xtensa_ee_ldf_128_xp_p, $fu3, $fu2, $fu1, $fu0, $as, $ad",
			[(int_xtensa_ee_ldf_128_xp FPR:$fu3, FPR:$fu2, FPR:$fu1, FPR:$fu0, AR:$as, AR:$ad)]>;

def EE_LDF_64_IP:  EE_Inst32<(outs FPR:$fu1, FPR:$fu0, AR:$asr), (ins AR:$as, offset_256_8:$imm8),
                   "ee.ldf.64.ip\t $fu1, $fu0, $as, $imm8", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> fu1;
  bits<4> fu0;
  bits<4> as;
  bits<8> imm8;
  
  let mayLoad = 1;
  
  let Inst{28-23} = 0x38;
  let Inst{22-16} = imm8{7-1};
  let Inst{15-12} = fu1{3-0};
  let Inst{11-8} = fu0{3-0};
  let Inst{7-5} = 0x2;
  let Inst{4} = imm8{0};
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_LDF_64_IP_P : Pseudo<(outs), (ins FPR:$fu1, FPR:$fu0, AR:$as, offset_256_8:$imm8),
				"!xtensa_ee_ldf_64_ip_p, $fu1, $fu0, $as, $imm8",
			[(int_xtensa_ee_ldf_64_ip FPR:$fu1, FPR:$fu0, AR:$as, timm:$imm8)]>;

def EE_LDF_64_XP:  EE_Inst24<(outs FPR:$fu1, FPR:$fu0, AR:$asr), (ins AR:$as, AR:$ad),
                   "ee.ldf.64.xp\t $fu1, $fu0, $as, $ad", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> fu1;
  bits<4> fu0;
  bits<4> as;
  bits<4> ad;
  
  let mayLoad = 1;
  
  let Inst{23-20} = fu0{3-0};
  let Inst{19-16} = 0x6;
  let Inst{15-12} = fu1{3-0};
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x0;
}

let usesCustomInserter = 1 in
def EE_LDF_64_XP_P : Pseudo<(outs), (ins FPR:$fu1, FPR:$fu0, AR:$as, AR:$ad),
				"!xtensa_ee_ldf_64_xp_p, $fu1, $fu0, $as, $ad",
			[(int_xtensa_ee_ldf_64_xp FPR:$fu1, FPR:$fu0, AR:$as, AR:$ad)]>;

def EE_LDQA_S16_128_IP:  EE_Inst24<(outs AR:$asr), (ins AR:$as, offset_256_16:$imm16),
                   "ee.ldqa.s16.128.ip\t $as, $imm16", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> as;
  bits<8> imm16;
  
  let mayLoad = 1;
  
  let Inst{23} = 0x0;
  let Inst{22} = imm16{7};
  let Inst{21-15} = 0x2;
  let Inst{14-8} = imm16{6-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_LDQA_S16_128_IP_P : Pseudo<(outs), (ins AR:$as, offset_256_16:$imm16),
				"!xtensa_ee_ldqa_s16_128_ip_p, $as, $imm16",
			[(int_xtensa_ee_ldqa_s16_128_ip AR:$as, timm:$imm16)]>;

def EE_LDQA_S16_128_XP:  EE_Inst24<(outs AR:$asr), (ins AR:$as, AR:$ad),
                   "ee.ldqa.s16.128.xp\t $as, $ad", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> as;
  bits<4> ad;
  
  let mayLoad = 1;
  
  let Inst{23-12} = 0x7e4;
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_LDQA_S16_128_XP_P : Pseudo<(outs), (ins AR:$as, AR:$ad),
				"!xtensa_ee_ldqa_s16_128_xp_p, $as, $ad",
			[(int_xtensa_ee_ldqa_s16_128_xp AR:$as, AR:$ad)]>;

def EE_LDQA_S8_128_IP:  EE_Inst24<(outs AR:$asr), (ins AR:$as, offset_256_16:$imm16),
                   "ee.ldqa.s8.128.ip\t $as, $imm16", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> as;
  bits<8> imm16;
  
  let mayLoad = 1;
  
  let Inst{23} = 0x0;
  let Inst{22} = imm16{7};
  let Inst{21-15} = 0x22;
  let Inst{14-8} = imm16{6-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_LDQA_S8_128_IP_P : Pseudo<(outs), (ins AR:$as, offset_256_16:$imm16),
				"!xtensa_ee_ldqa_s8_128_ip_p, $as, $imm16",
			[(int_xtensa_ee_ldqa_s8_128_ip AR:$as, timm:$imm16)]>;

def EE_LDQA_S8_128_XP:  EE_Inst24<(outs AR:$asr), (ins AR:$as, AR:$ad),
                   "ee.ldqa.s8.128.xp\t $as, $ad", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> as;
  bits<4> ad;
  
  let mayLoad = 1;
  
  let Inst{23-12} = 0x714;
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_LDQA_S8_128_XP_P : Pseudo<(outs), (ins AR:$as, AR:$ad),
				"!xtensa_ee_ldqa_s8_128_xp_p, $as, $ad",
			[(int_xtensa_ee_ldqa_s8_128_xp AR:$as, AR:$ad)]>;

def EE_LDQA_U16_128_IP:  EE_Inst24<(outs AR:$asr), (ins AR:$as, offset_256_16:$imm16),
                   "ee.ldqa.u16.128.ip\t $as, $imm16", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> as;
  bits<8> imm16;
  
  let mayLoad = 1;
  
  let Inst{23} = 0x0;
  let Inst{22} = imm16{7};
  let Inst{21-15} = 0xa;
  let Inst{14-8} = imm16{6-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_LDQA_U16_128_IP_P : Pseudo<(outs), (ins AR:$as, offset_256_16:$imm16),
				"!xtensa_ee_ldqa_u16_128_ip_p, $as, $imm16",
			[(int_xtensa_ee_ldqa_u16_128_ip AR:$as, timm:$imm16)]>;

def EE_LDQA_U16_128_XP:  EE_Inst24<(outs AR:$asr), (ins AR:$as, AR:$ad),
                   "ee.ldqa.u16.128.xp\t $as, $ad", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> as;
  bits<4> ad;
  
  let mayLoad = 1;
  
  let Inst{23-12} = 0x7a4;
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_LDQA_U16_128_XP_P : Pseudo<(outs), (ins AR:$as, AR:$ad),
				"!xtensa_ee_ldqa_u16_128_xp_p, $as, $ad",
			[(int_xtensa_ee_ldqa_u16_128_xp AR:$as, AR:$ad)]>;

def EE_LDQA_U8_128_IP:  EE_Inst24<(outs AR:$asr), (ins AR:$as, offset_256_16:$imm16),
                   "ee.ldqa.u8.128.ip\t $as, $imm16", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> as;
  bits<8> imm16;
  
  let mayLoad = 1;
  
  let Inst{23} = 0x0;
  let Inst{22} = imm16{7};
  let Inst{21-15} = 0x2a;
  let Inst{14-8} = imm16{6-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_LDQA_U8_128_IP_P : Pseudo<(outs), (ins AR:$as, offset_256_16:$imm16),
				"!xtensa_ee_ldqa_u8_128_ip_p, $as, $imm16",
			[(int_xtensa_ee_ldqa_u8_128_ip AR:$as, timm:$imm16)]>;

def EE_LDQA_U8_128_XP:  EE_Inst24<(outs AR:$asr), (ins AR:$as, AR:$ad),
                   "ee.ldqa.u8.128.xp\t $as, $ad", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> as;
  bits<4> ad;
  
  let mayLoad = 1;
  
  let Inst{23-12} = 0x704;
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_LDQA_U8_128_XP_P : Pseudo<(outs), (ins AR:$as, AR:$ad),
				"!xtensa_ee_ldqa_u8_128_xp_p, $as, $ad",
			[(int_xtensa_ee_ldqa_u8_128_xp AR:$as, AR:$ad)]>;

def EE_LDXQ_32:  EE_Inst32<(outs QR:$qu), (ins QR:$qs, AR:$as, select_4:$sel4, select_8:$sel8),
                   "ee.ldxq.32\t $qu, $qs, $as, $sel4, $sel8", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<3> qs;
  bits<4> as;
  bits<2> sel4;
  bits<3> sel8;
  
  let mayLoad = 1;
  
  let Inst{28-22} = 0x70;
  let Inst{21-20} = sel4{1-0};
  let Inst{19-17} = qu{2-0};
  let Inst{16-14} = qs{2-0};
  let Inst{13-11} = sel8{2-0};
  let Inst{10-4} = 0x7d;
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_LDXQ_32_P : Pseudo<(outs), (ins imm8:$qu, imm8:$qs, AR:$as, select_4:$sel4, select_8:$sel8),
				"!xtensa_ee_ldxq_32_p, $qu, $qs, $as, $sel4, $sel8",
			[(int_xtensa_ee_ldxq_32 timm:$qu, timm:$qs, AR:$as, timm:$sel4, timm:$sel8)]>;

def EE_LD_128_USAR_IP:  EE_Inst24<(outs QR:$qu, AR:$asr), (ins AR:$as, offset_256_16:$imm16),
                   "ee.ld.128.usar.ip\t $qu, $as, $imm16", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<8> imm16;
  
  let mayLoad = 1;
  
  let Inst{23} = 0x1;
  let Inst{22} = imm16{7};
  let Inst{21-20} = qu{2-1};
  let Inst{19-16} = 0x1;
  let Inst{15} = qu{0};
  let Inst{14-8} = imm16{6-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_LD_128_USAR_IP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, offset_256_16:$imm16),
				"!xtensa_ee_ld_128_usar_ip_p, $qu, $as, $imm16",
			[(int_xtensa_ee_ld_128_usar_ip timm:$qu, AR:$as, timm:$imm16)]>;

def EE_LD_128_USAR_XP:  EE_Inst24<(outs QR:$qu, AR:$asr), (ins AR:$as, AR:$ad),
                   "ee.ld.128.usar.xp\t $qu, $as, $ad", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<4> ad;
  
  let mayLoad = 1;
  
  let Inst{23-22} = 0x2;
  let Inst{21-20} = qu{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qu{0};
  let Inst{14-12} = 0x0;
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_LD_128_USAR_XP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, AR:$ad),
				"!xtensa_ee_ld_128_usar_xp_p, $qu, $as, $ad",
			[(int_xtensa_ee_ld_128_usar_xp timm:$qu, AR:$as, AR:$ad)]>;

def EE_LD_ACCX_IP:  EE_Inst24<(outs AR:$asr), (ins AR:$as, offset_256_8:$imm8),
                   "ee.ld.accx.ip\t $as, $imm8", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> as;
  bits<8> imm8;
  
  let mayLoad = 1;
  
  let Inst{23} = 0x0;
  let Inst{22} = imm8{7};
  let Inst{21-15} = 0x1c;
  let Inst{14-8} = imm8{6-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_LD_ACCX_IP_P : Pseudo<(outs), (ins AR:$as, offset_256_8:$imm8),
				"!xtensa_ee_ld_accx_ip_p, $as, $imm8",
			[(int_xtensa_ee_ld_accx_ip AR:$as, timm:$imm8)]>;

def EE_LD_QACC_H_H_32_IP:  EE_Inst24<(outs AR:$asr), (ins AR:$as, offset_256_4:$imm4),
                   "ee.ld.qacc_h.h.32.ip\t $as, $imm4", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> as;
  bits<8> imm4;
  
  let mayLoad = 1;
  
  let Inst{23} = 0x0;
  let Inst{22} = imm4{7};
  let Inst{21-15} = 0x3c;
  let Inst{14-8} = imm4{6-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_LD_QACC_H_H_32_IP_P : Pseudo<(outs), (ins AR:$as, offset_256_4:$imm4),
				"!xtensa_ee_ld_qacc_h_h_32_ip_p, $as, $imm4",
			[(int_xtensa_ee_ld_qacc_h_h_32_ip AR:$as, timm:$imm4)]>;

def EE_LD_QACC_H_L_128_IP:  EE_Inst24<(outs AR:$asr), (ins AR:$as, offset_256_16:$imm16),
                   "ee.ld.qacc_h.l.128.ip\t $as, $imm16", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> as;
  bits<8> imm16;
  
  let mayLoad = 1;
  
  let Inst{23} = 0x0;
  let Inst{22} = imm16{7};
  let Inst{21-15} = 0xc;
  let Inst{14-8} = imm16{6-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_LD_QACC_H_L_128_IP_P : Pseudo<(outs), (ins AR:$as, offset_256_16:$imm16),
				"!xtensa_ee_ld_qacc_h_l_128_ip_p, $as, $imm16",
			[(int_xtensa_ee_ld_qacc_h_l_128_ip AR:$as, timm:$imm16)]>;

def EE_LD_QACC_L_H_32_IP:  EE_Inst24<(outs AR:$asr), (ins AR:$as, offset_256_4:$imm4),
                   "ee.ld.qacc_l.h.32.ip\t $as, $imm4", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> as;
  bits<8> imm4;
  
  let mayLoad = 1;
  
  let Inst{23} = 0x0;
  let Inst{22} = imm4{7};
  let Inst{21-15} = 0x2c;
  let Inst{14-8} = imm4{6-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_LD_QACC_L_H_32_IP_P : Pseudo<(outs), (ins AR:$as, offset_256_4:$imm4),
				"!xtensa_ee_ld_qacc_l_h_32_ip_p, $as, $imm4",
			[(int_xtensa_ee_ld_qacc_l_h_32_ip AR:$as, timm:$imm4)]>;

def EE_LD_QACC_L_L_128_IP:  EE_Inst24<(outs AR:$asr), (ins AR:$as, offset_256_16:$imm16),
                   "ee.ld.qacc_l.l.128.ip\t $as, $imm16", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> as;
  bits<8> imm16;
  
  let mayLoad = 1;
  
  let Inst{23} = 0x0;
  let Inst{22} = imm16{7};
  let Inst{21-15} = 0x0;
  let Inst{14-8} = imm16{6-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_LD_QACC_L_L_128_IP_P : Pseudo<(outs), (ins AR:$as, offset_256_16:$imm16),
				"!xtensa_ee_ld_qacc_l_l_128_ip_p, $as, $imm16",
			[(int_xtensa_ee_ld_qacc_l_l_128_ip AR:$as, timm:$imm16)]>;

def EE_LD_UA_STATE_IP:  EE_Inst24<(outs AR:$asr), (ins AR:$as, offset_256_16:$imm16),
                   "ee.ld.ua_state.ip\t $as, $imm16", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> as;
  bits<8> imm16;
  
  let mayLoad = 1;
  
  let Inst{23} = 0x0;
  let Inst{22} = imm16{7};
  let Inst{21-15} = 0x20;
  let Inst{14-8} = imm16{6-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_LD_UA_STATE_IP_P : Pseudo<(outs), (ins AR:$as, offset_256_16:$imm16),
				"!xtensa_ee_ld_ua_state_ip_p, $as, $imm16",
			[(int_xtensa_ee_ld_ua_state_ip AR:$as, timm:$imm16)]>;

def EE_MOVI_32_A:  EE_Inst24<(outs AR:$au), (ins QR:$qs, select_4:$sel4),
                   "ee.movi.32.a\t $qs, $au, $sel4", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qs;
  bits<4> au;
  bits<2> sel4;
  
  
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qs{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qs{0};
  let Inst{14-12} = 0x7;
  let Inst{11-10} = sel4{1-0};
  let Inst{9-8} = 0x1;
  let Inst{7-4} = au{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_MOVI_32_A_P : Pseudo<(outs), (ins imm8:$qs, AR:$au, select_4:$sel4),
				"!xtensa_ee_movi_32_a_p, $qs, $au, $sel4",
			[(int_xtensa_ee_movi_32_a timm:$qs, AR:$au, timm:$sel4)]>;

def EE_MOVI_32_Q:  EE_Inst24<(outs QR:$qu), (ins AR:$as, select_4:$sel4),
                   "ee.movi.32.q\t $qu, $as, $sel4", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<2> sel4;
  
  
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qu{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qu{0};
  let Inst{14-12} = 0x3;
  let Inst{11-10} = sel4{1-0};
  let Inst{9-8} = 0x2;
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_MOVI_32_Q_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, select_4:$sel4),
				"!xtensa_ee_movi_32_q_p, $qu, $as, $sel4",
			[(int_xtensa_ee_movi_32_q timm:$qu, AR:$as, timm:$sel4)]>;

def EE_MOV_S16_QACC:  EE_Inst24<(outs), (ins QR:$qs),
                   "ee.mov.s16.qacc\t $qs", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qs;
  
  
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qs{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qs{0};
  let Inst{14-0} = 0x7f24;
}

let usesCustomInserter = 1 in
def EE_MOV_S16_QACC_P : Pseudo<(outs), (ins imm8:$qs),
				"!xtensa_ee_mov_s16_qacc_p, $qs",
			[(int_xtensa_ee_mov_s16_qacc timm:$qs)]>;

def EE_MOV_S8_QACC:  EE_Inst24<(outs), (ins QR:$qs),
                   "ee.mov.s8.qacc\t $qs", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qs;
  
  
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qs{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qs{0};
  let Inst{14-0} = 0x7f34;
}

let usesCustomInserter = 1 in
def EE_MOV_S8_QACC_P : Pseudo<(outs), (ins imm8:$qs),
				"!xtensa_ee_mov_s8_qacc_p, $qs",
			[(int_xtensa_ee_mov_s8_qacc timm:$qs)]>;

def EE_MOV_U16_QACC:  EE_Inst24<(outs), (ins QR:$qs),
                   "ee.mov.u16.qacc\t $qs", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qs;
  
  
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qs{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qs{0};
  let Inst{14-0} = 0x7f64;
}

let usesCustomInserter = 1 in
def EE_MOV_U16_QACC_P : Pseudo<(outs), (ins imm8:$qs),
				"!xtensa_ee_mov_u16_qacc_p, $qs",
			[(int_xtensa_ee_mov_u16_qacc timm:$qs)]>;

def EE_MOV_U8_QACC:  EE_Inst24<(outs), (ins QR:$qs),
                   "ee.mov.u8.qacc\t $qs", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qs;
  
  
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qs{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qs{0};
  let Inst{14-0} = 0x7f74;
}

let usesCustomInserter = 1 in
def EE_MOV_U8_QACC_P : Pseudo<(outs), (ins imm8:$qs),
				"!xtensa_ee_mov_u8_qacc_p, $qs",
			[(int_xtensa_ee_mov_u8_qacc timm:$qs)]>;

def EE_NOTQ:  EE_Inst24<(outs QR:$qa), (ins QR:$qx),
                   "ee.notq\t $qa, $qx", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qx;
  
  
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qa{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qa{0};
  let Inst{14-8} = 0x7f;
  let Inst{7-6} = qx{2-1};
  let Inst{5} = 0x0;
  let Inst{4} = qx{0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_NOTQ_P : Pseudo<(outs), (ins imm8:$qa, imm8:$qx),
				"!xtensa_ee_notq_p, $qa, $qx",
			[(int_xtensa_ee_notq timm:$qa, timm:$qx)]>;

def EE_ORQ:  EE_Inst24<(outs QR:$qa), (ins QR:$qx, QR:$qy),
                   "ee.orq\t $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;
  
  
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qa{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qa{0};
  let Inst{14-12} = 0x7;
  let Inst{11-10} = qy{2-1};
  let Inst{9-8} = 0x0;
  let Inst{7-6} = qx{2-1};
  let Inst{5} = qy{0};
  let Inst{4} = qx{0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_ORQ_P : Pseudo<(outs), (ins imm8:$qa, imm8:$qx, imm8:$qy),
				"!xtensa_ee_orq_p, $qa, $qx, $qy",
			[(int_xtensa_ee_orq timm:$qa, timm:$qx, timm:$qy)]>;

def EE_SLCI_2Q:  EE_Inst24<(outs QR:$qs1r, QR:$qs0r), (ins QR:$qs1, QR:$qs0, select_16:$sar16),
                   "ee.slci.2q\t $qs1, $qs0, $sar16", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qs1;
  bits<3> qs0;
  bits<4> sar16;
  
  
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qs1{2-1};
  let Inst{19-16} = 0xc;
  let Inst{15} = qs1{0};
  let Inst{14-12} = qs0{2-0};
  let Inst{11-8} = 0x6;
  let Inst{7-4} = sar16{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_SLCI_2Q_P : Pseudo<(outs), (ins imm8:$qs1, imm8:$qs0, select_16:$sar16),
				"!xtensa_ee_slci_2q_p, $qs1, $qs0, $sar16",
			[(int_xtensa_ee_slci_2q timm:$qs1, timm:$qs0, timm:$sar16)]>;

def EE_SLCXXP_2Q:  EE_Inst24<(outs QR:$qs1r, QR:$qs0r, AR:$asr), (ins QR:$qs1, QR:$qs0, AR:$as, AR:$ad),
                   "ee.slcxxp.2q\t $qs1, $qs0, $as, $ad", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qs1;
  bits<3> qs0;
  bits<4> as;
  bits<4> ad;
  
  
  let Inst{23-22} = 0x2;
  let Inst{21-20} = qs1{2-1};
  let Inst{19-16} = 0x6;
  let Inst{15} = qs1{0};
  let Inst{14-12} = qs0{2-0};
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_SLCXXP_2Q_P : Pseudo<(outs), (ins imm8:$qs1, imm8:$qs0, AR:$as, AR:$ad),
				"!xtensa_ee_slcxxp_2q_p, $qs1, $qs0, $as, $ad",
			[(int_xtensa_ee_slcxxp_2q timm:$qs1, timm:$qs0, AR:$as, AR:$ad)]>;

def EE_SRCI_2Q:  EE_Inst24<(outs QR:$qs1r, QR:$qs0r), (ins QR:$qs1, QR:$qs0, select_16:$sar16),
                   "ee.srci.2q\t $qs1, $qs0, $sar16", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qs1;
  bits<3> qs0;
  bits<4> sar16;
  
  
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qs1{2-1};
  let Inst{19-16} = 0xc;
  let Inst{15} = qs1{0};
  let Inst{14-12} = qs0{2-0};
  let Inst{11-8} = 0xa;
  let Inst{7-4} = sar16{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_SRCI_2Q_P : Pseudo<(outs), (ins imm8:$qs1, imm8:$qs0, select_16:$sar16),
				"!xtensa_ee_srci_2q_p, $qs1, $qs0, $sar16",
			[(int_xtensa_ee_srci_2q timm:$qs1, timm:$qs0, timm:$sar16)]>;

def EE_SRCMB_S16_QACC:  EE_Inst24<(outs QR:$qu), (ins AR:$as, select_2:$sel2),
                   "ee.srcmb.s16.qacc\t $qu, $as, $sel2", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<1> sel2;
  
  
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qu{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qu{0};
  let Inst{14-11} = 0xe;
  let Inst{10} = sel2{0};
  let Inst{9-8} = 0x2;
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_SRCMB_S16_QACC_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, select_2:$sel2),
				"!xtensa_ee_srcmb_s16_qacc_p, $qu, $as, $sel2",
			[(int_xtensa_ee_srcmb_s16_qacc timm:$qu, AR:$as, timm:$sel2)]>;

def EE_SRCMB_S8_QACC:  EE_Inst24<(outs QR:$qu), (ins AR:$as, select_2:$sel2),
                   "ee.srcmb.s8.qacc\t $qu, $as, $sel2", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<1> sel2;
  
  
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qu{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qu{0};
  let Inst{14-11} = 0xf;
  let Inst{10} = sel2{0};
  let Inst{9-8} = 0x2;
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_SRCMB_S8_QACC_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, select_2:$sel2),
				"!xtensa_ee_srcmb_s8_qacc_p, $qu, $as, $sel2",
			[(int_xtensa_ee_srcmb_s8_qacc timm:$qu, AR:$as, timm:$sel2)]>;

def EE_SRCQ_128_ST_INCP:  EE_Inst24<(outs AR:$asr), (ins QR:$qs0, QR:$qs1, AR:$as),
                   "ee.srcq.128.st.incp\t $qs0, $qs1, $as", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qs0;
  bits<3> qs1;
  bits<4> as;
  
  let mayStore = 1;
  
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qs1{2-1};
  let Inst{19-16} = 0xc;
  let Inst{15} = qs1{0};
  let Inst{14-12} = qs0{2-0};
  let Inst{11-8} = 0xe;
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_SRCQ_128_ST_INCP_P : Pseudo<(outs), (ins imm8:$qs0, imm8:$qs1, AR:$as),
				"!xtensa_ee_srcq_128_st_incp_p, $qs0, $qs1, $as",
			[(int_xtensa_ee_srcq_128_st_incp timm:$qs0, timm:$qs1, AR:$as)]>;

def EE_SRCXXP_2Q:  EE_Inst24<(outs QR:$qs1r, QR:$qs0r, AR:$asr), (ins QR:$qs1, QR:$qs0, AR:$as, AR:$ad),
                   "ee.srcxxp.2q\t $qs1, $qs0, $as, $ad", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qs1;
  bits<3> qs0;
  bits<4> as;
  bits<4> ad;
  
  
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qs1{2-1};
  let Inst{19-16} = 0x6;
  let Inst{15} = qs1{0};
  let Inst{14-12} = qs0{2-0};
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_SRCXXP_2Q_P : Pseudo<(outs), (ins imm8:$qs1, imm8:$qs0, AR:$as, AR:$ad),
				"!xtensa_ee_srcxxp_2q_p, $qs1, $qs0, $as, $ad",
			[(int_xtensa_ee_srcxxp_2q timm:$qs1, timm:$qs0, AR:$as, AR:$ad)]>;

def EE_SRC_Q:  EE_Inst24<(outs QR:$qa), (ins QR:$qs0, QR:$qs1),
                   "ee.src.q\t $qa, $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qs0;
  bits<3> qs1;
  
  
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qs1{2-1};
  let Inst{19-16} = 0xc;
  let Inst{15} = qs1{0};
  let Inst{14-12} = qs0{2-0};
  let Inst{11-7} = 0x6;
  let Inst{6-4} = qa{2-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_SRC_Q_P : Pseudo<(outs), (ins imm8:$qa, imm8:$qs0, imm8:$qs1),
				"!xtensa_ee_src_q_p, $qa, $qs0, $qs1",
			[(int_xtensa_ee_src_q timm:$qa, timm:$qs0, timm:$qs1)]>;

def EE_SRC_Q_LD_IP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qs0r), (ins AR:$as, offset_256_16:$imm16, QR:$qs0, QR:$qs1),
                   "ee.src.q.ld.ip\t $qu, $as, $imm16, $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<8> imm16;
  bits<3> qs0;
  bits<3> qs1;
  
  let mayLoad = 1;
  
  let Inst{28-23} = 0x38;
  let Inst{22-20} = imm16{7-5};
  let Inst{19-17} = qu{2-0};
  let Inst{16-14} = qs0{2-0};
  let Inst{13-11} = imm16{4-2};
  let Inst{10-8} = qs1{2-0};
  let Inst{7-6} = 0x0;
  let Inst{5-4} = imm16{1-0};
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_SRC_Q_LD_IP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, offset_256_16:$imm16, imm8:$qs0, imm8:$qs1),
				"!xtensa_ee_src_q_ld_ip_p, $qu, $as, $imm16, $qs0, $qs1",
			[(int_xtensa_ee_src_q_ld_ip timm:$qu, AR:$as, timm:$imm16, timm:$qs0, timm:$qs1)]>;

def EE_SRC_Q_LD_XP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qs0r), (ins AR:$as, AR:$ad, QR:$qs0, QR:$qs1),
                   "ee.src.q.ld.xp\t $qu, $as, $ad, $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<4> ad;
  bits<3> qs0;
  bits<3> qs1;
  
  let mayLoad = 1;
  
  let Inst{28-20} = 0x1d0;
  let Inst{19-17} = qu{2-0};
  let Inst{16-14} = qs0{2-0};
  let Inst{13-11} = 0x0;
  let Inst{10-8} = qs1{2-0};
  let Inst{7-4} = ad{3-0};
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_SRC_Q_LD_XP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, AR:$ad, imm8:$qs0, imm8:$qs1),
				"!xtensa_ee_src_q_ld_xp_p, $qu, $as, $ad, $qs0, $qs1",
			[(int_xtensa_ee_src_q_ld_xp timm:$qu, AR:$as, AR:$ad, timm:$qs0, timm:$qs1)]>;

def EE_SRC_Q_QUP:  EE_Inst24<(outs QR:$qa, QR:$qs0r), (ins QR:$qs0, QR:$qs1),
                   "ee.src.q.qup\t $qa, $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qs0;
  bits<3> qs1;
  
  
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qs1{2-1};
  let Inst{19-16} = 0xc;
  let Inst{15} = qs1{0};
  let Inst{14-12} = qs0{2-0};
  let Inst{11-7} = 0xe;
  let Inst{6-4} = qa{2-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_SRC_Q_QUP_P : Pseudo<(outs), (ins imm8:$qa, imm8:$qs0, imm8:$qs1),
				"!xtensa_ee_src_q_qup_p, $qa, $qs0, $qs1",
			[(int_xtensa_ee_src_q_qup timm:$qa, timm:$qs0, timm:$qs1)]>;

def EE_SRS_ACCX:  EE_Inst24<(outs AR:$au), (ins AR:$as, select_2:$sel2),
                   "ee.srs.accx\t $au, $as, $sel2", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> au;
  bits<4> as;
  bits<1> sel2;
  
  
  let Inst{23-15} = 0xfc;
  let Inst{14} = sel2{0};
  let Inst{13-12} = 0x1;
  let Inst{11-8} = au{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_SRS_ACCX_P : Pseudo<(outs), (ins AR:$au, AR:$as, select_2:$sel2),
				"!xtensa_ee_srs_accx_p, $au, $as, $sel2",
			[(int_xtensa_ee_srs_accx AR:$au, AR:$as, timm:$sel2)]>;

def EE_STF_128_IP:  EE_Inst32<(outs AR:$asr), (ins FPR:$fv3, FPR:$fv2, FPR:$fv1, FPR:$fv0, AR:$as, offset_16_16:$imm16f),
                   "ee.stf.128.ip\t $fv3, $fv2, $fv1, $fv0, $as, $imm16f", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> fv3;
  bits<4> fv2;
  bits<4> fv1;
  bits<4> fv0;
  bits<4> as;
  bits<4> imm16f;
  
  let mayStore = 1;
  
  let Inst{28-24} = 0x12;
  let Inst{23-20} = fv3{3-0};
  let Inst{19-16} = fv2{3-0};
  let Inst{15-12} = fv1{3-0};
  let Inst{11-8} = fv0{3-0};
  let Inst{7-4} = imm16f{3-0};
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_STF_128_IP_P : Pseudo<(outs), (ins FPR:$fv3, FPR:$fv2, FPR:$fv1, FPR:$fv0, AR:$as, offset_16_16:$imm16f),
				"!xtensa_ee_stf_128_ip_p, $fv3, $fv2, $fv1, $fv0, $as, $imm16f",
			[(int_xtensa_ee_stf_128_ip FPR:$fv3, FPR:$fv2, FPR:$fv1, FPR:$fv0, AR:$as, timm:$imm16f)]>;

def EE_STF_128_XP:  EE_Inst32<(outs AR:$asr), (ins FPR:$fv3, FPR:$fv2, FPR:$fv1, FPR:$fv0, AR:$as, AR:$ad),
                   "ee.stf.128.xp\t $fv3, $fv2, $fv1, $fv0, $as, $ad", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> fv3;
  bits<4> fv2;
  bits<4> fv1;
  bits<4> fv0;
  bits<4> as;
  bits<4> ad;
  
  let mayStore = 1;
  
  let Inst{28-24} = 0x13;
  let Inst{23-20} = fv3{3-0};
  let Inst{19-16} = fv2{3-0};
  let Inst{15-12} = fv1{3-0};
  let Inst{11-8} = fv0{3-0};
  let Inst{7-4} = ad{3-0};
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_STF_128_XP_P : Pseudo<(outs), (ins FPR:$fv3, FPR:$fv2, FPR:$fv1, FPR:$fv0, AR:$as, AR:$ad),
				"!xtensa_ee_stf_128_xp_p, $fv3, $fv2, $fv1, $fv0, $as, $ad",
			[(int_xtensa_ee_stf_128_xp FPR:$fv3, FPR:$fv2, FPR:$fv1, FPR:$fv0, AR:$as, AR:$ad)]>;

def EE_STF_64_IP:  EE_Inst32<(outs AR:$asr), (ins FPR:$fv1, FPR:$fv0, AR:$as, offset_256_8:$imm8),
                   "ee.stf.64.ip\t $fv1, $fv0, $as, $imm8", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> fv1;
  bits<4> fv0;
  bits<4> as;
  bits<8> imm8;
  
  let mayStore = 1;
  
  let Inst{28-23} = 0x38;
  let Inst{22-16} = imm8{7-1};
  let Inst{15-12} = fv1{3-0};
  let Inst{11-8} = fv0{3-0};
  let Inst{7-5} = 0x3;
  let Inst{4} = imm8{0};
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_STF_64_IP_P : Pseudo<(outs), (ins FPR:$fv1, FPR:$fv0, AR:$as, offset_256_8:$imm8),
				"!xtensa_ee_stf_64_ip_p, $fv1, $fv0, $as, $imm8",
			[(int_xtensa_ee_stf_64_ip FPR:$fv1, FPR:$fv0, AR:$as, timm:$imm8)]>;

def EE_STF_64_XP:  EE_Inst24<(outs AR:$asr), (ins FPR:$fv1, FPR:$fv0, AR:$as, AR:$ad),
                   "ee.stf.64.xp\t $fv1, $fv0, $as, $ad", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> fv1;
  bits<4> fv0;
  bits<4> as;
  bits<4> ad;
  
  let mayStore = 1;
  
  let Inst{23-20} = fv0{3-0};
  let Inst{19-16} = 0x7;
  let Inst{15-12} = fv1{3-0};
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x0;
}

let usesCustomInserter = 1 in
def EE_STF_64_XP_P : Pseudo<(outs), (ins FPR:$fv1, FPR:$fv0, AR:$as, AR:$ad),
				"!xtensa_ee_stf_64_xp_p, $fv1, $fv0, $as, $ad",
			[(int_xtensa_ee_stf_64_xp FPR:$fv1, FPR:$fv0, AR:$as, AR:$ad)]>;

def EE_STXQ_32:  EE_Inst32<(outs), (ins QR:$qv, QR:$qs, AR:$as, select_4:$sel4, select_8:$sel8),
                   "ee.stxq.32\t $qv, $qs, $as, $sel4, $sel8", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qv;
  bits<3> qs;
  bits<4> as;
  bits<2> sel4;
  bits<3> sel8;
  
  let mayStore = 1;
  
  let Inst{28-22} = 0x73;
  let Inst{21-20} = sel4{1-0};
  let Inst{19-17} = 0x0;
  let Inst{16-14} = qs{2-0};
  let Inst{13-11} = sel8{2-0};
  let Inst{10-8} = qv{2-0};
  let Inst{7-4} = 0x0;
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_STXQ_32_P : Pseudo<(outs), (ins imm8:$qv, imm8:$qs, AR:$as, select_4:$sel4, select_8:$sel8),
				"!xtensa_ee_stxq_32_p, $qv, $qs, $as, $sel4, $sel8",
			[(int_xtensa_ee_stxq_32 timm:$qv, timm:$qs, AR:$as, timm:$sel4, timm:$sel8)]>;

def EE_ST_ACCX_IP:  EE_Inst24<(outs AR:$asr), (ins AR:$as, offset_256_8:$imm8),
                   "ee.st.accx.ip\t $as, $imm8", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> as;
  bits<8> imm8;
  
  let mayStore = 1;
  
  let Inst{23} = 0x0;
  let Inst{22} = imm8{7};
  let Inst{21-15} = 0x4;
  let Inst{14-8} = imm8{6-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_ST_ACCX_IP_P : Pseudo<(outs), (ins AR:$as, offset_256_8:$imm8),
				"!xtensa_ee_st_accx_ip_p, $as, $imm8",
			[(int_xtensa_ee_st_accx_ip AR:$as, timm:$imm8)]>;

def EE_ST_QACC_H_H_32_IP:  EE_Inst24<(outs AR:$asr), (ins AR:$as, offset_256_4:$imm4),
                   "ee.st.qacc_h.h.32.ip\t $as, $imm4", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> as;
  bits<8> imm4;
  
  let mayStore = 1;
  
  let Inst{23} = 0x0;
  let Inst{22} = imm4{7};
  let Inst{21-15} = 0x24;
  let Inst{14-8} = imm4{6-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_ST_QACC_H_H_32_IP_P : Pseudo<(outs), (ins AR:$as, offset_256_4:$imm4),
				"!xtensa_ee_st_qacc_h_h_32_ip_p, $as, $imm4",
			[(int_xtensa_ee_st_qacc_h_h_32_ip AR:$as, timm:$imm4)]>;

def EE_ST_QACC_H_L_128_IP:  EE_Inst24<(outs AR:$asr), (ins AR:$as, offset_256_16:$imm16),
                   "ee.st.qacc_h.l.128.ip\t $as, $imm16", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> as;
  bits<8> imm16;
  
  let mayStore = 1;
  
  let Inst{23} = 0x0;
  let Inst{22} = imm16{7};
  let Inst{21-15} = 0x1a;
  let Inst{14-8} = imm16{6-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_ST_QACC_H_L_128_IP_P : Pseudo<(outs), (ins AR:$as, offset_256_16:$imm16),
				"!xtensa_ee_st_qacc_h_l_128_ip_p, $as, $imm16",
			[(int_xtensa_ee_st_qacc_h_l_128_ip AR:$as, timm:$imm16)]>;

def EE_ST_QACC_L_H_32_IP:  EE_Inst24<(outs AR:$asr), (ins AR:$as, offset_256_4:$imm4),
                   "ee.st.qacc_l.h.32.ip\t $as, $imm4", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> as;
  bits<8> imm4;
  
  let mayStore = 1;
  
  let Inst{23} = 0x0;
  let Inst{22} = imm4{7};
  let Inst{21-15} = 0x3a;
  let Inst{14-8} = imm4{6-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_ST_QACC_L_H_32_IP_P : Pseudo<(outs), (ins AR:$as, offset_256_4:$imm4),
				"!xtensa_ee_st_qacc_l_h_32_ip_p, $as, $imm4",
			[(int_xtensa_ee_st_qacc_l_h_32_ip AR:$as, timm:$imm4)]>;

def EE_ST_QACC_L_L_128_IP:  EE_Inst24<(outs AR:$asr), (ins AR:$as, offset_256_16:$imm16),
                   "ee.st.qacc_l.l.128.ip\t $as, $imm16", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> as;
  bits<8> imm16;
  
  let mayStore = 1;
  
  let Inst{23} = 0x0;
  let Inst{22} = imm16{7};
  let Inst{21-15} = 0x18;
  let Inst{14-8} = imm16{6-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_ST_QACC_L_L_128_IP_P : Pseudo<(outs), (ins AR:$as, offset_256_16:$imm16),
				"!xtensa_ee_st_qacc_l_l_128_ip_p, $as, $imm16",
			[(int_xtensa_ee_st_qacc_l_l_128_ip AR:$as, timm:$imm16)]>;

def EE_ST_UA_STATE_IP:  EE_Inst24<(outs AR:$asr), (ins AR:$as, offset_256_16:$imm16),
                   "ee.st.ua_state.ip\t $as, $imm16", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> as;
  bits<8> imm16;
  
  let mayStore = 1;
  
  let Inst{23} = 0x0;
  let Inst{22} = imm16{7};
  let Inst{21-15} = 0x38;
  let Inst{14-8} = imm16{6-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_ST_UA_STATE_IP_P : Pseudo<(outs), (ins AR:$as, offset_256_16:$imm16),
				"!xtensa_ee_st_ua_state_ip_p, $as, $imm16",
			[(int_xtensa_ee_st_ua_state_ip AR:$as, timm:$imm16)]>;

def EE_VADDS_S16:  EE_Inst24<(outs QR:$qa), (ins QR:$qx, QR:$qy),
                   "ee.vadds.s16\t $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;
  
  
  let Inst{23-22} = 0x2;
  let Inst{21-20} = qa{2-1};
  let Inst{19-16} = 0xe;
  let Inst{15} = qa{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x0;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0x64;
}

let usesCustomInserter = 1 in
def EE_VADDS_S16_P : Pseudo<(outs), (ins imm8:$qa, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vadds_s16_p, $qa, $qx, $qy",
			[(int_xtensa_ee_vadds_s16 timm:$qa, timm:$qx, timm:$qy)]>;

def EE_VADDS_S16_LD_INCP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qa), (ins AR:$as, QR:$qx, QR:$qy),
                   "ee.vadds.s16.ld.incp\t $qu, $as, $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;
  
  let mayLoad = 1;
  
  let Inst{28-23} = 0x38;
  let Inst{22-20} = qu{2-0};
  let Inst{19-17} = qa{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-4} = 0x2d;
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VADDS_S16_LD_INCP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, imm8:$qa, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vadds_s16_ld_incp_p, $qu, $as, $qa, $qx, $qy",
			[(int_xtensa_ee_vadds_s16_ld_incp timm:$qu, AR:$as, timm:$qa, timm:$qx, timm:$qy)]>;

def EE_VADDS_S16_ST_INCP:  EE_Inst32<(outs AR:$asr, QR:$qa), (ins QR:$qv, AR:$as, QR:$qx, QR:$qy),
                   "ee.vadds.s16.st.incp\t $qv, $as, $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qv;
  bits<4> as;
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;
  
  let mayStore = 1;
  
  let Inst{28-20} = 0x1c9;
  let Inst{19-17} = qa{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-8} = qv{2-0};
  let Inst{7-4} = 0x0;
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VADDS_S16_ST_INCP_P : Pseudo<(outs), (ins imm8:$qv, AR:$as, imm8:$qa, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vadds_s16_st_incp_p, $qv, $as, $qa, $qx, $qy",
			[(int_xtensa_ee_vadds_s16_st_incp timm:$qv, AR:$as, timm:$qa, timm:$qx, timm:$qy)]>;

def EE_VADDS_S32:  EE_Inst24<(outs QR:$qa), (ins QR:$qx, QR:$qy),
                   "ee.vadds.s32\t $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;
  
  
  let Inst{23-22} = 0x2;
  let Inst{21-20} = qa{2-1};
  let Inst{19-16} = 0xe;
  let Inst{15} = qa{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x0;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0x74;
}

let usesCustomInserter = 1 in
def EE_VADDS_S32_P : Pseudo<(outs), (ins imm8:$qa, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vadds_s32_p, $qa, $qx, $qy",
			[(int_xtensa_ee_vadds_s32 timm:$qa, timm:$qx, timm:$qy)]>;

def EE_VADDS_S32_LD_INCP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qa), (ins AR:$as, QR:$qx, QR:$qy),
                   "ee.vadds.s32.ld.incp\t $qu, $as, $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;
  
  let mayLoad = 1;
  
  let Inst{28-23} = 0x38;
  let Inst{22-20} = qu{2-0};
  let Inst{19-17} = qa{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-4} = 0x3d;
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VADDS_S32_LD_INCP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, imm8:$qa, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vadds_s32_ld_incp_p, $qu, $as, $qa, $qx, $qy",
			[(int_xtensa_ee_vadds_s32_ld_incp timm:$qu, AR:$as, timm:$qa, timm:$qx, timm:$qy)]>;

def EE_VADDS_S32_ST_INCP:  EE_Inst32<(outs AR:$asr, QR:$qa), (ins QR:$qv, AR:$as, QR:$qx, QR:$qy),
                   "ee.vadds.s32.st.incp\t $qv, $as, $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qv;
  bits<4> as;
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;
  
  let mayStore = 1;
  
  let Inst{28-20} = 0x1c9;
  let Inst{19-17} = qa{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-8} = qv{2-0};
  let Inst{7-4} = 0x1;
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VADDS_S32_ST_INCP_P : Pseudo<(outs), (ins imm8:$qv, AR:$as, imm8:$qa, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vadds_s32_st_incp_p, $qv, $as, $qa, $qx, $qy",
			[(int_xtensa_ee_vadds_s32_st_incp timm:$qv, AR:$as, timm:$qa, timm:$qx, timm:$qy)]>;

def EE_VADDS_S8:  EE_Inst24<(outs QR:$qa), (ins QR:$qx, QR:$qy),
                   "ee.vadds.s8\t $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;
  
  
  let Inst{23-22} = 0x2;
  let Inst{21-20} = qa{2-1};
  let Inst{19-16} = 0xe;
  let Inst{15} = qa{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x0;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0x84;
}

let usesCustomInserter = 1 in
def EE_VADDS_S8_P : Pseudo<(outs), (ins imm8:$qa, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vadds_s8_p, $qa, $qx, $qy",
			[(int_xtensa_ee_vadds_s8 timm:$qa, timm:$qx, timm:$qy)]>;

def EE_VADDS_S8_LD_INCP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qa), (ins AR:$as, QR:$qx, QR:$qy),
                   "ee.vadds.s8.ld.incp\t $qu, $as, $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;
  
  let mayLoad = 1;
  
  let Inst{28-23} = 0x38;
  let Inst{22-20} = qu{2-0};
  let Inst{19-17} = qa{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-4} = 0x1c;
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VADDS_S8_LD_INCP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, imm8:$qa, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vadds_s8_ld_incp_p, $qu, $as, $qa, $qx, $qy",
			[(int_xtensa_ee_vadds_s8_ld_incp timm:$qu, AR:$as, timm:$qa, timm:$qx, timm:$qy)]>;

def EE_VADDS_S8_ST_INCP:  EE_Inst32<(outs AR:$asr, QR:$qa), (ins QR:$qv, AR:$as, QR:$qx, QR:$qy),
                   "ee.vadds.s8.st.incp\t $qv, $as, $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qv;
  bits<4> as;
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;
  
  let mayStore = 1;
  
  let Inst{28-20} = 0x1c9;
  let Inst{19-17} = qa{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-8} = qv{2-0};
  let Inst{7-4} = 0x2;
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VADDS_S8_ST_INCP_P : Pseudo<(outs), (ins imm8:$qv, AR:$as, imm8:$qa, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vadds_s8_st_incp_p, $qv, $as, $qa, $qx, $qy",
			[(int_xtensa_ee_vadds_s8_st_incp timm:$qv, AR:$as, timm:$qa, timm:$qx, timm:$qy)]>;

def EE_VCMP_EQ_S16:  EE_Inst24<(outs QR:$qa), (ins QR:$qx, QR:$qy),
                   "ee.vcmp.eq.s16\t $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;
  
  
  let Inst{23-22} = 0x2;
  let Inst{21-20} = qa{2-1};
  let Inst{19-16} = 0xe;
  let Inst{15} = qa{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x0;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0x94;
}

let usesCustomInserter = 1 in
def EE_VCMP_EQ_S16_P : Pseudo<(outs), (ins imm8:$qa, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vcmp_eq_s16_p, $qa, $qx, $qy",
			[(int_xtensa_ee_vcmp_eq_s16 timm:$qa, timm:$qx, timm:$qy)]>;

def EE_VCMP_EQ_S32:  EE_Inst24<(outs QR:$qa), (ins QR:$qx, QR:$qy),
                   "ee.vcmp.eq.s32\t $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;
  
  
  let Inst{23-22} = 0x2;
  let Inst{21-20} = qa{2-1};
  let Inst{19-16} = 0xe;
  let Inst{15} = qa{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x0;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0xa4;
}

let usesCustomInserter = 1 in
def EE_VCMP_EQ_S32_P : Pseudo<(outs), (ins imm8:$qa, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vcmp_eq_s32_p, $qa, $qx, $qy",
			[(int_xtensa_ee_vcmp_eq_s32 timm:$qa, timm:$qx, timm:$qy)]>;

def EE_VCMP_EQ_S8:  EE_Inst24<(outs QR:$qa), (ins QR:$qx, QR:$qy),
                   "ee.vcmp.eq.s8\t $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;
  
  
  let Inst{23-22} = 0x2;
  let Inst{21-20} = qa{2-1};
  let Inst{19-16} = 0xe;
  let Inst{15} = qa{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x0;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0xb4;
}

let usesCustomInserter = 1 in
def EE_VCMP_EQ_S8_P : Pseudo<(outs), (ins imm8:$qa, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vcmp_eq_s8_p, $qa, $qx, $qy",
			[(int_xtensa_ee_vcmp_eq_s8 timm:$qa, timm:$qx, timm:$qy)]>;

def EE_VCMP_GT_S16:  EE_Inst24<(outs QR:$qa), (ins QR:$qx, QR:$qy),
                   "ee.vcmp.gt.s16\t $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;
  
  
  let Inst{23-22} = 0x2;
  let Inst{21-20} = qa{2-1};
  let Inst{19-16} = 0xe;
  let Inst{15} = qa{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x0;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0xc4;
}

let usesCustomInserter = 1 in
def EE_VCMP_GT_S16_P : Pseudo<(outs), (ins imm8:$qa, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vcmp_gt_s16_p, $qa, $qx, $qy",
			[(int_xtensa_ee_vcmp_gt_s16 timm:$qa, timm:$qx, timm:$qy)]>;

def EE_VCMP_GT_S32:  EE_Inst24<(outs QR:$qa), (ins QR:$qx, QR:$qy),
                   "ee.vcmp.gt.s32\t $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;
  
  
  let Inst{23-22} = 0x2;
  let Inst{21-20} = qa{2-1};
  let Inst{19-16} = 0xe;
  let Inst{15} = qa{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x0;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0xd4;
}

let usesCustomInserter = 1 in
def EE_VCMP_GT_S32_P : Pseudo<(outs), (ins imm8:$qa, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vcmp_gt_s32_p, $qa, $qx, $qy",
			[(int_xtensa_ee_vcmp_gt_s32 timm:$qa, timm:$qx, timm:$qy)]>;

def EE_VCMP_GT_S8:  EE_Inst24<(outs QR:$qa), (ins QR:$qx, QR:$qy),
                   "ee.vcmp.gt.s8\t $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;
  
  
  let Inst{23-22} = 0x2;
  let Inst{21-20} = qa{2-1};
  let Inst{19-16} = 0xe;
  let Inst{15} = qa{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x0;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0xe4;
}

let usesCustomInserter = 1 in
def EE_VCMP_GT_S8_P : Pseudo<(outs), (ins imm8:$qa, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vcmp_gt_s8_p, $qa, $qx, $qy",
			[(int_xtensa_ee_vcmp_gt_s8 timm:$qa, timm:$qx, timm:$qy)]>;

def EE_VCMP_LT_S16:  EE_Inst24<(outs QR:$qa), (ins QR:$qx, QR:$qy),
                   "ee.vcmp.lt.s16\t $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;
  
  
  let Inst{23-22} = 0x2;
  let Inst{21-20} = qa{2-1};
  let Inst{19-16} = 0xe;
  let Inst{15} = qa{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x0;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0xf4;
}

let usesCustomInserter = 1 in
def EE_VCMP_LT_S16_P : Pseudo<(outs), (ins imm8:$qa, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vcmp_lt_s16_p, $qa, $qx, $qy",
			[(int_xtensa_ee_vcmp_lt_s16 timm:$qa, timm:$qx, timm:$qy)]>;

def EE_VCMP_LT_S32:  EE_Inst24<(outs QR:$qa), (ins QR:$qx, QR:$qy),
                   "ee.vcmp.lt.s32\t $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;
  
  
  let Inst{23-22} = 0x2;
  let Inst{21-20} = qa{2-1};
  let Inst{19-16} = 0xe;
  let Inst{15} = qa{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x1;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_VCMP_LT_S32_P : Pseudo<(outs), (ins imm8:$qa, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vcmp_lt_s32_p, $qa, $qx, $qy",
			[(int_xtensa_ee_vcmp_lt_s32 timm:$qa, timm:$qx, timm:$qy)]>;

def EE_VCMP_LT_S8:  EE_Inst24<(outs QR:$qa), (ins QR:$qx, QR:$qy),
                   "ee.vcmp.lt.s8\t $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;
  
  
  let Inst{23-22} = 0x2;
  let Inst{21-20} = qa{2-1};
  let Inst{19-16} = 0xe;
  let Inst{15} = qa{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x1;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0x14;
}

let usesCustomInserter = 1 in
def EE_VCMP_LT_S8_P : Pseudo<(outs), (ins imm8:$qa, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vcmp_lt_s8_p, $qa, $qx, $qy",
			[(int_xtensa_ee_vcmp_lt_s8 timm:$qa, timm:$qx, timm:$qy)]>;

def EE_VLDBC_16:  EE_Inst24<(outs QR:$qu), (ins AR:$as),
                   "ee.vldbc.16\t $qu, $as", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  
  let mayLoad = 1;
  
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qu{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qu{0};
  let Inst{14-8} = 0x73;
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_VLDBC_16_P : Pseudo<(outs), (ins imm8:$qu, AR:$as),
				"!xtensa_ee_vldbc_16_p, $qu, $as",
			[(int_xtensa_ee_vldbc_16 timm:$qu, AR:$as)]>;

def EE_VLDBC_16_IP:  EE_Inst24<(outs QR:$qu, AR:$asr), (ins AR:$as, offset_128_2:$imm2),
                   "ee.vldbc.16.ip\t $qu, $as, $imm2", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<7> imm2;
  
  let mayLoad = 1;
  
  let Inst{23-22} = 0x2;
  let Inst{21-20} = qu{2-1};
  let Inst{19-16} = 0x5;
  let Inst{15} = qu{0};
  let Inst{14-8} = imm2{6-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_VLDBC_16_IP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, offset_128_2:$imm2),
				"!xtensa_ee_vldbc_16_ip_p, $qu, $as, $imm2",
			[(int_xtensa_ee_vldbc_16_ip timm:$qu, AR:$as, timm:$imm2)]>;

def EE_VLDBC_16_XP:  EE_Inst24<(outs QR:$qu, AR:$asr), (ins AR:$as, AR:$ad),
                   "ee.vldbc.16.xp\t $qu, $as, $ad", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<4> ad;
  
  let mayLoad = 1;
  
  let Inst{23-22} = 0x2;
  let Inst{21-20} = qu{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qu{0};
  let Inst{14-12} = 0x4;
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_VLDBC_16_XP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, AR:$ad),
				"!xtensa_ee_vldbc_16_xp_p, $qu, $as, $ad",
			[(int_xtensa_ee_vldbc_16_xp timm:$qu, AR:$as, AR:$ad)]>;

def EE_VLDBC_32:  EE_Inst24<(outs QR:$qu), (ins AR:$as),
                   "ee.vldbc.32\t $qu, $as", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  
  let mayLoad = 1;
  
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qu{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qu{0};
  let Inst{14-8} = 0x77;
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_VLDBC_32_P : Pseudo<(outs), (ins imm8:$qu, AR:$as),
				"!xtensa_ee_vldbc_32_p, $qu, $as",
			[(int_xtensa_ee_vldbc_32 timm:$qu, AR:$as)]>;

def EE_VLDBC_32_IP:  EE_Inst24<(outs QR:$qu, AR:$asr), (ins AR:$as, offset_256_4:$imm4),
                   "ee.vldbc.32.ip\t $qu, $as, $imm4", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<8> imm4;
  
  let mayLoad = 1;
  
  let Inst{23} = 0x1;
  let Inst{22} = imm4{7};
  let Inst{21-20} = qu{2-1};
  let Inst{19-16} = 0x2;
  let Inst{15} = qu{0};
  let Inst{14-8} = imm4{6-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_VLDBC_32_IP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, offset_256_4:$imm4),
				"!xtensa_ee_vldbc_32_ip_p, $qu, $as, $imm4",
			[(int_xtensa_ee_vldbc_32_ip timm:$qu, AR:$as, timm:$imm4)]>;

def EE_VLDBC_32_XP:  EE_Inst24<(outs QR:$qu, AR:$asr), (ins AR:$as, AR:$ad),
                   "ee.vldbc.32.xp\t $qu, $as, $ad", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<4> ad;
  
  let mayLoad = 1;
  
  let Inst{23-22} = 0x2;
  let Inst{21-20} = qu{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qu{0};
  let Inst{14-12} = 0x1;
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_VLDBC_32_XP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, AR:$ad),
				"!xtensa_ee_vldbc_32_xp_p, $qu, $as, $ad",
			[(int_xtensa_ee_vldbc_32_xp timm:$qu, AR:$as, AR:$ad)]>;

def EE_VLDBC_8:  EE_Inst24<(outs QR:$qu), (ins AR:$as),
                   "ee.vldbc.8\t $qu, $as", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  
  let mayLoad = 1;
  
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qu{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qu{0};
  let Inst{14-8} = 0x3b;
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_VLDBC_8_P : Pseudo<(outs), (ins imm8:$qu, AR:$as),
				"!xtensa_ee_vldbc_8_p, $qu, $as",
			[(int_xtensa_ee_vldbc_8 timm:$qu, AR:$as)]>;

def EE_VLDBC_8_IP:  EE_Inst24<(outs QR:$qu, AR:$asr), (ins AR:$as, offset_128_1:$imm1),
                   "ee.vldbc.8.ip\t $qu, $as, $imm1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<7> imm1;
  
  let mayLoad = 1;
  
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qu{2-1};
  let Inst{19-16} = 0x5;
  let Inst{15} = qu{0};
  let Inst{14-8} = imm1{6-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_VLDBC_8_IP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, offset_128_1:$imm1),
				"!xtensa_ee_vldbc_8_ip_p, $qu, $as, $imm1",
			[(int_xtensa_ee_vldbc_8_ip timm:$qu, AR:$as, timm:$imm1)]>;

def EE_VLDBC_8_XP:  EE_Inst24<(outs QR:$qu, AR:$asr), (ins AR:$as, AR:$ad),
                   "ee.vldbc.8.xp\t $qu, $as, $ad", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<4> ad;
  
  let mayLoad = 1;
  
  let Inst{23-22} = 0x2;
  let Inst{21-20} = qu{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qu{0};
  let Inst{14-12} = 0x5;
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_VLDBC_8_XP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, AR:$ad),
				"!xtensa_ee_vldbc_8_xp_p, $qu, $as, $ad",
			[(int_xtensa_ee_vldbc_8_xp timm:$qu, AR:$as, AR:$ad)]>;

def EE_VLDHBC_16_INCP:  EE_Inst24<(outs QR:$qu, QR:$qu1, AR:$asr), (ins AR:$as),
                   "ee.vldhbc.16.incp\t $qu, $qu1, $as", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<3> qu1;
  bits<4> as;
  
  let mayLoad = 1;
  
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qu{2-1};
  let Inst{19-16} = 0xc;
  let Inst{15} = qu{0};
  let Inst{14-12} = qu1{2-0};
  let Inst{11-8} = 0x2;
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_VLDHBC_16_INCP_P : Pseudo<(outs), (ins imm8:$qu, imm8:$qu1, AR:$as),
				"!xtensa_ee_vldhbc_16_incp_p, $qu, $qu1, $as",
			[(int_xtensa_ee_vldhbc_16_incp timm:$qu, timm:$qu1, AR:$as)]>;

def EE_VLD_128_IP:  EE_Inst24<(outs QR:$qu, AR:$asr), (ins AR:$as, offset_256_16:$imm16),
                   "ee.vld.128.ip\t $qu, $as, $imm16", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<8> imm16;
  
  let mayLoad = 1;
  
  let Inst{23} = 0x1;
  let Inst{22} = imm16{7};
  let Inst{21-20} = qu{2-1};
  let Inst{19-16} = 0x3;
  let Inst{15} = qu{0};
  let Inst{14-8} = imm16{6-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_VLD_128_IP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, offset_256_16:$imm16),
				"!xtensa_ee_vld_128_ip_p, $qu, $as, $imm16",
			[(int_xtensa_ee_vld_128_ip timm:$qu, AR:$as, timm:$imm16)]>;

def EE_VLD_128_XP:  EE_Inst24<(outs QR:$qu, AR:$asr), (ins AR:$as, AR:$ad),
                   "ee.vld.128.xp\t $qu, $as, $ad", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<4> ad;
  
  let mayLoad = 1;
  
  let Inst{23-22} = 0x2;
  let Inst{21-20} = qu{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qu{0};
  let Inst{14-12} = 0x2;
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_VLD_128_XP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, AR:$ad),
				"!xtensa_ee_vld_128_xp_p, $qu, $as, $ad",
			[(int_xtensa_ee_vld_128_xp timm:$qu, AR:$as, AR:$ad)]>;

def EE_VLD_H_64_IP:  EE_Inst24<(outs QR:$qu, AR:$asr), (ins AR:$as, offset_256_8:$imm8),
                   "ee.vld.h.64.ip\t $qu, $as, $imm8", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<8> imm8;
  
  let mayLoad = 1;
  
  let Inst{23} = 0x1;
  let Inst{22} = imm8{7};
  let Inst{21-20} = qu{2-1};
  let Inst{19-16} = 0x8;
  let Inst{15} = qu{0};
  let Inst{14-8} = imm8{6-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_VLD_H_64_IP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, offset_256_8:$imm8),
				"!xtensa_ee_vld_h_64_ip_p, $qu, $as, $imm8",
			[(int_xtensa_ee_vld_h_64_ip timm:$qu, AR:$as, timm:$imm8)]>;

def EE_VLD_H_64_XP:  EE_Inst24<(outs QR:$qu, AR:$asr), (ins AR:$as, AR:$ad),
                   "ee.vld.h.64.xp\t $qu, $as, $ad", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<4> ad;
  
  let mayLoad = 1;
  
  let Inst{23-22} = 0x2;
  let Inst{21-20} = qu{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qu{0};
  let Inst{14-12} = 0x6;
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_VLD_H_64_XP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, AR:$ad),
				"!xtensa_ee_vld_h_64_xp_p, $qu, $as, $ad",
			[(int_xtensa_ee_vld_h_64_xp timm:$qu, AR:$as, AR:$ad)]>;

def EE_VLD_L_64_IP:  EE_Inst24<(outs QR:$qu, AR:$asr), (ins AR:$as, offset_256_8:$imm8),
                   "ee.vld.l.64.ip\t $qu, $as, $imm8", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<8> imm8;
  
  let mayLoad = 1;
  
  let Inst{23} = 0x1;
  let Inst{22} = imm8{7};
  let Inst{21-20} = qu{2-1};
  let Inst{19-16} = 0x9;
  let Inst{15} = qu{0};
  let Inst{14-8} = imm8{6-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_VLD_L_64_IP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, offset_256_8:$imm8),
				"!xtensa_ee_vld_l_64_ip_p, $qu, $as, $imm8",
			[(int_xtensa_ee_vld_l_64_ip timm:$qu, AR:$as, timm:$imm8)]>;

def EE_VLD_L_64_XP:  EE_Inst24<(outs QR:$qu, AR:$asr), (ins AR:$as, AR:$ad),
                   "ee.vld.l.64.xp\t $qu, $as, $ad", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<4> ad;
  
  let mayLoad = 1;
  
  let Inst{23-22} = 0x2;
  let Inst{21-20} = qu{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qu{0};
  let Inst{14-12} = 0x3;
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_VLD_L_64_XP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, AR:$ad),
				"!xtensa_ee_vld_l_64_xp_p, $qu, $as, $ad",
			[(int_xtensa_ee_vld_l_64_xp timm:$qu, AR:$as, AR:$ad)]>;

def EE_VMAX_S16:  EE_Inst24<(outs QR:$qa), (ins QR:$qx, QR:$qy),
                   "ee.vmax.s16\t $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;
  
  
  let Inst{23-22} = 0x2;
  let Inst{21-20} = qa{2-1};
  let Inst{19-16} = 0xe;
  let Inst{15} = qa{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x1;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0x24;
}

let usesCustomInserter = 1 in
def EE_VMAX_S16_P : Pseudo<(outs), (ins imm8:$qa, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmax_s16_p, $qa, $qx, $qy",
			[(int_xtensa_ee_vmax_s16 timm:$qa, timm:$qx, timm:$qy)]>;

def EE_VMAX_S16_LD_INCP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qa), (ins AR:$as, QR:$qx, QR:$qy),
                   "ee.vmax.s16.ld.incp\t $qu, $as, $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;
  
  let mayLoad = 1;
  
  let Inst{28-23} = 0x38;
  let Inst{22-20} = qu{2-0};
  let Inst{19-17} = qa{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-4} = 0x1d;
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMAX_S16_LD_INCP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, imm8:$qa, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmax_s16_ld_incp_p, $qu, $as, $qa, $qx, $qy",
			[(int_xtensa_ee_vmax_s16_ld_incp timm:$qu, AR:$as, timm:$qa, timm:$qx, timm:$qy)]>;

def EE_VMAX_S16_ST_INCP:  EE_Inst32<(outs AR:$asr, QR:$qa), (ins QR:$qv, AR:$as, QR:$qx, QR:$qy),
                   "ee.vmax.s16.st.incp\t $qv, $as, $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qv;
  bits<4> as;
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;
  
  let mayStore = 1;
  
  let Inst{28-20} = 0x1c9;
  let Inst{19-17} = qa{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-8} = qv{2-0};
  let Inst{7-4} = 0x3;
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMAX_S16_ST_INCP_P : Pseudo<(outs), (ins imm8:$qv, AR:$as, imm8:$qa, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmax_s16_st_incp_p, $qv, $as, $qa, $qx, $qy",
			[(int_xtensa_ee_vmax_s16_st_incp timm:$qv, AR:$as, timm:$qa, timm:$qx, timm:$qy)]>;

def EE_VMAX_S32:  EE_Inst24<(outs QR:$qa), (ins QR:$qx, QR:$qy),
                   "ee.vmax.s32\t $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;
  
  
  let Inst{23-22} = 0x2;
  let Inst{21-20} = qa{2-1};
  let Inst{19-16} = 0xe;
  let Inst{15} = qa{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x1;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0x34;
}

let usesCustomInserter = 1 in
def EE_VMAX_S32_P : Pseudo<(outs), (ins imm8:$qa, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmax_s32_p, $qa, $qx, $qy",
			[(int_xtensa_ee_vmax_s32 timm:$qa, timm:$qx, timm:$qy)]>;

def EE_VMAX_S32_LD_INCP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qa), (ins AR:$as, QR:$qx, QR:$qy),
                   "ee.vmax.s32.ld.incp\t $qu, $as, $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;
  
  let mayLoad = 1;
  
  let Inst{28-23} = 0x38;
  let Inst{22-20} = qu{2-0};
  let Inst{19-17} = qa{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-4} = 0x1e;
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMAX_S32_LD_INCP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, imm8:$qa, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmax_s32_ld_incp_p, $qu, $as, $qa, $qx, $qy",
			[(int_xtensa_ee_vmax_s32_ld_incp timm:$qu, AR:$as, timm:$qa, timm:$qx, timm:$qy)]>;

def EE_VMAX_S32_ST_INCP:  EE_Inst32<(outs AR:$asr, QR:$qa), (ins QR:$qv, AR:$as, QR:$qx, QR:$qy),
                   "ee.vmax.s32.st.incp\t $qv, $as, $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qv;
  bits<4> as;
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;
  
  let mayStore = 1;
  
  let Inst{28-20} = 0x1ca;
  let Inst{19-17} = qa{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-8} = qv{2-0};
  let Inst{7-4} = 0x0;
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMAX_S32_ST_INCP_P : Pseudo<(outs), (ins imm8:$qv, AR:$as, imm8:$qa, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmax_s32_st_incp_p, $qv, $as, $qa, $qx, $qy",
			[(int_xtensa_ee_vmax_s32_st_incp timm:$qv, AR:$as, timm:$qa, timm:$qx, timm:$qy)]>;

def EE_VMAX_S8:  EE_Inst24<(outs QR:$qa), (ins QR:$qx, QR:$qy),
                   "ee.vmax.s8\t $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;
  
  
  let Inst{23-22} = 0x2;
  let Inst{21-20} = qa{2-1};
  let Inst{19-16} = 0xe;
  let Inst{15} = qa{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x1;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0x44;
}

let usesCustomInserter = 1 in
def EE_VMAX_S8_P : Pseudo<(outs), (ins imm8:$qa, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmax_s8_p, $qa, $qx, $qy",
			[(int_xtensa_ee_vmax_s8 timm:$qa, timm:$qx, timm:$qy)]>;

def EE_VMAX_S8_LD_INCP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qa), (ins AR:$as, QR:$qx, QR:$qy),
                   "ee.vmax.s8.ld.incp\t $qu, $as, $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;
  
  let mayLoad = 1;
  
  let Inst{28-23} = 0x38;
  let Inst{22-20} = qu{2-0};
  let Inst{19-17} = qa{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-4} = 0x1f;
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMAX_S8_LD_INCP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, imm8:$qa, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmax_s8_ld_incp_p, $qu, $as, $qa, $qx, $qy",
			[(int_xtensa_ee_vmax_s8_ld_incp timm:$qu, AR:$as, timm:$qa, timm:$qx, timm:$qy)]>;

def EE_VMAX_S8_ST_INCP:  EE_Inst32<(outs AR:$asr, QR:$qa), (ins QR:$qv, AR:$as, QR:$qx, QR:$qy),
                   "ee.vmax.s8.st.incp\t $qv, $as, $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qv;
  bits<4> as;
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;
  
  let mayStore = 1;
  
  let Inst{28-20} = 0x1cb;
  let Inst{19-17} = qa{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-8} = qv{2-0};
  let Inst{7-4} = 0x0;
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMAX_S8_ST_INCP_P : Pseudo<(outs), (ins imm8:$qv, AR:$as, imm8:$qa, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmax_s8_st_incp_p, $qv, $as, $qa, $qx, $qy",
			[(int_xtensa_ee_vmax_s8_st_incp timm:$qv, AR:$as, timm:$qa, timm:$qx, timm:$qy)]>;

def EE_VMIN_S16:  EE_Inst24<(outs QR:$qa), (ins QR:$qx, QR:$qy),
                   "ee.vmin.s16\t $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;
  
  
  let Inst{23-22} = 0x2;
  let Inst{21-20} = qa{2-1};
  let Inst{19-16} = 0xe;
  let Inst{15} = qa{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x1;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0x54;
}

let usesCustomInserter = 1 in
def EE_VMIN_S16_P : Pseudo<(outs), (ins imm8:$qa, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmin_s16_p, $qa, $qx, $qy",
			[(int_xtensa_ee_vmin_s16 timm:$qa, timm:$qx, timm:$qy)]>;

def EE_VMIN_S16_LD_INCP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qa), (ins AR:$as, QR:$qx, QR:$qy),
                   "ee.vmin.s16.ld.incp\t $qu, $as, $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;
  
  let mayLoad = 1;
  
  let Inst{28-23} = 0x38;
  let Inst{22-20} = qu{2-0};
  let Inst{19-17} = qa{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-4} = 0x2e;
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMIN_S16_LD_INCP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, imm8:$qa, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmin_s16_ld_incp_p, $qu, $as, $qa, $qx, $qy",
			[(int_xtensa_ee_vmin_s16_ld_incp timm:$qu, AR:$as, timm:$qa, timm:$qx, timm:$qy)]>;

def EE_VMIN_S16_ST_INCP:  EE_Inst32<(outs AR:$asr, QR:$qa), (ins QR:$qv, AR:$as, QR:$qx, QR:$qy),
                   "ee.vmin.s16.st.incp\t $qv, $as, $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qv;
  bits<4> as;
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;
  
  let mayStore = 1;
  
  let Inst{28-20} = 0x1ca;
  let Inst{19-17} = qa{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-8} = qv{2-0};
  let Inst{7-4} = 0x1;
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMIN_S16_ST_INCP_P : Pseudo<(outs), (ins imm8:$qv, AR:$as, imm8:$qa, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmin_s16_st_incp_p, $qv, $as, $qa, $qx, $qy",
			[(int_xtensa_ee_vmin_s16_st_incp timm:$qv, AR:$as, timm:$qa, timm:$qx, timm:$qy)]>;

def EE_VMIN_S32:  EE_Inst24<(outs QR:$qa), (ins QR:$qx, QR:$qy),
                   "ee.vmin.s32\t $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;
  
  
  let Inst{23-22} = 0x2;
  let Inst{21-20} = qa{2-1};
  let Inst{19-16} = 0xe;
  let Inst{15} = qa{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x1;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0x64;
}

let usesCustomInserter = 1 in
def EE_VMIN_S32_P : Pseudo<(outs), (ins imm8:$qa, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmin_s32_p, $qa, $qx, $qy",
			[(int_xtensa_ee_vmin_s32 timm:$qa, timm:$qx, timm:$qy)]>;

def EE_VMIN_S32_LD_INCP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qa), (ins AR:$as, QR:$qx, QR:$qy),
                   "ee.vmin.s32.ld.incp\t $qu, $as, $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;
  
  let mayLoad = 1;
  
  let Inst{28-23} = 0x38;
  let Inst{22-20} = qu{2-0};
  let Inst{19-17} = qa{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-4} = 0x3e;
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMIN_S32_LD_INCP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, imm8:$qa, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmin_s32_ld_incp_p, $qu, $as, $qa, $qx, $qy",
			[(int_xtensa_ee_vmin_s32_ld_incp timm:$qu, AR:$as, timm:$qa, timm:$qx, timm:$qy)]>;

def EE_VMIN_S32_ST_INCP:  EE_Inst32<(outs AR:$asr, QR:$qa), (ins QR:$qv, AR:$as, QR:$qx, QR:$qy),
                   "ee.vmin.s32.st.incp\t $qv, $as, $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qv;
  bits<4> as;
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;
  
  let mayStore = 1;
  
  let Inst{28-20} = 0x1cb;
  let Inst{19-17} = qa{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-8} = qv{2-0};
  let Inst{7-4} = 0x1;
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMIN_S32_ST_INCP_P : Pseudo<(outs), (ins imm8:$qv, AR:$as, imm8:$qa, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmin_s32_st_incp_p, $qv, $as, $qa, $qx, $qy",
			[(int_xtensa_ee_vmin_s32_st_incp timm:$qv, AR:$as, timm:$qa, timm:$qx, timm:$qy)]>;

def EE_VMIN_S8:  EE_Inst24<(outs QR:$qa), (ins QR:$qx, QR:$qy),
                   "ee.vmin.s8\t $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;
  
  
  let Inst{23-22} = 0x2;
  let Inst{21-20} = qa{2-1};
  let Inst{19-16} = 0xe;
  let Inst{15} = qa{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x1;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0x74;
}

let usesCustomInserter = 1 in
def EE_VMIN_S8_P : Pseudo<(outs), (ins imm8:$qa, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmin_s8_p, $qa, $qx, $qy",
			[(int_xtensa_ee_vmin_s8 timm:$qa, timm:$qx, timm:$qy)]>;

def EE_VMIN_S8_LD_INCP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qa), (ins AR:$as, QR:$qx, QR:$qy),
                   "ee.vmin.s8.ld.incp\t $qu, $as, $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;
  
  let mayLoad = 1;
  
  let Inst{28-23} = 0x38;
  let Inst{22-20} = qu{2-0};
  let Inst{19-17} = qa{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-4} = 0x2f;
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMIN_S8_LD_INCP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, imm8:$qa, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmin_s8_ld_incp_p, $qu, $as, $qa, $qx, $qy",
			[(int_xtensa_ee_vmin_s8_ld_incp timm:$qu, AR:$as, timm:$qa, timm:$qx, timm:$qy)]>;

def EE_VMIN_S8_ST_INCP:  EE_Inst32<(outs AR:$asr, QR:$qa), (ins QR:$qv, AR:$as, QR:$qx, QR:$qy),
                   "ee.vmin.s8.st.incp\t $qv, $as, $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qv;
  bits<4> as;
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;
  
  let mayStore = 1;
  
  let Inst{28-20} = 0x1ca;
  let Inst{19-17} = qa{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-8} = qv{2-0};
  let Inst{7-4} = 0x2;
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMIN_S8_ST_INCP_P : Pseudo<(outs), (ins imm8:$qv, AR:$as, imm8:$qa, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmin_s8_st_incp_p, $qv, $as, $qa, $qx, $qy",
			[(int_xtensa_ee_vmin_s8_st_incp timm:$qv, AR:$as, timm:$qa, timm:$qx, timm:$qy)]>;

def EE_VMULAS_S16_ACCX:  EE_Inst24<(outs), (ins QR:$qx, QR:$qy),
                   "ee.vmulas.s16.accx\t $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qx;
  bits<3> qy;
  
  
  let Inst{23-15} = 0x34;
  let Inst{14} = qy{2};
  let Inst{13} = 0x0;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0x84;
}

let usesCustomInserter = 1 in
def EE_VMULAS_S16_ACCX_P : Pseudo<(outs), (ins imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmulas_s16_accx_p, $qx, $qy",
			[(int_xtensa_ee_vmulas_s16_accx timm:$qx, timm:$qy)]>;

def EE_VMULAS_S16_ACCX_LD_IP:  EE_Inst32<(outs QR:$qu, AR:$asr), (ins AR:$as, offset_64_16:$imm16, QR:$qx, QR:$qy),
                   "ee.vmulas.s16.accx.ld.ip\t $qu, $as, $imm16, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<8> imm16;
  bits<3> qx;
  bits<3> qy;
  
  let mayLoad = 1;
  
  let Inst{28-25} = 0xf;
  let Inst{24-23} = imm16{5-4};
  let Inst{22-20} = qu{2-0};
  let Inst{19-17} = 0x0;
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-8} = 0x0;
  let Inst{7-4} = imm16{3-0};
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMULAS_S16_ACCX_LD_IP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, offset_64_16:$imm16, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmulas_s16_accx_ld_ip_p, $qu, $as, $imm16, $qx, $qy",
			[(int_xtensa_ee_vmulas_s16_accx_ld_ip timm:$qu, AR:$as, timm:$imm16, timm:$qx, timm:$qy)]>;

def EE_VMULAS_S16_ACCX_LD_IP_QUP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qs0r), (ins AR:$as, offset_64_16:$imm16, QR:$qx, QR:$qy, QR:$qs0, QR:$qs1),
                   "ee.vmulas.s16.accx.ld.ip.qup\t $qu, $as, $imm16, $qx, $qy, $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<8> imm16;
  bits<3> qx;
  bits<3> qy;
  bits<3> qs0;
  bits<3> qs1;
  
  let mayLoad = 1;
  
  let Inst{28-25} = 0x0;
  let Inst{24-23} = imm16{5-4};
  let Inst{22-20} = qu{2-0};
  let Inst{19-17} = qs1{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-8} = qs0{2-0};
  let Inst{7-4} = imm16{3-0};
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMULAS_S16_ACCX_LD_IP_QUP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, offset_64_16:$imm16, imm8:$qx, imm8:$qy, imm8:$qs0, imm8:$qs1),
				"!xtensa_ee_vmulas_s16_accx_ld_ip_qup_p, $qu, $as, $imm16, $qx, $qy, $qs0, $qs1",
			[(int_xtensa_ee_vmulas_s16_accx_ld_ip_qup timm:$qu, AR:$as, timm:$imm16, timm:$qx, timm:$qy, timm:$qs0, timm:$qs1)]>;

def EE_VMULAS_S16_ACCX_LD_XP:  EE_Inst32<(outs QR:$qu, AR:$asr), (ins AR:$as, AR:$ad, QR:$qx, QR:$qy),
                   "ee.vmulas.s16.accx.ld.xp\t $qu, $as, $ad, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<4> ad;
  bits<3> qx;
  bits<3> qy;
  
  let mayLoad = 1;
  
  let Inst{28-23} = 0x3c;
  let Inst{22-20} = qu{2-0};
  let Inst{19-17} = 0x0;
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-8} = 0x1;
  let Inst{7-4} = ad{3-0};
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMULAS_S16_ACCX_LD_XP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, AR:$ad, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmulas_s16_accx_ld_xp_p, $qu, $as, $ad, $qx, $qy",
			[(int_xtensa_ee_vmulas_s16_accx_ld_xp timm:$qu, AR:$as, AR:$ad, timm:$qx, timm:$qy)]>;

def EE_VMULAS_S16_ACCX_LD_XP_QUP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qs0r), (ins AR:$as, AR:$ad, QR:$qx, QR:$qy, QR:$qs0, QR:$qs1),
                   "ee.vmulas.s16.accx.ld.xp.qup\t $qu, $as, $ad, $qx, $qy, $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<4> ad;
  bits<3> qx;
  bits<3> qy;
  bits<3> qs0;
  bits<3> qs1;
  
  let mayLoad = 1;
  
  let Inst{28-23} = 0x2c;
  let Inst{22-20} = qu{2-0};
  let Inst{19-17} = qs1{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-8} = qs0{2-0};
  let Inst{7-4} = ad{3-0};
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMULAS_S16_ACCX_LD_XP_QUP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, AR:$ad, imm8:$qx, imm8:$qy, imm8:$qs0, imm8:$qs1),
				"!xtensa_ee_vmulas_s16_accx_ld_xp_qup_p, $qu, $as, $ad, $qx, $qy, $qs0, $qs1",
			[(int_xtensa_ee_vmulas_s16_accx_ld_xp_qup timm:$qu, AR:$as, AR:$ad, timm:$qx, timm:$qy, timm:$qs0, timm:$qs1)]>;

def EE_VMULAS_S16_QACC:  EE_Inst24<(outs), (ins QR:$qx, QR:$qy),
                   "ee.vmulas.s16.qacc\t $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qx;
  bits<3> qy;
  
  
  let Inst{23-15} = 0x34;
  let Inst{14} = qy{2};
  let Inst{13} = 0x1;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0x84;
}

let usesCustomInserter = 1 in
def EE_VMULAS_S16_QACC_P : Pseudo<(outs), (ins imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmulas_s16_qacc_p, $qx, $qy",
			[(int_xtensa_ee_vmulas_s16_qacc timm:$qx, timm:$qy)]>;

def EE_VMULAS_S16_QACC_LDBC_INCP:  EE_Inst24<(outs QR:$qu, AR:$asr), (ins AR:$as, QR:$qx, QR:$qy),
                   "ee.vmulas.s16.qacc.ldbc.incp\t $qu, $as, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qx;
  bits<3> qy;
  
  let mayLoad = 1;
  
  let Inst{23-21} = 0x4;
  let Inst{20} = qu{2};
  let Inst{19-16} = 0x7;
  let Inst{15} = qu{1};
  let Inst{14} = qy{2};
  let Inst{13} = qu{0};
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_VMULAS_S16_QACC_LDBC_INCP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmulas_s16_qacc_ldbc_incp_p, $qu, $as, $qx, $qy",
			[(int_xtensa_ee_vmulas_s16_qacc_ldbc_incp timm:$qu, AR:$as, timm:$qx, timm:$qy)]>;

def EE_VMULAS_S16_QACC_LDBC_INCP_QUP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qs0r), (ins AR:$as, QR:$qx, QR:$qy, QR:$qs0, QR:$qs1),
                   "ee.vmulas.s16.qacc.ldbc.incp.qup\t $qu, $as, $qx, $qy, $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qx;
  bits<3> qy;
  bits<3> qs0;
  bits<3> qs1;
  
  let mayLoad = 1;
  
  let Inst{28-23} = 0x38;
  let Inst{22-20} = qu{2-0};
  let Inst{19-17} = qs1{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-8} = qs0{2-0};
  let Inst{7-4} = 0x8;
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMULAS_S16_QACC_LDBC_INCP_QUP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, imm8:$qx, imm8:$qy, imm8:$qs0, imm8:$qs1),
				"!xtensa_ee_vmulas_s16_qacc_ldbc_incp_qup_p, $qu, $as, $qx, $qy, $qs0, $qs1",
			[(int_xtensa_ee_vmulas_s16_qacc_ldbc_incp_qup timm:$qu, AR:$as, timm:$qx, timm:$qy, timm:$qs0, timm:$qs1)]>;

def EE_VMULAS_S16_QACC_LD_IP:  EE_Inst32<(outs QR:$qu, AR:$asr), (ins AR:$as, offset_64_16:$imm16, QR:$qx, QR:$qy),
                   "ee.vmulas.s16.qacc.ld.ip\t $qu, $as, $imm16, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<8> imm16;
  bits<3> qx;
  bits<3> qy;
  
  let mayLoad = 1;
  
  let Inst{28-25} = 0xf;
  let Inst{24-23} = imm16{5-4};
  let Inst{22-20} = qu{2-0};
  let Inst{19-17} = 0x1;
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-8} = 0x0;
  let Inst{7-4} = imm16{3-0};
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMULAS_S16_QACC_LD_IP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, offset_64_16:$imm16, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmulas_s16_qacc_ld_ip_p, $qu, $as, $imm16, $qx, $qy",
			[(int_xtensa_ee_vmulas_s16_qacc_ld_ip timm:$qu, AR:$as, timm:$imm16, timm:$qx, timm:$qy)]>;

def EE_VMULAS_S16_QACC_LD_IP_QUP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qs0r), (ins AR:$as, offset_64_16:$imm16, QR:$qx, QR:$qy, QR:$qs0, QR:$qs1),
                   "ee.vmulas.s16.qacc.ld.ip.qup\t $qu, $as, $imm16, $qx, $qy, $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<8> imm16;
  bits<3> qx;
  bits<3> qy;
  bits<3> qs0;
  bits<3> qs1;
  
  let mayLoad = 1;
  
  let Inst{28-25} = 0x1;
  let Inst{24-23} = imm16{5-4};
  let Inst{22-20} = qu{2-0};
  let Inst{19-17} = qs1{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-8} = qs0{2-0};
  let Inst{7-4} = imm16{3-0};
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMULAS_S16_QACC_LD_IP_QUP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, offset_64_16:$imm16, imm8:$qx, imm8:$qy, imm8:$qs0, imm8:$qs1),
				"!xtensa_ee_vmulas_s16_qacc_ld_ip_qup_p, $qu, $as, $imm16, $qx, $qy, $qs0, $qs1",
			[(int_xtensa_ee_vmulas_s16_qacc_ld_ip_qup timm:$qu, AR:$as, timm:$imm16, timm:$qx, timm:$qy, timm:$qs0, timm:$qs1)]>;

def EE_VMULAS_S16_QACC_LD_XP:  EE_Inst32<(outs QR:$qu, AR:$asr), (ins AR:$as, AR:$ad, QR:$qx, QR:$qy),
                   "ee.vmulas.s16.qacc.ld.xp\t $qu, $as, $ad, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<4> ad;
  bits<3> qx;
  bits<3> qy;
  
  let mayLoad = 1;
  
  let Inst{28-23} = 0x3c;
  let Inst{22-20} = qu{2-0};
  let Inst{19-17} = 0x1;
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-8} = 0x1;
  let Inst{7-4} = ad{3-0};
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMULAS_S16_QACC_LD_XP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, AR:$ad, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmulas_s16_qacc_ld_xp_p, $qu, $as, $ad, $qx, $qy",
			[(int_xtensa_ee_vmulas_s16_qacc_ld_xp timm:$qu, AR:$as, AR:$ad, timm:$qx, timm:$qy)]>;

def EE_VMULAS_S16_QACC_LD_XP_QUP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qs0r), (ins AR:$as, AR:$ad, QR:$qx, QR:$qy, QR:$qs0, QR:$qs1),
                   "ee.vmulas.s16.qacc.ld.xp.qup\t $qu, $as, $ad, $qx, $qy, $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<4> ad;
  bits<3> qx;
  bits<3> qy;
  bits<3> qs0;
  bits<3> qs1;
  
  let mayLoad = 1;
  
  let Inst{28-23} = 0x2d;
  let Inst{22-20} = qu{2-0};
  let Inst{19-17} = qs1{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-8} = qs0{2-0};
  let Inst{7-4} = ad{3-0};
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMULAS_S16_QACC_LD_XP_QUP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, AR:$ad, imm8:$qx, imm8:$qy, imm8:$qs0, imm8:$qs1),
				"!xtensa_ee_vmulas_s16_qacc_ld_xp_qup_p, $qu, $as, $ad, $qx, $qy, $qs0, $qs1",
			[(int_xtensa_ee_vmulas_s16_qacc_ld_xp_qup timm:$qu, AR:$as, AR:$ad, timm:$qx, timm:$qy, timm:$qs0, timm:$qs1)]>;

def EE_VMULAS_S8_ACCX:  EE_Inst24<(outs), (ins QR:$qx, QR:$qy),
                   "ee.vmulas.s8.accx\t $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qx;
  bits<3> qy;
  
  
  let Inst{23-15} = 0x34;
  let Inst{14} = qy{2};
  let Inst{13} = 0x0;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0xc4;
}

let usesCustomInserter = 1 in
def EE_VMULAS_S8_ACCX_P : Pseudo<(outs), (ins imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmulas_s8_accx_p, $qx, $qy",
			[(int_xtensa_ee_vmulas_s8_accx timm:$qx, timm:$qy)]>;

def EE_VMULAS_S8_ACCX_LD_IP:  EE_Inst32<(outs QR:$qu, AR:$asr), (ins AR:$as, offset_64_16:$imm16, QR:$qx, QR:$qy),
                   "ee.vmulas.s8.accx.ld.ip\t $qu, $as, $imm16, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<8> imm16;
  bits<3> qx;
  bits<3> qy;
  
  let mayLoad = 1;
  
  let Inst{28-25} = 0xf;
  let Inst{24-23} = imm16{5-4};
  let Inst{22-20} = qu{2-0};
  let Inst{19-17} = 0x2;
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-8} = 0x0;
  let Inst{7-4} = imm16{3-0};
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMULAS_S8_ACCX_LD_IP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, offset_64_16:$imm16, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmulas_s8_accx_ld_ip_p, $qu, $as, $imm16, $qx, $qy",
			[(int_xtensa_ee_vmulas_s8_accx_ld_ip timm:$qu, AR:$as, timm:$imm16, timm:$qx, timm:$qy)]>;

def EE_VMULAS_S8_ACCX_LD_IP_QUP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qs0r), (ins AR:$as, offset_64_16:$imm16, QR:$qx, QR:$qy, QR:$qs0, QR:$qs1),
                   "ee.vmulas.s8.accx.ld.ip.qup\t $qu, $as, $imm16, $qx, $qy, $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<8> imm16;
  bits<3> qx;
  bits<3> qy;
  bits<3> qs0;
  bits<3> qs1;
  
  let mayLoad = 1;
  
  let Inst{28-25} = 0x2;
  let Inst{24-23} = imm16{5-4};
  let Inst{22-20} = qu{2-0};
  let Inst{19-17} = qs1{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-8} = qs0{2-0};
  let Inst{7-4} = imm16{3-0};
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMULAS_S8_ACCX_LD_IP_QUP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, offset_64_16:$imm16, imm8:$qx, imm8:$qy, imm8:$qs0, imm8:$qs1),
				"!xtensa_ee_vmulas_s8_accx_ld_ip_qup_p, $qu, $as, $imm16, $qx, $qy, $qs0, $qs1",
			[(int_xtensa_ee_vmulas_s8_accx_ld_ip_qup timm:$qu, AR:$as, timm:$imm16, timm:$qx, timm:$qy, timm:$qs0, timm:$qs1)]>;

def EE_VMULAS_S8_ACCX_LD_XP:  EE_Inst32<(outs QR:$qu, AR:$asr), (ins AR:$as, AR:$ad, QR:$qx, QR:$qy),
                   "ee.vmulas.s8.accx.ld.xp\t $qu, $as, $ad, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<4> ad;
  bits<3> qx;
  bits<3> qy;
  
  let mayLoad = 1;
  
  let Inst{28-23} = 0x3c;
  let Inst{22-20} = qu{2-0};
  let Inst{19-17} = 0x2;
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-8} = 0x1;
  let Inst{7-4} = ad{3-0};
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMULAS_S8_ACCX_LD_XP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, AR:$ad, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmulas_s8_accx_ld_xp_p, $qu, $as, $ad, $qx, $qy",
			[(int_xtensa_ee_vmulas_s8_accx_ld_xp timm:$qu, AR:$as, AR:$ad, timm:$qx, timm:$qy)]>;

def EE_VMULAS_S8_ACCX_LD_XP_QUP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qs0r), (ins AR:$as, AR:$ad, QR:$qx, QR:$qy, QR:$qs0, QR:$qs1),
                   "ee.vmulas.s8.accx.ld.xp.qup\t $qu, $as, $ad, $qx, $qy, $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<4> ad;
  bits<3> qx;
  bits<3> qy;
  bits<3> qs0;
  bits<3> qs1;
  
  let mayLoad = 1;
  
  let Inst{28-23} = 0x2e;
  let Inst{22-20} = qu{2-0};
  let Inst{19-17} = qs1{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-8} = qs0{2-0};
  let Inst{7-4} = ad{3-0};
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMULAS_S8_ACCX_LD_XP_QUP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, AR:$ad, imm8:$qx, imm8:$qy, imm8:$qs0, imm8:$qs1),
				"!xtensa_ee_vmulas_s8_accx_ld_xp_qup_p, $qu, $as, $ad, $qx, $qy, $qs0, $qs1",
			[(int_xtensa_ee_vmulas_s8_accx_ld_xp_qup timm:$qu, AR:$as, AR:$ad, timm:$qx, timm:$qy, timm:$qs0, timm:$qs1)]>;

def EE_VMULAS_S8_QACC:  EE_Inst24<(outs), (ins QR:$qx, QR:$qy),
                   "ee.vmulas.s8.qacc\t $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qx;
  bits<3> qy;
  
  
  let Inst{23-15} = 0x34;
  let Inst{14} = qy{2};
  let Inst{13} = 0x1;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0xc4;
}

let usesCustomInserter = 1 in
def EE_VMULAS_S8_QACC_P : Pseudo<(outs), (ins imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmulas_s8_qacc_p, $qx, $qy",
			[(int_xtensa_ee_vmulas_s8_qacc timm:$qx, timm:$qy)]>;

def EE_VMULAS_S8_QACC_LDBC_INCP:  EE_Inst24<(outs QR:$qu, AR:$asr), (ins AR:$as, QR:$qx, QR:$qy),
                   "ee.vmulas.s8.qacc.ldbc.incp\t $qu, $as, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qx;
  bits<3> qy;
  
  let mayLoad = 1;
  
  let Inst{23-21} = 0x5;
  let Inst{20} = qu{2};
  let Inst{19-16} = 0x7;
  let Inst{15} = qu{1};
  let Inst{14} = qy{2};
  let Inst{13} = qu{0};
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_VMULAS_S8_QACC_LDBC_INCP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmulas_s8_qacc_ldbc_incp_p, $qu, $as, $qx, $qy",
			[(int_xtensa_ee_vmulas_s8_qacc_ldbc_incp timm:$qu, AR:$as, timm:$qx, timm:$qy)]>;

def EE_VMULAS_S8_QACC_LDBC_INCP_QUP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qs0r), (ins AR:$as, QR:$qx, QR:$qy, QR:$qs0, QR:$qs1),
                   "ee.vmulas.s8.qacc.ldbc.incp.qup\t $qu, $as, $qx, $qy, $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qx;
  bits<3> qy;
  bits<3> qs0;
  bits<3> qs1;
  
  let mayLoad = 1;
  
  let Inst{28-23} = 0x38;
  let Inst{22-20} = qu{2-0};
  let Inst{19-17} = qs1{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-8} = qs0{2-0};
  let Inst{7-4} = 0x9;
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMULAS_S8_QACC_LDBC_INCP_QUP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, imm8:$qx, imm8:$qy, imm8:$qs0, imm8:$qs1),
				"!xtensa_ee_vmulas_s8_qacc_ldbc_incp_qup_p, $qu, $as, $qx, $qy, $qs0, $qs1",
			[(int_xtensa_ee_vmulas_s8_qacc_ldbc_incp_qup timm:$qu, AR:$as, timm:$qx, timm:$qy, timm:$qs0, timm:$qs1)]>;

def EE_VMULAS_S8_QACC_LD_IP:  EE_Inst32<(outs QR:$qu, AR:$asr), (ins AR:$as, offset_64_16:$imm16, QR:$qx, QR:$qy),
                   "ee.vmulas.s8.qacc.ld.ip\t $qu, $as, $imm16, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<8> imm16;
  bits<3> qx;
  bits<3> qy;
  
  let mayLoad = 1;
  
  let Inst{28-25} = 0xf;
  let Inst{24-23} = imm16{5-4};
  let Inst{22-20} = qu{2-0};
  let Inst{19-17} = 0x3;
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-8} = 0x0;
  let Inst{7-4} = imm16{3-0};
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMULAS_S8_QACC_LD_IP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, offset_64_16:$imm16, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmulas_s8_qacc_ld_ip_p, $qu, $as, $imm16, $qx, $qy",
			[(int_xtensa_ee_vmulas_s8_qacc_ld_ip timm:$qu, AR:$as, timm:$imm16, timm:$qx, timm:$qy)]>;

def EE_VMULAS_S8_QACC_LD_IP_QUP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qs0r), (ins AR:$as, offset_64_16:$imm16, QR:$qx, QR:$qy, QR:$qs0, QR:$qs1),
                   "ee.vmulas.s8.qacc.ld.ip.qup\t $qu, $as, $imm16, $qx, $qy, $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<8> imm16;
  bits<3> qx;
  bits<3> qy;
  bits<3> qs0;
  bits<3> qs1;
  
  let mayLoad = 1;
  
  let Inst{28-25} = 0x3;
  let Inst{24-23} = imm16{5-4};
  let Inst{22-20} = qu{2-0};
  let Inst{19-17} = qs1{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-8} = qs0{2-0};
  let Inst{7-4} = imm16{3-0};
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMULAS_S8_QACC_LD_IP_QUP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, offset_64_16:$imm16, imm8:$qx, imm8:$qy, imm8:$qs0, imm8:$qs1),
				"!xtensa_ee_vmulas_s8_qacc_ld_ip_qup_p, $qu, $as, $imm16, $qx, $qy, $qs0, $qs1",
			[(int_xtensa_ee_vmulas_s8_qacc_ld_ip_qup timm:$qu, AR:$as, timm:$imm16, timm:$qx, timm:$qy, timm:$qs0, timm:$qs1)]>;

def EE_VMULAS_S8_QACC_LD_XP:  EE_Inst32<(outs QR:$qu, AR:$asr), (ins AR:$as, AR:$ad, QR:$qx, QR:$qy),
                   "ee.vmulas.s8.qacc.ld.xp\t $qu, $as, $ad, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<4> ad;
  bits<3> qx;
  bits<3> qy;
  
  let mayLoad = 1;
  
  let Inst{28-23} = 0x3c;
  let Inst{22-20} = qu{2-0};
  let Inst{19-17} = 0x3;
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-8} = 0x1;
  let Inst{7-4} = ad{3-0};
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMULAS_S8_QACC_LD_XP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, AR:$ad, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmulas_s8_qacc_ld_xp_p, $qu, $as, $ad, $qx, $qy",
			[(int_xtensa_ee_vmulas_s8_qacc_ld_xp timm:$qu, AR:$as, AR:$ad, timm:$qx, timm:$qy)]>;

def EE_VMULAS_S8_QACC_LD_XP_QUP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qs0r), (ins AR:$as, AR:$ad, QR:$qx, QR:$qy, QR:$qs0, QR:$qs1),
                   "ee.vmulas.s8.qacc.ld.xp.qup\t $qu, $as, $ad, $qx, $qy, $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<4> ad;
  bits<3> qx;
  bits<3> qy;
  bits<3> qs0;
  bits<3> qs1;
  
  let mayLoad = 1;
  
  let Inst{28-23} = 0x2f;
  let Inst{22-20} = qu{2-0};
  let Inst{19-17} = qs1{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-8} = qs0{2-0};
  let Inst{7-4} = ad{3-0};
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMULAS_S8_QACC_LD_XP_QUP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, AR:$ad, imm8:$qx, imm8:$qy, imm8:$qs0, imm8:$qs1),
				"!xtensa_ee_vmulas_s8_qacc_ld_xp_qup_p, $qu, $as, $ad, $qx, $qy, $qs0, $qs1",
			[(int_xtensa_ee_vmulas_s8_qacc_ld_xp_qup timm:$qu, AR:$as, AR:$ad, timm:$qx, timm:$qy, timm:$qs0, timm:$qs1)]>;

def EE_VMULAS_U16_ACCX:  EE_Inst24<(outs), (ins QR:$qx, QR:$qy),
                   "ee.vmulas.u16.accx\t $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qx;
  bits<3> qy;
  
  
  let Inst{23-15} = 0x14;
  let Inst{14} = qy{2};
  let Inst{13} = 0x0;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0x84;
}

let usesCustomInserter = 1 in
def EE_VMULAS_U16_ACCX_P : Pseudo<(outs), (ins imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmulas_u16_accx_p, $qx, $qy",
			[(int_xtensa_ee_vmulas_u16_accx timm:$qx, timm:$qy)]>;

def EE_VMULAS_U16_ACCX_LD_IP:  EE_Inst32<(outs QR:$qu, AR:$asr), (ins AR:$as, offset_64_16:$imm16, QR:$qx, QR:$qy),
                   "ee.vmulas.u16.accx.ld.ip\t $qu, $as, $imm16, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<8> imm16;
  bits<3> qx;
  bits<3> qy;
  
  let mayLoad = 1;
  
  let Inst{28-25} = 0xf;
  let Inst{24-23} = imm16{5-4};
  let Inst{22-20} = qu{2-0};
  let Inst{19-17} = 0x4;
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-8} = 0x0;
  let Inst{7-4} = imm16{3-0};
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMULAS_U16_ACCX_LD_IP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, offset_64_16:$imm16, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmulas_u16_accx_ld_ip_p, $qu, $as, $imm16, $qx, $qy",
			[(int_xtensa_ee_vmulas_u16_accx_ld_ip timm:$qu, AR:$as, timm:$imm16, timm:$qx, timm:$qy)]>;

def EE_VMULAS_U16_ACCX_LD_IP_QUP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qs0r), (ins AR:$as, offset_64_16:$imm16, QR:$qx, QR:$qy, QR:$qs0, QR:$qs1),
                   "ee.vmulas.u16.accx.ld.ip.qup\t $qu, $as, $imm16, $qx, $qy, $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<8> imm16;
  bits<3> qx;
  bits<3> qy;
  bits<3> qs0;
  bits<3> qs1;
  
  let mayLoad = 1;
  
  let Inst{28-25} = 0x4;
  let Inst{24-23} = imm16{5-4};
  let Inst{22-20} = qu{2-0};
  let Inst{19-17} = qs1{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-8} = qs0{2-0};
  let Inst{7-4} = imm16{3-0};
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMULAS_U16_ACCX_LD_IP_QUP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, offset_64_16:$imm16, imm8:$qx, imm8:$qy, imm8:$qs0, imm8:$qs1),
				"!xtensa_ee_vmulas_u16_accx_ld_ip_qup_p, $qu, $as, $imm16, $qx, $qy, $qs0, $qs1",
			[(int_xtensa_ee_vmulas_u16_accx_ld_ip_qup timm:$qu, AR:$as, timm:$imm16, timm:$qx, timm:$qy, timm:$qs0, timm:$qs1)]>;

def EE_VMULAS_U16_ACCX_LD_XP:  EE_Inst32<(outs QR:$qu, AR:$asr), (ins AR:$as, AR:$ad, QR:$qx, QR:$qy),
                   "ee.vmulas.u16.accx.ld.xp\t $qu, $as, $ad, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<4> ad;
  bits<3> qx;
  bits<3> qy;
  
  let mayLoad = 1;
  
  let Inst{28-23} = 0x3c;
  let Inst{22-20} = qu{2-0};
  let Inst{19-17} = 0x4;
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-8} = 0x1;
  let Inst{7-4} = ad{3-0};
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMULAS_U16_ACCX_LD_XP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, AR:$ad, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmulas_u16_accx_ld_xp_p, $qu, $as, $ad, $qx, $qy",
			[(int_xtensa_ee_vmulas_u16_accx_ld_xp timm:$qu, AR:$as, AR:$ad, timm:$qx, timm:$qy)]>;

def EE_VMULAS_U16_ACCX_LD_XP_QUP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qs0r), (ins AR:$as, AR:$ad, QR:$qx, QR:$qy, QR:$qs0, QR:$qs1),
                   "ee.vmulas.u16.accx.ld.xp.qup\t $qu, $as, $ad, $qx, $qy, $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<4> ad;
  bits<3> qx;
  bits<3> qy;
  bits<3> qs0;
  bits<3> qs1;
  
  let mayLoad = 1;
  
  let Inst{28-23} = 0x30;
  let Inst{22-20} = qu{2-0};
  let Inst{19-17} = qs1{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-8} = qs0{2-0};
  let Inst{7-4} = ad{3-0};
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMULAS_U16_ACCX_LD_XP_QUP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, AR:$ad, imm8:$qx, imm8:$qy, imm8:$qs0, imm8:$qs1),
				"!xtensa_ee_vmulas_u16_accx_ld_xp_qup_p, $qu, $as, $ad, $qx, $qy, $qs0, $qs1",
			[(int_xtensa_ee_vmulas_u16_accx_ld_xp_qup timm:$qu, AR:$as, AR:$ad, timm:$qx, timm:$qy, timm:$qs0, timm:$qs1)]>;

def EE_VMULAS_U16_QACC:  EE_Inst24<(outs), (ins QR:$qx, QR:$qy),
                   "ee.vmulas.u16.qacc\t $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qx;
  bits<3> qy;
  
  
  let Inst{23-15} = 0x14;
  let Inst{14} = qy{2};
  let Inst{13} = 0x1;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0x84;
}

let usesCustomInserter = 1 in
def EE_VMULAS_U16_QACC_P : Pseudo<(outs), (ins imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmulas_u16_qacc_p, $qx, $qy",
			[(int_xtensa_ee_vmulas_u16_qacc timm:$qx, timm:$qy)]>;

def EE_VMULAS_U16_QACC_LDBC_INCP:  EE_Inst24<(outs QR:$qu, AR:$asr), (ins AR:$as, QR:$qx, QR:$qy),
                   "ee.vmulas.u16.qacc.ldbc.incp\t $qu, $as, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qx;
  bits<3> qy;
  
  let mayLoad = 1;
  
  let Inst{23-21} = 0x6;
  let Inst{20} = qu{2};
  let Inst{19-16} = 0x7;
  let Inst{15} = qu{1};
  let Inst{14} = qy{2};
  let Inst{13} = qu{0};
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_VMULAS_U16_QACC_LDBC_INCP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmulas_u16_qacc_ldbc_incp_p, $qu, $as, $qx, $qy",
			[(int_xtensa_ee_vmulas_u16_qacc_ldbc_incp timm:$qu, AR:$as, timm:$qx, timm:$qy)]>;

def EE_VMULAS_U16_QACC_LDBC_INCP_QUP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qs0r), (ins AR:$as, QR:$qx, QR:$qy, QR:$qs0, QR:$qs1),
                   "ee.vmulas.u16.qacc.ldbc.incp.qup\t $qu, $as, $qx, $qy, $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qx;
  bits<3> qy;
  bits<3> qs0;
  bits<3> qs1;
  
  let mayLoad = 1;
  
  let Inst{28-23} = 0x38;
  let Inst{22-20} = qu{2-0};
  let Inst{19-17} = qs1{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-8} = qs0{2-0};
  let Inst{7-4} = 0xa;
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMULAS_U16_QACC_LDBC_INCP_QUP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, imm8:$qx, imm8:$qy, imm8:$qs0, imm8:$qs1),
				"!xtensa_ee_vmulas_u16_qacc_ldbc_incp_qup_p, $qu, $as, $qx, $qy, $qs0, $qs1",
			[(int_xtensa_ee_vmulas_u16_qacc_ldbc_incp_qup timm:$qu, AR:$as, timm:$qx, timm:$qy, timm:$qs0, timm:$qs1)]>;

def EE_VMULAS_U16_QACC_LD_IP:  EE_Inst32<(outs QR:$qu, AR:$asr), (ins AR:$as, offset_64_16:$imm16, QR:$qx, QR:$qy),
                   "ee.vmulas.u16.qacc.ld.ip\t $qu, $as, $imm16, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<8> imm16;
  bits<3> qx;
  bits<3> qy;
  
  let mayLoad = 1;
  
  let Inst{28-25} = 0xf;
  let Inst{24-23} = imm16{5-4};
  let Inst{22-20} = qu{2-0};
  let Inst{19-17} = 0x5;
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-8} = 0x0;
  let Inst{7-4} = imm16{3-0};
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMULAS_U16_QACC_LD_IP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, offset_64_16:$imm16, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmulas_u16_qacc_ld_ip_p, $qu, $as, $imm16, $qx, $qy",
			[(int_xtensa_ee_vmulas_u16_qacc_ld_ip timm:$qu, AR:$as, timm:$imm16, timm:$qx, timm:$qy)]>;

def EE_VMULAS_U16_QACC_LD_IP_QUP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qs0r), (ins AR:$as, offset_64_16:$imm16, QR:$qx, QR:$qy, QR:$qs0, QR:$qs1),
                   "ee.vmulas.u16.qacc.ld.ip.qup\t $qu, $as, $imm16, $qx, $qy, $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<8> imm16;
  bits<3> qx;
  bits<3> qy;
  bits<3> qs0;
  bits<3> qs1;
  
  let mayLoad = 1;
  
  let Inst{28-25} = 0x5;
  let Inst{24-23} = imm16{5-4};
  let Inst{22-20} = qu{2-0};
  let Inst{19-17} = qs1{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-8} = qs0{2-0};
  let Inst{7-4} = imm16{3-0};
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMULAS_U16_QACC_LD_IP_QUP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, offset_64_16:$imm16, imm8:$qx, imm8:$qy, imm8:$qs0, imm8:$qs1),
				"!xtensa_ee_vmulas_u16_qacc_ld_ip_qup_p, $qu, $as, $imm16, $qx, $qy, $qs0, $qs1",
			[(int_xtensa_ee_vmulas_u16_qacc_ld_ip_qup timm:$qu, AR:$as, timm:$imm16, timm:$qx, timm:$qy, timm:$qs0, timm:$qs1)]>;

def EE_VMULAS_U16_QACC_LD_XP:  EE_Inst32<(outs QR:$qu, AR:$asr), (ins AR:$as, AR:$ad, QR:$qx, QR:$qy),
                   "ee.vmulas.u16.qacc.ld.xp\t $qu, $as, $ad, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<4> ad;
  bits<3> qx;
  bits<3> qy;
  
  let mayLoad = 1;
  
  let Inst{28-23} = 0x3c;
  let Inst{22-20} = qu{2-0};
  let Inst{19-17} = 0x5;
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-8} = 0x1;
  let Inst{7-4} = ad{3-0};
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMULAS_U16_QACC_LD_XP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, AR:$ad, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmulas_u16_qacc_ld_xp_p, $qu, $as, $ad, $qx, $qy",
			[(int_xtensa_ee_vmulas_u16_qacc_ld_xp timm:$qu, AR:$as, AR:$ad, timm:$qx, timm:$qy)]>;

def EE_VMULAS_U16_QACC_LD_XP_QUP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qs0r), (ins AR:$as, AR:$ad, QR:$qx, QR:$qy, QR:$qs0, QR:$qs1),
                   "ee.vmulas.u16.qacc.ld.xp.qup\t $qu, $as, $ad, $qx, $qy, $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<4> ad;
  bits<3> qx;
  bits<3> qy;
  bits<3> qs0;
  bits<3> qs1;
  
  let mayLoad = 1;
  
  let Inst{28-23} = 0x31;
  let Inst{22-20} = qu{2-0};
  let Inst{19-17} = qs1{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-8} = qs0{2-0};
  let Inst{7-4} = ad{3-0};
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMULAS_U16_QACC_LD_XP_QUP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, AR:$ad, imm8:$qx, imm8:$qy, imm8:$qs0, imm8:$qs1),
				"!xtensa_ee_vmulas_u16_qacc_ld_xp_qup_p, $qu, $as, $ad, $qx, $qy, $qs0, $qs1",
			[(int_xtensa_ee_vmulas_u16_qacc_ld_xp_qup timm:$qu, AR:$as, AR:$ad, timm:$qx, timm:$qy, timm:$qs0, timm:$qs1)]>;

def EE_VMULAS_U8_ACCX:  EE_Inst24<(outs), (ins QR:$qx, QR:$qy),
                   "ee.vmulas.u8.accx\t $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qx;
  bits<3> qy;
  
  
  let Inst{23-15} = 0x14;
  let Inst{14} = qy{2};
  let Inst{13} = 0x0;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0xc4;
}

let usesCustomInserter = 1 in
def EE_VMULAS_U8_ACCX_P : Pseudo<(outs), (ins imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmulas_u8_accx_p, $qx, $qy",
			[(int_xtensa_ee_vmulas_u8_accx timm:$qx, timm:$qy)]>;

def EE_VMULAS_U8_ACCX_LD_IP:  EE_Inst32<(outs QR:$qu, AR:$asr), (ins AR:$as, offset_64_16:$imm16, QR:$qx, QR:$qy),
                   "ee.vmulas.u8.accx.ld.ip\t $qu, $as, $imm16, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<8> imm16;
  bits<3> qx;
  bits<3> qy;
  
  let mayLoad = 1;
  
  let Inst{28-25} = 0xf;
  let Inst{24-23} = imm16{5-4};
  let Inst{22-20} = qu{2-0};
  let Inst{19-17} = 0x6;
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-8} = 0x0;
  let Inst{7-4} = imm16{3-0};
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMULAS_U8_ACCX_LD_IP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, offset_64_16:$imm16, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmulas_u8_accx_ld_ip_p, $qu, $as, $imm16, $qx, $qy",
			[(int_xtensa_ee_vmulas_u8_accx_ld_ip timm:$qu, AR:$as, timm:$imm16, timm:$qx, timm:$qy)]>;

def EE_VMULAS_U8_ACCX_LD_IP_QUP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qs0r), (ins AR:$as, offset_64_16:$imm16, QR:$qx, QR:$qy, QR:$qs0, QR:$qs1),
                   "ee.vmulas.u8.accx.ld.ip.qup\t $qu, $as, $imm16, $qx, $qy, $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<8> imm16;
  bits<3> qx;
  bits<3> qy;
  bits<3> qs0;
  bits<3> qs1;
  
  let mayLoad = 1;
  
  let Inst{28-25} = 0x6;
  let Inst{24-23} = imm16{5-4};
  let Inst{22-20} = qu{2-0};
  let Inst{19-17} = qs1{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-8} = qs0{2-0};
  let Inst{7-4} = imm16{3-0};
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMULAS_U8_ACCX_LD_IP_QUP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, offset_64_16:$imm16, imm8:$qx, imm8:$qy, imm8:$qs0, imm8:$qs1),
				"!xtensa_ee_vmulas_u8_accx_ld_ip_qup_p, $qu, $as, $imm16, $qx, $qy, $qs0, $qs1",
			[(int_xtensa_ee_vmulas_u8_accx_ld_ip_qup timm:$qu, AR:$as, timm:$imm16, timm:$qx, timm:$qy, timm:$qs0, timm:$qs1)]>;

def EE_VMULAS_U8_ACCX_LD_XP:  EE_Inst32<(outs QR:$qu, AR:$asr), (ins AR:$as, AR:$ad, QR:$qx, QR:$qy),
                   "ee.vmulas.u8.accx.ld.xp\t $qu, $as, $ad, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<4> ad;
  bits<3> qx;
  bits<3> qy;
  
  let mayLoad = 1;
  
  let Inst{28-23} = 0x3c;
  let Inst{22-20} = qu{2-0};
  let Inst{19-17} = 0x6;
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-8} = 0x1;
  let Inst{7-4} = ad{3-0};
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMULAS_U8_ACCX_LD_XP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, AR:$ad, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmulas_u8_accx_ld_xp_p, $qu, $as, $ad, $qx, $qy",
			[(int_xtensa_ee_vmulas_u8_accx_ld_xp timm:$qu, AR:$as, AR:$ad, timm:$qx, timm:$qy)]>;

def EE_VMULAS_U8_ACCX_LD_XP_QUP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qs0r), (ins AR:$as, AR:$ad, QR:$qx, QR:$qy, QR:$qs0, QR:$qs1),
                   "ee.vmulas.u8.accx.ld.xp.qup\t $qu, $as, $ad, $qx, $qy, $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<4> ad;
  bits<3> qx;
  bits<3> qy;
  bits<3> qs0;
  bits<3> qs1;
  
  let mayLoad = 1;
  
  let Inst{28-23} = 0x32;
  let Inst{22-20} = qu{2-0};
  let Inst{19-17} = qs1{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-8} = qs0{2-0};
  let Inst{7-4} = ad{3-0};
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMULAS_U8_ACCX_LD_XP_QUP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, AR:$ad, imm8:$qx, imm8:$qy, imm8:$qs0, imm8:$qs1),
				"!xtensa_ee_vmulas_u8_accx_ld_xp_qup_p, $qu, $as, $ad, $qx, $qy, $qs0, $qs1",
			[(int_xtensa_ee_vmulas_u8_accx_ld_xp_qup timm:$qu, AR:$as, AR:$ad, timm:$qx, timm:$qy, timm:$qs0, timm:$qs1)]>;

def EE_VMULAS_U8_QACC:  EE_Inst24<(outs), (ins QR:$qx, QR:$qy),
                   "ee.vmulas.u8.qacc\t $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qx;
  bits<3> qy;
  
  
  let Inst{23-15} = 0x14;
  let Inst{14} = qy{2};
  let Inst{13} = 0x1;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0xc4;
}

let usesCustomInserter = 1 in
def EE_VMULAS_U8_QACC_P : Pseudo<(outs), (ins imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmulas_u8_qacc_p, $qx, $qy",
			[(int_xtensa_ee_vmulas_u8_qacc timm:$qx, timm:$qy)]>;

def EE_VMULAS_U8_QACC_LDBC_INCP:  EE_Inst24<(outs QR:$qu, AR:$asr), (ins AR:$as, QR:$qx, QR:$qy),
                   "ee.vmulas.u8.qacc.ldbc.incp\t $qu, $as, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qx;
  bits<3> qy;
  
  let mayLoad = 1;
  
  let Inst{23-21} = 0x7;
  let Inst{20} = qu{2};
  let Inst{19-16} = 0x7;
  let Inst{15} = qu{1};
  let Inst{14} = qy{2};
  let Inst{13} = qu{0};
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_VMULAS_U8_QACC_LDBC_INCP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmulas_u8_qacc_ldbc_incp_p, $qu, $as, $qx, $qy",
			[(int_xtensa_ee_vmulas_u8_qacc_ldbc_incp timm:$qu, AR:$as, timm:$qx, timm:$qy)]>;

def EE_VMULAS_U8_QACC_LDBC_INCP_QUP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qs0r), (ins AR:$as, QR:$qx, QR:$qy, QR:$qs0, QR:$qs1),
                   "ee.vmulas.u8.qacc.ldbc.incp.qup\t $qu, $as, $qx, $qy, $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qx;
  bits<3> qy;
  bits<3> qs0;
  bits<3> qs1;
  
  let mayLoad = 1;
  
  let Inst{28-23} = 0x38;
  let Inst{22-20} = qu{2-0};
  let Inst{19-17} = qs1{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-8} = qs0{2-0};
  let Inst{7-4} = 0xb;
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMULAS_U8_QACC_LDBC_INCP_QUP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, imm8:$qx, imm8:$qy, imm8:$qs0, imm8:$qs1),
				"!xtensa_ee_vmulas_u8_qacc_ldbc_incp_qup_p, $qu, $as, $qx, $qy, $qs0, $qs1",
			[(int_xtensa_ee_vmulas_u8_qacc_ldbc_incp_qup timm:$qu, AR:$as, timm:$qx, timm:$qy, timm:$qs0, timm:$qs1)]>;

def EE_VMULAS_U8_QACC_LD_IP:  EE_Inst32<(outs QR:$qu, AR:$asr), (ins AR:$as, offset_64_16:$imm16, QR:$qx, QR:$qy),
                   "ee.vmulas.u8.qacc.ld.ip\t $qu, $as, $imm16, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<8> imm16;
  bits<3> qx;
  bits<3> qy;
  
  let mayLoad = 1;
  
  let Inst{28-25} = 0xf;
  let Inst{24-23} = imm16{5-4};
  let Inst{22-20} = qu{2-0};
  let Inst{19-17} = 0x7;
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-8} = 0x0;
  let Inst{7-4} = imm16{3-0};
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMULAS_U8_QACC_LD_IP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, offset_64_16:$imm16, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmulas_u8_qacc_ld_ip_p, $qu, $as, $imm16, $qx, $qy",
			[(int_xtensa_ee_vmulas_u8_qacc_ld_ip timm:$qu, AR:$as, timm:$imm16, timm:$qx, timm:$qy)]>;

def EE_VMULAS_U8_QACC_LD_IP_QUP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qs0r), (ins AR:$as, offset_64_16:$imm16, QR:$qx, QR:$qy, QR:$qs0, QR:$qs1),
                   "ee.vmulas.u8.qacc.ld.ip.qup\t $qu, $as, $imm16, $qx, $qy, $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<8> imm16;
  bits<3> qx;
  bits<3> qy;
  bits<3> qs0;
  bits<3> qs1;
  
  let mayLoad = 1;
  
  let Inst{28-25} = 0x7;
  let Inst{24-23} = imm16{5-4};
  let Inst{22-20} = qu{2-0};
  let Inst{19-17} = qs1{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-8} = qs0{2-0};
  let Inst{7-4} = imm16{3-0};
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMULAS_U8_QACC_LD_IP_QUP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, offset_64_16:$imm16, imm8:$qx, imm8:$qy, imm8:$qs0, imm8:$qs1),
				"!xtensa_ee_vmulas_u8_qacc_ld_ip_qup_p, $qu, $as, $imm16, $qx, $qy, $qs0, $qs1",
			[(int_xtensa_ee_vmulas_u8_qacc_ld_ip_qup timm:$qu, AR:$as, timm:$imm16, timm:$qx, timm:$qy, timm:$qs0, timm:$qs1)]>;

def EE_VMULAS_U8_QACC_LD_XP:  EE_Inst32<(outs QR:$qu, AR:$asr), (ins AR:$as, AR:$ad, QR:$qx, QR:$qy),
                   "ee.vmulas.u8.qacc.ld.xp\t $qu, $as, $ad, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<4> ad;
  bits<3> qx;
  bits<3> qy;
  
  let mayLoad = 1;
  
  let Inst{28-23} = 0x3c;
  let Inst{22-20} = qu{2-0};
  let Inst{19-17} = 0x7;
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-8} = 0x1;
  let Inst{7-4} = ad{3-0};
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMULAS_U8_QACC_LD_XP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, AR:$ad, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmulas_u8_qacc_ld_xp_p, $qu, $as, $ad, $qx, $qy",
			[(int_xtensa_ee_vmulas_u8_qacc_ld_xp timm:$qu, AR:$as, AR:$ad, timm:$qx, timm:$qy)]>;

def EE_VMULAS_U8_QACC_LD_XP_QUP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qs0r), (ins AR:$as, AR:$ad, QR:$qx, QR:$qy, QR:$qs0, QR:$qs1),
                   "ee.vmulas.u8.qacc.ld.xp.qup\t $qu, $as, $ad, $qx, $qy, $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<4> ad;
  bits<3> qx;
  bits<3> qy;
  bits<3> qs0;
  bits<3> qs1;
  
  let mayLoad = 1;
  
  let Inst{28-23} = 0x33;
  let Inst{22-20} = qu{2-0};
  let Inst{19-17} = qs1{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-8} = qs0{2-0};
  let Inst{7-4} = ad{3-0};
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMULAS_U8_QACC_LD_XP_QUP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, AR:$ad, imm8:$qx, imm8:$qy, imm8:$qs0, imm8:$qs1),
				"!xtensa_ee_vmulas_u8_qacc_ld_xp_qup_p, $qu, $as, $ad, $qx, $qy, $qs0, $qs1",
			[(int_xtensa_ee_vmulas_u8_qacc_ld_xp_qup timm:$qu, AR:$as, AR:$ad, timm:$qx, timm:$qy, timm:$qs0, timm:$qs1)]>;

def EE_VMUL_S16:  EE_Inst24<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "ee.vmul.s16\t $qz, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qz;
  bits<3> qx;
  bits<3> qy;
  
  
  let Inst{23-22} = 0x2;
  let Inst{21-20} = qz{2-1};
  let Inst{19-16} = 0xe;
  let Inst{15} = qz{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x1;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0x84;
}

let usesCustomInserter = 1 in
def EE_VMUL_S16_P : Pseudo<(outs), (ins imm8:$qz, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmul_s16_p, $qz, $qx, $qy",
			[(int_xtensa_ee_vmul_s16 timm:$qz, timm:$qx, timm:$qy)]>;

def EE_VMUL_S16_LD_INCP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qz), (ins AR:$as, QR:$qx, QR:$qy),
                   "ee.vmul.s16.ld.incp\t $qu, $as, $qz, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qz;
  bits<3> qx;
  bits<3> qy;
  
  let mayLoad = 1;
  
  let Inst{28-23} = 0x38;
  let Inst{22-20} = qu{2-0};
  let Inst{19-17} = qz{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-4} = 0x3f;
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMUL_S16_LD_INCP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, imm8:$qz, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmul_s16_ld_incp_p, $qu, $as, $qz, $qx, $qy",
			[(int_xtensa_ee_vmul_s16_ld_incp timm:$qu, AR:$as, timm:$qz, timm:$qx, timm:$qy)]>;

def EE_VMUL_S16_ST_INCP:  EE_Inst32<(outs AR:$asr, QR:$qz), (ins QR:$qv, AR:$as, QR:$qx, QR:$qy),
                   "ee.vmul.s16.st.incp\t $qv, $as, $qz, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qv;
  bits<4> as;
  bits<3> qz;
  bits<3> qx;
  bits<3> qy;
  
  let mayStore = 1;
  
  let Inst{28-20} = 0x1cb;
  let Inst{19-17} = qz{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-8} = qv{2-0};
  let Inst{7-4} = 0x2;
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMUL_S16_ST_INCP_P : Pseudo<(outs), (ins imm8:$qv, AR:$as, imm8:$qz, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmul_s16_st_incp_p, $qv, $as, $qz, $qx, $qy",
			[(int_xtensa_ee_vmul_s16_st_incp timm:$qv, AR:$as, timm:$qz, timm:$qx, timm:$qy)]>;

def EE_VMUL_S8:  EE_Inst24<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "ee.vmul.s8\t $qz, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qz;
  bits<3> qx;
  bits<3> qy;
  
  
  let Inst{23-22} = 0x2;
  let Inst{21-20} = qz{2-1};
  let Inst{19-16} = 0xe;
  let Inst{15} = qz{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x1;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0x94;
}

let usesCustomInserter = 1 in
def EE_VMUL_S8_P : Pseudo<(outs), (ins imm8:$qz, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmul_s8_p, $qz, $qx, $qy",
			[(int_xtensa_ee_vmul_s8 timm:$qz, timm:$qx, timm:$qy)]>;

def EE_VMUL_S8_LD_INCP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qz), (ins AR:$as, QR:$qx, QR:$qy),
                   "ee.vmul.s8.ld.incp\t $qu, $as, $qz, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qz;
  bits<3> qx;
  bits<3> qy;
  
  let mayLoad = 1;
  
  let Inst{28-23} = 0x38;
  let Inst{22-20} = qu{2-0};
  let Inst{19-17} = qz{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-4} = 0x4c;
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMUL_S8_LD_INCP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, imm8:$qz, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmul_s8_ld_incp_p, $qu, $as, $qz, $qx, $qy",
			[(int_xtensa_ee_vmul_s8_ld_incp timm:$qu, AR:$as, timm:$qz, timm:$qx, timm:$qy)]>;

def EE_VMUL_S8_ST_INCP:  EE_Inst32<(outs AR:$asr, QR:$qz), (ins QR:$qv, AR:$as, QR:$qx, QR:$qy),
                   "ee.vmul.s8.st.incp\t $qv, $as, $qz, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qv;
  bits<4> as;
  bits<3> qz;
  bits<3> qx;
  bits<3> qy;
  
  let mayStore = 1;
  
  let Inst{28-20} = 0x1ca;
  let Inst{19-17} = qz{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-8} = qv{2-0};
  let Inst{7-4} = 0x3;
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMUL_S8_ST_INCP_P : Pseudo<(outs), (ins imm8:$qv, AR:$as, imm8:$qz, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmul_s8_st_incp_p, $qv, $as, $qz, $qx, $qy",
			[(int_xtensa_ee_vmul_s8_st_incp timm:$qv, AR:$as, timm:$qz, timm:$qx, timm:$qy)]>;

def EE_VMUL_U16:  EE_Inst24<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "ee.vmul.u16\t $qz, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qz;
  bits<3> qx;
  bits<3> qy;
  
  
  let Inst{23-22} = 0x2;
  let Inst{21-20} = qz{2-1};
  let Inst{19-16} = 0xe;
  let Inst{15} = qz{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x1;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0xa4;
}

let usesCustomInserter = 1 in
def EE_VMUL_U16_P : Pseudo<(outs), (ins imm8:$qz, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmul_u16_p, $qz, $qx, $qy",
			[(int_xtensa_ee_vmul_u16 timm:$qz, timm:$qx, timm:$qy)]>;

def EE_VMUL_U16_LD_INCP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qz), (ins AR:$as, QR:$qx, QR:$qy),
                   "ee.vmul.u16.ld.incp\t $qu, $as, $qz, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qz;
  bits<3> qx;
  bits<3> qy;
  
  let mayLoad = 1;
  
  let Inst{28-23} = 0x38;
  let Inst{22-20} = qu{2-0};
  let Inst{19-17} = qz{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-4} = 0x5c;
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMUL_U16_LD_INCP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, imm8:$qz, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmul_u16_ld_incp_p, $qu, $as, $qz, $qx, $qy",
			[(int_xtensa_ee_vmul_u16_ld_incp timm:$qu, AR:$as, timm:$qz, timm:$qx, timm:$qy)]>;

def EE_VMUL_U16_ST_INCP:  EE_Inst32<(outs AR:$asr, QR:$qz), (ins QR:$qv, AR:$as, QR:$qx, QR:$qy),
                   "ee.vmul.u16.st.incp\t $qv, $as, $qz, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qv;
  bits<4> as;
  bits<3> qz;
  bits<3> qx;
  bits<3> qy;
  
  let mayStore = 1;
  
  let Inst{28-20} = 0x1cb;
  let Inst{19-17} = qz{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-8} = qv{2-0};
  let Inst{7-4} = 0x3;
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMUL_U16_ST_INCP_P : Pseudo<(outs), (ins imm8:$qv, AR:$as, imm8:$qz, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmul_u16_st_incp_p, $qv, $as, $qz, $qx, $qy",
			[(int_xtensa_ee_vmul_u16_st_incp timm:$qv, AR:$as, timm:$qz, timm:$qx, timm:$qy)]>;

def EE_VMUL_U8:  EE_Inst24<(outs QR:$qz), (ins QR:$qx, QR:$qy),
                   "ee.vmul.u8\t $qz, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qz;
  bits<3> qx;
  bits<3> qy;
  
  
  let Inst{23-22} = 0x2;
  let Inst{21-20} = qz{2-1};
  let Inst{19-16} = 0xe;
  let Inst{15} = qz{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x1;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0xb4;
}

let usesCustomInserter = 1 in
def EE_VMUL_U8_P : Pseudo<(outs), (ins imm8:$qz, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmul_u8_p, $qz, $qx, $qy",
			[(int_xtensa_ee_vmul_u8 timm:$qz, timm:$qx, timm:$qy)]>;

def EE_VMUL_U8_LD_INCP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qz), (ins AR:$as, QR:$qx, QR:$qy),
                   "ee.vmul.u8.ld.incp\t $qu, $as, $qz, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qz;
  bits<3> qx;
  bits<3> qy;
  
  let mayLoad = 1;
  
  let Inst{28-23} = 0x38;
  let Inst{22-20} = qu{2-0};
  let Inst{19-17} = qz{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-4} = 0x6c;
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMUL_U8_LD_INCP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, imm8:$qz, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmul_u8_ld_incp_p, $qu, $as, $qz, $qx, $qy",
			[(int_xtensa_ee_vmul_u8_ld_incp timm:$qu, AR:$as, timm:$qz, timm:$qx, timm:$qy)]>;

def EE_VMUL_U8_ST_INCP:  EE_Inst32<(outs AR:$asr, QR:$qz), (ins QR:$qv, AR:$as, QR:$qx, QR:$qy),
                   "ee.vmul.u8.st.incp\t $qv, $as, $qz, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qv;
  bits<4> as;
  bits<3> qz;
  bits<3> qx;
  bits<3> qy;
  
  let mayStore = 1;
  
  let Inst{28-20} = 0x1d1;
  let Inst{19-17} = qz{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-8} = qv{2-0};
  let Inst{7-4} = 0x0;
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VMUL_U8_ST_INCP_P : Pseudo<(outs), (ins imm8:$qv, AR:$as, imm8:$qz, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vmul_u8_st_incp_p, $qv, $as, $qz, $qx, $qy",
			[(int_xtensa_ee_vmul_u8_st_incp timm:$qv, AR:$as, timm:$qz, timm:$qx, timm:$qy)]>;

def EE_VPRELU_S16:  EE_Inst24<(outs QR:$qz), (ins QR:$qx, QR:$qy, AR:$ay),
                   "ee.vprelu.s16\t $qz, $qx, $qy, $ay", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qz;
  bits<3> qx;
  bits<3> qy;
  bits<4> ay;
  
  
  let Inst{23-22} = 0x2;
  let Inst{21-20} = qz{2-1};
  let Inst{19-16} = 0xc;
  let Inst{15} = qz{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x0;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-4} = ay{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_VPRELU_S16_P : Pseudo<(outs), (ins imm8:$qz, imm8:$qx, imm8:$qy, AR:$ay),
				"!xtensa_ee_vprelu_s16_p, $qz, $qx, $qy, $ay",
			[(int_xtensa_ee_vprelu_s16 timm:$qz, timm:$qx, timm:$qy, AR:$ay)]>;

def EE_VPRELU_S8:  EE_Inst24<(outs QR:$qz), (ins QR:$qx, QR:$qy, AR:$ay),
                   "ee.vprelu.s8\t $qz, $qx, $qy, $ay", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qz;
  bits<3> qx;
  bits<3> qy;
  bits<4> ay;
  
  
  let Inst{23-22} = 0x2;
  let Inst{21-20} = qz{2-1};
  let Inst{19-16} = 0xc;
  let Inst{15} = qz{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x1;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-4} = ay{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_VPRELU_S8_P : Pseudo<(outs), (ins imm8:$qz, imm8:$qx, imm8:$qy, AR:$ay),
				"!xtensa_ee_vprelu_s8_p, $qz, $qx, $qy, $ay",
			[(int_xtensa_ee_vprelu_s8 timm:$qz, timm:$qx, timm:$qy, AR:$ay)]>;

def EE_VRELU_S16:  EE_Inst24<(outs QR:$qsr), (ins QR:$qs, AR:$ax, AR:$ay),
                   "ee.vrelu.s16\t $qs, $ax, $ay", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qs;
  bits<4> ax;
  bits<4> ay;
  
  
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qs{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qs{0};
  let Inst{14-12} = 0x1;
  let Inst{11-8} = ax{3-0};
  let Inst{7-4} = ay{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_VRELU_S16_P : Pseudo<(outs), (ins imm8:$qs, AR:$ax, AR:$ay),
				"!xtensa_ee_vrelu_s16_p, $qs, $ax, $ay",
			[(int_xtensa_ee_vrelu_s16 timm:$qs, AR:$ax, AR:$ay)]>;

def EE_VRELU_S8:  EE_Inst24<(outs QR:$qsr), (ins QR:$qs, AR:$ax, AR:$ay),
                   "ee.vrelu.s8\t $qs, $ax, $ay", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qs;
  bits<4> ax;
  bits<4> ay;
  
  
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qs{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qs{0};
  let Inst{14-12} = 0x5;
  let Inst{11-8} = ax{3-0};
  let Inst{7-4} = ay{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_VRELU_S8_P : Pseudo<(outs), (ins imm8:$qs, AR:$ax, AR:$ay),
				"!xtensa_ee_vrelu_s8_p, $qs, $ax, $ay",
			[(int_xtensa_ee_vrelu_s8 timm:$qs, AR:$ax, AR:$ay)]>;

def EE_VSL_32:  EE_Inst24<(outs QR:$qa), (ins QR:$qs),
                   "ee.vsl.32\t $qa, $qs", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qs;
  
  
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qs{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qs{0};
  let Inst{14-7} = 0x7e;
  let Inst{6-4} = qa{2-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_VSL_32_P : Pseudo<(outs), (ins imm8:$qa, imm8:$qs),
				"!xtensa_ee_vsl_32_p, $qa, $qs",
			[(int_xtensa_ee_vsl_32 timm:$qa, timm:$qs)]>;

def EE_VSMULAS_S16_QACC:  EE_Inst24<(outs), (ins QR:$qx, QR:$qy, select_8:$sel8),
                   "ee.vsmulas.s16.qacc\t $qx, $qy, $sel8", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qx;
  bits<3> qy;
  bits<3> sel8;
  
  
  let Inst{23-22} = 0x2;
  let Inst{21-20} = sel8{2-1};
  let Inst{19-16} = 0xe;
  let Inst{15} = sel8{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x1;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0xc4;
}

let usesCustomInserter = 1 in
def EE_VSMULAS_S16_QACC_P : Pseudo<(outs), (ins imm8:$qx, imm8:$qy, select_8:$sel8),
				"!xtensa_ee_vsmulas_s16_qacc_p, $qx, $qy, $sel8",
			[(int_xtensa_ee_vsmulas_s16_qacc timm:$qx, timm:$qy, timm:$sel8)]>;

def EE_VSMULAS_S16_QACC_LD_INCP:  EE_Inst32<(outs QR:$qu, AR:$asr), (ins AR:$as, QR:$qx, QR:$qy, select_8:$sel8),
                   "ee.vsmulas.s16.qacc.ld.incp\t $qu, $as, $qx, $qy, $sel8", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qx;
  bits<3> qy;
  bits<3> sel8;
  
  let mayLoad = 1;
  
  let Inst{28-23} = 0x38;
  let Inst{22-20} = qu{2-0};
  let Inst{19-17} = sel8{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-4} = 0x7c;
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VSMULAS_S16_QACC_LD_INCP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, imm8:$qx, imm8:$qy, select_8:$sel8),
				"!xtensa_ee_vsmulas_s16_qacc_ld_incp_p, $qu, $as, $qx, $qy, $sel8",
			[(int_xtensa_ee_vsmulas_s16_qacc_ld_incp timm:$qu, AR:$as, timm:$qx, timm:$qy, timm:$sel8)]>;

def EE_VSMULAS_S8_QACC:  EE_Inst24<(outs), (ins QR:$qx, QR:$qy, select_16:$sel16),
                   "ee.vsmulas.s8.qacc\t $qx, $qy, $sel16", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qx;
  bits<3> qy;
  bits<4> sel16;
  
  
  let Inst{23-22} = 0x2;
  let Inst{21-20} = sel16{3-2};
  let Inst{19-16} = 0xe;
  let Inst{15} = sel16{1};
  let Inst{14} = qy{2};
  let Inst{13} = 0x0;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-5} = 0x2;
  let Inst{4} = sel16{0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_VSMULAS_S8_QACC_P : Pseudo<(outs), (ins imm8:$qx, imm8:$qy, select_16:$sel16),
				"!xtensa_ee_vsmulas_s8_qacc_p, $qx, $qy, $sel16",
			[(int_xtensa_ee_vsmulas_s8_qacc timm:$qx, timm:$qy, timm:$sel16)]>;

def EE_VSMULAS_S8_QACC_LD_INCP:  EE_Inst32<(outs QR:$qu, AR:$asr), (ins AR:$as, QR:$qx, QR:$qy, select_16:$sel16),
                   "ee.vsmulas.s8.qacc.ld.incp\t $qu, $as, $qx, $qy, $sel16", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qx;
  bits<3> qy;
  bits<4> sel16;
  
  let mayLoad = 1;
  
  let Inst{28-23} = 0x38;
  let Inst{22-20} = qu{2-0};
  let Inst{19-17} = sel16{3-1};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-9} = 0x1;
  let Inst{8} = sel16{0};
  let Inst{7-4} = 0xc;
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VSMULAS_S8_QACC_LD_INCP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, imm8:$qx, imm8:$qy, select_16:$sel16),
				"!xtensa_ee_vsmulas_s8_qacc_ld_incp_p, $qu, $as, $qx, $qy, $sel16",
			[(int_xtensa_ee_vsmulas_s8_qacc_ld_incp timm:$qu, AR:$as, timm:$qx, timm:$qy, timm:$sel16)]>;

def EE_VSR_32:  EE_Inst24<(outs QR:$qa), (ins QR:$qs),
                   "ee.vsr.32\t $qa, $qs", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qs;
  
  
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qs{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qs{0};
  let Inst{14-7} = 0x7f;
  let Inst{6-4} = qa{2-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_VSR_32_P : Pseudo<(outs), (ins imm8:$qa, imm8:$qs),
				"!xtensa_ee_vsr_32_p, $qa, $qs",
			[(int_xtensa_ee_vsr_32 timm:$qa, timm:$qs)]>;

def EE_VST_128_IP:  EE_Inst24<(outs AR:$asr), (ins QR:$qv, AR:$as, offset_256_16:$imm16),
                   "ee.vst.128.ip\t $qv, $as, $imm16", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qv;
  bits<4> as;
  bits<8> imm16;
  
  let mayStore = 1;
  
  let Inst{23} = 0x1;
  let Inst{22} = imm16{7};
  let Inst{21-20} = qv{2-1};
  let Inst{19-16} = 0xa;
  let Inst{15} = qv{0};
  let Inst{14-8} = imm16{6-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_VST_128_IP_P : Pseudo<(outs), (ins imm8:$qv, AR:$as, offset_256_16:$imm16),
				"!xtensa_ee_vst_128_ip_p, $qv, $as, $imm16",
			[(int_xtensa_ee_vst_128_ip timm:$qv, AR:$as, timm:$imm16)]>;

def EE_VST_128_XP:  EE_Inst24<(outs AR:$asr), (ins QR:$qv, AR:$as, AR:$ad),
                   "ee.vst.128.xp\t $qv, $as, $ad", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qv;
  bits<4> as;
  bits<4> ad;
  
  let mayStore = 1;
  
  let Inst{23-22} = 0x2;
  let Inst{21-20} = qv{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qv{0};
  let Inst{14-12} = 0x7;
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_VST_128_XP_P : Pseudo<(outs), (ins imm8:$qv, AR:$as, AR:$ad),
				"!xtensa_ee_vst_128_xp_p, $qv, $as, $ad",
			[(int_xtensa_ee_vst_128_xp timm:$qv, AR:$as, AR:$ad)]>;

def EE_VST_H_64_IP:  EE_Inst24<(outs AR:$asr), (ins QR:$qv, AR:$as, offset_256_8:$imm8),
                   "ee.vst.h.64.ip\t $qv, $as, $imm8", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qv;
  bits<4> as;
  bits<8> imm8;
  
  let mayStore = 1;
  
  let Inst{23} = 0x1;
  let Inst{22} = imm8{7};
  let Inst{21-20} = qv{2-1};
  let Inst{19-16} = 0xb;
  let Inst{15} = qv{0};
  let Inst{14-8} = imm8{6-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_VST_H_64_IP_P : Pseudo<(outs), (ins imm8:$qv, AR:$as, offset_256_8:$imm8),
				"!xtensa_ee_vst_h_64_ip_p, $qv, $as, $imm8",
			[(int_xtensa_ee_vst_h_64_ip timm:$qv, AR:$as, timm:$imm8)]>;

def EE_VST_H_64_XP:  EE_Inst24<(outs AR:$asr), (ins QR:$qv, AR:$as, AR:$ad),
                   "ee.vst.h.64.xp\t $qv, $as, $ad", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qv;
  bits<4> as;
  bits<4> ad;
  
  let mayStore = 1;
  
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qv{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qv{0};
  let Inst{14-12} = 0x0;
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_VST_H_64_XP_P : Pseudo<(outs), (ins imm8:$qv, AR:$as, AR:$ad),
				"!xtensa_ee_vst_h_64_xp_p, $qv, $as, $ad",
			[(int_xtensa_ee_vst_h_64_xp timm:$qv, AR:$as, AR:$ad)]>;

def EE_VST_L_64_IP:  EE_Inst24<(outs AR:$asr), (ins QR:$qv, AR:$as, offset_256_8:$imm8),
                   "ee.vst.l.64.ip\t $qv, $as, $imm8", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qv;
  bits<4> as;
  bits<8> imm8;
  
  let mayStore = 1;
  
  let Inst{23} = 0x1;
  let Inst{22} = imm8{7};
  let Inst{21-20} = qv{2-1};
  let Inst{19-16} = 0x4;
  let Inst{15} = qv{0};
  let Inst{14-8} = imm8{6-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_VST_L_64_IP_P : Pseudo<(outs), (ins imm8:$qv, AR:$as, offset_256_8:$imm8),
				"!xtensa_ee_vst_l_64_ip_p, $qv, $as, $imm8",
			[(int_xtensa_ee_vst_l_64_ip timm:$qv, AR:$as, timm:$imm8)]>;

def EE_VST_L_64_XP:  EE_Inst24<(outs AR:$asr), (ins QR:$qv, AR:$as, AR:$ad),
                   "ee.vst.l.64.xp\t $qv, $as, $ad", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qv;
  bits<4> as;
  bits<4> ad;
  
  let mayStore = 1;
  
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qv{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qv{0};
  let Inst{14-12} = 0x4;
  let Inst{11-8} = ad{3-0};
  let Inst{7-4} = as{3-0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_VST_L_64_XP_P : Pseudo<(outs), (ins imm8:$qv, AR:$as, AR:$ad),
				"!xtensa_ee_vst_l_64_xp_p, $qv, $as, $ad",
			[(int_xtensa_ee_vst_l_64_xp timm:$qv, AR:$as, AR:$ad)]>;

def EE_VSUBS_S16:  EE_Inst24<(outs QR:$qa), (ins QR:$qx, QR:$qy),
                   "ee.vsubs.s16\t $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;
  
  
  let Inst{23-22} = 0x2;
  let Inst{21-20} = qa{2-1};
  let Inst{19-16} = 0xe;
  let Inst{15} = qa{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x1;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0xd4;
}

let usesCustomInserter = 1 in
def EE_VSUBS_S16_P : Pseudo<(outs), (ins imm8:$qa, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vsubs_s16_p, $qa, $qx, $qy",
			[(int_xtensa_ee_vsubs_s16 timm:$qa, timm:$qx, timm:$qy)]>;

def EE_VSUBS_S16_LD_INCP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qa), (ins AR:$as, QR:$qx, QR:$qy),
                   "ee.vsubs.s16.ld.incp\t $qu, $as, $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;
  
  let mayLoad = 1;
  
  let Inst{28-23} = 0x38;
  let Inst{22-20} = qu{2-0};
  let Inst{19-17} = qa{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-4} = 0x4d;
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VSUBS_S16_LD_INCP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, imm8:$qa, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vsubs_s16_ld_incp_p, $qu, $as, $qa, $qx, $qy",
			[(int_xtensa_ee_vsubs_s16_ld_incp timm:$qu, AR:$as, timm:$qa, timm:$qx, timm:$qy)]>;

def EE_VSUBS_S16_ST_INCP:  EE_Inst32<(outs AR:$asr, QR:$qa), (ins QR:$qv, AR:$as, QR:$qx, QR:$qy),
                   "ee.vsubs.s16.st.incp\t $qv, $as, $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qv;
  bits<4> as;
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;
  
  let mayStore = 1;
  
  let Inst{28-20} = 0x1d1;
  let Inst{19-17} = qa{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-8} = qv{2-0};
  let Inst{7-4} = 0x1;
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VSUBS_S16_ST_INCP_P : Pseudo<(outs), (ins imm8:$qv, AR:$as, imm8:$qa, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vsubs_s16_st_incp_p, $qv, $as, $qa, $qx, $qy",
			[(int_xtensa_ee_vsubs_s16_st_incp timm:$qv, AR:$as, timm:$qa, timm:$qx, timm:$qy)]>;

def EE_VSUBS_S32:  EE_Inst24<(outs QR:$qa), (ins QR:$qx, QR:$qy),
                   "ee.vsubs.s32\t $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;
  
  
  let Inst{23-22} = 0x2;
  let Inst{21-20} = qa{2-1};
  let Inst{19-16} = 0xe;
  let Inst{15} = qa{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x1;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0xe4;
}

let usesCustomInserter = 1 in
def EE_VSUBS_S32_P : Pseudo<(outs), (ins imm8:$qa, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vsubs_s32_p, $qa, $qx, $qy",
			[(int_xtensa_ee_vsubs_s32 timm:$qa, timm:$qx, timm:$qy)]>;

def EE_VSUBS_S32_LD_INCP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qa), (ins AR:$as, QR:$qx, QR:$qy),
                   "ee.vsubs.s32.ld.incp\t $qu, $as, $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;
  
  let mayLoad = 1;
  
  let Inst{28-23} = 0x38;
  let Inst{22-20} = qu{2-0};
  let Inst{19-17} = qa{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-4} = 0x5d;
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VSUBS_S32_LD_INCP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, imm8:$qa, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vsubs_s32_ld_incp_p, $qu, $as, $qa, $qx, $qy",
			[(int_xtensa_ee_vsubs_s32_ld_incp timm:$qu, AR:$as, timm:$qa, timm:$qx, timm:$qy)]>;

def EE_VSUBS_S32_ST_INCP:  EE_Inst32<(outs AR:$asr, QR:$qa), (ins QR:$qv, AR:$as, QR:$qx, QR:$qy),
                   "ee.vsubs.s32.st.incp\t $qv, $as, $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qv;
  bits<4> as;
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;
  
  let mayStore = 1;
  
  let Inst{28-20} = 0x1d1;
  let Inst{19-17} = qa{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-8} = qv{2-0};
  let Inst{7-4} = 0x2;
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VSUBS_S32_ST_INCP_P : Pseudo<(outs), (ins imm8:$qv, AR:$as, imm8:$qa, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vsubs_s32_st_incp_p, $qv, $as, $qa, $qx, $qy",
			[(int_xtensa_ee_vsubs_s32_st_incp timm:$qv, AR:$as, timm:$qa, timm:$qx, timm:$qy)]>;

def EE_VSUBS_S8:  EE_Inst24<(outs QR:$qa), (ins QR:$qx, QR:$qy),
                   "ee.vsubs.s8\t $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;
  
  
  let Inst{23-22} = 0x2;
  let Inst{21-20} = qa{2-1};
  let Inst{19-16} = 0xe;
  let Inst{15} = qa{0};
  let Inst{14} = qy{2};
  let Inst{13} = 0x1;
  let Inst{12-11} = qy{1-0};
  let Inst{10-8} = qx{2-0};
  let Inst{7-0} = 0xf4;
}

let usesCustomInserter = 1 in
def EE_VSUBS_S8_P : Pseudo<(outs), (ins imm8:$qa, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vsubs_s8_p, $qa, $qx, $qy",
			[(int_xtensa_ee_vsubs_s8 timm:$qa, timm:$qx, timm:$qy)]>;

def EE_VSUBS_S8_LD_INCP:  EE_Inst32<(outs QR:$qu, AR:$asr, QR:$qa), (ins AR:$as, QR:$qx, QR:$qy),
                   "ee.vsubs.s8.ld.incp\t $qu, $as, $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qu;
  bits<4> as;
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;
  
  let mayLoad = 1;
  
  let Inst{28-23} = 0x38;
  let Inst{22-20} = qu{2-0};
  let Inst{19-17} = qa{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-4} = 0x6d;
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VSUBS_S8_LD_INCP_P : Pseudo<(outs), (ins imm8:$qu, AR:$as, imm8:$qa, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vsubs_s8_ld_incp_p, $qu, $as, $qa, $qx, $qy",
			[(int_xtensa_ee_vsubs_s8_ld_incp timm:$qu, AR:$as, timm:$qa, timm:$qx, timm:$qy)]>;

def EE_VSUBS_S8_ST_INCP:  EE_Inst32<(outs AR:$asr, QR:$qa), (ins QR:$qv, AR:$as, QR:$qx, QR:$qy),
                   "ee.vsubs.s8.st.incp\t $qv, $as, $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qv;
  bits<4> as;
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;
  
  let mayStore = 1;
  
  let Inst{28-20} = 0x1d1;
  let Inst{19-17} = qa{2-0};
  let Inst{16-14} = qx{2-0};
  let Inst{13-11} = qy{2-0};
  let Inst{10-8} = qv{2-0};
  let Inst{7-4} = 0x3;
  let Inst{3-0} = as{3-0};
}

let usesCustomInserter = 1 in
def EE_VSUBS_S8_ST_INCP_P : Pseudo<(outs), (ins imm8:$qv, AR:$as, imm8:$qa, imm8:$qx, imm8:$qy),
				"!xtensa_ee_vsubs_s8_st_incp_p, $qv, $as, $qa, $qx, $qy",
			[(int_xtensa_ee_vsubs_s8_st_incp timm:$qv, AR:$as, timm:$qa, timm:$qx, timm:$qy)]>;

def EE_VUNZIP_16:  EE_Inst24<(outs QR:$qs0r, QR:$qs1r), (ins QR:$qs0, QR:$qs1),
                   "ee.vunzip.16\t $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qs0;
  bits<3> qs1;
  
  
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qs1{2-1};
  let Inst{19-16} = 0xc;
  let Inst{15} = qs1{0};
  let Inst{14-12} = qs0{2-0};
  let Inst{11-0} = 0x384;
}

let usesCustomInserter = 1 in
def EE_VUNZIP_16_P : Pseudo<(outs), (ins imm8:$qs0, imm8:$qs1),
				"!xtensa_ee_vunzip_16_p, $qs0, $qs1",
			[(int_xtensa_ee_vunzip_16 timm:$qs0, timm:$qs1)]>;

def EE_VUNZIP_32:  EE_Inst24<(outs QR:$qs0r, QR:$qs1r), (ins QR:$qs0, QR:$qs1),
                   "ee.vunzip.32\t $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qs0;
  bits<3> qs1;
  
  
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qs1{2-1};
  let Inst{19-16} = 0xc;
  let Inst{15} = qs1{0};
  let Inst{14-12} = qs0{2-0};
  let Inst{11-0} = 0x394;
}

let usesCustomInserter = 1 in
def EE_VUNZIP_32_P : Pseudo<(outs), (ins imm8:$qs0, imm8:$qs1),
				"!xtensa_ee_vunzip_32_p, $qs0, $qs1",
			[(int_xtensa_ee_vunzip_32 timm:$qs0, timm:$qs1)]>;

def EE_VUNZIP_8:  EE_Inst24<(outs QR:$qs0r, QR:$qs1r), (ins QR:$qs0, QR:$qs1),
                   "ee.vunzip.8\t $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qs0;
  bits<3> qs1;
  
  
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qs1{2-1};
  let Inst{19-16} = 0xc;
  let Inst{15} = qs1{0};
  let Inst{14-12} = qs0{2-0};
  let Inst{11-0} = 0x3a4;
}

let usesCustomInserter = 1 in
def EE_VUNZIP_8_P : Pseudo<(outs), (ins imm8:$qs0, imm8:$qs1),
				"!xtensa_ee_vunzip_8_p, $qs0, $qs1",
			[(int_xtensa_ee_vunzip_8 timm:$qs0, timm:$qs1)]>;

def EE_VZIP_16:  EE_Inst24<(outs QR:$qs0r, QR:$qs1r), (ins QR:$qs0, QR:$qs1),
                   "ee.vzip.16\t $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qs0;
  bits<3> qs1;
  
  
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qs1{2-1};
  let Inst{19-16} = 0xc;
  let Inst{15} = qs1{0};
  let Inst{14-12} = qs0{2-0};
  let Inst{11-0} = 0x3b4;
}

let usesCustomInserter = 1 in
def EE_VZIP_16_P : Pseudo<(outs), (ins imm8:$qs0, imm8:$qs1),
				"!xtensa_ee_vzip_16_p, $qs0, $qs1",
			[(int_xtensa_ee_vzip_16 timm:$qs0, timm:$qs1)]>;

def EE_VZIP_32:  EE_Inst24<(outs QR:$qs0r, QR:$qs1r), (ins QR:$qs0, QR:$qs1),
                   "ee.vzip.32\t $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qs0;
  bits<3> qs1;
  
  
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qs1{2-1};
  let Inst{19-16} = 0xc;
  let Inst{15} = qs1{0};
  let Inst{14-12} = qs0{2-0};
  let Inst{11-0} = 0x3c4;
}

let usesCustomInserter = 1 in
def EE_VZIP_32_P : Pseudo<(outs), (ins imm8:$qs0, imm8:$qs1),
				"!xtensa_ee_vzip_32_p, $qs0, $qs1",
			[(int_xtensa_ee_vzip_32 timm:$qs0, timm:$qs1)]>;

def EE_VZIP_8:  EE_Inst24<(outs QR:$qs0r, QR:$qs1r), (ins QR:$qs0, QR:$qs1),
                   "ee.vzip.8\t $qs0, $qs1", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qs0;
  bits<3> qs1;
  
  
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qs1{2-1};
  let Inst{19-16} = 0xc;
  let Inst{15} = qs1{0};
  let Inst{14-12} = qs0{2-0};
  let Inst{11-0} = 0x3d4;
}

let usesCustomInserter = 1 in
def EE_VZIP_8_P : Pseudo<(outs), (ins imm8:$qs0, imm8:$qs1),
				"!xtensa_ee_vzip_8_p, $qs0, $qs1",
			[(int_xtensa_ee_vzip_8 timm:$qs0, timm:$qs1)]>;

def EE_XORQ:  EE_Inst24<(outs QR:$qa), (ins QR:$qx, QR:$qy),
                   "ee.xorq\t $qa, $qx, $qy", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  bits<3> qx;
  bits<3> qy;
  
  
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qa{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qa{0};
  let Inst{14-12} = 0x3;
  let Inst{11-10} = qy{2-1};
  let Inst{9-8} = 0x1;
  let Inst{7-6} = qx{2-1};
  let Inst{5} = qy{0};
  let Inst{4} = qx{0};
  let Inst{3-0} = 0x4;
}

let usesCustomInserter = 1 in
def EE_XORQ_P : Pseudo<(outs), (ins imm8:$qa, imm8:$qx, imm8:$qy),
				"!xtensa_ee_xorq_p, $qa, $qx, $qy",
			[(int_xtensa_ee_xorq timm:$qa, timm:$qx, timm:$qy)]>;

def EE_ZERO_ACCX:  EE_Inst24<(outs), (ins),
                   "ee.zero.accx\t", []>, Requires<[HasESP32S3Ops]>
{
  
  
  let Inst{23-0} = 0x250804;
}

let usesCustomInserter = 1 in
def EE_ZERO_ACCX_P : Pseudo<(outs), (ins),
				"!xtensa_ee_zero_accx_p",
			[(int_xtensa_ee_zero_accx)]>;

def EE_ZERO_Q:  EE_Inst24<(outs QR:$qa), (ins),
                   "ee.zero.q\t $qa", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> qa;
  
  
  let Inst{23-22} = 0x3;
  let Inst{21-20} = qa{2-1};
  let Inst{19-16} = 0xd;
  let Inst{15} = qa{0};
  let Inst{14-0} = 0x7fa4;
}

let usesCustomInserter = 1 in
def EE_ZERO_Q_P : Pseudo<(outs), (ins imm8:$qa),
				"!xtensa_ee_zero_q_p, $qa",
			[(int_xtensa_ee_zero_q timm:$qa)]>;

def EE_ZERO_QACC:  EE_Inst24<(outs), (ins),
                   "ee.zero.qacc\t", []>, Requires<[HasESP32S3Ops]>
{
  
  
  let Inst{23-0} = 0x250844;
}

let usesCustomInserter = 1 in
def EE_ZERO_QACC_P : Pseudo<(outs), (ins),
				"!xtensa_ee_zero_qacc_p",
			[(int_xtensa_ee_zero_qacc)]>;

def RUR_ACCX_0:  EE_Inst24<(outs AR:$arr), (ins),
                   "rur.accx_0\t $arr", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> arr;
  
  
  let Inst{23-16} = 0xe3;
  let Inst{15-12} = arr{3-0};
  let Inst{11-0} = 0x0;
}

def : Pat<(i32 (int_xtensa_rur_accx_0)), (RUR_ACCX_0)>;

def RUR_ACCX_1:  EE_Inst24<(outs AR:$arr), (ins),
                   "rur.accx_1\t $arr", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> arr;
  
  
  let Inst{23-16} = 0xe3;
  let Inst{15-12} = arr{3-0};
  let Inst{11-0} = 0x10;
}

def : Pat<(i32 (int_xtensa_rur_accx_1)), (RUR_ACCX_1)>;

def RUR_FFT_BIT_WIDTH:  EE_Inst24<(outs AR:$arr), (ins),
                   "rur.fft_bit_width\t $arr", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> arr;
  
  
  let Inst{23-16} = 0xe3;
  let Inst{15-12} = arr{3-0};
  let Inst{11-0} = 0xe0;
}

def : Pat<(i32 (int_xtensa_rur_fft_bit_width)), (RUR_FFT_BIT_WIDTH)>;

def RUR_GPIO_OUT:  EE_Inst24<(outs AR:$arr), (ins),
                   "rur.gpio_out\t $arr", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> arr;
  
  
  let Inst{23-16} = 0xe3;
  let Inst{15-12} = arr{3-0};
  let Inst{11-0} = 0xc0;
}

def : Pat<(i32 (int_xtensa_rur_gpio_out)), (RUR_GPIO_OUT)>;

def RUR_QACC_H_0:  EE_Inst24<(outs AR:$arr), (ins),
                   "rur.qacc_h_0\t $arr", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> arr;
  
  
  let Inst{23-16} = 0xe3;
  let Inst{15-12} = arr{3-0};
  let Inst{11-0} = 0x20;
}

def : Pat<(i32 (int_xtensa_rur_qacc_h_0)), (RUR_QACC_H_0)>;

def RUR_QACC_H_1:  EE_Inst24<(outs AR:$arr), (ins),
                   "rur.qacc_h_1\t $arr", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> arr;
  
  
  let Inst{23-16} = 0xe3;
  let Inst{15-12} = arr{3-0};
  let Inst{11-0} = 0x30;
}

def : Pat<(i32 (int_xtensa_rur_qacc_h_1)), (RUR_QACC_H_1)>;

def RUR_QACC_H_2:  EE_Inst24<(outs AR:$arr), (ins),
                   "rur.qacc_h_2\t $arr", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> arr;
  
  
  let Inst{23-16} = 0xe3;
  let Inst{15-12} = arr{3-0};
  let Inst{11-0} = 0x40;
}

def : Pat<(i32 (int_xtensa_rur_qacc_h_2)), (RUR_QACC_H_2)>;

def RUR_QACC_H_3:  EE_Inst24<(outs AR:$arr), (ins),
                   "rur.qacc_h_3\t $arr", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> arr;
  
  
  let Inst{23-16} = 0xe3;
  let Inst{15-12} = arr{3-0};
  let Inst{11-0} = 0x50;
}

def : Pat<(i32 (int_xtensa_rur_qacc_h_3)), (RUR_QACC_H_3)>;

def RUR_QACC_H_4:  EE_Inst24<(outs AR:$arr), (ins),
                   "rur.qacc_h_4\t $arr", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> arr;
  
  
  let Inst{23-16} = 0xe3;
  let Inst{15-12} = arr{3-0};
  let Inst{11-0} = 0x60;
}

def : Pat<(i32 (int_xtensa_rur_qacc_h_4)), (RUR_QACC_H_4)>;

def RUR_QACC_L_0:  EE_Inst24<(outs AR:$arr), (ins),
                   "rur.qacc_l_0\t $arr", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> arr;
  
  
  let Inst{23-16} = 0xe3;
  let Inst{15-12} = arr{3-0};
  let Inst{11-0} = 0x70;
}

def : Pat<(i32 (int_xtensa_rur_qacc_l_0)), (RUR_QACC_L_0)>;

def RUR_QACC_L_1:  EE_Inst24<(outs AR:$arr), (ins),
                   "rur.qacc_l_1\t $arr", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> arr;
  
  
  let Inst{23-16} = 0xe3;
  let Inst{15-12} = arr{3-0};
  let Inst{11-0} = 0x80;
}

def : Pat<(i32 (int_xtensa_rur_qacc_l_1)), (RUR_QACC_L_1)>;

def RUR_QACC_L_2:  EE_Inst24<(outs AR:$arr), (ins),
                   "rur.qacc_l_2\t $arr", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> arr;
  
  
  let Inst{23-16} = 0xe3;
  let Inst{15-12} = arr{3-0};
  let Inst{11-0} = 0x90;
}

def : Pat<(i32 (int_xtensa_rur_qacc_l_2)), (RUR_QACC_L_2)>;

def RUR_QACC_L_3:  EE_Inst24<(outs AR:$arr), (ins),
                   "rur.qacc_l_3\t $arr", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> arr;
  
  
  let Inst{23-16} = 0xe3;
  let Inst{15-12} = arr{3-0};
  let Inst{11-0} = 0xa0;
}

def : Pat<(i32 (int_xtensa_rur_qacc_l_3)), (RUR_QACC_L_3)>;

def RUR_QACC_L_4:  EE_Inst24<(outs AR:$arr), (ins),
                   "rur.qacc_l_4\t $arr", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> arr;
  
  
  let Inst{23-16} = 0xe3;
  let Inst{15-12} = arr{3-0};
  let Inst{11-0} = 0xb0;
}

def : Pat<(i32 (int_xtensa_rur_qacc_l_4)), (RUR_QACC_L_4)>;

def RUR_SAR_BYTE:  EE_Inst24<(outs AR:$arr), (ins),
                   "rur.sar_byte\t $arr", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> arr;
  
  
  let Inst{23-16} = 0xe3;
  let Inst{15-12} = arr{3-0};
  let Inst{11-0} = 0xd0;
}

def : Pat<(i32 (int_xtensa_rur_sar_byte)), (RUR_SAR_BYTE)>;

def RUR_UA_STATE_0:  EE_Inst24<(outs AR:$arr), (ins),
                   "rur.ua_state_0\t $arr", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> arr;
  
  
  let Inst{23-16} = 0xe3;
  let Inst{15-12} = arr{3-0};
  let Inst{11-0} = 0xf0;
}

def : Pat<(i32 (int_xtensa_rur_ua_state_0)), (RUR_UA_STATE_0)>;

def RUR_UA_STATE_1:  EE_Inst24<(outs AR:$arr), (ins),
                   "rur.ua_state_1\t $arr", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> arr;
  
  
  let Inst{23-16} = 0xe3;
  let Inst{15-12} = arr{3-0};
  let Inst{11-0} = 0x100;
}

def : Pat<(i32 (int_xtensa_rur_ua_state_1)), (RUR_UA_STATE_1)>;

def RUR_UA_STATE_2:  EE_Inst24<(outs AR:$arr), (ins),
                   "rur.ua_state_2\t $arr", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> arr;
  
  
  let Inst{23-16} = 0xe3;
  let Inst{15-12} = arr{3-0};
  let Inst{11-0} = 0x110;
}

def : Pat<(i32 (int_xtensa_rur_ua_state_2)), (RUR_UA_STATE_2)>;

def RUR_UA_STATE_3:  EE_Inst24<(outs AR:$arr), (ins),
                   "rur.ua_state_3\t $arr", []>, Requires<[HasESP32S3Ops]>
{
  bits<4> arr;
  
  
  let Inst{23-16} = 0xe3;
  let Inst{15-12} = arr{3-0};
  let Inst{11-0} = 0x120;
}

def : Pat<(i32 (int_xtensa_rur_ua_state_3)), (RUR_UA_STATE_3)>;

def WUR_ACCX_0:  EE_Inst24<(outs), (ins AR:$art),
                   "wur.accx_0\t $art", [(int_xtensa_wur_accx_0 AR:$art)]>, Requires<[HasESP32S3Ops]>
{
  bits<4> art;
  
  
  let Inst{23-8} = 0xf300;
  let Inst{7-4} = art{3-0};
  let Inst{3-0} = 0x0;
}

def WUR_ACCX_1:  EE_Inst24<(outs), (ins AR:$art),
                   "wur.accx_1\t $art", [(int_xtensa_wur_accx_1 AR:$art)]>, Requires<[HasESP32S3Ops]>
{
  bits<4> art;
  
  
  let Inst{23-8} = 0xf301;
  let Inst{7-4} = art{3-0};
  let Inst{3-0} = 0x0;
}

def WUR_FFT_BIT_WIDTH:  EE_Inst24<(outs), (ins AR:$art),
                   "wur.fft_bit_width\t $art", [(int_xtensa_wur_fft_bit_width AR:$art)]>, Requires<[HasESP32S3Ops]>
{
  bits<4> art;
  
  
  let Inst{23-8} = 0xf30e;
  let Inst{7-4} = art{3-0};
  let Inst{3-0} = 0x0;
}

def WUR_GPIO_OUT:  EE_Inst24<(outs), (ins AR:$art),
                   "wur.gpio_out\t $art", [(int_xtensa_wur_gpio_out AR:$art)]>, Requires<[HasESP32S3Ops]>
{
  bits<4> art;
  
  
  let Inst{23-8} = 0xf30c;
  let Inst{7-4} = art{3-0};
  let Inst{3-0} = 0x0;
}

def WUR_QACC_H_0:  EE_Inst24<(outs), (ins AR:$art),
                   "wur.qacc_h_0\t $art", [(int_xtensa_wur_qacc_h_0 AR:$art)]>, Requires<[HasESP32S3Ops]>
{
  bits<4> art;
  
  
  let Inst{23-8} = 0xf302;
  let Inst{7-4} = art{3-0};
  let Inst{3-0} = 0x0;
}

def WUR_QACC_H_1:  EE_Inst24<(outs), (ins AR:$art),
                   "wur.qacc_h_1\t $art", [(int_xtensa_wur_qacc_h_1 AR:$art)]>, Requires<[HasESP32S3Ops]>
{
  bits<4> art;
  
  
  let Inst{23-8} = 0xf303;
  let Inst{7-4} = art{3-0};
  let Inst{3-0} = 0x0;
}

def WUR_QACC_H_2:  EE_Inst24<(outs), (ins AR:$art),
                   "wur.qacc_h_2\t $art", [(int_xtensa_wur_qacc_h_2 AR:$art)]>, Requires<[HasESP32S3Ops]>
{
  bits<4> art;
  
  
  let Inst{23-8} = 0xf304;
  let Inst{7-4} = art{3-0};
  let Inst{3-0} = 0x0;
}

def WUR_QACC_H_3:  EE_Inst24<(outs), (ins AR:$art),
                   "wur.qacc_h_3\t $art", [(int_xtensa_wur_qacc_h_3 AR:$art)]>, Requires<[HasESP32S3Ops]>
{
  bits<4> art;
  
  
  let Inst{23-8} = 0xf305;
  let Inst{7-4} = art{3-0};
  let Inst{3-0} = 0x0;
}

def WUR_QACC_H_4:  EE_Inst24<(outs), (ins AR:$art),
                   "wur.qacc_h_4\t $art", [(int_xtensa_wur_qacc_h_4 AR:$art)]>, Requires<[HasESP32S3Ops]>
{
  bits<4> art;
  
  
  let Inst{23-8} = 0xf306;
  let Inst{7-4} = art{3-0};
  let Inst{3-0} = 0x0;
}

def WUR_QACC_L_0:  EE_Inst24<(outs), (ins AR:$art),
                   "wur.qacc_l_0\t $art", [(int_xtensa_wur_qacc_l_0 AR:$art)]>, Requires<[HasESP32S3Ops]>
{
  bits<4> art;
  
  
  let Inst{23-8} = 0xf307;
  let Inst{7-4} = art{3-0};
  let Inst{3-0} = 0x0;
}

def WUR_QACC_L_1:  EE_Inst24<(outs), (ins AR:$art),
                   "wur.qacc_l_1\t $art", [(int_xtensa_wur_qacc_l_1 AR:$art)]>, Requires<[HasESP32S3Ops]>
{
  bits<4> art;
  
  
  let Inst{23-8} = 0xf308;
  let Inst{7-4} = art{3-0};
  let Inst{3-0} = 0x0;
}

def WUR_QACC_L_2:  EE_Inst24<(outs), (ins AR:$art),
                   "wur.qacc_l_2\t $art", [(int_xtensa_wur_qacc_l_2 AR:$art)]>, Requires<[HasESP32S3Ops]>
{
  bits<4> art;
  
  
  let Inst{23-8} = 0xf309;
  let Inst{7-4} = art{3-0};
  let Inst{3-0} = 0x0;
}

def WUR_QACC_L_3:  EE_Inst24<(outs), (ins AR:$art),
                   "wur.qacc_l_3\t $art", [(int_xtensa_wur_qacc_l_3 AR:$art)]>, Requires<[HasESP32S3Ops]>
{
  bits<4> art;
  
  
  let Inst{23-8} = 0xf30a;
  let Inst{7-4} = art{3-0};
  let Inst{3-0} = 0x0;
}

def WUR_QACC_L_4:  EE_Inst24<(outs), (ins AR:$art),
                   "wur.qacc_l_4\t $art", [(int_xtensa_wur_qacc_l_4 AR:$art)]>, Requires<[HasESP32S3Ops]>
{
  bits<4> art;
  
  
  let Inst{23-8} = 0xf30b;
  let Inst{7-4} = art{3-0};
  let Inst{3-0} = 0x0;
}

def WUR_SAR_BYTE:  EE_Inst24<(outs), (ins AR:$art),
                   "wur.sar_byte\t $art", [(int_xtensa_wur_sar_byte AR:$art)]>, Requires<[HasESP32S3Ops]>
{
  bits<4> art;
  
  
  let Inst{23-8} = 0xf30d;
  let Inst{7-4} = art{3-0};
  let Inst{3-0} = 0x0;
}

def WUR_UA_STATE_0:  EE_Inst24<(outs), (ins AR:$art),
                   "wur.ua_state_0\t $art", [(int_xtensa_wur_ua_state_0 AR:$art)]>, Requires<[HasESP32S3Ops]>
{
  bits<4> art;
  
  
  let Inst{23-8} = 0xf30f;
  let Inst{7-4} = art{3-0};
  let Inst{3-0} = 0x0;
}

def WUR_UA_STATE_1:  EE_Inst24<(outs), (ins AR:$art),
                   "wur.ua_state_1\t $art", [(int_xtensa_wur_ua_state_1 AR:$art)]>, Requires<[HasESP32S3Ops]>
{
  bits<4> art;
  
  
  let Inst{23-8} = 0xf310;
  let Inst{7-4} = art{3-0};
  let Inst{3-0} = 0x0;
}

def WUR_UA_STATE_2:  EE_Inst24<(outs), (ins AR:$art),
                   "wur.ua_state_2\t $art", [(int_xtensa_wur_ua_state_2 AR:$art)]>, Requires<[HasESP32S3Ops]>
{
  bits<4> art;
  
  
  let Inst{23-8} = 0xf311;
  let Inst{7-4} = art{3-0};
  let Inst{3-0} = 0x0;
}

def WUR_UA_STATE_3:  EE_Inst24<(outs), (ins AR:$art),
                   "wur.ua_state_3\t $art", [(int_xtensa_wur_ua_state_3 AR:$art)]>, Requires<[HasESP32S3Ops]>
{
  bits<4> art;
  
  
  let Inst{23-8} = 0xf312;
  let Inst{7-4} = art{3-0};
  let Inst{3-0} = 0x0;
}

def mv_QR:  EE_Inst24<(outs QR:$a), (ins QR:$b),
                   "mv.qr\t $a, $b", []>, Requires<[HasESP32S3Ops]>
{
  bits<3> a;
  bits<3> b;
  
  
  let Inst{23-22} = 0x2;
  let Inst{21-20} = a{2-1};
  let Inst{19-16} = 0xf;
  let Inst{15} = a{0};
  let Inst{14-12} = 0x0;
  let Inst{11-10} = b{2-1};
  let Inst{9-6} = 0x0;
  let Inst{5} = b{0};
  let Inst{4-0} = 0x4;
}

let usesCustomInserter = 1 in
def mv_QR_P : Pseudo<(outs), (ins imm8:$a, imm8:$b),
				"!xtensa_mv_qr_p, $a, $b",
			[(int_xtensa_mv_qr timm:$a, timm:$b)]>;