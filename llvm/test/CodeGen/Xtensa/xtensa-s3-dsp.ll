; RUN: llc -O1 -mtriple=xtensa -mcpu=esp32s3 %s -o - | FileCheck %s

; CHECK: @test
define void @test(){
	tail call void @llvm.xtensa.ee.andq(i32 7, i32 0, i32 1)
	; CHECK: ee.andq
	tail call void @llvm.xtensa.ee.bitrev(i32 0, i32 3)
	; CHECK: ee.bitrev
	tail call void @llvm.xtensa.ee.cmul.s16(i32 2, i32 3, i32 2, i32 1)
	; CHECK: ee.cmul.s16
	tail call void @llvm.xtensa.ee.cmul.s16.ld.incp(i32 2, i32 1, i32 7, i32 5, i32 0, i32 3)
	; CHECK: ee.cmul.s16.ld.incp
	tail call void @llvm.xtensa.ee.cmul.s16.st.incp(i32 1, i32 6, i32 6, i32 1, i32 1, i32 1)
	; CHECK: ee.cmul.s16.st.incp
	tail call void @llvm.xtensa.ee.fft.ams.s16.ld.incp(i32 4, i32 10, i32 4, i32 2, i32 2, i32 2, i32 0, i32 1)
	; CHECK: ee.fft.ams.s16.ld.incp
	tail call void @llvm.xtensa.ee.fft.ams.s16.ld.incp.uaup(i32 0, i32 6, i32 1, i32 6, i32 2, i32 2, i32 6, i32 1)
	; CHECK: ee.fft.ams.s16.ld.incp.uaup
	tail call void @llvm.xtensa.ee.fft.ams.s16.ld.r32.decp(i32 7, i32 2, i32 5, i32 6, i32 1, i32 4, i32 0, i32 1)
	; CHECK: ee.fft.ams.s16.ld.r32.decp
	tail call void @llvm.xtensa.ee.fft.ams.s16.st.incp(i32 1, i32 1, i32 14, i32 2, i32 7, i32 0, i32 1, i32 0)
	; CHECK: ee.fft.ams.s16.st.incp
	tail call void @llvm.xtensa.ee.fft.cmul.s16.ld.xp(i32 7, i32 4, i32 9, i32 0, i32 2, i32 7, i32 1)
	; CHECK: ee.fft.cmul.s16.ld.xp
	tail call void @llvm.xtensa.ee.fft.cmul.s16.st.xp(i32 3, i32 0, i32 5, i32 11, i32 8, i32 7, i32 1, i32 0)
	; CHECK: ee.fft.cmul.s16.st.xp
	tail call void @llvm.xtensa.ee.fft.r2bf.s16(i32 7, i32 7, i32 2, i32 0, i32 1)
	; CHECK: ee.fft.r2bf.s16
	tail call void @llvm.xtensa.ee.fft.r2bf.s16.st.incp(i32 0, i32 5, i32 1, i32 10, i32 0)
	; CHECK: ee.fft.r2bf.s16.st.incp
	tail call void @llvm.xtensa.ee.fft.vst.r32.decp(i32 0, i32 10, i32 1)
	; CHECK: ee.fft.vst.r32.decp
	tail call void @llvm.xtensa.ee.ldf.128.ip(float 0x40099999A0000000, float 0x40099999A0000000, float 0x40099999A0000000, float 0x40099999A0000000, i32 0, i32 64)
	; CHECK: ee.ldf.128.ip
	tail call void @llvm.xtensa.ee.ldf.128.xp(float 0x40099999A0000000, float 0x40099999A0000000, float 0x40099999A0000000, float 0x40099999A0000000, i32 2, i32 13)
	; CHECK: ee.ldf.128.xp
	tail call void @llvm.xtensa.ee.ldf.64.ip(float 0x40099999A0000000, float 0x40099999A0000000, i32 10, i32 -232)
	; CHECK: ee.ldf.64.ip
	tail call void @llvm.xtensa.ee.ldf.64.xp(float 0x40099999A0000000, float 0x40099999A0000000, i32 13, i32 8)
	; CHECK: ee.ldf.64.xp
	tail call void @llvm.xtensa.ee.ldqa.s16.128.ip(i32 5, i32 160)
	; CHECK: ee.ldqa.s16.128.ip
	tail call void @llvm.xtensa.ee.ldqa.s16.128.xp(i32 13, i32 5)
	; CHECK: ee.ldqa.s16.128.xp
	tail call void @llvm.xtensa.ee.ldqa.s8.128.ip(i32 13, i32 640)
	; CHECK: ee.ldqa.s8.128.ip
	tail call void @llvm.xtensa.ee.ldqa.s8.128.xp(i32 2, i32 13)
	; CHECK: ee.ldqa.s8.128.xp
	tail call void @llvm.xtensa.ee.ldqa.u16.128.ip(i32 11, i32 -1184)
	; CHECK: ee.ldqa.u16.128.ip
	tail call void @llvm.xtensa.ee.ldqa.u16.128.xp(i32 1, i32 14)
	; CHECK: ee.ldqa.u16.128.xp
	tail call void @llvm.xtensa.ee.ldqa.u8.128.ip(i32 14, i32 736)
	; CHECK: ee.ldqa.u8.128.ip
	tail call void @llvm.xtensa.ee.ldqa.u8.128.xp(i32 2, i32 13)
	; CHECK: ee.ldqa.u8.128.xp
	tail call void @llvm.xtensa.ee.ldxq.32(i32 2, i32 4, i32 0, i32 1, i32 0)
	; CHECK: ee.ldxq.32
	tail call void @llvm.xtensa.ee.ld.128.usar.ip(i32 6, i32 7, i32 1904)
	; CHECK: ee.ld.128.usar.ip
	tail call void @llvm.xtensa.ee.ld.128.usar.xp(i32 0, i32 10, i32 8)
	; CHECK: ee.ld.128.usar.xp
	tail call void @llvm.xtensa.ee.ld.accx.ip(i32 6, i32 -560)
	; CHECK: ee.ld.accx.ip
	tail call void @llvm.xtensa.ee.ld.qacc.h.h.32.ip(i32 6, i32 68)
	; CHECK: ee.ld.qacc_h.h.32.ip
	tail call void @llvm.xtensa.ee.ld.qacc.h.l.128.ip(i32 7, i32 304)
	; CHECK: ee.ld.qacc_h.l.128.ip
	tail call void @llvm.xtensa.ee.ld.qacc.l.h.32.ip(i32 4, i32 52)
	; CHECK: ee.ld.qacc_l.h.32.ip
	tail call void @llvm.xtensa.ee.ld.qacc.l.l.128.ip(i32 7, i32 1040)
	; CHECK: ee.ld.qacc_l.l.128.ip
	tail call void @llvm.xtensa.ee.ld.ua.state.ip(i32 7, i32 -1760)
	; CHECK: ee.ld.ua_state.ip
	tail call void @llvm.xtensa.ee.movi.32.a(i32 0, i32 7, i32 0)
	; CHECK: ee.movi.32.a
	tail call void @llvm.xtensa.ee.movi.32.q(i32 0, i32 1, i32 2)
	; CHECK: ee.movi.32.q
	tail call void @llvm.xtensa.ee.mov.s16.qacc(i32 0)
	; CHECK: ee.mov.s16.qacc
	tail call void @llvm.xtensa.ee.mov.s8.qacc(i32 0)
	; CHECK: ee.mov.s8.qacc
	tail call void @llvm.xtensa.ee.mov.u16.qacc(i32 2)
	; CHECK: ee.mov.u16.qacc
	tail call void @llvm.xtensa.ee.mov.u8.qacc(i32 6)
	; CHECK: ee.mov.u8.qacc
	tail call void @llvm.xtensa.ee.notq(i32 6, i32 7)
	; CHECK: ee.notq
	tail call void @llvm.xtensa.ee.orq(i32 1, i32 2, i32 5)
	; CHECK: ee.orq
	tail call void @llvm.xtensa.ee.slci.2q(i32 5, i32 2, i32 8)
	; CHECK: ee.slci.2q
	tail call void @llvm.xtensa.ee.slcxxp.2q(i32 2, i32 4, i32 2, i32 11)
	; CHECK: ee.slcxxp.2q
	tail call void @llvm.xtensa.ee.srci.2q(i32 4, i32 0, i32 7)
	; CHECK: ee.srci.2q
	tail call void @llvm.xtensa.ee.srcmb.s16.qacc(i32 6, i32 5, i32 0)
	; CHECK: ee.srcmb.s16.qacc
	tail call void @llvm.xtensa.ee.srcmb.s8.qacc(i32 1, i32 7, i32 1)
	; CHECK: ee.srcmb.s8.qacc
	tail call void @llvm.xtensa.ee.srcq.128.st.incp(i32 5, i32 3, i32 6)
	; CHECK: ee.srcq.128.st.incp
	tail call void @llvm.xtensa.ee.srcxxp.2q(i32 1, i32 1, i32 1, i32 11)
	; CHECK: ee.srcxxp.2q
	tail call void @llvm.xtensa.ee.src.q(i32 7, i32 3, i32 0)
	; CHECK: ee.src.q
	tail call void @llvm.xtensa.ee.src.q.ld.ip(i32 1, i32 10, i32 1856, i32 4, i32 0)
	; CHECK: ee.src.q.ld.ip
	tail call void @llvm.xtensa.ee.src.q.ld.xp(i32 7, i32 1, i32 1, i32 0, i32 3)
	; CHECK: ee.src.q.ld.xp
	tail call void @llvm.xtensa.ee.src.q.qup(i32 5, i32 2, i32 0)
	; CHECK: ee.src.q.qup
	tail call void @llvm.xtensa.ee.srs.accx(i32 3, i32 6, i32 1)
	; CHECK: ee.srs.accx
	tail call void @llvm.xtensa.ee.stf.128.ip(float 0x40099999A0000000, float 0x40099999A0000000, float 0x40099999A0000000, float 0x40099999A0000000, i32 10, i32 -96)
	; CHECK: ee.stf.128.ip
	tail call void @llvm.xtensa.ee.stf.128.xp(float 0x40099999A0000000, float 0x40099999A0000000, float 0x40099999A0000000, float 0x40099999A0000000, i32 1, i32 6)
	; CHECK: ee.stf.128.xp
	tail call void @llvm.xtensa.ee.stf.64.ip(float 0x40099999A0000000, float 0x40099999A0000000, i32 8, i32 184)
	; CHECK: ee.stf.64.ip
	tail call void @llvm.xtensa.ee.stf.64.xp(float 0x40099999A0000000, float 0x40099999A0000000, i32 1, i32 7)
	; CHECK: ee.stf.64.xp
	tail call void @llvm.xtensa.ee.stxq.32(i32 2, i32 5, i32 2, i32 0, i32 4)
	; CHECK: ee.stxq.32
	tail call void @llvm.xtensa.ee.st.accx.ip(i32 4, i32 -136)
	; CHECK: ee.st.accx.ip
	tail call void @llvm.xtensa.ee.st.qacc.h.h.32.ip(i32 14, i32 96)
	; CHECK: ee.st.qacc_h.h.32.ip
	tail call void @llvm.xtensa.ee.st.qacc.h.l.128.ip(i32 5, i32 -496)
	; CHECK: ee.st.qacc_h.l.128.ip
	tail call void @llvm.xtensa.ee.st.qacc.l.h.32.ip(i32 12, i32 348)
	; CHECK: ee.st.qacc_l.h.32.ip
	tail call void @llvm.xtensa.ee.st.qacc.l.l.128.ip(i32 9, i32 592)
	; CHECK: ee.st.qacc_l.l.128.ip
	tail call void @llvm.xtensa.ee.st.ua.state.ip(i32 11, i32 -1568)
	; CHECK: ee.st.ua_state.ip
	tail call void @llvm.xtensa.ee.vadds.s16(i32 2, i32 3, i32 6)
	; CHECK: ee.vadds.s16
	tail call void @llvm.xtensa.ee.vadds.s16.ld.incp(i32 6, i32 5, i32 4, i32 6, i32 1)
	; CHECK: ee.vadds.s16.ld.incp
	tail call void @llvm.xtensa.ee.vadds.s16.st.incp(i32 6, i32 9, i32 7, i32 6, i32 2)
	; CHECK: ee.vadds.s16.st.incp
	tail call void @llvm.xtensa.ee.vadds.s32(i32 7, i32 3, i32 1)
	; CHECK: ee.vadds.s32
	tail call void @llvm.xtensa.ee.vadds.s32.ld.incp(i32 1, i32 8, i32 7, i32 1, i32 3)
	; CHECK: ee.vadds.s32.ld.incp
	tail call void @llvm.xtensa.ee.vadds.s32.st.incp(i32 5, i32 4, i32 3, i32 1, i32 7)
	; CHECK: ee.vadds.s32.st.incp
	tail call void @llvm.xtensa.ee.vadds.s8(i32 7, i32 2, i32 2)
	; CHECK: ee.vadds.s8
	tail call void @llvm.xtensa.ee.vadds.s8.ld.incp(i32 6, i32 2, i32 0, i32 3, i32 0)
	; CHECK: ee.vadds.s8.ld.incp
	tail call void @llvm.xtensa.ee.vadds.s8.st.incp(i32 1, i32 7, i32 2, i32 3, i32 4)
	; CHECK: ee.vadds.s8.st.incp
	tail call void @llvm.xtensa.ee.vcmp.eq.s16(i32 5, i32 3, i32 2)
	; CHECK: ee.vcmp.eq.s16
	tail call void @llvm.xtensa.ee.vcmp.eq.s32(i32 3, i32 3, i32 5)
	; CHECK: ee.vcmp.eq.s32
	tail call void @llvm.xtensa.ee.vcmp.eq.s8(i32 2, i32 6, i32 0)
	; CHECK: ee.vcmp.eq.s8
	tail call void @llvm.xtensa.ee.vcmp.gt.s16(i32 6, i32 2, i32 0)
	; CHECK: ee.vcmp.gt.s16
	tail call void @llvm.xtensa.ee.vcmp.gt.s32(i32 6, i32 7, i32 2)
	; CHECK: ee.vcmp.gt.s32
	tail call void @llvm.xtensa.ee.vcmp.gt.s8(i32 0, i32 3, i32 5)
	; CHECK: ee.vcmp.gt.s8
	tail call void @llvm.xtensa.ee.vcmp.lt.s16(i32 6, i32 6, i32 6)
	; CHECK: ee.vcmp.lt.s16
	tail call void @llvm.xtensa.ee.vcmp.lt.s32(i32 2, i32 2, i32 6)
	; CHECK: ee.vcmp.lt.s32
	tail call void @llvm.xtensa.ee.vcmp.lt.s8(i32 0, i32 3, i32 4)
	; CHECK: ee.vcmp.lt.s8
	tail call void @llvm.xtensa.ee.vldbc.16(i32 0, i32 7)
	; CHECK: ee.vldbc.16
	tail call void @llvm.xtensa.ee.vldbc.16.ip(i32 6, i32 10, i32 220)
	; CHECK: ee.vldbc.16.ip
	tail call void @llvm.xtensa.ee.vldbc.16.xp(i32 5, i32 10, i32 9)
	; CHECK: ee.vldbc.16.xp
	tail call void @llvm.xtensa.ee.vldbc.32(i32 1, i32 6)
	; CHECK: ee.vldbc.32
	tail call void @llvm.xtensa.ee.vldbc.32.ip(i32 4, i32 9, i32 -352)
	; CHECK: ee.vldbc.32.ip
	tail call void @llvm.xtensa.ee.vldbc.32.xp(i32 7, i32 10, i32 7)
	; CHECK: ee.vldbc.32.xp
	tail call void @llvm.xtensa.ee.vldbc.8(i32 6, i32 7)
	; CHECK: ee.vldbc.8
	tail call void @llvm.xtensa.ee.vldbc.8.ip(i32 1, i32 8, i32 124)
	; CHECK: ee.vldbc.8.ip
	tail call void @llvm.xtensa.ee.vldbc.8.xp(i32 1, i32 9, i32 14)
	; CHECK: ee.vldbc.8.xp
	tail call void @llvm.xtensa.ee.vldhbc.16.incp(i32 5, i32 0, i32 0)
	; CHECK: ee.vldhbc.16.incp
	tail call void @llvm.xtensa.ee.vld.128.ip(i32 5, i32 4, i32 1344)
	; CHECK: ee.vld.128.ip
	tail call void @llvm.xtensa.ee.vld.128.xp(i32 6, i32 9, i32 8)
	; CHECK: ee.vld.128.xp
	tail call void @llvm.xtensa.ee.vld.h.64.ip(i32 6, i32 3, i32 408)
	; CHECK: ee.vld.h.64.ip
	tail call void @llvm.xtensa.ee.vld.h.64.xp(i32 7, i32 6, i32 7)
	; CHECK: ee.vld.h.64.xp
	tail call void @llvm.xtensa.ee.vld.l.64.ip(i32 6, i32 4, i32 -752)
	; CHECK: ee.vld.l.64.ip
	tail call void @llvm.xtensa.ee.vld.l.64.xp(i32 6, i32 9, i32 8)
	; CHECK: ee.vld.l.64.xp
	tail call void @llvm.xtensa.ee.vmax.s16(i32 5, i32 6, i32 7)
	; CHECK: ee.vmax.s16
	tail call void @llvm.xtensa.ee.vmax.s16.ld.incp(i32 7, i32 5, i32 7, i32 5, i32 3)
	; CHECK: ee.vmax.s16.ld.incp
	tail call void @llvm.xtensa.ee.vmax.s16.st.incp(i32 2, i32 0, i32 7, i32 5, i32 2)
	; CHECK: ee.vmax.s16.st.incp
	tail call void @llvm.xtensa.ee.vmax.s32(i32 5, i32 5, i32 3)
	; CHECK: ee.vmax.s32
	tail call void @llvm.xtensa.ee.vmax.s32.ld.incp(i32 3, i32 8, i32 5, i32 2, i32 5)
	; CHECK: ee.vmax.s32.ld.incp
	tail call void @llvm.xtensa.ee.vmax.s32.st.incp(i32 0, i32 1, i32 2, i32 7, i32 3)
	; CHECK: ee.vmax.s32.st.incp
	tail call void @llvm.xtensa.ee.vmax.s8(i32 5, i32 2, i32 5)
	; CHECK: ee.vmax.s8
	tail call void @llvm.xtensa.ee.vmax.s8.ld.incp(i32 6, i32 7, i32 5, i32 4, i32 2)
	; CHECK: ee.vmax.s8.ld.incp
	tail call void @llvm.xtensa.ee.vmax.s8.st.incp(i32 2, i32 1, i32 1, i32 3, i32 1)
	; CHECK: ee.vmax.s8.st.incp
	tail call void @llvm.xtensa.ee.vmin.s16(i32 0, i32 1, i32 0)
	; CHECK: ee.vmin.s16
	tail call void @llvm.xtensa.ee.vmin.s16.ld.incp(i32 7, i32 0, i32 4, i32 7, i32 6)
	; CHECK: ee.vmin.s16.ld.incp
	tail call void @llvm.xtensa.ee.vmin.s16.st.incp(i32 4, i32 3, i32 7, i32 4, i32 5)
	; CHECK: ee.vmin.s16.st.incp
	tail call void @llvm.xtensa.ee.vmin.s32(i32 4, i32 0, i32 4)
	; CHECK: ee.vmin.s32
	tail call void @llvm.xtensa.ee.vmin.s32.ld.incp(i32 3, i32 8, i32 1, i32 0, i32 3)
	; CHECK: ee.vmin.s32.ld.incp
	tail call void @llvm.xtensa.ee.vmin.s32.st.incp(i32 4, i32 9, i32 7, i32 6, i32 6)
	; CHECK: ee.vmin.s32.st.incp
	tail call void @llvm.xtensa.ee.vmin.s8(i32 5, i32 7, i32 6)
	; CHECK: ee.vmin.s8
	tail call void @llvm.xtensa.ee.vmin.s8.ld.incp(i32 3, i32 3, i32 6, i32 1, i32 0)
	; CHECK: ee.vmin.s8.ld.incp
	tail call void @llvm.xtensa.ee.vmin.s8.st.incp(i32 4, i32 3, i32 7, i32 5, i32 1)
	; CHECK: ee.vmin.s8.st.incp
	tail call void @llvm.xtensa.ee.vmulas.s16.accx(i32 6, i32 7)
	; CHECK: ee.vmulas.s16.accx
	tail call void @llvm.xtensa.ee.vmulas.s16.accx.ld.ip(i32 2, i32 2, i32 96, i32 2, i32 5)
	; CHECK: ee.vmulas.s16.accx.ld.ip
	tail call void @llvm.xtensa.ee.vmulas.s16.accx.ld.ip.qup(i32 5, i32 5, i32 -96, i32 3, i32 5, i32 0, i32 5)
	; CHECK: ee.vmulas.s16.accx.ld.ip.qup
	tail call void @llvm.xtensa.ee.vmulas.s16.accx.ld.xp(i32 7, i32 1, i32 3, i32 2, i32 6)
	; CHECK: ee.vmulas.s16.accx.ld.xp
	tail call void @llvm.xtensa.ee.vmulas.s16.accx.ld.xp.qup(i32 5, i32 7, i32 8, i32 1, i32 3, i32 2, i32 5)
	; CHECK: ee.vmulas.s16.accx.ld.xp.qup
	tail call void @llvm.xtensa.ee.vmulas.s16.qacc(i32 5, i32 7)
	; CHECK: ee.vmulas.s16.qacc
	tail call void @llvm.xtensa.ee.vmulas.s16.qacc.ldbc.incp(i32 0, i32 8, i32 4, i32 3)
	; CHECK: ee.vmulas.s16.qacc.ldbc.incp
	tail call void @llvm.xtensa.ee.vmulas.s16.qacc.ldbc.incp.qup(i32 5, i32 12, i32 1, i32 7, i32 7, i32 4)
	; CHECK: ee.vmulas.s16.qacc.ldbc.incp.qup
	tail call void @llvm.xtensa.ee.vmulas.s16.qacc.ld.ip(i32 3, i32 9, i32 -112, i32 1, i32 6)
	; CHECK: ee.vmulas.s16.qacc.ld.ip
	tail call void @llvm.xtensa.ee.vmulas.s16.qacc.ld.ip.qup(i32 2, i32 12, i32 -112, i32 0, i32 2, i32 7, i32 0)
	; CHECK: ee.vmulas.s16.qacc.ld.ip.qup
	tail call void @llvm.xtensa.ee.vmulas.s16.qacc.ld.xp(i32 1, i32 1, i32 10, i32 0, i32 5)
	; CHECK: ee.vmulas.s16.qacc.ld.xp
	tail call void @llvm.xtensa.ee.vmulas.s16.qacc.ld.xp.qup(i32 5, i32 4, i32 14, i32 7, i32 1, i32 1, i32 6)
	; CHECK: ee.vmulas.s16.qacc.ld.xp.qup
	tail call void @llvm.xtensa.ee.vmulas.s8.accx(i32 5, i32 1)
	; CHECK: ee.vmulas.s8.accx
	tail call void @llvm.xtensa.ee.vmulas.s8.accx.ld.ip(i32 4, i32 1, i32 -112, i32 0, i32 1)
	; CHECK: ee.vmulas.s8.accx.ld.ip
	tail call void @llvm.xtensa.ee.vmulas.s8.accx.ld.ip.qup(i32 3, i32 11, i32 64, i32 6, i32 0, i32 1, i32 1)
	; CHECK: ee.vmulas.s8.accx.ld.ip.qup
	tail call void @llvm.xtensa.ee.vmulas.s8.accx.ld.xp(i32 1, i32 0, i32 9, i32 7, i32 7)
	; CHECK: ee.vmulas.s8.accx.ld.xp
	tail call void @llvm.xtensa.ee.vmulas.s8.accx.ld.xp.qup(i32 0, i32 6, i32 12, i32 2, i32 3, i32 6, i32 5)
	; CHECK: ee.vmulas.s8.accx.ld.xp.qup
	tail call void @llvm.xtensa.ee.vmulas.s8.qacc(i32 4, i32 0)
	; CHECK: ee.vmulas.s8.qacc
	tail call void @llvm.xtensa.ee.vmulas.s8.qacc.ldbc.incp(i32 6, i32 12, i32 6, i32 3)
	; CHECK: ee.vmulas.s8.qacc.ldbc.incp
	tail call void @llvm.xtensa.ee.vmulas.s8.qacc.ldbc.incp.qup(i32 7, i32 14, i32 2, i32 3, i32 1, i32 6)
	; CHECK: ee.vmulas.s8.qacc.ldbc.incp.qup
	tail call void @llvm.xtensa.ee.vmulas.s8.qacc.ld.ip(i32 6, i32 3, i32 0, i32 6, i32 1)
	; CHECK: ee.vmulas.s8.qacc.ld.ip
	tail call void @llvm.xtensa.ee.vmulas.s8.qacc.ld.ip.qup(i32 2, i32 14, i32 16, i32 1, i32 4, i32 0, i32 1)
	; CHECK: ee.vmulas.s8.qacc.ld.ip.qup
	tail call void @llvm.xtensa.ee.vmulas.s8.qacc.ld.xp(i32 0, i32 2, i32 5, i32 7, i32 4)
	; CHECK: ee.vmulas.s8.qacc.ld.xp
	tail call void @llvm.xtensa.ee.vmulas.s8.qacc.ld.xp.qup(i32 3, i32 2, i32 7, i32 5, i32 4, i32 3, i32 0)
	; CHECK: ee.vmulas.s8.qacc.ld.xp.qup
	tail call void @llvm.xtensa.ee.vmulas.u16.accx(i32 0, i32 0)
	; CHECK: ee.vmulas.u16.accx
	tail call void @llvm.xtensa.ee.vmulas.u16.accx.ld.ip(i32 0, i32 0, i32 64, i32 7, i32 2)
	; CHECK: ee.vmulas.u16.accx.ld.ip
	tail call void @llvm.xtensa.ee.vmulas.u16.accx.ld.ip.qup(i32 5, i32 3, i32 96, i32 1, i32 1, i32 0, i32 0)
	; CHECK: ee.vmulas.u16.accx.ld.ip.qup
	tail call void @llvm.xtensa.ee.vmulas.u16.accx.ld.xp(i32 6, i32 12, i32 9, i32 1, i32 5)
	; CHECK: ee.vmulas.u16.accx.ld.xp
	tail call void @llvm.xtensa.ee.vmulas.u16.accx.ld.xp.qup(i32 7, i32 11, i32 14, i32 3, i32 3, i32 3, i32 0)
	; CHECK: ee.vmulas.u16.accx.ld.xp.qup
	tail call void @llvm.xtensa.ee.vmulas.u16.qacc(i32 5, i32 7)
	; CHECK: ee.vmulas.u16.qacc
	tail call void @llvm.xtensa.ee.vmulas.u16.qacc.ldbc.incp(i32 6, i32 8, i32 0, i32 3)
	; CHECK: ee.vmulas.u16.qacc.ldbc.incp
	tail call void @llvm.xtensa.ee.vmulas.u16.qacc.ldbc.incp.qup(i32 6, i32 9, i32 7, i32 7, i32 6, i32 0)
	; CHECK: ee.vmulas.u16.qacc.ldbc.incp.qup
	tail call void @llvm.xtensa.ee.vmulas.u16.qacc.ld.ip(i32 4, i32 9, i32 16, i32 0, i32 4)
	; CHECK: ee.vmulas.u16.qacc.ld.ip
	tail call void @llvm.xtensa.ee.vmulas.u16.qacc.ld.ip.qup(i32 2, i32 8, i32 64, i32 6, i32 2, i32 6, i32 6)
	; CHECK: ee.vmulas.u16.qacc.ld.ip.qup
	tail call void @llvm.xtensa.ee.vmulas.u16.qacc.ld.xp(i32 4, i32 7, i32 14, i32 3, i32 2)
	; CHECK: ee.vmulas.u16.qacc.ld.xp
	tail call void @llvm.xtensa.ee.vmulas.u16.qacc.ld.xp.qup(i32 2, i32 9, i32 9, i32 2, i32 4, i32 5, i32 3)
	; CHECK: ee.vmulas.u16.qacc.ld.xp.qup
	tail call void @llvm.xtensa.ee.vmulas.u8.accx(i32 7, i32 6)
	; CHECK: ee.vmulas.u8.accx
	tail call void @llvm.xtensa.ee.vmulas.u8.accx.ld.ip(i32 2, i32 7, i32 -48, i32 1, i32 1)
	; CHECK: ee.vmulas.u8.accx.ld.ip
	tail call void @llvm.xtensa.ee.vmulas.u8.accx.ld.ip.qup(i32 5, i32 8, i32 16, i32 2, i32 3, i32 5, i32 5)
	; CHECK: ee.vmulas.u8.accx.ld.ip.qup
	tail call void @llvm.xtensa.ee.vmulas.u8.accx.ld.xp(i32 3, i32 8, i32 14, i32 3, i32 4)
	; CHECK: ee.vmulas.u8.accx.ld.xp
	tail call void @llvm.xtensa.ee.vmulas.u8.accx.ld.xp.qup(i32 4, i32 12, i32 3, i32 1, i32 6, i32 1, i32 7)
	; CHECK: ee.vmulas.u8.accx.ld.xp.qup
	tail call void @llvm.xtensa.ee.vmulas.u8.qacc(i32 0, i32 3)
	; CHECK: ee.vmulas.u8.qacc
	tail call void @llvm.xtensa.ee.vmulas.u8.qacc.ldbc.incp(i32 4, i32 8, i32 2, i32 4)
	; CHECK: ee.vmulas.u8.qacc.ldbc.incp
	tail call void @llvm.xtensa.ee.vmulas.u8.qacc.ldbc.incp.qup(i32 0, i32 2, i32 4, i32 5, i32 6, i32 7)
	; CHECK: ee.vmulas.u8.qacc.ldbc.incp.qup
	tail call void @llvm.xtensa.ee.vmulas.u8.qacc.ld.ip(i32 6, i32 2, i32 -128, i32 5, i32 3)
	; CHECK: ee.vmulas.u8.qacc.ld.ip
	tail call void @llvm.xtensa.ee.vmulas.u8.qacc.ld.ip.qup(i32 7, i32 1, i32 48, i32 5, i32 6, i32 5, i32 5)
	; CHECK: ee.vmulas.u8.qacc.ld.ip.qup
	tail call void @llvm.xtensa.ee.vmulas.u8.qacc.ld.xp(i32 1, i32 9, i32 12, i32 4, i32 2)
	; CHECK: ee.vmulas.u8.qacc.ld.xp
	tail call void @llvm.xtensa.ee.vmulas.u8.qacc.ld.xp.qup(i32 7, i32 13, i32 13, i32 0, i32 4, i32 2, i32 6)
	; CHECK: ee.vmulas.u8.qacc.ld.xp.qup
	tail call void @llvm.xtensa.ee.vmul.s16(i32 4, i32 1, i32 0)
	; CHECK: ee.vmul.s16
	tail call void @llvm.xtensa.ee.vmul.s16.ld.incp(i32 0, i32 5, i32 1, i32 3, i32 5)
	; CHECK: ee.vmul.s16.ld.incp
	tail call void @llvm.xtensa.ee.vmul.s16.st.incp(i32 3, i32 9, i32 3, i32 7, i32 7)
	; CHECK: ee.vmul.s16.st.incp
	tail call void @llvm.xtensa.ee.vmul.s8(i32 3, i32 7, i32 7)
	; CHECK: ee.vmul.s8
	tail call void @llvm.xtensa.ee.vmul.s8.ld.incp(i32 3, i32 0, i32 4, i32 3, i32 1)
	; CHECK: ee.vmul.s8.ld.incp
	tail call void @llvm.xtensa.ee.vmul.s8.st.incp(i32 7, i32 14, i32 1, i32 4, i32 4)
	; CHECK: ee.vmul.s8.st.incp
	tail call void @llvm.xtensa.ee.vmul.u16(i32 5, i32 6, i32 3)
	; CHECK: ee.vmul.u16
	tail call void @llvm.xtensa.ee.vmul.u16.ld.incp(i32 0, i32 12, i32 3, i32 0, i32 6)
	; CHECK: ee.vmul.u16.ld.incp
	tail call void @llvm.xtensa.ee.vmul.u16.st.incp(i32 0, i32 14, i32 0, i32 5, i32 7)
	; CHECK: ee.vmul.u16.st.incp
	tail call void @llvm.xtensa.ee.vmul.u8(i32 5, i32 1, i32 2)
	; CHECK: ee.vmul.u8
	tail call void @llvm.xtensa.ee.vmul.u8.ld.incp(i32 3, i32 6, i32 4, i32 4, i32 0)
	; CHECK: ee.vmul.u8.ld.incp
	tail call void @llvm.xtensa.ee.vmul.u8.st.incp(i32 4, i32 13, i32 4, i32 4, i32 2)
	; CHECK: ee.vmul.u8.st.incp
	tail call void @llvm.xtensa.ee.vprelu.s16(i32 5, i32 2, i32 5, i32 3)
	; CHECK: ee.vprelu.s16
	tail call void @llvm.xtensa.ee.vprelu.s8(i32 7, i32 7, i32 0, i32 14)
	; CHECK: ee.vprelu.s8
	tail call void @llvm.xtensa.ee.vrelu.s16(i32 3, i32 4, i32 4)
	; CHECK: ee.vrelu.s16
	tail call void @llvm.xtensa.ee.vrelu.s8(i32 5, i32 8, i32 5)
	; CHECK: ee.vrelu.s8
	tail call void @llvm.xtensa.ee.vsl.32(i32 6, i32 6)
	; CHECK: ee.vsl.32
	tail call void @llvm.xtensa.ee.vsmulas.s16.qacc(i32 2, i32 3, i32 5)
	; CHECK: ee.vsmulas.s16.qacc
	tail call void @llvm.xtensa.ee.vsmulas.s16.qacc.ld.incp(i32 5, i32 11, i32 7, i32 6, i32 6)
	; CHECK: ee.vsmulas.s16.qacc.ld.incp
	tail call void @llvm.xtensa.ee.vsmulas.s8.qacc(i32 6, i32 4, i32 9)
	; CHECK: ee.vsmulas.s8.qacc
	tail call void @llvm.xtensa.ee.vsmulas.s8.qacc.ld.incp(i32 3, i32 10, i32 7, i32 1, i32 2)
	; CHECK: ee.vsmulas.s8.qacc.ld.incp
	tail call void @llvm.xtensa.ee.vsr.32(i32 6, i32 6)
	; CHECK: ee.vsr.32
	tail call void @llvm.xtensa.ee.vst.128.ip(i32 1, i32 2, i32 160)
	; CHECK: ee.vst.128.ip
	tail call void @llvm.xtensa.ee.vst.128.xp(i32 1, i32 12, i32 0)
	; CHECK: ee.vst.128.xp
	tail call void @llvm.xtensa.ee.vst.h.64.ip(i32 5, i32 3, i32 400)
	; CHECK: ee.vst.h.64.ip
	tail call void @llvm.xtensa.ee.vst.h.64.xp(i32 7, i32 2, i32 0)
	; CHECK: ee.vst.h.64.xp
	tail call void @llvm.xtensa.ee.vst.l.64.ip(i32 0, i32 12, i32 -952)
	; CHECK: ee.vst.l.64.ip
	tail call void @llvm.xtensa.ee.vst.l.64.xp(i32 2, i32 7, i32 10)
	; CHECK: ee.vst.l.64.xp
	tail call void @llvm.xtensa.ee.vsubs.s16(i32 0, i32 3, i32 1)
	; CHECK: ee.vsubs.s16
	tail call void @llvm.xtensa.ee.vsubs.s16.ld.incp(i32 0, i32 13, i32 1, i32 2, i32 1)
	; CHECK: ee.vsubs.s16.ld.incp
	tail call void @llvm.xtensa.ee.vsubs.s16.st.incp(i32 4, i32 7, i32 6, i32 6, i32 3)
	; CHECK: ee.vsubs.s16.st.incp
	tail call void @llvm.xtensa.ee.vsubs.s32(i32 1, i32 6, i32 7)
	; CHECK: ee.vsubs.s32
	tail call void @llvm.xtensa.ee.vsubs.s32.ld.incp(i32 5, i32 11, i32 0, i32 6, i32 7)
	; CHECK: ee.vsubs.s32.ld.incp
	tail call void @llvm.xtensa.ee.vsubs.s32.st.incp(i32 4, i32 9, i32 0, i32 0, i32 2)
	; CHECK: ee.vsubs.s32.st.incp
	tail call void @llvm.xtensa.ee.vsubs.s8(i32 6, i32 1, i32 0)
	; CHECK: ee.vsubs.s8
	tail call void @llvm.xtensa.ee.vsubs.s8.ld.incp(i32 5, i32 12, i32 5, i32 4, i32 7)
	; CHECK: ee.vsubs.s8.ld.incp
	tail call void @llvm.xtensa.ee.vsubs.s8.st.incp(i32 1, i32 11, i32 5, i32 2, i32 3)
	; CHECK: ee.vsubs.s8.st.incp
	tail call void @llvm.xtensa.ee.vunzip.16(i32 3, i32 4)
	; CHECK: ee.vunzip.16
	tail call void @llvm.xtensa.ee.vunzip.32(i32 7, i32 3)
	; CHECK: ee.vunzip.32
	tail call void @llvm.xtensa.ee.vunzip.8(i32 4, i32 2)
	; CHECK: ee.vunzip.8
	tail call void @llvm.xtensa.ee.vzip.16(i32 2, i32 0)
	; CHECK: ee.vzip.16
	tail call void @llvm.xtensa.ee.vzip.32(i32 0, i32 3)
	; CHECK: ee.vzip.32
	tail call void @llvm.xtensa.ee.vzip.8(i32 5, i32 6)
	; CHECK: ee.vzip.8
	tail call void @llvm.xtensa.ee.xorq(i32 0, i32 7, i32 7)
	; CHECK: ee.xorq
	tail call void @llvm.xtensa.ee.zero.accx()
	; CHECK: ee.zero.accx
	tail call void @llvm.xtensa.ee.zero.q(i32 4)
	; CHECK: ee.zero.q
	tail call void @llvm.xtensa.ee.zero.qacc()
	; CHECK: ee.zero.qacc
	tail call i32 @llvm.xtensa.rur.accx.0()
	; CHECK: rur.accx_0
	tail call i32 @llvm.xtensa.rur.accx.1()
	; CHECK: rur.accx_1
	tail call i32 @llvm.xtensa.rur.fft.bit.width()
	; CHECK: rur.fft_bit_width
	tail call i32 @llvm.xtensa.rur.gpio.out()
	; CHECK: rur.gpio_out
	tail call i32 @llvm.xtensa.rur.qacc.h.0()
	; CHECK: rur.qacc_h_0
	tail call i32 @llvm.xtensa.rur.qacc.h.1()
	; CHECK: rur.qacc_h_1
	tail call i32 @llvm.xtensa.rur.qacc.h.2()
	; CHECK: rur.qacc_h_2
	tail call i32 @llvm.xtensa.rur.qacc.h.3()
	; CHECK: rur.qacc_h_3
	tail call i32 @llvm.xtensa.rur.qacc.h.4()
	; CHECK: rur.qacc_h_4
	tail call i32 @llvm.xtensa.rur.qacc.l.0()
	; CHECK: rur.qacc_l_0
	tail call i32 @llvm.xtensa.rur.qacc.l.1()
	; CHECK: rur.qacc_l_1
	tail call i32 @llvm.xtensa.rur.qacc.l.2()
	; CHECK: rur.qacc_l_2
	tail call i32 @llvm.xtensa.rur.qacc.l.3()
	; CHECK: rur.qacc_l_3
	tail call i32 @llvm.xtensa.rur.qacc.l.4()
	; CHECK: rur.qacc_l_4
	tail call i32 @llvm.xtensa.rur.sar.byte()
	; CHECK: rur.sar_byte
	tail call i32 @llvm.xtensa.rur.ua.state.0()
	; CHECK: rur.ua_state_0
	tail call i32 @llvm.xtensa.rur.ua.state.1()
	; CHECK: rur.ua_state_1
	tail call i32 @llvm.xtensa.rur.ua.state.2()
	; CHECK: rur.ua_state_2
	tail call i32 @llvm.xtensa.rur.ua.state.3()
	; CHECK: rur.ua_state_3
	tail call void @llvm.xtensa.wur.accx.0(i32 13)
	; CHECK: wur.accx_0
	tail call void @llvm.xtensa.wur.accx.1(i32 4)
	; CHECK: wur.accx_1
	tail call void @llvm.xtensa.wur.fft.bit.width(i32 7)
	; CHECK: wur.fft_bit_width
	tail call void @llvm.xtensa.wur.gpio.out(i32 0)
	; CHECK: wur.gpio_out
	tail call void @llvm.xtensa.wur.qacc.h.0(i32 8)
	; CHECK: wur.qacc_h_0
	tail call void @llvm.xtensa.wur.qacc.h.1(i32 6)
	; CHECK: wur.qacc_h_1
	tail call void @llvm.xtensa.wur.qacc.h.2(i32 12)
	; CHECK: wur.qacc_h_2
	tail call void @llvm.xtensa.wur.qacc.h.3(i32 7)
	; CHECK: wur.qacc_h_3
	tail call void @llvm.xtensa.wur.qacc.h.4(i32 2)
	; CHECK: wur.qacc_h_4
	tail call void @llvm.xtensa.wur.qacc.l.0(i32 0)
	; CHECK: wur.qacc_l_0
	tail call void @llvm.xtensa.wur.qacc.l.1(i32 6)
	; CHECK: wur.qacc_l_1
	tail call void @llvm.xtensa.wur.qacc.l.2(i32 10)
	; CHECK: wur.qacc_l_2
	tail call void @llvm.xtensa.wur.qacc.l.3(i32 2)
	; CHECK: wur.qacc_l_3
	tail call void @llvm.xtensa.wur.qacc.l.4(i32 6)
	; CHECK: wur.qacc_l_4
	tail call void @llvm.xtensa.wur.sar.byte(i32 13)
	; CHECK: wur.sar_byte
	tail call void @llvm.xtensa.wur.ua.state.0(i32 0)
	; CHECK: wur.ua_state_0
	tail call void @llvm.xtensa.wur.ua.state.1(i32 9)
	; CHECK: wur.ua_state_1
	tail call void @llvm.xtensa.wur.ua.state.2(i32 3)
	; CHECK: wur.ua_state_2
	tail call void @llvm.xtensa.wur.ua.state.3(i32 3)
	; CHECK: wur.ua_state_3
	tail call void @llvm.xtensa.mv.qr(i32 0, i32 1)
	; CHECK: mv.qr
	ret void
}

declare void @llvm.xtensa.ee.andq(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.bitrev(i32, i32) nounwind
declare void @llvm.xtensa.ee.cmul.s16(i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.cmul.s16.ld.incp(i32, i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.cmul.s16.st.incp(i32, i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.fft.ams.s16.ld.incp(i32, i32, i32, i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.fft.ams.s16.ld.incp.uaup(i32, i32, i32, i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.fft.ams.s16.ld.r32.decp(i32, i32, i32, i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.fft.ams.s16.st.incp(i32, i32, i32, i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.fft.cmul.s16.ld.xp(i32, i32, i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.fft.cmul.s16.st.xp(i32, i32, i32, i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.fft.r2bf.s16(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.fft.r2bf.s16.st.incp(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.fft.vst.r32.decp(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.ldf.128.ip(float, float, float, float, i32, i32) nounwind
declare void @llvm.xtensa.ee.ldf.128.xp(float, float, float, float, i32, i32) nounwind
declare void @llvm.xtensa.ee.ldf.64.ip(float, float, i32, i32) nounwind
declare void @llvm.xtensa.ee.ldf.64.xp(float, float, i32, i32) nounwind
declare void @llvm.xtensa.ee.ldqa.s16.128.ip(i32, i32) nounwind
declare void @llvm.xtensa.ee.ldqa.s16.128.xp(i32, i32) nounwind
declare void @llvm.xtensa.ee.ldqa.s8.128.ip(i32, i32) nounwind
declare void @llvm.xtensa.ee.ldqa.s8.128.xp(i32, i32) nounwind
declare void @llvm.xtensa.ee.ldqa.u16.128.ip(i32, i32) nounwind
declare void @llvm.xtensa.ee.ldqa.u16.128.xp(i32, i32) nounwind
declare void @llvm.xtensa.ee.ldqa.u8.128.ip(i32, i32) nounwind
declare void @llvm.xtensa.ee.ldqa.u8.128.xp(i32, i32) nounwind
declare void @llvm.xtensa.ee.ldxq.32(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.ld.128.usar.ip(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.ld.128.usar.xp(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.ld.accx.ip(i32, i32) nounwind
declare void @llvm.xtensa.ee.ld.qacc.h.h.32.ip(i32, i32) nounwind
declare void @llvm.xtensa.ee.ld.qacc.h.l.128.ip(i32, i32) nounwind
declare void @llvm.xtensa.ee.ld.qacc.l.h.32.ip(i32, i32) nounwind
declare void @llvm.xtensa.ee.ld.qacc.l.l.128.ip(i32, i32) nounwind
declare void @llvm.xtensa.ee.ld.ua.state.ip(i32, i32) nounwind
declare void @llvm.xtensa.ee.movi.32.a(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.movi.32.q(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.mov.s16.qacc(i32) nounwind
declare void @llvm.xtensa.ee.mov.s8.qacc(i32) nounwind
declare void @llvm.xtensa.ee.mov.u16.qacc(i32) nounwind
declare void @llvm.xtensa.ee.mov.u8.qacc(i32) nounwind
declare void @llvm.xtensa.ee.notq(i32, i32) nounwind
declare void @llvm.xtensa.ee.orq(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.slci.2q(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.slcxxp.2q(i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.srci.2q(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.srcmb.s16.qacc(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.srcmb.s8.qacc(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.srcq.128.st.incp(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.srcxxp.2q(i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.src.q(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.src.q.ld.ip(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.src.q.ld.xp(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.src.q.qup(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.srs.accx(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.stf.128.ip(float, float, float, float, i32, i32) nounwind
declare void @llvm.xtensa.ee.stf.128.xp(float, float, float, float, i32, i32) nounwind
declare void @llvm.xtensa.ee.stf.64.ip(float, float, i32, i32) nounwind
declare void @llvm.xtensa.ee.stf.64.xp(float, float, i32, i32) nounwind
declare void @llvm.xtensa.ee.stxq.32(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.st.accx.ip(i32, i32) nounwind
declare void @llvm.xtensa.ee.st.qacc.h.h.32.ip(i32, i32) nounwind
declare void @llvm.xtensa.ee.st.qacc.h.l.128.ip(i32, i32) nounwind
declare void @llvm.xtensa.ee.st.qacc.l.h.32.ip(i32, i32) nounwind
declare void @llvm.xtensa.ee.st.qacc.l.l.128.ip(i32, i32) nounwind
declare void @llvm.xtensa.ee.st.ua.state.ip(i32, i32) nounwind
declare void @llvm.xtensa.ee.vadds.s16(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vadds.s16.ld.incp(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vadds.s16.st.incp(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vadds.s32(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vadds.s32.ld.incp(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vadds.s32.st.incp(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vadds.s8(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vadds.s8.ld.incp(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vadds.s8.st.incp(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vcmp.eq.s16(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vcmp.eq.s32(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vcmp.eq.s8(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vcmp.gt.s16(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vcmp.gt.s32(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vcmp.gt.s8(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vcmp.lt.s16(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vcmp.lt.s32(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vcmp.lt.s8(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vldbc.16(i32, i32) nounwind
declare void @llvm.xtensa.ee.vldbc.16.ip(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vldbc.16.xp(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vldbc.32(i32, i32) nounwind
declare void @llvm.xtensa.ee.vldbc.32.ip(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vldbc.32.xp(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vldbc.8(i32, i32) nounwind
declare void @llvm.xtensa.ee.vldbc.8.ip(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vldbc.8.xp(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vldhbc.16.incp(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vld.128.ip(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vld.128.xp(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vld.h.64.ip(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vld.h.64.xp(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vld.l.64.ip(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vld.l.64.xp(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmax.s16(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmax.s16.ld.incp(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmax.s16.st.incp(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmax.s32(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmax.s32.ld.incp(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmax.s32.st.incp(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmax.s8(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmax.s8.ld.incp(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmax.s8.st.incp(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmin.s16(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmin.s16.ld.incp(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmin.s16.st.incp(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmin.s32(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmin.s32.ld.incp(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmin.s32.st.incp(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmin.s8(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmin.s8.ld.incp(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmin.s8.st.incp(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmulas.s16.accx(i32, i32) nounwind
declare void @llvm.xtensa.ee.vmulas.s16.accx.ld.ip(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmulas.s16.accx.ld.ip.qup(i32, i32, i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmulas.s16.accx.ld.xp(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmulas.s16.accx.ld.xp.qup(i32, i32, i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmulas.s16.qacc(i32, i32) nounwind
declare void @llvm.xtensa.ee.vmulas.s16.qacc.ldbc.incp(i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmulas.s16.qacc.ldbc.incp.qup(i32, i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmulas.s16.qacc.ld.ip(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmulas.s16.qacc.ld.ip.qup(i32, i32, i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmulas.s16.qacc.ld.xp(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmulas.s16.qacc.ld.xp.qup(i32, i32, i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmulas.s8.accx(i32, i32) nounwind
declare void @llvm.xtensa.ee.vmulas.s8.accx.ld.ip(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmulas.s8.accx.ld.ip.qup(i32, i32, i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmulas.s8.accx.ld.xp(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmulas.s8.accx.ld.xp.qup(i32, i32, i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmulas.s8.qacc(i32, i32) nounwind
declare void @llvm.xtensa.ee.vmulas.s8.qacc.ldbc.incp(i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmulas.s8.qacc.ldbc.incp.qup(i32, i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmulas.s8.qacc.ld.ip(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmulas.s8.qacc.ld.ip.qup(i32, i32, i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmulas.s8.qacc.ld.xp(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmulas.s8.qacc.ld.xp.qup(i32, i32, i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmulas.u16.accx(i32, i32) nounwind
declare void @llvm.xtensa.ee.vmulas.u16.accx.ld.ip(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmulas.u16.accx.ld.ip.qup(i32, i32, i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmulas.u16.accx.ld.xp(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmulas.u16.accx.ld.xp.qup(i32, i32, i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmulas.u16.qacc(i32, i32) nounwind
declare void @llvm.xtensa.ee.vmulas.u16.qacc.ldbc.incp(i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmulas.u16.qacc.ldbc.incp.qup(i32, i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmulas.u16.qacc.ld.ip(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmulas.u16.qacc.ld.ip.qup(i32, i32, i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmulas.u16.qacc.ld.xp(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmulas.u16.qacc.ld.xp.qup(i32, i32, i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmulas.u8.accx(i32, i32) nounwind
declare void @llvm.xtensa.ee.vmulas.u8.accx.ld.ip(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmulas.u8.accx.ld.ip.qup(i32, i32, i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmulas.u8.accx.ld.xp(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmulas.u8.accx.ld.xp.qup(i32, i32, i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmulas.u8.qacc(i32, i32) nounwind
declare void @llvm.xtensa.ee.vmulas.u8.qacc.ldbc.incp(i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmulas.u8.qacc.ldbc.incp.qup(i32, i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmulas.u8.qacc.ld.ip(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmulas.u8.qacc.ld.ip.qup(i32, i32, i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmulas.u8.qacc.ld.xp(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmulas.u8.qacc.ld.xp.qup(i32, i32, i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmul.s16(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmul.s16.ld.incp(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmul.s16.st.incp(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmul.s8(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmul.s8.ld.incp(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmul.s8.st.incp(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmul.u16(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmul.u16.ld.incp(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmul.u16.st.incp(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmul.u8(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmul.u8.ld.incp(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vmul.u8.st.incp(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vprelu.s16(i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vprelu.s8(i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vrelu.s16(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vrelu.s8(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vsl.32(i32, i32) nounwind
declare void @llvm.xtensa.ee.vsmulas.s16.qacc(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vsmulas.s16.qacc.ld.incp(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vsmulas.s8.qacc(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vsmulas.s8.qacc.ld.incp(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vsr.32(i32, i32) nounwind
declare void @llvm.xtensa.ee.vst.128.ip(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vst.128.xp(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vst.h.64.ip(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vst.h.64.xp(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vst.l.64.ip(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vst.l.64.xp(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vsubs.s16(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vsubs.s16.ld.incp(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vsubs.s16.st.incp(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vsubs.s32(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vsubs.s32.ld.incp(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vsubs.s32.st.incp(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vsubs.s8(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vsubs.s8.ld.incp(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vsubs.s8.st.incp(i32, i32, i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.vunzip.16(i32, i32) nounwind
declare void @llvm.xtensa.ee.vunzip.32(i32, i32) nounwind
declare void @llvm.xtensa.ee.vunzip.8(i32, i32) nounwind
declare void @llvm.xtensa.ee.vzip.16(i32, i32) nounwind
declare void @llvm.xtensa.ee.vzip.32(i32, i32) nounwind
declare void @llvm.xtensa.ee.vzip.8(i32, i32) nounwind
declare void @llvm.xtensa.ee.xorq(i32, i32, i32) nounwind
declare void @llvm.xtensa.ee.zero.accx() nounwind
declare void @llvm.xtensa.ee.zero.q(i32) nounwind
declare void @llvm.xtensa.ee.zero.qacc() nounwind
declare i32 @llvm.xtensa.rur.accx.0() nounwind
declare i32 @llvm.xtensa.rur.accx.1() nounwind
declare i32 @llvm.xtensa.rur.fft.bit.width() nounwind
declare i32 @llvm.xtensa.rur.gpio.out() nounwind
declare i32 @llvm.xtensa.rur.qacc.h.0() nounwind
declare i32 @llvm.xtensa.rur.qacc.h.1() nounwind
declare i32 @llvm.xtensa.rur.qacc.h.2() nounwind
declare i32 @llvm.xtensa.rur.qacc.h.3() nounwind
declare i32 @llvm.xtensa.rur.qacc.h.4() nounwind
declare i32 @llvm.xtensa.rur.qacc.l.0() nounwind
declare i32 @llvm.xtensa.rur.qacc.l.1() nounwind
declare i32 @llvm.xtensa.rur.qacc.l.2() nounwind
declare i32 @llvm.xtensa.rur.qacc.l.3() nounwind
declare i32 @llvm.xtensa.rur.qacc.l.4() nounwind
declare i32 @llvm.xtensa.rur.sar.byte() nounwind
declare i32 @llvm.xtensa.rur.ua.state.0() nounwind
declare i32 @llvm.xtensa.rur.ua.state.1() nounwind
declare i32 @llvm.xtensa.rur.ua.state.2() nounwind
declare i32 @llvm.xtensa.rur.ua.state.3() nounwind
declare void @llvm.xtensa.wur.accx.0(i32) nounwind
declare void @llvm.xtensa.wur.accx.1(i32) nounwind
declare void @llvm.xtensa.wur.fft.bit.width(i32) nounwind
declare void @llvm.xtensa.wur.gpio.out(i32) nounwind
declare void @llvm.xtensa.wur.qacc.h.0(i32) nounwind
declare void @llvm.xtensa.wur.qacc.h.1(i32) nounwind
declare void @llvm.xtensa.wur.qacc.h.2(i32) nounwind
declare void @llvm.xtensa.wur.qacc.h.3(i32) nounwind
declare void @llvm.xtensa.wur.qacc.h.4(i32) nounwind
declare void @llvm.xtensa.wur.qacc.l.0(i32) nounwind
declare void @llvm.xtensa.wur.qacc.l.1(i32) nounwind
declare void @llvm.xtensa.wur.qacc.l.2(i32) nounwind
declare void @llvm.xtensa.wur.qacc.l.3(i32) nounwind
declare void @llvm.xtensa.wur.qacc.l.4(i32) nounwind
declare void @llvm.xtensa.wur.sar.byte(i32) nounwind
declare void @llvm.xtensa.wur.ua.state.0(i32) nounwind
declare void @llvm.xtensa.wur.ua.state.1(i32) nounwind
declare void @llvm.xtensa.wur.ua.state.2(i32) nounwind
declare void @llvm.xtensa.wur.ua.state.3(i32) nounwind
declare void @llvm.xtensa.mv.qr(i32, i32) nounwind

