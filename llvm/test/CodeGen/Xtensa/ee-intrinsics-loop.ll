; RUN: llc -mtriple=xtensa -mcpu=esp32s3 < %s | FileCheck %s

; Function Attrs: nounwind
define dso_local void @test(i32 noundef %ptr.coerce, i32 noundef %ptr2.coerce) local_unnamed_addr #0 {
entry:
  %coerce.val.ip = inttoptr i32 %ptr.coerce to ptr
  %0 = load i32, ptr %coerce.val.ip, align 4, !tbaa !2
  tail call void @llvm.xtensa.ee.vld.128.ip(i32 1, i32 16, i32 0)
  tail call void @llvm.xtensa.ee.vld.128.ip(i32 1, i32 16, i32 0)
  tail call void @llvm.xtensa.ee.vld.128.ip(i32 1, i32 16, i32 0)
  tail call void @llvm.xtensa.ee.vld.128.ip(i32 1, i32 %0, i32 0)
  tail call void @llvm.xtensa.ee.vld.128.ip(i32 2, i32 %0, i32 0)
  tail call void @llvm.xtensa.ee.vld.128.ip(i32 3, i32 %0, i32 0)
  tail call void @llvm.xtensa.ee.vld.128.ip(i32 3, i32 %0, i32 0)
  tail call void @llvm.xtensa.ee.vld.128.ip(i32 3, i32 %0, i32 0)
  tail call void @llvm.xtensa.ee.vld.128.ip(i32 3, i32 %0, i32 0)
  tail call void @llvm.xtensa.ee.vld.128.ip(i32 3, i32 %0, i32 0)
  tail call void @llvm.xtensa.ee.vmax.s8(i32 1, i32 2, i32 3)
  tail call void @llvm.xtensa.ee.vmax.s16(i32 4, i32 5, i32 7)
  tail call void @llvm.xtensa.ee.stf.128.ip(float 0x40099999A0000000, float 0x4002666660000000, float 0x4011CCCCC0000000, float 0x401AA8F5C0000000, i32 %0, i32 16)
  tail call void @llvm.xtensa.ee.stf.128.ip(float undef, float undef, float undef, float undef, i32 %0, i32 16)
  tail call void @llvm.xtensa.ee.stf.128.xp(float undef, float undef, float undef, float undef, i32 16, i32 %0)
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  tail call void @llvm.xtensa.wur.sar.byte(i32 0)
  tail call void @llvm.xtensa.wur.accx.0(i32 0)
  tail call void @llvm.xtensa.wur.accx.1(i32 0)
  ret void

for.body:                                         ; preds = %entry, %for.body
  %i.029 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  tail call void @llvm.xtensa.ee.ld.128.usar.ip(i32 4, i32 %0, i32 16)
  tail call void @llvm.xtensa.ee.src.q.ld.ip(i32 3, i32 %0, i32 16, i32 4, i32 2)
  tail call void @llvm.xtensa.ee.vmulas.s16.accx.ld.ip(i32 0, i32 10, i32 16, i32 4, i32 6)
  tail call void @llvm.xtensa.ee.vmulas.s16.accx.ld.xp.qup(i32 4, i32 %0, i32 16, i32 0, i32 4, i32 2, i32 3)
  tail call void @llvm.xtensa.ee.ld.128.usar.xp(i32 4, i32 %0, i32 16)
  tail call void @llvm.xtensa.ee.vmulas.s16.accx.ld.ip.qup(i32 3, i32 %0, i32 16, i32 0, i32 3, i32 4, i32 2)
  %inc = add nuw nsw i32 %i.029, 1
  %exitcond.not = icmp eq i32 %inc, 32
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body, !llvm.loop !6
}

; CHECK-LABEL: test:                                   # @test
; CHECK: # %bb.0:                                # %entry
; CHECK-NEXT:    entry	a1, 32
; CHECK-NEXT:	l32i.n	a8, a2, 0
; CHECK-NEXT:	movi.n	a9, 16
; CHECK-NEXT:	ee.vld.128.ip	 q1, a9, 0
; CHECK-NEXT:	ee.vld.128.ip	 q1, a9, 0
; CHECK-NEXT:	ee.vld.128.ip	 q1, a9, 0
; CHECK-NEXT:	ee.vld.128.ip	 q1, a8, 0
; CHECK-NEXT:	ee.vld.128.ip	 q2, a8, 0
; CHECK-NEXT:	ee.vld.128.ip	 q3, a8, 0
; CHECK-NEXT:	ee.vld.128.ip	 q3, a8, 0
; CHECK-NEXT:	ee.vld.128.ip	 q3, a8, 0
; CHECK-NEXT:	ee.vld.128.ip	 q3, a8, 0
; CHECK-NEXT:	ee.vld.128.ip	 q3, a8, 0
; CHECK-NEXT:	ee.vmax.s8	 q1, q2, q3
; CHECK-NEXT:	ee.vmax.s16	 q4, q5, q7
; CHECK-NEXT:	l32r	a10, .LCPI0_0
; CHECK-NEXT:	wfr	f8, a10
; CHECK-NEXT:	l32r	a10, .LCPI0_1
; CHECK-NEXT:	wfr	f9, a10
; CHECK-NEXT:	l32r	a10, .LCPI0_2
; CHECK-NEXT:	wfr	f10, a10
; CHECK-NEXT:	l32r	a10, .LCPI0_3
; CHECK-NEXT:	wfr	f11, a10
; CHECK-NEXT:	ee.stf.128.ip	 f11, f10, f9, f8, a8, 16
; CHECK-NEXT:	ee.stf.128.ip	 f8, f8, f8, f8, a8, 16
; CHECK-NEXT:	ee.stf.128.xp	 f8, f8, f8, f8, a9, a8
; CHECK-NEXT:	movi.n	a10, 0
; CHECK-NEXT:	movi.n	a11, 10
; CHECK-NEXT: .LBB0_1:                                # %for.body
; CHECK-NEXT:                                       # =>This Inner Loop Header: Depth=1
; CHECK-NEXT:  ee.ld.128.usar.ip        q4, a8, 16
; CHECK-NEXT:  ee.src.q.ld.ip   q3, a8, 16, q4, q2
; CHECK-NEXT:  ee.vmulas.s16.accx.ld.ip         q0, a11, 16, q4, q6
; CHECK-NEXT:  ee.vmulas.s16.accx.ld.xp.qup     q4, a8, a9, q0, q4, q2, q3
; CHECK-NEXT:  ee.ld.128.usar.xp        q4, a8, a9
; CHECK-NEXT:  ee.vmulas.s16.accx.ld.ip.qup     q3, a8, 16, q0, q3, q4, q2
; CHECK-NEXT:  addi.n  a10, a10, 1
; CHECK-NEXT:  bnei    a10, 32, .LBB0_1
; CHECK-NEXT: # %bb.2:                                # %for.cond.cleanup
; CHECK-NEXT:  movi.n  a8, 0
; CHECK-NEXT:  wur.sar_byte     a8
; CHECK-NEXT:  wur.accx_0       a8
; CHECK-NEXT:  wur.accx_1       a8
; CHECK-NEXT:  retw.n



; Function Attrs: nounwind
declare void @llvm.xtensa.ee.vld.128.ip(i32 immarg, i32, i32 immarg) #1

; Function Attrs: nounwind
declare void @llvm.xtensa.ee.vmax.s8(i32 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nounwind
declare void @llvm.xtensa.ee.vmax.s16(i32 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nounwind
declare void @llvm.xtensa.ee.stf.128.ip(float, float, float, float, i32, i32 immarg) #1

; Function Attrs: nounwind
declare void @llvm.xtensa.ee.stf.128.xp(float, float, float, float, i32, i32) #1

; Function Attrs: nounwind
declare void @llvm.xtensa.ee.ld.128.usar.ip(i32 immarg, i32, i32 immarg) #1

; Function Attrs: nounwind
declare void @llvm.xtensa.ee.src.q.ld.ip(i32 immarg, i32, i32 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nounwind
declare void @llvm.xtensa.ee.vmulas.s16.accx.ld.ip(i32 immarg, i32, i32 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nounwind
declare void @llvm.xtensa.ee.vmulas.s16.accx.ld.xp.qup(i32 immarg, i32, i32, i32 immarg, i32 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nounwind
declare void @llvm.xtensa.ee.ld.128.usar.xp(i32 immarg, i32, i32) #1

; Function Attrs: nounwind
declare void @llvm.xtensa.ee.vmulas.s16.accx.ld.ip.qup(i32 immarg, i32, i32 immarg, i32 immarg, i32 immarg, i32 immarg, i32 immarg) #1

; Function Attrs: nounwind
declare void @llvm.xtensa.wur.sar.byte(i32) #1

; Function Attrs: nounwind
declare void @llvm.xtensa.wur.accx.0(i32) #1

; Function Attrs: nounwind
declare void @llvm.xtensa.wur.accx.1(i32) #1

attributes #0 = { nounwind "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-features"="+atomctl,+bool,+coprocessor,+debug,+density,+dfpaccel,+div32,+exception,+fp,+highpriinterrupts,+interrupt,+loop,+mac16,+memctl,+miscsr,+mul32,+mul32high,+nsa,+prid,+regprotect,+rvector,+s32c1i,+sext,+threadptr,+timerint,+windowed" }
attributes #1 = { nounwind }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 1}
!1 = !{!"clang version 15.0.0 (https://github.com/espressif/llvm-project.git a69f9a9e5e52d92f584c8268b1d041b2dfaee1e8)"}
!2 = !{!3, !3, i64 0}
!3 = !{!"int", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.mustprogress"}
