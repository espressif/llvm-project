; RUN: llc -O1 -mtriple=xtensa -mcpu=esp32 -mfix-esp32-psram-cache-issue  -mfix-esp32-psram-cache-strategy=nops %s -o - | FileCheck %s

@a = dso_local local_unnamed_addr global i32 0, align 4
@b = dso_local local_unnamed_addr global i32 0, align 4

; Function Attrs: nofree norecurse nounwind
define dso_local void @f(i32 %a1.coerce, i32 %a2.coerce, i32 %a3.coerce, i32 %a4.coerce, i32 %cond) local_unnamed_addr #0 {
entry:
  %coerce.val.ip = inttoptr i32 %a1.coerce to i8*
  %coerce.val.ip1 = inttoptr i32 %a2.coerce to i16*
  %coerce.val.ip2 = inttoptr i32 %a3.coerce to i32*
  %coerce.val.ip3 = inttoptr i32 %a4.coerce to i32*
  %0 = load i32, i32* %coerce.val.ip2, align 4
  %conv = trunc i32 %0 to i8
  store i8 %conv, i8* %coerce.val.ip, align 1
  %tobool.not = icmp eq i32 %cond, 0
  br i1 %tobool.not, label %if.end, label %if.then
; CHECK:	l32i.n	a8, a4, 0
; CHECK:	memw
; CHECK:	s8i	a8, a2, 0


if.then:                                          ; preds = %entry
  %1 = load i32, i32* %coerce.val.ip2, align 4
  %conv8 = trunc i32 %1 to i16
  store i16 %conv8, i16* %coerce.val.ip1, align 2
  %2 = load i32, i32* %coerce.val.ip3, align 4
  store i32 %2, i32* %coerce.val.ip2, align 4
  %conv10 = trunc i32 %2 to i8
  store i8 %conv10, i8* %coerce.val.ip, align 1
  br label %return
; CHECK:	nop
; CHECK:	nop
; CHECK:	nop
; CHECK:	nop
; CHECK:	l32i.n	a8, a4, 0
; CHECK:	s16i	a8, a3, 0
; CHECK:	memw
; CHECK:	nop
; CHECK:	nop
; CHECK:	nop
; CHECK:	nop
; CHECK:	l32i.n	a8, a5, 0
; CHECK:	s32i.n	a8, a4, 0
; CHECK:	memw
; CHECK:	s8i	a8, a2, 0

if.end:                                           ; preds = %entry
  %3 = load i32, i32* %coerce.val.ip3, align 4
  %conv9 = trunc i32 %3 to i16
  store i16 %conv9, i16* %coerce.val.ip1, align 2
  br label %return
; CHECK:	l32i.n	a8, a5, 0
; CHECK:	memw
; CHECK:	s16i	a8, a3, 0


return:                                           ; preds = %if.then, %if.end
  ret void
}
