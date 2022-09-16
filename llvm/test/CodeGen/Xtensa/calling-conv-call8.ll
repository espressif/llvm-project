; RUN: llc -mtriple=xtensa -mcpu=esp32 -verify-machineinstrs < %s \
; RUN:   | FileCheck -check-prefix=XTENSA-STRUCT16 %s
; RUN: llc -mtriple=xtensa -mcpu=esp32 -verify-machineinstrs < %s \
; RUN:   | FileCheck -check-prefix=XTENSA-I128 %s

%struct.S = type { [4 x i32] }

@v = dso_local global i32 0, align 4

define dso_local void @caller_struct_a128b_1([4 x i32] %0) {

; XTENSA-STRUCT16-LABEL: caller_struct_a128b_1:
; XTENSA-STRUCT16-NEXT: .cfi_startproc
; XTENSA-STRUCT16: #  %bb.0:
; XTENSA-STRUCT16-NEXT: entry a1, 32
; XTENSA-STRUCT16-NEXT: .cfi_def_cfa_offset 32
; XTENSA-STRUCT16-NEXT: l32r  a8, .LCPI0_0
; XTENSA-STRUCT16-NEXT: mov.n a10, a2
; XTENSA-STRUCT16-NEXT: mov.n a11, a3
; XTENSA-STRUCT16-NEXT: mov.n a12, a4
; XTENSA-STRUCT16-NEXT: mov.n a13, a5
; XTENSA-STRUCT16-NEXT: callx8 a8
; XTENSA-STRUCT16-NEXT: retw.n

  call void @callee_struct_a128b_1([4 x i32] %0)
  ret void
}

declare dso_local void @callee_struct_a128b_1([4 x i32])

define dso_local void @caller_struct_a128b_2([4 x i32] %0) {
; XTENSA-STRUCT16-LABEL: caller_struct_a128b_2:
; XTENSA-STRUCT16:      .cfi_startproc
; XTENSA-STRUCT16-NEXT: # %bb.0:
; XTENSA-STRUCT16-NEXT: entry a1, 32
; XTENSA-STRUCT16-NEXT: .cfi_def_cfa_offset 32
; XTENSA-STRUCT16-NEXT: l32r a8, .LCPI1_0
; XTENSA-STRUCT16-NEXT: l32i.n a14, a8, 0
; XTENSA-STRUCT16-NEXT: l32r   a8, .LCPI1_1
; XTENSA-STRUCT16-NEXT: mov.n  a10, a2
; XTENSA-STRUCT16-NEXT: mov.n  a11, a3
; XTENSA-STRUCT16-NEXT: mov.n  a12, a4
; XTENSA-STRUCT16-NEXT: mov.n  a13, a5
; XTENSA-STRUCT16-NEXT: callx8 a8
; XTENSA-STRUCT16-NEXT: retw.n

  %2 = load i32, i32* @v, align 4
  call void @callee_struct_a128b_2([4 x i32] %0, i32 noundef %2)
  ret void
}

declare dso_local void @callee_struct_a128b_2([4 x i32], i32 noundef)

define dso_local void @caller_struct_a128b_3([4 x i32] %0) {
; XTENSA-STRUCT16-LABEL: caller_struct_a128b_3:
; XTENSA-STRUCT16:      .cfi_startproc
; XTENSA-STRUCT16-NEXT: # %bb.0:
; XTENSA-STRUCT16-NEXT: entry a1, 64
; XTENSA-STRUCT16-NEXT: .cfi_def_cfa_offset 64
; XTENSA-STRUCT16-NEXT: s32i.n a5, a1, 28
; XTENSA-STRUCT16-NEXT: s32i.n a4, a1, 24
; XTENSA-STRUCT16-NEXT: s32i.n a3, a1, 20
; XTENSA-STRUCT16-NEXT: s32i.n a2, a1, 16
; XTENSA-STRUCT16-NEXT: l32r   a8, .LCPI2_0
; XTENSA-STRUCT16-NEXT: l32i.n a10, a8, 0
; XTENSA-STRUCT16-NEXT: l32i.n a8, a1, 28
; XTENSA-STRUCT16-NEXT: s32i.n a8, a1, 12
; XTENSA-STRUCT16-NEXT: l32i.n a8, a1, 24
; XTENSA-STRUCT16-NEXT: s32i.n a8, a1, 8
; XTENSA-STRUCT16-NEXT: l32i.n a8, a1, 20
; XTENSA-STRUCT16-NEXT: s32i.n a8, a1, 4
; XTENSA-STRUCT16-NEXT: l32i.n a8, a1, 16
; XTENSA-STRUCT16-NEXT: s32i.n a8, a1, 0
; XTENSA-STRUCT16-NEXT: l32r   a8, .LCPI2_1
; XTENSA-STRUCT16-NEXT: callx8 a8
; XTENSA-STRUCT16-NEXT: retw.n

  %2 = alloca %struct.S, align 16
  %3 = extractvalue [4 x i32] %0, 0
  %4 = getelementptr inbounds %struct.S, %struct.S* %2, i32 0, i32 0, i32 0
  store i32 %3, i32* %4, align 16
  %5 = extractvalue [4 x i32] %0, 1
  %6 = getelementptr inbounds %struct.S, %struct.S* %2, i32 0, i32 0, i32 1
  store i32 %5, i32* %6, align 4
  %7 = extractvalue [4 x i32] %0, 2
  %8 = getelementptr inbounds %struct.S, %struct.S* %2, i32 0, i32 0, i32 2
  store i32 %7, i32* %8, align 8
  %9 = extractvalue [4 x i32] %0, 3
  %10 = getelementptr inbounds %struct.S, %struct.S* %2, i32 0, i32 0, i32 3
  store i32 %9, i32* %10, align 4
  %11 = load i32, i32* @v, align 4
  call void @callee_struct_a128b_3(i32 noundef %11, %struct.S* noundef nonnull byval(%struct.S) align 16 %2)
  ret void
}

declare dso_local void @callee_struct_a128b_3(i32 noundef, %struct.S* noundef byval(%struct.S) align 16)

define dso_local void @caller_i128b_1(i128 noundef %0) {
; XTENSA-I128-LABEL: caller_i128b_1:
; XTENSA-I128:       .cfi_startproc
; XTENSA-I128-NEXT:  # %bb.0:
; XTENSA-I128-NEXT:  entry  a1, 32
; XTENSA-I128-NEXT:  .cfi_def_cfa_offset 32
; XTENSA-I128-NEXT:  l32r   a8, .LCPI3_0
; XTENSA-I128-NEXT:  mov.n  a10, a2
; XTENSA-I128-NEXT:  mov.n  a11, a3
; XTENSA-I128-NEXT:  mov.n  a12, a4
; XTENSA-I128-NEXT:  mov.n  a13, a5
; XTENSA-I128-NEXT:  callx8 a8
; XTENSA-I128-NEXT:  retw.n

  call void @callee_i128b_1(i128 noundef %0)
  ret void
}

declare dso_local void @callee_i128b_1(i128 noundef)

define dso_local void @caller_i128b_2(i128 noundef %0) {
; XTENSA-I128-LABEL: caller_i128b_2:
; XTENSA-I128:       .cfi_startproc
; XTENSA-I128-NEXT:  # %bb.0:
; XTENSA-I128-NEXT:  entry  a1, 32
; XTENSA-I128-NEXT:  .cfi_def_cfa_offset 32
; XTENSA-I128-NEXT:  l32r   a8, .LCPI4_0
; XTENSA-I128-NEXT:  l32i.n a14, a8, 0
; XTENSA-I128-NEXT:  l32r   a8, .LCPI4_1
; XTENSA-I128-NEXT:  mov.n  a10, a2
; XTENSA-I128-NEXT:  mov.n  a11, a3
; XTENSA-I128-NEXT:  mov.n  a12, a4
; XTENSA-I128-NEXT:  mov.n  a13, a5
; XTENSA-I128-NEXT:  callx8 a8
; XTENSA-I128-NEXT:  retw.n

  %2 = load i32, i32* @v, align 4
  call void @callee_i128b_2(i128 noundef %0, i32 noundef %2)
  ret void
}

declare dso_local void @callee_i128b_2(i128 noundef, i32 noundef)

define dso_local void @caller_i128b_3(i128 noundef %0) {
; XTENSA-I128-LABEL: caller_i128b_3:
; XTENSA-I128:       .cfi_startproc
; XTENSA-I128-NEXT:  # %bb.0:
; XTENSA-I128-NEXT:  entry a1, 64
; XTENSA-I128-NEXT:  .cfi_def_cfa_offset 64
; XTENSA-I128-NEXT:  s32i.n a5, a1, 28
; XTENSA-I128-NEXT:  s32i.n a4, a1, 24
; XTENSA-I128-NEXT:  s32i.n a3, a1, 20
; XTENSA-I128-NEXT:  s32i.n a2, a1, 16
; XTENSA-I128-NEXT:  l32r   a8, .LCPI5_0
; XTENSA-I128-NEXT:  l32i.n a10, a8, 0
; XTENSA-I128-NEXT:  l32i.n a8, a1, 28
; XTENSA-I128-NEXT:  s32i.n a8, a1, 12
; XTENSA-I128-NEXT:  l32i.n a8, a1, 24
; XTENSA-I128-NEXT:  s32i.n a8, a1, 8
; XTENSA-I128-NEXT:  l32i.n a8, a1, 20
; XTENSA-I128-NEXT:  s32i.n a8, a1, 4
; XTENSA-I128-NEXT:  l32i.n a8, a1, 16
; XTENSA-I128-NEXT:  s32i.n a8, a1, 0
; XTENSA-I128-NEXT:  l32r   a8, .LCPI5_1
; XTENSA-I128-NEXT:  callx8 a8
; XTENSA-I128-NEXT:  retw.n

  %2 = alloca i128, align 16
  %3 = load i32, i32* @v, align 4
  store i128 %0, i128* %2, align 16
  call void @callee_i128b_3(i32 noundef %3, i128* noundef nonnull byval(i128) align 16 %2)
  ret void
}

declare dso_local void @callee_i128b_3(i32 noundef, i128* noundef byval(i128) align 16) 
