// RUN: %clang_cc1 -triple xtensa -O0 -emit-llvm %s -o - | FileCheck %s

#define	__malloc_like	__attribute__((__malloc__))
char *bufalloc () __malloc_like ;//__result_use_check;
extern void* malloc (unsigned size);

char *bufalloc ()
{
  char* buf = malloc(1024);

  return buf;
}

// CHECK: define dso_local noalias ptr @bufalloc() #0 {

struct S16 { int a[4]; } __attribute__ ((aligned (16)));

void callee_struct_a16b_1(struct S16 a) {}

// CHECK: define dso_local void @callee_struct_a16b_1(i128 %a.coerce)

void callee_struct_a16b_2(struct S16 a, int b) {}

// CHECK: define dso_local void @callee_struct_a16b_2(i128 %a.coerce, i32 noundef %b)

void callee_struct_a16b_3(int a, struct S16 b) {}

// CHECK: define dso_local void @callee_struct_a16b_3(i32 noundef %a, %struct.S16* noundef byval(%struct.S16) align 16 %b)
