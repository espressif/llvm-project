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
