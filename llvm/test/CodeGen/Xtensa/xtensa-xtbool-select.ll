; RUN: llc -O0 -mtriple=xtensa -mcpu=esp32 %s -o - | FileCheck %s

define <1 x i1> @test_select(<1 x i1>  %cc, <1 x i1> %t, <1 x i1> %f) {
  ; CHECK-LABEL: test_select
  ; CHECK-DAG: andb  [[EQ:b[0-9]+]], {{b[0-9]+}}, {{b[0-9]+}}
  ; CHECK-DAG: andbc [[NE:b[0-9]+]], {{b[0-9]+}}, {{b[0-9]+}}
  ; CHECK:     orb   {{b[0-9]+}}, [[EQ]], [[NE]]
  %r = select <1 x i1> %cc, <1 x i1> %t, <1 x i1> %f
  ret <1 x i1> %r
}
