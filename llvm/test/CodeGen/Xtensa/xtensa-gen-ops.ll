# RUN: python3 %s > %t && ( llc -mtriple=xtensa -mcpu=cnl %t -o - | FileCheck %t )


FIXTURES = [
    ('add','ae_add64','<1 x i64>'),
    ('add','ae_add32','<1 x i32>'),
    ('add','ae_add32','<2 x i32>'),
    ('add','ae_add16','<4 x i16>'),
    
    ('sub','ae_sub64','<1 x i64>'),
    ('sub','ae_sub32','<1 x i32>'),
    ('sub','ae_sub32','<2 x i32>'),
    ('sub','ae_sub16','<4 x i16>'),

    ('mul','ae_mulp32x2','<2 x i32>'),
    ('mul','ae_mulp32x2','<1 x i32>'),
    ('mul','ae_mul16x4','<4 x i16>'),
]

REG_TYPES = ['<1 x i64>', '<2 x i32>', '<1 x i32>', '<4 x i16>']

BITWISE_OPS = [
    ('and', 'ae_and'),
    ('or', 'ae_or'),
    ('xor', 'ae_xor')
]

from dataclasses import dataclass

@dataclass
class F:
  op: str
  instr: str
  type : str

template = """
define {type} @test_{fun}({type} %a, {type} %b)  {{
  ; CHECK-LABEL: test_{fun}
  ; CHECK: {instr} aed0, {{{{aed[01]}}}}, {{{{aed[01]}}}}
  %r = {op} {type} %a, %b
  ret {type} %r
}}
"""

def v2s(typ):
  return typ.strip('<>').replace(' ','')

for f in FIXTURES:
    f = F(*f)
    f.fun = f.op + v2s(f.type)
    print(template.format(**f.__dict__))

for f in BITWISE_OPS:
    op, instr = f
    for typ in REG_TYPES:
        fun = op + v2s(typ)
        print(template.format(op=op, instr=instr,fun=fun,type=typ))

cmp_template = """
define {vtype} @test_sel_{fun}({ctype} %a, {ctype} %b, {vtype} %t, {vtype} %f) {{
  ; CHECK-LABEL: test_sel_{fun}
  ; CHECK-DAG: ae_movda32 {{{{aed[0-9]+}}}}, {{{{a[0-9]+}}}}
  ; CHECK-DAG: ae_movda32 {{{{aed[0-9]+}}}}, {{{{a[0-9]+}}}}
  ; CHECK: {cmp_inst} {{{{b[0-9]+}}}}, {{{{aed[0-9]+}}}}, {{{{aed[0-9]+}}}}
  ; CHECK: {mov_inst} {{{{aed[0-9]+}}}}, {{{{aed[0-9]+}}}}, {{{{b[0-9]+}}}}
  %cmp = icmp {cmp_bc} {ctype} %a, %b
  %cond = select i1 %cmp, {vtype} %f, {vtype} %t
  ret {vtype} %cond
}}
"""

CMP_FIXTURES = [
  ('eq','ae_eq64', 'ae_movt64'),
  ('ne','ae_eq64', 'ae_movf64'),
  ('ugt','ae_le64', 'ae_movf64'),
  ('uge','ae_lt64', 'ae_movf64'),
  ('ult','ae_lt64', 'ae_movt64'),
  ('ule','ae_le64', 'ae_movt64'),
  ('sgt','ae_le64', 'ae_movf64'),
  ('sge','ae_lt64', 'ae_movf64'),
  ('slt','ae_lt64', 'ae_movt64'),
  ('sle','ae_le64', 'ae_movt64'),
]

SCALARS = "i32 i16 i8".split()

for cmp_bc, cmp_inst, mov_inst in CMP_FIXTURES:
  for ctype in SCALARS:
    for vtype in REG_TYPES:
      fun = '_'.join((cmp_bc, ctype, v2s(vtype)))
      print(cmp_template.format(**locals()))
