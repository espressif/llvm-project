# RUN: python3 %s > %t && ( llc -O0 -mtriple=xtensa -mcpu=esp32 %t -o - | FileCheck %t )

from dataclasses import dataclass

@dataclass
class F:
  ret: str
  fun : str
  instr: str
  args : [str]

FIXTURES = [
('float', 'xt_addexp_s', 'addexp.s {{f[0-9]+}}, {{f[0-9]+}}', ['float', 'float']) ,
('float', 'xt_addexpm_s', 'addexpm.s {{f[0-9]+}}, {{f[0-9]+}}', ['float', 'float']) ,
('i32', 'xt_ceil_s', 'ceil.s {{a[0-9]+}}, {{f[0-9]+}}, 0', ['float', 0]) ,
('float', 'xt_div0_s', 'div0.s {{f[0-9]+}}, {{f[0-9]+}}', ['float']) ,
('float', 'xt_divn_s', 'divn.s {{f[0-9]+}}, {{f[0-9]+}}, {{f[0-9]+}}', ['float', 'float', 'float']) ,
('i32', 'xt_floor_s', 'floor.s {{a[0-9]+}}, {{f[0-9]+}}, 0', ['float', 0]) ,
('float', 'xt_lsi', 'lsi {{f[0-9]+}}, {{a[0-9]+}}, 0', ['ptr', 0]) ,
# skip xt_lsip ,
('float', 'xt_lsx', 'lsx {{f[0-9]+}}, {{a[0-9]+}}, {{a[0-9]+}}', ['ptr', 'i32']) ,
# skip xt_lsxp ,
('float', 'xt_madd_s', 'madd.s {{f[0-9]+}}, {{f[0-9]+}}, {{f[0-9]+}}', ['float', 'float', 'float']) ,
('float', 'xt_maddn_s', 'maddn.s {{f[0-9]+}}, {{f[0-9]+}}, {{f[0-9]+}}', ['float', 'float', 'float']) ,
('float', 'xt_mkdadj_s', 'mkdadj.s {{f[0-9]+}}, {{f[0-9]+}}', ['float', 'float']) ,
('float', 'xt_mksadj_s', 'mksadj.s {{f[0-9]+}}, {{f[0-9]+}}', ['float']) ,
('float', 'xt_mov_s', 'mov.s {{f[0-9]+}}, {{f[0-9]+}}', ['float']) ,
('float', 'xt_moveqz_s', 'moveqz.s {{f[0-9]+}}, {{f[0-9]+}}, {{a[0-9]+}}', ['float', 'float', 'i32']) ,
('float', 'xt_movgez_s', 'movgez.s {{f[0-9]+}}, {{f[0-9]+}}, {{a[0-9]+}}', ['float', 'float', 'i32']) ,
('float', 'xt_movltz_s', 'movltz.s {{f[0-9]+}}, {{f[0-9]+}}, {{a[0-9]+}}', ['float', 'float', 'i32']) ,
('float', 'xt_movnez_s', 'movnez.s {{f[0-9]+}}, {{f[0-9]+}}, {{a[0-9]+}}', ['float', 'float', 'i32']) ,
('float', 'xt_msub_s', 'msub.s {{f[0-9]+}}, {{f[0-9]+}}, {{f[0-9]+}}', ['float', 'float', 'float']) ,
('float', 'xt_neg_s', 'neg.s {{f[0-9]+}}, {{f[0-9]+}}', ['float']) ,
('float', 'xt_nexp01_s', 'nexp01.s {{f[0-9]+}}, {{f[0-9]+}}', ['float']) ,
('float', 'xt_recip0_s', 'recip0.s {{f[0-9]+}}, {{f[0-9]+}}', ['float']) ,
('i32', 'xt_rfr', 'rfr {{a[0-9]+}}, {{f[0-9]+}}', ['float']) ,
('i32', 'xt_round_s', 'round.s {{a[0-9]+}}, {{f[0-9]+}}, 0', ['float', 0]) ,
('float', 'xt_rsqrt0_s', 'rsqrt0.s {{f[0-9]+}}, {{f[0-9]+}}', ['float']) ,
('i32', 'xt_rur_fcr', 'rur {{a[0-9]+}}, fcr', []) ,
('i32', 'xt_rur_fsr', 'rur {{a[0-9]+}}, fsr', []) ,
('float', 'xt_sqrt0_s', 'sqrt0.s {{f[0-9]+}}, {{f[0-9]+}}', ['float']) ,
('void', 'xt_ssi', 'ssi {{f[0-9]+}}, {{a[0-9]+}}, 0', ['float', 'ptr', 0]) ,
('ptr', 'xt_ssip', 'ssip {{f[0-9]+}}, {{a[0-9]+}}, 0', ['float', 'ptr', 0]) ,
('void', 'xt_ssx', 'ssx {{f[0-9]+}}, {{a[0-9]+}}, {{a[0-9]+}}', ['float', 'ptr', 'i32']) ,
('ptr', 'xt_ssxp', 'ssxp {{f[0-9]+}}, {{a[0-9]+}}, {{a[0-9]+}}', ['float', 'ptr', 'i32']) ,
('float', 'xt_wfr', 'wfr {{f[0-9]+}}, {{a[0-9]+}}', ['i32']) ,
('void', 'xt_wur_fcr', 'wur.fcr {{a[0-9]+}}', ['i32']) ,
('void', 'xt_wur_fsr', 'wur.fsr {{a[0-9]+}}', ['i32']) ,
]

template = """
define {ret} @test_{fun}({fun_args})  {{
  ; CHECK-LABEL: {fun}
  ; CHECK: {instr} 
  {ret_var} {assign} call {ret} @llvm.xtensa.{builtin}({call_args})
  ret {ret} {ret_var}
}}
declare {ret} @llvm.xtensa.{builtin}({call_types});
"""

for f in FIXTURES:
    if isinstance(f, dict):
      f = F(**f)
    elif isinstance(f, tuple):
      f = F(*f)
    args = f.args
    f.fun_args = ",".join(
        ['%s %%a%d' % (a,i) for i,a, in enumerate(args) if isinstance(a,str)])
    f.builtin = f.fun.replace('_','.')
    f.call_args = ",".join(
                        [('%s %%a%d' % (a, i)) if isinstance(a,str) else ('i32 %d' % a)
                         for i,a, in enumerate(args)])
    f.call_types = ",".join([a if isinstance(a,str) else 'i32' for a in args])
    if f.ret == 'void':
      f.assign = ""
      f.ret_var = ""  
    else:
      f.assign = "="
      f.ret_var = "%r"
    print(template.format(**f.__dict__))
