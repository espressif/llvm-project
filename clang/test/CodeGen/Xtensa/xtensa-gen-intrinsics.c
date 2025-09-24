# Fails on Winodws due to incorrectly built command line: `'(': command not found`, so disable it temporarily
# UNSUPPORTED: system-windows

# RUN: python3 %s > %t && ( %clang_cc1 -Dxtfloat=float -O0 -triple=xtensa %t -o - -emit-llvm | FileCheck %t )

FIXTURES = [
('xtfloat', 'xt_addexp_s', ['xtfloat', 'xtfloat']) ,
('xtfloat', 'xt_addexpm_s', ['xtfloat', 'xtfloat']) ,
('int', 'xt_ceil_s', ['xtfloat', 0]) ,
('xtfloat', 'xt_div0_s', ['xtfloat']) ,
('xtfloat', 'xt_divn_s', ['xtfloat', 'xtfloat', 'xtfloat']) ,
('int', 'xt_floor_s', ['xtfloat', 0]) ,
('xtfloat', 'xt_lsi', ['xtfloat*', 0]) ,
('xtfloat', 'xt_lsx', ['xtfloat*', 'int']) ,
('xtfloat', 'xt_madd_s', ['xtfloat', 'xtfloat', 'xtfloat']) ,
('xtfloat', 'xt_maddn_s', ['xtfloat', 'xtfloat', 'xtfloat']) ,
('xtfloat', 'xt_mkdadj_s', ['xtfloat', 'xtfloat']) ,
('xtfloat', 'xt_mksadj_s', ['xtfloat']) ,
('xtfloat', 'xt_mov_s', ['xtfloat']) ,
('xtfloat', 'xt_moveqz_s', ['xtfloat', 'xtfloat', 'int']) ,
('xtfloat', 'xt_movgez_s', ['xtfloat', 'xtfloat', 'int']) ,
('xtfloat', 'xt_movltz_s', ['xtfloat', 'xtfloat', 'int']) ,
('xtfloat', 'xt_movnez_s', ['xtfloat', 'xtfloat', 'int']) ,
('xtfloat', 'xt_msub_s', ['xtfloat', 'xtfloat', 'xtfloat']) ,
('xtfloat', 'xt_neg_s', ['xtfloat']) ,
('xtfloat', 'xt_nexp01_s', ['xtfloat']) ,
('xtfloat', 'xt_recip0_s', ['xtfloat']) ,
('int', 'xt_rfr', ['xtfloat']) ,
('int', 'xt_round_s', ['xtfloat', 0]) ,
('xtfloat', 'xt_rsqrt0_s', ['xtfloat']) ,
('int', 'xt_rur_fcr', []) ,
('int', 'xt_rur_fsr', []) ,
('xtfloat', 'xt_sqrt0_s', ['xtfloat']) ,
('void', 'xt_ssi', ['xtfloat', 'xtfloat*', 0]) ,
('xtfloat*', 'xt_ssip', ['xtfloat', 'xtfloat*', 0]) ,
('void', 'xt_ssx', ['xtfloat', 'xtfloat*', 'int']) ,
('xtfloat*', 'xt_ssxp', ['xtfloat', 'xtfloat*', 'int']) ,
('xtfloat', 'xt_wfr', ['int']) ,
('void', 'xt_wur_fcr', ['int']) ,
('void', 'xt_wur_fsr', ['int']) ,
]

from dataclasses import dataclass

TYPES = {
    'xtfloat' : 'float',
    'int' : 'i32',
    'void':'void'
}

def ctype2llvm(typ):
    if '*' in typ:
        return 'ptr'
    else:
        return TYPES[typ]


template = """
{ret} test_{fun}({fun_args}) {{
  // CHECK: {assign} call {llvm_ret} @llvm.xtensa.{builtin}({llvm_args})
  return __builtin_xtensa_{fun}({call_args});
}}
"""

@dataclass
class F:
  ret: str
  fun : str
  args: [str]
  #builtin
  #llvm_ret 
  #llvm_args
  #call_args

for f in FIXTURES:
    f = F(*f)
    args = f.args
    f.fun_args = ", ".join(
        ['%s a%d' % (a,i) for i,a, in enumerate(args) if isinstance(a,str)])
    f.builtin = f.fun.replace('_','.')
    f.llvm_args = ", ".join(
                        [('%s {{.*}}' % ctype2llvm(a)) if isinstance(a,str) else ('i32 %d' % a)
                         for i,a, in enumerate(args)])
    f.call_args = ", ".join(['a%d' % i if isinstance(a,str) else str(a) 
                            for i,a in enumerate(args)])
    f.llvm_ret = ctype2llvm(f.ret)
    f.assign = '' if f.ret == 'void' else '{{.*}} ='
    print(template.format(**f.__dict__))
