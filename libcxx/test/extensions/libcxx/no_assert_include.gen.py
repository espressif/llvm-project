# ===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===----------------------------------------------------------------------===##

# Ensure that none of the standard C++ headers implicitly include cassert or
# assert.h (because assert() is implemented as a macro).

# RUN: %{python} %s %{libcxx-dir}/utils

# block Lit from interpreting a RUN/XFAIL/etc inside the generation script
# END.

import sys

sys.path.append(sys.argv[1])
from libcxx.header_information import (
    lit_header_restrictions,
    lit_header_undeprecations,
    public_headers,
)

for header in public_headers:
    if header == "cassert":
        continue

    print(
        f"""\
//--- {header}.compile.pass.cpp
// RUN: %{{cxx}} %s %{{flags}} %{{compile_flags}} -fmodules -fcxx-modules -fmodules-cache-path=%t -fsyntax-only

// TODO: Espressif newlib is configured with --enable-newlib-reent-check-verify
//       which enables asserts in _REENT_CHECK macros defined in reent.h.
//       So assert.h gets included into standard headers.
// UNSUPPORTED: LIBCXX-ESP-FIXME

{lit_header_restrictions.get(header, '')}
{lit_header_undeprecations.get(header, '')}

#include <{header}>

#ifdef assert
# error "Do not include cassert or assert.h in standard header files"
#endif
"""
    )
